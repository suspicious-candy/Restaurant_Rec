"""Build the restaurant vector index from MongoDB. The index's only writer.

Mongo is the source of truth for text; Chroma is a derived index this script
regenerates from scratch. That direction is what makes the embedding model, the
chunking and the text template all changeable later — none of that is possible
if accumulated text lives only inside the vector store.

Prereq:  ./.venv/Scripts/python.exe -m pip install pymongo

Run:
    ./.venv/Scripts/python.exe rebuild_index.py --limit 200      smoke test
    ./.venv/Scripts/python.exe rebuild_index.py --only-missing   top-up after a sync
    ./.venv/Scripts/python.exe rebuild_index.py --recreate       full rebuild

Restart the FastAPI service afterwards: Chroma's PersistentClient caches
in-process, so a running service cannot see a collection another process
dropped and rebuilt.
"""
# Preload pyarrow BEFORE torch/transformers get pulled in below — see the note
# at the top of service.py for the Windows OpenMP DLL crash this avoids.
import pyarrow  # noqa: F401
import pyarrow.dataset  # noqa: F401

import argparse
import os

import chromadb
import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = "chroma_db"
COLLECTION = "places_v2"

# Cap on chunks per restaurant. Without it a place with 400 tips gets a
# smoother, more averaged vector than one with 8 — systematically pulling
# popular restaurants toward the corpus centroid, where they match every query
# weakly and no query strongly. Equal treatment beats using all the data.
MAX_CHUNKS = 24

# Tokens shared between neighbouring chunks, so a sentence split across a
# boundary contributes a whole thought rather than two halves.
CHUNK_OVERLAP = 32

ENCODE_BATCH = 128

# Restaurants per upsert. Bounds peak memory and commits progressively, so a
# crash at restaurant 12,000 leaves 11,500 written rather than nothing.
DOC_BATCH = 512


def rating_phrase(r: float) -> str:
    """Turn a 0-10 rating into words the encoder can actually use.

    A text encoder has no useful representation of "7.4", but a rich one for
    "a good rating", because that phrasing saturates the review text it was
    trained on. With a 256-token budget, spend it on tokens the model knows.

    Args:
        r: rating on the schema's 0-10 scale.

    Returns:
        A noun phrase such as "a very high rating".
    """
    if r >= 9:
        return "an exceptional rating"
    if r >= 8:
        return "a very high rating"
    if r >= 7:
        return "a good rating"
    if r >= 6:
        return "a decent rating"
    return "a modest rating"


def build_text(doc: dict) -> str:
    """Text that gets EMBEDDED. Deliberately excludes the restaurant's name.

    The name is a proper noun and, in a ~15-token document, it dominated the
    vector: 'sushi omakase' returned Sushi Time / Sushi Iwa / Sushi Inc, and
    'pizza and pasta' scored 5% category precision while 100% of its hits had
    a pizza word in the NAME. experiment_template.py measured three templates
    over 15,169 restaurants and 5 cuisine probes:

        template   category@20   nameleak@20
        control          74%          88%
        cat_x3           78%          87%
        no_name          95%          41%     <- this one

    Repeating the category to outweigh the name (cat_x3) barely moved either
    number: a transformer is not bag-of-words, so duplicating a token does not
    linearly raise its weight, and a distinctive proper noun keeps winning.
    Removing it is what works.

    Finding a place BY name is a different job — that belongs in a lexical
    search over Mongo (/api/Restaurants/search), not in the taste index. The
    name is still stored in this collection's metadata for display and debug.

    Known cost: without the name, restaurants sharing categories and locality
    produce identical text and therefore identical vectors — 15,169 places
    collapse to 9,877 distinct ones, largest group 453 reading "A Indian
    Restaurant." Ties then break on Chroma's arbitrary insertion order. The
    fix is enrichment (tips give each place unique text) and structured
    tie-breaking on rating/distance, not putting the name back.

    Args:
        doc: a Mongo restaurant document with categories, location, rating, tips.

    Returns:
        The text to embed. Roughly 10 tokens today; dozens once tips land.
    """
    parts = []

    cats = [c.get("name") for c in (doc.get("categories") or []) if c.get("name")]
    if cats:
        parts.append("A " + ", ".join(dict.fromkeys(cats)) + ".")
    else:
        parts.append("A restaurant.")

    loc = doc.get("location") or {}
    where = loc.get("locality") or loc.get("region")
    if where:
        parts.append(f"Located in {where}.")

    rating = doc.get("rating")
    if isinstance(rating, (int, float)) and rating > 0:
        parts.append(f"Has {rating_phrase(rating)}.")

    tips = [t.get("text") for t in (doc.get("tips") or []) if t.get("text")]
    if tips:
        parts.append("What people say: " + " ".join(tips))

    return " ".join(parts)


def chunk(text: str, tokenizer, window: int) -> tuple[list[str], int]:
    """Split text into overlapping windows that fit under the model's limit.

    Measured in TOKENS, never characters. Characters-per-token varies enough
    across punctuation and rare words that a character budget either overflows
    the window — which sentence-transformers answers with silent truncation,
    not an error — or wastes a third of it.

    The window must come from SentenceTransformer.max_seq_length (256 for this
    model). AutoTokenizer.from_pretrained reports model_max_length 512 for the
    same checkpoint and is wrong about what actually truncates.

    Args:
        text: the document to split.
        tokenizer: the SentenceTransformer's own tokenizer.
        window: max tokens per chunk, already allowing for [CLS] and [SEP].

    Returns:
        (chunks, total_tokens). total_tokens is the untruncated length, which
        is what the run report's percentiles are built from — the instrument
        that catches truncation before it silently discards data.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return [], 0
    step = max(1, window - CHUNK_OVERLAP)
    chunks = []
    for start in range(0, len(ids), step):
        chunks.append(tokenizer.decode(ids[start:start + window]))
        if len(chunks) >= MAX_CHUNKS:
            break
    return chunks, len(ids)


def pool(unit_vectors: np.ndarray) -> np.ndarray:
    """Combine a restaurant's chunk vectors into one vector for the restaurant.

    Chunks arrive already unit-length (model.encode with
    normalize_embeddings=True), so this averages and re-normalizes — the mean
    of unit vectors is shorter than one.

    Normalize-then-average is NOT average-then-normalize. Skip the first step
    and longer chunks dominate by magnitude, reintroducing the exact
    text-length bias this whole rebuild exists to remove.

    Args:
        unit_vectors: (n_chunks, D), each row already unit length.

    Returns:
        (D,) unit vector representing the restaurant.
    """
    mean = unit_vectors.mean(axis=0)
    n = np.linalg.norm(mean)
    return mean / n if n > 0 else mean


def batches(seq, size):
    """Yield successive `size`-length slices of `seq`.

    itertools.batched is 3.12+; this keeps the script portable.
    """
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def main():
    """Read restaurants from Mongo, embed them, upsert into Chroma.

    Pipeline per batch: build_text -> chunk -> encode -> pool -> upsert, with
    ids set to fsqId so re-runs replace rather than duplicate.

    Only `source: "foursquare"` rows are indexed. The 5,444 yelp_seed rows
    carry real review text but embed into a different region of the space, and
    mixing the two populations means ranking incomparable things.

    Ends with a report whose token percentiles are the point: p99 climbing past
    the model window is the signal that chunking has started doing real work,
    and the absence of that instrument is how the original pipeline discarded
    93% of its review text without anyone noticing.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true", help="drop the collection first")
    ap.add_argument("--limit", type=int, default=0, help="only N restaurants (smoke test)")
    ap.add_argument("--only-missing", action="store_true",
                    help="index only restaurants that have no vector yet (run after a sync)")
    args = ap.parse_args()
    if args.only_missing and args.recreate:
        raise SystemExit("--only-missing and --recreate are contradictory")

    load_dotenv(os.path.join("..", "palate", ".env"))
    # .strip(): the value in .env has a leading space after the '=' and not
    # every loader trims it.
    mongo_url = (os.environ.get("mongo_url") or "").strip()
    if not mongo_url:
        raise SystemExit("mongo_url not found in ../palate/.env")

    # The URI carries no database name, so Node's driver silently falls back to
    # "test" — which is where the app's data actually lives. pymongo raises
    # rather than guessing, so name it explicitly.
    mongo = MongoClient(mongo_url)[os.environ.get("MONGO_DB", "test")]

    # `source` is backfilled by backfill_source.py and set going forward
    # by mapFoursquarePlace. Before that existed this filtered on field-presence
    # accidents (`stats` absent, `lastFetchedAt` present) — both of which were
    # exact by coincidence rather than by intent.
    query = {"source": "foursquare"}
    projection = {"fsqId": 1, "name": 1, "categories": 1,
                  "location": 1, "rating": 1, "tips": 1}

    cursor = mongo.restaurants.find(query, projection)
    if args.limit:
        cursor = cursor.limit(args.limit)
    docs = [d for d in cursor if d.get("fsqId")]
    print(f"{len(docs)} restaurants selected from Mongo")
    if not docs:
        raise SystemExit(
            "nothing to embed — has backfill_source.py been run?"
        )

    client = chromadb.PersistentClient(path=PERSIST_DIR)
    if args.recreate:
        try:
            client.delete_collection(COLLECTION)
            print(f"dropped '{COLLECTION}'")
        except Exception:
            pass
    col = client.get_or_create_collection(
        name=COLLECTION,
        configuration={"hnsw": {"space": "cosine"}},
    )

    if args.only_missing:
        # Cheap top-up after a Foursquare sync: everything already indexed keeps
        # the vector it has, so only genuinely new restaurants are encoded.
        # A `build_text` change still needs a full --recreate, since existing
        # vectors would otherwise be left on the old template.
        indexed = set(col.get(include=[]).get("ids") or [])
        before = len(docs)
        docs = [d for d in docs if d["fsqId"] not in indexed]
        print(f"--only-missing: {before - len(docs)} already indexed, {len(docs)} to add")
        if not docs:
            print("index is up to date")
            return

    model = SentenceTransformer(MODEL_NAME)
    tokenizer = model.tokenizer
    window = model.max_seq_length - 2
    print(f"model window: {model.max_seq_length} tokens (chunking at {window})")

    written, skipped, token_counts, chunk_counts, capped = 0, 0, [], [], 0

    for batch in batches(docs, DOC_BATCH):
        texts = [build_text(d) for d in batch]
        chunked = [chunk(t, tokenizer, window) for t in texts]

        flat = [c for cs, _ in chunked for c in cs]
        if not flat:
            skipped += len(batch)
            continue

        vecs = model.encode(flat, batch_size=ENCODE_BATCH,
                            normalize_embeddings=True, show_progress_bar=False)
        vecs = np.asarray(vecs, dtype=np.float32)

        ids, embs, documents, metas = [], [], [], []
        cursor_i = 0
        for doc, text, (cs, n_tokens) in zip(batch, texts, chunked):
            if not cs:
                skipped += 1
                continue
            block = vecs[cursor_i:cursor_i + len(cs)]
            cursor_i += len(cs)

            token_counts.append(n_tokens)
            chunk_counts.append(len(cs))
            if len(cs) >= MAX_CHUNKS:
                capped += 1

            ids.append(doc["fsqId"])
            embs.append(pool(block).tolist())
            documents.append(text[:2000])
            metas.append({
                "business_id": doc["fsqId"],
                "name": doc.get("name") or "",
                "n_chunks": len(cs),
                "n_tokens": n_tokens,
                "has_tips": bool(doc.get("tips")),
            })

        if ids:
            col.upsert(ids=ids, embeddings=embs, documents=documents, metadatas=metas)
            written += len(ids)
            print(f"  upserted {written}/{len(docs)}")

    pct = np.percentile(token_counts, [50, 90, 99, 100]).astype(int) if token_counts else [0] * 4
    print(f"\ncollection '{COLLECTION}' now holds {col.count()} vectors")
    print(f"written={written}  skipped={skipped}  hit MAX_CHUNKS={capped}")
    print(f"tokens per restaurant  p50={pct[0]}  p90={pct[1]}  p99={pct[2]}  max={pct[3]}")
    print(f"chunks per restaurant  mean={np.mean(chunk_counts):.2f}  max={max(chunk_counts)}")
    print(f"single-chunk restaurants: {sum(1 for c in chunk_counts if c == 1)}/{len(chunk_counts)}")


if __name__ == "__main__":
    main()