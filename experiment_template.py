"""Does the restaurant NAME dominate the embedding, and does dropping it help?

Symptom that prompted this: 'sushi omakase' returned Sushi Time, Sushi Iwa,
Sushi Inc — matches on the name string, not on what the place serves. With a
~15-token document ("Name. Category: X. Located in Y.") the proper noun leads
and carries a large share of the signal, so a great Japanese place called Nobu
is unreachable and anything generically named is invisible.

Three templates are scored against the same probes:

  control   name + category + locality        (what rebuild_index.py builds now)
  no_name   category + locality               (name removed entirely)
  cat_x3    name + category repeated 3x       (name kept but outweighed)

Two metrics per probe, both over the top 20 hits:

  category@20  fraction whose Mongo categories match the cuisine asked for.
               This is the number that matters — precision of the retrieval.
  nameleak@20  fraction whose NAME contains a query keyword. High values mean
               the index is doing string matching on proper nouns.

Nothing is written to Chroma. Every text here is under the 254-token window
(p99 was 25), so chunk-then-pool reduces to a single normalized encode and the
search is a plain dot product over unit vectors.

Run:  ./.venv/Scripts/python.exe experiment_template.py
"""
import pyarrow  # noqa: F401
import pyarrow.dataset  # noqa: F401

import os
import re

import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

TOP_K = 20

PROBES = [
    {"q": "spicy Indian curry house",
     "cat": r"indian|curry|tandoor|biryani|north indian|south indian",
     "kw": ["indian", "curry", "spice", "spicy", "tandoor"]},
    {"q": "coffee and pastries",
     "cat": r"coffee|caf|bakery|dessert|tea",
     "kw": ["coffee", "cafe", "café", "brew", "bakery", "pastr"]},
    {"q": "sushi omakase",
     "cat": r"sushi|japanese|asian",
     "kw": ["sushi", "omakase", "japan"]},
    {"q": "pizza and pasta",
     "cat": r"pizza|italian",
     "kw": ["pizza", "pasta", "italian"]},
    {"q": "fast food burgers and fries",
     "cat": r"fast food|burger|sandwich",
     "kw": ["burger", "fries", "fast food", "mcdonald", "kfc"]},
]


def texts_for(doc):
    """Build all three candidate templates for one restaurant.

    Args:
        doc: a Mongo restaurant document with name, categories, location.

    Returns:
        (control, no_name, cat_x3) — name+category, category only, and
        name+category repeated three times. Returned as a tuple so all three
        stay in lockstep and indices line up with the encode loop in main().
    """
    name = doc.get("name") or "Unnamed restaurant"
    cats = list(dict.fromkeys(
        c.get("name") for c in (doc.get("categories") or []) if c.get("name")
    ))
    cat_str = ", ".join(cats) if cats else "restaurant"

    loc = doc.get("location") or {}
    where = loc.get("locality") or loc.get("region")
    tail = f" Located in {where}." if where else ""

    return (
        f"{name}. Category: {cat_str}.{tail}",
        f"A {cat_str}.{tail}",
        f"{name}. Category: {cat_str}. Serves {cat_str}. Known for {cat_str}.{tail}",
    )


def main():
    """Score all three templates over the whole corpus and print the comparison.

    Nothing is written to Chroma — the corpus is embedded three times in memory
    (15,169 x 384 float32 is ~23MB per variant) and searched with a plain dot
    product, which is exact rather than approximate and needs no collection.

    Every text here is far under the 254-token window, so chunk-then-pool would
    reduce to a single normalized encode; it is skipped as mathematically
    identical.

    Result that settled it: control 74% category@20 / 88% nameleak@20, cat_x3
    78% / 87%, no_name 95% / 41%.
    """
    load_dotenv(os.path.join("..", "palate", ".env"))
    mongo = MongoClient((os.environ["mongo_url"]).strip())[os.environ.get("MONGO_DB", "test")]

    docs = list(mongo.restaurants.find(
        {"source": "foursquare"},
        {"fsqId": 1, "name": 1, "categories": 1, "location": 1},
    ))
    print(f"{len(docs):,} restaurants loaded\n")

    names = [(d.get("name") or "").lower() for d in docs]
    cat_blobs = [
        " ".join(c.get("name", "") for c in (d.get("categories") or [])).lower()
        for d in docs
    ]

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    variants = {}
    for i, label in enumerate(("control", "no_name", "cat_x3")):
        corpus = [texts_for(d)[i] for d in docs]
        print(f"encoding {label} ...")
        variants[label] = np.asarray(
            model.encode(corpus, batch_size=256, normalize_embeddings=True,
                         show_progress_bar=False),
            dtype=np.float32,
        )

    qvecs = {p["q"]: model.encode([p["q"]], normalize_embeddings=True)[0].astype(np.float32)
             for p in PROBES}

    print(f"\n{'probe':30s} {'variant':9s} {'category@20':>12s} {'nameleak@20':>12s}")
    print("-" * 68)

    totals = {v: {"cat": [], "leak": []} for v in variants}

    for p in PROBES:
        cat_re = re.compile(p["cat"])
        for label, matrix in variants.items():
            # Unit vectors on both sides, so a dot product IS cosine similarity.
            top = np.argpartition(-(matrix @ qvecs[p["q"]]), TOP_K)[:TOP_K]

            cat_hits = sum(1 for i in top if cat_re.search(cat_blobs[i]))
            leak = sum(1 for i in top if any(k in names[i] for k in p["kw"]))

            totals[label]["cat"].append(cat_hits / TOP_K)
            totals[label]["leak"].append(leak / TOP_K)
            print(f"{p['q']:30s} {label:9s} {cat_hits / TOP_K:11.0%} {leak / TOP_K:11.0%}")
        print()

    print("=" * 68)
    print(f"{'MEAN':40s} {'category@20':>12s} {'nameleak@20':>12s}")
    for label in variants:
        c = np.mean(totals[label]["cat"])
        l = np.mean(totals[label]["leak"])
        print(f"{label:40s} {c:11.0%} {l:11.0%}")

    print("\n--- top 5 neighbours per variant ---")
    for p in PROBES[:3]:
        print(f"\n  {p['q']!r}")
        for label, matrix in variants.items():
            sims = matrix @ qvecs[p["q"]]
            top = np.argsort(-sims)[:5]
            shown = ", ".join(f"{docs[i]['name'][:26]}" for i in top)
            print(f"    {label:9s} {shown}")


if __name__ == "__main__":
    main()
