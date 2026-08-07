# Restaurant Recommender

The recommendation service behind [Palate](../palate). A FastAPI app that ranks
restaurants by semantic similarity — for one person, or for a group of people
who have to agree on dinner.

Originally exploratory work on the [Yelp Open Dataset](https://www.yelp.com/dataset);
that data now serves as a seed and a test fixture, and the live index is built
from Palate's own MongoDB.

---

## How it fits together

```
Foursquare API ──► MongoDB `restaurants`  ◄── source of truth for text
                          │
                          │  rebuild_index.py   (the index's ONLY writer)
                          ▼
                   Chroma `places_v2`     ◄── derived index, disposable
                          │
                          │  service.py     (read-only)
                          ▼
              POST /recommend  ·  POST /recommend/group
                          ▲
                          │  HTTP
                   Palate (Next.js)
```

**Mongo is the source of truth; Chroma is a derived index.** That direction is
what makes the embedding model, the chunking, and the text template all
changeable later — none of which is possible if accumulated text lives only
inside the vector store. The index can be deleted and rebuilt at any time.

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate                      # Windows
pip install fastapi uvicorn pydantic numpy pandas pyarrow tqdm python-dotenv \
            pymongo chromadb sentence-transformers \
            langchain-chroma langchain-huggingface
```

Mongo credentials are read from `../palate/.env` (`mongo_url`) — one secret in
one place, rather than a copy that drifts. The URI carries no database name, so
the scripts default to `test`, which is where Node's driver has been writing.
Override with `MONGO_DB` if that changes.

Run the service **from this directory** — Chroma's `persist_directory` is the
relative path `chroma_db`:

```bash
./.venv/Scripts/python.exe -m uvicorn service:app --port 8000
```

Port 8000 matters: Palate's `RECOMMENDER_URL` defaults to
`http://localhost:8000`. Without the service running, `/api/Restaurants/nearby`
logs `ECONNREFUSED` and silently degrades to an unranked list.

---

## API

### `POST /recommend`

```jsonc
{ "query": "A Italian, Pizza restaurant", "k": 5, "candidateIds": ["4f24…"] }
→ { "businessIds": ["4f24…", …] }
```

### `POST /recommend/group`

```jsonc
{
  "queries":  ["spicy North Indian curry", "Chinese noodles"],  // or…
  "vectors":  [[0.01, …]],          // pre-computed per-person taste vectors
  "weights":  [1.0, 1.0],           // optional, defaults to equal
  "candidateIds": ["4f24…"],        // required unless strategy is "centroid"
  "strategy": "blend",              // blend | min | mean | centroid
  "alpha":    0.5,
  "k":        20
}
→ {
  "businessIds": [...],
  "results": [{ "businessId", "score", "memberSims", "worstMemberSim" }],
  "agreement": 0.24,     // mean pairwise cosine between members
  "consensus": 0.78,     // ||mean of unit member vectors||
  "memberCount": 2
}
```

`queries` and `vectors` are concatenated in that order, and `weights` must
follow it. Accepting both is deliberate: callers send preference text today,
and can send per-person taste vectors later with no change to this endpoint.

**Always pass `candidateIds`.** Retrieval should be a *filter* — the caller
decides what is reachable (nearby, open, in budget) and the vector search only
*orders* what survives. See "Filter, then rank" below.

`score` is higher-is-better on every strategy, but **ordinal within one
response**: centroid scores are cosine similarities in [-1, 1]; blend and min
scores are built from z-scores and can exceed 1. Fine for ranking, wrong for
thresholding.

---

## Files

| file | purpose |
|---|---|
| `service.py` | FastAPI read API. Never writes the index. |
| `rebuild_index.py` | Builds `places_v2` from Mongo. The only writer. |
| `backfill_source.py` | One-time: stamps every restaurant with its provenance. |
| `experiment_template.py` | Offline comparison of embedding text templates. |
| `export_seed.py` | One-time: Yelp parquet → `palate/scripts/data/restaurants.seed.json`. |
| `save_clean_data.py` | One-time: raw Yelp JSON → cleaned parquet under `data/`. |
| `main.py` | PyCharm scaffold. Unused, safe to delete. |
| `*.ipynb` | Exploratory notebooks from the original EDA. |

### Rebuilding the index

```bash
./.venv/Scripts/python.exe rebuild_index.py --limit 200      # smoke test
./.venv/Scripts/python.exe rebuild_index.py --only-missing   # top-up after a sync
./.venv/Scripts/python.exe rebuild_index.py --recreate       # full rebuild
```

`--only-missing` adds restaurants that have no vector yet and exits before even
loading the model if there are none. A change to `build_text` needs
`--recreate`, since existing vectors would otherwise be left on the old template.

**Restart the service after any rebuild.** Chroma's `PersistentClient` caches
in-process, so a running service cannot see a collection another process
dropped and rebuilt.

---

## The index

`places_v2` — 15,169 vectors, one per restaurant, dim 384, **cosine** space, all
unit length, ids are `fsqId`, `business_id` in metadata on every vector (which
is what makes candidate filtering work). Covers Mongo's
`{ source: "foursquare" }` population exactly.

Model: `sentence-transformers/all-MiniLM-L6-v2`, **256-token** window.

A dead `langchain` collection is still on disk (~490MB). It held 20,613 vectors
in L2 space, mixing Yelp review text with Foursquare template sentences — two
populations that were never comparable. Delete it once nothing references it.

---

## Design decisions, and the measurements behind them

These are the non-obvious ones. Each produced plausible-looking output while
being wrong, so none of them surfaced without deliberate measurement.

### Filter, then rank

`nearby/route.ts` used to take the 50 nearest restaurants from Mongo, separately
take the top-50 from Chroma across the whole corpus, and intersect. Measured
overlap: **0**, at k=50 and k=200; 1 at k=1000. Two independent top-50 draws
from a 15,169-item corpus are expected to share 0.16 items. Because of an
`if (ranked.length > 0)` guard the route silently fell back every time — the
recommender had never affected the dashboard.

**Never intersect two independently-retrieved top-K lists.** Retrieval is a
filter; ranking is an ordering. Ranking must also never *drop* candidates: rows
with no vector are appended in distance order rather than disappearing.

### The embedded text excludes the restaurant's name

Three templates, 15,169 restaurants × 5 cuisine probes, top-20:

| template | category@20 ↑ | nameleak@20 ↓ |
|---|---|---|
| control (name + category) | 74% | 88% |
| cat_x3 (category repeated) | 78% | 87% |
| **no_name** | **95%** | **41%** |

Worst control case: `"pizza and pasta"` scored **5%** category precision with
100% name leakage — 20 results with a pizza word in the *name*, one of which
was actually a pizza place.

Repeating the category to outweigh the name barely helped. A transformer is not
bag-of-words: duplicating a token does not linearly raise its weight, and a
distinctive proper noun keeps winning. Removing it is what works.

Finding a place *by name* is a different job, and belongs in lexical search over
Mongo.

**Known cost:** without names, restaurants sharing categories and locality
produce identical text and therefore identical vectors — 15,169 places collapse
to 9,877 distinct ones, largest group 453 reading `"A Indian Restaurant."` Ties
break on Chroma's arbitrary insertion order. The fix is enrichment and
structured tie-breaking, not putting the name back.

### Group aggregation: least misery, not the average

For unit vectors in cosine space, `cos(centroid, d) == mean_i cos(v_i, d)` over
a constant — so **centroid ranking *is* mean ranking**, identically. Verified:
the `centroid` and `mean` strategies return the same top pick and the same
worst-member rank in every test.

That makes the original centroid approach the *additive utilitarian* social
choice function, adopted by accident because `similarity_search` takes exactly
one vector. Averaging is known to fail on polarized groups.

Once the candidate set is bounded, the whole N-members × M-candidates matrix
fits in microseconds and any aggregation is affordable. Measured — **worst-member
rank of the group's #1 pick**, lower is better, over 67 NYC candidates:

| group | centroid / mean | min / blend |
|---|---|---|
| vegan + steakhouse | 18 | **1** |
| vegan + steak + Sichuan | 64 | **12** |
| two Indian queries (aligned) | 17 | 9–10 |

The first row is the argument in two restaurant names: mean picked
**Sweetgreen** (a salad chain — the vegan's answer, nothing for the steak fan);
min picked **Superiority Burger** (a vegetarian burger restaurant) — the
compromise a thoughtful human would name.

Rows are z-scored per member first, or the member whose taste vector sits
furthest from the corpus centre sets `min` for every candidate — looking like
the pickiest person purely because of where their vector sits. Caveat: this
optimizes *relative* satisfaction, so raw worst-member cosine can dip slightly
for already-aligned groups even as rank improves.

`blend` defaults to α=0.5 because pure `min` is brittle here: one outlier member
swings everything, and vector collisions make minima tie constantly.

### `agreement` over `consensus`

`consensus` is `‖mean of unit member vectors‖`. `agreement` inverts it in closed
form to the mean pairwise cosine:

```
agreement = (n · consensus² − 1) / (n − 1)
```

Verified against ground-truth pairwise cosines to machine epsilon, N=2 and N=3.
Two reasons to prefer it: ~3.2× the resolution in the 0.2–0.4 band where real
sentence pairs land (the square root is very flat there), and it is comparable
across group sizes — `consensus` for unrelated members floors near `1/√N`, so
the same value means "divided" for a pair and "aligned" for six.

### One writer, one template

A `/embed` endpoint used to let callers push their own text. That made two
writers with two templates — the caller's kept the restaurant name — and wrote
vectors Mongo could not reproduce, so any `--recreate` silently changed them.
Removed. Sync writes Mongo; `rebuild_index.py --only-missing` builds vectors.

---

## Known limitations

- **Text is thin.** p50 is ~10 tokens per restaurant. The index can currently
  distinguish cuisines and little else — it cannot tell a great Italian
  restaurant from a mediocre one, because nothing in the vector encodes
  quality, atmosphere, or price. Backfilling Foursquare tips is the fix.
- **48% of restaurants share a vector** with at least one other. See above.
- **Coverage is capped by a hardcoded limit**, not by geography. Both sync paths
  request `limit=50` per call and never paginate, so each synced city has
  exactly 50 restaurants. 51% of the corpus sits in localities with fewer than
  50 places; a random anchor sees a median of 50 candidates within 30 km.
- **No per-person taste vectors yet.** `/recommend/group` accepts them via
  `vectors`; nothing builds them.
- **No evaluation harness.** Every number above came from a throwaway script.
  Promoting them into a repeatable suite with a fixed probe set is what would
  make "did that change help?" answerable.

---

## Gotchas

- **Import `pyarrow` before torch/transformers.** On Windows they bundle
  separate native OpenMP runtimes; if pyarrow's DLL loads second the process
  dies with a native access violation (`0xC0000005`) and uvicorn never starts.
  Every script here does this first — keep it that way.
- **`SentenceTransformer.max_seq_length` is the real limit** (256 here).
  `AutoTokenizer.from_pretrained` reports `model_max_length: 512` for the same
  checkpoint and is wrong about what actually truncates — and truncation is
  silent, never an exception.
- **Chroma's distance metric is fixed at collection creation.** Changing it
  means a new collection.
- **Chroma caches per process.** Cross-process writes are invisible until a
  fresh process opens the store.

---

## Historical: the Yelp dataset

`save_clean_data.py` and `export_seed.py` built the original 5,444-restaurant
seed from the Yelp Open Dataset — real review text, reshaped so Yelp's
`business_id` lives in the `fsqId` field (deliberate stopgap, not a real
Foursquare id) and stars are doubled onto the schema's 0–10 scale. Those rows
are Mongo's `source: "yelp_seed"` population and are **excluded** from
`places_v2`.

They remain the only rich-text corpus available, which makes them useful as a
development fixture — the only way to test that centroid maths and group
aggregation behave before the live data is dense enough to tell a working
implementation from a broken one.

The dataset itself is not in this repo. Download via
[kagglehub](https://github.com/Kaggle/kagglehub) or the
[Yelp dataset page](https://www.yelp.com/dataset).
