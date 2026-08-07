"""One-time: stamp every restaurant with where it came from.

Two importers write into the same `restaurants` collection:

  * export_seed.py -> seedRestaurants.mjs   (Yelp academic dataset, 5,444 rows)
  * mapFoursquarePlace                      (Foursquare Places API, 15,169 rows)

Telling them apart mattered — they carry completely different text, so they
embed into different regions of the vector space and must not be ranked against
each other. But nothing recorded the provenance, so it had to be *inferred*
from which optional fields each importer happened to write (`stats` present =
Yelp). That inference is exact today purely by coincidence: add `stats` to the
Foursquare mapper and every downstream filter breaks silently, surfacing as bad
recommendations rather than an error.

This turns the inference into a stored fact, once. From here on `source` is set
at write time by mapFoursquarePlace and seedRestaurants.mjs.

Run:  ./.venv/Scripts/python.exe backfill_source.py [--dry-run]
"""
import argparse
import os

from dotenv import load_dotenv
from pymongo import MongoClient

# `stats` and `verified` are written only by export_seed.py; `tel`/`website`
# only by mapFoursquarePlace. Field presence is the only signal available
# before this script runs.
YELP = {"stats": {"$exists": True}}
FOURSQUARE = {"stats": {"$exists": False}}


def main():
    """Stamp every restaurant with `source`, inferring it once from field presence.

    Asserts the two filters partition the collection before writing anything —
    they are complements, so a mismatch means the collection changed mid-count.

    Idempotent: re-running just rewrites the same values. Reversible with
    `db.restaurants.updateMany({}, {$unset: {source: ""}})`.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="count only, write nothing")
    args = ap.parse_args()

    load_dotenv(os.path.join("..", "palate", ".env"))
    mongo_url = (os.environ.get("mongo_url") or "").strip()
    if not mongo_url:
        raise SystemExit("mongo_url not found in ../palate/.env")

    col = MongoClient(mongo_url)[os.environ.get("MONGO_DB", "test")]["restaurants"]

    total = col.count_documents({})
    n_yelp = col.count_documents(YELP)
    n_fsq = col.count_documents(FOURSQUARE)

    print(f"total       {total:>8,}")
    print(f"yelp_seed   {n_yelp:>8,}")
    print(f"foursquare  {n_fsq:>8,}")

    # The two filters are complements, so this can only fail if the collection
    # changed under us mid-count. Cheap assertion, catches a real class of bug.
    if n_yelp + n_fsq != total:
        raise SystemExit(f"split does not cover the collection: {n_yelp}+{n_fsq} != {total}")

    if args.dry_run:
        print("\ndry run — nothing written")
        return

    r1 = col.update_many(YELP, {"$set": {"source": "yelp_seed"}})
    r2 = col.update_many(FOURSQUARE, {"$set": {"source": "foursquare"}})
    print(f"\nstamped yelp_seed={r1.modified_count:,}  foursquare={r2.modified_count:,}")

    for value in ("yelp_seed", "foursquare"):
        print(f"  source={value:<11s} {col.count_documents({'source': value}):>8,}")
    missing = col.count_documents({"source": {"$exists": False}})
    print(f"  source missing        {missing:>8,}")


if __name__ == "__main__":
    main()
