"""One-time export: join the Yelp restaurants that were embedded into
chroma_db (data/restaurant_docs.parquet) with their location/category/rating
info (data/mapped.parquet), and write a JSON array shaped for the Palate
Mongoose `Restaurant` model (see palate/src/models/restaurantModel.js).

Run once:
    ./.venv/Scripts/python.exe export_seed.py

Output:
    ../palate/scripts/data/restaurants.seed.json
"""
import json
import math
import os

import pyarrow  # noqa: F401  (see service.py for why this import order matters)
import pyarrow.dataset  # noqa: F401
import pandas as pd

OUT_DIR = os.path.join("..", "palate", "scripts", "data")
OUT_PATH = os.path.join(OUT_DIR, "restaurants.seed.json")


def clean(v):
    """NaN/NaT -> None so json.dump produces valid `null` instead of NaN.

    pandas represents missing numeric values as float NaN, and json.dump
    happily emits a bare `NaN` token — which is not valid JSON and blows up on
    the Node side when seedRestaurants.mjs parses the file.

    Args:
        v: any cell value out of a DataFrame.

    Returns:
        None if the value is missing, otherwise the value unchanged.
    """
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def main():
    """Join Yelp business metadata with the embedded set and write the seed JSON.

    `mapped.parquet` has one row per (business, matched category), so
    categories are collapsed back to one list per business. Only businesses
    present in `restaurant_docs.parquet` are exported, which was meant to keep
    the seeded dashboard and the vector index in step.

    Two reshapes worth knowing about: Yelp's `business_id` is written into the
    `fsqId` field (it is NOT a real Foursquare id — a deliberate stopgap), and
    Yelp's 0-5 stars are doubled to match the schema's 0-10 rating comment.

    Historical: these 5,444 rows are the `source: "yelp_seed"` population.
    rebuild_index.py excludes them, because template-sentence and review-text
    embeddings are not comparable in the same ranking.
    """
    mapped = pd.read_parquet("data/mapped.parquet")
    docs = pd.read_parquet("data/restaurant_docs.parquet")

    # Only seed restaurants that are actually searchable via /recommend,
    # so every dashboard entry has a matching embedding and vice versa.
    embedded_ids = set(docs["business_id"])
    mapped = mapped[mapped["business_id"].isin(embedded_ids)].copy()

    missing = embedded_ids - set(mapped["business_id"])
    if missing:
        print(f"Warning: {len(missing)} embedded business_ids have no row in mapped.parquet, skipping them")

    base_cols = ["business_id", "name", "address", "city", "state", "postal_code",
                 "latitude", "longitude", "stars", "review_count", "is_open"]
    base = mapped.drop_duplicates("business_id")[base_cols].set_index("business_id")

    # mapped.parquet has one row per (business, matched category) — collect
    # every category a business matched into a single list per business.
    categories_by_biz = (
        mapped.drop_duplicates(["business_id", "fsqid"])
        .groupby("business_id")[["fsqid", "fsq_name"]]
        .apply(lambda g: [
            {"fsqCategoryId": str(int(row.fsqid)), "name": row.fsq_name}
            for row in g.itertuples()
        ])
    )

    records = []
    for business_id, row in base.iterrows():
        cats = categories_by_biz.get(business_id, [])
        address = clean(row["address"]) or ""
        city = clean(row["city"]) or ""
        state = clean(row["state"]) or ""
        postcode = clean(row["postal_code"]) or ""
        formatted = ", ".join(p for p in [address, city, f"{state} {postcode}".strip()] if p.strip())

        lat = clean(row["latitude"])
        lng = clean(row["longitude"])
        stars = clean(row["stars"])
        is_open = clean(row["is_open"])

        records.append({
            # Yelp business_id, not a real Foursquare id — see README note
            # added alongside this script for why the field is still fsqId.
            "fsqId": business_id,
            "name": clean(row["name"]),
            "categories": cats,
            "cuisine": sorted({c["name"] for c in cats}),
            "location": {
                "formattedAddress": formatted,
                "address": address or None,
                "locality": city or None,
                "region": state or None,
                "postcode": postcode or None,
                "country": "US",
            },
            "geocodes": {"latitude": lat, "longitude": lng},
            "geo": {
                "type": "Point",
                "coordinates": [lng, lat] if lat is not None and lng is not None else None,
            },
            "rating": stars * 2 if stars is not None else None,  # Yelp 0-5 -> Foursquare-like 0-10
            "stats": {"totalRatings": clean(row["review_count"])},
            "verified": bool(is_open) if is_open is not None else None,
        })

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    print(f"Wrote {len(records)} restaurants to {OUT_PATH}")


if __name__ == "__main__":
    main()
