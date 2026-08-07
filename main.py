"""PyCharm's default scaffold. Not part of the recommender — nothing imports it.

The real entry points are:
    service.py             FastAPI read API (uvicorn service:app --port 8000)
    rebuild_index.py       builds the vector index from Mongo (the only writer)
    backfill_source.py     one-time: stamps restaurants with their provenance
    experiment_template.py offline comparison of embedding text templates
    export_seed.py         one-time: Yelp parquet -> palate seed JSON
    save_clean_data.py     one-time: raw Yelp JSON -> cleaned parquet

Safe to delete.
"""


def print_hi(name):
    """Print a greeting. IDE scaffold, unused.

    Args:
        name: whatever to greet.
    """
    print(f'Hi, {name}')


if __name__ == '__main__':
    print_hi('PyCharm')
