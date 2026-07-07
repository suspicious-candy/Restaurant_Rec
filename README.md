# Restaurant Recommender

Exploratory data analysis and recommendation work on the [Yelp Open Dataset](https://www.yelp.com/dataset).

## Contents
- `data-explorer.ipynb` — data loading, missing-value checks, and correlation analysis of the Yelp `business` and `user` tables.
- `main.py` — entry point / scratch script.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install pandas numpy seaborn matplotlib kagglehub
```

## Data
The Yelp dataset is **not** included in this repo (it's large). Download it via
[kagglehub](https://github.com/Kaggle/kagglehub) or from the
[Yelp dataset page](https://www.yelp.com/dataset), then point the notebook's
`path` variable at your local copy.
