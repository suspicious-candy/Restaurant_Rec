"""Reproduce the data-explorer cleaning pipeline and persist all cleaned
DataFrames to disk as parquet files under ./data/.

Run once:  ./.venv/Scripts/python.exe save_clean_data.py
Then load in any notebook, e.g.:
    reviews_3UP = pd.read_parquet("data/reviews_3UP.parquet")
"""
import os
import kagglehub
import pandas as pd
import numpy as np
from ftlangdetect import detect as ft_detect

OUT = "data"
os.makedirs(OUT, exist_ok=True)

path = kagglehub.dataset_download("yelp-dataset/yelp-dataset")
print("Dataset path:", path)


def load(name):
    return pd.concat(
        pd.read_json(f"{path}/yelp_academic_dataset_{name}.json",
                     lines=True, chunksize=100_000, nrows=500_000)
    )


# ---- load raw ----
print("Loading raw JSON...")
reviews = load("review")
business = load("business")
user = load("user")
tip = load("tip")
checkin = pd.read_json(f"{path}/yelp_academic_dataset_checkin.json",
                       lines=True, convert_dates=False)

# ---- clean business ----
business["missingAddress"] = business["address"].isin(['', 'None']).astype(int)
business["missingpostal_code"] = business["postal_code"].isin(['', 'None']).astype(int)
business = business.dropna(subset=["categories"]).copy()
business["cat_list"] = business["categories"].str.split(", ")
exploded = business.explode("cat_list")
exploded["cat_list"] = exploded["cat_list"].str.strip()

YELP_TO_FSQ = {
    "Bakeries": (13002, "Bakery"), "Bars": (13003, "Bar"),
    "Breakfast & Brunch": (13028, "Breakfast Spot"), "Cafes": (13032, "Café"),
    "Coffee & Tea": (13034, "Coffee Shop"), "Bubble Tea": (13033, "Bubble Tea Shop"),
    "Desserts": (13040, "Dessert Shop"),
    "Ice Cream & Frozen Yogurt": (13046, "Ice Cream Parlor"),
    "Donuts": (13043, "Donut Shop"), "Juice Bars & Smoothies": (13059, "Juice Bar"),
    "Fast Food": (13145, "Fast Food Restaurant"), "Food Court": (13052, "Food Court"),
    "Food Trucks": (13054, "Food Truck"), "Pizza": (13064, "Pizzeria"),
    "Steakhouses": (13383, "Steakhouse"), "American (New)": (13068, "American"),
    "American (Traditional)": (13068, "American"), "Asian Fusion": (13072, "Asian"),
    "Chinese": (13099, "Chinese"), "Indian": (13199, "Indian"),
    "Italian": (13236, "Italian"), "Japanese": (13263, "Japanese"),
    "Korean": (13289, "Korean"), "Mediterranean": (13302, "Mediterranean"),
    "Mexican": (13303, "Mexican"), "Middle Eastern": (13306, "Middle Eastern"),
    "Thai": (13352, "Thai"), "Vietnamese": (13377, "Vietnamese"),
    "Seafood": (13338, "Seafood"), "Sushi Bars": (13276, "Sushi"),
    "Vegan": (13385, "Vegan / Vegetarian"), "Vegetarian": (13385, "Vegan / Vegetarian"),
}
exploded["fsqid"] = exploded["cat_list"].map(lambda c: YELP_TO_FSQ.get(c, (None, None))[0])
exploded["fsq_name"] = exploded["cat_list"].map(lambda c: YELP_TO_FSQ.get(c, (None, None))[1])
mapped = exploded.dropna(subset=["fsqid"])

# ---- clean user ----
user = user.drop(columns=["funny", "cool", "elite", "friends", "fans",
                          "compliment_cool", "compliment_funny", "compliment_plain",
                          "compliment_photos", "compliment_profile", "compliment_cute"])

# ---- clean reviews (drop cols, word count, language filter, >=25 words) ----
reviews = reviews.drop(columns=["cool", "funny"])
reviews["wordsInDesc"] = reviews["text"].str.split().str.len()


def fast_lang(text):
    if not isinstance(text, str) or len(text.strip()) < 20:
        return None
    return ft_detect(text.replace("\n", " "))["lang"]


print("Detecting language over reviews (this takes a few minutes)...")
reviews["lang"] = reviews["text"].map(fast_lang)
reviews = reviews[reviews["lang"] == "en"]
reviews_3UP = reviews[reviews["wordsInDesc"] >= 25]

# ---- clean checkin ----
checkin["checkinCount"] = checkin["date"].str.split(",").str.len()
split_dates = checkin["date"].str.split(", ")
checkin["firstCheckin"] = pd.to_datetime(split_dates.str[0])
checkin["latestCheckin"] = pd.to_datetime(split_dates.str[-1])
checkin = checkin.drop(columns=["date"])

# ---- merge reviews_3UP with restaurant/category info ----
food_biz = mapped[["business_id", "fsqid", "fsq_name", "stars"]].drop_duplicates("business_id")
reviews_3UP = reviews_3UP.merge(food_biz, on="business_id", how="inner")

# ---- build restaurant_docs for embedding ----
restaurant_docs = (
    reviews_3UP.groupby("business_id")
    .agg(all_text=("text", lambda x: " ".join(x.head(50))),
         avg_stars=("stars", "mean"),
         n_reviews=("text", "count"))
    .reset_index()
)

# ---- save everything ----
to_save = {
    "reviews_clean": reviews,
    "reviews_3UP": reviews_3UP,
    "restaurant_docs": restaurant_docs,
    "business_clean": business,
    "mapped": mapped,
    "user_clean": user,
    "tip_clean": tip,
    "checkin_clean": checkin,
}
print("\nSaving cleaned files to ./data/ :")
for name, df in to_save.items():
    fp = os.path.join(OUT, f"{name}.parquet")
    df.to_parquet(fp)
    print(f"  {name+'.parquet':30s} rows={len(df):>8,}  cols={df.shape[1]}")

print("\nDone.")
