import os
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "Appliances.json")

RANDOM_STATE = 42
MAX_REVIEWS = 20000      # keep runtime manageable on full dataset
MIN_USER_REVIEWS = 2     # users must have at least 2 reviews in the working subset
MIN_ITEM_REVIEWS = 2     # items must have at least 2 reviews in the working subset
ALPHA = 0.5              # weight for sentiment-enhanced ratings
TOP_N = 5

analyzer = SentimentIntensityAnalyzer()


def label_sentiment_score(text: str) -> float:
    return analyzer.polarity_scores(str(text))["compound"]  # [-1, +1]


print("Loading dataset...")
use_cols = ["reviewerID", "asin", "overall", "summary", "reviewText"]
df = pd.read_json(DATA_PATH, lines=True)
df = df[use_cols].copy()
df = df[df["reviewText"].notnull() & df["reviewerID"].notnull() & df["asin"].notnull()].copy()
df["summary"] = df["summary"].fillna("")
df["text"] = (df["summary"].astype(str) + " " + df["reviewText"].astype(str)).str.strip()

print(f"Full dataset rows after cleaning: {len(df)}")

# Sample from the full dataset so the baseline remains fast and reproducible
if len(df) > MAX_REVIEWS:
    df = df.sample(n=MAX_REVIEWS, random_state=RANDOM_STATE)
    print(f"Working subset sampled from full data: {len(df)} reviews")
else:
    print(f"Using all available reviews: {len(df)}")

# Keep only users/items with at least a small amount of interaction in the working subset
user_counts = df["reviewerID"].value_counts()
item_counts = df["asin"].value_counts()

df = df[
    df["reviewerID"].isin(user_counts[user_counts >= MIN_USER_REVIEWS].index)
    & df["asin"].isin(item_counts[item_counts >= MIN_ITEM_REVIEWS].index)
].copy()

print(f"Rows after user/item frequency filter: {len(df)}")
print(f"Unique users: {df['reviewerID'].nunique()}")
print(f"Unique items: {df['asin'].nunique()}")

# sentiment-enhanced ratings
df["sentiment_score"] = df["text"].apply(label_sentiment_score)
df["rating_adj"] = (df["overall"] + ALPHA * df["sentiment_score"]).clip(1, 5)

# user-item matrix (adjusted ratings)
R = df.pivot_table(index="reviewerID", columns="asin", values="rating_adj", aggfunc="mean").fillna(0)
print(f"User-item matrix shape: {R.shape}")

if R.shape[0] == 0 or R.shape[1] == 0:
    raise ValueError("User-item matrix is empty after filtering. Increase MAX_REVIEWS or lower the min frequency thresholds.")

# item-item cosine similarity
S = cosine_similarity(R.T)
items = R.columns.to_list()


def recommend(user_id: str, top_n: int = TOP_N):
    if user_id not in R.index:
        return []

    user_ratings = R.loc[user_id].values
    liked_idx = np.where(user_ratings >= 4.0)[0]
    if len(liked_idx) == 0:
        return []

    scores = S[liked_idx].mean(axis=0)
    scores[liked_idx] = -1  # do not recommend already-liked items
    rec_idx = np.argsort(scores)[::-1][:top_n]
    return [(items[i], float(scores[i])) for i in rec_idx if scores[i] > 0]


# Pick a demo user with the most non-zero ratings in the working subset
interaction_counts = (R > 0).sum(axis=1)
some_user = interaction_counts.sort_values(ascending=False).index[0]

print("\nDemo user:", some_user)
print("Observed items for demo user:", int(interaction_counts.loc[some_user]))
print("Top recommendations:", recommend(some_user, top_n=TOP_N))
