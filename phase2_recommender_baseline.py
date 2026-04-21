import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CANDIDATE_NAMES = (
    os.path.join("Appliances.json", "Appliances.json"),
    "Appliances.json.gz",
    "Appliances.json",
    "Appliances_5.json.gz",
    "Appliances_5.json",
)


def resolve_appliances_data_path(caller_file: str | None = None) -> str:
    env_path = os.environ.get("APPLIANCES_DATA_PATH")
    if env_path:
        candidate = Path(env_path).expanduser().resolve()
        if candidate.exists():
            return str(candidate)

    search_roots = []
    if caller_file:
        script_dir = Path(caller_file).resolve().parent
        search_roots.extend((script_dir, script_dir / "data", script_dir.parent))

    cwd = Path.cwd().resolve()
    search_roots.extend((cwd, cwd / "data"))

    seen = set()
    ordered_roots = []
    for root in search_roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            ordered_roots.append(root)

    for root in ordered_roots:
        for name in DEFAULT_CANDIDATE_NAMES:
            candidate = root / name
            if candidate.exists():
                return str(candidate)

    searched = [str(root / name) for root in ordered_roots for name in DEFAULT_CANDIDATE_NAMES]
    raise FileNotFoundError(
        "Could not find the appliances dataset. "
        "Set APPLIANCES_DATA_PATH or place the file in one of these locations:\n"
        + "\n".join(searched)
    )


DATA_PATH = resolve_appliances_data_path(__file__)

RANDOM_STATE = 42
MAX_REVIEWS = 20000
MIN_USER_REVIEWS = 2
MIN_ITEM_REVIEWS = 2
ALPHA = 0.5
TOP_N = 5
LIKE_THRESHOLD = 4.0

analyzer = SentimentIntensityAnalyzer()


def label_sentiment_score(text: str) -> float:
    """Return VADER compound sentiment score in the range [-1, 1]."""
    return analyzer.polarity_scores(str(text))["compound"]


def load_and_prepare_data() -> pd.DataFrame:
    print("Loading dataset...")
    print(f"Using dataset: {DATA_PATH}")

    use_cols = ["reviewerID", "asin", "overall", "summary", "reviewText"]
    df = pd.read_json(DATA_PATH, lines=True)
    df = df[use_cols].copy()

    df = df[
        df["reviewText"].notnull()
        & df["reviewerID"].notnull()
        & df["asin"].notnull()
    ].copy()

    df["summary"] = df["summary"].fillna("")
    df["text"] = (
        df["summary"].astype(str) + " " + df["reviewText"].astype(str)
    ).str.strip()

    print(f"Full dataset rows after cleaning: {len(df)}")

    if len(df) > MAX_REVIEWS:
        df = df.sample(n=MAX_REVIEWS, random_state=RANDOM_STATE)
        print(f"Working subset sampled from full data: {len(df)} reviews")
    else:
        print(f"Using all available reviews: {len(df)}")

    user_counts = df["reviewerID"].value_counts()
    item_counts = df["asin"].value_counts()

    df = df[
        df["reviewerID"].isin(user_counts[user_counts >= MIN_USER_REVIEWS].index)
        & df["asin"].isin(item_counts[item_counts >= MIN_ITEM_REVIEWS].index)
    ].copy()

    print(f"Rows after user/item frequency filter: {len(df)}")
    print(f"Unique users: {df['reviewerID'].nunique()}")
    print(f"Unique items: {df['asin'].nunique()}")

    df["sentiment_score"] = df["text"].apply(label_sentiment_score)
    df["rating_adj"] = (df["overall"] + ALPHA * df["sentiment_score"]).clip(1, 5)

    return df


def build_user_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
    R = df.pivot_table(
        index="reviewerID",
        columns="asin",
        values="rating_adj",
        aggfunc="mean"
    ).fillna(0)

    print(f"User-item matrix shape: {R.shape}")

    if R.shape[0] == 0 or R.shape[1] == 0:
        raise ValueError(
            "User-item matrix is empty after filtering. "
            "Increase MAX_REVIEWS or lower the min frequency thresholds."
        )

    return R


def build_item_similarity_matrix(R: pd.DataFrame) -> np.ndarray:
    return cosine_similarity(R.T)


def get_popular_items(df: pd.DataFrame) -> pd.Series:
    item_stats = df.groupby("asin").agg(
        mean_rating_adj=("rating_adj", "mean"),
        interaction_count=("asin", "count")
    )

    item_stats = item_stats.sort_values(
        by=["mean_rating_adj", "interaction_count"],
        ascending=[False, False]
    )

    return item_stats["mean_rating_adj"]


def recommend(
    user_id: str,
    R: pd.DataFrame,
    S: np.ndarray,
    items: list,
    df: pd.DataFrame,
    top_n: int = TOP_N
):
    popular_items = get_popular_items(df)

    if user_id not in R.index:
        return [(item, float(score)) for item, score in popular_items.head(top_n).items()]

    user_ratings = R.loc[user_id].values
    seen_idx = np.where(user_ratings > 0)[0]

    user_rows = df[df["reviewerID"] == user_id]
    liked_items = user_rows[user_rows["overall"] >= LIKE_THRESHOLD]["asin"].unique().tolist()
    liked_idx = [items.index(item) for item in liked_items if item in items]

    if len(liked_idx) == 0:
        seen_items = {items[i] for i in seen_idx}
        fallback = [
            (item, float(score))
            for item, score in popular_items.items()
            if item not in seen_items
        ]
        return fallback[:top_n]

    weights = user_ratings[liked_idx]
    if weights.sum() == 0:
        scores = S[liked_idx].mean(axis=0)
    else:
        scores = np.average(S[liked_idx], axis=0, weights=weights)

    scores[seen_idx] = -1

    rec_idx = np.argsort(scores)[::-1]
    recommendations = []

    for i in rec_idx:
        if scores[i] <= 0:
            continue
        recommendations.append((items[i], float(scores[i])))
        if len(recommendations) == top_n:
            break

    if not recommendations:
        seen_items = {items[i] for i in seen_idx}
        fallback = [
            (item, float(score))
            for item, score in popular_items.items()
            if item not in seen_items
        ]
        return fallback[:top_n]

    return recommendations


def plot_original_vs_adjusted_ratings(df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    plt.hist(df["overall"], bins=np.arange(0.5, 5.6, 0.5), alpha=0.7, label="Original Rating")
    plt.hist(df["rating_adj"], bins=20, alpha=0.7, label="Adjusted Rating")
    plt.xlabel("Rating Value")
    plt.ylabel("Frequency")
    plt.title("Original vs Adjusted Rating Distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_sentiment_distribution(df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    plt.hist(df["sentiment_score"], bins=30)
    plt.xlabel("Sentiment Score")
    plt.ylabel("Frequency")
    plt.title("Sentiment Score Distribution")
    plt.tight_layout()
    plt.show()


def plot_user_interaction_distribution(R: pd.DataFrame):
    interaction_counts = (R > 0).sum(axis=1)

    plt.figure(figsize=(8, 5))
    plt.hist(interaction_counts, bins=20)
    plt.xlabel("Number of Interacted Items per User")
    plt.ylabel("Number of Users")
    plt.title("User Interaction Distribution")
    plt.tight_layout()
    plt.show()


def print_sparsity(R: pd.DataFrame):
    total_cells = R.shape[0] * R.shape[1]
    non_zero = (R > 0).sum().sum()
    sparsity = 1 - (non_zero / total_cells)

    print(f"\nMatrix sparsity: {sparsity:.4f} ({sparsity * 100:.2f}%)")
    print(f"Non-zero entries: {non_zero}")
    print(f"Total entries: {total_cells}")


def print_top_recommendations(user_id: str, recommendations):
    print(f"\nTop recommendations for user {user_id}:")
    if not recommendations:
        print("No recommendations found.")
        return

    for rank, (item, score) in enumerate(recommendations, start=1):
        print(f"{rank}. Item: {item} | Score: {score:.4f}")


def main():
    df = load_and_prepare_data()
    R = build_user_item_matrix(df)
    S = build_item_similarity_matrix(R)
    items = R.columns.to_list()

    print_sparsity(R)

    interaction_counts = (R > 0).sum(axis=1)
    some_user = interaction_counts.sort_values(ascending=False).index[0]

    recommendations = recommend(some_user, R, S, items, df, top_n=TOP_N)

    print("\nDemo user:", some_user)
    print("Observed items for demo user:", int(interaction_counts.loc[some_user]))
    print_top_recommendations(some_user, recommendations)

    plot_original_vs_adjusted_ratings(df)
    plot_sentiment_distribution(df)
    plot_user_interaction_distribution(R)


if __name__ == "__main__":
    main()
