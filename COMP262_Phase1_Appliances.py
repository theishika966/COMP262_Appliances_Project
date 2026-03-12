# ============================================================
# COMP 262 - Project Phase 1
# Amazon Appliances Dataset (5-core)
# Lexicon-Based Sentiment Analysis (VADER + TextBlob)
# ============================================================

import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

sns.set_style("whitegrid")

# -------------------------
# USER SETTINGS
# -------------------------
# Option A (simple): put your dataset path here
DEFAULT_FILE_PATH = "Appliances_5.json"
# If you want to run with an argument instead:
# python script.py "C:\path\to\Appliances_5.json"
def get_dataset_path() -> str:
    if len(sys.argv) >= 2:
        return sys.argv[1]
    return DEFAULT_FILE_PATH


# -------------------------
# Helpers
# -------------------------
SENTIMENT_ORDER = ["Negative", "Neutral", "Positive"]

def require_file(path: str) -> None:
    if not os.path.isfile(path):
        print("\nERROR: Dataset file not found.")
        print("Tried path:", path)
        print("\nFix options:")
        print("1) Edit DEFAULT_FILE_PATH in the script to your real location")
        print(r'2) Or run: python script.py "C:\full\path\to\Appliances_5.json"')
        raise FileNotFoundError(f"File does not exist: {path}")

def safe_read_json_lines(path: str) -> pd.DataFrame:
    """
    Reads line-delimited JSON (one JSON per line) used by Amazon review datasets.
    Supports .json and .json.gz
    """
    # pandas can read gz directly if extension is .gz
    return pd.read_json(path, lines=True)

def label_sentiment(rating: float) -> str:
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def plot_and_save(fig_name: str, show: bool = True) -> None:
    """
    Saves current figure then shows (optional).
    """
    plt.tight_layout()
    plt.savefig(fig_name, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

def make_confusion_heatmap(y_true, y_pred, title: str, fig_name: str, show: bool = True):
    cm = confusion_matrix(y_true, y_pred, labels=SENTIMENT_ORDER)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=SENTIMENT_ORDER,
        yticklabels=SENTIMENT_ORDER
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plot_and_save(fig_name, show=show)


# ============================================================
# SECTION 2: Load Dataset
# ============================================================
file_path = get_dataset_path()
require_file(file_path)

df = safe_read_json_lines(file_path)

print("Dataset Shape:", df.shape)
print("\nColumns:\n", df.columns)

# Basic required columns check
required_cols = ["reviewText", "overall", "reviewerID", "asin"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing required columns: {missing}")


# ============================================================
# SECTION 3: Dataset Exploration
# ============================================================
print("\n===== BASIC STATISTICS =====")
print("Total Reviews:", len(df))
print("Unique Users:", df["reviewerID"].nunique())
print("Unique Products:", df["asin"].nunique())
print("Average Rating:", df["overall"].mean())

print("\n===== Rating Distribution =====")
print(df["overall"].value_counts().sort_index())

plt.figure(figsize=(7, 4))
sns.countplot(x="overall", data=df, order=sorted(df["overall"].unique()))
plt.title("Rating Distribution")
plot_and_save("rating_distribution.png", show=True)

# Reviews per product
reviews_per_product = df.groupby("asin").size()
plt.figure(figsize=(7, 4))
sns.histplot(reviews_per_product, bins=20)
plt.title("Distribution of Reviews per Product")
plt.xlabel("Reviews per Product")
plot_and_save("reviews_per_product.png", show=True)

# Reviews per user
reviews_per_user = df.groupby("reviewerID").size()
plt.figure(figsize=(7, 4))
sns.histplot(reviews_per_user, bins=20)
plt.title("Distribution of Reviews per User")
plt.xlabel("Reviews per User")
plot_and_save("reviews_per_user.png", show=True)

# Review length analysis
df["review_length"] = df["reviewText"].astype(str).apply(lambda x: len(x.split()))
print("\nMin Review Length:", df["review_length"].min())
print("Max Review Length:", df["review_length"].max())
print("Average Review Length:", df["review_length"].mean())

plt.figure(figsize=(7, 4))
sns.histplot(df["review_length"], bins=30)
plt.title("Review Length Distribution")
plt.xlabel("Words per Review")
plot_and_save("review_length_distribution.png", show=True)

# Missing values
print("\n===== Missing Values =====")
print(df.isnull().sum())

# Remove only empty reviews (DO NOT remove duplicates in 5-core)
df = df[df["reviewText"].notnull()].copy()
df = df[df["reviewText"].astype(str).str.strip() != ""].copy()


# ============================================================
# SECTION 4: Label Sentiment (ground truth from rating)
# ============================================================
df["sentiment"] = df["overall"].apply(label_sentiment)

print("\n===== Sentiment Distribution =====")
print(df["sentiment"].value_counts())

# ============================================================
# SECTION 5: Random Sampling (1000 Reviews)
# ============================================================
n_sample = min(1000, len(df))
sample_df = df.sample(n=n_sample, random_state=42).copy()

print("\nSample Size:", n_sample)
print("Sample Sentiment Distribution:")
print(sample_df["sentiment"].value_counts())

# ============================================================
# SECTION 6: VADER Model
# ============================================================
analyzer = SentimentIntensityAnalyzer()

def vader_predict(text: str) -> str:
    score = analyzer.polarity_scores(str(text))["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

sample_df.loc[:, "vader_pred"] = sample_df["reviewText"].apply(vader_predict)

vader_accuracy = accuracy_score(sample_df["sentiment"], sample_df["vader_pred"])
vader_precision = precision_score(sample_df["sentiment"], sample_df["vader_pred"], average="weighted", zero_division=0)
vader_recall = recall_score(sample_df["sentiment"], sample_df["vader_pred"], average="weighted", zero_division=0)
vader_f1 = f1_score(sample_df["sentiment"], sample_df["vader_pred"], average="weighted", zero_division=0)

print("\n===== VADER RESULTS =====")
print("Accuracy:", vader_accuracy)
print("Precision:", vader_precision)
print("Recall:", vader_recall)
print("F1 Score:", vader_f1)
print("\nClassification Report:\n")
print(classification_report(sample_df["sentiment"], sample_df["vader_pred"], labels=SENTIMENT_ORDER, zero_division=0))

make_confusion_heatmap(
    sample_df["sentiment"],
    sample_df["vader_pred"],
    "VADER Confusion Matrix",
    "vader_confusion_matrix.png",
    show=True
)

# ============================================================
# SECTION 7: Text Cleaning for TextBlob
# ============================================================
sample_df.loc[:, "clean_review"] = sample_df["reviewText"].apply(clean_text)

# ============================================================
# SECTION 8: TextBlob Model
# ============================================================
def textblob_predict(text: str) -> str:
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

sample_df.loc[:, "textblob_pred"] = sample_df["clean_review"].apply(textblob_predict)

tb_accuracy = accuracy_score(sample_df["sentiment"], sample_df["textblob_pred"])
tb_precision = precision_score(sample_df["sentiment"], sample_df["textblob_pred"], average="weighted", zero_division=0)
tb_recall = recall_score(sample_df["sentiment"], sample_df["textblob_pred"], average="weighted", zero_division=0)
tb_f1 = f1_score(sample_df["sentiment"], sample_df["textblob_pred"], average="weighted", zero_division=0)

print("\n===== TEXTBLOB RESULTS =====")
print("Accuracy:", tb_accuracy)
print("Precision:", tb_precision)
print("Recall:", tb_recall)
print("F1 Score:", tb_f1)
print("\nClassification Report:\n")
print(classification_report(sample_df["sentiment"], sample_df["textblob_pred"], labels=SENTIMENT_ORDER, zero_division=0))

make_confusion_heatmap(
    sample_df["sentiment"],
    sample_df["textblob_pred"],
    "TextBlob Confusion Matrix",
    "textblob_confusion_matrix.png",
    show=True
)

# ============================================================
# SECTION 9: Model Comparison Table
# ============================================================
comparison = pd.DataFrame({
    "Model": ["VADER", "TextBlob"],
    "Accuracy": [vader_accuracy, tb_accuracy],
    "Precision": [vader_precision, tb_precision],
    "Recall": [vader_recall, tb_recall],
    "F1 Score": [vader_f1, tb_f1]
})

print("\n===== MODEL COMPARISON =====")
print(comparison.to_string(index=False))

# Save comparison to CSV
comparison.to_csv("model_comparison.csv", index=False)
print("\nSaved: rating_distribution.png, reviews_per_product.png, reviews_per_user.png,")
print("       review_length_distribution.png, vader_confusion_matrix.png,")
print("       textblob_confusion_matrix.png, model_comparison.csv")