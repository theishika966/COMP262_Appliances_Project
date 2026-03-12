# ============================================================
# COMP 262 - Project Phase 2
# Amazon Appliances Dataset 
# Machine Learning Sentiment Analysis + Lexicon Comparison
# TF-IDF + Logistic Regression + Linear SVM
# ============================================================

import os
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

sns.set_style("whitegrid")

# -------------------------
# Settings
# -------------------------
SENTIMENT_ORDER = ["Negative", "Neutral", "Positive"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FILE_PATH = os.path.join(SCRIPT_DIR, "Appliances.json")

RANDOM_STATE = 42
MIN_REVIEWS_PHASE2 = 2000
TEST_SIZE = 0.30

# -------------------------
# Utilities
# -------------------------
def require_file(path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")


def safe_read_json_lines(path: str) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def label_sentiment(rating: float) -> str:
    if rating >= 4:
        return "Positive"
    if rating == 3:
        return "Neutral"
    return "Negative"


def clean_text_basic(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def plot_confusion(y_true, y_pred, title, out_png):
    cm = confusion_matrix(y_true, y_pred, labels=SENTIMENT_ORDER)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=SENTIMENT_ORDER, yticklabels=SENTIMENT_ORDER)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def eval_model(name, y_true, y_pred) -> dict:
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision_w": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall_w": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1_w": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


# -------------------------
# Lexicon models for comparison
# -------------------------
vader = SentimentIntensityAnalyzer()


def vader_predict(text: str) -> str:
    score = vader.polarity_scores(str(text))["compound"]
    if score >= 0.05:
        return "Positive"
    if score <= -0.05:
        return "Negative"
    return "Neutral"



def textblob_predict(text: str) -> str:
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0:
        return "Positive"
    if polarity < 0:
        return "Negative"
    return "Neutral"


# ============================================================
# 1) Load + Basic Prep
# ============================================================
file_path = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_FILE_PATH
require_file(file_path)

df = safe_read_json_lines(file_path)

# Keep only needed columns and non-empty review text
keep_cols = ["overall", "summary", "reviewText", "reviewerID", "asin"]
df = df[keep_cols].copy()
df = df[df["reviewText"].notnull()].copy()
df = df[df["reviewText"].astype(str).str.strip() != ""].copy()

# Ground truth labels from ratings
df["sentiment"] = df["overall"].apply(label_sentiment)

# Choose text field(s): combine summary + reviewText for richer representation
df["summary"] = df["summary"].fillna("")
df["reviewText"] = df["reviewText"].fillna("")
df["text"] = (df["summary"].astype(str) + " " + df["reviewText"].astype(str)).str.strip()
df["text_clean"] = df["text"].apply(clean_text_basic)
df = df[df["text_clean"].astype(str).str.strip() != ""].copy()

print("Dataset Shape:", df.shape)
print("Sentiment Distribution:\n", df["sentiment"].value_counts())

# ============================================================
# 2) Phase #2 subset selection (>=2000)
# ============================================================
if len(df) < MIN_REVIEWS_PHASE2:
    raise ValueError(f"Need at least {MIN_REVIEWS_PHASE2} reviews for Phase #2. Found {len(df)}.")

phase2_df = df.sample(n=MIN_REVIEWS_PHASE2, random_state=RANDOM_STATE).copy()

print("\nPhase2 subset size:", len(phase2_df))
print("Phase2 subset sentiment distribution:\n", phase2_df["sentiment"].value_counts())
print("\nPhase2 subset exploration:")
print("Unique users:", phase2_df["reviewerID"].nunique())
print("Unique products:", phase2_df["asin"].nunique())
print("Average rating:", round(phase2_df["overall"].mean(), 4))
print("Average review length (words):", round(phase2_df["reviewText"].astype(str).str.split().str.len().mean(), 2))
print("Median review length (words):", int(phase2_df["reviewText"].astype(str).str.split().str.len().median()))
print("Duplicate combined text rows:", int(phase2_df.duplicated(subset=["text"]).sum()))

# ============================================================
# 3) Train/Test split (70/30 stratified)
# ============================================================
train_df, test_df = train_test_split(
    phase2_df[["text", "text_clean", "sentiment"]].copy(),
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=phase2_df["sentiment"]
)

X_train = train_df["text_clean"].values
y_train = train_df["sentiment"].values
X_test = test_df["text_clean"].values
y_test = test_df["sentiment"].values

print("\nTrain size:", len(X_train), "Test size:", len(X_test))

# ============================================================
# 4) TF-IDF + Logistic Regression (with tuning)
# ============================================================
lr_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )),
    ("clf", LogisticRegression(
        max_iter=3000,
        class_weight="balanced"
    ))
])

lr_param_grid = {
    "tfidf__max_features": [20000, 40000],
    "clf__C": [0.5, 1.0, 2.0]
}

lr_search = GridSearchCV(
    lr_pipe,
    lr_param_grid,
    scoring="f1_macro",
    cv=3,
    n_jobs=-1
)

print("\nTraining Logistic Regression (GridSearchCV)...")
lr_search.fit(X_train, y_train)
best_lr = lr_search.best_estimator_
print("Best LR Params:", lr_search.best_params_)

lr_pred = best_lr.predict(X_test)

# ============================================================
# 5) TF-IDF + Linear SVM (with tuning)
# ============================================================
svm_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )),
    ("clf", LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=10000
    ))
])

svm_param_grid = {
    "tfidf__max_features": [20000, 40000],
    "clf__C": [0.5, 1.0, 2.0]
}

svm_search = GridSearchCV(
    svm_pipe,
    svm_param_grid,
    scoring="f1_macro",
    cv=3,
    n_jobs=-1
)

print("\nTraining Linear SVM (GridSearchCV)...")
svm_search.fit(X_train, y_train)
best_svm = svm_search.best_estimator_
print("Best SVM Params:", svm_search.best_params_)

svm_pred = best_svm.predict(X_test)

# ============================================================
# 6) Lexicon models on the SAME test set (apples-to-apples)
# ============================================================
lex_y_test = test_df["sentiment"].values
vader_pred = test_df["text"].apply(vader_predict).values
textblob_pred = test_df["text"].apply(textblob_predict).values

# ============================================================
# 7) Evaluation + Outputs
# ============================================================
results = [
    eval_model("LogReg (TF-IDF)", y_test, lr_pred),
    eval_model("LinearSVM (TF-IDF)", y_test, svm_pred),
    eval_model("VADER (Lexicon)", lex_y_test, vader_pred),
    eval_model("TextBlob (Lexicon)", lex_y_test, textblob_pred),
]
results_df = pd.DataFrame(results)

print("\n===== PHASE #2 RESULTS (Test Set) =====")
print(results_df.sort_values("F1_macro", ascending=False).to_string(index=False))

print("\n--- Logistic Regression Classification Report ---")
print(classification_report(y_test, lr_pred, labels=SENTIMENT_ORDER, zero_division=0))

print("\n--- Linear SVM Classification Report ---")
print(classification_report(y_test, svm_pred, labels=SENTIMENT_ORDER, zero_division=0))

print("\n--- VADER Classification Report ---")
print(classification_report(lex_y_test, vader_pred, labels=SENTIMENT_ORDER, zero_division=0))

print("\n--- TextBlob Classification Report ---")
print(classification_report(lex_y_test, textblob_pred, labels=SENTIMENT_ORDER, zero_division=0))

plot_confusion(y_test, lr_pred, "LogReg (TF-IDF) Confusion Matrix", "phase2_lr_confusion.png")
plot_confusion(y_test, svm_pred, "LinearSVM (TF-IDF) Confusion Matrix", "phase2_svm_confusion.png")
plot_confusion(lex_y_test, vader_pred, "VADER Confusion Matrix", "phase2_vader_confusion.png")
plot_confusion(lex_y_test, textblob_pred, "TextBlob Confusion Matrix", "phase2_textblob_confusion.png")

results_df.to_csv("phase2_model_comparison.csv", index=False)
print("\nSaved: phase2_model_comparison.csv + confusion matrix PNGs")
