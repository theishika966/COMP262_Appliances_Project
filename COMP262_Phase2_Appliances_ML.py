# ============================================================
# COMP 262 - Phase 2 
# ============================================================

import os
import re
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
# PATHS
# -------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "Appliances.json")
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# -------------------------
# SETTINGS
# -------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.30
N_SAMPLE = 2000
SENTIMENT_ORDER = ["Negative", "Neutral", "Positive"]

# -------------------------
# HELPERS
# -------------------------
def label_sentiment(r):
    if r >= 4: return "Positive"
    if r == 3: return "Neutral"
    return "Negative"

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def plot_confusion(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=SENTIMENT_ORDER)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=SENTIMENT_ORDER,
                yticklabels=SENTIMENT_ORDER)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()

# -------------------------
# LOAD DATA
# -------------------------
print("Loading dataset...")
df = pd.read_json(DATA_PATH, lines=True)

df = df[["overall","summary","reviewText","reviewerID","asin"]]
df = df[df["reviewText"].notnull()]
df["summary"] = df["summary"].fillna("")
df["text"] = (df["summary"] + " " + df["reviewText"]).astype(str)

# -------------------------
# LABEL
# -------------------------
df["sentiment"] = df["overall"].apply(label_sentiment)

# -------------------------
# SAMPLE 
# -------------------------
df = df.sample(n=N_SAMPLE, random_state=RANDOM_STATE).copy()

print("Dataset size:", len(df))

# ============================================================
# 🔍 DATA EXPLORATION 
# ============================================================

# Sentiment distribution
plt.figure()
sns.countplot(x="sentiment", data=df)
plt.title("Sentiment Distribution")
plt.savefig(os.path.join(PLOT_DIR, "sentiment_distribution.png"))
plt.close()

# Review length
df["length"] = df["text"].apply(lambda x: len(x.split()))
plt.figure()
sns.histplot(df["length"], bins=30)
plt.title("Review Length Distribution")
plt.savefig(os.path.join(PLOT_DIR, "review_length.png"))
plt.close()

# Reviews per user 
reviews_per_user = df.groupby("reviewerID").size()
plt.figure()
sns.histplot(reviews_per_user, bins=30)
plt.title("Reviews per User")
plt.savefig(os.path.join(PLOT_DIR, "reviews_per_user.png"))
plt.close()

# Reviews per product
reviews_per_product = df.groupby("asin").size()
plt.figure()
sns.histplot(reviews_per_product, bins=30)
plt.title("Reviews per Product")
plt.savefig(os.path.join(PLOT_DIR, "reviews_per_product.png"))
plt.close()

# Duplicate analysis 
duplicates = df.duplicated(subset=["text"]).sum()
dup_ratio = duplicates / len(df)

print("\nDuplicate reviews:", duplicates)
print("Duplicate ratio:", dup_ratio)

plt.figure()
plt.pie([len(df)-duplicates, duplicates],
        labels=["Unique", "Duplicate"],
        autopct="%1.1f%%")
plt.title("Duplicate Review Distribution")
plt.savefig(os.path.join(PLOT_DIR, "duplicate_distribution.png"))
plt.close()

# ============================================================
# PREPROCESSING
# ============================================================
df["text_clean"] = df["text"].apply(clean_text)

# ============================================================
# SPLIT (70/30 stratified)
# ============================================================
train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    stratify=df["sentiment"],
    random_state=RANDOM_STATE
)

X_train = train_df["text_clean"]
y_train = train_df["sentiment"]

X_test = test_df["text_clean"]
y_test = test_df["sentiment"]

# ============================================================
# MODEL 1: LOGISTIC REGRESSION
# ============================================================
lr_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))
])

lr_grid = GridSearchCV(lr_pipe, {"clf__C":[0.5,1,2]}, scoring="f1_macro", cv=3)
lr_grid.fit(X_train, y_train)

print("\nLR Best CV Score:", lr_grid.best_score_)
lr_pred = lr_grid.predict(X_test)

# ============================================================
# MODEL 2: SVM
# ============================================================
svm_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
    ("clf", LinearSVC(class_weight="balanced"))
])

svm_grid = GridSearchCV(svm_pipe, {"clf__C":[0.5,1,2]}, scoring="f1_macro", cv=3)
svm_grid.fit(X_train, y_train)

print("SVM Best CV Score:", svm_grid.best_score_)
svm_pred = svm_grid.predict(X_test)

# ============================================================
# LEXICON MODELS 
# ============================================================
vader = SentimentIntensityAnalyzer()

def vader_pred(t):
    s = vader.polarity_scores(t)["compound"]
    if s >= 0.05: return "Positive"
    if s <= -0.05: return "Negative"
    return "Neutral"

def tb_pred(t):
    p = TextBlob(t).sentiment.polarity
    if p > 0: return "Positive"
    if p < 0: return "Negative"
    return "Neutral"

raw_test = test_df["text"]

v_pred = raw_test.apply(vader_pred)
t_pred = raw_test.apply(tb_pred)

# ============================================================
# EVALUATION
# ============================================================
def evaluate(name, y, pred):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, average="weighted"),
        "Recall": recall_score(y, pred, average="weighted"),
        "F1": f1_score(y, pred, average="macro")
    }

results = pd.DataFrame([
    evaluate("LogReg", y_test, lr_pred),
    evaluate("SVM", y_test, svm_pred),
    evaluate("VADER", y_test, v_pred),
    evaluate("TextBlob", y_test, t_pred)
])

print("\n===== FINAL RESULTS =====")
print(results)

results.to_csv(os.path.join(PLOT_DIR, "phase2_results.csv"), index=False)

# ============================================================
# CONFUSION MATRICES
# ============================================================
plot_confusion(y_test, lr_pred, "LogReg Confusion", "lr_confusion.png")
plot_confusion(y_test, svm_pred, "SVM Confusion", "svm_confusion.png")
plot_confusion(y_test, v_pred, "VADER Confusion", "vader_confusion.png")
plot_confusion(y_test, t_pred, "TextBlob Confusion", "textblob_confusion.png")
