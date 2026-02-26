# ============================================================
# COMP 262 - Project Phase 1
# Amazon Appliances Dataset (5-core)
# Lexicon-Based Sentiment Analysis
# ============================================================

# =========================
# SECTION 1: Import Libraries
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

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

# =========================
# SECTION 2: Load Dataset
# =========================

file_path = "data/Appliances_5.json.gz"
df = pd.read_json(file_path, lines=True)

print("Dataset Shape:", df.shape)
print("\nColumns:\n", df.columns)

# =========================
# SECTION 3: Dataset Exploration
# =========================

print("\n===== BASIC STATISTICS =====")
print("Total Reviews:", len(df))
print("Unique Users:", df['reviewerID'].nunique())
print("Unique Products:", df['asin'].nunique())
print("Average Rating:", df['overall'].mean())

print("\n===== Rating Distribution =====")
print(df['overall'].value_counts())

plt.figure()
sns.countplot(x='overall', data=df)
plt.title("Rating Distribution")
plt.show()

# Reviews per product
reviews_per_product = df.groupby('asin').size()
plt.figure()
sns.histplot(reviews_per_product, bins=20)
plt.title("Distribution of Reviews per Product")
plt.show()

# Reviews per user
reviews_per_user = df.groupby('reviewerID').size()
plt.figure()
sns.histplot(reviews_per_user, bins=20)
plt.title("Distribution of Reviews per User")
plt.show()

# Review length analysis
df['review_length'] = df['reviewText'].astype(str).apply(lambda x: len(x.split()))

print("\nMin Review Length:", df['review_length'].min())
print("Max Review Length:", df['review_length'].max())
print("Average Review Length:", df['review_length'].mean())

plt.figure()
sns.histplot(df['review_length'], bins=30)
plt.title("Review Length Distribution")
plt.show()

# Missing values
print("\n===== Missing Values =====")
print(df.isnull().sum())

# Remove empty reviews
df = df[df['reviewText'].notnull()]
df = df[df['reviewText'].str.strip() != ""]

# =========================
# SECTION 4: Label Sentiment
# =========================

def label_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

df['sentiment'] = df['overall'].apply(label_sentiment)

print("\n===== Sentiment Distribution =====")
print(df['sentiment'].value_counts())

# =========================
# SECTION 5: Random Sampling (1000 Reviews)
# =========================

sample_df = df.sample(n=1000, random_state=42)

print("\nSample Sentiment Distribution:")
print(sample_df['sentiment'].value_counts())

# =========================
# SECTION 6: VADER Model
# =========================

analyzer = SentimentIntensityAnalyzer()

def vader_predict(text):
    score = analyzer.polarity_scores(str(text))['compound']
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

sample_df['vader_pred'] = sample_df['reviewText'].apply(vader_predict)

vader_accuracy = accuracy_score(sample_df['sentiment'], sample_df['vader_pred'])
vader_precision = precision_score(sample_df['sentiment'], sample_df['vader_pred'], average='weighted')
vader_recall = recall_score(sample_df['sentiment'], sample_df['vader_pred'], average='weighted')
vader_f1 = f1_score(sample_df['sentiment'], sample_df['vader_pred'], average='weighted')

print("\n===== VADER RESULTS =====")
print("Accuracy:", vader_accuracy)
print("Precision:", vader_precision)
print("Recall:", vader_recall)
print("F1 Score:", vader_f1)
print("\nClassification Report:\n")
print(classification_report(sample_df['sentiment'], sample_df['vader_pred']))

cm_vader = confusion_matrix(sample_df['sentiment'], sample_df['vader_pred'])

plt.figure()
sns.heatmap(cm_vader, annot=True, fmt='d',
            xticklabels=["Negative","Neutral","Positive"],
            yticklabels=["Negative","Neutral","Positive"])
plt.title("VADER Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================
# SECTION 7: Text Cleaning for TextBlob
# =========================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

sample_df['clean_review'] = sample_df['reviewText'].apply(clean_text)

# =========================
# SECTION 8: TextBlob Model
# =========================

def textblob_predict(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

sample_df['textblob_pred'] = sample_df['clean_review'].apply(textblob_predict)

tb_accuracy = accuracy_score(sample_df['sentiment'], sample_df['textblob_pred'])
tb_precision = precision_score(sample_df['sentiment'], sample_df['textblob_pred'], average='weighted')
tb_recall = recall_score(sample_df['sentiment'], sample_df['textblob_pred'], average='weighted')
tb_f1 = f1_score(sample_df['sentiment'], sample_df['textblob_pred'], average='weighted')

print("\n===== TEXTBLOB RESULTS =====")
print("Accuracy:", tb_accuracy)
print("Precision:", tb_precision)
print("Recall:", tb_recall)
print("F1 Score:", tb_f1)
print("\nClassification Report:\n")
print(classification_report(sample_df['sentiment'], sample_df['textblob_pred']))

cm_tb = confusion_matrix(sample_df['sentiment'], sample_df['textblob_pred'])

plt.figure()
sns.heatmap(cm_tb, annot=True, fmt='d',
            xticklabels=["Negative","Neutral","Positive"],
            yticklabels=["Negative","Neutral","Positive"])
plt.title("TextBlob Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================
# SECTION 9: Model Comparison
# =========================

comparison = pd.DataFrame({
    "Model": ["VADER", "TextBlob"],
    "Accuracy": [vader_accuracy, tb_accuracy],
    "Precision": [vader_precision, tb_precision],
    "Recall": [vader_recall, tb_recall],
    "F1 Score": [vader_f1, tb_f1]
})

print("\n===== MODEL COMPARISON =====")
print(comparison)