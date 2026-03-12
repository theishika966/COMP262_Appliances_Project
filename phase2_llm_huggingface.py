import os
import re
import warnings
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

warnings.filterwarnings("ignore", message=".*cache-system uses symlinks.*")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "Appliances.json")

MODEL_NAME = "google/flan-t5-base"
RANDOM_STATE = 42
MIN_WORDS_LONG_REVIEW = 100
TARGET_SUMMARY_WORDS = 50
MAX_INPUT_CHARS = 2200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def word_count(text: str) -> int:
    return len(normalize_space(text).split())


def truncate_for_model(text: str, max_chars: int = MAX_INPUT_CHARS) -> str:
    text = normalize_space(text)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_space = cut.rfind(" ")
    if last_space > 0:
        cut = cut[:last_space]
    return cut.strip()


def cap_to_approx_words(text: str, max_words: int = TARGET_SUMMARY_WORDS) -> str:
    words = normalize_space(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip(" ,;:-") + "..."


def is_question_like(text: str) -> bool:
    t = normalize_space(text)
    if "?" in t:
        return True
    t2 = re.sub(r"[^a-zA-Z\s]", " ", t.lower())
    t2 = normalize_space(t2)
    starters = (
        "how", "what", "why", "when", "where", "which",
        "does", "do", "did", "is", "are", "can", "could",
        "should", "would", "will", "has", "have"
    )
    return any(t2.startswith(s + " ") for s in starters)


class LocalSeq2SeqGenerator:
    def __init__(self, model_name: str):
        print(f"Loading local Hugging Face model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(DEVICE)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 96, min_new_tokens: int = 20) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=False,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.0
            )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return normalize_space(text)


print("Loading dataset...")
df = pd.read_json(DATA_PATH, lines=True)
df = df[df["reviewText"].notnull()].copy()
df = df[df["reviewText"].astype(str).str.strip() != ""].copy()
df["summary"] = df["summary"].fillna("")
df["text"] = (df["summary"].astype(str) + " " + df["reviewText"].astype(str)).map(normalize_space)

long_reviews = df[df["text"].apply(word_count) > MIN_WORDS_LONG_REVIEW].copy()
if long_reviews.empty:
    raise ValueError("No reviews longer than 100 words were found in the dataset.")

sample10 = long_reviews.sample(n=min(10, len(long_reviews)), random_state=RANDOM_STATE).reset_index(drop=True)
print(f"Found {len(long_reviews)} reviews longer than {MIN_WORDS_LONG_REVIEW} words.")
print(f"Selected {len(sample10)} long reviews for summarization.\n")

llm = LocalSeq2SeqGenerator(MODEL_NAME)


def summarize_to_50_words(text: str) -> str:
    review = truncate_for_model(text)
    prompt = (
        "Summarize the following appliance review in about 50 words. "
        "Keep the main product experience, major positives or negatives, and overall judgment.\n\n"
        f"Review: {review}\n\n"
        "Summary:"
    )
    output = llm.generate(prompt, max_new_tokens=90, min_new_tokens=35)
    return cap_to_approx_words(output, TARGET_SUMMARY_WORDS)


print("Generating 10 summaries...\n")
summaries = []
for _, row in sample10.iterrows():
    summaries.append(summarize_to_50_words(row["text"]))

print("=== FIRST TWO SUMMARIES (for report) ===")
for i in range(min(2, len(summaries))):
    print(f"\nReview #{i + 1} (original words: {word_count(sample10.iloc[i]['text'])})")
    print("Original Review:")
    print(sample10.iloc[i]["text"])
    print("\nSummary (~50 words):")
    print(summaries[i])

q_candidates = df[df["text"].apply(is_question_like)].copy()
if q_candidates.empty:
    print("\nNo question-like review found.")
else:
    q_text = q_candidates.sample(n=1, random_state=RANDOM_STATE).iloc[0]["text"]
    q_short = truncate_for_model(q_text, max_chars=1500)
    response_prompt = (
        "You are a customer service representative for an appliance seller. "
        "Write a polite and helpful response to the customer review below. "
        "Acknowledge the concern, answer what you can, and suggest a practical next step if needed. "
        "Keep the reply professional and concise.\n\n"
        f"Customer review: {q_short}\n\n"
        "Response:"
    )
    response = llm.generate(response_prompt, max_new_tokens=120, min_new_tokens=40)

    print("\n=== QUESTION-LIKE REVIEW (for report) ===")
    print(q_text)
    print("\n=== GENERATED CUSTOMER SERVICE RESPONSE (for report) ===")
    print(response)
