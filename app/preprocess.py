import os
import re

from underthesea import word_tokenize


def load_stopwords(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def preprocess_text(text: str, stopwords: set, remove_stopwords: bool = True) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^\w\s\u00C0-\u024F\u1E00-\u1EFF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = word_tokenize(text, format="text")
    if remove_stopwords and stopwords:
        words = [w for w in text.split() if w not in stopwords]
        text = " ".join(words)
    return text
