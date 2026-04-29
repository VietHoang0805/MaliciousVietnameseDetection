import os

MODEL_NAME = "vinai/phobert-base"
MODEL_DIR = os.path.join("weight", "phobert_malicious_detection")
MODEL_WEIGHTS = os.path.join(MODEL_DIR, "phobert_best.pth")
STOPWORDS_PATH = os.path.join("data", "vietnamese_stopwords.txt")

MAX_LEN = 128
LABEL_MAP = {0: "Clean", 1: "Malicious"}
