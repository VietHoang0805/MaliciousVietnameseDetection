import os


def _resolve_model_dir(candidates: list[str]) -> str:
	for path in candidates:
		if os.path.exists(path):
			return path
	return candidates[0]


MODEL_CATALOG = {
	"phobert": {
		"label": "PhoBERT",
		"arch": "phobert",
		"model_name": "vinai/phobert-base",
		"dir_candidates": [
			os.path.join("weight", "phobert_malicious_detection"),
			os.path.join("weight", "phobert", "phobert_malicious_detection"),
		],
		"weights_name": "phobert_best.pth",
		"use_fast": False,
	},
	"phobert_bilstm": {
		"label": "PhoBERT + BiLSTM",
		"arch": "phobert_bilstm",
		"model_name": "vinai/phobert-base",
		"dir_candidates": [
			os.path.join("weight", "phobert_bilstm", "phobert_bilstm"),
		],
		"weights_name": "phobert_bilstm_best.pth",
		"use_fast": False,
	},
	"xlmr_base": {
		"label": "XLM-RoBERTa Base",
		"arch": "xlmr_base",
		"model_name": "xlm-roberta-base",
		"dir_candidates": [
			os.path.join("weight", "xlmr_malicious_detection"),
			os.path.join("weight", "xlmr", "xlmr_malicious_detection"),
		],
		"weights_name": "xlmr_best.pth",
		"use_fast": True,
	},
}

DEFAULT_MODEL_KEY = "phobert"


def get_model_config(model_key: str) -> dict:
	if model_key not in MODEL_CATALOG:
		raise KeyError(f"Unknown model key: {model_key}")
	entry = MODEL_CATALOG[model_key].copy()
	entry["model_dir"] = _resolve_model_dir(entry["dir_candidates"])
	entry["model_weights"] = os.path.join(entry["model_dir"], entry["weights_name"])
	return entry


_default_config = get_model_config(DEFAULT_MODEL_KEY)
MODEL_NAME = _default_config["model_name"]
MODEL_DIR = _default_config["model_dir"]
MODEL_WEIGHTS = _default_config["model_weights"]

STOPWORDS_PATH = os.path.join("data", "vietnamese_stopwords.txt")

MAX_LEN = 128
LABEL_MAP = {0: "Clean", 1: "Malicious"}
