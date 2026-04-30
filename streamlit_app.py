import pandas as pd
import streamlit as st

from app.config import DEFAULT_MODEL_KEY, MODEL_CATALOG, STOPWORDS_PATH, get_model_config
from app.model import load_model_and_tokenizer
from app.predict import predict_text
from app.preprocess import load_stopwords


def main():
	st.set_page_config(page_title="Malicious Comment Detection", layout="wide")
	st.title("Malicious Comment Detection")
	st.caption("Binary classification: 0 = Clean, 1 = Malicious")

	with st.sidebar:
		st.subheader("Model")
		model_key = st.selectbox(
			"Choose model",
			options=list(MODEL_CATALOG.keys()),
			index=list(MODEL_CATALOG.keys()).index(DEFAULT_MODEL_KEY),
			format_func=lambda key: MODEL_CATALOG[key]["label"],
		)
		model_config = get_model_config(model_key)
		st.write(f"Tokenizer: {model_config['model_dir']}")
		st.write(f"Weights: {model_config['model_weights']}")
		if "reload_token" not in st.session_state:
			st.session_state["reload_token"] = 0
		if st.button("Reload model"):
			st.session_state["reload_token"] += 1
			st.rerun()
		st.divider()
		st.subheader("Preprocessing")
		remove_stopwords = st.checkbox("Remove stopwords", value=True)

	stopwords = load_stopwords(STOPWORDS_PATH)

	@st.cache_resource
	def cached_load_model(model_key: str, reload_token: int):
		return load_model_and_tokenizer(model_key)

	try:
		model, tokenizer = cached_load_model(model_key, st.session_state["reload_token"])
	except FileNotFoundError as exc:
		st.error(str(exc))
		st.stop()

	st.subheader("Single prediction")
	text_input = st.text_area("Enter Vietnamese text", height=120)
	if st.button("Predict"):
		if not text_input.strip():
			st.warning("Please enter some text.")
		else:
			result = predict_text(text_input, model, tokenizer, stopwords if remove_stopwords else set())
			st.write(f"Label: {result['label']}")
			st.write(f"Confidence: {result['confidence']:.4f}")
			st.write(
				f"Probabilities - Clean: {result['prob_clean']:.4f} | Malicious: {result['prob_malicious']:.4f}"
			)
			with st.expander("Show processed text"):
				st.code(result["processed"])

	st.divider()
	st.subheader("Batch prediction (CSV)")
	st.write("Upload a CSV with a 'content' column.")
	upload = st.file_uploader("Choose a CSV file", type=["csv"])
	if upload is not None:
		df = pd.read_csv(upload)
		if "content" not in df.columns:
			st.error("CSV must contain a 'content' column.")
		else:
			with st.spinner("Running predictions..."):
				results = []
				for text in df["content"].fillna("").astype(str).tolist():
					res = predict_text(text, model, tokenizer, stopwords if remove_stopwords else set())
					results.append(res)
				result_df = df.copy()
				result_df["prediction"] = [r["label"] for r in results]
				result_df["confidence"] = [r["confidence"] for r in results]
				result_df["prob_clean"] = [r["prob_clean"] for r in results]
				result_df["prob_malicious"] = [r["prob_malicious"] for r in results]

			st.dataframe(result_df.head(50))
			csv_bytes = result_df.to_csv(index=False).encode("utf-8")
			st.download_button("Download results CSV", csv_bytes, file_name="predictions.csv")


if __name__ == "__main__":
	main()
