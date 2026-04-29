# Vietnamese Malicious Comment Detection (PhoBERT)

Streamlit app for Vietnamese malicious comment detection using a PhoBERT classifier. The repository also includes a training notebook for building and exporting the model weights + tokenizer.

## Features
- Single-text prediction with confidence scores
- Batch CSV prediction with download
- Preprocessing with Vietnamese tokenization and optional stopword removal

## Project structure
- app/ - core modules (config, preprocessing, model, prediction)
- streamlit_app.py - Streamlit UI entrypoint
- data/ - dataset and stopwords
- weight/phobert_malicious_detection/ - exported tokenizer and model weights
- Notebook/PhoBert.ipynb - training and evaluation notebook

## Requirements
- Python 3.9+
- See requirements.txt

## Setup
```bash
pip install -r requirements.txt
```

## Run the app
```bash
streamlit run streamlit_app.py
```

## Batch prediction format
Upload a CSV file with at least one column:
- content: Vietnamese text to classify

The app returns extra columns:
- prediction, confidence, prob_clean, prob_malicious

## Model artifacts
The app expects these files to exist:
- weight/phobert_malicious_detection/phobert_best.pth
- weight/phobert_malicious_detection/tokenizer_config.json
- weight/phobert_malicious_detection/vocab.txt
- weight/phobert_malicious_detection/bpe.codes

## Training notebook
The notebook in Notebook/PhoBert.ipynb contains:
- preprocessing
- K-fold training
- evaluation + error analysis
- export of model weights and tokenizer

## Notes
- Stopwords are loaded from data/vietnamese_stopwords.txt
- The classifier predicts binary labels: 0 = Clean, 1 = Malicious
