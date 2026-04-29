import numpy as np
import torch

from .config import LABEL_MAP, MAX_LEN
from .preprocess import preprocess_text


def predict_text(text: str, model, tokenizer, stopwords: set):
    processed = preprocess_text(text, stopwords=stopwords)
    encoding = tokenizer(
        processed,
        max_length=MAX_LEN,
        truncation=True,
        add_special_tokens=True,
        padding="max_length",
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        confidence = float(np.max(probs))

    return {
        "label": LABEL_MAP[pred],
        "confidence": confidence,
        "prob_clean": float(probs[0]),
        "prob_malicious": float(probs[1]),
        "processed": processed,
    }
