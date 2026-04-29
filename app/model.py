import os

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .config import MODEL_DIR, MODEL_NAME, MODEL_WEIGHTS


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        pooled_output = outputs.pooler_output
        x = self.drop(pooled_output)
        return self.fc(x)


def load_model_and_tokenizer():
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Tokenizer folder not found: {MODEL_DIR}")
    if not os.path.exists(MODEL_WEIGHTS):
        raise FileNotFoundError(f"Model weights not found: {MODEL_WEIGHTS}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
    model = SentimentClassifier(n_classes=2)
    state = torch.load(MODEL_WEIGHTS, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer
