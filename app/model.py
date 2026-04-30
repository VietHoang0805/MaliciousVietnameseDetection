import os

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .config import DEFAULT_MODEL_KEY, get_model_config


class PhoBertClassifier(nn.Module):
    def __init__(self, model_name: str, n_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=dropout)
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


class PhoBertBiLSTMClassifier(nn.Module):
    def __init__(self, model_name: str, n_classes: int = 2, hidden_size: int = 256, dropout: float = 0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.bilstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        sequence_output = outputs.last_hidden_state
        _, (h_n, _) = self.bilstm(sequence_output)
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        pooled = torch.cat([h_forward, h_backward], dim=1)
        x = self.dropout(pooled)
        return self.fc(x)


class XLMRClassifier(nn.Module):
    def __init__(self, model_name: str, n_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(self.encoder.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        cls_token = outputs.last_hidden_state[:, 0, :]
        x = self.drop(cls_token)
        return self.fc(x)


def _build_model(model_config: dict) -> nn.Module:
    arch = model_config["arch"]
    model_name = model_config["model_name"]
    if arch == "phobert":
        return PhoBertClassifier(model_name, n_classes=2)
    if arch == "phobert_bilstm":
        return PhoBertBiLSTMClassifier(model_name, n_classes=2)
    if arch == "xlmr_base":
        return XLMRClassifier(model_name, n_classes=2)
    raise ValueError(f"Unsupported model architecture: {arch}")


def load_model_and_tokenizer(model_key: str = DEFAULT_MODEL_KEY):
    model_config = get_model_config(model_key)
    model_dir = model_config["model_dir"]
    model_weights = model_config["model_weights"]

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Tokenizer folder not found: {model_dir}")
    if not os.path.exists(model_weights):
        raise FileNotFoundError(f"Model weights not found: {model_weights}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=model_config["use_fast"])
    model = _build_model(model_config)
    state = torch.load(model_weights, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer
