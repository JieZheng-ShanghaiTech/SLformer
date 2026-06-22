"""Simple semantic encoder helpers for MedCPT / HF encoders.

This module defines a minimal wrapper to load a HuggingFace-compatible
sentence encoder and encode batches of texts to dense vectors.

Note: this file is intentionally simple and assumes the environment has
`transformers` and `torch` available. Do not execute here.
"""

from typing import List

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np


def load_encoder(model_path: str, device: str = "cpu"):
    """Load tokenizer and model from a local or HuggingFace path.

    Returns (tokenizer, model, device).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def encode_texts(tokenizer, model, device, texts: List[str], batch_size: int = 32):
    """Encode a list of texts and return a numpy array of shape (N, q).

    Uses mean pooling over the last hidden state.
    """
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            last = out.last_hidden_state  # (B, L, H)
            mask = attention_mask.unsqueeze(-1).float()
            summed = (last * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-9)
            emb = (summed / denom).cpu().numpy()
        all_emb.append(emb)
    return np.vstack(all_emb)
