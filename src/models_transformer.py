"""Transformer fine-tuning for multi-label toxicity (model-agnostic).

Defaults to MiniLM-L6-H384 (23M params) — chosen over DistilBERT after a CPU timing
probe (MiniLM ~92 min/epoch vs DistilBERT ~242 min/epoch at 36k rows / seq len 128).
Any HF encoder id works via ``model_name``.

A plain PyTorch loop is used instead of ``Trainer`` to keep memory overhead low and
the control flow transparent. Checkpoints are saved so an overnight run can be resumed
and its predictions reused without retraining.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import LABELS

DEFAULT_MODEL = "nreimers/MiniLM-L6-H384-uncased"
CKPT_DIR = Path(__file__).resolve().parent.parent / "models"


class EncodedDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.enc = tokenizer(
            list(texts), truncation=True, padding="max_length",
            max_length=max_len, return_tensors="pt",
        )
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            {"input_ids": self.enc["input_ids"][i],
             "attention_mask": self.enc["attention_mask"][i]},
            self.y[i],
        )


def _predict(model, dl) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for batch, _ in dl:
            logits = model(**batch).logits
            out.append(torch.sigmoid(logits).numpy())
    return np.concatenate(out)


def fine_tune(
    train_texts, train_labels, val_texts, val_labels,
    model_name: str = DEFAULT_MODEL,
    max_len: int = 128, epochs: int = 3, batch_size: int = 16, lr: float = 3e-5,
    seed: int = 42, threads: int = 8, save_as: str | None = "transformer",
    log=print,
):
    """Fine-tune an encoder and return (model, tokenizer, history, val_probs).

    Trains on the subsample; evaluate separately on the full test set via
    ``predict_proba``. Set ``epochs`` to a smaller value for a quick smoke test.
    """
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS), problem_type="multi_label_classification",
    )
    tr_ds = EncodedDataset(train_texts, train_labels, tok, max_len)
    val_ds = EncodedDataset(val_texts, val_labels, tok, max_len)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    history = []
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for batch, yb in tr_dl:
            opt.zero_grad()
            loss = loss_fn(model(**batch).logits, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb)
        val_probs = _predict(model, val_dl)
        vloss = loss_fn(
            torch.logit(torch.tensor(val_probs).clamp(1e-6, 1 - 1e-6)),
            torch.tensor(val_ds.y),
        ).item()
        history.append({"epoch": ep, "train_loss": tot / len(tr_ds),
                        "val_loss": vloss, "sec": time.time() - t0})
        log(f"  epoch {ep}: train {history[-1]['train_loss']:.4f} "
            f"val {vloss:.4f} ({history[-1]['sec']:.0f}s)")

    if save_as:
        CKPT_DIR.mkdir(exist_ok=True)
        model.save_pretrained(CKPT_DIR / save_as)
        tok.save_pretrained(CKPT_DIR / save_as)
        log(f"  saved checkpoint -> models/{save_as}")
    return model, tok, history, val_probs


@torch.no_grad()
def predict_proba(model, tokenizer, texts, max_len=128, batch_size=64,
                  threads=8) -> np.ndarray:
    """Predict on arbitrary texts (e.g. the full 63,978-row test set)."""
    torch.set_num_threads(threads)
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            list(texts[i:i + batch_size]), truncation=True, padding=True,
            max_length=max_len, return_tensors="pt",
        )
        out.append(torch.sigmoid(model(**enc).logits).numpy())
    return np.concatenate(out)
