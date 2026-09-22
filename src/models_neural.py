"""From-scratch neural models: TextCNN (1D-CNN) and BiLSTM.

Both are compact, CPU-trainable multi-label classifiers over a word-index sequence
representation. They form the middle rung of the model ladder (classical -> classic
deep learning -> transformer).

Design choices for CPU / low-RAM:
    * vocabulary capped (default 20k) built from the training subsample;
    * short max sequence length (default 200) with padding/truncation;
    * small hidden sizes; embeddings trainable from scratch by default, with an
      optional ``pretrained`` matrix (e.g. GloVe) for transfer learning.

Multi-label training uses BCEWithLogitsLoss (independent sigmoid per label).
"""
from __future__ import annotations

import re
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data import LABELS

PAD, UNK = "<pad>", "<unk>"
_TOK = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return _TOK.findall(text.lower())


class Vocab:
    def __init__(self, texts, max_size: int = 20_000, min_freq: int = 2):
        counter: Counter = Counter()
        for t in texts:
            counter.update(tokenize(t))
        self.itos = [PAD, UNK]
        for tok, freq in counter.most_common():
            if freq < min_freq or len(self.itos) >= max_size:
                break
            self.itos.append(tok)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self.stoi.get(tok, 1) for tok in tokenize(text)][:max_len]
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        return ids


class TextDataset(Dataset):
    def __init__(self, texts, labels: np.ndarray, vocab: Vocab, max_len: int = 200):
        self.x = [vocab.encode(t, max_len) for t in texts]
        self.y = labels.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.tensor(self.x[i], dtype=torch.long), torch.tensor(self.y[i])


class TextCNN(nn.Module):
    """Kim (2014) style 1D-CNN with multiple kernel widths."""

    def __init__(self, vocab_size, emb_dim=100, n_labels=6, kernels=(2, 3, 4),
                 n_filters=100, dropout=0.5, pretrained=None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if pretrained is not None:
            self.emb.weight.data.copy_(torch.tensor(pretrained))
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, n_filters, k, padding=k // 2) for k in kernels]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernels), n_labels)

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)          # (B, emb, L)
        feats = [torch.relu(c(e)).max(dim=2).values for c in self.convs]
        h = self.dropout(torch.cat(feats, dim=1))
        return self.fc(h)


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=100, n_labels=6, hidden=64,
                 dropout=0.5, pretrained=None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if pretrained is not None:
            self.emb.weight.data.copy_(torch.tensor(pretrained))
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, n_labels)

    def forward(self, x):
        e = self.emb(x)
        out, _ = self.lstm(e)
        pooled = out.max(dim=1).values                # global max-pool over time
        return self.fc(self.dropout(pooled))


def train_model(model, train_ds, val_ds, epochs=4, batch_size=64, lr=1e-3,
                seed=42, log=print):
    """Train a multi-label model; return per-epoch history and best val probs."""
    torch.manual_seed(seed)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=256)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for xb, yb in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        model.eval()
        vloss, probs = 0.0, []
        with torch.no_grad():
            for xb, yb in val_dl:
                logits = model(xb)
                vloss += loss_fn(logits, yb).item() * len(xb)
                probs.append(torch.sigmoid(logits).numpy())
        history.append(
            {"epoch": ep,
             "train_loss": tot / len(train_ds),
             "val_loss": vloss / len(val_ds),
             "sec": time.time() - t0}
        )
        log(f"  epoch {ep}: train {history[-1]['train_loss']:.4f} "
            f"val {history[-1]['val_loss']:.4f} ({history[-1]['sec']:.0f}s)")
    val_probs = np.concatenate(probs)
    return history, val_probs


@torch.no_grad()
def predict_proba(model, texts, vocab, max_len=200, batch_size=256) -> np.ndarray:
    model.eval()
    ids = [vocab.encode(t, max_len) for t in texts]
    out = []
    for i in range(0, len(ids), batch_size):
        xb = torch.tensor(ids[i:i + batch_size], dtype=torch.long)
        out.append(torch.sigmoid(model(xb)).numpy())
    return np.concatenate(out)
