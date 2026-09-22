"""Text cleaning and train/validation splitting.

Two cleaning levels are provided:
    * ``clean_text``    – light normalisation used for the TF-IDF baseline and the
      from-scratch neural models (lowercasing, URL/user/number masking, whitespace).
    * transformer inputs use the model's own tokenizer on *lightly* cleaned text
      (we keep casing information loss minimal — DistilBERT/MiniLM are uncased, so
      lowercasing is harmless, but we avoid aggressive stripping that removes signal).

The validation split is *iterative-stratified by hand* on the rarest label to make
sure ``threat``/``identity_hate`` positives appear in both folds.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from data import LABELS

_URL = re.compile(r"https?://\S+|www\.\S+")
_USER = re.compile(r"@\w+")
_IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_NUM = re.compile(r"\b\d+\b")
_NONWORD = re.compile(r"[^a-z0-9\s'!?.]")
_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Light, reversible-enough normalisation for classical / neural models."""
    text = text.lower()
    text = _URL.sub(" url ", text)
    text = _IP.sub(" ipaddr ", text)
    text = _USER.sub(" user ", text)
    text = _NUM.sub(" num ", text)
    text = _NONWORD.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text


def add_clean_column(df: pd.DataFrame, col: str = "comment_text") -> pd.DataFrame:
    df = df.copy()
    df["clean_text"] = df[col].map(clean_text)
    return df


def train_val_split(
    df: pd.DataFrame,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split that guarantees rare-label coverage in both folds.

    Rows are ordered by their rarest active label so that stratified sampling
    keeps ``threat``/``identity_hate`` positives on both sides.
    """
    rng = np.random.RandomState(seed)
    # Assign each row a stratification key = its rarest positive label (or "none").
    label_freq = df[LABELS].sum().to_dict()
    order = sorted(LABELS, key=lambda c: label_freq[c])  # rarest first

    def strat_key(row) -> str:
        for lab in order:
            if row[lab] == 1:
                return lab
        return "none"

    keys = df.apply(strat_key, axis=1)
    val_idx = []
    for key, group in df.groupby(keys):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac)))
        val_idx.extend(idx[:n_val].tolist())
    val_mask = df.index.isin(val_idx)
    return df[~val_mask].reset_index(drop=True), df[val_mask].reset_index(drop=True)
