"""Data loading, test-set filtering, and stratified subsampling.

The Jigsaw dataset ships as three gzipped CSVs in ./data:
    train.csv.gz       id, comment_text, + 6 binary label columns
    test.csv.gz        id, comment_text
    test_labels.csv.gz id, + 6 label columns (with -1 for unscored rows)

Two facts drive everything downstream:
    * `test_labels` uses -1 for rows Kaggle excluded from scoring. Those rows
      MUST be dropped before computing any test metric (63,978 scored rows remain).
    * Labels are heavily imbalanced (threat ~0.3%), so the transformer/neural
      subsample keeps every positive row and only downsamples the clean majority.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_train(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Load the full training set (~159,571 rows)."""
    df = pd.read_csv(Path(data_dir) / "train.csv.gz")
    df["comment_text"] = df["comment_text"].fillna("").astype(str)
    return df


def load_test(data_dir: Path | str = DATA_DIR, scored_only: bool = True) -> pd.DataFrame:
    """Load the test set joined with its labels.

    With ``scored_only=True`` (default) the -1 unscored rows are dropped, leaving
    the 63,978 rows Kaggle actually scores. This is the only correct basis for a
    reported test metric.
    """
    data_dir = Path(data_dir)
    text = pd.read_csv(data_dir / "test.csv.gz")
    labels = pd.read_csv(data_dir / "test_labels.csv.gz")
    df = text.merge(labels, on="id", how="inner")
    df["comment_text"] = df["comment_text"].fillna("").astype(str)
    if scored_only:
        # A row is unscored iff its labels are -1 (all six share the same flag).
        df = df[df["toxic"] != -1].reset_index(drop=True)
    return df


def label_matrix(df: pd.DataFrame) -> np.ndarray:
    """Return the (n, 6) binary label matrix as float32."""
    return df[LABELS].to_numpy(dtype=np.float32)


def stratified_subsample(
    df: pd.DataFrame,
    n_negatives: int = 20_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Keep every row with >=1 positive label; downsample the clean majority.

    A naive random subsample would leave only a handful of rare-label positives
    (e.g. ~60 threats out of 478). Keeping all positives preserves that signal
    while cutting compute by ~4x.
    """
    rng = np.random.RandomState(seed)
    positive_mask = df[LABELS].sum(axis=1) > 0
    positives = df[positive_mask]
    negatives = df[~positive_mask]
    n_negatives = min(n_negatives, len(negatives))
    neg_idx = rng.choice(negatives.index.to_numpy(), size=n_negatives, replace=False)
    sampled = pd.concat([positives, negatives.loc[neg_idx]])
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def describe_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Per-label positive counts and prevalence — used in EDA."""
    counts = df[LABELS].sum().astype(int)
    return pd.DataFrame(
        {"positives": counts, "prevalence_%": (counts / len(df) * 100).round(3)}
    ).sort_values("positives", ascending=False)
