"""Evaluation metrics and threshold tuning for multi-label classification.

Metrics reported (all per-label and aggregated):
    * ROC-AUC       – threshold-independent ranking quality.
    * PR-AUC (AP)   – far more informative than ROC-AUC under <1% prevalence.
    * mean col ROC-AUC – the official Jigsaw competition metric.
    * F1 (macro/micro) at per-label thresholds tuned on the validation set.

Thresholds are tuned on validation predictions ONLY, never on test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data import LABELS


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Per-label threshold that maximises F1 on the given (validation) set."""
    thresholds = np.zeros(y_true.shape[1])
    grid = np.linspace(0.05, 0.95, 19)
    for j in range(y_true.shape[1]):
        if y_true[:, j].sum() == 0:
            thresholds[j] = 0.5
            continue
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            f1 = f1_score(y_true[:, j], (y_prob[:, j] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[j] = best_t
    return thresholds


def per_label_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-label ROC-AUC, PR-AUC, precision/recall/F1 at the given thresholds."""
    if thresholds is None:
        thresholds = np.full(y_true.shape[1], 0.5)
    rows = []
    for j, lab in enumerate(LABELS):
        yt, yp = y_true[:, j], y_prob[:, j]
        pred = (yp >= thresholds[j]).astype(int)
        has_pos = yt.sum() > 0
        rows.append(
            {
                "label": lab,
                "roc_auc": roc_auc_score(yt, yp) if has_pos else np.nan,
                "pr_auc": average_precision_score(yt, yp) if has_pos else np.nan,
                "precision": precision_score(yt, pred, zero_division=0),
                "recall": recall_score(yt, pred, zero_division=0),
                "f1": f1_score(yt, pred, zero_division=0),
                "threshold": thresholds[j],
                "support": int(yt.sum()),
            }
        )
    return pd.DataFrame(rows)


def summary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict:
    """Aggregate headline metrics for the model-comparison table."""
    if thresholds is None:
        thresholds = np.full(y_true.shape[1], 0.5)
    pred = (y_prob >= thresholds).astype(int)
    per_label_auc = [
        roc_auc_score(y_true[:, j], y_prob[:, j])
        for j in range(y_true.shape[1])
        if y_true[:, j].sum() > 0
    ]
    return {
        "mean_col_roc_auc": float(np.mean(per_label_auc)),  # official Jigsaw metric
        "macro_pr_auc": float(
            np.mean(
                [
                    average_precision_score(y_true[:, j], y_prob[:, j])
                    for j in range(y_true.shape[1])
                    if y_true[:, j].sum() > 0
                ]
            )
        ),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, pred, average="micro", zero_division=0),
    }
