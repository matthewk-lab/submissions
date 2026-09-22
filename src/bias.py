"""Responsible-AI: identity-term bias probe.

The well-documented failure mode of models trained on this dataset is *unintended
identity bias*: benign mentions of identity terms ("gay", "muslim", "black", ...) are
disproportionately flagged as toxic because those terms co-occur with toxicity in the
training data. This module measures the effect by comparing, for a given label,
the model's **false positive rate** on the subgroup of test comments mentioning each
identity term against the background FPR.

A large positive gap = the model penalises mention of that identity regardless of the
comment's actual sentiment. This is the analysis that motivated the follow-up
Jigsaw Unintended Bias in Toxicity Classification competition.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Identity terms grouped by axis (from the Jigsaw Unintended Bias term list).
IDENTITY_TERMS = {
    "gender": ["woman", "women", "man", "men", "female", "male", "trans", "transgender"],
    "sexuality": ["gay", "lesbian", "homosexual", "bisexual", "lgbt", "queer"],
    "religion": ["muslim", "islam", "jewish", "jew", "christian", "catholic", "buddhist", "hindu"],
    "race": ["black", "white", "asian", "african", "latino", "hispanic", "indian"],
    "other": ["disabled", "deaf", "blind", "old", "young"],
}

ALL_TERMS = sorted({t for terms in IDENTITY_TERMS.values() for t in terms})


def _mentions(texts: pd.Series, term: str) -> np.ndarray:
    pat = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    return texts.str.contains(pat).to_numpy()


def subgroup_fpr(
    texts: pd.Series,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_index: int = 0,          # default: 'toxic'
    threshold: float = 0.5,
    terms: list[str] | None = None,
    min_support: int = 30,
) -> pd.DataFrame:
    """False positive rate per identity subgroup for one label.

    FPR is computed only over *negative* (non-toxic) examples: of the comments that
    are genuinely not toxic, what fraction does the model wrongly flag? Comparing the
    subgroup FPR to the background FPR isolates identity bias from real toxicity.
    """
    terms = terms or ALL_TERMS
    pred = (y_prob[:, label_index] >= threshold).astype(int)
    truth = y_true[:, label_index].astype(int)
    neg = truth == 0
    background_fpr = pred[neg].mean()

    rows = []
    for term in terms:
        m = _mentions(texts, term)
        sub_neg = m & neg
        support = int(sub_neg.sum())
        if support < min_support:
            continue
        fpr = pred[sub_neg].mean()
        rows.append(
            {"term": term, "subgroup_neg_n": support,
             "subgroup_fpr": fpr, "background_fpr": background_fpr,
             "fpr_gap": fpr - background_fpr}
        )
    return (
        pd.DataFrame(rows)
        .sort_values("fpr_gap", ascending=False)
        .reset_index(drop=True)
    )
