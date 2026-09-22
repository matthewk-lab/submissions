"""Explainability helpers.

Two complementary views:
    * Global — per-label Logistic Regression coefficients (see
      ``models_baseline.TfidfLogReg.top_features``): which n-grams push a label up.
    * Local — LIME on individual comments for any probabilistic model (baseline or
      transformer), showing which words drove a specific prediction.

LIME is model-agnostic: it perturbs the input text and fits a local linear surrogate,
so it works identically for the TF-IDF baseline and the transformer.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from lime.lime_text import LimeTextExplainer

from data import LABELS


def make_explainer() -> LimeTextExplainer:
    return LimeTextExplainer(class_names=["not", "is"])


def explain_comment(
    text: str,
    predict_fn: Callable[[list[str]], np.ndarray],
    label: str = "toxic",
    num_features: int = 10,
    num_samples: int = 500,
):
    """Explain one comment's prediction for a single label.

    ``predict_fn`` maps a list of texts -> (n, 6) probability array. LIME needs a
    2-column (P(not), P(is)) matrix for the target label, which we adapt here.
    """
    j = LABELS.index(label)

    def binary_predict(texts: list[str]) -> np.ndarray:
        p = predict_fn(texts)[:, j]
        return np.column_stack([1 - p, p])

    explainer = make_explainer()
    exp = explainer.explain_instance(
        text, binary_predict, num_features=num_features, num_samples=num_samples
    )
    return exp  # exp.as_list() -> [(word, weight), ...]; exp.as_html() for notebooks
