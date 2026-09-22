"""TF-IDF + One-vs-Rest Logistic Regression baseline.

A deliberately strong classical reference. On this dataset a well-tuned TF-IDF +
linear model reaches ~0.97-0.98 mean column ROC-AUC and is hard to beat — which is
itself an instructive finding for the report. Trains on the FULL data (sparse and
cheap) and doubles as an explainability tool via per-label coefficients.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from data import LABELS


class TfidfLogReg:
    def __init__(self, max_features: int = 50_000, C: float = 4.0, seed: int = 42):
        self.word_vec = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            ngram_range=(1, 2),
            max_features=max_features,
            min_df=3,
        )
        self.char_vec = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="char",
            ngram_range=(2, 4),
            max_features=max_features,
            min_df=3,
        )
        self.C = C
        self.seed = seed
        self.models: dict[str, LogisticRegression] = {}

    def _features(self, texts, fit: bool) -> csr_matrix:
        from scipy.sparse import hstack

        if fit:
            w = self.word_vec.fit_transform(texts)
            c = self.char_vec.fit_transform(texts)
        else:
            w = self.word_vec.transform(texts)
            c = self.char_vec.transform(texts)
        return hstack([w, c]).tocsr()

    def fit(self, texts, Y: np.ndarray) -> "TfidfLogReg":
        X = self._features(texts, fit=True)
        for j, lab in enumerate(LABELS):
            clf = LogisticRegression(
                C=self.C, max_iter=1000, solver="liblinear", random_state=self.seed
            )
            clf.fit(X, Y[:, j])
            self.models[lab] = clf
        return self

    def predict_proba(self, texts) -> np.ndarray:
        X = self._features(texts, fit=False)
        probs = np.column_stack(
            [self.models[lab].predict_proba(X)[:, 1] for lab in LABELS]
        )
        return probs

    def top_features(self, label: str, n: int = 15):
        """Most toxic-pushing n-grams for a label (explainability)."""
        names = np.concatenate(
            [self.word_vec.get_feature_names_out(), self.char_vec.get_feature_names_out()]
        )
        coefs = self.models[label].coef_.ravel()
        top = np.argsort(coefs)[-n:][::-1]
        return list(zip(names[top], coefs[top]))
