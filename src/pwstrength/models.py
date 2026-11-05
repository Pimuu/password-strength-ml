"""Model definitions, non-learned baselines and evaluation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .labeler import LABELS, STRONG_MIN_BITS, WEAK_MAX_BITS


def make_models(seed: int = 42) -> dict:
    """The three learned models compared in the paper.

    * Logistic regression: a linear, interpretable reference. Features are
      standardised so the coefficients can be compared.
    * Random forest: a non-linear bagged ensemble with impurity importances.
    * Histogram gradient boosting: a boosted-tree ensemble (LightGBM-style).
    All three use balanced class weights because 'medium' is the smallest class.
    """
    return {
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0),
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=1, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=seed,
        ),
    }


# ---------------------------------------------------------------------------
# Non-learned baselines (operate on the feature frame)
# ---------------------------------------------------------------------------
def luds_meter(X: pd.DataFrame) -> np.ndarray:
    """A typical composition-rule meter (Lowercase/Uppercase/Digits/Symbols + length).

    One point each for: length >= 8, length >= 12, both cases present, a digit,
    a symbol. 0-2 points = weak, 3 = medium, 4-5 = strong.
    """
    pts = ((X["length"] >= 8).astype(int) + (X["length"] >= 12).astype(int)
           + ((X["n_lower"] > 0) & (X["n_upper"] > 0)).astype(int)
           + (X["n_digit"] > 0).astype(int) + (X["n_symbol"] > 0).astype(int))
    return np.select([pts <= 2, pts == 3], [0, 1], default=2)


def pool_entropy_meter(X: pd.DataFrame) -> np.ndarray:
    """Naive entropy meter: length x log2(alphabet size), with the labeler's thresholds."""
    b = X["pool_entropy_bits"]
    return np.select([b < WEAK_MAX_BITS, b < STRONG_MIN_BITS], [0, 1], default=2)


BASELINES = {
    "Baseline_LUDS_meter": luds_meter,
    "Baseline_pool_entropy": pool_entropy_meter,
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred) -> dict:
    """Accuracy, macro and weighted P/R/F1, plus per-class P/R/F1."""
    out = {"accuracy": accuracy_score(y_true, y_pred)}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        out[f"precision_{avg}"], out[f"recall_{avg}"], out[f"f1_{avg}"] = p, r, f
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    for k, name in enumerate(LABELS):
        out[f"precision_{name}"], out[f"recall_{name}"], out[f"f1_{name}"] = p[k], r[k], f[k]
    return out


def confusion(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
