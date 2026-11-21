"""Validation-only decision-threshold selection; does not calibrate probabilities.

Callers must supply a dedicated validation set, never final-test labels. Threshold
selection changes decisions, not AP/AUPRC or ROC-AUC of the underlying scores.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)


def validate_scores(y, probabilities):
    y, p = np.asarray(y), np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or not len(y):
        raise ValueError("Labels and probabilities must be nonempty aligned vectors")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be finite and between zero and one")
    return y.astype(int), p


def select_threshold(y_validation, validation_probabilities):
    """Maximize validation F1; break exact ties with the highest threshold.

    PR thresholds enumerate every distinct achievable decision boundary using
    prediction >= threshold, including tied scores. Missing classes are an error:
    a useful validation estimate cannot be obtained from a single-class sample.
    """
    y, p = validate_scores(y_validation, validation_probabilities)
    if len(np.unique(y)) != 2:
        raise ValueError("Threshold selection needs both classes in validation")
    precision, recall, thresholds = precision_recall_curve(y, p)
    denom = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denom,
                   out=np.zeros_like(denom), where=denom > 0)
    winner = np.flatnonzero(f1 == f1.max())[-1]
    return {"threshold": float(thresholds[winner]),
            "validation_f1": float(f1[winner]),
            "validation_precision": float(precision[winner]),
            "validation_recall": float(recall[winner]),
            "validation_n": len(y), "validation_frauds": int(y.sum()),
            "objective": "maximum validation F1; highest threshold breaks ties"}


def score_probabilities(y, probabilities, threshold=0.5):
    y, p = validate_scores(y, probabilities)
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {
        "threshold": float(threshold), "n": len(y), "frauds": int(y.sum()),
        "auprc": float(average_precision_score(y, p)) if y.sum() else None,
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "accuracy": float(np.mean(pred == y)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "predicted_frauds": int(pred.sum()),
    }
