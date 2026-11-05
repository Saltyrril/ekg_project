"""Метрики multi-label классификации (общие для бейзлайна и нейросетей)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def multilabel_metrics(y_true: np.ndarray, y_score: np.ndarray, classes: list[str]) -> dict[str, object]:
    """AUROC и PR-AUC по классам + macro-среднее.

    y_true: (N, C) {0,1}; y_score: (N, C) вероятности/скоры. Оба метрики не зависят от порога.
    """
    auroc = {c: float(roc_auc_score(y_true[:, i], y_score[:, i])) for i, c in enumerate(classes)}
    pr_auc = {c: float(average_precision_score(y_true[:, i], y_score[:, i])) for i, c in enumerate(classes)}
    return {
        "macro_auroc": float(np.mean(list(auroc.values()))),
        "auroc": auroc,
        "macro_pr_auc": float(np.mean(list(pr_auc.values()))),
        "pr_auc": pr_auc,
    }
