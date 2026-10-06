"""Shared classification metrics and serialization."""

import json
from pathlib import Path
from typing import Sequence

from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score


def calculate_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_spam": float(precision_score(y_true, y_pred, pos_label="spam", zero_division=0)),
        "recall_spam": float(recall_score(y_true, y_pred, pos_label="spam", zero_division=0)),
        "f1_spam": float(f1_score(y_true, y_pred, pos_label="spam", zero_division=0)),
    }


def save_metrics(metrics: dict[str, float], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def report_text(y_true: Sequence[str], y_pred: Sequence[str]) -> str:
    return classification_report(y_true, y_pred, labels=["not_spam", "spam"], zero_division=0)
