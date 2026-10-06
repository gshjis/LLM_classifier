"""Evaluate a saved Hugging Face classifier on the test set."""

from pathlib import Path

import pandas as pd
from transformers import pipeline

from llm_classifier.llm.dataset import ID_TO_LABEL
from llm_classifier.metrics import calculate_metrics, report_text, save_metrics


def evaluate_llm(config: dict) -> dict[str, float]:
    model_dir = Path(config.get("output_dir", "outputs/models/llm"))
    if not model_dir.exists():
        raise FileNotFoundError(f"Trained LLM not found at {model_dir}; run train-llm first")
    test = pd.read_csv(config["data"]["test"])
    classifier = pipeline("text-classification", model=str(model_dir), tokenizer=str(model_dir), device=-1)
    outputs = classifier(test["text"].astype(str).tolist(), truncation=True, batch_size=int(config.get("batch_size", 4)))
    predictions = [ID_TO_LABEL[int(item["label"].split("_")[-1])] if item["label"].startswith("LABEL_") else item["label"] for item in outputs]
    metrics = calculate_metrics(test["label"].astype(str), predictions)
    save_metrics(metrics, Path(config.get("metrics_dir", "outputs/metrics")) / "llm.json")
    print(report_text(test["label"].astype(str), predictions))
    print(metrics)
    return metrics
