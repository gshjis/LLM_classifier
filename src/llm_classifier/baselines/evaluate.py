"""Evaluate saved baselines on the held-out test set."""

from pathlib import Path

import pandas as pd

from llm_classifier.baselines.models import load_model
from llm_classifier.metrics import calculate_metrics, report_text, save_metrics


def evaluate_baselines(config: dict) -> dict[str, dict[str, float]]:
    test = pd.read_csv(config["data"]["test"])
    output_dir = Path(config.get("output_dir", "outputs/models/baselines"))
    metrics_dir = Path(config.get("metrics_dir", "outputs/metrics"))
    results = {}
    for model_path in sorted(output_dir.glob("*.joblib")):
        model = load_model(model_path)
        predictions = model.predict(test["text"].astype(str))
        metrics = calculate_metrics(test["label"].astype(str), predictions)
        results[model_path.stem] = metrics
        save_metrics(metrics, metrics_dir / f"{model_path.stem}.json")
        print(f"\n{model_path.stem}\n{report_text(test['label'].astype(str), predictions)}")
        print(metrics)
    if not results:
        raise FileNotFoundError(f"No trained baseline models found in {output_dir}; run train-baselines first")
    return results
