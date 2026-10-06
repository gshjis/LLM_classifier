"""Train all configured classical baselines."""

from pathlib import Path

import pandas as pd

from llm_classifier.baselines.models import build_models, save_model


def train_baselines(config: dict) -> dict[str, Path]:
    train = pd.read_csv(config["data"]["train"])
    output_dir = Path(config.get("output_dir", "outputs/models/baselines"))
    saved = {}
    for name, model in build_models(config).items():
        model.fit(train["text"].astype(str), train["label"].astype(str))
        path = output_dir / f"{name}.joblib"
        save_model(model, path)
        saved[name] = path
        print(f"Saved {name}: {path}")
    return saved
