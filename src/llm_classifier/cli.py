"""Command-line interface for the project workflows."""

import argparse
import json
from pathlib import Path

from llm_classifier.baselines.evaluate import evaluate_baselines
from llm_classifier.baselines.models import load_model
from llm_classifier.baselines.train import train_baselines
from llm_classifier.config import load_config
from llm_classifier.data import prepare_data
from llm_classifier.data_download import download_sms_spam_collection
from llm_classifier.llm.evaluate import evaluate_llm
from llm_classifier.llm.predict import predict_llm
from llm_classifier.llm.train import train_llm


def _config_path(value: str | None, default: str) -> Path:
    return Path(value or default)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and run spam classification models")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Split raw CSV into train/validation/test")
    prepare.add_argument("--input", default="data/raw/spam.csv")
    prepare.add_argument("--output-dir", default="data/processed")
    prepare.add_argument("--random-state", type=int, default=42)

    for name in ("train-baselines", "evaluate-baselines", "train-llm", "evaluate-llm"):
        command = subparsers.add_parser(name)
        command.add_argument("--config")

    baseline_predict = subparsers.add_parser("predict-baseline")
    baseline_predict.add_argument("--text", required=True)
    baseline_predict.add_argument("--model", default="logistic_regression")
    baseline_predict.add_argument("--model-dir", default="outputs/models/baselines")

    download = subparsers.add_parser(
        "download-data",
        help="Download dataset into data/raw/spam.csv (via kagglehub)",
    )
    download.add_argument("--output", default="data/raw/spam.csv")

    llm_predict = subparsers.add_parser("predict-llm")
    llm_predict.add_argument("--text", required=True)
    llm_predict.add_argument("--model-dir", default="outputs/models/llm")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-data":
        paths = prepare_data(args.input, args.output_dir, random_state=args.random_state)
        print("Prepared splits:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
    elif args.command == "download-data":
        out = download_sms_spam_collection(output_path=args.output)
        print(f"Downloaded dataset to: {out}")
    elif args.command in {"train-baselines", "evaluate-baselines"}:
        config = load_config(_config_path(args.config, "configs/baseline.yaml"))
        (train_baselines if args.command == "train-baselines" else evaluate_baselines)(config)
    elif args.command in {"train-llm", "evaluate-llm"}:
        config = load_config(_config_path(args.config, "configs/llm.yaml"))
        (train_llm if args.command == "train-llm" else evaluate_llm)(config)
    elif args.command == "predict-baseline":
        model_path = Path(args.model_dir) / f"{args.model}.joblib"
        prediction = load_model(model_path).predict([args.text])[0]
        print(json.dumps({"label": str(prediction)}, ensure_ascii=False))
    elif args.command == "predict-llm":
        print(json.dumps(predict_llm(args.text, args.model_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
