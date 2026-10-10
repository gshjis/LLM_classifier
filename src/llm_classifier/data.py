"""Loading, validation, and splitting of message classification datasets."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_ALIASES = {
    "spam": "spam",
    "1": "spam",
    "not_spam": "not_spam",
    "not spam": "not_spam",
    "ham": "not_spam",
    "0": "not_spam",
}


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        # SMS Spam Collection is commonly distributed in latin-1/cp1252.
        return pd.read_csv(path, encoding="latin-1")


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = _read_csv(path)
    if {"text", "label"}.issubset(frame.columns):
        frame = frame[["text", "label"]].copy()
    elif {"v1", "v2"}.issubset(frame.columns):
        # SMS Spam Collection uses v1=label and v2=text, sometimes with
        # trailing empty columns in the CSV header and every row.
        frame = frame.rename(columns={"v1": "label", "v2": "text"})[["text", "label"]].copy()
    else:
        raise ValueError(
            "Dataset must contain either 'text' and 'label' columns or SMS Spam Collection columns 'v1' and 'v2'"
        )
    frame = frame.dropna(subset=["text", "label"]).copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame["label"] = frame["label"].astype(str).str.strip().str.lower().map(LABEL_ALIASES)
    frame = frame.dropna(subset=["label"])
    frame = frame[frame["text"] != ""].reset_index(drop=True)
    if frame["label"].nunique() != 2:
        raise ValueError("Dataset must contain examples of both spam and not_spam")
    if frame["label"].value_counts().min() < 10:
        raise ValueError("Each class must contain at least 10 examples for stratified train/validation/test splits")
    return frame


def split_dataset(
    frame: pd.DataFrame,
    train_size: float = 0.8,
    validation_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_size <= 0 or validation_size <= 0 or train_size + validation_size >= 1:
        raise ValueError("train_size and validation_size must be positive and sum to less than 1")
    train, remainder = train_test_split(
        frame, test_size=1 - train_size, random_state=random_state, stratify=frame["label"]
    )
    relative_validation_size = validation_size / (1 - train_size)
    validation, test = train_test_split(
        remainder,
        test_size=1 - relative_validation_size,
        random_state=random_state,
        stratify=remainder["label"],
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


def prepare_data(
    input_path: str | Path = "data/raw/spam.csv",
    output_dir: str | Path = "data/processed",
    train_size: float = 0.8,
    validation_size: float = 0.1,
    random_state: int = 42,
) -> dict[str, Path]:
    frame = load_dataset(input_path)
    splits = split_dataset(frame, train_size, validation_size, random_state)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, split in zip(("train", "validation", "test"), splits, strict=True):
        path = output_dir / f"{name}.csv"
        split.to_csv(path, index=False)
        paths[name] = path
    return paths
