"""Dataset encoding helpers for Transformers."""

from datasets import Dataset
import pandas as pd

LABEL_TO_ID = {"not_spam": 0, "spam": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


def to_hf_dataset(frame: pd.DataFrame, tokenizer, max_length: int) -> Dataset:
    frame = frame[["text", "label"]].copy()
    unknown = set(frame["label"].astype(str)) - set(LABEL_TO_ID)
    if unknown:
        raise ValueError(f"Unsupported labels: {sorted(unknown)}")
    dataset = Dataset.from_dict(
        {"text": frame["text"].astype(str).tolist(), "labels": [LABEL_TO_ID[label] for label in frame["label"].astype(str)]}
    )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return dataset.map(tokenize, batched=True, remove_columns=["text"])
