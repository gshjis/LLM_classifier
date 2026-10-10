import pandas as pd
import pytest

from llm_classifier.data import load_dataset, split_dataset


def test_load_dataset_normalizes_labels(tmp_path):
    path = tmp_path / "emails.csv"
    labels = [1, 0, "spam", "ham"] * 5
    pd.DataFrame({"text": [f"email {index}" for index in range(len(labels))], "label": labels}).to_csv(path, index=False)
    frame = load_dataset(path)
    assert frame["label"].tolist()[:4] == ["spam", "not_spam", "spam", "not_spam"]


def test_load_sms_spam_collection_with_trailing_empty_columns(tmp_path):
    path = tmp_path / "spam.csv"
    rows = ["v1,v2,,,\n"]
    rows.extend(f'{label},"message {index}",,,\n' for index, label in enumerate(["ham", "spam"] * 10))
    path.write_text("".join(rows), encoding="utf-8")

    frame = load_dataset(path)

    assert frame.columns.tolist() == ["text", "label"]
    assert frame.iloc[0].to_dict() == {"text": "message 0", "label": "not_spam"}
    assert set(frame["label"]) == {"spam", "not_spam"}


def test_load_dataset_falls_back_to_latin1_encoding(tmp_path):
    path = tmp_path / "spam.csv"
    rows = ["v1,v2,,,\n"]
    rows.extend(f'{label},"message {index} caf\xe9",,,\n' for index, label in enumerate(["ham", "spam"] * 10))
    path.write_bytes("".join(rows).encode("latin-1"))

    frame = load_dataset(path)

    assert frame.iloc[0]["text"] == "message 0 café"
    assert set(frame["label"]) == {"spam", "not_spam"}


def test_load_dataset_requires_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"email": [f"hello {i}" for i in range(20)], "category": ["spam", "ham"] * 10}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="columns"):
        load_dataset(path)


def test_split_is_stratified_and_complete(sample_frame):
    train, validation, test = split_dataset(sample_frame)
    assert len(train) + len(validation) + len(test) == len(sample_frame)
    assert all(set(part["label"]) == {"spam", "not_spam"} for part in (train, validation, test))
