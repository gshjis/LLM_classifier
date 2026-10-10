"""Download datasets (e.g., Kaggle) into expected local paths."""

from __future__ import annotations

import shutil
from pathlib import Path


def download_kaggle_dataset(*, handle: str, output_path: str | Path) -> Path:
    """Download a Kaggle dataset via kagglehub.

    Notes:
      - For public datasets, Kaggle token is typically not required.
      - The project expects SMS Spam Collection at data/raw/spam.csv.
    """

    # Import lazily so unit tests can monkeypatch kagglehub.
    import kagglehub

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    downloaded_dir = Path(kagglehub.dataset_download(handle))

    # SMS Spam Collection contains spam.csv inside the downloaded directory.
    # We copy the first spam.csv we can find.
    candidates = sorted(downloaded_dir.rglob("spam.csv"))
    if not candidates:
        raise FileNotFoundError(f"Could not find spam.csv inside downloaded dataset for handle={handle!r}")

    src = candidates[0]
    shutil.copyfile(src, output_path)
    return output_path


def download_sms_spam_collection(*, output_path: str | Path = "data/raw/spam.csv") -> Path:
    """Convenience wrapper for the dataset referenced in README."""

    return download_kaggle_dataset(
        handle="uciml/sms-spam-collection-dataset",
        output_path=output_path,
    )
