from __future__ import annotations

from pathlib import Path


def test_download_sms_spam_collection_copies_spam_csv(tmp_path, monkeypatch):
    # Arrange: fake kagglehub.dataset_download output directory structure.
    downloaded_dir = tmp_path / "kaggle"
    downloaded_dir.mkdir(parents=True, exist_ok=True)
    (downloaded_dir / "nested").mkdir(parents=True, exist_ok=True)
    (downloaded_dir / "nested" / "spam.csv").write_text("v1,v2\nham,hello\nspam,bad\n", encoding="utf-8")

    def fake_dataset_download(handle: str) -> str:
        assert handle == "uciml/sms-spam-collection-dataset"
        return str(downloaded_dir)

    import llm_classifier.data_download as dl

    monkeypatch.setattr(dl, "kagglehub", type("X", (), {"dataset_download": staticmethod(fake_dataset_download)}))

    # Act
    out = dl.download_sms_spam_collection(output_path=tmp_path / "data" / "raw" / "spam.csv")

    # Assert
    assert out.exists()
    assert out.read_text(encoding="utf-8").startswith("v1,v2")
