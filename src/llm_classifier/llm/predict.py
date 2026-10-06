"""Single-text inference using a fine-tuned sequence classifier."""

from pathlib import Path

from transformers import pipeline


def predict_llm(text: str, model_dir: str | Path = "outputs/models/llm") -> dict[str, str | float]:
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Trained LLM not found at {model_dir}; run train-llm first")
    classifier = pipeline("text-classification", model=str(model_dir), tokenizer=str(model_dir), device=-1)
    result = classifier(text, truncation=True)[0]
    label = result["label"]
    if label.startswith("LABEL_"):
        label = "spam" if label.endswith("1") else "not_spam"
    return {"label": label, "score": float(result["score"])}
