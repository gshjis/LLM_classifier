"""CPU fine-tuning for a Hugging Face sequence classifier."""

from pathlib import Path

import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed

from llm_classifier.llm.dataset import ID_TO_LABEL, LABEL_TO_ID, to_hf_dataset


def train_llm(config: dict) -> Path:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    model_name = config.get("model_name", "cointegrated/rubert-tiny2")
    output_dir = Path(config.get("output_dir", "outputs/models/llm"))
    max_length = int(config.get("max_length", 256))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, label2id=LABEL_TO_ID, id2label=ID_TO_LABEL
    )
    train_frame = pd.read_csv(config["data"]["train"])
    validation_frame = pd.read_csv(config["data"]["validation"])
    train_dataset = to_hf_dataset(train_frame, tokenizer, max_length)
    validation_dataset = to_hf_dataset(validation_frame, tokenizer, max_length)
    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=float(config.get("epochs", 1)),
        per_device_train_batch_size=int(config.get("batch_size", 4)),
        per_device_eval_batch_size=int(config.get("batch_size", 4)),
        learning_rate=float(config.get("learning_rate", 2e-5)),
        weight_decay=float(config.get("weight_decay", 0.01)),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        use_cpu=True,
        report_to="none",
        logging_strategy="steps",
        logging_steps=10,
        seed=seed,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=validation_dataset)
    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved fine-tuned model: {output_dir}")
    return output_dir
