from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _guess_text_column(features: dict[str, Any]) -> str:
    for name in ("text", "utterance", "sentence", "query"):
        if name in features:
            return name
    for name, feat in features.items():
        if getattr(feat, "dtype", None) == "string":
            return name
    raise ValueError(f"Could not find a text column in: {list(features.keys())}")


def _guess_label_column(features: dict[str, Any]) -> str:
    for name in ("label", "intent", "category"):
        if name in features:
            return name
    for name, feat in features.items():
        if hasattr(feat, "names"):
            return name
    for name, feat in features.items():
        if getattr(feat, "dtype", None) in {"int64", "int32", "int16", "int8"}:
            return name
    raise ValueError(f"Could not find a label column in: {list(features.keys())}")


def _make_splits(
    ds: DatasetDict,
    *,
    label_col: str,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> tuple[Dataset, Dataset, Dataset]:
    if {"train", "validation", "test"}.issubset(ds.keys()):
        return ds["train"], ds["validation"], ds["test"]

    if {"train", "test"}.issubset(ds.keys()):
        split = ds["train"].train_test_split(
            test_size=val_ratio, seed=seed, stratify_by_column=label_col
        )
        return split["train"], split["test"], ds["test"]

    split_name = next(iter(ds.keys()))
    all_ds = ds[split_name]
    tmp = all_ds.train_test_split(
        test_size=val_ratio + test_ratio, seed=seed, stratify_by_column=label_col
    )
    tmp2 = tmp["test"].train_test_split(
        test_size=test_ratio / (val_ratio + test_ratio), seed=seed, stratify_by_column=label_col
    )
    return tmp["train"], tmp2["train"], tmp2["test"]


def _build_label_maps(ds: Dataset, *, label_col: str) -> tuple[dict[str, int], dict[int, str]]:
    feat = ds.features.get(label_col)
    if feat is not None and hasattr(feat, "names"):
        id2label = {i: str(name) for i, name in enumerate(list(feat.names))}
        label2id = {name: i for i, name in id2label.items()}
        return label2id, id2label

    label_values = sorted({int(x) for x in ds[label_col]})
    id2label = {i: str(i) for i in label_values}
    label2id = {v: k for k, v in id2label.items()}
    return label2id, id2label


def _save_label_map(path: Path, *, label2id: dict[str, int], id2label: dict[int, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _delete_random_char(text: str, rng: random.Random) -> str:
    if len(text) < 4:
        return text
    i = rng.randrange(0, len(text))
    return text[:i] + text[i + 1 :]


def _swap_adjacent_chars(text: str, rng: random.Random) -> str:
    if len(text) < 4:
        return text
    i = rng.randrange(0, len(text) - 1)
    return text[:i] + text[i + 1] + text[i] + text[i + 2 :]


def _drop_short_word(text: str, rng: random.Random) -> str:
    words = text.split()
    short_idxs = [i for i, w in enumerate(words) if len(w) <= 3]
    if not short_idxs:
        return text
    idx = rng.choice(short_idxs)
    return " ".join(words[:idx] + words[idx + 1 :])


def _duplicate_word(text: str, rng: random.Random) -> str:
    words = text.split()
    if len(words) < 2:
        return text
    idx = rng.randrange(0, len(words))
    return " ".join(words[: idx + 1] + [words[idx]] + words[idx + 1 :])


def _apply_noise(text: str, rng: random.Random) -> str:
    ops = [_delete_random_char, _swap_adjacent_chars, _drop_short_word, _duplicate_word]
    op = rng.choice(ops)
    text2 = op(text, rng)
    if rng.random() < 0.25:
        op2 = rng.choice(ops)
        text2 = op2(text2, rng)
    return text2


def _maybe_noise(text: str, *, seed: int, idx: int, noise_prob: float) -> str:
    if noise_prob <= 0:
        return text
    rng = random.Random(seed + int(idx))
    if rng.random() >= noise_prob:
        return text
    return _apply_noise(text, rng)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a transformer intent model (DistilBERT by default).")
    parser.add_argument("--dataset", default="banking77", help="HF datasets name (default: banking77).")
    parser.add_argument("--model", default="distilbert-base-uncased", help="HF model name (default: distilbert-base-uncased).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Used only when dataset has no test split.")
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--noise-prob", type=float, default=0.15, help="ASR-like text noise prob (training only).")
    parser.add_argument("--output-dir", default=None, help="Default: models/slu_transformer")
    args = parser.parse_args()

    root = _project_root()
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir) if args.output_dir else (models_dir / "slu_transformer")
    label_map_out = root / "slu" / "label_map.json"

    ds = load_dataset(args.dataset)
    train_ref = ds["train"] if "train" in ds else ds[next(iter(ds.keys()))]
    text_col = _guess_text_column(train_ref.features)
    label_col = _guess_label_column(train_ref.features)
    train_ds, val_ds, test_ds = _make_splits(
        ds, label_col=label_col, seed=args.seed, val_ratio=args.val_ratio, test_ratio=args.test_ratio
    )

    label2id, id2label = _build_label_maps(train_ds, label_col=label_col)
    _save_label_map(label_map_out, label2id=label2id, id2label=id2label)

    if args.noise_prob > 0:
        train_ds = train_ds.map(
            lambda ex, idx: {text_col: _maybe_noise(ex[text_col], seed=args.seed, idx=idx, noise_prob=args.noise_prob)},
            with_indices=True,
            desc="Adding mild ASR-like noise",
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(batch[text_col], truncation=True, max_length=args.max_length)

    def add_labels(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return {"labels": batch[label_col]}

    train_tok = train_ds.map(add_labels, batched=True).map(
        tokenize, batched=True, remove_columns=[text_col, label_col]
    )
    val_tok = val_ds.map(add_labels, batched=True).map(tokenize, batched=True, remove_columns=[text_col, label_col])
    test_tok = test_ds.map(add_labels, batched=True).map(tokenize, batched=True, remove_columns=[text_col, label_col])

    config = AutoConfig.from_pretrained(
        args.model,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )
    model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1_macro": float(f1_score(labels, preds, average="macro")),
        }

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(args.epochs),
        per_device_train_batch_size=int(args.batch_size),
        per_device_eval_batch_size=int(args.batch_size),
        learning_rate=float(args.lr),
        weight_decay=float(args.weight_decay),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        logging_strategy="steps",
        logging_steps=50,
        report_to="none",
        seed=int(args.seed),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print(f"dataset={args.dataset} text_col={text_col} label_col={label_col} labels={len(label2id)}")
    trainer.train()
    val_metrics = trainer.evaluate(eval_dataset=val_tok)
    test_metrics = trainer.evaluate(eval_dataset=test_tok)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("val :", {k: round(float(v), 4) for k, v in val_metrics.items() if isinstance(v, (int, float))})
    print("test:", {k: round(float(v), 4) for k, v in test_metrics.items() if isinstance(v, (int, float))})
    print(f"saved: {output_dir}")
    print(f"saved: {label_map_out}")


if __name__ == "__main__":
    main()

