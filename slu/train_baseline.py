from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


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


def _evaluate(
    name: str,
    *,
    vectorizer: TfidfVectorizer,
    clf: LogisticRegression,
    texts: list[str],
    y_true: np.ndarray,
) -> dict[str, float]:
    x = vectorizer.transform(texts)
    y_pred = clf.predict(x)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a classical ML intent baseline (TF-IDF + LogisticRegression).")
    parser.add_argument("--dataset", default="banking77", help="HF datasets name (default: banking77).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Used only when dataset has no test split.")
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--model-out", default=None, help="Path to write model (default: models/slu_baseline.pkl).")
    args = parser.parse_args()

    root = _project_root()
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_out = Path(args.model_out) if args.model_out else (models_dir / "slu_baseline.pkl")
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

    x_train = list(train_ds[text_col])
    y_train = np.asarray(train_ds[label_col], dtype=np.int64)

    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=args.max_features)
    x_train_vec = vectorizer.fit_transform(x_train)

    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        multi_class="multinomial",
    )
    clf.fit(x_train_vec, y_train)

    metrics_train = _evaluate(
        "train",
        vectorizer=vectorizer,
        clf=clf,
        texts=list(train_ds[text_col]),
        y_true=np.asarray(train_ds[label_col], dtype=np.int64),
    )
    metrics_val = _evaluate(
        "val",
        vectorizer=vectorizer,
        clf=clf,
        texts=list(val_ds[text_col]),
        y_true=np.asarray(val_ds[label_col], dtype=np.int64),
    )
    metrics_test = _evaluate(
        "test",
        vectorizer=vectorizer,
        clf=clf,
        texts=list(test_ds[text_col]),
        y_true=np.asarray(test_ds[label_col], dtype=np.int64),
    )

    print(f"dataset={args.dataset} text_col={text_col} label_col={label_col} labels={len(label2id)}")
    print("train:", {k: round(v, 4) for k, v in metrics_train.items()})
    print("val  :", {k: round(v, 4) for k, v in metrics_val.items()})
    print("test :", {k: round(v, 4) for k, v in metrics_test.items()})

    artifact = {
        "dataset": args.dataset,
        "text_col": text_col,
        "label_col": label_col,
        "label2id": label2id,
        "id2label": id2label,
        "vectorizer": vectorizer,
        "classifier": clf,
    }
    dump(artifact, model_out)
    print(f"saved: {model_out}")
    print(f"saved: {label_map_out}")


if __name__ == "__main__":
    main()

