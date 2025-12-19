from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from joblib import load as joblib_load
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=-1, keepdims=True)


def _predict_transformer(
    texts: list[str],
    *,
    model_dir: Path,
    batch_size: int,
    max_length: int,
) -> tuple[np.ndarray, dict[int, str]]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    id2label: dict[int, str] = {int(k): str(v) for k, v in dict(model.config.id2label or {}).items()}

    probs_all: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        enc = {k: v.to(device) for k, v in enc.items() if torch.is_tensor(v)}
        with torch.inference_mode():
            out = model(**enc)
        logits = out.logits.detach().cpu().numpy()
        probs_all.append(_softmax(logits))
    return np.vstack(probs_all), id2label


def _predict_baseline(texts: list[str], *, baseline_path: Path) -> tuple[np.ndarray, dict[int, str]]:
    artifact = joblib_load(baseline_path)
    vectorizer = artifact["vectorizer"]
    clf = artifact["classifier"]
    x = vectorizer.transform(texts)
    probs = clf.predict_proba(x)

    id2label: dict[int, str] = {}
    raw = artifact.get("id2label") or {}
    if isinstance(raw, dict) and raw and all(isinstance(k, int) for k in raw.keys()):
        id2label = {int(k): str(v) for k, v in raw.items()}
    elif isinstance(raw, dict):
        id2label = {int(k): str(v) for k, v in raw.items()}

    # Align proba columns to label ids via classes_
    classes = np.asarray(getattr(clf, "classes_", np.arange(probs.shape[1])), dtype=np.int64)
    probs_aligned = np.zeros_like(probs)
    for j, cls_id in enumerate(classes):
        if int(cls_id) < probs.shape[1]:
            probs_aligned[:, int(cls_id)] = probs[:, j]

    if not id2label:
        id2label = {i: str(i) for i in range(probs_aligned.shape[1])}
    return probs_aligned, id2label


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def _threshold_sweep(
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: list[float],
) -> list[dict[str, float]]:
    p_max = np.max(probs, axis=1)
    y_pred = np.argmax(probs, axis=1)

    rows: list[dict[str, float]] = []
    for t in thresholds:
        accepted = p_max >= t
        coverage = float(np.mean(accepted)) if accepted.size else 0.0
        if np.any(accepted):
            acc_acc = float(accuracy_score(y_true[accepted], y_pred[accepted]))
            f1_acc = float(f1_score(y_true[accepted], y_pred[accepted], average="macro"))
        else:
            acc_acc = 0.0
            f1_acc = 0.0
        rows.append(
            {
                "threshold": float(t),
                "coverage": coverage,
                "acc_accepted": acc_acc,
                "f1_macro_accepted": f1_acc,
                "utility": float(acc_acc * coverage),
            }
        )
    return rows


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_config(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SLU model and pick an OOD threshold on validation set.")
    parser.add_argument("--dataset", default="banking77")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--model", choices=["transformer", "baseline"], default="transformer")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--thresholds", default="0.50,0.60,0.70,0.78,0.80,0.85,0.90")
    parser.add_argument("--write-config", action="store_true", help="Write chosen threshold to slu/intents_config.json")
    args = parser.parse_args()

    root = _project_root()
    cfg_path = root / "slu" / "intents_config.json"

    models_dir = root / "models"
    transformer_dir = models_dir / "slu_transformer"
    baseline_path = models_dir / "slu_baseline.pkl"

    ds = load_dataset(args.dataset)
    train_ref = ds["train"] if "train" in ds else ds[next(iter(ds.keys()))]
    text_col = _guess_text_column(train_ref.features)
    label_col = _guess_label_column(train_ref.features)
    _train_ds, val_ds, test_ds = _make_splits(
        ds, label_col=label_col, seed=args.seed, val_ratio=args.val_ratio, test_ratio=args.test_ratio
    )

    val_texts = list(val_ds[text_col])
    val_labels = np.asarray(val_ds[label_col], dtype=np.int64)
    test_texts = list(test_ds[text_col])
    test_labels = np.asarray(test_ds[label_col], dtype=np.int64)

    if args.model == "transformer":
        if not transformer_dir.exists():
            raise SystemExit(f"Missing transformer model dir: {transformer_dir} (run slu/train_transformer.py)")
        probs_val, _id2label = _predict_transformer(
            val_texts,
            model_dir=transformer_dir,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        probs_test, _ = _predict_transformer(
            test_texts,
            model_dir=transformer_dir,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
    else:
        if not baseline_path.exists():
            raise SystemExit(f"Missing baseline model: {baseline_path} (run slu/train_baseline.py)")
        probs_val, _id2label = _predict_baseline(val_texts, baseline_path=baseline_path)
        probs_test, _ = _predict_baseline(test_texts, baseline_path=baseline_path)

    pred_val = np.argmax(probs_val, axis=1)
    pred_test = np.argmax(probs_test, axis=1)
    print(f"val (no threshold):  {_metrics(val_labels, pred_val)}")
    print(f"test(no threshold):  {_metrics(test_labels, pred_test)}")

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    sweep = _threshold_sweep(val_labels, probs_val, thresholds)
    sweep = sorted(sweep, key=lambda r: (r["utility"], r["acc_accepted"]), reverse=True)

    print("\nthreshold sweep (sorted):")
    for row in sweep:
        print(
            f"  T={row['threshold']:.2f}  cov={row['coverage']:.3f}  "
            f"acc@cov={row['acc_accepted']:.3f}  f1@cov={row['f1_macro_accepted']:.3f}  util={row['utility']:.3f}"
        )

    best = sweep[0] if sweep else {"threshold": 0.78}
    best_t = float(best["threshold"])
    print(f"\nchosen ood_threshold={best_t:.2f}")

    # Report test coverage/acc at chosen threshold
    pmax_test = np.max(probs_test, axis=1)
    accepted = pmax_test >= best_t
    cov_test = float(np.mean(accepted)) if accepted.size else 0.0
    if np.any(accepted):
        acc_test = float(accuracy_score(test_labels[accepted], pred_test[accepted]))
        f1_test = float(f1_score(test_labels[accepted], pred_test[accepted], average="macro"))
    else:
        acc_test = 0.0
        f1_test = 0.0
    print(f"test @T={best_t:.2f}: coverage={cov_test:.3f} acc_accepted={acc_test:.3f} f1_macro_accepted={f1_test:.3f}")

    if args.write_config:
        cfg = _load_config(cfg_path)
        cfg["ood_threshold"] = best_t
        _save_config(cfg_path, cfg)
        print(f"updated: {cfg_path}")


if __name__ == "__main__":
    main()

