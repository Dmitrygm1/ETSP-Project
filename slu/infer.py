from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import torch

from slu.slots import extract_slots


_DEFAULT_OOD_THRESHOLD = 0.78


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_label_map(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    payload = _load_json(path)
    label2id_raw = payload.get("label2id") or {}
    id2label_raw = payload.get("id2label") or {}
    label2id = {str(k): int(v) for k, v in dict(label2id_raw).items()}
    id2label = {int(k): str(v) for k, v in dict(id2label_raw).items()}
    return label2id, id2label


def _keyword_rationale(text: str) -> list[str]:
    t = (text or "").lower()
    patterns = [
        (r"\b(charged twice|double charge|duplicate (charge|transaction))\b", "keyword: charged twice/duplicate"),
        (r"\b(chargeback|dispute)\b", "keyword: chargeback/dispute"),
        (r"\brefund\b", "keyword: refund"),
        (r"\b(lost card|stolen card|card stolen)\b", "keyword: lost/stolen card"),
        (r"\b(card (hasn'?t arrived|not arrived)|card arrival)\b", "keyword: card arrival"),
        (r"\b(freeze( my)? card|block( my)? card)\b", "keyword: freeze/block card"),
    ]
    why: list[str] = []
    for rx, msg in patterns:
        if re.search(rx, t):
            why.append(msg)
    return why[:4]


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _id2label_from_model(model) -> dict[int, str]:
    raw = getattr(getattr(model, "config", None), "id2label", None) or {}
    if isinstance(raw, dict):
        return {int(k): str(v) for k, v in raw.items()}
    return {}


def _format_topk(id2label: dict[int, str], probs: list[float], k: int = 3) -> list[str]:
    if not probs:
        return []
    idxs = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:k]
    lines = []
    for i in idxs:
        label = id2label.get(i, str(i))
        lines.append(f"top: {label} p={probs[i]:.2f}")
    return lines


@lru_cache(maxsize=1)
def _load_slu_backend() -> dict[str, Any]:
    root = _project_root()
    cfg_path = root / "slu" / "intents_config.json"
    label_map_path = root / "slu" / "label_map.json"
    transformer_dir = root / "models" / "slu_transformer"
    baseline_path = root / "models" / "slu_baseline.pkl"

    cfg = _load_json(cfg_path)
    threshold = float(cfg.get("ood_threshold", _DEFAULT_OOD_THRESHOLD))

    backend: dict[str, Any] = {
        "threshold": threshold,
        "cfg_path": cfg_path,
        "label_map_path": label_map_path,
        "transformer_dir": transformer_dir,
        "baseline_path": baseline_path,
        "backend": None,
        "load_error": None,
    }

    # Prefer transformer if available.
    if transformer_dir.exists():
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(transformer_dir))
            model = AutoModelForSequenceClassification.from_pretrained(str(transformer_dir))
            dev = _device()
            model.to(dev)
            model.eval()

            label2id, id2label = _load_label_map(label_map_path)
            id2label = _id2label_from_model(model) or id2label

            backend.update(
                {
                    "backend": "transformer",
                    "tokenizer": tokenizer,
                    "model": model,
                    "device": dev,
                    "label2id": label2id,
                    "id2label": id2label,
                }
            )
            return backend
        except Exception as exc:
            backend["load_error"] = f"Failed to load transformer SLU model: {exc}"

    # Fallback to baseline if available.
    if baseline_path.exists():
        try:
            try:
                from joblib import load as joblib_load
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "Baseline SLU model found, but joblib is not installed. "
                    "Install `slu/requirements.txt` to enable baseline inference."
                ) from exc

            artifact = joblib_load(baseline_path)
            vectorizer = artifact["vectorizer"]
            clf = artifact["classifier"]
            id2label = artifact.get("id2label") or {}
            if isinstance(id2label, dict) and id2label and all(isinstance(k, str) for k in id2label.keys()):
                id2label = {int(k): str(v) for k, v in id2label.items()}

            label2id, id2label_file = _load_label_map(label_map_path)
            id2label = id2label or id2label_file

            backend.update(
                {
                    "backend": "baseline",
                    "vectorizer": vectorizer,
                    "classifier": clf,
                    "label2id": label2id,
                    "id2label": id2label,
                }
            )
            return backend
        except Exception as exc:
            backend["load_error"] = f"Failed to load baseline SLU model: {exc}"

    backend["backend"] = None
    if not backend.get("load_error"):
        backend["load_error"] = (
            "No SLU model found. Train a model by running:\n"
            "  python -m slu.train_transformer\n"
            "or\n"
            "  python -m slu.train_baseline\n"
            "The app will still work with basic slot extraction."
        )
    return backend


def _predict_transformer(text: str, assets: dict[str, Any]) -> tuple[str, float, list[str]]:
    tokenizer = assets["tokenizer"]
    model = assets["model"]
    dev: torch.device = assets["device"]
    id2label: dict[int, str] = assets.get("id2label") or {}

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=False,
    )
    enc = {k: v.to(dev) for k, v in enc.items() if torch.is_tensor(v)}
    with torch.inference_mode():
        out = model(**enc)
    probs = torch.softmax(out.logits, dim=-1).squeeze(0).detach().cpu().tolist()

    pred_id = int(max(range(len(probs)), key=lambda i: probs[i]))
    pred_label = id2label.get(pred_id, str(pred_id))
    conf = float(probs[pred_id]) if probs else 0.0
    rationale = _format_topk(id2label, probs, k=3)
    return pred_label, conf, rationale


def _predict_baseline(text: str, assets: dict[str, Any]) -> tuple[str, float, list[str]]:
    vectorizer = assets["vectorizer"]
    clf = assets["classifier"]
    id2label: dict[int, str] = assets.get("id2label") or {}

    x = vectorizer.transform([text])
    probs = clf.predict_proba(x)[0].tolist()
    class_ids = list(getattr(clf, "classes_", range(len(probs))))

    best_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    pred_id = int(class_ids[best_idx])
    pred_label = id2label.get(pred_id, str(pred_id))
    conf = float(probs[best_idx]) if probs else 0.0

    idx2label = {i: id2label.get(int(class_ids[i]), str(class_ids[i])) for i in range(len(probs))}
    rationale = _format_topk(idx2label, probs, k=3)

    # Optional: add feature-based rationale if available.
    try:
        import numpy as np

        coef = getattr(clf, "coef_", None)
        if coef is not None and hasattr(x, "indices") and hasattr(vectorizer, "get_feature_names_out"):
            feat_names = vectorizer.get_feature_names_out()
            row_weights = np.asarray(coef[best_idx]).reshape(-1)
            idxs = x.indices
            vals = x.data
            scores = vals * row_weights[idxs]
            top_local = np.argsort(scores)[::-1]
            added = 0
            for j in top_local[:12]:
                if scores[j] <= 0:
                    continue
                feat = str(feat_names[idxs[j]])
                rationale.append(f"keyword: {feat}")
                added += 1
                if added >= 4:
                    break
    except Exception:
        pass

    return pred_label, conf, rationale


def run_slu(text: str) -> dict[str, object]:
    """Run intent classification (+ OOD thresholding) and optional slot extraction.

    Returns (minimum keys):
      - intent: string
      - confidence: float (0..1)
      - is_ood: bool
      - slots: dict[str,str]
    """

    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return {
            "intent": "unknown",
            "confidence": 0.0,
            "is_ood": True,
            "slots": {},
            "rationale": ["empty input"],
        }

    assets = _load_slu_backend()
    threshold = float(assets.get("threshold", _DEFAULT_OOD_THRESHOLD))

    if assets.get("backend") == "transformer":
        pred_intent, conf, rationale = _predict_transformer(text, assets)
    elif assets.get("backend") == "baseline":
        pred_intent, conf, rationale = _predict_baseline(text, assets)
    else:
        why = ["no SLU model found (run `python -m slu.train_transformer` or `python -m slu.train_baseline`)"]
        why.extend(_keyword_rationale(text))
        slots = extract_slots(text).slots
        return {
            "intent": "unknown",
            "confidence": 0.0,
            "is_ood": True,
            "slots": slots,
            "rationale": why,
        }

    is_ood = bool(conf < threshold)
    intent = "unknown" if is_ood else str(pred_intent)

    why: list[str] = []
    if is_ood:
        why.append(f"low_confidence: {conf:.2f} < {threshold:.2f}")
        why.append(f"best_guess: {pred_intent}")
    why.extend(rationale)
    why.extend(_keyword_rationale(text))

    slot_info = extract_slots(text, intent=intent if intent != "unknown" else None)
    why.extend(slot_info.rationale)

    return {
        "intent": intent,
        "confidence": float(conf),
        "is_ood": is_ood,
        "slots": slot_info.slots,
        "rationale": why[:12],
    }


if __name__ == "__main__":  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser(description="Quick SLU inference runner.")
    p.add_argument("text", nargs="+", help="Text to classify")
    args = p.parse_args()
    joined = " ".join(args.text)
    print(json.dumps(run_slu(joined), indent=2, ensure_ascii=False))
