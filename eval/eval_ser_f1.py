from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import f1_score

import ser
from eval.common import DEFAULT_RESULTS_DIR, ensure_16k_mono, ensure_dir


def _audio_from_example(example: dict) -> tuple[np.ndarray, int]:
    if "audio" in example and example["audio"] is not None:
        audio = example["audio"]
        if isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
            return np.asarray(audio["array"]), int(audio["sampling_rate"])
    if "file" in example and isinstance(example["file"], str) and example["file"]:
        audio_16k = ser.load_audio_16k_mono(example["file"])
        return audio_16k, 16000
    raise KeyError("Could not find audio in example (expected 'audio' or 'file').")


def _label_name(ds, example: dict) -> str:
    if "label" not in example:
        raise KeyError("Missing 'label' field.")
    lab = example["label"]
    if isinstance(lab, int):
        feat = ds.features.get("label")
        if hasattr(feat, "names") and feat.names:
            return str(feat.names[lab])
        return str(lab)
    return str(lab)


def _map_to_monalion_4(label: str) -> str:
    key = label.strip().lower()
    mapping = {
        "neutral": "Neutral",
        "neu": "Neutral",
        "happy": "Happy",
        "hap": "Happy",
        "angry": "Angry",
        "ang": "Angry",
        "sad": "Sad",
    }
    if key in mapping:
        return mapping[key]
    return label.strip().title()


def main() -> None:
    parser = argparse.ArgumentParser(description="SER evaluation (macro-F1, full vs chunked, RTF).")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window", type=float, default=4.0)
    parser.add_argument("--hop", type=float, default=2.0)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    ensure_dir(results_dir)

    ds = load_dataset("anton-l/superb_demo", "er", split="session1", streaming=True)
    if hasattr(ds, "take"):
        ds = ds.shuffle(seed=args.seed, buffer_size=1000).take(args.n)
    else:
        ds = ds.shuffle(seed=args.seed).select(range(min(args.n, len(ds))))

    items: list[dict] = []
    y_true: list[str] = []
    y_pred_full: list[str] = []
    y_pred_chunked: list[str] = []

    target_labels = ["Neutral", "Happy", "Sad", "Angry"]
    negative = {"Angry", "Sad"}
    non_negative = {"Neutral", "Happy"}

    for i, ex in enumerate(ds):
        audio, sr = _audio_from_example(ex)
        audio_16k = ensure_16k_mono(audio, sr) if sr != 16000 else np.asarray(audio, dtype=np.float32).reshape(-1)
        dur = audio_16k.size / 16000.0

        true_label = _map_to_monalion_4(_label_name(ds, ex))

        t0 = time.perf_counter()
        labels, probs, vad = ser.predict_ser_proba_with_vad(audio_16k)
        t_full = time.perf_counter() - t0
        pred_full = labels[int(np.argmax(probs))]

        t1 = time.perf_counter()
        _timeline, pred_chunk, scores_chunk, _vad_chunk = ser.predict_emotion_windows(
            audio_16k, window_s=args.window, hop_s=args.hop
        )
        t_chunk = time.perf_counter() - t1

        y_true.append(true_label)
        y_pred_full.append(pred_full)
        y_pred_chunked.append(pred_chunk or "")

        items.append(
            {
                "utt_id": str(ex.get("file", f"superb_er_{i}")),
                "duration_sec": round(dur, 4),
                "true": true_label,
                "pred_full": pred_full,
                "pred_chunked": pred_chunk,
                "time_full_sec": round(t_full, 6),
                "rtf_full": round(t_full / dur, 4) if dur > 0 else "",
                "time_chunked_sec": round(t_chunk, 6),
                "rtf_chunked": round(t_chunk / dur, 4) if dur > 0 else "",
                "vad_valence": round(float(vad.get("valence", 0.0)), 4),
                "vad_arousal": round(float(vad.get("arousal", 0.0)), 4),
                "vad_dominance": round(float(vad.get("dominance", 0.0)), 4),
            }
        )

    df = pd.DataFrame(items)
    items_csv = results_dir / "ser_items.csv"
    df.to_csv(items_csv, index=False)

    macro_f1_full = f1_score(y_true, y_pred_full, average="macro", labels=target_labels, zero_division=0)
    macro_f1_chunked = f1_score(y_true, y_pred_chunked, average="macro", labels=target_labels, zero_division=0)

    df_neg = df[df["true"].isin(list(negative))]
    df_nonneg = df[df["true"].isin(list(non_negative))]

    def subset_f1(sub: pd.DataFrame, mode: str) -> float:
        if sub.empty:
            return float("nan")
        return float(
            f1_score(
                sub["true"].tolist(),
                sub[mode].tolist(),
                average="macro",
                labels=sorted(sub["true"].unique().tolist()),
                zero_division=0,
            )
        )

    f1_neg_full = subset_f1(df_neg, "pred_full")
    f1_nonneg_full = subset_f1(df_nonneg, "pred_full")
    f1_neg_chunked = subset_f1(df_neg, "pred_chunked")
    f1_nonneg_chunked = subset_f1(df_nonneg, "pred_chunked")

    summary = {
        "n": int(df.shape[0]),
        "macro_f1_full": float(macro_f1_full),
        "macro_f1_chunked": float(macro_f1_chunked),
        "mean_rtf_full": float(df["rtf_full"].replace("", np.nan).astype(float).mean()),
        "mean_rtf_chunked": float(df["rtf_chunked"].replace("", np.nan).astype(float).mean()),
        "f1_negative_full": f1_neg_full,
        "f1_non_negative_full": f1_nonneg_full,
        "delta_f1_full_nonneg_minus_neg": float(f1_nonneg_full - f1_neg_full),
        "f1_negative_chunked": f1_neg_chunked,
        "f1_non_negative_chunked": f1_nonneg_chunked,
        "delta_f1_chunked_nonneg_minus_neg": float(f1_nonneg_chunked - f1_neg_chunked),
    }

    summary_csv = results_dir / "ser_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    (results_dir / "ser_meta.json").write_text(
        json.dumps({"window": args.window, "hop": args.hop, "seed": args.seed, "n": args.n}, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote: {items_csv}")
    print(f"Wrote: {summary_csv}")


if __name__ == "__main__":
    main()
