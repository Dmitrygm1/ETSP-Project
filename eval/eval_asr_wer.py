from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from datasets import load_dataset
from jiwer import wer

from eval.common import (
    DEFAULT_RESULTS_DIR,
    asr_transcribe,
    ensure_16k_mono,
    ensure_dir,
    normalize_text_for_wer,
    temp_wav_path,
)


def _audio_from_example(example: dict) -> tuple[np.ndarray, int]:
    if "audio" in example and example["audio"] is not None:
        audio = example["audio"]
        if isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
            return np.asarray(audio["array"]), int(audio["sampling_rate"])
    for key in ("file", "path", "audio_filepath", "wav"):
        if key in example and isinstance(example[key], str) and example[key]:
            import ser

            audio_16k = ser.load_audio_16k_mono(example[key])
            return audio_16k, 16000
    raise KeyError("Could not find audio in example (expected 'audio' or a file path key).")


def _ref_text_from_example(example: dict) -> str:
    for key in ("text", "sentence", "transcript", "transcription", "normalized_text", "prompt"):
        if key in example and isinstance(example[key], str) and example[key].strip():
            return example[key]
    raise KeyError("Could not find reference text in example (expected 'text'/'sentence'/...).")


def _utterance_id(example: dict, fallback: str) -> str:
    for key in ("id", "utt_id", "utterance_id", "uid", "file", "path"):
        if key in example and example[key] is not None:
            return str(example[key])
    return fallback


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser(description="ASR WER evaluation + disparity (accented vs clean).")
    parser.add_argument("--n", type=int, default=200, help="Samples per dataset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--whisper-model", default="base", help="faster-whisper model size (base/small/...)")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument(
        "--skip-accented",
        action="store_true",
        help="Skip accented dataset (useful if gated/unavailable).",
    )
    parser.add_argument("--accented-dataset", default="KoelLabs/L2ArcticSpontaneousSplit")
    parser.add_argument("--accented-split", default="", help="Optional split name for accented dataset")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    ensure_dir(results_dir)

    items: list[dict] = []

    def eval_dataset(name: str, ds) -> None:
        nonlocal items
        if hasattr(ds, "take"):
            ds = ds.shuffle(seed=args.seed, buffer_size=1000).take(args.n)
        else:
            ds = ds.shuffle(seed=args.seed).select(range(min(args.n, len(ds))))
        for i, ex in enumerate(ds):
            audio, sr = _audio_from_example(ex)
            ref = _ref_text_from_example(ex)
            audio_16k = ensure_16k_mono(audio, sr)
            with temp_wav_path(audio_16k) as wav_path:
                hyp = asr_transcribe(wav_path, model_size=args.whisper_model)
            ref_n = normalize_text_for_wer(ref)
            hyp_n = normalize_text_for_wer(hyp)
            w = float(wer(ref_n, hyp_n))
            items.append(
                {
                    "dataset": name,
                    "utt_id": _utterance_id(ex, f"{name}_{i}"),
                    "duration_sec": round(audio_16k.size / 16000.0, 4),
                    "ref": ref,
                    "hyp": hyp,
                    "wer": w,
                }
            )

    ds_clean = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
    eval_dataset("librispeech_test_clean", ds_clean)

    accented_loaded = False
    if not args.skip_accented:
        try:
            ds_l2 = load_dataset(args.accented_dataset, streaming=True)
            if hasattr(ds_l2, "keys"):
                split = args.accented_split.strip()
                if not split:
                    split = next((k for k in ds_l2.keys() if "script" in k.lower()), next(iter(ds_l2.keys())))
                ds_l2 = ds_l2[split]
                eval_dataset(f"{args.accented_dataset}:{split}", ds_l2)
                accented_loaded = True
            else:
                eval_dataset(args.accented_dataset, ds_l2)
                accented_loaded = True
        except Exception as exc:
            print(f"[warn] Could not load accented dataset ({args.accented_dataset}): {exc}")
            print("[warn] Re-run with --skip-accented or provide an accessible dataset/split.")

    df = pd.DataFrame(items)
    items_csv = results_dir / "asr_wer_items.csv"
    df.to_csv(items_csv, index=False)

    summary_rows: list[dict] = []
    for dataset_name, g in df.groupby("dataset"):
        values = g["wer"].to_numpy(dtype=float)
        row = {
            "dataset": dataset_name,
            "n": int(values.size),
            "wer_mean": float(values.mean()) if values.size else float("nan"),
        }
        if values.size >= 2:
            row["wer_ci95_lo"], row["wer_ci95_hi"] = _bootstrap_mean_ci(values, n_boot=500, seed=args.seed)
        summary_rows.append(row)

    if accented_loaded:
        clean_mean = df[df["dataset"] == "librispeech_test_clean"]["wer"].mean()
        acc_mean = df[df["dataset"] != "librispeech_test_clean"]["wer"].mean()
        summary_rows.append(
            {
                "dataset": "delta",
                "n": int(df.shape[0]),
                "wer_mean": float(acc_mean - clean_mean),
                "wer_ci95_lo": "",
                "wer_ci95_hi": "",
            }
        )

    summary_csv = results_dir / "asr_wer_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    meta = {
        "whisper_model": args.whisper_model,
        "n_per_dataset": args.n,
        "seed": args.seed,
        "accented_dataset": args.accented_dataset,
        "accented_split": args.accented_split,
    }
    (results_dir / "asr_wer_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote: {items_csv}")
    print(f"Wrote: {summary_csv}")


if __name__ == "__main__":
    main()
