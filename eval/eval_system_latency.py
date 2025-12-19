from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from datasets import load_dataset

import ser
from eval.common import (
    DEFAULT_RESULTS_DIR,
    asr_transcribe_with_language,
    ensure_16k_mono,
    ensure_dir,
    mt_translate,
    temp_wav_path,
)


def _audio_from_example(example: dict) -> tuple[np.ndarray, int]:
    if "audio" in example and example["audio"] is not None:
        audio = example["audio"]
        if isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
            return np.asarray(audio["array"]), int(audio["sampling_rate"])
    if "file" in example and isinstance(example["file"], str) and example["file"]:
        audio_16k = ser.load_audio_16k_mono(example["file"])
        return audio_16k, 16000
    raise KeyError("Could not find audio in example (expected 'audio' or 'file').")


def _init_clients_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS clients (
                phone_number TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                recent_cases TEXT NOT NULL,
                account_notes TEXT NOT NULL,
                last_contact_date TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO clients
                (phone_number, name, status, recent_cases, account_notes, last_contact_date)
            VALUES
                (?, ?, ?, ?, ?, ?)
            """,
            (
                "+41791234567",
                "Mila Novak",
                "VIP",
                "2025-11-20: Duplicate charge dispute.",
                "Prefers email follow-up.",
                "2025-11-20",
            ),
        )
        conn.commit()


def _db_lookup(db_path: Path, phone: str) -> float:
    t0 = time.perf_counter()
    with sqlite3.connect(db_path) as conn:
        conn.execute("SELECT name FROM clients WHERE phone_number = ?", (phone,)).fetchone()
    return time.perf_counter() - t0


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="System latency/RTF evaluation (ASR + SER + MT + DB).")
    parser.add_argument("--n", type=int, default=30, help="Samples per dataset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--ser-mode", choices=["full", "chunked"], default="chunked")
    parser.add_argument("--window", type=float, default=4.0)
    parser.add_argument("--hop", type=float, default=2.0)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--skip-accented", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    ensure_dir(results_dir)

    db_path = Path(__file__).resolve().parents[1] / "data" / "clients.db"
    _init_clients_db(db_path)

    datasets_to_run: list[tuple[str, Any]] = []
    datasets_to_run.append(("librispeech_test_clean", load_dataset("librispeech_asr", "clean", split="test", streaming=True)))
    datasets_to_run.append(("superb_er_session1", load_dataset("anton-l/superb_demo", "er", split="session1", streaming=True)))

    if not args.skip_accented:
        try:
            ds_l2 = load_dataset("KoelLabs/L2ArcticSpontaneousSplit", streaming=True)
            if hasattr(ds_l2, "keys"):
                split = next((k for k in ds_l2.keys() if "script" in k.lower()), next(iter(ds_l2.keys())))
                datasets_to_run.append((f"l2arctic:{split}", ds_l2[split]))
        except Exception as exc:
            print(f"[warn] Skipping accented dataset (L2Arctic): {exc}")

    rows: list[dict] = []

    for ds_name, ds in datasets_to_run:
        if hasattr(ds, "take"):
            ds = ds.shuffle(seed=args.seed, buffer_size=1000).take(args.n)
        else:
            ds = ds.shuffle(seed=args.seed).select(range(min(args.n, len(ds))))
        for i, ex in enumerate(ds):
            audio, sr = _audio_from_example(ex)
            audio_16k = ensure_16k_mono(audio, sr) if sr != 16000 else np.asarray(audio, dtype=np.float32).reshape(-1)
            dur = audio_16k.size / 16000.0

            with temp_wav_path(audio_16k) as wav_path:
                t0 = time.perf_counter()
                transcript, lang, _p = asr_transcribe_with_language(wav_path, model_size=args.whisper_model)
                t_asr = time.perf_counter() - t0

            t1 = time.perf_counter()
            if args.ser_mode == "full":
                _labels, _probs = ser.predict_ser_proba(audio_16k)
            else:
                ser.predict_emotion_windows(audio_16k, window_s=args.window, hop_s=args.hop)
            t_ser = time.perf_counter() - t1

            t_mt = 0.0
            if lang and lang != "en":
                try:
                    t2 = time.perf_counter()
                    _ = mt_translate(transcript, lang, "en")
                    t_mt = time.perf_counter() - t2
                except Exception:
                    t_mt = 0.0

            t_db = _db_lookup(db_path, "+41791234567")
            t_total = t_asr + t_ser + t_mt + t_db
            rtf_total = (t_total / dur) if dur > 0 else float("nan")

            rows.append(
                {
                    "dataset": ds_name,
                    "utt_id": str(ex.get("id", ex.get("file", f"{ds_name}_{i}"))),
                    "duration_sec": round(dur, 4),
                    "t_asr": round(t_asr, 6),
                    "t_ser": round(t_ser, 6),
                    "t_mt": round(t_mt, 6),
                    "t_db": round(t_db, 6),
                    "t_total": round(t_total, 6),
                    "rtf_total": round(rtf_total, 4) if np.isfinite(rtf_total) else "",
                }
            )

    df = pd.DataFrame(rows)
    items_csv = results_dir / "system_latency.csv"
    df.to_csv(items_csv, index=False)

    def summarize(group: pd.DataFrame) -> dict:
        vals = group["t_total"].to_numpy(dtype=float)
        rtf_vals = group["rtf_total"].replace("", np.nan).astype(float).to_numpy()
        return {
            "dataset": group["dataset"].iloc[0],
            "n": int(group.shape[0]),
            "t_total_mean": float(np.nanmean(vals)),
            "t_total_median": float(np.nanmedian(vals)),
            "t_total_p90": _percentile(vals, 90),
            "rtf_mean": float(np.nanmean(rtf_vals)),
            "rtf_median": float(np.nanmedian(rtf_vals)),
        }

    summary_rows = [summarize(g) for _, g in df.groupby("dataset")]
    summary_csv = results_dir / "system_latency_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    (results_dir / "system_latency_meta.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    print(f"Wrote: {items_csv}")
    print(f"Wrote: {summary_csv}")


if __name__ == "__main__":
    main()
