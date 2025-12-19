from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from datasets import load_dataset
from sacrebleu import corpus_bleu

from eval.common import DEFAULT_RESULTS_DIR, ensure_dir, mt_translate


def main() -> None:
    parser = argparse.ArgumentParser(description="MT evaluation (BLEU) using WMT16.")
    parser.add_argument("--pair", default="de-en", help="WMT16 config, e.g. de-en, ro-en, fr-en")
    parser.add_argument("--src", default="de", help="Source language key in dataset translation dict")
    parser.add_argument("--tgt", default="en", help="Target language key in dataset translation dict")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    ensure_dir(results_dir)

    ds = load_dataset("wmt/wmt16", args.pair, split="test", streaming=True)
    if hasattr(ds, "take"):
        ds = ds.shuffle(seed=args.seed, buffer_size=1000).take(args.n)
    else:
        ds = ds.shuffle(seed=args.seed).select(range(min(args.n, len(ds))))

    hyps: list[str] = []
    refs: list[str] = []
    rows: list[dict] = []

    for i, ex in enumerate(ds):
        tr = ex.get("translation") or {}
        src = tr.get(args.src, "")
        ref = tr.get(args.tgt, "")
        t0 = time.perf_counter()
        hyp = mt_translate(src, args.src, args.tgt)
        dt = time.perf_counter() - t0
        hyps.append(hyp)
        refs.append(ref)
        rows.append({"i": i, "time_sec": round(dt, 6), "src": src, "ref": ref, "hyp": hyp})

    items_csv = results_dir / "mt_items.csv"
    pd.DataFrame(rows).to_csv(items_csv, index=False)

    bleu = float(corpus_bleu(hyps, [refs]).score)
    summary = {
        "pair": args.pair,
        "n": int(len(rows)),
        "bleu": bleu,
        "mean_time_sec": float(pd.DataFrame(rows)["time_sec"].mean()),
    }
    summary_csv = results_dir / "mt_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    (results_dir / "mt_meta.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    print(f"Wrote: {items_csv}")
    print(f"Wrote: {summary_csv}")


if __name__ == "__main__":
    main()
