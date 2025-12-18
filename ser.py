from __future__ import annotations

from functools import lru_cache
from typing import Optional

import librosa
import numpy as np


SER_SAMPLE_RATE = 16000


def load_audio_16k_mono(path: str) -> np.ndarray:
    audio, _sr = librosa.load(path, sr=SER_SAMPLE_RATE, mono=True)
    return audio.astype(np.float32, copy=False)


def _default_device() -> int:
    try:
        import torch

        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1


@lru_cache(maxsize=2)
def _get_classifier(device: int):
    from transformers import pipeline

    return pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", device=device)


def predict_emotion_windows(
    audio_16k: np.ndarray,
    *,
    window_s: float = 4.0,
    hop_s: float = 2.0,
    top_k: int = 4,
) -> tuple[
    list[tuple[float, str, float, dict[str, float]]],
    Optional[str],
    dict[str, float],
]:
    audio_16k = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
    if audio_16k.size == 0:
        return [], None, {}

    classifier = _get_classifier(_default_device())

    window_n = max(1, int(window_s * SER_SAMPLE_RATE))
    hop_n = max(1, int(hop_s * SER_SAMPLE_RATE))

    if audio_16k.size <= window_n:
        starts = [0]
    else:
        starts = list(range(0, audio_16k.size - window_n + 1, hop_n))
        last_start = max(0, audio_16k.size - window_n)
        if starts[-1] != last_start:
            starts.append(last_start)

    timeline: list[tuple[float, str, float, dict[str, float]]] = []
    sums: dict[str, float] = {}

    for start in starts:
        end = min(start + window_n, audio_16k.size)
        chunk = audio_16k[start:end]

        outputs = classifier({"array": chunk, "sampling_rate": SER_SAMPLE_RATE}, top_k=top_k)
        scores = {str(o["label"]): float(o["score"]) for o in outputs}

        top_label, top_score = max(scores.items(), key=lambda kv: kv[1])
        timeline.append((start / SER_SAMPLE_RATE, top_label, top_score, scores))

        for label, score in scores.items():
            sums[label] = sums.get(label, 0.0) + score

    n = float(len(starts))
    aggregated_scores = {label: total / n for label, total in sums.items()}
    aggregated_label = max(aggregated_scores.items(), key=lambda kv: kv[1])[0] if aggregated_scores else None
    return timeline, aggregated_label, aggregated_scores
