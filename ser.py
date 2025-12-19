from __future__ import annotations

from functools import lru_cache
from typing import Optional

import inspect
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoModelForAudioClassification, AutoProcessor


SER_SAMPLE_RATE = 16000
REPO_ID = "MERaLiON/MERaLiON-SER-v1"
EMO_MAP = ["Neutral", "Happy", "Sad", "Angry", "Fearful", "Disgusted", "Surprised"]


def load_audio_16k_mono(path: str) -> np.ndarray:
    def read_soundfile(p: str) -> tuple[np.ndarray, int]:
        audio, sr = sf.read(p, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        return audio, int(sr)

    try:
        audio, sr = read_soundfile(path)
    except Exception:
        tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_wav.close()
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    path,
                    "-ac",
                    "1",
                    "-ar",
                    str(SER_SAMPLE_RATE),
                    tmp_wav.name,
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            audio, sr = read_soundfile(tmp_wav.name)
        finally:
            try:
                os.unlink(tmp_wav.name)
            except OSError:
                pass

    if sr != SER_SAMPLE_RATE:
        wav = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=SER_SAMPLE_RATE)
        audio = wav.cpu().numpy()

    return np.asarray(audio, dtype=np.float32).reshape(-1)


def _default_device() -> int:
    return 0 if torch.cuda.is_available() else -1


@lru_cache(maxsize=2)
def _get_ser(device: int):
    device_str = "cuda" if device == 0 and torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(REPO_ID, trust_remote_code=True)
    model = AutoModelForAudioClassification.from_pretrained(REPO_ID, trust_remote_code=True)
    model.to(device_str)
    model.eval()

    params = inspect.signature(model.forward).parameters
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    allowed_keys = {
        name
        for name, param in params.items()
        if name != "self" and param.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }

    return {
        "device": device_str,
        "processor": processor,
        "model": model,
        "accepts_kwargs": accepts_kwargs,
        "allowed_keys": allowed_keys,
    }


def _predict_ser(wav_16k: np.ndarray) -> dict[str, object]:
    wav_16k = np.asarray(wav_16k, dtype=np.float32).reshape(-1)
    if wav_16k.size == 0:
        raise ValueError("Empty audio.")

    ser = _get_ser(_default_device())
    processor: AutoProcessor = ser["processor"]
    model: AutoModelForAudioClassification = ser["model"]
    device_str: str = ser["device"]

    inputs = processor(
        wav_16k,
        sampling_rate=SER_SAMPLE_RATE,
        return_tensors="pt",
        return_attention_mask=True,
    )

    inputs = {k: v.to(device_str) for k, v in inputs.items() if torch.is_tensor(v)}
    if not ser["accepts_kwargs"]:
        inputs = {k: v for k, v in inputs.items() if k in ser["allowed_keys"]}

    with torch.inference_mode():
        out = model(**inputs)

    logits = out["logits"] if isinstance(out, dict) else out.logits
    dims = out["dims"] if isinstance(out, dict) else getattr(out, "dims", None)
    if dims is None:
        raise RuntimeError("Model output missing 'dims'.")

    probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
    dims = dims.squeeze(0).detach().cpu().numpy()

    if probs.shape[0] != len(EMO_MAP):
        raise RuntimeError(f"Unexpected logits size: {probs.shape[0]} (expected {len(EMO_MAP)}).")
    if dims.shape[0] != 3:
        raise RuntimeError(f"Unexpected dims size: {dims.shape[0]} (expected 3).")

    idx = int(np.argmax(probs))
    label = EMO_MAP[idx]
    confidence = float(probs[idx])

    valence = float(np.clip(dims[0], 0.0, 1.0))
    arousal = float(np.clip(dims[1], 0.0, 1.0))
    dominance = float(np.clip(dims[2], 0.0, 1.0))

    return {
        "label": label,
        "confidence": confidence,
        "valence": valence,
        "arousal": arousal,
        "dominance": dominance,
        "probs": {EMO_MAP[i]: float(probs[i]) for i in range(len(EMO_MAP))},
    }


def predict_emotion_windows(
    audio_16k: np.ndarray,
    *,
    window_s: float = 4.0,
    hop_s: float = 2.0,
) -> tuple[
    list[tuple[float, str, float, tuple[float, float, float], dict[str, float]]],
    Optional[str],
    dict[str, float],
    dict[str, float],
]:
    audio_16k = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
    if audio_16k.size == 0:
        return [], None, {}, {}

    window_n = max(1, int(window_s * SER_SAMPLE_RATE))
    hop_n = max(1, int(hop_s * SER_SAMPLE_RATE))

    if audio_16k.size <= window_n:
        starts = [0]
    else:
        starts = list(range(0, audio_16k.size - window_n + 1, hop_n))
        last_start = max(0, audio_16k.size - window_n)
        if starts[-1] != last_start:
            starts.append(last_start)

    timeline: list[tuple[float, str, float, tuple[float, float, float], dict[str, float]]] = []
    sums: dict[str, float] = {label: 0.0 for label in EMO_MAP}
    vad_sums = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}

    for start in starts:
        end = min(start + window_n, audio_16k.size)
        chunk = audio_16k[start:end]

        pred = _predict_ser(chunk)
        label = str(pred["label"])
        conf = float(pred["confidence"])
        v = float(pred["valence"])
        a = float(pred["arousal"])
        d = float(pred["dominance"])
        probs = {str(k): float(vv) for k, vv in dict(pred["probs"]).items()}

        timeline.append((start / SER_SAMPLE_RATE, label, conf, (v, a, d), probs))

        for lbl, score in probs.items():
            sums[lbl] = sums.get(lbl, 0.0) + score
        vad_sums["valence"] += v
        vad_sums["arousal"] += a
        vad_sums["dominance"] += d

    n = float(len(starts))
    aggregated_scores = {label: total / n for label, total in sums.items()}
    aggregated_label = max(aggregated_scores.items(), key=lambda kv: kv[1])[0] if aggregated_scores else None
    aggregated_vad = {k: v / n for k, v in vad_sums.items()}
    return timeline, aggregated_label, aggregated_scores, aggregated_vad


def predict_ser(audio_16k: np.ndarray) -> dict[str, object]:
    return _predict_ser(audio_16k)


def predict_ser_path(audio_path: str) -> dict[str, object]:
    return _predict_ser(load_audio_16k_mono(audio_path))


def predict_ser_proba(audio_16k: np.ndarray) -> tuple[list[str], np.ndarray]:
    labels, probs, _vad = predict_ser_proba_with_vad(audio_16k)
    return labels, probs


def predict_ser_proba_with_vad(audio_16k: np.ndarray) -> tuple[list[str], np.ndarray, dict[str, float]]:
    pred = _predict_ser(audio_16k)
    probs = pred["probs"]
    labels = list(EMO_MAP)
    prob_array = np.asarray([float(probs[lbl]) for lbl in labels], dtype=np.float32)
    vad = {
        "valence": float(pred["valence"]),
        "arousal": float(pred["arousal"]),
        "dominance": float(pred["dominance"]),
    }
    return labels, prob_array, vad


def predict_ser_proba_path(audio_path: str) -> tuple[list[str], np.ndarray]:
    return predict_ser_proba(load_audio_16k_mono(audio_path))


def predict_ser_proba_path_with_vad(audio_path: str) -> tuple[list[str], np.ndarray, dict[str, float]]:
    return predict_ser_proba_with_vad(load_audio_16k_mono(audio_path))
