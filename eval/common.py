from __future__ import annotations

import os
import re
import tempfile
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "results"
SR_16K = 16000


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_text_for_wer(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ensure_16k_mono(audio: np.ndarray, sr: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        # HF datasets often use (n, channels); some libraries use (channels, n).
        if audio.shape[0] <= 4 and audio.shape[1] > audio.shape[0]:
            audio = audio.mean(axis=0)
        else:
            audio = audio.mean(axis=1)
    audio = audio.reshape(-1)

    if int(sr) != SR_16K:
        wav = torch.from_numpy(audio)
        wav = torchaudio.functional.resample(wav, orig_freq=int(sr), new_freq=SR_16K)
        audio = wav.cpu().numpy()

    return np.asarray(audio, dtype=np.float32).reshape(-1)


def duration_seconds(audio_16k: np.ndarray) -> float:
    n = int(np.asarray(audio_16k).size)
    return n / float(SR_16K) if n > 0 else 0.0


def load_audio_16k_mono(audio_path: str) -> np.ndarray:
    import ser

    return ser.load_audio_16k_mono(audio_path)


def write_wav_16k_mono(audio_16k: np.ndarray, out_path: str) -> None:
    audio_16k = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
    sf.write(out_path, audio_16k, SR_16K, subtype="PCM_16")


@contextmanager
def temp_wav_path(audio_16k: np.ndarray):
    audio_16k = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    write_wav_16k_mono(audio_16k, tmp.name)
    try:
        yield tmp.name
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@lru_cache(maxsize=2)
def _get_whisper(model_size: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device=device, compute_type=compute_type)


def asr_transcribe_with_language(
    audio_path: str,
    *,
    model_size: str = "base",
    device: Optional[str] = None,
) -> tuple[str, str, float]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "int8" if device == "cpu" else "float16"
    model = _get_whisper(model_size, device, compute_type)

    segments, info = model.transcribe(audio_path, vad_filter=True)
    transcript = " ".join([seg.text.strip() for seg in segments if getattr(seg, "text", None)])
    lang = getattr(info, "language", None) or "unknown"
    lang_prob = float(getattr(info, "language_probability", 0.0) or 0.0)
    return transcript.strip(), lang, lang_prob


def asr_transcribe(audio_path: str, *, model_size: str = "base", device: Optional[str] = None) -> str:
    transcript, _lang, _prob = asr_transcribe_with_language(audio_path, model_size=model_size, device=device)
    return transcript


def run_asr(audio_path: str, *, model_size: str = "base") -> str:
    return asr_transcribe(audio_path, model_size=model_size)


TRANSLATION_MODELS = {
    ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
    ("nl", "en"): "Helsinki-NLP/opus-mt-nl-en",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
    ("it", "en"): "Helsinki-NLP/opus-mt-it-en",
    ("pt", "en"): "Helsinki-NLP/opus-mt-pt-en",
}


@lru_cache(maxsize=8)
def _get_translator(model_name: str, device: int):
    from transformers import pipeline

    return pipeline("translation", model=model_name, device=device)


def mt_translate(text: str, src_lang: str, tgt_lang: str = "en") -> str:
    src_lang = (src_lang or "").lower()
    tgt_lang = (tgt_lang or "").lower()
    if src_lang == tgt_lang:
        return text
    model_name = TRANSLATION_MODELS.get((src_lang, tgt_lang))
    if not model_name:
        raise ValueError(f"No translation model configured for {src_lang}->{tgt_lang}.")

    device = 0 if torch.cuda.is_available() else -1
    translator = _get_translator(model_name, device)
    out = translator(text, max_length=512)
    return out[0]["translation_text"]


def run_mt(text: str, src_lang: str, tgt_lang: str = "en") -> str:
    return mt_translate(text, src_lang, tgt_lang)


def ser_predict_proba(audio_path: str):
    import ser

    return ser.predict_ser_proba_path(audio_path)


def ser_predict_proba_from_audio(audio_16k: np.ndarray):
    import ser

    return ser.predict_ser_proba(audio_16k)


def run_ser_proba(audio_path: str):
    return ser_predict_proba(audio_path)


def timed(fn, *args, **kwargs) -> tuple[Any, float]:
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0
