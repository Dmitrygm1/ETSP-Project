# Call Support Copilot

#### Project for Essentials of Speech and Text Processing

## Overview

Upload an audio call + manually enter caller phone number, then get:

- Whisper transcription
- Language detection + translation to English (only when needed)
- Speech Emotion Recognition (SER): 7-class emotion + Valence/Arousal/Dominance (overall + timeline)
- Spoken Language Understanding (SLU): intent + OOD/unknown + regex slots
- Client record lookup from a tiny synthetic SQLite DB
- Suggested actions (intent-driven + rule-based)

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

First run downloads models (Whisper + translation + SER) and can take a while.

## Evals

Install eval dependencies:

```powershell
pip install -r requirements-eval.txt
```

Run scripts (writes CSVs into `results/`):

```powershell
python -m eval.eval_asr_wer --n 50 --skip-accented
python -m eval.eval_ser_f1 --n 50
python -m eval.eval_system_latency --n 10 --skip-accented
python -m eval.eval_mt_bleu --pair de-en --n 50
```

Notes:
- The accented ASR dataset (`KoelLabs/L2ArcticSpontaneousSplit`) used in `eval_asr_wer` and `eval_system_latency` may be gated; use `--skip-accented` with those scripts if it fails.
- SER eval uses SUPERB demo (IEMOCAP 4-class) and reports macro-F1 on `{Neutral, Happy, Sad, Angry}`.

## SLU (optional training)

See `slu/README.md` for training + evaluation scripts (baseline + transformer).

## Quick SER test (optional)

```powershell
curl.exe -L -o happy.wav "https://cdn-media.huggingface.co/speech_samples/IEMOCAP_Ses01F_impro03_F013.wav"
curl.exe -L -o neutral.wav "https://cdn-media.huggingface.co/speech_samples/IEMOCAP_Ses01F_impro04_F000.wav"
python ser_cli.py happy.wav
python ser_cli.py neutral.wav
```

## Demo phone numbers (synthetic DB)

- `+41 79 123 45 67`
- `+31 6 12345678`
- `+49 151 23456789`
- `+33 1 23 45 67 89`

## Notes
- Translation is supported for: `de`, `nl`, `fr`, `es`, `it`, `pt` (Whisper language codes).
- SER uses `MERaLiON/MERaLiON-SER-v1` on 4s windows with 2s hop.
- The client DB auto-creates at `data/clients.db` on first run.
