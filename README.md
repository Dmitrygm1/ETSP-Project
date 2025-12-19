# 📞 Call Support Copilot

> **AI-powered call analysis for customer support agents**

*Project for Essentials in Text and Speech Processing*

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Suite](#evaluation-suite)
- [SLU Training](#slu-training)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Demo Data](#demo-data)

---

## Overview

Call Support Copilot is a comprehensive speech processing application that analyzes uploaded customer service call recordings with low latency. Upload an audio recording with a caller's phone number to receive:

- **Automatic Speech Recognition (ASR)** - Whisper-based transcription
- **Language Detection & Translation** - Automatic translation to English when needed
- **Speech Emotion Recognition (SER)** - 7-class emotion detection with VAD dimensions
- **Spoken Language Understanding (SLU)** - Intent classification with slot extraction
- **Client Lookup** - CRM integration via SQLite database
- **Suggested Actions** - Context-aware agent guidance

---

## Features

### 🎤 Speech Recognition
- **Model**: [faster-whisper](https://github.com/guillaumekln/faster-whisper) (base/small)
- **Optimization**: INT8 quantization for CPU inference
- **VAD**: Voice Activity Detection for improved accuracy

### 🌍 Machine Translation
- **Models**: Helsinki-NLP OPUS neural MT
- **Languages**: German, Dutch, French, Spanish, Italian, Portuguese → English
- **Chunking**: Automatic text splitting for long transcripts (450 char max)

### 😊 Emotion Recognition
- **Model**: [MERaLiON-SER-v1](https://huggingface.co/MERaLiON/MERaLiON-SER-v1)
- **Classes**: Neutral, Happy, Sad, Angry, Fearful, Disgusted, Surprised
- **Dimensions**: Valence, Arousal, Dominance (0.0-1.0 scale)
- **Timeline**: 4-second windows with 2-second hop for emotion tracking

### 🧠 Intent Understanding
- **Dataset**: [BANKING77](https://huggingface.co/datasets/PolyAI/banking77) (77 intent classes)
- **Models**: 
  - Transformer: Fine-tuned DistilBERT (recommended)
  - Baseline: TF-IDF + Logistic Regression (fast fallback)
- **OOD Detection**: Confidence thresholding for out-of-domain queries
- **Slot Extraction**: Regex-based entity extraction (amounts, dates, references, etc.)

### 👤 Client Management
- **Database**: SQLite with automatic initialization
- **Records**: Phone number, name, VIP status, case history, account notes

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         STREAMLIT UI (app.py)                       │
│    📱 Phone Input → 🎤 Audio Upload → 🚀 Process Call → 📊 Results  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────────┐
         ▼                       ▼                           ▼
┌─────────────────┐   ┌─────────────────────┐   ┌──────────────────────┐
│  ASR (Whisper)  │   │  SER (MERaLiON)     │   │  SLU (DistilBERT)    │
│  faster-whisper │   │  7-class + VAD      │   │  BANKING77 dataset   │
│  base/small     │   │  Timeline tracking  │   │  77 intent classes   │
└────────┬────────┘   └──────────┬──────────┘   └──────────┬───────────┘
         │                       │                         │
         ▼                       ▼                         ▼
┌─────────────────┐   ┌─────────────────────┐   ┌──────────────────────┐
│  MT (Helsinki)  │   │  Emotion Timeline   │   │  Intent + Slots      │
│  6 lang → EN    │   │  + Overall VAD      │   │  + OOD Detection     │
└─────────────────┘   └─────────────────────┘   └──────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
         ┌─────────────────┐       ┌──────────────────────┐
         │  SQLite DB      │       │  Suggested Actions   │
         │  Client Records │       │  Intent + Rule-based │
         └─────────────────┘       └──────────────────────┘
```

---

## Installation

### Prerequisites
- Python 3.10+
- ffmpeg (for audio format conversion)

### macOS
```bash
# Install ffmpeg
brew install ffmpeg

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Windows (PowerShell)
```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Linux
```bash
# Install ffmpeg
sudo apt install ffmpeg

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

> ⚠️ **Note**: First run downloads models (~1GB total) and may take several minutes.

---

## Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

### Quick SER Test (CLI)

Test Speech Emotion Recognition directly from the command line:

```bash
# Using included sample audio
python ser_cli.py audio-samples/happy.wav
python ser_cli.py audio-samples/neutral.wav

# Or download IEMOCAP samples
curl -L -o happy.wav "https://cdn-media.huggingface.co/speech_samples/IEMOCAP_Ses01F_impro03_F013.wav"
python ser_cli.py happy.wav
```

### Quick SLU Test (CLI)

Test intent classification from the command line:

```bash
python -m slu.infer "I was charged twice for the same transaction yesterday"
```

---

## Evaluation Suite

A comprehensive evaluation framework for benchmarking system components.

### Installation

```bash
pip install -r requirements-eval.txt
```

### Running Evaluations

All evaluations write results to the `results/` directory.

#### ASR Word Error Rate (WER)
Evaluates transcription accuracy on LibriSpeech and optionally L2Arctic (accented speech).

```bash
python -m eval.eval_asr_wer --n 50 --skip-accented
```

| Option | Description |
|--------|-------------|
| `--n` | Samples per dataset (default: 200) |
| `--whisper-model` | Model size: base, small (default: base) |
| `--skip-accented` | Skip L2Arctic dataset if gated/unavailable |

#### Speech Emotion Recognition (SER) F1
Evaluates emotion classification on SUPERB demo (IEMOCAP 4-class).

```bash
python -m eval.eval_ser_f1 --n 50
```

| Option | Description |
|--------|-------------|
| `--n` | Number of samples (default: 200) |
| `--window` | Window length in seconds (default: 4.0) |
| `--hop` | Hop length in seconds (default: 2.0) |

#### Machine Translation BLEU
Evaluates translation quality on WMT16.

```bash
python -m eval.eval_mt_bleu --pair de-en --n 50
```

| Option | Description |
|--------|-------------|
| `--pair` | Language pair: de-en, fr-en, etc. |
| `--n` | Number of samples (default: 200) |

#### System Latency
End-to-end latency benchmark (ASR + SER + MT + DB).

```bash
python -m eval.eval_system_latency --n 10 --skip-accented
```

Outputs: `results/system_latency.csv` (per-sample) and `results/system_latency_summary.csv` (mean/median/p90 + real-time factor).

| Option | Description |
|--------|-------------|
| `--n` | Samples per dataset (default: 30) |
| `--ser-mode` | full or chunked (default: chunked) |

---

## SLU Training

Train intent classification models on BANKING77.

### Baseline Model (Fast)

TF-IDF + Logistic Regression:

```bash
pip install -r slu/requirements.txt
python -m slu.train_baseline
```

Output: `models/slu_baseline.pkl`

### Transformer Model (Recommended)

Fine-tuned DistilBERT with ASR noise augmentation:

```bash
pip install -r slu/requirements.txt
python -m slu.train_transformer
```

Output: `models/slu_transformer/`

### OOD Threshold Tuning

Evaluate and select optimal confidence threshold:

```bash
python -m slu.eval --write-config
```

This updates `slu/intents_config.json` with the chosen threshold.

---

## Project Structure

```
ESTP/
├── app.py                    # Streamlit web application
├── ser.py                    # Speech Emotion Recognition module
├── ser_cli.py                # CLI for SER testing
├── requirements.txt          # Core dependencies
├── requirements-eval.txt     # Evaluation dependencies
│
├── slu/                      # Spoken Language Understanding
│   ├── infer.py              # Intent inference (transformer/baseline)
│   ├── slots.py              # Regex-based slot extraction
│   ├── train_transformer.py  # DistilBERT fine-tuning
│   ├── train_baseline.py     # TF-IDF + LogReg training
│   ├── eval.py               # SLU evaluation & threshold tuning
│   ├── intents_config.json   # Intent actions & OOD threshold
│   ├── label_map.json        # BANKING77 label mappings
│   └── requirements.txt      # SLU training dependencies
│
├── eval/                     # Evaluation suite
│   ├── common.py             # Shared utilities
│   ├── eval_asr_wer.py       # ASR WER evaluation
│   ├── eval_ser_f1.py        # SER F1 evaluation
│   ├── eval_mt_bleu.py       # MT BLEU evaluation
│   └── eval_system_latency.py # End-to-end latency
│
├── audio-samples/            # Sample audio files
│   ├── happy.wav
│   └── neutral.wav
│
├── data/                     # Auto-generated
│   └── clients.db            # SQLite client database
│
├── models/                   # Trained models (after training)
│   ├── slu_transformer/      # Fine-tuned DistilBERT
│   └── slu_baseline.pkl      # TF-IDF + LogReg
│
└── results/                  # Evaluation outputs
    ├── asr_wer_*.csv
    ├── ser_*.csv
    ├── mt_*.csv
    └── system_latency*.csv
```

---

## Technical Details

### Models & Resources

| Component | Model | Size | Source |
|-----------|-------|------|--------|
| ASR | faster-whisper (base) | ~150MB | [HuggingFace](https://huggingface.co/Systran/faster-whisper-base) |
| SER | MERaLiON-SER-v1 | ~500MB | [HuggingFace](https://huggingface.co/MERaLiON/MERaLiON-SER-v1) |
| MT | Helsinki-NLP OPUS | ~300MB each | [HuggingFace](https://huggingface.co/Helsinki-NLP) |
| SLU | DistilBERT (fine-tuned) | ~250MB | Trained locally |

### SLU Intent Classes

The system recognizes 77 banking intents including:
- Card issues: `activate_my_card`, `card_arrival`, `lost_or_stolen_card`, `card_not_working`
- Transactions: `transaction_charged_twice`, `request_refund`, `Refund_not_showing_up`
- Transfers: `pending_transfer`, `failed_transfer`, `cancel_transfer`
- Account: `terminate_account`, `verify_my_identity`, `edit_personal_details`

### Slot Types

| Slot | Pattern Examples | Use Case |
|------|------------------|----------|
| `amount` | `49.90`, `CHF 100`, `€50` | Transaction disputes |
| `date` | `2025-12-01`, `December 1st` | Timeline queries |
| `reference_id` | `AB12C3D4`, `order #12345` | Case lookup |
| `card_last4` | `ending in 1234` | Card identification |
| `merchant` | `at Amazon`, `from PayPal` | Merchant disputes |

### Emotion Classes & VAD

| Emotion | Valence | Arousal | Typical Scenario |
|---------|---------|---------|------------------|
| Happy | High | Medium | Resolution satisfaction |
| Neutral | Medium | Low | Standard inquiry |
| Sad | Low | Low | Service disappointment |
| Angry | Low | High | Dispute escalation |
| Fearful | Low | High | Fraud concerns |
| Disgusted | Low | Medium | Poor experience |
| Surprised | Medium | High | Unexpected charges |

---

## Demo Data

### Test Phone Numbers

| Phone Number | Client | Status |
|--------------|--------|--------|
| `+41 79 123 45 67` | Mila Novak | VIP |
| `+33 1 23 45 67 89` | Hugo Martin | VIP |
| `+31 6 12345678` | Noah de Vries | Standard |
| `+49 151 23456789` | Lea Schneider | Standard |

### Sample Utterances

```
"I was charged twice for the same transaction yesterday"
"My card hasn't arrived yet, it's been two weeks"
"I want to dispute a charge of 49.90 CHF from Amazon"
"Can you check reference ID AB12C3D4?"
"My card ending in 1234 was stolen"
```

---

## License

This project was developed for academic purposes as part of the Essentials of Speech and Text Processing course.

---

## Acknowledgments

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) for efficient ASR
- [MERaLiON](https://huggingface.co/MERaLiON) for the SER model
- [Helsinki-NLP](https://huggingface.co/Helsinki-NLP) for OPUS translation models
- [PolyAI](https://huggingface.co/datasets/PolyAI/banking77) for the BANKING77 dataset
