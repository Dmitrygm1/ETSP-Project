# Call Support Copilot (MWI)

#### Project for Essentials of Speech and Text Processing

## Overview

Upload an audio call + manually enter caller phone number, then get:

- Whisper transcription
- Language detection + translation to English (only when needed)
- Text-based emotion label + confidence
- Client record lookup from a tiny synthetic SQLite DB
- Rule-based suggested actions

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

First run downloads models (Whisper + translation + emotion) and can take a while.

## Demo phone numbers (synthetic DB)

- `+41 79 123 45 67`
- `+31 6 12345678`
- `+49 151 23456789`
- `+33 1 23 45 67 89`

## Notes
- Translation is supported for: `de`, `nl`, `fr`, `es`, `it`, `pt` (Whisper language codes).
- The client DB auto-creates at `data/clients.db` on first run.
