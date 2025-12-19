# SLU (Spoken Language Understanding)

This folder adds a minimal end-to-end SLU module for the call-support demo:

- Intent classification (BANKING77 by default)
- Simple OOD handling via confidence thresholding
- Optional slot extraction (regex-based)

## Install (once)

If you already installed the root `requirements.txt` for the Streamlit app, you likely still need training deps:

```powershell
pip install -r slu/requirements.txt
```

## Train

Baseline (fast):

```powershell
python -m slu.train_baseline
```

Transformer (real model):

```powershell
python -m slu.train_transformer
```

Pick an OOD threshold on the validation split and write it to `slu/intents_config.json`:

```powershell
python -m slu.eval --write-config
```

## Inference (single entry point)

```powershell
python -c "from slu.infer import run_slu; import json; print(json.dumps(run_slu('I was charged twice yesterday for 49.90'), indent=2))"
```

## Demo utterances

- "I was charged twice for the same thing yesterday"
- "I want to do a chargeback for a transaction on 2025-12-01"
- "My card is gone, I think I lost it on the train"
- "Where is my new card, it hasn’t arrived"
- "I still don't see my refund, it was supposed to be there last week"
- "My card is ending in 1234"
- "Reference ID: AB12C3D4"
- "It was 49.90 CHF at Amazon on 01/12/2025"
- "Can you book me a hotel?" (should often be rejected as `unknown` due to low confidence)

