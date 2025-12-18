import os
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import streamlit as st


AGENT_LANGUAGE = "en"
DEFAULT_WHISPER_MODEL = "base"

TRANSLATION_MODELS = {
    "de": "Helsinki-NLP/opus-mt-de-en",
    "nl": "Helsinki-NLP/opus-mt-nl-en",
    "fr": "Helsinki-NLP/opus-mt-fr-en",
    "es": "Helsinki-NLP/opus-mt-es-en",
    "it": "Helsinki-NLP/opus-mt-it-en",
    "pt": "Helsinki-NLP/opus-mt-pt-en",
}


@st.cache_resource(show_spinner=False)
def get_whisper_model(model_size: str):
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device="cpu", compute_type="int8")


@st.cache_resource(show_spinner=False)
def get_translator(model_name: str):
    from transformers import pipeline

    return pipeline("translation", model=model_name)


@st.cache_resource(show_spinner=False)
def get_emotion_classifier():
    from transformers import pipeline

    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
    )


@dataclass(frozen=True)
class ClientRecord:
    phone_number: str
    name: str
    status: str
    recent_cases: str
    account_notes: str
    last_contact_date: str


SEED_CLIENTS: list[ClientRecord] = [
    ClientRecord(
        phone_number="+41791234567",
        name="Mila Novak",
        status="VIP",
        recent_cases="2025-11-20: Duplicate charge dispute (card present).",
        account_notes="Prefers email follow-up. Often travels; time zone CET.",
        last_contact_date="2025-11-20",
    ),
    ClientRecord(
        phone_number="+31612345678",
        name="Noah de Vries",
        status="Standard",
        recent_cases="2025-10-02: New debit card delivery status.",
        account_notes="Requested SMS-only contact. Mild hearing impairment noted.",
        last_contact_date="2025-10-02",
    ),
    ClientRecord(
        phone_number="+4915123456789",
        name="Lea Schneider",
        status="Standard",
        recent_cases="2025-09-12: Refund timeline question (merchant dispute).",
        account_notes="Calm caller; prefers concise explanations.",
        last_contact_date="2025-09-12",
    ),
    ClientRecord(
        phone_number="+33123456789",
        name="Hugo Martin",
        status="VIP",
        recent_cases="2025-08-18: Card lost while abroad; replacement issued.",
        account_notes="Priority routing. Verify security questions carefully.",
        last_contact_date="2025-08-18",
    ),
]


def canonicalize_phone_number(phone_number: str) -> str:
    digits = re.sub(r"\D", "", phone_number or "")
    return f"+{digits}" if digits else ""


def get_db_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "clients.db"


def ensure_client_db(db_path: Path) -> None:
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
        existing = conn.execute("SELECT COUNT(1) FROM clients").fetchone()
        if existing and existing[0] > 0:
            return
        conn.executemany(
            """
            INSERT OR REPLACE INTO clients
                (phone_number, name, status, recent_cases, account_notes, last_contact_date)
            VALUES
                (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    c.phone_number,
                    c.name,
                    c.status,
                    c.recent_cases,
                    c.account_notes,
                    c.last_contact_date,
                )
                for c in SEED_CLIENTS
            ],
        )
        conn.commit()


def lookup_client(db_path: Path, phone_number: str) -> Optional[ClientRecord]:
    phone_number = canonicalize_phone_number(phone_number)
    if not phone_number:
        return None
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT phone_number, name, status, recent_cases, account_notes, last_contact_date
            FROM clients
            WHERE phone_number = ?
            """,
            (phone_number,),
        ).fetchone()
    if not row:
        return None
    return ClientRecord(*row)


def write_uploaded_audio_to_temp(uploaded_bytes: bytes, suffix: str) -> str:
    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_bytes)
        return tmp.name


def split_text(text: str, max_chars: int = 450) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        if not sentence:
            continue
        if current_len + len(sentence) + (1 if current else 0) <= max_chars:
            current.append(sentence)
            current_len += len(sentence) + (1 if current_len else 0)
            continue
        if current:
            chunks.append(" ".join(current))
        if len(sentence) <= max_chars:
            current = [sentence]
            current_len = len(sentence)
            continue
        for i in range(0, len(sentence), max_chars):
            chunks.append(sentence[i : i + max_chars])
        current = []
        current_len = 0
    if current:
        chunks.append(" ".join(current))
    return chunks


def translate_to_english(text: str, src_lang: str) -> Optional[str]:
    model_name = TRANSLATION_MODELS.get(src_lang)
    if not model_name:
        return None
    translator = get_translator(model_name)
    parts = []
    for chunk in split_text(text):
        out = translator(chunk, max_length=512)
        parts.append(out[0]["translation_text"])
    return " ".join(parts).strip()


def predict_emotion(text: str) -> Optional[tuple[str, float]]:
    text = text.strip()
    if not text:
        return None
    classifier = get_emotion_classifier()
    result = classifier(text, truncation=True, max_length=512)
    # Handle various transformers return shapes across versions.
    if isinstance(result, list) and result and isinstance(result[0], list):
        candidates = result[0]
    elif isinstance(result, list):
        candidates = result
    elif isinstance(result, dict):
        candidates = [result]
    else:
        return None
    best = max(candidates, key=lambda d: float(d.get("score", 0.0)))
    return str(best.get("label", "unknown")), float(best.get("score", 0.0))


def generate_suggested_actions(transcript: str, emotion_label: Optional[str]) -> list[str]:
    t = transcript.lower()
    actions: list[str] = []
    if emotion_label and emotion_label.lower() in {"anger", "annoyance", "disgust"}:
        actions.append("Caller upset: acknowledge feelings and de-escalate; use calm tone.")
    if emotion_label and emotion_label.lower() in {"fear", "sadness"}:
        actions.append("Caller distressed: reassure and explain next steps clearly.")
    if re.search(r"\b(charged twice|double charge|duplicate (charge|transaction))\b", t):
        actions.append("Check duplicate transactions and start a dispute/refund if applicable.")
    if re.search(r"\b(refund|chargeback|dispute)\b", t):
        actions.append("Explain refund/dispute timeline; confirm merchant and transaction details.")
    if re.search(r"\b(lost card|stolen card|card stolen)\b", t):
        actions.append("Freeze the card immediately; order replacement; review recent transactions.")
    if re.search(r"\b(cancel|close account|terminate)\b", t):
        actions.append("Follow retention flow: ask reason, offer alternatives, confirm cancellation steps.")
    if not actions:
        actions.append("Clarify the issue and confirm caller identity; proceed with standard support flow.")
    return actions


def main() -> None:
    st.set_page_config(page_title="Call Support Copilot", layout="wide")
    st.title("Call Support Copilot (MWI)")
    st.caption("Demo: upload an audio call + caller number -> transcript, translation, emotion, client lookup.")

    db_path = get_db_path()
    ensure_client_db(db_path)

    with st.sidebar:
        st.subheader("Demo Scenario")
        st.write("Bank support (synthetic clients)")
        agent_language = st.selectbox("Agent language", options=["en"], index=0, disabled=True)
        whisper_model_size = st.selectbox(
            "Whisper model",
            options=["base", "small"],
            index=0,
            help="Use 'small' for better accuracy (slower + larger download).",
        )

    col_left, col_right = st.columns([2, 1], gap="large")

    with col_left:
        phone_input = st.text_input("Caller phone number", placeholder="+41 79 123 45 67")
        audio_file = st.file_uploader(
            "Upload call audio (WAV/MP3/M4A/FLAC/OGG)",
            type=["wav", "mp3", "m4a", "flac", "ogg"],
        )
        if audio_file is not None:
            st.audio(audio_file.getvalue())

        run = st.button("Process call", type="primary", disabled=(audio_file is None or not phone_input.strip()))

    if run and audio_file is not None and phone_input.strip():
        caller_phone = canonicalize_phone_number(phone_input)
        client = lookup_client(db_path, caller_phone)

        audio_bytes = audio_file.getvalue()
        suffix = Path(audio_file.name).suffix or ".wav"
        tmp_path = write_uploaded_audio_to_temp(audio_bytes, suffix=suffix)

        try:
            with st.spinner("Transcribing audio (Whisper)..."):
                whisper = get_whisper_model(whisper_model_size)
                segments, info = whisper.transcribe(tmp_path, vad_filter=True)
                transcript_parts: list[str] = []
                for seg in segments:
                    transcript_parts.append(seg.text.strip())
                transcript = " ".join([p for p in transcript_parts if p]).strip()
                detected_lang = getattr(info, "language", None) or "unknown"
                lang_prob = float(getattr(info, "language_probability", 0.0) or 0.0)

            translated = None
            if transcript and detected_lang != agent_language:
                with st.spinner("Translating to English..."):
                    translated = translate_to_english(transcript, detected_lang)

            text_for_emotion = translated if translated else transcript
            emotion = None
            if text_for_emotion:
                with st.spinner("Detecting emotion..."):
                    emotion = predict_emotion(text_for_emotion)

            actions = generate_suggested_actions(text_for_emotion or transcript, emotion[0] if emotion else None)

            st.session_state["call_result"] = {
                "caller_phone": caller_phone,
                "client": client,
                "transcript": transcript,
                "detected_lang": detected_lang,
                "lang_prob": lang_prob,
                "translated": translated,
                "emotion": emotion,
                "actions": actions,
            }
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    result = st.session_state.get("call_result")

    with col_left:
        if result:
            st.subheader("Transcription")
            st.write(f"Detected language: `{result['detected_lang']}` (p={result['lang_prob']:.2f})")
            st.text_area("Transcript", value=result["transcript"], height=220)
            if result["detected_lang"] != agent_language:
                if result["translated"]:
                    st.subheader("Translation (English)")
                    st.text_area("Translated transcript", value=result["translated"], height=220)
                else:
                    st.warning(
                        f"Translation not available for `{result['detected_lang']}`. "
                        "Try English, Dutch (nl), German (de), French (fr), Spanish (es), Italian (it), or Portuguese (pt)."
                    )

    with col_right:
        if not result:
            st.subheader("Outputs")
            st.write("Upload audio and click **Process call**.")
            return

        st.subheader("Emotion")
        emotion = result.get("emotion")
        if emotion:
            st.write(f"**{emotion[0]}** (confidence {emotion[1]:.2f})")
        else:
            st.write("No emotion result (empty transcript).")

        st.subheader("Client record")
        client = result.get("client")
        if client:
            st.write(f"**{client.name}** ({client.status})")
            st.write(f"Phone: `{client.phone_number}`")
            st.write(f"Last contact: `{client.last_contact_date}`")
            st.write("Recent cases:")
            st.write(client.recent_cases)
            st.write("Account notes:")
            st.write(client.account_notes)
        else:
            st.write("No client record found for this phone number.")

        st.subheader("Suggested actions")
        for item in result["actions"]:
            st.write(f"- {item}")


if __name__ == "__main__":
    main()
