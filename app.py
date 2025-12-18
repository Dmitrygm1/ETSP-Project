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


def generate_suggested_actions(transcript: str, emotion_label: Optional[str]) -> list[str]:
    t = transcript.lower()
    actions: list[str] = []
    if emotion_label and emotion_label.lower() in {"angry", "disgusted"}:
        actions.append("Caller upset: acknowledge feelings and de-escalate; use calm tone.")
    if emotion_label and emotion_label.lower() in {"sad", "fearful"}:
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
    st.title("Call Support Copilot")
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

            ser_error = None
            ser_timeline = []
            ser_label = None
            ser_scores: dict[str, float] = {}
            ser_vad: dict[str, float] = {}
            try:
                with st.spinner("Detecting emotion (SER)..."):
                    from ser import load_audio_16k_mono, predict_emotion_windows

                    audio_16k = load_audio_16k_mono(tmp_path)
                    ser_timeline, ser_label, ser_scores, ser_vad = predict_emotion_windows(
                        audio_16k,
                        window_s=4.0,
                        hop_s=2.0,
                    )
            except Exception as exc:
                ser_error = str(exc)

            actions = generate_suggested_actions(translated or transcript, ser_label)

            st.session_state["call_result"] = {
                "caller_phone": caller_phone,
                "client": client,
                "transcript": transcript,
                "detected_lang": detected_lang,
                "lang_prob": lang_prob,
                "translated": translated,
                "ser_timeline": ser_timeline,
                "ser_label": ser_label,
                "ser_scores": ser_scores,
                "ser_vad": ser_vad,
                "ser_error": ser_error,
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

        st.subheader("Emotion (SER)")
        if result.get("ser_error"):
            st.warning(f"SER failed: {result['ser_error']}")
        ser_label = result.get("ser_label")
        ser_scores = result.get("ser_scores") or {}
        ser_timeline = result.get("ser_timeline") or []
        ser_vad = result.get("ser_vad") or {}
        if ser_label:
            st.write(f"**Overall:** {ser_label} ({ser_scores.get(ser_label, 0.0):.2f})")
            if ser_vad:
                v_col, a_col, d_col = st.columns(3)
                v_col.metric("Valence", f"{ser_vad.get('valence', 0.0):.2f}")
                a_col.metric("Arousal", f"{ser_vad.get('arousal', 0.0):.2f}")
                d_col.metric("Dominance", f"{ser_vad.get('dominance', 0.0):.2f}")
            if ser_timeline:
                t0, last_label, last_conf, (v, a, d), _ = ser_timeline[-1]
                st.write(
                    f"**Current:** {last_label} ({last_conf:.2f}) at {t0:.1f}s "
                    f"| V/A/D: {v:.2f}/{a:.2f}/{d:.2f}"
                )
            with st.expander("Mean emotion scores"):
                st.json({k: round(v, 4) for k, v in ser_scores.items()})
            with st.expander("Emotion timeline"):
                rows = [
                    {
                        "t_start_s": round(t, 2),
                        "label": lbl,
                        "confidence": round(conf, 4),
                        "valence": round(v, 4),
                        "arousal": round(a, 4),
                        "dominance": round(d, 4),
                    }
                    for (t, lbl, conf, (v, a, d), _full_probs) in ser_timeline
                ]
                st.dataframe(rows, use_container_width=True, height=240)
        else:
            st.write("No SER result.")

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
