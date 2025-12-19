from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
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
    """Load and cache the Whisper model.
    
    Downloads the model from HuggingFace on first call.
    """
    from faster_whisper import WhisperModel

    try:
        return WhisperModel(model_size, device="cpu", compute_type="int8")
    except Exception as e:
        error_msg = str(e)
        if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            raise RuntimeError(
                f"Failed to download Whisper model. Check your internet connection.\n"
                f"Model size: {model_size}\n"
                f"Error: {error_msg}"
            )
        raise


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


@st.cache_data(show_spinner=False)
def load_slu_intents_config() -> dict:
    cfg_path = Path(__file__).resolve().parent / "slu" / "intents_config.json"
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def generate_intent_actions(slu_result: Optional[dict], emotion_label: Optional[str]) -> list[str]:
    intent = str((slu_result or {}).get("intent") or "unknown").strip()
    slots = (slu_result or {}).get("slots") or {}

    cfg = load_slu_intents_config()
    actions_map = cfg.get("intent_actions") or {}
    actions: list[str] = list(actions_map.get(intent) or actions_map.get(intent.lower()) or actions_map.get("unknown") or [])

    try:
        from slu.slots import suggest_followups

        followups = suggest_followups(intent, slots)
        actions = followups + actions
    except Exception:
        pass

    if emotion_label and emotion_label.lower() in {"angry", "disgusted"} and intent in {"chargeback", "refund_not_showing_up"}:
        actions = ["Caller upset: acknowledge feelings and de-escalate; use calm tone."] + actions

    return dedupe_preserve_order([a for a in actions if a])


def get_emotion_color(emotion: Optional[str]) -> str:
    """Returns color code for emotion"""
    if not emotion:
        return "#6B7280"
    emotion_lower = emotion.lower()
    colors = {
        "happy": "#10B981",
        "neutral": "#6B7280",
        "sad": "#3B82F6",
        "angry": "#EF4444",
        "fearful": "#8B5CF6",
        "disgusted": "#F59E0B",
        "surprised": "#EC4899",
    }
    return colors.get(emotion_lower, "#6B7280")


def get_status_badge_color(status: str) -> str:
    """Returns color for status badge"""
    return "#10B981" if status.upper() == "VIP" else "#6B7280"


def main() -> None:
    st.set_page_config(
        page_title="Call Support Copilot",
        page_icon="📞",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for better styling
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        }
        .main-header h1 {
            color: white;
            margin: 0;
            font-size: 2.5rem;
        }
        .main-header p {
            color: rgba(255, 255, 255, 0.9);
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
        }
        .info-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            margin: 1rem 0;
            color: #1F2937;
        }
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
            color: #1F2937;
        }
        .emotion-badge {
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            color: white;
            font-weight: 600;
            margin: 0.25rem;
        }
        .status-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            color: white;
            font-weight: 600;
            font-size: 0.85rem;
        }
        .action-item {
            background: #f0f9ff;
            padding: 1rem;
            border-radius: 6px;
            border-left: 3px solid #3B82F6;
            margin: 0.5rem 0;
            color: #1F2937;
        }
        .client-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            color: #1F2937;
        }
        .stButton>button {
            width: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            padding: 0.75rem;
            border-radius: 8px;
            border: none;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .upload-section {
            background: #f8f9fa;
            padding: 2rem;
            border-radius: 10px;
            border: 2px dashed #cbd5e1;
            text-align: center;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📞 Call Support Copilot</h1>
        <p>AI-powered call analysis for customer support agents</p>
    </div>
    """, unsafe_allow_html=True)

    db_path = get_db_path()
    ensure_client_db(db_path)

    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")
        
        agent_language = st.selectbox(
            "🌐 Agent Language",
            options=["en"],
            index=0,
            disabled=True,
            help="Currently only English is supported",
        )
        
        whisper_model_size = st.selectbox(
            "🎤 Whisper Model",
            options=["base", "small"],
            index=0,
            help="Base: Faster, smaller. Small: Better accuracy but slower.",
        )
        
        st.markdown("---")
        st.markdown("### 📋 Demo Phone Numbers")
        st.markdown("""
        <div style="font-size: 0.9rem; color: #6B7280;">
        <p><strong>VIP Clients:</strong></p>
        <ul style="margin: 0.5rem 0;">
            <li>+41 79 123 45 67</li>
            <li>+33 1 23 45 67 89</li>
        </ul>
        <p><strong>Standard Clients:</strong></p>
        <ul style="margin: 0.5rem 0;">
            <li>+31 6 12345678</li>
            <li>+49 151 23456789</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        <div style="font-size: 0.85rem; color: #6B7280;">
        <p>This demo processes customer service calls using:</p>
        <ul>
            <li>🎤 Speech Recognition</li>
            <li>🌍 Translation</li>
            <li>😊 Emotion Detection</li>
            <li>🧠 Intent Understanding</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    col_left, col_middle, col_right = st.columns([1, 1, 1], gap="large")

    with col_left:
        st.markdown("### 📥 Input")
        st.markdown("---")
        
        phone_input = st.text_input(
            "📱 Caller Phone Number",
            placeholder="+41 79 123 45 67",
            help="Enter the caller's phone number to lookup client information",
        )
        
        st.markdown("### 🎵 Audio Upload")
        audio_file = st.file_uploader(
            "Upload call audio file",
            type=["wav", "mp3", "m4a", "flac", "ogg"],
            help="Supported formats: WAV, MP3, M4A, FLAC, OGG",
            label_visibility="collapsed",
        )
        
        if audio_file is not None:
            st.markdown("**Audio Preview:**")
            st.audio(audio_file)
            st.success(f"✅ File loaded: {audio_file.name}")
        else:
            st.info("👆 Please upload an audio file to begin analysis")

        run = st.button("🚀 Process Call", type="primary", disabled=(audio_file is None), width="stretch")

    if run and audio_file is not None:
        # Default to demo number if empty
        phone_val = phone_input.strip() or "+41791234567"
        caller_phone = canonicalize_phone_number(phone_val)
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

            text_for_slu = translated or transcript

            slu_error = None
            slu_result = None
            try:
                with st.spinner("Understanding intent (SLU)..."):
                    from slu.infer import run_slu

                    slu_result = run_slu(text_for_slu or "")
            except Exception as exc:
                slu_error = str(exc)

            intent_actions = generate_intent_actions(slu_result, ser_label) if slu_result else []
            rule_actions = generate_suggested_actions(text_for_slu or "", ser_label)
            actions = dedupe_preserve_order(intent_actions + rule_actions)

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
                "slu_result": slu_result,
                "slu_error": slu_error,
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
            st.markdown("### 📝 Transcription")
            st.markdown("---")
            
            # Language detection badge
            lang_prob = result['lang_prob']
            lang_color = "#10B981" if lang_prob > 0.8 else "#F59E0B" if lang_prob > 0.5 else "#EF4444"
            st.markdown(f"""
            <div style="margin-bottom: 1rem;">
                <span style="background: {lang_color}; color: white; padding: 0.5rem 1rem; border-radius: 20px; font-weight: 600;">
                    🌐 {result['detected_lang'].upper()} ({(lang_prob*100):.1f}% confidence)
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            st.text_area(
                "**Original Transcript**",
                value=result["transcript"],
                height=200,
                label_visibility="visible",
                disabled=True,
            )
            
            if result["detected_lang"] != agent_language:
                if result["translated"]:
                    st.markdown("### 🌍 Translation")
                    st.markdown("---")
                    st.text_area(
                        "**English Translation**",
                        value=result["translated"],
                        height=200,
                        label_visibility="visible",
                        disabled=True,
                    )
                else:
                    st.warning(
                        f"⚠️ Translation not available for `{result['detected_lang']}`. "
                        "Supported languages: English, Dutch (nl), German (de), French (fr), Spanish (es), Italian (it), Portuguese (pt)."
                    )

    if not result:
        with col_middle:
            st.markdown("### 📊 Analysis Results")
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 10px; border: 2px dashed #cbd5e1;">
                <h3 style="color: #6B7280;">📊 Analysis Results</h3>
                <p style="color: #9CA3AF; margin-top: 1rem;">
                    Upload an audio file and click <strong>Process Call</strong> to see<br>
                    transcription, emotion analysis, intent detection, and suggested actions here.
                </p>
            </div>
            """, unsafe_allow_html=True)
        with col_right:
            st.markdown("### 👤 Client Information")
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 10px; border: 2px dashed #cbd5e1;">
                <h3 style="color: #6B7280;">👤 Client Information</h3>
                <p style="color: #9CA3AF; margin-top: 1rem;">
                    Enter a phone number and process a call<br>
                    to see client records and suggested actions here.
                </p>
            </div>
            """, unsafe_allow_html=True)
        return

    with col_middle:
        st.markdown("### 😊 Emotion Analysis")
        st.markdown("---")
        
        if result.get("ser_error"):
            st.error(f"❌ SER failed: {result['ser_error']}")
        
        ser_label = result.get("ser_label")
        ser_scores = result.get("ser_scores") or {}
        ser_timeline = result.get("ser_timeline") or []
        ser_vad = result.get("ser_vad") or {}
        
        if ser_label:
            emotion_color = get_emotion_color(ser_label)
            confidence = ser_scores.get(ser_label, 0.0)
            
            # Emotion badge
            st.markdown(f"""
            <div style="margin-bottom: 1.5rem;">
                <span class="emotion-badge" style="background: {emotion_color}; font-size: 1.1rem;">
                    {ser_label.upper()} ({(confidence*100):.1f}%)
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # VAD Metrics
            if ser_vad:
                st.markdown("**Emotional Dimensions:**")
                v_col, a_col, d_col = st.columns(3)
                v_col.metric(
                    "😊 Valence",
                    f"{ser_vad.get('valence', 0.0):.2f}",
                    help="Positive (1.0) to Negative (0.0)"
                )
                a_col.metric(
                    "⚡ Arousal",
                    f"{ser_vad.get('arousal', 0.0):.2f}",
                    help="Excited (1.0) to Calm (0.0)"
                )
                d_col.metric(
                    "👑 Dominance",
                    f"{ser_vad.get('dominance', 0.0):.2f}",
                    help="Dominant (1.0) to Submissive (0.0)"
                )
            
            # Current emotion
            if ser_timeline:
                t0, last_label, last_conf, (v, a, d), _ = ser_timeline[-1]
                st.markdown(f"""
                <div class="info-card">
                    <strong>Current Emotion:</strong> {last_label} ({(last_conf*100):.1f}%) at {t0:.1f}s<br>
                    <small>V/A/D: {v:.2f} / {a:.2f} / {d:.2f}</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Emotion scores expander
            with st.expander("📈 Detailed Emotion Scores"):
                emotion_df = pd.DataFrame([
                    {"Emotion": k, "Score": f"{v:.4f}", "Percentage": f"{(v*100):.2f}%"}
                    for k, v in sorted(ser_scores.items(), key=lambda x: x[1], reverse=True)
                ])
                st.dataframe(emotion_df, width="stretch", hide_index=True)
            
            # Timeline expander
            if ser_timeline:
                with st.expander("⏱️ Emotion Timeline"):
                    rows = [
                        {
                            "Time (s)": round(t, 2),
                            "Emotion": lbl,
                            "Confidence": f"{conf:.2%}",
                            "Valence": round(v, 3),
                            "Arousal": round(a, 3),
                            "Dominance": round(d, 3),
                        }
                        for (t, lbl, conf, (v, a, d), _full_probs) in ser_timeline
                    ]
                    timeline_df = pd.DataFrame(rows)
                    st.dataframe(timeline_df, width="stretch", hide_index=True)
        else:
            st.info("No emotion analysis available.")

        st.markdown("---")
        st.markdown("### 🧠 Intent Understanding")
        st.markdown("---")
        
        if result.get("slu_error"):
            st.error(f"❌ SLU failed: {result['slu_error']}")
        
        slu = result.get("slu_result") or {}
        if slu:
            intent = slu.get("intent", "unknown")
            confidence = float(slu.get("confidence", 0.0))
            is_ood = slu.get("is_ood", False)
            rationale = slu.get("rationale", [])
            
            # Check if model is missing
            no_model = any("no SLU model found" in str(r).lower() for r in rationale)
            
            # Intent display
            if no_model:
                st.info("ℹ️ **No trained SLU model found.** Using keyword-based slot extraction only.")
                st.caption("💡 Train a model: `python -m slu.train_transformer`")
            elif is_ood:
                st.warning(f"⚠️ **Intent:** `{intent}` (confidence: {(confidence*100):.1f}% - Below threshold)")
            else:
                intent_color = "#10B981" if confidence > 0.8 else "#F59E0B"
                st.markdown(f"""
                <div style="margin-bottom: 1rem;">
                    <span style="background: {intent_color}; color: white; padding: 0.5rem 1rem; border-radius: 20px; font-weight: 600; font-size: 1rem;">
                        🎯 {intent.replace('_', ' ').title()} ({(confidence*100):.1f}%)
                    </span>
                </div>
                """, unsafe_allow_html=True)
            
            # Slots display
            slots = slu.get("slots") or {}
            if slots:
                st.markdown("**📋 Extracted Information:**")
                for slot_name, slot_value in slots.items():
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>{slot_name.replace('_', ' ').title()}:</strong> {slot_value}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No structured information extracted.")
            
            # SLU details expander
            with st.expander("🔍 SLU Details"):
                st.json(slu)
        else:
            st.info("No intent analysis available.")

    with col_right:
        st.markdown("### 👤 Client Information")
        st.markdown("---")
        
        client = result.get("client")
        if client:
            status_color = get_status_badge_color(client.status)
            st.markdown(f"""
            <div class="client-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <h3 style="margin: 0; color: #1F2937;">{client.name}</h3>
                    <span class="status-badge" style="background: {status_color};">
                        {client.status}
                    </span>
                </div>
                <p style="margin: 0.5rem 0; color: #6B7280;">
                    <strong>📱 Phone:</strong> <code>{client.phone_number}</code>
                </p>
                <p style="margin: 0.5rem 0; color: #6B7280;">
                    <strong>📅 Last Contact:</strong> {client.last_contact_date}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**📋 Recent Cases:**")
            st.markdown(f"""
            <div class="info-card" style="background: #FEF3C7; border-left-color: #F59E0B;">
                {client.recent_cases}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**📝 Account Notes:**")
            st.markdown(f"""
            <div class="info-card" style="background: #DBEAFE; border-left-color: #3B82F6;">
                {client.account_notes}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ No client record found for this phone number.")
            st.info("💡 Try using one of the demo phone numbers from the sidebar.")

        st.markdown("---")
        st.markdown("### 💡 Suggested Actions")
        st.markdown("---")
        
        actions = result.get("actions", [])
        if actions:
            for idx, item in enumerate(actions, 1):
                st.markdown(f"""
                <div class="action-item">
                    <strong>{idx}.</strong> {item}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No specific actions suggested. Proceed with standard support flow.")


if __name__ == "__main__":
    main()
