"""Shared utility functions for the Streamlit app."""

from __future__ import annotations

import re
import tempfile
from datetime import datetime
from pathlib import Path

SAFETY_NOTE = "This result is only an estimate. It does not prove that the person is lying."
INITIAL_STATEMENT_PROMPT = "Describe the situation or activity you want to answer about."
DEFAULT_ADAPTIVE_QUESTIONS = [
    "What time did this happen?",
    "Where were you during this situation?",
    "Who was with you at that time?",
    "What happened after that?",
]


def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_")
    return cleaned or "audio_file"


def save_uploaded_file(uploaded_file, prefix: str = "audio") -> str:
    original_name = getattr(uploaded_file, "name", "audio.wav") or "audio.wav"
    suffix = Path(original_name).suffix or ".wav"
    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{safe_filename(original_name)}"
    temp_dir = Path(tempfile.gettempdir()) / "ai_voice_lie_possibility"
    temp_dir.mkdir(parents=True, exist_ok=True)
    file_path = temp_dir / filename
    if not file_path.suffix:
        file_path = file_path.with_suffix(suffix)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(file_path)


def percent_text(value: float | int) -> str:
    try:
        return f"{round(float(value), 2)}%"
    except Exception:
        return "0%"


def reset_test_state(session_state) -> None:
    keys = [
        "current_page",
        "normal_audio_path",
        "normal_features",
        "initial_audio_path",
        "initial_features",
        "initial_transcript",
        "initial_manual_text",
        "generated_questions",
        "followup_answers",
        "question_index",
        "questions_started",
        "analysis",
        "test_datetime",
    ]
    for key in keys:
        if key in session_state:
            del session_state[key]
