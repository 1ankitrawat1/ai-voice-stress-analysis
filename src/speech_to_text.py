"""English speech-to-text support for recorded or uploaded answers."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

DEFAULT_MODEL_NAME = os.getenv("FASTER_WHISPER_MODEL", "base.en")


@lru_cache(maxsize=1)
def _load_faster_whisper_model(model_name: str = DEFAULT_MODEL_NAME):
    from faster_whisper import WhisperModel  # type: ignore

    return WhisperModel(model_name, device="cpu", compute_type="int8")


@lru_cache(maxsize=1)
def _load_openai_whisper_model(model_name: str = "base.en"):
    import whisper  # type: ignore

    return whisper.load_model(model_name)


def _transcribe_with_faster_whisper(audio_path: str) -> Optional[str]:
    try:
        model = _load_faster_whisper_model(DEFAULT_MODEL_NAME)
        segments, _ = model.transcribe(
            audio_path,
            beam_size=1,
            language="en",
            vad_filter=True,
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
        return text.strip() or None
    except Exception:
        return None


def _transcribe_with_openai_whisper(audio_path: str) -> Optional[str]:
    try:
        model = _load_openai_whisper_model("base.en")
        result = model.transcribe(audio_path, language="en", fp16=False)
        text = str(result.get("text", "")).strip()
        return text or None
    except Exception:
        return None


def transcribe_audio_file(audio_path: str | Path) -> Optional[str]:
    """Return English transcript text when a supported Whisper package is ready."""
    path = Path(audio_path)
    if not path.exists():
        return None

    faster_text = _transcribe_with_faster_whisper(str(path))
    if faster_text:
        return faster_text

    whisper_text = _transcribe_with_openai_whisper(str(path))
    if whisper_text:
        return whisper_text

    return None


def speech_to_text_status() -> str:
    try:
        import faster_whisper  # noqa: F401

        return "Speech-to-text is ready for English audio using faster-whisper."
    except Exception:
        pass

    try:
        import whisper  # noqa: F401

        return "Speech-to-text is ready for English audio using whisper."
    except Exception:
        return "Speech-to-text package is not installed. Install requirements again or type the transcript after upload."
