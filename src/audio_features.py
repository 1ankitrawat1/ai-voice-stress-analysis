"""Audio feature extraction for the Lie Possibility project."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import librosa
import numpy as np


@dataclass
class AudioFeatureResult:
    """Container for extracted audio values."""

    is_valid: bool
    error_message: str
    duration: float
    rms_energy_mean: float
    rms_energy_std: float
    zero_crossing_rate_mean: float
    spectral_centroid_mean: float
    spectral_centroid_std: float
    mfcc_mean_values: list[float]
    silence_ratio: float
    estimated_pause_count: int
    speaking_stability_score: float
    pitch_mean: float
    pitch_std: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _empty_features(message: str = "Audio was not analyzed.", duration: float = 0.0) -> Dict[str, Any]:
    return AudioFeatureResult(
        is_valid=False,
        error_message=message,
        duration=round(float(duration or 0.0), 3),
        rms_energy_mean=0.0,
        rms_energy_std=0.0,
        zero_crossing_rate_mean=0.0,
        spectral_centroid_mean=0.0,
        spectral_centroid_std=0.0,
        mfcc_mean_values=[],
        silence_ratio=1.0,
        estimated_pause_count=0,
        speaking_stability_score=0.0,
        pitch_mean=0.0,
        pitch_std=0.0,
    ).to_dict()


def _safe_float(value: float | np.ndarray, default: float = 0.0) -> float:
    try:
        output = float(value)
        if np.isnan(output) or np.isinf(output):
            return default
        return output
    except Exception:
        return default


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _load_audio(audio_path: str | Path, sample_rate: int) -> tuple[np.ndarray, int]:
    path = str(audio_path)
    y, sr = librosa.load(path, sr=sample_rate, mono=True)
    return y.astype(np.float32), sr


def _count_pauses(rms_values: np.ndarray, threshold: float, frame_time: float, min_pause_seconds: float = 0.35) -> int:
    if rms_values.size == 0:
        return 0

    silent_frames = rms_values < threshold
    pause_count = 0
    current_silent_time = 0.0

    for is_silent in silent_frames:
        if is_silent:
            current_silent_time += frame_time
        else:
            if current_silent_time >= min_pause_seconds:
                pause_count += 1
            current_silent_time = 0.0

    if current_silent_time >= min_pause_seconds:
        pause_count += 1

    return int(pause_count)


def _estimate_pitch(y: np.ndarray, sr: int) -> tuple[float, float]:
    """Estimate pitch with librosa.yin and return stable numeric values."""
    if y.size < sr * 0.25 or np.max(np.abs(y)) < 1e-5:
        return 0.0, 0.0

    try:
        pitch_values = librosa.yin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        pitch_values = pitch_values[np.isfinite(pitch_values)]
        pitch_values = pitch_values[(pitch_values > 0) & (pitch_values < 2500)]
        if pitch_values.size == 0:
            return 0.0, 0.0
        return _safe_float(np.mean(pitch_values)), _safe_float(np.std(pitch_values))
    except Exception:
        return 0.0, 0.0


def extract_audio_features(audio_path: str | Path, sample_rate: int = 16000) -> Dict[str, Any]:
    """Extract audio features with defensive handling for invalid or silent files."""
    path = Path(audio_path)
    if not path.exists():
        return _empty_features("Audio file was not found.")

    try:
        y, sr = _load_audio(path, sample_rate)
    except Exception:
        return _empty_features("Audio could not be opened. Use WAV, MP3, or M4A.")

    if y.size == 0:
        return _empty_features("Audio file has no usable sound.")

    duration = _safe_float(librosa.get_duration(y=y, sr=sr))
    if duration < 0.30:
        return _empty_features("Audio is too short for analysis.", duration)

    if np.max(np.abs(y)) < 1e-5:
        return _empty_features("Audio appears silent.", duration)

    try:
        frame_length = 2048
        hop_length = 512
        frame_time = hop_length / sr

        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)

        rms_mean = _safe_float(np.mean(rms))
        rms_std = _safe_float(np.std(rms))
        zcr_mean = _safe_float(np.mean(zcr))
        centroid_mean = _safe_float(np.mean(centroid))
        centroid_std = _safe_float(np.std(centroid))
        mfcc_means = [_safe_float(v) for v in np.mean(mfcc, axis=1)]

        dynamic_threshold = max(0.01 * float(np.max(rms)), float(np.percentile(rms, 20)) * 0.85, 1e-5)
        silence_ratio = _safe_float(np.mean(rms < dynamic_threshold))
        pause_count = _count_pauses(rms, dynamic_threshold, frame_time)
        pitch_mean, pitch_std = _estimate_pitch(y, sr)

        energy_variability = rms_std / (rms_mean + 1e-6)
        centroid_variability = centroid_std / (centroid_mean + 1e-6)
        pitch_variability = pitch_std / (pitch_mean + 1e-6) if pitch_mean > 0 else 0.0
        instability = (energy_variability * 45) + (centroid_variability * 25) + (pitch_variability * 30)
        speaking_stability_score = _clip(100.0 - instability)

        return AudioFeatureResult(
            is_valid=True,
            error_message="",
            duration=round(duration, 3),
            rms_energy_mean=round(rms_mean, 6),
            rms_energy_std=round(rms_std, 6),
            zero_crossing_rate_mean=round(zcr_mean, 6),
            spectral_centroid_mean=round(centroid_mean, 3),
            spectral_centroid_std=round(centroid_std, 3),
            mfcc_mean_values=[round(x, 5) for x in mfcc_means],
            silence_ratio=round(silence_ratio, 4),
            estimated_pause_count=pause_count,
            speaking_stability_score=round(speaking_stability_score, 2),
            pitch_mean=round(pitch_mean, 3),
            pitch_std=round(pitch_std, 3),
        ).to_dict()
    except Exception:
        return _empty_features("Audio analysis failed for this file.", duration)
