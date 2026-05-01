"""Hybrid scoring engine for voice-based Lie Possibility estimation."""

from __future__ import annotations

from statistics import mean
from typing import Dict, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .answer_matching import (
    analyze_answer_matching,
    calculate_text_hesitation_score,
    estimate_words_per_minute,
    text_hesitation_level,
)


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def _safe_ratio(answer_value: float, normal_value: float) -> float:
    if normal_value <= 1e-8:
        return 0.0 if answer_value <= 1e-8 else 1.0
    return abs(answer_value - normal_value) / (abs(normal_value) + 1e-8)


def result_level_from_score(score: float) -> str:
    if score <= 35:
        return "Low"
    if score <= 65:
        return "Medium"
    return "High"


def label_from_score(score: float) -> str:
    return result_level_from_score(score)


def stability_label(score: float) -> str:
    return "Stable" if score < 45 else "Unstable"


def calculate_voice_difference_score(normal_features: Dict, answer_features: Dict) -> float:
    if not normal_features.get("is_valid") or not answer_features.get("is_valid"):
        return 45.0

    energy_change = _safe_ratio(answer_features.get("rms_energy_mean", 0.0), normal_features.get("rms_energy_mean", 0.0))
    pitch_change = _safe_ratio(answer_features.get("pitch_mean", 0.0), normal_features.get("pitch_mean", 0.0))
    centroid_change = _safe_ratio(answer_features.get("spectral_centroid_mean", 0.0), normal_features.get("spectral_centroid_mean", 0.0))
    stability_change = abs(answer_features.get("speaking_stability_score", 0.0) - normal_features.get("speaking_stability_score", 0.0)) / 100.0

    score = (energy_change * 25) + (pitch_change * 25) + (centroid_change * 20) + (stability_change * 30)
    return round(_clip(score), 2)


def calculate_pause_score(answer_features: Dict, answer_text: str) -> float:
    if not answer_features.get("is_valid"):
        return 45.0

    duration = max(answer_features.get("duration", 0.0), 0.1)
    pause_count = answer_features.get("estimated_pause_count", 0)
    silence_ratio = answer_features.get("silence_ratio", 0.0)
    pauses_per_minute = pause_count / (duration / 60.0)

    # Combine audio pauses with transcript hesitation. Repeated fillers such as
    # "uh uh uh" are counted per occurrence by calculate_text_hesitation_score.
    audio_pause_score = (silence_ratio * 55) + min(30, pauses_per_minute * 2.5)
    text_hesitation_score = calculate_text_hesitation_score([answer_text])
    score = (audio_pause_score * 0.62) + (text_hesitation_score * 0.38)
    return round(_clip(score), 2)


def _random_forest_support_score(features: Dict) -> float:
    """Use a small internal Random Forest support layer for score smoothing."""
    try:
        x_train = np.array(
            [
                [0.02, 0.20, 75, 120, 0.03],
                [0.04, 0.30, 65, 160, 0.05],
                [0.08, 0.45, 50, 230, 0.08],
                [0.12, 0.60, 35, 300, 0.10],
                [0.03, 0.22, 80, 130, 0.04],
                [0.09, 0.50, 45, 260, 0.09],
            ],
            dtype=float,
        )
        y_train = np.array([0, 0, 1, 1, 0, 1], dtype=int)
        model = RandomForestClassifier(n_estimators=40, random_state=7, max_depth=3)
        model.fit(x_train, y_train)

        vector = np.array(
            [
                [
                    float(features.get("rms_energy_std", 0.0)),
                    float(features.get("silence_ratio", 0.0)),
                    float(features.get("speaking_stability_score", 0.0)),
                    float(features.get("pitch_std", 0.0)),
                    float(features.get("zero_crossing_rate_mean", 0.0)),
                ]
            ],
            dtype=float,
        )
        probability = model.predict_proba(vector)[0][1]
        return round(_clip(probability * 100.0), 2)
    except Exception:
        return 50.0


def calculate_voice_stress_score(normal_features: Dict, answer_features: Dict) -> float:
    if not answer_features.get("is_valid"):
        return 45.0

    stability = answer_features.get("speaking_stability_score", 0.0)
    silence_ratio = answer_features.get("silence_ratio", 0.0)
    pitch_std = answer_features.get("pitch_std", 0.0)
    energy_std = answer_features.get("rms_energy_std", 0.0)
    difference = calculate_voice_difference_score(normal_features, answer_features)
    support_score = _random_forest_support_score(answer_features)

    instability_score = 100.0 - stability
    pitch_score = min(100.0, pitch_std / 4.0)
    energy_score = min(100.0, energy_std * 900.0)
    silence_score = silence_ratio * 100.0

    rule_score = (instability_score * 0.30) + (pitch_score * 0.18) + (energy_score * 0.17) + (silence_score * 0.15) + (difference * 0.20)
    score = (rule_score * 0.82) + (support_score * 0.18)
    return round(_clip(score), 2)


def calculate_speaking_speed_label(answer_texts: List[str], answer_features: List[Dict]) -> str:
    speeds = []
    for text, features in zip(answer_texts, answer_features):
        if features.get("is_valid"):
            speed = estimate_words_per_minute(text, features.get("duration", 0.0))
            if speed > 0:
                speeds.append(speed)
    if len(speeds) < 2:
        return "Stable"
    speed_range = max(speeds) - min(speeds)
    return "Unstable" if speed_range > 75 else "Stable"


def build_main_reasons(component_scores: Dict[str, float], answer_matching: Dict[str, object], speed_label: str) -> List[str]:
    reasons: List[str] = []

    if answer_matching.get("strong_contradiction"):
        reasons.append("An important answer detail changed during the test.")
    if answer_matching.get("hesitation_count", 0) >= 3:
        reasons.append("Repeated hesitation words were found.")
    if answer_matching.get("uncertainty_count", 0) >= 1:
        reasons.append("Some answers were unclear or uncertain.")
    if component_scores["pause_and_hesitation"] > 45:
        reasons.append("Long pauses or hesitation signs were found before some answers.")
    if component_scores["difference_from_normal"] > 45:
        reasons.append("Voice changed more than normal.")
    if answer_matching.get("answer_matching_level") == "Not Fully Matched":
        reasons.append("Some answers did not match properly.")
    if speed_label == "Unstable":
        reasons.append("Speaking speed was not stable.")
    if component_scores["voice_stress"] > 55:
        reasons.append("Voice stress level was higher in some answers.")

    if not reasons:
        reasons.append("Voice answers stayed within the lower concern range.")
    return list(dict.fromkeys(reasons))


def apply_final_score_calibration(base_score: float, component_scores: Dict[str, float], answer_matching: Dict[str, object], hesitation_label: str) -> float:
    """Apply conservative escalation rules for clear hesitation and answer shifts.

    The system still estimates only Lie Possibility. These rules prevent obvious
    transcript concerns from being averaged down too far by stable audio values.
    """
    calibrated = float(base_score)
    answer_score = float(answer_matching.get("answer_matching_score", 0.0))
    pause_score = float(component_scores.get("pause_and_hesitation", 0.0))
    voice_score = float(component_scores.get("voice_stress", 0.0))
    difference_score = float(component_scores.get("difference_from_normal", 0.0))

    if answer_matching.get("strong_contradiction"):
        calibrated = max(calibrated, 68.0)
    elif answer_matching.get("self_correction_count", 0) >= 1 and answer_score >= 55:
        calibrated = max(calibrated, 58.0)

    if hesitation_label == "High" and answer_matching.get("answer_matching_level") == "Not Fully Matched":
        calibrated = max(calibrated, 55.0)

    if answer_score >= 65 and pause_score >= 55:
        calibrated = max(calibrated, 66.0)
    elif answer_score >= 55 and pause_score >= 45:
        calibrated = max(calibrated, 55.0)

    if voice_score >= 65 and (pause_score >= 45 or difference_score >= 50):
        calibrated = max(calibrated, 58.0)

    return round(_clip(calibrated), 2)


def analyze_test(normal_features: Dict, answer_features: List[Dict], answer_texts: List[str]) -> Dict[str, object]:
    valid_answer_features = [features for features in answer_features if features.get("is_valid")]
    if not normal_features.get("is_valid") or not valid_answer_features:
        return {
            "ready": False,
            "error": "Normal voice and question answers are required for analysis.",
        }

    voice_stress_scores = [calculate_voice_stress_score(normal_features, features) for features in answer_features]
    pause_scores = [calculate_pause_score(features, text) for features, text in zip(answer_features, answer_texts)]
    difference_scores = [calculate_voice_difference_score(normal_features, features) for features in answer_features]
    answer_matching = analyze_answer_matching(answer_texts)

    voice_stress_avg = round(mean(voice_stress_scores), 2)
    audio_pause_avg = round(mean(pause_scores), 2)
    transcript_hesitation_score = calculate_text_hesitation_score(answer_texts)
    pause_avg = round(max(audio_pause_avg, transcript_hesitation_score), 2)
    difference_avg = round(mean(difference_scores), 2)
    answer_matching_score = float(answer_matching.get("answer_matching_score", 0.0))

    speed_label = calculate_speaking_speed_label(answer_texts, answer_features)
    hesitation_label, hesitation_count = text_hesitation_level(answer_texts)

    component_scores = {
        "voice_stress": voice_stress_avg,
        "pause_and_hesitation": pause_avg,
        "difference_from_normal": difference_avg,
        "answer_matching": answer_matching_score,
    }

    base_score = round(
        (voice_stress_avg * 0.30)
        + (pause_avg * 0.25)
        + (difference_avg * 0.25)
        + (answer_matching_score * 0.20),
        2,
    )
    final_score = apply_final_score_calibration(base_score, component_scores, answer_matching, hesitation_label)
    result_level = result_level_from_score(final_score)

    voice_change_label = stability_label(difference_avg)
    pause_label = label_from_score(pause_avg)
    voice_stress_label = label_from_score(voice_stress_avg)

    return {
        "ready": True,
        "final_score": final_score,
        "base_score_before_calibration": base_score,
        "result_level": result_level,
        "component_scores": component_scores,
        "transcript_hesitation_score": transcript_hesitation_score,
        "voice_stress_scores": [round(score, 2) for score in voice_stress_scores],
        "cards": {
            "Voice Stress": voice_stress_label,
            "Pause Level": pause_label,
            "Hesitation": hesitation_label,
            "Answer Matching": answer_matching.get("answer_matching_level", "Matched"),
            "Speaking Speed": speed_label,
            "Voice Change": voice_change_label,
        },
        "answer_matching": answer_matching,
        "hesitation_count": hesitation_count,
        "main_reasons": build_main_reasons(component_scores, answer_matching, speed_label),
    }
