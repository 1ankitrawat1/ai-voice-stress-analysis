"""Answer matching and transcript checks for the Lie Possibility project."""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Tuple

from .question_planner import extract_statement_context, normalize_text

HESITATION_PATTERNS = [
    r"\bum\b",
    r"\buh\b",
    r"\ber\b",
    r"\bah\b",
    r"\bmaybe\b",
    r"\bi think\b",
    r"\bnot sure\b",
    r"\bi guess\b",
    r"\bkind of\b",
    r"\bsort of\b",
]

TIME_PATTERN = re.compile(
    r"\b(?:[01]?\d|2[0-3])(?::[0-5]\d)?\s?(?:am|pm)?\b|"
    r"\b(?:morning|afternoon|evening|night|noon|midnight|yesterday|today|tomorrow)\b",
    re.IGNORECASE,
)

NEGATION_PATTERNS = [
    r"\bi did not\b",
    r"\bi didn't\b",
    r"\bi was not\b",
    r"\bi wasn't\b",
    r"\bnot really\b",
    r"\bno\b",
]

UNCERTAINTY_PHRASES = [
    "not remember",
    "do not remember",
    "don't remember",
    "dont remember",
    "no idea",
    "i forgot",
    "not sure",
    "i don't know",
    "i dont know",
    "don't know",
    "dont know",
    "i think",
    "maybe",
    "i guess",
]

CORRECTION_PHRASES = [
    "no no",
    "wait",
    "actually",
    "sorry",
    "i mean",
    "rather",
]

PERSON_WORDS = {"friend", "friends", "alone", "myself", "family", "brother", "sister", "classmate", "colleague"}
MOVEMENT_WORDS = {"went", "go", "gone", "visited", "visit", "left", "returned", "came", "reached", "travelled", "traveled"}
STAY_WORDS = {"stayed", "stay", "home", "house", "room"}


def _words(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z']+\b", normalize_text(text))


def count_hesitation_words(text: str) -> int:
    normalized = normalize_text(text)
    total = 0
    for pattern in HESITATION_PATTERNS:
        total += len(re.findall(pattern, normalized, flags=re.IGNORECASE))
    return total


def count_uncertainty_phrases(text: str) -> int:
    normalized = normalize_text(text)
    return sum(normalized.count(phrase) for phrase in UNCERTAINTY_PHRASES)


def count_correction_phrases(text: str) -> int:
    normalized = normalize_text(text)
    return sum(normalized.count(phrase) for phrase in CORRECTION_PHRASES)


def extract_times(text: str) -> List[str]:
    normalized = normalize_text(text)
    output: List[str] = []
    for match in TIME_PATTERN.finditer(normalized):
        item = match.group(0).strip().lower()
        if item not in output:
            output.append(item)
    return output


def _is_short_answer(text: str) -> bool:
    words = [w for w in normalize_text(text).split(" ") if w]
    return len(words) <= 2


def _repeated_word_count(text: str) -> int:
    words = re.findall(r"\b[a-zA-Z']+\b", normalize_text(text))
    if not words:
        return 0
    counts = Counter(words)
    return sum(count - 1 for count in counts.values() if count > 1)


def _has_negation(text: str) -> bool:
    normalized = normalize_text(text)
    return any(re.search(pattern, normalized) for pattern in NEGATION_PATTERNS)


def _detect_statement_shift(initial_statement: str, later_answers: List[str]) -> bool:
    initial_words = set(_words(initial_statement))
    combined_later = " ".join(later_answers)
    later_words = set(_words(combined_later))

    initial_movement = bool(initial_words & MOVEMENT_WORDS)
    later_stay = bool(later_words & STAY_WORDS) and _has_negation(combined_later)

    initial_stay = bool(initial_words & STAY_WORDS) and not bool(initial_words & MOVEMENT_WORDS)
    later_movement = bool(later_words & MOVEMENT_WORDS)

    return bool((initial_movement and later_stay) or (initial_stay and later_movement))


def _detect_time_review_needed(initial_statement: str, later_answers: List[str]) -> bool:
    initial_times = extract_times(initial_statement)
    later_times: List[str] = []
    for answer in later_answers:
        later_times.extend(extract_times(answer))

    broad_initial = {t for t in initial_times if t in {"morning", "afternoon", "evening", "night", "noon", "midnight"}}
    broad_later = {t for t in later_times if t in {"morning", "afternoon", "evening", "night", "noon", "midnight"}}

    if broad_initial and broad_later and broad_initial.isdisjoint(broad_later):
        return True

    return False


def _detect_person_contradiction(answer: str) -> bool:
    """Detect a same-answer shift such as 'my friend, no, I was alone'."""
    normalized = normalize_text(answer)
    words = set(_words(normalized))
    mentions_friend_or_group = bool(words & {"friend", "friends", "family", "brother", "sister", "classmate", "colleague"})
    mentions_alone = bool(words & {"alone", "myself"})
    has_correction = count_correction_phrases(normalized) > 0 or _has_negation(normalized)
    return bool(mentions_friend_or_group and mentions_alone and has_correction)


def _detect_strong_self_correction(answer: str) -> bool:
    normalized = normalize_text(answer)
    if count_correction_phrases(normalized) >= 1 and len(_words(normalized)) >= 4:
        return True
    if re.search(r"\b(yes|yeah)\b.*\b(no|not)\b|\b(no|not)\b.*\b(yes|yeah)\b", normalized):
        return True
    return False


def text_hesitation_level(answers: List[str]) -> Tuple[str, int]:
    count = sum(count_hesitation_words(answer) for answer in answers)
    uncertainty_count = sum(count_uncertainty_phrases(answer) for answer in answers)
    correction_count = sum(count_correction_phrases(answer) for answer in answers)
    combined_count = count + uncertainty_count + correction_count
    if combined_count <= 1:
        return "Low", combined_count
    if combined_count <= 4:
        return "Medium", combined_count
    return "High", combined_count


def calculate_text_hesitation_score(answers: List[str]) -> float:
    """Convert transcript hesitation signs into a 0-100 concern score."""
    if not answers:
        return 0.0

    hesitation_count = sum(count_hesitation_words(answer) for answer in answers)
    uncertainty_count = sum(count_uncertainty_phrases(answer) for answer in answers)
    correction_count = sum(count_correction_phrases(answer) for answer in answers)
    repeated_total = sum(_repeated_word_count(answer) for answer in answers)
    short_count = sum(1 for answer in answers if answer and _is_short_answer(answer))

    score = 0.0
    score += min(38.0, hesitation_count * 7.5)
    score += min(30.0, uncertainty_count * 10.0)
    score += min(24.0, correction_count * 12.0)
    score += min(14.0, repeated_total * 2.5)
    score += min(10.0, short_count * 4.0)

    if hesitation_count >= 4 and uncertainty_count >= 1:
        score = max(score, 62.0)
    if correction_count >= 1 and (hesitation_count >= 2 or uncertainty_count >= 1):
        score = max(score, 58.0)

    return round(max(0.0, min(100.0, score)), 2)


def analyze_answer_matching(answers: List[str]) -> Dict[str, object]:
    """Analyze transcripts and return a score for answer matching concerns."""
    cleaned = [normalize_text(answer) for answer in answers]
    initial_statement = cleaned[0] if cleaned else ""
    later_answers = cleaned[1:] if len(cleaned) > 1 else []
    context = extract_statement_context(initial_statement)

    reasons: List[str] = []
    score = 0.0

    if len(cleaned) < 5:
        reasons.append("All voice answers were not available for matching.")
        score += 18

    empty_count = sum(1 for answer in cleaned if not answer)
    if empty_count:
        reasons.append("Some answers were not available as text.")
        score += min(32, empty_count * 8)

    short_count = sum(1 for answer in cleaned if answer and _is_short_answer(answer))
    if short_count:
        reasons.append("Some answers were too short for strong matching.")
        score += min(24, short_count * 6)

    hesitation_count = sum(count_hesitation_words(answer) for answer in cleaned)
    if hesitation_count >= 2:
        reasons.append("Repeated hesitation words were found.")
        score += min(36, hesitation_count * 7)

    repeated_total = sum(_repeated_word_count(answer) for answer in cleaned)
    if repeated_total >= 3:
        reasons.append("Repeated words were found in the answers.")
        score += min(18, repeated_total * 3)

    uncertainty_count = sum(count_uncertainty_phrases(answer) for answer in cleaned)
    if uncertainty_count:
        reasons.append("Some answers were unclear or uncertain.")
        score += min(30, uncertainty_count * 12)

    correction_count = sum(count_correction_phrases(answer) for answer in cleaned)
    self_correction_count = sum(1 for answer in cleaned if _detect_strong_self_correction(answer))
    if self_correction_count:
        reasons.append("An answer changed during the response.")
        score += min(30, self_correction_count * 18)

    strong_contradiction = any(_detect_person_contradiction(answer) for answer in cleaned)
    if strong_contradiction:
        reasons.append("An important answer detail changed during the test.")
        score += 34

    if initial_statement and not context.is_clear:
        reasons.append("The first statement was not clear enough for strong matching.")
        score += 12

    if _detect_statement_shift(initial_statement, later_answers):
        reasons.append("Some answers did not match properly.")
        score += 24

    if _detect_time_review_needed(initial_statement, later_answers):
        reasons.append("The time answers need review.")
        score += 14

    if later_answers:
        location_terms = set(context.locations)
        if location_terms and not any(term in " ".join(later_answers) for term in location_terms):
            score += 6

    if strong_contradiction:
        score = max(score, 68.0)
    elif self_correction_count >= 1 and uncertainty_count >= 1:
        score = max(score, 58.0)
    elif hesitation_count >= 4 and uncertainty_count >= 1:
        score = max(score, 55.0)

    score = round(max(0.0, min(100.0, score)), 2)
    level = "Matched" if score <= 30 else "Not Fully Matched"

    if not reasons:
        reasons.append("Answers were clear enough for matching.")

    unique_reasons = list(dict.fromkeys(reasons))
    return {
        "answer_matching_score": score,
        "answer_matching_level": level,
        "hesitation_count": hesitation_count,
        "uncertainty_count": uncertainty_count,
        "correction_count": correction_count,
        "self_correction_count": self_correction_count,
        "strong_contradiction": strong_contradiction,
        "repeated_word_count": repeated_total,
        "statement_context": context.to_dict(),
        "reasons": unique_reasons,
    }


def estimate_words_per_minute(text: str, duration_seconds: float) -> float:
    if duration_seconds <= 0:
        return 0.0
    words = re.findall(r"\b\w+\b", normalize_text(text))
    return round((len(words) / duration_seconds) * 60.0, 2)
