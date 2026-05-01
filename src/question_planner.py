"""Adaptive follow-up question planning for the voice test."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

STOP_WORDS = {
    "i", "me", "my", "we", "our", "you", "he", "she", "they", "it", "the", "a", "an",
    "and", "or", "but", "to", "from", "in", "on", "at", "for", "with", "of", "was", "were",
    "am", "is", "are", "be", "been", "being", "did", "do", "does", "had", "have", "has",
    "that", "this", "there", "here", "then", "than", "so", "just", "very", "really",
    "yesterday", "today", "tomorrow", "last", "next", "morning", "afternoon", "evening", "night",
}

TIME_PATTERN = re.compile(
    r"\b(?:[01]?\d|2[0-3])(?::[0-5]\d)?\s?(?:am|pm)?\b|"
    r"\b(?:morning|afternoon|evening|night|noon|midnight|yesterday|today|tomorrow)\b",
    re.IGNORECASE,
)

LOCATION_TERMS = {
    "home", "house", "room", "hostel", "college", "school", "office", "market", "mall", "park",
    "gym", "library", "restaurant", "cafe", "hospital", "station", "temple", "cinema", "movie",
    "shop", "city", "village", "class", "coaching", "friend", "workplace", "bank",
}

STUDY_TERMS = {"study", "studied", "studying", "read", "reading", "exam", "class", "lecture", "python", "math", "science", "subject", "assignment"}
WORK_TERMS = {"work", "worked", "working", "office", "task", "project", "meeting", "client", "job", "shift"}
TRAVEL_TERMS = {"went", "go", "gone", "visited", "visit", "travel", "travelled", "reached", "left", "returned", "came", "market", "mall", "park", "station"}
SOCIAL_TERMS = {"friend", "friends", "family", "brother", "sister", "teacher", "sir", "madam", "colleague", "classmate"}
PURCHASE_TERMS = {"buy", "bought", "purchase", "purchased", "shopping", "shop", "ordered", "paid"}


@dataclass(frozen=True)
class StatementContext:
    statement_type: str
    times: List[str]
    locations: List[str]
    activity_terms: List[str]
    people_terms: List[str]
    is_clear: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "statement_type": self.statement_type,
            "times": self.times,
            "locations": self.locations,
            "activity_terms": self.activity_terms,
            "people_terms": self.people_terms,
            "is_clear": self.is_clear,
        }


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _words(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z']+\b", normalize_text(text))


def extract_statement_context(statement: str | None) -> StatementContext:
    normalized = normalize_text(statement)
    words = _words(normalized)
    word_set = set(words)

    times = []
    for match in TIME_PATTERN.finditer(normalized):
        value = match.group(0).strip().lower()
        if value not in times:
            times.append(value)

    locations = [word for word in words if word in LOCATION_TERMS]
    activity_terms = [word for word in words if word not in STOP_WORDS and len(word) > 2]
    people_terms = [word for word in words if word in SOCIAL_TERMS]

    if word_set & STUDY_TERMS:
        statement_type = "study"
    elif word_set & WORK_TERMS:
        statement_type = "work"
    elif word_set & PURCHASE_TERMS:
        statement_type = "purchase"
    elif word_set & TRAVEL_TERMS:
        statement_type = "travel"
    elif word_set & SOCIAL_TERMS:
        statement_type = "social"
    elif "home" in word_set or "house" in word_set or "room" in word_set:
        statement_type = "home"
    else:
        statement_type = "general"

    clear_signal_count = 0
    if len(words) >= 6:
        clear_signal_count += 1
    if times:
        clear_signal_count += 1
    if locations:
        clear_signal_count += 1
    if activity_terms:
        clear_signal_count += 1
    is_clear = clear_signal_count >= 2

    return StatementContext(
        statement_type=statement_type,
        times=times,
        locations=list(dict.fromkeys(locations)),
        activity_terms=list(dict.fromkeys(activity_terms[:8])),
        people_terms=list(dict.fromkeys(people_terms)),
        is_clear=is_clear,
    )


def generate_adaptive_questions(statement: str | None, total_questions: int = 4) -> List[str]:
    """Create follow-up questions from the user's first spoken statement."""
    context = extract_statement_context(statement)

    if not context.is_clear:
        questions = [
            "What activity or situation are you talking about?",
            "What time did it happen?",
            "Where were you at that time?",
            "Who was with you during that situation?",
        ]
    elif context.statement_type == "study":
        questions = [
            "What time did you start studying?",
            "Which subject or topic were you studying?",
            "Where were you studying?",
            "What time did you stop studying?",
        ]
    elif context.statement_type == "work":
        questions = [
            "What time did you start working on it?",
            "What task were you handling?",
            "Who else knew about this work?",
            "What time did you finish or stop working?",
        ]
    elif context.statement_type == "purchase":
        questions = [
            "Where did you make the purchase?",
            "What exactly did you buy or order?",
            "Who was with you at that time?",
            "What time did this happen?",
        ]
    elif context.statement_type == "travel":
        questions = [
            "What time did you leave?",
            "Which place did you visit?",
            "Who was with you?",
            "What time did you return?",
        ]
    elif context.statement_type == "social":
        questions = [
            "Who was with you during that time?",
            "Where did this happen?",
            "What did you do or discuss there?",
            "What time did the situation end?",
        ]
    elif context.statement_type == "home":
        questions = [
            "What were you doing at home?",
            "What time did you start that activity?",
            "Was anyone else present at that time?",
            "What time did that activity end?",
        ]
    else:
        questions = [
            "What time did this happen?",
            "Where were you during this situation?",
            "Who was with you at that time?",
            "What happened after that?",
        ]

    cleaned: List[str] = []
    for question in questions:
        if question not in cleaned:
            cleaned.append(question)

    fallback = [
        "What time did this happen?",
        "Where were you during this situation?",
        "Who was with you at that time?",
        "What happened after that?",
    ]
    for question in fallback:
        if len(cleaned) >= total_questions:
            break
        if question not in cleaned:
            cleaned.append(question)

    return cleaned[:total_questions]
