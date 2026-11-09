"""
Narrative options the Architect and Story Director choose from: hook styles,
how a beat follows the previous one, how a video may close, and audience
levels. Names and hints only; code and prompts decide what to do with them.
"""
from typing import Any, Dict, List

HOOK_STYLES: Dict[str, str] = {
    "scale_comparison": "A size or scale made relatable ('so big the whole solar system fits inside, with room to spare').",
    "relatable_number": "A surprising number tied to everyday experience.",
    "question_reveal": "Ask a question the viewer wants answered, then promise the reveal.",
    "myth_bust": "Start from a common belief that turns out to be wrong or incomplete.",
    "what_if": "Pose a what-if that the topic can answer.",
    "surprising_fact": "A single striking fact, stated plainly.",
}

# weak=True marks the flattest link; too many in a row feels episodic.
BEAT_LINKS: Dict[str, Dict[str, Any]] = {
    "opens": {"description": "The first beat.", "weak": False},
    "therefore": {"description": "Follows as a consequence of the previous beat.", "weak": False},
    "but": {"description": "Turns against or complicates the previous beat.", "weak": False},
    "raises_question": {"description": "The previous beat leaves a question; this one takes it up.", "weak": False},
    "callback": {"description": "Brings back something shown or said earlier.", "weak": False},
    "example": {"description": "Illustrates the previous beat with a concrete case.", "weak": False},
    "and_then": {"description": "Simply the next thing. Use rarely.", "weak": True},
}

CLOSING_STYLES: Dict[str, str] = {
    "callback": "Return to the hook or an earlier image.",
    "question_to_viewer": "Leave the viewer with a question to think about.",
    "reflection": "A short reflective line.",
    "light_joke": "A light joke in the character's own style.",
    "teaser": "Hint at a related topic.",
    "none": "No coda: the video ends on its conclusion.",
}

AUDIENCE_LEVELS: Dict[str, Dict[str, Any]] = {
    "kids": {"description": "children about 8 to 12: very simple words, short sentences, vivid everyday comparisons", "reading_grade": 5},
    "teens": {"description": "teenagers: clear and conversational", "reading_grade": 8},
    "general": {"description": "the general public with a mild interest: famous people and basic ideas are already known", "reading_grade": 8},
    "beginners": {"description": "complete beginners: every idea is put in plain words", "reading_grade": 6},
    "experts": {"description": "viewers who already know the field: technical terms are fine", "reading_grade": 12},
}


def hook_style_names() -> List[str]:
    return list(HOOK_STYLES)


def beat_link_names() -> List[str]:
    return list(BEAT_LINKS)


def closing_style_names() -> List[str]:
    return list(CLOSING_STYLES)


def is_weak_link(link: str) -> bool:
    return bool(BEAT_LINKS.get(link, {}).get("weak"))