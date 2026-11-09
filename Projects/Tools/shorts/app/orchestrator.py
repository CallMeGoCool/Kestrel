"""
Kestrel Shorts - Orchestrator
=============================

The receptionist. It reads the user's request and decides:

  - which registered channel it belongs to (from the registry's channel menu),
  - the topic and angle,
  - how long the video should be, when the user said so,
  - how fresh the information must be ("latest", "last 24 hours"),
  - whether it needs to ask the user a question first.

It makes ONE small LLM call. Numbers (a requested length, a freshness
window) are read from the text by code, because small models misread them.

It returns updates and does not write files; `apply_to_context()` copies
them into a RunContext (the workflow controller persists it).

    python -m app.orchestrator "Explain how Roman roads were built"
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from core import config, llm
from core.registry_loader import RegistryError, RegistryLoader

logger = logging.getLogger(__name__)

STAGE = "orchestrator"

_UNITS = {"s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
          "min": 60, "mins": 60, "minute": 60, "minutes": 60}


# ----------------------------------------------------------------------
# Deterministic extraction (numbers are read by code, not by the model)
# ----------------------------------------------------------------------

_DURATION = re.compile(
    r"(\d+(?:\.\d+)?)\s*-?\s*(seconds?|secs?|minutes?|mins?)\b", re.IGNORECASE
)
_NOT_A_LENGTH_BEFORE = re.compile(r"\b(last|past|within|previous|every|per|over the)\s+$", re.IGNORECASE)
_NOT_A_LENGTH_AFTER = re.compile(r"^\s*(ago|old)\b", re.IGNORECASE)


_LENGTH_AFTER = re.compile(r"^[\s-]*(?:long\b|video\b|short\b|clip\b|explainer\b|reel\b|version\b|summary\b|overview\b|"
                           r"animation\b|on\b|about\b|of\b|or\s+(?:less|so)\b|max\b|tops\b|total\b)", re.IGNORECASE)
_LENGTH_END = re.compile(r"^\s*(?:$|[.;!?])")
_LENGTH_BEFORE = re.compile(r"(?:\bin|\bunder|\baround|\babout|\bapprox\w*|\broughly|\bmake it|\bkeep it|\bat most|"
                            r"\bmax(?:imum)?|\bup to|\blasting|\blength|\bduration|\bruntime|\bnearly|\bwithin)\s+(?:about\s+|around\s+|roughly\s+)?(?:a\s+)?$",
                            re.IGNORECASE)


def _is_length(text: str, start: int, end: int, numeric: bool = True) -> bool:
    """A number of seconds is the video's length when the words around it say so ('in about 45 seconds',
    '20 second video'); '30 second rule' or 'a 5 minute walk' belong to the topic."""
    before = text[max(0, start - 32):start]
    after = text[end:end + 28]
    return bool(_LENGTH_BEFORE.search(before) or _LENGTH_AFTER.match(after)
                or (numeric and (start < 2 or _LENGTH_END.match(after))))


def extract_duration_seconds(text: str) -> Optional[float]:
    """A length the user asked for ('20 second video', 'in about a minute'), else None."""
    for match in _DURATION.finditer(text):
        if _NOT_A_LENGTH_BEFORE.search(text[max(0, match.start() - 14):match.start()]):
            continue   # "news from the last 30 minutes" is freshness, not length
        if _NOT_A_LENGTH_AFTER.match(text[match.end():match.end() + 8]):
            continue
        if not _is_length(text, match.start(), match.end()):
            continue   # "explain the 30 second rule" is the topic, not the length
        return float(match.group(1)) * _UNITS[match.group(2).lower()]
    for pattern, seconds in ((r"\bhalf\s+(?:a\s+)?minute\b|\bhalf-minute\b", 30.0), (r"\b(?:a|one)[\s-]+minute\b", 60.0)):
        match = re.search(pattern, text, re.IGNORECASE)
        if match and (seconds == 30.0 or _is_length(text, match.start(), match.end(), numeric=False)):
            return seconds
    return None


def _extract_duration_seconds_old(text: str) -> Optional[float]:
    """A length the user asked for ('20 second video', 'a minute'), else None."""
    for match in _DURATION.finditer(text):
        if _NOT_A_LENGTH_BEFORE.search(text[max(0, match.start() - 14):match.start()]):
            continue   # "news from the last 30 minutes" is freshness, not length
        if _NOT_A_LENGTH_AFTER.match(text[match.end():match.end() + 8]):
            continue
        return float(match.group(1)) * _UNITS[match.group(2).lower()]
    if re.search(r"\bhalf\s+(?:a\s+)?minute\b|\bhalf-minute\b", text, re.IGNORECASE):
        return 30.0
    if re.search(r"\b(?:a|one)\s+minute\b", text, re.IGNORECASE):
        return 60.0
    return None


_FRESHNESS = (
    ("24h", re.compile(r"\b(?:last|past)\s+\d+\s*(?:minutes?|mins?|hours?)\b|\b(?:last|past)\s+(?:day|24\s*hours?)\b|\btoday\b|\btonight\b|\bbreaking\b", re.I)),
    ("7d", re.compile(r"\b(?:this|last|past)\s+week\b|\b7\s*days\b", re.I)),
    ("30d", re.compile(r"\b(?:this|last|past)\s+month\b|\b30\s*days\b", re.I)),
)


def extract_freshness(text: str) -> Optional[str]:
    """An explicit freshness window in the text, else None."""
    for window, pattern in _FRESHNESS:
        if pattern.search(text):
            return window
    return None


_NO_AVATAR = re.compile(
    r"\b(?:no|without|skip)\s+(?:any\s+)?(?:avatars?|characters?\s+on\s+screen)\b|\bfaceless\b", re.IGNORECASE)


def extract_avatar_preference(text: str) -> Optional[str]:
    """'none' when the user asks for a faceless video, else None (use the channel's level)."""
    return "none" if _NO_AVATAR.search(text) else None


_AUDIENCE = (
    ("kids", re.compile(r"\b(?:for|aimed at|targeting)\s+(?:young\s+)?(?:kids|children|little ones)\b|\bkid[- ]friendly\b", re.I)),
    ("teens", re.compile(r"\bfor\s+teen(?:ager)?s?\b", re.I)),
    ("beginners", re.compile(r"\b(?:for|aimed at)\s+(?:complete\s+)?beginners\b|\beli5\b|\blike i'?m (?:five|5)\b", re.I)),
    ("experts", re.compile(r"\b(?:for|aimed at)\s+(?:experts|professionals|specialists|researchers)\b", re.I)),
)


def extract_audience_override(text: str) -> Optional[str]:
    """kids | teens | beginners | experts when the request names an audience, else None."""
    for level, pattern in _AUDIENCE:
        if pattern.search(text):
            return level
    return None


_STYLES = (
    ("stylized_illustration", re.compile(r"\b(?:cartoon|cartoony|illustrat\w*|anime|comic|animated|hand[- ]drawn)\b", re.I)),
    ("cinematic_realism", re.compile(r"\b(?:realistic|photo[- ]?realistic|cinematic|live[- ]action)\b", re.I)),
    ("documentary", re.compile(r"\bdocumentary\b", re.I)),
    ("graphic_conceptual", re.compile(r"\b(?:infographic|minimali\w+|flat design)\b", re.I)),
)


_STYLE_WORDS = (
    ("stylized_illustration", r"cartoon|cartoony|illustrat(?:ed|ion|ive)|anime|comic(?:[- ]book)?|hand[- ]drawn|animated"),
    ("cinematic_realism", r"realistic|photo[- ]?realistic|cinematic|live[- ]action"),
    ("documentary", r"documentary"),
    ("graphic_conceptual", r"infographic|minimali(?:st|stic)|flat[- ]design"),
)


def _style_pattern(words: str) -> "re.Pattern[str]":
    # A look is requested ('in a cartoon style', 'make it cinematic', 'anime-style'), not merely mentioned
    # ('a documentary about X', 'the illustration of the heart').
    return re.compile(rf"\b(?:in|as|with|using|make it|made in|done in)\s+(?:an?\s+|the\s+)?(?:{words})\b"
                      rf"|\b(?:{words})[- ]?(?:style|looking|look|aesthetic|visuals?|animation|vibe)\b", re.IGNORECASE)


_STYLE_PATTERNS = tuple((style, _style_pattern(words)) for style, words in _STYLE_WORDS)


def extract_visual_style_override(text: str) -> Optional[str]:
    """A visual_style_catalog id when the request asks for a look, else None (use the channel default)."""
    for style, pattern in _STYLE_PATTERNS:
        if pattern.search(text):
            return style
    return None


_ADULT = re.compile(r"\b(?:18\s*\+|18\s*plus|adults?[- ]only|adult\s+content|nsfw|mature\s+audiences?)(?!\w)", re.IGNORECASE)


def extract_content_rating(text: str) -> Optional[str]:
    """'adult' only when the user's own words ask for an 18+ video, else None (general)."""
    return "adult" if _ADULT.search(text) else None


def _extract_visual_style_override_old(text: str) -> Optional[str]:
    """A visual_style_catalog id when the request names a look, else None (use the channel default)."""
    for style, pattern in _STYLES:
        if pattern.search(text):
            return style
    return None


# ----------------------------------------------------------------------
# Prompt and schema
# ----------------------------------------------------------------------

def _schema(channel_ids: List[str]) -> Dict[str, Any]:
    text = {"type": "string"}
    return {
        "type": "object",
        "properties": {
            "channel_id": {"type": "string", "enum": channel_ids},
            "channel_rationale": text,
            "topic": text,
            "angle": text,
            "length_style": {"type": "string", "enum": ["none", "brief", "detailed"]},
            "freshness_window": {"type": "string", "enum": ["none", "24h", "7d", "30d"]},
            "needs_clarification": {"type": "boolean"},
            "clarification_question": text,
        },
        "required": ["channel_id", "channel_rationale", "topic", "angle", "length_style",
                     "freshness_window", "needs_clarification", "clarification_question"],
    }


def _build_prompt(prompt: str, menu: List[Dict[str, str]], time_line: str,
                  answers: List[str]) -> str:
    channels = "\n".join(
        f"- {c['id']}: {c['name']}. {c['description'][:200]}" for c in menu
    )
    follow_up = ""
    if answers:
        follow_up = "\nThe user's answers to earlier questions:\n" + "\n".join(f"- {a}" for a in answers)
    return f"""Now: {time_line}

Channels (pick the one whose description best fits the request):
{channels}

User request:
{prompt}
{follow_up}
Decide:
- channel_id: the best-fitting channel above. A request that is not for any channel (a personal or one-off video) belongs to a channel described as personal or one-off, if one exists.
- channel_rationale: one short sentence.
- topic: the subject, as a short phrase.
- angle: one sentence on the intended angle, or "" if the user gave none.
- length_style: "brief" if the user wants it short or quick, "detailed" if thorough or in depth, otherwise "none". Never guess a number of seconds.
- freshness_window: how recent the information must be ("24h", "7d", "30d"), or "none" for timeless topics.
- needs_clarification: true only if you cannot tell what the video is about or which channel fits. Then write ONE short question in clarification_question; otherwise "".

Return JSON only."""


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------

def _read_input(original_prompt: Any, context: Any, context_path: Any):
    """Return (prompt, explicit_preferences, time_line)."""
    if context is None and context_path:
        from core.run_context import RunContext
        context = RunContext.load(context_path)
    if context is None and not isinstance(original_prompt, str):
        context = original_prompt            # run_orchestrator(run_context)
    if context is not None:
        if hasattr(context, "time_line"):    # a RunContext
            return context.original_prompt, context.explicit_preferences, context.time_line
        if isinstance(context, dict):
            stamp = context.get("time_context") or config.now_stamp()
            return (str(context.get("original_prompt", "")),
                    context.get("explicit_preferences") or {}, config.describe_stamp(stamp))
    return original_prompt, {}, config.describe_stamp(config.now_stamp())


def run_orchestrator(
    original_prompt: Any = "",
    available_channels: Optional[List[Dict[str, Any]]] = None,
    explicit_preferences: Optional[Dict[str, Any]] = None,
    constraints: Optional[List[str]] = None,
    timeout: Optional[float] = None,
    *,
    context: Any = None,
    context_path: Optional[str] = None,
    registry: Optional[RegistryLoader] = None,
    enable_preliminary_search: bool = False,   # accepted for old callers; ignored
) -> Dict[str, Any]:
    """Interpret a request and choose its channel. See the module docstring."""
    prompt, preferences, time_line = _read_input(original_prompt, context, context_path)
    if explicit_preferences:
        preferences = {**preferences, **explicit_preferences}
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("The request must be a non-empty string.")
    prompt = prompt.strip()

    registry = registry or RegistryLoader()
    menu = registry.channel_menu()
    if available_channels:
        allowed = {c.get("channel_id") for c in available_channels if isinstance(c, dict)}
        menu = [c for c in menu if c["id"] in allowed]
    if not menu:
        raise RegistryError("No channels are registered (or none are available).")
    ids = [c["id"] for c in menu]

    answers = [str(a.get("response", "")).strip()
               for a in preferences.get("clarification_responses", []) if isinstance(a, dict)]
    answers = [a for a in answers if a]
    # Numbers come from everything the user has said, including answers.
    user_text = " ".join([prompt, *answers])

    def validate(data: Dict[str, Any]) -> Dict[str, Any]:
        if data.get("channel_id") not in ids:
            raise ValueError(f"channel_id must be one of {ids}.")
        if not str(data.get("topic", "")).strip():
            raise ValueError("topic must not be empty.")
        return data

    data = llm.generate_json(
        STAGE, _build_prompt(prompt, menu, time_line, answers),
        schema=_schema(ids), validate=validate, idle_timeout=timeout,
    )

    seconds = extract_duration_seconds(user_text)
    freshness = extract_freshness(user_text) or data.get("freshness_window", "none")
    style = data.get("length_style", "none")

    needs_clarification = bool(data.get("needs_clarification"))
    question = str(data.get("clarification_question", "")).strip()
    if needs_clarification and not question:
        names = ", ".join(c["name"] for c in menu)
        question = f"What would you like the video to be about? (Channels available: {names}.)"

    # The chosen channel must resolve cleanly (worlds, characters, voices, avatars).
    channel_id = data["channel_id"]
    try:
        registry.resolve_channel(channel_id)
    except RegistryError as exc:         # a configuration problem is not a question for the user: stop loudly
        raise RegistryError(f"Channel {channel_id!r} has a configuration problem: {exc}") from exc

    topic = data["topic"].strip()
    last = llm.LAST_RESULT
    return {
        "interpretation": {
            "topic": topic,
            "angle": str(data.get("angle", "")).strip(),
            "length_style": style,
            "freshness_window": freshness,
            "avatar_presence": extract_avatar_preference(user_text),
            "audience_override": extract_audience_override(user_text),
            "visual_style_override": extract_visual_style_override(user_text),
            "content_rating": extract_content_rating(user_text),
            "original_prompt": prompt,
        },
        "channel_selection": {
            "channel_id": channel_id,
            "rationale": str(data.get("channel_rationale", "")).strip(),
        },
        "duration_request": {"seconds": seconds, "style": style},
        "research_brief": {
            "topic": topic,
            "angle": str(data.get("angle", "")).strip(),
            "freshness_window": freshness,
        },
        "clarification": {"required": needs_clarification,
                          "question": question if needs_clarification else None},
        # Flat keys the workflow runner already looks for:
        "requires_clarification": needs_clarification,
        "clarification_question": question if needs_clarification else None,
        "metadata": {
            "agent": STAGE,
            "model": getattr(last, "model", None),
            "prompt_tokens": getattr(last, "prompt_tokens", None),
            "output_tokens": getattr(last, "output_tokens", None),
        },
    }


def apply_to_context(ctx: Any, result: Dict[str, Any]) -> None:
    """Copy an orchestrator result into a RunContext (caller saves it)."""
    ctx.update_section("interpretation", result["interpretation"])
    ctx.update_section("research_brief", result["research_brief"])
    ctx.update_section("clarification", result["clarification"])

    selection = result["channel_selection"]
    if selection.get("channel_id"):
        ctx.select_channel(selection["channel_id"], selection.get("rationale", ""))

    seconds = result["duration_request"].get("seconds")
    if seconds:
        info = ctx.set_duration(seconds, "user", "Length requested in the prompt.")
        if info["clamped"]:
            ctx.update_section("duration", {
                "reason": f"Requested {seconds:.0f}s, adjusted to the allowed range."})


execute_orchestrator = run_orchestrator


if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) or "Explain how the Antikythera mechanism worked"
    print(json.dumps(run_orchestrator(text), indent=2, ensure_ascii=False))