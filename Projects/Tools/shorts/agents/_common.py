"""Helpers shared by the agents."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from core import config

logger = logging.getLogger(__name__)


def read_brief(context: Any = None, context_path: Optional[str] = None) -> Dict[str, Any]:
    """
    What most agents need to know about the run, from a RunContext, a dict
    or a context file: topic, angle, freshness, length wishes, the current
    time line, and a one-line description of the chosen channel.
    """
    if context is None and context_path:
        from core.run_context import RunContext
        context = RunContext.load(context_path)
    data = context.to_dict() if hasattr(context, "to_dict") else (context or {})

    interpretation = data.get("interpretation") or {}
    brief = data.get("research_brief") or {}
    selection = data.get("channel_selection") or {}
    duration = data.get("duration") or {}
    channel_id = selection.get("channel_id") or selection.get("selected_channel_id")

    channel_line = ""
    audience, reading_grade, avatar_presence, default_style = "", 0, "", ""
    if channel_id:
        try:
            from core.registry_loader import RegistryLoader
            ch = RegistryLoader().get_channel(channel_id)
            style = ", ".join(ch.get("storytelling_style", []))
            channel_line = f"{ch.get('name', channel_id)}: {ch.get('description', '')} Style: {style}".strip()
            audience = str(ch.get("audience") or "")
            reading_grade = int(ch.get("reading_grade") or 0)
            avatar_presence = str(ch.get("avatar_presence") or "")
            default_style = str(ch.get("default_visual_style") or "")
        except Exception as exc:
            logger.warning("Could not read channel %r: %s", channel_id, exc)

    # A request such as "for kids" overrides the channel's audience for this video.
    from creative.catalogs import narrative_catalog
    level = narrative_catalog.AUDIENCE_LEVELS.get(interpretation.get("audience_override") or "")
    if level:
        audience, reading_grade = level["description"], level["reading_grade"]
    audience = audience or narrative_catalog.AUDIENCE_LEVELS["general"]["description"]
    reading_grade = reading_grade or narrative_catalog.AUDIENCE_LEVELS["general"]["reading_grade"]
    avatar_presence = interpretation.get("avatar_presence") or avatar_presence or "cameo"

    user_seconds = duration.get("target_seconds") if duration.get("source") == "user" else None
    return {
        "ctx": context if hasattr(context, "run_directory") else None,
        "channel_id": channel_id,
	"request": data.get("original_prompt") or interpretation.get("original_prompt") or "",
        "topic": brief.get("topic") or interpretation.get("topic") or data.get("original_prompt") or "",
        "angle": interpretation.get("angle") or brief.get("angle") or "",
        "freshness": interpretation.get("freshness_window") or brief.get("freshness_window") or "none",
        "length_style": interpretation.get("length_style") or "none",
        "user_seconds": user_seconds,
        "time_line": getattr(context, "time_line", None)
                     or config.describe_stamp(data.get("time_context") or config.now_stamp()),
        "channel_line": channel_line,
        "audience": audience,
        "reading_grade": reading_grade,
        "avatar_presence": avatar_presence,
        "visual_style": interpretation.get("visual_style_override") or default_style,
    }