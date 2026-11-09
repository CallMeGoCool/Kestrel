from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Any, Dict, List, Optional

from core.models._serde import coerce_fields


@dataclass
class BeatPlan:
    """One planned beat. A beat is a moment in the story, with no fixed timing."""

    beat_id: str
    summary: str
    purpose: str = ""
    key_points: List[str] = field(default_factory=list)
    link: str = ""          # how it follows the previous beat (beat_link catalog)
    difficulty: str = ""    # simple | medium | complex: how much explaining the beat needs


@dataclass
class GlossaryTerm:
    """A hard term the video needs, with plain words (and an analogy to say first)."""

    term: str
    plain: str
    analogy: str = ""


@dataclass
class DurationPlan:
    """The Architect's proposed length (the Story Director may adjust it)."""

    target_seconds: float
    depth: str = "brief"            # name | brief | detailed
    reason: str = ""


@dataclass
class ResearchRequest:
    question: str
    reason: str = ""


@dataclass
class StoryPlan:
    """Architect output: the blueprint the Script Refiner sketches from."""

    title: str
    premise: str = ""
    hook: str = ""
    narrative_goal: str = ""
    ending: str = ""
    narrative_approach: str = ""

    hook_style: str = ""                 # hook_style catalog
    hook_fact: str = ""                  # the researched fact the hook builds on, if any
    glossary: List[GlossaryTerm] = field(default_factory=list)
    assumed_knowledge: List[str] = field(default_factory=list)   # known to the audience: do not explain
    conclusion_beat_id: str = ""         # where the main takeaway lands (need not be the last beat)

    beats: List[BeatPlan] = field(default_factory=list)
    duration: Optional[DurationPlan] = None
    factual_caveats: List[str] = field(default_factory=list)
    research_requests: List[ResearchRequest] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    # Compatibility: StoryPlan(story_beats=["...", "..."]) builds beats.
    story_beats: InitVar[Optional[List[str]]] = None

    def __post_init__(self, story_beats: Optional[List[str]] = None) -> None:
        coerce_fields(self)
        if story_beats and not self.beats:
            self.beats = [
                BeatPlan(beat_id=f"beat_{i}", summary=text)
                for i, text in enumerate(story_beats, 1)
            ]


# Reading `plan.story_beats` returns the beat summaries, so older code keeps
# working. Defined after the class so the dataclass keeps its InitVar.
StoryPlan.story_beats = property(lambda self: [b.summary for b in self.beats])