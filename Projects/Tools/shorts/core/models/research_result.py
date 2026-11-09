from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from core.models._serde import coerce_fields


@dataclass
class Claim:
    """A factual claim and the evidence behind it."""

    claim: str
    source_urls: List[str] = field(default_factory=list)
    strength: str = "insufficient"   # high | medium | low | insufficient
    notes: str = ""


@dataclass
class HookCandidate:
    """A researched fact that could open the video (surprising, relatable)."""

    fact: str
    source_urls: List[str] = field(default_factory=list)
    why: str = ""


@dataclass
class ResearchResult:
    """
    Researcher output. The research plan and raw search statuses are kept
    in the run folder (stages/researcher.json), not here.

    `sources` lists only URLs the model actually read.
    """

    topic: str
    summary: str = ""
    findings: List[str] = field(default_factory=list)
    claims: List[Claim] = field(default_factory=list)
    hook_candidates: List[HookCandidate] = field(default_factory=list)
    essentials: List[str] = field(default_factory=list)   # the main points a good explainer covers
    uncertainties: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    is_niche: bool = False

    status: str = "completed"        # completed | completed_with_warnings | failed
    warnings: List[str] = field(default_factory=list)
    queries_executed: List[str] = field(default_factory=list)

    # When and where the research was done, and any freshness requirement
    # ("24h", "7d", ...) so later stages can judge how current it is.
    researched_at: str = ""
    location: str = ""
    freshness_window: str = ""

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coerce_fields(self)