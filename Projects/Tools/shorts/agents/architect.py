"""
Kestrel Shorts - Architect
==========================

Turns the research into a BLUEPRINT: approach, hook, beats, ending, and a
length. It does not write narration (the Script Refiner sketches that).

The model chooses structure and how deeply each point is explained
("depth"). Code turns that into seconds, because small models give erratic
numbers:

    seconds = 8 (hook + payoff) + points x seconds-per-point(depth)

    name = 4.5 s/point    brief = 13 s    detailed = 22 s

A length the user asked for always wins. Weak research claims become
factual caveats automatically. If the model cannot produce a valid plan
after retries the error is raised; there is no template fallback.

    from agents.architect import run_architect, apply_to_context
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from agents._common import read_brief
from core import config, llm
from core.models._serde import from_dict
from core.models.research_result import ResearchResult
from core.models.story_plan import BeatPlan, DurationPlan, GlossaryTerm, ResearchRequest, StoryPlan
from creative.catalogs import narrative_catalog

logger = logging.getLogger(__name__)

STAGE = "architect"
MAX_BEATS = 8
MAX_RESEARCH_REQUESTS = 2
SECONDS_PER_POINT = {"name": 4.5, "brief": 13.0, "detailed": 22.0}
FRAME_SECONDS = 8.0     # hook + payoff

_STR_LIST = {"type": "array", "items": {"type": "string"}}
SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "narrative_approach": {"type": "string"},
        "premise": {"type": "string"},
        "hook": {"type": "string"},
        "narrative_goal": {"type": "string"},
        "beats": {"type": "array", "items": {"type": "object", "properties": {
            "summary": {"type": "string"}, "purpose": {"type": "string"}, "key_points": _STR_LIST,
            "link": {"type": "string", "enum": narrative_catalog.beat_link_names()},
                        "difficulty": {"type": "string", "enum": ["simple", "medium", "complex"]}},
            "required": ["summary", "purpose", "key_points", "link", "difficulty"]}},
        "hook_style": {"type": "string", "enum": narrative_catalog.hook_style_names()},
        "hook_candidate": {"type": "integer"},
        "glossary": {"type": "array", "items": {"type": "object", "properties": {
            "term": {"type": "string"}, "plain": {"type": "string"}, "analogy": {"type": "string"}},
            "required": ["term", "plain", "analogy"]}},
        "assumed_knowledge": _STR_LIST,
        "conclusion_beat": {"type": "integer"},
        "ending": {"type": "string"},
        "depth": {"type": "string", "enum": list(SECONDS_PER_POINT)},
        "depth_reason": {"type": "string"},
        "factual_caveats": _STR_LIST,
        "research_requests": {"type": "array", "items": {"type": "object", "properties": {
            "question": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["question", "reason"]}},
    },
    "required": ["title", "narrative_approach", "premise", "hook", "narrative_goal", "beats",
                 "ending", "depth", "depth_reason", "factual_caveats", "research_requests",
                 "hook_style", "hook_candidate", "glossary", "assumed_knowledge", "conclusion_beat"],
}


def _clean(items: Any, limit: int) -> List[str]:
    out: List[str] = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, str) and item.strip() and item.strip() not in out:
            out.append(item.strip())
        if len(out) >= limit:
            break
    return out


def _limits(info: Dict[str, Any]) -> tuple:
    """(max beats, max key points per beat), from the length the user wants."""
    seconds = info["user_seconds"]
    if seconds:
        beats = max(1, min(MAX_BEATS, round(config.clamp_duration(seconds) / 20)))
        return beats, (1 if seconds <= 45 else 3)
    if info["length_style"] == "brief":
        return 4, 1
    if info["length_style"] == "detailed":
        return MAX_BEATS, 5
    return MAX_BEATS, 3


def _build_prompt(research: ResearchResult, info: Dict[str, Any]) -> str:
    max_beats, max_points = _limits(info)
    if info["user_seconds"]:
        seconds = config.clamp_duration(info["user_seconds"])
        length = f"{seconds:.0f} seconds (requested by the user; about {config.word_budget(seconds)} spoken words)"
    else:
        style = info["length_style"]
        length = (f"The user wants it {style}." if style != "none"
                  else "Not specified; choose depth to fit the topic.")
    sizing = (f"at most {max_beats} beats and at most {max_points} key point(s) each; use fewer if the "
              "topic allows, and put the most important points first")

    findings = "\n".join(f"{i}. {f}" for i, f in enumerate(research.findings, 1)) or "(none)"
    claims = "\n".join(
        f"- [{'WEAK' if c.strength in ('low', 'insufficient') else c.strength}] {c.claim}"
        for c in research.claims) or "(none)"
    open_items = "\n".join(f"- {u}" for u in research.uncertainties[:5]) or "(none)"
    hooks = "\n".join(f"{i}. {h.fact}" for i, h in enumerate(research.hook_candidates, 1)) or "(none)"
    essentials = "\n".join(f"- {e}" for e in research.essentials) or "(none)"
    hook_styles = ", ".join(narrative_catalog.hook_style_names())
    links = ", ".join(narrative_catalog.beat_link_names())

    return f"""Task: design the blueprint for a short video from the research below. You decide how the information is organized. You do not write narration.

Now: {info['time_line']}
Topic: {info['topic']}
Angle: {info['angle'] or 'none given'}
The user's request, word for word: {info['request'] or info['topic']}
Length: {length}
Audience: {info['audience']} (reading level about grade {info['reading_grade']})
{('Channel (editorial lens, not a limit on facts): ' + info['channel_line']) if info['channel_line'] else ''}

Research summary:
{research.summary}

Findings:
{findings}

Claims and how well supported they are:
{claims}

Open questions in the research:
{open_items}

Hook candidates (researched facts that could open the video):
{hooks}

Essential points an explainer of this topic covers:
{essentials}

Rules:
- narrative_approach: choose what suits this topic (question-driven, chronological, cause and effect, myth versus evidence, comparison, investigation...) and name it.
- Cover only what the request needs: if it says 'main' or 'key', pick the most important items.
- Make the beats cover the essential points above first (when the user asked for the main parts of something, the beats ARE those parts); add extras only if the length allows.
- hook: one sentence for a general audience: a surprising, relatable idea (a scale comparison, a number tied to everyday life, a common misconception, or a what-if). Build it on a hook candidate when one fits and give its number ONLY in the hook_candidate field (0 if you use none); never write the number or the word hook_candidate inside the hook sentence. hook_style: one of {hook_styles}. Never open with a generic statement. Honest, no clickbait.
- The audience is the general public, not specialists. Famous people, places and basic ideas are already known to them: list them in assumed_knowledge and do not plan to explain them.
- glossary: up to 5 hard terms the video really needs. For each give plain words and, for a hard idea, an everyday analogy to say first. Use only terms your beats actually use, and give each hard idea its OWN everyday comparison: never reuse one comparison (a camera, say) for several terms. Leave out terms the video can avoid.
- beats: use as few as the topic needs ({sizing}). Each beat has a one-sentence summary, its purpose, key_points (the specific facts it covers, kept short), and a link saying how it follows the previous beat: {links}. The first beat is 'opens'. Prefer therefore, but, raises_question, callback and example; use and_then rarely.
- difficulty (per beat): simple = an everyday idea any viewer already understands (name it plainly); medium = a technical term the viewer may not know but can grasp with an example; complex = a hard mechanism or abstract idea that needs an everyday picture first. Judge by the audience above, not by the topic.
- conclusion_beat: the number of the beat where the main takeaway lands. It need not be the last beat; at most one short, light coda may follow it. Do not make a beat the conclusion just because it is last.
- ending: the main takeaway in one sentence, no stronger than the evidence. It must not repeat the hook.
- depth: how deeply each point is explained. "name" = just name or show each point, "brief" = one or two sentences each, "detailed" = a fuller explanation. depth_reason: one sentence.
- Use only facts from the findings. Never state a WEAK claim as fact; frame it as uncertain.
- factual_caveats: what the video must not overstate.
- research_requests: up to 2 questions the research could not answer that the video truly needs; [] if none.
Return JSON only."""


def plan_duration(beats: List[BeatPlan], depth: str, reason: str,
                  user_seconds: Optional[float]) -> DurationPlan:
    """Seconds from structure and depth (or the user's requested length)."""
    if user_seconds:
        return DurationPlan(config.clamp_duration(user_seconds), depth,
                            "Length requested by the user.")
    points = sum(max(1, len(b.key_points)) for b in beats)
    wanted = FRAME_SECONDS + points * SECONDS_PER_POINT[depth]
    note = f"{points} point(s) at '{depth}' depth. {reason}".strip()
    if wanted > config.MAX_VIDEO_SECONDS:
        note += (f" The plan needs about {wanted:.0f}s but the ceiling is "
                 f"{config.MAX_VIDEO_SECONDS:.0f}s: the script must compress.")
    return DurationPlan(config.clamp_duration(wanted), depth, note)


def run_architect(
    research: Any,
    context: Any = None,
    context_path: Optional[str] = None,
) -> StoryPlan:
    if isinstance(research, dict):
        research = from_dict(ResearchResult, research)
    if not isinstance(research, ResearchResult):
        raise TypeError("run_architect() requires a ResearchResult.")
    if not research.topic.strip():
        raise ValueError("ResearchResult topic cannot be empty.")

    info = read_brief(context, context_path)
    info["topic"] = research.topic
    max_beats, max_points = _limits(info)

    def validate(data: Dict[str, Any]) -> Dict[str, Any]:
        for key in ("title", "hook", "ending"):
            if not str(data.get(key, "")).strip():
                raise ValueError(f"{key} must not be empty.")
        beats = [b for b in data.get("beats", []) if isinstance(b, dict) and str(b.get("summary", "")).strip()]
        if not beats:
            raise ValueError("beats must contain at least one beat with a summary.")
        data["beats"] = beats[:max_beats]
        return data

    data = llm.generate_json(STAGE, _build_prompt(research, info), schema=SCHEMA, validate=validate)

    beats = [
        BeatPlan(beat_id=f"beat_{i}", summary=b["summary"].strip(),
                 purpose=str(b.get("purpose", "")).strip(),
                 difficulty=b.get("difficulty") if b.get("difficulty") in ("simple", "medium", "complex") else "medium",
                 key_points=_clean(b.get("key_points"), max_points),
                 link="opens" if i == 1 else (b.get("link") if b.get("link") in narrative_catalog.BEAT_LINKS
                                              and b.get("link") != "opens" else "and_then"))
        for i, b in enumerate(data["beats"], 1)
    ]
    glossary = [GlossaryTerm(term=g["term"].strip(), plain=g["plain"].strip(), analogy=str(g.get("analogy", "")).strip())
                for g in data.get("glossary", []) if isinstance(g, dict)
                and str(g.get("term", "")).strip() and str(g.get("plain", "")).strip()][:5]
    pick = data.get("hook_candidate")
    hook_fact = (research.hook_candidates[pick - 1].fact
                 if isinstance(pick, int) and 1 <= pick <= len(research.hook_candidates) else "")
    conclusion = data.get("conclusion_beat")
    conclusion = conclusion if isinstance(conclusion, int) and 1 <= conclusion <= len(beats) else len(beats)

    caveats = _clean(data.get("factual_caveats"), 6)
    for claim in research.claims:                       # weak claims are never silently dropped
        if claim.strength in ("low", "insufficient"):
            caveats.append(f"Weakly supported: {claim.claim}")
    caveats = list(dict.fromkeys(caveats))

    requests = [
        ResearchRequest(question=r["question"].strip(), reason=str(r.get("reason", "")).strip())
        for r in data.get("research_requests", [])
        if isinstance(r, dict) and str(r.get("question", "")).strip()
    ][:MAX_RESEARCH_REQUESTS]

    last = llm.LAST_RESULT
    return StoryPlan(
        title=data["title"].strip(),
        premise=str(data.get("premise", "")).strip(),
        hook=re.sub(r"\s*\(?\bhook_candidate\b\s*\d*\)?", "", data["hook"]).strip(),
        narrative_goal=str(data.get("narrative_goal", "")).strip(),
        ending=data["ending"].strip(),
        narrative_approach=str(data.get("narrative_approach", "")).strip(),
        hook_style=data.get("hook_style", ""),
        hook_fact=hook_fact,
        glossary=glossary,
        assumed_knowledge=_clean(data.get("assumed_knowledge"), 6),
        conclusion_beat_id=f"beat_{conclusion}",
        beats=beats,
        duration=plan_duration(beats, data["depth"], str(data.get("depth_reason", "")).strip(),
                               info["user_seconds"]),
        factual_caveats=caveats,
        research_requests=requests,
        metadata={"agent": STAGE, "prompt_tokens": getattr(last, "prompt_tokens", None),
                  "output_tokens": getattr(last, "output_tokens", None)},
    )


def apply_to_context(ctx: Any, plan: StoryPlan) -> None:
    """Record the planned length in the context, unless the user already set one."""
    if plan.duration and ctx.duration.get("source") != "user":
        ctx.set_duration(plan.duration.target_seconds, "architect", plan.duration.reason)


generate_story_plan = run_architect
execute_architect = run_architect


if __name__ == "__main__":
    import sys
    from agents.researcher import run_research
    topic = " ".join(sys.argv[1:]) or "The Antikythera mechanism"
    found = run_research(topic)
    result = run_architect(found)
    print(f"\n{result.title}  [{result.narrative_approach}]")
    print("Hook:", result.hook)
    for b in result.beats:
        print(f"  {b.beat_id}: {b.summary}  ({len(b.key_points)} points)")
    print("Ending:", result.ending)
    print("Length:", result.duration)
    print("Caveats:", result.factual_caveats)