"""
Kestrel Shorts - Script Refiner
===============================

Writes the SKETCH: neutral, plain spoken narration that lays out every
beat of the Architect's plan within the word budget. The Story Director
later gives it character voices and final length, so this stage only
settles the content.

It also chooses the WORLD (the channel's sub-genre and tone) from the
channel's world menu. A channel with one world needs no model call.

Checks done in code, not by the model:
  - every plan beat has a section, in order, with its beat_id;
  - total length near the word budget (hard limits 45%-170%; the Story
    Director fixes the rest, recorded as `length_status`);
  - numbers in the sketch that do not appear in the research are listed
    in metadata["unsupported_numbers"].

    python -m agents.script_refiner "How Roman roads were built" [--channel history]
    python -m agents.script_refiner --from-run <path to shared_context.json>
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agents._common import read_brief
from core import config, llm
from core.models._serde import from_dict
from core.models.research_result import ResearchResult
from core.models.script import Script, ScriptSection
from core.models.story_plan import StoryPlan
from core.models.world import World
from core.registry_loader import RegistryLoader

logger = logging.getLogger(__name__)

STAGE = "script_refiner"
HARD_LOW, HARD_HIGH = 0.3, 1.7     # accepted range of the word budget; the Story Director fixes the rest
MIN_BEAT_WORDS = 8

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_NUMBER_WORD = re.compile(r"\b(hundred|thousand|million|billion|trillion)s?\b", re.IGNORECASE)
_EXPLAIN_AS = {
    "simple": "state it plainly; no analogy",
    "medium": "use the proper term with one short concrete example",
    "complex": "give an everyday picture or analogy FIRST, then the term if it is needed again",
}

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _word_budget(plan: StoryPlan, ctx: Any) -> int:
    if ctx is not None and getattr(ctx, "word_budget", None):
        return int(ctx.word_budget)
    if plan.duration:
        return config.word_budget(plan.duration.target_seconds)
    return config.word_budget(60)


def _allocate_words(plan: StoryPlan, budget: int) -> Dict[str, int]:
    """Split the budget over beats by weight (key points; hook and ending add weight)."""
    weights = {b.beat_id: max(1, len(b.key_points)) for b in plan.beats}
    first, last = plan.beats[0].beat_id, plan.beats[-1].beat_id
    weights[first] += 1
    weights[last] += 1
    total = sum(weights.values())
    return {bid: max(MIN_BEAT_WORDS, round(budget * w / total)) for bid, w in weights.items()}


def unsupported_numbers(text: str, research: ResearchResult) -> List[str]:
    """Numbers in `text` that appear nowhere in the research."""
    corpus = " ".join([research.topic, research.summary, *research.findings,
                       *(c.claim for c in research.claims),
                       *(h.fact for h in research.hook_candidates)])
    corpus = re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", " ", corpus)   # citation markers are not facts
    known = {n.replace(",", "") for n in _NUMBER.findall(corpus)}
    spelled = ({w.lower() for w in _NUMBER_WORD.findall(text)}
               - {w.lower() for w in _NUMBER_WORD.findall(corpus)})
    return sorted({n.replace(",", "") for n in _NUMBER.findall(text)} - known) + sorted(spelled)

def _world_block(world: World) -> str:
    conventions = "; ".join(world.storytelling_conventions[:4])
    return (f'World "{world.name}": {world.description}\n'
            f"Atmosphere: {world.atmosphere or 'not specified'}\n"
            f"Storytelling conventions: {conventions or 'none'}")


# ----------------------------------------------------------------------
# World choice
# ----------------------------------------------------------------------

def choose_world(info: Dict[str, Any], plan: StoryPlan, registry: RegistryLoader,
                 channel_id: str) -> Tuple[str, str]:
    """Pick a world from the channel's menu. Returns (world_id, rationale)."""
    menu = registry.world_menu(channel_id)
    if not menu:
        raise ValueError(f"Channel {channel_id!r} has no worlds registered.")
    if len(menu) == 1:
        return menu[0]["id"], "The only world registered for this channel."

    ids = [w["id"] for w in menu]
    listing = "\n".join(f"- {w['id']}: {w['name']}. {w['description'][:200]}" for w in menu)
    prompt = f"""Task: pick the world (sub-genre and tone) that best suits this video.

Topic: {info['topic']}
Angle: {info['angle'] or 'none given'}
Video: {plan.title} ({plan.narrative_approach}). Hook: {plan.hook}

Worlds:
{listing}

Pick the single best fit and give a one-sentence reason. Return JSON only."""
    schema = {"type": "object",
              "properties": {"world_id": {"type": "string", "enum": ids},
                             "rationale": {"type": "string"}},
              "required": ["world_id", "rationale"]}

    def validate(data: Dict[str, Any]) -> Dict[str, Any]:
        if data.get("world_id") not in ids:
            raise ValueError(f"world_id must be one of {ids}.")
        return data

    data = llm.generate_json(STAGE, prompt, schema=schema, validate=validate, num_predict=200)
    return data["world_id"], str(data.get("rationale", "")).strip()


# ----------------------------------------------------------------------
# Sketch
# ----------------------------------------------------------------------

def _build_prompt(plan: StoryPlan, research: ResearchResult, info: Dict[str, Any],
                  world: World, budget: int, targets: Dict[str, int]) -> str:
    beats = "\n".join(
        f"- {b.beat_id} (at most {targets[b.beat_id]} words, link: {b.link or 'follows on'}): {b.summary} | purpose: {b.purpose} | "
        f"key points: {'; '.join(b.key_points) or 'none'} | explain as: {_EXPLAIN_AS.get(b.difficulty or 'medium', _EXPLAIN_AS['medium'])}" for b in plan.beats)
    facts = "\n".join(f"{i}. {f[:220]}" for i, f in enumerate(research.findings[:10], 1))
    weak = "\n".join(f"- {c.claim}" for c in research.claims if c.strength in ("low", "insufficient"))
    caveats = "\n".join(f"- {c}" for c in plan.factual_caveats[:5])
    lo, hi = round(budget * (1 - config.WORD_BUDGET_TOLERANCE)), round(budget * (1 + config.WORD_BUDGET_TOLERANCE))
    glossary = "\n".join(f"- {g.term}: say '{g.analogy or g.plain}' first; plain meaning: {g.plain}" for g in plan.glossary) or "(none)"
    known = "; ".join(plan.assumed_knowledge) or "(none)"
    hook_line = f"Hook idea: {plan.hook}"
    if plan.hook_style or plan.hook_fact:
        hook_line += f" (style: {plan.hook_style or 'free'}" + (f"; built on this researched fact: {plan.hook_fact}" if plan.hook_fact else "") + ")"

    return f"""Task: write a SKETCH narration for a short video: neutral, plain spoken sentences that cover each beat. Later stages add character voices; you only lay out the content.

Now: {info['time_line']}
Topic: {info['topic']}
Angle: {info['angle'] or 'none given'}
{('Channel style: ' + info['channel_line']) if info['channel_line'] else ''}
{_world_block(world)}

Length: about {budget} words in total (between {lo} and {hi}).
Audience: {info['audience']} (reading level about grade {info['reading_grade']})

Plan: {plan.title}
{hook_line}
Main takeaway (lands in {plan.conclusion_beat_id or 'the last beat'}): {plan.ending}
Beats (write exactly one section per beat, in this order):
{beats}

Hard terms (say the everyday picture first, the term only if it is needed again):
{glossary}
Already known to the audience (do not explain):
{known}

Facts you may use (numbered):
{facts or '(none)'}

Do not state as fact (weakly supported):
{weak or '(none)'}
Must not be overstated:
{caveats or '(none)'}

Rules:
- Use only the facts above. Add no facts, numbers or dates that are not listed.
- Open with the hook idea in the first section. Land the main takeaway in the beat marked as the conclusion; a beat after it, if any, is a short light closing note. Never repeat the hook as the ending.
- Write for the audience above: short sentences (about 15 words or fewer), everyday words, no jargon. Explain each beat the way its 'explain as' note says. Use at most one analogy per beat, and never reuse the same comparison (a camera, say) in different beats. An analogy may add no new fact or number.
- Do not explain what the audience already knows: 'the Roman emperor Julius Caesar', not his biography.
- Each beat follows naturally from the previous one, as its link says; use plain connecting phrases (so, but, which raises the question), and begin each beat after the first by picking up an idea or word from the end of the previous one.
- Plain spoken sentences. No headings, no stage directions, no character names, no dialogue.
- Stay within each beat's word limit. Never say the same fact twice to fill space; stop when the key points are explained.
- performance_intent: 3 to 6 words on how the line should be delivered.
- source_findings: the numbers of the facts the section uses.
Return JSON only."""


def run_script_refiner(
    story_plan: Any,
    research: Any,
    channel: Any = None,
    world: Any = None,
    speaker_character: Any = None,          # accepted for old callers; ignored
    *,
    context: Any = None,
    context_path: Optional[str] = None,
    registry: Optional[RegistryLoader] = None,
) -> Script:
    if isinstance(story_plan, dict):
        story_plan = from_dict(StoryPlan, story_plan)
    if isinstance(research, dict):
        research = from_dict(ResearchResult, research)
    if not isinstance(story_plan, StoryPlan) or not isinstance(research, ResearchResult):
        raise TypeError("run_script_refiner() needs a StoryPlan and a ResearchResult.")
    if not story_plan.beats:
        raise ValueError("The story plan has no beats.")

    info = read_brief(context, context_path)
    info["topic"] = research.topic or info["topic"]
    ctx = info["ctx"]
    registry = registry or RegistryLoader()

    # World: given by the caller, or chosen from the channel's menu.
    if world is not None:
        world = from_dict(World, world) if isinstance(world, dict) else world
        world_id, world_rationale = world.world_id, "World supplied by the caller."
    else:
        channel_id = getattr(channel, "channel_id", None) or info["channel_id"]
        if not channel_id:
            raise ValueError("No channel selected; the Orchestrator must choose one first.")
        world_id, world_rationale = choose_world(info, story_plan, registry, channel_id)
        world = registry.load_world(world_id)
    print(f"[Script Refiner] world: {world_id} ({world_rationale})")

    budget = _word_budget(story_plan, ctx)
    targets = _allocate_words(story_plan, budget)
    by_id = {b.beat_id: b for b in story_plan.beats}
    beat_ids = list(by_id)
    lo, hi = round(budget * HARD_LOW), round(budget * HARD_HIGH)
    n_facts = len(research.findings[:10])

    def validate(data: Dict[str, Any]) -> Dict[str, Any]:
        found: Dict[str, Dict[str, Any]] = {}
        for item in data.get("sections") if isinstance(data.get("sections"), list) else []:
            if not isinstance(item, dict) or item.get("beat_id") not in by_id:
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            slot = found.setdefault(item["beat_id"], {"text": "", "intent": "", "facts": []})
            slot["text"] = (slot["text"] + " " + text).strip()
            slot["intent"] = slot["intent"] or str(item.get("performance_intent", "")).strip()
            slot["facts"] += [n for n in item.get("source_findings", [])
                              if isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= n_facts]
        missing = [b for b in beat_ids if b not in found]
        if missing:
            raise ValueError(f"These beats have no section text: {', '.join(missing)}.")
        words = sum(len(v["text"].split()) for v in found.values())
        if not lo <= words <= hi:
            raise ValueError(f"The script has {words} words; it must be about {budget} words "
                             f"(between {lo} and {hi}).")
        return {"title": str(data.get("title", "")).strip(), "found": found}

    schema = {"type": "object", "properties": {
        "title": {"type": "string"},
        "sections": {"type": "array", "items": {"type": "object", "properties": {
            "beat_id": {"type": "string", "enum": beat_ids},
            "text": {"type": "string"},
            "performance_intent": {"type": "string"},
            "source_findings": {"type": "array", "items": {"type": "integer"}}},
            "required": ["beat_id", "text", "performance_intent", "source_findings"]}}},
        "required": ["title", "sections"]}

    data = llm.generate_json(STAGE, _build_prompt(story_plan, research, info, world, budget, targets),
                             schema=schema, validate=validate, retries=1)
    last = llm.LAST_RESULT

    sections: List[ScriptSection] = []
    for beat_id in beat_ids:
        slot = data["found"][beat_id]
        beat = by_id[beat_id]
        sections.append(ScriptSection(
            section_id=f"{beat_id}a", text=slot["text"], beat_id=beat_id,
            beat_title=beat.summary[:80], performance_intent=slot["intent"] or None,
            narrative_purpose=beat.purpose,
            metadata={"source_findings": sorted(set(slot["facts"]))}))

    script = Script(
        title=data["title"] or story_plan.title,
        sections=sections,
        target_seconds=story_plan.duration.target_seconds if story_plan.duration else None,
        metadata={"agent": STAGE},
    )
    words = script.word_count
    tolerance = config.WORD_BUDGET_TOLERANCE
    script.metadata.update({
        "world_id": world_id,
        "world_rationale": world_rationale,
        "word_budget": budget,
        "word_count": words,
        "length_status": "long" if words > budget * (1 + tolerance)
                         else "short" if words < budget * (1 - tolerance) else "ok",
        "unsupported_numbers": unsupported_numbers(script.full_text, research),
        "prompt_tokens": getattr(last, "prompt_tokens", None),
        "output_tokens": getattr(last, "output_tokens", None),
    })
    if script.metadata["unsupported_numbers"]:
        print(f"[Script Refiner] WARNING: numbers not found in the research: "
              f"{script.metadata['unsupported_numbers']}")
    print(f"[Script Refiner] {len(sections)} sections, {words}/{budget} words "
          f"({script.metadata['length_status']})")
    return script


def apply_to_context(ctx: Any, script: Script) -> None:
    """Record the chosen world in the context."""
    ctx.select_world(script.metadata["world_id"], script.metadata.get("world_rationale", ""))


generate_script = run_script_refiner
execute_script_refiner = run_script_refiner


def _main() -> None:
    import argparse
    import dataclasses

    from core.run_context import RunContext

    parser = argparse.ArgumentParser()
    parser.add_argument("topic", nargs="*")
    parser.add_argument("--channel")
    parser.add_argument("--from-run", help="shared_context.json of an earlier run; skips research and architect")
    args = parser.parse_args()

    if args.from_run:
        source = Path(args.from_run)
        ctx = RunContext.load(source / "shared_context.json" if source.is_dir() else source)
        research = from_dict(ResearchResult, ctx.read_stage_output("researcher"))
        plan = from_dict(StoryPlan, ctx.read_stage_output("architect"))
    else:
        from agents import architect
        from app import orchestrator
        from agents.researcher import run_research

        prompt = " ".join(args.topic) or "How the Antikythera mechanism worked"
        ctx = RunContext(prompt)
        decision = orchestrator.run_orchestrator(context=ctx)
        if decision["requires_clarification"]:
            raise SystemExit(f"The orchestrator needs more information: {decision['clarification_question']}")
        orchestrator.apply_to_context(ctx, decision)
        if args.channel:
            ctx.select_channel(args.channel)
        print(f"Channel: {ctx.channel_id} | length: {ctx.duration.get('target_seconds', 'architect decides')} "
              f"| style: {ctx.interpretation.get('length_style')}")
        research = run_research(context=ctx)
        ctx.write_stage_output("researcher", dataclasses.asdict(research))
        plan = architect.run_architect(research, context=ctx)
        architect.apply_to_context(ctx, plan)
        ctx.write_stage_output("architect", dataclasses.asdict(plan))
        ctx.save()
        print("Saved run:", ctx.context_path)

    script = run_script_refiner(plan, research, context=ctx)
    apply_to_context(ctx, script)
    ctx.write_stage_output("script_refiner", dataclasses.asdict(script))
    ctx.save()

    print(f"\n{script.title}   (world: {script.metadata['world_id']})")
    for s in script.sections:
        print(f"\n[{s.section_id}] {s.performance_intent}\n{s.text}")


if __name__ == "__main__":
    _main()