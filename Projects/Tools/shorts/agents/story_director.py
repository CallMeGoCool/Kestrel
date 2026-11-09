"""
Kestrel Shorts - Story Director
===============================

Turns the Script Refiner's sketch into the FINAL script, spoken by the
channel's characters. It owns the narration and the video's length.

How the cast works (no stored specialities, so worlds can differ freely):
  1. For THIS video it writes a cast_plan: which 1-3 characters appear, each
     one's angle on the topic and what they contribute (from their
     personality and the topic). They are equals and experts.
  2. Then it writes the beats. Each line has ONE speaker, so several
     characters can share a beat. Humour follows each character's own
     personality (`personality.humor`), at most one pun per video.

Flow: every beat says how it follows the previous one (`link`). The
`conclusion` beat is where the main takeaway lands and need not be last; a
short `coda` may follow, with a closing style chosen from context.

Audience: plain language for the channel's audience. Hard ideas get the
everyday picture or analogy first, then the term; things the audience
already knows are not explained.

Code does the checking and the arithmetic:
  - speakers must be registered characters (schema enum);
  - emphasis must be an exact substring of its line (one per beat), else dropped;
  - unknown roles, links, voice options are repaired, not fatal;
  - total words must be near the word budget, else one retry;
  - numbers not found in the research are listed in metadata;
  - estimated seconds = words / WORDS_PER_SECOND + estimated pauses.

Revision: the model returns ONLY the beats it changes; the rest stay.

    python -m agents.story_director --from-run <path to shared_context.json>
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agents._common import read_brief
from agents.script_refiner import unsupported_numbers
from core import config, llm
from core.models._serde import from_dict
from core.models.character import Character
from core.models.research_result import ResearchResult
from core.models.script import Script, ScriptSection
from core.models.story_plan import StoryPlan
from core.models.world import World
from core.registry_loader import RegistryLoader
from creative.catalogs import beat_role_catalog, narrative_catalog, voice_catalog

logger = logging.getLogger(__name__)

STAGE = "story_director"
PHASES = ("draft", "revision", "finalize")
DECISIONS = ("accepted", "partially_accepted", "rejected", "unresolved")
MAX_CAST = 3
BOUNDS_USER = (0.75, 1.3)       # accepted share of the word budget when the user asked for a length
BOUNDS_DEFAULT = (0.3, 1.5)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _character_block(c: Character) -> str:
    p = c.personality or {}
    traits = ", ".join(list(p.get("primary_traits", []))[:5] + list(p.get("secondary_traits", []))[:2])
    biases = " ".join(b.get("description", "") for b in c.humanizing_biases[:2] if isinstance(b, dict))
    relations = " ".join(r.get("description", "") for r in c.relationships[:2] if isinstance(r, dict))
    return (f"- {c.character_id} ({c.display_name}): {c.description}\n"
            f"  Traits: {traits}. Style: {c.communication_style}\n"
            f"  Behaviour: {' '.join(c.behavior[:3])}\n"
            f"  Humour: {p.get('humor') or 'not specified (keep it light)'}\n"
            f"  Leanings: {biases or 'none'}  Relationships: {relations or 'none'}")


def _world_block(w: World) -> str:
    return (f'World "{w.name}": {w.description}\n'
            f"Atmosphere: {w.atmosphere or 'n/a'}. Storytelling conventions: "
            f"{'; '.join(w.storytelling_conventions[:4]) or 'none'}")


def _beats_from_script(script: Script) -> Dict[str, Dict[str, Any]]:
    """Internal beat dicts from a Script (sketch or final), keyed by beat_id."""
    beats: Dict[str, Dict[str, Any]] = {}
    for beat in script.beats:
        first = beat.sections[0]
        beats[beat.beat_id] = {
            "beat_id": beat.beat_id,
            "beat_title": beat.title or first.beat_title,
            "beat_role": first.beat_role or "explanation",
            "link": first.continuity_from_previous or "",
            "closing_style": first.metadata.get("closing_style", "none"),
            "intent": first.narrative_purpose,
            "setting": first.setting_context,
            "mood": first.atmosphere,
            "lines": [{
                "speaker": s.speaker_character_id,
                "text": s.text,
                "emphasis": s.emphasis,
                "pace": s.voice_direction.get("pace", voice_catalog.DEFAULTS["pace"]),
                "pause_after": s.voice_direction.get("pause_after", voice_catalog.DEFAULTS["pause_after"]),
            } for s in beat.sections],
        }
    return beats


def _words(beats: Dict[str, Dict[str, Any]]) -> int:
    return sum(len(line["text"].split()) for b in beats.values() for line in b["lines"])


def _estimated_seconds(beats: Dict[str, Dict[str, Any]]) -> float:
    pauses = voice_catalog.ESTIMATED_PAUSE_SECONDS
    return _words(beats) / config.WORDS_PER_SECOND + sum(
        pauses.get(line["pause_after"], 0.0) for b in beats.values() for line in b["lines"])


def _feedback_items(feedback: Any) -> List[Dict[str, str]]:
    """Normalize feedback (list, or dict of issues/suggestions) into numbered items."""
    if feedback is None:
        return []
    raw: List[Any] = []
    if isinstance(feedback, dict):
        for key in ("issues", "suggestions", "decisions_required"):
            raw += feedback.get(key) or []
    elif isinstance(feedback, list):
        raw = feedback
    else:
        raw = [feedback]
    items = []
    for i, entry in enumerate(raw, 1):
        e = entry if isinstance(entry, dict) else {"problem": str(entry)}
        items.append({
            "id": str(e.get("id") or f"f{i}"),
            "from": str(e.get("from") or e.get("source") or "reviewer"),
            "beat_id": str(e.get("beat_id") or e.get("section_id") or "any"),
            "problem": str(e.get("problem") or e.get("message") or e.get("issue") or "").strip(),
            "suggestion": str(e.get("suggestion") or e.get("proposal") or "").strip(),
        })
    return [it for it in items if it["problem"] or it["suggestion"]]


# ----------------------------------------------------------------------
# Prompt and schema
# ----------------------------------------------------------------------

def _schema(beat_ids: List[str], speakers: List[str], revision: bool) -> Dict[str, Any]:
    text = {"type": "string"}
    line = {"type": "object", "properties": {
        "speaker": {"type": "string", "enum": speakers} if speakers else text,
        "text": text,
        "emphasis": text,
        "pace": {"type": "string", "enum": voice_catalog.get_voice_option_names("pace")},
        "pause_after": {"type": "string", "enum": voice_catalog.get_voice_option_names("pause_after")},
    }, "required": ["speaker", "text"]}
    beat = {"type": "object", "properties": {
        "beat_id": {"type": "string", "enum": beat_ids},
        "beat_title": text,
        "beat_role": {"type": "string", "enum": beat_role_catalog.get_beat_role_names()},
        "link": {"type": "string", "enum": narrative_catalog.beat_link_names()},
        "closing_style": {"type": "string", "enum": narrative_catalog.closing_style_names()},
        "intent": text, "setting": text, "mood": text,
        "lines": {"type": "array", "items": line},
    }, "required": ["beat_id", "beat_title", "beat_role", "link", "intent", "setting", "mood", "lines"]}
    props: Dict[str, Any] = {"title": text, "beats": {"type": "array", "items": beat}}
    required = ["title", "beats"]
    if speakers:
        props["cast_plan"] = {"type": "array", "items": {"type": "object", "properties": {
            "character_id": {"type": "string", "enum": speakers}, "angle": text, "contribution": text},
            "required": ["character_id", "angle", "contribution"]}}
        if not revision:
            required.insert(1, "cast_plan")
    if revision:
        props["feedback_decisions"] = {"type": "array", "items": {"type": "object", "properties": {
            "item_id": text, "decision": {"type": "string", "enum": list(DECISIONS)},
            "rationale": text, "changed_beats": {"type": "array", "items": text}},
            "required": ["item_id", "decision", "rationale", "changed_beats"]}}
        required.append("feedback_decisions")
    return {"type": "object", "properties": props, "required": required}


def _rules(has_cast: bool, revision: bool) -> str:
    cast = ""
    if has_cast and not revision:
        cast = ("- First write cast_plan: choose 1 to 3 of the characters for THIS video (one is fine when that is best). "
                "Give each an angle on this topic (the part they know best or the way they look at it, drawn from their "
                "personality and the topic) and what they contribute (details, numbers, examples, doubts, wit). They are "
                "equals and experts: nobody plays the student. A character may ask a question, but only on behalf of the "
                "viewer or to hand over to the other's angle, never out of ignorance. Give each beat to whoever's angle fits it.\n")
    elif has_cast:
        cast = "- Keep the cast plan above: each character keeps their angle.\n"
    return f"""Rules:
- Facts come only from the material above. Personality changes how things are said, never what is true. Add no numbers, dates or claims that are not there.
- Audience: write for the audience above. Short sentences, everyday words. Keep the explanations the sketch already gives: add no new analogy, and never reuse a comparison used in another beat; an analogy adds no new fact or number. Do not explain what the audience already knows ("already known" list): 'the Roman emperor Julius Caesar', not his biography.
{cast}- Each beat has one or more lines; each line has ONE speaker. Characters may speak to the viewer directly now and then. Natural speech in each character's own style; no stage directions, no name labels, no emojis.
- Humour only as each character's own style allows; at most one pun in the whole video.
- Lists, tables and many items: never read them cell by cell. Name the pattern, mention two or three items, then summarise the rest ('and six more'). When feedback says a beat's screen is too full, rewrite that beat so its items come in separate parts, in the order they should appear on screen, one part per sentence.
- Flow: each beat follows from the previous one. link says how: therefore, but, raises_question, callback, example, and_then (weakest, use rarely). The first beat is 'opens'.
- beat_role: hook (the opening), explanation, question (a character asks the viewer or hands over), reaction, emphasis (a key moment), example, data (numbers, lists, comparisons), conclusion (where the main takeaway lands; it need not be the last beat), coda (a short light closing note after the conclusion, optional). Do not add a coda just to finish. For a coda set closing_style to fit THIS video: callback, question_to_viewer, reflection, light_joke, teaser or none.
- intent: one sentence on what the viewer should understand or feel. setting: what this beat is about, as something that could be pictured, in a few words (for example 'the retina at the back of the eye'). Never the characters' own room. mood: 2 or 3 words.
- Optional per line: emphasis (a word or short phrase copied exactly from the line; at most one per beat), pace (slow, natural, brisk), pause_after (none, short, medium, long).
- Open with the hook, built on the hook fact when one is given. The takeaway lands in the conclusion beat and does not repeat the hook."""


def _build_prompt(info, research, plan, world, characters, beats, budget, bounds, revision_items,
                  phase, cast_plan) -> str:
    lo, hi = round(budget * bounds[0]), round(budget * bounds[1])
    scale = budget / max(1, _words(beats))
    plan_links = {b.beat_id: b.link for b in plan.beats}
    beat_lines = []
    for b in beats.values():
        text = " ".join(f'[{l["speaker"] or "narrator"}] {l["text"]}' for l in b["lines"])
        head = f'{b["beat_id"]} (about {max(8, round(len(text.split()) * scale))} words'
        hint = b["link"] or plan_links.get(b["beat_id"], "")
        beat_lines.append(f'{head}{", suggested link: " + hint if hint else ""}): {text}')
    facts = "\n".join(f"- {f[:220]}" for f in research.findings[:10]) or "(none)"
    weak = "\n".join(f"- {c.claim}" for c in research.claims if c.strength in ("low", "insufficient")) or "(none)"
    caveats = "\n".join(f"- {c}" for c in plan.factual_caveats[:5]) or "(none)"
    glossary = "\n".join(f"- {g.term}: say '{g.analogy or g.plain}' first; plain meaning: {g.plain}"
                         for g in plan.glossary) or "(none)"
    known = "; ".join(plan.assumed_knowledge) or "(none)"
    cast = "\n".join(_character_block(c) for c in characters) or "(no characters: use an empty speaker)"
    hook = f"Hook idea: {plan.hook}"
    if plan.hook_style or plan.hook_fact:
        hook += f" (style: {plan.hook_style or 'free'}" + (f"; researched fact: {plan.hook_fact}" if plan.hook_fact else "") + ")"
    previous_cast = ""
    if cast_plan:
        previous_cast = "\nCast plan:\n" + "\n".join(
            f"- {c['character_id']}: angle: {c.get('angle', '')}. Contributes: {c.get('contribution', '')}"
            for c in cast_plan) + "\n"

    if phase == "draft":
        task = ("Task: turn the sketch script below into the final script, spoken by the characters. "
                "You write the lines; you do not choose visuals.")
        source, extra = "Sketch (rewrite each beat as lines):", ""
    else:
        feedback = "\n".join(
            f'- [{i["id"]}] from {i["from"]}, beat {i["beat_id"]}: {i["problem"]}'
            + (f' Suggestion: {i["suggestion"]}' if i["suggestion"] else "") for i in revision_items
        ) or "- (no specific feedback: polish only)"
        task = ("Task: revise the script using the feedback. Return ONLY the beats you change; every beat "
                "you leave out stays exactly as it is.")
        source = "Current script:"
        extra = (f"\nFeedback:\n{feedback}\n\nFor each feedback item add a feedback_decisions entry: "
                 "accepted, partially_accepted, rejected or unresolved, with a one-sentence rationale and the "
                 "beats changed. Accept what improves clarity, accuracy or fit with the visuals; reject what "
                 "conflicts with the facts or the characters; a purely visual concern needs no script change.\n")

    return f"""{task}

Now: {info['time_line']}
Topic: {info['topic']}
Angle: {info['angle'] or 'none given'}
Audience: {info['audience']} (reading level about grade {info['reading_grade']})
{('Channel style: ' + info['channel_line']) if info['channel_line'] else ''}
{_world_block(world)}

Length: about {budget} words of spoken lines in total (between {lo} and {hi}).

{hook}
Main takeaway lands in: {plan.conclusion_beat_id or 'the last beat'}

Characters (speakers must be one of these ids):
{cast}
{previous_cast}
{source}
{chr(10).join(beat_lines)}
{extra}
Hard terms (say the everyday picture first):
{glossary}
Already known to the audience (do not explain):
{known}

Facts you may use:
{facts}
Do not state as fact (weakly supported):
{weak}
Must not be overstated:
{caveats}

{_rules(bool(characters), phase != 'draft')}
Return JSON only."""


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------

def run_story_director(
    context: Any = None,
    research: Any = None,
    story_plan: Any = None,
    script: Any = None,
    channel: Any = None,
    world: Any = None,
    characters: Any = None,
    voice_catalog_unused: Any = None,       # old callers passed a voice catalog; the catalog is imported now
    revision_feedback: Any = None,
    previous_script: Any = None,
    phase: str = "draft",
    num_predict: Optional[int] = None,
    *,
    registry: Optional[RegistryLoader] = None,
    **ignored: Any,                          # tolerate old keyword arguments such as voice_catalog=
) -> Script:
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}.")
    research = from_dict(ResearchResult, research) if isinstance(research, dict) else research
    story_plan = from_dict(StoryPlan, story_plan) if isinstance(story_plan, dict) else story_plan
    for name, value, kind in (("research", research, ResearchResult), ("story_plan", story_plan, StoryPlan)):
        if not isinstance(value, kind):
            raise TypeError(f"{name} must be a {kind.__name__}.")

    def as_script(value: Any) -> Optional[Script]:
        return Script(**value) if isinstance(value, dict) else value

    script, previous_script = as_script(script), as_script(previous_script)
    current = (previous_script or script) if phase != "draft" else script
    if not isinstance(current, Script) or not current.sections:
        raise ValueError("A sketch (draft) or a previous script (revision) with sections is required.")

    info = read_brief(context)
    ctx = info["ctx"]
    registry = registry or RegistryLoader()
    channel_id = getattr(channel, "channel_id", None) or info["channel_id"]

    if characters is None:
        characters = ([registry.load_character(m["id"], resolve=False)
                       for m in registry.character_menu(channel_id)] if channel_id else [])
    characters = [from_dict(Character, c) if isinstance(c, dict) else c for c in characters]
    speakers = [c.character_id for c in characters]
    if not speakers:
        logger.warning("No characters available: lines will have no speaker.")

    if world is None:
        world_id = current.metadata.get("world_id") or (ctx.world_id if ctx else None)
        if not world_id:
            raise ValueError("No world selected; the Script Refiner must choose one first.")
        world = registry.load_world(world_id)
    world = from_dict(World, world) if isinstance(world, dict) else world

    beats_in = _beats_from_script(current)
    beat_ids = list(beats_in)
    if ctx is not None and ctx.word_budget:
        budget = int(ctx.word_budget)
    elif story_plan.duration:
        budget = config.word_budget(story_plan.duration.target_seconds)
    else:
        budget = max(20, _words(beats_in))
    user_set = bool(ctx and ctx.duration.get("source") == "user")
    bounds = BOUNDS_USER if user_set else BOUNDS_DEFAULT
    lo, hi = round(budget * bounds[0]), round(budget * bounds[1])
    items = _feedback_items(revision_feedback)
    item_ids = {i["id"] for i in items}
    previous_cast = list(current.metadata.get("cast_plan") or []) if phase != "draft" else []
    repaired: List[str] = []

    def validate(data: Dict[str, Any]) -> Dict[str, Any]:
        cast: List[Dict[str, str]] = []
        for c in data.get("cast_plan") if isinstance(data.get("cast_plan"), list) else []:
            if isinstance(c, dict) and c.get("character_id") in speakers \
                    and c["character_id"] not in {x["character_id"] for x in cast}:
                cast.append({"character_id": c["character_id"], "angle": str(c.get("angle", "")).strip(),
                             "contribution": str(c.get("contribution", "")).strip()})
        cast = cast[:MAX_CAST]
        if phase == "draft" and speakers and not cast:
            raise ValueError("cast_plan must name at least one of the available characters.")
        if phase != "draft" and not cast:
            cast = [dict(c) for c in previous_cast]

        merged = {bid: dict(b, lines=[dict(l) for l in b["lines"]]) for bid, b in beats_in.items()}
        returned = set()
        for item in data.get("beats") if isinstance(data.get("beats"), list) else []:
            bid = item.get("beat_id")
            if bid not in beats_in:
                continue
            lines, has_emphasis = [], False
            for raw in item.get("lines") if isinstance(item.get("lines"), list) else []:
                text = str(raw.get("text", "")).strip()
                if not text:
                    continue
                emphasis = str(raw.get("emphasis") or "").strip()
                emphasis = emphasis if emphasis and emphasis in text and not has_emphasis else None
                has_emphasis = has_emphasis or bool(emphasis)
                pace, pause = raw.get("pace"), raw.get("pause_after")
                lines.append({
                    "speaker": (raw.get("speaker") or None) if speakers else None,
                    "text": text, "emphasis": emphasis,
                    "pace": pace if voice_catalog.is_valid_voice_option("pace", pace) else voice_catalog.DEFAULTS["pace"],
                    "pause_after": pause if voice_catalog.is_valid_voice_option("pause_after", pause)
                                   else voice_catalog.DEFAULTS["pause_after"],
                })
            if not lines:
                raise ValueError(f"Beat {bid} has no lines.")
            role, link, closing = item.get("beat_role"), item.get("link"), item.get("closing_style")
            role = role if beat_role_catalog.is_valid_beat_role(role) else "explanation"
            merged[bid] = {
                "beat_id": bid,
                "beat_title": str(item.get("beat_title", "")).strip()[:80] or beats_in[bid]["beat_title"],
                "beat_role": role,
                "link": link if link in narrative_catalog.BEAT_LINKS else "and_then",
                "closing_style": closing if role == "coda" and closing in narrative_catalog.CLOSING_STYLES else "none",
                "intent": str(item.get("intent", "")).strip(),
                "setting": str(item.get("setting", "")).strip(),
                "mood": str(item.get("mood", "")).strip(),
                "lines": lines,
            }
            returned.add(bid)
        merged[beat_ids[0]]["link"] = "opens"          # the first beat always opens
        if phase == "draft":
            missing = [b for b in beat_ids if b not in returned]
            if missing:
                raise ValueError(f"These beats have no lines: {', '.join(missing)}.")
        words = _words(merged)
        if not lo <= words <= hi:
            raise ValueError(f"The script has {words} words; it must be about {budget} words "
                             f"(between {lo} and {hi}).")

        used = {l["speaker"] for b in merged.values() for l in b["lines"] if l["speaker"]}
        for cid in sorted(used - {c["character_id"] for c in cast}):   # a speaker missing from the plan joins it
            cast.append({"character_id": cid, "angle": "", "contribution": ""})
            repaired.append(cid)

        decisions = []
        for d in data.get("feedback_decisions") if isinstance(data.get("feedback_decisions"), list) else []:
            if isinstance(d, dict) and d.get("decision") in DECISIONS:
                decisions.append({"item_id": str(d.get("item_id", "")), "decision": d["decision"],
                                  "rationale": str(d.get("rationale", "")).strip(),
                                  "changed_beats": [b for b in d.get("changed_beats", []) if b in beats_in]})
        return {"title": str(data.get("title", "")).strip(), "beats": merged, "decisions": decisions, "cast": cast}

    prompt = _build_prompt(info, research, story_plan, world, characters, beats_in, budget, bounds,
                           items, phase, previous_cast)
    data = llm.generate_json(
        STAGE, prompt, schema=_schema(beat_ids, speakers, phase != "draft"), validate=validate,
        retries=1, num_predict=num_predict,
    )
    last = llm.LAST_RESULT

    sections: List[ScriptSection] = []
    for bid in beat_ids:
        b = data["beats"][bid]
        for i, line in enumerate(b["lines"]):
            sections.append(ScriptSection(
                section_id=f"{bid}{chr(97 + i)}", text=line["text"], beat_id=bid,
                beat_title=b["beat_title"], beat_role=b["beat_role"],
                speaker_character_id=line["speaker"], emphasis=line["emphasis"],
                voice_direction={"pace": line["pace"], "pause_after": line["pause_after"]},
                narrative_purpose=b["intent"], setting_context=b["setting"], atmosphere=b["mood"],
                continuity_from_previous=b["link"],
                metadata={"closing_style": b["closing_style"]} if b["beat_role"] == "coda" else {},
            ))

    estimate = _estimated_seconds(data["beats"])
    out = Script(title=data["title"] or current.title, sections=sections,
                 target_seconds=round(config.clamp_duration(estimate), 1),
                 metadata={"agent": STAGE})
    out.metadata.update({
        "phase": phase,
        "world_id": current.metadata.get("world_id") or getattr(world, "world_id", None),
        "cast_plan": data["cast"],
        "cast_repaired": repaired,
        "word_budget": budget,
        "word_count": out.word_count,
        "estimated_seconds": round(estimate, 1),
        "speakers": sorted({s.speaker_character_id for s in sections if s.speaker_character_id}),
        "unsupported_numbers": unsupported_numbers(out.full_text, research),
        "feedback_decisions": data["decisions"],
        "unanswered_feedback": sorted(item_ids - {d["item_id"] for d in data["decisions"]}),
        "prompt_tokens": getattr(last, "prompt_tokens", None),
        "output_tokens": getattr(last, "output_tokens", None),
    })
    if out.metadata["unsupported_numbers"]:
        print(f"[Story Director] WARNING: numbers not found in the research: {out.metadata['unsupported_numbers']}")
    print(f"[Story Director] {phase}: {len(sections)} lines in {len(beat_ids)} beats, "
          f"{out.word_count}/{budget} words, ~{estimate:.0f}s, speakers={out.metadata['speakers']}")
    return out


def apply_to_context(ctx: Any, script: Script) -> None:
    """Record the length the script really has, without replacing the planned budget."""
    if script.target_seconds:
        ctx.update_section("duration", {
            "final_estimate_seconds": script.target_seconds,
            "final_estimate_words": script.word_count,
        })


generate_story_script = run_story_director
execute_story_director = run_story_director


def _main() -> None:
    import argparse
    import dataclasses

    from core.run_context import RunContext

    parser = argparse.ArgumentParser()
    parser.add_argument("--from-run", required=True, help="shared_context.json of a run that has finished the script refiner")
    args = parser.parse_args()

    ctx = RunContext.load(args.from_run)
    research = from_dict(ResearchResult, ctx.read_stage_output("researcher"))
    plan = from_dict(StoryPlan, ctx.read_stage_output("architect"))
    sketch = Script(**ctx.read_stage_output("script_refiner"))

    script = run_story_director(context=ctx, research=research, story_plan=plan, script=sketch)
    apply_to_context(ctx, script)
    ctx.write_stage_output("story_director", dataclasses.asdict(script))
    ctx.save()

    print(f"\n{script.title}  (~{script.target_seconds:.0f}s)")
    print("Cast:")
    for c in script.metadata["cast_plan"]:
        print(f"  {c['character_id']}: {c['angle']}  [{c['contribution']}]")
    for beat in script.beats:
        first = beat.sections[0]
        print(f"\n== {beat.beat_id}: {beat.title} [{beat.role}, link: {first.continuity_from_previous}]")
        print(f"   intent: {first.narrative_purpose} | setting: {first.setting_context} | mood: {first.atmosphere}")
        for s in beat.sections:
            extra = f"  <{s.emphasis}>" if s.emphasis else ""
            print(f"   {s.speaker_character_id or 'narrator'}: {s.text}{extra}  "
                  f"({s.voice_direction.get('pace')}, pause {s.voice_direction.get('pause_after')})")
        if first.metadata.get("closing_style"):
            print(f"   closing style: {first.metadata['closing_style']}")
    if script.metadata["unsupported_numbers"]:
        print("\nNumbers to check:", script.metadata["unsupported_numbers"])


if __name__ == "__main__":
    _main()