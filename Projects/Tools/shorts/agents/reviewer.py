"""
Kestrel Shorts - Reviewer
=========================

The guard before anything expensive runs (image generation, voice, render).
It is the proof-reader: it checks that the script and the visual plan are
complete, consistent and faithful to the research. It never rewrites.

Two stages, with different verdicts:

  stage="draft"  -> cleared | cleared_with_suggestions | grave_error
  stage="final"  -> cleared | grave_error        (after the directors agree)

Severity of each issue:
  grave       the work must be redone or repaired (missing beats, invented
              or contradicting facts, an unregistered speaker, a length far
              off the budget, an invalid avatar pose, a cast image, a blocked
              image prompt, ...)
  suggestion  the directors may accept or reject it
  note        (final stage only) a leftover suggestion, recorded but not blocking

Most checks are plain code. ONE small LLM call proof-reads the script
against the research (does it deliver what was asked? are any claims
unsupported or contradicted?). Its findings are suggestions only: a small
model is not allowed to veto a script. If that call fails, the code checks
still run and the report says so.

The visual plan (version 2, described in engines/visual/visual_rules.py) is
optional. When given it is checked by this file's own shot checks (beats,
cues, layouts, reuse chains, avatars) and by the shared visual rules (briefs,
elements, text, screen load, kinetic use); with the image prompts, also the
prompts themselves.

    python -m agents.reviewer --from-run <shared_context.json> [--final] [--no-llm]
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from agents._common import read_brief
from agents.script_refiner import unsupported_numbers
from core import config, llm
from core.models._serde import from_dict
from core.models.research_result import ResearchResult
from core.models.script import Script
from core.models.story_plan import StoryPlan
from core.registry_loader import RegistryLoader
from creative.catalogs import beat_role_catalog, narrative_catalog, visual_catalog as vc, voice_catalog
from engines.visual import visual_rules as vr

logger = logging.getLogger(__name__)

STAGE = "reviewer"
GRAVE, SUGGESTION, NOTE = "grave", "suggestion", "note"

# (suggestion band, grave band) as shares of the word budget
LENGTH_USER = ((0.85, 1.15), (0.7, 1.3))
LENGTH_DEFAULT = ((0.75, 1.25), (0.5, 1.5))
UNSUPPORTED_GRAVE_AT = 3       # this many numbers missing from the research make it grave
UNSUPPORTED_CLAIMS_GRAVE = 4   # this many unsupported claims (from the proof-read) are worth a warning
CAMEO_MAX_SHARE, CAMEO_MAX_RUN = 0.4, 2


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

@dataclass
class Issue:
    id: str
    severity: str
    check: str
    problem: str
    beat_id: str = "any"
    suggestion: str = ""
    target: str = "story_director"      # story_director | visual_director | both
    source: str = "code"                # code | llm


@dataclass
class ReviewReport:
    stage: str
    verdict: str
    issues: List[Issue] = field(default_factory=list)
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def feedback_for(self, target: str) -> Dict[str, Any]:
        """Feedback in the shape the directors accept as `revision_feedback`."""
        def item(i: Issue) -> Dict[str, str]:
            return {"id": i.id, "from": "reviewer", "beat_id": i.beat_id,
                    "problem": i.problem, "suggestion": i.suggestion}
        mine = [i for i in self.issues if i.target in (target, "both")]
        return {"source": "reviewer", "status": self.verdict, "stage": self.stage,
                "issues": [item(i) for i in mine if i.severity == GRAVE],
                "suggestions": [item(i) for i in mine if i.severity in (SUGGESTION, NOTE)]}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class _Issues:
    def __init__(self) -> None:
        self.items: List[Issue] = []

    def add(self, severity: str, check: str, problem: str, beat_id: str = "any",
            suggestion: str = "", target: str = "story_director", source: str = "code") -> None:
        self.items.append(Issue(f"r{len(self.items) + 1}", severity, check, problem, beat_id,
                                suggestion, target, source))


# ----------------------------------------------------------------------
# Script checks (code)
# ----------------------------------------------------------------------

def _words(text: str) -> set:
    return set(re.findall(r"[a-z']+", text.lower()))


def _estimated_seconds(script: Script) -> float:
    pauses = voice_catalog.ESTIMATED_PAUSE_SECONDS
    return script.word_count / config.WORDS_PER_SECOND + sum(
        pauses.get(s.voice_direction.get("pause_after", "none"), 0.0) for s in script.sections)


def _syllables(word: str) -> int:
    word = word.lower()
    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and not word.endswith(("le", "ee")) and count > 1:
        count -= 1
    return max(1, count)


def _reading_grade(text: str) -> float:
    """Flesch-Kincaid grade level; 0 when the text is too short to judge."""
    words = re.findall(r"[A-Za-z']+", text)
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    if len(words) < 20 or not sentences:
        return 0.0
    syllables = sum(_syllables(w) for w in words)
    return 0.39 * len(words) / len(sentences) + 11.8 * syllables / len(words) - 15.59


def _check_script(out: _Issues, script: Script, research: ResearchResult, plan: StoryPlan,
                  registered: List[str], budget: int, user_set: bool, reading_grade: int = 0) -> None:
    if not script.sections:
        out.add(GRAVE, "structure", "The script has no lines.")
        return

    ids = [s.section_id for s in script.sections]
    if len(ids) != len(set(ids)):
        out.add(GRAVE, "structure", "Some section ids are duplicated.")

    beats = script.beats
    present = {b.beat_id for b in beats}
    for beat in plan.beats:
        if beat.beat_id not in present:
            out.add(GRAVE, "coverage", f"Planned beat '{beat.summary}' has no lines in the script.",
                    beat.beat_id, "Write the missing beat.")

    for s in script.sections:
        if registered and s.speaker_character_id not in registered:
            out.add(GRAVE, "speaker", f"Line {s.section_id} has speaker {s.speaker_character_id!r}, "
                    "which is not a registered character of this channel.", s.beat_id or "any",
                    f"Use one of: {', '.join(registered)}.")
        if s.emphasis and s.emphasis not in s.text:
            out.add(SUGGESTION, "format", f"Emphasis {s.emphasis!r} is not in line {s.section_id}.", s.beat_id or "any")
        for key, value in s.voice_direction.items():
            if not voice_catalog.is_valid_voice_option(key, value):
                out.add(SUGGESTION, "format", f"Unknown {key} {value!r} in line {s.section_id}.", s.beat_id or "any")

    # Roles and shape
    for b in beats:
        if not beat_role_catalog.is_valid_beat_role(b.role):
            out.add(SUGGESTION, "format", f"Beat {b.beat_id} has no valid beat_role.", b.beat_id)
        first = b.sections[0]
        if not first.narrative_purpose or not first.setting_context:
            out.add(SUGGESTION, "intent", f"Beat {b.beat_id} lacks an intent or a setting; the Visual "
                    "Director needs both.", b.beat_id, "Add what the beat means and what it is about.", "both")
    if beats and beats[0].role != "hook":
        out.add(SUGGESTION, "continuity", "The first beat is not a hook.", beats[0].beat_id)
    roles = [b.role for b in beats]
    if len(beats) > 1 and roles.count("conclusion") != 1:
        out.add(SUGGESTION, "continuity", f"The script has {roles.count('conclusion')} conclusion beats; it should have exactly one.")
    elif roles.count("conclusion") == 1:
        after = beats[roles.index("conclusion") + 1:]
        for b in after:
            if b.role != "coda":
                out.add(SUGGESTION, "continuity", f"Beat {b.beat_id} adds new material after the conclusion.", b.beat_id)
        if len(after) > 1:
            out.add(SUGGESTION, "continuity", "More than one beat follows the conclusion; keep at most one short coda.", after[-1].beat_id)
    settings = {b.sections[0].setting_context.strip().lower() for b in beats if b.sections[0].setting_context}
    if len(beats) >= 3 and len(settings) == 1:
        out.add(SUGGESTION, "continuity", "Every beat has the same setting, so the visuals will repeat.",
                "any", "Describe what each beat is about, not where the characters sit.", "both")

    # Length
    words, est = script.word_count, _estimated_seconds(script)
    suggest_band, grave_band = LENGTH_USER if user_set else LENGTH_DEFAULT
    ratio = words / max(1, budget)
    if not grave_band[0] <= ratio <= grave_band[1] or est > config.MAX_VIDEO_SECONDS:
        out.add(GRAVE, "length", f"The script has {words} words (~{est:.0f}s); the budget is {budget} words.",
                suggestion="Rewrite to the budget.")
    elif ratio > suggest_band[1] or (user_set and ratio < suggest_band[0]):
        out.add(SUGGESTION, "length", f"The script has {words} words (~{est:.0f}s) against a budget of {budget}.",
                suggestion="Trim or extend toward the budget.")
    elif est < config.MIN_VIDEO_SECONDS:
        out.add(SUGGESTION, "length", f"The video would run only ~{est:.0f}s.")

    # Numbers that appear nowhere in the research
    bad = unsupported_numbers(script.full_text, research)
    if len(bad) >= UNSUPPORTED_GRAVE_AT:
        out.add(GRAVE, "facts", f"Numbers not found in the research: {', '.join(bad)}.",
                suggestion="Remove them or tie each to a finding.")
    elif bad:
        out.add(SUGGESTION, "facts", f"Numbers not found in the research: {', '.join(bad)}.",
                suggestion="Remove them or tie each to a finding.")

    # Variety and repetition
    if len(registered) >= 2 and len(script.metadata.get("speakers", {s.speaker_character_id for s in script.sections})) < 2:
        out.add(SUGGESTION, "continuity", "Only one character speaks although several are available.")
    seen: Dict[str, str] = {}
    for s in script.sections:
        key = re.sub(r"\W+", " ", s.text.lower()).strip()
        if key in seen:
            out.add(SUGGESTION, "continuity", f"Line {s.section_id} repeats line {seen[key]}.", s.beat_id or "any")
        seen[key] = s.section_id
    if len(beats) > 1:
        hook, end = _words(beats[0].sections[0].text), _words(beats[-1].sections[-1].text)
        if hook and end and len(hook & end) / len(hook | end) > 0.6:
            out.add(SUGGESTION, "continuity", "The ending largely repeats the hook.", beats[-1].beat_id)

    # Flow: beats linked only by "and then" feel episodic
    if len(beats) >= 3:
        weak = [b for b in beats[1:] if not b.sections[0].continuity_from_previous
                or narrative_catalog.is_weak_link(b.sections[0].continuity_from_previous)]
        if len(weak) / (len(beats) - 1) > 0.4:
            out.add(SUGGESTION, "continuity", f"{len(weak)} of {len(beats) - 1} beats are linked only by 'and then' "
                    "(or not at all), so the video may feel episodic.",
                    suggestion="Say how each beat follows the one before: therefore, but, or a question it raises.",
                    target="both")

    # Balance between characters
    spoken = [s.speaker_character_id for s in script.sections if s.speaker_character_id]
    if len(set(spoken)) >= 2:
        for who in sorted(set(spoken)):
            share = spoken.count(who) / len(spoken)
            if share < 0.25:
                out.add(SUGGESTION, "balance", f"{who} has only {share:.0%} of the lines.",
                        suggestion="Give every character in the cast a real contribution.")
        askers = [s.speaker_character_id for s in script.sections
                  if s.speaker_character_id and s.text.strip().endswith("?")]
        if len(askers) >= 3 and len(set(askers)) == 1:
            out.add(SUGGESTION, "balance", f"{askers[0]} asks every question while the other only answers.",
                    suggestion="Let the characters trade angles instead of one asking and one answering.")

    # Reading level against the audience
    grade = _reading_grade(script.full_text)
    if reading_grade and grade and grade > reading_grade + 2:
        out.add(SUGGESTION, "readability", f"The script reads at about grade {grade:.0f}; the audience level is grade {reading_grade}.",
                suggestion="Shorter sentences and everyday words; give the picture before the term.")


# ----------------------------------------------------------------------
# Proof-read (one LLM call; its findings are suggestions only)
# ----------------------------------------------------------------------

def _llm_check(out: _Issues, script: Script, research: ResearchResult, plan: StoryPlan,
               info: Dict[str, Any], request: str) -> str:
    beat_ids = [b.beat_id for b in script.beats]
    body = "\n".join(
        f"{b.beat_id} [{b.role or '?'}]: " + " / ".join(f'{s.speaker_character_id or "narrator"}: "{s.text}"' for s in b.sections)
        for b in script.beats)
    known = [*research.findings[:10], *(h.fact for h in research.hook_candidates),
             *(c.claim for c in research.claims if c.strength not in ("low", "insufficient"))]
    unmark = lambda t: re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", t)
    facts = "\n".join("- " + unmark(f)[:220] for f in known) or "(none)"
    weak = "\n".join(f"- {c.claim}" for c in research.claims if c.strength in ("low", "insufficient")) or "(none)"
    caveats = "\n".join(f"- {c}" for c in plan.factual_caveats[:5]) or "(none)"
    planned = "; ".join(b.summary for b in plan.beats) or "(none)"
    prompt = f"""Task: proof-read a video script against its source facts. Do not rewrite it.

The user's request: {request}
Topic: {info['topic']}

Facts (the only allowed sources):
{facts}
Weak claims (must not be stated as fact):
{weak}
Must not be overstated:
{caveats}

Script, by beat:
{body}

Check:
1. covers_request: does the script deliver what the user asked for, at the scope the request implies? The video was planned to cover only: {planned}. List in missing only things the request itself requires that the script leaves out (up to 4); a brief or 'main parts' request does not require every detail in the facts.
2. claims: each SPECIFIC factual statement in the script (a number, name, date, or how or why something works) that the facts do not support (unsupported), that contradicts them (contradicts), or that states a weak claim as settled fact (overstated). Quote it briefly. Ignore opinions, rhetorical questions, expressions of wonder, the characters' reactions, and everyday analogies that add no new fact (such as 'like a camera lens'). A sentence that restates a listed fact in other words is supported.
3. continuity: up to 3 notes on beats that repeat each other, jump without a link, or leave a question unanswered.
4. jargon: up to 4 terms a general viewer would not know that the script uses without explaining them in plain words. Terms the user's request names, and terms the script explains with an analogy or a short definition, are not jargon.
Return JSON only."""
    text = {"type": "string"}
    schema = {"type": "object", "properties": {
        "covers_request": {"type": "boolean"}, "missing": {"type": "array", "items": text},
        "claims": {"type": "array", "items": {"type": "object", "properties": {
            "beat_id": {"type": "string", "enum": beat_ids}, "claim": text,
            "kind": {"type": "string", "enum": ["unsupported", "contradicts", "overstated"]}},
            "required": ["beat_id", "claim", "kind"]}},
        "continuity": {"type": "array", "items": {"type": "object", "properties": {
            "beat_id": {"type": "string", "enum": beat_ids}, "note": text}, "required": ["beat_id", "note"]}},
        "jargon": {"type": "array", "items": text}},
        "required": ["covers_request", "missing", "claims", "continuity", "jargon"]}
    try:
        data = llm.generate_json(STAGE, prompt, schema=schema, retries=1)
    except llm.LLMError as exc:
        if exc.kind in ("connection", "http", "server"):
            raise
        out.add(SUGGESTION, "proofread", f"The automatic proof-read could not run ({exc.kind}); "
                "only the code checks were applied.", source="code")
        return "failed"

    missing = [m for m in data.get("missing", []) if isinstance(m, str) and m.strip()]
    if not data.get("covers_request", True) or missing:
        out.add(SUGGESTION, "coverage", "The proof-read thinks the script may not cover: "
                + ("; ".join(missing[:4]) or "essential points") + " (check this by hand).",
                suggestion="Add it only if the request really needs it.", target="both", source="llm")
    soft = 0
    for c in data.get("claims", []):
        if c.get("kind") == "contradicts":
            out.add(SUGGESTION, "facts", f"Possible contradiction with the research: {c.get('claim', '')}", c["beat_id"],
                    "Check it against the facts.", source="llm")
        else:
            soft += 1
            out.add(SUGGESTION, "facts", f"{c.get('kind', 'unsupported').capitalize()} claim: {c.get('claim', '')}",
                    c["beat_id"], "Remove it or ground it in a finding.", source="llm")
    if soft >= UNSUPPORTED_CLAIMS_GRAVE:
        out.add(SUGGESTION, "facts", f"The proof-read doubts {soft} claims; check them by hand.", source="llm")
    # Jargon flags from the model hit the parts the video is about; the code reading-level check covers this.
    # The model's continuity notes were wrong in the real run (they contradicted the script);
    # the code checks already cover links, repeats and the ending.
    return "ok"


# ----------------------------------------------------------------------
# Visual plan checks (code)
# ----------------------------------------------------------------------

def _check_visual_plan(out: _Issues, plan: Any, script: Script, level: str,
                       registry: Optional[RegistryLoader], research: Optional[ResearchResult] = None) -> None:
    """Checks the Visual Director's plan: shots, images, text, avatars and their limits."""
    shots = plan.get("shots", []) if isinstance(plan, dict) else (plan or [])
    beats = script.beats
    beat_ids = [b.beat_id for b in beats]
    roles = {b.beat_id: b.role for b in beats}
    beat_text = {b.beat_id: " ".join(s.text for s in b.sections).lower() for b in beats}
    with_avatar: Dict[str, bool] = {b: False for b in beat_ids}
    pose_cache: Dict[str, set] = {}
    media_seen: set = set()
    generated, screen_text = 0, []

    def poses(character_id: str) -> set:
        if character_id not in pose_cache:
            try:
                pose_cache[character_id] = {p["pose_id"] for p in registry.pose_menu(character_id)} if registry else set()
            except Exception:
                pose_cache[character_id] = set()
        return pose_cache[character_id]

    for position, shot in enumerate(shots):
        sid, bid = str(shot.get("shot_id", f"#{position}")), shot.get("beat_id")
        if bid not in with_avatar:
            out.add(GRAVE, "visual", f"Shot {sid} points to an unknown beat {bid!r}.", "any", target="visual_director")
            continue
        if not str(shot.get("intent", "")).strip():
            out.add(SUGGESTION, "intent", f"Shot {sid} has no intent (why this visual is used here).", bid,
                    target="visual_director")
        cue = str(shot.get("cue") or "").strip()
        if cue and cue.lower() not in beat_text[bid]:
            out.add(SUGGESTION, "visual", f"Shot {sid} starts at a phrase that is not in the narration: {cue!r}.", bid,
                    target="visual_director")
        if shot.get("subtitle_style", "regular") not in ("regular", "kinetic"):
            out.add(SUGGESTION, "visual", f"Shot {sid} has an unknown subtitle style.", bid, target="visual_director")

        media = shot.get("media") or []
        layout = shot.get("layout")
        if layout not in vc.LAYOUTS:
            out.add(GRAVE, "visual", f"Shot {sid} uses an unknown layout {layout!r}.", bid, target="visual_director")
        else:
            low, high = vc.slot_range(layout)
            if not low <= len(media) <= high:
                out.add(GRAVE, "visual", f"Shot {sid}: layout {layout!r} holds {low}-{high} image(s) but the shot has "
                        f"{len(media)}.", bid, "Pick a layout that fits the images.", "visual_director")
        for m in media:
            if m.get("source") in ("reuse", "variation") and m.get("ref") not in media_seen:
                out.add(GRAVE, "visual", f"Shot {sid} {m.get('source')}s {m.get('ref')!r}, which is not an earlier image.",
                        bid, "Reuse only images already shown.", "visual_director")
            if m.get("source") in ("generate", "variation"):
                generated += 1
            media_seen.add(m.get("media_id"))           # a zoom chain may vary an image from the same shot

        for t in shot.get("text") or []:
            if t.get("kind") not in ("attention", "explanatory"):
                out.add(SUGGESTION, "visual", f"Shot {sid} has on-screen text of kind {t.get('kind')!r}; subtitles follow "
                        "the narration and are not written here.", bid, target="visual_director")
            screen_text.append(str(t.get("content", "")))

        avatar = shot.get("avatar")
        if avatar:
            with_avatar[bid] = True
            cid, pose = avatar.get("character_id"), avatar.get("pose_id")
            if pose not in poses(cid):
                out.add(GRAVE, "visual", f"Shot {sid} uses pose {pose!r} for {cid!r}, which is not registered.",
                        bid, "Pick a pose from that character's pose list.", "visual_director")

    shown = {s.get("beat_id") for s in shots}
    for bid in beat_ids:
        if bid not in shown:
            out.add(GRAVE, "visual", f"Beat {bid} has no shot.", bid, target="visual_director")

    if generated > config.MAX_GENERATED_IMAGES:
        out.add(GRAVE, "visual", f"The plan needs {generated} generated images; the limit is {config.MAX_GENERATED_IMAGES}.",
                suggestion="Reuse or vary earlier images.", target="visual_director")
    if research is not None and screen_text:
        bad = unsupported_numbers(" ".join(screen_text), research)
        if len(bad) >= UNSUPPORTED_GRAVE_AT:
            out.add(GRAVE, "facts", f"On-screen text has numbers not found in the research: {', '.join(bad)}.",
                    target="visual_director")
        elif bad:
            out.add(SUGGESTION, "facts", f"On-screen text has numbers not found in the research: {', '.join(bad)}.",
                    target="visual_director")

    # Avatar presence level
    count = sum(with_avatar.values())
    share = count / max(1, len(beat_ids))
    if level == "none" and count:
        out.add(GRAVE, "avatar", f"Avatars appear in {count} beat(s) but this video is faceless.", target="visual_director")
    elif level == "cameo":
        run = longest = 0
        for bid in beat_ids:
            run = run + 1 if with_avatar[bid] else 0
            longest = max(longest, run)
        if share > CAMEO_MAX_SHARE or longest > CAMEO_MAX_RUN:
            out.add(SUGGESTION, "avatar", f"Avatars appear in {share:.0%} of beats (longest run {longest}); "
                    "'cameo' means occasional.", suggestion="Keep them for hook, question, reaction and conclusion beats.",
                    target="visual_director")
        for bid in beat_ids:
            if with_avatar[bid] and beat_role_catalog.get_beat_role_catalog().get(roles[bid], {}).get("avatar_fit") == "low":
                out.add(SUGGESTION, "avatar", f"Beat {bid} ({roles[bid]}) suits graphics better than an avatar.",
                        bid, target="visual_director")
    elif level == "constant" and share < 0.8:
        out.add(SUGGESTION, "avatar", f"Avatars appear in only {share:.0%} of beats; this channel expects them almost always.",
                target="visual_director")
    elif level == "frequent" and share < 0.4:
        out.add(SUGGESTION, "avatar", f"Avatars appear in only {share:.0%} of beats; this channel expects them often.",
                target="visual_director")

    layouts = [s.get("layout") for s in shots]
    for i in range(len(layouts) - 3):
        if layouts[i] and len(set(layouts[i:i + 4])) == 1:
            out.add(SUGGESTION, "variety", f"Four shots in a row use the layout {layouts[i]!r}.", target="visual_director")
            break


def _check_visual_rules(out: _Issues, plan: Any, script: Script, research: ResearchResult, registry: RegistryLoader,
                        channel_id: Optional[str], prompts: Optional[Dict[str, Any]]) -> None:
    """The shared visual rules (engines/visual/visual_rules.py): briefs, elements, text, load, kinetic use and,
    when prompts exist, the image prompts. Overload and warnings are suggestions; real breaks are grave."""
    if not isinstance(plan, dict):
        return
    banned = ["avatar"]
    for member in (registry.character_menu(channel_id) if channel_id else []):
        banned += [member["id"].lower(), str(member["name"]).lower()]
        banned += [str(p["pose_id"]).lower() for p in registry.pose_menu(member["id"]) if p.get("pose_id")]
    theme = None
    try:
        theme_id = registry.get_channel(channel_id).get("theme_id") if channel_id else ""
        theme = registry.get_theme(theme_id) if theme_id else None
    except Exception:
        theme = None
    narration = {b.beat_id: " ".join(s.text for s in b.sections) for b in script.beats}
    issues = vr.check_plan(plan, research.findings, banned, theme, narration)
    if prompts:
        issues += vr.check_image_prompts(plan, prompts, banned)
    for i in issues:
        grave = i["severity"] == "error" and i["check"] != "load"
        target = "both" if i["check"] in ("load", "kinetic") else "visual_director"
        text = f"{i['shot_id']}: {i['problem']}" if i["shot_id"] else i["problem"]
        out.add(GRAVE if grave else SUGGESTION, f"visual:{i['check']}", text, i.get("beat_id") or "any",
                i.get("suggestion", ""), target)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def review(
    stage: str = "draft",
    script: Any = None,
    research: Any = None,
    story_plan: Any = None,
    visual_plan: Any = None,
    context: Any = None,
    registry: Optional[RegistryLoader] = None,
    use_llm: bool = True,
    image_prompts: Any = None,
) -> ReviewReport:
    if stage not in ("draft", "final"):
        raise ValueError("stage must be 'draft' or 'final'.")
    script = Script(**script) if isinstance(script, dict) else script
    research = from_dict(ResearchResult, research) if isinstance(research, dict) else research
    story_plan = from_dict(StoryPlan, story_plan) if isinstance(story_plan, dict) else story_plan
    if not isinstance(script, Script) or not isinstance(research, ResearchResult) or not isinstance(story_plan, StoryPlan):
        raise TypeError("review() needs a Script, a ResearchResult and a StoryPlan.")

    info = read_brief(context)
    ctx = info["ctx"]
    registry = registry or RegistryLoader()
    channel_id = info["channel_id"]
    registered = [m["id"] for m in registry.character_menu(channel_id)] if channel_id else []

    if ctx is not None and ctx.word_budget:
        budget = int(ctx.word_budget)
    elif story_plan.duration:
        budget = config.word_budget(story_plan.duration.target_seconds)
    else:
        budget = max(20, script.word_count)
    user_set = bool(ctx and ctx.duration.get("source") == "user")
    request = (ctx.original_prompt if ctx is not None else info["topic"])

    level = ((ctx.interpretation.get("avatar_presence") if ctx is not None else None)
             or (registry.get_channel(channel_id).get("avatar_presence") if channel_id else None) or "cameo")

    out = _Issues()
    _check_script(out, script, research, story_plan, registered, budget, user_set, info["reading_grade"])
    proofread = "skipped"
    if use_llm and script.sections:
        proofread = _llm_check(out, script, research, story_plan, info, request)
    if visual_plan is not None:
        _check_visual_plan(out, visual_plan, script, level, registry, research)
        _check_visual_rules(out, visual_plan, script, research, registry, channel_id, image_prompts)

    grave = [i for i in out.items if i.severity == GRAVE]
    soft = [i for i in out.items if i.severity == SUGGESTION]
    if stage == "final":                       # only cleared or grave_error
        for i in soft:
            i.severity = NOTE
        verdict = "grave_error" if grave else "cleared"
    else:
        verdict = "grave_error" if grave else ("cleared_with_suggestions" if soft else "cleared")

    summary = (f"{verdict}: {len(grave)} grave issue(s), {len(soft)} suggestion(s)."
               if stage == "draft" else f"{verdict}: {len(grave)} grave issue(s), {len(soft)} note(s).")
    return ReviewReport(stage, verdict, out.items, summary, {
        "word_budget": budget, "word_count": script.word_count,
        "estimated_seconds": round(_estimated_seconds(script), 1),
        "avatar_presence": level, "proofread": proofread, "visual_plan_checked": visual_plan is not None,
        "image_prompts_checked": bool(image_prompts)})


run_reviewer = review


def _main() -> None:
    import argparse

    from core.run_context import RunContext

    parser = argparse.ArgumentParser()
    parser.add_argument("--from-run", required=True)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    ctx = RunContext.load(args.from_run)
    report = review(
        "final" if args.final else "draft",
        script=ctx.read_stage_output("story_director"),
        research=ctx.read_stage_output("researcher"),
        story_plan=ctx.read_stage_output("architect"),
        visual_plan=ctx.read_stage_output("visual_plan"),
        image_prompts=ctx.read_stage_output("image_prompts"),
        context=ctx, use_llm=not args.no_llm)
    ctx.write_stage_output(f"review_{report.stage}", report.to_dict())
    print(f"\nVERDICT: {report.verdict}\n{report.summary}   {report.metadata}")
    for i in report.issues:
        print(f"  [{i.severity:<10}] {i.check:<16} {i.beat_id:<7} {i.problem}"
              + (f"\n{'':>38}-> {i.suggestion}" if i.suggestion else ""))


if __name__ == "__main__":
    _main()