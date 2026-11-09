"""
Kestrel Shorts - Researcher
===========================

Learns the topic in stages, because the model's own knowledge is stale:

  1. Plan 2 exploration searches        (LLM, small)
  2. Search them
  3. Plan deeper searches from what was found   (LLM, sees the evidence)
  4. Search them
  5. Synthesize numbered evidence into findings, claims and uncertainties (LLM)
  6. Optionally one follow-up round for an important gap (search + re-synthesize)

Rules it enforces in code:
  - No evidence means a ResearchError, never an answer written from memory.
  - Claims cite evidence NUMBERS; the code maps them to real URLs, so a
    source can never be invented. A claim with no valid source is
    "insufficient".
  - Evidence is shown round-robin across queries under a size cap, so later
    searches are not crowded out by earlier ones.
  - Raw searches are cached in the run folder, so a retry repeats no search.

    python -m agents.researcher "How did Roman roads work"
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agents._common import read_brief as _read_context
from core import config, llm, search
from core.models.research_result import Claim, HookCandidate, ResearchResult

logger = logging.getLogger(__name__)

STAGE = "researcher"

MAX_EXPLORATION_QUERIES = 2
MAX_DEEP_QUERIES = 3
MAX_FOLLOW_UP_QUERIES = 2
MAX_TOTAL_QUERIES = 6

EXPLORATION_EVIDENCE_CHARS = 3000
SYNTHESIS_EVIDENCE_CHARS = 6000
PER_SOURCE_CHARS = 600

STRENGTHS = ("high", "medium", "low", "insufficient")


class ResearchError(RuntimeError):
    """Research cannot produce trustworthy output (no evidence, service down...)."""


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------

import re as _re
from urllib.parse import urlparse as _urlparse

# Pages that are never evidence for a video: dictionaries, shops and social feeds.
_JUNK_HOSTS = _re.compile(
    r"(?:^|\.)(?:dictionary\.com|merriam-webster\.com|vocabulary\.com|thefreedictionary\.com|collinsdictionary\.com|"
    r"dictionary\.cambridge\.org|oxfordlearnersdictionaries\.com|wiktionary\.org|urbandictionary\.com|thesaurus\.com|"
    r"pinterest\.[a-z.]+|quora\.com|facebook\.com|instagram\.com|tiktok\.com|amazon\.[a-z.]+|ebay\.[a-z.]+|etsy\.com)$",
    _re.IGNORECASE)


def _is_junk(url: str) -> bool:
    host = (_urlparse(url).netloc or "").lower().split(":")[0]
    return bool(_JUNK_HOSTS.search(host))


_RECENCY_WORDS = _re.compile(r"\b(?:latest|newest|current(?:ly)?|recent(?:ly)?|today|up[- ]to[- ]date|updated?)\b", _re.IGNORECASE)


def _timeless(query: str) -> str:
    """For a timeless topic, drop recency words and the current or previous year from a query (a past year such as
    1969 is kept: it is part of the subject). A query that would become too short is returned as it was."""
    year = int(config.now_stamp()["date"][:4])
    cleaned = _re.sub(rf"\b(?:{year}|{year - 1})\b", " ", query.strip())
    cleaned = " ".join(_RECENCY_WORDS.sub(" ", cleaned).split())
    return cleaned if len(cleaned.split()) >= 2 else query.strip()


def _strings(value: Any, limit: int) -> List[str]:
    out: List[str] = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, str) and item.strip() and item.strip() not in out:
            out.append(item.strip())
        if len(out) >= limit:
            break
    return out


def _say(line: str) -> None:
    print(f"[Researcher] {line}")


@dataclass
class _Run:
    cache_dir: Any = None
    timeless: bool = False          # a timeless topic: queries lose recency words and the current year
    executed: List[str] = field(default_factory=list)
    outcomes: List[search.SearchOutcome] = field(default_factory=list)
    groups: List[List[Dict[str, str]]] = field(default_factory=list)

    def run_queries(self, queries: List[str], limit: int) -> None:
        seen = {q.casefold() for q in self.executed}
        for query in queries:
            query = _timeless(query) if self.timeless else query.strip()
            if limit <= 0 or len(self.executed) >= MAX_TOTAL_QUERIES:
                break
            if not query or query.casefold() in seen:
                continue
            seen.add(query.casefold())
            outcome = search.search(query, cache_dir=self.cache_dir)
            self.executed.append(query)
            self.outcomes.append(outcome)
            self.groups.append(outcome.results)
            limit -= 1
            _say(f"search {outcome.status}{' (cached)' if outcome.cached else ''}: {query}")

    def results(self) -> List[Dict[str, str]]:
        """All results, one from each query in turn, without duplicate URLs."""
        ordered, seen = [], set()
        for rank in range(max((len(g) for g in self.groups), default=0)):
            for group in self.groups:
                if rank < len(group) and group[rank]["url"] not in seen and not _is_junk(group[rank]["url"]):
                    seen.add(group[rank]["url"])
                    ordered.append(group[rank])
        return ordered


def _header(info: Dict[str, Any]) -> str:
    return (
        f"Now: {info['time_line']}\n"
        f"Topic: {info['topic']}\n"
        f"The user's request, word for word: {info['request'] or info['topic']}\n"
        f"Angle: {info['angle'] or 'none given'}\n"
        f"Freshness needed: {info['freshness']}\n"
        "Research exactly what the user asked about; the channel's theme must not change the topic.\n"
        "Respect the scope of the request: 'main parts' or 'overview' means the main, commonly taught points, not every detail.\n"
    )


def _plan(prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """A planning call. Failures fall back to code-made queries, except a dead Ollama."""
    try:
        return llm.generate_json(STAGE, prompt, schema=schema)
    except llm.LLMError as exc:
        if exc.kind in ("connection", "http", "server"):
            raise
        logger.warning("Planning call failed (%s); using fallback queries.", exc)
        return {}


_STR_LIST = {"type": "array", "items": {"type": "string"}}
EXPLORE_SCHEMA = {"type": "object", "properties": {"queries": _STR_LIST}, "required": ["queries"]}
DEEP_SCHEMA = {"type": "object", "properties": {"essential_points": _STR_LIST, "priority_questions": _STR_LIST,
                                                "queries": _STR_LIST, "known_gaps": _STR_LIST},
               "required": ["essential_points", "priority_questions", "queries", "known_gaps"]}
SYNTH_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "findings": _STR_LIST,
        "claims": {"type": "array", "items": {"type": "object", "properties": {
            "claim": {"type": "string"},
            "source_numbers": {"type": "array", "items": {"type": "integer"}},
            "strength": {"type": "string", "enum": list(STRENGTHS)},
            "notes": {"type": "string"}},
            "required": ["claim", "source_numbers", "strength", "notes"]}},
        "uncertainties": _STR_LIST,
        "is_niche": {"type": "boolean"},
        "follow_up_queries": _STR_LIST,
        "hook_candidates": {"type": "array", "items": {"type": "object", "properties": {
            "fact": {"type": "string"},
            "source_numbers": {"type": "array", "items": {"type": "integer"}},
            "why": {"type": "string"}},
            "required": ["fact", "source_numbers", "why"]}},
    },
    "required": ["summary", "findings", "claims", "uncertainties", "is_niche", "follow_up_queries",
                 "hook_candidates"],
}


# ----------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------

def _explore_queries(info: Dict[str, Any]) -> List[str]:
    prompt = (
        "Task: initial exploration for a short video. Plan up to 2 web search queries that "
        "establish what the topic is and its current state.\n\n"
        + _header(info) +
        "\nRules:\n- Queries are short, specific and different from each other.\n"
        "- If freshness is not 'none', include the current year or a recency word.\n"
        "- Do not assume what has or has not happened; establish the current state.\n"
        "Return JSON only."
    )
    queries = _strings(_plan(prompt, EXPLORE_SCHEMA).get("queries"), MAX_EXPLORATION_QUERIES)
    if queries:
        return queries
    year = config.now_stamp()["date"][:4]
    return [info["topic"], f"{info['topic']} {year}" if info["freshness"] != "none"
            else f"{info['topic']} overview"][:MAX_EXPLORATION_QUERIES]


def _deep_plan(info: Dict[str, Any], run: _Run) -> Dict[str, Any]:
    text, _ = search.format_evidence(run.results(), PER_SOURCE_CHARS, EXPLORATION_EVIDENCE_CHARS)
    prompt = (
        "Task: deeper research planning for a short video. You have seen the first search "
        "results; use them to choose what to look up next.\n\n"
        + _header(info) +
        f"\nWhat the first searches found:\n{text or '(nothing found)'}\n\n"
        "Rules:\n- priority_questions: up to 3 questions that matter most and the evidence does not yet answer.\n"
        "- queries: up to 3 short searches that would answer them; do not repeat earlier searches: "
        f"{'; '.join(run.executed)}\n"
        "- essential_points: the 4 to 8 main points any good explainer of this topic would cover (for example, for 'parts of the eye': cornea, lens, iris, retina, optic nerve). Your queries must first confirm the essentials the evidence does not yet cover.\n"
        "- known_gaps: what is still unclear.\n"
        "- Prefer direct, authoritative sources. Do not use your own memory as proof.\n"
        "- Cover the essential, commonly taught points of the topic first; only then chase obscure details.\n"
        "Return JSON only."
    )
    return _plan(prompt, DEEP_SCHEMA)


def _synthesize(info: Dict[str, Any], results: List[Dict[str, str]],
                essentials: Optional[List[str]] = None) -> Tuple[Dict[str, Any], List[str]]:
    text, urls = search.format_evidence(results, PER_SOURCE_CHARS, SYNTHESIS_EVIDENCE_CHARS)
    essentials_block = ("Essential points a good explainer covers: " + "; ".join(essentials) + "\n\n") if essentials else ""
    prompt = (
        "Task: synthesize a factual research brief for a short video from the numbered evidence below.\n\n"
        + _header(info) +
        f"\nEvidence (search snippets, numbered):\n{text}\n\n{essentials_block}"
        "Rules:\n- Use only the evidence above for facts. Snippets are leads, not proof. Add nothing from memory.\n"
        "- findings: 5 to 8 short, specific, supported statements. Cover each essential point the evidence supports; list any essential point the evidence does not cover in uncertainties.\n"
        "- claims: the 3 to 6 most important claims. In source_numbers list the evidence numbers that "
        "support each one ([] if none do). strength: high = several independent sources agree, "
        "medium = one solid source, low = weak or indirect, insufficient = not supported.\n"
        "- uncertainties: disputes, gaps, or things the evidence does not settle.\n"
        "- If freshness is not 'none', say in uncertainties when a claim's date cannot be confirmed "
        "or the evidence looks outdated.\n"
        "- Ignore garbled, cut-off or misspelled words in the snippets; never list them as facts or uncertainties.\n"
        "- is_niche: true if the topic is specialized or obscure.\n"
        "- follow_up_queries: up to 2 searches that would fill an important gap; [] if none.\n"
        "- hook_candidates: up to 3 facts from the evidence that could open a video for a general audience: "
        "a surprising scale comparison, a relatable number, or a common misconception. Use only the evidence, "
        "give source_numbers, and say in why what makes it surprising. [] if none.\n"
        "Return JSON only."
    )

    def validate(data: Dict[str, Any]) -> Dict[str, Any]:
        if not str(data.get("summary", "")).strip():
            raise ValueError("summary must not be empty.")
        findings = [re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", f).strip()
                    for f in _strings(data.get("findings"), 10)]
        if not findings:
            raise ValueError("findings must contain at least one supported statement.")
        claims = []
        for item in data.get("claims") if isinstance(data.get("claims"), list) else []:
            if not isinstance(item, dict) or not str(item.get("claim", "")).strip():
                continue
            numbers = [n for n in item.get("source_numbers", [])
                       if isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= len(urls)]
            strength = str(item.get("strength", "")).lower()
            if strength not in STRENGTHS or not numbers:
                strength = "insufficient"          # no real source, no confidence
            claims.append({"claim": item["claim"].strip(),
                           "source_urls": [urls[n - 1] for n in dict.fromkeys(numbers)],
                           "strength": strength, "notes": str(item.get("notes", "")).strip()})
        hooks = []
        for item in data.get("hook_candidates") if isinstance(data.get("hook_candidates"), list) else []:
            if not isinstance(item, dict) or not str(item.get("fact", "")).strip():
                continue
            nums = [n for n in item.get("source_numbers", [])
                    if isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= len(urls)]
            if nums:                                  # a hook must be grounded in a real source
                hooks.append({"fact": re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", item["fact"]).strip(), "source_urls": [urls[n - 1] for n in dict.fromkeys(nums)],
                              "why": str(item.get("why", "")).strip()})
        return {"summary": data["summary"].strip(), "findings": findings, "claims": claims,
                "hook_candidates": hooks[:3],
                "uncertainties": _strings(data.get("uncertainties"), 10),
                "is_niche": bool(data.get("is_niche")),
                "follow_up_queries": _strings(data.get("follow_up_queries"), MAX_FOLLOW_UP_QUERIES)}

    try:
        return llm.generate_json(STAGE, prompt, schema=SYNTH_SCHEMA, validate=validate), urls
    except llm.LLMError as exc:
        raise ResearchError(f"Could not synthesize the research ({exc.kind}): {exc}. "
                            "Searches are cached, so retrying is cheap.") from exc


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def run_research(
    topic: Optional[str] = None,
    *,
    context: Any = None,
    context_path: Optional[str] = None,
) -> ResearchResult:
    info = _read_context(context, context_path)
    if topic and str(topic).strip():
        info["topic"] = str(topic).strip()
    if not info["topic"]:
        raise ValueError("No research topic: pass one or provide a context with a topic.")

    ctx = info["ctx"]
    if not search.ensure_service():
        raise ResearchError("The search service is not running and could not be started. "
                            "Start search_service.py and retry.")

    _say(f"researching: {info['topic']}")
    run = _Run(cache_dir=(ctx.run_directory / "search") if ctx else None, timeless=info["freshness"] == "none")

    exploration = _explore_queries(info)
    run.run_queries(exploration, MAX_EXPLORATION_QUERIES)

    deep = _deep_plan(info, run)
    essentials = _strings(deep.get("essential_points"), 8)
    deep_queries = _strings(deep.get("queries"), MAX_DEEP_QUERIES) or [f"{info['topic']} key facts evidence"]
    run.run_queries(deep_queries, MAX_DEEP_QUERIES)

    if not run.results():
        statuses = {o.status for o in run.outcomes}
        if statuses == {"unavailable"}:
            raise ResearchError("The search service could not be reached. Start search_service.py and retry.")
        raise ResearchError("The searches returned no evidence. Refusing to write research from memory; "
                            "try rephrasing the request.")

    data, urls = _synthesize(info, run.results(), essentials)

    quick = info["length_style"] == "brief" or bool(info["user_seconds"] and info["user_seconds"] <= 45)
    follow_ups = [] if quick else [
        q for q in data["follow_up_queries"] if len(run.executed) < MAX_TOTAL_QUERIES]
    if follow_ups:
        before = len(run.results())
        run.run_queries(follow_ups, MAX_FOLLOW_UP_QUERIES)
        if len(run.results()) > before:
            data, urls = _synthesize(info, run.results(), essentials)

    warnings: List[str] = []
    failed = [o for o in run.outcomes if o.status in ("failed", "unavailable")]
    empty = [o for o in run.outcomes if o.status == "no_results"]
    if failed:
        warnings.append(f"{len(failed)} search(es) failed.")
    if empty:
        warnings.append(f"{len(empty)} search(es) returned nothing.")
    if info["freshness"] != "none":
        warnings.append("The search service gives no publication dates, so freshness "
                        f"({info['freshness']}) could not be verified from the sources.")
    weak = [c for c in data["claims"] if c["strength"] in ("low", "insufficient")]
    if weak:
        warnings.append(f"{len(weak)} claim(s) are weakly supported.")

    result = ResearchResult(
        topic=info["topic"],
        summary=data["summary"],
        findings=data["findings"],
        claims=[Claim(**c) for c in data["claims"]],
        hook_candidates=[HookCandidate(**h) for h in data["hook_candidates"]],
        essentials=essentials,
        uncertainties=data["uncertainties"],
        sources=urls,
        is_niche=data["is_niche"],
        status="completed_with_warnings" if warnings else "completed",
        warnings=warnings,
        queries_executed=list(run.executed),
        researched_at=config.now_stamp()["iso"],
        location=config.LOCATION,
        freshness_window="" if info["freshness"] == "none" else info["freshness"],
    )

    if ctx is not None:
        ctx.write_stage_output("researcher_trace", {
            "exploration_queries": exploration, "deep_plan": deep,
            "searches": [o.to_dict() | {"results": len(o.results)} for o in run.outcomes],
            "evidence_urls": urls,
        })
    _say(f"done: {len(result.findings)} findings, {len(result.claims)} claims, "
         f"{len(result.sources)} sources, status={result.status}")
    return result


run_researcher = run_research

if __name__ == "__main__":
    import sys
    out = run_research(" ".join(sys.argv[1:]) or "The Antikythera mechanism")
    print("\n" + out.summary)
    for f in out.findings:
        print("-", f)
    print("\nWarnings:", out.warnings or "none")