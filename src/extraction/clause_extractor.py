"""
Clause Extractor — LLM-based clause segmentation + risk classification.

Takes a chapter-level text chunk (from docling_loader.py) and asks a local
Ollama model to:
  1. Segment it into individual numbered clauses.
  2. Classify each clause's risk_level against a human-authored rubric.

Design:
- The RISK RUBRIC is human-authored and fixed in this file (not left to
  LLM judgment) — the LLM applies the rubric, it does not invent categories.
  Mirrors the project's core principle: LLM does extraction, all compliance
  DECISIONS (gap detection) stay in deterministic Cypher rules.
- Output is constrained to JSON — same pattern used in intent_classifier.py
  / reasoning_chain.py (hr-agent-hackathon project) to prevent free-form
  hallucinated prose.
- Calls Ollama's local REST API directly via `requests` (no langchain-ollama
  dependency — keeps this module dependency-light; LangGraph orchestration
  wraps this as a node later without this module knowing about LangChain).
- No knowledge of Neo4j — pure "text in, structured clause list out".
  graph_writer.py consumes this module's output; this module never imports
  graph_writer.
"""
import json
import re
import logging
from typing import Any, Dict, List
import requests
from functools import lru_cache
from difflib import SequenceMatcher


from config.settings import settings

logger = logging.getLogger(__name__)

_VALID_RISK_LEVELS = {"high", "medium", "low"}
_HIGH_SIGNAL_PATTERN = re.compile(r"\b(shall not|shall|must)\b", re.IGNORECASE)
_ILLUSTRATIVE_PREFIX_PATTERN = re.compile(r"^\s*(illustration|example)\s*:", re.IGNORECASE)
_last_call_metadata: Dict[str, Any] = {}
_MAX_PLAUSIBLE_CLAUSE_NUM_LEN = 30


RISK_RUBRIC = """
Risk classification rubric — apply EXACTLY these definitions, do not invent
your own criteria:

- high: clause imposes a hard deadline, monetary penalty, mandatory
  disclosure obligation, or explicit compliance requirement. Look for
  language like "shall", "must", "penalty of", "within X days".
- medium: clause describes a process or procedure with a customer-facing
  obligation, but has no hard penalty or deadline attached.
- low: definitional, background, or explanatory clause with no direct
  compliance action required.
"""


EXTRACTION_PROMPT_TEMPLATE = """You are a regulatory compliance analyst.

{rubric}

Extract each enforceable compliance obligation, operational mandate, restriction, and regulatory definition from the text below.

RULES:
1. CLAUSE COVERAGE — BE EXHAUSTIVE:
   - Extract EVERY distinct actionable obligation, prohibition, or restriction
     ("shall", "must", "shall not", "required to", "is prohibited"), even if
     several appear consecutively in the same paragraph or list.
   - Extract each distinct requirement as its OWN separate clause entry — do
     NOT merge multiple sub-items or consecutive sentences into one clause.
   - Extract defined statutory terms and their specific operational meanings
     from Definitions sections.
   - Do not stop early: a section with 5+ consecutive obligation sentences
     should produce 5+ clause entries, not a summarized subset.
   - Skip non-regulatory front matter, document titles, and table of contents entries.

2. ATOMICITY:
   - Extract each distinct requirement as its own clause entry.
   - Do not merge distinct sub-items into a single paragraph.

3. CLAUSE NUMBERING (clause_num):
   - Capture the visible structural label exactly as formatted (e.g., "11(1)", "(iv)", "5(12)", "B.1").
   - If a paragraph or sub-point has no explicit prefix number, use "".
   - Never generate or invent fake clause numbers.

4. QUOTES & JSON FORMATTING:
   - Use single quotes inside string values (e.g. 'Cardholder').
   - Do not use unescaped double quotes inside values.
   - risk_level must be exactly: "high", "medium", or "low".

5. EMPTY HANDLING:
   - If <CHAPTER_TEXT> contains only headers or table of contents, return: {{"clauses": []}}

<CHAPTER_TEXT>
{chapter_text}
</CHAPTER_TEXT>

Respond ONLY with valid JSON (no markdown fences, no conversational text):
{{
  "clauses": [
    {{"clause_num": "...", "text": "...", "risk_level": "...", "reason": "..."}}
  ]
}}
"""


def _call_ollama(prompt: str, model: str | None = None, timeout: int = 600, max_retries: int = 1) -> str:
    """
    Sends a generate request to Ollama's REST API. Retries ONCE on
    ReadTimeout (transient — machine sleep/wake, cold model load, slow
    chapter) with the same timeout. Does NOT retry on ConnectionError —
    that means Ollama is down, not slow, and retrying won't help; let it
    propagate immediately.
    Returns raw response text. Also logs (and the caller can inspect via
    _last_call_metadata) whether Ollama's context window was exceeded —
    see done_reason handling below.
    """
    model = model or settings.OLLAMA_MODEL

    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0.0,
                        "repeat_penalty": 1.1,
                        "repeat_last_n": 256,
                        "num_ctx": 6144,
                        "num_predict": 3500,
                        "keep_alive": "15m"
                    }
                },
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()

            done_reason = data.get("done_reason")
            prompt_tokens = data.get("prompt_eval_count")
            output_tokens = data.get("eval_count")
            logger.info(
                f"[ClauseExtractor] Ollama call done_reason={done_reason} "
                f"prompt_tokens={prompt_tokens} output_tokens={output_tokens} model={model}"
            )
            if done_reason == "length":
                logger.warning(
                    f"[ClauseExtractor] TRUNCATED — output cut off before natural stop "
                    f"(prompt_tokens={prompt_tokens}, output_tokens={output_tokens})."
                )
            _last_call_metadata["done_reason"] = done_reason
            _last_call_metadata["prompt_tokens"] = prompt_tokens
            _last_call_metadata["output_tokens"] = output_tokens

            return data["response"]
        except requests.exceptions.ReadTimeout as e:
            last_exc = e
            logger.warning(
                f"[ClauseExtractor] Ollama read timeout (attempt {attempt + 1}/{max_retries + 1}, timeout={timeout}), model={model}.")
    raise last_exc


def _salvage_truncated_json(cleaned: str) -> dict | None:
    """
    Recovers complete clause objects from a truncated 'clauses' array by
    trimming back to the last complete '}' before the cut point, then
    closing the array/object. Returns None if no complete objects exist.
    """
    start = cleaned.find('"clauses"')
    if start == -1:
        return None
    last_complete = cleaned.rfind('}')
    if last_complete == -1:
        return None
    salvaged = cleaned[:last_complete + 1] + "]}"
    try:
        return json.loads(salvaged)
    except json.JSONDecodeError:
        return None


def _parse_clauses(raw_output: str) -> List[Dict[str, Any]]:
    """
    Parses and validates the LLM's JSON output.
    Drops any individual clause with an invalid/missing field rather than
    discarding the whole batch — one malformed clause shouldn't cost a
    chapter's worth of otherwise-valid extraction.
    """
    cleaned = raw_output.strip()
    cleaned = (
        cleaned.removesuffix("```")
        .removeprefix("```json")
        .removeprefix("```")
        .strip()
    )
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = _salvage_truncated_json(cleaned)
        if parsed is None:
            logger.warning("[ClauseExtractor] JSON parse failed, salvage also failed. Raw output: %r", raw_output)
            return []
        logger.warning("[ClauseExtractor] JSON truncated — salvaged %d clause(s) from partial output.", len(parsed.get("clauses", [])))
    logger.info(f"[ClauseExtractor] PARSED JSON: {parsed}")
    valid_clauses = []
    for idx, c in enumerate(parsed.get("clauses", [])):
        if not isinstance(c, dict):
            continue
        text = str(c.get("text", "")).strip()
        if not text:
            logger.warning("[ClauseExtractor] Dropping clause — missing text: %r", c)
            continue  # nothing to display or ground — genuinely unrecoverable

        risk_raw = str(c.get("risk_level", "")).strip().lower()
        is_valid_risk = risk_raw in _VALID_RISK_LEVELS

        clause_num = str(c.get("clause_num", "")).strip()
        if not clause_num or len(clause_num) > _MAX_PLAUSIBLE_CLAUSE_NUM_LEN:
            if len(clause_num) > _MAX_PLAUSIBLE_CLAUSE_NUM_LEN:
                logger.warning(
                    "[ClauseExtractor] clause_num implausibly long (%d chars) — "
                    "treating as leaked text, not a real label: %r",
                    len(clause_num), clause_num[:60]
                )
            clause_num = f"unnumbered_{idx}"
            logger.info("[ClauseExtractor] Assigned placeholder %r for: %r", clause_num, text[:60])

        entry = {
            "clause_num": clause_num,
            "text": text,
            "risk_level": risk_raw if is_valid_risk else "invalid",
            "reason": str(c.get("reason", "")).strip(),
        }
        if not is_valid_risk:
            logger.warning(
                "[ClauseExtractor] Invalid risk_level %r — kept in output as "
                "dropped_invalid_risk: %r", risk_raw, c
            )
            entry["_invalid_risk"] = True
        valid_clauses.append(entry)
    return valid_clauses


def _normalize_text_clean(text: str) -> str:
    # Remove page markers, punctuation, quotes, and non-breaking spaces
    text = re.sub(r"\[p\.\d+\]", " ", text)
    text = re.sub(r"[\"'\u2018\u2019\u201c\u201d]", "", text)
    text = re.sub(r"\s*/\s*", "/", text)  # "his / her" -> "his/her" — PDF slash-spacing artifact
    return re.sub(r"\s+", " ", text).lower().strip()


@lru_cache(maxsize=16)
def _normalize_source_for_grounding(source_text: str) -> str:
    """
    Normalizes SOURCE text once and caches it. Same section text gets
    normalized only on first call — every subsequent clause/fragment
    check against it, and every repeat call in benchmark_model.py
    (same section, different candidate models), hits the cache instead
    of re-running regex over a multi-KB chapter string.
    maxsize=16 is plenty — a benchmark run touches at most a couple
    sections per invocation.
    """
    text = re.sub(r"\[p\.\d+\]", " ", source_text)
    return _normalize_text_clean(text)

@lru_cache(maxsize=16)
def _tokenize_source_for_grounding(source_text: str) -> tuple:
    return tuple(_normalize_source_for_grounding(source_text).split())


def _fragment_grounded(fragment: str, source_text: str, min_ratio: float = 0.82) -> bool:
    """
    Fuzzy substring check: slides a window of source tokens roughly the
    same length as the fragment and compares via SequenceMatcher. This
    tolerates single-word insertions/deletions ("The"), source PDF typos
    ("perc ent"), and Rule-5 defined-term substitutions, while still
    rejecting genuinely fabricated content (near-zero ratio anywhere).
    """
    frag_tokens = _normalize_text_clean(fragment).split()
    if len(frag_tokens) < 4:
        return False  # too short to fuzzy-match reliably

    frag_norm = " ".join(frag_tokens)
    source_tokens = _tokenize_source_for_grounding(source_text)
    n = len(frag_tokens)

    for window_size in (n, max(n - 2, 1), n + 2):
        if window_size > len(source_tokens):
            continue
        for i in range(len(source_tokens) - window_size + 1):
            window = " ".join(source_tokens[i:i + window_size])
            if SequenceMatcher(None, window, frag_norm).ratio() >= min_ratio:
                return True
    return False

def _is_grounded_in_source(clause_text: str, source_text: str, min_fragment_ratio: float = 0.6) -> bool:
    """
    Split on ':' too — merged list clauses ("aspects: X") need the
    intro and the bullet checked independently, since the LLM may
    merge the intro with a non-adjacent bullet from the source.

    Fragments are weighted by token length, not count. A short heading
    fragment ("Underwriting Standards") failing fuzzy match must not
    outvote a long, fully-grounded substantive sentence sitting right
    next to it — equal-weight-per-fragment was dropping legitimate
    clauses whose only "ungrounded" fragment was a markdown section
    heading swallowed into the clause text by the ':' split.

    Verified via scripts/analyse_drops.py across all 5 RBI docs:
    4/80 dropped_ungrounded clauses flip to included under this
    weighting, zero regressions (prompt-leak noise fragments like
    "penalty of" stay at ratio 0.00 under both formulas).
    """
    raw_fragments = [f.strip() for f in re.split(r"[;.:\n]", clause_text) if len(f.strip()) > 15]
    if not raw_fragments:
        return False

    weights = [len(f.split()) for f in raw_fragments]
    grounded_weight = sum(
        w for f, w in zip(raw_fragments, weights) if _fragment_grounded(f, source_text)
    )
    total_weight = sum(weights)
    return (grounded_weight / total_weight) >= min_fragment_ratio if total_weight else False


def _enforce_risk_rubric(clauses: list) -> list:
    """
    Deterministic override: any clause containing a hard signal word is
    forced to 'high', regardless of what the LLM assigned. Only applies to
    clauses still eligible for inclusion — dropped_invalid_risk clauses
    are dropped precisely because we couldn't trust their risk_level in
    the first place, so overriding it here would mask that they were
    ever invalid when inspecting the JSON.
    """
    for c in clauses:
        if c["status"] != "included":
            continue
        if _HIGH_SIGNAL_PATTERN.search(c["text"]) and c["risk_level"] != "high":
            logger.info("[RiskRubric] Overriding %s: %s -> high", c["clause_num"], c["risk_level"])
            c["risk_level"] = "high"
    return clauses


def _filter_illustrative(clauses: list) -> list:
    """
    Drops clauses that are illustrative examples of a rule, not the rule
    itself. These frequently contain "shall"/"must" (describing the
    scenario the rule applies to) and would otherwise get force-labeled
    high by _enforce_risk_rubric — a false positive on both extraction
    and risk classification.
    """
    dropped = 0

    for c in clauses:
        if _ILLUSTRATIVE_PREFIX_PATTERN.match(c["text"]):
            c["status"] = "dropped_illustrative"
            dropped += 1
    if dropped > 0:
        logger.info("[ClauseExtractor] Filtered %d illustrative clause(s)", dropped)
    return clauses


def attach_section_metadata(
    clauses: List[Dict[str, Any]],
    chapter_title: str,
    page_start: int | None = None,
    page_end: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Stamps each clause with its source section's chapter_title, page
    range, and the 'truncated' flag from the most recent extract_clauses()
    call (_last_call_metadata). Single source of truth for this step —
    extract_to_json.py and reextract_sections.py both call this instead
    of re-implementing the loop, so a patched section's clause schema
    can never silently diverge from a fresh full-run's schema.
    """
    truncated = _last_call_metadata.get("done_reason") == "length"
    for c in clauses:
        c["chapter_title"] = chapter_title
        c["page_start"] = page_start
        c["page_end"] = page_end
        c["truncated"] = truncated
    return clauses


def extract_clauses(chapter_text: str, model: str | None = None) -> List[Dict[str, Any]]:
    """
    Extract and risk-classify clauses from a chapter of regulatory text.

    Args:
        chapter_text: chapter_text field from docling_loader.py's chunk
                       output (includes inline [p.N] page markers — left
                       in intentionally as harmless context; a future
                       extension could have the LLM report per-clause
                       page numbers using them).

    Returns:
        List of dicts: {clause_num, text, risk_level, reason}.
        Returns [] if LLM output could not be parsed at all — never raises
        on parse failure. DOES raise requests.RequestException if Ollama
        itself is unreachable — that failure must not be masked.
    """
    if len(chapter_text.strip()) < 150:
        logger.info("[ClauseExtractor] Section too short (%d chars) — skipping LLM call.", len(chapter_text.strip()))
        return []
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        rubric=RISK_RUBRIC, chapter_text=chapter_text
    )
    raw_output = _call_ollama(prompt, model=model)
    logger.info(f"[ClauseExtractor] RAW OUTPUT: {raw_output}")
    clauses = _parse_clauses(raw_output)
    for c in clauses:
        if c.pop("_invalid_risk", False):
            c["status"] = "dropped_invalid_risk"
            continue
        if _is_grounded_in_source(c["text"], chapter_text):
            c["status"] = "included"
        else:
            logger.warning("[ClauseExtractor] Dropping non-grounded (fabricated?) clause: %r", c)
            c["status"] = "dropped_ungrounded"
    clauses = _filter_illustrative(clauses)
    clauses = _enforce_risk_rubric(clauses)
    return clauses
