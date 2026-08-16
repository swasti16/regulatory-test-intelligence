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
import logging
from typing import Any, Dict, List
import re
import requests

from config.settings import settings

logger = logging.getLogger(__name__)

_VALID_RISK_LEVELS = {"high", "medium", "low"}

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

Given the following chapter text from a banking regulation document,
identify each individual numbered clause and classify its risk level.

Rules:
- If clauses are explicitly numbered in the text (e.g. "3.1", "(a)"), use
  that exact numbering as clause_num.
- If no explicit numbering exists, assign sequential numbers starting at "1"
  in the order clauses appear.
- Each clause's "text" must be the clause content verbatim from the source
  — do not paraphrase or summarize.
- risk_level must be exactly one of: high, medium, low.
- CRITICAL: - If the clause text contains a quoted/defined term (e.g. "Cardholder"),
  represent it using single quotes instead: 'Cardholder'. Do NOT use double
  quotes anywhere inside a JSON string value — double quotes are reserved
  for JSON syntax only.

Chapter Text:
{chapter_text}

Respond ONLY with valid JSON in this exact shape, no extra text, no markdown
fences:
{{
  "clauses": [
    {{"clause_num": "...", "text": "...", "risk_level": "...", "reason": "..."}}
  ]
}}
"""


def _call_ollama(prompt: str) -> str:
    """
    Sends a single generate request to the local Ollama REST API.
    Raises requests.RequestException on network/connection failure —
    callers let this propagate (unlike JSON-parse failures below, a dead
    Ollama connection is a real error, not something to silently mask).
    """
    response = requests.post(
        f"{settings.OLLAMA_BASE_URL}/api/generate",
        json={
            "model": settings.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0},
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


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
        logger.warning("[ClauseExtractor] JSON parse failed. Raw output: %r", raw_output)
        return []
    logger.info(f"[ClauseExtractor] PARSED JSON: {parsed}")
    valid_clauses = []
    for c in parsed.get("clauses", []):
        if not isinstance(c, dict):
            continue
        risk = str(c.get("risk_level", "")).strip().lower()
        if risk not in _VALID_RISK_LEVELS:
            logger.warning("[ClauseExtractor] Dropping clause — invalid risk_level: %r", c)
            continue
        if not c.get("clause_num") or not c.get("text"):
            logger.warning("[ClauseExtractor] Dropping clause — missing required field: %r", c)
            continue
        valid_clauses.append({
            "clause_num": str(c["clause_num"]).strip(),
            "text": str(c["text"]).strip(),
            "risk_level": risk,
            "reason": str(c.get("reason", "")).strip(),
        })
    return valid_clauses


def extract_clauses(chapter_text: str) -> List[Dict[str, Any]]:
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
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(rubric=RISK_RUBRIC, chapter_text=chapter_text)
    raw_output = _call_ollama(prompt)
    logger.info(f"[ClauseExtractor] RAW OUTPUT: {raw_output}")
    return _parse_clauses(raw_output)
