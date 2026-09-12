"""
LangGraph nodes for the scoped single-PDF extraction pipeline.

Each node wraps existing, already-tested functions from
src/ingestion/docling_loader.py and src/extraction/clause_extractor.py —
no extraction/parsing logic is duplicated here. Nodes only handle state
bookkeeping and sequencing.
"""
import logging
from typing import Any, Dict
import re
import json
import os
from datetime import datetime
from src.ingestion.docling_loader import (
    load_pdf,
    split_chapter_into_sections,
    split_definitions_section,
)
from src.extraction.clause_extractor import extract_clauses, attach_section_metadata
from src.orchestration.state import PipelineState
from config.settings import Settings
from collections import Counter


logger = logging.getLogger(__name__)
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

_LEAKED_CLAUSE_NUM_MAX_LEN = 30
_VALID_RISK_LEVELS = {"high", "medium", "low"}
_NOISE_TEXT_LITERALS = {"penalty of", "within x days", "shall", "must"}


def load_node(state: PipelineState) -> Dict[str, Any]:
    """Loads the PDF into chapter-level chunks (unchanged docling_loader call)."""
    chapters = load_pdf(state["pdf_path"])
    doc_id = chapters[0]["doc_id"] if chapters else ""
    logger.info(f"[load_node] Loaded {len(chapters)} chapter(s) from {state['pdf_path']}")
    return {"doc_id": doc_id, "_chapters": chapters}


def split_node(state: PipelineState) -> Dict[str, Any]:
    """
    Splits every chapter into sections (same branching logic
    extract_to_json.py's worker uses: Definitions chapters get
    split_definitions_section(), everything else gets
    split_chapter_into_sections()).
    """
    chapters = state["_chapters"]
    doc_id = state["doc_id"]
    all_sections = []

    for chapter in chapters:
        if "Definitions" in chapter["chapter_title"]:
            sections = split_definitions_section(
                chapter["chapter_text"], chapter["chapter_title"], doc_id, chapter["page_start"]
            )
        else:
            sections = split_chapter_into_sections(
                chapter_text=chapter["chapter_text"],
                chapter_title=chapter["chapter_title"],
                doc_id=doc_id,
                fallback_page=chapter["page_start"],
            )
        all_sections.extend(sections)

    logger.info(f"[split_node] {len(chapters)} chapter(s) -> {len(all_sections)} section(s)")
    return {
        "sections": all_sections,
        "current_section_idx": 0,
        "current_section_retried": False,
        "all_clauses": [],
        "failed_sections": [],
        "current_section_included_count": 0,
    }


def _split_with_overlap(text: str, overlap_sentences: int = 2) -> tuple[str, str]:
    """
    Splits text into two halves at a sentence boundary (never mid-sentence),
    with the last `overlap_sentences` of half_a repeated at the start of
    half_b. This trades some duplicate extraction (deduped downstream via
    the same clause_num#N logic extract_to_json.py already uses) for
    guaranteeing no clause is silently truncated across the cut point.

    Logs the split point and overlap content — needed to diagnose cases
    where a section STILL fails after retry: was the split itself badly
    placed (e.g., overlap landed inside a table/list, not prose), or is
    this a genuine extraction failure independent of chunking?
    """
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    if len(sentences) < 2:
        logger.warning(
            f"[split_retry] Section too short to split ({len(sentences)} sentence(s)) "
            f"— falling back to full text for both halves."
        )
        return text, ""

    mid = len(sentences) // 2
    overlap_start = max(0, mid - overlap_sentences)
    overlap_text = " ".join(sentences[overlap_start:mid])

    half_a = " ".join(sentences[:mid])
    half_b = " ".join(sentences[overlap_start:])

    logger.info(
        f"[split_retry] Split at sentence {mid}/{len(sentences)}. "
        f"Overlap ({len(sentences[overlap_start:mid])} sentence(s)): {overlap_text!r}"
    )
    logger.info(f"[split_retry] half_a ({len(half_a)} chars): {half_a[:200]!r}...")
    logger.info(f"[split_retry] half_b ({len(half_b)} chars): {half_b[:200]!r}...")

    return half_a, half_b


def extract_node(state: PipelineState) -> Dict[str, Any]:
    """
    Extracts clauses for the CURRENT section only (state["current_section_idx"]).
    Does not advance the index — that happens in advance_node after the
    router decides no retry is needed.
    """
    section = state["sections"][state["current_section_idx"]]
    logger.info(f"[extract_node] Extracting: {section['chapter_title']}")

    # extract_clauses() returns [] without calling the LLM if the section
    # text is under its internal 150-char threshold — that's a legitimate
    # "nothing here" result, not a failure. Track it explicitly so the
    # router doesn't misclassify empty headings as extraction failures.
    was_attempted = len(section["section_text"].strip()) >= 150

    try:
        clauses = extract_clauses(section["section_text"])
        clauses = attach_section_metadata(
            clauses, section["chapter_title"], section["page_start"], section["page_end"]
        )
        included_count = sum(1 for c in clauses if c["status"] == "included")
    except Exception as e:
        # Mirrors extract_to_json.py's per-section try/except — a timeout
        # or other Ollama failure on ONE section must not lose every
        # clause already extracted for this PDF. Treat as "attempted but
        # 0 survived" so the router sends it straight to mark_failed
        # (skips a pointless retry — a stalled Ollama call won't succeed
        # faster on retry within the same run).
        logger.error(f"[extract_node] '{section['chapter_title']}' extraction failed: {e}")
        clauses = []
        included_count = 0
    return {
        "all_clauses": state["all_clauses"] + clauses,
        "current_section_included_count": included_count,
        "current_section_extraction_attempted": was_attempted,
    }


def split_retry_node(state: PipelineState) -> Dict[str, Any]:
    """
    Retry path: splits the current section's text at a SENTENCE boundary
    (never mid-clause) with a small sentence-overlap between halves, so a
    clause near the midpoint can't be silently truncated. Overlap means
    some clauses appear in both halves' output — deduped by the same
    clause_num#N collision logic extract_to_json.py already applies
    (see post_process_extraction.py._dedupe_clause_nums for the pattern).
    """
    section = state["sections"][state["current_section_idx"]]
    half_a, half_b = _split_with_overlap(section["section_text"])

    logger.info(f"[split_retry_node] Retrying '{section['chapter_title']}' as 2 overlapping halves")

    merged_clauses = []
    for half_text in (half_a, half_b):
        if not half_text.strip():
            continue
        clauses = extract_clauses(half_text)
        clauses = attach_section_metadata(
            clauses, section["chapter_title"], section["page_start"], section["page_end"]
        )
        merged_clauses.extend(clauses)

    # Dedupe collisions from the overlap region — same (chapter_title,
    # clause_num) pattern as post_process_extraction.py._dedupe_clause_nums.
    from collections import Counter
    key_counts = Counter((c["chapter_title"], c["clause_num"]) for c in merged_clauses)
    seen = Counter()
    for c in merged_clauses:
        key = (c["chapter_title"], c["clause_num"])
        if key_counts[key] > 1:
            seen[key] += 1
            c["clause_num"] = f"{c['clause_num']}#{seen[key]}"

    included_count = sum(1 for c in merged_clauses if c["status"] == "included")
    logger.info(f"[split_retry_node] Retry produced {included_count} included clause(s)")

    return {
        "all_clauses": state["all_clauses"] + merged_clauses,
        "current_section_included_count": included_count,
        "current_section_retried": True,
    }


def mark_failed_node(state: PipelineState) -> Dict[str, Any]:
    """Retry (if any) still produced 0 clauses — record and move on."""
    section = state["sections"][state["current_section_idx"]]
    logger.warning(f"[mark_failed_node] '{section['chapter_title']}' — 0 clauses even after retry")
    return {"failed_sections": state["failed_sections"] + [section["chapter_title"]]}


def advance_node(state: PipelineState) -> Dict[str, Any]:
    """Moves to the next section, resetting per-section transient flags."""
    return {
        "current_section_idx": state["current_section_idx"] + 1,
        "current_section_retried": False,
        "current_section_included_count": 0,
    }


def write_node(state: PipelineState) -> Dict[str, Any]:
    """
    Writes final output to data/extracted_clauses/{doc_id}.json — same
    shape as extract_to_json.py's worker output, so downstream tools
    (post_process_extraction.py, upload_to_neo4j.py) work unchanged
    regardless of which pipeline produced the file.
    """
    clauses = state["all_clauses"]

    def _count(status):
        return sum(1 for c in clauses if c["status"] == status)

    result = {
        "doc_id": state["doc_id"],
        "source_pdf": state["pdf_path"],
        "extracted_at": datetime.now().isoformat(),
        "model": Settings.OLLAMA_MODEL,
        "pipeline": "langgraph",  # distinguishes from extract_to_json.py output
        "summary": {
            "total_extracted": len(clauses),
            "included": _count("included"),
            "dropped_ungrounded": _count("dropped_ungrounded"),
            "dropped_illustrative": _count("dropped_illustrative"),
            "dropped_invalid_risk": _count("dropped_invalid_risk"),
            "failed_sections": state["failed_sections"],
        },
        "validation_issues": state["validation_issues"],
        "clauses": clauses,
    }

    out_dir = "data/extracted_clauses"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{state['doc_id']}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"[write_node] Wrote {len(clauses)} clause(s) -> {out_path}")
    return {}


def fix_leaked_clause_nums_node(state: PipelineState) -> Dict[str, Any]:
    """
    Mirrors post_process_extraction.py._fix_leaked_clause_nums() — a
    clause_num longer than the plausible-label threshold means the LLM
    echoed clause TEXT into the numbering field. Reassigns a placeholder
    unique by list index (monotonic, file-wide unique by construction).
    Kept local rather than importing the standalone script, matching the
    existing pattern in split_retry_node's dedupe logic.
    """
    clauses = state["all_clauses"]
    fixed = 0
    for idx, c in enumerate(clauses):
        if len(c["clause_num"]) > _LEAKED_CLAUSE_NUM_MAX_LEN:
            c["clause_num"] = f"leaked_fixed_{idx}"
            fixed += 1
    if fixed:
        logger.info(f"[fix_leaked_clause_nums_node] Fixed {fixed} leaked clause_num(s)")
    return {"all_clauses": clauses}


def dedupe_clause_nums_node(state: PipelineState) -> Dict[str, Any]:
    """
    Document-wide dedupe — mirrors post_process_extraction.py._dedupe_clause_nums().
    split_retry_node only dedupes WITHIN its own overlap output; this
    catches collisions across the WHOLE document (e.g. two unrelated
    sections producing the same (chapter_title, clause_num) by chance),
    which split_retry_node's local dedupe cannot see.
    """
    clauses = state["all_clauses"]
    key_counts = Counter((c["chapter_title"], c["clause_num"]) for c in clauses)
    seen = Counter()
    changed = 0
    for c in clauses:
        key = (c["chapter_title"], c["clause_num"])
        if key_counts[key] > 1:
            seen[key] += 1
            new_num = f"{c['clause_num']}#{seen[key]}"
            if new_num != c["clause_num"]:
                changed += 1
            c["clause_num"] = new_num
    if changed:
        logger.info(f"[dedupe_clause_nums_node] Deduped {changed} clause_num collision(s)")
    return {"all_clauses": clauses}


def validate_node(state: PipelineState) -> Dict[str, Any]:
    """
    Mirrors post_process_extraction.py._validate_for_upload() — read-only
    FAIL/WARN checks before Neo4j upload. Does NOT block write_node;
    issues are logged + written into the output JSON's validation_issues
    field for human review, same as the standalone script's printed
    checklist (this project's "human review checkpoint" principle).
    """
    issues = []
    clauses = state["all_clauses"]
    doc_id = state["doc_id"]

    included = [c for c in clauses if c.get("status") == "included"]
    seen_ids = set()
    for c in included:
        cid = (doc_id, c.get("chapter_title"), c.get("clause_num"))
        if cid in seen_ids:
            issues.append(f"FAIL: duplicate clause_id for Neo4j MERGE key: {cid}")
        seen_ids.add(cid)

        if not c.get("text", "").strip():
            issues.append(f"FAIL: included clause with empty text — clause_num={c.get('clause_num')}")
        if c.get("risk_level") not in _VALID_RISK_LEVELS:
            issues.append(f"FAIL: invalid risk_level {c.get('risk_level')!r} — clause_num={c.get('clause_num')}")
        if len(c.get("clause_num", "")) > _LEAKED_CLAUSE_NUM_MAX_LEN:
            issues.append(f"FAIL: clause_num still leaked (>{_LEAKED_CLAUSE_NUM_MAX_LEN} chars) — {c['clause_num'][:50]}...")
        norm_text = c.get("text", "").strip().lower()
        if norm_text in _NOISE_TEXT_LITERALS or len(norm_text) < 15:
            issues.append(f"WARN: suspiciously short/noise-like included clause text: {c.get('text')!r}")
        if c.get("truncated"):
            issues.append(
                f"WARN: truncated clause (Ollama output cut off, may be incomplete) — "
                f"clause_num={c.get('clause_num')}"
            )

    null_pages = sum(1 for c in included if c.get("page_start") is None)
    if null_pages:
        issues.append(f"WARN: {null_pages}/{len(included)} included clause(s) have page_start=None")

    if len(clauses) < 20:
        issues.append(f"WARN: only {len(clauses)} total clauses extracted — unusually low, check for a botched/partial run")

    if not issues:
        logger.info("[validate_node] READY — no issues found.")
    else:
        fail_count = sum(1 for i in issues if i.startswith("FAIL:"))
        logger.warning(f"[validate_node] {len(issues)} issue(s) found ({fail_count} FAIL, {len(issues)-fail_count} WARN)")
        for issue in issues:
            logger.warning(f"[validate_node]   {issue}")

    return {"validation_issues": issues}
