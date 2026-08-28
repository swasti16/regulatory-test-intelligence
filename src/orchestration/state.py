"""
Shared state schema for the LangGraph extraction pipeline.

Design: a single TypedDict flows through every node — each node reads
what it needs, writes back an updated dict. LangGraph merges returned
dict keys into the running state automatically.

Scope: single-PDF extraction with grounding-failure retry (split-half
strategy). Does NOT cover multi-PDF batch, subprocess isolation, or
Neo4j writes — those remain script-driven (extract_to_json.py,
upload_to_neo4j.py) per the project's scoped-demo decision.
"""
from typing import TypedDict, List, Dict, Any


class PipelineState(TypedDict):
    pdf_path: str
    doc_id: str

    # Populated by split_node — each dict matches split_chapter_into_sections()
    # output: {doc_id, chapter_title, section_text, page_start, page_end}
    sections: List[Dict[str, Any]]

    # Index into `sections` — which section extract_node is currently on
    current_section_idx: int

    # Whether the CURRENT section has already used its one retry attempt
    current_section_retried: bool

    # Accumulates across all sections — final output written by write_node
    all_clauses: List[Dict[str, Any]]

    # Section chapter_titles where even the retry produced 0 survived clauses
    failed_sections: List[str]

        # Transient: how many clauses had status=="included" after the most
    # recent extract_node/split_retry_node call — read by the router to
    # decide whether to retry, and reset each time a section starts.
    current_section_included_count: int

    # Transient: True if extract_clauses() actually made an LLM call for
    # the current section (text >= 150 chars). If False, a 0-clause result
    # means "legitimately empty section" (e.g. a heading-only chunk), NOT
    # a failure — retry must be skipped, matching extract_to_json.py's
    # `if clauses and included_count == 0` behavior.
    current_section_extraction_attempted: bool

    # Transient: chapters from load_node, consumed by split_node.
    _chapters: List[Dict[str, Any]]
