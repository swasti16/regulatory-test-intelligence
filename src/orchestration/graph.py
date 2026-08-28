"""
LangGraph wiring for the scoped single-PDF extraction pipeline.

Flow:
  load -> split -> extract -[router]-> advance | split_retry | mark_failed
  split_retry -[retry_router]-> advance | mark_failed
  mark_failed -> advance
  advance -[loop_router]-> extract (more sections) | write (done)
  write -> END

Scope: single PDF, in-process (no subprocess isolation — see
docling_loader.py comments on why extract_to_json.py uses subprocess
isolation for batch runs; that protection is intentionally NOT
replicated here, per the scoped-demo decision).
"""
from langgraph.graph import StateGraph, END

from src.orchestration.state import PipelineState
from src.orchestration.nodes import (
    load_node,
    split_node,
    extract_node,
    split_retry_node,
    mark_failed_node,
    advance_node,
    write_node,
)


def _route_after_extract(state: PipelineState) -> str:
    """
    Retry only if the LLM was actually called and produced zero survivors
    (real failure). If the section was too short to attempt (legitimately
    empty, e.g. a bare heading), skip straight to advance — matches
    extract_to_json.py's `if clauses and included_count == 0` semantics.
    """
    if not state["current_section_extraction_attempted"]:
        return "advance"
    if state["current_section_included_count"] > 0:
        return "advance"
    return "split_retry"


def _route_after_retry(state: PipelineState) -> str:
    """After split-retry: still 0 survived -> mark failed, else advance."""
    if state["current_section_included_count"] > 0:
        return "advance"
    return "mark_failed"


def _route_after_advance(state: PipelineState) -> str:
    """More sections left to process, or done -> write output."""
    if state["current_section_idx"] < len(state["sections"]):
        return "extract"
    return "write"


def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("load", load_node)
    graph.add_node("split", split_node)
    graph.add_node("extract", extract_node)
    graph.add_node("split_retry", split_retry_node)
    graph.add_node("mark_failed", mark_failed_node)
    graph.add_node("advance", advance_node)
    graph.add_node("write", write_node)

    graph.set_entry_point("load")
    graph.add_edge("load", "split")
    graph.add_edge("split", "extract")

    graph.add_conditional_edges(
        "extract",
        _route_after_extract,
        {"advance": "advance", "split_retry": "split_retry"},
    )
    graph.add_conditional_edges(
        "split_retry",
        _route_after_retry,
        {"advance": "advance", "mark_failed": "mark_failed"},
    )
    graph.add_edge("mark_failed", "advance")

    graph.add_conditional_edges(
        "advance",
        _route_after_advance,
        {"extract": "extract", "write": "write"},
    )
    graph.add_edge("write", END)

    return graph.compile()
