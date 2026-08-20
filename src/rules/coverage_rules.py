"""
Coverage Rules — deterministic gap detection over the traceability graph.

Implements the MVP rules from README.md:
  1. Missing Coverage      — clauses with zero linked test cases
  2. Low Coverage Threshold — regulations below LOW_COVERAGE_THRESHOLD % clause coverage
  3. High-Risk Prioritization — missing-coverage clauses filtered by risk_level: high

Design:
- Every rule here is a plain, parameterized Cypher query — no LLM involved in
  deciding what counts as a gap. This is the module that makes the project's
  core claim ("deterministic, auditable gap detection") true, not aspirational.
- All functions are read-only (execute_read) — this module never writes to
  the graph. Coverage state is a derived view, not stored state.
- doc_id=None means "across all regulations" — every query supports both a
  single-regulation and portfolio-wide view, since a QA lead may want either.
- Does NOT filter on is_test_fixture — that's a test-cleanup concern owned by
  tests/graph/conftest.py, not a coverage-semantics concern. Fixture nodes in
  a live query would only appear if cleanup itself failed, which is a bug the
  fixture teardown should catch, not something this module should mask.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from config.settings import settings
from src.graph.neo4j_client import Neo4jClient


def find_missing_coverage(
    client: Neo4jClient, doc_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Rule 1 — Missing Coverage.

    Returns every Clause with zero COVERED_BY relationships to a TestCase.
    Ordered by risk_level (high first) so the highest-consequence gaps surface
    at the top of any report built on this.

    Args:
        client: Connected Neo4jClient.
        doc_id: Optional Regulation.doc_id to scope to a single regulation.
                None checks across all regulations in the graph.

    Returns:
        List of dicts: {doc_id, regulation_title, clause_id, chapter_title,
        text, risk_level}. Empty list means full coverage (or no clauses).
    """
    def _work(tx, **params):
        result = tx.run(
            """
            MATCH (r:Regulation)-[:HAS_CLAUSE]->(c:Clause)
            WHERE NOT (c)-[:COVERED_BY]->(:TestCase)
              AND ($doc_id IS NULL OR r.doc_id = $doc_id)
            RETURN r.doc_id AS doc_id,
                   r.title AS regulation_title,
                   c.clause_id AS clause_id,
                   c.chapter_title AS chapter_title,
                   c.text AS text,
                   c.risk_level AS risk_level
            ORDER BY
                CASE c.risk_level
                    WHEN 'high' THEN 0
                    WHEN 'medium' THEN 1
                    WHEN 'low' THEN 2
                    ELSE 3
                END,
                c.chapter_title, c.clause_id
            """,
            **params,
        )
        return [record.data() for record in result]

    return client.execute_read(_work, doc_id=doc_id)


def get_coverage_summary(
    client: Neo4jClient, doc_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Coverage percentage per regulation — the building block for Rule 2
    (Low Coverage Threshold) and for any future dashboard.

    Args:
        client: Connected Neo4jClient.
        doc_id: Optional Regulation.doc_id to scope to a single regulation.
                None returns a row per regulation in the graph.

    Returns:
        List of dicts: {doc_id, title, total_clauses, covered_clauses,
        coverage_pct}. A regulation with zero clauses returns coverage_pct: 0.0
        rather than dividing by zero.
    """
    def _work(tx, **params):
        result = tx.run(
            """
            MATCH (r:Regulation)-[:HAS_CLAUSE]->(c:Clause)
            WHERE $doc_id IS NULL OR r.doc_id = $doc_id
            WITH r,
                 count(c) AS total_clauses,
                 count(CASE WHEN (c)-[:COVERED_BY]->(:TestCase) THEN 1 END) AS covered_clauses
            RETURN r.doc_id AS doc_id,
                   r.title AS title,
                   total_clauses,
                   covered_clauses,
                   CASE WHEN total_clauses = 0 THEN 0.0
                        ELSE toFloat(covered_clauses) / total_clauses * 100
                   END AS coverage_pct
            ORDER BY coverage_pct ASC
            """,
            **params,
        )
        return [record.data() for record in result]

    return client.execute_read(_work, doc_id=doc_id)


def find_low_coverage_regulations(
    client: Neo4jClient,
    threshold: Optional[float] = None,
    doc_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Rule 2 — Low Coverage Threshold.

    Returns regulations whose clause coverage % falls below `threshold`.
    Filtering happens in Python over get_coverage_summary()'s output rather
    than a second Cypher query — coverage_pct is already computed once here,
    no need to duplicate the aggregation.

    Args:
        client: Connected Neo4jClient.
        threshold: Coverage % floor. Defaults to settings.LOW_COVERAGE_THRESHOLD
                   (80, per config/settings.py) if not given explicitly.
        doc_id: Optional single-regulation scope.

    Returns:
        Subset of get_coverage_summary()'s rows where coverage_pct < threshold.
    """
    effective_threshold = (
        threshold if threshold is not None else settings.LOW_COVERAGE_THRESHOLD
    )
    summary = get_coverage_summary(client, doc_id=doc_id)
    return [row for row in summary if row["coverage_pct"] < effective_threshold]


def find_high_risk_gaps(
    client: Neo4jClient, doc_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Rule 3 — High-Risk Prioritization.

    Returns missing-coverage clauses (Rule 1) filtered to risk_level: high.
    This is the report a compliance/QA lead should look at first — clauses
    carrying hard deadlines, penalties, or mandatory disclosure obligations
    (per the risk rubric in clause_extractor.py) that currently have zero
    test coverage.

    Args:
        client: Connected Neo4jClient.
        doc_id: Optional single-regulation scope.

    Returns:
        Subset of find_missing_coverage()'s rows where risk_level == "high".
    """
    return [
        row for row in find_missing_coverage(client, doc_id=doc_id)
        if row["risk_level"] == "high"
    ]
