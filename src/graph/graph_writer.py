"""
Graph Writer — persists extracted regulatory clauses into Neo4j.

Design:
- Pure "dict -> graph" persistence layer. Takes already-extracted clause
  data (doc_id, chapter_title, clause_num, text, risk_level) as plain
  arguments — has NO knowledge of Docling, LLM extraction, or Ollama.
  Extraction (src/extraction/, not yet built) is a separate module whose
  job ends where this one begins.
- All writes go through Neo4jClient.execute_write() using MERGE for
  idempotency — safe to re-run against the same document without
  duplicating nodes.
- clause_id is a composite key: {doc_id}_{chapter_title}_{clause_num}.
  Required because clause numbers repeat across chapters/documents —
  raw clause_num alone is not unique.
- is_test_fixture: bool = False on node-creating functions exists ONLY
  for test cleanup (see tests/graph/conftest.py). Production callers
  never pass it. Kept as a single atomic MERGE+SET rather than a
  separate "mark as fixture" write, so a mid-test crash can't leave an
  unmarked (uncleanable) node behind.
"""
from typing import Optional

from src.graph.neo4j_client import Neo4jClient


def clause_id(doc_id: str, chapter_title: str, clause_num: str) -> str:
    """Composite clause ID — the single source of truth for this format."""
    return f"{doc_id}_{chapter_title}_{clause_num}"


def write_regulation(
    client: Neo4jClient,
    doc_id: str,
    title: str,
    is_test_fixture: bool = False,
) -> None:
    """MERGE a Regulation node. Idempotent — safe to call repeatedly."""
    def _work(tx, **params):
        tx.run(
            """
            MERGE (r:Regulation {doc_id: $doc_id})
            ON CREATE SET r.title = $title, r.is_test_fixture = $is_test_fixture
            """,
            **params,
        )
    client.execute_write(
        _work, doc_id=doc_id, title=title, is_test_fixture=is_test_fixture
    )


def write_clause(
    client: Neo4jClient,
    doc_id: str,
    chapter_title: str,
    clause_num: str,
    text: str,
    risk_level: str,
    page_start: Optional[int] = None,
    page_end: Optional[int] = None,
    is_test_fixture: bool = False,
) -> str:
    """
    MERGE a Clause node. Returns the composite clause_id so the caller
    (pipeline orchestration) can immediately pass it to
    link_clause_to_regulation() without recomputing it.
    """
    cid = clause_id(doc_id, chapter_title, clause_num)

    def _work(tx, **params):
        tx.run(
            """
            MERGE (c:Clause {clause_id: $clause_id})
            ON CREATE SET
                c.text = $text,
                c.risk_level = $risk_level,
                c.chapter_title = $chapter_title,
                c.page_start = $page_start,
                c.page_end = $page_end,
                c.is_test_fixture = $is_test_fixture
            """,
            **params,
        )
    client.execute_write(
        _work,
        clause_id=cid,
        text=text,
        risk_level=risk_level,
        chapter_title=chapter_title,
        page_start=page_start,
        page_end=page_end,
        is_test_fixture=is_test_fixture,
    )
    return cid


def link_clause_to_regulation(client: Neo4jClient, doc_id: str, clause_id: str) -> None:
    """
    MERGE the HAS_CLAUSE relationship. Requires both nodes to already
    exist (MATCH, not MERGE, on the nodes) — this function only owns
    the relationship, not node creation.
    """
    def _work(tx, **params):
        tx.run(
            """
            MATCH (r:Regulation {doc_id: $doc_id})
            MATCH (c:Clause {clause_id: $clause_id})
            MERGE (r)-[:HAS_CLAUSE]->(c)
            """,
            **params,
        )
    client.execute_write(_work, doc_id=doc_id, clause_id=clause_id)


def link_clause_to_testcase(
    client: Neo4jClient,
    clause_id: str,
    test_case_id: str,
    test_case_title: Optional[str] = None,
) -> None:
    """
    MERGE a TestCase node (creating it if this is the first time we've
    seen this test_case_id) and the COVERED_BY relationship to an
    existing Clause. Clause must already exist (MATCH).
    """
    def _work(tx, **params):
        tx.run(
            """
            MATCH (c:Clause {clause_id: $clause_id})
            MERGE (t:TestCase {test_case_id: $test_case_id})
            ON CREATE SET t.title = $test_case_title
            MERGE (c)-[:COVERED_BY]->(t)
            """,
            **params,
        )
    client.execute_write(
        _work,
        clause_id=clause_id,
        test_case_id=test_case_id,
        test_case_title=test_case_title,
    )
