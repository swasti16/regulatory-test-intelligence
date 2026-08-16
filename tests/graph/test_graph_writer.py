"""
Tests for src.graph.graph_writer.

Fast tests: mock Neo4jClient.execute_write, assert on the Cypher text
and params passed — verifies query CONSTRUCTION, not database behavior.
Slow tests (@pytest.mark.slow): real Aura via conftest's real_client
fixture — verifies actual MERGE idempotency, which a mock cannot prove.
"""
from unittest.mock import MagicMock
import pytest

from src.graph.graph_writer import (
    clause_id,
    write_regulation,
    write_clause,
    link_clause_to_regulation,
    link_clause_to_testcase,
)


# ======== Fast Unit Tests — Query Construction ==================================

class TestClauseIdComposition:
    def test_composite_id_format(self):
        assert clause_id("155MD", "Chapter II - Conduct", "3.1") == \
            "155MD_Chapter II - Conduct_3.1"

    def test_same_clause_num_different_chapters_differ(self):
        id1 = clause_id("155MD", "Chapter I", "1")
        id2 = clause_id("155MD", "Chapter II", "1")
        assert id1 != id2


class TestWriteRegulationCallsExecuteWrite:
    def test_calls_execute_write_with_work_fn(self):
        mock_client = MagicMock()
        write_regulation(mock_client, doc_id="155MD", title="Credit Card Directions")
        mock_client.execute_write.assert_called_once()
        _, kwargs = mock_client.execute_write.call_args
        assert kwargs["doc_id"] == "155MD"
        assert kwargs["title"] == "Credit Card Directions"
        assert kwargs["is_test_fixture"] is False  # production default


class TestWriteClauseReturnsCompositeId:
    def test_returns_clause_id(self):
        mock_client = MagicMock()
        result = write_clause(
            mock_client, doc_id="155MD", chapter_title="Chapter I",
            clause_num="1", text="...", risk_level="high",
        )
        assert result == clause_id("155MD", "Chapter I", "1")

    def test_optional_page_fields_default_none(self):
        mock_client = MagicMock()
        write_clause(
            mock_client, doc_id="155MD", chapter_title="Chapter I",
            clause_num="1", text="...", risk_level="low",
        )
        _, kwargs = mock_client.execute_write.call_args
        assert kwargs["page_start"] is None
        assert kwargs["page_end"] is None


class TestLinkFunctionsCallExecuteWrite:
    def test_link_clause_to_regulation(self):
        mock_client = MagicMock()
        link_clause_to_regulation(mock_client, doc_id="155MD", clause_id="155MD_ChI_1")
        mock_client.execute_write.assert_called_once()

    def test_link_clause_to_testcase(self):
        mock_client = MagicMock()
        link_clause_to_testcase(
            mock_client, clause_id="155MD_ChI_1",
            test_case_id="TC001", test_case_title="Notice period enforced",
        )
        mock_client.execute_write.assert_called_once()
        _, kwargs = mock_client.execute_write.call_args
        assert kwargs["test_case_title"] == "Notice period enforced"


# ======== Slow Integration Tests — Real Neo4j Aura ==================================

@pytest.mark.slow
class TestRealIdempotency:
    """
    The actual thing worth proving with a real database: writing the
    same clause twice must NOT create two nodes. A mock cannot verify
    this — it has no concept of graph state.
    """

    def test_write_regulation_twice_creates_one_node(self, real_client):
        write_regulation(real_client, doc_id="TEST_reg_1", title="Fixture Reg",
                         is_test_fixture=True)
        write_regulation(real_client, doc_id="TEST_reg_1", title="Fixture Reg",
                         is_test_fixture=True)

        def _count(tx):
            result = tx.run(
                "MATCH (r:Regulation {doc_id: $doc_id}) RETURN count(r) AS c",
                doc_id="TEST_reg_1",
            )
            return result.single()["c"]

        count = real_client.execute_read(_count)
        assert count == 1, "MERGE should not have created a duplicate node"

    def test_write_clause_twice_creates_one_node(self, real_client):
        write_clause(
            real_client, doc_id="TEST_reg_1", chapter_title="Chapter I",
            clause_num="1", text="Fixture clause text", risk_level="high",
            is_test_fixture=True,
        )
        write_clause(
            real_client, doc_id="TEST_reg_1", chapter_title="Chapter I",
            clause_num="1", text="Fixture clause text", risk_level="high",
            is_test_fixture=True,
        )

        def _count(tx):
            result = tx.run(
                "MATCH (c:Clause {clause_id: $cid}) RETURN count(c) AS c",
                cid=clause_id("TEST_reg_1", "Chapter I", "1"),
            )
            return result.single()["c"]

        count = real_client.execute_read(_count)
        assert count == 1

    def test_full_chain_regulation_clause_relationship(self, real_client):
        """End-to-end: write both nodes, link them, verify the edge exists."""
        write_regulation(real_client, doc_id="TEST_reg_2", title="Fixture Reg 2",
                         is_test_fixture=True)
        cid = write_clause(
            real_client, doc_id="TEST_reg_2", chapter_title="Chapter I",
            clause_num="1", text="...", risk_level="medium",
            is_test_fixture=True,
        )
        link_clause_to_regulation(real_client, doc_id="TEST_reg_2", clause_id=cid)

        def _check_edge(tx):
            result = tx.run(
                """
                MATCH (r:Regulation {doc_id: $doc_id})-[:HAS_CLAUSE]->(c:Clause {clause_id: $cid})
                RETURN count(*) AS c
                """,
                doc_id="TEST_reg_2", cid=cid,
            )
            return result.single()["c"]

        assert real_client.execute_read(_check_edge) == 1
