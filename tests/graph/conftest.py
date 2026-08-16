"""
Real Neo4j Aura fixture for slow integration tests in tests/graph/.
Requires valid NEO4J_URI/NEO4J_PASSWORD in .env — tests using this
fixture are marked @pytest.mark.slow and skip cleanly if unreachable.
"""
import pytest
from src.graph.neo4j_client import Neo4jClient


@pytest.fixture
def real_client():
    """
    Yields a real Neo4jClient connected to Aura. Teardown deletes ONLY
    nodes explicitly marked is_test_fixture=true — see graph_writer.py
    docstring for why a boolean flag was chosen over a doc_id string
    prefix (collision risk with real seed data like TestCase IDs).
    """
    client = Neo4jClient()
    try:
        client.verify_connectivity()
    except Exception as exc:
        pytest.skip(f"Neo4j Aura unreachable — skipping integration test: {exc}")

    yield client

    def _cleanup(tx):
        # DETACH DELETE removes this node's relationships too, but does
        # NOT cascade to the other end — every fixture-created node
        # (Regulation, Clause, TestCase) must independently carry the
        # flag, or it survives as an orphan after this query.
        tx.run("MATCH (n) WHERE n.is_test_fixture = true DETACH DELETE n")

    client.execute_write(_cleanup)
    client.close()
