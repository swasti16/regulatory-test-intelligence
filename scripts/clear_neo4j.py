"""
Wipes ALL nodes/relationships from the configured Neo4j instance.
Use when clause data needs a full reset (e.g. after fixing extraction
bugs and wanting a clean re-upload). Irreversible — no confirmation
beyond the interactive prompt below.

Run:
    python scripts/clear_neo4j.py
"""
from src.graph.neo4j_client import Neo4jClient


def main():
    confirm = input("This will DELETE ALL nodes/relationships in Neo4j. Type 'DELETE' to confirm: ")
    if confirm != "DELETE":
        print("Aborted.")
        return

    with Neo4jClient() as client:
        def _work(tx):
            result = tx.run("MATCH (n) DETACH DELETE n")
            return result.consume().counters

        counters = client.execute_write(_work)
        print(f"Deleted {counters.nodes_deleted} nodes, {counters.relationships_deleted} relationships.")


if __name__ == "__main__":
    main()
