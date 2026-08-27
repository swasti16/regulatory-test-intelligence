"""
Uploads data/sample_testcases.json into Neo4j as TestCase nodes + COVERED_BY
edges. Only status=="confirmed" links are written — "suggested" links (from
a future LLM-assisted matcher, Post-MVP) are silently skipped until a human
promotes them to "confirmed" in the source file.

This script is deliberately shaped like a Jira/TestRail import adapter would
be: read structured test-case data -> resolve clause references -> write
graph edges. Swapping "manual_seed" for a real Jira API pull would only
change how `test_cases` is populated, not this script's logic.

Idempotent: link_clause_to_testcase() uses MERGE, safe to re-run.

Run:
    python scripts/upload_test_cases.py
    python scripts/upload_test_cases.py path/to/other_testcases.json
"""
import json
import logging
import sys

from src.graph.neo4j_client import Neo4jClient
from src.graph.graph_writer import clause_id, link_clause_to_testcase
from config.logging_config import setup_logging

setup_logging("upload_test_cases")
logger = logging.getLogger(__name__)

DEFAULT_PATH = "data/sample_testcases.json"


def upload_file(client: Neo4jClient, json_path: str) -> None:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    test_cases = data["test_cases"]
    links_written = 0
    links_skipped_unconfirmed = 0
    missing_clauses = []

    for tc in test_cases:
        for cover in tc["covers"]:
            if cover.get("status") != "confirmed":
                links_skipped_unconfirmed += 1
                continue

            cid = clause_id(cover["doc_id"], cover["chapter_title"], cover["clause_num"])

            # link_clause_to_testcase's MATCH silently no-ops if the clause
            # doesn't exist in the graph — verify existence first so a typo'd
            # chapter_title/clause_num fails loud, not silent.
            def _exists(tx, **params):
                result = tx.run(
                    "MATCH (c:Clause {clause_id: $clause_id}) RETURN count(c) AS c",
                    **params,
                )
                return result.single()["c"] > 0

            if not client.execute_read(_exists, clause_id=cid):
                missing_clauses.append((tc["test_case_id"], cid))
                continue

            link_clause_to_testcase(
                client, clause_id=cid,
                test_case_id=tc["test_case_id"], test_case_title=tc["title"],
            )
            links_written += 1

    logger.info(f"{json_path}: {links_written} confirmed link(s) written, "
                f"{links_skipped_unconfirmed} unconfirmed link(s) skipped")
    print(f"Wrote {links_written} TestCase links from {len(test_cases)} test case(s).")
    if links_skipped_unconfirmed:
        print(f"Skipped {links_skipped_unconfirmed} unconfirmed (status != 'confirmed') link(s).")
    if missing_clauses:
        print(f"WARNING: {len(missing_clauses)} link(s) referenced clauses NOT FOUND in Neo4j "
              f"(check doc_id/chapter_title/clause_num against the uploaded extraction JSON):")
        for tc_id, cid in missing_clauses:
            print(f"  {tc_id} -> {cid}")


def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    with Neo4jClient() as client:
        upload_file(client, json_path)


if __name__ == "__main__":
    main()
