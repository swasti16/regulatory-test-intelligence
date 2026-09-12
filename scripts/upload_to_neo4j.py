"""
Upload reviewed clause JSON (from extract_to_json.py) into Neo4j.

Reads data/extracted_clauses/{doc_id}.json, uploads ONLY clauses with
status == "included". Human review happens between extract_to_json.py
and this script — edit/delete entries (or flip status) in the JSON if a
reviewer disagrees before running this.

Validation gate: if data["validation_issues"] contains any "FAIL:" entries
(written by validate_node / post_process_extraction's checks), upload is
blocked for that file — the human-review checkpoint must be resolved
first. WARN entries do not block.
Run:
    python scripts/upload_to_neo4j.py                          # all JSON files
    python scripts/upload_to_neo4j.py data/extracted_clauses/X.json
"""
import json
import logging
import sys
import glob
import os

from src.graph.neo4j_client import Neo4jClient
from src.graph.graph_writer import write_regulation, write_clause, link_clause_to_regulation
from config.logging_config import setup_logging

setup_logging("upload_to_neo4j")
logger = logging.getLogger(__name__)

INPUT_DIR = "data/extracted_clauses"


def _blocking_failures(data: dict) -> list:
    """Returns FAIL: entries from validation_issues (post_process_extraction.py
    and LangGraph's validate_node both populate this field with FAIL:/WARN:
    prefixed strings). Missing/absent field = no known failures (older JSON
    predating this check) — treated as pass, not silently blocked."""
    issues = data.get("validation_issues", [])
    return [i for i in issues if i.startswith("FAIL:")]


def upload_file(client: Neo4jClient, json_path: str) -> bool:
    """Returns True if uploaded, False if blocked by validation FAILs."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    doc_id = data["doc_id"]

    failures = _blocking_failures(data)
    if failures:
        logger.error(f"{doc_id}: BLOCKED — {len(failures)} validation FAIL(s) unresolved:")
        for f_ in failures:
            logger.error(f"  {f_}")
        print(f"\n{doc_id}: UPLOAD BLOCKED — {len(failures)} FAIL(s). "
              f"Run post_process_extraction.py and resolve before uploading.")
        for f_ in failures:
            print(f"  {f_}")
        return False

    included = [c for c in data["clauses"] if c.get("status") == "included"]
    skipped = len(data["clauses"]) - len(included)

    logger.info(f"{doc_id}: uploading {len(included)} included clauses ({skipped} skipped)")

    write_regulation(client, doc_id=doc_id, title=doc_id)

    written = 0
    for c in included:
        cid = write_clause(
            client, doc_id=doc_id, chapter_title=c["chapter_title"],
            clause_num=c["clause_num"], text=c["text"],
            risk_level=c["risk_level"],
            page_start=c.get("page_start"), page_end=c.get("page_end"),
        )
        link_clause_to_regulation(client, doc_id=doc_id, clause_id=cid)
        written += 1

    print(f"{doc_id}: wrote {written} clauses to Neo4j ({skipped} skipped)")
    return True


def main():
    json_paths = [sys.argv[1]] if len(sys.argv) > 1 else \
        sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))

    if not json_paths:
        print(f"No JSON files found in {INPUT_DIR}. Run extract_to_json.py first.")
        return

    blocked = []
    with Neo4jClient() as client:
        try:
            client.verify_connectivity()
        except Exception as e:
            print(f"FATAL: Neo4j unreachable — {e}")
            sys.exit(1)
        for json_path in json_paths:
            try:
                if not upload_file(client, json_path):
                    blocked.append(json_path)
            except Exception as e:
                logger.error(f"Skipping {json_path} — upload failed: {e}")
                continue

    if blocked:
        print(f"\n{len(blocked)} file(s) blocked by validation FAILs: {blocked}")
        sys.exit(1)


if __name__ == "__main__":
    main()
