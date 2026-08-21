"""
Upload reviewed clause JSON (from extract_to_json.py) into Neo4j.

Reads data/extracted_clauses/{doc_id}.json, uploads ONLY clauses with
status == "included". Human review happens between extract_to_json.py
and this script — edit/delete entries (or flip status) in the JSON if a
reviewer disagrees before running this.

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


def upload_file(client: Neo4jClient, json_path: str) -> None:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    doc_id = data["doc_id"]
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


def main():
    json_paths = [sys.argv[1]] if len(sys.argv) > 1 else \
        sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))

    if not json_paths:
        print(f"No JSON files found in {INPUT_DIR}. Run extract_to_json.py first.")
        return

    with Neo4jClient() as client:
        for json_path in json_paths:
            try:
                upload_file(client, json_path)
            except Exception as e:
                logger.error(f"Skipping {json_path} — upload failed: {e}")
                continue


if __name__ == "__main__":
    main()
