"""
Manual pipeline dry-run — NOT a pytest test.

Wires ingestion -> extraction -> graph_writer end-to-end against a real
regulatory PDF, so extracted clauses can be inspected in Neo4j Browser
before hand-authoring TestCase seed data (data/sample_testcases.json).

Run directly:
    python scripts/run_pipeline.py

Writes real (non-fixture) Regulation/Clause nodes to your configured
Neo4j Aura instance — these are NOT cleaned up automatically (unlike
test fixtures marked is_test_fixture=true). This is intentional: this
IS your real demo graph data.
"""
import logging

from src.ingestion.docling_loader import load_pdf
from src.extraction.clause_extractor import extract_clauses
from src.graph.neo4j_client import Neo4jClient
from src.graph.graph_writer import write_regulation, write_clause, link_clause_to_regulation
from config.settings import Settings

import glob
import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run() -> None:
    pdf_paths = sorted(glob.glob(os.path.join(Settings.RBI_PDF_DIR, "*.pdf")))
    logger.info("Found %d PDFs to process: %s", len(pdf_paths), pdf_paths)

    with Neo4jClient() as client:
        for pdf_path in pdf_paths:
            chapters = load_pdf(pdf_path)
            doc_id = chapters[0]["doc_id"]
            logger.info("Processing %s -> doc_id=%s (%d chapters)", pdf_path, doc_id, len(chapters))

            write_regulation(client, doc_id=doc_id, title=doc_id)  # filename as title placeholder
            logger.info("Wrote Regulation node: %s", doc_id)

            total_clauses = 0
            for chapter in chapters:
                logger.info("Extracting clauses from: %s", chapter["chapter_title"])
                clauses = extract_clauses(chapter["chapter_text"])
                logger.info("  -> extracted %d clauses", len(clauses))

                for clause in clauses:
                    cid = write_clause(
                        client,
                        doc_id=doc_id,
                        chapter_title=chapter["chapter_title"],
                        clause_num=clause["clause_num"],
                        text=clause["text"],
                        risk_level=clause["risk_level"],
                        page_start=chapter["page_start"],
                        page_end=chapter["page_end"],
                    )
                    link_clause_to_regulation(client, doc_id=doc_id, clause_id=cid)
                    total_clauses += 1

            logger.info("Finished %s. Total clauses written: %d", doc_id, total_clauses)

if __name__ == "__main__":
    run()