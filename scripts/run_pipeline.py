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

from src.ingestion.docling_loader import load_pdf, split_chapter_into_sections
from src.extraction.clause_extractor import extract_clauses
from src.graph.neo4j_client import Neo4jClient
from src.graph.graph_writer import write_regulation, write_clause, link_clause_to_regulation
from config.settings import Settings
from config.logging_config import setup_logging
import glob
import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

setup_logging("run_pipeline")
logger = logging.getLogger(__name__)


def run() -> None:
    pdf_paths = sorted(glob.glob(os.path.join(Settings.RBI_PDF_DIR, "*.pdf")))
    logger.info(f"Found {len(pdf_paths)} PDFs to process: {pdf_paths}")

    with Neo4jClient() as client:
        for pdf_path in pdf_paths:
            try:
                chapters = load_pdf(pdf_path)
                doc_id = chapters[0]["doc_id"]
                logger.info(f"Processing {pdf_path} -> doc_id={doc_id} ({len(chapters)} chapters)")

                write_regulation(client, doc_id=doc_id, title=doc_id)
                logger.info(f"Wrote Regulation node: {doc_id}")

                total_clauses = 0
                for chapter in chapters:
                    sections = split_chapter_into_sections(
                        chapter_text=chapter["chapter_text"],
                        chapter_title=chapter["chapter_title"],
                        doc_id=doc_id,
                        fallback_page=chapter["page_start"],
                    )
                    logger.info(
                        f"{chapter['chapter_title']}: split into {len(sections)} section(s) for extraction"
                    )

                    for section in sections:
                        logger.info(f"Extracting clauses from: {section['chapter_title']}")
                        try:
                            clauses = extract_clauses(section["section_text"])
                        except Exception as e:
                            logger.error(
                                f"Skipping section '{section['chapter_title']}' — extraction failed: {e}")
                            continue
                        logger.info(f"  -> extracted {len(clauses)} clauses")

                        for clause in clauses:
                            cid = write_clause(
                                client, doc_id=doc_id, chapter_title=section["chapter_title"],
                                clause_num=clause["clause_num"], text=clause["text"],
                                risk_level=clause["risk_level"],
                                page_start=section["page_start"], page_end=section["page_end"],
                            )
                            link_clause_to_regulation(client, doc_id=doc_id, clause_id=cid)
                            total_clauses += 1
                logger.info(f"Finished {doc_id}. Total clauses written: {total_clauses}")

            except Exception as e:
                logger.error(f"Skipping PDF {pdf_path} entirely — unrecoverable error: {e}")
                continue


if __name__ == "__main__":
    run()
