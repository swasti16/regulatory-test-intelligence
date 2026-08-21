"""
Extract clauses from a PDF (or all PDFs in Settings.RBI_PDF_DIR) and dump
them to data/extracted_clauses/{doc_id}.json — NO Neo4j writes.

Human-review checkpoint: every clause carries its `status` field
(included / dropped_ungrounded / dropped_illustrative) so a reviewer sees
exactly what will/won't be uploaded before anything touches the graph.
Separates the slow/non-deterministic step (LLM extraction) from the
fast/deterministic step (graph write) — see upload_to_neo4j.py.

Run:
    python scripts/extract_to_json.py                    # all PDFs in RBI_PDF_DIR
    python scripts/extract_to_json.py path/to/one.pdf     # single PDF
"""
import json
import logging
import os
import sys
import glob
from datetime import datetime
from collections import Counter
from src.ingestion.docling_loader import load_pdf, split_chapter_into_sections, split_definitions_section
from src.extraction.clause_extractor import extract_clauses, _last_call_metadata
from config.settings import Settings
from config.logging_config import setup_logging

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

setup_logging("extract_to_json")
logger = logging.getLogger(__name__)

OUTPUT_DIR = "data/extracted_clauses"


def extract_pdf(pdf_path: str) -> dict:
    """Runs extraction across every chapter/section of one PDF. No uploads."""
    chapters = load_pdf(pdf_path)
    doc_id = chapters[0]["doc_id"]
    logger.info(f"Processing {pdf_path} -> doc_id={doc_id} ({len(chapters)} chapters)")

    all_clauses = []
    failed_sections = []

    for chapter in chapters:
        if "Definitions" in chapter["chapter_title"]:
            sections = split_definitions_section(chapter["chapter_text"], chapter["chapter_title"], doc_id, chapter["page_start"])
        else:
            sections = split_chapter_into_sections(
                chapter_text=chapter["chapter_text"],
                chapter_title=chapter["chapter_title"],
                doc_id=doc_id,
                fallback_page=chapter["page_start"],
            )
        for section in sections:
            logger.info(f"Extracting: {section['chapter_title']}")
            try:
                clauses = extract_clauses(section["section_text"])
                included_count = sum(1 for c in clauses if c["status"] == "included")
                if clauses and included_count == 0:
                    logger.error(
                        f"ZERO clauses survived for '{section['chapter_title']}' "
                        f"({len(clauses)} attempted, all dropped) — likely repetition-loop "
                        f"or systematic grounding failure. Flagging for manual re-run."
                    )
                    failed_sections.append(f"{section['chapter_title']} (0 survived — needs retry)")
            except Exception as e:
                logger.error(f"Skipping section '{section['chapter_title']}' — extraction failed: {e}")
                failed_sections.append(section["chapter_title"])
                continue

            call_meta = dict(_last_call_metadata)  # snapshot before next call overwrites it
            if call_meta.get("done_reason") == "length":
                logger.error(f"TRUNCATED OUTPUT for '{section['chapter_title']}' — results may be incomplete")

            for c in clauses:
                c["chapter_title"] = section["chapter_title"]
                c["page_start"] = section["page_start"]
                c["page_end"] = section["page_end"]
                c["truncated"] = call_meta.get("done_reason") == "length"  # <-- per-clause flag
                all_clauses.append(c)

    # Disambiguate clause_num collisions within the same chapter_title —
    # happens when one LLM call's source text spans multiple numbered
    # lists that each restart at "(i)"/"1" (Docling didn't tag them as
    # separate section headers). Without this, Neo4j MERGE on the
    # composite key silently collapses distinct clauses into one node.
    key_counts = Counter((c["chapter_title"], c["clause_num"]) for c in all_clauses)
    seen = Counter()
    for c in all_clauses:
        key = (c["chapter_title"], c["clause_num"])
        if key_counts[key] > 1:
            seen[key] += 1
            c["clause_num"] = f"{c['clause_num']}#{seen[key]}"

    def _count(status):
        return sum(1 for c in all_clauses if c["status"] == status)

    return {
        "doc_id": doc_id,
        "source_pdf": pdf_path,
        "extracted_at": datetime.now().isoformat(),
        "model": Settings.OLLAMA_MODEL,
        "summary": {
            "total_extracted": len(all_clauses),
            "included": _count("included"),
            "dropped_ungrounded": _count("dropped_ungrounded"),
            "dropped_illustrative": _count("dropped_illustrative"),
            "failed_sections": failed_sections,
        },
        "clauses": all_clauses,
    }


def save_result(result: dict) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{result['doc_id']}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {out_path}")
    return out_path


def main():
    pdf_paths = [sys.argv[1]] if len(sys.argv) > 1 else \
        sorted(glob.glob(os.path.join(Settings.RBI_PDF_DIR, "*.pdf")))

    logger.info(f"Found {len(pdf_paths)} PDF(s) to process")

    for pdf_path in pdf_paths:
        try:
            result = extract_pdf(pdf_path)
            out_path = save_result(result)
            s = result["summary"]
            print(f"\n{result['doc_id']}: {s['included']} included, "
                  f"{s['dropped_ungrounded']} dropped(ungrounded), "
                  f"{s['dropped_illustrative']} dropped(illustrative) -> {out_path}")
        except Exception as e:
            logger.error(f"Skipping PDF {pdf_path} entirely — unrecoverable error: {e}")
            continue


if __name__ == "__main__":
    main()
