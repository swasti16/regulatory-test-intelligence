"""
Re-extract specific sections of an already-processed PDF and patch the
results back into its existing data/extracted_clauses/{doc_id}.json —
without re-running the whole PDF.

Use this after fixing a bug in clause_extractor.py (e.g. repeat_penalty,
num_ctx tuning) that only affected a few known-bad sections, rather than
re-running all 5 PDFs from scratch.

Run:
    python scripts/reextract_sections.py RBI_Credit_Debit_Card \
        "Chapter II - Conduct of Credit Card Business :: B. Role of the Board" \
        "Chapter II - Conduct of Credit Card Business :: C. Issue of Credit Cards" \
        "Chapter I - Preliminary :: Reserve Bank of India (Commercial Banks - Credit Cards and Debit Cards: Issuance and Conduct) Directions, 2025" \
        "Chapter I - Preliminary :: C. Definitions"

Matches chapter_title args EXACTLY against section["chapter_title"] as
produced by split_chapter_into_sections() — copy-paste from the existing
JSON's "chapter_title" field or the failed_sections list to avoid typos.
"""
import json
import logging
import os
import sys
import glob
from datetime import datetime

from src.ingestion.docling_loader import load_pdf, split_chapter_into_sections
from src.extraction.clause_extractor import extract_clauses
from config.settings import Settings
from config.logging_config import setup_logging

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

setup_logging("reextract_sections")
logger = logging.getLogger(__name__)

OUTPUT_DIR = "data/extracted_clauses"


def _find_pdf_path(doc_id: str) -> str:
    """Locates the source PDF for doc_id under Settings.RBI_PDF_DIR."""
    matches = glob.glob(os.path.join(Settings.RBI_PDF_DIR, f"{doc_id}.pdf"))
    if not matches:
        raise FileNotFoundError(f"No PDF found for doc_id={doc_id!r} in {Settings.RBI_PDF_DIR}")
    return matches[0]


def _load_existing_json(doc_id: str) -> dict:
    path = os.path.join(OUTPUT_DIR, f"{doc_id}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No existing extraction found at {path} — run extract_to_json.py first."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _find_all_sections(pdf_path: str, doc_id: str) -> list:
    """Reconstructs every section across the whole PDF, same logic as
    extract_to_json.py — needed so we can look up section_text for the
    requested chapter_titles."""
    chapters = load_pdf(pdf_path)
    all_sections = []
    for chapter in chapters:
        sections = split_chapter_into_sections(
            chapter_text=chapter["chapter_text"],
            chapter_title=chapter["chapter_title"],
            doc_id=doc_id,
            fallback_page=chapter["page_start"],
        )
        all_sections.extend(sections)
    return all_sections


def reextract(doc_id: str, target_titles: list) -> None:
    pdf_path = _find_pdf_path(doc_id)
    existing = _load_existing_json(doc_id)

    logger.info(f"Reloading {pdf_path} to locate {len(target_titles)} target section(s)...")
    all_sections = _find_all_sections(pdf_path, doc_id)
    sections_by_title = {s["chapter_title"]: s for s in all_sections}

    missing_titles = [t for t in target_titles if t not in sections_by_title]
    if missing_titles:
        logger.error(f"Could not find these chapter_titles in the PDF's sections: {missing_titles}")
        logger.error("Available titles (first 10 shown):")
        for t in list(sections_by_title.keys())[:10]:
            logger.error(f"  {t!r}")
        raise ValueError(f"{len(missing_titles)} target title(s) not found — aborting, no changes made.")

    # Strip out ALL old clauses belonging to the sections being redone —
    # avoids duplicate/stale entries sitting alongside the new ones.
    old_clauses = existing["clauses"]
    kept_clauses = [c for c in old_clauses if c["chapter_title"] not in target_titles]
    removed_count = len(old_clauses) - len(kept_clauses)
    logger.info(f"Removing {removed_count} stale clause(s) from {len(target_titles)} target section(s).")

    new_clauses = []
    still_failed = []
    for title in target_titles:
        section = sections_by_title[title]
        logger.info(f"Re-extracting: {title}")
        try:
            clauses = extract_clauses(section["section_text"])
        except Exception as e:
            logger.error(f"Section '{title}' failed again: {e}")
            still_failed.append(title)
            continue

        included_count = sum(1 for c in clauses if c["status"] == "included")
        logger.info(f"  -> {len(clauses)} extracted, {included_count} included")
        if clauses and included_count == 0:
            logger.error(f"  ZERO clauses survived for '{title}' — still broken, needs further investigation.")

        for c in clauses:
            c["chapter_title"] = title
            c["page_start"] = section["page_start"]
            c["page_end"] = section["page_end"]
            new_clauses.append(c)

    # Merge back
    merged_clauses = kept_clauses + new_clauses

    def _count(status):
        return sum(1 for c in merged_clauses if c["status"] == status)

    # failed_sections: drop any of our target_titles that succeeded this
    # time, keep everything else from the original run, add any that
    # failed again.
    old_failed = existing.get("summary", {}).get("failed_sections", [])
    remaining_old_failed = [f for f in old_failed if f not in target_titles]
    new_failed_sections = remaining_old_failed + still_failed

    existing["clauses"] = merged_clauses
    existing["summary"] = {
        "total_extracted": len(merged_clauses),
        "included": _count("included"),
        "dropped_ungrounded": _count("dropped_ungrounded"),
        "dropped_illustrative": _count("dropped_illustrative"),
        "failed_sections": new_failed_sections,
    }
    existing["last_reextracted_at"] = datetime.now().isoformat()
    existing["last_reextracted_sections"] = target_titles

    out_path = os.path.join(OUTPUT_DIR, f"{doc_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"\nPatched {out_path}")
    print(f"  Re-extracted: {len(target_titles) - len(still_failed)}/{len(target_titles)} section(s) succeeded")
    if still_failed:
        print(f"  Still failing: {still_failed}")
    print(f"  New summary: {existing['summary']}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    doc_id = sys.argv[1]
    target_titles = sys.argv[2:]
    reextract(doc_id, target_titles)


if __name__ == "__main__":
    main()
