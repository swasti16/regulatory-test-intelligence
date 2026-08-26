"""
Model Benchmark — Compares candidate Ollama models on extraction speed,
clause count, and output quality against Chapter II (long) and Chapter III (short)
from RBI_Credit_Debit_Card.pdf using sub-chapter section splitting.

NO Neo4j writes — pure isolated evaluation.

Run: python scripts/benchmark_model.py
"""
import glob
import json
import logging
import os
import time
from typing import Any, Dict, List

from config.logging_config import setup_logging
from config.settings import Settings
from src.extraction.clause_extractor import extract_clauses
from src.ingestion.docling_loader import load_pdf, split_chapter_into_sections

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

RESULTS_PATH = "benchmark_results.json"

setup_logging("benchmark_models", level=logging.WARNING)

PDF_PATH = os.path.join(Settings.RBI_PDF_DIR, "RBI_Credit_Debit_Card.pdf")

# Models to compare — ensure these are pulled in Ollama
CANDIDATE_MODELS = [
    "llama3.2:3b",
    # "llama3.2:1b",
    # "qwen2.5:1.5b",
]

# Minimum character threshold to skip empty parent headings (e.g., "## B. Role of the Board")
MIN_SECTION_LEN = 50


def _save_incremental(results: List[Dict[str, Any]]) -> None:
    """Writes current results after every model+chapter combo completes —
    a multi-hour CPU run losing all progress on a crash/hang isn't
    acceptable; each combo now costs real wall-clock time to redo."""
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def run_benchmark() -> None:
    print(f"Loading and parsing {PDF_PATH} via Docling (one-time cost)...")
    chapters = load_pdf(PDF_PATH)
    doc_id = chapters[0]["doc_id"]

    long_chapter = next((c for c in chapters if "Chapter II" in c["chapter_title"]), None)
    short_chapter = next((c for c in chapters if "Chapter III" in c["chapter_title"]), None)

    if not long_chapter or not short_chapter:
        raise RuntimeError("Could not locate Chapter II or Chapter III in parsed PDF.")

    test_chapters = {
        "long (Chapter II)": long_chapter,
        "short (Chapter III)": short_chapter,
    }

    results: List[Dict[str, Any]] = []

    for model in CANDIDATE_MODELS:
        print(f"\n{'='*80}\nBENCHMARKING MODEL: {model}\n{'='*80}")

        for chapter_label, chapter in test_chapters.items():
            sections = split_chapter_into_sections(
                chapter_text=chapter["chapter_text"],
                chapter_title=chapter["chapter_title"],
                doc_id=doc_id,
                fallback_page=chapter["page_start"],
            )

            # Filter out empty parent header fragments
            active_sections = [
                s for s in sections if len(s["section_text"].strip()) >= MIN_SECTION_LEN
            ]

            print(f"\n--- Chapter: {chapter_label} ({len(active_sections)} active sections) ---")
            
            chapter_start_time = time.perf_counter()
            all_extracted_clauses: List[Dict[str, Any]] = []
            failed_sections = 0

            for idx, sec in enumerate(active_sections, start=1):
                sec_title = sec["chapter_title"]
                sec_text = sec["section_text"]

                try:
                    clauses = extract_clauses(sec_text, model=model)
                    all_extracted_clauses.extend(clauses)
                except Exception as e:
                    failed_sections += 1
                    print(f"  [!] Section {idx}/{len(active_sections)} failed ({sec_title}): {e}")

            elapsed = time.perf_counter() - chapter_start_time
            status = "OK" if failed_sections == 0 else f"PARTIAL ({failed_sections} failed)"

            # Print quick sample
            print(f"  Time: {elapsed:.1f}s | Status: {status} | Total Clauses: {len(all_extracted_clauses)}")
            for c in all_extracted_clauses[:3]:
                print(f"    [{c.get('risk_level', 'N/A')}] Clause {c.get('clause_num', '?')}: {c.get('text', '')[:90]}...")

            results.append({
                "model": model,
                "chapter": chapter_label,
                "sections_processed": len(active_sections),
                "failed_sections": failed_sections,
                "elapsed_seconds": round(elapsed, 1),
                "status": status,
                "clause_count": len(all_extracted_clauses),
                "clauses": all_extracted_clauses,
            })
            _save_incremental(results)
            print(f"  [saved progress to {RESULTS_PATH}]")

    print(f"\n[+] Full benchmark results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run_benchmark()
