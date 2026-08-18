"""
Model benchmark — compares candidate Ollama models on extraction speed
and output quality against two fixed real chapters (one long, one short)
from RBI_Credit_Debit_Card.pdf. NO Neo4j writes — pure isolated comparison.

Run: python scripts/benchmark_models.py
Requires each candidate model already pulled: `ollama pull <model>`
"""
from src.ingestion.docling_loader import load_pdf
from src.extraction.clause_extractor import extract_clauses
from config.settings import Settings
from config.logging_config import setup_logging
import logging
import time
import json
import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

setup_logging("benchmark_models", level=logging.WARNING)

PDF_PATH = os.path.join(Settings.RBI_PDF_DIR, "RBI_Credit_Debit_Card.pdf")

# Candidates to compare — edit this list based on what you've pulled
CANDIDATE_MODELS = [
    "llama3.2:3b",   # current baseline — known too slow, included for reference
    "llama3.2:1b",
    "qwen2.5:1.5b",
]

CHAPTER_TITLES_TO_TEST = {
    "long": None,   # filled in after load — will match "Chapter II"
    "short": None,  # will match "Chapter III"
}


def run_benchmark():
    print(f"Loading and parsing {PDF_PATH} via Docling (one-time cost, not counted per-model)...")
    chapters = load_pdf(PDF_PATH)

    long_chapter = next((c for c in chapters if "Chapter II" in c["chapter_title"]), None)
    short_chapter = next((c for c in chapters if "Chapter III" in c["chapter_title"]), None)

    if not long_chapter or not short_chapter:
        raise RuntimeError("Could not find Chapter II / Chapter III in parsed chapters — check titles.")

    test_chapters = {"long (Chapter II)": long_chapter, "short (Chapter III)": short_chapter}

    results = []

    for model in CANDIDATE_MODELS:
        for label, chapter in test_chapters.items():
            print(f"\n{'='*70}\nModel: {model} | Chapter: {label}\n{'='*70}")
            start = time.perf_counter()
            try:
                clauses = extract_clauses(chapter["chapter_text"], model=model)
                elapsed = time.perf_counter() - start
                status = "OK"
            except Exception as e:
                elapsed = time.perf_counter() - start
                clauses = []
                status = f"FAILED: {e}"

            print(f"  Time: {elapsed:.1f}s | Status: {status} | Clauses extracted: {len(clauses)}")
            for c in clauses[:3]:  # print first 3 for a quick quality glance
                print(f"    [{c['risk_level']}] {c['clause_num']}: {c['text'][:100]}...")

            results.append({
                "model": model,
                "chapter": label,
                "elapsed_seconds": round(elapsed, 1),
                "status": status,
                "clause_count": len(clauses),
                "clauses": clauses,
            })

    # Save full output for detailed comparison later
    out_path = "benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out_path}")

    # Summary table
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"{'Model':<20}{'Chapter':<20}{'Time(s)':<10}{'Clauses':<10}{'Status'}")
    for r in results:
        print(f"{r['model']:<20}{r['chapter']:<20}{r['elapsed_seconds']:<10}{r['clause_count']:<10}{r['status']}")


if __name__ == "__main__":
    run_benchmark()