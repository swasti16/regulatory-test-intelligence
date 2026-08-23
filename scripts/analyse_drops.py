"""
Diagnoses why status=="dropped_ungrounded" clauses were dropped, and
quantifies how many would flip to "included" under the length-weighted
grounding fix vs the original count-weighted logic.

Re-derives each dropped clause's source section_text by re-running the
same Docling load + section-split pipeline extract_to_json.py used —
NOT re-calling the LLM, purely re-checking grounding against the
already-extracted clause text.

Run:
    python scripts/analyse_drops.py                      # all docs
    python scripts/analyse_drops.py RBI_Credit_Debit_Card # one doc
"""
import json
import os
import sys
import glob
import re

from src.ingestion.docling_loader import (
    load_pdf, split_chapter_into_sections, split_definitions_section,
)
from src.extraction.clause_extractor import _fragment_grounded

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

INPUT_DIR = "data/extracted_clauses"


def _build_section_lookup(source_pdf: str, doc_id: str) -> dict:
    chapters = load_pdf(source_pdf)
    lookup = {}
    for chapter in chapters:
        if "Definitions" in chapter["chapter_title"]:
            sections = split_definitions_section(
                chapter["chapter_text"], chapter["chapter_title"], doc_id, chapter["page_start"]
            )
        else:
            sections = split_chapter_into_sections(
                chapter["chapter_text"], chapter["chapter_title"], doc_id, chapter["page_start"]
            )
        for s in sections:
            lookup[s["chapter_title"]] = s["section_text"]
    return lookup


def _ratio_unweighted(fragments, source_text) -> float:
    if not fragments:
        return 0.0
    found = sum(1 for f in fragments if _fragment_grounded(f, source_text))
    return found / len(fragments)


def _ratio_weighted(fragments, source_text) -> float:
    if not fragments:
        return 0.0
    weights = [len(f.split()) for f in fragments]
    grounded_weight = sum(w for f, w in zip(fragments, weights) if _fragment_grounded(f, source_text))
    total_weight = sum(weights)
    return grounded_weight / total_weight if total_weight else 0.0


def analyse_file(json_path: str) -> None:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    doc_id = data["doc_id"]
    dropped = [c for c in data["clauses"] if c["status"] == "dropped_ungrounded"]
    if not dropped:
        print(f"{doc_id}: 0 dropped_ungrounded clauses — skipping")
        return

    print(f"\n{'='*70}\n{doc_id} — {len(dropped)} dropped_ungrounded clause(s)\n{'='*70}")
    lookup = _build_section_lookup(data["source_pdf"], doc_id)

    would_flip = []
    still_dropped = []
    no_section_found = []

    for c in dropped:
        source_text = lookup.get(c["chapter_title"])
        if source_text is None:
            no_section_found.append(c)
            continue

        fragments = [f.strip() for f in re.split(r"[;.:\n]", c["text"]) if len(f.strip()) > 15]
        old_ratio = _ratio_unweighted(fragments, source_text)
        new_ratio = _ratio_weighted(fragments, source_text)

        if old_ratio < 0.6 <= new_ratio:
            would_flip.append((c, old_ratio, new_ratio))
        else:
            still_dropped.append((c, old_ratio, new_ratio))

    print(f"Would flip to INCLUDED under weighted fix: {len(would_flip)}")
    for c, old, new in would_flip:
        print(f"  [{c['clause_num']}] old_ratio={old:.2f} new_ratio={new:.2f}: {c['text'][:100]}...")

    print(f"\nStill correctly dropped (genuine noise): {len(still_dropped)}")
    for c, old, new in still_dropped[:5]:  # first 5 only, avoid noise
        print(f"  [{c['clause_num']}] old_ratio={old:.2f} new_ratio={new:.2f}: {c['text'][:80]}...")
    if len(still_dropped) > 5:
        print(f"  ... and {len(still_dropped) - 5} more")

    if no_section_found:
        print(f"\nWARNING: {len(no_section_found)} clause(s) — section text not found "
              f"(chapter_title mismatch, needs manual check)")
        for c in no_section_found:
            print(f"  [{c['clause_num']}] chapter_title={c['chapter_title']!r}")


def main():
    if len(sys.argv) > 1:
        json_paths = [os.path.join(INPUT_DIR, f"{sys.argv[1]}.json")]
    else:
        json_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))

    for path in json_paths:
        try:
            analyse_file(path)
        except Exception as e:
            print(f"ERROR analysing {path}: {e}")
            continue


if __name__ == "__main__":
    main()
