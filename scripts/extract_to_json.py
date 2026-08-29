"""
Extract clauses from a PDF (or all PDFs in Settings.RBI_PDF_DIR) and dump
them to data/extracted_clauses/{doc_id}.json — NO Neo4j writes.

PER-PDF SUBPROCESS ISOLATION:
Each PDF runs in its own fresh `python` subprocess so torch/docling model
memory is guaranteed released back to the OS on subprocess exit — root
cause of the earlier silent OOM kill was relying on gc.collect() inside a
single long-lived interpreter across multiple Docling model reloads.

Two modes in this one file:
  ORCHESTRATOR (default): loops PDFs, spawns one child subprocess per PDF,
  inspects each child's exit code, logs+skips on crash/timeout, continues.
  WORKER (--worker flag, used internally): runs extraction for exactly ONE
  PDF in the current (freshly-spawned) process, with its own isolated log
  file and a background peak-RSS sampler (see _PeakMemoryMonitor).

Run:
    python scripts/extract_to_json.py                    # all PDFs, isolated
    python scripts/extract_to_json.py path/to/one.pdf     # single PDF, isolated
"""
import json
import logging
import os
import sys
import glob
import subprocess
import threading
from datetime import datetime
from collections import Counter

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

logger = logging.getLogger(__name__)

OUTPUT_DIR = "data/extracted_clauses"
_SUBPROCESS_TIMEOUT_SEC = int(os.environ.get("EXTRACT_TIMEOUT_SEC", 14400))


class _PeakMemoryMonitor:
    """
    Background-thread RSS sampler. A single memory_info().rss call is a
    snapshot, not a peak — Docling's layout model load is a short, sharp
    spike between samples that point-in-time checks miss entirely. This
    polls every 0.5s and tracks the high-water mark for the life of the
    `with` block.
    """
    def __init__(self, interval_sec: float = 0.5):
        self._interval = interval_sec
        self._peak_bytes = 0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self):
        import psutil
        proc = psutil.Process(os.getpid())
        while not self._stop_event.is_set():
            try:
                rss = proc.memory_info().rss
                if rss > self._peak_bytes:
                    self._peak_bytes = rss
            except Exception:
                pass
            self._stop_event.wait(self._interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop_event.set()
        self._thread.join(timeout=2)

    @property
    def peak_gb(self) -> float:
        return self._peak_bytes / (1024 ** 3)


def _derive_doc_id(pdf_path: str) -> str:
    return os.path.splitext(os.path.basename(pdf_path))[0]


def _run_worker(pdf_path: str) -> int:
    """
    Runs extraction for exactly ONE pdf. Only ever called inside a
    freshly-spawned subprocess — calling this directly from the
    orchestrator's own process defeats isolation.
    Returns 0 on success, 1 on failure (orchestrator reads via
    subprocess.run(...).returncode).
    """
    from config.logging_config import setup_logging
    doc_id = _derive_doc_id(pdf_path)
    log_path = setup_logging(f"extract_to_json_{doc_id}")

    # Imported here, not at module top — keeps the ORCHESTRATOR process's
    # own footprint free of torch/docling; only the worker (which exits and
    # releases everything on completion) ever loads them.
    from src.ingestion.docling_loader import load_pdf, split_chapter_into_sections, split_definitions_section
    from src.extraction.clause_extractor import extract_clauses, _last_call_metadata, attach_section_metadata
    from config.settings import Settings

    logger.info(f"[Worker] Starting isolated extraction for {pdf_path}")

    with _PeakMemoryMonitor() as mem:
        try:
            chapters = load_pdf(pdf_path)
            doc_id = chapters[0]["doc_id"]
            logger.info(f"Processing {pdf_path} -> doc_id={doc_id} ({len(chapters)} chapters)")

            all_clauses = []
            failed_sections = []

            for chapter in chapters:
                if "Definitions" in chapter["chapter_title"]:
                    sections = split_definitions_section(
                        chapter["chapter_text"], chapter["chapter_title"], doc_id, chapter["page_start"]
                    )
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
                                f"({len(clauses)} attempted, all dropped) — flagging for manual re-run."
                            )
                            failed_sections.append(f"{section['chapter_title']} (0 survived — needs retry)")
                    except Exception as e:
                        logger.error(f"Skipping section '{section['chapter_title']}' — extraction failed: {e}")
                        failed_sections.append(section["chapter_title"])
                        continue

                    call_meta = dict(_last_call_metadata)
                    if call_meta.get("done_reason") == "length":
                        logger.error(f"TRUNCATED OUTPUT for '{section['chapter_title']}' — results may be incomplete")

                    clauses = attach_section_metadata(
                        clauses, section["chapter_title"], section["page_start"], section["page_end"]
                    )
                    all_clauses.extend(clauses)

            key_counts = Counter((c["chapter_title"], c["clause_num"]) for c in all_clauses)
            seen = Counter()
            for c in all_clauses:
                key = (c["chapter_title"], c["clause_num"])
                if key_counts[key] > 1:
                    seen[key] += 1
                    c["clause_num"] = f"{c['clause_num']}#{seen[key]}"

            def _count(status):
                return sum(1 for c in all_clauses if c["status"] == status)

            result = {
                "doc_id": doc_id,
                "source_pdf": pdf_path,
                "extracted_at": datetime.now().isoformat(),
                "model": Settings.OLLAMA_MODEL,
                "summary": {
                    "total_extracted": len(all_clauses),
                    "included": _count("included"),
                    "dropped_ungrounded": _count("dropped_ungrounded"),
                    "dropped_illustrative": _count("dropped_illustrative"),
                    "dropped_invalid_risk": _count("dropped_invalid_risk"),
                    "failed_sections": failed_sections,
                },
                "clauses": all_clauses,
            }

            os.makedirs(OUTPUT_DIR, exist_ok=True)
            out_path = os.path.join(OUTPUT_DIR, f"{doc_id}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            logger.info(f"[Worker] Saved: {out_path}")
            logger.info(f"[Worker][MEM] Peak RSS this process: {mem.peak_gb:.3f} GB")
            logger.info(f"{doc_id}: {result['summary']['included']} included, "
                  f"{result['summary']['dropped_ungrounded']} dropped(ungrounded), "
                  f"{result['summary']['dropped_illustrative']} dropped(illustrative) "
                  f"{result['summary']['dropped_invalid_risk']} dropped(invalid_risk) "
                  f"| peak RSS {mem.peak_gb:.3f} GB -> {out_path}")
            return 0

        except Exception:
            logger.exception(f"[Worker] UNRECOVERABLE — {pdf_path} failed")
            logger.info(f"[Worker][MEM] Peak RSS before failure: {mem.peak_gb:.3f} GB")
            return 1


def _run_orchestrator(pdf_paths: list) -> None:
    logger.info(f"Found {len(pdf_paths)} PDF(s) to process (each in an isolated subprocess)")
    results = []

    for pdf_path in pdf_paths:
        doc_id = _derive_doc_id(pdf_path)
        logger.info(f"\n{'='*70}\nSpawning isolated subprocess for: {doc_id}\n{'='*70}")

        try:
            proc = subprocess.run(
                [sys.executable, __file__, pdf_path, "--worker"],
                timeout=_SUBPROCESS_TIMEOUT_SEC,
            )
            if proc.returncode != 0:
                logger.info(f"  [FAILED] {doc_id} — subprocess exit code {proc.returncode} "
                      f"(often an OS-level kill, e.g. OOM). "
                      f"Check logs/extract_to_json_{doc_id}_*.log.")
                results.append((doc_id, "FAILED", proc.returncode))
            else:
                results.append((doc_id, "OK", 0))
        except subprocess.TimeoutExpired:
            log_glob = sorted(glob.glob(f"logs/extract_to_json_{doc_id}_*.log"))
            stall_note = ""
            if log_glob:
                last_log = log_glob[-1]
                age_sec = datetime.now().timestamp() - os.path.getmtime(last_log)
                stall_note = f" | log last written {age_sec:.0f}s ago ({'likely genuinely stuck' if age_sec > 300 else 'was still actively progressing'})"
            logger.info(f"  [TIMEOUT] {doc_id} — exceeded {_SUBPROCESS_TIMEOUT_SEC}s, killed.{stall_note}")
            results.append((doc_id, "TIMEOUT", None))

    logger.info(f"\n{'='*70}\nBATCH SUMMARY\n{'='*70}")
    for doc_id, status, code in results:
        suffix = f" (exit code {code})" if status == "FAILED" else ""
        logger.info(f"  {doc_id:<30} {status}{suffix}")


def main():
    args = sys.argv[1:]
    worker_mode = "--worker" in args
    args = [a for a in args if a != "--worker"]

    if worker_mode:
        if not args:
            logger.info("Worker mode requires exactly one PDF path.")
            sys.exit(1)
        sys.exit(_run_worker(args[0]))

    from config.settings import Settings
    pdf_paths = [args[0]] if args else sorted(glob.glob(os.path.join(Settings.RBI_PDF_DIR, "*.pdf")))

    if not pdf_paths:
        logger.info("No PDFs found to process.")
        return

    _run_orchestrator(pdf_paths)


if __name__ == "__main__":
    main()
