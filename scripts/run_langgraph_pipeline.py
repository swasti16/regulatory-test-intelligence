"""
Runs the LangGraph-orchestrated extraction pipeline.

ORCHESTRATOR (default): loops all PDFs in Settings.RBI_PDF_DIR, spawns one
child subprocess per PDF — mirrors extract_to_json.py's per-PDF subprocess
isolation. Docling model memory must be released back to the OS between
PDFs; gc.collect() alone is insufficient (confirmed by RBI_Managing_Risks.pdf
stalling/OOM-ing when run in-process after prior PDFs in this session).
WORKER (--worker <pdf_path>): runs the pipeline for exactly ONE PDF in the
current (freshly-spawned) process.

Run:
    python scripts/run_langgraph_pipeline.py                                   # all PDFs, isolated
    python scripts/run_langgraph_pipeline.py data/RBI_regulations/RBI_KYC.pdf   # single PDF, isolated
"""
import glob
import os
import subprocess
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

_SUBPROCESS_TIMEOUT_SEC = int(os.environ.get("EXTRACT_TIMEOUT_SEC", 14400))


def _run_worker(pdf_path: str) -> int:
    """Runs the LangGraph pipeline for exactly ONE pdf. Only called inside
    a freshly-spawned subprocess — calling directly from the orchestrator's
    own process defeats isolation."""
    from config.logging_config import setup_logging
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
    setup_logging(f"langgraph_pipeline_{doc_id}")

    # Imported here, not at module top — keeps the ORCHESTRATOR process's
    # own footprint free of torch/docling/langgraph; only the worker (which
    # exits and releases everything) ever loads them.
    from src.orchestration.graph import build_graph
    import logging
    logger = logging.getLogger(__name__)

    app = build_graph()
    try:
        final_state = app.invoke({
            "pdf_path": pdf_path,
            "doc_id": "",
            "sections": [],
            "current_section_idx": 0,
            "current_section_retried": False,
            "all_clauses": [],
            "failed_sections": [],
            "current_section_included_count": 0,
            "_chapters": [],
            "current_section_extraction_attempted": False,
            "validation_issues": [],
        })
    except Exception:
        logger.exception(f"[Worker] UNRECOVERABLE — {pdf_path} failed")
        return 1

    print(f"\nDone. doc_id={final_state['doc_id']}")
    print(f"Total clauses: {len(final_state['all_clauses'])}")
    print(f"Failed sections: {final_state['failed_sections']}")
    print(f"Validation issues: {len(final_state['validation_issues'])}")
    return 0


def _run_orchestrator(pdf_paths: list) -> None:
    print(f"Found {len(pdf_paths)} PDF(s) to process (each in an isolated subprocess)")
    results = []

    for pdf_path in pdf_paths:
        doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"\n{'='*70}\nSpawning isolated subprocess for: {doc_id}\n{'='*70}")

        try:
            proc = subprocess.run(
                [sys.executable, __file__, pdf_path, "--worker"],
                timeout=_SUBPROCESS_TIMEOUT_SEC,
            )
            status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        except subprocess.TimeoutExpired:
            status = f"TIMEOUT (>{_SUBPROCESS_TIMEOUT_SEC}s)"
        results.append((doc_id, status))

    print(f"\n{'='*70}\nBATCH SUMMARY\n{'='*70}")
    for doc_id, status in results:
        print(f"  {doc_id:<30} {status}")


def main():
    args = sys.argv[1:]
    worker_mode = "--worker" in args
    args = [a for a in args if a != "--worker"]

    if worker_mode:
        if not args:
            print("Worker mode requires exactly one PDF path.")
            sys.exit(1)
        sys.exit(_run_worker(args[0]))

    from config.settings import Settings
    pdf_paths = [args[0]] if args else sorted(glob.glob(os.path.join(Settings.RBI_PDF_DIR, "*.pdf")))

    if not pdf_paths:
        print("No PDFs found to process.")
        return

    _run_orchestrator(pdf_paths)


if __name__ == "__main__":
    main()
