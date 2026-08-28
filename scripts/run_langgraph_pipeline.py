"""
Runs the LangGraph-orchestrated extraction pipeline on a single PDF.

Run:
    python scripts/run_langgraph_pipeline.py data/RBI_regulations/RBI_KYC.pdf
"""
import os
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from config.logging_config import setup_logging
from src.orchestration.graph import build_graph


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_langgraph_pipeline.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    setup_logging("langgraph_pipeline")

    app = build_graph()
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
    })

    print(f"\nDone. doc_id={final_state['doc_id']}")
    print(f"Total clauses: {len(final_state['all_clauses'])}")
    print(f"Failed sections: {final_state['failed_sections']}")


if __name__ == "__main__":
    main()
