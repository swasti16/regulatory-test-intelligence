"""
Docling PDF Loader — Single-File Regulatory Document Parser.

This module converts a single regulatory PDF into a structured list of 
chapter-level chunks with inline page-marker provenance and deterministic 
page-range citations.

==============================================================================
DESIGN PRINCIPLES & ARCHITECTURAL DECISIONS
==============================================================================

1. Single-File Responsibility:
   - Operates strictly on ONE input file path per invocation.
   - Keeps the loader isolated, pure, and easily unit-testable.
   - Treats document discovery (e.g., globbing `data/sample_regulations/`)
     and multi-process/async fan-out as orchestration concerns.

2. Chunking Strategy (Chapter-Level Granularity):
   - Chunks are partitioned along regulatory Chapter boundaries 
     (e.g., `## Chapter I - ...`, `## Chapter 2 - ...`).
   - Why Chapter-level over Clause-level in the Loader:
     * Maintains full semantic and co-reference context across related clauses.
     * Leaves fine-grained extraction, validation, and schema mapping to the 
       downstream LLM extraction step.
   - Any document preamble (title block, gazette notifications, enactment 
     dates before Chapter I) is prepended to the first chapter rather than 
     dropped, preserving legal context.

3. Citation & Provenance Design (Inline Markers + Metadata):
   - Problem with raw markdown export (`export_to_markdown`):
     Docling's default markdown export strips page provenance, making 
     per-clause source verification impossible.
   - Solution (Tree Walk + Inline Marker Injection):
     * Traverses Docling's structured document tree (`iterate_items()`).
     * Extracts physical page origins from each node's provenance (`item.prov`).
     * Injects inline `[p.N]` markers into the reconstructed text stream at 
       every page boundary transition.
   - Downstream Auditability:
     * `page_start` and `page_end` in the chunk dictionary provide instant 
       metadata filtering.
     * The `[p.N]` markers persist directly in `chapter_text`, enabling the 
       downstream LLM to attach verifiable page-level citations to individual 
       clauses, obligations, and penalties.

4. Page Boundary State Tracking:
   - Chapters starting mid-page look back to the preceding text stream to 
     carry forward the active `running_page`.
   - Single-page chapters without an internal boundary crossing correctly 
     inherit the active page rather than evaluating to `None`.
==============================================================================
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)

# Matches chapter headings rendered as "## Chapter X - Title"
# Handles Roman numerals (I, IV, X) and Arabic numerals (1, 2, 3)
_CHAPTER_HEADING_MD_RE = re.compile(
    r"^##\s*(Chapter\s+[IVXLC\d]+.*)$", 
    re.MULTILINE | re.IGNORECASE
)

# Regex pattern used to extract injected inline page markers, e.g., "[p.3]"
_PAGE_MARKER_RE = re.compile(r"\[p\.(\d+)\]")


def load_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Parse a single regulatory PDF into chapter-level structured chunks with page provenance.

    Args:
        pdf_path: Absolute or relative file path to the PDF document.

    Returns:
        A list of dictionaries, where each dict represents a discrete chapter chunk:
        [
            {
                "doc_id": str,          # filename without extension
                "chapter_title": str,   # e.g. "Chapter I - Preliminary"
                "chapter_text": str,    # chapter text WITH inline [p.N]
                                        # page markers — extraction stage
                                        # uses these for per-clause citation
                "page_start": int | None,
                "page_end": int | None,
            },
            ...
        ]

    Raises:
        FileNotFoundError: If the provided `pdf_path` does not exist on disk.
        docling.exceptions.ConversionError: If Docling fails to parse or render the PDF.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"Target PDF not found: {pdf_path}")

    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]

    # Convert document via Docling pipeline (OCR, layout parsing, reading order)
    converter = DocumentConverter()
    result = converter.convert(pdf_path)

    # Reconstruct text with inline page markers
    full_text, doc_first_page = _reconstruct_with_page_markers(result.document)

    # Split text into chapter-level chunks with robust page attribution
    return _split_into_chapters(full_text, doc_id, default_page=doc_first_page)


def _get_item_pages(item: Any) -> List[int]:
    """
    Extract unique 1-based page numbers from a Docling document item's provenance.

    Args:
        item: A Docling structural node (e.g., TextItem, SectionHeaderItem, TableItem).

    Returns:
        List of distinct integer page numbers where this item physically appears.
    """
    pages: List[int] = []
    prov_list = getattr(item, "prov", None)
    if prov_list:
        for prov in prov_list:
            page_no = getattr(prov, "page_no", None)
            if page_no is not None and page_no not in pages:
                pages.append(page_no)
    return pages


def _render_item(item: Any, document: Any) -> Optional[str]:
    """
    Render an individual Docling document item into Markdown format.

    - Table items are serialized to GitHub-Flavored Markdown pipe tables.
    - Section headers and titles are normalized with markdown heading prefixes (`## `)
      to serve as split anchors.
    - Paragraphs and general items are emitted as clean, stripped text.

    Args:
        item: The current Docling item being processed.
        document: The root DoclingDocument instance (required for table markdown export).

    Returns:
        Rendered string representation, or None if the item contains no displayable text.
    """
    label = getattr(item, "label", None)
    label_value = getattr(label, "value", str(label)).lower()

    # Tables: use Docling's native markdown table exporter
    if "table" in label_value:
        try:
            return item.export_to_markdown(document)
        except Exception as exc:
            logger.warning("Table export_to_markdown failed (%s); falling back to plain text.", exc)
            return getattr(item, "text", None)

    text = getattr(item, "text", None)
    if not text or not text.strip():
        return None

    # Normalize section headers and titles for deterministic regex matching
    if "section_header" in label_value or "title" in label_value:
        return f"## {text.strip()}"

    return text.strip()


def _reconstruct_with_page_markers(document: Any) -> Tuple[str, Optional[int]]:
    """
    Traverse the Docling document tree and synthesize a linear Markdown stream
    annotated with inline `[p.N]` page markers at page transitions.

    Args:
        document: The parsed DoclingDocument object.

    Returns:
        A tuple containing:
            - Reconstructed full document text with inline `[p.N]` markers.
            - The first detected page number in the document (usually 1).
    """
    lines: List[str] = []
    first_detected_page: Optional[int] = None

    # Buffer (item, top) per page; flush+sort when page changes.
    page_buffer: List[Tuple[Any, float]] = []
    buffer_page_no: Optional[int] = None

    def flush_buffer():
        """Sort buffered items by vertical position (top, descending) and render."""
        nonlocal lines
        page_buffer.sort(key=lambda pair: pair[1], reverse=True)
        for buffered_item, _top in page_buffer:
            rendered = _render_item(buffered_item, document)
            if rendered:
                lines.append(rendered)

    for item, _level in document.iterate_items():
        pages = _get_item_pages(item)
        if not pages:
            continue  # items without page provenance can't be spatially sorted — skip from buffer path

        page_no = pages[0]
        if first_detected_page is None:
            first_detected_page = page_no

        if page_no != buffer_page_no:
            if page_buffer:
                flush_buffer()
                page_buffer.clear()
            lines.append(f"[p.{page_no}]")
            buffer_page_no = page_no

        prov = getattr(item, "prov", None)
        top = prov[0].bbox.t if prov else 0.0
        page_buffer.append((item, top))

    if page_buffer:
        flush_buffer()
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(lines))
    return "\n\n".join(lines), first_detected_page


def _split_into_chapters(
    full_text: str, 
    doc_id: str, 
    default_page: Optional[int] = 1
) -> List[Dict[str, Any]]:
    """
    Partition reconstructed document text into structured chapter-level records.

    Ensures that chapters starting mid-page correctly retain their start page
    by inspecting preceding text markers, preventing orphaned `page_start: None`.

    Args:
        full_text: Full synthesized document text containing `[p.N]` markers.
        doc_id: Identifier derived from the source filename.
        default_page: Fallback page number if no markers are detected.

    Returns:
        List of chapter dictionaries with complete citation and text payload.
    """
    matches = list(_CHAPTER_HEADING_MD_RE.finditer(full_text))

    # Fallback if no chapter headers match the regex
    if not matches:
        return [_build_chunk(doc_id, "Full Document (no chapter headings detected)", full_text, fallback_page=default_page)]

    chunks: List[Dict[str, Any]] = []
    running_page = default_page

    for i, match in enumerate(matches):
        chapter_title = match.group(1).strip()
        start_idx = match.start()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)

        # Attach preamble/metadata before Chapter I to the first chapter chunk
        if i == 0 and start_idx > 0:
            raw_chapter = full_text[:end_idx].strip()
        else:
            raw_chapter = full_text[start_idx:end_idx].strip()

        # Check the last page marker that appeared BEFORE this chapter boundary
        preceding_text = full_text[:start_idx]
        preceding_markers = [int(n) for n in _PAGE_MARKER_RE.findall(preceding_text)]
        if preceding_markers:
            running_page = preceding_markers[-1]

        # Ensure chapter text explicitly carries its active starting page marker
        chapter_markers = [int(n) for n in _PAGE_MARKER_RE.findall(raw_chapter)]
        if not chapter_markers and running_page is not None:
            chapter_text = f"[p.{running_page}]\n{raw_chapter}"
        elif chapter_markers and chapter_markers[0] != running_page and running_page is not None:
            chapter_text = f"[p.{running_page}]\n{raw_chapter}"
        else:
            chapter_text = raw_chapter

        chunk = _build_chunk(doc_id, chapter_title, chapter_text, fallback_page=running_page)
        chunks.append(chunk)

        # Carry the end page forward as the baseline for the next chapter
        if chunk["page_end"] is not None:
            running_page = chunk["page_end"]

    return chunks


def _build_chunk(
    doc_id: str, 
    chapter_title: str, 
    chapter_text: str, 
    fallback_page: Optional[int] = None
) -> Dict[str, Any]:
    """
    Construct a finalized chunk dictionary, calculating min/max page citations.

    Args:
        doc_id: Document identifier string.
        chapter_title: Extracted heading for the chapter.
        chapter_text: Reconstructed text with inline markers.
        fallback_page: Active running page to use if no internal markers exist.

    Returns:
        Structured chunk dictionary.
    """
    page_numbers = [int(n) for n in _PAGE_MARKER_RE.findall(chapter_text)]

    if page_numbers:
        page_start = min(page_numbers)
        page_end = max(page_numbers)
    else:
        page_start = fallback_page
        page_end = fallback_page

    return {
        "doc_id": doc_id,
        "chapter_title": chapter_title,
        "chapter_text": chapter_text,
        "page_start": page_start,
        "page_end": page_end,
    }
