"""
Tests for src.ingestion.docling_loader.

Fast tests (default): pure functions on hand-crafted text/stub items —
no Docling model loading, milliseconds each.
Slow tests (@pytest.mark.slow): run real Docling against sample PDFs,
~15-20s model load. Run with: pytest -m "not slow" for fast-only.
"""
from types import SimpleNamespace
import pytest
from config.settings import Settings


from src.ingestion.docling_loader import (
    load_pdf,
    _split_into_chapters,
    _build_chunk,
    _reconstruct_with_page_markers,
)


# ======== Stub Docling item ==================================
# Mirrors ONLY the attributes our loader code actually touches:
# item.label.value, item.text, item.prov[0].page_no, item.prov[0].bbox.t
# (verified against real docling_core.types.doc TextItem/ProvenanceItem/
# BoundingBox — see handoff notes / session log for the verification).
# Using a plain stub (not unittest.mock.Mock) so a typo'd attribute access
# raises AttributeError instead of silently returning a new Mock.
class FakeItem:
    def __init__(self, text, page_no, top, label="paragraph"):
        self.text = text
        self.label = SimpleNamespace(value=label)
        self.prov = [SimpleNamespace(page_no=page_no, bbox=SimpleNamespace(t=top))]


# ======== Fast Unit Tests — Chapter Splitting ==================================

class TestChapterTitleDetection:
    def test_happy_path_roman_numeral(self):
        text = "## Chapter I - Preliminary\n\nSome intro text."
        chunks = _split_into_chapters(text, doc_id="doc1")
        assert len(chunks) == 1
        assert chunks[0]["chapter_title"] == "Chapter I - Preliminary"

    def test_happy_path_arabic_numeral(self):
        text = "## Chapter 2 - Definitions\n\nBody text."
        chunks = _split_into_chapters(text, doc_id="doc1")
        assert chunks[0]["chapter_title"] == "Chapter 2 - Definitions"

    def test_extra_spaces_in_heading(self):
        text = "##   Chapter   III  -  Scope \n\nBody."
        chunks = _split_into_chapters(text, doc_id="doc1")
        assert chunks[0]["chapter_title"].startswith("Chapter")

    def test_case_insensitive_chapter_keyword(self):
        text = "## chapter IV - lowercase heading\n\nBody."
        chunks = _split_into_chapters(text, doc_id="doc1")
        assert len(chunks) == 1


# ======== Fast Unit Tests — Page Metadata ==================================

class TestPageMetadata:
    def test_single_page_multiple_chapters(self):
        text = (
            "[p.1]\n## Chapter I - A\nText A.\n\n"
            "## Chapter II - B\nText B."
        )
        chunks = _split_into_chapters(text, doc_id="doc1")
        assert len(chunks) == 2
        assert chunks[0]["page_start"] == 1 and chunks[0]["page_end"] == 1
        assert chunks[1]["page_start"] == 1 and chunks[1]["page_end"] == 1

    def test_chapter_spanning_multiple_pages(self):
        text = (
            "[p.1]\n## Chapter I - Long\nStart.\n[p.2]\nMiddle.\n[p.3]\nEnd."
        )
        chunks = _split_into_chapters(text, doc_id="doc1")
        assert chunks[0]["page_start"] == 1
        assert chunks[0]["page_end"] == 3

    def test_chapter_starting_mid_page_regression_bug1(self):
        """
        Regression: chapter heading appears mid-page (no [p.N] marker
        immediately preceding it) — must inherit running_page from the
        last marker BEFORE the heading, not default to None.
        """
        text = (
            "[p.5]\nSome preamble text on page 5.\n"
            "## Chapter II - Mid Page Start\nBody continues on page 5."
        )
        chunks = _split_into_chapters(text, doc_id="doc1")
        assert chunks[0]["page_start"] == 5
        assert chunks[0]["page_end"] == 5

    def test_chapter_completing_in_one_page(self):
        text = "[p.7]\n## Chapter I - Short\nAll on one page."
        chunks = _split_into_chapters(text, doc_id="doc1")
        assert chunks[0]["page_start"] == 7 == chunks[0]["page_end"]


class TestNoPageMarkers:
    def test_no_page_numbers_present(self):
        text = "## Chapter I - No Markers\nJust text, no [p.N] anywhere."
        chunks = _split_into_chapters(text, doc_id="doc1", default_page=None)
        assert chunks[0]["page_start"] is None
        assert chunks[0]["page_end"] is None


class TestFileNotFound:
    def test_missing_pdf_raises(self):
        with pytest.raises(FileNotFoundError):
            load_pdf("nonexistent_file_xyz.pdf")


class TestChunkCountMatchesHeadings:
    def test_three_headings_three_chunks(self):
        text = (
            "## Chapter I - A\nText.\n"
            "## Chapter II - B\nText.\n"
            "## Chapter III - C\nText."
        )
        chunks = _split_into_chapters(text, doc_id="doc1")
        assert len(chunks) == 3

    def test_no_headings_returns_single_fallback_chunk(self):
        text = "No chapter headings here at all."
        chunks = _split_into_chapters(text, doc_id="doc1")
        assert len(chunks) == 1
        assert "no chapter headings detected" in chunks[0]["chapter_title"].lower()


class TestDocIdDisambiguation:
    def test_same_chapter_title_different_doc_ids(self):
        """
        Two different documents both have 'Chapter I - Preliminary'.
        doc_id must disambiguate — ties to composite Neo4j clause ID
        design ({doc_id}_{chapter}_{clause_num}).
        """
        text = "## Chapter I - Preliminary\nSame title, different doc."
        chunks_a = _split_into_chapters(text, doc_id="regulation_155MD")
        chunks_b = _split_into_chapters(text, doc_id="regulation_999XY")

        assert chunks_a[0]["chapter_title"] == chunks_b[0]["chapter_title"]
        assert chunks_a[0]["doc_id"] != chunks_b[0]["doc_id"]
        assert chunks_a[0]["doc_id"] == "regulation_155MD"
        assert chunks_b[0]["doc_id"] == "regulation_999XY"


# ======== Fast Unit Test — Bbox-Sort Fix (Bug #2 regression) ==================

class TestBboxSortOrdering:
    """
    Regression for bug #2: Docling's iterate_items() traversal order is
    unreliable within a single page. A section header can be emitted
    AFTER its own subsections even though its bbox.t (top coordinate,
    origin bottom-left -> higher t = higher on page) is spatially above
    them. Fix: buffer items per page, sort by bbox.t descending before
    rendering, instead of trusting iterate_items() order.
    """

    def test_out_of_order_items_are_sorted_by_bbox_top(self):
        # Simulates iterate_items() yielding subsection BEFORE its parent
        # heading (the exact real-world 155MD.pdf scenario), despite the
        # heading having a higher bbox.t (higher on the page).
        subsection = FakeItem(
            text="A. Eligibility", page_no=2, top=300.0, label="section_header"
        )
        heading = FakeItem(
            text="Chapter II - Conduct of Credit Card Business",
            page_no=2, top=700.0, label="section_header"
        )

        class FakeDoc:
            def iterate_items(self):
                # Deliberately out of visual order — subsection first
                yield (subsection, 0)
                yield (heading, 0)

        full_text, first_page = _reconstruct_with_page_markers(FakeDoc())

        heading_idx = full_text.index("Chapter II")
        subsection_idx = full_text.index("A. Eligibility")
        assert heading_idx < subsection_idx, (
            "Heading must appear before its subsection after bbox-sort fix"
        )
        assert first_page == 2

    def test_items_on_different_pages_get_page_markers(self):
        item_p1 = FakeItem(text="Page one content", page_no=1, top=100.0)
        item_p2 = FakeItem(text="Page two content", page_no=2, top=100.0)

        class FakeDoc:
            def iterate_items(self):
                yield (item_p1, 0)
                yield (item_p2, 0)

        full_text, first_page = _reconstruct_with_page_markers(FakeDoc())
        assert "[p.1]" in full_text
        assert "[p.2]" in full_text
        assert full_text.index("[p.1]") < full_text.index("Page one content")
        assert full_text.index("[p.2]") < full_text.index("Page two content")
        assert first_page == 1


# ======== Slow Integration Tests — Real Docling + Real PDFs ==================

REAL_RBI_PDF = f"{Settings.RBI_PDF_DIR}/RBI_Credit_Debit_Card.pdf"


@pytest.mark.slow
class TestRealPdfIntegration:
    def test_table_renders_correctly(self):
        chunks = load_pdf(REAL_RBI_PDF)
        full_text = "\n".join(c["chapter_text"] for c in chunks)
        assert "|" in full_text  # markdown table pipe syntax present

    def test_multipage_table_renders_correctly(self):
        # 155MD.pdf confirmed to contain a genuine multi-page table —
        # reuse it directly (no separate fixture needed, see handoff notes).
        chunks = load_pdf(REAL_RBI_PDF)
        assert len(chunks) > 0
        assert any(c["page_start"] != c["page_end"] for c in chunks), (
            "Expected at least one chapter chunk to span multiple pages"
        )

    def test_reading_order_regression_chapter_ii(self):
        """
        Full end-to-end reading-order check on the real RBI PDF:
        Chapter II heading must precede its known subsection
        'A. Eligibility' in the reconstructed/chunked text — the exact
        bug #2 scenario, validated against the real document this time.
        """
        chunks = load_pdf(REAL_RBI_PDF)
        chapter_ii = next(
            (c for c in chunks if "Chapter II" in c["chapter_title"]), None
        )
        assert chapter_ii is not None, "Chapter II not found in parsed chunks"
        assert "A. Eligibility" in chapter_ii["chapter_text"]
        assert (
            chapter_ii["chapter_text"].index("Chapter II")
            < chapter_ii["chapter_text"].index("A. Eligibility")
        )
