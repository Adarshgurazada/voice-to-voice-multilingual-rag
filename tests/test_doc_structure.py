import os
import sys

import pymupdf
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from doc_structure import DocumentLayout, SectionTracker, chunk_items, Item, structured_page_chunks

BODY = ("Refunds are issued within fourteen days. The customer must keep the receipt. "
        "Digital goods are refunded only if they were never opened. Contact support to start a claim. ")


def _build(path, pages=4):
    doc = pymupdf.open()
    for n in range(1, pages + 1):
        page = doc.new_page(width=612, height=792)
        page.insert_text((50, 24), "ACME Handbook", fontsize=8, fontname="helv")  # running header
        page.insert_text((50, 775), f"Page {n}", fontsize=8, fontname="helv")  # running footer
        if n == 1:
            page.insert_text((50, 70), "SECTION 2", fontsize=9, fontname="hebo")
            page.insert_text((50, 95), "Refund policy", fontsize=22, fontname="hebo")
            page.insert_textbox(pymupdf.Rect(50, 110, 560, 220), BODY * 2, fontsize=10, fontname="helv")
            page.insert_text((50, 250), "Exceptions", fontsize=15, fontname="hebo")
            page.insert_textbox(pymupdf.Rect(50, 262, 560, 360), "Gift cards are never refunded. " * 3,
                                fontsize=10, fontname="helv")
        elif n == 2:
            for x, label in ((50, "Question"), (210, "What it means"), (380, "Reply")):
                page.insert_text((x, 100), label, fontsize=10, fontname="hebo")
            rows = [("Can I return it?", "Wants a refund", "Yes, within fourteen days."),
                    ("Is it opened?", "Digital goods", "Only if never opened."),
                    ("Gift card?", "Special case", "Never refunded.")]
            for i, row in enumerate(rows):
                for x, cell in zip((50, 210, 380), row):
                    page.insert_text((x, 135 + 40 * i), cell, fontsize=10, fontname="helv")
        else:
            page.insert_textbox(pymupdf.Rect(50, 80, 560, 600), BODY * 7, fontsize=10, fontname="helv")
    doc.save(path)
    return pymupdf.open(path)


@pytest.fixture()
def doc(tmp_path):
    return _build(str(tmp_path / "handbook.pdf"))


def test_running_header_and_footer_are_removed(doc):
    layout = DocumentLayout(doc)
    assert "acme handbook" in layout.boilerplate and "page #" in layout.boilerplate
    tracker = SectionTracker()
    chunks = [c for page in doc for c in structured_page_chunks(page, layout, tracker)]
    assert chunks and not any("ACME Handbook" in c["content"] or "Page " in c["content"] for c in chunks)


def test_headings_become_a_section_path_carried_across_pages(doc):
    layout = DocumentLayout(doc)
    tracker = SectionTracker()
    by_page = [structured_page_chunks(page, layout, tracker) for page in doc]
    # The author's own wording is kept ("SECTION 2"), joined to the title under it.
    assert by_page[0][0]["section"] == "SECTION 2 \u00b7 Refund policy"
    assert any(c["section"] == "SECTION 2 \u00b7 Refund policy \u203a Exceptions" for c in by_page[0])
    # Pages 3 and 4 have no heading of their own and stay under the last one.
    assert {c["section"] for c in by_page[2]} == {"SECTION 2 \u00b7 Refund policy \u203a Exceptions"}


def test_unruled_table_becomes_one_chunk_per_row_keyed_by_headings(doc):
    layout = DocumentLayout(doc)
    tracker = SectionTracker()
    chunks = structured_page_chunks(doc[1], layout, tracker)
    rows = [c for c in chunks if c["kind"] == "row"]
    assert len(rows) == 3
    assert rows[0]["content"] == ("Question: Can I return it? | What it means: Wants a refund | "
                                  "Reply: Yes, within fourteen days.")


def test_text_chunks_end_on_sentence_boundaries_and_stay_small(doc):
    layout = DocumentLayout(doc)
    tracker = SectionTracker()
    chunks = [c for c in structured_page_chunks(doc[2], layout, tracker) if c["kind"] == "text"]
    assert len(chunks) >= 2
    assert all(c["content"].rstrip().endswith(".") for c in chunks)
    assert all(len(c["content"]) <= 1100 for c in chunks)


def test_chunker_does_not_split_a_long_sentence_mid_word():
    items = [Item("para", "word " * 500 + "end.")]
    chunks = chunk_items(items, SectionTracker())
    assert all(len(c["content"]) <= 1100 for c in chunks)
    assert "".join(c["content"] for c in chunks).replace(" ", "").startswith("word")


def test_part_headings_sit_above_sections():
    tracker = SectionTracker()
    chunks = chunk_items([Item("part", "Part D · Run the cycle"), Item("heading", "Section 17 · Objections", 1),
                          Item("para", "Answer in three moves.")], tracker)
    assert chunks[0]["section"] == "Part D · Run the cycle › Section 17 · Objections"
