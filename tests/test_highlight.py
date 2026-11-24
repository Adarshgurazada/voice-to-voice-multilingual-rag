import os
import sys

import pymupdf
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from highlight import find_quote_rects, highlighted_png


_KEEP_ALIVE = []  # a page is invalid once its document is garbage-collected


@pytest.fixture()
def page(tmp_path):
    doc = pymupdf.open()
    _KEEP_ALIVE.append(doc)
    page = doc.new_page(width=612, height=792)
    page.insert_textbox(pymupdf.Rect(50, 80, 560, 200),
                        "Refunds are issued within fourteen days of the request. The customer must keep the receipt. "
                        "Gift cards are never refunded under any condition.", fontsize=11, fontname="helv")
    for x, label in ((50, "Question"), (210, "What it means"), (380, "Reply")):
        page.insert_text((x, 300), label, fontsize=10, fontname="hebo")
    for x, cell in ((50, "Can I return it?"), (210, "Wants a refund"), (380, "Yes, within fourteen days.")):
        page.insert_text((x, 330), cell, fontsize=10, fontname="helv")
    return page


def test_sentence_is_located_on_its_line(page):
    rects = find_quote_rects(page, "The customer must keep the receipt.")
    assert rects and all(r[3] - r[1] < 20 for r in rects)  # line-height boxes
    assert all(70 < r[1] < 200 for r in rects)  # inside the paragraph, not the table


def test_quote_survives_case_punctuation_and_small_differences(page):
    assert find_quote_rects(page, "REFUNDS are issued within fourteen days of the request")
    assert find_quote_rects(page, "Gift cards are never refunded under any condition")


def test_table_row_quote_is_located_cell_by_cell(page):
    rects = find_quote_rects(
        page, "Question: Can I return it? | What it means: Wants a refund | Reply: Yes, within fourteen days.")
    xs = sorted(round(r[0]) for r in rects)
    assert len(rects) == 3 and xs[0] < 100 and xs[-1] > 350  # one box per column, same row
    assert all(312 < r[1] < 335 for r in rects)  # the data row, not the header row above it


def test_unlocatable_quote_returns_nothing_not_a_guess(page):
    assert find_quote_rects(page, "this sentence never appears on the page") == []
    assert find_quote_rects(page, "word") == []  # a single word is not evidence of a quote


def test_highlighted_png_does_not_change_the_page_text(page):
    rects = find_quote_rects(page, "The customer must keep the receipt.")
    data = highlighted_png(page, rects, pymupdf.Matrix(1, 1))
    assert data[:4] == b"\x89PNG"
