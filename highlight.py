"""Find a verified quote on its PDF page, so the page image can show exactly where an answer comes from.

The answer pipeline already checks that every quote really occurs in the passage it cites. This module takes
the next step for a learner: it locates those words on the rendered page (word boxes from the PDF's text layer,
or from OCR for scanned pages) and returns one rectangle per line of text, ready to be drawn as a highlight.

Matching is by word sequence, tolerant of punctuation, case, hyphenation and the small differences between a
chunk's text and the page (table rows, for example, carry column headings that are not printed beside each
cell). A quote that cannot be located well enough returns nothing, never a guess.
"""

import difflib
import re
import unicodedata

MIN_COVERAGE = 0.6  # share of the quote's words that must be found, in order, close together


def _tokens(text: str) -> list[str]:
    cleaned = "".join(ch if unicodedata.category(ch)[0] in "LNM" else " " for ch in text.casefold())
    return cleaned.split()


def _page_words(page, ocr_languages: str | None):
    words = page.get_text("words")
    if not words and ocr_languages:
        try:
            textpage = page.get_textpage_ocr(language=ocr_languages, dpi=150, full=True)
            words = page.get_text("words", textpage=textpage)
        except Exception:  # noqa: BLE001 - OCR is a best effort; no boxes just means no highlight
            words = []
    return words


def _segments(quote: str) -> list[str]:
    """A table-row quote is "Heading: cell | Heading: cell"; each cell is located on its own, since the printed
    column headings sit elsewhere on the page."""
    if " | " not in quote:
        return [quote]
    cells = []
    for part in quote.split(" | "):
        label, sep, rest = part.partition(": ")
        cells.append(rest if sep and len(label.split()) <= 6 else part)
    return cells


def _locate(tokens: list[str], wanted: list[str]) -> tuple[int, int] | None:
    """Start and end (exclusive) of the page words that best carry ``wanted``, or None."""
    matcher = difflib.SequenceMatcher(None, tokens, wanted, autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size]
    if not blocks:
        return None
    best: list = []
    for anchor in blocks:  # matching runs that agree on where the quote starts belong together
        start = anchor.a - anchor.b
        group = [b for b in blocks if abs((b.a - b.b) - start) <= max(3, len(wanted) // 3)]
        if sum(b.size for b in group) > sum(b.size for b in best):
            best = group
    if sum(b.size for b in best) / len(wanted) < MIN_COVERAGE:
        return None
    return min(b.a for b in best), max(b.a + b.size for b in best)


def find_quote_rects(page, quote: str, ocr_languages: str | None = None) -> list[tuple[float, float, float, float]]:
    """One (x0, y0, x1, y1) box per text line that carries the quote, or [] when it cannot be located."""
    boxes, tokens = [], []
    for x0, y0, x1, y1, text, block, line, _ in _page_words(page, ocr_languages):
        for token in _tokens(text):  # a printed word can hold punctuation-joined parts
            tokens.append(token)
            boxes.append((x0, y0, x1, y1, block, line))
    if not tokens:
        return []
    lines: dict[tuple[int, int], list[float]] = {}
    for segment in _segments(quote):
        wanted = _tokens(segment)
        if len(wanted) < 2 and " | " not in quote:
            continue
        span = _locate(tokens, wanted) if wanted else None
        if span is None:
            continue
        for x0, y0, x1, y1, block, line in boxes[span[0]:span[1]]:
            box = lines.setdefault((block, line), [x0, y0, x1, y1])
            box[0], box[1], box[2], box[3] = min(box[0], x0), min(box[1], y0), max(box[2], x1), max(box[3], y1)
    return [tuple(box) for box in lines.values()]


def highlighted_png(page, rects, matrix) -> bytes:
    """The page as a PNG with the rectangles highlighted (the document on disk is never changed)."""
    import pymupdf
    for x0, y0, x1, y1 in rects:
        page.add_highlight_annot(pymupdf.Rect(x0, y0, x1, y1))
    return page.get_pixmap(matrix=matrix, annots=True).tobytes("png")
