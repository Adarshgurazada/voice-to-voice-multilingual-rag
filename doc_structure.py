"""Structure-aware reading of text-layer PDFs.

A plain text dump loses what a learner relies on: which section a sentence belongs to, which lines are a
running header, and which cells make up a table row. This module reads the page layout (positions, font
sizes, boldness) to recover that structure, then chunks by section and sentence:

* repeated headers/footers (the same line near the top or bottom of many pages) are dropped;
* headings become a section path such as "Section 17 · Objection handling › Objections", carried across pages;
* unruled tables (columns that merely line up) become one chunk per row, keyed by the column headings;
* paragraphs are split on sentence boundaries, never mid-sentence.

Everything is deterministic and offline. Scanned pages (OCR or vision text) have no layout and keep the old
chunker; they inherit the current section path.
"""

import re
from collections import Counter
from dataclasses import dataclass

SIZE = 800  # soft target characters per chunk
MAX_SIZE = 1100  # a chunk is only allowed past SIZE to finish a sentence
OVERLAP_SENTENCE_MAX = 220  # carry the last sentence into the next chunk only if it is this short
PATH_SEPARATOR = " › "
_KICKER_WORDS = ("section", "part", "chapter", "appendix", "module", "unit", "lesson")
_PART_WORDS = ("part", "chapter", "appendix", "unit", "module")  # these sit above sections in the path
_BULLET = re.compile(r"^\s*(?:[•·▪●*\-–]|\d{1,2}[.)])\s+")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z“\"(0-9•])")


@dataclass
class Line:
    text: str
    x0: float
    x1: float
    y0: float
    y1: float
    size: float
    bold: bool
    block: int

    @property
    def height(self) -> float:
        return max(self.y1 - self.y0, 1.0)


@dataclass
class Item:
    kind: str  # "heading" | "para" | "row"
    text: str
    level: int = 0  # headings only: 1 is the highest


def page_lines(page) -> list[Line]:
    """Every text line on the page with its geometry and font."""
    lines: list[Line] = []
    for block_index, block in enumerate(page.get_text("dict")["blocks"]):
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = "".join(s["text"] for s in line["spans"]).strip()
            bold = all((s["flags"] & 16) or "bold" in s["font"].lower() for s in spans)
            x0, y0, x1, y1 = line["bbox"]
            lines.append(Line(text, x0, x1, y0, y1, round(max(s["size"] for s in spans), 1), bool(bold), block_index))
    return lines


def _normalize(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower()).strip()


def _looks_like_heading(text: str) -> bool:
    """Short and not a sentence: a lead paragraph set in a larger font is body text, not a heading."""
    text = text.strip()
    if len(text) > 110 or re.search(r"[.!?]\s+[A-Z\u201c\"]", text):
        return False
    return not (text.endswith(".") and len(text) > 60)


def _in_margin(line: Line, page_height: float) -> bool:
    return line.y1 < 0.09 * page_height or line.y0 > 0.91 * page_height


class DocumentLayout:
    """Document-wide facts needed to read any one page: boilerplate lines, body size, heading sizes."""

    def __init__(self, doc):
        pages = len(doc)
        margin_counts: Counter = Counter()
        sizes: Counter = Counter()
        per_page = []
        for page in doc:
            lines = page_lines(page)
            per_page.append((page.rect.height, page.rect.width, lines))
            for line in lines:
                if _in_margin(line, page.rect.height):
                    margin_counts[_normalize(line.text)] += 1
        threshold = max(3, int(0.3 * pages))
        self.boilerplate = {text for text, count in margin_counts.items() if count >= threshold}
        for height, _, lines in per_page:
            for line in lines:
                if _normalize(line.text) not in self.boilerplate or not _in_margin(line, height):
                    sizes[line.size] += len(line.text)
        self.body_size = sizes.most_common(1)[0][0] if sizes else 10.0
        heading_like: Counter = Counter()
        for height, _, lines in per_page:
            for line in lines:
                if line.size >= 1.3 * self.body_size and _looks_like_heading(line.text):
                    heading_like[line.size] += 1
        self.heading_sizes = sorted(heading_like, reverse=True)[:3]

    def body_lines(self, page) -> list[Line]:
        height = page.rect.height
        return [
            line for line in page_lines(page)
            if not (_in_margin(line, height) and _normalize(line.text) in self.boilerplate)
        ]

    # -- classification -------------------------------------------------------------------------
    def heading_level(self, line: Line) -> int:
        """0 for body text, otherwise 1 (highest) to 4."""
        if line.size >= 1.3 * self.body_size and self.heading_sizes and _looks_like_heading(line.text):
            nearest = min(range(len(self.heading_sizes)), key=lambda i: abs(self.heading_sizes[i] - line.size))
            return nearest + 1
        if line.bold and len(line.text) <= 80 and line.size >= 1.15 * self.body_size:
            return 4
        return 0

    def is_kicker(self, line: Line) -> bool:
        """A small bold label such as "SECTION 17" that introduces the title below it."""
        text = line.text.strip()
        return (
            line.bold and len(text) <= 70 and line.size <= 1.25 * self.body_size
            and text.split(" ")[0].lower().rstrip(".:") in _KICKER_WORDS and any(c.isdigit() for c in text)
        ) or (line.bold and text.isupper() and 3 <= len(text) <= 40 and line.size <= 1.25 * self.body_size
              and not any(c.isdigit() for c in text) and len(text.split()) <= 4)


# -- tables -----------------------------------------------------------------------------------------
def _find_table(lines: list[Line], layout: DocumentLayout, page_width: float):
    """Find one unruled table: a bold header row of 3+ columns followed by rows aligned to those columns.

    Returns (headers, rows, used_line_ids) or None. Columns are the x positions of the header cells; a row
    starts at each gap in the first column; a line that spills past its column ends the table.
    """
    ordered = sorted(range(len(lines)), key=lambda i: (round(lines[i].y0 / 2.5), lines[i].x0))
    seen_y: set[float] = set()
    for first in ordered:
        anchor = lines[first]
        key = round(anchor.y0 / 2.5)
        if key in seen_y or not anchor.bold:
            continue
        seen_y.add(key)
        row = sorted(
            (i for i, l in enumerate(lines) if abs(l.y0 - anchor.y0) <= 2.5 and l.bold and len(l.text) <= 40),
            key=lambda i: lines[i].x0,
        )
        starts = [lines[i].x0 for i in row]
        if len(row) < 3 or any(b - a < 40 for a, b in zip(starts, starts[1:])):
            continue
        height = anchor.height
        # Header cells may wrap onto a second bold line directly under the first.
        headers = [lines[i].text for i in row]
        used = set(row)
        for column, i in enumerate(row):
            previous = lines[i]
            for j in sorted(range(len(lines)), key=lambda k: lines[k].y0):
                other = lines[j]
                if j in used or not other.bold or other.y0 <= previous.y0 or abs(other.x0 - previous.x0) > 3:
                    continue
                if other.y0 - previous.y1 <= 0.8 * height and other.y0 - anchor.y0 <= 2.4 * height:
                    headers[column] += " " + other.text
                    used.add(j)
                    previous = other
        header_bottom = max(lines[i].y1 for i in used)
        right = [s - 2 for s in starts[1:]] + [page_width - 20]
        cells: list[tuple[int, Line, int]] = []  # (column, line, index)
        end_y = None
        # y is quantized: bold and regular text on the same visual row differ by a fraction of a point.
        for j in sorted(range(len(lines)), key=lambda k: (round(lines[k].y0 / 3), lines[k].x0)):
            line = lines[j]
            if j in used or line.y0 < header_bottom - 1:
                continue
            column = max((c for c, s in enumerate(starts) if line.x0 >= s - 4), default=None)
            if (column is None or line.x1 > right[column] + 6
                    or line.size > 1.2 * layout.body_size or layout.heading_level(line)):
                end_y = line.y0
                break
            cells.append((column, line, j))
        if not cells:
            continue
        rows: list[dict[int, list[str]]] = []
        starts_y: list[float] = []
        last_first_column: Line | None = None
        for column, line, _ in cells:
            if column == 0 and (last_first_column is None or line.y0 - last_first_column.y1 > 0.6 * line.height):
                rows.append({})
                starts_y.append(line.y0)
            if column == 0:
                last_first_column = line
            if not rows:
                continue
            rows[-1].setdefault(column, []).append(line.text)
        if len(rows) < 2:
            continue
        table_rows = [[" ".join(r.get(c, [])).strip() for c in range(len(starts))] for r in rows]
        used |= {j for _, _, j in cells}
        return [h.strip() for h in headers], table_rows, used
    return None


# -- page items -------------------------------------------------------------------------------------
def _paragraphs(lines: list[Line]) -> list[str]:
    """Group lines into paragraphs: by block, starting a new one at each bullet."""
    paragraphs: list[list[str]] = []
    block = None
    for line in lines:
        if block != line.block or _BULLET.match(line.text) or not paragraphs:
            paragraphs.append([])
            block = line.block
        paragraphs[-1].append(line.text)
    out = []
    for parts in paragraphs:
        text = ""
        for part in parts:
            if text.endswith("-") and part[:1].islower():
                text = text[:-1] + part
            else:
                text = f"{text} {part}".strip()
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            out.append(text)
    return out


def page_items(page, layout: DocumentLayout) -> list[Item]:
    """Headings, paragraphs and table rows of one page in reading order."""
    lines = layout.body_lines(page)
    if not lines:
        return []
    table = _find_table(lines, layout, page.rect.width)
    table_lines = set()
    items: list[tuple[float, Item]] = []
    if table is not None:
        headers, rows, table_lines = table
        top = min(lines[i].y0 for i in table_lines)
        for row in rows:
            record = " | ".join(f"{h}: {c}" for h, c in zip(headers, row) if c)
            if record:
                items.append((top, Item("row", record)))
    rest = sorted((l for i, l in enumerate(lines) if i not in table_lines), key=lambda l: (l.block, l.y0, l.x0))
    # A block of several large-font lines that reads as sentences is a lead paragraph, not a heading.
    paragraph_like: set[tuple[int, float]] = set()
    for key in {(l.block, l.size) for l in rest}:
        joined = " ".join(l.text for l in rest if (l.block, l.size) == key)
        if not _looks_like_heading(joined):
            paragraph_like.add(key)

    def level_of(line: Line) -> int:
        return 0 if (line.block, line.size) in paragraph_like and line.size >= 1.3 * layout.body_size \
            else layout.heading_level(line)
    flow: list[tuple[float, Item]] = []
    body: list[Line] = []

    def flush_body():
        nonlocal body
        for text in _paragraphs(body):
            flow.append((body[0].y0, Item("para", text)))
        body = []

    i = 0
    while i < len(rest):
        line = rest[i]
        level = level_of(line)
        kicker = layout.is_kicker(line)
        if level or kicker:
            flush_body()
            text, y, top_level = line.text, line.y0, level or 1
            # A kicker plus the title under it, or a title wrapped over several lines, is one heading.
            while i + 1 < len(rest):
                nxt = rest[i + 1]
                follows = nxt.y0 - rest[i].y1 <= 3 * rest[i].height and nxt.block == rest[i].block or (
                    kicker and nxt.y0 - rest[i].y1 <= 3 * rest[i].height)
                nxt_level = level_of(nxt)
                if follows and nxt_level and (kicker or nxt_level == top_level):
                    text = f"{text} · {nxt.text}" if kicker else f"{text} {nxt.text}"
                    top_level, kicker = min(top_level, nxt_level), False
                    i += 1
                else:
                    break
            if text.isupper() and len(text) > 3:
                text = text.title()
            first_word = text.split(" ")[0].lower().rstrip(".:")
            is_part = first_word in _PART_WORDS and any(c.isdigit() or c.isupper() for c in text.split(" ")[1:2][0:1] or [""])
            flow.append((y, Item("part" if is_part else "heading", text, top_level)))
        else:
            body.append(line)
        i += 1
    flush_body()
    merged = sorted(flow + items, key=lambda pair: pair[0])
    return [item for _, item in merged]


# -- chunking ---------------------------------------------------------------------------------------
class SectionTracker:
    """The current heading path, carried from page to page."""

    def __init__(self):
        self.stack: list[tuple[int, str]] = []
        self.part = ""

    def set_part(self, text: str) -> None:
        self.part = text
        self.stack.clear()

    def push(self, level: int, text: str) -> None:
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, text))

    @property
    def path(self) -> str:
        return PATH_SEPARATOR.join(([self.part] if self.part else []) + [text for _, text in self.stack])


def _sentences(paragraph: str) -> list[str]:
    parts = [s.strip() for s in _SENTENCE_BREAK.split(paragraph) if s.strip()]
    out: list[str] = []
    for sentence in parts:
        while len(sentence) > MAX_SIZE:  # a runaway sentence is hard-cut at a word boundary
            cut = sentence.rfind(" ", 0, SIZE)
            cut = cut if cut > 0 else SIZE
            out.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            out.append(sentence)
    return out


def chunk_items(items: list[Item], tracker: SectionTracker) -> list[dict]:
    """Turn one page's items into chunks ``{"content", "section", "kind"}`` (sentence-aligned)."""
    chunks: list[dict] = []
    buffer: list[str] = []
    buffer_section = tracker.path

    def flush(keep_overlap: bool = False):
        nonlocal buffer, buffer_section
        if buffer:
            chunks.append({"content": " ".join(buffer).strip(), "section": buffer_section, "kind": "text"})
        carry = buffer[-1] if keep_overlap and buffer and len(buffer[-1]) <= OVERLAP_SENTENCE_MAX else None
        buffer = [carry] if carry else []
        buffer_section = tracker.path

    for item in items:
        if item.kind == "heading":
            flush()
            tracker.push(item.level, item.text)
            buffer_section = tracker.path
        elif item.kind == "part":
            flush()
            tracker.set_part(item.text)
            buffer_section = tracker.path
        elif item.kind == "row":
            flush()
            chunks.append({"content": item.text, "section": tracker.path, "kind": "row"})
        else:
            for sentence in _sentences(item.text):
                size = sum(len(s) + 1 for s in buffer)
                if buffer and size + len(sentence) > SIZE:
                    flush(keep_overlap=True)
                buffer.append(sentence)
    flush()
    # An overlap carried into an otherwise empty tail would just repeat the previous chunk.
    deduped: list[dict] = []
    for chunk in chunks:
        if deduped and chunk["content"] == deduped[-1]["content"][-len(chunk["content"]):] and chunk["kind"] == "text":
            continue
        deduped.append(chunk)
    return deduped


def structured_page_chunks(page, layout: DocumentLayout, tracker: SectionTracker) -> list[dict]:
    """Chunks for one text-layer page, or an empty list when the page has no usable layout."""
    return chunk_items(page_items(page, layout), tracker)
