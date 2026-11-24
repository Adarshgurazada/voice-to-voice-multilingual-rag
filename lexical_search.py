"""Keyword (BM25) search over chunk text, to run next to the dense embedding search.

Embeddings are good at meaning and weak at exact tokens: a product name, an acronym ("SOC 2"), a figure, or a
rare word in a scanned page. BM25 is the opposite. Fusing the two rankings (reciprocal-rank fusion, as the
other rankings already are) recovers hits that either misses alone.

No dependency and no network. Tokens are runs of letters, digits and combining marks, so Indic scripts are
kept whole (Python's ``\\w`` would split a Telugu word at its vowel signs).
"""

import math
import re
import weakref
from collections import Counter, defaultdict

# \w plus the Devanagari-to-Sinhala blocks (their vowel signs and viramas are combining marks, not \w).
_TOKEN = re.compile(r"[\wऀ-෿‌‍]+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was", "were", "be", "it", "its",
    "this", "that", "what", "which", "who", "whom", "how", "do", "does", "did", "with", "as", "at", "by", "from",
    "about", "tell", "me", "us", "you", "your", "can", "could", "should", "would", "please", "according",
}


def tokenize(text: str) -> list[str]:
    return [t for t in (m.lower() for m in _TOKEN.findall(text)) if t not in _STOP and (len(t) > 1 or not t.isascii())]


class BM25:
    """Okapi BM25 over a list of texts; document ids are list positions."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for position, text in enumerate(texts):
            counts = Counter(tokenize(text))
            self.lengths.append(sum(counts.values()))
            for term, count in counts.items():
                self.postings[term].append((position, count))
        self.size = len(texts)
        self.avg_length = (sum(self.lengths) / self.size) if self.size else 0.0

    def scores(self, query: str) -> dict[int, float]:
        found: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = math.log(1 + (self.size - len(posting) + 0.5) / (len(posting) + 0.5))
            for position, count in posting:
                norm = 1 - self.b + self.b * self.lengths[position] / (self.avg_length or 1.0)
                found[position] += idf * count * (self.k1 + 1) / (count + self.k1 * norm)
        return found

    def top(self, query: str, limit: int = 20) -> list[int]:
        ranked = sorted(self.scores(query).items(), key=lambda item: (-item[1], item[0]))
        return [position for position, _ in ranked[:limit]]


_CACHE: dict = {}


def cached_for(frame, key: str, build):
    """Compute ``build()`` once per DataFrame and keep it beside the frame, not inside ``frame.attrs``.

    pandas deep-copies ``attrs`` every time it derives a new frame (each ``iloc``, ``fillna`` or ``iterrows``),
    so a large cache stored there is copied on every query. Entries are dropped when the frame is collected.
    """
    ident = (id(frame), key)
    entry = _CACHE.get(ident)
    if entry is not None and entry[0]() is frame and entry[1] == len(frame):
        return entry[2]
    value = build()
    _CACHE[ident] = (weakref.ref(frame, lambda _ref, ident=ident: _CACHE.pop(ident, None)), len(frame), value)
    return value


def _texts(metadata_db) -> list[str]:
    sections = metadata_db["section"].fillna("").astype(str).tolist() if "section" in metadata_db else [""] * len(metadata_db)
    return [f"{s} {c}" for s, c in zip(sections, metadata_db["content"].astype(str).tolist())]


def lexical_ranking(query: str, metadata_db, limit: int = 20) -> list[int]:
    """Positions of the chunks that best match the query's exact words (empty when no word matches)."""
    index = cached_for(metadata_db, "bm25", lambda: BM25(_texts(metadata_db)))
    return index.top(query, limit)
