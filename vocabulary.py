"""Document vocabulary: terms that bias transcription and repair near-miss spellings.

Speech recognition does not know that a company is called "Brandel" and hears "Brendel" or "brandy".
Terms that the learner's own documents use (names, product words, acronyms) are extracted once after
indexing. They are then (a) given to the transcription model as a hint and (b) used to repair words
that sound like a document term but are spelled differently in a transcript or a search query.

Everything here is deterministic and offline, so a correction is explainable and costs nothing.
"""

import difflib
import re
from collections import Counter

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'’]*[A-Za-z0-9]|[A-Za-z]")
_TITLE_NOISE = {"pdf", "doc", "docx", "final", "copy", "draft", "version", "new", "the", "and", "of", "for"}
_SENTENCE_END = ".!?:•*#|-"
# Capitalized in tables and headings but never a name worth biasing speech recognition toward.
_GENERIC = {
    "yes", "no", "none", "section", "part", "appendix", "chapter", "role", "limits", "limit", "unclear",
    "proposed", "validate", "evaluate", "english", "note", "example", "step", "steps", "page", "table",
    "figure", "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "confidential", "draft", "internal", "total", "other", "all", "any",
}
# Letters that are easily confused by ear (first letter of the heard word vs. the term).
_FIRST_LETTER_GROUPS = [set("szc"), set("kcq"), set("jg"), set("vw"), set("fp"), set("td"), set("bp"), set("mn")]


def _is_sentence_start(text: str, start: int) -> bool:
    prefix = text[max(0, start - 3):start]
    if "\n" in prefix or not prefix.strip():
        return True
    return prefix.strip()[-1:] in _SENTENCE_END


def _tokens(text: str):
    for match in _WORD.finditer(text):
        yield match.group(0), _is_sentence_start(text, match.start())


def extract_vocabulary(texts: list[str], titles: list[str] | tuple[str, ...] = (), limit: int = 40) -> list[str]:
    """Return up to ``limit`` distinctive terms (proper nouns, product words, acronyms) in display casing.

    A word counts as a term when it is capitalized in the middle of sentences much more often than
    it appears in plain lowercase, or when it only ever appears in capitals (an acronym). Words
    from the document titles are always included.
    """
    capitalized, lowercase, upper = Counter(), Counter(), Counter()
    casing: dict[str, str] = {}
    for text in texts:
        for word, initial in _tokens(text):
            lower = word.lower()
            if len(word) >= 2 and word.isupper():
                upper[lower] += 1
                casing.setdefault(lower, word)
            elif word[0].isupper() and not initial or (word[0].islower() and any(c.isupper() for c in word[1:])):
                capitalized[lower] += 1
                casing[lower] = word
            else:
                lowercase[lower] += 1  # lowercase, or capitalized only because it starts a sentence

    scored: dict[str, int] = {}
    for lower, count in capitalized.items():
        if lower in _GENERIC:
            continue
        if len(lower) >= 3 and count >= 2 and count >= 2 * lowercase[lower]:
            scored[lower] = count + upper[lower]
    for lower, count in upper.items():
        if lower in scored or lower in _GENERIC:
            continue
        if len(lower) >= 3 and count >= 2 and lowercase[lower] == 0 and capitalized[lower] == 0:
            scored[lower] = count

    ordered: list[str] = []
    for title in titles:
        for word, _ in _tokens(re.sub(r"[_\-.]+", " ", title)):
            if len(word) >= 3 and word.lower() not in _TITLE_NOISE and not re.fullmatch(r"[0-9a-f]{12,}", word.lower()):
                ordered.append(word)
    for lower, _ in sorted(scored.items(), key=lambda item: (-item[1], item[0])):
        ordered.append(casing[lower])

    seen, result = set(), []
    for word in ordered:
        key = word.lower()
        if key not in seen:
            seen.add(key)
            result.append(word)
        if len(result) >= limit:
            break
    return result


def prompt_hint(vocabulary: list[str], max_chars: int = 320) -> str:
    """The vocabulary as a comma-separated string that fits a transcription prompt."""
    chosen, size = [], 0
    for term in vocabulary:
        if size + len(term) + 2 > max_chars:
            break
        chosen.append(term)
        size += len(term) + 2
    return ", ".join(chosen)


def is_prompt_echo(text: str, vocabulary: list[str]) -> bool:
    """True when a transcript is only the vocabulary hint read back.

    Transcription models sometimes answer a stretch of silence or noise with their own prompt. Such a "transcript"
    must never be treated as something the learner said.
    """
    parts = [part.strip(" .").casefold() for part in text.replace("\n", ",").split(",")]
    parts = [part for part in parts if part]
    if len(parts) < 3 or not vocabulary:
        return False
    known = {term.casefold() for term in vocabulary}
    return sum(part in known for part in parts) / len(parts) >= 0.8


def _phonetic(word: str) -> str:
    """A tiny sound-alike key: Brendel, Brendle and Brandel all become 'brndl'."""
    w = word.lower()
    for old, new in (("ph", "f"), ("ck", "k"), ("ee", "i"), ("kn", "n"), ("wr", "r")):
        w = w.replace(old, new)
    w = re.sub(r"c(?=[eiy])", "s", w)
    w = w.replace("c", "k").replace("z", "s").replace("q", "k").replace("x", "ks")
    key = w[:1] + re.sub(r"[aeiouyh]", "", w[1:])
    return re.sub(r"(.)\1+", r"\1", key)


def _first_letters_compatible(a: str, b: str) -> bool:
    return a == b or any(a in group and b in group for group in _FIRST_LETTER_GROUPS)


class Corrector:
    """Repair words that sound like a document term but are spelled differently."""

    def __init__(self, vocabulary: list[str], known_words: set[str] | None = None):
        self.terms = {term.lower(): term for term in vocabulary if len(term) >= 3}
        # Words that really occur in the documents are never "corrected".
        self.known = {w for w in (known_words or set()) if w not in self.terms}
        self._phonetic = {lower: _phonetic(lower) for lower in self.terms}

    def _best_term(self, lower: str) -> str | None:
        best, best_ratio = None, 0.0
        key = _phonetic(lower)
        for term_lower, canonical in self.terms.items():
            if abs(len(lower) - len(term_lower)) > 2 or not _first_letters_compatible(lower[0], term_lower[0]):
                continue
            ratio = difflib.SequenceMatcher(None, lower, term_lower).ratio()
            sounds_alike = key == self._phonetic[term_lower] and len(key) >= 3
            if (ratio >= 0.8 or (sounds_alike and ratio >= 0.6)) and ratio > best_ratio:
                best, best_ratio = canonical, ratio
        return best

    def correct(self, text: str) -> tuple[str, list[dict]]:
        """Return the text with near-miss spellings replaced, and what was changed."""
        if not text or not self.terms:
            return text, []
        fixes: list[dict] = []

        def replace(match: re.Match) -> str:
            word = match.group(0)
            lower = word.lower()
            if lower in self.terms:
                return self.terms[lower]  # right word, right casing (not reported as a correction)
            if len(lower) < 4 or lower in self.known:
                return word
            canonical = self._best_term(lower)
            if canonical is None:
                return word
            fixes.append({"from": word, "to": canonical})
            return canonical

        return _WORD.sub(replace, text), fixes


def build_known_words(texts: list[str]) -> set[str]:
    """Every lowercase word that occurs in the documents."""
    known: set[str] = set()
    for text in texts:
        known.update(word.lower() for word, _ in _tokens(text))
    return known
