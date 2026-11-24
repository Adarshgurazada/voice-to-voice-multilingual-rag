import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from lexical_search import BM25, lexical_ranking, tokenize

TEXTS = [
    "Brandel records live meetings and drafts follow-ups.",
    "Security: we hold a SOC 2 report and encrypt data with AES-256.",
    "Pricing plans: Pro, Ultra and Limited.",
    "General notes about meetings and customers.",
]


def test_rare_exact_token_ranks_first():
    index = BM25(TEXTS)
    assert index.top("Do you have SOC 2?")[0] == 1
    assert index.top("AES-256 encryption")[0] == 1


def test_stopwords_alone_match_nothing():
    assert BM25(TEXTS).top("what is the") == []


def test_indic_words_are_kept_whole():
    word = "పోతన"  # Telugu "Potana" with a vowel sign that \w alone would split on
    assert tokenize(f"{word} కవి") [0] == word


def test_telugu_query_finds_telugu_chunk():
    texts = ["పోతన భాగవతం రచించాడు", "english only text"]
    assert BM25(texts).top("పోతన ఎవరు")[0] == 0


def test_section_path_is_searchable_and_index_is_cached():
    frame = pd.DataFrame({"content": ["Reply politely."], "section": ["Section 17 · Objection handling"]})
    assert lexical_ranking("objection handling", frame) == [0]
    assert frame.attrs == {}  # caches must not live in attrs: pandas deep-copies them on every derived frame


def test_hybrid_lifts_an_exact_match_the_embedding_missed(monkeypatch):
    import numpy as np
    import module_v2v_rag as rag
    rows = pd.DataFrame([
        {"doc": "d", "page": 1, "type": "text", "content": "General notes about meetings."},
        {"doc": "d", "page": 2, "type": "text", "content": "General remarks about customers."},
        {"doc": "d", "page": 3, "type": "text", "content": "We hold a SOC 2 report."},
    ])

    class Index:
        ntotal = 3

        def search(self, matrix, k):
            return np.array([[0.9, 0.8, 0.1]]), np.array([[0, 1, 2]])  # dense puts the SOC 2 chunk last

    monkeypatch.setattr(rag, "get_text_embeddings", lambda client, texts: [[1.0, 0.0] for _ in texts])
    monkeypatch.setattr(rag, "rerank_candidates", lambda client, q, texts, k: list(range(min(k, len(texts)))))
    monkeypatch.setattr(rag, "expand_query", lambda client, q: [q])
    dense = rag.retrieve_documents("SOC 2", Index(), rows, None, 3, hybrid=False)
    hybrid = rag.retrieve_documents("SOC 2", Index(), rows, None, 3, hybrid=True)
    assert [c["page"] for c in dense.candidates] == [1, 2, 3]
    assert hybrid.candidates[0]["page"] == 3 or hybrid.candidates[1]["page"] == 3 and hybrid.candidates[0]["page"] == 1
    assert [c["page"] for c in hybrid.candidates].index(3) < [c["page"] for c in dense.candidates].index(3)


def test_followup_gate_and_rewrite_fall_back_safely():
    from types import SimpleNamespace
    import module_v2v_rag as rag
    history = [("What does Callipolis mean?", "Beautiful city.")]
    assert rag.looks_like_followup("Who are the military elite in it?", history)
    assert rag.looks_like_followup("And the second one?", history)
    assert not rag.looks_like_followup("Who are the military elite in it?", [])
    assert not rag.looks_like_followup(
        "Explain in detail how the Republic argues that justice in the city mirrors justice in the soul", history)

    def boom(**kwargs):
        raise RuntimeError("down")

    client = SimpleNamespace(with_options=lambda **k: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=boom))))
    assert rag.rewrite_followup(client, "And the second one?", history) == "And the second one?"


def test_rewritten_followup_is_searched_and_reported(monkeypatch):
    import numpy as np
    import module_v2v_rag as rag
    rows = pd.DataFrame([{"doc": "d", "page": 1, "type": "text", "content": "Callipolis guardians defend the city."}])
    searched, said = [], []

    class Index:
        ntotal = 1

        def search(self, matrix, k):
            return np.array([[0.9] * len(matrix)]).reshape(len(matrix), 1), np.array([[0]] * len(matrix))

    monkeypatch.setattr(rag, "get_text_embeddings", lambda client, texts: searched.extend(texts) or [[1.0, 0.0] for _ in texts])
    monkeypatch.setattr(rag, "rerank_candidates", lambda client, q, texts, k: [0])
    monkeypatch.setattr(rag, "expand_query", lambda client, q: [q])
    monkeypatch.setattr(rag, "rewrite_followup", lambda client, q, h: "Who are the military elite in Callipolis?")
    rag.retrieve_documents("elite", Index(), rows, None, 1, progress=said.append,
                           question_hint="Who are the military elite in it?",
                           history=[("What does Callipolis mean?", "Beautiful city.")])
    assert "Who are the military elite in Callipolis?" in searched
    assert any(s.startswith("Understanding this as") for s in said)


def test_cache_is_built_once_and_dropped_with_the_frame():
    import gc
    import lexical_search as ls
    frame = pd.DataFrame({"content": ["alpha beta", "gamma"]})
    built = []
    first = ls.cached_for(frame, "demo", lambda: built.append(1) or object())
    assert ls.cached_for(frame, "demo", lambda: built.append(1) or object()) is first and len(built) == 1
    ident = (id(frame), "demo")
    del frame
    gc.collect()
    assert ident not in ls._CACHE
