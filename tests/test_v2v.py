"""Unit tests for the PDF indexing, retrieval, and Realtime payload core."""

import os
import sys
import asyncio
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

# config.py exits without a key; provide a dummy before importing the module.
os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import module_v2v_rag as m  # noqa: E402


# --------------------------------------------------------------------------- #
# test_chunking
# --------------------------------------------------------------------------- #
def test_chunking_empty_and_whitespace():
    assert m.chunk_text("") == []
    assert m.chunk_text("   \n\t ") == []


def test_chunking_small_single_chunk():
    text = "hello world this is a short document"
    chunks = m.chunk_text(text, size=800, overlap=150)
    assert chunks == [text]


def test_chunking_no_split_words_and_overlap():
    words = [f"word{i}" for i in range(200)]
    text = " ".join(words)
    chunks = m.chunk_text(text, size=50, overlap=20)
    assert len(chunks) > 1
    # No chunk exceeds size (words fit; each word <= size).
    for ch in chunks:
        assert len(ch) <= 50
        # Every whitespace-split token is a whole original word (never split).
        for tok in ch.split():
            assert tok in words
    # Overlap: consecutive chunks share at least one trailing/leading word.
    for a, b in zip(chunks, chunks[1:]):
        assert a.split()[-1] == b.split()[0] or set(a.split()) & set(b.split())


def test_chunking_forward_progress_long_token():
    long = "x" * 2500
    chunks = m.chunk_text(long, size=800, overlap=150)
    # Hard-cut into <=size pieces; reassembles to the original.
    assert all(len(c) <= 800 for c in chunks)
    assert "".join(chunks) == long


# --------------------------------------------------------------------------- #
# test_rerank_parse
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, content):
        self.choices = [mock.Mock(message=mock.Mock(content=content))]


def _client_returning(content=None, exc=None):
    client = mock.Mock()
    # rerank_candidates calls client.with_options(...).chat.completions.create(...);
    # make with_options return the same configured mock so the stub is honored.
    client.with_options.return_value = client
    if exc is not None:
        client.chat.completions.create.side_effect = exc
    else:
        client.chat.completions.create.return_value = _FakeResp(content)
    return client


def test_rerank_valid():
    candidates = ["a", "b", "c", "d", "e"]
    client = _client_returning(json.dumps({"indices": [3, 1, 0, 2, 4]}))
    result = m.rerank_candidates(client, "q", candidates, top_k=3)
    assert result == [3, 1, 0]


def test_rerank_out_of_range_falls_back():
    candidates = ["a", "b", "c"]
    client = _client_returning(json.dumps({"indices": [0, 5, 1]}))
    result = m.rerank_candidates(client, "q", candidates, top_k=3)
    assert result == [0, 1, 2]


def test_rerank_duplicate_falls_back():
    candidates = ["a", "b", "c", "d"]
    client = _client_returning(json.dumps({"indices": [0, 0, 1]}))
    result = m.rerank_candidates(client, "q", candidates, top_k=3)
    assert result == [0, 1, 2]


def test_rerank_non_int_falls_back():
    candidates = ["a", "b", "c"]
    client = _client_returning(json.dumps({"indices": ["0", 1]}))
    result = m.rerank_candidates(client, "q", candidates, top_k=2)
    assert result == [0, 1]


def test_rerank_malformed_json_falls_back():
    candidates = ["a", "b", "c", "d"]
    client = _client_returning("not-json{{{")
    result = m.rerank_candidates(client, "q", candidates, top_k=2)
    assert result == [0, 1]


def test_rerank_timeout_falls_back():
    candidates = ["a", "b", "c"]
    client = _client_returning(exc=TimeoutError("deadline"))
    result = m.rerank_candidates(client, "q", candidates, top_k=2)
    assert result == [0, 1]


def test_rerank_incomplete_list_falls_back():
    # top_k=5 but the model returns fewer than min(top_k, n) indices -> FAISS order.
    candidates = ["a", "b", "c", "d", "e"]
    client = _client_returning(json.dumps({"indices": [3]}))
    result = m.rerank_candidates(client, "q", candidates, top_k=5)
    assert result == [0, 1, 2, 3, 4]


def test_rerank_short_candidates_accepts_complete_list():
    # Fewer candidates than top_k: required == len(candidates); a complete list is OK.
    candidates = ["a", "b"]
    client = _client_returning(json.dumps({"indices": [1, 0]}))
    result = m.rerank_candidates(client, "q", candidates, top_k=5)
    assert result == [1, 0]

    # ...but an incomplete list for short candidates still falls back.
    client2 = _client_returning(json.dumps({"indices": [1]}))
    result2 = m.rerank_candidates(client2, "q", candidates, top_k=5)
    assert result2 == [0, 1]


def test_rerank_disables_client_retries():
    candidates = ["a", "b", "c"]
    client = _client_returning(json.dumps({"indices": [2, 0, 1]}))
    m.rerank_candidates(client, "q", candidates, top_k=3)
    # Retries disabled + deadline enforced via with_options.
    client.with_options.assert_called_once_with(max_retries=0, timeout=m.RERANK_TIMEOUT_S)


# --------------------------------------------------------------------------- #
# test_language_mapping
# --------------------------------------------------------------------------- #
def test_language_mapping_all_supported():
    for code in m.SUPPORTED_LANGUAGES:
        iso = m.iso_lang(code)
        assert len(iso) == 2
        assert iso == code.split("-")[0].lower() or iso in m.LANG_ISO_MAP.values()


def test_language_mapping_fallback_prefix():
    assert m.iso_lang("fr-FR") == "fr"
    assert m.iso_lang("zh-Hans-CN") == "zh"
    assert m.iso_lang("") == "en"


# --------------------------------------------------------------------------- #
# test_session_payload
# --------------------------------------------------------------------------- #
def test_session_payload():
    payload = m.build_session_payload("hi-IN", "coral")
    assert payload["output_modalities"] == ["audio"]
    assert payload["type"] == "realtime"
    assert payload["audio"]["output"]["voice"] == "coral"
    assert payload["audio"]["output"]["voice"] in m.AVAILABLE_VOICES
    assert payload["audio"]["input"]["transcription"]["language"] == "hi"
    assert payload["audio"]["input"]["format"]["rate"] == 24000
    assert payload["audio"]["output"]["format"]["rate"] == 24000
    assert payload["tool_choice"] == "required"
    assert "model" not in payload
    tool = payload["tools"][0]
    assert tool["type"] == "function"
    assert tool["name"] == "search_documents"
    assert tool["parameters"]["required"] == ["query"]
    assert tool["parameters"]["properties"]["query"]["type"] == "string"


# --------------------------------------------------------------------------- #
# test_stale_index_rejection
# --------------------------------------------------------------------------- #
def test_stale_index_rejection(tmp_path, monkeypatch):
    import faiss

    monkeypatch.chdir(tmp_path)
    # Create a fake PDF path (existence not needed for the load branch).
    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")

    # Write a stale 768-dim index + matching meta.
    stale_dim = 768
    vecs = np.random.rand(3, stale_dim).astype("float32")
    faiss.normalize_L2(vecs)
    idx = faiss.IndexFlatIP(stale_dim)
    idx.add(vecs)
    faiss.write_index(idx, "fake.faissindex")
    pd.DataFrame(
        [{"doc": "fake", "page": 1, "type": "text", "content": f"chunk {i}"} for i in range(3)]
    ).to_parquet("fake.meta.parquet")

    inst = m.VoiceRAGMultimodalFAISS.__new__(m.VoiceRAGMultimodalFAISS)
    inst.pdf_paths = [str(pdf)]
    inst.cache_dir = tmp_path
    inst.image_save_dir = tmp_path / "images"
    inst.client = mock.Mock()
    inst.faiss_index = None
    inst.metadata_db = None
    inst.is_active = True

    build_called = {"n": 0}

    def fake_build(pdf_path_str, client, image_save_dir):
        build_called["n"] += 1
        # Return a valid 1536-dim single-vector index build.
        emb = np.random.rand(m.EMBEDDING_DIM).astype("float32")
        return (
            [{"doc": "fake", "page": 1, "type": "text", "content": "rebuilt"}],
            [emb],
            m.EMBEDDING_DIM,
        )

    with mock.patch.object(m, "build_or_load_index", side_effect=fake_build):
        asyncio.run(inst.setup_rag_database())

    # Stale index rejected + removed, rebuild invoked.
    assert build_called["n"] == 1
    assert not Path("fake.faissindex").exists() or faiss.read_index("fake.faissindex").d == m.EMBEDDING_DIM
    assert inst.faiss_index is not None
    assert inst.faiss_index.d == m.EMBEDDING_DIM
