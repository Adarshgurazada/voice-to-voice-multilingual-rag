"""Unit tests for the OpenAI-migrated voice-to-voice RAG module.

Set OPENAI_API_KEY before import (config exits without one):
    OPENAI_API_KEY=x python -m pytest -q
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

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


# --------------------------------------------------------------------------- #
# test_tool_state_machine
# --------------------------------------------------------------------------- #
class _Event:
    def __init__(self, type, **kw):
        self.type = type
        for k, v in kw.items():
            setattr(self, k, v)


class _FakeConn:
    """Scripted async fake realtime connection for the tool-call state machine."""

    def __init__(self, script):
        # script: list of events emitted for the initial (tool) response.
        self._script = list(script)
        self.function_outputs = []  # (call_id, output)
        self.response_creates = []  # response kwargs / None
        self._continuation_emitted = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._script:
            return self._script.pop(0)
        raise StopAsyncIteration

    # conn.conversation.item.create(item=...)
    @property
    def conversation(self):
        outer = self

        class _Item:
            async def create(self, item=None):
                if item and item.get("type") == "function_call_output":
                    outer.function_outputs.append((item["call_id"], item["output"]))

        class _Conv:
            item = _Item()

        return _Conv()

    # conn.response.create(response=...)
    @property
    def response(self):
        outer = self

        class _Response:
            async def create(self, response=None):
                outer.response_creates.append(response)
                # Simulate the continuation's terminal response.done.
                if response and response.get("tool_choice") == "none" and not outer._continuation_emitted:
                    outer._continuation_emitted = True
                    outer._script.append(_Event("response.done"))

        return _Response()


def test_tool_state_machine():
    inst = m.VoiceRAGMultimodalFAISS.__new__(m.VoiceRAGMultimodalFAISS)
    inst.is_active = True
    inst.playback_queue = asyncio.Queue()
    inst.last_transcript = ""
    inst.faiss_index = object()
    inst.metadata_db = object()
    inst.client = mock.Mock()

    async def fake_retrieval(query):
        assert query == "what is justice"
        return "CONTEXT: grounded passage"

    inst._run_retrieval = fake_retrieval

    script = [
        _Event("conversation.item.input_audio_transcription.completed", transcript="tell me about justice"),
        _Event("response.function_call_arguments.done",
               arguments=json.dumps({"query": "what is justice"}), call_id="call_abc"),
        _Event("response.output_audio_transcript.done", transcript="Justice is..."),
        _Event("response.done"),  # tool-call response done -> continuation
    ]
    conn = _FakeConn(script)

    asyncio.run(inst._consume_turn(conn))

    # function_call_output sent with the right call_id + retrieved context.
    assert conn.function_outputs == [("call_abc", "CONTEXT: grounded passage")]
    # A continuation response.create(tool_choice="none") was issued.
    assert conn.response_creates == [{"tool_choice": "none"}]


def test_tool_state_machine_does_not_end_on_first_done():
    """The turn must end only on the continuation's response.done, not the first."""
    inst = m.VoiceRAGMultimodalFAISS.__new__(m.VoiceRAGMultimodalFAISS)
    inst.is_active = True
    inst.playback_queue = asyncio.Queue()
    inst.last_transcript = "fallback transcript"
    inst.faiss_index = object()
    inst.metadata_db = object()
    inst.client = mock.Mock()

    async def fake_retrieval(query):
        # No query in args -> falls back to saved transcript.
        assert query == "fallback transcript"
        return "CTX"

    inst._run_retrieval = fake_retrieval

    script = [
        _Event("conversation.item.input_audio_transcription.completed", transcript="fallback transcript"),
        _Event("response.function_call_arguments.done", arguments="{}", call_id="cid"),
        _Event("response.done"),  # first done -> should NOT end the turn
    ]
    conn = _FakeConn(script)
    asyncio.run(inst._consume_turn(conn))

    # Continuation issued (proves first done did not terminate).
    assert conn.response_creates == [{"tool_choice": "none"}]
    assert conn.function_outputs == [("cid", "CTX")]


def test_tool_state_machine_malformed_args_uses_transcript():
    """Malformed / non-dict tool args must not crash; fall back to the transcript."""
    inst = m.VoiceRAGMultimodalFAISS.__new__(m.VoiceRAGMultimodalFAISS)
    inst.is_active = True
    inst.playback_queue = asyncio.Queue()
    inst.last_transcript = "spoken question"
    inst.faiss_index = object()
    inst.metadata_db = object()
    inst.client = mock.Mock()

    seen = {}

    async def fake_retrieval(query):
        seen["query"] = query
        return "CTX"

    inst._run_retrieval = fake_retrieval

    # arguments is a JSON list (not a dict) -> AttributeError would crash the old code.
    script = [
        _Event("conversation.item.input_audio_transcription.completed", transcript="spoken question"),
        _Event("response.function_call_arguments.done", arguments="[1, 2, 3]", call_id="cid"),
        _Event("response.done"),
    ]
    conn = _FakeConn(script)
    asyncio.run(inst._consume_turn(conn))

    assert seen["query"] == "spoken question"
    assert conn.function_outputs == [("cid", "CTX")]
    assert conn.response_creates == [{"tool_choice": "none"}]


class _GatedConn:
    """Async fake conn whose continuation response.done is gated behind an Event.

    Emits the tool-phase events, then blocks (no more events) until the test sets
    ``release`` -- at which point the continuation response.done is delivered.
    This proves the consumer waits for the second response.done rather than
    ending on the first.
    """

    def __init__(self):
        self._queue = asyncio.Queue()
        self.release = asyncio.Event()
        self.function_outputs = []
        self.response_creates = []
        self._continuation_started = False
        for ev in (
            _Event("conversation.item.input_audio_transcription.completed", transcript="q"),
            _Event("response.function_call_arguments.done",
                   arguments=json.dumps({"query": "the query"}), call_id="cid"),
            _Event("response.done"),  # tool-phase done
        ):
            self._queue.put_nowait(ev)

    def __aiter__(self):
        return self

    async def __anext__(self):
        # Once the continuation is pending, wait until released before emitting
        # its response.done. If empty and not yet gated, wait for the queue.
        if self._continuation_started and self._queue.empty():
            await self.release.wait()
            self._continuation_started = False  # deliver exactly one done
            return _Event("response.done")
        return await self._queue.get()

    @property
    def conversation(self):
        outer = self

        class _Item:
            async def create(self, item=None):
                if item and item.get("type") == "function_call_output":
                    outer.function_outputs.append((item["call_id"], item["output"]))

        class _Conv:
            item = _Item()

        return _Conv()

    @property
    def response(self):
        outer = self

        class _Response:
            async def create(self, response=None):
                outer.response_creates.append(response)
                if response and response.get("tool_choice") == "none":
                    outer._continuation_started = True

        return _Response()


def test_tool_state_machine_waits_for_continuation_done():
    """Consumer must remain pending after the first done and finish only once the
    continuation's response.done is delivered."""
    inst = m.VoiceRAGMultimodalFAISS.__new__(m.VoiceRAGMultimodalFAISS)
    inst.is_active = True
    inst.playback_queue = asyncio.Queue()
    inst.last_transcript = ""
    inst.faiss_index = object()
    inst.metadata_db = object()
    inst.client = mock.Mock()

    async def fake_retrieval(query):
        return "CTX"

    inst._run_retrieval = fake_retrieval

    async def scenario():
        conn = _GatedConn()
        task = asyncio.create_task(inst._consume_turn(conn))
        # Let the consumer process the tool phase + first response.done and issue
        # the continuation. It should now be waiting (gated) for the second done.
        for _ in range(20):
            await asyncio.sleep(0)
            if conn.response_creates:
                break
        await asyncio.sleep(0.05)
        assert conn.response_creates == [{"tool_choice": "none"}]
        assert conn.function_outputs == [("cid", "CTX")]
        # First done did NOT end the turn: task still pending.
        assert not task.done()
        # Release the continuation's response.done; now the turn should end.
        conn.release.set()
        await asyncio.wait_for(task, timeout=1.0)
        assert task.done()

    asyncio.run(scenario())


# --------------------------------------------------------------------------- #
# Optional live integration test (guarded; skipped without env flag + key)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    os.getenv("RUN_REALTIME_IT") != "1",
    reason="Live realtime integration test requires RUN_REALTIME_IT=1 and a real key.",
)
def test_realtime_integration_smoke():  # pragma: no cover - manual only
    pytest.skip("Manual/no-hardware integration test placeholder.")
