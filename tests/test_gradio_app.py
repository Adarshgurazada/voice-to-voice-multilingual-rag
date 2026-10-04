"""Focused tests for browser audio, uploads, and source references."""

import asyncio
import base64
import io
import os
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio_app as app  # noqa: E402
import module_v2v_rag as rag  # noqa: E402


def test_audio_to_pcm16_resamples_mono():
    source = np.full((1600, 2), 2000, dtype=np.int16)
    pcm = app._audio_to_pcm16((16000, source))
    samples = np.frombuffer(pcm, dtype="<i2")
    assert abs(len(samples) - 2400) <= 1
    assert 1500 < samples[100] < 2500


def test_audio_rejects_empty_and_long_recordings():
    with pytest.raises(ValueError, match="empty"):
        app._audio_to_pcm16((16000, np.array([], dtype=np.int16)))
    with pytest.raises(ValueError, match="45 seconds"):
        app._audio_to_pcm16((16000, np.zeros(46 * 16000, dtype=np.int16)))


def test_pcm_to_wav_is_playable():
    pcm = np.array([0, 1000, -1000], dtype="<i2").tobytes()
    result = app._pcm_to_wav(pcm)
    with wave.open(io.BytesIO(result), "rb") as wav_file:
        assert wav_file.getframerate() == rag.OUTPUT_RATE
        assert wav_file.getnchannels() == 1
        assert wav_file.readframes(3) == pcm


def test_store_pdf_uses_content_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "DOCUMENT_DIR", tmp_path / "stored")
    uploaded = tmp_path / "chapter.pdf"
    uploaded.write_bytes(b"%PDF-1.4\nfirst version")
    first, name = app._store_pdf(str(uploaded))
    assert name == "chapter.pdf"
    assert first.read_bytes() == uploaded.read_bytes()
    uploaded.write_bytes(b"%PDF-1.4\nsecond version")
    second, _ = app._store_pdf(str(uploaded))
    assert second != first


def test_retrieval_returns_selected_source_rows(monkeypatch):
    class FakeIndex:
        ntotal = 2

        def search(self, vector, count):
            assert count == 2
            return np.array([[0.9, 0.8]]), np.array([[0, 1]])

    rows = pd.DataFrame([
        {"doc": "lesson", "page": 2, "type": "text", "content": "First passage"},
        {"doc": "lesson", "page": 5, "type": "image", "content": "A diagram", "image_path": "diagram.png"},
    ])
    monkeypatch.setattr(rag, "get_text_embedding", lambda client, query: [1.0, 0.0])
    monkeypatch.setattr(rag, "rerank_candidates", lambda client, query, candidates, top_k: [1, 0])
    context, sources = rag.retrieve_context_and_sources("diagram", FakeIndex(), rows, None, 2)
    assert sources == [
        {"doc": "lesson", "page": 5, "type": "image"},
        {"doc": "lesson", "page": 2, "type": "text"},
    ]
    assert "Pg 5" in context


def test_browser_turn_waits_for_search_then_spoken_answer(monkeypatch):
    class FakeConnection:
        def __init__(self):
            self.events = [
                SimpleNamespace(type="conversation.item.input_audio_transcription.completed", transcript="What is a cell?"),
                SimpleNamespace(type="response.function_call_arguments.done", arguments='{"query":"cell"}', call_id="call-1"),
                SimpleNamespace(type="response.done"),
            ]
            self.tool_output = None
            self.responses = []
            self.session = SimpleNamespace(update=self.update)
            self.input_audio_buffer = SimpleNamespace(append=self.append, commit=self.commit)
            self.response = SimpleNamespace(create=self.create_response)
            self.conversation = SimpleNamespace(item=SimpleNamespace(create=self.create_item))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.events:
                return self.events.pop(0)
            raise StopAsyncIteration

        async def update(self, session):
            assert session["tools"][0]["name"] == "search_documents"

        async def append(self, audio):
            assert base64.b64decode(audio) == b"\x00\x00"

        async def commit(self):
            pass

        async def create_item(self, item):
            self.tool_output = item

        async def create_response(self, response=None):
            self.responses.append(response)
            if response == {"tool_choice": "none"}:
                self.events.extend([
                    SimpleNamespace(type="response.output_audio.delta", delta=base64.b64encode(b"\x00\x00").decode()),
                    SimpleNamespace(type="response.output_audio_transcript.done", transcript="A cell is a basic unit of life."),
                    SimpleNamespace(type="response.done"),
                ])

    connection = FakeConnection()
    client = SimpleNamespace(realtime=SimpleNamespace(connect=lambda model: connection))
    engine = SimpleNamespace(async_client=client, faiss_index=object(), metadata_db=object(), client=object())
    session = app.BrowserSession(engine, {}, {})
    monkeypatch.setattr(app, "retrieve_context_and_sources", lambda *args: (
        "[Text Context - Source: book, Pg 3] A cell is a basic unit of life.",
        [{"doc": "book", "page": 3, "type": "text"}],
    ))

    question, answer, audio, sources = asyncio.run(app._answer_turn(session, b"\x00\x00", "en-US", "alloy"))
    assert question == "What is a cell?"
    assert answer == "A cell is a basic unit of life."
    assert audio.startswith(b"RIFF")
    assert sources[0]["page"] == 3
    assert connection.tool_output["call_id"] == "call-1"
    assert connection.responses == [None, {"tool_choice": "none"}]


def test_gradio_layout_builds_without_starting_server():
    demo = app.build_app()
    assert demo is not None
