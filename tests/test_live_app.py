"""Offline checks for the continuous browser Realtime path."""

import asyncio
import base64
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import live_app as live  # noqa: E402
from voice_session import BrowserSession  # noqa: E402
from voice_session import _apply_preference_update  # noqa: E402


def _session(tmp_path):
    path = tmp_path / "lesson.pdf"
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Study source")
    document.save(path)
    document.close()
    browser = BrowserSession(SimpleNamespace(), {"lesson": path}, {"lesson": "Lesson.pdf"})
    return live.LiveSession("test-session", browser, [path], status="ready")


def test_live_payload_has_server_vad_and_required_retrieval():
    payload = live.live_session_payload(BrowserSession(None, {}, {}))
    vad = payload["audio"]["input"]["turn_detection"]
    assert vad["type"] == "server_vad"
    assert vad["create_response"] is True
    assert vad["interrupt_response"] is True
    assert payload["tool_choice"] == "required"
    assert "language" not in payload["audio"]["input"]["transcription"]


def test_preference_update_validates_whole_request():
    assert _apply_preference_update({"language": "hi-IN", "voice": "cedar"}, "auto", "alloy")[:2] == (
        "hi-IN", "cedar",
    )
    assert _apply_preference_update({"language": "hi-IN", "voice": "unknown"}, "auto", "alloy")[:2] == (
        "auto", "alloy",
    )


def test_retrieval_returns_distinct_source_pages(monkeypatch):
    from module_v2v_rag import retrieve_context_and_sources, retrieve_documents
    import module_v2v_rag as rag

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
    context, sources = retrieve_context_and_sources("diagram", FakeIndex(), rows, None, 2)
    assert [(source["doc"], source["page"]) for source in sources] == [("lesson", 5), ("lesson", 2)]
    assert "Pg 5" in context
    detailed = retrieve_documents("diagram", FakeIndex(), rows, None, 2)
    assert [(source["doc"], source["page"]) for source in detailed.candidates] == [
        ("lesson", 2), ("lesson", 5),
    ]
    assert detailed.candidates[0]["score"] == pytest.approx(0.9)


def test_source_page_is_previewed_and_validated(tmp_path, monkeypatch):
    session = _session(tmp_path)
    monkeypatch.setitem(live.sessions, session.id, session)
    client = TestClient(live.app)
    assert client.get("/").status_code == 200
    assert client.get("/assets/app.js").status_code == 200
    page = client.get(f"/api/sessions/{session.id}/pages/lesson/1")
    assert page.status_code == 200
    assert page.headers["content-type"] == "image/png"
    assert page.content.startswith(b"\x89PNG")
    assert client.get(f"/api/sessions/{session.id}/pages/lesson/2").status_code == 404
    assert client.get(f"/api/sessions/{session.id}/pages/other/1").status_code == 404


def test_browser_audio_is_forwarded_without_manual_turn_commit():
    class FakeWebSocket:
        def __init__(self):
            self.messages = [
                {"type": "websocket.receive", "bytes": b"\x01\x00\x02\x00"},
                {"type": "websocket.disconnect"},
            ]

        async def receive(self):
            return self.messages.pop(0)

    class FakeBuffer:
        def __init__(self):
            self.audio = []

        async def append(self, *, audio):
            self.audio.append(audio)

    buffer = FakeBuffer()
    conn = SimpleNamespace(input_audio_buffer=buffer)
    asyncio.run(live._from_browser(FakeWebSocket(), conn))
    assert buffer.audio == [base64.b64encode(b"\x01\x00\x02\x00").decode()]


def test_realtime_relays_transcript_audio_and_turn_events(tmp_path):
    session = _session(tmp_path)

    class FakeWebSocket:
        def __init__(self):
            self.json_events = []
            self.audio = []

        async def send_json(self, event):
            self.json_events.append(event)

        async def send_bytes(self, audio):
            self.audio.append(audio)

    class FakeConnection:
        def __init__(self):
            self.events = iter([
                SimpleNamespace(type="input_audio_buffer.speech_started"),
                SimpleNamespace(type="input_audio_buffer.speech_stopped"),
                SimpleNamespace(type="conversation.item.input_audio_transcription.completed", item_id="u1", transcript="Explain the page"),
                SimpleNamespace(type="response.created", response=SimpleNamespace(id="r1")),
                SimpleNamespace(type="response.output_audio.delta", delta=base64.b64encode(b"\x01\x00").decode()),
                SimpleNamespace(type="response.output_audio_transcript.done", item_id="a1", transcript="Here is the answer"),
                SimpleNamespace(type="response.done", response=SimpleNamespace(id="r1", status="completed")),
            ])

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.events)
            except StopIteration:
                raise StopAsyncIteration

    ws = FakeWebSocket()
    asyncio.run(live._from_openai(ws, FakeConnection(), session))
    assert ws.audio == [b"\x01\x00"]
    assert [event["type"] for event in ws.json_events if event["type"] != "sources"] == [
        "speech_started", "speech_stopped", "transcript", "transcript", "metrics", "turn_done",
    ]
    assert session.metrics[0]["time_to_first_audio_ms"] >= 0
    assert session.browser.turns == [("Explain the page", "Here is the answer")]


def test_document_tool_searches_and_continues_spoken_response(tmp_path, monkeypatch):
    session = _session(tmp_path)
    session.browser.engine = SimpleNamespace(faiss_index=object(), metadata_db=object(), client=object())
    queries = []

    def fake_retrieve(query, *args):
        queries.append(query)
        from module_v2v_rag import RetrievalResult
        return RetrievalResult(
            "Source: lesson, Pg 1 - Study source",
            passages=[{"doc": "lesson", "page": 1, "type": "text", "content": "Study source"}],
        )

    monkeypatch.setattr(live, "retrieve_documents", fake_retrieve)

    class FakeWebSocket:
        def __init__(self):
            self.events = []

        async def send_json(self, event):
            self.events.append(event)

    class FakeConnection:
        def __init__(self):
            self.events = iter([
                SimpleNamespace(type="conversation.item.input_audio_transcription.completed", item_id="u1", transcript="Explain the page"),
                SimpleNamespace(type="response.created", response=SimpleNamespace(id="r1")),
                SimpleNamespace(type="response.function_call_arguments.done", response_id="r1", call_id="c1", name="search_documents", arguments='{"query":"study source"}'),
                SimpleNamespace(type="response.done", response=SimpleNamespace(id="r1", status="completed")),
                SimpleNamespace(type="response.created", response=SimpleNamespace(id="r2")),
                SimpleNamespace(type="response.output_audio_transcript.done", item_id="a1", transcript="The page says study source."),
                SimpleNamespace(type="response.done", response=SimpleNamespace(id="r2", status="completed")),
            ])
            self.outputs = []
            self.responses = []
            self.conversation = SimpleNamespace(item=SimpleNamespace(create=self.create_item))
            self.response = SimpleNamespace(create=self.create_response)
            self.session = SimpleNamespace(update=self.update_session)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.events)
            except StopIteration:
                raise StopAsyncIteration

        async def create_item(self, *, item):
            self.outputs.append(item)

        async def create_response(self, *, response):
            self.responses.append(response)

        async def update_session(self, *, session):
            pass

    ws = FakeWebSocket()
    conn = FakeConnection()
    asyncio.run(live._from_openai(ws, conn, session))
    assert queries == ["study source"]
    assert conn.outputs[0]["type"] == "function_call_output"
    assert "[S1]" in conn.outputs[0]["output"]
    assert conn.responses == [{"tool_choice": "none"}]
    assert session.browser.turns == [("Explain the page", "The page says study source.")]
    assert any(event["type"] == "sources" and event["sources"] for event in ws.events)
    assert any(event["type"] == "sources" and event["sources"] and
               event["sources"][0]["id"] == "S1" for event in ws.events)


def test_quiz_answer_is_not_exposed_on_question_turn(tmp_path):
    session = _session(tmp_path)
    quiz = SimpleNamespace(question="What is X?", answer="Secret", source={"doc": "lesson", "page": 1})
    session.browser.quiz = quiz
    session.quiz_started_this_turn = True
    result = asyncio.run(live._run_tool(session, "skip_quiz", {}, "skip"))
    assert json.loads(result) == {"status": "wait_for_learner_answer"}
    assert session.browser.quiz is quiz
