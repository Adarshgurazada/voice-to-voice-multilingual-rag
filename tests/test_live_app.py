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
    monkeypatch.setattr(rag, "lexical_ranking", lambda *a, **k: [])  # this test is about dense order and reranking
    context, sources = retrieve_context_and_sources("diagram", FakeIndex(), rows, None, 2)
    assert [(source["doc"], source["page"]) for source in sources] == [("lesson", 5), ("lesson", 2)]
    assert "Pg 5" in context
    detailed = retrieve_documents("diagram", FakeIndex(), rows, None, 2, hybrid=False)  # pure dense order
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
    asyncio.run(live._from_browser(FakeWebSocket(), conn, None))
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
    assert [event["type"] for event in ws.json_events if event["type"] not in ("sources", "evidence")] == [
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
    monkeypatch.setattr(live, "GROUNDED_VOICE", False)

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


def test_language_comes_from_transcript_script_not_the_models_guess():
    from voice_session import language_from_transcript
    assert language_from_transcript("పోతన ఏ గ్రామానికి చెందినవాడు?", "ml-IN") == "te-IN"
    assert language_from_transcript("तोते की आयु कितनी है?", "mr-IN") == "mr-IN"
    assert language_from_transcript("तोते की आयु कितनी है?", "ta-IN") == "hi-IN"
    assert language_from_transcript("Who led the Thirty?", "ml-IN") == "en-US"
    assert language_from_transcript("", "te-IN") is None


def test_tool_call_overrides_a_wrong_claimed_language(tmp_path):
    session = _session(tmp_path)
    out = asyncio.run(live._run_tool(
        session, "set_preferences", {"spoken_language": "ml-IN"}, "పోతన ఏ గ్రామానికి చెందినవాడు?"))
    assert session.browser.last_spoken_language == "te-IN"
    assert session.turn_metrics["language_claim_overridden"] is True
    assert "No language or voice change" in out


def test_language_decision_waits_for_the_late_transcript(tmp_path):
    session = _session(tmp_path)

    async def scenario():
        async def transcript_arrives_late():
            await asyncio.sleep(0.05)
            session.current_question = "పోతన ఏ గ్రామానికి చెందినవాడు?"
            session.transcript_event.set()

        task = asyncio.create_task(transcript_arrives_late())
        await live._run_tool(session, "set_preferences", {"spoken_language": "en-US"}, "")
        await task

    asyncio.run(scenario())
    assert session.browser.last_spoken_language == "te-IN"
    assert session.browser.language_confident is True


def test_instruction_only_asserts_a_language_when_it_was_detected():
    from voice_session import _browser_instructions
    unsure = _browser_instructions("auto", "en-US", [], None, False)
    assert "do not default to English" in unsure and "server detected" not in unsure
    sure = _browser_instructions("auto", "te-IN", [], None, True)
    assert "server detected it from their transcript as Telugu" in sure


def _grounded_session(tmp_path, monkeypatch, result):
    from module_v2v_rag import RetrievalResult
    session = _session(tmp_path)
    session.browser.engine = SimpleNamespace(faiss_index=object(), metadata_db=object(), client=object())
    passages = [{"doc": "lesson", "page": 1, "type": "text", "content": "Study source"},
                {"doc": "lesson", "page": 2, "type": "text", "content": "Other page"}]
    monkeypatch.setattr(live, "retrieve_documents", lambda *a: RetrievalResult("ctx", passages=passages))
    monkeypatch.setattr(live, "GROUNDED_VOICE", True)
    calls = []

    def fake_answer(client, question, numbered, language, *extra):
        calls.append((question, [p["id"] for p in numbered], language))
        return result

    monkeypatch.setattr(live, "answer_question", fake_answer)
    return session, calls


def test_grounded_tool_returns_only_verified_claims_and_sources(tmp_path, monkeypatch):
    from grounded_answer import GroundedAnswer
    verified = GroundedAnswer(
        "The page says study source.", False, "en",
        claims=[{"text": "The page says study source.", "source_id": "S1", "quote": "Study source"}],
        dropped=[{"text": "x", "source_id": "S2", "quote": "y", "reason": "not_supported"}],
    )
    session, calls = _grounded_session(tmp_path, monkeypatch, verified)
    out = json.loads(asyncio.run(live._run_tool(
        session, "search_documents", {"query": "study source"}, "What is the study source?")))
    assert out["status"] == "verified_answer"
    assert out["claims"] == [{"role": "answer", "source": "S1", "claim": "The page says study source."}]
    assert calls == [("What is the study source?", ["S1", "S2"], "en")]
    assert [s["id"] for s in session.turn_sources] == ["S1"]  # the dropped S2 is never shown
    assert session.turn_metrics["claims_verified"] == 1 and session.turn_metrics["claims_dropped"] == 1
    assert session.turn_metrics["grounded_abstained"] is False
    assert "The page says study source." in session.followup_instructions
    assert "Add no other facts" in session.followup_instructions


def test_grounded_tool_abstains_without_sources(tmp_path, monkeypatch):
    from grounded_answer import GroundedAnswer
    session, _ = _grounded_session(
        tmp_path, monkeypatch, GroundedAnswer("I couldn't find that in your documents.", True, "en"))
    out = json.loads(asyncio.run(live._run_tool(
        session, "search_documents", {"query": "capital of mars"}, "capital of mars")))
    assert out["status"] == "not_found" and out["message"].startswith("I couldn't find")
    assert session.turn_sources == []
    assert session.turn_metrics["grounded_abstained"] is True
    assert "nothing else" in session.followup_instructions and "own knowledge" in session.followup_instructions


def test_grounded_tool_passes_detected_language(tmp_path, monkeypatch):
    from grounded_answer import GroundedAnswer
    session, calls = _grounded_session(tmp_path, monkeypatch, GroundedAnswer("x", True, "te"))
    asyncio.run(live._run_tool(session, "search_documents", {"query": "study source"}, "ఈ పేజీ ఏమి చెబుతుంది"))
    assert calls[0][2] == "te"


def _language_harness(tmp_path, messages):
    session = _session(tmp_path)

    class FakeWebSocket:
        def __init__(self):
            self.sent = []
            self.messages = list(messages) + [{"type": "websocket.disconnect"}]

        async def receive(self):
            return self.messages.pop(0)

        async def send_json(self, event):
            self.sent.append(event)

    class FakeConnection:
        def __init__(self):
            self.updates = []
            self.session = SimpleNamespace(update=self.update)

        async def update(self, *, session):
            self.updates.append(session)

    ws, conn = FakeWebSocket(), FakeConnection()
    asyncio.run(live._from_browser(ws, conn, session))
    return session, ws, conn


def test_dropdown_language_sticks_and_updates_instructions(tmp_path):
    session, ws, conn = _language_harness(
        tmp_path, [{"type": "websocket.receive", "text": json.dumps({"type": "set_language", "language": "te-IN"})}])
    assert session.browser.language_preference == "te-IN"
    assert "Answer in Telugu until the user changes" in conn.updates[0]["instructions"]
    assert ws.sent == [{"type": "preferences", "language": "te-IN"}]


def test_dropdown_auto_restores_following_speech(tmp_path):
    session, _, conn = _language_harness(tmp_path, [
        {"type": "websocket.receive", "text": json.dumps({"type": "set_language", "language": "hi-IN"})},
        {"type": "websocket.receive", "text": json.dumps({"type": "set_language", "language": "auto"})},
    ])
    assert session.browser.language_preference == "auto"
    assert "Answer in the language" in conn.updates[-1]["instructions"]


def test_dropdown_rejects_unknown_language_and_garbage(tmp_path):
    session, ws, conn = _language_harness(tmp_path, [
        {"type": "websocket.receive", "text": json.dumps({"type": "set_language", "language": "xx-XX"})},
        {"type": "websocket.receive", "text": "not json"},
    ])
    assert session.browser.language_preference == "auto"
    assert conn.updates == []
    assert ws.sent[0]["type"] == "error"


def test_fixed_language_overrides_spoken_language_and_skips_transcript_wait(tmp_path, monkeypatch):
    from grounded_answer import GroundedAnswer
    session, calls = _grounded_session(tmp_path, monkeypatch, GroundedAnswer("x", True, "te"))
    session.browser.language_preference = "hi-IN"
    session.current_question = ""  # the transcript has not arrived and never will in this test
    out = asyncio.run(asyncio.wait_for(live._run_tool(
        session, "search_documents", {"query": "study source", "spoken_language": "te-IN"}, ""), 1.0))
    assert calls[0][2] == "hi"  # Hindi was chosen, whatever the model heard
    assert "Hindi" in json.loads(out)["instruction"]


def test_realtime_model_is_the_21_mini():
    import config
    assert config.REALTIME_MODEL_ID == "gpt-realtime-2.1-mini"


def test_one_word_transcript_falls_back_to_search_query(tmp_path, monkeypatch):
    from grounded_answer import GroundedAnswer
    session, calls = _grounded_session(tmp_path, monkeypatch, GroundedAnswer("x", True, "en"))
    asyncio.run(live._run_tool(session, "search_documents", {"query": "objection handling"}, "kindly."))
    assert calls[0][0] == "objection handling"


def _preview_conn(fail=False):
    class Buffer:
        def __init__(self):
            self.audio, self.commits = [], 0

        async def append(self, *, audio):
            if fail:
                raise RuntimeError("preview socket closed")
            self.audio.append(audio)

        async def commit(self):
            if fail:
                raise RuntimeError("preview socket closed")
            self.commits += 1

    return SimpleNamespace(input_audio_buffer=Buffer())


def test_microphone_audio_is_also_sent_to_the_live_preview(tmp_path):
    session = _session(tmp_path)
    session.preview = _preview_conn()

    class FakeWebSocket:
        def __init__(self):
            self.messages = [{"type": "websocket.receive", "bytes": b"\x01\x00\x02\x00"},
                             {"type": "websocket.disconnect"}]

        async def receive(self):
            return self.messages.pop(0)

    main = SimpleNamespace(input_audio_buffer=SimpleNamespace(append=lambda *, audio: asyncio.sleep(0)))
    asyncio.run(live._from_browser(FakeWebSocket(), main, session))
    assert session.preview.input_audio_buffer.audio == [base64.b64encode(b"\x01\x00\x02\x00").decode()]


def test_preview_failure_is_swallowed_and_disables_the_preview(tmp_path):
    session = _session(tmp_path)
    session.preview = _preview_conn(fail=True)
    asyncio.run(live._preview_append(session, "AAAA"))
    assert session.preview is None
    asyncio.run(live._preview_commit(session))  # no preview: a no-op, not an error


def test_speech_stopped_commits_the_preview_segment(tmp_path):
    session = _session(tmp_path)
    session.preview = _preview_conn()

    class FakeWebSocket:
        async def send_json(self, event):
            pass

    class FakeConnection:
        def __init__(self):
            self.events = iter([SimpleNamespace(type="input_audio_buffer.speech_stopped")])

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.events)
            except StopIteration:
                raise StopAsyncIteration

    asyncio.run(live._from_openai(FakeWebSocket(), FakeConnection(), session))
    assert session.preview.input_audio_buffer.commits == 1


def test_preview_relays_accumulating_live_text_and_starts_fresh_per_item(tmp_path):
    session = _session(tmp_path)
    sent = []

    class FakeWebSocket:
        async def send_json(self, event):
            sent.append(event)

    def delta(item, text):
        return SimpleNamespace(type="conversation.item.input_audio_transcription.delta", item_id=item, delta=text)

    class FakePreview:
        def __init__(self):
            self.events = iter([delta("a", "Hello"), delta("a", " there"), delta("b", "Next")])

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.events)
            except StopIteration:
                raise StopAsyncIteration

    asyncio.run(live._from_preview(FakeWebSocket(), session, FakePreview()))
    assert [e["text"] for e in sent] == ["Hello", "Hello there", "Next"]


def test_preview_error_event_turns_the_preview_off(tmp_path):
    session = _session(tmp_path)
    session.preview = object()

    class FakePreview:
        def __aiter__(self):
            return self

        async def __anext__(self):
            if getattr(self, "done", False):
                raise StopAsyncIteration
            self.done = True
            return SimpleNamespace(type="error", error="boom")

    asyncio.run(live._from_preview(SimpleNamespace(send_json=None), session, FakePreview()))
    assert session.preview is None


def test_preamble_before_tool_does_not_flag_language_and_answer_audio_is_timed(tmp_path, monkeypatch):
    session = _session(tmp_path)
    session.speech_stopped_at = 0.0
    session.browser.last_spoken_language = "te-IN"
    session.browser.language_preference = "te-IN"
    monkeypatch.setattr(live, "GROUNDED_VOICE", False)
    monkeypatch.setattr(live, "retrieve_documents", lambda *a: __import__("module_v2v_rag").RetrievalResult(
        "ctx", passages=[{"doc": "lesson", "page": 1, "type": "text", "content": "Study source"}]))
    session.browser.engine = SimpleNamespace(faiss_index=object(), metadata_db=object(), client=object())

    class FakeWebSocket:
        async def send_json(self, event):
            pass

        async def send_bytes(self, data):
            pass

    class FakeConnection:
        def __init__(self):
            self.events = iter([
                SimpleNamespace(type="response.created", response=SimpleNamespace(id="r1")),
                # Spoken English filler (Latin script) before the tool call while the preference is Telugu.
                SimpleNamespace(type="response.output_audio_transcript.done", item_id="a0", transcript="Let me check."),
                SimpleNamespace(type="response.function_call_arguments.done", response_id="r1", call_id="c1",
                                name="search_documents", arguments='{"query":"study source"}'),
                SimpleNamespace(type="response.done", response=SimpleNamespace(id="r1", status="completed")),
                SimpleNamespace(type="response.created", response=SimpleNamespace(id="r2")),
                SimpleNamespace(type="response.output_audio.delta", delta="AAAA"),
                SimpleNamespace(type="response.output_audio_transcript.done", item_id="a1", transcript="ఇది సమాధానం."),
                SimpleNamespace(type="response.done", response=SimpleNamespace(id="r2", status="completed")),
            ])
            self.conversation = SimpleNamespace(item=SimpleNamespace(create=self._noop))
            self.response = SimpleNamespace(create=self._noop)
            self.session = SimpleNamespace(update=self._noop)

        async def _noop(self, **kwargs):
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.events)
            except StopIteration:
                raise StopAsyncIteration

    asyncio.run(live._from_openai(FakeWebSocket(), FakeConnection(), session))
    metrics = session.metrics[0]
    assert "answer_language_mismatch" not in metrics  # the English preamble is not the answer
    assert metrics["time_to_answer_audio_ms"] >= 0


def _error_events(code):
    class FakeWebSocket:
        async def send_json(self, event):
            pass

    class FakeConnection:
        def __init__(self):
            self.events = iter([
                SimpleNamespace(type="error", error=SimpleNamespace(code=code, message="x")),
                SimpleNamespace(type="input_audio_buffer.speech_stopped"),
            ])

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.events)
            except StopIteration:
                raise StopAsyncIteration

    return FakeWebSocket(), FakeConnection()


def test_lost_race_error_does_not_end_the_call(tmp_path):
    session = _session(tmp_path)
    ws, conn = _error_events("conversation_already_has_active_response")
    asyncio.run(live._from_openai(ws, conn, session))  # reaches the event after the error
    assert session.turn_metrics["ignored_errors"] == ["conversation_already_has_active_response"]
    assert session.speech_stopped_at is not None


def test_other_realtime_errors_still_end_the_call(tmp_path):
    session = _session(tmp_path)
    ws, conn = _error_events("invalid_api_key")
    with pytest.raises(RuntimeError):
        asyncio.run(live._from_openai(ws, conn, session))


def test_search_streams_real_progress_and_closes_the_card(tmp_path, monkeypatch):
    from grounded_answer import GroundedAnswer
    from module_v2v_rag import RetrievalResult
    session = _session(tmp_path)
    session.browser.engine = SimpleNamespace(faiss_index=object(), metadata_db=object(), client=object())
    passages = [{"doc": "lesson", "page": 4, "type": "text", "content": "Study source"}]
    events = []
    session.emit = events.append

    def fake_retrieve(query, index, meta, client, top_k, expand, progress, *extra):
        progress("Searching 185 passages")
        progress("Found 20 candidate passages on pages 4, 9")
        return RetrievalResult("ctx", passages=passages)

    def fake_answer(client, question, numbered, language, check, progress, *extra):
        progress("Reading 1 passages and drafting an answer")
        return GroundedAnswer("Done.", False, "en", claims=[{"text": "Done.", "source_id": "S1", "quote": "Study"}])

    monkeypatch.setattr(live, "retrieve_documents", fake_retrieve)
    monkeypatch.setattr(live, "answer_question", fake_answer)
    monkeypatch.setattr(live, "GROUNDED_VOICE", True)
    asyncio.run(live._run_tool(session, "search_documents", {"query": "study source"}, "What is the study source?"))
    status = [e for e in events if e["type"] == "status"]
    assert [e["state"] for e in status] == ["start", "step", "step", "step", "end"]
    assert status[0]["query"] == "study source"
    assert status[-1]["summary"].startswith("Checked 1 passages · 1 verified claim")
    assert [e["type"] for e in events].count("evidence") == 1  # one card set per verified answer
    assert session.status_open is False


def test_search_query_is_respelled_from_document_vocabulary(tmp_path, monkeypatch):
    from module_v2v_rag import RetrievalResult
    from vocabulary import Corrector
    session = _session(tmp_path)
    session.browser.engine = SimpleNamespace(faiss_index=object(), metadata_db=object(), client=object())
    session.browser.corrector = Corrector(["Brandel"])
    seen = []
    monkeypatch.setattr(live, "retrieve_documents",
                        lambda query, *a: seen.append(query) or RetrievalResult("ctx", passages=[]))
    monkeypatch.setattr(live, "GROUNDED_VOICE", False)
    asyncio.run(live._run_tool(session, "search_documents", {"query": "Brendle what does Brendle do"}, ""))
    assert seen == ["Brandel what does Brandel do"]


def test_final_user_transcript_is_respelled_and_reports_corrections(tmp_path):
    from vocabulary import Corrector
    session = _session(tmp_path)
    session.browser.corrector = Corrector(["Brandel"])
    sent = []

    class FakeWebSocket:
        async def send_json(self, event):
            sent.append(event)

    class FakeConnection:
        def __init__(self):
            self.events = iter([SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed", item_id="u1",
                transcript="What does Brendel actually do?")])

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.events)
            except StopIteration:
                raise StopAsyncIteration

    asyncio.run(live._from_openai(FakeWebSocket(), FakeConnection(), session))
    final = sent[0]
    assert final["text"] == "What does Brandel actually do?" and final["heard"] == "What does Brendel actually do?"
    assert final["corrections"] == [{"from": "Brendel", "to": "Brandel"}]
    assert session.current_question == "What does Brandel actually do?"


def test_vocabulary_echo_never_replaces_the_real_question(tmp_path):
    session = _session(tmp_path)
    session.browser.vocabulary = ["Brandel", "Sales", "Playbook", "CRM", "Gong", "Slack"]
    sent = []

    class FakeWebSocket:
        async def send_json(self, event):
            sent.append(event)

    class FakeConnection:
        def __init__(self):
            done = "conversation.item.input_audio_transcription.completed"
            self.events = iter([
                SimpleNamespace(type=done, item_id="u1", transcript="Explain the Brandel positioning statement."),
                SimpleNamespace(type=done, item_id="u2", transcript="Brandel, Sales, Playbook, CRM, Gong, Slack"),
            ])

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.events)
            except StopIteration:
                raise StopAsyncIteration

    asyncio.run(live._from_openai(FakeWebSocket(), FakeConnection(), session))
    assert session.current_question == "Explain the Brandel positioning statement."
    assert sent[0]["final"] is True and sent[1] == {"type": "transcript_discard", "item_id": "u2"}


def test_status_card_is_closed_when_a_tool_ends_without_reporting(tmp_path, monkeypatch):
    from module_v2v_rag import RetrievalResult
    session = _session(tmp_path)
    session.browser.engine = SimpleNamespace(faiss_index=object(), metadata_db=object(), client=object())
    monkeypatch.setattr(live, "GROUNDED_VOICE", False)  # ungrounded path never reports an end itself
    monkeypatch.setattr(live, "retrieve_documents", lambda *a: RetrievalResult(
        "ctx", passages=[{"doc": "lesson", "page": 1, "type": "text", "content": "Study source"}]))
    events = []
    session.emit = events.append

    class FakeWebSocket:
        async def send_json(self, event):
            pass

    class FakeConnection:
        def __init__(self):
            self.events = iter([
                SimpleNamespace(type="response.created", response=SimpleNamespace(id="r1")),
                SimpleNamespace(type="response.function_call_arguments.done", response_id="r1", call_id="c1",
                                name="search_documents", arguments='{"query":"study source"}'),
            ])
            self.conversation = SimpleNamespace(item=SimpleNamespace(create=self._noop))
            self.session = SimpleNamespace(update=self._noop)

        async def _noop(self, **kwargs):
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.events)
            except StopIteration:
                raise StopAsyncIteration

    asyncio.run(live._from_openai(FakeWebSocket(), FakeConnection(), session))
    assert [e["state"] for e in events] == ["start", "end"]
    assert events[-1]["summary"] == "Search finished" and session.status_open is False


def test_instructions_ask_for_a_spoken_preamble_and_carry_document_terms():
    from voice_session import _browser_instructions
    text = _browser_instructions("auto", "en-US", [], None, False, ["Brandel", "RevOps"])
    assert "Let me check what your materials actually say about" in text
    assert "Brandel, RevOps" in text


def test_transcription_prompt_carries_document_terms():
    from voice_session import _browser_session_payload
    session = SimpleNamespace(last_spoken_language="en-US", turns=[], quiz=None, language_confident=False,
                              vocabulary=["Brandel", "RevOps"])
    payload = _browser_session_payload(session, "auto", "marin")
    prompt = payload["audio"]["input"]["transcription"]["prompt"]
    # Bare list only: sentence-style prompts made the model answer the question instead of transcribing it.
    assert prompt == "Brandel, RevOps" and "." not in prompt and "ranscribe" not in prompt


def test_preview_payload_has_no_prompt_because_the_streaming_model_rejects_it():
    assert "prompt" not in live.preview_session_payload()["audio"]["input"]["transcription"]


def test_transcription_prompt_without_documents_keeps_only_the_language_hint():
    from voice_session import _browser_session_payload
    session = SimpleNamespace(last_spoken_language="en-US", turns=[], quiz=None, language_confident=False,
                              vocabulary=[])
    prompt = _browser_session_payload(session, "auto", "marin")["audio"]["input"]["transcription"]["prompt"]
    assert "Telugu" in prompt and "Brandel" not in prompt


def test_audio_chunks_are_tagged_with_their_response_item_once(tmp_path):
    session = _session(tmp_path)
    session.speech_stopped_at = 0.0
    sent_json, sent_bytes = [], []

    class FakeWebSocket:
        async def send_json(self, event):
            sent_json.append(event)

        async def send_bytes(self, data):
            sent_bytes.append(data)

    class FakeConnection:
        def __init__(self):
            self.events = iter([
                SimpleNamespace(type="response.output_audio.delta", item_id="a1", delta="AAAA"),
                SimpleNamespace(type="response.output_audio.delta", item_id="a1", delta="AAAA"),
                SimpleNamespace(type="response.output_audio.delta", item_id="a2", delta="AAAA"),
            ])

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.events)
            except StopIteration:
                raise StopAsyncIteration

    asyncio.run(live._from_openai(FakeWebSocket(), FakeConnection(), session))
    assert [e for e in sent_json if e["type"] == "audio_item"] == [
        {"type": "audio_item", "item_id": "a1"}, {"type": "audio_item", "item_id": "a2"}]
    assert len(sent_bytes) == 3


def test_short_section_and_where_label_read_naturally():
    assert live.short_section("Part D › SECTION 17 · Objection handling › Cards") == \
        "SECTION 17 · Objection handling › Cards"
    assert live.where_label(["Part D › SECTION 17 · Objection handling"], [47, 47]) == \
        "Part D › SECTION 17 · Objection handling (page 47)"
    assert live.where_label([], [4, 5, 6]) == "pages 4–6"
    assert live.where_label([], []) == ""


def test_teaching_instructions_differ_by_depth_and_carry_only_verified_text():
    flow = [{"role": "answer", "source": "S1", "claim": "Acknowledge first."},
            {"role": "reason", "source": "S1", "claim": "So the buyer feels heard."}]
    brief = live.teaching_instructions("English", "brief", flow, "Section 17 (page 47)", "")
    detailed = live.teaching_instructions("English", "detailed", flow, "Section 17 (page 47)", "It is about listening.")
    assert "two to four short spoken sentences" in brief and "In plain words" not in brief
    assert "never invent a reason" in detailed and "It is about listening." in detailed and "quiz" in detailed
    for text in (brief, detailed):
        assert "[reason] So the buyer feels heard." in text and "Section 17 (page 47)" in text


def test_depth_argument_selects_the_lesson_style(tmp_path, monkeypatch):
    from grounded_answer import GroundedAnswer
    answer = GroundedAnswer("x", False, "en", claims=[{"text": "A fact.", "source_id": "S1", "quote": "Study",
                                                        "role": "answer", "page": 1, "section": ""}],
                            explanation="Plainly: a fact.", sections=[])
    session, _ = _grounded_session(tmp_path, monkeypatch, answer)
    asyncio.run(live._run_tool(session, "search_documents", {"query": "study source", "depth": "detailed"},
                               "Explain the study source step by step"))
    assert "never invent a reason" in session.followup_instructions
    session2, _ = _grounded_session(tmp_path, monkeypatch, answer)
    asyncio.run(live._run_tool(session2, "search_documents", {"query": "study source"}, "What is the source?"))
    assert "two to four short spoken sentences" in session2.followup_instructions


def test_source_labels_show_where_in_the_document(tmp_path):
    session = _session(tmp_path)
    session.browser.documents["lesson"] = tmp_path / "lesson.pdf"
    session.browser.display_names["lesson"] = "Lesson"
    session.turn_sources = [{"id": "S1", "doc": "lesson", "page": 47, "content": "x",
                             "section": "Part D › SECTION 17 · Objection handling"}]
    sent = []

    class FakeWebSocket:
        async def send_json(self, event):
            sent.append(event)

    asyncio.run(live._send_sources(FakeWebSocket(), session))
    assert sent[0]["sources"][0]["label"] == "Lesson · Part D › SECTION 17 · Objection handling (p. 47)"


def test_list_questions_are_drafted_in_full_but_spoken_briefly(tmp_path, monkeypatch):
    from grounded_answer import GroundedAnswer
    session, _ = _grounded_session(tmp_path, monkeypatch, GroundedAnswer("x", True, "en"))
    seen = []
    monkeypatch.setattr(live, "answer_question",
                        lambda client, q, numbered, language, check, progress, depth: seen.append(depth)
                        or GroundedAnswer("x", True, "en"))
    asyncio.run(live._run_tool(session, "search_documents", {"query": "objections", "covers_all": True}, "all objections"))
    asyncio.run(live._run_tool(session, "search_documents", {"query": "objections"}, "an objection"))
    assert seen == ["detailed", "brief"]


def test_brief_reply_is_given_only_the_leading_claims(tmp_path, monkeypatch):
    from grounded_answer import GroundedAnswer
    claims = [{"text": f"Fact {i}.", "source_id": "S1", "quote": "Study", "role": "answer", "page": 1, "section": ""}
              for i in range(10)]
    answer = GroundedAnswer("x", False, "en", claims=claims)
    session, _ = _grounded_session(tmp_path, monkeypatch, answer)
    brief = json.loads(asyncio.run(live._run_tool(session, "search_documents", {"query": "study source"}, "What is it?")))
    assert len(brief["claims"]) == live.BRIEF_CLAIMS and brief["more_available"] is True
    assert "Fact 9." not in session.followup_instructions
    assert "never say how many items there are" in session.followup_instructions
    session2, _ = _grounded_session(tmp_path, monkeypatch, answer)
    full = json.loads(asyncio.run(live._run_tool(
        session2, "search_documents", {"query": "study source", "depth": "detailed"}, "Explain it step by step")))
    assert len(full["claims"]) == 10 and full["more_available"] is False


def test_evidence_cards_carry_role_quote_location_and_a_highlight_link(tmp_path):
    import pymupdf
    pdf = tmp_path / "evidence_source.pdf"  # not "lesson.pdf": the session helper writes its own file there
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(50, 80, 560, 200), "Handle objections in three moves. Never argue with a buyer.",
                        fontsize=11, fontname="helv")
    doc.save(str(pdf))
    session = _session(tmp_path)
    session.browser.documents["lesson"] = pdf
    session.browser.display_names["lesson"] = "Lesson"
    by_id = {"S1": {"doc": "lesson", "page": 1, "content": "x", "section": "Part D › SECTION 17 · Objections"}}
    claims = [{"text": "Never argue.", "source_id": "S1", "quote": "Never argue with a buyer.", "role": "caution"},
              {"text": "Not on the page.", "source_id": "S1", "quote": "this quote is not on the page", "role": "step"}]
    cards = live.build_evidence_cards(session, claims, by_id)
    assert [c["role"] for c in cards] == ["caution", "step"]
    assert cards[0]["located"] is True and cards[1]["located"] is False
    assert cards[0]["label"] == "Lesson · Part D › SECTION 17 · Objections (p. 1)"
    assert cards[0]["url"] == f"/api/sessions/{session.id}/evidence/{cards[0]['id']}"
    assert session.evidence[cards[0]["id"]]["quote"] == "Never argue with a buyer."
    png = live.evidence_page.__wrapped__(session.id, cards[0]["id"]) if hasattr(live.evidence_page, "__wrapped__") else None
    assert png is None or png.body[:4] == b"\x89PNG"


def test_new_question_clears_stale_evidence(tmp_path):
    session = _session(tmp_path)
    sent = []

    class FakeWebSocket:
        async def send_json(self, event):
            sent.append(event)

    class FakeConnection:
        def __init__(self):
            self.events = iter([SimpleNamespace(type="input_audio_buffer.speech_started")])

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.events)
            except StopIteration:
                raise StopAsyncIteration

    asyncio.run(live._from_openai(FakeWebSocket(), FakeConnection(), session))
    assert {"type": "evidence", "cards": [], "reading": ""} in sent


def test_unreadable_page_never_breaks_evidence_cards(tmp_path):
    session = _session(tmp_path)  # its PDF has one page; the passage claims page 99
    by_id = {"S1": {"doc": "lesson", "page": 99, "content": "x", "section": ""}}
    cards = live.build_evidence_cards(
        session, [{"text": "A fact.", "source_id": "S1", "quote": "Study source", "role": "answer"}], by_id)
    assert len(cards) == 1 and cards[0]["located"] is None


def test_recent_turns_reach_retrieval_so_follow_ups_can_be_rewritten(tmp_path, monkeypatch):
    from module_v2v_rag import RetrievalResult
    session = _session(tmp_path)
    session.browser.engine = SimpleNamespace(faiss_index=object(), metadata_db=object(), client=object())
    session.browser.turns = [(f"Question {i}", f"Answer {i}") for i in range(5)]
    seen = {}

    def fake_retrieve(query, index, meta, client, top_k, expand, progress, covers_all, hint, history, hybrid, rewrite):
        seen.update(history=history, hybrid=hybrid, rewrite=rewrite, hint=hint)
        return RetrievalResult("ctx", passages=[])

    monkeypatch.setattr(live, "retrieve_documents", fake_retrieve)
    monkeypatch.setattr(live, "GROUNDED_VOICE", False)
    asyncio.run(live._run_tool(session, "search_documents", {"query": "his parents"}, "What are his parents called?"))
    assert seen["history"] == [("Question 2", "Answer 2"), ("Question 3", "Answer 3"), ("Question 4", "Answer 4")]
    assert seen["hybrid"] is True and seen["rewrite"] is True
