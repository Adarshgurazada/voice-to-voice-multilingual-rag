"""Always-on browser voice RAG with one Realtime session per connected learner."""

import asyncio
import base64
import contextlib
import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import pymupdf
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from config import REALTIME_MODEL_ID
from citation_audit import audit_citations
from glossary_mcp import lookup_glossary
from voice_session import (
    AUTO_LANGUAGE,
    BrowserSession,
    _apply_preference_update,
    _browser_instructions,
    _browser_session_payload,
    _tool_args,
)
from module_v2v_rag import (
    RAG_TOP_K,
    SUPPORTED_LANGUAGES,
    VoiceRAGMultimodalFAISS,
    _event_response_id,
    _event_response_status,
    _extract_query,
    retrieve_documents,
)
from study_quiz import generate_quiz, grade_quiz


ROOT = Path(__file__).resolve().parent
DATA = ROOT / ".local_data"
WEB = ROOT / "live_web"
MAX_FILES = 5
MAX_BYTES = 25 * 1024 * 1024

app = FastAPI(title="Voice Study Companion")
app.mount("/assets", StaticFiles(directory=WEB), name="assets")


@dataclass
class LiveSession:
    id: str
    browser: BrowserSession
    paths: list[Path]
    status: str = "indexing"
    progress: int = 0
    detail: str = "Preparing documents"
    connected: bool = False
    audio_emitted: bool = False
    voice_reconnect: bool = False
    current_question: str = ""
    current_answer: str = ""
    current_answer_item_id: str = ""
    turn_sources: list[dict] = field(default_factory=list)
    quiz_started_this_turn: bool = False
    speech_stopped_at: float | None = None
    response_active: bool = False
    turn_metrics: dict = field(default_factory=dict)
    metrics: list[dict] = field(default_factory=list)


sessions: dict[str, LiveSession] = {}


def live_session_payload(session: BrowserSession) -> dict:
    payload = _browser_session_payload(
        session, session.language_preference, session.selected_voice,
    )
    payload["audio"]["input"]["turn_detection"] = {
        "type": "server_vad",
        "threshold": 0.5,
        "prefix_padding_ms": 300,
        "silence_duration_ms": 600,
        "create_response": True,
        "interrupt_response": True,
    }
    return payload


def _session_or_404(session_id: str) -> LiveSession:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "Session not found. Upload documents again.")
    return session


async def _save_pdf(upload: UploadFile) -> tuple[Path, str]:
    name = Path(upload.filename or "").name
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    content = await upload.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise HTTPException(400, f"{name} exceeds the 25 MB limit.")
    if not content.startswith(b"%PDF-"):
        raise HTTPException(400, f"{name} is not a valid PDF.")
    digest = hashlib.sha256(content).hexdigest()[:16]
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._-")[:70] or "document"
    destination = DATA / "documents" / f"{digest}_{safe_stem}.pdf"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.write_bytes(content)
    return destination, name


async def _index(session: LiveSession) -> None:
    path_positions = {str(path): i for i, path in enumerate(session.paths)}

    def on_page(path: str, page: int, total: int) -> None:
        position = path_positions[path]
        session.progress = min(99, round(5 + 90 * (position + page / total) / len(session.paths)))
        session.detail = f"{session.browser.display_names[Path(path).stem]}: page {page}/{total}"

    try:
        await session.browser.engine.setup_rag_database(progress_callback=on_page)
        engine = session.browser.engine
        if not engine.is_active or engine.faiss_index is None or engine.metadata_db is None:
            raise RuntimeError("No searchable content was found in the uploaded PDFs.")
        session.status = "ready"
        session.progress = 100
        session.detail = "Documents ready"
    except Exception as exc:
        session.status = "error"
        session.detail = str(exc)


@app.get("/")
def home():
    return FileResponse(WEB / "index.html")


@app.post("/api/sessions")
async def create_session(files: list[UploadFile] = File(...)):
    if not 1 <= len(files) <= MAX_FILES:
        raise HTTPException(400, "Choose one to five PDFs.")
    stored = []
    for upload in files:
        stored.append(await _save_pdf(upload))
    paths = [path for path, _ in stored]
    if len({path.stem for path in paths}) != len(paths):
        raise HTTPException(400, "Choose distinct PDFs.")
    engine = VoiceRAGMultimodalFAISS(
        pdf_paths=[str(path) for path in paths],
        cache_dir=DATA / "indexes",
        image_save_dir=DATA / "images",
    )
    documents = {path.stem: path for path, _ in stored}
    display_names = {path.stem: name for path, name in stored}
    session_id = uuid.uuid4().hex
    session = LiveSession(
        session_id, BrowserSession(engine, documents, display_names), paths,
    )
    sessions[session_id] = session
    asyncio.create_task(_index(session))
    return {"session_id": session_id}


@app.get("/api/sessions/{session_id}")
def session_status(session_id: str):
    session = _session_or_404(session_id)
    return {
        "status": session.status,
        "progress": session.progress,
        "detail": session.detail,
        "documents": list(session.browser.display_names.values()),
    }


@app.get("/api/sessions/{session_id}/metrics")
def session_metrics(session_id: str):
    session = _session_or_404(session_id)
    return {"turns": session.metrics, "transport": "websocket", "clock": "server_monotonic"}


@app.get("/api/sessions/{session_id}/pages/{doc}/{page}")
def source_page(session_id: str, doc: str, page: int):
    session = _session_or_404(session_id)
    path = session.browser.documents.get(doc)
    if path is None:
        raise HTTPException(404, "Document not found.")
    with pymupdf.open(path) as document:
        if not 1 <= page <= len(document):
            raise HTTPException(404, "Page not found.")
        image = document.load_page(page - 1).get_pixmap(matrix=pymupdf.Matrix(1.3, 1.3))
        return Response(image.tobytes("png"), media_type="image/png")


async def _send_sources(ws: WebSocket, session: LiveSession) -> None:
    sources = []
    for source in session.turn_sources:
        doc, page = source["doc"], source["page"]
        if doc not in session.browser.documents:
            continue
        sources.append({
            "id": source.get("id"),
            "label": f"{session.browser.display_names[doc]} - page {page}",
            "url": f"/api/sessions/{session.id}/pages/{quote(doc, safe='')}/{page}",
            "excerpt": source.get("content", "")[:500],
        })
    await ws.send_json({"type": "sources", "sources": sources})


async def _run_tool(session: LiveSession, name: str, args: dict, transcript: str) -> str:
    browser = session.browser
    engine = browser.engine
    claimed_language = args.get("spoken_language")
    if claimed_language in SUPPORTED_LANGUAGES:
        browser.last_spoken_language = claimed_language
    answer_language = (
        browser.language_preference if browser.language_preference != AUTO_LANGUAGE
        else browser.last_spoken_language
    )
    if name == "search_documents":
        query = _extract_query(json.dumps(args), transcript)
        if not query:
            return json.dumps({"status": "no_question"})
        retrieval = await asyncio.to_thread(
            retrieve_documents,
            query, engine.faiss_index, engine.metadata_db, engine.client, RAG_TOP_K,
        )
        if retrieval.error:
            return f"Document search failed: {retrieval.error}"
        if not retrieval.passages:
            return retrieval.context
        context = []
        for passage in retrieval.passages:
            source_id = f"S{len(session.turn_sources) + 1}"
            session.turn_sources.append({**passage, "id": source_id})
            context.append(
                f"[{source_id}] {passage['doc']}, page {passage['page']}, "
                f"{passage['type']} passage:\n{passage['content']}"
            )
        return "\n---\n".join(context)
    if name == "set_preferences":
        language, voice, output = _apply_preference_update(
            args, browser.language_preference, browser.selected_voice,
        )
        if language != browser.language_preference or voice != browser.selected_voice:
            if voice != browser.selected_voice and session.audio_emitted:
                session.voice_reconnect = True
            browser.language_preference = language
            browser.selected_voice = voice
        return output
    if name == "lookup_glossary":
        term = args.get("term")
        if not isinstance(term, str) or not term.strip():
            return json.dumps({"status": "invalid_term"})
        return await lookup_glossary(term, answer_language)
    if name == "start_quiz":
        topic = args.get("topic")
        if not isinstance(topic, str):
            topic = "important ideas in the document"
        browser.quiz = await asyncio.to_thread(generate_quiz, engine, topic, answer_language)
        session.quiz_started_this_turn = True
        session.turn_sources.append(browser.quiz.source)
        return json.dumps({
            "status": "ready", "question": browser.quiz.question,
            "source": browser.quiz.source,
            "next_step": "Ask this question aloud and wait for the learner's answer.",
        })
    if name == "grade_quiz":
        if browser.quiz is None:
            return json.dumps({"status": "no_pending_quiz"})
        if session.quiz_started_this_turn:
            return json.dumps({"status": "wait_for_learner_answer"})
        learner_answer = transcript or args.get("answer", "")
        if not isinstance(learner_answer, str) or not learner_answer.strip():
            return json.dumps({"status": "no_spoken_answer"})
        assessment = await asyncio.to_thread(
            grade_quiz, engine.client, browser.quiz, learner_answer, answer_language,
        )
        session.turn_sources.append(browser.quiz.source)
        result = json.dumps({
            "status": "graded", "correct": assessment.correct,
            "feedback": assessment.feedback, "expected_answer": browser.quiz.answer,
            "source": browser.quiz.source,
        })
        browser.quiz = None
        return result
    if name == "skip_quiz":
        if browser.quiz is None:
            return json.dumps({"status": "no_pending_quiz"})
        if session.quiz_started_this_turn:
            return json.dumps({"status": "wait_for_learner_answer"})
        session.turn_sources.append(browser.quiz.source)
        result = json.dumps({
            "status": "skipped", "answer": browser.quiz.answer,
            "source": browser.quiz.source,
        })
        browser.quiz = None
        return result
    raise RuntimeError(f"Unexpected tool call: {name}")


async def _from_browser(ws: WebSocket, conn) -> None:
    while True:
        message = await ws.receive()
        if message["type"] == "websocket.disconnect":
            return
        audio = message.get("bytes")
        if audio is None:
            continue
        if len(audio) < 2 or len(audio) % 2 or len(audio) > 48000:
            await ws.send_json({"type": "error", "message": "Invalid microphone audio frame."})
            continue
        await conn.input_audio_buffer.append(audio=base64.b64encode(audio).decode("ascii"))


async def _send_citation_audit(ws: WebSocket, session: LiveSession, answer: str,
                               item_id: str, passages: list[dict]) -> None:
    try:
        spans = await asyncio.to_thread(
            audit_citations, session.browser.engine.client, answer, passages,
        )
        await ws.send_json({"type": "citations", "item_id": item_id, "spans": spans})
    except Exception as exc:
        print(f"Citation audit unavailable: {exc}")
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "citations", "item_id": item_id, "spans": [],
                                "status": "unavailable"})


async def _from_openai(ws: WebSocket, conn, session: LiveSession) -> None:
    calls_by_response: dict[str, set[str]] = {}
    citation_tasks: set[asyncio.Task] = set()
    active_response_id = ""
    try:
      async for event in conn:
        kind = getattr(event, "type", "")
        if kind == "error":
            raise RuntimeError(str(getattr(event, "error", event)))
        if kind == "response.created":
            active_response_id = _event_response_id(event) or ""
            session.response_active = True
        elif kind == "input_audio_buffer.speech_started":
            if session.response_active:
                session.turn_metrics["interrupted_previous_response"] = True
            session.turn_sources.clear()
            session.quiz_started_this_turn = False
            session.turn_metrics = {"interrupted_previous_response": session.response_active}
            session.speech_stopped_at = None
            await ws.send_json({"type": "speech_started"})
            await _send_sources(ws, session)
        elif kind == "input_audio_buffer.speech_stopped":
            session.speech_stopped_at = time.perf_counter()
            await ws.send_json({"type": "speech_stopped"})
        elif kind == "conversation.item.input_audio_transcription.delta":
            await ws.send_json({
                "type": "transcript", "speaker": "user", "item_id": getattr(event, "item_id", ""),
                "delta": getattr(event, "delta", ""), "final": False,
            })
        elif kind == "conversation.item.input_audio_transcription.completed":
            session.current_question = getattr(event, "transcript", "") or ""
            await ws.send_json({
                "type": "transcript", "speaker": "user", "item_id": getattr(event, "item_id", ""),
                "text": session.current_question, "final": True,
            })
        elif kind == "response.function_call_arguments.done":
            response_id = getattr(event, "response_id", None) or active_response_id
            call_id = getattr(event, "call_id", None)
            if not call_id:
                raise RuntimeError("Tool call had no call ID.")
            calls = calls_by_response.setdefault(response_id, set())
            if call_id in calls:
                continue
            output = await _run_tool(
                session, getattr(event, "name", ""),
                _tool_args(getattr(event, "arguments", None)), session.current_question,
            )
            await conn.conversation.item.create(item={
                "type": "function_call_output", "call_id": call_id, "output": output,
            })
            calls.add(call_id)
            await _send_sources(ws, session)
            browser = session.browser
            await conn.session.update(session={
                "type": "realtime",
                "instructions": _browser_instructions(
                    browser.language_preference, browser.last_spoken_language,
                    browser.turns, browser.quiz,
                ),
            })
        elif kind == "response.output_audio.delta":
            session.audio_emitted = True
            if "time_to_first_audio_ms" not in session.turn_metrics and session.speech_stopped_at is not None:
                session.turn_metrics["time_to_first_audio_ms"] = round(
                    (time.perf_counter() - session.speech_stopped_at) * 1000, 1,
                )
            await ws.send_bytes(base64.b64decode(getattr(event, "delta", "")))
        elif kind == "response.output_audio_transcript.delta":
            delta = getattr(event, "delta", "") or ""
            session.current_answer += delta
            await ws.send_json({
                "type": "transcript", "speaker": "assistant", "item_id": getattr(event, "item_id", ""),
                "delta": delta, "final": False,
            })
        elif kind == "response.output_audio_transcript.done":
            session.current_answer = getattr(event, "transcript", "") or session.current_answer
            session.current_answer_item_id = getattr(event, "item_id", "") or ""
            await ws.send_json({
                "type": "transcript", "speaker": "assistant", "item_id": getattr(event, "item_id", ""),
                "text": session.current_answer, "final": True,
            })
        elif kind == "response.done":
            session.response_active = False
            response_id = _event_response_id(event) or active_response_id
            status = _event_response_status(event)
            calls = calls_by_response.pop(response_id, set())
            if status and status not in ("completed", "cancelled"):
                raise RuntimeError(f"Realtime response failed ({status}).")
            if status == "cancelled":
                session.turn_metrics["cancelled_response"] = True
                session.current_question = ""
                session.current_answer = ""
                session.quiz_started_this_turn = False
                await ws.send_json({"type": "turn_done"})
            elif status == "completed" and calls:
                await conn.response.create(response={"tool_choice": "none"})
            elif not calls:
                if session.current_answer and any(source.get("id") for source in session.turn_sources):
                    task = asyncio.create_task(_send_citation_audit(
                        ws, session, session.current_answer, session.current_answer_item_id,
                        list(session.turn_sources),
                    ))
                    citation_tasks.add(task)
                    task.add_done_callback(citation_tasks.discard)
                session.turn_metrics["transcription_missing"] = not bool(session.current_question)
                session.turn_metrics["answer_audio_missing"] = "time_to_first_audio_ms" not in session.turn_metrics
                session.metrics.append(dict(session.turn_metrics))
                await ws.send_json({"type": "metrics", "metrics": session.turn_metrics})
                if session.current_question and session.current_answer:
                    session.browser.turns.append((session.current_question, session.current_answer))
                session.current_question = ""
                session.current_answer = ""
                session.current_answer_item_id = ""
                session.quiz_started_this_turn = False
                await ws.send_json({"type": "turn_done"})
                if session.voice_reconnect:
                    session.voice_reconnect = False
                    await ws.send_json({"type": "reconnect"})
                    return
    finally:
        for task in citation_tasks:
            task.cancel()
        if citation_tasks:
            await asyncio.gather(*citation_tasks, return_exceptions=True)


@app.websocket("/ws/{session_id}")
async def live_voice(ws: WebSocket, session_id: str):
    session = sessions.get(session_id)
    if session is None or session.status != "ready" or session.connected:
        await ws.close(code=1008)
        return
    await ws.accept()
    session.connected = True
    session.audio_emitted = False
    try:
        async with session.browser.engine.async_client.realtime.connect(model=REALTIME_MODEL_ID) as conn:
            await conn.session.update(session=live_session_payload(session.browser))
            await ws.send_json({"type": "ready"})
            incoming = asyncio.create_task(_from_browser(ws, conn))
            outgoing = asyncio.create_task(_from_openai(ws, conn, session))
            done, pending = await asyncio.wait({incoming, outgoing}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            for task in pending:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            for task in done:
                task.result()
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    except Exception as exc:
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "error", "message": str(exc)})
    finally:
        session.connected = False
        with contextlib.suppress(Exception):
            await ws.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
