"""Always-on browser voice RAG with one Realtime session per connected learner."""

import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import os
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import pymupdf
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from config import (
    FOLLOWUP_REWRITE, GROUNDED_VOICE, HYBRID_SEARCH, LIVE_PREVIEW_MODEL_ID, LIVE_TRANSCRIPT_PREVIEW, REALTIME_MODEL_ID,
)
from citation_audit import audit_citations
from glossary_mcp import lookup_glossary
from voice_session import (
    AUTO_LANGUAGE,
    language_from_transcript,
    BrowserSession,
    LANGUAGE_NAMES,
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
from grounded_answer import LANGUAGE_NAMES as GROUNDED_LANGUAGES, answer_question
from highlight import find_quote_rects, highlighted_png
from study_quiz import generate_quiz, grade_quiz
from vocabulary import Corrector, build_known_words, extract_vocabulary, is_prompt_echo


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("voice_rag")

# Realtime errors that describe a lost race, not a broken session. Anything else still ends the call.
BENIGN_REALTIME_ERRORS = {
    "conversation_already_has_active_response",
    "response_cancel_not_active",
    "input_audio_buffer_commit_empty",
}

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
    transcript_event: asyncio.Event = field(default_factory=asyncio.Event)
    current_answer: str = ""
    current_answer_item_id: str = ""
    turn_sources: list[dict] = field(default_factory=list)
    quiz_started_this_turn: bool = False
    speech_stopped_at: float | None = None
    response_active: bool = False
    evidence: dict = field(default_factory=dict)  # card id -> {doc, page, quote, rects}
    status_open: bool = False  # a progress card is showing in the browser
    emit: Callable[[dict], None] = lambda event: None  # thread-safe: queue an event for the browser
    followup_instructions: str = ""
    audio_item_id: str = ""  # the response item the audio being streamed belongs to
    answer_language_check: tuple = (None, None)
    preview_text: str = ""  # latest live-preview words of the current question
    answer_phase: bool = False  # a tool result has been returned; the next audio is the real answer
    preview: object | None = None  # streaming transcription connection, display only
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


def _current_instructions(browser: BrowserSession) -> str:
    return _browser_instructions(
        browser.language_preference, browser.last_spoken_language, browser.turns, browser.quiz,
        browser.language_confident, browser.vocabulary,
    )


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
        texts = engine.metadata_db["content"].astype(str).tolist()
        browser = session.browser
        browser.vocabulary = extract_vocabulary(texts, list(browser.display_names.values()))
        browser.corrector = Corrector(browser.vocabulary, build_known_words(texts))
        log.info("document vocabulary (%d terms): %s", len(browser.vocabulary), ", ".join(browser.vocabulary[:25]))
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


@app.get("/api/sessions/{session_id}/evidence/{card_id}")
def evidence_page(session_id: str, card_id: str):
    """The cited page with the exact quote highlighted (the PDF on disk is never modified)."""
    session = _session_or_404(session_id)
    item = session.evidence.get(card_id)
    path = session.browser.documents.get(item["doc"]) if item else None
    if item is None or path is None:
        raise HTTPException(404, "Evidence not found.")
    with pymupdf.open(path) as document:
        page = document.load_page(item["page"] - 1)
        if item["rects"] is None:  # a scanned page: read word positions with OCR once, on demand
            item["rects"] = find_quote_rects(page, item["quote"], os.getenv("OCR_LANGUAGES", "eng").replace(",", "+"))
        return Response(highlighted_png(page, item["rects"], pymupdf.Matrix(1.3, 1.3)), media_type="image/png")


def build_evidence_cards(session: LiveSession, claims: list[dict], by_id: dict) -> list[dict]:
    """One card per verified claim: what it says, the exact quote, where it sits and a link to the page with the
    quote highlighted. ``located`` tells the browser whether the quote could be found on the page image."""
    cards, opened = [], {}
    try:
        for claim in claims[:14]:
            passage = by_id[claim["source_id"]]
            doc, page_number = passage["doc"], passage["page"]
            path = session.browser.documents.get(doc)
            rects, located = None, None
            if path is not None:
                try:  # evidence is a nicety: a page that cannot be read must never break the answer
                    if doc not in opened:
                        opened[doc] = pymupdf.open(path)
                    page = opened[doc].load_page(page_number - 1)
                    if page.get_text("words"):
                        rects = find_quote_rects(page, claim["quote"])
                        located = bool(rects)
                except Exception as exc:  # noqa: BLE001
                    log.warning("could not locate quote on %s p.%s: %s", doc, page_number, exc)
                    rects, located = None, None
            card_id = uuid.uuid4().hex[:12]
            session.evidence[card_id] = {"doc": doc, "page": page_number, "quote": claim["quote"], "rects": rects}
            section = passage.get("section", "")
            name = session.browser.display_names.get(doc, doc)
            cards.append({
                "id": card_id, "role": claim.get("role", "answer"), "claim": claim["text"], "quote": claim["quote"],
                "source_id": claim["source_id"], "page": page_number, "section": short_section(section, 3),
                "label": f"{name} \u00b7 {short_section(section)} (p. {page_number})" if section else f"{name} (p. {page_number})",
                "located": located, "url": f"/api/sessions/{session.id}/evidence/{card_id}",
            })
    finally:
        for document in opened.values():
            document.close()
    return cards


async def _send_sources(ws: WebSocket, session: LiveSession) -> None:
    sources = []
    for source in session.turn_sources:
        doc, page = source["doc"], source["page"]
        if doc not in session.browser.documents:
            continue
        sources.append({
            "id": source.get("id"),
            "label": (f"{session.browser.display_names[doc]} \u00b7 {short_section(source['section'])} (p. {page})"
                      if source.get("section") else f"{session.browser.display_names[doc]} - page {page}"),
            "url": f"/api/sessions/{session.id}/pages/{quote(doc, safe='')}/{page}",
            "excerpt": source.get("content", "")[:500],
        })
    await ws.send_json({"type": "sources", "sources": sources})


async def _await_transcript(session: LiveSession, current: str, timeout: float = 3.0) -> str:
    """The user transcript lands ~2 s after speech stops, usually after the model has already
    called a tool. Wait (bounded) for it so language decisions use what was actually said."""
    if current:
        return current
    try:
        await asyncio.wait_for(session.transcript_event.wait(), timeout)
    except asyncio.TimeoutError:
        log.warning("user transcript not ready after %.1fs", timeout)
    return session.current_question


def _resolve_language(session: LiveSession, name: str, args: dict, transcript: str) -> str:
    """Pick the answer language from the transcript's script; the model's guess is a fallback."""
    browser = session.browser
    claimed = args.get("spoken_language")
    detected = language_from_transcript(transcript, claimed)
    if detected in SUPPORTED_LANGUAGES:
        browser.last_spoken_language = detected
        browser.language_confident = True
    else:
        browser.language_confident = False
        if claimed in SUPPORTED_LANGUAGES:
            browser.last_spoken_language = claimed
    log.info("tool=%s transcript=%r transcript_language=%s model_claimed_language=%s using=%s confident=%s args=%s",
             name, transcript[:80], detected, claimed, browser.last_spoken_language,
             browser.language_confident, json.dumps(args, ensure_ascii=False)[:300])
    if claimed and detected and claimed != detected:
        log.warning("language mismatch: model claimed %s but transcript script says %s", claimed, detected)
        session.turn_metrics["language_claim_overridden"] = True
    return (
        browser.language_preference if browser.language_preference != AUTO_LANGUAGE
        else browser.last_spoken_language
    )


def short_section(path: str, depth: int = 2) -> str:
    """The last few levels of a section path: "Part D › Section 17 · Objection handling" -> the last two."""
    return " \u203a ".join(path.split(" \u203a ")[-depth:]) if path else ""


def where_label(sections: list[str], pages: list[int]) -> str:
    """Where an answer comes from, in a few words: its most-used section and the pages."""
    place = short_section(sections[0]) if sections else ""
    pages = sorted(set(pages))
    page_text = ("page " + str(pages[0])) if len(pages) == 1 else (
        f"pages {pages[0]}\u2013{pages[-1]}" if pages and pages[-1] - pages[0] == len(pages) - 1
        else "pages " + ", ".join(str(n) for n in pages[:5]))
    return f"{place} ({page_text})" if place and pages else place or (page_text if pages else "")


BRIEF_CLAIMS = 4  # claims a brief spoken reply is given


def teaching_instructions(name: str, depth: str, flow: list[dict], where: str, plain_words: str,
                          more_available: bool = False) -> str:
    """What the voice model is told for this reply: how to teach the verified claims, nothing more."""
    claims = " ".join(f"[{c['role']}] {c['claim']}" for c in flow)
    place = f"Say in a few words that this comes from {where}. " if where else ""
    more = ("These claims are only the first few of a longer verified list: introduce them with 'for example' or "
            "'including', never say how many items there are, and offer the rest. " if more_available else "")
    plain = (f"Finish with a sentence that starts with the equivalent of 'In plain words,' and says: {plain_words} "
             if plain_words else "")
    no_guessing = ("If the learner asked why or what something means and no [reason] claim is listed, say plainly that "
                   "the document does not state it; never guess. ")  # used for detailed lessons only
    if depth == "detailed":
        return (
            f"Teach the learner in {name} as a short spoken lesson of at most about eight sentences, using only these "
            f"verified claims. Give the direct answer first; then any definitions; then each step, adding why it "
            f"matters right after it only when a 'reason' claim for it is listed (never invent a reason); then an "
            f"example or caution if listed. {no_guessing}{place}{plain}End with one sentence offering a quick quiz question. "
            f"Add no other facts. Do not read source ids aloud. Claims: {claims}")
    return (
        f"Answer the learner in {name} using only these verified claims, in two to four short spoken sentences: the "
        f"direct answer first, then the one or two points that matter most; for a list, name a few of the "
        f"items and say you can go through the rest (do not state a total). {place}{plain}{more}Then offer, in one short sentence, to go deeper "
        f"(each item, the reasons, an example, or a quick quiz question). Add no other facts. Do not "
        f"read source ids aloud. Claims: {claims}")


async def _grounded_tool_output(
    session: LiveSession, question: str, language: str, passages: list[dict], progress=None,
    depth: str = "brief", draft_depth: str | None = None,
) -> str:
    """Answer from retrieved passages through the claim verifier and hand the voice model only
    verified claims (or a fixed "not found"). Sources shown to the learner are the verified ones."""
    numbered = [{**passage, "id": f"S{i}"} for i, passage in enumerate(passages, start=1)]
    short = language[:2]
    target = short if short in GROUNDED_LANGUAGES else LANGUAGE_NAMES.get(language, "English")
    started = time.perf_counter()
    result = await asyncio.to_thread(
        answer_question, session.browser.engine.client, question, numbered, target, True, progress,
        draft_depth or depth)
    session.turn_metrics.update({
        "grounding_seconds": round(time.perf_counter() - started, 2),
        "grounded_abstained": result.abstained,
        "claims_verified": len(result.claims),
        "claims_dropped": len(result.dropped),
    })
    name = LANGUAGE_NAMES.get(language, "English")
    pages = sorted({p["page"] for p in numbered})
    session.status_open = False
    session.emit({
        "type": "status", "state": "end", "abstained": result.abstained,
        "summary": ("Nothing found to back an answer" if result.abstained else
                    f"Checked {len(numbered)} passages · {len(result.claims)} verified claim"
                    f"{'s' if len(result.claims) != 1 else ''}") + f" · {session.turn_metrics['grounding_seconds'] + session.turn_metrics.get('retrieval_seconds', 0):.1f} s",
        "pages": pages,
    })
    log.info("grounded answer: abstained=%s verified=%d dropped=%d in %.2fs",
             result.abstained, len(result.claims), len(result.dropped), session.turn_metrics["grounding_seconds"])
    if result.abstained:
        log.warning("abstained on question=%r; passages: %s", question[:160],
                    "; ".join(f"p{p['page']} {p.get('section', '')[-40:]}" for p in numbered))
        session.followup_instructions = (
            f"Say this to the learner in {name}, in one short sentence, and nothing else: {result.text} "
            "Do not answer from your own knowledge and do not guess."
        )
        return json.dumps({
            "status": "not_found",
            "message": result.text,
            "instruction": f"Tell the learner this in {name}, briefly. Do not answer from your own knowledge "
                           "and do not guess.",
        }, ensure_ascii=False)
    by_id = {p["id"]: p for p in numbered}
    flow, seen = [], set()
    for claim in result.claims:
        passage = by_id[claim["source_id"]]
        if claim["source_id"] not in seen:
            seen.add(claim["source_id"])
            session.turn_sources.append(dict(passage))
        flow.append({"role": claim.get("role", "answer"), "source": claim["source_id"], "claim": claim["text"]})
    where = where_label(result.sections, [c["page"] for c in result.claims if "page" in c])
    # A model reads out whatever it is handed, so a brief reply is given only the leading claims; everything is
    # given for a detailed lesson. The rest stays available for the learner to ask for.
    shown = flow if depth == "detailed" else flow[:BRIEF_CLAIMS]
    more = len(flow) > len(shown)
    cards = await asyncio.to_thread(build_evidence_cards, session, result.claims, by_id)
    session.emit({"type": "evidence", "cards": cards, "reading": result.explanation})
    session.followup_instructions = teaching_instructions(name, depth, shown, where, result.explanation, more)
    return json.dumps({
        "status": "verified_answer",
        "where": where,
        "claims": shown,
        "more_available": more,
        "plain_words": result.explanation,
        "depth": depth,
        "instruction": f"Teach this in {name} using only these verified claims. Add no other facts.",
    }, ensure_ascii=False)


async def _run_tool(session: LiveSession, name: str, args: dict, transcript: str) -> str:
    browser = session.browser
    engine = browser.engine
    # Start waiting for the transcript now so retrieval and transcription overlap.
    transcript_task = asyncio.ensure_future(_await_transcript(session, transcript))
    if name != "search_documents":
        transcript = await transcript_task
        answer_language = _resolve_language(session, name, args, transcript)
    if name == "search_documents":
        query = _extract_query(json.dumps(args), "")
        if not query:
            query = _extract_query(None, await transcript_task)
        if not query:
            return json.dumps({"status": "no_question"})
        if browser.corrector is not None:
            query, fixes = browser.corrector.correct(query)
            if fixes:
                log.info("search query respelled from document vocabulary: %s", fixes)
        session.status_open = True
        session.emit({"type": "status", "state": "start", "query": query})

        def progress(text: str) -> None:
            session.emit({"type": "status", "state": "step", "text": text})

        started = time.perf_counter()
        hint = session.current_question or session.preview_text
        retrieval = await asyncio.to_thread(
            retrieve_documents,
            query, engine.faiss_index, engine.metadata_db, engine.client, RAG_TOP_K, True, progress,
            bool(args.get("covers_all")), hint, list(browser.turns[-3:]) or None, HYBRID_SEARCH, FOLLOWUP_REWRITE,
        )
        session.turn_metrics["retrieval_seconds"] = round(time.perf_counter() - started, 2)
        if browser.language_preference != AUTO_LANGUAGE:
            # The learner chose the answer language, so the late transcript is not needed for it.
            transcript_task.cancel()
            transcript = session.current_question
        else:
            transcript = await transcript_task
        language = _resolve_language(session, name, args, transcript)
        if retrieval.error:
            return f"Document search failed: {retrieval.error}"
        if not retrieval.passages:
            return retrieval.context
        if GROUNDED_VOICE:
            question = transcript if len(transcript.split()) >= 3 else query
            depth = "detailed" if args.get("depth") == "detailed" else "brief"
            # A list needs every item drafted even when it is spoken briefly.
            draft_depth = "detailed" if args.get("covers_all") else depth
            return await _grounded_tool_output(session, question, language, retrieval.passages, progress, depth,
                                               draft_depth)
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


def preview_session_payload() -> dict:
    # No vocabulary prompt: the streaming model rejects one. The preview is raw on purpose; the accurate
    # transcript (with document terms) replaces it and respells anything the preview got wrong.
    return {"type": "transcription", "audio": {"input": {
        "format": {"type": "audio/pcm", "rate": 24000},
        "transcription": {"model": LIVE_PREVIEW_MODEL_ID},
        "turn_detection": None,  # these streaming models do not take server VAD; we commit by hand
    }}}


async def _drop_preview(session: LiveSession, reason: Exception | str) -> None:
    """The preview is a nicety: on any problem turn it off and keep the call going."""
    if session.preview is not None:
        log.warning("live transcript preview disabled: %s", reason)
        session.preview = None


async def _preview_append(session: LiveSession, audio_b64: str) -> None:
    if session.preview is None:
        return
    try:
        await session.preview.input_audio_buffer.append(audio=audio_b64)
    except Exception as exc:  # noqa: BLE001
        await _drop_preview(session, exc)


async def _preview_commit(session: LiveSession) -> None:
    """End of an utterance: flush the preview's buffer so the next one starts a fresh segment."""
    if session.preview is None:
        return
    try:
        await session.preview.input_audio_buffer.commit()
    except Exception as exc:  # noqa: BLE001
        await _drop_preview(session, exc)


async def _status_pump(ws: WebSocket, queue: asyncio.Queue) -> None:
    """Send queued progress events to the browser in order."""
    try:
        while True:
            await ws.send_json(await queue.get())
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - progress is cosmetic; the call carries on
        log.warning("status stream stopped: %s", exc)


async def _from_preview(ws: WebSocket, session: LiveSession, preview) -> None:
    """Relay streaming transcript text to the browser as an accumulating live line."""
    current_item, text = None, ""
    try:
        async for event in preview:
            kind = getattr(event, "type", "")
            if kind == "error":
                await _drop_preview(session, str(getattr(event, "error", event)))
                return
            if kind == "conversation.item.input_audio_transcription.delta":
                item = getattr(event, "item_id", "")
                if item != current_item:
                    current_item, text = item, ""
                text += getattr(event, "delta", "") or ""
                session.preview_text = text
                await ws.send_json({"type": "live_transcript", "text": text})
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        await _drop_preview(session, exc)


async def _set_answer_language(ws: WebSocket, conn, session: LiveSession, language: str) -> None:
    """Apply the learner's dropdown choice: a fixed language sticks, ``auto`` follows speech."""
    if language not in (AUTO_LANGUAGE, *SUPPORTED_LANGUAGES):
        await ws.send_json({"type": "error", "message": "Unsupported answer language."})
        return
    browser = session.browser
    browser.language_preference = language
    log.info("answer language set to %s from the UI", language)
    await conn.session.update(session={
        "type": "realtime",
        "instructions": _current_instructions(browser),
    })
    await ws.send_json({"type": "preferences", "language": language})


async def _from_browser(ws: WebSocket, conn, session: LiveSession) -> None:
    while True:
        message = await ws.receive()
        if message["type"] == "websocket.disconnect":
            return
        audio = message.get("bytes")
        if audio is None:
            try:
                control = json.loads(message.get("text") or "")
            except ValueError:
                continue
            if isinstance(control, dict) and control.get("type") == "set_language":
                await _set_answer_language(ws, conn, session, str(control.get("language", "")))
            continue
        if len(audio) < 2 or len(audio) % 2 or len(audio) > 48000:
            await ws.send_json({"type": "error", "message": "Invalid microphone audio frame."})
            continue
        encoded = base64.b64encode(audio).decode("ascii")
        await conn.input_audio_buffer.append(audio=encoded)
        if session is not None:
            await _preview_append(session, encoded)


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
            error = getattr(event, "error", event)
            if getattr(error, "code", None) in BENIGN_REALTIME_ERRORS:
                # A race such as speaking while a tool call is finishing: the call can carry on.
                log.warning("realtime error event ignored: %s", error)
                session.turn_metrics.setdefault("ignored_errors", []).append(getattr(error, "code", ""))
                continue
            log.error("realtime error event: %s", error)
            raise RuntimeError(str(error))
        if kind == "response.created":
            active_response_id = _event_response_id(event) or ""
            session.response_active = True
        elif kind == "input_audio_buffer.speech_started":
            if session.response_active:
                session.turn_metrics["interrupted_previous_response"] = True
            session.turn_sources.clear()
            session.followup_instructions = ""
            session.preview_text = ""
            session.answer_phase = False
            session.quiz_started_this_turn = False
            session.current_question = ""
            session.transcript_event.clear()
            session.turn_metrics = {"interrupted_previous_response": session.response_active}
            session.speech_stopped_at = None
            await ws.send_json({"type": "speech_started"})
            await ws.send_json({"type": "evidence", "cards": [], "reading": ""})
            await _send_sources(ws, session)
        elif kind == "input_audio_buffer.speech_stopped":
            session.speech_stopped_at = time.perf_counter()
            await _preview_commit(session)
            await ws.send_json({"type": "speech_stopped"})
        elif kind == "conversation.item.input_audio_transcription.delta":
            await ws.send_json({
                "type": "transcript", "speaker": "user", "item_id": getattr(event, "item_id", ""),
                "delta": getattr(event, "delta", ""), "final": False,
            })
        elif kind == "conversation.item.input_audio_transcription.completed":
            heard = getattr(event, "transcript", "") or ""
            if is_prompt_echo(heard, session.browser.vocabulary):
                # The transcription model read its own vocabulary hint back for a noisy or silent stretch. It
                # is not something the learner said, so it must not replace the real question.
                log.warning("ignored a transcript that was only the vocabulary hint read back")
                await ws.send_json({"type": "transcript_discard", "item_id": getattr(event, "item_id", "")})
                continue
            corrector = session.browser.corrector
            fixed, fixes = corrector.correct(heard) if corrector is not None else (heard, [])
            session.current_question = fixed
            log.info("user transcript: %s", heard)
            if fixes:
                log.info("transcript respelled from document vocabulary: %s -> %s", fixes, fixed)
                session.turn_metrics["transcript_corrections"] = len(fixes)
            session.transcript_event.set()
            await ws.send_json({
                "type": "transcript", "speaker": "user", "item_id": getattr(event, "item_id", ""),
                "text": fixed, "final": True, "heard": heard, "corrections": fixes,
            })
        elif kind == "response.function_call_arguments.done":
            response_id = getattr(event, "response_id", None) or active_response_id
            call_id = getattr(event, "call_id", None)
            if not call_id:
                raise RuntimeError("Tool call had no call ID.")
            calls = calls_by_response.setdefault(response_id, set())
            if call_id in calls:
                continue
            tool_name = getattr(event, "name", "")
            try:
                output = await _run_tool(
                    session, tool_name,
                    _tool_args(getattr(event, "arguments", None)), session.current_question,
                )
            except Exception as exc:  # a failing tool must not end the call
                log.warning("Tool %s failed: %s", tool_name, exc)
                session.turn_metrics.setdefault("tool_errors", []).append(tool_name)
                output = json.dumps({
                    "status": "tool_error", "tool": tool_name,
                    "message": "The tool failed. Tell the user briefly and offer to try again.",
                })
            await conn.conversation.item.create(item={
                "type": "function_call_output", "call_id": call_id, "output": output,
            })
            calls.add(call_id)
            session.answer_phase = True
            if session.status_open:  # the tool ended without reporting (error, no passages, ungrounded mode)
                session.status_open = False
                session.emit({"type": "status", "state": "end", "abstained": False, "summary": "Search finished"})
            await _send_sources(ws, session)
            browser = session.browser
            await ws.send_json({"type": "preferences", "language": browser.language_preference})
            await conn.session.update(session={
                "type": "realtime",
                "instructions": _current_instructions(browser),
            })
        elif kind == "response.output_audio.delta":
            session.audio_emitted = True
            if "time_to_first_audio_ms" not in session.turn_metrics and session.speech_stopped_at is not None:
                session.turn_metrics["time_to_first_audio_ms"] = round(
                    (time.perf_counter() - session.speech_stopped_at) * 1000, 1,
                )
            if (session.answer_phase and "time_to_answer_audio_ms" not in session.turn_metrics
                    and session.speech_stopped_at is not None):
                session.turn_metrics["time_to_answer_audio_ms"] = round(
                    (time.perf_counter() - session.speech_stopped_at) * 1000, 1,
                )
            item_id = getattr(event, "item_id", "") or ""
            if item_id and item_id != session.audio_item_id:
                # Lets the browser pace each response's text to its own audio.
                session.audio_item_id = item_id
                await ws.send_json({"type": "audio_item", "item_id": item_id})
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
            preference = session.browser.language_preference
            expected_code = session.browser.last_spoken_language if preference == AUTO_LANGUAGE else preference
            expected = LANGUAGE_NAMES.get(expected_code, "")
            spoken = language_from_transcript(session.current_answer)
            log.info("assistant transcript (%s, expected %s): %s", spoken, expected, session.current_answer[:300])
            session.answer_language_check = (spoken, expected_code)
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
                followup = {"tool_choice": "none"}
                if session.followup_instructions:
                    followup["instructions"] = session.followup_instructions
                    log.info("spoken-answer instructions (%d chars): %s", len(session.followup_instructions),
                             session.followup_instructions[:240])
                    session.followup_instructions = ""
                await conn.response.create(response=followup)
            elif not calls:
                spoken, expected_code = getattr(session, "answer_language_check", (None, None))
                if spoken and expected_code and spoken[:2] != expected_code[:2]:
                    log.warning("answer language %s differs from expected %s", spoken, expected_code)
                    session.turn_metrics["answer_language_mismatch"] = True
                session.answer_language_check = (None, None)
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
                log.info("turn metrics (grounded_voice=%s): %s", GROUNDED_VOICE,
                         json.dumps(session.turn_metrics, ensure_ascii=False))
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
        async_client = session.browser.engine.async_client
        async with contextlib.AsyncExitStack() as stack:
            conn = await stack.enter_async_context(async_client.realtime.connect(model=REALTIME_MODEL_ID))
            await conn.session.update(session=live_session_payload(session.browser))
            preview_task = None
            session.preview = None
            if LIVE_TRANSCRIPT_PREVIEW:
                try:
                    preview = await stack.enter_async_context(
                        async_client.realtime.connect(extra_query={"intent": "transcription"}))
                    await preview.session.update(session=preview_session_payload())
                    session.preview = preview
                    preview_task = asyncio.create_task(_from_preview(ws, session, preview))
                except Exception as exc:  # noqa: BLE001 - the call works without the preview
                    await _drop_preview(session, exc)
            loop = asyncio.get_running_loop()
            status_queue: asyncio.Queue = asyncio.Queue()
            session.emit = lambda event: loop.call_soon_threadsafe(status_queue.put_nowait, event)
            status_task = asyncio.create_task(_status_pump(ws, status_queue))
            await ws.send_json({"type": "ready"})
            incoming = asyncio.create_task(_from_browser(ws, conn, session))
            outgoing = asyncio.create_task(_from_openai(ws, conn, session))
            try:
                done, pending = await asyncio.wait({incoming, outgoing}, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                for task in pending:
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                for task in done:
                    task.result()
            finally:
                session.preview = None
                session.emit = lambda event: None
                status_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await status_task
                if preview_task is not None:
                    preview_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await preview_task
    except (WebSocketDisconnect, asyncio.CancelledError):
        log.info("browser disconnected")
    except Exception as exc:
        log.exception("voice session ended with an error: %s", exc)
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "error", "message": str(exc)})
    finally:
        session.connected = False
        with contextlib.suppress(Exception):
            await ws.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
