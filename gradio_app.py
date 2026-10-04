"""Local, Python-only browser interface for the voice RAG prototype."""

import asyncio
import base64
import contextlib
import hashlib
import io
import math
import re
import shutil
import wave
from dataclasses import dataclass, field
from pathlib import Path

import fitz
import gradio as gr
import numpy as np
from PIL import Image
from scipy.signal import resample_poly

from module_v2v_rag import (
    AVAILABLE_VOICES,
    INPUT_RATE,
    OUTPUT_RATE,
    RAG_TOP_K,
    SUPPORTED_LANGUAGES,
    TURN_TIMEOUT_S,
    VoiceRAGMultimodalFAISS,
    _event_response_id,
    _event_response_status,
    _extract_query,
    build_session_payload,
    retrieve_context_and_sources,
)
from config import REALTIME_MODEL_ID


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / ".local_data"
DOCUMENT_DIR = DATA_DIR / "documents"
MAX_DOCUMENTS = 5
MAX_PDF_BYTES = 25 * 1024 * 1024
MAX_AUDIO_SECONDS = 45


@dataclass
class BrowserSession:
    engine: VoiceRAGMultimodalFAISS
    documents: dict[str, Path]
    display_names: dict[str, str]
    turns: list[tuple[str, str]] = field(default_factory=list)
    source_pages: dict[str, tuple[Path, int]] = field(default_factory=dict)


_sessions: dict[str, BrowserSession] = {}


def _session_key(request: gr.Request) -> str:
    if request is None or not request.session_hash:
        raise ValueError("Browser session unavailable. Reload the page and try again.")
    return request.session_hash


def _store_pdf(uploaded_path: str) -> tuple[Path, str]:
    source = Path(uploaded_path)
    if source.suffix.lower() != ".pdf" or not source.is_file():
        raise ValueError("Only PDF files are supported.")
    if source.stat().st_size > MAX_PDF_BYTES:
        raise ValueError(f"{source.name} exceeds the 25 MB document limit.")
    with source.open("rb") as pdf_file:
        signature = pdf_file.read(5)
    if source.stat().st_size < 5 or signature != b"%PDF-":
        raise ValueError(f"{source.name} is not a valid PDF file.")

    digest = hashlib.sha256()
    with source.open("rb") as pdf_file:
        for chunk in iter(lambda: pdf_file.read(1024 * 1024), b""):
            digest.update(chunk)
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source.stem).strip("._-")[:70] or "document"
    destination = DOCUMENT_DIR / f"{digest.hexdigest()[:16]}_{safe_stem}.pdf"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copyfile(source, destination)
    return destination, source.name


def _audio_to_pcm16(audio: tuple[int, np.ndarray]) -> bytes:
    if audio is None:
        raise ValueError("Record a question with the microphone first.")
    sample_rate, samples = audio
    if sample_rate <= 0 or samples is None or samples.size == 0:
        raise ValueError("The recording is empty.")
    if samples.ndim not in (1, 2):
        raise ValueError("Unsupported audio shape.")
    if len(samples) / sample_rate > MAX_AUDIO_SECONDS:
        raise ValueError("Keep each question under 45 seconds.")
    signal = samples.astype(np.float32)
    if samples.ndim == 2:
        signal = signal.mean(axis=1)
    if np.issubdtype(samples.dtype, np.integer):
        signal /= max(abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max)
    if sample_rate != INPUT_RATE:
        factor = math.gcd(sample_rate, INPUT_RATE)
        signal = resample_poly(signal, INPUT_RATE // factor, sample_rate // factor)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    return (np.clip(signal, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def _pcm_to_wav(pcm: bytes) -> bytes | None:
    if not pcm:
        return None
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(OUTPUT_RATE)
        wav_file.writeframes(pcm)
    return output.getvalue()


async def _answer_turn(
    session: BrowserSession, pcm: bytes, language: str, voice: str
) -> tuple[str, str, bytes | None, list[dict]]:
    engine = session.engine
    user_transcript = ""
    answer_parts: list[str] = []
    answer_done = ""
    audio_parts: list[bytes] = []
    sources: list[dict] = []
    initial_response_id = None
    continuation_response_id = None
    tool_output_sent = False
    phase = "search"

    async with engine.async_client.realtime.connect(model=REALTIME_MODEL_ID) as conn:
        payload = build_session_payload(language, voice)
        if session.turns:
            recent = session.turns[-3:]
            history = "\n".join(f"User: {q[:400]}\nAssistant: {a[:700]}" for q, a in recent)
            payload["instructions"] += (
                "\nRecent conversation for resolving references only; search results remain "
                f"the sole source of factual answers:\n{history}"
            )
        await conn.session.update(session=payload)
        await conn.input_audio_buffer.append(audio=base64.b64encode(pcm).decode("ascii"))
        await conn.input_audio_buffer.commit()
        await conn.response.create()

        async def consume():
            nonlocal user_transcript, answer_done, initial_response_id
            nonlocal continuation_response_id, tool_output_sent, phase, sources
            async for event in conn:
                event_type = getattr(event, "type", "")
                if event_type == "error":
                    raise RuntimeError(str(getattr(event, "error", event)))
                if event_type == "conversation.item.input_audio_transcription.completed":
                    user_transcript = getattr(event, "transcript", "") or ""
                elif event_type == "response.created":
                    response_id = _event_response_id(event)
                    if phase == "search" and initial_response_id is None:
                        initial_response_id = response_id
                    elif phase == "answer" and continuation_response_id is None:
                        continuation_response_id = response_id
                elif event_type == "response.function_call_arguments.done" and phase == "search":
                    query = _extract_query(getattr(event, "arguments", None), user_transcript)
                    if not query:
                        raise RuntimeError("Could not understand the question. Please try again.")
                    context, sources = await asyncio.to_thread(
                        retrieve_context_and_sources,
                        query, engine.faiss_index, engine.metadata_db, engine.client, RAG_TOP_K,
                    )
                    await conn.conversation.item.create(item={
                        "type": "function_call_output",
                        "call_id": getattr(event, "call_id", None),
                        "output": context,
                    })
                    tool_output_sent = True
                    if not user_transcript:
                        user_transcript = query
                elif event_type in ("response.output_audio.delta", "response.audio.delta"):
                    delta = getattr(event, "delta", None)
                    if delta:
                        audio_parts.append(base64.b64decode(delta))
                elif event_type in (
                    "response.output_audio_transcript.delta", "response.audio_transcript.delta",
                ):
                    answer_parts.append(getattr(event, "delta", "") or "")
                elif event_type in (
                    "response.output_audio_transcript.done", "response.audio_transcript.done",
                ):
                    answer_done = getattr(event, "transcript", "") or ""
                elif event_type == "response.done":
                    response_id = _event_response_id(event)
                    status = _event_response_status(event)
                    if phase == "search":
                        if not tool_output_sent:
                            continue
                        if initial_response_id and response_id and response_id != initial_response_id:
                            continue
                        if status and status != "completed":
                            raise RuntimeError(f"Document search did not complete ({status}).")
                        phase = "answer"
                        await conn.response.create(response={"tool_choice": "none"})
                    else:
                        if continuation_response_id and response_id and response_id != continuation_response_id:
                            continue
                        if status and status != "completed":
                            raise RuntimeError(f"Spoken answer did not complete ({status}).")
                        return

        await asyncio.wait_for(consume(), timeout=TURN_TIMEOUT_S)

    transcript = answer_done or "".join(answer_parts)
    return user_transcript, transcript, _pcm_to_wav(b"".join(audio_parts)), sources


def _preview_page(session: BrowserSession, key: str | None) -> Image.Image | None:
    if not key or key not in session.source_pages:
        return None
    path, page_number = session.source_pages[key]
    with fitz.open(path) as document:
        if page_number < 1 or page_number > len(document):
            return None
        pixmap = document.load_page(page_number - 1).get_pixmap(matrix=fitz.Matrix(1.4, 1.4))
    with Image.open(io.BytesIO(pixmap.tobytes("png"))) as image:
        return image.copy()


async def index_documents(
    files: list[str] | None, request: gr.Request, progress: gr.Progress = gr.Progress()
):
    if not files:
        raise gr.Error("Choose at least one PDF.")
    if len(files) > MAX_DOCUMENTS:
        raise gr.Error("Choose up to five PDFs at a time.")
    key = _session_key(request)
    try:
        progress(0, desc="Preparing PDFs")
        stored = [_store_pdf(path) for path in files]
        paths = [str(path) for path, _ in stored]
        engine = VoiceRAGMultimodalFAISS(
            pdf_paths=paths,
            language="en-US",
            cache_dir=DATA_DIR / "indexes",
            image_save_dir=DATA_DIR / "images",
        )
        progress(0.1, desc="Indexing document pages")
        await engine.setup_rag_database()
        if not engine.is_active or engine.faiss_index is None:
            raise RuntimeError("No searchable content was found in these PDFs.")
        documents = {path.stem: path for path, _ in stored}
        display_names = {path.stem: name for path, name in stored}
        previous = _sessions.get(key)
        _sessions[key] = BrowserSession(engine, documents, display_names)
        if previous is not None:
            with contextlib.suppress(Exception):
                previous.engine.client.close()
                await previous.engine.async_client.close()
        progress(1, desc="Documents ready")
    except Exception as exc:
        raise gr.Error(f"Could not index documents: {exc}") from exc
    names = ", ".join(name for _, name in stored)
    return f"Ready: {names}", [], gr.update(interactive=True), gr.update(choices=[], value=None), None


async def ask_question(audio, history, language, voice, request: gr.Request):
    key = _session_key(request)
    session = _sessions.get(key)
    if session is None:
        raise gr.Error("Index documents before asking a question.")
    if language not in SUPPORTED_LANGUAGES or voice not in AVAILABLE_VOICES:
        raise gr.Error("Choose a supported language and voice.")
    try:
        pcm = _audio_to_pcm16(audio)
        question, answer, answer_audio, sources = await _answer_turn(session, pcm, language, voice)
        if not answer and not answer_audio:
            raise RuntimeError("The assistant returned no answer. Please try again.")
    except Exception as exc:
        raise gr.Error(f"Could not answer this question: {exc}") from exc

    question = question or "Spoken question"
    answer = answer or "Spoken answer"
    session.turns.append((question, answer))
    updated_history = list(history or []) + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    source_pages = {}
    choices = []
    for source in sources:
        doc_name = source["doc"]
        page = source["page"]
        path = session.documents.get(doc_name)
        if path is None:
            continue
        source_key = f"{doc_name}:{page}"
        if source_key in source_pages:
            continue
        source_pages[source_key] = (path, page)
        label = f"{session.display_names[doc_name]} - page {page}"
        choices.append((label, source_key))
    session.source_pages = source_pages
    first = choices[0][1] if choices else None
    status = "Answer ready. Retrieved pages are shown alongside the response."
    if not choices:
        status = "Answer ready. No document pages were retrieved."
    return (
        updated_history,
        answer_audio,
        gr.update(choices=choices, value=first, interactive=bool(choices)),
        _preview_page(session, first),
        status,
        gr.update(value=None),
    )


def show_page(source_key: str | None, request: gr.Request):
    session = _sessions.get(_session_key(request))
    return _preview_page(session, source_key) if session else None


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Voice Study Companion") as demo:
        gr.Markdown("# Voice Study Companion")
        status = gr.Markdown("Upload study PDFs to begin.")
        with gr.Row():
            with gr.Column(scale=2):
                files = gr.File(label="Study PDFs", file_count="multiple", file_types=[".pdf"], type="filepath")
                index_button = gr.Button("Index documents", variant="primary")
                sources = gr.Dropdown(label="Retrieved pages", choices=[], interactive=False)
                page_preview = gr.Image(label="Source page", interactive=False, height=500)
            with gr.Column(scale=3):
                conversation = gr.Chatbot(label="Conversation", height=360)
                microphone = gr.Audio(label="Ask by voice", sources=["microphone"], type="numpy", interactive=False)
                answer_audio = gr.Audio(label="Spoken answer", autoplay=True, interactive=False)
                with gr.Row():
                    language = gr.Dropdown(SUPPORTED_LANGUAGES, value="en-US", label="Question and answer language")
                    voice = gr.Dropdown(AVAILABLE_VOICES, value="alloy", label="Voice")

        index_button.click(
            index_documents,
            inputs=[files],
            outputs=[status, conversation, microphone, sources, page_preview],
        )
        microphone.stop_recording(
            ask_question,
            inputs=[microphone, conversation, language, voice],
            outputs=[conversation, answer_audio, sources, page_preview, status, microphone],
        )
        sources.change(show_page, inputs=[sources], outputs=[page_preview])
    return demo


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", server_port=7860, share=False)
