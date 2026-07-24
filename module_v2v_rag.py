# module_v2v_rag.py

import asyncio
import sys
import os
import base64
import json
import io
import argparse
import time  # For simple delays / backoff
import pandas as pd
import fitz  # PyMuPDF for image extraction
from PIL import Image as PIL_Image
from pathlib import Path

# --- FAISS ---
try:
    import faiss
except ImportError:
    print("ERROR: faiss-cpu (or faiss-gpu) not found. Install: pip install faiss-cpu")
    sys.exit(1)
try:
    import pyarrow  # Required by pandas for parquet
except ImportError:
    print("ERROR: pyarrow not found. Install: pip install pyarrow")
    sys.exit(1)

import numpy as np

# OpenAI clients (sync for embeddings/vision/rerank, async for realtime).
from openai import OpenAI, AsyncOpenAI

# Import configuration loader / model IDs.
try:
    from config import (
        OPENAI_API_KEY,
        REALTIME_MODEL_ID,
        EMBEDDING_MODEL_ID,
        EMBEDDING_DIM,
        VISION_MODEL_ID,
        RERANK_MODEL_ID,
        TRANSCRIBE_MODEL_ID,
    )
except ImportError:
    print("Error: Could not import config.py.", file=sys.stderr)
    sys.exit(1)

# --- Audio Constants ---
# OpenAI realtime audio is pcm16, 24kHz, mono in both directions.
INPUT_RATE = 24000
OUTPUT_RATE = 24000
CHANNELS = 1
CHUNK_SIZE = 1024
# PyAudio format constant is resolved lazily (pyaudio only imported in voice mode).

# --- RAG Constants ---
RAG_TEXT_CHUNK_SIZE = 800
RAG_CHUNK_OVERLAP = 150
RAG_TOP_K = 5
RAG_CANDIDATE_K = 20
RERANK_TIMEOUT_S = 4.0
IMAGE_SAVE_DIR = "extracted_images"

# --- FAISS Index File Suffixes ---
FAISS_INDEX_SUFFIX = ".faissindex"
METADATA_SUFFIX = ".meta.parquet"

# --- Realtime turn handling ---
TURN_TIMEOUT_S = 30
# Retrieval mode: "tool" (model calls search_documents) or "inject" (deterministic
# fallback that always retrieves on the transcript and injects context as text).
RAG_MODE = os.getenv("RAG_MODE", "tool").lower()

# --- Session Configuration ---
DEFAULT_VOICE = "alloy"
DEFAULT_LANGUAGE = "hi-IN"
SUPPORTED_LANGUAGES = [
    "en-US", "ta-IN", "gu-IN", "mr-IN", "te-IN", "bn-IN",
    "kn-IN", "hi-IN", "ml-IN", "en-GB", "en-IN",
]
# OpenAI realtime voices (marin and cedar are current additions).
AVAILABLE_VOICES = [
    "alloy", "ash", "ballad", "coral", "echo",
    "sage", "shimmer", "verse", "marin", "cedar",
]

# BCP-47 -> ISO-639-1 mapping used for the transcription language hint.
LANG_ISO_MAP = {
    "hi-IN": "hi",
    "en-US": "en",
    "en-GB": "en",
    "en-IN": "en",
    "ta-IN": "ta",
    "te-IN": "te",
    "bn-IN": "bn",
    "gu-IN": "gu",
    "mr-IN": "mr",
    "kn-IN": "kn",
    "ml-IN": "ml",
}


def iso_lang(code: str) -> str:
    """Map a BCP-47 code to ISO-639-1; fall back to the code prefix."""
    if not code:
        return "en"
    if code in LANG_ISO_MAP:
        return LANG_ISO_MAP[code]
    return code.split("-")[0]


def _event_response_id(event):
    """Extract a response id from a realtime event (GA nested or flat), or None."""
    resp = getattr(event, "response", None)
    if resp is not None:
        rid = getattr(resp, "id", None)
        if rid is not None:
            return rid
    return getattr(event, "response_id", None)


def _event_response_status(event):
    """Extract a response status from a realtime event, or None if unavailable."""
    resp = getattr(event, "response", None)
    if resp is not None:
        return getattr(resp, "status", None)
    return None


def _extract_query(raw_args, fallback: str) -> str:
    """Parse tool-call arguments defensively and return a usable query string.

    ``raw_args`` may be malformed JSON, or decode to None/list/dict-with-non-string
    ``query``. In any such case (or when ``query`` is empty/whitespace), return the
    ``fallback`` transcript. Never raises.
    """
    try:
        args = json.loads(raw_args) if raw_args else None
    except Exception:  # noqa: BLE001
        args = None
    query = None
    if isinstance(args, dict):
        candidate = args.get("query")
        if isinstance(candidate, str):
            query = candidate.strip()
    if query:
        return query
    return (fallback or "").strip()


# --- Retry helper (embeddings + vision only) ---
def with_retry(fn, *, attempts=3, base_delay=1.0):
    """Run ``fn`` with exponential backoff on transient/rate-limit errors.

    Returns ``fn()``'s result, or None after exhausting attempts. Used by
    embeddings and vision captioning only -- NOT by rerank, which enforces a
    single ~4s deadline with a FAISS-order fallback (retries would blow the
    latency budget).
    """
    last_err = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - broad by design; SDK error types vary
            last_err = e
            if attempt < attempts - 1:
                delay = base_delay * (2 ** attempt)
                print(
                    f"\nWarning: transient API error ({type(e).__name__}: {e}); "
                    f"retrying in {delay:.1f}s...",
                    file=sys.stderr,
                )
                time.sleep(delay)
            else:
                print(
                    f"\nWarning: API call failed after {attempts} attempts: "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr,
                )
    return None


# --- Helper Functions (RAG Logic - FAISS Integrated) ---

def get_text_embedding(client: OpenAI, text: str) -> list | None:
    """Generate a 1536-dim text embedding via OpenAI."""
    if not text or text.isspace():
        return None

    def _call():
        resp = client.embeddings.create(model=EMBEDDING_MODEL_ID, input=text)
        return resp.data[0].embedding

    return with_retry(_call)


def chunk_text(text: str, size: int = RAG_TEXT_CHUNK_SIZE, overlap: int = RAG_CHUNK_OVERLAP) -> list[str]:
    """Word-boundary chunker with whole-word overlap and forward progress.

    - Accumulates words until adding the next would exceed ``size`` chars.
    - Carries back whole words totalling up to ``overlap`` chars to start the
      next chunk (never splits a word).
    - Hard-cuts a single token longer than ``size`` into ``size``-length pieces
      so the loop always makes forward progress.
    - Skips empty / whitespace-only results.
    """
    if not text or text.isspace():
        return []

    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0  # chars including single-space separators

    def flush():
        if current:
            chunk = " ".join(current).strip()
            if chunk and not chunk.isspace():
                chunks.append(chunk)

    i = 0
    while i < len(words):
        word = words[i]
        # Hard-cut over-long single tokens to guarantee forward progress.
        if len(word) > size:
            flush()
            current = []
            current_len = 0
            for j in range(0, len(word), size):
                piece = word[j:j + size]
                if piece and not piece.isspace():
                    chunks.append(piece)
            i += 1
            continue

        addition = len(word) if not current else len(word) + 1
        if current and current_len + addition > size:
            # Emit current chunk, then build overlap of whole words.
            flush()
            overlap_words: list[str] = []
            overlap_len = 0
            for w in reversed(current):
                add = len(w) if not overlap_words else len(w) + 1
                if overlap_len + add > overlap:
                    break
                overlap_words.insert(0, w)
                overlap_len += add
            current = overlap_words
            current_len = overlap_len
            # Do not advance i; retry adding this word to the new chunk.
            continue

        current.append(word)
        current_len += addition
        i += 1

    flush()
    return chunks


def get_image_description(client: OpenAI, image_bytes: bytes, mime: str, prompt: str) -> str | None:
    """Caption an image via OpenAI vision chat.

    Takes already-normalized bytes + MIME (conversion happens in
    ``build_or_load_index``). Returns the caption text, or a string containing
    "Error:" / "blocked" on failure so ``build_or_load_index``'s filter drops it.
    """
    try:
        b64 = base64.b64encode(image_bytes).decode("ascii")
    except Exception as e:  # noqa: BLE001
        print(f"\nWarning: Could not base64-encode image: {e}", file=sys.stderr)
        return f"Error: {type(e).__name__}"

    def _call():
        resp = client.chat.completions.create(
            model=VISION_MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                    ],
                }
            ],
        )
        return resp.choices[0].message.content

    content = with_retry(_call)
    if content is None:
        return "Error: image description failed"
    content = content.strip()
    if not content:
        return "Empty description generated"
    return content


# --- MIME handling for image captioning ---
# PyMuPDF ext -> MIME for formats OpenAI vision accepts directly.
_DIRECT_MIME = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
}
# Formats OpenAI vision does not accept -> convert with Pillow before captioning.
_CONVERT_EXTS = {"jpx", "jp2", "bmp", "tiff", "tif"}


def _normalize_image(image_bytes: bytes, ext: str) -> tuple[bytes, str, str] | None:
    """Return (bytes, mime, final_ext) ready for captioning, or None on failure.

    Directly-supported formats pass through. Unsupported formats (jpx/jp2/bmp/
    tiff) are re-encoded via Pillow to PNG (or JPEG for RGB/L without alpha).
    """
    ext = (ext or "").lower()
    if ext in _DIRECT_MIME:
        return image_bytes, _DIRECT_MIME[ext], ext
    if ext in _CONVERT_EXTS:
        try:
            with PIL_Image.open(io.BytesIO(image_bytes)) as im:
                buf = io.BytesIO()
                if im.mode in ("RGBA", "P", "LA"):
                    im.convert("RGBA").save(buf, format="PNG")
                    return buf.getvalue(), "image/png", "png"
                im.convert("RGB").save(buf, format="JPEG")
                return buf.getvalue(), "image/jpeg", "jpg"
        except Exception as e:  # noqa: BLE001
            print(f"\nWarning: Pillow conversion failed for .{ext}: {e}", file=sys.stderr)
            return None
    # Unknown format: attempt Pillow conversion to PNG as a last resort.
    try:
        with PIL_Image.open(io.BytesIO(image_bytes)) as im:
            buf = io.BytesIO()
            im.convert("RGBA").save(buf, format="PNG")
            return buf.getvalue(), "image/png", "png"
    except Exception:  # noqa: BLE001
        return None


# --- FAISS Index Building / Loading ---
def build_or_load_index(pdf_path_str: str, client: OpenAI, image_save_dir: str) -> tuple[list, list, int | None]:
    """Process a single PDF: extract text/images, embed, return metadata + embeddings.

    Returns (metadata_list, embeddings_list, detected_dimension).
    """
    pdf_file = Path(pdf_path_str)
    doc_basename = pdf_file.stem

    img_dir_path = Path(image_save_dir)
    img_dir_path.mkdir(parents=True, exist_ok=True)
    image_description_prompt = """
    Describe this image in detail. If it's a table, extract the data. If it's a chart, explain its key findings.
    If it's an image with text on it, try transcribing the text in great detail,
    so that it is useful for the user to understand the content as well.
    """

    embeddings_list = []
    metadata_list = []
    embedding_dimension = None

    def _add_item(embedding, meta):
        """Detect/verify dim, then append the embedding + metadata. Returns True if added."""
        nonlocal embedding_dimension
        if not embedding:
            return False
        if embedding_dimension is None:
            embedding_dimension = len(embedding)
            print(f"\n  [Detected Embedding Dim: {embedding_dimension}]")
        elif len(embedding) != embedding_dimension:
            print("\nError: Embed dim mismatch! Skipping.", file=sys.stderr)
            return False
        metadata_list.append(meta)
        embeddings_list.append(np.array(embedding, dtype=np.float32))
        return True

    print(f"\nProcessing document: {pdf_file.name}")
    try:
        doc = fitz.open(pdf_path_str)
        num_pages = len(doc)
        print(f"  - Found {num_pages} pages.")
        for page_num in range(num_pages):
            page = doc.load_page(page_num)
            page_id = page_num + 1
            print(f"\r  - Processing page {page_id}/{num_pages}...", end="")
            # 1. Process Text
            page_text = page.get_text("text")
            if page_text and not page_text.isspace():
                for chunk in chunk_text(page_text):
                    if not chunk or chunk.isspace():
                        continue
                    embedding = get_text_embedding(client, chunk)
                    _add_item(
                        embedding,
                        {"doc": doc_basename, "page": page_id, "type": "text", "content": chunk},
                    )

            # 2. Process Images
            image_list = page.get_images(full=True)
            if image_list:
                print(f" Found {len(image_list)} images... ", end="")
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    ext = base_image["ext"]
                    if len(image_bytes) < 5000:
                        continue
                    normalized = _normalize_image(image_bytes, ext)
                    if normalized is None:
                        print(
                            f"\nWarning: Could not normalize image p{page_id} img{img_index} (.{ext}); skipping.",
                            file=sys.stderr,
                        )
                        continue
                    norm_bytes, mime, final_ext = normalized
                    image_filename = f"{doc_basename}_p{page_id}_img{img_index}.{final_ext}"
                    image_path = img_dir_path / image_filename
                    with open(image_path, "wb") as img_file:
                        img_file.write(norm_bytes)
                    description = get_image_description(client, norm_bytes, mime, image_description_prompt)
                    if not description or "Error:" in description or "blocked" in description:
                        print(
                            f"\nWarning: Skipping image {image_filename} (desc issue): {description}",
                            file=sys.stderr,
                        )
                        continue
                    # Embed the description text
                    desc_embedding = get_text_embedding(client, description)
                    _add_item(
                        desc_embedding,
                        {
                            "doc": doc_basename,
                            "page": page_id,
                            "type": "image",
                            "content": description,
                            "image_path": str(image_path),
                        },
                    )
                except Exception as img_err:  # noqa: BLE001
                    print(f"\nError processing image {img_index} on page {page_id}: {img_err}", file=sys.stderr)
        print()  # Newline after document
    except Exception as e:  # noqa: BLE001
        print(f"Error processing document {pdf_path_str}: {e}", file=sys.stderr)

    if not metadata_list or not embeddings_list:
        print(f"Error: No content could be processed/embedded for {pdf_file.name}.", file=sys.stderr)
        return [], [], None

    print(f"  - Finished processing {pdf_file.name}. Found {len(metadata_list)} items.")
    return metadata_list, embeddings_list, embedding_dimension


# --- Reranking ---
def rerank_candidates(client: OpenAI, query: str, candidates: list[str], top_k: int) -> list[int]:
    """Rerank candidate chunks via OpenAI; FAISS-order fallback on any failure.

    Returns a list of candidate indices (most relevant first). On any parse /
    validation error (non-int, out-of-range, duplicate, incomplete list, or
    malformed JSON) or on timeout, returns ``list(range(min(top_k, len(
    candidates))))`` (FAISS order). A single ~RERANK_TIMEOUT_S deadline is
    enforced with the client's retries disabled (max_retries=0); no backoff.
    """
    n = len(candidates)
    fallback = list(range(min(top_k, n)))
    if n == 0:
        return []

    # Build a numbered list of full candidate texts. For unusually long chunks,
    # send a query-centered excerpt so the tail is not hidden.
    max_excerpt = 1200
    lines = []
    for idx, cand in enumerate(candidates):
        text = cand or ""
        if len(text) > max_excerpt:
            lowered = text.lower()
            q = (query or "").lower().strip()
            pos = lowered.find(q.split()[0]) if q else -1
            if pos < 0:
                excerpt = text[:max_excerpt]
            else:
                start = max(0, pos - max_excerpt // 2)
                excerpt = text[start:start + max_excerpt]
            text = excerpt
        lines.append(f"[{idx}] {text}")
    candidate_block = "\n".join(lines)

    prompt = (
        f"You are reranking passages for the question: {query!r}.\n"
        f"Below are {n} candidate passages, each prefixed with its index.\n\n"
        f"{candidate_block}\n\n"
        f"Return the {min(top_k, n)} most relevant passage indices, most relevant first, "
        f'as a JSON object of the form {{"indices": [int, ...]}}. '
        f"Use only integer indices from 0 to {n - 1}, with no duplicates."
    )

    def _call():
        # Disable the SDK's default retries (2) so the ~RERANK_TIMEOUT_S deadline
        # is a true single-attempt bound; on any failure we fall back to FAISS order.
        resp = client.with_options(
            max_retries=0, timeout=RERANK_TIMEOUT_S
        ).chat.completions.create(
            model=RERANK_MODEL_ID,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    try:
        content = _call()
    except Exception as e:  # noqa: BLE001 - includes timeouts
        print(f"\nWarning: rerank call failed ({type(e).__name__}); using FAISS order.", file=sys.stderr)
        return fallback

    if not content:
        return fallback
    required = min(top_k, n)
    try:
        data = json.loads(content)
        indices = data["indices"]
        if not isinstance(indices, list):
            return fallback
        seen = set()
        validated = []
        for x in indices:
            # Reject booleans and non-ints; ints must be in range and unique.
            if isinstance(x, bool) or not isinstance(x, int):
                return fallback
            if x < 0 or x >= n:
                return fallback
            if x in seen:
                return fallback
            seen.add(x)
            validated.append(x)
        # Require a complete list: exactly `required` unique valid indices.
        if len(validated) < required:
            return fallback
        return validated[:required]
    except Exception:  # noqa: BLE001 - any parse/validation error
        return fallback


# --- FAISS Retrieval ---
def _format_context(rows) -> str:
    context_parts = []
    for _, row in rows.iterrows():
        if row["type"] == "text":
            context_parts.append(
                f"[Text Context - Source: {row['doc']}, Pg {row['page']}]\n{row['content']}"
            )
        elif row["type"] == "image":
            img_name = os.path.basename(row.get("image_path", "Unknown Image"))
            context_parts.append(
                f"[Image Context - Source: {row['doc']}, Pg {row['page']}, Image: {img_name}]\n"
                f"Description: {row['content']}"
            )
        context_parts.append("---")
    return "\n".join(context_parts)


def find_relevant_context_faiss(query: str, faiss_index, metadata_db: pd.DataFrame, client: OpenAI, top_k: int) -> str:
    """Find relevant items via FAISS over-fetch + rerank; return formatted context."""
    print("Status: Finding relevant document context using FAISS...")
    if faiss_index is None or metadata_db is None or metadata_db.empty:
        return "No document context available."

    try:
        query_embedding = get_text_embedding(client, query)
        if query_embedding is None:
            return "Could not process query embedding."

        query_embedding_np = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_embedding_np)

        k = min(RAG_CANDIDATE_K, faiss_index.ntotal)
        if k <= 0:
            return "No relevant context found in documents."
        print(f" - Searching FAISS index (k={k})...")
        _, indices = faiss_index.search(query_embedding_np, k)

        if indices.size == 0:
            print(" - No relevant items found in index.")
            return "No relevant context found in documents."

        retrieved_indices = indices[0]
        retrieved_indices = retrieved_indices[retrieved_indices != -1]
        if len(retrieved_indices) == 0:
            print(" - No relevant items found in index (all results were invalid).")
            return "No relevant context found in documents."

        candidate_rows = metadata_db.iloc[retrieved_indices]
        candidate_texts = [str(c) for c in candidate_rows["content"].tolist()]

        print(f" - Reranking {len(candidate_texts)} candidates -> top {top_k}...")
        reranked = rerank_candidates(client, query, candidate_texts, top_k)
        # Map reranked positions back to metadata rows.
        selected_positions = reranked[:top_k]
        selected_rows = candidate_rows.iloc[selected_positions]

        print(f" - Top {len(selected_rows)} relevant items selected.")
        return _format_context(selected_rows)

    except Exception as e:  # noqa: BLE001
        print(f"Error searching FAISS index: {e}", file=sys.stderr)
        return "Error retrieving context from index."


# --- Realtime session payload (pure function; no clients/PyAudio/socket) ---
def build_session_payload(language: str, voice: str) -> dict:
    """Build the GA realtime session payload.

    Notes:
    - ``output_modalities`` is ``["audio"]`` only; audio already carries an
      output transcript, so ``["audio","text"]`` is invalid.
    - ``voice`` lives under ``audio.output.voice``.
    - ``model`` is fixed at connect time and must NOT appear in this payload.
    """
    instructions = (
        "You are a helpful voice assistant answering questions grounded ONLY in the "
        "results returned by the search_documents tool. Always call search_documents "
        "before answering. If the documents do not contain the answer, say that you "
        "could not find it in the documents. Do not invent facts. Keep a natural, "
        f"positive, conversational tone. Respond in {language}."
    )
    return {
        "type": "realtime",
        "output_modalities": ["audio"],
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "transcription": {"model": TRANSCRIBE_MODEL_ID, "language": iso_lang(language)},
                "turn_detection": None,
            },
            "output": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "voice": voice,
            },
        },
        "instructions": instructions,
        "tools": [
            {
                "type": "function",
                "name": "search_documents",
                "description": "Search the indexed PDF documents for passages relevant to the user's question.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ],
        "tool_choice": "required",
    }


# --- Main Application Class ---
class VoiceRAGMultimodalFAISS:
    def __init__(self, pdf_paths: list[str], language=DEFAULT_LANGUAGE, voice=DEFAULT_VOICE):
        self.language_code = language
        self.selected_voice = voice
        self.pdf_paths = pdf_paths
        self.faiss_index = None  # Combined FAISS index
        self.metadata_db = None  # Combined metadata DataFrame
        self.playback_queue = asyncio.Queue()
        self.p_audio = None  # PyAudio handle; created lazily in voice mode only.
        self.output_stream = None
        self.is_active = True
        self.client = None  # sync OpenAI client
        self.async_client = None  # AsyncOpenAI client for realtime
        # Saved transcript for the current turn (display / query fallback).
        self.last_transcript = ""
        self._validate_config()
        self._initialize_clients()
        print("--- OpenAI Voice RAG (FAISS - Realtime STT+LLM+TTS) ---")
        print(f"Output Voice: {self.selected_voice}, Language: {self.language_code}")
        print(f"Input: Mic ({INPUT_RATE} Hz), Output: Speaker ({OUTPUT_RATE} Hz)")
        print(f"Processing documents: {', '.join(os.path.basename(p) for p in pdf_paths)}")
        print("Status: Initializing...")

    def _validate_config(self):
        if self.selected_voice not in AVAILABLE_VOICES:
            print(
                f"Warning: Voice '{self.selected_voice}' not available; using '{DEFAULT_VOICE}'.",
                file=sys.stderr,
            )
            self.selected_voice = DEFAULT_VOICE
        if self.language_code not in SUPPORTED_LANGUAGES:
            print(f"Warning: Lang code '{self.language_code}' not known.", file=sys.stderr)

    def _initialize_clients(self):
        """Initialize OpenAI clients. PyAudio is NOT touched here."""
        try:
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            print("Status: OpenAI sync + async clients initialized.")
        except Exception as e:  # noqa: BLE001
            print(f"Error initializing clients: {e}", file=sys.stderr)
            sys.exit(1)

    def _ensure_audio(self):
        """Create the PyAudio handle on demand (voice mode only)."""
        if self.p_audio is None:
            import pyaudio  # imported lazily so --selfcheck/--rebuild never need it
            self.p_audio = pyaudio.PyAudio()

    async def setup_rag_database(self):
        """Load or build FAISS index and metadata for all specified PDFs."""
        if not self.client:
            print("Error: Clients not initialized.", file=sys.stderr)
            self.is_active = False
            return

        all_metadata = []
        all_embeddings = []
        final_dimension = None

        for pdf_path in self.pdf_paths:
            base_name = Path(pdf_path).stem
            faiss_file = Path(f"{base_name}{FAISS_INDEX_SUFFIX}")
            meta_file = Path(f"{base_name}{METADATA_SUFFIX}")

            if faiss_file.exists() and meta_file.exists():
                print(f"Status: Loading existing index for {os.path.basename(pdf_path)}...")
                need_rebuild = False
                try:
                    loaded_index = faiss.read_index(str(faiss_file))
                    loaded_meta = pd.read_parquet(meta_file)

                    # Stale-index safety: reject any index whose dim != EMBEDDING_DIM.
                    if loaded_index.d != EMBEDDING_DIM:
                        print(
                            f"Warning: Stale index dim for {pdf_path} "
                            f"({loaded_index.d} != {EMBEDDING_DIM}). Rebuilding...",
                            file=sys.stderr,
                        )
                        need_rebuild = True
                    elif loaded_index.ntotal != len(loaded_meta):
                        raise ValueError(
                            f"Index/metadata count mismatch for {pdf_path} "
                            f"({loaded_index.ntotal} vs {len(loaded_meta)})"
                        )
                    else:
                        # Verify image_path files exist; if any image row lacks a
                        # non-empty existing string path (missing column, None, NaN,
                        # empty, or a path that no longer exists), invalidate the
                        # ENTIRE per-PDF cache and rebuild (keeps ntotal==len(meta)).
                        image_rows = loaded_meta[loaded_meta["type"] == "image"]
                        if not image_rows.empty:
                            has_col = "image_path" in loaded_meta.columns
                            missing = 0
                            if not has_col:
                                missing = len(image_rows)
                            else:
                                for p in image_rows["image_path"].tolist():
                                    if not isinstance(p, str) or not p.strip() or not Path(p).exists():
                                        missing += 1
                            if missing:
                                print(
                                    f"Warning: {missing} image row(s) with missing/invalid "
                                    f"image_path for {pdf_path}; invalidating cache and rebuilding...",
                                    file=sys.stderr,
                                )
                                need_rebuild = True

                    if not need_rebuild:
                        if final_dimension is None:
                            final_dimension = loaded_index.d
                        elif loaded_index.d != final_dimension:
                            raise ValueError(
                                f"Inconsistent embedding dimensions found "
                                f"({loaded_index.d} vs {final_dimension})"
                            )
                        loaded_embeddings = loaded_index.reconstruct_n(0, loaded_index.ntotal)
                        all_metadata.append(loaded_meta)
                        all_embeddings.append(loaded_embeddings)
                        print(f"  - Loaded {loaded_index.ntotal} vectors.")
                        continue  # loaded successfully
                except Exception as e:  # noqa: BLE001
                    print(f"Warning: Failed to load index for {pdf_path}: {e}. Rebuilding...", file=sys.stderr)
                    need_rebuild = True

                if need_rebuild:
                    try:
                        faiss_file.unlink(missing_ok=True)
                        meta_file.unlink(missing_ok=True)
                    except OSError:
                        pass

            # Build if loading failed or files don't exist
            print(f"Status: Building index for {os.path.basename(pdf_path)}...")
            meta_list, embeds_list, dim = await asyncio.to_thread(
                build_or_load_index, pdf_path, self.client, IMAGE_SAVE_DIR
            )

            if not meta_list or not embeds_list or dim is None:
                print(f"Warning: Failed to process {pdf_path}, skipping.", file=sys.stderr)
                continue

            if final_dimension is None:
                final_dimension = dim
            elif dim != final_dimension:
                print(
                    f"Error: Inconsistent embedding dimensions ({dim} vs {final_dimension}). "
                    f"Skipping {pdf_path}.",
                    file=sys.stderr,
                )
                continue

            current_meta_df = pd.DataFrame(meta_list)
            current_embeddings_np = np.array(embeds_list).astype("float32")

            try:
                print(f"  - Saving index & metadata for {os.path.basename(pdf_path)}...")
                faiss.normalize_L2(current_embeddings_np)
                index_single = faiss.IndexFlatIP(final_dimension)
                index_single.add(current_embeddings_np)
                faiss.write_index(index_single, str(faiss_file))
                current_meta_df.to_parquet(meta_file)
                print("  - Saved successfully.")
            except Exception as e:  # noqa: BLE001
                print(f"Error saving index/metadata for {pdf_path}: {e}. Skipping.", file=sys.stderr)
                continue

            all_metadata.append(current_meta_df)
            all_embeddings.append(current_embeddings_np)

        if not all_metadata or not all_embeddings:
            print("Fatal Error: No data processed for any PDF.", file=sys.stderr)
            self.is_active = False
            return

        print("\nStatus: Combining data from all documents...")
        self.metadata_db = pd.concat(all_metadata, ignore_index=True)
        combined_embeddings_np = np.vstack(all_embeddings)

        if final_dimension is None or combined_embeddings_np.shape[1] != final_dimension:
            print("Fatal Error: Final dimension mismatch during combination.", file=sys.stderr)
            self.is_active = False
            return

        print(
            f"Status: Building final combined FAISS index "
            f"(Dim: {final_dimension}, Vectors: {combined_embeddings_np.shape[0]})..."
        )
        self.faiss_index = faiss.IndexFlatIP(final_dimension)
        self.faiss_index.add(combined_embeddings_np)
        print(f"Status: Combined FAISS index ready ({self.faiss_index.ntotal} total vectors).")

        if self.faiss_index.ntotal != len(self.metadata_db):
            print(
                f"FATAL ERROR: Final combined index/metadata mismatch! "
                f"({self.faiss_index.ntotal} vs {len(self.metadata_db)})",
                file=sys.stderr,
            )
            self.is_active = False
            return

    async def start(self):
        self.is_active = True
        await self.setup_rag_database()
        if not self.is_active:
            print("Exiting due to RAG setup failure.")
            return
        # Open audio devices only now (voice mode). --selfcheck/--rebuild never reach here.
        self._ensure_audio()
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self.connect_and_manage_session())
                tg.create_task(self.play_audio_output())
            print("Status: TaskGroup finished.")
        except* Exception as eg:
            print("\nError group exceptions occurred:", file=sys.stderr)
            for i, exc in enumerate(eg.exceptions):
                if isinstance(exc, asyncio.CancelledError):
                    print("- Task cancelled.")
                else:
                    print(f"- Exception {i + 1}/{len(eg.exceptions)}: {type(exc).__name__}: {exc}", file=sys.stderr)
        finally:
            await self.stop()

    async def stop(self):
        if not self.is_active:
            return
        print("\nStatus: Initiating shutdown...")
        self.is_active = False
        await asyncio.sleep(0.2)
        if getattr(self, "output_stream", None):
            print("Status: Waiting for playback queue...")
            try:
                await asyncio.wait_for(self.playback_queue.join(), timeout=1.0)
                print("Status: Playback queue empty.")
            except asyncio.TimeoutError:
                print("Warning: Timeout waiting playback.", file=sys.stderr)
            except Exception as q_err:  # noqa: BLE001
                print(f"Warning: Error joining queue: {q_err}", file=sys.stderr)
            try:
                if self.output_stream.is_active():
                    self.output_stream.stop_stream()
                self.output_stream.close()
                print("Status: Output stream closed.")
            except Exception as e:  # noqa: BLE001
                print(f"Warning: Error closing output stream: {e}", file=sys.stderr)
        if self.p_audio:
            self.p_audio.terminate()
            print("Status: PyAudio terminated.")
        print("--- OpenAI Voice RAG Multimodal Finished ---")

    def _clear_playback_queue(self):
        try:
            while True:
                self.playback_queue.get_nowait()
                self.playback_queue.task_done()
        except asyncio.QueueEmpty:
            pass

    async def _run_retrieval(self, query: str) -> str:
        """Run FAISS+rerank retrieval off the event loop."""
        return await asyncio.to_thread(
            find_relevant_context_faiss,
            query,
            self.faiss_index,
            self.metadata_db,
            self.client,
            RAG_TOP_K,
        )

    async def _handle_media_event(self, event, etype, state) -> bool:
        """Handle the media/error event branches shared by both turn consumers.

        ``state`` is a mutable dict carrying the accumulated ``transcript``.
        Returns True if the event was handled here; False otherwise. Raises
        RuntimeError on an ``error`` event so the per-turn recovery can reset.
        """
        if etype in ("response.output_audio.delta", "response.audio.delta"):
            delta = getattr(event, "delta", None)
            if delta:
                try:
                    await self.playback_queue.put(base64.b64decode(delta))
                except Exception:  # noqa: BLE001
                    pass
            return True

        if etype in (
            "response.output_audio_transcript.delta",
            "response.audio_transcript.delta",
        ):
            state["transcript"] += getattr(event, "delta", "") or ""
            if self.playback_queue.empty():
                print(f"   Assistant (Text): {state['transcript']}", end="\r", flush=True)
            return True

        if etype in (
            "response.output_audio_transcript.done",
            "response.audio_transcript.done",
        ):
            done_text = getattr(event, "transcript", None) or state["transcript"]
            if done_text:
                print(f"\n   Assistant: {done_text}", flush=True)
            return True

        if etype == "response.output_text.delta":
            state["transcript"] += getattr(event, "delta", "") or ""
            return True

        if etype == "error":
            err = getattr(event, "error", event)
            print(f"\nError event from realtime API: {err}", file=sys.stderr)
            raise RuntimeError(f"realtime error: {err}")

        return False

    async def _consume_turn(self, conn):
        """Consume events for a single turn via the tool-call state machine.

        Tracks the initial (tool-call) and continuation response ids and a
        ``tool_output_sent`` flag so a stale ``response.done`` from a prior
        failed/timed-out turn cannot be mistaken for this turn's completion:
        - Transition tool -> continuation only after the tool output was sent AND
          the completing ``response.done`` matches the initial response id (when
          the SDK exposes ids) and reports status == "completed" (when available).
        - End the turn only on a ``response.done`` matching the continuation
          response id.
        """
        state_name = "tool"
        pending_call_id = None
        tool_output_sent = False
        initial_response_id = None
        continuation_response_id = None
        ev_state = {"transcript": ""}

        async for event in conn:
            if not self.is_active:
                break
            etype = getattr(event, "type", "")

            if etype == "conversation.item.input_audio_transcription.completed":
                self.last_transcript = getattr(event, "transcript", "") or ""
                print(f"   >> You asked: {self.last_transcript}", flush=True)

            elif etype == "response.created":
                rid = _event_response_id(event)
                if state_name == "tool" and initial_response_id is None:
                    initial_response_id = rid
                elif state_name == "continuation" and continuation_response_id is None:
                    continuation_response_id = rid

            elif etype == "response.function_call_arguments.done":
                if state_name != "tool":
                    # Guard: continuation should not call the tool again.
                    continue
                query = _extract_query(getattr(event, "arguments", None), self.last_transcript)
                pending_call_id = getattr(event, "call_id", None)
                context = await self._run_retrieval(query)
                await conn.conversation.item.create(
                    item={
                        "type": "function_call_output",
                        "call_id": pending_call_id,
                        "output": context,
                    }
                )
                tool_output_sent = True

            elif etype == "response.done":
                rid = _event_response_id(event)
                status = _event_response_status(event)
                if state_name == "tool":
                    # Only advance once the tool output was sent and this done
                    # matches the initial response (id + status where available).
                    if not tool_output_sent:
                        continue
                    if initial_response_id is not None and rid is not None and rid != initial_response_id:
                        continue
                    if status is not None and status != "completed":
                        # Failed/incomplete tool response -> abort turn for recovery.
                        raise RuntimeError(f"tool response not completed (status={status})")
                    state_name = "continuation"
                    await conn.response.create(response={"tool_choice": "none"})
                else:
                    # Continuation done -> end the turn (match id when available).
                    if continuation_response_id is not None and rid is not None and rid != continuation_response_id:
                        continue
                    if status is not None and status not in ("completed", None):
                        raise RuntimeError(f"continuation response not completed (status={status})")
                    return

            else:
                await self._handle_media_event(event, etype, ev_state)

    async def _consume_turn_inject(self, conn):
        """Deterministic fallback turn (RAG_MODE='inject'): no tool call.

        On the input transcription completing, retrieve context, inject it as a
        user text message, then request a spoken answer with tool_choice='none'.
        The turn ends on the response.done for that answer (matched by id when
        the SDK exposes it).
        """
        state_name = "await_transcript"
        answer_response_id = None
        ev_state = {"transcript": ""}

        async for event in conn:
            if not self.is_active:
                break
            etype = getattr(event, "type", "")

            if etype == "conversation.item.input_audio_transcription.completed":
                self.last_transcript = getattr(event, "transcript", "") or ""
                print(f"   >> You asked: {self.last_transcript}", flush=True)
                context = await self._run_retrieval(self.last_transcript)
                await conn.conversation.item.create(
                    item={
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    f"Answer based ONLY on the context below.\n\n"
                                    f"CONTEXT:\n{context}\n\n"
                                    f"QUESTION: {self.last_transcript}"
                                ),
                            }
                        ],
                    }
                )
                state_name = "answer"
                await conn.response.create(response={"tool_choice": "none"})

            elif etype == "response.created":
                if state_name == "answer" and answer_response_id is None:
                    answer_response_id = _event_response_id(event)

            elif etype == "response.done":
                if state_name != "answer":
                    continue
                rid = _event_response_id(event)
                if answer_response_id is not None and rid is not None and rid != answer_response_id:
                    continue
                return

            else:
                await self._handle_media_event(event, etype, ev_state)

    async def _cancel_active_response(self, conn):
        """Best-effort cancel of any in-flight response after a failed turn."""
        try:
            await conn.response.cancel()
        except Exception:  # noqa: BLE001 - cancel may be unsupported or nothing active
            pass

    async def _drain_buffered_events(self, conn):
        """Drain any buffered events so a stale response.done can't leak into the
        next turn. Bounded by a short deadline; safe if nothing is queued."""
        try:
            await asyncio.wait_for(self._drain_loop(conn), timeout=1.0)
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001
            pass

    async def _drain_loop(self, conn):
        async for _ in conn:
            # Keep consuming until the transport briefly goes idle (timeout above
            # ends the drain); we intentionally discard these stale events.
            await asyncio.sleep(0)

    async def connect_and_manage_session(self):
        if self.client is None or self.metadata_db is None or self.faiss_index is None:
            print("Error: OpenAI Client or RAG DB not ready.", file=sys.stderr)
            self.is_active = False
            return

        print("Status: Connecting to OpenAI realtime API...")
        try:
            async with self.async_client.realtime.connect(model=REALTIME_MODEL_ID) as conn:
                await conn.session.update(
                    session=build_session_payload(self.language_code, self.selected_voice)
                )
                print("Status: Connected.")
                print(f"Status: RAG_MODE={RAG_MODE}")
                print("\n>>> Press ENTER to record, ENTER again to stop. (Ctrl+C to quit) <<<")

                while self.is_active:
                    await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                    if not self.is_active:
                        break
                    audio_frames = await self.record_audio()
                    if not audio_frames or not self.is_active:
                        continue

                    # Stream captured pcm16 frames to the input buffer.
                    audio_b64 = base64.b64encode(b"".join(audio_frames)).decode("ascii")
                    await conn.input_audio_buffer.append(audio=audio_b64)
                    await conn.input_audio_buffer.commit()

                    self.last_transcript = ""
                    try:
                        if RAG_MODE == "inject":
                            # Inject mode retrieves on the transcript; no initial tool call.
                            await asyncio.wait_for(
                                self._consume_turn_inject(conn), timeout=TURN_TIMEOUT_S
                            )
                        else:
                            # Tool mode: kick off the required tool-call response.
                            await conn.response.create()
                            await asyncio.wait_for(
                                self._consume_turn(conn), timeout=TURN_TIMEOUT_S
                            )
                    except asyncio.TimeoutError:
                        print("\nWarning: Turn timed out; resetting.", file=sys.stderr)
                        self.last_transcript = ""
                        await self._cancel_active_response(conn)
                        await self._drain_buffered_events(conn)
                        self._clear_playback_queue()
                    except RuntimeError as turn_err:
                        print(f"\nWarning: Turn aborted ({turn_err}); resetting.", file=sys.stderr)
                        self.last_transcript = ""
                        await self._cancel_active_response(conn)
                        await self._drain_buffered_events(conn)
                        self._clear_playback_queue()

                    if not self.is_active:
                        break
                    print("\n>>> Press ENTER to record... <<<")
        except asyncio.CancelledError:
            print("Status: Connection/Manage task cancelled.")
        except Exception as e:  # noqa: BLE001
            print(f"\nError in connect_and_manage: {type(e).__name__}: {e}", file=sys.stderr)
            self.is_active = False
            raise
        finally:
            print("Status: Exiting connect_and_manage loop.")

    async def record_audio(self):
        self._ensure_audio()
        if not self.is_active or self.p_audio is None:
            return None
        import pyaudio
        frames = []
        input_stream_obj = None
        print("[Recording... Press ENTER to stop]", end="", flush=True)
        try:
            input_stream_obj = self.p_audio.open(
                format=pyaudio.paInt16, channels=CHANNELS, rate=INPUT_RATE,
                input=True, frames_per_buffer=CHUNK_SIZE,
            )
            recording_task = asyncio.create_task(self._read_audio_chunks(input_stream_obj, frames))
            await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            print(" [Stopping Recording...]")
        except OSError as e:
            print(f"\nError opening input stream: {e}", file=sys.stderr)
            return None
        except Exception as e:  # noqa: BLE001
            print(f"\nError during recording setup: {e}", file=sys.stderr)
            return None
        finally:
            if "recording_task" in locals() and recording_task and not recording_task.done():
                recording_task.cancel()
                try:
                    await recording_task
                except asyncio.CancelledError:
                    pass
            if input_stream_obj:
                try:
                    input_stream_obj.stop_stream()
                    input_stream_obj.close()
                except Exception as close_e:  # noqa: BLE001
                    print(f"Warning: Error closing recording stream: {close_e}", file=sys.stderr)
        return frames if self.is_active else None

    async def _read_audio_chunks(self, stream, frames):
        try:
            while True:
                data = await asyncio.to_thread(stream.read, CHUNK_SIZE, exception_on_overflow=False)
                frames.append(data)
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            print(f"Error reading audio chunk: {e}", file=sys.stderr)

    async def play_audio_output(self):
        import pyaudio
        print("Status: Waiting for PyAudio to start output stream...")
        while self.p_audio is None and self.is_active:
            await asyncio.sleep(0.1)
        if not self.is_active or self.p_audio is None:
            print("Status: Playback task exiting.")
            return
        print("Status: Starting audio output stream...")
        try:
            self.output_stream = self.p_audio.open(
                format=pyaudio.paInt16, channels=CHANNELS, rate=OUTPUT_RATE, output=True
            )
            print("Status: Output stream opened.")
            while self.is_active:
                try:
                    data = await asyncio.wait_for(self.playback_queue.get(), timeout=1.0)
                    if data is None:
                        continue
                    await asyncio.to_thread(self.output_stream.write, data)
                    self.playback_queue.task_done()
                except asyncio.TimeoutError:
                    if not self.is_active:
                        break
                    continue
                except asyncio.CancelledError:
                    print("Status: Playback task inner cancelled.")
                    break
                except OSError as write_err:
                    print(f"Error writing to output stream: {write_err}", file=sys.stderr)
                    self.is_active = False
                    break
                except Exception as inner_e:  # noqa: BLE001
                    print(f"Error during playback loop: {type(inner_e).__name__}: {inner_e}", file=sys.stderr)
                    self.is_active = False
                    break
        except asyncio.CancelledError:
            print("Status: Playback task setup cancelled.")
        except OSError as e:
            print(f"\nPyAudio OSError in playback: {e}", file=sys.stderr)
            self.is_active = False
        except Exception as e:  # noqa: BLE001
            print(f"\nUnexpected error in playback: {e}", file=sys.stderr)
            self.is_active = False
            raise
        finally:
            print("Status: Exiting play_audio_output loop.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAI Realtime Voice RAG Multimodal w/ FAISS")
    parser.add_argument("pdf_paths", nargs="+", help="Path(s) to the PDF document(s) to use for RAG.")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, help=f"Input/Output language code (BCP-47). Default: {DEFAULT_LANGUAGE}")
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help=f"Output voice. One of: {', '.join(AVAILABLE_VOICES)}. Default: {DEFAULT_VOICE}",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="Print the realtime session payload JSON and exit (no clients, no audio, no network).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Build/save the FAISS indexes for the given PDFs and exit (no mic).",
    )
    args = parser.parse_args()

    lang_arg, voice_arg = args.language, args.voice
    if lang_arg not in SUPPORTED_LANGUAGES:
        print(f"Warning: Invalid language '{args.language}', using default '{DEFAULT_LANGUAGE}'.")
        lang_arg = DEFAULT_LANGUAGE
    if voice_arg not in AVAILABLE_VOICES:
        print(f"Warning: Invalid voice '{args.voice}', using default '{DEFAULT_VOICE}'.")
        voice_arg = DEFAULT_VOICE

    # --selfcheck: pure payload build; no clients / PyAudio / socket / PDFs required.
    if args.selfcheck:
        print(json.dumps(build_session_payload(lang_arg, voice_arg), indent=2))
        sys.exit(0)

    valid_paths = [p for p in args.pdf_paths if os.path.isfile(p) and p.lower().endswith(".pdf")]
    if not valid_paths:
        print("Error: No valid PDF documents provided.", file=sys.stderr)
        sys.exit(1)
    if len(valid_paths) != len(args.pdf_paths):
        print("Warning: Some invalid paths were skipped.", file=sys.stderr)

    client = VoiceRAGMultimodalFAISS(pdf_paths=valid_paths, language=lang_arg, voice=voice_arg)

    # --rebuild: build indexes only, then exit (no mic).
    if args.rebuild:
        async def _rebuild():
            await client.setup_rag_database()

        try:
            asyncio.run(_rebuild())
            print("Status: Rebuild complete." if client.is_active else "Status: Rebuild finished with errors.")
        except KeyboardInterrupt:
            print("\nRebuild cancelled by user (Ctrl+C).")
        sys.exit(0 if client.is_active else 1)

    try:
        print("Starting application...")
        asyncio.run(client.start())
    except KeyboardInterrupt:
        print("\nExecution cancelled by user (Ctrl+C).")
    except Exception as e:  # noqa: BLE001
        print(f"\nAn unexpected error occurred in main execution: {type(e).__name__}: {e}", file=sys.stderr)
    finally:
        print("Application exiting.")
