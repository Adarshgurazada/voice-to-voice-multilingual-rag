# Voice Study Companion

A Python voice assistant for asking questions about study PDFs. It is aimed at multilingual learners who benefit from hearing an explanation in a language they understand while being able to inspect the source page. The repository is a prototype, not a finished tutoring product.

The browser app is a continuous voice conversation. Upload up to five PDFs, wait for indexing, and connect the microphone once. OpenAI Realtime server-side VAD detects turns and interruptions; the browser streams audio, plays replies as they arrive, and shows a live transcript and retrieved source pages.

## What works now

- PDF text extraction and overlapping text chunks with PyMuPDF.
- Image extraction and OpenAI vision descriptions. Image descriptions are embedded as text; this is not native image-vector retrieval.
- OpenAI text embeddings, FAISS similarity search, and model-based reranking of the top candidates.
- OpenAI Realtime function calling through `search_documents` and `set_preferences`. Document answers are instructed to use only retrieved content and to say when an answer is absent.
- A live browser interface with one persistent Realtime connection, continuous mic streaming, server VAD, interruptible output, incremental transcript, and retrieved-page previews.
- Automatic answer-language selection based on each clear spoken request. Spoken commands can change the answer language or output voice.
- A read-only local glossary MCP tool for learner-friendly explanations of a few curated terms.
- Voice-led, one-question practice quizzes generated from retrieved PDF passages, with spoken answers and formative feedback.

The live interface requires one **Connect** click for microphone permission; after that, there is no recording button or manual turn submission. The model detects when you stop speaking and can be interrupted while replying. Audio is streamed to OpenAI and is not saved by this app, though OpenAI API data policies still apply. Auto language behavior is model-inferred from speech, not a separately measured language detector. Retrieved pages are candidate evidence, not verified sentence-by-sentence citations.

## Requirements

- Python 3.11 or newer.
- An OpenAI API key with access to the models in [`config.py`](config.py).
- A microphone and speakers for voice use.
- A modern browser with microphone permission and AudioWorklet support. Use headphones to reduce echo.

The app makes paid OpenAI API calls while indexing and answering. Indexing a new PDF can take time because each text chunk is embedded and extracted images are described. Reusing the same uploaded content loads its local index.

If a voice question reports `You need to install openai[realtime]`, reinstall with `python -m pip install -r requirements-web.txt`. The Realtime extra supplies the WebSocket dependency used for spoken turns.

## Browser quickstart

```bash
git clone https://github.com/Adarshgurazada/voice-to-voice-multilingual-rag.git
cd voice-to-voice-multilingual-rag
git switch wip/voice-study-companion
python -m venv .venv
```

Activate the environment, then install the browser dependencies:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-web.txt
```

```bash
# macOS or Linux
source .venv/bin/activate
python -m pip install -r requirements-web.txt
```

Create a `.env` file in the repository root:

```dotenv
OPENAI_API_KEY=your_api_key_here
```

Start the live browser app:

```bash
python live_app.py
```

Open `http://127.0.0.1:8000`. The app binds to localhost and does not create a public share link.

1. Choose one to five PDFs, each at most 25 MB, and select **Index documents**.
2. Wait until indexing says `Documents ready`, then select **Connect** and allow microphone access.
3. Ask a question aloud, then pause. Keep talking naturally for follow-ups; speak over an answer to interrupt it. The transcript and retrieved-page previews update during the conversation.
4. Say, for example, "Answer in Hindi", "Use the cedar voice", or "Follow my language automatically" to change preferences by voice. A voice change may briefly reconnect the call. **Mute mic** sends silence while connected; **Disconnect** ends the call.

The browser contains a small amount of JavaScript for microphone capture, audio playback, and the transcript. Python/FastAPI handles upload, indexing, retrieval, tool calls, and the OpenAI session. The API key never goes to the browser. Only local-origin access is intended; do not expose this prototype on a public host without authentication, rate limits, and session cleanup. On a shared machine, remove `.local_data/` when you no longer need uploaded documents or indexes.

For a vocabulary question, try "What does photosynthesis mean?" The local glossary has a few sample terms in English and Hindi. It explicitly reports when a term or language is not covered. PDF questions still use document search; a glossary meaning is not evidence for a claim about the uploaded PDF.

To practice, say "Quiz me on this PDF" or name a topic. The assistant asks one question aloud; answer aloud on the next turn. Say "skip" or "tell me the answer" to reveal it. The question generator must cite a retrieved page and include an exact quote found on that page before the question is accepted. The answer key stays in the local server session until you answer or skip. Grading uses a separate model call and is **formative automated feedback**, not a verified assessment. A new quiz request replaces any pending question. Reloading or re-indexing clears quiz state.

The live browser does not set a transcription language hint; this lets the model infer the input language. Spoken commands and automatic language behavior need a live multilingual walkthrough before being treated as validated.

The live Realtime connection retains conversation context while connected. On reconnect, a short text history helps resolve follow-up references. Every document question is instructed to run retrieval again; previous answers are not source evidence.

## How retrieval works

1. For each PDF, PyMuPDF extracts page text and images. Text is split into overlapping chunks. Images are converted when needed, described by a vision model, and those descriptions are treated as searchable text.
2. `text-embedding-3-small` embeds the text and image descriptions. Normalized vectors are saved in a FAISS index alongside page metadata in Parquet.
3. A question is embedded and FAISS retrieves up to 20 candidates. A model reranks them to five passages, falling back to FAISS order if reranking fails or times out.
4. For document questions, the Realtime model calls `search_documents`; Python runs retrieval and returns source-labeled context. For spoken setting changes, it calls `set_preferences`; a setting-only turn does not run retrieval. The model then streams the spoken answer. Server VAD determines when a user turn ends and interrupts replies when speech begins again.

For standalone vocabulary questions, the model can call `lookup_glossary`. The browser app acts as an MCP client and launches [`study_mcp.py`](study_mcp.py) locally over stdio on demand. The server reads [`study_glossary.json`](study_glossary.json); it does not need a separate terminal, network endpoint, or API key. Only the glossary tool is exposed, and its content is separate from PDF evidence. The sample glossary is intentionally small, so expand its vetted explanations before presenting it as broad language support.

The browser stores uploaded PDFs and generated indexes under `.local_data/`. This directory and `.env` are ignored by Git. Files remain on the local machine until you delete them.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Unit tests cover chunking, rerank fallbacks, index validation, VAD configuration, PCM forwarding, Realtime transcript and tool events, spoken preferences, quiz routing and evidence checks, and retrieved source metadata. They do not prove that a live API session, microphone, or speaker works in your environment; perform a manual browser walkthrough for live verification.

## Retrieval evaluation

`eval_rag.py` evaluates the **same embedding, FAISS, and reranking path** used by the app. Prepare a JSONL file with one answerable question per line, a unique `id`, and all pages that contain evidence for the answer. Use [`examples/eval_dataset.example.jsonl`](examples/eval_dataset.example.jsonl) as a format example, not as benchmark data. Each `doc` is the PDF filename *without* `.pdf`; pages are one-based. Use your own PDFs and manually verified page labels. Keep private or copyrighted PDFs and their evaluation questions outside Git if necessary.

```bash
python eval_rag.py --dataset path/to/your_cases.jsonl --pdf path/to/biology.pdf --dry-run
python eval_rag.py --dataset path/to/your_cases.jsonl --pdf path/to/biology.pdf
```

Repeat `--pdf` to search several documents. The dry run validates JSONL syntax, IDs, labels, and PDF paths without an API key or API calls. A full run loads or builds indexes in `.local_data/`, checks that labeled pages have indexed content, and makes paid embedding/reranking calls. The default report is `.local_data/eval_report.json`; use `--output` to choose another location. It contains per-question retrieved pages, Hit@5, Recall@5, reciprocal rank, and retrieval time, plus macro averages. Retrieved chunks from the same page count once. The report also records SHA-256 hashes of the dataset/PDFs and the configured model IDs. Results depend on the PDFs, model versions, and index state; keep those fixed when comparing changes. The current index cache checks embedding dimensions but does not track embedding model version, so rebuild cached indexes if you change that model.

These are **page-level retrieval** metrics, not answer correctness, groundedness, multilingual speech quality, or abstention metrics. The example file is not a scored dataset. A credible benchmark needs a versioned, human-labeled question set with multiple topics and languages, including a separate set of unanswerable questions for later answer-behavior evaluation.

## Current limits and next steps

- Auto language behavior, spoken preferences, and quizzes have not been measured for multilingual accuracy or tested end to end with the live API. They depend on the model interpreting speech correctly. Quiz question/answer quality and grading consistency need their own labeled evaluation.
- Page previews show retrieved pages. Grounded answer claims and exact citation accuracy still need evaluation.
- The PDF pipeline may miss scanned text, complicated tables, and chart details. OCR and table-aware extraction are candidates for measured improvements.
- The retrieval evaluation runner is available, but no project-specific labeled benchmark or measured scores are checked in. Answer correctness, unsupported claims, language behavior, end-to-end latency, and cost still need evaluation.
- The first terminology MCP integration is local and read-only. Additional document-source integrations and write-capable learning tools remain planned; they will need explicit privacy and approval rules.

This project should be presented as a working prototype with measurable next steps, not as a validated learning aid.
