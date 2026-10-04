# Voice Study Companion

A Python voice assistant for asking questions about study PDFs. It is aimed at multilingual learners who benefit from hearing an explanation in a language they understand while being able to inspect the source page. The repository is a prototype, not a finished tutoring product.

The browser app uses Gradio. It accepts up to five PDFs, records a spoken question, retrieves relevant passages, and returns a spoken answer with a transcript and previews of the pages retrieved for that turn. The original terminal voice app remains available.

## What works now

- PDF text extraction and overlapping text chunks with PyMuPDF.
- Image extraction and OpenAI vision descriptions. Image descriptions are embedded as text; this is not native image-vector retrieval.
- OpenAI text embeddings, FAISS similarity search, and model-based reranking of the top candidates.
- OpenAI Realtime function calling through `search_documents`. The answer is instructed to use only retrieved document content and to say when an answer is absent.
- A Gradio browser interface with PDF upload, indexing, recorded microphone questions, answer audio, transcripts, and retrieved-page previews.
- A command-line push-to-talk interface using PyAudio.

The browser interface is **turn-based**: recording stops before the question is sent, and playback begins after the answer is complete. It is not yet a continuous, interruptible voice call. Language and voice are currently chosen with UI controls; automatic language detection and spoken settings are planned. Retrieved pages are candidate evidence, not verified sentence-by-sentence citations.

## Requirements

- Python 3.11 or newer.
- An OpenAI API key with access to the models in [`config.py`](config.py).
- A microphone and speakers for voice use.
- A browser with microphone permission for the Gradio app.

The app makes paid OpenAI API calls while indexing and answering. Indexing a new PDF can take time because each text chunk is embedded and extracted images are described. Reusing the same uploaded content loads its local index.

## Browser quickstart

```bash
git clone https://github.com/Adarshgurazada/voice-to-voice-multilingual-rag.git
cd voice-to-voice-multilingual-rag
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

Start the local browser app:

```bash
python gradio_app.py
```

Open `http://127.0.0.1:7860`. The app binds to localhost and does not create a public share link.

1. Choose one to five PDFs, each at most 25 MB, and select **Index documents**.
2. When the status says `Ready`, record a question and stop recording to submit it.
3. Listen to the answer and read its transcript. Select a retrieved page to inspect it.
4. Choose the question/answer language and voice in the controls if needed, then record the next question. The language is also passed as a transcription hint.

The interface keeps a short text history for resolving follow-up references. Every document question still runs retrieval again; previous answers are not treated as source evidence.

## Command-line mode

Install the CLI audio dependency in the same environment:

```bash
python -m pip install -r requirements-cli.txt
```

PyAudio may require PortAudio system libraries on some platforms. Then run:

```bash
python module_v2v_rag.py path/to/document.pdf --language en-US --voice alloy
```

Press Enter to start recording and Enter again to stop. Use `Ctrl+C` to exit. `--selfcheck` prints the Realtime session configuration without making API calls, and `--rebuild` rebuilds indexes without microphone use. See `python module_v2v_rag.py --help` for all options.

## How retrieval works

1. For each PDF, PyMuPDF extracts page text and images. Text is split into overlapping chunks. Images are converted when needed, described by a vision model, and those descriptions are treated as searchable text.
2. `text-embedding-3-small` embeds the text and image descriptions. Normalized vectors are saved in a FAISS index alongside page metadata in Parquet.
3. A question is embedded and FAISS retrieves up to 20 candidates. A model reranks them to five passages, falling back to FAISS order if reranking fails or times out.
4. The Realtime model calls `search_documents`; Python runs retrieval and returns source-labeled context. The model then generates the spoken answer.

The browser stores uploaded PDFs and generated indexes under `.local_data/`. This directory and `.env` are ignored by Git. Files remain on the local machine until you delete them. The terminal app uses index files in the current working directory and extracted images in `extracted_images/`.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Unit tests cover chunking, rerank fallbacks, index validation, the Realtime tool state machine, browser audio conversion, content-based upload naming, and retrieved source metadata. They do not prove that a live API session, microphone, or speaker works in your environment. A separate live smoke test requires `RUN_REALTIME_IT=1` and an API key; it is not part of the default suite.

## Current limits and next steps

- The browser uses a manually selected answer language. Automatic input language detection, spoken preference changes, and quizzes are future features.
- Page previews show retrieved pages. Grounded answer claims and exact citation accuracy still need evaluation.
- The PDF pipeline may miss scanned text, complicated tables, and chart details. OCR and table-aware extraction are candidates for measured improvements.
- The project has component tests but no versioned RAG benchmark yet. Planned evaluation covers retrieval recall, answer correctness, unsupported claims, language behavior, latency, and cost.
- MCP document-source and terminology integrations are planned. No MCP service is connected in this version.

This project should be presented as a working prototype with measurable next steps, not as a validated learning aid.
