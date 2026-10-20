# Voice Study Companion

A Python voice assistant for asking questions about study PDFs. It is aimed at multilingual learners who benefit from hearing an explanation in a language they understand while being able to inspect the source page. The repository is a prototype, not a finished tutoring product.

The browser app is a continuous voice conversation. Upload up to five PDFs, wait for indexing, and connect the microphone once. OpenAI Realtime server-side VAD detects turns and interruptions; the browser streams audio, plays replies as they arrive, and shows a live transcript and retrieved source pages.

## What works now

- PDF text extraction and overlapping text chunks with PyMuPDF. Low-text image pages use optional Tesseract OCR, with configurable language data.
- Image extraction and OpenAI vision descriptions. Image descriptions are embedded as text; this is not native image-vector retrieval.
- OpenAI text embeddings, FAISS similarity search, and model-based reranking of the top candidates.
- OpenAI Realtime function calling through `search_documents` and `set_preferences`. Document answers are instructed to use only retrieved content and to say when an answer is absent.
- A live browser interface with one persistent Realtime connection, continuous mic streaming, server VAD, interruptible output, incremental transcript, and retrieved-page previews.
- Automatic answer-language selection based on each clear spoken request. Spoken commands can change the answer language or output voice.
- A read-only local glossary MCP tool for learner-friendly explanations of a few curated terms.
- Voice-led, one-question practice quizzes generated from retrieved PDF passages, with spoken answers and formative feedback.
- Passage-linked claim citations after a spoken document answer. A second model proposes claim/evidence pairs; the server accepts a link only when the claim occurs in the answer and the quoted evidence occurs verbatim in the retrieved passage.
- Per-turn server-side time to first audio, interrupted-response flags, missing transcription/audio flags, plus an offline scorer for human-labeled transcription, language, and VAD errors.

The live interface requires one **Connect** click for microphone permission; after that, there is no recording button or manual turn submission. The model detects when you stop speaking and can be interrupted while replying. Audio is streamed to OpenAI and is not saved by this app, though OpenAI API data policies still apply. Auto language behavior is model-inferred from speech, not a separately measured language detector. Citation links validate an exact quote against an indexed chunk, but semantic support is model-judged and still needs human evaluation.

## Requirements

- Python 3.11 or newer.
- An OpenAI API key with access to the models in [`config.py`](config.py).
- A microphone and speakers for voice use.
- A modern browser with microphone permission and AudioWorklet support. Use headphones to reduce echo.
- For scanned PDFs: a local Tesseract installation with `tessdata` for the document languages. PyMuPDF needs the language data; OCR does not call a separate cloud OCR service.

The app makes paid OpenAI API calls while indexing and answering. Indexing a new PDF can take time because each text chunk is embedded and extracted images are described. Reusing the same uploaded content loads its local index.

If a voice question reports `You need to install openai[realtime]`, reinstall with `python -m pip install -r requirements-web.txt`. The Realtime extra supplies the WebSocket dependency used for spoken turns.

For scanned documents, install Tesseract and its trained language files before indexing. Set `OCR_LANGUAGES` to a `+`-separated list of Tesseract codes, for example `eng+hin` for English and Hindi. Set `TESSDATA_PREFIX` to the directory containing `eng.traineddata` and `hin.traineddata` if PyMuPDF cannot find it. In PowerShell, run `$env:OCR_LANGUAGES = "eng+hin"` before starting the app. The default is `eng`. Missing OCR data produces an indexing error rather than silently claiming a scanned page was indexed. Selectable-text pages do not require OCR. Changing languages or OCR mode uses a new index cache; older cache files are left in `.local_data/`.

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
3. Ask a question aloud, then pause. Keep talking naturally for follow-ups; speak over an answer to interrupt it. The transcript and retrieved-page previews update during the conversation. After a document answer, linked citation badges may appear next to supported claims; select one to inspect its passage and page.
4. Say, for example, "Answer in Hindi", "Use the cedar voice", or "Follow my language automatically" to change preferences by voice. A voice change may briefly reconnect the call. **Mute mic** sends silence while connected; **Disconnect** ends the call.

The browser contains a small amount of JavaScript for microphone capture, audio playback, and the transcript. Python/FastAPI handles upload, indexing, retrieval, tool calls, and the OpenAI session. The API key never goes to the browser. Only local-origin access is intended; do not expose this prototype on a public host without authentication, rate limits, and session cleanup. On a shared machine, remove `.local_data/` when you no longer need uploaded documents or indexes.

For a vocabulary question, try "What does photosynthesis mean?" The local glossary has a few sample terms in English and Hindi. It explicitly reports when a term or language is not covered. PDF questions still use document search; a glossary meaning is not evidence for a claim about the uploaded PDF.

To practice, say "Quiz me on this PDF" or name a topic. The assistant asks one question aloud; answer aloud on the next turn. Say "skip" or "tell me the answer" to reveal it. The question generator must cite a retrieved page and include an exact quote found on that page before the question is accepted. The answer key stays in the local server session until you answer or skip. Grading uses a separate model call and is **formative automated feedback**, not a verified assessment. A new quiz request replaces any pending question. Reloading or re-indexing clears quiz state.

The live browser does not set a transcription language hint; this lets the model infer the input language. Spoken commands and automatic language behavior need a live multilingual walkthrough before being treated as validated.

The live Realtime connection retains conversation context while connected. On reconnect, a short text history helps resolve follow-up references. Every document question is instructed to run retrieval again; previous answers are not source evidence.

## How retrieval works

1. For each PDF, PyMuPDF extracts page text and images. An image-bearing page with fewer than 40 non-whitespace selectable characters is OCRed at 150 DPI. OCR text and ordinary text are split into overlapping chunks. Images are converted when needed, described by a vision model, and those descriptions are treated as searchable text. Embedded images are not natively understood by FAISS, and charts/diagrams depend on the accuracy of the generated description.
2. `text-embedding-3-small` embeds the text and image descriptions. Normalized vectors are saved in a FAISS index alongside page metadata in Parquet.
3. A question is embedded and FAISS retrieves up to 20 candidates. A model reranks them to five passages, falling back to FAISS order if reranking fails or times out.
4. For document questions, the Realtime model calls `search_documents`; Python runs retrieval and returns source-labeled context. For spoken setting changes, it calls `set_preferences`; a setting-only turn does not run retrieval. The model then streams the spoken answer. Server VAD determines when a user turn ends and interrupts replies when speech begins again.
5. After a document answer, a separate model proposes claim-to-passage citations. The server rejects nonexistent IDs and quotes not present in the indexed passage; the browser links accepted claims to passage excerpts and source-page previews. This adds a paid API call per cited answer and may take a moment after the audio finishes. The quote check is mechanical, not proof that the passage entails the claim. Image-description citations quote the generated description, not the original image pixels.

For standalone vocabulary questions, the model can call `lookup_glossary`. The browser app acts as an MCP client and launches [`study_mcp.py`](study_mcp.py) locally over stdio on demand. The server reads [`study_glossary.json`](study_glossary.json); it does not need a separate terminal, network endpoint, or API key. Only the glossary tool is exposed, and its content is separate from PDF evidence. The sample glossary is intentionally small, so expand its vetted explanations before presenting it as broad language support.

The browser stores uploaded PDFs and generated indexes under `.local_data/`. This directory and `.env` are ignored by Git. Files remain on the local machine until you delete them.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Unit tests cover chunking, OCR triggering and failure, rerank fallbacks, index validation, VAD configuration, PCM forwarding, Realtime transcript and tool events, citation quote validation, voice scoring, spoken preferences, quiz routing and evidence checks, and retrieved source metadata. They do not prove that local Tesseract, a live API session, microphone, or speaker works in your environment; perform a manual browser walkthrough for live verification.

## Retrieval evaluation

`eval_rag.py` evaluates the **production embedding, FAISS, and reranking path**, not a substitute retriever. It scores whether a question retrieves human-labeled *evidence pages*. It does not generate or grade answers.

Create a UTF-8 JSONL dataset with one answerable question per line. Use [`examples/eval_dataset.example.jsonl`](examples/eval_dataset.example.jsonl) for the schema only; it is not benchmark data. Each row needs a unique `id`, a `query`, and all `expected_pages` that genuinely support the answer. `doc` is the PDF filename without `.pdf`, and page numbers start at 1. Add an exact `quote` when possible: the dry run checks that the quote occurs on the labeled page (using local OCR for scanned pages). Optional `category` and `language` labels produce subgroup summaries. Have someone other than the question writer review the evidence labels before reporting scores. Keep private PDFs, labels, and generated reports under `.local_data/`, which Git ignores.

A useful first benchmark covers several document layouts, languages, and question types. Keep answerable retrieval questions separate from unanswerable/abstention tests. Freeze the PDF bytes and dataset before comparing pipeline changes; changing the questions between runs invalidates a before/after claim.

```bash
python eval_rag.py --dataset path/to/your_cases.jsonl --pdf path/to/biology.pdf --dry-run
python eval_rag.py --dataset path/to/your_cases.jsonl --pdf path/to/biology.pdf --ocr off --output .local_data/ocr_off.json
python eval_rag.py --dataset path/to/your_cases.jsonl --pdf path/to/biology.pdf --ocr auto --compare .local_data/ocr_off.json --output .local_data/ocr_on.json
```

Repeat `--pdf` to search multiple documents. `--dry-run` validates schema, page bounds, and quotes without OpenAI calls; scanned-page quote checks still need local Tesseract OCR. Full runs make paid embedding and reranking calls and can invoke the vision model during indexing. They store fingerprinted indexes under `.local_data/eval_cache/`; the fingerprint includes PDF hashes, models, chunk settings, OCR mode/languages, and pipeline version. A comparison requires identical dataset and PDF content hashes and the same top-k setting. The report records how many pages produced OCR chunks. If `--ocr off` yields no searchable content at all, it cannot produce a retrieval baseline; report that as a coverage failure instead of inventing a score.

Read `summary.faiss` and `summary.reranked` separately. FAISS shows initial ranking; reranked shows the five passages selected for the assistant. At k = 1, 3, or 5:

- `hit_at_k`: fraction of questions with at least one labeled evidence page among the first k distinct retrieved pages.
- `recall_at_k`: fraction of all labeled evidence pages retrieved by k, averaged over questions.
- `mrr_at_k`: mean reciprocal rank of the first labeled page; a first-place hit contributes 1, second place 1/2, and no hit contributes 0.
- `candidate_hit_rate`: fraction of questions with evidence anywhere in the FAISS candidate pool before reranking.

The report also includes mean embedding, FAISS, reranking, and total retrieval time; per-question candidate scores and selected pages; and breakdowns by `category` and `language`. Repeated chunks from one page count once. The diagnostic `outcome` distinguishes `candidate_miss` (FAISS never found evidence), `not_promoted` (evidence was a candidate but not in FAISS top five or final top five), `rerank_drop` (FAISS top five had evidence but reranking lost it), `late_rank` (final top five had evidence but not first), and `top1_hit`. Inspect misses manually before changing the pipeline; OCR is only one possible cause.

These are **page-level retrieval metrics**, not answer correctness, claim groundedness, citation accuracy, multilingual speech quality, or abstention. No human-reviewed benchmark or measured score is included in this repository. Do not put results from the illustrative JSONL or a tiny self-labeled smoke set on a resume; report the dataset size, document mix, label method, metrics, and limitations alongside any eventual score.

## Voice evaluation

While a call is open, the footer shows server-side time from VAD `speech_stopped` to the first outgoing audio chunk. `GET /api/sessions/{session_id}/metrics` returns per-turn timings, interrupted-response flags, and missing transcription/audio flags in memory. It does not store raw audio or transcripts. The timer excludes browser playback buffering and is not end-to-end perceived latency. Session metrics disappear when the process exits.

Use [`examples/voice_eval.example.jsonl`](examples/voice_eval.example.jsonl) as a schema example, not as a real score. For representative, consented turns, manually enter a reference transcription, the observed transcription, expected and observed answer language, and whether VAD split or merged a turn incorrectly. Copy timings and interruption flags from the local metrics endpoint when relevant. Then run:

```bash
python eval_voice.py --dataset path/to/labeled_voice_turns.jsonl --output .local_data/voice_report.json
```

The report computes word error rate, answer-language accuracy, turn-detection error rate, interruption count, and p50/p95 server-side time to first audio. Metrics without ground-truth labels are `null`, not guessed. Keep voice labels private; avoid including children's recordings or identifying transcripts in Git.

## Current limits and next steps

- Auto language behavior, spoken preferences, and quizzes have not been measured for multilingual accuracy or tested end to end with the live API. They depend on the model interpreting speech correctly. Quiz question/answer quality and grading consistency need their own labeled evaluation.
- The citation auditor verifies that quoted evidence exists in an indexed passage, but grounded claim accuracy and citation coverage still need human review.
- OCR has not been run end to end on this machine because Tesseract language data is not installed. Multilingual OCR needs the corresponding traineddata files. Complicated tables and charts remain model-description dependent.
- The retrieval evaluation runner is available, but no human-reviewed benchmark or measured scores are checked in. Answer correctness, unsupported claims, language behavior, end-to-end latency, and cost still need evaluation.
- WebRTC is deferred until a WebSocket latency/interruption baseline has been measured. The current browser uses a server-proxied WebSocket and keeps the API key server-side. A WebRTC migration must preserve server-side document tools and be judged against the same tests, not assumed faster.
- The first terminology MCP integration is local and read-only. Additional document-source integrations and write-capable learning tools remain planned; they will need explicit privacy and approval rules.

This project should be presented as a working prototype with measurable next steps, not as a validated learning aid.
