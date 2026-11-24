# Voice-to-Voice Multilingual RAG, with receipts

Talk to your PDFs out loud, in English, Telugu or Hindi, and hear the answer in the language you
asked in. Every answer is built to be traceable to a page of the source, and the repository ships
the evaluation harness and measured results that show how often that holds.

> **Status: working prototype with a small, honest evaluation.** The numbers below come from a
> 32-question development set I labeled myself over four scanned PDFs. They show what each change
> did on that set; they are not a validated benchmark. See [Limitations](#limitations).

## What it does

1. **Upload up to five PDFs** (text or scanned) in the browser. They are read (text layer, OCR, or
   image captions), chunked, embedded and indexed locally with FAISS.
2. **Click Connect once and just talk.** A single OpenAI Realtime session handles speech-in and
   speech-out with server-side voice-activity detection, so you can interrupt it mid-sentence.
3. **The assistant searches your documents through a tool call** before it answers. Retrieved pages
   appear next to the live transcript, with excerpts and page previews.
4. **Answer language is yours to control.** The *Answer language* dropdown defaults to **Auto**,
   which follows the language you speak (decided from the script of the transcript, not the voice
   model's guess). Pick a language and answers stay in it, however you speak; saying "answer in
   Hindi" does the same and updates the dropdown.

The spoken answer is built from claims that passed the grounded-answer verifier
(`grounded_answer.py`, see [Architecture](#architecture)), not from raw passages.

## Architecture

```mermaid
flowchart LR
    subgraph Ingest["Indexing (per PDF, cached)"]
        P[PDF] --> T{Text layer?}
        T -- yes --> C[Section-aware chunks<br/>headings, table rows, sentences]
        T -- no --> O[Tesseract OCR<br/>or vision-model page read]
        O --> C
        P --> I[Embedded images -> vision captions]
        I --> E
        C --> E[Batch embeddings<br/>text-embedding-3-small]
        E --> F[(FAISS inner-product index<br/>+ parquet metadata)]
    end

    subgraph Retrieve["Retrieval"]
        Q[Question] --> FU{Follow-up?}
        FU -- yes --> RW[Rewrite from the last turns]
        FU -- no --> X
        RW --> X{Non-English?}
        X -- yes --> TR[English translation]
        X -- no --> S
        TR --> S[Search original + English<br/>top 20 each]
        X -.-> BM[BM25 keyword search]
        S --> RRF[Reciprocal-rank fusion]
        BM --> RRF
        RRF --> RR[LLM rerank -> top 5]
    end

    F --> S
    RR --> A

    subgraph Voice["Live voice (WebSocket)"]
        M((Mic)) --> RT[OpenAI Realtime<br/>server VAD + barge-in]
        RT -->|search_documents tool| SRV[FastAPI server]
        SRV --> Q
        A[Passages with source IDs] --> RT
        RT --> SP((Speaker))
        SRV --> UI[Transcript + sources + metrics]
    end
```

Key design points:

- **Server-side tools.** The browser never sees the API key; the server runs retrieval and sends
  results back to the Realtime session. A failing tool returns an error to the model instead of
  ending the call.
- **Two-stage retrieval.** Dense search (fused with BM25 keyword search) over-fetches 20 candidates; an LLM reranker picks the final
  5 with strict output validation and a FAISS-order fallback on timeout (single 4 s deadline).
- **Cross-language bridge.** Dense retrievers favour same-language passages, so a Hindi question can
  miss the English page that answers it. Non-English queries are also translated to English, both
  rankings are searched and fused with reciprocal-rank fusion.
- **Language chosen by script.** The transcript arrives about 2 s after the model has already
  called its tool, so the server waits (bounded, overlapped with retrieval) for it and maps its
  writing system to a language. This replaced trusting the voice model's own guess, which once
  answered Telugu in Malayalam.
- **Receipts.** `search_documents` runs the grounded answer pipeline (below) and gives the voice
  model only claims whose quotes were verified in the cited passage (or a fixed "not found"). The UI
  shows only those verified sources, and a post-hoc citation auditor links the spoken answer to
  exact quotes. The model's wording is still its own: see Limitations.

### Grounded answer pipeline (evaluated on text, used in the voice path)

`grounded_answer.py` turns retrieved passages into an answer that can be audited:

1. The model drafts an answer as **claims**, each with the passage ID and an exact **quote**.
2. **Quote check:** a claim survives only if its quote occurs in the cited passage (exact, or at
   least 90% character match to tolerate OCR hyphenation and typos).
3. **Support check (optional):** a second model call asks whether each quoted claim is actually
   *supported* by the passage. Quote presence alone proved insufficient: a fabricated "translated
   into English in 1804" claim carried a real quote from an unrelated passage.
4. The final answer is **rebuilt from verified claims only**. If none survive, the answer is a fixed
   "I couldn't find that in your documents" in the question's language (en/te/hi/bn).

### Study-companion layer

The goal is a companion that explains what a document means, not one that reads passages back.

- **Section-aware index** (`doc_structure.py`). The PDF layout (positions, font sizes, boldness) is read to
  drop repeated headers and footers, build a section path such as
  `PART 2 · Selling › SECTION 5 · Handling objections`, split at sentence boundaries, and turn
  unruled tables into one chunk per row keyed by the column headings (including a "what it often means"
  column, which is where the intent lives). The section path is embedded with every chunk and shown next to
  every source.
- **Retrieval that reaches the whole answer.** Section titles are matched against the question, and
  list-style questions ("all the objections", asked by the voice model or heard) pull in the whole matching
  section; other questions add the neighbouring chunks. The learner's own words are searched as well as the
  model's often terse query.
- **Hybrid search and follow-ups.** BM25 keyword search (`lexical_search.py`) is fused with the embedding
  search, and a message like "which village was he from?" is rewritten into a standalone question from the
  last three turns before searching (`HYBRID_SEARCH`, `FOLLOWUP_REWRITE`). Measured effects are in
  `benchmarks/RESULTS.md`: rewriting fixed every follow-up that lost its subject; hybrid search raised
  Hit@3 from 0.93 to 0.99 but not Hit@1.
- **Evidence cards.** Every verified claim becomes a card with its role (answer, definition, step, why,
  example, caution), the exact quote, where it sits in the document, and the page image with that quote
  highlighted (`highlight.py`). The companion's own plain-words reading, when it passes the check, is shown in a
  separate dashed card labelled as not quoted from the document.
- **Teaching-style answers** (`grounded_answer.py`). Every verified claim has a role (answer, definition,
  step, reason, example, caution) and the lesson follows that order. A reason is only spoken when the
  document states it; otherwise the companion says the document does not. An optional "in plain words"
  reading is used only if a second check finds that it adds no fact, number, name, step or reason.
- **Depth.** A brief reply gives the direct answer and the few points that matter, and offers more; a
  detailed reply ("explain", "why", "step by step") walks through the steps and their reasons.

## Results

All numbers are from the local development set in `benchmarks/` (32 questions, four scanned PDFs:
an English philosophy text, a Telugu textbook lesson, a Hindi science chapter and a Bengali poem
page). The PDFs are not in this repo (copyright); bring your own and relabel. Full tables, caveats
and reproduction commands are in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

### Retrieval (page-level)

"Reranked" is what the assistant actually sees. Hit@k means a labeled evidence page is among the
first k distinct pages.

| Configuration | Candidate recall | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Time |
|---|---|---|---|---|---|---|
| Tesseract OCR, no query expansion (baseline) | 0.938 | 0.875 | 0.938 | 0.938 | 0.901 | 1.4 s |
| **Tesseract OCR + English query expansion (shipped)** | **1.000** | **0.906** | **0.969** | **0.969** | **0.932** | 1.7 s |
| Vision-model page reading, no expansion | 0.875 | 0.812 | 0.844 | 0.844 | 0.828 | 1.4 s |
| Vision-model page reading + expansion | 0.938 | 0.906 | 0.906 | 0.906 | 0.906 | 1.6 s |

- LLM reranking lifted Hit@1 from 0.625 (FAISS order) to 0.875 on the baseline.
- Query expansion fixed both Telugu and Hindi questions about the English text, which previously
  retrieved the Telugu and Hindi documents instead.
- Vision-model page reading is available (`--ocr vision`) but did not win overall: it fixed one
  Telugu question and lost two others.

### Answer-level grounding (24 answerable, 8 unanswerable questions)

A separate, stronger judge model (`gpt-5.5`) scores correctness against a reference answer and
checks every proposed claim against its full cited passage. **The judge is a model, not ground
truth, and is not yet calibrated against human labels.**

| Run | Abstained on unanswerable | False abstain | Correct | Shipped claims supported | Answer language correct |
|---|---|---|---|---|---|
| Quote check only | 7/8 | 2/24 | 21/24 | 30/32 | 100% |
| Quote + support check, run 1 | 8/8 | 3/24 | 19/24 | 27/28 | 100% |
| Quote + support check, run 2 (repeat) | 6/8 | 2/24 | 18/24 | 29/31 | 100% |

What I take from it:

- **Run-to-run noise is as large as the effect.** The same configuration abstained on 8/8 and 6/8
  unanswerable questions in two runs. I do not claim the support check improves abstention until the
  set is larger and runs are repeated.
- **Answer language went from 77% to 100%** once the target language was passed explicitly. Before,
  the model copied source-language quotes into its answer.
- **Grounding is only as good as the index.** An elephant life-span question (unanswerable; the
  figure leaves it blank) is answered with the crocodile's number because the image *caption* in the
  index misattributed the labels. The cited passage "supports" the claim; the index itself is wrong.
- **OCR noise causes false abstentions.** The model "corrects" typos (e.g. `7imaeus` to `Timaeus`)
  inside its quote, and Telugu OCR is often garbled.
- The support check adds about 0.7 s per answer.

### Section-aware index on a private 66-page sales playbook

The playbook cannot be shared, so this is a local check, not a benchmark. Same questions, same retrieval
code, old plain-chunk index versus the new one. "Found" counts how many expected facts reach the answer step.

| Question | Index | Passages | Characters | Passages starting with the page header | Expected facts found |
|---|---|---|---|---|---|
| List: all the objections (and what each means) | plain chunks | 5 | 3,984 | 3 of 5 | 13 of 24 |
| | **section-aware** | 12 | 3,569 | **0 of 12** | **22 of 24** |
| Explain one objection and why | plain chunks | 5 | 3,988 | 3 of 5 | 5 of 5 |
| | section-aware | 7 | 2,424 | 0 of 7 | 5 of 5 |
| Fact: is our data safe? | plain chunks | 5 | 3,979 | 1 of 5 | 5 of 5 |
| | section-aware | 7 | 1,875 | 0 of 7 | 5 of 5 |

The gain is on list questions; for single-fact questions coverage is unchanged but the context is 40-50%
smaller. Three questions, one document, one run each: treat it as a sanity check.

### Not yet measured

Spoken input and output quality (word error rate, turn-detection errors, time to first audio on real
sessions), a speech-to-speech vs cascaded comparison, human calibration of the judge, repeated runs
with confidence intervals, and any public redistributable corpus.

## Quickstart

Requirements: Python 3.11+, an OpenAI API key with access to the models in
[`config.py`](config.py), a browser with microphone access (headphones recommended to avoid echo),
and Tesseract if you index scanned PDFs.

```bash
git clone https://github.com/Adarshgurazada/voice-to-voice-multilingual-rag.git
cd voice-to-voice-multilingual-rag
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements-dev.txt     # app + tests

cp .env.example .env                              # then edit .env
```

`.env`:

```dotenv
OPENAI_API_KEY=your_key_here
# Tesseract language codes joined with "+" (not commas). Needed only for scanned PDFs.
OCR_LANGUAGES=eng+tel+hin+ben
```

Install Tesseract and its language data for scanned documents (macOS: `brew install tesseract
tesseract-lang`). The code finds Homebrew and common Linux `tessdata` folders automatically; set
`TESSDATA_PREFIX` otherwise. Missing language data raises an indexing error rather than silently
skipping a scanned page.

Run:

```bash
python live_app.py        # serves http://127.0.0.1:8000 (localhost only)
```

Open the page, choose PDFs, click **Index documents**, then **Connect** and speak. Indexing a new
PDF makes paid OpenAI calls (embeddings and image captions) and can take minutes; re-uploading the
same file reuses the local cache in `.local_data/`.

The server logs every turn: the user transcript, the language detected from its script, what the
voice model claimed, the language it answered in, and a warning on any mismatch.

## Running the evaluations

```bash
export OCR_LANGUAGES=eng+tel+hin+ben
PDFS="--pdf docs/plato.pdf --pdf docs/telugu.pdf --pdf docs/hindi.pdf --pdf docs/bengali.pdf"

# 1. Validate labels offline (no API calls)
python eval_rag.py --dataset benchmarks/dev_local.jsonl $PDFS --dry-run

# 2. Retrieval: FAISS vs reranked, per-language and per-category breakdowns
python eval_rag.py --dataset benchmarks/dev_local.jsonl $PDFS [--no-expand] [--ocr auto|vision|off]

# 3. Answer-level grounding, abstention, correctness, answer language
python eval_answers.py --dataset benchmarks/answers_dev_local.jsonl $PDFS [--no-support-check]

# 4. Spoken-turn metrics from labeled turns (word error rate, language accuracy, VAD errors, p50/p95)
python eval_voice.py --dataset path/to/labeled_voice_turns.jsonl
```

Reports are written to `.local_data/`. Index caches are fingerprinted by PDF hashes, models, chunking,
OCR mode and pipeline version, so a comparison is only valid between runs on identical inputs.

**Retrieval cases** (`benchmarks/dev_local.jsonl`): `id`, `query`, `expected_pages` (`doc`, `page`,
optional `quote`), optional `category` and `language`. Failures are labeled `candidate_miss` (FAISS
never found it), `not_promoted`, `rerank_drop`, `late_rank` or `top1_hit`.

**Answer cases** (`benchmarks/answers_dev_local.jsonl`): `id`, `query`, boolean `answerable`, and for
answerable cases a short `reference` and `expected_pages`. Unanswerable cases include near-misses (a
figure whose value is blank) and off-topic questions.

**Answer metrics:** abstention recall (unanswerable), false-abstain rate (answerable), correctness vs
reference, citation page hit rate, raw vs shipped claim support rate (judge, against the full
passage), unsupported shipped claims, answer-language match (by script), and latency.

## Configuration

| Setting | Where | Default |
|---|---|---|
| Realtime voice model | `config.py` `REALTIME_MODEL_ID` | `gpt-realtime-2.1-mini` |
| Embeddings | `EMBEDDING_MODEL_ID` | `text-embedding-3-small` (1536-dim) |
| Vision captions, rerank, quiz | `VISION_MODEL_ID`, `RERANK_MODEL_ID`, `QUIZ_MODEL_ID` | `gpt-5.4-mini` |
| Answer drafting and in-pipeline support check | `ANSWER_MODEL_ID`, `SUPPORT_MODEL_ID` | `gpt-5.4-mini` |
| Evaluation judge (independent of the pipeline) | `JUDGE_MODEL_ID` | `gpt-5.5` |
| Transcription | `TRANSCRIBE_MODEL_ID` | `gpt-4o-mini-transcribe` |
| Verified voice answers | `GROUNDED_VOICE` env | `1` (set `0` to give the voice model raw passages) |
| Hybrid search, follow-up rewriting | `HYBRID_SEARCH`, `FOLLOWUP_REWRITE` env | both on (`0` turns one off) |
| Live transcript preview | `LIVE_TRANSCRIPT_PREVIEW`, `LIVE_PREVIEW_MODEL_ID` env | on, `gpt-realtime-whisper` |
| Answer drafting model | `ANSWER_MODEL_ID` env | `gpt-5.4-mini` |
| Retrieval | `module_v2v_rag.py` | 20 candidates, top 5, 800-char chunks, 150 overlap |
| OCR languages | `OCR_LANGUAGES` env | `eng` |

## Project layout

```
live_app.py            FastAPI server: uploads, WebSocket voice loop, tools, turn logging
voice_session.py       Session state, Realtime tool schemas, language detection by script
module_v2v_rag.py      PDF reading (text/OCR/vision), chunking, batched embeddings, FAISS,
                       query expansion, reciprocal-rank fusion, LLM reranking
doc_structure.py       Layout reading: headers, headings, section paths, table rows, sentence chunks
lexical_search.py      BM25 keyword search fused with the embedding search
highlight.py           Locate a verified quote on its page for the evidence cards
vocabulary.py          Document terms for transcription hints and spelling repair
grounded_answer.py     Claim + quote answers with roles, quote verifier, support and explanation checks
citation_audit.py      Post-hoc claim-to-quote spans for the live transcript
eval_rag.py            Retrieval benchmark (FAISS vs reranked, ablation flags)
eval_answers.py        Answer-level grounding / abstention / correctness benchmark
eval_voice.py          Spoken-turn scoring (word error rate, language, VAD, latency)
study_quiz.py, study_mcp.py, glossary_mcp.py, study_glossary.json
                       Voice quiz, and a small local MCP glossary tool (see Limitations)
live_web/              Browser UI (AudioWorklet mic streaming, transcript, sources)
benchmarks/            Development datasets and RESULTS.md
tests/                 155 offline tests (no network or API key needed)
```

## Tests and CI

```bash
python -m pytest -q          # 155 tests, offline, runs without OPENAI_API_KEY
```

GitHub Actions runs the same suite on every push (`.github/workflows/ci.yml`).

## Limitations

- **Structure reading is heuristic.** Headings, running headers and tables are recovered from font sizes
  and positions. It worked on the one document it was tuned on (a designed, text-layer PDF) and is covered
  by synthetic tests, but other layouts (multi-column articles, scanned pages) fall back to plain chunking.
- **Roles and the plain-words reading come from a small model.** Roles are sometimes mislabelled, and
  the explanation check is deliberately strict, so the plain-words line is often absent.
- **A brief spoken reply is given only the first few verified claims** so the voice model does not read
  everything out; the rest is available on request.
- **The live voice path was exercised by replaying recorded speech and browser events, and by hand,
  not by an automated microphone test.** Transcription hints can occasionally be echoed back by the
  transcription model on silence; those transcripts are detected and ignored.
- **The evaluation is small and self-labeled.** 32 questions, one labeler, single runs. One question
  is about 3 points of retrieval accuracy; many differences above are one or two questions.
- **The LLM judge is uncalibrated** and I have not yet compared it with human labels.
- **The verifier protects what the voice model is *given*, not what it says.** `search_documents`
  returns only verified claims (or a fixed "not found"), but the voice model still rephrases them
  aloud and could add a sentence of its own. The post-hoc citation audit flags that case; nothing
  blocks it. Verified voice answers are also not yet scored by the answer-level evaluation.
- **The verifier adds latency to every spoken answer.** In the text evaluation, drafting plus
  checking takes about 2.1 s on top of about 2.0 s of retrieval. Time to first audio in the live app
  with and without the verifier has not been measured on real sessions yet (`GROUNDED_VOICE=0`
  turns it off for that comparison).
- **Indexed text quality bounds answer quality.** Garbled Indic OCR and mislabeled image captions
  both produced wrong or missing answers in the evaluation.
- **Telugu, Hindi and Bengali coverage is thin** in the benchmark (a handful of pages each).
- **Voice quality is unmeasured on real sessions.** Short utterances can still be mis-transcribed,
  and if the transcript is late the voice model's own language guess is used as a fallback.
- **The quiz and glossary features are side experiments.** The glossary holds three entries and
  starts a subprocess per lookup, so it is a demonstration of the MCP wiring, not a real source.
- Sessions live in memory in one process and are never expired. This is a local single-user demo.

## Roadmap

1. Measure the verifier's latency/faithfulness trade-off in the live voice path
   (`GROUNDED_VOICE=1` vs `0`, both log per-turn metrics), and compare against a cascaded
   transcription, grounded answer, speech-synthesis pipeline on the same spoken questions.
2. Calibrate the judge against human labels and repeat every run to report confidence intervals.
3. A combined Tesseract + vision index as a measured ablation, and keyword weighting tuned so hybrid
   search stops costing top-1 accuracy.
4. A public, redistributable multilingual corpus so the benchmark can live in the repo.
5. An answer panel with claim-to-quote highlights (the exact quote drawn on the page image) and evidence
   cards showing claim, quote, section and page.
6. Follow-up awareness (rewrite "and the second one?" from recent turns) and cross-reference following
   ("see Section 12").
7. A public document set so the section-aware index and list-completeness checks can be scored in the repo.

## Data, privacy and copyright

Audio is streamed to OpenAI and is not saved by this app; OpenAI's API data policies still apply.
Uploaded PDFs, indexes and evaluation reports stay under `.local_data/` on your machine (git-ignored),
and `.env` is git-ignored. The source PDFs used for development are third-party material and are not
redistributed with this repository; evaluation labels in `benchmarks/` reference them by name only.
