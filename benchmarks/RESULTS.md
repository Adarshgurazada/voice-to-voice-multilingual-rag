# Dev benchmark results (local corpus)

These numbers come from a **small, self-labeled development set** over four local PDFs
(`plato`, `telugu`, `hindi`, `bengali`; scanned pages, not redistributable). They show whether a
change helped on *this* set. They are not a validated benchmark: 32 questions, labels written by
one person (with an AI assistant), single runs, LLM-based components that vary between runs.
Do not quote them externally without the caveats below.

Reproduce (needs `OPENAI_API_KEY`; Tesseract with `eng+tel+hin+ben` for OCR):

```bash
OCR_LANGUAGES=eng+tel+hin+ben python eval_rag.py --dataset benchmarks/dev_local.jsonl \
  --pdf docs/plato.pdf --pdf docs/telugu.pdf --pdf docs/hindi.pdf --pdf docs/bengali.pdf [--no-expand] [--ocr vision]
OCR_LANGUAGES=eng+tel+hin+ben python eval_answers.py --dataset benchmarks/answers_dev_local.jsonl \
  --pdf docs/plato.pdf --pdf docs/telugu.pdf --pdf docs/hindi.pdf --pdf docs/bengali.pdf [--no-support-check]
```

## 1. Retrieval (page-level, 32 questions)

Hit@k = a labeled evidence page is among the first k distinct pages. "Reranked" is what the
assistant actually sees (top 5 after LLM reranking of 20 candidates).

| Configuration | Candidate recall | FAISS Hit@5 | Reranked Hit@1 | Hit@3 | Hit@5 | MRR@5 | Retrieval time |
|---|---|---|---|---|---|---|---|
| Tesseract OCR, no query expansion (baseline) | 0.938 | 0.906 | 0.875 | 0.938 | 0.938 | 0.901 | 1.4 s |
| **Tesseract OCR + English query expansion** | **1.000** | **1.000** | **0.906** | **0.969** | **0.969** | **0.932** | 1.7 s |
| Vision-model page reading, no expansion | 0.875 | 0.844 | 0.812 | 0.844 | 0.844 | 0.828 | 1.4 s |
| Vision-model page reading + expansion | 0.938 | 0.906 | 0.906 | 0.906 | 0.906 | 0.906 | 1.6 s |

What changed and why:

- **Query expansion fixed the cross-language misses.** Telugu and Hindi questions about the English
  Plato text (`xl-01`, `xl-02`) used to retrieve pages from the Telugu and Hindi documents instead:
  dense search favours same-language passages. Translating non-English queries to English and fusing
  both rankings (reciprocal-rank fusion) found the right page for both, at about +0.3 s per query.
- **Vision-model page reading is not a clear win.** It fixed one Telugu question (`te-03`) but lost
  two others (`xl-03`, `bn-01`). It stays available as `--ocr vision`; a hybrid that indexes both
  Tesseract and vision text is the obvious next experiment.
- **Remaining misses.** `te-06` ("author of the lesson") ranks page 2 before the labeled page 1; both
  pages name the author, so this is partly label ambiguity (labels were left unchanged). One
  question dropped in reranking (`rerank_drop`).

Sample-size note: one question is 3.1 points. Differences of one or two questions are noise.

## 2. Answer-level grounding (32 questions: 24 answerable, 8 unanswerable)

The answer pipeline drafts claims with quotes, then (a) verifies each quote against the cited passage
(exact, or at least 90% character match to tolerate OCR noise) and (b) optionally has a model check
that the claim is actually supported. Only verified claims form the answer; otherwise it abstains.
A separate, stronger judge model (`gpt-5.5`) scores correctness against a reference and checks every
proposed claim against its full passage. **The judge is a model, not ground truth, and has not yet
been calibrated against human labels.**

| Run (same index, expansion on) | Abstained on unanswerable | False abstain (answerable) | Correct | Shipped claims supported | Answer language correct |
|---|---|---|---|---|---|
| Quote check only | 7/8 | 2/24 | 21/24 | 30/32 | 100% |
| Quote + support check, run 1 | 8/8 | 3/24 | 19/24 | 27/28 | 100% |
| Quote + support check, run 2 (repeat) | 6/8 | 2/24 | 18/24 | 29/31 | 100% |

Findings:

- **Quote verification alone is not enough.** A fabricated claim ("translated into English in 1804")
  carried a real quote from an unrelated passage and passed the quote check. That is why the
  support stage exists. But the repeat run shows the effect on abstention is **within run-to-run
  noise** (8/8 vs 6/8 on the same configuration); we cannot claim it helps until the set is larger
  and runs are repeated.
- **Answer language is now correct in every run** after the answer language was passed explicitly
  (it was 77% when the model copied the source-language quote into the answer).
- **Known failure the guardrail cannot catch:** the elephant life-span question (`un-01`) is
  answered with the crocodile's number because the *image caption* in the index misattributes the
  figure labels. The cited passage "supports" the claim; the index itself is wrong. Grounding is only
  as good as the indexed text.
- **OCR noise causes false abstentions.** Several answerable questions abstain because the model
  "corrects" OCR typos inside its quote (e.g. `7imaeus` for `Timaeus`) or the Telugu OCR is garbled.
- Added latency of the support stage: about +0.7 s per answer (1.2 s to 1.9–2.1 s).

## 3. Hybrid keyword search and follow-up rewriting

Two retrieval changes, each switchable for measurement: BM25 keyword search fused with the embedding search
(`--no-hybrid` turns it off) and rewriting of follow-up questions from the last three turns (`--no-rewrite`).
**Neither was tuned on these sets**; the defaults were chosen before running them.

Reproduce (needs `OPENAI_API_KEY`; the same four PDFs as above):

```bash
OCR_LANGUAGES=eng+tel+hin+ben python eval_rag.py --dataset benchmarks/dev_local.jsonl \
  --pdf docs/plato.pdf --pdf docs/telugu.pdf --pdf docs/hindi.pdf --pdf docs/bengali.pdf [--no-hybrid]
OCR_LANGUAGES=eng+tel+hin+ben python eval_rag.py --dataset benchmarks/followups_dev_local.jsonl \
  --pdf docs/plato.pdf --pdf docs/telugu.pdf --pdf docs/hindi.pdf --pdf docs/bengali.pdf [--no-hybrid] [--no-rewrite]
```

**Hybrid search, original 32 questions** (3 runs each; mean, with the range in brackets). These runs use the
current index version, so they are not comparable with the numbers in section 1.

| Configuration | Candidate recall | Reranked Hit@1 | Hit@3 | Hit@5 | MRR@5 | Seconds per query |
|---|---|---|---|---|---|---|
| Dense (embeddings) only | 0.979 | **0.823** (0.81-0.84) | 0.927 (0.91-0.94) | 0.958 (0.94-0.97) | 0.882 (0.87-0.90) | 1.71 |
| Dense + BM25 (hybrid) | **0.990** | 0.792 (0.75-0.81) | **0.990** (0.97-1.00) | **0.990** (0.97-1.00) | 0.891 (0.86-0.91) | 1.68 |

Hybrid search finds the labelled page more often within the first three or five results (+0.06 at Hit@3,
+0.03 at Hit@5) and costs no measurable time. It does **not** improve the top-ranked result: Hit@1 is about
one question lower, and the ranges overlap. The answer step reads all five reranked passages, so Hit@5 is
the number that matters most for answers; but this is one question set, and the Hit@1 loss is real.

**Follow-up rewriting, 12 follow-up questions with conversation history** (2-3 runs each). Examples:
"Which village was he from?" after a question about Potana; "And how about the crocodile?" after a question
about the tortoise.

| Configuration | Candidate recall | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|
| Dense, no rewrite | 0.917 | 0.833 | 0.833 | 0.833 | 0.833 |
| **Dense + rewrite** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| Hybrid, no rewrite | 0.917 | 0.694 (0.67-0.75) | 0.833 | 0.833 | 0.755 |
| Hybrid + rewrite | 1.000 | 0.889 (0.75-1.00) | 0.972 (0.92-1.00) | 1.000 | 0.936 (0.88-1.00) |

- **Rewriting is the clear win.** It fixed the cases where a pronoun hid the subject: "Which village was he
  from?" and "What are his parents called?" searched the Plato PDF without the history and found the right
  Telugu page with it. It costs about one second, only on messages that look like follow-ups.
- **Hybrid search hurts context-free follow-ups.** Without a rewrite, a vague message like "Which village was
  he from?" gives BM25 only generic words, and Hit@1 falls from 0.833 to 0.694. With the rewrite the keyword
  search gets real terms and the gap closes (0.889 against 1.000 for dense + rewrite at Hit@1, equal at Hit@5).
- **Caveats.** 12 conversations written by one person with an AI assistant from facts already in the labelled
  set, one document family, and LLM reranking that varies run to run. Differences of one question are 8 points
  here. Treat the direction as the finding, not the decimals.

## Not yet measured

Spoken input and output (word error rate, turn detection, time to first audio in the live app), the
speech-to-speech vs cascaded comparison, a human-labeled calibration set for the judge, repeated runs
with confidence intervals, and any public, redistributable corpus.
