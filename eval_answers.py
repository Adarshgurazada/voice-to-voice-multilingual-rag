"""Answer-level evaluation: grounding, abstention, correctness and answer language.

For every question the pipeline retrieves passages, drafts a claim-cited answer and keeps
only claims whose quotes verify. A separate judge model then checks (a) each proposed claim
against the full cited passage and (b) the final answer against a reference. The judge is
a model, not ground truth: spot-check its verdicts in the report before quoting numbers.
"""

import argparse
import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from grounded_answer import answer_question, detect_language

REQUIRED = {"id", "query", "answerable"}


class ClaimVerdict(BaseModel):
    index: int
    supported: bool
    note: str


class Judgement(BaseModel):
    correctness: Literal["correct", "partial", "incorrect"]
    claims: list[ClaimVerdict]


def load_cases(path: Path, document_stems: set[str]) -> list[dict]:
    cases, seen = [], set()
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            case = json.loads(line)
            if not REQUIRED <= set(case) or not isinstance(case["answerable"], bool):
                raise ValueError(f"Line {number}: id, query and boolean answerable are required.")
            if case["id"] in seen:
                raise ValueError(f"Line {number}: duplicate id {case['id']}.")
            seen.add(case["id"])
            if case["answerable"]:
                if not case.get("reference"):
                    raise ValueError(f"Line {number}: answerable cases need a reference answer.")
                for page in case.get("expected_pages", []):
                    if page["doc"] not in document_stems:
                        raise ValueError(f"Line {number}: unknown document {page['doc']}.")
            cases.append(case)
    if not cases:
        raise ValueError("The dataset contains no cases.")
    return cases


JUDGE_PROMPT = (
    "You grade a document question-answering system. You receive the question, the reference "
    "(or NONE when the documents do not contain the answer), the system's final answer, and a list "
    "of claims the system proposed, each with the full text of the passage it cited.\n"
    "1. correctness: compare the FINAL ANSWER with the reference. 'correct' = same essential facts; "
    "'partial' = some but not all essential facts, or extra unsupported detail; 'incorrect' = wrong, "
    "missing the key fact, or it answers when the reference is NONE.\n"
    "2. For each claim, supported=true only if the cited passage text directly states or clearly "
    "implies the claim. Judge the passage, not your own knowledge. Add a short note.\n"
    "Return one verdict per claim index."
)


def judge(client, model: str, case: dict, final_text: str, raw_claims: list[dict], passages: dict[str, str]) -> Judgement:
    payload = {
        "question": case["query"],
        "reference": case.get("reference") or "NONE",
        "final_answer": final_text,
        "claims": [
            {"index": i, "claim": c["text"], "passage": passages.get(c["source_id"], "")[:3000]}
            for i, c in enumerate(raw_claims)
        ],
    }
    completion = client.chat.completions.parse(
        model=model,
        messages=[{"role": "system", "content": JUDGE_PROMPT},
                  {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        response_format=Judgement,
        timeout=40,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("Judge returned no result.")
    return parsed


def score_case(case: dict, answer, judgement: Judgement | None, retrieval_seconds: float) -> dict:
    verdicts = {v.index: v.supported for v in judgement.claims} if judgement else {}
    verified_keys = {(c["text"], c["source_id"], c["quote"]) for c in answer.claims}
    raw = []
    for i, claim in enumerate(answer.raw_claims):
        raw.append({**claim, "supported": verdicts.get(i),
                    "verified": (claim["text"].strip(), claim["source_id"], claim["quote"].strip()) in verified_keys})
    expected = {(p["doc"], p["page"]) for p in case.get("expected_pages", [])}
    cited = {(c["doc"], c["page"]) for c in answer.claims}
    return {
        "id": case["id"], "query": case["query"], "language": case.get("language"),
        "category": case.get("category"), "answerable": case["answerable"],
        "abstained": answer.abstained, "answer": answer.text,
        "answer_language": answer.language, "query_language": detect_language(case["query"]),
        "correctness": judgement.correctness if judgement and not answer.abstained else None,
        "raw_claims": raw, "dropped": answer.dropped,
        "citation_page_hit": bool(expected & cited) if case["answerable"] and not answer.abstained else None,
        "retrieval_seconds": round(retrieval_seconds, 3), "answer_seconds": round(answer.seconds, 3),
    }


def _rate(numerator: int, denominator: int):
    return round(numerator / denominator, 3) if denominator else None


def summarize(results: list[dict]) -> dict:
    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]
    shown = [r for r in answerable if not r["abstained"]]
    raw = [c for r in results for c in r["raw_claims"] if c["supported"] is not None]
    kept = [c for c in raw if c["verified"]]
    non_abstained = [r for r in results if not r["abstained"]]
    summary = {
        "cases": len(results), "answerable": len(answerable), "unanswerable": len(unanswerable),
        "abstention_recall": _rate(sum(r["abstained"] for r in unanswerable), len(unanswerable)),
        "false_abstain_rate": _rate(sum(r["abstained"] for r in answerable), len(answerable)),
        "answered_unanswerable": [r["id"] for r in unanswerable if not r["abstained"]],
        "correct_rate": _rate(sum(r["correctness"] == "correct" for r in answerable), len(answerable)),
        "correct_or_partial_rate": _rate(sum(r["correctness"] in ("correct", "partial") for r in answerable), len(answerable)),
        "citation_page_hit_rate": _rate(sum(bool(r["citation_page_hit"]) for r in shown), len(shown)),
        "raw_claims": len(raw), "verified_claims": len(kept),
        "raw_claim_support_rate": _rate(sum(c["supported"] for c in raw), len(raw)),
        "shipped_claim_support_rate": _rate(sum(c["supported"] for c in kept), len(kept)),
        "unsupported_shipped_claims": sum(not c["supported"] for c in kept),
        "dropped_claims": sum(len(r["dropped"]) for r in results),
        "answer_language_match_rate": _rate(
            sum(r["answer_language"] == r["query_language"] for r in non_abstained), len(non_abstained)),
        "mean_retrieval_seconds": round(sum(r["retrieval_seconds"] for r in results) / len(results), 2),
        "mean_answer_seconds": round(sum(r["answer_seconds"] for r in results) / len(results), 2),
    }
    for key in ("language", "category"):
        groups: dict[str, list[dict]] = {}
        for r in results:
            groups.setdefault(r.get(key) or "unspecified", []).append(r)
        summary[f"by_{key}"] = {
            name: {
                "cases": len(g),
                "abstained_correctly_or_answered": _rate(
                    sum((r["abstained"] != r["answerable"]) for r in g), len(g)),
                "correct_rate_answerable": _rate(
                    sum(r["correctness"] == "correct" for r in g if r["answerable"]),
                    sum(r["answerable"] for r in g)),
            }
            for name, g in groups.items()
        }
    return summary


async def run(
    cases: list[dict], pdf_paths: list[Path], ocr_mode: str, expand: bool, check: bool = True, workers: int = 4,
) -> dict:
    from config import ANSWER_MODEL_ID, JUDGE_MODEL_ID
    from eval_rag import index_cache
    from module_v2v_rag import RAG_TOP_K, VoiceRAGMultimodalFAISS, retrieve_documents

    fingerprint, pdf_hashes, cache_root = index_cache(pdf_paths, ocr_mode)
    engine = VoiceRAGMultimodalFAISS(
        pdf_paths=[str(p) for p in pdf_paths], cache_dir=cache_root / "indexes",
        image_save_dir=cache_root / "images", ocr_mode=ocr_mode,
    )
    try:
        await engine.setup_rag_database()
        if not engine.is_active:
            raise RuntimeError("Could not load or build a searchable index.")

        def one(case: dict) -> dict:
            started = time.perf_counter()
            retrieval = retrieve_documents(
                case["query"], engine.faiss_index, engine.metadata_db, engine.client, RAG_TOP_K, expand)
            retrieval_seconds = time.perf_counter() - started
            if retrieval.error:
                raise RuntimeError(f"{case['id']}: retrieval failed: {retrieval.error}")
            passages = [{**p, "id": f"S{i}"} for i, p in enumerate(retrieval.passages, 1)]
            answer = answer_question(engine.client, case["query"], passages, check=check)
            judgement = None
            if answer.raw_claims or (case["answerable"] and not answer.abstained):
                judgement = judge(engine.client, JUDGE_MODEL_ID, case, answer.text, answer.raw_claims,
                                  {p["id"]: p["content"] for p in passages})
            return score_case(case, answer, judgement, retrieval_seconds)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(await asyncio.gather(*[asyncio.to_thread(one, c) for c in cases]))
        return {
            "schema_version": 1,
            "configuration": {"answer_model": ANSWER_MODEL_ID, "judge_model": JUDGE_MODEL_ID,
                              "ocr_mode": ocr_mode, "query_expansion": expand, "support_check": check, "index_fingerprint": fingerprint,
                              "top_k": RAG_TOP_K},
            "inputs": {"pdfs": pdf_hashes},
            "summary": summarize(results),
            "results": results,
        }
    finally:
        engine.client.close()
        await engine.async_client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path(".local_data/answers_report.json"))
    parser.add_argument("--ocr", choices=("auto", "vision", "off"), default="auto")
    parser.add_argument("--no-expand", action="store_true")
    parser.add_argument("--no-support-check", action="store_true",
                        help="Skip the semantic support stage (ablation: quote check only)")
    args = parser.parse_args()
    try:
        pdfs = [p.resolve() for p in args.pdf]
        cases = load_cases(args.dataset, {p.stem for p in pdfs})
        report = asyncio.run(run(cases, pdfs, args.ocr, not args.no_expand, not args.no_support_check))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {report['summary']['cases']} cases to {args.output}")
    print(json.dumps({k: v for k, v in report["summary"].items() if not k.startswith("by_")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
