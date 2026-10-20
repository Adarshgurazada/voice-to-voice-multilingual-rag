"""Run a page-level retrieval evaluation against the application's PDF pipeline."""

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path


def load_cases(path: Path, document_stems: set[str]) -> list[dict]:
    cases = []
    seen_ids = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(case, dict):
                raise ValueError(f"Line {line_number}: expected a JSON object.")
            case_id = case.get("id")
            query = case.get("query")
            expected = case.get("expected_pages")
            if not isinstance(case_id, str) or not case_id.strip() or case_id in seen_ids:
                raise ValueError(f"Line {line_number}: id must be a unique nonempty string.")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"Line {line_number}: query must be a nonempty string.")
            if not isinstance(expected, list) or not expected:
                raise ValueError(f"Line {line_number}: expected_pages must be a nonempty list.")
            page_keys = set()
            for page in expected:
                if not isinstance(page, dict) or set(page) != {"doc", "page"}:
                    raise ValueError(f"Line {line_number}: each expected page needs doc and page.")
                doc, number = page["doc"], page["page"]
                if not isinstance(doc, str) or doc not in document_stems or type(number) is not int or number < 1:
                    raise ValueError(f"Line {line_number}: invalid document stem or page number.")
                page_keys.add((doc, number))
            if len(page_keys) != len(expected):
                raise ValueError(f"Line {line_number}: duplicate expected page.")
            seen_ids.add(case_id)
            cases.append({"id": case_id, "query": query, "expected_pages": expected})
    if not cases:
        raise ValueError("The dataset contains no evaluation cases.")
    return cases


def score_case(case: dict, sources: list[dict], k: int) -> dict:
    expected = {(page["doc"], page["page"]) for page in case["expected_pages"]}
    predicted = []
    for source in sources:
        key = (source["doc"], source["page"])
        if key not in predicted:
            predicted.append(key)
        if len(predicted) == k:
            break
    hits = expected.intersection(predicted)
    first_hit_rank = next((rank for rank, key in enumerate(predicted, 1) if key in expected), None)
    return {
        "id": case["id"],
        "expected_pages": case["expected_pages"],
        "retrieved_pages": [{"doc": doc, "page": page} for doc, page in predicted],
        "hit_at_k": bool(hits),
        "recall_at_k": len(hits) / len(expected),
        "reciprocal_rank": 1 / first_hit_rank if first_hit_rank else 0.0,
    }


def summarize(results: list[dict], k: int) -> dict:
    count = len(results)
    return {
        "cases": count,
        "hit_at_k": sum(result["hit_at_k"] for result in results) / count,
        "recall_at_k": sum(result["recall_at_k"] for result in results) / count,
        "mrr_at_k": sum(result["reciprocal_rank"] for result in results) / count,
        "mean_retrieval_seconds": sum(result["retrieval_seconds"] for result in results) / count,
        "k": k,
    }


async def run_retrieval(cases: list[dict], pdf_paths: list[Path]) -> dict:
    from config import EMBEDDING_MODEL_ID, RERANK_MODEL_ID
    from module_v2v_rag import (
        RAG_CANDIDATE_K, RAG_TOP_K, VoiceRAGMultimodalFAISS, retrieve_context_and_sources,
    )

    engine = VoiceRAGMultimodalFAISS(
        pdf_paths=[str(path) for path in pdf_paths],
        cache_dir=Path(".local_data") / "indexes",
        image_save_dir=Path(".local_data") / "images",
    )
    try:
        await engine.setup_rag_database()
        if not engine.is_active or engine.faiss_index is None or engine.metadata_db is None:
            raise RuntimeError("Could not load or build a searchable index for the PDFs.")
        available_pages = {
            (str(row["doc"]), int(row["page"])) for _, row in engine.metadata_db.iterrows()
        }
        for case in cases:
            for page in case["expected_pages"]:
                if (page["doc"], page["page"]) not in available_pages:
                    raise ValueError(
                        f"Case {case['id']} labels {page['doc']} page {page['page']}, "
                        "which has no indexed content."
                    )
        results = []
        for case in cases:
            started = time.perf_counter()
            context, sources = await asyncio.to_thread(
                retrieve_context_and_sources,
                case["query"], engine.faiss_index, engine.metadata_db, engine.client, RAG_TOP_K,
            )
            if context.startswith(("Error retrieving", "Could not process query embedding")):
                raise RuntimeError(f"Retrieval failed for case {case['id']}: {context}")
            result = score_case(case, sources, RAG_TOP_K)
            result["retrieval_seconds"] = round(time.perf_counter() - started, 3)
            results.append(result)
        return {
            "configuration": {
                "embedding_model": EMBEDDING_MODEL_ID,
                "rerank_model": RERANK_MODEL_ID,
                "candidate_k": RAG_CANDIDATE_K,
                "top_k": RAG_TOP_K,
            },
            "summary": summarize(results, RAG_TOP_K),
            "results": results,
        }
    finally:
        if engine.client is not None:
            engine.client.close()
        if engine.async_client is not None:
            await engine.async_client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Labeled JSONL question set")
    parser.add_argument("--pdf", type=Path, action="append", required=True, help="PDF to search; repeat for multiple files")
    parser.add_argument("--output", type=Path, default=Path(".local_data/eval_report.json"))
    parser.add_argument("--dry-run", action="store_true", help="Validate labels and paths without API calls")
    args = parser.parse_args()

    try:
        pdf_paths = [path.resolve() for path in args.pdf]
        if any(not path.is_file() or path.suffix.lower() != ".pdf" for path in pdf_paths):
            raise ValueError("Every --pdf path must be an existing PDF file.")
        stems = [path.stem for path in pdf_paths]
        if len(set(stems)) != len(stems):
            raise ValueError("PDF filenames must have unique stems.")
        cases = load_cases(args.dataset, set(stems))
        if args.dry_run:
            print(f"Valid dataset: {len(cases)} cases across {len(pdf_paths)} PDF(s). No API calls made.")
            return 0
        report = asyncio.run(run_retrieval(cases, pdf_paths))
        with args.dataset.open("rb") as handle:
            dataset_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        pdf_hashes = []
        for path in pdf_paths:
            with path.open("rb") as handle:
                pdf_hashes.append({"name": path.name, "sha256": hashlib.file_digest(handle, "sha256").hexdigest()})
        report["inputs"] = {"dataset_sha256": dataset_hash, "pdfs": pdf_hashes}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        summary = report["summary"]
        print(f"Saved {summary['cases']} cases to {args.output}")
        print(f"Hit@{summary['k']}: {summary['hit_at_k']:.3f}  "
              f"Recall@{summary['k']}: {summary['recall_at_k']:.3f}  "
              f"MRR@{summary['k']}: {summary['mrr_at_k']:.3f}")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
