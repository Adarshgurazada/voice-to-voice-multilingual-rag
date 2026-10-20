"""Run a page-level retrieval evaluation against the application's PDF pipeline."""

import argparse
import asyncio
import hashlib
import json
import os
import re
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
                if not isinstance(page, dict) or not {"doc", "page"} <= set(page) or set(page) - {"doc", "page", "quote"}:
                    raise ValueError(f"Line {line_number}: each expected page needs doc and page; quote is optional.")
                doc, number = page["doc"], page["page"]
                if not isinstance(doc, str) or doc not in document_stems or type(number) is not int or number < 1:
                    raise ValueError(f"Line {line_number}: invalid document stem or page number.")
                if "quote" in page and (not isinstance(page["quote"], str) or not page["quote"].strip()):
                    raise ValueError(f"Line {line_number}: quote must be nonempty text.")
                page_keys.add((doc, number))
            if len(page_keys) != len(expected):
                raise ValueError(f"Line {line_number}: duplicate expected page.")
            seen_ids.add(case_id)
            labeled = {"id": case_id, "query": query, "expected_pages": expected}
            for key in ("category", "language"):
                if key in case:
                    if not isinstance(case[key], str) or not case[key].strip():
                        raise ValueError(f"Line {line_number}: {key} must be nonempty text.")
                    labeled[key] = case[key]
            cases.append(labeled)
    if not cases:
        raise ValueError("The dataset contains no evaluation cases.")
    return cases


def validate_pdf_labels(cases: list[dict], pdf_paths: list[Path]) -> None:
    """Check page bounds and optional evidence quotes without making API calls."""
    import pymupdf

    documents = {}
    try:
        for path in pdf_paths:
            documents[path.stem] = pymupdf.open(path)
        for case in cases:
            for label in case["expected_pages"]:
                document = documents[label["doc"]]
                page = label["page"]
                if page > len(document):
                    raise ValueError(f"Case {case['id']} labels {label['doc']} page {page}, but the PDF has {len(document)} pages.")
                if "quote" in label:
                    pdf_page = document[page - 1]
                    text = re.sub(r"\s+", " ", pdf_page.get_text()).casefold()
                    quote = re.sub(r"\s+", " ", label["quote"]).casefold().strip()
                    if quote not in text and pdf_page.get_images(full=True):
                        try:
                            textpage = pdf_page.get_textpage_ocr(
                                language=os.getenv("OCR_LANGUAGES", "eng"), dpi=150, full=True,
                            )
                            text = re.sub(r"\s+", " ", pdf_page.get_text(textpage=textpage)).casefold()
                        except Exception as exc:
                            raise ValueError(f"OCR needed to validate {label['doc']} page {page}: {exc}") from exc
                    if quote not in text:
                        raise ValueError(f"Case {case['id']} quote was not found on {label['doc']} page {page}.")
    finally:
        for document in documents.values():
            document.close()


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


K_VALUES = (1, 3, 5)


def score_retrieval(case: dict, candidates: list[dict], selected: list[dict]) -> dict:
    """Score both ranking stages and identify where evidence was lost."""
    stages = {
        "faiss": {str(k): score_case(case, candidates, k) for k in K_VALUES},
        "reranked": {str(k): score_case(case, selected, k) for k in K_VALUES},
    }
    candidate_score = score_case(case, candidates, len(candidates) or 1)
    candidate_hit = candidate_score["hit_at_k"]
    if not candidate_hit:
        failure = "candidate_miss"
    elif not stages["reranked"]["5"]["hit_at_k"]:
        failure = "rerank_drop" if stages["faiss"]["5"]["hit_at_k"] else "not_promoted"
    elif not stages["reranked"]["1"]["hit_at_k"]:
        failure = "late_rank"
    else:
        failure = "top1_hit"
    return {
        "id": case["id"], "query": case["query"],
        "expected_pages": case["expected_pages"],
        "category": case.get("category"), "language": case.get("language"),
        "candidate_pages": candidate_score["retrieved_pages"],
        "candidate_chunks": candidates,
        "retrieved_pages": stages["reranked"]["5"]["retrieved_pages"],
        "selected_chunks": selected,
        "candidate_hit": candidate_hit,
        "stages": stages,
        "outcome": failure,
    }


def summarize_benchmark(results: list[dict]) -> dict:
    summary = {"cases": len(results), "outcomes": {}}
    for stage in ("faiss", "reranked"):
        summary[stage] = {}
        for k in K_VALUES:
            values = [result["stages"][stage][str(k)] for result in results]
            summary[stage][f"hit_at_{k}"] = sum(value["hit_at_k"] for value in values) / len(values)
            summary[stage][f"recall_at_{k}"] = sum(value["recall_at_k"] for value in values) / len(values)
            summary[stage][f"mrr_at_{k}"] = sum(value["reciprocal_rank"] for value in values) / len(values)
    summary["candidate_hit_rate"] = sum(result["candidate_hit"] for result in results) / len(results)
    for result in results:
        summary["outcomes"][result["outcome"]] = summary["outcomes"].get(result["outcome"], 0) + 1
    for name in ("retrieval_seconds", "embedding_seconds", "faiss_seconds", "rerank_seconds"):
        summary[f"mean_{name}"] = sum(result["timings"][name] for result in results) / len(results)
    for key in ("category", "language"):
        groups = {}
        for result in results:
            value = result.get(key) or "unspecified"
            groups.setdefault(value, []).append(result)
        summary[f"by_{key}"] = {
            value: {
                "cases": len(group),
                "candidate_hit_rate": sum(item["candidate_hit"] for item in group) / len(group),
                **{f"hit_at_{k}": sum(item["stages"]["reranked"][str(k)]["hit_at_k"] for item in group) / len(group)
                   for k in K_VALUES},
            }
            for value, group in groups.items()
        }
    return summary


def compare_reports(current: dict, baseline: dict) -> dict:
    if baseline.get("schema_version") != 2:
        raise ValueError("Baseline report uses a different benchmark schema.")
    if current["inputs"] != baseline.get("inputs"):
        raise ValueError("Baseline must use the same dataset and PDF content hashes.")
    if current["configuration"]["top_k"] != baseline["configuration"]["top_k"]:
        raise ValueError("Baseline and current top_k must match.")
    return {
        stage: {
            f"{metric}_at_{k}": current["summary"][stage][f"{metric}_at_{k}"]
            - baseline["summary"][stage][f"{metric}_at_{k}"]
            for k in K_VALUES for metric in ("hit", "recall", "mrr")
        }
        for stage in ("faiss", "reranked")
    }


async def run_retrieval(cases: list[dict], pdf_paths: list[Path], ocr_mode: str = "auto") -> dict:
    from config import EMBEDDING_MODEL_ID, RERANK_MODEL_ID, VISION_MODEL_ID
    from module_v2v_rag import (
        RAG_CANDIDATE_K, RAG_TOP_K, RAG_TEXT_CHUNK_SIZE, RAG_CHUNK_OVERLAP,
        INDEX_PIPELINE_VERSION,
        VoiceRAGMultimodalFAISS, retrieve_documents,
    )

    pdf_hashes = []
    for path in pdf_paths:
        with path.open("rb") as handle:
            pdf_hashes.append({"name": path.name, "sha256": hashlib.file_digest(handle, "sha256").hexdigest()})
    index_config = {
        "pdfs": pdf_hashes, "embedding_model": EMBEDDING_MODEL_ID,
        "vision_model": VISION_MODEL_ID, "chunk_size": RAG_TEXT_CHUNK_SIZE,
        "chunk_overlap": RAG_CHUNK_OVERLAP, "pipeline_version": INDEX_PIPELINE_VERSION,
        "ocr_mode": ocr_mode, "ocr_languages": os.getenv("OCR_LANGUAGES", "eng"),
    }
    fingerprint = hashlib.sha256(json.dumps(index_config, sort_keys=True).encode()).hexdigest()[:16]
    cache_root = Path(".local_data") / "eval_cache" / fingerprint
    engine = VoiceRAGMultimodalFAISS(
        pdf_paths=[str(path) for path in pdf_paths],
        cache_dir=cache_root / "indexes",
        image_save_dir=cache_root / "images",
        ocr_mode=ocr_mode,
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
            retrieval = await asyncio.to_thread(
                retrieve_documents,
                case["query"], engine.faiss_index, engine.metadata_db, engine.client, RAG_TOP_K,
            )
            if retrieval.error:
                raise RuntimeError(f"Retrieval failed for case {case['id']}: {retrieval.error}")
            result = score_retrieval(case, retrieval.candidates, retrieval.sources)
            result["timings"] = {
                "retrieval_seconds": round(time.perf_counter() - started, 4),
                **{key: round(value, 4) for key, value in retrieval.timings.items()},
            }
            results.append(result)
        return {
            "schema_version": 2,
            "configuration": {
                "embedding_model": EMBEDDING_MODEL_ID,
                "vision_model": VISION_MODEL_ID,
                "rerank_model": RERANK_MODEL_ID,
                "candidate_k": RAG_CANDIDATE_K,
                "top_k": RAG_TOP_K,
                "index_fingerprint": fingerprint,
                "ocr_mode": ocr_mode,
                "indexed_ocr_pages": len({
                    (str(row["doc"]), int(row["page"]))
                    for _, row in engine.metadata_db[engine.metadata_db["type"] == "ocr"].iterrows()
                }),
            },
            "summary": summarize_benchmark(results),
            "results": results,
            "inputs": {"pdfs": pdf_hashes},
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
    parser.add_argument("--compare", type=Path, help="Compare with a report from the same dataset and PDFs")
    parser.add_argument("--ocr", choices=("auto", "off"), default="auto",
                        help="Use OCR on low-text image pages, or disable it for a baseline")
    args = parser.parse_args()

    try:
        pdf_paths = [path.resolve() for path in args.pdf]
        if any(not path.is_file() or path.suffix.lower() != ".pdf" for path in pdf_paths):
            raise ValueError("Every --pdf path must be an existing PDF file.")
        stems = [path.stem for path in pdf_paths]
        if len(set(stems)) != len(stems):
            raise ValueError("PDF filenames must have unique stems.")
        cases = load_cases(args.dataset, set(stems))
        validate_pdf_labels(cases, pdf_paths)
        if args.dry_run:
            print(f"Valid dataset: {len(cases)} cases across {len(pdf_paths)} PDF(s). No API calls made.")
            return 0
        with args.dataset.open("rb") as handle:
            dataset_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        pdf_hashes = []
        for path in pdf_paths:
            with path.open("rb") as handle:
                pdf_hashes.append({"name": path.name, "sha256": hashlib.file_digest(handle, "sha256").hexdigest()})
        baseline = None
        if args.compare:
            baseline = json.loads(args.compare.read_text(encoding="utf-8"))
            if baseline.get("schema_version") != 2 or baseline.get("inputs") != {
                "dataset_sha256": dataset_hash, "pdfs": pdf_hashes,
            }:
                raise ValueError("Baseline must be a version 2 report for the same dataset and PDFs.")
            from module_v2v_rag import RAG_TOP_K
            if baseline.get("configuration", {}).get("top_k") != RAG_TOP_K:
                raise ValueError("Baseline and current top_k must match.")
        report = asyncio.run(run_retrieval(cases, pdf_paths, args.ocr))
        report["inputs"]["dataset_sha256"] = dataset_hash
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        summary = report["summary"]
        print(f"Saved {summary['cases']} cases to {args.output}")
        for stage in ("faiss", "reranked"):
            scores = summary[stage]
            print(f"{stage}: Hit@1={scores['hit_at_1']:.3f}  Hit@3={scores['hit_at_3']:.3f}  "
                  f"Hit@5={scores['hit_at_5']:.3f}  MRR@5={scores['mrr_at_5']:.3f}")
        print(f"Outcomes: {summary['outcomes']}")
        if baseline is not None:
            deltas = compare_reports(report, baseline)
            print("Metric deltas vs baseline:", deltas)
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
