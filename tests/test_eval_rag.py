"""Pure retrieval-evaluation checks; no OpenAI calls."""

import asyncio
import json
import sys

import pandas as pd
import pytest
import pymupdf

from eval_rag import (
    compare_reports, load_cases, main, score_case, score_retrieval,
    run_retrieval, summarize_benchmark, validate_pdf_labels,
)


def test_load_cases_accepts_labeled_pages(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(json.dumps({
        "id": "q1", "query": "What is a cell?",
        "expected_pages": [{"doc": "biology", "page": 3}],
    }) + "\n", encoding="utf-8")

    cases = load_cases(dataset, {"biology"})

    assert cases[0]["expected_pages"] == [{"doc": "biology", "page": 3}]


@pytest.mark.parametrize("expected", [
    [], [{"doc": "other", "page": 1}], [{"doc": [], "page": 1}],
    [{"doc": "biology", "page": True}], [{"doc": "biology", "page": 0}],
    [{"doc": "biology", "page": 1}, {"doc": "biology", "page": 1}],
])
def test_load_cases_rejects_bad_labels(tmp_path, expected):
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(json.dumps({
        "id": "q1", "query": "Question", "expected_pages": expected,
    }) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_cases(dataset, {"biology"})


def test_load_cases_rejects_duplicate_ids(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    case = {"id": "q1", "query": "Question", "expected_pages": [{"doc": "biology", "page": 1}]}
    dataset.write_text(json.dumps(case) + "\n" + json.dumps(case) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unique"):
        load_cases(dataset, {"biology"})


def test_validate_pdf_labels_checks_page_and_quote(tmp_path):
    pdf = tmp_path / "biology.pdf"
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Cells have membranes")
    document.save(pdf)
    document.close()
    case = {"id": "q1", "expected_pages": [{"doc": "biology", "page": 1, "quote": "Cells have membranes"}]}
    validate_pdf_labels([case], [pdf])
    case["expected_pages"][0]["quote"] = "not on page"
    with pytest.raises(ValueError, match="quote was not found"):
        validate_pdf_labels([case], [pdf])
    case["expected_pages"][0]["page"] = 2
    with pytest.raises(ValueError, match="PDF has 1 pages"):
        validate_pdf_labels([case], [pdf])


def test_score_case_deduplicates_pages_and_computes_metrics():
    case = {
        "id": "q1", "query": "Question",
        "expected_pages": [{"doc": "biology", "page": 2}, {"doc": "biology", "page": 4}],
    }
    sources = [
        {"doc": "biology", "page": 1, "type": "text"},
        {"doc": "biology", "page": 1, "type": "image"},
        {"doc": "biology", "page": 2, "type": "text"},
        {"doc": "biology", "page": 3, "type": "text"},
    ]

    result = score_case(case, sources, k=3)

    assert result["retrieved_pages"] == [
        {"doc": "biology", "page": 1}, {"doc": "biology", "page": 2},
        {"doc": "biology", "page": 3},
    ]
    assert result["hit_at_k"] is True
    assert result["recall_at_k"] == 0.5
    assert result["reciprocal_rank"] == 0.5


def test_score_case_miss():
    case = {"id": "q1", "expected_pages": [{"doc": "biology", "page": 5}]}
    miss = score_case(case, [{"doc": "biology", "page": 1}], k=5)
    assert miss["hit_at_k"] is False
    assert miss["recall_at_k"] == 0
    assert miss["reciprocal_rank"] == 0


def test_stage_scores_distinguish_candidate_miss_and_rerank_drop():
    case = {"id": "q1", "query": "Question", "expected_pages": [{"doc": "biology", "page": 4}]}
    wrong = [{"doc": "biology", "page": 1}]
    right = [{"doc": "biology", "page": 4}]
    miss = score_retrieval(case, wrong, wrong)
    lost = score_retrieval(case, wrong + right, wrong)
    hit = score_retrieval(case, wrong + right, right)
    not_promoted = score_retrieval(
        case, [{"doc": "biology", "page": page} for page in (1, 2, 3, 5, 6, 4)],
        [{"doc": "biology", "page": page} for page in (1, 2, 3, 5, 6)],
    )
    assert miss["outcome"] == "candidate_miss"
    assert lost["outcome"] == "rerank_drop"
    assert not_promoted["outcome"] == "not_promoted"
    assert hit["outcome"] == "top1_hit"
    assert lost["stages"]["faiss"]["3"]["hit_at_k"] is True
    for result in (miss, lost, hit):
        result["timings"] = {
            "retrieval_seconds": 0.2, "embedding_seconds": 0.1,
            "faiss_seconds": 0.01, "rerank_seconds": 0.09,
        }
    summary = summarize_benchmark([miss, lost, hit])
    assert summary["faiss"]["hit_at_3"] == pytest.approx(2 / 3)
    assert summary["reranked"]["hit_at_1"] == pytest.approx(1 / 3)
    assert summary["outcomes"] == {"candidate_miss": 1, "rerank_drop": 1, "top1_hit": 1}
    assert summary["by_category"]["unspecified"]["cases"] == 3


def test_comparison_requires_same_inputs():
    metrics = {f"{metric}_at_{k}": 0.5 for k in (1, 3, 5) for metric in ("hit", "recall", "mrr")}
    lower = {key: 0.25 for key in metrics}
    current = {"schema_version": 2, "inputs": {"dataset_sha256": "same"}, "configuration": {"top_k": 5},
               "summary": {"faiss": metrics, "reranked": metrics}}
    baseline = {"schema_version": 2, "inputs": {"dataset_sha256": "same"}, "configuration": {"top_k": 5},
                "summary": {"faiss": lower, "reranked": metrics}}
    assert compare_reports(current, baseline)["faiss"]["hit_at_1"] == 0.25
    baseline["inputs"]["dataset_sha256"] = "different"
    with pytest.raises(ValueError, match="same dataset"):
        compare_reports(current, baseline)


def test_run_retrieval_reports_both_stages_without_api(tmp_path, monkeypatch):
    import module_v2v_rag as rag

    pdf = tmp_path / "biology.pdf"
    pdf.write_bytes(b"benchmark bytes")
    case = {"id": "q1", "query": "What is a cell?", "category": "fact",
            "expected_pages": [{"doc": "biology", "page": 2}]}
    cache_dirs = []

    class FakeAsyncClient:
        async def close(self):
            pass

    class FakeEngine:
        def __init__(self, *, pdf_paths, cache_dir, image_save_dir, ocr_mode):
            cache_dirs.append(cache_dir)
            self.is_active = True
            self.faiss_index = object()
            self.metadata_db = pd.DataFrame([{"doc": "biology", "page": 2, "type": "text"}])
            self.client = type("Client", (), {"close": lambda self: None})()
            self.async_client = FakeAsyncClient()

        async def setup_rag_database(self):
            pass

    def fake_retrieve(*args):
        return rag.RetrievalResult(
            "context", [{"doc": "biology", "page": 2}],
            [{"doc": "biology", "page": 1}, {"doc": "biology", "page": 2}],
            {"embedding_seconds": 0.01, "faiss_seconds": 0.01, "rerank_seconds": 0.02},
        )

    monkeypatch.setattr(rag, "VoiceRAGMultimodalFAISS", FakeEngine)
    monkeypatch.setattr(rag, "retrieve_documents", fake_retrieve)
    monkeypatch.setenv("OPENAI_API_KEY", "test-dummy-key")
    monkeypatch.chdir(tmp_path)
    report = asyncio.run(run_retrieval([case], [pdf]))
    assert report["schema_version"] == 2
    assert report["summary"]["faiss"]["hit_at_1"] == 0
    assert report["summary"]["reranked"]["hit_at_1"] == 1
    assert report["summary"]["outcomes"] == {"top1_hit": 1}
    assert len(report["configuration"]["index_fingerprint"]) == 16
    assert report["configuration"]["ocr_mode"] == "auto"
    assert "eval_cache" in str(cache_dirs[0])


def test_dry_run_validates_without_importing_api_backend(tmp_path, monkeypatch, capsys):
    pdf = tmp_path / "biology.pdf"
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Cells have membranes")
    document.save(pdf)
    document.close()
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(json.dumps({
        "id": "q1", "query": "Question",
        "expected_pages": [{"doc": "biology", "page": 1}],
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "eval_rag.py", "--dataset", str(dataset), "--pdf", str(pdf), "--dry-run",
    ])

    assert main() == 0
    assert "No API calls made" in capsys.readouterr().out
    assert not (tmp_path / "eval_report.json").exists()
