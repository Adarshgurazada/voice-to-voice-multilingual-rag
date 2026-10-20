"""Pure retrieval-evaluation checks; no OpenAI calls."""

import json
import sys

import pytest

from eval_rag import load_cases, main, score_case, summarize


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


def test_score_case_miss_and_summary():
    case = {"id": "q1", "expected_pages": [{"doc": "biology", "page": 5}]}
    miss = score_case(case, [{"doc": "biology", "page": 1}], k=5)
    miss["retrieval_seconds"] = 0.2

    summary = summarize([miss], k=5)

    assert summary["hit_at_k"] == 0
    assert summary["recall_at_k"] == 0
    assert summary["mrr_at_k"] == 0
    assert summary["mean_retrieval_seconds"] == 0.2


def test_dry_run_validates_without_importing_api_backend(tmp_path, monkeypatch, capsys):
    pdf = tmp_path / "biology.pdf"
    pdf.write_bytes(b"placeholder for validation-only mode")
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
