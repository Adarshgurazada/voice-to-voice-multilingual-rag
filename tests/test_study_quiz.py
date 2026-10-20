"""Grounding and grading tests for the one-question voice quiz."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from study_quiz import QuizAssessment, QuizDraft, QuizItem, _validate_draft, generate_quiz, grade_quiz  # noqa: E402


def _completion(parsed):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, refusal=None))])


def _engine(parsed):
    calls = []

    def parse(**kwargs):
        calls.append(kwargs)
        return _completion(parsed)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
    metadata = pd.DataFrame([{
        "doc": "book", "page": 2, "type": "text", "content": "A cell is a tiny unit of life.",
    }])
    return SimpleNamespace(client=client, faiss_index=object(), metadata_db=metadata), calls


def test_generation_accepts_exact_quote_from_retrieved_page(monkeypatch):
    draft = QuizDraft(
        question="What is a cell?", answer="A tiny unit of life",
        evidence_quote="A cell is a tiny unit of life.", source_doc="book", source_page=2,
    )
    engine, calls = _engine(draft)
    monkeypatch.setattr("study_quiz.retrieve_context_and_sources", lambda *args: (
        "[Text Context - Source: book, Pg 2] A cell is a tiny unit of life.",
        [{"doc": "book", "page": 2, "type": "text"}],
    ))

    item = generate_quiz(engine, "cells", "en-US")

    assert item.question == "What is a cell?"
    assert item.answer == "A tiny unit of life"
    assert calls[0]["response_format"] is QuizDraft


@pytest.mark.parametrize("source_doc,evidence_quote", [
    ("other", "A cell is a tiny unit of life."),
    ("book", "Plants make their own food."),
])
def test_generation_rejects_unretrieved_or_unquoted_evidence(source_doc, evidence_quote):
    draft = QuizDraft(
        question="What is a cell?", answer="A tiny unit of life",
        evidence_quote=evidence_quote, source_doc=source_doc, source_page=2,
    )
    sources = [{"doc": "book", "page": 2}]
    metadata = pd.DataFrame([{"doc": "book", "page": 2, "content": "A cell is a tiny unit of life."}])

    with pytest.raises(RuntimeError):
        _validate_draft(draft, sources, metadata)


def test_generation_rejects_empty_question_and_oversized_quote():
    metadata = pd.DataFrame([{"doc": "book", "page": 2, "content": "A cell is a tiny unit of life."}])
    sources = [{"doc": "book", "page": 2}]
    for question, quote in [(" ", "A cell is a tiny unit of life."), ("What is a cell?", "x" * 161)]:
        draft = QuizDraft(
            question=question, answer="A tiny unit", evidence_quote=quote,
            source_doc="book", source_page=2,
        )
        with pytest.raises(RuntimeError):
            _validate_draft(draft, sources, metadata)


def test_grade_uses_hidden_key_and_returns_formative_assessment():
    engine, calls = _engine(QuizAssessment(correct=True, feedback="Good explanation."))
    item = QuizItem("What is a cell?", "A tiny unit of life", "A cell is a tiny unit.", "book", 2)

    result = grade_quiz(engine.client, item, "A little living building block", "en-US")

    assert result.correct is True
    assert calls[0]["response_format"] is QuizAssessment
    assert "Expected answer: A tiny unit of life" in calls[0]["messages"][1]["content"]
