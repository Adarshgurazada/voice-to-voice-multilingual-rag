"""Grounded one-question practice quizzes for the browser voice session."""

from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel

from config import QUIZ_MODEL_ID
from module_v2v_rag import RAG_TOP_K, retrieve_context_and_sources


class QuizDraft(BaseModel):
    question: str
    answer: str
    evidence_quote: str
    source_doc: str
    source_page: int


class QuizAssessment(BaseModel):
    correct: bool
    feedback: str


@dataclass
class QuizItem:
    question: str
    answer: str
    evidence_quote: str
    source_doc: str
    source_page: int

    @property
    def source(self) -> dict:
        return {"doc": self.source_doc, "page": self.source_page, "type": "text"}


def _parsed(completion, kind):
    message = completion.choices[0].message
    if getattr(message, "refusal", None):
        raise RuntimeError("The model declined the quiz request.")
    parsed = getattr(message, "parsed", None)
    if not isinstance(parsed, kind):
        raise RuntimeError("The model did not return a complete quiz result.")
    return parsed


def _validate_draft(draft: QuizDraft, sources: list[dict], metadata: pd.DataFrame) -> QuizItem:
    if not draft.question.strip() or not draft.answer.strip():
        raise RuntimeError("The quiz question or answer was empty.")
    if not 8 <= len(draft.evidence_quote.strip()) <= 160:
        raise RuntimeError("The quiz evidence quote has an invalid length.")
    if draft.source_page < 1:
        raise RuntimeError("The quiz source page was invalid.")
    source_key = (draft.source_doc, draft.source_page)
    if source_key not in {(source["doc"], source["page"]) for source in sources}:
        raise RuntimeError("The quiz question named a page that was not retrieved.")
    rows = metadata[
        (metadata["doc"] == draft.source_doc) & (metadata["page"] == draft.source_page)
    ]
    quote = " ".join(draft.evidence_quote.split()).casefold()
    if not any(quote in " ".join(str(content).split()).casefold() for content in rows["content"]):
        raise RuntimeError("The quiz evidence quote was not found on the cited page.")
    return QuizItem(
        question=draft.question.strip(), answer=draft.answer.strip(),
        evidence_quote=draft.evidence_quote.strip(), source_doc=draft.source_doc,
        source_page=draft.source_page,
    )


def generate_quiz(engine, topic: str, language: str) -> QuizItem:
    """Retrieve evidence, then make one question whose quote exists on a source page."""
    topic = topic.strip()[:200] or "important ideas in the document"
    context, sources = retrieve_context_and_sources(
        topic, engine.faiss_index, engine.metadata_db, engine.client, RAG_TOP_K,
    )
    if not sources:
        raise RuntimeError("No document evidence was found for a quiz question.")
    completion = engine.client.chat.completions.parse(
        model=QUIZ_MODEL_ID,
        messages=[
            {"role": "system", "content": (
                "Create one short, age-appropriate open question answerable only from the provided "
                "PDF passages. Write the question and answer in the requested language. Include "
                "one exact 8-160 character quote from a retrieved passage, and its source document "
                "stem and one-based page. Do not use outside facts or reveal the answer in the question."
            )},
            {"role": "user", "content": f"Language: {language}\nTopic: {topic}\nRetrieved passages:\n{context}"},
        ],
        response_format=QuizDraft,
        timeout=20,
    )
    return _validate_draft(_parsed(completion, QuizDraft), sources, engine.metadata_db)


def grade_quiz(client, item: QuizItem, learner_answer: str, language: str) -> QuizAssessment:
    """Compare a spoken answer with the stored key; return advisory feedback."""
    if not learner_answer.strip():
        raise ValueError("No spoken answer was captured. Please try again.")
    completion = client.chat.completions.parse(
        model=QUIZ_MODEL_ID,
        messages=[
            {"role": "system", "content": (
                "Evaluate the learner's answer using only the expected answer and evidence quote. "
                "Accept accurate paraphrases and minor speech transcription errors. Mark incorrect "
                "when the essential meaning is absent or contradicted. Give one brief, kind explanation "
                "in the requested language. This is formative feedback, not a formal grade."
            )},
            {"role": "user", "content": (
                f"Language: {language}\nQuestion: {item.question}\nExpected answer: {item.answer}"
                f"\nEvidence: {item.evidence_quote}\nLearner answer: {learner_answer[:1000]}"
            )},
        ],
        response_format=QuizAssessment,
        timeout=20,
    )
    assessment = _parsed(completion, QuizAssessment)
    if not assessment.feedback.strip():
        raise RuntimeError("The quiz grader returned no feedback.")
    return assessment
