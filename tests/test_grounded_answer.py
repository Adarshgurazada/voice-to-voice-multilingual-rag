"""Offline checks for claim verification, abstention and answer-level scoring."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import eval_answers as ev  # noqa: E402
import grounded_answer as ga  # noqa: E402

PASSAGES = [
    {"id": "S1", "doc": "plato", "page": 1, "content": "Critias son of Callaeschrus made himself leader of a thirty-man junta."},
    {"id": "S2", "doc": "plato", "page": 2, "content": "The decisive battle took place by the temple of Bendis."},
]


def _client(draft):
    class Completions:
        def parse(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=draft))])
    return SimpleNamespace(chat=SimpleNamespace(completions=Completions()))


def _draft(claims, answerable=True, answer=None, plain_words="", roles=None):
    text = answer if answer is not None else " ".join(c[0] for c in claims)
    return ga.GroundedDraft(answerable=answerable, answer=text,
                            plain_words=plain_words,
                            claims=[ga.AnswerClaim(text=t, source_id=s, quote=q,
                                                   role=(roles[i] if roles else "answer"))
                                    for i, (t, s, q) in enumerate(claims)])


def test_detect_language_by_script():
    assert ga.detect_language("Who led the Thirty?") == "en"
    assert ga.detect_language("ముప్పై మంది నియంతల నాయకుడు ఎవరు?") == "te"
    assert ga.detect_language("तोते की आयु कितनी है?") == "hi"
    assert ga.detect_language("কোন কবিতা?") == "bn"


def test_verified_claims_form_the_answer():
    draft = _draft([("Critias led the junta.", "S1", "Critias son of Callaeschrus")])
    result = ga.answer_question(_client(draft), "Who led the Thirty?", PASSAGES, check=False)
    assert not result.abstained and result.text == "Critias led the junta."
    assert result.claims[0]["page"] == 1 and not result.dropped


def test_fabricated_quote_is_dropped_and_unsupported_text_is_not_spoken():
    draft = _draft(
        [("Critias led the junta.", "S1", "Critias son of Callaeschrus"),
         ("He was born in Sparta.", "S1", "born in Sparta in 450 BC")],
        answer="Critias led the junta. He was born in Sparta. He liked sailing.",
    )
    result = ga.answer_question(_client(draft), "Who led the Thirty?", PASSAGES, check=False)
    assert result.text == "Critias led the junta."
    assert [d["reason"] for d in result.dropped] == ["quote_not_in_passage"]


def test_unknown_source_and_tiny_quote_are_dropped():
    draft = _draft([("A fact.", "S9", "something here"), ("Another fact.", "S1", "Cri")])
    verified, dropped = ga.verify_draft(draft, PASSAGES)
    assert not verified
    assert {d["reason"] for d in dropped} == {"unknown_source", "empty_claim_or_quote"}


def test_quote_match_tolerates_ocr_hyphenation_but_not_fabrication():
    passage = "The decisive battle took place by the temple of Ben- dis, the goddess whose festival began it."
    assert ga.quote_match("the temple of Bendis, the goddess", passage) == "exact"
    assert ga.quote_match("the temple of Bendes the goddess", passage) == "fuzzy"
    assert ga.quote_match("the decisive battle took place in Athens harbour", passage) is None


def test_abstains_in_question_language_when_unanswerable_or_nothing_verifies():
    result = ga.answer_question(_client(_draft([], answerable=False, answer="")), "తెలుగు ప్రశ్న?", PASSAGES, check=False)
    assert result.abstained and result.text == ga.ABSTAIN_MESSAGES["te"]
    bad = _draft([("X", "S1", "not in the passage at all")])
    result = ga.answer_question(_client(bad), "What is it?", PASSAGES, check=False)
    assert result.abstained and result.text == ga.ABSTAIN_MESSAGES["en"] and result.dropped


def test_load_cases_validates_schema(tmp_path):
    good = tmp_path / "ok.jsonl"
    good.write_text('{"id":"a","query":"q","answerable":true,"reference":"r","expected_pages":[{"doc":"plato","page":1}]}\n'
                    '{"id":"b","query":"q2","answerable":false}\n', encoding="utf-8")
    assert len(ev.load_cases(good, {"plato"})) == 2
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id":"a","query":"q","answerable":true}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        ev.load_cases(bad, {"plato"})


def test_summary_metrics():
    ok = ga.GroundedAnswer("Critias led the junta.", False, "en",
                           claims=[{"text": "Critias led the junta.", "source_id": "S1", "quote": "Critias son of", "doc": "plato", "page": 1}],
                           raw_claims=[{"text": "Critias led the junta.", "source_id": "S1", "quote": "Critias son of"}])
    refusal = ga.GroundedAnswer(ga.ABSTAIN_MESSAGES["en"], True, "en")
    judged = ev.Judgement(correctness="correct", claims=[ev.ClaimVerdict(index=0, supported=True, note="ok")])
    answerable = {"id": "a", "query": "Who led the Thirty?", "answerable": True, "language": "en",
                  "reference": "Critias", "expected_pages": [{"doc": "plato", "page": 1}]}
    unanswerable = {"id": "u", "query": "Capital of France?", "answerable": False, "language": "en"}
    results = [ev.score_case(answerable, ok, judged, 1.0), ev.score_case(unanswerable, refusal, None, 1.0)]
    summary = ev.summarize(results)
    assert summary["abstention_recall"] == 1.0 and summary["false_abstain_rate"] == 0.0
    assert summary["correct_rate"] == 1.0 and summary["citation_page_hit_rate"] == 1.0
    assert summary["shipped_claim_support_rate"] == 1.0 and summary["unsupported_shipped_claims"] == 0


def _two_stage_client(draft, verdicts):
    class Completions:
        def parse(self, **kwargs):
            parsed = draft if kwargs["response_format"] is ga.GroundedDraft else ga.SupportCheck(
                verdicts=[ga.SupportVerdict(index=i, supported=v) for i, v in enumerate(verdicts)])
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))])
    return SimpleNamespace(chat=SimpleNamespace(completions=Completions()))


def test_semantic_check_drops_claims_whose_quote_does_not_support_them():
    draft = _draft([("Critias led the junta.", "S1", "Critias son of Callaeschrus"),
                    ("The Republic was translated in 1804.", "S2", "temple of Bendis")])
    result = ga.answer_question(_two_stage_client(draft, [True, False]), "Who led the Thirty?", PASSAGES)
    assert result.text == "Critias led the junta."
    assert [d["reason"] for d in result.dropped] == ["not_supported"]


def test_semantic_check_failure_fails_closed():
    draft = _draft([("Critias led the junta.", "S1", "Critias son of Callaeschrus")])

    class Boom:
        def parse(self, **kwargs):
            if kwargs["response_format"] is ga.GroundedDraft:
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=draft))])
            raise RuntimeError("support check down")
    client = SimpleNamespace(chat=SimpleNamespace(completions=Boom()))
    result = ga.answer_question(client, "Who led the Thirty?", PASSAGES)
    assert result.abstained and result.dropped[0]["reason"] == "not_supported"


def test_claim_citing_document_name_is_repaired_when_quote_is_in_one_passage():
    passage_text = PASSAGES[0]["content"]
    quote = passage_text[:40]
    draft = _draft([("A fact.", "some_document_name", quote)])
    verified, dropped = ga.verify_draft(draft, PASSAGES)
    assert [c["source_id"] for c in verified] == [PASSAGES[0]["id"]] and not dropped


def test_misnamed_source_is_still_dropped_when_quote_matches_nothing():
    verified, dropped = ga.verify_draft(_draft([("A fact.", "doc_name", "nothing like this anywhere")]), PASSAGES)
    assert not verified and dropped[0]["reason"] == "unknown_source"


def test_answer_question_reports_each_real_stage():
    from types import SimpleNamespace
    draft = _draft([("Critias led the junta.", "S1", "Critias son of Callaeschrus")])

    class Completions:
        def parse(self, **kwargs):
            parsed = draft if kwargs["response_format"] is ga.GroundedDraft else ga.SupportCheck(
                verdicts=[ga.SupportVerdict(index=0, supported=True)])
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    steps = []
    ga.answer_question(client, "Who led the junta?", PASSAGES, "en", True, steps.append)
    assert steps == [
        f"Reading {len(PASSAGES)} passages and drafting an answer",
        "Checking 1 claim against the exact quotes",
        "1 of 1 quotes found on the page",
        "Confirming each claim is really supported by its passage",
    ]


def test_a_failing_progress_callback_never_breaks_answering():
    from types import SimpleNamespace
    draft = _draft([("Critias led the junta.", "S1", "Critias son of Callaeschrus")])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        parse=lambda **k: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=draft))]))))

    def boom(text):
        raise RuntimeError("socket gone")

    assert ga.answer_question(client, "Who led?", PASSAGES, "en", False, boom).abstained is False


def _teaching_client(draft, support_ok=True, explanation_ok=True):
    from types import SimpleNamespace

    class Completions:
        def parse(self, **kwargs):
            fmt = kwargs["response_format"]
            if fmt is ga.GroundedDraft:
                parsed = draft
            elif fmt is ga.ExplanationVerdict:
                parsed = ga.ExplanationVerdict(ok=explanation_ok)
            else:
                parsed = ga.SupportCheck(verdicts=[ga.SupportVerdict(index=i, supported=support_ok) for i in range(5)])
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))])

    return SimpleNamespace(chat=SimpleNamespace(completions=Completions()))


def test_claims_keep_their_roles_and_sections_for_teaching():
    passages = [{**PASSAGES[0], "section": "Part A › Section 1 · Start"}, *PASSAGES[1:]]
    draft = _draft([("Critias led the junta.", "S1", "Critias son of Callaeschrus")], roles=["answer"])
    answer = ga.answer_question(_teaching_client(draft), "Who led?", passages, "en", True)
    assert answer.claims[0]["role"] == "answer"
    assert answer.sections == ["Part A › Section 1 · Start"]


def test_plain_words_survive_only_when_the_checker_agrees():
    draft = _draft([("Critias led the junta.", "S1", "Critias son of Callaeschrus")],
                   plain_words="In short, Critias was the leader.")
    kept = ga.answer_question(_teaching_client(draft, explanation_ok=True), "Who led?", PASSAGES, "en", True)
    assert kept.explanation == "In short, Critias was the leader." and not kept.abstained
    dropped = ga.answer_question(_teaching_client(draft, explanation_ok=False), "Who led?", PASSAGES, "en", True)
    assert dropped.explanation == "" and not dropped.abstained  # the verified claims are still spoken


def test_plain_words_are_discarded_when_a_claim_they_describe_is_unsupported():
    draft = _draft([("Critias led the junta.", "S1", "Critias son of Callaeschrus")],
                   plain_words="In short, Critias was the leader.")
    answer = ga.answer_question(_teaching_client(draft, support_ok=False), "Who led?", PASSAGES, "en", True)
    assert answer.abstained and answer.explanation == ""


def test_explanation_check_fails_closed_on_error():
    from types import SimpleNamespace

    def boom(**kwargs):
        raise RuntimeError("api down")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=boom)))
    assert ga.check_explanation(client, "anything", [{"text": "A claim."}]) is False
    assert ga.check_explanation(client, "", [{"text": "A claim."}]) is False
