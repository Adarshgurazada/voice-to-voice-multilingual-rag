import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from vocabulary import Corrector, build_known_words, extract_vocabulary, prompt_hint

DOCS = [
    "Brandel records calls. Why use Brandel? Our CRM and the CRM sync work well.\n"
    "Brandel drafts follow-ups. SECTION 3 pricing. The pricing page lists plans.",
    "Many sales teams use Brandel daily. A handoff to the CRM is automatic. Yes, it works.\nYes, really.",
]


def test_extracts_names_and_acronyms_but_not_generic_words():
    vocab = extract_vocabulary(DOCS, ["3f9a1c2b7d4e8a60_Brandel_Sales_Playbook.pdf"])
    assert "Brandel" in vocab and "CRM" in vocab
    assert "Sales" in vocab and "Playbook" in vocab  # words from the document title
    assert "Yes" not in vocab and "SECTION" not in vocab and "pricing" not in vocab
    assert not any(len(word) >= 12 and word.lower() == word and word.isalnum() and any(c.isdigit() for c in word)
                   for word in vocab)  # no file hash from the filename


def test_prompt_hint_respects_budget():
    hint = prompt_hint(["Brandel", "Playbook", "RevOps"], max_chars=14)
    assert hint == "Brandel, Playbook"[:len(hint)] and len(hint) <= 14


def test_corrector_repairs_misheard_term_and_reports_it():
    corrector = Corrector(["Brandel", "Playbook"], build_known_words(DOCS))
    text, fixes = corrector.correct("What does Brendel actually do?")
    assert text == "What does Brandel actually do?" and fixes == [{"from": "Brendel", "to": "Brandel"}]
    assert corrector.correct("Tell me about Brendle pricing")[0] == "Tell me about Brandel pricing"


def test_corrector_leaves_real_words_and_unrelated_words_alone():
    corrector = Corrector(["Brandel", "Playbook"], build_known_words(DOCS))
    for sentence in ("What does brandy actually do?", "Is the handle total correct?", "How do I handle pricing?"):
        assert corrector.correct(sentence) == (sentence, [])


def test_correct_term_with_wrong_casing_is_fixed_silently():
    text, fixes = Corrector(["Brandel"]).correct("what is brandel")
    assert text == "what is Brandel" and fixes == []


def test_empty_vocabulary_changes_nothing():
    assert Corrector([]).correct("Brendel") == ("Brendel", [])


def test_prompt_echo_is_recognised_and_real_speech_is_not():
    from vocabulary import is_prompt_echo
    vocab = ["Brandel", "Sales", "Playbook", "CRM", "Gong", "Slack"]
    assert is_prompt_echo("Brandel, Sales, Playbook, CRM, Gong, Slack", vocab)
    assert is_prompt_echo("Brandel, Sales, Playbook, CRM,", vocab)  # cut off mid-list
    assert not is_prompt_echo("Explain the Brandel positioning statement in plain words.", vocab)
    assert not is_prompt_echo("Tell me about Gong, Slack and the CRM", vocab)
    assert not is_prompt_echo("Brandel", vocab)
    assert not is_prompt_echo("Brandel, Sales, Playbook", [])
