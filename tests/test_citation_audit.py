from types import SimpleNamespace

from citation_audit import CitationClaim, CitationDraft, audit_citations


def test_citation_audit_rejects_unverified_quotes_and_ids():
    draft = CitationDraft(claims=[
        CitationClaim(claim="Plants use sunlight", source_id="S1", evidence_quote="sunlight"),
        CitationClaim(claim="Plants use sunlight", source_id="S2", evidence_quote="sunlight"),
        CitationClaim(claim="make food", source_id="S1", evidence_quote="invented quote"),
    ])
    completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=draft))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=lambda **kwargs: completion)))
    answer = "Plants use sunlight to make food."
    spans = audit_citations(client, answer, [
        {"id": "S1", "content": "Plants use sunlight to make food."},
    ])
    assert spans == [{"start": 0, "end": 19, "source_id": "S1", "evidence_quote": "sunlight"}]


def test_citation_audit_without_evidence_avoids_api_call():
    assert audit_citations(None, "An answer", []) == []
