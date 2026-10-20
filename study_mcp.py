"""Read-only local MCP server for curated study vocabulary."""

import json
from pathlib import Path

from mcp.server import MCPServer


GLOSSARY_PATH = Path(__file__).with_name("study_glossary.json")
mcp = MCPServer("Voice Study Glossary")


def find_term(term: str, language: str) -> dict:
    """Find a curated explanation without using a model or external service."""
    normalized = term.strip().casefold()
    if not normalized or len(normalized) > 80:
        return {"status": "invalid_term"}
    glossary = json.loads(GLOSSARY_PATH.read_text(encoding="utf-8"))
    for entry in glossary["terms"]:
        names = [entry["term"], *entry.get("aliases", [])]
        if normalized in (name.casefold() for name in names):
            explanation = entry["explanations"].get(language)
            if explanation is None:
                return {"status": "language_unavailable", "term": entry["term"], "language": language}
            return {
                "status": "found", "term": entry["term"], "language": language,
                "explanation": explanation, "source": "curated_study_glossary",
            }
    return {"status": "not_found", "term": term.strip()}


@mcp.tool()
def lookup_term(term: str, language: str = "en-US") -> str:
    """Look up a learner-friendly definition in the local curated glossary."""
    return json.dumps(find_term(term, language), ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")
