"""The local glossary server and its MCP protocol boundary."""

import asyncio
import json

from mcp import Client

from glossary_mcp import lookup_glossary
from study_mcp import find_term, mcp


def test_curated_glossary_has_explicit_language_coverage():
    found = find_term("Cells", "hi-IN")
    missing_language = find_term("cell", "ta-IN")
    missing_term = find_term("mitochondrion", "en-US")

    assert found["status"] == "found"
    assert found["source"] == "curated_study_glossary"
    assert missing_language["status"] == "language_unavailable"
    assert missing_term["status"] == "not_found"


def test_mcp_server_exposes_read_only_lookup_tool():
    async def check():
        async with Client(mcp) as client:
            tools = await client.list_tools()
            assert [tool.name for tool in tools.tools] == ["lookup_term"]
            result = await client.call_tool("lookup_term", {"term": "fraction", "language": "en-US"})
            assert not result.is_error
            return json.loads(result.content[0].text)

    assert asyncio.run(check())["status"] == "found"


def test_browser_bridge_launches_local_stdio_server():
    result = asyncio.run(lookup_glossary("photosynthesis", "en-US"))

    assert json.loads(result)["status"] == "found"
