"""MCP client bridge used by the browser voice tool loop."""

import json
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters


SERVER_PATH = Path(__file__).with_name("study_mcp.py")


async def lookup_glossary(term: str, language: str) -> str:
    """Call the local stdio server and return its text result to Realtime."""
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER_PATH)])
    try:
        async with Client(params) as client:
            result = await client.call_tool(
                "lookup_term", {"term": term, "language": language}, read_timeout_seconds=8,
            )
        if result.is_error:
            return json.dumps({"status": "unavailable", "reason": "Glossary tool failed."})
        return "\n".join(block.text for block in result.content if getattr(block, "type", "") == "text")
    except Exception:
        return json.dumps({"status": "unavailable", "reason": "Local glossary is unavailable."})
