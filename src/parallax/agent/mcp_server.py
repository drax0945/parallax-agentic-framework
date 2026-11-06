"""Stdio MCP server exposing the Parallax tools.

Register it with any MCP client — Claude Code, the Claude desktop app, an IDE —
and the chemistry tools, memory and citation machinery become available inside
that agent, alongside the skills in `.claude/skills/`.

    {
      "mcpServers": {
        "parallax": { "command": "parallax", "args": ["mcp"] }
      }
    }
"""

from __future__ import annotations

import json
from typing import Any

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from ..config import settings
from .tools import TOOLS, call_tool

INSTRUCTIONS = """
Parallax: cheminformatics tools with provenance.

Keep the evidence tiers apart and say which one a number belongs to: measured (an
experiment, carries a DOI), computed (a deterministic calculation from structure), or
modelled (a prediction, carries an applicability-domain verdict and the model's RMSE).

Start a task with `memory_recall`; finish by writing durable conclusions with
`memory_remember`. Cite everything from tool results, verify DOIs with `check_doi`, and
end a report with `citation_bibliography`.

Retrosynthesis tools generate hypotheses, not precedent: confirm each step with
`reaction_precedent_search` before standing behind it. Restricted chemistry classes are
screened automatically and blocked verdicts are final.
""".strip()


def _result(payload: dict[str, Any]) -> types.CallToolResult:
    text = json.dumps(payload, indent=2, default=str)
    is_error = bool(payload.get("error")) or bool(payload.get("refused"))
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)], isError=is_error
    )


def build_server() -> Server:
    async def on_list_tools(ctx, params):  # type: ignore[no-untyped-def]
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name=tool.name,
                    description=tool.description,
                    inputSchema=tool.input_schema,
                    annotations=types.ToolAnnotations(
                        readOnlyHint=tool.read_only,
                        destructiveHint=False,
                        openWorldHint=tool.group in {"literature", "bioactivity", "structure", "synthesis"},
                    ),
                )
                for tool in TOOLS
            ]
        )

    async def on_call_tool(ctx, params):  # type: ignore[no-untyped-def]
        payload = call_tool(params.name, dict(params.arguments or {}))
        return _result(payload)

    return Server(
        "parallax",
        version="0.1.0",
        title="Parallax cheminformatics",
        instructions=INSTRUCTIONS,
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


async def serve() -> None:
    settings().ensure_dirs()
    server = build_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    import anyio

    anyio.run(serve)


if __name__ == "__main__":
    main()
