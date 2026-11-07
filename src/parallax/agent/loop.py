"""The agent session.

Runs Claude with the Parallax tools (in-process, no subprocess hop), the skills
in `.claude/skills/`, and a memory briefing carried over from previous sessions.

Authentication piggybacks on the local `claude` CLI, so an existing Claude Code
login works and no API key is needed. If `ANTHROPIC_API_KEY` is set it is used
instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from ..config import settings, utcnow
from ..provenance import registry
from .prompts import build_system_prompt
from .tools import TOOLS, call_tool

SERVER_NAME = "parallax"


@dataclass
class Turn:
    kind: str  # text | tool_use | tool_result | thinking | result | error
    text: str = ""
    tool: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


def _sdk_tools() -> list[Any]:
    """Wrap the Parallax tools as in-process SDK MCP tools."""
    from claude_agent_sdk import tool as sdk_tool

    wrapped = []
    for definition in TOOLS:

        def make(name: str) -> Callable[[dict[str, Any]], Any]:
            async def handler(args: dict[str, Any]) -> dict[str, Any]:
                payload = call_tool(name, args)
                return {
                    "content": [
                        {"type": "text", "text": json.dumps(payload, indent=2, default=str)}
                    ],
                    "isError": bool(payload.get("error") or payload.get("refused")),
                }

            return handler

        wrapped.append(
            sdk_tool(definition.name, definition.description, definition.input_schema)(
                make(definition.name)
            )
        )
    return wrapped


def _options(
    *,
    model: str | None = None,
    max_turns: int = 60,
    extra_system: str | None = None,
    permission_mode: str = "bypassPermissions",
) -> Any:
    from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server

    cfg = settings()
    cfg.ensure_dirs()
    server = create_sdk_mcp_server(name=SERVER_NAME, version="0.1.0", tools=_sdk_tools())

    allowed = [f"mcp__{SERVER_NAME}__{t.name}" for t in TOOLS]
    # The agent also benefits from reading/writing workspace files (reports, notes).
    allowed += ["Read", "Write", "Glob", "Grep", "Skill"]

    return ClaudeAgentOptions(
        model=model or cfg.model,
        system_prompt=build_system_prompt(extra=extra_system),
        mcp_servers={SERVER_NAME: server},
        allowed_tools=allowed,
        disallowed_tools=["WebSearch", "WebFetch", "Bash"],
        permission_mode=permission_mode,  # tools here are read-mostly and screened
        max_turns=max_turns,
        cwd=str(cfg.workspace),
        setting_sources=["project"],
        skills="all",
        env={"PARALLAX_WORKSPACE": str(cfg.workspace)},
    )


async def stream_session(
    prompt: str,
    *,
    model: str | None = None,
    max_turns: int = 60,
    extra_system: str | None = None,
    resume: str | None = None,
) -> AsyncIterator[Turn]:
    """Run one task, yielding turns as they happen."""
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeSDKClient,
        ResultMessage,
        TextBlock,
        ThinkingBlock,
        ToolResultBlock,
        ToolUseBlock,
    )

    options = _options(model=model, max_turns=max_turns, extra_system=extra_system)
    if resume:
        options.resume = resume

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        yield Turn(kind="text", text=block.text)
                    elif isinstance(block, ThinkingBlock):
                        yield Turn(kind="thinking", text=getattr(block, "thinking", ""))
                    elif isinstance(block, ToolUseBlock):
                        yield Turn(
                            kind="tool_use",
                            tool=str(block.name).replace(f"mcp__{SERVER_NAME}__", ""),
                            payload=dict(block.input or {}),
                        )
            elif isinstance(message, ResultMessage):
                yield Turn(
                    kind="result",
                    text=getattr(message, "result", "") or "",
                    payload={
                        "session_id": getattr(message, "session_id", None),
                        "turns": getattr(message, "num_turns", None),
                        "cost_usd": getattr(message, "total_cost_usd", None),
                        "duration_ms": getattr(message, "duration_ms", None),
                    },
                )


async def run_session(
    prompt: str,
    *,
    model: str | None = None,
    max_turns: int = 60,
    extra_system: str | None = None,
    on_turn: Callable[[Turn], None] | None = None,
    save_run: bool = True,
) -> dict[str, Any]:
    """Run one task to completion and return a transcript summary."""
    texts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}

    async for turn in stream_session(
        prompt, model=model, max_turns=max_turns, extra_system=extra_system
    ):
        if on_turn:
            on_turn(turn)
        if turn.kind == "text":
            texts.append(turn.text)
        elif turn.kind == "tool_use":
            tool_calls.append({"tool": turn.tool, "input": turn.payload})
        elif turn.kind == "result":
            meta = turn.payload

    answer = "\n".join(t for t in texts if t.strip())
    result = {
        "prompt": prompt,
        "answer": answer,
        "tool_calls": tool_calls,
        "n_tool_calls": len(tool_calls),
        "citations": len(registry()),
        "meta": meta,
        "finished": utcnow(),
    }

    problem = diagnose_auth_failure(answer, len(tool_calls))
    if problem:
        result["auth_problem"] = problem

    if save_run:
        result["run_dir"] = _save_run(result)
    return result


AUTH_SIGNATURES: list[tuple[str, str, str]] = [
    (
        "not logged in",
        "no credentials",
        "Authenticate once:  parallax login\n"
        "  or use an API key instead:  export ANTHROPIC_API_KEY=sk-ant-...",
    ),
    (
        "invalid api key",
        "ANTHROPIC_API_KEY is set but not accepted",
        "Check the key is current and has credit, or unset it to fall back to a logged-in "
        "session:  unset ANTHROPIC_API_KEY",
    ),
    (
        "credit balance",
        "the account has no credit",
        "Top up the Anthropic account, or use a subscription login instead of an API key.",
    ),
    (
        "please run /login",
        "no credentials",
        "Run `parallax login`, or set ANTHROPIC_API_KEY.",
    ),
]


def bundled_cli() -> Any:
    """Path to the claude binary that ships inside claude-agent-sdk, if present."""
    from pathlib import Path

    try:
        import claude_agent_sdk
    except ImportError:
        return None
    candidate = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"
    return candidate if candidate.exists() else None


def diagnose_auth_failure(answer: str, n_tool_calls: int) -> dict[str, Any] | None:
    """Spot an auth failure that would otherwise read like a one-line answer.

    The underlying CLI reports "Not logged in" or "Invalid API key" as ordinary
    assistant text and exits cleanly, so without this a user sees a cryptic
    sentence, zero tool calls, and no indication that nothing actually ran.
    """
    if n_tool_calls or not answer:
        return None
    lowered = answer.lower()
    if len(answer) > 400:
        return None
    for needle, reason, fix in AUTH_SIGNATURES:
        if needle in lowered:
            return {"reason": reason, "message": answer.strip(), "fix": fix}
    return None


def _save_run(result: dict[str, Any]) -> str:
    cfg = settings()
    stamp = utcnow().replace(":", "").replace("-", "")
    directory = cfg.runs_dir / stamp
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    (directory / "answer.md").write_text(result["answer"], encoding="utf-8")
    registry().save(directory)
    return str(directory)
