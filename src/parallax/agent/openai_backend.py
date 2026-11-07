"""A provider-agnostic agent loop over any OpenAI-compatible chat endpoint.

Why this exists
---------------
The Claude backend is excellent but ties the conversational agent to one
account. The chemistry underneath needs no particular model, so the agent
should not either. This loop speaks the `/chat/completions` tool-calling
protocol, which is implemented by essentially everything:

* **Ollama** — local, free, private, no account at all
* **OpenAI, Mistral, Groq, Together, Fireworks, DeepSeek, OpenRouter**
* **vLLM / llama.cpp / LM Studio** — self-hosted
* **Azure OpenAI** and other enterprise gateways

It is the same tool set, the same skills and the same memory as the Claude
backend; only the thing doing the reasoning changes.

A caveat worth stating plainly: the discipline this framework depends on —
separating measured from computed from modelled, refusing to quote an
out-of-domain prediction, never inventing a citation — is enforced partly by
the tools and partly by the model's willingness to follow instructions. A small
local model will follow them less reliably than a frontier one. The tools still
refuse to fabricate, and `audit_draft` still catches invented references, so
the floor holds; the ceiling depends on the model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from ..config import settings, utcnow
from ..net import SourceError
from ..provenance import registry
from .prompts import build_system_prompt
from .skills import load_skill, skill_index
from .tools import TOOLS, call_tool

DEFAULT_TIMEOUT = 300.0


@dataclass
class Provider:
    name: str
    base_url: str
    model: str
    api_key: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"


PRESETS: dict[str, dict[str, str]] = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5:14b",
        "key_env": "",
        "note": "local and free; no account. Install ollama, then `ollama pull qwen2.5:14b`. "
        "Pick a model with solid tool-calling — qwen2.5, llama3.1 or mistral-nemo.",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "key_env": "OPENAI_API_KEY",
        "note": "pay per token",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "anthropic/claude-sonnet-4.5",
        "key_env": "OPENROUTER_API_KEY",
        "note": "one key, many models including Claude",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "key_env": "GROQ_API_KEY",
        "note": "fast, generous free tier",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "key_env": "TOGETHER_API_KEY",
        "note": "hosted open models",
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "model": "mistral-large-latest",
        "key_env": "MISTRAL_API_KEY",
        "note": "hosted, EU-based",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "key_env": "DEEPSEEK_API_KEY",
        "note": "inexpensive",
    },
}


def resolve_provider(
    name: str | None = None, *, model: str | None = None, base_url: str | None = None
) -> Provider:
    """Work out which endpoint to talk to, from flags then environment."""
    cfg = settings()
    name = (name or cfg.credential("PARALLAX_LLM_PROVIDER") or "").lower().strip()

    if base_url or cfg.credential("PARALLAX_LLM_BASE_URL"):
        url = base_url or cfg.credential("PARALLAX_LLM_BASE_URL")
        return Provider(
            name=name or "custom",
            base_url=url,
            model=model or cfg.credential("PARALLAX_LLM_MODEL") or "gpt-4o",
            api_key=cfg.credential("PARALLAX_LLM_API_KEY"),
        )

    if not name:
        # Pick the first preset whose key is actually present.
        for candidate, spec in PRESETS.items():
            if spec["key_env"] and cfg.credential(spec["key_env"]):
                name = candidate
                break
        else:
            name = "ollama"  # the only one needing no account

    spec = PRESETS.get(name)
    if spec is None:
        raise ValueError(
            f"unknown provider {name!r}; known: {sorted(PRESETS)}. "
            "For anything else set PARALLAX_LLM_BASE_URL to its OpenAI-compatible endpoint."
        )
    return Provider(
        name=name,
        base_url=spec["base_url"],
        model=model or cfg.credential("PARALLAX_LLM_MODEL") or spec["model"],
        api_key=cfg.credential(spec["key_env"]) if spec["key_env"] else None,
    )


def available_providers() -> list[dict[str, Any]]:
    cfg = settings()
    out = []
    for name, spec in PRESETS.items():
        key_present = bool(cfg.credential(spec["key_env"])) if spec["key_env"] else None
        out.append(
            {
                "provider": name,
                "model": spec["model"],
                "needs_key": spec["key_env"] or None,
                "key_present": key_present,
                "ready": key_present if spec["key_env"] else "if ollama is running",
                "note": spec["note"],
            }
        )
    return out


# --- tool schema translation -------------------------------------------


def _openai_tools() -> list[dict[str, Any]]:
    """Parallax tools plus `load_skill`, in OpenAI function format."""
    functions = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description[:1024],
                "parameters": tool.input_schema,
            },
        }
        for tool in TOOLS
    ]
    functions.append(
        {
            "type": "function",
            "function": {
                "name": "load_skill",
                "description": (
                    "Read the full procedure for one of the available skills. Call this "
                    "before starting a task the skill covers — it encodes how to avoid the "
                    "mistakes that matter in this domain."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
            },
        }
    )
    return functions


def _dispatch(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "load_skill":
        return load_skill(arguments.get("name", ""))
    return call_tool(name, arguments)


# --- the loop ------------------------------------------------------------


@dataclass
class Step:
    kind: str  # text | tool_use | tool_result | error
    text: str = ""
    tool: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


def run(
    prompt: str,
    *,
    provider: Provider | None = None,
    max_turns: int = 40,
    temperature: float = 0.0,
    on_step: Callable[[Step], None] | None = None,
) -> dict[str, Any]:
    """Drive the tools with any OpenAI-compatible model."""
    import httpx

    target = provider or resolve_provider()
    system = build_system_prompt()
    index = skill_index()
    if index:
        system += "\n\n---\n\n" + index

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    tools = _openai_tools()
    tool_calls_made: list[dict[str, Any]] = []
    texts: list[str] = []

    headers = {"Content-Type": "application/json", **target.extra_headers}
    if target.api_key:
        headers["Authorization"] = f"Bearer {target.api_key}"

    with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
        for _ in range(max_turns):
            try:
                response = client.post(
                    target.endpoint,
                    headers=headers,
                    json={
                        "model": target.model,
                        "messages": messages,
                        "tools": tools,
                        "tool_choice": "auto",
                        "temperature": temperature,
                    },
                )
            except httpx.HTTPError as exc:
                return _failure(
                    target, f"could not reach {target.endpoint}: {exc}", texts, tool_calls_made
                )

            if response.status_code >= 400:
                return _failure(
                    target,
                    f"HTTP {response.status_code} from {target.name}: {response.text[:300]}",
                    texts,
                    tool_calls_made,
                )

            try:
                body = response.json()
                choice = body["choices"][0]["message"]
            except (ValueError, KeyError, IndexError) as exc:
                return _failure(
                    target, f"unexpected response shape: {exc}", texts, tool_calls_made
                )

            content = choice.get("content") or ""
            requested = choice.get("tool_calls") or []

            if content.strip():
                texts.append(content)
                if on_step:
                    on_step(Step(kind="text", text=content))

            if not requested:
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": requested,
                }
            )

            for call in requested:
                function = call.get("function", {})
                name = function.get("name", "")
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                if on_step:
                    on_step(Step(kind="tool_use", tool=name, payload=arguments))

                try:
                    result = _dispatch(name, arguments)
                except SourceError as exc:
                    result = {"error": exc.message, "source": exc.source}
                except Exception as exc:  # the loop must survive a bad tool call
                    result = {"error": f"{type(exc).__name__}: {exc}"}

                tool_calls_made.append({"tool": name, "input": arguments})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", name),
                        "content": json.dumps(result, default=str)[:60000],
                    }
                )
                if on_step:
                    on_step(Step(kind="tool_result", tool=name, payload=result))

    return {
        "prompt": prompt,
        "answer": "\n".join(t for t in texts if t.strip()),
        "tool_calls": tool_calls_made,
        "n_tool_calls": len(tool_calls_made),
        "citations": len(registry()),
        "provider": {"name": target.name, "model": target.model, "endpoint": target.endpoint},
        "finished": utcnow(),
    }


def _failure(
    target: Provider, message: str, texts: list[str], calls: list[dict[str, Any]]
) -> dict[str, Any]:
    advice = (
        "Is ollama running? Start it with `ollama serve`, then `ollama pull "
        f"{target.model}`."
        if target.name == "ollama"
        else f"Check the endpoint and that the API key for {target.name} is set and valid."
    )
    return {
        "error": message,
        "advice": advice,
        "provider": {"name": target.name, "model": target.model, "endpoint": target.endpoint},
        "answer": "\n".join(texts),
        "tool_calls": calls,
        "n_tool_calls": len(calls),
        "citations": len(registry()),
    }
