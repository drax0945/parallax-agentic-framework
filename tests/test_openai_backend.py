"""The provider-agnostic loop, driven by a stub OpenAI-compatible server.

This proves the protocol handling end to end — tool schema translation, the
tool-call round trip, skill loading, error survival — without needing any model
provider or network access.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from parallax.agent.openai_backend import (
    PRESETS,
    Provider,
    _openai_tools,
    available_providers,
    resolve_provider,
    run,
)


class _StubModel:
    """Replays a scripted sequence of assistant turns."""

    def __init__(self, script):
        self.script = list(script)
        self.seen_requests = []


def _make_server(stub: _StubModel) -> tuple[HTTPServer, str]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            stub.seen_requests.append(body)
            message = stub.script.pop(0) if stub.script else {"content": "done"}
            payload = {"choices": [{"message": {"role": "assistant", **message}}]}
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}/v1"


def _tool_call(name: str, arguments: dict, call_id: str = "c1") -> dict:
    return {
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


class TestToolTranslation:
    def test_every_tool_is_exposed_in_openai_format(self):
        from parallax.agent.tools import TOOLS

        functions = _openai_tools()
        # all Parallax tools, plus load_skill
        assert len(functions) == len(TOOLS) + 1
        names = {f["function"]["name"] for f in functions}
        assert "load_skill" in names
        assert "predict_potency" in names

    def test_schemas_are_wellformed_for_the_api(self):
        for function in _openai_tools():
            spec = function["function"]
            assert spec["name"] and spec["description"]
            assert spec["parameters"]["type"] == "object"
            # OpenAI rejects descriptions beyond 1024 chars
            assert len(spec["description"]) <= 1024


class TestProviderResolution:
    def test_known_presets_resolve(self):
        for name in PRESETS:
            provider = resolve_provider(name)
            assert provider.endpoint.endswith("/chat/completions")
            assert provider.model

    def test_unknown_provider_is_rejected_with_guidance(self):
        with pytest.raises(ValueError, match="unknown provider"):
            resolve_provider("not-a-provider")

    def test_custom_base_url_wins(self):
        provider = resolve_provider(base_url="http://example.test/v1", model="my-model")
        assert provider.endpoint == "http://example.test/v1/chat/completions"
        assert provider.model == "my-model"

    def test_ollama_needs_no_key(self):
        entry = next(p for p in available_providers() if p["provider"] == "ollama")
        assert entry["needs_key"] is None


class TestLoop:
    def test_a_tool_call_round_trip_executes_real_chemistry(self):
        stub = _StubModel(
            [
                _tool_call("computed_properties", {"smiles": "CC(=O)Oc1ccccc1C(=O)O"}),
                {"content": "Aspirin has a molecular weight of 180.16."},
            ]
        )
        server, base = _make_server(stub)
        try:
            result = run(
                "What is aspirin's molecular weight?",
                provider=Provider(name="stub", base_url=base, model="stub-model"),
            )
        finally:
            server.shutdown()

        assert result["n_tool_calls"] == 1
        assert result["tool_calls"][0]["tool"] == "computed_properties"
        assert "180.16" in result["answer"]
        # the real tool ran: its output went back to the model
        tool_message = next(
            m for m in stub.seen_requests[-1]["messages"] if m.get("role") == "tool"
        )
        assert "180.16" in tool_message["content"]

    def test_skills_are_advertised_and_loadable(self):
        stub = _StubModel(
            [
                _tool_call("load_skill", {"name": "potency-modelling"}),
                {"content": "Loaded the skill."},
            ]
        )
        server, base = _make_server(stub)
        try:
            result = run(
                "How should I model potency?",
                provider=Provider(name="stub", base_url=base, model="stub-model"),
            )
        finally:
            server.shutdown()

        system = stub.seen_requests[0]["messages"][0]["content"]
        assert "Available skills" in system
        assert "potency-modelling" in system
        tool_message = next(
            m for m in stub.seen_requests[-1]["messages"] if m.get("role") == "tool"
        )
        assert "scaffold" in tool_message["content"].lower()
        assert result["n_tool_calls"] == 1

    def test_a_failing_tool_does_not_kill_the_loop(self):
        stub = _StubModel(
            [
                _tool_call("computed_properties", {"smiles": "not a molecule"}),
                {"content": "That structure could not be parsed."},
            ]
        )
        server, base = _make_server(stub)
        try:
            result = run(
                "Profile this",
                provider=Provider(name="stub", base_url=base, model="stub-model"),
            )
        finally:
            server.shutdown()
        assert result["n_tool_calls"] == 1
        assert "could not be parsed" in result["answer"]

    def test_unreachable_endpoint_is_reported_with_advice(self):
        result = run(
            "hello",
            provider=Provider(name="ollama", base_url="http://127.0.0.1:1/v1", model="x"),
        )
        assert result["error"]
        assert "ollama" in result["advice"].lower()

    def test_http_error_is_surfaced_not_swallowed(self):
        class ErrHandler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                self.send_response(401)
                self.send_header("Content-Length", "0")
                self.end_headers()

        server = HTTPServer(("127.0.0.1", 0), ErrHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            result = run(
                "hello",
                provider=Provider(
                    name="openai", base_url=f"http://127.0.0.1:{server.server_port}/v1", model="x"
                ),
            )
        finally:
            server.shutdown()
        assert "401" in result["error"]
