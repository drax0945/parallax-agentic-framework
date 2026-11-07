"""The console's server layer: scope, runtime reporting and the grounding check."""

from __future__ import annotations

import json

import pytest

from parallax.ui.server import is_ungrounded, runtime_payload, scope_payload


class TestScope:
    def test_scope_describes_the_real_system(self):
        from parallax.agent.tools import TOOLS

        scope = scope_payload()
        assert scope["n_tools"] == len(TOOLS)
        assert len(scope["skills"]) >= 11
        assert {t["tier"] for t in scope["tiers"]} == {"measured", "computed", "modelled"}
        assert scope["n_reaction_templates"] >= 30
        assert scope["limits"] and scope["chemical_scope"]

    def test_scope_is_json_serialisable(self):
        json.dumps(scope_payload(), default=str)

    def test_every_tool_appears_in_some_group(self):
        from parallax.agent.tools import TOOLS

        scope = scope_payload()
        listed = {t["name"] for group in scope["tool_groups"].values() for t in group}
        assert listed == {t.name for t in TOOLS}

    def test_organometallics_are_declared_unsupported(self):
        areas = " ".join(c["area"] + c["status"] for c in scope_payload()["chemical_scope"])
        assert "rganometallic" in areas
        assert "not supported" in areas


class TestRuntimeReporting:
    def test_claude_is_the_default_backend(self):
        payload = runtime_payload()
        assert payload["backend"] == "claude"
        assert payload["local"] is False

    def test_ollama_is_reported_as_local(self):
        payload = runtime_payload("ollama")
        assert payload["backend"] == "ollama"
        assert payload["local"] is True

    def test_unknown_provider_is_reported_not_raised(self):
        payload = runtime_payload("nonsense-provider")
        assert payload["ready"] is False

    def test_runtime_is_json_serialisable(self):
        for provider in (None, "ollama", "groq"):
            json.dumps(runtime_payload(provider), default=str)


class TestGroundingCheck:
    def test_fabricated_numbers_after_a_failed_tool_are_flagged(self):
        # the exact failure seen from a 3B local model: tool rejected, answer invented
        answer = "The output is: {'molecular_weight': 280.3}"
        assert is_ungrounded(answer, citations=0, tools_ok=0) is True

    def test_a_properly_sourced_answer_is_not_flagged(self):
        assert is_ungrounded("Aspirin is 180.16 g/mol.", citations=12, tools_ok=1) is False

    def test_a_successful_tool_call_grounds_the_answer(self):
        assert is_ungrounded("180.16", citations=0, tools_ok=1) is False

    def test_prose_without_numbers_is_not_flagged(self):
        assert is_ungrounded("I could not find that compound.", citations=0, tools_ok=0) is False

    def test_empty_answer_is_not_flagged(self):
        assert is_ungrounded("", citations=0, tools_ok=0) is False
