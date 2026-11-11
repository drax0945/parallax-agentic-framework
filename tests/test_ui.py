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
        assert [t["tier"] for t in scope["tiers"]] == [
            "observed", "measured", "computed", "modelled"
        ]  # same order as the prompt: in-house evidence first
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


class TestAnswerRendering:
    def test_a_table_after_prose_still_renders(self):
        from parallax.ui.server import render_markdown

        html = render_markdown("Here are the values:\n| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |")
        assert "<table>" in html
        assert html.count("<tr>") == 3  # header + two rows

    def test_rows_are_not_split_into_paragraphs(self):
        from parallax.ui.server import render_markdown

        html = render_markdown("Text:\n| A |\n|---|\n| 1 |\n| 2 |")
        assert "<table>" in html
        assert "<p>|" not in html

    def test_heading_after_prose_renders(self):
        from parallax.ui.server import render_markdown

        assert "<h2>" in render_markdown("Some text.\n## Results\nmore")

    def test_pipes_inside_a_code_fence_are_left_alone(self):
        from parallax.ui.server import render_markdown

        html = render_markdown("Run:\n```\n| not | a | table |\n```")
        assert "<table>" not in html

    def test_raw_html_in_the_answer_is_escaped(self):
        from parallax.ui.server import render_markdown

        assert "<script>" not in render_markdown("<script>alert(1)</script>")


class TestStructureExtraction:
    def test_real_smiles_are_recognised(self):
        from parallax.ui.server import looks_like_structure

        assert looks_like_structure("CC(=O)Oc1ccccc1C(=O)O")
        assert looks_like_structure("CCO")

    def test_prose_and_identifiers_are_not_mistaken_for_structures(self):
        from parallax.ui.server import looks_like_structure

        for text in ("ALK", "the answer is 5", "CHEMBL4247", "", "a potent inhibitor", None):
            assert not looks_like_structure(text)

    def test_structures_are_collected_from_nested_tool_payloads(self):
        from parallax.ui.server import collect_structures

        found: dict[str, str] = {}
        collect_structures(
            {"rows": [{"smiles": "CC(=O)Oc1ccccc1C(=O)O", "pref_name": "Aspirin"}], "count": 1},
            found,
        )
        assert "CC(=O)Oc1ccccc1C(=O)O" in found

    def test_depiction_returns_svg_or_none(self):
        from parallax.ui.server import depict

        assert (depict("CC(=O)Oc1ccccc1C(=O)O") or "").lstrip().startswith("<svg")
        assert depict("not a molecule") is None


class TestClaudeToolResults:
    """Tool results arrive on the user side of the Claude transcript; without
    them the console shows calls that start and never finish."""

    def test_result_text_flattens_block_shapes(self):
        from parallax.agent.loop import _result_text

        assert _result_text("plain") == "plain"
        assert _result_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a\nb"
        assert _result_text(None) == ""

    def test_summarise_text_handles_json_and_prose(self):
        import json

        from parallax.ui.server import _summarise_text

        assert "180" in _summarise_text(json.dumps({"summary": "MW is 180.16"}))
        assert _summarise_text("not json at all") == "not json at all"
        assert _summarise_text("") == ""

    def test_stream_handles_tool_result_turns(self):
        """run_session must not mistake a tool result for part of the answer."""
        import inspect

        from parallax.agent import loop

        source = inspect.getsource(loop.run_session)
        assert "tool_result" in source
        stream = inspect.getsource(loop.stream_session)
        assert "UserMessage" in stream and "ToolResultBlock" in stream


class TestLogbookInTheConsole:
    """The logbook is the top evidence tier, so the console has to show it."""

    def test_the_observed_tier_is_advertised_first(self):
        tiers = scope_payload()["tiers"]
        assert tiers[0]["tier"] == "observed"
        assert "this lab" in tiers[0]["meaning"].lower()

    def test_the_scope_carries_live_logbook_state(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PARALLAX_WORKSPACE", str(tmp_path))
        from parallax.config import settings

        settings.cache_clear()
        from parallax.logbook import add

        add(title="Suzuki trial", outcome="failure", date="2026-05-04",
            transform="Suzuki-Miyaura coupling", conditions={"solvent": "toluene"},
            observations="Protodeboronation.", project="X")

        book = scope_payload()["logbook"]
        assert book["n_entries"] == 1
        assert book["by_outcome"] == {"failure": 1}
        assert book["failures_recorded"] == 1
        assert book["recent"][0]["title"] == "Suzuki trial"
        assert book["date_range"] == ["2026-05-04", "2026-05-04"]

        # and it went to the temporary workspace, not the real one
        assert (tmp_path / "logbook" / "entries").exists()
        settings.cache_clear()

    def test_an_empty_logbook_is_reported_not_hidden(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PARALLAX_WORKSPACE", str(tmp_path))
        from parallax.config import settings

        settings.cache_clear()
        book = scope_payload()["logbook"]
        assert book["n_entries"] == 0 and book["recent"] == []
        settings.cache_clear()

    def test_the_page_renders_the_logbook_tab(self):
        import pathlib

        page = (pathlib.Path("src/parallax/ui/index.html")).read_text(encoding="utf-8")
        assert 'data-pane="logbook"' in page
        assert 'id="pane-logbook"' in page
        assert 'data-t="observed"' in page


class TestLogbookStatsShape:
    def test_the_search_index_stats_are_not_overwritten(self, tmp_path):
        """`index` was assigned twice in the same dict; the stats were lost."""
        from parallax.logbook.store import LogbookStore

        book = LogbookStore(root=tmp_path / "lb")
        book.add(title="x", outcome="success", product_smiles="CCO")
        summary = book.stats()
        assert summary["search_index"]["indexed_entries"] == 1
        assert summary["index"].endswith("LOGBOOK.md")  # the human index path

    def test_recent_is_newest_first(self, tmp_path):
        from parallax.logbook.store import LogbookStore

        book = LogbookStore(root=tmp_path / "lb")
        for day in ("2026-01-05", "2026-03-20", "2026-02-11"):
            book.add(title=f"run {day}", outcome="success", date=day)
        assert [r["date"] for r in book.recent()] == [
            "2026-03-20", "2026-02-11", "2026-01-05"
        ]


class TestStoresFollowTheWorkspace:
    """A store must never keep writing to the workspace it first saw."""

    def test_the_logbook_store_follows_a_workspace_change(self, tmp_path, monkeypatch):
        from parallax.config import settings
        from parallax.logbook import store

        first, second = tmp_path / "a", tmp_path / "b"
        monkeypatch.setenv("PARALLAX_WORKSPACE", str(first))
        settings.cache_clear()
        assert store().root == first / "logbook"

        monkeypatch.setenv("PARALLAX_WORKSPACE", str(second))
        settings.cache_clear()
        assert store().root == second / "logbook"  # was stuck on `first`
        settings.cache_clear()

    def test_the_memory_store_follows_a_workspace_change(self, tmp_path, monkeypatch):
        from parallax.config import settings
        from parallax.memory import store

        first, second = tmp_path / "a", tmp_path / "b"
        monkeypatch.setenv("PARALLAX_WORKSPACE", str(first))
        settings.cache_clear()
        assert store().root == first / "memory"

        monkeypatch.setenv("PARALLAX_WORKSPACE", str(second))
        settings.cache_clear()
        assert store().root == second / "memory"
        settings.cache_clear()
