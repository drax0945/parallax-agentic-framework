"""Memory, citations and tool dispatch."""

from __future__ import annotations

import pytest

from parallax.memory.store import MemoryStore
from parallax.provenance import Citation, CitationRegistry, cite_software, reset_registry


@pytest.fixture()
def memory(tmp_path):
    return MemoryStore(root=tmp_path / "memory")


class TestMemory:
    def test_write_and_recall(self, memory):
        memory.remember(
            type="heuristic",
            title="Closing a 12-membered macrolactam removes rotatable bonds",
            body="Seen in the crizotinib to lorlatinib series.",
            tags=["alk", "macrocycle"],
            citations=["doi:10.1021/jm1012374"],
            confidence=0.7,
        )
        hits = memory.recall("macrocycle rotatable bonds ALK", k=3)
        assert hits["n_results"] == 1
        assert hits["results"][0]["confidence"] == 0.7
        assert "doi:10.1021/jm1012374" in hits["results"][0]["citations"]

    def test_entries_round_trip_through_markdown(self, memory):
        created = memory.remember(
            type="finding",
            title="Lorlatinib ALK IC50 is 0.2 nM",
            body="Biochemical assay. Value from the discovery paper.",
            tags=["alk"],
            citations=["doi:10.1021/jm500261q"],
            entities={"targets": ["CHEMBL4247"]},
        )
        reloaded = MemoryStore(root=memory.root)
        entry = reloaded.get(created["id"])
        assert entry["title"] == "Lorlatinib ALK IC50 is 0.2 nM"
        assert entry["entities"]["targets"] == ["CHEMBL4247"]
        assert entry["citations"] == ["doi:10.1021/jm500261q"]

    def test_near_duplicates_corroborate_instead_of_piling_up(self, memory):
        first = memory.remember(
            type="heuristic",
            title="SNAr beats Buchwald at scale on electron-poor heteroaryl chlorides",
            body="No palladium means no metal removal burden in the API.",
        )
        second = memory.remember(
            type="heuristic",
            title="SNAr beats Buchwald at scale on electron poor heteroaryl chlorides",
            body="No palladium, so no metal removal burden in the API.",
        )
        assert second["created_new"] is False
        assert second["merged_into"] == first["id"]
        assert memory.stats()["active_entries"] == 1
        assert memory.get(first["id"])["support"] == 2

    def test_confidence_rises_on_reinforce_and_halves_on_contradict(self, memory):
        created = memory.remember(type="finding", title="X is Y", body="once observed", confidence=0.5)
        after_support = memory.reinforce(created["id"], note="seen again")
        assert after_support["confidence"] > 0.5
        after_conflict = memory.contradict(created["id"], note="a second source disagrees")
        assert after_conflict["confidence"] < after_support["confidence"]
        assert memory.get(created["id"])["contradictions"] == 1
        assert "contradicted" in memory.get(created["id"])["body"]

    def test_contradict_requires_a_note_via_the_tool(self):
        from parallax.agent.tools import call_tool

        result = call_tool("memory_update", {"entry_id": "nope", "action": "contradict"})
        assert "error" in result

    def test_superseded_entries_leave_recall(self, memory):
        old = memory.remember(type="finding", title="old claim", body="superseded soon")
        new = memory.remember(type="finding", title="corrected claim", body="this one is right")
        memory.supersede(old["id"], new["id"])
        titles = [r["title"] for r in memory.recall("claim", k=5)["results"]]
        assert "old claim" not in titles
        assert "corrected claim" in titles

    def test_human_index_is_written(self, memory):
        memory.remember(type="preference", title="Quote pIC50 not nM", body="House style.")
        index = memory.readme_path.read_text(encoding="utf-8")
        assert "Parallax memory" in index
        assert "Quote pIC50 not nM" in index

    def test_consolidate_merges_and_reports(self, memory):
        memory.remember(type="finding", title="alpha finding about kinase potency", body="detail one here")
        memory.remember(type="finding", title="beta finding about kinase potency", body="detail one here")
        result = memory.consolidate()
        assert "active_entries" in result
        assert isinstance(result["merged"], list)

    def test_unknown_type_rejected(self, memory):
        with pytest.raises(ValueError):
            memory.remember(type="nonsense", title="t", body="b")


class TestCitations:
    def test_doi_deduplication_merges_metadata(self):
        registry = CitationRegistry()
        first = registry.add(
            Citation(key="doi:10.1/x", kind="journal", title="A paper", source_db="a", doi="10.1/X")
        )
        second = registry.add(
            Citation(
                key="other-key",
                kind="journal",
                title="A paper",
                source_db="b",
                doi="https://doi.org/10.1/x",
                year=2020,
            )
        )
        assert first == second
        assert len(registry) == 1
        assert registry.get(first).year == 2020  # metadata merged in

    def test_reference_and_bibtex_render(self):
        registry = CitationRegistry()
        key = registry.add(
            Citation(
                key="doi:10.1021/jm500261q",
                kind="journal",
                title="Discovery of lorlatinib",
                source_db="test",
                authors=["Johnson, Ted W."],
                year=2014,
                venue="Journal of Medicinal Chemistry",
                doi="10.1021/jm500261q",
            )
        )
        assert "Johnson" in registry.get(key).reference()
        assert "doi.org/10.1021/jm500261q" in registry.get(key).reference()
        bibtex = registry.to_bibtex()
        assert bibtex.startswith("@article{")
        assert "doi = {10.1021/jm500261q}" in bibtex
        assert "Johnson, Ted W." in bibtex

    def test_audit_catches_dois_written_from_memory(self):
        """The registry cannot see prose. This is the check that can."""
        from parallax.provenance import Citation, audit_text, registry, reset_registry

        reset_registry()
        registry().add(
            Citation(
                key="doi:10.1021/jm500261q",
                kind="journal",
                title="retrieved by a tool",
                source_db="chembl",
                doi="10.1021/jm500261q",
            )
        )
        draft = (
            "Measured ALK Ki 0.9 nM [doi:10.1021/jm500261q]. "
            "Macrocyclisation helps [doi:10.1021/acs.jmedchem.6b00906]."
        )
        result = audit_text(draft, check_crossref=False)
        assert result["clean"] is False
        unbacked = [d["doi"] for d in result["unbacked_dois"]]
        assert "10.1021/acs.jmedchem.6b00906" in unbacked
        assert "10.1021/jm500261q" not in unbacked  # this one came from a tool

    def test_audit_passes_a_properly_sourced_draft(self):
        from parallax.provenance import Citation, audit_text, registry, reset_registry

        reset_registry()
        registry().add(
            Citation(
                key="doi:10.1021/jm500261q",
                kind="journal",
                title="t",
                source_db="chembl",
                doi="10.1021/jm500261q",
            )
        )
        result = audit_text("Everything here cites doi:10.1021/jm500261q only.", check_crossref=False)
        assert result["clean"] is True
        assert "traces to a tool result" in result["verdict"]

    def test_audit_flags_unknown_citation_keys(self):
        from parallax.provenance import audit_text, reset_registry

        reset_registry()
        result = audit_text("As shown in `chembl:CHEMBL999999`.", check_crossref=False)
        assert "chembl:CHEMBL999999" in result["unknown_citation_keys"]
        assert result["clean"] is False

    def test_bibliography_reports_requested_keys_it_does_not_have(self):
        from parallax.agent.tools import call_tool
        from parallax.provenance import reset_registry

        reset_registry()
        result = call_tool(
            "citation_bibliography", {"format": "markdown", "keys": ["doi:10.1/never-retrieved"]}
        )
        assert result["unknown_keys"] == ["doi:10.1/never-retrieved"]
        assert "no tool produced them" in result["warning"]

    def test_empty_registry_is_explicit(self):
        assert "No citations" in CitationRegistry().to_markdown()

    def test_software_citations_register(self):
        reset_registry()
        keys = cite_software("software:rdkit", "method:sa_score")
        assert "software:rdkit" in keys
        assert "method:sa_score" in keys


class TestToolDispatch:
    def test_unknown_tool_returns_error_not_exception(self):
        from parallax.agent.tools import call_tool

        result = call_tool("no_such_tool", {})
        assert "error" in result
        assert "available" in result

    def test_bad_arguments_return_the_schema(self):
        from parallax.agent.tools import call_tool

        result = call_tool("computed_properties", {"wrong_arg": 1})
        assert "error" in result
        assert "expected_schema" in result

    def test_handler_exception_is_wrapped(self):
        from parallax.agent.tools import call_tool

        result = call_tool("computed_properties", {"smiles": "definitely not a molecule"})
        assert "error" in result
        assert result["tool"] == "computed_properties"

    def test_every_tool_has_a_schema_and_description(self):
        from parallax.agent.tools import TOOLS

        for tool in TOOLS:
            assert tool.description.strip()
            assert tool.input_schema["type"] == "object"
            assert len(tool.description) > 40, f"{tool.name} description is too thin"

    def test_tool_names_are_unique(self):
        from parallax.agent.tools import TOOLS

        names = [t.name for t in TOOLS]
        assert len(names) == len(set(names))

    def test_mcp_server_builds_with_all_tools(self):
        from parallax.agent.mcp_server import build_server
        from parallax.agent.tools import TOOLS

        server = build_server()
        assert server is not None
        assert len(TOOLS) > 30
