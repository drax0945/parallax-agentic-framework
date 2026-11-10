"""The lab logbook: primary records, structural recall, and in-house precedence."""

from __future__ import annotations

import pytest

from parallax.logbook.store import LogbookStore, split_reaction

AMIDE = "CC(=O)O.Nc1c(Cl)cccc1Cl>>CC(=O)Nc1c(Cl)cccc1Cl"
PRODUCT = "CC(=O)Nc1c(Cl)cccc1Cl"


@pytest.fixture()
def book(tmp_path):
    store = LogbookStore(root=tmp_path / "logbook")
    store.add(
        title="HATU coupling onto 2,6-dichloroaniline", outcome="failure",
        reaction_smiles=AMIDE, transform="Amide coupling (acid + amine)",
        conditions={"reagent": "HATU", "solvent": "DMF"},
        observations="No conversion by LCMS after 18 h. Aniline too hindered.",
        project="ALK series",
    )
    store.add(
        title="T3P coupling, same substrate", outcome="success", yield_percent=71,
        reaction_smiles=AMIDE, transform="Amide coupling (acid + amine)",
        conditions={"reagent": "T3P", "solvent": "EtOAc"},
        observations="Clean conversion at 80 C.", project="ALK series",
    )
    store.add(
        title="Suzuki on bromopyridine", outcome="success", yield_percent=88,
        reaction_smiles="Brc1ccncc1.OB(O)c1ccccc1>>c1ccc(-c2ccncc2)cc1",
        transform="Suzuki-Miyaura coupling", project="ALK series",
    )
    return store


class TestRecording:
    def test_reaction_smiles_is_split_into_reactants_and_product(self):
        reactants, product = split_reaction(AMIDE)
        assert reactants == ["CC(=O)O", "Nc1c(Cl)cccc1Cl"]
        assert product == "CC(=O)Nc1c(Cl)cccc1Cl"

    def test_agents_notation_with_middle_section_is_tolerated(self):
        reactants, product = split_reaction("CCO>[H+]>CC=O")
        assert reactants == ["CCO"] and product == "CC=O"

    def test_a_bad_outcome_is_rejected(self, book):
        with pytest.raises(ValueError, match="outcome must be"):
            book.add(title="x", outcome="worked fine")

    def test_entries_round_trip_through_markdown(self, book):
        reloaded = LogbookStore(root=book.root)
        assert len(reloaded.all_entries()) == 3
        entry = next(e for e in reloaded.all_entries() if e.outcome == "failure")
        assert entry.conditions["reagent"] == "HATU"
        assert entry.transform == "Amide coupling (acid + amine)"
        assert "hindered" in entry.observations

    def test_structures_are_canonicalised_on_the_way_in(self, tmp_path):
        store = LogbookStore(root=tmp_path / "lb")
        store.add(title="a", outcome="success", product_smiles="OC(=O)C")  # non-canonical
        assert store.all_entries()[0].product_smiles == "CC(=O)O"

    def test_a_human_readable_index_is_written(self, book):
        index = book.index_path.read_text(encoding="utf-8")
        assert "Lab logbook" in index and "ALK series" in index
        assert "FAILURE" in index


class TestStructuralRecall:
    def test_product_structure_finds_runs_with_no_shared_words(self, book):
        """The point of structural search: wording differs, chemistry does not."""
        hits = book.search(product_smiles=PRODUCT)["results"]
        assert len(hits) == 2
        assert all(h["matched_on"].get("product_similarity") == 1.0 for h in hits)

    def test_an_unrelated_product_does_not_match(self, book):
        assert book.search(product_smiles="CCO")["n_results"] == 0

    def test_reaction_smiles_is_accepted_directly(self, book):
        assert book.search(reaction_smiles=AMIDE)["n_results"] == 2

    def test_text_search_still_works(self, book):
        hits = book.search("hindered aniline no conversion")["results"]
        assert hits and hits[0]["outcome"] == "failure"

    def test_transform_name_matches_exactly_and_loosely(self, book):
        assert book.search(transform="Amide coupling (acid + amine)")["n_results"] == 2
        assert book.search(transform="Amide coupling")["n_results"] == 2

    def test_results_can_be_filtered_to_failures(self, book):
        hits = book.search(product_smiles=PRODUCT, outcome="failure")["results"]
        assert len(hits) == 1 and hits[0]["outcome"] == "failure"

    def test_an_empty_logbook_says_so_rather_than_failing(self, tmp_path):
        result = LogbookStore(root=tmp_path / "empty").search("anything")
        assert result["n_results"] == 0
        assert "empty" in result["note"].lower()


class TestTrackRecord:
    def test_success_rate_is_computed_from_real_runs(self, book):
        record = book.outcomes_for_transform("Amide coupling")
        assert record["n_runs"] == 2
        assert record["in_house_success_rate"] == 0.5
        assert record["median_yield_percent"] == 71.0

    def test_failures_are_returned_with_their_notes(self, book):
        failures = book.outcomes_for_transform("Amide coupling")["failures"]
        assert len(failures) == 1
        assert "hindered" in failures[0]["observations"]

    def test_a_rate_always_carries_its_sample_size(self, book):
        assert "2 run(s)" in book.outcomes_for_transform("Amide coupling")["caveat"]

    def test_an_untried_transform_defers_to_the_template_prior(self, book):
        record = book.outcomes_for_transform("Mitsunobu etherification")
        assert record["enough_data"] is False
        assert "template prior" in record["note"]

    def test_stats_highlight_recorded_failures(self, book):
        stats = book.stats()
        assert stats["n_entries"] == 3
        assert stats["failures_recorded"] == 1
        assert stats["with_structures"] == 3


class TestPrecedenceInTheAgentSurface:
    def test_precedent_search_reports_in_house_separately(self):
        from parallax.agent.tools import call_tool

        result = call_tool("reaction_precedent_search", {"query": "amide coupling", "limit": 2})
        assert "in_house" in result
        assert "weight" in result["in_house"]

    def test_logbook_tools_are_registered(self):
        from parallax.agent.tools import TOOL_MAP

        for name in ("logbook_add", "logbook_search", "logbook_track_record", "logbook_stats"):
            assert name in TOOL_MAP
        assert TOOL_MAP["logbook_add"].read_only is False
        assert TOOL_MAP["logbook_search"].read_only is True

    def test_the_prompt_ranks_observed_above_measured(self):
        from parallax.agent.prompts import SYSTEM_PROMPT

        assert "**observed**" in SYSTEM_PROMPT
        assert SYSTEM_PROMPT.index("**observed**") < SYSTEM_PROMPT.index("**measured**")
        assert "outranks everything below" in SYSTEM_PROMPT
