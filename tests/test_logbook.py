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


class TestIndexScaling:
    """The markdown stays authoritative; the index only makes it fast."""

    def test_search_runs_through_the_index(self, book):
        index = book.index()
        assert len(index) == 3
        assert index.stats()["indexed_entries"] == 3
        assert (book.root / ".index" / "fingerprints.npy").exists()

    def test_a_hand_edited_file_is_picked_up(self, book):
        target = next(book.entries_dir.glob("*hatu*.md"))
        target.write_text(
            target.read_text(encoding="utf-8").replace(
                "Aniline too hindered.", "Aniline too hindered. Retried with T3P."
            ),
            encoding="utf-8",
        )
        hits = LogbookStore(root=book.root).search("retried T3P")["results"]
        assert hits and "Retried with T3P" in hits[0]["observations"]

    def test_a_deleted_file_leaves_the_index(self, book):
        next(book.entries_dir.glob("*suzuki*.md")).unlink()
        assert LogbookStore(root=book.root).search("bromopyridine")["store_size"] == 2

    def test_deleting_the_index_loses_nothing(self, book):
        import shutil

        book.index()  # make sure there is one to delete
        shutil.rmtree(book.root / ".index")
        rebuilt = LogbookStore(root=book.root)
        assert rebuilt.search(product_smiles=PRODUCT)["n_results"] == 2

    def test_a_corrupt_index_is_rebuilt_rather_than_raising(self, book):
        book.index()
        (book.root / ".index" / "meta.json").write_text("{ not json", encoding="utf-8")
        assert LogbookStore(root=book.root).search(product_smiles=PRODUCT)["n_results"] == 2

    def test_unchanged_files_are_not_reparsed(self, book):
        from parallax.logbook.store import Experiment

        book.index()  # first pass parses everything
        changes = LogbookStore(root=book.root).index().refresh(parse=Experiment.from_markdown)
        assert changes["reparsed"] == 0 and changes["added"] == 0
        assert changes["unchanged"] == 3

    def test_a_narrow_query_does_not_score_the_whole_store(self, book):
        result = book.search("hindered aniline")
        assert result["candidates_considered"] < result["store_size"]

    def test_bulk_tanimoto_matches_rdkit(self):
        import numpy as np
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator

        from parallax.logbook.index import bulk_tanimoto, pack_fingerprint

        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        smiles = ["CC(=O)Nc1ccccc1", "CC(=O)Nc1c(Cl)cccc1Cl", "CCO"]
        packed = [pack_fingerprint(s) for s in smiles]
        matrix = np.vstack([np.frombuffer(p, dtype=np.uint8) for p in packed])
        from parallax.logbook.index import _POPCOUNT

        popcounts = _POPCOUNT[matrix].sum(axis=1).astype(np.float32)
        ours = bulk_tanimoto(matrix, popcounts, packed[0])

        reference = [
            DataStructs.TanimotoSimilarity(
                gen.GetFingerprint(Chem.MolFromSmiles(smiles[0])),
                gen.GetFingerprint(Chem.MolFromSmiles(s)),
            )
            for s in smiles
        ]
        assert np.allclose(ours, reference, atol=1e-6)


@pytest.fixture()
def dated_book(tmp_path):
    """Three runs on three days, each recording a different set of details.

    The field names are deliberately inconsistent — `catalyst` in one entry,
    `cat` in another, `stir rate` and `eln` in a third. A logbook does not get
    to dictate vocabulary, and retrieval must not depend on one.
    """
    store = LogbookStore(root=tmp_path / "dated")
    store.add(
        title="Suzuki on the chloropyridine core", outcome="success", yield_percent=64,
        date="2026-03-12", operator="RK", scale="250 mg",
        reaction_smiles="Clc1ccncc1.OB(O)c1ccccc1>>c1ccc(-c2ccncc2)cc1",
        transform="Suzuki-Miyaura coupling",
        conditions={"catalyst": "Pd(OAc)2", "ligand": "SPhos", "base": "K3PO4",
                    "solvent": "dioxane/water", "temperature": "100 C", "time": "3 h"},
        observations="Full conversion in 3 h.", analysis="LCMS, single peak",
        project="ALK series", links={"eln": "ELN-4471"},
    )
    store.add(
        title="Suzuki retry on a bigger scale", outcome="failure",
        date="2026-03-14", operator="RK", scale="12 g",
        reaction_smiles="Clc1ccncc1.OB(O)c1ccccc1>>c1ccc(-c2ccncc2)cc1",
        transform="Suzuki-Miyaura coupling",
        conditions={"cat": "Pd2(dba)3", "solv": "toluene", "stir rate": "600 rpm",
                    "degassing": "three freeze-pump-thaw cycles"},
        observations="Protodeboronation dominated.", project="ALK series",
    )
    store.add(
        title="Buchwald amination of the same core", outcome="success", yield_percent=52,
        date="2026-04-02", operator="JD", transform="Buchwald-Hartwig amination",
        conditions={"catalyst": "Pd(OAc)2", "base": "NaOtBu"},
        project="ALK series",
    )
    return store


class TestDateAddressableRecords:
    """A run is identified by when it happened, not only by what it made."""

    def test_a_date_pins_one_day(self, dated_book):
        hits = dated_book.search(date="2026-03-12")["results"]
        assert len(hits) == 1
        assert hits[0]["title"] == "Suzuki on the chloropyridine core"

    def test_a_bare_date_needs_no_search_text(self, dated_book):
        """'What did we run on the 14th' is a complete question."""
        result = dated_book.search(date="2026-03-14")
        assert result["n_results"] == 1
        assert result["ordering"] == "date, newest first"
        assert result["filters"] == {"date": "2026-03-14"}

    def test_a_window_bounds_both_ends(self, dated_book):
        march = dated_book.search(date_from="2026-03-01", date_to="2026-03-31")["results"]
        assert len(march) == 2
        assert [h["date"] for h in march] == ["2026-03-14", "2026-03-12"]  # newest first

    def test_an_open_ended_window_works(self, dated_book):
        assert dated_book.search(date_from="2026-04-01")["n_results"] == 1
        assert dated_book.search(date_to="2026-03-12")["n_results"] == 1

    def test_a_dated_hit_carries_the_whole_record(self, dated_book):
        """Any question about a run, answered from the record rather than guessed."""
        hits = dated_book.search("Suzuki", date="2026-03-12")["results"]
        assert len(hits) == 1
        record = hits[0]["record"]

        # whatever was asked about, it is in there, under the name it was written
        assert record["conditions"]["catalyst"] == "Pd(OAc)2"
        assert record["conditions"]["time"] == "3 h"
        assert record["operator"] == "RK"
        assert record["scale"] == "250 mg"
        assert record["yield_percent"] == 64
        assert record["analysis"] == "LCMS, single peak"
        assert record["links"]["eln"] == "ELN-4471"
        assert record["date"] == "2026-03-12"

    def test_the_record_is_the_entry_and_nothing_added(self, dated_book):
        from dataclasses import asdict

        hit = dated_book.search(date="2026-04-02")["results"][0]
        assert hit["record"] == asdict(dated_book.get(hit["id"]))

    def test_an_absent_field_is_absent_not_invented(self, dated_book):
        """The Buchwald entry records no solvent. Nothing may supply one."""
        record = dated_book.search(date="2026-04-02")["results"][0]["record"]
        assert "solvent" not in record["conditions"]
        assert record["scale"] is None
        assert record["conditions"] == {"catalyst": "Pd(OAc)2", "base": "NaOtBu"}

    def test_a_date_narrows_an_otherwise_ambiguous_query(self, dated_book):
        """Two Suzukis on the same substrate; the date separates them."""
        both = dated_book.search(transform="Suzuki-Miyaura coupling")["results"]
        assert len(both) == 2
        one = dated_book.search(transform="Suzuki-Miyaura coupling", date="2026-03-14")["results"]
        assert len(one) == 1 and one[0]["conditions"]["cat"] == "Pd2(dba)3"

    def test_a_date_filter_combines_with_outcome(self, dated_book):
        result = dated_book.search(date_from="2026-03-01", date_to="2026-03-31", outcome="failure")
        assert result["n_results"] == 1
        assert result["results"][0]["date"] == "2026-03-14"

    def test_a_month_name_reaches_the_entry(self, dated_book):
        """Nobody types an ISO date half the time."""
        hits = dated_book.search("suzuki run in march")["results"]
        assert len(hits) == 2
        assert all(h["date"].startswith("2026-03") for h in hits)

    def test_the_year_alone_does_not_surface_an_entry(self, dated_book):
        """A shared year is too weak to be a reason for a hit; the filter is."""
        top = dated_book.search("suzuki 2026")["results"][:2]
        assert all(h["date"].startswith("2026-03") for h in top)

    def test_an_unused_date_returns_nothing_rather_than_the_nearest(self, dated_book):
        assert dated_book.search(date="2026-03-13")["n_results"] == 0


class TestTheRecordIsNotRewritten:
    """A logbook entry is a primary record. Nothing renames what a lab wrote."""

    def test_keys_are_stored_exactly_as_given(self, tmp_path):
        store = LogbookStore(root=tmp_path / "lb")
        store.add(
            title="x", outcome="success",
            conditions={"microwave power": "150 W", "Cat": "CuI", "pH": "7.40"},
        )
        conditions = store.all_entries()[0].conditions
        assert conditions == {"microwave power": "150 W", "Cat": "CuI", "pH": "7.40"}

    def test_a_number_keeps_the_figures_it_was_written_with(self, tmp_path):
        """Conditions round-trip as written text, deliberately.

        Coercing `7.40` to a float would return `7.4` and throw away a
        significant figure the chemist chose. A primary record comes back the
        way it went in; the typed fields (`yield_percent`) are where numbers
        are numbers.
        """
        store = LogbookStore(root=tmp_path / "lb")
        store.add(title="x", outcome="success", yield_percent=64,
                  conditions={"pH": 7.40, "equiv": 1.50})
        entry = LogbookStore(root=store.root).all_entries()[0]
        assert entry.conditions == {"pH": "7.4", "equiv": "1.5"}  # as python wrote them
        assert entry.yield_percent == 64.0  # a typed field stays a number

    def test_hand_written_keys_survive_a_reload(self, dated_book):
        """A chemist editing the markdown keeps their own field names."""
        target = next(dated_book.entries_dir.glob("*buchwald*.md"))
        target.write_text(
            target.read_text(encoding="utf-8").replace(
                "  catalyst: Pd(OAc)2", "  Pd source: Pd(OAc)2"
            ),
            encoding="utf-8",
        )
        hit = LogbookStore(root=dated_book.root).search(date="2026-04-02")["results"][0]
        assert hit["conditions"]["Pd source"] == "Pd(OAc)2"
        assert "catalyst" not in hit["conditions"]

    def test_a_field_name_this_lab_invented_is_searchable(self, dated_book):
        """Whatever a lab records becomes reachable by the words it used."""
        for term in ("stir rate", "degassing", "freeze-pump-thaw", "ELN-4471"):
            assert dated_book.search(term)["n_results"] >= 1, term

    def test_searching_by_a_field_name_finds_entries_that_have_it(self, dated_book):
        hits = dated_book.search("degassing")["results"]
        assert len(hits) == 1 and hits[0]["date"] == "2026-03-14"

    def test_every_recorded_field_reaches_the_text_index(self):
        from parallax.logbook.store import Experiment

        text = Experiment(
            id="x", title="t", outcome="success", date="2026-03-12",
            operator="RK", scale="12 g", analysis="NMR clean",
            conditions={"odd key": "odd value"}, links={"eln": "ELN-1"},
            tags=["scaleup"], observations="worked",
        ).searchable()
        for fragment in ("RK", "12 g", "NMR clean", "odd key", "odd value",
                         "eln", "ELN-1", "scaleup", "worked"):
            assert fragment in text, fragment

    def test_the_prompt_forbids_filling_a_gap_with_something_plausible(self):
        from parallax.agent.prompts import SYSTEM_PROMPT

        assert "Never fill a gap with something plausible" in SYSTEM_PROMPT
        assert "quoted from" in SYSTEM_PROMPT


class TestDateTokens:
    def test_the_iso_form_month_and_year_month_are_indexed(self):
        from parallax.logbook import date_tokens

        assert date_tokens("2026-03-12") == ["2026-03-12", "2026-03", "march"]

    def test_the_year_comes_free_from_the_tokeniser(self):
        from parallax.memory.store import _tokens

        assert "2026" in _tokens("2026-03-12")

    def test_a_missing_or_odd_date_does_not_raise(self):
        from parallax.logbook import date_tokens

        assert date_tokens(None) == [] and date_tokens("") == []
        assert date_tokens("unknown") == ["unknown"]


class TestFilterOnlyQueries:
    def test_an_outcome_alone_lists_the_failures(self, dated_book):
        """Previously scored to zero and returned nothing."""
        result = dated_book.search(outcome="failure")
        assert result["n_results"] == 1
        assert result["ordering"] == "date, newest first"

    def test_a_project_alone_lists_the_project(self, dated_book):
        assert dated_book.search(project="ALK series")["n_results"] == 3

    def test_a_ranked_query_still_ranks(self, dated_book):
        result = dated_book.search("protodeboronation")
        assert result["ordering"] == "relevance"
        assert result["results"][0]["outcome"] == "failure"


class TestIndexVersioning:
    def test_an_index_from_an_older_version_is_rebuilt(self, dated_book):
        import json

        dated_book.index()
        meta_path = dated_book.root / ".index" / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["version"] = 1  # as written before searchable() changed
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        reopened = LogbookStore(root=dated_book.root)
        assert reopened.search("suzuki run in march")["n_results"] == 2
        assert json.loads(meta_path.read_text(encoding="utf-8"))["version"] >= 2
