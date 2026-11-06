"""Live network tests. Run with: pytest -m network"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.network


def test_resolve_by_name_gives_the_right_structure():
    from parallax.chem import resolve

    structure = resolve("lorlatinib")
    assert structure.formula == "C21H19FN6O2"
    assert structure.inchikey.startswith("IIXWYSCJSQVBQM")
    assert any(c.startswith("pubchem:") for c in structure.citations)


def test_chembl_activities_carry_source_dois():
    from parallax.sources import chembl

    result = chembl.activities(target_chembl_id="CHEMBL4247", standard_type="IC50", limit=10)
    assert result["count"] > 0
    cited = [row for row in result["rows"] if row.get("citation")]
    assert cited, "every activity row should resolve to a source document"
    assert any(row["citation"].startswith("doi:") for row in cited)


def test_europepmc_search_returns_citations():
    from parallax.sources import europepmc

    result = europepmc.search("lorlatinib macrocyclic ALK inhibitor", limit=5)
    assert result["returned"] > 0
    assert all(record["citation"] for record in result["results"])


def test_patent_text_search_states_its_coverage_limit():
    from parallax.sources import patents

    result = patents.search("ALK inhibitor", limit=5)
    assert all(record["is_patent"] for record in result["results"])
    # The free patent index is partial, so an empty result must not read as
    # "nothing is patented".
    assert "coverage_note" in result
    assert "NOT shown" in result["coverage_note"] or "US grants only" in result["coverage_note"]


def test_structure_anchored_patent_lookup_is_the_reliable_route():
    from parallax.sources import patents

    result = patents.for_compound("crizotinib", limit=10)
    assert result["structure_linked_total"] > 10
    assert result["structure_linked_patents"]
    assert all(p["citation"].startswith("patent:") for p in result["structure_linked_patents"])


def test_doi_verification_distinguishes_real_from_fabricated():
    from parallax.provenance import verify_doi

    real = verify_doi("10.1021/jm500261q")
    assert real["exists"] is True
    assert "lorlatinib" in real["title"].lower() or "macrocyclic" in real["title"].lower()

    fake = verify_doi("10.9999/this.does.not.exist.12345")
    assert fake["exists"] is not True


def test_reaxys_reports_unavailable_rather_than_inventing():
    from parallax.sources import reaxys

    result = reaxys.reactions(query="macrolactamisation", limit=5)
    if not result.get("available"):
        assert result["results"] == []
        assert "use_instead" in result
        assert "fallback" in result


def test_measured_properties_carry_references():
    from parallax.agent.tools import call_tool

    result = call_tool("measured_properties", {"query": "aspirin"})
    assert result.get("n_values", 0) > 0
    assert any(value.get("citation") for value in result["values"])


@pytest.mark.slow
def test_model_training_beats_the_mean_baseline():
    from parallax.models import predict, train

    result = train("CHEMBL4247", endpoint="inhibition", split="scaffold", max_records=1200)
    card = result["card"]
    assert card["n_compounds"] >= 100
    assert card["metrics"]["rmse"] < card["baseline_metrics"]["rmse"]
    assert card["source_documents"] > 10

    prediction = predict(card["name"], "C[C@H]1Oc2cc(cnc2N)-c2c(nn(C)c2C#N)CN(C)C(=O)c2ccc(F)cc21")
    first = prediction["predictions"][0]
    assert 5 < first["predicted_value"] < 12
    assert first["applicability_domain"]["verdict"]
    assert first["uncertainty_log_units"] > 0
