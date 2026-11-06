"""Deterministic chemistry tests — no network required."""

from __future__ import annotations

import pytest

from parallax.chem.analogs import enumerate_analogs, matched_pairs, suggest_macrocyclization
from parallax.chem.descriptors import alerts, druglikeness, physchem, profile, similarity
from parallax.chem.reactions import (
    TEMPLATES,
    compiled_templates,
    procedural_templates,
    validate_reaction_smiles,
)
from parallax.chem.retro import disconnect, feasibility
from parallax.chem.structure import canonical_smiles, classify_input, parse, scaffold, svg

LORLATINIB = "C[C@H]1Oc2cc(cnc2N)-c2c(nn(C)c2C#N)CN(C)C(=O)c2ccc(F)cc21"
CRIZOTINIB = "C[C@@H](Oc1cc(-c2cnn(C3CCNCC3)c2)cnc1N)c1c(Cl)ccc(F)c1Cl"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


class TestStructure:
    def test_parse_and_canonicalise(self):
        assert canonical_smiles("C(C)O") == canonical_smiles("CCO")

    def test_bad_smiles_raises(self):
        with pytest.raises(ValueError):
            parse("this is not a molecule")

    def test_salt_is_stripped_by_default(self):
        # Sodium acetate -> acetate (largest fragment)
        assert "Na" not in canonical_smiles("CC(=O)[O-].[Na+]")

    def test_scaffold(self):
        core = scaffold(ASPIRIN)
        assert "c1ccccc1" in core.replace("C1=CC=CC=C1", "c1ccccc1")

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3", "inchi"),
            ("BSYNRYMUTXBXSQ-UHFFFAOYSA-N", "inchikey"),
            ("CHEMBL601719", "chembl_id"),
            ("50-78-2", "cas"),
            ("lorlatinib", "name"),
            (ASPIRIN, "smiles"),
        ],
    )
    def test_input_classification(self, text, expected):
        assert classify_input(text) == expected

    def test_svg_renders(self):
        drawing = svg(ASPIRIN)
        assert drawing.strip().startswith("<?xml") or "<svg" in drawing
        assert "</svg>" in drawing


class TestDescriptors:
    def test_lorlatinib_is_a_12_membered_macrocycle(self):
        values = physchem(LORLATINIB)["values"]
        assert values["is_macrocycle"] is True
        assert 12 in values["macrocycle_ring_sizes"]

    def test_macrocyclisation_removes_rotatable_bonds(self):
        assert physchem(LORLATINIB)["values"]["rotatable_bonds"] == 0
        assert physchem(CRIZOTINIB)["values"]["rotatable_bonds"] > 0

    def test_known_property_values(self):
        values = physchem(ASPIRIN)["values"]
        assert 180.0 < values["mw"] < 180.3
        assert values["formula"] == "C9H8O4"
        assert 1.0 < values["clogp"] < 1.5

    def test_profile_separates_tiers_and_cites(self):
        result = profile(ASPIRIN)
        assert result["tier"].startswith("computed")
        assert "method:crippen_logp" in result["citations"]
        assert "software:rdkit" in result["citations"]

    def test_bro5_interpretation_for_macrocycle(self):
        rules = druglikeness(LORLATINIB)
        assert rules["summary"]["beyond_rule_of_5"] is True
        assert "weak predictors" in rules["interpretation"]

    def test_alert_catalog_runs(self):
        result = alerts("O=C(Cc1ccccc1)Nc1ccc(N(=O)=O)cc1")
        assert isinstance(result["n_alerts"], int)
        assert "interpretation" in result

    def test_similarity_bounds(self):
        assert similarity(LORLATINIB, LORLATINIB) == 1.0
        assert 0.0 <= similarity(LORLATINIB, ASPIRIN) < 0.3


class TestReactions:
    def test_every_smarts_template_compiles(self):
        failures = [t.name for t in TEMPLATES if not t.procedural and t.reaction() is None]
        assert failures == []

    def test_library_is_split_between_smarts_and_procedural(self):
        assert len(compiled_templates()) >= 20
        assert {t.procedural for t in procedural_templates()} == {"amide", "ester", "ether", "alkene"}

    def test_templates_carry_conditions_and_precedent_queries(self):
        for template in TEMPLATES:
            assert template.conditions, f"{template.name} has no conditions"
            assert template.precedent_query, f"{template.name} has no precedent query"
            assert 0 < template.reliability <= 1

    def test_atom_balance_detection(self):
        balanced = validate_reaction_smiles("CCO.CC(=O)O>>CCOC(C)=O.O")
        assert balanced["valid"] is True
        assert balanced["atom_balanced"] is True

        unbalanced = validate_reaction_smiles("CCO>>CCOCC")
        assert unbalanced["valid"] is True
        assert unbalanced["atom_balanced"] is False
        assert unbalanced["imbalance_product_minus_reactant"]

    def test_unparsable_reaction_is_reported_not_raised(self):
        assert validate_reaction_smiles("not>>a>>reaction")["valid"] is False


class TestRetro:
    def test_macrolactam_disconnection_found_for_lorlatinib(self):
        options = disconnect(LORLATINIB)
        names = [o.template for o in options]
        assert "Macrolactamization" in names
        macro = next(o for o in options if o.template == "Macrolactamization")
        assert macro.ring_size == 12
        assert len(macro.precursors) == 1  # ring opening gives ONE linear precursor
        parse(macro.precursors[0])  # and it must be a valid molecule

    def test_ring_opening_products_are_valid_molecules(self):
        for option in disconnect(LORLATINIB):
            for precursor in option.precursors:
                parse(precursor)

    def test_disconnections_are_ranked(self):
        scores = [o.score for o in disconnect(CRIZOTINIB)]
        assert scores == sorted(scores, reverse=True)

    def test_ring_opening_outranks_steps_that_leave_the_macrocycle_intact(self):
        # The strategically decisive problem for a macrocycle is the closure, so a
        # disconnection that leaves the ring untouched must not outrank one that opens it.
        options = disconnect(LORLATINIB)
        best_opening = next(o for o in options if o.macrocyclization)
        retained = [o for o in options if any("macrocycle is retained" in c for c in o.concerns)]
        assert retained, "steps leaving the ring intact should be flagged"
        assert best_opening.score > max(o.score for o in retained)

    def test_acyclic_targets_are_not_penalised(self):
        for option in disconnect(ASPIRIN):
            assert not any("macrocycle is retained" in c for c in option.concerns)

    def test_macrocyclisation_concerns_mention_dilution(self):
        macro = next(o for o in disconnect(LORLATINIB) if o.macrocyclization)
        assert any("dilution" in c for c in macro.concerns)

    def test_feasibility_names_drivers(self):
        result = feasibility(LORLATINIB)
        factors = {d["factor"] for d in result["drivers"]}
        assert "synthetic accessibility" in factors
        assert "macrocyclisation" in factors
        assert "no single number" in result["verdict_scale"]


class TestDesign:
    def test_analogs_carry_rationale_and_deltas(self):
        result = enumerate_analogs(CRIZOTINIB, max_total=10)
        assert result["n_analogs"] > 0
        for analog in result["analogs"]:
            assert analog["rationale"]
            assert analog["expected_effect"]
            assert "clogp" in analog["property_deltas"]
            parse(analog["smiles"])

    def test_macrocyclisation_designs_are_real_macrocycles(self):
        result = suggest_macrocyclization(CRIZOTINIB, max_designs=6)
        assert result["n_designs"] > 0
        for design in result["designs"]:
            values = physchem(design["smiles"])["values"]
            assert values["is_macrocycle"] is True
            assert 11 <= design["ring_size"] <= 16
            assert design["rotatable_bond_change"] <= 0

    def test_mmpa_finds_a_known_transformation(self):
        records = [
            {"smiles": "c1ccc(C(=O)Nc2ccccc2)cc1", "y": 6.0, "citations": ["doi:10.1/a"]},
            {"smiles": "c1ccc(C(=O)Nc2ccc(F)cc2)cc1", "y": 7.0, "citations": ["doi:10.1/b"]},
            {"smiles": "c1ccc(C(=O)Nc2ccc(Cl)cc2)cc1", "y": 7.2, "citations": ["doi:10.1/c"]},
        ]
        result = matched_pairs(records, min_pairs_per_transform=1)
        assert result["n_pairs_found"] > 0
        assert result["n_transformations"] > 0
        assert all("delta" in example for t in result["transformations"] for example in t["examples"])
