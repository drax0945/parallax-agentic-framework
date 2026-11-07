"""Multi-parameter optimisation, covalent detection and endpoint generality."""

from __future__ import annotations

import pytest

from parallax.chem.optimisation import (
    PROFILES,
    Objective,
    covalent_profile,
    list_profiles,
    score_candidates,
)
from parallax.models.qsar import CONCENTRATION_ENDPOINTS, ENDPOINT_PRESETS, normalise_value

LORLATINIB = "C[C@H]1Oc2cc(cnc2N)-c2c(nn(C)c2C#N)CN(C)C(=O)c2ccc(F)cc21"
CRIZOTINIB = "C[C@@H](Oc1cc(-c2cnn(C3CCNCC3)c2)cnc1N)c1c(Cl)ccc(F)c1Cl"
GREASY = "CCCCCCCCCCCCCCCCc1ccc(cc1)C(=O)Nc1ccc(cc1)C(=O)O"


class TestEndpointGenerality:
    def test_non_kinase_endpoints_have_presets(self):
        for endpoint in ("antimicrobial", "cytotoxicity", "binding_affinity_relative", "admet"):
            assert endpoint in ENDPOINT_PRESETS

    def test_pchembl_is_used_when_present(self):
        value, how = normalise_value({"pchembl_value": 8.1}, endpoint_types=["IC50"])
        assert value == 8.1
        assert how == "pchembl_value"

    def test_micrograms_per_ml_converts_to_pmic(self):
        # 2 ug/mL at MW 450 -> 4.44 uM -> pMIC 5.35
        value, how = normalise_value(
            {"standard_type": "MIC", "standard_value": 2.0, "standard_units": "ug.mL-1"},
            endpoint_types=["MIC"],
            mol_weight=450.0,
        )
        assert value == pytest.approx(5.35, abs=0.02)
        assert "MW" in how

    def test_molar_units_convert(self):
        value, _ = normalise_value(
            {"standard_type": "IC50", "standard_value": 500, "standard_units": "nM"},
            endpoint_types=["IC50"],
        )
        assert value == pytest.approx(6.3, abs=0.01)

    def test_non_concentration_endpoints_keep_their_scale(self):
        value, how = normalise_value(
            {"standard_type": "RBA", "standard_value": 37, "standard_units": "%"},
            endpoint_types=["RBA"],
        )
        assert value == 37.0
        assert "native scale" in how

    def test_unusable_values_are_rejected_not_guessed(self):
        for row in (
            {"standard_type": "IC50", "standard_value": 5, "standard_units": "weird"},
            {"standard_type": "IC50", "standard_value": -1, "standard_units": "nM"},
            {"standard_type": "IC50", "standard_value": None, "standard_units": "nM"},
        ):
            value, _ = normalise_value(row, endpoint_types=["IC50"])
            assert value is None

    def test_mic_is_treated_as_a_concentration(self):
        assert {"MIC", "MIC50", "MIC90"} <= CONCENTRATION_ENDPOINTS


class TestMultiParameterScoring:
    def test_cns_profile_ranks_a_real_cns_drug_above_a_known_non_penetrant(self):
        """Lorlatinib was designed for brain penetration where crizotinib failed."""
        result = score_candidates(
            [
                {"smiles": LORLATINIB, "potency": 8.3, "label": "lorlatinib"},
                {"smiles": CRIZOTINIB, "potency": 7.7, "label": "crizotinib"},
            ],
            profile="cns",
        )
        ranked = [r["label"] for r in result["ranked"]]
        assert ranked[0] == "lorlatinib"

    def test_potency_alone_does_not_win(self):
        result = score_candidates(
            [
                {"smiles": LORLATINIB, "potency": 8.3, "label": "balanced"},
                {"smiles": GREASY, "potency": 9.0, "label": "greasy"},
            ],
            profile="cns",
        )
        assert result["ranked"][0]["label"] == "balanced"

    def test_every_candidate_names_its_limiting_objective(self):
        result = score_candidates([{"smiles": CRIZOTINIB, "potency": 7.0}], profile="oral_small_molecule")
        row = result["ranked"][0]
        assert row["limiting_objective"]
        assert 0.0 <= row["limiting_desirability"] <= 1.0

    def test_missing_potency_drops_the_term_rather_than_scoring_zero(self):
        result = score_candidates([{"smiles": LORLATINIB}], profile="oral_small_molecule")
        row = result["ranked"][0]
        assert "potency" in row["missing_inputs"]
        assert row["mpo_score"] > 0

    def test_scores_are_bounded_and_ordered(self):
        result = score_candidates(
            [{"smiles": s, "potency": 7.0} for s in (LORLATINIB, CRIZOTINIB, GREASY)],
            profile="oral_small_molecule",
        )
        scores = [r["mpo_score"] for r in result["ranked"]]
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_antimicrobial_profile_inverts_the_lipophilicity_preference(self):
        oral = next(o for o in PROFILES["oral_small_molecule"]["objectives"] if o.name == "clogp")
        anti = next(o for o in PROFILES["antimicrobial"]["objectives"] if o.name == "clogp")
        assert anti.good < oral.good  # Gram-negative entry favours lower logP

    def test_custom_objectives_are_honoured(self):
        result = score_candidates(
            [{"smiles": LORLATINIB}],
            custom_objectives=[Objective("mw", "minimise", good=100.0, bad=200.0)],
        )
        assert result["profile"] == "custom"
        assert result["ranked"][0]["mpo_score"] < 0.1  # MW 406 fails that objective

    def test_unknown_profile_rejected(self):
        with pytest.raises(ValueError):
            score_candidates([{"smiles": LORLATINIB}], profile="nonsense")

    def test_profiles_are_listable(self):
        assert set(list_profiles()["profiles"]) == set(PROFILES)


class TestCovalentDetection:
    @pytest.mark.parametrize(
        "smiles,warhead",
        [
            ("C=CC(=O)N1CCCC1c1ncnc2[nH]ccc12", "acrylamide"),
            ("ClCC(=O)Nc1ccccc1", "chloroacetamide"),
            ("CC(C)CC(NC(=O)C1CC1)C#N", "nitrile (reversible covalent)"),
            ("CC(C)CC(NC(=O)c1ccccc1)B(O)O", "boronic acid"),
        ],
    )
    def test_known_warheads_are_detected(self, smiles, warhead):
        result = covalent_profile(smiles)
        assert result["is_putative_covalent"] is True
        assert warhead in [w["warhead"] for w in result["warheads"]]

    def test_non_covalent_drug_is_not_flagged(self):
        assert covalent_profile(LORLATINIB)["is_putative_covalent"] is False

    def test_covalent_hits_explain_the_modelling_consequence(self):
        result = covalent_profile("C=CC(=O)N1CCCC1c1ncnc2[nH]ccc12")
        joined = " ".join(result["implications"]).lower()
        assert "time-dependent" in joined
        assert "kinact" in joined


class TestAgentRuntimeDiagnosis:
    """An auth failure arrives as ordinary assistant text, so it must be caught."""

    def test_not_logged_in_is_diagnosed(self):
        from parallax.agent.loop import diagnose_auth_failure

        problem = diagnose_auth_failure("Not logged in · Please run /login", 0)
        assert problem is not None
        assert "parallax login" in problem["fix"]

    def test_invalid_api_key_is_diagnosed(self):
        from parallax.agent.loop import diagnose_auth_failure

        problem = diagnose_auth_failure("Invalid API key · Fix external API key", 0)
        assert problem is not None
        assert "ANTHROPIC_API_KEY" in problem["reason"]

    def test_a_real_answer_is_not_mistaken_for_a_failure(self):
        from parallax.agent.loop import diagnose_auth_failure

        assert diagnose_auth_failure("Lorlatinib has an ALK pChEMBL of 9.7.", 3) is None
        # even with no tool calls, a substantive answer is not an auth error
        assert diagnose_auth_failure("x" * 500, 0) is None

    def test_the_runtime_binary_is_resolvable(self):
        """`parallax ask` needs a claude binary; the SDK bundles one."""
        import shutil

        from parallax.agent.loop import bundled_cli

        assert shutil.which("claude") or bundled_cli(), (
            "no claude binary on PATH and none bundled with claude-agent-sdk"
        )

    def test_login_command_is_registered(self):
        from parallax.cli import app

        names = {c.name or c.callback.__name__ for c in app.registered_commands}
        assert "login" in names
