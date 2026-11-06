"""Computed molecular properties, drug-likeness rules and structural alerts.

Three tiers of property information are kept strictly separate, because
conflating them is how cheminformatics reports go wrong:

1. **measured** — a value somebody determined experimentally, cited to the
   paper or handbook that reported it (see `sources.pubchem.experimental_properties`);
2. **computed** — a deterministic calculation from the structure, cited to the
   method paper (Crippen logP, Ertl TPSA, Ertl SA score, QED ...);
3. **modelled** — a prediction from a model trained on cited data
   (see `models.qsar`), always carrying an applicability-domain judgement.

This module owns tier 2.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Any

from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors
from rdkit.Chem import FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams

from ..provenance import cite_software
from .structure import parse, scaffold

RDLogger.DisableLog("rdApp.*")

_SA_PATH = os.path.join(RDConfig.RDContribDir, "SA_Score")
_NP_PATH = os.path.join(RDConfig.RDContribDir, "NP_Score")
for _path in (_SA_PATH, _NP_PATH):
    if _path not in sys.path:
        sys.path.append(_path)

import sascorer  # noqa: E402  (RDKit contrib, path-injected above)
import npscorer  # noqa: E402


@lru_cache(maxsize=1)
def _np_model() -> Any:
    # npscorer chatters on stdout when it loads its model, which would corrupt
    # JSON tool output; keep it quiet.
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return npscorer.readNPModel()


@lru_cache(maxsize=1)
def _alert_catalog() -> FilterCatalog.FilterCatalog:
    params = FilterCatalogParams()
    for catalog in (
        FilterCatalogParams.FilterCatalogs.PAINS_A,
        FilterCatalogParams.FilterCatalogs.PAINS_B,
        FilterCatalogParams.FilterCatalogs.PAINS_C,
        FilterCatalogParams.FilterCatalogs.BRENK,
        FilterCatalogParams.FilterCatalogs.NIH,
        FilterCatalogParams.FilterCatalogs.ZINC,
    ):
        params.AddCatalog(catalog)
    return FilterCatalog.FilterCatalog(params)


def physchem(smiles_or_mol: str | Chem.Mol) -> dict[str, Any]:
    """Core physicochemical descriptors, each from a published method."""
    mol = parse(smiles_or_mol) if isinstance(smiles_or_mol, str) else smiles_or_mol
    citations = cite_software(
        "software:rdkit", "method:crippen_logp", "method:tpsa", "method:qed", "method:sa_score"
    )
    ring_info = mol.GetRingInfo()
    macrocycle_sizes = [len(ring) for ring in ring_info.AtomRings() if len(ring) >= 12]

    values = {
        "mw": round(Descriptors.MolWt(mol), 2),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "clogp": round(Crippen.MolLogP(mol), 2),
        "mr": round(Crippen.MolMR(mol), 2),
        "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "rings": ring_info.NumRings(),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "aliphatic_rings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "heteroatoms": rdMolDescriptors.CalcNumHeteroatoms(mol),
        "fraction_csp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        "stereocentres": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)),
        "formal_charge": Chem.GetFormalCharge(mol),
        "qed": round(QED.qed(mol), 3),
        "sa_score": round(sascorer.calculateScore(mol), 2),
        "np_likeness": round(npscorer.scoreMol(mol, _np_model()), 2),
        "bertz_complexity": round(Descriptors.BertzCT(mol), 1),
        "num_amide_bonds": rdMolDescriptors.CalcNumAmideBonds(mol),
        "num_spiro": rdMolDescriptors.CalcNumSpiroAtoms(mol),
        "num_bridgehead": rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        "largest_ring": max((len(r) for r in ring_info.AtomRings()), default=0),
        "is_macrocycle": bool(macrocycle_sizes),
        "macrocycle_ring_sizes": macrocycle_sizes,
        "formula": rdMolDescriptors.CalcMolFormula(mol),
    }
    return {
        "smiles": Chem.MolToSmiles(mol),
        "scaffold": scaffold(mol),
        "values": values,
        "method_notes": {
            "clogp": "Wildman-Crippen atomic contribution logP (RDKit implementation)",
            "tpsa": "Ertl fragment-based topological polar surface area",
            "sa_score": "Ertl-Schuffenhauer synthetic accessibility, 1 (easy) to 10 (hard)",
            "np_likeness": "Ertl natural-product likeness, -5 (synthetic) to +5 (NP-like)",
            "qed": "Bickerton weighted quantitative estimate of drug-likeness, 0-1",
        },
        "citations": citations,
    }


# --- rule sets -----------------------------------------------------------


def druglikeness(smiles_or_mol: str | Chem.Mol) -> dict[str, Any]:
    """Published developability rule sets, evaluated explicitly."""
    mol = parse(smiles_or_mol) if isinstance(smiles_or_mol, str) else smiles_or_mol
    p = physchem(mol)["values"]
    citations = cite_software(
        "method:lipinski", "method:veber", "method:ghose", "method:macrocycle_review"
    )

    def rule(name: str, checks: list[tuple[str, bool]], reference: str) -> dict[str, Any]:
        failures = [label for label, ok in checks if not ok]
        return {
            "name": name,
            "passes": not failures,
            "violations": failures,
            "n_violations": len(failures),
            "reference": reference,
        }

    rules = [
        rule(
            "Lipinski Ro5",
            [
                ("MW <= 500", p["mw"] <= 500),
                ("cLogP <= 5", p["clogp"] <= 5),
                ("HBD <= 5", p["hbd"] <= 5),
                ("HBA <= 10", p["hba"] <= 10),
            ],
            "method:lipinski",
        ),
        rule(
            "Veber oral",
            [
                ("rotatable bonds <= 10", p["rotatable_bonds"] <= 10),
                ("TPSA <= 140", p["tpsa"] <= 140),
                ("HBD+HBA <= 12", (p["hbd"] + p["hba"]) <= 12),
            ],
            "method:veber",
        ),
        rule(
            "Ghose",
            [
                ("160 <= MW <= 480", 160 <= p["mw"] <= 480),
                ("-0.4 <= cLogP <= 5.6", -0.4 <= p["clogp"] <= 5.6),
                ("20 <= heavy atoms <= 70", 20 <= p["heavy_atoms"] <= 70),
                ("40 <= MR <= 130", 40 <= p["mr"] <= 130),
            ],
            "method:ghose",
        ),
        rule(
            "Egan permeability",
            [("TPSA <= 131.6", p["tpsa"] <= 131.6), ("cLogP <= 5.88", p["clogp"] <= 5.88)],
            "method:egan",
        ),
        rule(
            "Pfizer 3/75 (tox risk)",
            [("not (cLogP > 3 and TPSA < 75)", not (p["clogp"] > 3 and p["tpsa"] < 75))],
            "method:pfizer_375",
        ),
        rule(
            "GSK 4/400",
            [("cLogP <= 4", p["clogp"] <= 4), ("MW <= 400", p["mw"] <= 400)],
            "method:gsk_4400",
        ),
        rule(
            "Lead-likeness",
            [
                ("MW <= 350", p["mw"] <= 350),
                ("cLogP <= 3.5", p["clogp"] <= 3.5),
                ("rotatable bonds <= 7", p["rotatable_bonds"] <= 7),
            ],
            "method:lead_like",
        ),
    ]

    beyond_ro5 = p["mw"] > 500 or p["is_macrocycle"]
    return {
        "rules": rules,
        "summary": {
            "rules_passed": sum(1 for r in rules if r["passes"]),
            "rules_total": len(rules),
            "qed": p["qed"],
            "beyond_rule_of_5": beyond_ro5,
        },
        "interpretation": (
            "Macrocyclic / bRo5 chemotype: Ro5 and Veber cut-offs are weak predictors here. "
            "Judge permeability on measured data, chameleonicity and 3D polar-surface shielding "
            "rather than on 2D counts."
            if beyond_ro5
            else "Within classical small-molecule property space; rule violations are meaningful signals."
        ),
        "citations": citations,
    }


def cns_mpo(smiles_or_mol: str | Chem.Mol) -> dict[str, Any]:
    """CNS multiparameter optimisation desirability score (Wager et al.).

    Uses cLogP, cLogD (approximated), MW, TPSA, HBD and most-basic pKa. pKa is
    not computed here — when it is unknown the term is scored neutrally and the
    result says so, rather than silently assuming a value.
    """
    mol = parse(smiles_or_mol) if isinstance(smiles_or_mol, str) else smiles_or_mol
    p = physchem(mol)["values"]
    citations = cite_software("method:cns_mpo")

    def ramp(value: float, good: float, bad: float) -> float:
        if good == bad:
            return 1.0
        if good < bad:
            if value <= good:
                return 1.0
            if value >= bad:
                return 0.0
            return (bad - value) / (bad - good)
        if value >= good:
            return 1.0
        if value <= bad:
            return 0.0
        return (value - bad) / (good - bad)

    terms = {
        "clogp (<=3 good, >=5 bad)": ramp(p["clogp"], 3.0, 5.0),
        "clogd approx = clogp-1 (<=2 good, >=4 bad)": ramp(p["clogp"] - 1.0, 2.0, 4.0),
        "mw (<=360 good, >=500 bad)": ramp(p["mw"], 360.0, 500.0),
        "tpsa (40-90 window)": min(ramp(p["tpsa"], 40.0, 20.0), ramp(p["tpsa"], 90.0, 120.0)),
        "hbd (<=0.5 good, >=3.5 bad)": ramp(p["hbd"], 0.5, 3.5),
    }
    score = sum(terms.values()) * 6.0 / 5.0  # scale 5 scored terms onto the 0-6 scale
    return {
        "cns_mpo_score": round(score, 2),
        "max_score": 6.0,
        "terms": {k: round(v, 2) for k, v in terms.items()},
        "pka_term": "omitted — most-basic pKa not computed; score rescaled over 5 of 6 terms",
        "reading": (
            "≥4 is the desirability threshold Wager et al. associate with higher CNS exposure."
        ),
        "citations": citations,
    }


def alerts(smiles_or_mol: str | Chem.Mol) -> dict[str, Any]:
    """PAINS / Brenk / NIH / ZINC structural alerts with their catalogue of origin."""
    mol = parse(smiles_or_mol) if isinstance(smiles_or_mol, str) else smiles_or_mol
    citations = cite_software("method:pains", "method:brenk_filter")
    matches = []
    for entry in _alert_catalog().GetMatches(mol):
        try:
            catalog = entry.GetProp("FilterSet")
        except Exception:
            catalog = "unknown"
        matches.append({"catalog": catalog, "alert": entry.GetDescription()})
    by_catalog: dict[str, list[str]] = {}
    for match in matches:
        by_catalog.setdefault(match["catalog"], []).append(match["alert"])
    return {
        "n_alerts": len(matches),
        "by_catalog": by_catalog,
        "interpretation": (
            "No catalogued liability substructures."
            if not matches
            else "Alerts flag substructures associated with assay interference or tox liabilities. "
            "They are heuristics: approved drugs do trip them. Treat as a prompt to check the "
            "specific assay readout, not as a veto."
        ),
        "citations": citations,
    }


def liability_flags(smiles_or_mol: str | Chem.Mol) -> dict[str, Any]:
    """Pharmacophore-style flags for common attrition risks.

    Each flag states the structural basis and what to measure next. These are
    screening heuristics drawn from published SAR, not predictions, and the
    output says so.
    """
    mol = parse(smiles_or_mol) if isinstance(smiles_or_mol, str) else smiles_or_mol
    p = physchem(mol)["values"]
    smarts_flags: list[tuple[str, str, str, str]] = [
        (
            "hERG risk pharmacophore",
            "[NX3;H0,H1;!$(N[C,S]=[O,S,N])]",
            "basic amine present; combined with high lipophilicity this is the classic hERG motif",
            "measure hERG patch-clamp; consider reducing basicity or logP",
        ),
        (
            "reactive Michael acceptor",
            "[CX3]=[CX3][CX3]=[OX1]",
            "enone / acrylamide-type electrophile",
            "check covalent intent; run GSH trapping and reactive-metabolite screen",
        ),
        (
            "aniline (metabolic activation)",
            "[NX3;H2,H1][c]",
            "aromatic amine — possible nitroso/quinone-imine bioactivation",
            "run AMES and reactive-metabolite assays; consider blocking para position",
        ),
        (
            "nitroaromatic",
            "[$([NX3](=O)=O),$([NX3+](=O)[O-])][c]",
            "nitroaromatic groups are frequently mutagenic via nitroreduction",
            "AMES test; consider replacing with cyano/sulfonyl",
        ),
        (
            "thiophene / furan (CYP bioactivation)",
            "[c]1[c][c][s,o][c]1",
            "electron-rich 5-membered heteroaryl prone to epoxidation / S-oxidation",
            "reactive-metabolite screen; consider ring substitution",
        ),
        (
            "ester (plasma instability)",
            "[CX3](=O)O[CX4]",
            "alkyl ester — likely esterase hydrolysis in plasma",
            "measure plasma stability; consider amide or hindered ester if prodrug not intended",
        ),
        (
            "sulfonamide (CA inhibition / solubility)",
            "[SX4](=O)(=O)[NX3]",
            "primary/secondary sulfonamide",
            "check carbonic-anhydrase selectivity if primary",
        ),
    ]

    flags = []
    for name, smarts, basis, action in smarts_flags:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None and mol.HasSubstructMatch(pattern):
            flags.append({"flag": name, "structural_basis": basis, "next_experiment": action})

    if p["clogp"] > 3 and any(f["flag"].startswith("hERG") for f in flags):
        flags.append(
            {
                "flag": "hERG risk elevated by lipophilicity",
                "structural_basis": f"basic centre with cLogP {p['clogp']}",
                "next_experiment": "prioritise hERG counter-screen before potency optimisation",
            }
        )
    if p["clogp"] > 5:
        flags.append(
            {
                "flag": "high lipophilicity",
                "structural_basis": f"cLogP {p['clogp']} > 5",
                "next_experiment": "expect solubility and promiscuity issues; measure kinetic solubility and CYP inhibition",
            }
        )
    if p["tpsa"] < 60 and p["mw"] > 450:
        flags.append(
            {
                "flag": "large and apolar",
                "structural_basis": f"MW {p['mw']} with TPSA {p['tpsa']}",
                "next_experiment": "measure solubility and permeability directly; 2D rules are unreliable here",
            }
        )

    return {
        "flags": flags,
        "n_flags": len(flags),
        "disclaimer": (
            "Structure-based heuristics from published medicinal-chemistry practice. "
            "They indicate what to measure, not what the measurement will be."
        ),
        "citations": cite_software("method:brenk_filter", "method:pains"),
    }


def fingerprint(smiles_or_mol: str | Chem.Mol, *, radius: int = 2, bits: int = 2048):
    """Morgan/ECFP fingerprint (the workhorse for similarity and QSAR)."""
    from rdkit.Chem import rdFingerprintGenerator

    mol = parse(smiles_or_mol) if isinstance(smiles_or_mol, str) else smiles_or_mol
    cite_software("method:morgan_fp")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    return generator.GetFingerprint(mol)


def similarity(smiles_a: str, smiles_b: str) -> float:
    from rdkit import DataStructs

    return round(
        DataStructs.TanimotoSimilarity(fingerprint(smiles_a), fingerprint(smiles_b)), 3
    )


def profile(smiles_or_mol: str | Chem.Mol) -> dict[str, Any]:
    """Everything tier-2 about one structure, in one call."""
    mol = parse(smiles_or_mol) if isinstance(smiles_or_mol, str) else smiles_or_mol
    core = physchem(mol)
    rules = druglikeness(mol)
    alert_result = alerts(mol)
    liabilities = liability_flags(mol)
    cns = cns_mpo(mol)
    citations = sorted(
        set(core["citations"] + rules["citations"] + alert_result["citations"] + liabilities["citations"] + cns["citations"])
    )
    return {
        "smiles": core["smiles"],
        "scaffold": core["scaffold"],
        "computed_properties": core["values"],
        "method_notes": core["method_notes"],
        "druglikeness": rules,
        "cns_mpo": cns,
        "structural_alerts": alert_result,
        "liability_flags": liabilities,
        "tier": "computed (deterministic from structure; not measured, not modelled)",
        "citations": citations,
    }
