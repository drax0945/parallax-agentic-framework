"""Multi-parameter optimisation and modality-specific profiling.

Drug discovery is multi-objective. A compound that gains a log of potency while
adding two logs of lipophilicity is usually a step backwards, and ranking on any
single number hides that. This module scores a set of candidates against an
explicit, weighted objective profile and shows which objective each candidate
failed — so a decision can be argued with rather than just accepted.

Desirability curves follow the standard Derringer-Suich formulation used across
process and molecular optimisation: each property maps onto 0-1, and the overall
score is a weighted geometric mean, so a zero on any objective sinks the
candidate rather than being averaged away.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

from rdkit import Chem, RDLogger

from ..provenance import Citation, cite_software, registry
from .descriptors import alerts, physchem
from .structure import parse

RDLogger.DisableLog("rdApp.*")


@dataclass
class Objective:
    """One optimisation criterion and how desirable each value is."""

    name: str
    kind: str  # maximise | minimise | window
    good: float
    bad: float
    weight: float = 1.0
    upper_good: float | None = None  # for window objectives
    upper_bad: float | None = None
    note: str = ""

    def desirability(self, value: float | None) -> float | None:
        if value is None:
            return None
        if self.kind == "window":
            rising = _ramp(value, self.good, self.bad)
            falling = _ramp(value, self.upper_good or self.good, self.upper_bad or self.bad)
            return min(rising, falling)
        if self.kind == "maximise":
            return _ramp(value, self.good, self.bad)
        return _ramp(value, self.good, self.bad)


def _ramp(value: float, good: float, bad: float) -> float:
    """Linear desirability between a good and a bad anchor, clamped to 0-1."""
    if good == bad:
        return 1.0
    if good < bad:  # lower is better
        if value <= good:
            return 1.0
        if value >= bad:
            return 0.0
        return (bad - value) / (bad - good)
    if value >= good:  # higher is better
        return 1.0
    if value <= bad:
        return 0.0
    return (value - bad) / (good - bad)


# --- profiles -------------------------------------------------------------

PROFILES: dict[str, dict[str, Any]] = {
    "oral_small_molecule": {
        "description": "Conventional oral drug: potent, soluble enough, not too greasy or large.",
        "objectives": [
            Objective("potency", "maximise", good=8.0, bad=5.0, weight=2.0,
                      note="pIC50/pKi/pEC50; weighted double because it is the primary endpoint"),
            Objective("clogp", "window", good=1.0, bad=-1.0, upper_good=3.5, upper_bad=5.5,
                      note="lipophilicity sweet spot; high cLogP drives promiscuity and clearance"),
            Objective("mw", "minimise", good=400.0, bad=550.0,
                      note="smaller leaves room to grow during optimisation"),
            Objective("tpsa", "window", good=40.0, bad=20.0, upper_good=120.0, upper_bad=150.0,
                      note="too low hurts solubility, too high hurts permeability"),
            Objective("hbd", "minimise", good=2.0, bad=5.0, note="HBD count drives permeability"),
            Objective("sa_score", "minimise", good=3.5, bad=6.5, weight=0.8,
                      note="synthetic accessibility — a design nobody can make is not a design"),
            Objective("n_alerts", "minimise", good=0.0, bad=3.0, weight=0.8,
                      note="PAINS/Brenk structural liabilities"),
        ],
    },
    "cns": {
        "description": "CNS-penetrant: smaller, less polar, fewer HBDs, lower basicity.",
        "objectives": [
            Objective("potency", "maximise", good=8.0, bad=5.5, weight=2.0),
            Objective("clogp", "window", good=1.5, bad=-0.5, upper_good=3.0, upper_bad=5.0),
            Objective("mw", "minimise", good=360.0, bad=500.0, weight=1.2,
                      note="Wager CNS MPO anchors: desirable <=360, zero by 500"),
            Objective("tpsa", "window", good=40.0, bad=20.0, upper_good=90.0, upper_bad=120.0,
                      weight=1.5,
                      note="Wager CNS MPO window; lorlatinib sits at 110 and is brain-penetrant, "
                           "so this tapers rather than cutting off at 90"),
            Objective("hbd", "minimise", good=1.0, bad=4.0, weight=1.5),
            Objective("sa_score", "minimise", good=3.5, bad=6.5, weight=0.6),
            Objective("n_alerts", "minimise", good=0.0, bad=3.0, weight=0.8),
        ],
    },
    "lead_like": {
        "description": "Early lead: deliberately small and simple, leaving room to optimise.",
        "objectives": [
            Objective("potency", "maximise", good=7.0, bad=4.5, weight=1.5),
            Objective("mw", "minimise", good=350.0, bad=450.0, weight=1.5),
            Objective("clogp", "window", good=1.0, bad=-1.0, upper_good=3.0, upper_bad=4.5),
            Objective("fraction_csp3", "maximise", good=0.4, bad=0.1, weight=0.8,
                      note="three-dimensionality correlates with developability"),
            Objective("sa_score", "minimise", good=3.0, bad=5.5),
            Objective("n_alerts", "minimise", good=0.0, bad=2.0),
        ],
    },
    "antimicrobial": {
        "description": "Antibacterial: different property space — more polar, often zwitterionic.",
        "objectives": [
            Objective("potency", "maximise", good=6.0, bad=4.0, weight=2.0,
                      note="pMIC; whole-cell activity already includes permeability and efflux"),
            Objective("clogp", "window", good=-1.0, bad=-3.0, upper_good=2.0, upper_bad=4.0,
                      note="Gram-negative entry favours low logP, unlike classical oral rules"),
            Objective("tpsa", "maximise", good=120.0, bad=60.0, weight=0.8,
                      note="polarity aids porin uptake in Gram-negatives"),
            Objective("mw", "minimise", good=500.0, bad=700.0, weight=0.8),
            Objective("sa_score", "minimise", good=4.0, bad=6.5, weight=0.6),
        ],
    },
    "fragment": {
        "description": "Fragment screening hit: rule of three.",
        "objectives": [
            Objective("potency", "maximise", good=5.0, bad=3.0, weight=1.0),
            Objective("mw", "minimise", good=250.0, bad=350.0, weight=1.5),
            Objective("clogp", "window", good=0.5, bad=-1.5, upper_good=3.0, upper_bad=4.0),
            Objective("hbd", "minimise", good=2.0, bad=4.0),
            Objective("sa_score", "minimise", good=3.0, bad=5.0),
        ],
    },
}

PROFILE_CITATIONS = {
    "oral_small_molecule": ("method:lipinski", "method:veber", "method:gsk_4400"),
    "cns": ("method:cns_mpo",),
    "lead_like": ("method:lead_like",),
    "antimicrobial": (),
    "fragment": (),
}


def _derringer_citation() -> str:
    return registry().add(
        Citation(
            key="doi:10.1080/00224065.1980.11980968",
            kind="method",
            title="Simultaneous optimization of several response variables",
            source_db="parallax/mpo",
            authors=["Derringer, George", "Suich, Ronald"],
            year=1980,
            venue="Journal of Quality Technology",
            doi="10.1080/00224065.1980.11980968",
            note="desirability-function formulation used for multi-parameter scoring",
        )
    )


def score_candidates(
    candidates: Sequence[dict[str, Any]],
    *,
    profile: str = "oral_small_molecule",
    custom_objectives: Sequence[Objective] | None = None,
    potency_key: str = "potency",
) -> dict[str, Any]:
    """Rank candidates on a weighted desirability profile.

    `candidates` are dicts with at least `smiles`, optionally a measured or
    predicted `potency` (as a pX value). Properties are computed where absent.
    """
    if custom_objectives:
        objectives = list(custom_objectives)
        profile_description = "custom objective set"
        profile_name = "custom"
    else:
        spec = PROFILES.get(profile)
        if spec is None:
            raise ValueError(f"unknown profile {profile!r}; options: {sorted(PROFILES)}")
        objectives = list(spec["objectives"])
        profile_description = spec["description"]
        profile_name = profile

    citations = [_derringer_citation(), *cite_software("software:rdkit")]
    citations += cite_software(*PROFILE_CITATIONS.get(profile_name, ()))

    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        smiles = candidate.get("smiles")
        if not smiles:
            continue
        try:
            mol = parse(smiles)
        except Exception as exc:
            scored.append({"smiles": smiles, "error": str(exc)})
            continue

        values = physchem(mol)["values"]
        available: dict[str, float | None] = dict(values)
        available["n_alerts"] = float(alerts(mol)["n_alerts"])
        available["potency"] = candidate.get(potency_key)

        terms: list[dict[str, Any]] = []
        product = 1.0
        total_weight = 0.0
        missing: list[str] = []
        worst: tuple[float, str] | None = None

        for objective in objectives:
            value = available.get(objective.name)
            desirability = objective.desirability(value)
            if desirability is None:
                missing.append(objective.name)
                continue
            terms.append(
                {
                    "objective": objective.name,
                    "value": round(float(value), 3),
                    "desirability": round(desirability, 3),
                    "weight": objective.weight,
                    "note": objective.note,
                }
            )
            # Geometric mean: a zero anywhere sinks the candidate, which is the
            # behaviour you want — you cannot average your way out of a hard fail.
            product *= max(desirability, 1e-6) ** objective.weight
            total_weight += objective.weight
            if worst is None or desirability < worst[0]:
                worst = (desirability, objective.name)

        score = product ** (1 / total_weight) if total_weight else 0.0
        scored.append(
            {
                "smiles": smiles,
                "mpo_score": round(score, 3),
                "limiting_objective": worst[1] if worst else None,
                "limiting_desirability": round(worst[0], 3) if worst else None,
                "terms": terms,
                "missing_inputs": missing,
                "properties": {
                    k: values[k] for k in ("mw", "clogp", "tpsa", "hbd", "sa_score", "qed")
                },
                **{k: v for k, v in candidate.items() if k not in {"smiles"}},
            }
        )

    scored.sort(key=lambda row: row.get("mpo_score", -1), reverse=True)

    return {
        "profile": profile_name,
        "profile_description": profile_description,
        "objectives": [asdict(o) for o in objectives],
        "n_candidates": len(scored),
        "ranked": scored,
        "how_to_read": (
            "mpo_score is a weighted geometric mean of per-objective desirabilities (0-1). "
            "`limiting_objective` names the single property holding each candidate back — "
            "that is the one to fix. A missing potency value simply drops that term rather "
            "than scoring zero, so compare only candidates with the same inputs available."
        ),
        "citations": sorted(set(citations)),
    }


def list_profiles() -> dict[str, Any]:
    return {
        "profiles": {
            name: {
                "description": spec["description"],
                "objectives": [
                    {
                        "name": o.name,
                        "kind": o.kind,
                        "good": o.good,
                        "bad": o.bad,
                        "weight": o.weight,
                    }
                    for o in spec["objectives"]
                ],
            }
            for name, spec in PROFILES.items()
        },
        "note": "Pass custom objectives when none of these match the project's criteria.",
    }


# --- modality detection ---------------------------------------------------

COVALENT_WARHEADS: list[tuple[str, str, str, str]] = [
    ("acrylamide", "[NX3][CX3](=[OX1])[CX3]=[CX3]", "irreversible",
     "the dominant warhead for kinase cysteines (ibrutinib, osimertinib class)"),
    ("chloroacetamide", "[NX3][CX3](=[OX1])[CH2][Cl]", "irreversible",
     "more reactive than acrylamide; higher off-target risk"),
    ("vinyl sulfone", "[SX4](=[OX1])(=[OX1])[CX3]=[CX3;!$(C=O)]", "irreversible",
     "cysteine protease warhead"),
    ("epoxide", "[CX4]1[OX2][CX4]1", "irreversible",
     "reactive and often non-selective"),
    ("beta-lactam", "[NX3]1[CX3](=[OX1])[CX4][CX4]1", "irreversible",
     "serine acylation — the penicillin mechanism"),
    ("nitrile (reversible covalent)", "[CX4][CX2]#[NX1]", "reversible",
     "reversible cysteine/serine adduct; nirmatrelvir class"),
    ("aldehyde (reversible covalent)", "[CX3H1](=[OX1])[#6]", "reversible",
     "forms hemiacetal/thiohemiacetal; often metabolically unstable"),
    ("boronic acid", "[BX3]([OX2H1])[OX2H1]", "reversible",
     "serine/threonine protease warhead; bortezomib class"),
    ("fluorosulfate / sulfonyl fluoride", "[SX4](=[OX1])(=[OX1])[F]", "irreversible",
     "SuFEx chemistry; tyrosine/lysine labelling"),
    ("cyanoacrylamide", "[NX3][CX3](=[OX1])[CX3](=[CX3])[CX2]#[NX1]", "reversible",
     "tunable reversible-covalent acrylamide"),
]


def covalent_profile(smiles: str) -> dict[str, Any]:
    """Detect covalent warheads and say what follows for assay design.

    Covalent inhibitors are a mainstream modality, and they break the usual
    assumptions: potency is time-dependent, so a single IC50 is close to
    meaningless without an incubation time, and kinact/KI is the real measure.
    """
    mol = parse(smiles)
    cite_software("software:rdkit")
    found = []
    for name, smarts, reversibility, note in COVALENT_WARHEADS:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None and mol.HasSubstructMatch(pattern):
            found.append(
                {
                    "warhead": name,
                    "reversibility": reversibility,
                    "note": note,
                    "n_occurrences": len(mol.GetSubstructMatches(pattern)),
                }
            )

    return {
        "smiles": Chem.MolToSmiles(mol),
        "is_putative_covalent": bool(found),
        "warheads": found,
        "implications": (
            [
                "Potency is time-dependent: an IC50 without a stated pre-incubation time is "
                "not interpretable. Ask for kinact/KI.",
                "Any QSAR model trained on equilibrium IC50 data is the wrong tool for ranking "
                "these — the model does not know about residence time or reactivity.",
                "Selectivity depends on intrinsic warhead reactivity as much as on recognition; "
                "a glutathione reactivity assay is the standard counter-screen.",
                "Check the target actually has an addressable nucleophile positioned for the "
                "warhead; without structural evidence a warhead is just a liability.",
            ]
            if found
            else ["No covalent warhead detected; standard equilibrium assumptions apply."]
        ),
        "caveat": (
            "Substructure detection only. It cannot tell a designed warhead from an incidental "
            "electrophile, and it says nothing about whether the compound actually forms an adduct."
        ),
    }
