"""Retrosynthetic analysis: disconnections, routes and feasibility.

How this works, and what it is not
----------------------------------
Disconnections come from two places: a SMARTS template library for ordinary
intermolecular bond formations, and explicit bond surgery for ring-opening
(macrocyclisation) steps, where a 1->2 template would produce nonsense.

Routes are then scored on a small number of things that actually determine
whether a chemist can run them: whether the ends are buyable, how many steps,
how reliable each transform is, whether there is a chemoselectivity conflict,
and whether protecting groups will be needed.

What this is **not** is a precedent. A disconnection with a high score is a
hypothesis. Every step carries a `precedent_query` so the literature tools can
look for real examples, and `feasibility()` reports precedent coverage
separately from template score. A route with no precedents found is reported as
unprecedented.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Iterable

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

from ..provenance import cite_software
from ..safety import guard_synthesis
from .descriptors import physchem
from .reactions import (
    Template,
    compiled_templates,
    procedural_templates,
    register_template_citation,
)
from .structure import canonical_smiles, parse

RDLogger.DisableLog("rdApp.*")

MIN_MACRO_RING = 10


@dataclass
class Disconnection:
    template: str
    reaction_class: str
    forward_description: str
    conditions: str
    reliability_prior: float
    product: str
    precursors: list[str]
    reagents_note: str = ""
    macrocyclization: bool = False
    ring_size: int | None = None
    concerns: list[str] = field(default_factory=list)
    tolerates: str = ""
    fails_when: str = ""
    scale_notes: str = ""
    precedent_query: str = ""
    citation: str | None = None
    score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RouteStep:
    step: int
    forward_description: str
    template: str
    reaction_class: str
    conditions: str
    reactants: list[str]
    product: str
    reliability_prior: float
    concerns: list[str]
    precedent_query: str
    citation: str | None
    macrocyclization: bool = False
    scale_notes: str = ""


@dataclass
class Route:
    target: str
    steps: list[RouteStep]
    starting_materials: list[dict[str, Any]]
    n_steps: int
    score: float
    longest_linear_sequence: int
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "n_steps": self.n_steps,
            "score": round(self.score, 3),
            "longest_linear_sequence": self.longest_linear_sequence,
            "steps": [asdict(s) for s in self.steps],
            "starting_materials": self.starting_materials,
            "notes": self.notes,
        }


# --- availability --------------------------------------------------------


@lru_cache(maxsize=4096)
def availability(smiles: str) -> tuple[bool, str, str | None]:
    """Is this plausibly a catalogue compound?

    There is no free, authoritative buyability API, so this is an explicit
    proxy: a PubChem record exists, the molecule is small/simple, and its
    synthetic accessibility is low. The answer says which test it passed so the
    judgement is never mistaken for a supplier quote.
    """
    try:
        mol = parse(smiles)
    except Exception:
        return False, "unparsable structure", None

    heavy = mol.GetNumHeavyAtoms()
    weight = Descriptors.MolWt(mol)
    sa = physchem(mol)["values"]["sa_score"]

    if heavy <= 6 or weight < 120:
        return True, f"trivially simple reagent (MW {weight:.0f}, {heavy} heavy atoms)", None

    try:
        from ..sources import pubchem

        record = pubchem.lookup(canonical_smiles(mol), kind="smiles")
        cid = record["cid"]
        citation = record["citations"][0]
        known = True
    except Exception:
        return (
            False,
            f"no PubChem record found for this exact structure (MW {weight:.0f}, SA {sa})",
            None,
        )

    if known and (heavy <= 24 or sa <= 3.6):
        return (
            True,
            f"known compound (PubChem CID {cid}), small/simple enough to be a likely catalogue item "
            f"(MW {weight:.0f}, {heavy} heavy atoms, SA score {sa})",
            citation,
        )
    return (
        False,
        f"known compound (PubChem CID {cid}) but complex (MW {weight:.0f}, {heavy} heavy atoms, "
        f"SA score {sa}) — treat as an intermediate to be made, or check a supplier catalogue",
        citation,
    )


# --- procedural ring opening --------------------------------------------


def _macro_bonds(mol: Chem.Mol, kind: str) -> list[tuple[int, int, int]]:
    """Find breakable bonds inside a macro-ring. Returns (begin, end, ring_size)."""
    ring_info = mol.GetRingInfo()
    found: list[tuple[int, int, int]] = []

    for bond in mol.GetBonds():
        if bond.GetIsAromatic():
            continue
        sizes = [
            size
            for size in range(MIN_MACRO_RING, 40)
            if bond.IsInRingSize(size)
        ]
        if not sizes:
            continue
        ring_size = min(sizes)
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()

        if kind == "amide":
            for carbon, nitrogen in ((a, b), (b, a)):
                if (
                    carbon.GetSymbol() == "C"
                    and nitrogen.GetSymbol() == "N"
                    and bond.GetBondType() == Chem.BondType.SINGLE
                    and any(
                        nb.GetSymbol() == "O"
                        and mol.GetBondBetweenAtoms(carbon.GetIdx(), nb.GetIdx()).GetBondType()
                        == Chem.BondType.DOUBLE
                        for nb in carbon.GetNeighbors()
                    )
                ):
                    found.append((carbon.GetIdx(), nitrogen.GetIdx(), ring_size))

        elif kind == "ester":
            for carbon, oxygen in ((a, b), (b, a)):
                if (
                    carbon.GetSymbol() == "C"
                    and oxygen.GetSymbol() == "O"
                    and bond.GetBondType() == Chem.BondType.SINGLE
                    and any(
                        nb.GetSymbol() == "O"
                        and mol.GetBondBetweenAtoms(carbon.GetIdx(), nb.GetIdx()).GetBondType()
                        == Chem.BondType.DOUBLE
                        for nb in carbon.GetNeighbors()
                    )
                ):
                    found.append((carbon.GetIdx(), oxygen.GetIdx(), ring_size))

        elif kind == "ether":
            for oxygen, carbon in ((a, b), (b, a)):
                if (
                    oxygen.GetSymbol() == "O"
                    and carbon.GetSymbol() == "C"
                    and bond.GetBondType() == Chem.BondType.SINGLE
                    and oxygen.GetDegree() == 2
                    and not any(
                        nb.GetSymbol() == "O"
                        and mol.GetBondBetweenAtoms(carbon.GetIdx(), nb.GetIdx()).GetBondType()
                        == Chem.BondType.DOUBLE
                        for nb in carbon.GetNeighbors()
                    )
                    and not carbon.GetIsAromatic()
                ):
                    found.append((oxygen.GetIdx(), carbon.GetIdx(), ring_size))

        elif kind == "alkene":
            if (
                bond.GetBondType() == Chem.BondType.DOUBLE
                and a.GetSymbol() == "C"
                and b.GetSymbol() == "C"
            ):
                found.append((a.GetIdx(), b.GetIdx(), ring_size))

    return found


def _open_ring(mol: Chem.Mol, kind: str, first: int, second: int) -> str | None:
    """Break one macro-ring bond and cap the ends to give a linear precursor."""
    editable = Chem.RWMol(mol)
    editable.RemoveBond(first, second)

    def add_h(index: int, count: int = 1) -> None:
        atom = editable.GetAtomWithIdx(index)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(atom.GetNumExplicitHs() + count)

    if kind == "amide":  # C(=O)-N  ->  C(=O)OH + N-H
        oxygen = editable.AddAtom(Chem.Atom(8))
        editable.AddBond(first, oxygen, Chem.BondType.SINGLE)
        editable.GetAtomWithIdx(oxygen).SetNumExplicitHs(1)
        add_h(second)
    elif kind == "ester":  # C(=O)-O  ->  C(=O)OH + HO-
        oxygen = editable.AddAtom(Chem.Atom(8))
        editable.AddBond(first, oxygen, Chem.BondType.SINGLE)
        editable.GetAtomWithIdx(oxygen).SetNumExplicitHs(1)
        add_h(second)
    elif kind == "ether":  # O-C  ->  O-H + C-Br
        add_h(first)
        bromine = editable.AddAtom(Chem.Atom(35))
        editable.AddBond(second, bromine, Chem.BondType.SINGLE)
    elif kind == "alkene":  # C=C  ->  two terminal alkenes
        for index in (first, second):
            methylene = editable.AddAtom(Chem.Atom(6))
            editable.AddBond(index, methylene, Chem.BondType.DOUBLE)
            editable.GetAtomWithIdx(methylene).SetNumExplicitHs(2)
    else:
        return None

    try:
        product = editable.GetMol()
        Chem.SanitizeMol(product)
        return Chem.MolToSmiles(product)
    except Exception:
        return None


# --- chemoselectivity screening -----------------------------------------

CONFLICT_PATTERNS: list[tuple[str, str, str]] = [
    ("[CX3](=O)[OX2H1]", "free carboxylic acid", "will compete in couplings and poison Pd catalysts — protect as an ester"),
    ("[NX3;H2][CX4]", "free primary aliphatic amine", "will compete in acylations/alkylations — Boc protection likely needed"),
    ("[NX3;H2][c]", "free aniline NH2", "nucleophilic and oxidation-prone; may need protection"),
    ("[OX2H1][CX4]", "free aliphatic alcohol", "competes in esterification/etherification — protect or exploit deliberately"),
    ("[OX2H1][c]", "free phenol", "acidic and nucleophilic; may be deprotonated by the base used elsewhere"),
    ("[SX2H1]", "free thiol", "oxidises readily and poisons metal catalysts"),
    ("[CX3H1](=O)", "aldehyde", "electrophilic; survives few basic or nucleophilic conditions unprotected"),
]


def _conflicts(smiles: str, template: Template) -> list[str]:
    """Functional groups that will interfere with the proposed step."""
    try:
        mol = parse(smiles)
    except Exception:
        return []
    issues: list[str] = []
    for smarts, label, consequence in CONFLICT_PATTERNS:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue
        matches = mol.GetSubstructMatches(pattern)
        if not matches:
            continue
        relevant = (
            ("amine" in label and "C-N" in template.reaction_class)
            or ("acid" in label and template.reaction_class in {"C-N bond formation", "C-O bond formation"})
            or ("alcohol" in label and "C-O" in template.reaction_class)
            or ("phenol" in label and "C-O" in template.reaction_class)
            or ("thiol" in label)
            or ("aldehyde" in label)
        )
        if relevant:
            issues.append(f"{label} present ({len(matches)}x): {consequence}")

    # Duplicate reactive sites mean the step cannot be regioselective as drawn.
    reaction = template.reaction()
    if reaction is not None:
        try:
            n_matches = len(reaction.RunReactants((mol,)))
            if n_matches > 1:
                issues.append(
                    f"{n_matches} equivalent sites match this transform — the step is not "
                    "regioselective as drawn; a blocking/protecting strategy or a different "
                    "disconnection is needed"
                )
        except Exception:
            pass
    return issues


# --- one-step disconnection ---------------------------------------------


def disconnect(smiles: str, *, include_macro: bool = True) -> list[Disconnection]:
    """All single-step disconnections for one target."""
    target = canonical_smiles(smiles)
    mol = parse(target)
    cite_software("software:rdkit", "method:retrosynthesis_review")
    target_is_macrocycle = physchem(mol)["values"]["is_macrocycle"]
    results: list[Disconnection] = []

    # SMARTS-driven intermolecular disconnections.
    for template in compiled_templates():
        reaction = template.reaction()
        if reaction is None:
            continue
        try:
            outcomes = reaction.RunReactants((mol,))
        except Exception:
            continue
        seen: set[tuple[str, ...]] = set()
        for outcome in outcomes:
            precursors: list[str] = []
            ok = True
            for product in outcome:
                try:
                    Chem.SanitizeMol(product)
                    precursors.append(Chem.MolToSmiles(product))
                except Exception:
                    ok = False
                    break
            if not ok or not precursors:
                continue
            key = tuple(sorted(precursors))
            if key in seen:
                continue
            seen.add(key)
            concerns = _conflicts(target, template)
            # A disconnection that leaves a macrocycle intact has not addressed the
            # hardest part of the synthesis — say so, because the ranking should not
            # imply otherwise.
            main_precursor = max(
                (p for p in precursors if _parsable(p)),
                key=lambda p: physchem(p)["values"]["heavy_atoms"],
                default=None,
            )
            if (
                target_is_macrocycle
                and main_precursor
                and physchem(main_precursor)["values"]["is_macrocycle"]
            ):
                concerns.append(
                    "the macrocycle is retained in the precursor: this step does not solve the "
                    "ring closure, which is the strategically decisive problem here"
                )
            citation = register_template_citation(template)
            results.append(
                Disconnection(
                    template=template.name,
                    reaction_class=template.reaction_class,
                    forward_description=template.forward_description,
                    conditions=template.conditions,
                    reliability_prior=template.reliability,
                    product=target,
                    precursors=sorted(precursors),
                    macrocyclization=False,
                    concerns=concerns,
                    tolerates=template.tolerates,
                    fails_when=template.fails_when,
                    scale_notes=template.scale_notes,
                    precedent_query=template.precedent_query,
                    citation=citation,
                )
            )

    # Procedural ring-opening disconnections.
    if include_macro:
        for template in procedural_templates():
            kind = template.procedural
            for first, second, ring_size in _macro_bonds(mol, kind):
                precursor = _open_ring(mol, kind, first, second)
                if not precursor:
                    continue
                concerns = [
                    f"macrocyclisation of a {ring_size}-membered ring: run at high dilution "
                    "(1-10 mM) with slow addition; oligomerisation is the competing pathway",
                ]
                if ring_size <= 11:
                    concerns.append(
                        f"{ring_size}-membered rings are transannular-strained and are the "
                        "hardest size class to close"
                    )
                if kind == "alkene":
                    concerns.append(
                        "RCM gives an E/Z mixture unless controlled; the alkene usually needs "
                        "hydrogenation afterwards, which adds a diastereoselectivity problem"
                    )
                results.append(
                    Disconnection(
                        template=template.name,
                        reaction_class=template.reaction_class,
                        forward_description=template.forward_description,
                        conditions=template.conditions,
                        reliability_prior=template.reliability,
                        product=target,
                        precursors=[precursor],
                        reagents_note="intramolecular — one linear precursor",
                        macrocyclization=True,
                        ring_size=ring_size,
                        concerns=concerns,
                        tolerates=template.tolerates,
                        fails_when=template.fails_when,
                        scale_notes=template.scale_notes,
                        precedent_query=template.precedent_query,
                        citation=register_template_citation(template),
                    )
                )

    for disconnection in results:
        disconnection.score = _score_step(disconnection)

    results.sort(key=lambda d: d.score, reverse=True)
    return results


def _score_step(disconnection: Disconnection) -> float:
    """Rank a disconnection. Priors and penalties only — never a probability."""
    score = disconnection.reliability_prior

    # Penalise unresolved selectivity problems.
    score -= 0.12 * len([c for c in disconnection.concerns if "not regioselective" in c])
    score -= 0.05 * len([c for c in disconnection.concerns if "protect" in c.lower()])

    # Reward simplification: precursors should be simpler than the product.
    try:
        product_complexity = physchem(disconnection.product)["values"]["sa_score"]
        precursor_complexities = [
            physchem(p)["values"]["sa_score"] for p in disconnection.precursors
        ]
        if precursor_complexities:
            simplification = product_complexity - max(precursor_complexities)
            score += max(min(simplification * 0.15, 0.25), -0.15)
    except Exception:
        pass

    # Reward buyable precursors.
    buyable = sum(1 for p in disconnection.precursors if availability(p)[0])
    if disconnection.precursors:
        score += 0.2 * (buyable / len(disconnection.precursors))

    if disconnection.macrocyclization and (disconnection.ring_size or 99) <= 11:
        score -= 0.1

    # On a macrocyclic target, a step that leaves the ring intact is strategically weak.
    if any("macrocycle is retained" in c for c in disconnection.concerns):
        score -= 0.3

    return round(min(max(score, 0.01), 1.0), 3)


# --- multi-step planning -------------------------------------------------


def plan(
    smiles: str,
    *,
    max_depth: int = 3,
    beam: int = 3,
    max_routes: int = 5,
) -> dict[str, Any]:
    """Search for routes back to plausible starting materials."""
    verdict = guard_synthesis(smiles=smiles)
    if not verdict.allowed:
        return {
            "refused": True,
            "safety": verdict.as_dict(),
            "target": smiles,
            "routes": [],
        }

    target = canonical_smiles(smiles)
    routes: list[Route] = []

    def expand(current: str, depth: int, trail: list[Disconnection]) -> None:
        if len(routes) >= max_routes * 3:
            return
        available, reason, citation = availability(current)
        if available or depth >= max_depth:
            if trail:
                routes.append(_assemble(target, trail))
            return
        options = disconnect(current)[:beam]
        if not options:
            if trail:
                route = _assemble(target, trail)
                route.notes.append(
                    f"search stopped at {current}: no template disconnection found and the "
                    "structure does not look like a catalogue item"
                )
                routes.append(route)
            return
        for option in options:
            # Follow the most complex precursor; the simpler ones are leaves.
            ordered = sorted(
                option.precursors,
                key=lambda p: physchem(p)["values"]["sa_score"] if _parsable(p) else 0,
                reverse=True,
            )
            if not ordered:
                continue
            expand(ordered[0], depth + 1, trail + [option])

    expand(target, 0, [])

    unique: dict[str, Route] = {}
    for route in routes:
        signature = "|".join(f"{s.template}:{s.product}" for s in route.steps)
        if signature not in unique or route.score > unique[signature].score:
            unique[signature] = route
    ranked = sorted(unique.values(), key=lambda r: (r.score, -r.n_steps), reverse=True)[:max_routes]

    return {
        "target": target,
        "safety": verdict.as_dict(),
        "n_routes": len(ranked),
        "routes": [r.as_dict() for r in ranked],
        "search": {"max_depth": max_depth, "beam": beam},
        "caveat": (
            "Template-generated hypotheses ranked by heuristics. No step here has been "
            "checked against the literature yet — run precedent search on each step's "
            "`precedent_query` before treating a route as real."
        ),
        "citations": cite_software("software:rdkit", "method:retrosynthesis_review", "method:sa_score"),
    }


def _parsable(smiles: str) -> bool:
    try:
        parse(smiles)
        return True
    except Exception:
        return False


def _assemble(target: str, trail: list[Disconnection]) -> Route:
    """Turn a retro trail into a forward-ordered route."""
    steps: list[RouteStep] = []
    forward = list(reversed(trail))
    for index, disconnection in enumerate(forward, start=1):
        steps.append(
            RouteStep(
                step=index,
                forward_description=disconnection.forward_description,
                template=disconnection.template,
                reaction_class=disconnection.reaction_class,
                conditions=disconnection.conditions,
                reactants=disconnection.precursors,
                product=disconnection.product,
                reliability_prior=disconnection.reliability_prior,
                concerns=disconnection.concerns,
                precedent_query=disconnection.precedent_query,
                citation=disconnection.citation,
                macrocyclization=disconnection.macrocyclization,
                scale_notes=disconnection.scale_notes,
            )
        )

    leaves: list[str] = []
    intermediates = {d.product for d in trail}
    for disconnection in trail:
        for precursor in disconnection.precursors:
            if precursor not in intermediates:
                leaves.append(precursor)

    starting_materials = []
    for leaf in dict.fromkeys(leaves):
        available, reason, citation = availability(leaf)
        starting_materials.append(
            {
                "smiles": leaf,
                "likely_available": available,
                "assessment": reason,
                "citation": citation,
            }
        )

    step_scores = [d.score for d in trail] or [0.0]
    product_of_scores = 1.0
    for score in step_scores:
        product_of_scores *= score
    availability_fraction = (
        sum(1 for s in starting_materials if s["likely_available"]) / len(starting_materials)
        if starting_materials
        else 0.0
    )
    score = (product_of_scores ** (1 / len(step_scores))) * (0.6 + 0.4 * availability_fraction)
    score *= 0.92 ** max(len(steps) - 1, 0)  # step count costs something

    return Route(
        target=target,
        steps=steps,
        starting_materials=starting_materials,
        n_steps=len(steps),
        score=round(score, 3),
        longest_linear_sequence=len(steps),
        notes=[],
    )


# --- feasibility ---------------------------------------------------------


def feasibility(smiles: str, *, route: dict[str, Any] | None = None) -> dict[str, Any]:
    """Structured feasibility judgement for a target, optionally for a route."""
    verdict = guard_synthesis(smiles=smiles)
    target = canonical_smiles(smiles)
    values = physchem(target)["values"]
    cite_software("method:sa_score", "software:rdkit")

    drivers: list[dict[str, str]] = []

    sa = values["sa_score"]
    if sa < 3:
        drivers.append({"factor": "synthetic accessibility", "reading": f"SA score {sa} — routine", "weight": "favourable"})
    elif sa < 4.5:
        drivers.append({"factor": "synthetic accessibility", "reading": f"SA score {sa} — moderate, standard medicinal-chemistry effort", "weight": "neutral"})
    else:
        drivers.append({"factor": "synthetic accessibility", "reading": f"SA score {sa} — demanding; expect a long sequence or specialised chemistry", "weight": "adverse"})

    if values["is_macrocycle"]:
        sizes = values["macrocycle_ring_sizes"]
        hardest = min(sizes)
        drivers.append(
            {
                "factor": "macrocyclisation",
                "reading": (
                    f"ring size(s) {sizes}. Closure needs high dilution and conformational "
                    "pre-organisation; "
                    + (
                        "11-membered and smaller rings are the hardest class"
                        if hardest <= 11
                        else "12-14 membered rings close reasonably with the right tether"
                    )
                ),
                "weight": "adverse" if hardest <= 11 else "neutral",
            }
        )

    centres = values["stereocentres"]
    if centres >= 3:
        drivers.append({"factor": "stereochemistry", "reading": f"{centres} stereocentres — asymmetric steps or resolution needed, and the step count grows", "weight": "adverse"})
    elif centres:
        drivers.append({"factor": "stereochemistry", "reading": f"{centres} stereocentre(s) — manageable with a chiral building block or asymmetric step", "weight": "neutral"})

    protect = [
        label
        for smarts, label, _ in CONFLICT_PATTERNS
        if (pattern := Chem.MolFromSmarts(smarts)) is not None
        and parse(target).HasSubstructMatch(pattern)
    ]
    if protect:
        drivers.append({"factor": "protecting groups", "reading": f"reactive groups present ({', '.join(protect)}) — expect protection/deprotection steps", "weight": "neutral"})

    if values["mw"] > 600:
        drivers.append({"factor": "size", "reading": f"MW {values['mw']} — fragment coupling strategy likely, purification gets harder", "weight": "adverse"})

    assessment = {
        "target": target,
        "safety": verdict.as_dict(),
        "computed": {
            "sa_score": sa,
            "mw": values["mw"],
            "stereocentres": centres,
            "is_macrocycle": values["is_macrocycle"],
            "macrocycle_ring_sizes": values["macrocycle_ring_sizes"],
            "rings": values["rings"],
            "fraction_csp3": values["fraction_csp3"],
        },
        "drivers": drivers,
        "verdict_scale": "favourable | neutral | adverse per driver; no single number is given because the drivers are not commensurable",
    }

    if route:
        steps = route.get("steps", [])
        unprecedented = [s for s in steps if not s.get("precedent_checked")]
        macro_steps = [s for s in steps if s.get("macrocyclization")]
        low_reliability = [s for s in steps if (s.get("reliability_prior") or 1) < 0.65]
        assessment["route_assessment"] = {
            "n_steps": len(steps),
            "steps_with_template_reliability_below_0.65": [s["template"] for s in low_reliability],
            "macrocyclisation_steps": [s["template"] for s in macro_steps],
            "steps_without_literature_precedent_checked": len(unprecedented),
            "starting_material_availability": route.get("starting_materials"),
            "overall": (
                "The weakest link sets the route's feasibility: "
                + (
                    ", ".join(s["template"] for s in (low_reliability + macro_steps)[:3])
                    or "no individually weak step identified"
                )
            ),
        }

    assessment["next_actions"] = [
        "run precedent search on each step's precedent_query to replace priors with real examples",
        "check starting-material availability against a supplier catalogue (this tool uses a proxy)",
        "for any macrocyclisation, look for a reported closure on the same ring size and tether",
    ]
    return assessment


def summarize_disconnections(smiles: str, *, top: int = 8) -> dict[str, Any]:
    """Compact one-step analysis, suitable for a tool result."""
    verdict = guard_synthesis(smiles=smiles)
    if not verdict.allowed:
        return {"refused": True, "safety": verdict.as_dict(), "disconnections": []}
    options = disconnect(smiles)
    return {
        "target": canonical_smiles(smiles),
        "n_disconnections": len(options),
        "safety": verdict.as_dict(),
        "disconnections": [d.as_dict() for d in options[:top]],
        "caveat": (
            "Ranked by template priors and heuristics, not by precedent. Confirm each with a "
            "literature search before relying on it."
        ),
    }
