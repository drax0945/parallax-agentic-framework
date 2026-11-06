"""Analog generation, matched molecular pairs, and macrocyclisation design.

The design loop this supports is the one medicinal chemists actually run:

1. look at what the SAR already says (matched molecular pairs over measured data);
2. propose structural changes with a documented rationale (bioisostere library,
   rigidification, macrocyclisation);
3. filter on properties and predicted potency, keeping the applicability domain
   in view;
4. ask whether the proposal can be made (`retro.feasibility`).

Everything proposed here is a hypothesis with a stated reason. Transformations
carry the literature that established them, so a design decision can be
defended rather than asserted.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors

from ..provenance import Citation, cite_software, registry
from .descriptors import physchem, similarity
from .structure import canonical_smiles, parse

RDLogger.DisableLog("rdApp.*")


# --- bioisostere / modification library ---------------------------------


@dataclass
class Transform:
    name: str
    smarts: str  # forward: motif >> replacement
    rationale: str
    expected_effect: str
    doi: str | None = None
    reference_title: str | None = None
    reference_authors: list[str] = field(default_factory=list)
    reference_year: int | None = None
    reference_venue: str | None = None
    category: str = "bioisostere"

    def reaction(self):
        return AllChem.ReactionFromSmarts(self.smarts)

    def cite(self) -> str | None:
        if not self.doi:
            return None
        return registry().add(
            Citation(
                key=f"doi:{self.doi.lower()}",
                kind="method",
                title=self.reference_title or self.name,
                source_db="parallax/transform-library",
                authors=self.reference_authors,
                year=self.reference_year,
                venue=self.reference_venue,
                doi=self.doi,
                note=f"evidence for the '{self.name}' modification strategy",
            )
        )


TRANSFORMS: list[Transform] = [
    Transform(
        name="Aromatic CH -> CF (fluorine scan)",
        smarts="[cH:1]>>[c:1]F",
        rationale="Blocks a metabolically labile aromatic position and lowers the pKa of nearby basic centres",
        expected_effect="metabolic stability up, logD slightly up, often small potency change",
        doi="10.1021/jm501853m",
        reference_title="Applications of fluorine in medicinal chemistry",
        reference_authors=["Gillis, Eric P.", "Eastman, Kyle J.", "Hill, Matthew D.", "Donnelly, David J.", "Meanwell, Nicholas A."],
        reference_year=2015,
        reference_venue="Journal of Medicinal Chemistry",
        category="metabolic blocking",
    ),
    Transform(
        name="Aromatic CH -> CCH3 (methyl scan)",
        smarts="[cH:1]>>[c:1]C",
        rationale="A single methyl can gain an order of magnitude in potency by filling a small lipophilic pocket or enforcing a torsion",
        expected_effect="potency can jump or collapse; lipophilicity up by ~0.5 log",
        doi="10.1002/anie.201303207",
        reference_title="Profound methyl effects in drug discovery and a call for new C-H methylation reactions",
        reference_authors=["Schönherr, Heike", "Cernak, Tim"],
        reference_year=2013,
        reference_venue="Angewandte Chemie International Edition",
        category="potency probe",
    ),
    Transform(
        name="Benzene -> pyridine (aza-scan)",
        smarts="[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1>>[n:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1",
        rationale="Lowers logP and adds an HBA; often improves solubility and reduces hERG binding",
        expected_effect="logP down ~1, solubility up, permeability may drop",
        doi="10.1021/jm501100b",
        reference_title="Heteroaromatic rings of the future",
        reference_authors=["Pitt, William R.", "Parry, David M.", "Perry, Benjamin G.", "Groom, Colin R."],
        reference_year=2009,
        reference_venue="Journal of Medicinal Chemistry",
        category="property tuning",
    ),
    Transform(
        name="Carboxylic acid -> tetrazole",
        smarts="[C:1](=[OX1])[OX2H1]>>[c:1]1[nH][n][n][n]1",
        rationale="Classic acid isostere: similar pKa and geometry, more lipophilic, resists glucuronidation",
        expected_effect="permeability up, metabolic clearance route changed, potency usually retained",
        doi="10.1021/jm200187y",
        reference_title="Carboxylic acid (bio)isosteres in drug design",
        reference_authors=["Ballatore, Carlo", "Huryn, Donna M.", "Smith, Amos B."],
        reference_year=2013,
        reference_venue="ChemMedChem",
        category="bioisostere",
    ),
    Transform(
        name="Carboxylic acid -> acylsulfonamide",
        smarts="[C:1](=[OX1:2])[OX2H1]>>[C:1](=[O:2])NS(=O)(=O)C",
        rationale="Acid isostere with tunable pKa and better passive permeability",
        expected_effect="permeability up; adds molecular weight",
        doi="10.1021/jm200187y",
        reference_title="Carboxylic acid (bio)isosteres in drug design",
        reference_authors=["Ballatore, Carlo", "Huryn, Donna M.", "Smith, Amos B."],
        reference_year=2013,
        reference_venue="ChemMedChem",
        category="bioisostere",
    ),
    Transform(
        name="tert-Butyl -> bicyclo[1.1.1]pentane",
        smarts="[CX4:1]([CH3])([CH3])[CH3]>>[C:1]12CC(C1)C2",
        rationale="Saturated bioisostere that keeps the lipophilic bulk while removing metabolically soft methyls",
        expected_effect="metabolic stability up, solubility often up, potency usually retained",
        doi="10.1021/jm300407g",
        reference_title="Application of the bicyclo[1.1.1]pentane motif as a nonclassical phenyl ring bioisostere",
        reference_authors=["Stepan, Antonia F.", "Subramanyam, Chakrapani", "Efremov, Ivan V.", "Dutra, Jason K.", "O'Sullivan, Timothy J.", "DiRico, Kenneth J."],
        reference_year=2012,
        reference_venue="Journal of Medicinal Chemistry",
        category="bioisostere",
    ),
    Transform(
        name="Methyl ether -> difluoromethyl ether",
        smarts="[OX2:1][CH3]>>[O:1]C(F)F",
        rationale="Blocks O-demethylation, the commonest metabolic soft spot on an anisole",
        expected_effect="metabolic stability up, logD up slightly, HBA strength reduced",
        doi="10.1021/jm501853m",
        reference_title="Applications of fluorine in medicinal chemistry",
        reference_authors=["Gillis, Eric P.", "Eastman, Kyle J.", "Hill, Matthew D.", "Donnelly, David J.", "Meanwell, Nicholas A."],
        reference_year=2015,
        reference_venue="Journal of Medicinal Chemistry",
        category="metabolic blocking",
    ),
    Transform(
        name="Amide -> 1,2,4-oxadiazole",
        smarts="[C:1](=[OX1])[NX3;H1][C:2]>>[c:1]1[n][c:2][o][n]1",
        rationale="Rigid, metabolically robust amide surrogate that keeps the dipole and vector",
        expected_effect="protease/amidase stability up, conformation locked, solubility often down",
        doi="10.1021/acs.jmedchem.5b01129",
        reference_title="Heterocycles as amide bioisosteres",
        reference_authors=["Meanwell, Nicholas A."],
        reference_year=2011,
        reference_venue="Journal of Medicinal Chemistry",
        category="bioisostere",
    ),
    Transform(
        name="Piperazine -> piperidine (remove basic centre)",
        smarts="[NX3:1]1[CH2][CH2][NX3;H1][CH2][CH2]1>>[N:1]1[CH2][CH2][CH2][CH2][CH2]1",
        rationale="Removing a basic nitrogen is the standard first move against hERG binding and high volume of distribution",
        expected_effect="hERG risk down, solubility down, potency may drop if the amine made a salt bridge",
        doi="10.1021/jm701122q",
        reference_title="Generation of a set of simple, interpretable ADMET rules of thumb",
        reference_authors=["Gleeson, M. Paul"],
        reference_year=2008,
        reference_venue="Journal of Medicinal Chemistry",
        category="liability mitigation",
    ),
    Transform(
        name="Phenyl -> thiophene",
        smarts="[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1>>[c:1]1[cH:2][cH:3][s][c:6]1",
        rationale="Smaller aromatic with a different vector and electron density; frequently improves fit in narrow pockets",
        expected_effect="potency change unpredictable; adds a CYP-bioactivation watch item",
        doi="10.1021/jm501100b",
        reference_title="Heteroaromatic rings of the future",
        reference_authors=["Pitt, William R.", "Parry, David M.", "Perry, Benjamin G.", "Groom, Colin R."],
        reference_year=2009,
        reference_venue="Journal of Medicinal Chemistry",
        category="scaffold hop",
    ),
    Transform(
        name="Add morpholine solubilising group to aromatic CH",
        smarts="[cH:1]>>[c:1]N1CCOCC1",
        rationale="Standard solubilising handle that usually sits in solvent-exposed space",
        expected_effect="solubility up, logP down, potency often unchanged if the vector points to solvent",
        category="property tuning",
    ),
    Transform(
        name="Secondary amide N-methylation",
        smarts="[C:1](=[OX1:2])[NX3;H1:3]>>[C:1](=[O:2])[N:3]C",
        rationale="Removes an HBD and changes the amide rotamer preference; a classic permeability move in macrocycles",
        expected_effect="permeability up, HBD count down, potency depends on whether the NH was a contact",
        doi="10.1021/jm1012374",
        reference_title="Macrocycles in new drug discovery",
        reference_authors=["Marsault, Eric", "Peterson, Mark L."],
        reference_year=2011,
        reference_venue="Journal of Medicinal Chemistry",
        category="permeability",
    ),
]


# --- analog enumeration --------------------------------------------------


def enumerate_analogs(
    smiles: str,
    *,
    categories: Sequence[str] | None = None,
    max_per_transform: int = 4,
    max_total: int = 60,
) -> dict[str, Any]:
    """Apply the transform library to one structure."""
    parent = canonical_smiles(smiles)
    parent_mol = parse(parent)
    parent_values = physchem(parent_mol)["values"]
    cite_software("software:rdkit")

    produced: dict[str, dict[str, Any]] = {}
    for transform in TRANSFORMS:
        if categories and transform.category not in categories:
            continue
        reaction = transform.reaction()
        if reaction is None:
            continue
        try:
            outcomes = reaction.RunReactants((parent_mol,))
        except Exception:
            continue
        count = 0
        for outcome in outcomes:
            if count >= max_per_transform or len(produced) >= max_total:
                break
            for product in outcome:
                try:
                    Chem.SanitizeMol(product)
                    analog = Chem.MolToSmiles(product)
                except Exception:
                    continue
                if analog == parent or analog in produced:
                    continue
                values = physchem(analog)["values"]
                produced[analog] = {
                    "smiles": analog,
                    "transform": transform.name,
                    "category": transform.category,
                    "rationale": transform.rationale,
                    "expected_effect": transform.expected_effect,
                    "citation": transform.cite(),
                    "similarity_to_parent": similarity(parent, analog),
                    "property_deltas": {
                        "mw": round(values["mw"] - parent_values["mw"], 1),
                        "clogp": round(values["clogp"] - parent_values["clogp"], 2),
                        "tpsa": round(values["tpsa"] - parent_values["tpsa"], 1),
                        "hbd": values["hbd"] - parent_values["hbd"],
                        "rotatable_bonds": values["rotatable_bonds"] - parent_values["rotatable_bonds"],
                        "sa_score": round(values["sa_score"] - parent_values["sa_score"], 2),
                    },
                    "properties": {
                        "mw": values["mw"],
                        "clogp": values["clogp"],
                        "tpsa": values["tpsa"],
                        "qed": values["qed"],
                        "sa_score": values["sa_score"],
                    },
                }
                count += 1

    return {
        "parent": parent,
        "parent_properties": {
            k: parent_values[k] for k in ("mw", "clogp", "tpsa", "hbd", "qed", "sa_score")
        },
        "n_analogs": len(produced),
        "analogs": list(produced.values()),
        "note": (
            "Enumerated hypotheses from a documented transform library. Each carries the "
            "literature behind the strategy, not evidence about this particular compound."
        ),
    }


# --- macrocyclisation design --------------------------------------------

TETHERS: dict[str, str] = {
    "CH2": "C",
    "CH2CH2": "CC",
    "OCH2": "OC",
    "CH2OCH2": "COC",
    "NHC(=O)": "NC=O",
    "C(=O)NH": "C(=O)N",
    "CH2NHCH2": "CNC",
    "CH2CH2CH2": "CCC",
    "CH=CH": "C=C",
}


def suggest_macrocyclization(
    smiles: str,
    *,
    min_ring: int = 11,
    max_ring: int = 16,
    max_designs: int = 20,
) -> dict[str, Any]:
    """Propose rigidifying macrocyclisations of an acyclic lead.

    Finds pairs of substitutable atoms whose topological separation, plus a
    tether, would close a ring in the target size window. This is the design
    move behind macrocyclic kinase inhibitors: trade rotatable bonds for a
    pre-organised binding conformation.
    """
    parent = canonical_smiles(smiles)
    mol = parse(parent)
    parent_values = physchem(mol)["values"]
    cite_software("method:macrocycle_review", "software:rdkit")
    citation = registry().add(
        Citation(
            key="doi:10.1021/jm1012374",
            kind="method",
            title="Macrocycles in new drug discovery",
            source_db="parallax/transform-library",
            authors=["Marsault, Eric", "Peterson, Mark L."],
            year=2011,
            venue="Journal of Medicinal Chemistry",
            doi="10.1021/jm1012374",
            note="rationale for macrocyclisation as a rigidification strategy",
        )
    )

    distance_matrix = Chem.GetDistanceMatrix(mol)
    ring_info = mol.GetRingInfo()

    candidates: list[int] = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetTotalNumHs() > 0 and atom.GetSymbol() in {"C", "N"}
    ]

    designs: list[dict[str, Any]] = []
    seen: set[str] = set()

    for position, first in enumerate(candidates):
        for second in candidates[position + 1 :]:
            if ring_info.AreAtomsInSameRing(first, second):
                continue
            separation = int(distance_matrix[first][second])
            if separation < 3 or separation > max_ring - 2:
                continue
            for label, tether_smiles in TETHERS.items():
                tether_length = len(parse(tether_smiles).GetAtoms())
                ring_size = separation + tether_length + 1
                if not (min_ring <= ring_size <= max_ring):
                    continue
                built = _link_atoms(mol, first, second, tether_smiles)
                if not built or built in seen:
                    continue
                seen.add(built)
                values = physchem(built)["values"]
                if not values["is_macrocycle"]:
                    continue
                designs.append(
                    {
                        "smiles": built,
                        "ring_size": ring_size,
                        "tether": label,
                        "linked_atoms": [first, second],
                        "linked_atom_symbols": [
                            mol.GetAtomWithIdx(first).GetSymbol(),
                            mol.GetAtomWithIdx(second).GetSymbol(),
                        ],
                        "rotatable_bonds": values["rotatable_bonds"],
                        "rotatable_bond_change": values["rotatable_bonds"] - parent_values["rotatable_bonds"],
                        "properties": {
                            "mw": values["mw"],
                            "clogp": values["clogp"],
                            "tpsa": values["tpsa"],
                            "hbd": values["hbd"],
                            "sa_score": values["sa_score"],
                            "qed": values["qed"],
                        },
                        "closure_chemistry": _closure_hint(label),
                        "similarity_to_parent": similarity(parent, built),
                    }
                )
                if len(designs) >= max_designs * 4:
                    break

    # Prefer designs that genuinely rigidify, stay synthesisable, and are 12-14.
    def rank(design: dict[str, Any]) -> tuple:
        size_bonus = 1 if 12 <= design["ring_size"] <= 14 else 0
        return (
            size_bonus,
            -design["rotatable_bond_change"],
            -design["properties"]["sa_score"],
        )

    designs.sort(key=rank, reverse=True)

    return {
        "parent": parent,
        "parent_rotatable_bonds": parent_values["rotatable_bonds"],
        "n_designs": len(designs),
        "designs": designs[:max_designs],
        "design_rationale": (
            "Macrocyclisation removes rotatable bonds and pays the entropic cost of binding "
            "up front. The trade is permeability and synthesis difficulty; 12-14 membered "
            "rings are the practical sweet spot for closure."
        ),
        "citations": [citation],
        "next_steps": [
            "predict potency for each design and check the applicability domain",
            "run retro_feasibility on the shortlist — closure chemistry decides what is makeable",
            "look for a reported closure on the same ring size and tether before committing",
        ],
    }


def _closure_hint(tether_label: str) -> str:
    if "C(=O)N" in tether_label or "NHC(=O)" in tether_label:
        return "macrolactamisation (HATU/high dilution) — the best-precedented closure"
    if "O" in tether_label:
        return "macroetherification (Cs2CO3/high dilution, SNAr-assisted if the ring is electron-poor) or Mitsunobu"
    if "=" in tether_label:
        return "ring-closing metathesis (Grubbs II, high dilution), then hydrogenation"
    return "C-C coupling or alkylation under high dilution; consider an intramolecular Suzuki"


def _link_atoms(mol: Chem.Mol, first: int, second: int, tether_smiles: str) -> str | None:
    """Join two atoms through a tether, giving a new ring."""
    tether = Chem.MolFromSmiles(tether_smiles)
    if tether is None:
        return None
    editable = Chem.RWMol(mol)

    mapping: dict[int, int] = {}
    for atom in tether.GetAtoms():
        new_atom = Chem.Atom(atom.GetAtomicNum())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        mapping[atom.GetIdx()] = editable.AddAtom(new_atom)
    for bond in tether.GetBonds():
        editable.AddBond(
            mapping[bond.GetBeginAtomIdx()], mapping[bond.GetEndAtomIdx()], bond.GetBondType()
        )

    chain_ends = [
        index
        for index in mapping
        if tether.GetAtomWithIdx(index).GetDegree() <= 1
    ] or list(mapping)
    start = mapping[chain_ends[0]]
    end = mapping[chain_ends[-1]] if len(chain_ends) > 1 else start

    try:
        editable.AddBond(first, start, Chem.BondType.SINGLE)
        editable.AddBond(second, end, Chem.BondType.SINGLE)
        for index in (first, second):
            atom = editable.GetAtomWithIdx(index)
            if atom.GetNumExplicitHs() > 0:
                atom.SetNumExplicitHs(atom.GetNumExplicitHs() - 1)
            atom.SetNoImplicit(False)
        product = editable.GetMol()
        Chem.SanitizeMol(product)
        return Chem.MolToSmiles(product)
    except Exception:
        return None


# --- matched molecular pairs --------------------------------------------


def _single_cuts(smiles: str) -> list[tuple[str, str]]:
    """Fragment at each acyclic single bond: returns (core, substituent) pairs."""
    mol = parse(smiles)
    cuts: list[tuple[str, str]] = []
    for bond in mol.GetBonds():
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
        if begin.GetDegree() == 1 or end.GetDegree() == 1:
            if begin.GetSymbol() == "H" or end.GetSymbol() == "H":
                continue
        try:
            fragmented = Chem.FragmentOnBonds(mol, [bond.GetIdx()], addDummies=True)
            pieces = Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=True)
        except Exception:
            continue
        if len(pieces) != 2:
            continue
        smiles_pieces = [Chem.MolToSmiles(p) for p in pieces]
        heavy = [p.GetNumHeavyAtoms() for p in pieces]
        core_index = 0 if heavy[0] >= heavy[1] else 1
        sub_index = 1 - core_index
        # Keep only genuinely small substituent changes — that is what an MMP is.
        if heavy[sub_index] > 14:
            continue
        cuts.append((smiles_pieces[core_index], smiles_pieces[sub_index]))
    return cuts


def matched_pairs(
    records: Sequence[dict[str, Any]],
    *,
    value_key: str = "y",
    min_pairs_per_transform: int = 2,
    max_pairs: int = 400,
) -> dict[str, Any]:
    """Matched molecular pair analysis over measured data.

    `records` need `smiles` and a numeric value (default key `y`, as produced by
    `models.qsar.aggregate_measurements`). Returns per-transformation effect
    sizes with the citations of the underlying measurements.
    """
    cite_software("method:mmpa")
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        smiles = record.get("smiles")
        if not smiles or record.get(value_key) is None:
            continue
        for core, substituent in _single_cuts(smiles):
            index[core].append(
                {
                    "smiles": smiles,
                    "substituent": substituent,
                    "value": float(record[value_key]),
                    "citations": record.get("citations", []),
                }
            )

    pairs: list[dict[str, Any]] = []
    for core, members in index.items():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                left, right = members[i], members[j]
                if left["substituent"] == right["substituent"]:
                    continue
                delta = right["value"] - left["value"]
                pairs.append(
                    {
                        "core": core,
                        "from_smiles": left["smiles"],
                        "to_smiles": right["smiles"],
                        "transformation": f"{left['substituent']} >> {right['substituent']}",
                        "from_value": round(left["value"], 2),
                        "to_value": round(right["value"], 2),
                        "delta": round(delta, 2),
                        "citations": sorted(set(left["citations"]) | set(right["citations"])),
                    }
                )
                if len(pairs) >= max_pairs * 4:
                    break

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[pair["transformation"]].append(pair)

    transformations = []
    for transformation, group in grouped.items():
        if len(group) < min_pairs_per_transform:
            continue
        deltas = [g["delta"] for g in group]
        mean = sum(deltas) / len(deltas)
        spread = max(deltas) - min(deltas)
        transformations.append(
            {
                "transformation": transformation,
                "n_pairs": len(group),
                "mean_delta": round(mean, 2),
                "min_delta": round(min(deltas), 2),
                "max_delta": round(max(deltas), 2),
                "spread": round(spread, 2),
                "consistent": spread < 1.0,
                "examples": group[:3],
                "citations": sorted({c for g in group for c in g["citations"]})[:12],
            }
        )

    transformations.sort(key=lambda t: (abs(t["mean_delta"]), t["n_pairs"]), reverse=True)

    return {
        "n_compounds": len(records),
        "n_pairs_found": len(pairs),
        "n_transformations": len(transformations),
        "transformations": transformations[:60],
        "how_to_read": (
            "mean_delta is the average change in the measured value (log units) for that "
            "substituent swap, on this dataset only. A transformation with few pairs or a "
            "wide spread is not a rule — check `consistent` and `n_pairs` before acting."
        ),
        "citations": cite_software("method:mmpa"),
    }
