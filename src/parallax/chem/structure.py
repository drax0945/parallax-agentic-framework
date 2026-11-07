"""Structure input, standardisation and depiction.

Accepts whatever a chemist is likely to paste: a name, a SMILES, an InChI, an
InChIKey, a ChEMBL ID, a PubChem CID or a CAS number. Name lookups go through
PubChem (and the NCI resolver as a fallback) so the resolved structure carries a
citation back to the record it came from, rather than being recalled from a
model's memory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, inchi
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

from ..provenance import cite_software

RDLogger.DisableLog("rdApp.*")

INCHIKEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
INCHI_RE = re.compile(r"^InChI=1S?/")
CHEMBL_RE = re.compile(r"^CHEMBL\d+$", re.IGNORECASE)
CID_RE = re.compile(r"^(CID[:\s]*)?\d{1,9}$", re.IGNORECASE)
CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")


@dataclass
class Structure:
    """A resolved, standardised structure plus where it came from."""

    input: str
    smiles: str
    mol: Any = field(repr=False, default=None)
    name: str | None = None
    inchi: str | None = None
    inchikey: str | None = None
    formula: str | None = None
    identifiers: dict[str, Any] = field(default_factory=dict)
    citations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    components_dropped: list[str] = field(default_factory=list)

    @property
    def reliable(self) -> bool:
        """False when standardisation discarded chemistry the result depends on."""
        return not any(n.startswith("WARNING") for n in self.notes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "input": self.input,
            "name": self.name,
            "smiles": self.smiles,
            "inchikey": self.inchikey,
            "formula": self.formula,
            "identifiers": self.identifiers,
            "citations": self.citations,
            "notes": self.notes,
            "components_dropped": self.components_dropped,
            "properties_reliable": self.reliable,
        }


class StructureError(ValueError):
    pass


_normalizer = rdMolStandardize.Normalizer()
_uncharger = rdMolStandardize.Uncharger()
_largest = rdMolStandardize.LargestFragmentChooser()


def parse(smiles: str, *, standardize: bool = True, keep_salt: bool = False) -> Chem.Mol:
    """SMILES -> standardised RDKit mol."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise StructureError(f"could not parse SMILES: {smiles!r}")
    if not standardize:
        return mol
    try:
        mol = _normalizer.normalize(mol)
        if not keep_salt:
            mol = _largest.choose(mol)
        mol = _uncharger.uncharge(mol)
        Chem.SanitizeMol(mol)
    except Exception:  # standardisation is best-effort; never lose the molecule
        mol = Chem.MolFromSmiles(smiles)
    return mol


def canonical_smiles(smiles_or_mol: str | Chem.Mol, *, isomeric: bool = True) -> str:
    mol = parse(smiles_or_mol) if isinstance(smiles_or_mol, str) else smiles_or_mol
    return Chem.MolToSmiles(mol, isomericSmiles=isomeric)


def scaffold(smiles_or_mol: str | Chem.Mol, *, generic: bool = False) -> str:
    """Bemis-Murcko scaffold SMILES."""
    mol = parse(smiles_or_mol) if isinstance(smiles_or_mol, str) else smiles_or_mol
    core = MurckoScaffold.GetScaffoldForMol(mol)
    if generic:
        core = MurckoScaffold.MakeScaffoldGeneric(core)
    return Chem.MolToSmiles(core)


def classify_input(text: str) -> str:
    text = text.strip()
    if INCHI_RE.match(text):
        return "inchi"
    if INCHIKEY_RE.match(text):
        return "inchikey"
    if CHEMBL_RE.match(text):
        return "chembl_id"
    if CAS_RE.match(text):
        return "cas"
    if CID_RE.match(text) and not text.isalpha():
        return "cid"
    if Chem.MolFromSmiles(text) is not None and not text.isalpha():
        return "smiles"
    # Single-word all-alpha strings could be either; prefer name lookup.
    if Chem.MolFromSmiles(text) is not None and len(text) <= 3:
        return "smiles"
    return "name"


def resolve(query: str, *, standardize: bool = True) -> Structure:
    """Resolve any common identifier into a `Structure` with provenance."""
    from ..sources import pubchem, chembl, nci  # local import avoids a cycle

    query = (query or "").strip()
    if not query:
        raise StructureError("empty structure query")

    kind = classify_input(query)
    citations: list[str] = cite_software("software:rdkit")
    notes: list[str] = []
    identifiers: dict[str, Any] = {}
    name: str | None = None
    smiles: str | None = None

    if kind == "smiles":
        smiles = query
        notes.append("structure taken directly from supplied SMILES")

    elif kind == "inchi":
        mol = inchi.MolFromInchi(query)
        if mol is None:
            raise StructureError("could not parse InChI")
        smiles = Chem.MolToSmiles(mol)
        notes.append("structure parsed from supplied InChI")

    elif kind == "chembl_id":
        record = chembl.molecule(query.upper())
        smiles = record.get("smiles")
        name = record.get("pref_name")
        identifiers.update(record.get("identifiers", {}))
        citations.extend(record.get("citations", []))

    elif kind in {"name", "inchikey", "cid", "cas"}:
        try:
            record = pubchem.lookup(query, kind=kind)
            smiles = record.get("smiles")
            name = record.get("name")
            identifiers.update(record.get("identifiers", {}))
            citations.extend(record.get("citations", []))
        except Exception as exc:  # fall back to the NCI resolver for odd synonyms
            notes.append(f"PubChem lookup failed ({exc}); trying NCI resolver")
            record = nci.resolve_name(query)
            smiles = record.get("smiles")
            name = record.get("name") or query
            citations.extend(record.get("citations", []))

    if not smiles:
        raise StructureError(f"could not resolve a structure for {query!r}")

    # Standardisation keeps the largest fragment, which is what a medicinal
    # chemist wants for a salt (amlodipine besylate -> amlodipine) and badly
    # wrong for a coordination compound (cisplatin -> ammonia). Record what was
    # dropped either way, and refuse to stay quiet about the dangerous case.
    components = [c for c in smiles.split(".") if c]
    mol = parse(smiles, standardize=standardize)
    canonical = Chem.MolToSmiles(mol)
    dropped: list[str] = []
    if standardize and len(components) > 1:
        kept = set(Chem.MolToSmiles(mol).split("."))
        for component in components:
            parsed_component = Chem.MolFromSmiles(component)
            if parsed_component is None:
                continue
            if Chem.MolToSmiles(parsed_component) not in kept:
                dropped.append(Chem.MolToSmiles(parsed_component))

    if dropped:
        metals = {
            "Fe", "Pt", "Pd", "Ru", "Rh", "Ir", "Os", "Co", "Ni", "Cu", "Zn", "Mn", "Cr",
            "Ti", "V", "Mo", "W", "Au", "Ag", "Hg", "Cd", "Pb", "Sn", "Al", "Gd", "Tc",
        }
        source_atoms = {
            atom.GetSymbol()
            for component in components
            if (m := Chem.MolFromSmiles(component)) is not None
            for atom in m.GetAtoms()
        }
        is_coordination = bool(source_atoms & metals)
        notes.append(
            f"input had {len(components)} components; kept the largest and dropped: "
            + ", ".join(dropped)
        )
        if is_coordination:
            notes.append(
                "WARNING: this looks like a metal coordination compound or organometallic. "
                "Keeping only the largest organic fragment makes every downstream property "
                "WRONG for this structure. Parallax's descriptors, rules, models and "
                "retrosynthesis are built for organic small molecules and do not apply here. "
                "Treat the computed values as invalid."
            )
        else:
            notes.append(
                "This is the usual, intended behaviour for a salt or co-crystal: properties "
                "are reported for the free acid/base, not the salt form. Molecular weight and "
                "solubility of the supplied salt will differ."
            )

    structure_components_dropped = dropped
    try:
        structure_inchi = inchi.MolToInchi(mol)
        structure_key = inchi.InchiToInchiKey(structure_inchi) if structure_inchi else None
    except Exception:
        structure_inchi, structure_key = None, None

    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    return Structure(
        input=query,
        smiles=canonical,
        mol=mol,
        name=name,
        inchi=structure_inchi,
        inchikey=structure_key,
        formula=CalcMolFormula(mol),
        identifiers=identifiers,
        citations=list(dict.fromkeys(citations)),
        notes=notes,
        components_dropped=structure_components_dropped,
    )


def svg(
    smiles_or_mol: str | Chem.Mol,
    *,
    width: int = 380,
    height: int = 300,
    legend: str = "",
    highlight_smarts: str | None = None,
) -> str:
    """Render a molecule as inline SVG (for reports and dashboards)."""
    mol = parse(smiles_or_mol) if isinstance(smiles_or_mol, str) else smiles_or_mol
    mol = Chem.Mol(mol)
    Chem.rdDepictor.Compute2DCoords(mol)
    highlight_atoms: list[int] = []
    highlight_bonds: list[int] = []
    if highlight_smarts:
        pattern = Chem.MolFromSmarts(highlight_smarts)
        if pattern is not None:
            for match in mol.GetSubstructMatches(pattern):
                highlight_atoms.extend(match)
                for bond in mol.GetBonds():
                    if bond.GetBeginAtomIdx() in match and bond.GetEndAtomIdx() in match:
                        highlight_bonds.append(bond.GetIdx())
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.addStereoAnnotation = True
    options.clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        legend=legend,
        highlightAtoms=highlight_atoms or None,
        highlightBonds=highlight_bonds or None,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def grid_svg(
    smiles: list[str], legends: list[str] | None = None, *, per_row: int = 4, size: int = 240
) -> str:
    mols = []
    for smi in smiles:
        try:
            mols.append(parse(smi))
        except StructureError:
            continue
    if not mols:
        return "<svg xmlns='http://www.w3.org/2000/svg'></svg>"
    return Draw.MolsToGridImage(
        mols,
        molsPerRow=per_row,
        subImgSize=(size, size),
        legends=(legends or [""] * len(mols))[: len(mols)],
        useSVG=True,
    ).replace("xmlns:svg", "xmlns")
