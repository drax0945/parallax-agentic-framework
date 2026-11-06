"""PubChem PUG-REST connector.

PubChem is the structure-of-record for name resolution, experimental
physicochemical annotations, synonyms/CAS numbers, patent cross-references and
bioassay summaries.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from ..config import settings
from ..net import SourceError, get_json
from ..provenance import Citation, registry

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
VIEW = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data"

PROPERTIES = [
    "SMILES",
    "ConnectivitySMILES",
    "InChI",
    "InChIKey",
    "MolecularFormula",
    "MolecularWeight",
    "XLogP",
    "TPSA",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
    "HeavyAtomCount",
    "Complexity",
    "Charge",
    "Title",
]

PUBCHEM_DB_CITATION = Citation(
    key="database:pubchem",
    kind="database",
    title="PubChem 2025 update",
    source_db="pubchem",
    authors=["Kim, Sunghwan", "Chen, Jie", "Cheng, Tiejun", "Gindulyte, Asta", "He, Jia", "He, Siqian", "Li, Qingliang", "Shoemaker, Benjamin A.", "Thiessen, Paul A.", "Yu, Bo", "Zaslavsky, Leonid", "Zhang, Jian", "Bolton, Evan E."],
    venue="Nucleic Acids Research",
    year=2025,
    doi="10.1093/nar/gkae1059",
    url="https://pubchem.ncbi.nlm.nih.gov",
)


def _cite_compound(cid: int, title: str | None = None) -> str:
    registry().add(PUBCHEM_DB_CITATION)
    return registry().add(
        Citation(
            key=f"pubchem:CID{cid}",
            kind="database",
            title=f"PubChem Compound Summary for CID {cid}" + (f" — {title}" if title else ""),
            source_db="pubchem",
            url=f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
            identifiers={"pubchem_cid": cid},
            note="structure and annotated properties retrieved from PubChem PUG-REST",
        )
    )


def _path_for(query: str, kind: str) -> str:
    quoted = quote(str(query).strip(), safe="")
    if kind == "cid":
        return f"{BASE}/compound/cid/{quoted.lstrip('CIDcid: ')}"
    if kind == "inchikey":
        return f"{BASE}/compound/inchikey/{quoted}"
    if kind == "smiles":
        return f"{BASE}/compound/smiles/{quoted}"
    return f"{BASE}/compound/name/{quoted}"


def lookup(query: str, *, kind: str = "name") -> dict[str, Any]:
    """Resolve an identifier to a structure plus PubChem's annotated properties."""
    path = _path_for(query, "name" if kind == "cas" else kind)
    payload = get_json(
        f"{path}/property/{','.join(PROPERTIES)}/JSON",
        source="pubchem",
    )
    rows = payload.get("PropertyTable", {}).get("Properties", [])
    if not rows:
        raise SourceError("pubchem", f"no compound found for {query!r}")
    row = rows[0]
    cid = int(row["CID"])
    smiles = row.get("SMILES") or row.get("ConnectivitySMILES")
    if not smiles:
        raise SourceError("pubchem", f"record {cid} has no SMILES")

    citation = _cite_compound(cid, row.get("Title"))
    numeric = {
        "molecular_weight": _as_float(row.get("MolecularWeight")),
        "xlogp_experimental_model": _as_float(row.get("XLogP")),
        "tpsa": _as_float(row.get("TPSA")),
        "hbd": row.get("HBondDonorCount"),
        "hba": row.get("HBondAcceptorCount"),
        "rotatable_bonds": row.get("RotatableBondCount"),
        "heavy_atoms": row.get("HeavyAtomCount"),
        "complexity": row.get("Complexity"),
        "formal_charge": row.get("Charge"),
    }
    return {
        "cid": cid,
        "name": row.get("Title"),
        "smiles": smiles,
        "connectivity_smiles": row.get("ConnectivitySMILES"),
        "inchi": row.get("InChI"),
        "inchikey": row.get("InChIKey"),
        "formula": row.get("MolecularFormula"),
        "pubchem_properties": {k: v for k, v in numeric.items() if v is not None},
        "identifiers": {"pubchem_cid": cid, "inchikey": row.get("InChIKey")},
        "citations": [citation],
        "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
    }


def synonyms(cid: int, limit: int = 25) -> dict[str, Any]:
    payload = get_json(f"{BASE}/compound/cid/{cid}/synonyms/JSON", source="pubchem")
    names = payload.get("InformationList", {}).get("Information", [{}])[0].get("Synonym", [])
    cas = [n for n in names if n.count("-") == 2 and n.replace("-", "").isdigit()]
    return {
        "cid": cid,
        "synonyms": names[:limit],
        "cas_numbers": cas[:10],
        "citations": [_cite_compound(cid)],
    }


def patents(cid: int, limit: int = 40) -> dict[str, Any]:
    """Patent documents PubChem links to a compound."""
    try:
        payload = get_json(f"{BASE}/compound/cid/{cid}/xrefs/PatentID/JSON", source="pubchem")
    except SourceError as exc:
        if exc.status == 404:
            return {"cid": cid, "patents": [], "citations": [], "note": "no linked patents"}
        raise
    ids = payload.get("InformationList", {}).get("Information", [{}])[0].get("PatentID", [])
    keys = []
    records = []
    for patent_id in ids[:limit]:
        key = registry().add(
            Citation(
                key=f"patent:{patent_id}",
                kind="patent",
                title=f"Patent {patent_id} (linked to PubChem CID {cid})",
                source_db="pubchem/patents",
                url=f"https://patents.google.com/patent/{patent_id.replace('-', '')}",
                identifiers={"patent_id": patent_id, "pubchem_cid": cid},
                note="compound-patent link asserted by PubChem cross-reference",
            )
        )
        keys.append(key)
        records.append({"patent_id": patent_id, "citation": key})
    return {
        "cid": cid,
        "count_total": len(ids),
        "patents": records,
        "citations": keys[:limit],
    }


def assay_summary(cid: int, limit: int | None = None) -> dict[str, Any]:
    """Bioassay results PubChem holds for a compound (AID-level provenance)."""
    limit = limit or settings().max_tool_rows
    try:
        payload = get_json(f"{BASE}/compound/cid/{cid}/assaysummary/JSON", source="pubchem")
    except SourceError as exc:
        if exc.status == 404:
            return {"cid": cid, "rows": [], "citations": [], "note": "no assay summary"}
        raise
    table = payload.get("Table", {})
    columns = table.get("Columns", {}).get("Column", [])
    rows = []
    keys = set()
    for raw in table.get("Row", [])[:limit]:
        cells = raw.get("Cell", [])
        record = dict(zip(columns, cells))
        aid = record.get("AID")
        if aid:
            key = registry().add(
                Citation(
                    key=f"pubchem:AID{aid}",
                    kind="database",
                    title=record.get("Assay Name") or f"PubChem BioAssay AID {aid}",
                    source_db="pubchem/bioassay",
                    url=f"https://pubchem.ncbi.nlm.nih.gov/bioassay/{aid}",
                    identifiers={"pubchem_aid": aid},
                    note=f"depositor: {record.get('Source') or 'unknown'}",
                )
            )
            keys.add(key)
            record["citation"] = key
        rows.append(record)
    return {"cid": cid, "rows": rows, "citations": sorted(keys)}


def similar(smiles: str, *, threshold: int = 90, limit: int = 50) -> dict[str, Any]:
    """2D similarity search (Tanimoto on PubChem fingerprints)."""
    quoted = quote(smiles, safe="")
    payload = get_json(
        f"{BASE}/compound/fastsimilarity_2d/smiles/{quoted}/property/SMILES,Title,MolecularWeight/JSON",
        source="pubchem",
        params={"Threshold": threshold, "MaxRecords": limit},
    )
    rows = payload.get("PropertyTable", {}).get("Properties", [])
    out = []
    for row in rows:
        cid = int(row["CID"])
        out.append(
            {
                "cid": cid,
                "name": row.get("Title"),
                "smiles": row.get("SMILES"),
                "molecular_weight": _as_float(row.get("MolecularWeight")),
                "citation": _cite_compound(cid, row.get("Title")),
            }
        )
    return {"query": smiles, "threshold": threshold, "hits": out, "count": len(out)}


def substructure(smarts_or_smiles: str, *, limit: int = 50) -> dict[str, Any]:
    quoted = quote(smarts_or_smiles, safe="")
    payload = get_json(
        f"{BASE}/compound/fastsubstructure/smiles/{quoted}/property/SMILES,Title/JSON",
        source="pubchem",
        params={"MaxRecords": limit},
    )
    rows = payload.get("PropertyTable", {}).get("Properties", [])
    return {
        "query": smarts_or_smiles,
        "hits": [
            {
                "cid": int(r["CID"]),
                "name": r.get("Title"),
                "smiles": r.get("SMILES"),
                "citation": _cite_compound(int(r["CID"]), r.get("Title")),
            }
            for r in rows
        ],
        "count": len(rows),
    }


def experimental_properties(cid: int) -> dict[str, Any]:
    """Pull PubChem's annotated experimental data with per-value references.

    This is where measured melting points, solubilities, logP values and the
    like live, each with the handbook or paper that reported them.
    """
    try:
        payload = get_json(f"{VIEW}/compound/{cid}/JSON", source="pubchem")
    except SourceError:
        return {"cid": cid, "properties": [], "citations": []}

    wanted = {
        "Experimental Properties",
        "Chemical and Physical Properties",
        "Solubility",
        "LogP",
        "Melting Point",
        "Boiling Point",
    }
    found: list[dict[str, Any]] = []
    keys: set[str] = set()
    reference_map: dict[int, dict[str, Any]] = {}

    record = payload.get("Record", {})
    for ref in record.get("Reference", []):
        reference_map[ref.get("ReferenceNumber")] = ref

    def walk(section: dict[str, Any], trail: list[str]) -> None:
        heading = section.get("TOCHeading", "")
        path = trail + [heading]
        for info in section.get("Information", []):
            value = info.get("Value", {})
            strings = [s.get("String", s) if isinstance(s, dict) else s for s in value.get("StringWithMarkup", [])]
            text = "; ".join(str(s) for s in strings if s)
            if not text and "Number" in value:
                text = ", ".join(str(n) for n in value["Number"])
                if value.get("Unit"):
                    text += f" {value['Unit']}"
            if not text:
                continue
            ref = reference_map.get(info.get("ReferenceNumber"), {})
            citation_key = None
            if ref:
                source_name = ref.get("SourceName") or "PubChem annotation"
                citation_key = registry().add(
                    Citation(
                        key=f"pubchem:ref{cid}:{info.get('ReferenceNumber')}",
                        kind="database",
                        title=ref.get("Name") or f"{source_name} record for CID {cid}",
                        source_db=f"pubchem/{source_name}",
                        url=ref.get("URL"),
                        identifiers={"pubchem_cid": cid},
                        note=f"annotation under {' > '.join(p for p in path if p)}",
                    )
                )
                keys.add(citation_key)
            found.append(
                {
                    "section": " > ".join(p for p in path if p),
                    "name": info.get("Name") or heading,
                    "value": text[:500],
                    "citation": citation_key,
                }
            )

        for child in section.get("Section", []):
            if trail or heading in wanted or any(w in heading for w in wanted) or not trail:
                walk(child, path)

    for section in record.get("Section", []):
        if section.get("TOCHeading") in wanted or "Properties" in section.get("TOCHeading", ""):
            walk(section, [])

    return {"cid": cid, "properties": found[:120], "citations": sorted(keys)}


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
