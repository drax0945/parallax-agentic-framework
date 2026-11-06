"""ChEMBL connector: bioactivity with per-measurement provenance.

ChEMBL is the backbone of potency work here because every activity row carries
a `document_chembl_id` that resolves to a DOI or PubMed ID. That means a
measured IC50 — and any model trained on it — can be traced to the paper or
patent that reported it.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from ..config import settings
from ..net import SourceError, get_json
from ..provenance import Citation, registry

BASE = "https://www.ebi.ac.uk/chembl/api/data"
PAGE_MAX = 1000

CHEMBL_DB_CITATION = Citation(
    key="database:chembl",
    kind="database",
    title="The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods",
    source_db="chembl",
    authors=["Zdrazil, Barbara", "Felix, Eloy", "Hunter, Fiona", "Manners, Emma J.", "Blackshaw, James", "Corbett, Sybilla", "de Veij, Marleen", "Ioannidis, Harris", "Mendez Lopez, David", "Mosquera, Juan F.", "Magarinos, Maria Paula", "Bosc, Nicolas", "Arcila, Ricardo", "Kizilören, Tevfik", "Gaulton, Anna", "Bento, A. Patricia", "Adasme, Melissa F.", "Monecke, Peter", "Landrum, Gregory A.", "Leach, Andrew R."],
    venue="Nucleic Acids Research",
    year=2024,
    doi="10.1093/nar/gkad1004",
    url="https://www.ebi.ac.uk/chembl/",
)


def _db_citation() -> str:
    return registry().add(CHEMBL_DB_CITATION)


def _molecule_citation(chembl_id: str, name: str | None) -> str:
    _db_citation()
    return registry().add(
        Citation(
            key=f"chembl:{chembl_id}",
            kind="database",
            title=f"ChEMBL compound record {chembl_id}" + (f" ({name})" if name else ""),
            source_db="chembl",
            url=f"https://www.ebi.ac.uk/chembl/compound_report_card/{chembl_id}/",
            identifiers={"chembl_id": chembl_id},
        )
    )


def _flatten_molecule(record: dict[str, Any]) -> dict[str, Any]:
    structures = record.get("molecule_structures") or {}
    properties = record.get("molecule_properties") or {}
    chembl_id = record.get("molecule_chembl_id")
    name = record.get("pref_name")
    return {
        "chembl_id": chembl_id,
        "pref_name": name,
        "smiles": structures.get("canonical_smiles"),
        "inchikey": structures.get("standard_inchi_key"),
        "max_phase": record.get("max_phase"),
        "first_approval": record.get("first_approval"),
        "molecule_type": record.get("molecule_type"),
        "indication_class": record.get("indication_class"),
        "atc_codes": record.get("atc_classifications") or [],
        "oral": record.get("oral"),
        "black_box_warning": record.get("black_box_warning"),
        "chembl_properties": {
            "mw": properties.get("full_mwt"),
            "alogp": properties.get("alogp"),
            "cx_logp": properties.get("cx_logp"),
            "cx_logd": properties.get("cx_logd"),
            "cx_most_apka": properties.get("cx_most_apka"),
            "cx_most_bpka": properties.get("cx_most_bpka"),
            "psa": properties.get("psa"),
            "hbd": properties.get("hbd"),
            "hba": properties.get("hba"),
            "rtb": properties.get("rtb"),
            "aromatic_rings": properties.get("aromatic_rings"),
            "heavy_atoms": properties.get("heavy_atoms"),
            "qed_weighted": properties.get("qed_weighted"),
            "ro3_pass": properties.get("ro3_pass"),
            "num_ro5_violations": properties.get("num_ro5_violations"),
            "np_likeness_score": properties.get("np_likeness_score"),
        },
        "identifiers": {"chembl_id": chembl_id, "inchikey": structures.get("standard_inchi_key")},
        "citations": [_molecule_citation(chembl_id, name)] if chembl_id else [],
    }


def molecule(chembl_id: str) -> dict[str, Any]:
    payload = get_json(f"{BASE}/molecule/{chembl_id}?format=json", source="chembl")
    return _flatten_molecule(payload)


def search_molecule(query: str, *, limit: int = 10) -> dict[str, Any]:
    payload = get_json(
        f"{BASE}/molecule/search",
        source="chembl",
        params={"q": query, "format": "json", "limit": limit},
    )
    molecules = [_flatten_molecule(m) for m in payload.get("molecules", [])]
    return {"query": query, "results": molecules, "count": len(molecules)}


def search_target(query: str, *, limit: int = 10, organism: str | None = "Homo sapiens") -> dict[str, Any]:
    payload = get_json(
        f"{BASE}/target/search",
        source="chembl",
        params={"q": query, "format": "json", "limit": max(limit * 3, 20)},
    )
    results = []
    for target in payload.get("targets", []):
        if organism and target.get("organism") and target["organism"] != organism:
            continue
        chembl_id = target.get("target_chembl_id")
        citation = registry().add(
            Citation(
                key=f"chembl:{chembl_id}",
                kind="database",
                title=f"ChEMBL target record {chembl_id} ({target.get('pref_name')})",
                source_db="chembl",
                url=f"https://www.ebi.ac.uk/chembl/target_report_card/{chembl_id}/",
                identifiers={"chembl_id": chembl_id},
            )
        )
        results.append(
            {
                "target_chembl_id": chembl_id,
                "pref_name": target.get("pref_name"),
                "target_type": target.get("target_type"),
                "organism": target.get("organism"),
                "accession": next(
                    (
                        c.get("accession")
                        for c in (target.get("target_components") or [])
                        if c.get("accession")
                    ),
                    None,
                ),
                "citation": citation,
            }
        )
        if len(results) >= limit:
            break
    _db_citation()
    return {"query": query, "targets": results, "count": len(results)}


def documents(document_ids: Sequence[str]) -> dict[str, str]:
    """Resolve ChEMBL document IDs to citation keys (DOI where available)."""
    unique = [d for d in dict.fromkeys(document_ids) if d and d != "CHEMBL1158643"]
    mapping: dict[str, str] = {}
    chunk_size = 40
    for start in range(0, len(unique), chunk_size):
        chunk = unique[start : start + chunk_size]
        try:
            payload = get_json(
                f"{BASE}/document",
                source="chembl",
                params={
                    "document_chembl_id__in": ",".join(chunk),
                    "format": "json",
                    "limit": chunk_size,
                },
            )
            records = payload.get("documents", [])
        except SourceError:
            records = []
            for doc_id in chunk:
                try:
                    records.append(get_json(f"{BASE}/document/{doc_id}?format=json", source="chembl"))
                except SourceError:
                    continue
        for record in records:
            doc_id = record.get("document_chembl_id")
            if not doc_id:
                continue
            doi = record.get("doi")
            patent_id = record.get("patent_id")
            authors = [a.strip() for a in (record.get("authors") or "").split(",") if a.strip()]
            kind = "patent" if patent_id and not doi else "journal"
            key = f"doi:{doi.lower()}" if doi else (f"patent:{patent_id}" if patent_id else f"chembl:{doc_id}")
            mapping[doc_id] = registry().add(
                Citation(
                    key=key,
                    kind=kind,
                    title=record.get("title") or f"ChEMBL document {doc_id}",
                    source_db="chembl/document",
                    authors=authors[:20],
                    year=record.get("year"),
                    venue=record.get("journal_full_title") or record.get("journal") or ("patent" if patent_id else None),
                    doi=doi,
                    url=(
                        f"https://doi.org/{doi}"
                        if doi
                        else f"https://www.ebi.ac.uk/chembl/document_report_card/{doc_id}/"
                    ),
                    identifiers={
                        "chembl_document": doc_id,
                        "pubmed_id": record.get("pubmed_id"),
                        "patent_id": patent_id,
                        "volume": record.get("volume"),
                        "first_page": record.get("first_page"),
                    },
                    note="source document for ChEMBL bioactivity measurements",
                )
            )
    return mapping


def activities(
    *,
    target_chembl_id: str | None = None,
    molecule_chembl_id: str | None = None,
    standard_type: str | None = None,
    assay_type: str | None = None,
    pchembl_min: float | None = None,
    limit: int = 500,
    resolve_documents: bool = True,
) -> dict[str, Any]:
    """Fetch activity measurements with citations for each source document."""
    if not target_chembl_id and not molecule_chembl_id:
        raise ValueError("provide target_chembl_id or molecule_chembl_id")

    params: dict[str, Any] = {"format": "json", "limit": min(limit, PAGE_MAX)}
    if target_chembl_id:
        params["target_chembl_id"] = target_chembl_id
    if molecule_chembl_id:
        params["molecule_chembl_id"] = molecule_chembl_id
    if standard_type:
        params["standard_type"] = standard_type
    if assay_type:
        params["assay_type"] = assay_type
    if pchembl_min is not None:
        params["pchembl_value__gte"] = pchembl_min

    rows: list[dict[str, Any]] = []
    offset = 0
    total = None
    while len(rows) < limit:
        page_params = dict(params)
        page_params["offset"] = offset
        page_params["limit"] = min(PAGE_MAX, limit - len(rows))
        payload = get_json(f"{BASE}/activity", source="chembl", params=page_params)
        meta = payload.get("page_meta", {})
        total = meta.get("total_count", total)
        page = payload.get("activities", [])
        if not page:
            break
        for activity in page:
            rows.append(
                {
                    "molecule_chembl_id": activity.get("molecule_chembl_id"),
                    "molecule_name": activity.get("molecule_pref_name"),
                    "smiles": activity.get("canonical_smiles"),
                    "standard_type": activity.get("standard_type"),
                    "standard_relation": activity.get("standard_relation"),
                    "standard_value": _as_float(activity.get("standard_value")),
                    "standard_units": activity.get("standard_units"),
                    "pchembl_value": _as_float(activity.get("pchembl_value")),
                    "assay_chembl_id": activity.get("assay_chembl_id"),
                    "assay_description": activity.get("assay_description"),
                    "assay_type": activity.get("assay_type"),
                    "bao_label": activity.get("bao_label"),
                    "target_chembl_id": activity.get("target_chembl_id"),
                    "target_name": activity.get("target_pref_name"),
                    "document_chembl_id": activity.get("document_chembl_id"),
                    "document_year": activity.get("document_year"),
                    "document_journal": activity.get("document_journal"),
                    "data_validity_comment": activity.get("data_validity_comment"),
                    "activity_comment": activity.get("activity_comment"),
                }
            )
        offset += len(page)
        if not meta.get("next"):
            break

    citation_map: dict[str, str] = {}
    if resolve_documents and rows:
        citation_map = documents([r["document_chembl_id"] for r in rows if r.get("document_chembl_id")])
        for row in rows:
            row["citation"] = citation_map.get(row.get("document_chembl_id") or "")

    _db_citation()
    return {
        "query": {k: v for k, v in params.items() if k != "format"},
        "rows": rows,
        "count": len(rows),
        "total_available": total,
        "citations": sorted(set(citation_map.values())) + [CHEMBL_DB_CITATION.key],
    }


def mechanisms(chembl_id: str) -> dict[str, Any]:
    """Mechanism of action records, each with its supporting references."""
    payload = get_json(
        f"{BASE}/mechanism",
        source="chembl",
        params={"molecule_chembl_id": chembl_id, "format": "json", "limit": 50},
    )
    out = []
    keys: set[str] = set()
    for record in payload.get("mechanisms", []):
        refs = []
        for ref in record.get("mechanism_refs") or []:
            ref_type = (ref.get("ref_type") or "").upper()
            ref_id = ref.get("ref_id")
            url = ref.get("ref_url")
            key = registry().add(
                Citation(
                    key=(
                        f"pmid:{ref_id}"
                        if ref_type == "PUBMED"
                        else f"{ref_type.lower() or 'ref'}:{ref_id}"
                    ),
                    kind="journal" if ref_type in {"PUBMED", "DOI"} else "database",
                    title=f"{ref_type} {ref_id} — mechanism evidence for {chembl_id}",
                    source_db=f"chembl/mechanism/{ref_type.lower()}",
                    url=url,
                    identifiers={"pubmed_id": ref_id if ref_type == "PUBMED" else None},
                )
            )
            keys.add(key)
            refs.append(key)
        out.append(
            {
                "action_type": record.get("action_type"),
                "mechanism_of_action": record.get("mechanism_of_action"),
                "target_chembl_id": record.get("target_chembl_id"),
                "max_phase": record.get("max_phase"),
                "references": refs,
            }
        )
    return {"chembl_id": chembl_id, "mechanisms": out, "citations": sorted(keys)}


def similarity(smiles: str, *, threshold: int = 70, limit: int = 50) -> dict[str, Any]:
    from urllib.parse import quote

    payload = get_json(
        f"{BASE}/similarity/{quote(smiles, safe='')}/{threshold}",
        source="chembl",
        params={"format": "json", "limit": limit},
    )
    hits = []
    for record in payload.get("molecules", []):
        flat = _flatten_molecule(record)
        flat["similarity"] = _as_float(record.get("similarity"))
        hits.append(flat)
    return {"query": smiles, "threshold": threshold, "hits": hits, "count": len(hits)}


def training_set(
    target_chembl_id: str,
    *,
    standard_types: Iterable[str] = ("IC50", "Ki", "Kd", "EC50"),
    limit: int = 4000,
    min_pchembl: float | None = None,
) -> dict[str, Any]:
    """Assemble a modelling dataset for one target, keeping full provenance."""
    collected: list[dict[str, Any]] = []
    citation_keys: set[str] = set()
    per_type = max(limit // max(len(list(standard_types)), 1), 200)
    for standard_type in standard_types:
        try:
            result = activities(
                target_chembl_id=target_chembl_id,
                standard_type=standard_type,
                pchembl_min=min_pchembl,
                limit=per_type,
            )
        except SourceError:
            continue
        collected.extend(result["rows"])
        citation_keys.update(k for k in result["citations"] if k)

    clean: list[dict[str, Any]] = []
    for row in collected:
        if not row.get("smiles") or row.get("pchembl_value") is None:
            continue
        if row.get("standard_relation") not in (None, "=", "~"):
            continue
        if row.get("data_validity_comment"):
            continue
        clean.append(row)

    return {
        "target_chembl_id": target_chembl_id,
        "rows": clean,
        "count": len(clean),
        "standard_types": list(standard_types),
        "citations": sorted(citation_keys),
        "filters_applied": [
            "pchembl_value present",
            "standard_relation in {=, ~}",
            "no data_validity_comment",
        ],
    }


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
