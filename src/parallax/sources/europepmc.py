"""Europe PMC connector: journal literature and patent abstracts.

Europe PMC indexes MEDLINE, PMC full text, preprints *and* EPO patent
abstracts (source `PAT`), which makes it the single most useful free endpoint
for medicinal-chemistry literature mining.
"""

from __future__ import annotations

from typing import Any

from ..net import get_json
from ..provenance import Citation, registry

BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"


def _cite(record: dict[str, Any]) -> str:
    doi = record.get("doi")
    pmid = record.get("pmid")
    pmcid = record.get("pmcid")
    source = record.get("source")
    identifier = record.get("id")
    is_patent = source == "PAT"

    if doi:
        key = f"doi:{doi.lower()}"
    elif is_patent:
        key = f"patent:{identifier}"
    elif pmid:
        key = f"pmid:{pmid}"
    else:
        key = f"epmc:{source}:{identifier}"

    authors = [a.strip() for a in (record.get("authorString") or "").rstrip(".").split(",") if a.strip()]
    year = record.get("pubYear")
    url = (
        f"https://doi.org/{doi}"
        if doi
        else (
            f"https://europepmc.org/article/{source}/{identifier}"
            if source and identifier
            else None
        )
    )
    return registry().add(
        Citation(
            key=key,
            kind="patent" if is_patent else "journal",
            title=(record.get("title") or "untitled").rstrip("."),
            source_db="europepmc",
            authors=authors[:20],
            year=int(year) if str(year).isdigit() else None,
            venue=record.get("journalTitle") or ("patent (EPO via Europe PMC)" if is_patent else None),
            doi=doi,
            url=url,
            identifiers={
                "pmid": pmid,
                "pmcid": pmcid,
                "europepmc_id": identifier,
                "source": source,
                "is_open_access": record.get("isOpenAccess"),
                "cited_by_count": record.get("citedByCount"),
            },
        )
    )


def search(
    query: str,
    *,
    limit: int = 20,
    include_patents: bool = False,
    patents_only: bool = False,
    sort: str | None = None,
    year_from: int | None = None,
) -> dict[str, Any]:
    """Search Europe PMC. Returns records with abstracts and citation keys."""
    full_query = query
    if patents_only:
        full_query = f"({query}) AND SRC:PAT"
    elif not include_patents:
        full_query = f"({query}) NOT SRC:PAT"
    if year_from:
        full_query += f" AND (FIRST_PDATE:[{year_from}-01-01 TO 3000-12-31])"

    params = {
        "query": full_query,
        "format": "json",
        "pageSize": min(limit, 100),
        "resultType": "core",
    }
    if sort:
        params["sort"] = sort

    payload = get_json(f"{BASE}/search", source="europepmc", params=params)
    results = payload.get("resultList", {}).get("result", [])
    records = []
    keys = []
    for record in results[:limit]:
        key = _cite(record)
        keys.append(key)
        abstract = (record.get("abstractText") or "").strip()
        records.append(
            {
                "citation": key,
                "title": (record.get("title") or "").rstrip("."),
                "authors": record.get("authorString"),
                "year": record.get("pubYear"),
                "venue": record.get("journalTitle"),
                "doi": record.get("doi"),
                "pmid": record.get("pmid"),
                "pmcid": record.get("pmcid"),
                "is_patent": record.get("source") == "PAT",
                "open_access": record.get("isOpenAccess") == "Y",
                "cited_by": record.get("citedByCount"),
                "abstract": abstract[:2500],
                "has_full_text": bool(record.get("pmcid")),
            }
        )
    return {
        "query": full_query,
        "hit_count": payload.get("hitCount"),
        "returned": len(records),
        "results": records,
        "citations": keys,
    }


def full_text(pmcid: str, *, max_chars: int = 60000) -> dict[str, Any]:
    """Retrieve open-access full text (XML) for close reading of a method."""
    pmcid = pmcid if pmcid.upper().startswith("PMC") else f"PMC{pmcid}"
    from ..net import get_text

    text = get_text(
        f"{BASE}/{pmcid}/fullTextXML",
        source="europepmc",
        accept="application/xml",
    )
    import re

    stripped = re.sub(r"<[^>]+>", " ", text)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    key = registry().add(
        Citation(
            key=f"pmcid:{pmcid}",
            kind="journal",
            title=f"Open-access full text {pmcid}",
            source_db="europepmc/fulltext",
            url=f"https://europepmc.org/article/PMC/{pmcid}",
            identifiers={"pmcid": pmcid},
        )
    )
    return {"pmcid": pmcid, "text": stripped[:max_chars], "citations": [key]}


def citing_works(doi_or_pmid: str, *, limit: int = 20) -> dict[str, Any]:
    """Who cited this work — useful for following a scaffold's lineage forward."""
    source, identifier = ("MED", doi_or_pmid)
    if "/" in doi_or_pmid:
        found = search(f'DOI:"{doi_or_pmid}"', limit=1)
        if not found["results"]:
            return {"query": doi_or_pmid, "results": [], "citations": []}
        identifier = found["results"][0].get("pmid") or ""
        if not identifier:
            return {"query": doi_or_pmid, "results": [], "citations": []}
    payload = get_json(
        f"{BASE}/{source}/{identifier}/citations",
        source="europepmc",
        params={"format": "json", "pageSize": min(limit, 100)},
    )
    items = payload.get("citationList", {}).get("citation", [])
    records = [
        {
            "title": item.get("title"),
            "year": item.get("pubYear"),
            "authors": item.get("authorString"),
            "pmid": item.get("id"),
            "citation": _cite({**item, "source": item.get("source", "MED"), "pmid": item.get("id")}),
        }
        for item in items[:limit]
    ]
    return {"query": doi_or_pmid, "results": records, "citations": [r["citation"] for r in records]}
