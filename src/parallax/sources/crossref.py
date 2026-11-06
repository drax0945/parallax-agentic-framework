"""Crossref connector: authoritative bibliographic metadata and DOI checking."""

from __future__ import annotations

from typing import Any

from ..config import settings
from ..net import SourceError, get_json
from ..provenance import Citation, registry

BASE = "https://api.crossref.org/works"


def _cite(item: dict[str, Any]) -> str:
    doi = (item.get("DOI") or "").lower()
    authors = [
        f"{a.get('family', '')}, {a.get('given', '')}".strip(", ")
        for a in item.get("author", [])
        if a.get("family")
    ]
    issued = (item.get("issued", {}).get("date-parts") or [[None]])[0]
    year = issued[0] if issued else None
    return registry().add(
        Citation(
            key=f"doi:{doi}" if doi else f"crossref:{item.get('title', ['?'])[0][:40]}",
            kind="patent" if item.get("type") == "patent" else "journal",
            title=(item.get("title") or ["untitled"])[0],
            source_db="crossref",
            authors=authors[:20],
            year=year,
            venue=(item.get("container-title") or [None])[0],
            doi=doi or None,
            url=item.get("URL"),
            identifiers={
                "type": item.get("type"),
                "publisher": item.get("publisher"),
                "volume": item.get("volume"),
                "page": item.get("page"),
                "referenced_by_count": item.get("is-referenced-by-count"),
            },
            verified=True,  # came straight from Crossref, so the DOI exists
        )
    )


def search(
    query: str,
    *,
    limit: int = 20,
    year_from: int | None = None,
    kind: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "query.bibliographic": query,
        "rows": min(limit, 100),
        "select": "DOI,title,author,container-title,issued,type,publisher,volume,page,URL,abstract,is-referenced-by-count",
        "mailto": settings().contact_email,
    }
    filters = []
    if year_from:
        filters.append(f"from-pub-date:{year_from}-01-01")
    if kind:
        filters.append(f"type:{kind}")
    if filters:
        params["filter"] = ",".join(filters)

    payload = get_json(BASE, source="crossref", params=params)
    items = payload.get("message", {}).get("items", [])
    records = []
    for item in items:
        key = _cite(item)
        abstract = item.get("abstract") or ""
        import re

        abstract = re.sub(r"<[^>]+>", " ", abstract)
        records.append(
            {
                "citation": key,
                "title": (item.get("title") or ["untitled"])[0],
                "doi": item.get("DOI"),
                "year": ((item.get("issued", {}).get("date-parts") or [[None]])[0] or [None])[0],
                "venue": (item.get("container-title") or [None])[0],
                "type": item.get("type"),
                "cited_by": item.get("is-referenced-by-count"),
                "abstract": re.sub(r"\s+", " ", abstract).strip()[:1500] or None,
            }
        )
    return {
        "query": query,
        "total": payload.get("message", {}).get("total-results"),
        "results": records,
        "citations": [r["citation"] for r in records],
    }


def work(doi: str) -> dict[str, Any]:
    """Fetch one DOI's metadata; raises if the DOI does not exist."""
    clean = doi.lower().replace("https://doi.org/", "").replace("doi:", "").strip()
    payload = get_json(f"{BASE}/{clean}", source="crossref", params={"mailto": settings().contact_email})
    item = payload.get("message", {})
    key = _cite(item)
    citation = registry().get(key)
    return {
        "doi": clean,
        "citation": key,
        "reference": citation.reference() if citation else None,
        "title": (item.get("title") or ["untitled"])[0],
        "type": item.get("type"),
        "references_count": item.get("references-count"),
    }


def references_of(doi: str, *, limit: int = 60) -> dict[str, Any]:
    """The reference list of a paper — how to walk a synthesis lineage back."""
    clean = doi.lower().strip()
    try:
        payload = get_json(
            f"{BASE}/{clean}", source="crossref", params={"mailto": settings().contact_email}
        )
    except SourceError as exc:
        return {"doi": clean, "references": [], "error": exc.message}
    refs = payload.get("message", {}).get("reference", []) or []
    out = []
    for ref in refs[:limit]:
        ref_doi = (ref.get("DOI") or "").lower() or None
        key = registry().add(
            Citation(
                key=f"doi:{ref_doi}" if ref_doi else f"crossref:ref:{ref.get('key', '')[:40]}",
                kind="journal",
                title=ref.get("article-title") or ref.get("unstructured") or "cited reference",
                source_db="crossref/references",
                authors=[ref["author"]] if ref.get("author") else [],
                year=int(ref["year"]) if str(ref.get("year", "")).isdigit() else None,
                venue=ref.get("journal-title"),
                doi=ref_doi,
                note=f"cited by {clean}",
            )
        )
        out.append({"citation": key, "doi": ref_doi, "title": ref.get("article-title") or ref.get("unstructured")})
    return {"doi": clean, "count": len(out), "references": out, "citations": [r["citation"] for r in out]}
