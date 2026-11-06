"""OpenAlex connector: broad scholarly search with citation graph context.

Anonymous access is rate-limited; set `OPENALEX_API_KEY` for reliable use. The
connector degrades to an explicit "unavailable" result rather than failing a
whole analysis, since Europe PMC and Crossref cover the same ground.
"""

from __future__ import annotations

from typing import Any

from ..config import settings
from ..net import SourceError, get_json
from ..provenance import Citation, registry

BASE = "https://api.openalex.org/works"


def _cite(work: dict[str, Any]) -> str:
    doi = (work.get("doi") or "").lower().replace("https://doi.org/", "") or None
    authors = [
        a.get("author", {}).get("display_name")
        for a in work.get("authorships", [])[:20]
        if a.get("author", {}).get("display_name")
    ]
    venue = (work.get("primary_location") or {}).get("source") or {}
    return registry().add(
        Citation(
            key=f"doi:{doi}" if doi else f"openalex:{(work.get('id') or '').rsplit('/', 1)[-1]}",
            kind="journal",
            title=work.get("display_name") or "untitled",
            source_db="openalex",
            authors=[a for a in authors if a],
            year=work.get("publication_year"),
            venue=venue.get("display_name"),
            doi=doi,
            url=work.get("doi") or work.get("id"),
            identifiers={
                "openalex_id": work.get("id"),
                "cited_by_count": work.get("cited_by_count"),
                "type": work.get("type"),
            },
        )
    )


def search(query: str, *, limit: int = 20, year_from: int | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {
        "search": query,
        "per-page": min(limit, 50),
        "mailto": settings().contact_email,
        "select": "id,doi,display_name,publication_year,cited_by_count,type,authorships,primary_location",
    }
    if year_from:
        params["filter"] = f"from_publication_date:{year_from}-01-01"
    api_key = settings().credential("OPENALEX_API_KEY")
    if api_key:
        params["api_key"] = api_key

    try:
        payload = get_json(BASE, source="openalex", params=params)
    except SourceError as exc:
        return {
            "query": query,
            "available": False,
            "reason": exc.message,
            "results": [],
            "citations": [],
            "advice": "OpenAlex throttles anonymous search; use europepmc_search or crossref_search instead, or set OPENALEX_API_KEY.",
        }

    results = payload.get("results", [])
    records = []
    for work in results:
        key = _cite(work)
        records.append(
            {
                "citation": key,
                "title": work.get("display_name"),
                "year": work.get("publication_year"),
                "cited_by": work.get("cited_by_count"),
                "doi": work.get("doi"),
                "type": work.get("type"),
            }
        )
    return {
        "query": query,
        "available": True,
        "total": payload.get("meta", {}).get("count"),
        "results": records,
        "citations": [r["citation"] for r in records],
    }
