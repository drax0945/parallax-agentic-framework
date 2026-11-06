"""Patent search across free sources.

Patents are where most medicinal-chemistry SAR first appears, so route design
and analog work lean on them heavily. Three complementary free routes:

* Europe PMC `SRC:PAT` — EPO patent abstracts, full-text searchable;
* PubChem compound-patent cross-references — structure-anchored;
* PatentsView (US) — structured claims/assignee data, needs a free API key.
"""

from __future__ import annotations

from typing import Any

from ..config import settings
from ..net import SourceError, get_json
from ..provenance import Citation, registry
from . import europepmc, pubchem

PATENTSVIEW = "https://search.patentsview.org/api/v1/patent/"


COVERAGE_NOTE = (
    "Free-text patent search here runs against Europe PMC's patent index, which is a "
    "partial subset of EPO/WO filings — a query that returns nothing has NOT shown that no "
    "patent exists. For reliable coverage use `patents_for_compound` (structure-anchored "
    "through PubChem's compound-patent cross-references, which is far more complete), set "
    "PATENTSVIEW_API_KEY for full US grant search, or consult Espacenet/SciFinder directly."
)


def search(query: str, *, limit: int = 20, year_from: int | None = None) -> dict[str, Any]:
    """Free-text patent search.

    Uses PatentsView (full US grants) when a key is configured, otherwise Europe
    PMC's partial patent index. Either way the coverage limit is stated, because
    an empty patent search is routinely mistaken for 'nothing is patented'.
    """
    if settings().credential("PATENTSVIEW_API_KEY"):
        result = uspto(query, limit=limit, year_from=year_from)
        if result.get("available"):
            return {
                "query": query,
                "source": "patentsview (US grants, full text search)",
                "hit_count": result.get("total"),
                "results": result["results"],
                "citations": result["citations"],
                "coverage_note": "US grants only; use Espacenet for non-US filings.",
            }

    result = europepmc.search(query, limit=limit, patents_only=True, year_from=year_from)
    return {
        "query": query,
        "source": "europepmc:SRC:PAT (partial EPO/WO abstract index)",
        "hit_count": result.get("hit_count"),
        "results": result["results"],
        "citations": result["citations"],
        "coverage_note": COVERAGE_NOTE,
        "n_results": len(result["results"]),
    }


def for_compound(query: str, *, limit: int = 30) -> dict[str, Any]:
    """Patents linked to a specific structure, anchored via PubChem."""
    record = pubchem.lookup(query) if not query.isdigit() else {"cid": int(query)}
    cid = record["cid"]
    linked = pubchem.patents(cid, limit=limit)
    name = record.get("name") or query
    text_hits = search(f'"{name}"', limit=min(limit, 10))
    return {
        "compound": name,
        "cid": cid,
        "structure_linked_patents": linked["patents"],
        "structure_linked_total": linked.get("count_total"),
        "text_search_hits": text_hits["results"],
        "citations": sorted(set(linked["citations"] + text_hits["citations"])),
    }


def uspto(query: str, *, limit: int = 20, year_from: int | None = None) -> dict[str, Any]:
    """PatentsView search (US grants). Requires PATENTSVIEW_API_KEY."""
    api_key = settings().credential("PATENTSVIEW_API_KEY")
    if not api_key:
        return {
            "available": False,
            "reason": "PATENTSVIEW_API_KEY is not set",
            "setup": "Free key: https://patentsview.org/apis/keyrequest",
            "fallback": "patent_search (Europe PMC EPO index) covers most pharma filings",
            "results": [],
            "citations": [],
        }

    criteria: dict[str, Any] = {
        "_or": [
            {"_text_any": {"patent_title": query}},
            {"_text_any": {"patent_abstract": query}},
        ]
    }
    if year_from:
        criteria = {"_and": [criteria, {"_gte": {"patent_date": f"{year_from}-01-01"}}]}

    try:
        payload = get_json(
            PATENTSVIEW,
            source="patentsview",
            params={
                "q": __import__("json").dumps(criteria),
                "f": __import__("json").dumps(
                    ["patent_id", "patent_title", "patent_date", "patent_abstract", "assignees.assignee_organization"]
                ),
                "o": __import__("json").dumps({"size": min(limit, 100)}),
            },
            headers={"X-Api-Key": api_key},
        )
    except SourceError as exc:
        return {"available": False, "reason": exc.message, "results": [], "citations": []}

    records = []
    for patent in payload.get("patents", []) or []:
        patent_id = patent.get("patent_id")
        assignees = [
            a.get("assignee_organization")
            for a in patent.get("assignees", []) or []
            if a.get("assignee_organization")
        ]
        key = registry().add(
            Citation(
                key=f"patent:US{patent_id}",
                kind="patent",
                title=patent.get("patent_title") or f"US patent {patent_id}",
                source_db="patentsview",
                authors=assignees[:5],
                year=int((patent.get("patent_date") or "0")[:4]) or None,
                venue="US Patent and Trademark Office",
                url=f"https://patents.google.com/patent/US{patent_id}",
                identifiers={"patent_id": f"US{patent_id}", "assignees": assignees},
            )
        )
        records.append(
            {
                "citation": key,
                "patent_id": f"US{patent_id}",
                "title": patent.get("patent_title"),
                "date": patent.get("patent_date"),
                "assignees": assignees,
                "abstract": (patent.get("patent_abstract") or "")[:1200],
            }
        )
    return {
        "available": True,
        "query": query,
        "total": payload.get("total_hits"),
        "results": records,
        "citations": [r["citation"] for r in records],
    }
