"""Reaxys (and other licensed reaction databases) through a pluggable adapter.

Reaxys has no open endpoint: access is by institutional subscription, and the
exact API surface differs between deployments (hosted API key, on-site proxy,
or an internal mirror). Rather than guess at a protocol, Parallax defines the
adapter contract and routes to whatever the institution has:

1. **Custom adapter** — set ``REAXYS_ADAPTER`` to an import path
   (``mypkg.mymodule:client``) for an object implementing `ReaxysAdapter`.
   This is the supported path for most sites, since it lets your existing
   authenticated client do the talking.
2. **Generic HTTP** — set ``REAXYS_API_URL`` and ``REAXYS_API_KEY`` and
   Parallax will POST the query to that endpoint with the key attached, passing
   the response through with citations attached to each hit.
3. **Nothing configured** — every call returns ``available: false`` together
   with the public fallback that covers the same question. Parallax never
   invents Reaxys content, and the agent is instructed to say so plainly.

Adapter contract
----------------
```python
class ReaxysAdapter(Protocol):
    def reactions(self, *, product_smiles: str | None = None,
                  reactant_smiles: str | None = None,
                  query: str | None = None, limit: int = 20) -> list[dict]: ...
    def substances(self, *, smiles: str | None = None,
                   query: str | None = None, limit: int = 20) -> list[dict]: ...
```
Each returned dict should carry whatever of these it has: ``reaction_smiles``,
``product``, ``reactants``, ``reagents``, ``catalyst``, ``solvent``,
``temperature``, ``time``, ``yield_percent``, ``steps``, ``title``, ``authors``,
``year``, ``journal``, ``doi``, ``patent_number``, ``reaxys_id``, ``url``.
"""

from __future__ import annotations

import importlib
from typing import Any, Protocol, runtime_checkable

from ..config import settings
from ..net import SourceError, request
from ..provenance import Citation, registry

FALLBACK_NOTE = (
    "Public fallback in use: reaction and SAR evidence is assembled from ChEMBL source "
    "documents, Europe PMC (journals + EPO patent abstracts) and PubChem patent "
    "cross-references. Coverage of preparative conditions and yields is thinner than "
    "Reaxys; treat absence of a precedent as 'not found here', not 'not known'."
)


@runtime_checkable
class ReaxysAdapter(Protocol):
    def reactions(self, **kwargs: Any) -> list[dict[str, Any]]: ...
    def substances(self, **kwargs: Any) -> list[dict[str, Any]]: ...


def _load_adapter() -> Any | None:
    spec = settings().credential("REAXYS_ADAPTER")
    if not spec:
        return None
    module_path, _, attribute = spec.partition(":")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise SourceError("reaxys", f"could not import adapter {spec!r}: {exc}") from exc
    target = getattr(module, attribute or "client", None)
    if target is None:
        raise SourceError("reaxys", f"adapter {spec!r} has no attribute {attribute or 'client'}")
    return target() if isinstance(target, type) else target


def availability() -> dict[str, Any]:
    """Report how licensed-database access is configured."""
    cfg = settings()
    adapter = cfg.credential("REAXYS_ADAPTER")
    api = cfg.credential("REAXYS_API_URL")
    key = cfg.credential("REAXYS_API_KEY") or cfg.credential("ELSEVIER_API_KEY")
    mode = "adapter" if adapter else ("http" if (api and key) else "unavailable")
    return {
        "available": mode != "unavailable",
        "mode": mode,
        "adapter": adapter,
        "endpoint": api,
        "cas_common_chemistry": bool(cfg.credential("CAS_API_KEY")),
        "setup": {
            "option_1_adapter": "REAXYS_ADAPTER=yourpkg.reaxys_client:client  (implements ReaxysAdapter)",
            "option_2_http": "REAXYS_API_URL=<your endpoint>  REAXYS_API_KEY=<key>",
            "notes": "Reaxys requires an Elsevier subscription; check your institution's terms before automated querying.",
        },
        "fallback": FALLBACK_NOTE,
    }


def _cite_hit(hit: dict[str, Any], index: int) -> str:
    doi = (hit.get("doi") or "").lower() or None
    patent = hit.get("patent_number")
    reaxys_id = hit.get("reaxys_id")
    if doi:
        key = f"doi:{doi}"
    elif patent:
        key = f"patent:{patent}"
    elif reaxys_id:
        key = f"reaxys:{reaxys_id}"
    else:
        key = f"reaxys:hit{index}"
    authors = hit.get("authors") or []
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.split(",") if a.strip()]
    return registry().add(
        Citation(
            key=key,
            kind="patent" if patent and not doi else "journal",
            title=hit.get("title") or f"Reaxys record {reaxys_id or index}",
            source_db="reaxys",
            authors=authors[:20],
            year=hit.get("year"),
            venue=hit.get("journal") or ("patent" if patent else None),
            doi=doi,
            url=hit.get("url"),
            identifiers={
                "reaxys_id": reaxys_id,
                "patent_number": patent,
                "yield_percent": hit.get("yield_percent"),
            },
            note="retrieved from Reaxys via licensed access",
        )
    )


def _unavailable(question: str) -> dict[str, Any]:
    status = availability()
    return {
        "available": False,
        "question": question,
        "reason": "no licensed Reaxys access is configured in this workspace",
        "setup": status["setup"],
        "fallback": FALLBACK_NOTE,
        "use_instead": [
            "reaction_precedent_search (Europe PMC journals + EPO patents)",
            "chembl_activities (SAR with per-measurement DOIs)",
            "patents_for_compound (PubChem structure-linked patents)",
        ],
        "results": [],
        "citations": [],
    }


def _via_http(payload: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = settings()
    url = cfg.credential("REAXYS_API_URL")
    key = cfg.credential("REAXYS_API_KEY") or cfg.credential("ELSEVIER_API_KEY")
    response = request(
        url,
        method="POST",
        json_body=payload,
        headers={"X-API-Key": key, "Accept": "application/json"},
        source="reaxys",
        cache=True,
    )
    body = response.json_body
    if isinstance(body, dict):
        for field in ("results", "hits", "data", "reactions", "substances"):
            if isinstance(body.get(field), list):
                return body[field]
        return [body]
    return body if isinstance(body, list) else []


def reactions(
    *,
    product_smiles: str | None = None,
    reactant_smiles: str | None = None,
    query: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Preparative reaction precedents for a product (or free-text query)."""
    question = f"reactions product={product_smiles} reactant={reactant_smiles} query={query}"
    adapter = _load_adapter()
    hits: list[dict[str, Any]] | None = None

    if adapter is not None:
        hits = adapter.reactions(
            product_smiles=product_smiles,
            reactant_smiles=reactant_smiles,
            query=query,
            limit=limit,
        )
    elif availability()["mode"] == "http":
        hits = _via_http(
            {
                "type": "reaction",
                "product_smiles": product_smiles,
                "reactant_smiles": reactant_smiles,
                "query": query,
                "limit": limit,
            }
        )

    if hits is None:
        return _unavailable(question)

    records = []
    for index, hit in enumerate(hits[:limit]):
        key = _cite_hit(hit, index)
        records.append(
            {
                "citation": key,
                "reaction_smiles": hit.get("reaction_smiles"),
                "reagents": hit.get("reagents"),
                "catalyst": hit.get("catalyst"),
                "solvent": hit.get("solvent"),
                "temperature": hit.get("temperature"),
                "time": hit.get("time"),
                "yield_percent": hit.get("yield_percent"),
                "steps": hit.get("steps"),
                "reference": hit.get("title"),
                "year": hit.get("year"),
            }
        )
    return {
        "available": True,
        "mode": availability()["mode"],
        "query": question,
        "count": len(records),
        "results": records,
        "citations": [r["citation"] for r in records],
    }


def substances(*, smiles: str | None = None, query: str | None = None, limit: int = 20) -> dict[str, Any]:
    """Substance records (identity, properties, supplier/availability signals)."""
    question = f"substances smiles={smiles} query={query}"
    adapter = _load_adapter()
    hits: list[dict[str, Any]] | None = None
    if adapter is not None:
        hits = adapter.substances(smiles=smiles, query=query, limit=limit)
    elif availability()["mode"] == "http":
        hits = _via_http({"type": "substance", "smiles": smiles, "query": query, "limit": limit})
    if hits is None:
        return _unavailable(question)
    records = []
    for index, hit in enumerate(hits[:limit]):
        records.append({**hit, "citation": _cite_hit(hit, index)})
    return {
        "available": True,
        "mode": availability()["mode"],
        "count": len(records),
        "results": records,
        "citations": [r["citation"] for r in records],
    }


# --- CAS Common Chemistry (free tier, needs a key) -----------------------


def cas_lookup(query: str) -> dict[str, Any]:
    """CAS Common Chemistry record (authoritative CAS RN + name)."""
    key = settings().credential("CAS_API_KEY")
    if not key:
        return {
            "available": False,
            "reason": "CAS_API_KEY is not set",
            "setup": "Free key: https://www.cas.org/services/commonchemistry-api",
            "fallback": "PubChem synonyms include CAS numbers for most known compounds",
        }
    try:
        payload = request(
            "https://commonchemistry.cas.org/api/search",
            params={"q": query},
            headers={"X-API-KEY": key},
            source="cas",
        ).json()
    except SourceError as exc:
        return {"available": False, "reason": exc.message}
    results = payload.get("results", [])
    out = []
    for record in results[:10]:
        rn = record.get("rn")
        citation = registry().add(
            Citation(
                key=f"cas:{rn}",
                kind="database",
                title=f"CAS Common Chemistry record {rn} ({record.get('name')})",
                source_db="cas-common-chemistry",
                url=f"https://commonchemistry.cas.org/detail?cas_rn={rn}",
                identifiers={"cas_rn": rn},
            )
        )
        out.append({"cas_rn": rn, "name": record.get("name"), "smiles": record.get("smile"), "citation": citation})
    return {"available": True, "query": query, "results": out, "citations": [r["citation"] for r in out]}
