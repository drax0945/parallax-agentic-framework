"""NCI/CADD Chemical Identifier Resolver — fallback name->structure service."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from ..net import SourceError, get_text
from ..provenance import Citation, registry

BASE = "https://cactus.nci.nih.gov/chemical/structure"


def resolve_name(name: str) -> dict[str, Any]:
    quoted = quote(name.strip(), safe="")
    try:
        smiles = get_text(f"{BASE}/{quoted}/smiles", source="nci-resolver").strip()
    except SourceError as exc:
        raise SourceError("nci-resolver", f"could not resolve {name!r}: {exc.message}") from exc
    if not smiles or "<" in smiles or len(smiles.split()) > 1:
        raise SourceError("nci-resolver", f"no structure returned for {name!r}")
    key = registry().add(
        Citation(
            key=f"nci:{quoted[:60]}",
            kind="database",
            title=f"NCI/CADD Chemical Identifier Resolver record for '{name}'",
            source_db="nci-resolver",
            url=f"{BASE}/{quoted}/smiles",
            note="structure resolved from a chemical name by the NCI CADD group service",
        )
    )
    return {"name": name, "smiles": smiles, "citations": [key]}
