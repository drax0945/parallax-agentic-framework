"""Connectors to chemistry, bioactivity, literature and patent databases.

Every connector follows the same contract:

* it returns plain dicts/lists (JSON-serialisable, ready for a tool result);
* it registers a `Citation` for each record it returns and includes the
  citation keys in its payload;
* it raises `SourceError` rather than returning a guess.

Public sources (PubChem, ChEMBL, Europe PMC, Crossref, OpenAlex, NCI) need no
credentials. Licensed sources (Reaxys, CAS, SciFinder) are reached through
adapters that activate only when credentials are present and otherwise report
clearly that they are unavailable.
"""

from . import chembl, crossref, europepmc, nci, openalex, patents, pubchem, reaxys

__all__ = [
    "chembl",
    "crossref",
    "europepmc",
    "nci",
    "openalex",
    "patents",
    "pubchem",
    "reaxys",
]
