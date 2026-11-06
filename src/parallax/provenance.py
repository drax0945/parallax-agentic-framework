"""Citation and provenance tracking.

The rule this module enforces: **no number without a source**. Every tool in
Parallax returns citation keys alongside its payload, and those keys resolve to
records in a per-run `CitationRegistry`. A registry can be exported as
markdown, BibTeX or JSON, and any DOI can be checked against Crossref so a
reference that does not exist is caught before it reaches a report.

Citation kinds
--------------
``journal``   peer-reviewed article (has DOI/PMID)
``patent``    patent document
``database``  a database record (ChEMBL activity, PubChem compound)
``software``  a tool or toolkit that computed a value
``method``    a published method/algorithm (e.g. an SA score paper)
``model``     a model trained inside Parallax, itself backed by cited data
``dataset``   an assembled dataset (points to its member citations)
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

from .config import utcnow
from .net import SourceError, get_json

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:a-z0-9A-Z]+")


@dataclass
class Citation:
    key: str
    kind: str
    title: str
    source_db: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    url: str | None = None
    identifiers: dict[str, Any] = field(default_factory=dict)
    note: str | None = None
    retrieved: str = field(default_factory=utcnow)
    verified: bool | None = None

    def short(self) -> str:
        first = self.authors[0].split(",")[0].split()[-1] if self.authors else self.source_db
        if self.authors and len(self.authors) > 1:
            first += " et al."
        bits = [first]
        if self.venue:
            bits.append(self.venue)
        if self.year:
            bits.append(str(self.year))
        return ", ".join(bits)

    def reference(self) -> str:
        """One-line human reference string."""
        authors = ", ".join(self.authors[:4])
        if len(self.authors) > 4:
            authors += " et al."
        parts = [p for p in (authors or None, self.title, self.venue, str(self.year) if self.year else None) if p]
        line = ". ".join(parts)
        if self.doi:
            line += f". https://doi.org/{self.doi}"
        elif self.url:
            line += f". {self.url}"
        return line

    def bibtex(self) -> str:
        entry_type = {
            "journal": "article",
            "patent": "patent",
            "database": "misc",
            "software": "software",
            "method": "article",
            "model": "misc",
            "dataset": "misc",
        }.get(self.kind, "misc")
        cite_key = re.sub(r"[^A-Za-z0-9]+", "_", self.key).strip("_")
        lines = [f"@{entry_type}{{{cite_key},"]
        if self.authors:
            lines.append(f"  author = {{{' and '.join(self.authors)}}},")
        lines.append(f"  title = {{{self.title}}},")
        if self.venue:
            field_name = "journal" if entry_type == "article" else "howpublished"
            lines.append(f"  {field_name} = {{{self.venue}}},")
        if self.year:
            lines.append(f"  year = {{{self.year}}},")
        if self.doi:
            lines.append(f"  doi = {{{self.doi}}},")
        if self.url:
            lines.append(f"  url = {{{self.url}}},")
        lines.append(f"  note = {{retrieved {self.retrieved}; source {self.source_db}}}")
        lines.append("}")
        return "\n".join(lines)


class CitationRegistry:
    """Deduplicating store of citations used in one analysis."""

    def __init__(self) -> None:
        self._items: dict[str, Citation] = {}
        self._by_doi: dict[str, str] = {}

    def add(self, citation: Citation) -> str:
        if citation.doi:
            citation.doi = citation.doi.lower().replace("https://doi.org/", "").strip()
            existing = self._by_doi.get(citation.doi)
            if existing:
                # Merge any newly discovered metadata into the existing record.
                current = self._items[existing]
                for name, value in asdict(citation).items():
                    if value and not getattr(current, name, None):
                        setattr(current, name, value)
                return existing
            self._by_doi[citation.doi] = citation.key
        if citation.key in self._items:
            return citation.key
        self._items[citation.key] = citation
        return citation.key

    def add_many(self, citations: Iterable[Citation]) -> list[str]:
        return [self.add(c) for c in citations]

    def get(self, key: str) -> Citation | None:
        return self._items.get(key)

    def keys(self) -> list[str]:
        return list(self._items)

    def all(self) -> list[Citation]:
        return sorted(self._items.values(), key=lambda c: (-(c.year or 0), c.title[:40]))

    def subset(self, keys: Sequence[str]) -> list[Citation]:
        return [self._items[k] for k in keys if k in self._items]

    def __len__(self) -> int:
        return len(self._items)

    def to_markdown(self, keys: Sequence[str] | None = None) -> str:
        items = self.subset(keys) if keys else self.all()
        if not items:
            return "_No citations recorded._"
        lines = []
        for citation in items:
            flag = ""
            if citation.verified is True:
                flag = " ✓"
            elif citation.verified is False:
                flag = " ⚠ unverified DOI"
            lines.append(f"- `{citation.key}` — {citation.reference()}{flag}")
        return "\n".join(lines)

    def to_bibtex(self, keys: Sequence[str] | None = None) -> str:
        items = self.subset(keys) if keys else self.all()
        return "\n\n".join(c.bibtex() for c in items)

    def to_json(self) -> str:
        return json.dumps([asdict(c) for c in self.all()], indent=2)

    def save(self, directory: Any) -> None:
        from pathlib import Path

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "citations.json").write_text(self.to_json(), encoding="utf-8")
        (path / "citations.bib").write_text(self.to_bibtex(), encoding="utf-8")
        (path / "citations.md").write_text(self.to_markdown(), encoding="utf-8")


# A single process-wide registry keeps tool calls composable: a tool deep in a
# call chain can register a source and return only its key.
REGISTRY = CitationRegistry()


def registry() -> CitationRegistry:
    return REGISTRY


def reset_registry() -> CitationRegistry:
    global REGISTRY
    REGISTRY = CitationRegistry()
    return REGISTRY


# --- standing citations for the deterministic machinery ------------------

SOFTWARE_CITATIONS: dict[str, Citation] = {
    "software:rdkit": Citation(
        key="software:rdkit",
        kind="software",
        title="RDKit: Open-source cheminformatics",
        source_db="rdkit",
        authors=["Landrum, Greg"],
        url="https://www.rdkit.org",
        note="descriptor, fingerprint, reaction and scaffold computations",
    ),
    "method:sa_score": Citation(
        key="method:sa_score",
        kind="method",
        title="Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions",
        source_db="literature",
        authors=["Ertl, Peter", "Schuffenhauer, Ansgar"],
        venue="Journal of Cheminformatics",
        year=2009,
        doi="10.1186/1758-2946-1-8",
    ),
    "method:qed": Citation(
        key="method:qed",
        kind="method",
        title="Quantifying the chemical beauty of drugs",
        source_db="literature",
        authors=["Bickerton, G. Richard", "Paolini, Gaia V.", "Besnard, Jérémy", "Muresan, Sorel", "Hopkins, Andrew L."],
        venue="Nature Chemistry",
        year=2012,
        doi="10.1038/nchem.1243",
    ),
    "method:lipinski": Citation(
        key="method:lipinski",
        kind="method",
        title="Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings",
        source_db="literature",
        authors=["Lipinski, Christopher A.", "Lombardo, Franco", "Dominy, Beryl W.", "Feeney, Paul J."],
        venue="Advanced Drug Delivery Reviews",
        year=1997,
        doi="10.1016/s0169-409x(96)00423-1",
    ),
    "method:veber": Citation(
        key="method:veber",
        kind="method",
        title="Molecular properties that influence the oral bioavailability of drug candidates",
        source_db="literature",
        authors=["Veber, Daniel F.", "Johnson, Stephen R.", "Cheng, Hung-Yuan", "Smith, Brian R.", "Ward, Keith W.", "Kopple, Kenneth D."],
        venue="Journal of Medicinal Chemistry",
        year=2002,
        doi="10.1021/jm020017n",
    ),
    "method:ghose": Citation(
        key="method:ghose",
        kind="method",
        title="A knowledge-based approach in designing combinatorial or medicinal chemistry libraries for drug discovery",
        source_db="literature",
        authors=["Ghose, Arup K.", "Viswanadhan, Vellarkad N.", "Wendoloski, John J."],
        venue="Journal of Combinatorial Chemistry",
        year=1999,
        doi="10.1021/cc9800071",
    ),
    "method:egan": Citation(
        key="method:egan",
        kind="method",
        title="Prediction of drug absorption using multivariate statistics",
        source_db="literature",
        authors=["Egan, William J.", "Merz, Kenneth M.", "Baldwin, John J."],
        venue="Journal of Medicinal Chemistry",
        year=2000,
        doi="10.1021/jm000292e",
    ),
    "method:pfizer_375": Citation(
        key="method:pfizer_375",
        kind="method",
        title="Physiochemical drug properties associated with in vivo toxicological outcomes",
        source_db="literature",
        authors=["Hughes, Jason D.", "Blagg, Julian", "Price, David A.", "Bailey, Simon", "DeCrescenzo, Gary A.", "Devraj, Rajesh V.", "Ellsworth, Edmund", "Fobian, Yvette M.", "Gibbs, Michael E.", "Gilles, Richard W."],
        venue="Bioorganic & Medicinal Chemistry Letters",
        year=2008,
        doi="10.1016/j.bmcl.2008.07.071",
        note="the 'Pfizer 3/75' lipophilicity-polarity toxicity risk rule",
    ),
    "method:gsk_4400": Citation(
        key="method:gsk_4400",
        kind="method",
        title="Generation of a set of simple, interpretable ADMET rules of thumb",
        source_db="literature",
        authors=["Gleeson, M. Paul"],
        venue="Journal of Medicinal Chemistry",
        year=2008,
        doi="10.1021/jm701122q",
        note="the 'GSK 4/400' rule",
    ),
    "method:lead_like": Citation(
        key="method:lead_like",
        kind="method",
        title="The design of leadlike combinatorial libraries",
        source_db="literature",
        authors=["Teague, Simon J.", "Davis, Andrew M.", "Leeson, Paul D.", "Oprea, Tudor"],
        venue="Angewandte Chemie International Edition",
        year=1999,
        doi="10.1002/(sici)1521-3773(19991216)38:24<3743::aid-anie3743>3.0.co;2-u",
    ),
    "method:cns_mpo": Citation(
        key="method:cns_mpo",
        kind="method",
        title="Moving beyond rules: the development of a central nervous system multiparameter optimization (CNS MPO) approach to enable alignment of druglike properties",
        source_db="literature",
        authors=["Wager, Travis T.", "Hou, Xinjun", "Verhoest, Patrick R.", "Villalobos, Anabella"],
        venue="ACS Chemical Neuroscience",
        year=2010,
        doi="10.1021/cn100008c",
    ),
    "method:ligand_efficiency": Citation(
        key="method:ligand_efficiency",
        kind="method",
        title="Ligand efficiency: a useful metric for lead selection",
        source_db="literature",
        authors=["Hopkins, Andrew L.", "Groom, Colin R.", "Alex, Alexander"],
        venue="Drug Discovery Today",
        year=2004,
        doi="10.1016/s1359-6446(04)03069-7",
    ),
    "method:lle": Citation(
        key="method:lle",
        kind="method",
        title="The influence of drug-like concepts on decision-making in medicinal chemistry",
        source_db="literature",
        authors=["Leeson, Paul D.", "Springthorpe, Brian"],
        venue="Nature Reviews Drug Discovery",
        year=2007,
        doi="10.1038/nrd2445",
        note="lipophilic ligand efficiency (LLE)",
    ),
    "method:retrosynthesis_review": Citation(
        key="method:retrosynthesis_review",
        kind="method",
        title="Computer-assisted synthesis planning",
        source_db="literature",
        authors=["Szymkuć, Sara", "Gajewska, Ewa P.", "Klucznik, Tomasz", "Molga, Karol", "Dittwald, Piotr", "Startek, Michał", "Bajczyk, Michał", "Grzybowski, Bartosz A."],
        venue="Angewandte Chemie International Edition",
        year=2016,
        doi="10.1002/anie.201506101",
        note="context for template-based retrosynthetic analysis and its limits",
    ),
    "method:morgan_fp": Citation(
        key="method:morgan_fp",
        kind="method",
        title="Extended-connectivity fingerprints",
        source_db="literature",
        authors=["Rogers, David", "Hahn, Mathew"],
        venue="Journal of Chemical Information and Modeling",
        year=2010,
        doi="10.1021/ci100050t",
    ),
    "method:murcko": Citation(
        key="method:murcko",
        kind="method",
        title="The properties of known drugs. 1. Molecular frameworks",
        source_db="literature",
        authors=["Bemis, Guy W.", "Murcko, Mark A."],
        venue="Journal of Medicinal Chemistry",
        year=1996,
        doi="10.1021/jm9602928",
    ),
    "method:mmpa": Citation(
        key="method:mmpa",
        kind="method",
        title="Matched molecular pairs as a medicinal chemistry tool",
        source_db="literature",
        authors=["Griffen, Ed", "Leach, Andrew G.", "Robb, Graeme R.", "Warner, Daniel J."],
        venue="Journal of Medicinal Chemistry",
        year=2011,
        doi="10.1021/jm200452d",
    ),
    "method:scaffold_split": Citation(
        key="method:scaffold_split",
        kind="method",
        title="MoleculeNet: a benchmark for molecular machine learning",
        source_db="literature",
        authors=["Wu, Zhenqin", "Ramsundar, Bharath", "Feinberg, Evan N.", "Gomes, Joseph", "Geniesse, Caleb", "Pappu, Aneesh S.", "Leswing, Karl", "Pande, Vijay"],
        venue="Chemical Science",
        year=2018,
        doi="10.1039/c7sc02664a",
        note="basis for scaffold-split out-of-distribution evaluation",
    ),
    "method:applicability_domain": Citation(
        key="method:applicability_domain",
        kind="method",
        title="Applicability domain for QSAR models: where theory meets reality",
        source_db="literature",
        authors=["Sahigara, Faizan", "Mansouri, Kamel", "Ballabio, Davide", "Mauri, Andrea", "Consonni, Viviana", "Todeschini, Roberto"],
        venue="International Journal of Quantitative Structure-Property Relationships",
        year=2012,
        doi="10.4018/ijqspr.2016010101",
        note="nearest-neighbour similarity based applicability domain",
    ),
    "method:random_forest": Citation(
        key="method:random_forest",
        kind="method",
        title="Random forests",
        source_db="literature",
        authors=["Breiman, Leo"],
        venue="Machine Learning",
        year=2001,
        doi="10.1023/a:1010933404324",
    ),
    "software:scikit_learn": Citation(
        key="software:scikit_learn",
        kind="software",
        title="Scikit-learn: Machine learning in Python",
        source_db="scikit-learn",
        authors=["Pedregosa, Fabian", "Varoquaux, Gaël", "Gramfort, Alexandre", "Michel, Vincent", "Thirion, Bertrand"],
        venue="Journal of Machine Learning Research",
        year=2011,
        url="https://scikit-learn.org",
    ),
    "method:crippen_logp": Citation(
        key="method:crippen_logp",
        kind="method",
        title="Prediction of physicochemical parameters by atomic contributions",
        source_db="literature",
        authors=["Wildman, Scott A.", "Crippen, Gordon M."],
        venue="Journal of Chemical Information and Computer Sciences",
        year=1999,
        doi="10.1021/ci990307l",
    ),
    "method:tpsa": Citation(
        key="method:tpsa",
        kind="method",
        title="Fast calculation of molecular polar surface area as a sum of fragment-based contributions",
        source_db="literature",
        authors=["Ertl, Peter", "Rohde, Bernhard", "Selzer, Paul"],
        venue="Journal of Medicinal Chemistry",
        year=2000,
        doi="10.1021/jm000942e",
    ),
    "method:brenk_filter": Citation(
        key="method:brenk_filter",
        kind="method",
        title="Lessons learnt from assembling screening libraries for drug discovery for neglected diseases",
        source_db="literature",
        authors=["Brenk, Ruth", "Schipani, Alessandro", "James, Daniel", "Krasowski, Agata", "Gilbert, Ian Hugh", "Frearson, Julie", "Wyatt, Paul Graham"],
        venue="ChemMedChem",
        year=2008,
        doi="10.1002/cmdc.200700139",
    ),
    "method:pains": Citation(
        key="method:pains",
        kind="method",
        title="New substructure filters for removal of pan assay interference compounds (PAINS) from screening libraries and for their exclusion in bioassays",
        source_db="literature",
        authors=["Baell, Jonathan B.", "Holloway, Georgina A."],
        venue="Journal of Medicinal Chemistry",
        year=2010,
        doi="10.1021/jm901137j",
    ),
    "method:macrocycle_review": Citation(
        key="method:macrocycle_review",
        kind="method",
        title="Macrocycles in new drug discovery",
        source_db="literature",
        authors=["Marsault, Eric", "Peterson, Mark L."],
        venue="Journal of Medicinal Chemistry",
        year=2011,
        doi="10.1021/jm1012374",
    ),
}


def cite_software(*keys: str) -> list[str]:
    """Register standing software/method citations and return their keys."""
    registered = []
    for key in keys:
        citation = SOFTWARE_CITATIONS.get(key)
        if citation:
            # Copy so per-run retrieval stamps stay accurate.
            registered.append(REGISTRY.add(Citation(**{**asdict(citation), "retrieved": utcnow()})))
    return registered


# --- verification ---------------------------------------------------------


def verify_doi(doi: str) -> dict[str, Any]:
    """Check a DOI against Crossref. Guards against fabricated references."""
    clean = doi.lower().replace("https://doi.org/", "").replace("doi:", "").strip()
    if not DOI_RE.fullmatch(clean):
        return {"doi": doi, "exists": False, "reason": "not a syntactically valid DOI"}
    try:
        payload = get_json(
            f"https://api.crossref.org/works/{clean}",
            source="crossref",
            params={"mailto": None},
        )
    except SourceError as exc:
        if exc.status == 404:
            return {"doi": clean, "exists": False, "reason": "no Crossref record"}
        return {"doi": clean, "exists": None, "reason": exc.message}
    message = payload.get("message", {})
    titles = message.get("title") or ["(untitled)"]
    return {
        "doi": clean,
        "exists": True,
        "title": titles[0],
        "venue": (message.get("container-title") or [None])[0],
        "year": (message.get("issued", {}).get("date-parts") or [[None]])[0][0],
        "type": message.get("type"),
        "authors": [
            f"{a.get('family', '')}, {a.get('given', '')}".strip(", ")
            for a in message.get("author", [])[:10]
        ],
    }


CITATION_KEY_RE = re.compile(r"`?\b((?:doi|pmid|pmcid|patent|chembl|pubchem|cas|model|method|software|nci|reaxys|epmc|fda):[^\s`,\]\)]+)")


def audit_text(text: str, *, check_crossref: bool = True) -> dict[str, Any]:
    """Audit a draft for citations that no tool actually produced.

    `verify_registry` can only check what reached the registry. A model that
    types a DOI straight into its prose bypasses that entirely — which is the
    exact failure this whole module exists to prevent. This scans the finished
    text instead, and reports every DOI and citation key that is not backed by
    a tool result.

    A DOI can be real and still be unbacked: real-but-unbacked means the claim
    was not actually sourced from the data, which is its own problem.
    """
    registry_dois = {c.doi for c in REGISTRY.all() if c.doi}
    registry_keys = set(REGISTRY.keys())

    found_dois = {
        d.rstrip(".,;:)]}\\").lower() for d in DOI_RE.findall(text or "")
    }
    found_keys = {m.rstrip(".,;:)]}`") for m in CITATION_KEY_RE.findall(text or "")}
    # A `doi:10.x/y` key also asserts the bare DOI; don't double-report it.
    found_keys = {k for k in found_keys if not k.lower().startswith("doi:")}

    unbacked_dois: list[dict[str, Any]] = []
    for doi in sorted(found_dois - registry_dois):
        record: dict[str, Any] = {
            "doi": doi,
            "in_registry": False,
            "problem": "cited in the text but no tool returned it",
        }
        if check_crossref:
            resolved = verify_doi(doi)
            record["resolves_at_crossref"] = resolved.get("exists")
            record["crossref_title"] = resolved.get("title")
            record["severity"] = (
                "fabricated — this DOI does not resolve and must be removed"
                if resolved.get("exists") is not True
                else "unsourced — the DOI is real, but it came from background knowledge "
                "rather than from the data, so the claim it supports is not grounded"
            )
        unbacked_dois.append(record)

    unknown_keys = sorted(found_keys - registry_keys)

    return {
        "dois_in_text": len(found_dois),
        "citation_keys_in_text": len(found_keys),
        "backed_by_tool_results": len(found_dois & registry_dois) + len(found_keys & registry_keys),
        "unbacked_dois": unbacked_dois,
        "unknown_citation_keys": unknown_keys,
        "clean": not unbacked_dois and not unknown_keys,
        "verdict": (
            "Every citation in this text traces to a tool result."
            if not unbacked_dois and not unknown_keys
            else f"{len(unbacked_dois)} DOI(s) and {len(unknown_keys)} citation key(s) in this "
            "text are not backed by any tool result. Remove them, or replace them with a "
            "source you actually retrieved."
        ),
    }


def verify_registry(keys: Sequence[str] | None = None) -> dict[str, Any]:
    """Verify every DOI-bearing citation in the registry."""
    items = REGISTRY.subset(keys) if keys else REGISTRY.all()
    checked, bad = 0, []
    for citation in items:
        if not citation.doi:
            continue
        result = verify_doi(citation.doi)
        checked += 1
        citation.verified = result.get("exists") is True
        if result.get("exists") is not True:
            bad.append({"key": citation.key, "doi": citation.doi, "reason": result.get("reason")})
    return {"checked": checked, "failed": bad, "total_citations": len(items)}
