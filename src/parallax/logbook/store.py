"""Storage and retrieval for lab experiment records.

Storage is plain markdown with frontmatter under `logbook/entries/`, matching
the memory store: readable, editable, diffable, and no database to run.

Retrieval is **hybrid and structural-first**, which is the part that makes this
different from document search. Asking "have we run this reaction before?" is a
chemistry question, not a text question — "amide coupling failed" and "HATU
gave no conversion" share no words but describe the same event. So the ranking
is driven by product and reactant similarity, with the named transform and the
written notes as supporting signals. It is deterministic and needs no embedding
service, which keeps the whole system offline-capable.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from ..config import settings, today, utcnow
from ..memory.store import _tokens  # same tokeniser, so recall behaves alike

RDLogger.DisableLog("rdApp.*")

OUTCOMES = {"success", "partial", "failure", "inconclusive"}
_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str, limit: int = 44) -> str:
    return SLUG_RE.sub("-", (text or "").lower()).strip("-")[:limit] or "entry"


def _fingerprint(smiles: str | None):
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return _generator.GetFingerprint(mol) if mol is not None else None


def split_reaction(reaction_smiles: str) -> tuple[list[str], str | None]:
    """'A.B>>C' -> (['A','B'], 'C').

    Both standard forms are accepted: `reactants>>products`, and
    `reactants>agents>products` where reagents and catalysts sit in the middle.
    Splitting on '>' handles both, since 'A>>C' yields an empty middle field.
    """
    if not reaction_smiles or ">" not in reaction_smiles:
        return [], None
    parts = reaction_smiles.split(">")
    if len(parts) < 3:
        return [], None
    left, right = parts[0], parts[-1]
    reactants = [p for p in left.split(".") if p.strip()]
    products = [p for p in right.split(".") if p.strip()]
    main = max(products, key=lambda p: Chem.MolFromSmiles(p).GetNumHeavyAtoms()
               if Chem.MolFromSmiles(p) else 0) if products else None
    return reactants, main


@dataclass
class Experiment:
    id: str
    title: str
    outcome: str
    date: str = field(default_factory=today)
    operator: str | None = None
    transform: str | None = None
    reaction_smiles: str | None = None
    product_smiles: str | None = None
    reactant_smiles: list[str] = field(default_factory=list)
    yield_percent: float | None = None
    scale: str | None = None
    conditions: dict[str, Any] = field(default_factory=dict)
    observations: str = ""
    analysis: str | None = None
    project: str | None = None
    tags: list[str] = field(default_factory=list)
    links: dict[str, Any] = field(default_factory=dict)
    created: str = field(default_factory=utcnow)

    def searchable(self) -> str:
        condition_text = " ".join(f"{k} {v}" for k, v in self.conditions.items())
        return " ".join(
            [
                self.title, self.title, self.transform or "", condition_text,
                self.observations, self.project or "", " ".join(self.tags),
                self.outcome,
            ]
        )

    def one_line(self) -> str:
        bits = [self.date, self.outcome.upper()]
        if self.yield_percent is not None:
            bits.append(f"{self.yield_percent:g}%")
        if self.transform:
            bits.append(self.transform)
        return " · ".join(bits) + f" — {self.title}"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    # --- serialisation -------------------------------------------------
    def to_markdown(self) -> str:
        lines = ["---", f"id: {self.id}", f"date: {self.date}", f"outcome: {self.outcome}"]
        lines.append(f"title: {_quote(self.title)}")
        if self.operator:
            lines.append(f"operator: {_quote(self.operator)}")
        if self.transform:
            lines.append(f"transform: {_quote(self.transform)}")
        if self.reaction_smiles:
            lines.append(f"reaction_smiles: {self.reaction_smiles}")
        if self.product_smiles:
            lines.append(f"product_smiles: {self.product_smiles}")
        if self.reactant_smiles:
            lines.append(f"reactant_smiles: [{', '.join(self.reactant_smiles)}]")
        if self.yield_percent is not None:
            lines.append(f"yield_percent: {self.yield_percent}")
        if self.scale:
            lines.append(f"scale: {_quote(self.scale)}")
        if self.conditions:
            lines.append("conditions:")
            for key, value in self.conditions.items():
                lines.append(f"  {key}: {_quote(str(value))}")
        if self.analysis:
            lines.append(f"analysis: {_quote(self.analysis)}")
        if self.project:
            lines.append(f"project: {_quote(self.project)}")
        if self.tags:
            lines.append(f"tags: [{', '.join(self.tags)}]")
        if self.links:
            lines.append("links:")
            for key, value in self.links.items():
                lines.append(f"  {key}: {_quote(str(value))}")
        lines.append(f"created: {self.created}")
        lines.append("---")
        lines.append("")
        lines.append(f"# {self.title}")
        lines.append("")
        if self.observations.strip():
            lines.append(self.observations.strip())
        return "\n".join(lines) + "\n"

    @classmethod
    def from_markdown(cls, text: str) -> "Experiment | None":
        if not text.startswith("---"):
            return None
        _, _, rest = text.partition("---")
        front, _, body = rest.partition("\n---")
        meta = _parse_frontmatter(front)
        if not meta.get("id"):
            return None
        body = re.sub(r"^\s*#.*\n", "", body.strip(), count=1).strip()
        yield_value = meta.get("yield_percent")
        try:
            yield_percent = float(yield_value) if yield_value not in (None, "") else None
        except (TypeError, ValueError):
            yield_percent = None
        return cls(
            id=str(meta["id"]),
            title=str(meta.get("title", "untitled")),
            outcome=str(meta.get("outcome", "inconclusive")),
            date=str(meta.get("date", today())),
            operator=meta.get("operator") or None,
            transform=meta.get("transform") or None,
            reaction_smiles=meta.get("reaction_smiles") or None,
            product_smiles=meta.get("product_smiles") or None,
            reactant_smiles=_as_list(meta.get("reactant_smiles")),
            yield_percent=yield_percent,
            scale=meta.get("scale") or None,
            conditions={k: v for k, v in (meta.get("conditions") or {}).items()},
            observations=body,
            analysis=meta.get("analysis") or None,
            project=meta.get("project") or None,
            tags=_as_list(meta.get("tags")),
            links={k: v for k, v in (meta.get("links") or {}).items()},
            created=str(meta.get("created", utcnow())),
        )


def _quote(text: str) -> str:
    return f'"{text}"' if any(c in text for c in ":#[]{}") else text


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [p.strip() for p in str(value).strip().strip("[]").split(",") if p.strip()]


def _parse_frontmatter(text: str) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    current: str | None = None
    for raw in text.splitlines():
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        indented = raw.startswith(("  ", "\t"))
        line = raw.strip()
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if indented and current:
            nested = meta.setdefault(current, {})
            if isinstance(nested, dict):
                nested[key] = value
            continue
        if not value:
            meta[key] = {}
            current = key
        else:
            meta[key] = value
            current = None
    return meta


class LogbookStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or (settings().workspace / "logbook")
        self.entries_dir = self.root / "entries"
        self.index_path = self.root / "LOGBOOK.md"
        self.entries_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Experiment] | None = None
        self._fps: dict[str, Any] = {}
        self._index: Any = None

    # --- index ------------------------------------------------------------
    def index(self):
        """The search index, refreshed against what is actually on disk."""
        from .index import LogbookIndex

        if self._index is None:
            self._index = LogbookIndex(self.root, self.entries_dir)
        self._index.refresh(parse=Experiment.from_markdown)
        return self._index

    def load_entry(self, entry_id: str) -> Experiment | None:
        """Read one entry's full text. Search never needs this for the whole store."""
        if self._cache and entry_id in self._cache:
            return self._cache[entry_id]
        path = self.entries_dir / f"{entry_id}.md"
        if not path.exists():
            return None
        return Experiment.from_markdown(path.read_text(encoding="utf-8"))

    # --- io -------------------------------------------------------------
    def _load(self) -> dict[str, Experiment]:
        if self._cache is not None:
            return self._cache
        entries: dict[str, Experiment] = {}
        for path in sorted(self.entries_dir.glob("*.md")):
            try:
                entry = Experiment.from_markdown(path.read_text(encoding="utf-8"))
            except Exception:
                entry = None
            if entry:
                entries[entry.id] = entry
        self._cache = entries
        return entries

    def _write(self, entry: Experiment) -> None:
        (self.entries_dir / f"{entry.id}.md").write_text(entry.to_markdown(), encoding="utf-8")
        if self._cache is not None:
            self._cache[entry.id] = entry
        self._fps.pop(entry.id, None)
        self._index = None
        self._write_index()

    def _write_index(self) -> None:
        entries = sorted(self._load().values(), key=lambda e: e.date, reverse=True)
        counts: dict[str, int] = {}
        for entry in entries:
            counts[entry.outcome] = counts.get(entry.outcome, 0) + 1
        lines = [
            "# Lab logbook",
            "",
            "Experiments run in this lab — successes and failures both. These are primary",
            "records: the agent reads them and treats them as outranking published",
            "precedent, so keep them accurate and correct anything that is wrong.",
            "",
            f"_{len(entries)} entries · "
            + ", ".join(f"{n} {k}" for k, n in sorted(counts.items()))
            + f" · updated {utcnow()}_",
            "",
        ]
        by_project: dict[str, list[Experiment]] = {}
        for entry in entries:
            by_project.setdefault(entry.project or "unfiled", []).append(entry)
        for project, group in sorted(by_project.items()):
            lines.append(f"## {project}")
            lines.append("")
            for entry in group:
                lines.append(f"- [{entry.one_line()}](entries/{entry.id}.md)")
            lines.append("")
        self.index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- write ----------------------------------------------------------
    def add(
        self,
        *,
        title: str,
        outcome: str,
        reaction_smiles: str | None = None,
        product_smiles: str | None = None,
        reactant_smiles: Sequence[str] | None = None,
        transform: str | None = None,
        conditions: dict[str, Any] | None = None,
        yield_percent: float | None = None,
        observations: str = "",
        date: str | None = None,
        operator: str | None = None,
        scale: str | None = None,
        analysis: str | None = None,
        project: str | None = None,
        tags: Sequence[str] = (),
        links: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        outcome = (outcome or "").strip().lower()
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {sorted(OUTCOMES)}, got {outcome!r}")

        reactants = list(reactant_smiles or [])
        product = product_smiles
        if reaction_smiles and not (product and reactants):
            parsed_reactants, parsed_product = split_reaction(reaction_smiles)
            reactants = reactants or parsed_reactants
            product = product or parsed_product

        # Canonicalise so structural search matches regardless of how it was typed.
        def canonical(smiles: str | None) -> str | None:
            if not smiles:
                return None
            mol = Chem.MolFromSmiles(smiles)
            return Chem.MolToSmiles(mol) if mol is not None else smiles

        product = canonical(product)
        reactants = [canonical(r) or r for r in reactants]

        entry = Experiment(
            id=f"{(date or today()).replace('-', '')}-{_slug(title, 40)}-{uuid.uuid4().hex[:4]}",
            title=title.strip(),
            outcome=outcome,
            date=date or today(),
            operator=operator,
            transform=transform,
            reaction_smiles=reaction_smiles,
            product_smiles=product,
            reactant_smiles=reactants,
            yield_percent=yield_percent,
            scale=scale,
            conditions=conditions or {},
            observations=observations.strip(),
            analysis=analysis,
            project=project,
            tags=[t.strip().lower() for t in tags if t.strip()],
            links=links or {},
        )
        self._write(entry)
        return {
            "id": entry.id,
            "path": str(self.entries_dir / f"{entry.id}.md"),
            "summary": entry.one_line(),
            "product_smiles": entry.product_smiles,
            "note": "Recorded as an in-house observation. It now outranks literature "
            "precedent for this lab.",
        }

    def get(self, entry_id: str) -> Experiment | None:
        return self._load().get(entry_id)

    def all_entries(self) -> list[Experiment]:
        return list(self._load().values())

    # --- retrieval --------------------------------------------------------
    def _entry_fp(self, entry: Experiment):
        if entry.id not in self._fps:
            self._fps[entry.id] = {
                "product": _fingerprint(entry.product_smiles),
                "reactants": [fp for fp in (_fingerprint(r) for r in entry.reactant_smiles) if fp],
            }
        return self._fps[entry.id]

    def search(
        self,
        query: str | None = None,
        *,
        product_smiles: str | None = None,
        reactant_smiles: Sequence[str] | None = None,
        reaction_smiles: str | None = None,
        transform: str | None = None,
        outcome: str | None = None,
        project: str | None = None,
        k: int = 8,
        min_score: float = 0.05,
    ) -> dict[str, Any]:
        """Hybrid search: structure first, then transform name, then text.

        Runs against the persistent index, so nothing re-reads the markdown and
        nothing re-computes a fingerprint. Only the entries actually returned
        have their bodies loaded.
        """
        index = self.index()
        if not len(index):
            return {"n_results": 0, "results": [], "store_size": 0,
                    "note": "The logbook is empty. Add experiments with logbook_add."}

        if reaction_smiles and not product_smiles:
            parsed_reactants, parsed_product = split_reaction(reaction_smiles)
            product_smiles = product_smiles or parsed_product
            reactant_smiles = reactant_smiles or parsed_reactants

        structural = index.structural_scores(product_smiles)
        text_scores = index.token_overlap(_tokens(query or ""))
        transform_key = (transform or "").strip().lower()

        # Only consider rows that something actually matched, so a big logbook
        # costs no more than a small one for a narrow query.
        candidates: set[int] = set(text_scores)
        if structural is not None:
            candidates |= set(np.flatnonzero(structural > 0.3).tolist())
        if transform_key:
            candidates |= {
                position for position, row in enumerate(index.rows)
                if row.get("transform") and transform_key in row["transform"].lower()
            }
        if not candidates:
            candidates = set(range(len(index))) if not (query or product_smiles or transform_key) else set()

        reactant_scores = None
        if reactant_smiles and len(candidates) <= 4000:
            reactant_scores = index.reactant_scores(list(reactant_smiles))

        scored: list[tuple[float, dict[str, Any], int]] = []
        for position in candidates:
            row = index.rows[position]
            if outcome and row["outcome"] != outcome:
                continue
            if project and (row.get("project") or "").lower() != project.lower():
                continue

            why: dict[str, Any] = {}
            score = 0.0

            if structural is not None:
                similarity = float(structural[position])
                if similarity > 0.3:
                    score += 0.55 * similarity
                    why["product_similarity"] = round(similarity, 3)

            if reactant_scores is not None:
                best = float(reactant_scores[position])
                if best > 0.3:
                    score += 0.2 * best
                    why["reactant_similarity"] = round(best, 3)

            if transform_key and row.get("transform"):
                entry_transform = row["transform"].lower()
                if transform_key == entry_transform:
                    score += 0.3
                    why["transform"] = "exact match"
                elif transform_key in entry_transform or entry_transform in transform_key:
                    score += 0.18
                    why["transform"] = "related"

            overlap = text_scores.get(position, 0.0)
            if overlap:
                score += 0.3 * overlap
                why["text_overlap"] = round(overlap, 3)

            if score >= min_score:
                scored.append((score, why, position))

        scored.sort(key=lambda row: row[0], reverse=True)
        top = scored[:k]

        results = []
        for score, why, position in top:
            summary = index.entry_at(position)
            entry = self.load_entry(summary.id)  # body loaded only for a hit
            results.append(
                {
                    "id": summary.id,
                    "date": summary.date,
                    "title": summary.title,
                    "outcome": summary.outcome,
                    "yield_percent": summary.yield_percent,
                    "transform": summary.transform,
                    "reaction_smiles": entry.reaction_smiles if entry else None,
                    "product_smiles": summary.product_smiles,
                    "conditions": entry.conditions if entry else {},
                    "observations": (entry.observations[:600] if entry else ""),
                    "operator": entry.operator if entry else None,
                    "project": summary.project,
                    "relevance": round(score, 3),
                    "matched_on": why,
                }
            )

        return {
            "n_results": len(results),
            "store_size": len(index),
            "candidates_considered": len(candidates),
            "results": results,
            "how_to_read": (
                "In-house records, ranked by structural similarity first, then the named "
                "transform, then text. These are observations from this lab: they outrank "
                "published precedent here, but each is a single run — check the counts "
                "before treating one as a rule."
            ),
        }

    # --- aggregate --------------------------------------------------------
    def outcomes_for_transform(self, transform: str, *, min_runs: int = 1) -> dict[str, Any]:
        """How this lab actually fares at a given transform."""
        key = (transform or "").strip().lower()
        index = self.index()
        matching = [
            index.entry_at(position) for position, row in enumerate(index.rows)
            if row.get("transform") and (key == row["transform"].lower() or key in row["transform"].lower())
        ]
        if len(matching) < min_runs:
            return {
                "transform": transform,
                "n_runs": len(matching),
                "enough_data": False,
                "note": "Too few in-house runs to say anything; use the template prior.",
            }
        counts: dict[str, int] = {}
        for entry in matching:
            counts[entry.outcome] = counts.get(entry.outcome, 0) + 1
        successes = counts.get("success", 0)
        decided = successes + counts.get("failure", 0) + counts.get("partial", 0)
        yields = [e.yield_percent for e in matching if e.yield_percent is not None]
        return {
            "transform": transform,
            "n_runs": len(matching),
            "enough_data": True,
            "outcomes": counts,
            "in_house_success_rate": round(successes / decided, 2) if decided else None,
            "median_yield_percent": (
                round(sorted(yields)[len(yields) // 2], 1) if yields else None
            ),
            "failures": [
                {"id": e.id, "date": e.date, "title": e.title,
                 "observations": ((self.load_entry(e.id) or e).observations[:240]
                                  if hasattr(self.load_entry(e.id) or e, "observations") else "")}
                for e in matching if e.outcome == "failure"
            ][:6],
            "caveat": (
                f"Based on {len(matching)} run(s) in this lab. A handful of runs is a hint, "
                "not a rate — read the failures rather than the number."
            ),
        }

    def stats(self) -> dict[str, Any]:
        index = self.index()
        entries = [index.entry_at(i) for i in range(len(index))]
        counts: dict[str, int] = {}
        transforms: dict[str, int] = {}
        projects: dict[str, int] = {}
        for entry in entries:
            counts[entry.outcome] = counts.get(entry.outcome, 0) + 1
            if entry.transform:
                transforms[entry.transform] = transforms.get(entry.transform, 0) + 1
            projects[entry.project or "unfiled"] = projects.get(entry.project or "unfiled", 0) + 1
        yields = [e.yield_percent for e in entries if e.yield_percent is not None]
        return {
            "n_entries": len(entries),
            "by_outcome": counts,
            "failures_recorded": counts.get("failure", 0),
            "by_transform": dict(sorted(transforms.items(), key=lambda kv: -kv[1])[:12]),
            "by_project": projects,
            "with_structures": sum(1 for e in entries if e.product_smiles),
            "index": index.stats(),
            "median_yield_percent": round(sorted(yields)[len(yields) // 2], 1) if yields else None,
            "date_range": [min((e.date for e in entries), default=None),
                           max((e.date for e in entries), default=None)],
            "index": str(self.index_path),
            "note": (
                "Failures are the most valuable records here: the published literature "
                "almost never reports them."
            ),
        }

    def import_rows(self, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
        """Bulk ingest, e.g. from an ELN export."""
        added, skipped = [], []
        for row in rows:
            try:
                added.append(self.add(**row)["id"])
            except Exception as exc:
                skipped.append({"row": str(row)[:120], "error": str(exc)})
        return {"added": len(added), "ids": added[:50], "skipped": skipped[:20]}


_STORE: LogbookStore | None = None


def store() -> LogbookStore:
    global _STORE
    if _STORE is None:
        _STORE = LogbookStore()
    return _STORE


def add(**kwargs: Any) -> dict[str, Any]:
    return store().add(**kwargs)


def search(query: str | None = None, **kwargs: Any) -> dict[str, Any]:
    return store().search(query, **kwargs)


def outcomes_for_transform(transform: str, **kwargs: Any) -> dict[str, Any]:
    return store().outcomes_for_transform(transform, **kwargs)


def stats() -> dict[str, Any]:
    return store().stats()
