"""The memory store.

Entry types
-----------
``finding``     something determined about a specific compound/target, with citations
``heuristic``   a generalisation across cases ("in this series, X tends to Y")
``sop``         a procedure that worked, worth repeating
``model``       a trained model and what its honest performance was
``failure``     something that did not work, and why — the most valuable type,
                because it stops the agent repeating a dead end
``preference``  how this user wants work done
``entity``      a durable fact about a target, assay or chemotype

Each entry carries a confidence that moves with evidence: corroboration raises
it, contradiction lowers it, and entries nobody uses decay. `consolidate()`
merges near-duplicates and rewrites the human index.

Retrieval is lexical (TF-IDF over title, body and tags) with boosts for tag
hits, confidence and recency. No embedding service, so recall is deterministic
and works offline.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..config import settings, utcnow

VALID_TYPES = {"finding", "heuristic", "sop", "model", "failure", "preference", "entity"}

SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str, limit: int = 48) -> str:
    slug = SLUG_RE.sub("-", text.lower()).strip("-")
    return slug[:limit] or "entry"


@dataclass
class MemoryEntry:
    id: str
    type: str
    title: str
    body: str
    tags: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    entities: dict[str, list[str]] = field(default_factory=dict)
    confidence: float = 0.6
    support: int = 1
    contradictions: int = 0
    uses: int = 0
    created: str = field(default_factory=utcnow)
    updated: str = field(default_factory=utcnow)
    superseded_by: str | None = None
    related: list[str] = field(default_factory=list)
    source_run: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def searchable(self) -> str:
        entity_text = " ".join(v for values in self.entities.values() for v in values)
        return f"{self.title} {self.title} {' '.join(self.tags)} {entity_text} {self.body}"

    def to_markdown(self) -> str:
        lines = ["---"]
        lines.append(f"id: {self.id}")
        lines.append(f"type: {self.type}")
        lines.append(f"title: {_quote(self.title)}")
        lines.append(f"tags: [{', '.join(self.tags)}]")
        lines.append(f"citations: [{', '.join(self.citations)}]")
        if self.entities:
            lines.append("entities:")
            for key, values in self.entities.items():
                lines.append(f"  {key}: [{', '.join(values)}]")
        lines.append(f"confidence: {self.confidence}")
        lines.append(f"support: {self.support}")
        lines.append(f"contradictions: {self.contradictions}")
        lines.append(f"uses: {self.uses}")
        lines.append(f"created: {self.created}")
        lines.append(f"updated: {self.updated}")
        if self.superseded_by:
            lines.append(f"superseded_by: {self.superseded_by}")
        if self.related:
            lines.append(f"related: [{', '.join(self.related)}]")
        if self.source_run:
            lines.append(f"source_run: {self.source_run}")
        lines.append("---")
        lines.append("")
        lines.append(f"# {self.title}")
        lines.append("")
        lines.append(self.body.strip())
        if self.citations:
            lines.append("")
            lines.append("**Evidence:** " + ", ".join(f"`{c}`" for c in self.citations))
        return "\n".join(lines) + "\n"

    @classmethod
    def from_markdown(cls, text: str) -> "MemoryEntry | None":
        if not text.startswith("---"):
            return None
        _, _, rest = text.partition("---")
        front, _, body = rest.partition("\n---")
        meta = _parse_frontmatter(front)
        if not meta.get("id"):
            return None
        body = re.sub(r"^\s*#.*\n", "", body.strip(), count=1).strip()
        body = re.sub(r"\n\*\*Evidence:\*\*.*$", "", body, flags=re.S).strip()
        return cls(
            id=str(meta["id"]),
            type=str(meta.get("type", "finding")),
            title=str(meta.get("title", "untitled")),
            body=body,
            tags=_as_list(meta.get("tags")),
            citations=_as_list(meta.get("citations")),
            entities={k: _as_list(v) for k, v in (meta.get("entities") or {}).items()},
            confidence=float(meta.get("confidence", 0.6) or 0.6),
            support=int(float(meta.get("support", 1) or 1)),
            contradictions=int(float(meta.get("contradictions", 0) or 0)),
            uses=int(float(meta.get("uses", 0) or 0)),
            created=str(meta.get("created", utcnow())),
            updated=str(meta.get("updated", utcnow())),
            superseded_by=meta.get("superseded_by") or None,
            related=_as_list(meta.get("related")),
            source_run=meta.get("source_run") or None,
        )


def _quote(text: str) -> str:
    return f'"{text}"' if any(c in text for c in ":#[]{}") else text


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip().strip("[]")
    return [part.strip() for part in text.split(",") if part.strip()]


def _parse_frontmatter(text: str) -> dict[str, Any]:
    """Minimal YAML-subset parser: scalars, inline lists, one nested level."""
    meta: dict[str, Any] = {}
    current_key: str | None = None
    for raw in text.splitlines():
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        indented = raw.startswith(("  ", "\t"))
        line = raw.strip()
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if indented and current_key:
            nested = meta.setdefault(current_key, {})
            if isinstance(nested, dict):
                nested[key] = _as_list(value)
            continue
        if not value:
            meta[key] = {}
            current_key = key
        else:
            meta[key] = value
            current_key = None
    return meta


class MemoryStore:
    def __init__(self, root: Path | None = None) -> None:
        cfg = settings()
        self.root = root or cfg.memory_dir
        self.entries_dir = self.root / "entries"
        self.index_path = self.root / "index.json"
        self.readme_path = self.root / "MEMORY.md"
        self.entries_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, MemoryEntry] | None = None

    # --- io ------------------------------------------------------------
    def _load(self) -> dict[str, MemoryEntry]:
        if self._cache is not None:
            return self._cache
        entries: dict[str, MemoryEntry] = {}
        for path in sorted(self.entries_dir.glob("*.md")):
            try:
                entry = MemoryEntry.from_markdown(path.read_text(encoding="utf-8"))
            except Exception:
                entry = None
            if entry:
                entries[entry.id] = entry
        self._cache = entries
        return entries

    def _path_for(self, entry: MemoryEntry) -> Path:
        return self.entries_dir / f"{entry.id}.md"

    def _write(self, entry: MemoryEntry) -> None:
        self._path_for(entry).write_text(entry.to_markdown(), encoding="utf-8")
        if self._cache is not None:
            self._cache[entry.id] = entry
        self._write_index()

    def _write_index(self) -> None:
        entries = self._load()
        index = {
            "updated": utcnow(),
            "count": len(entries),
            "entries": [
                {
                    "id": e.id,
                    "type": e.type,
                    "title": e.title,
                    "tags": e.tags,
                    "confidence": e.confidence,
                    "support": e.support,
                    "contradictions": e.contradictions,
                    "uses": e.uses,
                    "updated": e.updated,
                    "superseded_by": e.superseded_by,
                    "n_citations": len(e.citations),
                }
                for e in sorted(entries.values(), key=lambda x: x.updated, reverse=True)
            ],
        }
        self.index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
        self._write_readme()

    def _write_readme(self) -> None:
        entries = [e for e in self._load().values() if not e.superseded_by]
        by_type: dict[str, list[MemoryEntry]] = {}
        for entry in entries:
            by_type.setdefault(entry.type, []).append(entry)

        lines = [
            "# Parallax memory",
            "",
            "What this workspace has learned. One file per fact in `entries/`;",
            "this page is the index. Edit or delete anything that is wrong — the agent",
            "reads these files and will not re-derive what you have corrected.",
            "",
            f"_{len(entries)} active entries, updated {utcnow()}_",
            "",
        ]
        headings = {
            "heuristic": "Heuristics — generalisations to apply",
            "finding": "Findings — specific, cited results",
            "failure": "Failures — what not to repeat",
            "sop": "Procedures that worked",
            "model": "Models built here",
            "entity": "Targets, assays and chemotypes",
            "preference": "How this user wants work done",
        }
        for entry_type, heading in headings.items():
            group = sorted(
                by_type.get(entry_type, []),
                key=lambda e: (e.confidence, e.support),
                reverse=True,
            )
            if not group:
                continue
            lines.append(f"## {heading}")
            lines.append("")
            for entry in group:
                marks = f"confidence {entry.confidence:.2f}"
                if entry.support > 1:
                    marks += f", {entry.support}× corroborated"
                if entry.contradictions:
                    marks += f", {entry.contradictions}× contradicted"
                tags = f" `{' '.join(entry.tags[:5])}`" if entry.tags else ""
                lines.append(f"- [{entry.title}](entries/{entry.id}.md) — {marks}{tags}")
            lines.append("")
        self.readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- write operations ----------------------------------------------
    def remember(
        self,
        *,
        type: str,
        title: str,
        body: str,
        tags: Sequence[str] = (),
        citations: Sequence[str] = (),
        entities: dict[str, list[str]] | None = None,
        confidence: float = 0.6,
        related: Sequence[str] = (),
        source_run: str | None = None,
        merge_threshold: float = 0.72,
    ) -> dict[str, Any]:
        """Store a new memory, or corroborate an existing near-duplicate."""
        if type not in VALID_TYPES:
            raise ValueError(f"unknown memory type {type!r}; valid: {sorted(VALID_TYPES)}")

        existing = self._find_duplicate(title, body, type, threshold=merge_threshold)
        if existing:
            return self.reinforce(
                existing.id,
                note=f"corroborated: {title}",
                citations=citations,
            ) | {"merged_into": existing.id, "created_new": False}

        entry = MemoryEntry(
            id=f"{datetime.now(timezone.utc):%Y%m%d}-{_slug(title, 40)}-{uuid.uuid4().hex[:4]}",
            type=type,
            title=title.strip(),
            body=body.strip(),
            tags=[t.strip().lower() for t in tags if t.strip()],
            citations=list(dict.fromkeys(citations)),
            entities=entities or {},
            confidence=max(0.05, min(confidence, 0.99)),
            related=list(related),
            source_run=source_run,
        )
        self._write(entry)
        return {
            "id": entry.id,
            "created_new": True,
            "path": str(self._path_for(entry)),
            "title": entry.title,
            "type": entry.type,
            "confidence": entry.confidence,
        }

    def _find_duplicate(
        self, title: str, body: str, type: str, *, threshold: float
    ) -> MemoryEntry | None:
        candidates = [e for e in self._load().values() if e.type == type and not e.superseded_by]
        if not candidates:
            return None
        query_tokens = _tokens(f"{title} {body}")
        best: tuple[float, MemoryEntry] | None = None
        for entry in candidates:
            entry_tokens = _tokens(f"{entry.title} {entry.body}")
            if not entry_tokens or not query_tokens:
                continue
            overlap = len(query_tokens & entry_tokens) / len(query_tokens | entry_tokens)
            title_overlap = _jaccard(_tokens(title), _tokens(entry.title))
            score = max(overlap, title_overlap * 0.95)
            if best is None or score > best[0]:
                best = (score, entry)
        if best and best[0] >= threshold:
            return best[1]
        return None

    def reinforce(
        self, entry_id: str, *, note: str = "", citations: Sequence[str] = ()
    ) -> dict[str, Any]:
        entries = self._load()
        entry = entries.get(entry_id)
        if not entry:
            raise KeyError(f"no memory entry {entry_id!r}")
        entry.support += 1
        entry.confidence = min(0.99, entry.confidence + (1 - entry.confidence) * 0.35)
        entry.updated = utcnow()
        for citation in citations:
            if citation not in entry.citations:
                entry.citations.append(citation)
        if note:
            entry.body = entry.body.rstrip() + f"\n\n- {utcnow()[:10]}: {note}"
        self._write(entry)
        return {
            "id": entry.id,
            "confidence": round(entry.confidence, 3),
            "support": entry.support,
            "action": "reinforced",
        }

    def contradict(self, entry_id: str, *, note: str) -> dict[str, Any]:
        entries = self._load()
        entry = entries.get(entry_id)
        if not entry:
            raise KeyError(f"no memory entry {entry_id!r}")
        entry.contradictions += 1
        entry.confidence = max(0.05, entry.confidence * 0.5)
        entry.updated = utcnow()
        entry.body = entry.body.rstrip() + f"\n\n- {utcnow()[:10]} **contradicted**: {note}"
        self._write(entry)
        retired = entry.confidence < 0.15
        return {
            "id": entry.id,
            "confidence": round(entry.confidence, 3),
            "contradictions": entry.contradictions,
            "action": "contradicted",
            "below_usable_confidence": retired,
            "advice": (
                "Confidence is now low enough that this should be replaced by a corrected "
                "entry; write the new one and call supersede()."
                if retired
                else "Entry kept with reduced confidence."
            ),
        }

    def supersede(self, old_id: str, new_id: str) -> dict[str, Any]:
        entries = self._load()
        old, new = entries.get(old_id), entries.get(new_id)
        if not old or not new:
            raise KeyError("both old and new entries must exist")
        old.superseded_by = new_id
        old.updated = utcnow()
        new.related = list(dict.fromkeys(new.related + [old_id]))
        self._write(old)
        self._write(new)
        return {"superseded": old_id, "by": new_id}

    def forget(self, entry_id: str) -> dict[str, Any]:
        entries = self._load()
        entry = entries.get(entry_id)
        if not entry:
            raise KeyError(f"no memory entry {entry_id!r}")
        self._path_for(entry).unlink(missing_ok=True)
        entries.pop(entry_id, None)
        self._write_index()
        return {"forgotten": entry_id, "title": entry.title}

    # --- read operations -----------------------------------------------
    def recall(
        self,
        query: str,
        *,
        k: int = 6,
        types: Iterable[str] | None = None,
        tags: Iterable[str] | None = None,
        min_confidence: float = 0.15,
        record_use: bool = True,
    ) -> dict[str, Any]:
        entries = [
            e
            for e in self._load().values()
            if not e.superseded_by
            and e.confidence >= min_confidence
            and (not types or e.type in set(types))
            and (not tags or set(t.lower() for t in tags) & set(e.tags))
        ]
        if not entries:
            return {"query": query, "n_results": 0, "results": [], "store_size": len(self._load())}

        query_tokens = _tokens(query)
        tag_set = {t.lower() for t in (tags or [])}
        scored: list[tuple[float, MemoryEntry]] = []
        for entry in entries:
            entry_tokens = _tokens(entry.searchable())
            lexical = _jaccard(query_tokens, entry_tokens) if query_tokens else 0.0
            overlap_count = len(query_tokens & entry_tokens)
            coverage = overlap_count / max(len(query_tokens), 1)
            tag_bonus = 0.15 if tag_set & set(entry.tags) else 0.0
            recency = _recency_weight(entry.updated)
            score = (
                0.55 * coverage
                + 0.25 * lexical
                + tag_bonus
                + 0.1 * entry.confidence
                + 0.05 * recency
            )
            if score > 0.02:
                scored.append((score, entry))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = scored[:k]

        if record_use:
            for _, entry in top:
                entry.uses += 1
                self._path_for(entry).write_text(entry.to_markdown(), encoding="utf-8")

        return {
            "query": query,
            "n_results": len(top),
            "store_size": len(self._load()),
            "results": [
                {
                    "id": entry.id,
                    "type": entry.type,
                    "title": entry.title,
                    "body": entry.body,
                    "tags": entry.tags,
                    "confidence": round(entry.confidence, 2),
                    "support": entry.support,
                    "contradictions": entry.contradictions,
                    "citations": entry.citations,
                    "relevance": round(score, 3),
                    "updated": entry.updated,
                }
                for score, entry in top
            ],
        }

    def get(self, entry_id: str) -> dict[str, Any] | None:
        entry = self._load().get(entry_id)
        return entry.as_dict() if entry else None

    def all_entries(self) -> list[MemoryEntry]:
        return list(self._load().values())

    def stats(self) -> dict[str, Any]:
        entries = self._load()
        active = [e for e in entries.values() if not e.superseded_by]
        by_type: dict[str, int] = {}
        for entry in active:
            by_type[entry.type] = by_type.get(entry.type, 0) + 1
        return {
            "total_entries": len(entries),
            "active_entries": len(active),
            "superseded": len(entries) - len(active),
            "by_type": by_type,
            "mean_confidence": (
                round(sum(e.confidence for e in active) / len(active), 3) if active else None
            ),
            "contradicted": sum(1 for e in active if e.contradictions),
            "most_used": sorted(
                ({"id": e.id, "title": e.title, "uses": e.uses} for e in active),
                key=lambda x: x["uses"],
                reverse=True,
            )[:5],
            "index": str(self.readme_path),
        }

    # --- maintenance ----------------------------------------------------
    def consolidate(self, *, decay_after_days: int = 120, duplicate_threshold: float = 0.7) -> dict[str, Any]:
        """Merge near-duplicates, decay stale entries, rewrite the index."""
        entries = [e for e in self._load().values() if not e.superseded_by]
        merged: list[dict[str, str]] = []
        decayed: list[str] = []

        # Merge duplicates within the same type.
        by_type: dict[str, list[MemoryEntry]] = {}
        for entry in entries:
            by_type.setdefault(entry.type, []).append(entry)

        for group in by_type.values():
            group.sort(key=lambda e: (e.support, e.confidence), reverse=True)
            for index, keeper in enumerate(group):
                if keeper.superseded_by:
                    continue
                for candidate in group[index + 1 :]:
                    if candidate.superseded_by:
                        continue
                    overlap = _jaccard(
                        _tokens(f"{keeper.title} {keeper.body}"),
                        _tokens(f"{candidate.title} {candidate.body}"),
                    )
                    if overlap >= duplicate_threshold:
                        keeper.support += candidate.support
                        keeper.citations = list(
                            dict.fromkeys(keeper.citations + candidate.citations)
                        )
                        keeper.confidence = min(
                            0.99, max(keeper.confidence, candidate.confidence) + 0.05
                        )
                        keeper.updated = utcnow()
                        candidate.superseded_by = keeper.id
                        candidate.updated = utcnow()
                        self._write(candidate)
                        merged.append({"merged": candidate.id, "into": keeper.id})
                self._write(keeper)

        # Decay entries that nothing has used or confirmed in a long time.
        for entry in entries:
            if entry.superseded_by or entry.type in {"preference", "model"}:
                continue
            age_days = _age_days(entry.updated)
            if age_days > decay_after_days and entry.uses == 0 and entry.support <= 1:
                entry.confidence = max(0.05, entry.confidence * 0.8)
                entry.updated = utcnow()
                self._write(entry)
                decayed.append(entry.id)

        self._write_index()
        return {
            "merged": merged,
            "decayed": decayed,
            "active_entries": len([e for e in self._load().values() if not e.superseded_by]),
            "index_rewritten": str(self.readme_path),
        }


# --- helpers -------------------------------------------------------------

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "with", "is", "are", "was",
    "be", "by", "at", "as", "it", "this", "that", "from", "has", "have", "which", "but", "not",
}


def _tokens(text: str) -> set[str]:
    """Tokens for lexical matching.

    Hyphenated and slashed terms contribute both the joined form and their
    parts, so "electron-poor" and "electron poor" match each other — otherwise
    trivial punctuation differences defeat duplicate detection.
    """
    words = re.findall(r"[a-z0-9][a-z0-9\-\./]{1,}", (text or "").lower())
    tokens: set[str] = set()
    for word in words:
        stripped = word.strip("-./")
        if len(stripped) > 2 and stripped not in STOPWORDS:
            tokens.add(stripped)
        for part in re.split(r"[-/.]", stripped):
            if len(part) > 2 and part not in STOPWORDS:
                tokens.add(part)
    return tokens


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _age_days(timestamp: str) -> float:
    try:
        then = datetime.fromisoformat(timestamp)
    except ValueError:
        return 0.0
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - then).total_seconds() / 86400


def _recency_weight(timestamp: str) -> float:
    age = _age_days(timestamp)
    return max(0.0, 1.0 - age / 365)


# --- module-level convenience -------------------------------------------

_STORE: MemoryStore | None = None


def store() -> MemoryStore:
    global _STORE
    if _STORE is None:
        _STORE = MemoryStore()
    return _STORE


def remember(**kwargs: Any) -> dict[str, Any]:
    return store().remember(**kwargs)


def recall(query: str, **kwargs: Any) -> dict[str, Any]:
    return store().recall(query, **kwargs)


def consolidate(**kwargs: Any) -> dict[str, Any]:
    return store().consolidate(**kwargs)


def stats() -> dict[str, Any]:
    return store().stats()
