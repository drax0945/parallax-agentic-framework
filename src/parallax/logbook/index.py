"""A persistent, incrementally-maintained index over the logbook.

Why this exists
---------------
The logbook is plain markdown so a chemist can read and correct it, but a lab
with years of history has tens of thousands of entries, and re-reading every
file on every query is not viable. Measured on 25,000 drug-sized records, a
cold query cost 6.6 s: 4.3 s parsing markdown and 2.3 s rebuilding
fingerprints. The CLI and each `parallax ask` pay that from scratch.

The index removes both costs and keeps the markdown authoritative:

* **fingerprints are packed once** into a uint8 matrix and compared with a
  vectorised popcount Tanimoto — 25,000 comparisons in ~8 ms instead of 2.3 s;
* **tokens live in an inverted index**, so a text query touches only the
  entries that share a word rather than re-tokenising everything;
* **bodies are never loaded to search**, only for the handful of results
  actually returned, which keeps the resident footprint flat;
* **refresh is incremental** — files are compared by size and mtime, so only
  what a human actually edited is re-parsed.

The index is a cache and never the source of truth. Delete `.index/` and it
rebuilds from the markdown.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

FP_BITS = 2048
FP_BYTES = FP_BITS // 8
INDEX_VERSION = 1
_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)

# popcount lookup, so Tanimoto over the whole matrix is a couple of numpy ops
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def pack_fingerprint(smiles: str | None) -> bytes | None:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return DataStructs.BitVectToBinaryText(_generator.GetFingerprint(mol))


def bulk_tanimoto(matrix: np.ndarray, row_popcounts: np.ndarray, query: bytes) -> np.ndarray:
    """Tanimoto of one packed fingerprint against every row. Vectorised."""
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float32)
    q = np.frombuffer(query, dtype=np.uint8)
    intersection = _POPCOUNT[matrix & q].sum(axis=1).astype(np.float32)
    union = row_popcounts + int(_POPCOUNT[q].sum()) - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, intersection / union, 0.0)


@dataclass
class IndexedEntry:
    """Everything search needs, without touching the entry's body."""

    id: str
    date: str
    title: str
    outcome: str
    transform: str | None
    project: str | None
    product_smiles: str | None
    yield_percent: float | None


class LogbookIndex:
    def __init__(self, root: Path, entries_dir: Path) -> None:
        self.dir = root / ".index"
        self.entries_dir = entries_dir
        self.meta_path = self.dir / "meta.json"
        self.fp_path = self.dir / "fingerprints.npy"
        self.rows: list[dict[str, Any]] = []
        self.inverted: dict[str, list[int]] = {}
        self.matrix: np.ndarray = np.zeros((0, FP_BYTES), dtype=np.uint8)
        self.row_popcounts: np.ndarray = np.zeros(0, dtype=np.float32)
        self._loaded = False

    # --- persistence ----------------------------------------------------
    def _read(self) -> bool:
        if not (self.meta_path.exists() and self.fp_path.exists()):
            return False
        try:
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if meta.get("version") != INDEX_VERSION:
                return False
            matrix = np.load(self.fp_path)
        except Exception:
            return False
        if matrix.shape[0] != len(meta.get("rows", [])):
            return False
        self.rows = meta["rows"]
        self.inverted = meta.get("inverted", {})
        self.matrix = matrix
        self.row_popcounts = _POPCOUNT[matrix].sum(axis=1).astype(np.float32)
        return True

    def _write(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        np.save(self.fp_path, self.matrix)
        self.meta_path.write_text(
            json.dumps(
                {"version": INDEX_VERSION, "rows": self.rows, "inverted": self.inverted},
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )

    # --- build ----------------------------------------------------------
    def refresh(self, *, parse) -> dict[str, Any]:
        """Bring the index in line with the markdown on disk.

        `parse` turns a file's text into an `Experiment`; it is injected so this
        module stays free of the store's import cycle.
        """
        if not self._loaded:
            self._read()
            self._loaded = True

        on_disk: dict[str, os.stat_result] = {}
        if self.entries_dir.is_dir():
            with os.scandir(self.entries_dir) as scan:
                for item in scan:
                    if item.name.endswith(".md"):
                        on_disk[item.name] = item.stat()

        known = {row["file"]: position for position, row in enumerate(self.rows)}
        unchanged, reparsed, added, removed = 0, 0, 0, 0

        kept_rows: list[dict[str, Any]] = []
        kept_fps: list[np.ndarray] = []

        for name, stat in sorted(on_disk.items()):
            position = known.get(name)
            if position is not None:
                row = self.rows[position]
                if row.get("size") == stat.st_size and row.get("mtime") == stat.st_mtime_ns:
                    kept_rows.append(row)
                    kept_fps.append(self.matrix[position])
                    unchanged += 1
                    continue
                reparsed += 1
            else:
                added += 1

            text = (self.entries_dir / name).read_text(encoding="utf-8")
            entry = parse(text)
            if entry is None:
                continue
            packed = pack_fingerprint(entry.product_smiles)
            reactant_packed = [
                p for p in (pack_fingerprint(r) for r in entry.reactant_smiles) if p
            ]
            kept_rows.append(
                {
                    "file": name,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime_ns,
                    "id": entry.id,
                    "date": entry.date,
                    "title": entry.title,
                    "outcome": entry.outcome,
                    "transform": entry.transform,
                    "project": entry.project,
                    "product_smiles": entry.product_smiles,
                    "yield_percent": entry.yield_percent,
                    "tokens": sorted(_entry_tokens(entry)),
                    "reactant_fps": [p.hex() for p in reactant_packed[:4]],
                }
            )
            kept_fps.append(
                np.frombuffer(packed, dtype=np.uint8)
                if packed
                else np.zeros(FP_BYTES, dtype=np.uint8)
            )

        removed = len(self.rows) - unchanged - reparsed
        self.rows = kept_rows
        self.matrix = (
            np.vstack(kept_fps) if kept_fps else np.zeros((0, FP_BYTES), dtype=np.uint8)
        )
        self.row_popcounts = (
            _POPCOUNT[self.matrix].sum(axis=1).astype(np.float32)
            if self.matrix.size
            else np.zeros(0, dtype=np.float32)
        )
        self._rebuild_inverted()

        if reparsed or added or removed:
            self._write()
        return {
            "entries": len(self.rows),
            "unchanged": unchanged,
            "reparsed": reparsed,
            "added": added,
            "removed": max(removed, 0),
        }

    def _rebuild_inverted(self) -> None:
        inverted: dict[str, list[int]] = {}
        for position, row in enumerate(self.rows):
            for token in row.get("tokens", []):
                inverted.setdefault(token, []).append(position)
        self.inverted = inverted

    # --- query ----------------------------------------------------------
    def structural_scores(self, product_smiles: str | None) -> np.ndarray | None:
        packed = pack_fingerprint(product_smiles)
        if packed is None or not len(self.rows):
            return None
        return bulk_tanimoto(self.matrix, self.row_popcounts, packed)

    def reactant_scores(self, reactant_smiles: Sequence[str]) -> np.ndarray | None:
        """Best similarity between any query reactant and any recorded one."""
        queries = [p for p in (pack_fingerprint(r) for r in reactant_smiles) if p]
        if not queries or not len(self.rows):
            return None
        best = np.zeros(len(self.rows), dtype=np.float32)
        for position, row in enumerate(self.rows):
            stored = row.get("reactant_fps") or []
            if not stored:
                continue
            matrix = np.vstack([np.frombuffer(bytes.fromhex(h), dtype=np.uint8) for h in stored])
            popcounts = _POPCOUNT[matrix].sum(axis=1).astype(np.float32)
            best[position] = max(
                float(bulk_tanimoto(matrix, popcounts, q).max()) for q in queries
            )
        return best

    def token_overlap(self, tokens: set[str]) -> dict[int, float]:
        """Only entries sharing a word are touched, via the inverted index."""
        if not tokens:
            return {}
        hits: dict[int, int] = {}
        for token in tokens:
            for position in self.inverted.get(token, ()):
                hits[position] = hits.get(position, 0) + 1
        return {position: count / len(tokens) for position, count in hits.items()}

    def entry_at(self, position: int) -> IndexedEntry:
        row = self.rows[position]
        return IndexedEntry(
            id=row["id"], date=row["date"], title=row["title"], outcome=row["outcome"],
            transform=row.get("transform"), project=row.get("project"),
            product_smiles=row.get("product_smiles"),
            yield_percent=row.get("yield_percent"),
        )

    def __len__(self) -> int:
        return len(self.rows)

    def stats(self) -> dict[str, Any]:
        return {
            "indexed_entries": len(self.rows),
            "distinct_tokens": len(self.inverted),
            "fingerprint_matrix_kb": int(self.matrix.nbytes / 1024),
            "index_dir": str(self.dir),
        }


def _entry_tokens(entry: Any) -> set[str]:
    from ..memory.store import _tokens

    return _tokens(entry.searchable())
