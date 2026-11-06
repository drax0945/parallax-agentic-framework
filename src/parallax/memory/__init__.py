"""Continual-learning memory: what the agent has learned, in plain markdown.

Memory is a directory of markdown files with frontmatter, plus an index. It is
deliberately human-readable and human-editable: a chemist can open
`memory/MEMORY.md`, disagree with something, and fix it. Nothing is hidden in a
vector store.
"""

from .store import (
    MemoryEntry,
    MemoryStore,
    consolidate,
    recall,
    remember,
    stats,
    store,
)

__all__ = [
    "MemoryEntry",
    "MemoryStore",
    "consolidate",
    "recall",
    "remember",
    "stats",
    "store",
]
