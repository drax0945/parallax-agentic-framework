"""Runtime configuration and filesystem layout for Parallax.

Everything the framework writes lives under the workspace root so a project is
self-contained and inspectable: memory is plain markdown, caches are plain
files, runs are directories you can open.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


def _find_workspace(start: Path | None = None) -> Path:
    """Walk up from cwd looking for a Parallax workspace marker."""
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "parallax").exists():
            return candidate
        if (candidate / ".parallax").is_dir():
            return candidate
    return here


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no dependency, never overrides real env vars)."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class Settings:
    workspace: Path
    contact_email: str
    model: str
    small_model: str
    cache_ttl_hours: int
    request_timeout: float
    max_tool_rows: int
    offline: bool
    extra: dict[str, str] = field(default_factory=dict)

    # --- derived paths -------------------------------------------------
    @property
    def state_dir(self) -> Path:
        return self.workspace / ".parallax"

    @property
    def cache_dir(self) -> Path:
        return self.state_dir / "cache"

    @property
    def runs_dir(self) -> Path:
        return self.state_dir / "runs"

    @property
    def models_dir(self) -> Path:
        return self.state_dir / "models"

    @property
    def audit_log(self) -> Path:
        return self.state_dir / "safety-audit.jsonl"

    @property
    def memory_dir(self) -> Path:
        return self.workspace / "memory"

    @property
    def memory_index_file(self) -> Path:
        return self.memory_dir / "MEMORY.md"

    @property
    def memory_entries_dir(self) -> Path:
        return self.memory_dir / "entries"

    @property
    def skills_dir(self) -> Path:
        candidates = [
            self.workspace / ".claude" / "skills",
            self.workspace / "skills",
        ]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        return candidates[0]

    @property
    def reports_dir(self) -> Path:
        return self.workspace / "reports"

    def ensure_dirs(self) -> None:
        for path in (
            self.state_dir,
            self.cache_dir,
            self.runs_dir,
            self.models_dir,
            self.memory_dir,
            self.memory_entries_dir,
            self.reports_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    # --- credentials ---------------------------------------------------
    def credential(self, name: str) -> str | None:
        value = os.environ.get(name)
        return value.strip() or None if value else None

    @property
    def has_reaxys(self) -> bool:
        return bool(self.credential("REAXYS_API_KEY") or self.credential("ELSEVIER_API_KEY"))

    @property
    def has_cas(self) -> bool:
        return bool(self.credential("CAS_API_KEY"))

    @property
    def has_anthropic_key(self) -> bool:
        return bool(self.credential("ANTHROPIC_API_KEY"))


@lru_cache(maxsize=1)
def settings() -> Settings:
    workspace = Path(os.environ.get("PARALLAX_WORKSPACE") or _find_workspace())
    _load_dotenv(workspace / ".env")
    return Settings(
        workspace=workspace,
        contact_email=os.environ.get("PARALLAX_CONTACT_EMAIL", "parallax-agent@example.org"),
        model=os.environ.get("PARALLAX_MODEL", "claude-opus-4-5-20251101"),
        small_model=os.environ.get("PARALLAX_SMALL_MODEL", "claude-haiku-4-5-20251001"),
        cache_ttl_hours=int(os.environ.get("PARALLAX_CACHE_TTL_HOURS", "168")),
        request_timeout=float(os.environ.get("PARALLAX_TIMEOUT", "40")),
        max_tool_rows=int(os.environ.get("PARALLAX_MAX_ROWS", "300")),
        offline=os.environ.get("PARALLAX_OFFLINE", "").lower() in {"1", "true", "yes"},
    )


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def today() -> str:
    return datetime.now(timezone.utc).date().isoformat()
