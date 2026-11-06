"""Polite, cached HTTP access to public chemistry and literature APIs.

Design notes
------------
* Every response is cached on disk keyed by the full request, so re-running an
  analysis is fast and reproducible, and a cited record can be re-read exactly
  as it was retrieved.
* Per-host minimum intervals keep us inside the published fair-use limits of
  PubChem (<=5 req/s), ChEMBL, Europe PMC and Crossref.
* Failures are retried with backoff, then raised as `SourceError` so tools can
  degrade gracefully instead of inventing data.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

from .config import settings, utcnow

USER_AGENT_TEMPLATE = "Parallax-Cheminformatics/0.1 (+research agent; mailto:{email})"

# Minimum seconds between requests per host.
HOST_INTERVALS: dict[str, float] = {
    "pubchem.ncbi.nlm.nih.gov": 0.22,
    "www.ebi.ac.uk": 0.15,
    "api.crossref.org": 0.12,
    "api.openalex.org": 0.15,
    "api.semanticscholar.org": 0.4,
    "cactus.nci.nih.gov": 0.3,
    "api.elsevier.com": 0.5,
    "commonchemistry.cas.org": 0.5,
    "search.patentsview.org": 0.5,
}
DEFAULT_INTERVAL = 0.25


class SourceError(RuntimeError):
    """A data source could not be reached or returned an unusable payload."""

    def __init__(self, source: str, message: str, status: int | None = None):
        super().__init__(f"{source}: {message}")
        self.source = source
        self.status = status
        self.message = message


class OfflineError(SourceError):
    pass


@dataclass
class Response:
    url: str
    status: int
    json_body: Any | None
    text: str
    from_cache: bool
    retrieved: str

    def json(self) -> Any:
        if self.json_body is None:
            raise SourceError(self.url, "response was not JSON")
        return self.json_body


_last_call: dict[str, float] = {}
_lock = threading.Lock()


def _throttle(host: str) -> None:
    interval = HOST_INTERVALS.get(host, DEFAULT_INTERVAL)
    with _lock:
        last = _last_call.get(host, 0.0)
        wait = interval - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
        _last_call[host] = time.monotonic()


def _cache_path(method: str, url: str, params: dict[str, Any] | None, body: Any) -> Path:
    blob = json.dumps(
        {"m": method, "u": url, "p": params or {}, "b": body}, sort_keys=True, default=str
    )
    digest = hashlib.sha256(blob.encode()).hexdigest()[:40]
    host = httpx.URL(url).host or "unknown"
    directory = settings().cache_dir / host
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{digest}.json"


def _read_cache(path: Path, ttl_hours: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if ttl_hours > 0:
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if age_hours > ttl_hours:
            return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def request(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    method: str = "GET",
    json_body: Any | None = None,
    headers: dict[str, str] | None = None,
    source: str = "http",
    ttl_hours: int | None = None,
    attempts: int = 3,
    accept: str = "application/json",
    cache: bool = True,
) -> Response:
    """Fetch a URL with caching, throttling and retries."""
    cfg = settings()
    ttl = cfg.cache_ttl_hours if ttl_hours is None else ttl_hours
    path = _cache_path(method, url, params, json_body)

    if cache:
        cached = _read_cache(path, ttl)
        if cached is not None:
            return Response(
                url=cached.get("url", url),
                status=cached.get("status", 200),
                json_body=cached.get("json"),
                text=cached.get("text", ""),
                from_cache=True,
                retrieved=cached.get("retrieved", "unknown"),
            )

    if cfg.offline:
        raise OfflineError(source, "offline mode is enabled and no cached copy exists")

    request_headers = {
        "User-Agent": USER_AGENT_TEMPLATE.format(email=cfg.contact_email),
        "Accept": accept,
    }
    request_headers.update(headers or {})

    host = httpx.URL(url).host or "unknown"
    last_error: Exception | None = None

    for attempt in range(attempts):
        _throttle(host)
        try:
            with httpx.Client(timeout=cfg.request_timeout, follow_redirects=True) as client:
                response = client.request(
                    method, url, params=params, json=json_body, headers=request_headers
                )
        except httpx.HTTPError as exc:  # network-level failure
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
            continue

        if response.status_code in (429, 500, 502, 503, 504):
            last_error = SourceError(source, f"HTTP {response.status_code}", response.status_code)
            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if (retry_after or "").isdigit() else 2.0 * (attempt + 1)
            time.sleep(min(delay, 20))
            continue

        if response.status_code == 404:
            raise SourceError(source, "not found", 404)
        if response.status_code >= 400:
            raise SourceError(
                source, f"HTTP {response.status_code}: {response.text[:200]}", response.status_code
            )

        parsed: Any | None = None
        content_type = response.headers.get("content-type", "")
        if "json" in content_type:
            try:
                parsed = response.json()
            except ValueError:
                parsed = None

        record = {
            "url": str(response.url),
            "status": response.status_code,
            "json": parsed,
            "text": "" if parsed is not None else response.text[:2_000_000],
            "retrieved": utcnow(),
        }
        if cache:
            try:
                path.write_text(json.dumps(record), encoding="utf-8")
            except OSError:
                pass
        return Response(
            url=record["url"],
            status=record["status"],
            json_body=parsed,
            text=record["text"],
            from_cache=False,
            retrieved=record["retrieved"],
        )

    raise SourceError(source, f"failed after {attempts} attempts ({last_error})")


def get_json(url: str, *, source: str, params: dict[str, Any] | None = None, **kwargs: Any) -> Any:
    return request(url, params=params, source=source, **kwargs).json()


def get_text(url: str, *, source: str, params: dict[str, Any] | None = None, **kwargs: Any) -> str:
    kwargs.setdefault("accept", "text/plain")
    response = request(url, params=params, source=source, **kwargs)
    return response.text


def build_url(base: str, params: dict[str, Any]) -> str:
    return f"{base}?{urlencode(params)}" if params else base
