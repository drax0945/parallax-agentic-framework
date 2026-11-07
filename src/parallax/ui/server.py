"""The console's HTTP layer.

Standard library only — no web framework — so the console adds no dependencies
to a scientific install. Tool calls stream to the browser over server-sent
events as they actually happen, with real durations.
"""

from __future__ import annotations

import json
import queue
import re
import threading
import time
import traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..config import settings

HERE = Path(__file__).parent

SMILES_KEYS = {"smiles", "query", "product_smiles", "reactant_smiles", "smiles_list", "candidates"}


def depict(smiles: str, *, width: int = 230, height: int = 190, legend: str = "") -> str | None:
    """SVG for a structure, or None if it is not a parsable molecule."""
    from ..chem.structure import svg

    try:
        return re.sub(r"<\?xml[^>]*\?>", "", svg(smiles, width=width, height=height, legend=legend)).strip()
    except Exception:
        return None


def looks_like_structure(text: Any, min_atoms: int = 3) -> bool:
    from rdkit import Chem

    if not isinstance(text, str) or not (2 < len(text) < 400) or " " in text.strip():
        return False
    mol = Chem.MolFromSmiles(text.strip())
    return mol is not None and mol.GetNumHeavyAtoms() >= min_atoms


def collect_structures(value: Any, found: dict[str, str], *, caption: str = "", depth: int = 0) -> None:
    """Walk a tool payload for anything that is actually a molecule."""
    if depth > 4 or len(found) > 12:
        return
    if isinstance(value, str):
        if looks_like_structure(value):
            found.setdefault(value.strip(), caption)
    elif isinstance(value, dict):
        for key, item in value.items():
            label = caption
            if key in {"name", "pref_name", "title", "transform", "template"} and isinstance(item, str):
                label = item[:60]
            collect_structures(item, found, caption=label, depth=depth + 1)
    elif isinstance(value, (list, tuple)):
        for item in value[:12]:
            collect_structures(item, found, caption=caption, depth=depth + 1)


def _open_block_starts(text: str) -> str:
    """Ensure a blank line before a table or heading that follows prose.

    Markdown needs one, and models routinely write a table straight after a
    sentence. The blank line goes only before the FIRST row of a table — a
    naive regex puts one between every row and destroys it.
    """
    lines = text.split("\n")
    out: list[str] = []
    in_fence = False
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_fence = not in_fence
        previous = lines[index - 1] if index else ""
        previous_stripped = previous.lstrip()
        if not in_fence and previous.strip():
            starts_table = stripped.startswith("|") and not previous_stripped.startswith("|")
            starts_heading = stripped.startswith("#") and not previous_stripped.startswith("#")
            if starts_table or starts_heading:
                out.append("")
        out.append(line)
    return "\n".join(out)


def render_markdown(text: str) -> str:
    """Agent prose -> HTML, with tables. Raw HTML in the source is escaped."""
    import html as _html

    import markdown as _md

    escaped = _html.escape(text or "", quote=False)
    escaped = _open_block_starts(escaped)
    return _md.markdown(escaped, extensions=["tables", "fenced_code", "sane_lists"])


# --- what the agent can do, read from the code rather than written twice ---


def scope_payload() -> dict[str, Any]:
    from ..agent.openai_backend import available_providers
    from ..agent.skills import available_skills
    from ..agent.tools import TOOLS
    from ..chem.optimisation import PROFILES
    from ..chem.reactions import TEMPLATES
    from ..models.qsar import ENDPOINT_PRESETS

    groups: dict[str, list[dict[str, str]]] = {}
    for tool in TOOLS:
        groups.setdefault(tool.group, []).append(
            {
                "name": tool.name,
                "summary": tool.description.split(". ")[0].rstrip(".") + ".",
            }
        )

    return {
        "tiers": [
            {
                "tier": "measured",
                "meaning": "Somebody determined it experimentally.",
                "cites": "the DOI of the study that measured it",
                "sources": "ChEMBL, PubChem annotations, the literature",
            },
            {
                "tier": "computed",
                "meaning": "A deterministic calculation from the structure.",
                "cites": "the method paper (Crippen, Ertl, Bickerton)",
                "sources": "RDKit, run locally",
            },
            {
                "tier": "modelled",
                "meaning": "A prediction from a model trained on cited data.",
                "cites": "the model card, which lists its training-set DOIs",
                "sources": "built on demand from ChEMBL",
            },
        ],
        "tool_groups": groups,
        "n_tools": len(TOOLS),
        "skills": [
            {"name": s.name, "description": s.description} for s in available_skills()
        ],
        "endpoints": [
            {"name": k, "label": v["label"], "note": v["note"]}
            for k, v in ENDPOINT_PRESETS.items()
        ],
        "mpo_profiles": [
            {"name": k, "description": v["description"]} for k, v in PROFILES.items()
        ],
        "n_reaction_templates": len(TEMPLATES),
        "providers": available_providers(),
        "chemical_scope": [
            {"area": "Drug-like small molecules, kinase inhibitors, macrocycles", "status": "designed for this"},
            {"area": "Natural products, peptides, carbohydrates", "status": "works; drug-likeness rules less meaningful"},
            {"area": "Salts and co-crystals", "status": "largest fragment kept, reported in notes"},
            {"area": "Organometallics, coordination compounds", "status": "not supported — flagged unreliable"},
            {"area": "Polymers, materials, inorganics", "status": "out of scope"},
        ],
        "limits": [
            "Retrosynthesis is template-based: it proposes, it does not know. Precedent search is what makes a route real.",
            "No yield prediction.",
            "Starting-material availability is a proxy, not a supplier quote.",
            "Models are only as good as ChEMBL is for that target.",
            "Free patent text search is partial; structure-anchored lookup is more complete.",
            "Macrocycle permeability is not predicted.",
            "Every computational design needs experimental validation.",
        ],
    }


def runtime_payload(provider: str | None = None) -> dict[str, Any]:
    """Which model is actually going to answer, and is it reachable."""
    import shutil

    from ..agent.loop import bundled_cli
    from ..agent.openai_backend import resolve_provider

    cfg = settings()
    choice = (provider or cfg.credential("PARALLAX_LLM_PROVIDER") or "claude").lower()

    if choice == "claude":
        binary = shutil.which("claude") or bundled_cli()
        return {
            "backend": "claude",
            "label": "Claude",
            "model": cfg.model,
            "detail": "Claude Agent SDK" + (" (bundled runtime)" if binary and not shutil.which("claude") else ""),
            "credentials": "ANTHROPIC_API_KEY" if cfg.has_anthropic_key else "logged-in session",
            "local": False,
            "ready": bool(binary),
        }

    try:
        target = resolve_provider(choice)
    except ValueError as exc:
        return {"backend": choice, "label": choice, "ready": False, "detail": str(exc)}

    reachable = None
    if target.name == "ollama":
        import httpx

        try:
            httpx.get(target.base_url.replace("/v1", "") + "/api/tags", timeout=2.0)
            reachable = True
        except Exception:
            reachable = False

    return {
        "backend": target.name,
        "label": target.name.capitalize(),
        "model": target.model,
        "detail": target.endpoint,
        "credentials": "API key set" if target.api_key else "no key needed",
        "local": target.name == "ollama" or "localhost" in target.base_url or "127.0.0.1" in target.base_url,
        "ready": reachable if reachable is not None else True,
    }


# --- the agent run, as a stream of events ---------------------------------


def _run_agent(question: str, provider: str, events: queue.Queue) -> None:
    started = time.monotonic()
    pending: dict[str, float] = {}
    outcomes: dict[str, int] = {"ok": 0, "failed": 0}
    structures: dict[str, str] = {}

    def emit(kind: str, **payload: Any) -> None:
        events.put(
            {
                "kind": kind,
                "at": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                "elapsed": round(time.monotonic() - started, 2),
                **payload,
            }
        )

    try:
        if provider == "claude":
            import asyncio

            from ..agent.loop import Turn, run_session

            def on_turn(turn: Turn) -> None:
                if turn.kind == "tool_use":
                    pending[turn.tool or ""] = time.monotonic()
                    for key, value in (turn.payload or {}).items():
                        if key in SMILES_KEYS:
                            collect_structures(value, structures, caption=f"input to {turn.tool}")
                    emit("tool_call", tool=turn.tool, args=turn.payload)
                elif turn.kind == "text" and turn.text.strip():
                    emit("text", text=turn.text)

            result = asyncio.run(run_session(question, on_turn=on_turn))
            if result.get("auth_problem"):
                emit(
                    "error",
                    message=result["auth_problem"]["message"],
                    advice=result["auth_problem"]["fix"],
                )
        else:
            from ..agent.openai_backend import Step, resolve_provider, run

            target = resolve_provider(provider)

            def on_step(step: Step) -> None:
                if step.kind == "tool_use":
                    pending[step.tool or ""] = time.monotonic()
                    for key, value in (step.payload or {}).items():
                        if key in SMILES_KEYS:
                            collect_structures(value, structures, caption=f"input to {step.tool}")
                    emit("tool_call", tool=step.tool, args=step.payload)
                elif step.kind == "tool_result":
                    start = pending.pop(step.tool or "", None)
                    ok = not (isinstance(step.payload, dict) and step.payload.get("error"))
                    outcomes["ok" if ok else "failed"] += 1
                    if ok:
                        collect_structures(step.payload, structures)
                    emit(
                        "tool_result",
                        tool=step.tool,
                        ok=ok,
                        took=round(time.monotonic() - start, 2) if start else None,
                        summary=_summarise(step.payload),
                    )
                elif step.kind == "text" and step.text.strip():
                    emit("text", text=step.text)

            result = run(question, provider=target, on_step=on_step)
            if result.get("error"):
                emit("error", message=result["error"], advice=result.get("advice", ""))

        answer = result.get("answer", "") or ""
        citations = result.get("citations", 0)
        calls = result.get("n_tool_calls", 0)
        # A model can fabricate a number after a tool refused it. If the prose
        # carries figures that nothing successful produced, say so loudly.
        ungrounded = is_ungrounded(answer, citations, outcomes["ok"])

        # any structure the agent wrote in backticks counts too
        for candidate in re.findall(r"`([^`\n]{3,200})`", answer):
            if looks_like_structure(candidate):
                structures.setdefault(candidate.strip(), "")

        drawn = []
        for smiles, caption in list(structures.items())[:8]:
            image = depict(smiles, legend="")
            if image:
                drawn.append({"smiles": smiles, "caption": caption, "svg": image})

        emit(
            "done",
            answer=answer,
            answer_html=render_markdown(answer),
            molecules=drawn,
            n_tool_calls=calls,
            tools_ok=outcomes["ok"],
            tools_failed=outcomes["failed"],
            citations=citations,
            ungrounded=ungrounded,
        )
    except Exception as exc:  # the stream must always terminate
        emit("error", message=f"{type(exc).__name__}: {exc}", advice=traceback.format_exc(limit=2))
        emit("done", answer="", n_tool_calls=0, citations=0)


def is_ungrounded(answer: str, citations: int, tools_ok: int) -> bool:
    """True when the prose carries figures that nothing successful produced.

    A model can fabricate a number straight after a tool refuses its arguments.
    The tools hold the floor — they never invent a value — but the model's own
    text is not similarly constrained, so the console has to say when an answer
    rests on nothing.
    """
    if not answer:
        return False
    return any(ch.isdigit() for ch in answer) and citations == 0 and tools_ok == 0


def _summarise(payload: Any) -> str:
    if not isinstance(payload, dict):
        return str(payload)[:140]
    if payload.get("error"):
        return f"error: {payload['error']}"[:160]
    for key in ("summary", "verdict", "assessment", "how_to_read", "note"):
        if isinstance(payload.get(key), str):
            return payload[key][:160]
    counts = {
        k: (len(v) if isinstance(v, (list, dict)) else v)
        for k, v in payload.items()
        if k.startswith("n_") or k in {"count", "returned", "n_results"}
    }
    if counts:
        return ", ".join(f"{k}={v}" for k, v in list(counts.items())[:3])
    keys = [k for k in payload if not k.startswith("_")][:5]
    return "returned " + ", ".join(keys)


# --- http ----------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "Parallax"

    def log_message(self, *args: Any) -> None:  # keep the terminal clean
        pass

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, status: int = 200) -> None:
        self._send(json.dumps(payload, default=str).encode(), "application/json", status)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path
        params = parse_qs(parsed.query)

        if route in ("/", "/index.html"):
            self._send((HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif route == "/api/scope":
            self._json(scope_payload())
        elif route == "/api/runtime":
            self._json(runtime_payload((params.get("provider") or [None])[0]))
        elif route == "/api/depict":
            smiles = (params.get("smiles") or [""])[0]
            image = depict(smiles, width=int((params.get("w") or [230])[0]), height=190)
            if image is None:
                self._json({"error": "not a parsable structure"}, 400)
            else:
                self._send(image.encode(), "image/svg+xml")
        elif route == "/api/ask":
            self._stream(
                (params.get("q") or [""])[0],
                (params.get("provider") or ["claude"])[0],
            )
        else:
            self._json({"error": "not found"}, 404)

    def _stream(self, question: str, provider: str) -> None:
        if not question.strip():
            self._json({"error": "empty question"}, 400)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        events: queue.Queue = queue.Queue()
        worker = threading.Thread(
            target=_run_agent, args=(question, provider, events), daemon=True
        )
        worker.start()

        while True:
            try:
                event = events.get(timeout=1.0)
            except queue.Empty:
                try:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                continue
            try:
                self.wfile.write(f"data: {json.dumps(event, default=str)}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            if event.get("kind") == "done":
                return


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    settings().ensure_dirs()
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    if open_browser:
        import webbrowser

        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    print(f"  Parallax console on {url}   (ctrl-c to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        httpd.server_close()
