"""Parallax command line.

Two ways in:

* `parallax ask "..."` — the agent, with skills, tools and memory.
* everything else — the tools directly, for when you know what you want.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Optional

import typer

from .config import settings

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Agentic cheminformatics: cited data, deterministic chemistry, continual memory.",
)
memory_app = typer.Typer(no_args_is_help=True, help="Inspect and maintain the memory store.")
app.add_typer(memory_app, name="memory")


# --- output helpers ------------------------------------------------------

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
OFF = "\033[0m"


def _plain() -> bool:
    return not sys.stdout.isatty()


def _c(text: str, colour: str) -> str:
    return text if _plain() else f"{colour}{text}{OFF}"


def head(text: str) -> None:
    typer.echo("\n" + _c(text, BOLD))
    typer.echo(_c("─" * min(len(text), 72), DIM))


def kv(label: str, value: Any, width: int = 24) -> None:
    typer.echo(f"  {label.ljust(width)} {value}")


def dump(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, default=str))


def _fail(message: str) -> None:
    typer.echo(_c(f"error: {message}", RED), err=True)
    raise typer.Exit(1)


# --- agent ---------------------------------------------------------------


@app.command()
def ask(
    prompt: list[str] = typer.Argument(..., help="What you want the agent to do."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override the model."),
    max_turns: int = typer.Option(60, "--max-turns", help="Cap the agent's tool-use turns."),
    show_tools: bool = typer.Option(True, "--show-tools/--quiet", help="Print tool calls as they run."),
    json_out: bool = typer.Option(False, "--json", help="Emit the full transcript as JSON."),
) -> None:
    """Run a research task with the full agent (skills, tools, memory)."""
    from .agent.loop import Turn, run_session

    question = " ".join(prompt)
    settings().ensure_dirs()

    def on_turn(turn: Turn) -> None:
        if json_out:
            return
        if turn.kind == "tool_use" and show_tools:
            arguments = ", ".join(
                f"{k}={str(v)[:48]}" for k, v in list(turn.payload.items())[:3]
            )
            typer.echo(_c(f"  · {turn.tool}({arguments})", DIM))
        elif turn.kind == "text" and turn.text.strip():
            typer.echo(turn.text)

    try:
        result = asyncio.run(
            run_session(question, model=model, max_turns=max_turns, on_turn=on_turn)
        )
    except FileNotFoundError as exc:
        _fail(
            f"could not start the agent ({exc}). The `claude` CLI must be installed and "
            "logged in, or set ANTHROPIC_API_KEY."
        )
        return

    if json_out:
        dump(result)
        return

    meta = result.get("meta") or {}
    typer.echo()
    typer.echo(
        _c(
            f"  {result['n_tool_calls']} tool calls · {result['citations']} sources cited"
            + (f" · {meta.get('turns')} turns" if meta.get("turns") else "")
            + (f" · ${meta['cost_usd']:.3f}" if meta.get("cost_usd") else ""),
            DIM,
        )
    )
    if result.get("run_dir"):
        typer.echo(_c(f"  run saved to {result['run_dir']}", DIM))


@app.command()
def mcp() -> None:
    """Run the stdio MCP server (for Claude Code, the desktop app, or any MCP client)."""
    from .agent.mcp_server import main as serve

    serve()


# --- direct tools --------------------------------------------------------


@app.command()
def compound(
    query: str = typer.Argument(..., help="Name, SMILES, InChIKey, CAS, CID or ChEMBL ID."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Profile a compound: identity, properties, measured bioactivity, all cited."""
    from .agent.tools import call_tool

    result = call_tool("compound_profile", {"query": query})
    if json_out:
        dump(result)
        return
    if result.get("error"):
        _fail(result["error"])

    identity = result["identity"]
    head(identity.get("name") or query)
    kv("SMILES", identity["smiles"])
    kv("Formula", identity.get("formula"))
    kv("InChIKey", identity.get("inchikey"))
    kv("Identifiers", json.dumps(identity.get("identifiers", {})))

    screen = result.get("safety_screen", {})
    if screen.get("level") != "clear":
        typer.echo(
            _c(f"  safety: {screen['level']} — {screen.get('reason')}", YELLOW)
        )

    values = result["computed"]["computed_properties"]
    head("Computed properties")
    for key in ("mw", "clogp", "tpsa", "hbd", "hba", "rotatable_bonds", "fraction_csp3", "qed", "sa_score"):
        kv(key, values.get(key))
    if values.get("is_macrocycle"):
        kv("macrocycle", f"yes, ring size(s) {values['macrocycle_ring_sizes']}")

    rules = result["computed"]["druglikeness"]
    kv("rules passed", f"{rules['summary']['rules_passed']}/{rules['summary']['rules_total']}")
    for rule in rules["rules"]:
        if not rule["passes"]:
            typer.echo(_c(f"    ✗ {rule['name']}: {', '.join(rule['violations'])}", YELLOW))

    flags = result["computed"]["liability_flags"]["flags"]
    if flags:
        head("Liability flags")
        for flag in flags:
            typer.echo(f"  • {flag['flag']}")
            typer.echo(_c(f"      next: {flag['next_experiment']}", DIM))

    measured = result.get("measured_bioactivity")
    if measured and measured.get("by_target"):
        head("Measured bioactivity (each value cited)")
        for target, data in list(measured["by_target"].items())[:6]:
            typer.echo(f"  {target} — best pChEMBL {data['best_pchembl']} ({data['n']} records)")
            for example in data["examples"][:2]:
                typer.echo(
                    _c(
                        f"      {example['type']} {example['value_nM']} nM · {example['citation']}",
                        DIM,
                    )
                )

    mechanisms = (result.get("mechanisms") or {}).get("mechanisms") or []
    if mechanisms:
        head("Mechanism")
        for mechanism in mechanisms[:4]:
            typer.echo(f"  • {mechanism['mechanism_of_action']} ({mechanism.get('action_type')})")

    _print_sources()


@app.command()
def properties(
    smiles: str = typer.Argument(..., help="SMILES string."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Computed property profile with drug-likeness rules and alerts."""
    from .agent.tools import call_tool

    result = call_tool("computed_properties", {"smiles": smiles})
    if json_out or result.get("error"):
        dump(result)
        return
    head("Computed properties")
    for key, value in result["computed_properties"].items():
        kv(key, value)
    head("Drug-likeness")
    for rule in result["druglikeness"]["rules"]:
        mark = _c("pass", GREEN) if rule["passes"] else _c("fail", YELLOW)
        typer.echo(f"  {mark}  {rule['name']}" + (f" — {', '.join(rule['violations'])}" if rule["violations"] else ""))
    typer.echo(_c(f"\n  {result['druglikeness']['interpretation']}", DIM))
    kv("CNS MPO", f"{result['cns_mpo']['cns_mpo_score']} / 6")
    alerts = result["structural_alerts"]
    kv("structural alerts", alerts["n_alerts"] or "none")
    for catalog, items in (alerts.get("by_catalog") or {}).items():
        typer.echo(f"    {catalog}: {', '.join(items[:4])}")


@app.command()
def train(
    target: str = typer.Argument(..., help="Target name or ChEMBL ID."),
    endpoint: str = typer.Option("inhibition", "--endpoint", "-e", help="inhibition | binding | activation | cell_potency | any_potency"),
    split: str = typer.Option("scaffold", "--split", "-s", help="scaffold | time | random"),
    max_records: int = typer.Option(3000, "--max-records"),
    name: Optional[str] = typer.Option(None, "--name"),
) -> None:
    """Train a potency/activation model from ChEMBL data and report honest performance."""
    from .agent.tools import call_tool

    typer.echo(_c(f"  pulling {endpoint} data for {target} …", DIM))
    result = call_tool(
        "train_potency_model",
        {"target": target, "endpoint": endpoint, "split": split, "max_records": max_records, "name": name},
    )
    if result.get("error"):
        _fail(result["error"])
    card = result["card"]
    head(f"Model: {card['name']}")
    kv("target", f"{card['target_name']} ({card['target_chembl_id']})")
    kv("endpoint", card["endpoint_label"])
    kv("algorithm", card["algorithm"])
    kv("compounds", f"{card['n_compounds']} (from {card['n_raw_measurements']} measurements)")
    kv("source documents", card["source_documents"])
    kv("split", f"{card['split']} ({card['n_train']} train / {card['n_test']} test)")
    head("Held-out performance")
    for key in ("r2", "rmse", "mae", "spearman"):
        kv(key, card["metrics"].get(key))
    kv("baseline rmse", card["baseline_metrics"]["rmse"])
    gain = round(card["baseline_metrics"]["rmse"] - card["metrics"]["rmse"], 2)
    kv("signal over baseline", f"{gain} log units")
    if gain < 0.1:
        typer.echo(_c("  this model barely beats predicting the mean — do not use its numbers", YELLOW))
    for warning in card.get("warnings", []):
        typer.echo(_c(f"  ! {warning}", YELLOW))
    for note in card.get("data_notes", []):
        typer.echo(_c(f"  · {note}", DIM))
    typer.echo(f"\n  predict with: parallax predict {card['name']} '<SMILES>'")


@app.command()
def predict(
    model: str = typer.Argument(..., help="Model name (see `parallax models`)."),
    smiles: list[str] = typer.Argument(..., help="One or more SMILES."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Predict potency with applicability domain and uncertainty."""
    from .agent.tools import call_tool

    result = call_tool("predict_potency", {"model": model, "smiles": list(smiles)})
    if json_out or result.get("error"):
        dump(result)
        return
    head(f"{result['endpoint']} — {result['target']}")
    performance = result["model_performance"]
    typer.echo(
        _c(
            f"  model: {performance['split']} split, RMSE {performance['rmse']} "
            f"(baseline {performance['baseline_rmse_predicting_the_mean']}), ρ {performance['spearman']}",
            DIM,
        )
    )
    for prediction in result["predictions"]:
        typer.echo()
        typer.echo(f"  {prediction['query']}")
        kv("predicted", f"{prediction['predicted_value']} ({prediction['predicted_nM']} nM)")
        kv("68% interval", prediction["interval_68"])
        domain = prediction["applicability_domain"]
        colour = GREEN if domain.get("in_domain") else YELLOW
        typer.echo(f"  {'domain'.ljust(24)} {_c(domain['verdict'], colour)}")
        if prediction.get("in_training_set"):
            typer.echo(_c(f"  ! {prediction['caveat']}", YELLOW))
        for neighbour in prediction["nearest_training_compounds"][:2]:
            typer.echo(
                _c(f"      nearest: {neighbour['measured']} measured, similarity {neighbour['similarity']}", DIM)
            )
    typer.echo(_c(f"\n  {result['how_to_read']}", DIM))


@app.command()
def models() -> None:
    """List models trained in this workspace."""
    from .agent.tools import call_tool

    result = call_tool("list_potency_models", {})
    if not result["n_models"]:
        typer.echo("  no models yet — build one with `parallax train <target>`")
        return
    head(f"{result['n_models']} model(s)")
    for model in result["models"]:
        typer.echo(f"  {model['name']}")
        typer.echo(
            _c(
                f"      {model['target']} · {model['endpoint']} · {model['algorithm']} · "
                f"n={model['n_compounds']} · {model['split']} R² {model['test_r2']}, RMSE {model['test_rmse']}",
                DIM,
            )
        )


@app.command()
def retro(
    smiles: str = typer.Argument(..., help="Target SMILES."),
    plan_routes: bool = typer.Option(False, "--plan", help="Search multi-step routes instead of one step."),
    depth: int = typer.Option(3, "--depth"),
    top: int = typer.Option(6, "--top"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Retrosynthetic disconnections or full route search."""
    from .agent.tools import call_tool

    if plan_routes:
        result = call_tool("retro_plan", {"smiles": smiles, "max_depth": depth, "max_routes": top})
        if json_out:
            dump(result)
            return
        if result.get("refused"):
            _fail(f"refused: {result['safety']['reason']}")
        head(f"{result['n_routes']} route(s)")
        for index, route in enumerate(result["routes"], start=1):
            typer.echo(f"\n  Route {index} — score {route['score']}, {route['n_steps']} step(s)")
            for step in route["steps"]:
                typer.echo(f"    {step['step']}. {step['template']}")
                typer.echo(_c(f"       {' + '.join(step['reactants'])}", DIM))
                typer.echo(_c(f"       → {step['product']}", DIM))
                typer.echo(_c(f"       {step['conditions']}", DIM))
                for concern in step["concerns"][:1]:
                    typer.echo(_c(f"       ! {concern}", YELLOW))
            for material in route["starting_materials"]:
                mark = _c("available", GREEN) if material["likely_available"] else _c("to be made", YELLOW)
                typer.echo(f"    SM [{mark}] {material['smiles']}")
        typer.echo(_c(f"\n  {result['caveat']}", DIM))
    else:
        result = call_tool("retro_disconnect", {"smiles": smiles, "top": top})
        if json_out:
            dump(result)
            return
        if result.get("refused"):
            _fail(f"refused: {result['safety']['reason']}")
        head(f"{result['n_disconnections']} disconnection(s)")
        for disconnection in result["disconnections"]:
            typer.echo(f"\n  [{disconnection['score']}] {disconnection['template']}")
            for precursor in disconnection["precursors"]:
                typer.echo(f"      ← {precursor}")
            typer.echo(_c(f"      {disconnection['conditions']}", DIM))
            if disconnection.get("ring_size"):
                typer.echo(_c(f"      ring size {disconnection['ring_size']}", DIM))
            for concern in disconnection["concerns"][:2]:
                typer.echo(_c(f"      ! {concern}", YELLOW))
            typer.echo(_c(f"      precedent query: {disconnection['precedent_query']}", DIM))
        typer.echo(_c(f"\n  {result['caveat']}", DIM))


@app.command()
def feasibility(smiles: str = typer.Argument(...), json_out: bool = typer.Option(False, "--json")) -> None:
    """Can this be made, and what is the dominant difficulty?"""
    from .agent.tools import call_tool

    result = call_tool("synthesis_feasibility", {"smiles": smiles})
    if json_out:
        dump(result)
        return
    head("Feasibility")
    for driver in result["drivers"]:
        colour = {"favourable": GREEN, "neutral": DIM, "adverse": YELLOW}[driver["weight"]]
        typer.echo(f"  {_c(driver['weight'].ljust(11), colour)} {driver['factor']}: {driver['reading']}")
    head("Next actions")
    for action in result["next_actions"]:
        typer.echo(f"  • {action}")


@app.command()
def design(
    smiles: str = typer.Argument(..., help="Lead compound SMILES."),
    macrocycle: bool = typer.Option(False, "--macrocycle", help="Propose macrocyclisations instead of analogs."),
    model: Optional[str] = typer.Option(None, "--model", help="Score designs with a trained model."),
    limit: int = typer.Option(10, "--limit"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Generate analogs, or macrocyclic designs, from a lead."""
    from .agent.tools import call_tool

    if macrocycle:
        result = call_tool("design_macrocycles", {"smiles": smiles, "max_designs": limit})
        if json_out:
            dump(result)
            return
        head(f"{result['n_designs']} macrocyclisation(s) — parent has {result['parent_rotatable_bonds']} rotatable bonds")
        candidates = [d["smiles"] for d in result["designs"]]
        for item in result["designs"]:
            typer.echo(f"\n  ring {item['ring_size']} via {item['tether']}  (rotatable bonds {item['rotatable_bond_change']:+d})")
            typer.echo(f"    {item['smiles']}")
            typer.echo(_c(f"    SA {item['properties']['sa_score']} · cLogP {item['properties']['clogp']} · TPSA {item['properties']['tpsa']}", DIM))
            typer.echo(_c(f"    closure: {item['closure_chemistry']}", DIM))
        typer.echo(_c(f"\n  {result['design_rationale']}", DIM))
    else:
        result = call_tool("design_analogs", {"smiles": smiles, "max_total": limit})
        if json_out:
            dump(result)
            return
        head(f"{result['n_analogs']} analog(s)")
        candidates = [a["smiles"] for a in result["analogs"]]
        for analog in result["analogs"]:
            deltas = analog["property_deltas"]
            typer.echo(f"\n  {analog['transform']}")
            typer.echo(f"    {analog['smiles']}")
            typer.echo(_c(f"    ΔcLogP {deltas['clogp']:+} · ΔTPSA {deltas['tpsa']:+} · ΔSA {deltas['sa_score']:+} · similarity {analog['similarity_to_parent']}", DIM))
            typer.echo(_c(f"    {analog['rationale']}", DIM))

    if model and candidates:
        scored = call_tool("score_designs", {"smiles_list": candidates, "model": model})
        head(f"Ranked by predicted potency ({model})")
        for row in scored["designs"][:limit]:
            prediction = row.get("predicted") or {}
            typer.echo(
                f"  {prediction.get('value', '—')}  {row['smiles'][:60]}"
                + (f"  [{prediction.get('applicability_domain', '')}]" if prediction else "")
            )
        typer.echo(_c(f"\n  {scored['how_to_read']}", DIM))


@app.command()
def mmpa(
    target: str = typer.Argument(..., help="Target name or ChEMBL ID."),
    endpoint: str = typer.Option("inhibition", "--endpoint", "-e"),
    limit: int = typer.Option(1000, "--limit"),
    top: int = typer.Option(15, "--top"),
) -> None:
    """Matched molecular pair analysis over measured activity data."""
    from .agent.tools import call_tool

    typer.echo(_c(f"  assembling {endpoint} data for {target} …", DIM))
    result = call_tool("matched_molecular_pairs", {"target": target, "endpoint": endpoint, "limit": limit})
    if result.get("error"):
        _fail(result["error"])
    head(f"{result['target']} — {result['endpoint']}")
    kv("compounds", result["n_compounds"])
    kv("pairs found", result["n_pairs_found"])
    kv("transformations", result["n_transformations"])
    head("Largest consistent effects")
    for transformation in result["transformations"][:top]:
        mark = _c("consistent", GREEN) if transformation["consistent"] else _c("variable", YELLOW)
        typer.echo(
            f"  {transformation['mean_delta']:+.2f} log  n={transformation['n_pairs']:<3} {mark}  "
            f"{transformation['transformation'][:64]}"
        )
    typer.echo(_c(f"\n  {result['how_to_read']}", DIM))


@app.command()
def lit(
    query: list[str] = typer.Argument(..., help="Search terms."),
    limit: int = typer.Option(10, "--limit", "-n"),
    patents: bool = typer.Option(False, "--patents", help="Search patent abstracts instead."),
    year_from: Optional[int] = typer.Option(None, "--since"),
) -> None:
    """Search the literature or patents, with citation keys."""
    from .agent.tools import call_tool

    text = " ".join(query)
    tool = "patent_search" if patents else "literature_search"
    result = call_tool(tool, {"query": text, "limit": limit, "year_from": year_from})
    if result.get("error"):
        _fail(result["error"])
    head(f"{result.get('hit_count') or result.get('total') or len(result['results'])} hit(s) — showing {len(result['results'])}")
    for record in result["results"]:
        typer.echo(f"\n  {record['title']}")
        typer.echo(
            _c(
                f"    {record.get('authors') or ''} · {record.get('venue') or ''} {record.get('year') or ''}",
                DIM,
            )
        )
        typer.echo(_c(f"    {record['citation']}", CYAN))
        if record.get("abstract"):
            typer.echo(_c(f"    {record['abstract'][:220]}…", DIM))


@app.command()
def precedent(query: list[str] = typer.Argument(..., help="The transformation to look for.")) -> None:
    """Search for real reported examples of a transformation."""
    from .agent.tools import call_tool

    result = call_tool("reaction_precedent_search", {"query": " ".join(query)})
    head(f"{result['n_precedents_found']} precedent(s)")
    typer.echo(_c(f"  {result['verdict']}", YELLOW if not result["n_precedents_found"] else GREEN))
    licensed = result["licensed_database"]
    kv("licensed database", "available" if licensed["available"] else "not configured")
    for group, label in (("journal_precedents", "Journals"), ("patent_precedents", "Patents")):
        records = result.get(group) or []
        if records:
            head(label)
            for record in records[:6]:
                typer.echo(f"  {record['title'][:90]}")
                typer.echo(_c(f"    {record.get('year')} · {record['citation']}", DIM))


@app.command("cite")
def cite(
    verify: bool = typer.Option(False, "--verify", help="Check every DOI against Crossref."),
    format: str = typer.Option("markdown", "--format", "-f", help="markdown | bibtex | json"),
    doi: Optional[str] = typer.Option(None, "--doi", help="Verify one DOI and show its real metadata."),
) -> None:
    """Show or verify the citation registry for this process."""
    from .agent.tools import call_tool

    if doi:
        dump(call_tool("check_doi", {"doi": doi}))
        return
    if verify:
        dump(call_tool("verify_all_citations", {}))
        return
    result = call_tool("citation_bibliography", {"format": format})
    typer.echo(result["bibliography"])


@app.command()
def screen(
    smiles: Optional[str] = typer.Option(None, "--smiles"),
    text: Optional[str] = typer.Option(None, "--text"),
    mode: str = typer.Option("synthesis", "--mode", help="analysis | synthesis"),
) -> None:
    """Safety screen a structure or request."""
    from .agent.tools import call_tool

    if not smiles and not text:
        _fail("give --smiles or --text")
    dump(call_tool("safety_screen", {"smiles": smiles, "text": text, "mode": mode}))


# --- memory --------------------------------------------------------------


@memory_app.command("recall")
def memory_recall_cmd(
    query: list[str] = typer.Argument(...),
    k: int = typer.Option(6, "-k"),
) -> None:
    """Search memory."""
    from .agent.tools import call_tool

    result = call_tool("memory_recall", {"query": " ".join(query), "k": k})
    head(f"{result['n_results']} of {result['store_size']} memories")
    for entry in result["results"]:
        typer.echo(f"\n  [{entry['relevance']}] {entry['title']}")
        typer.echo(
            _c(
                f"    {entry['type']} · confidence {entry['confidence']} · support {entry['support']}"
                + (f" · contradicted {entry['contradictions']}×" if entry["contradictions"] else "")
                + f" · {entry['id']}",
                DIM,
            )
        )
        typer.echo(_c(f"    {' '.join(entry['body'].split())[:260]}", DIM))


@memory_app.command("list")
def memory_list() -> None:
    """Show the human-readable memory index."""
    path = settings().memory_index_file
    if not path.exists():
        typer.echo("  memory is empty")
        return
    typer.echo(path.read_text(encoding="utf-8"))


@memory_app.command("stats")
def memory_stats_cmd() -> None:
    """Memory store health."""
    from .agent.tools import call_tool

    dump(call_tool("memory_stats", {}))


@memory_app.command("consolidate")
def memory_consolidate_cmd() -> None:
    """Merge duplicates, decay stale entries, rewrite the index."""
    from .agent.tools import call_tool

    dump(call_tool("memory_consolidate", {}))


@memory_app.command("add")
def memory_add(
    title: str = typer.Argument(...),
    body: str = typer.Argument(...),
    type: str = typer.Option("finding", "--type", "-t"),
    tags: str = typer.Option("", "--tags", help="comma separated"),
    confidence: float = typer.Option(0.6, "--confidence"),
) -> None:
    """Write a memory by hand."""
    from .agent.tools import call_tool

    dump(
        call_tool(
            "memory_remember",
            {
                "type": type,
                "title": title,
                "body": body,
                "tags": [t.strip() for t in tags.split(",") if t.strip()],
                "confidence": confidence,
            },
        )
    )


# --- workspace -----------------------------------------------------------


@app.command()
def tools() -> None:
    """List the tools the agent can call, by group."""
    from .agent.prompts import tool_overview

    grouped = tool_overview()
    total = sum(len(v) for v in grouped.values())
    head(f"{total} tools")
    for group, items in sorted(grouped.items()):
        typer.echo(f"\n  {_c(group, BOLD)}")
        for item in items:
            typer.echo(f"    {item['name'].ljust(30)} {_c(item['summary'][:88], DIM)}")


@app.command()
def doctor() -> None:
    """Check the environment: dependencies, credentials, data sources, memory."""
    cfg = settings()
    head("Workspace")
    kv("root", cfg.workspace)
    kv("memory", f"{cfg.memory_dir} ({'exists' if cfg.memory_dir.exists() else 'not created yet'})")
    kv("skills", f"{cfg.skills_dir} ({len(list(cfg.skills_dir.glob('*/SKILL.md'))) if cfg.skills_dir.exists() else 0} skills)")
    kv("models", len(list(cfg.models_dir.glob('*.pkl'))) if cfg.models_dir.exists() else 0)
    kv("cache", f"{sum(1 for _ in cfg.cache_dir.rglob('*.json'))} entries" if cfg.cache_dir.exists() else "empty")

    head("Dependencies")
    for module, label in (
        ("rdkit", "RDKit"),
        ("sklearn", "scikit-learn"),
        ("httpx", "httpx"),
        ("markdown", "markdown"),
        ("mcp", "MCP"),
        ("claude_agent_sdk", "Claude Agent SDK"),
    ):
        try:
            imported = __import__(module)
            version = getattr(imported, "__version__", "ok")
            typer.echo(f"  {_c('✓', GREEN)} {label.ljust(22)} {version}")
        except ImportError:
            typer.echo(f"  {_c('✗', RED)} {label.ljust(22)} missing")

    head("Agent runtime")
    import shutil

    claude_cli = shutil.which("claude")
    if claude_cli:
        typer.echo(f"  {_c('✓', GREEN)} claude CLI              {claude_cli}")
    elif cfg.has_anthropic_key:
        typer.echo(f"  {_c('✓', GREEN)} ANTHROPIC_API_KEY set")
    else:
        typer.echo(
            f"  {_c('✗', RED)} no agent runtime — install the `claude` CLI or set ANTHROPIC_API_KEY"
        )
    kv("model", cfg.model)

    head("Data sources")
    checks = [
        ("PubChem", "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/aspirin/cids/JSON"),
        ("ChEMBL", "https://www.ebi.ac.uk/chembl/api/data/status?format=json"),
        ("Europe PMC", "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=aspirin&format=json&pageSize=1"),
        ("Crossref", "https://api.crossref.org/works/10.1021/jm1012374"),
    ]
    from .net import SourceError, request

    for label, url in checks:
        try:
            response = request(url, source=label.lower(), attempts=1, ttl_hours=1)
            state = "cached" if response.from_cache else "live"
            typer.echo(f"  {_c('✓', GREEN)} {label.ljust(22)} reachable ({state})")
        except SourceError as exc:
            typer.echo(f"  {_c('✗', RED)} {label.ljust(22)} {exc.message[:50]}")

    head("Licensed sources")
    from .sources import reaxys

    status = reaxys.availability()
    mark = _c("✓", GREEN) if status["available"] else _c("—", YELLOW)
    typer.echo(f"  {mark} Reaxys                 {status['mode']}")
    typer.echo(f"  {_c('✓', GREEN) if status['cas_common_chemistry'] else _c('—', YELLOW)} CAS Common Chemistry   {'configured' if status['cas_common_chemistry'] else 'no CAS_API_KEY'}")
    if not status["available"]:
        typer.echo(_c(f"      {status['setup']['option_1_adapter']}", DIM))
        typer.echo(_c(f"      {status['setup']['option_2_http']}", DIM))

    head("Memory")
    from .memory import stats as memory_stats_fn

    try:
        dump(memory_stats_fn())
    except Exception as exc:
        typer.echo(f"  memory unavailable: {exc}")


@app.command()
def init(
    force: bool = typer.Option(False, "--force", help="Overwrite existing config files."),
) -> None:
    """Create the workspace layout, .env.example and MCP registration."""
    cfg = settings()
    cfg.ensure_dirs()
    created: list[str] = []

    mcp_config = cfg.workspace / ".mcp.json"
    if force or not mcp_config.exists():
        # Portable on purpose: no absolute paths, so this file can be committed.
        # The workspace root is found by walking up from the server's cwd.
        mcp_config.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "parallax": {
                            # `uv run` resolves the project venv from the cwd, so this
                            # works without the venv being active or on PATH.
                            "command": "uv",
                            "args": ["run", "parallax", "mcp"],
                        }
                    }
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        created.append(str(mcp_config))

    env_example = cfg.workspace / ".env.example"
    if force or not env_example.exists():
        env_example.write_text(
            "\n".join(
                [
                    "# Parallax configuration — copy to .env and fill in what you have.",
                    "# Nothing here is required: every core source is free and needs no key.",
                    "",
                    "# Polite contact address sent to PubChem/ChEMBL/Crossref (good practice).",
                    "PARALLAX_CONTACT_EMAIL=you@institution.edu",
                    "",
                    "# Model used by `parallax ask`.",
                    "# PARALLAX_MODEL=claude-opus-4-5-20251101",
                    "",
                    "# Only if you are not using the logged-in `claude` CLI.",
                    "# ANTHROPIC_API_KEY=",
                    "",
                    "# Licensed reaction databases (optional).",
                    "# Option 1 — your own authenticated client:",
                    "# REAXYS_ADAPTER=yourpkg.reaxys_client:client",
                    "# Option 2 — a generic HTTP endpoint:",
                    "# REAXYS_API_URL=",
                    "# REAXYS_API_KEY=",
                    "",
                    "# Optional free-tier keys.",
                    "# CAS_API_KEY=",
                    "# PATENTSVIEW_API_KEY=",
                    "# OPENALEX_API_KEY=",
                    "",
                    "# Behaviour.",
                    "# PARALLAX_CACHE_TTL_HOURS=168",
                    "# PARALLAX_OFFLINE=0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        created.append(str(env_example))

    head("Workspace ready")
    for path in (cfg.memory_dir, cfg.memory_entries_dir, cfg.reports_dir, cfg.state_dir):
        kv(path.name, str(path))
    for path in created:
        typer.echo(f"  created {path}")
    typer.echo(f"\n  {len(list(cfg.skills_dir.glob('*/SKILL.md')))} skills in {cfg.skills_dir}")
    typer.echo("\n  next: parallax doctor")


def _print_sources() -> None:
    from .provenance import registry

    citations = registry().all()
    if not citations:
        return
    head(f"Sources ({len(citations)})")
    for citation in citations[:14]:
        typer.echo(_c(f"  {citation.key}", CYAN))
        typer.echo(_c(f"    {citation.reference()[:150]}", DIM))
    if len(citations) > 14:
        typer.echo(_c(f"  … {len(citations) - 14} more — `parallax cite` for the full list", DIM))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
