"""The tool surface the agent reasons with.

One definition, three consumers: the in-process SDK server (`parallax ask`), the
stdio MCP server (`parallax mcp`, for Claude Code and the desktop app) and the
CLI. Handlers are ordinary functions returning JSON-serialisable dicts, and
they never raise at the boundary — a failure comes back as a structured error
the agent can reason about and route around.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Any, Callable

from ..config import settings
from ..provenance import registry, verify_doi, verify_registry

Handler = Callable[..., dict[str, Any]]


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Handler
    group: str = "general"
    read_only: bool = True


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


STR = {"type": "string"}
INT = {"type": "integer"}
NUM = {"type": "number"}
BOOL = {"type": "boolean"}


def _str_list(description: str) -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}, "description": description}


# =====================================================================
# structure and identity
# =====================================================================


def resolve_compound(query: str) -> dict[str, Any]:
    from ..chem import resolve

    structure = resolve(query)
    return structure.as_dict()


def compound_profile(query: str, include_measured: bool = True, include_bioactivity: bool = True) -> dict[str, Any]:
    from ..chem import resolve
    from ..chem.descriptors import profile
    from ..safety import screen
    from ..sources import chembl, pubchem

    structure = resolve(query)
    out: dict[str, Any] = {
        "identity": structure.as_dict(),
        "safety_screen": screen(smiles=structure.smiles, text=query).as_dict(),
        "computed": profile(structure.smiles),
    }

    cid = structure.identifiers.get("pubchem_cid")
    if include_measured and cid:
        try:
            measured = pubchem.experimental_properties(int(cid))
            out["measured_properties"] = {
                "tier": "measured — experimental values with their source references",
                "n_values": len(measured["properties"]),
                "values": measured["properties"][:40],
                "citations": measured["citations"][:40],
            }
        except Exception as exc:
            out["measured_properties"] = {"error": str(exc)}

    if include_bioactivity:
        try:
            found = chembl.search_molecule(structure.name or query, limit=1)
            if found["results"]:
                record = found["results"][0]
                out["chembl_record"] = record
                chembl_id = record["chembl_id"]
                out["mechanisms"] = chembl.mechanisms(chembl_id)
                activities = chembl.activities(molecule_chembl_id=chembl_id, limit=60)
                by_target: dict[str, list[dict[str, Any]]] = {}
                for row in activities["rows"]:
                    if row.get("pchembl_value") is None:
                        continue
                    by_target.setdefault(row.get("target_name") or "unknown", []).append(row)
                out["measured_bioactivity"] = {
                    "tier": "measured — each value cites the paper that reported it",
                    "n_rows": activities["count"],
                    "by_target": {
                        target: {
                            "n": len(rows),
                            "best_pchembl": max(r["pchembl_value"] for r in rows),
                            "examples": [
                                {
                                    "type": r["standard_type"],
                                    "value_nM": r["standard_value"],
                                    "pchembl": r["pchembl_value"],
                                    "assay": (r.get("assay_description") or "")[:140],
                                    "citation": r.get("citation"),
                                }
                                for r in sorted(rows, key=lambda x: -x["pchembl_value"])[:3]
                            ],
                        }
                        for target, rows in sorted(
                            by_target.items(), key=lambda kv: -max(r["pchembl_value"] for r in kv[1])
                        )[:10]
                    },
                }
        except Exception as exc:
            out["chembl_record"] = {"error": str(exc)}

    return out


def computed_properties(smiles: str) -> dict[str, Any]:
    from ..chem.descriptors import profile

    return profile(smiles)


def measured_properties(query: str) -> dict[str, Any]:
    from ..chem import resolve
    from ..sources import pubchem

    structure = resolve(query)
    cid = structure.identifiers.get("pubchem_cid")
    if not cid:
        return {
            "query": query,
            "error": "no PubChem record, so no annotated experimental data",
            "advice": "use computed_properties, and search the literature for measured values",
        }
    result = pubchem.experimental_properties(int(cid))
    return {
        "query": query,
        "smiles": structure.smiles,
        "tier": "measured",
        "n_values": len(result["properties"]),
        "values": result["properties"],
        "citations": result["citations"],
    }


def similarity_search(smiles: str, source: str = "chembl", threshold: int = 70, limit: int = 30) -> dict[str, Any]:
    from ..sources import chembl, pubchem

    if source == "pubchem":
        return pubchem.similar(smiles, threshold=max(threshold, 80), limit=limit)
    return chembl.similarity(smiles, threshold=threshold, limit=limit)


def substructure_search(query: str, limit: int = 30) -> dict[str, Any]:
    from ..sources import pubchem

    return pubchem.substructure(query, limit=limit)


# =====================================================================
# bioactivity
# =====================================================================


def chembl_find_target(query: str, organism: str | None = "Homo sapiens", limit: int = 8) -> dict[str, Any]:
    from ..sources import chembl

    return chembl.search_target(query, limit=limit, organism=organism)


def chembl_activities(
    target: str | None = None,
    compound: str | None = None,
    standard_type: str | None = None,
    assay_type: str | None = None,
    pchembl_min: float | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    from ..sources import chembl

    target_id = target
    if target and not target.upper().startswith("CHEMBL"):
        found = chembl.search_target(target, limit=1)
        if not found["targets"]:
            return {"error": f"no ChEMBL target matched {target!r}"}
        target_id = found["targets"][0]["target_chembl_id"]

    compound_id = compound
    if compound and not compound.upper().startswith("CHEMBL"):
        found = chembl.search_molecule(compound, limit=1)
        if not found["results"]:
            return {"error": f"no ChEMBL compound matched {compound!r}"}
        compound_id = found["results"][0]["chembl_id"]

    result = chembl.activities(
        target_chembl_id=target_id,
        molecule_chembl_id=compound_id,
        standard_type=standard_type,
        assay_type=assay_type,
        pchembl_min=pchembl_min,
        limit=min(limit, settings().max_tool_rows * 4),
    )
    rows = result["rows"]
    if len(rows) > settings().max_tool_rows:
        result["rows"] = rows[: settings().max_tool_rows]
        result["truncated"] = {
            "shown": settings().max_tool_rows,
            "available": len(rows),
            "note": "use a narrower standard_type or a pchembl_min to focus the set",
        }
    return result


def chembl_mechanisms(compound: str) -> dict[str, Any]:
    from ..sources import chembl

    chembl_id = compound
    if not compound.upper().startswith("CHEMBL"):
        found = chembl.search_molecule(compound, limit=1)
        if not found["results"]:
            return {"error": f"no ChEMBL compound matched {compound!r}"}
        chembl_id = found["results"][0]["chembl_id"]
    return chembl.mechanisms(chembl_id)


def pubchem_assays(query: str, limit: int = 60) -> dict[str, Any]:
    from ..chem import resolve
    from ..sources import pubchem

    structure = resolve(query)
    cid = structure.identifiers.get("pubchem_cid")
    if not cid:
        return {"error": "no PubChem CID for this structure"}
    return pubchem.assay_summary(int(cid), limit=limit)


# =====================================================================
# literature and patents
# =====================================================================


def literature_search(
    query: str,
    limit: int = 15,
    year_from: int | None = None,
    include_patents: bool = False,
    source: str = "europepmc",
) -> dict[str, Any]:
    from ..sources import crossref, europepmc, openalex

    if source == "crossref":
        return crossref.search(query, limit=limit, year_from=year_from)
    if source == "openalex":
        return openalex.search(query, limit=limit, year_from=year_from)
    return europepmc.search(query, limit=limit, year_from=year_from, include_patents=include_patents)


def patent_search(query: str, limit: int = 15, year_from: int | None = None) -> dict[str, Any]:
    from ..sources import patents

    return patents.search(query, limit=limit, year_from=year_from)


def patents_for_compound(query: str, limit: int = 25) -> dict[str, Any]:
    from ..sources import patents

    return patents.for_compound(query, limit=limit)


def read_full_text(pmcid: str, max_chars: int = 40000) -> dict[str, Any]:
    from ..sources import europepmc

    return europepmc.full_text(pmcid, max_chars=max_chars)


def reaction_precedent_search(
    query: str, limit: int = 12, include_patents: bool = True, year_from: int | None = None
) -> dict[str, Any]:
    """Look for real reported examples of a transformation."""
    from ..sources import europepmc, reaxys

    licensed = reaxys.reactions(query=query, limit=limit)
    journals = europepmc.search(query, limit=limit, include_patents=False, year_from=year_from)
    patent_hits = (
        europepmc.search(query, limit=max(limit // 2, 4), patents_only=True, year_from=year_from)
        if include_patents
        else {"results": [], "citations": [], "hit_count": 0}
    )
    found = len(journals["results"]) + len(patent_hits["results"]) + len(licensed.get("results", []))
    return {
        "query": query,
        "licensed_database": {
            "available": licensed.get("available", False),
            "results": licensed.get("results", []),
            "note": licensed.get("fallback") or licensed.get("mode"),
        },
        "journal_precedents": journals["results"],
        "patent_precedents": patent_hits["results"],
        "n_precedents_found": found,
        "verdict": (
            "precedented — at least one report describes this transformation in a related context"
            if found
            else "no precedent found in the free sources searched; this does not prove the step is "
            "unknown (Reaxys/SciFinder coverage of preparative conditions is deeper), but it "
            "must be reported as unprecedented-as-far-as-checked"
        ),
        "citations": sorted(
            set(journals["citations"] + patent_hits["citations"] + licensed.get("citations", []))
        ),
    }


def reaxys_status() -> dict[str, Any]:
    from ..sources import reaxys

    return reaxys.availability()


def reaxys_reactions(
    product_smiles: str | None = None,
    reactant_smiles: str | None = None,
    query: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    from ..sources import reaxys

    return reaxys.reactions(
        product_smiles=product_smiles, reactant_smiles=reactant_smiles, query=query, limit=limit
    )


def cas_lookup(query: str) -> dict[str, Any]:
    from ..sources import reaxys

    return reaxys.cas_lookup(query)


# =====================================================================
# modelling
# =====================================================================


def train_potency_model(
    target: str,
    endpoint: str = "inhibition",
    split: str = "scaffold",
    name: str | None = None,
    max_records: int = 3000,
    standard_types: list[str] | None = None,
    assay_type: str | None = None,
) -> dict[str, Any]:
    from ..memory import remember
    from ..models import train

    result = train(
        target,
        endpoint=endpoint,
        split=split,
        name=name,
        max_records=max_records,
        standard_types=standard_types,
        assay_type=assay_type,
    )
    card = result["card"]
    # A model's honest performance is exactly the kind of thing worth remembering.
    try:
        remember(
            type="model",
            title=f"QSAR model for {card['target_name'] or card['target_chembl_id']} ({card['endpoint_label']})",
            body=(
                f"{result['summary']}\n\n"
                f"Trained on {card['n_compounds']} compounds from {card['source_documents']} "
                f"source documents. Baseline (predicting the training mean) RMSE is "
                f"{card['baseline_metrics']['rmse']} log units, so the model adds "
                f"{round(card['baseline_metrics']['rmse'] - card['metrics']['rmse'], 2)} log units "
                f"of signal. Model name for prediction: `{card['name']}`."
            ),
            tags=["model", "qsar", endpoint, card["target_chembl_id"].lower()],
            citations=card["citations"][:30],
            entities={"targets": [card["target_chembl_id"]]},
            confidence=0.8,
        )
    except Exception:
        pass
    return result


def predict_potency(model: str, smiles: list[str] | str) -> dict[str, Any]:
    from ..models import predict

    return predict(model, smiles)


def list_potency_models() -> dict[str, Any]:
    from ..models import list_models

    cards = list_models()
    return {
        "n_models": len(cards),
        "models": [
            {
                "name": c["name"],
                "target": c.get("target_name") or c["target_chembl_id"],
                "endpoint": c["endpoint_label"],
                "algorithm": c["algorithm"],
                "n_compounds": c["n_compounds"],
                "split": c["split"],
                "test_r2": c["metrics"].get("r2"),
                "test_rmse": c["metrics"].get("rmse"),
                "created": c["created"],
            }
            for c in cards
        ],
        "note": "train_potency_model builds a new one; models are stored in .parallax/models",
    }


def matched_molecular_pairs(
    target: str, endpoint: str = "inhibition", limit: int = 1200, min_pairs: int = 2
) -> dict[str, Any]:
    """MMPA over measured activity data for a target."""
    from ..chem.analogs import matched_pairs
    from ..models.qsar import ENDPOINT_PRESETS, aggregate_measurements
    from ..sources import chembl

    preset = ENDPOINT_PRESETS.get(endpoint, ENDPOINT_PRESETS["inhibition"])
    target_id = target
    target_name = None
    if not target.upper().startswith("CHEMBL"):
        found = chembl.search_target(target, limit=1)
        if not found["targets"]:
            return {"error": f"no ChEMBL target matched {target!r}"}
        target_id = found["targets"][0]["target_chembl_id"]
        target_name = found["targets"][0]["pref_name"]

    rows: list[dict[str, Any]] = []
    for standard_type in preset["standard_types"]:
        result = chembl.activities(
            target_chembl_id=target_id,
            standard_type=standard_type,
            assay_type=preset["assay_type"],
            limit=limit,
        )
        rows.extend(result["rows"])

    aggregated = list(aggregate_measurements(rows).values())
    if len(aggregated) < 10:
        return {
            "error": f"only {len(aggregated)} compounds with usable data — too few for MMPA",
            "target": target_id,
        }
    analysis = matched_pairs(aggregated, min_pairs_per_transform=min_pairs)
    analysis["target"] = target_name or target_id
    analysis["endpoint"] = preset["label"]
    return analysis


# =====================================================================
# synthesis
# =====================================================================


def retro_disconnect(smiles: str, top: int = 8) -> dict[str, Any]:
    from ..chem.retro import summarize_disconnections

    return summarize_disconnections(smiles, top=top)


def retro_plan(smiles: str, max_depth: int = 3, beam: int = 3, max_routes: int = 4) -> dict[str, Any]:
    from ..chem.retro import plan

    return plan(smiles, max_depth=max_depth, beam=beam, max_routes=max_routes)


def synthesis_feasibility(smiles: str, route: dict[str, Any] | None = None) -> dict[str, Any]:
    from ..chem.retro import feasibility

    return feasibility(smiles, route=route)


def validate_reaction(reaction_smiles: str) -> dict[str, Any]:
    from ..chem.reactions import validate_reaction_smiles

    return validate_reaction_smiles(reaction_smiles)


def reaction_library() -> dict[str, Any]:
    from ..chem.reactions import library_summary

    templates = library_summary()
    return {
        "n_templates": len(templates),
        "templates": templates,
        "note": (
            "Reliability figures are coarse priors for ranking disconnections, not success "
            "probabilities. Use reaction_precedent_search for evidence about a specific step."
        ),
    }


def starting_material_check(smiles: str) -> dict[str, Any]:
    from ..chem.retro import availability

    available, reason, citation = availability(smiles)
    return {
        "smiles": smiles,
        "likely_available": available,
        "assessment": reason,
        "citation": citation,
        "caveat": (
            "This is a proxy (PubChem presence plus simplicity), not a supplier quote. "
            "Confirm against a catalogue before planning around it."
        ),
    }


# =====================================================================
# design
# =====================================================================


def design_analogs(
    smiles: str, categories: list[str] | None = None, max_total: int = 40
) -> dict[str, Any]:
    from ..chem.analogs import enumerate_analogs

    return enumerate_analogs(smiles, categories=categories, max_total=max_total)


def design_macrocycles(
    smiles: str, min_ring: int = 11, max_ring: int = 16, max_designs: int = 15
) -> dict[str, Any]:
    from ..chem.analogs import suggest_macrocyclization

    return suggest_macrocyclization(
        smiles, min_ring=min_ring, max_ring=max_ring, max_designs=max_designs
    )


def score_designs(
    smiles_list: list[str],
    model: str | None = None,
    max_sa_score: float = 6.0,
) -> dict[str, Any]:
    """Property-filter and (optionally) potency-rank a set of designs."""
    from ..chem.descriptors import physchem
    from ..models import predict

    scored: list[dict[str, Any]] = []
    predictions: dict[str, dict[str, Any]] = {}
    if model:
        try:
            result = predict(model, smiles_list)
            for item in result["predictions"]:
                predictions[item["query"]] = item
        except Exception as exc:
            predictions = {}
            model_error = str(exc)
        else:
            model_error = None
    else:
        model_error = None

    for smiles in smiles_list:
        try:
            values = physchem(smiles)["values"]
        except Exception as exc:
            scored.append({"smiles": smiles, "error": str(exc)})
            continue
        prediction = predictions.get(smiles)
        flags = []
        if values["sa_score"] > max_sa_score:
            flags.append(f"SA score {values['sa_score']} above the {max_sa_score} cut-off")
        if values["mw"] > 600:
            flags.append(f"MW {values['mw']} is high")
        if values["clogp"] > 5:
            flags.append(f"cLogP {values['clogp']} is high")
        scored.append(
            {
                "smiles": smiles,
                "properties": {
                    k: values[k]
                    for k in ("mw", "clogp", "tpsa", "hbd", "rotatable_bonds", "qed", "sa_score", "is_macrocycle")
                },
                "predicted": (
                    {
                        "value": prediction["predicted_value"],
                        "endpoint": prediction["endpoint"],
                        "uncertainty_log_units": prediction["uncertainty_log_units"],
                        "applicability_domain": prediction["applicability_domain"]["verdict"],
                        "in_training_set": prediction.get("in_training_set"),
                    }
                    if prediction
                    else None
                ),
                "flags": flags,
            }
        )

    if model and predictions:
        scored.sort(
            key=lambda row: (row.get("predicted") or {}).get("value") or -99, reverse=True
        )

    return {
        "n_designs": len(scored),
        "model_used": model,
        "model_error": model_error,
        "designs": scored,
        "how_to_read": (
            "Ranked by predicted potency when a model is given. Differences smaller than the "
            "model's RMSE are noise, and designs flagged outside the applicability domain are "
            "ordering hints only."
        ),
    }


# =====================================================================
# memory
# =====================================================================


def memory_recall(
    query: str, k: int = 6, types: list[str] | None = None, tags: list[str] | None = None
) -> dict[str, Any]:
    from ..memory import recall

    return recall(query, k=k, types=types, tags=tags)


def memory_remember(
    type: str,
    title: str,
    body: str,
    tags: list[str] | None = None,
    citations: list[str] | None = None,
    confidence: float = 0.6,
    entities: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    from ..memory import remember

    return remember(
        type=type,
        title=title,
        body=body,
        tags=tags or [],
        citations=citations or [],
        confidence=confidence,
        entities=entities or {},
    )


def memory_update(entry_id: str, action: str, note: str = "", citations: list[str] | None = None) -> dict[str, Any]:
    from ..memory import store

    memory = store()
    if action == "reinforce":
        return memory.reinforce(entry_id, note=note, citations=citations or [])
    if action == "contradict":
        if not note:
            return {"error": "contradicting an entry requires a note explaining the conflict"}
        return memory.contradict(entry_id, note=note)
    if action == "forget":
        return memory.forget(entry_id)
    return {"error": f"unknown action {action!r}; use reinforce, contradict or forget"}


def memory_consolidate() -> dict[str, Any]:
    from ..memory import consolidate

    return consolidate()


def memory_stats() -> dict[str, Any]:
    from ..memory import stats

    return stats()


# =====================================================================
# citations, safety, reporting
# =====================================================================


def citation_bibliography(format: str = "markdown", keys: list[str] | None = None) -> dict[str, Any]:
    reg = registry()
    if format == "bibtex":
        body = reg.to_bibtex(keys)
    elif format == "json":
        body = reg.to_json()
    else:
        body = reg.to_markdown(keys)
    result = {"format": format, "n_citations": len(reg), "bibliography": body}
    if keys:
        # Silently dropping a requested key would hide exactly the mistake that
        # matters: citing something no tool ever returned.
        unknown = [k for k in keys if k not in set(reg.keys())]
        if unknown:
            result["unknown_keys"] = unknown
            result["warning"] = (
                f"{len(unknown)} requested key(s) are not in the registry, so no tool "
                "produced them. They are omitted from the bibliography above and must not "
                "appear in your text either."
            )
    return result


def audit_draft(text: str) -> dict[str, Any]:
    """Catch citations that were written from memory rather than retrieved."""
    from ..provenance import audit_text

    return audit_text(text)


def check_doi(doi: str) -> dict[str, Any]:
    return verify_doi(doi)


def verify_all_citations() -> dict[str, Any]:
    result = verify_registry()
    result["interpretation"] = (
        "Every DOI in this run's citation registry was checked against Crossref. "
        + (
            f"{len(result['failed'])} did not resolve and must not be cited."
            if result["failed"]
            else "All resolved."
        )
    )
    return result


def safety_screen(smiles: str | None = None, text: str | None = None, mode: str = "analysis") -> dict[str, Any]:
    from ..safety import screen

    return screen(smiles=smiles, text=text, mode=mode).as_dict()


def write_report(
    title: str,
    markdown: str,
    molecules: list[dict[str, str]] | None = None,
    filename: str | None = None,
    include_bibliography: bool = True,
) -> dict[str, Any]:
    from ..report import render_report

    return render_report(
        title=title,
        markdown=markdown,
        molecules=molecules or [],
        filename=filename,
        include_bibliography=include_bibliography,
    )



def selectivity_profile(compound: str, primary_target: str | None = None, limit: int = 400) -> dict[str, Any]:
    """Measured activity across every target, with selectivity ratios."""
    from ..sources import chembl

    chembl_id = compound
    if not compound.upper().startswith("CHEMBL"):
        found = chembl.search_molecule(compound, limit=1)
        if not found["results"]:
            return {"error": f"no ChEMBL compound matched {compound!r}"}
        chembl_id = found["results"][0]["chembl_id"]
        name = found["results"][0].get("pref_name") or compound
    else:
        name = compound

    result = chembl.activities(molecule_chembl_id=chembl_id, limit=limit)
    by_target: dict[str, dict[str, Any]] = {}
    for row in result["rows"]:
        if row.get("pchembl_value") is None or not row.get("target_name"):
            continue
        entry = by_target.setdefault(
            row["target_name"],
            {
                "target": row["target_name"],
                "target_chembl_id": row.get("target_chembl_id"),
                "values": [],
                "assay_types": set(),
                "citations": set(),
            },
        )
        entry["values"].append(float(row["pchembl_value"]))
        if row.get("assay_type"):
            entry["assay_types"].add(row["assay_type"])
        if row.get("citation"):
            entry["citations"].add(row["citation"])

    if not by_target:
        return {
            "compound": name,
            "error": "no pChEMBL-quantified activity found for this compound",
            "advice": "try pubchem_assays for broader screening data",
        }

    rows = []
    for entry in by_target.values():
        values = entry["values"]
        rows.append(
            {
                "target": entry["target"],
                "target_chembl_id": entry["target_chembl_id"],
                "best_pchembl": round(max(values), 2),
                "median_pchembl": round(sorted(values)[len(values) // 2], 2),
                "n_measurements": len(values),
                "assay_formats": sorted(entry["assay_types"]),
                "citations": sorted(entry["citations"])[:5],
            }
        )
    rows.sort(key=lambda r: r["best_pchembl"], reverse=True)

    anchor = rows[0]
    if primary_target:
        match = [r for r in rows if primary_target.lower() in (r["target"] or "").lower()
                 or primary_target.upper() == (r["target_chembl_id"] or "")]
        if match:
            anchor = match[0]

    for row in rows:
        delta = anchor["best_pchembl"] - row["best_pchembl"]
        row["log_selectivity_vs_primary"] = round(delta, 2)
        row["fold_selectivity_vs_primary"] = round(10 ** delta, 1) if delta >= 0 else None

    off_targets = [r for r in rows if r["target"] != anchor["target"]]
    close = [r for r in off_targets if r["log_selectivity_vs_primary"] < 1.0]

    return {
        "compound": name,
        "chembl_id": chembl_id,
        "primary_target": anchor["target"],
        "primary_pchembl": anchor["best_pchembl"],
        "n_targets_with_data": len(rows),
        "targets": rows[:40],
        "off_targets_within_10x": [
            {"target": r["target"], "fold": r["fold_selectivity_vs_primary"]} for r in close
        ],
        "assessment": (
            f"{len(close)} target(s) are within 10-fold of the primary — selectivity is a "
            "live liability here"
            if close
            else "no profiled off-target within 10-fold, on the data available"
        ),
        "caveat": (
            "Selectivity is only as broad as what has been tested. ChEMBL coverage is "
            "opportunistic, not a panel: absence of an off-target here is not evidence of "
            "selectivity against it. For kinases, ask for a published kinome panel."
        ),
        "citations": sorted({c for r in rows for c in r["citations"]})[:60],
    }


def multiparameter_score(
    candidates: list[dict[str, Any]] | list[str],
    profile: str = "oral_small_molecule",
    model: str | None = None,
) -> dict[str, Any]:
    """Rank candidates on a weighted multi-objective desirability profile."""
    from ..chem.optimisation import score_candidates

    records: list[dict[str, Any]] = []
    for candidate in candidates:
        records.append({"smiles": candidate} if isinstance(candidate, str) else dict(candidate))

    if model:
        from ..models import predict

        try:
            prediction = predict(model, [r["smiles"] for r in records])
            lookup = {p["query"]: p for p in prediction["predictions"]}
            for record in records:
                hit = lookup.get(record["smiles"])
                if hit and record.get("potency") is None:
                    record["potency"] = hit["predicted_value"]
                    record["potency_source"] = f"predicted by {model}"
                    record["applicability_domain"] = hit["applicability_domain"]["verdict"]
        except Exception as exc:
            return {"error": f"model {model!r} failed: {exc}"}

    return score_candidates(records, profile=profile)


def optimisation_profiles() -> dict[str, Any]:
    """List the available multi-parameter optimisation profiles."""
    from ..chem.optimisation import list_profiles

    return list_profiles()


def covalent_check(smiles: str) -> dict[str, Any]:
    """Detect covalent warheads and what they imply for assays and modelling."""
    from ..chem.optimisation import covalent_profile

    return covalent_profile(smiles)


# =====================================================================
# registry
# =====================================================================

TOOLS: list[Tool] = [
    Tool(
        "resolve_compound",
        "Resolve a chemical name, SMILES, InChI, InChIKey, CAS number, PubChem CID or ChEMBL ID "
        "into a standardised structure with a citation to the record it came from. Use this "
        "first whenever the user names a compound — never recall a structure from memory.",
        _schema({"query": STR | {"description": "name, SMILES, InChI, InChIKey, CAS, CID or ChEMBL ID"}}, ["query"]),
        resolve_compound,
        group="structure",
    ),
    Tool(
        "compound_profile",
        "Full picture of one compound: resolved identity, computed properties, published "
        "experimental values with references, ChEMBL record, mechanism of action, and measured "
        "bioactivity per target with a citation on every number. The best single starting call "
        "for 'tell me about compound X'.",
        _schema(
            {
                "query": STR,
                "include_measured": BOOL | {"description": "fetch PubChem experimental annotations (default true)"},
                "include_bioactivity": BOOL | {"description": "fetch ChEMBL activity and mechanism data (default true)"},
            },
            ["query"],
        ),
        compound_profile,
        group="structure",
    ),
    Tool(
        "computed_properties",
        "Deterministic physicochemical profile from the structure: MW, cLogP, TPSA, HBD/HBA, "
        "rotatable bonds, Fsp3, QED, SA score, NP-likeness, macrocycle detection, drug-likeness "
        "rule sets (Lipinski/Veber/Ghose/Egan/Pfizer 3-75/GSK 4-400), CNS MPO, PAINS and Brenk "
        "alerts, and liability flags. Each value cites its method paper.",
        _schema({"smiles": STR}, ["smiles"]),
        computed_properties,
        group="properties",
    ),
    Tool(
        "measured_properties",
        "Experimentally measured properties for a compound from PubChem's annotated data "
        "(melting point, solubility, logP, stability and so on), each with the handbook or paper "
        "that reported it. Prefer these over computed values whenever they exist.",
        _schema({"query": STR}, ["query"]),
        measured_properties,
        group="properties",
    ),
    Tool(
        "similarity_search",
        "Find structurally similar compounds in ChEMBL (default) or PubChem, with citations. "
        "Use to find an existing series around a hit, or to check whether a design is novel.",
        _schema(
            {
                "smiles": STR,
                "source": STR | {"enum": ["chembl", "pubchem"]},
                "threshold": INT | {"description": "percent Tanimoto, default 70 (ChEMBL) / 90 (PubChem)"},
                "limit": INT,
            },
            ["smiles"],
        ),
        similarity_search,
        group="structure",
    ),
    Tool(
        "substructure_search",
        "PubChem substructure search from a SMILES or SMARTS query — use to find compounds "
        "sharing a scaffold.",
        _schema({"query": STR, "limit": INT}, ["query"]),
        substructure_search,
        group="structure",
    ),
    Tool(
        "chembl_find_target",
        "Resolve a target name (e.g. 'ALK', 'EGFR', 'CDK9') to ChEMBL target IDs with organism "
        "and UniProt accession. Do this before pulling activity data so you know which target "
        "record you are using.",
        _schema({"query": STR, "organism": STR, "limit": INT}, ["query"]),
        chembl_find_target,
        group="bioactivity",
    ),
    Tool(
        "chembl_activities",
        "Measured bioactivity rows from ChEMBL for a target and/or compound, each carrying the "
        "DOI of the paper or patent that reported it. Filter by standard_type (IC50, Ki, Kd, "
        "EC50), assay_type (B = binding/biochemical, F = functional/cell) and a minimum pChEMBL. "
        "This is the primary source of real potency data.",
        _schema(
            {
                "target": STR | {"description": "target name or CHEMBL id"},
                "compound": STR | {"description": "compound name or CHEMBL id"},
                "standard_type": STR | {"description": "IC50 | Ki | Kd | EC50 | ..."},
                "assay_type": STR | {"enum": ["B", "F", "A", "T", "P", "U"]},
                "pchembl_min": NUM,
                "limit": INT,
            },
        ),
        chembl_activities,
        group="bioactivity",
    ),
    Tool(
        "chembl_mechanisms",
        "Mechanism of action records for a drug or clinical compound, with supporting references.",
        _schema({"compound": STR}, ["compound"]),
        chembl_mechanisms,
        group="bioactivity",
    ),
    Tool(
        "pubchem_assays",
        "PubChem BioAssay summary for a compound — broad screening data with AID-level "
        "provenance. Useful for selectivity and off-target context.",
        _schema({"query": STR, "limit": INT}, ["query"]),
        pubchem_assays,
        group="bioactivity",
    ),
    Tool(
        "literature_search",
        "Search the scientific literature (Europe PMC by default, which covers MEDLINE, PMC "
        "full text and preprints; or Crossref/OpenAlex). Returns abstracts and citation keys. "
        "Use for SAR, synthesis, mechanism and property evidence.",
        _schema(
            {
                "query": STR,
                "limit": INT,
                "year_from": INT,
                "include_patents": BOOL,
                "source": STR | {"enum": ["europepmc", "crossref", "openalex"]},
            },
            ["query"],
        ),
        literature_search,
        group="literature",
    ),
    Tool(
        "patent_search",
        "Search patent abstracts (EPO index via Europe PMC). Most medicinal-chemistry SAR "
        "appears in patents before papers, so check here for series and route information.",
        _schema({"query": STR, "limit": INT, "year_from": INT}, ["query"]),
        patent_search,
        group="literature",
    ),
    Tool(
        "patents_for_compound",
        "Patents linked to a specific structure, anchored through PubChem's compound-patent "
        "cross-references plus a text search. Use to find the originator filing for a compound.",
        _schema({"query": STR, "limit": INT}, ["query"]),
        patents_for_compound,
        group="literature",
    ),
    Tool(
        "read_full_text",
        "Read the open-access full text of a paper by PMCID — use when an abstract is not "
        "enough, for example to extract reaction conditions or an assay protocol.",
        _schema({"pmcid": STR, "max_chars": INT}, ["pmcid"]),
        read_full_text,
        group="literature",
    ),
    Tool(
        "reaction_precedent_search",
        "Look for real reported examples of a transformation, across a licensed reaction "
        "database (if configured), journals and patents. Call this on every synthetic step you "
        "propose: it is what turns a template guess into a precedented step, and it tells you "
        "plainly when no precedent was found.",
        _schema({"query": STR, "limit": INT, "include_patents": BOOL, "year_from": INT}, ["query"]),
        reaction_precedent_search,
        group="synthesis",
    ),
    Tool(
        "reaxys_status",
        "Report whether licensed reaction-database access (Reaxys, CAS) is configured in this "
        "workspace, and how to configure it. Call this before telling a user what is or is not "
        "in Reaxys.",
        _schema({}),
        reaxys_status,
        group="synthesis",
    ),
    Tool(
        "reaxys_reactions",
        "Query Reaxys for preparative reaction precedents by product or reactant structure. "
        "Requires licensed access; returns an explicit 'unavailable' result with the public "
        "fallback when none is configured.",
        _schema(
            {"product_smiles": STR, "reactant_smiles": STR, "query": STR, "limit": INT},
        ),
        reaxys_reactions,
        group="synthesis",
    ),
    Tool(
        "cas_lookup",
        "CAS Common Chemistry lookup for an authoritative CAS Registry Number and name "
        "(requires CAS_API_KEY; falls back to PubChem synonyms).",
        _schema({"query": STR}, ["query"]),
        cas_lookup,
        group="structure",
    ),
    Tool(
        "train_potency_model",
        "Build a QSAR model for a target from ChEMBL data, right now, and report its honest "
        "out-of-distribution performance. endpoint: 'inhibition' (IC50, biochemical), 'binding' "
        "(Ki/Kd), 'activation' (EC50, functional/cellular response), 'cell_potency', or "
        "'any_potency', 'antimicrobial' (MIC), 'cytotoxicity', 'binding_affinity_relative' "
        "(RBA) or 'admet'. For anything else, pass standard_types explicitly — ANY ChEMBL "
        "standard_type can be modelled, and values in ug/mL are converted to molar pX using "
        "molecular weight. split: 'scaffold' (default, pessimistic and useful), 'time' "
        "(chronological, closest to prospective) or 'random' (optimistic). Check which "
        "endpoints a target actually has with chembl_activities before assuming a preset fits.",
        _schema(
            {
                "target": STR | {"description": "target name or CHEMBL id"},
                "endpoint": STR | {"enum": ["inhibition", "binding", "activation", "cell_potency", "any_potency", "antimicrobial", "cytotoxicity", "binding_affinity_relative", "admet"]},
                "standard_types": _str_list("explicit ChEMBL standard_type(s), overriding the preset — e.g. ['MIC'] or ['Papp']"),
                "assay_type": STR | {"enum": ["B", "F", "A", "T", "P", "U"]},
                "split": STR | {"enum": ["scaffold", "time", "random"]},
                "name": STR,
                "max_records": INT,
            },
            ["target"],
        ),
        train_potency_model,
        group="modelling",
        read_only=False,
    ),
    Tool(
        "predict_potency",
        "Predict potency or activation for one or more structures with a trained model. Returns "
        "the value, an uncertainty band, an applicability-domain verdict, the nearest measured "
        "training compounds, and a flag if the structure was in the training set. Never quote a "
        "prediction without its applicability-domain verdict.",
        _schema(
            {
                "model": STR | {"description": "model name from train_potency_model or list_potency_models"},
                "smiles": {
                    "anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}],
                    "description": "one SMILES or a list",
                },
            },
            ["model", "smiles"],
        ),
        predict_potency,
        group="modelling",
    ),
    Tool(
        "list_potency_models",
        "List models already trained in this workspace, with their performance and endpoints.",
        _schema({}),
        list_potency_models,
        group="modelling",
    ),
    Tool(
        "matched_molecular_pairs",
        "Matched molecular pair analysis over measured data for a target: which single "
        "substituent changes move potency, by how much, how consistently, and with which "
        "citations. This is evidence from real SAR rather than a model's opinion.",
        _schema(
            {
                "target": STR,
                "endpoint": STR | {"enum": ["inhibition", "binding", "activation", "cell_potency", "any_potency"]},
                "limit": INT,
                "min_pairs": INT,
            },
            ["target"],
        ),
        matched_molecular_pairs,
        group="modelling",
    ),
    Tool(
        "retro_disconnect",
        "One-step retrosynthetic disconnections for a target, ranked by template priors and "
        "heuristics, including macrocyclisation ring-openings. Each carries conditions, what it "
        "tolerates, where it fails, chemoselectivity concerns and a literature query to find "
        "precedent.",
        _schema({"smiles": STR, "top": INT}, ["smiles"]),
        retro_disconnect,
        group="synthesis",
    ),
    Tool(
        "retro_plan",
        "Search multi-step routes back to plausibly available starting materials. Returns "
        "forward-ordered steps with conditions and concerns, the starting-material assessment, "
        "and a route score. Hypotheses only until each step is precedent-checked.",
        _schema({"smiles": STR, "max_depth": INT, "beam": INT, "max_routes": INT}, ["smiles"]),
        retro_plan,
        group="synthesis",
    ),
    Tool(
        "synthesis_feasibility",
        "Structured feasibility judgement for making a compound: synthetic accessibility, "
        "macrocyclisation difficulty by ring size, stereocentre burden, protecting-group demand, "
        "size, and — if a route is supplied — its weakest steps and precedent coverage.",
        _schema({"smiles": STR, "route": {"type": "object", "description": "a route object from retro_plan"}}, ["smiles"]),
        synthesis_feasibility,
        group="synthesis",
    ),
    Tool(
        "validate_reaction",
        "Check a reaction SMILES for parsability and atom balance. Run this on any reaction you "
        "write down — an unbalanced reaction is the commonest error in machine-written chemistry.",
        _schema({"reaction_smiles": STR}, ["reaction_smiles"]),
        validate_reaction,
        group="synthesis",
    ),
    Tool(
        "reaction_library",
        "List the reaction templates available for retrosynthesis, with classes, conditions and "
        "reliability priors.",
        _schema({}),
        reaction_library,
        group="synthesis",
    ),
    Tool(
        "starting_material_check",
        "Assess whether a structure is plausibly a catalogue compound. A proxy, not a supplier "
        "quote — the result says which test it passed.",
        _schema({"smiles": STR}, ["smiles"]),
        starting_material_check,
        group="synthesis",
    ),
    Tool(
        "design_analogs",
        "Generate analogs from a documented transform library: fluorine and methyl scans, "
        "aza-scan, acid bioisosteres, amide surrogates, metabolic blocking, solubilising groups, "
        "basicity removal. Each analog states the rationale, the expected property effect and "
        "the literature behind the strategy, with property deltas against the parent.",
        _schema(
            {
                "smiles": STR,
                "categories": _str_list(
                    "optional filter: bioisostere, metabolic blocking, potency probe, property tuning, "
                    "liability mitigation, permeability, scaffold hop"
                ),
                "max_total": INT,
            },
            ["smiles"],
        ),
        design_analogs,
        group="design",
    ),
    Tool(
        "design_macrocycles",
        "Propose macrocyclisations of an acyclic lead: finds atom pairs whose linking would "
        "close an 11-16 membered ring, builds each design, reports the rotatable-bond saving, "
        "properties and the closure chemistry needed. This is the rigidification strategy behind "
        "macrocyclic kinase inhibitors.",
        _schema({"smiles": STR, "min_ring": INT, "max_ring": INT, "max_designs": INT}, ["smiles"]),
        design_macrocycles,
        group="design",
    ),
    Tool(
        "score_designs",
        "Property-filter and potency-rank a list of candidate structures with a trained model. "
        "Returns properties, predictions with applicability domain, and flags.",
        _schema(
            {
                "smiles_list": _str_list("candidate structures"),
                "model": STR | {"description": "optional trained model name"},
                "max_sa_score": NUM,
            },
            ["smiles_list"],
        ),
        score_designs,
        group="design",
    ),
    Tool(
        "selectivity_profile",
        "Measured activity for one compound across every target ChEMBL has data for, with "
        "fold-selectivity against the primary target and a flag for any off-target within "
        "10-fold. Selectivity is a first-class question in drug discovery and this is the "
        "evidence-based answer to it — bounded by what has actually been tested.",
        _schema(
            {
                "compound": STR,
                "primary_target": STR | {"description": "anchor for the ratios; defaults to the most potent"},
                "limit": INT,
            },
            ["compound"],
        ),
        selectivity_profile,
        group="bioactivity",
    ),
    Tool(
        "multiparameter_score",
        "Rank candidates on several objectives at once — potency, lipophilicity, size, "
        "polarity, HBD, synthetic accessibility and structural alerts — using a weighted "
        "desirability profile ('oral_small_molecule', 'cns', 'lead_like', 'antimicrobial', "
        "'fragment'). Returns each candidate's score and, more usefully, the single "
        "limiting objective holding it back. Pass a model name to fill in predicted potency. "
        "Use this instead of ranking on potency alone, which hides the trade-offs.",
        _schema(
            {
                "candidates": {
                    "type": "array",
                    "items": {"anyOf": [{"type": "string"}, {"type": "object"}]},
                    "description": "SMILES strings, or {smiles, potency} objects",
                },
                "profile": STR | {"enum": ["oral_small_molecule", "cns", "lead_like", "antimicrobial", "fragment"]},
                "model": STR | {"description": "optional trained model to supply potency"},
            },
            ["candidates"],
        ),
        multiparameter_score,
        group="design",
    ),
    Tool(
        "optimisation_profiles",
        "List the multi-parameter optimisation profiles and the objectives each one weights.",
        _schema({}),
        optimisation_profiles,
        group="design",
    ),
    Tool(
        "covalent_check",
        "Detect covalent warheads (acrylamide, chloroacetamide, vinyl sulfone, epoxide, "
        "beta-lactam, nitrile, boronic acid, sulfonyl fluoride) and report what follows: "
        "potency becomes time-dependent, equilibrium IC50 models no longer apply, and "
        "glutathione reactivity becomes the key counter-screen. Run this before trusting any "
        "potency model on a compound bearing an electrophile.",
        _schema({"smiles": STR}, ["smiles"]),
        covalent_check,
        group="properties",
    ),
    Tool(
        "memory_recall",
        "Search what this workspace has learned before: findings, heuristics, failures, "
        "procedures, models, user preferences. Call this at the start of any non-trivial task — "
        "it is how the agent avoids repeating work and re-deriving conclusions.",
        _schema(
            {
                "query": STR,
                "k": INT,
                "types": _str_list("filter: finding, heuristic, sop, model, failure, preference, entity"),
                "tags": _str_list("filter by tag"),
            },
            ["query"],
        ),
        memory_recall,
        group="memory",
    ),
    Tool(
        "memory_remember",
        "Write a durable memory. Use 'finding' for a cited result about a specific compound or "
        "target, 'heuristic' for a generalisation, 'failure' for a dead end worth not repeating, "
        "'sop' for a procedure that worked, 'model' for a trained model, 'preference' for how "
        "this user wants work done, 'entity' for durable facts about a target or chemotype. "
        "Include citation keys for anything evidential. Near-duplicates are merged and "
        "corroborate the existing entry instead of piling up.",
        _schema(
            {
                "type": STR | {"enum": ["finding", "heuristic", "sop", "model", "failure", "preference", "entity"]},
                "title": STR | {"description": "a specific, self-contained claim"},
                "body": STR | {"description": "the detail, including why it is believed and any caveat"},
                "tags": _str_list("lowercase tags: target, chemotype, endpoint, tooling"),
                "citations": _str_list("citation keys supporting this"),
                "confidence": NUM | {"description": "0-1; be honest, low for single observations"},
                "entities": {"type": "object", "description": "e.g. {\"targets\": [\"CHEMBL4247\"]}"},
            },
            ["type", "title", "body"],
        ),
        memory_remember,
        group="memory",
        read_only=False,
    ),
    Tool(
        "memory_update",
        "Reinforce a memory that was confirmed, contradict one that new evidence undercuts "
        "(which halves its confidence and records why), or forget one that is simply wrong.",
        _schema(
            {
                "entry_id": STR,
                "action": STR | {"enum": ["reinforce", "contradict", "forget"]},
                "note": STR | {"description": "required when contradicting: what conflicts and how"},
                "citations": _str_list("new supporting citations"),
            },
            ["entry_id", "action"],
        ),
        memory_update,
        group="memory",
        read_only=False,
    ),
    Tool(
        "memory_consolidate",
        "Housekeeping pass: merge near-duplicate memories, decay stale unused ones, rewrite the "
        "human-readable index at memory/MEMORY.md. Worth running at the end of a long session.",
        _schema({}),
        memory_consolidate,
        group="memory",
        read_only=False,
    ),
    Tool(
        "memory_stats",
        "Size and health of the memory store: counts by type, mean confidence, contradicted "
        "entries, most-used entries.",
        _schema({}),
        memory_stats,
        group="memory",
    ),
    Tool(
        "citation_bibliography",
        "The bibliography for everything cited in this session, as markdown, BibTeX or JSON. "
        "Every analysis you report should end with this.",
        _schema({"format": STR | {"enum": ["markdown", "bibtex", "json"]}, "keys": _str_list("optional subset of citation keys")}),
        citation_bibliography,
        group="citations",
    ),
    Tool(
        "audit_draft",
        "Scan your finished text for citations no tool produced. verify_all_citations only "
        "checks the registry; this checks the prose, which is where a DOI recalled from "
        "background knowledge slips in. Reports every DOI and citation key in the text that "
        "is not backed by a tool result, and says whether each is fabricated (does not "
        "resolve) or merely unsourced (real, but you did not retrieve it). Run this on any "
        "report or analysis before delivering it.",
        _schema({"text": STR | {"description": "the draft you are about to deliver"}}, ["text"]),
        audit_draft,
        group="citations",
    ),
    Tool(
        "check_doi",
        "Verify one DOI against Crossref and return its real metadata. Use this before printing "
        "any DOI you did not get from a tool result — it is the guard against fabricated "
        "references.",
        _schema({"doi": STR}, ["doi"]),
        check_doi,
        group="citations",
    ),
    Tool(
        "verify_all_citations",
        "Check every DOI in this session's citation registry against Crossref and report any "
        "that do not resolve. Run before delivering a report.",
        _schema({}),
        verify_all_citations,
        group="citations",
    ),
    Tool(
        "safety_screen",
        "Screen a structure and/or a request against restricted chemistry classes (chemical "
        "weapons, energetic materials, controlled substances, acute toxins). Returns clear, "
        "caution or blocked with the matched category. Synthesis tools screen automatically; "
        "call this directly when you want the judgement up front.",
        _schema({"smiles": STR, "text": STR, "mode": STR | {"enum": ["analysis", "synthesis"]}}),
        safety_screen,
        group="safety",
    ),
    Tool(
        "write_report",
        "Render a standalone HTML report with molecule depictions and the full bibliography, "
        "saved into reports/. Give it your analysis as markdown and a list of "
        "{smiles, caption} molecules to depict.",
        _schema(
            {
                "title": STR,
                "markdown": STR | {"description": "the report body in markdown"},
                "molecules": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"smiles": STR, "caption": STR},
                        "required": ["smiles"],
                    },
                },
                "filename": STR,
                "include_bibliography": BOOL,
            },
            ["title", "markdown"],
        ),
        write_report,
        group="reporting",
        read_only=False,
    ),
]

TOOL_MAP: dict[str, Tool] = {tool.name: tool for tool in TOOLS}


def tool_names() -> list[str]:
    return list(TOOL_MAP)


def tool_schemas() -> list[dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in TOOLS
    ]


def call_tool(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Invoke a tool by name, converting failures into structured results."""
    tool = TOOL_MAP.get(name)
    if tool is None:
        return {"error": f"unknown tool {name!r}", "available": tool_names()}
    arguments = {k: v for k, v in (arguments or {}).items() if v is not None}
    try:
        return tool.handler(**arguments)
    except TypeError as exc:
        return {
            "error": f"bad arguments for {name}: {exc}",
            "expected_schema": tool.input_schema,
        }
    except Exception as exc:  # noqa: BLE001 — the boundary must not raise
        from ..net import SourceError

        payload: dict[str, Any] = {
            "error": str(exc),
            "error_type": type(exc).__name__,
            "tool": name,
        }
        if isinstance(exc, SourceError):
            payload["source"] = exc.source
            payload["advice"] = (
                "The data source failed or had no record. Try a different identifier or source "
                "rather than filling the gap from memory."
            )
        else:
            payload["traceback"] = traceback.format_exc(limit=3)
        return payload
