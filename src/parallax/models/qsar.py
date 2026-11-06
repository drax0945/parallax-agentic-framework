"""QSAR model building with honest, out-of-distribution evaluation.

What this module deliberately does
----------------------------------
* **Scaffold splits by default.** Random splits on congeneric medicinal-chemistry
  series flatter a model badly; a scaffold split estimates what happens on the
  next chemotype. A chronological split is available when publication years are
  present, which is the closest analogue of a prospective test.
* **Aggregates replicates** per compound (median pChEMBL) instead of letting a
  well-measured compound dominate.
* **Separates binding from functional assays**, because an IC50 against an
  enzyme and an EC50 in cells are not the same quantity.
* **Reports an applicability domain** with every prediction, based on the
  nearest-neighbour Tanimoto similarity to the training set.
* **Writes a model card** carrying the source DOIs behind the training data, so
  a prediction inherits its provenance.
"""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..config import settings, utcnow
from ..provenance import Citation, cite_software, registry
from ..chem.descriptors import physchem
from ..chem.structure import parse, scaffold

RDLogger.DisableLog("rdApp.*")

FP_RADIUS = 2
FP_BITS = 2048

ENDPOINT_PRESETS: dict[str, dict[str, Any]] = {
    "inhibition": {
        "standard_types": ["IC50"],
        "assay_type": "B",
        "label": "pIC50 (biochemical inhibition)",
        "note": "enzyme/target inhibition potency measured in a binding or biochemical format",
    },
    "binding": {
        "standard_types": ["Ki", "Kd"],
        "assay_type": "B",
        "label": "pKi/pKd (binding affinity)",
        "note": "equilibrium binding affinity",
    },
    "activation": {
        "standard_types": ["EC50"],
        "assay_type": "F",
        "label": "pEC50 (functional activation/response)",
        "note": "functional cellular response — agonist activation or cellular potency; "
        "mechanistically distinct from biochemical inhibition",
    },
    "cell_potency": {
        "standard_types": ["IC50", "GI50"],
        "assay_type": "F",
        "label": "pIC50 (cellular)",
        "note": "cell-based antiproliferative or functional potency",
    },
    "any_potency": {
        "standard_types": ["IC50", "Ki", "Kd", "EC50"],
        "assay_type": None,
        "label": "pActivity (mixed endpoints)",
        "note": "mixed endpoint model — use only for coarse ranking, since binding and "
        "functional readouts are pooled",
    },
}


@dataclass
class ModelCard:
    name: str
    target_chembl_id: str
    target_name: str | None
    endpoint: str
    endpoint_label: str
    standard_types: list[str]
    assay_type: str | None
    algorithm: str
    n_compounds: int
    n_raw_measurements: int
    n_train: int
    n_test: int
    split: str
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    candidates: dict[str, dict[str, float]]
    feature_spec: dict[str, Any]
    training_value_range: list[float]
    created: str
    citations: list[str] = field(default_factory=list)
    source_documents: int = 0
    warnings: list[str] = field(default_factory=list)
    data_notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        m = self.metrics
        return (
            f"{self.name}: {self.algorithm} on {self.n_compounds} compounds "
            f"({self.endpoint_label}); {self.split}-split test R²={m.get('r2', float('nan')):.2f}, "
            f"RMSE={m.get('rmse', float('nan')):.2f} log units, Spearman ρ={m.get('spearman', float('nan')):.2f}"
        )


# --- featurisation -------------------------------------------------------

_generator = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)

DESCRIPTOR_KEYS = [
    "mw",
    "clogp",
    "tpsa",
    "hbd",
    "hba",
    "rotatable_bonds",
    "aromatic_rings",
    "aliphatic_rings",
    "fraction_csp3",
    "heavy_atoms",
    "stereocentres",
    "num_amide_bonds",
    "largest_ring",
    "sa_score",
    "qed",
    "bertz_complexity",
]


def _fp_array(mol: Chem.Mol) -> np.ndarray:
    fingerprint = _generator.GetFingerprint(mol)
    array = np.zeros((FP_BITS,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fingerprint, array)
    return array


def featurize(smiles: Sequence[str], *, with_descriptors: bool = True) -> tuple[np.ndarray, list[int]]:
    """SMILES -> feature matrix. Returns (X, indices_kept)."""
    rows: list[np.ndarray] = []
    kept: list[int] = []
    for index, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            continue
        features = [_fp_array(mol)]
        if with_descriptors:
            values = physchem(mol)["values"]
            features.append(np.array([float(values[k]) for k in DESCRIPTOR_KEYS], dtype=np.float64))
        rows.append(np.concatenate([f.astype(np.float64) for f in features]))
        kept.append(index)
    if not rows:
        return np.zeros((0, FP_BITS)), []
    return np.vstack(rows), kept


def _fingerprints(smiles: Sequence[str]) -> list[Any]:
    out = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        out.append(_generator.GetFingerprint(mol) if mol is not None else None)
    return out


# --- splitting -----------------------------------------------------------


def scaffold_split(
    smiles: Sequence[str], *, test_fraction: float = 0.2, seed: int = 0
) -> tuple[list[int], list[int]]:
    """Bemis-Murcko scaffold split: whole scaffolds held out together."""
    groups: dict[str, list[int]] = {}
    for index, smi in enumerate(smiles):
        try:
            key = scaffold(smi, generic=False) or f"__none_{index}"
        except Exception:
            key = f"__bad_{index}"
        groups.setdefault(key, []).append(index)

    # Largest scaffold families into train; singletons dominate the test set,
    # which is what makes this a pessimistic (useful) estimate.
    ordered = sorted(groups.values(), key=len, reverse=True)
    n_test_target = int(len(smiles) * test_fraction)
    train: list[int] = []
    test: list[int] = []
    for family in ordered:
        if len(test) < n_test_target and len(family) <= max(2, n_test_target // 4):
            test.extend(family)
        else:
            train.extend(family)
    if not test:  # degenerate case: one scaffold family
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(smiles))
        cut = int(len(smiles) * (1 - test_fraction))
        train, test = list(indices[:cut]), list(indices[cut:])
    return train, test


def time_split(years: Sequence[int | None], *, test_fraction: float = 0.2) -> tuple[list[int], list[int]]:
    """Chronological split — train on the past, test on the most recent work."""
    known = [(index, year) for index, year in enumerate(years) if year]
    if len(known) < 20:
        raise ValueError("not enough year-stamped records for a chronological split")
    known.sort(key=lambda pair: pair[1])
    cut = int(len(known) * (1 - test_fraction))
    train = [index for index, _ in known[:cut]]
    test = [index for index, _ in known[cut:]]
    missing = [index for index, year in enumerate(years) if not year]
    train.extend(missing)
    return train, test


# --- training ------------------------------------------------------------


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    rank_a = np.argsort(np.argsort(a)).astype(float)
    rank_b = np.argsort(np.argsort(b)).astype(float)
    if rank_a.std() == 0 or rank_b.std() == 0:
        return float("nan")
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 3),
        "rmse": round(float(math.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "spearman": round(float(_spearman(y_true, y_pred)), 3),
        "n": int(len(y_true)),
    }


def aggregate_measurements(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """One record per canonical structure: median value, spread, provenance."""
    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        smi = row.get("smiles")
        value = row.get("pchembl_value")
        if not smi or value is None:
            continue
        try:
            canonical = Chem.MolToSmiles(parse(smi))
        except Exception:
            continue
        bucket = buckets.setdefault(
            canonical,
            {
                "smiles": canonical,
                "values": [],
                "citations": set(),
                "years": [],
                "chembl_ids": set(),
                "assay_descriptions": set(),
            },
        )
        bucket["values"].append(float(value))
        if row.get("citation"):
            bucket["citations"].add(row["citation"])
        if row.get("document_year"):
            bucket["years"].append(row["document_year"])
        if row.get("molecule_chembl_id"):
            bucket["chembl_ids"].add(row["molecule_chembl_id"])
        if row.get("assay_description"):
            bucket["assay_descriptions"].add(row["assay_description"][:160])

    out: dict[str, dict[str, Any]] = {}
    for canonical, bucket in buckets.items():
        values = np.array(bucket["values"], dtype=float)
        out[canonical] = {
            "smiles": canonical,
            "y": float(np.median(values)),
            "n_measurements": int(values.size),
            "spread": round(float(values.max() - values.min()), 2) if values.size > 1 else 0.0,
            "citations": sorted(bucket["citations"]),
            "year": min(bucket["years"]) if bucket["years"] else None,
            "chembl_ids": sorted(bucket["chembl_ids"]),
            "assays": sorted(bucket["assay_descriptions"])[:3],
        }
    return out


def train(
    target: str,
    *,
    endpoint: str = "inhibition",
    split: str = "scaffold",
    name: str | None = None,
    max_records: int = 3000,
    min_compounds: int = 40,
    test_fraction: float = 0.2,
    seed: int = 0,
    save: bool = True,
) -> dict[str, Any]:
    """Build a potency/activation model for one target from ChEMBL data."""
    from ..sources import chembl

    preset = ENDPOINT_PRESETS.get(endpoint)
    if preset is None:
        raise ValueError(f"unknown endpoint {endpoint!r}; options: {sorted(ENDPOINT_PRESETS)}")

    # Resolve the target.
    target_chembl_id = target
    target_name = None
    if not target.upper().startswith("CHEMBL"):
        found = chembl.search_target(target, limit=1)
        if not found["targets"]:
            raise ValueError(f"no ChEMBL target matched {target!r}")
        target_chembl_id = found["targets"][0]["target_chembl_id"]
        target_name = found["targets"][0]["pref_name"]

    raw_rows: list[dict[str, Any]] = []
    citation_keys: set[str] = set()
    per_type = max(max_records // len(preset["standard_types"]), 300)
    for standard_type in preset["standard_types"]:
        result = chembl.activities(
            target_chembl_id=target_chembl_id,
            standard_type=standard_type,
            assay_type=preset["assay_type"],
            limit=per_type,
        )
        raw_rows.extend(result["rows"])
        citation_keys.update(k for k in result["citations"] if k)
        if not target_name:
            target_name = next((r.get("target_name") for r in result["rows"] if r.get("target_name")), None)

    usable = [
        row
        for row in raw_rows
        if row.get("pchembl_value") is not None
        and row.get("smiles")
        and row.get("standard_relation") in (None, "=", "~")
        and not row.get("data_validity_comment")
    ]
    aggregated = aggregate_measurements(usable)
    if len(aggregated) < min_compounds:
        raise ValueError(
            f"only {len(aggregated)} usable compounds for {target_chembl_id}/{endpoint} "
            f"(need {min_compounds}); try endpoint='any_potency' or a different target"
        )

    records = list(aggregated.values())
    smiles = [r["smiles"] for r in records]
    y_all = np.array([r["y"] for r in records], dtype=float)
    X_all, kept = featurize(smiles)
    records = [records[i] for i in kept]
    smiles = [smiles[i] for i in kept]
    y_all = y_all[kept]

    warnings: list[str] = []
    if split == "time":
        try:
            train_idx, test_idx = time_split([r["year"] for r in records], test_fraction=test_fraction)
            split_used = "chronological"
        except ValueError as exc:
            warnings.append(f"chronological split unavailable ({exc}); fell back to scaffold split")
            train_idx, test_idx = scaffold_split(smiles, test_fraction=test_fraction, seed=seed)
            split_used = "scaffold"
    elif split == "random":
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(smiles))
        cut = int(len(order) * (1 - test_fraction))
        train_idx, test_idx = list(order[:cut]), list(order[cut:])
        split_used = "random"
        warnings.append(
            "random split: optimistic for congeneric series — read the scaffold-split number instead"
        )
    else:
        train_idx, test_idx = scaffold_split(smiles, test_fraction=test_fraction, seed=seed)
        split_used = "scaffold"

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]

    candidates: dict[str, Any] = {
        "random_forest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=1, n_jobs=-1, random_state=seed
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=seed),
        "ridge": RidgeCV(alphas=np.logspace(-2, 3, 12)),
    }
    scored: dict[str, dict[str, float]] = {}
    fitted: dict[str, Any] = {}
    for label, estimator in candidates.items():
        try:
            estimator.fit(X_train, y_train)
            scored[label] = _metrics(y_test, estimator.predict(X_test))
            fitted[label] = estimator
        except Exception as exc:  # a candidate failing must not sink the run
            warnings.append(f"{label} failed to fit: {exc}")

    if not fitted:
        raise RuntimeError("no model could be fitted")

    best_label = max(scored, key=lambda k: (scored[k]["spearman"] if not math.isnan(scored[k]["spearman"]) else -9, scored[k]["r2"]))
    best_model = fitted[best_label]

    baseline = _metrics(y_test, np.full_like(y_test, float(np.mean(y_train))))

    cite_software(
        "software:scikit_learn",
        "method:random_forest",
        "method:morgan_fp",
        "method:scaffold_split",
        "method:applicability_domain",
        "software:rdkit",
    )

    model_name = name or f"{target_chembl_id}_{endpoint}".lower()
    card = ModelCard(
        name=model_name,
        target_chembl_id=target_chembl_id,
        target_name=target_name,
        endpoint=endpoint,
        endpoint_label=preset["label"],
        standard_types=list(preset["standard_types"]),
        assay_type=preset["assay_type"],
        algorithm=best_label,
        n_compounds=len(records),
        n_raw_measurements=len(usable),
        n_train=len(train_idx),
        n_test=len(test_idx),
        split=split_used,
        metrics=scored[best_label],
        baseline_metrics=baseline,
        candidates=scored,
        feature_spec={
            "morgan_radius": FP_RADIUS,
            "morgan_bits": FP_BITS,
            "descriptors": DESCRIPTOR_KEYS,
        },
        training_value_range=[round(float(y_all.min()), 2), round(float(y_all.max()), 2)],
        created=utcnow(),
        citations=sorted(citation_keys),
        source_documents=len({c for r in records for c in r["citations"]}),
        warnings=warnings,
        data_notes=[
            preset["note"],
            "replicate measurements aggregated as the median pChEMBL value per structure",
            f"{sum(1 for r in records if r['spread'] > 1.0)} compounds have >1 log unit spread "
            "between replicate measurements (assay noise floor)",
        ],
    )

    # Register the model itself as a citable artefact pointing at its data.
    model_citation = registry().add(
        Citation(
            key=f"model:{model_name}",
            kind="model",
            title=f"Parallax QSAR model '{model_name}' — {card.endpoint_label} for {target_name or target_chembl_id}",
            source_db="parallax",
            year=int(utcnow()[:4]),
            note=(
                f"{best_label} on {card.n_compounds} compounds from {card.source_documents} ChEMBL "
                f"source documents; {split_used}-split test R²={card.metrics['r2']}, "
                f"RMSE={card.metrics['rmse']} log units"
            ),
            identifiers={
                "target_chembl_id": target_chembl_id,
                "endpoint": endpoint,
                "training_citations": card.citations[:200],
            },
        )
    )
    card.citations = sorted(set(card.citations + [model_citation]))

    payload = {
        "card": card,
        "model": best_model,
        "train_fingerprints": _fingerprints([smiles[i] for i in train_idx]),
        "train_smiles": [smiles[i] for i in train_idx],
        "train_y": y_train,
        "records": records,
    }

    if save:
        _save(model_name, payload)

    return {
        "model": model_name,
        "card": card.as_dict(),
        "summary": card.summary(),
        "citations": card.citations,
        "held_out_examples": [
            {
                "smiles": smiles[i],
                "measured": round(float(y_all[i]), 2),
                "predicted": round(float(best_model.predict(X_all[i : i + 1])[0]), 2),
            }
            for i in test_idx[:8]
        ],
    }


# --- persistence ---------------------------------------------------------


def _model_path(name: str) -> Path:
    return settings().models_dir / f"{name}.pkl"


def _card_path(name: str) -> Path:
    return settings().models_dir / f"{name}.card.json"


def _save(name: str, payload: dict[str, Any]) -> None:
    settings().models_dir.mkdir(parents=True, exist_ok=True)
    with _model_path(name).open("wb") as handle:
        pickle.dump(payload, handle)
    _card_path(name).write_text(json.dumps(payload["card"].as_dict(), indent=2), encoding="utf-8")


def load_model(name: str) -> dict[str, Any]:
    path = _model_path(name)
    if not path.exists():
        available = [p.stem for p in settings().models_dir.glob("*.pkl")]
        raise FileNotFoundError(f"no model named {name!r}; available: {available}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def list_models() -> list[dict[str, Any]]:
    out = []
    for card_file in sorted(settings().models_dir.glob("*.card.json")):
        try:
            out.append(json.loads(card_file.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return out


# --- prediction ----------------------------------------------------------


def applicability(fingerprint: Any, train_fingerprints: Sequence[Any]) -> dict[str, Any]:
    """Nearest-neighbour similarity to the training set."""
    valid = [fp for fp in train_fingerprints if fp is not None]
    if not valid or fingerprint is None:
        return {"max_similarity": None, "in_domain": None, "verdict": "unknown"}
    sims = DataStructs.BulkTanimotoSimilarity(fingerprint, valid)
    sims_sorted = sorted(sims, reverse=True)
    top = sims_sorted[0]
    mean_top5 = float(np.mean(sims_sorted[:5]))
    if top >= 0.7:
        verdict = "inside domain — close analogues in training data"
    elif top >= 0.5:
        verdict = "edge of domain — related chemotype, widen the error bars"
    elif top >= 0.35:
        verdict = "outside domain — extrapolation, treat as a ranking hint only"
    else:
        verdict = "far outside domain — not a usable prediction"
    return {
        "max_similarity": round(float(top), 3),
        "mean_top5_similarity": round(mean_top5, 3),
        "in_domain": bool(top >= 0.5),
        "verdict": verdict,
    }


def predict(
    model_name: str,
    smiles: str | Iterable[str],
    *,
    include_neighbours: bool = True,
) -> dict[str, Any]:
    """Predict with an applicability-domain judgement and inherited citations."""
    payload = load_model(model_name)
    card: ModelCard = payload["card"]
    model = payload["model"]
    queries = [smiles] if isinstance(smiles, str) else list(smiles)

    X, kept = featurize(queries)
    results: list[dict[str, Any]] = []
    fingerprints = _fingerprints(queries)
    train_fps = payload["train_fingerprints"]
    train_smiles = payload["train_smiles"]
    train_y = payload["train_y"]

    for position, index in enumerate(kept):
        row = X[position : position + 1]
        value = float(model.predict(row)[0])
        spread = None
        if hasattr(model, "estimators_") and isinstance(model, RandomForestRegressor):
            per_tree = np.array([est.predict(row)[0] for est in model.estimators_])
            spread = float(per_tree.std())

        domain = applicability(fingerprints[index], train_fps)
        rmse = card.metrics.get("rmse") or 0.0
        uncertainty = max(rmse, (spread or 0.0))
        if domain["max_similarity"] is not None and domain["max_similarity"] < 0.5:
            uncertainty *= 1.5

        neighbours = []
        if include_neighbours and fingerprints[index] is not None:
            sims = DataStructs.BulkTanimotoSimilarity(
                fingerprints[index], [fp for fp in train_fps if fp is not None]
            )
            order = np.argsort(sims)[::-1][:3]
            for position_n in order:
                neighbours.append(
                    {
                        "smiles": train_smiles[int(position_n)],
                        "measured": round(float(train_y[int(position_n)]), 2),
                        "similarity": round(float(sims[int(position_n)]), 3),
                    }
                )

        in_range = card.training_value_range[0] <= value <= card.training_value_range[1]
        try:
            canonical_query = Chem.MolToSmiles(parse(queries[index]))
        except Exception:
            canonical_query = queries[index]
        in_training = canonical_query in set(train_smiles)
        results.append(
            {
                "query": queries[index],
                "in_training_set": in_training,
                "caveat": (
                    "This exact structure is in the model's training set, so the prediction "
                    "is a fit, not a test. Use the measured value instead."
                    if in_training
                    else None
                ),
                "endpoint": card.endpoint_label,
                "predicted_value": round(value, 2),
                "predicted_nM": round(10 ** (9 - value), 1) if 0 < value < 15 else None,
                "uncertainty_log_units": round(uncertainty, 2),
                "interval_68": [round(value - uncertainty, 2), round(value + uncertainty, 2)],
                "applicability_domain": domain,
                "within_training_range": in_range,
                "nearest_training_compounds": neighbours,
                "model": card.name,
            }
        )

    skipped = [queries[i] for i in range(len(queries)) if i not in kept]
    return {
        "model": card.name,
        "endpoint": card.endpoint_label,
        "target": card.target_name or card.target_chembl_id,
        "model_performance": {
            "split": card.split,
            **card.metrics,
            "baseline_rmse_predicting_the_mean": card.baseline_metrics.get("rmse"),
        },
        "predictions": results,
        "unparsable_inputs": skipped,
        "how_to_read": (
            f"Values are {card.endpoint_label}; higher is more potent. The model's own "
            f"{card.split}-split RMSE is {card.metrics.get('rmse')} log units, so differences "
            "smaller than that are not meaningful. Predictions outside the applicability "
            "domain are ranking hints, not numbers to design against."
        ),
        "citations": card.citations,
    }
