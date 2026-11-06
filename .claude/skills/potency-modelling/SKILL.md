---
name: potency-modelling
description: Build and use QSAR models for potency, activation or cellular response with honest out-of-distribution evaluation. Use when asked to predict activity/potency/activation/IC50/EC50/Ki for a compound or a set of designs, to build a model for a target, to rank candidates by predicted activity, or to judge whether a prediction can be trusted.
---

# Potency and activation modelling

A prediction is only worth as much as the evaluation behind it. The job here is to build
a model whose reported performance is what a chemist would actually get on the next
chemotype, and then to be disciplined about when its numbers mean anything.

## Choosing the endpoint — this decides what you are predicting

| Endpoint | Data | Means |
|---|---|---|
| `inhibition` | IC50, biochemical/binding format | target inhibition potency |
| `binding` | Ki, Kd | equilibrium affinity |
| `activation` | EC50, functional format | a **functional response** — agonist activation or cellular potency |
| `cell_potency` | cellular IC50, GI50 | antiproliferative/cellular potency |
| `any_potency` | all of the above pooled | coarse ranking only |

"Activation" is not inhibition measured differently; it is a different experiment. If the
user asks about activation, use the `activation` endpoint. If they ask vaguely about
"activity", ask which readout — or build both and say what each covers.

## Procedure

### 1. Recall and check for an existing model

```
memory_recall(query="<target> model potency")
list_potency_models()
```

Reuse an existing model where the endpoint matches. Retrain when the endpoint differs or
the data is materially older.

### 2. Resolve the target

```
chembl_find_target(query="ALK")
```

Confirm organism and that you have the right record — a kinase name often matches several
targets and the mouse ortholog.

### 3. Check the data before trusting a model of it

```
chembl_activities(target="CHEMBL4247", standard_type="IC50", assay_type="B", limit=200)
```

Look for: how many compounds, how many source documents, whether the series is congeneric
(all one scaffold) or diverse, and how large the replicate spread is. A model over 60
compounds from one paper is a model of that paper.

### 4. Train

```
train_potency_model(target="CHEMBL4247", endpoint="inhibition", split="scaffold")
```

- **`scaffold`** (default) holds out whole Bemis-Murcko scaffold families. Pessimistic and
  the number you should quote.
- **`time`** trains on earlier publications and tests on the most recent — the closest
  thing to a prospective test. Use it when the data is well year-stamped and say so.
- **`random`** is optimistic on congeneric series. Only use it to show the gap.

Read three numbers from the model card: test **RMSE**, the **baseline RMSE** (predicting
the training mean), and **Spearman ρ**. A model whose RMSE is not clearly below the
baseline has learned nothing useful, and you should say that instead of reporting its
predictions.

### 5. Predict, and respect the domain

```
predict_potency(model="chembl4247_inhibition", smiles=["<design 1>", "<design 2>"])
```

Every prediction comes back with:

- `uncertainty_log_units` and a 68% interval,
- an `applicability_domain` verdict from nearest-neighbour similarity,
- the nearest measured training compounds,
- `in_training_set` — if true, quote the **measured** value, not the prediction.

Hard rules:

- Never report a prediction without its domain verdict.
- Treat differences smaller than the model's RMSE as noise. If RMSE is 0.7 log units, 7.9
  and 8.3 are the same prediction.
- "Outside domain" predictions order candidates at best. Say that rather than giving a
  number with a decimal point.

### 6. Use SAR evidence alongside the model

```
matched_molecular_pairs(target="CHEMBL4247", endpoint="inhibition")
```

MMPA is measured evidence about which substituent changes actually move potency, with
citations. Where MMPA and the model disagree, the measured pairs win — and that
disagreement is worth reporting.

## Reporting format

State for each compound: predicted value (with units — pIC50 or pEC50), the interval, the
domain verdict, and the nearest measured analogue with its real value. Then one line on
what the model is: algorithm, compound count, source-document count, split, test RMSE vs
baseline.

Close with the limits, specifically: assay noise in the training data, how congeneric the
training set was, and that any computational design needs experimental validation.

### 7. Remember

```
memory_remember(type="model", title="<model> for <target> (<endpoint>)",
                body="<summary + honest performance + what it is good for>",
                tags=["model","qsar","<target>"], citations=[...], confidence=0.8)
```

Also record a `heuristic` if the modelling exposed something general about the series
(e.g. "the model systematically under-predicts macrocycles in this series"), and a
`failure` memory if the data was too sparse to model — that saves the next session the
same dead end.
