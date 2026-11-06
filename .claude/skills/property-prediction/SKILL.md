---
name: property-prediction
description: Predict or retrieve physicochemical, developability and ADMET-relevant properties, keeping measured, computed and modelled values strictly apart. Use when asked for properties of a compound, logP/solubility/permeability/stability questions, drug-likeness or developability assessment, or whether a compound will be orally or CNS available.
---

# Property prediction

Most property mistakes come from one habit: presenting a calculation as if it were a
measurement. The whole discipline here is tier separation.

| Tier | Source | What it is | Example |
|---|---|---|---|
| **measured** | `measured_properties`, literature | an experiment, with a reference | measured logP 2.1 (handbook, cited) |
| **computed** | `computed_properties` | a deterministic calculation from structure | cLogP 2.80 (Wildman-Crippen) |
| **modelled** | `predict_potency` and trained models | a statistical prediction with a domain | predicted pIC50 7.9 ± 0.7 |

Always name the tier. Never let a computed value stand in for a measurement without
saying so.

## Procedure

### 1. Look for measurements first

```
measured_properties(query="<compound>")
```

PubChem's annotated experimental data carries the handbook or paper for each value —
melting point, solubility, logP, stability. If a measured value exists, it supersedes any
calculation, and you report the calculation only as a cross-check.

If nothing is annotated, search:

```
literature_search(query="<compound> solubility OR permeability OR logD measured")
```

### 2. Compute

```
computed_properties(smiles="<smiles>")
```

Returns MW, cLogP, TPSA, HBD/HBA, rotatable bonds, Fsp3, stereocentres, QED, SA score,
NP-likeness, macrocycle detection, plus rule sets and alerts, each citing its method paper.

What these do and do not mean:

- **cLogP** is a calculated octanol/water partition coefficient for the neutral species.
  It is **not** logD at pH 7.4, which is what matters for a basic or acidic compound.
  Say which one the question needs.
- **TPSA** is a 2D sum of fragment contributions. It correlates with permeability across
  classical small molecules and correlates poorly for macrocycles, which can shield polar
  surface conformationally.
- **SA score** is complexity-based synthetic accessibility, 1 (easy) to 10 (hard). It is
  not a route.
- **QED** is a weighted desirability score, not a probability of success.

### 3. Apply the rule sets with judgement

The tool evaluates Lipinski, Veber, Ghose, Egan, Pfizer 3/75, GSK 4/400 and lead-likeness,
and returns an `interpretation` field. Respect it:

- For a **classical small molecule**, violations are meaningful signals worth reporting.
- For a **macrocycle or beyond-rule-of-5 compound**, these cut-offs are weak predictors.
  Do not declare a macrocyclic drug "non-drug-like" on a Ro5 violation; lorlatinib and
  most macrocyclic kinase inhibitors violate these rules and are oral drugs.

For CNS questions use the CNS MPO score the tool returns, and note that the pKa term is
omitted (pKa is not computed) so the score is rescaled over five of six terms.

### 4. Liabilities

The profile returns PAINS/Brenk/NIH/ZINC alerts and liability flags with a
`next_experiment` field for each. Report them as "what to measure", not as predictions.
The common ones and what they mean:

- basic amine + cLogP > 3 → hERG counter-screen before further potency work
- aniline, nitroaromatic → AMES and reactive-metabolite screens
- thiophene, furan → CYP bioactivation screen
- alkyl ester → plasma stability
- cLogP > 5 → expect solubility and promiscuity problems

### 5. Where a model is needed

Parallax trains models from cited data on demand rather than shipping pre-trained ADMET
black boxes. If the user needs a modelled property that has a ChEMBL-measurable endpoint,
build it (see `potency-modelling`) and report its honest performance. If the endpoint
cannot be assembled from available data, **say so** and name the assay that would answer
the question. That is a better answer than an unvalidated number.

## Reporting format

A single table with a `tier` column, so nothing is ambiguous:

| Property | Value | Tier | Source |
|---|---|---|---|
| logP | 2.1 | measured | doi:... |
| cLogP | 2.80 | computed | Wildman-Crippen |
| TPSA | 110.1 Å² | computed | Ertl |
| pIC50 (ALK) | 7.9 ± 0.7 | modelled | model card, in-domain |

Then: developability read-out (which rules pass, with the macrocycle caveat if it
applies), liabilities with the next experiment for each, and explicitly what is **not**
measured anywhere you looked.

### 6. Remember

Record measured values you had to dig for as a `finding` with their citation — that search
should not need repeating.
