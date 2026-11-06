---
name: analog-design
description: Design new analogs of a lead — bioisosteric replacement, metabolic blocking, property tuning, rigidification and macrocyclisation — then rank them on predicted potency, properties and makeability. Use when asked to improve a compound, propose analogs or next compounds to make, fix a liability, or design a macrocyclic version of a lead.
---

# Analog design

Design is a loop: read the SAR, propose a change with a reason, predict what it does,
check it can be made, and say what would falsify the idea. Proposals without the last two
steps waste a chemist's time.

## Procedure

### 1. Read what the SAR already says

```
memory_recall(query="<target> <chemotype> SAR")
matched_molecular_pairs(target="<target>", endpoint="<endpoint>")
```

MMPA is **measured evidence**: which substituent swaps moved potency, by how much, how
consistently, with citations. Check `n_pairs` and `consistent` before treating any
transformation as a rule. A transformation seen twice with a 2-log spread is not a rule.

Also look at the existing series:

```
similarity_search(smiles="<lead>", source="chembl", threshold=70)
chembl_activities(target="<target>", standard_type="IC50", assay_type="B")
```

Knowing what has already been made stops you proposing a compound from a 2014 paper.

### 2. Define what you are optimising, and what must not regress

State it explicitly before designing: "raise ALK potency without adding lipophilicity and
keeping TPSA under 110". A design that improves one number and wrecks another is not an
improvement, and without a stated constraint you will not notice.

### 3. Generate proposals

```
design_analogs(smiles="<lead>", categories=["metabolic blocking","bioisostere"])
```

Available categories and what they are for:

| Category | Use when |
|---|---|
| `metabolic blocking` | a soft spot (aromatic CH, O-methyl, benzylic position) is driving clearance |
| `bioisostere` | a group is causing a problem but is needed for binding (acid, amide) |
| `potency probe` | exploring a pocket — methyl scan |
| `property tuning` | solubility or logP needs fixing (aza-scan, solubilising group) |
| `liability mitigation` | removing a basic centre for hERG, for example |
| `permeability` | removing HBDs, N-methylation |
| `scaffold hop` | the current core is patented or metabolically poor |

Each analog comes back with the rationale, the expected property effect, the literature
behind the strategy, and property deltas against the parent.

### 4. Rigidification and macrocyclisation

When the lead is flexible (many rotatable bonds) and potency or selectivity needs work:

```
design_macrocycles(smiles="<lead>", min_ring=11, max_ring=16)
```

This finds atom pairs whose linking closes an 11–16 membered ring, builds each design and
reports the rotatable-bond saving and the closure chemistry needed. The logic: a
macrocycle pays the conformational entropy cost of binding up front, which can buy
potency and selectivity, at the price of permeability risk and a harder synthesis.

Judge designs on: ring size (12–14 is the practical window), rotatable bonds removed, SA
score, and whether the closure chemistry has precedent. Say which bond closes the ring.

### 5. Predict and filter

```
score_designs(smiles_list=[...], model="<trained model>", max_sa_score=6.0)
```

If no model exists for the target, build one first (see the `potency-modelling` skill) and
quote its honest scaffold-split RMSE. Then:

- discard nothing silently — report what was filtered and why;
- treat predicted differences smaller than the model's RMSE as ties;
- flag every design whose applicability-domain verdict is "outside domain". Macrocyclic
  designs from an acyclic training set usually *are* outside the domain. Say so; that is
  the honest state of the art, not a failure of the tool.

### 6. Check they can be made

```
synthesis_feasibility(smiles="<each shortlisted design>")
retro_disconnect(smiles="<each shortlisted design>")
reaction_precedent_search(query="<closure or key step>")
```

A design that needs an unprecedented macrocyclisation on an 11-membered ring is a
different proposition from one closing a 13-membered macrolactam with ample precedent.

## Reporting format

A shortlist of 3–8 designs, each with:

- structure (SMILES) and what changed from the parent
- the reason for the change, with the literature behind the strategy
- predicted potency ± uncertainty, **with the domain verdict**
- key property deltas (cLogP, TPSA, HBD, rotatable bonds, MW)
- makeability: SA score, the closure/key step, whether it is precedented
- **what would falsify it** — the single assay to run first

Then one paragraph on the design hypothesis as a whole: what you are betting on, and what
result would tell you the bet is wrong.

### 7. Remember

```
memory_remember(type="heuristic", title="<transformation> in <series> does <effect>",
                body="<evidence, MMPA pair count, spread>", citations=[...], confidence=<honest>)
```

Record the designs proposed as a `finding` so a later session can check what was made and
what the measured result was — that is how this loop gets better over time.
