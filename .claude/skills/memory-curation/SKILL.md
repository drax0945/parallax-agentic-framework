---
name: memory-curation
description: Decide what to write to long-term memory, how to phrase it, and when to correct or retire an entry. Use at the end of substantial work, when the user corrects you or states a preference, when a recalled memory conflicts with new evidence, or when asked what the workspace has learned.
---

# Memory curation

Memory is what makes this workspace improve rather than restart. It lives in
`memory/entries/*.md` with a human-readable index at `memory/MEMORY.md`, so a chemist can
read it, disagree and edit it.

## What to write

| Type | Write it when | Example title |
|---|---|---|
| `finding` | a specific, cited result worth not re-deriving | "Lorlatinib ALK IC50 is 1.3 nM (biochemical), doi:..." |
| `heuristic` | a generalisation across cases, with the evidence behind it | "In this ALK series, 12-membered macrolactam closure held potency and removed 5 rotatable bonds" |
| `failure` | a dead end — the most valuable type | "SNAr closure on the 11-membered analogue gave only oligomer in the reported attempt" |
| `sop` | a procedure that worked and would be repeated | "Dataset curation for ALK: IC50 + assay_type B, median per structure, drop validity-flagged rows" |
| `model` | a model was trained, with its honest performance | "QSAR for ALK pIC50: RF, 1540 compounds, scaffold R² 0.62, RMSE 0.69" |
| `preference` | the user states how they want work done | "Reports should quote pIC50 not nM, and always name the weakest synthetic step" |
| `entity` | a durable fact about a target, assay or chemotype | "ALK = CHEMBL4247 (human); mouse ortholog CHEMBL5771 is a common mis-selection" |

## What not to write

- Anything the tools recompute cheaply and deterministically (a cLogP value, a canonical
  SMILES). Memory is for what was *learned*, not what is *derivable*.
- Restatements of a database record — cite the record instead.
- Vague impressions. "Macrocycles are promising" helps nobody.
- Session mechanics ("the user asked about ALK today").

## How to phrase an entry

A good title is a **self-contained claim** that still makes sense in six months with no
context. Compare:

- bad: "Macrocycle results"
- good: "Closing the 12-membered macrolactam in the crizotinib series removed all 5 rotatable bonds while retaining ALK potency"

The body carries the detail, the evidence, and — importantly — the caveat: sample size,
assay format, whether it is one observation or many, and what would change your mind.

```
memory_remember(
  type="heuristic",
  title="<the claim>",
  body="<evidence, numbers, assay format, caveat, what would falsify it>",
  tags=["alk","macrocycle","potency"],
  citations=["doi:...", "chembl:CHEMBL..."],
  confidence=0.65,
  entities={"targets": ["CHEMBL4247"]})
```

## Confidence, honestly

| Confidence | Means |
|---|---|
| 0.9+ | multiple independent sources, or a direct user instruction |
| 0.7–0.85 | solid evidence, one good source or a consistent MMPA set |
| 0.5–0.65 | a single observation, or a plausible inference |
| < 0.5 | a hunch worth recording but not acting on |

Near-duplicates are merged automatically and corroborate the existing entry, so writing
the same thing twice raises confidence rather than cluttering the store. That is intended
— corroboration is signal.

## Correcting memory

This matters more than adding to it. When a recalled memory conflicts with new evidence:

```
memory_update(entry_id="<id>", action="contradict",
              note="<what conflicts, with the citation>")
```

Confidence is halved and the conflict is recorded in the entry. If the old entry is simply
wrong, write the corrected entry and then retire the old one. If it was never more than
noise, `forget` it.

When a memory is confirmed by new work:

```
memory_update(entry_id="<id>", action="reinforce", note="<how>", citations=["doi:..."])
```

## Housekeeping

At the end of a long session:

```
memory_consolidate()
```

This merges near-duplicates, decays entries nothing has used or confirmed, and rewrites
`memory/MEMORY.md`. Then tell the user what was learned this session in one or two lines —
they should know what their workspace now believes.

```
memory_stats()
```

gives counts by type, mean confidence and contradicted entries, which is the health check
worth reporting if the store is getting large.
