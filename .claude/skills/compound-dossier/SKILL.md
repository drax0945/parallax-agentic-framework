---
name: compound-dossier
description: Build a complete, cited profile of a single compound — identity, measured and computed properties, bioactivity per target, mechanism, patents and liabilities. Use when asked "tell me about X", "what do we know about X", "profile this compound", "is this a known compound", or when a compound enters a project for the first time.
---

# Compound dossier

A dossier answers four questions in order: **what is it**, **what has been measured**,
**what can be calculated**, and **what is unknown**. The last one is the part people
skip, and it is usually the most useful.

## Procedure

### 1. Recall before researching

```
memory_recall(query="<compound name> <target or project>", k=6)
```

If a previous session already profiled this compound, build on it rather than starting
over. Note what has changed since.

### 2. Establish identity — never from memory

```
resolve_compound(query="<whatever the user gave you>")
```

A name is not a structure. Report the canonical SMILES, InChIKey, formula and the
database record the structure came from. If the user supplied a SMILES, say so — that is
a different provenance from a database lookup.

If resolution fails, try the alternative identifier (CAS, InChIKey, ChEMBL ID) and say
plainly that the compound could not be resolved. Do not substitute a similar compound.

### 3. Pull everything at once

```
compound_profile(query="<resolved name or SMILES>")
```

This returns identity, computed properties, PubChem experimental annotations, the ChEMBL
record, mechanism of action, and measured bioactivity grouped by target with a citation
on every value. Read it before deciding what else you need.

### 4. Fill the gaps that matter

Depending on the question:

- selectivity or off-target context → `pubchem_assays`
- the originator filing and composition-of-matter claims → `patents_for_compound`
- what the literature says about a specific property → `literature_search`
- authoritative CAS number → `cas_lookup` (or the PubChem synonyms already returned)
- close analogues and whether a series exists → `similarity_search`

### 5. Read the liabilities honestly

`compound_profile` returns structural alerts and liability flags. These are heuristics
drawn from published practice, not predictions. Approved drugs trip them. Report them as
"what to measure next", which is what the tool's `next_experiment` field gives you.

## Reporting format

Lead with a two-line identity statement, then:

| | |
|---|---|
| **Measured** | the strongest experimental facts, each with a DOI |
| **Computed** | MW, cLogP, TPSA, HBD/HBA, rotatable bonds, Fsp3, QED, SA score; note macrocycle/bRo5 status |
| **Bioactivity** | best measured potency per target, assay format, citation |
| **Mechanism** | from ChEMBL mechanism records with references |
| **Liabilities** | flagged substructures and the experiment that would settle each |
| **Gaps** | what is not measured anywhere you looked |

Rules that matter here:

- If the compound is beyond rule-of-5 or macrocyclic, say that Ro5/Veber cut-offs are weak
  predictors for it and judge permeability on measured data instead. The tool says this
  too — do not contradict it with a confident 2D verdict.
- Distinguish biochemical (IC50/Ki, `assay_type` B) from cellular (EC50, `assay_type` F)
  potency explicitly. They are different quantities.
- When replicate measurements differ by more than a log unit, report the range, not a
  single number.

### 6. Close the loop

```
verify_all_citations()
citation_bibliography(format="markdown")
```

Then write what is durable:

```
memory_remember(type="finding", title="<specific cited claim>", body="<detail + caveat>",
                tags=[...], citations=[...], confidence=0.7)
```

Record an `entity` memory for the target or chemotype if you learned something durable
about it, and a `failure` memory if a source you expected to have data did not.

For a substantial dossier, offer `write_report` — the chemist gets an HTML file with
depicted structures and the full bibliography.
