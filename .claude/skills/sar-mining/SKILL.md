---
name: sar-mining
description: Mine literature and patents for a compound series, assemble a curated SAR dataset with per-measurement citations, and analyse what drives activity. Use when asked to find analogues of a compound, build a dataset for a target or series, review the SAR around a chemotype, or survey what has been published or patented in an area.
---

# SAR mining

The output of this skill is a dataset a chemist can trust: every row has a structure, a
measured value, an assay format and the document that reported it. Curation decisions are
stated, not silent.

## Procedure

### 1. Frame the series

```
memory_recall(query="<target> <chemotype> series dataset")
chembl_find_target(query="<target>")
resolve_compound(query="<reference compound>")
```

Decide and write down: which target record, which endpoint, which assay format, and what
defines membership of the series (a scaffold, a similarity cut-off, or a named programme).

### 2. Structured data first

```
chembl_activities(target="CHEMBL4247", standard_type="IC50", assay_type="B", limit=1000)
```

This is the backbone: structures with measured values, each carrying its source DOI. Pull
each endpoint separately rather than pooling — binding and functional data answer
different questions.

### 3. Widen by structure

```
similarity_search(smiles="<reference>", source="chembl", threshold=70)
substructure_search(query="<scaffold SMARTS>")
```

Use these to find series members that the target-based query missed, for example
compounds tested only in a cell assay.

### 4. Literature and patents

```
literature_search(query="<chemotype> <target> inhibitor SAR", limit=20, year_from=2015)
patent_search(query="<chemotype> <target> inhibitor")
patents_for_compound(query="<reference compound>")
```

Patents matter: most medicinal-chemistry SAR appears there first, and the originator
filing usually contains the broadest table of analogues. For open-access papers worth
reading closely:

```
read_full_text(pmcid="PMC...")
```

### 5. Curate, and say what you removed

The tools already drop rows with a non-equal relation (`>`, `<`), a missing pChEMBL value
or a data-validity comment. Beyond that, state your own filters explicitly:

- buffers, solvents and reagents that appear as "compounds" in extracted data
- duplicate structures across sources — aggregate to a median per structure and report
  the replicate spread
- mixed assay formats — keep them separate or say you pooled them and why
- values outside a plausible range for the assay

Report the funnel: records retrieved → after relation/validity filters → unique
structures → final dataset. A dataset whose size is stated without a funnel is not
curated.

### 6. Analyse

```
matched_molecular_pairs(target="<target>", endpoint="<endpoint>")
```

Then look at the distribution of the dataset itself: how many distinct scaffolds, whether
one paper dominates, how the values are spread, and what the replicate noise looks like.
This determines what any conclusion — or any model — can support.

Useful questions to answer concretely:

- which substituent changes move potency, by how much, and how consistently (MMPA)
- whether potency tracks lipophilicity in this series (compute cLogP across the set and
  look; if it does, raw potency gains may be lipophilicity artefacts)
- where the series is unexplored — which positions nobody has varied

### 7. Report

| | |
|---|---|
| **Dataset** | n structures, n measurements, n source documents, year range |
| **Funnel** | what was retrieved and what was removed, with the rule for each |
| **Assay mix** | formats present, and whether they were pooled |
| **Noise** | how many compounds have >1 log unit replicate spread |
| **SAR drivers** | MMPA transformations with pair counts and consistency |
| **Gaps** | positions and property space nobody has explored |

Offer `write_report` for the full version with depicted structures, and
`citation_bibliography` for the source list — a mined dataset is only as useful as its
provenance.

### 8. Remember

```
memory_remember(type="finding", title="<target> <series> dataset: n=<N> from <M> documents",
                body="<how it was assembled, filters, noise level, where it lives>",
                tags=["dataset","<target>"], citations=[...], confidence=0.8)
memory_remember(type="heuristic", title="<what the SAR says>", ...)
```

Record the curation rules as an `sop` memory so the next pass is reproducible.
