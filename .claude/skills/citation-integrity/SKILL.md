---
name: citation-integrity
description: Verify every reference before delivery and produce the bibliography. Use before delivering any report or analysis, when asked for references or a bibliography, when checking whether a citation is real, or when a DOI needs validating.
---

# Citation integrity

A fabricated reference destroys the credibility of everything around it. This skill is
the check that runs before anything leaves.

## The rules

1. **Citations come from tool results.** Every tool that returns data registers a citation
   and hands back its key. Those keys are what you cite.
2. **Never write a DOI from memory.** If you believe a paper exists and want to cite it,
   verify it first:
   ```
   check_doi(doi="10.1021/jm1012374")
   ```
   The result gives the real title, venue, year and authors. If it does not resolve, do
   not cite it — say the claim is from background knowledge and unverified.
3. **Verify the whole registry before delivery.**
   ```
   verify_all_citations()
   ```
   Every DOI is checked against Crossref. Any that fail are named; remove those claims or
   mark them clearly as unverified.
4. **Audit the prose, not just the registry.**
   ```
   audit_draft(text="<your finished report>")
   ```
   `verify_all_citations` can only check what reached the registry. A DOI typed straight
   into the text bypasses it completely — and that is the easiest mistake to make, because
   a half-remembered reference looks exactly like a retrieved one. `audit_draft` scans the
   finished text and separates two failures:

   - **fabricated** — the DOI does not resolve. Remove it.
   - **unsourced** — the DOI is real, but no tool returned it. The reference exists; the
     *claim* is still ungrounded. Either retrieve it properly with `literature_search` or
     `check_doi`, or drop the claim.

5. **The bibliography ships with the analysis.**
   ```
   citation_bibliography(format="markdown")   # or "bibtex" for a manuscript
   ```

## Matching a claim to the right citation

Cite the thing that actually supports the claim:

| Claim | Correct citation |
|---|---|
| "IC50 = 4 nM against ALK" | the source document of that measurement (`doi:...` from `chembl_activities`) |
| "cLogP is 2.8" | the method paper (Wildman-Crippen) — it is a calculation, and `computed_properties` returns the key |
| "predicted pIC50 7.9" | the model citation (`model:...`), which itself lists the training-set documents |
| "amide couplings tolerate heterocycles" | the methodology reference returned with the template |
| "this compound is a known compound" | the database record (`pubchem:CID...`, `chembl:CHEMBL...`) |
| "no precedent found for this step" | nothing — state which sources you searched instead |

A database record and a journal article are different kinds of evidence. The tools mark
the `kind` field; keep that distinction in your prose.

## Honest absence

When a search finds nothing, that is a result and it is reported as one:

> No precedent was found for this closure in Europe PMC (journals + EPO patent abstracts)
> or ChEMBL source documents. Licensed reaction databases were not available in this
> workspace, and their coverage of preparative conditions is deeper, so this is
> "not found", not "not known".

Never let absence of evidence become a confident claim in either direction.

## Before delivering, check

- [ ] every number has a tier (measured / computed / modelled) and a citation
- [ ] `verify_all_citations` run, and no unresolved DOI remains in the text
- [ ] `audit_draft` run on the final text, and it reports `clean: true`
- [ ] no DOI appears that did not come from a tool result or pass `check_doi`
- [ ] model predictions carry their applicability-domain verdict
- [ ] bibliography appended
- [ ] gaps and unchecked assumptions stated explicitly
