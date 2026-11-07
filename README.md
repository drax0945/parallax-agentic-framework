# Parallax

An agentic cheminformatics framework. An LLM agent orchestrates; deterministic chemistry
tools do the calculations; every number carries a source; and what the workspace learns
persists in plain markdown you can read and correct.

Built for medicinal and synthetic chemistry work: profile a compound, mine SAR out of the
literature and patents, train potency models on cited data, design analogs and
macrocycles, plan syntheses, and judge whether any of it can actually be made.

```bash
parallax ask "Profile lorlatinib, build an ALK potency model, and propose three
              macrocyclic analogs with routes and citations"
```

---

## The one idea this is built around

Cheminformatics reports go wrong in a predictable way: a calculated number gets reported
as if it were measured, a model's guess as if it were a fact, and a plausible-looking
citation turns out not to exist. Parallax makes those three mistakes hard.

**Every quantitative claim belongs to exactly one tier, and the tier is always stated:**

| Tier | Where it comes from | What it cites |
|---|---|---|
| **measured** | ChEMBL, PubChem annotations, a paper | the DOI of the report that measured it |
| **computed** | RDKit, deterministic from structure | the method paper (Crippen, Ertl, Bickerton…) |
| **modelled** | a model trained here, on cited data | the model card, which lists its training-set DOIs |

A model prediction inherits the provenance of its training data. Ask where a predicted
pIC50 came from and the chain is: value → model → training set → the papers that reported
every point in it.

And citations are verified, not trusted, in two places: `verify_all_citations` checks
every DOI in the session registry against Crossref, and `audit_draft` scans the finished
prose for references no tool produced — because a DOI recalled from background knowledge
never enters the registry and would otherwise slip through. It separates *fabricated*
(does not resolve) from *unsourced* (real paper, but you did not retrieve it, so the claim
is still ungrounded). This caught a real lapse during development: a live agent run cited
a genuine J. Med. Chem. DOI it had never looked up.

---

## Install

Needs Python 3.11+ and either the [`claude` CLI](https://claude.com/claude-code) logged in
(no API key needed — it reuses your session) or `ANTHROPIC_API_KEY`.

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install -e .
.venv/bin/parallax init
.venv/bin/parallax doctor
```

`doctor` checks dependencies, the agent runtime, every data source and the memory store.
No credentials are required: PubChem, ChEMBL, Europe PMC, Crossref and the NCI resolver
are all free and keyless.

---

## Two ways to use it

### The agent

```bash
parallax ask "What is known about crizotinib's ALK potency, and how does lorlatinib's
              macrocyclisation change the property profile?"
```

The agent loads the skills in `.claude/skills/`, recalls what previous sessions learned,
calls the tools, and reports with a bibliography. Runs are saved to `.parallax/runs/`.

### Inside Claude Code or the desktop app

`parallax init` writes `.mcp.json`, so the 41 tools appear in any MCP client:

```json
{ "mcpServers": { "parallax": { "command": "parallax", "args": ["mcp"] } } }
```

The skills in `.claude/skills/` load automatically in the same workspace, so you get the
same behaviour with your normal agent.

### Directly, when you know what you want

```bash
parallax compound lorlatinib              # cited dossier
parallax properties '<SMILES>'            # computed profile + rule sets + alerts
parallax train ALK --endpoint activation  # model, honestly evaluated
parallax predict chembl4247_inhibition '<SMILES>'
parallax mmpa ALK                         # what the measured SAR actually says
parallax retro '<SMILES>' --plan          # routes
parallax design '<SMILES>' --macrocycle --model chembl4247_inhibition
parallax precedent "macrolactamization 12-membered ring high dilution"
parallax feasibility '<SMILES>'
parallax lit "lorlatinib brain penetration" --since 2015
parallax cite --verify
parallax memory recall "ALK macrocycle"
```

---

## What it does

### Compound intelligence
Resolve a name, SMILES, InChI, InChIKey, CAS, CID or ChEMBL ID into a standardised
structure *with a citation to the record it came from* — never from the model's memory.
Then: computed properties, PubChem's annotated experimental values with their original
references, the ChEMBL record, mechanism of action, and measured bioactivity per target
with a DOI on every value.

### Properties, developability, liabilities
MW, cLogP, TPSA, HBD/HBA, rotatable bonds, Fsp3, stereocentres, QED, SA score,
NP-likeness, Bertz complexity, macrocycle detection. Lipinski, Veber, Ghose, Egan,
Pfizer 3/75, GSK 4/400, lead-likeness, CNS MPO. PAINS/Brenk/NIH/ZINC alerts. Liability
flags (hERG pharmacophore, aniline and nitroaromatic bioactivation, thiophene/furan CYP
risk, ester plasma instability) each paired with *the experiment that would settle it*.

Macrocycles are handled correctly: the tool says so when Ro5 and Veber cut-offs stop
being predictive, instead of calling a macrocyclic drug non-drug-like.

### Potency and activation prediction
Models are built when needed, from data pulled live out of ChEMBL — nothing pre-trained
and opaque.

- **Endpoints are kept distinct**: `inhibition` (IC50, biochemical), `binding` (Ki/Kd),
  `activation` (EC50, functional response), `cell_potency`, `any_potency`. Activation is
  a different experiment from inhibition, not the same one relabelled.
- **Scaffold splits by default**, because random splits flatter congeneric series.
  Chronological splits available when years are present — the closest thing to a
  prospective test.
- Replicates aggregated per structure; validity-flagged and censored (`>`/`<`) rows dropped.
- Every prediction returns an **applicability-domain verdict**, an uncertainty band, the
  nearest measured training compounds, and a flag if the query was *in* the training set
  (in which case you should quote the measurement, not the prediction).
- Reported against a **mean-prediction baseline**, so a model that learned nothing is
  visible as such.

A real run: ALK, 1540 compounds, 420 source documents, scaffold-split RMSE **0.69** log
units against a baseline of **1.16**.

### SAR mining and matched molecular pairs
Assemble a curated dataset from ChEMBL, similarity and substructure search, literature and
patents — with the curation funnel reported, not hidden. Then MMPA over the measured data:
which substituent swaps move potency, by how much, how consistently, with the citations
behind each pair. Measured evidence, as a counterweight to any model.

### Synthesis
A 30-template retrosynthesis library (couplings, C–N and C–O bond formation, SNAr,
reductive amination, Mitsunobu, RCM, click, cyanation, protection and FGI) with
conditions, tolerances, failure modes and scale notes. Macrocyclisation ring-openings are
done by explicit bond surgery rather than SMARTS, so a ring-opening gives **one** valid
linear precursor instead of mangled fragments.

Multi-step route search, starting-material assessment, atom-balance validation, and a
feasibility review that names drivers rather than inventing a single score.

The important discipline: **templates propose, literature confirms.** Every step carries a
`precedent_query`, and `reaction_precedent_search` reports precedented and unprecedented
steps differently — including saying plainly when nothing was found and that free sources
cover preparative conditions less well than Reaxys.

Asked for a route to lorlatinib, the top-ranked disconnection is macrolactamisation of the
12-membered ring — which is how it is actually made.

### Design
A transform library with the literature behind each strategy: fluorine and methyl scans,
aza-scan, acid bioisosteres (tetrazole, acylsulfonamide), amide surrogates, BCP for
tert-butyl, difluoromethyl ether, basicity removal for hERG, N-methylation for
permeability. Each analog reports its rationale, expected effect and property deltas.

Plus macrocyclisation design: finds atom pairs whose linking closes an 11–16 membered
ring, builds each design, and reports the rotatable-bond saving and closure chemistry.
Run on crizotinib it proposes macrocycles with **5 rotatable bonds removed** — the
crizotinib → lorlatinib rigidification logic, derived rather than recalled.

### Reports
`write_report` produces a standalone HTML file with embedded structure depictions, the
analysis, and the full bibliography with verification status per DOI. Nothing loads
remotely. Sources are also exported as JSON and BibTeX.

---

## Continual memory

Memory is a directory of markdown files with an index you can read:

```
memory/
  MEMORY.md          # the human index, grouped by type
  entries/*.md       # one file per fact, frontmatter + body
  index.json         # machine index
```

Seven types: `finding` (a cited result), `heuristic` (a generalisation),
`failure` (a dead end — the most valuable), `sop`, `model`, `preference`, `entity`.

What makes it continual rather than a log:

- **Near-duplicates merge and corroborate.** Writing the same conclusion twice raises its
  confidence instead of cluttering the store.
- **Confidence moves with evidence.** `reinforce` raises it; `contradict` halves it and
  records the conflict in the entry; unused entries decay.
- **Entries can be superseded**, and superseded ones leave recall.
- **Retrieval is lexical and deterministic** — no embedding service, works offline.
- **A memory briefing is injected at session start**, so the agent begins knowing what the
  workspace established, which dead ends to avoid, and how you like to work.

It is plain markdown on purpose. Open `memory/MEMORY.md`, disagree with something, edit or
delete it — the agent reads your correction and will not re-derive what you fixed.

---

## Licensed databases (Reaxys, CAS)

Reaxys has no open endpoint, and the API surface differs between institutional
deployments. Parallax does not guess at a protocol. It defines the adapter contract and
routes to whatever you have:

```bash
# Option 1 — your own authenticated client (the supported path for most sites)
REAXYS_ADAPTER=yourpkg.reaxys_client:client

# Option 2 — a generic HTTP endpoint
REAXYS_API_URL=...   REAXYS_API_KEY=...
```

With nothing configured, every Reaxys call returns `available: false` with the setup
instructions and the public fallback that answers the same question (ChEMBL source
documents + Europe PMC + PubChem patent cross-references). **It never invents Reaxys
content**, and the agent is instructed to say which sources it actually searched.
`parallax doctor` and the `reaxys_status` tool report the true state, so "it's not in
Reaxys" is never claimed on the basis of a database that was never queried.

`CAS_API_KEY` (free tier), `PATENTSVIEW_API_KEY` (free) and `OPENALEX_API_KEY` are
optional and unlock the corresponding sources.

---

## Safety

Parallax is for legitimate research, and there is a gate that keeps it there. Structural
SMARTS screens (so renaming does not bypass them) plus intent screening cover chemical
warfare agents, energetic materials, controlled-substance scaffolds and clandestine
production, and acute toxins.

Synthesis tools screen in `synthesis` mode, where a match is **binding** — no routes, no
precursors, no reagents, no conditions. Analysis stays available in `analysis` mode,
because knowing a compound is dangerous is defensive information: identity, hazard
classification, detection and regulatory context are not restricted. Every non-clear
verdict is written to `.parallax/safety-audit.jsonl`.

The agent is instructed not to help route around a block, and the planner refuses before
generating anything.

---

## Layout

```
src/parallax/
  config.py        workspace layout, settings, credentials
  net.py           cached, throttled, retrying HTTP (reproducible + polite)
  provenance.py    Citation, CitationRegistry, DOI verification, method citations
  safety.py        structural + intent screening, audit log
  report.py        standalone HTML reports
  cli.py           the command line
  chem/            structure, descriptors, reactions, retro, analogs
  sources/         pubchem, chembl, europepmc, crossref, openalex, patents, reaxys, nci
  models/          QSAR with scaffold/time splits, applicability domain, model cards
  memory/          the continual-learning store
  agent/           tools (41), skills loader, prompts, session loop, MCP server
.claude/skills/    10 workflow skills
tests/             88 tests (`-m network` for the live ones)
```

## Tests

```bash
.venv/bin/python -m pytest -m "not network"   # offline: chemistry, safety, memory, dispatch
.venv/bin/python -m pytest -m network         # live: sources, DOI verification, training
```

The safety tests specifically check that renaming a restricted structure does not bypass
the screen, and that the route planner refuses before generating.

---

## Chemical scope

Built for **organic small-molecule drug discovery**. It runs on anything RDKit parses, but
judgement quality degrades outside that space, and the degradation is not always obvious:

| Chemistry | Status |
|---|---|
| Drug-like small molecules, kinase inhibitors, macrocycles | Designed for this |
| Natural products, peptides, carbohydrates | Descriptors and retro work; drug-likeness rules are less meaningful |
| Salts and co-crystals | Largest fragment kept (the free acid/base), reported in `notes` |
| Organometallics, coordination compounds | **Not supported** — flagged `properties_reliable: false` |
| Polymers, materials, inorganics | Out of scope |

The reaction templates (30) and transform library (12) are medicinal-chemistry oriented:
amide couplings, cross-couplings, SNAr, macrocyclisation, bioisosteres. There is nothing
for organometallic synthesis, polymerisation, C-H activation or photoredox.

Targets are not restricted: any ChEMBL target with enough measured data can be modelled.
Nothing is hardcoded to kinases.

## Honest limits

- **Retrosynthesis is template-based.** It proposes; it does not know. Precedent search is
  what makes a route real, and an unprecedented step is reported as unprecedented.
- **No yield prediction.** The tools do not estimate yields and the agent is told not to.
- **Starting-material availability is a proxy** (PubChem presence + simplicity), not a
  supplier quote. The result says which test it passed.
- **Models are as good as ChEMBL is for that target.** Sparse data and assay noise set a
  floor: compounds with >1 log unit replicate spread are counted and reported.
- **Free patent text search is partial.** Structure-anchored lookup through PubChem is
  much more complete; the tool says so rather than letting an empty result read as
  "nothing is patented".
- **Macrocycle permeability is not predicted.** 2D descriptors do not capture
  chameleonicity, and the tools say so instead of guessing.
- **Every computational design needs experimental validation.** That sentence is in the
  footer of every report for a reason.
