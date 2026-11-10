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

## Setup

Three required steps, then one optional one. Nothing here installs a model or a model
runtime — those are separate and covered below.

### 1. Install

Needs **Python 3.11+** and nothing else. `claude-agent-sdk` ships its own `claude`
binary, so Claude Code does **not** have to be installed separately.

```bash
git clone https://github.com/drax0945/parallax-agentic-framework.git
cd parallax-agentic-framework
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install -e .
```

No `uv`? `python3 -m venv .venv && .venv/bin/pip install -e .` works the same.

### 2. Create the workspace

```bash
.venv/bin/parallax init
```

Writes `memory/`, `reports/`, `.parallax/` (cache, models, run transcripts), plus
`.mcp.json` and `.env.example`. Safe to re-run; it will not overwrite what exists.

### 3. Check it

```bash
.venv/bin/parallax doctor
```

Verifies dependencies, the agent runtime, all four data sources and the memory store, and
names anything missing. **If `doctor` is happy, the chemistry works** — run
`parallax compound aspirin` to confirm.

Running `source .venv/bin/activate` once lets you drop the `.venv/bin/` prefix everywhere.

### 4. Optional — credentials, only for the conversational agent

Everything above works with no account anywhere. Only `parallax ask` and `parallax ui`
need a model. See **Where things run** below for how to pick one.

```bash
parallax login                        # Claude, the default
```

---

## Where things run

Three separate places, and it matters which is which.

```
   YOUR MACHINE                                    THE INTERNET
┌──────────────────────────────┐
│  chemistry engine            │
│    RDKit, scikit-learn       │     read-only     ┌─────────────────────┐
│    retrosynthesis, MPO       │ ────────────────▶ │ PubChem   ChEMBL    │
│    memory/  (markdown)       │   cached locally  │ EuropePMC Crossref  │
│    .parallax/ (models,cache) │ ◀──────────────── │ free, no key        │
│                              │                   └─────────────────────┘
│  console  127.0.0.1:8765     │
│  MCP server (stdio)          │                   ┌─────────────────────┐
│                              │   only `ask`/`ui` │ Claude / OpenAI /   │
│  the 50 tools ───────────────│ ────────────────▶ │ Groq / …            │
│  (no model, ever)            │   prompts + tool  │ needs an account    │
└──────────────────────────────┘   results         └─────────────────────┘
            │
            │ or, with Ollama, the model is on your machine too:
            └──────▶ localhost:11434   (no account, nothing leaves the box)
```

**On your machine, always:** every calculation. Structure parsing, descriptors, QSAR
training and prediction, retrosynthesis, MMPA, multi-parameter scoring, the memory store,
the citation registry, the console server (bound to `127.0.0.1`, not your network) and the
MCP server. **None of the 50 tools calls a model.** Verify it yourself:

```bash
grep -rl -e anthropic -e claude_agent_sdk src/parallax/chem src/parallax/sources \
         src/parallax/models src/parallax/memory src/parallax/agent/tools.py
# no matches — the chemistry, data, modelling, memory and tool layers never touch an LLM
```

The only files that reach a model are `config.py`, `cli.py`, `ui/server.py` and the two
agent backends.

**Remote, read-only, no credentials:** the chemistry databases. Responses are cached under
`.parallax/cache/` keyed by the full request, so a repeat analysis is fast and reproducible.
`PARALLAX_OFFLINE=1` forbids new requests and serves only the cache.

**Remote, only when you use the agent:** the model. It receives your question, the tool
descriptions and the tool *results* — so database records and computed values do travel to
whichever provider you choose. With Ollama they do not leave your machine.

### How each model backend works

| Backend | Where the model runs | What it needs | How it is reached |
|---|---|---|---|
| **claude** (default) | Anthropic's servers | `parallax login`, or `ANTHROPIC_API_KEY` | the bundled `claude` binary, spawned as a subprocess |
| **ollama** | **your machine** | Ollama installed + a model pulled | HTTP to `localhost:11434` |
| openai, groq, together, mistral, deepseek, openrouter | that provider's servers | their API key | HTTPS to their `/chat/completions` |
| anything else | wherever you host it | `PARALLAX_LLM_BASE_URL` | same protocol |

**Claude.** `parallax login` opens a browser sign-in and stores credentials in your OS
keychain; the SDK then spawns the bundled binary per request. An `ANTHROPIC_API_KEY` is
used instead when set. Usage counts against whichever you authenticated with.

**Ollama.** Parallax never installs or pulls anything — you do that yourself:

```bash
brew install ollama          # or https://ollama.com/download
ollama pull qwen2.5:14b      # llama3.1 and mistral-nemo also handle tools well
parallax ask --provider ollama "Profile lorlatinib"
```

Parallax asks the local daemon which models are installed and picks a tool-capable one, so
it will not default to something you do not have. Nothing leaves your machine.

**Any other provider.** Set its key and name it:

```bash
export GROQ_API_KEY=...
parallax ask --provider groq "Build an ALK model"

export PARALLAX_LLM_BASE_URL=https://my-gateway/v1   # vLLM, LM Studio, Azure, …
export PARALLAX_LLM_MODEL=my-model
```

`parallax providers` shows what is configured; `PARALLAX_LLM_PROVIDER` makes a choice
permanent. The console's header badge always names the model actually answering, and turns
red when it cannot be reached.

If no model is set up, `parallax ask` exits with a diagnosis and the fix — the underlying
runtime reports "Not logged in" as ordinary assistant text, which is easy to mistake for an
answer.

### One caveat about model choice

The discipline here — separating measured from computed from modelled, refusing
out-of-domain predictions, never inventing a citation — is enforced partly by the tools and
partly by the model following instructions. A small local model follows them less reliably
than a frontier one. The tools still refuse to fabricate, `audit_draft` still catches
invented references, and the console flags an answer as **not grounded** when no tool call
succeeded. The floor holds whatever you run; the ceiling depends on the model.

---

## The console

```bash
parallax ui
```

A local web console at `http://127.0.0.1:8765`. Ask in plain language and watch every
tool call stream in as it runs, with real durations and pass/fail state. The header shows
which model is answering — Claude or a local Ollama model — and a **What can it do?**
drawer explains the evidence tiers, all 50 tools by group, the 11 skills, modellable
endpoints, optimisation profiles, chemical scope and the limits. That drawer is generated
from the code, so it cannot drift from what the agent actually does.

While a run is in flight a **side loader** shows that the agentic loop is still going:
the current phase, elapsed time, and a row of microplate wells that fill as each tool call
completes — so a 120-second run reads as progress rather than a hang. The reply arrives
whole at the end, and the loader says so.

Answers come back **structured**, not as a wall of text: markdown is rendered server-side
so numbers land in real tables with a tier and source column, and every structure the
agent touches is **drawn** — depicted with RDKit from the SMILES in its tool calls and in
its prose, shown above the answer.

It also flags an answer as **not grounded** when the prose carries figures but no tool
call succeeded and nothing was cited — which is what a small local model does after a
tool rejects its arguments.

## Four ways to use it

### The agent

```bash
parallax ask "What is known about crizotinib's ALK potency, and how does lorlatinib's
              macrocyclisation change the property profile?"
```

The agent loads the skills in `.claude/skills/`, recalls what previous sessions learned,
calls the tools, and reports with a bibliography. Runs are saved to `.parallax/runs/`.

### Inside Claude Code or the desktop app

`parallax init` writes `.mcp.json`, so the 50 tools appear in any MCP client:

```json
{ "mcpServers": { "parallax": { "command": "uv", "args": ["run", "parallax", "mcp"] } } }
```

`uv run` resolves the project venv from the working directory, so the server starts
without the venv being active or `parallax` being on your PATH.

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

### As a Python library, for batch work

No agent and no model involved — the chemistry, data and provenance layers import directly.
Results compose into pandas, the citation registry accumulates across calls, and
structures render inline in Jupyter via `parallax.chem.svg()`.

```python
from parallax.chem import resolve
from parallax.chem.analogs import suggest_macrocyclization
from parallax.models import predict
from parallax.sources import chembl
from parallax.provenance import registry

lead = resolve("crizotinib")
data = chembl.activities(target_chembl_id="CHEMBL4247", standard_type="IC50", limit=500)
designs = suggest_macrocyclization(lead.smiles, max_designs=10)["designs"]
scored = predict("chembl4247_inhibition", [d["smiles"] for d in designs])

print(len(registry()), "sources cited so far")
```

This is the right mode for scoring a library, running a nightly job, or anything
reproducible: it is deterministic and needs no Claude access at all.

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

### Selectivity and multi-parameter optimisation
`selectivity_profile` returns measured activity across every target ChEMBL holds for a
compound, with fold-selectivity against the primary and a flag for anything within
10-fold — and it says plainly that an untested off-target is absence of testing, not
evidence of selectivity.

`multiparameter_score` ranks candidates on potency, lipophilicity, size, polarity, HBD,
synthetic accessibility and structural alerts together, using a weighted geometric mean so
a hard failure on one objective sinks a candidate rather than being averaged away. Five
profiles ship — `oral_small_molecule`, `cns`, `lead_like`, `antimicrobial`, `fragment` —
and the antimicrobial profile deliberately inverts the lipophilicity preference, because
Gram-negative entry is not oral drug-likeness.

The most useful output is `limiting_objective`: the single property holding each candidate
back, which tells a chemist what to fix rather than just how it ranked. Validated against
clinical reality — on the CNS profile lorlatinib scores 0.75 and crizotinib 0.15, which is
the difference that actually sent lorlatinib into development.

`covalent_check` detects warheads (acrylamide, chloroacetamide, nitrile, boronic acid,
sulfonyl fluoride and others) and flags that equilibrium IC50 models do not apply to them.

### Potency and activation prediction
Models are built when needed, from data pulled live out of ChEMBL — nothing pre-trained
and opaque.

- **Endpoints are kept distinct**: `inhibition` (IC50, biochemical), `binding` (Ki/Kd),
  `activation` (EC50, functional response), `cell_potency`, `antimicrobial` (MIC),
  `cytotoxicity`, `binding_affinity_relative` (RBA) and `admet`. Activation is a different
  experiment from inhibition, not the same one relabelled.
- **Any ChEMBL endpoint can be modelled**, not just the presets: pass `standard_types`
  explicitly. Values in ug/mL convert to molar pX via molecular weight, which is what makes
  antibacterial MIC data usable — 251 of 300 *S. aureus* rows are MIC, and relying on
  ChEMBL's `pchembl_value` alone would have silently excluded the entire field.
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

## The lab logbook

```bash
parallax log add "HATU coupling onto 2,6-dichloroaniline" -o failure \
  -r "CC(=O)O.Nc1c(Cl)cccc1Cl>>CC(=O)Nc1c(Cl)cccc1Cl" \
  -t "Amide coupling (acid + amine)" --reagent HATU --solvent DMF \
  -n "No conversion by LCMS after 18 h. Aniline too hindered."
```

Primary records of what this lab actually ran — **successes and failures both**. Kept in
`logbook/entries/*.md`, separate from memory on purpose: memory holds the agent's
conclusions and the agent may revise them, while the logbook holds evidence that only a
human should change.

**In-house results are a fourth evidence tier, `observed`, and they outrank everything
else here** — published work included. A paper reporting that a step works does not
overrule a failure on the bench with the substrate in front of you. It is also the only
source of negative results: the literature publishes almost none.

### Retrieval is structural, not text search

Asking "have we run this before?" is a chemistry question. *"Amide coupling failed"* and
*"HATU gave no conversion"* share no words but describe the same event, so ranking is
driven by **product and reactant similarity** first, then the named transform, then text.
Deterministic, no embedding service, works offline.

```bash
parallax log search --product "CC(=O)Nc1c(Cl)cccc1Cl"   # finds both runs above
parallax log record "Amide coupling"                     # this lab's real success rate
```

`reaction_precedent_search` now queries the logbook before the literature and reports the
two separately, so a step you have already failed is flagged before it is proposed again.
Asked how to make that anilide, the agent opens with the logbook rather than a generic
recommendation:

> Your logbook already has the answer for this exact substrate. HATU/DIPEA/DMF failed;
> T3P/pyridine/EtOAc at 80 °C gave 71%. **Use T3P.**

### Every record is addressable by date

A logbook is also a record of *when*, so "what ligand did we use on the Suzuki on
12 March?" has to be answerable. Each condition goes in its own field, which is what makes
it retrievable later — a catalyst written into a sentence cannot be looked up:

```bash
parallax log add "Suzuki onto the ALK core" -o success -y 64 \
  --date 2026-03-12 --operator RK -t "Suzuki-Miyaura coupling" \
  --catalyst "Pd(OAc)2" --ligand SPhos --base K3PO4 --temp "100 C" --time "3 h" \
  -c atmosphere=N2 -n "Full conversion by LCMS in 3 h."
```

Then a date is a complete query — no search text needed:

```bash
parallax log search --date 2026-03-12                 # everything run that day
parallax log search --from 2026-03-01 --to 2026-03-31 # a window
parallax log search suzuki --date 2026-03-12          # pin an ambiguous query
```

```
  SUCCESS · 64%  Suzuki onto the ALK core
      2026-03-12 · Suzuki-Miyaura coupling · RK
      catalyst=Pd(OAc)2, ligand=SPhos, base=K3PO4, temperature=100 C, time=3 h
```

The agent reads the value straight out of the entry and quotes it with the entry id — and
if the field is absent it says the logbook does not record it rather than naming a
plausible catalyst. Three details make this work in practice:

- **Common abbreviations are folded onto one spelling**, so `cat`, `lig`, `temp` and
  `solv` land as `catalyst`, `ligand`, `temperature` and `solvent`. Two chemists typing
  the same thing differently stay searchable as one field. Anything unrecognised is kept
  verbatim.
- **Month names are indexed**, because `2026-03-12` shares no words with "the Suzuki we
  ran in March". The bare year deliberately is not a ranking signal — a shared year should
  never be the reason an entry surfaces, and that is what the date window is for.
- **Filters alone are a valid search.** A bare `--date`, or `--outcome failure` with no
  query, lists what matches newest first instead of scoring everything to zero.

### It stays fast as it grows

The markdown is authoritative, but re-reading every file per query does not scale. A
persistent index under `logbook/.index/` fixes that without changing where the data lives.
Measured on **25,000 drug-sized entries**:

| | before | after |
|---|---|---|
| cold load | 4.28 s | **0.27 s** |
| structural search | 2.31 s | **0.23 s** |
| text search | 0.75 s | **0.14 s** |
| **one cold query** | **6.59 s** | **0.50 s** |

Three things do the work. Fingerprints are packed once into a `uint8` matrix and compared
with a vectorised popcount Tanimoto — 25,000 comparisons in ~8 ms rather than 2.3 s.
Tokens live in an inverted index, so a text query touches only entries sharing a word.
And entry bodies are never loaded to search, only for the handful of results returned.

Refresh is incremental: files are compared by size and mtime, so editing one entry by hand
re-parses one file. The index is a cache and never the source of truth — delete it, or run
`parallax log reindex --rebuild`, and it rebuilds from the markdown. A corrupt index is
discarded rather than raising.

**Memory needs none of this.** It holds the agent's conclusions, which accumulate slowly
and are merged by `memory_consolidate`; at 8,000 entries a cold recall is 0.55 s, so it
stays a plain linear scan. Different growth curve, different answer.

### Two things to keep in mind

**One run is not a rate.** Every in-house claim carries its sample size, and
`logbook_track_record` says so explicitly rather than reporting a bare percentage.

**The logbook is your unpublished IP.** Tool results reach whichever model you use, so a
hosted provider would see your failed routes. If that matters, run the agent against a
local Ollama model — then nothing leaves the machine.

**Neither `logbook/` nor `memory/` is committed.** Both are gitignored, so a clone starts
with an empty workspace and nothing from one lab's bench — or from a test run — can reach
another's. This is not only privacy: a shipped logbook entry would be read as `observed`
in-house precedent by somebody else's agent, which is the one tier that outranks published
work. Both directories are created on first use. To keep a scratch workspace out of the
real one entirely, point `PARALLAX_WORKSPACE` somewhere else:

```bash
PARALLAX_WORKSPACE=$(mktemp -d) parallax log add "trial run" -o success
```

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
  chem/            structure, descriptors, reactions, retro, analogs, optimisation
  sources/         pubchem, chembl, europepmc, crossref, openalex, patents, reaxys, nci
  models/          QSAR with scaffold/time splits, applicability domain, model cards
  memory/          the continual-learning store (the agent's conclusions)
  logbook/         lab experiment records (primary evidence, human-written)
                   + .index/, a rebuildable cache that keeps search fast at scale
  ui/              local web console (stdlib HTTP + SSE, no framework)
  agent/           tools (50), skills loader, prompts, MCP server,
                   and two interchangeable backends: Claude SDK or any
                   OpenAI-compatible endpoint
.claude/skills/    11 workflow skills
tests/            202 tests (`-m network` for the live ones)
```

The 11 skills, which are plain markdown you can read and edit:

| Skill | Covers |
|---|---|
| `compound-dossier` | profiling a compound end to end |
| `property-prediction` | the three tiers, developability, liabilities |
| `potency-modelling` | building and trusting a QSAR model |
| `sar-mining` | assembling a cited dataset from literature and patents |
| `analog-design` | bioisosteres, rigidification, macrocyclisation |
| `lead-optimisation` | selectivity, multi-parameter choice, what to make next |
| `synthesis-planning` | routes, conditions, precedent, the weakest step |
| `route-variation` | generating and comparing alternative strategies |
| `feasibility-review` | can it be made, and what dominates the difficulty |
| `citation-integrity` | verifying references before delivery |
| `memory-curation` | what to remember, and how to correct it |

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
