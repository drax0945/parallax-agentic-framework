---
name: synthesis-planning
description: Plan a synthesis of a given compound — retrosynthetic disconnections, multi-step routes, conditions, precedent search and the weakest-step analysis. Use when asked how to make/synthesise a compound, for a synthetic route or scheme, to find a precedent for a step, or to assess a proposed route.
---

# Synthesis planning

The purpose of this skill is to produce a route a chemist could actually take to the
bench, and to be explicit about which parts are precedented and which are hypotheses.
A route presented without that distinction is worse than no route.

## Procedure

### 1. Screen and resolve

```
resolve_compound(query="<target>")
safety_screen(smiles="<smiles>", text="<the user's request>", mode="synthesis")
```

The synthesis tools screen automatically, but doing it first means you know where you
stand. A `blocked` verdict is final: report it, keep identity/hazard/regulatory
information available, and do not look for a way around it.

### 2. Recall

```
memory_recall(query="synthesis <chemotype> <key transform>", types=["sop","failure","heuristic"])
```

Previous sessions may have established that a particular closure failed or that a
supplier route exists. Failures are the most valuable memories here.

### 3. Understand the target before cutting it up

```
computed_properties(smiles="<smiles>")
synthesis_feasibility(smiles="<smiles>")
```

Note: stereocentre count, macrocycle and ring size, SA score, reactive groups that will
need protection. This tells you what kind of problem you have before you see any
disconnection.

### 4. One-step disconnections

```
retro_disconnect(smiles="<smiles>", top=8)
```

Each disconnection gives conditions, what it tolerates, where it fails,
chemoselectivity concerns, and a `precedent_query`. Read the concerns: a step flagged
"N equivalent sites match — not regioselective as drawn" is not a step, it is a problem
to solve.

For a macrocyclic target, the ring-opening disconnections are usually the strategic
choice. The question is not only *can it close* but *which bond leaves the most
convergent fragment pair*.

### 5. Full routes

```
retro_plan(smiles="<smiles>", max_depth=3, beam=3, max_routes=4)
```

Returns forward-ordered steps, starting materials with an availability assessment, and a
route score. The score ranks hypotheses; it is not a probability of success.

### 6. Precedent — this is the step that makes the route real

For **every** step you intend to stand behind:

```
reaction_precedent_search(query="<the step's precedent_query, made specific to your substrate>")
```

Make the query specific: "macrolactamisation 12-membered ring aminopyridine HATU high
dilution" finds more than "amide coupling". Then:

- **precedented** — cite the report, and note how close the precedent substrate is.
- **no precedent found** — say exactly that. Add that free sources cover preparative
  conditions less thoroughly than Reaxys/SciFinder, so this is "not found", not "not
  known". If the user wants the licensed answer, run `reaxys_status` and tell them
  whether access is configured here.

Never present a template-generated step as though it had a precedent.

### 7. Check your chemistry on paper

```
validate_reaction(reaction_smiles="<reactants>>><product>")
```

Atom imbalance is the commonest error in machine-written chemistry. Small imbalances are
fine where leaving groups or water are implicit — but you should know which case you are
in.

```
starting_material_check(smiles="<each starting material>")
```

This is a proxy (PubChem presence + simplicity), not a supplier quote. Say so.

## Reporting format

Give the route as a numbered forward sequence. For each step:

**Step n — <transform name>**
- Reaction: `reactant(s) >> product`
- Conditions: reagent, base, solvent, temperature
- Precedent: citation, or "none found in the sources searched"
- Watch out for: the chemoselectivity or stereochemistry problem, and what you would try
  if it fails

Then:

- **Starting materials** — with availability assessment
- **Longest linear sequence** and step count
- **Weakest step** — name it. A route is as good as its worst step, and saying which one
  is the single most useful sentence in a route proposal.
- **Scale notes** — where a step is impractical beyond milligrams (high dilution,
  stoichiometric phosphine, cyanide handling, Pd removal)

## Macrocyclisation specifics

If a step closes a macrocycle, always state: ring size; that high dilution (1–10 mM) and
slow addition are needed; that oligomerisation is the competing pathway; and whether a
conformational bias (turn element, pre-organised tether) exists to help. 11-membered and
smaller rings are the hardest class; 12–14 is the practical window. For RCM, note the E/Z
outcome and that hydrogenation afterwards introduces a diastereoselectivity question.

### 8. Remember

```
memory_remember(type="sop", title="Route to <chemotype>: <key step> works under <conditions>",
                body="<what worked, what the precedent was, what to watch>",
                tags=[...], citations=[...], confidence=0.6)
```

Record a `failure` memory for any disconnection you rejected and why — that is what stops
the next session proposing it again.
