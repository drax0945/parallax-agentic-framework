---
name: route-variation
description: Generate and compare alternative synthetic routes to the same target, trading off step count, convergence, precedent, cost and scalability. Use when asked for other ways to make a compound, variations on a route, a more scalable or shorter route, or to compare two routes.
---

# Route variation

One route is an answer; several routes with a comparison is a decision. This skill
generates genuinely different strategies — not cosmetic variations of the same
disconnection — and compares them on the axes that decide which one gets run.

## What counts as a real variation

Two routes are meaningfully different if they differ in at least one of:

- **the strategic bond** — which bond is formed last (for a macrocycle, where the ring
  closes);
- **convergence** — linear vs fragment-coupling, and where the longest sequence sits;
- **the chemistry class** — e.g. Pd-catalysed coupling vs metal-free SNAr for the same bond;
- **the order of operations** — building the sensitive group early vs late;
- **protecting-group strategy** — whether a group is protected at all.

Four routes that all close the same amide with different coupling agents are one route.

## Procedure

### 1. Enumerate strategic options

```
retro_disconnect(smiles="<target>", top=10)
```

Group the disconnections by strategic bond and reaction class. Pick the distinct
strategies — typically 2–4 — rather than the top N by score.

### 2. Develop each into a route

```
retro_plan(smiles="<target>", max_depth=3, beam=2, max_routes=4)
```

Where a strategy you care about is not represented, drive it manually: disconnect the
precursor you want with `retro_disconnect` and continue from there.

### 3. Look for the metal-free and the convergent alternative deliberately

Two variations worth generating every time, because they usually win at scale:

- **Metal-free substitution for a coupling.** If a C–N bond is being made by
  Buchwald-Hartwig on an electron-poor heteroaryl halide, check whether plain SNAr will
  do it: no Pd, no ligand, no metal removal. `retro_disconnect` proposes both — compare them.
- **Convergent fragment coupling.** Splitting a 7-step linear sequence into two 3-step
  fragments plus a coupling shortens the longest linear sequence, which is what actually
  governs throughput.

### 4. Precedent each distinct strategy

```
reaction_precedent_search(query="<strategy-specific query>")
```

Precedent is often the deciding factor between two otherwise equal routes. A route whose
key step has a close precedent beats a theoretically shorter one that does not.

### 5. Assess each

```
synthesis_feasibility(smiles="<target>", route=<route object>)
starting_material_check(smiles="<each starting material>")
```

## The comparison table

Produce exactly this, one row per route:

| | Route A | Route B | Route C |
|---|---|---|---|
| Strategic bond formed last | | | |
| Steps / longest linear sequence | | | |
| Convergent? | | | |
| Weakest step | | | |
| Precedent for the key step | citation or "none found" | | |
| Starting materials likely available | | | |
| Metal catalysis needed | | | |
| Scale concerns | high dilution / stoichiometric reagent / metal removal | | |
| Stereochemistry risk | | | |

Then a recommendation in two or three sentences: which route for a first-pass milligram
synthesis (usually the best-precedented), which for scale (usually the most convergent,
metal-free one), and what single experiment would discriminate them.

Be explicit that these are hypotheses ranked by heuristics and precedent coverage, not
validated routes.

### 6. Remember the comparison

```
memory_remember(type="heuristic",
  title="For <chemotype>, <strategy> beats <alternative> because <reason>",
  body="<the comparison and what decided it>", tags=[...], citations=[...], confidence=0.6)
```
