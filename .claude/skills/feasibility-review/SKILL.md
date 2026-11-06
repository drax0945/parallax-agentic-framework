---
name: feasibility-review
description: Judge whether a compound or a proposed route can actually be made, and at what cost and risk. Use when asked if something is synthesisable or feasible, to review a proposed route, to choose between candidates on makeability, or to assess scalability.
---

# Feasibility review

Feasibility is not one number. It is a set of named drivers that are not commensurable,
and the honest answer identifies which driver dominates.

## Procedure

### 1. Assess the target

```
synthesis_feasibility(smiles="<target>")
```

Returns the drivers, each marked favourable / neutral / adverse:

- **synthetic accessibility** (SA score): <3 routine, 3–4.5 standard medicinal chemistry,
  >4.5 demanding
- **macrocyclisation**: ring size decides difficulty. ≤11 is the hardest class; 12–14 is
  the practical window; high dilution is mandatory and oligomerisation is the competitor
- **stereochemistry**: each centre means an asymmetric step, a chiral building block, or a
  resolution — and resolution costs half the material per centre
- **protecting groups**: free acids, primary amines, alcohols, phenols and thiols each
  imply protection/deprotection steps and a Pd-poisoning risk
- **size**: MW >600 usually forces a fragment-coupling strategy and harder purification

Deliberately there is no single score. Report the dominant adverse driver.

### 2. Assess the route, if there is one

```
retro_plan(smiles="<target>")
synthesis_feasibility(smiles="<target>", route=<route object>)
```

Then the questions that decide whether it runs:

- **Which is the weakest step?** Name it. Everything else is secondary.
- **Is the key step precedented?** `reaction_precedent_search` on each step. Unprecedented
  is not fatal, but it changes the risk and the timeline.
- **Are the starting materials real?** `starting_material_check` is a proxy — PubChem
  presence plus simplicity — not a supplier quote, and you must say that.
- **Longest linear sequence**, not total steps, governs throughput.
- **Chemoselectivity**: does any step match multiple equivalent sites? The disconnection
  output flags this; an unresolved flag means the step as drawn gives a mixture.

### 3. Scale and practicality

Flag these specifically, because they are invisible in a scheme and decisive at scale:

| Flag | Why it matters |
|---|---|
| high dilution (macrocyclisation) | solvent volume dominates cost; often the limiting step |
| stoichiometric phosphine (Mitsunobu) | phosphine oxide removal; poor atom economy |
| cyanide source (cyanation) | handling, quench and permitting |
| Pd catalysis | metal removal to <10 ppm in API |
| cryogenic organometallics | equipment constraints |
| chromatographic purification of an intermediate | fine at mg, a problem at kg |
| base-sensitive stereocentre with a strong base step | epimerisation risk |

### 4. Verdict

Write it as:

> **Feasible with one hard step.** SA score 4.9 and a 12-membered macrolactam closure.
> The closure is the dominant risk: high dilution required, oligomerisation competing, and
> [precedent status]. The other four steps are routine (amide coupling, SNAr, Boc
> protection/removal). Starting materials look like catalogue items by the structural
> proxy, unconfirmed against a supplier. Longest linear sequence 5.

Then the one experiment that would most reduce uncertainty — usually a model study of the
weakest step on a simplified substrate.

Avoid "feasible" or "not feasible" without naming the driver. Avoid implying certainty
about yields; the tools do not predict yields and neither should you.

### 5. Remember

```
memory_remember(type="finding", title="<target> feasibility: dominated by <driver>",
                body="<drivers, weakest step, precedent status>", citations=[...], confidence=0.7)
```

If a route was rejected, record it as a `failure` with the reason — that is what stops the
next session proposing it.
