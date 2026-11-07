---
name: lead-optimisation
description: Run a multi-objective lead optimisation cycle — selectivity profiling, multi-parameter scoring, and choosing which compound to make next. Use when asked which compound to progress, how to balance potency against properties, whether something is selective enough, to triage a design set, or to plan the next round of a medicinal chemistry campaign.
---

# Lead optimisation

Optimising one property is easy and usually wrong. A lead optimisation decision is a
trade: potency against lipophilicity, selectivity against synthetic cost, speed against
confidence. This skill makes the trade explicit rather than hiding it behind one number.

## The cycle

### 1. Establish where you are

```
memory_recall(query="<project/target> lead optimisation")
compound_profile(query="<current lead>")
selectivity_profile(compound="<current lead>", primary_target="<target>")
```

`selectivity_profile` is the question people forget. It returns measured activity across
every target ChEMBL holds for the compound, with fold-selectivity against your primary and
a flag for anything within 10-fold.

Read its caveat seriously: selectivity is only as broad as what has been tested. An
off-target that nobody assayed does not appear. For kinases, ask whether a published
kinome panel exists.

### 2. State the objective explicitly, before designing

Write down what must improve and what must not regress. Then pick the profile that
matches the project:

| Profile | For |
|---|---|
| `oral_small_molecule` | conventional oral drug |
| `cns` | brain penetration required |
| `lead_like` | early series, keeping room to grow |
| `antimicrobial` | antibacterials — different property space entirely, polarity helps |
| `fragment` | fragment hits, rule of three |

`optimisation_profiles()` shows the objectives and weights in each. If none fits, say so
and define custom objectives rather than forcing a bad match — an antibacterial scored on
oral-drug rules will rank the wrong compounds.

### 3. Generate and score

```
design_analogs(smiles="<lead>")            # or design_macrocycles for rigidification
multiparameter_score(candidates=[...], profile="cns", model="<trained model>")
```

The score is a weighted geometric mean of per-objective desirabilities, so a hard failure
on any one objective sinks a candidate instead of being averaged away. That is deliberate:
a compound with no solubility is not rescued by excellent potency.

**The most useful field is `limiting_objective`** — the single property holding each
candidate back. Report it for every shortlisted compound, because it tells the chemist
what to fix rather than just how the compound ranked.

### 4. Check the modality assumptions hold

```
covalent_check(smiles="<candidate>")
```

If a warhead is present, the usual framework breaks: potency becomes time-dependent, an
IC50 without a stated pre-incubation time is uninterpretable, and an equilibrium QSAR
model is the wrong tool for ranking. Say so rather than quoting a predicted IC50.

### 5. Confirm it can be made

```
synthesis_feasibility(smiles="<each shortlisted design>")
reaction_precedent_search(query="<the key step>")
```

### 6. Recommend, with the trade stated

Lead with the recommendation, then the trade you are making and what would change your
mind:

> **Make compound 4 first.** Predicted pIC50 7.9 (in domain), MPO 0.71 on the CNS profile,
> limited by TPSA at 105 — which is tolerable given lorlatinib ships at 110. Selectivity
> against MET is only 8-fold, so run the MET counter-screen in the same plate. Two steps
> from a catalogue aminopyridine; the SNAr closure has precedent [doi:...].
>
> The trade: compound 7 is predicted half a log more potent but needs a 6-step route and
> an unprecedented macrocyclisation. If the MET counter-screen on 4 fails, 7 becomes the
> fallback.

Always name: the compound, the limiting objective, the key risk, and the one experiment
that would discriminate.

## Things that go wrong here

- **Ranking on predicted potency alone.** The model's RMSE is usually 0.6–0.8 log units,
  so the top three predictions are frequently a tie. Use MPO to break it on properties
  that are *calculated exactly* rather than predicted.
- **Scoring macrocycles or covalent compounds with a model trained on neither.** Check the
  applicability domain; say "ranking hint" and mean it.
- **Treating absence of off-target data as selectivity.** It is absence of testing.
- **Using the oral profile for an antibacterial.** Gram-negative entry favours low logP and
  higher polarity — the opposite of oral drug-likeness.

### 7. Remember the decision

```
memory_remember(type="finding", title="<which compound was chosen and why>",
                body="<the trade, the limiting objective, what would falsify it>",
                citations=[...], confidence=0.6)
```

Record rejected designs as `failure` entries with the reason — that is what stops the next
cycle re-proposing them.
