---
id: 20261106-macrocyclising-crizotinib-removes-all-5--44d8
type: finding
title: Macrocyclising crizotinib removes all 5 rotatable bonds; predictions fall outside the model's domain
tags: [alk, macrocycle, design, crizotinib, applicability-domain]
citations: [model:chembl4247_inhibition, doi:10.1021/jm1012374]
entities:
  targets: [CHEMBL4247]
confidence: 0.6
support: 1
contradictions: 0
uses: 3
created: 2026-11-06T17:27:47+00:00
updated: 2026-11-06T17:27:47+00:00
---

# Macrocyclising crizotinib removes all 5 rotatable bonds; predictions fall outside the model's domain

Design C[C@H]1Oc2cc(cnc2N)-c2cnn(c2)C2CCNCC2Cc2cc(F)c(Cl)c1c2Cl closes a 14-membered ring via a CH2 tether, taking crizotinib from 5 rotatable bonds to 0. Predicted pIC50 (biochemical inhibition) 8.36 ±1.06, but the applicability domain says: "outside domain — extrapolation, treat as a ranking hint only" (max Tanimoto 0.419 to training data). The general lesson, which outlasts this particular design: a model trained on a largely acyclic ALK series cannot score macrocyclic designs quantitatively. Treat its output as ranking only, and get measured data on a macrocyclic subset before trusting numbers. Makeability: SA score 6.0 and the closure is C-C coupling or alkylation under high dilution; consider an intramolecular Suzuki

**Evidence:** `model:chembl4247_inhibition`, `doi:10.1021/jm1012374`
