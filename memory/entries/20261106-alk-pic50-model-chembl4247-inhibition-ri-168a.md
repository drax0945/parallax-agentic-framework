---
id: 20261106-alk-pic50-model-chembl4247-inhibition-ri-168a
type: model
title: "ALK pIC50 model `chembl4247_inhibition`: ridge, scaffold RMSE 0.71"
tags: [model, qsar, alk, chembl4247]
citations: [model:chembl4247_inhibition, database:chembl]
entities:
  targets: [CHEMBL4247]
confidence: 0.8
support: 1
contradictions: 0
uses: 0
created: 2026-11-06T17:27:45+00:00
updated: 2026-11-06T17:27:45+00:00
---

# ALK pIC50 model `chembl4247_inhibition`: ridge, scaffold RMSE 0.71

ridge on 1699 compounds drawn from 137 ChEMBL source documents; endpoint pIC50 (biochemical inhibition). Scaffold-split test R²=0.52, RMSE=0.71 log units, Spearman ρ=0.641. Predicting the training mean gives RMSE 1.046, so the model adds 0.34 log units of signal. Differences below 0.71 log units are noise and must not be interpreted. 1 data note(s) record replicate spread in the training set.

**Evidence:** `model:chembl4247_inhibition`, `database:chembl`
