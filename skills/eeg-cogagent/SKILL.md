---
name: eeg-cogagent
description: Orchestrate and audit reproducible BIDS EEG biomarker studies for dementia and neurological cohorts. Use when Codex needs to inspect or run EEG-CogAgent; plan YAML-driven MNE preprocessing; extract spectral, ratio, connectivity, or graph features; run leakage-safe statistics and machine learning; verify artifacts and provenance; generate publication figures; or draft conservative Methods and Results from deterministic outputs.
---

# EEG-CogAgent

Treat the language model as an analysis orchestrator and scientific-reporting assistant. Keep all numerical claims grounded in deterministic Python outputs. Never infer participant-level diagnoses.

## Route the request

- For a new dataset or configuration, read `references/datasets.md`, validate its BIDS task and labels, then follow the full workflow.
- For an existing result directory, run the audit first. Recompute only missing, failed, or explicitly requested stages.
- For manuscript writing or figure interpretation, read `references/reporting.md` and source every number from CSV/JSON artifacts.
- For exact commands and artifact paths, read `references/workflow.md`.

## Execute the workflow

1. Resolve the project root and YAML config. Confirm the BIDS root, task, label column, group mapping, sampling frequency, channel types, line frequency, and output directory.
2. Print the analysis contract with `eeg-cogagent plan <config>`.
3. Run a stratified smoke test before any full computation. Inspect `failed_subjects.csv`, feature dimensions, class counts, and epoch retention.
4. Run the full subject-level pipeline only after the smoke test passes.
5. Run requested extensions independently: quality control, adjusted statistics, pairwise classification, nuisance residualization, connectivity/graph analysis, cross-condition validation, and manuscript figures.
6. Run `eeg-cogagent audit <config>` after computations. Treat failed checks as blockers. Explain warnings rather than silently deleting them.
7. Draft text only after the audit. Distinguish exploratory biomarker evidence, internal validation, paired cross-condition validation, and independent external validation.

## Preserve validation integrity

- Split and evaluate at the participant level; never treat epochs as independent samples.
- Fit imputation, scaling, residualization, feature selection, and hyperparameter search inside training folds.
- Use nested stratified cross-validation for model comparison.
- Keep ds004504 and ds006036 participants separated by outer fold during cross-condition testing. Do not call ds006036 an external cohort because it contains the same participants.
- Report failed participants and sensitivity analyses. Do not relax thresholds solely to recover a failed case.
- Prefer balanced accuracy, class-specific recall, and out-of-fold AUC over accuracy alone.

## Enforce claim boundaries

- Say “LLM-assisted orchestration,” not “LLM diagnosis.”
- Say “biomarker screening” or “research workflow,” not “clinical decision system.”
- Do not claim novelty as generic dementia EEG classification.
- Do not claim state of the art without a matched benchmark.
- Do not equate internal or paired cross-condition validation with independent external validation.
- Make reproducibility, interpretability, artifact auditing, and automated reporting the primary contribution.

## Finish with evidence

Return the config used, processed and failed participant counts, key output paths, audit status, best internally validated metrics, principal interpretable biomarkers, and explicit limitations. Link to the generated audit and report rather than copying unverified numbers into prose.
