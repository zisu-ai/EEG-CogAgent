# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project intent and claim boundaries (read first)

EEG-CogAgent is a reproducible, **agent-orchestrated** resting-state EEG biomarker workflow for dementia differential analysis (AD / FTD / HC) on public BIDS data. The language model's role is to *standardize, document, audit, and draft* — not to diagnose.

These boundaries are enforced everywhere (README, `agent.py`, the skill in `skills/eeg-cogagent/SKILL.md`) and any code or prose change must respect them:

- Say "LLM-assisted orchestration," never "LLM diagnosis." Say "biomarker screening," never "clinical decision system."
- Do not claim novelty as generic dementia EEG classification — the priority dataset already has ML/DL studies.
- Do not claim state-of-the-art without a matched benchmark.
- Do not equate cross-condition (ds006036) or internal CV with **independent external** validation. ds006036 contains the **same participants** as ds004504 (photomark task) and must never be called an external cohort.
- Reproducibility, interpretability, artifact auditing, and automated reporting are the contribution — not the classifier itself.

## Environment setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .          # installs the `eeg-cogagent` console script
python -m pip install -e ".[test]"  # pytest
```

Optional extras (pyproject.toml): `boosting` (xgboost), `download` (openneuro-py), `submission` (python-docx, Pillow for DOCX/PDF builds).

Data is **not** in the repo. Download OpenNeuro `ds004504` into `data/ds004504` (see `scripts/download_ds004504.ps1`). The OSF external-validation archive lives at `data/osf_2v5md/EEG_data.zip`. This is **not** a git repository.

## Common commands

CLI (defined in `eeg_cogagent/cli.py`, registered as `eeg-cogagent` in pyproject):

```
eeg-cogagent plan  configs/ds004504_minimal.yaml              # print the analysis contract
eeg-cogagent run   configs/ds004504_minimal.yaml              # full pipeline
eeg-cogagent run   configs/ds004504_minimal.yaml --subjects-limit 6   # stratified smoke test
eeg-cogagent audit configs/ds004504_minimal.yaml [--strict]   # artifact/provenance audit
```

Tests:

```
python -m pytest tests/                       # full suite (also covers external-validation via tests/test_*.py)
python -m pytest tests/test_audit.py          # one file
python -m pytest tests/test_ml.py -k residualizer   # one test
```

Post-hoc extension scripts (independent of the `run` command — each reads a config and a prior result directory):

```
python scripts/qc_ds004504.py            --config configs/ds004504_minimal.yaml    # QC + confound + sensitivity
python scripts/adjusted_analysis.py      --config configs/ds004504_minimal.yaml    # covariate-adjusted feature tests
python scripts/pairwise_analysis.py      --config configs/ds004504_minimal.yaml
python scripts/residualized_analysis.py  --config configs/ds004504_minimal.yaml    # leakage-safe nuisance residualization
python scripts/connectivity_analysis.py  --config configs/ds004504_minimal.yaml --workers 6
python scripts/generate_framework_figure.py --config configs/ds004504_minimal.yaml
python scripts/cross_condition_validation.py                                   # after also running ds006036
python scripts/osf_walk.py                                                     # inspect the OSF archive
```

A standard complete workflow (from README): smoke-test with `--subjects-limit 6`, then full `run`, then the extension scripts above, then `audit`.

## Architecture

### Config-driven pipeline
Everything is parameterized by a YAML file in `configs/` (see `ds004504_minimal.yaml`, `ds006036_cross_condition.yaml`). `config.load_config` resolves relative paths against the config file's parent's parent (`_project_root`) via `project_path()`. Key sections: `preprocessing` (filter band, notch, reference, epoch length, `max_minutes`, `reject_uv`), `features.bands` / `ratios` / `regions`, `connectivity` (metrics + wPLI graph threshold), `statistics`, `ml`, `report`.

### Core pipeline (`eeg-cogagent run`, orchestrated by `cli.py`)
Per-subject loop with failure capture, then aggregate-and-analyze:

1. `bids.load_participants` — reads `participants.tsv`, maps `Group` (A/F/C) → label (AD/FTD/HC) via `label_map`.
2. `preprocess.make_epochs` — MNE-BIDS load → bandpass → notch → average reference → crop to `max_minutes` → fixed-length epochs → `drop_bad` with `reject_uv`.
3. `features.extract_subject_features` — Welch PSD → log10 band power per channel / region / global, plus theta/alpha and delta/alpha log-power ratios. This is the **absolute** feature space.
4. Aggregate into `features.csv` (+ covariates Age/Gender/MMSE/n_epochs per subject).
5. `stats.run_feature_statistics` — Kruskal-Wallis + Benjamini-Hochberg FDR (`q_value`). `run_pairwise_feature_statistics` does per-comparison Mann-Whitney with rank-biserial effect size.
6. `ml.evaluate_models` — nested stratified k-fold (outer = `cv_folds` capped by smallest class; inner grid search) over `logistic_regression`, `svm_rbf`, `random_forest`. Writes `model_metrics.csv` + out-of-fold `model_predictions.csv`.
7. `viz.*` → figures/ ; `report.write_report` → `auto_report.md` (Jinja template, Methods/Results drafted from deterministic numbers only).

All `run` outputs land in `results/<output_dir>/`. `failed_subjects.csv` is written (and removed if empty). `agent_plan.md` records the analysis contract.

### Leakage-safe ML patterns (must not regress)
- Splits and metrics are **participant-level only** — never epochs/channels/conditions as samples.
- Imputation, scaling, feature selection, and hyperparameter tuning are fit **inside training folds** (nested CV).
- `NuisanceResidualizer` (`ml.py`) is a scikit-learn transformer that residualizes EEG variance against Age / binary-gender / log-epoch-count using **training-fold data only**, then returns only residualized EEG features (nuisance never reaches the classifier). Used by `residualized_analysis.py` via `evaluate_residualized_models`.

### Audit and provenance (`audit.py`)
`eeg-cogagent audit` checks artifact presence, unique-participant integrity, finite feature values, q-value/metric ranges in [0,1], out-of-fold prediction coverage, optional extension artifacts, plus a SHA-256 `artifact_manifest.json` and recorded software versions. Overall status is `fail` > `warn` > `pass`; `--strict` exits non-zero on fail. Treat failed checks as blockers, warnings as things to explain — never silently delete `failed_subjects.csv`.

### External validation (`external_osf.py`, Phase 1)
AD-vs-HC only (binary; FTD/three-class stays internal). Reads the OSF `EEG_data.zip` **directly from the ZIP** in memory (never extracts/mutates, preserving its SHA-256) with strict path-slip validation. All learned components (imputation, scaling, feature selection, classifier, hyperparameters, threshold) are **fit on ds004504 only** — never on OSF. Because the OSF files are 8 s / 1024 samples at 128 Hz with no calibration metadata, it uses **scale-invariant relative powers + band ratios** (not the absolute log10 PSD of the discovery set) — see `feature_mapping()`. Domain-shift is audited without using external labels for selection. Cohort is **80 AD + 12 HC, eyes-closed only** (public 160/24 counts are recordings/conditions, not unique people).

### Configs vs datasets
- `ds004504` (task `eyesclosed`) — discovery set. 19 channels of standard 10-20. Cohort AD 36 / FTD 23 / HC 29.
- `ds006036` (task `photomark`) — **same participants**, used for paired cross-condition testing only.
- OSF node `2v5md` — the only genuinely independent external data (binary AD vs HC).

## Manuscript / submission tooling (secondary)

`docs/` holds the manuscript(s) (`FULL_MANUSCRIPT.md`, `FULL_MANUSCRIPT_JNM.md`, figure-caption and plan markdowns) plus generated DOCX/PDF. `scripts/build_*_docx.py` and `build_jnm_*` regenerate the Journal of Nuclear Medicine submission bundle into `submission/JNM/` and QA renders into `work/`. `literature/` holds the search record, BibTeX, and evidence library. When editing manuscripts, source every number from CSV/JSON artifacts rather than restating it from memory.

## Conventions

- Windows-first repo; examples use PowerShell and backslash paths. A Bash tool is available for POSIX scripts.
- `from __future__ import annotations` and type hints are used throughout — match this style.
- Subject IDs in `participants.tsv` carry the `sub-` prefix; `bids.subject_code` / `preprocess.make_epochs` strip it for `BIDSPath`.
- The reserved metadata columns `{participant_id, label, Group, Gender, Age, MMSE, n_epochs}` are auto-excluded from both statistics and ML feature columns — preserve these exclusions when adding feature sources.
- `eeg_cogagent.egg-info/`, `.pytest_cache/`, `.venv/`, `results/`, `work/`, `.codex/`, `.claude-runs/`, and `prompts/*.stream.log` are generated/runtime artifacts — do not hand-edit.
