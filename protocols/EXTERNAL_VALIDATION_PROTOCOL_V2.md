# External Validation Protocol v2 — OSF AD vs HC

Status: **predeclared, frozen on 2026-07-01 before any v2 OSF evaluation.**
This protocol is version-controlled at `protocols/EXTERNAL_VALIDATION_PROTOCOL_V2.md`
and copied verbatim into each v2 result directory at publish time.

## Scope and claim boundaries

- **Independent external evaluation / generalization check** of a discovery
  model on the OSF `2v5md` archive. Not a clinical diagnostic claim, not a
  clinical-validity claim, not a state-of-the-art claim.
- **Binary AD vs HC only.** FTD is kept internal/exploratory; it never enters
  the external model training or the primary domain-shift baseline.
- `ds006036` is the same participants as `ds004504` and is **not** treated as an
  external cohort.
- **Not prospective / not blinded.** OSF labels and prior v1 results were
  already inspected on 2026-07-01. v2 is therefore a *method-audited* external
  evaluation, not a confirmatory validation on never-seen outcomes. This is
  stated explicitly in the report and review request.

## Data

- **Discovery:** OpenNeuro `ds004504` (CC0, DOI `10.18112/openneuro.ds004504.v1.0.9`,
  BIDS v1.2.1), 88 subjects (36 AD / 23 FTD / 29 HC). Training uses **AD+HC only
  (65)**.
- **External:** OSF node `2v5md`, file `EEG_data.zip`, canonical SHA-256
  `f5b30df4fd0d18e3224dde0bd564e2a5cea61845ae5a9b8142ae722c5d99ba93`. Eyes-closed
  only: **80 AD + 12 HC = 92 subjects**, 19 common 10-20 channels, 1024 samples
  @ 128 Hz (8 s).
- **License:** the associated article (DOI `10.1038/s41598-023-32664-8`) is
  CC BY 4.0, but the OSF dataset node `node_license` is `null`. The article
  license does not override the dataset-node license, so the **dataset reuse
  license is `UNRESOLVED`** (both are recorded in provenance).
- **Source preprocessing of OSF:** per the article, the OSF signals are
  source band-limited to **0.5–30 Hz** and movement artifacts were removed
  manually by an EEG technician. v2 does not additionally FIR-filter the short
  OSF records.
- **Archive vs publication discrepancy:** the archive also contains F1/F2 in
  addition to the 19 channels listed in the article; v2 uses the common 19 and
  excludes F1/F2 (recorded in provenance, not hidden).
- No OSF age/sex/MMSE metadata is available, so demographic confounds cannot be
  tested or adjusted.

## Feature space (predeclared)

- **Bands (half-open [low, high), 1–30 Hz common support):** delta [1,4),
  theta [4,8), alpha [8,13), beta [13,30). **Gamma is excluded** because the OSF
  source is band-limited to 0.5–30 Hz.
- **Relative-power denominator = the four common bands only** (1–30 Hz).
- **36 features:** 4 global relative powers + 20 regional relative powers
  (4 bands × 5 regions) + 2 global ratios (theta/alpha, delta/alpha) + 10
  regional ratios (2 ratios × 5 regions). Per-channel columns are excluded.
- **Same function, band edges, half-open rule, and Welch 0.5 Hz resolution on
  both datasets** (OSF 128 Hz / nperseg 256; ds004504 500 Hz / nperseg 1000).
  Feature *definition* and *analysis frequency support* are identical;
  acquisition device, source preprocessing, record length, and local QC differ
  (genuine domain shift).
- Relative powers + log band-power ratios are scale-invariant; the OSF text
  files carry no unit/calibration metadata.

## Model and training (predeclared, discovery-only)

- **Primary model:** L2 Logistic Regression, `class_weight="balanced"`.
  No deep learning, no ensembles, no model comparison.
- **C grid:** {0.1, 1.0, 10.0}, selected by inner stratified CV, scoring
  `balanced_accuracy`.
- **Internal nested-CV estimate (unbiased):** outer 5-fold (capped by the
  smaller class). Per outer fold, C **and** the decision threshold are chosen on
  the outer-*training* data only (C via inner GridSearchCV; threshold via
  inner cross-fitted OOF probabilities), then frozen and applied to the
  outer-test fold. Outer-test labels never enter C/threshold selection.
- **Final external model:** fit on **all** ds004504 AD/HC; C selected by CV on
  all discovery; threshold selected by cross-fitted OOF on all discovery. Both
  are discovery-only. OSF is used only at `predict` time.
- **Threshold search rule:** `argmax balanced_accuracy` over
  `np.linspace(0.01, 0.99, 99)`. **Tie-break: lowest threshold** (deterministic).
- **Sensitivity analysis (predeclared):** also report external metrics at the
  fixed threshold 0.5. The primary threshold is not changed after seeing OSF
  results.

## Statistics

- **Primary metrics:** balanced accuracy, ROC AUC, sensitivity, specificity,
  confusion matrix. Accuracy is reported but is **not** primary (80/12 imbalance).
- **BA and AUC:** subject-level, class-stratified bootstrap 95% CI, **10,000
  resamples**, fixed seed (42). CIs are *conditional on the fitted discovery
  model*; they do not account for discovery training-sample uncertainty.
- **Sensitivity/specificity:** Wilson score 95% CI (binomial). Specificity has
  only n=12 HC, so its interval is very wide.
- **No connectivity / graph external validation** (8 s records do not support it).

## Hard gates (the run does not publish unless all pass)

Canonical archive SHA-256; condition is `Eyes_closed`; cohort audit has no
`fail`; external exactly 80 AD + 12 HC, 92 unique IDs, all common-19, 1024
finite samples/channel; discovery exactly 88 unique (36 AD / 23 FTD / 29 HC) all
extracted; training strictly 36 AD + 29 HC; no duplicate IDs, no unknown labels,
no missing columns, no non-finite features; predictions exactly 92 rows with IDs
equal to the audited set and labels ⊆ {AD, HC}. Unknown external labels are
never silently mapped to HC. Outputs are written to a staging directory and
published atomically only on success.

## Results policy

v2 is methodological hardening, **not score-chasing**. Whatever the external
metrics, they are saved and reported in full. No model/feature/threshold is
chosen or adjusted using OSF labels, features, or metrics. If the real run
exposes a pure engineering bug, it is fixed and logged in the review request's
"Protocol deviations / rerun count" section.

## Reproducibility (no Git history available)

`validation_provenance.json` records the exact command, UTC time, interpreter,
Python/platform and key package versions, the seed, SHA-256 of key code and
input files, `environment.txt` (pip freeze), and `artifact_manifest.json`
(relative path, bytes, SHA-256 for every published artifact; the manifest
excludes itself).
