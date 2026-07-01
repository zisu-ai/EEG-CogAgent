# Reporting and claim rules

## Evidence hierarchy

Use numerical evidence in this order:

1. Audited CSV/JSON artifacts.
2. Generated QC and analysis reports.
3. Manuscript drafts generated from those artifacts.
4. Narrative interpretation.

If prose conflicts with a machine-readable artifact, correct the prose.

## Required reporting

Report participant counts and failures, retained epoch distribution, preprocessing parameters, feature definitions, FDR family, participant-level split strategy, inner and outer cross-validation, metrics with uncertainty, nuisance analyses, and validation scope.

Describe functional connectivity as sensor-level unless source reconstruction was performed. State that PLI and wPLI reduce sensitivity to zero-lag coupling but do not remove volume conduction, reference dependence, or other confounding.

## Approved framing

Primary contribution: a lightweight, configuration-driven, LLM-assisted skill that orchestrates deterministic BIDS EEG preprocessing, interpretable biomarker extraction, statistics, validation, artifact auditing, figure generation, and Methods/Results drafting.

Validation case: public dementia EEG data demonstrate disease-versus-control slowing and network alterations, with moderate multiclass prediction and weaker AD-versus-FTD separation.

## Prohibited overclaims

Do not state that the LLM analyzed raw EEG, diagnosed dementia, eliminated all confounding, established causality, achieved clinical-grade performance, or generalized externally unless a genuinely independent cohort was tested.
