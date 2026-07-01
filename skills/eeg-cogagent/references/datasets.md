# Dataset routing

## ds004504: primary validation case

Use OpenNeuro ds004504 for the main eyes-closed resting-state analysis. It contains 88 participants: 36 AD, 23 FTD, and 29 healthy controls; 19 EEG channels; 500-Hz sampling; and labels in `participants.tsv` with `A=AD`, `F=FTD`, and `C=HC`.

Default config: `configs/ds004504_minimal.yaml`.

## ds006036: paired condition shift

Use OpenNeuro ds006036 only for paired cross-condition robustness. It contains photomark/open-eyes recordings from the same cohort as ds004504. Maintain identical spectral feature definitions and participant-disjoint outer folds.

Default config: `configs/ds006036_cross_condition.yaml`.

Do not describe ds006036 as independent external validation. Recruitment, site, device, labels, and participant identities are shared.

## New BIDS datasets

Before adapting a config, inspect:

- `dataset_description.json` and license;
- `participants.tsv` names, encodings, and missingness;
- task and suffix names;
- channel names, channel types, units, reference, and sampling frequency;
- line frequency and recording duration;
- event structure and whether the recording is rest, task, or stimulation;
- whether participants overlap any training dataset.

Create a new YAML config instead of modifying a completed analysis config in place.
