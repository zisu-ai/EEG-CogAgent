# EEG-CogAgent

Lightweight project scaffold for an LLM-assisted resting-state EEG biomarker workflow in dementia.

Core idea: do not claim that an LLM diagnoses dementia. The agent standardizes and documents the analysis workflow: BIDS loading, MNE preprocessing, interpretable EEG feature extraction, statistics, machine-learning validation, figures, and report drafting.

Priority dataset: OpenNeuro `ds004504`, a CC0 BIDS EEG dataset with Alzheimer's disease, frontotemporal dementia, and healthy controls.

## Quick Start

```powershell
cd EEG-CogAgent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Download the dataset with OpenNeuro or DataLad into `data/ds004504`, then run:

```powershell
eeg-cogagent plan configs\ds004504_minimal.yaml
eeg-cogagent run configs\ds004504_minimal.yaml --subjects-limit 6 --output-dir results\smoke\ds004504_minimal
eeg-cogagent run configs\ds004504_minimal.yaml
```

The first run with `--subjects-limit 6` is a smoke test. The full run should write features, statistics, model metrics, figures, and a Markdown report under `results/ds004504_minimal`.

Run the post hoc quality-control and confound-sensitivity analyses with:

```powershell
python scripts\qc_ds004504.py --config configs\ds004504_minimal.yaml
python scripts\adjusted_analysis.py --config configs\ds004504_minimal.yaml
python scripts\pairwise_analysis.py --config configs\ds004504_minimal.yaml
python scripts\residualized_analysis.py --config configs\ds004504_minimal.yaml
python scripts\connectivity_analysis.py --config configs\ds004504_minimal.yaml --workers 6
python scripts\generate_framework_figure.py --config configs\ds004504_minimal.yaml
eeg-cogagent run configs\ds006036_cross_condition.yaml
python scripts\cross_condition_validation.py
eeg-cogagent audit configs\ds004504_minimal.yaml
```

The audit command verifies artifact completeness, cohort and prediction consistency, metric ranges, optional extensions, software versions, and SHA-256 provenance. It writes `agent_audit.json`, `agent_audit.md`, and `artifact_manifest.json` inside the selected result directory.

`ds006036` contains photomark recordings from the same participants as `ds004504`. It is used for subject-disjoint cross-condition testing, not described as an independent external cohort.

## Paper Angle

Working title:

`EEG-CogAgent: An Auditable Language-Model Agent for Reproducible Dementia EEG Biomarker Analysis`

The safest novelty claim is:

> EEG-CogAgent packages a reproducible, agent-readable resting-state EEG biomarker workflow for dementia differential analysis, combining interpretable EEG features, standardized validation, and automated Methods/Results drafting on public BIDS data.

Avoid saying "no one has done dementia EEG classification." The dataset already has ML and deep-learning studies. The contribution is the reusable skill/agent layer plus transparent reporting.

## Manuscript and Literature Archive

- Complete manuscript: [`docs/FULL_MANUSCRIPT.md`](docs/FULL_MANUSCRIPT.md)
- Nature-skills review and terminology audit: [`docs/NATURE_STYLE_AUDIT.md`](docs/NATURE_STYLE_AUDIT.md)
- Search record: [`literature/search_strategy.md`](literature/search_strategy.md)
- Evidence library: [`literature/library.csv`](literature/library.csv)
- BibTeX library: [`literature/references.bib`](literature/references.bib)
- Structural synthesis: [`literature/structure_synthesis.md`](literature/structure_synthesis.md)
- Open full texts: [`literature/fulltext/`](literature/fulltext/)
