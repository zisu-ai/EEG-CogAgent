# Workflow and artifact map

## Core ds004504 run

```powershell
python -m pip install -e .
eeg-cogagent plan configs\ds004504_minimal.yaml
eeg-cogagent run configs\ds004504_minimal.yaml --subjects-limit 6 --output-dir results\smoke\ds004504_minimal
eeg-cogagent run configs\ds004504_minimal.yaml
```

The core run writes `features.csv`, `feature_statistics.csv`, `model_metrics.csv`, out-of-fold predictions, topomaps, baseline tables, and `auto_report.md` under `results/ds004504_minimal`.

## Analysis extensions

```powershell
python scripts\qc_ds004504.py --config configs\ds004504_minimal.yaml
python scripts\adjusted_analysis.py --config configs\ds004504_minimal.yaml
python scripts\pairwise_analysis.py --config configs\ds004504_minimal.yaml
python scripts\residualized_analysis.py --config configs\ds004504_minimal.yaml
python scripts\connectivity_analysis.py --config configs\ds004504_minimal.yaml --workers 6
python scripts\generate_framework_figure.py --config configs\ds004504_minimal.yaml
```

Run extensions independently so a failed optional analysis does not overwrite the core feature matrix.

## Paired cross-condition analysis

```powershell
eeg-cogagent run configs\ds006036_cross_condition.yaml --subjects-limit 9 --output-dir results\smoke\ds006036_cross_condition
eeg-cogagent run configs\ds006036_cross_condition.yaml
python scripts\cross_condition_validation.py
```

The cross-condition script trains only on ds004504 training participants and evaluates ds006036 recordings from held-out participants. Its outputs live under `results/ds006036_cross_condition/cross_condition`.

## Deterministic audit

```powershell
eeg-cogagent audit configs\ds004504_minimal.yaml
eeg-cogagent audit configs\ds004504_minimal.yaml --strict
```

The audit creates:

- `agent_audit.json`: machine-readable checks, versions, and run summary.
- `agent_audit.md`: human-readable audit report.
- `artifact_manifest.json`: relative paths, byte sizes, and SHA-256 hashes.

Use `--strict` in automation when failed checks should return a nonzero exit code.

## Expected manuscript artifacts

- Figure 1: `figures/figure1_workflow.pdf`
- Figure 2: `figures/topomap_*.png`
- Figure 3: `figures/figure3_connectivity.pdf`
- Figure 4: `figures/figure4_roc.pdf`
- Figure 5: `results/ds006036_cross_condition/figures/figure5_cross_condition.pdf`
- Draft Methods/Results: `docs/MANUSCRIPT_METHODS_RESULTS.md`
