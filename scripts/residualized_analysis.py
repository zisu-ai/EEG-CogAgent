from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eeg_cogagent.config import load_config, project_path
from eeg_cogagent.ml import evaluate_residualized_models
from eeg_cogagent.viz import plot_multiclass_roc


def write_report(
    output_dir: Path,
    residualized: pd.DataFrame,
    unadjusted: pd.DataFrame,
) -> None:
    comparison = residualized.merge(
        unadjusted[["model", "accuracy", "balanced_accuracy", "auc_ovr"]],
        on="model",
        suffixes=("_residualized", "_unadjusted"),
    )
    for metric in ["accuracy", "balanced_accuracy", "auc_ovr"]:
        comparison[f"delta_{metric}"] = (
            comparison[f"{metric}_residualized"] - comparison[f"{metric}_unadjusted"]
        )
    comparison.to_csv(output_dir / "residualized_model_comparison.csv", index=False)

    best = residualized.sort_values("balanced_accuracy", ascending=False).iloc[0]
    lines = [
        "# Leakage-Safe Residualized Classification",
        "",
        "## Method",
        "",
        "Age, binary gender, and log-transformed retained epoch count were regressed from every EEG feature. "
        "The nuisance regression was fitted independently on each training fold and applied to its held-out fold. "
        "Residualization, imputation, scaling, and hyperparameter selection were all contained within the nested "
        "cross-validation pipeline. Nuisance variables were not supplied to the final classifier.",
        "",
        "## Performance",
        "",
        residualized.to_markdown(index=False, floatfmt=".3f"),
        "",
        "Comparison with the original unadjusted EEG classifiers:",
        "",
        comparison.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Interpretation",
        "",
        f"The best residualized model was `{best['model']}` with balanced accuracy "
        f"{best['balanced_accuracy']:.3f} and multiclass OVR AUC {best['auc_ovr']:.3f}. "
        "This is a confound-sensitivity analysis, not proof that all biological or technical confounding has been removed. "
        "External validation remains necessary before any diagnostic-performance claim.",
    ]
    (output_dir / "residualized_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leakage-safe nuisance-residualized nested-CV analysis."
    )
    parser.add_argument("--config", default="configs/ds004504_minimal.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result_dir = project_path(cfg, cfg["paths"]["output_dir"])
    output_dir = result_dir / "qc"
    output_dir.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(result_dir / "features.csv")

    metrics, predictions = evaluate_residualized_models(features, cfg)
    metrics.to_csv(output_dir / "residualized_model_metrics.csv", index=False)
    predictions.to_csv(output_dir / "residualized_model_predictions.csv", index=False)
    plot_multiclass_roc(predictions, result_dir / "figures")
    write_report(
        output_dir,
        metrics,
        pd.read_csv(result_dir / "model_metrics.csv"),
    )
    print(f"Residualized outputs written to {output_dir}")


if __name__ == "__main__":
    main()
