from __future__ import annotations

import argparse
import warnings

import pandas as pd

from eeg_cogagent.config import load_config, project_path
from eeg_cogagent.ml import evaluate_models


COMPARISONS = [("AD", "HC"), ("FTD", "HC"), ("AD", "FTD")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Nested-CV pairwise ds004504 classification.")
    parser.add_argument("--config", default="configs/ds004504_minimal.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result_dir = project_path(cfg, cfg["paths"]["output_dir"])
    output_dir = result_dir / "qc"
    output_dir.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(result_dir / "features.csv")

    metric_frames = []
    prediction_frames = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        for group_a, group_b in COMPARISONS:
            comparison = f"{group_a}_vs_{group_b}"
            subset = features[features["label"].isin([group_a, group_b])].reset_index(drop=True)
            metrics, predictions = evaluate_models(subset, cfg)
            metrics.insert(0, "comparison", comparison)
            predictions.insert(0, "comparison", comparison)
            metric_frames.append(metrics)
            prediction_frames.append(predictions)

    all_metrics = pd.concat(metric_frames, ignore_index=True)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    all_metrics.to_csv(output_dir / "pairwise_model_metrics.csv", index=False)
    all_predictions.to_csv(output_dir / "pairwise_model_predictions.csv", index=False)

    best = (
        all_metrics.sort_values(["comparison", "balanced_accuracy"], ascending=[True, False])
        .groupby("comparison", as_index=False)
        .first()
    )
    lines = [
        "# Pairwise Nested-CV Classification",
        "",
        "The existing 137 EEG features were evaluated without feature changes. Hyperparameters were selected within the inner folds of five-fold stratified nested cross-validation.",
        "",
        "## All Models",
        "",
        all_metrics.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Best Model Per Comparison",
        "",
        best.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Interpretation",
        "",
        "Pairwise performance should be used to distinguish disease-versus-control detection from AD-versus-FTD differential classification. "
        "These are internal cross-validation estimates and are not substitutes for external validation.",
    ]
    (output_dir / "pairwise_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Pairwise outputs written to {output_dir}")


if __name__ == "__main__":
    main()
