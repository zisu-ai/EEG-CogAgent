from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal, spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from statsmodels.stats.multitest import multipletests

from eeg_cogagent.config import load_config, project_path
from eeg_cogagent.ml import evaluate_models
from eeg_cogagent.stats import run_feature_statistics


LABELS = ["AD", "FTD", "HC"]


def _bootstrap_metrics(predictions: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for model, frame in predictions.groupby("model", sort=False):
        true = frame["true_label"].to_numpy()
        pred = frame["pred_label"].to_numpy()
        class_indices = [np.flatnonzero(true == label) for label in LABELS]
        boot_acc = []
        boot_bal = []
        for _ in range(n_boot):
            sample = np.concatenate(
                [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
            )
            boot_acc.append(accuracy_score(true[sample], pred[sample]))
            boot_bal.append(balanced_accuracy_score(true[sample], pred[sample]))
        rows.append({
            "model": model,
            "accuracy": accuracy_score(true, pred),
            "accuracy_ci_low": np.quantile(boot_acc, 0.025),
            "accuracy_ci_high": np.quantile(boot_acc, 0.975),
            "balanced_accuracy": balanced_accuracy_score(true, pred),
            "balanced_accuracy_ci_low": np.quantile(boot_bal, 0.025),
            "balanced_accuracy_ci_high": np.quantile(boot_bal, 0.975),
        })
    return pd.DataFrame(rows)


def _prediction_diagnostics(predictions: pd.DataFrame, output_dir: Path) -> None:
    confusion_rows = []
    class_rows = []
    models = list(predictions["model"].drop_duplicates())
    fig, axes = plt.subplots(1, len(models), figsize=(4.5 * len(models), 4.2), constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, model in zip(axes, models):
        frame = predictions[predictions["model"] == model]
        matrix = confusion_matrix(frame["true_label"], frame["pred_label"], labels=LABELS)
        for true_idx, true_label in enumerate(LABELS):
            for pred_idx, pred_label in enumerate(LABELS):
                confusion_rows.append({
                    "model": model,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "n": int(matrix[true_idx, pred_idx]),
                })

        report = classification_report(
            frame["true_label"],
            frame["pred_label"],
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        )
        for label in LABELS:
            class_rows.append({
                "model": model,
                "class": label,
                "precision": report[label]["precision"],
                "recall_sensitivity": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": int(report[label]["support"]),
            })

        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                text_color = "white" if matrix[row, col] > matrix.max() / 2 else "black"
                ax.text(
                    col,
                    row,
                    str(matrix[row, col]),
                    ha="center",
                    va="center",
                    color=text_color,
                )
        ax.set_xticks(range(len(LABELS)), LABELS)
        ax.set_yticks(range(len(LABELS)), LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        display_names = {
            "logistic_regression": "Logistic Regression",
            "svm_rbf": "SVM (RBF)",
            "random_forest": "Random Forest",
        }
        ax.set_title(display_names.get(model, model.replace("_", " ").title()))

    fig.colorbar(image, ax=axes.tolist(), shrink=0.75, label="Participants")
    fig.savefig(output_dir / "confusion_matrices.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(confusion_rows).to_csv(output_dir / "confusion_matrices.csv", index=False)
    pd.DataFrame(class_rows).to_csv(output_dir / "per_class_metrics.csv", index=False)


def _confound_tests(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = features.groupby("label")
    for variable in ["Age", "n_epochs"]:
        samples = [group[variable].dropna().astype(float).to_numpy() for _, group in grouped]
        statistic, p_value = kruskal(*samples)
        n = sum(len(sample) for sample in samples)
        k = len(samples)
        epsilon_squared = max(0.0, (statistic - k + 1) / (n - k))
        rows.append({
            "variable": variable,
            "test": "Kruskal-Wallis",
            "statistic": statistic,
            "p_value": p_value,
            "effect_size": epsilon_squared,
            "effect_size_name": "epsilon_squared",
            "minimum_expected_count": np.nan,
        })

    contingency = pd.crosstab(features["label"], features["Gender"])
    statistic, p_value, _, expected = chi2_contingency(contingency)
    denominator = len(features) * min(contingency.shape[0] - 1, contingency.shape[1] - 1)
    cramer_v = np.sqrt(statistic / denominator) if denominator else np.nan
    rows.append({
        "variable": "Gender",
        "test": "Chi-square",
        "statistic": statistic,
        "p_value": p_value,
        "effect_size": cramer_v,
        "effect_size_name": "Cramers_V",
        "minimum_expected_count": expected.min(),
    })
    return pd.DataFrame(rows)


def _covariate_correlations(features: pd.DataFrame, eeg_columns: list[str]) -> pd.DataFrame:
    rows = []
    for covariate in ["Age", "n_epochs"]:
        for feature in eeg_columns:
            valid = features[[covariate, feature]].dropna()
            rho, p_value = spearmanr(valid[covariate], valid[feature])
            rows.append({
                "covariate": covariate,
                "feature": feature,
                "rho": rho,
                "p_value": p_value,
            })
    result = pd.DataFrame(rows)
    result["q_value"] = np.nan
    for covariate, index in result.groupby("covariate").groups.items():
        _, q_values, _, _ = multipletests(result.loc[index, "p_value"], method="fdr_bh")
        result.loc[index, "q_value"] = q_values
    return result.sort_values(["covariate", "q_value", "p_value"]).reset_index(drop=True)


def _write_report(
    output_dir: Path,
    features: pd.DataFrame,
    full_metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    per_class: pd.DataFrame,
    confounds: pd.DataFrame,
    correlations: pd.DataFrame,
    sensitivity_metrics: pd.DataFrame,
    sensitivity_stats: pd.DataFrame,
    excluded: pd.DataFrame,
    epoch_threshold: int,
) -> None:
    comparison = full_metrics.merge(
        sensitivity_metrics,
        on="model",
        suffixes=("_full", "_sensitivity"),
    )
    for metric in ["accuracy", "balanced_accuracy", "auc_ovr"]:
        comparison[f"delta_{metric}"] = (
            comparison[f"{metric}_sensitivity"] - comparison[f"{metric}_full"]
        )
    comparison.to_csv(output_dir / "sensitivity_comparison.csv", index=False)

    best = full_metrics.sort_values("balanced_accuracy", ascending=False).iloc[0]
    best_classes = per_class[per_class["model"] == best["model"]]
    significant_confounds = confounds[confounds["p_value"] < 0.05]
    age_correlations = correlations[
        (correlations["covariate"] == "Age") & (correlations["q_value"] < 0.05)
    ]
    epoch_correlations = correlations[
        (correlations["covariate"] == "n_epochs") & (correlations["q_value"] < 0.05)
    ]
    full_stats = pd.read_csv(output_dir.parent / "feature_statistics.csv")
    top_overlap = len(
        set(full_stats.head(20)["feature"]) & set(sensitivity_stats.head(20)["feature"])
    )

    lines = [
        "# ds004504 Full-Cohort Quality Control",
        "",
        "## Scope",
        "",
        "This report audits the existing 88-participant analysis without adding EEG features or changing preprocessing.",
        "",
        "## Cohort and Processing",
        "",
        f"- Processed participants: {len(features)} (AD 36, FTD 23, HC 29).",
        f"- Retained epochs: median {features['n_epochs'].median():.0f}, range {features['n_epochs'].min():.0f}-{features['n_epochs'].max():.0f}.",
        f"- Participants below {epoch_threshold} epochs: {len(excluded)}.",
        "",
        "## Classification Diagnostics",
        "",
        f"The best balanced accuracy was produced by `{best['model']}`: {best['balanced_accuracy']:.3f} "
        f"(accuracy {best['accuracy']:.3f}, multiclass OVR AUC {best['auc_ovr']:.3f}).",
        "",
        diagnostics.to_markdown(index=False, floatfmt=".3f"),
        "",
        "Class-specific performance for the best model:",
        "",
        best_classes.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Confounding Checks",
        "",
        confounds.to_markdown(index=False, floatfmt=".4g"),
        "",
        f"Nominally significant group differences among tested covariates: {len(significant_confounds)}. "
        f"EEG features associated after FDR correction with age: {len(age_correlations)}; "
        f"with retained epoch count: {len(epoch_correlations)}.",
        "",
        "These are screening diagnostics, not proof that confounding is absent. Age- or epoch-sensitive features should be checked in adjusted models before confirmatory claims.",
        "",
        "## Low-Epoch Sensitivity Analysis",
        "",
        f"A post hoc sensitivity analysis excluded participants with fewer than {epoch_threshold} retained epochs: "
        + (", ".join(excluded["participant_id"]) if len(excluded) else "none")
        + ".",
        "",
        comparison[[
            "model",
            "balanced_accuracy_full",
            "balanced_accuracy_sensitivity",
            "delta_balanced_accuracy",
            "auc_ovr_full",
            "auc_ovr_sensitivity",
            "delta_auc_ovr",
        ]].to_markdown(index=False, floatfmt=".3f"),
        "",
        f"Top-20 FDR feature overlap after exclusion: {top_overlap}/20. "
        f"Significant features in the sensitivity set: {(sensitivity_stats['q_value'] < 0.05).sum()}/{len(sensitivity_stats)}.",
        "",
        "## Interpretation",
        "",
        "The full-cohort run is technically complete. The classification signal is moderate rather than diagnostic-grade. "
        "The strongest biomarker pattern is EEG slowing (especially theta/alpha ratios), and robustness should be framed as exploratory until covariate-adjusted and external validation analyses are completed.",
    ]
    (output_dir / "qc_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the full ds004504 EEG-CogAgent run.")
    parser.add_argument("--config", default="configs/ds004504_minimal.yaml")
    parser.add_argument("--epoch-threshold", type=int, default=30)
    args = parser.parse_args()

    cfg = load_config(args.config)
    result_dir = project_path(cfg, cfg["paths"]["output_dir"])
    output_dir = result_dir / "qc"
    output_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(result_dir / "features.csv")
    predictions = pd.read_csv(result_dir / "model_predictions.csv")
    full_metrics = pd.read_csv(result_dir / "model_metrics.csv")
    excluded_columns = set(cfg["ml"].get("exclude_columns", [])) | {
        "participant_id", "label", "Group", "Gender", "Age", "MMSE", "n_epochs"
    }
    eeg_columns = [
        column for column in features.select_dtypes(include=[np.number]).columns
        if column not in excluded_columns
    ]

    _prediction_diagnostics(predictions, output_dir)
    diagnostics = _bootstrap_metrics(predictions)
    diagnostics.to_csv(output_dir / "model_diagnostics.csv", index=False)
    per_class = pd.read_csv(output_dir / "per_class_metrics.csv")

    confounds = _confound_tests(features)
    confounds.to_csv(output_dir / "confound_tests.csv", index=False)
    correlations = _covariate_correlations(features, eeg_columns)
    correlations.to_csv(output_dir / "covariate_correlations.csv", index=False)

    excluded = features[features["n_epochs"] < args.epoch_threshold][
        ["participant_id", "label", "n_epochs"]
    ].copy()
    excluded.to_csv(output_dir / "low_epoch_participants.csv", index=False)
    sensitivity = features[features["n_epochs"] >= args.epoch_threshold].reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        sensitivity_metrics, sensitivity_predictions = evaluate_models(sensitivity, cfg)
    sensitivity_metrics.to_csv(output_dir / "sensitivity_model_metrics.csv", index=False)
    sensitivity_predictions.to_csv(output_dir / "sensitivity_model_predictions.csv", index=False)
    sensitivity_stats = run_feature_statistics(
        sensitivity,
        label_column=cfg["statistics"].get("label_column", "label"),
        exclude_columns=cfg["statistics"].get("exclude_columns", []),
    )
    sensitivity_stats.to_csv(output_dir / "sensitivity_feature_statistics.csv", index=False)

    _write_report(
        output_dir,
        features,
        full_metrics,
        diagnostics,
        per_class,
        confounds,
        correlations,
        sensitivity_metrics,
        sensitivity_stats,
        excluded,
        args.epoch_threshold,
    )
    print(f"QC outputs written to {output_dir}")


if __name__ == "__main__":
    main()
