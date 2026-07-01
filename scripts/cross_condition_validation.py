from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from eeg_cogagent.config import load_config, project_path
from eeg_cogagent.ml import _model_grid


META_COLUMNS = {"participant_id", "label", "Group", "Gender", "Age", "MMSE", "n_epochs"}
CONDITION_COLORS = {"eyesclosed": "#0072B2", "photomark": "#D55E00"}


def _metrics_with_ci(
    truth: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    random_state: int,
    n_bootstrap: int = 2000,
) -> dict[str, float]:
    accuracy = accuracy_score(truth, predicted)
    balanced = balanced_accuracy_score(truth, predicted)
    auc_value = roc_auc_score(truth, probabilities, multi_class="ovr", labels=classes)
    rng = np.random.default_rng(random_state)
    by_class = [np.flatnonzero(truth == class_code) for class_code in classes]
    accuracy_samples = []
    balanced_samples = []
    auc_samples = []
    for _ in range(n_bootstrap):
        indices = np.concatenate([
            rng.choice(class_indices, size=len(class_indices), replace=True)
            for class_indices in by_class
        ])
        accuracy_samples.append(accuracy_score(truth[indices], predicted[indices]))
        balanced_samples.append(balanced_accuracy_score(truth[indices], predicted[indices]))
        try:
            auc_samples.append(
                roc_auc_score(
                    truth[indices], probabilities[indices], multi_class="ovr", labels=classes
                )
            )
        except ValueError:
            continue
    return {
        "accuracy": accuracy,
        "accuracy_ci_low": float(np.percentile(accuracy_samples, 2.5)),
        "accuracy_ci_high": float(np.percentile(accuracy_samples, 97.5)),
        "balanced_accuracy": balanced,
        "balanced_accuracy_ci_low": float(np.percentile(balanced_samples, 2.5)),
        "balanced_accuracy_ci_high": float(np.percentile(balanced_samples, 97.5)),
        "auc_ovr": auc_value,
        "auc_ci_low": float(np.percentile(auc_samples, 2.5)),
        "auc_ci_high": float(np.percentile(auc_samples, 97.5)),
    }


def evaluate_cross_condition(
    source: pd.DataFrame,
    target: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired_ids = sorted(set(source["participant_id"]) & set(target["participant_id"]))
    source = source.set_index("participant_id").loc[paired_ids].reset_index().copy()
    target = target.set_index("participant_id").loc[paired_ids].reset_index().copy()
    if not source["label"].equals(target["label"]):
        raise ValueError("Diagnostic labels do not match across paired datasets.")

    source_numeric = set(source.select_dtypes(include=[np.number]).columns) - META_COLUMNS
    target_numeric = set(target.select_dtypes(include=[np.number]).columns) - META_COLUMNS
    feature_columns = sorted(source_numeric & target_numeric)
    if not feature_columns:
        raise ValueError("No common numeric EEG features were found.")

    encoder = LabelEncoder()
    labels = source["label"].astype(str)
    y = encoder.fit_transform(labels)
    class_codes = np.arange(len(encoder.classes_))
    ml_cfg = cfg["ml"]
    folds = min(int(ml_cfg.get("cv_folds", 5)), int(labels.value_counts().min()))
    random_state = int(ml_cfg.get("random_state", 42))
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    prediction_rows = []

    for model_name in ml_cfg.get("models", ["logistic_regression"]):
        for fold, (train_indices, test_indices) in enumerate(
            outer.split(source[feature_columns], y), start=1
        ):
            estimator, grid = _model_grid(model_name, random_state + fold)
            inner = StratifiedKFold(
                n_splits=min(folds - 1, int(np.bincount(y[train_indices]).min())),
                shuffle=True,
                random_state=random_state + fold,
            )
            fitted = GridSearchCV(
                estimator,
                grid,
                cv=inner,
                scoring=ml_cfg.get("scoring", "balanced_accuracy"),
                n_jobs=-1,
            )
            fitted.fit(source.iloc[train_indices][feature_columns], y[train_indices])

            for condition, frame in [("eyesclosed", source), ("photomark", target)]:
                predicted = fitted.predict(frame.iloc[test_indices][feature_columns])
                probabilities = fitted.predict_proba(frame.iloc[test_indices][feature_columns])
                for row_index, pred_code, probability in zip(test_indices, predicted, probabilities):
                    row = {
                        "model": model_name,
                        "evaluation_condition": condition,
                        "fold": fold,
                        "participant_id": source.iloc[row_index]["participant_id"],
                        "true_label": labels.iloc[row_index],
                        "pred_label": encoder.inverse_transform([pred_code])[0],
                    }
                    row.update({
                        f"prob_{class_name}": float(probability[class_index])
                        for class_index, class_name in enumerate(encoder.classes_)
                    })
                    prediction_rows.append(row)

    predictions = pd.DataFrame(prediction_rows)
    metric_rows = []
    for (model_name, condition), group in predictions.groupby(
        ["model", "evaluation_condition"], sort=False
    ):
        truth = encoder.transform(group["true_label"])
        predicted = encoder.transform(group["pred_label"])
        probabilities = group[[f"prob_{name}" for name in encoder.classes_]].to_numpy()
        row = {
            "model": model_name,
            "evaluation_condition": condition,
            "n_participants": len(group),
            "n_features": len(feature_columns),
            "cv_folds": folds,
        }
        row.update(
            _metrics_with_ci(
                truth,
                predicted,
                probabilities,
                class_codes,
                random_state=random_state + len(metric_rows),
            )
        )
        metric_rows.append(row)
    return pd.DataFrame(metric_rows), predictions


def plot_results(metrics: pd.DataFrame, output_dir: Path) -> list[Path]:
    models = list(metrics["model"].drop_duplicates())
    conditions = ["eyesclosed", "photomark"]
    display_names = {
        "logistic_regression": "Logistic regression",
        "svm_rbf": "SVM (RBF)",
        "random_forest": "Random forest",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
    x_positions = np.arange(len(models))
    width = 0.34
    for axis, metric, ylabel, panel in [
        (axes[0], "balanced_accuracy", "Balanced accuracy", "A"),
        (axes[1], "auc_ovr", "Multiclass OVR AUC", "B"),
    ]:
        for offset_index, condition in enumerate(conditions):
            subset = metrics.set_index(["model", "evaluation_condition"]).loc[
                [(model, condition) for model in models]
            ]
            values = subset[metric].to_numpy(dtype=float)
            low_name = f"{metric}_ci_low" if metric != "auc_ovr" else "auc_ci_low"
            high_name = f"{metric}_ci_high" if metric != "auc_ovr" else "auc_ci_high"
            errors = np.vstack([
                values - subset[low_name].to_numpy(dtype=float),
                subset[high_name].to_numpy(dtype=float) - values,
            ])
            positions = x_positions + (offset_index - 0.5) * width
            axis.bar(
                positions, values, width=width, color=CONDITION_COLORS[condition],
                alpha=0.86, label="Eyes-closed test" if condition == "eyesclosed" else "Photomark test",
                yerr=errors, capsize=2.5, error_kw={"linewidth": 0.8},
            )
        axis.axhline(1 / 3 if metric == "balanced_accuracy" else 0.5,
                     color="#6B7280", linestyle="--", linewidth=0.8)
        axis.set_ylim(0, 1)
        axis.set_ylabel(ylabel, fontsize=8)
        axis.set_xticks(x_positions, [display_names.get(model, model) for model in models],
                        rotation=18, ha="right")
        axis.tick_params(labelsize=7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(-0.13, 1.04, panel, transform=axis.transAxes,
                  fontsize=10, fontweight="bold", va="top")
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("Subject-disjoint cross-condition validation", fontsize=9.5)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "figure5_cross_condition.png"
    pdf = output_dir / "figure5_cross_condition.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def write_report(output_dir: Path, metrics: pd.DataFrame, figures: list[Path]) -> Path:
    pivot = metrics.pivot(index="model", columns="evaluation_condition", values="balanced_accuracy")
    deltas = (pivot["photomark"] - pivot["eyesclosed"]).rename("delta_balanced_accuracy")
    comparison = metrics.merge(deltas, left_on="model", right_index=True)
    target = metrics[metrics["evaluation_condition"] == "photomark"]
    best = target.sort_values("balanced_accuracy", ascending=False).iloc[0]
    lines = [
        "# Subject-Disjoint Cross-Condition Validation",
        "",
        "## Design",
        "",
        "The same participants contributed eyes-closed recordings in ds004504 and open-eyes "
        "photomark recordings in ds006036. Five stratified outer folds were defined at the participant "
        "level. For each fold, models and hyperparameters were fitted using only ds004504 recordings "
        "from the training participants. Performance was evaluated both on held-out ds004504 recordings "
        "and on ds006036 recordings from those same held-out participants. Thus, no participant whose "
        "ds006036 recording was tested contributed either condition to model fitting.",
        "",
        "## Results",
        "",
        comparison.to_markdown(index=False, floatfmt=".3f"),
        "",
        f"The best photomark-condition balanced accuracy was {best['balanced_accuracy']:.3f} "
        f"(95% bootstrap CI {best['balanced_accuracy_ci_low']:.3f}–"
        f"{best['balanced_accuracy_ci_high']:.3f}) with `{best['model']}`. Its multiclass OVR AUC "
        f"was {best['auc_ovr']:.3f} (95% CI {best['auc_ci_low']:.3f}–{best['auc_ci_high']:.3f}).",
        "",
        "## Interpretation",
        "",
        "This analysis tests robustness to a substantial acquisition-condition shift, not external-cohort "
        "generalization: ds004504 and ds006036 contain the same clinical cohort. Subject-disjoint folds "
        "prevent direct identity leakage into model fitting, but shared recruitment, device, site, and "
        "participant characteristics remain. An independent cohort is still required for external validation.",
        "",
        "Generated figures: " + ", ".join(path.name for path in figures),
    ]
    path = output_dir / "cross_condition_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Subject-disjoint ds004504-to-ds006036 validation.")
    parser.add_argument("--source-config", default="configs/ds004504_minimal.yaml")
    parser.add_argument("--target-config", default="configs/ds006036_cross_condition.yaml")
    args = parser.parse_args()

    source_cfg = load_config(args.source_config)
    target_cfg = load_config(args.target_config)
    source_dir = project_path(source_cfg, source_cfg["paths"]["output_dir"])
    target_dir = project_path(target_cfg, target_cfg["paths"]["output_dir"])
    output_dir = target_dir / "cross_condition"
    figure_dir = target_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(source_dir / "features.csv")
    target = pd.read_csv(target_dir / "features.csv")
    metrics, predictions = evaluate_cross_condition(source, target, source_cfg)
    metrics.to_csv(output_dir / "cross_condition_metrics.csv", index=False)
    predictions.to_csv(output_dir / "cross_condition_predictions.csv", index=False)
    figures = plot_results(metrics, figure_dir)
    report = write_report(output_dir, metrics, figures)
    print(f"Cross-condition outputs written to {output_dir}")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
