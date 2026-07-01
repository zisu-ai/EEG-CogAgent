from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.stats.multitest import multipletests

from eeg_cogagent.config import load_config, project_path


def _design_matrix(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    complete = features.dropna(subset=["Age", "Gender", "n_epochs", "label"]).copy()
    design = pd.DataFrame(index=complete.index)
    design["const"] = 1.0
    design["group_FTD"] = (complete["label"] == "FTD").astype(float)
    design["group_HC"] = (complete["label"] == "HC").astype(float)
    design["age_z"] = (complete["Age"] - complete["Age"].mean()) / complete["Age"].std(ddof=0)
    design["gender_M"] = (complete["Gender"].astype(str).str.upper() == "M").astype(float)
    log_epochs = np.log1p(complete["n_epochs"].astype(float))
    design["log_epochs_z"] = (log_epochs - log_epochs.mean()) / log_epochs.std(ddof=0)
    return complete, design


def adjusted_feature_tests(features: pd.DataFrame, eeg_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    complete, design = _design_matrix(features)
    omnibus_rows = []
    contrast_rows = []
    group_indices = [design.columns.get_loc("group_FTD"), design.columns.get_loc("group_HC")]
    restriction = np.zeros((2, design.shape[1]))
    restriction[0, group_indices[0]] = 1.0
    restriction[1, group_indices[1]] = 1.0
    contrasts = {
        "FTD_vs_AD": np.array([0, 1, 0, 0, 0, 0], dtype=float),
        "HC_vs_AD": np.array([0, 0, 1, 0, 0, 0], dtype=float),
        "FTD_vs_HC": np.array([0, 1, -1, 0, 0, 0], dtype=float),
    }

    reduced_design = design.drop(columns=["group_FTD", "group_HC"])
    for feature in eeg_columns:
        y = complete[feature].astype(float)
        y_sd = y.std(ddof=0)
        if not np.isfinite(y_sd) or y_sd == 0:
            continue
        y_z = (y - y.mean()) / y_sd
        fitted = sm.OLS(y_z, design).fit(cov_type="HC3")
        reduced = sm.OLS(y_z, reduced_design).fit()
        group_test = fitted.wald_test(restriction, use_f=False, scalar=True)
        partial_r2 = max(0.0, (reduced.ssr - fitted.ssr) / reduced.ssr)
        omnibus_rows.append({
            "feature": feature,
            "wald_chi2": float(group_test.statistic),
            "df": 2,
            "p_value": float(group_test.pvalue),
            "partial_r2_group": partial_r2,
            "n": int(fitted.nobs),
        })
        for comparison, vector in contrasts.items():
            test = fitted.t_test(vector)
            contrast_rows.append({
                "feature": feature,
                "comparison": comparison,
                "adjusted_standardized_difference": np.asarray(test.effect).item(),
                "robust_se": np.asarray(test.sd).item(),
                "z_value": np.asarray(test.tvalue).item(),
                "p_value": np.asarray(test.pvalue).item(),
            })

    omnibus = pd.DataFrame(omnibus_rows)
    _, omnibus["q_value"], _, _ = multipletests(omnibus["p_value"], method="fdr_bh")
    omnibus = omnibus.sort_values(["q_value", "p_value"]).reset_index(drop=True)

    pairwise = pd.DataFrame(contrast_rows)
    pairwise["q_value"] = np.nan
    for comparison, indices in pairwise.groupby("comparison").groups.items():
        _, q_values, _, _ = multipletests(pairwise.loc[indices, "p_value"], method="fdr_bh")
        pairwise.loc[indices, "q_value"] = q_values
    pairwise = pairwise.sort_values(["comparison", "q_value", "p_value"]).reset_index(drop=True)
    return omnibus, pairwise


def nuisance_only_nested_cv(features: pd.DataFrame, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    encoded = pd.DataFrame(index=features.index)
    encoded["Age"] = pd.to_numeric(features["Age"], errors="coerce")
    encoded["Gender_M"] = (features["Gender"].astype(str).str.upper() == "M").astype(float)
    encoded["log_epochs"] = np.log1p(pd.to_numeric(features["n_epochs"], errors="coerce"))
    feature_sets = {
        "age_gender": ["Age", "Gender_M"],
        "epoch_count": ["log_epochs"],
        "all_nuisance": ["Age", "Gender_M", "log_epochs"],
    }
    encoder = LabelEncoder()
    y = encoder.fit_transform(features["label"].astype(str))
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    metric_rows = []
    prediction_rows = []

    for name, columns in feature_sets.items():
        y_true_all = []
        y_pred_all = []
        for fold, (train_index, test_index) in enumerate(outer.split(encoded, y), start=1):
            estimator = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ])
            inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state + fold)
            search = GridSearchCV(
                estimator,
                {"model__C": [0.1, 1.0, 10.0]},
                cv=inner,
                scoring="balanced_accuracy",
                n_jobs=-1,
            )
            search.fit(encoded.iloc[train_index][columns], y[train_index])
            predicted = search.predict(encoded.iloc[test_index][columns])
            y_true_all.extend(y[test_index])
            y_pred_all.extend(predicted)
            for index, prediction in zip(test_index, encoder.inverse_transform(predicted)):
                prediction_rows.append({
                    "feature_set": name,
                    "fold": fold,
                    "participant_id": features.iloc[index]["participant_id"],
                    "true_label": features.iloc[index]["label"],
                    "pred_label": prediction,
                })
        metric_rows.append({
            "feature_set": name,
            "n_features": len(columns),
            "accuracy": accuracy_score(y_true_all, y_pred_all),
            "balanced_accuracy": balanced_accuracy_score(y_true_all, y_pred_all),
        })
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def write_report(
    output_dir: Path,
    adjusted: pd.DataFrame,
    pairwise: pd.DataFrame,
    nuisance_metrics: pd.DataFrame,
    unadjusted: pd.DataFrame,
    eeg_metrics: pd.DataFrame,
) -> None:
    adjusted_significant = adjusted[adjusted["q_value"] < 0.05]
    top_overlap = len(set(adjusted.head(20)["feature"]) & set(unadjusted.head(20)["feature"]))
    pairwise_counts = (
        pairwise.assign(significant=pairwise["q_value"] < 0.05)
        .groupby("comparison", as_index=False)["significant"]
        .sum()
        .rename(columns={"significant": "fdr_significant_features"})
    )
    best_eeg = eeg_metrics.sort_values("balanced_accuracy", ascending=False).iloc[0]
    best_nuisance = nuisance_metrics.sort_values("balanced_accuracy", ascending=False).iloc[0]

    lines = [
        "# Covariate-Adjusted ds004504 Analysis",
        "",
        "## Model",
        "",
        "Each standardized EEG feature was modeled with diagnostic group, age, gender, and log-transformed retained epoch count. "
        "Group effects were tested jointly with a two-degree-of-freedom Wald test using HC3 heteroskedasticity-robust standard errors. "
        "Benjamini-Hochberg FDR correction was applied across EEG features. Pairwise contrasts were corrected within each comparison.",
        "",
        "## Adjusted Biomarker Results",
        "",
        f"Adjusted FDR-significant EEG features: {len(adjusted_significant)}/{len(adjusted)}. "
        f"Top-20 overlap with the unadjusted Kruskal-Wallis ranking: {top_overlap}/20.",
        "",
        adjusted.head(20).to_markdown(index=False, floatfmt=".4g"),
        "",
        "Pairwise FDR-significant feature counts:",
        "",
        pairwise_counts.to_markdown(index=False),
        "",
        "## Nuisance-Only Classification",
        "",
        nuisance_metrics.to_markdown(index=False, floatfmt=".3f"),
        "",
        f"The strongest nuisance-only baseline was `{best_nuisance['feature_set']}` "
        f"(balanced accuracy {best_nuisance['balanced_accuracy']:.3f}), compared with the best EEG model "
        f"`{best_eeg['model']}` ({best_eeg['balanced_accuracy']:.3f}).",
        "",
        "## Interpretation",
        "",
        "Persistence of FDR-significant group effects after adjustment supports a disease-associated EEG slowing signal, but does not establish causality. "
        "The nuisance-only benchmark quantifies how much classification is available from cohort composition and signal-retention differences alone. "
        "A leakage-safe residualized sensitivity analysis is available in `residualized_report.md`; "
        "external validation is still required for a strong diagnostic claim.",
    ]
    (output_dir / "adjusted_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Covariate-adjusted ds004504 biomarker analysis.")
    parser.add_argument("--config", default="configs/ds004504_minimal.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result_dir = project_path(cfg, cfg["paths"]["output_dir"])
    output_dir = result_dir / "qc"
    output_dir.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(result_dir / "features.csv")
    excluded = {
        "participant_id", "label", "Group", "Gender", "Age", "MMSE", "n_epochs"
    }
    eeg_columns = [
        column for column in features.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]

    adjusted, pairwise = adjusted_feature_tests(features, eeg_columns)
    adjusted.to_csv(output_dir / "adjusted_feature_statistics.csv", index=False)
    pairwise.to_csv(output_dir / "adjusted_pairwise_contrasts.csv", index=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        nuisance_metrics, nuisance_predictions = nuisance_only_nested_cv(
            features,
            random_state=int(cfg["ml"].get("random_state", 42)),
        )
    nuisance_metrics.to_csv(output_dir / "nuisance_baseline_metrics.csv", index=False)
    nuisance_predictions.to_csv(output_dir / "nuisance_baseline_predictions.csv", index=False)

    write_report(
        output_dir,
        adjusted,
        pairwise,
        nuisance_metrics,
        pd.read_csv(result_dir / "feature_statistics.csv"),
        pd.read_csv(result_dir / "model_metrics.csv"),
    )
    print(f"Adjusted outputs written to {output_dir}")


if __name__ == "__main__":
    main()
