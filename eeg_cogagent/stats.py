from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests


def run_feature_statistics(
    features: pd.DataFrame,
    label_column: str = "label",
    exclude_columns: list[str] | None = None,
) -> pd.DataFrame:
    exclude = set(exclude_columns or [])
    exclude.update({"participant_id", label_column, "label", "Group", "Gender", "Age", "MMSE", "n_epochs"})
    numeric_cols = [
        col for col in features.select_dtypes(include=[np.number]).columns
        if col not in exclude
    ]
    rows = []
    grouped = list(features.groupby(label_column))
    for col in numeric_cols:
        samples = [group[col].dropna().to_numpy() for _, group in grouped]
        samples = [sample for sample in samples if len(sample) >= 2]
        if len(samples) < 2:
            continue
        stat, p_value = kruskal(*samples)
        rows.append({"feature": col, "statistic": float(stat), "p_value": float(p_value)})

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    _, q_values, _, _ = multipletests(result["p_value"], method="fdr_bh")
    result["q_value"] = q_values
    return result.sort_values(["q_value", "p_value"]).reset_index(drop=True)


def run_pairwise_feature_statistics(
    features: pd.DataFrame,
    label_column: str = "label",
    exclude_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Run pairwise Mann–Whitney tests with FDR correction per comparison."""
    exclude = set(exclude_columns or [])
    exclude.update({"participant_id", label_column, "label", "Group", "Gender", "Age", "MMSE", "n_epochs"})
    numeric_cols = [
        col for col in features.select_dtypes(include=[np.number]).columns
        if col not in exclude
    ]
    labels = sorted(features[label_column].dropna().astype(str).unique())
    rows = []
    for left_index, left_label in enumerate(labels):
        for right_label in labels[left_index + 1:]:
            comparison = f"{left_label}_vs_{right_label}"
            for column in numeric_cols:
                left = features.loc[features[label_column].astype(str) == left_label, column].dropna().to_numpy()
                right = features.loc[features[label_column].astype(str) == right_label, column].dropna().to_numpy()
                if len(left) < 2 or len(right) < 2:
                    continue
                statistic, p_value = mannwhitneyu(left, right, alternative="two-sided")
                rank_biserial = 2.0 * statistic / (len(left) * len(right)) - 1.0
                rows.append({
                    "comparison": comparison,
                    "feature": column,
                    "n_left": len(left),
                    "n_right": len(right),
                    "median_left": float(np.median(left)),
                    "median_right": float(np.median(right)),
                    "rank_biserial": float(rank_biserial),
                    "p_value": float(p_value),
                })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["q_value"] = np.nan
    for _, indices in result.groupby("comparison").groups.items():
        _, q_values, _, _ = multipletests(result.loc[indices, "p_value"], method="fdr_bh")
        result.loc[indices, "q_value"] = q_values
    return result.sort_values(["comparison", "q_value", "p_value"]).reset_index(drop=True)
