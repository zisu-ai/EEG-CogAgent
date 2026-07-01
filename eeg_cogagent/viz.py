from __future__ import annotations

from pathlib import Path

import math

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize


GROUP_COLORS = {"AD": "#D55E00", "FTD": "#56B4E9", "HC": "#009E73"}

GROUP_ORDER = ["AD", "FTD", "HC"]
TOPOMAP_CMAP = "cividis"
TOPOMAP_CBAR_LABEL = "Mean log band power (a.u.)"


def _shared_topomap_limits(group_means: dict) -> tuple[float, float]:
    """Finite, padded color range shared across all group mean vectors."""
    stacked = np.concatenate(list(group_means.values())) if group_means else np.array([])
    finite = stacked[np.isfinite(stacked)]
    if finite.size == 0 or not (np.isfinite(finite.min()) and np.isfinite(finite.max())):
        return 0.0, 1.0
    low, high = float(finite.min()), float(finite.max())
    if low == high:
        pad = max(abs(low) * 1e-3, 1e-6)
        low -= pad
        high += pad
    return low, high


def _band_topomap_inputs(features: pd.DataFrame, band: str, labels: list[str]):
    """Build the MNE info, absolute group-mean vectors, and shared ``vlim`` for one band.

    Normalization is identical to ``plot_band_topomaps``: one finite color range
    derived from all group-channel means pooled across groups, never rescaled per group.
    Returns ``None`` when the band has no channel columns.
    """
    import mne

    prefix = f"band_ch__{band}__"
    cols = [col for col in features.columns if col.startswith(prefix)]
    if not cols:
        return None
    channels = [col.removeprefix(prefix) for col in cols]
    info = mne.create_info(channels, sfreq=500, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, on_missing="ignore")

    group_means = {
        label: features.loc[features["label"] == label, cols].mean(axis=0).to_numpy(dtype=float)
        for label in labels
    }
    vlim = _shared_topomap_limits(group_means)
    return info, group_means, vlim


def plot_band_topomaps(features: pd.DataFrame, cfg: dict, output_dir: Path) -> list[Path]:
    import mne

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    available = set(features["label"].dropna().unique())
    labels = [label for label in GROUP_ORDER if label in available] or sorted(available)
    bands = cfg["features"]["bands"].keys()

    for band in bands:
        inputs = _band_topomap_inputs(features, band, labels)
        if inputs is None:
            continue
        info, group_means, vlim = inputs

        fig, axes = plt.subplots(1, len(labels), figsize=(4 * len(labels), 3.5), constrained_layout=True)
        if len(labels) == 1:
            axes = [axes]
        image = None
        for ax, label in zip(axes, labels):
            image, _ = mne.viz.plot_topomap(
                group_means[label], info, axes=ax, cmap=TOPOMAP_CMAP, vlim=vlim,
                contours=6, show=False,
            )
            ax.set_title(label)
        fig.colorbar(image, ax=list(axes), shrink=0.7, label=TOPOMAP_CBAR_LABEL)
        fig.suptitle(f"{band.capitalize()} band")
        png = output_dir / f"topomap_{band}.png"
        pdf = output_dir / f"topomap_{band}.pdf"
        fig.savefig(png, dpi=220)
        fig.savefig(pdf)
        plt.close(fig)
        paths.extend([png, pdf])
    return paths


def plot_topomap_composite(features: pd.DataFrame, cfg: dict, output_dir: Path) -> list[Path]:
    """Publication composite of group-mean spectral topomaps.

    Five band rows (delta, theta, alpha, beta, gamma) by three group columns
    (AD, FTD, HC) with one compact per-row colorbar. Each band reuses the exact
    normalization from :func:`plot_band_topomaps`: a single finite ``vlim`` shared
    across all three groups, the ``cividis`` colormap, and absolute group-mean
    log band power (no per-group rescaling, no threshold changes).
    """
    import mne
    from matplotlib.gridspec import GridSpec

    output_dir.mkdir(parents=True, exist_ok=True)
    available = set(features["label"].dropna().unique())
    labels = [label for label in GROUP_ORDER if label in available] or sorted(available)

    rows: list[tuple[str, tuple]] = []
    for band in cfg["features"]["bands"].keys():
        inputs = _band_topomap_inputs(features, band, labels)
        if inputs is not None:
            rows.append((band, inputs))
    if not rows:
        return []

    n_rows = len(rows)
    n_groups = len(labels)
    fig = plt.figure(figsize=(7.2, 8.8), facecolor="white")
    grid = GridSpec(
        n_rows, n_groups + 1,
        width_ratios=[1.0] * n_groups + [0.05],
        left=0.155, right=0.88, top=0.905, bottom=0.045,
        wspace=0.05, hspace=0.16,
    )

    column_centers: list[float] = []
    for row_index, (band, (info, group_means, vlim)) in enumerate(rows):
        row_image = None
        row_position = None
        for col_index, label in enumerate(labels):
            ax = fig.add_subplot(grid[row_index, col_index])
            position = ax.get_position()
            if col_index == 0:
                row_position = position
            if row_index == 0:
                column_centers.append(position.x0 + position.width / 2)
            row_image, _ = mne.viz.plot_topomap(
                group_means[label], info, axes=ax, cmap=TOPOMAP_CMAP,
                vlim=vlim, contours=6, show=False,
            )
            ax.set_axis_off()

        cax = fig.add_subplot(grid[row_index, n_groups])
        colorbar = fig.colorbar(row_image, cax=cax)
        ticks = np.linspace(vlim[0], vlim[1], 3)
        colorbar.set_ticks(ticks)
        colorbar.set_ticklabels([f"{value:.2f}" for value in ticks])
        colorbar.ax.tick_params(labelsize=6, length=2.2, pad=2)
        colorbar.outline.set_linewidth(0.5)

        y_center = row_position.y0 + row_position.height / 2
        fig.text(0.048, y_center, chr(65 + row_index), fontsize=12,
                 fontweight="bold", va="center", ha="left")
        fig.text(0.082, y_center, band.capitalize(), fontsize=9.5,
                 va="center", ha="left", color="#1F2937")

    for label, x_center in zip(labels, column_centers):
        fig.text(x_center, 0.928, label, fontsize=10, fontweight="bold",
                 ha="center", va="bottom")

    fig.text(0.5, 0.968, "Group-mean scalp spectral topomaps",
             fontsize=11, ha="center", va="center")
    fig.text(0.955, 0.5, TOPOMAP_CBAR_LABEL, rotation=90, fontsize=8,
             ha="center", va="center", color="#374151")

    png = output_dir / "figure2_spectral_topomaps.png"
    pdf = output_dir / "figure2_spectral_topomaps.pdf"
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return [png, pdf]


def plot_model_metrics(metrics: pd.DataFrame, output_dir: Path) -> Path | None:
    if metrics.empty:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    ax = metrics.set_index("model")[["balanced_accuracy", "auc_ovr"]].plot(kind="bar", ylim=(0, 1), figsize=(7, 4))
    ax.set_ylabel("score")
    ax.set_xlabel("")
    ax.legend(loc="lower right")
    plt.tight_layout()
    out = output_dir / "model_metrics.png"
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def plot_connectivity_figure(
    features: pd.DataFrame,
    edges: pd.DataFrame,
    statistics: pd.DataFrame,
    output_dir: Path,
    proportional_threshold: float = 0.2,
) -> list[Path]:
    """Create a publication-ready group wPLI network and feature distribution figure."""
    import mne

    candidates = statistics[
        statistics["feature"].str.startswith(("conn_global__wpli__", "graph__wpli__"))
    ]
    if candidates.empty:
        return []
    top = candidates.iloc[0]
    feature_name = str(top["feature"])
    parts = feature_name.split("__")
    band = parts[2] if feature_name.startswith("conn_global") else parts[2]
    band_edges = edges[(edges["metric"] == "wpli") & (edges["band"] == band)].copy()
    if band_edges.empty:
        return []

    channels = sorted(set(band_edges["source"]) | set(band_edges["target"]))
    montage = mne.channels.make_standard_montage("standard_1020")
    positions_3d = montage.get_positions()["ch_pos"]
    positions = {channel: positions_3d[channel][:2] for channel in channels if channel in positions_3d}
    channels = [channel for channel in channels if channel in positions]

    group_means = (
        band_edges.groupby(["label", "source", "target"], as_index=False)["value"].mean()
    )
    all_group_weights = group_means["value"].to_numpy(dtype=float)
    weight_min = float(np.nanmin(all_group_weights))
    weight_max = float(np.nanmax(all_group_weights))
    denominator = max(weight_max - weight_min, np.finfo(float).eps)

    labels = [label for label in ["AD", "FTD", "HC"] if label in set(features["label"])]
    fig, axes = plt.subplots(1, len(labels) + 1, figsize=(7.2, 2.45), constrained_layout=True)
    for panel_index, (axis, label) in enumerate(zip(axes[:-1], labels)):
        group = group_means[group_means["label"] == label].sort_values("value", ascending=False)
        n_possible = len(channels) * (len(channels) - 1) // 2
        group = group.head(max(1, int(math.ceil(n_possible * proportional_threshold))))
        for _, edge in group.iterrows():
            if edge["source"] not in positions or edge["target"] not in positions:
                continue
            start = positions[edge["source"]]
            end = positions[edge["target"]]
            scaled = (float(edge["value"]) - weight_min) / denominator
            axis.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color="#6B7280",
                alpha=0.18 + 0.62 * scaled,
                linewidth=0.35 + 1.8 * scaled,
                zorder=1,
            )
        xy = np.array([positions[channel] for channel in channels])
        axis.scatter(
            xy[:, 0], xy[:, 1], s=17, color=GROUP_COLORS.get(label, "#777777"),
            edgecolor="white", linewidth=0.35, zorder=2,
        )
        for channel, (x_value, y_value) in zip(channels, xy):
            axis.text(x_value, y_value + 0.008, channel, fontsize=4.5, ha="center", va="bottom")
        axis.set_title(f"{label} (n={(features['label'] == label).sum()})", fontsize=8)
        axis.set_aspect("equal")
        axis.axis("off")
        axis.text(-0.08, 1.03, chr(65 + panel_index), transform=axis.transAxes,
                  fontsize=9, fontweight="bold", va="top")

    distribution_axis = axes[-1]
    rng = np.random.default_rng(42)
    values_by_group = [
        features.loc[features["label"] == label, feature_name].dropna().to_numpy(dtype=float)
        for label in labels
    ]
    box = distribution_axis.boxplot(
        values_by_group, positions=np.arange(len(labels)), widths=0.55,
        patch_artist=True, showfliers=False,
        medianprops={"color": "#111827", "linewidth": 1.1},
        whiskerprops={"color": "#4B5563", "linewidth": 0.8},
        capprops={"color": "#4B5563", "linewidth": 0.8},
    )
    for patch, label in zip(box["boxes"], labels):
        patch.set_facecolor(GROUP_COLORS.get(label, "#999999"))
        patch.set_alpha(0.55)
        patch.set_edgecolor("#374151")
    for position, (label, values) in enumerate(zip(labels, values_by_group)):
        jitter = rng.normal(0, 0.055, len(values))
        distribution_axis.scatter(
            position + jitter, values, s=8, alpha=0.55,
            color=GROUP_COLORS.get(label, "#777777"), edgecolor="none",
        )
    readable_feature = feature_name.replace("__", " · ").replace("_", " ")
    distribution_axis.set_xticks(np.arange(len(labels)), labels)
    distribution_axis.set_ylabel(readable_feature, fontsize=7)
    distribution_axis.set_title(f"Kruskal–Wallis FDR q={float(top['q_value']):.2g}", fontsize=7.5)
    distribution_axis.spines[["top", "right"]].set_visible(False)
    distribution_axis.tick_params(labelsize=6.5)
    distribution_axis.text(-0.16, 1.03, chr(65 + len(labels)), transform=distribution_axis.transAxes,
                           fontsize=9, fontweight="bold", va="top")
    fig.suptitle(f"{band.capitalize()}-band wPLI networks (top {proportional_threshold:.0%} edges)", fontsize=9)

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "figure3_connectivity.png"
    pdf = output_dir / "figure3_connectivity.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def plot_multiclass_roc(
    predictions: pd.DataFrame,
    output_dir: Path,
    stem: str = "figure4_roc",
) -> list[Path]:
    """Plot one-vs-rest ROC curves from out-of-fold class probabilities."""
    probability_columns = [column for column in predictions if column.startswith("prob_")]
    if not probability_columns:
        return []
    classes = [column.removeprefix("prob_") for column in probability_columns]
    models = list(predictions["model"].drop_duplicates())
    fig, axes = plt.subplots(1, len(models), figsize=(7.2, 2.45), constrained_layout=True)
    if len(models) == 1:
        axes = [axes]
    class_colors = {"AD": "#D55E00", "FTD": "#56B4E9", "HC": "#009E73"}

    for panel_index, (axis, model) in enumerate(zip(axes, models)):
        subset = predictions[predictions["model"] == model]
        truth = label_binarize(subset["true_label"], classes=classes)
        for class_index, class_name in enumerate(classes):
            false_positive, true_positive, _ = roc_curve(
                truth[:, class_index], subset[f"prob_{class_name}"].to_numpy(dtype=float)
            )
            class_auc = auc(false_positive, true_positive)
            axis.plot(
                false_positive,
                true_positive,
                color=class_colors.get(class_name, "#555555"),
                linewidth=1.5,
                label=f"{class_name} ({class_auc:.2f})",
            )
        axis.plot([0, 1], [0, 1], linestyle="--", color="#9CA3AF", linewidth=0.8)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.02)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("False-positive rate", fontsize=7)
        if panel_index == 0:
            axis.set_ylabel("True-positive rate", fontsize=7)
        title = model.replace("_", " ").title().replace("Svm Rbf", "SVM (RBF)")
        axis.set_title(title, fontsize=8)
        axis.legend(title="Class (AUC)", frameon=False, fontsize=6, title_fontsize=6.5, loc="lower right")
        axis.tick_params(labelsize=6.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(-0.17, 1.04, chr(65 + panel_index), transform=axis.transAxes,
                  fontsize=9, fontweight="bold", va="top")
    fig.suptitle("Leakage-safe residualized nested-CV ROC curves", fontsize=9)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def plot_workflow_figure(output_dir: Path) -> list[Path]:
    """Create a vector workflow schematic for the EEG-CogAgent framework."""
    fig, axis = plt.subplots(figsize=(7.2, 3.1))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    stages = [
        ("BIDS EEG\n+ YAML", "Data + analysis\ncontract", "#E8F1FA"),
        ("MNE\npreprocessing", "Filter · reference\nEpochs · QC", "#E7F5EF"),
        ("Interpretable\nfeatures", "Power · ratios\nConnectivity · graphs", "#FFF3D6"),
        ("Statistical & ML\nvalidation", "FDR statistics\nNested CV · sensitivity", "#F4E9F7"),
        ("Research\noutputs", "Tables · figures\nReports · drafts", "#FCE8E6"),
    ]
    left_margin = 0.025
    gap = 0.017
    width = (0.95 - gap * (len(stages) - 1)) / len(stages)
    y_value = 0.32
    height = 0.34
    centers = []
    for index, (title, subtitle, color) in enumerate(stages):
        x_value = left_margin + index * (width + gap)
        centers.append((x_value + width / 2, y_value + height / 2))
        patch = FancyBboxPatch(
            (x_value, y_value), width, height,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            facecolor=color, edgecolor="#4B5563", linewidth=0.8,
        )
        axis.add_patch(patch)
        axis.text(x_value + width / 2, y_value + 0.225, title,
                  ha="center", va="center", fontsize=6.8, fontweight="bold", linespacing=1.1)
        axis.text(x_value + width / 2, y_value + 0.095, subtitle,
                  ha="center", va="center", fontsize=5.25, color="#374151", linespacing=1.25)
        if index:
            previous = centers[index - 1]
            arrow = FancyArrowPatch(
                (previous[0] + width / 2, previous[1]),
                (x_value - 0.004, y_value + height / 2),
                arrowstyle="-|>", mutation_scale=8, linewidth=0.9, color="#4B5563",
            )
            axis.add_patch(arrow)

    orchestration = FancyBboxPatch(
        (0.17, 0.76), 0.66, 0.14,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor="#1F4E79", edgecolor="#163A5B", linewidth=0.9,
    )
    axis.add_patch(orchestration)
    axis.text(0.5, 0.845, "LLM-assisted orchestration layer", color="white",
              fontsize=9, fontweight="bold", ha="center", va="center")
    axis.text(0.5, 0.795, "plans tools · checks artifacts · summarizes evidence · never diagnoses",
              color="#E5EEF6", fontsize=6.4, ha="center", va="center")
    for center in centers:
        axis.add_patch(FancyArrowPatch(
            (0.5, 0.76), (center[0], y_value + height + 0.008),
            connectionstyle="arc3,rad=0", arrowstyle="-|>", mutation_scale=6,
            linewidth=0.55, color="#6B7280", alpha=0.75,
        ))

    guardrail = FancyBboxPatch(
        (0.09, 0.08), 0.82, 0.13,
        boxstyle="round,pad=0.01,rounding_size=0.018",
        facecolor="#F8FAFC", edgecolor="#9CA3AF", linewidth=0.8,
    )
    axis.add_patch(guardrail)
    axis.text(0.5, 0.155, "Reproducibility and safety guardrails", fontsize=7.4,
              fontweight="bold", ha="center", va="center", color="#1F2937")
    axis.text(0.5, 0.105,
              "deterministic numerical modules · machine-readable provenance · subject-level CV · research-only claims",
              fontsize=6.1, ha="center", va="center", color="#4B5563")
    axis.text(0.01, 0.97, "A", fontsize=10, fontweight="bold", va="top")

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "figure1_workflow.png"
    pdf = output_dir / "figure1_workflow.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [png, pdf]
