from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from eeg_cogagent.bids import load_participants
from eeg_cogagent.config import load_config, project_path
from eeg_cogagent.connectivity import extract_connectivity_features, matrices_to_edge_rows
from eeg_cogagent.preprocess import make_epochs
from eeg_cogagent.stats import run_feature_statistics, run_pairwise_feature_statistics
from eeg_cogagent.viz import plot_connectivity_figure


def _process_subject(cfg: dict, participant: dict) -> tuple[dict, list[dict]]:
    bids_root = project_path(cfg, cfg["paths"]["bids_root"])
    participant_id = participant["participant_id"]
    epochs = make_epochs(bids_root, participant_id, cfg)
    features, matrices = extract_connectivity_features(epochs, cfg)
    features.update({
        "participant_id": participant_id,
        "label": participant["label"],
        "Group": participant.get("Group"),
        "Age": participant.get("Age"),
        "MMSE": participant.get("MMSE"),
        "Gender": participant.get("Gender"),
        "n_epochs": len(epochs),
    })
    edges = matrices_to_edge_rows(
        matrices, list(epochs.ch_names), participant_id, participant["label"]
    )
    return features, edges


def write_report(
    output_dir: Path,
    features: pd.DataFrame,
    statistics: pd.DataFrame,
    pairwise: pd.DataFrame,
    failures: pd.DataFrame,
) -> Path:
    significant = statistics[statistics["q_value"] < 0.05]
    top_table = statistics.head(20)
    pairwise_counts = (
        pairwise.assign(significant=pairwise["q_value"] < 0.05)
        .groupby("comparison", as_index=False)["significant"]
        .sum()
        .rename(columns={"significant": "fdr_significant_features"})
    )
    lines = [
        "# Functional Connectivity and Graph Analysis",
        "",
        "## Methods",
        "",
        "Epoch-wise Fourier coefficients were calculated after demeaning and Hann tapering. "
        "Magnitude-squared coherence, phase-lag index (PLI), and weighted phase-lag index (wPLI) "
        "were estimated separately in the delta, theta, alpha, beta, and gamma bands. The wPLI "
        "matrices were proportionally thresholded at a fixed 20% edge density. Weighted clustering "
        "coefficient, global efficiency using inverse connection weight as path length, and mean "
        "node strength were then calculated. Group differences were screened using Kruskal–Wallis "
        "tests with Benjamini–Hochberg false discovery rate correction.",
        "",
        "## Processing",
        "",
        f"Connectivity features were generated for {len(features)} participants; "
        f"{len(failures)} participants failed processing. Each participant contributed 15 global "
        "connectivity summaries and 15 wPLI graph features.",
        "",
        "## Results",
        "",
        f"FDR-significant connectivity or graph features: {len(significant)}/{len(statistics)}.",
        "",
        top_table.to_markdown(index=False, floatfmt=".4g"),
        "",
        "Pairwise Mann–Whitney results, FDR-corrected within each comparison:",
        "",
        pairwise_counts.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "These source-space-unresolved scalp networks are exploratory biomarkers. PLI and wPLI "
        "reduce sensitivity to zero-lag coupling but do not remove volume-conduction effects, "
        "reference dependence, or all preprocessing-related bias. Network findings should therefore "
        "support the interpretable workflow demonstration rather than a clinical diagnostic claim.",
    ]
    path = output_dir / "connectivity_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ds004504 connectivity and graph analysis.")
    parser.add_argument("--config", default="configs/ds004504_minimal.yaml")
    parser.add_argument("--subjects-limit", type=int)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    bids_root = project_path(cfg, cfg["paths"]["bids_root"])
    result_dir = project_path(cfg, cfg["paths"]["output_dir"])
    output_dir = result_dir / "connectivity"
    figure_dir = result_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    participants = load_participants(bids_root, cfg)
    if args.subjects_limit:
        participants = participants.groupby("label", group_keys=False).head(
            max(1, args.subjects_limit // participants["label"].nunique())
        ).head(args.subjects_limit)
    records = participants.to_dict(orient="records")
    workers = args.workers or int(cfg.get("connectivity", {}).get("workers", 1))
    feature_rows: list[dict] = []
    edge_rows: list[dict] = []
    failure_rows: list[dict] = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_subject, cfg, participant): participant["participant_id"]
            for participant in records
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="connectivity subjects"):
            participant_id = futures[future]
            try:
                subject_features, subject_edges = future.result()
                feature_rows.append(subject_features)
                edge_rows.extend(subject_edges)
            except Exception as exc:  # noqa: BLE001 - preserve batch progress and report failures.
                failure_rows.append({"participant_id": participant_id, "error": repr(exc)})

    if not feature_rows:
        raise RuntimeError("No participants completed connectivity processing.")
    features = pd.DataFrame(feature_rows).sort_values("participant_id").reset_index(drop=True)
    edges = pd.DataFrame(edge_rows)
    failures = pd.DataFrame(failure_rows, columns=["participant_id", "error"])
    statistics = run_feature_statistics(features, "label")
    pairwise = run_pairwise_feature_statistics(features, "label")

    features.to_csv(output_dir / "connectivity_features.csv", index=False)
    edges.to_csv(output_dir / "connectivity_edges.csv.gz", index=False, compression="gzip")
    statistics.to_csv(output_dir / "connectivity_statistics.csv", index=False)
    pairwise.to_csv(output_dir / "connectivity_pairwise_statistics.csv", index=False)
    failures.to_csv(output_dir / "failed_subjects.csv", index=False)
    figure_paths = plot_connectivity_figure(
        features,
        edges,
        statistics,
        figure_dir,
        proportional_threshold=float(cfg["connectivity"].get("proportional_threshold", 0.2)),
    )
    report = write_report(output_dir, features, statistics, pairwise, failures)
    print(f"Connectivity outputs written to {output_dir}")
    print(f"Report: {report}")
    print(f"Figures: {', '.join(str(path) for path in figure_paths)}")


if __name__ == "__main__":
    main()
