from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from tqdm import tqdm

from .agent import write_agent_plan, workflow_checklist
from .audit import write_audit
from .bids import baseline_table, load_participants
from .config import load_config, project_path
from .features import extract_subject_features
from .ml import evaluate_models
from .preprocess import make_epochs
from .report import write_report
from .stats import run_feature_statistics
from .viz import plot_band_topomaps, plot_model_metrics, plot_topomap_composite

app = typer.Typer(help="Run EEG-CogAgent workflows.")


def _stratified_head(participants: pd.DataFrame, n: int) -> pd.DataFrame:
    if n >= len(participants):
        return participants
    group_count = participants["label"].nunique()
    per_group = max(1, n // group_count)
    selected = participants.groupby("label", group_keys=False).head(per_group)
    remaining = n - len(selected)
    if remaining > 0:
        selected_ids = set(selected["participant_id"])
        extras = participants.loc[~participants["participant_id"].isin(selected_ids)].head(remaining)
        selected = pd.concat([selected, extras], ignore_index=False)
    return selected.sort_index().head(n)


@app.command()
def plan(config: Path) -> None:
    """Print the configured agent workflow."""
    cfg = load_config(config)
    typer.echo(workflow_checklist(cfg))


@app.command()
def audit(
    config: Path,
    strict: bool = False,
    output_dir: Path | None = typer.Option(None, "--output-dir"),
) -> None:
    """Audit output completeness, consistency, provenance, and claim guardrails."""
    cfg = load_config(config)
    if output_dir is not None:
        cfg["paths"]["output_dir"] = str(output_dir)
    json_path, markdown_path, manifest_path, result = write_audit(cfg)
    typer.echo(
        f"Audit {result['status'].upper()}: {result['status_counts']['pass']} passed, "
        f"{result['status_counts']['warn']} warnings, {result['status_counts']['fail']} failed"
    )
    typer.echo(f"JSON: {json_path}")
    typer.echo(f"Report: {markdown_path}")
    typer.echo(f"Manifest: {manifest_path}")
    if strict and result["status"] == "fail":
        raise typer.Exit(code=1)


@app.command()
def run(
    config: Path,
    subjects_limit: int | None = None,
    output_dir: Path | None = typer.Option(None, "--output-dir"),
) -> None:
    """Run preprocessing, feature extraction, statistics, ML, figures, and report."""
    cfg = load_config(config)
    bids_root = project_path(cfg, cfg["paths"]["bids_root"])
    configured_output = output_dir if output_dir is not None else cfg["paths"]["output_dir"]
    run_output_dir = project_path(cfg, configured_output)
    figure_dir = run_output_dir / "figures"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    participants = load_participants(bids_root, cfg)
    if subjects_limit:
        participants = _stratified_head(participants, subjects_limit)

    baseline = baseline_table(participants)
    baseline.to_csv(run_output_dir / "table1_baseline.csv", index=False)
    write_agent_plan(cfg, run_output_dir)

    rows = []
    failures = []
    for _, row in tqdm(participants.iterrows(), total=len(participants), desc="subjects"):
        participant_id = row["participant_id"]
        try:
            epochs = make_epochs(bids_root, participant_id, cfg)
            feat = extract_subject_features(epochs, cfg)
            feat.update({
                "participant_id": participant_id,
                "label": row["label"],
                "Group": row.get("Group"),
                "Age": row.get("Age"),
                "MMSE": row.get("MMSE"),
                "Gender": row.get("Gender"),
            })
            rows.append(feat)
        except Exception as exc:  # noqa: BLE001 - keep batch runs alive and report failed subjects.
            failures.append({"participant_id": participant_id, "error": repr(exc)})

    failure_path = run_output_dir / "failed_subjects.csv"
    if failures:
        pd.DataFrame(failures).to_csv(failure_path, index=False)
    elif failure_path.exists():
        failure_path.unlink()
    if not rows:
        raise RuntimeError("No subjects were processed successfully.")

    features = pd.DataFrame(rows)
    features.to_csv(run_output_dir / "features.csv", index=False)

    stats = run_feature_statistics(
        features,
        cfg["statistics"].get("label_column", "label"),
        cfg["statistics"].get("exclude_columns", []),
    )
    stats.to_csv(run_output_dir / "feature_statistics.csv", index=False)

    metrics, predictions = evaluate_models(features, cfg)
    metrics.to_csv(run_output_dir / "model_metrics.csv", index=False)
    predictions.to_csv(run_output_dir / "model_predictions.csv", index=False)

    figures = []
    figures.extend(plot_band_topomaps(features, cfg, figure_dir))
    figures.extend(plot_topomap_composite(features, cfg, figure_dir))
    metric_figure = plot_model_metrics(metrics, figure_dir)
    if metric_figure:
        figures.append(metric_figure)

    report = write_report(cfg, baseline, features, stats, metrics, figures, run_output_dir)
    typer.echo(f"Wrote {run_output_dir}")
    typer.echo(f"Report: {report}")


if __name__ == "__main__":
    app()
