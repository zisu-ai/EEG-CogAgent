from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import project_path


def _check(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _software_versions() -> dict[str, str]:
    packages = ["mne", "mne-bids", "numpy", "pandas", "scikit-learn", "scipy", "statsmodels"]
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _artifact_manifest(output_dir: Path) -> list[dict[str, Any]]:
    ignored = {"agent_audit.json", "agent_audit.md", "artifact_manifest.json"}
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name in ignored or "__pycache__" in path.parts:
            continue
        rows.append({
            "path": path.relative_to(output_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    return rows


def audit_run(cfg: dict[str, Any]) -> dict[str, Any]:
    output_dir = project_path(cfg, cfg["paths"]["output_dir"])
    bids_root = project_path(cfg, cfg["paths"]["bids_root"])
    checks: list[dict[str, str]] = []
    required = {
        "participants table": bids_root / cfg["participants"].get("file", "participants.tsv"),
        "feature matrix": output_dir / "features.csv",
        "baseline table": output_dir / "table1_baseline.csv",
        "feature statistics": output_dir / "feature_statistics.csv",
        "model metrics": output_dir / "model_metrics.csv",
        "model predictions": output_dir / "model_predictions.csv",
        "automated report": output_dir / "auto_report.md",
        "agent plan": output_dir / "agent_plan.md",
    }
    for label, path in required.items():
        status = "pass" if path.exists() and path.stat().st_size > 0 else "fail"
        checks.append(_check(f"artifact:{label}", status, str(path)))

    summary: dict[str, Any] = {
        "dataset_id": cfg["project"].get("dataset_id"),
        "task": cfg["project"].get("task"),
        "output_dir": str(output_dir),
    }
    features_path = output_dir / "features.csv"
    metrics_path = output_dir / "model_metrics.csv"
    predictions_path = output_dir / "model_predictions.csv"
    statistics_path = output_dir / "feature_statistics.csv"
    baseline_path = output_dir / "table1_baseline.csv"

    features = pd.read_csv(features_path) if features_path.exists() else pd.DataFrame()
    if not features.empty:
        unique_participants = int(features["participant_id"].nunique())
        duplicate_count = int(features["participant_id"].duplicated().sum())
        summary["processed_participants"] = unique_participants
        summary["group_counts"] = features["label"].value_counts().sort_index().to_dict()
        checks.append(_check(
            "features:unique-participants",
            "pass" if duplicate_count == 0 else "fail",
            f"rows={len(features)}, unique_participants={unique_participants}, duplicates={duplicate_count}",
        ))
        numeric = features.select_dtypes(include=[np.number]).columns
        nonfinite = int((~np.isfinite(features[numeric].to_numpy(dtype=float))).sum())
        checks.append(_check(
            "features:finite-values",
            "pass" if nonfinite == 0 else "warn",
            f"nonfinite_numeric_values={nonfinite}",
        ))

        if baseline_path.exists():
            baseline = pd.read_csv(baseline_path)
            baseline_n = int(baseline["n"].sum())
            checks.append(_check(
                "cohort:baseline-feature-consistency",
                "pass" if baseline_n == unique_participants else "fail",
                f"baseline_n={baseline_n}, feature_n={unique_participants}",
            ))

    if statistics_path.exists():
        statistics = pd.read_csv(statistics_path)
        q_valid = bool(statistics["q_value"].between(0, 1, inclusive="both").all())
        checks.append(_check(
            "statistics:q-value-range",
            "pass" if q_valid else "fail",
            f"rows={len(statistics)}, fdr_significant={int((statistics['q_value'] < 0.05).sum())}",
        ))

    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    if not metrics.empty:
        metric_columns = ["accuracy", "balanced_accuracy", "auc_ovr"]
        valid = all(metrics[column].dropna().between(0, 1, inclusive="both").all() for column in metric_columns)
        best = metrics.sort_values("balanced_accuracy", ascending=False).iloc[0]
        summary["best_model"] = str(best["model"])
        summary["best_balanced_accuracy"] = float(best["balanced_accuracy"])
        summary["best_auc_ovr"] = float(best["auc_ovr"])
        checks.append(_check(
            "models:metric-range",
            "pass" if valid else "fail",
            f"models={len(metrics)}, best={best['model']}, balanced_accuracy={best['balanced_accuracy']:.3f}",
        ))

    if predictions_path.exists() and not features.empty:
        predictions = pd.read_csv(predictions_path)
        expected_models = set(metrics["model"]) if not metrics.empty else set(predictions["model"])
        coverage = predictions.groupby("model")["participant_id"].nunique().to_dict()
        complete = all(coverage.get(model, 0) == len(features) for model in expected_models)
        duplicates = int(predictions.duplicated(["model", "participant_id"]).sum())
        checks.append(_check(
            "models:out-of-fold-coverage",
            "pass" if complete and duplicates == 0 else "fail",
            f"coverage={coverage}, duplicate_model_participants={duplicates}",
        ))

    failed_path = output_dir / "failed_subjects.csv"
    if failed_path.exists():
        failures = pd.read_csv(failed_path)
        checks.append(_check(
            "processing:failed-subjects",
            "warn" if len(failures) else "pass",
            f"failed_subjects={len(failures)}",
        ))
    else:
        checks.append(_check("processing:failed-subjects", "pass", "failed_subjects=0"))

    optional_artifacts = {
        "quality-control report": output_dir / "qc" / "qc_report.md",
        "adjusted analysis": output_dir / "qc" / "adjusted_report.md",
        "residualized analysis": output_dir / "qc" / "residualized_report.md",
        "connectivity report": output_dir / "connectivity" / "connectivity_report.md",
        "workflow figure": output_dir / "figures" / "figure1_workflow.pdf",
        "connectivity figure": output_dir / "figures" / "figure3_connectivity.pdf",
        "ROC figure": output_dir / "figures" / "figure4_roc.pdf",
    }
    for label, path in optional_artifacts.items():
        checks.append(_check(
            f"extension:{label}",
            "pass" if path.exists() and path.stat().st_size > 0 else "warn",
            str(path),
        ))

    status_counts = {
        status: sum(check["status"] == status for check in checks)
        for status in ["pass", "warn", "fail"]
    }
    overall_status = "fail" if status_counts["fail"] else ("warn" if status_counts["warn"] else "pass")
    manifest = _artifact_manifest(output_dir) if output_dir.exists() else []
    return {
        "status": overall_status,
        "status_counts": status_counts,
        "summary": summary,
        "checks": checks,
        "software": _software_versions(),
        "artifacts": manifest,
    }


def write_audit(cfg: dict[str, Any]) -> tuple[Path, Path, Path, dict[str, Any]]:
    output_dir = project_path(cfg, cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_run(cfg)
    json_path = output_dir / "agent_audit.json"
    markdown_path = output_dir / "agent_audit.md"
    manifest_path = output_dir / "artifact_manifest.json"
    json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(audit["artifacts"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    summary = audit["summary"]
    lines = [
        "# EEG-CogAgent Run Audit",
        "",
        f"Overall status: **{audit['status'].upper()}**. "
        f"Checks: {audit['status_counts']['pass']} passed, "
        f"{audit['status_counts']['warn']} warnings, {audit['status_counts']['fail']} failed.",
        "",
        "## Run Summary",
        "",
        f"Dataset `{summary.get('dataset_id')}` task `{summary.get('task')}` processed "
        f"{summary.get('processed_participants', 'unknown')} participants. "
        f"The best internally evaluated model was `{summary.get('best_model', 'unavailable')}` "
        f"with balanced accuracy {summary.get('best_balanced_accuracy', float('nan')):.3f}.",
        "",
        "## Checks",
        "",
        pd.DataFrame(audit["checks"]).to_markdown(index=False),
        "",
        "## Interpretation Guardrail",
        "",
        "This audit verifies artifact completeness and internal numerical consistency. It does not "
        "establish clinical validity, absence of unmeasured confounding, or external generalizability.",
    ]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path, manifest_path, audit
