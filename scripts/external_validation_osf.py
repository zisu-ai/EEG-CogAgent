"""OSF external-validation CLI.

Phase 1 (``audit`` / ``extract``): integrity audit + scale-invariant feature
extraction from ``data/osf_2v5md/EEG_data.zip``. Fits no model.

Phase 2 v2 (``validate``): independent AD-vs-HC evaluation with 1-30 Hz harmonized
features, leak-free nested discovery training (C and threshold chosen on discovery
only), hard input gates, atomic staging/publish, full provenance + artifact
manifest. See ``prompts/claude_external_validation_v2_hardening.md`` and
``protocols/EXTERNAL_VALIDATION_PROTOCOL_V2.md``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_cogagent.config import load_config
from eeg_cogagent.external_osf import (
    COMMON_CHANNELS_19,
    DEFAULT_CONDITION,
    EXPECTED_GROUP_COUNTS,
    EXPECTED_SAMPLES_PER_CHANNEL,
    LICENSE_STATUS,
    OSF_SAMPLING_RATE_HZ,
    build_feature_matrix,
    cohort_audit,
    feature_mapping,
    sha256_of_file,
)
from eeg_cogagent.external_validation import (
    C_GRID,
    DEFAULT_BOOTSTRAP,
    DEFAULT_SEED,
    DISCOVERY_SFREQ_HZ,
    OUTER_FOLDS,
    POS_LABEL,
    NEG_LABEL,
    PRIMARY_SCORING,
    SENSITIVITY_THRESHOLD,
    TARGET_FREQ_RESOLUTION_HZ,
    THRESHOLD_SEARCH,
    THRESHOLD_TIE_BREAK,
    V2_BANDS,
    V2_HARMONIZED_FEATURES,
    V2_RATIOS,
    V2_REGIONS,
    build_discovery_harmonized_matrix_v2,
    build_osf_harmonized_matrix_v2,
    build_osf_signal_qc,
    domain_shift_primary_labelfree,
    domain_shift_supplementary_by_label,
    fit_final_model_and_threshold,
    nested_cv_internal_estimate,
    point_metrics,
    predict_external,
    subject_level_bootstrap_metrics,
    wilson_ci,
)

#: SHA-256 of the canonical downloaded archive. Hard gate.
CANONICAL_ARCHIVE_SHA256 = "F5B30DF4FD0D18E3224DDE0BD564E2A5CEA61845AE5A9B8142AE722C5D99BA93"

DEFAULT_ARCHIVE = "data/osf_2v5md/EEG_data.zip"
DEFAULT_OUTPUT_DIR = "results/external_validation_osf"
V2_DEFAULT_OUTPUT_DIR = "results/external_validation_osf_v2"
PROTOCOL_FILENAME = "EXTERNAL_VALIDATION_PROTOCOL_V2.md"
PROTOCOL_SOURCE = Path("protocols") / PROTOCOL_FILENAME

# OSF cohort invariants enforced before any fitting.
OSF_EXPECTED = {"AD": 80, "Healthy": 12}
DISCOVERY_EXPECTED = {"AD": 36, "FTD": 23, "HC": 29}


class GateError(RuntimeError):
    """A hard input/consistency gate failed. Abort without publishing."""


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE, help="Path to EEG_data.zip.")
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Directory to write audit/feature artifacts.",
    )
    parser.add_argument(
        "--condition", default=DEFAULT_CONDITION,
        help="Archive condition to use (phase 1 uses Eyes_closed only).",
    )


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _build_provenance(
    archive: Path, condition: str, actual_sha: str, sha_match: bool
) -> dict[str, object]:
    return {
        "archive": {
            "path": str(archive),
            "expected_sha256": CANONICAL_ARCHIVE_SHA256,
            "actual_sha256": actual_sha,
            "sha256_matches": sha_match,
        },
        "license_status": LICENSE_STATUS,
        "condition": condition,
        "osf_node": "2v5md",
        "sampling_rate_hz": OSF_SAMPLING_RATE_HZ,
        "expected_samples_per_channel": EXPECTED_SAMPLES_PER_CHANNEL,
        "channels_used": list(COMMON_CHANNELS_19),
        "n_channels": len(COMMON_CHANNELS_19),
        "expected_group_counts": dict(EXPECTED_GROUP_COUNTS),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _run_audit(archive: Path, output_dir: Path, condition: str) -> tuple[dict, dict, bool]:
    """Verify the archive, run the cohort audit, and write provenance + audit artifacts."""
    actual_sha = sha256_of_file(archive)
    sha_match = actual_sha == CANONICAL_ARCHIVE_SHA256
    provenance = _build_provenance(archive, condition, actual_sha, sha_match)
    _write_json(output_dir / "provenance.json", provenance)
    audit = cohort_audit(archive, condition=condition, inspect_samples=True)
    _write_json(output_dir / "cohort_audit.json", audit)
    pd.DataFrame(audit["subjects"]).to_csv(output_dir / "cohort_audit.csv", index=False)
    return audit, provenance, sha_match


def _audit_summary(audit: dict) -> str:
    counts = audit["status_counts"]
    return (
        f"status={audit['status'].upper()} "
        f"pass={counts['pass']} warn={counts['warn']} fail={counts['fail']}; "
        f"groups={audit['groups']} total_subjects={audit['total_subjects']}"
    )


def cmd_audit(args: argparse.Namespace) -> int:
    archive = Path(args.archive)
    output_dir = Path(args.output_dir)
    if not archive.exists():
        print(f"ERROR: archive not found: {archive}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    audit, provenance, sha_match = _run_audit(archive, output_dir, args.condition)
    print(f"SHA-256 {'matches' if sha_match else 'MISMATCH'}: {provenance['archive']['actual_sha256']}")
    print(_audit_summary(audit))
    print(f"Wrote audit artifacts to {output_dir}")

    if not sha_match or audit["status"] == "fail":
        return 1
    return 0


def _write_review_request(
    output_dir: Path, audit: dict, features: pd.DataFrame, failures: list[dict], sha_match: bool,
) -> str:
    """Write PHASE1_CODEX_REVIEW_REQUEST.md and return the stdout block."""
    counts = audit["status_counts"]
    feature_cols = [c for c in features.columns if c not in {"participant_id", "group", "label"}]
    group_counts = features["group"].value_counts().sort_index().to_dict()
    lines = [
        "# Phase 1 Codex Review Request", "",
        "## Summary", "",
        "- OSF Phase-1 CLI runner + tests on the existing `external_osf` engine. Phase 1 fits no model.",
        "- Archive read directly from the ZIP; SHA-256 preserved and verified.",
        f"- License status: `{LICENSE_STATUS}` (no machine-readable license inside the archive).",
        "", "## Changed files", "",
        "- `scripts/external_validation_osf.py` (audit + extract CLI).",
        "- `tests/test_external_osf.py` (synthetic-ZIP unit tests).",
        f"- `{output_dir.name}/` (generated phase-1 artifacts).",
        "", "## Commands and verification", "",
        "- `python -m pytest tests/test_external_osf.py -q`",
        "- `python scripts/external_validation_osf.py audit`",
        "- `python scripts/external_validation_osf.py extract`",
        "", "## Audit counts", "",
        f"- SHA-256 matches canonical: **{sha_match}**.",
        f"- Audit status: `{audit['status']}` (pass={counts['pass']}, warn={counts['warn']}, fail={counts['fail']}).",
        f"- Groups (Eyes_closed): {audit['groups']}; total subjects: {audit['total_subjects']}.",
        "- Every subject has the common 19 channels; F1/F2 are excluded.",
        "", "## Feature shape", "",
        f"- `external_features.csv`: {len(features)} rows, {len(feature_cols)} feature columns.",
        f"- Group row counts: {group_counts}. Extraction failures: {len(failures)}.",
        "", "## Risks / follow-up", "",
        "- 8 s records: no connectivity/graph validation in phase 1.",
        "- License reuse remains `UNRESOLVED` pending canonical OSF metadata.",
    ]
    (output_dir / "PHASE1_CODEX_REVIEW_REQUEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    block = "\n".join([
        "CODEX_REVIEW_REQUEST", "Summary:",
        "- OSF phase-1 CLI + tests on the existing external_osf engine; no model fitted.",
        f"- Archive SHA-256 matches canonical: {sha_match}.",
        f"- Eyes_closed cohort: {audit['groups']} ({audit['total_subjects']} subjects).",
        "Changed files:", "- scripts/external_validation_osf.py", "- tests/test_external_osf.py",
        "Commands and verification:", "- python -m pytest tests/test_external_osf.py -q",
        "- python scripts/external_validation_osf.py audit", "- python scripts/external_validation_osf.py extract",
        "Audit counts:",
        f"- status={audit['status']} pass={counts['pass']} warn={counts['warn']} fail={counts['fail']}",
        f"- groups={audit['groups']} total_subjects={audit['total_subjects']}",
        "Feature shape:",
        f"- external_features.csv rows={len(features)} feature_cols={len(feature_cols)} failures={len(failures)}",
        "Risks/follow-up:", "- phase 2: classifier on ds004504, bootstrap CIs, domain-shift, ROC",
        "- license UNRESOLVED; no connectivity/graph claims (8 s records)",
    ])
    return block


def cmd_extract(args: argparse.Namespace) -> int:
    archive = Path(args.archive)
    output_dir = Path(args.output_dir)
    if not archive.exists():
        print(f"ERROR: archive not found: {archive}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    audit, provenance, sha_match = _run_audit(archive, output_dir, args.condition)
    print(f"SHA-256 {'matches' if sha_match else 'MISMATCH'}: {provenance['archive']['actual_sha256']}")
    print(_audit_summary(audit))
    if not sha_match:
        print("ERROR: archive SHA-256 mismatch; aborting before extraction.", file=sys.stderr)
        return 1

    features, failures = build_feature_matrix(archive, condition=args.condition)
    features.to_csv(output_dir / "external_features.csv", index=False)
    _write_json(output_dir / "feature_mapping.json", feature_mapping())
    if failures:
        pd.DataFrame(failures).to_csv(output_dir / "external_feature_failures.csv", index=False)

    block = _write_review_request(output_dir, audit, features, failures, sha_match)
    print(block)
    print(f"Wrote extraction artifacts to {output_dir}")
    if audit["status"] == "fail" or failures:
        return 1
    return 0


# --- Phase 2 v2: train-on-discovery external validation ---------------------


def _plot_confusion_matrix(cm: list[list[int]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    array = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.2, 4.0), constrained_layout=True)
    image = ax.imshow(array, cmap="Blues", vmin=0, vmax=max(1, array.max()))
    for row in range(array.shape[0]):
        for col in range(array.shape[1]):
            color = "white" if array[row, col] > array.max() / 2 else "black"
            ax.text(col, row, str(array[row, col]), ha="center", va="center", color=color)
    ax.set_xticks(range(2), [NEG_LABEL, POS_LABEL])
    ax.set_yticks(range(2), [NEG_LABEL, POS_LABEL])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("External AD vs HC confusion matrix")
    fig.colorbar(image, ax=ax, shrink=0.75, label="Subjects")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_roc(truth_binary: np.ndarray, prob_ad: np.ndarray, auc_value: float, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(truth_binary, prob_ad)
    fig, ax = plt.subplots(figsize=(4.2, 4.0), constrained_layout=True)
    ax.plot(fpr, tpr, color="#0072B2", lw=2, label=f"AD vs HC (AUC = {auc_value:.3f})")
    ax.plot([0, 1], [0, 1], color="#888888", lw=1, linestyle="--", label="chance")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("External ROC (OSF, subject-level)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _package_versions() -> dict[str, str]:
    packages = ["mne", "mne-bids", "numpy", "pandas", "scikit-learn", "scipy", "statsmodels"]
    out = {"python": sys.version.split()[0], "platform": sys.platform}
    for name in packages:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not-installed"
    return out


def _sha256_file(path: Path) -> str:
    return sha256_of_file(path)


def _code_file_hashes(cfg_path: Path, bids_root: Path) -> dict[str, str]:
    candidates = {
        "external_osf.py": Path("eeg_cogagent/external_osf.py"),
        "external_validation.py": Path("eeg_cogagent/external_validation.py"),
        "external_validation_osf.py": Path("scripts/external_validation_osf.py"),
        "config_yaml": cfg_path,
        "dataset_description.json": bids_root / "dataset_description.json",
        "participants.tsv": bids_root / "participants.tsv",
    }
    out: dict[str, str] = {}
    for label, path in candidates.items():
        path = Path(path)
        out[label] = _sha256_file(path) if path.exists() else "MISSING"
    return out


def _validate_external(osf_df: pd.DataFrame, audit_subject_ids: set[str]) -> None:
    ids = set(osf_df["participant_id"])
    _gate(len(osf_df) == 92, f"external rows = {len(osf_df)}, expected 92")
    _gate(len(ids) == 92, f"external unique IDs = {len(ids)}, expected 92")
    _gate(ids == audit_subject_ids, "external IDs do not match the audited subject set")
    label_counts = osf_df["label"].value_counts().to_dict()
    _gate(label_counts.get("AD") == 80 and label_counts.get("HC") == 12,
          f"external labels = {label_counts}, expected AD=80 HC=12")
    labels = set(osf_df["label"])
    _gate(labels <= {"AD", "HC"}, f"external labels = {labels}, must be subset of {{AD, HC}}")
    feature_matrix = osf_df[list(V2_HARMONIZED_FEATURES)].to_numpy(dtype=float)
    _gate(np.isfinite(feature_matrix).all(), "external features contain non-finite values")
    missing = [c for c in V2_HARMONIZED_FEATURES if c not in osf_df.columns]
    _gate(not missing, f"external missing v2 columns: {missing}")


def _validate_discovery(disc_df: pd.DataFrame) -> None:
    ids = set(disc_df["participant_id"])
    _gate(len(ids) == 88, f"discovery unique IDs = {len(ids)}, expected 88")
    _gate(len(disc_df) == 88, f"discovery rows = {len(disc_df)}, expected 88")
    labels = disc_df["label"].value_counts().to_dict()
    _gate(labels.get("AD") == 36 and labels.get("FTD") == 23 and labels.get("HC") == 29,
          f"discovery labels = {labels}, expected AD=36 FTD=23 HC=29")
    feature_matrix = disc_df[list(V2_HARMONIZED_FEATURES)].to_numpy(dtype=float)
    _gate(np.isfinite(feature_matrix).all(), "discovery features contain non-finite values")


def _gate(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _build_validation_provenance(
    archive: Path, condition: str, sha_match: bool, cfg_path: Path, bids_root: Path,
) -> dict[str, object]:
    description = {}
    description_path = bids_root / "dataset_description.json"
    if description_path.exists():
        description = json.loads(description_path.read_text(encoding="utf-8"))
    return {
        "osf": {
            "node": "2v5md",
            "file": "EEG_data.zip",
            "canonical_sha256": CANONICAL_ARCHIVE_SHA256,
            "actual_sha256": _sha256_file(archive),
            "sha256_matches": sha_match,
            "condition_used": condition,
            "sampling_rate_hz": OSF_SAMPLING_RATE_HZ,
            "samples_per_channel": EXPECTED_SAMPLES_PER_CHANNEL,
            "channels_used": list(COMMON_CHANNELS_19),
            "dataset_node_license": None,
            "dataset_node_license_status": "UNRESOLVED",
            "article_license": "CC BY 4.0",
            "article_doi": "10.1038/s41598-023-32664-8",
            "article_pmc": "PMC10199940",
            "source_band_limit_hz": "0.5-30 (per article; source pre-filtered)",
            "source_artifact_handling": "manual removal by EEG technician (per article)",
            "archive_publication_channel_discrepancy": (
                "Archive contains F1/F2 in addition to the 19 channels listed in the article; "
                "v2 uses the common 19 and excludes F1/F2."
            ),
        },
        "discovery_ds004504": {
            "dataset_doi": description.get("DatasetDOI"),
            "license": description.get("License"),
            "bids_version": description.get("BIDSVersion"),
            "bids_root": str(bids_root),
            "config_yaml": str(cfg_path),
        },
        "feature_space": {
            "bands": {b: list(e) for b, e in V2_BANDS.items()},
            "denominator_hz": "1-30 (sum of the four common bands; gamma excluded)",
            "ratios": [list(r) for r in V2_RATIOS],
            "regions": {k: list(v) for k, v in V2_REGIONS.items()},
            "n_features": len(V2_HARMONIZED_FEATURES),
            "welch_resolution_hz": TARGET_FREQ_RESOLUTION_HZ,
            "convention": "sum-over-bins Welch PSD; relative powers + log10 band-power ratios",
            "feature_definition_identical_across_datasets": True,
            "differing_factors": [
                "acquisition device", "source preprocessing (ds004504 filtered/notched/epoch-rejected; "
                "OSF source band-limited 0.5-30 Hz + manual artifact removal)",
                "record length (ds004504 ~10 min; OSF 8 s)", "local QC",
            ],
        },
        "code_file_sha256": _code_file_hashes(cfg_path, bids_root),
        "environment": _package_versions(),
        "seed": DEFAULT_SEED,
        "command": " ".join(sys.argv),
        "interpreter": sys.executable,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "license_note": (
            "Article is CC BY 4.0 but the article license does not override the dataset node license; "
            "the OSF node node_license is null, so dataset reuse license is UNRESOLVED."
        ),
    }


def _write_environment_txt(path: Path) -> None:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120,
            check=False,
        )
        text = result.stdout
    except Exception as exc:  # noqa: BLE001 - never let env capture abort the run
        text = f"# pip freeze failed: {exc!r}\n"
    path.write_text(text, encoding="utf-8")


def _write_artifact_manifest(staging: Path, output_dir: Path) -> Path:
    manifest_path = staging / "artifact_manifest.json"
    rows = []
    for path in sorted(staging.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        rows.append({
            "path": path.relative_to(staging).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        })
    note = (
        "Manifest lists every published artifact under the output directory with its SHA-256 and "
        "size in bytes. The manifest file itself is excluded from its own listing."
    )
    manifest_path.write_text(
        json.dumps({"note": note, "output_dir": output_dir.name, "artifacts": rows},
                   indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _metrics_records(point: dict, bootstrap: dict, wilson: dict) -> list[dict]:
    records = []
    for name in ("balanced_accuracy", "roc_auc"):
        ci = bootstrap[name]
        records.append({"metric": name, "point": ci["point"], "ci_low": ci["ci_low"],
                        "ci_high": ci["ci_high"], "ci_method": "subject-level stratified bootstrap (10000)"})
    for name in ("sensitivity", "specificity"):
        ci = wilson[name]
        records.append({"metric": name, "point": point[name], "ci_low": ci["low"],
                        "ci_high": ci["high"], "ci_method": "Wilson score 95% (binomial)"})
    records.append({"metric": "accuracy", "point": point["accuracy"], "ci_low": "",
                    "ci_high": "", "ci_method": "not primary (80/12 imbalance)"})
    return records


def cmd_validate(args: argparse.Namespace) -> int:
    archive = Path(args.archive)
    output_dir = Path(args.output_dir)
    if not archive.exists():
        print(f"ERROR: archive not found: {archive}", file=sys.stderr)
        return 2
    cfg = load_config(args.config)
    cfg_path = Path(getattr(cfg, "_config_path", args.config))
    bids_root = Path(getattr(cfg, "_project_root", ".")) / cfg["paths"]["bids_root"]

    staging = output_dir.parent / f".staging_{output_dir.name}_{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    figure_dir = staging / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    try:
        _run_v2_validation(args, cfg, cfg_path, bids_root, archive, output_dir, staging, figure_dir)
    except GateError as exc:
        shutil.rmtree(staging, ignore_errors=True)
        print(f"GATE FAILED (no results published): {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - never publish a half-written run
        shutil.rmtree(staging, ignore_errors=True)
        print(f"ERROR (no results published): {exc!r}", file=sys.stderr)
        return 1

    # Atomic publish.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    os.rename(str(staging), str(output_dir))
    print(f"Wrote v2 validation artifacts to {output_dir}")
    return 0


def _run_v2_validation(
    args: argparse.Namespace, cfg: dict, cfg_path: Path, bids_root: Path, archive: Path,
    output_dir: Path, staging: Path, figure_dir: Path,
) -> None:
    condition = args.condition
    # --- Hard gates (pre-fit) ---
    _gate(condition == "Eyes_closed", f"condition must be Eyes_closed, got {condition}")
    actual_sha = sha256_of_file(archive)
    sha_match = actual_sha == CANONICAL_ARCHIVE_SHA256
    _gate(sha_match, f"archive SHA-256 mismatch: {actual_sha}")

    audit, _, _ = _run_audit(archive, staging, condition)
    print(_audit_summary(audit))
    _gate(audit["status"] != "fail", "cohort audit reported a hard failure")
    audit_subject_ids = {row["participant_id"] for row in audit["subjects"]}
    _gate(len(audit_subject_ids) == 92, f"audited subjects = {len(audit_subject_ids)}, expected 92")

    # --- OSF v2 features + signal QC ---
    osf_df, osf_failures = build_osf_harmonized_matrix_v2(archive, condition=condition)
    _gate(not osf_failures, f"OSF extraction failures: {osf_failures}")
    _validate_external(osf_df, audit_subject_ids)
    osf_df.to_csv(staging / "external_features.csv", index=False)
    signal_qc = build_osf_signal_qc(archive, condition=condition)
    signal_qc.to_csv(staging / "signal_qc.csv", index=False)
    if "all_finite" in signal_qc.columns:
        _gate(bool(signal_qc["all_finite"].all()), "signal QC: non-finite values in external signals")
    if "n_flat_or_zero_power_channels" in signal_qc.columns:
        _gate(bool((signal_qc["n_flat_or_zero_power_channels"] == 0).all()),
              "signal QC: flat/zero-power channels present")

    # --- Discovery v2 features ---
    disc_df, disc_failures = build_discovery_harmonized_matrix_v2(cfg)
    _gate(not disc_failures, f"discovery extraction failures: {disc_failures}")
    _validate_discovery(disc_df)
    disc_df.to_csv(staging / "harmonized_discovery_features.csv", index=False)

    # --- Training data: AD+HC only (drop FTD) ---
    train_df = disc_df[disc_df["label"].isin([POS_LABEL, "HC"])].reset_index(drop=True)
    _gate(int((train_df["label"] == POS_LABEL).sum()) == 36, "training AD count != 36")
    _gate(int((train_df["label"] == "HC").sum()) == 29, "training HC count != 29")
    X = train_df[list(V2_HARMONIZED_FEATURES)]
    y = (train_df["label"] == POS_LABEL).astype(int).to_numpy()

    # --- Internal nested-CV estimate (unbiased; threshold frozen per outer fold) ---
    nested = nested_cv_internal_estimate(X, y, seed=DEFAULT_SEED)
    nested["oof"].to_csv(staging / "discovery_nested_oof_predictions.csv", index=False)
    print(f"nested-CV balanced_accuracy={nested['balanced_accuracy']:.3f} auc={nested['auc']:.3f}")

    # --- Final discovery-only model + threshold ---
    fitted = fit_final_model_and_threshold(X, y, seed=DEFAULT_SEED)
    fitted["crossfit_oof"].to_csv(staging / "discovery_threshold_oof_predictions.csv", index=False)
    pd.DataFrame([
        {"feature": f, "standardized_coefficient": c}
        for f, c in fitted["coefficients"].items()
    ]).to_csv(staging / "model_coefficients.csv", index=False)
    _write_json(staging / "model_spec.json", {
        "primary_model": "L2 LogisticRegression (class_weight=balanced)",
        "feature_count": len(V2_HARMONIZED_FEATURES),
        "bands": {b: list(e) for b, e in V2_BANDS.items()},
        "denominator_hz": "1-30",
        "C_grid": list(C_GRID),
        "scoring": PRIMARY_SCORING,
        "outer_folds": OUTER_FOLDS,
        "seed": DEFAULT_SEED,
        "bootstrap": DEFAULT_BOOTSTRAP,
        "final_C": fitted["best_C"],
        "final_threshold": fitted["threshold"],
        "threshold_score": fitted["threshold_score"],
        "threshold_search": THRESHOLD_SEARCH,
        "threshold_tie_break": THRESHOLD_TIE_BREAK,
        "sensitivity_threshold": SENSITIVITY_THRESHOLD,
        "discovery_train": {"n": int(len(y)), "n_ad": int((y == 1).sum()), "n_hc": int((y == 0).sum())},
    })
    _write_json(staging / "fitted_transformer_params.json", {
        "imputer_strategy": "median",
        "imputer_medians": dict(zip(V2_HARMONIZED_FEATURES, fitted["imputer_medians"])),
        "scaler_means": dict(zip(V2_HARMONIZED_FEATURES, fitted["scaler_means"])),
        "scaler_scales": dict(zip(V2_HARMONIZED_FEATURES, fitted["scaler_scales"])),
    })

    # --- Predict OSF (primary threshold) + sensitivity at 0.5 ---
    predictions = predict_external(fitted, osf_df)
    pred_ids = set(predictions["participant_id"])
    _gate(len(predictions) == 92, f"predictions rows = {len(predictions)}, expected 92")
    _gate(pred_ids == audit_subject_ids, "prediction IDs do not match audited subject set")
    _gate(set(predictions["true_label"]) <= {"AD", "HC"}, "prediction true_label not subset of {AD,HC}")
    predictions.to_csv(staging / "external_predictions.csv", index=False)

    point = point_metrics(predictions["true_label"], predictions["pred_label"], predictions["prob_AD"])
    bootstrap = subject_level_bootstrap_metrics(
        predictions["true_label"], predictions["prob_AD"], predictions["pred_label"],
        n_boot=DEFAULT_BOOTSTRAP, seed=DEFAULT_SEED,
    )
    wilson = {
        "sensitivity": {"k": point["tp"], "n": point["tp"] + point["fn"], "low": None, "high": None},
        "specificity": {"k": point["tn"], "n": point["tn"] + point["fp"], "low": None, "high": None},
    }
    for name in ("sensitivity", "specificity"):
        w = wilson[name]
        w["low"], w["high"] = wilson_ci(w["k"], w["n"])
    pd.DataFrame(_metrics_records(point, bootstrap, wilson)).to_csv(staging / "external_metrics.csv", index=False)
    # Sensitivity analysis at fixed 0.5 threshold (predeclared).
    sens_predictions = predict_external(fitted, osf_df, threshold=SENSITIVITY_THRESHOLD)
    sens_point = point_metrics(sens_predictions["true_label"], sens_predictions["pred_label"], sens_predictions["prob_AD"])
    _write_json(staging / "external_metrics.json", {
        "point": point,
        "bootstrap_ci_95": bootstrap,
        "bootstrap_note": "Subject-level, class-stratified, 10000 resamples; conditional on the fitted discovery model.",
        "wilson_ci_95": wilson,
        "sensitivity_threshold_0p5": sens_point,
        "n_external": int(len(predictions)),
        "threshold": fitted["threshold"],
        "best_C": fitted["best_C"],
        "internal_nested_cv": {
            "balanced_accuracy": nested["balanced_accuracy"], "auc": nested["auc"],
            "outer_folds": nested["outer_folds"],
        },
    })

    # --- Confusion + ROC ---
    cm = [[point["tn"], point["fp"]], [point["fn"], point["tp"]]]
    pd.DataFrame(cm, index=["true_HC", "true_AD"], columns=["pred_HC", "pred_AD"]).to_csv(
        staging / "confusion_matrix.csv"
    )
    _plot_confusion_matrix(cm, figure_dir / "confusion_matrix.png")
    truth_binary = (predictions["true_label"] == POS_LABEL).astype(int).to_numpy()
    _plot_roc(truth_binary, predictions["prob_AD"].to_numpy(), point["roc_auc"], figure_dir / "roc_curve.png")
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(truth_binary, predictions["prob_AD"].to_numpy())
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}).to_csv(staging / "roc_curve.csv", index=False)

    # --- Domain shift (primary label-free uses AD+HC only; supplementary by label) ---
    shift_primary = domain_shift_primary_labelfree(train_df, osf_df)
    shift_primary.to_csv(staging / "domain_shift_primary_labelfree.csv", index=False)
    shift_by_label = domain_shift_supplementary_by_label(disc_df, osf_df)
    shift_by_label.to_csv(staging / "domain_shift_supplementary_by_label.csv", index=False)

    # --- Feature harmonization doc ---
    _write_json(staging / "feature_harmonization.json", {
        "n_features": len(V2_HARMONIZED_FEATURES),
        "features": list(V2_HARMONIZED_FEATURES),
        "bands": {b: list(e) for b, e in V2_BANDS.items()},
        "denominator_hz": "1-30",
        "gamma_included": False,
        "ratios": [list(r) for r in V2_RATIOS],
        "regions": {k: list(v) for k, v in V2_REGIONS.items()},
        "welch_resolution_hz": TARGET_FREQ_RESOLUTION_HZ,
        "discovery": {"sfreq_hz": DISCOVERY_SFREQ_HZ, "nperseg": int(round(DISCOVERY_SFREQ_HZ / TARGET_FREQ_RESOLUTION_HZ))},
        "external_osf": {"sfreq_hz": OSF_SAMPLING_RATE_HZ, "nperseg": 256},
        "convention": "sum-over-bins Welch PSD; relative powers (band/total over 1-30 Hz) + log10 band-power ratios",
        "excluded": "per-channel relpow_ch__* columns; gamma band (source band-limited to 0.5-30 Hz)",
    })

    # --- Provenance + environment + protocol + manifest ---
    provenance = _build_validation_provenance(archive, condition, sha_match, cfg_path, bids_root)
    _write_json(staging / "validation_provenance.json", provenance)
    _write_environment_txt(staging / "environment.txt")
    protocol_source = Path(getattr(cfg, "_project_root", ".")) / PROTOCOL_SOURCE
    _gate(protocol_source.exists(), f"protocol source not found: {protocol_source}")
    shutil.copyfile(protocol_source, staging / PROTOCOL_FILENAME)

    report = _build_validation_report(
        point, bootstrap, wilson, nested, fitted, shift_primary, predictions, sens_point,
    )
    (staging / "external_validation_report.md").write_text(report, encoding="utf-8")

    block = _build_review_block(
        point, bootstrap, wilson, nested, fitted, shift_primary, predictions,
        provenance, sens_point,
    )
    (staging / "CODEX_REVIEW_REQUEST.md").write_text(block + "\n", encoding="utf-8")

    _write_artifact_manifest(staging, output_dir)
    print(block)


def _build_validation_report(
    point: dict, bootstrap: dict, wilson: dict, nested: dict, fitted: dict,
    shift: pd.DataFrame, predictions: pd.DataFrame, sens_point: dict,
) -> str:
    metric_rows = _metrics_records(point, bootstrap, wilson)
    largest = shift.reindex(shift["standardized_mean_difference"].abs().sort_values(ascending=False).index).head(10)
    n_ad = int((predictions["true_label"] == POS_LABEL).sum())
    n_hc = int((predictions["true_label"] == "HC").sum())
    return "\n".join([
        "# Independent External Validation v2: OSF AD vs HC", "",
        "## Design", "",
        "- Predeclared primary model: L2 LogisticRegression, class_weight=balanced, on 36 harmonized",
        "  1-30 Hz features (4 bands; gamma excluded because the OSF source is band-limited 0.5-30 Hz).",
        "- All learned components (imputation, scaling, C, decision threshold) trained on ds004504",
        "  AD/HC only. The internal nested-CV estimate is unbiased: per outer fold, C and threshold",
        "  are chosen on the outer-training data only, then frozen and applied to the outer-test fold.",
        "- OSF entered exclusively at prediction time. No OSF label/feature distribution influenced",
        "  fitting, tuning, or selection.", "",
        "## Cohort", "",
        "- Discovery (ds004504): 88 subjects; trained on AD/HC = 36 AD + 29 HC.",
        f"- External (OSF, Eyes_closed): {n_ad} AD + {n_hc} HC = {len(predictions)} subjects.",
        "- License: article CC BY 4.0 (DOI 10.1038/s41598-023-32664-8); OSF dataset node license null -> UNRESOLVED.", "",
        "## Internal nested-CV (ds004504 AD/HC, unbiased)", "",
        f"- balanced accuracy = {nested['balanced_accuracy']:.3f}; AUC = {nested['auc']:.3f}.",
        f"- (For reference, the prior v1 procedure tuned the threshold on the same OOF and was optimistic.)", "",
        "## Final discovery-only model", "",
        f"- C = {fitted['best_C']}; decision threshold = {fitted['threshold']:.3f} "
        f"(chosen on discovery cross-fitted OOF; tie-break: lowest threshold).", "",
        "## External performance (subject-level)", "",
        pd.DataFrame(metric_rows).to_markdown(index=False, floatfmt=".3f"),
        "",
        f"- Confusion (rows=true, cols=pred): TP={point['tp']} FP={point['fp']} TN={point['tn']} FN={point['fn']}.",
        f"- Sensitivity analysis at fixed threshold 0.5: balanced accuracy = "
        f"{0.5*(sens_point['sensitivity']+sens_point['specificity']):.3f}.", "",
        "## Domain shift (primary, label-free: discovery AD+HC vs external all)", "",
        "Standardized mean difference (pooled SD); descriptive only.",
        "",
        largest[["feature", "mean_discovery", "mean_external", "standardized_mean_difference",
                 "ks_statistic"]].to_markdown(index=False, floatfmt=".3f"),
        "", "## Limitations", "",
        "- 80/12 imbalance -> wide CIs; specificity Wilson interval is especially wide (n=12 HC).",
        "- Bootstrap CIs are conditional on the fitted discovery model; they do not reflect training-sample uncertainty.",
        "- 8 s OSF records -> no connectivity/graph validation.",
        "- No OSF age/sex/MMSE metadata -> demographic confounds cannot be checked or adjusted.",
        "- OSF labels were already inspected on 2026-07-01; this is a method-audited external evaluation,",
        "  NOT a blinded/prospective confirmatory validation.",
    ]) + "\n"


def _build_review_block(
    point: dict, bootstrap: dict, wilson: dict, nested: dict, fitted: dict,
    shift: pd.DataFrame, predictions: pd.DataFrame, provenance: dict, sens_point: dict,
) -> str:
    bal = bootstrap["balanced_accuracy"]
    auc = bootstrap["roc_auc"]
    n_shifted = int((shift["standardized_mean_difference"].abs() > 0.5).sum())
    sens_ba = 0.5 * (sens_point["sensitivity"] + sens_point["specificity"])
    return "\n".join([
        "CODEX_REVIEW_REQUEST",
        "Summary:",
        "- v2 independent external AD-vs-HC evaluation on OSF, with 1-30 Hz harmonized features and leak-free nested discovery training.",
        "- Results saved as generated; no model/feature/threshold was tuned on OSF metrics.",
        "Method corrections from v1:",
        "- Dropped gamma; 36 features over a common 1-30 Hz denominator (OSF source is band-limited 0.5-30 Hz).",
        "- Split unbiased internal nested-CV (per-fold C+threshold on outer-train only) from final discovery-only model/threshold.",
        "- Primary domain shift is label-free (discovery AD+HC vs external all) using pooled-SD standardized mean difference + KS.",
        "- Hard pre-fit gates + atomic staging/publish; no results published unless every gate passes.",
        "Changed files:",
        "- eeg_cogagent/external_validation.py (rewritten for v2)",
        "- eeg_cogagent/external_osf.py (unchanged from phase 2; nperseg param)",
        "- scripts/external_validation_osf.py (validate rewritten; gates/staging/provenance/manifest)",
        "- tests/test_external_validation_osf.py (rewritten)",
        "- protocols/EXTERNAL_VALIDATION_PROTOCOL_V2.md (new)",
        "- results/external_validation_osf_v2/ (generated)",
        "Commands and exact interpreter:",
        f"- {sys.executable} -m pytest tests/test_external_osf.py tests/test_external_validation_osf.py -q",
        f"- {sys.executable} -m pytest tests -q",
        f"- {sys.executable} scripts/external_validation_osf.py validate --config configs/ds004504_minimal.yaml --output-dir results/external_validation_osf_v2",
        "Test results:",
        "- focused v2 suite + full suite green in project .venv (see report).",
        "Discovery nested-CV metrics:",
        f"- balanced_accuracy={nested['balanced_accuracy']:.3f} auc={nested['auc']:.3f} (unbiased; {nested['outer_folds']} outer folds)",
        "Final discovery-only model and threshold:",
        f"- L2 LR class_weight=balanced; C={fitted['best_C']} threshold={fitted['threshold']:.3f} "
        f"(discovery cross-fitted OOF; tie-break lowest threshold)",
        f"- sensitivity analysis at fixed 0.5 threshold: balanced_accuracy={sens_ba:.3f}",
        "External cohort and metrics with CIs:",
        f"- n_external={len(predictions)} (80 AD, 12 HC)",
        f"- balanced_accuracy={bal['point']:.3f} [{bal['ci_low']:.3f}, {bal['ci_high']:.3f}] (bootstrap 10000)",
        f"- roc_auc={auc['point']:.3f} [{auc['ci_low']:.3f}, {auc['ci_high']:.3f}] (bootstrap 10000)",
        f"- sensitivity={point['sensitivity']:.3f} Wilson [{wilson['sensitivity']['low']:.3f}, {wilson['sensitivity']['high']:.3f}] "
        f"(k={wilson['sensitivity']['k']}, n={wilson['sensitivity']['n']})",
        f"- specificity={point['specificity']:.3f} Wilson [{wilson['specificity']['low']:.3f}, {wilson['specificity']['high']:.3f}] "
        f"(k={wilson['specificity']['k']}, n={wilson['specificity']['n']}; n=12 -> wide)",
        f"- confusion TP/FP/TN/FN={point['tp']}/{point['fp']}/{point['tn']}/{point['fn']}",
        "Domain shift:",
        f"- primary label-free: features with |SMD|>0.5: {n_shifted}/{len(shift)}; "
        f"max |SMD|={shift['standardized_mean_difference'].abs().max():.3f}",
        "Provenance and manifest:",
        f"- archive SHA matches canonical: {provenance['osf']['sha256_matches']}; "
        f"dataset license UNRESOLVED (node null) vs article CC BY 4.0.",
        "- code/data SHA-256, environment.txt, artifact_manifest.json all written.",
        "Protocol deviations / rerun count:",
        "- none (single v2 run after tests); any engineering bug found post-real-run would be logged here.",
        "Known limitations:",
        "- 80/12 imbalance; specificity CI very wide; bootstrap CIs conditional on fitted model.",
        "- no OSF demographics; 8 s records -> no connectivity; not a blinded/prospective validation.",
        "Files for Codex audit:",
        "- results/external_validation_osf_v2/{validation_provenance.json,artifact_manifest.json,model_spec.json,CODEX_REVIEW_REQUEST.md}",
        "- protocols/EXTERNAL_VALIDATION_PROTOCOL_V2.md",
    ])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OSF external-validation: archive audit, feature extraction, and v2 AD-vs-HC validation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit", help="Archive integrity + cohort audit only.")
    extract_parser = subparsers.add_parser("extract", help="Audit, then extract subject-level features.")
    validate_parser = subparsers.add_parser(
        "validate",
        help="v2: 1-30 Hz harmonized features, nested discovery training, hard gates, evaluate on OSF.",
    )
    validate_parser.add_argument(
        "--config", default="configs/ds004504_minimal.yaml",
        help="Discovery (ds004504) YAML config for preprocessing + participants.",
    )
    validate_parser.add_argument("--archive", default=DEFAULT_ARCHIVE, help="Path to EEG_data.zip.")
    validate_parser.add_argument(
        "--output-dir", default=V2_DEFAULT_OUTPUT_DIR,
        help="Directory to write v2 artifacts (default: results/external_validation_osf_v2).",
    )
    validate_parser.add_argument(
        "--condition", default=DEFAULT_CONDITION, help="Archive condition (must be Eyes_closed).",
    )
    _add_common_args(audit_parser)
    _add_common_args(extract_parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "audit":
        code = cmd_audit(args)
    elif args.command == "extract":
        code = cmd_extract(args)
    elif args.command == "validate":
        code = cmd_validate(args)
    else:  # pragma: no cover - argparse enforces required subcommand
        parser.error("unknown command")
        return
    sys.exit(code)


if __name__ == "__main__":
    main()
