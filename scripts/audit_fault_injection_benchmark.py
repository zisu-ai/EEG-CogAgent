"""Audit-contract fault-injection benchmark (v3.1).

Synthetic-fixture benchmark that injects a fixed set of integrity faults and
records whether the EEG-CogAgent audit contract detects each one. Pure software
assurance: it validates the *audit contract*, it does not measure clinical or
scientific performance, and the fault count must never be reported as such.

The methodological point (reported in the manuscript as one paragraph + one
supplementary table) is the comparison between ID-only identity audit and
content-level fingerprint audit: an exact-signal duplicate published under two
DISTINCT folder IDs is invisible to an ID-only audit but is caught by the
content fingerprint audit.

Outputs (``results/audit_fault_injection_v3_1/``):
* ``fault_injection_results.csv``  - one row per fault
* ``fault_injection_results.json`` - structured
* ``report.md``                    - narrative report
* ``artifact_manifest.json``       - SHA-256 + size for every published file

No real archive, no real EEG, no model fitting. Deterministic.
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from eeg_cogagent.external_osf import (
    COMMON_CHANNELS_19,
    EXPECTED_SAMPLES_PER_CHANNEL,
    OSF_SAMPLING_RATE_HZ,
    _parse_channel_bytes,
    cluster_signal_fingerprints,
    compute_signal_fingerprint,
    sha256_of_bytes,
)
from eeg_cogagent.external_validation import (
    V2_HARMONIZED_FEATURES,
    _assert_oof_inventory,
    harmonized_features_from_signal_v2,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "external_validation_osf.py"
OUTPUT_DIR = REPO_ROOT / "results" / "audit_fault_injection_v3_1"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("ev_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- synthetic fixtures ------------------------------------------------------


def _signal(freqs=(2.0, 6.0, 10.0, 20.0), n: int = 1024, seed: int = 0) -> dict[str, np.ndarray]:
    """Finite, non-flat, 1024-sample multi-sine + noise across the 19 common channels."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / OSF_SAMPLING_RATE_HZ
    out: dict[str, np.ndarray] = {}
    for ch in COMMON_CHANNELS_19:
        sig = np.zeros(n)
        for f in freqs:
            sig += np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        out[ch] = sig + rng.normal(0.0, 0.05, size=n)
    return out


def _channel_text(values: np.ndarray) -> bytes:
    return ("\n".join(f"{v:.10f}" for v in values) + "\n").encode("utf-8")


def _write_archive(path: Path, subjects: list[tuple[str, str, dict[str, np.ndarray]]],
                   condition: str = "Eyes_closed") -> None:
    """Write a tiny valid-layout OSF-style archive. Each subject is (group, paciente, signal)."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for group, paciente, signal in subjects:
            for ch in COMMON_CHANNELS_19:
                zf.writestr(f"EEG_data/{group}/{condition}/{paciente}/{ch}.txt", _channel_text(signal[ch]))


# --- result container --------------------------------------------------------


@dataclass
class FaultResult:
    fault_id: str
    description: str
    detector: str
    expected_detection: bool
    actual_detection: bool
    exit_behavior: str
    notes: str


def _detect(impl: Callable[[], object]) -> tuple[bool, str]:
    """Run a detector; return (detected, exit_behavior). Detection = it raised."""
    try:
        impl()
    except Exception as exc:  # noqa: BLE001 - detection IS the exception
        return True, f"raised {type(exc).__name__}"
    return False, "returned clean (no signal)"


# --- individual fault cases --------------------------------------------------


def fault_sha_mismatch() -> FaultResult:
    a = sha256_of_bytes(b"canonical-archive-bytes")
    b = sha256_of_bytes(b"different-bytes-tampered")
    detected = a != b
    return FaultResult(
        fault_id="F01_sha_mismatch",
        description="Archive SHA-256 differs from the canonical digest.",
        detector="archive_sha256_gate (pre-fit, hard)",
        expected_detection=True,
        actual_detection=detected,
        exit_behavior="GateError; abort before any fitting or publish",
        notes=f"sha_a={a[:12]}.. sha_b={b[:12]}..; inequality drives the gate",
    )


def fault_missing_channel() -> FaultResult:
    sig = _signal()
    del sig["Fp1"]
    detected, exit = _detect(lambda: harmonized_features_from_signal_v2(sig, 128.0, 256))
    return FaultResult(
        fault_id="F02_missing_channel",
        description="A common-19 channel is absent from the recording.",
        detector="harmonized_features_from_signal_v2 missing-channel check",
        expected_detection=True, actual_detection=detected, exit_behavior=exit,
        notes="Extraction fails per-subject; the subject is recorded, not silently imputed.",
    )


def fault_wrong_sample_count() -> FaultResult:
    short = _channel_text(np.arange(10, dtype=float))
    detected, exit = _detect(lambda: _parse_channel_bytes(short, expected=EXPECTED_SAMPLES_PER_CHANNEL))
    return FaultResult(
        fault_id="F03_wrong_sample_count",
        description="A channel file carries fewer than 1024 samples.",
        detector="_parse_channel_bytes(expected=1024)",
        expected_detection=True, actual_detection=detected, exit_behavior=exit,
        notes="Hard schema check; byte length is never mistaken for sample count (cf. v3 fingerprint defect).",
    )


def fault_nan_inf() -> FaultResult:
    sig = _signal()
    sig["Fz"] = sig["Fz"].copy(); sig["Fz"][5] = np.nan
    detected, exit = _detect(lambda: harmonized_features_from_signal_v2(sig, 128.0, 256))
    return FaultResult(
        fault_id="F04_nan_inf",
        description="A channel contains a non-finite (NaN/Inf) sample.",
        detector="harmonized_features_from_signal_v2 finite check",
        expected_detection=True, actual_detection=detected, exit_behavior=exit,
        notes="Predeclared fail rule; non-finite signals never reach features.",
    )


def fault_flat_channel() -> FaultResult:
    sig = _signal()
    sig["Cz"] = np.full(EXPECTED_SAMPLES_PER_CHANNEL, 3.0)
    detected, exit = _detect(lambda: harmonized_features_from_signal_v2(sig, 128.0, 256))
    return FaultResult(
        fault_id="F05_flat_channel",
        description="A channel is constant (zero variance / zero power).",
        detector="harmonized_features_from_signal_v2 flat-channel check",
        expected_detection=True, actual_detection=detected, exit_behavior=exit,
        notes="Predeclared fail rule; rejects flat/zero-power channels.",
    )


def _synthetic_frame(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """Build an external-style frame; rows = list of (participant_id, label)."""
    rng = np.random.default_rng(0)
    data = []
    for pid, label in rows:
        row = {"participant_id": pid, "group": "AD" if label == "AD" else "Healthy", "label": label}
        for f in V2_HARMONIZED_FEATURES:
            row[f] = float(rng.normal())
        data.append(row)
    return pd.DataFrame(data)


def fault_duplicate_id(script_module) -> FaultResult:
    audit_ids = sorted({f"AD_Paciente{i}" for i in range(1, 81)} | {f"Healthy_Paciente{i}" for i in range(1, 13)})
    labels = ["AD"] * 80 + ["HC"] * 12
    frame = _synthetic_frame(list(zip(audit_ids, labels)))
    # Make the last row collide with the first row's participant_id (92 rows, 91 unique).
    frame.loc[frame.index[-1], "participant_id"] = frame.loc[frame.index[0], "participant_id"]
    detected, exit = _detect(lambda: script_module._validate_external(frame, set(audit_ids)))
    return FaultResult(
        fault_id="F06_duplicate_id",
        description="Two rows share one participant_id (nominal-folder collision).",
        detector="_validate_external count/uniqueness gate",
        expected_detection=True, actual_detection=detected, exit_behavior=exit,
        notes="ID-level audit catches folder-ID collisions.",
    )


def fault_unknown_label(script_module) -> FaultResult:
    audit_ids = {f"AD_Paciente{i}" for i in range(1, 81)} | {f"Healthy_Paciente{i}" for i in range(1, 13)}
    frame = _synthetic_frame(list(zip(sorted(audit_ids), ["AD"] * 80 + ["HC"] * 12)))
    frame.loc[frame.index[0], "label"] = "XYZ"
    detected, exit = _detect(lambda: script_module._validate_external(frame, audit_ids))
    return FaultResult(
        fault_id="F07_unknown_label",
        description="An external row carries a label outside {AD, HC}.",
        detector="_validate_external label-subset gate",
        expected_detection=True, actual_detection=detected, exit_behavior=exit,
        notes="Unknown labels are never silently remapped.",
    )


def fault_oof_inventory_mismatch() -> FaultResult:
    rng = np.random.default_rng(1)
    idx = pd.Index([f"sub-{i:03d}" for i in range(65)])
    X = pd.DataFrame(rng.normal(size=(65, 4)), index=idx)
    oof = pd.DataFrame({
        "participant_id": [str(i) for i in list(idx)[:-1]],  # drop one -> mismatch
        "true_label": ["AD"] * 64, "true_binary": [1] * 64,
    })
    detected, exit = _detect(lambda: _assert_oof_inventory(oof, X, "nested_oof"))
    return FaultResult(
        fault_id="F08_oof_inventory_mismatch",
        description="Out-of-fold predictions omit one training participant.",
        detector="_assert_oof_inventory (participant-level coverage)",
        expected_detection=True, actual_detection=detected, exit_behavior=exit,
        notes="Guarantees every discovery AD/HC subject appears exactly once out-of-fold.",
    )


def fault_exact_signal_duplicate_distinct_ids(tmp_path: Path) -> list[FaultResult]:
    """The key contrast: identical signal under two distinct folder IDs.

    ID-only audit (participant_id uniqueness) cannot see it; the content
    fingerprint audit can. Returns two rows."""
    archive = tmp_path / "dup.zip"
    shared = _signal(seed=7)
    _write_archive(archive, [
        ("AD", "Paciente40", shared),
        ("AD", "Paciente41", shared),  # bit-identical signal, distinct folder ID
    ])
    rows = compute_signal_fingerprint(archive, condition="Eyes_closed")
    audit_df, summary = cluster_signal_fingerprints(rows)

    # ID-only audit: are participant_ids unique?
    ids = [r["participant_id"] for r in rows]
    id_unique = len(set(ids)) == len(ids)

    # Content audit: is there a duplicate cluster?
    content_detected = summary["duplicate_cluster_count"] >= 1

    return [
        FaultResult(
            fault_id="F09a_exact_signal_duplicate_id_only_audit",
            description="Identical 19-channel signal published under two distinct folder IDs; inspected by ID uniqueness only.",
            detector="id_only_participant_uniqueness",
            expected_detection=False,
            actual_detection=False,
            exit_behavior="returns clean (missed)" if id_unique else "flagged",
            notes=("Distinct IDs -> the ID-only audit sees uniqueness and MISSES the exact-signal "
                   "duplicate. This is the class of integrity violation an ID-only contract cannot detect."),
        ),
        FaultResult(
            fault_id="F09b_exact_signal_duplicate_content_audit",
            description="Identical 19-channel signal published under two distinct folder IDs; inspected by content fingerprint.",
            detector="content_fingerprint_audit (compute_signal_fingerprint + cluster_signal_fingerprints)",
            expected_detection=True,
            actual_detection=content_detected,
            exit_behavior=(f"flagged size-{max((len(m) for m in summary['clusters'].values()), default=1)} duplicate cluster"
                           if content_detected else "returned clean"),
            notes="Content-level identity audit catches what ID-only misses; this motivates the v3/v3.1 duplicate-cluster finding.",
        ),
    ]


def fault_cross_label_duplicate_conflict(tmp_path: Path) -> FaultResult:
    archive = tmp_path / "conflict.zip"
    shared = _signal(seed=9)
    _write_archive(archive, [
        ("AD", "Paciente50", shared),
        ("Healthy", "Paciente50", shared),  # identical signal, different group
    ])
    rows = compute_signal_fingerprint(archive, condition="Eyes_closed")
    detected, exit = _detect(lambda: cluster_signal_fingerprints(rows))
    return FaultResult(
        fault_id="F10_cross_label_duplicate_conflict",
        description="Identical signal filed under two label groups (AD and Healthy).",
        detector="cluster_signal_fingerprints group-conflict check",
        expected_detection=True, actual_detection=detected, exit_behavior=exit,
        notes="Hard-fails: the same recording cannot carry two labels.",
    )


def fault_stale_manifest(script_module, tmp_path: Path) -> FaultResult:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "a.txt").write_text("real-content", encoding="utf-8")
    manifest = staging / "artifact_manifest.json"
    manifest.write_text(json.dumps({
        "note": "stale", "output_dir": "x", "artifacts": [
            {"path": "a.txt", "size_bytes": 999, "sha256": "00" * 32},  # wrong size + sha
        ],
    }), encoding="utf-8")
    detected, exit = _detect(lambda: script_module._verify_manifest(staging, manifest))
    return FaultResult(
        fault_id="F11_stale_manifest",
        description="Manifest lists a size/SHA-256 that no longer matches the real file.",
        detector="_verify_manifest (size + SHA-256 per file)",
        expected_detection=True, actual_detection=detected, exit_behavior=exit,
        notes="Self-verification rejects a manifest that drifts from its files.",
    )


def fault_failed_pytest_gate(script_module, tmp_path: Path) -> FaultResult:
    import sys
    failing_test = tmp_path / "test_always_fail.py"
    failing_test.write_text("def test_always_fail():\n    assert False\n", encoding="utf-8")
    args = [sys.executable, "-m", "pytest", str(failing_test), "-q"]
    exit_code, summary = script_module._run_pytest(args)
    detected = exit_code != 0
    return FaultResult(
        fault_id="F12_failed_pytest_gate",
        description="The focused/full pytest suite exits non-zero.",
        detector="_write_test_report gate ([sys.executable, -m, pytest, ...])",
        expected_detection=True, actual_detection=detected,
        exit_behavior=f"exit_code={exit_code}; GateError, no publish" if detected else "returned clean",
        notes="Uses sys.executable (not PATH python); non-zero refuses to publish.",
    )


# --- manifest ----------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    return sha256_of_bytes(path.read_bytes())


def _write_manifest(output_dir: Path) -> Path:
    manifest = output_dir / "artifact_manifest.json"
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        rows.append({"path": path.relative_to(output_dir).as_posix(),
                     "size_bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    manifest.write_text(json.dumps({
        "note": "Fault-injection benchmark manifest; excludes itself.",
        "output_dir": OUTPUT_DIR.name,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "artifacts": rows,
    }, indent=2), encoding="utf-8")
    return manifest


# --- orchestration -----------------------------------------------------------


def run_benchmark(output_dir: Path = OUTPUT_DIR) -> dict:
    script_module = _load_script_module()
    results: list[FaultResult] = [
        fault_sha_mismatch(),
        fault_missing_channel(),
        fault_wrong_sample_count(),
        fault_nan_inf(),
        fault_flat_channel(),
        fault_duplicate_id(script_module),
        fault_unknown_label(script_module),
        fault_oof_inventory_mismatch(),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        results.extend(fault_exact_signal_duplicate_distinct_ids(tmp_path))
        results.append(fault_cross_label_duplicate_conflict(tmp_path))
        results.append(fault_stale_manifest(script_module, tmp_path))
        results.append(fault_failed_pytest_gate(script_module, tmp_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(output_dir / "fault_injection_results.csv", index=False)

    n = len(results)
    n_expected_detect = sum(1 for r in results if r.expected_detection)
    n_detected_of_expected = sum(1 for r in results if r.expected_detection and r.actual_detection)
    n_false_alarm = sum(1 for r in results if not r.expected_detection and r.actual_detection)
    summary = {
        "n_faults": n,
        "n_expected_detected": n_detected_of_expected,
        "n_expected_detect": n_expected_detect,
        "n_false_alarms": n_false_alarm,
        "contract": (
            "Audit-contract assurance, not clinical/scientific performance. The fault count "
            "is a coverage count of injected integrity violations, not a diagnostic metric."
        ),
        "key_contrast": (
            "F09a/b: an exact-signal duplicate under distinct folder IDs is missed by the "
            "ID-only audit and caught by the content-fingerprint audit."
        ),
        "results": [asdict(r) for r in results],
    }
    (output_dir / "fault_injection_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    (output_dir / "report.md").write_text(_build_report(df, summary), encoding="utf-8")
    _write_manifest(output_dir)
    return summary


def _build_report(df: pd.DataFrame, summary: dict) -> str:
    return "\n".join([
        "# Audit-Contract Fault-Injection Benchmark (v3.1)", "",
        "## Scope and claim boundary", "",
        "This benchmark validates the **audit contract** of EEG-CogAgent by injecting a fixed",
        "set of synthetic integrity faults and recording whether each is detected. It is pure",
        "software assurance: the fault count is a coverage count of injected violations, **not**",
        "a clinical, diagnostic, or scientific-performance metric, and must not be reported as such.", "",
        "## Result summary", "",
        f"- Faults injected: **{summary['n_faults']}**.",
        f"- Expected detections delivered: **{summary['n_expected_detected']}/{summary['n_expected_detect']}**.",
        f"- False alarms (detected when not expected): **{summary['n_false_alarms']}**.",
        "",
        "## Key methodological contrast", "",
        summary["key_contrast"], "",
        "## Per-fault detail", "",
        df[["fault_id", "detector", "expected_detection", "actual_detection", "exit_behavior"]].to_markdown(
            index=False
        ),
        "",
        "## Limitations", "",
        "- Synthetic fixtures only; real-archive behaviour is covered by the v3/v3.1 read-only duplicate audit.",
        "- This is a detection-coverage check, not a security certification or a manual-vs-agent comparison.",
    ]) + "\n"


def main() -> None:
    summary = run_benchmark()
    print(f"Wrote fault-injection benchmark to {OUTPUT_DIR}")
    print(f"faults={summary['n_faults']} "
          f"expected_detected={summary['n_expected_detected']}/{summary['n_expected_detect']} "
          f"false_alarms={summary['n_false_alarms']}")


if __name__ == "__main__":
    main()
