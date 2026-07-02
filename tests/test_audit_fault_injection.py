"""Focused tests for the audit-contract fault-injection benchmark (v3.1).

The benchmark is a software-assurance coverage check, NOT a clinical or
scientific-performance metric. These tests verify (a) it runs and emits the
expected artifacts, (b) every expected-detection fault is actually detected,
(c) the key contrast holds — an exact-signal duplicate under distinct IDs is
missed by the ID-only audit and caught by the content-fingerprint audit, and
(d) the claim-boundary disclaimer is present so the fault count is not mis-read
as a diagnostic metric.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import importlib.util
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_PATH = REPO_ROOT / "scripts" / "audit_fault_injection_benchmark.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location("fib", BENCH_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["fib"] = module  # required so @dataclass can resolve __module__
    spec.loader.exec_module(module)
    return module


def _run_to_tmp(tmp_path: Path) -> dict:
    fib = _load_benchmark()
    out = tmp_path / "audit_fault_injection_v3_1"
    summary = fib.run_benchmark(output_dir=out)
    return summary


def test_benchmark_runs_and_emits_artifacts(tmp_path):
    summary = _run_to_tmp(tmp_path)
    out = tmp_path / "audit_fault_injection_v3_1"
    for rel in ("fault_injection_results.csv", "fault_injection_results.json",
                "report.md", "artifact_manifest.json"):
        assert (out / rel).exists(), f"missing {rel}"


def test_all_expected_detections_fire(tmp_path):
    summary = _run_to_tmp(tmp_path)
    df = pd.DataFrame(summary["results"])
    expected = df[df["expected_detection"]]
    assert (expected["actual_detection"] == True).all()  # noqa: E712
    # No false alarms on faults that are expected to detect.
    assert summary["n_false_alarms"] == 0


def test_id_only_audit_misses_but_content_audit_catches(tmp_path):
    """The methodological punchline of the benchmark."""
    summary = _run_to_tmp(tmp_path)
    by_id = {r["fault_id"]: r for r in summary["results"]}
    id_only = by_id["F09a_exact_signal_duplicate_id_only_audit"]
    content = by_id["F09b_exact_signal_duplicate_content_audit"]
    # ID-only audit must MISS (distinct IDs look unique).
    assert id_only["expected_detection"] is False
    assert id_only["actual_detection"] is False
    # Content-fingerprint audit must DETECT the duplicate cluster.
    assert content["expected_detection"] is True
    assert content["actual_detection"] is True


def test_fault_count_is_not_dressed_as_clinical_metric(tmp_path):
    fib = _load_benchmark()
    out = tmp_path / "audit_fault_injection_v3_1"
    fib.run_benchmark(output_dir=out)
    report = (out / "report.md").read_text(encoding="utf-8")
    js = json.loads((out / "fault_injection_results.json").read_text(encoding="utf-8"))
    # The disclaimer must be present in both the report and the JSON contract.
    assert "not" in report.lower() and "clinical" in report.lower()
    assert "clinical" in js["contract"].lower() or "diagnostic" in js["contract"].lower()


def test_manifest_self_consistent(tmp_path):
    fib = _load_benchmark()
    out = tmp_path / "audit_fault_injection_v3_1"
    fib.run_benchmark(output_dir=out)
    manifest = json.loads((out / "artifact_manifest.json").read_text(encoding="utf-8"))
    listed = {row["path"] for row in manifest["artifacts"]}
    actual = {p.relative_to(out).as_posix() for p in out.rglob("*")
              if p.is_file() and p.name != "artifact_manifest.json"}
    assert listed == actual
