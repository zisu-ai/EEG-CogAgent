"""Unit tests for external-validation v2 (1-30 Hz, 36 features, leak-free nested CV).

Offline and fast by default: synthetic signals/frames only, no real archive or
BIDS data. CLI hard-gate behaviour is exercised through (a) importing the script
module to call its pure gate helpers directly, and (b) two cheap subprocess
invocations (condition != Eyes_closed; archive SHA mismatch) that fail before
touching ds004504. Maps prompts/claude_external_validation_v2_hardening.md §5.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_cogagent import external_validation as ev
from eeg_cogagent.external_osf import COMMON_CHANNELS_19

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "external_validation_osf.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("ev_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_matrix(n_ad: int, n_hc: int, n_ftd: int = 0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, pid = [], 0
    for label, count in (("AD", n_ad), ("HC", n_hc), ("FTD", n_ftd)):
        for _ in range(count):
            pid += 1
            offset = 1.5 if label == "AD" else (-1.5 if label == "HC" else 0.0)
            row = {"participant_id": f"syn-{pid:03d}", "group": label, "label": label}
            for index, feature in enumerate(ev.V2_HARMONIZED_FEATURES):
                row[feature] = float(rng.normal(loc=offset if index == 0 else 0.0, scale=1.0))
            rows.append(row)
    return pd.DataFrame(rows)


def _signal(freqs, sfreq, n_samples, seed=0, amplitude=1.0):
    """Multi-sine + small-noise signal across the 19 common channels (per-channel phase)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / sfreq
    channels = {}
    for ch in COMMON_CHANNELS_19:
        sig = np.zeros(n_samples)
        for f in freqs:
            sig += amplitude * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        channels[ch] = sig + rng.normal(0.0, 0.05, size=n_samples)
    return channels


class FeatureSetTests(unittest.TestCase):
    def test_36_features_no_gamma_globals_sum_to_one(self):
        feats = ev.V2_HARMONIZED_FEATURES
        self.assertEqual(len(feats), 36)
        self.assertFalse(any("gamma" in f for f in feats))
        self.assertEqual(len([f for f in feats if f.startswith("relpow_global__")]), 4)
        self.assertEqual(len([f for f in feats if f.startswith("relpow_region__")]), 20)
        self.assertEqual(len([f for f in feats if f.startswith("ratio_global__")]), 2)
        self.assertEqual(len([f for f in feats if f.startswith("ratio_region__")]), 10)
        channels = _signal([2.0, 6.0, 10.0, 20.0], 500.0, 4000)
        out = ev.harmonized_features_from_signal_v2(channels, 500.0, 1000)
        self.assertEqual(set(out.keys()), set(feats))
        globals_ = [out[f] for f in feats if f.startswith("relpow_global__")]
        for value in globals_:
            self.assertTrue(0.0 <= value <= 1.0)
        self.assertAlmostEqual(sum(globals_), 1.0, places=6)

    def test_resolution_aligned_features_match_across_sfreqs(self):
        # Same 4 in-band sines, 8 s at 128 Hz (1024 samples) and 500 Hz (4000 samples),
        # both at 0.5 Hz Welch resolution -> comparable global relative powers.
        low = ev.harmonized_features_from_signal_v2(_signal([2, 6, 10, 20], 128.0, 1024, seed=1), 128.0, 256)
        high = ev.harmonized_features_from_signal_v2(_signal([2, 6, 10, 20], 500.0, 4000, seed=1), 500.0, 1000)
        bands = ["delta", "theta", "alpha", "beta"]
        for band in bands:
            a = low[f"relpow_global__{band}"]
            b = high[f"relpow_global__{band}"]
            self.assertLess(abs(a - b), 0.06, msg=f"{band}: {a:.4f} vs {b:.4f}")

    def test_35hz_out_of_band_does_not_change_features(self):
        base = _signal([2, 6, 10, 20], 128.0, 1024, seed=2)
        feats_a = ev.harmonized_features_from_signal_v2(base, 128.0, 256)
        rng = np.random.default_rng(99)
        with35 = {ch: base[ch] + 2.0 * np.sin(2 * np.pi * 35.0 * np.arange(1024) / 128.0 + rng.uniform(0, 2 * np.pi))
                  for ch in COMMON_CHANNELS_19}
        feats_b = ev.harmonized_features_from_signal_v2(with35, 128.0, 256)
        for feature in ev.V2_HARMONIZED_FEATURES:
            self.assertLess(abs(feats_a[feature] - feats_b[feature]), 0.02,
                            msg=f"35Hz leaked into {feature}")

    def test_fail_rules_on_bad_signals(self):
        good = _signal([2, 6, 10, 20], 128.0, 1024, seed=3)
        # Missing common channel.
        missing = dict(good); missing.pop("Fp1")
        with self.assertRaises(ValueError):
            ev.harmonized_features_from_signal_v2(missing, 128.0, 256)
        # Non-finite values.
        nan_sig = dict(good); nan_sig["Fz"] = nan_sig["Fz"].copy(); nan_sig["Fz"][5] = np.nan
        with self.assertRaises(ValueError):
            ev.harmonized_features_from_signal_v2(nan_sig, 128.0, 256)
        # Flat / zero-power channel.
        flat = dict(good); flat["Cz"] = np.full(1024, 3.0)
        with self.assertRaises(ValueError):
            ev.harmonized_features_from_signal_v2(flat, 128.0, 256)


class TrainingAndShiftTests(unittest.TestCase):
    def _fit(self, seed=42):
        train = _synthetic_matrix(36, 29, 23, seed=1)
        binary = train[train["label"].isin(["AD", "HC"])].reset_index(drop=True)
        X = binary[list(ev.V2_HARMONIZED_FEATURES)]
        y = (binary["label"] == "AD").astype(int).to_numpy()
        return train, binary, X, y, ev.fit_final_model_and_threshold(X, y, seed=seed)

    def test_ftd_excluded_from_training_and_primary_shift(self):
        train, binary, X, y, fitted = self._fit()
        # Training data is AD+HC only (65), not 88.
        self.assertEqual(len(y), 65)
        self.assertEqual(int((y == 1).sum()), 36)
        self.assertEqual(int((y == 0).sum()), 29)
        # Primary shift is label-free: discovery side must be AD+HC (65), never 88.
        external = _synthetic_matrix(80, 12, 0, seed=2)
        shift = ev.domain_shift_primary_labelfree(binary, external)
        self.assertEqual(set(shift["n_discovery"]), {65})
        self.assertEqual(set(shift["n_external"]), {92})
        # The function never references the external label column.
        import inspect
        self.assertNotIn("external_df[\"label\"]", inspect.getsource(ev.domain_shift_primary_labelfree))

    def test_external_label_shuffle_does_not_change_model_outputs(self):
        train, binary, X, y, fitted = self._fit()
        external = _synthetic_matrix(80, 12, 0, seed=2)
        original = ev.predict_external(fitted, external)
        swapped = external.copy()
        swapped["label"] = "HC"
        relabeled = ev.predict_external(fitted, swapped)
        np.testing.assert_array_equal(original["pred_label"].to_numpy(), relabeled["pred_label"].to_numpy())
        np.testing.assert_allclose(original["prob_AD"].to_numpy(), relabeled["prob_AD"].to_numpy())
        # Coefficients, C, threshold, imputer/scaler are unchanged (model never refit).
        refit = ev.fit_final_model_and_threshold(X, y, seed=42)
        self.assertEqual(refit["best_C"], fitted["best_C"])
        self.assertEqual(refit["threshold"], fitted["threshold"])
        np.testing.assert_allclose(list(fitted["coefficients"].values()), list(refit["coefficients"].values()))


class NestedCVIsolationTests(unittest.TestCase):
    def test_outer_test_labels_excluded_from_fold_selection(self):
        train = _synthetic_matrix(36, 29, 0, seed=1)
        X = train[list(ev.V2_HARMONIZED_FEATURES)]
        y = (train["label"] == "AD").astype(int).to_numpy()

        seen = []
        real_select_C = ev.select_C

        def spy_select_C(X_tr, y_tr, seed, fold=0):
            seen.append((fold, set(X_tr.index.tolist())))
            return real_select_C(X_tr, y_tr, seed, fold)

        ev.select_C = spy_select_C
        try:
            ev.nested_cv_internal_estimate(X, y, seed=42)
        finally:
            ev.select_C = real_select_C

        # Reconstruct the exact outer splits and assert each fold's selector saw
        # only that fold's outer-training rows.
        outer = __import__("sklearn.model_selection", fromlist=["StratifiedKFold"]).StratifiedKFold(
            n_splits=5, shuffle=True, random_state=42)
        for (fold, seen_indices), (train_idx, _test_idx) in zip(
                sorted(seen), outer.split(X, y), strict=True):
            self.assertEqual(seen_indices, set(train_idx.tolist()),
                             msg=f"fold {fold} selector saw indices outside outer-train")


class BootstrapAndCITests(unittest.TestCase):
    def test_bootstrap_stratified_deterministic_and_missing_class_error(self):
        rng = np.random.default_rng(0)
        truth = np.array(["AD"] * 80 + ["HC"] * 12)
        pred = truth.copy()
        prob = np.concatenate([rng.uniform(0.6, 1.0, 80), rng.uniform(0.0, 0.4, 12)])
        a = ev.subject_level_bootstrap_metrics(truth, prob, pred, n_boot=300, seed=42)
        b = ev.subject_level_bootstrap_metrics(truth, prob, pred, n_boot=300, seed=42)
        self.assertEqual(a, b)
        for metric in a.values():
            self.assertLessEqual(metric["ci_low"], metric["point"])
            self.assertGreaterEqual(metric["ci_high"], metric["point"])
            self.assertTrue(0.0 <= metric["ci_low"] <= 1.0 <= metric["ci_high"] or metric["ci_high"] <= 1.0)
        # All-HC truth -> no AD class -> clear error.
        with self.assertRaises(ValueError):
            ev.subject_level_bootstrap_metrics(np.array(["HC"] * 10), prob[:10], np.array(["HC"] * 10), n_boot=50)

    def test_wilson_ci_known_values(self):
        # k=8, n=12: Wilson 95% ~ [0.391, 0.862] (standard reference).
        low, high = ev.wilson_ci(8, 12)
        self.assertAlmostEqual(low, 0.391, places=2)
        self.assertAlmostEqual(high, 0.862, places=2)
        # k=0 -> lower bound 0; k=n -> upper bound 1.
        self.assertAlmostEqual(ev.wilson_ci(0, 12)[0], 0.0, places=10)
        self.assertLessEqual(ev.wilson_ci(0, 12)[1], 1.0)
        low_n, high_n = ev.wilson_ci(12, 12)
        self.assertGreaterEqual(low_n, 0.6)
        self.assertAlmostEqual(high_n, 1.0, places=6)


class CliGateTests(unittest.TestCase):
    def test_validate_external_frame_rejects_bad_inputs(self):
        script = _load_script_module()
        audit_ids = {f"AD_Paciente{i}" for i in range(1, 81)} | {f"Healthy_Paciente{i}" for i in range(1, 13)}

        good = _synthetic_matrix(80, 12, 0, seed=2)
        good["participant_id"] = list(audit_ids)
        script._validate_external(good, audit_ids)  # no raise

        # Unknown label.
        bad_label = good.copy(); bad_label.loc[bad_label.index[0], "label"] = "XYZ"
        with self.assertRaises(script.GateError):
            script._validate_external(bad_label, audit_ids)
        # Wrong count / duplicate ID.
        dup = pd.concat([good, good.iloc[:1]], ignore_index=True)
        with self.assertRaises(script.GateError):
            script._validate_external(dup, audit_ids)
        # ID set mismatch.
        bad_ids = good.copy(); bad_ids["participant_id"] = [f"x{i}" for i in range(len(good))]
        with self.assertRaises(script.GateError):
            script._validate_external(bad_ids, audit_ids)

    def test_cli_condition_not_eyes_closed_fails_without_publishing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "ev2"
            result = subprocess.run(
                [sys.executable, str(SCRIPT_PATH), "validate",
                 "--config", str(REPO_ROOT / "configs/ds004504_minimal.yaml"),
                 "--archive", str(REPO_ROOT / "data/osf_2v5md/EEG_data.zip"),
                 "--output-dir", str(out_dir), "--condition", "Eyes_open"],
                capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=180,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(out_dir.exists() and (out_dir / "CODEX_REVIEW_REQUEST.md").exists())

    def test_cli_sha_mismatch_fails_without_publishing(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_zip = Path(tmp) / "fake.zip"
            fake_zip.write_bytes(b"not a real archive")
            out_dir = Path(tmp) / "ev2"
            result = subprocess.run(
                [sys.executable, str(SCRIPT_PATH), "validate",
                 "--config", str(REPO_ROOT / "configs/ds004504_minimal.yaml"),
                 "--archive", str(fake_zip), "--output-dir", str(out_dir)],
                capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=180,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(out_dir.exists() and (out_dir / "CODEX_REVIEW_REQUEST.md").exists())


class DeterminismTests(unittest.TestCase):
    def test_offline_extraction_and_fit_are_deterministic(self):
        channels = _signal([2, 6, 10, 20], 500.0, 4000, seed=5)
        a = ev.harmonized_features_from_signal_v2(channels, 500.0, 1000)
        b = ev.harmonized_features_from_signal_v2(channels, 500.0, 1000)
        self.assertEqual(a, b)
        train = _synthetic_matrix(36, 29, 0, seed=1)
        X = train[list(ev.V2_HARMONIZED_FEATURES)]
        y = (train["label"] == "AD").astype(int).to_numpy()
        nested_a = ev.nested_cv_internal_estimate(X, y, seed=42)
        nested_b = ev.nested_cv_internal_estimate(X, y, seed=42)
        self.assertAlmostEqual(nested_a["balanced_accuracy"], nested_b["balanced_accuracy"], places=9)
        pd.testing.assert_frame_equal(nested_a["oof"], nested_b["oof"])


if __name__ == "__main__":
    unittest.main()
