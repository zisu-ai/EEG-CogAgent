"""Unit tests for the OSF Phase-1 archive parser + spectral feature extractor.

Tests use a small synthetic ZIP built in a temporary directory; the real OSF
archive is never touched (kept offline and fast). Covers path safety, cohort
indexing and condition filtering, the cohort audit, one-row-per-subject
extraction with F1/F2 exclusion, the relative-power unit-interval property,
128 Hz / sfreq threading, and determinism.
"""
from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from eeg_cogagent.external_osf import (
    COMMON_CHANNELS_19,
    DEFAULT_BANDS,
    EXCLUDED_CHANNELS,
    OSF_SAMPLING_RATE_HZ,
    build_feature_matrix,
    cohort_audit,
    index_archive,
    is_safe_member_name,
)

#: Channels written into the synthetic archive: the 19 common ones plus the two
#: deliberately-excluded channels, so exclusion is exercised end-to-end.
ZIP_CHANNELS = (*COMMON_CHANNELS_19, *sorted(EXCLUDED_CHANNELS))


def _write_synthetic_zip(
    path: Path,
    rng: np.random.Generator,
    subjects: list[tuple[str, str]],
    *,
    condition: str = "Eyes_closed",
    n_samples: int = 1024,
    channels: tuple[str, ...] = ZIP_CHANNELS,
    signal_hz: float | None = None,
    noise_scale: float = 1.0,
    per_channel_phase: bool = False,
    extra_members: list[str] | None = None,
) -> None:
    """Write a minimal but valid-layout archive to ``path``.

    Each requested channel for each subject is a text file of one float per line.
    When ``signal_hz`` is set, a sine at that frequency is added (optionally with
    a per-channel random phase); Gaussian noise is always added so every band has
    nonzero power and ratios stay finite.
    """
    t = np.arange(n_samples) / OSF_SAMPLING_RATE_HZ
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for group, subject in subjects:
            for channel in channels:
                signal = np.zeros(n_samples)
                if signal_hz is not None:
                    phase = float(rng.uniform(0, 2 * np.pi)) if per_channel_phase else 0.0
                    signal = np.sin(2 * np.pi * signal_hz * t + phase)
                noise = rng.normal(0.0, noise_scale, size=n_samples)
                values = signal + noise
                member = f"EEG_data/{group}/{condition}/{subject}/{channel}.txt"
                payload = "\n".join(f"{v:.10f}" for v in values) + "\n"
                archive.writestr(member, payload)
        for member in extra_members or []:
            archive.writestr(member, "0\n")


class PathSafetyTests(unittest.TestCase):
    def test_is_safe_member_name_rejects_traversal(self):
        for bad in ("../evil.txt", "/etc/passwd", "a\\b.txt", "C:x.txt", "", "a/..", "a/./b"):
            self.assertFalse(is_safe_member_name(bad), msg=f"expected reject: {bad!r}")
        self.assertTrue(is_safe_member_name("EEG_data/AD/Eyes_closed/Paciente1/Fp1.txt"))


class ArchiveIndexTests(unittest.TestCase):
    def test_index_archive_group_identity_and_condition_filtering(self):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "EEG_data.zip"
            rng = np.random.default_rng(0)
            # 2 AD + 2 Healthy Eyes_closed, plus 1 AD Eyes_open that must be excluded.
            closed = [("AD", "Paciente1"), ("AD", "Paciente2"),
                      ("Healthy", "Paciente1"), ("Healthy", "Paciente2")]
            _write_synthetic_zip(zip_path, rng, closed)
            # Append an Eyes_open subject to the same archive via a fresh write.
            eyes_open_path = Path(tmp) / "EO.zip"
            rng2 = np.random.default_rng(1)
            _write_synthetic_zip(
                eyes_open_path, rng2, [("AD", "Paciente3")], condition="Eyes_open"
            )
            # Merge both conditions into one archive.
            merged = Path(tmp) / "merged.zip"
            with zipfile.ZipFile(zip_path) as a, zipfile.ZipFile(eyes_open_path) as b:
                with zipfile.ZipFile(merged, "w", zipfile.ZIP_DEFLATED) as out:
                    for src in (a, b):
                        for name in src.namelist():
                            out.writestr(name, src.read(name))

            summary = index_archive(merged, condition="Eyes_closed")
            self.assertEqual(summary.groups,
                             {"AD": {"subjects": 2}, "Healthy": {"subjects": 2}})
            self.assertEqual(len(summary.subjects), 4)
            self.assertEqual({r["group"] for r in summary.subjects}, {"AD", "Healthy"})
            # The Eyes_open Paciente3 must not appear among Eyes_closed subjects.
            self.assertFalse(any(r["subject"] == "Paciente3" for r in summary.subjects))


class CohortAuditTests(unittest.TestCase):
    def test_audit_passes_with_expected_counts_and_common_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "EEG_data.zip"
            rng = np.random.default_rng(7)
            subjects = [("AD", "Paciente1"), ("AD", "Paciente2"),
                        ("Healthy", "Paciente1"), ("Healthy", "Paciente2")]
            _write_synthetic_zip(zip_path, rng, subjects)
            audit = cohort_audit(
                zip_path,
                expected_groups={"AD": 2, "Healthy": 2},
                inspect_samples=False,
            )
        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["status_counts"]["fail"], 0)
        self.assertTrue(all(r["has_all_common_19"] for r in audit["subjects"]))
        # F1/F2 were present in the archive and recorded as intentionally excluded.
        self.assertTrue(all(r["ex_channels_present"] for r in audit["subjects"]))


class FeatureMatrixTests(unittest.TestCase):
    @staticmethod
    def _noise_zip(tmp: str) -> Path:
        zip_path = Path(tmp) / "EEG_data.zip"
        rng = np.random.default_rng(42)
        subjects = [("AD", "Paciente1"), ("AD", "Paciente2"),
                    ("Healthy", "Paciente1"), ("Healthy", "Paciente2")]
        _write_synthetic_zip(zip_path, rng, subjects)
        return zip_path

    def test_one_row_per_subject_label_mapping_and_channel_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = self._noise_zip(tmp)
            features, failures = build_feature_matrix(zip_path)
        self.assertEqual(failures, [])
        self.assertEqual(len(features), 4)
        self.assertEqual(features["participant_id"].nunique(), 4)
        # Healthy -> HC, AD -> AD.
        label_map = dict(zip(features["group"], features["label"]))
        self.assertEqual(label_map["AD"], "AD")
        self.assertEqual(label_map["Healthy"], "HC")
        # No excluded channel leaked into feature columns.
        feature_cols = [c for c in features.columns
                        if c not in {"participant_id", "group", "label", "n_samples"}]
        for col in feature_cols:
            self.assertNotIn("__F1", col)
            self.assertNotIn("__F2", col)
        # Per-channel relative-power columns cover exactly the 19 common channels.
        alpha_channels = sorted(
            col.split("__")[2] for col in feature_cols if col.startswith("relpow_ch__alpha__")
        )
        self.assertEqual(alpha_channels, sorted(COMMON_CHANNELS_19))

    def test_relative_powers_in_unit_interval_and_finite(self):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = self._noise_zip(tmp)
            features, _ = build_feature_matrix(zip_path)
        bands = list(DEFAULT_BANDS.keys())
        for _, row in features.iterrows():
            for channel in COMMON_CHANNELS_19:
                total = sum(row[f"relpow_ch__{band}__{channel}"] for band in bands)
                self.assertAlmostEqual(total, 1.0, places=6,
                                       msg=f"relpowers != 1 for {channel}")
        numeric = features.drop(columns=["participant_id", "group", "label"])
        self.assertTrue(np.isfinite(numeric.to_numpy(dtype=float)).all(),
                        "all feature values must be finite")

    def test_128hz_threads_into_welch(self):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "EEG_data.zip"
            rng = np.random.default_rng(3)
            subjects = [("AD", "Paciente1"), ("Healthy", "Paciente1")]
            # 10 Hz sine + small noise per channel, avg-ref OFF so the sine is not
            # removed as common mode; this isolates the sfreq -> Welch path.
            _write_synthetic_zip(
                zip_path, rng, subjects, signal_hz=10.0, noise_scale=0.05,
                per_channel_phase=True,
            )
            feats, _ = build_feature_matrix(zip_path, average_reference=False)
            bands = list(DEFAULT_BANDS.keys())
            # The 10 Hz sine must be classified as alpha dominance on every subject.
            for _, row in feats.iterrows():
                globals_ = {b: row[f"relpow_global__{b}"] for b in bands}
                self.assertEqual(max(globals_, key=globals_.get), "alpha")

            # sfreq is actually used: changing it changes frequency-bin allocation.
            same_default = build_feature_matrix(zip_path, average_reference=False)[0]
            same_explicit = build_feature_matrix(
                zip_path, sfreq=128.0, average_reference=False
            )[0]
            self.assertTrue(same_default.equals(same_explicit))
            shifted = build_feature_matrix(
                zip_path, sfreq=256.0, average_reference=False
            )[0]
            self.assertFalse(same_default.equals(shifted))

    def test_determinism(self):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = self._noise_zip(tmp)
            first, _ = build_feature_matrix(zip_path)
            second, _ = build_feature_matrix(zip_path)
        self.assertTrue(first.equals(second))


if __name__ == "__main__":
    unittest.main()
