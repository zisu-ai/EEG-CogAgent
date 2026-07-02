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


class CanonicalFingerprintTests(unittest.TestCase):
    """v3: duplicate-signal integrity audit (label-free, schema-versioned)."""

    @staticmethod
    def _bytes_for(value: float = 0.0, n: int = 1024) -> bytes:
        return (f"{value}\n" * n).encode("utf-8")

    def _row(self, group: str, subject: str, channel_bytes: dict) -> dict:
        from eeg_cogagent.external_osf import (
            compute_signal_fingerprint,
        )
        # Use the library to construct a real fingerprint row.
        import tempfile, zipfile
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "EEG_data.zip"
            archive_parent = f"EEG_data/{group}/Eyes_closed/{subject}"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
                for ch in COMMON_CHANNELS_19:
                    zf.writestr(f"{archive_parent}/{ch}.txt", channel_bytes[ch])
            rows = compute_signal_fingerprint(archive, condition="Eyes_closed")
        self.assertEqual(len(rows), 1)
        return rows[0]

    def test_two_distinct_ids_same_signal_form_cluster(self):
        from eeg_cogagent.external_osf import cluster_signal_fingerprints

        bytes_a = {ch: self._bytes_for(0.1 * (i + 1)) for i, ch in enumerate(COMMON_CHANNELS_19)}
        bytes_b = {ch: self._bytes_for(0.1 * (i + 1)) for i, ch in enumerate(COMMON_CHANNELS_19)}
        rows = [
            self._row("AD", "Paciente40", bytes_a),
            self._row("AD", "Paciente41", bytes_b),
        ]
        audit_df, summary = cluster_signal_fingerprints(rows)
        self.assertEqual(summary["nominal_count"], 2)
        self.assertEqual(summary["unique_fingerprint_count"], 1)
        self.assertEqual(summary["duplicate_cluster_count"], 1)
        sizes = audit_df["cluster_size"].tolist()
        self.assertEqual(sizes, [2, 2])
        representatives = set(audit_df["representative_id"].tolist())
        self.assertEqual(representatives, {"AD_Paciente40"})  # lexicographically smallest
        included = audit_df["included_primary"].tolist()
        self.assertIn(True, included)
        self.assertIn(False, included)

    def test_one_sample_change_changes_fingerprint(self):
        bytes_a = {ch: self._bytes_for(0.1 * (i + 1)) for i, ch in enumerate(COMMON_CHANNELS_19)}
        bytes_b = dict(bytes_a)
        bytes_b["Fp1"] = self._bytes_for(0.1 * 99)  # different content
        row_a = self._row("AD", "Paciente1", bytes_a)
        row_b = self._row("AD", "Paciente1", bytes_b)
        self.assertNotEqual(row_a["signal_sha256"], row_b["signal_sha256"])

    def test_same_values_different_text_formatting_same_fingerprint(self):
        # v3.1 repair: the digest is over parsed float64 values, so identical
        # numbers expressed with different whitespace / decimal precision must
        # collide (v1 hashed raw text bytes and would have diverged here).
        # Values are multiples of 0.5 so that fixed-precision formatting round-trips
        # to the exact same float64 (no rounding).
        from eeg_cogagent.external_osf import _fingerprint_one_subject
        vals = {ch: 0.5 * (i + 1) for i, ch in enumerate(COMMON_CHANNELS_19)}
        plain = {ch: (f"{vals[ch]}\n" * 1024).encode("utf-8") for ch in COMMON_CHANNELS_19}
        spaced = {ch: (f" {vals[ch]} \n" * 1024).encode("utf-8") for ch in COMMON_CHANNELS_19}
        padded = {ch: (f"{vals[ch]:.6f}\n" * 1024).encode("utf-8") for ch in COMMON_CHANNELS_19}
        self.assertEqual(_fingerprint_one_subject(plain), _fingerprint_one_subject(spaced))
        self.assertEqual(_fingerprint_one_subject(plain), _fingerprint_one_subject(padded))

    def test_wrong_sample_count_raises(self):
        from eeg_cogagent.external_osf import _fingerprint_one_subject
        short = {ch: (f"{0.1 * (i + 1)}\n" * 10).encode("utf-8") for i, ch in enumerate(COMMON_CHANNELS_19)}
        with self.assertRaises(ValueError):
            _fingerprint_one_subject(short)

    def test_non_finite_values_raise(self):
        from eeg_cogagent.external_osf import _fingerprint_one_subject
        bad = {ch: (f"{0.1 * (i + 1)}\n" * 1024).encode("utf-8") for i, ch in enumerate(COMMON_CHANNELS_19)}
        bad["Fp1"] = ("nan\n" * 1024).encode("utf-8")
        with self.assertRaises(ValueError):
            _fingerprint_one_subject(bad)

    def test_channel_order_does_not_affect_fingerprint(self):
        from eeg_cogagent.external_osf import (
            FINGERPRINT_VERSION,
            _fingerprint_one_subject,
        )
        bytes_per_channel = {ch: self._bytes_for(0.5 * i) for i, ch in enumerate(COMMON_CHANNELS_19)}
        fp1 = _fingerprint_one_subject(bytes_per_channel)
        reordered = dict(reversed(list(bytes_per_channel.items())))
        fp2 = _fingerprint_one_subject(reordered)
        self.assertEqual(fp1, fp2)
        self.assertTrue(fp1.startswith(FINGERPRINT_VERSION) or len(fp1) == 64)

    def test_same_fingerprint_conflicting_groups_hard_fails(self):
        from eeg_cogagent.external_osf import cluster_signal_fingerprints

        bytes_a = {ch: self._bytes_for(0.7) for ch in COMMON_CHANNELS_19}
        rows = [
            self._row("AD", "Paciente50", bytes_a),
            self._row("Healthy", "Paciente50", bytes_a),
        ]
        with self.assertRaises(ValueError):
            cluster_signal_fingerprints(rows)

    def test_clustering_does_not_read_labels_or_predictions(self):
        from eeg_cogagent.external_osf import cluster_signal_fingerprints
        import ast, inspect
        source = inspect.getsource(cluster_signal_fingerprints)
        tree = ast.parse(source)
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        attrs = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                attrs.add(node.attr)
        forbidden_names = {"prob", "pred", "pred_label", "score", "predict"}
        self.assertTrue(forbidden_names.isdisjoint(names),
                        msg=f"forbidden name(s) referenced in cluster_signal_fingerprints body: {forbidden_names & names}")
        self.assertTrue({"prob_AD", "pred_label"}.isdisjoint(attrs))


class RealArchiveDuplicateTests(unittest.TestCase):
    """v3 doc §8.13: read-only assertion on the canonical archive's duplicate cluster.

    Skipped if the canonical archive is absent (offline-friendly).
    """

    ARCHIVE = Path("data/osf_2v5md/EEG_data.zip")

    def _summary(self, condition: str):
        from eeg_cogagent.external_osf import (
            compute_signal_fingerprint, cluster_signal_fingerprints,
        )
        if not self.ARCHIVE.exists():
            self.skipTest(f"canonical archive not present at {self.ARCHIVE}")
        rows = compute_signal_fingerprint(self.ARCHIVE, condition=condition)
        _, summary = cluster_signal_fingerprints(rows)
        return summary

    def test_canonical_eyes_closed_duplicate_cluster(self):
        summ = self._summary("Eyes_closed")
        self.assertEqual(summ["nominal_count"], 92)
        self.assertEqual(summ["unique_fingerprint_count"], 88)
        self.assertEqual(summ["duplicate_cluster_count"], 1)
        members = [m for m in summ["clusters"].values() if len(m) > 1]
        self.assertEqual(len(members), 1)
        self.assertEqual(set(members[0]),
                         {"AD_Paciente40", "AD_Paciente41", "AD_Paciente42", "AD_Paciente43", "AD_Paciente44"})

    def test_canonical_eyes_open_reproduces_cluster(self):
        # Per the v3 doc: Eyes_open has 91 nominal (80 AD + 11 Healthy) and the
        # same AD_Paciente40-44 duplicate cluster reproduces.
        summ = self._summary("Eyes_open")
        self.assertEqual(summ["nominal_count"], 91)
        self.assertEqual(summ["unique_fingerprint_count"], 87)
        self.assertEqual(summ["duplicate_cluster_count"], 1)
        members = [m for m in summ["clusters"].values() if len(m) > 1]
        self.assertEqual(set(members[0]),
                         {"AD_Paciente40", "AD_Paciente41", "AD_Paciente42", "AD_Paciente43", "AD_Paciente44"})


if __name__ == "__main__":
    unittest.main()
