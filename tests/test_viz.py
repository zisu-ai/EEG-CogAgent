import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from eeg_cogagent.viz import _shared_topomap_limits, plot_band_topomaps, plot_topomap_composite


CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
]
BANDS = {
    "delta": [1.0, 4.0],
    "theta": [4.0, 8.0],
    "alpha": [8.0, 13.0],
    "beta": [13.0, 30.0],
    "gamma": [30.0, 45.0],
}
CFG = {"features": {"bands": BANDS}}


def _make_features(offsets: dict) -> pd.DataFrame:
    """Synthetic negative log-power features with within-scalp spatial gradients."""
    ramp = np.linspace(0.0, 0.8, len(CHANNELS))
    rows = []
    for label, offset in offsets.items():
        for subject in range(3):
            row = {"label": label}
            for band in BANDS:
                for channel, rise in zip(CHANNELS, ramp):
                    row[f"band_ch__{band}__{channel}"] = float(-12.0 + offset + rise + 0.05 * subject)
            rows.append(row)
    return pd.DataFrame(rows)


class SharedLimitsTests(unittest.TestCase):
    def test_limits_span_all_groups(self):
        group_means = {
            "AD": np.array([-11.0, -11.5, -12.0]),
            "FTD": np.array([-12.5, -12.0, -11.8]),
            "HC": np.array([-11.2, -11.9, -12.2]),
        }
        low, high = _shared_topomap_limits(group_means)
        self.assertAlmostEqual(low, -12.5)
        self.assertAlmostEqual(high, -11.0)

    def test_constant_values_get_padded_range(self):
        group_means = {"AD": np.array([-11.0, -11.0]), "FTD": np.array([-11.0])}
        low, high = _shared_topomap_limits(group_means)
        self.assertLess(low, -11.0)
        self.assertGreater(high, -11.0)
        self.assertTrue(np.isfinite(low) and np.isfinite(high))

    def test_non_finite_falls_back_to_unit_range(self):
        group_means = {"AD": np.array([np.nan, -np.inf]), "FTD": np.array([np.nan])}
        self.assertEqual(_shared_topomap_limits(group_means), (0.0, 1.0))

    def test_empty_dict_falls_back_to_unit_range(self):
        self.assertEqual(_shared_topomap_limits({}), (0.0, 1.0))


class PlotBandTopomapsTests(unittest.TestCase):
    def test_produces_png_pdf_with_shared_vlim_per_band(self):
        import mne

        features = _make_features({"AD": 0.2, "FTD": -0.3, "HC": 0.0})
        captured = []
        original = mne.viz.plot_topomap

        def spy(data, pos, **kwargs):
            captured.append({
                "vlim": tuple(kwargs.get("vlim", (None, None))),
                "cmap": kwargs.get("cmap"),
            })
            return original(data, pos, **kwargs)

        mne.viz.plot_topomap = spy
        try:
            with tempfile.TemporaryDirectory() as tmp:
                outputs = plot_band_topomaps(features, CFG, Path(tmp))
                self.assertEqual(len(outputs), len(BANDS) * 2)
                for band in BANDS:
                    self.assertTrue((Path(tmp) / f"topomap_{band}.png").exists())
                    self.assertTrue((Path(tmp) / f"topomap_{band}.pdf").exists())
        finally:
            mne.viz.plot_topomap = original

        # Five bands x three groups = fifteen topomap calls.
        self.assertEqual(len(captured), len(BANDS) * 3)
        for entry in captured:
            self.assertEqual(entry["cmap"], "cividis")
            self.assertNotEqual(entry["vlim"], (None, None))
            low, high = entry["vlim"]
            self.assertTrue(np.isfinite(low) and np.isfinite(high))
            self.assertLess(low, high)
        # Each consecutive block of three calls (one band) must share identical vlim.
        for start in range(0, len(captured), 3):
            block = [entry["vlim"] for entry in captured[start:start + 3]]
            self.assertTrue(all(vlim == block[0] for vlim in block),
                            f"vlim differs within band starting at call {start}: {block}")

    def test_degenerate_constant_data_does_not_raise(self):
        rows = []
        for label in ["AD", "FTD", "HC"]:
            for _ in range(2):
                row = {"label": label}
                for band in BANDS:
                    for channel in CHANNELS:
                        row[f"band_ch__{band}__{channel}"] = -11.0
                rows.append(row)
        features = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as tmp:
            outputs = plot_band_topomaps(features, CFG, Path(tmp))
            self.assertEqual(len(outputs), len(BANDS) * 2)
            for band in BANDS:
                self.assertTrue((Path(tmp) / f"topomap_{band}.png").exists())


class PlotTopomapCompositeTests(unittest.TestCase):
    def test_produces_nonempty_composite_png_pdf_with_shared_vlim_per_band(self):
        import mne

        features = _make_features({"AD": 0.2, "FTD": -0.3, "HC": 0.0})
        captured = []
        original = mne.viz.plot_topomap

        def spy(data, pos, **kwargs):
            captured.append({"vlim": tuple(kwargs.get("vlim", (None, None)))})
            return original(data, pos, **kwargs)

        mne.viz.plot_topomap = spy
        try:
            with tempfile.TemporaryDirectory() as tmp:
                outputs = plot_topomap_composite(features, CFG, Path(tmp))
                self.assertEqual(len(outputs), 2)
                png = Path(tmp) / "figure2_spectral_topomaps.png"
                pdf = Path(tmp) / "figure2_spectral_topomaps.pdf"
                self.assertTrue(png.exists() and png.stat().st_size > 0)
                self.assertTrue(pdf.exists() and pdf.stat().st_size > 0)
        finally:
            mne.viz.plot_topomap = original

        # Five bands x three groups = fifteen topomap calls.
        self.assertEqual(len(captured), len(BANDS) * 3)
        for entry in captured:
            low, high = entry["vlim"]
            self.assertTrue(np.isfinite(low) and np.isfinite(high))
            self.assertLess(low, high)
        # Each consecutive block of three calls (one band row) must share identical vlim.
        for start in range(0, len(captured), 3):
            block = [entry["vlim"] for entry in captured[start:start + 3]]
            self.assertTrue(all(vlim == block[0] for vlim in block),
                            f"vlim differs within band row starting at call {start}: {block}")


if __name__ == "__main__":
    unittest.main()
