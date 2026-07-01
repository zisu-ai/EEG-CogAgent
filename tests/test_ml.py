import unittest

import numpy as np

from eeg_cogagent.ml import NuisanceResidualizer


class NuisanceResidualizerTests(unittest.TestCase):
    def test_removes_linear_nuisance_association(self):
        rng = np.random.default_rng(42)
        nuisance = rng.normal(size=(60, 3))
        coefficients = np.array([[2.0, 0.0], [-1.0, 3.0], [0.5, 0.2]])
        eeg = nuisance @ coefficients + rng.normal(scale=0.01, size=(60, 2))
        matrix = np.column_stack([eeg, nuisance])

        residuals = NuisanceResidualizer(n_eeg_features=2).fit_transform(matrix)

        correlations = np.corrcoef(
            np.column_stack([residuals, nuisance]), rowvar=False
        )[:2, 2:]
        self.assertEqual(residuals.shape, eeg.shape)
        self.assertLess(np.max(np.abs(correlations)), 1e-10)

    def test_rejects_invalid_eeg_feature_count(self):
        matrix = np.ones((5, 3))
        with self.assertRaises(ValueError):
            NuisanceResidualizer(n_eeg_features=3).fit(matrix)


if __name__ == "__main__":
    unittest.main()
