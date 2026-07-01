import unittest

import networkx as nx
import numpy as np

from eeg_cogagent.connectivity import _connectivity_from_fourier, graph_metrics


class ConnectivityTests(unittest.TestCase):
    def test_fixed_phase_lag_has_unit_connectivity(self):
        epochs = 20
        frequencies = 5
        base = np.ones((epochs, frequencies), dtype=complex)
        lagged = base * np.exp(1j * np.pi / 2)
        fourier = np.stack([base, lagged], axis=1)

        coherence, pli, wpli = _connectivity_from_fourier(fourier)

        self.assertAlmostEqual(coherence[0, 1], 1.0, places=12)
        self.assertAlmostEqual(pli[0, 1], 1.0, places=12)
        self.assertAlmostEqual(wpli[0, 1], 1.0, places=12)

    def test_graph_metrics_are_finite(self):
        matrix = np.array([
            [1.0, 0.9, 0.2, 0.1],
            [0.9, 1.0, 0.8, 0.3],
            [0.2, 0.8, 1.0, 0.7],
            [0.1, 0.3, 0.7, 1.0],
        ])
        metrics = graph_metrics(matrix, ["A", "B", "C", "D"], 0.5)
        self.assertEqual(set(metrics), {"clustering", "global_efficiency", "mean_strength"})
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_networkx_weighted_clustering_contract(self):
        graph = nx.Graph()
        graph.add_weighted_edges_from([("A", "B", 1.0), ("B", "C", 1.0), ("A", "C", 1.0)])
        self.assertAlmostEqual(nx.average_clustering(graph, weight="weight"), 1.0)


if __name__ == "__main__":
    unittest.main()
