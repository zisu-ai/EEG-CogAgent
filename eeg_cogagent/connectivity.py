from __future__ import annotations

import math

import networkx as nx
import numpy as np


def _connectivity_from_fourier(fourier: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute coherence, PLI, and wPLI from epoch-wise Fourier coefficients."""
    if fourier.ndim != 3:
        raise ValueError("Expected Fourier data shaped epochs x channels x frequencies.")
    n_channels = fourier.shape[1]
    coherence = np.eye(n_channels, dtype=float)
    pli = np.zeros((n_channels, n_channels), dtype=float)
    wpli = np.zeros((n_channels, n_channels), dtype=float)

    auto = np.mean(np.abs(fourier) ** 2, axis=0)
    eps = np.finfo(float).eps
    for left in range(n_channels):
        for right in range(left + 1, n_channels):
            cross = fourier[:, left, :] * np.conjugate(fourier[:, right, :])
            cross_mean = np.mean(cross, axis=0)
            denominator = auto[left] * auto[right]
            coh_value = np.mean(np.abs(cross_mean) ** 2 / np.maximum(denominator, eps))

            imaginary = np.imag(cross)
            pli_value = np.mean(np.abs(np.mean(np.sign(imaginary), axis=0)))
            wpli_frequency = np.abs(np.mean(imaginary, axis=0)) / np.maximum(
                np.mean(np.abs(imaginary), axis=0), eps
            )
            wpli_value = np.mean(wpli_frequency)

            coherence[left, right] = coherence[right, left] = float(np.clip(coh_value, 0, 1))
            pli[left, right] = pli[right, left] = float(np.clip(pli_value, 0, 1))
            wpli[left, right] = wpli[right, left] = float(np.clip(wpli_value, 0, 1))
    return coherence, pli, wpli


def _weighted_global_efficiency(graph: nx.Graph) -> float:
    nodes = list(graph.nodes)
    if len(nodes) < 2:
        return float("nan")
    lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight="distance"))
    inverse_sum = 0.0
    for source in nodes:
        for target in nodes:
            if source == target:
                continue
            distance = lengths.get(source, {}).get(target)
            if distance is not None and distance > 0:
                inverse_sum += 1.0 / distance
    return inverse_sum / (len(nodes) * (len(nodes) - 1))


def graph_metrics(
    matrix: np.ndarray,
    channel_names: list[str],
    proportional_threshold: float = 0.2,
) -> dict[str, float]:
    """Calculate weighted graph metrics after a fixed-density threshold."""
    n_channels = len(channel_names)
    if matrix.shape != (n_channels, n_channels):
        raise ValueError("Connectivity matrix shape does not match channel names.")
    if not 0 < proportional_threshold <= 1:
        raise ValueError("proportional_threshold must be in (0, 1].")

    upper = np.triu_indices(n_channels, k=1)
    weights = matrix[upper]
    edge_count = max(1, int(math.ceil(len(weights) * proportional_threshold)))
    selected = np.argsort(weights)[-edge_count:]

    graph = nx.Graph()
    graph.add_nodes_from(channel_names)
    for index in selected:
        left = upper[0][index]
        right = upper[1][index]
        weight = float(weights[index])
        if not np.isfinite(weight) or weight <= 0:
            continue
        graph.add_edge(
            channel_names[left],
            channel_names[right],
            weight=weight,
            distance=1.0 / max(weight, np.finfo(float).eps),
        )

    strengths = [
        sum(attributes["weight"] for _, _, attributes in graph.edges(node, data=True))
        / max(n_channels - 1, 1)
        for node in channel_names
    ]
    return {
        "clustering": float(nx.average_clustering(graph, weight="weight")),
        "global_efficiency": float(_weighted_global_efficiency(graph)),
        "mean_strength": float(np.mean(strengths)),
    }


def extract_connectivity_features(
    epochs,
    cfg: dict,
) -> tuple[dict[str, float], dict[str, dict[str, np.ndarray]]]:
    """Extract band-specific connectivity and wPLI graph features."""
    connection_cfg = cfg.get("connectivity", {})
    configured_bands = cfg["features"]["bands"]
    requested_bands = connection_cfg.get("bands", list(configured_bands))
    bands = {name: configured_bands[name] for name in requested_bands}
    threshold = float(connection_cfg.get("proportional_threshold", 0.2))

    data = epochs.get_data(copy=True)
    data -= data.mean(axis=-1, keepdims=True)
    window = np.hanning(data.shape[-1])
    fourier = np.fft.rfft(data * window, axis=-1)
    frequencies = np.fft.rfftfreq(data.shape[-1], d=1.0 / float(epochs.info["sfreq"]))
    channel_names = list(epochs.ch_names)
    upper = np.triu_indices(len(channel_names), k=1)

    features: dict[str, float] = {}
    matrices: dict[str, dict[str, np.ndarray]] = {}
    for band, (low, high) in bands.items():
        frequency_mask = (frequencies >= float(low)) & (frequencies < float(high))
        if not frequency_mask.any():
            continue
        coherence, pli, wpli = _connectivity_from_fourier(fourier[:, :, frequency_mask])
        matrices[band] = {"coherence": coherence, "pli": pli, "wpli": wpli}
        for metric, matrix in matrices[band].items():
            features[f"conn_global__{metric}__{band}"] = float(np.mean(matrix[upper]))
        for metric, value in graph_metrics(wpli, channel_names, threshold).items():
            features[f"graph__wpli__{band}__{metric}"] = value
    return features, matrices


def matrices_to_edge_rows(
    matrices: dict[str, dict[str, np.ndarray]],
    channel_names: list[str],
    participant_id: str,
    label: str,
) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for band, metric_matrices in matrices.items():
        for metric, matrix in metric_matrices.items():
            for left in range(len(channel_names)):
                for right in range(left + 1, len(channel_names)):
                    rows.append({
                        "participant_id": participant_id,
                        "label": label,
                        "band": band,
                        "metric": metric,
                        "source": channel_names[left],
                        "target": channel_names[right],
                        "value": float(matrix[left, right]),
                    })
    return rows
