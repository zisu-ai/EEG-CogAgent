from __future__ import annotations

import math

import numpy as np


def _safe_log10(value: float) -> float:
    if value <= 0 or not math.isfinite(value):
        return float("nan")
    return float(np.log10(value))


def extract_subject_features(epochs, cfg: dict) -> dict[str, float]:
    feature_cfg = cfg["features"]
    bands = feature_cfg["bands"]
    fmin = min(pair[0] for pair in bands.values())
    fmax = max(pair[1] for pair in bands.values())
    psd = epochs.compute_psd(method="welch", fmin=fmin, fmax=fmax, verbose="ERROR")
    data = psd.get_data()  # epochs x channels x frequencies
    freqs = psd.freqs
    ch_names = psd.ch_names

    features: dict[str, float] = {}
    band_by_channel: dict[str, dict[str, float]] = {}

    for band, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        if not mask.any():
            continue
        values = data[:, :, mask].mean(axis=(0, 2))
        band_by_channel[band] = {}
        for channel, value in zip(ch_names, values):
            log_power = _safe_log10(float(value))
            band_by_channel[band][channel] = log_power
            features[f"band_ch__{band}__{channel}"] = log_power
        features[f"band_global__{band}"] = float(np.nanmean(list(band_by_channel[band].values())))

        for region, channels in feature_cfg.get("regions", {}).items():
            present = [band_by_channel[band][ch] for ch in channels if ch in band_by_channel[band]]
            if present:
                features[f"band_region__{band}__{region}"] = float(np.nanmean(present))

    for numerator, denominator in feature_cfg.get("ratios", []):
        if numerator not in band_by_channel or denominator not in band_by_channel:
            continue
        if f"band_global__{numerator}" in features and f"band_global__{denominator}" in features:
            features[f"ratio_global__{numerator}_{denominator}"] = (
                features[f"band_global__{numerator}"] - features[f"band_global__{denominator}"]
            )
        for region in feature_cfg.get("regions", {}):
            num_key = f"band_region__{numerator}__{region}"
            den_key = f"band_region__{denominator}__{region}"
            if num_key in features and den_key in features:
                features[f"ratio_region__{numerator}_{denominator}__{region}"] = features[num_key] - features[den_key]

    features["n_epochs"] = int(len(epochs))
    return features

