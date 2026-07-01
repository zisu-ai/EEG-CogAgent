"""Phase-2 v2 external validation: 1-30 Hz harmonized features, leak-free nested
discovery training, and independent OSF AD-vs-HC evaluation.

v2 methodological corrections (see prompts/claude_external_validation_v2_hardening.md):

* Frequency support is genuinely homogeneous. The OSF archive is source
  band-limited to 0.5-30 Hz (per the associated paper, DOI 10.1038/s41598-023-32664-8),
  so v2 uses four common bands delta/theta/alpha/beta over 1-30 Hz and **drops
  gamma entirely**. Relative-power denominators span only the four common bands.
* The internal nested-CV estimate is unbiased: per outer fold, both ``C`` and the
  decision threshold are chosen on the outer-*training* data only (via inner
  cross-fitting), then frozen and applied to the outer-test fold. The final
  external model is trained on all ds004504 AD/HC with C and threshold chosen by
  discovery-only cross-fitting. OSF never enters fitting, tuning, or selection.
* The primary domain-shift audit is label-free (discovery AD+HC vs external all,
  never reading the external label column) and reports standardized mean
  differences with a correct pooled-SD formula plus KS statistics.

No deep learning, no ensembles, no model comparison. The predeclared primary
model is an L2 Logistic Regression with ``class_weight="balanced"``.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .bids import load_participants
from .config import project_path
from .external_osf import (
    COMMON_CHANNELS_19,
    EXPECTED_SAMPLES_PER_CHANNEL,
    SubjectKey,
    extract_subject_features,
    index_archive,
    read_subject_channels,
)
from .preprocess import make_epochs

# --- v2 frequency support (no gamma; 1-30 Hz common denominator) ------------

#: v2 common bands. Edges half-open [low, high). Gamma is deliberately excluded:
#: the OSF archive is source band-limited to 0.5-30 Hz.
V2_BANDS: "OrderedDict[str, tuple[float, float]]" = OrderedDict([
    ("delta", (1.0, 4.0)),
    ("theta", (4.0, 8.0)),
    ("alpha", (8.0, 13.0)),
    ("beta", (13.0, 30.0)),
])

#: Band-power ratios (scale-invariant). Same pairings as the discovery config.
V2_RATIOS: tuple[tuple[str, str], ...] = (("theta", "alpha"), ("delta", "alpha"))

#: Scalp regions over the common 19 channels (10-20 temporal/parietal names,
#: matching both datasets).
V2_REGIONS: dict[str, tuple[str, ...]] = {
    "frontal": ("Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"),
    "temporal": ("T3", "T4", "T5", "T6"),
    "central": ("C3", "C4", "Cz"),
    "parietal": ("P3", "P4", "Pz"),
    "occipital": ("O1", "O2"),
}

#: Welch frequency resolution matched between datasets (0.5 Hz).
TARGET_FREQ_RESOLUTION_HZ: float = 0.5

#: Discovery sampling rate (verified ds004504 value; nperseg derived at runtime).
DISCOVERY_SFREQ_HZ: float = 500.0

#: Binary encoding: AD is the positive class.
POS_LABEL = "AD"
NEG_LABEL = "HC"

# --- Predeclared analysis constants (fixed before reading any OSF labels) ----

C_GRID: tuple[float, ...] = (0.1, 1.0, 10.0)
PRIMARY_SCORING = "balanced_accuracy"
OUTER_FOLDS = 5
DEFAULT_SEED = 42
DEFAULT_BOOTSTRAP = 10000
#: Fixed-threshold sensitivity analysis (predeclared, not the primary threshold).
SENSITIVITY_THRESHOLD = 0.5
#: Threshold search rule and tie-breaking (lowest threshold wins ties).
THRESHOLD_SEARCH = "argmax balanced_accuracy over np.linspace(0.01, 0.99, 99)"
THRESHOLD_TIE_BREAK = "lowest threshold on ties (deterministic)"


def _v2_feature_names() -> tuple[str, ...]:
    """The predeclared 36 harmonized v2 features, in fixed column order."""
    bands = list(V2_BANDS.keys())
    regions = list(V2_REGIONS.keys())
    names: list[str] = []
    for band in bands:
        names.append(f"relpow_global__{band}")
    for region in regions:
        for band in bands:
            names.append(f"relpow_region__{band}__{region}")
    for num, den in V2_RATIOS:
        names.append(f"ratio_global__{num}_{den}")
    for region in regions:
        for num, den in V2_RATIOS:
            names.append(f"ratio_region__{num}_{den}__{region}")
    return tuple(names)


#: 36 v2 features (4 global + 20 regional relpowers + 2 + 10 ratios). Per-channel
#: columns are excluded; relative-power denominators span only the four 1-30 Hz bands.
V2_HARMONIZED_FEATURES: tuple[str, ...] = _v2_feature_names()
assert len(V2_HARMONIZED_FEATURES) == 36


# --- Harmonized feature extraction ------------------------------------------


def harmonized_features_from_signal_v2(
    channels_data: dict[str, np.ndarray],
    sfreq: float,
    nperseg: int,
) -> dict[str, float]:
    """Project one subject's channel signals onto the 36 v2 features.

    Same sum-over-bins Welch convention as :mod:`external_osf`, but restricted to
    the four 1-30 Hz common bands, so relative powers are fractions of 1-30 Hz
    power. Raises ``KeyError`` on schema drift, ``ValueError`` on a missing common
    channel, non-finite input, or a flat/zero-power channel (predeclared fail rules).
    """
    missing = [ch for ch in COMMON_CHANNELS_19 if ch not in channels_data]
    if missing:
        raise ValueError(f"missing common channels: {missing}")
    for ch in COMMON_CHANNELS_19:
        array = np.asarray(channels_data[ch], dtype=float)
        if not np.isfinite(array).all():
            raise ValueError(f"non-finite values in channel {ch}")
        if array.size > 1 and float(np.std(array)) == 0.0:
            raise ValueError(f"flat/zero-power channel: {ch}")
    full = extract_subject_features(
        channels_data,
        sfreq=sfreq,
        nperseg=nperseg,
        bands=V2_BANDS,
        ratios=V2_RATIOS,
        regions=V2_REGIONS,
        channels=COMMON_CHANNELS_19,
        average_reference=True,
    )
    return {name: float(full[name]) for name in V2_HARMONIZED_FEATURES}


def _select_v2_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return metadata + the 36 v2 feature columns in fixed order."""
    keep = ["participant_id", "group", "label", *V2_HARMONIZED_FEATURES]
    missing = [c for c in keep if c not in dataframe.columns]
    if missing:
        raise KeyError(f"v2 feature columns missing: {missing}")
    return dataframe[keep]


def build_discovery_harmonized_matrix_v2(
    cfg: dict,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Re-extract v2 (1-30 Hz) features for every ds004504 subject from clean epochs."""
    bids_root = project_path(cfg, cfg["paths"]["bids_root"])
    participants = load_participants(bids_root, cfg)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for _, participant in tqdm(
        participants.iterrows(), total=len(participants), desc="discovery subjects"
    ):
        participant_id = participant["participant_id"]
        try:
            epochs = make_epochs(bids_root, participant_id, cfg)
            data = epochs.get_data()
            if data.shape[0] == 0:
                raise ValueError("no clean epochs retained")
            channel_names = list(epochs.ch_names)
            concatenated = np.concatenate(data, axis=1)
            sfreq = float(epochs.info["sfreq"])
            nperseg = max(2, int(round(sfreq / TARGET_FREQ_RESOLUTION_HZ)))
            channels_data = {ch: concatenated[i] for i, ch in enumerate(channel_names)}
            feats = harmonized_features_from_signal_v2(channels_data, sfreq, nperseg)
        except Exception as exc:  # noqa: BLE001 - keep batch alive, report failures
            failures.append({"participant_id": participant_id, "error": repr(exc)})
            continue
        feats["participant_id"] = participant_id
        feats["group"] = participant.get("Group")
        feats["label"] = participant["label"]
        rows.append(feats)
    if not rows:
        raise RuntimeError("no discovery subjects extracted successfully")
    return _select_v2_columns(pd.DataFrame(rows)), failures


def build_osf_harmonized_matrix_v2(
    archive: str,
    condition: str = "Eyes_closed",
    sfreq: float = 128.0,
    nperseg: int = 256,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Re-extract v2 features for the OSF archive via the shared low-level function."""
    from .external_osf import build_feature_matrix

    dataframe, failures = build_feature_matrix(
        archive,
        condition=condition,
        sfreq=sfreq,
        channels=COMMON_CHANNELS_19,
        bands=V2_BANDS,
        ratios=V2_RATIOS,
        regions=V2_REGIONS,
        average_reference=True,
        nperseg=nperseg,
    )
    return _select_v2_columns(dataframe), failures


def build_osf_signal_qc(archive: str, condition: str = "Eyes_closed") -> pd.DataFrame:
    """Label-blind per-subject signal QC: finite values, sample count, channel
    completeness, flat/zero-power channels. Group is carried for reference only;
    no QC metric uses it."""
    summary = index_archive(archive, condition=condition)
    rows: list[dict[str, Any]] = []
    for row in summary.subjects:
        key = SubjectKey(group=row["group"], condition=condition, subject=row["subject"])
        record: dict[str, Any] = {
            "participant_id": row["participant_id"],
            "group": row["group"],
            "has_all_common_19": bool(row["has_all_common_19"]),
        }
        try:
            data = read_subject_channels(archive, key, COMMON_CHANNELS_19)
        except Exception as exc:  # noqa: BLE001 - report and continue
            record.update({"status": "fail", "error": repr(exc)})
            rows.append(record)
            continue
        sample_counts = {ch: int(data[ch].size) for ch in COMMON_CHANNELS_19}
        stds = {ch: float(np.std(data[ch])) for ch in COMMON_CHANNELS_19}
        all_finite = all(bool(np.isfinite(data[ch]).all()) for ch in COMMON_CHANNELS_19)
        n_flat = int(sum(1 for s in stds.values() if s == 0.0))
        record.update({
            "status": "pass",
            "samples_per_channel": int(set(sample_counts.values()).pop()) if len(set(sample_counts.values())) == 1 else -1,
            "expected_samples": EXPECTED_SAMPLES_PER_CHANNEL,
            "sample_count_uniform": bool(len(set(sample_counts.values())) == 1),
            "all_finite": all_finite,
            "n_flat_or_zero_power_channels": n_flat,
        })
        rows.append(record)
    return pd.DataFrame(rows)


# --- Classifier building blocks (fit on discovery only) ---------------------


def _base_pipeline(random_state: int, C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            C=C, max_iter=2000, class_weight="balanced", random_state=random_state,
        )),
    ])


def _cv_splits(n_splits: int, min_class: int) -> int:
    if min_class < 2:
        raise ValueError("need at least two samples per class for cross-validation")
    return min(n_splits, min_class)


def _best_threshold(y_true: np.ndarray, prob_pos: np.ndarray) -> tuple[float, float]:
    """Threshold maximizing balanced accuracy; lowest threshold wins ties."""
    best_threshold, best_score = 0.5, -1.0
    for threshold in np.linspace(0.01, 0.99, 99):
        score = balanced_accuracy_score(y_true, (prob_pos >= threshold).astype(int))
        if score > best_score:  # strict > => first (lowest) threshold kept on ties
            best_score, best_threshold = score, threshold
    return float(best_threshold), float(best_score)


def select_C(X: pd.DataFrame, y: np.ndarray, seed: int, fold: int = 0) -> float:
    """Choose regularization C via GridSearchCV on the given (training) data only."""
    min_class = int(np.bincount(y).min())
    inner = StratifiedKFold(
        n_splits=_cv_splits(OUTER_FOLDS, min_class), shuffle=True, random_state=seed + fold
    )
    search = GridSearchCV(
        _base_pipeline(seed), {"model__C": list(C_GRID)},
        cv=inner, scoring=PRIMARY_SCORING, refit=False,
    )
    search.fit(X, y)
    return float(search.best_params_["model__C"])


def crossfit_oof_proba(
    X: pd.DataFrame, y: np.ndarray, C: float, seed: int, fold: int = 0,
) -> np.ndarray:
    """Cross-fitted out-of-fold AD probabilities using a fixed C (training data only)."""
    min_class = int(np.bincount(y).min())
    skf = StratifiedKFold(
        n_splits=_cv_splits(OUTER_FOLDS, min_class), shuffle=True, random_state=seed + fold
    )
    oof = np.full(len(y), np.nan, dtype=float)
    for train_idx, test_idx in skf.split(X, y):
        model = _base_pipeline(seed, C).fit(X.iloc[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
    if not np.isfinite(oof).all():
        raise RuntimeError("cross-fitting left missing probabilities")
    return oof


def select_threshold_crossfit(
    X: pd.DataFrame, y: np.ndarray, C: float, seed: int, fold: int = 0,
) -> float:
    """Choose the decision threshold from cross-fitted OOF on the given training data."""
    oof_prob = crossfit_oof_proba(X, y, C, seed, fold)
    threshold, _ = _best_threshold(y, oof_prob)
    return float(threshold)


def nested_cv_internal_estimate(
    X: pd.DataFrame, y: np.ndarray, seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Unbiased internal nested-CV estimate.

    Per outer fold, C and threshold are selected on the outer-*training* data only
    (C via inner GridSearchCV; threshold via inner cross-fitted OOF), then frozen
    and applied to the outer-test fold. Outer-test labels never influence selection.
    """
    min_class = int(np.bincount(y).min())
    outer_splits = _cv_splits(OUTER_FOLDS, min_class)
    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    oof_rows: list[dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        fold_C = select_C(X_train, y_train, seed, fold)
        fold_threshold = select_threshold_crossfit(X_train, y_train, fold_C, seed, fold)
        model = _base_pipeline(seed, fold_C).fit(X_train, y_train)
        prob = model.predict_proba(X.iloc[test_idx])[:, 1]
        pred = (prob >= fold_threshold).astype(int)
        for position, idx in enumerate(test_idx):
            oof_rows.append({
                "fold": fold,
                "row_index": int(idx),
                "true_label": int(y[idx]),
                "prob_AD": float(prob[position]),
                "pred_label": int(pred[position]),
                "fold_C": float(fold_C),
                "fold_threshold": float(fold_threshold),
            })
    oof = pd.DataFrame(oof_rows)
    truth = oof["true_label"].to_numpy()
    pred = oof["pred_label"].to_numpy()
    prob = oof["prob_AD"].to_numpy()
    try:
        auc = float(roc_auc_score(truth, prob))
    except ValueError:
        auc = float("nan")
    return {
        "oof": oof,
        "outer_folds": outer_splits,
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "auc": auc,
        "n_train_total": int(len(y)),
    }


def fit_final_model_and_threshold(
    X: pd.DataFrame, y: np.ndarray, seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Final external model: C and threshold chosen by discovery-only cross-fitting,
    then the pipeline is refit on all ds004504 AD/HC."""
    best_C = select_C(X, y, seed, fold=0)
    crossfit_prob = crossfit_oof_proba(X, y, best_C, seed, fold=0)
    threshold, threshold_score = _best_threshold(y, crossfit_prob)
    pipeline = _base_pipeline(seed, best_C).fit(X, y)
    lr = pipeline.named_steps["model"]
    imputer = pipeline.named_steps["impute"]
    scaler = pipeline.named_steps["scale"]
    crossfit_oof = pd.DataFrame({
        "row_index": np.arange(len(y)),
        "true_label": y.astype(int),
        "prob_AD": crossfit_prob,
        "pred_label": (crossfit_prob >= threshold).astype(int),
        "fold_C": float(best_C),
        "fold_threshold": float(threshold),
    })
    return {
        "pipeline": pipeline,
        "classes": [int(c) for c in lr.classes_],
        "best_C": float(best_C),
        "threshold": float(threshold),
        "threshold_score": float(threshold_score),
        "crossfit_oof": crossfit_oof,
        "coefficients": dict(zip(V2_HARMONIZED_FEATURES, [float(c) for c in lr.coef_[0]])),
        "imputer_medians": [float(v) for v in imputer.statistics_],
        "scaler_means": [float(v) for v in scaler.mean_],
        "scaler_scales": [float(v) for v in scaler.scale_],
    }


def predict_external(
    fitted: dict[str, Any], external_df: pd.DataFrame, threshold: float | None = None,
) -> pd.DataFrame:
    """Predict AD probability per external subject. ``pred_label`` is model-derived
    (threshold on prob_AD); ``true_label`` is carried through untouched. No refit."""
    if threshold is None:
        threshold = fitted["threshold"]
    pipeline = fitted["pipeline"]
    X = external_df[list(V2_HARMONIZED_FEATURES)]
    probabilities = pipeline.predict_proba(X)
    ad_index = fitted["classes"].index(1)
    prob_ad = probabilities[:, ad_index]
    pred_positive = (prob_ad >= threshold).astype(int)
    return pd.DataFrame({
        "participant_id": external_df["participant_id"].to_numpy(),
        "true_label": external_df["label"].to_numpy(),
        "prob_AD": prob_ad,
        "threshold": float(threshold),
        "pred_label": np.where(pred_positive == 1, POS_LABEL, NEG_LABEL),
    })


# --- Metrics ----------------------------------------------------------------


def _recall(truth: np.ndarray, pred: np.ndarray, positive: int) -> float:
    mask = truth == positive
    denominator = int(mask.sum())
    if denominator == 0:
        return float("nan")
    return float(((pred == positive) & mask).sum() / denominator)


def point_metrics(truth_label, pred_label, prob_ad) -> dict[str, float]:
    """Point binary metrics. AD is the positive class."""
    truth = np.where(np.asarray(truth_label) == POS_LABEL, 1, 0)
    pred = np.where(np.asarray(pred_label) == POS_LABEL, 1, 0)
    prob = np.asarray(prob_ad, dtype=float)
    sensitivity = _recall(truth, pred, 1)
    specificity = _recall(truth, pred, 0)
    balanced = 0.5 * (sensitivity + specificity)
    try:
        auc = float(roc_auc_score(truth, prob))
    except ValueError:
        auc = float("nan")
    return {
        "balanced_accuracy": float(balanced),
        "accuracy": float((pred == truth).mean()),
        "roc_auc": auc,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tp": int(((pred == 1) & (truth == 1)).sum()),
        "fp": int(((pred == 1) & (truth == 0)).sum()),
        "tn": int(((pred == 0) & (truth == 0)).sum()),
        "fn": int(((pred == 0) & (truth == 1)).sum()),
    }


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """95% (default) Wilson score interval for a binomial proportion."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(center - margin), float(center + margin)


def subject_level_bootstrap_metrics(
    truth_label, prob_ad, pred_label,
    n_boot: int = DEFAULT_BOOTSTRAP, seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, float]]:
    """Subject-level class-stratified bootstrap 95% CIs for BA and AUC.

    Conditional on the fitted discovery model (does not account for discovery
    training-sample uncertainty). Raises ``ValueError`` if either class is absent.
    """
    truth = np.where(np.asarray(truth_label) == POS_LABEL, 1, 0)
    pred = np.where(np.asarray(pred_label) == POS_LABEL, 1, 0)
    prob = np.asarray(prob_ad, dtype=float)
    class_indices = [np.flatnonzero(truth == cls) for cls in (0, 1)]
    for cls, indices in zip(("HC", "AD"), class_indices):
        if indices.size == 0:
            raise ValueError(f"class {cls} absent; cannot stratify bootstrap")

    def _metrics(t: np.ndarray, p: np.ndarray, pr: np.ndarray) -> tuple[float, float]:
        bal = 0.5 * (_recall(t, p, 1) + _recall(t, p, 0))
        try:
            auc = float(roc_auc_score(t, pr))
        except ValueError:
            auc = float("nan")
        return float(bal), auc

    names = ["balanced_accuracy", "roc_auc"]
    point = _metrics(truth, pred, prob)
    rng = np.random.default_rng(seed)
    samples = np.empty((n_boot, 2), dtype=float)
    for boot in range(n_boot):
        idx = np.concatenate([
            rng.choice(indices, size=indices.size, replace=True) for indices in class_indices
        ])
        samples[boot] = _metrics(truth[idx], pred[idx], prob[idx])
    return {
        name: {
            "point": float(point[i]),
            "ci_low": float(np.percentile(samples[:, i], 2.5)),
            "ci_high": float(np.percentile(samples[:, i], 97.5)),
        }
        for i, name in enumerate(names)
    }


# --- Domain-shift audit -----------------------------------------------------


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def _sd(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if values.size > 1 else float("nan")


def _median(values: np.ndarray) -> float:
    return float(np.median(values)) if values.size else float("nan")


def _iqr(values: np.ndarray) -> float:
    if not values.size:
        return float("nan")
    q1, q3 = np.percentile(values, [25, 75])
    return float(q3 - q1)


def _standardized_mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD standardized mean difference (external b minus discovery a).

    Uses Cohen's d pooled SD: sqrt((sd_a^2 + sd_b^2) / 2). Returns NaN if either
    side is empty or both SDs are zero/non-finite.
    """
    if a.size == 0 or b.size == 0:
        return float("nan")
    sd_a, sd_b = _sd(a), _sd(b)
    if not (np.isfinite(sd_a) and np.isfinite(sd_b)):
        return float("nan")
    pooled = np.sqrt((sd_a ** 2 + sd_b ** 2) / 2.0)
    if pooled == 0 or not np.isfinite(pooled):
        return float("nan")
    return float((_mean(b) - _mean(a)) / pooled)


def _shift_row(feature: str, a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    try:
        ks = float(ks_2samp(a, b).statistic) if a.size and b.size else float("nan")
    except Exception:  # noqa: BLE001 - KS can fail on degenerate inputs
        ks = float("nan")
    return {
        "feature": feature,
        "n_discovery": int(a.size),
        "n_external": int(b.size),
        "mean_discovery": _mean(a),
        "mean_external": _mean(b),
        "sd_discovery": _sd(a),
        "sd_external": _sd(b),
        "median_discovery": _median(a),
        "median_external": _median(b),
        "iqr_discovery": _iqr(a),
        "iqr_external": _iqr(b),
        "standardized_mean_difference": _standardized_mean_difference(a, b),
        "ks_statistic": ks,
    }


def domain_shift_primary_labelfree(
    train_adhc_df: pd.DataFrame, external_df: pd.DataFrame,
) -> pd.DataFrame:
    """Primary label-free shift audit: discovery AD+HC vs external ALL.

    The external label column is never read. ``train_adhc_df`` must already be
    filtered to AD+HC by the caller. Descriptive only.
    """
    rows = [
        _shift_row(feature, train_adhc_df[feature].dropna().to_numpy(float),
                   external_df[feature].dropna().to_numpy(float))
        for feature in V2_HARMONIZED_FEATURES
    ]
    return pd.DataFrame(rows)


def domain_shift_supplementary_by_label(
    train_df: pd.DataFrame, external_df: pd.DataFrame,
) -> pd.DataFrame:
    """Supplementary label-aware shift (discovery-AD vs external-AD, discovery-HC vs
    external-HC). Label-aware descriptive only; computed after frozen predictions."""
    rows: list[dict[str, Any]] = []
    for label in (POS_LABEL, NEG_LABEL):
        a = train_df.loc[train_df["label"] == label]
        b = external_df.loc[external_df["label"] == label]
        for feature in V2_HARMONIZED_FEATURES:
            row = _shift_row(feature, a[feature].dropna().to_numpy(float),
                             b[feature].dropna().to_numpy(float))
            row["comparison"] = f"discovery_{label}_vs_external_{label}"
            rows.append(row)
    out = pd.DataFrame(rows)
    return out[["comparison", *out.columns[:-1].tolist()]]
