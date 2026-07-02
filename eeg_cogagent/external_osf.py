"""Local OSF archive parser and deterministic spectral feature extraction.

Phase 1 (external validation) only. This module reads the OSF EEG archive
(``EEG_data.zip`` from OSF node 2v5md) directly from the ZIP, audits the cohort,
validates the common 19-channel montage, and extracts Welch-PSD spectral
features. It never fits a model and never treats channels or conditions as
samples: one row is produced per subject.

Design notes
------------
* Direct-ZIP parsing. Members are read via ``zipfile`` in memory; the archive is
  never extracted to disk and never modified, so its SHA-256 is preserved.
* Path safety. Only members matching a strict regex over the expected layout are
  ever opened; every candidate member name is validated against zip-slip rules
  (no absolute paths, no ``..`` segments, no backslashes, no drive letters).
* Determinism. Channel iteration order, subject ordering, Welch parameters and
  column ordering are all fixed. There is no randomness anywhere in the path.
* Feature semantics. Bands match the discovery (ds004504) configuration exactly.
  Because the OSF text files carry no unit/calibration metadata and may originate
  from different acquisition systems than the discovery set, features are
  **relative powers and band-power ratios only** (scale-invariant), not the
  absolute log10 PSD used by the discovery pipeline. See ``feature_mapping()``.
"""

from __future__ import annotations

import hashlib
import io
import re
import sys
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.signal import welch

# --- Fixed constants matching the archive + discovery configuration ----------

#: Sampling frequency of the OSF archive (Hz). Documented by the dataset card.
OSF_SAMPLING_RATE_HZ: float = 128.0

#: Expected number of samples per channel file.
EXPECTED_SAMPLES_PER_CHANNEL: int = 1024

#: The 19 common 10-20 channels used for discovery-compatible features, in a
#: fixed canonical order. F1 and F2 are excluded (not present in the discovery
#: montage / dropped for comparability).
COMMON_CHANNELS_19: tuple[str, ...] = (
    "Fp1", "Fp2",
    "F3", "F4",
    "C3", "C4",
    "P3", "P4",
    "O1", "O2",
    "F7", "F8",
    "T3", "T4", "T5", "T6",
    "Fz", "Cz", "Pz",
)

#: Channels present in the archive but deliberately excluded from features.
EXCLUDED_CHANNELS: frozenset[str] = frozenset({"F1", "F2"})

#: Condition used for external validation.
DEFAULT_CONDITION: str = "Eyes_closed"

#: Groups present in the archive and their expected Eyes_closed subject counts.
EXPECTED_GROUP_COUNTS: dict[str, int] = {"AD": 80, "Healthy": 12}

#: Frequency bands. Edges are identical to configs/ds004504_minimal.yaml.
DEFAULT_BANDS: "OrderedDict[str, tuple[float, float]]" = OrderedDict([
    ("delta", (1.0, 4.0)),
    ("theta", (4.0, 8.0)),
    ("alpha", (8.0, 13.0)),
    ("beta", (13.0, 30.0)),
    ("gamma", (30.0, 45.0)),
])

#: Band-power ratios, identical to the discovery configuration.
DEFAULT_RATIOS: tuple[tuple[str, str], ...] = (("theta", "alpha"), ("delta", "alpha"))

#: Scalp regions restricted to the 19 common channels. The OSF archive uses the
#: classic 10-20 temporal/parietal names (T3/T4/T5/T6) rather than the 10-10
#: aliases (T7/T8/P7/P8), so the discovery temporal list is mapped accordingly.
DEFAULT_REGIONS: dict[str, tuple[str, ...]] = {
    "frontal": ("Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"),
    "temporal": ("T3", "T4", "T5", "T6"),
    "central": ("C3", "C4", "Cz"),
    "parietal": ("P3", "P4", "Pz"),
    "occipital": ("O1", "O2"),
}

#: Welch configuration. nperseg=256 at 128 Hz = 2 s segments; matches scipy/MNE
#: defaults and yields 4 non-overlapping segments over the 8 s record.
WELCH_NPERSEG: int = 256
WELCH_WINDOW: str = "hann"
WELCH_NOVERLAP: int = 0

#: License state for the OSF archive. Intentionally unresolved for phase 1.
LICENSE_STATUS: str = "UNRESOLVED"

#: Strict regex for the only layout we will ever open. The channel basename is
#: validated separately against the allowed channel set.
_MEMBER_RE: re.Pattern[str] = re.compile(
    r"^EEG_data/(AD|Healthy)/(Eyes_closed|Eyes_open)/(Paciente\d+)/([^/]+)\.txt$"
)


# --- Path safety -------------------------------------------------------------


def is_safe_member_name(name: str) -> bool:
    """Return True if a ZIP member name is safe to consider.

    Rejects absolute paths, drive letters, backslashes, ``..`` traversal and
    any non-relative segment. This is belt-and-braces: we additionally only ever
    open names matching ``_MEMBER_RE``.
    """
    if not name or name.startswith("/"):
        return False
    if "\\" in name or ":" in name:
        return False
    parts = name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return False
    return True


# --- Provenance --------------------------------------------------------------


def sha256_of_file(path: str | Path) -> str:
    """Stream a file and return its uppercase SHA-256 hex digest."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


# --- ZIP parsing -------------------------------------------------------------


@dataclass(frozen=True)
class SubjectKey:
    """A unique subject location inside the archive."""

    group: str        # "AD" | "Healthy"
    condition: str    # "Eyes_closed" | "Eyes_open"
    subject: str      # "PacienteN"

    @property
    def participant_id(self) -> str:
        # AD and Healthy both use Paciente1.. so the group must namespace the id.
        return f"{self.group}_{self.subject}"

    def member_for(self, channel: str) -> str:
        return f"EEG_data/{self.group}/{self.condition}/{self.subject}/{channel}.txt"


@dataclass
class CohortSummary:
    groups: dict[str, dict[str, int]] = field(default_factory=dict)
    subjects: list[dict[str, Any]] = field(default_factory=list)
    channel_sets: dict[str, list[str]] = field(default_factory=dict)
    ignored_entries: list[str] = field(default_factory=list)
    sample_counts: dict[str, dict[str, int]] = field(default_factory=dict)


def _open_zip(path: str | Path) -> zipfile.ZipFile:
    return zipfile.ZipFile(str(path), mode="r")


def list_archive_members(path: str | Path) -> list[str]:
    """Return the sorted list of safe, non-directory member names in the archive."""
    with _open_zip(path) as archive:
        names = [n for n in archive.namelist() if is_safe_member_name(n) and not n.endswith("/")]
    return sorted(names)


def _member_subject_key(name: str) -> tuple[SubjectKey, str] | None:
    """Map a member name to ``(SubjectKey, channel)`` or ``None`` if it is not a
    subject channel file."""
    match = _MEMBER_RE.match(name)
    if match is None:
        return None
    group, condition, subject, channel = match.groups()
    return SubjectKey(group=group, condition=condition, subject=subject), channel


def index_archive(path: str | Path, condition: str = DEFAULT_CONDITION) -> CohortSummary:
    """Build a cohort summary of the archive for ``condition``.

    The summary enumerates subjects, their channel sets, sample counts per
    channel, and any archive members that were ignored (not matching the
    expected layout). Sample counts require reading each file; pass
    ``inspect_samples=False`` via :func:`cohort_audit` to skip when only
    structural counts are needed.
    """
    summary = CohortSummary()
    # channels per (group, subject) across *all* conditions, for completeness.
    channels_by_subject: dict[tuple[str, str], set[str]] = {}
    condition_subjects: dict[tuple[str, str], set[str]] = {}

    with _open_zip(path) as archive:
        all_names = archive.namelist()
        for name in all_names:
            mapped = _member_subject_key(name)
            if mapped is None:
                if name and not name.endswith("/") and is_safe_member_name(name):
                    summary.ignored_entries.append(name)
                continue
            key, channel = mapped
            channels_by_subject.setdefault((key.group, key.subject), set()).add(channel)
            condition_subjects.setdefault((key.group, key.condition), set()).add(key.subject)

    # Group counts for the requested condition.
    groups: dict[str, dict[str, int]] = {}
    for (group, cond), subs in condition_subjects.items():
        if cond != condition:
            continue
        groups.setdefault(group, {"subjects": 0})["subjects"] = len(subs)
    summary.groups = {g: groups[g] for g in sorted(groups)}

    # Per-subject rows for the requested condition, with channel set.
    for (group, subject), channels in sorted(
        channels_by_subject.items(), key=lambda kv: (kv[0][0], _paciente_num(kv[0][1]))
    ):
        # Only include subjects that have the requested condition present.
        if subject not in condition_subjects.get((group, condition), set()):
            continue
        missing_common = [c for c in COMMON_CHANNELS_19 if c not in channels]
        extra = sorted(channels - set(COMMON_CHANNELS_19) - EXCLUDED_CHANNELS)
        summary.subjects.append({
            "participant_id": f"{group}_{subject}",
            "group": group,
            "subject": subject,
            "condition": condition,
            "n_channels_present": len(channels),
            "has_all_common_19": len(missing_common) == 0,
            "missing_common_19": ",".join(missing_common),
            "ex_channels_present": ",".join(sorted(channels & EXCLUDED_CHANNELS)),
            "unexpected_channels": ",".join(extra),
        })
        cs = sorted(channels)
        summary.channel_sets.setdefault(group, [])
        if cs not in summary.channel_sets[group]:
            summary.channel_sets[group].append(cs)

    return summary


def _paciente_num(subject: str) -> int:
    """Extract the integer index from ``PacienteN`` for deterministic sorting."""
    match = re.search(r"\d+", subject)
    return int(match.group()) if match else 0


def _parse_channel_bytes(raw: bytes, expected: int | None = EXPECTED_SAMPLES_PER_CHANNEL) -> tuple[np.ndarray, int]:
    """Parse a single channel text file (one float per line) into a 1-D array.

    Returns ``(values, n_parsed)``. Blank lines are skipped. Non-numeric lines
    raise ``ValueError`` so callers can record the failure rather than silently
    corrupting features.
    """
    text = raw.decode("utf-8", errors="strict")
    values: list[float] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        values.append(float(line))
    if not values:
        raise ValueError("channel file is empty")
    arr = np.asarray(values, dtype=np.float64)
    if expected is not None and arr.size != expected:
        raise ValueError(f"channel has {arr.size} samples, expected {expected}")
    return arr, arr.size


def read_subject_channels(
    path: str | Path,
    key: SubjectKey,
    channels: Sequence[str],
) -> dict[str, np.ndarray]:
    """Read the requested channels for one subject directly from the ZIP.

    Raises ``KeyError`` if a channel member is absent and ``ValueError`` if a
    file cannot be parsed or has the wrong sample count.
    """
    out: dict[str, np.ndarray] = {}
    wanted = set(channels)
    with _open_zip(path) as archive:
        for channel in channels:
            member = key.member_for(channel)
            try:
                raw = archive.read(member)
            except KeyError as exc:  # pragma: no cover - re-raised with context
                raise KeyError(f"missing channel {channel} for {key.participant_id}") from exc
            values, _ = _parse_channel_bytes(raw)
            out[channel] = values
            wanted.discard(channel)
    if wanted:
        missing = sorted(wanted)
        raise KeyError(f"missing channels {missing} for {key.participant_id}")
    return out


# --- Audit -------------------------------------------------------------------


def cohort_audit(
    path: str | Path,
    condition: str = DEFAULT_CONDITION,
    expected_groups: Mapping[str, int] | None = None,
    inspect_samples: bool = True,
) -> dict[str, Any]:
    """Audit the cohort and return a structured report.

    When ``inspect_samples`` is True, every channel file for every subject in
    ``condition`` is read to confirm the sample count and parse-ability. This is
    the integrity check used before feature extraction.
    """
    expected_groups = dict(expected_groups or EXPECTED_GROUP_COUNTS)
    summary = index_archive(path, condition=condition)

    checks: list[dict[str, Any]] = []

    def add(check_id: str, status: str, detail: str) -> None:
        checks.append({"id": check_id, "status": status, "detail": detail})

    total_subjects = sum(g["subjects"] for g in summary.groups.values())
    add(
        "cohort:total-subjects",
        "pass" if total_subjects == sum(expected_groups.values()) else "fail",
        f"total={total_subjects} expected={sum(expected_groups.values())}",
    )

    for group, expected in sorted(expected_groups.items()):
        got = summary.groups.get(group, {"subjects": 0})["subjects"]
        add(
            f"cohort:group:{group}",
            "pass" if got == expected else "fail",
            f"{group}={got} expected={expected}",
        )

    all_common = all(row["has_all_common_19"] for row in summary.subjects)
    missing = [r["participant_id"] for r in summary.subjects if not r["has_all_common_19"]]
    add(
        "channels:common-19-present",
        "pass" if all_common else "fail",
        f"subjects_missing_common_19={missing}" if missing else "all subjects have common 19",
    )

    unexpected = [r["participant_id"] for r in summary.subjects if r["unexpected_channels"]]
    add(
        "channels:no-unexpected",
        "pass" if not unexpected else "warn",
        f"subjects_with_unexpected_channels={unexpected}" if unexpected else "none",
    )

    add(
        "channels:f1f2-excluded",
        "pass",
        f"excluded={sorted(EXCLUDED_CHANNELS)}",
    )

    # Channel-set uniformity within each group.
    uniform = True
    for group, sets in summary.channel_sets.items():
        if len(sets) > 1:
            uniform = False
    add(
        "channels:uniform-within-group",
        "pass" if uniform else "warn",
        "channel sets per group: " + ", ".join(f"{g}={len(s)}" for g, s in summary.channel_sets.items()),
    )

    sample_info: dict[str, Any] = {}
    if inspect_samples:
        sample_counts: dict[str, dict[str, int]] = {}
        bad: list[dict[str, str]] = []
        with _open_zip(path) as archive:
            for row in summary.subjects:
                key = SubjectKey(group=row["group"], condition=condition, subject=row["subject"])
                per_channel: dict[str, int] = {}
                for channel in COMMON_CHANNELS_19:
                    member = key.member_for(channel)
                    try:
                        _, n = _parse_channel_bytes(archive.read(member))
                    except Exception as exc:  # noqa: BLE001 - report, do not abort
                        bad.append({"participant_id": row["participant_id"], "channel": channel, "error": repr(exc)})
                        per_channel[channel] = -1
                        continue
                    per_channel[channel] = int(n)
                sample_counts[row["participant_id"]] = per_channel
        sample_info = {
            "expected_samples_per_channel": EXPECTED_SAMPLES_PER_CHANNEL,
            "counts": sample_counts,
            "failures": bad,
        }
        ok_samples = not bad and all(
            n == EXPECTED_SAMPLES_PER_CHANNEL
            for per in sample_counts.values()
            for n in per.values()
        )
        add(
            "integrity:sample-counts",
            "pass" if ok_samples else "fail",
            f"failures={len(bad)}; all channels = {EXPECTED_SAMPLES_PER_CHANNEL} samples"
            if ok_samples
            else f"failures={len(bad)}",
        )

    status_counts = {
        s: sum(c["status"] == s for c in checks)
        for s in ("pass", "warn", "fail")
    }
    overall = "fail" if status_counts["fail"] else ("warn" if status_counts["warn"] else "pass")
    return {
        "status": overall,
        "status_counts": status_counts,
        "condition": condition,
        "groups": summary.groups,
        "total_subjects": total_subjects,
        "expected_group_counts": expected_groups,
        "channel_sets": summary.channel_sets,
        "subjects": summary.subjects,
        "ignored_entries": summary.ignored_entries,
        "checks": checks,
        "sample_integrity": sample_info,
    }


# --- Spectral features -------------------------------------------------------


def _band_mask(freqs: np.ndarray, low: float, high: float) -> np.ndarray:
    """Half-open frequency mask ``[low, high)`` matching discovery semantics."""
    return (freqs >= low) & (freqs < high)


def _channel_band_powers(
    signal: np.ndarray,
    sfreq: float,
    bands: Mapping[str, tuple[float, float]],
    nperseg: int = WELCH_NPERSEG,
) -> dict[str, float]:
    """Return absolute band power per band for one channel via Welch PSD.

    Power is the sum of PSD density bins inside the (half-open) band. The bin
    width is common across bands and cancels in ratios / relative powers, so it
    is not multiplied back in; the value is proportional to band energy.
    """
    nperseg_eff = min(nperseg, signal.size)
    freqs, psd = welch(
        signal,
        fs=sfreq,
        window=WELCH_WINDOW,
        nperseg=nperseg_eff,
        noverlap=min(WELCH_NOVERLAP, nperseg_eff - 1) if nperseg_eff > 1 else 0,
        detrend="constant",
        scaling="density",
    )
    powers: dict[str, float] = {}
    for band, (low, high) in bands.items():
        mask = _band_mask(freqs, low, high)
        powers[band] = float(np.sum(psd[mask])) if mask.any() else 0.0
    return powers


def _safe_log_ratio(numerator: float, denominator: float) -> float:
    if numerator <= 0.0 or denominator <= 0.0:
        return float("nan")
    return float(np.log10(numerator / denominator))


def _average_reference(channel_matrix: np.ndarray) -> np.ndarray:
    """Subtract the instantaneous mean across channels (discovery reference)."""
    return channel_matrix - channel_matrix.mean(axis=0, keepdims=True)


def extract_subject_features(
    channels_data: Mapping[str, np.ndarray],
    sfreq: float = OSF_SAMPLING_RATE_HZ,
    bands: Mapping[str, tuple[float, float]] | None = None,
    ratios: Sequence[tuple[str, str]] = DEFAULT_RATIOS,
    regions: Mapping[str, Sequence[str]] = DEFAULT_REGIONS,
    channels: Sequence[str] = COMMON_CHANNELS_19,
    average_reference: bool = True,
    nperseg: int = WELCH_NPERSEG,
) -> dict[str, float]:
    """Extract deterministic relative-power + ratio features for one subject.

    The feature schema mirrors the discovery pipeline (per-channel, global and
    region aggregates, plus global/region ratios) but replaces absolute log10
    PSD with scale-invariant relative power. See module docstring.

    ``nperseg`` is forwarded to the Welch estimator so the same scale-invariant
    convention can be applied to recordings at a different sampling rate (e.g.
    the 500 Hz discovery set at nperseg=1000 keeps the 0.5 Hz resolution of the
    128 Hz OSF archive at nperseg=256). The default preserves phase-1 behaviour.
    """
    bands = OrderedDict(bands or DEFAULT_BANDS)
    channels = [c for c in channels if c in channels_data]
    if not channels:
        raise ValueError("no usable channels supplied")

    n_samples = {c: int(channels_data[c].size) for c in channels}
    sample_set = set(n_samples.values())
    if len(sample_set) != 1:
        raise ValueError(f"inconsistent sample counts across channels: {n_samples}")
    n_samp = sample_set.pop()

    matrix = np.vstack([channels_data[c] for c in channels])
    if average_reference:
        matrix = _average_reference(matrix)

    # Per-channel absolute band powers: channel -> band -> power.
    per_channel: dict[str, dict[str, float]] = {}
    for idx, channel in enumerate(channels):
        per_channel[channel] = _channel_band_powers(matrix[idx], sfreq=sfreq, bands=bands, nperseg=nperseg)

    features: dict[str, float] = {}
    # Per-channel relative power = band_abs / total_abs (sum over all bands).
    relpow_ch: dict[str, dict[str, float]] = {b: {} for b in bands}
    for channel in channels:
        total = sum(per_channel[channel][b] for b in bands)
        for b in bands:
            rel = per_channel[channel][b] / total if total > 0 else float("nan")
            relpow_ch[b][channel] = rel
            features[f"relpow_ch__{b}__{channel}"] = float(rel)

    # Global + region aggregates for relative power (mean over channels).
    for b in bands:
        values = [relpow_ch[b][c] for c in channels]
        features[f"relpow_global__{b}"] = float(np.nanmean(values))
    for region, region_channels in regions.items():
        present = [c for c in region_channels if c in channels]
        if not present:
            continue
        for b in bands:
            values = [relpow_ch[b][c] for c in present]
            features[f"relpow_region__{b}__{region}"] = float(np.nanmean(values))

    # Ratios: log10(P_num / P_den). Aggregated as log of mean band powers, to
    # keep a single, explicit definition (documented in feature_mapping()).
    band_mean_power: dict[str, dict[str, float]] = {
        level: {b: float(np.nanmean([per_channel[c][b] for c in scope])) for b in bands}
        for level, scope in (
            ("global", list(channels)),
            *[(f"region__{r}", [c for c in rc if c in channels]) for r, rc in regions.items()],
        )
    }
    for num, den in ratios:
        if num not in bands or den not in bands:
            continue
        features[f"ratio_global__{num}_{den}"] = _safe_log_ratio(
            band_mean_power["global"][num], band_mean_power["global"][den]
        )
        for region, region_channels in regions.items():
            scope = [c for c in region_channels if c in channels]
            if not scope:
                continue
            features[f"ratio_region__{num}_{den}__{region}"] = _safe_log_ratio(
                band_mean_power[f"region__{region}"][num],
                band_mean_power[f"region__{region}"][den],
            )

    features["n_samples"] = int(n_samp)
    return features


# --- Feature matrix + mapping ------------------------------------------------


def build_feature_matrix(
    path: str | Path,
    condition: str = DEFAULT_CONDITION,
    sfreq: float = OSF_SAMPLING_RATE_HZ,
    channels: Sequence[str] = COMMON_CHANNELS_19,
    bands: Mapping[str, tuple[float, float]] | None = None,
    ratios: Sequence[tuple[str, str]] = DEFAULT_RATIOS,
    regions: Mapping[str, Sequence[str]] = DEFAULT_REGIONS,
    average_reference: bool = True,
    nperseg: int = WELCH_NPERSEG,
    fail_fast: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Build the one-row-per-subject feature matrix for ``condition``.

    Returns ``(dataframe, failures)``. Failed subjects are recorded rather than
    raised unless ``fail_fast`` is True.
    """
    bands = OrderedDict(bands or DEFAULT_BANDS)
    summary = index_archive(path, condition=condition)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with _open_zip(path) as archive:
        for row in summary.subjects:
            key = SubjectKey(group=row["group"], condition=condition, subject=row["subject"])
            try:
                channels_data: dict[str, np.ndarray] = {}
                for channel in channels:
                    raw = archive.read(key.member_for(channel))
                    values, _ = _parse_channel_bytes(raw)
                    channels_data[channel] = values
                feats = extract_subject_features(
                    channels_data,
                    sfreq=sfreq,
                    bands=bands,
                    ratios=ratios,
                    regions=regions,
                    channels=channels,
                    average_reference=average_reference,
                    nperseg=nperseg,
                )
            except Exception as exc:  # noqa: BLE001 - record and continue
                failures.append({"participant_id": row["participant_id"], "error": repr(exc)})
                if fail_fast:
                    raise
                continue
            feats = {
                "participant_id": row["participant_id"],
                "group": row["group"],
                "label": "AD" if row["group"] == "AD" else "HC",
                **feats,
            }
            rows.append(feats)

    if not rows:
        raise RuntimeError("no subjects extracted successfully")
    dataframe = pd.DataFrame(rows)
    return dataframe, failures


def feature_column_order(n_bands: int, n_channels: int, regions: Mapping[str, Sequence[str]], ratios: Sequence[tuple[str, str]]) -> list[str]:
    """Deterministic feature column ordering (excluding metadata)."""
    columns: list[str] = []
    # This helper is informational; pandas builds columns from row dict order,
    # which is already deterministic. Kept for documentation/tests.
    bands = list(DEFAULT_BANDS.keys())
    for b in bands:
        for ch in COMMON_CHANNELS_19[:n_channels]:
            columns.append(f"relpow_ch__{b}__{ch}")
    for b in bands:
        columns.append(f"relpow_global__{b}")
    for region in regions:
        for b in bands:
            columns.append(f"relpow_region__{b}__{region}")
    for num, den in ratios:
        columns.append(f"ratio_global__{num}_{den}")
        for region in regions:
            columns.append(f"ratio_region__{num}_{den}__{region}")
    columns.append("n_samples")
    return columns


def feature_mapping() -> dict[str, Any]:
    """Document the OSF feature semantics and their relation to discovery.

    The bands and ratios are identical to ``configs/ds004504_minimal.yaml``.
    The amplitude semantics differ: discovery uses log10(absolute PSD); OSF uses
    relative power (band / total within 1-45 Hz) and log10 band-power ratios, so
    that features are invariant to the archive's unknown amplitude calibration.
    """
    return {
        "archive": {
            "osf_node": "2v5md",
            "file": "EEG_data.zip",
            "sampling_rate_hz": OSF_SAMPLING_RATE_HZ,
            "samples_per_channel": EXPECTED_SAMPLES_PER_CHANNEL,
            "condition_used": DEFAULT_CONDITION,
            "license_status": LICENSE_STATUS,
        },
        "preprocessing": {
            "average_reference": {
                "applied": True,
                "channels": list(COMMON_CHANNELS_19),
                "note": "Matches discovery 'reference: average' across the montage.",
            },
            "bandpass_filter": {
                "applied": False,
                "reason": (
                    "Only 8 s (1024 samples) of resting data per channel; a designed "
                    "FIR bandpass would introduce edge artifacts. Welch detrend="
                    "'constant' removes per-segment DC instead."
                ),
            },
            "notch_filter": {"applied": False, "reason": "Bands restricted to <=45 Hz; 50/60 Hz line noise is out of band."},
            "epoching_rejection": {
                "applied": False,
                "reason": "Single contiguous 8 s record per channel; no artifact rejection in phase 1.",
            },
        },
        "welch": {
            "nperseg": WELCH_NPERSEG,
            "window": WELCH_WINDOW,
            "noverlap": WELCH_NOVERLAP,
            "detrend": "constant",
            "scaling": "density",
            "frequency_resolution_hz": OSF_SAMPLING_RATE_HZ / WELCH_NPERSEG,
        },
        "bands": {b: list(edges) for b, edges in DEFAULT_BANDS.items()},
        "ratios": [list(r) for r in DEFAULT_RATIOS],
        "regions": {k: list(v) for k, v in DEFAULT_REGIONS.items()},
        "channels_used": list(COMMON_CHANNELS_19),
        "channels_excluded": sorted(EXCLUDED_CHANNELS),
        "feature_schema": {
            "relpow_ch__{band}__{channel}": "Per-channel relative power: band_abs_power / total_abs_power (1-45 Hz). Scale-invariant. Analog of discovery 'band_ch__{band}__{channel}' which used log10 absolute PSD.",
            "relpow_global__{band}": "Mean over channels of per-channel relative power. Analog of 'band_global__{band}'.",
            "relpow_region__{band}__{region}": "Mean over region channels of per-channel relative power. Analog of 'band_region__{band}__{region}'.",
            "ratio_global__{num}_{den}": "log10(mean P_num / mean P_den) over channels. Scale-invariant. Analog of 'ratio_global__{num}_{den}'.",
            "ratio_region__{num}_{den}__{region}": "log10(mean P_num / mean P_den) within region. Analog of 'ratio_region__{num}_{den}__{region}'.",
            "n_samples": "Number of samples per channel used (provenance; analog of discovery 'n_epochs').",
        },
        "mismatch_with_discovery": [
            "Amplitude units: discovery uses log10(absolute Welch PSD uV^2/Hz); OSF uses relative power (proportion in [0,1]) because the archive text files carry no unit/calibration metadata and may originate from a different acquisition system.",
            "Aggregation of ratios: discovery differences mean-of-log-power across channels; OSF takes log of mean-power across channels. Both are scale-invariant; they agree only when channel powers are equal. This is documented and deterministic.",
            "Preprocessing: discovery bandpass (1-45 Hz), 50 Hz notch, 4 s epochs with amplitude rejection, average reference. OSF applies average reference only (see 'preprocessing'). No filtering or epoch rejection is performed on the 8 s records.",
            "Temporal nomenclature: OSF uses T3/T4/T5/T6 (10-20); discovery regions list T7/T8/P7/P8 (10-10 aliases). These refer to the same electrodes and are mapped accordingly.",
            "Label space: discovery labels {AD, FTD, HC}; OSF groups {AD, Healthy}, mapped to label 'AD' and 'HC' for downstream compatibility. No FTD exists in this archive.",
        ],
    }


# --- Canonical signal fingerprint + cluster audit (v3) ----------------------
#
# Label-free content hash over the 19 common channels in fixed order; written
# into the digest as a schema version string. Used to detect exact-signal
# duplicates that share one canonical archive member across multiple nominal
# folder IDs. Does NOT read labels, predictions, or probabilities to form a
# cluster; that is enforced by the doc test and read-only code review.


#: Schema-version tag written into the digest; bump on channel-set / dtype /
#: byte-order / sample-count-encoding changes.
#:
#: v1 (``osf-common19-float64-v1``) claimed to hash parsed little-endian
#: float64 sample bytes but in fact hashed the raw ZIP text bytes and used the
#: text byte length as the "sample count". v2 fixes this: each channel is parsed
#: to float64 (strict, one float per line), hard-checked for exactly
#: ``EXPECTED_SAMPLES_PER_CHANNEL`` finite samples, and written as explicit
#: little-endian (``<f8``) contiguous bytes with the sample count as a fixed
#: little-endian int64. The cluster structure is unchanged (identical signals
#: still hash identically), but individual digest strings differ from v1, so the
#: version tag is bumped as a future-break. See repair log.
FINGERPRINT_VERSION: str = "osf-common19-float64-v2"


def _fingerprint_one_subject(channel_bytes: dict[str, bytes], schema_version: str = FINGERPRINT_VERSION) -> str:
    """SHA-256 over ``schema_version(UTF-8) + ch_name(UTF-8) + sample_count(int64 LE)
    + float64(<f8) bytes`` for each ``COMMON_CHANNELS_19`` channel in strict fixed order.

    The raw ZIP text bytes are first parsed to float64 (strict: one float per
    line, blank lines skipped) and hard-checked for exactly
    ``EXPECTED_SAMPLES_PER_CHANNEL`` finite samples. The sample count is written
    as a fixed-width little-endian int64 and the values as explicit
    little-endian ``<f8`` contiguous bytes, so the digest is independent of text
    whitespace / decimal formatting and reproducible across machine endianness,
    while any single-sample numeric change alters it. Channel iteration order is
    always ``COMMON_CHANNELS_19`` regardless of the input dict's iteration order.
    """
    digest = hashlib.sha256()
    digest.update(schema_version.encode("utf-8"))
    for channel in COMMON_CHANNELS_19:
        values, n_samples = _parse_channel_bytes(
            channel_bytes[channel], expected=EXPECTED_SAMPLES_PER_CHANNEL
        )
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite values in fingerprint channel {channel}")
        digest.update(channel.encode("utf-8"))
        digest.update(np.array(n_samples, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(values, dtype="<f8").tobytes())
    return digest.hexdigest().upper()


def compute_signal_fingerprint(
    archive: str | Path, condition: str = DEFAULT_CONDITION,
    schema_version: str = FINGERPRINT_VERSION,
) -> list[dict[str, str]]:
    """Label-free: one row per nominal folder-ID with participant_id, group,
    condition, signal_sha256 computed in the canonical channel order."""
    summary = index_archive(archive, condition=condition)
    fingerprint_rows: list[dict[str, str]] = []
    with _open_zip(archive) as archive_zip:
        for row in summary.subjects:
            key = SubjectKey(group=row["group"], condition=condition, subject=row["subject"])
            per_channel: dict[str, bytes] = {}
            for channel in COMMON_CHANNELS_19:
                per_channel[channel] = archive_zip.read(key.member_for(channel))
            fingerprint_rows.append({
                "participant_id": f"{row['group']}_{row['subject']}",
                "group": row["group"],
                "condition": row["condition"],
                "signal_sha256": _fingerprint_one_subject(per_channel, schema_version),
            })
    return fingerprint_rows


def cluster_signal_fingerprints(
    rows: list[dict[str, str]],
    excluded_channels_for_conflict: tuple[str, ...] = ("group",),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Cluster rows by ``signal_sha256`` (no label/probability fields are read
    to form a cluster). Hard-fails on label/group conflict inside a cluster.

    Returns ``(audit_df, cluster_summary)``. ``representative_id`` is the
    lexicographically smallest ``participant_id`` in the cluster; ties (which
    do not arise in this archive because folder IDs are unique) would still be
    deterministic.
    """
    fingerprint_rows = list(rows)  # do not mutate caller's list
    digest_to_members: dict[str, list[dict[str, str]]] = {}
    for row in fingerprint_rows:
        digest_to_members.setdefault(row["signal_sha256"], []).append(row)
    per_digest_representative: dict[str, str] = {
        digest: min(member["participant_id"] for member in members)
        for digest, members in digest_to_members.items()
    }
    audit_rows: list[dict[str, Any]] = []
    for row in fingerprint_rows:
        fingerprint = row["signal_sha256"]
        members = digest_to_members[fingerprint]
        for field in excluded_channels_for_conflict:
            distinct = {member[field] for member in members}
            if len(distinct) > 1:
                raise ValueError(
                    f"label conflict in cluster {fingerprint}: {field} values {distinct} on members "
                    f"{sorted(member['participant_id'] for member in members)}"
                )
        audit_rows.append({
            "participant_id": row["participant_id"],
            "group": row["group"],
            "condition": row["condition"],
            "signal_sha256": fingerprint,
            "cluster_id": fingerprint,
            "cluster_size": len(members),
            "representative_id": per_digest_representative[fingerprint],
            "included_primary": row["participant_id"] == per_digest_representative[fingerprint],
            "exclusion_reason": "" if row["participant_id"] == per_digest_representative[fingerprint]
                               else f"exact_signal_duplicate_of:{per_digest_representative[fingerprint]}",
        })
    audit_df = pd.DataFrame(audit_rows)
    cluster_sizes = [len(members) for members in digest_to_members.values()]
    summary = {
        "nominal_count": int(audit_df["participant_id"].nunique()),
        "unique_fingerprint_count": int(audit_df["signal_sha256"].nunique()),
        "duplicate_cluster_count": int(sum(1 for size in cluster_sizes if size > 1)),
        "clusters": {
            digest: sorted(member["participant_id"] for member in members)
            for digest, members in digest_to_members.items()
        },
    }
    return audit_df, summary
