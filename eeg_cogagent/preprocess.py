from __future__ import annotations

from pathlib import Path


def make_epochs(bids_root: Path, participant_id: str, cfg: dict):
    import mne
    from mne_bids import BIDSPath, read_raw_bids

    prep = cfg["preprocessing"]
    subject = participant_id.removeprefix("sub-")
    bids_path = BIDSPath(
        root=bids_root,
        subject=subject,
        task=cfg["project"]["task"],
        datatype="eeg",
        suffix="eeg",
    )

    raw = read_raw_bids(bids_path, verbose="ERROR")
    raw.load_data()
    montage = mne.channels.make_standard_montage("standard_1020")
    if "eeg" not in raw.get_channel_types():
        standard_channels = set(montage.ch_names)
        recoverable = {
            channel: "eeg"
            for channel, channel_type in zip(raw.ch_names, raw.get_channel_types())
            if channel_type == "misc" and channel in standard_channels
        }
        if recoverable:
            raw.set_channel_types(recoverable, verbose="ERROR")
    raw.pick(picks="eeg", exclude=[])
    raw.filter(prep["l_freq"], prep["h_freq"], verbose="ERROR")

    notch_freqs = prep.get("notch_freqs") or []
    if notch_freqs:
        raw.notch_filter(notch_freqs, verbose="ERROR")

    if prep.get("reference", "average") == "average":
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    raw.set_montage(montage, on_missing="ignore", verbose="ERROR")

    max_minutes = prep.get("max_minutes")
    if max_minutes:
        raw.crop(tmax=min(float(raw.times[-1]), float(max_minutes) * 60.0))

    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=float(prep["epoch_length"]),
        preload=True,
        reject_by_annotation=True,
        verbose="ERROR",
    )

    reject_uv = prep.get("reject_uv")
    if reject_uv:
        epochs.drop_bad(reject={"eeg": float(reject_uv) * 1e-6}, verbose="ERROR")
    return epochs
