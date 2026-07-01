from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_participants(bids_root: Path, cfg: dict) -> pd.DataFrame:
    participants_cfg = cfg["participants"]
    table = pd.read_csv(bids_root / participants_cfg["file"], sep="\t")
    table.columns = [str(column).strip() for column in table.columns]
    label_col = participants_cfg["label_column"]
    label_map = participants_cfg.get("label_map", {})
    table["label"] = table[label_col].map(label_map).fillna(table[label_col])
    return table


def subject_code(participant_id: str) -> str:
    return participant_id.removeprefix("sub-")


def baseline_table(participants: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in participants.groupby("label"):
        row = {"group": label, "n": len(group)}
        if "Age" in group:
            row["age_mean"] = round(group["Age"].astype(float).mean(), 2)
            row["age_sd"] = round(group["Age"].astype(float).std(), 2)
        if "MMSE" in group:
            row["mmse_mean"] = round(group["MMSE"].astype(float).mean(), 2)
            row["mmse_sd"] = round(group["MMSE"].astype(float).std(), 2)
        if "Gender" in group:
            row["female_n"] = int((group["Gender"] == "F").sum())
            row["male_n"] = int((group["Gender"] == "M").sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("group")
