from __future__ import annotations

from pathlib import Path


def workflow_checklist(cfg: dict) -> str:
    dataset = cfg["project"]["dataset_id"]
    task = cfg["project"]["task"]
    bands = ", ".join(cfg["features"]["bands"].keys())
    models = ", ".join(cfg["ml"]["models"])
    connectivity = cfg.get("connectivity", {})
    connectivity_summary = ", ".join(connectivity.get("metrics", [])) or "disabled"
    return f"""# EEG-CogAgent Workflow Checklist

- Dataset: {dataset}
- BIDS task: {task}
- Preprocessing: {cfg['preprocessing']['l_freq']}-{cfg['preprocessing']['h_freq']} Hz, notch {cfg['preprocessing'].get('notch_freqs')}, average reference, fixed-length epochs
- Features: {bands}, theta/alpha, delta/alpha, regional summaries
- Connectivity: {connectivity_summary}; fixed-density wPLI graph metrics
- Statistics: Kruskal-Wallis + FDR
- Models: {models}, nested stratified CV
- Report: baseline table, feature statistics, connectivity networks, model table, publication figures

Agent rule: never present model output as diagnosis. Present it as reproducible biomarker-screening evidence on public research data.
"""


def write_agent_plan(cfg: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "agent_plan.md"
    out.write_text(workflow_checklist(cfg), encoding="utf-8")
    return out
