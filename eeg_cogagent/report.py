from __future__ import annotations

from pathlib import Path

import pandas as pd
from jinja2 import Template


REPORT_TEMPLATE = """# {{ title }}

## Cohort

{{ baseline }}

## Automated Methods Draft

Resting-state eyes-closed EEG data were loaded from a BIDS-formatted public dataset. Signals were band-pass filtered from {{ l_freq }} to {{ h_freq }} Hz, notch-filtered at {{ notch_freqs }}, re-referenced to the average reference, and segmented into {{ epoch_length }}-s non-overlapping epochs. Epochs exceeding {{ reject_uv }} microvolts peak-to-peak were excluded. Power spectral features were extracted for delta, theta, alpha, beta, and gamma bands at channel, regional, and global levels. Theta/alpha and delta/alpha log-power ratios were computed as interpretable slowing markers. Group-level feature differences were assessed using Kruskal-Wallis tests with Benjamini-Hochberg FDR correction. Machine-learning models were evaluated using nested stratified cross-validation.

## Results Draft

The automated workflow processed {{ n_subjects }} participants and retained a median of {{ median_epochs }} epochs per participant. The strongest FDR-ranked EEG features are listed below. These results should be interpreted as biomarker-screening outputs rather than clinical diagnostic claims.

{{ top_stats }}

## Model Performance

{{ metrics }}

## Generated Figures

{% for figure in figures %}- {{ figure }}
{% endfor %}
"""


def _markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    shown = df.head(max_rows) if max_rows else df
    return shown.to_markdown(index=False)


def write_report(
    cfg: dict,
    baseline: pd.DataFrame,
    features: pd.DataFrame,
    stats: pd.DataFrame,
    metrics: pd.DataFrame,
    figures: list[Path],
    output_dir: Path,
) -> Path:
    prep = cfg["preprocessing"]
    template = Template(REPORT_TEMPLATE)
    text = template.render(
        title=cfg["report"]["title"],
        baseline=_markdown_table(baseline),
        l_freq=prep["l_freq"],
        h_freq=prep["h_freq"],
        notch_freqs=", ".join(map(str, prep.get("notch_freqs", []))),
        epoch_length=prep["epoch_length"],
        reject_uv=prep.get("reject_uv", "NA"),
        n_subjects=len(features),
        median_epochs=round(float(features["n_epochs"].median()), 1) if "n_epochs" in features else "NA",
        top_stats=_markdown_table(stats, cfg["report"].get("top_features", 20)),
        metrics=_markdown_table(metrics),
        figures=[str(path.name) for path in figures],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "auto_report.md"
    out.write_text(text, encoding="utf-8")
    return out

