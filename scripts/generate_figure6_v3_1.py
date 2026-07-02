"""Generate Figure 6 (v3.1): independent external archive integrity + evaluation.

Four-panel publication composite, recomputable from v3.1 artifacts only:
  A. Integrity flow: 92 nominal -> one size-5 duplicate cluster -> 88 unique records.
  B. Primary unique-record ROC (AUC 0.967, 95% CI).
  C. Primary confusion matrix (axes/colorbar labelled 'unique records').
  D. Top domain-shift Cohen's d (label-free; external minus discovery).

Colorblind-safe; >=300 dpi PNG + vector PDF; no 'subject-level' wording. Reads:
  results/external_validation_osf_v3_1/{roc_curve_primary.csv,
  confusion_matrix_primary.csv, domain_shift_primary_labelfree.csv,
  signal_fingerprint_audit_eyes_closed.json, external_metrics.json}
Writes:
  results/external_validation_osf_v3_1/figures/figure6_external_integrity.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

OSF = Path(__file__).resolve().parent.parent / "results" / "external_validation_osf_v3_1"
OUT = OSF / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# colorblind-safe palette (Wong 2011): blue / orange / green / vermillion
C_BLUE, C_ORANGE, C_GREEN, C_VERM = "#0072B2", "#E69F00", "#009E73", "#D55E00"


def main() -> None:
    roc = pd.read_csv(OSF / "roc_curve_primary.csv")
    cm = pd.read_csv(OSF / "confusion_matrix_primary.csv", index_col=0)
    shift = pd.read_csv(OSF / "domain_shift_primary_labelfree.csv")
    fp = json.loads((OSF / "signal_fingerprint_audit_eyes_closed.json").read_text(encoding="utf-8"))
    metrics = json.loads((OSF / "external_metrics.json").read_text(encoding="utf-8"))

    nominal = fp["nominal_count"]
    unique = fp["unique_fingerprint_count"]
    dup_members = next((m for m in fp["clusters"].values() if len(m) > 1), [])
    dup_size = len(dup_members)
    auc = metrics["point"]["roc_auc"]
    auc_ci = metrics["bootstrap_ci_95"]["roc_auc"]

    fig = plt.figure(figsize=(11.5, 9.0), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.30,
                          left=0.06, right=0.97, top=0.93, bottom=0.07)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    # --- Panel A: integrity flow -------------------------------------------------
    axA.set_xlim(0, 10)
    axA.set_ylim(0, 10)
    axA.axis("off")

    def box(cx, cy, w, h, text, face, edge=C_BLUE):
        patch = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                               boxstyle="round,pad=0.08,rounding_size=0.18",
                               linewidth=1.5, edgecolor=edge, facecolor=face)
        axA.add_patch(patch)
        axA.text(cx, cy, text, ha="center", va="center", fontsize=8.5, wrap=True)

    def arrow(x0, y0, x1, y1):
        axA.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                      mutation_scale=14, color="#444444", lw=1.4))

    box(2.5, 8.3, 3.6, 1.5, f"Nominal records\n{nominal} (80 AD + 12 HC)", "#DEEBF7")
    box(7.5, 8.3, 3.6, 1.5,
        f"Content fingerprint audit\n{nominal} digests (parsed float64)", "#DEEBF7")
    arrow(4.3, 8.3, 5.7, 8.3)
    box(5.0, 5.5, 5.0, 1.6,
        f"Exact-signal duplicate cluster (size {dup_size})\n"
        f"{', '.join(dup_members)}\nReproduces in Eyes_open",
        "#FEE0B6", edge=C_ORANGE)
    arrow(7.5, 7.55, 6.2, 6.3)
    arrow(3.5, 7.55, 4.2, 6.3)
    box(2.5, 2.6, 3.8, 1.5, "Excluded as\ndeterministic\nduplicates", "#FDE0DD", edge=C_VERM)
    box(7.5, 2.6, 3.8, 1.5,
        f"Primary unique records\n{unique} (76 AD + 12 HC)", "#D9ECD9", edge=C_GREEN)
    arrow(4.0, 5.0, 2.8, 3.35)
    arrow(6.0, 5.0, 7.2, 3.35)
    axA.text(5.0, 9.7, "A.  Archive integrity flow", ha="center", fontsize=11, fontweight="bold")
    axA.text(5.0, 0.6, "88 unique recordings, not proven unique persons",
             ha="center", fontsize=7.5, style="italic", color="#555555")

    # --- Panel B: ROC ------------------------------------------------------------
    axB.plot(roc["fpr"], roc["tpr"], color=C_BLUE, lw=2.2,
             label=f"Unique records (AUC = {auc:.3f})")
    axB.fill_between(roc["fpr"], roc["tpr"], 0, color=C_BLUE, alpha=0.06)
    axB.plot([0, 1], [0, 1], color="#888888", lw=1, linestyle="--", label="chance")
    axB.set_xlabel("False positive rate (1 - specificity)")
    axB.set_ylabel("True positive rate (sensitivity)")
    axB.set_title("B.  External ROC (OSF, unique records)")
    axB.set_xlim(0, 1)
    axB.set_ylim(0, 1.02)
    axB.legend(loc="lower right", fontsize=8)
    axB.text(0.97, 0.06, f"95% CI {auc_ci['ci_low']:.3f} to {auc_ci['ci_high']:.3f}",
             ha="right", va="bottom", fontsize=7.5, color="#444444",
             transform=axB.transAxes)

    # --- Panel C: confusion matrix ----------------------------------------------
    array = cm.to_numpy().astype(float)  # rows true_HC,true_AD ; cols pred_HC,pred_AD
    im = axC.imshow(array, cmap="Blues", vmin=0, vmax=max(1, array.max()))
    for r in range(2):
        for c in range(2):
            color = "white" if array[r, c] > array.max() / 2 else "black"
            axC.text(c, r, str(int(array[r, c])), ha="center", va="center",
                     color=color, fontsize=13, fontweight="bold")
    axC.set_xticks(range(2), ["pred HC", "pred AD"])
    axC.set_yticks(range(2), ["true HC", "true AD"])
    axC.set_title("C.  Confusion matrix (unique records)")
    cb = fig.colorbar(im, ax=axC, shrink=0.78)
    cb.set_label("Unique records")

    # --- Panel D: domain shift --------------------------------------------------
    s = shift.reindex(shift["cohens_d"].abs().sort_values(ascending=False).index).head(10).iloc[::-1]
    short = [f.replace("relpow_region__", "").replace("relpow_global__", "global ")
             .replace("ratio_region__", "").replace("ratio_global__", "global ")
             .replace("__", " ") for f in s["feature"]]
    colors = [C_VERM if v > 0 else C_BLUE for v in s["cohens_d"]]
    axD.barh(range(len(s)), s["cohens_d"].to_numpy(), color=colors)
    axD.set_yticks(range(len(s)), short, fontsize=7.5)
    axD.axvline(0, color="#444444", lw=0.8)
    axD.axvline(0.5, color=C_ORANGE, lw=0.8, linestyle="--", alpha=0.7)
    axD.axvline(-0.5, color=C_ORANGE, lw=0.8, linestyle="--", alpha=0.7)
    axD.set_xlabel("Cohen's d (external minus discovery)")
    axD.set_title("D.  Top domain shift (label-free)")
    axD.text(0.02, 0.06, f"{int((shift['cohens_d'].abs()>0.5).sum())}/36 |d|>0.5",
             transform=axD.transAxes, fontsize=7.5, color="#444444", va="bottom")

    for path in (OUT / "figure6_external_integrity.png", OUT / "figure6_external_integrity.pdf"):
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote figure6_external_integrity.png/.pdf to {OUT}")


if __name__ == "__main__":
    main()
