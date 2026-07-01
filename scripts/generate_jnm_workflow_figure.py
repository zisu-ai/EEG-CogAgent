from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ds004504_minimal" / "figures"


def box(ax, xy, wh, title, subtitle, fill, edge="#4B5563"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        facecolor=fill, edgecolor=edge, linewidth=0.9,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
            fontsize=7.1, fontweight="bold", linespacing=1.05, color="#111827")
    ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center",
            fontsize=5.6, linespacing=1.15, color="#374151")
    return x + w / 2, y + h / 2


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.6, 3.45))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    top = FancyBboxPatch(
        (0.14, 0.77), 0.72, 0.16,
        boxstyle="round,pad=0.014,rounding_size=0.030",
        facecolor="#1F4E79", edgecolor="#163A5B", linewidth=1.0,
    )
    ax.add_patch(top)
    ax.text(0.50, 0.875, "Constrained LLM orchestration", ha="center", va="center",
            fontsize=10.0, fontweight="bold", color="white")
    ax.text(0.50, 0.823,
            "plans authorized tools · checks required artifacts · drafts claim-bounded text",
            ha="center", va="center", fontsize=6.7, color="#E5EEF6")
    ax.text(0.50, 0.785,
            "never diagnoses participants · never alters numerical outputs",
            ha="center", va="center", fontsize=6.7, color="#FDE68A", fontweight="bold")

    stages = [
        ("BIDS EEG\n+ YAML", "open data\nanalysis contract", "#E8F1FA"),
        ("MNE\npreprocessing", "filter · reference\nepoch · QC", "#E7F5EF"),
        ("Biomarker\nfeatures", "power · ratios\nconnectivity · graph", "#FFF3D6"),
        ("Statistics\n+ ML", "FDR · contrasts\nnested CV · sensitivity", "#F4E9F7"),
        ("Research\noutputs", "tables · figures\naudit · report draft", "#FCE8E6"),
    ]
    xs = [0.04, 0.235, 0.43, 0.625, 0.82]
    centers = []
    for x, (title, subtitle, fill) in zip(xs, stages):
        centers.append(box(ax, (x, 0.34), (0.145, 0.28), title, subtitle, fill))
    for i in range(len(centers) - 1):
        ax.add_patch(FancyArrowPatch(
            (centers[i][0] + 0.078, centers[i][1]),
            (centers[i + 1][0] - 0.078, centers[i + 1][1]),
            arrowstyle="-|>", mutation_scale=9, linewidth=0.9, color="#4B5563",
        ))
    for c in centers:
        ax.add_patch(FancyArrowPatch(
            (0.50, 0.77), (c[0], 0.63),
            arrowstyle="-|>", mutation_scale=6, linewidth=0.55, color="#6B7280", alpha=0.75,
        ))

    bottom = FancyBboxPatch(
        (0.09, 0.08), 0.82, 0.14,
        boxstyle="round,pad=0.012,rounding_size=0.022",
        facecolor="#F8FAFC", edgecolor="#9CA3AF", linewidth=0.9,
    )
    ax.add_patch(bottom)
    ax.text(0.50, 0.165, "Audit contract and safety guardrails", ha="center", va="center",
            fontsize=8.1, fontweight="bold", color="#1F2937")
    ax.text(0.50, 0.116,
            "participant uniqueness · leakage-safe predictions · probability bounds · artifact manifest · research-only claims",
            ha="center", va="center", fontsize=6.15, color="#4B5563")

    ax.text(0.005, 0.975, "A", fontsize=10.5, fontweight="bold", va="top")
    fig.savefig(OUT / "figure1_workflow_jnm.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "figure1_workflow_jnm.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT / "figure1_workflow_jnm.png")
    print(OUT / "figure1_workflow_jnm.pdf")


if __name__ == "__main__":
    main()
