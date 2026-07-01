from __future__ import annotations

import csv
import re
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "literature" / "fulltext" / "jnm_target"
OUT_TXT = ROOT / "literature" / "jnm_target_extracted"
OUT_CSV = ROOT / "literature" / "jnm_target_gap_table.csv"
OUT_MD = ROOT / "literature" / "jnm_target_summary.md"


MANUAL = {
    "EEGLLM.pdf": {
        "short_name": "EEG-AI",
        "title": "EEG-AI: An agentic system for AI-assisted semi-automated EEG preprocessing and artifact removal",
        "journal": "Journal of Neuroscience Methods",
        "task": "Semi-automated EEG preprocessing and ICA artifact removal",
        "llm_agent": "Yes",
        "disease_biomarker": "No",
        "audit_report": "Partial: closed-loop policy and expert oversight",
        "difference": "Closest JNM precedent; EEG-CogAgent should avoid claiming novelty as a generic EEG agent and instead emphasize disease-focused biomarker discovery, audit contract and constrained report generation.",
    },
    "EEG Agent A Unified Framework for Automated EEG Analysis Using Large Language Models.pdf": {
        "short_name": "EEG Agent",
        "title": "EEG Agent: A Unified Framework for Automated EEG Analysis Using Large Language Models",
        "journal": "AAAI",
        "task": "General automated EEG analysis tool orchestration",
        "llm_agent": "Yes",
        "disease_biomarker": "Not dementia-specific",
        "audit_report": "General report generation; not focused on disease validation boundaries",
        "difference": "Establishes that LLM-based EEG tool orchestration exists; EEG-CogAgent must be framed as a narrower dementia BIDS-to-biomarker validation workflow.",
    },
    "EEGUnity_Open-Source_Tool_in_Facilitating_Unified_EEG_Datasets_Toward_Large-Scale_EEG_Model.pdf": {
        "short_name": "EEGUnity",
        "title": "EEGUnity: open-source tool in facilitating unified EEG datasets toward large-scale EEG model",
        "journal": "IEEE Transactions on Neural Systems and Rehabilitation Engineering",
        "task": "EEG dataset harmonization and large-scale EEG model preparation",
        "llm_agent": "Possibly LLM-supported, but primarily harmonization software",
        "disease_biomarker": "No dementia-specific biomarker validation",
        "audit_report": "Dataset unification rather than manuscript/report audit",
        "difference": "Supports the need for standardized EEG data infrastructure; EEG-CogAgent sits downstream, turning BIDS datasets into audited disease-analysis artifacts.",
    },
    "data-08-00095.pdf": {
        "short_name": "ds004504 dataset",
        "title": "A dataset of scalp EEG recordings of Alzheimer's disease, frontotemporal dementia and healthy subjects from routine EEG",
        "journal": "Data",
        "task": "Public AD/FTD/HC EEG dataset release",
        "llm_agent": "No",
        "disease_biomarker": "Dataset resource",
        "audit_report": "No",
        "difference": "Primary validation dataset for EEG-CogAgent; must cite as dataset source and respect its cohort limitations.",
    },
    "s41598-026-35316-9.pdf": {
        "short_name": "Functional connectivity AD/FTD",
        "title": "EEG-based classification of Alzheimer's disease and frontotemporal dementia using functional connectivity",
        "journal": "Scientific Reports",
        "task": "AD/FTD/HC classification using functional connectivity",
        "llm_agent": "No",
        "disease_biomarker": "Yes: connectivity classification",
        "audit_report": "No LLM/report workflow",
        "difference": "Shows ds004504 disease classification has already been studied; EEG-CogAgent novelty cannot be classification accuracy alone.",
    },
    "12984_2026_Article_1897.pdf": {
        "short_name": "Multi-dimensional EEG dementia",
        "title": "Multi-dimensional EEG analysis reveals distinct neurophysiological patterns in Alzheimer's and frontotemporal dementia",
        "journal": "Journal of NeuroEngineering and Rehabilitation",
        "task": "Multi-feature AD/FTD/HC EEG characterization",
        "llm_agent": "No",
        "disease_biomarker": "Yes: multidimensional EEG patterns",
        "audit_report": "No LLM/report workflow",
        "difference": "A direct disease-biomarker competitor; our manuscript should present lower-claim, audit-first workflow value rather than outperforming multidimensional classifiers.",
    },
}


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            text = f"[page extraction failed: {exc}]"
        parts.append(f"\n---PAGE {i + 1}---\n{text}")
    text = "\n".join(parts).replace("\x00", "")
    return text


def word_flag(text: str, patterns: list[str]) -> str:
    lower = text.lower()
    return "Yes" if any(p.lower() in lower for p in patterns) else "No"


def first_match(text: str, patterns: list[str]) -> str:
    for pat in patterns:
        m = re.search(pat, text, flags=re.I | re.S)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()[:700]
    return ""


def main() -> None:
    OUT_TXT.mkdir(parents=True, exist_ok=True)
    rows = []
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        text = extract_text(pdf)
        (OUT_TXT / f"{pdf.stem}.txt").write_text(text, encoding="utf-8")
        manual = MANUAL.get(pdf.name, {})
        abstract = first_match(text, [r"ABSTRACT\s*(.*?)(?:\n1\.|\nIntroduction|Keywords:)", r"Abstract\s*(.*?)(?:\n1\.|\nIntroduction|Keywords:)"])
        row = {
            "file": pdf.name,
            "short_name": manual.get("short_name", pdf.stem),
            "title": manual.get("title", first_match(text, [r"---PAGE 1---\s*(.*?)\n[A-Z][a-z]+ [A-Z]"])),
            "journal": manual.get("journal", ""),
            "task": manual.get("task", ""),
            "dataset_or_population": first_match(text, [r"(ds\d{6}.*?)\.", r"(Alzheimer.*?healthy.*?subjects.*?)\."]),
            "llm_or_agent": manual.get("llm_agent", word_flag(text, ["large language model", "LLM", "agentic", "agent"])),
            "automated_preprocessing": word_flag(text, ["preprocessing", "artifact", "ICA", "bad channel"]),
            "disease_biomarker": manual.get("disease_biomarker", word_flag(text, ["Alzheimer", "frontotemporal", "dementia", "biomarker"])),
            "audit_or_report_generation": manual.get("audit_report", word_flag(text, ["audit", "report generation", "reproducib", "closed-loop"])),
            "abstract_or_key_point": abstract,
            "difference_for_eeg_cogagent": manual.get("difference", ""),
        }
        rows.append(row)

    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# JNM target literature positioning summary",
        "",
        "Purpose: use the downloaded target papers to reposition EEG-CogAgent for first submission to Journal of Neuroscience Methods.",
        "",
        "## Main conclusion",
        "",
        "The manuscript should not be framed as a generic LLM EEG agent or as another AD/FTD classifier. The strongest niche is an auditable, dementia-focused BIDS-to-biomarker workflow in which deterministic EEG modules generate numerical artifacts and the language-model layer only plans, checks, and drafts constrained reports.",
        "",
        "## Gap table",
        "",
        "| Short name | Main task | LLM/agent | Dementia biomarker | Difference for EEG-CogAgent |",
        "|---|---|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['short_name']} | {r['task']} | {r['llm_or_agent']} | {r['disease_biomarker']} | {r['difference_for_eeg_cogagent']} |"
        )
    lines += [
        "",
        "## Required manuscript changes",
        "",
        "1. Add a Related Work subsection contrasting EEG-AI, EEG Agent, EEGUnity, DISCOVER-EEG, and ds004504 dementia classifiers.",
        "2. Add an Agent boundary and audit contract subsection: LLM does not diagnose, inspect raw EEG, change thresholds post hoc, or alter numerical outputs.",
        "3. Add a comparison table in the manuscript or supplement with columns: tool/study, EEG task, LLM/agent role, disease validation, audit/report support, claim boundary.",
        "4. Revise Figure 1 to explicitly show deterministic modules below and constrained LLM orchestration above.",
        "5. Strengthen limitations: small single-site cohort, no external validation, AD-versus-FTD weakness, ds006036 condition transfer only.",
        "6. Cover letter should state novelty as method/workflow validation, not diagnostic superiority.",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_CSV)
    print(OUT_MD)


if __name__ == "__main__":
    main()
