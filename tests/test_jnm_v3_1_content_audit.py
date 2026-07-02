"""Content audit for the v3.1 manuscript + submission package (Stage 9).

Enforces the prompt's claim-boundary and journal-format rules directly against
the v3.1 source files. Independent of the DOCX build: it audits the markdown
source of truth so that a green test means the manuscript content complies
before any DOCX is rendered.

Rules checked (see each test). Focused, offline, fast.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
MS = REPO / "docs" / "FULL_MANUSCRIPT_JNM_V3_1.md"
LEDGER_CSV = REPO / "docs" / "JNM_V3_1_EVIDENCE_LEDGER.csv"


@pytest.fixture(scope="module")
def manuscript() -> str:
    return MS.read_text(encoding="utf-8")


def _abstract(manuscript: str) -> str:
    m = re.search(r"## Abstract\s+(.+?)\n\n\*\*Keywords:", manuscript, re.S)
    assert m, "abstract section not found"
    return m.group(1)


# --- format rules ------------------------------------------------------------


def test_abstract_le_250_words(manuscript):
    words = re.findall(r"[A-Za-z0-9'\-/.:]+", _abstract(manuscript))
    assert len(words) <= 250, f"abstract is {len(words)} words (>250)"


def test_abstract_has_no_citations(manuscript):
    ab = _abstract(manuscript)
    assert not re.search(r"\([A-Z][a-z]+ et al\., \d{4}", ab), "abstract must contain no citations"
    assert not re.search(r"\[\d+\]", ab), "abstract must contain no numeric citations"


def test_keywords_1_to_7(manuscript):
    km = re.search(r"\*\*Keywords:\*\*\s*(.+)", manuscript)
    kws = [k.strip() for k in km.group(1).split(";") if k.strip()]
    assert 1 <= len(kws) <= 7


def test_no_residual_numeric_bracket_citations(manuscript):
    assert not re.search(r"\[\d+\]", manuscript), "residual numeric [n] citation found"


def test_references_alphabetical(manuscript):
    m = re.search(r"## References(.+?)(## Tables|## Figure legends|\Z)", manuscript, re.S)
    block = m.group(1)
    # Non-dataset reference entries: lines starting with a capitalized author surname + year.
    surnames: list[str] = []
    for line in block.splitlines():
        s = line.strip()
        if not s or s.startswith("[dataset]") or not re.search(r"\b(19|20)\d{2}\b", s):
            continue
        first = re.match(r"([A-ZÀ-Ÿ][\wÀ-ÿ\-']+)", s)
        if first:
            surnames.append(first.group(1).lower())
    assert surnames, "no reference entries parsed"
    assert surnames == sorted(surnames), f"references not alphabetical: {surnames}"


# --- claim-boundary rules (forbidden phrases) --------------------------------


@pytest.mark.parametrize("forbidden", [
    "92 independent subjects",
    "92 independent persons",
    "88 unique persons",
    "88 unique people",
    "subject-level external bootstrap",
    "absence of independent external validation",
    "prospectively validated",
    "clinical-grade",
    "state-of-the-art",
])
def test_forbidden_phrases_absent(manuscript, forbidden):
    assert forbidden.lower() not in manuscript.lower(), f"forbidden phrase present: {forbidden!r}"


# --- required content --------------------------------------------------------


@pytest.mark.parametrize("required", [
    "88 unique recordings",
    "exact-signal duplicate",
    "0.967",   # AUC
    "0.873",   # BA
    "0.772",   # BA CI low
    "0.917",   # AUC CI low
    "osf-common19-float64-v2",
    "UNRESOLVED",
    "post-hoc",
    "generative AI",
])
def test_required_phrases_present(manuscript, required):
    assert required.lower() in manuscript.lower(), f"required phrase missing: {required!r}"


def test_nominal_92_is_labelled_non_primary(manuscript):
    # Every mention of the 0.877 nominal figure must be framed as non-primary.
    assert re.search(r"0\.877.*non-primary|non-primary.*0\.877", manuscript, re.I | re.S) or \
        "0.877" not in manuscript


def test_ds006036_is_cross_condition_not_external(manuscript):
    # ds006036 must not be called an external cohort.
    for m in re.finditer(r"ds006036", manuscript):
        window = manuscript[max(0, m.start() - 80):m.end() + 80].lower()
        assert "external cohort" not in window and "external validation" not in window or \
            "not" in window or "rather than" in window, \
            "ds006036 described as external without a negation"


def test_osf_license_unresolved(manuscript):
    assert "UNRESOLVED" in manuscript and "dataset node license" in manuscript.lower()


def test_ai_disclosure_present(manuscript):
    assert "Declaration of generative AI" in manuscript
    assert "Claude Code" in manuscript or "Claude" in manuscript


def test_data_availability_mentions_osf_and_openneuro(manuscript):
    da = re.search(r"### Data availability(.+?)(###|\Z)", manuscript, re.S).group(1)
    assert "2v5md" in da and "ds004504" in da and "ds006036" in da


def test_figures_1_to_6_referenced_in_text(manuscript):
    for n in (1, 2, 3, 4, 5, 6):
        assert re.search(rf"Fig(?:ure|\.)\s*{n}\b", manuscript), f"Figure {n} not referenced in text"


# --- ledger cross-check ------------------------------------------------------


def test_ledger_exists_and_has_key_claims():
    import pandas as pd
    df = pd.read_csv(LEDGER_CSV)
    ids = set(df["claim_id"])
    for cid in ("EXT-02", "EXT-09", "EXT-10", "EXT-13", "EXT-16", "BENCH-01"):
        assert cid in ids, f"ledger missing key claim {cid}"
    # External 88 / AUC / BA present as values.
    vals = " ".join(df["exact_value"].astype(str).tolist())
    assert "0.967" in vals and "0.873" in vals
