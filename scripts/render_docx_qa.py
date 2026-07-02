"""Render a DOCX to per-page PNGs for visual QA (Windows, MS Word via COM).

Pipeline: DOCX -> PDF (Word COM SaveAs FileFormat=17) -> PNG per page (PyMuPDF).
Used by the v3.1 submission visual-QA workflow (build -> render -> inspect ->
fix -> re-render). Not imported by the deterministic pipeline; a rendering tool only.

CLI:
    python scripts/render_docx_qa.py <input.docx> <out_dir> [--dpi 150]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def docx_to_pdf(docx_path: Path, pdf_path: Path) -> None:
    """Convert DOCX -> PDF via MS Word COM automation (Windows)."""
    import win32com.client as win32  # type: ignore
    docx_abs = str(docx_path.resolve())
    pdf_abs = str(pdf_path.resolve())
    word = win32.gencache.EnsureDispatch("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    try:
        doc = word.Documents.Open(docx_abs, ReadOnly=True)
        # FileFormat 17 = wdFormatPDF
        doc.SaveAs(pdf_abs, FileFormat=17)
        doc.Close(SaveChanges=False)
    finally:
        word.Quit()


def pdf_to_pngs(pdf_path: Path, out_dir: Path, dpi: int = 150) -> list[Path]:
    """Render each PDF page to PNG {stem}-page-{n}.png at the given DPI."""
    import fitz  # PyMuPDF
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    written: list[Path] = []
    with fitz.open(str(pdf_path)) as doc:
        for idx, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            out = out_dir / f"{stem}-page-{idx:02d}.png"
            pix.save(str(out))
            written.append(out)
    return written


def render(docx_path: Path, out_dir: Path, dpi: int = 150) -> list[Path]:
    pdf_path = out_dir / (Path(docx_path).stem + ".pdf")
    docx_to_pdf(Path(docx_path), pdf_path)
    return pdf_to_pngs(pdf_path, out_dir, dpi=dpi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("docx")
    ap.add_argument("out_dir")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()
    pages = render(Path(args.docx), Path(args.out_dir), dpi=args.dpi)
    print(f"Rendered {len(pages)} pages to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
