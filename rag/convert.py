"""One-time: convert knowledge/pdfs/*.pdf -> knowledge/md/*.md.

Run:  uv run python rag/convert.py

Uses pymupdf4llm (layout-aware PDF -> Markdown). It handles the multi-column
academic-journal layout of these ADA/clinical guideline PDFs correctly —
preserving reading order, paragraph boundaries, headings, and tables — which
markitdown's pdfminer backend does not (it interleaves columns). Those Markdown
headings are what rag/chunk.py's structure-aware pass splits on.

Idempotent — re-running overwrites knowledge/md/.
"""
from pathlib import Path

import pymupdf4llm

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "knowledge" / "pdfs"
MD_DIR = ROOT / "knowledge" / "md"


def main() -> None:
    MD_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(PDF_DIR.glob("*.pdf")) # turn pdfs into a list
    if not pdfs:
        print(f"No PDFs found in {PDF_DIR}")
        return

    print(f"Converting {len(pdfs)} PDF(s) from {PDF_DIR} -> {MD_DIR}\n")
    ok = 0
    for pdf in pdfs: # Loop through each PDf in list of PDFs
        out = MD_DIR / f"{pdf.stem}.md" # md file
        try:
            text = pymupdf4llm.to_markdown(str(pdf), show_progress=False) or ""
        except Exception as e:  # keep going; a bad file shouldn't stop the batch
            print(f"  x {pdf.name:14} FAILED: {e}")
            continue
        out.write_text(text, encoding="utf-8") # if successful, write to md file
        headings = sum(1 for ln in text.splitlines() if ln.lstrip().startswith("#")) # count headings
        flag = "empty" if len(text.strip()) < 50 else "ok" # after removing whitespace, if less than 50 chars, flag as empty
        print(f"  {pdf.name:14} -> {out.name:14} {len(text):>8,} chars  {headings:>3} headings  {flag}")
        ok += 1

    print(f"\nDone: {ok}/{len(pdfs)} converted into {MD_DIR}")
    print("Review any 'empty' files — likely scanned/image-only PDFs needing OCR.")


if __name__ == "__main__":
    main()
