"""
utils/pdf_parser.py — Extract plain text from an uploaded PDF.

Uses pypdf (pure-Python, no system dependencies).

Raises
------
ValueError  — file is not a valid PDF, is encrypted, or yields no text
              (likely a scanned/image-only PDF).
"""

import io
import re
from pypdf import PdfReader
from pypdf.errors import PdfReadError

# Max upload size accepted by the API endpoint (bytes)
MAX_PDF_BYTES = 10 * 1024 * 1024  # 10 MB


def extract_pdf_text(data: bytes) -> tuple[str, int]:
    """
    Parse *data* as a PDF and return ``(text, page_count)``.

    The returned text has:
    - Pages separated by a blank line
    - Runs of 3+ blank lines collapsed to two (keeps spacing readable)
    - Leading/trailing whitespace stripped per page

    Raises
    ------
    ValueError  if the file cannot be read, is encrypted, or has no text.
    """
    if len(data) > MAX_PDF_BYTES:
        raise ValueError(
            f"PDF is too large ({len(data) // (1024*1024)} MB). "
            "Maximum allowed size is 10 MB."
        )

    try:
        reader = PdfReader(io.BytesIO(data))
    except PdfReadError as exc:
        raise ValueError(f"Could not read PDF: {exc}") from exc

    if reader.is_encrypted:
        raise ValueError(
            "This PDF is password-protected. "
            "Please remove the password before uploading."
        )

    page_texts: list[str] = []
    for page in reader.pages:
        raw = _extract_page_text(page)
        cleaned = _clean_page_text(raw)
        if cleaned:
            page_texts.append(cleaned)

    if not page_texts:
        raise ValueError(
            "No text could be extracted from this PDF. "
            "It may be a scanned or image-only document. "
            "Please paste the text manually instead."
        )

    combined = "\n\n".join(page_texts)
    # Collapse 3+ consecutive blank lines → 2
    combined = re.sub(r"\n{3,}", "\n\n", combined)
    return combined, len(reader.pages)


def _extract_page_text(page) -> str:
    """Extract one page's text, preferring the layout-preserving mode.

    The default extraction mode frequently mangles text from multi-column /
    flowed PDFs: it doubles the spaces between words and scatters individual
    words onto their own lines, so the result looks "cropped" in the UI even
    though every character is present.  ``extraction_mode="layout"`` keeps
    words on their original lines with single spacing (and preserves table
    columns).  It is newer code, so we fall back to the default mode if it
    raises or yields nothing.
    """
    try:
        text = page.extract_text(extraction_mode="layout") or ""
    except Exception:
        text = ""
    if not text.strip():
        text = page.extract_text() or ""
    return text


def _clean_page_text(raw: str) -> str:
    """Trim trailing whitespace from each line and strip the page edges.

    Layout mode pads lines with trailing spaces and may leave a leading
    blank line; removing them keeps the textarea tidy without disturbing the
    leading indentation that aligns table columns.
    """
    lines = [line.rstrip() for line in raw.splitlines()]
    return "\n".join(lines).strip()
