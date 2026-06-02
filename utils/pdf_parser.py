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
        raw = page.extract_text() or ""
        cleaned = raw.strip()
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
