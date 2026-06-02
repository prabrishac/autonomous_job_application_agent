"""
utils/pdf_export.py — Convert markdown-formatted document text to a PDF.

Uses fpdf2 (pure-Python, no system dependencies).

Supported markdown:
  # H1 / ## H2 / ### H3 / #### H4
  **bold** / *italic* / `code` / [link](url)
  - / * / + bullet lists
  1. / 1) numbered lists
  --- horizontal rules
  Blank lines → paragraph spacing
"""

import re
from fpdf import FPDF

# ── Indigo palette (matches the app header gradient) ─────────────────────────
_INDIGO       = (67,  56, 202)
_GRAY_800     = (31,  41,  55)
_GRAY_700     = (55,  65,  81)
_GRAY_300     = (209, 213, 219)
_BLACK        = (0,   0,   0)


# ── Public API ────────────────────────────────────────────────────────────────

def build_pdf(title: str, content: str) -> bytes:
    """
    Return PDF bytes for *content* (markdown text) with a styled *title* header.
    Safe to stream directly to the browser as application/pdf.
    """
    pdf = _new_pdf()

    # ── Title block ───────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*_INDIGO)
    pdf.multi_cell(0, 10, _to_latin1(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*_INDIGO)
    pdf.set_line_width(0.6)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(5)
    pdf.set_text_color(*_BLACK)

    _render_markdown(pdf, content)

    return bytes(pdf.output())


# ── Internal helpers ──────────────────────────────────────────────────────────

def _new_pdf() -> FPDF:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=22)
    pdf.set_margins(left=20, top=20, right=20)
    pdf.add_page()
    return pdf


def _strip_inline(text: str) -> str:
    """Remove inline markdown markers, keeping the display text."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)    # **bold**
    text = re.sub(r"__(.+?)__",     r"\1", text)    # __bold__
    text = re.sub(r"\*(.+?)\*",     r"\1", text)    # *italic*
    text = re.sub(r"_(.+?)_",       r"\1", text)    # _italic_
    text = re.sub(r"`(.+?)`",       r"\1", text)    # `code`
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text) # [text](url)
    return text


# Characters outside latin-1 that commonly appear in LLM output
_UNICODE_MAP: dict[str, str] = {
    "—": "--",   # em dash
    "–": "-",    # en dash
    "‘": "'",    # left single quote
    "’": "'",    # right single quote
    "“": '"',    # left double quote
    "”": '"',    # right double quote
    "•": "-",    # bullet
    "…": "...",  # ellipsis
    "·": "-",    # middle dot
    "‒": "-",    # figure dash
    "―": "--",   # horizontal bar
    " ": " ",    # non-breaking space
}


def _to_latin1(text: str) -> str:
    """Replace common Unicode characters with latin-1 equivalents, then
    drop anything that still cannot be encoded so fpdf never chokes."""
    for ch, repl in _UNICODE_MAP.items():
        text = text.replace(ch, repl)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _render_markdown(pdf: FPDF, content: str) -> None:
    lm = pdf.l_margin
    indent = 6  # mm indent for list items

    for raw_line in content.split("\n"):
        line    = raw_line.rstrip()
        stripped = line.strip()

        # ── Blank line → spacing ───────────────────────────────────────────
        if not stripped:
            pdf.ln(3)
            continue

        # ── H1 ────────────────────────────────────────────────────────────
        if re.match(r"^# [^#]", stripped):
            text = _to_latin1(_strip_inline(stripped[2:]))
            pdf.set_font("Helvetica", "B", 15)
            pdf.set_text_color(*_GRAY_800)
            pdf.multi_cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
            pdf.set_text_color(*_BLACK)

        # ── H2 ────────────────────────────────────────────────────────────
        elif re.match(r"^## [^#]", stripped):
            text = _to_latin1(_strip_inline(stripped[3:]))
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(*_INDIGO)
            pdf.multi_cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
            # underline rule
            pdf.set_draw_color(*_INDIGO)
            pdf.set_line_width(0.2)
            y = pdf.get_y()
            pdf.line(lm, y, pdf.w - pdf.r_margin, y)
            pdf.ln(3)
            pdf.set_text_color(*_BLACK)

        # ── H3 ────────────────────────────────────────────────────────────
        elif re.match(r"^### [^#]", stripped):
            text = _to_latin1(_strip_inline(stripped[4:]))
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(*_GRAY_700)
            pdf.multi_cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
            pdf.set_text_color(*_BLACK)

        # ── H4 ────────────────────────────────────────────────────────────
        elif re.match(r"^#### ", stripped):
            text = _to_latin1(_strip_inline(stripped[5:]))
            pdf.set_font("Helvetica", "BI", 10)
            pdf.set_text_color(*_GRAY_700)
            pdf.multi_cell(0, 5.5, text, new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(*_BLACK)

        # ── Horizontal rule ───────────────────────────────────────────────
        elif re.match(r"^-{3,}$", stripped) or stripped in ("***", "___"):
            pdf.ln(2)
            pdf.set_draw_color(*_GRAY_300)
            pdf.set_line_width(0.3)
            pdf.line(lm, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(4)

        # ── Bullet list ───────────────────────────────────────────────────
        elif re.match(r"^[-*+] ", stripped):
            text = _to_latin1(_strip_inline(stripped[2:]))
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(*_GRAY_700)
            pdf.set_x(lm + indent)
            pdf.multi_cell(
                pdf.w - pdf.r_margin - lm - indent, 5.5,
                f"-  {text}",
                new_x="LMARGIN", new_y="NEXT",
            )
            pdf.set_text_color(*_BLACK)

        # ── Numbered list ─────────────────────────────────────────────────
        elif re.match(r"^\d+[.)]\s", stripped):
            m = re.match(r"^(\d+)[.)]\s+(.*)", stripped)
            if m:
                num  = m.group(1)
                text = _to_latin1(_strip_inline(m.group(2)))
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(*_GRAY_700)
                pdf.set_x(lm + indent)
                pdf.multi_cell(
                    pdf.w - pdf.r_margin - lm - indent, 5.5,
                    f"{num}.  {text}",
                    new_x="LMARGIN", new_y="NEXT",
                )
                pdf.set_text_color(*_BLACK)

        # ── Plain paragraph ───────────────────────────────────────────────
        else:
            text = _to_latin1(_strip_inline(stripped))
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(*_GRAY_700)
            pdf.multi_cell(0, 5.5, text, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
            pdf.set_text_color(*_BLACK)
