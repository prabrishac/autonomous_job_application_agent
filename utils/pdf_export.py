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
from pathlib import Path
from fpdf import FPDF

# ── Indigo palette (matches the app header gradient) ─────────────────────────
_INDIGO       = (67,  56, 202)
_GRAY_800     = (31,  41,  55)
_GRAY_700     = (55,  65,  81)
_GRAY_300     = (209, 213, 219)
_BLACK        = (0,   0,   0)

# ── Body font ─────────────────────────────────────────────────────────────────
# Gelasio (an openly-licensed serif, metric-compatible with Georgia) is bundled
# under assets/fonts and embedded so the PDF renders identically on every
# device.  If those files are ever missing we fall back to the base-14 core
# "Helvetica" so export still works.  See assets/fonts/OFL.txt for the license.
_FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
_SERIF_FILES = {
    "":   _FONT_DIR / "Gelasio-Regular.ttf",
    "B":  _FONT_DIR / "Gelasio-Bold.ttf",
    "I":  _FONT_DIR / "Gelasio-Italic.ttf",
    "BI": _FONT_DIR / "Gelasio-BoldItalic.ttf",
}
_USE_SERIF = all(p.exists() for p in _SERIF_FILES.values())
_FONT = "Gelasio" if _USE_SERIF else "Helvetica"


# ── Public API ────────────────────────────────────────────────────────────────

def build_pdf(title: str, content: str) -> bytes:
    """
    Return PDF bytes for *content* (markdown text) with a styled *title* header.
    Safe to stream directly to the browser as application/pdf.
    """
    pdf = _new_pdf()

    # ── Title block (skipped when no title is supplied) ───────────────────────
    if title and title.strip():
        pdf.set_font(_FONT, "B", 18)
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
    if _USE_SERIF:
        for style, path in _SERIF_FILES.items():
            pdf.add_font(_FONT, style, str(path))
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


# Matches one inline markdown span; order matters (** before *, __ before _)
_INLINE_RE = re.compile(
    r"(\*\*.+?\*\*|__.+?__|\*.+?\*|_.+?_|`.+?`|\[.+?\]\(.+?\))"
)


def _inline_segments(text: str) -> list[tuple[str, str]]:
    """Split *text* into (display_text, kind) runs.

    kind is one of: "" (plain), "B" (bold), "I" (italic), "code",
    or "link:<url>".  Markers are consumed; display text is returned raw.
    """
    segments: list[tuple[str, str]] = []
    pos = 0
    for m in _INLINE_RE.finditer(text):
        if m.start() > pos:
            segments.append((text[pos:m.start()], ""))
        tok = m.group(0)
        if tok.startswith("**") or tok.startswith("__"):
            segments.append((tok[2:-2], "B"))
        elif tok.startswith("`"):
            segments.append((tok[1:-1], "code"))
        elif tok.startswith("["):
            mm = re.match(r"\[(.+?)\]\((.+?)\)", tok)
            segments.append((mm.group(1), "link:" + mm.group(2)))
        else:  # *italic* / _italic_
            segments.append((tok[1:-1], "I"))
        pos = m.end()
    if pos < len(text):
        segments.append((text[pos:], ""))
    return segments


def _render_inline(pdf: FPDF, text: str, h: float, size: float,
                   base_style: str = "") -> None:
    """Flow *text* applying inline bold/italic/code/link styling.

    Uses pdf.write() so runs share a line and wrap at the right margin /
    current left margin.  Leaves the cursor at the end of the text — the
    caller is responsible for the trailing ln().
    """
    for raw, kind in _inline_segments(text):
        seg = _to_latin1(raw)
        if not seg:
            continue
        if kind == "B":
            pdf.set_font(_FONT, (base_style + "B"), size)
            pdf.write(h, seg)
        elif kind == "I":
            pdf.set_font(_FONT, (base_style + "I"), size)
            pdf.write(h, seg)
        elif kind == "code":
            pdf.set_font("Courier", "", size)
            pdf.write(h, seg)
        elif kind.startswith("link:"):
            pdf.set_font(_FONT, (base_style + "U"), size)
            pdf.write(h, seg, link=kind[5:])
        else:
            pdf.set_font(_FONT, base_style, size)
            pdf.write(h, seg)
    pdf.set_font(_FONT, base_style, size)


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
            pdf.set_font(_FONT, "B", 15)
            pdf.set_text_color(*_GRAY_800)
            pdf.multi_cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
            pdf.set_text_color(*_BLACK)

        # ── H2 ────────────────────────────────────────────────────────────
        elif re.match(r"^## [^#]", stripped):
            text = _to_latin1(_strip_inline(stripped[3:]))
            pdf.ln(2)
            pdf.set_font(_FONT, "B", 12)
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
            pdf.set_font(_FONT, "B", 11)
            pdf.set_text_color(*_GRAY_700)
            pdf.multi_cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
            pdf.set_text_color(*_BLACK)

        # ── H4 ────────────────────────────────────────────────────────────
        elif re.match(r"^#### ", stripped):
            text = _to_latin1(_strip_inline(stripped[5:]))
            pdf.set_font(_FONT, "BI", 10)
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
            pdf.set_text_color(*_GRAY_700)
            pdf.set_left_margin(lm + indent)
            pdf.set_x(lm + indent)
            _render_inline(pdf, f"-  {stripped[2:]}", 5.5, 10)
            pdf.ln(5.5)
            pdf.set_left_margin(lm)
            pdf.set_x(lm)
            pdf.set_text_color(*_BLACK)

        # ── Numbered list ─────────────────────────────────────────────────
        elif re.match(r"^\d+[.)]\s", stripped):
            m = re.match(r"^(\d+)[.)]\s+(.*)", stripped)
            if m:
                num  = m.group(1)
                pdf.set_text_color(*_GRAY_700)
                pdf.set_left_margin(lm + indent)
                pdf.set_x(lm + indent)
                _render_inline(pdf, f"{num}.  {m.group(2)}", 5.5, 10)
                pdf.ln(5.5)
                pdf.set_left_margin(lm)
                pdf.set_x(lm)
                pdf.set_text_color(*_BLACK)

        # ── Plain paragraph ───────────────────────────────────────────────
        else:
            pdf.set_text_color(*_GRAY_700)
            _render_inline(pdf, stripped, 5.5, 10)
            pdf.ln(5.5)
            pdf.ln(1)
            pdf.set_text_color(*_BLACK)
