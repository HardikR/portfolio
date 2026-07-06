"""
================================================================================
DVS Task 1 — Unified PDF Extraction Pipeline  (PaddleOCR-VL-1.5)
================================================================================
Extracts text AND tables from a contract PDF — digital or scanned pages alike —
and writes a structured JSON file.

Design — ONE engine, ONE path:
  PaddleOCR-VL-1.5 is a complete document parser. It handles digital and
  scanned pages itself and emits text + tables in a single pass, so this
  pipeline has no scanned/digital branch. A page-type DETECTION step is still
  run, but only as metadata (it is reported in the JSON) — both page types go
  through the same engine.

Stages:
  Stage 1  Detect    — classify each page digital/scanned (metadata only)
  Stage 2  Parse     — PaddleOCR-VL-1.5: every page -> Markdown (text + tables)
  Stage 3  Clean     — strip repeated banner / footer / page-number noise
  Stage 4  Clauses   — parse clause / subclause structure from the Markdown
  Stage 5  Tables    — pull tables out of the Markdown the model already emitted
  Stage 6  Output    — assemble + write the JSON
  Stage 7  TOC check — verify clauses against the TOC (ALWAYS runs;
                       --toc-pages is required on every run)

Engine:
  Primary  : PaddleOCR-VL-1.5  (research pick — degraded-scan specialist)
  Fallback : Tesseract         (only if PaddleOCR-VL cannot be loaded; the run
                                is then flagged degraded in the JSON)

Install (Colab):
  pip install paddleocr paddlepaddle pymupdf pillow
  # CPU fallback only:  apt-get install tesseract-ocr ; pip install pytesseract

Usage (--toc-pages is REQUIRED — name the Table-of-Contents page(s) every run):
  python dvs_extract_paddleocrvl.py contract.pdf --toc-pages 3-9
  python dvs_extract_paddleocrvl.py contract.pdf --toc-pages 3 4 --output-dir out
  python dvs_extract_paddleocrvl.py contract.pdf --toc-pages 3-9 --max-pages 5     # quick test
  python dvs_extract_paddleocrvl.py contract.pdf --toc-pages 3-9 --pages 10 11 50  # specific pages
  python dvs_extract_paddleocrvl.py contract.pdf --toc-pages 3-9 --range 1 40      # a batch

Note on Colab GPU: PaddleOCR-VL-1.5 runs on CPU but is much faster on a GPU.
For a large PDF, process in batches with --range (e.g. --range 1 40, then
--range 41 80, ...). Each batch writes its own JSON named with the page span,
so batches never overwrite each other; merge the JSONs afterwards.
================================================================================
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF


# ==============================================================================
# CONFIGURATION
# ==============================================================================

RENDER_DPI         = 200    # page raster resolution handed to the OCR model
MIN_CHARS_DIGITAL  = 200    # >= this much good text => page tagged 'digital'
NOISE_REPEAT_RATIO = 0.25   # a line on >25% of pages is treated as banner/footer
NOISE_MIN_REPEAT   = 3
NOISE_MIN_LINELEN  = 4
WORD_QUALITY_THRESH = 0.40  # short-page tie-breaker for the detector


# ==============================================================================
# SHARED HELPERS
# ==============================================================================

def _word_quality(text: str) -> float:
    """Fraction of tokens that are real alphabetic words (len > 1)."""
    tokens = text.split()
    if not tokens:
        return 0.0
    real = sum(1 for t in tokens if t.isalpha() and len(t) > 1)
    return real / len(tokens)


def _render_page(fitz_page: fitz.Page, dpi: int = RENDER_DPI):
    """Render a fitz page to a PIL RGB image."""
    from PIL import Image
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = fitz_page.get_pixmap(matrix=mat, alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def _normalise_for_boilerplate(text: str) -> str:
    """Normalise page text so a repeated stamp collapses to a constant."""
    t = re.sub(r"\d+", "#", text.strip())
    return re.sub(r"\s+", " ", t)


# ==============================================================================
# STAGE 1 — PAGE TYPE DETECTION  (metadata only — both types use one engine)
# ==============================================================================
#
# PaddleOCR-VL-1.5 processes digital and scanned pages with the same call, so
# detection does not branch the pipeline. It is kept because the JSON should
# still report how many pages were scanned vs digital, and because a future
# optimisation could read a digital page's native text layer directly.
#
# The detector is boilerplate-aware: a page whose entire text is only a
# document-wide repeated stamp (e.g. a release banner) is 'scanned' even though
# it has a small text layer; character count is the primary signal otherwise.
# ==============================================================================

def _find_boilerplate(doc: fitz.Document) -> set[str]:
    """Find normalised page-text repeated across many pages (stamps/banners)."""
    total = len(doc)
    if total < 8:
        return set()
    counts: Counter = Counter()
    for page in doc:
        norm = _normalise_for_boilerplate(page.get_text("text"))
        if norm:
            counts[norm] += 1
    threshold = max(5, total * 0.20)
    return {txt for txt, c in counts.items() if c >= threshold}


def _classify_page(page: fitz.Page, boilerplate: set[str]) -> str:
    """Classify one page as 'scanned' or 'digital' (metadata tag)."""
    text = page.get_text("text")
    chars = len(text.strip())

    # GlyphlessFont => page was previously OCR'd => scanned
    fonts = page.get_fonts(full=False)
    if any("glyphless" in f[3].lower() or "invisible" in f[3].lower()
           for f in fonts):
        return "scanned"

    # page text is only the repeated boilerplate stamp => scanned
    if text.strip() and _normalise_for_boilerplate(text) in boilerplate:
        return "scanned"

    # substantial text => digital (regardless of word-quality: TOC dot-leaders
    # drag quality down on legitimately digital pages)
    if chars >= MIN_CHARS_DIGITAL:
        return "digital"

    # short page: word quality as final tie-breaker
    if text.strip() and _word_quality(text) >= WORD_QUALITY_THRESH:
        return "digital"
    return "scanned"


def detect_pages(pdf_path: Path) -> list[dict]:
    """Classify every page. Returns [{page, type, char_count}, ...]."""
    doc = fitz.open(str(pdf_path))
    boilerplate = _find_boilerplate(doc)
    results = []
    for i, page in enumerate(doc):
        results.append({
            "page":       i + 1,
            "type":       _classify_page(page, boilerplate),
            "char_count": len(page.get_text("text").strip()),
        })
    doc.close()
    return results


# ==============================================================================
# STAGE 2 — OCR ENGINE  (PaddleOCR-VL-1.5, with Tesseract fallback)
# ==============================================================================

class OCREngine:
    """
    Document-parsing engine. Primary: PaddleOCR-VL-1.5. Fallback: Tesseract.

    ocr_page(pil_image) -> {"markdown": str, "engine": str}
      markdown holds text AND tables (GitHub-flavoured Markdown tables) in
      reading order — produced by the VLM in a single pass.
    """

    def __init__(self):
        self.engine_name = "none"
        self._paddle = None
        self._init_engine()

    def _init_engine(self):
        # ---- PaddleOCR-VL-1.5 (primary) ----
        try:
            from paddleocr import PaddleOCRVL
            self._paddle = PaddleOCRVL()      # downloads weights on first run
            self.engine_name = "paddleocr-vl-1.5"
            print("  [OK] OCR engine: PaddleOCR-VL-1.5")
            return
        except Exception as exc:
            print(f"  [WARN] PaddleOCR-VL-1.5 unavailable: {exc}")

        # ---- Tesseract (fallback) ----
        try:
            import pytesseract  # noqa: F401
            self.engine_name = "tesseract-fallback"
            print("  [WARN] Falling back to Tesseract — weaker, run flagged degraded.")
            print("         For best results: pip install paddleocr paddlepaddle")
            return
        except Exception:
            pass

        print("  [ERROR] No OCR engine available. "
              "Install: pip install paddleocr  (or)  pip install pytesseract")

    # ----------------------------------------------------------------------
    def ocr_page(self, pil_image) -> dict:
        """Run the engine on one page image."""
        if self.engine_name == "paddleocr-vl-1.5":
            return {"markdown": self._ocr_paddle(pil_image),
                    "engine": self.engine_name}
        if self.engine_name == "tesseract-fallback":
            return {"markdown": self._ocr_tesseract(pil_image),
                    "engine": self.engine_name}
        return {"markdown": "", "engine": "none"}

    # ---- PaddleOCR-VL-1.5 -------------------------------------------------
    def _ocr_paddle(self, pil_image) -> str:
        """
        PaddleOCR-VL-1.5 predicts layout + text + tables and returns a
        structured result per image. We ask for Markdown so tables arrive
        as Markdown tables and text keeps reading order.
        """
        import numpy as np
        result = self._paddle.predict(np.array(pil_image))

        md_parts: list[str] = []
        for res in result:
            md = None
            # newer PaddleOCR result objects expose .markdown
            if hasattr(res, "markdown") and res.markdown:
                md = (res.markdown.get("markdown_texts")
                      if isinstance(res.markdown, dict) else res.markdown)
            if md is None and isinstance(res, dict):
                md = res.get("markdown") or res.get("text")
            if md:
                md_parts.append(md if isinstance(md, str) else str(md))
        return "\n\n".join(md_parts).strip()

    # ---- Tesseract fallback ----------------------------------------------
    def _ocr_tesseract(self, pil_image) -> str:
        """
        Plain-text OCR — fallback only. Tesseract has no reading-order model,
        so on multi-column pages it may split a clause number from its text.
        --psm 6 keeps lines as intact as possible.
        """
        import pytesseract
        return pytesseract.image_to_string(
            pil_image, lang="eng",
            config="--psm 6 -c preserve_interword_spaces=1").strip()


# ==============================================================================
# STAGE 3 — NOISE REMOVAL
# ==============================================================================

_TOC_LINE_RE = re.compile(r"^.{3,}\s*\.{3,}\s*\d{1,4}\s*$")   # "3. TITLE ...... 10"
_PAGENUM_RE  = re.compile(r"^\d{1,4}$")                       # bare page number
_STAMP_RE    = re.compile(r"^\d{1,4}\s+of\s+\d{1,4}$", re.IGNORECASE)  # "N of 196"


def detect_noise_lines(pages: list[dict]) -> set[str]:
    """Find lines repeated across many pages — running headers/footers/banners."""
    total = len(pages)
    if total < 4:
        return set()
    counts: Counter = Counter()
    for p in pages:
        seen = {ln.strip() for ln in p["markdown"].splitlines()
                if len(ln.strip()) >= NOISE_MIN_LINELEN}
        counts.update(seen)
    threshold = max(NOISE_MIN_REPEAT, total * NOISE_REPEAT_RATIO)
    return {ln for ln, c in counts.items() if c >= threshold}


def clean_page(markdown: str, noise: set[str]) -> str:
    """Remove repeated banner/footer noise and stray page numbers."""
    out = []
    for ln in markdown.splitlines():
        s = ln.strip()
        if s in noise or _PAGENUM_RE.match(s) or _STAMP_RE.match(s):
            continue
        if _TOC_LINE_RE.match(s):
            continue
        out.append(ln)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


# ==============================================================================
# STAGE 4 — CLAUSE PARSING
# ==============================================================================
#
# PaddleOCR-VL emits clause structure as Markdown headings and numbered lines.
# The Harbour contract uses TWO heading conventions in different sections, so
# the clause matcher must accept both:
#   - "### 1. Recitals"            number + dot   (Part-A clause style)
#   - "## 2 Description of Services" number, no dot (Service-Requirements style)
#   - "## I Introduction"          Roman numeral   (Service-Requirements style)
# Subclauses are numbered lines "N.N ..." or "N.N.N ..." — usually plain text,
# sometimes promoted by the VLM to a "### N.N" heading.
# ALLCAPS section labels ("## PRICING") are sub-headings inside a clause, kept
# as body text — NOT treated as clauses (that was the bug that collapsed every
# clause into one).
# ==============================================================================

# Clause: a Markdown heading whose text begins with a clause identifier.
# The identifier is EITHER:
#   - an arabic number, optional letter, optional dot:  "1", "12", "1A", "3."
#     (PaddleOCR-VL sometimes inserts a space, e.g. "1 A." for "1A." — the
#      optional space between number and letter is tolerated here)
#   - a Roman numeral:                                  "I", "II", "IV"
# followed by whitespace and a Title-case / ALLCAPS title.
# A heading whose number is multi-level (N.N, or the OCR-split "N A.N") is NOT
# a clause — that is a subclause; it is handled by _SUBCLAUSE_RE.
_CLAUSE_HEADING_RE = re.compile(
    r"^#{1,4}\s+"
    r"(\d{1,3}\s?[A-Z]?|[IVXL]{1,5})"  # arabic (1, 12, 1A, "1 A") OR roman
    r"\.?\s+"                          # optional dot, then whitespace
    r"(?![\d.]*\.\d)"                  # reject digit multi-level (1.1)
    r"(?![A-Z]\.?\d)"                  # reject letter multi-level (A.1, A1)
    r"([A-Z][A-Za-z].{2,90})\s*$"      # the title (>=2 leading letters)
)

# Subclause: a TWO-level number "N.N" — this is a true subclause.
# The leading number tolerates the SAME OCR artefacts the clause-heading regex
# does: PaddleOCR-VL sometimes splits "1A.1" as "1 A.1" (inserted space) or
# renders the leading "1" as "l"/"I". Without this, an alphanumeric clause like
# 1A loses every subclause (1A.1/1A.2/...), which then fall through to the body
# branch and — with no subclause open — used to be discarded entirely.
_SUBCLAUSE_RE = re.compile(
    r"^#{0,4}\s*([\dlI]{1,3}\s?[A-Z]?\.\d{1,3}[A-Z]?)\s+(.+?)\s*$"
)
# Sub-point: a THREE-level number "N.N.N" — this nests UNDER a subclause,
# it is NOT counted as a separate subclause (that was the count-inflation bug).
_SUBPOINT_RE = re.compile(
    r"^#{0,4}\s*([\dlI]{1,3}\s?[A-Z]?\.\d{1,3}[A-Z]?\.\d{1,3})\s+(.+?)\s*$"
)

# A plain ALLCAPS section sub-heading inside a clause (kept as body text).
_SECTION_LABEL_RE = re.compile(r"^#{1,4}\s+[A-Z][A-Z0-9 ,/&'\-()]{2,70}$")

# An appendix / schedule / annexure HEADING (start of a new appendix section).
# Tightened so it only fires on a real heading, not on sentence fragments that
# merely start with the word — e.g. "Schedule.", "Schedule; or", "Schedule 16;",
# "Appendix 2. The Service Provider...". A real heading is: the keyword, an
# identifier (number or roman), and then EITHER nothing (a bare "SCHEDULE 2")
# or a Title-case title — and the line must not run on as a sentence (no comma,
# semicolon, or sentence-ending dot, and no lower-case continuation word).
_APPENDIX_BOUNDARY_RE = re.compile(
    r"^#{0,4}\s*"
    r"(?i:appendix|annexure|annex|schedule|attachment|exhibit)\s+"  # keyword
    r"(?:\d{1,3}[A-Z]?|[IVXL]{1,6})"          # an identifier (number or roman)
    r"(?![.;,])"                              # not a sentence fragment after it
    r"(?:[ \-:]+[A-Z][^.;]*)?"                # optional Title-case title
    r"\s*$"                                   # ends cleanly (no run-on sentence)
)


def _clean_num(num: str) -> str:
    """Normalise an OCR'd clause / subclause number.

    Removes the space PaddleOCR-VL sometimes inserts ("1 A.1" -> "1A.1") and
    fixes a leading "1" that OCR rendered as "l" or "I" ("lA.1" -> "1A.1").
    Only the leading character is touched, so trailing clause letters (the "A"
    in "24A") are preserved.
    """
    num = re.sub(r"\s+", "", num.strip())   # "1 A.1" -> "1A.1"
    num = re.sub(r"^[lI]", "1", num)        # leading 1 mis-read as l / I
    return num


def parse_clauses(pages: list[dict]) -> dict:
    """
    Build a clause / subclause tree from the cleaned page Markdown.

    Hierarchy handled:
      - clause      "### N. Title" / "## N Title" / "## I Title"
      - subclause   "N.N ..."   — a true subclause, counted
      - sub-point   "N.N.N ..." — nests inside its subclause's text, NOT
                      counted as a separate subclause (prevents the count
                      from being inflated by deep numbering)
    Appendix / Schedule / Annexure headings are captured as their OWN clause
    (clause_number = e.g. "Appendix 3"), so the appendix's prose is preserved
    and its internal numbering (3.1, 3.1.1) nests under it rather than being
    absorbed into the preceding contract clause or discarded.

    Any body text that arrives before a clause's first subclause is kept in a
    clause-level preamble (folded into the rebuilt "clause" full-text), so a
    clause whose first subclause fails to parse never loses its content.

    Returns {"clauses": [...], "clause_count", "subclause_count"}.
    """
    clauses: list[dict] = []
    current_clause: dict | None = None
    current_sub: dict | None = None

    def _flush_sub():
        nonlocal current_sub
        if current_sub and current_clause is not None:
            current_sub["subclause_text"] = current_sub["subclause_text"].strip()
            current_clause["subclauses"].append(current_sub)
        current_sub = None

    def _add_body(text: str, sep: str = " "):
        """Append body text to the open subclause, or — if none is open — to
        the current clause's preamble, so it is never silently dropped."""
        if current_sub is not None:
            current_sub["subclause_text"] += sep + text
        elif current_clause is not None:
            current_clause["preamble"] += sep + text

    def _new_clause(num: str, title: str, pno: int, is_appendix: bool = False):
        nonlocal current_clause
        _flush_sub()
        if is_appendix:
            clause_number = num.strip()
        else:
            # space-removal only ("1 A" -> "1A"); do NOT run the leading
            # l/I -> 1 fix here, because a clause number can be a Roman
            # numeral (I, II, IV). That fix is applied to subclause numbers,
            # which are always digit-based.
            clause_number = re.sub(r"(\d)\s+([A-Z])", r"\1\2", num.strip())
        current_clause = {
            "clause_number": clause_number,
            "clause_title":  title.strip(),
            "start_page":    pno,
            "is_appendix":   is_appendix,
            "preamble":      "",
            "subclauses":    [],
        }
        clauses.append(current_clause)

    for page in pages:
        pno = page["page"]
        for raw in page["markdown"].splitlines():
            line = raw.strip()
            if not line:
                if current_sub:
                    current_sub["subclause_text"] += "\n"
                continue

            # ---- appendix / schedule boundary: start a NEW appendix clause ----
            # Previously this closed the clause and skipped everything that
            # followed, which silently dropped substantive prose appendices
            # (e.g. "Appendix 3 Detailed Service Plans Requirements"). Now the
            # appendix becomes its own clause: its prose is preserved and its
            # internal numbering (3.1, 3.1.1) nests under it, not the preceding
            # contract clause.
            if _APPENDIX_BOUNDARY_RE.match(line):
                heading = re.sub(r"^#+\s*", "", line).strip()
                m_app = re.match(
                    r"(?i:(appendix|annexure|annex|schedule|attachment|exhibit)"
                    r"\s+(\d{1,3}[A-Z]?|[IVXL]{1,6}))(.*)$", heading)
                if m_app:
                    app_num = re.sub(r"\s+", " ",
                                     f"{m_app.group(1).title()} {m_app.group(2)}").strip()
                    app_title = m_app.group(3).strip(" -:") or app_num
                else:
                    app_num = app_title = heading
                _new_clause(app_num, app_title, pno, is_appendix=True)
                continue

            m_clause = _CLAUSE_HEADING_RE.match(line)
            m_subpoint = _SUBPOINT_RE.match(line)   # N.N.N — check before N.N
            m_sub = _SUBCLAUSE_RE.match(line)

            # ---- a real clause heading ----
            if m_clause:
                _new_clause(m_clause.group(1), m_clause.group(2), pno)
                continue

            # ---- a sub-point "N.N.N" : nest inside the current subclause ----
            # It is body content of its parent subclause, not a new subclause.
            if m_subpoint:
                num = _clean_num(m_subpoint.group(1))
                body = m_subpoint.group(2).strip()
                if current_sub is not None:
                    current_sub["subclause_text"] += f"\n{num} {body}"
                elif current_clause is not None:
                    # a sub-point with no open subclause — start the parent
                    # subclause implicitly from its first two number levels
                    parent = ".".join(num.split(".")[:2])
                    current_sub = {
                        "subclause_number": parent,
                        "subclause_title":  "(untitled)",
                        "subclause_text":   f"{num} {body}",
                        "page":             pno,
                    }
                continue

            # ---- a subclause line "N.N" ----
            if m_sub:
                _flush_sub()
                sub_num = _clean_num(m_sub.group(1))
                if current_clause is None:
                    # no clause open: stray numbered line — skip rather than
                    # invent a clause
                    continue
                current_sub = {
                    "subclause_number": sub_num,
                    "subclause_title":  m_sub.group(2).strip()[:120],
                    "subclause_text":   m_sub.group(2).strip(),
                    "page":             pno,
                }
                continue

            # ---- ALLCAPS section label: keep as body text ----
            if _SECTION_LABEL_RE.match(line):
                label = re.sub(r"^#+\s*", "", line)
                _add_body(label, sep="\n")
                continue

            # ---- ordinary body text: keep it (subclause, else clause preamble) ----
            _add_body(line)

    _flush_sub()
    return {
        "clauses":         clauses,
        "clause_count":    len(clauses),
        "subclause_count": sum(len(c["subclauses"]) for c in clauses),
    }


# ==============================================================================
# STAGE 5 — TABLE EXTRACTION
# ==============================================================================
#
# PaddleOCR-VL-1.5 emits tables in the SAME pass as the text, but in one of
# two formats depending on the table:
#   - simple tables  -> GitHub-flavoured Markdown:   | a | b | c |
#   - complex tables -> HTML:  <table><tr><td rowspan=2>..</td>..</tr></table>
#     (used when the table has merged cells — rowspan / colspan)
#
# Stage 5 parses BOTH. The HTML branch expands rowspan/colspan so every row
# ends up rectangular (merged cells are repeated into the spanned positions),
# which is what a downstream consumer needs. This is still a parse step, not
# a separate vision model — the VLM already did the table recognition.
# ==============================================================================

import html as _html
from html.parser import HTMLParser

_MD_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")             # | a | b | c |
_MD_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")       # |---|:--:|---|
_HTML_TABLE_RE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)


def _split_md_row(line: str) -> list[str]:
    """Split a Markdown table row into stripped cell strings."""
    return [c.strip() for c in _MD_ROW_RE.match(line).group(1).split("|")]


class _TableHTMLParser(HTMLParser):
    """
    Parse a single <table> into a rectangular list-of-rows, expanding
    rowspan and colspan so every cell occupies its full grid footprint.
    """

    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []      # finished grid
        self._cur: list[str] = []            # current <tr> being built
        self._cell = None                    # text buffer for current cell
        self._span = (1, 1)                  # (rowspan, colspan) of current cell
        self._pending: dict[int, tuple] = {} # col -> (text, remaining_rowspan)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "tr":
            self._cur = []
        elif tag in ("td", "th"):
            self._cell = []
            try:
                rs = int(a.get("rowspan", 1))
            except ValueError:
                rs = 1
            try:
                cs = int(a.get("colspan", 1))
            except ValueError:
                cs = 1
            self._span = (max(1, rs), max(1, cs))

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            text = _html.unescape(" ".join(self._cell).strip())
            text = re.sub(r"\s+", " ", text)
            self._cur.append((text, self._span))
            self._cell = None
        elif tag == "tr":
            self._emit_row()

    def _emit_row(self):
        """Place the collected cells into the grid, honouring row/col spans
        carried down from earlier rows."""
        row_idx = len(self.rows)
        out: list[str] = []
        col = 0
        cells = list(self._cur)
        while cells or col in self._pending:
            # a cell spanning down from a previous row occupies this column
            if col in self._pending:
                text, remaining = self._pending[col]
                out.append(text)
                if remaining - 1 > 0:
                    self._pending[col] = (text, remaining - 1)
                else:
                    del self._pending[col]
                col += 1
                continue
            if not cells:
                break
            text, (rs, cs) = cells.pop(0)
            for _ in range(cs):
                out.append(text)
                if rs > 1:
                    self._pending[col] = (text, rs - 1)
                col += 1
        self.rows.append(out)
        self._cur = []


def _parse_html_table(html_str: str) -> tuple[list[str], list[list[str]]]:
    """Return (headers, body_rows) for one HTML <table> string."""
    p = _TableHTMLParser()
    try:
        p.feed(html_str)
    except Exception:
        return [], []
    rows = [r for r in p.rows if any(c.strip() for c in r)]
    if not rows:
        return [], []
    width = max(len(r) for r in rows)
    rows = [(r + [""] * width)[:width] for r in rows]
    return rows[0], rows[1:]


def extract_tables(pages: list[dict]) -> dict:
    """
    Pull tables out of the OCR Markdown — both HTML <table> blocks and
    GitHub-flavoured Markdown pipe tables.

    Returns {"tables": [...], "table_count": int}.
    """
    tables: list[dict] = []

    def _add(pno: int, headers: list[str], body: list[list[str]], fmt: str):
        if not headers or len(body) < 1:
            return
        n_on_page = len([t for t in tables if t["page"] == pno])
        tables.append({
            "table_id":  f"T{pno:03d}_{n_on_page + 1}",
            "page":      pno,
            "format":    fmt,
            "headers":   headers,
            "rows":      body,
            "row_count": len(body),
        })

    for page in pages:
        pno = page["page"]
        md = page["markdown"]

        # ---- HTML tables (complex tables with merged cells) ----
        for m in _HTML_TABLE_RE.finditer(md):
            headers, body = _parse_html_table(m.group(0))
            _add(pno, headers, body, "html")

        # ---- Markdown pipe tables (simple tables) ----
        # skip any lines that were inside an HTML table block
        lines = md.splitlines()
        i = 0
        while i < len(lines):
            if not _MD_ROW_RE.match(lines[i]):
                i += 1
                continue
            block = []
            while i < len(lines) and (_MD_ROW_RE.match(lines[i])
                                      or _MD_SEP_RE.match(lines[i])):
                block.append(lines[i])
                i += 1
            data_rows = [r for r in block if not _MD_SEP_RE.match(r)]
            if len(data_rows) < 2:
                continue
            headers = _split_md_row(data_rows[0])
            width = len(headers)
            body = [(_split_md_row(r) + [""] * width)[:width]
                    for r in data_rows[1:]]
            _add(pno, headers, body, "markdown")

    return {"tables": tables, "table_count": len(tables)}


# ==============================================================================
# STAGE 8 — TABLE-OF-CONTENTS VERIFICATION
# ==============================================================================
#
# Coverage check. The user names the TOC page(s) with --toc-pages (REQUIRED).
# Those pages are OCR'd SEPARATELY (they never pass through Stage 3 noise
# removal, so the existing parsing is untouched), their entries are parsed
# into a checklist, and the checklist is compared against the clauses that
# were actually extracted.
#
# A TOC line looks like:  "9.  GENERAL WARRANTIES ............ 17"
# i.e. <number> <title> <dot leaders> <page>.
# ==============================================================================

# TOC entry: leading clause number, a title, dot-leaders, a trailing page no.
_TOC_ENTRY_RE = re.compile(
    r"^\s*#{0,4}\s*"
    r"(\d{1,3}[A-Z]?(?:\.\d{1,3})?)\.?\s+"   # 1  group: clause/subclause number
    r"(.+?)"                                 # 2  group: the title
    r"\s*\.{2,}\s*"                          # dot leaders
    r"(\d{1,4})\s*$"                         # 3  group: the stated page number
)
# fallback: a TOC line where OCR dropped the dot-leaders — number, title, page
_TOC_ENTRY_NODOTS_RE = re.compile(
    r"^\s*#{0,4}\s*(\d{1,3}[A-Z]?)\.?\s+([A-Z][A-Za-z][^\d]{2,70}?)\s+(\d{1,4})\s*$"
)


def _normalise_title(text: str) -> str:
    """Lower-case, strip punctuation and spaces — for fuzzy title matching."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def parse_toc(toc_markdown_pages: list[str]) -> list[dict]:
    """
    Parse TOC page text into a list of {number, title, stated_page} entries.
    Only top-level clause entries (number without a dot) are kept as the
    checklist — subclause lines are ignored so the check stays clause-level.
    """
    entries: list[dict] = []
    for md in toc_markdown_pages:
        for raw in md.splitlines():
            line = raw.strip()
            if not line:
                continue
            m = _TOC_ENTRY_RE.match(line) or _TOC_ENTRY_NODOTS_RE.match(line)
            if not m:
                continue
            num = m.group(1).strip()
            # keep only top-level clauses (no dot in the number)
            if "." in num:
                continue
            entries.append({
                "number":      num,
                "title":       m.group(2).strip(),
                "stated_page": int(m.group(3)),
            })
    # de-duplicate on number (a TOC entry can wrap onto two lines)
    seen, out = set(), []
    for e in entries:
        if e["number"] not in seen:
            seen.add(e["number"])
            out.append(e)
    return out


def verify_against_toc(toc_entries: list[dict],
                       clauses: list[dict],
                       extracted_range: tuple[int, int] | None = None) -> dict:
    """
    Compare the TOC checklist against the extracted clauses.

    A TOC entry is 'matched' when a clause has the same number AND a
    sufficiently similar title (normalised).

    extracted_range: if given, e.g. (43, 100), the check is narrowed to
      clauses that should live in that PDF-page range. Each matched clause
      contributes its (TOC stated_page -> real start_page) offset; the
      median offset is then used to project unmatched TOC entries onto
      real PDF pages, so we can tell a 'should-have-been-in-this-batch'
      failure apart from a 'lives-outside-this-batch' expected absence.
    """
    by_num: dict[str, list[dict]] = {}
    for c in clauses:
        by_num.setdefault(str(c.get("clause_number", "")).strip(), []).append(c)

    matched, in_range_missing, out_of_range, title_mismatch = [], [], [], []
    offsets: list[int] = []   # (real_pdf_page - toc_stated_page) for matches

    # ---- first pass: find matches and gather the TOC-vs-PDF page offset ----
    match_records = []  # (toc_entry, hit_clause) for matched ones
    unresolved   = []   # (toc_entry, candidates)  for unmatched
    for e in toc_entries:
        candidates = by_num.get(e["number"], [])
        toc_t = _normalise_title(e["title"])
        hit = None
        for c in candidates:
            ct = _normalise_title(c.get("clause_title", ""))
            if ct and toc_t and (ct in toc_t or toc_t in ct
                                 or ct[:12] == toc_t[:12]):
                hit = c
                break
        if hit:
            match_records.append((e, hit))
            sp, real = e.get("stated_page"), hit.get("start_page")
            if isinstance(sp, int) and isinstance(real, int):
                offsets.append(real - sp)
        else:
            unresolved.append((e, candidates))

    # median offset — robust to one bad match
    page_offset = 0
    if offsets:
        offsets.sort()
        page_offset = offsets[len(offsets) // 2]

    def _in_range(page: int | None) -> bool:
        if extracted_range is None or page is None:
            return True
        lo, hi = extracted_range
        return lo <= page <= hi

    # ---- second pass: classify everything using the inferred offset ----
    for e, hit in match_records:
        matched.append({**e, "found_on_page": hit.get("start_page")})

    for e, candidates in unresolved:
        # estimate where this TOC clause SHOULD have been in the PDF
        stated = e.get("stated_page")
        estimated_pdf = (stated + page_offset
                          if isinstance(stated, int) else None)
        if candidates:
            # number found, title differs
            cand_page = candidates[0].get("start_page")
            entry = {
                "number":           e["number"],
                "toc_title":        e["title"],
                "extracted_title":  candidates[0].get("clause_title", ""),
                "found_on_page":    cand_page,
            }
            if _in_range(cand_page):
                title_mismatch.append(entry)
            else:
                out_of_range.append({**e, "estimated_pdf_page": estimated_pdf,
                                      "reason": "title mismatch outside batch"})
        else:
            # number not found at all — use the estimate to decide
            if _in_range(estimated_pdf):
                in_range_missing.append({**e,
                                          "estimated_pdf_page": estimated_pdf})
            else:
                out_of_range.append({**e,
                                      "estimated_pdf_page": estimated_pdf,
                                      "reason": "outside this batch"})

    # extracted clauses NOT in the TOC (e.g. mis-detected headings)
    toc_nums = {e["number"] for e in toc_entries}
    extra = [str(c.get("clause_number", "")) for c in clauses
             if str(c.get("clause_number", "")).strip() not in toc_nums]

    expected_in_range = len(matched) + len(in_range_missing) + len(title_mismatch)
    coverage = (round(100.0 * len(matched) / expected_in_range, 1)
                if expected_in_range else 0.0)

    return {
        "toc_clause_count":         len(toc_entries),
        "extracted_clause_count":   len(clauses),
        "extracted_range":          list(extracted_range) if extracted_range else None,
        "toc_to_pdf_page_offset":   page_offset,
        "expected_in_this_range":   expected_in_range,
        "matched_count":            len(matched),
        "coverage_percent":         coverage,
        "in_range_missing":         in_range_missing,
        "title_mismatches":         title_mismatch,
        "out_of_range":             out_of_range,
        "extra_not_in_toc":         sorted(set(extra)),
    }


# ==============================================================================
# STAGE 7 — OUTPUT ASSEMBLY
# ==============================================================================
#
# The output JSON is written DIRECTLY in the final delivery schema:
#
#   { "clauses": [ { "clause", "clause number", "clause title",
#                    "subclauses": [ { "subclause number",
#                                      "subclause title",
#                                      "subclause text" } ] } ],
#     "tables":  [ { "table_id", "page", "format",
#                    "headers", "rows", "row_count" } ] }
#
# Keys use SPACES (not underscores). Each clause carries a "clause" field =
# the full reconstructed clause text. No run metadata is written — the file
# is ready to consume by Task 2 with no post-processing.
# ==============================================================================

def _clause_full_text(clause: dict) -> str:
    """Rebuild the 'clause' field: title line, optional preamble (body text
    that appeared before the first subclause), then every subclause's
    number+title header and body text, in order."""
    parts: list[str] = []
    if clause.get("clause_title"):
        parts.append(clause["clause_title"])
    if clause.get("preamble", "").strip():
        parts.append(clause["preamble"].strip())
    for sub in clause.get("subclauses", []):
        header = " ".join(x for x in (sub.get("subclause_number", ""),
                                      sub.get("subclause_title", "")) if x).strip()
        if header:
            parts.append(header)
        if sub.get("subclause_text"):
            parts.append(sub["subclause_text"])
    return "\n".join(parts).strip()


def build_output(pdf_path: Path, detect: list[dict], ocr_pages: list[dict],
                  parsed: dict, table_result: dict, engine_name: str,
                  elapsed: float) -> dict:
    """
    Assemble the final JSON in the delivery schema (clauses + tables only,
    spaced keys). pdf_path / detect / ocr_pages / engine_name / elapsed are
    accepted for signature compatibility but not written to the output.
    """
    clauses_out = []
    for c in parsed["clauses"]:
        clauses_out.append({
            "clause":        _clause_full_text(c),
            "clause number": c.get("clause_number", ""),
            "clause title":  c.get("clause_title", ""),
            "subclauses": [
                {
                    "subclause number": s.get("subclause_number", ""),
                    "subclause title":  s.get("subclause_title", ""),
                    "subclause text":   s.get("subclause_text", ""),
                }
                for s in c.get("subclauses", [])
            ],
        })

    tables_out = []
    for t in table_result["tables"]:
        rows = t.get("rows", [])
        tables_out.append({
            "table_id":  t.get("table_id", ""),
            "page":      t.get("page", 0),
            "format":    t.get("format", ""),
            "headers":   t.get("headers", []),
            "rows":      rows,
            "row_count": t.get("row_count", len(rows)),
        })

    return {
        "clauses": clauses_out,
        "tables":  tables_out,
    }


def validate(output: dict) -> list[str]:
    """Light schema check. Returns a list of issue strings (empty = ok)."""
    issues = []
    for f in ["clauses", "tables"]:
        if f not in output:
            issues.append(f"missing field: {f}")
    return issues


# ==============================================================================
# ORCHESTRATOR
# ==============================================================================

def run(pdf_path: Path, output_dir: Path,
        max_pages: int | None = None,
        only_pages: list[int] | None = None,
        page_range: tuple[int, int] | None = None,
        toc_pages: list[int] | None = None,
        progress_callback=None) -> dict:
    """
    Run the full pipeline.

    toc_pages: list of page numbers that contain the Table of Contents.
      Stage 7 now ALWAYS runs and compares the TOC against the extracted
      clauses (coverage check). The page(s) must be supplied on every run —
      the CLI makes --toc-pages required, so a normal command-line run always
      has them. Those pages are OCR'd separately and do NOT pass through noise
      removal or clause parsing — the existing stages are unaffected. If this
      is None (a programmatic call that forgot to pass TOC pages) Stage 7 is
      skipped with a warning rather than crashing.

    progress_callback: optional callable(done, total, page_no, message).
      Called once per page during Stage 2 so a GUI can show live progress.
      Ignored when None (normal CLI use).
    """
    t0 = time.time()
    print(f"\n{'='*64}\n  DVS TASK 1 — UNIFIED EXTRACTION PIPELINE\n"
          f"  Build: 2026-06-01 (no Stage6 enrichment; Stage7 TOC always-on, console-only)\n"
          f"  Engine: PaddleOCR-VL-1.5  |  Input: {pdf_path.name}\n{'='*64}")

    def _stage(msg: str):
        """Print a stage banner and report it to the GUI callback (if any).
        A done value of -1 marks this as a stage message, not page progress."""
        print(f"\n{msg}")
        if progress_callback is not None:
            try:
                progress_callback(-1, 0, 0, msg)
            except Exception:
                pass

    # ---- Stage 1: detect ----
    _stage("[Stage 1] Detecting page types (metadata)...")
    detect = detect_pages(pdf_path)
    print(f"  Total: {len(detect)}  |  digital: "
          f"{sum(1 for p in detect if p['type']=='digital')}  |  scanned: "
          f"{sum(1 for p in detect if p['type']=='scanned')}")

    # which pages to process — by default ALL of them (one engine, one path)
    if page_range:
        lo, hi = page_range
        target = [p["page"] for p in detect if lo <= p["page"] <= hi]
    elif only_pages:
        target = [p for p in only_pages if 1 <= p <= len(detect)]
    else:
        target = [p["page"] for p in detect]
    if max_pages:
        target = target[:max_pages]
    print(f"  Pages to process: {len(target)}"
          + (f"  (range {target[0]}-{target[-1]})" if target else ""))

    # ---- Stage 2: OCR ----
    _stage(f"[Stage 2] Loading OCR engine (first run downloads ~2GB model)...")
    engine = OCREngine()
    if engine.engine_name == "none":
        print("  Aborting — no OCR engine available.")
        return {}
    _stage(f"[Stage 2] Parsing {len(target)} pages with PaddleOCR-VL-1.5...")

    doc = fitz.open(str(pdf_path))
    ocr_pages: list[dict] = []
    empty_pages: list[int] = []
    for idx, pno in enumerate(target, 1):
        img = _render_page(doc[pno - 1])
        res = engine.ocr_page(img)
        md = res["markdown"]
        stripped = md.strip()
        # an empty page = the OCR engine returned nothing usable
        is_empty = len(stripped) == 0
        # low quality = some text, but it looks like noise.
        # Measure quality on the prose only: strip HTML table markup first,
        # otherwise a perfectly good table page scores low just for its tags.
        prose_only = re.sub(r"<[^>]+>", " ", md)          # drop HTML tags
        prose_only = re.sub(r"\|", " ", prose_only)        # drop md table pipes
        is_low_q = (not is_empty) and _word_quality(prose_only) < 0.20
        if is_empty:
            empty_pages.append(pno)
        ocr_pages.append({
            "page": pno, "markdown": md, "engine": res["engine"],
            "type": next(p["type"] for p in detect if p["page"] == pno),
            "flag_low_quality": is_low_q,
            "flag_empty": is_empty,
        })
        if is_empty:
            tag = "  [EMPTY — engine returned no text]"
        elif is_low_q:
            tag = "  [LOW QUALITY]"
        else:
            tag = ""
        print(f"  ({idx}/{len(target)}) page {pno}: {len(stripped)} chars{tag}")
        if progress_callback is not None:
            try:
                progress_callback(idx, len(target), pno,
                                  f"page {pno}: {len(stripped)} chars{tag}")
            except Exception:
                pass   # a GUI callback must never break the pipeline
    doc.close()
    if empty_pages:
        print(f"  NOTE: {len(empty_pages)} page(s) returned no text: {empty_pages}")

    # ---- Stage 3: clean ----
    _stage("[Stage 3] Removing repeated banner/footer noise...")
    noise = detect_noise_lines(ocr_pages)
    for p in ocr_pages:
        p["markdown"] = clean_page(p["markdown"], noise)
    print(f"  Removed {len(noise)} repeated noise patterns")

    # ---- Stage 4: clauses ----
    _stage("[Stage 4] Parsing clause structure...")
    parsed = parse_clauses(ocr_pages)
    print(f"  Clauses: {parsed['clause_count']}  |  "
          f"Subclauses: {parsed['subclause_count']}")

    # ---- Stage 5: tables ----
    _stage("[Stage 5] Extracting tables from the Markdown...")
    table_result = extract_tables(ocr_pages)
    print(f"  Tables: {table_result['table_count']}")

    # ---- Stage 6: output ----
    _stage("[Stage 6] Building JSON output...")
    elapsed = time.time() - t0
    output = build_output(pdf_path, detect, ocr_pages, parsed,
                          table_result, engine.engine_name, elapsed)

    # ---- Stage 7: TOC verification (ALWAYS runs; --toc-pages is required) ----
    # The CLI requires --toc-pages, so a normal run always lands in the first
    # branch. The else-branch only guards a programmatic caller that forgot to
    # pass TOC pages — it warns and skips rather than crashing the run.
    if toc_pages:
        _stage(f"[Stage 7] Verifying against Table of Contents "
               f"(pages {toc_pages})...")
        toc_md: list[str] = []
        doc2 = fitz.open(str(pdf_path))
        for tp in toc_pages:
            if 1 <= tp <= len(doc2):
                res = engine.ocr_page(_render_page(doc2[tp - 1]))
                toc_md.append(res["markdown"])
        doc2.close()
        toc_entries = parse_toc(toc_md)

        # narrow the check to the PDF pages we actually extracted, using
        # each extracted clause's start_page (so any TOC-vs-PDF page offset
        # is sidestepped — clauses know their own real PDF sheet number)
        if target:
            extracted_range = (min(target), max(target))
        else:
            extracted_range = None

        report = verify_against_toc(toc_entries, parsed["clauses"],
                                    extracted_range=extracted_range)
        # NOTE: the TOC report is a console-only diagnostic for the operator.
        # It is intentionally NOT written into the output JSON, which stays a
        # clean { clauses, tables } deliverable for Task 2.

        rng_txt = (f" (pages {extracted_range[0]}-{extracted_range[1]})"
                   if extracted_range else "")
        print(f"  TOC lists {report['toc_clause_count']} clauses total; "
              f"{report['expected_in_this_range']} expected in this batch"
              f"{rng_txt}")
        print(f"  Page offset (TOC->PDF): {report['toc_to_pdf_page_offset']}")
        print(f"  Matched {report['matched_count']}  |  "
              f"coverage in this batch: {report['coverage_percent']}%")
        if report["in_range_missing"]:
            miss = ", ".join(f"{m['number']} {m['title']}"
                             for m in report["in_range_missing"])
            print(f"  MISSING (should have been in this batch): {miss}")
        if report["title_mismatches"]:
            print(f"  {len(report['title_mismatches'])} title mismatch(es) "
                  f"in this batch — number found but title differs:")
            for tm in report["title_mismatches"]:
                print(f"      [{tm['number']}] TOC: '{tm['toc_title']}'  vs  "
                      f"extracted: '{tm['extracted_title']}'")
        if report["out_of_range"]:
            print(f"  {len(report['out_of_range'])} TOC clause(s) outside "
                  f"this batch — not counted (expected)")
        if report["extra_not_in_toc"]:
            print(f"  Extracted but not in TOC: "
                  f"{', '.join(report['extra_not_in_toc'])}")
    else:
        _stage("[Stage 7] SKIPPED — no TOC pages supplied. "
               "Pass --toc-pages to run the coverage check (it is required "
               "on the CLI; this skip only happens on a programmatic call).")

    issues = validate(output)
    print("  Schema OK" if not issues else "  Schema issues: " + "; ".join(issues))

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = re.sub(r"[^\w]", "_", pdf_path.stem)[:40]
    # range suffix keeps batch outputs from overwriting each other
    if target:
        span = f"_p{target[0]}-{target[-1]}"
    else:
        span = ""
    out_path = output_dir / f"dvs_extract_{stem}{span}_{stamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*64}")
    print(f"  DONE in {elapsed:.1f}s")
    print(f"  Engine    : {engine.engine_name}")
    print(f"  Pages     : {len(ocr_pages)}")
    print(f"  Clauses   : {parsed['clause_count']}")
    print(f"  Tables    : {table_result['table_count']}")
    print(f"  Output    : {out_path}")
    if engine.engine_name != "paddleocr-vl-1.5":
        print(f"  NOTE      : degraded run — PaddleOCR-VL-1.5 was unavailable, "
              f"Tesseract fallback used")
    empty = [p["page"] for p in ocr_pages if p.get("flag_empty")]
    if empty:
        print(f"  NOTE      : {len(empty)} page(s) returned no text: {empty}")
    print(f"{'='*64}\n")
    return output


# ==============================================================================
# CLI
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="DVS Task 1 — unified PDF extraction (PaddleOCR-VL-1.5)")
    ap.add_argument("pdf_path", type=Path, help="path to the PDF")
    ap.add_argument("--output-dir", type=Path, default=Path("output"),
                    help="directory for the JSON output (default: ./output)")
    ap.add_argument("--max-pages", type=int, default=None,
                    help="process only the first N pages (quick test)")
    ap.add_argument("--pages", type=int, nargs="+", default=None,
                    help="process only these specific page numbers")
    ap.add_argument("--range", type=int, nargs=2, metavar=("START", "END"),
                    default=None, dest="page_range",
                    help="process pages START..END inclusive (for batching a "
                         "large PDF, e.g. --range 1 40 then --range 41 80)")
    ap.add_argument("--toc-pages", type=str, nargs="+", required=True,
                    metavar="P",
                    help="REQUIRED. Page(s) of the Table of Contents. Accepts "
                         "single pages, ranges, or both. Stage 7 ALWAYS runs "
                         "and checks the extracted clauses against the TOC. "
                         "Examples: --toc-pages 3 4    |    --toc-pages 3-9    "
                         "|    --toc-pages 3-5 8 10-12")
    args = ap.parse_args()

    if not args.pdf_path.exists():
        print(f"ERROR: file not found: {args.pdf_path}")
        sys.exit(1)

    # Expand --toc-pages tokens. Each token is either "N" or "N-M".
    # --toc-pages is required, so args.toc_pages is always a non-empty list.
    toc_pages_list = []
    for tok in args.toc_pages:
        if "-" in tok:
            try:
                lo, hi = (int(x) for x in tok.split("-", 1))
            except ValueError:
                print(f"ERROR: bad --toc-pages range: {tok!r}")
                sys.exit(1)
            if lo > hi:
                lo, hi = hi, lo
            toc_pages_list.extend(range(lo, hi + 1))
        else:
            try:
                toc_pages_list.append(int(tok))
            except ValueError:
                print(f"ERROR: bad --toc-pages value: {tok!r}")
                sys.exit(1)
    # de-duplicate while preserving order
    seen = set()
    toc_pages_list = [p for p in toc_pages_list
                      if not (p in seen or seen.add(p))]

    page_range = tuple(args.page_range) if args.page_range else None
    run(args.pdf_path, args.output_dir, args.max_pages, args.pages, page_range,
        toc_pages=toc_pages_list)


if __name__ == "__main__":
    main()