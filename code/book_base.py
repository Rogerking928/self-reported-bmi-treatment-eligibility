"""Word layout helpers used by book.py (build and the functions it calls).

Vendored unchanged from the author's earlier project (book.py of a study on obstetric volume and maternal death), where
the same layout was used: US Letter landscape, 1.0 in / 0.9 in margins, one item per page, Times New Roman, 8 pt
three-rule tables, 700 dpi figures scaled into an 8.2 x 5.4 in box.
"""
import re
from PIL import Image
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Inches

FONT = "Times New Roman"
PAGE_W, PAGE_H = Inches(11.0), Inches(8.5)
MARGIN_LR, MARGIN_TB = Inches(1.0), Inches(0.9)
BOX_W, BOX_H = 8.2, 5.4
MAX_SCALE = 1.5


# ------------------------------------------------------------------ docx
def run(p, text, size=None, bold=None, italic=None):
    r = p.add_run(text)
    r.font.name = FONT
    r._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), FONT)
    if size: r.font.size = Pt(size)
    if bold is not None: r.bold = bold
    if italic is not None: r.italic = italic
    return r


def run_sci(p, text, size, bold=None):
    """把學名切出來設斜體，其餘正體。"""
    for part in re.split(r"(<i>.*?</i>)", text):
        if part.startswith("<i>"):
            run(p, part[3:-4], size, bold=bold, italic=True)
        elif part:
            run(p, part, size, bold=bold, italic=False)


def para(doc, before=0, after=0, align=None, page_break=False, keep=False):
    p = doc.add_paragraph()
    f = p.paragraph_format
    f.space_before, f.space_after, f.line_spacing = Pt(before), Pt(after), 1.0
    f.page_break_before, f.keep_with_next, f.keep_together = page_break, keep, keep
    if align is not None: p.alignment = align
    return p


def borders(cell, *edges):
    tcPr = cell._tc.get_or_add_tcPr()
    for old in tcPr.findall(qn("w:tcBorders")): tcPr.remove(old)
    b = OxmlElement("w:tcBorders")
    for e in edges:
        el = OxmlElement(f"w:{e}")
        for k, v in (("val", "single"), ("sz", "8"), ("space", "0"), ("color", "000000")):
            el.set(qn(f"w:{k}"), v)
        b.append(el)
    tcPr.append(b)


def add_table(doc, item, first):
    p = para(doc, after=4, page_break=not first, keep=True)
    label, rest = item["caption"].split(". ", 1)
    run(p, label + ". ", 11.5, bold=True); run(p, rest, 11.5, bold=True)
    rows = item["rows"]
    ncol = len(rows[0])
    t = doc.add_table(rows=len(rows), cols=ncol)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    first_w = (2.6 if ncol > 4 else min(3.6, BOX_W * 0.34)) if ncol > 2 else BOX_W * 0.55
    widths = [first_w] + [(BOX_W - first_w) / (ncol - 1)] * (ncol - 1)
    for i, row in enumerate(rows):
        is_group = i > 0 and all(c == "" for c in row[1:])
        for j, text in enumerate(row):
            c = t.cell(i, j)
            c.width = Inches(widths[j])
            cp = c.paragraphs[0]
            cp.paragraph_format.space_after = Pt(0); cp.paragraph_format.line_spacing = 1.0
            cp.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            run_sci(cp, str(text), 8, bold=(i == 0 or (is_group and j == 0)))
    last = len(rows) - 1
    for j in range(ncol):
        borders(t.cell(0, j), "top", "bottom")
        if last >= 1: borders(t.cell(1, j), *(("top", "bottom") if last == 1 else ("top",)))
        if last > 1: borders(t.cell(last, j), "bottom")
    trPr = t.rows[0]._tr.get_or_add_trPr()
    h = OxmlElement("w:tblHeader"); h.set(qn("w:val"), "true"); trPr.append(h)
    if item.get("note"):
        n = para(doc, before=3)
        run_sci(n, item["note"], 9)


def add_figure(doc, item, first):
    p = para(doc, after=4, page_break=not first, keep=True)
    run(p, item["caption"], 11.5, bold=True)
    if item.get("note"):
        n = para(doc, after=6, keep=True)
        run_sci(n, item["note"], 9.5)
    w_px, h_px = Image.open(item["path"]).size
    w_in, h_in = w_px / 700, h_px / 700
    s = min(MAX_SCALE, BOX_W / w_in, BOX_H / h_in)
    ip = para(doc, align=WD_ALIGN_PARAGRAPH.CENTER)
    ip.add_run().add_picture(str(item["path"]), width=Inches(w_in * s), height=Inches(h_in * s))


def build(items, path, supp=()):
    """一份 Word：主文的表、主文的圖，接著補充的表、補充的圖（使用者 2026-10-04：圖表 Word 一個就好）。"""
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name, st.font.size = FONT, Pt(8)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width, sec.page_height = PAGE_W, PAGE_H
    sec.left_margin = sec.right_margin = MARGIN_LR
    sec.top_margin = sec.bottom_margin = MARGIN_TB
    tabs = [i for i in items if i["kind"] == "table"]
    figs = [i for i in items if i["kind"] == "figure"]
    stabs = [i for i in supp if i["kind"] == "table"]
    sfigs = [i for i in supp if i["kind"] == "figure"]
    for k, it in enumerate(tabs + figs + stabs + sfigs):
        (add_table if it["kind"] == "table" else add_figure)(doc, it, k == 0)
    doc.save(path)
    return len(tabs) + len(stabs), len(figs) + len(sfigs)

