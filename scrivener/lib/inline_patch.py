"""Monkey-patch ReportLab's inline backColor rendering to use
rounded rectangles with padding derived from font metrics."""

from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import paragraph as _mod

_orig = _mod._do_post_text

RADIUS = 2


def _patched(tx):
    xs = tx.XtraState
    if xs.backColors:
        f = xs.f
        fs = f.fontSize
        fn = getattr(f, 'fontName', None) or getattr(f, 'name', 'Helvetica')
        y0 = xs.cur_y
        leading = xs.style.leading

        from reportlab.pdfbase.pdfmetrics import getFont
        face = getFont(fn).face
        em = face.ascent + abs(face.descent)
        asc = face.ascent / em * fs
        desc = abs(face.descent) / em * fs

        gap = leading - asc - desc
        pad_x = stringWidth(' ', fn, fs) / 3
        bot = y0 - desc - gap / 3
        top = y0 + asc + gap / 3
        h = top - bot

        from reportlab.lib.colors import HexColor
        stroke_color = HexColor("#e0ddd8")

        c = tx._canvas
        for x1, x2, color in xs.backColors:
            c.saveState()
            c.setFillColor(color)
            c.setStrokeColor(stroke_color)
            c.setLineWidth(0.5)
            c.roundRect(x1 - pad_x, bot, (x2 - x1) + 2 * pad_x,
                        h, RADIUS, stroke=1, fill=1)
            c.restoreState()
        xs.backColors = []
        xs.backColor = None
    _orig(tx)


_mod._do_post_text = _patched
