"""Native ReportLab Mermaid diagram renderer.

Uses merm's parser and Sugiyama layout engine, then draws nodes, edges,
arrowheads, edge labels, and subgraphs using ReportLab drawing primitives
with the document's own registered fonts.
"""

import logging
import math

from reportlab.graphics.shapes import (
    Circle,
    Drawing,
    Group,
    Line,
    PolyLine,
    Polygon,
    Rect,
    String,
)
from reportlab.lib.colors import Color, HexColor
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import Spacer

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _parse_color(value):
    if isinstance(value, str) and value.startswith("#"):
        return HexColor(value)
    if isinstance(value, Color):
        return value
    return HexColor("#333333")


def _lighten(color, amount=0.85):
    """Mix *color* toward white by *amount* (0 = unchanged, 1 = white)."""
    r = color.red + (1 - color.red) * amount
    g = color.green + (1 - color.green) * amount
    b = color.blue + (1 - color.blue) * amount
    return Color(r, g, b)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def draw_mermaid(code, content_width, page_h, style,
                 body_font="Body", gap_sm=6):
    """Render Mermaid *code* to a list of ReportLab flowables.

    Returns ``None`` on parse or layout failure so the caller can fall back
    to rendering the code as a plain code block.
    """
    try:
        font_size = style.get("body_size", 10)
        diagram, layout = _parse_and_layout(code, body_font, font_size)
        drawing = _build_drawing(diagram, layout, content_width, page_h,
                                 style, body_font, font_size)
        if drawing is None:
            return None
        return [Spacer(1, gap_sm), drawing, Spacer(1, gap_sm * 2)]
    except Exception as e:
        log.warning("mermaid native render failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Parse + layout
# ---------------------------------------------------------------------------

def _parse_and_layout(code, body_font, font_size):
    from merm import parse_flowchart
    from merm.layout import layout_diagram
    from merm.layout.config import LayoutConfig
    import merm.layout.sugiyama as _sg

    saved = (_sg._DEFAULT_FONT_SIZE, _sg._NODE_PADDING_H,
             _sg._NODE_PADDING_V, _sg._NODE_MIN_HEIGHT, _sg._NODE_MIN_WIDTH,
             _sg._line_width, _sg._wrap_line)

    _sg._DEFAULT_FONT_SIZE = font_size
    _sg._NODE_PADDING_H = font_size * 2.0
    _sg._NODE_PADDING_V = font_size * 1.0
    _sg._NODE_MIN_HEIGHT = font_size * 2.2
    _sg._NODE_MIN_WIDTH = font_size * 4.0

    def _rl_line_width(text, fs):
        return stringWidth(text, body_font, fs)

    def _rl_wrap_line(text, fs, max_width):
        if _rl_line_width(text, fs) <= max_width:
            return [text]
        words = text.split()
        if not words:
            return [text]
        lines, cur = [], words[0]
        for w in words[1:]:
            trial = cur + " " + w
            if _rl_line_width(trial, fs) <= max_width:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
        return lines

    _sg._line_width = _rl_line_width
    _sg._wrap_line = _rl_wrap_line

    try:
        diagram = parse_flowchart(code)
        layout_cfg = LayoutConfig(
            rank_sep=font_size * 3.0,
            node_sep=font_size * 1.2,
            direction=diagram.direction,
        )

        def measure_fn(text, fs):
            """Return raw text dimensions — merm adds padding separately."""
            lines = text.split("\\n") if "\\n" in text else text.split("\n")
            if not lines:
                lines = [""]
            max_w = max(stringWidth(ln.strip(), body_font, fs) for ln in lines)
            return (max_w, fs * 1.3 * len(lines))

        result = layout_diagram(diagram, measure_fn, layout_cfg)
    finally:
        (_sg._DEFAULT_FONT_SIZE, _sg._NODE_PADDING_H,
         _sg._NODE_PADDING_V, _sg._NODE_MIN_HEIGHT, _sg._NODE_MIN_WIDTH,
         _sg._line_width, _sg._wrap_line) = saved

    return diagram, result


# ---------------------------------------------------------------------------
# Drawing builder
# ---------------------------------------------------------------------------

def _build_drawing(diagram, layout, content_width, page_h, style,
                   body_font, font_size):
    if layout.width <= 0 or layout.height <= 0:
        return None

    max_h = page_h * style.get("mermaid_max_height_ratio", 0.8)
    scale = min(content_width / layout.width, 1.0)
    if layout.height * scale > max_h:
        scale = max_h / layout.height

    dw = layout.width * scale
    dh = layout.height * scale
    drawing = Drawing(dw, dh)
    drawing.hAlign = "CENTER"

    canvas_h = layout.height

    stroke_color = _parse_color(style.get("heading_rule_color", "#333333"))
    text_color = _parse_color(style.get("text_color",
                              style.get("body_fg", "#333333")))
    accent = _parse_color(style.get("accent_color", "#9370DB"))
    node_fill = _lighten(accent, 0.85)
    node_stroke = stroke_color
    sg_fill = _lighten(HexColor("#AAAA33"), 0.92)
    sg_stroke = HexColor("#AAAA33")

    fs = font_size * scale

    node_map = {n.id: n for n in diagram.nodes}

    # --- subgraphs (drawn first, behind everything else) ---
    if layout.subgraphs:
        _draw_subgraphs(drawing, layout.subgraphs, canvas_h, scale,
                        sg_fill, sg_stroke, body_font, fs, text_color)

    # --- edges (behind nodes) ---
    edge_ir_map = {}
    for edge in diagram.edges:
        edge_ir_map.setdefault((edge.source, edge.target), edge)

    for el in layout.edges:
        ir_edge = edge_ir_map.get((el.source, el.target))
        if ir_edge is None:
            continue
        _draw_edge(drawing, el, ir_edge, canvas_h, scale,
                   stroke_color, body_font, fs * 0.85, text_color)

    # --- nodes (on top) ---
    for node_id, nl in layout.nodes.items():
        ir_node = node_map.get(node_id)
        if ir_node is None:
            continue
        _draw_node(drawing, ir_node, nl, canvas_h, scale,
                   node_fill, node_stroke, body_font, fs, text_color)

    return drawing


# ---------------------------------------------------------------------------
# Coordinate flip helper
# ---------------------------------------------------------------------------

def _flip_y(merm_y, canvas_h):
    """Convert merm top-left y to ReportLab bottom-left y."""
    return canvas_h - merm_y


# ---------------------------------------------------------------------------
# Subgraph drawing
# ---------------------------------------------------------------------------

def _draw_subgraphs(drawing, subgraphs, canvas_h, scale, fill, stroke,
                    font, fs, text_color):
    for sg_id, sg in subgraphs.items():
        x = sg.x * scale
        w = sg.width * scale
        h = sg.height * scale
        y = _flip_y(sg.y + sg.height, canvas_h) * scale

        drawing.add(Rect(x, y, w, h,
                         fillColor=fill,
                         strokeColor=stroke,
                         strokeWidth=1 * scale,
                         strokeDashArray=[4 * scale, 3 * scale],
                         fillOpacity=0.5))

        if sg.title:
            drawing.add(String(x + 4 * scale, y + h - fs - 2 * scale,
                               sg.title,
                               fontName=font + "-Bold" if _font_exists(font + "-Bold") else font,
                               fontSize=fs * 0.9,
                               fillColor=text_color))


def _font_exists(name):
    try:
        from reportlab.pdfbase.pdfmetrics import getFont
        getFont(name)
        return True
    except KeyError:
        return False


# ---------------------------------------------------------------------------
# Node drawing - all 14 shapes
# ---------------------------------------------------------------------------

def _draw_node(drawing, ir_node, nl, canvas_h, scale,
               fill, stroke, font, fs, text_color):
    x = nl.x * scale
    w = nl.width * scale
    h = nl.height * scale
    y = _flip_y(nl.y + nl.height, canvas_h) * scale
    cx = x + w / 2
    cy = y + h / 2
    sw = max(1, 1 * scale)

    shape = ir_node.shape.value if hasattr(ir_node.shape, 'value') else str(ir_node.shape)
    _SHAPE_DISPATCH.get(shape, _shape_rect)(
        drawing, x, y, w, h, cx, cy, sw, scale, fill, stroke)

    _draw_node_label(drawing, ir_node.label, cx, cy, font, fs, text_color)


def _shape_rect(drawing, x, y, w, h, cx, cy, sw, scale, fill, stroke):
    drawing.add(Rect(x, y, w, h,
                     fillColor=fill, strokeColor=stroke, strokeWidth=sw))


def _shape_rounded(drawing, x, y, w, h, cx, cy, sw, scale, fill, stroke):
    r = min(w, h) * 0.15
    drawing.add(Rect(x, y, w, h, rx=r, ry=r,
                     fillColor=fill, strokeColor=stroke, strokeWidth=sw))


def _shape_stadium(drawing, x, y, w, h, cx, cy, sw, scale, fill, stroke):
    r = h / 2
    drawing.add(Rect(x, y, w, h, rx=r, ry=r,
                     fillColor=fill, strokeColor=stroke, strokeWidth=sw))


def _shape_subroutine(drawing, x, y, w, h, cx, cy, sw, scale, fill, stroke):
    drawing.add(Rect(x, y, w, h,
                     fillColor=fill, strokeColor=stroke, strokeWidth=sw))
    inset = min(w * 0.08, 8 * scale)
    drawing.add(Line(x + inset, y, x + inset, y + h,
                     strokeColor=stroke, strokeWidth=sw))
    drawing.add(Line(x + w - inset, y, x + w - inset, y + h,
                     strokeColor=stroke, strokeWidth=sw))


def _shape_diamond(drawing, x, y, w, h, cx, cy, sw, scale, fill, stroke):
    pts = [cx, y + h, x + w, cy, cx, y, x, cy]
    drawing.add(Polygon(pts, fillColor=fill, strokeColor=stroke, strokeWidth=sw))


def _shape_circle(drawing, x, y, w, h, cx, cy, sw, scale, fill, stroke):
    r = min(w, h) / 2
    drawing.add(Circle(cx, cy, r,
                       fillColor=fill, strokeColor=stroke, strokeWidth=sw))


def _shape_double_circle(drawing, x, y, w, h, cx, cy, sw, scale, fill, stroke):
    r = min(w, h) / 2
    drawing.add(Circle(cx, cy, r,
                       fillColor=fill, strokeColor=stroke, strokeWidth=sw))
    drawing.add(Circle(cx, cy, r * 0.85,
                       fillColor=fill, strokeColor=stroke, strokeWidth=sw))


def _shape_hexagon(drawing, x, y, w, h, cx, cy, sw, scale, fill, stroke):
    inset = min(w * 0.15, 12 * scale)
    pts = [x + inset, y,
           x + w - inset, y,
           x + w, cy,
           x + w - inset, y + h,
           x + inset, y + h,
           x, cy]
    drawing.add(Polygon(pts, fillColor=fill, strokeColor=stroke, strokeWidth=sw))


def _shape_parallelogram(drawing, x, y, w, h, cx, cy, sw, scale, fill, stroke):
    skew = min(w * 0.15, 10 * scale)
    pts = [x + skew, y, x + w, y, x + w - skew, y + h, x, y + h]
    drawing.add(Polygon(pts, fillColor=fill, strokeColor=stroke, strokeWidth=sw))


def _shape_parallelogram_alt(drawing, x, y, w, h, cx, cy, sw, scale, fill, stroke):
    skew = min(w * 0.15, 10 * scale)
    pts = [x, y, x + w - skew, y, x + w, y + h, x + skew, y + h]
    drawing.add(Polygon(pts, fillColor=fill, strokeColor=stroke, strokeWidth=sw))


def _shape_trapezoid(drawing, x, y, w, h, cx, cy, sw, scale, fill, stroke):
    inset = min(w * 0.15, 10 * scale)
    pts = [x + inset, y, x + w - inset, y, x + w, y + h, x, y + h]
    drawing.add(Polygon(pts, fillColor=fill, strokeColor=stroke, strokeWidth=sw))


def _shape_trapezoid_alt(drawing, x, y, w, h, cx, cy, sw, scale, fill, stroke):
    inset = min(w * 0.15, 10 * scale)
    pts = [x, y, x + w, y, x + w - inset, y + h, x + inset, y + h]
    drawing.add(Polygon(pts, fillColor=fill, strokeColor=stroke, strokeWidth=sw))


def _shape_cylinder(drawing, x, y, w, h, cx, cy, sw, scale, fill, stroke):
    from reportlab.graphics.shapes import Ellipse
    ell_h = min(h * 0.15, 8 * scale)
    body_y = y + ell_h / 2
    body_h = h - ell_h
    drawing.add(Rect(x, body_y, w, body_h,
                     fillColor=fill, strokeColor=fill, strokeWidth=0))
    drawing.add(Ellipse(cx, y + ell_h / 2, w / 2, ell_h,
                        fillColor=fill, strokeColor=stroke, strokeWidth=sw))
    drawing.add(Ellipse(cx, y + h - ell_h / 2, w / 2, ell_h,
                        fillColor=fill, strokeColor=stroke, strokeWidth=sw))
    drawing.add(Line(x, y + ell_h / 2, x, y + h - ell_h / 2,
                     strokeColor=stroke, strokeWidth=sw))
    drawing.add(Line(x + w, y + ell_h / 2, x + w, y + h - ell_h / 2,
                     strokeColor=stroke, strokeWidth=sw))


def _shape_asymmetric(drawing, x, y, w, h, cx, cy, sw, scale, fill, stroke):
    notch = min(w * 0.15, 10 * scale)
    pts = [x, y, x + w, y, x + w - notch, cy, x + w, y + h, x, y + h]
    drawing.add(Polygon(pts, fillColor=fill, strokeColor=stroke, strokeWidth=sw))


_SHAPE_DISPATCH = {
    "rect": _shape_rect,
    "rounded": _shape_rounded,
    "stadium": _shape_stadium,
    "subroutine": _shape_subroutine,
    "diamond": _shape_diamond,
    "circle": _shape_circle,
    "double_circle": _shape_double_circle,
    "hexagon": _shape_hexagon,
    "parallelogram": _shape_parallelogram,
    "parallelogram_alt": _shape_parallelogram_alt,
    "trapezoid": _shape_trapezoid,
    "trapezoid_alt": _shape_trapezoid_alt,
    "cylinder": _shape_cylinder,
    "asymmetric": _shape_asymmetric,
}


# ---------------------------------------------------------------------------
# Node label (multi-line aware)
# ---------------------------------------------------------------------------

def _draw_node_label(drawing, label, cx, cy, font, fs, text_color):
    if not label:
        return
    lines = label.split("\\n") if "\\n" in label else label.split("\n")
    n = len(lines)
    line_h = fs * 1.3
    start_y = cy - fs * 0.25 + (n - 1) * line_h / 2

    for i, ln in enumerate(lines):
        drawing.add(String(cx, start_y - i * line_h, ln.strip(),
                           fontName=font, fontSize=fs,
                           fillColor=text_color,
                           textAnchor="middle"))


# ---------------------------------------------------------------------------
# Edge drawing
# ---------------------------------------------------------------------------

def _draw_edge(drawing, edge_layout, ir_edge, canvas_h, scale,
               stroke_color, font, fs, text_color):
    from merm.ir.enums import EdgeType, ArrowType

    et = ir_edge.edge_type
    if et == EdgeType.invisible:
        return

    points = edge_layout.points
    if len(points) < 2:
        return

    scaled = []
    for p in points:
        scaled.append(p.x * scale)
        scaled.append(_flip_y(p.y, canvas_h) * scale)

    sw = 1.5 * scale
    dash = None

    if et in (EdgeType.thick, EdgeType.thick_arrow):
        sw = 3 * scale
    if et in (EdgeType.dotted, EdgeType.dotted_arrow):
        dash = [4 * scale, 3 * scale]

    drawing.add(PolyLine(scaled, strokeColor=stroke_color,
                         strokeWidth=sw, strokeDashArray=dash))

    # Target arrow (at last point)
    has_target_arrow = et in (EdgeType.arrow, EdgeType.dotted_arrow,
                              EdgeType.thick_arrow)
    ta = ir_edge.target_arrow
    if has_target_arrow or (ta and ta != ArrowType.none):
        at = ta if ta and ta != ArrowType.none else ArrowType.arrow
        x2, y2 = scaled[-2], scaled[-1]
        x1, y1 = scaled[-4], scaled[-3]
        _draw_arrowhead(drawing, x1, y1, x2, y2, at, stroke_color,
                        6 * scale)

    # Source arrow (at first point)
    sa = ir_edge.source_arrow
    if sa and sa != ArrowType.none:
        x1, y1 = scaled[2], scaled[3]
        x2, y2 = scaled[0], scaled[1]
        _draw_arrowhead(drawing, x1, y1, x2, y2, sa, stroke_color,
                        6 * scale)

    # Edge label
    if ir_edge.label:
        mid_idx = len(points) // 2
        mp = points[mid_idx]
        lx = mp.x * scale
        ly = _flip_y(mp.y, canvas_h) * scale
        drawing.add(String(lx, ly + fs * 0.3, ir_edge.label,
                           fontName=font, fontSize=fs,
                           fillColor=text_color,
                           textAnchor="middle"))


# ---------------------------------------------------------------------------
# Arrowhead drawing
# ---------------------------------------------------------------------------

def _draw_arrowhead(drawing, x1, y1, x2, y2, arrow_type, color, size):
    """Draw an arrowhead at (x2, y2) pointing away from (x1, y1)."""
    from merm.ir.enums import ArrowType

    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length < 0.01:
        return

    ux, uy = dx / length, dy / length
    px, py = -uy, ux

    if arrow_type == ArrowType.arrow:
        tip_x, tip_y = x2, y2
        base_x, base_y = x2 - ux * size, y2 - uy * size
        pts = [tip_x, tip_y,
               base_x + px * size * 0.4, base_y + py * size * 0.4,
               base_x - px * size * 0.4, base_y - py * size * 0.4]
        drawing.add(Polygon(pts, fillColor=color, strokeColor=color,
                            strokeWidth=0))

    elif arrow_type == ArrowType.circle:
        r = size * 0.35
        drawing.add(Circle(x2 - ux * r, y2 - uy * r, r,
                           fillColor=color, strokeColor=color))

    elif arrow_type == ArrowType.cross:
        arm = size * 0.4
        cx, cy = x2 - ux * size * 0.3, y2 - uy * size * 0.3
        drawing.add(Line(cx - px * arm, cy - py * arm,
                         cx + px * arm, cy + py * arm,
                         strokeColor=color, strokeWidth=1.5 * (size / 6)))
        drawing.add(Line(cx - ux * arm, cy - uy * arm,
                         cx + ux * arm, cy + uy * arm,
                         strokeColor=color, strokeWidth=1.5 * (size / 6)))
