"""Unit tests for the native ReportLab Mermaid renderer."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.mermaid import draw_mermaid, _parse_and_layout


SIMPLE_STYLE = {
    "body_size": 10,
    "heading_rule_color": "#333333",
    "accent_color": "#9370DB",
    "mermaid_max_height_ratio": 0.8,
}


def test_parse_and_layout_simple(font_registered):
    code = "flowchart TD\n    A[Start] --> B[End]"
    diagram, layout = _parse_and_layout(code, "Body", 8.5)
    assert len(layout.nodes) == 2
    assert len(layout.edges) == 1
    assert layout.width > 0
    assert layout.height > 0


def test_draw_mermaid_returns_flowables(font_registered):
    code = "flowchart TD\n    A[Start] --> B[End]"
    result = draw_mermaid(code, 500, 700, SIMPLE_STYLE, body_font="Body")
    assert result is not None
    assert len(result) == 3  # Spacer, Drawing, Spacer


def test_draw_mermaid_with_diamond(font_registered):
    code = "flowchart TD\n    A[Start] --> B{Decision}\n    B --> C[End]"
    result = draw_mermaid(code, 500, 700, SIMPLE_STYLE, body_font="Body")
    assert result is not None


def test_draw_mermaid_with_edge_labels(font_registered):
    code = "flowchart TD\n    A[Start] -->|yes| B[End]\n    A -.->|no| C[Alt]"
    result = draw_mermaid(code, 500, 700, SIMPLE_STYLE, body_font="Body")
    assert result is not None


def test_draw_mermaid_with_subgraph(font_registered):
    code = (
        "flowchart TD\n"
        "    subgraph grp [My Group]\n"
        "        A[One] --> B[Two]\n"
        "    end\n"
        "    B --> C[Three]"
    )
    result = draw_mermaid(code, 500, 700, SIMPLE_STYLE, body_font="Body")
    assert result is not None


def test_draw_mermaid_multiline_label(font_registered):
    code = 'flowchart TD\n    A["Line One\\nLine Two"] --> B[End]'
    result = draw_mermaid(code, 500, 700, SIMPLE_STYLE, body_font="Body")
    assert result is not None


def test_draw_mermaid_lr_direction(font_registered):
    code = "flowchart LR\n    A[Start] --> B[End]"
    result = draw_mermaid(code, 500, 700, SIMPLE_STYLE, body_font="Body")
    assert result is not None


def test_draw_mermaid_returns_none_on_bad_input(font_registered):
    result = draw_mermaid("not valid mermaid", 500, 700, SIMPLE_STYLE)
    assert result is None


def test_draw_mermaid_circle_shape(font_registered):
    code = "flowchart TD\n    A((Circle)) --> B[End]"
    result = draw_mermaid(code, 500, 700, SIMPLE_STYLE, body_font="Body")
    assert result is not None


def test_drawing_fits_content_width(font_registered):
    code = "flowchart TD\n    A[Start] --> B[End]"
    width = 400
    result = draw_mermaid(code, width, 700, SIMPLE_STYLE, body_font="Body")
    assert result is not None
    drawing = result[1]
    assert drawing.width <= width + 1  # allow rounding tolerance


@pytest.mark.parametrize("code,desc", [
    (
        'flowchart TD\n'
        '    A["parent resumes - slot = parent.frame_alloc"]\n'
        '    A --> B["parent calls child()"]\n'
        '    B --> C["child operator new - reads slot"]',
        "long labels (d4172-style)",
    ),
    (
        'flowchart TD\n    A["Line One\\nLine Two"] --> B[End]',
        "multi-line label",
    ),
    (
        'flowchart TD\n'
        '    A[Short] --> B{Decision Point}\n'
        '    B --> C((Circle))',
        "mixed shapes",
    ),
])
def test_text_fits_in_boxes(font_registered, code, desc):
    """Verify that every node label fits inside its laid-out box."""
    from reportlab.pdfbase.pdfmetrics import stringWidth

    font_size = SIMPLE_STYLE["body_size"]
    diagram, layout = _parse_and_layout(code, "Body", font_size)

    content_width = 460
    page_h = 700
    scale = content_width / layout.width
    max_h = page_h * SIMPLE_STYLE.get("mermaid_max_height_ratio", 0.8)
    if layout.height * scale > max_h:
        scale = max_h / layout.height

    fs = font_size * scale
    node_map = {n.id: n for n in diagram.nodes}

    for nid, nl in layout.nodes.items():
        ir = node_map.get(nid)
        if not ir or not ir.label:
            continue
        label = ir.label
        lines = label.split("\\n") if "\\n" in label else label.split("\n")

        box_w = nl.width * scale
        box_h = nl.height * scale
        text_w = max(stringWidth(ln.strip(), "Body", fs) for ln in lines)
        text_h = fs * 1.3 * len(lines)

        assert text_w < box_w, (
            f"[{desc}] node {nid}: text width {text_w:.1f} > box width {box_w:.1f}"
        )
        assert text_h < box_h, (
            f"[{desc}] node {nid}: text height {text_h:.1f} > box height {box_h:.1f}"
        )
