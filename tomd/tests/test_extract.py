"""Tests for lib.pdf.extract."""

from unittest.mock import MagicMock
from lib.pdf.extract import extract_spatial


def _make_page(chars_by_span):
    """Build a mock page whose rawdict returns chars grouped by span.

    chars_by_span: list of (font, size, [(char, x0, y0, x1, y1), ...])
    """
    spans = []
    for font, size, char_list in chars_by_span:
        chars = []
        for c, x0, y0, x1, y1 in char_list:
            chars.append({
                "c": c,
                "bbox": (x0, y0, x1, y1),
                "origin": (x0, y0),
            })
        spans.append({
            "font": font,
            "size": size,
            "flags": 0,
            "color": 0,
            "chars": chars,
        })
    page = MagicMock()
    page.get_text.return_value = {
        "blocks": [{
            "type": 0,
            "lines": [{"spans": spans}],
        }],
    }
    return page


def test_spatial_sorts_by_x_within_same_y():
    """Chars at the same y but reversed x-order should come out left-to-right."""
    page = _make_page([
        ("Font", 12.0, [
            ("B", 100.0, 10.0, 112.0, 22.0),
            ("A", 10.0, 10.0, 22.0, 22.0),
        ]),
    ])
    blocks = extract_spatial(page, 0)
    full_text = " ".join(ln.text for b in blocks for ln in b.lines)
    assert full_text.index("A") < full_text.index("B")
