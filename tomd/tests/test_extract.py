"""Tests for lib.pdf.extract."""

from unittest.mock import MagicMock
from lib.pdf.extract import extract_spatial, attach_links
from lib.pdf.types import Span, Line, Block


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


def _make_block_with_span(text, bbox):
    span = Span(text=text, bbox=bbox)
    line = Line(spans=[span])
    return Block(lines=[line]), span


class TestAttachLinks:
    def test_best_overlap_wins(self):
        """Link is assigned to the span with the greatest overlap area."""
        block_a, span_a = _make_block_with_span("hello", (10, 10, 100, 20))
        block_b, span_b = _make_block_with_span("world", (200, 10, 300, 20))
        link = {"uri": "https://example.com", "bbox": (5, 8, 110, 22)}
        attach_links([block_a, block_b], [link])
        assert span_a.link_url == "https://example.com"
        assert span_b.link_url is None

    def test_multiple_links_each_win_their_best_span(self):
        """Each link independently assigns to its own best-overlap span."""
        block_a, span_a = _make_block_with_span("hello", (10, 10, 100, 20))
        block_b, span_b = _make_block_with_span("world", (200, 10, 300, 20))
        link_a = {"uri": "https://a.com", "bbox": (5, 8, 110, 22)}
        link_b = {"uri": "https://b.com", "bbox": (195, 8, 305, 22)}
        attach_links([block_a, block_b], [link_a, link_b])
        assert span_a.link_url == "https://a.com"
        assert span_b.link_url == "https://b.com"

    def test_later_link_overwrites_if_better_overlap(self):
        """A later link with larger overlap replaces an earlier assignment."""
        block, span = _make_block_with_span("text", (10, 10, 100, 20))
        small_link = {"uri": "https://small.com", "bbox": (10, 10, 50, 20)}
        big_link = {"uri": "https://big.com", "bbox": (5, 8, 110, 22)}
        attach_links([block], [small_link, big_link])
        assert span.link_url == "https://big.com"

    def test_no_overlap_no_assignment(self):
        """A link entirely outside all spans leaves link_url unchanged."""
        block, span = _make_block_with_span("text", (10, 10, 100, 20))
        link = {"uri": "https://far.com", "bbox": (500, 500, 600, 510)}
        attach_links([block], [link])
        assert span.link_url is None
