"""Tests for lib.pdf.wording."""

from conftest import make_span, make_line, make_block
from lib.pdf.wording import (
    classify_wording, _color_int_to_rgb, _rgb_to_hue_sat,
    _is_chromatic, _classify_hue, _match_underline, _match_strikethrough,
)


def _color_to_int(r, g, b):
    """Convert 0-255 RGB to MuPDF integer color."""
    return (r << 16) | (g << 8) | b


class TestColorConversion:
    def test_black(self):
        assert _color_int_to_rgb(0) == (0.0, 0.0, 0.0)

    def test_green(self):
        r, g, b = _color_int_to_rgb(_color_to_int(0, 133, 71))
        assert g > r and g > b

    def test_red(self):
        r, g, b = _color_int_to_rgb(_color_to_int(204, 0, 0))
        assert r > g and r > b


class TestHSVClassification:
    def test_green_hue(self):
        h, s = _rgb_to_hue_sat(0.0, 0.52, 0.28)
        assert _is_chromatic(s)
        assert _classify_hue(h) == "green"

    def test_red_hue(self):
        h, s = _rgb_to_hue_sat(0.8, 0.0, 0.0)
        assert _is_chromatic(s)
        assert _classify_hue(h) == "red"

    def test_blue_hue(self):
        h, s = _rgb_to_hue_sat(0.02, 0.34, 0.76)
        assert _is_chromatic(s)
        assert _classify_hue(h) == "blue"

    def test_black_not_chromatic(self):
        _, s = _rgb_to_hue_sat(0.0, 0.0, 0.0)
        assert not _is_chromatic(s)

    def test_gray_not_chromatic(self):
        _, s = _rgb_to_hue_sat(0.5, 0.5, 0.5)
        assert not _is_chromatic(s)

    def test_teal_is_green(self):
        h, s = _rgb_to_hue_sat(0.0, 0.6, 0.6)
        assert _is_chromatic(s)
        assert _classify_hue(h) == "green"


class TestDrawingMatch:
    def test_underline_match(self):
        span_bbox = (100, 50, 200, 60)
        drawings = [(60.5, 100, 200, (0, 0.5, 0))]
        assert _match_underline(span_bbox, drawings, 1.5)

    def test_underline_no_match_too_far(self):
        span_bbox = (100, 50, 200, 60)
        drawings = [(65, 100, 200, (0, 0.5, 0))]
        assert not _match_underline(span_bbox, drawings, 1.5)

    def test_strikethrough_match(self):
        span_bbox = (100, 50, 200, 60)
        drawings = [(55, 100, 200, (0.8, 0, 0))]
        assert _match_strikethrough(span_bbox, drawings, 2.0)

    def test_strikethrough_no_match_too_far(self):
        span_bbox = (100, 50, 200, 60)
        drawings = [(48, 100, 200, (0.8, 0, 0))]
        assert not _match_strikethrough(span_bbox, drawings, 2.0)

    def test_empty_drawings(self):
        span_bbox = (100, 50, 200, 60)
        assert not _match_underline(span_bbox, [], 1.5)
        assert not _match_strikethrough(span_bbox, [], 2.0)


class TestClassifyWording:
    def test_green_span_classified_as_ins(self):
        green = _color_to_int(0, 133, 71)
        from lib.pdf.types import Span, Line, Block
        spans = [Span(text=f"added {i}", color=green,
                       bbox=(10, 50, 200, 60)) for i in range(6)]
        lines = [Line(spans=[s]) for s in spans]
        block = Block(lines=lines, page_num=0)
        drawings = {0: [(60.5, 0, 300, (0, 0.5, 0.3))]}
        problems = classify_wording([block], drawings)
        assert block.lines[0].spans[0].wording_role == "ins"

    def test_red_span_classified_as_del(self):
        red = _color_to_int(204, 0, 0)
        block = make_block(["removed text"] * 5, page_num=0)
        for ln in block.lines:
            ln.spans[0].color = red
        problems = classify_wording([block], {})
        assert block.lines[0].spans[0].wording_role == "del"

    def test_below_threshold_no_classification(self):
        green = _color_to_int(0, 133, 71)
        block = make_block(["tiny"], page_num=0)
        block.lines[0].spans[0].color = green
        problems = classify_wording([block], {})
        assert block.lines[0].spans[0].wording_role is None

    def test_black_span_not_classified(self):
        block = make_block(["normal text"] * 10, page_num=0)
        problems = classify_wording([block], {})
        assert all(s.wording_role is None
                   for ln in block.lines for s in ln.spans)

    def test_blue_span_not_classified(self):
        blue = _color_to_int(5, 85, 193)
        block = make_block(["link text"] * 10, page_num=0)
        for ln in block.lines:
            ln.spans[0].color = blue
        problems = classify_wording([block], {})
        assert all(s.wording_role is None
                   for ln in block.lines for s in ln.spans)

    def test_medium_confidence_generates_problem(self):
        green = _color_to_int(0, 133, 71)
        blocks = [make_block(["added"] * 10, page_num=0)]
        for ln in blocks[0].lines:
            ln.spans[0].color = green
        problems = classify_wording(blocks, {})
        assert any("MEDIUM" in p or "medium" in p.lower() for p in problems)

    def test_high_confidence_no_problem(self):
        green = _color_to_int(0, 133, 71)
        from lib.pdf.types import Span, Line, Block
        spans = [Span(text=f"added {i}", color=green,
                       bbox=(10, 50, 200, 60)) for i in range(10)]
        lines = [Line(spans=[s]) for s in spans]
        blocks = [Block(lines=lines, page_num=0)]
        drawings = {0: [(60.5, 0, 500, (0, 0.5, 0.3))]}
        problems = classify_wording(blocks, drawings)
        assert len(problems) == 0
