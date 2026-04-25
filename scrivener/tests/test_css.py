"""Layer 1: Pure-function tests for CSS generation.

No fonts, no ReportLab. Tests verify generate_css() output
and the su() helper arithmetic.
"""

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib.config import load_style, resolve_style_path
from lib.css import generate_css, su, _su_abs


@pytest.fixture
def style():
    path = resolve_style_path(None)
    return copy.deepcopy(load_style(path))


class TestSu:
    def test_su_one(self):
        assert su(1) == "var(--u)"

    def test_su_ratio(self):
        assert su(0.8) == "calc(var(--u) * 0.8)"

    def test_su_large(self):
        assert su(1.6) == "calc(var(--u) * 1.6)"

    def test_su_small(self):
        assert su(0.3) == "calc(var(--u) * 0.3)"

    def test_su_abs(self):
        cfg = {"body_size": 10}
        assert _su_abs(cfg, 15) == "calc(var(--u) * 1.5)"

    def test_su_abs_bar_width(self):
        cfg = {"body_size": 10}
        assert _su_abs(cfg, 3) == "calc(var(--u) * 0.3)"


class TestGenerateCss:
    def test_returns_string(self, style):
        css = generate_css(style)
        assert isinstance(css, str)
        assert len(css) > 100

    def test_fragment_mode_has_px_unit(self, style):
        css = generate_css(style, mode="fragment")
        assert "--u:" in css
        assert "px" in css

    def test_full_mode_has_html_rule(self, style):
        css = generate_css(style, mode="full")
        assert "html {" in css
        assert "font-size:" in css
        assert "%" in css.split("html {")[1].split("}")[0]

    def test_full_mode_unit_is_em(self, style):
        css = generate_css(style, mode="full")
        article_block = css.split("article.scrivener {")[1].split("}")[0]
        assert "--u: 1em" in article_block

    def test_contains_color_vars(self, style):
        from lib.colors import resolve_colors
        resolve_colors(style, None)
        css = generate_css(style)
        assert "--accent:" in css
        assert "--link-color:" in css
        assert "--code-fg:" in css
        assert "--code-bg:" in css

    def test_contains_heading_rules(self, style):
        css = generate_css(style)
        for level in range(1, 7):
            assert f"article.scrivener h{level}" in css

    def test_contains_code_block_rule(self, style):
        css = generate_css(style)
        assert "pre.code-block" in css

    def test_contains_blockquote_rule(self, style):
        css = generate_css(style)
        assert "blockquote" in css

    def test_contains_table_rules(self, style):
        css = generate_css(style)
        assert "article.scrivener table" in css
        assert "article.scrivener th" in css
        assert "article.scrivener td" in css

    def test_contains_syntax_classes(self, style):
        css = generate_css(style)
        assert ".hl-k" in css
        assert ".hl-s" in css
        assert ".hl-c" in css

    def test_contains_wording_rules(self, style):
        css = generate_css(style)
        assert ".wording-add" in css
        assert ".wording-remove" in css

    def test_contains_front_matter_rule(self, style):
        css = generate_css(style)
        assert "dl.front-matter" in css

    def test_flexbox_layout(self, style):
        css = generate_css(style)
        assert "display: flex" in css
        assert "flex-direction: column" in css

    def test_max_width_set(self, style):
        css = generate_css(style)
        assert "max-width:" in css

    def test_page_break_rule(self, style):
        css = generate_css(style)
        assert "hr.page-break" in css

    def test_heading_scale_arithmetic(self, style):
        """Verify h1 font-size uses the scale from the style."""
        h1_scale = style.get("headings", {}).get("h1", {}).get("scale", 1.6)
        css = generate_css(style)
        assert f"calc(var(--u) * {h1_scale})" in css

    def test_body_space_after(self, style):
        css = generate_css(style)
        assert "calc(var(--u) * 0.8)" in css
