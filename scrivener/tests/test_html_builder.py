"""Layer 3: HTML builder integration tests.

Full build_html pipeline against markdown fixtures.
No fonts or ReportLab needed.
"""

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib.colors import resolve_colors
from lib.config import load_style, resolve_style_path
from lib.html_builder import build_css, build_html

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def style():
    s = load_style(resolve_style_path(None))
    resolve_colors(s, None)
    return s


def _assert_valid_html(path):
    assert path.exists(), f"HTML not created: {path}"
    assert path.stat().st_size > 0, f"HTML is empty: {path}"
    text = path.read_text(encoding="utf-8")
    assert "article" in text.lower() or "<!doctype" in text.lower()


class TestBuildHtmlFullMode:
    def test_minimal(self, style, tmp_path):
        md = FIXTURES / "minimal.md"
        out = tmp_path / "minimal.html"
        result = build_html(md, out, {}, copy.deepcopy(style), mode="full")
        _assert_valid_html(result)
        text = result.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in text
        assert "<html" in text
        assert "article" in text

    def test_css_file_written(self, style, tmp_path):
        md = FIXTURES / "minimal.md"
        out = tmp_path / "minimal.html"
        build_html(md, out, {}, copy.deepcopy(style), mode="full")
        css_files = list(tmp_path.glob("*.css"))
        assert len(css_files) == 1

    def test_stylesheet_link(self, style, tmp_path):
        md = FIXTURES / "minimal.md"
        out = tmp_path / "minimal.html"
        build_html(md, out, {}, copy.deepcopy(style), mode="full")
        text = out.read_text(encoding="utf-8")
        assert '<link rel="stylesheet"' in text

    def test_front_matter(self, style, tmp_path):
        md = FIXTURES / "front-matter.md"
        out = tmp_path / "front-matter.html"
        result = build_html(md, out, {}, copy.deepcopy(style), mode="full")
        _assert_valid_html(result)
        text = result.read_text(encoding="utf-8")
        assert "front-matter" in text

    def test_headings(self, style, tmp_path):
        md = FIXTURES / "headings.md"
        out = tmp_path / "headings.html"
        result = build_html(md, out, {}, copy.deepcopy(style), mode="full")
        _assert_valid_html(result)
        text = result.read_text(encoding="utf-8")
        assert "<h2" in text

    def test_code(self, style, tmp_path):
        md = FIXTURES / "code.md"
        out = tmp_path / "code.html"
        result = build_html(md, out, {}, copy.deepcopy(style), mode="full")
        _assert_valid_html(result)
        text = result.read_text(encoding="utf-8")
        assert "code-block" in text

    def test_table(self, style, tmp_path):
        md = FIXTURES / "table.md"
        out = tmp_path / "table.html"
        result = build_html(md, out, {}, copy.deepcopy(style), mode="full")
        _assert_valid_html(result)
        text = result.read_text(encoding="utf-8")
        assert "<table>" in text

    def test_wording(self, style, tmp_path):
        md = FIXTURES / "wording.md"
        out = tmp_path / "wording.html"
        result = build_html(md, out, {}, copy.deepcopy(style), mode="full")
        _assert_valid_html(result)

    def test_with_toc(self, style, tmp_path):
        md = FIXTURES / "headings.md"
        out = tmp_path / "headings-toc.html"
        result = build_html(md, out, {"toc": True}, copy.deepcopy(style),
                            mode="full")
        _assert_valid_html(result)
        text = result.read_text(encoding="utf-8")
        assert "toc" in text

    def test_output_dir_created(self, style, tmp_path):
        md = FIXTURES / "minimal.md"
        out = tmp_path / "sub" / "dir" / "output.html"
        result = build_html(md, out, {}, copy.deepcopy(style), mode="full")
        _assert_valid_html(result)


class TestBuildHtmlFragmentMode:
    def test_fragment_no_doctype(self, style, tmp_path):
        md = FIXTURES / "minimal.md"
        out = tmp_path / "minimal.html"
        build_html(md, out, {}, copy.deepcopy(style), mode="fragment")
        text = out.read_text(encoding="utf-8")
        assert "<!DOCTYPE" not in text
        assert "<html" not in text
        assert "<article" in text

    def test_fragment_css_file(self, style, tmp_path):
        md = FIXTURES / "minimal.md"
        out = tmp_path / "minimal.html"
        build_html(md, out, {}, copy.deepcopy(style), mode="fragment")
        css_files = list(tmp_path.glob("*.css"))
        assert len(css_files) == 1

    def test_fragment_px_unit(self, style, tmp_path):
        md = FIXTURES / "minimal.md"
        out = tmp_path / "minimal.html"
        build_html(md, out, {}, copy.deepcopy(style), mode="fragment")
        css = list(tmp_path.glob("*.css"))[0].read_text(encoding="utf-8")
        assert "px" in css
        assert "html {" not in css


class TestInlineCss:
    def test_full_inline(self, style, tmp_path):
        md = FIXTURES / "minimal.md"
        out = tmp_path / "minimal.html"
        build_html(md, out, {}, copy.deepcopy(style),
                   mode="full", inline_css=True)
        text = out.read_text(encoding="utf-8")
        assert "<style>" in text
        assert '<link rel="stylesheet"' not in text
        css_files = list(tmp_path.glob("*.css"))
        assert len(css_files) == 0

    def test_fragment_inline(self, style, tmp_path):
        md = FIXTURES / "minimal.md"
        out = tmp_path / "minimal.html"
        build_html(md, out, {}, copy.deepcopy(style),
                   mode="fragment", inline_css=True)
        text = out.read_text(encoding="utf-8")
        assert "<style>" in text
        assert "<article" in text
        css_files = list(tmp_path.glob("*.css"))
        assert len(css_files) == 0


class TestBuildCss:
    def test_build_css_default(self, style, tmp_path):
        out = tmp_path / "test.css"
        result = build_css(copy.deepcopy(style), output_path=out)
        assert result.exists()
        text = result.read_text(encoding="utf-8")
        assert "article.scrivener" in text

    def test_build_css_default_path(self, style):
        result = build_css(copy.deepcopy(style))
        assert result.exists()
        assert result.suffix == ".css"
        text = result.read_text(encoding="utf-8")
        assert "--accent:" in text
