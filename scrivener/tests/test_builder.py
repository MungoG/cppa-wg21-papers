"""Layer 3: Builder integration tests - full build_pdf pipeline.

Each test loads a markdown fixture, runs build_pdf, and verifies
the output PDF exists and is structurally valid.
"""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.builder import build_pdf
from lib.colors import resolve_colors
from lib.config import load_style, resolve_style_path

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def default_style(font_registered):
    style = load_style(resolve_style_path(None))
    resolve_colors(style, None)
    return style


@pytest.fixture
def wg21_style(font_registered):
    style = load_style(resolve_style_path("cpp-al"))
    resolve_colors(style, None)
    return style


def _assert_valid_pdf(path):
    assert path.exists(), f"PDF not created: {path}"
    assert path.stat().st_size > 0, f"PDF is empty: {path}"
    header = path.read_bytes()[:5]
    assert header == b"%PDF-", f"Not a valid PDF: {path}"


def _read_pdf_title(path):
    """Extract the /Title string from a PDF's Info dictionary."""
    import re
    raw = path.read_bytes()
    m = re.search(rb'/Title\s*\(([^)]*)\)', raw)
    if m:
        return m.group(1).decode("latin-1")
    m = re.search(rb'/Title\s*<([0-9A-Fa-f]+)>', raw)
    if m:
        return bytes.fromhex(m.group(1).decode("ascii")).decode("utf-16-be")
    return None


def test_build_minimal(default_style, tmp_path):
    md = FIXTURES / "minimal.md"
    out = tmp_path / "minimal.pdf"
    result = build_pdf(md, out, {}, copy.deepcopy(default_style))
    _assert_valid_pdf(result)


def test_build_front_matter(wg21_style, tmp_path):
    md = FIXTURES / "front-matter.md"
    out = tmp_path / "front-matter.pdf"
    result = build_pdf(md, out, {}, copy.deepcopy(wg21_style))
    _assert_valid_pdf(result)


def test_build_headings(default_style, tmp_path):
    md = FIXTURES / "headings.md"
    out = tmp_path / "headings.pdf"
    result = build_pdf(md, out, {}, copy.deepcopy(default_style))
    _assert_valid_pdf(result)


def test_build_code(default_style, tmp_path):
    md = FIXTURES / "code.md"
    out = tmp_path / "code.pdf"
    result = build_pdf(md, out, {}, copy.deepcopy(default_style))
    _assert_valid_pdf(result)


def test_build_wording(default_style, tmp_path):
    md = FIXTURES / "wording.md"
    out = tmp_path / "wording.pdf"
    result = build_pdf(md, out, {}, copy.deepcopy(default_style))
    _assert_valid_pdf(result)


def test_build_table(default_style, tmp_path):
    md = FIXTURES / "table.md"
    out = tmp_path / "table.pdf"
    result = build_pdf(md, out, {}, copy.deepcopy(default_style))
    _assert_valid_pdf(result)


def test_build_unknown_page_size(default_style, tmp_path):
    md = FIXTURES / "minimal.md"
    out = tmp_path / "bogus.pdf"
    style = copy.deepcopy(default_style)
    style["page_size"] = "bogus"
    with pytest.raises(ValueError, match="unknown page_size"):
        build_pdf(md, out, {}, style)


def test_build_output_dir_created(default_style, tmp_path):
    md = FIXTURES / "minimal.md"
    out = tmp_path / "sub" / "dir" / "output.pdf"
    result = build_pdf(md, out, {}, copy.deepcopy(default_style))
    _assert_valid_pdf(result)


def test_build_with_toc(default_style, tmp_path):
    md = FIXTURES / "headings.md"
    out = tmp_path / "headings-toc.pdf"
    result = build_pdf(md, out, {"toc": True}, copy.deepcopy(default_style))
    _assert_valid_pdf(result)


def test_build_no_toc(default_style, tmp_path):
    md = FIXTURES / "headings.md"
    out = tmp_path / "headings-notoc.pdf"
    result = build_pdf(md, out, {"no_toc": True}, copy.deepcopy(default_style))
    _assert_valid_pdf(result)


def test_build_intent_info(wg21_style, tmp_path):
    md = FIXTURES / "intent-info.md"
    out = tmp_path / "intent-info.pdf"
    result = build_pdf(md, out, {}, copy.deepcopy(wg21_style))
    _assert_valid_pdf(result)
    title = _read_pdf_title(result)
    assert title is not None, "Could not read PDF title metadata"
    assert title.startswith("Info: "), f"Expected 'Info: ' prefix, got: {title!r}"
    assert "Symmetric Transfer" in title


def test_build_intent_ask(wg21_style, tmp_path):
    md = FIXTURES / "intent-ask.md"
    out = tmp_path / "intent-ask.pdf"
    result = build_pdf(md, out, {}, copy.deepcopy(wg21_style))
    _assert_valid_pdf(result)
    title = _read_pdf_title(result)
    assert title is not None, "Could not read PDF title metadata"
    assert title.startswith("Ask: "), f"Expected 'Ask: ' prefix, got: {title!r}"
    assert "Coroutine Execution Model" in title


def test_build_no_intent_no_prefix(wg21_style, tmp_path):
    md = FIXTURES / "front-matter.md"
    out = tmp_path / "no-intent.pdf"
    result = build_pdf(md, out, {}, copy.deepcopy(wg21_style))
    _assert_valid_pdf(result)
    title = _read_pdf_title(result)
    assert title is not None, "Could not read PDF title metadata"
    assert title == "Test Paper", f"Expected bare title, got: {title!r}"


def test_build_mermaid(default_style, tmp_path):
    md = FIXTURES / "mermaid.md"
    out = tmp_path / "mermaid.pdf"
    result = build_pdf(md, out, {}, copy.deepcopy(default_style))
    _assert_valid_pdf(result)
