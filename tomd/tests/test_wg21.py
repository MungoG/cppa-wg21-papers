"""Tests for lib.pdf.wg21."""

from conftest import make_block, make_span, make_line
from lib.pdf.types import Block, Line, Span
from lib.pdf.wg21 import extract_metadata_from_blocks, _parse_authors, _lookup_lightness


def _meta_block(lines_text, page_num=0, font_size=9.0):
    lines = []
    for text in lines_text:
        span = Span(text=text, font_size=font_size)
        lines.append(Line(spans=[span], page_num=page_num))
    return Block(lines=lines, page_num=page_num)


def test_extracts_doc_number():
    b = _meta_block(["Document Number: P4003R0", "Date: 2026-01-01"])
    meta, consumed = extract_metadata_from_blocks([b])
    assert meta.get("document") == "P4003R0"


def test_extracts_date():
    b = _meta_block(["Document Number: P1234R0", "Date: 2026-03-15"])
    meta, consumed = extract_metadata_from_blocks([b])
    assert meta.get("date") == "2026-03-15"


def test_extracts_audience():
    b = _meta_block(["Document Number: P1234R0", "Audience: LEWG"])
    meta, consumed = extract_metadata_from_blocks([b])
    assert meta.get("audience") == "LEWG"


def test_extracts_reply_to():
    b = _meta_block(["Document Number: P1234R0",
                      "Reply-to: Alice <alice@x.com>"])
    meta, consumed = extract_metadata_from_blocks([b])
    assert "reply-to" in meta
    assert any("Alice" in a for a in meta["reply-to"])


def test_title_picks_largest_font():
    title_block = _meta_block(["My Paper Title"], font_size=16.0)
    label_block = _meta_block(["Subtitle Line"], font_size=9.0)
    meta_block = _meta_block(["Document Number: P1234R0"], font_size=9.0)
    meta, consumed = extract_metadata_from_blocks(
        [label_block, title_block, meta_block])
    assert meta.get("title") == "My Paper Title"


def test_pre_label_blocks_consumed():
    cat_block = _meta_block(["WG21 PROPOSAL"], font_size=9.0)
    title_block = _meta_block(["Real Title"], font_size=16.0)
    meta_block = _meta_block(["Document Number: P1234R0"], font_size=9.0)
    meta, consumed = extract_metadata_from_blocks(
        [cat_block, title_block, meta_block])
    assert 0 in consumed
    assert 1 in consumed


def test_color_lightness_lookup():
    colors = {100.0: 0.42, 200.0: 0.17}
    assert _lookup_lightness(colors, 100.0) == 0.42
    assert _lookup_lightness(colors, 102.0) == 0.42
    assert _lookup_lightness(colors, 300.0) == 0.0


def test_color_lightness_empty():
    assert _lookup_lightness(None, 100.0) == 0.0
    assert _lookup_lightness({}, 100.0) == 0.0


def test_parse_authors_name_email():
    lines = ["Alice Smith alice@example.com"]
    authors = _parse_authors(lines)
    assert len(authors) == 1
    assert "Alice Smith" in authors[0]
    assert "alice@example.com" in authors[0]


def test_parse_authors_name_then_email():
    lines = ["Bob Jones", "bob@example.com"]
    authors = _parse_authors(lines)
    assert len(authors) == 1
    assert "Bob Jones" in authors[0]
