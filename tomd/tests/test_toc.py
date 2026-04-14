"""Tests for lib.toc."""

from lib.toc import _normalize_toc_entry, find_toc_indices


def test_normalize_strips_dot_leaders():
    assert "abstract" in _normalize_toc_entry("Abstract ......... 5")


def test_normalize_strips_trailing_page_number():
    result = _normalize_toc_entry("Abstract 42")
    assert "42" not in result


def test_normalize_strips_section_prefix():
    assert _normalize_toc_entry("2.1 Introduction") == "introduction"


def test_normalize_lowercases():
    assert _normalize_toc_entry("ABSTRACT") == "abstract"


def test_normalize_collapses_whitespace():
    assert "  " not in _normalize_toc_entry("Some   Entry")


def test_find_toc_basic_run():
    texts = ["Abstract", "Introduction", "Motivation", "Body text here"]
    headings = {"Abstract", "Introduction", "Motivation"}
    indices = find_toc_indices(texts, headings)
    assert 0 in indices
    assert 1 in indices
    assert 2 in indices
    assert 3 not in indices


def test_find_toc_gap_bridging():
    texts = ["Abstract", "non-match", "Introduction", "Motivation"]
    headings = {"Abstract", "Introduction", "Motivation"}
    indices = find_toc_indices(texts, headings)
    assert 1 in indices


def test_find_toc_too_few_matches():
    texts = ["Abstract", "Introduction"]
    headings = {"Abstract", "Introduction"}
    indices = find_toc_indices(texts, headings)
    assert len(indices) == 0


def test_find_toc_duplicate_stops_scan():
    texts = ["Abstract", "Introduction", "Motivation",
             "Abstract"]
    headings = {"Abstract", "Introduction", "Motivation"}
    indices = find_toc_indices(texts, headings)
    assert 3 not in indices


def test_find_toc_label_included():
    texts = ["Table of Contents", "Abstract", "Introduction", "Motivation"]
    headings = {"Abstract", "Introduction", "Motivation"}
    indices = find_toc_indices(texts, headings)
    assert 0 in indices


def test_find_toc_empty_inputs():
    assert find_toc_indices([], set()) == set()
    assert find_toc_indices(["x"], set()) == set()
    assert find_toc_indices([], {"x"}) == set()
