"""Tests for lib.pdf.structure."""

from conftest import make_block, make_section
from lib.pdf.types import (
    Block, Line, Span, Section, SectionKind, Confidence,
)
from lib.pdf.structure import (
    compare_extractions, structure_sections,
    _heading_confidence, _word_similarity, _detect_body_size,
    _merge_paragraphs,
)


class TestHeadingConfidence:
    def test_number_font_bold(self):
        level, conf = _heading_confidence(True, 2, 2, True, False)
        assert level == 2
        assert conf == Confidence.HIGH

    def test_number_font_no_bold(self):
        level, conf = _heading_confidence(True, 2, 2, False, False)
        assert level == 2
        assert conf == Confidence.MEDIUM

    def test_number_font_disagree(self):
        level, conf = _heading_confidence(True, 2, 3, False, False)
        assert level == 2
        assert conf == Confidence.MEDIUM

    def test_number_bold_no_font(self):
        level, conf = _heading_confidence(True, 2, None, True, False)
        assert level == 2
        assert conf == Confidence.MEDIUM

    def test_number_alone(self):
        level, conf = _heading_confidence(True, 2, None, False, False)
        assert level == 2
        assert conf == Confidence.LOW

    def test_font_known_bold(self):
        level, conf = _heading_confidence(False, 0, 1, True, True)
        assert level == 2
        assert conf == Confidence.HIGH

    def test_font_known_no_bold(self):
        level, conf = _heading_confidence(False, 0, 1, False, True)
        assert level == 2
        assert conf == Confidence.MEDIUM

    def test_font_bold(self):
        level, conf = _heading_confidence(False, 0, 1, True, False)
        assert level == 2
        assert conf == Confidence.MEDIUM

    def test_font_alone(self):
        level, conf = _heading_confidence(False, 0, 1, False, False)
        assert level == 2
        assert conf == Confidence.LOW

    def test_known_alone(self):
        level, conf = _heading_confidence(False, 0, None, False, True)
        assert level == 2
        assert conf == Confidence.LOW

    def test_nothing(self):
        level, conf = _heading_confidence(False, 0, None, False, False)
        assert level == 0
        assert conf == Confidence.UNCERTAIN


class TestWordSimilarity:
    def test_identical(self):
        assert _word_similarity(["a", "b"], ["a", "b"]) == 1.0

    def test_disjoint(self):
        assert _word_similarity(["a", "b"], ["c", "d"]) == 0.0

    def test_partial_overlap(self):
        sim = _word_similarity(["a", "b", "c"], ["a", "b", "d"])
        assert 0.5 < sim < 1.0

    def test_both_empty(self):
        assert _word_similarity([], []) == 1.0

    def test_one_empty(self):
        assert _word_similarity(["a"], []) == 0.0


class TestCompareExtractions:
    def test_identical_blocks_confident(self):
        m = [make_block(["hello world"], page_num=0)]
        s = [make_block(["hello world"], page_num=0)]
        sections = compare_extractions(m, s)
        assert all(sec.kind != SectionKind.UNCERTAIN for sec in sections)

    def test_different_blocks_uncertain(self):
        m = [make_block(["The quick brown fox jumps over the lazy dog and then some more words"], page_num=0)]
        s = [make_block(["Completely unrelated text about different topics entirely with enough words here"], page_num=0)]
        sections = compare_extractions(m, s)
        assert any(sec.kind == SectionKind.UNCERTAIN for sec in sections)

    def test_tiny_uncertain_demoted(self):
        m = [make_block(["short"], page_num=0)]
        s = [make_block(["diff"], page_num=0)]
        sections = compare_extractions(m, s)
        uncertain = [s for s in sections if s.kind == SectionKind.UNCERTAIN]
        assert len(uncertain) == 0


class TestMergeParagraphs:
    def test_merges_continuation(self):
        s1 = make_section("Some text without terminal")
        s2 = make_section("continuation here")
        result = _merge_paragraphs([s1, s2])
        assert len(result) == 1

    def test_no_merge_with_terminal(self):
        s1 = make_section("Some text with terminal.")
        s2 = make_section("Next paragraph.")
        result = _merge_paragraphs([s1, s2])
        assert len(result) == 2

    def test_no_mutation(self):
        s1 = make_section("Some text without terminal")
        s2 = make_section("continuation here")
        original_text = s1.text
        _merge_paragraphs([s1, s2])
        assert s1.text == original_text


class TestDetectBodySize:
    def test_returns_mode(self):
        sections = [
            make_section("body text", font_size=10.0),
            make_section("body text", font_size=10.0),
            make_section("heading", font_size=14.0),
        ]
        assert _detect_body_size(sections) == 10.0

    def test_empty_sections(self):
        from lib.pdf.types import FALLBACK_BODY_SIZE
        assert _detect_body_size([]) == FALLBACK_BODY_SIZE
