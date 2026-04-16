"""Tests for citation extraction, reference parsing, and unversioned ref detection."""
from helpers import _lines
from cite import (
    build_exclusion_ranges,
    extract_body_citations,
    find_refs_section,
    parse_references,
    check_unversioned_refs,
)


def _lines(text):
    return [line + '\n' for line in text.strip().split('\n')]


class TestExtractBodyCitations:
    def test_consecutive_order(self):
        lines = _lines("""\
First <sup>[1]</sup> then <sup>[2]</sup>.
Next <sup>[3]</sup>.
## References
[1] Ref one
[2] Ref two
[3] Ref three""")
        excluded = build_exclusion_ranges(lines)
        refs_start, _ = find_refs_section(lines)
        cites, first_app, old_to_new = extract_body_citations(
            lines, excluded, refs_start)
        assert first_app == [1, 2, 3]
        assert old_to_new == {1: 1, 2: 2, 3: 3}

    def test_out_of_order(self):
        lines = _lines("""\
First <sup>[3]</sup> then <sup>[1]</sup>.
Next <sup>[2]</sup>.
## References
[1] Ref
[2] Ref
[3] Ref""")
        excluded = build_exclusion_ranges(lines)
        refs_start, _ = find_refs_section(lines)
        cites, first_app, old_to_new = extract_body_citations(
            lines, excluded, refs_start)
        assert first_app == [3, 1, 2]
        assert old_to_new == {3: 1, 1: 2, 2: 3}

    def test_skips_code_blocks(self):
        lines = _lines("""\
Body <sup>[1]</sup>.
```
Code <sup>[5]</sup>.
```
More <sup>[2]</sup>.
## References""")
        excluded = build_exclusion_ranges(lines)
        refs_start, _ = find_refs_section(lines)
        cites, first_app, old_to_new = extract_body_citations(
            lines, excluded, refs_start)
        assert first_app == [1, 2]
        assert 5 not in old_to_new

    def test_repeated_citation(self):
        lines = _lines("""\
First <sup>[2]</sup> and <sup>[1]</sup>.
Again <sup>[2]</sup>.
## References""")
        excluded = build_exclusion_ranges(lines)
        refs_start, _ = find_refs_section(lines)
        cites, first_app, old_to_new = extract_body_citations(
            lines, excluded, refs_start)
        assert first_app == [2, 1]
        assert old_to_new == {2: 1, 1: 2}
        assert len(cites) == 3


class TestFindRefsSection:
    def test_h2_references(self):
        lines = _lines("""\
body
## References
[1] Ref""")
        start, end = find_refs_section(lines)
        assert start == 1
        assert end == 3

    def test_h1_references(self):
        lines = _lines("""\
body
# References
[1] Ref""")
        start, end = find_refs_section(lines)
        assert start == 1

    def test_no_references(self):
        lines = _lines("Just a body\nNo refs here")
        start, end = find_refs_section(lines)
        assert start == -1

    def test_section_after_references(self):
        lines = _lines("""\
body
## References
[1] Ref
## Appendix
appendix text""")
        start, end = find_refs_section(lines)
        assert start == 1
        assert end == 3


class TestParseReferences:
    def test_format_a(self):
        lines = _lines("""\
## References

[1] C++ Working Draft, https://eel.is/c++draft/
[2] Boost.URL, https://github.com/boostorg/url""")
        refs = parse_references(lines, 0, len(lines))
        assert 1 in refs
        assert 2 in refs
        assert refs[1].format == 'A'
        assert 'C++ Working Draft' in refs[1].text

    def test_format_b(self):
        lines = _lines("""\
## References

1. [C++ Working Draft](https://eel.is/c++draft/) - description
2. [Boost.URL](https://github.com/boostorg/url) - description""")
        refs = parse_references(lines, 0, len(lines))
        assert 1 in refs
        assert 2 in refs
        assert refs[1].format == 'B'

    def test_format_a_with_dash_prefix(self):
        lines = _lines("""\
## References

- [1] First ref
- [2] Second ref""")
        refs = parse_references(lines, 0, len(lines))
        assert 1 in refs
        assert 2 in refs
        assert refs[1].format == 'A'


class TestCheckUnversionedRefs:
    def test_finds_bare_paper_number(self):
        lines = _lines("See P2300 for details.\n## References")
        excluded = build_exclusion_ranges(lines)
        results = check_unversioned_refs(lines, excluded, 1)
        assert len(results) == 1
        assert results[0][1] == 'P2300'

    def test_versioned_not_flagged(self):
        lines = _lines("See P2300R10 for details.\n## References")
        excluded = build_exclusion_ranges(lines)
        results = check_unversioned_refs(lines, excluded, 1)
        assert len(results) == 0

    def test_d_prefix(self):
        lines = _lines("See D4035 for details.\n## References")
        excluded = build_exclusion_ranges(lines)
        results = check_unversioned_refs(lines, excluded, 1)
        assert len(results) == 1
        assert results[0][1] == 'D4035'

    def test_skips_code_blocks(self):
        lines = _lines("""\
```
P2300
```
## References""")
        excluded = build_exclusion_ranges(lines)
        results = check_unversioned_refs(lines, excluded, 3)
        assert len(results) == 0
