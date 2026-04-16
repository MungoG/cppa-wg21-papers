"""Tests for fix operations."""
from helpers import _lines
from cite import (
    RefEntry,
    ResolveResult,
    PaperMetadata,
    remove_orphan_refs,
    add_missing_ref_entries,
    fix_unversioned_refs,
    fix_title_mismatches,
    normalize_ref_format,
    normalize_title,
    strip_trailing_urls,
    space_ref_entries,
    demote_h1_refs,
)


class TestRemoveOrphanRefs:
    def test_blanks_orphan_lines(self):
        lines = _lines("""\
## References

[1] First ref
[2] Second ref
[3] Third ref""")
        refs = {
            1: RefEntry(1, 'First ref', 2, 'A'),
            2: RefEntry(2, 'Second ref', 3, 'A'),
            3: RefEntry(3, 'Third ref', 4, 'A'),
        }
        result = remove_orphan_refs(lines, [2], refs)
        assert result[3] == '\n'
        assert '[1] First ref' in result[2]
        assert '[3] Third ref' in result[4]

    def test_no_orphans(self):
        lines = _lines("## References\n[1] Ref")
        refs = {1: RefEntry(1, 'Ref', 1, 'A')}
        result = remove_orphan_refs(lines, [], refs)
        assert result == lines

    def test_does_not_mutate_input(self):
        lines = _lines("## References\n[1] Ref\n[2] Orphan")
        refs = {
            1: RefEntry(1, 'Ref', 1, 'A'),
            2: RefEntry(2, 'Orphan', 2, 'A'),
        }
        original = list(lines)
        remove_orphan_refs(lines, [2], refs)
        assert lines == original


class TestFixUnversionedRefs:
    def test_uses_ref_lookup(self):
        lines = _lines("See P2300 for details.\n## References\n[1] P2300R10, stuff")
        refs = {1: RefEntry(1, 'P2300R10, stuff', 2, 'A')}
        result = fix_unversioned_refs(
            lines, [(0, 'P2300')], refs, ResolveResult())
        assert 'P2300R10' in result[0]

    def test_uses_metadata_fallback(self):
        lines = _lines("See P1234 for details.\n## References")
        resolved = ResolveResult()
        resolved.metadata['P1234R3'] = PaperMetadata(
            paper_id='P1234R3', title='Test', authors='A', date='', url='')
        result = fix_unversioned_refs(
            lines, [(0, 'P1234')], {}, resolved)
        assert 'P1234R3' in result[0]

    def test_no_match_unchanged(self):
        lines = _lines("See P9999 for details.\n## References")
        result = fix_unversioned_refs(
            lines, [(0, 'P9999')], {}, ResolveResult())
        assert 'P9999' in result[0]


class TestNormalizeTitle:
    def test_strips_quotes(self):
        assert normalize_title('"std::execution"') == 'std::execution'

    def test_strips_backticks(self):
        assert normalize_title('`std::execution`') == 'std::execution'

    def test_strips_trailing_punct(self):
        assert normalize_title('std::execution,') == 'std::execution'

    def test_collapses_whitespace(self):
        assert normalize_title('a   b  c') == 'a b c'

    def test_case_insensitive(self):
        assert normalize_title('Std::Execution') == 'std::execution'


class TestFixTitleMismatches:
    def test_updates_title(self):
        lines = _lines("""\
## References

[1] P2300R10, "old title," Eric, https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html""")
        refs = {1: RefEntry(1, lines[2].strip()[4:], 2, 'A')}
        resolved = ResolveResult()
        resolved.metadata['P2300R10'] = PaperMetadata(
            paper_id='P2300R10', title='new title',
            authors='Eric', date='', url='')
        result = fix_title_mismatches(lines, refs, resolved)
        assert '"new title"' in result[2]

    def test_no_mismatch_unchanged(self):
        lines = _lines("""\
## References

[1] P2300R10, "std::execution," Eric, https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html""")
        refs = {1: RefEntry(1, lines[2].strip()[4:], 2, 'A')}
        resolved = ResolveResult()
        resolved.metadata['P2300R10'] = PaperMetadata(
            paper_id='P2300R10', title='std::execution',
            authors='Eric', date='', url='')
        result = fix_title_mismatches(lines, refs, resolved)
        assert result[2] == lines[2]

    def test_correct_index_title_applied(self):
        """Index.json has correct titles, so updates should always apply."""
        lines = _lines("""\
## References

[1] N2406, "old title," Hinnant, https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2406.html""")
        refs = {1: RefEntry(1, lines[2].strip()[4:], 2, 'A')}
        resolved = ResolveResult()
        resolved.metadata['N2406'] = PaperMetadata(
            paper_id='N2406',
            title='Mutex, Lock, Condition Variable Rationale',
            authors='Hinnant', date='', url='')
        result = fix_title_mismatches(lines, refs, resolved)
        assert 'Mutex, Lock' in result[2]


class TestNormalizeRefFormat:
    def test_converts_b_to_a(self):
        lines = _lines("""\
## References

1. First ref
2. Second ref""")
        result = normalize_ref_format(lines, 0, len(lines))
        assert '[1] First ref' in result[2]
        assert '[2] Second ref' in result[3]

    def test_leaves_a_unchanged(self):
        lines = _lines("""\
## References

[1] First ref
[2] Second ref""")
        result = normalize_ref_format(lines, 0, len(lines))
        assert result == lines

    def test_skips_blank_lines(self):
        lines = _lines("""\
## References

1. First ref

2. Second ref""")
        result = normalize_ref_format(lines, 0, len(lines))
        assert '[1] First ref' in result[2]
        assert result[3] == '\n'
        assert '[2] Second ref' in result[4]


class TestStripTrailingUrls:
    def test_removes_duplicate_url(self):
        lines = _lines("""\
## References

[1] [P2300R10](https://example.com/p2300r10.html) - title. https://example.com/p2300r10.html""")
        result = strip_trailing_urls(lines, 0, len(lines))
        assert 'https://example.com' not in result[2].split(')', 1)[-1]
        assert '[P2300R10](https://example.com/p2300r10.html)' in result[2]

    def test_wraps_bare_url(self):
        lines = _lines("""\
## References

[1] Capy, https://github.com/cppalliance/capy""")
        result = strip_trailing_urls(lines, 0, len(lines))
        assert '[Capy](https://github.com/cppalliance/capy)' in result[2]

    def test_no_url_unchanged(self):
        lines = _lines("""\
## References

[1] Some reference without a URL""")
        result = strip_trailing_urls(lines, 0, len(lines))
        assert result == lines


class TestSpaceRefEntries:
    def test_inserts_blank_lines(self):
        lines = _lines("""\
## References

[1] First ref
[2] Second ref
[3] Third ref""")
        result = space_ref_entries(lines, 0, len(lines))
        content = ''.join(result)
        assert '[1] First ref\n\n[2] Second ref' in content
        assert '[2] Second ref\n\n[3] Third ref' in content

    def test_no_double_spacing(self):
        lines = _lines("""\
## References

[1] First ref

[2] Second ref

[3] Third ref""")
        result = space_ref_entries(lines, 0, len(lines))
        assert result == lines

    def test_mixed_spacing(self):
        lines = _lines("""\
## References

[1] First ref
[2] Second ref

[3] Third ref""")
        result = space_ref_entries(lines, 0, len(lines))
        content = ''.join(result)
        assert '[1] First ref\n\n[2] Second ref' in content
        assert '[2] Second ref\n\n[3] Third ref' in content


class TestDemoteH1Refs:
    def test_demotes_h1(self):
        lines = _lines("""\
# References

[1] Ref""")
        result = demote_h1_refs(lines, 0)
        assert result[0].startswith('## References')

    def test_leaves_h2_unchanged(self):
        lines = _lines("""\
## References

[1] Ref""")
        result = demote_h1_refs(lines, 0)
        assert result[0] == lines[0]

    def test_handles_extra_whitespace(self):
        lines = _lines("""\
#  References

[1] Ref""")
        result = demote_h1_refs(lines, 0)
        assert result[0].startswith('##  References')


class TestAddMissingRefEntries:
    def test_creates_stub(self):
        lines = _lines("""\
Body <sup>[1]</sup> text.

## References
""")
        result = add_missing_ref_entries(
            lines, [1], [(0, 1)], 3, ResolveResult())
        content = ''.join(result)
        assert '[1] [TODO: Add reference]' in content

    def test_creates_from_link_context(self):
        lines = _lines("""\
See [example](https://example.com)<sup>[1]</sup>.

## References
""")
        result = add_missing_ref_entries(
            lines, [1], [(0, 1)], 3, ResolveResult())
        content = ''.join(result)
        assert '[1] example, https://example.com' in content
