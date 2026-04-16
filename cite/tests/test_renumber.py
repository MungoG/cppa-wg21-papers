"""Tests for renumbering, reordering, and orphan detection."""
from helpers import _lines
from cite import (
    RefEntry,
    renumber_content,
    reorder_refs,
)


class TestRenumberContent:
    def test_identity(self):
        lines = _lines("See <sup>[1]</sup> and <sup>[2]</sup>.")
        old_to_new = {1: 1, 2: 2}
        result = renumber_content(old_to_new, set(), lines)
        assert '<sup>[1]</sup>' in result
        assert '<sup>[2]</sup>' in result

    def test_swap(self):
        lines = _lines("First <sup>[3]</sup> then <sup>[1]</sup>.")
        old_to_new = {3: 1, 1: 2}
        result = renumber_content(old_to_new, set(), lines)
        assert 'First <sup>[1]</sup>' in result
        assert 'then <sup>[2]</sup>' in result

    def test_skips_excluded_lines(self):
        lines = _lines("""\
Body <sup>[2]</sup>.
```
Code <sup>[2]</sup>.
```""")
        excluded = {1, 2, 3}
        old_to_new = {2: 1}
        result = renumber_content(old_to_new, excluded, lines)
        result_lines = result.split('\n')
        assert '<sup>[1]</sup>' in result_lines[0]
        assert '<sup>[2]</sup>' in result_lines[2]

    def test_no_collision(self):
        """Renumbering [1]->[2] and [2]->[1] must not collide."""
        lines = _lines("<sup>[1]</sup> and <sup>[2]</sup>.")
        old_to_new = {1: 2, 2: 1}
        result = renumber_content(old_to_new, set(), lines)
        assert '<sup>[2]</sup> and <sup>[1]</sup>' in result

    def test_multiple_on_one_line(self):
        lines = _lines("See <sup>[3]</sup><sup>[1]</sup><sup>[2]</sup>.")
        old_to_new = {3: 1, 1: 2, 2: 3}
        result = renumber_content(old_to_new, set(), lines)
        assert '<sup>[1]</sup><sup>[2]</sup><sup>[3]</sup>' in result


class TestReorderRefs:
    def test_basic_reorder_format_a(self):
        lines = _lines("""\
## References

[3] Third ref
[1] First ref
[2] Second ref""")
        refs = {
            3: RefEntry(3, 'Third ref', 2, 'A'),
            1: RefEntry(1, 'First ref', 3, 'A'),
            2: RefEntry(2, 'Second ref', 4, 'A'),
        }
        old_to_new = {3: 1, 1: 2, 2: 3}
        result = reorder_refs(lines, refs, old_to_new)
        assert '[1] Third ref' in result[2]
        assert '[2] First ref' in result[3]
        assert '[3] Second ref' in result[4]

    def test_format_b_emits_format_a(self):
        """reorder_refs always emits [N] format regardless of input."""
        lines = _lines("""\
## References

3. Third ref
1. First ref
2. Second ref""")
        refs = {
            3: RefEntry(3, 'Third ref', 2, 'B'),
            1: RefEntry(1, 'First ref', 3, 'B'),
            2: RefEntry(2, 'Second ref', 4, 'B'),
        }
        old_to_new = {3: 1, 1: 2, 2: 3}
        result = reorder_refs(lines, refs, old_to_new)
        assert '[1] Third ref' in result[2]
        assert '[2] First ref' in result[3]
        assert '[3] Second ref' in result[4]

    def test_preserves_non_entry_lines(self):
        lines = _lines("""\
## References

### Papers

[1] First
[2] Second

### Other

[3] Third""")
        refs = {
            1: RefEntry(1, 'First', 4, 'A'),
            2: RefEntry(2, 'Second', 5, 'A'),
            3: RefEntry(3, 'Third', 9, 'A'),
        }
        old_to_new = {1: 1, 2: 2, 3: 3}
        result = reorder_refs(lines, refs, old_to_new)
        content = ''.join(result)
        assert '### Papers' in content
        assert '### Other' in content
        assert '[1] First' in content
        assert '[3] Third' in content
