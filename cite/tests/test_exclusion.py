"""Tests for build_exclusion_ranges."""
from helpers import _lines
from cite import build_exclusion_ranges


def test_no_fences():
    lines = _lines("Hello world\nNo fences here\n")
    assert build_exclusion_ranges(lines) == set()


def test_simple_fence():
    lines = _lines("""\
before
```
code line 1
code line 2
```
after""")
    excluded = build_exclusion_ranges(lines)
    assert 0 not in excluded
    assert 1 in excluded  # opening fence
    assert 2 in excluded
    assert 3 in excluded
    assert 4 in excluded  # closing fence
    assert 5 not in excluded


def test_language_tagged_fence():
    lines = _lines("""\
text
```cpp
int x = 0;
```
more text""")
    excluded = build_exclusion_ranges(lines)
    assert 0 not in excluded
    assert 1 in excluded
    assert 2 in excluded
    assert 3 in excluded
    assert 4 not in excluded


def test_multiple_fences():
    lines = _lines("""\
text
```
block 1
```
middle
```
block 2
```
end""")
    excluded = build_exclusion_ranges(lines)
    assert excluded == {1, 2, 3, 5, 6, 7}


def test_indented_fence():
    lines = _lines("""\
text
  ```
  indented code
  ```
after""")
    excluded = build_exclusion_ranges(lines)
    assert 1 in excluded
    assert 2 in excluded
    assert 3 in excluded


def test_citation_inside_fence_excluded():
    lines = _lines("""\
body <sup>[1]</sup>
```
code <sup>[2]</sup>
```
body <sup>[3]</sup>""")
    excluded = build_exclusion_ranges(lines)
    assert 0 not in excluded
    assert 2 in excluded
    assert 4 not in excluded
