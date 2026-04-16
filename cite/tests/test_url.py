"""Tests for wg21.link detection, HTTP fallback, and URL replacement."""
from unittest.mock import patch, MagicMock
from urllib.error import URLError

from helpers import _lines
from cite import (
    find_wg21_links,
    _resolve_via_http,
    apply_wg21_replacements,
    build_exclusion_ranges,
)


class TestFindWg21Links:
    def test_finds_links(self):
        lines = _lines("""\
See [P2300R10](https://wg21.link/p2300r10) for details.
Also https://wg21.link/p3552r3 here.""")
        excluded = build_exclusion_ranges(lines)
        results = find_wg21_links(lines, excluded)
        assert len(results) == 2
        assert results[0].slug.lower() == 'p2300r10'
        assert results[1].slug.lower() == 'p3552r3'

    def test_skips_code_blocks(self):
        lines = _lines("""\
Body https://wg21.link/p2300r10.
```
Code https://wg21.link/p3552r3.
```""")
        excluded = build_exclusion_ranges(lines)
        results = find_wg21_links(lines, excluded)
        assert len(results) == 1
        assert results[0].slug.lower() == 'p2300r10'

    def test_no_wg21_links(self):
        lines = _lines("See https://open-std.org/paper.html for details.")
        excluded = build_exclusion_ranges(lines)
        results = find_wg21_links(lines, excluded)
        assert len(results) == 0


class TestResolveViaHttp:
    @patch('cite.urlopen')
    def test_resolves_url(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.url = (
            'https://www.open-std.org/jtc1/sc22/wg21/'
            'docs/papers/2024/p2300r10.html')
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = _resolve_via_http('https://wg21.link/p2300r10')
        assert result is not None
        assert 'open-std.org' in result

    @patch('cite.urlopen', side_effect=URLError('network error'))
    def test_returns_none_on_failure(self, _mock):
        result = _resolve_via_http('https://wg21.link/p2300r10')
        assert result is None


class TestApplyWg21Replacements:
    def test_replaces_urls(self):
        lines = _lines("""\
See [P2300R10](https://wg21.link/p2300r10)<sup>[1]</sup>.
Also https://wg21.link/p2300r10 in refs.""")
        replacements = {
            'https://wg21.link/p2300r10':
                'https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html',
        }
        result = apply_wg21_replacements(lines, replacements, set())
        for line in result:
            assert 'wg21.link' not in line
            if 'P2300R10' in line or 'p2300r10' in line:
                assert 'open-std.org' in line

    def test_no_replacements(self):
        lines = _lines("No wg21 links here.")
        result = apply_wg21_replacements(lines, {}, set())
        assert result == lines

    def test_skips_excluded_lines(self):
        lines = _lines("""\
Body https://wg21.link/p2300r10 here.
```
https://wg21.link/p2300r10
```
After.""")
        replacements = {
            'https://wg21.link/p2300r10':
                'https://www.open-std.org/.../p2300r10.html',
        }
        excluded = {1, 2, 3}
        result = apply_wg21_replacements(lines, replacements, excluded)
        assert 'open-std.org' in result[0]
        assert 'wg21.link' in result[2]
