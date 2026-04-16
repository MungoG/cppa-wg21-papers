"""Tests for wg21.link index lookup and HTML entity encoding."""
from cite import (
    lookup_paper_metadata,
    _find_latest_revision,
    to_html_entities,
    PaperMetadata,
)


SAMPLE_INDEX = {
    'P2300R10': {
        'type': 'paper',
        'title': 'std::execution',
        'author': 'Eric Niebler, Michal Dominiak',
        'date': '2024-06-28',
        'long_link': 'http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html',
        'link': 'https://wg21.link/p2300r10',
    },
    'P2300R9': {
        'type': 'paper',
        'title': 'std::execution',
        'author': 'Eric Niebler, Michal Dominiak',
        'date': '2024-01-01',
        'long_link': 'http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r9.html',
        'link': 'https://wg21.link/p2300r9',
    },
    'N2406': {
        'type': 'paper',
        'title': 'Mutex, Lock, Condition Variable Rationale',
        'author': 'Howard E. Hinnant',
        'date': '2007-09-09',
        'long_link': 'http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2406.html',
        'link': 'https://wg21.link/n2406',
    },
    'CWG1': {
        'type': 'issue',
        'title': 'Some CWG issue',
        'long_link': 'https://cplusplus.github.io/CWG/issues/1.html',
    },
}


class TestLookupPaperMetadata:
    def test_finds_paper(self):
        meta = lookup_paper_metadata('P2300R10', SAMPLE_INDEX)
        assert meta is not None
        assert meta.paper_id == 'P2300R10'
        assert meta.title == 'std::execution'
        assert 'Eric Niebler' in meta.authors

    def test_case_insensitive(self):
        meta = lookup_paper_metadata('p2300r10', SAMPLE_INDEX)
        assert meta is not None
        assert meta.paper_id == 'P2300R10'

    def test_n_paper(self):
        meta = lookup_paper_metadata('N2406', SAMPLE_INDEX)
        assert meta is not None
        assert meta.title == 'Mutex, Lock, Condition Variable Rationale'
        assert meta.authors == 'Howard E. Hinnant'

    def test_not_found(self):
        meta = lookup_paper_metadata('P9999R0', SAMPLE_INDEX)
        assert meta is None

    def test_non_paper_type_not_returned(self):
        meta = lookup_paper_metadata('CWG1', SAMPLE_INDEX)
        assert meta is None

    def test_url_from_long_link(self):
        meta = lookup_paper_metadata('P2300R10', SAMPLE_INDEX)
        assert 'open-std.org' in meta.url
        assert 'p2300r10' in meta.url


class TestFindLatestRevision:
    def test_finds_max_revision(self):
        entry = _find_latest_revision('P2300', SAMPLE_INDEX)
        assert entry is not None
        assert 'p2300r10' in entry['long_link']

    def test_returns_none_if_not_found(self):
        entry = _find_latest_revision('P9999', SAMPLE_INDEX)
        assert entry is None

    def test_single_revision(self):
        entry = _find_latest_revision('N2406', SAMPLE_INDEX)
        assert entry is not None


class TestToHtmlEntities:
    def test_ascii_unchanged(self):
        assert to_html_entities('Hello World') == 'Hello World'

    def test_named_entity(self):
        result = to_html_entities('Ren\u00e9')
        assert result == 'Ren&eacute;'

    def test_numeric_entity(self):
        result = to_html_entities('Micha\u0142')
        assert result == 'Micha&#322;'

    def test_mixed(self):
        result = to_html_entities('caf\u00e9 Micha\u0142')
        assert '&eacute;' in result
        assert '&#322;' in result
        assert result.isascii()

    def test_empty_string(self):
        assert to_html_entities('') == ''

    def test_all_ascii_output(self):
        text = 'Ga\u0161per A\u017eman'
        result = to_html_entities(text)
        assert result.isascii()
