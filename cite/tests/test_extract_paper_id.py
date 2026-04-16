"""Tests for extract_paper_id_from_url."""
from cite import extract_paper_id_from_url


class TestExtractPaperId:
    def test_open_std_url(self):
        url = ('https://www.open-std.org/jtc1/sc22/wg21/'
               'docs/papers/2024/p2300r10.html')
        result = extract_paper_id_from_url(url)
        assert result is not None
        pid, year = result
        assert pid == 'P2300R10'
        assert year == 2024

    def test_open_std_n_paper(self):
        url = ('https://www.open-std.org/jtc1/sc22/wg21/'
               'docs/papers/2007/n2406.html')
        result = extract_paper_id_from_url(url)
        assert result is not None
        pid, year = result
        assert pid == 'N2406'
        assert year == 2007

    def test_wg21_link(self):
        url = 'https://wg21.link/p2300r10'
        result = extract_paper_id_from_url(url)
        assert result is not None
        pid, year = result
        assert pid == 'P2300R10'
        assert year is None

    def test_non_paper_url(self):
        url = 'https://github.com/boostorg/url'
        result = extract_paper_id_from_url(url)
        assert result is None

    def test_open_std_pdf(self):
        url = ('https://www.open-std.org/jtc1/sc22/wg21/'
               'docs/papers/2024/p0843r10.html')
        result = extract_paper_id_from_url(url)
        assert result is not None
        pid, year = result
        assert pid == 'P0843R10'
        assert year == 2024
