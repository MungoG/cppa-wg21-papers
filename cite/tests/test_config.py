"""Tests for config loading and exemption matching."""
from pathlib import Path

from cite import load_config, is_exempt_link, is_exempt_section

import pytest


class TestLoadConfig:
    def test_none_path(self):
        config = load_config(None)
        assert config['exempt_sections'] == []
        assert config['exempt_links'] == []
        assert config['exempt_orphans'] == []

    def test_missing_file(self):
        config = load_config('/nonexistent/path/cite.yaml')
        assert config['exempt_sections'] == []

    def test_valid_config(self, tmp_path):
        p = tmp_path / 'cite.yaml'
        p.write_text("""\
exempt_sections:
  - Disclosure
  - Acknowledgements
exempt_links:
  - "https://example.com/*"
""")
        config = load_config(str(p))
        assert 'Disclosure' in config['exempt_sections']
        assert config['exempt_links'] == ['https://example.com/*']
        assert config['exempt_orphans'] == []

    def test_unknown_key_raises(self, tmp_path):
        p = tmp_path / 'cite.yaml'
        p.write_text("bogus_key: true\n")
        with pytest.raises(ValueError, match='unknown config keys'):
            load_config(str(p))

    def test_empty_file(self, tmp_path):
        p = tmp_path / 'cite.yaml'
        p.write_text("")
        config = load_config(str(p))
        assert config['exempt_sections'] == []

    def test_null_value_treated_as_empty(self, tmp_path):
        p = tmp_path / 'cite.yaml'
        p.write_text("exempt_sections:\n")
        config = load_config(str(p))
        assert config['exempt_sections'] == []

    def test_non_list_value_raises(self, tmp_path):
        p = tmp_path / 'cite.yaml'
        p.write_text("exempt_sections: just-a-string\n")
        with pytest.raises(ValueError, match='must be a list'):
            load_config(str(p))


class TestIsExemptLink:
    def test_glob_match(self):
        config = {'exempt_links': ['https://github.com/cppalliance/*']}
        assert is_exempt_link('https://github.com/cppalliance/corosio', config)
        assert not is_exempt_link('https://github.com/boostorg/url', config)

    def test_exact_match(self):
        config = {'exempt_links': ['https://example.com/exact']}
        assert is_exempt_link('https://example.com/exact', config)
        assert not is_exempt_link('https://example.com/exact/sub', config)

    def test_empty_config(self):
        assert not is_exempt_link('https://anything.com', {})
        assert not is_exempt_link('https://anything.com', {'exempt_links': []})


class TestIsExemptSection:
    def test_case_insensitive(self):
        config = {'exempt_sections': ['Disclosure']}
        assert is_exempt_section('Disclosure', config)
        assert is_exempt_section('disclosure', config)
        assert is_exempt_section('DISCLOSURE', config)

    def test_no_match(self):
        config = {'exempt_sections': ['Disclosure']}
        assert not is_exempt_section('Introduction', config)

    def test_empty(self):
        assert not is_exempt_section('Anything', {})
