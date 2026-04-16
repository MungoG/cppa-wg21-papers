"""Tests for exempt_orphans config wiring."""
import tempfile
import os

from cite import scan, load_config


def test_exempt_orphans_suppresses_finding():
    content = """\
Body text <sup>[1]</sup>.

## References

[1] Cited ref, https://example.com

[2] Orphan github ref, https://github.com/cppalliance/capy
"""
    config = {
        'exempt_sections': [],
        'exempt_links': [],
        'exempt_orphans': ['*github.com/cppalliance*'],
    }
    fd, tmp = tempfile.mkstemp(suffix='.md')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(content)
        result = scan(tmp, config)
    finally:
        os.unlink(tmp)

    assert 2 not in result.orphans
    orphan_findings = [
        f for f in result.findings if f.rule == 'orphan-ref']
    assert len(orphan_findings) == 0


def test_non_exempt_orphan_still_flagged():
    content = """\
Body text <sup>[1]</sup>.

## References

[1] Cited ref

[2] Non-exempt orphan
"""
    config = {
        'exempt_sections': [],
        'exempt_links': [],
        'exempt_orphans': ['*github.com*'],
    }
    fd, tmp = tempfile.mkstemp(suffix='.md')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(content)
        result = scan(tmp, config)
    finally:
        os.unlink(tmp)

    assert 2 in result.orphans
