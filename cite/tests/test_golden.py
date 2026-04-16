"""Golden file tests - run cite.py end-to-end and compare output."""
from pathlib import Path

from cite import scan, write, load_config, ResolveResult

GOLDEN_DIR = Path(__file__).resolve().parent / 'golden'


def _golden_pairs():
    """Yield (input_path, expected_path, fix_mode) for each golden test."""
    inputs = sorted(GOLDEN_DIR.glob('*.input.md'))
    for inp in inputs:
        name = inp.name.replace('.input.md', '')
        exp = inp.parent / f'{name}.expected.md'
        fix_mode = name.startswith('fix-')
        if exp.exists():
            yield inp, exp, fix_mode


def _compare(inp_name, output, expected):
    if output != expected:
        output_lines = output.splitlines(keepends=True)
        expected_lines = expected.splitlines(keepends=True)
        for i, (ol, el) in enumerate(zip(output_lines, expected_lines)):
            if ol != el:
                assert False, (
                    f"{inp_name} line {i+1} differs:\n"
                    f"  got:      {ol.rstrip()}\n"
                    f"  expected: {el.rstrip()}"
                )
        if len(output_lines) != len(expected_lines):
            assert False, (
                f"{inp_name}: line count differs "
                f"(got {len(output_lines)}, "
                f"expected {len(expected_lines)})"
            )


def test_golden_pairs_exist():
    pairs = list(_golden_pairs())
    assert len(pairs) > 0, "No golden file pairs found"


def test_golden_files():
    config = load_config(None)
    empty_resolved = ResolveResult()

    for inp, exp, fix_mode in _golden_pairs():
        result = scan(str(inp), config)
        output = write(result, empty_resolved, fix=fix_mode)
        expected = exp.read_text(encoding='utf-8')
        _compare(inp.name, output, expected)


def test_idempotency():
    """Running scan+write twice on the same input produces identical output."""
    config = load_config(None)
    empty_resolved = ResolveResult()

    for inp, _exp, fix_mode in _golden_pairs():
        result1 = scan(str(inp), config)
        output1 = write(result1, empty_resolved, fix=fix_mode)

        import tempfile, os
        fd, tmp = tempfile.mkstemp(suffix='.md')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(output1)
            result2 = scan(tmp, config)
            output2 = write(result2, empty_resolved, fix=fix_mode)
        finally:
            os.unlink(tmp)

        assert output1 == output2, (
            f"Idempotency failed for {inp.name}: "
            f"second pass produced different output"
        )
