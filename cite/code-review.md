# Code Review: cite

- **Date:** 2026-04-16
- **Model:** Opus 4.6
- **Scope:** cite/cite.py, cite/tests/

Solid single-module auditor with a well-tested happy path; the few real bugs live in docstring-to-implementation drift and exclusion-range bypass in the replacement pass.

## Executive Summary

cite.py is a focused ~1600-line CLI tool that scans, resolves, and rewrites WG21 markdown citations. The three-pass architecture (scan / resolve / write) is sound, the sentinel-based renumbering is collision-proof, and file writes are correctly atomic. Test coverage is reasonable - renumbering, exclusion, config, parsing, fix operations, and golden-file round-trips are all exercised.

Four should-fix findings survive the adversarial challenge: a docstring that promises a ValueError never raised, a heading demotion that can silently fail on extra whitespace, a `--check` predicate that is narrower than `--fix`, and an exclusion-range bypass in `apply_wg21_replacements`. None are data-loss bugs; all are correctness gaps that can produce silent wrong behavior in edge cases.

## Codebase Profile

### Core Implementation

Single-module CLI tool: scan, resolve, write passes for WG21 citation auditing and rewriting.

### cite/cite.py

Monolithic but well-structured 1623-line CLI tool. Dataclass-driven state (AuditResult, ResolveResult), clean three-pass pipeline, atomic file I/O, sentinel renumbering. Main risks: write-pass mutation chain re-derives refs boundaries after each step; `apply_wg21_replacements` ignores exclusion ranges; `parse_mailing_index` docstring/implementation contract mismatch.

### cite/tests/test_url.py

12 assertions across 3 test classes covering wg21.link detection, HTTP resolution (mocked), and URL replacement.

### cite/tests/test_extract_paper_id.py

5 test cases for `extract_paper_id_from_url`. Covers open-std.org, wg21.link, N-papers, and non-paper URLs. Clean.

### cite/tests/test_mailing_index.py

13 tests across 2 classes: HTML table parsing and HTML entity encoding. Good coverage of multi-table pages and edge cases. Does not test the ValueError path the docstring promises.

### cite/tests/test_fix_ops.py

8 test classes covering orphan removal, unversioned refs, title normalization, title mismatches, format normalization, trailing URL stripping, ref spacing, and H1 demotion. No coverage for `add_missing_ref_entries`.

### cite/tests/test_renumber.py

8 tests across 2 classes for `renumber_content` and `reorder_refs`. Covers identity, swap, collision, exclusion, and format-B-to-A conversion. Solid.

### cite/tests/test_config.py

13 tests across 3 classes for config loading, link exemption, and section exemption.

### cite/tests/test_parse.py

14 tests across 4 classes covering body citation extraction, refs section detection, reference parsing, and unversioned ref checks.

### cite/tests/test_golden.py

2 tests: existence check for golden pairs and parameterized round-trip comparison. End-to-end coverage through scan+write pipeline.

### cite/tests/test_exclusion.py

6 tests for `build_exclusion_ranges`. Covers simple/tagged/multiple/indented fences and citation-inside-fence exclusion. Clean.

## Cross-cutting Analysis

The codebase is a single-module tool with a test suite - no module boundary issues in the traditional sense. The internal architecture does have implicit boundaries worth noting.

The scan/resolve/write pipeline is cleanly separated at the function level, but the write pass breaks the abstraction by re-running scan helpers (`find_refs_section`, `parse_references`, `build_exclusion_ranges`, `extract_body_citations`) after each mutation step. This is a pragmatic choice - mutations shift line indices - but it means `write()` is tightly coupled to scan internals. A mutation that produces malformed output silently propagates through subsequent steps rather than failing fast.

Duplication across test files is the main hygiene issue. The `_lines()` helper is copy-pasted into 5 of 8 test files. The `sys.path.insert` hack appears in all 9 test files. A `conftest.py` with the path fix and a shared `_lines` fixture would eliminate both.

Naming is mostly coherent. The `find_*` functions return typed tuples consistently, with one inconsistency: `find_wg21_links` returns `(line_idx, url, slug)` while `find_uncited_links` returns `(line_idx, text, url)` - similar shapes with different semantics in the same positional slots. Named tuples or small dataclasses would make these explicit.

The exclusion-range pattern is the most critical cross-cutting concern. `build_exclusion_ranges` correctly feeds into all scan functions, but `apply_wg21_replacements` operates on raw lines without checking exclusion. If a wg21.link URL appears both in body text and inside a fenced code block, the replacement rewrites both.

The `_MailingIndexParser.validate_and_get_rows()` method name promises validation it does not perform, connecting directly to the docstring/implementation drift in `parse_mailing_index`. `EXPECTED_MAILING_HEADERS` is defined at module level but never compared against parsed headers.

## Findings

### Should fix

1. `cite.py:707`, `cite.py:699`, `cite.py:819` - implement header validation in `validate_and_get_rows` or remove the ValueError promise from the docstring and delete the dead `except ValueError` in `fetch_paper_metadata`
   (`parse_mailing_index` docstring says "Raises ValueError if table headers don't match expected format" but implementation never validates; `EXPECTED_MAILING_HEADERS` is defined but unused; `fetch_paper_metadata` catches a ValueError that can never be raised)

2. `cite.py:1232` - use regex-based replacement in `demote_h1_refs` instead of literal string replace
   (`REF_HEADING_RE` allows extra whitespace after `#` but `replace` uses the literal `'# References'`, so demotion silently fails when the heading has multiple spaces)

3. `cite.py:1577` - align `--check` predicate with all changes `--fix` would apply
   (`--check` only keys off `needs_renumber`, orphans, `wg21_links`, and error-severity findings; flag-severity issues like uncited-link, unversioned-ref, h1-refs, format-b-ref, and malformed-cite leave exit 0 while `--fix` would still edit the file)

4. `cite.py:905` - pass the exclusion set to `apply_wg21_replacements` and skip excluded lines
   (`find_wg21_links` skips excluded lines but replacement runs on every line; a wg21.link URL inside a fenced code block gets rewritten if the same URL also appears in body text)

### Nice to have

5. `cite.py:951` - remove unused `refs_start`/`refs_end` parameters from `reorder_refs`
   (function accepts but never uses these parameters, misleading readers about boundary enforcement)

6. `cite.py:919` - fix docstring: Pass 1 rewrites all `CITE_RE` matches, not only N in `old_to_new`
   (docstring claims Pass 1 is only for N in `old_to_new` but the loop unconditionally rewrites all `CITE_RE` matches)

7. `cite.py:868` - define merge policy for `all_paper_ids` when same `paper_id` appears with different hint years
   (later assignments overwrite earlier `hint_year` values with no deterministic merge rule)

8. `cite.py:390` - allow optional whitespace between `)` and `<sup>` in uncited-link detection
   (whitespace between markdown link closing `)` and `<sup>` bypasses both citation check branches, producing false uncited-link flags)

9. `cite/tests/*` - extract `_lines()` helper into `conftest.py`
   (identical 2-line helper duplicated across 5 test files)

10. `cite/tests/*` - replace per-file `sys.path.insert` hack with `conftest.py` or proper package install
    (same path hack in all 9 test files)

11. `cite.py:408`, `cite.py:368` - use named tuples for `find_wg21_links` and `find_uncited_links` return types
    (both return `list[tuple[int, str, str]]` but the str fields mean different things in the same positions)

12. `cite/tests/test_fix_ops.py:18` - make `_normalize_title` public or fold its tests into `fix_title_mismatches` tests
    (testing a private function creates refactoring friction; 5 dedicated tests make it de-facto public API)

13. `cite/tests/*` - remove 4 unused imports; consider adding `add_missing_ref_entries` test coverage
    (`WG21_LINK_RE` in test_url.py, `add_missing_ref_entries` in test_fix_ops.py, `tempfile` in test_config.py, `extract_paper_id_from_url` in test_parse.py)

10 files / 50 functions reviewed. 41 functions reviewed clean.
