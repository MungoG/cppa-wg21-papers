# Scrivener Test Suite

## Running Tests

```
cd scrivener
pip install -r requirements.txt
pytest tests/ -v
```

Layer 1 tests (pure functions) and HTML tests run without font
downloads. PDF Layer 2-3 tests download fonts on first run and
cache them in `.fonts/`.

## Architecture

The test suite is organized in four layers, from fast/cheap
to slow/expensive.

### Layer 1 - Pure-function unit tests

No ReportLab font registration needed. Tests cover the data
transformation functions that underpin everything else:

- `test_config.py` - deep_merge, palette resolution, front
  matter extraction, config merge priority, option validation,
  style path resolution, style inheritance
- `test_colors.py` - HSL round-trips, color derivation,
  accent resolution, parse_color
- `test_escape.py` - XML entity escaping
- `test_highlight.py` - Pygments-to-ReportLab markup
- `test_renderer_utils.py` - static/utility methods on
  ASTRenderer (_nobr_numbers, _only_text, _propagate_keep)
- `test_css.py` - CSS generation (su() arithmetic, generate_css
  output structure, full vs fragment mode, color vars, heading
  scale arithmetic)

### Layer 2 - Renderer token tests

**PDF renderer** - Require registered fonts. Construct an
ASTRenderer with a real style dict and feed it hand-built
mistune AST tokens. Assert on flowable types and key
properties - not pixel output.

- `test_renderer.py` - block-level rendering (paragraph,
  heading, code, list, table, blockquote, wording divs),
  inline rendering (emphasis, strong, codespan, link,
  strikethrough, ins/del), front-matter and TOC flowable
  generation, table word-wrap prevention (splitLongWords),
  min-word-width column floors, font-size shrink-retry

**HTML renderer** - No fonts or ReportLab needed. Construct
an HTMLRenderer and feed it the same token shapes.

- `test_html_renderer.py` - block tokens (paragraph, heading,
  code, list, table, blockquote, wording divs, image, block
  html), inline tokens (emphasis, strong, codespan, link,
  strikethrough, softbreak, linebreak, inline html), front
  matter, TOC, URI scheme restriction, wording div wrapping

### Layer 3 - Builder integration tests

**PDF builder** - Full `build_pdf` pipeline. Each test loads a
small markdown fixture from `tests/fixtures/`, runs the
pipeline, and verifies the output PDF exists and has a valid
header.

- `test_builder.py` - minimal, front-matter, headings, code,
  wording, table fixtures; error cases (bad page_size);
  output directory creation; TOC toggle

**HTML builder** - Full `build_html` pipeline. Tests all four
mode combinations (full/fragment x linked/inline CSS) and the
standalone `build_css` path.

- `test_html_builder.py` - full-page output (doctype, stylesheet
  link, CSS file written), fragment output (no doctype, no html
  wrapper), inline CSS (style tag, no separate file), build_css
  standalone, output directory creation, TOC integration

### Layer 4 - Style and catalog tests

Verify the style system and JSON catalog API against the
real `styles/` directory.

- `test_catalog.py` - list_styles shape and content,
  list_images, load_style key completeness, palette
  resolution, inheritance, circular inheritance detection

## Layer 5 - Visual Regression (future)

Not yet implemented. The approach when ready:

1. Render `test-kitchen-sink.md` to PDF using `build_pdf`
2. Rasterize each page to PNG (via `pdf2image` or
   `PyMuPDF`/`fitz`)
3. Compare against checked-in baseline images using a
   perceptual diff tool (e.g. `pixelmatch` thresholds)
4. Fail the test if any page exceeds the diff threshold

Dependencies needed: `pdf2image` or `PyMuPDF` for
rasterization, plus a diff library.

Trade-off: baseline images break on any intentional visual
change (font tweaks, spacing adjustments, color updates).
Maintenance cost is high until the feature set and visual
design stabilize. Best introduced after the style system
is mature and changes are infrequent.

The existing `test-kitchen-sink.md` already serves as a
manual visual regression fixture - render it and inspect
the PDF by eye after significant changes.
