# Scrivener Gap Report

Remaining gaps between scrivener (ReportLab + mistune) and
paperkiller (Pandoc + Chromium). Items already implemented in
scrivener are excluded.

## Rendering Pipeline

| | Paperkiller | Scrivener |
|---|---|---|
| Parser | Pandoc (full CommonMark + extensions) | Mistune v3 (AST, strikethrough + table plugins) |
| Layout engine | Chromium print-to-PDF | ReportLab platypus |
| Mermaid | mermaid-filter (official CLI, full syntax) | merm (pure Python, limited syntax) with mermaido fallback |
| Deployment | Docker microservice (Flask + REST API) | Standalone Python script |

## Gaps

| # | Feature | Priority | Notes |
|---|---------|----------|-------|
| 1 | Mermaid full syntax | Medium | `merm` fails on chained arrows and named task IDs. Workaround: rewrite Mermaid source to use explicit edge pairs. Paperkiller uses official mermaid-cli with full syntax. |
| 2 | PDF bookmarks/outline | Medium | Neither tool creates a bookmark tree from headings. ReportLab supports this via `bookmarkPage` + `addOutlineEntry`. |
| 3 | Footnotes | Medium | Pandoc handles footnotes natively. Mistune has no footnote plugin. Rendering footnotes at page bottom in ReportLab is non-trivial. |
| 4 | Figure captions | Medium | Paperkiller supports `<figure>` + `<figcaption>`. Scrivener inserts block images with no caption. |
| 5 | Running headers | Low | Paperkiller has "WG21 Proposal" static strip. Scrivener has page numbers only. Adding text to `PageChrome` header bar would be straightforward. |
| 6 | Classification watermark | Low | Paperkiller has diagonal watermark and colored header/footer bars for C/S/TS. Not needed for current use cases. |
| 7 | Definition lists | Low | Pandoc supports natively. Mistune has no plugin. Rarely used in WG21 papers. |
| 8 | Inline images | Low | Scrivener renders alt text only for inline images. Block images work. |
| 9 | Raw HTML rendering | Low | Scrivener escapes raw HTML into text. Paperkiller renders it via Chromium. |
| 10 | Math/LaTeX | Low | Neither tool has it wired. ReportLab would need an external renderer. |
| 11 | Citations/bibliography | Low | Pandoc has `--citeproc`. ReportLab would need a full citation engine. |
| 12 | Cross-references | Low | Neither tool has explicit anchor/ref support beyond heading links. |
| 13 | Task lists | Low | Neither tool supports `- [ ]` / `- [x]` checkboxes. |

## Scrivener Advantages

Features scrivener has that paperkiller does not:

- Style cascade with `inherits:` and deep merge
- Logical font manifest with on-demand download
- Variable font axis instantiation (arbitrary width/weight)
- JSON style catalog API (`--list-styles`) with images array
- CLI option overrides with schema validation
- Smart table column sizing
- Proportional spacing system (`sp()`)
- Shared image asset directory
- Standalone script (no Docker/Node/Chromium dependency)
- Per-character CJK cmap fallback detection
- Configurable front matter fields per style
- GitHub-style blockquote callouts (`[!NOTE]`, `[!WARNING]`, `[!CAUTION]`)
- Heading widow/orphan control with keepWithNext propagation
- Inline code rounded background rectangles

## Summary

The highest-impact gaps are Mermaid full syntax (1), PDF bookmarks (2),
footnotes (3), and figure captions (4). Mermaid is partially working
but limited by the pure-Python `merm` parser. Bookmarks would make
long papers navigable in PDF readers. Footnotes are blocked on a
mistune plugin plus non-trivial page-bottom placement in ReportLab.
Everything else is low priority.
