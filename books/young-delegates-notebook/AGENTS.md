# Agent instructions: A Young Delegate's Notebook

This repo holds one product: A Young Delegate's Notebook. It walks a reader from "I've never heard of WG21" to "I'm attending meetings and finding my niche." Each chapter is a stable plateau, a resting point where someone could stop and still be contributing. The guide accumulates: each chapter builds on the one before it.

The full plan, with the complete topic list, pedagogy rules, and writing algorithm, lives at `.cursor/plans/young_delegates_notebook_acddb630.plan.md`. That plan is the source of truth. This file is the short version.

## What this book is

- A practical navigation guide for newcomers. Dead simple, accumulative, builds like a pyramid.
- Opinionated. It has a position: stability over innovation, users first, evidence over enthusiasm.
- Voiced by the Patron, an unnamed experienced delegate speaking straight to the reader. The Patron is never named in the text.

## What this book is not

- Not institutional analysis. That is the job of `my-books/wg21-bible/`.
- Not a reform manifesto. That is the job of the Reform Codex.
- Not a textbook, manual, or encyclopedia.
- It does not use coined terms from the Bible. No Consensus Ratchet, no Peerage, no Empty Seat, no Silence As Consensus.
- It does not psychoanalyze the committee. Advice is addressed to the reader about their own judgment, in plain words, not jargon.

## Structure

- `young-delegates-notebook.md` - the assembled manuscript. Generated. Do not edit by hand.
- `chapters/intro.md` - the introduction. Sets the voice for the whole book.
- `chapters/ch-01.md` through `ch-13.md` - the thirteen chapters.
- `build.py` - the assembler. Run `python build.py` from this directory to regenerate the manuscript and its table of contents.

Edit chapter files. Rebuild. Never edit the assembled manuscript directly.

## Section numbering

- Decimal hierarchy. Chapters are `1`, `2`. Sections are `1.1`, `1.2`. Subsections are `1.2.1`.
- Chapter headings are `## 1. Title` (level 2). Section headings are `### 1.1 Title` (level 3). Subsection headings are `#### 1.1.1 Title` (level 4).
- Headings carry their full number. Headings do not use the section symbol.
- In prose, section references use the section symbol: `§2.4`, `§5.2.1`.

## Linking and citation

- No citations, footnotes, endnotes, or bibliography. Every reference is an inline hyperlink.
- Dense inline links. The text should read like a well-linked wiki, not an academic paper.
- Every paper number is a live link via wg21.link: `[P1234R0](https://wg21.link/p1234r0)`. No bare paper numbers.
- Every named document, site, or resource is linked on first mention in each chapter file. Later mentions in the same file use the plain name.

## Voice

- Patient colleague sitting next to the reader. Not above the reader.
- Contractions. Informal. Address the reader as "you" always.
- Conclusion first per subsection. State the destination, then build toward it.
- Average sentence 15 words, hard cap 25. Paragraphs 3 sentences max.
- No em dashes, no double dashes, no semicolons. Single dash only.
- Banned words include delve, tapestry, landscape, ecosystem, realm, robust, leverage, utilize, facilitate, navigate, streamline. The full draft rules are in the plan.

## The Delegate's Oath

Exact wording, do not paraphrase:

> I vow to do what is best for the language, to make no unnecessary proposals, and to put the needs of sixteen million users ahead of my own.

The Oath belongs to Chapter 1. The introduction carries its spirit ("put the users first") but does not state the formal Oath.

## Reform Codex boundaries

The Notebook draws principles from the Reform Codex but never its partisan or operational content. Use Sections 1-4 (the weight of the standard, paper writing, voting, delegate behavior). Do not use Sections 5-9 (reform communication, institutional diagnosis, structural remedies, NB strategy, counter-tactics). The reader should come away thinking "good principles for responsible participation," not "recruited into a faction."
