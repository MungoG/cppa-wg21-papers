# WG21 Paper Style Rules

Rules for all work on WG21 papers in this repository.

## Formatting

1. Markdown files must be ASCII only - no Unicode characters
2. Diacritics in personal names must never be omitted -
   represent them using HTML character references. Prefer
   named character references (`&uuml;`) over numeric
   character references (`&#252;`),
   e.g. `Dietmar K&uuml;hl`, `Micha&lstrok; Dominiak`
3. Single dashes (`-`) for all dashes - no em-dashes, no double dashes
4. Human-written tone - not wordy, no AI-characteristic phrasing
5. No git operations - user handles all git manually
6. The `archive/` folder is restricted for human use only - do
   not read, write, or modify any file in that folder
7. Padded tables with aligned pipe characters in every table on every rewrite
8. WG21-appropriate style
9. `[P####R#](https://wg21.link/p####r#)` link convention for all paper references
10. Paper titles on first mention
11. Backticks for code identifiers in headings

## Front Matter

12. Every paper begins with a valid YAML front matter block
    delimited by `---`. Required and optional fields:

    ```yaml
    ---
    title: "Paper Title"
    document: D0000R0
    date: YYYY-MM-DD
    reply-to:
      - "Author Name <author@example.com>"
    audience: SG1, LEWG
    ---
    ```

    - `title` - required, double-quoted string
    - `document` - required, unquoted paper number
      (e.g. D4007R0)
    - `date` - required, unquoted ISO 8601 date
    - `reply-to` - required, YAML list of double-quoted
      strings in the form `"Name <email>"`
    - `audience` - required, unquoted comma-separated
      target audience (e.g. SG1, LEWG)
13. Do not add front matter fields that the Pandoc template
    (`tools/wg21.html5`) does not consume

## Language and Grammar

14. American English unless otherwise specified
15. Follow the Chicago Manual of Style unless otherwise specified
16. Use the Oxford comma
17. Avoid dangling "This" - every sentence beginning with "This" must
    have a clear, unambiguous antecedent. If "This" could refer to
    more than one thing, replace it with a specific noun phrase
18. Do not end sentences with a preposition in formal papers -
    e.g. "worth building upon" not "worth building on"
19. "bound" = attached/tied to; "bounded" = limited -
    e.g. "an allocator bound to that tenant's quota"
20. Possessive before a gerund -
    e.g. "coroutines' interacting" not "coroutines interacting"
21. "may" = permitted/possible; "might" = hypothetical/enabled but
    not currently supported. Use precisely
22. Prefer "deeper" or "more thorough" over "full" for appendix
    cross-references. Understatement is better than hyperbole
23. Headings must not contain a dangling "This" - replace with a
    specific noun phrase. e.g. "ABI Lock-In Makes the Architectural
    Gap Permanent" not "ABI Lock-In Makes This Permanent"
24. Transitive verbs must not leave their object implicit when the
    referent is not immediately obvious. Supply the object explicitly.
    e.g. "forgetting to forward the allocator" not "forgetting to
    forward"
25. Avoid ambiguous prepositional attachment. When a prepositional
    phrase could modify either a nearby noun or the main verb,
    restructure to make the attachment clear.
    e.g. "The allocator - along with its state - must be
    discoverable" not "The allocator - with its state - must be
    discoverable" (where "with its state" could attach to
    "discoverable")
26. Place "only" immediately before the word, phrase, or clause it
    modifies to avoid ambiguity.
    e.g. "provides only one mechanism" not "only provides one
    mechanism"
27. Do not use contractions (it's, they're, don't, etc.) in formal
    papers. Use the expanded form

## Code Examples

28. Consistent comment style within a code block - either all
    comments are sentences (capitalized, with terminating punctuation)
    or all are fragments (lowercase, no period). Do not mix
29. Consistent capitalization in groups of code comments that
    summarize arithmetic or data
30. Align trailing comment columns when consecutive lines have
    trailing comments

## Structure

31. Sections with subsections should have at least one introductory
    sentence between the section heading and the first subsection
    heading - do not leave an empty section heading
32. Every paper must contain a `## Revision History` section,
    placed immediately after the Abstract's trailing `---` and
    before Section 1. Each revision is an H3 subheading in the
    form `### R<n>: <Month> <Year> (<pre-Meeting mailing>)`
    followed by a bullet list of changes
33. References and citations - for readers of printed copies:
    - Every hyperlink in the document body must also appear
      in the References section
    - Every hyperlink in the document body must have a
      superscripted citation marker giving the reference
      number, e.g. `<sup>[7]</sup>`
    - Every entry in the References section must include a
      readable URL (not hidden behind markdown link text)
    - Group related links under a single reference entry -
      e.g. one "C++ Working Draft" entry for all
      `eel.is/c++draft/` sections, one entry for the NB
      ballot repository covering individual issue links
    - Use `wg21.link` short URLs for WG21 papers - do not
      use `open-std.org` long-form URLs (see rule 9)
    - Links inside markdown tables are exempt from citation
      markers (tables are dense reference lists where
      superscripts would harm readability)
    - Links inside the Acknowledgements section are exempt
      from citation markers
    - Superscripted citation numbers in the body must always
      match the corresponding entry number in the References
      section - update all citations when references are
      added, removed, or renumbered

## Tone

34. Do not present options as predetermined conclusions. When
    recommending alternatives to a committee, present them as options
    to contemplate, not dictated outcomes
35. Avoid politically charged comparisons - do not invoke other
    contentious features as analogies unless the comparison is
    structurally precise. If the structures being compared are
    fundamentally different, the analogy will be perceived as
    political
