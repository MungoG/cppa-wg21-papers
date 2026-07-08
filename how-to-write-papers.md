# How to Write a Paper

<!--
When this file is mentioned or loaded, adopt it as system context in full.
Follow its rules while drafting or revising a WG21 paper. Do not summarize
it or discuss it abstractly. Operate from it.
-->

You are drafting a WG21 paper for a delegate audience. Write for a delegate who reads in passes and stops when a pass fails: show, then assert.

## The Delegate

The delegate has two hundred papers in the mailing and reads yours in up to three passes, stopping the moment a pass fails. The first pass takes 5 minutes and touches only the surface: title, abstract, headings, conclusion. The second pass takes up to 1 hour and follows the body: sections, figures, code. The third pass is hostile: the delegate challenges every assumption and hunts for what the paper omits. Most delegates make only the first pass, so the surface decides whether anything else is read.

Terms used throughout: the "contribution" is what the paper provides, not what it discusses; the "ask" is the specific request an ask-paper (front matter `intent: ask`) makes of its audience; the "surface" is the title, abstract, headings, and conclusion together.

Stage the rules: apply rules 1-6 when writing the surface, rules 7-16 when writing each body section, and rules 17-19 before calling the draft done. After drafting, apply all 19 again as audit criteria, one at a time. The rules are staged, not simultaneous.

## The Surface (rules 1-6)

Rules 1-6 govern what the first pass touches.

**1. Title.** Name the contribution in the title, not the topic area. A delegate classifies the paper from the title alone: proposal, analysis, experience report, or position.

**2. Abstract.** Open with the brutal summary: 1 sentence on its own line, no citations, no paper numbers, no hedging (source/CLAUDE.md rule 26; for ask-papers, "This paper asks [request]" satisfies the line). After a blank line, funnel in over 3-6 sentences: 1 sentence of context the audience already shares, 1-2 sentences narrowing to the specific problem, 1 sentence naming the contribution, and, for ask-papers, 1 sentence stating the ask. Each sentence narrows the scope of the one before it; the funnel does not open with code and does not jump from context to contribution.

WRONG:

```
co_await f() suspends the caller and resumes it later. The resumption
thread is unspecified. We propose executor affinity.
```

RIGHT:

```
The standard does not say which thread a suspended coroutine resumes on.

Asio, cppcoro, folly, and libunifex all ship coroutine task types. Each
one decides where a suspended coroutine resumes, and because the
standard is silent, each invents its own incompatible rule. This paper
proposes executor affinity for coroutine task types and asks LEWG to
forward it to LWG.
```

**3. Introduction.** Name the related work and enumerate the contributions as a numbered list in the introduction. State the paper's assumptions in the introduction, where the first pass can see them (an assumption the delegate cannot find reads the same as one that is invalid).

**4. Headings.** Write headings that carry the argument when read alone, in sequence, with everything between them removed. If a heading names a topic ("Background") and its section argues a point, rename the heading to state the point ("Detached Execution Fails Under Load").

**5. Conclusion.** Restate the contribution as the body's evidence refined it, in words that do not repeat the abstract. State what C++ gains if the ask is granted and what it keeps paying if it is not. Name who builds on the work next. Restate the ask so a delegate who reads only the conclusion can vote. Widen with these named consequences, not with slogans (widening without content is drift, and drift reads as marketing).

**6. Surface last.** After the body is complete, rewrite the surface against the finished body: re-derive the abstract from what the paper now shows, re-check that the headings still carry the argument, and rewrite the conclusion from the body's actual evidence (the surface sells the paper that exists, not the paper that was planned).

## The Argument (rules 7-16)

Rules 7-16 govern every body section, not only the first.

**7. Section preparation.** Open every section and subsection with 1-3 sentences stating what it covers and why the delegate needs it to follow the argument; put code, definitions, and technical material after those sentences, not first. When a section contains 2 or more subsections, add a map between the section heading and the first subsection heading: what the subsections cover and how they relate.

**8. Topic sentences.** Begin every paragraph with a topic sentence that advances the argument. Read in sequence, the topic sentences reproduce the paper's argument (the note-taking delegate harvests them as the summary).

**9. Self-containment.** Give every domain term 1 sentence of context before it carries weight in the argument. Give every reference the argument leans on a 1-sentence inline summary of its takeaway. Add a glossary section when the paper introduces 5 or more new terms. Call each concept by exactly one name for the whole paper; when two similar terms name genuinely different things, state the distinction where the second term first appears. The test: a delegate follows the entire argument without opening another document.

**10. Evidence before evaluation.** Place evidence (code, data, measurements, enumeration) before the value word it supports, in the same or the preceding paragraph. Never write a value word ("simpler", "significant", "elegant") whose evidence has not yet appeared; move the evidence ahead of the word or delete the word. (Evaluation read before evidence frames the evidence; evaluation without evidence is a bare assertion.)

**11. Structural claims.** Follow every claim of minimality, completeness, necessity, or exclusivity ("minimal", "complete", "necessary", "the only way") with what breaks when the thing is removed or why no alternative achieves the property. Delete the claim when the justification does not exist.

**12. Enumerate or delete.** Replace every vague quantifier ("some", "many", "various", "several", "often", "widely") with the actual items or the actual count. Delete the claim when the items cannot be named.

**13. Declarative register.** Write every sentence as an assertion of fact, evidence, or argument. Convert every rhetorical question into the statement it implies. Replace every slogan with the enumeration it compresses.

WRONG:

```
But on which thread? Under whose control? Small protocol, big rewards.
```

RIGHT:

```
The specification does not determine which thread the coroutine resumes
on or which component controls resumption. The protocol contains three
concepts, and each one removes a class of framework-specific workaround.
```

**14. Flow.** Connect consecutive ideas with transitions; give no paragraph a sentence fragment as its opener. Expand any passage a first-time delegate would re-read to parse (terse is acceptable; incomprehensible is not). Repeat a phrase of 4 or more words verbatim across sections only when each repetition adds meaning the prior instance did not carry; otherwise reword or cut.

**15. Code and figure context.** Precede every code block with 1-3 sentences stating its provenance (where the code comes from), its status (proposed, existing, or hypothetical), and its purpose (why the delegate is seeing it). Caption every figure with the point it demonstrates, label every axis with units, and attach error bars wherever a conclusion depends on the difference shown (delegates read sloppy graphs as sloppy work).

**16. Generalist calibration.** When the front matter `audience:` names LEWG, EWG, or WG21, write the problem statement, contribution, and conclusion so a competent C++ programmer with no domain expertise follows them. When the audience is a single study group (for example SG1), assume that group's domain expertise and apply rule 9 unchanged.

## The Audit (rules 17-19)

Rules 17-19 anticipate the hostile third pass: the delegate who challenges every assumption and hunts for what the paper omits.

**17. Disclose first.** State every assumption and every limitation in the paper before a delegate can discover it independently (a disclosed limitation is a scoping decision; a discovered one is a credibility failure that spreads to every other claim). An objection answered in the text is a finding that does not land.

**18. Checkable detail.** Provide enough detail for a delegate to check the work: implementation experience with a link, measurements with their setup, and every considered alternative with the reason it was rejected. Work that cannot be checked from the text reads the same as wrong.

**19. Cite fully, summarize inline.** Attach a citation to every claim that rests on prior work: prior papers in the series, poll history, and the references a survey of the area would surface. Never delete a citation to reduce delegate effort; add the 1-sentence inline summary instead (the citation exists for verification, the summary for comprehension). Citation format follows source/CLAUDE.md rules 27-37.

WRONG:

```
See P4172R0 for the design rationale.
```

RIGHT:

```
[P4172R0](https://www.open-std.org/...)<sup>[3]</sup> concluded that
awaitables compose with senders without an adapter layer; this paper
builds on that conclusion.
```

## Self-Scan

Run these 3 checks on the finished draft. Each answers yes or no; each no returns to the named rules.

1. **Surface check.** Reading only the title, abstract, headings, and conclusion, name the paper's category, context, assumptions, contribution, and ask. A missing answer returns to rules 1-6.
2. **Thrust check.** Read the topic sentences in sequence and confirm they reproduce the argument with its evidence. A gap returns to rules 8 and 10.
3. **Hostile check.** List the paper's weaknesses as an opponent would state them. Every weakness the text does not already acknowledge returns to rule 17.

Human colleague tests remain the pre-submission standard: a five-minute reader and a non-specialist reader catch what self-scanning cannot.

## Scope

These rules govern every paper drafted in `source/` and apply to every section of a paper, not only the first. Formatting, front matter, citation format, and wording-div mechanics live in `source/CLAUDE.md`; when that file and this one conflict, `source/CLAUDE.md` wins. Mechanical verification belongs to code, not prose: run `cite/cite.py --fix` for citations and invoke the Auditor (`situation-room/tools/auditor.md`) for structural checks.

When a rule cannot be satisfied truthfully - no implementation experience exists, no poll history is on record - never fabricate the missing evidence; state the gap in the paper in 1 sentence and continue.

## References

The delegate model is Keshav's three-pass reading method. These sources hold the evidence behind the rules.

1. S. Keshav, "How to Read a Paper," ACM SIGCOMM Computer Communication Review, 37(3), 2007 - the three-pass model these rules invert. Formatted copy: [how-to-read-papers.md](how-to-read-papers.md).
2. S. Peyton Jones, "Research Skills" - writing alongside the rest of the research craft. [http://research.microsoft.com/~simonpj/Papers/giving-a-talk/giving-a-talk.htm](http://research.microsoft.com/~simonpj/Papers/giving-a-talk/giving-a-talk.htm)
3. H. Schulzrinne, "Writing Technical Articles" - structure, style, and prose mechanics. [http://www.cs.columbia.edu/~hgs/etc/writing-style.html](http://www.cs.columbia.edu/~hgs/etc/writing-style.html)
4. G.M. Whitesides, "Whitesides' Group: Writing a Paper" - building a paper from its outline. [http://www.che.iitm.ac.in/misc/dd/writepaper.pdf](http://www.che.iitm.ac.in/misc/dd/writepaper.pdf)
5. T. Roscoe, "Writing Reviews for Systems Conferences" - the hostile reviewer's playbook. [http://people.inf.ethz.ch/troscoe/pubs/review-writing.pdf](http://people.inf.ethz.ch/troscoe/pubs/review-writing.pdf)

---

Written against Fable 5 / Opus 4.8 era guidance (2026-07). Re-audit on model upgrade; delete rules the model no longer violates before adding new ones.

Write for a delegate who reads in passes and stops when a pass fails: show, then assert.
