# How to Write a Paper

*The complement of [how-to-read-papers.md](how-to-read-papers.md): how to write for the reader described there.*

## Abstract

Most advice on writing papers treats the reader as an attentive stranger who starts at the title and reads to the end. Real readers do not work this way. They read in up to three passes, most make only one, and they decide within minutes whether your paper deserves more of their time. This article inverts the three-pass reading method into a writing method: treat each pass as a set of requirements, and write the paper so that every pass succeeds on its own terms.

- **Keywords:** Paper, Writing, Hints

## 1. Introduction

A paper is written once and read many times, so the economics favor the reader. The reading side of the transaction has been described precisely: Keshav's three-pass method [1] tells readers to make a five-to-ten minute scan first, a careful pass of up to an hour second, and a full re-implementation pass third, stopping as soon as a pass tells them they can. Reviewers, colleagues, and citation-hunting strangers all read approximately this way, whether or not they ever learned it from a paper.

This is good news for the writer. If you know the algorithm your readers run, you can optimize against it. Each pass examines specific parts of the paper, in a specific order, looking for answers to specific questions. Those parts, that order, and those questions are your requirements.

The uncomfortable corollary: the default outcome of a first pass is that the reader stops. A paper that saves its virtues for section 4 will never have section 4 read. This article walks through the three passes from the writer's side and describes what each one demands.

## 2. Writing for the Three Passes

The reading method is cumulative: each pass builds on the previous one. The writing method is therefore layered: the paper must be complete at three different depths. Pass one is the paper in miniature - title, abstract, headings, conclusion. Pass two is the argument - figures and key points. Pass three is the evidence - assumptions, details, and everything a skeptical expert needs to re-create the work. A well-written paper is three nested papers, each self-sufficient at its own resolution.

### 2.1 Writing for the First Pass

In five to ten minutes, the first-pass reader reads your title, abstract, and introduction, scans your section headings, reads your conclusion, and glances over your references. Everything else is invisible. Then they answer five questions and decide whether you get another hour of their life. Each element the first pass touches has a job:

1. **Title:** State the contribution, not the topic area. A reader should be able to guess the category of the paper - measurement, analysis, prototype, position - from the title alone.
2. **Abstract:** Write a complete miniature of the paper: the problem, the approach, the result, and why it matters. The abstract is not a trailer that withholds the ending; the first pass never reaches the ending anywhere else.
3. **Introduction:** Establish context and state the contributions explicitly, preferably as an enumerated list. The first-pass reader is asking which work this relates to and what it claims; answer both in so many words.
4. **Headings:** Choose section and sub-section titles that read as a coherent outline on their own. The reader will read them as one, with everything in between removed. If the headings do not tell the story, the story does not get told.
5. **Conclusion:** Restate the contributions concretely, with the results attached. The conclusion is read minutes after the abstract; it should confirm and sharpen, not repeat or drift.
6. **References:** Cite the works your readers already know. The reader ticks off the references they have read, and each tick tells them where you sit on the citation graph and whether you know the field.

At the end of the first pass the reader scores you on the five Cs. Restated as targets for the writer:

1. **Category:** Make the type of paper unmistakable from the title and abstract. A reader who cannot classify your paper cannot evaluate it.
2. **Context:** Name the related work and the theoretical bases yourself, early. Do not make the reader reconstruct where the paper sits.
3. **Correctness:** Surface your assumptions where the first pass can see them. An assumption the reader cannot find reads the same as one that is invalid.
4. **Contributions:** Enumerate them. Contributions the reader must infer are contributions you do not get credit for.
5. **Clarity:** Treat writing quality as a gate, not a polish step. Most reviewers make only one pass, and if the gist does not survive it, the paper is rejected.

The first pass is where papers die. If a reviewer cannot understand the gist after one pass, the paper will likely be rejected; if a reader cannot see the highlights in five minutes, the paper will never be read. Spend a disproportionate share of your writing time on the surface the first pass touches: it is a small fraction of the text, and it decides everything.

### 2.2 Writing for the Second Pass

The second-pass reader gives you up to an hour. They read with care but skip the proofs, they study your figures, and they jot down key points as they go. Their goal is to be able to summarize the main thrust of the paper, with supporting evidence, to someone else. Your job is to make that summary easy to construct and hard to get wrong.

1. **Figures carry the argument.** The second-pass reader examines figures, diagrams, and graphs with special attention, often before the surrounding prose. Make each one self-contained: a caption that states what the figure shows and why it matters, axes properly labeled with units, and error bars wherever a conclusion depends on statistical significance. Readers are told outright to use sloppy graphs to separate rushed, shoddy work from the truly excellent; a mislabeled axis costs credibility far beyond the figure it appears in.
2. **Topic sentences are the summary.** A reader jotting key points harvests the first sentence of each paragraph. Write topic sentences so that, read in sequence, they reproduce the argument of the paper. If the margin notes come out wrong, the writing failed before the reading did.
3. **Curate the references.** The second-pass reader marks relevant unread references for further reading, using them to learn the background of the paper. Cite the background a newcomer actually needs, so the follow-up reading you trigger builds your case.

The reading method also lists the reasons a second pass fails: terminology and acronyms the reader does not know, a proof or experimental technique they cannot follow, unsubstantiated assertions, and numerous forward references. Every item on that list is a writing defect before it is a reading problem. Define terms on first use, expand acronyms, introduce techniques before relying on them, support assertions when you make them, and structure the paper to be readable front to back. A reader who abandons the second pass sets your paper aside hoping never to need it, and most never return.

### 2.3 Writing for the Third Pass

The third-pass reader, typically a reviewer, tries to virtually re-implement your paper: making your assumptions, re-creating your work, and comparing the result against what you wrote. They challenge every assumption in every statement and hunt for hidden failings, implicit assumptions, missing citations, and problems with your experimental or analytical techniques. Write for this reader as if for an auditor.

1. **State every assumption explicitly.** The third pass exists to find the assumptions you did not state. An assumption you disclose is a scoping decision; an assumption the reader discovers is a hidden failing.
2. **Provide enough detail to re-create the work.** Parameters, configurations, datasets, derivation steps; where space does not permit, point to an appendix or an artifact. If the work cannot be re-created from the text, the third pass concludes the paper cannot be checked, which reads the same as wrong.
3. **Disclose your limitations yourself.** The third-pass reader will find them; the only question is whether they find them in your text or in their own notes. A disclosed limitation is a boundary. A discovered one is a credibility failure that spreads to every other claim in the paper.
4. **Cite everything relevant.** Missing citations to relevant work are on the third-pass reader's explicit checklist, and the one subject on which a reviewer is guaranteed to be expert is the literature.

The third pass takes an experienced reader about an hour, and that hour is spent probing exactly the items above. A paper that survives it does more than avoid rejection: it wins over the one reader who understood it completely.

## 3. Writing for the Literature Surveyor

Not every reader arrives through your abstract. The literature surveyor arrives sideways: they searched an academic search engine with well-chosen keywords, found a few recent papers, and read the related work sections first, looking for a thumbnail of the field, shared citations, and repeated author names. Their method makes three demands:

1. **Be findable.** The surveyor finds papers by keyword search. Put the words people would actually search for in the title and abstract - the established terms of the field, not private coinages. A paper that renames its own subject is invisible.
2. **Write the related work section as a service.** Surveyors read your related work looking for a thumbnail summary of recent work, and if you are lucky they treat it as a pointer to the field. Make it organized, fair, and current. A generous related work section is also self-interested: it is the part of your paper most likely to be read by people who have not yet decided to read your paper.
3. **Sit on the citation graph.** Surveyors identify the key papers and researchers by finding shared citations across bibliographies. Cite the key papers of your field, every time. A paper missing the citations everyone else shares falls off the graph the surveyor is walking.

## 4. A Checklist Before You Submit

Each pass suggests a test. Run all three before submitting.

1. **The five-minute test.** Give a colleague the title, abstract, introduction, headings, and conclusion for five minutes, then ask for the five Cs: what kind of paper it is, what it relates to, what it assumes, what it contributes, and whether it was readable. Wrong answers mark first-pass defects.
2. **The one-hour test.** Ask a colleague outside your specialty to read the paper, skipping the proofs, and then summarize its main thrust with supporting evidence. If the summary comes back wrong or thin, the figures and topic sentences are not carrying the argument.
3. **The re-implementation test.** Ask your most skeptical reader whether they could re-create the work from the text alone, and to list its weaknesses. Every weakness they find that the paper does not already acknowledge is a finding you have donated to a hostile reviewer.

The tests are cheap relative to a rejection: a few hours of colleagues' time against months of resubmission delay.

## 5. Related Work

The reading method this article inverts is S. Keshav's "How to Read a Paper" [1]; read it first, since it is the specification this article compiles against. For the mechanics of structure, style, and prose, Henning Schulzrinne's guide to writing technical articles [3] and George Whitesides's overview of building a paper from its outline [4] are complementary. Simon Peyton Jones's materials on research skills [2] cover writing alongside the rest of the craft. And to understand the reviewer you are writing for, read Timothy Roscoe's "Writing Reviews for Systems Conferences" [5]: it is the other side's playbook.

## 6. References

1. S. Keshav, "How to Read a Paper," ACM SIGCOMM Computer Communication Review, 37(3), 2007. Formatted copy: [how-to-read-papers.md](how-to-read-papers.md).
2. S. Peyton Jones, "Research Skills," [http://research.microsoft.com/~simonpj/Papers/giving-a-talk/giving-a-talk.htm](http://research.microsoft.com/~simonpj/Papers/giving-a-talk/giving-a-talk.htm).
3. H. Schulzrinne, "Writing Technical Articles," [http://www.cs.columbia.edu/~hgs/etc/writing-style.html](http://www.cs.columbia.edu/~hgs/etc/writing-style.html).
4. G.M. Whitesides, "Whitesides' Group: Writing a Paper," [http://www.che.iitm.ac.in/misc/dd/writepaper.pdf](http://www.che.iitm.ac.in/misc/dd/writepaper.pdf).
5. T. Roscoe, "Writing Reviews for Systems Conferences," [http://people.inf.ethz.ch/troscoe/pubs/review-writing.pdf](http://people.inf.ethz.ch/troscoe/pubs/review-writing.pdf).

---

*Companion to [how-to-read-papers.md](how-to-read-papers.md). The reading model is S. Keshav's three-pass method (ACM SIGCOMM Computer Communication Review, 2007).*
