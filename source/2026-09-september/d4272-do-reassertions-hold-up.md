---
title: "Assessing P3846R1 Against the Public Record: Eighteen Responses on C++26 Contract Assertions"
document: P4272R0
date: 2026-09-03
intent: info
audience: EWG
reply-to:
  - "Vinnie Falco <vinnie.falco@gmail.com>"
---

## Abstract

The eighteen responses defending C++26 contract assertions resolve one of the eighteen objections.

P3846R1 answers eighteen concerns raised about P2900, the contract assertions facility adopted into the C++26 working draft. Assessed against the public record, on a five-value support rating and a three-value resolution criterion defined in this paper and applied by its author, its responses come out in their authors' favor on support, with two supported, six substantially supported, ten mixed, and none unsupported or contradicted. Fourteen of the eighteen nonetheless contain a material assertion the record does not support, and three carry a subclaim the record contradicts. On resolution the record comes out for the objecting position, with one objection answered, sixteen partly answered, and one not resolved. Because P3846R1 presents the objections as previously addressed, the objecting position prevails on closure, and seventeen objections remain open for any future contracts design.

---

## Revision History

### R0: September 2026

- Initial version.

---

## 1. Introduction

This paper assesses the eighteen responses in P3846R1<sup>[1]</sup> against the public record. P3846R1, "C++26 Contract Assertions, Reasserted", answers eighteen concerns raised about P2900<sup>[2]</sup>, the contract assertions facility adopted into the C++26 working draft at the February 2025 Hagenberg meeting. Each response is tested against the public sources available at two dates: 2025-11-03, the date P3846R1 prints on its title page, and 2026-03-23, the date the authors supplied the artifact that entered the April 2026 mailing. For every concern the assessment records three independent fields: whether the record supports the complete response, whether the response contains a material unsupported subclaim, and whether the response resolves the original objection on its own terms. The paper proposes no wording and requests no poll.

The concerns under assessment come from the objecting papers P3835R0<sup>[3]</sup>, P3829R0<sup>[4]</sup>, P3849R0<sup>[5]</sup>, P3506R0<sup>[6]</sup>, and P3878R0<sup>[7]</sup>, and from national body comments on the C++26 draft whose dispositions the Kona November 2025 minutes record<sup>[8]</sup>. P3846R1 is the consolidated response by twenty-two authors, including two of the three authors of P2900R14, written for that meeting and published in the April 2026 mailing. Its abstract states that "Almost all objections are repetitions of those raised in earlier papers, addressed in subsequent responses, and extensively discussed in EWG" (p. 1)<sup>[1]</sup>. That characterization is what makes closure the operative question below, and the hedge "almost all" is P3846R1's own. The third P2900R14 author, Andrzej Krzemie&nacute;ski, is not among the twenty-two. His absence matters to this assessment because he wrote P3859R0<sup>[9]</sup> and P3896R0<sup>[10]</sup> and co-authored P1995R1<sup>[11]</sup>, all three of which enter at Concern 16. Section 3 records the two errors that a single-verdict rating produces: one incorrect sentence can fail an entire response, and a committee decision can be read as proof of a technical proposition. The three-field structure used here makes both errors structurally impossible.

P2900 provides contract assertions in declaration position - preconditions, postconditions, and `contract_assert` - together with four evaluation semantics selectable per translation unit and a global contract-violation handler invoked when a checked assertion fails. This combination enables both the traditional documentation use of assertions and checked enforcement under a single syntax, and its const-ification of predicate parameters enables the compiler to reject predicates that modify their arguments. P3846R1's structure provides one self-contained response per concern, with its reasoning, citations, and page references collected under a single heading, which enables each response to be checked against the record independently of the others.

The assessment provides four contributions:

1. A dual-cutoff temporal method (Section 2). Sources carrying a document date on or before 2025-11-03 are the test of whether the responses were accurate when written; sources dated between that day and 2026-03-23 are the test of whether the artifact was current when the authors supplied it; later evidence is held separately in Section 9 and never enters the primary score.

2. The separation of overall support from subclaim defects from objection resolution (Section 3). A single defective sentence changes only the subclaim flag, and a committee vote changes only the procedural record.

3. The eighteen-concern assessment itself (Sections 4 through 8): one four-part evidence unit per concern, stating the response, the evidence that supports it, the limitation under the same standard, and the rating.

4. Corrections to the public citation record. The 2016 lost-optimization report is LLVM issue 28170, renumbered from Bugzilla 27796 when the LLVM tracker migrated to GitHub<sup>[12]</sup>. The Kona minutes N5031 are a 2025 document authored by Nina Dinka Ranns<sup>[8]</sup>. P3626R0 is the alternative wording prepared by Timur Doumler, lead author of P2900R14 and P3846R1, so that the Evolution Working Group (EWG) could poll the alternative; it is not an independent rival proposal<sup>[13]</sup>. P3097R0 was merged into P2900R8, a revision of the proposal, never into the C++26 working draft, and the November 2024 Wroc&lstrok;aw meeting took place between the 2024 St. Louis poll and the Hagenberg poll<sup>[14]</sup>.

The assessment makes three assumptions. First, a paper's printed date fixes what its authors could have known: a source dated after 2025-11-03 cannot retroactively falsify a sentence written by that day, and a source available by that day needs no later corroboration. Second, a committee vote establishes procedural disposition without settling a technical question, and Section 3 applies this rule to favorable and unfavorable polls alike. Third, public sources are the only admissible evidence, so a claim resting on unpublished discussion is recorded as unsourced rather than as false.

The result is a dual-cutoff, dual-axis assessment that preserves the reasoning the record supports while identifying which objections remain open.

---

## 2. Two Cutoffs: Accuracy When Written, Currency When Supplied

This section fixes which public record each claim in P3846R1 is tested against, and states the source of the poll evidence used later in this review. A delegate who wants to check a date or a tally needs both before reading the summary table.

The audited artifact is the P3846R1 PDF published in the April 2026 WG21 mailing<sup>[1]</sup>: 494,439 bytes, SHA-256 `0cbbdc9c27987d5694b5d4f6d48d97c3244d8c40a547225ce79cf060bd46035c`, with the live open-std.org copy, an Internet Archive capture, and the copy quoted throughout this review all hashing to that value. The artifact prints "Date: 2025-11-03" on its title page and describes itself in its revision history as "Revision 1 (Update for Kona November 2025 Meeting)". Its PDF creation and modification metadata both read 2026-03-23T13:16:09Z. P3846R1's public tracking issue records "Authors provided updated version" ten minutes later the same day<sup>[15]</sup>, and the official WG21 2026 index dates the paper 2026-03-23 and places it in the April 2026 mailing<sup>[16]</sup>. The registry therefore contradicts the title page as a matter of public record: the artifact prints one date and is registered under another. The same registry row also gives the title as "C++26 Contracts, reasserted" and the author list as two names, where the artifact's own front matter carries "C++26 Contract Assertions, Reasserted" and 22 reply-to authors; the artifact's front matter governs.

The printed date, 2025-11-03, is the substantive cutoff: sources carrying a document date on or before that day are the test of whether the responses were accurate when written. The build date, 2026-03-23, is the publication-state cutoff: sources dated after 2025-11-03 and on or before 2026-03-23 are the test of whether the artifact placed before the committee was current when the authors supplied it. The axis keys to document date because document date and public availability diverge for P3846R1's own cited material. The revision history names seven papers as added discussion - P3853R0, P3878R0, P3889R0, P3893R0, P3896R0, P3909R0, and P3910R0 - and six of the seven were first mailed in December 2025, after the printed date, yet every one carries a document date on or before 2025-11-03, with P3909R0 dated exactly on the boundary<sup>[17]</sup>. P3893R0 is the clearest illustration: its source artifact and printed date are 2025-10-24, while its public tracking issue and its mailing are December 2025<sup>[18]</sup>. A rule keyed to public availability on 2025-11-03 would exclude P3846R1's own cited material from its own source set; a rule keyed to document date does not.

Evidence dated between the cutoffs can show that the March artifact was stale, incomplete, or left a claim unqualified. It does not make a November claim retroactively false, and it creates no obligation to cite a source that did not exist when the text was written. P3878R1 illustrates the boundary: printed 2025-11-06 and indexed 2025-11-08, it could not inform the November manuscript, but it had been public for over four months when the artifact was built<sup>[19]</sup>. Evidence dated after 2026-03-23 establishes later chronology or outcomes only; it is held separately in Section 9 and never enters the primary score.

No public artifact establishes that text byte-identical to the audited PDF circulated on 2025-11-03. The canonical 2025 directory contains no P3846R1, the 2025 index lists none, and the Internet Archive holds thirteen captures of P3846R0 spread across the whole window but no capture of any P3846R1 before June 2026<sup>[17]</sup>. The March copy named in the tracking issue cannot be byte-compared today, because its URL is access-gated and has never been archived; the hash identity stated above covers the April artifact. The two-cutoff treatment is a disclosed method of this review, adopted so that the responses are judged against the record available when the text was written as well as when the artifact was supplied.

Poll evidence comes from one primary location. No published WG21 minutes record subgroup straw-poll tallies: the minutes for St. Louis, Wroc&lstrok;aw, Hagenberg, and Kona record plenary motions and prose subgroup reports, and none contains an SF/F/N/A/SA tally<sup>[20]</sup><sup>[21]</sup><sup>[22]</sup><sup>[8]</sup>. The Hagenberg minutes instead direct readers to "the P2900 tracker" for the contracts polls<sup>[22]</sup>. The primary public record for every poll cited below is therefore the chair-posted comment thread on the public cplusplus/papers tracker<sup>[23]</sup>, and each tally is also corroborated by the proposal's rationale paper where that paper records the same poll<sup>[24]</sup>.

Later evidence can establish staleness or later outcomes, and it changes neither historical score.

---

## 3. Support, Subclaims, and Resolution Are Independent Judgments

This section defines the three fields scored for every concern and the evidence standard applied to both sides. The summary table in Section 4 is unreadable without them, because the fields answer different questions and are never combined.

Overall support is the rating of the complete P3846R1 response against the public record at the cutoffs of Section 2. It takes exactly one of five exclusive values.

**Table 1. Overall support ratings for a complete P3846R1 response. Each response receives exactly one rating. A material unsupported subclaim does not by itself change the rating; it is recorded in a separate field.**

| Rating | Meaning |
|---|---|
| Supported | The record supports the response's central claims and contains no equally material evidentiary weakness. |
| Substantially supported | The record supports the central response, while identified limitations do not create an equally strong contrary case. |
| Mixed | Material support and material weakness both affect the central response, so neither controls the complete answer. |
| Not supported | The response contains a recognizable argument, but the record does not support its central claim. |
| Contradicted | The best contemporaneous evidence defeats the response's central claim. |

The unsupported-subclaim flag is a separate yes/no field. It is Yes when the response contains a specific material assertion - factual, causal, quantitative, or historical - that lacks support or is false, regardless of the complete response's rating. A Supported response carries no such defect. A Substantially supported or Mixed response can carry one without the defect consuming the whole response.

Resolution status answers a third question: whether the response meets the original objection on its own terms. "Answered" means the supported response meets the stated objection. "Partly answered" means the response explains a tradeoff or documents a procedure while leaving a material technical issue open. "Not resolved" means the response does not meet the objection's central mechanism or evidence. A committee vote establishes procedural disposition; it does not by itself convert a technical status from Not resolved to Answered.

The fields are separate because combining them produces two characteristic errors. The first error is to treat one incorrect sentence as the failure of an entire response, erasing whatever else the response established. The second error is to treat a committee decision as proof of a technical proposition, converting a procedural outcome into a technical one. Three fields make the two errors structurally impossible: a single defective sentence changes only the subclaim flag, and a vote changes only the procedural record.

The evidence standard is symmetric, and it limits the counter-evidence presented below as much as it limits P3846R1. Expert testimony may support a qualitative judgment when it is public and attributed; it cannot support an unattributed quantitative comparison. A coding rule proves recognition of a risk, not frequency. A later signature on a proposal proves agreement at that date, not dependence of earlier implementation work. Affiliation affects weight only through a stated rule applied to the response papers and the objecting papers alike. This standard is the author's own construction, and the author has a material stake in the question under assessment: the author is a co-author of P4238R0<sup>[25]</sup>, whose position this paper's findings support, as the Disclosure records. If the standard is rejected, the eighteen quoted sentences and the cited artifacts stand independently of the rating scheme.

Support for a response and resolution of an objection are independent judgments, and they cannot share one category. Section 4 reports both, separately, for all eighteen concerns.

---

## 4. The Summary Table: Authors Favored on Support, the Objecting Position on Resolution

This section consolidates the eighteen assessments into one table so that the distribution is visible before the evidence. Sections 5 through 8 give the four-part evidence unit behind each row.

**Table 2. Assessment of the eighteen P3846R1 responses. Overall support rates the complete response against the public record at the cutoffs defined in Section 2. Unsupported subclaim records whether the response contains a material unsupported factual, causal, quantitative, or historical assertion. Resolution records whether the response meets the original objection on its own terms. The three fields are independent by construction (Section 3).**

| Concern | Overall support | Unsupported subclaim | Resolution |
|---|---|---|---|
| 1. Safety and non-ignorable checks | Mixed | Yes | Partly answered |
| 2. Cross-translation-unit semantics | Mixed | Yes | Partly answered |
| 3. Dependency management | Mixed | Yes | Partly answered |
| 4. One Definition Rule | Substantially supported | Yes | Partly answered |
| 5. Modules | Supported | No | Partly answered |
| 6. Implementation-defined behavior | Mixed | Yes | Partly answered |
| 7. Uncheckable guidance | Substantially supported | Yes | Partly answered |
| 8. Constification | Substantially supported | No | Partly answered |
| 9. Global violation handler | Substantially supported | Yes | Partly answered |
| 10. Consecutive assertions | Supported | No | Answered |
| 11. Predicate exceptions | Mixed | Yes | Not resolved |
| 12. Static analysis | Mixed | Yes | Partly answered |
| 13. Complexity | Mixed | Yes | Partly answered |
| 14. Missing features | Substantially supported | Yes | Partly answered |
| 15. Future features | Mixed | Yes | Partly answered |
| 16. Decomposition | Mixed | Yes | Partly answered |
| 17. Deployment experience | Substantially supported | Yes | Partly answered |
| 18. Library hardening | Mixed | No | Partly answered |

The counts are two Supported, six Substantially supported, ten Mixed, zero Not supported, and zero Contradicted. Fourteen responses carry the unsupported-subclaim flag. The resolution counts are one Answered, sixteen Partly answered, and one Not resolved. Converting them into a percentage of objections settled would misstate them: the support column holds ratings of responses rather than of objections, and the resolution column records an evaluative judgment against the stated criterion.

The two axes diverge. On overall support, the record comes out for the authors of P3846R1: eight responses receive favorable ratings, ten are Mixed, and no complete response is Not supported or Contradicted. On resolution, the record comes out for the objecting position: one objection is Answered, sixteen are Partly answered, and one is Not resolved.

---

## 5. Two Fully Supported Responses, One Resolved Objection

This section covers the two responses the public record fully supports, Concerns 5 and 10. A delegate needs them side by side because they agree on support and differ on resolution, which is the distinction the summary table rests on.

### Concern 5: Modules

Concern 5 is the objection that P2900 does not work well with modules. P3835R0<sup>[3]</sup> asks how modules relate to contract assertions and whether they could address the concerns about configuring contract-evaluation semantics - the per-translation-unit rule selecting whether an assertion is evaluated and what happens when it is false. P3846R1's response is a bounded architectural claim stated in bounded terms: "Modules are not a solution to every concern, but they do provide capabilities unavailable in a purely header-based model and, therefore, represent a distinct avenue for configuring contract-evaluation semantics" (p. 15). The Details subsection opens conditionally - "In principle, inline functions in a BMI could carry additional information, such as contract-evaluation semantics" (p. 16), a built module interface (BMI) being the compiled artifact produced from a module unit that importers consume - and the same paragraph limits the claim: "because a BMI can itself be compiled multiple times with different flags in the same build (often matching the flags used by the importer), modules remain only a partial solution to the broader problem of ensuring flag consistency across object files" (p. 16).

P3321R0<sup>[26]</sup>, which P3846R1 cites, discusses module translation units as one configuration strategy. In GCC, the contracts change to `gcc/cp/module.cc` adds 30 lines: it streams references to the outlined pre and post helper functions so that importers need not regenerate them, and it appends a single boolean element to the BMI dialect string recording that contracts were enabled<sup>[27]</sup>. That is an implementation of exactly the bounded kind of information-carrying the response describes. The content was public on the GCC development fork from 2025-10-19, before the substantive cutoff; the same change was merged into GCC master on 2026-01-28 as commit `64674a2`<sup>[28]</sup>, between the two cutoffs.

The limits of the demonstration are equally visible under the same standard. No evaluation semantic is written to or read from a BMI anywhere in the change: the dialect element is a compatibility marker, and there is no cross-module policy resolution of any kind<sup>[27]</sup>. No cutoff-era implementation demonstrates semantic configuration through module metadata, and P3835R0 describes the mixed-mode question - translation units compiled with different evaluation semantics linked into one program - as orthogonal to modules<sup>[3]</sup>. These limits confine the practical value of the claim. They do not contradict it, because the claim was made "in principle" and expressly called partial.

The response is Supported, with no material unsupported subclaim. The objection is Partly answered: the architectural avenue exists and is demonstrated at the level the response claims, while the practical question of who controls the semantic across module boundaries remains open.

### Concern 10: Consecutive assertions - the one objection answered

Concern 10 is the objection that observing consecutive contract assertions is dangerous, because an earlier assertion may be a precondition for safely evaluating a later one. P3846R1's response supplies a mechanism, a counterexample, and a committee record. The mechanism: "The idiomatic solution is to combine dependent predicates into a single assertion, thus avoiding the risk of evaluating the second condition after the first fails" (p. 23). The residual risk is stated in the same response: "continuing past a failed assertion always comes with a risk" (p. 23). The counterexample addresses the proposed alternative: "Proposals to automatically skip subsequent assertions after an observed contract violation are problematic because doing so could suppress an unrelated enforced check and result in a worse outcome" (p. 23).

Short-circuit conjunction in `pre(p && p->is_runnable())` prevents the dependent second predicate from being evaluated after the first fails, so the mechanism works for the canonical case. The alternative was considered and declined on the record. The Contracts Study Group (SG21) polled forwarding P3582R0<sup>[29]</sup>, the automatic-skipping proposal, to EWG for C++26 on 2025-02-06: SF 0, F 0, N 1, A 13, SA 7, official result "Consensus against"<sup>[30]</sup>, corroborated by the rationale paper<sup>[24]</sup>. The poll question was forwarding; P3582R0 itself states that no general method distinguishes related predicates from unrelated ones, which is the gap the counterexample exploits.

One clause in P3846R1's discussion cannot be independently verified and is attributed here to P3846R1 rather than to the public record. P3846R1 states that after SG21's discussion of P3582R0, no one, including that paper's author, was in favor of pursuing the mitigation. The tally entails the aggregate: with SF 0 and F 0, no attendee voted in favor. The individual attribution asserts more, and no public source records individual votes in any SG21 poll.

The mechanism's limits are real and stated in the response. Conjunction does not combine every dependent precondition cleanly, and the observe semantic retains the risk the response names; P3846R1 notes that vendor-specific analysis could implement other conforming strategies.

The response is Supported, with no material unsupported subclaim. The objection is Answered: the response meets the concern on its own terms, supplying a working mitigation for the canonical case, a counterexample against the proposed alternative, and a recorded committee decision declining that alternative.

Concern 5's claim is supported because it was stated conditionally, and its practical value remains untested; Concern 10's response is supported and complete.

---

## 6. Six Supported Central Claims, Each With a Material Limit

This section covers the six responses - Concerns 4, 7, 8, 9, 14, and 17 - whose central claims the public record supports, each with a material limit that preserves part of the original objection. A delegate needs the limits stated with the same precision as the support, because these are the rows where both are real.

### Concern 4: One Definition Rule

Concern 4 is the objection that P2900 violates the spirit of the One Definition Rule. P3846R1 classifies the reported failure as a general compiler defect. It states that the optimizations at issue are unsound across replaceable inline definitions and that "both Clang (LLVMPR26774) and GCC (GCCBug70018) disabled them nearly a decade ago" (p. 15). It characterizes the behavior reported in P3829R0<sup>[4]</sup> as "a regression of the same issue in GCC 14, entirely unrelated to contract assertions" (p. 15), states that "the incorrect behaviour caused by the GCC bug is not conforming" (p. 13), and records that during upstreaming of GCC's P2900 implementation the authors filed the bug report and reproducer themselves (p. 15). On performance, the response states: "Concerns about harming performance by disabling these optimisations are equally unfounded. The optimisations are suppressed only when an inline function is not inlined and is instead invoked indirectly, which already incurs overhead that likely dwarfs any benefits from such optimisations. Clang made this tradeoff long ago without user complaints." (p. 15).

Clang's fix for the defect recorded as LLVM issue 27148<sup>[31]</sup> was committed on 2016-04-08 under the title "Don't IPO over functions that can be de-refined", introducing a predicate marking definitions that may be replaced by differently optimized variants at link time and changing a set of interprocedural optimization passes to skip them<sup>[32]</sup>. GCC's fix for the same defect class was merged into trunk the same month and appeared in the GCC 7 release series<sup>[33]</sup>. Measured from the printed date, that is nine years and seven months, which "nearly a decade" fairly describes. The 2025 bug record supports the classification as general: its title concerns invalid optimization based on bodies of vague-linkage functions, its reproducer contains no contract assertion of any kind and does not use the contracts flags, and it was filed as an interprocedural-optimization regression<sup>[34]</sup>. Inside that bug, Richard Smith states in [comment 12](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=121936#c12): "LLVM has (to the best of my knowledge) fully addressed this and it doesn't seem to have been a problem at scale."<sup>[34]</sup> The classification and the compiler history are on the record.

The contrary evidence on performance sits inside a source the response cites. Jan Hubi&ccaron;ka, GCC's interprocedural optimization maintainer, wrote in [comment 23](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=121936#c23) of that bug, five weeks before the printed date, that "these optimisations may have a large performance impact", and he quantified one workload, reporting that the optimization "improved jpeg-xl encoding speed by 47%"<sup>[34]</sup>. Asked about the narrow case the response describes - the callee neither inlined nor cloned - he answered in [comment 26](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=121936#c26) that disabling the optimization there forgoes a "quite considerable amount of current and future optimization oppurtunities" (spelling as in the source)<sup>[34]</sup>. "Equally unfounded" is therefore contradicted by the cited record, not only unsupported. The symmetric standard of Section 3 requires the complementary disclosure: Hubicka's figure is a measure of the gain from the optimization in general, and no measurement of the narrow indirect-call case exists on either side. The adjacent sentence, "Clang made this tradeoff long ago without user complaints", carries no source, and a primary record in that class exists: thirty-nine days after the Clang fix, Warren Ristow, a Sony toolchain engineer and LLVM contributor, filed the report now numbered LLVM issue 28170, writing that the change was "causing a lost optimization in variadic cases, that I believe can still safely be optimized"<sup>[12]</sup>. His report was confined to variadics, stated that the suppression was unneeded "in this case" rather than in general, and was resolved in 2018 by teaching the inliner to handle variadic functions, leaving the conservatism standing<sup>[12]</sup>. The sentence is overstated rather than refuted.

Against the categorical wording: in [comment 9](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=121936#c9) of the same bug, Smith enumerates "contract checking mode" among the properties that can trigger the defect<sup>[34]</sup>, and the GCC contracts implementation carries a dedicated workaround for it, the `-fcontracts-conservative-ipa` option, enabled by default, that was public on the development fork from 2025-10-19<sup>[35]</sup> and was merged into GCC master on 2026-01-28<sup>[36]</sup>. The workaround shows that contracts exposed the defect and required operational mitigation; it does not show that P2900 specifies or causes the invalid optimization, and the commit message says so, calling the wrapper sufficient "while a suitable general fix is evaluated"<sup>[36]</sup>. In the response's favor on the central classification: the bug remained NEW at both cutoffs, and a GCC contributor, Matthias Kretz, later disputed that the optimization is nonconforming at all, in [comments 28](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=121936#c28) and [31](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=121936#c31) of 2026-02-13 and 2026-02-16, between the cutoffs<sup>[34]</sup>. Neither side can treat the bug as a settled miscompilation, and the response's flat "is not conforming" is contested in the bug's own thread.

The response is Substantially supported. The "equally unfounded" performance sentence is a material unsupported subclaim, contradicted by a source the response cites. The objection is Partly answered: the classification and the compiler history are supported, while the performance dismissal and the categorical "entirely unrelated" wording leave the operational core of the concern open.

### Concern 7: Uncheckable guidance

Concern 7 is the objection that P2900 relies on guidelines the compiler cannot check. P3846R1's response separates the design question from the frequency claim. On design, it states that the requirement "is not specific to P2900's design; it applies to any correct usage of an assertion facility in any programming language, including existing C++ facilities such as C assert and other preprocessor-based assertion macros" (p. 19), and it argues that enforcing such constraints is impossible without making all but the most primitive C++ expressions inside a predicate ill-formed, citing P3499R1. On frequency, it states: "Decades of experience with these facilities have shown that destructive side effects from predicates are easily identified during development and testing and are rarely an issue" (p. 19), supported in the same paragraph by teaching history: "We have been teaching for a long time that assertions should not have side effects (e.g., Rule 68 of [Sutter04]), and users have learned to use them correctly and effectively" (p. 19).

P3499R1<sup>[37]</sup> analyzes strict compiler-enforced predicate restrictions and shows that broad restrictions reject useful C++ expressions. That is evidence that the enforcement difficulty is real rather than asserted, and the design half of the response has support. The response also notes that const-ification supplies a partial compiler-enforced guard against accidental modification.

For the frequency claim, the response supplies no frequency data, no survey, and no defect study; the teaching history establishes that the rule is taught; it says nothing about how often the bug occurs. The symmetric standard limits the counter-evidence in the same way, and it is limited: CERT's PRE31-C rule recognizes the bug class but labels its likelihood "Unlikely" in its own risk table<sup>[38]</sup>, and the two static-analysis checks in this class target other languages, SonarQube S3346 being a C# rule and PVS-Studio V6055 a Java diagnostic<sup>[39]</sup><sup>[40]</sup>. These diagnostics establish recognition of the bug class without establishing its incidence in C++. Neither side supplies frequency measurements.

The response is Substantially supported. The "rarely an issue" frequency claim is a material unsupported subclaim. The objection is Partly answered: the design difficulty is evidenced, and the empirical characterization is not.

### Concern 8: Constification

Concern 8 is the objection that const-ification is problematic: that it changes the meaning of predicates, complicates teaching, and obstructs automatic assertion insertion. P3846R1's response grants the mechanism and contests the harm. It states that const-ification does not change the const-ness of a parameter used in a postcondition assertion, that it "can change overload resolution in contract-assertion predicates, but this change impacts only code that is already unusable in practice ([P3261R2])", and that "No compelling real-world examples of correct assertions rendered incorrect by const-ification have been produced" (p. 20). It reports that "when existing implementations of P2900 in GCC and Clang were applied as a replacement for legacy assertion libraries, const-ification revealed genuine bugs in existing libraries" (p. 20), and it records that "In Wroc&lstrok;aw, EWG reached consensus against removing const-ification, which was reaffirmed in Hagenberg" (p. 20).

The migration experiments produced defect counts in two large codebases. Applied to BDE, the experiment found six assignment-versus-equality defects<sup>[41]</sup>. Applied to LLVM, it found approximately seventy-five const-correctness defects before about 98.5 percent of assertions compiled, a figure reported by P3261R2<sup>[42]</sup>; the implementers report records the qualitative failure categories and about one day of cleanup work<sup>[43]</sup>. The adoption and retention of the feature were polled three times. SG21's 2023-12-14 teleconference adopted the modification-protection design then circulating as D3071R1, published days later as P3071R1<sup>[44]</sup>, with SF 6, F 10, N 3, A 0, SA 0, recorded result "Consensus"<sup>[45]</sup>, corroborated by the rationale paper<sup>[24]</sup>. EWG polled removing const-ification at Wroc&lstrok;aw, "P3261R1 / P3478R0: P2900 shall not have `const`-ification by default", SF 10, F 4, N 9, A 19, SA 12, "consensus against"<sup>[46]</sup>, and again at Hagenberg, "P2900: remove constification", SF 9, F 7, N 6, A 37, SA 14, "Consensus against"<sup>[23]</sup> - the second margin wider than the first, which is what "reaffirmed" describes. The two questions are not worded alike: Wroc&lstrok;aw polled removing const-ification as the default while leaving it available, and Hagenberg polled removing it outright. The wider margin was recorded against the stronger question, so the wording difference does not change the direction of the margin comparison. One characterization note: the response describes the adoption poll as having "strong consensus" (p. 20); the recorded label is "Consensus", and WG21 subgroups use a distinct stronger label that was not applied here<sup>[45]</sup>. The 16-0 tally supports the substance, if not the label.

The limits preserve part of the objection. The experiments test existing large codebases; they demonstrate limited migration cost and defect detection in the tested code; whether overload-selection differences are harmless in every correct program remains undemonstrated. The BDE result is not attributable to const-ification alone, because P3336R0 states that both const-ification and the restricted contract-predicate grammar prevent the six assignment mistakes<sup>[41]</sup>. The teachability and automatic-insertion concerns are addressed by analysis in P3261R2<sup>[42]</sup>, without new evidence.

The response is Substantially supported, with no material unsupported subclaim. The objection is Partly answered: const-ification was implementable, exposed real defects, and was retained through two removal polls, while the overload-selection and teachability questions are answered by analysis rather than demonstrated harmlessness.

### Concern 9: Global violation handler

Concern 9 is the objection that global contract-violation handlers are problematic. P3846R1's response grounds its analogy in the standard library before it reaches beyond it: "C++ already includes several global handlers for this purpose (e.g., std::set_new_handler, std::set_terminate, signal handlers), and similar mechanisms are widely and successfully used in major frameworks such as Qt and in game engines" (pp. 22-23). The "purpose" named in that sentence is the role the preceding sentence assigns to the contract-violation handler - the function invoked when a checked assertion fails: "This latter behaviour is exactly the role of P2900's contract-violation handler" (p. 22). The response distinguishes the removed `std::unexpected` on the following line: "Unlike std::unexpected, these facilities are not intended for recovery, and experience with them has been positive" (p. 23).

The rationale paper records the history of per-assertion and global-handler designs<sup>[24]</sup>, and P2811R7 states that BDE deployed user-provided contract-violation handlers in 2004 and continued using them<sup>[47]</sup>. The production history therefore sits inside the response's own cited sources. The `std::unexpected` distinction also holds: its removal followed the removal of dynamic exception specifications, so its history does not establish that every global diagnostic handler is defective.

The extension beyond the standard library is unsupported. The Qt and game-engine half of the sentence names no deployments and no outcomes; "widely and successfully used" is asserted for those two ecosystems without evidence. The response's further claim that a single global handler is essential for using contract assertions at scale is a design judgment the BDE experience supports as one data point, which falls short of a universal result.

The response is Substantially supported. The Qt and game-engine success claim is a material unsupported subclaim; the standard-library half of the analogy stands. The objection is Partly answered.

### Concern 14: Missing features

Concern 14 is the objection that P2900 lacks important features. P3846R1's response is an incremental-delivery argument with one categorical sentence in the middle of it: "SG21 and EWG iteratively selected among the many potential features that should be included in P2900. All the requested features have been discussed in various papers; no proposals that included them gained consensus in EWG" (p. 28). Around that sentence, the response describes the minimal-viable-product strategy, inventories the requested extensions, and names active or exploratory proposals for them. One page later, the same section records the virtual-function history itself: "pre and post on virtual functions do have a proposal ([P3097R0]) that is fully specified, has been reviewed and approved with strong consensus in EWG, has been reviewed by CWG, has been implemented in GCC, and could be re-added to the C++ working draft any time EWG wishes to do so" (pp. 29-30). CWG is the Core Working Group, the subgroup that reviews language wording before it enters the working draft.

EWG at St. Louis polled "we would like to see this paper merged into P2900 and progress contracts with virtual function support" for P3097R0<sup>[14]</sup>: SF 18, F 15, N 5, A 1, SA 2, posted on the tracker without a result line<sup>[48]</sup> and recorded as "Result: Consensus" by the rationale paper<sup>[24]</sup>. The feature was merged into P2900R8, a revision of the proposal, never into the C++26 working draft. At Hagenberg, EWG polled "P2900: disallow pre/post contracts on virtual functions entirely": SF 20, F 24, N 13, A 14, SA 2, consensus in favor<sup>[23]</sup>, and the feature was struck from P2900R14<sup>[2]</sup> before P2900R14 entered the working draft that same week. Wroc&lstrok;aw, 18-23 November 2024<sup>[21]</sup>, took place between St. Louis and Hagenberg, so Hagenberg was not the next meeting after St. Louis. The history as the response itself tells it is accurate.

The defect is the categorical sentence. The virtual-function history appears in the same section of P3846R1, one page after the sentence it contradicts, so the sentence needs a virtual-function exception: the St. Louis poll is a proposal that included a requested feature and gained EWG consensus. The label "strong consensus" is the authors' characterization; no public source applies it to the St. Louis poll<sup>[48]</sup><sup>[24]</sup>, and the 33-3 tally supports "consensus" either way. The incremental-delivery response around the defective sentence is substantive: the requested features were individually polled at Hagenberg with no consensus to add them, and the named extension proposals demonstrate an evolution path. That path supports extensibility; it does not give users the capabilities in C++26.

The response is Substantially supported. The categorical consensus sentence is a material unsupported subclaim. The objection is Partly answered.

### Concern 17: Deployment experience

Concern 17 is the objection that P2900 has insufficient deployment experience. P3846R1's response accepts the standard the objection invokes and then states its strongest sentence: "Expecting a reasonable level of implementation experience before standardising a novel language feature is good engineering practice that we strongly support. P2900 has been fully implemented in two major compilers." (p. 33). Eight lines later on the same page, the Details subsection qualifies both halves: "Two complete implementations of P2900 exist in GCC and Clang ([P3460R0]) and are publicly available (including on Compiler Explorer), with upstreaming in progress" (p. 33), and "With these implementations nearly complete, contract assertions are much closer to being merged to mainline branches of these compilers than are other significant C++ features (such as modules) at this point in the standardisation process" (pp. 33-34). The tension among "fully implemented", "nearly complete", and "upstreaming in progress" is P3846R1's own. The response also states the deployment gap expressly: "While it has not been deployed to production, neither has any other major language feature adopted by C++ in any previous or current Standard" (p. 33).

Both implementations were public and nightly-built on Compiler Explorer at both cutoffs<sup>[49]</sup>, which supports the availability half of the response.

**Table 3. Provenance ledger for the deployment evidence cited under Concern 17. Each row records the implementer or authors, the tested component, the codebase, the publication date, the relationship to P2900 at that date, and the limitation of the result.**

| Source | Implementer or authors | Tested component | Codebase | Date | Relationship to P2900 at that date | Limitation |
|---|---|---|---|---|---|---|
| P3460R0<sup>[43]</sup> | Nina Dinka Ranns, Iain Sandoe | Full language implementation | GCC development fork | 2024-10 | Implements P2900R8; N4820-era `contract_violation` object | Lambda capture rules and contract violation on exception unimplemented; code merged into no official branch |
| P3460R0<sup>[43]</sup> | Eric Fiselier | Full language implementation | Clang development fork | 2024-10 | Implements P2900R8 | Virtual functions unimplemented; libc++ experience limited to `contract_assert` |
| P3336R0<sup>[41]</sup> | Joshua Berne | Migration experiment replacing assertion macros | BDE | 2024-06 | Development versions of P2900 | Experiment, not production deployment |
| P3191R0<sup>[50]</sup> | Louis Dionne, Yeoul Na, Konstantin Varlamov | Contract-violation handler requirements | libc++ and bounds-safety deployments | 2024-03 | Handler experience later adopted into P2900 as quick-enforce | Library-side requirements, not language deployment |
| P3471R4<sup>[51]</sup> | Konstantin Varlamov, Louis Dionne | Standard-library hardening | libc++ | 2025-02 | Hardening defined over contract-violation semantics | Library hardening experience, not the complete language feature |

The strongest sentence is not supported by the only implementation report the response cites. P3460R0 states that "At the time of writing this paper, the most recent version of P2900 is P2900R8" (p. 1), describes an N4820-era violation object, and records that "The code has not been merged into any official branch" (p. 5)<sup>[43]</sup>. Its feature matrix records complementary gaps in the two compilers - Clang lacked virtual functions, GCC lacked lambda capture rules and contract violation on exception - and its libc++ deployment experience is expressly "limited to contract_assert, as no new pre and post conditions were added" (p. 4)<sup>[43]</sup>. Read against P2900R14<sup>[2]</sup>, "fully implemented in two major compilers" is not what the cited report documents. The upstream record confirms the gap on the Clang side: the Clang status page recorded P2900R14 as not implemented at both cutoff dates, no upstream branch or pull request was located, and the implementation remained a single-maintainer out-of-tree fork<sup>[52]</sup>.

Between the cutoffs, the prediction in the response came true for GCC. On 2026-01-28, a nine-commit series was merged into GCC master including the base implementation commit "c++, contracts: C++26 base implementation as per P2900R14"<sup>[53]</sup>. The March artifact still says "with upstreaming in progress" two months after that merge, so the publication-state record is stale in the authors' own disfavor; for Clang, no upstreaming had occurred at either cutoff. Later agreement with P2900 by an author listed in Table 3 proves agreement at that later date, and no result in the table is discounted on that basis.

The response is Substantially supported. "Fully implemented in two major compilers" is a material unsupported subclaim as applied to P2900R14. The objection is Partly answered: substantial implementation and component experience exist and are stated to fall short of production deployment of the complete feature, and whether the available level suffices before standardization is a policy judgment the evidence does not settle.

Concerns 4, 7, 8, 9, 14, and 17 have supported central claims with material limits that preserve part of each objection. In each row the support and the limit are both on the record, and neither cancels the other.

---

## 7. Ten Responses Where Support and Weakness Are Both Material

This section covers the ten responses - Concerns 1, 2, 3, 6, 11, 12, 13, 15, 16, and 18 - where material support and material weakness both affect the central response, so neither controls the complete answer. A delegate needs the two halves separated within each concern, because these are the rows where quoting either half alone misstates the record.

### Concern 1: Safety and non-ignorable checks

Concern 1 is the objection that contract assertions make C++ less safe because they can be switched off: as P3846R1 records it, P3835R0<sup>[3]</sup> and three national body comments characterize P2900 as providing no method to guarantee in code that a particular assertion will always be checked. P3846R1's response separates the checked and ignored cases and then makes its strongest claim: "Assertions do not make C++ 'less safe'. When checked, they can detect bugs; when ignored, they have no runtime effect while documenting intent. The ability to configure their evaluation semantics externally is a prerequisite for widespread adoption, not a defect." (p. 6). The response grounds the necessity claim in adoption history, arguing that the ability to ignore assertions "is what enables it, as proven by decades of successful use of C assert" (p. 7), and it lists routes to non-ignorable checks that P2900 leaves open: build-time selection of a checked semantic, duplicating the check in source, a vendor attribute, and ordinary control flow (p. 7).

The narrower claim within the categorical one has documented support. P2877R0<sup>[54]</sup> reports production settings where observing or disabling individual assertions reduces the risk of outages and removes a disincentive to liberal use. P3500R1<sup>[55]</sup> identifies production, gaming, low-latency, server, and high-performance computing environments with conflicting enforcement needs. Together they establish that configurability reduces adoption barriers in the environments that asked for it.

The categorical sentence asserts more than that evidence establishes. The historical coexistence of `NDEBUG` and adoption does not establish causation, and the response supplies no study, survey, or usage data linking the mechanism to the adoption; the predicted consequences of removing the ignore semantic - checks omitted in performance-sensitive code, removed after initial development, libraries abandoned for more performant alternatives (p. 8) - are likewise asserted without comparative usage evidence. Of the enumerated routes to non-ignorable checks, one also falls short of demonstration: the Clang vendor attribute shown in the response's own code line has no located public implementation. The Rust comparison available to the objecting side has its own limits, and they belong in the same paragraph as the analogy: Rust checks a fixed class of operations, permits source-level `unsafe` bypass, and benefits from optimizer removal of redundant checks, so it is not structurally equivalent to arbitrary user-written predicates. The 2021 study of restoring Rust's elided checks serves both sides: little, no, or negative benefit in 76.4 percent of tested benchmarks and meaningful gains in 23.6 percent, with the authors cautioning that restoring every check is not always realistic<sup>[56]</sup>. The November 2025 Android report, which describes checked-by-default Rust deployed at scale<sup>[57]</sup>, is dated 2025-11-13: it is publication-state evidence that qualifies the adoption claim for the March artifact, and it was not available at the November cutoff.

The response is Mixed. The categorical prerequisite claim is a material unsupported subclaim: the record supports the narrower proposition that configurability reduces adoption barriers, and stops short of the proposition that ignorable semantics are a prerequisite for widespread adoption. The objection is Partly answered: the checked-versus-ignored analysis and the enumerated routes to non-ignorable checks are substantive, while the necessity claim and the predicted harms of removing ignore remain unestablished.

### Concern 2: Cross-translation-unit semantics

Concern 2 is the objection that P2900 does not provide consistent semantics across translation units: in a program combining translation units compiled with different evaluation semantics, no rule dictates which semantic governs an assertion shared between them. P3846R1's response is a five-item strategy list, introduced as "Implementation strategies for mitigating the issue highlighted by [P3835R0]" with none requiring any change to P2900's specification (p. 10). The first item is "The naive implementation (implemented in GCC and Clang): compile each function with the contract-evaluation semantic specified for that TU; if multiple definitions exist, the compiler will choose one arbitrarily" (p. 10), and its worst case is scoped and qualified: "The worst case (barring compiler bugs such as those described in Concern 4) is that a contract assertion intended to be checked is instead ignored, which is no worse than if contract assertions did not exist" (p. 10). The remaining items defer selection to link, load, or run time ("prototyped in GCC"), refine that approach with a link-time constant ("not yet implemented"), encode the semantic in the application binary interface (ABI), the binary-level naming and calling contract between separately compiled components ("prototyped for the Itanium ABI"), and await future linker support (p. 10).

The taxonomy and the first strategy are on the record. P3267R1<sup>[58]</sup> documents the caller-side, callee-side, double-sided, delayed, runtime, and load-time selection strategies, and P3321R0<sup>[26]</sup> describes compile-time, link-time, and runtime selection. The naive strategy is implemented and public: P3460R0<sup>[43]</sup> documents it in both compilers, and the GCC development fork's option table selects the evaluation semantic per translation unit at compile time through `-fcontract-evaluation-semantic=`<sup>[59]</sup>. The qualified worst-case sentence is accurate as written: it is the second sentence of the naive-strategy bullet, it is scoped to that strategy, and its parenthetical excludes the compiler-bug case by construction, so the regression recorded in GCC Bug 121936<sup>[34]</sup> does not contradict it.

What the record does not contain is any public artifact for the two claimed prototypes. The deferred-selection bullet reports that link-time optimization "has been shown to work reliably in the GCC prototype" (p. 10); no public code, measurement, or build log for that prototype exists, and the absence is positive rather than a failed search: GCC's complete `fcontract*` option table contains no link-time, load-time, or runtime selection option at either cutoff<sup>[59]</sup><sup>[60]</sup>. The ABI bullet states that "A proof of concept for this approach has been shared with WG21" (p. 10). The one public contracts-ABI repository, efcs/contracts-abi<sup>[61]</sup>, consists of a README and a `.gitignore`, 22 KB, frozen since 2025-06-27; P3846R1 never cites it, so it is not a P3846 source artifact; and it specifies the violation-entrypoint ABI rather than the user-function symbol-selection strategy this concern describes. The response itself states the gap: writing of an inline function `f` in a shared header, it says that on the naive implementation "users who do not fully control their build environment cannot reliably predict which evaluation semantic applies to non-inlined calls to f" (p. 10), and P3267R1 states that mixed compilation modes can reduce the minimum evaluation count to zero<sup>[58]</sup>.

The response is Mixed. The claimed deferred-selection prototype and the reported link-time reclamation result are material unsupported subclaims. The objection is Partly answered: the strategy space is documented and the naive strategy is implemented in both compilers, while predictable control over the semantic of a precompiled dependency remains undemonstrated in any public artifact.

### Concern 3: Dependency management

Concern 3 is the objection that the impact of P2900 on dependency management is unclear. P3849R0<sup>[5]</sup>, the paper P3846R1 names as raising it, states that "Contracts introduce several new build configurations, but we have not yet seen concrete examples of how they interact with real-world build systems or complex dependency graphs." P3846R1's response is a replacement argument followed by a build-system example: "P2900 introduces no new configuration dimension; rather, it replaces a proliferation of custom flags to control macro-based assertions with a single mechanism" (p. 11), and "Boost.Build already added such support on top of the available GCC and Clang implementations of P2900. Adding this support took less than an hour of implementation effort, demonstrating that contract assertions fit naturally into existing build models." (p. 12). The same paragraph states: "The support includes documentation covering scenarios such as static and dynamic linking against a library where both the library and main() are independently compiled with either ignore or enforce" (p. 12). The example demonstrates the static half of that list, pairing an independently compiled library with its client across the ignore-and-enforce combinations.

The example is genuine. Boost.Build commit 3b20a4e1<sup>[62]</sup>, dated 2025-09-30, carries 149 additions across 9 files covering contract flags, a feature declaration, documentation, and an example, and its flag mapping is a correct identity mapping from build-system value to compiler flag, so builds behave properly. The cited example repository<sup>[63]</sup> combines enforce and ignore for a library with enforce and ignore for its client and confirms that the library's compiled semantic governs a non-inline function: a coherent, runnable demonstration of the per-translation-unit model. The response is also candid about the boundary of its claim, stating that whether and how mixing different choices across translation units is supported "is, and remains, outside of the scope of the C++ Standard" (p. 11).

Three limits qualify the demonstration. First, the elapsed-time figure is unverifiable: the quotation is exact, but no public record establishes the duration - the commit's parent is eleven weeks earlier, and there is no pull request, issue, or review thread - so the figure is the author's own unwitnessed report about his own work<sup>[62]</sup>. Second, the documentation added by that commit mispairs all four evaluation semantics with descriptions belonging to different semantics: `enforce` is documented as "Contract assertions are not evaluated (contracts are disabled)" and `observe` as "Contract assertions are evaluated and violations terminate the program", descriptions belonging to `ignore` and `enforce` respectively, and the file remains uncorrected on the development branch<sup>[62]</sup>. Third, the cited example does not demonstrate dynamic linking: line 42 of its `build.jam` reads `for local link_lib in static #shared`, the shared case was commented out from file creation, and the matrix it builds is static-only across four targets<sup>[63]</sup>. The response's sentence names the documentation rather than the example, and "such as" makes the list illustrative, so the record does not contradict the sentence; the defect is that the demonstrated artifact reaches only the static half of the scenarios the documentation describes.

Two further facts bound the example in time and provenance. B2 5.4.0, the first release carrying the support, was published 2025-12-20, so at the November cutoff the support existed only on the development branch<sup>[64]</sup>. The commit author, Ren&eacute; Ferdinand Rivera Morell, is both the B2 maintainer and a P3846R1 coauthor, which is a disclosure fact that does not invalidate a public commit. The symmetric rule applies to the response as well: P2877R0, whose authors Joshua Berne and Tom Honermann are both P3846R1 signatories, states that "A Contracts design that requires build-time decisions regarding whether contracts are evaluated and what the consequences of contract violation are creates a significant burden for package managers"<sup>[54]</sup>, and Concern 3 does not engage it.

The response is Mixed. The elapsed-time figure is a material unsupported subclaim, and the documentation sentence outruns the demonstrated example, which reaches static linking only. The objection is Partly answered: one mature build system exposed the basic controls through a small, correct patch, while the package-manager burden documented by the response's own signatories and the complex-dependency-graph half of the objection remain unaddressed.

### Concern 6: Implementation-defined behavior

Concern 6 is the objection that too much of P2900 is implementation-defined. P3846R1's response explains that C++ makes behaviors implementation-defined when they are platform-dependent or when the same choice does not suit every program on a platform, reports that only the selection of evaluation semantics will need regular user attention, and compares the choices made in the GCC and Clang development branches in a table (pp. 16-18). It then enumerates: "P2900 introduces exactly five implementation-defined properties:" (p. 17) - the termination mode used by enforce and quick-enforce, the behavior of the default contract-violation handler, whether the handler is replaceable, the choice of evaluation semantic, and the maximum number of repeated evaluations.

The operational account is substantive, and the enumerated items are real. Four of the five map to properties carried by the incorporated working draft's index: the evaluation semantic, the maximum repeated evaluations, the contract-termination method, and handler replaceability<sup>[65]</sup>. The remaining item is not invented: P2900R14's own design prose states that the default contract-violation handler "has implementation-defined effects"<sup>[2]</sup>. The response's explanation of why the principal choices are platform- or configuration-dependent stands undisturbed.

The completeness claim fails against the draft the response invokes. N5008, the C++26 working draft dated 2025-03-15, lists seven contract-related entries in its own index of implementation-defined behavior<sup>[65]</sup>: the four above, plus the virtual-destructor choice for `contract_violation`, the `comment()` contents, and the `location()` value. That enumeration was public 233 days before P3846R1's printed date. The arithmetic is not "five of seven": four of P3846R1's five items map to indexed properties, its second item has no index entry because the normative wording expresses the default handler's behavior as Recommended practice rather than as an implementation-defined designation, and three indexed properties are omitted. The count is bounded on both sides so that it cannot be read as selective: a naive search of P2900R14's proposed wording returns ten "implementation-defined" hits, and the three beyond the seven are excluded with reasons, two being pre-existing C++23 text<sup>[66]</sup> and one being Recommended practice. The omission is not attributable to obscurity: P3321R0, which the same paragraph of P3846R1 describes as discussing "the full list of implementation-defined behaviours" (p. 17), contains a section titled "What strings get put in a contract-violation object?" on exactly the `comment` and `location` contents the list omits<sup>[26]</sup>.

The response is Mixed. The "exactly five" completeness claim is a material unsupported subclaim, disproved by the working draft P3846R1 repeatedly invokes. The objection is Partly answered: the operational explanation of the principal choices is substantive and undisturbed, while the enumerated list is not complete against the wording's own index.

### Concern 11: Predicate exceptions - the one objection not resolved

Concern 11 is the objection that treating exceptions thrown during predicate evaluation as contract violations is infeasible. As P3846R1 records it, "[FI-071] comments that no implementation or deployment experience of P2900 exists for non-Itanium ABIs and adds that Microsoft considers it infeasible to treat exceptions thrown from the evaluation of contract predicates as contract violations" (p. 25). P3846R1's response describes the two constituencies its design serves - one requiring that predicate exceptions never escape, an approach that "caused issues in the C++20 Contracts proposal", the other requiring recovery from exceptions such as `bad_alloc` regardless of origin - and states its central claim: "The approach in P2900 is the only known solution that satisfies both groups: exceptions thrown during predicate evaluation are treated as contract violations and passed to the violation handler, allowing user-defined recovery strategies while maintaining sound control-flow semantics" (p. 25).

The mechanism is coherent on its own terms, and the cost argument has a documented basis. Routing predicate exceptions to the violation handler lets the handler terminate, continue according to the selected semantic, or rethrow, which prevents predicate exceptions from creating check-dependent control paths while preserving a route for recovery. P3591R0<sup>[67]</sup>, which P3846R1 records as its address to the Microsoft position raised previously in P3506R0<sup>[6]</sup>, explains why exception-handling support code disappears for predicates statically known not to throw.

The response states that "The overwhelming majority of predicates are trivially non-throwing" (p. 25), and the Details subsection grounds that in "our experience" (p. 25): author experience, with no cited dataset. On the platform the objection named, the response supplies no implementation or measurement on a non-Itanium interface, and none was located in the public record.

The procedural record shows division, not resolution. EWG at Hagenberg polled "P2900: unconditionally unwind exceptions when they leave predicate evaluation": SF 12, F 18, N 11, A 15, SA 7, official result "No consensus for change"<sup>[23]</sup>, corroborated by the rationale paper<sup>[24]</sup>. P3846R1 characterizes that history as both groups concluding that "prioritising the speculative overhead of a small subset of predicates over ensuring that contract assertions have well-defined, analysable control flow would be unsound" (p. 25). The poll records an unresolved division of preference, and the thirty votes favoring propagation establish a substantial constituency for a different tradeoff. Two documented alternatives exist, and neither satisfies both constituencies. P3626R0<sup>[13]</sup> is "Make predicate exceptions propagate by default" by the lead author of P2900R14 and P3846R1, described by the rationale paper as the wording diff prepared so EWG could poll the alternative<sup>[24]</sup>; unconditional propagation does not satisfy the no-escape constituency at all. P3909R0<sup>[68]</sup> describes a different programmatic architecture and notes that translation costs differ on non-Itanium interfaces, without specifying a complete alternative that satisfies both groups. P4009R0<sup>[69]</sup> supplied another customizable architecture by the March cutoff, likewise without measured implementation evidence.

The response is Mixed. The claim that the overwhelming majority of predicates are trivially non-throwing is a material unsupported subclaim, resting on author experience rather than a cited dataset. The objection is Not resolved: the response supplies a coherent mechanism for its two stated constituencies, and it supplies no implementation or measurement on the non-Itanium interface where the objection reported infeasibility.

### Concern 12: Static analysis

Concern 12 is the objection that P2900 does not support static analysis. P3846R1's response is a syntax argument followed by a vendor-activity claim. The syntax argument: macro-based assertions are limited by non-uniform syntax, by preprocessing removal, and by placement inside function bodies, and contract assertions on declarations remove all three (p. 26). The vendor-activity claim: "Some static analysis providers (such as CodeQL) are already actively pursuing support for P2900 contract assertions in their tools" (p. 26), supported in the Details subsection by the CppCon work, which "demonstrated how P2900 can enable static proofs beyond conventional flow and range analysis" and combined "the CodeQL static analyser with the Z3 constraint solver to validate a wide range of contracts" (p. 27).

The syntax advantages are corroborated from both directions. P3386R1<sup>[70]</sup> explains how declaration-level standardized syntax can improve interprocedural analysis. P3893R0<sup>[18]</sup>, written by the talk's CodeQL copresenter, acknowledges that declaration-level contract specifiers can simplify dependency modeling for CodeQL, and the prototype repository states the forward intent plainly: "In the future, we hope to support C++26 contract specifiers `pre(...)` and `post(...)`"<sup>[71]</sup>.

The limitation is the prototype's scope. P3893R0 states that the demonstrated prototype targets traditional assertions rather than P2900 contract specifiers and warns against using it to judge the overall feasibility of P2900 static analysis; its statement that "the portions of this talk presented by GitHub are not an endorsement of P2900"<sup>[18]</sup> is a corporate non-endorsement, and a corporate non-endorsement is not a technical contradiction. The repository limits the demonstration further: plain `assert` is not supported either, only "`assert` macros that have been annotated `/*@ requires @*/`"<sup>[71]</sup>, a bespoke comment annotation of the kind P3846R1's own Concern 12 lists among the limitations of the status quo (p. 26). The repository has been frozen since 2025-09-19, so the score is identical at both cutoffs<sup>[71]</sup>. One process note completes the record: P3846R1's revision history records adding discussion of P3893R0, while the body of Concern 12 does not cite or answer it<sup>[1]</sup>.

The response is Mixed. The characterization of the CppCon work as validating a wide range of contracts is a material unsupported subclaim: the demonstrated prototype validates annotated assertion macros; it does not reach P2900 contract specifiers. The objection is Partly answered: the syntax-level advantages are real and acknowledged from both directions, while demonstrated P2900-specific analysis remains future work in the cited prototype.

### Concern 13: Complexity

Concern 13 is the objection that P2900 is too complex. P3846R1's response separates user-facing simplicity from specification length, reports that "Complete implementations of P2900 in GCC and Clang were produced relatively quickly by a tiny team ([P3460R0]) and with limited impact on those compilers", and then makes its comparative claim: "The implementers reported that P2900 is orders of magnitude simpler to support than modules, concepts, reflection, or even lambdas" (p. 27).

The tractability half is documented. P3460R0<sup>[43]</sup>, the cited report, states that "P2900's specification was clear and implementable" and records no major difficulty with the implemented components. P3591R0<sup>[67]</sup> makes a similar qualitative comparison publicly, and P4020R0<sup>[72]</sup>, available by the March cutoff, calls the minimal form "fairly simple to implement". P4020R0's author is Andrzej Krzemie&nacute;ski, and its Recommendations section declines to choose between shipping as-is and another vehicle: "There seem to be only two coherent responses to RO 2-056: Take no action (ship as is). Implement solution 'v4' from [P3911R0] (move C++ contracts to other shipping vehicle)." Committee adoption, which the response also cites, establishes perceived value sufficient for standardization.

The comparative magnitude is not documented. P3460R0 contains no comparison to modules, concepts, reflection, or lambdas<sup>[43]</sup>, and the sentence attributing "orders of magnitude simpler" to the implementers has no cited report, measurement, or attributed quotation. The sentence is attributed judgment rather than a result P3460R0 establishes, and quick implementation by a small team supports relative tractability without quantifying the comparison. Committee adoption, likewise, establishes perceived value sufficient for standardization without establishing an objective complexity ratio.

The response is Mixed. The "orders of magnitude" comparison is a material unsupported subclaim. The objection is Partly answered: straightforward implementation and introductory use are documented, while no source supplies a measured comparison against the named features.

### Concern 15: Future features

Concern 15 is the objection that adopting P2900 now forecloses or complicates future features, principally deep const and generic decorators. P3846R1's response is procedural and technical. Procedurally, it quotes SD-4, the WG21 Practices and Procedures standing document: "we do not significantly delay progress on concrete proposals in order to wait for alternative proposals we might get in the future" (p. 31)<sup>[73]</sup>. Technically, it explains why deep const and decorators would each require their own design work and why decorators differ from contract assertions. P3846R1 agrees on the point of principle - "We agree with the authors of [P3829R0] that having deep const in the language might be beneficial" - and then states its historical claim: "Yet in more than four decades of C++ evolution, no proposal for deep const has ever been brought forward, and it appears doubtful that one will ever materialise" (p. 31).

The procedural case is intact. SD-4 states the rule the response applies<sup>[73]</sup>, P3829R0<sup>[4]</sup> raises future compatibility questions but demonstrates no incompatibility with a complete deep-const or decorator design, and P3261R2<sup>[42]</sup> reports independent difficulties with proposed deep-const approaches, which supports the response's assessment that specifying deep const is a daunting undertaking.

The historical sentence is false, and the falsity is narrow. P1974R0<sup>[74]</sup> explicitly proposes `propconst`, a language-level qualifier for deep constness, and P2670R1<sup>[75]</sup> revises that design line, so a proposal for deep const has been brought forward. One false sentence does not refute the complete response: P1974R0 predates P2900 constification and does not show that constification obstructs or otherwise affects a future deep-const facility, and no concrete compatibility analysis between P2900 and a complete deep-const or decorator design was located on either side. The compatibility question remains open on both sides.

The response is Mixed. The four-decades historical claim is a material unsupported subclaim, and it is false. The objection is Partly answered: the procedural case against blocking P2900 on an unspecified future design stands, while the compatibility of constification with a future deep-const facility is analyzed by neither side.

### Concern 16: Decomposition

Concern 16 is the objection that contract assertions could be composed from more primitive features standardized individually. P3846R1's response is that the proposed decomposition is incomplete and that the idea has already failed: "The idea to redesign contract assertions as a composition of more primitive features was first proposed in [P1893R0] and subsequently shown to be inadequate for the real-world use cases for contract assertions ([P1995R1])" (p. 32). Around that sentence, P3846R1 argues that "P2900 clearly qualifies as 'in flight', whereas the decomposition sketch in [P3829R0] lacks the required detailed discussion" (p. 32), invoking P2000R4's rule that a radical change to a proposal already in flight may not delay it "unless it comes with a paper with a detailed discussion of design, use, and implementation"<sup>[76]</sup>.

The substantive rebuttal is concrete and stands. P3829R0 calls its own decomposition a straw-man sketch<sup>[4]</sup>. P3846R1 identifies the semantics the sketch is missing: a global contract-violation handler shared across libraries, control over when checking code is injected or omitted, and the assertion marker that tools consume (pp. 32-33). P2000R4 supplies the procedural reason not to delay an advanced proposal for a late, incomplete redesign<sup>[76]</sup>. P3859R0<sup>[9]</sup> and P3896R0<sup>[10]</sup>, both by Andrzej Krzemie&nacute;ski, a P2900R14 co-author who did not sign P3846R1, argue that an atomic assertion marker supports tooling. P4009R0<sup>[69]</sup>, available by the March cutoff, made a library-oriented alternative more concrete without providing a deployed replacement.

The defect is citation mismatch. P3846R1 attaches ([P1995R1]) directly to the "shown to be inadequate" sentence, so a citation exists; P1995R1<sup>[11]</sup> catalogs and polls use cases but does not mention or evaluate P1893R0<sup>[77]</sup>, so the citation does not support the sentence. Because P3846R1 does not cite P2899R1 for that sentence, P1893R0's absence from the rationale paper<sup>[24]</sup> is not independently probative and is not treated as such. The broader claim that contract assertions are non-decomposable, "a very contract-specific and non-decomposable property" (p. 33), is an argument from required properties; it is not a demonstrated impossibility result.

The response is Mixed. The claim that the earlier design was shown to be inadequate is a material unsupported subclaim: the attached citation does not support it. The objection is Partly answered: the identified omissions in the proposed decomposition are concrete and unanswered, while non-decomposability remains an argument from required properties rather than a demonstrated result.

### Concern 18: Library hardening

Concern 18 is the objection that standard-library hardening cannot depend on contract assertions. P3846R1's response names the four national body comments seeking decoupling in its first sentence - "[FR-001-014], [US 3-015], [US 61-112], and [FR-010-113] ask for standard-library hardening to be specified independently from contract assertions due to a lack of deployment experience with the latter" (p. 35) - and engages P3878R0<sup>[7]</sup> directly. It explains that hardening can be specified through the contract-violation model without the literal syntax: "the specification of standard library hardening requires that a violation of a hardened precondition behaves as if a contract assertion had been violated, but this does not require the use of a literal contract assertion (the syntactic construct) for that purpose" (p. 36). It identifies vendor extensions, compiler recognition of hardening assertions, and preprocessor emulation of quick-enforce as implementation mechanisms (pp. 35-36), and it states that "Both the libc++ and libstdc++ implementation currently being planned once contracts are available are conforming implementations of C++26 standard-library hardening on top of P2900" (p. 35).

At the November cutoff, the response also anticipated the resolution the committee later adopted. P3846R1 describes the reduction of hardening to the enforce and quick-enforce semantics as "a sound decision that the committee could make" (p. 36) and observes that observe-like behavior could remain available under a term other than hardening. P3912R0<sup>[78]</sup>, written by six P3846R1 co-authors, states both halves of the vendor position: "the C++26 working draft lacks the additional annotations needed to implement a practically deployable hardened standard library. Labels as proposed in [P3400R2] address this gap. In their absence, vendors must rely on vendor-specific attributes (implemented in Clang) or use vendor-specific macros that behave as-if they were a contract_assert statement (a conforming approach currently used by both libc++ and libstdc++)." The first half identifies a deployability gap in the current working draft; the second half records the conforming macro approach, and both halves are consistent with the planned-implementation claim, which names the same vendor-attribute and preprocessor routes. That gap does not change the unsupported-subclaim flag: the claim under test is that the planned libc++ and libstdc++ implementations are conforming, the passage itself calls the macro approach conforming, and the deployability gap it names is one the response's own mechanism list presupposes. The flag remains No. The viable mechanisms and the direct engagement with P3878R0 are on the record.

Between the cutoffs, the committee acted. P3878R1<sup>[19]</sup>, dated 2025-11-06, required terminating semantics for a hardened implementation, and N5031 records its adoption at the Kona plenary by unanimous consent in a motion stating that the change "addresses ballot comments RU-016, FR-001-014, FR-010-113, US 3-015, and US 61-112"<sup>[8]</sup>. The disposition record contains no contrast between RU-016 and the four decoupling comments: all four were rejected alongside RU-016, each marked "Rejected (no change, intention Accepted)"<sup>[8]</sup>, meaning the requested textual change was not adopted while the underlying aim was met another way, with implementation-level decoupling preserved and observe excluded. The adopted direction narrowed and formalized an option P3846R1 had accepted rather than defeating its response. What the March publication shows is staleness: the artifact did not report the compromise reached more than four months before it was built. The phrase "on top of P2900" remains ambiguous between literal language syntax and the normative contract-violation model, which is why this response carries no unsupported-subclaim flag.

The response is Mixed, with no material unsupported subclaim. The objection is Partly answered: the November response identified the decoupling requests, supplied viable mechanisms, and accepted the restriction the committee later adopted, while the March artifact did not report that adoption and the central phrase remains ambiguous between the literal syntax and the contract-violation model.

Concerns 1, 2, 3, 6, 11, 12, 13, 15, 16, and 18 contain material support and material weakness, so neither position controls the complete response.

---

## 8. Fourteen Material Unsupported Subclaims, None Invalidating a Whole Response

This section collects the fourteen responses that carry a Yes unsupported-subclaim flag - Concerns 1, 2, 3, 4, 6, 7, 9, 11, 12, 13, 14, 15, 16, and 17 - so that the pattern is visible in one place. The collection exists because overall support is rated separately for the complete response in the summary table: a single defective sentence cannot invalidate everything around it, and the flag is the field that records the defect.

**Table 4. The fourteen material unsupported subclaims in P3846R1, in concern order. Each row names the concern and the section holding its full evidence unit, quotes the subclaim with its page in P3846R1, states what the public record does or does not establish, and classifies the defect. A false statement is contradicted by the record. An unsourced claim has no cited support. An unverifiable existence claim asserts an artifact that no public record contains. A citation mismatch attaches a citation that does not support its sentence. A claim outrunning its evidence has real evidence that does not reach the stated conclusion.**

| Concern | Unsupported subclaim | What the record establishes | Defect kind |
|---|---|---|---|
| 1 (Section 7) | "The ability to configure their evaluation semantics externally is a prerequisite for widespread adoption, not a defect" (p. 6) | Configurability reduces adoption barriers in the environments that asked for it; no study, survey, or usage data links the mechanism to the adoption | Unsourced causal claim |
| 2 (Section 7) | Deferred selection "prototyped in GCC", with link-time optimization that "has been shown to work reliably in the GCC prototype" (p. 10) | No public code, measurement, or build log; GCC's complete contract option table contains no deferred-selection option at either cutoff | Unverifiable existence claim |
| 3 (Section 7) | "Adding this support took less than an hour of implementation effort"; the support includes "documentation covering scenarios such as static and dynamic linking" (p. 12) | No public record establishes the duration; the example's shared-library case was commented out from file creation, so the runnable demonstration reaches static linking only | Unsourced self-report; claim outrunning its evidence |
| 4 (Section 6) | Performance concerns are "equally unfounded" (p. 15) | The cited bug contains the GCC interprocedural-optimization maintainer's report of a 47 percent gain on one workload and "quite considerable" surrendered opportunity in the narrow case | False statement |
| 6 (Section 7) | "P2900 introduces exactly five implementation-defined properties" (p. 17) | The incorporated working draft's index lists seven contract-related entries, public 233 days before the printed date; four of the five map to indexed properties, one has no index entry, three are omitted | Claim outrunning its evidence |
| 7 (Section 6) | Destructive side effects from predicates "are rarely an issue" (p. 19) | No frequency data, survey, or defect study; the teaching history establishes that the rule is taught, not how often the bug occurs | Unsourced quantitative claim |
| 9 (Section 6) | Similar mechanisms are "widely and successfully used in major frameworks such as Qt and in game engines" (pp. 22-23) | No deployments and no outcomes are named for either ecosystem; the standard-library half of the analogy is supported | Unsourced empirical claim |
| 11 (Section 7) | "The overwhelming majority of predicates are trivially non-throwing" (p. 25) | The response grounds the proportion in the authors' experience; no dataset is cited | Unsourced quantitative claim |
| 12 (Section 7) | The CppCon work combined CodeQL with Z3 "to validate a wide range of contracts" (p. 27) | The prototype validates annotated assertion macros only; support for P2900<sup>[2]</sup> specifiers is stated as future hope | Claim outrunning its evidence |
| 13 (Section 7) | "The implementers reported that P2900 is orders of magnitude simpler to support than modules, concepts, reflection, or even lambdas" (p. 27) | The cited implementers' report contains no such comparison; no measurement or attributed quotation exists | Unsourced quantitative claim |
| 14 (Section 6) | "no proposals that included them gained consensus in EWG" (p. 28) | P3097R0<sup>[14]</sup>, which included a requested feature, gained EWG consensus at St. Louis, a history the same section of P3846R1<sup>[1]</sup> records one page later | False statement |
| 15 (Section 7) | "Yet in more than four decades of C++ evolution, no proposal for deep const has ever been brought forward" (p. 31) | P1974R0<sup>[74]</sup> proposes propconst, a language-level deep-const qualifier, and P2670R1<sup>[75]</sup> revises that design line | False statement |
| 16 (Section 7) | The decomposition idea was "subsequently shown to be inadequate for the real-world use cases for contract assertions" (p. 32), citing P1995R1 | P1995R1<sup>[11]</sup> catalogs and polls use cases and does not mention or evaluate P1893R0<sup>[77]</sup> | Citation mismatch |
| 17 (Section 6) | "P2900 has been fully implemented in two major compilers" (p. 33) | The cited report documents P2900R8<sup>[2]</sup> with complementary gaps in the two compilers and code merged into no official branch; upstream Clang recorded P2900R14<sup>[2]</sup> as not implemented at both cutoffs | Claim outrunning its evidence |

The defects are not uniform in kind, and Concern 3 carries two of different kinds, so the counts in this paragraph sum to fifteen across fourteen rows. Three rows contain a false statement, a sentence the record contradicts (Concerns 4, 14, and 15). Six rows contain an unsourced quantitative, causal, empirical, or self-reported claim, where the response asserts a magnitude, a frequency, a cause, a duration, or a deployment success that no cited source establishes (Concerns 1, 3, 7, 9, 11, and 13). One row is an unverifiable existence claim, and the absence is demonstrated positively from the compiler's own option table rather than by a failed search (Concern 2). One row is a citation mismatch, where a citation exists and does not support its sentence (Concern 16). Four rows outrun their evidence, where evidence exists and does not reach the stated conclusion (Concerns 3, 6, 12, and 17). The four responses without a flag - Concerns 5, 8, 10, and 18 - show that the flag is not automatic: each was tested against the same standard, and no material unsupported subclaim was found.

Fourteen responses contain a material unsupported subclaim. Because overall support is rated separately, each defect changes only the subclaim flag: five of the fourteen responses remain Substantially supported and nine remain Mixed, and no flag by itself invalidates the complete response that carries it.

---

## 9. Evidence After the Cutoffs Shows Staleness and Outcomes, Not Retroactive Error

This section holds the evidence that postdates the 2026-03-23 publication-state cutoff, kept separate so that none of it enters the primary score, together with two intervening events between the cutoffs whose classification needs care. Later evidence can establish that the March artifact was stale or that a predicted outcome later occurred; under the rule of Section 2 it changes neither historical score.

The intervening events concern implementations. On 2026-01-28, between the two cutoffs, nine commits were merged into GCC master as one contiguous block, including the base implementation commit "c++, contracts: C++26 base implementation as per P2900R14"<sup>[53]</sup>, and the series was included in GCC 16.1.0, released 2026-04-30<sup>[79]</sup>. The merge is publication-state evidence: it predates the March artifact by two months, so P3846R1's prediction that contract assertions were "much closer to being merged to mainline branches of these compilers" had already come true for GCC when the artifact was supplied, and the artifact's own "with upstreaming in progress" was stale in its authors' disfavor. The GCC 16.1.0 release is subsequent and enters no score. For Clang the record is the reverse: upstream LLVM held no contracts branch at either cutoff, no upstream pull request was located, and the Clang status page recorded P2900R14 as "No" on 2025-11-03 and on 2026-03-23<sup>[52]</sup>; the current status page records both P2900R14 and P3097R3 as "No"<sup>[80]</sup>. The second intervening event qualifies the Concern 2 prototype account without changing it: the repository efcs/contracts-abi-specification appeared on 2025-12-15 and gained a compilable example implementation of the contract-violation entrypoint ABI on 2026-02-22<sup>[81]</sup>. It is not an implementation of the link-time semantic-selection strategy Concern 2 describes, it is not a compiler prototype, and it postdates the November text.

The first subsequent item is a CMake Discourse thread on build-system support for contracts. On 2026-05-08, Lieven de Cock asked how CMake would deal with C++26 contracts: whether the compiler and linker flags would be left to the user or abstracted as a per-target property, and how the evaluation semantic would be specified<sup>[82]</sup>. On 2026-06-29, fifty-two days later, Vito Gamberini, a Kitware developer with a public commit history in CMake, replied in full<sup>[82]</sup>:

> Historically CMake does not add mechanisms for experimental compiler features.
>
> For example, `-lstdc++fs` for `std::filesystem` was never given an abstraction, nor are there plans to support `-freflection`.
>
> It is unlikely `-fcontracts` will be treated differently.
>
> The enforcement semantic is a different story. Once compilers have stable implementations and interfaces for enforcement semantics, some collection of properties will likely need to be implemented for them.

The reply's final paragraph supports P3846R1: on the enabling flag the answer is negative, and on semantic selection - the property Concern 6 of P3846R1 identifies as the one users will regularly configure - the answer is that properties will likely need to be implemented. Both posts postdate the publication-state cutoff, so the thread is subsequent evidence that cannot reduce the Concern 2 or Concern 3 score, and it is reported for chronology only.

The second subsequent item corroborates the Concern 6 count. The live index of implementation-defined behavior in the working draft<sup>[83]</sup>, retrieved 2026-08-14, lists the same seven contract-related entries established from N5008<sup>[65]</sup> in Section 7. It adds nothing to the count: the index is regenerated from the working-draft trunk, it postdates both cutoffs, and it is a mutable document, so it enters no score.

The remaining subsequent items concern the extension proposals named in the Concern 1 and Concern 14 responses. P3097 continued after the March artifact: EWG at Brno polled "Forward P3097R2 to CWG for inclusion in C++29" on 2026-06-10, with SF 8, F 14, N 4, A 1, SA 5 and recorded result "consensus"; the Core Working Group (CWG) approved D3097R3 for a plenary straw poll two days later, and P3097R3 was published on 2026-07-17<sup>[84]</sup>. The progression is subsequent evidence that the re-addition path P3846R1 describes for virtual-function contracts is advancing, aimed at C++29 rather than C++26, and it enters no score. For P3400, the record located for this review establishes only an intervening datum: [P3912R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3912r0.pdf)<sup>[78]</sup>, dated 2025-12-15, describes the labels feature as "[P3400R2], which is currently being pursued for C++29". No later P3400 revision was located or verified, and none is characterized here.

Evidence after each cutoff can establish that the March artifact was stale, as with the completed GCC upstreaming it did not report, or that an outcome later occurred, as with the GCC 16.1.0 release and the P3097 progression. It cannot retroactively change what the public record supported on 2025-11-03 or on 2026-03-23, and nothing in this section enters the primary score.

---

## 10. Eight Limitations Bound What the Counts Can Mean

This section states the limitations of this review, each stated plainly and before a delegate can discover it independently. They bound what the counts in Section 4 can mean.

1. **Byte identity of the November artifact.** No public artifact establishes that text byte-identical to the audited March PDF circulated on 2025-11-03: the canonical 2025 directory contains no P3846R1, the 2025 index lists none, the Internet Archive holds no capture of any P3846R1 before June 2026, and the March copy named in the tracking issue is access-gated and has never been archived. The dual-date treatment of Section 2 is a disclosed method of this review.

2. **The Boost.Build elapsed time.** The "less than an hour" figure is the author's own unwitnessed report about his own work. The quotation is exact; no public record establishes the duration; the commit's parent is eleven weeks earlier, and there is no pull request, issue, or review thread.

3. **The individual attribution in the Concern 10 poll.** No public source records individual votes in any SG21 poll, so the clause "including the author of the proposal" cannot be independently verified. The tally entails the aggregate - with SF 0 and F 0, no attendee voted in favor - and Section 5 attributes the clause to P3846R1 rather than to the public record.

4. **The GCC deferred-selection prototype.** No public code, measurement, or build log was located for the prototype or for the "shown to work reliably" link-time result. The absence is positive rather than a failed search - GCC's complete contract option table contains no link-time, load-time, or runtime selection option at either cutoff - with one stated recall gap: the gcc.gnu.org web archives sit behind an interactive bot filter, so a gcc-patches posting describing the prototype would not have been found.

5. **The characterization of the 2016 Ristow report.** Whether the report counts as a "user complaint" in P3846R1's sense is a judgment. The underlying facts are verified - the reporter is a Sony toolchain engineer and LLVM contributor, the report was confined to variadics, and it was resolved in 2018 without relaxing the conservatism - and no measurement of the narrow indirect-call case exists on either side.

6. **Concerns not re-verified by a dedicated task.** Concerns 1, 7, 8, 13, and 15 were not re-verified by a dedicated verification task; their ratings rest on the quoted sentences and the analysis of the cited sources recorded in Sections 5 through 7. A character-level recheck of those five was not performed, and one could in principle move a rating.

7. **Ratings are judgments, not measurements.** Another reviewer applying the same standard could place a boundary case one category higher or lower, particularly Concerns 2, 7, 9, 11, and 17. The concern-level evidence matters more than the aggregate count.

8. **Confinement to the public record.** Where P3846R1 relies on discussion that was not published, no public source could be located, and that absence is reported as an absence rather than treated as evidence of anything.

The aggregate result is bounded at three points: the artifact chronology, because the November text cannot be byte-compared (item 1); prototype completeness, because the deferred-selection record is an absence with one stated recall gap (item 4); and the qualitative sufficiency judgments, because the ratings are analytical and five concerns were not independently re-verified (items 6 and 7). Within those bounds, the concern-level findings rest on quoted sentences and cited artifacts, and the counts in Section 4 carry these eight qualifications.

---

## 11. Conclusion: The Objecting Position Prevails on Closure

The eighteen evidence units establish what the counts alone cannot. Each unit quotes a response's load-bearing sentences, tests them against the public record at the two cutoffs, and records support, subclaim, and resolution separately, so every aggregate below traces to a quotation and a dated artifact. On support, the record favors the authors of P3846R1: two responses are Supported, six are Substantially supported, ten are Mixed, and none is Not supported or Contradicted. Fourteen responses contain a material unsupported subclaim: three false statements, six unsourced claims of magnitude, frequency, cause, duration, or deployment success, one unverifiable existence claim, one citation mismatch, and four claims that outrun their evidence, with Concern 3 carrying two defects of different kinds. On resolution, one objection is Answered, sixteen are Partly answered, and one is Not resolved. P3846R1 presents the objections as previously addressed; the units show what that presentation is worth on the record, which fully supports only the responses to Concerns 5 and 10 and resolves only Concern 10.

The record also contains four points in P3846R1's favor. Under Concern 2, the sentence that reads as a general worst-case claim is scoped to the naive implementation strategy and excludes the compiler-bug case by construction, so the miscompilation documented under Concern 4 does not contradict it. Under Concern 16, the disputed sentence carries a citation; the defect is that the cited paper does not support the sentence. The Concern 18 response names all four national body comments seeking decoupling in its first sentence. Between the two cutoffs, on 2026-01-28, the base implementation of P2900R14 was merged into GCC master<sup>[53]</sup>: the Concern 17 prediction had already come true for GCC when the March artifact was supplied, the artifact's "with upstreaming in progress" was stale in its authors' own disfavor, and no Clang upstreaming had occurred at either cutoff<sup>[52]</sup>.

Three groups can build on this assessment. Authors evaluating contract designs can build on the supported mechanisms: the bounded modules claim of Concern 5, demonstrated in GCC's module serialization; the consecutive-assertion mechanism of Concern 10, backed by a recorded committee decision; the const-ification migration evidence of Concern 8. Authors preparing a future reassertion can work from the fourteen flagged subclaims, which name the sentences to source, qualify, or remove. Reviewers assessing other response papers can reuse the dual-cutoff rule and the three independent fields, which transfer to any document tested against a dated public record.

Sixteen objections remain partly answered and one remains not resolved, and because a committee vote establishes procedural disposition without settling a technical question, those objections stay open for any future contracts design, whatever its vehicle. The gaps are specific and on the record: predictable control of the evaluation semantic across precompiled dependencies is undemonstrated in any public artifact, no implementation or measurement exists on the non-Itanium interface where infeasibility was reported, and no frequency data supports the claimed rarity of destructive predicate side effects. The counts carry the eight qualifications of Section 10, and within those bounds the concern-level findings rest on quoted sentences and cited artifacts.

---

## Disclosure

The author provides information and serves at the pleasure of the committee.

The author is president of the C++ Alliance and maintains coroutine-native I/O libraries under it.

This paper assesses P3846R1's eighteen responses against the public record at the two cutoffs stated in Section 2, a record broader than the sources P3846R1 itself cites: the responses are tested against the working draft, compiler repositories, bug trackers, build-system history, and the public poll record, whether or not P3846R1 cites them. It proposes no wording and requests no poll.

The C++ Alliance has published a position, in [P4238R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4238r0.pdf)<sup>[25]</sup>, that the National Bodies vote No on the C++26 DIS ballot and return the draft over Contracts. This paper's findings support that position, and the author is a co-author of P4238R0. This co-authorship is a material stake in the question under assessment.

One limitation of the method is the author's own: the audit tests the sentences the author selected as load-bearing in each response, and a different selection could produce a different distribution of subclaim flags even if every verdict here were upheld.

This paper was prepared with the assistance of generative tools. The author is responsible for its content.

This paper asks for nothing.

---

## References

[1] [P3846R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p3846r1.pdf) - "C++26 Contract Assertions, Reasserted" (Timur Doumler, Joshua Berne, Ga&scaron;per A&zcaron;man, Peter Bindels, Peter Dimov, Louis Dionne, Eric Fiselier, Mungo Gill, Pablo Halpern, Tom Honermann, Corentin Jabot, John Lakos, Nevin Liber, Lisa Lippincott, Ryan McDougall, Jason Merrill, Roger Orr, Nina Dinka Ranns, Ren&eacute; Ferdinand Rivera Morell, Oliver Rosten, Iain Sandoe, Hui Xie, 2025).

[2] [P2900R14](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p2900r14.pdf) - "Contracts for C++" (Joshua Berne, Timur Doumler, Andrzej Krzemie&nacute;ski, 2025).

[3] [P3835R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3835r0.html) - "Contracts make C++ less safe - full stop" (John Spicer, Ville Voutilainen, Jos&eacute; Daniel Garc&iacute;a S&aacute;nchez, 2025).

[4] [P3829R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3829r0.pdf) - "Contracts do not belong in the language" (David Chisnall, John Spicer, Ville Voutilainen, Gabriel Dos Reis, Jos&eacute; Daniel Garc&iacute;a S&aacute;nchez, 2025).

[5] [P3849R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3849r0.pdf) - "SIS/TK611 considerations on Contract Assertions" (Harald Achitz, 2025).

[6] [P3506R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3506r0.pdf) - "P2900 Is Still Not Ready for C++26" (Gabriel Dos Reis, 2025).

[7] [P3878R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3878r0.html) - "C++26 Contracts are not a good fit for standard library hardening" (Ville Voutilainen, Jonathan Wakely, John Spicer, Stephan T. Lavavej, 2025).

[8] [N5031](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/n5031.pdf) - "WG21 November 2025 Kona Hybrid meeting Minutes of Meeting" (Nina Dinka Ranns, 2025).

[9] [P3859R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3859r0.html) - "Assertions are not necessarily for changing program behavior" (Andrzej Krzemie&nacute;ski, 2025).

[10] [P3896R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3896r0.html) - "Design goals for a contract support facility" (Andrzej Krzemie&nacute;ski, 2025).

[11] [P1995R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p1995r1.html) - "Contracts - Use Cases" (Joshua Berne, Timur Doumler, Andrzej Krzemie&nacute;ski, Ryan McDougall, Herb Sutter, 2020).

[12] [LLVM Issue 28170](https://github.com/llvm/llvm-project/issues/28170) - "Calls to empty variadic functions in comdat no longer optimized out" (Warren Ristow, 2016).

[13] [P3626R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3626r0.pdf) - "Make predicate exceptions propagate by default" (Timur Doumler, 2025).

[14] [P3097R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p3097r0.pdf) - "Contracts for C++: Support for Virtual Functions" (Timur Doumler, Joshua Berne, Ga&scaron;per A&zcaron;man, 2024).

[15] [cplusplus/papers issue 2455](https://github.com/cplusplus/papers/issues/2455) - P3846 tracking issue, cplusplus/papers (2025).

[16] [WG21 2026 paper index](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/) - Official WG21 paper index, 2026 directory, retrieved 2026-09-02.

[17] [WG21 2025 paper index](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/) - Official WG21 paper index, 2025 directory, retrieved 2026-09-02.

[18] [P3893R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3893r0.pdf) - "The CppCon 2025 Talk on Contracts and CodeQL in Context" (Mike Fairhurst, 2025).

[19] [P3878R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3878r1.html) - "Standard library hardening should not use the 'observe' semantic" (Ville Voutilainen, Jonathan Wakely, John Spicer, Stephan T. Lavavej, 2025).

[20] [N4985](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/n4985.pdf) - "WG21 June 2024 Hybrid meeting Minutes of Meeting" (Nina Dinka Ranns, 2024).

[21] [N5000](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/n5000.pdf) - "WG21 November 2024 Hybrid meeting Minutes of Meeting" (Nina Dinka Ranns, 2024).

[22] [N5007](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/n5007.pdf) - "WG21 February 2025 Hybrid meeting Minutes of Meeting" (Nina Dinka Ranns, 2025).

[23] [cplusplus/papers issue 1648](https://github.com/cplusplus/papers/issues/1648#issuecomment-2651224887) - EWG Hagenberg contracts polls, posted by the EWG chair (JF Bastien, 2025).

[24] [P2899R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p2899r1.pdf) - "Contracts for C++ - Rationale" (Joshua Berne, Timur Doumler, Rostislav Khlebnikov, Andrzej Krzemie&nacute;ski, 2025).

[25] [P4238R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4238r0.pdf) - "Returning C++26 for the Evaluation It Skipped" (Vinnie Falco, Ville Voutilainen, Jos&eacute; Daniel Garc&iacute;a S&aacute;nchez, John Spicer, 2026).

[26] [P3321R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p3321r0.pdf) - "Contracts Interaction With Tooling" (Joshua Berne, 2024).

[27] [gcc/cp/module.cc at 436aff90](https://raw.githubusercontent.com/villevoutilainen/gcc/436aff90fc62a9637f475c2ea34840b1e9bc1a79/gcc/cp/module.cc) - GCC contracts development fork (villevoutilainen/gcc), module serialization source at the branch head of 2025-10-19.

[28] [GCC commit 64674a2](https://github.com/gcc-mirror/gcc/commit/64674a295b63f46ac9b6776348ae6bbda63fd1ef) - "c++, contracts: Allow contract checks as outlined functions." (Nina Ranns, Iain Sandoe, Ville Voutilainen, 2026).

[29] [P3582R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3582r0.html) - "Observed a contract violation? Skip subsequent assertions!" (Andrzej Krzemie&nacute;ski, 2025).

[30] [cplusplus/papers issue 2225](https://github.com/cplusplus/papers/issues/2225#issuecomment-2641031934) - SG21 poll on forwarding P3582R0, posted by the SG21 chair (Timur Doumler, 2025).

[31] [LLVM Issue 27148](https://github.com/llvm/llvm-project/issues/27148) - "Doing certain kinds of IPO over comdat functions is unsound" (Sanjoy Das, 2016).

[32] [LLVM commit 5ce3272](https://github.com/llvm/llvm-project/commit/5ce32728330fe7684f24d1b9c418c152db988830) - "Don't IPO over functions that can be de-refined" (Sanjoy Das, 2016).

[33] [GCC Bug 70018](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=70018) - "[6 Regression] Possible issue around IPO and C++ comdats discovered as pure/const" (Sanjoy Das, 2016).

[34] [GCC Bug 121936](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=121936) - "[14/15/16/17 Regression] Invalid optimisation (at O3) based on bodies of vague linkage functions" (Iain Sandoe, 2025).

[35] [GCC fork commit 100e6a9](https://github.com/villevoutilainen/gcc/commit/100e6a95b732638952de68bc087ec16efcb0b320) - "c++, contracts: Add a noipa wrapper around terminate calls." (Nina Ranns, Iain Sandoe, 2025).

[36] [GCC commit cac7958](https://github.com/gcc-mirror/gcc/commit/cac79586e1ab11fdb5480d7d1d93a48181fb3973) - "c++, contracts: Work around GCC IPA bug, PR121936 by wrapping terminate." (Nina Ranns, Iain Sandoe, 2026).

[37] [P3499R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3499r1.pdf) - "Exploring strict contract predicates" (Timur Doumler, Lisa Lippincott, Joshua Berne, 2025).

[38] [PRE31-C. Avoid side effects in arguments to unsafe macros](https://wiki.sei.cmu.edu/confluence/display/c/PRE31-C.+Avoid+side+effects+in+arguments+to+unsafe+macros) - SEI CERT C Coding Standard (Carnegie Mellon University Software Engineering Institute).

[39] [SonarQube S3346](https://github.com/SonarSource/sonar-dotnet/releases/tag/5.11.0.1761) - "Expressions used in Debug.Assert should not produce side effects", SonarQube static-analysis rule for C#, introduced in sonar-dotnet 5.11 (SonarSource, 2017); retrieved 2026-09-02.

[40] [PVS-Studio V6055](https://pvs-studio.com/en/docs/warnings/v6055/) - PVS-Studio static-analysis diagnostic for Java, retrieved 2026-09-02.

[41] [P3336R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p3336r0.pdf) - "Usage Experience for Contracts with BDE" (Joshua Berne, 2024).

[42] [P3261R2](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p3261r2.pdf) - "Revisiting const-ification in Contract Assertions" (Joshua Berne, 2024).

[43] [P3460R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p3460r0.pdf) - "C++ Contracts Implementers Report" (Eric Fiselier, Nina Dinka Ranns, Iain Sandoe, 2024).

[44] [P3071R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p3071r1.html) - "Protection against modifications in contracts" (Jens Maurer, 2023).

[45] [cplusplus/papers issue 1732](https://github.com/cplusplus/papers/issues/1732#issuecomment-2181379281) - SG21 adoption poll for D3071R1, posted by the SG21 chair (Timur Doumler, 2024).

[46] [cplusplus/papers issue 2062](https://github.com/cplusplus/papers/issues/2062#issuecomment-2485786122) - EWG Wroc&lstrok;aw poll on removing const-ification, posted by the EWG chair (JF Bastien, 2024).

[47] [P2811R7](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p2811r7.pdf) - "Contract-Violation Handlers" (Joshua Berne, 2023).

[48] [cplusplus/papers issue 1822](https://github.com/cplusplus/papers/issues/1822#issuecomment-2197580410) - EWG St. Louis polls on P3097R0, posted by the EWG chair (JF Bastien, 2024).

[49] [Compiler Explorer C++ compiler configuration](https://github.com/compiler-explorer/compiler-explorer/blob/main/etc/config/c%2B%2B.amazon.properties) - compiler-explorer/compiler-explorer, `etc/config/c++.amazon.properties`; the contracts toolchains appear identically in the versions at both cutoffs.

[50] [P3191R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p3191r0.pdf) - "Feedback on the scalability of contract violation handlers in P2900" (Louis Dionne, Yeoul Na, Konstantin Varlamov, 2024).

[51] [P3471R4](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3471r4.html) - "Standard library hardening" (Konstantin Varlamov, Louis Dionne, 2025).

[52] [Clang C++ support status](https://github.com/llvm/llvm-project/blob/e65522e596522faca391eea0adb440542b9f8f15/clang/www/cxx_status.html) - Clang C++ support status page, version at the 2025-11-03 cutoff; the version at the 2026-03-23 cutoff records the same status.

[53] [GCC commit c928dc51](https://github.com/gcc-mirror/gcc/commit/c928dc51966d) - "c++, contracts: C++26 base implementation as per P2900R14." (Iain Sandoe, 2026).

[54] [P2877R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p2877r0.pdf) - "Contract Build Modes, Semantics, and Implementation Strategies" (Joshua Berne, Tom Honermann, 2023).

[55] [P3500R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3500r1.pdf) - "Are Contracts 'safe'?" (Timur Doumler, Ga&scaron;per A&zcaron;man, Joshua Berne, Ryan McDougall, 2025).

[56] [Safer at Any Speed: Automatic Context-Aware Safety Enhancement for Rust](https://doi.org/10.1145/3485480) - Proceedings of the ACM on Programming Languages, Volume 5, Issue OOPSLA, Article 103 (Natalie Popescu, Ziyang Xu, Sotiris Apostolakis, David I. August, Amit Levy, 2021).

[57] [Rust in Android: move fast and fix things](https://blog.google/security/rust-in-android-move-fast-fix-things/) - Google Security Blog (Jeff Vander Stoep, 2025).

[58] [P3267R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p3267r1.html) - "C++ contracts implementation strategies" (Peter Bindels, Tom Honermann, 2024).

[59] [gcc/c-family/c.opt at 436aff90](https://raw.githubusercontent.com/villevoutilainen/gcc/436aff90fc62a9637f475c2ea34840b1e9bc1a79/gcc/c-family/c.opt) - GCC contracts development fork (villevoutilainen/gcc), compiler option table at the branch head of 2025-10-19.

[60] [gcc/c-family/c.opt at bd0dde45](https://raw.githubusercontent.com/gcc-mirror/gcc/bd0dde45a3d0cd9fbf88b4b20515d477c555c335/gcc/c-family/c.opt) - GCC master compiler option table at the last commit touching the file on or before the 2026-03-23 cutoff.

[61] [efcs/contracts-abi](https://github.com/efcs/contracts-abi) - Contract-violation entrypoint ABI design document (Eric Fiselier, 2025); README and .gitignore only, frozen since 2025-06-27.

[62] [Boost.Build commit 3b20a4e](https://github.com/boostorg/build/commit/3b20a4e16594b19a38f006a7af051c775bf0e1c9) - "Add initial support for {CPP}-26 Contracts for GCC based toolsets (like clang)." (Ren&eacute; Ferdinand Rivera Morell, 2025).

[63] [grafikrobot/cpp_contracts_example](https://github.com/grafikrobot/cpp_contracts_example) - C++ Contracts example repository (Ren&eacute; Ferdinand Rivera Morell, 2025).

[64] [B2 release record](https://github.com/bfgroup/b2/releases) - bfgroup/b2; release 5.4.0, the first tagged release carrying contracts support, published 2025-12-20.

[65] [N5008](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/n5008.pdf) - "Working Draft, Programming Languages - C++" (Thomas K&ouml;ppe, 2025).

[66] [N4950](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/n4950.pdf) - "Working Draft, Standard for Programming Language C++" (Thomas K&ouml;ppe, 2023).

[67] [P3591R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3591r0.pdf) - "Contextualizing Contracts Concerns" (Joshua Berne, Timur Doumler, 2025).

[68] [P3909R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3909r0.html) - "Contracts should go into a White Paper - even at this late point" (Ville Voutilainen, 2025).

[69] [P4009R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4009r0.html) - "A proposal for solving all of the contracts concerns" (Ville Voutilainen, 2026).

[70] [P3386R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p3386r1.pdf) - "Static Analysis of Contracts with P2900" (Joshua Berne, 2024).

[71] [advanced-security/codeql-contracts-smt-z3](https://github.com/advanced-security/codeql-contracts-smt-z3) - SMT constraint solving in CodeQL with Z3; frozen since 2025-09-19.

[72] [P4020R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4020r0.html) - "Concerns about contract assertions" (Andrzej Krzemie&nacute;ski, 2026).

[73] [SD-4](https://isocpp.org/std/standing-documents/sd-4-wg21-practices-and-procedures) - "WG21 Practices and Procedures" (Guy Davidson, 2026).

[74] [P1974R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p1974r0.pdf) - "Non-transient constexpr allocation using propconst" (Jeff Snyder, Louis Dionne, Daveed Vandevoorde, 2020).

[75] [P2670R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p2670r1.html) - "Non-transient constexpr allocation" (Barry Revzin, 2023).

[76] [P2000R4](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2000r4.pdf) - "Direction for ISO C++" (Howard Hinnant, Roger Orr, Bjarne Stroustrup, Daveed Vandevoorde, Michael Wong, 2022).

[77] [P1893R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1893r0.pdf) - "Proposal of Contract Primitives" (Andrew Tomazos, 2019).

[78] [P3912R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3912r0.pdf) - "Design considerations for always-enforced contract assertions" (Timur Doumler, Joshua Berne, Ga&scaron;per A&zcaron;man, Oliver Rosten, Lisa Lippincott, Peter Bindels, 2025).

[79] [GCC 16 release changes](https://gcc.gnu.org/gcc-16/changes.html) - GCC 16 release series changes page; the C++26 feature list records 'P2900R14, Contracts (PR119061)'; GCC 16.1.0 released 2026-04-30; retrieved 2026-09-02.

[80] [Clang C++ support status](https://github.com/llvm/llvm-project/blob/main/clang/www/cxx_status.html) - Clang C++ support status page, current version, retrieved 2026-09-02; records P2900R14 and P3097R3 as not implemented.

[81] [efcs/contracts-abi-specification](https://github.com/efcs/contracts-abi-specification) - Contracts ABI specification (Eric Fiselier); created 2025-12-15, with a compilable example implementation of the contract-violation entrypoint ABI added 2026-02-22.

[82] [C++26 contracts](https://discourse.cmake.org/t/c-26-contracts/15644) - CMake Discourse thread 15644; question posted 2026-05-08 by Lieven de Cock, reply posted 2026-06-29 by Vito Gamberini, Kitware.

[83] [Index of implementation-defined behavior](https://eel.is/c++draft/impldefindex) - C++ working draft, live eel.is rendering, retrieved 2026-08-14.

[84] [cplusplus/papers issue 1822](https://github.com/cplusplus/papers/issues/1822#issuecomment-4671031912) - EWG Brno poll on forwarding P3097R2 to CWG, posted by Jeff Snyder; the same tracking issue records CWG's approval of D3097R3 for a plenary straw poll on 2026-06-12 and the publication of P3097R3 on 2026-07-17 (2026).
