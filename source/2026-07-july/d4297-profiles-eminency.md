---
title: "A Paper That Changes Nothing Is Settling Everything: P3100, Profiles, and the Consensus Ratchet"
document: P4297R0
date: 2026-07-07
intent: info
audience: WG21
reply-to:
  - "Vinnie Falco <vinnie.falco@gmail.com>"
---

## Abstract

A proposal that by its own account changes nothing is settling which safety framework governs C++.

[P3100R6](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p3100r6.pdf)<sup>[1]</sup> defines Profiles as "a named configuration preset" over its framework of implicit contract assertions, and states that all existing implementations of C++ already conform to its proposed wording. This paper examines how that architectural claim is advancing. The method is the public record: the proposal's own text and self-reported history, the published poll trail, vendor documentation, and a disclosed full-text search of the 2025 and 2026 mailings. The findings: six self-reported polls, none of which adopts anything; a per-clause review process that was voted at Sofia and, in the proposal's own words, never ran; a no-normative-effect design that lowers the stakes of every poll taken about it; a deployment record that runs opposite to the proposed hierarchy; and no published paper that contests the recasting. The conclusion is stated plainly in Section 10: the question of which framework governs safety configuration is being settled by accretion, the published record does not support the direction being locked in, and a question of this order is worth deciding as a question.

---

## Revision History

### R0: July 2026

- Initial version.

---

## 1. Disclosure

The author provides information and serves at the pleasure of the committee.

The author is the founder of the C++ Alliance and maintains Boost.Beast and related networking libraries. He is not an author of any Profiles paper and maintains no Profiles implementation. His published work includes [P4094R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4094r1.pdf)<sup>[2]</sup>, an examination of the P0443R14 unification history, to which this paper is a companion.

This paper takes no position on the technical merits of implicit contract assertions as a mechanism. Section 3 records what the proposal achieves before anything else is said about it.

The author works only from the published record: papers in the public mailings, public vendor documentation, and public meeting artifacts. Committee-internal deliberations - wiki minutes, reflector threads, hallway conversations - are neither cited nor characterized in this paper. This is a real limitation: the room may contain answers the record does not.

This paper uses machine-assisted drafting.

This paper asks for nothing.

---

## 2. Introduction

Two bodies of safety work are converging on one question. [P3100R6](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p3100r6.pdf)<sup>[1]</sup> ("A framework for systematically addressing undefined behaviour in the C++ Standard", Doumler and Berne) enumerates 80 cases of core language undefined behavior and proposes to guard the 77 runtime-checkable cases with implicit contract assertions, configured through the Labels facility proposed in [P3400R3](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p3400r3.pdf)<sup>[3]</sup>. The Profiles work - the framework in [P3589R2](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3589r2.pdf)<sup>[4]</sup> (Dos Reis) and the individual profiles in [P3081R2](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3081r2.pdf)<sup>[5]</sup> (Sutter), [P3984R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p3984r0.pdf)<sup>[6]</sup> (Stroustrup), and others - proposes named, enforceable guarantees with the framework as the user-facing configuration mechanism. The question the two bodies of work now share: which one is the architecture, and which one is a feature inside the other's architecture?

P3100R6 answers that question in its Section 4.4: Profiles are "a higher-level feature building on top of" its tools, and a concrete profile is "a named configuration preset". This paper is about how that answer is advancing through the committee - not through the paper's technical content, but through its procedural posture.

This paper contributes five things:

1. It documents the recasting from the proposal's own text (Section 3).
2. It identifies the proposal's no-normative-effect posture and the procedural effect that posture has (Section 4).
3. It reconstructs the proposal's advancement from its own self-reported poll history, and names the mechanism that advancement follows (Section 5).
4. It assembles the public deployment record of the two candidate architectures (Section 6).
5. It verifies, by a disclosed and re-runnable method, that no published WG21 paper contests the recasting, and that none outside the proposal's own companion papers engages it (Section 7).

One evaluative standard is used, and it is stated now so the reader can weigh everything below against it: seniority among safety frameworks is earned by deployment. The standard is the committee's own, written in its direction paper and restated in this problem domain by three senior committee members (Section 6 quotes both). Readers who reject that standard will find the facts in this paper unchanged but the conclusion unearned; readers who accept it will find the conclusion follows.

The conclusion, in brief: a first-order architecture decision is being settled through an accumulation of low-stakes polls about a paper that, by its own account, changes nothing - and the published record, measured by the committee's own standard, supports the opposite hierarchy. The full statement, with the evidence attached, is Section 10.

---

## 3. The Claim: Profiles as a Preset over the Framework

P3100R6 earns three acknowledgments before any analysis. First, its Appendix A enumerates every case of explicit core language undefined behavior in the standard - 80 cases, categorized and classified by diagnosability, and useful to every safety effort regardless of vehicle. Second, its five evaluation semantics - ignore, observe, enforce, quick-enforce, and the new assume - give one coherent vocabulary to a landscape of existing vendor mechanisms: it maps `-ftrapv`, `-fwrapv`, and the sanitizers into that model<sup>[1]</sup>. Third, its wording is engineered for backward compatibility: no existing implementation is invalidated by it. These are real achievements.

The claim this paper examines appears in Section 4.4 of P3100R6, under "Configuration":

> It therefore seems logical to define Profiles as a higher-level feature building on top of these three basic tools (see Figure 4). Given that these three features are configurable, a concrete profile could be defined as being a named configuration preset for these features.

Figure 4 of P3100R6 draws the architecture, and its caption states it:

> Figure 4: Overview of the proposed holistic strategy for removing UB from the C++ language: seven orthogonal tools plus Profiles as a higher-level feature specified on top of these tools. The red rectangle in the centre illustrates the scope of the proposal in Sections 5 and 6 of this paper.

Section 7.2 of P3100R6 then states what the layering means for the Profiles framework specifically. Granular control of evaluation semantics must belong to exactly one feature:

> For granular, in-source control of the evaluation semantics of implicit contract assertions, we need to agree whether this happens via directives such as the ones proposed in [P3400R3] and shown here, or by using the syntax proposed in the Profiles framework as proposed in [P3589R2]. If we want to have both, we need to specify one in terms of the other to avoid an incoherent and messy design.

And it offers Profiles two futures: a profile "can be defined as essentially a declaration that expands to [P3400R3] directives", or Profiles can be redesigned "as an auditing feature rather than a configuration feature", where a profile no longer configures anything and instead renders a program ill-formed when configuration chosen elsewhere violates its guarantees<sup>[1]</sup>.

The proposal has already acted on this architecture once. Section 5.6 of P3100R6 withdraws the `detection_mode` enumerators that [P3081R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3081r1.pdf)<sup>[7]</sup> had adopted from earlier revisions of the proposal into its own proposed wording:

> ... unlike earlier revisions of this paper and unlike [P3081R1], which adopted its library API from those earlier revisions, we no longer propose to add new enumerators to the enumeration detection_mode to encode the category of error (Initialization, Bounds, and so on); instead, this encoding can be accomplished more effectively and flexibly via Labels (see Section 7.1).

P3081R2's proposed wording depends on enumerators that the substrate it adopted them from no longer proposes. No later revision of that paper has appeared since R2 (February 2025)<sup>[5]</sup>; the withdrawal stands unanswered in print. Whoever owns the configuration mechanism owns the design, and the proposal is already exercising that ownership.

Together, Section 4.4, Figure 4, Section 5.6, and Section 7.2 are a seniority claim. The framework is the substrate; Profiles are a preset over it, or an auditor beside it.

---

## 4. The Posture: A Change That Changes Nothing

Section 5.2 of P3100R6 states the proposal's conformance posture:

> Note that no implementation is actually required to implement these checks: a valid implementation choice is to make all 77 cases always have the ignore semantic. It follows that all existing implementations of C++ are already conforming with this wording transformation.

Every evaluation semantic is implementation-defined per case; there is, in the proposal's own wording, "no guarantee that there is a way to select any particular semantic"<sup>[1]</sup>. A conforming implementation may do nothing at all.

This design has a legitimate engineering rationale: it makes the wording adoptable without breaking any implementation. It also has a procedural effect, and the procedural effect is the subject of this paper. A proposal that changes nothing looks like it decides nothing. A poll about a paper with no normative effect reads as low-stakes: no vendor is compelled, no code breaks, nothing ships differently the next day. Each vote is easy to cast and easy to justify. What the votes accumulate toward - the architecture in Figure 4, the configuration ownership of Section 7.2 - has appeared on a ballot exactly once, as "a good basis" for a white paper that no longer exists (Table 1, poll 4); it has never been polled as the design question for the proposal now advancing. No normative effect is not no effect.

---

## 5. The Procedure: Six Polls, None of Them Adoption

P3100R6's Section 2, "History and polls", self-reports what it presents as the proposal's complete committee trail; no independent public enumeration exists to check it against. Table 1 reproduces every poll in it, quoted from the paper, with the tallies the paper itself gives.

Table 1: All six polls in P3100R6's self-reported history (its Section 2). Poll text abridged only by ellipsis; tallies and result labels as printed in P3100R6. The rightmost column classifies what each poll asked for.

| # | Body, meeting | Poll (as quoted in P3100R6) | SF/F/N/A/SA | Result (as printed) | What was asked |
|---|---|---|---|---|---|
| 1 | SG21, Wroc&lstrok;aw, 2024-11 | "We support the direction of P3100R1 and encourage the authors to come back with a fully specified proposal." | 19/6/0/0/0 | Consensus | Direction |
| 2 | EWG, Hagenberg, 2025-02 | "Pursue a language safety white paper in the C++26 timeframe containing systematic treatment of core language Undefined Behavior in C++, covering Erroneous Behavior, Profiles, and Contracts. Appoint Herb and Ga&scaron;per as editors." | 32/31/6/4/4 | Consensus | A vehicle and its editors |
| 3 | EWG, Sofia, 2025-06 | "EWG encourages more work on P3100R2 and wants a step-by-step systematic review of P3100R2 to do per-clause approval for inclusion in the core language UB whitepaper (in telecons)" | 15/27/2/0/0 | Consensus | A review process |
| 4 | EWG, Sofia, 2025-06 | "EWG agrees that Timur's (magic) slide 53 in P3754R0 is a good basis for the core language UB whitepaper, and asks that the Whitepaper editors make it so." | 14/28/2/1/0 | Consensus | A diagram as a basis for the vehicle |
| 5 | SG23, Kona, 2025-11 | "SG23 supports the direction of P3100R4 and recommends its inclusion in C++29" | 15/12/1/1/2 | Consensus | Direction and a target |
| 6 | EWG, Croydon, 2026-03 | "Update P3100R5 by applying the presented rules to all cases of runtime-checkable UB in the standard, as listed in appendix A, and bring it back to EWG for case-by-case wording review" | 39/19/5/2/1 | Strong consensus | A wording update across all 77 cases, and more review |

The Hagenberg poll and its tally are independently public in [P3656R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3656r1.pdf)<sup>[8]</sup>, the white paper process document by its appointed editors. The remaining tallies are as self-reported in P3100R6.

Three observations follow from the table, each checkable against it.

First, none of the six polls adopts anything. Two are direction polls, one creates a vehicle and appoints its editors, one endorses a diagram as a basis for that vehicle, one recommends a target, and one requests a wording update and further review. The one poll that established a content-approval process - Sofia's per-clause approval in telecons - is the process the paper itself reports never ran:

> However, the telecons for per-clause approval of this paper into the white paper were never scheduled, and the pursuit of the white paper as a ship vehicle has stalled. To make progress, we decided to instead target C++29 with the present proposal.

Second, the endorsements attach to a vehicle that no longer exists. The Hagenberg poll created a white paper "in the C++26 timeframe" covering "Erroneous Behavior, Profiles, and Contracts" jointly, under two appointed editors. The Sofia polls endorsed the strategy diagram and the per-clause process for that white paper. The white paper stalled; the proposal retargeted to C++29 as an ordinary IS-track paper; and the endorsements travel with it in its history section. A poll about the basis of a joint, editor-curated document is now part of the case for a single-author-team proposal on the standard track - a vehicle the poll never mentioned.

Third, the compression is already visible inside the proposal's own text. Section 1 of P3100R6 states: "The proposed design has been reviewed and approved by SG21, SG23, and EWG." The reader who has just read Table 1 can weigh the word "approved" against what each poll asked. Within Section 2 itself, the prose introducing the Kona poll says SG23 "approved it with strong consensus"; the poll box directly beneath records "Result: Consensus", with one Against and two Strongly Against. Direction becomes approval; consensus becomes strong consensus. That compression is the mechanism this section names, operating in the paper's own pages.

The mechanism is observable in this trail. Each recorded consensus becomes the floor for the next question: the Hagenberg vehicle poll builds on Wroclaw's direction, the Sofia polls build on Hagenberg's vehicle, and the Kona and Croydon polls carry the accumulated record forward. A sequence of individually modest polls - direction, vehicle, basis, target, review - functions as a ratchet: no single poll decides the architecture, and every poll raises the cost of contesting it. Reversal is possible; the committee removed the earlier Contracts design from the C++20 working draft after adopting it<sup>[9]</sup>, and it walked away from P0443R14 (below). What both reversals cost was years, and both arrived only after implementation and deployment pressure from outside the room - the corrective input a proposal with no implementations does not generate. Case-by-case wording review completes the pattern. The Croydon poll routes the proposal into review of 77 cases, one at a time. Each case review is a small, technical, reasonable question. None of them is the question of Section 4.4. When the last case is approved, the architecture will have been settled without ever having been polled as its own question - its one ballot appearance was as a "good basis" for the abandoned white paper.

The committee has run the consensus-without-deployment pattern before. In 2014, SG1 directed the authors of three deployed executor models - networking, GPU dispatch, and thread pools - to unify them into a single abstraction. The result, [P0443R14](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p0443r14.html)<sup>[10]</sup>, absorbed fourteen published revisions of committee direction over four years and was never adopted; the effort was closed in 2021 in favor of a successor, and no unified executor model was ever deployed as designed. [P4094R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4094r1.pdf)<sup>[2]</sup> documents that history and its costs. The property that record documents: a unification advancing on committee consensus while deployment evidence for the unified design stayed at zero, with the costs surfacing afterward, in the field. A paper that changes nothing is settling everything - and the settling is the kind that history shows is corrected late, expensively, and from outside.

---

## 6. The Record: What Ships and What Does Not

The standard this paper measures by comes from the committee's direction paper. [P2000R5](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p2000r5.pdf)<sup>[11]</sup> ("Direction for ISO C++", the Direction Group, February 2026): "We change the language and standard library by gradually building on previous work or by providing a better alternative to an existing feature." Either disjunct grounds the same test - a proposal is measured against the practice that exists, whether it builds on that practice or claims to better it. Three committee members stated the test in this exact domain: [P3608R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3608r0.html)<sup>[12]</sup> (Voutilainen, Wakely, Dos Reis, January 2025) observes that "the standard library hardening is existing practice, and comes with very positive field experience reports. Two out of three of our major library vendors already ship it", and closes: "Ship the stable and mature existing practice. Don't ship wild guesses." Its proposed C++26 shipping set was the general profiles framework, library hardening ([P3471R4](https://open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3471r4.html)<sup>[13]</sup>), and one profile enabling that hardening - the framework in the set, the substrate machinery out of it.

The provenance of this standard is disclosed rather than assumed away: P2000R5, the safety opinion [P2759R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p2759r1.pdf)<sup>[14]</sup>, and the 2026 statement [P3970R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p3970r0.pdf)<sup>[15]</sup> are Direction Group opinions rather than committee-adopted policy, and Stroustrup, an author on one side of the dispute this paper examines, is a co-author of all three. The reader should weigh the standard knowing that. The deployment facts below stand on vendor documentation and do not depend on it.

The Profiles direction has its own trail of direction signals, and this paper discounts them exactly as it discounts Table 1. P2759R1 and P3970R0 name Profiles as the direction to build on, a lineage running through [P2687R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2687r0.pdf)<sup>[16]</sup> (Stroustrup and Dos Reis, 2022) back to the C++ Core Guidelines, announced in September 2015<sup>[17]</sup>. At Issaquah in February 2023, the direction as presented drew a recorded vote of 47 for and 2 against, reported in the preface of [P2816R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p2816r0.pdf)<sup>[18]</sup> without the poll wording. By this paper's own accounting that vote is a direction signal, not an adoption. Lineage is not the argument of this section. Deployment is.

One symmetry must be stated before any table. Neither specification is deployed. The Profiles framework of P3589R2 has no implementation of its proposed syntax; the implicit-contract-assertion framework of P3100R6 has none of its machinery. Both are undeployed specifications competing to govern a deployed practice. The deployment question that separates them is not which proposal has shipped - neither has - but which architecture the shipped practice already embodies, and which requires machinery the field has not built.

Table 2: The deployed practice both architectures claim. Public vendor documentation; each row names the check-set, when it first shipped, and how it is configured and responds to a failed check as deployed.

| Practice | First shipped | Configuration and failure response, as deployed |
|---|---|---|
| Core Guidelines checkers | clang-tidy `cppcoreguidelines-` checks in LLVM 3.8 (March 2016)<sup>[19]</sup>; MSVC checker in VS 2015 Update 1, installed by default from VS 2017<sup>[20]</sup> | Named rule sets; compile-time diagnostics; no runtime machinery<sup>[20]</sup> |
| Hardened libc++ | LLVM 18 (March 2024)<sup>[21]</sup> | Four named modes (none, fast, extensive, debug); a failed check "reliably terminated" per Apple's documentation<sup>[22]</sup>; a build setting in Xcode 16<sup>[22]</sup>; deployed across Google server-side production at approximately 0.30% average cost<sup>[23]</sup> |
| libstdc++ assertions | `_GLIBCXX_ASSERTIONS`, GCC 6 (2016)<sup>[24]</sup> | Macro-enabled precondition assertions; enabled by default for unoptimized builds since GCC 15 (2025)<sup>[24]</sup> |
| MSVC STL hardening | `_MSVC_STL_HARDENING`, VS 2022 17.14 (May 2025)<sup>[25]</sup> | Coarse macro plus per-class macros; a failed check calls `__fastfail()`<sup>[25]</sup> |

Table 3: The substrate's distinctive machinery. Status from vendor documentation and the proposal's own text.

| Component | Status | Public record |
|---|---|---|
| Contract-violation runtime (P2900R14, C++26)<sup>[26]</sup> | Adopted into the working draft, February 2025<sup>[27]</sup> | One compiler implementation: GCC 16.1 (April 2026), opt-in `-fcontracts`, under GCC's blanket experimental C++26 label<sup>[28]</sup>. Clang: "No"<sup>[29]</sup>. MSVC: "not yet implemented"<sup>[25]</sup> |
| Implicit contract assertions (P3100R6) | Proposed | No implementation. The word "experience" does not occur in P3100R6<sup>[1]</sup> |
| The assume semantic, the proposal's fifth (P3100R6 Section 5.4) | Proposed | In no shipping compiler's documented contracts support: GCC 16.1 documents four evaluation semantics<sup>[28]</sup>; Clang and MSVC document none<sup>[29]</sup> |
| Labels (P3400R3) | Proposed | Future tense in P3100R6 itself: Labels "will provide the ability to choose and constrain the evaluation semantic in code"<sup>[1]</sup> |

Read together, the tables say one thing. Every deployed check-set in Table 2 is a named, vendor-defined collection whose failure response is fixed by its definition: a diagnostic, a termination, a `__fastfail()`. None routes through a user-replaceable contract-violation handler, none is configured by a Label, and none needs an assume semantic. That is the shape the Profiles papers describe - a profile defines its guarantee and its error action - and it is not the shape the substrate requires, which is the machinery of Table 3. The proposal's mapping of `-ftrapv`, `-fwrapv`, and the sanitizers into its vocabulary (Section 3) is a genuine contribution of nomenclature, and it points the same way: the tools it maps predate the proposal, and none of them uses its machinery.

Standard library hardening is where the two ledgers touch, and both halves belong in the record. C++26 specifies hardened preconditions in contract terms - P3471R4 is the first user of the adopted Contracts feature<sup>[27]</sup> - so the specification has already assigned hardening's future to the substrate. The deployments run ahead of that assignment: libc++'s hardening shipped two years before any compiler implemented the contracts runtime, and MSVC ships hardening today by calling `__fastfail()` "As C++26 Contracts are not yet implemented"<sup>[25]</sup>. The assignment is standardized; the field has not yet exercised it. What deployment validates today is the named-guarantee shape, not the handler-and-semantics machinery.

Measured by the direction paper's test, the question of this section is which architecture is the gradual building on previous work, and which is the new substrate awaiting its first deployment. The deployed practice answers it: named check-sets with vendor-defined failure responses ship at scale today, and the violation-handler-and-Labels machinery does not yet ship at all. A diagram does not make a framework senior. Deployment does.

---

## 7. The Silence: No Published Answer

The recasting has been in print since June 2025 - first on slide 53 of [P3754R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3754r0.pdf)<sup>[30]</sup> ("Configurable Profiles - Named configuration presets"), endorsed by the Sofia diagram poll, then in P3100R4 (August 2025) and every revision since. Through the 2026-05 mailing, no published WG21 paper contests it, and none outside the proposal's own companion papers engages it.

That is an absence claim, so its method is disclosed here and can be re-run. All papers in the public open-std 2025 and 2026 mailing directories whose titles match profile, safety, harden, undefined behaviour, UB, erroneous, contract, preset, P3100, or P3754 were enumerated, together with every revision of P3100 and P3754 and the P2687/P3274/P3081/P3589/P3970/P3984 lineage: 121 documents (fetched 2026-07-07; the newest available mailing is 2026-05). The full text of each was searched for P3100, P3754, preset, configuration preset, implicit contract assertion, and layering language (built on top of, higher-level feature, substrate). Web searches for response papers and the public GitHub tracker issue for P3100 were checked in addition, and the full WG21 paper index was scanned by author for 2025-2026 papers by Stroustrup, Dos Reis, Sutter, and Vandevoorde. The recasting occurs only in P3100R4/R5/R6 and P3754R0/R1, all by the proposal's own authors. The proposal's companion papers argue for the subordination and are excluded from the claim by construction; no paper contests it. A contesting paper with a title outside the keyword net would escape the enumeration; the author scan, the web searches, and the tracker check are the mitigation.

The one-directional record is worth setting out, because the silence is not symmetric.

On the proposal's side, the subordination thesis is asserted in print, repeatedly. [P3543R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p3543r0.pdf)<sup>[31]</sup> (December 2024, co-authored by Doumler and Berne) states that P3081R1's runtime checks "are already preconditions introduced as implicit preconditions into the language itself by [P3100]". [P3599R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3599r0.pdf)<sup>[32]</sup> (February 2025) proposes to "restrict [P3081R1] (Profiles) to static checks". P3100R6 Section 4.4 and Figure 4 state the preset architecture; Section 5.6 exercises it.

On the Profiles side, the published record asserts an incompatible architecture and does not engage the rival claim. P3589R2<sup>[4]</sup>, the framework paper, contains no occurrence of "contract", "label", or "preset" in any revision - its architecture stands entirely apart from the substrate - though its latest revision (May 2025) predates the recasting's first publication, so its silence is chronology, not evidence. P3984R0<sup>[6]</sup> post-dates the recasting by eight months and asserts a profile that owns its semantics directly: "For example, signed arithmetic overflow is UB so a profile can define it to be wraparound like unsigned arithmetic (though I wouldn't do that), to be saturated arithmetic, or to throw an exception." Its term for the response to a failed runtime check is "error action"; the strings "P3100", "contract", and "preset" do not occur anywhere in it. [P3970R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p3970r0.pdf)<sup>[15]</sup>, the Direction Group's January 2026 statement, comes nearest: it reports "a stream of uncoordinated proposals to address problems from different perspectives, often not even mentioning Profiles" and reasserts the P3589R2 framework as the way forward - without naming P3100 or addressing the layering claim. Gabriel Dos Reis has no published paper of any kind in 2026 through the 2026-05 mailing, per the author scan above.

The interaction question itself was named as open, in print, eighteen months ago. [P3608R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3608r0.pdf)<sup>[12]</sup> (Voutilainen, Wakely, Dos Reis, January 2025): "We have a very unclear picture on how contracts and profiles should interact and interoperate." The question was asked before the recasting existed. The recasting is the proposal's answer. Nothing in print has answered back.

The gap is structural, and Section 4 explains it. A paper with no normative effect creates no moment that demands a response. There is no wording to object to that changes anyone's implementation, no forwarding poll to vote against, no deadline. Each individual poll is about direction, process, or a diagram; opposing the architecture at any single point means opposing something that, at that point, does nothing. The result is a claim that advances precisely because it never presents a surface to push against - while the papers that could contest it continue to assert their own architecture in parallel, as if the two could both be true.

---

## 8. The Cost: What the Recasting Forecloses

The costs below are stated as design costs, from the published texts. Each is what the preset architecture removes from the Profiles design space if it becomes the settled premise.

- **Profile-defined semantics.** P3984R0 states its model in one sentence: "A profile cannot change the semantics of a program beyond defining the meaning of some forms of undefined behavior" - and its overflow example (Section 7 quotes it) has the profile choose wraparound, saturation, or an exception. Under Section 4.4 of P3100R6, a profile does not define semantics; it selects configurations of the framework's semantics. The difference is ownership: an author of a profile under P3984's model writes the guarantee; an author of a preset under P3100's model chooses from the substrate's menu.
- **Configuration ownership.** Section 7.2 of P3100R6 requires that either Labels or the Profiles framework be specified in terms of the other. The proposal nominates Labels: Section 5.4 assigns granular semantic selection to Labels in the future tense, Section 5.6 reroutes error-category identification to Labels, and Section 7.2 sketches a profile as "essentially a declaration that expands to [P3400R3] directives". If per-clause review normalizes that architecture case by case, the framework of P3589R2 arrives at its own EWG review pre-defined as syntax sugar over another proposal's facility.
- **The auditing demotion.** The alternative Section 7.2 offers - Profiles as "an auditing feature rather than a configuration feature" - removes Profiles from the configuration business entirely. An auditing profile cannot enable anything; it can only reject programs whose configuration, chosen through the substrate's mechanisms, violates its guarantees. That is a real and possibly useful feature. It is also a strictly smaller one than the framework the Direction Group endorsed<sup>[15]</sup>.
- **The dialect standard, applied asymmetrically.** Section 4.3.2 of P3100R6 rejects making refined behavior conditional, in these words: "We also cannot have two different language dialects where the same expression means two different things (overflow or wraparound)." Section 5.4 of the same paper then maps signed integer overflow, for the same expression, to wraparound under the ignore semantic, a diagnostic under observe or enforce, an abort under quick-enforce, and undefined behavior under assume - selected per case by an implementation-defined mechanism with "no guarantee that there is a way to select any particular semantic"<sup>[1]</sup>. The paper's own dialect standard is applied to the tool it rejects and not to the framework it proposes. Whether five per-case, implementation-selected meanings for one expression constitute dialects is exactly the kind of question that per-clause wording review will never ask, because it is not a property of any single clause.

---

## 9. Objections

Each heading below is an objection this paper expects, stated in its strongest form; each response draws only on evidence already presented.

### "Nothing normative changes, so nothing is decided."

This is the proposal's own strongest defense, and Section 4 concedes its premise: the wording obligates no implementation to do anything. What it decides is architecture, not behavior. Section 7.2 states that one configuration feature must be specified in terms of the other - that is a decision, and the proposal makes it in Labels' favor. Section 5.6 already executed a piece of it, withdrawing an API that another paper's published wording depends on. And the proposed wording amends [defns.undefined] and adds [basic.contract.implicit], so that every case of undefined behavior definitionally carries an implicit precondition assertion that it does not occur<sup>[1]</sup>. Definitions are the part of the standard every later paper must write against. A change to what undefined behavior *is* does not need runtime effects to have consequences.

### "Direction polls are just encouragement; the real decision comes later."

The real decision arrives at a room whose defaults the direction polls have set. By the time an adoption poll exists, the recasting will have six recorded consensus results and 77 case-by-case approvals behind it, and the burden of proof will sit on whoever contests the accumulated record. The Sofia diagram poll illustrates the conversion: a poll about the basis of a stalled, editor-curated white paper now functions, in the proposal's history section, as EWG endorsement of the strategy behind an IS-track proposal. Encouragement compounds. That is what Section 5 documents.

### "The framework and Profiles are complementary; the layering is a detail."

Complementary on whose terms? Section 7.2 of P3100R6 states the terms question directly and answers it: one feature must be specified in terms of the other, and the paper's architecture places Labels underneath. P3081R1 took the complementary path in print - it adopted the substrate's API into its own wording - and Section 5.6 of P3100R6 then withdrew that API. Complementarity without settled configuration ownership has already produced one stranded dependency. The layering is not a detail; it is the decision.

### "This analysis is itself procedural alarmism."

Every input to this paper is the proposal's own published text, the public mailings, and public vendor documentation, and the one absence claim ships with a re-runnable method (Section 7). The proposal's technical achievements are recorded in Section 3. The reader who disputes the conclusion can dispute it against the same sources.

---

## 10. Conclusion

P3100R6 recasts Profiles as "a named configuration preset" over a substrate of implicit contract assertions (its Section 4.4 and Figure 4). Its own text states that all existing implementations of C++ already conform to its proposed wording (Section 5.2). Stated plainly: that combination is a decision procedure. A paper that by its own account changes nothing has accumulated six polls - direction, vehicle, diagram, target, and review, never adoption - and case-by-case review will now convert that direction into 77 small consensus determinations, individually easy and collectively expensive to reverse. What the procedure settles is which framework is senior.

The published record, measured by the committee's own existing-practice standard, runs the other way. Neither specification is deployed, and the deployed practice sides with one of them. Hardened standard libraries in three implementations and a decade of Core Guidelines checkers ship as named check-sets with vendor-defined failure responses - the shape the Profiles model describes - with production deployment documented by Apple and Google. The substrate's distinctive machinery has no such record: its base facility has one opt-in compiler implementation, released this April under an experimental label; the framework built on top of it claims no implementation experience in its own text; the fifth evaluation semantic it requires is in no shipping compiler's contracts support; and the Labels facility it routes configuration through is described by the proposal itself in the future tense.

And the recasting has advanced without a single published paper contesting it. The interaction question was named as open in January 2025; the proposal answered it in June 2025; through the 2026-05 mailing, nothing in print has answered back. The no-normative-effect posture gave the room no moment that demanded an answer - that is the mechanism, and it is still running.

This paper's contributions, restated with their results: the recasting is documented from the proposal's own text (Section 3); the no-op posture and its procedural effect are named (Section 4); the ratchet is reconstructed entirely from the proposal's self-reported history (Section 5); the deployment record of the two architectures points opposite to the proposed hierarchy (Section 6); and the absence of any published contest is verified by a disclosed method (Section 7).

Seniority among safety frameworks is earned by deployment. On the record assembled here, the architecture the Profiles model describes is the one the deployed practice already embodies, and the substrate's machinery has not yet begun to earn it. Which framework governs safety configuration in C++ is a first-order design question with a decade of field evidence bearing on it. It is worth deciding as a question, on that evidence, rather than inheriting as the accumulated side effect of polls about a paper that changes nothing.

---

## Acknowledgments

The verbatim extraction and cross-checking of P3100R6's text, the verification of paper numbers and titles against the public indexes, the deployment documentation survey, and the mailing-corpus absence search were performed with machine assistance and verified against the cited sources.

---

## References

[1] [P3100R6](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p3100r6.pdf) - "A framework for systematically addressing undefined behaviour in the C++ Standard" (Timur Doumler, Joshua Berne, 2026).

[2] [P4094R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4094r1.pdf) - "Info: The Unification of Executors and P0443" (Vinnie Falco, 2026).

[3] [P3400R3](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p3400r3.pdf) - "Controlling Contract-Assertion Properties" (Joshua Berne, 2026).

[4] [P3589R2](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3589r2.pdf) - "C++ Profiles: The Framework" (Gabriel Dos Reis, 2025).

[5] [P3081R2](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3081r2.pdf) - "Core safety profiles for C++26" (Herb Sutter, 2025).

[6] [P3984R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p3984r0.pdf) - "A type-safety profile" (Bjarne Stroustrup, 2026).

[7] [P3081R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3081r1.pdf) - "Core safety profiles for C++26" (Herb Sutter, 2025).

[8] [P3656R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3656r1.pdf) - "Initial draft proposal for core language UB white paper: Process and major work items" (Herb Sutter, Ga&scaron;per A&zcaron;man, 2025).

[9] [Trip report: Summer ISO C++ standards meeting (Cologne)](https://herbsutter.com/2019/07/20/trip-report-summer-iso-c-standards-meeting-cologne/) - "Contracts moved from draft C++20 to a new Study Group" (Herb Sutter, 2019).

[10] [P0443R14](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p0443r14.html) - "A Unified Executors Proposal for C++" (Jared Hoberock, Michael Garland, Chris Kohlhoff, Chris Mysen, H. Carter Edwards, Gordon Brown, David Hollman, 2020).

[11] [P2000R5](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p2000r5.pdf) - "Direction for ISO C++" (Jeff Garland, Paul E. McKenney, Roger Orr, Bjarne Stroustrup, David Vandevoorde, Michael Wong, 2026).

[12] [P3608R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3608r0.html) - "Contracts and profiles: what can we reasonably ship in C++26" (Ville Voutilainen, Jonathan Wakely, Gabriel Dos Reis, 2025).

[13] [P3471R4](https://open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3471r4.html) - "Standard library hardening" (Konstantin Varlamov, Louis Dionne, 2025).

[14] [P2759R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p2759r1.pdf) - "DG Opinion on Safety for ISO C++" (Howard Hinnant, Roger Orr, Bjarne Stroustrup, David Vandevoorde, Michael Wong, 2023).

[15] [P3970R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p3970r0.pdf) - "Profiles and Safety: a call to action" (David Vandevoorde, Jeff Garland, Paul E. McKenney, Roger Orr, Bjarne Stroustrup, Michael Wong, 2026).

[16] [P2687R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2687r0.pdf) - "Design Alternatives for Type-and-Resource Safe C++" (Bjarne Stroustrup, Gabriel Dos Reis, 2022).

[17] [Bjarne Stroustrup announces C++ Core Guidelines](https://isocpp.org/blog/2015/09/bjarne-stroustrup-announces-cpp-core-guidelines) - "Bjarne Stroustrup announces C++ Core Guidelines" (isocpp.org, 2015).

[18] [P2816R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p2816r0.pdf) - "Safety Profiles: Type-and-resource Safe programming in ISO Standard C++" (Bjarne Stroustrup, Gabriel Dos Reis, 2023).

[19] [clang-tidy checks, LLVM 3.8](https://releases.llvm.org/3.8.0/tools/clang/tools/extra/docs/clang-tidy/checks/cppcoreguidelines-pro-bounds-array-to-pointer-decay.html) - "cppcoreguidelines-pro-bounds-array-to-pointer-decay" (LLVM Project, 2016).

[20] [Using the C++ Core Guidelines checkers](https://learn.microsoft.com/en-us/cpp/code-quality/using-the-cpp-core-guidelines-checkers) - "Use the C++ Core Guidelines checkers" (Microsoft Learn, retrieved 2026).

[21] [libc++ Hardening Modes](https://libcxx.llvm.org/Hardening.html) - "Hardening Modes" (LLVM Project, retrieved 2026).

[22] [C++ in Xcode](https://developer.apple.com/xcode/cpp/) - "C++ and Xcode" (Apple Developer, retrieved 2026).

[23] [Retrofitting spatial safety to hundreds of millions of lines of C++](https://security.googleblog.com/2024/11/retrofitting-spatial-safety-to-hundreds.html) - "Retrofitting spatial safety to hundreds of millions of lines of C++" (Alexander Rebert, Christoph Kern, Google Security Blog, 2024).

[24] [libstdc++ macros documentation](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_macros.html) - "Macros" (GCC, retrieved 2026); API history at [libstdc++ API evolution](https://gcc.gnu.org/onlinedocs/libstdc++/manual/api.html).

[25] [microsoft/STL VS 2022 changelog](https://github.com/microsoft/STL/wiki/VS-2022-Changelog) - "VS 2022 Changelog" (Microsoft STL team, retrieved 2026); hardening details at [STL Hardening wiki](https://github.com/microsoft/STL/wiki/STL-Hardening).

[26] [P2900R14](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p2900r14.pdf) - "Contracts for C++" (Joshua Berne, Timur Doumler, Andrzej Krzemie&nacute;ski, 2025).

[27] [Trip report: February 2025 ISO C++ standards meeting (Hagenberg, Austria)](https://herbsutter.com/2025/02/17/trip-report-february-2025-iso-c-standards-meeting-hagenberg-austria/) - (Herb Sutter, 2025).

[28] [GCC 16 release notes](https://gcc.gnu.org/gcc-16/changes.html) - "GCC 16 Release Series: Changes, New Features, and Fixes" (GCC, 2026); flag documentation in the [GCC 16.1 manual](https://gcc.gnu.org/onlinedocs/gcc-16.1.0/gcc/C_002b_002b-Dialect-Options.html); experimental C++26 label at [C++ Standards Support in GCC](https://gcc.gnu.org/projects/cxx-status.html).

[29] [C++ Support in Clang](https://clang.llvm.org/cxx_status.html) - "C++ Support in Clang" (LLVM Project, retrieved 2026).

[30] [P3754R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3754r0.pdf) - "Slides for P3100R2 presentation to EWG" (Timur Doumler, 2025).

[31] [P3543R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p3543r0.pdf) - "Response to Core Safety Profiles (P3081)" (Mungo Gill, Corentin Jabot, John Lakos, Joshua Berne, Timur Doumler, 2024).

[32] [P3599R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3599r0.pdf) - "Initial Implicit Contract Assertions" (Joshua Berne, Timur Doumler, 2025).
