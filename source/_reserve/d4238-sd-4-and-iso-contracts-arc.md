---
title: "Reconciliation Capacity: The Contracts Arc Under Two Rule Sets"
document: P4238R0
date: 2026-07-01
intent: info
audience: WG21
reply-to:
  - "Vinnie Falco <vinnie.falco@gmail.com>"
---

## Abstract

Twenty-one polls, zero documented reconciliation processes. The "ignore" semantic was contested in 2022 and is still unresolved in 2026. Under SD-4, "no consensus for change" preserves the status quo. Under the ISO Directives, unresolved objections require reconciliation at each stage. The contracts arc demonstrates what happens when a contentious design choice survives through repeated status-quo preservation rather than documented engagement: minority concerns accumulate without resolution, surfacing as a 19-NB revolt at the ballot stage.

This paper traces the trajectory of P2900's "ignore" semantic from its establishment in 2021 through the NB ballot in 2025. For each decision point, it asks: what did the ISO Directives require that SD-4 did not? The answer is not better decisions. It is documented engagement - the structural capacity to reconcile rather than override.

---

## Revision History

### R0: July 2026 (pre-Wroclaw mailing)

- Initial version.

---

## 1. Disclosure

The author provides information and serves at the pleasure of the committee.

The author has no competing contracts proposal. The author did not participate in SG21. The author's analysis is limited to the published record. Committee discussions occur in rooms, hallways, and private channels that leave no public trace. If a reader is aware of reconciliation processes that this paper's research did not reach, the author welcomes the correction.

A forthcoming companion paper, [P4240R0](https://wg21.link/p4240r0)<sup>[1]</sup>, examines the NB ballot phase of P2900. This paper covers the developmental trajectory from 2021 to the ballot. P4240 is the endpoint. This paper is the path that produced it.

This paper asks for nothing.

---

## 2. What P2900 Achieved

SG21 spent five years gathering use cases, exploring the design space, and refining the proposal before forwarding it to EWG. [P2900R14](https://wg21.link/p2900r14)<sup>[2]</sup> was adopted into the C++26 working draft at Hagenberg in February 2025 with strong consensus. Complete implementations exist in publicly available forks of GCC and Clang. The implementers reported P2900's specification as "clear and implementable."

[P2899R1](https://wg21.link/p2899r1)<sup>[3]</sup> provides a systematic design rationale. [P3846R1](https://wg21.link/p3846r1)<sup>[4]</sup> carries twenty-two co-authors addressing eighteen concerns raised in NB comments. The quality of the technical work is not in question. What is in question is which procedural system governs how unresolved objections interact with that work during development.

---

## 3. The Thesis

One point of jurisdiction first, because it frames everything that follows. The ISO Directives do not reach inside a working group. SG21 polls, EWG polls, and WG21 plenary votes have no procedural standing in ISO terms - the Directives neither require reconciliation there nor forbid its absence. The reconciliation machinery of Directive 2.5.6, the comment-resolution obligations of 2.6.5, and the appeal chain of 5.1 operate at the committee level, where National Bodies act. The comparison in this paper is therefore a counterfactual: what the development process would have looked like had the work been organized in ISO-governed bodies where those rules apply.

The counterfactual matters because of what the actual structure produces. The ISO Directives would not have prevented contracts from being adopted. In an ISO-governed body, they would have required the "ignore" semantic, the enforcement question, and the implementation experience concern to be reconciled during development - producing iterative design evolution rather than terminal confrontation. In the actual structure, every one of those decisions was made by internal polls that ISO does not recognize, under SD-4 rules that bind no National Body, and the first decision point with ISO standing was the SC 22 ballot - which is exactly where the unreconciled dissent surfaced.

Networking lacked progress. Contracts had momentum without course correction. Both are consequences of the same structural gap: the entire development pipeline runs in a space where no reconciliation obligation exists.

Under SD-4, "no consensus for change" preserves the status quo. A minority that cannot muster a 2:1 supermajority to change the design has no structural remedy. The design proceeds. Under the ISO Directives, at committee level, consensus requires "seeking to take into account the views of all parties concerned and to reconcile any conflicting arguments" (2.5.6). The minority is directed to the appeal chain (5.1), not told to accept the outcome.

The difference is not in whether the decision is made. It is in whether the minority's concerns are engaged with substantively or simply outvoted - and in whether any of the intermediate decisions carry procedural weight at all.

---

## 4. The Record of Non-Reconciliation

[P2521R4](https://wg21.link/p2521r4)<sup>[5]</sup>, "Contract support - Record of SG21 consensus," is the authoritative document of SG21's decisions from 2021 through 2023. It records twenty-one polls with vote tallies. It documents zero reconciliation processes.

Each section states the design choice, provides rationale, then shows a poll table with SF/F/N/A/SA counts and a one-line disposition: "Consensus," "Not consensus," or "Consensus against." There is no text anywhere in the paper describing what was done to address minority concerns between polls, what the dissenters' specific objections were, or whether any attempt was made to find a compromise position before moving to a vote.

Four polls resulted in "No consensus" or "Not consensus": Poll 3 (safe-predicate model, 5-2-0-7-4), Poll 4 (side-effect-free predicates, 6-2-1-5-4), Poll 9 (throwing predicate as violation, 7-7-3-3-7), Poll 10 (exception propagation, 5-5-4-3-10). In each case the paper records the vote, prints the disposition, and moves on. No documentation of follow-up discussion. No description of compromise proposals attempted. No indication of whether the items were revisited with modified wording.

No Directive was violated by this absence - the Directives do not apply to SG21, and none of these twenty-one polls has ISO standing. That is the structural gap. In an ISO-governed body, Directive 2.5.6 requires the process, not just the outcome. Inside a working group, no rule requires it, no rule forbids it, and the record shows it did not happen.

---

## 5. The "Ignore" Semantic

### 2021: Establishment

[P2388R4](https://wg21.link/p2388r4)<sup>[6]</sup> established "No_eval" - the precursor to "ignore." Section 5.1 documented "two irreconcilable programming models" within SG21: one viewing contract violation as fatal (comparable to null-pointer dereference), the other viewing it as "just a piece of information." The deferral of continuation mode preserved "ignore" as the only non-terminating option.

SG21 adopted P2388R0 as the basis for the MVP in July 2021 (SF:8 / F:4 / N:2 / A:0 / SA:0).

### 2022: Dissent

Stroustrup published [P2698R0](https://wg21.link/p2698r0)<sup>[7]</sup>, "Unconditional Termination Is a Serious Problem," arguing that the binary choice (abort or ignore) made contracts unusable for programs that cannot unconditionally terminate - the Linux kernel, financial systems, medical devices. The paper drove the 2023 addition of "observe" as a third semantic.

### 2023: Formalization

[P2521R4](https://wg21.link/p2521r4)<sup>[5]</sup> Poll 11 (Varna 2023) adopted [P2877R0](https://wg21.link/p2877r0)<sup>[8]</sup>'s three implementation-defined semantics - Ignore, Enforce, Observe - with SF:10 / F:11 / N:4 / A:3 / SA:1. Four opposing votes. Their concerns are entirely undocumented.

[P2852R0](https://wg21.link/p2852r0)<sup>[9]</sup> formalized "ignore" as well-defined: "Predicate evaluation is not performed; execution continues."

[P2932R1](https://wg21.link/p2932r1)<sup>[10]</sup> (Berne) established Principle 2: "Program Semantics Are Independent of Chosen CCA Semantics." This bounds the "ignore" semantic rather than eliminating it - the closest documented reconciliation attempt. But it is framed as a design constraint, not a negotiated compromise with named dissenters.

### 2024: The Tokyo Poll

EWG polled at Tokyo (March 2024):

> "P2900R6: contracts should have `enforce` semantics only, and should not have `ignore` nor `observe`."
>
> SF:6 / F:1 / N:3 / A:15 / SA:24 - Consensus against.

The minority who wanted guaranteed enforcement - seven votes (SF:6 / F:1) - were structurally blocked. Under SD-4, "no consensus for change" preserves the status quo. The seven who wanted enforce-only have no further remedy.

In an ISO-governed body, the minority's unresolved objection to the proceeding - that "ignore" undermines safety - would require reconciliation (2.5.6). Engagement with the substance of their concern, not merely a vote confirming the status quo. In EWG, a body ISO does not recognize, neither the poll nor any obligation attached to it exists in ISO terms.

[P3362R0](https://wg21.link/p3362r0)<sup>[11]</sup> (Voutilainen/Corden, 2024) articulated the substance: without guaranteed enforcement, static analyzers cannot reason about contract predicates. Contracts become useless for safety proofs.

### 2025: The NB Ballot

Romania (NB comment RO 2-056) specifically targeted the "ignore" semantic: remove it, or remove Contracts entirely.

### 2026: Still Unresolved

[P3911R2](https://wg21.link/p3911r2)<sup>[12]</sup> responded to RO 2-056 with `pre!` syntax for non-ignorable contracts. EWG Kona 2025 voted consensus to pursue a change (SF:18 / F:25 / N:22 / A:7 / SA:0). A later telecon voted against the specific direction of P3911R2 (SF:4 / F:4 / N:9 / A:13 / SA:12).

The concern was raised in 2022 (P2698R0). It is unresolved in 2026. Four years.

---

## 6. The "Strict Contracts" Arc

A second non-reconciliation arc runs parallel to the "ignore" trajectory. [P2899R1](https://wg21.link/p2899r1)<sup>[3]</sup> Section 3.6.1 documents it in detail.

[P2680R0](https://wg21.link/p2680r0)<sup>[23]</sup> proposed "strict contracts" - contract predicates constrained to expressions provably free of undefined behavior. The concern: without such constraints, an optimizer can exploit UB in a contract predicate to elide the check entirely, even under the enforce semantic. The concern is real. P2899R1 reproduces a concrete example where signed integer overflow in a precondition allows the compiler to remove the check.

SG21 polled at Kona (November 2022):

> "Given our limited resources, we encourage the author of P2680 to come back with a revision addressing the currently open questions."
>
> SF:9 / F:5 / N:1 / A:3 / SA:7 - No consensus.

SG21 polled again (December 2022 telecon), twice:

> "SG21 should attempt to design a model for safe programming in C++ as part of the contracts MVP."
>
> SF:5 / F:2 / N:0 / A:7 / SA:4 - No consensus.

> "The contracts MVP should contain contract-checking predicates that are free of certain types of UB and free of side effects outside of their cone of evaluation, as proposed in D2680R1."
>
> SF:5 / F:2 / N:0 / A:7 / SA:4 - No consensus.

P2899R1 records what happened next: "At this point, the topic of 'strict contracts' was closed as far as SG21 was concerned."

Under SD-4, "no consensus" means the topic does not advance. The SG21 co-chairs closed it. No reconciliation with the seven members (SF:5 / F:2) who supported the direction is documented.

The author of P2680 then published [P3173R0](https://wg21.link/p3173r0)<sup>[24]</sup>, targeting EWG directly. EWG showed significant interest at Tokyo (March 2024):

> "P2900R6: contracts should expose less undefined behavior than regular C++ code does."
>
> SF:17 / F:12 / N:12 / A:12 / SA:5 - No consensus.

P2899R1 then records: "Following procedural complaints that the author of 'strict contracts' had not been given a fair chance to present his ideas during that joint SG21/SG23 session, it was agreed that 'strict contracts' should be reviewed again in SG23."

SG23 polled (May 2024):

> "We should promise more SG23 committee time to pursuing the approach of handling certain kinds of undefined behaviour differently inside and outside of contract predicates."
>
> F:7 / N:3 / A:13 - No consensus.

The concern surfaced in 2022. It was voted down in SG21, escalated to EWG, redirected to SG23 after procedural complaints, and voted down there. At no stage does the published record document reconciliation - engagement with the substance of the minority's objection between polls, compromise proposals attempted, or reasons why the concern was found unpersuasive rather than merely outvoted.

In an ISO-governed body, "no consensus" does not close a topic. It triggers the obligation to reconcile (2.5.6), and the minority retains standing with access to the appeal chain (5.1). None of that machinery reaches inside WG21: the SG21 co-chairs' discretionary re-hearing in SG23 is not an appeal process. It is an accommodation that the chair could have declined. At committee level the Directives provide structural remedy. Inside the working group, where this entire arc played out, there is none - by anyone's rules.

---

## 7. Implementation Experience

At Tokyo (March 2024), EWG polled:

> "P2900R6: contracts - there should be some usage experience of contracts in an implementation of the STL (without necessarily having a paper to adopt these changes) before contracts can move to plenary."
>
> SF:16 / F:30 / N:12 / A:3 / SA:0 - Strong consensus.

Forty-six in favor, three against. Near-unanimity.

Eight months later at Wroclaw (November 2024), LEWG forwarded P2900R11 to LWG. The SA comments:

> "Categorically opposed to forwarding this proposal, as I view implementability and deployment feasibility as a guess."
>
> "I'd be strongly in favor with a TS like we did with concepts thereby realizing that the design was totally wrong resulting in substantial changes."

[P3506R0](https://wg21.link/p3506r0)<sup>[13]</sup> (Dos Reis/Microsoft, November 2024) documented: "the libc++ experience report only replaced existing asserts with `contract_assert`, never using pre/post-conditions."

P2899R1<sup>[3]</sup> Section 2.9 describes the same deployment experience from the proponents' perspective: "Deployment experience was collected from replacing C assert in LLVM and dependencies and from replacing the existing hardening, validation, and debugging macros in the libc++ Standard Library implementation." The proponents' own rationale document confirms the scope that P3506R0 characterized as insufficient.

Three months later at Hagenberg (February 2025), P2900R14 was adopted. LWG: F:13 / A:0 / N:0.

Under SD-4, the near-unanimous Tokyo poll created no binding obligation. Ship pressure overrode it. Nor did it create an ISO obligation - an EWG poll has no ISO standing, so there was nothing in the ISO framework for the forwarding to be inconsistent with. In an ISO-governed body, a near-unanimous consensus requirement would be treated as a prerequisite condition, and the reconciliation framework (2.5.6) would flag the forwarding as inconsistent with the prior consensus unless the implementation experience had been provided.

---

## 8. P4005: The Minority That Could Not Escalate

[P4005R0](https://wg21.link/p4005r0)<sup>[14]</sup> (Voutilainen, 2026) proposed guaranteed enforcement: `entry_cond`, `return_cond`, and `mandatory_assert` that can never be set to "ignore." The proposal responded directly to NB comment RO 2-056.

EWG telecon (February 2025):

> "Encourage more work in the direction of P4005 for C++26 in response to RO 2-056."
>
> SF:8 / F:5 / N:2 / A:9 / SA:14 - Not consensus.

> "Encourage more work in the direction of P4005 for C++29."
>
> SF:9 / F:4 / N:8 / A:8 / SA:6 - Not consensus.

Under SD-4, the paper was rejected twice. No further procedural remedy exists within WG21. But this is the one point in the arc where the ISO framework genuinely applies, because RO 2-056 is a National Body ballot comment at SC level: Directive 2.6.5 requires it to be *addressed* - substantive engagement, not merely a vote demonstrating that the majority disagrees - and the NB retains standing with access to the three-level appeal chain (5.1). Two EWG telecon polls, which have no ISO standing, cannot discharge that obligation. No WG-internal vote can.

---

## 9. The Wroclaw Forwarding

EWG (November 2024):

> "P2900R11: send to CWG and LEWG for inclusion in C++26."
>
> SF:25 / F:17 / N:0 / A:3 / SA:12 - Consensus in favor.

Fifteen votes against out of fifty-seven voters. Twenty-six percent opposition.

[P3573R0](https://wg21.link/p3573r0)<sup>[15]</sup> (January 2025) was filed by nine authors - Stroustrup, Dos Reis, Spicer, Voutilainen, Vandevoorde, van Winkel, Regev, Hava, Garcia - expressing "grave concerns" about P2900's design and direction. The paper functions as a de facto minority report from a group that includes an EDG representative, a Microsoft representative, and a former convenor.

Under SD-4, the 2:1 threshold was met (42 in favor vs 15 against). Consensus was declared. The paper was forwarded.

Measured against the ISO definition of consensus - "seeking to take into account the views of all parties concerned and to reconcile any conflicting arguments" (2.5.6) - nine senior committee members filing "grave concerns" is not reconciled consensus. It is a supermajority override of a substantial minority. The override broke no ISO rule, because an EWG forwarding vote is not an ISO decision. That is precisely the problem: a decision with no ISO standing carried the design to the one stage where NB objections could no longer be reconciled, only balloted.

---

## 10. The Accumulation

The escalation trajectory of the opposition papers:

| Paper | Date | Authors | Position |
|---|---|---|---|
| [P2698R0](https://wg21.link/p2698r0)<sup>[7]</sup> | Nov 2022 | Stroustrup | "Unconditional Termination Is a Serious Problem" |
| [P3362R0](https://wg21.link/p3362r0)<sup>[11]</sup> | 2024 | Voutilainen, Corden | Contracts useless for safety without guaranteed enforcement |
| [P3506R0](https://wg21.link/p3506r0)<sup>[13]</sup> | Nov 2024 | Dos Reis (Microsoft) | "P2900 Is Still not Ready for C++26" |
| [P3573R0](https://wg21.link/p3573r0)<sup>[15]</sup> | Jan 2025 | 9 authors | "Grave concerns" - de facto minority report |
| [P3829R0](https://wg21.link/p3829r0)<sup>[16]</sup> | Jul 2025 | 5 authors | "Contracts do not belong in the language" |
| [P3835R0](https://wg21.link/p3835r0)<sup>[17]</sup> | Sep 2025 | Spicer, Voutilainen, Garcia | "Contracts make C++ less safe" - concrete safety hole |
| [P3851R0](https://wg21.link/p3851r0)<sup>[18]</sup> | Sep 2025 | Garcia et al. | Spain recommends removal |
| NB ballot | 2025 | 19/26 NBs | 4 request removal; Romania targets "ignore" |

No single decision was wrong. The "ignore" semantic has genuine use cases: gradual deployment in large codebases where enabling enforcement everywhere simultaneously is impractical. The MVP approach was a rational response to C++20's removal. Ship pressure is real. Experienced practitioners made these calls. [P3846R1](https://wg21.link/p3846r1)<sup>[4]</sup>'s twenty-two-author defense is thorough.

But the minority's concerns were expressed in 2022 and are still unresolved in 2026. Had the work been organized in ISO-governed bodies, forced reconciliation at each stage would have required engaging with the substance of each objection rather than preserving the status quo through repeated "no consensus for change." The Directives do not guarantee better outcomes. They guarantee documented engagement. The difference is that the minority's concerns either evolve the design iteratively or are reconciled through explicit compromise - rather than accumulating until the NB ballot forces a terminal confrontation. As it was, none of the decisions that carried the design forward had any ISO standing, and the ballot was the first point where ISO procedure touched the project at all.

No formal ISO-process appeal was filed. The opposition took the form of papers, not procedural escalation - and the appeal right belongs to National Bodies (5.1), not to individual participants, so for most of the dissenters papers were the only channel that existed. For the NBs, the appeal chain remains the structural remedy that no WG-internal rule can foreclose. Under SD-4, the system provided no structural path to resolution other than "bring a paper and outvote the majority."

---

## 11. Anticipated Objections

**Q: SG21 did reconcile. They spent five years.**

A: They spent five years producing a design. P2521R4 - the official record of that work - documents twenty-one polls and zero reconciliation processes. The time spent is not disputed. What is absent from the published record is documentation of engagement with the minority's specific objections between polls. Time spent is not reconciliation. Reconciliation is documented engagement with the substance of dissent.

**Q: The MVP was the reconciliation. Stroustrup's concerns drove the addition of "observe."**

A: The addition of "observe" in 2023 was a genuine response to P2698R0. It is the one documented case where a dissenting concern produced a design change. The paper credits this (Section 5). The question is whether the subsequent concerns - guaranteed enforcement, implementation experience, mixed-mode safety holes - received the same engagement. The published record shows they did not.

**Q: "No consensus for change" IS consensus. The committee considered it and decided not to change.**

A: Under SD-4's 2:1 threshold, this is procedurally correct. Under the ISO Directives (2.5.6), consensus requires reconciliation of conflicting arguments, not merely demonstration that the majority disagrees. The two systems define "consensus" differently. This paper documents what each definition produces.

**Q: The Directives would have just slowed things down.**

A: Possibly. The paper concedes this (Section 10). The Directives guarantee documented engagement, not superior outcomes. Whether iterative reconciliation would have produced a better design is unknowable. What is observable is that it would have produced a documented path from objection to resolution at each stage rather than an accumulation that detonated at the ballot.

---

## Acknowledgments

The author thanks Joshua Berne, Timur Doumler, and Andrzej Krzemienski for [P2900R14](https://wg21.link/p2900r14) and five years of SG21 work; Bjarne Stroustrup for [P2698R0](https://wg21.link/p2698r0), which drove the addition of "observe"; Ville Voutilainen for [P4005R0](https://wg21.link/p4005r0) and [P3362R0](https://wg21.link/p3362r0); Gabriel Dos Reis for [P3506R0](https://wg21.link/p3506r0); the nine authors of [P3573R0](https://wg21.link/p3573r0) for documenting their concerns on the record; Erich Keane, JF Bastien, Inbal Levi, and Timur Doumler for the poll records on [cplusplus/papers#1648](https://github.com/cplusplus/papers/issues/1648); Arthur O'Dwyer for the public catalog of NB comments; and Steve Gerbino for feedback on this paper.

---

## References

[1] [P4240R0](https://wg21.link/p4240r0) - "Did ISO or SD-4 Govern the P2900 Ballot?" (Vinnie Falco, forthcoming).

[2] [P2900R14](https://wg21.link/p2900r14) - "Contracts for C++" (Joshua Berne, Timur Doumler, Andrzej Krzemienski, 2025).

[3] [P2899R1](https://wg21.link/p2899r1) - "Contracts for C++ - Rationale" (Joshua Berne, Timur Doumler, Andrzej Krzemienski, 2025).

[4] [P3846R1](https://wg21.link/p3846r1) - "C++26 Contract Assertions, Reasserted" (Timur Doumler et al., 2026).

[5] [P2521R4](https://wg21.link/p2521r4) - "Contract support - Record of SG21 consensus" (Joshua Berne, 2023).

[6] [P2388R4](https://wg21.link/p2388r4) - "Minimum Contract Support: either No_eval or Eval_and_abort" (Andrzej Krzemienski, Gasper Azman, 2021).

[7] [P2698R0](https://wg21.link/p2698r0) - "Unconditional Termination Is a Serious Problem" (Bjarne Stroustrup, 2022).

[8] [P2877R0](https://wg21.link/p2877r0) - "Contract Build Levels and Semantics" (2023).

[9] [P2852R0](https://wg21.link/p2852r0) - "Contract Violation Handling Semantics for the Contracts MVP" (Tom Honermann, 2023).

[10] [P2932R1](https://wg21.link/p2932r1) - "Principled Approach to Open Design Questions for Contracts" (Joshua Berne, 2023).

[11] [P3362R0](https://wg21.link/p3362r0) - "Static analysis and 'safety' of Contracts, P2900 vs. P2680/P3285" (Ville Voutilainen, Richard Corden, 2024).

[12] [P3911R2](https://wg21.link/p3911r2) - "Make Contracts Reliably Non-Ignorable" (2026).

[13] [P3506R0](https://wg21.link/p3506r0) - "P2900 Is Still not Ready for C++26" (Gabriel Dos Reis, 2024).

[14] [P4005R0](https://wg21.link/p4005r0) - "A proposal for guaranteed-(quick-)enforced contracts" (Ville Voutilainen, 2026).

[15] [P3573R0](https://wg21.link/p3573r0) - "Contract concerns" (Bjarne Stroustrup, Gabriel Dos Reis, John Spicer, Ville Voutilainen, Daveed Vandevoorde, Nir Friedman, Gaby Dos Reis, Ran Regev, Ohad Hava, J. Daniel Garcia, 2025).

[16] [P3829R0](https://wg21.link/p3829r0) - "Contracts do not belong in the language" (David Chisnall, John Spicer, Ville Voutilainen, Gabriel Dos Reis, J. Daniel Garcia, 2025).

[17] [P3835R0](https://wg21.link/p3835r0) - "Contracts make C++ less safe - full stop!" (John Spicer, Ville Voutilainen, J. Daniel Garcia, 2025).

[18] [P3851R0](https://wg21.link/p3851r0) - "Spain's Position on Contracts for C++26" (J. Daniel Garcia et al., 2025).

[19] [cplusplus/papers#1648](https://github.com/cplusplus/papers/issues/1648) - GitHub issue tracking P2900.

[20] [cplusplus/papers#2641](https://github.com/cplusplus/papers/issues/2641) - GitHub issue tracking P4005.

[21] ISO/IEC. "ISO/IEC Directives, Part 1 - Consolidated JTC 1 Supplement." 2023.

[22] Davidson, G. "SD-4: WG21 Practices and Procedures." ISO/IEC JTC1/SC22/WG21/SD-4, 2026-05-11.

[23] [P2680R0](https://wg21.link/p2680r0) - "Contracts for C++: Prioritizing Safety" (Gabriel Dos Reis, 2022).

[24] [P3173R0](https://wg21.link/p3173r0) - "P2900R6 may not be sufficiently implementable" (Gabriel Dos Reis, 2024).

[25] [P3285R0](https://wg21.link/p3285r0) - "Contracts and Safety" (Gabriel Dos Reis, 2024).
