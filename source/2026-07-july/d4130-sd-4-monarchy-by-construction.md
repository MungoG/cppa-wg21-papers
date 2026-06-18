---
title: "SD-4: A Monarchy By Construction"
document: P4130R0
date: 2026-07-01
intent: info
audience: WG21
reply-to:
  - "Vinnie Falco <vinnie.falco@gmail.com>"
---

## Abstract

[SD-4](https://isocpp.org/std/standing-documents/sd-4-wg21-practices-and-procedures)<sup>[1]</sup> concentrates power and accelerates feature delivery beyond design capacity.

This paper proposes six corrections to SD-4. Each replaces a single passage with text that aligns the practice with the ISO/IEC Directives. The convenor can adopt all six immediately.

---

## Revision History

### R0: July 2026 (post-Brno mailing)

- Initial version.

---

## 1. Disclosure

The author provides information and serves at the pleasure of the committee.

The author's position is that C++ should prioritize stability over new features.

## 2. Concentration of Power

The convenor appoints every subgroup chair with no oversight. Chairs control the schedule, frame the polls, and determine consensus. SD-4 confirms that chairs "have no fixed term."<sup>[1]</sup> Study group chairs function identically<sup>[2]</sup>. The committee's own description states that the convenor "determines consensus, chairs the WG, sets the WG meeting schedule," and "appoints Study Groups," and that the subgroups operate "with the authority of the convenor."<sup>[3]</sup> The Directives govern to the working-group level and are silent on WG-internal structure<sup>[4]</sup>; SD-4 fills that silence with concentrated, unreviewable discretionary power. It is a single appointment chain, and every decision downstream inherits its judgment.

### 2.1 The Consensus Ratchet

Once a chair declares consensus, five provisions of SD-4 make that declaration nearly impossible to reverse:

1. A proposal "normally advances if there are more than twice as many in favor of a proposal as against"<sup>[1]</sup> - a vote ratio, with no requirement to reconcile the concerns of those against.
2. The minority, once heard, must "accept group decisions."<sup>[1]</sup>
3. A national body ballot comment that revisits a decided question is "out of harmony with the ISO Code of Conduct."<sup>[1]</sup>
4. A competing approach that lacks a paper "does not exist and will not block progress."<sup>[1]</sup>
5. Repeated escalation "erodes their credibility."<sup>[1]</sup>

The provisions create compounding effects. A decision reached on a bare two-to-one vote, in a room where most delegates did not read the paper, becomes a result the minority must accept, may not revisit at ballot, dare not escalate, and cannot displace with a later alternative. The ratchet turns once, and it locks.

---

## 3. Six Proposed Corrections

The ratchet is tuned for fast feature delivery, which is the one thing a language as large as C++ cannot afford to optimize for. Each correction below replaces a single passage of SD-4.

### 3.1 Chair Terms

The Directives require fixed terms with confirmation for the offices they govern (Directive 1.12.1)<sup>[4]</sup>; SD-4 grants subgroup chairs indefinite tenure with neither.

:::wording

<del>Subgroup chairs are appointed by the convenor, and are selected to match the current needs of the subgroup. They have no fixed term.</del>

<ins>Subgroup chairs are appointed by the convenor for a term of three years, subject to confirmation by the committee. Chairs may be reappointed through the same process.</ins>

:::

### 3.2 The 2:1 Threshold

The Directives define consensus as the reconciliation of conflicting arguments, not a vote ratio (Directive 2.5.6)<sup>[4]</sup>. The two-to-one rule appears nowhere in the Directives.

:::wording

<del>Subgroup polls, especially in design subgroups, should favor progress. A proposal normally advances if there are more than twice as many in favor of a proposal as against, after discussion of the concerns of those voting against and possibly a re-poll to see if opinions have improved. This is true even if a large number vote Neutral, though it can be concerning if a majority of all those voting vote Neutral.</del>

<ins>Subgroup polls should ensure all views are heard. The chair determines whether consensus exists after discussion of the concerns of those voting against and, where possible, reconciliation of conflicting arguments. The reconciliation shall be minuted and circulated within four weeks of the meeting end (Directive 1.9.2c).</ins>

:::

### 3.3 Ballot Comments

The Directives place no content restriction on national body ballot comments (Directive 2.6.2) and require every comment to be addressed (Directive 2.6.5)<sup>[4]</sup>; an internal practices document cannot override either obligation.

:::wording

<del>A ballot comment that requests a change that was already considered and decided otherwise at a WG21 meeting, and comes from a national body that was present at the meeting and had an opportunity to have their objections be heard and considered, is out of harmony with the ISO Code of Conduct's commitment to 'accept group decisions.' Once the WG has consensus to send a document for ballot, to repeat as an NB comment an objection that previously failed to carry the day is actually making, not a new technical objection, but an objection to the consensus of the WG.</del>

<ins>National body ballot rights are governed by the ISO/IEC Directives (2.6.2, 2.6.5). All technical and editorial comments are in scope and all comments shall be addressed.</ins>

:::

### 3.4 The Escalation Trap

The Directives direct objectors to a formal appeal process and attach no penalty to its use (Directive 2.5.6, Clause 5.1)<sup>[4]</sup>. SD-4's credibility language penalizes the exercise of a right the Directives protect.

:::wording

<del>...or (b) when a participant or national body regularly uses the escalation process to express a pattern of strong disagreement on topic after topic, which erodes their credibility and is not the purpose of the escalation resolution process (like exception handling, escalation handling is for hard errors, and is not designed for expressing less serious conditions or for what should be ordinary control flow).</del>

<ins>...or (b) when escalation is used routinely for matters that could be resolved through normal discussion. The formal appeal process (ISO/IEC Directives, Clause 5.1) is available to any participant at any time without prejudice.</ins>

:::

### 3.5 The Information Seal

SD-4's quotation prohibition has no basis in the Directives or in JTC 1 Standing Document 19<sup>[5]</sup>. WG14, WG5, and WG9 publish their minutes.

:::wording

<del>Meeting records of subgroup discussion, meeting wikis, and non-public committee email lists (aka reflectors), which often include personal positions and discussion. It is not allowed to quote from these publicly (e.g., in papers and blog posts) except that the following are allowed: (a) quoting straw poll questions and numeric results; and (b) quoting words or positions attributed to a specific person with that person's prior consent.</del>

<ins>Meeting records are governed by the ISO/IEC Directives (1.8.2e, 1.9.2c) and JTC 1 Standing Document 19 Section 9.</ins>

:::

### 3.6 Silence As Consensus

The Directives define consensus as "seeking to take into account the views of all parties concerned" (Directive 2.5.6)<sup>[4]</sup>; silence is the absence of a view, not its expression.

:::wording

<del>Unanimous consent, where if there are no objections then it is known that everyone is either in favor or neutral, without having to count hands. This is typically used to save time when there may already be broad agreement.</del>

<ins>Unanimous consent is appropriate only for editorial corrections, procedural motions, and matters previously decided by an explicit poll. For substantive design or specification questions, the chair takes an explicit poll. Silence is not agreement.</ins>

:::

---

## 4. What the Rules Produce

### 4.1 The Bandwidth Gap

The median delegate cannot fully absorb the papers, and falls back on social evaluation to fill the gap. When the delegate is silent, SD-4 records that silence as consensus.

### 4.2 No Evidentiary Bar

No rule requires an author to steel-man the case against a proposal, or to build a competing approach and measure it before dismissing it. [P2300R10](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html)<sup>[6]</sup> Section 1.9.2 sets aside coroutines as the foundation for asynchronous code in roughly five paragraphs of prose, with no coroutine-native model built and no measurement taken. The omissions are now in the record. [P4255R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4255r0.pdf)<sup>[7]</sup> shows that the sender model never accounted for the synchronous case, which the awaitable design handles at zero cost. [P4251R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4251r0.pdf)<sup>[8]</sup> shows that the sender model's own headline domain, GPU data movement, composes naturally with coroutines - a result reached independently by groups at NVIDIA Research, CERN, the University of Wisconsin, and Schr&ouml;dinger. The authors did not notice, because nothing required them to look.

### 4.3 A Nash Equilibrium

For headline features, where competing designs contend for one slot and the loser ships nothing, these incentives form a stable equilibrium. Claim the widest possible scope; every domain claimed is territory a rival cannot enter. Disclose the minimum evidence needed to pass the next gate; every benchmark and every admitted limitation is ammunition for an opponent. Dig in rather than concede. Treat non-engagement as the dominant response to a competitor, because under a consensus model silence reads as the absence of support. [P4263R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4263r0.pdf)<sup>[9]</sup> documents this mechanism in full. Eighteen implementers across every major compiler reported the result in [P3962R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p3962r0.pdf)<sup>[10]</sup>:

> "We would like the committee to consider ways of slowing down the addition of features into the standard to allow implementers to catch up, and to allow the existing features to improve in quality."

The committee ships faster than the language can be designed or implemented. The rules in Section 2 are why.

## 5. Conclusion

These six corrections are not exhaustive. Each is a single passage in a document the convenor maintains and revises between meetings without a plenary vote. Adopting all six requires no poll, no study group, and no national body ballot. The instrument of the problem and the instrument of its repair are the same file.

---

## References

[1] [SD-4](https://isocpp.org/std/standing-documents/sd-4-wg21-practices-and-procedures) - "WG21 Practices and Procedures" (Guy Davidson, 2026). The provisions quoted here predate the current convenor.

[2] [SD-3](https://isocpp.org/std/standing-documents/sd-3-study-group-organizational-information) - "Study Group Organizational Information."

[3] [The Committee](https://isocpp.org/std/the-committee) - "The Committee."

[4] [ISO/IEC Directives, Part 1, Consolidated JTC 1 Supplement](https://jtc1info.org/wp-content/uploads/2023/11/ISO-IEC-Consolidated-JTC-1-Supplement-2023.pdf) - "ISO/IEC Directives, Part 1, Consolidated JTC 1 Supplement" (ISO/IEC, 2023).

[5] JTC 1 Standing Document 19 - "Meetings," Section 9 (JTC 1, 2022).

[6] [P2300R10](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html) - "`std::execution`" (Eric Niebler, Micha&lstrok; Dominiak, Georgy Evtushenko, Lewis Baker, Lucian Radu Teodorescu, Lee Howes, Kirk Shoop, Michael Garland, Bryce Adelstein Lelbach, 2024).

[7] [P4255R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4255r0.pdf) - "Awaitables And Senders For Synchronous I/O" (Vinnie Falco, 2026).

[8] [P4251R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4251r0.pdf) - "IoAwaitables for GPU Data Movement: Convergent Findings" (Vinnie Falco, 2026).

[9] [P4263R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4263r0.pdf) - "Why I Write" (Vinnie Falco, 2026).

[10] [P3962R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p3962r0.pdf) - "Implementation reality of WG21 standardization" (Nina Ranns, et al., 2026).
