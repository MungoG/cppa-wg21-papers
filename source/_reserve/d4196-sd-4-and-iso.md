---
title: "Discrepancies Between SD-4 and the ISO Directives"
document: P4196R0
date: 2026-04-19
intent: info
audience: WG21
reply-to:
  - "Vinnie Falco <vinnie.falco@gmail.com>"
---

> **SHELVED (2026-06-10). DO NOT CIRCULATE.** The thesis of this paper is incorrect: ISO deliberately specified no rules for WG subgroups, so SD-4's subgroup provisions deviate from nothing and require no authorization. A written commitment was made not to circulate a paper with this premise. Retained as a record. Surviving material for a future inverted-thesis paper: the SD-4 vs ISO Code of Conduct mismatch (sections 3.4/3.6) and the NB appeal rights inventory (5.1.x); dead material: sections 3.1, 3.3, the 1.13.2 row of section 4, and the 1.4 path in section 5.

## Abstract

WG21 has two procedural documents. One is binding. The other is followed.

This paper places SD-4<sup>[1]</sup> and the ISO/IEC Directives Part 1<sup>[2]</sup> side by side on twelve points: seven where SD-4 deviates from the Directives' framework, and five where SD-4 is silent on Directive provisions within its declared scope.

---

## Revision History

### R0: August 2026

- Initial version.

---

## 1. Disclosure

The author provides information and serves at the pleasure of the committee.

The analysis is procedural.

This paper asks for nothing.

---

## 2. Two Documents

WG21 participants follow SD-4<sup>[1]</sup>. SD-4 describes itself as the "How We Work Cheat Sheet" and states that "everyone who participates in WG21 is expected to be familiar with this information." SD-4 does not appear in SD-1<sup>[6]</sup> (the WG21 document registry) for any year examined.

All JTC 1 working groups, including WG21, are bound by the ISO/IEC Directives Part 1<sup>[2]</sup><sup>[3]</sup>. The Directives exist to safeguard six WTO principles: Transparency, Openness, Impartiality and consensus, Effectiveness and relevance, Coherence, and Development dimension<sup>[3]</sup>.

Other WG21 standing documents - SD-3 (study groups), SD-5 (meeting information), SD-7 (mailing procedures) - complement the Directives with logistics. SD-4 is the only standing document that interprets the ISO consensus definition, creates voting thresholds, restricts National Body ballot comments, establishes escalation deadlines, and penalizes repeated objection.

**What the current system produces.** Four consecutive on-time releases from C++14 through C++23. Major compiler implementations tracking the standard within months of publication. A volunteer workforce that has produced modules, concepts, ranges, coroutines, `std::expected`, `std::print`, and `std::mdspan`. The procedural deviations documented below purchased something - speed, decisiveness, simplified administration. The cost is measured in Directive compliance.

---

## 3. What SD-4 Changes

Each entry places an SD-4 provision against the applicable Directive provision. Directive clause numbers refer to the ISO/IEC Directives Part 1 - Consolidated JTC 1 Supplement 2024<sup>[2]</sup>. Where a provision originates in the Consolidated ISO Supplement, the reference is to <sup>[3]</sup>.

### 3.1 Subgroup Chair Appointment

Directive 1.12.1<sup>[2]</sup>:

> "Working group Convenors shall be appointed by the committee for up to three-year terms. Such appointments shall be confirmed by the National Body (or liaison)."

SD-4<sup>[1]</sup>:

> "Subgroup chairs are appointed by the convener, and are selected to match the current needs of the subgroup. They have no fixed term."

| Requirement | Directives (1.12.1) | SD-4 |
|---|---|---|
| Appointing authority | The committee | The Convener |
| Term length | Up to three years | No fixed term |
| NB confirmation | Required | Not mentioned |

### 3.2 Consensus Threshold

Directive 2.5.6<sup>[2]</sup> (quoting ISO/IEC Guide 2:2004):

> "General agreement, characterized by the absence of sustained opposition to substantial issues by any important part of the concerned interests and by a process that involves seeking to take into account the views of all parties concerned and to reconcile any conflicting arguments. NOTE Consensus need not imply unanimity."

SD-4<sup>[1]</sup>:

> "A proposal normally advances if there are more than twice as many in favor of a proposal as against, after discussion of the concerns of those voting against and possibly a re-poll to see if opinions have improved."

SD-4 allows "each person in the room" to vote in subgroup polls<sup>[1]</sup>. The Directives require working group participants to be "Experts individually appointed by the P-members" and registered in the ISO Global Directory (1.12.1, 1.12.2)<sup>[2]</sup>.

### 3.3 Priority Allocation

SD-4<sup>[1]</sup>:

> "The direction group is a small by-invitation group of experienced participants who are asked to recommend priorities for WG21."

> "The design group chairs use that list to prioritize work at meetings."

The Directives define three types of subsidiary body: working groups (1.12)<sup>[2]</sup>, groups having advisory functions within a committee (1.13)<sup>[2]</sup>, and ad hoc groups (1.14)<sup>[2]</sup>. The Direction Group matches none. Advisory groups (1.13) are the closest functional match, but 1.13 is available to committees (TCs/SCs), not to working groups. Ad hoc groups (1.14) require committee-approved convenors, terms of reference, and a target completion date. The Direction Group has none of these.

### 3.4 Ballot Comment Scope

SD-4<sup>[1]</sup> declares two categories of NB ballot comments "not appropriate":

> "A ballot comment that requests adding an additional feature that is not already in the document is out of scope."

> "A ballot comment that requests a change that was already considered and decided otherwise at a WG21 meeting, and comes from a national body that was present at the meeting and had an opportunity to have their objections be heard and considered, is out of harmony with the ISO Code of Conduct's commitment to 'accept group decisions.'"

Directive 0.7(c)<sup>[2]</sup> places the responsibility for timely position-taking on the NB. SD-4 places a characterization on the comment itself. The Directives contain no provision authorizing a working group to pre-categorize NB ballot comments.

### 3.5 Escalation and Credibility

SD-4<sup>[1]</sup> states that escalation becomes inappropriate:

> "when a participant or national body regularly uses the escalation process to express a pattern of strong disagreement on topic after topic, which erodes their credibility."

SD-4 imposes a deadline: serious concerns must be posted "no later than the deadline of 5pm the evening before the closing plenary session"<sup>[1]</sup>. Objections not so escalated "should not be given weight in plenary."

The Directives contain no provision that penalizes repeated use of dispute resolution mechanisms and no cutoff after which objections lose standing.

### 3.6 Code of Conduct Attribution

SD-4 attributes to the ISO Code of Conduct the obligation to "accept group decisions."

The ISO Code of Ethics and Conduct<sup>[4]</sup>:

> "we accept and respect consensus decisions"

| | ISO Code of Ethics and Conduct<sup>[4]</sup> | SD-4<sup>[1]</sup> |
|---|---|---|
| Acceptance language | "accept and respect **consensus** decisions" | "accept **group** decisions" |
| Source | ISO Code of Ethics and Conduct (PUB100011) | Attributed to "ISO Code of Conduct"; phrase appears on a JTC 1 slide |
| Paired with | Full participant rights including escalation | Credibility erosion, 5pm objection deadline |

### 3.7 Meeting Record Transparency

SD-4<sup>[1]</sup>:

> "Meeting records of subgroup discussion, meeting wikis, and non-public committee email lists (aka reflectors)... often include personal positions and discussion. It is not allowed to quote from these publicly."

The Directives require meeting minutes circulated within 4 weeks (Annex SK, Clause SK.4)<sup>[3]</sup>, decisions posted within 48 hours (1.9.2(c))<sup>[2]</sup>, and use of the electronic platform "for transparency and traceability" (1.12.6)<sup>[2]</sup>. Directive SF.10<sup>[3]</sup> provides a consent-based framework for meeting recordings. The Directives' Foreword lists Transparency as the first of six WTO principles the procedures exist to safeguard<sup>[3]</sup>.

---

## 4. What SD-4 Does Not Mention

SD-4 describes itself as a comprehensive procedural guide. The following Directive provisions address topics within SD-4's scope but do not appear in SD-4.

| Directive Provision | Subject | SD-4 |
|---|---|---|
| 1.12.1 | Convenor "shall act in a purely international capacity" | |
| 5.1.1 | NB right of appeal to parent committee, TMB, and council board | |
| 5.1.2 | P-member may appeal any action "not in accordance with the ISO/IEC Directives" | |
| 5.3.4 | TMB Chair "shall form a conciliation panel" to hear appeals | |
| 1.13.2 | Advisory groups require committee approval of convenor, membership type, and terms of reference<sup>[3]</sup> | |

The right column is empty because SD-4 is silent on each provision.

---

## 5. Available Processes

Directive 1.4<sup>[2]</sup>:

> "Deviations from the procedures set out in the present document shall not be made without the authorization of the Chief Executive Officers of ISO or IEC or the technical management boards for deviations in the respective organizations."

The 2023 ISO Supplement<sup>[5]</sup> provided a Committee Specific Procedures mechanism: proposed CSPs submitted with a rationale to the TMB for review by the Directives Maintenance Team. SD-4 was not submitted through this process.

Directive 5.1.2<sup>[2]</sup>: a P-member of a committee may appeal against any action or inaction "not in accordance with" the Statutes and Rules of Procedure or the ISO/IEC Directives.

---

## References

[1] [SD-4](https://isocpp.org/std/standing-documents/sd-4-wg21-practices-and-procedures) - "SD-4: WG21 Practices and Procedures" (Guy Davidson, 2026). SD-4 provisions quoted in this paper predate the current convenor and were authored and maintained by Herb Sutter during his tenure as convenor through 2025.

[2] [JTC 1 Supplement 2024](https://www.pkn.pl/sites/default/files/sites/default/files/imce/files/ISO-IEC%20Consolidated%20JTC%201%20Supplement%202024.pdf) - "ISO/IEC Directives, Part 1 - Consolidated JTC 1 Supplement" (ISO/IEC, 2024). Hosted by the Polish Committee for Standardization, an ISO national body.

[3] [Consolidated ISO Supplement 2024](https://www.iso.org/sites/directives/current/consolidated/index.html) - "ISO/IEC Directives, Part 1 - Consolidated ISO Supplement" (ISO/IEC, 2024).

[4] [PUB100011](https://www.iso.org/files/live/sites/isoorg/files/store/en/PUB100011.pdf) - "ISO Code of Ethics and Conduct" (ISO, 2023).

[5] [Consolidated ISO Supplement 2023](https://agenturacas.gov.cz/wp-content/uploads/ISO-IEC-Directives-Part-1-with-ISO-Supplement-2023-PDF.pdf) - "Consolidated ISO Supplement" (ISO, 2023). Foreword paragraph h).

[6] [SD-1 2024](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/sd-1.htm) - "PL22.16/WG21 Document List" (WG21, 2024).
