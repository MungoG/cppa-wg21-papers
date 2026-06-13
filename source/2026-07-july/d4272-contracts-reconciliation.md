---
title: "Contracts Reconciliation - Process and Documentation"
document: P4272R0
date: 2026-07-01
intent: info
audience: WG21
reply-to:
  - "Vinnie Falco <vinnie.falco@gmail.com>"
---

## Abstract

WG21 reconciled on Contracts. It was not obligated to.

The ISO/IEC Directives describe reconciliation but do not mandate it for WG internal subgroup decisions. WG21 performed it voluntarily: P2900R14 was adopted 100-14-12 at Hagenberg after named objectors spoke at length, six contested design points were individually polled, and one change was adopted in response. Because no formal internal rule governs reconciliation documentation, the record of that process was authored exclusively by one faction. This paper examines the gap between what WG21 did and what the record shows.

---

## Revision History

### R0: July 2026 (post-Brno mailing)

- Initial revision.

---

## 1. Disclosure

The author provides information and serves at the pleasure of the committee.

This paper uses machine-assisted drafting.

The author maintains Boost.Beast and related networking libraries. He is a WG21 participant. He has no position on whether P2900 should remain in or be removed from C++26. The technical evidence examined supports P2900 on its merits.

The author was not present during SG21 deliberations (2020-2025). This analysis is constructed from published papers, official meeting minutes, the public SG15 archive, and GitHub issue records. The author initially believed reconciliation had not occurred. Further examination of EWG session records showed otherwise. This paper reflects the corrected finding.

This paper asks for nothing.

---

## 2. The ISO Reconciliation Model

The ISO/IEC Directives<sup>[1]</sup> (cl. 2.5.6) define consensus:

> General agreement, characterized by the absence of sustained opposition to substantial issues by any important part of the concerned interests and by a process that involves seeking to take into account the views of all parties concerned and to reconcile any conflicting arguments. Consensus need not imply unanimity.

The Directives specify the procedure for sustained opposition<sup>[1]</sup>:

> Those expressing sustained opposition have a right to be heard [...] The leadership will register the opposition (i.e. in the minutes, records, etc.) and continue to lead the work on the document.

And the limits of that obligation<sup>[1]</sup>:

> A sustained opposition is not akin to a right to veto. The obligation to address the sustained oppositions does not imply an obligation to resolve them successfully.

These provisions regulate decisions at the Working Group level - plenary votes, DIS ballots, and formal committee resolutions. The Directives intentionally leave unspecified how a WG organizes its internal subgroups. SD-4<sup>[2]</sup> governs WG21's internal operations: how subgroup chairs run sessions, how straw polls are interpreted, and how papers progress through the pipeline. SD-4 does not require reconciliation at the subgroup level. No formal rule mandates that EWG or LEWG document how minority concerns were addressed before forwarding a paper.

The Directives also describe a transparency model (cl. 1.12.6)<sup>[1]</sup>:

> For transparency and traceability, the electronic platform provided by the Office of the CEO shall be used for the circulation of WG documents and communication with members.

This clause applies to WG-level document circulation. It provides a model for what transparent records look like, without mandating that subgroups follow the same practice.

---

## 3. What WG21 Did

Despite no formal obligation to reconcile at the subgroup level, WG21 performed substantive reconciliation on P2900. The following is documented in public records:

**Opposition was formally expressed.** Nine co-authors published P3573R0<sup>[3]</sup> ("Contract concerns"): Michael Hava, J. Daniel Garcia Sanchez, Ran Regev, Gabriel Dos Reis, John Spicer, Bjarne Stroustrup, J.C. van Winkel, David Vandevoorde, and Ville Voutilainen. Additional papers included P3506R0<sup>[4]</sup> (Microsoft), P3829R0<sup>[5]</sup>, and P3835R0<sup>[6]</sup>.

**Specific design concerns were individually polled at Hagenberg EWG** (February 2025)<sup>[7]</sup>:

| Concern | Poll result |
|---------|-------------|
| Remove constification | SF:9 F:7 N:6 A:37 SA:14 - consensus against |
| Unconditionally unwind exceptions | SF:12 F:18 N:11 A:15 SA:7 - no consensus for change |
| Reduce UB in contracts | SF:9 F:7 N:9 A:22 SA:19 - consensus against |
| Add ODR to contracts | SF:6 F:5 N:10 A:25 SA:17 - consensus against |
| Disallow pre/post on virtual functions | SF:20 F:24 N:13 A:14 SA:2 - consensus in favor |
| Remove P2900 from C++26 | SF:9 F:8 N:3 A:19 SA:41 - consensus against |

**One change was adopted** in direct response to room discussion: contracts on virtual functions were removed from P2900.

**NB comments were processed.** National body comments on contracts were assigned to EWG at Kona<sup>[8]</sup>. Comments requesting removal or deferral included ES-049, ES-050, US 25-052, FR-004-053, FR-005-054, and FI-071. Two comments were accepted and produced design changes.<sup>[9]</sup>

**Two comprehensive response papers exist.** P2899R1<sup>[10]</sup> (150 pages, approximately 204 poll records documenting alternatives considered for each design decision) and P3846R1<sup>[9]</sup> (18 numbered concerns with Summary, Discussion Status, Response, and Details sections for each).

---

## 4. The Documentation Gap

Because reconciliation is voluntary and unformalized in SD-4, its documentation depends on individual chair practice. The result is asymmetric.

**P3846R1 carries 22 co-authors.** All 22 support P2900. No opponent co-authored or validated the document. The determination of whether 18 objections were "adequately addressed" was made exclusively by those being objected to.

**P2899R1 frames opposition as "sustained."** The paper characterizes P3506R0 and P3573R0 as "not new issues but restatements of known, sustained opposition."<sup>[10]</sup> In the Directives' reconciliation model, "sustained opposition" is a procedural status that triggers engagement - not a pejorative characterization. The terminological tension between the Directives' meaning and the paper's usage illustrates what happens when reconciliation language is borrowed informally rather than adopted as procedure.

**Detailed session records exist but are not publicly accessible.** EWG meeting notes documenting the Hagenberg contracts discussion reside on a wiki accessible to WG21 members. The Directives' transparency model (cl. 1.12.6) provides one approach to making such records traceable, though WG21 is not formally bound to follow it at the subgroup level.

**No jointly-validated summary exists.** No document carries the signature or co-authorship of both factions confirming that the minority's concerns were heard and the majority's responses were substantive. The absence of such a document allows the minority to credibly assert that their concerns were polled but not reconciled - even when the room record suggests otherwise. A formal rule requiring bilateral acknowledgment would foreclose this narrative.

---

## 5. The Public Thread Record

The SG15 mailing list archive for October 2025<sup>[11]</sup> contains a 355-message thread on P3835R0. One exchange illustrates the documentation gap in operation.

Oliver Rosten (message 2816<sup>[12]</sup>) raised a counter-example to the claim that P2900's mixed-mode behavior is novel:

> You can compile one TU with fast-math and one without. The header code they consume is token-identical. There is no ODR violation but you can get differences depending on which version the linker chooses.

John Spicer responded (message 2848<sup>[13]</sup>):

> If you use compiler options that explicitly change the meaning of code, then you should know what they do. With contracts, you use facilities that the standard endorses and says should work fine, but you get surprising results.

Oliver followed up (message 2851<sup>[14]</sup>):

> I have given an example which I think shows that this is a pre-existing problem. But I cannot see anywhere where you have acknowledged this.

This is not evidence that reconciliation failed. It is evidence that email threads are structurally poor venues for reconciliation - arguments can go unacknowledged without consequence because no chair manages the process and no summary captures the resolution.

---

## 6. Cross-Body Comparison

Several standards bodies have formalized reconciliation practices. The table below shows what formalization produces:

| Practice | WG14 | TC39 | IETF | ARG | WG21 |
|----------|------|------|------|-----|------|
| Chair interprets vote meaning | Yes | Yes | Yes | Yes | In room, not in formal record |
| Opposition registered in minutes | Yes | Yes | Yes | Yes | In wiki, not in N-documents |
| Bilateral response document | N/A | N/A | Chair decides | N/A | Proponent-only |
| Alternatives enumerated before resolution | Yes | N/A | Yes | Yes | In papers and wiki |

Sources: WG14 N3227<sup>[15]</sup>, TC39 meeting notes<sup>[16]</sup>, RFC 7282<sup>[17]</sup>, ARG minutes<sup>[18]</sup>.

Bodies with formal reconciliation rules produce consistent bilateral documentation. WG21's voluntary practice produces the same substance but inconsistent records - because the documentation practice depends on which chair runs the session and whether they choose to formalize the outcome.

---

## 7. Conclusion

The distance between reconciliation and its documentation is the distance between what happened in the room and what the paper trail shows to those who were not there.

WG21 reconciled on Contracts without being obligated to. Six concerns were polled individually. One was accepted. That the committee performed this voluntarily suggests it values the practice. The formal documentation of that reconciliation was authored by one faction - not because anyone acted improperly, but because no rule required otherwise. Formalizing what already happens in substance would produce consistent documentation regardless of which chair runs the session.

The process worked. The record of the process is one-sided. These are both true.

---

## References

[1] [ISO/IEC Directives, Part 1](https://www.iso.org/sites/directives/current/consolidated/) - "Consolidated ISO Supplement - Procedures specific to ISO" (ISO/IEC, 2023). Consensus definition from ISO/IEC Guide 2:2004, cl. 2.5.6.

[2] [SD-4](https://isocpp.org/std/standing-documents/sd-4-wg21-practices-and-procedures) - "WG21 Practices and Procedures" (WG21, 2024).

[3] [P3573R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3573r0.pdf) - "Contract concerns" (Michael Hava, J. Daniel Garcia Sanchez, Ran Regev, Gabriel Dos Reis, John Spicer, Bjarne Stroustrup, J.C. van Winkel, David Vandevoorde, Ville Voutilainen, 2025).

[4] [P3506R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3506r0.pdf) - "P2900 Is Still Not Ready for C++26" (Gabriel Dos Reis, 2025).

[5] [P3829R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3829r0.pdf) - "Contracts do not belong in the language" (David Chisnall, John Spicer, Ville Voutilainen, Gabriel Dos Reis, Jose Daniel Garcia Sanchez, 2025).

[6] [P3835R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3835r0.html) - "Contracts make C++ less safe - full stop!" (John Spicer, Ville Voutilainen, Jose Daniel Garcia Sanchez, 2025).

[7] [cplusplus/papers#1648](https://github.com/cplusplus/papers/issues/1648) - "P2900 Contracts for C++ (GitHub issue tracker)" (WG21, 2024-2025).

[8] [N5028](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/n5028.pdf) - "C++26 CD summary of voting and comments" (Herb Sutter, 2025).

[9] [P3846R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p3846r1.pdf) - "C++26 Contracts, reasserted" (Timur Doumler, Joshua Berne, et al., 2026).

[10] [P2899R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p2899r1.pdf) - "Contracts for C++ - Rationale" (Joshua Berne, Timur Doumler, Dmitry Khlebnikov, Andrzej Krzemienski, 2025).

[11] [SG15 Archive, October 2025](https://lists.isocpp.org/sg15/2025/10/subject.php) - ISOCPP SG15 Mailing List (public archive).

[12] [SG15 Message 2816](https://lists.isocpp.org/sg15/2025/10/2816.php) - Oliver Rosten (2025-10-20).

[13] [SG15 Message 2848](https://lists.isocpp.org/sg15/2025/10/2848.php) - John Spicer (2025-10-20).

[14] [SG15 Message 2851](https://lists.isocpp.org/sg15/2025/10/2851.php) - Oliver Rosten (2025-10-20).

[15] [N3227](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3227.htm) - "WG14 Minutes, January 2024, Strasbourg" (WG14, 2024).

[16] [TC39 Meeting Notes, July 2023](https://github.com/tc39/notes/blob/HEAD/meetings/2023-07/july-12.md) - (TC39, 2023).

[17] [RFC 7282](https://datatracker.ietf.org/doc/html/rfc7282) - "On Consensus and Humming in the IETF" (Pete Resnick, 2014).

[18] [ARG Minutes, Meeting 62T](http://www.ada-auth.org/ai-files/minutes/min-2301.html) - (WG9/ARG, 2023).
