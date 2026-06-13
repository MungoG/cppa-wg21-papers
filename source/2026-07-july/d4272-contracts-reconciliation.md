---
title: "C++26 Contracts: The Form of the Resolution"
document: P4272R0
date: 2026-06-13
intent: info
audience: WG21
reply-to:
  - "Vinnie Falco <vinnie.falco@gmail.com>"
---

## Abstract

Every sustained objection to C++26 Contracts drew a response; for the contested capabilities, that response was deferral to features that do not yet exist.

The ISO/IEC Directives place the reconciliation obligation at the Working Group level, and WG21 satisfied it: P2900R14 was adopted into the C++26 working draft at the Hagenberg plenary. The Directives do not require reconciliation at the subgroup level, yet EWG engaged the opposition across four sessions, polled the contested design points individually, and adopted one change. This paper examines five technical arguments advanced against P2900 between June 2024 and November 2025 and the responses recorded in the two comprehensive proponent papers, P3846R1 and P2899R1. For the contested capabilities - guaranteed enforcement, bounded predicates, in-source semantic selection, and contracts on virtual functions and function pointers - the recorded resolution was a poll together with deferral to a named post-C++26 extension. A deferral closes the argument it answers only if that extension is delivered, and each is graded below against the public evidence for delivery.

---

## Revision History

### R0: July 2026 (post-Brno mailing)

- Initial revision.

---

## 1. Disclosure

The author provides information and serves at the pleasure of the committee.

The author maintains Boost.Beast and related networking libraries and is a WG21 participant. He has no position on whether P2900 should remain in or be removed from C++26. The technical evidence examined supports P2900 on its merits, and P2899R1<sup>[1]</sup> documents approximately five years of design work, on the order of 200 recorded polls, and two complete implementations.

The author was not present during SG21 deliberations (2020-2025); this analysis is constructed from published papers, official meeting minutes, the public SG15 archive, GitHub issue records, and the EWG session records identified in Section 2. The author initially believed the opposition's arguments had not been engaged; examination of the published responses and the EWG session records showed otherwise. The question this paper examines is therefore not whether the arguments were engaged but in what form the committee recorded their resolution.

This paper uses machine-assisted drafting.

This paper asks for nothing.

---

## 2. The Five Contested Questions

Five technical arguments were advanced against P2900 in published WG21 papers between June 2024 and November 2025. Each subsection states the argument with its public source, summarizes the answer recorded in the proponents' papers P3846R1<sup>[2]</sup> and P2899R1<sup>[1]</sup>, and records how the committee resolved it. Two of the five were resolved without deferral: ODR tooling (2.1) on quality of implementation, and the macro comparison (2.2) through distributed written engagement. The remaining three resolved by poll together with deferral to a named post-C++26 extension. Where the resolution was a deferral, the watch-list at the end of this section grades the deferred feature against the public evidence for its delivery.

### 2.1 ODR Tooling

- *The argument.* P2900 makes mixed-mode inline definitions ODR-equivalent. P3835R0<sup>[3]</sup>, SG15 message 2667<sup>[4]</sup>, and P3573R0<sup>[5]</sup> hold that this removes the normative basis on which an implementation could reject or diagnose inconsistent mixed-mode combinations.
- *The answer on record.* P3846R1<sup>[2]</sup> Concern 2 engages this directly: mixing translation units built with different flags is inherent to the C++ compilation model, and P2900, unlike macro-based assertions, keeps the behavior bounded to one of the compiled semantics. The paper enumerates strategies that preserve the chosen semantic to link time - up to encoding it in the mangled name or through symbol versioning - so selection is deterministic. The "no worse than if contract assertions did not exist" sentence is the worst case of the naive strategy alone.
- *The resolution.* The skeptic seeks a normatively mandated diagnosis; P2900 permits the deterministic mechanisms but does not require them. EWG polled "add ODR to contracts": consensus against (SF:6 F:5 N:10 A:25 SA:17)<sup>[6]</sup>. The resolution rests on quality of implementation, not a future feature. The skeptics maintained the concern after the response: P3835R0<sup>[3]</sup> and the public SG15 archive<sup>[4]</sup> characterize "no worse than if contract assertions did not exist" as a status-quo comparison rather than an answer, holding that making mixed-mode definitions ODR-equivalent removes the normative basis a checker would use to reject them - that it "disables such checkers"<sup>[4]</sup> rather than enabling them. They also pressed the converse question - what practical benefit the ODR-equivalence of mixed-mode inline definitions provides - and framed the change as a net loss of diagnostic capability rather than a gain<sup>[4]</sup>.

### 2.2 Macro Comparison

- *The argument.* SG15 message 2871<sup>[7]</sup> and P3835R0<sup>[3]</sup> enumerate six dimensions on which macro-based assertion facilities differ from P2900: header consistency, exception-mapping control, violation-handler selection, the absence of constification, code-generation cost, and semantic transparency.
- *The answer on record.* Each dimension is engaged in the published responses, in separate places: exception mapping in P3846R1<sup>[2]</sup> Concern 11, violation-handler selection in Concern 9, constification in Concern 8, semantic transparency and the ODR in Concerns 2 and 4, and code-generation cost in Concerns 2 and 11 (an ignored assertion reduced to about two instructions; no overhead for non-throwing predicates).
- *The resolution.* No capability is deferred. What the papers do not contain is a single assembled, side-by-side comparison against the six dimensions as a set; the engagement is real but distributed. Assembled from the published responses, the mapping is:

| Dimension | Engaged in P3846R1<sup>[2]</sup> | Status |
|-----------|----------------------------------|--------|
| Header consistency | Concerns 2, 4 | Engaged via the mixed-mode/ODR analysis, not as a distinct point |
| Exception-mapping control | Concern 11 | Engaged directly |
| Violation-handler selection | Concern 9 | Engaged directly |
| Absence of constification | Concern 8 | Engaged directly |
| Code-generation cost | Concerns 2, 11 | Engaged directly |
| Semantic transparency | Concerns 2, 4 | Engaged via the ODR/semantic analysis |

Each dimension is engaged somewhere in the published responses; none is assembled by the proponents as a point-by-point reply to the enumeration as a set.

### 2.3 Guaranteed Enforcement

- *The argument.* P3835R0<sup>[3]</sup>, P3573R0<sup>[5]</sup>, and SG15 message 2978<sup>[8]</sup> hold that a facility is needed for guaranteed non-continuation past a failed check, expressed at the declaration and not removable by the function definition.
- *The answer on record.* P3846R1<sup>[2]</sup> Concern 1 offers four present-day paths to non-ignorable checks: build the defining translation unit with a checked semantic; duplicate the check with `pre` on the declaration and an `if` in the body; use a vendor attribute such as Clang's `contract_semantic("quick_enforce")`; or use an `if` statement until the extension lands. P2899R1<sup>[1]</sup> Section 2.3 lists the in-language guarantee under "Features Not Proposed" and names the labels extension, P3400R1, as the eventual mechanism.
- *The resolution.* The skeptic seeks an in-language, definition-proof guarantee, which P2900 does not provide; the response concedes this and defers it to labels (P3400R1), which would also supply in-source semantic selection. EWG declined mandatory enforcement at Tokyo (March 2024): an "enforce semantics only" poll reached consensus against (SF:6 F:1 N:3 A:15 SA:24)<sup>[6]</sup>. The skeptics characterize the four present-day paths as workarounds that do not meet the stated requirement. In the SG15 thread that requirement was given its sharpest form as a "Will Not Continue" guarantee<sup>[8]</sup>: non-continuation past a failed check, fixed at the function declaration and not removable by the definition - one the definition "cannot fail to deliver and cannot forget to deliver"<sup>[8]</sup>. An `if` in the body supplies a check the definition can still drop or alter, which is the property a declaration-level guarantee exists to exclude.

### 2.4 Bounded Predicates

- *The argument.* P3506R0<sup>[9]</sup>, P3362R0<sup>[10]</sup>, and P3573R0<sup>[5]</sup> seek bounded or strict predicates that a compiler could constrain, including the narrower position that an arithmetic-only form would serve.
- *The answer on record.* P3846R1<sup>[2]</sup> Concern 7 argues that enforcing side-effect-freedom would make "all but the most primitive C++ expressions" ill-formed (citing P3499R1), and distinguishes destructive from merely observable side effects. P2899R1<sup>[1]</sup> records that strict predicates were judged too restrictive for the runtime-checking use case P2900 prioritizes.
- *The resolution.* The skeptic seeks a defined, checkable predicate sublanguage; the response holds the broad direction infeasible and the arithmetic-only form insufficiently developed. The narrower arithmetic-only position was advanced as a concrete fallback in P3362R0<sup>[10]</sup> and P3506R0<sup>[9]</sup>; the response judged it insufficiently developed rather than adopting it. The direction, pursued across P2680R1, P3285R0, and P3362R0, has not achieved consensus in SG21, SG23, or EWG.

### 2.5 Design Foreclosure

- *The argument.* P3573R0<sup>[5]</sup> and P3829R0<sup>[11]</sup> hold that P2900's design may foreclose later addition of the capabilities it omits, including guaranteed enforcement and contracts on virtual functions and function pointers.
- *The answer on record.* P3846R1<sup>[2]</sup> Concern 15 analyzes the two interactions P3829R0 names: deep const (no proposal in four decades, and it would first have to resolve the const-ification questions) and decorators (orthogonal to assertions, since decorators run once and may mutate while contract assertions are ghost code that may run zero or many times). Concern 14 addresses the concrete omissions: contracts on virtual functions have a fully specified proposal, P3097R0, already EWG-approved, CWG-reviewed, and implemented in GCC; contracts on function pointers were explored in P3327R0 and found infeasible as a direct feature, requiring a novel facility such as function types with usage (P3271R1); in-source grouping is deferred to labels (P3400R1).
- *The resolution.* The skeptic seeks assurance that the omitted capabilities remain addable without an ABI break or redesign; the response points to the named extensions. EWG removed contracts on virtual functions from P2900 (SF:20 F:24 N:13 A:14 SA:2, consensus in favor of disallowing)<sup>[6]</sup>, and at Kona declined to pursue re-adding them (ES-072: SF:7 F:7 N:15 A:30 SA:13) or adding function pointers (ES-074: SF:4 F:2 N:2 A:22 SA:38) for C++26<sup>[6]</sup>. The skeptics put the foreclosure concern more sharply than addability: P3829R0<sup>[11]</sup> holds that P2900 occupies the syntax and design space a stronger facility would need, so shipping it first risks blocking the better successor rather than paving the way for it.

This analysis also draws on four EWG session records:

- [EWG Minutes, St. Louis, June 2024](https://wiki.isocpp.org/2024-06_St_Louis:NotesEWGP3097)<sup>[12]</sup>
- [EWG Minutes, Wroc&lstrok;aw, November 2024](https://wiki.isocpp.org/2024-11_Wroclaw:NotesEWGContracts)<sup>[13]</sup>
- [EWG Minutes, Hagenberg, February 2025](https://wiki.isocpp.org/2025-02_Hagenberg:NotesEWGContracts)<sup>[14]</sup>
- [EWG Minutes, Kona, November 2025](https://wiki.isocpp.org/2025-11_Kona:EWGContractsVirtualAndPointers)<sup>[15]</sup>

Access requires WG21 wiki credentials. The author examined these four records and found substantive technical engagement with the contested arguments at each session: at St. Louis, EWG reviewed the virtual-functions proposal (P3097R0) in technical detail before adopting it into P2900; at Wroc&lstrok;aw, the room discussed the contested design points before the forwarding poll (SF:25 F:17 N:0 A:3 SA:12)<sup>[6]</sup>; at Hagenberg, each contested point was discussed and then polled individually (Section 3); at Kona, the virtual-functions and function-pointer questions were taken up before the ES-072 and ES-074 polls<sup>[6]</sup>. SD-4<sup>[16]</sup> permits public quotation of straw-poll questions and numeric results but not of the discussion itself, so the characterizations above are broad paraphrase rather than quotation; WG21 participants can verify them against the credentialed records, and other readers rely on the author's representation.

### Deferral watch-list

The contested capabilities were resolved by deferral to named extensions; each deferral closes its argument only if the extension is delivered. The grades reflect the public evidence today, and the delivery status of each extension is drawn from the proponents' own papers.

- **Contracts on virtual functions** - P3097R0, reported by P3846R1<sup>[2]</sup> Concern 14 as fully specified, approved with strong consensus in EWG, CWG-reviewed, and implemented in GCC. Risk low for eventual delivery; for C++26, EWG declined to pursue it (Kona ES-072)<sup>[6]</sup>.
- **Guaranteed enforcement and in-source semantic selection** - the labels extension, P3400R1. Risk medium: P3846R1<sup>[2]</sup> reports it as actively pursued but not yet at consensus in any subgroup.
- **Contracts on function pointers** - a novel facility, since P3327R0 found direct support infeasible; function types with usage (P3271R1) is one candidate. Risk high: no viable design exists yet, and the Kona poll on pursuing it for C++26 did not reach consensus (ES-074)<sup>[6]</sup>.
- **Bounded or strict predicates** - a strict-predicates facility along the lines of P3499R1. Risk high: the direction has not achieved consensus after several years and multiple proposals (P2680R1, P3285R0, P3362R0).

---

## 3. The Poll Record

The chair's official poll postings in the GitHub issue tracker<sup>[6]</sup>, corroborated by P2899R1<sup>[1]</sup>, record the EWG polls. The Hagenberg session (February 2025) polled the contested design points individually. The numbers below are quoted from that public posting.

| Concern | Poll result |
|---------|-------------|
| Remove constification | SF:9 F:7 N:6 A:37 SA:14 - consensus against |
| Unconditionally unwind exceptions | SF:12 F:18 N:11 A:15 SA:7 - no consensus for change |
| Reduce UB in contracts | SF:9 F:7 N:9 A:22 SA:19 - consensus against |
| Add ODR to contracts | SF:6 F:5 N:10 A:25 SA:17 - consensus against |
| Disallow pre/post on virtual functions | SF:20 F:24 N:13 A:14 SA:2 - consensus in favor |
| Remove P2900 from C++26 | SF:9 F:8 N:3 A:19 SA:41 - consensus against |

One change was adopted in response to the room on the contested design points: contracts on virtual functions were removed from P2900. A separate objection was also resolved by an adopted change: at Kona (November 2025) EWG voted to adopt the wording proposed in P3878R0<sup>[17]</sup>, forbidding the `observe` semantic for hardened standard-library preconditions (SF:49 F:14 N:5 A:0 SA:0, consensus)<sup>[18]</sup>; the change shipped as P3878R1<sup>[19]</sup>, which records that the authors of the standard-library hardening proposal consider it a bug fix and that the national-body comment submitters confirmed it resolves their concern. The earlier forwarding poll at Wroc&lstrok;aw (November 2024) recorded SF:25 F:17 N:0 A:3 SA:12, and SG21 recorded consensus against a Technical Specification ship vehicle (SF:1 F:3 N:3 A:12 SA:10).

These numbers record the room's disposition toward each change, a different record from a per-argument determination on the merits (Section 4). That disposition was not static across the period: contracts on virtual functions were adopted into P2900 at St. Louis and removed at Hagenberg<sup>[6]</sup>.

---

## 4. Compliance and the Form of Resolution

The reconciliation obligation in the consensus definition (ISO/IEC Guide 2:2004, cl. 2.5.6)<sup>[20]</sup> binds at the Working Group level, and WG21 met it: P2900R14 was adopted at the Hagenberg plenary. Sustained opposition was present - P3573R0<sup>[5]</sup> carried nine co-authors, and national-body comments were filed by multiple delegations<sup>[21]</sup> - and the Directives provide that the obligation is to address such opposition, not to resolve it: "A sustained opposition is not akin to a right to veto [...] The obligation to address the sustained oppositions does not imply an obligation to resolve them successfully." SD-4<sup>[16]</sup> leaves the internal organization of subgroups unspecified, so EWG reconciliation is a voluntary practice, and the EWG session records reside on a credentialed wiki rather than in the N-document series.

Within that practice, two records should be separated. A poll measures the disposition of the room toward a proposed change, and one was taken at each session. A determination - a recorded finding that a contested argument was met on its merits, as distinct from a record that the room declined the change - is a different artifact. The substantive engagement exists, in the proponents' published responses and the session discussions, but no document in the examined record classifies any of the five arguments as met-on-the-merits rather than declined-by-vote. One narrower concern outside the five shows the contrasting form: the standard-library hardening objection in P3878R0<sup>[17]</sup> produced an adopted wording change for C++26, recorded in P3878R1<sup>[19]</sup> together with the national-body submitters' confirmation that it resolved their concern - a determination, not only a poll. RFC 7282<sup>[22]</sup>, the IETF statement on consensus, draws the same line: a hum measures the room, while addressing an objection means engaging its substance rather than counting it. Polling is what the rules require, and WG21 met it; a poll result and a per-argument determination are different records, and for the five, the committee recorded the first.

---

## 5. Conclusion

Five technical arguments. Four EWG sessions. Two comprehensive response papers. Sixteen months. Each was heard, engaged on its merits, and polled.

WG21 performed the procedure the Directives require: opposition was heard, registered, and polled, and the work continued. The reconciliation obligation binds at the Working Group level, where it was met when P2900R14 was adopted at plenary; EWG was not obligated to reconcile at the subgroup level and did so anyway. For the contested capabilities, the recorded resolution was a poll together with deferral to a named post-C++26 extension. Two features of the record remain: the comprehensive written responses are proponent-authored, and the recorded resolution of each of the five arguments is the poll, not a determination that the argument was met on its merits. One narrower concern outside the five, standard-library hardening, did receive a determination: an adopted change, recorded together with the national-body submitters' confirmation that it resolved their concern.

A deferral is a conditional answer. It closes the argument it was given to answer on the day the promised feature ships, and not before.

---

## References

[1] [P2899R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p2899r1.pdf) - "Contracts for C++ - Rationale" (Joshua Berne, Timur Doumler, Rostislav Khlebnikov, Andrzej Krzemie&nacute;ski, 2025).

[2] [P3846R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p3846r1.pdf) - "C++26 Contract Assertions, Reasserted" (Timur Doumler, Joshua Berne, et al., 2026).

[3] [P3835R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3835r0.html) - "Contracts make C++ less safe - full stop!" (John Spicer, Ville Voutilainen, Jose Daniel Garcia Sanchez, 2025).

[4] [SG15 Message 2667](https://lists.isocpp.org/sg15/2025/10/2667.php) - Ville Voutilainen (2025-10-14).

[5] [P3573R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3573r0.pdf) - "Contract concerns" (Michael Hava, J. Daniel Garcia Sanchez, Ran Regev, Gabriel Dos Reis, John Spicer, Bjarne Stroustrup, J.C. van Winkel, David Vandevoorde, Ville Voutilainen, 2025).

[6] [cplusplus/papers#1648](https://github.com/cplusplus/papers/issues/1648) - "P2900 Contracts for C++ (GitHub issue tracker)" (WG21, 2024-2025).

[7] [SG15 Message 2871](https://lists.isocpp.org/sg15/2025/10/2871.php) - John Spicer (2025-10-20).

[8] [SG15 Message 2978](https://lists.isocpp.org/sg15/2025/10/2978.php) - Ville Voutilainen (2025-10-27).

[9] [P3506R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3506r0.pdf) - "P2900 Is Still Not Ready for C++26" (Gabriel Dos Reis, 2025).

[10] [P3362R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p3362r0.pdf) - "Static analysis and 'safety' of Contracts, P2900 vs. P2680/P3285" (Ville Voutilainen, 2024).

[11] [P3829R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3829r0.pdf) - "Contracts do not belong in the language" (David Chisnall, John Spicer, Ville Voutilainen, Gabriel Dos Reis, Jose Daniel Garcia Sanchez, 2025).

[12] [EWG Minutes, St. Louis, June 2024](https://wiki.isocpp.org/2024-06_St_Louis:NotesEWGP3097) - "NotesEWGP3097" (WG21 wiki, credentialed access, 2024).

[13] [EWG Minutes, Wroc&lstrok;aw, November 2024](https://wiki.isocpp.org/2024-11_Wroclaw:NotesEWGContracts) - "NotesEWGContracts" (WG21 wiki, credentialed access, 2024).

[14] [EWG Minutes, Hagenberg, February 2025](https://wiki.isocpp.org/2025-02_Hagenberg:NotesEWGContracts) - "NotesEWGContracts" (WG21 wiki, credentialed access, 2025).

[15] [EWG Minutes, Kona, November 2025](https://wiki.isocpp.org/2025-11_Kona:EWGContractsVirtualAndPointers) - "EWGContractsVirtualAndPointers" (WG21 wiki, credentialed access, 2025).

[16] [SD-4](https://isocpp.org/std/standing-documents/sd-4-wg21-practices-and-procedures) - "WG21 Practices and Procedures" (WG21, 2024).

[17] [P3878R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3878r0.html) - "C++26 Contracts are not a good fit for standard library hardening" (Ville Voutilainen, Jonathan Wakely, John Spicer, Stephan T. Lavavej, 2025).

[18] [cplusplus/papers#2532](https://github.com/cplusplus/papers/issues/2532) - "P3878 Standard library hardening should not use the 'observe' semantic (GitHub issue tracker)" (WG21, 2025).

[19] [P3878R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3878r1.html) - "Standard library hardening should not use the 'observe' semantic" (Ville Voutilainen, Jonathan Wakely, John Spicer, Stephan T. Lavavej, 2025).

[20] [ISO/IEC Directives, Part 1](https://www.iso.org/sites/directives/current/consolidated/) - "Consolidated ISO Supplement - Procedures specific to ISO" (ISO/IEC, 2024). Consensus definition from ISO/IEC Guide 2:2004, cl. 2.5.6.

[21] [N5028](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/n5028.pdf) - "C++26 CD summary of voting and comments" (Herb Sutter, 2025).

[22] [RFC 7282](https://datatracker.ietf.org/doc/html/rfc7282) - "On Consensus and Humming in the IETF" (Pete Resnick, 2014).
