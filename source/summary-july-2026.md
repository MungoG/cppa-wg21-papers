---
title: "A Reader's Guide to the July 2026 Mailing"
document: D0000R0
date: 2026-07-31
intent: info
audience: LEWG
reply-to:
  - "Vinnie Falco <vinnie.falco@gmail.com>"
---

## Abstract

Eight papers by one author in a single July 2026 mailing lay out the complete design space for runtime-checked undefined behavior - who configures it, how a program responds when a check fires, and what the choice costs - built on a decade of production hardening data and framed by two reforms to how the committee admits proposals and calls its votes.

This paper summarizes 8 papers published in the
July 2026 mailing. It is a reading guide: an executive summary
that identifies the logical series within the collection, describes
what each series delivers, and provides individual summaries of every
paper. It asks for nothing.

---

## 1. Disclosure

The author provides information and serves at the pleasure of the
committee.

This paper asks for nothing.

---

## 2. Executive Summary

Three papers set the two rival architectures for runtime-checking core-language undefined behavior side by side and let the committee's own record decide between them. P4306 places P3100's implicit contract assertions against the Profiles framework of P3589 and P3984, measures both against four criteria already in the record - existing practice, field experience, systematic coverage of undefined behavior, and freedom from dialects - and finds that, taken one by one, none of them settles who should own the configuration. P4317 supplies the Profiles answer in full: std::core_ub, a single profile that guards the 77 runtime-checkable cases of core-language undefined behavior with zero changes to the definitional machinery of the standard, where the competing routing demands six. P4297 isolates those six, reconstructing the seven consecutive polls that advanced P3100 to wording review, showing that six foundational clauses - not seventy-seven equal edits - carry the claim that Profiles are merely a preset on P3100's machinery, and documenting that a re-runnable search of 121 documents finds no published paper contesting the point. Read together, the three turn a diffuse standoff into a bounded decision about six wording clauses and one profile.

Three more papers take up what happens after a check detects a violation, and they widen the question before they answer it. P4308 shows that the four responses under discussion for a throwing implicit contract assertion are not the whole space - there are at least eight - restores the value-reporting noexcept operator as Option 0, and exposes a trilemma in which keeping the operator's value, unwinding the stack, and preserving the operator's meaning cannot all hold at once. P4310 argues the answer is to terminate: every hardened implementation the authors could find - libc++, libstdc++, MSVC STL, glibc, Android IntSan, UBSan, Abseil, Folly, WebKit, and Google's production fleet - terminates or traps on a detected core-language violation, and none makes continuation its production default. P4318 prices the one semantic that would continue - P3100's observe response - and concludes the committee would trade a permanent cost for a benefit that expires, since the slice would have to return roughly seven times the entire cross-implementation tax in a single year to break even. Together they establish that the response space is twice as wide as advertised, that deployment runs in one direction, and that the continue-semantic carries a bill the field has already declined to pay.

Two papers step back from any single feature to reform how the committee admits proposals and calls its votes. P4133 supplies a rule the committee has gone twenty-five years without writing - as of its 2024 revision the entire Library Evolution policy list is a single sentence about [[nodiscard]] - and replaces the two verdicts the room renders today with three: reject, not ready, or admit. Its eight named quantities reduce to arithmetic calibrated against the published admission record, and that record is the payoff: std::regex measured at 160 times slower than the fastest downloadable alternative and unfixable within nine years, the executor lineage that produced 115 distinct papers and 219 revision documents across fourteen years, a 2,679-page working draft that is 77 percent library text against a name index that quadrupled to 14,278 entries in fifteen years. P4302 addresses the vote itself, proposing a single bright-line rule - no poll on a paper unless the polled revision appeared in a pre-meeting mailing - and produces the record that motivates it: four std::execution revisions adopted at Croydon whose first mailing was dated the following month, and a Brno poll recording consensus on a P3100R7 that three independent link checks confirm never appeared in any mailing. The author discloses, at length, that he maintains competing proposals in the very area the rule would most constrain, and argues why that conflict cuts against his own case.

The value compounds at three levels. A single paper hands the reader a finished tool: P4317 a profile scored against all twelve SD-10 and D&E principles in the open, P4318 a cost model with marginal benefit, reach, an interaction tax, a discount rate, and a return-on-complexity threshold, P4308 an eight-by-six scoring table with every deployment cell backed by a citation. A cluster hands over something no single paper reaches: the five technical papers together render the whole runtime-checking decision - configuration ownership, response space, and cost - as one coherent map rather than a series of isolated votes, so a delegate can see how a choice on termination constrains a choice on configuration. And the full collection delivers a picture available from no cluster alone - that the substance of the core-language-checking debate and the process by which the committee will decide it are the same problem viewed twice, since the admission gate of P4133 and the mailing rule of P4302 are precisely the mechanisms that would have surfaced the six-clause architecture claim of P4297 and the priced observe-semantic of P4318 before any poll reached them.

Three entry points suit three readers. A delegate who must vote on P3100 and Profiles should start with P4306: it frames the entire comparison neutrally, names both proposals and the criteria that judge them, and makes every later paper legible. A reader drawn to the technical core of what a detected violation should do should start with P4310, the most self-contained argument in the set, then follow it into P4308 for the full option space and P4318 for the price of continuing. A reader concerned with committee process rather than the feature itself should start with P4133, which builds its criteria from arithmetic anyone can re-run, and then read P4302 to watch the same evidentiary discipline turned on the committee's own poll and mailing record.

---

## 3. Individual Papers

### 3.1. P4133R0 - Should WG21 Even See This Paper? Admission Gates for Library and Language Proposals

For twenty-five years the committee has asked whether it needs a rule for what belongs in the standard library and never written one - as of its 2024 revision, the entire Library Evolution policy list reads as a single sentence about `[[nodiscard]]`. This paper supplies the missing rule: a two-step admission gate that calibrates the room's expectation before advocacy frames it, then measures the paper against that expectation to render three verdicts - reject, not ready, or admit - rather than the two the committee renders today. The library instruments are eight named quantities reduced to arithmetic and calibrated against the published admission record, and that record is the reason to read on - `std::regex` measured at one hundred sixty times slower than the fastest downloadable alternative and unfixable within nine years, the executor lineage that produced 115 distinct papers and 219 revision documents across fourteen years because no verdict ever attached to the component, and a 2,679-page working draft that is 77 percent library text against a name index that quadrupled to 14,278 entries in fifteen years. Every figure is sourced to numbered committee documents, vendor issue trackers, and public minutes, so a reader who rejects the value model still keeps the criteria vacuum, the priced Complexity Budget, and the penalty ledger intact. The paper builds the library gate in full and then deliberately leaves the language gate open, marking the space for the compiler implementers and educators whose expertise it declines to counterfeit.

### 3.2. P4297R0 - Severing P3100's Profiles Claim from Its Case-by-Case Review

Seven consecutive polls advanced P3100 to case-by-case wording review, and not one of them ever adopted the architecture claim now riding along with it. This paper reconstructs that poll record from the committee's own history and the public tracker, then shows that six foundational wording clauses - not seventy-seven equal edits - carry a claim that Profiles are merely a preset layered on P3100's machinery, so the architecture is effectively settled once those six pass. It documents a decade of shipped deployment evidence, from hardened libc++ running across Google production at roughly 0.30% average cost to MSVC and libstdc++ hardening, against a proposal whose distinctive machinery ships nowhere and whose own text never once uses the word "experience." A disclosed, re-runnable search of 121 documents finds no published paper contesting the characterization, and the reversals of C++20 Contracts and the fourteen-revision P0443 executors design are cited as proof that undoing a settled architecture costs years. It asks EWG for three polls that let the wording proceed on its merits while forcing the ownership question onto a ballot written for it.

### 3.3. P4302R0 - Require One Published Mailing Before Any Poll

At two consecutive meetings the committee took recorded polls on paper revisions its own national body review chain had never received, and this paper produces the receipts. The mailing-date columns on open-std.org show four `std::execution` revisions adopted at Croydon in March 2026 whose first mailing was dated the following month, including one paper that narrowed three wording options to one and another born in the room with zero days of review. At Brno the pattern sharpened into a poll that recorded consensus on a P3100R7 that three independent link checks confirm does not exist in any mailing, while the short link members were pointed to still resolved to R6. Against this record the author proposes a single bright-line rule - no poll on a paper unless the polled revision appeared in a pre-meeting mailing, with one narrow final-meeting exception for wording corrections - and grounds it in prior art from a 2021 cooling-period proposal, a request from eighteen implementers to slow down, and the sibling C committee's existing deadline. The disclosure is unusually pointed - the author maintains competing proposals in the very feature area the rule would constrain most, and argues at length why that conflict cuts against, not for, the case he makes.

### 3.4. P4306R0 - Configuring Runtime Checking: Profiles and Implicit Contract Assertions

No deployment anywhere - not in a decade of production safety checking across three vendors' standard libraries, not in the Linux kernel, Chrome, or Apple's millions of bounds-checked lines of C - selects checking semantics per assertion in source or routes a failed check through a replaceable violation handler, and that one fact governs the whole comparison. This paper sets two proposals that answer a single question - how a program configures the runtime checking of core-language undefined behavior - side by side: P3100's implicit contract assertions, configured through the C++26 Contracts evaluation semantics and the Labels of P3400, against the Profiles framework of P3589 and P3984, under which a profile owns the guarantee directly. Measuring both against four criteria already in the committee's record - existing practice, deployment and field experience, systematic coverage of undefined behavior, and freedom from dialects - it finds that, taken one by one, those criteria settle configuration ownership for neither, and it backs the finding with a citable deployment ledger (Google's measured cost near 0.30 percent, WebKit's reported "zero", Firefox's "negligible") and on-the-record field statements from the sanitizer authors themselves that production checking terminates rather than continues. It then walks the committee's own graveyard - set_unexpected removed in C++17, C's Annex K constraint handler condemned by field-experience review, glibc's malloc hooks deleted as an exploit primitive - to test the layering's premise that one handler slot can be the base every checking facility is specified in terms of, and shows that where the unified model is said to deploy today, conformance is satisfied by containing none of its machinery.

### 3.5. P4308R0 - Eight Responses to a Throwing Implicit Contract Assertion

The four options EWG is weighing for a throwing implicit contract assertion are not the whole space - there are at least eight, and the two that let an exception escape are the only two nobody has implemented. This paper restores Option 0, the value-reporting `noexcept` operator that P3100's premise forecloses, adds Options E, F, and G, and scores all eight against six requirements drawn from P3100R8's own prose, exposing a trilemma in which keeping the operator's value, unwinding the stack, and preserving the operator's meaning cannot all hold at once. It backs every deployment claim cell by cell: terminate-on-escape ships since C++11 and in GCC 16.1's experimental Contracts, the trap response runs across hundreds of millions of lines at Google through libc++ hardening, `assert` ends in `abort()` across glibc, musl, the MSVC CRT, and bionic, while Options A and 0 have only capability analogues like GCC's `-fnon-call-exceptions`. It reconstructs the SG21 poll record, traces a security lineage from NDSS unwinding exploits to a Rust double-free CVE, and reproduces the exact one-definition-rule break that sinks the value-reporting operator through `std::is_nothrow_invocable_v`. The paper names no winner and requests nothing - it only shows that the ground before EWG is twice as wide as advertised.

### 3.6. P4310R0 - Hasta la Vista, Undefined Behavior: Why Implicit Contract Violations Should Terminate

Every hardened C++ implementation the authors could find - libc++, libstdc++, MSVC STL, glibc, Google's production fleet, Android IntSan, UBSan, Abseil, Folly, WebKit - terminates or traps on a detected core-language violation, and not one makes continuation its production default. From that uniform record the paper argues that when P3100R8's implicit contract assertions guard the 77 runtime-checkable cases of core-language undefined behaviour, the answer to what happens after the handler runs is to terminate, and it splits the disputed `observe` semantic into a "hook" that logs and a "continuation" that runs on, showing that every telemetry need survives termination while only execution past an undefined, possibly corrupted state is removed. It reads the availability-first objection on its own terms through Erlang's "let it crash" supervision and functional-safety practice, adds the exception-handling cost the libc++ team declined and the corrupted-state hazard drawn from Microsoft fail-fast, glibc, CERT ERR56-CPP, and the CHOP unwinding-exploit research, and answers with a terminating response that reuses the C++26 `enforce` semantic and the existing non-throwing-boundary rule, adding no new semantic and leaving `noexcept` honest. Even Bloomberg's `bsls_review`, the strongest deployed counter-example, is documented as a bounded adoption aid rather than a steady-state default, which is exactly the opt-in, non-portable shape the paper concedes to anyone who still must continue. The intent is `info`: it argues a position, discloses its authors' stakes plainly, and asks the committee for nothing.

### 3.7. P4317R0 - A Profile for Runtime-Checkable Core-Language Undefined Behavior: std::core_ub

The hardening that eight production systems already ship - from libc++ and libstdc++ to Apple's -fbounds-safety, Android's sanitizers, and Chrome's control-flow integrity - is precisely the form this paper proposes to standardize, and it does so with zero changes to the definitional machinery of the standard where the competing routing demands six. The proposal is `std::core_ub`, a single profile under the P3589R2 framework that guards the 77 runtime-checkable cases of core-language undefined behavior enumerated by Doumler and Berne in P3100R8: over an enforced region, a violated precondition ends the program rather than proceeding into undefined behavior. It leaves the meaning of the `noexcept` operator untouched, offers three deployed candidate responses to a violation (trap, diagnostic-then-abort, and a non-returning handler), and defines fixed replacement values for the 15 cases that admit one, so signed overflow wraps rather than crashes. It carries the field's numbers with it - Google's fleet-scale libc++ hardening measured at an average 0.30% overhead, a roughly 30% cut in segmentation faults, and more than 1,000 bugs surfaced during rollout - and it scores itself against all twelve SD-10 and D&E principles in the open, inviting any delegate who disputes a verdict to re-score the row.

### 3.8. P4318 - Transient Benefit, Perpetual Cost: Implicit Core-Language Assertions

This paper puts a price tag on a single evaluation semantic and concludes the C++ committee would be trading a permanent bill for a benefit that expires. It builds a cost model - marginal benefit, reach, an interaction tax, a discount rate, and a return-on-complexity threshold borrowed from library reasoning and retooled for a language feature - and points it at exactly one slice of P3100R8: the observe response that continues past core-language undefined state, standardized as a portable guarantee. The verdict lands twice over on independent grounds, first because libc++ and Bloomberg's BDE already ship the capability as opt-in build flags their own documentation scopes to an adoption period, driving the marginal value near zero, and second because a decaying finite benefit set against a perpetuity requires the slice to return roughly seven times the entire cross-implementation tax in a single year to break even. The perpetual cost is located precisely - in the exception-handling machinery the libc++ team stated on the record it will not generate, in the noexcept operator quietly changing meaning, and in the concept every C++ programmer must now carry - and pointedly not in runtime, where the ignore default sets it to zero.

---

## 4. Conclusion

This reading guide covers 8 papers from the July 2026 mailing.
The author hopes it helps the reader find the papers most relevant to
their work and interests.

---

## References

[1] P4133R0 - "Should WG21 Even See This Paper? Admission Gates for Library and Language Proposals" (Vinnie Falco, 2026).

[2] P4297R0 - "Severing P3100's Profiles Claim from Its Case-by-Case Review" (Vinnie Falco, 2026).

[3] P4302R0 - "Require One Published Mailing Before Any Poll" (Vinnie Falco, 2026).

[4] P4306R0 - "Configuring Runtime Checking: Profiles and Implicit Contract Assertions" (Vinnie Falco, 2026).

[5] P4308R0 - "Eight Responses to a Throwing Implicit Contract Assertion" (Vinnie Falco, 2026).

[6] P4310R0 - "Hasta la Vista, Undefined Behavior: Why Implicit Contract Violations Should Terminate" (Vinnie Falco, 2026).

[7] P4317R0 - "A Profile for Runtime-Checkable Core-Language Undefined Behavior: std::core_ub" (Vinnie Falco, 2026).

[8] P4318 - "Transient Benefit, Perpetual Cost: Implicit Core-Language Assertions" (Vinnie Falco, 2026).
