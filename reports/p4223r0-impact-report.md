---
date: 2026-06-09
title: "P4223R0 Impact Analysis"
---

## P4223R0 Impact Analysis

P4223R0 "Towards Senders in Interfaces" (Ian Petersen, 2026-05-11) proposes `std::execution::function` (a type-erasing sender for separately-compiled async functions) and `std::execution::get_frame_allocator` (a receiver environment query for controlling operation state allocation). This report identifies which papers in the reserve and active queue interact with P4223R0 and what changes are suggested.

---

## Direct Technical Impact

These papers should cite P4223R0 or update their arguments in light of its proposals.

### P4127 - The Coroutine Frame Allocator Timing Problem

P4223R0 independently validates P4127's central thesis. Petersen's `get_frame_allocator` is the sender-world equivalent of the out-of-band delivery mechanism P4127 proves is one of exactly two paths. Petersen's observation that "operation states are analogous to stack frames" and his citation of P4712R2 Section 6.3 directly validates P4127's recycling analysis.

**Suggested change:** Add a citation to P4223R0 as independent confirmation that the timing constraint is real, the design space is closed, and the sender model arrived at the same conclusion.

### P4172 - IoAwaitable for Coroutine-Native Byte-Oriented I/O

P4172R2 already references Petersen's work in Section 11.1 and the Acknowledgements, citing `exec::function` and the GitHub PR.

**Suggested change:** Upgrade the citation from the GitHub PR/source link to the formal paper number P4223R0. No substantive argument changes needed.

### P4007 - Open Issues in `std::execution::task`

P4223R0's `get_frame_allocator` directly addresses P4007's "Allocator Propagation" issue, which states that "frame allocator propagation remains absent" and that the current design "forecloses automatic frame allocator propagation through the coroutine call tree."

**Suggested change:** Note that P4223R0 proposes environment-based frame allocator propagation via a receiver query, providing a sender-side mechanism that bypasses the coroutine `operator new` timing constraint.

### P4123 - The Cost of Senders for Coroutine I/O

P4223R0 relates to P4123's analysis of `any_sender` requiring per-operation heap allocation for type-erased streams. `function` provides a different allocation profile (connect-time only, no construction-time). The `get_frame_allocator` query relates to P4123's frame allocator timing discussion.

**Suggested change:** Acknowledge `function` as an alternative allocation profile for type-erased senders. Note Petersen's observation that operation state sizes repeat and lifetimes nest as evidence for the allocator patterns P4123 assumes are achievable.

### P4088 - What C++20 Coroutines Already Buy The Standard

P4088 Section 9 argues that `connect(any_sender, receiver)` stamps the receiver type into the operation state, requiring per-operation heap allocation under type erasure. P4223R0's `function` type-erases a sender factory instead, avoiding construction-time allocation.

**Suggested change:** Acknowledge `function` as the sender community's alternative design point in the allocation comparison sections. The connect-time allocation remains, but the construction-time allocation is eliminated.

### P4166 - Benefits of Frame-Visible Coroutines for Senders

P4166 argues frame-visible coroutines would eliminate `task`'s heap allocation entirely. P4223R0 takes a different approach: accept connect-time allocation but manage it via `get_frame_allocator`.

**Suggested change:** Reference `function` as evidence that the sender community acknowledges the allocation problem and is addressing it through allocator-management mechanisms. The two proposals are complementary: frame-visible coroutines eliminate the allocation; `get_frame_allocator` manages it when it exists.

### P2583 - Symmetric Transfer and Sender Composition

P4223R0's `function` is a type-erasing sender that must participate in the connect/start protocol. Its connect-time allocation is exactly where the symmetric transfer gap bites hardest.

**Suggested change:** Cite P4223R0 as a concrete example of a type-erased sender that validates the need for the protocol-level fix. If completion functions return `coroutine_handle<>`, `function`'s operation state must participate in that protocol.

---

## Structural/Landscape Changes

These papers describe a design space that P4223R0 has shifted.

### P4034 - On Universal Models

P4034's evidence table lists "Type erasure | libunifex #244 (5 years open) | Structural gap" as a domain where std::execution has no solution.

**Suggested change:** Acknowledge that P4223R0's `std::execution::function` is a proposed solution to this exact gap. The row should note the gap is being actively addressed.

### P4041 - Is `std::execution` a Universal Async Model?

P4041 Section 6 observes "Senders get the allocator they do not need. Coroutines need the frame allocator they do not get." P4223R0's `get_frame_allocator` partially addresses this.

**Suggested change:** Note that P4223R0 proposes a receiver-environment-based frame allocation query, partially closing the observation. The allocator now has a delivery mechanism in the sender world.

### P4126 - A Universal Continuation Model

P4126 proposes callback handles to give senders zero-allocation access to IoAwaitables. P4223R0's `function` addresses the same boundary from the sender side, accepting connect-time allocation managed via `get_frame_allocator`.

**Suggested change:** Reference `function` as the sender community's complementary approach. Compare allocation profiles: zero allocation (callback handles consuming awaitables) vs managed connect-time allocation (`function` returning type-erased senders).

### P4048 - Networking for C++29: A Call to Action

P4223R0 provides concrete interop mechanisms relevant to P4048's bridge requirements and compatibility team scope.

**Suggested change:** Cite P4223R0 as evidence that the std::execution side is actively developing bridge mechanisms (type-erased sender returns, frame allocator queries) that the Network Endeavor pipeline depends on.

---

## External Validation

These papers can cite P4223R0 as independent confirmation of their diagnoses.

### P4202 - Big Claims Require Big Evidence

P4223R0, by an external author (not C++ Alliance), explicitly credits Falco for "articulating shortcomings of senders/receivers when applied to network IO" and proposes remediation.

**Suggested change:** Cite P4223R0 in the case study timeline as a 2026 data point where an independent author validated the diagnosis and proposed remediation. This is external confirmation that the structural problems identified in the timeline were real.

### P4239 - Correction Capacity: The Networking Arc Under Two Rule Sets

P4223R0 represents organic correction from outside the P2300 authorship group: an external author proposing mechanisms to address the type-erasure and allocation problems that kept senders from serving I/O.

**Suggested change:** Update the "Open Loop" factual record in Section 6 (Decision 5). P4223R0 doesn't deliver sender-based networking (that claim remains true), but it represents evidence that correction mechanisms exist - relevant to the paper's thesis about whether correction occurs under each rule set.

---

## Not Informed

P4255 (synchronous senders) and 60 other papers in `_reserve/` have no interaction with P4223R0. The sender protocol's 7-step ceremony overhead (P4255's thesis) is unaffected by `function`, which still goes through connect/start. Committee-process, governance, contracts, profiles, int128, and other unrelated papers are orthogonal.
