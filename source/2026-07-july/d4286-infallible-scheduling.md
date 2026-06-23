---
title: "P3552: The Return of Networking TS Executors"
document: P4286R0
date: 2026-07-01
intent: info
audience: SG1, LEWG
reply-to:
  - "Vinnie Falco <vinnie.falco@gmail.com>"
---

## Abstract

The executor property rejected in 2021 returned, in 2026, as a requirement.

In 2021 the committee set aside the Networking TS. One stated deficiency was that its executors had no way to report a failure. In 2026 `std::execution::task` requires the schedulers that drive it to have no way to report a failure. The two constraints have the same shape. This paper traces the path from one to the other.

---

## Revision History

### R0: July 2026 (post-Brno mailing)

- Initial revision.

---

## 1. Disclosure

The author provides information and serves at the pleasure of the committee.

The author developed and maintains [Capy](https://github.com/cppalliance/capy)<sup>[1]</sup> and [Corosio](https://github.com/cppalliance/corosio)<sup>[2]</sup>, coroutine-native I/O libraries under the C++ Alliance. The author has a stake in the coroutine model's adoption.

This paper rests on three earlier papers in the same series ([P4094R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4094r1.html)<sup>[3]</sup>, [P4095R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4095r1.html)<sup>[4]</sup>, [P4096R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4096r1.html)<sup>[5]</sup>), which document the framing distinction the analysis below uses.

This paper asks for nothing.

## 2. The Diagnosis

In 2019, [P1525R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1525r0.pdf)<sup>[6]</sup> examined the one-way executor concept of the unified executors proposal<sup>[7]</sup>, whose basis operation accepted a callable and returned nothing:

```cpp
void execute(F&& f);
```

The first deficiency the paper named was error propagation. Errors arising during or after submission were handled, it observed, in an implementation-defined manner that varied from one executor to the next:

> "The implication is that no generic code can respond to asynchronous errors in a portable way."

In 2021, [P2464R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2021/p2464r0.html)<sup>[8]</sup>, written on behalf of the Finnish national body, applied the same standard to the Networking TS and reached three deficiencies. The first was the absence of an error channel. The committee polled on whether to keep pursuing the Networking TS. The poll reached no consensus, and the chair's guidance directed the committee's asynchronous work toward the sender/receiver model. A networking library with more than a decade of field deployment was set aside, and a missing error channel led the list of reasons.

## 3. The Framing

Why was the missing channel a deficiency? [P4094R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4094r1.html)<sup>[3]</sup> documents that `execute(F&&)` replaced three older primitives - `dispatch`, `post`, and `defer` - that scheduled a continuation rather than submitting work. Two readings of the same signature follow. Under the work framing, the callable is a unit of work, the caller is still running, and a missing error channel strands any failure with nowhere to go. Under the continuation framing, the callable is a resumption handle, the caller has suspended or returned, and there is no live caller waiting to receive an error.

[P4095R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4095r1.html)<sup>[4]</sup> and [P4096R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4096r1.html)<sup>[5]</sup> show that both P1525R0 and P2464R0 analyzed the operation under the work framing alone. The continuation framing - Kohlhoff's original definition in [P0113R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/p0113r0.html)<sup>[9]</sup> - had been erased from the API surface by the time either paper was written.

Under the continuation framing, an infallible scheduling operation is not a defect. It is the correct shape. A suspended coroutine does not need a channel to receive an error, because it is not running to act on one.

## 4. The Return

The sender/receiver model, [P2300R10](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html)<sup>[10]</sup>, shipped in C++26 as `std::execution`. [P3552R3](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3552r3.html)<sup>[11]</sup> added `task`, a coroutine type with one defining guarantee: scheduler affinity. After a `co_await`, a task resumes on the same scheduler it suspended on. `task` implements the guarantee by wrapping every awaited expression in `affine_on`, which schedules the continuation back onto the task's scheduler.

That scheduling operation must not fail. [P3941R4](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p3941r4.html)<sup>[12]</sup> establishes the requirement, in its own words:

> "If this scheduling operation fails, i.e., it completes with `set_error(e)`, or if it gets cancelled, i.e., it completes with `set_stopped()`, the execution agent on which the scheduling operation resumes is unclear and `affine_on` cannot guarantee its promise. Thus, it seems reasonable to require that a scheduler used with `affine_on` is infallible."

A scheduler used with `affine_on` may complete only with `set_value()`. No `set_error`. No `set_stopped`. The scheduling operation that drives `task` has no way to report a failure.

## 5. The Coroutine Executor

[P4003R3](https://isocpp.org/files/papers/P4003R3.pdf)<sup>[13]</sup> constrains the argument to `continuation` - a coroutine handle paired with an intrusive list pointer:

```cpp
std::coroutine_handle<> dispatch(
    continuation& c) const;

void post(continuation& c) const;
```

`dispatch` returns a handle for symmetric transfer. `post` defers. Both accept a suspended coroutine and resume it on a context. Neither delivers a value.

## 6. The Symmetry

| Property | P0443R14 executor | Infallible scheduler (P3941R4) | Coroutine executor (P4003R3) |
| --- | --- | --- | --- |
| Error channel | None | None - `set_error` omitted | None after acceptance. `post` throws before. |
| Cancellation | None | None - `set_stopped` omitted | `destroy()` on the handle |
| Value delivered | None | None - `set_value()` nullary | None |
| Encoded in the type by | `void` return | completion signatures | `continuation` argument type |
| Scope | every executor | schedulers used with `affine_on` | every coroutine executor |
| Why the caller tolerates it | it has returned | it is suspended | it is suspended |

The match across three columns is not exact. P0443R14 required infallibility of every executor and said so nowhere - the `void` return left no room to report a failure. P3941R4 requires infallibility of one scheduler in one role, and says so in the completion signatures. P4003R3 constrains the argument type to a coroutine handle, making infallibility a consequence of the continuation framing rather than a separate requirement. The old constraint was universal and implicit. The new ones are narrow and explicit.

**The committee rediscovered that continuations need a continuation-framed executor.**

## Acknowledgments

The author thanks Dietmar K&uuml;hl for P3941R4, which specifies the infallibility requirement and states its rationale precisely; Christopher Kohlhoff for the continuation framing in P0113R0 and for the Networking TS; Ville Voutilainen for P2464R0; Eric Niebler, Kirk Shoop, Lewis Baker, and Lee Howes for P1525R0 and the sender/receiver model that resolved the deficiencies they identified; and Steve Gerbino and Mungo Gill for co-developing the coroutine executor in P4003R3.

## References

[1] [Capy](https://github.com/cppalliance/capy) - Coroutine I/O primitives library (Vinnie Falco).

[2] [Corosio](https://github.com/cppalliance/corosio) - Coroutine-native networking library (Vinnie Falco).

[3] [P4094R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4094r1.html) - "The Unification of Executors and P0443" (Vinnie Falco, 2026).

[4] [P4095R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4095r1.html) - "The Basis Operation and P1525" (Vinnie Falco, 2026).

[5] [P4096R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4096r1.html) - "Coroutine Executors and P2464R0" (Vinnie Falco, 2026).

[6] [P1525R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1525r0.pdf) - "One-Way execute is a Poor Basis Operation" (Eric Niebler, Kirk Shoop, Lewis Baker, Lee Howes, 2019).

[7] [P0443R14](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p0443r14.html) - "A Unified Executors Proposal for C++" (Jared Hoberock, Michael Garland, Chris Kohlhoff, Chris Mysen, Carter Edwards, Gordon Brown, David Hollman, 2020).

[8] [P2464R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2021/p2464r0.html) - "Ruminations on networking and executors" (Ville Voutilainen, 2021).

[9] [P0113R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/p0113r0.html) - "Executors and Asynchronous Operations, Revision 2" (Christopher Kohlhoff, 2015).

[10] [P2300R10](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html) - "std::execution" (Micha&lstrok; Dominiak, Lewis Baker, Lee Howes, Kirk Shoop, Michael Garland, Eric Niebler, Bryce Adelstein Lelbach, 2024).

[11] [P3552R3](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3552r3.html) - "Add a Coroutine Task Type" (Dietmar K&uuml;hl, Maikel Nadolski, 2025).

[12] [P3941R4](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p3941r4.html) - "Scheduler Affinity" (Dietmar K&uuml;hl, 2026).

[13] [P4003R3](https://isocpp.org/files/papers/P4003R3.pdf) - "A Minimal Coroutine Execution Model" (Vinnie Falco, Steve Gerbino, Mungo Gill, 2026).
