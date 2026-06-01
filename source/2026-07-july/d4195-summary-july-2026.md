---
title: "A Reader's Guide to My July 2026 Papers"
document: D4195R0
date: 2026-07-31
intent: info
audience: WG21
reply-to:
  - "Vinnie Falco <vinnie.falco@gmail.com>"
---

## Abstract

Thirty-one papers deliver the complete Network Endeavor: thirteen ask-and-rationale pairs covering every layer from buffer descriptors through TLS, a coroutine task and executor utilities pair, an updated coroutine execution model, a GPU data-movement convergence study, and three governance papers placing WG21's procedures under structural and empirical scrutiny.

This paper summarizes 31 papers published in the July 2026 mailing. It is a reading guide: an executive summary that identifies the logical series within the collection, describes what each series delivers, and provides individual summaries of every paper. It asks for nothing.

---

## 1. Disclosure

The author provides information and serves at the pleasure of the committee.

This paper asks for nothing.

---

## 2. Executive Summary

Twenty-seven papers constitute the Network Endeavor series defined by P4100R1. At the foundation sits P4003R4<sup>[1]</sup>, the IoAwaitable coroutine execution model that provides executor affinity, stop-token propagation, and frame-allocator delivery through three operations in `await_suspend`. Above this protocol, thirteen ask-and-rationale pairs build the complete networking library in two stages. Stage One (Papers 2-7) delivers the portable coroutine and I/O vocabulary: `task<T>` with `run()` and `run_async()`, `strand<Ex>` and `any_executor`, buffer descriptors and sequence concepts, the `DynamicBuffer` growable buffer, `ReadStream`/`WriteStream`/`Stream` concepts with type-erasing wrappers, and `when_all`/`when_any` structured concurrency combinators. Stage Two (Papers 8-14) delivers platform I/O: timers, async signal handling, file I/O, TCP sockets with acceptors and IP addresses, DNS resolution, UDP datagrams, and TLS encryption. Every proposal comes from a shipping implementation in Capy and Corosio, validated across three platforms, and every design paper carries a multi-ecosystem convergence record. A companion study, P4251R0<sup>[30]</sup>, documents independent convergence on the IoAwaitable pattern for GPU data movement at NVIDIA Research, CERN, the University of Wisconsin-Madison, and Schrodinger.

Three papers stand outside the networking series. P4193R0<sup>[28]</sup> measures the interval between ISO publication and compiler conformance across five standards and finds that each successive standard takes longer, with C++20 still incomplete on two of three major compilers after sixty-four months. P4196R0<sup>[29]</sup> places SD-4 and the ISO/IEC Directives side by side on twelve points and documents deviations on every one, noting that no other JTC 1 working group maintains a comparable procedural supplement. P4238R0<sup>[31]</sup> traces the trajectory of P2900's "ignore" semantic from 2021 through the NB ballot, applying ISO Directive requirements at each decision point and finding twenty-one polls with zero documented reconciliation processes.

Two entry points serve different reader profiles. A library author or implementer evaluating what ships in C++29 should begin with P4003R4<sup>[1]</sup>, which defines the execution model the entire series builds on, then scan the Stage Two ask papers (Timers through TLS) to see the library surface. A delegate concerned with governance should begin with P4196R0<sup>[29]</sup>, which provides the procedural comparison table, then read P4238R0<sup>[31]</sup> for the case study.

---

## 3. Individual Papers

### 3.1. P4003R4 - A Minimal Coroutine Execution Model

The IoAwaitable protocol provides exactly three things a coroutine needs: executor affinity, stop-token propagation, and frame-allocator delivery. P4003R4 asks the committee to advance this protocol as a standard coroutine execution model. The paper starts from `co_await f()` and shows what the language requires to make that expression work across library boundaries. Everything comes from a complete implementation on three platforms in Capy and Corosio. The companion design paper P4172R0 carries the rationale, evidence framework, and analysis of alternatives. This is Paper 1 in the Network Endeavor series and the foundation every subsequent paper depends on.

### 3.2. D0003R0 / D0004R0 - Coroutine Task

D0003R0 asks the committee to advance `task<T>` - a lazy coroutine task with one template parameter - as standard library vocabulary, together with `run()`, `run_async()`, a `thread_pool`, and a `system_context`. The type has one template parameter because `coroutine_handle<>` erases the coroutine's type structurally. The IoAwaitable protocol propagates context through `await_suspend`, so the task type stays simple. D0004R0 carries the design rationale, the five-ecosystem convergence record, and the implementation inventory. Paper 2 in the series.

### 3.3. D0005R0 / D0006R0 - Executor Utilities

D0005R0 asks the committee to advance `strand<Ex>` and `any_executor`. `strand<Ex>` serializes coroutine resumption over any executor without locks. `any_executor` type-erases any `Executor` behind an owning handle. Neither has an I/O dependency or a platform dependency. Both ship in Capy. D0006R0 carries the design rationale and convergence record. Paper 3 in the series.

### 3.4. D0007R0 / D0008R0 - I/O Buffer Ranges

D0007R0 asks the committee to advance a buffer descriptor vocabulary: `const_buffer`, `mutable_buffer`, `ConstBufferSequence`, `MutableBufferSequence`, and four algorithms. The shape is the Networking TS shape, deployed for over twenty years in Boost.Asio. D0008R0 documents the seven-ecosystem convergence record, the relationship to `std::ranges`, and four anticipated objections. Paper 4 in the series; no async dependency.

### 3.5. D0009R0 / D0010R0 - Dynamic Buffer

D0009R0 asks the committee to advance the `DynamicBuffer` concept - a growable byte buffer with two-phase write (`prepare`/`commit`) and two-phase read (`data`/`consume`). Four implementations ship in Capy: flat, circular, vector-backed, and string-backed. D0010R0 carries the two-phase model rationale, the four-implementation tour, the three-ecosystem convergence record (Asio, .NET, Go), and the deferrals. Paper 5 in the series.

### 3.6. D0011R0 / D0012R0 - Stream Concepts

D0011R0 asks the committee to advance coroutine stream concepts - `ReadStream`, `WriteStream`, `Stream`, plus source/sink refinements and seven `any_*` type-erasing wrappers - as standard I/O vocabulary. The concepts are the coroutine-native successors to the Networking TS stream requirements. Boost.Http compiles once against `any_stream` for ABI-stable HTTP processing. D0012R0 documents the six-ecosystem convergence record and the design rationale. Paper 6 in the series.

### 3.7. D0013R0 / D0014R0 - Combinators

D0013R0 asks the committee to advance `when_all` and `when_any` as structured concurrency combinators for coroutine-native I/O. `when_all` awaits every child and returns when all complete. `when_any` awaits the first to complete and cancels the rest. Both propagate errors and integrate with `std::stop_token`. D0014R0 documents the five-ecosystem convergence and the design constraints that forced the shapes. Paper 7 in the series; completes Stage One.

### 3.8. D0015R0 / D0016R0 - Timers

D0015R0 asks the committee to advance `std::io::timer` - an awaitable deadline enforced by the kernel. One type, five member functions, one clock. The timer is the simplest kernel interaction in the series and proves the IoAwaitable protocol works end-to-end with a real operating system. D0016R0 documents the platform mapping to three OS primitives and the six-ecosystem convergence. Paper 8 in the series; first Stage Two paper.

### 3.9. D0017R0 / D0018R0 - Signals

D0017R0 asks the committee to advance `signal_set` for async signal handling. `<csignal>` is from 1989; its `signal()` function runs handlers in signal context where almost nothing is safe to call. Modern servers need to wait for `SIGINT` or `SIGTERM` the same way they wait for a timer or a socket: with `co_await`. D0018R0 carries the design rationale and six-ecosystem convergence. Paper 9 in the series.

### 3.10. D0019R0 / D0020R0 - Files

D0019R0 asks the committee to advance `stream_file` and `random_access_file` for async file I/O. Both satisfy the same stream concepts as TCP sockets: same buffer sequences, same error model, same cancellation. On Windows, IOCP with `FILE_FLAG_OVERLAPPED`; on Linux, POSIX file I/O with `io_uring` on the roadmap. D0020R0 carries the design rationale. Paper 10 in the series.

### 3.11. D0021R0 / D0022R0 - TCP

D0021R0 asks the committee to advance `tcp_socket`, `tcp_acceptor`, `endpoint`, `ipv4_address`, `ipv6_address`, and `ip_address` as standard networking vocabulary. A `tcp_socket` satisfies `Stream`. This delivers what the committee has been trying to standardize since N1925 (2005). D0022R0 documents the twenty-year deployment history and the convergence record. Paper 11 in the series.

### 3.12. D0023R0 / D0024R0 - DNS

D0023R0 asks the committee to advance `std::io::resolver` for async name resolution. Without DNS, TCP sockets can only connect to hardcoded IP addresses. One type, two overloads, one results range. D0024R0 documents why none of the existing C/POSIX interfaces are async or portable at the interface level. Paper 12 in the series.

### 3.13. D0025R0 / D0026R0 - UDP

D0025R0 asks the committee to advance `std::io::udp_socket` for datagram I/O. UDP is the transport beneath DNS, QUIC/HTTP/3, game networking, media streaming, and IoT protocols. The type provides `send_to`, `receive_from`, `bind`, multicast group management, and an optional connected mode. D0026R0 carries the design rationale. Paper 13 in the series.

### 3.14. D0027R0 / D0028R0 - TLS

D0027R0 asks the committee to advance `tls_context` for portable certificate management and `tls_stream` for encrypted I/O. Every other major language has this; C++ does not. A `tls_stream` wraps any `Stream` and satisfies `Stream` itself. The cryptographic engine is implementation-defined - the same model as `std::filesystem`. A TLS vulnerability disclosed on Monday is patchable by Tuesday without waiting for a standard library release cycle. D0028R0 documents the seven-ecosystem convergence and thirty years of stable wrapper design. Paper 14 in the series; final paper.

### 3.15. P4251R0 - IoAwaitables for GPU Data Movement: Convergent Findings

A protocol handler compiled once links against TCP, RDMA, or GPU device memory without recompilation. P4251R0 documents how independent projects at NVIDIA Research, CERN, the University of Wisconsin-Madison, and Schrodinger converged on the same coroutine-based async completion pattern for GPU data movement that the IoAwaitable protocol captures. Bidirectional bridges connect IoAwaitables and senders where byte-oriented I/O meets GPU dispatch, allowing each model to serve its natural domain.

### 3.16. P4193R0 - Each Standard Takes Longer

Each successive C++ standard takes longer for compilers to implement, and C++20 - published over five years ago - remains incomplete on two of three major compilers. P4193R0 measures the interval between ISO publication and compiler conformance across five standards, three compilers, and fifteen years. C++14 achieved near-complete conformance within months. C++20 has not achieved it after sixty-four. The paper asks what changed and what institutional forces produce the pattern.

### 3.17. P4196R0 - Discrepancies Between SD-4 and the ISO Directives

The procedural document that governs WG21's daily operation - SD-4 - deviates from the binding ISO/IEC Directives on every one of the twelve points this paper examines, and does not appear in any WG21 document list. P4196R0 places SD-4 side by side with the ISO/IEC Directives Part 1 on subgroup chair appointment, consensus thresholds, ballot comment scope, escalation procedures, priority allocation, and meeting record transparency, documenting that WG21's four main subgroups satisfy every functional criterion of an ISO working group while operating outside every governance requirement the Directives impose on one. The paper finds that no other JTC 1 working group maintains a comparable procedural supplement - the only body that grew to similar scale, MPEG, was formally restructured into proper working groups rather than allowed to write its own governance document. The analysis names the two Directive mechanisms available to resolve the gap: formalization under Directive 1.4 and National Body objection under Directive 5.1.2.

### 3.18. P4238R0 - Reconciliation Capacity: The Contracts Arc Under Two Rule Sets

Twenty-one polls, zero documented reconciliation processes. P4238R0 traces the trajectory of P2900's "ignore" semantic from its establishment in 2021 through the NB ballot in 2025. For each decision point, the paper asks what the ISO Directives required that SD-4 did not. Under SD-4, "no consensus for change" preserves the status quo; under the Directives, unresolved objections require reconciliation at each stage. The contracts arc demonstrates what happens when a contentious design choice survives through repeated status-quo preservation rather than documented engagement: minority concerns accumulate without resolution, surfacing as a 19-NB revolt at the ballot stage. The answer is not better decisions. It is documented engagement - the structural capacity to reconcile rather than override.

---

## 4. Conclusion

This reading guide covers 31 papers from the July 2026 mailing. The author hopes it helps the reader find the papers most relevant to their work and interests.

---

## References

[1] [P4003R4](https://wg21.link/p4003r4) - "A Minimal Coroutine Execution Model" (Vinnie Falco, Steve Gerbino, Mungo Gill, 2026).

[2] D0003R0 - "Coroutine Task" (Vinnie Falco, 2026).

[3] D0004R0 - "Coroutine Task: Design Rationale" (Vinnie Falco, 2026).

[4] D0005R0 - "Executor Utilities" (Vinnie Falco, 2026).

[5] D0006R0 - "Executor Utilities: Design Rationale" (Vinnie Falco, 2026).

[6] D0007R0 - "I/O Buffer Ranges" (Vinnie Falco, 2026).

[7] D0008R0 - "I/O Buffer Ranges: Design Rationale" (Vinnie Falco, 2026).

[8] D0009R0 - "Dynamic Buffer" (Vinnie Falco, 2026).

[9] D0010R0 - "Dynamic Buffer: Design Rationale" (Vinnie Falco, 2026).

[10] D0011R0 - "Stream Concepts" (Vinnie Falco, 2026).

[11] D0012R0 - "Stream Concepts: Design Rationale" (Vinnie Falco, 2026).

[12] D0013R0 - "Combinators" (Vinnie Falco, 2026).

[13] D0014R0 - "Combinators: Design Rationale" (Vinnie Falco, 2026).

[14] D0015R0 - "Timers" (Vinnie Falco, 2026).

[15] D0016R0 - "Timers: Design Rationale" (Vinnie Falco, 2026).

[16] D0017R0 - "Signals" (Vinnie Falco, 2026).

[17] D0018R0 - "Signals: Design Rationale" (Vinnie Falco, 2026).

[18] D0019R0 - "Files" (Vinnie Falco, 2026).

[19] D0020R0 - "Files: Design Rationale" (Vinnie Falco, 2026).

[20] D0021R0 - "TCP" (Vinnie Falco, 2026).

[21] D0022R0 - "TCP: Design Rationale" (Vinnie Falco, 2026).

[22] D0023R0 - "DNS" (Vinnie Falco, 2026).

[23] D0024R0 - "DNS: Design Rationale" (Vinnie Falco, 2026).

[24] D0025R0 - "UDP" (Vinnie Falco, 2026).

[25] D0026R0 - "UDP: Design Rationale" (Vinnie Falco, 2026).

[26] D0027R0 - "TLS" (Vinnie Falco, 2026).

[27] D0028R0 - "TLS: Design Rationale" (Vinnie Falco, 2026).

[28] [P4193R0](https://wg21.link/p4193r0) - "Each Standard Takes Longer" (Vinnie Falco, 2026).

[29] [P4196R0](https://wg21.link/p4196r0) - "Discrepancies Between SD-4 and the ISO Directives" (Vinnie Falco, 2026).

[30] [P4251R0](https://wg21.link/p4251r0) - "IoAwaitables for GPU Data Movement: Convergent Findings" (Vinnie Falco, 2026).

[31] [P4238R0](https://wg21.link/p4238r0) - "Reconciliation Capacity: The Contracts Arc Under Two Rule Sets" (Vinnie Falco, 2026).
