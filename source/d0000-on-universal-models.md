::: {.document-info}
| Document | D0000 |
|----------|-------|
| Date:       | 2026-02-09
| Reply-to:   | Vinnie Falco \<vinnie.falco@gmail.com\>
| Audience:   | All of WG21
::: 

# On Universal Models

## Abstract

Software engineering has a recurring pattern: smart people see commonality across domains and conclude that one abstraction should serve them all. Sometimes they are right. TCP/IP and IEEE 754 are genuine universal models that emerged from practice and proved themselves across decades of deployment. More often they are wrong, and the universal framework loses to pragmatic, specialized alternatives. This paper examines the evidence for and against a universal execution model in C++, focusing on `std::execution` ([P2300](https://wg21.link/p2300)). It proposes no wording changes. It asks the committee to consider whether the evidence supports the current direction, or whether specialization with interoperation might serve the C++ community better.

---

## 1. Introduction

The desire for a universal model is one of the most natural instincts in software design. A programmer looks at callbacks, futures, coroutines, and sender/receiver pipelines and thinks: these are all doing the same thing. Surely one abstraction can unify them.

That instinct is often productive. But it has a well-documented failure mode.

The MIT Exokernel paper put it in formal terms:

> "It is fundamentally impossible to define abstractions that are appropriate for all areas and implement them efficiently in all situations."
>
> [Engler et al., "Exokernel: An Operating System Architecture for Application-Level Resource Management"](https://people.eecs.berkeley.edu/~brewer/cs262b/hotos-exokernel.pdf) (1995)

Butler Lampson, in his Turing Award lecture on systems design, put the corrective plainly:

> "An interface should capture the minimum essentials of an abstraction. Don't generalize; generalizations are generally wrong."
>
> [Butler Lampson, "Hints for Computer System Design"](http://research.microsoft.com/en-us/um/people/blampson/33-Hints/Acrobat.pdf) (1983)

And Ted Kaminski captured the tradeoff precisely:

> "An all-powerful abstraction is a meaningless one. You've just got a new word for 'thing'."
>
> [Ted Kaminski, "The One Ring Problem"](https://tedinski.com/2018/01/30/the-one-ring-problem-abstraction-and-power.html) (2018)

None of this means universal models are impossible. They exist and they matter. But the history of computing suggests they are rare, and that the instinct to create them runs far ahead of the evidence needed to justify them.

This paper examines `std::execution` through that lens. The people who designed it are talented. The work they have done contains genuine insights. The question is whether the evidence supports declaring it the universal execution model for C++, or whether the direction has gotten ahead of itself in a domain that is genuinely difficult.

The asymmetry of risk makes this question worth asking. If `std::execution` is truly universal, it will prove itself through adoption and this paper will have been an unnecessary caution. If it is not universal and we mandate it anyway, the cost compounds for decades. Every major programming language ships standard networking. C++ does not. Getting the execution model wrong makes that problem harder, not easier.

---

## 2. The Historical Record

### 2.1 CORBA vs. REST

CORBA (Common Object Request Broker Architecture) is the closest historical parallel to `std::execution`. It was designed by a standards consortium (the OMG) as a universal middleware framework for distributed computing. It bundled object lifecycle, naming, transactions, security, concurrency, and interface definition into a single specification. It had institutional backing, vendor support, and a decade of investment. It failed.

Michi Henning, who spent years inside the CORBA effort, wrote the post-mortem:

> "Developers who gained experience with CORBA found that writing any nontrivial CORBA application was surprisingly difficult. Many of the APIs were complex, inconsistent, and downright arcane, forcing the developer to take care of a lot of detail."
>
> [Michi Henning, "The Rise and Fall of CORBA"](https://dl.acm.org/doi/10.1145/1142031.1142044), ACM Queue (2006)

The CORBA Component Model (CCM) was supposed to fix the usability problem. It did not:

> "The specification was large and complex and much of it had never been implemented, not even as a proof of concept. Reading the document made it clear that CCM was technically immature; sections of it were essentially unimplementable."
>
> Henning (2006)

David Chappell, writing in 1998 while CORBA was still considered viable, identified the deeper issue:

> "Is it even possible for committees to successfully create new technology? History is not encouraging. Confusing the production of paper with the production of products is perhaps the classic error for a standardization body."
>
> [David Chappell, "The Trouble With CORBA"](https://davidchappell.com/writing/article_Trouble_CORBA.php) (1998)

REST and HTTP won because they were narrow: resources, verbs, and status codes over a single protocol. CORBA did not fail before deployment. It failed *after* deployment, when the weight of the specification became apparent in practice. Chappell saw where it was heading:

> "The opportunity for a true standard, a TCP/IP for distributed objects, has been lost."
>
> Chappell (1998)

The structural parallels to `std::execution` are difficult to ignore:

| | CORBA | `std::execution` |
|---|---|---|
| Designed by | Standards consortium (OMG) | Standards committee (WG21) |
| Scope | Universal distributed object middleware | Universal async execution model |
| Bundled concerns | Object lifecycle, naming, transactions, security, concurrency, IDL | Scheduling, context propagation, error handling, cancellation, algorithm dispatch, hardware backends |
| Companion specs in flight | CCM, security service, transaction service | P2079, P3164, P3373, P3388, P3425, P3481, P3552, P3557, P3564, P3826 |
| Competing narrow alternatives | REST/HTTP, EJB, then microservices | Asio, folly, Seastar, TooManyCooks, Taskflow |
| Primary use case deferred | Web integration (arrived too late) | Networking (deferred to C++29) |
| Institutional backing | ISO, telecom vendors, government contracts | WG21, NVIDIA, national body votes |

The pattern has older precedent. The OSI model had government procurement mandates ([GOSIP](https://en.wikipedia.org/wiki/Government_Open_Systems_Interconnection_Profile), 1988), ISO backing, and major corporate support. TCP/IP won anyway because it was simpler to deploy ([IEEE Spectrum, "OSI: The Internet That Wasn't"](https://spectrum.ieee.org/osi-the-internet-that-wasnt)). Institutional momentum could not overcome the weight of the abstraction.

CORBA's failure was not inevitable. A narrower middleware - one that solved object communication without trying to standardize lifecycle, naming, transactions, and security in the same specification - might have succeeded. The lesson is not that distributed computing is impossible to standardize. It is that the scope of the standardization attempt determines its fate.

### 2.2 "Everything Is an Object"

Object-oriented programming went through a similar cycle. The claim that "everything is an object" positioned OOP as a universal modeling philosophy. Richard Gabriel argued at OOPSLA 2002 that this claim gave OOP a privileged position that starved research into alternative paradigms ([Gabriel, "Objects Have Failed"](https://dreamsongs.com/Files/ObjectsHaveFailed.pdf)).

The Gang of Four's *Design Patterns* (1994) articulated the corrective: "Favor object composition over class inheritance." The industry eventually learned that deep inheritance hierarchies were brittle and that composition through narrow interfaces produced better designs ([Wikipedia: Composition over inheritance](https://en.wikipedia.org/wiki/Composition_over_inheritance)). The structural parallel to async is direct: a single wide execution model that subsumes scheduling, error handling, cancellation, and hardware dispatch is the async equivalent of a single deep inheritance hierarchy. The corrective is the same - narrow interfaces that compose.

### 2.3 The Pattern

Universal models designed top-down consistently lose to specialized models that emerge from practice. This is not a new observation. The question is whether C++ async is repeating it.

---

## 3. Universal Models in C++

### 3.1 The "Grand Unified Model" Vote

On 2021-09-28, SG1 polled:

> "We believe we need one grand unified model for asynchronous execution in the C++ Standard Library, that covers structured concurrency, event based programming, active patterns, etc."

The result was **no consensus**: 4 SF, 9 WF, 5 N, 5 WA, 1 SA. [P2453R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2453r0.html) documents this poll and its interpretation. The committee did not achieve consensus that a universal model was even needed. Yet the direction proceeded as though it had.

This is worth pausing on. The foundational premise of the current direction did not achieve consensus among the people voting on it.

### 3.2 Ranges

Ranges are the most recent precedent for a universal abstraction in C++. They were positioned as the universal iteration and algorithm model. Adoption tells a more nuanced story.

Google bans `<ranges>` from most of its codebase. Daisy Hollman (Google) explained at [CppCon 2024](https://cppcon2024.sched.com/event/1gZgc/why-google-doesnt-allow-ranges-in-our-codebase) that ranges are "perhaps the largest and most ambitious single feature ever added to the C++ standard library" but "it's unreasonable to expect that those trade-offs will result in the same cost-benefit ratio in every context."

Compile-time overhead is substantial. Range-V3, the precursor library that informed the standard design, compiles its headers in 3.44 seconds versus 0.44 seconds for STL `<algorithm>`, roughly an 8x slowdown ([NanoRange wiki: Compile times](https://github.com/tcbrindle/NanoRange/wiki/Compile-times)). Deeply nested range adapters exhibit a "cubic stack blowup" in template instantiation ([Hacker News discussion](https://news.ycombinator.com/item?id=40317350)). Daniel Lemire's analysis suggests that ["std::ranges may not deliver the performance that you expect"](https://lemire.me/blog/2025/10/05/stdranges-may-not-deliver-the-performance-that-you-expect/) (2025).

Ranges brought real value to the language. The parallel to `std::execution` is imperfect - ranges shipped, they work, and many codebases use them productively. But the difference in *kind* of cost matters more than the difference in degree. With ranges, Google can ban the header; the cost is compile time and ergonomic complexity, real but recoverable. With `std::execution`, the cost is architectural: ABI lock-in on an execution model that has not proven itself for I/O, the domain that started the executor discussion a decade ago. WG21's insistence on ABI stability means that each feature must land right, because mistakes cannot be corrected without breaking the world. Every wide abstraction that lands imperfectly adds permanent technical debt to the standard - and the rate of that accumulation increases with the scope of what is frozen. Ranges show that even successful universal abstractions can prove too costly for major adopters. The question is whether C++ can afford to discover that *after* the ABI is frozen, rather than before.

A narrower ranges - one with the discernment to recognize that reaching for the last few percent of unlikely use cases would add disproportionate complexity - would have avoided these costs entirely. The instinct to make an abstraction cover every case is what turns a useful tool into an expensive one. `std::execution` makes the same reach, at higher stakes.

But the ranges experience also points toward a solution. The abstractions that succeed in C++ are narrow: iterators, RAII, allocators. A narrow async abstraction - one that captures the essential I/O operation and leaves everything else to the user - would follow the pattern that works.

### 3.3 IETF TAPS

The IETF Transport Services initiative ([RFC 9621](https://datatracker.ietf.org/doc/html/rfc9621), [RFC 9622](https://datatracker.ietf.org/doc/html/rfc9622), January 2025) proposes replacing the traditional socket API with a property-based model. Applications declare requirements - reliability, multistreaming, congestion control, multipath support, checksums, keep-alive, security protocols, certificate policies, preference levels - and the system selects the transport. [P3482](https://wg21.link/p3482) (Rodgers & Kühl) proposes basing C++ networking on this model. The result is an abstraction with an opinion about everything, in a domain where the abstractions that survive have opinions about almost nothing.

The POSIX socket interface - `socket`, `connect`, `read`, `write`, `close`, `shutdown` - has endured for four decades not despite its age but because of it. An interface that has survived the transition from 10 Mbps Ethernet to 400 Gbps, from single-threaded servers to million-connection event loops, from plaintext to ubiquitous TLS, and remained unchanged, is an interface whose abstraction is correct. As Ousterhout observes: "Implementations of the Unix I/O interface have evolved radically over the years, but the five basic kernel calls have not changed" ([*A Philosophy of Software Design*](https://web.stanford.edu/~ouster/cgi-bin/aposd.php)). What C++ needs is the async equivalent of what POSIX already provides - read bytes, write bytes, connect, close - not a new transport selection framework.

To make this concrete, we looked up the TAPS equivalent of a POSIX connect-and-send. In POSIX, the entire operation is three function calls:

```c
int fd = socket(AF_INET, SOCK_STREAM, 0);
connect(fd, &addr, sizeof(addr));
write(fd, buf, len);
```

We expected the TAPS version to be somewhat larger. It was so much larger that a side-by-side comparison is physically impossible - the TAPS client example from [RFC 9622](https://datatracker.ietf.org/doc/html/rfc9622) Section 3.1.2 occupies its own page and is reproduced in full in Appendix A. The specification that surrounds it runs to over 150 pages across two Standards Track RFCs. Instead, the following table places the POSIX code next to a summary of what TAPS requires for the same operation:

| POSIX                                      | TAPS (RFC 9622) Summary                                          |
|--------------------------------------------|------------------------------------------------------------------|
| `socket(AF_INET, SOCK_STREAM, 0);`         | Create `RemoteEndpoint`, configure hostname and service           |
| `connect(fd, &addr, sizeof(addr));`        | Create `TransportProperties`, set 18 Selection Properties (each with a 5-valued Preference Enumeration: Prohibit, Avoid, No Preference, Prefer, Require) |
|                                            | Create `SecurityParameters`, configure 8 security parameter categories (certificates, ALPN, ciphersuites, session cache, PSK, callbacks) |
|                                            | Create `Preconnection` from endpoints, properties, and security   |
|                                            | Call `Initiate()`, which triggers DNS resolution, candidate gathering, and protocol racing |
|                                            | Handle asynchronous `Ready` event                                 |
| `write(fd, buf, len);`                     | Create `MessageContext`, configure 8 per-message properties (lifetime, priority, ordering, reliability, checksum coverage, capacity profile, fragmentation, segmentation) |
|                                            | Call `Connection.Send(messageData, messageContext)`                |
|                                            | Handle asynchronous `Sent`, `Expired`, or `SendError` events      |

Three function calls versus four object types, 18 selection properties, 8 security parameter categories, 8 message properties, and a mandatory asynchronous event-driven interaction pattern. RFC 9622 itself states that implementations "SHOULD implement each Selection Property, Connection Property, and MessageContext Property specified in this document" and adds: "These features SHOULD be implemented even when, in a specific implementation, it will always result in no operation." The specification requires implementations to carry the weight of the full abstraction even when the application needs none of it.

One proprietary implementation of TAPS exists: Apple's Network.framework (2018). [P3482R1](https://wg21.link/p3482) acknowledges: "Unfortunately, at present, Apple's Network Framework is the only such implementation." The one open-source attempt, NEAT, was [abandoned when EU funding ended](https://www.neat-project.org/). Standardizing C++ networking on an API with no portable implementation experience would repeat the top-down pattern this paper examines. The proven alternative is simpler: take the narrow waist that has worked for forty years and make it async.

### 3.4 The Networking TS Schism

The Networking TS was really two things bundled together: an execution model (io_context, completion handlers, executors) and portable wrappers for I/O objects (sockets, timers, etc.). The controversy was never about the sockets. It was entirely about the execution model.

The polls in [P2453R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2453r0.html) make this clear:

- **Poll 1** asked whether the Networking TS/Asio async model is "a good basis for most asynchronous use cases, including networking, parallelism, and GPUs." Result: weak consensus against. But the interpretation noted: "It doesn't mean that the Networking TS async model isn't a good fit for networking. There were many comments to the contrary."

- **Poll 3** asked to "Stop pursuing the Networking TS/Asio design as the C++ Standard Library's answer for networking." Result: **no consensus** (13 SF, 13 WF, 8 N, 6 WA, 10 SA). The Networking TS is not dead.

- **Poll 5** asked whether it is "acceptable to ship socket-based networking... that does not support secure sockets." Result: **no consensus**. No objection to the sockets themselves, only to shipping without TLS.

Nobody objected to portable socket wrappers. The schism was entirely about which execution model async operations should use. The I/O objects were uncontroversial because they did not claim universality.

### 3.5 std::execution (P2300)

`std::execution` is the central case study. It proposes sender/receiver as the universal async model for C++. The question is whether the evidence supports declaring it universal.

#### 3.5.1 Design Priorities

`std::execution` has been visibly proven to excel for two specific domains: heterogeneous computing (GPU dispatch, data-parallel execution, hardware backend selection) and latency-critical compute (compile-time work graph construction, zero-allocation pipelines, deterministic execution). These are worthy domains, and the design serves them well. The evidence is in the specification itself:

- [P2300R10](https://open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html) §1.1 frames the motivation around "GPUs in the world's fastest supercomputer."
- P2300R10 §1.2 prioritizes "the diversity of execution resources and execution agents, because not all execution agents are created equal."
- The second end-user example (§1.3.2) is "Asynchronous inclusive scan," a classic GPU parallel primitive using `bulk` to spawn data-parallel execution agents. This is not an I/O pattern.
- `bulk` (§4.20.9) spawns N execution agents for data-parallel work. No networking analog exists.
- `continues_on` / transfer moves work between execution contexts, a CPU-to-GPU pattern. Networking does not transfer between hardware backends.
- Completion domains dispatch algorithms based on execution resource, so GPU backends can substitute custom implementations. TCP reads have one implementation per platform, not multiple hardware backends.

The entire sender algorithm customization lineage ([P2999R3](https://wg21.link/p2999r3), [P3303R1](https://open-std.org/jtc1/sc22/wg21/docs/papers/2024/p3303r1.html), [P3826](https://wg21.link/p3826)) is about domain-based algorithm dispatch. These papers contain zero mentions of networking, sockets, or I/O. Robert Leahy of Hudson River Trading, a core contributor who authored [P3373](https://wg21.link/p3373) and [P3388](https://wg21.link/p3388), [presented at CppCon 2025](https://cppcon2025.sched.com/event/27bQ1/stdexecution-in-asio-codebases-adopting-senders-without-a-rewrite) on adopting sender/receiver in financial infrastructure - precisely the domain where compile-time work graph construction and zero-overhead pipelines matter most.

But the properties that serve these domains are precisely what makes `std::execution` unsuitable as a universal model. The compile-time visibility that HFT values means everything is templates - separate compilation is impossible. The zero-allocation pipeline that embedded systems need means type erasure is structurally resisted. The very properties that serve GPU dispatch and latency-critical compute *underserve* the ordinary developer who needs fast compile times, stable ABIs, and type-erased interfaces behind shared libraries.

That developer - the one writing a web service, a database client, a monitoring agent, a chat server - needs to open a socket, read data, handle errors, and compile in reasonable time. They need `pimpl`, shared libraries, and `any_completion_handler`. The framework does not serve them. What they want to write is:

```cpp
auto [ec, n] = co_await stream.read(buf);
```

What `std::execution` requires them to write is:

```cpp
auto s = just(buf)
       | let_value([&](auto b) {
           return async_read(stream, b);
         })
       | then([](std::size_t n) { /* ... */ });
sync_wait(on(system_ctx.get_scheduler(), std::move(s)));
```

[P3552](https://wg21.link/p3552)'s `task` type narrows the ergonomic gap - a coroutine using `task` and `co_await` on a sender-based operation would look less verbose. But the underlying model differences persist regardless of surface syntax: the architectural costs remain. [P2079](https://wg21.link/p2079) (`system_context`) provides a default scheduler - a place to run the second snippet. It does not let you write the first one. The problem is not the absence of a scheduler. It is the programming model itself: the template machinery underneath, which cannot be hidden behind `pimpl` or compiled separately.

The design reflects its origins in GPU and HFT infrastructure. Of the companion papers in flight (see §3.5.5), zero address networking. The specification examples are parallel primitives, not I/O operations. The algorithm customization machinery serves hardware backend dispatch, not socket reads. A framework optimized for these domains will naturally optimize for these domains, and the claim of universality requires evidence from outside them.

The ordinary developer's needs are simpler. A design that serves them is correspondingly smaller. And smaller is achievable.

#### 3.5.1a Structured Concurrency

A natural response is that `std::execution` provides structured concurrency guarantees - structured lifetimes for async operations that prevent use-after-free - and that these guarantees justify the framework's scope.

Structured concurrency is a genuine contribution. The question is whether it is uniquely achievable through sender/receiver, or whether multiple execution models can provide it, each optimized for different domains.

Sender/receiver achieves structured lifetimes through operation states with explicit connect/start phases and compile-time work graph construction. This serves GPU dispatch and latency-critical compute well, where the entire execution graph is known at compile time and zero-allocation pipelines matter. Coroutines achieve structured lifetimes through a different mechanism: the coroutine frame lives as long as its handle, `co_await` suspends without detaching, and RAII destructors run at scope exit. Coroutine-based primitives - `when_all`, `when_any`, `async_scope`, `async_mutex`, channels - are implementable on top of coroutines without sender/receiver ([D4003](https://wg21.link/p4003) §4.1-4.2). This serves I/O-oriented code well, where the execution graph is dynamic and type erasure matters.

Each approach optimizes for its domain. The sender/receiver model pays for compile-time visibility with template weight and resistance to erasure. The coroutine model pays for erasure and fast compilation with runtime frame allocation. These are legitimate engineering tradeoffs, and different users will make different choices.

A natural concern is that coroutines lack portable heap allocation elision (HALO), making them unsuitable as a standardization foundation. This concern is less impactful for async I/O than it first appears. At the end of an I/O call chain, the coroutine escapes the caller: it suspends and its handle is passed to the OS reactor (epoll, IOCP, io_uring). HALO can never apply at the I/O boundary because the coroutine inherently outlives the call that created it. HALO matters for short-lived compute coroutines that return inline to the caller. I/O coroutines do not. The domain where HALO matters most - tight compute loops - is the domain sender/receiver already serves well. The domain where coroutines serve I/O - long-lived suspended operations waiting on the OS - is the domain where HALO is structurally inapplicable.

Both approaches have open problems. `std::execution`'s own structured concurrency record is mixed: [P3373](https://wg21.link/p3373) reworks operation-state lifetimes, [LWG 4368](https://cplusplus.github.io/LWG/issue4368) is a Priority 1 dangling-reference vulnerability in `transform_sender`, and `ensure_started`, `start_detached`, and `split` were removed precisely because they broke structured concurrency ([P3187R1](https://wg21.link/p3187r1)). Coroutine-based models face their own challenges in allocator propagation, which [D4007](https://wg21.link/p4007) and [D4003](https://wg21.link/p4003) address. Neither approach has a monopoly on safety.

Structured concurrency is a property worth having, not an argument for a single framework. The question before the committee is not which mechanism achieves it best in the abstract, but whether mandating one forecloses the others - and whether that foreclosure is justified by the evidence.

A related objection is that sender pipelines compose at the type level - `then | when_all | continues_on` - in ways that coroutines do not. This is true. Coroutines compose at the expression level: `co_await when_all(a(), b())`. The two styles optimize for different things. Type-level composition gives the compiler full visibility into the work graph at compile time, which serves GPU dispatch and latency-critical compute. Expression-level composition gives the programmer fast compilation and ABI stability, which serves I/O. But the claim that one composition style must subsume the other is itself the universality instinct this paper examines. Different domains compose differently. Demanding that every async abstraction compose identically is the same error as demanding that one execution model serve all domains. The question is not whether coroutines replicate sender composition, but whether the committee should foreclose coroutine-native composition by mandating a single model.

A further response is that `std::execution` provides a vocabulary type for async - a common currency that lets libraries interoperate regardless of which runtime they use. This is the strongest form of the universality claim, and it deserves a direct answer. A vocabulary type must be widely adopted, cheaply convertible at library boundaries, and stable across ABI versions. The evidence shows `std::execution` satisfies none of these: adoption outside GPU and HFT is thin (Section 3.5.2), the type-erasure boundary needed for cheap conversion does not exist (Section 3.5.3), and the ABI surface is maximally fragile because the framework is template-heavy and structurally resists erasure (Section 3.5.3). A narrower vocabulary - coroutine-based I/O concepts with type-erased `coroutine_handle` boundaries - would serve the same interoperability purpose with less cost, less risk, and a smaller ABI footprint.

#### 3.5.2 Networking Deferred

The official stdexec documentation states under "Standardization Status (as of 2025)":

> "Interop with networking is being explored for C++29."
>
> [nvidia.github.io/stdexec](https://nvidia.github.io/stdexec/)

The framework ships in C++26 without the use case that started the entire executor discussion a decade ago.

stdexec's only I/O example ([io_uring.cpp](https://github.com/NVIDIA/stdexec/blob/main/examples/io_uring.cpp)) contains zero socket operations, zero reads, zero writes. It demonstrates only timers (`schedule_after`, `schedule_at`). When a user asked about file reading, the maintainer directed them to a third-party repo ([NVIDIA/stdexec#1062](https://github.com/NVIDIA/stdexec/issues/1062)).

The reference implementations - [stdexec](https://github.com/NVIDIA/stdexec) (NVIDIA, 2.2k stars) and [libunifex](https://github.com/facebookexperimental/libunifex) (Meta, 1.7k stars) - have been freely available for years. Adoption stories exist in the domains the framework was designed for: GPU computing, financial infrastructure, heterogeneous hardware. But the broader open-source community has not reached for them. The framework has not achieved the broad voluntary adoption that genuine universal models achieve before standardization. Section 6.4 examines the adoption evidence in detail.

#### 3.5.3 Type Erasure and ABI

[libunifex issue #244](https://github.com/facebookexperimental/libunifex/issues/244) (opened March 2021, still open) documents the structural difficulty of type-erasing senders. The evidence chain is worth tracing in detail.

A user ([Garcia6l20](https://github.com/facebookexperimental/libunifex/issues/244)) tried to implement SSL sockets with libunifex. The SSL handshake requires type-erasing different sender types into a single `any_sender_of<>`. The code failed to compile because `any_sender_of<>` cannot forward the `error_code` that io_uring propagates through `set_error`.

Eric Niebler [diagnosed the structural limitation](https://github.com/facebookexperimental/libunifex/issues/244#issuecomment-810854513):

> "The problem is that the io_uring_context can propagate errors of two types: `std::exception_ptr` and `std::error_code`. `any_sender_of<>` is only able to handle `std::exception_ptr`. There currently isn't a generalization of `any_sender_of<>` than can handle more than `std::exception_ptr`."

His proposed workaround: convert the `error_code` into an `exception_ptr` before it crosses the type-erasure boundary. Niebler [called this](https://github.com/facebookexperimental/libunifex/issues/244#issuecomment-810854513) "not a super-awesome solution."

Lewis Baker [provided the specific mechanism](https://github.com/facebookexperimental/libunifex/issues/244#issuecomment-812866957): the type-erasing receiver would have two `set_error` overloads - one taking `exception_ptr` (passed through), one taking `error_code` that wraps it:

> "the receiver it passes down could have two overloads of `set_error()`. One taking `exception_ptr` and the other taking an `error_code` that forwards on to `set_error(*this, std::make_exception_ptr(std::system_error(ec)))`"

Baker also proposed a proper generalization - parameterizing `any_sender_of` with both value and error overload lists:

> "Longer term, it probably makes sense to allow the any_sender_of type to be parameterisable with both a list of set_value overloads and a list of set_error overloads that it expects/supports, rather than have them hard-coded to exception_ptr or some other error-types."

Neither fix was ever implemented. The issue remains open five years later. The only existing `any_sender_of` implementation - libunifex's [`any_sender_of.hpp`](https://github.com/facebookexperimental/libunifex/blob/main/include/unifex/any_sender_of.hpp) - confirms the structural limitation in source code: its type-erased receiver hard-codes `set_error` to accept only `std::exception_ptr`. NVIDIA's [stdexec](https://nvidia.github.io/stdexec/reference/index.html) ships no type-erased sender at all. No WG21 paper proposes one for C++26 or C++29.

C++26 `std::execution` ships without a type-erased sender, and the gap is not a missing paper - it is a structural consequence of the multi-channel completion model. The sender/receiver completion model has multiple error channels, the type-erasure boundary can only erase one, and the framework's own authors' workaround is to convert `error_code` to `exception_ptr` via `make_exception_ptr(system_error(ec))` - a mechanism the lead author calls "not a super-awesome solution." Whether the three-channel model is itself universal is an open question; [D4000](https://wg21.link/p4000) ("Where Does the Error Code Go?") examines the classification problem and finds that the three-channel model (`set_value`, `set_error`, `set_stopped`) forces every I/O library to choose which channel carries the error code, with no convention for the correct answer, and silently different `when_all` behavior depending on which convention a given sender follows. This matters for three reasons.

**ABI stability.** WG21 has maintained ABI stability across C++14, C++17, and C++20 ([P1863R1](https://open-std.org/jtc1/sc22/wg21/docs/papers/2020/p1863r1.pdf)). Breaking ABI costs the ecosystem "engineer-millennia." Templates and header-only libraries create ABI fragility because "template implementations must appear in headers, forcing recompilation into every translation unit that uses them" ([Gentoo C++ ABI analysis](https://blogs.gentoo.org/mgorny/2012/08/20/the-impact-of-cxx-templates-on-library-abi/)). Type-erased interfaces provide stable ABI boundaries: implementation changes stay behind the erasure wall, no recompilation needed. A framework that structurally resists type erasure maximizes ABI risk. A design that embraces type erasure, like coroutine handles which are inherently type-erased, would give the committee what it wants: async capabilities with ABI stability. This is a design that solves the committee's own stated problem.

**Error model mismatch.** Networking code prefers `error_code` over exceptions because errors are frequent - they are normal operational outcomes, not exceptional events. Connections reset, reads return EOF, operations get cancelled. These happen constantly. Exceptions carry performance overhead and dislocate control flow to unrelated areas of code; because errors occur with such frequency in networking, it is more performant and more convenient to handle them immediately at the call site. The resulting code is easier to audit and the code path is already hot. Beyond failure reporting, `error_code` can also indicate a success condition - why an operation succeeded. The end of the body was reached. The end of the stream was reached. The end of the file was reached. These are not failures, and the caller gains value from distinguishing these outcomes. Boost.Asio's documentation makes the point concretely: "An EOF error may be used to distinguish the end of a stream from a successful read of size 0" ([Boost.Asio Design](https://www.boost.org/doc/libs/release/doc/html/boost_asio/design/eof.html)). An `exception_ptr` cannot carry this information without absurdity - throwing an exception to report success. In [libunifex #244](https://github.com/facebookexperimental/libunifex/issues/244), Niebler confirms `any_sender_of` is hard-coded to `exception_ptr` while io_uring propagates `error_code`. The framework that claims universality produces a type-erased form that is itself not universal.

**Compile times.** [Asio issue #1100](https://github.com/chriskohlhoff/asio/issues/1100) is a feature request for type-erased handlers. The author states: "Some of us still care about compile times and being able to apply the Pimpl idiom. This is not possible when our libraries are forced to be header-only because of Asio." Kohlhoff responded by implementing `any_completion_handler`. A framework that structurally resists type erasure forces the template-heavy model on every user.

#### 3.5.4 Design Immaturity

[D4007R0](https://wg21.link/p4007) ("std::execution Needs More Time") documents a fundamental timing gap between the backward-flow context model and coroutine frame allocation. Eric Niebler, in [P3826R3](https://wg21.link/p3826) (2026-01-05), characterizes part of the design in his own words:

> "The receiver is not known during early customization. Therefore, early customization is irreparably broken."

When the lead author of a framework describes part of its design as "irreparably broken," that is evidence worth weighing.

The companion paper ["The Velocity of Change in `std::execution`"](https://github.com/cppalliance/wg21-papers/blob/master/source/D0000-execution-churn.md) surveys the published record systematically: since `std::execution` was approved for C++26 at Tokyo in March 2024, the committee has processed 50 items - 34 papers, 11 LWG defects, and 5 national body comments - modifying a single feature in 22 months. The rate of change has accelerated from 0.88 items/month to 2.80 items/month over the first four complete meeting periods, the subjects are not converging, and the severity of discovered defects has not decreased - two Priority 1 safety defects were filed 16 months after approval. A feature approaching stability would show the opposite trajectory on all three measures.

The closest precedent for comparison is `<ranges>`. After its adoption for C++20, `<ranges>` accumulated roughly 30 LWG issues in its first two years, most at Priority 2-3. `std::execution` has accumulated 11 LWG issues in less time, including two Priority 1 safety defects affecting core mechanisms (`connect` and `transform_sender`) that most sender/receiver programs exercise. The defect count may be comparable; the severity is not. Ranges' P1 defects at comparable maturity were design-quality issues: return types ([LWG 3186](https://cplusplus.github.io/LWG/issue3186)), naming ([LWG 3379](https://cplusplus.github.io/LWG/issue3379)), and semantic constraints ([LWG 3363](https://cplusplus.github.io/LWG/issue3363)). None were safety defects. `std::execution`'s P1 defects are dangling-reference vulnerabilities and unconstrained type aliases in core mechanisms. As of February 2026, the [LWG priority tracker](https://cplusplus.github.io/LWG/unresolved-prioritized.html) lists 8 Priority 1 issues in section 33 (`std::execution`) and zero in ranges. The severity profile is qualitatively different. Beyond defects, `std::execution` has required 34 papers to fix, rework, or complete - a volume that `<ranges>` did not require in a comparable period.

#### 3.5.5 Papers Still in Flight

["The Velocity of Change in `std::execution`"](https://github.com/cppalliance/wg21-papers/blob/master/source/D0000-execution-churn.md) catalogues all 50 items in detail. The summary: of the 34 post-approval papers, 2 removed features entirely (`ensure_started`, `start_detached`, `split`), 13 reworked fundamental design aspects (sender algorithm customization was rewritten three times), 11 added missing functionality (including `system_context`, `async_scope`, and `task` itself), and the remaining papers fix wording or address post-adoption issues. The 11 LWG defects include two at Priority 1 - a dangling-reference vulnerability in `transform_sender` and an unconstrained alias in `connect_result_t`. Five national body comments target `task`'s allocator model, requiring architectural changes to the coroutine type adopted only six months earlier. Switzerland's CD ballot comment identifies signal-safety as "a serious defect."

Of the papers that address design rather than wording, zero are about networking. The subjects span GPU dispatch, operation-state lifetimes, scheduler affinity, forward progress guarantees, allocator propagation, and diagnostic quality - the breadth of a framework still finding its shape, not one converging toward stability.

#### 3.5.5a The Universality Test

Beyond the velocity of change, three companion papers examine specific points where `std::execution` fails the universality test - cases where the framework cannot accommodate features already in the C++ language or error handling models already dominant in practice:

1. **Stop token gap.** C++20 added `std::stop_token` as a standard cancellation mechanism. When a plain awaitable - not a sender - is `co_await`ed inside `std::execution::task`, it has no standard mechanism to receive the stop token. The sender path creates an `awaitable-receiver` that bridges to the promise's environment; the plain awaitable path does not. [P3552R3](https://wg21.link/p3552) requires `task` to be "awaiter/awaitable friendly." [P3796R1](https://wg21.link/p3796r1) acknowledges the opposite: "awaitable non-senders are not supported." [D0000](https://wg21.link/p0000) ("How Do Plain Awaitables Receive a Stop Token?") documents this gap and demonstrates that the design space for solving it is not empty.

2. **Error classification gap.** The three-channel completion model forces every I/O library to classify its error codes into channels, with no convention for the correct answer. [D4000](https://wg21.link/p4000) ("Where Does the Error Code Go?") shows that when two libraries make different choices, generic algorithms like `when_all` produce silently different behavior. The framework that claims universality cannot agree with itself on how errors should be reported.

3. **Allocator timing gap.** [D4007](https://wg21.link/p4007) ("std::execution Needs More Time") documents that the backward-flow context model delivers the allocator too late for coroutine frame allocation. The receiver - and with it the allocator - is not known until `connect()`, but the coroutine frame must be allocated before `connect()` can be called.

A framework that cannot propagate `std::stop_token` to plain awaitables, cannot type-erase its own senders without losing `error_code`, and has no convention for which completion channel carries the error code - can it be called universal?

These are not edge cases. `std::stop_token` is a C++20 language feature. `error_code` is the dominant error model in networking code. Plain awaitables are the mechanism that C++20 coroutines provide for async composition. A universal execution model must accommodate the language's own features. The evidence shows `std::execution` does not.

#### 3.5.6 The Design Space Remains Open

[D4003](https://wg21.link/p4003) ("IoAwaitables: A Coroutines-Only Framework") demonstrates an alternative execution model purpose-built for I/O that diverges significantly from `std::execution`. Its existence proves the design space has not converged.

If WG21 commits to one execution model as universal, it may close off the design space for alternatives like IoAwaitables, [TooManyCooks](https://github.com/tzcnt/TooManyCooks) (a C++20 coroutine runtime optimized for raw performance), and Asio's completion token model. The history of C++ suggests that enabling multiple approaches, and letting the ecosystem converge naturally, has served the language better than mandating convergence from above.

---

## 4. Narrow Abstractions Win

C++ has a strong track record with universal abstractions. But the ones that succeed share a distinctive property: they are narrow.

### 4.1 Where Universality Succeeded in C++

**STL Iterators.** Stepanov's traversal protocol works across every container type. His key insight: "One seldom needs to know the exact type of data on which an algorithm works since most algorithms work on many similar types" ([Stepanov, "The Standard Template Library," 1994](https://stepanovpapers.com/Stepanov-The_Standard_Template_Library-1994.pdf)). Iterators capture one essential property, traversal, and leave everything else to the user. The same `std::sort` works on arrays, vectors, and deques. It works because the abstraction is narrow.

**RAII.** Constructor acquires, destructor releases. This pattern works across every resource type: memory, files, sockets, locks, GPU handles. Bjarne Stroustrup introduced it in the 1980s as a way to make resource management exception-safe ([cppreference: RAII](https://en.cppreference.com/w/cpp/language/raii)). It emerged from practice, not from committee design. It is minimal, stable, and universal across every C++ domain. It works because the abstraction is narrow.

**Allocators.** The allocator model lets every standard container work with any memory strategy: pool allocators, arena allocators, `std::pmr` resources, or the default heap. Containers become composable building blocks regardless of the memory strategy underneath ([cppreference: Allocator](https://en.cppreference.com/w/cpp/named_req/Allocator)). It works because the abstraction is narrow.

Iterators abstract over traversal. Allocators abstract over memory strategy. RAII abstracts over resource lifetime. Each captures one essential property and leaves everything else to the user.

`std::execution` tries to abstract over scheduling, context propagation, error handling, cancellation, algorithm dispatch, and hardware backend selection all at once. That is not one essential property. That is six.

### 4.2 Should C++ Choose Tradeoffs for Its Users?

C++ has historically given programmers control over their tradeoffs. You don't pay for what you don't use. You choose the abstractions appropriate to your domain. The narrow abstractions above embody this principle: iterators don't choose your container, allocators don't choose your memory strategy, RAII doesn't choose your resource.

A mandated universal execution model would represent a departure from this tradition. It would say: we have decided which tradeoffs are right for async, across all domains, for all users.

Perhaps that is the right call. But consider the evidence.

### 4.3 Multiple Valid Execution Models in the Wild

In async, no similarly narrow universal abstraction has emerged. Instead, multiple valid execution models coexist, each optimized for different tradeoffs:

**[TooManyCooks](https://github.com/tzcnt/TooManyCooks)** is a C++20 coroutine runtime with its own execution model (work-stealing thread pool, `tmc::task`, `tmc::spawn_tuple`), optimized for raw coroutine throughput via continuation stealing. It does not use `std::execution`. It is equally valid for its domain.

**[Boost.Asio](https://www.boost.org/doc/libs/release/doc/html/boost_asio.html) / Networking TS** has its own execution model, optimized for I/O completion with platform-native proactors (IOCP on Windows, epoll on Linux, kqueue on BSD). It supports multiple continuation styles (callbacks, futures, stackful coroutines, C++20 coroutines) and user-defined ones through completion tokens. Kohlhoff's [N3747](https://open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3747.pdf) ("A Universal Model for Asynchronous Operations," 2013) showed how a single initiating function can adapt to caller-chosen completion styles. This is a different kind of universality, proven across two decades of production use.

**[Capy](https://github.com/cppalliance/capy)** has its own execution model, optimized for coroutine-first ergonomics with forward-flowing context (executor, allocator, stop token propagated from caller to callee). [Corosio](https://github.com/cppalliance/corosio) uses it. Even within the same domain (async I/O), multiple valid execution models coexist because they optimize for different things.

**[gRPC](https://github.com/grpc/grpc)**, 44.3k GitHub stars. Google's production RPC framework, written primarily in C++. It has its own async model with both completion-queue-based and [callback-based](https://grpc.io/docs/languages/cpp/callback) APIs. It does not use `std::execution`.

**[libuv](https://github.com/libuv/libuv)**, 26.5k GitHub stars. The event-driven async I/O library that powers Node.js. It provides its own event loop backed by epoll, kqueue, IOCP, and event ports, with cross-platform support for TCP/UDP sockets, DNS, file I/O, IPC, and child processes ([libuv docs](https://docs.libuv.org/en/stable)). It does not use `std::execution`.

**[Seastar](https://github.com/scylladb/seastar)**, 9.1k GitHub stars. The framework behind ScyllaDB. Seastar uses a shared-nothing design that "shards all requests onto individual cores" with "explicit message passing rather than shared memory between threads" ([seastar.io](https://www.seastar.io/)). Its futures-and-promises execution model is fundamentally incompatible with sender/receiver. It does not use `std::execution`.

**[cppcoro](https://github.com/lewissbaker/cppcoro)**, 3.8k GitHub stars. A C++ coroutine abstractions library created by Lewis Baker, who went on to co-author [P2300](https://wg21.link/p2300). cppcoro predates P2300, and Baker's experience with it informed the sender/receiver design. It provides its own task types, schedulers, and async primitives. It does not use `std::execution`. The point is not that Baker should have stuck with cppcoro, but that the design space accommodates fundamentally different execution models - and it has not converged on one.

The existence of these models is not a failure to be corrected. It is evidence that the problem space is too rich for a single wide model to dominate.

If the cost of mandating the wrong model is borne by every C++ programmer for decades, this question deserves careful, unhurried consideration.

But the diversity of these models is not just a warning - it is an asset. Each has solved real problems for real users. The question is not which one to anoint, but what narrow contract would let them interoperate. That contract is smaller than any of these frameworks, and it may already exist.

---

## 5. What Works and What Doesn't

Genuine universal models do exist. It would be intellectually dishonest to ignore them. But they share characteristics that distinguish them from the models that fail.

**TCP/IP.** David Clark (1988) describes how TCP/IP's design philosophy evolved through "the repeated pattern of implementation and testing that occurred before the standards were set" ([Clark, "The Design Philosophy of the DARPA Internet Protocols"](https://www.cs.princeton.edu/~jrex/teaching/spring2005/reading/clark88.pdf)). Key features like the datagram service and the IP/TCP layering "were not part of the original proposal" but emerged from iterative deployment. The standard formalized what had already proven successful in operational use.

**IEEE 754.** Before the standard, floating-point arithmetic was chaos. William Kahan recalls numbers that "could behave as non-zero in comparisons but as zeros in multiplication" ([Kahan interview](https://people.eecs.berkeley.edu/~wkahan/ieee754status/754story.html)). IEEE 754 emerged from Intel's practical need for a coprocessor, designed by Kahan based on decades of hands-on experience with IBM, Cray, and CDC systems. It codified best practices from existing hardware, not theoretical ideals.

Both emerged from practice, not from committee-designed frameworks. Both were minimal and stable. Both achieved broad voluntary adoption across disparate domains without coercion. Both proved themselves through deployment before being standardized.

The models that fail look different. They are wide, comprehensive, and designed top-down. The models that succeed alongside them look different too. They are narrow, specialized, and composable.

**Unix pipes.** Doug McIlroy (1978): "Make each program do one thing well... Expect the output of every program to become the input to another" ([Wikipedia: Unix philosophy](https://en.wikipedia.org/wiki/Unix_philosophy)). The pipe, a byte stream, is the narrowest possible contract. It enabled an ecosystem of specialized tools that compose freely.

**TCP/IP's narrow waist.** The Internet's hourglass architecture puts a single, simple spanning layer (IP) at the center. "Simplicity and generality at the waist outperform richer, feature-heavy designs in real-world adoption and evolution" ([Wikipedia: Hourglass model](https://en.wikipedia.org/wiki/Hourglass_model)). Innovation happens above and below the waist, not at the waist. OSI tried to make the waist wide. It failed.

**Asio's completion tokens.** Kohlhoff's [N3747](https://open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3747.pdf) ("A Universal Model for Asynchronous Operations," 2013) showed how a single initiating function adapts to caller-chosen completion styles: callbacks, futures, stackful coroutines, C++20 coroutines, and user-defined ones. The narrow contract is the completion token. The model does not impose a continuation style; it lets each caller choose. Two decades of production use validate this approach.

The pattern is consistent. Narrow contracts enable broad ecosystems. Wide abstractions constrain them. When universality succeeds, it is minimal. When it fails, it is comprehensive.

If a universal model exists for async in C++, it will follow the practice-first pattern: broad voluntary adoption across disparate domains, without mandating it. If you have to mandate it, it is not universal. `std::execution` has not yet passed this test. If it does not, the alternative is not chaos. It is specialization with interoperation through narrow contracts, the approach that has worked everywhere else.

The ingredients for async C++'s narrow waist are not hypothetical. Coroutines provide the suspension mechanism. Buffer sequences provide the I/O vocabulary. `error_code` provides the error model. `stop_token` provides cancellation. The pieces exist. What remains is assembling them into the contract that lets the ecosystem build.

---

## 6. The Problem Is Already Solved

Before presenting that contract, we address the argument that standardization is needed to solve a coordination problem - and show that no such problem exists.

A key premise of `std::execution` is that C++ needs a standard framework for heterogeneous and parallel computing. But the ecosystem has not been waiting. The coordination problems that `std::execution` aims to solve are already solved by widely adopted, production-proven libraries.

### 6.1 GPU and Heterogeneous Computing

**NVIDIA CCCL** (Thrust, CUB, libcu++), 2.1k GitHub stars ([GitHub](https://github.com/NVIDIA/cccl)). Thrust is described as "the C++ parallel algorithms library which inspired the introduction of parallel algorithms to the C++ Standard Library" ([NVIDIA docs](https://nvidia.github.io/cccl/thrust/)). It is included in the NVIDIA HPC SDK and CUDA Toolkit. NVIDIA is not waiting for `std::execution` to ship GPU parallel algorithms. They already ship them.

**Kokkos**, 2.4k GitHub stars ([GitHub](https://github.com/kokkos/kokkos)). A performance portability layer from Sandia National Laboratories that enables "manycore performance portability through polymorphic memory access patterns" ([Kokkos documentation](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Introduction.html)). Code written with Kokkos runs on CPUs, Intel Xeon Phi, and GPUs without platform-specific rewrites.

**RAJA**, 560 GitHub stars ([GitHub](https://github.com/llnl/RAJA)). Lawrence Livermore National Laboratory's performance portability layer for DOE exascale applications. RAJA has been "proven in production with most LLNL ASC applications and numerous ECP applications" across NVIDIA, AMD, and Intel GPUs ([LLNL project page](https://computing.llnl.gov/projects/raja-managing-application-portability-next-generation-platforms)).

**OpenMP** accounts for "45% of all analyzed parallel APIs" on GitHub, making it the "dominant parallel programming model" with "steady and continuous growth in popularity over the past decade" ([Quantifying OpenMP, 2023](https://arxiv.org/pdf/2308.08002)). OpenMP 6.0 (November 2024) provides full GPU offload support ([OpenMP 6.0 announcement](https://openmp.org/home-news/openmp-arb-releases-openmp-6-0-for-easier-programming)).

### 6.2 Task Parallelism and Async

**Taskflow**, 11.6k GitHub stars ([GitHub](https://github.com/taskflow/taskflow)). "A General-purpose Task-parallel Programming System" that supports heterogeneous CPU-GPU computing and has demonstrated solving "a large-scale machine learning workload up to 29% faster, 1.5x less memory, and 1.9x higher throughput than the industrial system, oneTBB, on a machine of 40 CPUs and 4 GPUs" ([Taskflow paper](https://arxiv.org/pdf/2004.10908)).

**oneTBB** (Intel), 6.5k GitHub stars ([GitHub](https://github.com/uxlfoundation/oneTBB)). "A flexible performance library" for parallel computing that has been in production use for over 15 years ([oneTBB documentation](https://uxlfoundation.github.io/oneTBB/)).

**HPX**, 2.8k GitHub stars ([GitHub](https://github.com/STEllAR-GROUP/hpx)). "A general purpose C++ runtime system for parallel and distributed applications of any scale" that has demonstrated 96.8% parallel efficiency on 643,280 cores ([HPX website](https://hpx.stellar-group.org/)).

**folly** (Meta), 30.2k GitHub stars ([GitHub](https://github.com/facebook/folly)). Meta's production C++ library including `folly::Futures` and `folly::coro`, which power async operations across Meta's infrastructure at scale. Meta is not waiting for `std::execution`. They ship folly.

**Abseil** (Google), 17k GitHub stars ([GitHub](https://github.com/abseil/abseil-cpp)). "The fundamental building blocks that underpin most of what Google runs," drawn from Google's internal codebase and "production-tested and fully maintained" ([Abseil about page](https://abseil.io/about/)). Google is not waiting for `std::execution` either.

### 6.3 The Cost of Adding More

Every feature added to the C++ standard must be implemented and maintained by standard library vendors. There are exactly three major implementations: libstdc++ (GNU), libc++ (LLVM), and Microsoft's STL. Christopher Di Bella (Google, libc++ contributor) observes: "Due to its vast complexity, there have only been a handful of standard library implementations to date" ([C++Now 2024 talk](https://www.youtube.com/watch?v=bXlm3taD6lw)).

The committee itself is strained. Bryce Adelstein Lelbach notes the committee has received "10x more proposals over the past decade" and describes it as "300 individual authors, not 1 team" ([Convenor candidacy](https://brycelelbach.github.io/cpp_convenor/)). [P2656R2](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p2656r2.html) observes that "the community is struggling to manage the challenges of the complexity and variability of the tools, technologies, and systems that make C++ possible."

David Sankel (Adobe) captured the risk in [P3023R1](https://open-std.org/jtc1/sc22/wg21/docs/papers/2023/p3023r1.html):

> "The surest way to sabotage a standard is to say yes to everything."

Adding `std::execution` to the standard carries a cost. That cost must be weighed against the benefit.

### 6.4 The Costs Are Socialized, the Benefits Are Not

The strongest argument for standardization is that it solves a coordination problem: if everyone needs the same thing and nobody can agree on which library to use, the standard breaks the deadlock by picking one.

But no such coordination problem exists for `std::execution`. And the domains it serves do not need what standardization provides.

**GPU developers do not need the standard.** They use NVIDIA's compiler with CUDA extensions and a GPU-specific fork (`nvexec`) that standard C++ cannot express (see §6.5). They need `__device__`, `__global__`, `<<<>>>`, and `cudaStream_t`. The standard cannot give them any of these. They will use `nvexec` regardless of what WG21 decides.

**HFT and financial infrastructure do not need the standard.** They use user-mode networking stacks on specified hardware. They routinely fork entire libraries for nanosecond-level optimization. They do not want portability - they want the opposite: deterministic, hardware-specific, fully controllable execution. Standardization's benefits - long-term maintenance, cross-platform guarantees, canonical implementations - are properties these domains actively avoid.

**The ordinary developer gets no benefit either.** The ordinary async C++ developer - the majority - would benefit from portability and long-term maintenance. But they get a framework that does not serve their use case: networking and I/O with the build-time and ABI properties they need (Section 3.5.1). They bear the cost of standardization without receiving the benefit. An objection: ordinary developers doing task parallelism (not I/O) could benefit from a standard `when_all` and `task` without needing networking. This is true but does not change the calculus - those users are equally well-served by [oneTBB](https://github.com/uxlfoundation/oneTBB), [Taskflow](https://github.com/taskflow/taskflow), or coroutine-based alternatives, all available on vcpkg today without consuming pages of the standard.

**The library is already available.** `stdexec` is on [vcpkg](https://vcpkg.link/ports/stdexec): `vcpkg install stdexec`. Anyone who wants sender/receiver can use it right now. Boost has thrived for decades without standardization ([boost.org](https://www.boost.org/users/)). Google published [Abseil](https://github.com/abseil/abseil-cpp) (17k GitHub stars) as a standalone library without consuming a single page of the standard ([abseil.io/about](https://abseil.io/about/)). Package managers have eliminated the distribution problem - vcpkg offers over 2,600 libraries ([vcpkg.io](https://vcpkg.io/en/)), Conan hosts nearly 1,900 recipes ([conan.io/center](https://conan.io/center)). Nobody is blocked. NVIDIA ships CUDA. Meta ships folly. Intel ships oneTBB. ScyllaDB ships Seastar. The ecosystem is not waiting.

**The framework's complexity is not hypothetical.** Consider a tree search from the stdexec examples repository ([backtrack.cpp](https://github.com/steve-downey/sender-examples/blob/main/src/examples/backtrack.cpp)):

```cpp
auto search_tree(auto                    test,
                 tree::NodePtr<int>      tree,
                 stdexec::scheduler auto sch,
                 any_node_sender&&       fail) -> any_node_sender {
    if (tree == nullptr) {
        return std::move(fail);
    }
    if (test(tree)) {
        return stdexec::just(tree);
    }
    return stdexec::on(sch, stdexec::just()) |
           stdexec::let_value([=, fail = std::move(fail)]() mutable {
               return search_tree(
                   test,
                   tree->left(),
                   sch,
                   stdexec::on(sch, stdexec::just()) |
                       stdexec::let_value(
                           [=, fail = std::move(fail)]() mutable {
                               return search_tree(
                                   test, tree->right(), sch, std::move(fail));
                           }));
           });
}
```

The failure continuation is itself a sender pipeline (`on(sch, just()) | let_value(...)`) passed as a parameter to a recursive function. Each recursive call nests another `let_value` lambda that [captures and moves](https://stackoverflow.com/questions/32486623/moving-a-lambda-once-youve-move-captured-a-move-only-type-how-can-the-lambda) the failure sender. This is [continuation-passing style](https://en.wikipedia.org/wiki/Continuation-passing_style) expressed as sender composition - a technique [used more frequently by compilers than by programmers](https://en.wikipedia.org/wiki/Continuation-passing_style) as an intermediate representation, not as a style intended for human authorship. The reader must trace the `fail` parameter through three levels of `std::move` to understand which path executes - the same [pyramid of nesting](https://callbackhell.com/) that drove JavaScript's evolution from callbacks to promises to `async`/`await`. For further illustration, the same repository demonstrates [a fold operation requiring type-erased recursive sender returns](https://github.com/steve-downey/sender-examples/blob/main/src/examples/fold.cpp) and [a loop using mutable lambda captures with reference-captured locals inside nested senders](https://github.com/steve-downey/sender-examples/blob/main/src/examples/loop.cpp). This is what the committee is asking developers to adopt as their universal async model.

**The asymmetry is stark.** If `std::execution` is deferred from C++26 and offered as a standalone library:

- Users who want it lose nothing. They install it from vcpkg.
- Implementers gain relief from a massive implementation burden.
- The design gains time to mature, fix the 10+ papers in flight, and prove itself across domains.
- The committee gains bandwidth for higher-priority work. Every major programming language ships networking in its standard library: [Python](https://docs.python.org/3/library/socket.html), [Java](https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/net/package-summary.html), [Go](https://pkg.go.dev/net), [Rust](https://doc.rust-lang.org/std/net/), [C#](https://learn.microsoft.com/en-us/dotnet/api/system.net.sockets), [JavaScript/Node.js](https://nodejs.org/api/net.html). None of them ship a sender/receiver execution framework. Networking is the more universal need, and C++ is the only major language that still lacks it.

If `std::execution` stays in C++26 and the design proves wrong:

- Every C++ programmer inherits the cost.
- Three standard library teams must implement and maintain it indefinitely.
- The ABI is locked. Mistakes cannot be corrected without breaking the world.
- The standard's credibility is diminished.

The cost of including `std::execution` in the standard is enormous and permanent. The benefit accrues to few, and those few are already well-served without it.

But the alternative is not nothing. The alternative is a standard that gives ordinary developers what they actually need: async I/O that compiles fast, erases types cleanly, and lets them build the applications that every other language already supports.

### 6.5 NVIDIA Already Ships Sender/Receiver for GPU Without the Standard

The GPU use case is the primary motivation for `std::execution`'s design. But NVIDIA already ships a complete sender/receiver GPU integration as a standalone library, and it requires their own non-standard compiler. The standard is not involved.

**The code lives in `nvexec`, not `std::execution`.** NVIDIA's stdexec repository contains a separate namespace, `nvexec`, with GPU-specific sender algorithms ([github.com/NVIDIA/stdexec/tree/main/include/nvexec](https://github.com/NVIDIA/stdexec/tree/main/include/nvexec)). The files are `.cuh` (CUDA header) files, not standard C++ headers. They include GPU-specific reimplementations of `bulk`, `then`, `when_all`, `continues_on`, `let_value`, `split`, `reduce`, and more.

**The GPU scheduler uses CUDA-specific types that standard C++ cannot express.** The `stream_context.cuh` file ([source](https://github.com/NVIDIA/stdexec/blob/main/include/nvexec/stream_context.cuh)) defines a `stream_scheduler` whose completion signatures include `cudaError_t`, a CUDA-specific error type:

```cpp
using completion_signatures =
    STDEXEC::completion_signatures<set_value_t(), set_error_t(cudaError_t)>;
```

The scheduler's `schedule()` method is annotated with CUDA execution space specifiers:

```cpp
STDEXEC_ATTRIBUTE(nodiscard, host, device) auto schedule() const noexcept {
    return sender{ctx_};
}
```

The `host, device` annotation maps to CUDA's `__host__ __device__`, a non-standard extension. The `stream_context` constructor calls `cudaGetDevice()`, a CUDA runtime API function.

**GPU programming requires non-standard language extensions.** NVIDIA's [CUDA C/C++ Language Extensions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html) documentation enumerates the extensions that standard C++ cannot express:

- Execution space specifiers: "The execution space specifiers `__host__`, `__device__`, and `__global__` indicate whether a function executes on the host or the device."
- Memory space specifiers: "The memory space specifiers `__device__`, `__managed__`, `__constant__`, and `__shared__` indicate the storage location of a variable on the device."
- Kernel launch syntax: "The execution configuration is specified by inserting an expression in the form `<<<grid_dim, block_dim, dynamic_smem_bytes, stream>>>` between the function name and the parenthesized argument list."

None of these are valid C++. Code that uses them cannot be compiled by GCC, MSVC, or Clang without CUDA support. Every NVIDIA GPU user already depends on a non-standard compiler.

**What this means.** The primary use case for `std::execution`'s completion domains, `bulk` algorithm, and algorithm customization machinery is GPU dispatch. But GPU dispatch already works, today, in a standalone library (`nvexec`), distributed through stdexec, available on vcpkg, requiring NVIDIA's own compiler. Adding `std::execution` to the C++ standard does not change this. NVIDIA's users need `nvcc` regardless. The standard cannot express `__device__`, `__global__`, `<<<>>>`, or `cudaStream_t`. The GPU integration lives outside the standard by necessity, and it will continue to live outside the standard no matter what WG21 decides. The cost of standardization falls on implementers and the committee; the use case that motivates much of the design complexity cannot benefit from it.

---

## 7. Conclusion

Every genuine universal model examined in this paper shares two properties: narrow scope and practice-first emergence. TCP/IP, IEEE 754, iterators, RAII, and allocators each capture one essential property and leave everything else to the user. None was designed top-down by committee and then mandated into existence. Each proved itself through broad voluntary adoption before standardization codified the outcome.

`std::execution` follows the opposite pattern. The design is wide, the companion papers reveal fundamental open questions, and the framework ships in C++26 with a contested coroutine task type ([P3801](https://wg21.link/p3801), [P3796](https://wg21.link/p3796)), without type-erased senders, and without the networking use case that started the executor discussion a decade ago. The domain is genuinely difficult. Acknowledging that difficulty and taking more time is not a failure - it is prudent engineering.

The asymmetry of risk favors caution. Deferring `std::execution` costs nothing: the library is on vcpkg today, and the ecosystem is not waiting. Mandating a design that proves wrong costs decades of ABI lock-in, implementer burden, and foreclosed alternatives.

### Recommendations

This paper makes two asks. They are different in scope and political feasibility, and the paper deliberately separates them so the committee can act on either independently.

**The primary recommendation: do not gate networking on `std::execution`.** Networking should proceed on its own terms, with I/O-optimized concepts that serve the ordinary developer, without waiting for `std::execution` to solve problems it was not designed for. This is the paper's strongest ask and the one most directly supported by the evidence. The SG1 poll (§3.1), the GPU-oriented design artifacts (§3.5.1), the networking deferral (§3.5.2), and the deployment reality (§6.4, §6.5) all point to the same conclusion: `std::execution` should not be a prerequisite for networking. If the committee does nothing else, it should ensure that networking need not wait for an execution model that has not proven itself for I/O.

**The secondary recommendation: defer `std::execution` from C++26.** The evidence supports this too - the 10+ companion papers in flight, the contested task type, the absence of type-erased senders, the mismatch with the needs of most async developers. If the committee is willing to take this larger step, the paper argues it is justified. But it is a bigger ask with lower political probability, and the primary recommendation does not depend on it.

Other options exist - shipping `std::execution` in C++26 while pursuing networking in parallel, or shipping the sender/receiver core without the contested `task` type. Both are compatible with the primary recommendation but neither addresses the disproportionate risk of freezing a wide abstraction before it has stabilized. Fifty post-approval items, two Priority 1 safety defects, and an accelerating rate of change are properties of the sender/receiver framework itself, not of `task` alone.

Beyond these two asks, this paper respectfully suggests:

1. **Let execution models compete.** The ecosystem already has multiple production-proven models (Asio, folly, Seastar, TooManyCooks, Taskflow, and others). Rather than picking a winner, let voluntary adoption identify which abstractions deserve standardization - the same process that produced TCP/IP, IEEE 754, and the STL.

2. **Standardize the narrow async I/O contract, not the execution model.** [D4003](https://wg21.link/p4003) ("IoAwaitables: A Coroutines-Only Framework") demonstrates what such a contract looks like in practice and provides production benchmarks. The contract is small:

    ```cpp
    // error code and byte count delivered together
    struct io_result
    {
        std::error_code ec;
        std::size_t bytes_transferred = 0;
    };

    // An awaitable whose await_suspend receives the I/O environment
    template<typename A>
    concept IoAwaitable =
        requires(A a, std::coroutine_handle<> h, io_env const* env)
        { a.await_suspend(h, env); };

    // A writable byte stream
    template<typename T>
    concept WriteStream =
        requires(T& stream, const_buffer buffers)
        {
            { stream.write_some(buffers) } -> IoAwaitable;
            requires awaitable_decomposes_to<
                decltype(stream.write_some(buffers)),
                std::error_code, std::size_t>;
        };
    ```

    A concrete `write_sink` satisfying this concept. The implementation of `do_write` lives in a `.cpp` file behind an ABI boundary:

    ```cpp
    class write_sink
    {
        void* impl_;

        std::coroutine_handle<>
        do_write(
            std::coroutine_handle<> h,
            const_buffer buf,
            io_env const* env,
            std::error_code* ec,
            std::size_t* n);

    public:
        struct write_some_awaitable
        {
            write_sink* self_;
            const_buffer buf_;
            std::error_code ec_;
            std::size_t n_ = 0;

            bool await_ready() const noexcept { return false; }

            std::coroutine_handle<>
            await_suspend(
                std::coroutine_handle<> h,
                io_env const* env)
            {
                // stop token, executor, allocator available via env
                return self_->do_write(h, buf_, env, &ec_, &n_);
            }

            io_result await_resume() noexcept { return {ec_, n_}; }
        };

        write_some_awaitable
        write_some(const_buffer buf)
        {
            return write_some_awaitable{this, buf};
        }
    };
    ```

    A coroutine uses it:

    ```cpp
    task<> send_response(write_sink& sink, std::string_view msg)
    {
        auto [ec, n] = co_await sink.write_some(
            const_buffer(msg.data(), msg.size()));
        if (ec)
            log("write stopped after", n, "bytes:", ec.message());
    }
    ```

    The `io_env` carries the stop token, executor, and allocator - all delivered through `await_suspend` without coupling the awaitable to any promise type. The `do_write` signature takes `std::coroutine_handle<>` and `io_env const*`, both fixed types. The implementation can change without recompiling callers. `ReadStream`, `Stream`, `connect`, and `close` follow the same pattern. The network runtime itself - the event loop, the thread pool, the platform reactor - stays in the ecosystem, where it belongs. The standard provides only the contract through which libraries interoperate.

    Ousterhout captures the design principle:

    > "The best modules are deep: they have a lot of functionality hidden behind a simple interface. A deep module is a good abstraction because only a small fraction of its internal complexity is visible to its users."
    >
    > [Ousterhout, *A Philosophy of Software Design*](https://web.stanford.edu/~ouster/cgi-bin/aposd.php) (2018)

    An async I/O contract that standardizes `read`, `write`, `connect`, and `close` is a deep module: a small interface hiding the event loop, the reactor, and the platform behind a few fixed types. The benefits are substantial:

    - **Second-order library effects.** A deep I/O abstraction enables an entire ecosystem of libraries built on top of it: HTTP, WebSocket, database clients, message queues, monitoring agents - all interoperable, all composable, without mandating a single runtime. This is the tower of abstraction that [D4008](https://wg21.link/p4008) identifies as C++'s missing piece.

    - **Runtime choice becomes less consequential.** When the standard provides the I/O concepts, the user's choice of network runtime - Asio, libuv, a custom io_uring loop - matters less. Libraries written against the standard concepts work with any runtime that satisfies them. Interoperability comes from the contract, not from mandating a single framework.

    - **ABI stability through type erasure.** Coroutines are inherently type-erased at the `coroutine_handle` boundary. A coroutine-based I/O contract takes enormous pressure off the ABI stability mandate because the runtime is behind the erasure wall. Implementation changes stay behind it. No recompilation needed. This gives the committee what it has always wanted: async capabilities with ABI stability.

    - **Faster compile times and better encapsulation.** I/O concepts behind type-erased coroutine interfaces enable separate compilation, `pimpl`, and shared libraries. The developer writing a web service gets fast builds and clean module boundaries.

    - **TLS stays in the ecosystem, where it belongs.** TLS/SSL has been a point of contention for WG21 since the Networking TS, and a specification of that size and complexity would overwhelm the committee. A narrow I/O contract sidesteps this entirely: the standard specifies async read/write/connect/close on byte streams. TLS wraps a byte stream and produces another byte stream. The TLS implementation - OpenSSL, BoringSSL, SChannel, SecureTransport - lives in the ecosystem where domain experts maintain it and platform vendors integrate their native APIs. Apple uses SecureTransport. Microsoft uses SChannel. The standard does not get in their way. The committee gets 80% of what it needs for 20% of the cost.

A note on maturity. The bigger the framework, the wider the abstraction, the greater the need for evidence of maturity before standardization. Scope determines the maturity bar, not the other way around.

`std::execution` bundles six concerns - scheduling, context propagation, error handling, cancellation, algorithm dispatch, hardware backend selection - and has produced 50 post-approval items including two Priority 1 safety defects. A framework of this scope demands a high maturity bar, and the evidence shows it has not met it.

[D4003](https://wg21.link/p4003) and its implementations (Capy, Corosio) are deliberately deep - a small interface over a six-function I/O contract. They ask less of the standard: fewer concepts, a smaller ABI surface, no algorithm customization machinery, no completion domains. They cover sockets, timers, TLS, and DNS resolution on multiple platforms, with production benchmarks showing parity or better than Asio callbacks (1.01x-1.36x throughput depending on thread count; no comparison against `stdexec` is possible because `stdexec` has no networking implementation to benchmark). They do not yet cover every edge case of a decade-old framework. Specific gaps include: TCP only (no UDP or other transports), no io_uring backend (epoll, IOCP, and kqueue are implemented; io_uring is planned), heap allocation elision (HALO) depends on a Clang-specific attribute (`[[clang::coro_await_elidable]]`) and only applies to immediately-awaited tasks, and allocator propagation uses a thread-local window mechanism that requires a specific two-call invocation syntax. These are real limitations - and they are disclosed here so the committee can weigh them alongside `std::execution`'s own open problems.

But these are the kind of gaps a narrow contract can close incrementally. TCP-only becomes TCP+UDP in a point release. An io_uring backend is an implementation detail behind an existing concept. These are bounded problems with bounded solutions. The open problems in `std::execution` - type-erasure, the error model, algorithm customization rewritten three times ([P2999R3](https://wg21.link/p2999r3), [P3303R1](https://open-std.org/jtc1/sc22/wg21/docs/papers/2024/p3303r1.html), [P3826R3](https://wg21.link/p3826)) - are fundamental design questions that a wide framework may never resolve. A narrow contract carries less risk and should be afforded a correspondingly lower bar for maturation evidence. After all, it asks less of the standard. The standard's job is the narrow waist, not the full stack. The asymmetry in maturity is itself a consequence of the asymmetry in scope. That is the point.

A narrow async I/O contract in the standard library would unlock the ecosystem that every other major language already enjoys. HTTP clients and servers. WebSocket connections for real-time data. REST API endpoints. Interactive chat applications. Multiplayer game networking. gRPC services. Database connection pools. Cloud service integrations. Message queue consumers. Monitoring agents. All of these exist today in Python, Go, Rust, Java, and JavaScript as composable libraries built on a standard async I/O foundation. C++ has the users who need them, the performance characteristics they demand, and the coroutine machinery to express them elegantly. What it lacks is the six-function contract at the bottom that makes everything above it composable across libraries without mandating a single runtime.

The committee's job is not to design a universal execution framework. It is to standardize that narrow waist, that *deep module*, that **elegant minimal abstraction** - and let practitioners build on top.

---

## Appendix A: TAPS Client Example (RFC 9622 Section 3.1.2)

The following is the complete client example from [RFC 9622](https://datatracker.ietf.org/doc/html/rfc9622), Section 3.1.2, "Client Example." It shows how an application opens two Connections to a remote server, sends a request, and receives a response on each. This is the TAPS equivalent of a POSIX `socket` / `connect` / `write` / `read` / `close` sequence. It is reproduced verbatim from the specification.

```
RemoteSpecifier := NewRemoteEndpoint()
RemoteSpecifier.WithHostName("example.com")
RemoteSpecifier.WithService("https")

TransportProperties := NewTransportProperties()
TransportProperties.Require(preserve-msg-boundaries)
// Reliable data transfer and preserve order are required by default

SecurityParameters := NewSecurityParameters()
TrustCallback := NewCallback({
    // Verify the identity of the Remote Endpoint and return the result
})
SecurityParameters.SetTrustVerificationCallback(TrustCallback)

// Specifying a Local Endpoint is optional when using Initiate
Preconnection := NewPreconnection(RemoteSpecifier,
                                  TransportProperties,
                                  SecurityParameters)

Connection := Preconnection.Initiate()
Connection2 := Connection.Clone()

Connection -> Ready<>
Connection2 -> Ready<>

//---- Ready event handler for any Connection C begin ----
C.Send(messageDataRequest)

// Only receive complete messages
C.Receive()
//---- Ready event handler for any Connection C end ----

Connection -> Received<messageDataResponse, messageContext>
Connection2 -> Received<messageDataResponse, messageContext>

// Close the Connection in a Receive event handler
Connection.Close()
Connection2.Close()
```

The example above is the *happy path*. It does not show the configuration surface that surrounds it. RFC 9622 defines 18 Selection Properties (Section 6.2), each taking a 5-valued Preference Enumeration (`Prohibit`, `Avoid`, `No Preference`, `Prefer`, `Require`):

1. Reliable Data Transfer
2. Preservation of Message Boundaries
3. Configure Per-Message Reliability
4. Preservation of Data Ordering
5. Use 0-RTT Session Establishment with a Safely Replayable Message
6. Multistream Connections in a Group
7. Full Checksum Coverage on Sending
8. Full Checksum Coverage on Receiving
9. Congestion Control
10. Keep-Alive Packets
11. Interface Instance or Type
12. Provisioning Domain Instance or Type
13. Use Temporary Local Address
14. Multipath Transport
15. Advertisement of Alternative Addresses
16. Direction of Communication
17. Notification of ICMP Soft Error Message Arrival
18. Initiating Side Is Not the First to Write

8 Security Parameter categories (Section 6.3):

1. Allowed Security Protocols
2. Certificate Bundles
3. Pinned Server Certificate
4. Application-Layer Protocol Negotiation
5. Groups, Ciphersuites, and Signature Algorithms
6. Session Cache Options
7. Pre-Shared Key
8. Connection Establishment Callbacks

11 Connection Properties for runtime tuning (Section 8.1):

1. Required Minimum Corruption Protection Coverage for Receiving
2. Connection Priority
3. Timeout for Aborting Connection
4. Timeout for Keep-Alive Packets
5. Connection Group Transmission Scheduler
6. Capacity Profile
7. Policy for Using Multipath Transports
8. Bounds on Send or Receive Rate
9. Group Connection Limit
10. Isolate Session
11. Read-Only Connection Properties

8 per-Message Properties set on each Send (Section 9.1.3):

1. Lifetime (`msgLifetime`)
2. Priority (`msgPriority`)
3. Ordered (`msgOrdered`)
4. Reliable (`msgReliable`)
5. Checksum Coverage (`msgChecksumLen`)
6. Capacity Profile (`msgCapacityProfile`)
7. No Fragmentation (`noFragmentation`)
8. No Segmentation (`noSegmentation`)

And 10+ asynchronous events the application must be prepared to handle, including `Ready`, `Sent`, `Expired`, `SendError`, `Received`, `ReceivedPartial`, `ReceiveError`, `SoftError`, and `PathChange`.

The specification states:

> "It is intended to replace the BSD Socket API as the common interface to the transport layer."
>
> [RFC 9622](https://datatracker.ietf.org/doc/html/rfc9622), Abstract

> "These features SHOULD be implemented even when, in a specific implementation, it will always result in no operation."
>
> [RFC 9622](https://datatracker.ietf.org/doc/html/rfc9622), Section 5

This is the model that [P3482](https://wg21.link/p3482) proposes as the basis for C++ networking.

---

## References

1. Butler Lampson. "Hints for Computer System Design." 1983. http://research.microsoft.com/en-us/um/people/blampson/33-Hints/Acrobat.pdf

2. Engler et al. "Exokernel: An Operating System Architecture for Application-Level Resource Management." MIT, 1995. https://people.eecs.berkeley.edu/~brewer/cs262b/hotos-exokernel.pdf

3. Ted Kaminski. "The One Ring Problem." 2018. https://tedinski.com/2018/01/30/the-one-ring-problem-abstraction-and-power.html

4. Andrew Russell. "OSI: The Internet That Wasn't." IEEE Spectrum, 2013. https://spectrum.ieee.org/osi-the-internet-that-wasnt

5. Richard Gabriel. "Objects Have Failed." OOPSLA, 2002. https://dreamsongs.com/Files/ObjectsHaveFailed.pdf

6. Wikipedia. "Composition over inheritance." https://en.wikipedia.org/wiki/Composition_over_inheritance

7. P2453R0. "2021 October Library Evolution and Concurrency Networking and Executors Poll Outcomes." WG21, 2022. https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2453r0.html

8. CppCon 2024. "Why Google Doesn't Allow Ranges in Our Codebase." Daisy Hollman. https://cppcon2024.sched.com/event/1gZgc/why-google-doesnt-allow-ranges-in-our-codebase

9. NanoRange wiki. "Compile times." https://github.com/tcbrindle/NanoRange/wiki/Compile-times

10. Daniel Lemire. "std::ranges may not deliver the performance that you expect." 2025. https://lemire.me/blog/2025/10/05/stdranges-may-not-deliver-the-performance-that-you-expect/

11. IETF TAPS Working Group. Charter page. https://datatracker.ietf.org/wg/taps/about/

12. P3482R1. Rodgers & Kühl. "Design for C++ networking based on IETF TAPS." WG21. https://wg21.link/p3482

13. NEAT Project. https://www.neat-project.org/

14. P2300R10. Dominiak et al. "std::execution." WG21, 2024. https://open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html

15. P1241R0. "Merging Coroutines into C++." WG21, 2018. https://open-std.org/jtc1/sc22/wg21/docs/papers/2018/p1241r0.html

16. libunifex. Facebook Experimental. https://github.com/facebookexperimental/libunifex

17. stdexec. NVIDIA. https://nvidia.github.io/stdexec/

18. P2999R3. Niebler. "Sender Algorithm Customization." WG21, 2023. https://wg21.link/p2999r3

19. P3303R1. Niebler. "Fixing Lazy Sender Algorithm Customization." WG21, 2024. https://open-std.org/jtc1/sc22/wg21/docs/papers/2024/p3303r1.html

20. P3826R3. Niebler. "Fix Sender Algorithm Customization." WG21, 2026. https://wg21.link/p3826

21. D4007R0. Falco. "std::execution Needs More Time." WG21, 2026. https://wg21.link/p4007

22. stdexec io_uring.cpp. https://github.com/NVIDIA/stdexec/blob/main/examples/io_uring.cpp

23. NVIDIA/stdexec#1062. "io_uring reading files." https://github.com/NVIDIA/stdexec/issues/1062

24. libunifex issue #244. "Question about any_sender_of usage." https://github.com/facebookexperimental/libunifex/issues/244

25. P1863R1. "ABI breakage." WG21, 2020. https://open-std.org/jtc1/sc22/wg21/docs/papers/2020/p1863r1.pdf

26. Gentoo. "The impact of C++ templates on library ABI." 2012. https://blogs.gentoo.org/mgorny/2012/08/20/the-impact-of-cxx-templates-on-library-abi/

27. Boost.Asio. "Why EOF is an error." https://www.boost.org/doc/libs/release/doc/html/boost_asio/design/eof.html

28. Asio issue #1100. "Feature request: Type-erased handler wrapper." https://github.com/chriskohlhoff/asio/issues/1100

29. P2079. "System execution context." WG21. https://wg21.link/p2079

30. P3164. "Improving diagnostics for sender expressions." WG21. https://wg21.link/p3164

31. P3373. "Of Operation States and Their Lifetimes." WG21. https://wg21.link/p3373

32. P3388. "When Do You Know connect Doesn't Throw?" WG21. https://wg21.link/p3388

33. P3425. "Reducing operation-state sizes for subobject child operations." WG21. https://wg21.link/p3425

34. P3481. "std::execution::bulk() issues." WG21. https://wg21.link/p3481

35. P3552. "Add a Coroutine Task Type." WG21. https://wg21.link/p3552

36. P3557. "High-Quality Sender Diagnostics with Constexpr Exceptions." WG21. https://wg21.link/p3557

37. P3564. "Make the concurrent forward progress guarantee usable in bulk." WG21. https://wg21.link/p3564

38. D4003. Falco et al. "IoAwaitables: A Coroutines-Only Framework." WG21. https://wg21.link/p4003

39. TooManyCooks. https://github.com/tzcnt/TooManyCooks

40. N3747. Kohlhoff. "A Universal Model for Asynchronous Operations." WG21, 2013. https://open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3747.pdf

41. Wikipedia. "Unix philosophy." https://en.wikipedia.org/wiki/Unix_philosophy

42. Wikipedia. "Hourglass model." https://en.wikipedia.org/wiki/Hourglass_model

43. David Clark. "The Design Philosophy of the DARPA Internet Protocols." 1988. https://www.cs.princeton.edu/~jrex/teaching/spring2005/reading/clark88.pdf

44. William Kahan. "An Interview with the Old Man of Floating-Point." https://people.eecs.berkeley.edu/~wkahan/ieee754status/754story.html

45. Hacker News discussion on std::ranges. https://news.ycombinator.com/item?id=40317350

46. Boost.Asio. https://www.boost.org/doc/libs/release/doc/html/boost_asio.html

47. Capy. https://github.com/cppalliance/capy

48. Corosio. https://github.com/cppalliance/corosio

49. NVIDIA CCCL (Thrust, CUB, libcu++). https://github.com/NVIDIA/cccl

50. NVIDIA Thrust documentation. https://nvidia.github.io/cccl/thrust/

51. Kokkos. https://github.com/kokkos/kokkos

52. Kokkos Programming Guide: Introduction. https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Introduction.html

53. RAJA. Lawrence Livermore National Laboratory. https://github.com/llnl/RAJA

54. RAJA project page. LLNL. https://computing.llnl.gov/projects/raja-managing-application-portability-next-generation-platforms

55. "Quantifying OpenMP: Statistical Insights into Usage and Adoption." 2023. https://arxiv.org/pdf/2308.08002

56. OpenMP 6.0 announcement. https://openmp.org/home-news/openmp-arb-releases-openmp-6-0-for-easier-programming

57. Taskflow. https://github.com/taskflow/taskflow

58. Taskflow paper. https://arxiv.org/pdf/2004.10908

59. oneTBB (Intel). https://github.com/uxlfoundation/oneTBB

60. oneTBB documentation. https://uxlfoundation.github.io/oneTBB/

61. HPX. https://github.com/STEllAR-GROUP/hpx

62. HPX website. https://hpx.stellar-group.org/

63. folly (Meta). https://github.com/facebook/folly

64. Abseil (Google). https://github.com/abseil/abseil-cpp

65. Abseil about page. https://abseil.io/about/

66. Christopher Di Bella. "What Does It Take to Implement the C++ Standard Library?" C++Now 2024. https://www.youtube.com/watch?v=bXlm3taD6lw

67. Bryce Adelstein Lelbach. Convenor candidacy. https://brycelelbach.github.io/cpp_convenor/

68. P2656R2. "C++ Ecosystem International Standard." WG21, 2023. https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p2656r2.html

69. P3023R1. Sankel. "C++ Should Be C++." WG21, 2023. https://open-std.org/jtc1/sc22/wg21/docs/papers/2023/p3023r1.html

70. Stepanov. "The Standard Template Library." 1994. https://stepanovpapers.com/Stepanov-The_Standard_Template_Library-1994.pdf

71. gRPC. https://github.com/grpc/grpc

72. gRPC C++ Callback API Tutorial. https://grpc.io/docs/languages/cpp/callback

73. libuv. https://github.com/libuv/libuv

74. libuv documentation. https://docs.libuv.org/en/stable

75. Seastar. https://github.com/scylladb/seastar

76. Seastar website. https://www.seastar.io/

77. cppcoro. Lewis Baker. https://github.com/lewissbaker/cppcoro

78. stdexec on vcpkg. https://vcpkg.link/ports/stdexec

79. vcpkg. Microsoft. https://vcpkg.io/en/

80. Conan Center. https://conan.io/center

81. Boost background information. https://www.boost.org/users/

82. cppreference. "RAII." https://en.cppreference.com/w/cpp/language/raii

83. cppreference. "Allocator (named requirement)." https://en.cppreference.com/w/cpp/named_req/Allocator

84. Python standard library: socket. https://docs.python.org/3/library/socket.html

85. Java standard library: java.net. https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/net/package-summary.html

86. Go standard library: net. https://pkg.go.dev/net

87. Rust standard library: std::net. https://doc.rust-lang.org/std/net/

88. C# standard library: System.Net.Sockets. https://learn.microsoft.com/en-us/dotnet/api/system.net.sockets

89. Node.js standard library: net. https://nodejs.org/api/net.html

90. nvexec (NVIDIA GPU sender/receiver integration). https://github.com/NVIDIA/stdexec/tree/main/include/nvexec

91. nvexec stream_context.cuh source. https://github.com/NVIDIA/stdexec/blob/main/include/nvexec/stream_context.cuh

92. NVIDIA CUDA C/C++ Language Extensions. https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html

93. Michi Henning. "The Rise and Fall of CORBA." ACM Queue 4, no. 5 (June 2006). https://dl.acm.org/doi/10.1145/1142031.1142044

94. David Chappell. "The Trouble With CORBA." 1998. https://davidchappell.com/writing/article_Trouble_CORBA.php

95. RFC 9621. "Architecture and Requirements for Transport Services." IETF, January 2025. https://datatracker.ietf.org/doc/html/rfc9621

96. RFC 9622. "An Abstract Application Programming Interface (API) for Transport Services." IETF, January 2025. https://datatracker.ietf.org/doc/html/rfc9622

97. John Ousterhout. *A Philosophy of Software Design.* Yaknyam Press, 2018. https://web.stanford.edu/~ouster/cgi-bin/aposd.php

98. Robert Leahy. "std::execution in Asio Codebases: Adopting Senders Without a Rewrite." CppCon 2025. https://cppcon2025.sched.com/event/27bQ1/stdexecution-in-asio-codebases-adopting-senders-without-a-rewrite

99. Wikipedia. "Continuation-passing style." https://en.wikipedia.org/wiki/Continuation-passing_style

100. callbackhell.com. "Callback Hell: Taming JavaScript's Async Complexity." https://callbackhell.com/

101. steve-downey/sender-examples. "backtrack.cpp." https://github.com/steve-downey/sender-examples/blob/main/src/examples/backtrack.cpp

102. steve-downey/sender-examples. "fold.cpp." https://github.com/steve-downey/sender-examples/blob/main/src/examples/fold.cpp

103. steve-downey/sender-examples. "loop.cpp." https://github.com/steve-downey/sender-examples/blob/main/src/examples/loop.cpp

104. Stack Overflow. "Moving a lambda: once you've move-captured a move-only type, how can the lambda be used?" https://stackoverflow.com/questions/32486623/moving-a-lambda-once-youve-move-captured-a-move-only-type-how-can-the-lambda

105. D4008. Falco. "The C++ Standard Cannot Connect to the Internet." WG21. https://wg21.link/p4008

106. GitHub dependency graph: NVIDIA/stdexec. https://github.com/NVIDIA/stdexec/network/dependents

107. GitHub dependency graph: facebookexperimental/libunifex. https://github.com/facebookexperimental/libunifex/network/dependents

108. libunifex `any_sender_of.hpp` source. https://github.com/facebookexperimental/libunifex/blob/main/include/unifex/any_sender_of.hpp

109. D4000. Falco & Gill. "Where Does the Error Code Go?" WG21, 2026. https://wg21.link/p4000

110. LWG 3186. "ranges removal, partition, and partial_sort_copy algorithms discard useful information." https://cplusplus.github.io/LWG/issue3186

111. LWG 3363. "drop_while_view should opt-out of sized_range." https://cplusplus.github.io/LWG/issue3363

112. LWG 3379. "safe in several library names is misleading." https://cplusplus.github.io/LWG/issue3379

113. LWG unresolved prioritized list. https://cplusplus.github.io/LWG/unresolved-prioritized.html

114. D0000. Falco. "How Do Plain Awaitables Receive a Stop Token?" WG21, 2026. https://wg21.link/p0000
