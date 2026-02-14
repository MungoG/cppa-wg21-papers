::: {.document-info}
| Document   | D4000R0                                    |
|------------|--------------------------------------------|
| Date:      | 2026-02-09                                 |
| Reply-to:  | Vinnie Falco \<vinnie.falco@gmail.com\>    |
| Audience:  | All of WG21                                |
::: 

# The C++ Standard Cannot Connect to the Internet

## Abstract

This paper discusses a key observation: The C++ Standard cannot connect to the Internet. The problem is not missing sockets. Third-party networking libraries exist. The problem is that C++ lacks a standard asynchronous execution model designed for I/O, which prevents asynchronous algorithms from composing across library boundaries. Every other major programming language solved this problem by standardizing an async I/O foundation, and the result was an explosion of higher-level frameworks. C++ has no such ecosystem because there is no agreed-upon foundation to build upon. This paper examines the evidence, proposes priorities that a standard async model should reflect, and asks the committee to consider whether networking deserves higher priority than it currently receives.

---

## 1. The Observation

The C++ standard cannot connect to the internet, and it is not for lack of trying. Third-party networking libraries exist and are widely used. [Boost.Asio](https://www.boost.org/doc/libs/release/doc/html/boost_asio.html) has been in production for over twenty years. But Asio itself ships as two incompatible published versions, Boost.Asio and [standalone Asio](https://github.com/chriskohlhoff/asio), with different namespaces and different build configurations. This is the coordination problem in miniature: even the most popular, most mature C++ networking library cannot agree with itself on a single interface.

The deeper problem is not sockets. It is the absence of a standard.

If a programmer wants to write a composable async algorithm, for example an HTTP conversation, a WebSocket message exchange, or an asynchronous file transfer with optional compression, what execution model should they use? There is no answer. There are Asio completion tokens, Meta's [folly::coro](https://github.com/facebook/folly/blob/main/folly/experimental/coro/README.md), [Qt signals and slots](https://doc.qt.io/qt-6/signalsandslots.html), custom callback systems, promise/future chains, and ad-hoc event loops. Each is an island. Algorithms expressed in one model cannot compose with algorithms from another.

Not every committee member works with network programming daily. WG21 is full of experts in language design, template metaprogramming, numerics, concurrency, safety, and many other domains. This paper is written from a networking practitioner's perspective, for colleagues who may not have the same daily experience with this problem. It aims to explain, with evidence, why the networking gap matters and what it costs. Readers seeking background on asynchronous I/O concepts - event loops, completion handlers, coroutine-based networking - may find the tutorial in [D4003](https://wg21.link/p4003) Appendix A helpful before proceeding.

---

## 2. The Tower of Abstraction

Every major programming language standardized an async foundation. In some cases this was a full I/O model in the standard library. In others it was a narrow composability trait that third-party runtimes build upon. Either way, the standardized foundation enabled tall towers of abstraction. The foundation came first. The frameworks followed.

**Python** standardized `asyncio` in Python 3.4 (2014) via [PEP 3156](https://peps.python.org/pep-3156/), providing both the async model and I/O integration in the standard library. The ecosystem built upward. [Django](https://github.com/django/django) (86.7k GitHub stars), the web framework "for perfectionists with deadlines," sits atop this foundation.

**JavaScript** standardized `Promise` in ES2015 (June 2015) and `async`/`await` in ES2017 ([TC39 proposal](https://tc39.es/proposal-async-await/)). The language standard defines the composability primitive; the I/O runtime comes from the host environment (Node.js, browsers, Deno). Because every runtime implements the same `Promise` contract, libraries like [Express.js](https://github.com/expressjs/express) (68.7k stars) and [Next.js](https://github.com/vercel/next.js) (137.6k stars) compose across all of them.

**Go** shipped goroutines, channels, and `net/http` in the standard library from day one (2009). The `net/http` package is imported by [1,705,800 known packages](https://pkg.go.dev/net/http). The entire Go microservices ecosystem is built on this foundation.

**Rust** stabilized `async`/`await` in Rust 1.39 (November 2019; [Rust blog announcement](https://blog.rust-lang.org/2019/11/07/Async-await-stable/)) and standardized the [`Future`](https://doc.rust-lang.org/std/future/trait.Future.html) trait in `std`. The I/O runtime is not in the standard library; [Tokio](https://github.com/tokio-rs/tokio) (31k stars) fills that role. But because every runtime implements the same `Future` trait, libraries like [hyper](https://github.com/hyperium/hyper) and [Axum](https://github.com/tokio-rs/axum) (24.9k stars) are runtime-agnostic. Rust standardized the narrow composability contract, and the ecosystem built on it.

**Java** has had `java.net` in the standard library since JDK 1.0 (1996). The ecosystem built upward. [Spring Boot](https://github.com/spring-projects/spring-boot) (79.9k stars) sits atop this foundation.

**C#** shipped `async`/`await` and the `Task` type in C# 5.0 (2012; [overview](https://dotnetcurry.com/csharp/869/async-await-csharp-dotnet)), with a built-in thread pool and I/O completion in the standard library. The ecosystem built upward. [ASP.NET Core](https://github.com/dotnet/aspnetcore) (37.7k stars) sits atop this foundation.

In every case, standardization of an async foundation, whether a full I/O model or a narrow composability trait, enabled the ecosystem to build upward. The frameworks came after the foundation. The combined GitHub stars of these higher-level frameworks exceed 500,000 - an imperfect proxy, but one that reflects broad adoption and active communities.

**C++** added coroutines in C++20 but standardized neither an async I/O model nor a composability trait for async operations. The language machinery is there. The foundation is not. C++ bottoms out at [Boost.Asio](https://github.com/boostorg/asio) (1.5k stars, third party, no standard status), or raw POSIX/Winsock. There is no Django, no Express, no Spring Boot of C++. Not because C++ programmers are less capable, but because there is no standard foundation to build upon.

---

## 3. The Coordination Problem

The C++ standard exists to solve coordination problems. [P2000R4](https://wg21.link/p2000r4) ("Direction for ISO C++") was created specifically to address concerns about C++ "losing coherency due to proposals based on differing and sometimes mutually contradictory design philosophies." The standard provides a shared foundation so that independently developed libraries can interoperate.

The async I/O domain is the textbook case where that foundation is missing.

Without a standard async model for I/O, every library invents its own. [Boost.Asio](https://www.boost.org/doc/libs/release/doc/html/boost_asio.html) uses completion tokens. Meta's [folly::coro](https://github.com/facebook/folly/blob/main/folly/experimental/coro/README.md) is "a developer-friendly asynchronous C++ framework based on the Coroutines TS." [Qt](https://doc.qt.io/qt-6/signalsandslots.html) uses signals and slots, described as "Qt's central communication mechanism between objects, serving as an alternative to callbacks." Beyond these, there are custom callback systems, promise/future chains, and ad-hoc event loops in countless codebases.

Algorithms written for one model cannot compose with another. An HTTP library built on Asio completion tokens cannot be used by code built on folly::coro. A TLS wrapper using Qt's event loop cannot plug into [Boost.Beast](https://www.boost.org/doc/libs/release/libs/beast/). Each library is an island.

This fragmentation is the direct cause of the shallow abstraction tower. Nobody builds Django-scale frameworks in C++ because doing so would require building the entire stack from scratch: HTTP parsing, request routing, session management, templating - with no standard composable libraries at each level to build upon. In Python, each layer exists as a reusable library because the layers below it are agreed upon. In C++, each layer is an island, so the next layer up does not get built.

### The "No Dependencies" Culture

C++ libraries routinely advertise "no external dependencies" or "header-only" as a selling point. [nlohmann/json](https://github.com/nlohmann/json) (48.8k stars) advertises "no external dependencies beyond a C++11 compliant compiler." [cpp-httplib](https://github.com/yhirose/cpp-httplib) (16k stars) bills itself as "a C++ header-only HTTP/HTTPS server and client library" with no dependencies. [fmt](https://github.com/fmtlib/fmt) (23.2k stars) highlights "no external dependencies."

This cultural norm is a symptom of the missing foundation. In a healthy ecosystem, depending on shared infrastructure is normal. In Python, depending on `asyncio` is not a liability. In C++, depending on Boost.Asio is treated as a burden.

John Lakos's *Large-Scale C++ Software Design* ([Addison-Wesley, 1996](https://informit.com/store/large-scale-c-plus-plus-software-design-9780201633627)) established that well-structured systems with clear hierarchical dependencies are "fundamentally easier and more economical to maintain, test, and reuse." The "no dependencies" instinct inverts this principle. It treats isolation as a virtue when shared foundations would be more productive.

The absence of a standard async model makes this dysfunction worse. Since no foundation exists, every library must reinvent it or avoid async entirely. cpp-httplib, the most popular C++ HTTP library listed above, warns in its own README:

> "This library uses 'blocking' socket I/O. If you are looking for a library with 'non-blocking' socket I/O, this is not the one that you want."

A 16k-star HTTP library cannot offer async because there is no standard async foundation to build upon.

---

## 4. How We Got Here

In 2014, the C++ committee decided to "adopt existing practice" for networking, basing a proposal on [Boost.Asio](https://www.boost.org/doc/libs/release/doc/html/boost_asio.html) ([P3185R0](https://wg21.link/p3185r0) "A Proposed Direction for C++ Standard Networking Based on IETF TAPS" documents this history). Chris Kohlhoff published a [reference implementation of the Networking TS](https://github.com/chriskohlhoff/networking-ts-impl).

By 2021, the executor debate had consumed the effort. On 2021-09-28, SG1 polled whether "one grand unified model" for asynchronous execution was needed. The result was no consensus: 4 SF, 9 WF, 5 N, 5 WA, 1 SA. [P2453R0](https://wg21.link/p2453r0) documents these polls. Nobody objected to portable socket wrappers. The schism was entirely about the execution model.

The committee chose [P2300](https://wg21.link/p2300) (`std::execution`), a sender/receiver framework. It is now in the C++26 working draft.

C++26 is about to ship `std::execution`, an asynchronous execution model. The committee decided it needed a standard async framework. That instinct was correct. But the framework that landed was designed for GPU and parallel computing (section 5.1 presents the evidence; [On Universal Models](d0000-on-universal-models.md) section 3.5.1 develops it in detail). The [stdexec documentation](https://nvidia.github.io/stdexec/) confirms: "Interop with networking is being explored for C++29." Networking standardization is deferred to C++29 at the earliest.

The preceding account is not blame. The Networking TS was proposed in 2014. It is now 2026. The C++ standard still cannot connect to the internet. The domain is genuinely difficult. The instinct to find a universal model is natural. But the cost of the delay is real. The committee recognized the need for a standard async model, and that recognition was right. The question is whether the model that landed serves the most important use case.

The preceding sections argue that networking deserves higher priority than it currently receives. That argument stands regardless of what async model the committee pursues. The following section is separable: it proposes design criteria for whatever model the committee explores. A reader who accepts the priority argument need not endorse these specific criteria, and vice versa.

---

## 5. Priorities for a Standard Async Model

If C++ is going to standardize an asynchronous execution model, it should reflect the following priorities, each grounded in evidence. This section proposes those priorities, evaluates `std::execution` against each, and addresses the questions that remain.

### 5.1 It Should Put Networking First

Every other major programming language standardized an async foundation for networking, whether a full I/O model or a narrow composability trait (section 2). None of them ship a sender/receiver execution framework. C++ ships neither.

GPU users already have CUDA, which requires NVIDIA's non-standard compiler. The `__device__`, `__global__`, and `<<<>>>` syntax are not valid C++ ([CUDA C/C++ Language Extensions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)). Adding `std::execution` to the standard provides no benefit to GPU users that [stdexec on vcpkg](https://vcpkg.link/ports/stdexec) does not already provide.

`std::execution`'s design priorities explicitly target GPU computing. The specification itself provides the evidence. [P2300R10](https://wg21.link/p2300r10) section 1.1 frames the motivation around "GPUs in the world's fastest supercomputer." Section 1.2 prioritizes "the diversity of execution resources and execution agents, because not all execution agents are created equal." Section 1.3.2's second end-user example is "Asynchronous inclusive scan," a GPU parallel primitive using `bulk`. The `bulk` algorithm has no networking analog. The entire sender algorithm customization lineage ([P2999R3](https://wg21.link/p2999r3) "Sender Algorithm Customization," [P3303R1](https://wg21.link/p3303r1) "Fixing Lazy Sender Algorithm Customization," [P3826](https://wg21.link/p3826) "Fix Sender Algorithm Customization") is about domain-based algorithm dispatch. These papers contain zero mentions of networking, sockets, or I/O.

The post-approval record tells the same story. Since `std::execution` was approved for C++26 at Tokyo in March 2024, the committee has processed over 50 items - papers, LWG defects, and national body comments. Of those that address design rather than wording, zero are about networking. The subjects span GPU dispatch, operation-state lifetimes, scheduler affinity, forward progress guarantees, allocator propagation, and diagnostic quality. ([On Universal Models](d0000-on-universal-models.md) section 3.5.5 catalogues all 50 items.)

The reference implementations confirm the gap. stdexec's only I/O example ([io_uring.cpp](https://github.com/NVIDIA/stdexec/blob/main/examples/io_uring.cpp)) contains zero socket operations, zero reads, zero writes - only timers. When a user asked about file reading, the maintainer directed them to a third-party repository ([NVIDIA/stdexec#1062](https://github.com/NVIDIA/stdexec/issues/1062)). The reference implementations - [stdexec](https://github.com/NVIDIA/stdexec) (2.2k stars) and [libunifex](https://github.com/facebookexperimental/libunifex) (1.7k stars) - have been freely available for years. The broader open-source community has not reached for them.

Meanwhile, every major programming language ships standard networking: [Python](https://docs.python.org/3/library/socket.html) `socket`, [Java](https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/net/package-summary.html) `java.net`, [Go](https://pkg.go.dev/net) `net`, [Rust](https://doc.rust-lang.org/std/net/) `std::net`, [C#](https://learn.microsoft.com/en-us/dotnet/api/system.net.sockets) `System.Net.Sockets`, [Node.js](https://nodejs.org/api/net.html) `net`. None of them ship a sender/receiver execution framework. C++ ships `std::execution` but not networking. The priorities are inverted.

What does the committee's chosen model actually look like for networking? Maikel Nadolski's [senders-io](https://github.com/maikel/senders-io) library is the most complete attempt to build networking on top of `stdexec` senders. The following is the complete TCP echo server from that repository - the simplest possible network program that reads data and writes it back:

```cpp
#include <sio/net_concepts.hpp>
#include <sio/ip/tcp.hpp>
#include <sio/io_uring/socket_handle.hpp>
#include <sio/sequence/let_value_each.hpp>
#include <sio/sequence/ignore_all.hpp>

#include <exec/repeat_effect_until.hpp>
#include <exec/variant_sender.hpp>
#include <exec/when_any.hpp>

#include <iostream>

template <class ThenSender, class ElseSender>
exec::variant_sender<ThenSender, ElseSender>
  if_then_else(bool condition, ThenSender then, ElseSender otherwise) {
  if (condition) {
    return then;
  }
  return otherwise;
}

using tcp_socket = sio::io_uring::socket_handle<sio::ip::tcp>;
using tcp_acceptor = sio::io_uring::acceptor_handle<sio::ip::tcp>;

auto echo_input(tcp_socket client) {
  return stdexec::let_value(
    stdexec::just(client, std::array<std::byte, 1024>{}),
    [](auto socket, std::span<std::byte> buffer) {
      return sio::async::read_some(socket, sio::mutable_buffer{buffer}) //
           | stdexec::let_value([=](std::size_t nbytes) {
               return if_then_else(
                 nbytes != 0,
                 sio::async::write(socket, sio::const_buffer{buffer}.prefix(nbytes)),
                 stdexec::just(0));
             })
           | stdexec::then([](std::size_t nbytes) { return nbytes == 0; })
           | exec::repeat_effect_until()
           | stdexec::then([] { std::cout << "Connection closed.\n"; });
    });
}

int main() {
  exec::io_uring_context context{};
  auto endpoint = sio::ip::endpoint{sio::ip::address_v4::any(), 1080};
  auto acceptor = sio::io_uring::acceptor{&context, sio::ip::tcp::v4(), endpoint};

  auto accept_connections = sio::async::use_resources(
    [&](tcp_acceptor acceptor) {
      return sio::async::accept(acceptor) //
           | sio::let_value_each([](tcp_socket client) {
               return exec::finally(echo_input(client), sio::async::close(client));
             })
           | sio::ignore_all();
    },
    acceptor);

  stdexec::sync_wait(exec::when_any(std::move(accept_connections), context.run()));
}
```

Source: [senders-io/examples/tcp_echo_server.cpp](https://github.com/maikel/senders-io/blob/main/examples/tcp_echo_server.cpp)

This is an echo server. It accepts a connection, reads bytes, and writes them back. In Python this is a ten-line program. In Go it is twelve. In C++ built on the committee's chosen execution model, it is 60 lines of `variant_sender`, `let_value`, `repeat_effect_until`, and `use_resources`. A simple conditional branch requires a helper function template returning `exec::variant_sender`. The program requires expertise in sender composition that most C++ programmers do not have and is far removed from the way the C++ community writes code today. If this is the standard's answer to networking, the standard is not speaking to the people who need networking.

For comparison, the session handler from the same echo server built on a coroutines-first model ([Corosio](https://github.com/cppalliance/corosio)). The full Corosio program is longer than the sender version - 145 lines including the worker class, factory, argument parsing, and `ioc.run()`. Only the session handler is shown here because that is where the execution model lives. The rest is ordinary C++ that looks the same regardless of the async model. In the senders-io version above, the sender model pervades every line of the program: the helper template, the type aliases, the accept loop, and `main` are all sender composition. The contrast is not program size; it is how deeply the execution model penetrates the code:

```cpp
capy::task<> do_session() {
    for (;;) {
        buf_.resize(4096);
        auto [ec, n] = co_await sock_.read_some(
            capy::mutable_buffer(buf_.data(), buf_.size()));
        buf_.resize(n);
        auto [wec, wn] = co_await capy::write(
            sock_, capy::const_buffer(buf_.data(), buf_.size()));
        if (wec || ec)
            break;
    }
    sock_.close();
}
```

Source: [corosio/example/echo-server/echo_server.cpp](https://github.com/cppalliance/corosio/blob/develop/example/echo-server/echo_server.cpp)

No `variant_sender`. No `let_value`. No `repeat_effect_until`. The control flow is a `for` loop with `break`. Error handling is a structured binding and an `if`. The coroutines-first model produces code that an ordinary C++ programmer can read, write, and maintain. Corosio is not a finished library - it is early and evolving - but the ergonomic difference is not a matter of library maturity. It follows from the execution model: coroutines express sequential I/O naturally because that is what they were designed for.

**Networking creates towers. GPU does not.** The tower of abstraction argument only works if standardization actually enables higher-level libraries to proliferate. For networking, the evidence is overwhelming: sockets lead to HTTP, which leads to REST, which leads to web frameworks, which lead to full-stack applications. Django, Express, Spring Boot, Axum all sit atop standardized async I/O foundations. Each layer builds on the one below.

The GPU *C++* ecosystem is wide but not tall. CUDA leads to cuDNN, cuBLAS, cuFFT, OptiX, and custom kernels. Each library uses the foundation directly. GPU workloads are domain-specific: custom kernels for physics simulations, ML training loops, rendering pipelines. The vertical stacking that does exist in GPU - the ML/AI tower from CUDA to cuDNN to PyTorch to Hugging Face Transformers to application-level AI services - was built in Python, not in C++ on top of `std::execution`.

But that tower's interface to the world consists of network connections to API endpoints. Every major inference service - OpenAI, Anthropic, Hugging Face, AWS SageMaker - is accessed via REST or gRPC over HTTP. Consider the implications:

**To use the most consequential computing stack of this decade, you must connect to the internet. The C++ standard cannot do that.**

Prioritizing `std::execution` for GPU while deferring the networking foundation is prioritizing the compute engine while deferring the only mechanism through which its results reach users. The return on investment is asymmetric: networking standardization enables layered ecosystem growth (each layer building on the one below), while GPU standardization enables breadth (more libraries at the same level).

### 5.1.1 Why Not Wait for C++29?

The [stdexec documentation](https://nvidia.github.io/stdexec/) states: "Interop with networking is being explored for C++29." A natural response is: wait three years and build networking on top of the sender model that is already in the standard. This section explains why that path does not address the problem.

**The timeline.** C++29 ships around 2031-2032. The Networking TS was proposed in 2014. If networking arrives in C++29, the gap will be 17 years. No other major programming language has taken this long. Python standardized `asyncio` in under a year. Rust shipped `async`/`await` and the `Future` trait in four. Seventeen years is not iteration. It is a generation of C++ programmers who wrote their async code without a standard foundation.

**The ergonomic cost is structural.** The senders-io echo server above is not verbose because the library is immature. The verbosity follows from the sender model itself: `variant_sender` for branching, `let_value` for sequencing, `repeat_effect_until` for loops. These are inherent to sender composition, not artifacts of an early implementation. Time does not fix them. The coroutines-first echo server demonstrates that the alternative already exists.

**The error classification problem is unsolved.** [D4000](https://wg21.link/p4000) ("Where Does the Error Code Go?") documents that the three-channel completion model (`set_value` / `set_error` / `set_stopped`) forces every I/O library to classify its error codes into channels, with no convention for the correct answer. When two libraries make different choices, `when_all` produces silently different behavior depending on which child completes first. A coroutine-based model that returns `io_result<Ts...>` - delivering the error code and all operation-specific values together through `await_resume` - has no channels to choose between, regardless of the number of return values. This is not a maturity gap. It is structural to the sender model.

**Plain awaitables cannot receive a stop token.** [D0000](https://wg21.link/p0000) ("How Do Plain Awaitables Receive a Stop Token?") documents that a plain awaitable `co_await`ed inside `std::execution::task` has no standard mechanism to receive the stop token. The sender path creates an `awaitable-receiver` that bridges to the promise's environment; the plain awaitable path does not. [P3552R3](https://wg21.link/p3552) ("Add a Coroutine Task Type") requires `task` to be "awaiter/awaitable friendly." [P3796R1](https://wg21.link/p3796r1) ("Coroutine Task Issues") acknowledges the opposite: "awaitable non-senders are not supported." Networking-on-senders inherits this gap.

**The universality argument does not hold.** The strongest case for waiting is that a universal model serving GPU, CPU parallelism, and networking is more valuable than separate models because algorithms compose across domains. [On Universal Models](d0000-on-universal-models.md) examines this claim in detail and finds that:

- Type-erased senders do not exist. [libunifex issue #244](https://github.com/facebookexperimental/libunifex/issues/244) has been open for five years. NVIDIA's stdexec ships no type-erased sender. No WG21 paper proposes one. A model that cannot type-erase cannot serve as vocabulary across library boundaries.

- Structured concurrency does not require sender/receiver. `when_all`, `when_any`, `async_scope`, and `async_mutex` are implementable on coroutines alone ([D4003](https://wg21.link/p4003) sections 4.1-4.2).

- The narrow abstractions that succeed in C++ (iterators, RAII, allocators) each capture one essential property. Wide abstractions that bundle six concerns are the pattern that fails (CORBA, OSI).

The design space is not empty. [D4003](https://wg21.link/p4003) (IoAwaitables) demonstrates a coroutines-first execution model with working positions on error handling (structured `io_result<Ts...>`), cancellation (`std::stop_token` propagated forward through `io_env`), and platform abstraction (deliberately left to the transport). These are not hypothetical designs - they have running implementations with production benchmarks. Waiting for C++29 is not the only option, and the structural problems described above suggest it is not the best one.

### 5.2 It Should Put the User Experience First

The intuitive asynchronous operation from C++20 looks like this:

```cpp
auto [ec, n] = co_await stream.read( buf );
```

A standard async model should let users write this. Every language that added async I/O converged on the same user experience: sequential control flow with `async`/`await`. Python shipped it in [2015](https://peps.python.org/pep-0492/). JavaScript in [2017](https://tc39.es/proposal-async-await/). C# in [2012](https://dotnetcurry.com/csharp/869/async-await-csharp-dotnet). Rust in [2019](https://blog.rust-lang.org/2019/11/07/Async-await-stable/). Kotlin in [2018](https://kotlinlang.org/docs/coroutines-overview.html). Zero chose sender pipelines. The industry has answered this question.

C++20 added `co_await`, `co_return`, and `co_yield` as keywords in the language ([P0912R5](https://wg21.link/p0912r5)). The committee made the language design choice. But the framework that shipped does not build on it. `std::execution` arrived in C++26 without a coroutine task type. [P3552](https://wg21.link/p3552) ("Add a Coroutine Task Type") is still in flight, listing 12 unsolved design objectives. The primary way users are expected to write async code was not included in the framework.

The gap is not surface syntax. A natural compromise is to keep `std::execution` as shipped, adopt [P3552](https://wg21.link/p3552)'s `task` type, and build networking I/O operations as senders that users `co_await` through `task`. This would narrow the ergonomic difference at the surface, but the structural problems survive the layer. The three-channel error classification remains: every I/O operation must still choose whether to send its error code through `set_value` or `set_error`, and `when_all` still produces silently different behavior when two libraries choose differently (section 5.1.1). The allocator still arrives after the coroutine frame is allocated (section 5.2 below). Plain awaitables still cannot receive a stop token ([P3796R1](https://wg21.link/p3796r1)). The coroutine sugar wraps sender pipelines; it does not replace them. The template machinery underneath cannot be hidden behind `pimpl` or compiled separately. [P2079](https://wg21.link/p2079) ("System Execution Context") provides a default scheduler - a place to run sender pipelines. It does not let you write the above line of code. The problem is the programming model itself. (For a fuller treatment, see [On Universal Models](d0000-on-universal-models.md) section 3.5.1.)

There is a deeper incompatibility. [D4007R0](https://wg21.link/p4007) ("std::execution Needs More Time") documents that `std::execution`'s backward-flow context model provides the allocator after the coroutine frame is already allocated. Eric Niebler, the lead author of the framework, characterizes the issue in [P3826R3](https://wg21.link/p3826):

> "The receiver is not known during early customization. Therefore, early customization is irreparably broken."

When the lead author of a framework describes part of its design as "irreparably broken," that is evidence worth weighing. This is a fundamental incompatibility with coroutine-based I/O, where the allocator must be known at frame allocation time.

[P3796R1](https://wg21.link/p3796r1) (Kuhl, "Coroutine Task Issues") identifies another gap: "awaitable non-senders are not supported." A plain awaitable - the mechanism C++20 coroutines provide for async composition - cannot participate in the sender/receiver protocol without being wrapped as a sender. The framework that claims to serve all async use cases does not support the language's own async mechanism.

A natural objection is that coroutines lack portable heap allocation elision (HALO), making them unsuitable as a standardization foundation. This concern is less impactful for async I/O than it appears. At the end of an I/O call chain, the coroutine escapes the caller: it suspends and its handle is passed to the OS reactor (epoll, IOCP, io_uring). HALO can never apply at the I/O boundary because the coroutine inherently outlives the call that created it. The domain where HALO matters most - tight compute loops - is the domain sender/receiver already serves well. The domain where coroutines serve I/O - long-lived suspended operations waiting on the OS - is the domain where HALO is structurally inapplicable. (See [On Universal Models](d0000-on-universal-models.md) section 3.5.1a for the full argument.)

Alternative models are coroutines-first and do not have these problems. [D4003](https://wg21.link/p4003) (IoAwaitables) flows context forward through coroutine chains, with the allocator known at the launch site. [TooManyCooks](https://github.com/tzcnt/TooManyCooks) is a C++20 coroutine runtime built around `tmc::task` as the core type. [Capy](https://github.com/cppalliance/capy) is a coroutine-first execution model with forward-flowing context. Each of these designs treats coroutines as the primary user interface for async code, not as an afterthought bolted onto a different execution model. The echo server comparison in section 5.1 shows the result: a `for` loop with `co_await` and structured bindings versus 60 lines of `variant_sender` and `let_value`.

### 5.3 It Should Be Small Enough to Get Right

Butler Lampson, in his Turing Award lecture on systems design, put the principle plainly:

> "An interface should capture the minimum essentials of an abstraction. Don't generalize; generalizations are generally wrong."
>
> [Lampson, "Hints for Computer System Design"](http://research.microsoft.com/en-us/um/people/blampson/33-Hints/Acrobat.pdf) (1983)

Ted Kaminski captured the tradeoff precisely:

> "An all-powerful abstraction is a meaningless one. You've just got a new word for 'thing'."
>
> [Kaminski, "The One Ring Problem"](https://tedinski.com/2018/01/30/the-one-ring-problem-abstraction-and-power.html) (2018)

The MIT Exokernel paper put it in formal terms:

> "It is fundamentally impossible to define abstractions that are appropriate for all areas and implement them efficiently in all situations."
>
> [Engler et al., "Exokernel"](https://people.eecs.berkeley.edu/~brewer/cs262b/hotos-exokernel.pdf) (1995)

The C++ abstractions that succeed follow this pattern. Iterators abstract over traversal: "One seldom needs to know the exact type of data on which an algorithm works since most algorithms work on many similar types" ([Stepanov, "The Standard Template Library," 1994](https://stepanovpapers.com/Stepanov-The_Standard_Template_Library-1994.pdf)). Allocators abstract over memory strategy ([cppreference: Allocator](https://en.cppreference.com/w/cpp/named_req/Allocator)). RAII abstracts over resource lifetime ([cppreference: RAII](https://en.cppreference.com/w/cpp/language/raii)). Each captures one essential property and leaves everything else to the user.

`std::execution` abstracts over six concerns simultaneously. [P2300R10](https://wg21.link/p2300r10) section 1.2 lists them explicitly:

- "diversity of execution resources" (hardware backend selection)
- "cancellation" (stop token propagation)
- "error propagation" (three-channel completion model)
- "where things execute" (scheduling and context transfer)
- "manage lifetimes asynchronously" (operation-state ownership)
- algorithm dispatch via completion domains (GPU backend substitution)

That is not one essential property. That is six. Each is a separate axis of design complexity. Each generates its own companion papers. Each must be gotten right simultaneously, under ABI stability, forever.

The history of computing suggests this does not work. CORBA bundled object lifecycle, naming, transactions, security, concurrency, and interface definition into a single specification. Michi Henning, who spent years inside the effort, wrote the post-mortem:

> "Writing any nontrivial CORBA application was surprisingly difficult. Many of the APIs were complex, inconsistent, and downright arcane."
>
> [Henning, "The Rise and Fall of CORBA"](https://dl.acm.org/doi/10.1145/1142031.1142044), ACM Queue (2006)

David Chappell saw where it was heading:

> "The opportunity for a true standard, a TCP/IP for distributed objects, has been lost."
>
> [Chappell, "The Trouble With CORBA"](https://davidchappell.com/writing/article_Trouble_CORBA.php) (1998)

REST and HTTP won because they were narrow. The structural parallels between CORBA and `std::execution` are developed in detail in [On Universal Models](d0000-on-universal-models.md) section 2.1.

The specification size tells the story. The C++26 `[exec]` section ([eel.is/c++draft/exec](https://eel.is/c++draft/exec)) spans 16 major subsections (33.1 through 33.16) with thousands of lines of normative wording. By contrast, [D4003](https://wg21.link/p4003)'s IoAwaitables wording section is roughly 665 lines. D4003's own non-normative note states:

> "The wording below is not primarily intended for standardization. Its purpose is to demonstrate how a networking-focused, use-case-first design produces a dramatically leaner specification footprint. Compare this compact specification against the machinery required by P2300/P3826."

John Ousterhout captures the design principle that makes this work:

> "The best modules are deep: they have a lot of functionality hidden behind a simple interface."
>
> [Ousterhout, *A Philosophy of Software Design*](https://web.stanford.edu/~ouster/cgi-bin/aposd.php) (2018)

A narrow model designed for networking requires a fraction of the specification that a wide model designed for everything requires. This matters for implementers, for reviewers, and for the long-term maintainability of the standard. (For a fuller treatment of the narrow-vs-wide argument, see [On Universal Models](d0000-on-universal-models.md) section 4.)

The evidence from `std::execution` itself confirms the risk. [D0000](https://github.com/cppalliance/wg21-papers/blob/master/source/D0000-execution-churn.md) ("The Velocity of Change in `std::execution`") surveys the published record systematically: since approval at Tokyo in March 2024, the committee has processed 50 items - 34 papers, 11 LWG defects, and 5 national body comments - modifying a single feature in 22 months. The rate of change has accelerated, not slowed. The sender algorithm customization mechanism has been rewritten three times ([P2999R3](https://wg21.link/p2999r3), [P3303R1](https://wg21.link/p3303r1), [P3826R3](https://wg21.link/p3826)). Two Priority 1 safety defects were filed 16 months after approval. For comparison, `<ranges>` accumulated roughly 30 LWG issues in its first two years, mostly at Priority 2-3; `std::execution`'s defect count may be comparable, but the severity and the 34-paper rework volume are not (D0000 section 3). A framework with this many open design questions has not yet demonstrated the stability that standardization requires. A smaller model would have fewer questions to answer and fewer ways to get them wrong.

### 5.4 It Should Answer the Hard Questions

A standard async I/O model must take positions on error handling, cancellation, and platform abstraction. `std::execution` provides its own answers to all three. An alternative model must do the same or explain why the question does not apply. This section states the positions that the coroutines-first approach takes.

**Error handling.** Networking needs to return the error code *and* the result together. The sender model's three-channel design (`set_value` / `set_error` / `set_stopped`) forces a choice: the error goes through one channel, the result through another. A coroutine returns both in one value. The result type is variadic - `io_result<>` for `connect`, `io_result<std::size_t>` for byte transfers, `io_result<T1,T2>` for `accept` - and all specializations support structured bindings:

```cpp
auto [ec]       = co_await sock.connect(endpoint);
auto [ec, n]    = co_await sock.read_some(buf);
auto [ec, peer] = co_await acceptor.accept();
```

There is no channel classification. `await_resume` returns the full structured result as a single value, regardless of arity. [D4000](https://wg21.link/p4000) ("Where Does the Error Code Go?") documents why the sender model's three-channel alternative (`set_value` / `set_error` / `set_stopped`) is structurally unsuitable for I/O: it forces every library to choose a channel for the error code, loses partial-success information when errors go through `set_error`, and produces silently different `when_all` behavior when two libraries make different choices. The classification problem gets worse with higher-arity returns because the library author must decide which values go through which channel.

**Cancellation.** `std::stop_token` is propagated forward through the coroutine chain via an `io_env` structure passed to `await_suspend`. The stop token reaches the I/O object at the end of the chain, which cancels via the appropriate platform primitive (`CancelIoEx` on Windows, `IORING_OP_ASYNC_CANCEL` on Linux, `close()` on POSIX). Cancellation is cooperative: no operation is terminated mid-flight. Algorithms like `when_all` and `when_any` hook the caller's token to an internal stop source with fanout. [D0000](https://wg21.link/p0000) ("How Do Plain Awaitables Receive a Stop Token?") documents this mechanism and its limitations (non-standard `await_suspend` signature, protocol coupling to promise types that inject `io_env`). These are real limitations, and they are disclosed so the committee can weigh them alongside `std::execution`'s own open problems in the same area.

**Platform I/O.** The standard gives you the vocabulary types: concepts like `ReadStream`, `WriteStream`, `BufferSource`, and `BufferSink`, with type-erased wrappers like `any_read_stream` and `write_sink` ([D4003](https://wg21.link/p4003) section 3; [Capy](https://github.com/cppalliance/capy)). The ecosystem brings the platform implementation: Asio, io_uring, IOCP, kqueue. TLS is a byte stream that wraps another byte stream; the implementation (OpenSSL, BoringSSL, SChannel, SecureTransport) stays out of the standard where platform experts maintain it.

### 5.5 It Might Not Even Need Sockets!

After everything above - the priority gap, the user experience gap, the scope risk, and the answers to the hard questions - here is the surprise: the standard might not need to standardize sockets at all.

The vocabulary types from section 5.4 - `ReadStream`, `WriteStream`, and their type-erased wrappers - are enough. Algorithms written against these concepts are sharable across the ecosystem regardless of which transport satisfies them.

This is the model already demonstrated in practice.

**Boost.HTTP** follows a Sans-I/O architecture. Its documentation states: "The library itself does not perform any I/O operations or asynchronous flow control. Instead, it provides interfaces for consuming and producing buffers of data and events." It works with "any I/O framework (Asio, io_uring, platform APIs)" ([Boost.HTTP](https://github.com/cppalliance/http)).

**Boost.Capy** provides buffer-oriented stream concepts (`ReadStream`, `WriteStream`, `BufferSource`, `BufferSink`) with type-erased wrappers. Its documentation states: "It is not a networking library, yet it is the perfect foundation upon which networking libraries, or any libraries that perform I/O, may be built." Algorithms written against `any_stream` have "zero knowledge of Asio" and achieve "complete portability with no Asio dependency in algorithm code" ([Boost.Capy](https://github.com/cppalliance/capy)).

Both Boost.HTTP and Boost.Capy are still in active development. The designs are early and may change. We are showing this work before it is finished because C++26 is about to ship, and the window for reconsidering the execution model direction is closing. These libraries are presented not as finished proposals but as evidence that the approach is viable. The pattern they demonstrate, writing I/O algorithms against abstract buffer concepts independent of the transport, is what matters. The specific libraries are proof of concept, not the final answer.

The pattern is clear: standardize the buffer-oriented async I/O concepts and type-erased wrappers. Let the ecosystem provide the transports. The algorithms compose because they share the standard concepts, not because they share a specific socket implementation. This is a dramatically smaller standardization surface than either `std::execution` or full networking. And it solves the coordination problem: if everyone writes algorithms against the same buffer concepts, those algorithms interoperate regardless of the transport.

To make this concrete, the following algorithm writes an HTTP response body to any stream that satisfies the `WriteStream` concept. It has zero knowledge of the transport underneath:

```cpp
using namespace capy;
io_task<>
send_body( WriteStream auto& stream, std::string_view body )
{
    auto header = "Content-Length: " +
        std::to_string(body.size()) + "\r\n\r\n";
    auto [ec1, _] = co_await write( stream, make_buffer(header) );
    if (ec1)
      co_return ec1;
    auto [ec2, _] = co_await write( stream, make_buffer(body) );
    co_return ec2;
}
```

The same `send_body` works with an Asio TCP socket, an io_uring socket, a TLS stream, a test mock, or any transport that satisfies `WriteStream`. The standard provides the concept; the ecosystem provides the transports. This is the coordination mechanism that every other language provides and C++ lacks.

---

## 6. Conclusion

The current trajectory leaves the C++ standard in a unique position: it cannot connect to the internet services that its own standardization effort enables. The committee standardized an execution model for GPU computing. The ML/AI tower built on that computing - the most consequential computing stack of this decade - delivers its results over HTTP. Every major programming language can connect to it. C++ cannot.

Standard async I/O foundations shipped in Python, JavaScript, Go, Rust, Java, C#. Each one enabled an ecosystem of composable frameworks that C++ does not have. Not because the problem is unsolvable. Because the committee spent a decade solving a different problem.

C++20 gave us coroutines. The building blocks are in the language. The design space is not empty - IoAwaitables, Capy, and twenty years of Asio practice show that a small, coroutines-first async I/O model is achievable. The standard might not even need sockets. It needs the narrow waist: read, write, connect, close.

The C++ Standard cannot connect to the Internet. It should.

---

## References

1. Boost.Asio. https://www.boost.org/doc/libs/release/doc/html/boost_asio.html

2. Standalone Asio. https://github.com/chriskohlhoff/asio

3. folly::coro. Meta. https://github.com/facebook/folly/blob/main/folly/experimental/coro/README.md

4. Qt Signals and Slots. https://doc.qt.io/qt-6/signalsandslots.html

5. PEP 3156. "Asynchronous IO Support Rebooted: the asyncio Module." https://peps.python.org/pep-3156/

6. Django. https://github.com/django/django

7. TC39 Async/Await Proposal. https://tc39.es/proposal-async-await/

8. Express.js. https://github.com/expressjs/express

9. Next.js. https://github.com/vercel/next.js

10. Go net/http package. https://pkg.go.dev/net/http

11. Rust Blog. "Async-await on stable Rust!" 2019. https://blog.rust-lang.org/2019/11/07/Async-await-stable/

12. Tokio. https://github.com/tokio-rs/tokio

13. Axum. https://github.com/tokio-rs/axum

14. Spring Boot. https://github.com/spring-projects/spring-boot

15. C# async/await overview. https://dotnetcurry.com/csharp/869/async-await-csharp-dotnet

16. ASP.NET Core. https://github.com/dotnet/aspnetcore

17. Boost.Asio (GitHub). https://github.com/boostorg/asio

18. P2000R4. "Direction for ISO C++." WG21, 2022. https://wg21.link/p2000r4

19. nlohmann/json. https://github.com/nlohmann/json

20. cpp-httplib. https://github.com/yhirose/cpp-httplib

21. fmt. https://github.com/fmtlib/fmt

22. Lakos, John. *Large-Scale C++ Software Design.* Addison-Wesley, 1996. https://informit.com/store/large-scale-c-plus-plus-software-design-9780201633627

23. Boost.Beast. https://www.boost.org/doc/libs/release/libs/beast/

24. P3185R0. "A proposed direction for C++ Standard Networking based on IETF TAPS." WG21. https://wg21.link/p3185r0

25. Networking TS reference implementation. Kohlhoff. https://github.com/chriskohlhoff/networking-ts-impl

26. P2453R0. "2021 October Library Evolution and Concurrency Networking and Executors Poll Outcomes." WG21, 2022. https://wg21.link/p2453r0

27. P2300R10. Dominiak et al. "std::execution." WG21, 2024. https://wg21.link/p2300r10

28. stdexec. NVIDIA. https://nvidia.github.io/stdexec/

29. NVIDIA CUDA C/C++ Language Extensions. https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html

30. stdexec on vcpkg. https://vcpkg.link/ports/stdexec

31. P0912R5. "Merging Coroutines into C++20." WG21. https://wg21.link/p0912r5

32. P3552. "Add a Coroutine Task Type." WG21. https://wg21.link/p3552

33. D4007R0. Falco. "std::execution Needs More Time." WG21, 2026. https://wg21.link/p4007

34. P3826R3. Niebler. "Fix Sender Algorithm Customization." WG21, 2026. https://wg21.link/p3826

35. D4003. Falco et al. "IoAwaitables: A Coroutines-Only Execution Model." WG21. https://wg21.link/p4003

36. TooManyCooks. https://github.com/tzcnt/TooManyCooks

37. Capy. https://github.com/cppalliance/capy

38. Stepanov. "The Standard Template Library." 1994. https://stepanovpapers.com/Stepanov-The_Standard_Template_Library-1994.pdf

39. cppreference. "Allocator (named requirement)." https://en.cppreference.com/w/cpp/named_req/Allocator

40. cppreference. "RAII." https://en.cppreference.com/w/cpp/language/raii

41. C++26 Working Draft, [exec] section. https://eel.is/c++draft/exec

42. Boost.HTTP. https://github.com/cppalliance/http

43. senders-io. Nadolski. https://github.com/maikel/senders-io

44. Corosio. https://github.com/cppalliance/corosio

45. D4000. Falco & Gill. "Where Does the Error Code Go?" WG21, 2026. https://wg21.link/p4000

46. D0000. Falco. "How Do Plain Awaitables Receive a Stop Token?" WG21, 2026. https://wg21.link/p0000

47. D0000. Falco. "On Universal Models." WG21, 2026.

48. libunifex issue #244. "Question about any_sender_of usage." https://github.com/facebookexperimental/libunifex/issues/244

49. P3796R1. Kuhl. "Coroutine Task Issues." WG21. https://wg21.link/p3796r1

50. P2430R0. Kohlhoff. "Partial success scenarios with P2300." WG21, 2021. https://wg21.link/p2430r0

51. Butler Lampson. "Hints for Computer System Design." 1983. http://research.microsoft.com/en-us/um/people/blampson/33-Hints/Acrobat.pdf

52. Ted Kaminski. "The One Ring Problem." 2018. https://tedinski.com/2018/01/30/the-one-ring-problem-abstraction-and-power.html

53. Engler et al. "Exokernel: An Operating System Architecture for Application-Level Resource Management." MIT, 1995. https://people.eecs.berkeley.edu/~brewer/cs262b/hotos-exokernel.pdf

54. Michi Henning. "The Rise and Fall of CORBA." ACM Queue 4, no. 5 (June 2006). https://dl.acm.org/doi/10.1145/1142031.1142044

55. David Chappell. "The Trouble With CORBA." 1998. https://davidchappell.com/writing/article_Trouble_CORBA.php

56. John Ousterhout. *A Philosophy of Software Design.* Yaknyam Press, 2018. https://web.stanford.edu/~ouster/cgi-bin/aposd.php

57. P2999R3. Niebler. "Sender Algorithm Customization." WG21, 2023. https://wg21.link/p2999r3

58. P3303R1. Niebler. "Fixing Lazy Sender Algorithm Customization." WG21, 2024. https://wg21.link/p3303r1

59. NVIDIA/stdexec#1062. "io_uring reading files." https://github.com/NVIDIA/stdexec/issues/1062

60. stdexec io_uring.cpp. https://github.com/NVIDIA/stdexec/blob/main/examples/io_uring.cpp

61. libunifex. Facebook Experimental. https://github.com/facebookexperimental/libunifex

62. P2079. "System Execution Context." WG21. https://wg21.link/p2079

63. PEP 492. "Coroutines with async and await syntax." 2015. https://peps.python.org/pep-0492/

64. Kotlin Coroutines. https://kotlinlang.org/docs/coroutines-overview.html

65. Python standard library: socket. https://docs.python.org/3/library/socket.html

66. Java standard library: java.net. https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/net/package-summary.html

67. Go standard library: net. https://pkg.go.dev/net

68. Rust standard library: std::net. https://doc.rust-lang.org/std/net/

69. C# standard library: System.Net.Sockets. https://learn.microsoft.com/en-us/dotnet/api/system.net.sockets

70. Node.js standard library: net. https://nodejs.org/api/net.html

71. D0000. Falco. "The Velocity of Change in std::execution." WG21, 2026. https://github.com/cppalliance/wg21-papers/blob/master/source/D0000-execution-churn.md
