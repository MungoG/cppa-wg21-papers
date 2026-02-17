---
title: "Is the Sender Sub-Language C++?"
document: D4007R1
date: 2026-02-16
reply-to:
  - "Vinnie Falco <vinnie.falco@gmail.com>"
  - "Mungo Gill <mungo.gill@me.com>"
audience: SG1, LEWG
---

## Abstract

With `std::execution` ([P2300R10](https://wg21.link/p2300r10)), the committee has adopted the Sender Sub-Language, a programming model for asynchronous computation rooted in [continuation-passing style](https://en.wikipedia.org/wiki/Continuation-passing_style) (CPS), with its own control flow, variable binding, error handling, and type system ([D4014R0](https://wg21.link/d4014)). This is a significant architectural decision with real value for the domains it serves. Programming language theory establishes that CPS and direct-style are structurally incompatible models of computation. This paper maps the consequences of that decision for coroutines. It predicts friction wherever the Sender Sub-Language interfaces with regular C++, then demonstrates it with three structural gaps in the coroutine integration and a broader survey of eight friction points in the Sub-Language's own type system, semantics, and specification. The gaps are not design defects. They are structural consequences of the committee's decision to adopt a CPS-based model. Understanding them is a prerequisite for deciding how coroutines should relate to `std::execution` going forward.

---

## 2. Two Models of Computation

Coroutines let developers write asynchronous code that reads like sequential code:

```cpp
task<void> handle_request(tcp::socket sock)
{
    auto [ec, n] = co_await sock.async_read(buf);
    if (ec) co_return;
    auto doc = co_await parse_json(buf);
    co_await sock.async_write(build_response(doc));
}
```

Eric Niebler wrote in ["Structured Concurrency"](https://ericniebler.com/2020/11/08/structured-concurrency/) (2020): *"I think that 90% of all async code in the future should be coroutines simply for maintainability."*

The Sender Sub-Language expresses the same logic differently:

```cpp
auto sndr = just(std::move(socket))                       // pure/return
          | let_value([](tcp_socket& s) {                 // monadic bind (>>=)
                return async_read(s, buf)                 // Kleisli arrow
                     | then([](auto data) {               // fmap/functor lift
                           return parse(data);            // VALUE -> success path
                       });
            })
          | upon_error([](auto e) {                       // ERROR -> error path
                log(e);
            })
          | upon_stopped([] {                             // STOPPED -> cancellation path
                log("cancelled");
            });
auto op = connect(std::move(sndr), rcvr);                 // reify continuation
start(op);                                                // begin execution
```

Niebler [asked the question himself](https://ericniebler.com/2020/11/08/structured-concurrency/): *"Why would anybody write that when we have coroutines? You would certainly need a good reason."*

Herb Sutter, Technical Fellow at Citadel Securities, called `std::execution` *["the biggest usability improvement yet to use the coroutine support we already have."](https://herbsutter.com/2024/07/02/trip-report-summer-iso-c-standards-meeting-st-louis-mo-usa)*

The annotations in the sender pipeline reveal what the committee brought into the standard. [`just`](https://hackage.haskell.org/package/base/docs/Data-Maybe.html) is Haskell's `return`/`pure`, lifting a value into a monadic context. [`let_value`](https://hackage.haskell.org/package/base/docs/Control-Monad.html#v:-62--62--61-) is [monadic bind](https://hackage.haskell.org/package/base/docs/Control-Monad.html#v:-62--62--61-) (`>>=`), threading a value into the next computation. `then` is [fmap](https://hackage.haskell.org/package/base/docs/Data-Functor.html), applying a function inside a context. The three completion channels are a fixed [algebraic effect system](https://en.wikipedia.org/wiki/Effect_system). `connect` reifies the continuation into a concrete operation state. This is [continuation-passing style](https://en.wikipedia.org/wiki/Continuation-passing_style) expressed as composable value types. [D4014R0](https://wg21.link/d4014) ("The Sender Sub-Language") provides the full treatment.

Coroutines inhabit direct-style C++, code where values return to callers, errors propagate through the call stack, and resources are scoped to lexical lifetimes. This paper will call that *regular C++* for the remainder.

Niebler [described](https://ericniebler.com/2020/11/08/structured-concurrency/) what regular C++ means for async programming: *"We sprinkle `co_await` in our code and we get to continue using all our familiar idioms: exceptions for error handling, state in local variables, destructors for releasing resources, arguments passed by value or by reference, and all the other hallmarks of good, safe, and idiomatic Modern C++."*

The Sender Sub-Language provides compile-time work graph construction, zero-allocation pipelines, and vendor extensibility. Coroutines provide the programming model that most developers already know. The committee chose to standardize a CPS-based framework alongside an existing direct-style language feature. This paper examines what follows from that choice.

Structured bindings, `if` with initializer, range-for, the direct-style features the committee has invested in over the last decade, have no purchase on a sender pipeline. Values flow forward into continuations as arguments, not backward to callers as returns. There is no aggregate to destructure at intermediate stages.

Niebler [characterized](https://ericniebler.com/2020/11/08/structured-concurrency/) the trade-off: *"That style of programming makes a different tradeoff, however: it is far harder to write and read than the equivalent coroutine."*

---

## 3. What the Theory Predicts

By adopting the Sender Sub-Language, the committee placed a CPS-based model alongside an imperative language. Programming language theory can predict what follows. The incompatibility between CPS and direct-style is not a conjecture. It is a result with fifty years of literature behind it.

**Danvy, ["Back to Direct Style"](https://static.aminer.org/pdf/PDF/001/056/774/back_to_direct_style.pdf) (1992):** *"Not all lambda-terms are CPS terms, and not all CPS terms encode a left-to-right call-by-value evaluation."*

The CPS transform is asymmetric. Consider `when_all`: it dispatches on which channel fired (`set_value` vs `set_error`) at the type level, cancelling siblings on error without inspecting the payload. A coroutine has no way to express "cancel sibling operations based on the type-level path the result took." The Sub-Language can express things that have no regular C++ equivalent. The transform goes one way.

**Plotkin, ["Call-by-Name, Call-by-Value and the lambda-Calculus"](https://homepages.inf.ed.ac.uk/gdp/publications/cbn_cbv_lambda.pdf) (1975):** *"Operational equality is not preserved by either of the simulations."*

You can simulate one model in the other, but the simulation changes program behavior. `task<T>` is that simulation. A coroutine that returns `std::expected<size_t, error_code>` through `co_return` is operationally different from a sender that routes `error_code` through `set_error`, even though both describe "return the byte count or the error." The first is invisible to `upon_error`. The second loses the byte count. Same intent, different operational behavior.

The simulation works. It just does not preserve what the original meant.

**Strachey & Wadsworth, ["Continuations: A Mathematical Semantics for Handling Full Jumps"](https://www.cs.ox.ac.uk/publications/publication3729-abstract.html) (1974), as cited in later transformation work:** *"Transforming the representation of a direct-style semantics into continuation style usually does not yield the expected representation of a continuation-style semantics (i.e., one written by hand)."*

A hand-written sender operation allocates inside `connect()`, after the receiver's environment (including the allocator) is available. A coroutine's `promise_type::operator new` fires at the function call, before any sender machinery runs. The same "allocate a frame" operation, represented in two models, executes at structurally different points in time.

Niebler wrote in 2024: *["If your library exposes asynchrony, then returning a sender is a great choice: your users can await the sender in a coroutine if they like."](https://ericniebler.com/2024/02/04/what-are-senders-good-for-anyway/)* The phrase "if they like" implies this is straightforward. Plotkin's result says it is not.

Bob Nystrom gave the practitioner's version in ["What Color is Your Function?"](https://journal.stuffwithstuff.com/2015/02/01/what-color-is-your-function/) (2015): *"You still have divided your entire world into asynchronous and synchronous halves and all of the misery that entails."* The Sub-Language is the "red" (CPS) world. Regular C++ is the "blue" world.

C++ now contains a CPS sub-language. The committee put it there. Wherever the two meet, we can expect friction. Coroutines are the first and most visible boundary, but the incompatibility is between CPS and regular C++ itself. Any C++ construct that assumes values return to callers, errors propagate through the call stack, or resources are scoped to lexical lifetimes will encounter the same structural mismatch. The coroutine boundary is the first test of this prediction. The three gaps documented in this paper are three instances.

One might point to Haskell's `do`-notation as a successful bridge between CPS and direct-style. The bridge works because Haskell is pure, with no "direct-style with side effects" to conflict with, and because `do`-notation is a language feature designed for this purpose. C++ coroutines are not `do`-notation. They are an independent language feature with their own semantics that predate senders.

C++ is unique among languages that have attempted this bridge. JavaScript is dynamically typed and garbage collected; its async/await is syntactic sugar with no type-level routing and no lifetime hazards. Rust is strongly typed and memory safe; the borrow checker enforces lifetime safety across the async boundary, at the cost of `Pin` complexity. C# is strongly typed and garbage collected; `Task<T>` has no dangling reference problem. C++ is the only language that is simultaneously strongly typed enough to support a genuine CPS sub-language with compile-time routing and memory unsafe enough that the boundary between the two models creates lifetime hazards that neither garbage collection nor a borrow checker can resolve. [LWG 4368](https://cplusplus.github.io/LWG/issue4368) (dangling reference from `transform_sender`) is an example of exactly this hazard.

**What happens if we try?**

---

## 4. Where Does the Error Code Go?

The Sub-Language the committee adopted has three completion channels: `set_value`, `set_error`, and `set_stopped`. These channels are its type system. Type-level routing is a structural requirement of CPS reification (Appendix A.4). This section examines what happens when that type system meets runtime I/O completion semantics.

[P2300R10](https://wg21.link/p2300r10) provides three completion channels. Only one can be called for a given operation. The authors describe the model in [Section 1.3.1](https://open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html):

> *"A sender describes asynchronous work and sends a signal (value, error, or stopped) to some recipient(s) when that work completes."*

### 4.1 Senders

Every I/O operation in [Boost.Asio](https://www.boost.org/doc/libs/release/doc/html/boost_asio.html) completes with `void(error_code, size_t)`, an error code and a byte count, delivered together. A `read` may transfer 47 bytes and then encounter EOF. A `write` may transfer 1000 bytes out of 4096 before the connection resets. The error code and the byte count are not alternatives. They are returned together, because partial success is the normal case in I/O. [P2762R2](https://wg21.link/p2762r2) ("Sender/Receiver Interface for Networking") preserves this pattern.

A developer implementing an async read sender must choose. The natural instinct is to route the error code through `set_error`:

```cpp
if (!ec)
    set_value(std::move(rcvr_), n);
else
    set_error(std::move(rcvr_), ec); // bytes lost
```

But this loses the byte count on error. A `read` that transferred 47 bytes before EOF is now indistinguishable from a `read` that transferred zero.

The alternative is to put the error code into `set_value`, delivering both values together as [P2762R2](https://wg21.link/p2762r2) does:

```cpp
set_value(std::move(rcvr_), ec, n);
```

The `set_value` approach preserves the byte count. It matches twenty years of Asio practice. But now the developer is reporting an error through the success channel. The name says "value." They are sending an error code. Chris Kohlhoff identified this tension in 2021 in [P2430R0](https://open-std.org/jtc1/sc22/wg21/docs/papers/2021/p2430r0.pdf) ("Partial success scenarios with P2300"):

> *"Due to the limitations of the set_error channel (which has a single 'error' argument) and set_done channel (which takes no arguments), partial results must be communicated down the set_value channel."*

The naming is misleading. `set_error` looks right for an `error_code`, and for non-I/O senders it probably is. But for I/O, the natural choice silently misbehaves. The developer who follows the API's own naming convention produces code that silently misbehaves under composition. The developer who does the right thing must override that naming convention and put an error into the value channel.

### 4.2 Coroutines

The same forced choice reaches coroutine authors through `std::execution::task`. [P3552R3](https://wg21.link/p3552r3) ("Add a Coroutine Task Type") provides two mechanisms for completing a `task`:

- `co_return value` routes to `set_value` on the receiver
- `co_yield with_error(e)` suspends the coroutine and calls `set_error` on the receiver

An I/O coroutine author must choose between three return types, each with consequences:

**Option A: Return both values together.**

```cpp
std::execution::task<std::pair<std::error_code, std::size_t>>
do_read(tcp_socket& s, buffer& buf)
{
    auto [ec, n] = co_await s.async_read(buf);
    co_return {ec, n};
}
```

The error is invisible to `when_all`, `upon_error`, `let_error`, and every generic error-handling algorithm in `std::execution`.

**Option B: Signal the error through the error channel.**

```cpp
std::execution::task<std::size_t>
do_read(tcp_socket& s, buffer& buf)
{
    auto [ec, n] = co_await s.async_read(buf);
    if (ec)
        co_yield with_error(ec);                      // bytes lost
    co_return n;
}
```

The byte count is lost when `co_yield with_error(ec)` fires.

**Option C: Return only the byte count.**

```cpp
std::execution::task<std::size_t>
do_read(tcp_socket& s, buffer& buf)
{
    auto [ec, n] = co_await s.async_read(buf);
    co_return n;                                      // error silently discarded
}
```

None of these options is correct. The developer chooses which kind of wrong they prefer.

### 4.3 `when_all` Is Broken for I/O

[`when_all`](https://eel.is/c++draft/exec.when.all) launches multiple child operations concurrently. If any child completes with `set_error` or `set_stopped`, `when_all` requests cancellation of the remaining children. The channel the sender author chose now determines the behavior of every algorithm that consumes it:

```cpp
auto result = co_await when_all(
    socket_a.async_read(buf_a),
    socket_b.async_read(buf_b));
```

If the sender author chose `set_error` for EOF, `when_all` cancels `socket_b` and propagates the error. The 47 bytes already transferred are discarded.

If the sender author chose `set_value` for EOF, `when_all` treats the error as success. The sibling is not cancelled. A failure goes undetected.

When two libraries choose different conventions and are composed in the same `when_all`, the outcome depends on completion order. Same two EOF events. Same `when_all`. The program is correct on some runs and incorrect on others.

### 4.4 The Problem Is Structural

The mismatch is not a missing convention waiting to be supplied. It is a structural incompatibility between the Sub-Language's three-channel type system and I/O completion semantics. No convention, no adaptor, and no additional algorithm can resolve it without changing the model:

- **Converge on `set_value` for error codes.** Every generic error-handling algorithm in the Sub-Language is inaccessible to I/O senders. The error channel exists but I/O never uses it.

- **Converge on `set_error` for error codes.** Partial success is inexpressible. The byte count is lost.

- **Write an adaptor.** The adaptor must know which convention the inner sender follows. This reintroduces the forced choice one level up.

- **Add a fourth channel.** This would be a breaking change to the completion signature model.

The three-channel model assumes a clean separation: values are values, errors are errors. I/O does not have a clean separation. An `error_code` is a status report. EOF is not a failure. A partial write is not an error. The model and the domain are structurally incompatible. Appendix A.4 explains why: the three channels are a structural requirement of the compile-time work graph, and eliminating them would collapse the Sub-Language's type-level routing into runtime dispatch.

### 4.5 Always Use `set_value`?

One might argue that the channels are generic and I/O should simply adopt the `set_value` convention.

But `when_all` still breaks: when every I/O sender delivers `(error_code, size_t)` through `set_value`, I/O failure is indistinguishable from I/O success. Sibling cancellation on failure is impossible.

The completion signature `void(error_code, size_t)` predates senders by 25 years. It is not a quirk of one library. It is the completion signature of every I/O operation in POSIX, Win32, Asio, and every networking library built on them. The Sub-Language's three-channel model is optimized for type-level routing in sender pipelines (Appendix A.4 examines the design rationale). The friction with I/O completion semantics is a consequence of that optimization.

Kohlhoff identified the channel routing tension in [P2430R0](https://open-std.org/jtc1/sc22/wg21/docs/papers/2021/p2430r0.pdf) (2021). This paper extends the analysis to composition consequences. The `when_all` analysis: regardless of which convention a sender author chooses, correct behavior under `when_all` is impossible by design. When two libraries choose differently, the result is a race condition over correctness determined by completion order.

---

## 5. Where Is the `co_return`?

The Sub-Language requires errors to reach a separate channel. That requirement comes from the committee's decision to standardize type-level routing. Regular C++ coroutines have no native path to a second channel. `co_yield with_error` is one model of computation being asked to express a concept that belongs to the other.

### 5.1 Senders

In a sender's operation state, signaling an error is one line:

```cpp
set_error(std::move(rcvr), ec);
```

The call is direct. The name matches the intent.

### 5.2 Coroutines

Returning errors from a coroutine is customarily accomplished through `co_return`:

```cpp
my_task<std::expected<std::size_t, std::error_code>>
do_read(tcp_socket& s, buffer& buf)
{
    auto [ec, n] = co_await s.async_read(buf);
    if (ec)
        co_return std::unexpected(ec);
    co_return n;
}
```

Yet [P3552R3](https://wg21.link/p3552r3) innovates:

```cpp
std::execution::task<std::size_t>
do_read(tcp_socket& s, buffer& buf)
{
    auto [ec, n] = co_await s.async_read(buf);
    if (ec)
        co_yield with_error(ec);           // not co_return
        // never gets here?
    co_return n;
}
```

For six years, every C++ coroutine library has taught the same two conventions: `co_return` for final values, `co_yield` for intermediate values that the coroutine continues to produce. The second example above breaks both: `co_yield` does not produce a value, and the coroutine does not continue.

### 5.3 Established Practice

No production C++ coroutine library uses `co_yield` for error signaling:

- **cppcoro** (Lewis Baker): errors are exceptions or values. `co_return` delivers both.
- **folly::coro::Task** (Facebook/Meta): errors are exceptions, propagated automatically through the coroutine chain via `co_return` and `co_await`.
- **Boost.Cobalt** (Klemens Morgenstern): errors are exceptions or values. `co_return` delivers both.
- **Boost.Asio awaitable** (Chris Kohlhoff): `co_return` for values; errors delivered via `as_tuple(use_awaitable)` as return values.
- **libcoro** (Josh Baldwin): errors are exceptions or values. `co_return` delivers both.

There is no third path. `std::execution::task` introduces one.

The Sub-Language requires errors to reach the receiver through `set_error`. The C++ language specification defines `co_return expr` as calling `promise.return_value(expr)`, and P3552R3's `return_value` routes to `set_value`. There is no way to make `co_return` call `set_error`. The reason is a language constraint: a coroutine promise can only define `return_void` or `return_value`, but not both. Since `std::execution::task<void>` needs `co_return;` (which requires `return_void`), a `task<void>` coroutine cannot also have `return_value`.

P3552R3 needed a different mechanism. The only other coroutine keyword that accepts an expression and passes it to the promise is `co_yield`. P3552R3 exploits `yield_value` with a special overload for `with_error<E>`.

The committee is aware of this problem. Jonathan Wakely's [P3801R0](https://wg21.link/p3801r0) ("Concerns about the design of `std::execution::task`," 2025) explains:

> *"The reason `co_yield` is used, is that a coroutine promise can only specify `return_void` or `return_value`, but not both. If we want to allow `co_return;`, we cannot have `co_return with_error(error_code);`. This is unfortunate, but could be fixed by changing the language to drop that restriction."*

The proposed fix, a language change to the `return_void`/`return_value` mutual exclusion, is not part of C++26 and has no paper proposing it. The syntax ships as-is.

One might argue that this is a flaw in the coroutine language feature, not in `std::execution`. But the `return_void`/`return_value` mutual exclusion has existed since C++20 and has not prevented any other coroutine library from delivering errors through `co_return`. The constraint becomes a problem only when the framework requires errors to reach a separate channel, a requirement that originates in the Sub-Language, not in coroutines.

`co_yield with_error(e)` actively misleads. A developer who has learned what `co_yield` means, produce a value and continue, will read `co_yield with_error(ec)` and conclude that the coroutine continues after the yield point. It does not. The keyword's established meaning predicts the wrong behavior.

---

## 6. Where Is the Allocator?

In the Sub-Language the committee standardized, the `connect`/`start` reification pattern binds the continuation late, after the coroutine frame is already allocated. The receiver's environment, including the allocator, is not available until after the frame exists.

Eric Niebler wrote in 2021: *["The overwhelming benefit of coroutines in C++ is its ability to make your async scopes line up with lexical scopes."](https://ericniebler.com/2021/08/29/asynchronous-stacks-and-scopes/)* The allocator sequencing gap is the cost of misaligning those scopes: the frame is allocated at the lexical scope, but the allocator arrives at `connect` time.

Niebler [identified the cost](https://ericniebler.com/2020/11/08/structured-concurrency/) in 2020:

> *"With coroutines, you have an allocation when a coroutine is first called, and an indirect function call each time it is resumed. The compiler can sometimes eliminate that overhead, but sometimes not."*

Each `task<T>` call invokes the coroutine promise's `operator new` (`promise_type::operator new`). In a server handling thousands of connections at high request rates, frame allocations can reach the order of millions per second. A recycling allocator tuned for coroutine frames can outperform even state-of-the-art general-purpose allocators. [P4003R0](https://wg21.link/p4003r0) ("IoAwaitables: A Coroutines-Only Framework") benchmarks a 4-deep coroutine call chain (2 million iterations):

| Platform    | Allocator        | Time (ms) | vs std::allocator |
|-------------|------------------|----------:|------------------:|
| MSVC        | Recycling        |   1265.2  |          +210.4%  |
| MSVC        | mimalloc         |   1622.2  |          +142.1%  |
| MSVC        | `std::allocator` |   3926.9  |                 - |
| Apple clang | Recycling        |   2297.08 |           +55.2%  |
| Apple clang | `std::allocator` |   3565.49 |                 - |

### 6.1 Senders

The [P2300](https://wg21.link/p2300) authors solved the allocator propagation problem for senders with care and precision. The receiver's environment carries the allocator via `get_allocator(get_env(rcvr))`, and the sender algorithms propagate it automatically through every level of nesting:

```cpp
auto work =
    just(std::move(socket))                           // pure/return
  | let_value([](tcp_socket& s) {                     // monadic bind
        return async_read(s, buf)                     // Kleisli arrow
             | let_value([&](auto data) {             // nested bind
                   return parse(data)
                        | let_value([&](auto doc) {   // third level bind
                              return async_write(
                                  s, build_response(doc));
                          });
               });
    });

recycling_allocator<> alloc;                          // set once here
auto op = connect(
    write_env(std::move(work),
        prop(get_allocator, alloc)),
    rcvr);
start(op);
```

The allocator is set once at the launch site and reaches every operation in the tree without any intermediate sender mentioning it. This is genuinely good design.

However...

Nothing here allocates. The standard sender algorithms compose as value types. The operation state produced by `connect` is a single concrete type, a deeply nested template instantiation with no heap allocation. The allocator propagates elegantly through a pipeline that does not allocate.

### 6.2 Coroutines

Consider the standard usage pattern for spawning a connection handler:

```cpp
namespace ex = std::execution;

ex::counting_scope scope;

ex::spawn(
    ex::on(sch, handle_connection(std::move(conn))),
    scope.get_token());
```

Two independent problems prevent the allocator from reaching the coroutine frame:

**First, there is no API.** Neither `spawn` nor `on` accept an allocator parameter. The sender algorithms that launch coroutines have no mechanism to forward an allocator into the coroutine's `operator new`.

**Second, even if there were, the timing is wrong.** The expression `handle_connection(std::move(conn))` is a function call. The compiler evaluates it, including `promise_type::operator new`, before `spawn` or `on` execute. The frame is already allocated by the time any sender algorithm runs.

The only way to get an allocator into the coroutine frame is through the coroutine's own parameter list:

```cpp
// What users expect to write:
std::execution::task<>
handle_connection( tcp_socket conn );

// What they must actually write:
template<class Allocator>
std::execution::task<>
handle_connection( std::allocator_arg_t, Allocator alloc, tcp_socket conn );
```

The environment propagation that the P2300 authors designed so carefully is structurally unreachable for coroutine frame allocation.

**Senders get the allocator they do not need. Coroutines need the allocator they do not get.**

### 6.3 Coroutines Work for What Senders Get Free

[P3552R3](https://wg21.link/p3552r3) provides `std::allocator_arg_t` as a creation-time mechanism: the caller passes the allocator explicitly at the call site, and `promise_type::operator new` can use it. This solves the initial allocation.

Propagation remains unsolved. When a coroutine calls a child coroutine, the child's `operator new` fires during the function-call expression, before the parent's promise has any opportunity to intervene. The only workaround is manual forwarding. Three properties distinguish this from a minor inconvenience:

1. **Composability loss.** Generic Sub-Language algorithms like `let_value` and `when_all` launch child operations without knowledge of the caller's allocator. Manual forwarding cannot cross algorithm boundaries.

2. **Silent fallback.** Omitting `allocator_arg` from one call does not produce a compile error. The child silently falls back to the default heap allocator with no diagnostic.

3. **Protocol asymmetry.** Schedulers and stop tokens propagate automatically through the receiver environment. Allocators are the only execution resource that the Sub-Language forces coroutine users to propagate by hand.

The pattern has a recognized name in software engineering: [tramp data](https://softwareengineering.stackexchange.com/questions/335005), parameters passed through intermediate functions that do not use them. The React ecosystem calls it [prop drilling](https://kentcdodds.com/blog/prop-drilling). The principle is familiar to C++ developers too. [C++ Core Guidelines F.7](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines.html#Rf-smart): a function that uses an object should not be coupled to the caller's ownership or lifetime policy.

Senders receive the allocator through the environment, automatically, at every level of nesting, with no signature pollution. Coroutines receive it through `allocator_arg`, manually, at every call site, with silent fallback on any mistake. The same framework, the same resource, two completely different levels of support.

### 6.4 A Viral Signature?

Here is what allocator propagation looks like in a coroutine call chain under [P3552R3](https://wg21.link/p3552r3):

```cpp
template<typename Allocator>
task<void> level_three(
    std::allocator_arg_t = {},
    Allocator alloc = {})
{
    co_return;
}

template<typename Allocator>
task<void> level_two(
    int x,
    std::allocator_arg_t = {},
    Allocator alloc = {})
{
    co_await level_three(std::allocator_arg, alloc);
}

template<typename Allocator>
task<int> level_one(
    int v,
    std::allocator_arg_t = {},
    Allocator alloc = {})
{
    co_await level_two(42, std::allocator_arg, alloc);
    co_return v;
}
```

Every function signature carries allocator forwarding machinery that has nothing to do with the function's purpose.

### 6.5 Domain Freedom?

No workaround, global PMR, thread-local registries, or otherwise, can bypass the promise. Every allocated coroutine frame must run through `promise_type::operator new`. If the promise does not cooperate, the allocator does not reach the frame.

The escape hatch is to stop searching for a universal allocator model in one promise type. Let each domain's task type cooperate with its own allocator strategy. A networking task type can use thread-local propagation. A GPU task type can use device memory APIs. Solutions exist when the promise is free to serve its domain rather than forced to serve a Sub-Language it does not participate in.

---

## 7. Friction Beyond Coroutines

The three gaps in Sections 4-6 demonstrate friction at the coroutine boundary. This section broadens the lens. The same CPS/regular-C++ mismatch generates friction within the Sub-Language's own machinery and at every boundary with regular C++, not only coroutines.

The following friction points were identified through a survey of active papers and LWG issues, but not analyzed to the depth of the three gaps above. The authors present them as data points, not conclusions. Each appears consistent with the thesis that CPS and regular C++ produce friction at their boundary, but the committee is better positioned to judge whether these reflect normal end-of-cycle specification polish or structural consequences of the adopted model. Large features generate proportionate review activity. The question is whether the classification pattern, every item clustering at the boundary between two models of computation, is better explained by scale or by structure.

### 7.1 Type-System Friction

**Algorithm customization broken ([P3826R3](https://wg21.link/p3826r3), [P3718R0](https://wg21.link/p3718r0), [P2999R3](https://wg21.link/p2999r3), [P3303R1](https://wg21.link/p3303r1)).** The Sub-Language's CPS reification sequence means a sender cannot know its completion domain until it knows where it will start. Early customization is, in Eric Niebler's words, "irreparably broken." `starts_on(gpu, just()) | then(fn)` silently uses the CPU implementation for GPU work. Four papers and counting. In regular C++, calling `f(x)` on a thread pool just works.

**Incomprehensible diagnostics ([P3557R3](https://wg21.link/p3557r3), [P3164R4](https://wg21.link/p3164r4)).** Type checking deferred to `connect` time means `just(42) | then([](){})` (nullary lambda, wrong arity) produces dozens of lines of template backtrace far from the source. In regular C++, `g(f(42))` with a type mismatch produces a one-line error at the call site. The deferral is a structural consequence of CPS reification: the pipeline must be fully assembled before types are checked.

**`connect_result_t` SFINAE breakage ([LWG 4206](https://cplusplus.github.io/LWG/issue4206), Priority 1).** Changing `connect`'s constraints to `Mandates` made `connect` unconstrained, but `let_value` uses `connect_result_t` in SFINAE contexts expecting substitution failure. It gets hard errors instead. The Sub-Language's layered constraint model violates normal template metaprogramming conventions.

### 7.2 Semantic Friction

**`split` and `ensure_started` removed ([P3682R0](https://wg21.link/p3682r0), [P3187R1](https://wg21.link/p3187r1)).** The two most natural ways to share or reuse an async result violate structured concurrency. Both removed by plenary vote. In regular C++, `auto x = f(); use(x); use(x);` is trivial. The Sub-Language's structured concurrency guarantees, a consequence of CPS reification requiring stable operation state lifetimes, make this pattern fundamentally unsafe.

**Optional vs. variant interface mismatch ([P3570R2](https://wg21.link/p3570r2)).** A concurrent queue's `async_pop` naturally returns `optional<T>` in regular C++ ("value or nothing"). The Sub-Language requires the same operation to complete through `set_value(T)` or `set_stopped()`, two channels dispatched at the type level. API designers must choose which world to serve.

**Partial success in pure sender context ([P2430R0](https://wg21.link/p2430r0)).** This is the same error channel mismatch documented in Section 4, viewed from the sender side rather than the coroutine side. The forced choice between `set_value(ec, n)` and `set_error(ec)` exists for raw sender authors writing operation states, independent of coroutines. We include it here to show that the friction is not specific to the coroutine boundary.

### 7.3 Specification Friction

**`transform_sender` dangling reference ([LWG 4368](https://cplusplus.github.io/LWG/issue4368), Priority 1).** The specification itself has a use-after-free. The layered transform architecture creates a lifetime hazard: `default_domain::transform_sender` forwards a temporary as an xvalue after the temporary is destroyed. The reference implementation (stdexec) works around it by non-conformantly returning prvalues. This hazard does not exist in direct function calls.

**Circular `completion-signatures-for` ([LWG 4190](https://cplusplus.github.io/LWG/issue4190), Priority 2).** The spec for `completion-signatures-for<Sndr, Env>` tests `sender_in<Sndr, Env>`, which requires `get_completion_signatures(sndr, env)` to be well-formed, the very thing being defined. Circular specifications are a symptom of the type-level machinery's self-referential complexity.

If the committee determines that these items are routine specification work unrelated to the model of computation, that finding does not affect the three structural gaps documented in Sections 4-6. Those gaps stand on their own evidence. The question this section poses is narrower: does the pattern, eight items from three independent categories, each clustering at the boundary between CPS and regular C++, suggest a structural cause, or is it coincidence? The authors leave the classification to the committee.

---

## 8. The Gaps Cannot Be Fixed Later

The Sub-Language entered the standard for specific properties: compile-time routing, zero-allocation reification, type-level dispatch. Fixing the gaps would require removing those properties. The gaps are not oversights in the integration. They are the Sub-Language doing what it was designed to do.

### 8.1 `await_transform` Cannot Help

In `co_await child_coro(args...)`, the function call `child_coro(args...)` is evaluated first. The child's `promise_type::operator new` fires and the frame is allocated before `co_await` processing begins. By the time `await_transform` sees the returned `task<T>`, the child's frame has already been allocated without the parent's allocator.

### 8.2 No Path to the Promise's `operator new`

[P3552R3](https://wg21.link/p3552r3) establishes a two-tier model: the allocator is a creation-time concern passed at the call site, while the scheduler and stop token are connection-time concerns from the receiver. Propagating the allocator through a chain of nested coroutine calls remains the harder and unsolved problem. With standard containers, the allocator type is part of the type signature (`vector<T, Alloc>`), and `uses_allocator` construction gives generic code a standard way to propagate. With coroutines, the return type is `task<T>`. It does not carry the allocator type, and there is no `uses_allocator` equivalent for coroutine frame allocation.

### 8.3 P3826R3 and the Allocator Sequencing Gap

[P3826R3](https://wg21.link/p3826r3) ("Fix Sender Algorithm Customization") addresses sender algorithm customization, an important problem worth solving. We include it here not to criticize its scope but to clarify that its solutions do not change when the allocator becomes available to coroutines. P3826 offers five solutions. All target algorithm dispatch. Four of the five do not change when the allocator becomes available. The fifth, remove all `std::execution`, resolves the gap by deferral. See Appendix A.3 for an analysis of each solution.

### 8.4 ABI Lock-In

Once standardized, the relationship between the promise's `operator new` and `connect()` becomes part of the ABI. The standard does not break ABI on standardized interfaces. A fix would likely need `connect()` to propagate allocator context before the coroutine frame is allocated, a structural change to the sender protocol.

### 8.5 The Three-Channel Model Is the ABI

The three-channel completion model is the foundation of the Sub-Language's completion signature machinery. Every sender algorithm's behavior is defined in terms of which channel fires. Adding a fourth channel, changing `when_all` semantics, or redefining the relationship between error codes and channels would be a breaking change to the model. The channel routing problem identified by Kohlhoff in 2021 has no resolution within the current model, and the ABI freeze would make the model permanent.

The natural compromise, ship `task` with known limitations and fix via DR or C++29 addendum, assumes the fix is a minor adjustment. Sections 8.1 through 8.4 show it is not. The allocator sequencing gap requires changing the relationship between the promise's `operator new` and `connect()`. The channel routing gap requires changing the completion model. Both are structural, and both become ABI once shipped. A DR that changes ABI is not a DR. It is a new framework.

---

## 9. The Committee's Own Record

The committee's own proceedings confirm the gaps are known and unresolved.

LEWG polled the allocator question directly ([P3796R1](https://wg21.link/p3796r1), September 2025):

> "We would like to use the allocator provided by the receivers env instead of the one from the coroutine frame"
>
> | SF | F | N | A | SA |
> |----|---|---|---|----|
> |  0 | 0 | 5 | 0 |  0 |
>
> Attendance: 14. Outcome: strictly neutral.

The entire room abstained. Without a mechanism to propagate allocator context through nested coroutine calls, the committee had no direction to endorse. [D3980R0](https://isocpp.org/files/papers/D3980R0.html) (Kuhl, 2026-01-25) subsequently reworked the allocator propagation model relative to [P3552R3](https://wg21.link/p3552r3), adopted only six months earlier at Sofia. LWG 4356 confirms the gap has been filed as a specification defect.

The task type itself was contested. The forwarding poll (LEWG, 2025-05-06):

> "Forward P3552R1 to LWG for C++29"
>
> SF:5 / F:7 / N:0 / A:0 / SA:0 - unanimous.
>
> "Forward P3552R1 to LWG with a recommendation to apply for C++26 (if possible)."
>
> SF:5 / F:3 / N:4 / A:1 / SA:0 - weak consensus, with "if possible" qualifier.

The earlier design approval poll for P3552R1 was notably soft: SF:5 / F:6 / N:6 / A:1 / SA:0, six neutral votes matching six favorable votes. C++29 forwarding was unanimous. C++26 was conditional and weak. [P3796R1](https://wg21.link/p3796r1) ("Coroutine Task Issues") catalogues sixteen distinct open concerns about `task`. [P3801R0](https://wg21.link/p3801r0) ("Concerns about the design of `std::execution::task`," Jonathan Wakely, 2025) was filed in July 2025. P2300 was previously deferred from C++23 for maturity concerns; the same pattern of ongoing design changes is present again.

---

## 10. Working With the Grain

Herb Sutter reported that Citadel Securities already uses `std::execution` in production: *["We already use C++26's `std::execution` in production for an entire asset class, and as the foundation of our new messaging infrastructure."](https://herbsutter.com/2025/04/23/living-in-the-future-using-c26-at-work)* This confirms `std::execution` works well in its own domain: high-frequency trading, compile-time work graphs. When Senders operate in their domain, the design is elegant. Working with the grain of CPS.

[P4003R0](https://wg21.link/p4003r0) ("IoAwaitables: A Coroutines-Only Framework") is not proposed for standardization. The paper is not yet ready for that step. Yet a working library exists, compiles on three major toolchains, has benchmarks and unit tests, and serves as the foundation for a coroutine-only portable network library also in development. Here is what a coroutine-only networking design looks like when it is free to serve its own domain:

```cpp
// main.cpp - the launch site decides allocation policy
int main()
{
    io_context ioc;
    pmr::monotonic_buffer_resource pool;

    // allocator set once at launch site
    run_async(ioc.get_executor(), &pool)(accept_connections(ioc));

    ioc.run();
}

// server.cpp - coroutines just do their job
task<> accept_connections(io_context& ioc)
{
    auto stop_token = co_await this_coro::stop_token;
    tcp::acceptor acc(ioc, {tcp::v4(), 8080});
    while (! stop_token.stop_requested())
    {
        auto [ec, sock] = co_await acc.accept();
        if (ec) co_return;
        run_async(ioc.get_executor())(
            handle_request(std::move(sock)));
    }
}

task<> handle_request(tcp::socket sock)
{
    auto [ec, n] = co_await sock.read(buf);
    buf.commit(n);                          // partial success: use what arrived
    if (ec) co_return;                      // then check the status

    auto doc = co_await parse_json(buf);
    auto resp = co_await build_response(doc);

    pmr::unsynchronized_pool_resource bg_pool;
    auto [ec2] = co_await run(bg_executor, &bg_pool)(
        write_audit_log(resp));

    co_await sock.write(resp);
}
```

No `allocator_arg` in any signature. No forwarding. No `Environment` template parameter. No error channel routing decision. The task type is `template<class T> class task`, one parameter, matching established practice. None of the three gaps exist.

P4003R0 achieves this by using `thread_local` propagation to deliver the allocator to `promise_type::operator new` before `connect()`. The timing gap is solvable when the promise cooperates with a non-receiver mechanism.

P4003R0 does not provide compile-time work graph construction, zero-allocation pipelines, or vendor extensibility. It does not serve GPU dispatch or heterogeneous computing. That is the point. Different models of computation serve different domains. When each works with the grain of its own model, the design is elegant. When `task<T>` forces coroutines against the grain of CPS, three gaps appear. The gaps disappear not because P4003R0 is cleverer, but because it is not fighting the CPS/direct-style mismatch.

---

## 11. Suggested Straw Polls

We ask the committee to consider the following straw polls, which address the consequences documented in this paper.

**Diagnosis:**

> "The pattern documented in Section 7 and Appendix B, friction points clustering at the boundary between the Sender Sub-Language and regular C++, is best explained as a structural consequence of integrating a CPS-based model with direct-style C++, rather than typical end-of-cycle specification polish."

**Task type:**

> "The ergonomic cost of `co_yield with_error` is an acceptable trade-off for `std::execution::task` in C++26."

> "`std::execution::task` should not ship in C++26. The coroutine integration should iterate independently for C++29."

**Framework:**

> "The behavior of `when_all` under mixed channel conventions is acceptable for I/O use cases."

> "The allocator sequencing gap, where the receiver's environment is structurally unavailable at coroutine frame allocation time, is acceptable for C++26."

**Broader question:**

> "`std::execution` should be considered a domain-specific model for compile-time work graph construction, not a universal foundation for all asynchronous C++."

> "There is room in the standard library for a coroutine-native I/O model alongside `std::execution`."

---

## 12. Conclusion

The committee has adopted the Sender Sub-Language into C++26. This paper does not argue otherwise. The Sub-Language has real value for the domains it serves, and the committee made that decision with good reason.

But the decision has consequences for coroutines, and this paper asks the committee to address those consequences explicitly.

1. **Is the Sender Sub-Language still considered a universal model of asynchronous computation?** The evidence in this paper suggests it serves specific domains, GPU dispatch, high-frequency trading, heterogeneous computing, but not I/O. The three gaps are structural consequences of applying a CPS model to a domain whose completion semantics are inherently runtime.

2. **Are coroutines the primary tool for writing asynchronous C++ that most developers will use?** If yes, the coroutine integration deserves the same level of design investment as the sender pipeline itself, not an adaptation layer absorbing every cost of a model it does not use.

3. **Should coroutines be required to work through the Sender Sub-Language to access asynchronous I/O?** The SG4 poll (Kona 2023, SF:5/F:5/N:1/A:0/SA:1) answers this implicitly. This paper asks the committee to answer it explicitly, with full knowledge of the structural costs.

4. **Should `task<T>` ship in C++26 with these structural costs, or should the coroutine integration iterate independently?** Ship `std::execution` for the domains it serves. Let the coroutine integration develop on its own timeline.

The suggested straw polls in Section 11 offer the committee a way to record its answers.

The namespace `std::execution` claims universality. The evidence in this paper suggests the claim is too broad. The Sub-Language serves compile-time work graph construction, zero-allocation pipelines, and vendor-extensible heterogeneous dispatch exceptionally well. Coroutine-based I/O is not in that list. Perhaps the right question is not how to make `std::execution::task` serve I/O, but whether there is room in the standard for `std::io` alongside `std::execution`, each serving its own domain, with interoperation at the boundary.

---

## Appendix A - Code Examples

### A.1 Why HALO Cannot Help

HALO allows compilers to elide coroutine frame allocation when the frame's lifetime is provably bounded by its caller. When an I/O coroutine is launched onto an execution context, the frame must outlive the launching function:

```cpp
namespace ex = std::execution;

task<size_t> read_data(socket& s, buffer& buf)
{
    co_return co_await s.async_read(buf);
}

void start_read(ex::counting_scope& scope, auto sch)
{
    ex::spawn(
        ex::on(sch, read_data(sock, buf)),
        scope.get_token());
}
```

The compiler cannot prove bounded lifetime, so HALO cannot apply and allocation is mandatory.

### A.2 The Full Ceremony for Allocator-Aware Coroutines

The Sub-Language requires five layers of machinery to propagate a custom allocator through a coroutine call chain:

```cpp
namespace ex = std::execution;

// 1. Define a custom environment with the allocator
struct my_env
{
    using allocator_type = recycling_allocator<>;
    allocator_type alloc;

    friend auto tag_invoke(
        ex::get_allocator_t, my_env const& e) noexcept
    {
        return e.alloc;
    }
};

// 2. Alias the task type with the custom allocator
using my_task = ex::basic_task<
    ex::task_traits<my_env::allocator_type>>;

// 3. Every coroutine accepts and forwards the allocator
template<typename Allocator>
my_task level_two(
    int x,
    std::allocator_arg_t = {},
    Allocator alloc = {})
{
    co_return;
}

template<typename Allocator>
my_task level_one(
    int v,
    std::allocator_arg_t = {},
    Allocator alloc = {})
{
    co_await level_two(42, std::allocator_arg, alloc);
    co_return;
}

// At the launch site: inject the allocator via write_env
void launch(ex::io_context& ctx)
{
    my_env env{recycling_allocator<>{}};
    auto sndr =
        ex::write_env(level_one(0), env)
      | ex::continues_on(ctx.get_scheduler());
    ex::spawn(std::move(sndr), ctx.get_token());
}
```

Forgetting any one of the five steps silently falls back to the default allocator. The compiler provides no diagnostic.

### A.3 P3826R3 and Algorithm Dispatch

[P3826R3](https://wg21.link/p3826r3) addresses sender algorithm customization. P3826 offers five solutions. All target algorithm dispatch:

**Solution 4.1: Remove all `std::execution`.** Resolves the allocator sequencing gap by deferral.

**Solution 4.2: Remove customizable sender algorithms.** Does not change when the allocator becomes available.

**Solution 4.3: Remove sender algorithm customization.** Does not change when the allocator becomes available.

**Solution 4.4: Ship as-is, fix via DR.** Defers the fix. Does not change when the allocator becomes available.

**Solution 4.5: Fix algorithm customization now.** Restructures `transform_sender` to take the receiver's environment, changing information flow at `connect()` time. This enables correct algorithm dispatch but does not change when the allocator becomes available. This restructuring could enable future allocator solutions, but none has been proposed.

### A.4 Why the Three-Channel Model Exists

The Sub-Language constructs the entire work graph at compile time as a deeply-nested template type. [`connect(sndr, rcvr)`](https://eel.is/c++draft/exec.connect) collapses the pipeline into a single concrete type. For this to work, every control flow path must be distinguishable at the type level, not the value level.

The three completion channels provide exactly this. [Completion signatures](https://eel.is/c++draft/exec.getcomplsigs) declare three distinct type-level paths:

```cpp
using completion_signatures =
    stdexec::completion_signatures<
        set_value_t(int),                                 // value path
        set_error_t(error_code),                          // error path
        set_stopped_t()>;                                 // stopped path
```

`when_all` dispatches on which channel fired without inspecting payloads. `upon_error` attaches to the error path at the type level. `let_value` attaches to the value path at the type level. The routing is in the types, not in the values.

If errors were delivered as values (for example, `expected<int, error_code>` through `set_value`), the compiler would see one path carrying one type. `when_all` could not cancel siblings on error without runtime inspection of the payload. Every algorithm would need runtime branching logic to inspect the expected and route accordingly.

The three channels exist because the Sender Sub-Language is a compile-time language.

Compile-time languages route on types. Runtime languages route on values. A coroutine returns `auto [ec, n] = co_await read(buf)` and branches with `if (ec)` at runtime. The Sub-Language encodes `set_value` and `set_error` as separate types in the completion signature and routes at compile time. The three-channel model is not an arbitrary design choice. It is a structural requirement of the compile-time work graph.

A compile-time language cannot express partial success. I/O operations return `(error_code, size_t)` together because partial success is normal. The three-channel model demands that the sender author choose one channel. No choice is correct because the compile-time type system cannot represent "both at once."

Eliminating the three-channel model would remove the type-level routing that makes the compile-time work graph possible. The three channels are not a design flaw. They are the price of compile-time analysis. I/O cannot pay that price because I/O's completion semantics are inherently runtime.

---

## Appendix B - Chronological Churn Data

This appendix provides a chronological view of all 50 items modifying `std::execution` since Tokyo (March 2024). The rate accelerated from 0.88 to 2.80 items/month over the first four complete periods, and the categories of change are expanding rather than contracting. Whether viewed chronologically (this appendix) or by friction type (Section 7), the data tells the same story: all complexity flows from the Sub-Language into its boundaries with regular C++. All data is gathered from the published [WG21 paper mailings](https://open-std.org/jtc1/sc22/wg21/docs/papers/), the [LWG issues list](https://cplusplus.github.io/LWG/lwg-toc.html), and the [C++26 national body ballot comments](https://github.com/cplusplus/nbballot).

| Period                          | Months | Removals | Reworks | Wording | Missing | LWG  | Total  |
|---------------------------------|:------:|:--------:|:-------:|:-------:|:-------:|:----:|:------:|
| Pre-Wroclaw (Mar-Oct 2024)      |    8   |    1     |    5    |    0    |    1    |  0   |     7  |
| Pre-Hagenberg (Nov 2024-Feb 25) |    4   |    1     |    0    |    2    |    2    |  3   |     8  |
| Pre-Sofia (Mar-Jun 2025)        |    4   |    0     |    2    |    0    |    7    |  1   |    10  |
| Pre-Kona (Jul-Nov 2025)         |    5   |    0     |    3    |    3    |    1    |  7   |    14  |
| Pre-London (Dec 2025-Feb 2026)  |    3   |    0     |    5    |    1    |    0    |  0   |     6  |
| **Total**                       | **24** |  **2**   | **15**  |  **6**  | **11**  |**11**| **45** |

Five NB comments on `task` allocator support and signal safety bring the total to 50 items.

| Period                          | Items | Months | Items/Month |
|---------------------------------|:-----:|:------:|:-----------:|
| Pre-Wroclaw (Mar-Oct 2024)      |     7 |      8 |        0.88 |
| Pre-Hagenberg (Nov 2024-Feb 25) |     8 |      4 |        2.00 |
| Pre-Sofia (Mar-Jun 2025)        |    10 |      4 |        2.50 |
| Pre-Kona (Jul-Nov 2025)         |    14 |      5 |        2.80 |
| Pre-London (Dec 2025-Feb 2026)  |     6 |      3 |        2.00 |

Every one of the 50 items falls into one of two categories:

| Category                           | Items | Examples                                                                     |
|------------------------------------|:-----:|------------------------------------------------------------------------------|
| Sub-Language machinery             |  ~35  | Algorithm customization (4 papers), removals, operation states, environments |
| Sub-Language-to-coroutine bridge   |  ~15  | `task`, allocator model, `connect-awaitable`, `as.awaitable`, NB comments    |
| Coroutine-intrinsic issues         |   0   | (none)                                                                       |

---

## Appendix C - Complete Item Catalogue

### Removals

| Paper                                          | Title                                                   | Date       | Status  |
|------------------------------------------------|---------------------------------------------------------|------------|---------|
| [P3187R1](https://wg21.link/p3187r1)          | Remove `ensure_started` and `start_detached` from P2300 | 2024-10-15 | Adopted |
| [P3682R0](https://wg21.link/p3682r0)          | Remove `std::execution::split`                          | 2025-02-04 | Adopted |

### Major Design Reworks

| Paper                                                    | Title                                                     | Date       | Status      |
|----------------------------------------------------------|-----------------------------------------------------------|------------|-------------|
| [P2855R1](https://wg21.link/p2855r1)                    | Member customization points for Senders and Receivers     | 2024-03-18 | Adopted     |
| [P2999R3](https://wg21.link/p2999r3)                    | Sender Algorithm Customization                            | 2024-04-16 | Adopted     |
| [P3303R1](https://wg21.link/p3303r1)                    | Fixing Lazy Sender Algorithm Customization                | 2024-10-15 | Adopted     |
| [P3175R3](https://wg21.link/p3175r3)                    | Reconsidering the `std::execution::on` algorithm          | 2024-10-15 | Adopted     |
| [P3557R3](https://wg21.link/p3557r3)                    | High-Quality Sender Diagnostics with Constexpr Exceptions | 2025-06-10 | Adopted     |
| [P3570R2](https://wg21.link/p3570r2)                    | Optional variants in sender/receiver                      | 2025-06-14 | Adopted     |
| [P3718R0](https://wg21.link/p3718r0)                    | Fixing Lazy Sender Algorithm Customization, Again         | 2025-07-24 | In Progress |
| [P3826R3](https://wg21.link/p3826r3)                    | Fix Sender Algorithm Customization                        | 2025-11-14 | In Progress |
| [P3927R0](https://wg21.link/p3927r0)                    | `task_scheduler` Support for Parallel Bulk Execution      | 2026-01-15 | In Progress |
| [D3980R0](https://isocpp.org/files/papers/D3980R0.html) | Task's Allocator Use                                      | 2026-01-25 | In Progress |
| [P3373R2](https://wg21.link/p3373r2)                    | Of Operation States and Their Lifetimes                   | 2025-12-29 | In Progress |
| [P3941R1](https://wg21.link/p3941r1)                    | Scheduler Affinity                                        | 2026-01-14 | In Progress |
| [P3950R0](https://wg21.link/p3950r0)                    | `return_value` & `return_void` Are Not Mutually Exclusive | 2025-12-21 | In Progress |

### LWG Defects

| Issue                                                    | Title                                                            | Priority/Status    |
|----------------------------------------------------------|------------------------------------------------------------------|--------------------|
| [LWG 4368](https://cplusplus.github.io/LWG/issue4368)   | Potential dangling reference from `transform_sender`             | Open - Priority 1  |
| [LWG 4206](https://cplusplus.github.io/LWG/issue4206)   | `connect_result_t` should be constrained with `sender_to`        | Open - Priority 1  |
| [LWG 4215](https://cplusplus.github.io/LWG/issue4215)   | `run_loop::finish` should be `noexcept`                          | Open               |
| [LWG 4190](https://cplusplus.github.io/LWG/issue4190)   | `completion-signatures-for` specification is recursive           | Open               |
| [LWG 4356](https://cplusplus.github.io/LWG/issue4356)   | `connect()` should use `get_allocator(get_env(rcvr))`            | Open               |

### NB Comments

| NB Comment                                                       | Title                                                   | Status             |
|------------------------------------------------------------------|---------------------------------------------------------|--------------------|
| [US 255-384](https://github.com/cplusplus/nbballot/issues/959)  | Use allocator from receiver's environment               | Wording in D3980R0 |
| [US 253-386](https://github.com/cplusplus/nbballot/issues/961)  | Allow use of arbitrary allocators for coroutine frame    | Wording in D3980R0 |
| [US 254-385](https://github.com/cplusplus/nbballot/issues/960)  | Constrain `allocator_arg` argument position              | Wording in D3980R0 |
| [US 261-391](https://github.com/cplusplus/nbballot/issues/966)  | Bad specification of parameter type                      | Wording in D3980R0 |
| CH                                                               | Signal-safety defect                                     | Needs resolution   |

---

## References

### WG21 Papers

1. [P2300R10](https://wg21.link/p2300r10) - "std::execution" (Michal Dominiak, Georgy Evtushenko, Lewis Baker, Lucian Radu Teodorescu, Lee Howes, Kirk Shoop, Eric Niebler, 2024)
2. [P2300R4](https://wg21.link/p2300r4) - "std::execution" (Michal Dominiak, et al., 2022)
3. [P2430R0](https://open-std.org/jtc1/sc22/wg21/docs/papers/2021/p2430r0.pdf) - "Partial success scenarios with P2300" (Chris Kohlhoff, 2021)
4. [P2762R2](https://wg21.link/p2762r2) - "Sender/Receiver Interface For Networking" (Dietmar Kuhl, 2023)
5. [P2855R1](https://wg21.link/p2855r1) - "Member customization points for Senders and Receivers" (Ville Voutilainen, 2024)
6. [P2999R3](https://wg21.link/p2999r3) - "Sender Algorithm Customization" (Eric Niebler, 2024)
7. [P3149R11](https://wg21.link/p3149r11) - "async_scope" (Ian Petersen, Jessica Wong, Kirk Shoop, et al., 2025)
8. [P3164R4](https://wg21.link/p3164r4) - "Improving Diagnostics for Sender Expressions" (Eric Niebler, 2024)
9. [P3175R3](https://wg21.link/p3175r3) - "Reconsidering the std::execution::on algorithm" (Eric Niebler, 2024)
10. [P3187R1](https://wg21.link/p3187r1) - "Remove ensure_started and start_detached from P2300" (Lewis Baker, Eric Niebler, 2024)
11. [P3303R1](https://wg21.link/p3303r1) - "Fixing Lazy Sender Algorithm Customization" (Eric Niebler, 2024)
12. [P3373R2](https://wg21.link/p3373r2) - "Of Operation States and Their Lifetimes" (Robert Leahy, 2025)
13. [P3552R3](https://wg21.link/p3552r3) - "Add a Coroutine Task Type" (Dietmar Kuhl, Maikel Nadolski, 2025)
14. [P3557R3](https://wg21.link/p3557r3) - "High-Quality Sender Diagnostics with Constexpr Exceptions" (Eric Niebler, 2025)
15. [P3570R2](https://wg21.link/p3570r2) - "Optional variants in sender/receiver" (Fabio Fracassi, 2025)
16. [P3682R0](https://wg21.link/p3682r0) - "Remove std::execution::split" (Eric Niebler, 2025)
17. [P3718R0](https://wg21.link/p3718r0) - "Fixing Lazy Sender Algorithm Customization, Again" (Eric Niebler, 2025)
18. [P3796R1](https://wg21.link/p3796r1) - "Coroutine Task Issues" (Dietmar Kuhl, 2025)
19. [P3801R0](https://wg21.link/p3801r0) - "Concerns about the design of std::execution::task" (Jonathan Wakely, 2025)
20. [P3826R3](https://wg21.link/p3826r3) - "Fix Sender Algorithm Customization" (Eric Niebler, 2026)
21. [P3927R0](https://wg21.link/p3927r0) - "task_scheduler Support for Parallel Bulk Execution" (Lee Howes, 2026)
22. [P3941R1](https://wg21.link/p3941r1) - "Scheduler Affinity" (Dietmar Kuhl, 2026)
23. [P3950R0](https://wg21.link/p3950r0) - "return_value & return_void Are Not Mutually Exclusive" (Robert Leahy, 2025)
24. [D3980R0](https://isocpp.org/files/papers/D3980R0.html) - "Task's Allocator Use" (Dietmar Kuhl, 2026)
25. [P4003R0](https://wg21.link/p4003r0) - "IoAwaitables: A Coroutines-Only Framework" (Vinnie Falco, 2026)
26. [D4014R0](https://wg21.link/d4014) - "The Sender Sub-Language" (Vinnie Falco, 2026)
27. [N5028](https://wg21.link/n5028) - "Result of voting on ISO/IEC CD 14882" (2025)

### LWG Issues

28. [LWG 4368](https://cplusplus.github.io/LWG/issue4368) - "Potential dangling reference from `transform_sender`" (Priority 1)
29. [LWG 4206](https://cplusplus.github.io/LWG/issue4206) - "`connect_result_t` should be constrained with `sender_to`" (Priority 1)
30. [LWG 4190](https://cplusplus.github.io/LWG/issue4190) - "`completion-signatures-for` specification is recursive" (Priority 2)
31. [LWG 4215](https://cplusplus.github.io/LWG/issue4215) - "`run_loop::finish` should be `noexcept`"
32. [LWG 4356](https://cplusplus.github.io/LWG/issue4356) - "`connect()` should use `get_allocator(get_env(rcvr))`"

### Blog Posts

33. Eric Niebler, ["Ranges, Coroutines, and React: Early Musings on the Future of Async in C++"](https://ericniebler.com/2017/08/17/ranges-coroutines-and-react-early-musings-on-the-future-of-async-in-c/) (2017)
34. Eric Niebler, ["Structured Concurrency"](https://ericniebler.com/2020/11/08/structured-concurrency/) (2020)
35. Eric Niebler, ["Asynchronous Stacks and Scopes"](https://ericniebler.com/2021/08/29/asynchronous-stacks-and-scopes/) (2021)
36. Eric Niebler, ["What are Senders Good For, Anyway?"](https://ericniebler.com/2024/02/04/what-are-senders-good-for-anyway/) (2024)
37. Herb Sutter, ["Trip report: Summer ISO C++ standards meeting (St Louis, MO, USA)"](https://herbsutter.com/2024/07/02/trip-report-summer-iso-c-standards-meeting-st-louis-mo-usa) (2024)
38. Herb Sutter, ["Living in the future: Using C++26 at work"](https://herbsutter.com/2025/04/23/living-in-the-future-using-c26-at-work) (2025)
39. Bob Nystrom, ["What Color is Your Function?"](https://journal.stuffwithstuff.com/2015/02/01/what-color-is-your-function/) (2015)

### Programming Language Theory

40. Olivier Danvy, ["Back to Direct Style"](https://static.aminer.org/pdf/PDF/001/056/774/back_to_direct_style.pdf) (1992)
41. Gordon Plotkin, ["Call-by-Name, Call-by-Value and the lambda-Calculus"](https://homepages.inf.ed.ac.uk/gdp/publications/cbn_cbv_lambda.pdf) (1975)
42. Christopher Strachey & Christopher Wadsworth, ["Continuations: A Mathematical Semantics for Handling Full Jumps"](https://www.cs.ox.ac.uk/publications/publication3729-abstract.html) (1974)
43. Eugenio Moggi, ["Notions of Computation and Monads"](https://person.dibris.unige.it/moggi-eugenio/ftp/ic91.pdf) (1991)
44. John Reynolds, ["The Discoveries of Continuations"](https://homepages.inf.ed.ac.uk/wadler/papers/papers-we-love/reynolds-discoveries.pdf) (1993)
45. Guy Steele & Gerald Sussman, [The Lambda Papers](https://en.wikisource.org/wiki/Lambda_Papers) (1975-1980)

### Other

46. [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines.html) - F.7, R.30 (Bjarne Stroustrup, Herb Sutter, eds.)
47. Herb Sutter, [GotW #91: Smart Pointer Parameters](https://herbsutter.com/2013/06/05/gotw-91-solution-smart-pointer-parameters/) (2013)
48. Alisdair Meredith & Pablo Halpern, ["Getting Allocators out of Our Way"](https://www.youtube.com/watch?v=RLezJuqNcEQ) (CppCon 2019)
49. Gregor Kiczales et al., ["Aspect-Oriented Programming"](https://www.cs.ubc.ca/~gregor/papers/kiczales-ECOOP1997-AOP.pdf) (ECOOP 1997)
50. Butler Lampson, "Hints for Computer System Design" (1983)
