---
title: "How Do Plain Awaitables Receive a Stop Token?"
document: D0000
date: 2026-02-14
reply-to:
  - "Vinnie Falco \<vinnie.falco@gmail.com\>"
audience: SG1, LEWG
---

## Abstract

[P3552R3](https://wg21.link/p3552) ("Add a Coroutine Task Type") requires that `std::execution::task` be "awaiter/awaitable friendly." The sender/receiver protocol delivers the stop token through `get_stop_token(get_env(receiver))`, available only after `connect()`. A plain awaitable - one that is not a sender - has no receiver, no environment, and no standard mechanism to receive a stop token. Sections 1-3 describe the gap. Section 4 demonstrates that at least one library-level solution exists. Section 5 examines that solution's shortcomings honestly.

---

## 1. Introduction

[P3552R3](https://wg21.link/p3552) states that a coroutine task "needs to be awaiter/awaitable friendly, i.e., it should be possible to `co_await` awaitables which includes both library provided and user provided ones" (Section 3). [P3796R1](https://wg21.link/p3796r1) Section 3.5.8 identifies that "awaitable non-senders are not supported."

When a plain awaitable is `co_await`ed inside `std::execution::task`, how does it receive a stop token?

---

## 2. How Senders Receive the Stop Token

When a sender is `co_await`ed inside a `task`, the stop token is delivered through the following path:

1. The `co_await` expression triggers `with_awaitable_senders::await_transform` ([[exec.with.awaitable.senders]](https://eel.is/c++draft/exec.with.awaitable.senders)/3), which calls `as_awaitable(sender, promise)`.

2. `as_awaitable` recognizes the sender and constructs `sender-awaitable` ([[exec.as.awaitable]](https://eel.is/c++draft/exec.as.awaitable)/7.4), which calls `connect(sender, awaitable-receiver{...})` ([[exec.as.awaitable]](https://eel.is/c++draft/exec.as.awaitable)/5).

3. The `awaitable-receiver` forwards environment queries to the promise ([[exec.as.awaitable]](https://eel.is/c++draft/exec.as.awaitable)/4.4):

   > `tag(get_env(as_const(crcvr.continuation.promise())), as...)`

4. The sender's operation state queries `get_stop_token(get_env(receiver))` ([P2300R10](https://wg21.link/p2300)) and receives the token.

In code:

```cpp
// as_awaitable creates an awaitable-receiver for senders:
auto op = connect(sender, awaitable_receiver{&result, handle});

// The receiver forwards queries to the promise:
auto token = get_stop_token(get_env(receiver));
// Token is available. Cancellation works.
```

This path depends on `connect` producing an `awaitable-receiver` that bridges the sender to the promise's environment. Every link in the chain is well-specified.

---

## 3. Plain Awaitables

When `as_awaitable` receives a type that is already an awaitable - not a sender - it falls through to [[exec.as.awaitable]](https://eel.is/c++draft/exec.as.awaitable)/7.2 or /7.5: the expression is returned unchanged. No `awaitable-receiver` is created. No `connect` is called.

```cpp
struct my_timer
{
    bool await_ready() const noexcept { return false; }

    void await_suspend(std::coroutine_handle<> h)
    {
        // as_awaitable passed us through unchanged.
        // No receiver was created. No environment.
    }

    void await_resume() {}
};
```

Questions arise:

1. Where is the receiver?

2. Where is the environment?

3. Where is the stop token?

The awaitable can template `await_suspend` on the promise type. The `task::promise_type` does expose `get_env()` ([[task.promise]](https://eel.is/c++draft/exec.task)):

```cpp
template<typename Promise>
void await_suspend(std::coroutine_handle<Promise> h)
{
    auto token = get_stop_token(h.promise().get_env());
    // This compiles for a specific promise type.
    // But a generic timer, channel, or async mutex
    // cannot know what Promise is at definition time.
}
```

Four observations:

1. `task::promise_type` exposes `get_env()` ([[task.promise]](https://eel.is/c++draft/exec.task)), but this is specific to [P3552R3](https://wg21.link/p3552). There is no general requirement in [P2300R10](https://wg21.link/p2300) or elsewhere that promise types expose `get_env()`. `with_awaitable_senders` ([[exec.with.awaitable.senders]](https://eel.is/c++draft/exec.with.awaitable.senders)) does not add it.

2. A plain awaitable calling `h.promise().get_env()` is coupled specifically to `std::execution::task::promise_type` - not to coroutine promise types in general.

3. The same awaitable might be `co_await`ed by `std::execution::task`, by a user's custom coroutine type, by `cppcoro::task`, or by any other coroutine. Each has a different promise type. Most do not expose `get_env()`.

4. Senders do not have this problem. A sender receives its environment through `connect(sender, receiver)` - a well-defined protocol that does not depend on who is awaiting the result. The sender does not need to know its caller's promise type. Plain awaitables have no such protocol.

Consider a concrete example. A `string_sink` whose `write` member function returns a plain awaitable. The implementation of `do_write` is declared but not defined - it lives in a `.cpp` file behind an ABI boundary:

```cpp
class string_sink
{
    void* impl_;

    std::coroutine_handle<>
    do_write( std::coroutine_handle<> h, std::string_view sv );

public:
    struct write_awaitable
    {
        string_sink* self_;
        std::string_view sv_;

        bool await_ready() const noexcept { return false; }

        std::coroutine_handle<>
        await_suspend( std::coroutine_handle<> h )
        {
            // returns std::noop_coroutine
            return self_->do_write( h, sv_ );
        }
        void await_resume() noexcept {}
    };

    // write the string asynchronously
    write_awaitable write( std::string_view sv )
    {
        return write_awaitable{ this, sv };
    }
};
```

How do we get the stop token into `do_write`?

---

## 4. An Alternative Approach

The gap described above is not inherent to C++ coroutines. The following listing demonstrates a library-level mechanism that delivers the stop token to a plain awaitable without requiring the awaitable to know the promise type. The listing is not a proposal for standardization; it demonstrates that the design space is not empty.

The idea: the promise type wraps each co_awaited awaitable so that `await_suspend` receives a second argument - a pointer to an environment struct carrying the stop token. The awaitable accepts `std::coroutine_handle<>` (ABI-stable) and never couples to the promise type.

```cpp
#include <coroutine>
#include <stop_token>
#include <cassert>

// the environment passed to every awaitable
struct io_env
{
    std::stop_token stop_token;
};

// a plain awaitable that receives the stop token
// through await_suspend - no promise type knowledge
struct cancellable_op
{
    bool was_stopped = false;

    bool await_ready() const noexcept { return false; }

    void await_suspend(
        std::coroutine_handle<> h, io_env const* env)
    {
        // the stop token arrives here
        was_stopped = env->stop_token.stop_requested();
        h.resume();
    }

    bool await_resume() noexcept { return was_stopped; }
};

// the promise wraps awaitables to inject the environment
struct promise_type
{
    io_env const* env_ = nullptr;

    template<class Awaitable>
    struct wrapper
    {
        Awaitable& a_;
        io_env const* env_;

        bool await_ready() { return a_.await_ready(); }

        void await_suspend(std::coroutine_handle<> h)
        {
            // inject the environment as a second argument
            a_.await_suspend(h, env_);
        }

        auto await_resume() { return a_.await_resume(); }
    };

    template<class A>
    auto await_transform(A&& a)
    {
        return wrapper<A>{a, env_};
    }

    // --- minimal coroutine boilerplate ---
    struct task
    {
        using promise_type = ::promise_type;
        std::coroutine_handle<::promise_type> h_;
    };
    task get_return_object()
    {
        return {std::coroutine_handle<promise_type>
            ::from_promise(*this)};
    }
    std::suspend_always initial_suspend() noexcept
    {
        return {};
    }
    std::suspend_always final_suspend() noexcept
    {
        return {};
    }
    void return_void() {}
    void unhandled_exception() {}
};

using task = promise_type::task;

// a coroutine that co_awaits the plain awaitable
task example()
{
    cancellable_op op;
    bool stopped = co_await op;
    assert(stopped);
}

int main()
{
    std::stop_source src;
    src.request_stop();
    io_env env{src.get_token()};

    auto t = example();
    t.h_.promise().env_ = &env;
    t.h_.resume();
    t.h_.destroy();
}
```

The awaitable `cancellable_op` receives the stop token through `await_suspend(std::coroutine_handle<>, io_env const*)`. It does not know the promise type. Its `await_suspend` takes `coroutine_handle<>` - an ABI-stable signature suitable for use behind a shared library boundary.

The `string_sink` from Section 3 would look like this under the same model:

```cpp
class string_sink
{
    void* impl_;

    std::coroutine_handle<>
    do_write(
        std::coroutine_handle<> h,
        std::string_view sv,
        io_env const* env);

public:
    struct write_awaitable
    {
        string_sink* self_;
        std::string_view sv_;

        bool await_ready() const noexcept { return false; }

        std::coroutine_handle<>
        await_suspend(
            std::coroutine_handle<> h,
            io_env const* env)
        {
            // stop token available via env->stop_token
            return self_->do_write(h, sv_, env);
        }
        void await_resume() noexcept {}
    };

    write_awaitable write(std::string_view sv)
    {
        return write_awaitable{this, sv};
    }
};
```

The signature remains ABI-stable: `do_write` takes `std::coroutine_handle<>` and `io_env const*`, both of which are fixed types. The stop token reaches the implementation behind the ABI boundary without templating on the promise type.

[P4003R0](https://wg21.link/p4003r0) ("IoAwaitables: A Coroutines-Only Framework") provides a complete execution model built on this mechanism, including automatic propagation through nested coroutine chains, `when_all` with sibling cancellation, and production benchmarks.

---

## 5. Shortcomings

The approach demonstrated in Section 4 has real limitations.

**Non-standard `await_suspend` signature.** The C++ coroutine specification defines `await_suspend` as taking `std::coroutine_handle<>` or `std::coroutine_handle<Promise>`. The two-argument form is not part of the language. The promise's `await_transform` wraps the awaitable to inject the second argument, so the mechanism works - but awaitables must be designed for the protocol. Plain awaitables written for the standard one-argument signature still have no path without wrapping.

**Protocol coupling.** An awaitable designed for `await_suspend(handle, io_env const*)` works only with promise types that inject `io_env`. The awaitable does not work with `std::execution::task`, `cppcoro::task`, or other coroutine types without adaptation. The coupling is to a protocol rather than to a specific promise type, but it is coupling nonetheless.

**No language-level support.** A language-level mechanism - such as a standard way for promise types to inject context into `await_suspend` - would be more general than any library-level wrapper. Whether such a mechanism is feasible or desirable is a question for CWG, but the direction is worth exploring.

**Does not fix `std::execution::task`.** The listing demonstrates that the problem is solvable, but it does not propose changes to `std::execution`. The gap in `task` remains. Closing it would require changes to how `as_awaitable` handles plain awaitables - either by creating a receiver for them (analogous to the sender path) or by some other mechanism that delivers the environment.

---

## 6. Conclusion

A plain awaitable `co_await`ed inside `std::execution::task` has no standard mechanism to receive a stop token. The sender path creates an `awaitable-receiver` that bridges the sender to the promise's environment; the plain awaitable path does not. [P3552R3](https://wg21.link/p3552) explicitly requires "awaiter/awaitable friendly" behavior. [P3796R1](https://wg21.link/p3796r1) explicitly acknowledges that "awaitable non-senders are not supported."

At least one library-level solution exists, demonstrating that the design space is not empty. The stop token gap - like the allocator sequencing gap described in [D4007R0](https://wg21.link/p4007) - arises from fitting coroutine-based I/O into a protocol designed for a different use case. C++ might be better served by allowing each major use case - GPU dispatch, CPU-bound parallelism, networked I/O - to have an asynchronous execution model optimized for its requirements, rather than searching for a single universal model that serves all of them equally well.

We suggest the committee address this gap before freezing the `std::execution` API in the IS.

---

## References

1. [P2300R10](https://wg21.link/p2300) - std::execution (Michal Dominiak, Georgy Evtushenko, Lewis Baker, Lucian Radu Teodorescu, Lee Howes, Kirk Shoop, Eric Niebler)
2. [P3552R3](https://wg21.link/p3552) - Add a Coroutine Task Type (Dietmar Kuhl, Maikel Nadolski)
3. [P3796R1](https://wg21.link/p3796r1) - Coroutine Task Issues (Dietmar Kuhl)
4. [P4003R0](https://wg21.link/p4003r0) - IoAwaitables: A Coroutines-Only Framework (Vinnie Falco)
