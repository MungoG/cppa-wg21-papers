---
title: "How Do Plain Awaitables Receive a Stop Token?"
document: D0000
date: 2026-02-14
reply-to:
  - "Vinnie Falco <vinnie.falco@gmail.com>"
audience: WG21
---

## Abstract

[P3552R3](https://wg21.link/p3552) requires that `std::execution::task` be "awaiter/awaitable friendly." The sender/receiver protocol delivers the stop token through `get_stop_token(get_env(receiver))`, available only after `connect()`. A plain awaitable - one that is not a sender - has no receiver, no environment, and no standard mechanism to receive a stop token. This paper asks how that gap should be addressed.

---

## 1. Introduction

[P3552R3](https://wg21.link/p3552) states that a coroutine task "needs to be awaiter/awaitable friendly, i.e., it should be possible to `co_await` awaitables which includes both library provided and user provided ones" (Section 3). [P3796R1](https://open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3796r1.html) Section 3.5.8 identifies that "awaitable non-senders are not supported."

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
            return self_->do_write(h, sv_, impl_);
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

## 4. Conclusion

We suggest this question be answered before `std::execution` ships in C++26.

---

## References

1. [P2300R10](https://wg21.link/p2300) - std::execution (Michal Dominiak, Georgy Evtushenko, Lewis Baker, Lucian Radu Teodorescu, Lee Howes, Kirk Shoop, Eric Niebler)
2. [P3552R3](https://wg21.link/p3552) - Add a Coroutine Task Type (Dietmar Kuhl, Maikel Nadolski)
3. [P3796R1](https://open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3796r1.html) - Coroutine Task Issues (Dietmar Kuhl)
