---
title: "How Do Child Coroutines Receive a Frame Allocator?"
document: D0000
date: 2026-02-14
reply-to:
  - "Vinnie Falco <vinnie.falco@gmail.com>"
audience: WG21
---

## Abstract

[P3552R3](https://wg21.link/p3552) solves initial coroutine frame allocation by passing the allocator via `std::allocator_arg_t` at the call site. But when a coroutine calls a child coroutine, the child's `operator new` fires during the function-call expression - before `connect()`, before the receiver exists, before the allocator is reachable through the sender/receiver protocol. This paper asks how child coroutines receive a custom frame allocator.

---

## 1. Introduction

Coroutine frame allocation has a fundamental sequencing constraint: `promise_type::operator new` executes before the coroutine body ([dcl.fct.def.coroutine]). The allocator must be known at that moment. Any mechanism that provides the allocator later cannot help.

[P3552R3](https://wg21.link/p3552) addresses initial allocation cleanly: the caller passes the allocator via `std::allocator_arg_t`, and `promise_type::operator new` receives it through the coroutine's parameter list ([[task.promise]](https://eel.is/c++draft/exec.task)).

When a parent coroutine calls a child coroutine, how does the child receive the allocator?

---

## 2. How Initial Allocation Works

[P3552R3](https://wg21.link/p3552) specifies that if the coroutine's parameter list contains `std::allocator_arg_t`, the next parameter is the allocator ([[task.promise]](https://eel.is/c++draft/exec.task)/3-4). `promise_type::operator new` receives the coroutine's arguments and can extract it:

```cpp
// Initial allocation: caller provides the allocator
auto t = my_task(std::allocator_arg, alloc, args...);
// operator new receives (size, allocator_arg, alloc, args...)
// Allocator is available. Frame allocation works.
```

The sender/receiver protocol provides the allocator through a different path: `get_allocator(get_env(receiver))`. But `connect()` runs after the coroutine frame is already allocated. For initial allocation, the `allocator_arg_t` mechanism sidesteps this timing issue entirely.

---

## 3. Child Coroutine Allocation

Consider a parent coroutine that calls a child:

```cpp
task<void> parent(std::allocator_arg_t, Alloc alloc)
{
    // child's operator new fires HERE, during
    // the function-call expression
    co_await child();
}
```

The child's `operator new` executes during evaluation of `child()` - before `co_await`, before `await_transform`, before any sender/receiver machinery has an opportunity to intervene. The C++ evaluation order is:

1. `child()` is evaluated - `operator new` allocates the frame
2. `child()` returns `task<T>`
3. `co_await` triggers `await_transform`
4. If the child is a sender, `connect(child, receiver)` runs
5. `get_allocator(get_env(receiver))` becomes available

The allocator arrives at step 5. The frame was allocated at step 1.

For the child to use a custom allocator, the caller must forward it explicitly:

```cpp
task<void> parent(std::allocator_arg_t, Alloc alloc)
{
    co_await child(std::allocator_arg, alloc);
}
```

Every coroutine in the chain must accept the allocator. Every call site must forward it. The allocator becomes part of every coroutine's parameter list:

```cpp
task<void> child(std::allocator_arg_t, Alloc alloc)
{
    co_await grandchild(std::allocator_arg, alloc);
}

task<void> grandchild(std::allocator_arg_t, Alloc alloc)
{
    co_await great_grandchild(std::allocator_arg, alloc);
}
```

---

## 4. Questions

- The receiver's environment carries the allocator via `get_allocator(get_env(receiver))` ([P2300R10](https://wg21.link/p2300)). This environment is available after `connect()`. `operator new` runs before `connect()`. How does a child coroutine use the receiver's allocator for its frame?

- If the answer is manual forwarding via `allocator_arg_t`, does that not require every coroutine in a chain to accept and forward the allocator through its parameter list?

- Generic sender algorithms such as `let_value` launch child operations without knowledge of the caller's allocator. How does a coroutine invoked by a generic algorithm receive a custom frame allocator?

- [P3552R3](https://wg21.link/p3552) solves initial allocation. Is automatic propagation to child coroutines a problem that the committee intends to address, or is manual forwarding the intended design?

---

## 5. Conclusion

We suggest these questions be answered before `std::execution` ships in C++26.

---

## References

1. [P2300R10](https://wg21.link/p2300) - std::execution (Michal Dominiak, Georgy Evtushenko, Lewis Baker, Lucian Radu Teodorescu, Lee Howes, Kirk Shoop, Eric Niebler)
2. [P3552R3](https://wg21.link/p3552) - Add a Coroutine Task Type (Dietmar Kuhl, Maikel Nadolski)
3. [D4007R0](https://wg21.link/p4007) - Does std::execution Need More Time? (Vinnie Falco, Mungo Gill)
