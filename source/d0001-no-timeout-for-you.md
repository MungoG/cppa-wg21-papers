---
title: "No Timeout For You"
document: D0001R0
date: 2026-02-18
reply-to:
  - "Vinnie Falco <vinnie.falco@gmail.com>"
  - "Mungo Gill <mungo.gill@me.com>"
audience: SG1, LEWG
---

## Abstract

C++26's `std::execution` provides `when_all` but not `when_any` or `stop_when`. Timeouts cannot be expressed with the standard sender algorithms.

---

## Revision History

### R0: March 2026 (pre-Croydon mailing)

* Initial version.

---

## 1. The Motivating Example

[P2300R7](https://wg21.link/p2300r7)<sup>[1]</sup> ("`std::execution`," Section 1.3) presents this example of composed cancellation:

```cpp
sender auto composed_cancellation_example(auto query) {
  return stop_when(
    timeout(
      when_all(
        first_successful(
          query_server_a(query),
          query_server_b(query)),
        load_file("some_file.jpg")),
      5s),
    cancelButton.on_click());
}
```

This example uses four algorithms: `stop_when`, `timeout`, `first_successful`, and `when_all`.

Of the four, one shipped.

---

## 2. What C++26 Provides

The C++26 working draft<sup>[2]</sup> defines the following sender adaptors in [[exec]](https://eel.is/c++draft/exec)<sup>[3]</sup>:

| Sender factories   | Sender adaptors            | Sender consumers             |
|---------------------|----------------------------|------------------------------|
| `just`              | `starts_on`                | `sync_wait`                  |
| `just_error`        | `continues_on`             | `sync_wait_with_variant`     |
| `just_stopped`      | `on`                       |                              |
| `read_env`          | `schedule_from`            |                              |
| `schedule`          | `then`                     |                              |
|                     | `upon_error`               |                              |
|                     | `upon_stopped`             |                              |
|                     | `let_value`                |                              |
|                     | `let_error`                |                              |
|                     | `let_stopped`              |                              |
|                     | `bulk`                     |                              |
|                     | `split`                    |                              |
|                     | `when_all`                 |                              |
|                     | `when_all_with_variant`    |                              |
|                     | `into_variant`             |                              |
|                     | `stopped_as_optional`      |                              |
|                     | `stopped_as_error`         |                              |

`when_any` is absent. `stop_when` is absent. No timer facility exists.

`stop_when` appeared in [P2300R7](https://wg21.link/p2300r7)<sup>[1]</sup> (April 2023) and was removed before [P2300R10](https://wg21.link/p2300r10)<sup>[4]</sup> (June 2024). `when_any` exists in the [stdexec](https://github.com/NVIDIA/stdexec)<sup>[5]</sup> reference implementation but was never proposed for standardization across ten revisions of P2300.

---

## 3. What Networking Needs

Every production async framework provides a timeout facility:

| Framework                                                                                 | Timeout mechanism                    |
|-------------------------------------------------------------------------------------------|--------------------------------------|
| [Boost.Asio](https://www.boost.org/doc/libs/release/doc/html/boost_asio.html)<sup>[6]</sup> | `steady_timer::async_wait`           |
| [Go](https://pkg.go.dev/context#WithTimeout)<sup>[7]</sup>                                | `context.WithTimeout` + `select`     |
| [Trio](https://trio.readthedocs.io/en/stable/reference-core.html#cancellation-and-timeouts)<sup>[8]</sup> | `move_on_after`, `fail_after`        |
| [Kotlin](https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/with-timeout.html)<sup>[9]</sup> | `withTimeout` + `select`             |
| [Tokio](https://docs.rs/tokio/latest/tokio/time/fn.timeout.html)<sup>[10]</sup>           | `tokio::time::timeout`               |
| [Swift](https://developer.apple.com/documentation/swift/taskgroup)<sup>[11]</sup>          | `TaskGroup` + `race`                 |

A timeout is a race: run the operation and a timer concurrently, take whichever finishes first. This is the dual of `when_all`. Every framework in the table provides both.

---

## 4. The Code That Cannot Be Written

With a coroutine library that provides `with_timeout`:

```cpp
auto [ec, n] = co_await with_timeout(5s, sock.async_read(buf));
if (ec == errc::timed_out) { /* handle timeout */ }
```

With C++26 `std::execution`, the natural expression would be:

```cpp
auto work = when_any(async_read(sock, buf), sleep(5s));
```

This does not compile. `when_any` is not in C++26.

### 4.1 The Workaround

The [stdexec](https://github.com/NVIDIA/stdexec)<sup>[5]</sup> issue tracker contains a [user report](https://github.com/NVIDIA/stdexec/issues/1169)<sup>[12]</sup> of attempting to use the non-standard `when_any`. The user discovered that inner operations must manually query the stop token and poll it in a loop:

```cpp
stdexec::sender auto job1 = stdexec::schedule(sch)
  | stdexec::let_value([]{ return stdexec::get_stop_token(); })
  | stdexec::then([](auto const& token){
      int i = 0;
      while (!token.stop_requested()) {
          do_work(i++);
          std::this_thread::sleep_for(200ms);
      }
  });
```

Manual stop-token polling inside lambdas is the workaround. It is the manual orchestration that senders were designed to eliminate.

---

## 5. The Cancelled Read

Suppose `when_any` were added tomorrow. A timeout cancels a read mid-flight. The read has transferred 47 bytes. The three completion channels offer no correct path for the result:

- `set_stopped()` carries no arguments. The 47 bytes are lost.
- `set_error(ec)` carries only the error code. The 47 bytes are lost.
- `set_value(ec, n)` preserves both, but `upon_error` and `upon_stopped` become unreachable for I/O senders. Algorithms that dispatch on channel, such as `when_all` cancelling siblings on error, can no longer distinguish failure from success.

[P4007R0](https://wg21.link/p4007r0)<sup>[13]</sup> ("Senders and C++") Section 3 documents the full analysis. The three-channel completion model and I/O completion semantics are structurally incompatible. Five choices exist; none is correct.

The timeout gap and the channel gap are independent. Closing either one does not close the other.

---

## 6. Counterarguments

**"`when_any` is coming."** `stop_when` was proposed in [P2175R0](https://wg21.link/p2175r0)<sup>[14]</sup> ("Composable cancellation for sender-based async operations," 2020) and still appeared in [P2300R7](https://wg21.link/p2300r7)<sup>[1]</sup> (2023). It was removed before [P2300R10](https://wg21.link/p2300r10)<sup>[4]</sup> (2024). The motivating example still uses it. No replacement was proposed.

**"Manual stop-token wiring works."** It does. This is what senders were supposed to replace.

**"Networking is not yet standardized."** SG4 polled at Kona (November 2023) on [P2762R2](https://wg21.link/p2762r2)<sup>[15]</sup> ("Sender/Receiver Interface For Networking"):

> *"Networking should support only a sender/receiver model for asynchronous operations; the Networking TS's executor model should be removed"*
>
> | SF | F | N | A | SA |
> |----|---|---|---|----|
> |  5 | 5 | 1 | 0 |  1 |

The mandated model cannot express a timeout.

---

## 7. Suggested Straw Polls

> 1. "Is `std::execution` without a selection algorithm (`when_any`, `stop_when`, or equivalent) suitable as the mandated model for C++ networking?"

> 2. "WG21 should not mandate senders for networking until the sender algorithm set can express timeouts."

---

# Acknowledgements

This document is written in Markdown and depends on the extensions in [`pandoc`](https://pandoc.org/MANUAL.html#pandocs-markdown) and [`mermaid`](https://github.com/mermaid-js/mermaid), and we would like to thank the authors of those extensions and associated libraries.

---

## References

1. [P2300R7](https://wg21.link/p2300r7) - "`std::execution`" (Niebler, Dominiak, Baker, et al., 2023). https://wg21.link/p2300r7
2. C++26 Execution control library - cppreference. https://en.cppreference.com/w/cpp/execution.html
3. C++26 Working Draft, [exec] - eel.is. https://eel.is/c++draft/exec
4. [P2300R10](https://wg21.link/p2300r10) - "`std::execution`" (Niebler, Dominiak, Baker, et al., 2024). https://wg21.link/p2300r10
5. stdexec - NVIDIA reference implementation of `std::execution`. https://github.com/NVIDIA/stdexec
6. Boost.Asio documentation. https://www.boost.org/doc/libs/release/doc/html/boost_asio.html
7. Go `context.WithTimeout` documentation. https://pkg.go.dev/context#WithTimeout
8. Trio cancellation and timeouts documentation. https://trio.readthedocs.io/en/stable/reference-core.html#cancellation-and-timeouts
9. Kotlin `withTimeout` documentation. https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/with-timeout.html
10. Tokio `timeout` documentation. https://docs.rs/tokio/latest/tokio/time/fn.timeout.html
11. Swift `TaskGroup` documentation. https://developer.apple.com/documentation/swift/taskgroup
12. stdexec issue #1169 - "`when_any` question". https://github.com/NVIDIA/stdexec/issues/1169
13. [P4007R0](https://wg21.link/p4007r0) - "Senders and C++" (Falco, Gill, 2026). https://wg21.link/p4007r0
14. [P2175R0](https://wg21.link/p2175r0) - "Composable cancellation for sender-based async operations" (Baker, 2020). https://wg21.link/p2175r0
15. [P2762R2](https://wg21.link/p2762r2) - "Sender/Receiver Interface For Networking" (K&uuml;hl, 2023). https://wg21.link/p2762r2
