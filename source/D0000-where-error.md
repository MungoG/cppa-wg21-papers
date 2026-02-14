---
title: "Where Does the Error Code Go?"
document: D0000
date: 2026-02-14
reply-to: "Vinnie Falco <vinnie.falco@gmail.com>"
audience: WG21
---

## Abstract

Every I/O operation in [Boost.Asio](https://www.boost.org/doc/libs/release/doc/html/boost_asio.html) completes with `void(error_code, size_t)` - an error code and a byte count, delivered together. The sender/receiver model in [P2300R10](https://wg21.link/p2300) provides three separate completion channels: `set_value`, `set_error`, and `set_stopped`. This paper asks which channel the error code belongs in, and what happens when libraries disagree.

---

## 1. The I/O Completion Signature

Boost.Asio's stream operations complete with both an error code and a transfer count:

```cpp
// Boost.Asio completion signature for stream I/O
void(boost::system::error_code ec, std::size_t bytes_transferred)
```

This signature reflects reality. A `read` may transfer 47 bytes and then encounter EOF. A `write` may transfer 1000 bytes out of 4096 before the connection resets. The error code and the byte count are not alternatives - they are returned together, because partial success is the normal case in I/O.

[P2762R2](https://wg21.link/p2762) ("Sender/Receiver Interface for Networking") preserves this pattern. Dietmar Kuhl's networking sender completes with `set_value(receiver, error_code, size_t)`.

---

## 2. The Classification Problem

[P2300R10](https://wg21.link/p2300) provides three completion channels:

```cpp
set_value(receiver, values...);   // success
set_error(receiver, error);       // failure
set_stopped(receiver);            // cancellation
```

An I/O operation that returns `(error_code, size_t)` must choose: does the error code go through `set_value` or `set_error`?

**Option A: Error code in `set_value`.**

```cpp
// The Asio/P2762 approach
set_value(std::move(receiver), ec, bytes_transferred);
```

Both values are delivered together. The caller inspects `ec`. This matches twenty years of Asio practice.

**Option B: Error code in `set_error`.**

```cpp
// Separate channels
if (!ec)
    set_value(std::move(receiver), bytes_transferred);
else
    set_error(std::move(receiver), ec);
```

The byte count is lost on error. Partial success becomes indistinguishable from total failure.

Chris Kohlhoff identified this problem in [P2430R0](https://open-std.org/jtc1/sc22/wg21/docs/papers/2021/p2430r0.pdf) ("Partial success scenarios with P2300"), presented to LEWG/SG1 in August 2021:

> "Due to the limitations of the set_error channel (which has a single 'error' argument) and set_done channel (which takes no arguments), partial results must be communicated down the set_value channel."

P2430R0 concludes that Option A is the only viable path for I/O. But this conclusion has consequences for generic sender algorithms.

---

## 3. What `when_all` Does

[P2300R10](https://wg21.link/p2300) specifies that `when_all` cancels remaining children when any child completes with `set_error` or `set_stopped` ([[exec.when.all]](https://eel.is/c++draft/exec.when.all)).

Consider two concurrent I/O operations:

```cpp
auto result = co_await when_all(
    socket_a.read_some(buf_a),
    socket_b.read_some(buf_b));
```

If `socket_a` reports EOF through `set_error(receiver, eof)`, `when_all` cancels `socket_b`. The read on `socket_b` is aborted. The bytes already transferred from `socket_a` are lost.

If `socket_a` reports EOF through `set_value(receiver, eof, 47)`, `when_all` treats it as success. Both operations complete. The caller sees the 47 bytes and the EOF code.

Same underlying event. The `when_all` behavior depends entirely on which channel the library author chose for the error code.

---

## 4. Mixing Conventions

The problem is sharpest when two libraries are composed in the same `when_all`. Suppose Library A follows the Asio convention (error in `set_value`) and Library B follows the "pure" sender convention (error in `set_error`):

```cpp
// Library A (Asio-style): EOF delivered through set_value
auto read_a = lib_a::async_read_some(socket_a, buf_a);
// completes: set_value(receiver, error_code::eof, 47)

// Library B (pure sender): EOF delivered through set_error
auto read_b = lib_b::async_read_some(socket_b, buf_b);
// completes: set_error(receiver, error_code::eof)
```

```cpp
auto result = co_await when_all(read_a, read_b);
```

If `read_b` hits EOF first: `when_all` sees `set_error`, cancels `read_a`, propagates the error. The 47 bytes that `read_a` may have already transferred are lost.

If `read_a` hits EOF first: `when_all` sees `set_value`, keeps waiting for `read_b`. No cancellation. The caller eventually gets both results.

Same event. Same `when_all`. Different behavior depending on which socket hit EOF first and which library convention that socket's sender follows. This is not a hypothetical - it is what happens when an application composes senders from two libraries that made different classification choices.

---

## 5. The Broader Problem

A generic sender algorithm cannot know which convention a given I/O sender follows. Consider a retry algorithm:

```cpp
auto result = co_await retry_on_error(
    socket.read_some(buf));
```

If the I/O sender reports errors through `set_error`, `retry_on_error` sees them and can retry. If the I/O sender reports errors through `set_value`, `retry_on_error` never fires - the error is invisible to it.

The sender/receiver model's three-channel design assumes a clean classification: values are values, errors are errors. I/O does not have a clean classification. An `error_code` is a status report. EOF is not a failure - it is information. A partial write is not an error - it is a progress report. The classification decision is forced on every I/O library author, and there is no convention for which answer is correct.

---

## 6. Questions

- [P2762R2](https://wg21.link/p2762) delivers `error_code` through `set_value`. Should all I/O senders follow this convention? If so, how do generic error-handling algorithms like `retry` or `log_errors` distinguish I/O errors from successful completions?

- If I/O senders should use `set_error` instead, what happens to the bytes already transferred? Is partial success expressible through `set_error`?

- How does `when_all` behave correctly with I/O senders when the ecosystem has not agreed on which channel carries the error code?

---

## 7. Conclusion

We suggest these questions be answered before `std::execution` ships in C++26.

---

## References

1. [P2300R10](https://wg21.link/p2300) - std::execution (Michal Dominiak, Georgy Evtushenko, Lewis Baker, Lucian Radu Teodorescu, Lee Howes, Kirk Shoop, Eric Niebler)
2. [P2430R0](https://open-std.org/jtc1/sc22/wg21/docs/papers/2021/p2430r0.pdf) - Partial success scenarios with P2300 (Chris Kohlhoff)
3. [P2762R2](https://wg21.link/p2762) - Sender/Receiver Interface for Networking (Dietmar Kuhl)
4. [Boost.Asio](https://www.boost.org/doc/libs/release/doc/html/boost_asio.html) - Asynchronous I/O library (Chris Kohlhoff)
