---
title: "Test Paper"
document: D9999R0
date: 2026-04-15
audience: LEWG
reply-to:
  - "Test Author <test@example.com>"
---

## Abstract

A test paper for citation normalization.

---

## 1. Disclosure

The author provides information.

---

## 2. Body

First we cite [P2300R10](https://www.open-std.org/p2300r10.html)<sup>[3]</sup>, then [Boost.URL](https://github.com/boostorg/url)<sup>[1]</sup>.

Later we reference [cppcoro](https://github.com/lewissbaker/cppcoro)<sup>[2]</sup> again and [P2300R10](https://www.open-std.org/p2300r10.html)<sup>[3]</sup> a second time.

```cpp
// Citation inside code: <sup>[3]</sup> should be untouched.
int x = 0;
```

---

## References

[1] Boost.URL, https://github.com/boostorg/url

[2] cppcoro, Lewis Baker, https://github.com/lewissbaker/cppcoro

[3] P2300R10, std::execution, https://www.open-std.org/p2300r10.html

[4] Orphan reference that nobody cites
