---
title: "Viable Candidates for Expansion-Iterable"
document: DXXXXR0
date: 2026-02-20
reply-to:
  - "TBD"
audience: EWG
---

## Abstract

CWG 3123 fixes the expansion-iterable check in 8.7 [stmt.expand] paragraph 3. The current wording checks whether ADL finds any `begin`/`end` declaration. The proposed resolution checks whether ADL finds a viable candidate. CWG approved the resolution and requested EWG input (paper issue #2615).

---

## Revision History

### R0: March 2026 (pre-Croydon mailing)

* Initial version.

---

## 1. The Defect

[P1306R5](https://wg21.link/p1306r5)<sup>[1]</sup> ("Expansion Statements") introduced three forms of expansion statement: enumerating, iterating, and destructuring. The choice between iterating and destructuring depends on whether the expression is expansion-iterable, defined in 8.7 [stmt.expand] paragraph 3:

> An expression is expansion-iterable if it does not have array type and either
>
> - `begin-expr` and `end-expr` are of the form `E.begin()` and `E.end()`, or
> - argument-dependent lookups for `begin(E)` and for `end(E)` each find at least one function or function template.

Consider `std::tuple<int, double>`. The associated namespace is `std`. Namespace `std` declares:

```cpp
template<class C>
  constexpr auto begin(C& c) -> decltype(c.begin());

template<class C>
  constexpr auto end(C& c) -> decltype(c.end());
```

ADL finds these function templates. The wording requires "at least one function or function template" - condition satisfied. The tuple is classified as expansion-iterable, even though `std::tuple` has no `begin()` or `end()` member and calling `std::begin` on it is ill-formed.

The wording was modeled on a pattern that works well elsewhere: 7.6.2.4 [expr.await] uses "find at least one declaration" for `await_transform`, and 9.7 [dcl.struct.bind] uses it for member `get`. In both cases the lookup is class-scoped - the type author controls what names are visible. ADL is different. It searches entire associated namespaces, where unrelated declarations like `std::begin` reside alongside the type.

## 2. The Fix

The proposed resolution replaces "function or function template" with "viable candidate" (12.2.3 [over.match.viable]). For `std::tuple<int, double>`, template argument deduction for `std::begin(C&)` succeeds, but substitution into the trailing return type `decltype(c.begin())` fails in the immediate context. SFINAE removes the template from the candidate set, no viable candidates remain, and the tuple falls through to destructuring.

This is the same overload resolution that range-based for (8.6.5 [stmt.ranged]) already performs on `begin`/`end` names found by ADL<sup>[3]</sup>.

## 3. Alternatives

Four other approaches were considered.

**Restrict to member `begin`/`end`.** This breaks types that customize iteration through ADL free functions, which is the standard customization pattern for ranges.

**Require an opt-in trait.** A new `expansion_iterable` specialization would require every existing range type to opt in. The defect is a single phrase of wording; a new customization point is not warranted.

**Carve out tuple-like types.** Excluding types with a `tuple_size` specialization addresses the motivating case but not the general problem: any type whose associated namespaces contain unrelated `begin`/`end` declarations is affected.

**Check argument count.** `std::begin(C&)` takes one argument. A tuple expression is one argument. The check passes and the problem persists.

## 4. Instantiation

CWG noted three potential drawbacks of checking viability, each related to the template instantiations that overload resolution can trigger.

**Cost.** The viability check may instantiate templates that the current wording does not. In practice, the check is a subset of what range-for already performs on the same `begin`/`end` names - the incremental work is small.

**Hard errors.** A viability check can trigger template instantiation outside the immediate context, producing a hard error rather than a substitution failure. For `std::begin` and `std::end`, the trailing return type `decltype(c.begin())` keeps the failure in the immediate context. User-defined `begin`/`end` templates with weaker constraints could cause hard errors, but the same hazard already exists in range-for, concept checks, and `requires` expressions.

**Observability via reflection.** Template instantiation is permanent, and [P2996R10](https://wg21.link/p2996r10)<sup>[2]</sup> ("Reflection for C++26") makes it observable. Every concept check and every SFINAE-based trait already produces instantiations with the same observability. If implicit instantiation observability is a problem, the solution belongs in the reflection specification, not in expansion statements.

## 5. Proposed Wording

This paper endorses the proposed resolution approved by CWG on 2026-01-23.

Change in 8.7 [stmt.expand] paragraph 3 as follows:

> An expression is expansion-iterable if it does not have array type and either
>
> - `begin-expr` and `end-expr` are of the form `E.begin()` and `E.end()`, or
> - argument-dependent lookups for `begin(E)` and for `end(E)` each find at least one ~~function or function template~~ viable candidate (12.2.3 [over.match.viable]).

---

# Acknowledgements

This document is written in Markdown and depends on the extensions in [`pandoc`](https://pandoc.org/MANUAL.html#pandocs-markdown) and [`mermaid`](https://github.com/mermaid-js/mermaid), and we would like to thank the authors of those extensions and associated libraries.

The author thanks Jens Maurer and CWG for identifying the defect and crafting the proposed resolution.

---

# References

1. [P1306R5](https://wg21.link/p1306r5) - "Expansion Statements" (Dan Katz, Andrew Sutton, Sam Goodrick, Daveed Vandevoorde, 2025). https://wg21.link/p1306r5
2. [P2996R10](https://wg21.link/p2996r10) - "Reflection for C++26" (Wyatt Childers, Peter Dimov, Dan Katz, Barry Revzin, Andrew Sutton, Daveed Vandevoorde, Faisal Vali, 2025). https://wg21.link/p2996r10
3. [C++ Working Draft](https://eel.is/c++draft/) - (Richard Smith, ed.). Sections: [stmt.expand], [stmt.ranged], [iterator.range], [over.match.viable], [dcl.struct.bind], [expr.await]. https://eel.is/c++draft/
