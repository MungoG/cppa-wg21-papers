---
title: "C++ Design By Contract: Minimum Language, Maximum Library"
document: P4324R0
date: 2026-08-01
intent: info
audience: EWG, SG21
reply-to:
  - "Vinnie Falco <vinnie.falco@gmail.com>"
  - "Ville Voutilainen <ville.voutilainen@gmail.com>"
---

## Abstract

In C++26 (P2900), a contract predicate is evaluated with the entities it names treated as `const`, and an exception that escapes the predicate becomes a contract violation rather than propagating. The language currently fixes both.

This paper makes them per-assertion policies. The language does two things: it binds a predicate to a declaration or statement, and it names an assertion-control object that governs the predicate. That object is an ordinary library type; it carries the constification and exception-handling policies and selects the evaluation semantic. The semantics join the violation handler and violation object that P2900 already specifies as library code. The `pre`, `post`, and `contract_assert` syntax is unchanged in the common case. The design synthesizes four published proposals<sup>[4]</sup><sup>[5]</sup><sup>[2]</sup><sup>[3]</sup> on that library surface.

---

## Revision History

### R0: August 2026

- Initial version.

---

## 1. Introduction

C++26 contracts (P2900R14<sup>[1]</sup>) fix two behaviors in every predicate: the predicate is constified, and an exception that escapes it becomes a contract violation. Both are language rules, the same for every assertion.

This paper makes constification and exception propagation per-assertion policies. The language does two things: it binds a predicate to a declaration or statement, and it names an assertion-control object that governs it. The object exposes a few compile-time properties the compiler reads during code generation, and one call operator it invokes on a violation. The common syntax is unchanged, `pre(cond)`, `post(r: cond)`, `contract_assert(cond)`; an assertion that needs a specific policy names it, as in `pre<review>(cond)`.

Section 2 motivates the two policies. Section 3 specifies the design as a three-step compiler algorithm over the control object's compile-time properties. Section 4 relates the design to the C++26 working draft and the cost of changing it now. Section 5 turns to a separate question, the ownership of core-language undefined behavior, and places contracts and profiles as independent peers so that no such behavior is routed through the contract-violation handler. Section 6 gives a reference-implementation plan on existing compiler forks.

The design rests on one principle: the language carries only what only the language can do, and everything else goes in a library. Appendix B compares both designs against that principle and the others in Stroustrup's *The Design and Evolution of C++*<sup>[22]</sup>. Appendix A maps the design onto the prior work it unifies.

---

## 2. Motivation

P2900 fixes constification and exception-to-violation conversion as language rules, the same for every predicate. Each has a cost in ordinary code: a predicate can silently call a different function than the body, and a predicate that reuses a throwing validation routine loses the exception the caller would otherwise catch. This section shows both, and why the design makes the two per-assertion policies with constification off by default.

### 2.1 Constification silently changes which function a predicate calls

A programmer who knows nothing about constification expects a predicate to mean what the same expression means in the function body. Under P2900 it need not. Constifying the predicate treats the object as `const`, so an overload set with a `const` and a non-`const` member binds the `const` overload in the predicate and the non-`const` one in the body. The two can be different functions, and nothing reports the difference.

```cpp
struct handle {
  bool ready();        // non-const overload
  bool ready() const;  // const overload: a different function
};

void use(handle& h)
  pre(h.ready())                 // P2900: constified, binds handle::ready() const
{
  if (h.ready()) { /* ... */ }   // body: binds handle::ready()
}
```

For a const-correct type the two overloads observe the same state, so the result is the same and only the called function differs; the cost is that "the same sequence of tokens in immediate proximity ... mean different things," which P3071R1<sup>[23]</sup>, the proposal that made predicate entities `const`, itself calls "confusing." For a type whose overloads differ, the predicate checks something other than what the body relies on: "overload resolution might quietly invoke different functions in the two contexts"<sup>[8]</sup>. P2900's constification does not report either case. It errors only when no `const` overload exists at all, and silently switches when one does, so its alarm is inconsistent.

The design makes constification a per-assertion policy (Section 3.2) that defaults off, so a predicate means what the same expression means in the body, and an author who wants it opts in on the control object. This default is the more self-contained one. P2900 applies a single rule to every predicate, but the rule makes the predicate bind a different function than the same tokens in the body, so the predicate reads like ordinary code and is not. With the default here the predicate is ordinary code, with no separate rule to learn. Naming a control object, as in `pre<review>(cond)`, is a visible opt-in, the same indirection as an allocator or a comparator.

Consistency also serves static analysis, and the evidence cuts both ways. An analyzer applies value numbering, which gives equivalent computations the same value number<sup>[24]</sup>. A constified predicate binds a different function than the body, so the two receive different value numbers and the analyzer cannot establish that the predicate checks what the body relies on; a tool aware of constification still sees two functions, so awareness does not close the gap. Off by default, the predicate and the body share a value number. Constification has the opposite benefit: it lets an analyzer trust that a parameter passed by `const` reference is not modified in the predicate, which P3261R0<sup>[7]</sup> gives as a reason to keep it. The two pull against each other, so the design leaves the choice per assertion. A build-wide flag cannot, since different assertions want different answers, and tying a predicate's meaning to a build-wide switch is the semantic instability P2834R1<sup>[25]</sup> calls "anathema."

### 2.2 Converting an escaping exception into a violation hides the real error

A predicate can call a function that throws. Under P2900 an exception that escapes the predicate becomes a contract violation instead of propagating.

```cpp
bool validate(const settings&);   // throws std::invalid_argument on malformed input

void configure(const settings& s)
  pre(validate(s))   // P2900: a thrown std::invalid_argument becomes a contract violation
{
  // ...
}
```

The caller can no longer catch the `std::invalid_argument` it would catch from the body, and the backtrace points into the contract machinery rather than the throw site, which an implementer records as "useless"<sup>[10]</sup>. Because any predicate might throw, the compiler must "generate the correct instructions for exception handling around every contract assertion"<sup>[10]</sup>, and that scaffolding is what raises the `noexcept` trilemma P4308R0<sup>[11]</sup> describes. With propagation as the default the exception behaves as it does in the body, `noexcept` keeps its meaning, and an assertion that cannot throw carries no scaffolding.

The use cases are ordinary. A precondition often reuses a validation routine written for the body, and such routines commonly report malformed input by throwing; propagation lets the caller catch that exception where it would catch it from the body, instead of losing it to the violation handler. The value of an exception from a contract check reaching the caller is on record for the related case of a throwing handler, where a long-running service catches at a request boundary and abandons one suspect request while it keeps serving the rest (P3318R0<sup>[26]</sup>; P4308R0<sup>[11]</sup> §5.1).

### 2.3 What constification protects, and why it is a policy not a rule

Constification is not without value: it catches a mutating call in a predicate. `std::unordered_map::operator[]` is the clearest case, because it is non-`const` and inserts a value-initialized element when the key is absent.

```cpp
void f(std::unordered_map<int, int>& m, int k)
  pre(m[k] == 0)   // without constification this inserts into m; constification rejects it
{
  // ...
}
```

Here constification is the feature working as intended: it rejects a predicate that would mutate the container, a protection P3071R1<sup>[23]</sup> was written to provide. The design keeps that protection available rather than removing it. An author who wants it sets `constify` to true on the control object. Today the escape from constification runs the other way, per expression: an author writes a `const_cast` wrapper, which P3261R0<sup>[7]</sup> notes "clearly conveys that the author ... is intentionally working around const-ification," or reaches for the `mutable` specifier P3592R0<sup>[8]</sup> proposes to turn it off for a whole predicate. The control object reaches the same place per assertion, with no cast in the expression, and it carries the exception and semantic policies through one object rather than one keyword per behavior.

There is a real case for making constification the default. Contracts are a bug-finding feature, and requiring a predicate to hold on a `const` object makes an author confront code that is not const-correct: where no `const` path exists, constification errors rather than let the predicate bind a mutating call. On that view, awareness of a problem is worth more than agreement between the predicate and the body. The case has force, and the design does not dismiss it.

The alarm is incomplete, however. Constification errors only when no `const` overload exists; when both exist, it switches silently and reports nothing (Section 2.1). It catches the ill-formed case and misses the divergence case, so it does not deliver the awareness the argument wants. A programmer who has not learned about constification reads the predicate as the body, and the default matches that reading. Which tradeoff to prefer is an engineering judgment; the design makes consistency the default and leaves constification one control-object property away.

Making constification a control-object property adds a choice and removes nothing. An author who wants the guardrail keeps it with `constify` set to true; one who wants the predicate to read as ordinary code takes the default. P2900's single rule serves neither the second author nor the author of a non-const-correct API, and the per-assertion property serves both. The objection that configuring constification from outside the predicate is worse than P2900's fixed rule, whatever the default, is a different preference: one uniform evaluation rule, against the predicate meaning what ordinary code means. That is a difference in values, not a defect in either position. This design takes the second view and leaves the first available per assertion.

This also keeps the question out of the "remove it or keep it" frame the committee has already settled: at Hagenberg, EWG declined to remove constification (Section 4). This design does not remove constification. It relocates the choice into a per-assertion policy and makes consistency the default.

### 2.4 A diagnosing default is not viable

Between silent acceptance and off, a third option suggests itself: diagnose divergence. The compiler resolves the predicate twice, once constified and once as written, compares the two, and rejects the assertion when they bind different functions, leaving the author to choose `on` or `off` for that assertion. When the two resolutions agree, no choice arises. This extends the alarm of Section 2.1 from the ill-formed case to the divergence case without silent switching, and it was the design's default in an earlier draft, with a prototype on the GCC contracts fork that handled the simple case, where both overloads resolve and the two trees can be compared.

It does not work in general. Resolving the constified form is not a side-effect-free query. It can instantiate templates, and it can bind a deleted or otherwise ill-formed overload, and that ill-formedness is not always in a SFINAE context. When it is not, the error is hard and the translation fails before the comparison runs. Tentative resolution of the constified predicate then turns a program that would compile into one that does not, with an error the author did not write and cannot suppress from the control object.

This is the reason C++ has no `if (compiles(expr))` construct: asking whether code is well-formed can itself make the translation ill-formed. The design drops the diagnosing default and keeps the binary policy, off by default (Section 3.2). It records the attempt here so a later proposal starts from the result rather than re-deriving it.

---

## 3. Design

This section specifies the design as an algorithm, not wording. It covers what the language guarantees, the compile-time interface the control object exposes, the syntax a programmer writes, and the library code that handles a violation. The compiler reads a few compile-time properties off the control-object type to decide what code to generate, and on a violation it makes one ordinary call on the control object. Everything else is library code.

### 3.1 The two jobs

`pre`, `post`, and `contract_assert` do two things. Binding a predicate to a declaration or statement is the first, and P2900 already does it. Naming an assertion-control object that governs the predicate is the second, and it is new: each assertion carries a control object whose compile-time properties the compiler reads.

The compiler decides whether and when the predicate runs. An ignored assertion does not evaluate the predicate (Section 3.8); an enforced one does. That is P2900 behavior, not a change. It is also why the predicate is part of the construct rather than an argument to a function, which would evaluate it before the call. The control object's properties are read at compile time, before any evaluation, so they cost nothing at run time and do not conflict with an ignored assertion evaluating nothing.

This is also why the control object names a type rather than wrapping the predicate. An in-predicate wrapper such as `unconst(pred(x))` cannot be an ordinary function: the call would evaluate `pred(x)` first, already constified, before `unconst` ran, so suppressing constification in the expression needs a language construct. A compile-time property on the type is read before evaluation and needs no new mechanism.

### 3.2 The three-step compiler algorithm

For each contract assertion with control object of type `T` and a build-selected configuration `cfg`, the compiler performs three steps.

```text
1. If T::is_ignored(cfg) is true:
       if T::assumable is true, emit an optimizer assumption of the predicate;
       otherwise emit nothing.
       Stop.
2. If T::constify is true, evaluate the constified predicate; otherwise evaluate the predicate as written.
3. If the predicate is violated, call the control object with the violation data.
       If the call returns terminate, contract-terminate; otherwise proceed.
```

The configuration `cfg` is a compile-time constant within a translation unit, selected by an implementation-defined mechanism, as P2900's build-time semantic selection is chosen today. The three steps read three compile-time properties (`is_ignored`, `constify`, `assumable`) and make one runtime call. Nothing else about the assertion is a language rule.

### 3.3 The assertion-control concept

An assertion-control object is a stateless type exposing three compile-time members and a call operator.

```cpp
namespace std::contracts {

enum class evaluation_config : unsigned {
  ignore = 0, observe = 1, enforce = 2, quick_enforce = 3,
  // [4 .. 0xFFFF] reserved to the standard; [0x1'0000 ..] reserved to vendors and users
};

enum class violation_response { proceed, terminate };

template <class T>
concept assertion_control =
  std::is_empty_v<T> &&
  requires (T c, const char* comment, std::source_location loc, evaluation_config cfg) {
    { T::is_ignored(cfg)      } -> std::same_as<bool>;          // step 1
    { T::constify             } -> std::convertible_to<bool>;   // step 2
    { T::assumable            } -> std::convertible_to<bool>;   // optimizer
    { c(comment, loc, cfg)    } -> std::same_as<violation_response>;  // step 3
  };

}
```

The compiler reads only these member names and the call-operator signature. It does not depend on the contents of the `<contracts>` header. Gustafsson gives the reason: a dependency from the core compiler to the contents of a standard library header "is novel and something we usually want to avoid"<sup>[4]</sup>. P3968R0<sup>[4]</sup> reaches the same shape with boolean members named `constify`, `ignorable`, and `assumable`; P3400R4<sup>[5]</sup> reaches it with a `const` member the implementation may "query the value at any time."

### 3.4 Surface syntax

The control object is named with a template argument, and the common case has a default.

```cpp
void f(int x) pre(x >= 0);                 // uses std::contracts::default_v
void g(int* p) pre<review>(p != nullptr);  // names a specific control object
int  h() post(r: r >= 0);
contract_assert(index < size);
```

`pre(cond)` is `pre<std::contracts::default_v>(cond)`, and likewise for `post` and `contract_assert`. A program that never names a control object writes the P2900 syntax and gets a checked, terminating semantic by default.

### 3.5 The library owns the response

The call operator is where a violation is handled. It branches on the configuration and does whatever that configuration means.

```cpp
namespace std::contracts {

struct default_control {
  static constexpr bool is_ignored(evaluation_config cfg) { return cfg == evaluation_config::ignore; }
  static constexpr bool constify  = false;
  static constexpr bool assumable = false;

  violation_response operator()(const char* comment, std::source_location loc,
                                evaluation_config cfg) const {
    switch (cfg) {
      case evaluation_config::observe:
        invoke_default_contract_violation_handler(/* built from comment, loc */);
        return violation_response::proceed;
      case evaluation_config::enforce:
        invoke_default_contract_violation_handler(/* ... */);
        return violation_response::terminate;
      case evaluation_config::quick_enforce:
        return violation_response::terminate;
      default:
        return dispatch_extended(cfg, comment, loc);   // vendor and user range
    }
  }
};
inline constexpr default_control default_v{};

}
```

The branch table is a library function rather than a language rule or a program-wide compiler-owned handler. P2900R14 already places the violation handler and the violation object in the library: the handler is "a function named `::handle_contract_violation`," the library provides `invoke_default_contract_violation_handler`, and the properties of a `contract_violation` "are all accessed by const, non-virtual member functions"<sup>[1]</sup>. Moving the branch that selects among semantics into the same library is a small step from what P2900 already specifies.

A build-time flag selects which semantic applies; a control object defines what that semantic does. Without the control object, every semantic definition is a language rule, and adding or changing one requires a standard revision. With it, the definition is library code.

Extension is intrinsic. A platform or a user adds a configuration value in the reserved range and a library function that recognizes it; the compiler passes the selected `cfg` through without knowing its meaning. The semantics the committee has added or is debating illustrate the point. `quick_enforce` was added to P2900 by committee action; `noexcept-observe` and `noexcept-enforce` are proposed now in P4308R0 Option C<sup>[11]</sup> and implemented in experimental GCC and Clang branches. Under this design each is a library value, field-tested and then standardized from experience.

### 3.6 Configuration vocabulary

The standard defines the baseline vocabulary `ignore`, `observe`, `enforce`, and `quick_enforce`. The `evaluation_config` representation is an open enumeration with a reserved standard range and a reserved vendor-and-user range. A platform can define additional values without a language change, and library dispatch handles values it does not recognize by forwarding to the platform function. This delivers both behavioral configurability (what a semantic does) and vocabulary extensibility (adding a semantic), library-side.

### 3.7 Worked control objects

Two control objects show that behavior the committee has treated as language design is expressible as library code with no language change.

```cpp
// Log and continue at the library-defined level, always checked. The shape of
// Bloomberg's bsls_review, expressed as a control object.
struct review {
  static constexpr bool is_ignored(evaluation_config) { return false; }
  static constexpr bool constify  = true;
  static constexpr bool assumable = false;
  violation_response operator()(const char*, std::source_location, evaluation_config) const {
    log_to_telemetry(/* ... */);
    return violation_response::proceed;
  }
};

// Guaranteed-enforced and optimizable, the strand of P4005R0.
struct mandatory {
  static constexpr bool is_ignored(evaluation_config) { return false; }
  static constexpr bool constify  = false;
  static constexpr bool assumable = true;
  violation_response operator()(const char*, std::source_location, evaluation_config) const {
    return violation_response::terminate;
  }
};
```

`review` is the log-and-continue behavior Bloomberg deploys as `bsls_review`<sup>[21]</sup>, expressed without a language feature. `mandatory` is the always-checked, optimizable strand P4005R0<sup>[2]</sup> proposes, where the assertion's presence is ODR-affecting so an optimizer may rely on the predicate.

### 3.8 Zero-cost ignore and diagnostic quality

Because `is_ignored` is a compile-time query on the type, an ignored assertion produces no code and the predicate is never evaluated. A design that instead wraps the predicate in a library call, as in the plain-function form of P4009R0<sup>[3]</sup>, evaluates the predicate to make the call, even when the library ignores the result. The type-based form gives up nothing at the ignore configuration.

The call operator receives the predicate text as `comment` and a `std::source_location`. The compiler holds the predicate text already, so a violation message can name the expression without the library reconstructing it from a bare `bool`.

### 3.9 Scope

The design as specified here covers `pre`, `post`, and `contract_assert` on non-virtual functions and on statements. It does not cover old-value capture in postconditions or contracts on virtual functions. Both are the subjects of separate, already-prototyped extension papers: virtual functions in P3097R3<sup>[17]</sup> and postcondition captures in P3098R2<sup>[18]</sup>, with the caller-facing and callee-facing distinction the virtual-function work rests on developed in P3097R3<sup>[17]</sup>. Contracts on function pointers are not a special exclusion. Contracts attach to a declaration rather than to the function type, so an indirect call carries no declaration contract. The guaranteed-enforced strand of Section 3.7 is the mechanism when a check must run regardless of how the function is reached.

---

## 4. Relationship to the C++26 working draft

Two differences are substantive. First, constification is a property of the control object rather than a language rule: `default_control` does not constify, so a predicate means what the same expression means in the function body, and a control object enables it by setting `constify` to true. Second, an exception that escapes a predicate propagates as an ordinary exception, stopped at a `noexcept` boundary as any exception is, rather than becoming a contract violation. Both behaviors remain available as control-object properties; they stop being imposed by the language.

Both behaviors are separable, and the separation has implementation experience. P4005R0<sup>[2]</sup> reports "implementation experience with a mix of contract assertions with 'constification' and without" and, likewise, with and without converting an escaping predicate exception into a violation. Berne's P3261R0<sup>[7]</sup> enumerates removing constification as "Proposal 1: No const-ification," and P3592R0<sup>[8]</sup> proposes a `mutable` specifier "to simply turn off const-ification entirely within a contract assertion predicate." The design here makes the off state the default and the on state a control-object property.

The committee has polled removal of these elements. At Hagenberg in February 2025, EWG polled "Remove constification" at SF:9 F:7 N:6 A:37 SA:14, consensus against, and "Remove P2900 from C++26" at SF:9 F:8 N:3 A:19 SA:41, consensus against (EWG poll record, Hagenberg, February 2025). Both polls asked about removal without a replacement in hand.

The cost of these behaviors has been recorded since those polls. P4308R0<sup>[11]</sup> identifies a trilemma in which the `noexcept` operator's value, stack unwinding, and the meaning of `noexcept` cannot all be preserved at once when a predicate exception can escape a core-language check. P4306R0<sup>[10]</sup> records that converting predicate exceptions into violations requires the compiler to "generate the correct instructions for exception handling around every contract assertion," with an implementer noting the resulting backtrace "is useless." P4318R0<sup>[13]</sup> accounts for the standardization cost of a continuing semantic on the undefined class and finds it "returns less than it costs." A design in which an escaping exception propagates by default does not incur the trilemma or the exception-handling scaffolding, because there is no exception to convert.

The stage of the cycle bears on the cost of change. As of this writing the C++26 Draft International Standard has not been balloted. Amending one of these behaviors in undeployed draft text changes text that no implementation ships as a default; amending it after the standard ships changes a deployed default. That technical cost is lower before the standard is published, and it falls on text that the removability evidence above shows is separable. Against it stands a process cost: revisiting an element the committee has twice declined to remove is itself work, borne whenever a settled question is reopened. Both are facts of the current stage, recorded without a recommendation.

---

## 5. Contracts and profiles as independent peers

The question here is one of ownership: which feature owns the response when a checked operation fails.

A contract-violation handler that is invoked for core-language undefined behavior owns the response to that behavior. P4297R0<sup>[14]</sup> puts the ownership relation plainly: "Whoever owns the handler owns the response." If the response to a core-language operation, a signed-overflow check or a null-dereference check, is routed through the program-wide contract-violation handler, then the handler, and whoever configures it, determines what happens on that failure. The alternative is for the feature that guards the operation to own its guarantee.

This design scopes the contracts core to explicit, author-written assertions. `pre`, `post`, and `contract_assert` bind predicates the author wrote; they do not guard core-language operations the author did not annotate. Core-language undefined behavior is guarded by a profile instead. P4317R0<sup>[12]</sup> specifies `std::core_ub`, a profile that "owns its guarantee, its enumeration, and its response, and it leaves the meaning of `noexcept` untouched," where checking is a quality-of-implementation matter and a violated check ends the program rather than proceeding into undefined behavior. The 77 runtime-checkable cases split into 62 that terminate and 15 that have a defined replacement<sup>[12]</sup>.

The two facilities are therefore peers, not layers. The contracts core owns author-written assertions and their library-carried response. The profile owns core-language undefined behavior and its response. Neither is defined in terms of the other, and the contracts core has no claim on core-language undefined behavior, so it cannot become the substrate through which that behavior is configured.

This placement matches shipping practice. The log-and-continue behavior Bloomberg deploys, `bsls_review`<sup>[21]</sup>, operates at the library level where the state after a failed check is defined; on the harder cases Bloomberg's `bsls_assert`<sup>[21]</sup> terminates. The defined and undefined cases are handled by different tools, which is the same division this architecture draws between the contracts core and the `std::core_ub` profile.

---

## 6. Implementation

A reference implementation is in progress; this section states the plan and the components it builds on, and a later revision will report results.

The compiler part is a fork. The three-step algorithm (Section 3.2) is a small change to an existing P2900 compiler: the compiler already parses `pre`, `post`, and `contract_assert` and selects a semantic per build, so the change is to read the control-object's compile-time members and to call the control object rather than run a built-in semantic. The experimental GCC and Clang contracts branches on Compiler Explorer, which already implement the `noexcept-observe` and `noexcept-enforce` semantics<sup>[11]</sup>, and the GCC 16.1 experimental contracts implementation are the starting points.

The library part is a header. The `assertion_control` concept, `default_control`, the branch table, and the worked control objects of Section 3.7 are ordinary library code. The header also provides the plain-function surface of P4009R0<sup>[3]</sup> over the same machinery for the contexts that prefer it.

The peer profile is a second header. A `std::core_ub` profile modeled on P4317R0<sup>[12]</sup>, not routed through the contract-violation handler, demonstrates the architecture of Section 5 in code.

Each demonstration maps to a claim: an ignored assertion emits no code and no exception-handling scaffolding; a predicate exception propagates without conversion and `noexcept` keeps its meaning; a user-defined semantic is added with a library function and no compiler change; and no core-language undefined behavior is routed through the contract-violation handler. Old-value capture and contracts on virtual functions are out of scope for the reference implementation, consistent with Section 3.9; the P3097R3<sup>[17]</sup> and P3098R2<sup>[18]</sup> branches are the starting points if those are added later.

---

## 7. The default checked semantic

One design choice remains open: whether the default checked semantic, the one an unadorned `pre(cond)` gets, should be `enforce` or a non-throwing `enforce`. The choice interacts with the exception question. Because an escaping exception propagates by default, `enforce` already leaves `noexcept` with its usual meaning, which removes the reason a separate non-throwing semantic was introduced. A non-throwing default would nonetheless guarantee that a violation handler cannot itself throw across a `noexcept` boundary.

This design does not choose between them. A librarized design allows both to be prototyped as control objects, with the choice made from field experience. The two options carry different costs. `enforce` with a propagating exception preserves the ordinary exception behavior of the predicate and the usual meaning of `noexcept`, at the cost that a throwing handler under `enforce` interacts with `noexcept` as any throwing function does. A non-throwing default removes that interaction, at the cost of a second semantic to specify and a constraint on what a handler may do.

---

## 8. Objections

The two strongest objections to the design, each answered from evidence already presented.

### 8.1 "Contracts are a library problem, so no language change is warranted"

The objection. If Boost.Contract<sup>[19]</sup> already provides preconditions, postconditions, and customizable failure actions, and libc++ already ships hardening built as close to contracts as possible without the language feature<sup>[15]</sup>, and P2900 already places the handler and violation object in the library, then a library delivers contracts and the standard should not add language machinery for them.

The response, from the library author. The author of Boost.Contract wrote in 2016 that "language support for Contract Programming remains the ultimate solution even if Boost.Contract no longer uses crazy macros," because language support provides "a more concise syntax, compiler optimizations, and put the contracts with function declaration instead of definitions"<sup>[9]</sup>. Those three capabilities are what only the language can provide: a library predicate sits in the definition rather than the declaration; only the compiler can optimize on a guaranteed-enforced predicate, which is why P4005R0's assertions are ODR-affecting<sup>[2]</sup>; and a compiler that uses a contract for quality-of-implementation must see it in the source rather than behind a library call. The design keeps those three in the language, in its two syntactic jobs, and moves everything else to the library.

### 8.2 "One common syntax is the point, and only P2900 provides it"

The objection. Contracts are useful to static-analysis tools and documentation generators because everyone writes them the same way; a design that fragments the spelling, or that has no single short form, defeats that uniformity.

The response. The design keeps the single short form: `pre(cond)`, `post(r: cond)`, and `contract_assert(cond)` are the common spelling, and naming a control object is uncommon (Section 3.4). The uniformity the objection wants is a property of a facility that every toolchain implements. As of this writing, P2900 is implemented in experimental GCC and Clang branches but not across all major implementations, so portable code that must build on a toolchain without P2900 wraps its contracts in macros whose spelling varies by project, and the uniform, greppable form is not available in practice. A facility with a smaller language surface has a lower bar to implementation across toolchains, which is the condition under which a single greppable spelling actually exists.

---

## 9. Conclusion

A contracts facility can put two things in the language, binding a predicate to a declaration and naming the object that governs it, and place everything else in a library. The compiler reads a few compile-time properties off the control-object type and makes one call on a violation; constification, exception handling, the semantics, the handler, and the violation object are library code. The result keeps the capabilities only the language can provide, a declaration-attached syntax, optimization on a guaranteed-enforced predicate, and a contract the compiler can see, while making constification and exception propagation per-assertion properties that default off. A reference implementation on the existing compiler forks (Section 6) and field prototyping of the open default-semantic question (Section 7) are what build on this next.

---

## 10. Disclosure

The authors provide information and serve at the pleasure of the committee.

Vinnie Falco is affiliated with The C++ Alliance. Ville Voutilainen is the author of P4005R0<sup>[2]</sup> and P4009R0<sup>[3]</sup>.

This is an information paper.

The authors have written companion papers on C++26 contracts and profiles, including P4306R0<sup>[10]</sup>, P4308R0<sup>[11]</sup>, P4317R0<sup>[12]</sup>, P4318R0<sup>[13]</sup>, and P4297R0<sup>[14]</sup>. This paper offers a design alternative to P2900<sup>[1]</sup>, which the authors did not write, and the authors have a stated preference for the profiles-based architecture of Section 5.

One limitation is genuine and stated plainly: the reference implementation of Section 6 is not yet complete, so the evaluation in Appendix B rests on the design and on published prior work rather than on measured results from this design.

This paper is one of a series of C++ Alliance papers on contracts and profiles, listed above.

Methodology: the design synthesizes published papers and public documentation, and does not rely on private committee records. The paper was prepared with machine assistance.

This paper asks for nothing.

---

## Acknowledgments

The assertion object whose compiler-read members steer code generation is Bengt Gustafsson's design in P3968R0<sup>[4]</sup>. The assertion-control object composed from compile-time labels is Joshua Berne's design in P3400R4<sup>[5]</sup>. The design synthesizes their work with the guaranteed-enforced and librarized-semantics designs of P4005R0<sup>[2]</sup> and P4009R0<sup>[3]</sup>.

---

## References

[1] [P2900R14](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p2900r14.pdf) - "Contracts for C++" (Joshua Berne, Timur Doumler, Andrzej Krzemie&#324;ski, 2025).

[2] [P4005R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4005r0.html) - "A proposal for guaranteed-(quick-)enforced contracts" (Ville Voutilainen, 2026).

[3] [P4009R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4009r0.html) - "A proposal for solving all of the contracts concerns" (Ville Voutilainen, 2026).

[4] [P3968R0](https://wg21.link/p3968r0) - "A Framework For Contracts" (Bengt Gustafsson, 2026).

[5] [P3400R4](https://wg21.link/p3400r4) - "Controlling Contract-Assertion Properties" (Joshua Berne, 2026).

[6] [P4275R0](https://wg21.link/p4275r0) - "Assertion-Control Objects (P3400R4)" [EWG presentation] (Joshua Berne, 2026).

[7] [P3261R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p3261r0.pdf) - "Revisiting const-ification in Contract Assertions" (Joshua Berne, 2024).

[8] [P3592R0](https://wg21.link/p3592r0) - "Resolving Concerns with const-ification" (Joshua Berne, Timur Doumler, Lisa Lippincott, 2025).

[9] [Boost developers mailing list](https://listarchives.boost.org/Archives/boost/2016/06/230240.php) - "Re: [boost] [contract] Without the macros" (Lorenzo Caminiti, 2016-06-15).

[10] [P4306R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4306r0.pdf) - "Configuring Runtime Checking: Profiles and Implicit Contract Assertions" (Vinnie Falco, Ville Voutilainen, 2026).

[11] [P4308R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4308r0.pdf) - "Eight Responses to a Throwing Implicit Contract Assertion" (Vinnie Falco, Ville Voutilainen, 2026).

[12] [P4317R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4317r0.pdf) - "A Profile for Runtime-Checkable Core-Language Undefined Behavior: std::core_ub" (Vinnie Falco, 2026).

[13] [P4318R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4318r0.pdf) - "Transient Benefit, Perpetual Cost: Implicit Core-Language Assertions" (Vinnie Falco, 2026).

[14] [P4297R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4297r0.pdf) - "Severing P3100's Profiles Claim from Its Case-by-Case Review" (Vinnie Falco, Ville Voutilainen, 2026).

[15] [Practical Security in Production](https://queue.acm.org/detail.cfm?id=3773097) - "Practical Security in Production: Hardening the C++ Standard Library at Massive Scale" (Louis Dionne, Alex Rebert, Max Shavrick, Konstantin Varlamov, ACM Queue Vol. 23 Iss. 5, 2025).

[16] [P3294R2](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p3294r2.html) - "Code Injection with Token Sequences" (Andrei Alexandrescu, Barry Revzin, Daveed Vandevoorde, 2024).

[17] [P3097R3](https://wg21.link/p3097r3) - "Contracts for C++: Virtual functions" (Timur Doumler, Joshua Berne, Ga&#353;per A&#382;man, 2026).

[18] [P3098R2](https://wg21.link/p3098r2) - "Contracts for C++: Postcondition captures" (Timur Doumler, Ga&#353;per A&#382;man, Joshua Berne, 2026).

[19] [Boost.Contract](https://www.boost.org/doc/libs/release/libs/contract/doc/html/index.html) - Boost.Contract library documentation (Lorenzo Caminiti).

[20] [cppfront](https://hsutter.github.io/cppfront/cpp2/contracts/) - cppfront contracts and metafunctions documentation (Herb Sutter).

[21] [bsls_assert](https://bloomberg.github.io/bde-resources/doxygen/bde_api_prod/group__bsls__assert.html) and [bsls_review](https://bloomberg.github.io/bde-resources/doxygen/bde_api_prod/group__bsls__review.html) component documentation (Bloomberg BDE, retrieved 2026).

[22] B. Stroustrup, *The Design and Evolution of C++* (Addison-Wesley, 1994).

[23] [P3071R1](https://wg21.link/p3071r1) - "Protection against modifications in contracts" (Jens Maurer, 2024).

[24] [Value numbering](https://en.wikipedia.org/wiki/Value_numbering) - "Value numbering", Wikipedia (retrieved 2026).

[25] [P2834R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p2834r1.pdf) - "Semantic Stability Across Contract-Checking Build Modes" (Joshua Berne, John Lakos, 2023).

[26] [P3318R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p3318r0.html) - "Throwing violation handlers, from an application programming perspective" (Ville Voutilainen, 2024).

---

## Appendix A: Relationship to prior work

This design does not originate the idea of a small contracts core with a library-carried response. It unifies four published lines of work, and it uses a library surface that P2900 already specifies.

The assertion object with codegen-steering members is Gustafsson's. P3968R0<sup>[4]</sup> states the goal directly: "remove the dependency from the compiler to the standard library and delegate as much as possible of the functionality to library code for maximum flexibility," with the compiler reading boolean members named `constify`, `ignorable`, and `assumable` and calling an `operator()` on a violation. The `assertion_control` concept in Section 3.3 is the same shape with `is_ignored` in place of `ignorable`.

The compile-time control object is Berne's. P3400R4<sup>[5]</sup> introduces "assertion-control objects ... composed from constexpr labels, that let developers customize contract-assertion behavior directly in C++ source code," whose logic "must execute at compile time so that it can influence code generation." P3400R4 is authored by a co-author of P2900 and P3100, and P4275R0<sup>[6]</sup> presents it to EWG with an explicit division into a core-language part with "very little library dependency" and library utilities that "most can be written by users themselves if needed." The design here adopts that division.

The librarized semantics and the guaranteed-enforced strand are Voutilainen's. P4009R0<sup>[3]</sup> expresses the semantics as library functions, so that "adding new semantics and new functionality is simple and straightforward," and P4005R0<sup>[2]</sup> supplies a guaranteed-enforced form whose assertions carry "no 'constification' or other additional transform" and are ODR-affecting; the mangling "makes it safe to enable compiler optimizations based on knowledge of the results of guaranteed-enforced assertions." Section 3.7's `mandatory` control is that strand.

The response is already a library in P2900. P2900R14<sup>[1]</sup> specifies the handler as "a function named `::handle_contract_violation`," provides `invoke_default_contract_violation_handler`, and defines the `contract_violation` object with properties "accessed by const, non-virtual member functions." The design moves the remaining piece, the branch that selects among semantics, into the same library.

Library implementations of contract behavior exist and are shipping. Boost.Contract<sup>[19]</sup> supports "customizable actions on assertion failure (e.g., terminate the program or throw exceptions)" entirely in a library, and its author wrote in 2016 that "language support for Contract Programming remains the ultimate solution" because it provides "a more concise syntax, compiler optimizations, and put the contracts with function declaration instead of definitions"<sup>[9]</sup>. cppfront<sup>[20]</sup> ships contract groups with customizable violation handlers as library objects. P3294R2<sup>[16]</sup> describes the modern replacement for the preprocessor macros such libraries have relied on, where "a macro is a function that takes a token sequence and returns a token sequence."

---

## Appendix B: Evaluation against the design principles

This appendix compares the minimal core and P2900 against the principles in Stroustrup's *The Design and Evolution of C++*<sup>[22]</sup>. The principles are Stroustrup's; the comparison covers the principles on which the two designs differ. On the others, including expressing contracts in the language rather than in macros and driving the feature from real problems, the two designs are alike, because both keep the `pre`, `post`, and `contract_assert` syntax and address the same use cases. The P2900 column is left for the reader to fill in. Each row states the factual difference; the reader may judge whether a difference constitutes satisfaction of the principle or not.

Table 1. Design-principle comparison: the principles on which P2900 and the minimal core differ.

| Principle (D&E<sup>[22]</sup>) | P2900 | Minimal core | Difference |
|---|---|---|---|
| Zero-overhead: what is not used is not paid for | | Yes | P2900's default conversion of an escaping exception into a violation makes the compiler generate exception-handling code around every contract assertion<sup>[10]</sup>; the minimal core lets an escaping exception propagate by default, and an ignored assertion emits nothing (Section 3.8). |
| Prefer compile-time to run-time resolution | | Yes | P2900 applies constification, and converts an escaping exception into a violation, at run time; the minimal core resolves ignoredness, constification, and assumability as compile-time queries on the control-object type (Section 3.2). |
| Do not force one style | | Yes | P2900 fixes the semantics and these behaviors in the language; the minimal core lets a platform or user add a semantic library-side (Section 3.5), so a log-and-continue style is expressible without committee action. |
| General mechanisms over special-purpose features | | Yes | P2900 builds the semantics into the compiler; the minimal core selects a semantic by naming a different object, the most familiar extension mechanism in the language (Section 3.5). |
| Verifiable by local inspection | | Yes | Under constification a predicate can bind a different overload than the same text in the body, so "overload resolution might quietly invoke different functions in the two contexts"<sup>[8]</sup>; the minimal core evaluates the predicate under the usual rules by default<sup>[2]</sup>. Build-time semantic selection remains in both, so the difference is the meaning of the expression, not the selected semantic. |
| Integrate with the language, not a sub-language | | Yes | Constification is a new evaluation rule for a predicate; the minimal core reuses ordinary expression evaluation and an ordinary function call for the response. |

The scoring for the minimal core is the scoring of its default control object. A control object that sets `constify` to true reintroduces the local-inspection cost for the assertions that name it, and one that converts an escaping exception into a violation reintroduces the exception-handling scaffolding, so those rows would read for such an object as they do for P2900. The difference the table records is that P2900 applies the two behaviors unconditionally, while here they are opt-in per assertion and off by default.

The P2900 column is blank because the answer depends on which cost the reader weighs higher. Constification protects against accidental mutation in predicates, a problem P3071R1<sup>[23]</sup> addresses. Whether that protection outweighs the local-inspection cost is a technical judgment. Converting an escaping exception into a violation preserves a single response path through the violation handler. Whether that uniformity outweighs the zero-overhead cost is the same kind of question. P2900's authors made deliberate trade-offs, and different weights on the same evidence yield different answers. Both are sound engineering. Build-time semantic selection, the property P2900's authors treat as central, is preserved in both designs and is not at issue in any row. The table records the differences. The reader scores them.

The principle of providing a better alternative before removing an older facility<sup>[22]</sup> is addressed by Section 4: the alternative is specified here, so any change to the draft is a replacement rather than a removal.

---

## Appendix C: The assertion-control interface

The interface and the three control objects of Section 3, assembled as a single listing.

```cpp
namespace std::contracts {

enum class evaluation_config : unsigned {
  ignore = 0, observe = 1, enforce = 2, quick_enforce = 3,
  // [4 .. 0xFFFF] reserved to the standard; [0x1'0000 ..] reserved to vendors and users
};

enum class violation_response { proceed, terminate };

template <class T>
concept assertion_control =
  std::is_empty_v<T> &&
  requires (T c, const char* comment, std::source_location loc, evaluation_config cfg) {
    { T::is_ignored(cfg)      } -> std::same_as<bool>;
    { T::constify             } -> std::convertible_to<bool>;
    { T::assumable            } -> std::convertible_to<bool>;
    { c(comment, loc, cfg)    } -> std::same_as<violation_response>;
  };

struct default_control {
  static constexpr bool is_ignored(evaluation_config cfg) { return cfg == evaluation_config::ignore; }
  static constexpr bool constify  = false;
  static constexpr bool assumable = false;
  violation_response operator()(const char* comment, std::source_location loc,
                                evaluation_config cfg) const;
};
inline constexpr default_control default_v{};

struct review {     // log-and-continue at the library level, always checked
  static constexpr bool is_ignored(evaluation_config) { return false; }
  static constexpr bool constify  = true;
  static constexpr bool assumable = false;
  violation_response operator()(const char*, std::source_location, evaluation_config) const;
};

struct mandatory {  // guaranteed-enforced, optimizable
  static constexpr bool is_ignored(evaluation_config) { return false; }
  static constexpr bool constify  = false;
  static constexpr bool assumable = true;
  violation_response operator()(const char*, std::source_location, evaluation_config) const;
};

}
```
