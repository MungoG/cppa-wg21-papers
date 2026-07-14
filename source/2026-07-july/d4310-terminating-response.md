---
title: "Hasta la Vista, Undefined Behavior: Why Implicit Contract Violations Should Terminate"
document: D4310R0
date: 2026-07-13
intent: info
audience: EWG, SG21
reply-to:
  - "Vinnie Falco <vinnie.falco@gmail.com>"
---

## Abstract

Everything that ships terminates or traps on a detected core-language violation; no hardened implementation makes log-and-continue its default.

C++26 Contracts define four evaluation semantics, and P3100R6<sup>[1]</sup> extends them to the core language, so that a detected instance of runtime-checkable undefined behaviour invokes the contract-violation handler. This paper examines one question that extension raises: after the handler runs, should the program continue past the violation, or terminate? It separates the handler (a telemetry hook that logs the violation) from the continuation (executing past the detected violation), because only the second is contested. It then places the deployment record, the cost, the security literature, and the committee's own adopted decisions beside the two responses. No implementation yet provides implicit assertions with any semantic, so the comparison reasons from deployed analogues. Across the hardened implementations surveyed, the terminating response is the production default and continuing appears only as an opt-in or an adoption aid; the continuation carries an exception-handling cost the reference implementers decline to incur; and a C++26 decision already forbids the continuation for standard-library hardening. The paper's finding is that a terminating response, invoke the handler then terminate, is the response the evidence supports as the default. If a continuing response is to exist, the deployed precedents point to one shape, an opt-in and non-portable facility bounded to an adoption period, rather than a default. The paper proposes no wording and requests no poll.

---

## Revision History

### R0: July 2026

- Initial version.

---

## 1. Disclosure

The author provides information and serves at the pleasure of the committee.

The author is the founder of the C++ Alliance, which maintains a Clang fork for Profiles work, and prefers the family of responses in which no exception escapes an implicit contract assertion. That is a stake in the outcome, and the reader should weigh what follows accordingly.

The intent of this paper is `info`. It argues a position, that a terminating response is the one the evidence supports for implicit core-language assertions, but it proposes no wording and requests no poll.

This paper changes nothing in ratified C++26. C++26 Contracts (P2900R14<sup>[2]</sup>) are treated as fixed: the four evaluation semantics, the single violation handler, and the deliberate allowance that a handler may throw from an explicit contract assertion all stand unchanged. The question here belongs to the open C++29 work on implicit assertions (P3100R6<sup>[1]</sup>), and the paper addresses only that.

One limitation is disclosed up front. The comparison rests on a deployment survey and on the class of core-language checks whose continuation is undefined; it concedes in Section 5 the narrower class where continuation is defined. And no compiler yet implements implicit contract assertions with any semantic, so the paper reasons from deployed analogues, not from a conforming implementation of the feature itself.

This paper is one of a set in the July 2026 mailing on runtime-checking configuration. P4306R0<sup>[3]</sup> compares the configuration-ownership models on the public record, and P4297R0<sup>[4]</sup> asks EWG to decide the ownership relationship by an explicit poll. This paper is scoped to the response question and cross-references those rather than repeating them.

This paper was prepared with the assistance of generative tools; the author is responsible for its content.

This paper places the record for the committee's use and makes no request.

---

## 2. Two questions inside the one word 'observe'

This section separates the parts of the `observe` semantic, because the stated semantic treats as one decision what is really two. A reader who knows the C++26 Contracts model can skip to the last paragraph.

C++26 Contracts (P2900R14<sup>[2]</sup>, `[basic.contract.eval]`) define four evaluation semantics for a contract assertion. Under `ignore`, the assertion has no effect. Under `observe`, the contract-violation handler is invoked and, if it returns normally, control continues past the point of evaluation. Under `enforce`, the handler is invoked and the program is then contract-terminated. Under `quick-enforce`, the program is contract-terminated without invoking the handler. The handler is a single, program-wide function, `::handle_contract_violation`, and P3100R6<sup>[1]</sup> Section 5.6 keeps it single for implicit assertions rather than introducing a second one.

P3100R6<sup>[1]</sup> respecifies the runtime-checkable cases of core-language undefined behaviour as implicit contract assertions evaluated with these same four semantics. That extension raises a question: on a detected core-language violation, such as a signed-integer overflow or an out-of-bounds access, what happens after the check fails?

The word `observe` bundles two things that can be separated. The first is the handler invocation (termed "hook" henceforth): the handler is invoked, and it logs the violation, giving a deployment one place to record and report it. The second is the continue-past-violation response (termed "continuation" henceforth): after the handler returns, execution proceeds past the violation. The hook is uncontested and is preserved by every response this paper discusses, `enforce` included. The continuation is the contested part. The single difference between `enforce` and `observe` is what happens on a normal return from the handler: `enforce` terminates, `observe` continues. The sections that follow credit the hook and examine only the continuation.

Scope: this paper concerns implicit assertions on core-language undefined behaviour, where a detected violation means the program has already entered a state the language does not define. Explicit, author-written contract assertions, where a precondition can encode a recoverable condition and continuation can be meaningful, are outside this scope.

The terminating response preserves the hook. Under `enforce`, the handler is invoked and logs the violation before the program terminates. Every telemetry need the handler serves - recording the violation site, its kind, the predicate text - survives the terminating response unchanged. What `enforce` removes is the continuation alone: execution past a state the language does not define. The deployer's ability to observe a violation is not at stake; only the program's ability to keep running after one is.

---

## 3. What ships, terminates

This section reports what deployed hardening does on a detected core-language violation. The pattern is uniform.

Table 1. Response to a detected violation in deployed hardened implementations. Every entry terminates or traps; none continues by default.

| Implementation | Response on violation | Source |
|---|---|---|
| libc++ `fast` / `extensive` | trap (`quick-enforce`) | <sup>[5]</sup> |
| libc++ `debug` | log + terminate (`enforce`) | <sup>[5]</sup> |
| libstdc++ `_GLIBCXX_ASSERTIONS` | `abort` | <sup>[6]</sup> |
| MSVC STL `_MSVC_STL_HARDENING` | `__fastfail` | <sup>[7]</sup> |
| glibc `_FORTIFY_SOURCE` 1/2/3 | `SIGABRT` | <sup>[8]</sup> |
| Google server fleet (libc++) | trap (`quick-enforce`); ~0.3% avg overhead | <sup>[9]</sup> |
| Android IntSan | `abort` (log mode is testing only) | <sup>[10]</sup> |
| UBSan (production guidance) | trap (`-fsanitize-trap`); recover is "meant for testing purposes" | <sup>[11]</sup> |
| Abseil `CHECK`, Folly `XCHECK`, WebKit `RELEASE_ASSERT` | terminate | <sup>[12]</sup> |

The survey covers standard-library hardening modes, production sanitizer configurations, and critical-assertion facilities - the implementations that check for core-language violations in production. It does not cover availability-first domains that might rationally choose a different response.

Every surveyed implementation that detected a core-language violation and chose a response chose termination. Zero chose continuation as a production default. The absence of a conforming implementation of implicit contract assertions is symmetrical - no compiler has one with any semantic. The deployment analogues are not symmetrical: the terminating response has nine deployed analogues that selected it; the continuing response has none. The argument is asymmetrical precedent.

The implementations that log before terminating - libc++ debug mode, glibc `_FORTIFY_SOURCE`, Android IntSan in its production configuration - perform exactly the operation `enforce` specifies: invoke a reporting facility, then terminate. The handler is not absent from the survey; it is present in every entry that reports before it stops.

Bloomberg's own contract-checking machinery records the same conclusion in its commit history. The public BDE repository's `bsls_assert` facility terminates by default, and one commit message states the reasoning directly: a build "with a continuing handler, could cause execution to continue past that point," that "execution path is library undefined behavior and the program would be out of contract anyway," and the change makes the program "always terminate."<sup>[13]</sup> This is a deployer of a log-and-continue facility describing continuation past a detected violation as the defect and termination as the fix.

The deployment record therefore establishes one fact: on a detected core-language violation, the terminating response is the one in production use, and the continuing response is not.

---

## 4. The cost, and the rule it already breaks

This section adds two findings to the deployment record: the continuation carries a cost the reference implementers decline to pay, and it runs against a decision the committee has already adopted for C++26.

The cost of continuing is not a matter of a branch. Turning a core-language operation into a checked operation that can invoke a handler and continue requires exception-handling machinery around the check: because whether the program's handler is `noexcept` is a link-time decision, P2900R14<sup>[2]</sup> Section 3.6.6 notes that "the compiler ... has to generate the correct instructions for exception handling around every contract assertion." The implementers who ship hardening declined exactly this. P3191R0<sup>[14]</sup>, from the libc++ team, sets the production requirement that a contract violation "should generate no code at all beyond the equivalent of a branch and a `__builtin_trap()`," with "no exception-handling code being generated around contract predicates," and describes the handler-and-object path as "a lot of code and data being generated for a single assertion." The committee's own response to this cost was to add the `quick-enforce` semantic, which skips the handler entirely, recorded in P3198R0<sup>[15]</sup>. The isolated cost of the continuation over a trap has not been published as a measured figure; what the record shows is that the reference implementers decline the continuation path in production and the committee added a semantic to avoid its overhead.

The committee has also already decided this question for the adjacent case. P3878R1<sup>[16]</sup>, adopted into C++26, established that a standard-library hardened precondition may not be evaluated with a non-terminating semantic, on the reasoning that continuing past such a check "can result in violations of hardened preconditions being undefined behaviour, rather than guaranteed to be diagnosed, which defeats the purpose of using a hardened implementation." For the core-language checks whose continuation is likewise undefined, a detected null dereference or out-of-bounds access, the same reasoning applies one level down; it does not reach the class in Section 5, where continuation is into a defined result. One disclosure belongs here: the lead author of P3878R1 is a co-author of this paper's companions, though the decision was the whole committee's, and the argument stands on the deployment record and the security literature that follow even if the precedent is set aside.

On this premise, the present analysis and the Contracts proposal agree, though they reach a different conclusion about the default. Doumler and Berne write in P3097R2<sup>[17]</sup> that once a program

> is found to be in a possibly corrupted state, executing any user-defined code could result in a vulnerability.

They keep the `observe` semantic available nonetheless; the same hazard is the reason a corrupted-state continuation should not be the default. The agreement is on the danger; the resolution is where the two diverge.

The security literature is consistent with it. Microsoft's fail-fast documentation states that on a detected corruption "no exception handlers are invoked because the program is expected to be in a corrupted state."<sup>[18]</sup> The glibc maintainers removed even the backtrace from the heap-corruption path, on the reasoning that "doing more work at this point risks ... enabling code execution exploits."<sup>[19]</sup> The CERT C++ secure-coding rule ERR56-CPP states that "a violated invariant leaves the program in a state where graceful continued execution is likely to introduce security vulnerabilities."<sup>[20]</sup> Work on exception unwinding as an exploit surface (CHOP, NDSS 2023<sup>[21]</sup>) shows that running the unwinder over corrupted state can defeat shadow stacks. No published incident names a C++ contract-violation handler, because C++26 Contracts are not yet deployed; the evidence here is transfer from mechanisms that faced the identical choice and chose to terminate. This security argument is strongest where continuation runs on corrupted or undefined state; for the defined-replacement class of Section 5, it does not apply. For the checks whose continuation is undefined but whose violation is not memory corruption, the transfer is from the principle rather than from the mechanism: the state is undefined, and executing further on undefined state is what the fail-stop doctrine treats as the hazard.

---

## 5. Where continuing is defined

One class of core-language checks continues into defined behaviour, and the objection in Section 4 does not reach it.

Not every core-language check continues into undefined behaviour. P3100R6<sup>[1]</sup> gives some cases a defined replacement: a signed-integer overflow, for instance, can be specified to produce a wrapped result, so that continuing after the check yields a defined value rather than undefined behaviour. For that class, the objection in Section 4 does not apply, because continuation is into defined behaviour, and `observe` there is coherent.

What the concession does not supply is a deployed user. The field's response to signed overflow is either `-fwrapv`, which defines wraparound with no handler and no log, or `-ftrapv`, which terminates; neither is the log-and-continue that `observe` would add. So even in the class where continuation is defined, no surveyed deployment logs and continues by default. The question the evidence leaves standing is who requires that response; the record does not answer it. Where a deployer selects a continuing semantic for this defined class, that is a defensible choice, and the terminating finding of this paper does not reach it.

A second constraint applies to the defined-replacement class regardless of the safety objection: the cost. The exception-handling machinery the reference implementers decline in Section 4 must be generated around every checked operation if any semantic can continue, because whether the deployer selects `observe` or `enforce` is a build-time or link-time decision unknown to the compiler at code-generation time. This cost is class-agnostic. It applies to a signed-overflow check that continues into a defined wrapped result exactly as it applies to an out-of-bounds check that continues into undefined behaviour. The implementers' objection in P3191R0<sup>[14]</sup> - "no exception-handling code being generated around contract predicates" - does not distinguish between the two classes, because the generated code cannot.

---

## 6. A terminating response

This section states the response the evidence supports and shows it in code. It reuses the C++26 semantics without adding to them.

The response is to invoke the handler for its telemetry and then terminate, and to prevent a handler throw from escaping the checked expression. Both halves already exist in C++26. The first is the `enforce` semantic: the handler is invoked and, on a normal return, the program is contract-terminated (P2900R14<sup>[2]</sup>, `[basic.contract.eval]`). The second is the existing rule that an escaping exception at a non-throwing boundary results in termination; applied to an implicit assertion, a handler throw contract-terminates rather than propagating. Expressed as a restriction on the C++29 feature, implicit core-language assertions are limited to the terminating semantics, and a handler throw from one contract-terminates. This adds no fifth semantic; it reuses `enforce` and the existing termination rule. This is the response the C assert integration already takes: P3290R4<sup>[22]</sup> Section 2.2 specifies that `assert` invokes the handler nonthrowing and then terminates, and SG22 (C/C++ Liaison) reached consensus in July 2026 against letting handler exceptions propagate from `assert`.<sup>[23]</sup>

A terminating default is not a new idea. The co-authors of P3100R6<sup>[1]</sup> and P2900R14<sup>[2]</sup> recommend the same default. Berne and Lakos, in P3558R1<sup>[24]</sup>, recommend

> a default evaluation semantic, when nothing else is specified, of `enforce` for all core-language preconditions.

The finding does not depart from that recommendation; it reinforces it with the deployment record, the implementer evidence, and the committee's own adopted decision for the adjacent case. The former convener of WG21 records the same recommended practice. For an enforced safety check, P3081R2<sup>[25]</sup> advises implementations to

> use the evaluation semantic `quick_enforce` or `enforce`, and emit a diagnostic if the evaluation semantic used in execution may be `observe` or `ignore`.

Both recommendations state the terminating response as the default.

Consider the canonical example from related discussions, a detected overflow inside a function marked non-throwing:

```cpp
int f(int x) noexcept { return x + 1; }
```

Under a throwing response, a detected overflow in `x + 1` invokes a handler that may throw, and the exception attempts to leave a function the `noexcept` operator reports as non-throwing. The exception cannot both leave `f` and honor `noexcept(f)` reporting `true`. Doumler and Berne identify this trade-off in P3100R6<sup>[1]</sup> Section 5.5: the only way to keep this response without a source-breaking change is to "redefine the meaning of the `noexcept` operator to be 'can never throw an exception unless there is a contract violation'." The `noexcept` operator has been part of the function type since P0012R1<sup>[26]</sup> in C++17; the throwing response keeps its value at `true` while changing what that `true` guarantees. The terminating response leaves the operator alone. Under it, the handler for a violation in `x + 1` logs and the program terminates; `noexcept(x + 1)` remains `true` and remains honest, because nothing escapes. The security concern this raises is not hypothetical: N3103<sup>[27]</sup>, from 2010, records that a `noexcept` violation allowed to continue "can be exploited by a malicious user to bypass security restrictions," and recommends immediate termination.

There is a second reason to contain the throw, visible without leaving the standard library. Turning a detected violation into a thrown exception unwinds the stack through code that did not anticipate a throw at that point, running destructors on objects whose invariants are momentarily broken, so a frequently benign overflow becomes a double-free or a half-destroyed object. The standard library already refuses to throw at exactly these moments: `std::vector` reallocation uses `move_if_noexcept` so that a throwing move cannot corrupt the container mid-operation. Bloomberg's test infrastructure records the same collision from the other side: a commit notes that when a destructor "becomes implicitly `noexcept`, so throwing the test exception type out of the assert handler triggers a call to `terminate`," and the workaround is a non-throwing, terminating handler.<sup>[28]</sup> The terminating response is the one that does not manufacture this defect.

The terminating response leaves build-time configurability intact. A deployer still selects among `enforce` (invoke the handler, then terminate), `quick-enforce` (trap without the handler), and `ignore` (no check) for an implicit core-language assertion, and keeps full control of the handler body. What the finding narrows is the menu for one class of assertion, not the principle that the evaluation semantic is a build-time choice.

The bifurcation this draws, implicit core-language assertions terminate while explicit contracts may still throw, is deliberate and defensible. It is the same line P3878R1<sup>[16]</sup> drew for hardening, and it rests on the difference the scope in Section 2 named: an author-written precondition can encode a recoverable condition, whereas a detected core-language violation means the program is already in a state the language does not define. This is also the answer to the objection that a bug is a bug and the semantics should be uniform: the committee has already treated the two differently, one level up.

---

## 7. If continuing must be possible

This section addresses the case in which the committee concludes a continuing response must be available to someone. Where it is available, the deployed precedents share one shape: an opt-in, non-portable facility, bounded to an adoption period rather than a default or a portable guarantee.

The case for a continuing response deserves its strongest statement, and two forms of it are real. The first is availability: a long-running service or a fault-tolerant embedded system may prefer bounded degradation to a hard stop, so that for such a system a violation that terminates is itself the failure. The second is migration: a large codebase that adds a new check needs a way to find the violations it surfaces before it enforces them, so that turning the check on does not stop a program that works today. Both are legitimate.

The answer distinguishes them. The migration need is met by the hook without the continuation: the terminating response already invokes the handler and logs, so a team sees every violation before it enforces and moves from find to fix to enforce without ever running past a live violation. The availability need is real where continuation is into a defined result, the class of Section 5, and there the objection does not apply. Where continuation is into undefined behaviour, keeping the program running is not availability but execution on a corrupted state, which the security record in Section 4 treats as the more dangerous failure. What remains, a team that has weighed this and still chooses to continue past an undefined-state violation, is served by the facility below, not by a default.

A continuing response has a defensible shape, and the field already uses it. It is the shape of an explicit, non-default opt-in that its own documentation labels as undefined behaviour and disclaims. The libc++ `observe` semantic is documented in exactly these terms: "Continuing execution after a hardening check fails results in undefined behavior; the `observe` semantic is meant to make adopting hardening easier but should not be used outside of the adoption period."<sup>[5]</sup> Bloomberg's `bsls_review` is the same idea deployed: an explicit, temporary downgrade from `assert` to log-and-continue while a newly tightened check is rolled out, on library-level checks rather than core-language undefined behaviour.<sup>[29]</sup> The .NET platform offers the closest full-lifecycle precedent: it once let managed code opt into catching corrupted-state exceptions, then discouraged it with an analyzer whose guidance is "the safest option is to allow the process to crash," and ultimately made the opt-in inert.<sup>[30]</sup>

These precedents share three properties, and a continuing facility for core-language checks would need all three. It is opt-in, so the default remains the terminating response. It is non-portable and implementation-defined, so it does not become a guarantee every implementation must carry, which is what would re-import the cost in Section 4. And it is marked and disclaimed at the point of use, so that continuing past a detected violation is an affirmative choice recorded in the source, owned by whoever makes it, rather than a behaviour hidden in a default. A facility with those three properties answers the migration case that motivates `observe`, because the transitional need, to see violations before enforcing them, is met by the hook, which the terminating response already provides, and continuing during a rollout is a per-project, non-portable concern that a build already expresses today.

Whether such a facility should be built is a separate question; what matters is the shape: not the default, and not a portable semantic every implementation must support.

---

## 8. Problems with this analysis

This analysis has problems, and they are worth stating. It reasons from analogy rather than from a conforming implementation. It draws on security literature for a feature that is not a security feature. It proposes to restrict a semantic whose availability matters to teams with large legacy codebases. And the strongest deployed counter-example to its survey comes from the same institution whose co-authors recommend the default this analysis finds the evidence supports.

### The first problem: the security literature does not apply to a correctness tool

Contracts are a correctness tool for finding bugs, not a security mechanism. Citing fail-fast policies, CERT rules, and exploit research imports a category that does not belong. The category distinction is correct: contracts are not a security feature. The literature is cited for its subject, not its category. Doumler and Berne write, in P3097R2<sup>[17]</sup>, that once a program "is found to be in a possibly corrupted state, executing any user-defined code could result in a vulnerability." The hazard of continuing past a detected violation is documented by the authors of the feature this analysis examines.

### The second problem: restricting the semantic limits deployer choice

The standard should offer all four semantics and let each deployment decide. Restricting the menu takes a legitimate choice away from teams that need it. Deployer autonomy matters, and the terminating response preserves three of four semantics and the full customizability of the handler body. What it removes is execution past a state the language does not define. P3878R1<sup>[16]</sup>, adopted into C++26, made the same restriction for standard-library hardened preconditions. The committee adopted it.

### The third problem: the migration story requires seeing all violations before enforcing

A large codebase that turns on a new check needs to find every violation it surfaces before it enforces. If the first violation terminates the program, the team sees one signal per deployment. Under the terminating response, the handler fires, logs the violation, and the program terminates. The next deployment, with the first violation fixed, reveals the second. The hardened implementations in Table 1 rolled out checks across codebases measured in hundreds of millions of lines - Google's production fleet, libc++ across Apple's platforms, glibc across every Linux distribution - with the terminating response and without a log-and-continue semantic.

### The fourth problem: Bloomberg's `bsls_review` is the deployed counter-example

Bloomberg maintains `bsls_review`, a companion to `bsls_assert` that logs and continues while a newly tightened check is rolled out. Bloomberg confines log-and-continue to the library level in its own deployment - preconditions whose violation leaves the program in a state the library does not define but the language still does.<sup>[31]</sup> It does not operate on core-language undefined behaviour. Libc++ documents its `observe` semantic in the same terms: "should not be used outside of the adoption period."<sup>[5]</sup> And the co-authors of Bloomberg's contracts framework recommend, in P3558R1<sup>[24]</sup>, "a default evaluation semantic, when nothing else is specified, of `enforce` for all core-language preconditions."

---

## 9. The configuration question is separate

The response question and the configuration question are independent, and a boundary between them is worth stating.

The first is what the response to a detected core-language violation should be. The second is who configures that response: whether the selection of a semantic for a core-language check is expressed through the Contracts configuration facility (P3400R3<sup>[32]</sup>, which specifies runtime selection of the evaluation semantic, including for implicit assertions) or through the Profiles framework (P3589R2<sup>[33]</sup>, which today specifies structural request and suppression, with any runtime effect left implementation-defined). Routing core-language checks through the contract-violation handler is the layering that P4297R0<sup>[4]</sup> examines and would put to an explicit poll.

The layering is stated here as a fact and not resolved. The two questions can be answered independently, and the finding that the response should terminate holds regardless of which facility owns the selection. The comparison of the ownership models is in P4306R0<sup>[3]</sup>.

---

## 10. Conclusion

Of the responses to a detected core-language contract violation, one is in production use across every hardened implementation surveyed here, and it is the terminating response: invoke the handler, log, and terminate. The continuing response is not in production use for core-language undefined behaviour, carries a cost the reference implementers decline to pay, and runs against P3878R1<sup>[16]</sup>, the decision C++26 already adopted for the adjacent case. Where continuation is defined, the record still shows no deployment that logs and continues.

The finding of this paper is that the terminating response is the one the evidence supports as the default, and that it can be had by reusing the C++26 `enforce` semantic and the existing termination rule, without a new semantic and without changing the meaning of `noexcept`. If a continuing response is to exist, the deployed precedents point to one shape: an opt-in, non-portable facility bounded to an adoption period rather than a default. Whether to build such a facility is a question for the committee, not this paper. The record is placed here for the committee's use; the paper makes no request.

---

## Acknowledgements

Ville Voutilainen, for the observation that the exception-safety hazard of a throwing implicit handler and the `noexcept` interaction are two views of one problem. The authors of P2900R14<sup>[2]</sup> and P3100R6<sup>[1]</sup>, whose careful statement of the design made the questions in this paper precise. Any errors are the author's own.

---

## References

[1] [P3100R6](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p3100r6.pdf) - "A framework for systematically addressing undefined behaviour in the C++ Standard" (Timur Doumler, Joshua Berne, 2026).

[2] [P2900R14](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p2900r14.pdf) - "Contracts for C++" (Joshua Berne, Timur Doumler, Andrzej Krzemie&nacute;ski, Ville Voutilainen, 2025).

[3] [P4306R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4306r0.pdf) - "Configuring Runtime Checking: Profiles and Implicit Contract Assertions" (Vinnie Falco, Ville Voutilainen, 2026).

[4] [P4297R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4297r0.pdf) - "Severing the Profiles Configuration Question from Case-by-Case Review" (Vinnie Falco, Ville Voutilainen, 2026).

[5] [libc++ Hardening Modes](https://libcxx.llvm.org/Hardening.html) - "Hardening Modes" (LLVM Project, 2025).

[6] [Using libstdc++ Macros](https://gcc.gnu.org/onlinedocs/libstdc%2B%2B/manual/using_macros.html) - "The GNU C++ Library Manual: Macros" (GNU Project, 2025).

[7] [MSVC STL Hardening](https://learn.microsoft.com/en-us/cpp/overview/cpp-conformance-improvements?view=msvc-170) - "C++ conformance improvements in Visual Studio" (Microsoft, 2025).

[8] [Source Fortification](https://www.sourceware.org/glibc/manual/latest/html_node/Source-Fortification.html) - "The GNU C Library: Source Fortification" (GNU Project, 2025).

[9] [Practical Security in Production](https://queue.acm.org/detail.cfm?id=3773097) - "Practical Security in Production: Hardening the C++ Standard Library at Massive Scale" (Louis Dionne, Alexander Rebert, Max Shavrick, Konstantin Varlamov, 2025).

[10] [Android UBSan](https://source.android.com/docs/security/test/ubsan) - "UndefinedBehaviorSanitizer" (Android Open Source Project, 2025).

[11] [Clang UBSan](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html) - "UndefinedBehaviorSanitizer" (LLVM Project, 2025).

[12] [P3911R2](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p3911r2.html) - "Make Contracts Reliably Non-Ignorable" (Darius Ne&abreve;&tcommaaccent;u, Andrei Alexandrescu, Lucian Radu Teodorescu, Radu Nichita, Herb Sutter, 2026).

[13] [BDE commit 03fdb2e1e](https://github.com/bloomberg/bde/commit/03fdb2e1ea0fe99e65cb429f69ba4a144da27417) - "Reduce clang-tidy and coverity warnings when using 'bslmt_once'" (Nathan Burgers, 2019).

[14] [P3191R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p3191r0.pdf) - "Feedback on the scalability of contract violation handlers in P2900" (Louis Dionne, Yeoul Na, Konstantin Varlamov, 2024).

[15] [P3198R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p3198r0.html) - "A takeaway from the Tokyo LEWG meeting on Contracts MVP" (Andrzej Krzemie&nacute;ski, 2024).

[16] [P3878R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3878r1.html) - "Standard library hardening should not use the observe semantic" (Ville Voutilainen, Jonathan Wakely, John Spicer, Stephan T. Lavavej, 2025).

[17] [P3097R2](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p3097r2.pdf) - "Contracts for C++: Virtual functions" (Timur Doumler, Joshua Berne, 2026).

[18] [__fastfail intrinsic](https://learn.microsoft.com/en-us/cpp/intrinsics/fastfail) - "__fastfail" (Microsoft, 2023).

[19] [glibc BZ #21754](https://sourceware.org/legacy-ml/libc-alpha/2017-08/msg00853.html) - "malloc: Abort on heap corruption without a backtrace" (Florian Weimer, 2017).

[20] [ERR56-CPP](https://wiki.sei.cmu.edu/confluence/display/cplusplus/ERR56-CPP.+Guarantee+exception+safety) - "ERR56-CPP. Guarantee exception safety" (SEI CERT C++ Coding Standard, 2023).

[21] [CHOP](https://download.vusec.net/papers/chop_ndss23.pdf) - "Let Me Unwind That For You: Exceptions to Backward-Edge Protection" (Victor Duta, Fabian Freyer, Fabio Pagani, Marius Muench, Cristiano Giuffrida, 2023).

[22] [P3290R4](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p3290r4.pdf) - "Integrating Existing Assertions with Contracts" (Joshua Berne, Timur Doumler, John Lakos, 2026).

[23] [cplusplus/papers #1943](https://github.com/cplusplus/papers/issues/1943) - WG21 public paper tracker issue for P3290, recording the SG22 2026-07-08 poll on assert exception propagation.

[24] [P3558R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3558r1.pdf) - "Prevent Undefined Behavior By Default" (Joshua Berne, John Lakos, 2025).

[25] [P3081R2](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3081r2.pdf) - "Core safety profiles for C++26" (Herb Sutter, 2025).

[26] [P0012R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/p0012r1.html) - "Make exception specifications be part of the type system, version 5" (Jens Maurer, 2015).

[27] [N3103](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3103.pdf) - "Security impact of noexcept" (David Kohlbrenner, David Svoboda, Andrew Wesie, 2010).

[28] [BDE commit c73a697a1](https://github.com/bloomberg/bde/commit/c73a697a193d65ab4ed3df5387fb3f3401f33486) - "Fix bsltf::AllocTestType test driver on C++11" (Alisdair Meredith, 2015).

[29] [P2877R0](https://isocpp.org/files/papers/P2877R0.pdf) - "Contract Build Modes, Semantics, and Implementation Strategies" (Joshua Berne, Tom Honermann, 2023).

[30] [CA2153](https://learn.microsoft.com/en-us/dotnet/fundamentals/code-analysis/quality-rules/ca2153) - "CA2153: Avoid handling Corrupted State Exceptions" (Microsoft, 2023).

[31] [bsls_review](https://github.com/bloomberg/bde/blob/main/groups/bsl/bsls/bsls_review.h) - "bsls_review: Provide facilities to identify and report on defense reviews" (Bloomberg BDE, 2019).

[32] [P3400R3](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p3400r3.pdf) - "Controlling Contract-Assertion Properties" (Joshua Berne, 2026).

[33] [P3589R2](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3589r2.pdf) - "C++ Profiles: The Framework" (Gabriel Dos Reis, 2025).
