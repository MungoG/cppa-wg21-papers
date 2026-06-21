# GMP, Boost.Multiprecision, and the Case for a Standard Big Integer

The interoperability that supposedly proves the standard needs a big integer type proves the opposite: the ecosystem already solved coordination without one, the only fast backend is the one the standard cannot legally ship, and the honest path is to standardize the shared primitives that make every existing library better rather than freeze a sixth competitor that serious consumers have already shown they will abandon.

## Executive Summary

A Boost.Multiprecision co-maintainer argued that GMP and Boost.Multiprecision work together, and that this interoperability is evidence for a standard vocabulary type at the interface boundary. The claim is half right, and the half that is wrong matters more.

The interoperability is real. Boost.Multiprecision wraps GMP behind a single front-end type, and code written against that front-end runs on GMP, on a header-only backend, or on four other backends without changes. That is a working vocabulary-type pattern. It proves people want a stable interface independent of the math engine underneath.

It also proves the ecosystem already solved the coordination problem without the standard. The same fact supports both conclusions, so on its own it decides nothing. Worse for the claim: the GMP backend is the one piece that cannot become the standard type. GMP is LGPL, and the libstdc++ maintainers have stated that LGPL plus C++ templates collapses into GPL. A standard `big_int` would ship the slower header-only implementation, and the high-performance users who motivate the abstraction would keep calling GMP directly. "They work together" does not transfer to the standard library.

The strongest evidence for standardization is narrower and better than the interoperability argument. One library inside Boost refuses to depend on Boost.Multiprecision, even at zero external dependency cost, and returns big numbers as strings instead. The standard library already carries a private big integer inside `from_chars`. These are real coordination costs. They are also concentrated in a handful of libraries, not spread across the ecosystem.

The recommended path is not a monolithic `std::big_int`. It is the pipeline the same authors are already building: standardize `_BitInt(N)`, 128-bit integers, carry and widening builtins, and non-transient constexpr allocation as independent primitives. Each delivers value on its own and makes every third-party big integer library better. Each generates the adoption data that would tell us whether a standard vocabulary type is still needed afterward. Four prior monolithic proposals died over 22 years on the same unresolved questions. A fifth that skips to the end of the queue invites the same fate.

Verdict: the interoperability claim is weak as stated (confidence: high). The coordination problem is real but smaller than for strings or paths (confidence: high). A decomposed pipeline beats a monolithic type (confidence: medium-high, resting on a clear WG21 pattern). A standard big integer may still be warranted at the end of that pipeline, on evidence not yet collected (confidence: medium).

---

## 1. The Verdict on the Claim

The claim breaks into four arguments that can be steel-manned and scored independently. One supports standardization. Three cut against it.

| Argument | Score | One-line basis |
|---|---|---|
| For: the front-end is a working vocabulary type | 3/5 | Pattern works, but one confirmed friction case plus an unelaborated "I have seen the friction" |
| Against: the ecosystem already solved it | 3/5 | Clean logic, vulnerable to the conditional-probability reframe |
| Against: unbundle the primitives | 4/5 | Constructive, already shipping, sidesteps the measurement deadlock |
| Against: decompose into a pipeline | 3/5 | Strong historical pattern, weak attribution of past failures to scope alone |

### What was actually said versus what the argument implies

The maintainer said three concrete things. Boost.Multiprecision offers conversion boilerplate so people do not write it repeatedly. There are people, including the maintainer, who do not want to pull in all of Boost.Multiprecision for a small part. People have requested new backends.

The argument needs more than this. It needs the conversion boilerplate to represent friction that only a standard type removes. It needs the "small part" problem to be unsolvable except by the standard. It needs the backend requests to mean users want one interface over many engines.

The backend detail does not survive contact with the transcript. When asked about the new backend, the maintainer clarified it was for double-double floating point, not an integer backend. The "people request backends" point, read as evidence that users want a vocabulary type over multiple integer engines, does not hold. The request was for a different number category entirely.

### "Works as a backend" is not "needed at the interface"

GMP working as a backend proves the abstraction can wrap a fast C library. It says nothing about whether a standard type must exist at the interface boundary. These are separate claims. The first is a statement about Boost.Multiprecision's internal design. The second is a statement about the ecosystem's need. The interoperability evidence establishes the first and gets borrowed to argue the second.

The borrow fails on licensing. The fastest backend is the one a standard library cannot adopt. So the interoperability that supposedly proves the need is built on a component that the standard type would have to exclude.

---

## 2. The Three Counter-Proposals

### Unbundle the primitives (strongest, 4/5)

The case for a standard type leans on "language blessings" a third-party library cannot get: compiler intrinsics for carry chains and widening multiply, constexpr evaluation through a compiler's internal big integer, structural-type support for use as a template argument. Bundle these with the type and only the standard can deliver them.

They are separable, and they are already being separated:

- `_BitInt(N)` standardizes fixed-width extended integers with no library type (P3666R4, on the C++29 track). It covers the compile-time-known-width cases that need no allocation.
- 128-bit integers provide the widening-multiply primitive every bignum implementation needs. The 64-by-64-to-128 multiply is the core operation. Once standardized, any library gets it portably.
- Carry-chain and widening builtins are proposed directly (P3161R5: `add_carry`, `sub_borrow`, `mul_wide`, `div_wide`). These are the operations bignum authors currently reach for through compiler-specific intrinsics.
- Non-transient constexpr allocation is a general language feature (P2670R1, P3032). Solve it once and every allocating type gains constexpr support, not just a standard big integer.
- Structural types are the one blessing that may genuinely need per-type compiler work. Whether that alone justifies a full library type is the open question.

Standardize the first four and a third-party big integer reaches performance parity, gains constexpr, and loses its portability excuse. The residual case for a standard type shrinks to pure coordination value: a common name everyone can assume exists. That is a legitimate argument. It is not the same argument as "only the standard can do this."

The authors proposing the standard type are themselves authors on the primitive papers. The pipeline is already running. The question is whether to acknowledge it and sequence deliberately, or to ship the endpoint first.

### Decompose into a pipeline (3/5)

Big proposals carry structural risk in WG21. The record is consistent. Staged features succeed: Ranges shipped a knowingly incomplete core in C++20 and filled tiers across C++23 and C++26. Coroutines shipped language support in C++20 and library types later. Formatting shipped before printing. Concepts succeeded on the fourth attempt after the monolithic 2008 design was pulled, by shrinking scope and shipping incrementally. Contracts succeeded on the fourth attempt with an explicit minimum-viable strategy. Executors succeeded only after the monolithic version was dropped and replaced by three orthogonal abstractions.

Monolithic proposals stall. Networking has been twenty-one years in progress, blocked each time its bundled async model was reworked. Four big integer proposals died over two decades.

A pipeline assigns each stage a job. Ship 128-bit integers, then measure whether bignum libraries adopt them and whether the portability complaint fades. Ship carry builtins, then measure whether the header-only backend closes the gap with GMP. Ship constexpr allocation, then measure whether third-party big integers gain compile-time evaluation. Each measurement informs the go/no-go for the next stage. A monolithic type designed today commits to a memory layout, an allocation strategy, and a constexpr model before any of these answers exist.

The weak point: a pipeline can strand the ecosystem. If stages one through four ship and the final type never does, libraries keep using incompatible big integers and the coordination problem persists. "Conditional" can mean "perpetually deferred." The mitigation is the train model. If the residual friction is real and measurable after the primitives ship, the evidence makes the final type easier to pass, not harder.

### Concept instead of concrete type (supporting)

Could a `big_integer` concept solve the problem without a concrete type? It would let generic algorithms accept GMP, the header-only backend, or any conforming type. The standard already works this way for `std::integral`, for iterators, for ranges.

A concept solves generic algorithms and nothing else. It cannot cross an ABI boundary, because a shared library export needs a concrete type with a fixed layout. It cannot be stored in a container. It cannot be a default function argument. It cannot appear in the non-template function signatures that make up most real code. A JSON value cannot hold "some type modeling `big_integer`"; it has to hold a specific type.

The decisive evidence is cross-language. Every language that solved this chose a concrete type. Java's `BigInteger` is a class, not an interface. Go's `big.Int` is a concrete struct, deliberately not an interface. Rust ships both a concrete `BigInt` and numeric traits, and concrete types dominate at API boundaries. Python and JavaScript built the big integer into the language. None solved coordination with an interface alone.

Boost.Multiprecision is itself the proof. Its backend protocol is a concept. Its `number<Backend>` front-end is still a concrete type, and that is what users store, return, and pass. The concept enabled multiple engines. The concrete type carried the vocabulary load. The strongest design provides both, which is what the standard would end up doing anyway.

---

## 3. The Coordination Problem

### The theory

A vocabulary type is the type that crosses interfaces between independently developed libraries. P2125R0 defines the category and its cost: when libraries each invent their own representation of the same concept, interoperation needs a conversion path between every pair, and the cost grows with the square of the number of types. A shared standard type collapses that to one conversion per library.

A standard type wins by salience, not by quality. This is a Schelling point. Strangers told to meet in a city with no prior contact converge on a famous landmark, not because it is the best meeting place but because it is the obvious one. A standard string won over every proprietary string class not by being faster than the alternatives but by being the type every developer could assume. A standard big integer would coordinate the same way. Even at a measured speed deficit, it would be the type a library author reaches for first, because it carries no dependency and no license negotiation.

### The evidence that it is real

The N-squared problem is not hypothetical. Production code converts between big integer representations through hand-written adapters, and where limb layouts disagree the fallback is a hex-string round trip. The OpenSSL GMP engine converts between its own BIGNUM and GMP's `mpz_t`, with a comment calling the string path "extremely inefficient." A widely used crypto library wraps GMP with import/export adapters to and from its own BigInt. The compiler itself converts its internal wide integer to `mpz_t`. A numerics library round-trips through decimal strings to reach GMP. The pattern repeats: each library invented a type, and the bridges are manual.

These bridges rarely surface as complaints. A conversion adapter is written once and then forgotten, which is why seventeen years of trillion-dollar cryptographic software produced no bug reports, forum threads, or conference talks lamenting the missing standard type. The coordination cost does not show up as loud friction. It shows up as interfaces that were never built, which is the harder thing to count.

At least eight distinct big integer types are in active C++ use: GMP, the header-only Boost backend, the Boost GMP wrapper, OpenSSL BIGNUM, and the bignums inside Botan, Crypto++, NTL, and several TLS stacks. The crypto types dominate by deployment. They are also carved out of the standard type's constituency, because they need constant-time guarantees a general type should not promise.

The single best specimen sits inside Boost. The JSON library represents numbers as 64-bit integers or doubles, and silently loses precision on anything larger. Users who need exact large integers must abandon the high-level API and drop to a streaming parser with a custom handler that receives the number as a string. The JSON library will not depend on Boost.Multiprecision to fix this, even though both libraries live in the same project at zero external dependency cost. When a co-author was pressed for an example of a library author who refuses big integers over a dependency, the answer named the other co-author of that same JSON library. If intra-project coupling is too expensive, the coordination cost is established.

The standard library already needs the type it does not expose. The `from_chars` implementations in major compilers carry a private big integer for exact rounding when the fast path fails. It ships inside several toolchains and a browser engine. The need is universal enough that the standard library implements it privately while declining to offer it.

A boundary scan of the most-used Java libraries found `BigInteger` in roughly a quarter of public APIs. The equivalent C++ number is near zero. Two explanations compete. C++ developers work closer to hardware and genuinely need big integers at boundaries less often. And the dependency cost suppresses the usage that would otherwise appear. Both are partly true. The absence of usage in a world with no standard type cannot be read as absence of demand for a standard type, because the absence is exactly what the missing type would produce.

### The historical analogy and its limit

Strings, callbacks, and filesystem paths each fragmented into a dozen incompatible types before the standard provided a focal point. Each standardization stopped the proliferation of new types and gave the old ones a clear conversion target, without ever eliminating them.

The analogy has a ceiling. Strings crossed nearly every interface. Big integers cross far fewer. The coordination benefit is real but smaller, which means the bar for justifying the standardization cost is higher, not lower.

---

## 4. The Technical Architecture

### How the wrapping works

Boost.Multiprecision splits into two layers. The front-end, `number<Backend, ExpressionTemplates>`, provides every operator, conversion, and I/O path. The backend implements raw arithmetic through a protocol of free functions named `eval_*`. A user picks a backend through a typedef and never touches it directly. `mpz_int` is `number<gmp_int>`.

Dispatch is argument-dependent lookup with a default fallback, not tag dispatch or concepts. The front-end calls an unqualified `eval_add` after a `using default_ops::eval_add` declaration. ADL finds the backend's overload in its own namespace first and uses the default only if none exists. For GMP, `eval_add` calls `mpz_add` directly. One call, no temporary.

The backend contract is specific. A backend lists its interoperable signed, unsigned, and float types, provides the four arithmetic `eval_*` functions plus modulus and bitwise operations for integers, supplies conversions to the built-in types, and declares a category tag. Optional overrides let a backend supply three-argument forms that avoid aliasing temporaries and fused operations like multiply-add, which GMP maps to `mpz_addmul`.

### What the abstraction costs

Expression templates are the signature optimization and the signature hazard. They defer evaluation so that a compound expression evaluates in one pass without intermediate heap allocations. They also store operands by reference, which creates the dangling-reference trap: an `auto` variable holding `a + b` holds references to `a` and `b`, and a temporary operand is destroyed at the semicolon, leaving the expression dangling. The official guidance is to never mix `auto` with expression templates. The library plants `static_assert` traps to catch the most common form. Recursive generic code hits a second version of the problem, because the deferred type is not the number type, and naive recursion deduces a fresh type at every level.

The benefit is questionable on modern compilers, by the original author's own statement, and fixed-precision types ship with the optimization off by default because it makes little difference and slows compilation.

### Where the backends diverge

Interchangeability holds for basic arithmetic on the same number category. It breaks at the edges, and the edges are numerous. The header-only integer uses sign-magnitude; GMP's behavior differs at the bit level, so bitwise operations and exported bit patterns disagree. A 128-bit sign-magnitude integer has a symmetric range, unlike a two's-complement type, so its bounds differ from the built-in expectation. Formatted output of negative values in hex or octal throws on the GMP backend and does not on the header-only one. Checked backends throw on overflow; unchecked ones wrap; GMP cannot overflow at all. Fixed and arbitrary precision diverge on truncation. Signed and unsigned diverge on negation. The `numeric_limits` values differ. Mixed-precision arithmetic across configurations is, by a maintainer's own description, really hard, and supported for only a small set of type pairs.

The interface is uniform. The behavioral contract is not. The abstraction resembles the way two standard containers can model the same concept while differing in invalidation and performance.

### The licensing trap

GMP is the fastest implementation, acknowledged as such by Boost's own documentation, and it is LGPL. The libstdc++ FAQ states the problem directly: the LGPL's replaceability requirement cannot be satisfied for a template-heavy C++ library, because templates expand inline into user code, so the LGPL becomes equivalent to the GPL. A standard library cannot adopt GMP without forcing that obligation on every user.

So a standard type ships the header-only implementation, which trails GMP by factors of two to five at the hundreds-to-thousands-of-digits range where multiprecision matters. The fast engine is legally unavailable; the available engine is slow. Performance-critical users keep calling GMP directly, which is the outcome the vocabulary type was meant to prevent.

---

## 5. The Historical Record

### Four failures over twenty-two years

The first attempts proposed an unlimited-precision integer and were never taken up; the committee asked for an analysis of use cases and which communities needed the type, and never received one. A second author submitted a single revision and did not return. A third reworked the design across multiple papers into an inheritance hierarchy with a non-standard allocator model and virtual functions on a value type, which was incompatible with how C++ value types work. The longest-running attempt spanned five revisions over two years, got the furthest, and died in the numerics study group when the broader technical-specification strategy that carried it was abandoned and the author stopped.

The recurring killers are consistent. Author attrition, every time. No reference implementation. Scope creep into signed plus unsigned plus modular plus rational plus primality plus random. And the allocator dilemma: templating on an allocator forces an overload for every built-in type in mixed arithmetic, while not templating loses customization. No proposal resolved it.

### The ecosystem routed around it

While the committee stalled, the ecosystem shipped. The clearest case involves the very library the interoperability claim cites. The Ethereum C++ toolchain, the EVM engine and the Solidity compiler among them, ran on Boost.Multiprecision, judged it too heavy on speed and binary size, and migrated to intx, a purpose-built header-only library. The migrations are recorded in the projects' own issue trackers. The boundary change was an improvement, not a breakage.

That is the vocabulary-type claim tested at a real boundary. A major consumer evaluated the candidate standard type, found it wanting, and replaced it. The need the committee debated for two decades, the ecosystem resolved in a release cycle by building something better and switching to it. A type frozen into the standard cannot make that move, which is the cost the abstraction argument leaves out.

### Why this attempt is different, and where it still bleeds

The current draft mitigates most of the historical failure modes. It has a three-person team with implementation experience and a sustained committee presence. It has a reference implementation. It narrows scope to one type. It stays out of a technical specification.

It still faces the question that killed the predecessors in new clothing. The draft floats reference counting with copy-on-write, and a committee member raised the exact failure of the reference-counted standard string: the threading requirements. Atomic reference counting costs around a dozen cycles per copy even single-threaded, and non-atomic counting fails the standard's cross-thread expectations. The same source that froze the standard string into a small-buffer redesign sits waiting here. The allocator question remains open. The representation wording remains vague.

The four failures are not only evidence of an unresolved design space. They are evidence that the monolithic approach is itself the failure mode. That is the case for sequencing the primitives first.

---

## 6. Supporting Detail

### The pipeline's measurement function

Each stage answers a question the next stage depends on.

- After 128-bit integers ship: do bignum libraries adopt them as the limb type, does throughput improve, does the portability complaint recede.
- After carry and widening builtins ship: does the header-only backend close the gap with GMP, do new libraries appear that use the portable primitives, how much intrinsic-specific code gets replaced.
- After constexpr allocation ships: do third-party big integers gain compile-time evaluation, does the "only the standard can be constexpr" motivation weaken, does the chosen mechanism even admit reference-counted types.

A monolithic type forecloses these answers by committing before they exist. Boost.Multiprecision's own history is the pipeline in miniature. It started with one backend, added a generic front-end, then added backends and optimizations one at a time across releases. The library that the claim cites as evidence was itself built incrementally, not shipped whole.

### What the standard library is for

The recurring committee position is that the standard library should hold vocabulary types, generic algorithms, and things that need compiler support, and should not become a package manager for high-quality libraries that already reached their audience. Standardization trims features and freezes the result. For a fast-moving design space that is a loss. Multiprecision arithmetic is a stable domain at the algorithm level, which weakens that objection. The implementation strategies, reference counting, expression templates, allocation, intrinsic access, are not stable, which restores part of it.

### The demand-suppression problem is structural

The cleanest argument for the type is also the hardest to measure. Usage observed in a world without a standard type cannot estimate usage in a world with one. The Boost.JSON specimen and the private `from_chars` big integer are the visible edge of the suppressed demand. The dozens of homebrew implementations and the six-backend abstraction layer inside one geometry library are the engineering cost made concrete. They are real. They are also concentrated in a small number of high-profile libraries rather than spread across the ecosystem, which is why the evidence persuades on quality and not on breadth.

---

## Conclusion

The interoperability of GMP and Boost.Multiprecision is a genuine achievement and a poor argument. It demonstrates the vocabulary-type pattern while proving the ecosystem already implemented it without the standard, and it rests on a backend the standard cannot ship. The coordination problem is real, narrower than the canonical precedents, and supported by a few strong specimens rather than a broad base.

The constructive move is to stop treating the type as a monolith. Ship the primitives that make every big integer library better and that the standard alone can provide. Measure what friction survives. Decide on the vocabulary type with evidence the four failed attempts never had. The same people making the case for the type are already building the pipeline. The recommendation is to name it and follow it, front to back, advancing only on evidence.

*2026-06-21 07:20 - claude-4.8-opus*
