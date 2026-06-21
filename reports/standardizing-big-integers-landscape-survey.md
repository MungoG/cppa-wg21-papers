# Standardizing Big Integers: Landscape Survey

The interoperability that supposedly proves the standard needs a big integer type proves the opposite: the ecosystem already solved coordination without one, the only fast backend is the one the standard cannot legally ship, and the honest path is to standardize the primitives that make every existing library better rather than freeze a sixth competitor that serious consumers have already abandoned.

## Executive Summary

A Boost.Multiprecision co-maintainer argued that GMP and Boost.Multiprecision work together, and that this interoperability is evidence for a standard vocabulary type at the interface boundary. The claim is half right. The half that is wrong matters more.

The interoperability is real. Boost.Multiprecision wraps GMP behind a single front-end type. Code written against that front-end runs on GMP, on a header-only backend, or on four other backends without changes. That is a working vocabulary-type pattern. It proves people want a stable interface independent of the math engine underneath.

It also proves the ecosystem already solved the coordination problem without the standard. The same fact supports both conclusions, so on its own it decides nothing. Worse: the GMP backend cannot become the standard type. GMP is LGPL. The libstdc++ maintainers have stated that LGPL plus C++ templates collapses into GPL. A standard `big_int` would ship the slower header-only implementation. High-performance users would keep calling GMP directly. "They work together" does not transfer to the standard library.

The strongest evidence for standardization is narrower and better than the interoperability argument. One library inside Boost refuses to depend on Boost.Multiprecision, even at zero external dependency cost, and returns big numbers as strings instead. The standard library already carries a private big integer inside `from_chars`. These are real coordination costs. They are concentrated in a handful of libraries, not spread across the ecosystem.

The recommended path is not a monolithic `std::big_int`. It is a pipeline: standardize 128-bit integers, carry and widening builtins (P3161R5), and non-transient constexpr allocation as independent primitives. Each delivers standalone value. Each makes every third-party big integer library better. Each generates adoption data that would tell us whether a standard vocabulary type is still needed afterward. Four prior monolithic proposals died over 22 years on the same unresolved questions. A fifth that skips to the end of the queue invites the same fate.

`_BitInt(N)` (P3666R4) is a separate track. It standardizes fixed-width extended integers for a different constituency. P3666's author frames it as competing with P3161's carry and widening operations, not enabling them. The primitives that matter for arbitrary precision - `mul_wide`, `add_carry`, `__int128` - are independently motivated and do not depend on `_BitInt`.

Verdict: the interoperability claim is weak as stated (confidence: high). The coordination problem is real but smaller than for strings or paths (confidence: high). A decomposed pipeline beats a monolithic type (confidence: medium-high - clear WG21 pattern). A standard big integer may still be warranted at the end of that pipeline, on evidence not yet collected (confidence: medium).

---

## 1. The Verdict on the Claim

The claim breaks into four arguments. One supports standardization. Three cut against it.

| Argument | Score | Basis |
|---|---|---|
| For: the front-end is a working vocabulary type | 3/5 | Pattern works. One confirmed friction case plus an unelaborated "I have seen the friction." |
| Against: the ecosystem already solved it | 3/5 | Clean logic. Vulnerable to the conditional-probability reframe. |
| Against: unbundle the primitives | 4/5 | Constructive. Already shipping. Sidesteps the measurement deadlock. |
| Against: decompose into a pipeline | 3/5 | Strong historical pattern. Weak attribution of past failures to scope alone. |

### What was said versus what the argument implies

The maintainer said three concrete things. Boost.Multiprecision offers conversion boilerplate so people do not write it repeatedly. There are people, including the maintainer, who do not want to pull in all of Boost.Multiprecision for a small part. People have requested new backends.

The argument needs more. It needs the conversion boilerplate to represent friction that only a standard type removes. It needs the "small part" problem to be unsolvable except by the standard. It needs the backend requests to mean users want one interface over many engines.

The backend detail does not survive the transcript. When asked about the new backend, the maintainer clarified it was for double-double floating point, not an integer backend. The "people request backends" point, read as evidence for a vocabulary type over multiple integer engines, collapses. The request was for a different number category.

### "Works as a backend" is not "needed at the interface"

GMP working as a backend proves the abstraction can wrap a fast C library. It says nothing about whether a standard type must exist at the interface boundary. These are separate claims. The first describes Boost.Multiprecision's internal design. The second describes the ecosystem's need. The interoperability evidence establishes the first and gets borrowed to argue the second.

The borrow fails on licensing. The fastest backend is the one a standard library cannot adopt. The interoperability that supposedly proves the need rests on a component the standard type would exclude.

---

## 2. The Three Counter-Proposals

### Unbundle the primitives (strongest, 4/5)

The case for a standard type leans on "language blessings" a third-party library cannot get: compiler intrinsics for carry chains and widening multiply, constexpr evaluation through a compiler's internal big integer, structural-type support for template arguments. Bundle these with the type and only the standard can deliver them.

They are separable. They are already being separated:

- 128-bit integers provide the widening-multiply primitive every bignum implementation needs. The 64-by-64-to-128 multiply is the core limb operation. Standardize it and any library gets it portably.
- P3161R5 proposes carry-chain and widening builtins directly: `add_carry`, `sub_borrow`, `mul_wide`, `div_wide`. These are the operations bignum authors currently reach for through compiler-specific intrinsics.
- Non-transient constexpr allocation is a general language feature (P2670R1, P3032). Solve it once and every allocating type gains constexpr support, not just a standard big integer.
- Structural types are the one blessing that may genuinely need per-type compiler work. Whether that alone justifies a full library type is the open question.

`_BitInt(N)` (P3666R4) covers fixed-width extended integers. It does not provide the primitives arbitrary precision needs. P3666 explicitly frames `_BitInt` as making P3161's carry and widening operations "arguably obsolete" - the opposite of enabling them. The two papers are independent tracks with different constituencies. `_BitInt` matters for embedded and crypto users who know their width at compile time. It is not a dependency of the arbitrary-precision pipeline.

Arbitrary precision has three layers.

First, dynamic storage: the growable limb vector. Constexpr allocation and the existing allocator machinery address this.

Second, the limb-level inner loop: schoolbook add, subtract, multiply-accumulate. `mul_wide`, `add_carry`, and `__int128` target this directly.

Third, the asymptotically fast algorithms. Karatsuba (1962). Toom-Cook (1963). FFT multiplication (1971). Newton division. Subquadratic base conversion.

The primitives address layers one and two. Layer three stays in the ecosystem. That is the point. GMP, Boost.Multiprecision, and any future library keep iterating on those algorithms with better building blocks underneath. The standard does not freeze them.

Below roughly two thousand bits the inner loop dominates. The primitives close the performance gap there. Above that the fast algorithms take over, but that is the crypto and number-theory regime the current proposal concedes a standard type will not serve. The primitives capture most of the portable-performance value at the operand sizes the general-purpose constituency uses.

Standardize the first three and a third-party big integer gains a fast portable inner loop, constexpr support, and no portability excuse. The residual case for a standard type shrinks to coordination value: a common name everyone can assume exists. That is legitimate. It is not the same argument as "only the standard can do this," and it is separable from the primitives that deliver the performance.

### Decompose into a pipeline (3/5)

Big proposals carry structural risk in WG21. The record is consistent.

Staged features succeed. Ranges shipped a knowingly incomplete core in C++20 and filled tiers across C++23 and C++26. Coroutines shipped language support in C++20 and library types later. Formatting shipped before printing. Concepts succeeded on the fourth attempt after the monolithic 2008 design was pulled and the scope shrank. Contracts succeeded on the fourth attempt with a minimum-viable strategy. Executors succeeded only after the monolithic version was dropped and replaced by three orthogonal abstractions.

Monolithic proposals stall. Networking has been twenty-one years in progress, blocked each time its bundled async model was reworked. Four big integer proposals died over two decades.

A pipeline assigns each stage a job. Ship 128-bit integers, then measure whether bignum libraries adopt them and whether the portability complaint fades. Ship carry builtins, then measure whether the header-only backend closes the gap with GMP. Ship constexpr allocation, then measure whether third-party big integers gain compile-time evaluation. Each measurement informs the go/no-go for the next stage. A monolithic type designed today commits to a memory layout, an allocation strategy, and a constexpr model before any of those answers exist.

Two weak points. First, a pipeline can strand the ecosystem. If every stage ships and the final type never does, libraries keep using incompatible big integers. "Conditional" can mean "perpetually deferred." The mitigation is the train model. If the residual friction is real and measurable after the primitives ship, the evidence makes the final type easier to pass, not harder.

Second, no one has named this pipeline. The primitive papers are independent proposals by different authors. Framing them as a coordinated sequence is a retrospective narrative, not an organizational reality the committee recognizes. Making the pipeline real requires someone to champion it as a whole, not just champion the individual papers. That governance work has not been done.

### Concept instead of concrete type (supporting)

Could a `big_integer` concept solve the problem without a concrete type? It would let generic algorithms accept GMP, the header-only backend, or any conforming type. The standard already works this way for `std::integral`, for iterators, for ranges.

A concept solves generic algorithms. Nothing else. It cannot cross an ABI boundary. It cannot be stored in a container. It cannot be a default function argument. It cannot appear in the non-template function signatures that make up most real code. A JSON value cannot hold "some type modeling `big_integer`." It has to hold a specific type.

Every language that solved this chose a concrete type. Java's `BigInteger` is a class, not an interface. Go's `big.Int` is a concrete struct, deliberately not an interface. Rust ships both a concrete `BigInt` and numeric traits. Concrete types dominate at API boundaries. Python and JavaScript built the big integer into the language. None solved coordination with an interface alone.

Boost.Multiprecision is itself the proof. Its backend protocol is a concept. Its `number<Backend>` front-end is still a concrete type. That is what users store, return, and pass. The concept enabled multiple engines. The concrete type carried the vocabulary load.

---

## 3. The Coordination Problem

### The theory

A vocabulary type crosses interfaces between independently developed libraries. P2125R0 defines the category and its cost: when libraries each invent their own representation of the same concept, interoperation needs a conversion path between every pair. The cost grows with the square of the number of types. A shared standard type collapses it to one conversion per library.

A standard type wins by salience, not by quality. This is a Schelling point. A standard string won over every proprietary string class not by being faster but by being the type every developer could assume. A standard big integer would coordinate the same way: no dependency, no license negotiation, the obvious default.

### The evidence that it is real

The N-squared problem is not hypothetical. Production code converts between big integer representations through hand-written adapters. Where limb layouts disagree, the fallback is a hex-string round trip.

The OpenSSL GMP engine converts between its own BIGNUM and GMP's `mpz_t`. A comment in the source calls the string path "extremely inefficient." A widely used crypto library wraps GMP with import/export adapters. The compiler itself converts its internal wide integer to `mpz_t`. A numerics library round-trips through decimal strings to reach GMP. Each library invented a type. The bridges are manual.

These bridges rarely surface as complaints. A conversion adapter is written once and forgotten. Seventeen years of trillion-dollar cryptographic software produced no bug reports, forum threads, or conference talks about the missing standard type. The coordination cost does not present as loud friction. It presents as interfaces that were never built. That is harder to count.

Eight distinct big integer types are in active C++ use: GMP, the header-only Boost backend, the Boost GMP wrapper, OpenSSL BIGNUM, and the bignums inside Botan, Crypto++, NTL, and several TLS stacks. The crypto types dominate by deployment. They are carved out of the standard type's constituency. They need constant-time guarantees a general type should not promise.

The single best specimen sits inside Boost. The JSON library represents numbers as 64-bit integers or doubles. Anything larger silently loses precision. Users who need exact large integers must abandon the high-level API and drop to a streaming parser that receives the number as a string.

The JSON library will not depend on Boost.Multiprecision. Both libraries live in the same project. The external dependency cost is zero. When a co-author was pressed for an example of a library author who refuses big integers over a dependency, the answer named the other co-author of that same JSON library. If intra-project coupling is too expensive, the coordination cost is established.

The standard library already needs the type it does not expose. The `from_chars` implementations in major compilers carry a private big integer for exact rounding when the fast path fails. The code ships inside several toolchains and a browser engine.

An original boundary scan of the 148 most-used Java libraries found `BigInteger` in roughly a quarter of public APIs. The equivalent C++ number is near zero. Two explanations compete: C++ developers work closer to hardware and genuinely need big integers at boundaries less often, and the dependency cost suppresses the usage that would otherwise appear. Both are partly true. Absence of usage in a world with no standard type cannot be read as absence of demand, because the absence is exactly what the missing type would produce.

### The historical analogy and its limit

Strings, callbacks, and filesystem paths each fragmented into a dozen incompatible types before the standard provided a focal point. Each standardization stopped the proliferation of new types and gave the old ones a clear conversion target, without eliminating them.

The analogy has a ceiling. Strings crossed nearly every interface. Big integers cross far fewer. The coordination benefit is real but smaller. The bar for justifying standardization cost is higher.

---

## 4. The Technical Architecture

### How the wrapping works

Boost.Multiprecision splits into two layers. The front-end, `number<Backend, ExpressionTemplates>`, provides every operator, conversion, and I/O path. The backend implements raw arithmetic through free functions named `eval_*`. A user picks a backend through a typedef. `mpz_int` is `number<gmp_int>`.

Dispatch is argument-dependent lookup with a default fallback. The front-end calls an unqualified `eval_add` after a `using default_ops::eval_add` declaration. ADL finds the backend's overload first. For GMP, `eval_add` calls `mpz_add` directly. One call, no temporary.

A backend lists its interoperable signed, unsigned, and float types, provides the four arithmetic `eval_*` functions plus modulus and bitwise operations for integers, supplies conversions to the built-in types, and declares a category tag. Optional overrides let a backend supply three-argument forms and fused operations. GMP maps multiply-add to `mpz_addmul`.

### What the abstraction costs

Expression templates defer evaluation so a compound expression evaluates in one pass without intermediate heap allocations. They also store operands by reference. An `auto` variable holding `a + b` holds references to `a` and `b`. A temporary operand is destroyed at the semicolon, leaving a dangling reference. The official guidance: never mix `auto` with expression templates. The library plants `static_assert` traps for the most common form.

The benefit is questionable on modern compilers, by the original author's own statement. Fixed-precision types ship with the optimization off by default.

### Where the backends diverge

Interchangeability holds for basic arithmetic on the same number category. It breaks at the edges.

The header-only integer uses sign-magnitude. GMP differs at the bit level. Bitwise operations and exported bit patterns disagree. A 128-bit sign-magnitude integer has a symmetric range unlike two's complement. Formatted output of negative values in hex throws on the GMP backend and not on the header-only one. Checked backends throw on overflow. Unchecked ones wrap. GMP cannot overflow at all. Fixed and arbitrary precision diverge on truncation. Mixed-precision arithmetic across configurations is, by a maintainer's description, "really hard," and supported for only a small set of type pairs.

The interface is uniform. The behavioral contract is not.

### The licensing trap

GMP is the fastest implementation, acknowledged by Boost's own documentation. It is LGPL. The libstdc++ FAQ states the problem: the LGPL's replaceability requirement cannot be satisfied for a template-heavy C++ library, because templates expand inline into user code. The LGPL becomes equivalent to the GPL. A standard library cannot adopt GMP without forcing that obligation on every user.

A standard type ships the header-only implementation. Below about fifty digits, the header-only backend matches or beats GMP because it uses stack allocation where GMP hits the heap. Above that, at hundreds to thousands of digits, GMP leads by factors of two to five. That is the range where multiprecision matters. The fast engine is legally unavailable. The available engine is slow where it counts. Performance-critical users keep calling GMP directly. That is the outcome the vocabulary type was meant to prevent.

---

## 5. The Historical Record

### Four failures over twenty-two years

The first attempts proposed an unlimited-precision integer. The committee asked for use-case analysis and never received one. A second author submitted a single revision and did not return. A third reworked the design into an inheritance hierarchy with virtual functions on a value type, incompatible with C++ idioms. The longest-running attempt spanned five revisions, got the furthest, and died in the numerics study group when the broader technical-specification strategy was abandoned.

The recurring killers: author attrition, every time. No reference implementation. Scope creep into signed plus unsigned plus modular plus rational plus primality. The allocator dilemma: templating on an allocator forces an overload for every built-in type in mixed arithmetic. Not templating loses customization. No proposal resolved it.

### The ecosystem routed around it

While the committee stalled, the ecosystem shipped. The Ethereum C++ toolchain - the EVM engine and the Solidity compiler among them - ran on Boost.Multiprecision, judged it too heavy on speed and binary size, and migrated to intx, a purpose-built header-only library. The migrations are recorded in the projects' issue trackers. The boundary change was an improvement, not a breakage.

That is the vocabulary-type claim tested at a real boundary. A major consumer evaluated the candidate standard type, found it wanting, and replaced it. The need the committee debated for two decades, the ecosystem resolved in a release cycle. A type frozen into the standard cannot make that move.

### Where the current attempt differs, and where it bleeds

The current draft mitigates most of the historical failure modes. Three-person team with implementation experience and sustained committee presence. A reference implementation. Scope narrowed to one type. No technical specification.

The draft floats reference counting with copy-on-write. A committee member raised the exact failure of the reference-counted standard string: the threading requirements. Atomic reference counting costs around a dozen cycles per copy even single-threaded. Non-atomic counting fails the standard's cross-thread expectations. The same trap that forced the standard string into a small-buffer redesign sits waiting here. The allocator question remains open. The representation wording remains vague.

Four failures are evidence of an unresolved design space. They are also evidence the monolithic approach is the failure mode. That is the case for sequencing the primitives first.

---

## 6. Supporting Detail

### The pipeline's measurement function

Each stage answers a question the next stage depends on.

- After 128-bit integers ship: do bignum libraries adopt them as the limb type, does throughput improve, does the portability complaint recede.
- After carry and widening builtins ship: does the header-only backend close the gap with GMP, do new libraries appear on the portable primitives, how much intrinsic-specific code gets replaced.
- After constexpr allocation ships: do third-party big integers gain compile-time evaluation, does the "only the standard can be constexpr" motivation weaken, does the chosen mechanism admit reference-counted types.

A monolithic type forecloses these answers by committing before they exist. Boost.Multiprecision's own history is the pipeline in miniature. It started with one backend, added a generic front-end, then added backends and optimizations one at a time across releases. The library cited as evidence was itself built incrementally, not shipped whole.

### The `_BitInt` tension

P3666R4 and P3161R5 occupy competing positions. P3666 argues that `_BitInt(N)` makes P3161's widening and carry operations "arguably obsolete" - cast the operands to double width before multiplying and the dedicated primitives become unnecessary. P3161 makes zero reference to `_BitInt`. The two proposals are independent tracks motivated differently: P3666 from C compatibility and fixed-width use cases, P3161 from hardware ISA mapping.

The compiler machinery for `_BitInt` - GCC's `__mulbitint3`, LLVM's `APInt::tcMultiply` - uses the same widening multiply and carry propagation that P3161 standardizes. But those builtins predate `_BitInt` by over a decade. `_BitInt` consumes the machinery. It did not create it.

For the pipeline, this means `_BitInt` and P3161 are alternatives at the limb level, not a dependency chain. The arbitrary-precision pipeline does not require `_BitInt` as a precursor. It requires `mul_wide`, `add_carry`, `__int128`, and constexpr allocation. Those stand on their own.

### What the standard library is for

The recurring committee position: the standard library should hold vocabulary types, generic algorithms, and things needing compiler support. It should not become a package manager for high-quality libraries that already reached their audience. Standardization trims features and freezes the result. Multiprecision arithmetic is stable at the algorithm level, which weakens that objection. The implementation strategies - reference counting, expression templates, allocation, intrinsic access - are not stable, which restores part of it.

### The demand-suppression problem

The cleanest argument for the type is the hardest to measure. Usage observed in a world without a standard type cannot estimate usage in a world with one. The Boost.JSON specimen and the private `from_chars` big integer are the visible edge of suppressed demand. The dozens of homebrew implementations and the six-backend abstraction layer inside one geometry library are the engineering cost made concrete. They are real. They are concentrated in a small number of high-profile libraries rather than spread across the ecosystem. The evidence persuades on quality, not breadth.

---

## Conclusion

The interoperability of GMP and Boost.Multiprecision is a genuine achievement and a poor argument. It demonstrates the vocabulary-type pattern while proving the ecosystem already implemented it without the standard. It rests on a backend the standard cannot ship. The coordination problem is real, narrower than the canonical precedents, supported by a few strong specimens rather than a broad base.

Ship the primitives that make every big integer library better. Measure what friction survives. Decide on the vocabulary type with evidence the four failed attempts never had. The same people making the case for the type are already building the pipeline. Name it. Follow it front to back. Advance only on evidence.

*2026-06-21 10:16 - claude-4.8-opus*
