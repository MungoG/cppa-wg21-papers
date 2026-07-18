---
title: "An Initialization Profile: Compile-Time Guarantees for Uninitialized Memory in C++"
document: D4222R3
date: 2026-07-17
intent: info
audience: EWG, SG12, SG20, SG23
reply-to:
  - "Bjarne Stroustrup <bjarne@stroustrup.com>"
---

<!--
DEMONSTRATION DRAFT - NOT FOR PUBLICATION.
This is a papersmith rewrite of P4222R2 prepared as a workflow demonstration for
Bjarne Stroustrup. It is not an official revision and is not represented as the
author's own text. The R2 -> R3 increment is provisional and requires the author's
approval. Every design decision in R2 is preserved; only presentation was changed.
Working source: a provisional capture of P4222R2 (canonical PDF unavailable on 2026-07-17).
-->

**Demonstration draft, not for publication.** This document is a rewrite of P4222R2 produced to demonstrate a drafting workflow. It preserves every technical decision of the author's R2 and changes only presentation. The revision increment to R3 is provisional and is not adopted without the author's approval.

## Abstract

The C++ initialization profile closes the read-or-write-before-initialization gap for the code that opts in, entirely at compile time and at no run-time cost.

Reading or writing an object before it is initialized is one of the oldest error sources in C and C++, and the language has never fully closed it. The profile is foundational, because most other profiles and most ordinary code rely on the objects they touch being initialized, and for such code C++26's erroneous behavior is too weak, since its reaction is implementation-defined. The hard part is that C++ deliberately passes uninitialized memory around in performance buffers, memory pools, and slot-based containers, which the profile must keep supporting. It marks that deliberate use with three attributes (`[[uninit]]`, `[[ref_to_uninit]]`, `[[must_init]]`) and one trivial function template, `now_init()`, which confines each unverifiable step to a single reviewable point, all enforced by local static analysis so that ordinary code is untouched and only deliberate uses are checked. It rejects definite assignment and any partial static guarantee, requiring suppression where initialization cannot be proven. Being opt-in, it does not compel C, old-style code, or system headers to adopt it, and an implementation exists but for the concept-based overloading of Section 4.8, which turns on an unresolved type-deduction question. The design is offered as input to implementation work, with further change expected, and rests on one property: because these attributes stay out of the type system, code means the same thing whether or not the profile is enforced.

## Revision History

### R3 (demonstration draft): July 2026

- Presentation-only rewrite of R2. No design decision was changed: the three attributes (`[[uninit]]`, `[[ref_to_uninit]]`, `[[must_init]]`), `now_init()`, the choice to keep the attributes out of the type system, local-analysis-only enforcement, the same-meaning principle, the rejection of definite assignment, and every rejected alternative are preserved.
- Restructured for a reader who reads in passes: the finding leads each section, evidence precedes evaluation, and every code block carries a caption stating its provenance, status, and purpose.
- Corrected accidental defects in the code examples so they compile as intended, while preserving every deliberate `// error` illustration unchanged.
- Added a glossary for the profile vocabulary (Appendix B), a numbered list of contributions and stated assumptions (Section 1), and a Conclusion.

### R2: 2026-07-11

- Shortened `[[uninitialized]]` to `[[uninit]]` (Section 4.1) and elaborated the rationale for using attributes rather than modifying the type system (Section 4.1).
- Explained the use of `void*` (Section 4.3) and the interaction between `[[ref_to_uninit]]` and templates (Section 4.4).
- Presented the beginnings of the library techniques for simpler safe code (Sections 4.4, 4.5); discussed `construct_at()` and `destroy_at()` (Section 4.5).
- Introduced `now_init()` to limit the need for profile suppression (Sections 4.2, 4.4) and `[[must_init]]` to ease initialization functions.
- Elaborated the initialization of classes without constructors (Section 5.3); discussed and partly rejected definite assignment (Section 1.4).
- Extended the draft wording (Section 9) and discussed the choice of notation (Appendix).

## 1. Introduction

Uninitialized memory has been a source of error in C and C++ since the beginning, and no revision of either language has fully closed it. The initialization profile addresses this gap for the code that opts in. The profile is foundational, because most other profiles and most ordinary code rely on the objects they touch being initialized. It should therefore be the easiest profile to define. It is not, because the rules governing initialization and uninitialized memory in C++ are more intricate than they first appear. The design goal is to isolate that intricacy so that ordinary code stays simple and the complexity falls only on the code that deliberately manipulates uninitialized memory.

An earlier design of the profile was reviewed by EWG<sup>[1]</sup>, and profiles are now processed by SG23. This revision emphasizes the rationale, weighs a few alternatives, and suggests simplifications; it is written to inform implementation work and to let readers gauge the profile's likely effect on existing code, so further design change is expected. With the exception of the concept-based overloading in Section 4.8 and some notational details still being matched to the latest specification, an implementation exists.

The design rests on a small set of requirements:

- Every object is initialized, or marked as uninitialized, at its point of definition.
- Reading or writing uninitialized memory is an error.
- No complex flow analysis is required, and no run-time tests are needed.
- A class ensures that every accessible data member is initialized before use.
- Implicit initialization, through a default constructor or a language rule, is initialization.
- There is a way to pass uninitialized memory out of a scope (Section 4.3).
- Initializing an object twice is an error; destruction makes an object uninitialized; destroying an object twice is an error, because the second destruction operates on raw memory.
- The design is usable by most developers without an understanding of tricky concepts or implementation detail, and existing well-written code works unmodified.
- Code too complex for the profile can be expressed by suppressing the profile.
- The initialization attributes do not modify the type system; they carry the information the profile's enforcement needs.

Use of profiles is optional. Code that cannot reasonably be expressed under the initialization profile can be written without the profile, or with the profile suppressed for the region that needs it. The guarantees most language rules already assume, that no uninitialized object is read from (or, for a class with an assignment operator, written to), are what this profile enforces.

For arbitrarily complex code the "every object is initialized" property cannot be established statically, so the profile provides notation that carries enough information for every potential use of an uninitialized object to be detected at compile time. A profile is best specified by the guarantee it offers, here "no use of uninitialized objects," rather than by an enumeration of every affected construct (Section 8); the guarantee is what simplifies both understanding and use.

The mechanisms are small: three attributes and one trivial function template. The rejected alternatives and the cases that motivate the mechanisms occupy the sections that follow; had it not been for the need to manipulate uninitialized memory (Section 1.1), the profile would fit in a few pages. This profile is designed to fit the proposed profiles framework<sup>[2]</sup>.

This paper contributes:

1. A rationale for enforcing initialization through attributes that inform analysis rather than through a change to the type system (Section 4.1), and the argument that this choice is what preserves identical meaning with and without the profile.
2. Three attributes (`[[uninit]]`, `[[ref_to_uninit]]`, `[[must_init]]`) and one function template (`now_init()`) that together express the intentional use of uninitialized memory, with their rules (Sections 4, 6).
3. The beginnings of the library techniques that keep the attributes out of most application code (Sections 4.4, 4.7).
4. An account of what the profile deliberately does not attempt, and why, including the rejection of definite assignment (Section 1.4) and the open concept-overloading problem (Section 4.8).

The design rests on three assumptions, stated here so a reader can test them against what follows. First, code compiled with the profile and the same code compiled without it must mean the same thing; this rules out any mechanism that changes overload resolution or type identity. Second, only local static analysis is available; no whole-program or symbolic analysis is assumed. Third, the profile is opt-in, and some code, particularly code in older styles and C system headers, will not adopt it for a long time.

### 1.1. Buffers and memory pools

The complexity the profile must absorb comes from performance-critical memory buffers and user-defined memory pools, a pattern C++ uses to a degree that Ada, Java, C#, and other garbage-collected languages do not. When a buffer is filled under severe performance constraints, initializing it first with default values and overwriting them later adds overhead that some applications cannot afford, and the hardware support that would hide that overhead is not available everywhere. `std::vector` itself lives with this: it manages a mixture of initialized and uninitialized memory, usually through `construct_at()` and `destroy_at()`.

Allocators routinely hand out an uninitialized region of memory for some other class or function to manage. I would like a single abstraction to handle this, but we do not have one, and I suspect the range of uses is too broad for us to get one soon. I would love to be proven wrong, but I will proceed as though I will not. What we have is `void*`, a pointer to memory of unknown type (Section 4.3). Large uninitialized buffers are common for good reasons, and that use is why local variables were left uninitialized in C (I once asked Dennis). Consequently, the profile needs a concrete thing: a way to say that a pointer or smart pointer passed into or out of a function refers to an object that contains uninitialized memory, such as an array from an allocator.

Two simpler designs were considered and rejected. Banning the passing of objects with directly accessible uninitialized memory, by requiring such data to be private members so that a class owns the obligation to initialize, would forbid passing pointers and spans to structs and arrays that have uninitialized members, and would require the profile to be suppressed across much existing critical code. Leaving programmers to suppress the profile with no in-code hint of why is unmanageable for some applications and compromises the profile's value. Neither is acceptable, so the profile provides explicit notation instead.

### 1.2. Uninitialized memory

Memory is deliberately left uninitialized in several recurring places, each with the intent of turning it into a properly initialized object later:

- A local variable.
- A member of a class.
- Free-store memory for a type without default initialization, allocated by `new`.
- Free-store memory from a non-initializing allocator such as `malloc()`.

These objects are usually uninitialized for good reasons, but a variable left uninitialized by mistake becomes a later source of error. The profile aims to ban the use of uninitialized memory except where it is explicitly requested, to use that ban to support and simplify reasonable initialization techniques, and to make the remaining uses less error-prone. Initialization, or the explicit suppression of it, is stated at the point of definition, and every point where uninitialized memory is passed out of a scope is marked.

This is the subset-of-superset strategy: notation and libraries are added so that analysis becomes tractable, and only then is the language subset to the part over which the guarantee holds. The simplest code follows from not defining an object until there is a value to initialize it with, and that is the recommended default. The profile does not aim to admit every technique; those too error-prone or too complex to validate statically are relegated to code that suppresses the profile. Using random access to selectively initialize elements of an uninitialized array, for instance, cannot be validated statically (Section 5.4), so it is either banned or relegated to suppressed code.

Because the profile's guarantees are relied on by most code and most other profiles, its enforcement cost must be low at compile time and zero at run time. C++26 already improved matters by making a read of an uninitialized variable erroneous behavior rather than undefined behavior<sup>[3]</sup>, but erroneous behavior is not what the profile needs, because its response is implementation-defined:

- Testing for erroneous behavior can be a run-time action, and portable code must cope with whatever the violation handling does.
- If the response is to substitute a default value, the result can be a logical error.
- If the response is termination, possibly delayed, the code is unusable in applications that cannot tolerate unconditional termination.

The profile offers a stronger guarantee at the cost of stricter rules of use. Nearly all of its complexity comes from supporting the handling of uninitialized memory rather than from the plain requirement that every object be initialized. Like every profile, it is opt-in, and some code, typically written in older styles, will never adopt it (Section 7).

### 1.3. Static analysis without complex flow analysis

The profile relies on static analysis, ideally in the compiler, to reject unsafe uses, and it is designed so that the analysis stays cheap. One question is local - "does this definition have an initializer?" - and covers the majority of definitions. A second requires flow analysis: "is this uninitialized object constructed before use?" That second kind is what lets the profile handle buffers of uninitialized memory, as in the implementation of `std::vector`, and members initialized in a constructor body rather than in a member initializer.

To keep the analysis affordable, every branch of a run-time conditional is treated as executed for the purpose of the guarantee. The alternative, global analysis or symbolic execution, is unaffordable, and adding run-time checks is ruled out by the zero-cost requirement. The profiles considered here, including this one, require only local static analysis. Random access into an uninitialized array is banned for the same reason: allowing it would make the initialize-before-use guarantee impossible to enforce statically (Section 5.4). Errors of misuse after initialization, such as access after deletion (Section 4.6), are left to other profiles.

### 1.4. Definite assignment is rejected

Some languages take a different route. Ada and C# enforce that a variable is assigned before use, a rule called "definite assignment"; Java default-initializes every object. Neither fits C++, for several reasons:

- Initialization and assignment can mean very different things.
- The address of some memory can be passed out of a scope to be initialized elsewhere, for example in a separate translation unit.
- Delayed initialization can confuse a reader, especially during maintenance.
- Some constructs, such as loops and conditionals, make it impossible to determine at compile time whether an object has been initialized.
- Requiring the analysis to be exact, even where it is theoretically possible, would place a heavy burden on implementers.
- A compiler that can delay initialization without changing semantics may already do so as an optimization.

This design does not change the semantics of correct C++. The following illustrates the ambiguity that definite assignment would introduce; it is hypothetical code showing why a deferred initialization disguised as assignment cannot be reasoned about locally:

```cpp
void f(int v)
{
    X x;   // maybe uninitialized
    // ...
    x = v; // maybe an initialization
}
```

The meaning depends entirely on what `X` is:

- If `X` is `std::byte`, `x` is uninitialized and the assignment is valid.
- If `X` is `int`, `x` is uninitialized; depending on the type of `v` the assignment is implementation-defined or undefined in older compilers, and erroneous behavior in C++26.
- If `X` is `std::string`, `x` is default-constructed and the assignment is well defined, but unless the default assignment is optimized away, the generated code differs from a simple initialization.
- If `X` is a class without a default constructor, `x` is uninitialized and the assignment is undefined in older compilers, erroneous behavior in C++26.
- If `X` is a class where initialization and assignment differ, code compiled with and without the profile would differ.

Writing generic code over such a definite-assignment extension would produce code whose meaning changed with and without the profile, and that is the one outcome the design does not permit. Ways to delay initialization safely are shown in Section 5.4.

## 2. Implicit initialization is initialization

The profile treats initialization that the language already performs as initialization, so that existing well-formed code is not disturbed. A definition that invokes a default constructor is initialized (Section 5.1). A static variable with a default initializer is initialized (Section 3). A dynamically created object, using `new` with a default initializer or a default constructor, is initialized. In each case the profile adds no obligation beyond what the language already guarantees.

## 3. Static objects must be initialized before run time

Static objects in different translation units are initialized in an implementation-defined order, which can let one be read before it is initialized. Consider two translation units that read each other's globals during initialization, existing C++ that shows the hazard the profile must rule out:

```cpp
int f() { extern int y; return y; } // f.c
int x = f();

int g() { extern int x; return x; } // g.c
int y = g();
```

Either initialization order reads an uninitialized variable. Good developers avoid examples this stark, but not every case is this obvious, and the profile aims to guarantee initialization before use. The rule is therefore:

- Non-local static objects are initialized at compile time or link time. No run-time initialization is allowed for such objects.

This rule seems Draconian, but it conforms to common practice and has language support. There are four established ways to obey it:

- Do not use global variables.
- Use only `constinit` global variables.
- Make default constructors `consteval`.
- Wrap statics in functions that guarantee initialization before use.

Variations of all four have been used for decades; `constinit` is only a recent, direct form of an old technique. The function-wrapping form looks like this; it is existing C++, shown because it is the general technique for controlling when a static is initialized:

```cpp
X& var() { static X v = init(); return v; }
```

Wrapping a static in a function or class controls the timing of its initialization. When an initialization is too complex for these techniques to guarantee, the profile is suppressed for that one initialization. The rule keeps the initialization order of statics out of the class of things that can silently read uninitialized memory.

## 4. Marking uninitialized memory without changing the type system

A profile that guarantees initialization still needs a way to say "not initialized," because C++ programs pass uninitialized memory around, usually in order to initialize it (Section 1.1), and in rare cases must postpone the initialization of a class member. Most of the profile's complexity traces to those two uses. For everything else, "just initialize all objects" is the right rule: treat every introduction of an uninitialized object as an optimization technique, and therefore as a potential source of complexity and logic error. The profile's contribution here is to make the absence of initialization visible in the code and to remove the errors it would otherwise cause.

C++26 already has `[[indeterminate]]`, but it is explicitly not meant to be misused to document an intentional lack of initialization<sup>[3]</sup>, so the profile uses something distinct.

The central design decision is whether "uninitialized" should be a property of the type system. I think not, for four reasons:

- Tracing uninitialized memory through a program is often too expensive at compile time or run time, because it is flow-dependent and crosses function boundaries.
- Tracing it is often impossible to do statically.
- Some invoked code is C or C-style C++, such as an operating system.
- Not all code can be compiled with a profile, and will not be for years.

Keeping the information out of the type system is what preserves identical meaning with and without the profile. To enforce "every object is initialized before use," the profile therefore provides two markers: an "uninitialized" marker on a definition, which suppresses the error the profile would otherwise raise for a definition without immediate initialization; and a "refers to uninitialized" marker, which records that a pointer or reference to an uninitialized object has been created. Together they let the profile enforce initialization-before-use over a restricted, well-defined subset of C++ (Section 1). The subsections that follow give the notation (4.1), the rules for each marker (4.2, 4.3), the library support that keeps them out of application code (4.4, 4.7), and the harder cases: templates (4.5), lifetimes (4.6), overloading (4.8), and mixed data (4.9).

### 4.1. Notation: an attribute, not a keyword or a type change

There are three plausible ways to spell "leave this uninitialized": a pseudo-value (`X x = uninitialized;`), an attribute (`X x [[uninit]];`), or nothing at all, letting the compiler note that an object is uninitialized (`X x;`). The choice is syntactic, and the attribute is preferred. An attribute states plainly that the mark is information for optional analysis, so the profile can be verified on a modern compiler and the same code compiled on an older compiler that ignores the attribute. It also avoids introducing a keyword that I hope would be rarely used.

The attribute form leaves one case uncovered, an uninitialized slot inside an initializer list. The first line below is the attribute form and does not parse; the second is the pseudo-value form that would, showing the single case the attribute cannot express:

```cpp
tuple<int, int, int> t = {1, [[uninit]], 3}; // syntax error
tuple<int, int, int> t = {1, uninitialized, 3}; // "magic" non-value
```

I suggest this case is rare enough to leave without special coverage; where it arises, I would suppress the profile rather than add new syntax. Suppressing the whole profile for the common cases would be verbose, open to misuse, and likely bypassed with workarounds, so it is not the primary tool. A default value is often acceptable instead, as in this ordinary initialization:

```cpp
int uninitialized = 0B1010101010101010;
```

The remaining option, letting the compiler silently remember that an object is uninitialized, shares the advantages of `[[uninit]]` but has two drawbacks: being implicit, it cannot record whether the lack of initialization was intended, and it does not extend to carrying the initialized-or-not status across function boundaries (Section 5.2).

Two spelling questions have the same answer, and I think not to both. I do not spell it `[[profiles::uninit]]`, because the information it conveys is useful independently of the profile: it helps code review and will likely be used by other profiles and by people who do not enable this one. I do not spell it `[[uninitialized]]` either, because the longer form is verbose and invites an English-versus-American spelling problem; I tried the longer form for a while and found it a source of misspelling. Attributes are assumed to be optionally enforced, and for a profile "optionally enforced" means "enforced when the profile is opted into."

### 4.2. `[[uninit]]` rejects use before initialization

The `[[uninit]]` attribute has six rules:

- An uninitialized object not marked `[[uninit]]` is an error.
- An initialized object marked `[[uninit]]` is an error.
- An object marked `[[uninit]]` is left uninitialized and must be initialized before use, for example with `construct_at()` (Section 6); after initialization it is no longer `[[uninit]]`.
- The function template `now_init()` provides a pointer to something initialized from a pointer to something previously uninitialized (Section 4.4).
- A pointer or reference to an `[[uninit]]` object can be passed only to a `[[ref_to_uninit]]` (Section 4.3).
- A pointer or reference to something initialized cannot be passed to a `[[ref_to_uninit]]` (Section 4.3).

For example, where the `// error` lines are the cases the profile is meant to reject and the `// OK` lines those it accepts:

```cpp
int glob;            // OK: default initialized
int glob2 [[uninit]]; // error: initialized and [[uninit]]

void f()
{
    int loc;             // error: no initialization
    int loc2 [[uninit]]; // OK
    int loc3 = 3;        // OK
    int loc4 [[uninit]] = 4; // error: initialized and [[uninit]]
    vector<int> v1;      // OK: default initialized
    vector<int> v2 [[uninit]]; // error: initialized and [[uninit]]
    string s1;           // OK: default initialized
    string s2 [[uninit]]; // error: initialized and [[uninit]]

    int loc5 [[uninit]];
    int x = loc5;        // error: reads an uninitialized object

    int loc6 [[uninit]];
    loc6 = 7;            // initializing
    int y = loc6;        // OK
}
```

Using `[[uninit]]` to suppress default initialization is possible in principle but would make code more error-prone rather than less, so it is not done.

### 4.3. `[[ref_to_uninit]]` tracks uninitialized memory through pointers

A reference cannot be uninitialized, and the profile requires the same of pointers; where feasible, a pointer that would be left uninitialized is initialized to `nullptr` instead. What still needs expressing is that the memory a pointer refers to is uninitialized. `[[uninit]]` cannot say this, because it is not the pointer or reference that is uninitialized but the memory it points to.

Since about 1983, C++ has had `void*` for a pointer to memory of unknown type. For compatibility a `void*` is assumed by default to point to something initialized. Consider two pointers where `p1` points to an initialized object and `p2` to an uninitialized one, yet both have the same type:

```cpp
int x1 = 7;
void* p1 = &x1; // OK
int x2 [[uninit]];
void* p2 = &x2;
```

Every `void*` must be cast before use, but what `p1` and `p2` point to differs fundamentally with respect to initialization, and the type system does not record the difference. To record it, the profile introduces `[[ref_to_uninit]]`, "reference to uninitialized." Demanding instead that all elements of an array be uniformly initialized or uninitialized would be simpler but would leave major areas of C++ unserved. Consider the four combinations, where the `// error` lines are mismatches between the pointer's mark and the target's state:

```cpp
int x1 = 7;
void* p1 = &x1; // OK
void* p2 [[ref_to_uninit]] = &x1; // error: target is initialized
int x2 [[uninit]];
void* p3 = &x2; // error: target is uninitialized
void* p4 [[ref_to_uninit]] = &x2; // OK
```

To use what `p1` points to, a cast is still required (relying, as ever, on casting correctly or on the casting profile); dereferencing `p4` yields an `[[uninit]]` object. Allocators fit this model directly:

- `malloc()` returns a `void*` to uninitialized memory.
- `calloc()` returns a `void*` to zero-initialized memory.

The natural annotation is shown below; it is proposed library declarations illustrating where `[[ref_to_uninit]]` would belong:

```cpp
void* malloc [[ref_to_uninit]] ( std::size_t size );
void* calloc( std::size_t num, std::size_t size );
```

Whether adding `[[ref_to_uninit]]` to every allocator is manageable needs exploration, because system headers may not be modified for the benefit of C++ profiles. Where they are not, functions like `malloc()` are either called only under suppression or made known to the analyzer, which implementations that already diagnose erroneous behavior will do. The detailed rules for `[[ref_to_uninit]]` on a `void*` are:

- By default a `void*` points to memory initialized to some type.
- A `void*` marked `[[ref_to_uninit]]` must point to something `[[uninit]]`.
- On assignment between two `void*`s, neither or both must be `[[ref_to_uninit]]`.
- Casting a `[[ref_to_uninit]]` `void*` to another pointer type yields a `[[ref_to_uninit]]` result.
- After the memory a `[[ref_to_uninit]]` refers to is initialized, for example by `construct_at()`, it is no longer `[[uninit]]`.
- Dereferencing a `[[ref_to_uninit]]` yields an `[[uninit]]`.

The name `[[ref_to_uninit]]` will strike some as ugly. Because it belongs in inherently complex foundational code, where it and its direct users make it relatively common, I consider it acceptable and conventional; a longer, more descriptive name would be verbose. `[[points_or_refers_to_uninitialized]]` is fully explicit but sets a record in verbosity, and `[[ref_to_uninitialized]]` is a muddled mix of abbreviation and explicitness. Consider the attribute on function parameters and local pointers, where the `// error` lines mark passing an initialized pointer where an uninitialized target is required, or leaving a pointer uninitialized:

```cpp
void f1(int* p [[ref_to_uninit]]); // *p must be uninitialized
void f2(int& r [[ref_to_uninit]]); // r must refer to uninitialized

int* p0; // static, thus initialized

void g(int x)
{
    int* p1;             // error: uninitialized pointer
    int* p2 = nullptr;   // OK
    int* p3 [[ref_to_uninit]] = &x; // x must be uninitialized
    f1(p1);              // error: p1 is initialized
    f1(p2);              // OK: p2 is initialized
    int* p4 [[uninit]];  // p4 is uninitialized
}
```

Making `[[uninit]]` part of the type system, like `const`, would break a great deal of old code and C libraries unless it were opt-in; when enforced, `[[uninit]]` is viral. I hope it becomes universal, but that will take time, and a gradual introduction is essential.

### 4.4. `construct_at()` and `now_init()` keep the attributes out of application code

Containers like `std::vector` and slab or pool allocators store objects of a known type in "slots" of uninitialized memory rather than through `void*`. To avoid replicating the messy code that manages such slots, the standard library should supply the support directly:

- The iterator arguments of the `uninitialized_*()` family are annotated `[[ref_to_uninit]]`.
- `construct_at()` takes a `[[ref_to_uninit]]` and returns a pointer to an initialized object.
- The object passed to `destroy_at()` becomes uninitialized.
- A `now_init()` takes a `[[ref_to_uninit]]` pointer and returns a pointer to the now-initialized objects in the formerly uninitialized memory.

`now_init()` is the deliberate hole in the enforcement, and its definition is trivial. Consider the function itself:

```cpp
template<class T> T* now_init(T* p [[ref_to_uninit]]) { return p; }
```

This function cannot be compiled with the profile on, because it is a cast in disguise: it deliberately and explicitly breaks the enforcement. Compiled without the profile it is a no-op. If it could be written with the profile on, the profile would be flawed. Its value is that it lets a programmer avoid suppressing the whole profile and marks an easy-to-spot point for code review. Something like it is needed in any code that manages a mixture of initialized objects and uninitialized slots, such as a vector implementation (Section 4.6).

### 4.5. Template arguments are assumed initialized by default

By default, objects of a template argument type are assumed initialized, which avoids chaos; uninitialized objects of such types are nonetheless possible and sometimes useful. `unique_ptr<T>` and `span<T>` are the important examples, and like built-in pointers they need the profile to distinguish initialized from uninitialized objects they point to. For a class that can return a pointer or reference to what it holds, there are three choices:

- It cannot return a reference to an uninitialized object; this is the default.
- It always returns a reference to an uninitialized object, and the analyzer knows that.
- It returns initialized and uninitialized objects uniformly.

The third choice cannot be given a purely static guarantee. Consider why; `get()` is an error because a function has a single return type:

```cpp
class Wrong {
    int x = 7;
    int y [[uninit]];
public:
    int& get(bool b) { return (b) ? x : y; } // error
};
```

Even if `[[uninit]]` were part of the type, `get()` would still be an error, because it would attempt to return two different things through one return type; separating the alternatives requires a run-time test. Variants of this recur across use cases, so the returned object must carry its status: the default is that a returned object is initialized, and the uninitialized case is marked. The following illustrates the marked form; the `// error` lines are the misuses the profile catches:

```cpp
class Slot {
    int x = 0b101010101010;
    int y [[uninit]];
public:
    Slot(int xx) : x{ xx } {}
    Slot() {}
    int& get_init() { return x; }
    int& get_uninit [[ref_to_uninit]] () { return y; }
};

Slot ii = 7;
Slot uu; // leaves uu.y uninitialized
int i1 = ii.get_init();   // OK
int i2 = ii.get_uninit(); // error: reads from an [[uninit]]
int u1 [[uninit]] = uu.get_init();   // error: initializes an [[uninit]]
int u2 [[uninit]] = uu.get_uninit(); // OK
```

Until more experience is gathered, the simple solution is to templatize `Slot`, which is compile-time enforced like the rest of the profile. Its weakness is that a slot cannot be asked whether it is initialized; that has to be tracked externally, as a vector built on an array of slots would track the boundary between its initialized elements and its empty slots. Classes, including class templates such as `std::vector`, that fully control their own allocation and consistently return initialized objects are no separate problem (Section 5.4); they can often manipulate their slots directly through `[[uninit]]` and `[[ref_to_uninit]]` without a `Slot` type at all. A "dynamic slot" that used a `bool` to record which alternative is present would be easy, but it would add run-time cost, memory cost, and run-time error handling, so it is not the default.

### 4.6. Lifetimes: destruction leaves memory uninitialized

Handling uninitialized data means handling destruction as well as initialization, because destruction is the reverse of construction. Under the profile, the result of destruction is equivalent to an `[[uninit]]` mark. Initializing an object twice, for example with `construct_at()`, is an error, and so is destroying it twice, for example with `destroy_at()`. Reading an uninitialized object, except a `std::byte`, is erroneous behavior, and where the profile is enforced it is prevented at compile time. Writing an uninitialized object is an initialization: for a built-in type a plain write or `construct_at()`, for a class object `construct_at()`. Because initializing an object with a constructor twice is an error, `construct_at()` requires an uninitialized argument. Enforcing all of this needs only the simple flow analysis of Section 1.2.

Some unstructured techniques are thereby relegated to suppressed code. For example, where the `// error` comments mark the cases static analysis cannot clear because the index is not a compile-time constant:

```cpp
void init(span<X> s [[ref_to_uninit]], int i1, int i2)
{
    construct_at(&s[2], 10); // only s[2] is initialized
    X y = s[i2];             // error: reading uninitialized? i2 could be != 2
    construct_at(&s[i1], 10); // error: double initialization? i1 could be == 2
}
```

Realistic versions involve loops and conditions. There are two options: allow only code simple enough for static analysis to guarantee initialization and require suppression for more complex control flow, or fall back on erroneous behavior for the complex cases. A purely compile-time guarantee requires the first. Destruction has the same shape and the same constraint, shown here with a double-destruction that cannot be cleared statically:

```cpp
void init(span<X> s, int i1, int i2)
{
    destroy_at(&s[2]);
    destroy_at(&s[i1]); // error: double destruction? i1 could be == 2
}
```

What counts as "sufficiently simple" is left informal here and needs a precise definition in the wording. One coping technique keeps a separate record of which slots are live; it handles every case I can think of but provides no static guarantee. Consider that technique:

```cpp
vector<Slot> v; // Slot, Section 4.5
enum class Live { uninit, init };
vector<Live> live_slot;
// ...
if (live_slot[x] == Live::init) {
    int xx = v[x];
    // ...
}
```

Run-time testing, adding something like `Live` to a `Slot` and checking it on every access, would be simple but too expensive for latency-sensitive allocators and containers. The alternative is the vector pattern: keep the first part of a buffer initialized and the second part uninitialized, with a range check and operations that move the barrier between them. Here is a much-simplified sketch of the two cooperating templates. `Vector_memory` owns only uninitialized memory:

```cpp
template<class T>
struct Vector_memory {
    T* elem [[ref_to_uninit]];
    int no_of_elem;
    int no_of_slots;

    Vector_memory(int ne, int ns)
        : elem{(T*)malloc((ne+ns)*sizeof(T))},
          no_of_elem{ne},
          no_of_slots{ns}
    {
    }
    ~Vector_memory() { free(elem); }
};
```

`Vector` builds a usable container on top of it. Its subscript operator is the one place the profile cannot verify statically, so it calls `now_init()`, which marks that point for review:

```cpp
template<class T>
class Vector {
    Vector_memory<T> mem;
public:
    Vector(int ne, int ns, const T& val = T{})
        : mem{ne, ns}
    {
        uninitialized_fill(mem.elem, mem.elem+ne, val);
    }

    T& operator[](int i) {
        if (0 <= i && i < mem.no_of_elem)
            return *now_init(mem.elem[i]); // this slot has been initialized
        throw Bad_index{};
    }
    ~Vector() { destroy(mem.elem, mem.elem+mem.no_of_elem); }

    void reserve(int newsz)
    {
        if (newsz <= mem.no_of_elem+mem.no_of_slots) return;
        void* p = malloc(newsz*sizeof(T));
        uninitialized_move(mem.elem, mem.elem+mem.no_of_elem, (T*)p);
        mem.no_of_slots = newsz - mem.no_of_elem;
        void* pp = mem.elem;
        mem.elem = (T*)p;
        free(pp);
    }

    void push_back(const T& x) {
        if (mem.no_of_elem == mem.no_of_slots)
            reserve(mem.no_of_elem == 0 ? 8 : 2*mem.no_of_slots);
        construct_at(mem.elem+mem.no_of_elem, x);
        ++mem.no_of_elem;
        --mem.no_of_slots;
    }
};
```

The code is conventional for its kind, and it is free of initialization errors except at the `now_init()` call in the subscript operator, which is reached only from within an initialized range; verifying that statically would be expensive and, in similar cases, impossible. `now_init()` saves the user from suppressing the profile and marks a point for review. Other kinds of error remain possible (error handling is essentially missing here), but those belong to other profiles and, for logic errors, to the programmer.

A second important use is recycling a buffer, where messages are read and written once and no allocation happens while a stream of messages passes. This pattern appears, in many variations, throughout high-performance and low-latency code. Again, this is just a sketch of the initialization aspect:

```cpp
std::byte ibuf [[uninit]] [imax]; // or allocated in dynamic storage
class Message { /* read typed members from a byte-buffer representation */ };
int read(std::byte* p [[ref_to_uninit]], int max); // read at most max bytes into *p,
                                                    // return the number of bytes read

while (live) { // read into the buffer and process the messages stored as bytes
    int n = read(ibuf, imax);
    span<Message> messages {
        *(Message*)now_init(ibuf), // this buffer now holds Messages
        n / sizeof(Message)
    };
    for (Message& m : messages) {
        // ... process a message from the buffer ...
    }
    destroy(messages);
}
```

Assigning to an uninitialized `std::byte`, as `read` must, is allowed. Beyond initialization, the code needs one explicit type conversion, from bytes to messages in `now_init()`, and must avoid range errors; this is the low-level code where suppression of guarantees is tempting and common even in "safe" languages. In performance-critical cases `Message` will usually have no destructor, since it is an access mechanism over a byte buffer, so `destroy(messages)` is a no-op. The profile confines the one unverifiable step to a single, reviewable `now_init()` call rather than to a suppressed region.

### 4.7. Vocabulary templates need the marks only at their boundaries

The vocabulary types that matter here - `unique_ptr`, `span`, `vector`, and their relatives - are templates. Those that fully manage their own memory, like `std::vector`, are no separate problem. Others can be given uninitialized objects, and the profile must catch the misuse without breaking the large body of code that is not broken. Consider:

```cpp
unique_ptr<T> p (new T);
```

Whether the `T` is initialized matters, and users should care; the profile must catch misuses. Code like this must work as written when `new T` yields an initialized object, and the profile must give the user a way to handle the uninitialized cases. `make_unique()` has the same initialized-versus-uninitialized problem. Even `vector` is vulnerable; the following is proposed code where the element is uninitialized:

```cpp
int x [[uninit]];
vector v(10, x); // error: filling from an uninitialized value
```

Some templates are meant to take uninitialized memory. By default a template function's argument must be initialized, or marked `[[uninit]]`; a pointer argument must point to an initialized object, or be marked `[[ref_to_uninit]]`. For example, with an annotated `uninitialized_fill`:

```cpp
template<forward_range auto R, auto V>
    requires constructible_from<*iterator_v(R), V>
void uninitialized_fill(R r [[ref_to_uninit]], V val);

int a1[] = {1, 2, 3};
int a2[3] [[uninit]];
uninitialized_fill(a1, 10); // error: a1 is already initialized
uninitialized_fill(a2, 10); // OK
```

How does the analyzer know `R` refers to something, and to what? It does not have to. When `uninitialized_fill()` is compiled, writes through the things that refer to elements are done with `construct_at()` (Section 5.2) or under suppression. Class and function templates that take only initialized or only uninitialized arguments are not a problem: the wrong alternative is caught at the call site, and the rules of the last decades already work, so the profile only adds a simple check. Consider how the analyzer learns that `uninitialized_fill()` initializes its argument:

```cpp
int a2 [[uninit]] [3];
int x = a2[0];          // error: a2 is uninitialized
uninitialized_fill(a2, 10);
int y = a2[0];          // OK: a2 is now initialized
```

Requiring the compiler to know the semantics of specific standard-library functions such as `uninitialized_fill()` is not general; an attribute `[[now_init]]` (Section 4.4) is the general form. Better still, the test can be placed in a concept. Placing a check of `[[uninit]]` in `constructible_from<>`, where it belongs, returns the user to conventional code:

```cpp
template<forward_range auto R, auto V>
    requires constructible_from<*iterator_v(R), V>
void uninitialized_fill(R r, V val);

int a1[] = {1, 2, 3};
int a2 [[uninit]] [3];
uninitialized_fill(a1, 10); // error: a1 is already initialized
uninitialized_fill(a2, 10); // OK
```

This relies on the static analyzer seeing attributes such as `[[uninit]]` and `[[ref_to_uninit]]` when it checks a template argument, which requires-clauses allow. It does not make `[[uninit]]` part of the type system the way `const` is; doing that would let code under the profile mean something different from the same code compiled without it.

### 4.8. Concept-based overloading turns on an unresolved type-deduction question

This is the one part of the design without an implementation, and it turns on a type-deduction problem the implementers are still investigating. Defining a function template callable only with a `[[ref_to_uninit]]` argument is easy. For example, where `f(&ii)` is rejected because `ii` is initialized:

```cpp
template<class T> concept Pointer_to_uninit
    = Pointer<T> && requires(T* p) { T* q [[ref_to_uninit]] = p; };

template<Pointer_to_uninit T> int f(T* p) { /* ... */ }

int ii = 7;
int ui [[uninit]];
int y = f(&ii); // error: ii is initialized
int x = f(&ui); // OK
```

Overloading a pair that chooses between a `[[ref_to_uninit]]` and an ordinary pointer is also fine:

```cpp
template<Pointer_to_uninit T> int f(T* p) { /* ... */ }
template<Pointer T> int f(T* p) { /* ... */ }

int ii = 7;
int ui [[uninit]];
int y = f(&ii); // OK: calls the f() for initialized
int x = f(&ui); // OK: calls the f() for uninitialized
```

This compiles under the profile and fails to compile without it, which is acceptable, because the code relies on a function that is part of the profile's support; without the profile the two `f()` definitions are a double definition. The problem arises only when a pair is written to resolve to *different* alternatives with and without the profile. The following defines two concepts that distinguish initialized from uninitialized pointers of the same type:

```cpp
template<class T> concept Pointer_to_uninit
    = Pointer<T> && requires(T* p) { T* q [[ref_to_uninit]] = p; };
template<class T> concept Pointer_to_init
    = Pointer<T> && requires(T* p) { *p = 7; };
```

Overloading on those, and assuming for the moment it works, gives different resolutions. With the profile:

```cpp
template<Pointer_to_uninit T> int g(T* p) { /* ... */ }
template<Pointer_to_init T> int g(T* p) { /* ... */ }

int y = g(&ii); // OK: calls the g() for initialized
int x = g(&ui); // OK: calls the g() for uninitialized
```

Without the profile, the `g()` that assumes initialized has the stricter requirement, so both calls resolve to it:

```cpp
int y = g(&ii); // OK: calls the g() for initialized
int x = g(&ui); // OK: calls the g() for initialized
```

That violates the rule that code means the same thing with and without the profile. The example is contrived, but it must be prevented. There are three options:

- Accept such examples, allowing a violation of a fundamental principle. This is not acceptable.
- Catch them at the point of definition, which implies a check for each template function.
- Catch them at the point of use, which implies a check at each call of a template function.

The checks run only when the profile is requested. The point-of-use option is the most flexible but would draw the profile into every template call, which is undesirable. Because requires-clauses can be arbitrarily complex, the check is made tractable with the subset-of-superset technique:

- Unless a template declaration is an overload, nothing need be done.
- If it is an overload, look further only when `[[uninit]]` or `[[ref_to_uninit]]` appears in the constraints.
- Accept the overload only if the overloads involving `[[uninit]]` or `[[ref_to_uninit]]` are identical once those markers are ignored in the constraints.

That rule permits simple initialized-versus-uninitialized overloads at minimal cost; anyone wanting more general rules must supply a use case and a plausible acceptance algorithm. The open problem sits underneath this. Consider the call whose resolution depends on how the requires-clause reads a type:

```cpp
template<class T> concept Pointer_to_uninit
    = Pointer<T> && requires(T* p) { T* q [[ref_to_uninit]] = p; };

int ui [[uninit]];
int x = f(&ui); // OK: calls the f() for uninitialized
```

The question is how the information that `ui` is uninitialized reaches the requires-clause. Is the `T` in `T* q` the deduced type or the type of `ui` as presented as an argument? If it is the deduced type, the `[[ref_to_uninit]]` of `ui` has been stripped and the requires-clause never fails; if it is the argument's type, all is well. The implementers are looking into this problem.

If deduction strips the mark, an alternative mechanism is needed to carry the initialized-or-not information into a template. This is possible, though the concept solution is more elegant. Using `[[uninit]]` and `[[ref_to_uninit]]`, right and wrong uses can be distinguished, which is all a static analyzer needs. It is possible to define a `span<int>` that takes initialized ints and not uninitialized ones, and an `Uninitialized_span<int>` that takes uninitialized ints and not initialized ones. What cannot be defined without help is a single `span<int>` that distinguishes initialized from uninitialized ints, which is what simple, elegant use would want.

Failing the overloading, the initialized-versus-uninitialized distinction can be turned into a check that guides overloading, a kind of tag dispatch, by adding one or two intrinsics for what the compiler already knows, `is_init()` and `is_ref_to_init()`. These let the code work around the inability to encode attributes in types and smuggle them through type deduction, a difficulty I underestimated. For example, using such an intrinsic to select an implementation at run time from information the compiler holds:

```cpp
int x [[uninit]];
int* p [[ref_to_uninit]] = &x;
bool init = is_ref_to_init(p);

template<class T> class span {
    span(T* p, bool init) {
        if (init)
            init_span_implementation(p);
        else
            uninit_span_implementation(p);
    }
    // ...
};
```

Concept-based overloading is the profile's one open design question, and the demonstration keeps it stated rather than resolved.

### 4.9. Mixed initialized and uninitialized data requires suppression

Passing along a value that is sometimes initialized and sometimes not is the hard residual case, and static analysis cannot resolve it perfectly. For example:

```cpp
int* f(int x)
{
    return (0 < x) ? (int*)malloc(x*sizeof(int)) : new int[x];
}
```

A guarantee that is incomplete, especially one that holds often, breeds overconfidence and becomes a source of subtle bugs, which is exactly what the profile exists to prevent. Mixing of this kind therefore requires suppressing the profile. Adding the initialized-versus-uninitialized distinction to the type system would add another alternative to every read and write for the compiler to weigh, slowing compilation for all code, not only profiled code. It would also mean code could no longer be validated under the profile on a modern compiler and then relied on when compiled with an older one. The available tool is concept-based overloading of templates (Section 4.8). The profile draws the line at incomplete static guarantees: where it cannot prove initialization, it requires suppression rather than a partial promise.

## 5. Classes carry the initialization guarantee to their members

The profile's member-initialization rule is what lets a class carry the initialization guarantee to its users. A class with a constructor must have every member not marked `[[uninit]]` initialized by that constructor, and any member marked `[[uninit]]` must be initialized before it is exposed to users of the class. A class without a constructor must have every member initialized at the point of definition, unless the member is marked `[[uninit]]` there. Together these let a profile ensure type safety as it relates to initialization.

### 5.1. Constructors must initialize every unmarked member

A constructor that initializes every member is usually both the easiest form to use and the easiest for the compiler to validate. When a constructor is compiled with the profile requested, it is checked to initialize every member except those marked `[[uninit]]` or `[[ref_to_uninit]]`. If every translation unit is compiled with the profile, all is well; if not, as will be common for years, this check is still the best available. Reconciling different profiles across translation units and modules is the job of the profiles framework<sup>[2]</sup>, not of an individual profile. Complex code in a constructor can defeat static analysis and is rejected (Section 1.2); avoiding that is not hard.

### 5.2. Member initializers are the simplest path

The simplest way to initialize a member is directly, in a member initializer or a member initializer list. For example, in the two direct forms the profile expects:

```cpp
class X {
    int m1 = 7;
    int m2;
public:
    X(int x) : m2{x} {}
    // ...
};
```

A member initialized in the constructor body rather than in one of those two ways must be declared `[[uninit]]`. Static analysis (Section 1.2) can often show it is given a value before use. For example, where `p` is set in the body:

```cpp
class X {
    int* p [[uninit]];
    int x;
public:
    X(int v) : x{v}
    {
        if (v < 0 or sys_max <= v) error(v);
        p = new int(v);
    }
    // ...
};
```

Whether `p` is still `[[uninit]]` after construction is the hard question: answering "no" requires the compiler to examine every flow path in every constructor. An alternative is to initialize `p` elsewhere and use `now_init()` or suppression to expose the unverified `p` or `*p`. Often a "pseudo initialization" is the better choice, shown here setting `p` to `nullptr` first:

```cpp
class X {
    int* p = nullptr;
    int x;
public:
    X(int v) : x{v}
    {
        if (v < 0 or sys_max <= v) error(v);
        p = new int(v);
    }
};
```

The optimizer can usually eliminate the redundant initialization. Better still is a member of a type that performs its own checks rather than a primitive; the following uses `unique_ptr`:

```cpp
class X {
    unique_ptr<int> p;
public:
    X(int x) : p{make_unique<int>(x)} {}
};
```

Default construction works as ever. For example, with one member initialized in the list and another assigned in the body:

```cpp
class X {
    string s1;
    string s2;
public:
    X(const char* ss1, const char* ss2) : s1{ss1} { s2 = ss2; }
};
```

Initializing `s1` in the member initializer is preferred to assigning `s2` in the body, being more readable and more efficient unless the optimizer eliminates the default initialization of `s2`.

### 5.3. Classes that expose uninitialized memory

Some classes, such as memory pools, must expose uninitialized members to their users. Consider a class whose three flagged members would expose uninitialized objects with no indication:

```cpp
struct X {
    Y m1;         // maybe error
    Y* m2;        // error, see constructor
    int arr[10];  // error, see constructor
    X(int x) : m2{ (Y*)malloc(x) } {}
    // ...
};
```

The profile flags `m1` (if `Y` has no default constructor), `m2` (`malloc()` returns a pointer to uninitialized memory), and `arr` (it has uninitialized members), because each would expose users of `X` to uninitialized objects silently. The fix is to mark them:

```cpp
struct X {
    Y m1;
    Y* m2 [[ref_to_uninit]];
    int arr [[uninit]] [10];
    X(int x) : m1{x}, m2{ (Y*)malloc(x) } {}
    // ...
};
```

For an uninitialized class object, initialization uses `construct_at()` rather than plain assignment. The marks put the exposure of uninitialized memory into the interface, where a user and the analyzer can both see it.

### 5.4. Constructorless classes must initialize all members at definition

Objects of a class without a constructor must have all members initialized in their definition. For example, where the `// error` lines are the definitions that leave a member uninitialized:

```cpp
struct S { int x; string s; };

void f()
{
    S s1 = {1, "foo"}; // OK
    S s2 = {1};        // OK: s2.s defaults to ""
    S s3;              // error: s3.x uninitialized
    s3.x = 1;
    s3.s = "foo";
    S s4 [[uninit]];   // error: s4.s is default constructed
    s4.x = 1;
    s4.s = "foo";
}
```

`s4.s` is a `string`, and `string` has a default constructor, so `S s4 [[uninit]];` declares `s4.s` as both initialized and uninitialized, which is an error even before delayed initialization is considered. Delayed initialization of the whole object is accepted with `construct_at()`:

```cpp
S s5 [[uninit]];
construct_at(&s5, {1, "bar"});
s5.s = "bar"; // OK: s5 is now initialized
```

This works, but now the programmer and the compiler must track whether the initialization has happened (Section 1.3). For a single object that is complicating, unnecessary, and a source of error. Immediate initialization is simpler and clearer:

```cpp
S y = {1, "foobar"};
```

The rule of thumb is to not introduce a variable until there is an initializer for it.

### 5.5. Arrays are initialized at definition or marked `[[uninit]]`

Arrays, like structs, are either initialized in the definition or marked `[[uninit]]`. For example:

```cpp
int arr1[] = {1, 2, 3, 4};
int arr2[10] [[uninit]];
```

Large arrays are often, and reasonably, initialized later by algorithms that can be complex, as in the input-buffer example of Section 4.6. The `uninitialized_*()` family of range algorithms is close to ideal for this and can be verified, though it is tricky to use. For example, where the default constructor forces one error or the other:

```cpp
X arr3[20];                          // error unless X has a default constructor
ranges::uninitialized_fill(arr3, xval); // error if X has a default constructor
```

Marking the array `[[uninit]]` resolves it when the programmer knows `X`:

```cpp
X arr4[20] [[uninit]];               // error if X has a default constructor
ranges::uninitialized_fill(arr4, xval);
```

### 5.6. Unions cannot be delayed-initialized

The simplest treatment of unions aligns with classes that have public members. Consider it, where the `// OK? (no)` lines are the delayed initializations the profile rejects:

```cpp
union S {
    int x;
    string s;
};

S x1 = 7;
S x2;            // error
S x3 [[uninit]];
x3.x = 9;        // OK? (no)
S x4 [[uninit]];
x4.s = "foo";    // OK? (no)
```

The handling of `x2` and `x3` is straightforward. Delayed initialization of `x4` is rejected, because it would be an erroneous assignment, and to avoid treating union members of different types very differently, `x3` is banned as well. The union rule keeps delayed initialization out of a place where member types differ and the assignment cannot be verified.

## 6. Four operations count as initialization

To specify and implement the rules that clear `[[uninit]]` and `[[ref_to_uninit]]` once an object is initialized, the profile must define what initialization is. The current proposal is:

- For class objects, use `construct_at()`.
- For built-in types, use `construct_at()` or ordinary assignment.
- For ranges, use the `uninitialized_*()` family.
- For more complex initialization, use functions marked `[[now_init]]` (Section 6.2).

### 6.1. Loop-based initialization is too complex for static proof

Even simple-looking extensions are trickier than they appear, and the recurring obstacle is deciding what a completed initialization is. Consider whether range-for loops should be accepted, initializing an array in a loop:

```cpp
int aar6 [[uninit]] [10];
for (int& x : aar6) x = 0b1010101010101010;
```

Accepting this requires the profile to recognize the loop as an initialization, to confirm that the loop body really initializes, and that the loop really completes. Partial initialization is harder still. For example, initializing even and odd elements in separate loops:

```cpp
int aar7 [[uninit]] [10];
// ... no use of aar7 here ...
for (int x = 0; x < 10; x += 2) aar7[x] = 2; // initialize even
// ... restricted use of aar7 here ...
for (int x = 1; x < 10; x += 2) aar7[x] = 1; // initialize odd
```

Even the most reasonable generalizations leave serious verification problems. Consider an example whose completeness depends on run-time input:

```cpp
X aar8 [[uninit]] [10];
// ... no use of aar8 here ...
int count = 0;
while (cin && count < 10)
    aar8[count++] << cin;
// ... we can use aar8 here ...
```

Verifying this would mean tracking `count` and handling the case where `cin` supplied only five values; real examples are far more complex. Delayed initialization leads straight into these complexities. The alternative is to suppress the profile for the initialization step and use a recognized method such as `uninitialized_fill()`, which delays initialization without suppressing the profile:

```cpp
X arr9 [[uninit]] [20];
// ... any recognized initialization technique used here ...
// ... use arr9 here ...
```

Initialization too complex for the analyzer to recognize is best avoided but can be handled by the unverified `now_init()`. For example:

```cpp
X arr10 [[uninit]] [20];
// ... any initialization technique used here ...
span<X> s{*now_init(arr10)};
// ... use s, not arr10, here ...
```

This technique is not verifiable, but it is far more manageable than suppressing the profile, and it marks an obvious target for code review and deeper analysis.

### 6.2. `[[must_init]]` marks functions that must leave their argument initialized

A pointer to uninitialized memory is often passed to a function that initializes it. For example:

```cpp
T* initialize1(T* p [[ref_to_uninit]]);
void initialize2(T* p [[ref_to_uninit]]);
```

As the proposal stands, only `now_init()` can tell the analyzer that after a call to `initialize1()` or `initialize2()`, `*p` is initialized. Such functions are common, as are functions that only forward an uninitialized pointer to another function that does the work, so a verifiable way to express this is needed. Two capabilities are required: a way to say a pointer refers to uninitialized memory, which the profile has, and a way to say a pointer must refer to initialized objects after the call, which it does not. The `[[must_init]]` attribute supplies the second. Consider it, contrasted with `[[ref_to_uninit]]`:

```cpp
T* initialize1(T* p [[ref_to_uninit]]);
void initialize2(T* q [[must_init]]);
```

Here `q`, like `p`, must point to something uninitialized on entry, but after the call what `q` points to must have been initialized. Together `[[ref_to_uninit]]` and `[[must_init]]` cover the use cases the author has found. `[[must_init]]` implies `[[ref_to_uninit]]`, which avoids verbosity.

## 7. Code that does not obey the initialization profile

The C++ type system does not distinguish initialized objects from uninitialized memory; that is the profile's job. C code, trusted old-style C++, and code behind system headers does not obey the profile and probably never will. Range errors can break this profile, as they can break others, but preventing them belongs to the invalidation and range profiles; they are assumed prevented here and are not treated.

The major problem is system headers. Annotating C-style code with `[[uninit]]` and `[[ref_to_uninit]]` is not difficult, but system headers and much other foundational C-style code are controlled by organizations that, at least in the short term, will not accept C++ attributes. This is a genuine and unresolved difficulty, and the profile's opt-in design is what keeps it from blocking adoption elsewhere.

## 8. Specify the guarantee, not a list of affected cases

A profile should be specified by the guarantee it offers rather than by a long list of the places in the language it affects. Such lists are necessary in addition to the guarantee, but on their own they tend to be incomprehensible to non-experts and incomplete: if a list enumerates 27 cases, there is no way to be sure the right number is not 26 or 28. Every such list must be checked against the general guarantee it exists to deliver. Having both a guarantee and a detailed specification also gives a way to consistency-check the specification against the guarantee. A common style for specifying standard profiles is needed.

## 9. Early draft wording

This section collects the draft normative wording, which is incomplete by design.

### 9.1. Status

This is an early draft and needs input and review. The `???` markers below are the author's own placeholders for wording not yet drafted; they are preserved here rather than filled, because completing them is a drafting task for the committee, not a presentation change.

### 9.2. Guarantees

(???)

- Every object is initialized at the point of definition or marked `[[uninit]]`.
- Every object marked `[[uninit]]` at its point of definition is initialized by `construct_at()` or an equivalent before use.
- Only local, and only very simple, static analysis is used. For run-time alternatives, all execution paths are considered, and acceptance requires that all alternatives provide the desired result.
- Random access to uninitialized arrays is banned; range algorithms are used instead (Section 5.4). When random access is needed, the profile is suppressed.

### 9.3. Attributes

The profile defines three attributes:

- `[[uninit]]` marks an object as uninitialized.
- `[[ref_to_uninit]]` marks a pointer, reference, or smart pointer as pointing to zero or more uninitialized objects.
- `[[must_init]]` implies `[[ref_to_uninit]]` and marks that the pointer, reference, or smart pointer is to be initialized.

### 9.4. Classes

(???)

### 9.5. Pointers

(???)

- Pointers to functions behave exactly like other pointers with respect to initialized and uninitialized.
- Virtual functions are guaranteed to be initialized by language rules, as ever.
- References are guaranteed to be initialized by language rules, as ever.

### 9.6. Concepts

Add uninitialized requirements to the concepts that express the requirement to take uninitialized memory as arguments.

- (???)

### 9.7. Library functions

The standard-library support is a single function plus the use of concepts that enforce the requirement to take uninitialized arguments on existing functions that already require uninitialized memory.

- Apply `[[ref_to_uninit]]` to arguments and return types that refer to uninitialized objects.
- Add `now_init()` to the library.

## Conclusion

What the body establishes is how little machinery the guarantee needs. The mechanisms are deliberately small: three attributes (`[[uninit]]`, `[[ref_to_uninit]]`, `[[must_init]]`) and one function template (`now_init()`). The length of the design comes from the cases that must still work, uninitialized buffers, memory pools, slot-based containers, and from the alternatives that were tried and rejected, chiefly definite assignment and any scheme that would make initialized-versus-uninitialized part of the type system. Keeping that distinction out of the type system is what preserves the profile's central property: code means the same thing whether or not the profile is enforced.

The design provides a foundational guarantee that most other profiles and most ordinary code can rely on, in exchange for stricter rules only where uninitialized memory is deliberately used. Without it, the standing class of uninitialized-memory errors remains, alongside a C++26 response to them, erroneous behavior, whose reaction is implementation-defined and therefore not the hard guarantee foundational code needs.

Two things are unfinished. The concept-based overloading of Section 4.8 has no implementation and turns on a type-deduction question the implementers are still investigating; if deduction strips the initialization mark, the fallback is a pair of intrinsics for tag dispatch. And the standard wording of Section 9 is an early draft with sections still to be written. The implementers investigating Section 4.8, the authors of the profiles framework, and whoever drafts the remaining wording build on this work next. The design is offered as input to that work, and refinement is expected.

## Disclosure

The author provides information and serves at the pleasure of the committee.

The author is affiliated with Columbia University and works on the C++ profiles effort. This paper presents the design of the initialization profile as input to implementation work and to readers assessing the profile's effect on existing code; further design change is expected. P3402<sup>[1]</sup> is the other active design for the same initialization profile, with different notation (`[[indeterminate]]` and `[[profiles::suppress]]` in place of `[[uninit]]`, `[[ref_to_uninit]]`, and `now_init()`); this paper and P3402 are companion efforts toward the same goal. The profile is foundational to the profiles framework<sup>[2]</sup> and to other profiles, so its design choices have stakes beyond this paper. One genuine limitation of the approach is disclosed in the body and repeated here: the concept-based overloading of Section 4.8 has no implementation and depends on an unresolved type-deduction question, so the claim that the whole design is implemented holds with that one exception. This draft was produced with machine assistance.

This paper asks for nothing.

## Acknowledgments

Thanks to the work and comments of Vinnie Falco, Marc-Andr&eacute; Laverdi&egrave;re, Christopher Lapkowski, Charles-Henri Gros, Thomas K&ouml;ppe, Jason Merrill, Gabriel Dos Reis, Herb Sutter, David Vandevoorde, Vassil Vassilev, Michael Wong, the LEWG, and the EWG for work on the initialization profile.

## References

[1] [P3402R3](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3402r3.html) - "A Safety Profile Verifying Initialization" (Marc-Andr&eacute; Laverdi&egrave;re, Christopher Lapkowski, Charles-Henri Gros, 2025).

[2] [P3589R2](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3589r2.pdf) - "C++ Profiles: The Framework" (Gabriel Dos Reis, 2025).

[3] [P2795R5](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2795r5.html) - "Erroneous behaviour for uninitialized reads" (Thomas K&ouml;ppe, 2024).

## Appendix A: Attribute placement

Should `[[uninit]]` and `[[ref_to_uninit]]` be placed after the type, after the name, or after any part of the type? The answer depends on logic, aesthetics, and implementability in the major compilers. The conclusion is used in the main text; the alternatives and the reasoning are here. Consider the candidate positions:

```cpp
int x [[uninit]];
int [[uninit]] y;
int* [[ref_to_uninit]] p;
int* q [[ref_to_uninit]];
```

For a pointer to uninitialized memory, there are three placements:

- `int* p3 [[ref_to_uninit]] = &x;` - plausible: `p3` points to something uninitialized.
- `int* [[ref_to_uninit]] p4 = &x;` - plausible: `p4` is an `int*` that points to something uninitialized, but someone might want the marker in a typedef, which could lead to serious complexity.
- `int* p5 = &x [[ref_to_uninit]];` - odd: it is unclear whether `x` or `p5` refers to something uninitialized.

The same choice arises for arrays:

- `int* [[ref_to_uninit]] arr1 [10];` - plausible: the attribute is next to the type it describes.
- `int* arr2 [[ref_to_uninit]] [10];` - plausible: the attribute is just after the name, as in variable definitions.
- `int* arr3 [10] [[ref_to_uninit]];` - plausible: the attribute is at the end, where the initializer would be.

In my own examples, I had used the first alternative consistently for functions and the last for arrays, which is neither consistent, likely to generalize, nor easy to explain with a single rule. The middle choice, the attribute just after the name, is the consistent one. So I place the attribute after the name, where the initializer would be for an initialized variable, and it applies to the object rather than to the type. The attribute does not change the meaning of a program: it does not change the type; it gives the analyzer the information needed to reject programs that fail to initialize properly.

Digging deeper, `[[ref_to_uninit]]` could be dropped entirely by allowing `[[uninit]]` anywhere in a type, for example `int [[uninit]] * p6 = &x;`, meaning `p6` points to an uninitialized int. This is tempting and general, but the generality brings complexity. Consider what such code, which is *not* proposed, would allow, and the ambiguity it would create:

```cpp
// not proposed:
int [[uninit]]* p7 = nullptr;       // p7 points to something uninitialized
int* p8 [[uninit]];                 // p8 is uninitialized; must point to initialized
int [[uninit]]* p9 [[uninit]];      // p9 is uninitialized; must point to uninitialized
```

Types can be arbitrarily complex, as in this reference-to-array-of-uninitialized-ints:

```cpp
void f88(int [[uninit]] (&x)[10]); // reference to array of uninitialized ints
```

Implementers point out that embedding attributes in types is distinctly nontrivial, that the implications are not fully specified in the standard, and that existing implementations do not cover these cases. The fundamental problem is that it is hard, in current compiler architectures, to carry the information from where it is stated, somewhere in the type specification, to where it is used, in the declaration, without losing the attribute's meaning as placement varies. I fear that people will come to see `[[uninit]]` as part of the type and demand that it become part of the type. That would add complexity, development time, and compile time, would affect the overload rules, and would break the principle that profiles do not change the meaning of code. This design does not take that path.

The placement problem cannot be escaped entirely, because of smart pointers and function declarations. For example, where there is no name to attach the attribute to:

```cpp
smart_ptr<int [[uninit]]> p;   // pointer to uninitialized int
void fct1(int [[uninit]]);     // error: argument passing is initialization
int fct(int* [[ref_to_uninit]]);
```

Here the attribute comes last, in the position it would occupy after a name. This is where implementer input is essential; I proceed on the assumption that the notation is manageable. A variant arises with function return types:

- `int* [[ref_to_uninit]] f1(int);` - plausible: the attribute is next to the type it describes.
- `int* f2 [[ref_to_uninit]] (int);` - plausible: the attribute is just after the name, as for variables.
- `int* f3 (int) [[ref_to_uninit]];` - odd: the attribute is far from what it refers to.

The first is cleanest, the second easiest to implement and consistent with what is proposed for variables, the third potentially confusing. For now I use what I consider cleanest for arrays, functions, and template arguments: annotations for return types by the name, annotations on argument types after the argument type. The adopted set is:

```cpp
int* f1 [[ref_to_uninit]] (int);
int arr [[uninit]] [7];
void f2(int* [[ref_to_uninit]]);
void f3(int* arg [[ref_to_uninit]]);
auto f4 [[ref_to_uninit]] (int) -> int*;
unique_ptr<int [[uninit]]> up;
```

When doing initial implementations, cleaner and easier-to-implement alternatives that do not change semantics should be sought. As ever, removing the initialization attributes yields a program with unchanged semantics. The placement of `[[now_init]]` follows the return-type rule, next to the name it describes:

```cpp
void fct1 [[now_init]] (int* p [[ref_to_uninit]]);
```

This is what is needed to fit existing initialization functions into the validation framework (the corresponding wording remains to be drafted).

## Appendix B: Glossary

- **Initialization profile** - the opt-in profile defined by this paper: it guarantees, at compile time and zero run-time cost, that no object is read or written before it is initialized.
- **Profile** - a named, opt-in set of enforceable guarantees, processed within the profiles framework<sup>[2]</sup>; enforcing a profile never changes the meaning of conforming code.
- **Suppression** - turning the profile off for a region of code, so that code the profile cannot express can still be written.
- **`[[uninit]]`** - an attribute marking an object as intentionally uninitialized; the object must be initialized before use, after which it is no longer `[[uninit]]`.
- **`[[ref_to_uninit]]`** - an attribute marking a pointer, reference, or smart pointer as referring to zero or more uninitialized objects; the referent is treated as `[[uninit]]`.
- **`[[must_init]]`** - an attribute implying `[[ref_to_uninit]]` and also requiring that what the pointer refers to be initialized after the call.
- **`now_init()`** - a function template that returns an initialized pointer from a `[[ref_to_uninit]]` pointer; it is a deliberate, reviewable hole in enforcement that cannot compile with the profile on and is a no-op with it off.
- **`[[now_init]]`** - an attribute form of `now_init()` for functions that leave their pointed-to argument initialized after the call.
- **`construct_at()` / `destroy_at()`** - the standard operations that initialize an object in raw memory and that end an object's lifetime; under the profile, `construct_at()` requires an uninitialized argument, and `destroy_at()` leaves its argument uninitialized.
- **`uninitialized_*()` family** - the standard range algorithms (for example `uninitialized_fill`) that initialize a range of raw memory; their iterator arguments are annotated `[[ref_to_uninit]]`.
- **Local static analysis** - analysis that examines individual declarations, statements, and functions without whole-program or symbolic execution; the only analysis the profile assumes.
- **Definite assignment** - the rule, used by Ada and C#, that a variable be assigned before use; rejected here (Section 1.4) because it would give code different meanings with and without the profile.
- **Subset-of-superset strategy** - adding notation and libraries so that analysis becomes tractable, then restricting the language to the part over which the guarantee holds.
- **Slot** - a unit of storage that holds an object of a known type, used in place of `void*` when implementing containers over uninitialized memory.
