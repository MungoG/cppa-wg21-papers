---
title: "Per-Item Wording for std::core_ub: What the Profile Delivers"
document: D0000R0
date: 2026-08-10
intent: info
audience: EWG, SG22
reply-to:
  - "Vinnie Falco <vinnie.falco@gmail.com>"
---

## Abstract

This paper states, for each of the 77 runtime-checkable cases of core-language undefined behavior, what the standard says when `std::core_ub` (P4317R1) is enforced.

P4317R1 specifies the profile's guarantee, activation, and response but defers the per-item normative text. This companion supplies it. For 62 cases the response is termination: the program ends rather than proceeding into undefined behavior. For 15 cases the profile defines a replacement value, fixed for every conforming implementation. No foundational wording changes are required; no contract-violation handler is named; no semantic selection exists. The profile owns the guarantee and the response directly.

---

## Revision History

### R0: August 2026

- Initial version.

---

## 1 Introduction

P4317R1 specifies `std::core_ub`, a profile under P3589R2 that guards the 77 runtime-checkable cases of core-language undefined behavior enumerated in P3100R8. When enforced, a violated precondition ends the program rather than proceeding into undefined behavior. For 15 cases where a well-defined replacement exists, the profile defines the operation's meaning directly.

This paper provides the per-item normative text. Each item states the standard paragraph as it reads under enforcement, organized by the natural category of operation. The wording patterns are two:

**Terminate (62 cases).** The operation has undefined behavior. When `std::core_ub` is enforced, detection of the condition ends the program. Before termination, the implementation is permitted to issue a runtime diagnostic. An implementation may offer a throwing alternative to termination as a non-portable extension; the interaction with `noexcept` is analyzed in P4308R0.

**Replacement (15 cases).** When `std::core_ub` is enforced, the operation produces a defined erroneous value rather than undefined behavior. The value is fixed for every conforming implementation.

Contributions:

1. Normative wording for 15 replacement-value items under profile enforcement.
2. Normative wording for 62 terminate items under profile enforcement.
3. Demonstration that all 77 items are specified without foundational wording changes, handler routing, or semantic selection.

The 3 cases P3100R8 identifies as not runtime-checkable are excluded (Appendix A).

---

## 2 Arithmetic Overflow and Division

All items in this group have replacement behavior: the operation produces an unspecified erroneous value.

### 2.1 `[expr.expr.eval]` - Signed integer overflow

When `std::core_ub` is enforced: if the mathematical result of evaluating an arithmetic expression is neither in the range of representable values for its type nor a negative infinity, positive infinity, or NaN that is among the values of the type, the result is an unspecified erroneous value.

### 2.2 `[expr.mul.div.by.zero]` - Division and remainder by zero

When `std::core_ub` is enforced: if the second operand of / or % is zero, the result is an unspecified erroneous value.

### 2.3 `[expr.mul.representable.type.result]` - Overflowing division/remainder

When `std::core_ub` is enforced: if the quotient a/b is not representable in the type of the result, the results of both a/b and a%b are unspecified erroneous values.

---

## 3 Shift Operations

### 3.1 `[expr.shift.neg.and.width]` - Invalid shift amount

When `std::core_ub` is enforced: if the right operand is negative, or greater than or equal to the width of the promoted left operand, the result is an unspecified erroneous value.

---

## 4 Type Conversions

All items in this group have replacement behavior.

### 4.1 `[conv.double.out.of.range]` - Floating-point narrowing out of range

When `std::core_ub` is enforced: if the source value cannot be exactly represented and is not between two adjacent destination values, the result is an unspecified erroneous value.

### 4.2 `[conv.fpint.float.not.represented]` - Float-to-integer out of range

When `std::core_ub` is enforced: if the truncated value cannot be represented in the destination integer type, the result is an unspecified erroneous value.

### 4.3 `[conv.fpint.int.not.represented]` - Integer-to-float out of range

When `std::core_ub` is enforced: if the value being converted is outside the range of values that can be represented in the destination floating-point type, the result is an unspecified erroneous value.

### 4.4 `[expr.static.cast.enum.outside.range]` - Enum cast outside range

When `std::core_ub` is enforced: if the enumeration type does not have a fixed underlying type and the value is not within the range of the enumeration values, the result is an unspecified erroneous value.

### 4.5 `[expr.static.cast.fp.outside.range]` - Explicit FP cast outside range

When `std::core_ub` is enforced: if the source floating-point value is not exactly representable and is not between two adjacent values of the destination type, the result is an unspecified erroneous value.

### 4.6 `[conv.lval.valid.representation]` - Invalid bit pattern

When `std::core_ub` is enforced: if the bits in the value representation of the object are not valid for the object's type, the prvalue result is an unspecified erroneous value.

---

## 5 Indeterminate Values

### 5.1 `[basic.indet.value]` - Reading uninitialized storage

When `std::core_ub` is enforced: if an indeterminate value is produced by an evaluation of a built-in type, the result is an unspecified erroneous value. For non-built-in types, the program terminates.

---

## 6 Pointer and Array Operations

### 6.1 `[expr.sub.pointers.representable]` - Pointer difference overflow (replacement)

When `std::core_ub` is enforced: if the value i-j is not in the range of representable values of type std::ptrdiff_t, the result is an unspecified erroneous value.

### 6.2 `[expr.add.out.of.bounds]` - Pointer arithmetic out of bounds (terminate)

When `std::core_ub` is enforced: if adding to or subtracting from a pointer produces a result that does not point to an element of the same array or one past the end, the program terminates.

### 6.3 `[expr.add.sub.diff.pointers]` - Subtracting pointers into different arrays (terminate)

When `std::core_ub` is enforced: if P and Q do not point to elements of the same array object (or one past the end), the program terminates.

### 6.4 `[expr.add.not.similar]` - Pointer arithmetic on dissimilar type (terminate)

When `std::core_ub` is enforced: if pointer arithmetic is performed through a pointer to an object that is not similar to the element type of the array, the program terminates.

### 6.5 `[expr.assign.overlap]` - Overlapping assignment (terminate)

When `std::core_ub` is enforced: if the value of the right operand is stored to a memory location that overlaps with the left operand, the program terminates.

### 6.6 `[basic.stc.alloc.zero.dereference]` - Dereferencing zero-size allocation (terminate)

When `std::core_ub` is enforced: if a pointer returned by a zero-size allocation is dereferenced, the program terminates.

---

## 7 Control Flow

### 7.1 `[stmt.return.flow.off]` - Flowing off a non-void function (replacement/terminate)

When `std::core_ub` is enforced: if a non-void function that is neither main nor a coroutine flows off its end, the return storage is initialized with an unspecified erroneous value for built-in return types. For non-built-in return types, the program terminates.

### 7.2 `[stmt.return.coroutine.flow.off]` - Flowing off a coroutine (replacement/terminate)

When `std::core_ub` is enforced: if a coroutine whose promise type does not declare `return_void` flows off its end, the return storage is initialized with an unspecified erroneous value for built-in return types. For non-built-in return types, the program terminates.

### 7.3 `[dcl.attr.noreturn.eventually.returns]` - [[noreturn]] function returning (terminate)

When `std::core_ub` is enforced: if a function declared `[[noreturn]]` returns to its caller, the program terminates.

### 7.4 `[dcl.attr.assume.false]` - Violated assumption, pure subgroup (terminate)

When `std::core_ub` is enforced: if an assumption's side-effect-free predicate would evaluate to false, the program terminates.

---

## 8 Object Lifetime

All items in this group terminate.

### 8.1 `[lifetime.outside.pointer.delete]` - Delete through pointer to dead object

When `std::core_ub` is enforced: if a delete-expression is applied through a pointer to an object whose lifetime has ended, the program terminates.

### 8.2 `[lifetime.outside.pointer.member]` - Member access through pointer to dead object

When `std::core_ub` is enforced: if a non-static data member or non-static member function is accessed through a pointer to an object whose lifetime has ended, the program terminates.

### 8.3 `[lifetime.outside.pointer.virtual]` - Virtual call through pointer to dead object

When `std::core_ub` is enforced: if a virtual member function is called through a pointer to an object whose lifetime has ended, the program terminates.

### 8.4 `[lifetime.outside.pointer.dynamic.cast]` - dynamic_cast through pointer to dead object

When `std::core_ub` is enforced: if dynamic_cast is applied through a pointer to an object whose lifetime has ended, the program terminates.

### 8.5 `[lifetime.outside.glvalue.access]` - Glvalue access to dead object

When `std::core_ub` is enforced: if an lvalue-to-rvalue conversion is applied to a glvalue referring to an object whose lifetime has ended, the program terminates.

### 8.6 `[lifetime.outside.glvalue.member]` - Member access on glvalue of dead object

When `std::core_ub` is enforced: if a non-static data member is accessed through a glvalue referring to an object whose lifetime has ended, the program terminates.

### 8.7 `[lifetime.outside.glvalue.virtual]` - Virtual call on glvalue of dead object

When `std::core_ub` is enforced: if a virtual member function is called on a glvalue referring to an object whose lifetime has ended, the program terminates.

### 8.8 `[lifetime.outside.glvalue.dynamic.cast]` - dynamic_cast on glvalue of dead object

When `std::core_ub` is enforced: if dynamic_cast is applied to a glvalue referring to an object whose lifetime has ended, the program terminates.

### 8.9 `[original.type.implicit.destructor]` - Implicit destructor on wrong dynamic type

When `std::core_ub` is enforced: if an implicit destructor call occurs on an object whose original type differs from its dynamic type in a way that makes the destructor call undefined, the program terminates.

### 8.10 `[expr.type.reference.lifetime]` - Reference bound to object whose lifetime ended

When `std::core_ub` is enforced: if a reference is used to access an object whose lifetime has ended, the program terminates.

### 8.11 `[expr.dynamic.cast.pointer.lifetime]` - dynamic_cast on pointer to dead object

When `std::core_ub` is enforced: if dynamic_cast is applied to a pointer to an object whose lifetime has ended, the program terminates.

### 8.12 `[expr.dynamic.cast.glvalue.lifetime]` - dynamic_cast on glvalue to dead object

When `std::core_ub` is enforced: if dynamic_cast is applied to a glvalue referring to an object whose lifetime has ended, the program terminates.

### 8.13 `[class.dtor.no.longer.exists]` - Destructor on object that no longer exists

When `std::core_ub` is enforced: if a destructor is invoked on an object whose lifetime has already ended, the program terminates.

---

## 9 Object Model and Type System

All items in this group terminate.

### 9.1 `[intro.object.implicit.create]` - Implicit object creation failure

When `std::core_ub` is enforced: if no set of implicitly-created objects would give the program defined behavior, the program terminates.

### 9.2 `[intro.object.implicit.pointer]` - Invalid implicit-lifetime pointer

When `std::core_ub` is enforced: if a pointer to implicitly-created storage is used in a way that no valid object could justify, the program terminates.

### 9.3 `[basic.align.object.alignment]` - Misaligned access

When `std::core_ub` is enforced: if an object is accessed through storage that does not satisfy the object's alignment requirement, the program terminates.

### 9.4 `[creating.within.const.complete.obj]` - Modifying/creating within const complete object

When `std::core_ub` is enforced: if a new object is created in storage occupied by a const complete object, or a const complete object is modified, the program terminates.

### 9.5 `[expr.basic.lvalue.strict.aliasing.violation]` - Strict aliasing violation

When `std::core_ub` is enforced: if a glvalue of one type accesses the stored value of an object of an unrelated type in violation of the aliasing rules, the program terminates.

### 9.6 `[expr.basic.lvalue.union.initialization]` - Accessing inactive union member

When `std::core_ub` is enforced: if a non-active member of a union is accessed through a glvalue in violation of the union access rules, the program terminates.

### 9.7 `[dcl.type.cv.modify.const.obj]` - Modifying a const object

When `std::core_ub` is enforced: if an attempt is made to modify a const object through a non-const access path, the program terminates.

### 9.8 `[dcl.type.cv.access.volatile]` - Volatile access through non-volatile glvalue

When `std::core_ub` is enforced: if a volatile object is accessed through a non-volatile glvalue, the program terminates.

### 9.9 `[basic.compound.invalid.pointer]` - Use of an invalid pointer value

When `std::core_ub` is enforced: if an invalid pointer value is used, the program terminates.

### 9.10 `[expr.unary.dereference]` - Dereferencing an invalid pointer

When `std::core_ub` is enforced: if the unary * operator is applied to a pointer that does not point to a valid object or function, the program terminates.

### 9.11 `[expr.reinterpret.cast.invalid.pointer.value]` - reinterpret_cast producing invalid pointer

When `std::core_ub` is enforced: if a reinterpret_cast produces a pointer value that is invalid in a context where it is used, the program terminates.

---

## 10 Function Calls and Dispatch

All items in this group terminate.

### 10.1 `[expr.call.different.type]` - Calling through wrong function type

When `std::core_ub` is enforced: if a function is called through an expression whose type differs from the function's type, the program terminates.

### 10.2 `[expr.ref.member.not.similar]` - Member access on unrelated type

When `std::core_ub` is enforced: if a class member access is performed through a pointer to an object that is not of a type similar to the class, the program terminates.

### 10.3 `[expr.mptr.oper.not.contain.member]` - Pointer-to-member on wrong object

When `std::core_ub` is enforced: if a pointer-to-member is applied to an object that does not contain the member, the program terminates.

### 10.4 `[expr.mptr.oper.member.func.null]` - Calling through null pointer-to-member-function

When `std::core_ub` is enforced: if a null pointer-to-member-function is called, the program terminates.

### 10.5 `[conv.member.missing.member]` - Pointer-to-member conversion when member absent

When `std::core_ub` is enforced: if a pointer-to-member is converted and the destination class does not contain the original member, the program terminates.

### 10.6 `[conv.ptr.virtual.base]` - Pointer conversion involving virtual base

When `std::core_ub` is enforced: if a pointer to a derived class is converted to a pointer to a virtual base class and the object is not of the correct dynamic type, the program terminates.

### 10.7 `[dcl.ref.incompatible.function]` - Reference bound to incompatible function type

When `std::core_ub` is enforced: if a reference is bound to a function whose type is incompatible with the reference type, the program terminates.

### 10.8 `[dcl.ref.incompatible.type]` - Reference bound to incompatible object type

When `std::core_ub` is enforced: if a reference is bound to an object whose type is incompatible with the reference type, the program terminates.

### 10.9 `[dcl.ref.uninitialized.reference]` - Uninitialized reference

When `std::core_ub` is enforced: if a reference is used without having been initialized, the program terminates.

---

## 11 Memory Management

All items in this group terminate.

### 11.1 `[expr.new.non.allocating.null]` - Non-allocating new returning null

When `std::core_ub` is enforced: if a non-allocating allocation function returns null and the new-expression proceeds to construct, the program terminates.

### 11.2 `[expr.delete.mismatch]` - Delete type mismatch

When `std::core_ub` is enforced: if a delete-expression is applied and the static type of the operand differs from the dynamic type of the object, the program terminates.

### 11.3 `[expr.delete.array.mismatch]` - Delete[] type mismatch

When `std::core_ub` is enforced: if a delete[]-expression is applied and the static type of the operand differs from the dynamic type of the array elements, the program terminates.

### 11.4 `[expr.delete.dynamic.type.differ]` - Delete with wrong dynamic type

When `std::core_ub` is enforced: if a scalar delete-expression is applied and the object's dynamic type differs from its static type without a virtual destructor, the program terminates.

### 11.5 `[expr.delete.dynamic.array.dynamic.type.differ]` - Delete[] with wrong array element type

When `std::core_ub` is enforced: if a delete[]-expression is applied and the array's element dynamic type differs from its static type, the program terminates.

### 11.6 `[basic.stc.alloc.dealloc.constraint]` - Replacement function violating constraints

When `std::core_ub` is enforced: if a replacement allocation or deallocation function violates its required behavior, the program terminates.

### 11.7 `[expr.static.cast.base.class]` - static_cast to base when object not of type

When `std::core_ub` is enforced: if a static_cast converts a pointer or reference to a base class and the object is not of the correct derived type, the program terminates.

### 11.8 `[expr.static.cast.downcast.wrong.derived.type]` - static_cast downcast mismatch

When `std::core_ub` is enforced: if a static_cast performs a downcast and the object's dynamic type is not the target derived type, the program terminates.

### 11.9 `[expr.static.cast.does.not.contain.original.member]` - static_cast member pointer mismatch

When `std::core_ub` is enforced: if a static_cast converts a pointer-to-member and the destination class does not contain the original member, the program terminates.

---

## 12 Construction and Destruction

All items in this group terminate.

### 12.1 `[class.abstract.pure.virtual]` - Calling a pure virtual function

When `std::core_ub` is enforced: if a pure virtual function is called, the program terminates.

### 12.2 `[class.base.init.mem.fun]` - Member function call during base initialization

When `std::core_ub` is enforced: if a virtual member function is called for an object under construction during base-class initialization in a way that would dispatch to the derived class, the program terminates.

### 12.3 `[class.cdtor.before.ctor]` - Use before construction

When `std::core_ub` is enforced: if a non-static member or base class is referred to before the object's construction begins, the program terminates.

### 12.4 `[class.cdtor.after.dtor]` - Use after destruction

When `std::core_ub` is enforced: if a non-static member or base class is referred to after the object's destructor completes, the program terminates.

### 12.5 `[class.cdtor.convert.pointer]` - Converting pointer during ctor/dtor

When `std::core_ub` is enforced: if a pointer to an object under construction or destruction is converted to a pointer to a class that is not a base of the constructor's or destructor's class, the program terminates.

### 12.6 `[class.cdtor.form.pointer]` - Forming pointer-to-member during ctor/dtor

When `std::core_ub` is enforced: if a pointer-to-member referring to a member of a not-yet-constructed or already-destroyed subobject is formed, the program terminates.

### 12.7 `[class.cdtor.virtual.not.x]` - Virtual call resolving to wrong class

When `std::core_ub` is enforced: if a virtual function call during construction or destruction resolves to a function in a not-yet-constructed or already-destroyed derived class, the program terminates.

### 12.8 `[class.cdtor.typeid]` - typeid during construction/destruction

When `std::core_ub` is enforced: if typeid is applied to an object under construction or destruction and would resolve to a not-yet-constructed or already-destroyed derived class, the program terminates.

### 12.9 `[class.cdtor.dynamic.cast]` - dynamic_cast during construction/destruction

When `std::core_ub` is enforced: if dynamic_cast is applied to an object under construction or destruction in a way that would resolve to a not-yet-constructed or already-destroyed class, the program terminates.

### 12.10 `[except.handle.handler.ctor.dtor]` - Exception handler during object construction/destruction

When `std::core_ub` is enforced: if a handler is entered for an exception thrown during the construction or destruction of an object in a way that produces undefined behavior, the program terminates.

---

## 13 Concurrency

### 13.1 `[intro.races.data]` - Data race (replacement)

When `std::core_ub` is enforced: if a data race occurs, the implementation makes the conflicting accesses implicitly atomic (the operations execute in some unspecified interleaving without undefined behavior).

### 13.2 `[intro.execution.unsequenced.modification]` - Unsequenced modification (replacement)

When `std::core_ub` is enforced: if two unsequenced side effects on the same scalar object occur, the operations are sequenced in some unspecified order.

---

## 14 Program Lifecycle and Miscellaneous

### 14.1 `[basic.start.main.exit.during.destruction]` - Calling exit during static destruction (terminate)

When `std::core_ub` is enforced: if exit is called during the destruction of an object with static or thread storage duration, the program terminates.

### 14.2 `[basic.start.term.use.after.destruction]` - Using object after static destructor (terminate)

When `std::core_ub` is enforced: if an object with static storage duration is used after its destructor has completed, the program terminates.

### 14.3 `[stmt.dcl.local.static.init.recursive]` - Recursive static local initialization (terminate)

When `std::core_ub` is enforced: if control re-enters the initialization of a block-scope variable with static or thread storage duration during that initialization, the program terminates.

### 14.4 `[dcl.fct.def.coroutine.resume.not.suspended]` - Resuming non-suspended coroutine (terminate)

When `std::core_ub` is enforced: if a coroutine is resumed that is not in a suspended state, the program terminates.

### 14.5 `[dcl.fct.def.coroutine.destroy.not.suspended]` - Destroying non-suspended coroutine (terminate)

When `std::core_ub` is enforced: if a coroutine is destroyed that is not in a suspended state, the program terminates.

---

## Appendix A: Items Not Covered

The following 3 cases are identified by P3100R8 as not runtime-checkable. The profile does not guard them.

| Item | Reason |
|------|--------|
| `[intro.progress.stops]` | Determining whether a thread of execution makes progress is equivalent to solving the halting problem |
| `[dcl.attr.assume.false]` (nonpure subgroup) | The assumption predicate may have side effects; evaluating it to check would change program behavior |
| `[basic.compound.pointer.before.storage.duration]` | Requires knowing whether any potential pointer value will encounter undefined behavior in the future |

Additionally, `[temp.inst.inf.recursion]` is excluded by D4277R0 as addressed by CWG3034 (not a runtime evaluation issue).

---

## Conclusion

The 77 runtime-checkable cases of core-language undefined behavior each have a stated response under `std::core_ub` enforcement: 62 terminate, 15 produce a defined replacement value. The per-item wording requires no redefinition of "undefined behavior," no implicit-assertion concept, no contract-violation handler, and no semantic selection. The profile owns the guarantee and the response directly.

What the profile standardizes is what production hardening ships: a named set of checks, terminating on a violation (or producing a defined value where one exists), with the checking mechanism left to quality of implementation. The per-item text in this paper is the normative substance behind P4317R1's guarantee.

---

## Disclosure

The author is a participant in the profiles coalition and co-author of P4317, P4310, and P4297. This paper is a companion to P4317R1 and makes no independent request.

---

## References

- [1] P4317R1 - "A Profile for Runtime-Checkable Core-Language Undefined Behavior: std::core_ub" (Vinnie Falco, 2026)
- [2] P3100R8 - "Contracts for C++" (Joshua Berne, Timur Doumler, 2026)
- [3] D4277R0 - "Overview and Implementation Report for P3100" (Joshua Berne, 2026)
- [4] P3589R2 - "Profiles" (Herb Sutter, 2025)
- [5] P2795 - "Erroneous behaviour for uninitialized reads" (Thomas K&ouml;ppe, 2023)
- [6] P4310R1 - "Hasta la Vista, Undefined Behavior" (Vinnie Falco, Ville Voutilainen, 2026)
- [7] P4297 - "Severing the Profiles Claim" (Vinnie Falco, 2026)
