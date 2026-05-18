# On Bugfixing

Vinnie Falco
March 2026

Audience: Directions Group

---

## Abstract

An organization that cannot correct its mistakes will become overwhelmed by them.

---

## 1. The Premise

Software has bugs. Standards have bugs. Every revision of every standard in the history of computing has shipped known defects alongside intended features.

The question is not whether WG21 makes mistakes. It does. The question is whether WG21 can fix them.

---

## 2. The Asymmetry

Adding a feature to the standard requires one paper, one set of favorable polls, and one revision cycle. Removing or replacing a feature requires deprecation, a migration path, ABI coordination across every major implementation, and the political will to tell users that the committee was wrong.

The process is structurally asymmetric. D&E records the observation:

> "It is much easier to accept a proposal than to reject it."

This asymmetry is not a flaw in the process. It is the process. The committee is designed to add. It is not designed to subtract. The result is a ratchet: each cycle turns one direction.

---

## 3. The Record

The committee has acknowledged mistakes it has not corrected.

`std::regex` was standardized in C++11. Its performance characteristics are widely understood to be unacceptable. Fifteen years later, the interface is unchanged. No replacement has been proposed. No fix has been applied.

The Networking TS was proposed in 2005. Twenty-one years later, C++ has no standard networking. The committee could not ship it. The committee could not cancel it. The committee could not replace it. The proposal exists in a state that is none of these.

`auto_ptr` was deprecated in C++11 and removed in C++17. This is the single example of the committee completing a full removal cycle for a standard library component. It took six years from deprecation to removal, for a type whose replacement (`unique_ptr`) was shipped in the same standard that deprecated it.

`volatile` compound assignment and increment/decrement were deprecated in C++20. Six years later, the deprecation stands. No removal has been proposed.

The pattern is visible. The committee can add. The committee can deprecate. The committee has completed one removal in its history.

---

## 4. The Consequence

D&E records the constraint directly:

> "Often, it is not feasible to eliminate a feature or correct a mistake."

If the ratchet turns only one direction, the standard accumulates mistakes at the rate they are made minus the rate they are corrected. The rate of addition is measured in dozens of features per cycle. The rate of correction is measured in one removal across the committee's entire history.

Each cycle, the surface area grows. Each cycle, the ratio of known mistakes to total surface area either holds steady or increases. No mechanism in the current process reverses this. Deprecation is not reversal. Deprecation is a label. The feature remains. The ABI remains. The teaching burden remains. The interaction surface with every future feature remains.

---

## 5. The Test

This paper makes one claim: WG21 must demonstrate, concretely, that it can fix at least one significant mistake. Not deprecate. Fix. Remove, replace, or correct.

The specific mistake does not matter. The capability matters.

An organization that has never reversed a significant decision has no evidence that it can. This is not a criticism of the committee's membership, its leadership, or its intentions. It is an observation about what the institution has demonstrated.

**Capacity that has never been exercised is indistinguishable from capacity that does not exist.**

---

## 6. The Cost of Waiting

ABI makes every unfixed mistake permanent on a timeline the committee does not control. Implementations ship. Users compile against shipped interfaces. The cost of correction increases monotonically with time.

Each standard that ships without a correction raises the cost of the next correction. The window does not stay open. It narrows. An organization that defers correction indefinitely is an organization that has chosen, by inaction, never to correct.

The committee has demonstrated that it can build. The question is whether it can also repair. The answer is not yet in evidence.
