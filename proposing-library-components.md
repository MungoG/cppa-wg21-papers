# So you want to propose your library component to the standard?

**Document**: DXXXX  
**Date**: 2025-01-26  
**Audience**: WG21, Library Evolution  
**Author**: Vinnie Falco

## Abstract

Every few months, someone approaches WG21 with a library component they want standardized. Sometimes it's a container they've battle-tested. Sometimes it's a utility they believe fills an obvious gap. Sometimes it's an algorithm they invented last Tuesday. These proposals often end up languishing in committee, or getting rejected after years of effort, or being responded to with polite skepticism (in keeping with committee tradition).

This paper explains why.

---

## 1. Introduction

WG21 receives a steady stream of library proposals. Most don't succeed. The failure rate isn't because the committee is hostile or the ideas are bad. It's because the bar for standardization is extraordinarily high—higher than most proposers realize—and the evidence required to clear that bar is enormous.

This paper aims to be useful to prospective proposers by explaining:

- Why your component might not be as essential as you think (§2)
- Why standardization is harder than you think (§3)
- What evidence you must provide (§4)
- Why most people don't provide that evidence (§5)
- What alternatives exist (§6)

If you read this paper and still want to propose your library component, you'll be better prepared. If you read it and decide to pursue alternatives, you'll save years of your life and contribute more effectively to the C++ ecosystem.

---

## 2. Your Component Might Not Be as Essential as You Think

### 2.1 The Committee Has Seen Hundreds of Ideas

Experienced committee members have evaluated hundreds of library proposals over decades. Most end up discarded—not because they're bad, but because the boundary between what belongs in the standard and what doesn't is very sharp.

The reason many standard library components are similar in their essential nature is not for lack of imagination on the proposers' part. It's because the characteristics that make something suitable for standardization are rare:

- **Types requiring compiler intrinsics** — `std::initializer_list`, `std::coroutine_traits`
- **Core vocabulary types** — `std::optional`, `std::string_view`, `std::span`
- **Cross-platform OS abstractions** — `std::filesystem`, `std::thread`
- **Fundamental algorithms and data structures** — `std::vector`, `std::sort`

Your component probably doesn't fit these categories. Most don't. That doesn't make it bad—it makes it unsuitable for *this particular distribution mechanism*.

### 2.2 The Vocabulary Type Illusion

Many proposers claim their component should be a "vocabulary type"—something the ecosystem should agree on so libraries can interoperate. But vocabulary necessity requires evidence:

- **Multiple libraries currently suffer from type incompatibility.** Where?
- **Specific examples of failed interoperability.** What broke?
- **Type disagreement causes real problems.** For whom?

If you cannot point to coordination failures happening *today* that standardization would solve, your component is likely a utility, not a vocabulary type. Utilities can live happily in external libraries. Only vocabulary types have a strong case for standardization.

### 2.3 The "Everyone Needs This" Fallacy

"Everyone needs a good X" is not evidence. The committee has learned—often painfully—that generalizing from personal experience to universal need is unreliable.

`std::regex` seemed like something everyone needed. It shipped. Its performance is so poor that respected experts tell users to never use it. `std::unordered_map` seemed essential. Its mandated node-based implementation prevents performance optimizations that users actually need.

What you think everyone needs and what the ecosystem actually requires are often different. The gap between them is measured in committee time, implementation burden, and user frustration.

---

## 3. Standardization Is Harder Than You Think

### 3.1 Implementation Time Grows Exponentially

A simple utility can be written in an evening. A utility with allocator support, exception safety, constexpr compatibility, and interaction with 40 years of existing standard library facilities takes months. These time frames are not hypothetical.

Even experienced proposers often have serious difficulty estimating the complexity of specification work. If you think you know how hard it is to standardize your component, you are likely wrong by a factor of 3-10x.

### 3.2 The Perpetual Cost Problem

The cost to propose is finite. The cost to maintain is unbounded.

Once your component enters the standard:

- Every future proposal must analyze interaction with your component
- Every compiler vendor must implement and maintain it forever
- Every defect report consumes committee time indefinitely
- Every ABI concern constrains future evolution permanently
- Every teaching resource must explain when to use it versus alternatives

You pay once to get in. The ecosystem pays forever to keep you. This asymmetry explains committee reluctance: a "yes" creates eternal obligations; a "no" creates none.

### 3.3 The Three-Year Death March

Standardization operates on three-year cycles with immovable deadlines. Your component must:

- Be proposed (with working implementation)
- Survive SG review
- Survive LEWG design review (multiple meetings, typically)
- Survive LWG wording review (multiple meetings, typically)
- Survive two rounds of national body comment
- Not regress under straw polls in plenary

This process typically takes 3-7 years for significant library components. Every step requires you to attend meetings, respond to concerns, revise papers, and update implementations. If your motivation is "I'll propose it and the committee will handle it," you've already failed.

---

## 4. The Evidence You Must Provide

If you want committee members to be motivated by your idea, you must sell the idea. Here's the evidence required:

### 4.1 Demonstrated Need (Not Asserted Need)

- **User surveys.** Does your component rank highly in what developers want? The C++ ecosystem surveys (JetBrains, ISO, etc.) provide data. If your component isn't appearing in these surveys, demand is unproven.
- **StackOverflow questions.** Are users struggling to solve the problem your component addresses? Quantity and recurrence matter.
- **Existing workarounds.** What are users doing today without your component? Document the inadequacy of these workarounds with specifics, not generalizations.

### 4.2 Production Deployment Experience

- **Years of use.** Your component should have been deployed in production for years, not months.
- **Multiple independent users.** One company's internal use is insufficient. Multiple organizations using your library in different domains demonstrates generality.
- **Known failure modes.** What went wrong? What did you fix? Evolution through real-world iteration is evidence of maturity.

### 4.3 Implementation Reality

- **Reference implementation.** You need one that works. Not "mostly works." Not "will work after some cleanup." Works.
- **Multiple implementations.** Ideally, your component has been implemented by multiple parties. This proves the interface is implementable and reveals design issues.
- **Performance data.** Real benchmarks from real deployments. Marketing numbers don't count.

### 4.4 Alternative Analysis

- **Why not Boost?** Seriously answer this. Boost provides ecosystem benefits without standardization costs.
- **Why not vcpkg/conan?** Package managers solve distribution. Why do you need the standard?
- **Why not header-only?** If your component can be distributed as a header, standardization provides minimal benefit over GitHub.

### 4.5 Cost Acknowledgment

Your proposal should explicitly quantify:

- Implementation burden for vendors (estimated person-months)
- Ongoing maintenance burden (defect rate from comparable facilities)
- Teaching burden (where does this fit in C++ education?)
- Opportunity cost (what doesn't get done while implementing this?)

Proposals that ignore costs are proposals that don't understand the system they're entering.

---

## 5. Why Most People Don't Provide This Evidence

### 5.1 They Don't Know It's Required

Nothing in the WG21 submission process explicitly demands this evidence. You can submit a paper with just an idea and some proposed wording. The process accepts it. The committee then rejects it—politely, after consuming everyone's time.

This paper exists to make requirements explicit.

### 5.2 The Evidence Is Expensive to Gather

Demonstrating production deployment takes years. Gathering survey data requires resources. Building multiple implementations requires collaborators. Most proposers don't have years and resources to invest before the "real work" of standardization even begins.

But the evidence isn't optional. It's just not formally required. Proposals that skip it fail informally instead of formally. The failure is slower, more frustrating, and consumes more of everyone's time.

### 5.3 Enthusiasm Substitutes for Evidence

"This would be really useful" feels like evidence. It isn't. Enthusiasm is necessary but insufficient. The committee sees dozens of enthusiastic proposers. What distinguishes success is proof, not passion.

### 5.4 The "Obviously Good" Trap

Some components seem so obviously useful that evidence feels unnecessary. Why document need for a container everyone obviously needs?

`std::hive` seemed obviously useful. Game developers need stable iterators with fast insertion/deletion! Except the games industry has their own solutions, doesn't use `plf::colony`, and won't use `std::hive`. "Obviously" and "actually" diverge more than enthusiasts expect.

---

## 6. Alternatives to Standardization

If your goal is to help C++ programmers, standardization is often the wrong path. Here are alternatives that deliver value faster with less cost:

### 6.1 Boost

**Advantages:**

- Rigorous review process that improves your design
- Wide distribution through a trusted brand
- No ABI freeze—you can fix mistakes
- No three-year cycle—ship when ready
- Maintainer retains control

**Disadvantages:**

- You must maintain it (but you'd maintain a standard component too, through defect reports)
- Less prestige than "in the standard" (but more than "rejected by the standard")

If your component is good, Boost acceptance provides ecosystem benefits comparable to standardization without the costs. Many standard library components (smart pointers, filesystem, regex, optional, variant, any, string_view) came from Boost. Let Boost prove the design before proposing.

### 6.2 Standalone Library with Package Manager Distribution

**Advantages:**

- Total control over design and evolution
- Ship immediately
- Fix bugs without committee process
- Users who want it get it; users who don't aren't burdened

**Disadvantages:**

- Discoverability requires marketing
- No "blessed" status

For most library components, this is the right answer. The standard library is not meant to be comprehensive—it's meant to provide what *cannot be provided elsewhere*. If your component can be distributed via vcpkg, conan, or direct inclusion, it probably should be.

### 6.3 Contribute to an Existing Standard Proposal

**Advantages:**

- Leverage someone else's evidence gathering
- Join a team rather than going alone
- Improve something already in motion

**Disadvantages:**

- Less control over direction
- May require compromising your vision

If your component overlaps with an active proposal, joining forces is often more effective than competing. The committee rewards collaboration over fragmentation.

### 6.4 Write Papers That Help, Not Papers That Propose

**Advantages:**

- Lower bar for acceptance
- Immediate impact on committee direction
- Build reputation without multi-year commitment

**Disadvantages:**

- Doesn't get your component in the standard

Papers analyzing problems, surveying existing solutions, or evaluating design approaches are valuable to the committee and don't require implementation or deployment evidence. If you want to contribute to WG21 but lack the resources for a full library proposal, this path builds credibility while helping others.

---

## 7. If You Still Want to Propose

After reading this paper, if you still believe standardization is the right path for your component, here's how to succeed:

### 7.1 Invest Years Before Proposing

Build the library. Deploy it in production. Gather users. Fix bugs. Document failure modes. Do this for 3-5 years. Then propose.

The committee is not a design shop. It's a standardization body. Bring a finished design backed by evidence, not an idea backed by enthusiasm.

### 7.2 Find Allies

No one succeeds alone. You need:

- Committee members who believe in your proposal
- Implementers willing to prototype
- Users willing to testify to need
- Reviewers willing to improve your wording

If you can't find these allies, the evidence for your component is probably insufficient. The failure to attract allies is itself diagnostic.

### 7.3 Be Prepared for Rejection

Most proposals fail. Failure after years of work is painful but common. If rejection would devastate you, standardization is the wrong path. Your emotional well-being matters more than any library component.

### 7.4 Consider Whether You Want This

The proposal process is exhausting. Meetings cost thousands of dollars in travel. Discussions consume hundreds of hours. NB comments arrive years after you thought you were done. Defect reports continue indefinitely.

Is this how you want to spend the next 5-10 years of your life? For some people, yes—they find the work meaningful and the impact worthwhile. For most people, the alternatives deliver more value with less cost.

Only you can make that judgment.

---

## 8. Conclusion

The bar for standardization is high because the costs of standardization are high. The committee isn't being hostile when it demands evidence—it's being responsible about perpetual maintenance burdens.

Your library component might be excellent. That doesn't make it a candidate for standardization. The questions are:

- Does it solve a problem that *cannot be solved elsewhere*?
- Is there *evidence* of widespread need, not just assertion?
- Has it been *validated* in production over years?
- Are you prepared for *5-10 years* of committee engagement?

If yes to all four, bring your proposal and your evidence. If no to any, consider the alternatives. The C++ ecosystem is better served by a thriving external library than by a half-baked standard component.

To put it simply: standardization is hard work, and the committee needs motivation. If you want them to be motivated by the idea, you have to sell the idea—with evidence, not enthusiasm. If you can't provide that evidence, or won't invest the years required to gather it, the alternatives aren't consolation prizes. They're often the right answer.

---

## 9. Acknowledgments

This paper is inspired by the forum post "README - For non-programmers with great ideas" from KVR Audio (Rock Hardbuns et al., 2007), which explained to non-programmers why their VST plugin ideas weren't getting implemented. The dynamics are remarkably similar: enthusiastic proposers underestimating the work required, overestimating the uniqueness of their ideas, and not understanding what evidence would actually motivate implementers.

Thanks to the committee members who have patiently explained these dynamics to proposers over the years, and to the proposers whose painful experiences inform this paper's warnings.

---

## 10. References

### WG21 Papers

1. P3001R0. Müller, Jonathan; Laine, Zach; Lelbach, Bryce Adelstein; Sankel, David. "std::hive and containers like it are not a good fit for the standard library." October 2023.

2. P0939R4. Dos Reis, Gabriel. "Direction for ISO C++."

### External Sources

3. "README - For non-programmers with great ideas." KVR Audio Developer Forum. 2007.

4. Jabot, Corentin. "A cake for your cherry: what should go in the C++ standard library?"

5. Winters, Titus. "What Should Go Into the C++ Standard Library." Abseil Blog.

6. Müller, Jonathan. "What should be part of the C++ standard library?"

7. Lelbach, Bryce Adelstein. "What Belongs In The C++ Standard Library?" CppCon.

---

## Revision History

### R0 (2025-01-26)

- Initial version explaining evidence requirements for library standardization
- Identifies why proposals typically fail to meet the bar
- Suggests alternatives to standardization for library authors
