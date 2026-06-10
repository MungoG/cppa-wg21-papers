---
title: "Why I Write So Many Papers"
document: P4263R0
date: 2026-06-10
intent: info
audience: WG21
reply-to:
  - "Vinnie Falco <vinnie.falco@gmail.com>"
---

## Abstract

Thirty-six papers in four consecutive mailings, and the question most asked about them is why.

Between February and May 2026, thirty-six papers from the author and his collaborators entered the WG21 mailings, thirty-five of them information-only. Chairs and senior members keep asking why - and, separately, what the author wants. This paper answers both questions completely. The short form: the published record behind consequential committee decisions is thin; the thinness is rational, because the committee runs its papers as a zero-sum tournament whose dominant strategy is to claim the largest possible domain on the least possible evidence; nothing in the ISO framework requires any of this, so the structure is a choice, renewable and changeable at will; and an institution that performs the ceremonies of evidence without the incentives that make evidence matter cannot examine itself. The author supplies the record from outside the incentive structure - information-only papers that ask for nothing, the one move that is not a play in the tournament. The paper closes with the question it cannot answer from the record: in a structure where scoped claims with evidence do not prevail over unscoped claims without evidence, where do good technical outcomes come from?

---

## Revision History

### R0: July 2026 (Post-Brno)

- Initial revision.

---

## 1. Disclosure

The author provides information and serves at the pleasure of the committee.

The author developed and maintains [Capy](https://github.com/cppalliance/capy) and [Corosio](https://github.com/cppalliance/corosio) and believes coroutine-native I/O is a practical foundation for networking in C++.

Coroutine-native I/O and `std::execution` are complementary. Each serves the domain where its design choices pay off.

The author maintains proposals in the coroutine I/O space that address the same problem domain as sender-based networking.

This paper examines the published record. That effort requires re-examining consequential papers, including papers written by people the author respects.

This paper is drafted with AI.

This paper asks for nothing.

---

## 2. The Question

Since February 2026 the author has been asked, by chairs and by senior members, two questions more often than any technical question: why did you write so many papers, and what do you want? This section answers the first. Section 13 answers the second, in concrete terms, so that it does not need to be asked again.

The shape of the question is itself a finding, so it is recorded first. Thirty-five of the thirty-six papers are information-only: they request no floor time, ask for no polls, and compete with no proposal for a slot in any working draft; the thirty-sixth proposes an execution model through the ordinary channel. The question these papers draw is not "which claims are wrong?" - as of this writing, no specific claim in any of the thirty-six has been challenged in any committee venue. The question is "why do they exist?" A body that evaluated papers by their content would ask the first question. The body asks the second. The committee does not evaluate papers; it evaluates people, and the papers are being read as a fact about a person.

There is a structural twin that goes unasked: why is no one else meeting this evidentiary standard? Thirty-six papers in which every factual claim carries a citation, every quote is verified against its source, every prediction is dated and falsifiable - this is not a personal eccentricity. It is what an engineering record looks like. The asked question challenges one person's standing to produce so much. The unasked question would examine the process that makes such a record exceptional rather than ordinary.

The complete answer to the asked question, compressed. The remainder of the paper is the evidence, arranged so that a reader may stop at the end of any section and leave with a correct, if less detailed, understanding.

1. The author's technical interest is byte-oriented serial I/O - networking - built on C++20 coroutines.
2. C++26 adopted `std::execution`, a strong design in the domains it was built for, carrying a claim of universality that extends over the author's domain.
3. The author went into the published record expecting to find the evidence behind that claim. Section 3 documents what the search found: consensus recorded in the minutes, and an evidence column that is, for networking, empty.
4. Section 4 documents why the record is thin: papers compete in a zero-sum tournament whose dominant strategy is to claim the largest possible domain and disclose the least possible evidence. The thinness is not negligence. It is the winning move.
5. Section 5 documents the frame: nothing in the ISO Directives requires the tournament. The structure is self-authored, fully permitted, and changeable by the same local authority that built it, with no ISO process whatsoever.
6. Sections 6 through 8 document the consequence: an institution that performs the ceremonies of evidence - papers, polls, revision numbers - without the incentive structure that once made evidence matter, and the predictable ways such an institution receives evidence supplied from outside.
7. So the author writes papers: information-only, asking for nothing, addressed to the permanent public record - the one channel the tournament does not control, and the one move that is not a play in it. Sections 9 through 13 lay out that program, its term, and the conditions under which it would no longer be needed.

A reader who stops here holds the whole argument. What remains open is the question the author cannot answer from the record, and it is the question this paper exists to put on file: in a structure where a scoped claim with evidence cannot prevail over an unscoped claim without evidence, where do good technical outcomes come from?

**The asked question challenges a person. The unasked question examines a process.**

---

## 3. The Thin Record

The whole argument rests on one observable fact: the published record behind consequential decisions does not contain the evidence the decisions were announced as resting on. This section documents that fact through one centerpiece exhibit, told in the order the author lived it, and two secondary exhibits. Each exhibit is one paragraph where one paragraph suffices; the cited papers carry the depth.

### 3.1. What `std::execution` achieves

[P2300R10](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html)<sup>[1]</sup>, now `std::execution` in the C++26 working draft, provides compile-time sender composition with completion signatures checked by the compiler. It enables structured concurrency, with cancellation propagated through stop tokens as part of the protocol rather than as a convention. It serves heterogeneous compute dispatch, with deployments in GPU and infrastructure settings documented in the record. These properties are real, they ship in the reference implementation, and the author's own libraries consume them through bridges described below.

### 3.2. The search

The author's interest is byte-oriented serial I/O on C++20 coroutines; the corpus makes that obvious. P2300R10<sup>[1]</sup> entered the working draft carrying a wider claim - a basis for "most asynchronous use cases, including networking" - that extends over the author's domain. The author went into the historical record expecting to find the evidence behind the claim. The result of that search is six published papers tracing the decision chain end to end: the unification of executors ([P4094R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4094r1.pdf)<sup>[2]</sup>), the basis-operation pivot ([P4095R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4095r1.pdf)<sup>[3]</sup>), the P2464R0 diagnosis of the Networking TS ([P4096R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4096r1.pdf)<sup>[4]</sup>), the networking poll ([P4097R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4097r1.pdf)<sup>[5]</sup>), the claims-versus-evidence survey ([P4098R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4098r1.pdf)<sup>[6]</sup>), and the assembled causal chain ([P4099R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4099r1.pdf)<sup>[7]</sup>).

### 3.3. What the search found

In October 2021, LEWG polled: "The sender/receiver model (P2300) is a good basis for most asynchronous use cases, including networking, parallelism, and GPUs" - SF:24 / WF:16 / N:3 / WA:6 / SA:3, consensus in favor ([P2453R0](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2453r0.html)<sup>[8]</sup>). A Weakly Favor voter wrote: "I think this is a good basis for parallelism/GPUs but can't judge its suitability for networking." The chair's published interpretation: "In the short term, this poll result doesn't mean much. We don't have a paper in hand that proposes networking based on the [P2300R2] model." [P4097R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4097r1.pdf)<sup>[5]</sup> documents the full poll record. The word "networking" entered the consensus; no networking evidence accompanied it.

The unification that preceded the poll rests on one code example. [P0761R2](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0761r2.pdf)<sup>[9]</sup>, the Executors Design Document, argued that separate executor models force an N x M explosion of implementations, illustrated by a hypothetical `parallel_for` constructed by the proposal's own authors. [P4094R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4094r1.pdf)<sup>[2]</sup> searched the record and found that this snippet is the only code-level evidence published for any unification rationale - no measurement from a deployed standard library, no survey of applications, no deployment data.

[P4098R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4098r1.pdf)<sup>[6]</sup> tabulated twenty years of published claims about executors and networking against the published evidence for each. The evidence bar was set deliberately low: a code snippet from a real codebase counts, a prototype counts, a small user survey counts. The GPU and infrastructure deployments are real and documented. The networking cells are empty.

One absence is structural rather than evidentiary. C++20 coroutines prevent unbounded stack growth through symmetric transfer: a coroutine that awaits N synchronously-completing senders in a loop accumulates O(N) stack frames, while with symmetric transfer the same loop executes in O(1) stack space ([P2583R4](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p2583r4.pdf)<sup>[10]</sup>). The phrase "symmetric transfer" appears in no revision of the proposal, R0 through R10. No design rationale discusses the void-returning completion functions as a tradeoff. No revision history entry, across ten revisions and four years, records the alternative as considered and rejected.

Here the author offers testimony rather than analysis, because his convictions are data about why he writes. Coroutines entered the language in C++20. The committee spent the years that followed standardizing a competing asynchronous model whose specification never mentions the coroutine facility's central composition mechanism. It shipped. It was a mistake. The reader is not asked to adopt this verdict; the reader is asked to observe that the record permitted it to form.

The post-adoption record continues the pattern. [P4041R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4041r0.pdf)<sup>[11]</sup> enumerates the changes since adoption: twenty-four post-adoption items modified the sender sub-language or its integration; zero modified coroutines. The same paper surveys the ecosystem: the reference implementation has 1,300 stars; the sender I/O ecosystem built on it totals 21.

The field experience of the model's largest production deployment points the same way. Ian Petersen, a maintainer of Meta's libunifex, [wrote](https://github.com/facebookexperimental/libunifex/issues/586#issuecomment-1845934903)<sup>[12]</sup> in December 2023: "Our experience at Meta has been that coroutines are easier to read, write, debug, and just generally maintain than composition-of-sender algorithms-style code. The cost of that ease is basically overhead; coroutines don't optimize as well as raw senders (either for size or speed). The advice we give to internal teams adopting Unifex is that they should prefer coroutines until they know that the overheads are unacceptable, at which point they can refactor to the lower-level abstraction of raw senders."

### 3.4. The scoped alternative

The author's response to the empty evidence column was to fill it. The bridges [P4092R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4092r1.pdf)<sup>[13]</sup> and [P4093R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4093r1.pdf)<sup>[14]</sup> connect coroutine-native code and senders in both directions, built so that neither model forecloses the other. The scoped claim - coroutine-native I/O is the best fit for byte-oriented serial I/O - ships with its evidence: type-erased I/O at 36.4 ns and zero allocations per operation against 53.4 ns and one allocation for the sender equivalent ([P4088R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4088r1.pdf)<sup>[15]</sup>); at ten thousand connections performing one hundred operations per second each, the difference is a million allocations per second. The `retry` algorithm that P2300R10's own text presents at approximately 125 lines as a sender is seven lines as a coroutine, both drawn from the same specification ([P4178R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4178r0.pdf)<sup>[16]</sup>). A derivatives-exchange port measured the coroutine-native path at 21-27% lower P99 latency in its sustained-rate trade scenario ([P4125R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4125r1.pdf)<sup>[17]</sup>). The domain claim is narrow, the evidence is published, and the implementation runs on three platforms.

### 3.5. The same pattern elsewhere

Contracts. [P2900R14](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p2900r14.pdf)<sup>[18]</sup> is a significant achievement - fourteen revisions, sixteen design principles, wording review completed, two compiler implementations. It advanced through five years of subgroup polls in which preserving the status quo required no documented engagement with minority objections, and was adopted into the C++26 working draft with strong consensus. The objections did not disappear; they surfaced at the ballot, where seven national body comments requested removal ([P4208R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4208r0.pdf)<sup>[19]</sup>). Section 5 returns to why the objections surfaced exactly there.

Modules. Modules shipped in C++20. Six years later, the implementers' joint paper [P3962R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p3962r0.pdf)<sup>[20]</sup> records that "full conformance to recent standards remains difficult in practice, with some implementations still working toward C++20 conformance with limited capacity to adopt newer standards."

**The vote is in the minutes. The evidence is not.**

---

## 4. The Incentive: Over-Claim, Under-Evidence

Section 3 documented thinness. This section documents why thinness is rational - why the record looks the way it looks because of what the structure rewards, and not because of who its participants are.

### 4.1. The tournament

When two designs address the same problem, the committee's process selects one. The other does not ship smaller; it does not ship at all. Papers therefore compete in a zero-sum tournament, and the tournament has a dominant strategy with two halves.

Claim maximally. Every domain a paper claims is territory a rival cannot enter. A proposal scoped to GPU dispatch leaves networking open for a competitor; a proposal claiming "most asynchronous use cases, including networking" forecloses the competitor without ever producing a networking design. Universal claims are free real estate - the record documents that the universality was never evidenced ([P4041R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4041r0.pdf)<sup>[11]</sup>), and the tournament never priced that in.

Evidence minimally. Every benchmark, every disclosed tradeoff, every named limitation beyond the minimum needed to advance is ammunition for opponents. A paper that names its own costs hands the other side a poll argument. The rational author discloses exactly enough to pass the next gate and nothing more.

Claim maximally, evidence minimally: the twin distortion is the move the structure pays for, available to every author and rewarded whoever plays it. The quantity the structure optimizes is positional - which design wins - rather than informational: what is true, what was measured, what the users of the language would choose. User welfare does not appear in the objective function. It appears in the marketing.

### 4.2. The polls

The tournament resolves through polls, and the poll mechanics complete the loop. A contested design produces a bimodal poll: a strong-for bloc, a strong-against bloc, and a hollow middle, decided by which side shows up in force. The poll that stopped the Networking TS reads SF:13 / WF:13 / N:8 / WA:6 / SA:10 - no consensus, two mobilized blocs ([P2453R0](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2453r0.html)<sup>[8]</sup>). Poll tallies are composites of who was in the room and how the question was framed. The question polled is "do I want this?"; the question never polled is "is this correct?" - and a body that can choose between verification and social consensus converges on the cheaper method, because it is easier to achieve and harder to falsify.

The median delegate cannot independently evaluate most of the polls taken in a meeting week, so silence and deference are the dominant strategy, and the consensus model converts that rational ignorance into procedural legitimacy. Where expertise is missing, reputation substitutes for evidence; where seniors are silent, the silence reads as consent. M&uuml;ller [wrote](https://www.think-cell.com/en/career/devblog/trip-report-summer-iso-cpp-meeting-in-st-louis-usa)<sup>[21]</sup> of the St. Louis adoption: "Concerns were raised that maybe it wasn't reviewed properly, as committee members were not able to fully understand the intricate design details, and instead just trusted the authors that they did a good enough job." In such a system, correctness is the entry fee, not the prize.

### 4.3. The presentation trap

Within this structure, an invitation to present a challenging analysis is not an invitation to reconcile it. It is an invitation to an unfavorable poll. The room votes at the end of the session; the vote tracks the room's composition; the composition favors the incumbent whose ecosystem fills the seats. The tournament has exactly one way to process evidence - vote on it - and a vote is a measurement of mobilization, not of evidence. The author has watched this mechanism operate from inside the room.

### 4.4. No revisit

The equilibrium compounds, because thin records cannot be revisited. The basis-operation pivot was decided under a single framing of the design space, with the second framing never analyzed ([P4095R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4095r1.pdf)<sup>[3]</sup>); the Networking TS diagnosis evaluated the specification under that same inherited framing ([P4096R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4096r1.pdf)<sup>[4]</sup>). When a decision's file contains a poll tally and no rationale, there is nothing on record to check the decision against when conditions change. The decision becomes permanent not because it was confirmed but because it cannot be examined.

### 4.5. Scored predictions

The cost is measurable. [P4047R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4047r0.pdf)<sup>[22]</sup> collected twenty-seven dated, public, falsifiable predictions about `std::execution` from proponents and critics alike and scored each against the record: eighteen confirmed. The critics' warnings on safety, correctness, and networking proved predictive. The proponents' universality claims did not. Timeline estimates failed on all sides. No mechanism exists to attach consequences to either outcome - the predictions that proved wrong cost their authors nothing, and the warnings that proved right earned their authors nothing.

### 4.6. The countermeasure that exists

Adversarial review of a committee paper now costs under a dollar and takes about fifteen minutes. [P4207R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4207r0.pdf)<sup>[23]</sup> demonstrates the method on the contracts proposal: approximately fifteen candidate charges filed against [P2900R14](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p2900r14.pdf)<sup>[18]</sup>, twelve killed under cross-examination, eleven sections certified as battle-hardened, one objection surviving ([P4208R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4208r0.pdf)<sup>[19]</sup>). The tool exists, it is cheap, and it is self-inflicted - an author runs it on their own paper before the committee sees it. Nothing in the current structure rewards an author for doing so. The tournament pays for concealed weaknesses, not confessed ones.

**The record is thin because thinness wins polls.**

---

## 5. The Chosen Game

A natural reading of Sections 3 and 4 is that some external rulebook forces this. The opposite is true, and the truth locates the authority to change it.

### 5.1. What the Directives say, and where

The [ISO/IEC Directives, Part 1](https://jtc1info.org/wp-content/uploads/2023/11/ISO-IEC-Consolidated-JTC-1-Supplement-2023.pdf)<sup>[24]</sup> defines procedures at the level where ISO rules operate: technical committees, subcommittees, and the establishment of working groups. The procedures stop at the working group's boundary. For the subgroups inside one, the Directives specify nothing: no required structure, no chair qualifications, no terms of office, no confirmation procedure, no consensus mechanics, no appeal channel. The silence is deliberate: matters internal to a working group are left to the working group. Silence means neither required nor forbidden.

Where ISO rules do operate, the Directives define consensus as: "General agreement, characterized by the absence of sustained opposition to substantial issues by any important part of the concerned interests and by a process that involves seeking to take into account the views of all parties concerned and to reconcile any conflicting arguments" (Clause 2.5.6)<sup>[24]</sup>. The operative word is reconcile. The ISO ideal, at the levels where it applies, is not that the majority prevails; it is that conflicting arguments are taken into account and sustained opposition is addressed.

### 5.2. What WG21 built

WG21's operating rules are self-authored, recorded in [SD-4](https://isocpp.org/std/standing-documents/sd-4-wg21-practices-and-procedures)<sup>[25]</sup>: subgroup chairs appointed by the convener with no fixed term, consensus thresholds applied by the chairs, and a first-mover rule for competing work - "if a competing alternative does not have a paper, it does not exist and will not block progress of a proposal that we do have before us." Everything SD-4 builds is allowed. WG21 violates nothing.

That is precisely the finding. Because the Directives impose nothing at this level, the tournament of Section 4 - the appointment chain, the indefinite tenure, the single-winner poll mechanics, the first-mover rule - is a construction, authored locally, renewable locally, and repealable locally, tomorrow, with no ISO process whatsoever. A pathology required by a charter is a tragedy. A pathology chosen freely and renewable at will is a decision, renewed every cycle it persists.

### 5.3. Two design philosophies

Set the two systems side by side, each at the level where it operates. The ISO ideal is reconciliation: sustained opposition is engaged until it is addressed or genuinely isolated. The local practice is single-winner competition: opposition is outvoted, and "no consensus for change" preserves whatever advanced first. This is a contrast of design philosophies, and the contracts arc is its cleanest demonstration. National bodies hold no formal power at the working-group level; their authority begins at the subcommittee ballot - structurally the last possible moment in the pipeline. The contracts design advanced for five years through internal polls in which preserving the status quo required no documented engagement ([P4208R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4208r0.pdf)<sup>[19]</sup>); an EWG poll on an enforce-only model was rejected 6-1-3-15-24; and the accumulated objections surfaced at the ballot, where seven national body comments requested removal. The objections surfaced exactly where the structure gave them their first formal channel - exactly late. No rule was broken. The structure performed as designed. The design is the finding.

### 5.4. Downstream of the choice

The symptoms documented elsewhere in the corpus sit downstream of this chosen structure. Twenty-one years after the first networking proposal (N1925, 2005), the standard contains no sockets, no DNS, and no TLS ([P4099R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4099r1.pdf)<sup>[7]</sup>, [P4048R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4048r0.pdf)<sup>[26]</sup>). The C++26 working draft carries a coroutine task type with sixteen open issues documented after design approval ([P4007R3](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4007r3.pdf)<sup>[27]</sup>) and a composition gap that is permanent without a pervasive fix ([P2583R4](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p2583r4.pdf)<sup>[10]</sup>). Twenty-nine implementers gathered at Kona in 2025 - twenty in person, nine remote - and their joint summary records conformance falling behind the standard's growth ([P3962R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p3962r0.pdf)<sup>[20]</sup>). And adoption is not accelerating to compensate: JetBrains' ecosystem survey measured first-year adoption at 12% for C++17, 12% for C++20, and 10% for C++23 ([The C++ Ecosystem in 2023](https://blog.jetbrains.com/clion/2024/01/the-cpp-ecosystem-in-2023/)<sup>[28]</sup>).

**No rule is broken. A choice is renewed.**

---

## 6. Three Rational Responses

Game theory predicts how a tournament responds to a player who publishes evidence anyway. For every other player, engaging the substance is the worst available move: engagement legitimizes the evidence, costs preparation time, and risks losing on the merits in front of the room. The dominant strategies, in order of cost, are three. All three predictions are already confirmed by observation.

### 6.1. Attack the volume

Reframe the evidence as flooding. When quantity becomes the offense, content never has to be read; the act of publishing is converted into the violation, and the conversation moves from "is this correct?" to "is this appropriate?" - a question the room can answer without opening a single paper. The author has been told to stop publishing. The corpus is information-only and requests nothing from anyone's calendar; the volume attack does not engage that fact, because engaging facts is what the attack exists to avoid.

### 6.2. Attack the provenance

Dismiss on authorship. This is the cheapest move available: it requires no reading, no identification of a specific flaw, and no counter-evidence, and procedure absorbs the debate that substance would cost. The author's papers are AI-assisted and disclose it, in the same slot, in every paper. The dismissal converts the disclosure into the charge.

The provenance attack has a measurable signature: the asymmetric bar. The author has watched a single imperfect citation in an AI-assisted paper presented as proof of hallucination, and the hallucination presented as grounds to dismiss the paper entire. The mailing archives contain decades of human-authored papers with citation errors, and the archives also contain their treatment: errata lists, corrected revisions, no inference drawn about the author's capacity to produce valid work. If the concern were quality, the bar would be uniform.

The companion move deserves one dry sentence. The author - a C++ developer of thirty years with libraries in Boost - was asked publicly by a senior committee member whether he had read his own paper. The question is not a question. It converts a conversation about a paper's claims into a conversation about the paper's origin, and the conversion is the point, because the claims are where the engagement would have to happen.

### 6.3. Go silent

When neither attack lands, non-engagement. Silence is free, carries no poll risk, and under a consensus model reads as the absence of support - the system scores an unengaged paper and a refuted paper identically. Silence is also the only one of the three moves that never generates a quotable mistake.

### 6.4. The live confirmation

The predictions are not hypothetical. In sequence: the author has been told he writes too many papers; the author has been told the work is machine slop; the author has been told that the path to regaining the room's trust is to write fewer papers, by hand. Volume, provenance, and a trust frame - three responses, arriving through three channels, none accompanied by the identification of a single defective claim. Meanwhile the benchmarks of Section 3.4 and a working coroutine-native networking implementation - the artifact the universal model has not produced for its claimed domain in five years - remain unengaged.

These are the moves the incentive structure pays for. The same analysis that explains the thin evidence of Section 3 explains the reception of thick evidence: the tournament processes evidence as a threat because, within the tournament, that is exactly what it is.

**A bar that moves with provenance is a gate, not a standard.**

---

## 7. The Diagnosis: A Dead Player

Sections 3 through 6 are measurements. This section is the interpretation, and it borrows its instrument openly: Samo Burja's [Great Founder Theory](https://samoburja.com/gft/)<sup>[29]</sup>. Institutions are built by founders who possess generating principles - the live reasoning that produces the institution's visible forms. While the principles are alive, the institution is a live player: it can respond to novelty by reasoning from first principles, repair its own processes, and tell the difference between a ritual and its purpose. When the principles die, the forms persist. The institution becomes a dead player: it executes scripts, performs the ceremonies the founder designed, and cannot ask what the ceremonies were for. From the outside, the two states look identical for years. The difference appears only when the environment changes.

WG21 performs the ceremonies of evidence: papers, polls, revision numbers, wording review. The ceremonies were generated by principles - field experience before standardization, rationale on the record, mistakes corrected when found. Section 4 documented the incentive structure that now occupies the space where those principles lived. The ceremonies continue on schedule. The committee polls, schedules, publishes, and ships a standard every three years. What it cannot do is respond to novelty, and the last decade supplied three tests: a competing systems language reached production adoption in the committee's core domains, memory safety became a subject of regulatory attention, and the cost of producing rigorous analysis collapsed by orders of magnitude. Each was processed by the existing scripts - scheduling, polling, deferral - as though it were a paper to be slotted rather than an environment to be answered.

WG21 lost its great founder while he is still in the room. The forms of a technical meritocracy persist; what operates inside them is a structure in which standing determines audibility - the room reads the person before it reads the paper.

The people who carry the original principles have said so, on the record. Howard Hinnant, quoted by permission in [P4046R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4046r0.pdf)<sup>[30]</sup>: "I should quit asking: 'Has it been implemented?' The correct question is: What has been the field experience? Is there positive feedback from anyone outside your immediate family or people who could have a perceived conflict of interest (such as employees of your company)? Having your Mom and your direct reports say the proposal is wonderful is nice, but not sufficient." Sean Parent, in the same paper: "Every decision that the standards committee tends to make tends to almost be made in isolation. And they don't then document the rationale for that decision. And so when a similar decision is made, they may come up with a different answer."

The script symptoms accumulate quietly. Poll numbers exist; decision records do not - a vote discarding years of work receives the same ceremony as a one-line tweak. The committee transmits conclusions but not the judgment that generated them; the tacit knowledge lives in veterans and leaves with them, which is the gap [P4046R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4046r0.pdf)<sup>[30]</sup> documents and attempts to close. A direction can be endorsed by overwhelming vote and still not ship, because reaching consensus and honoring consensus are different acts. Adding a feature takes one paper; removing one takes the political will to admit error - a one-direction ratchet, and capacity that is never exercised is indistinguishable from capacity that does not exist. A market library earns each of its users one at a time; a standard library is imposed on all of them at once, so the feedback loop that disciplines the former never reaches the latter.

**The ceremonies survived the principles that generated them.**

---

## 8. What a Live Player Would Ask

The principles are recoverable, because they were never esoteric. They are the questions a live player would ask of any proposal, and the corpus is, among other things, an attempt to ask them from outside. Seven, each one sentence of doctrine and one of consequence.

1. **Evidence of utility is not evidence of need.** A library that thrives on GitHub has proven utility; the question standardization answers is what fails if the standard does not absorb it, and that question is almost never asked.
2. **The coordination test.** `string` and `thread` are coordination problems - every library must agree on them or nothing composes. Big integers and graphs are libraries. The first category needs a committee; the second needs a package manager. Every proposal belongs to one of the two.
3. **The complexity budget is a commons.** Every feature consumes specification pages, implementer hours, and teaching time that belong to everyone. The tournament of Section 4 is the commons tragedy in action - each author grabs budget because unclaimed budget is a rival's opportunity. The real test of a proposal is value per unit of budget consumed.
4. **Stability is the asset; spend it knowingly.** The standard's unique offer - the reason work lands there rather than on GitHub - is that what enters does not churn. Every admission is therefore irreversible in practice, which is an argument for the evidence bar, not against the feature.
5. **Adoption is ratification.** Every standardized design is a hypothesis about what users need. Deployment is the experiment. The committee ships the hypothesis and never runs the experiment - no retrospective, no success criteria, no scheduled look back. [P4047R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4047r0.pdf)<sup>[22]</sup> is what the measurement looks like when someone runs it from outside.
6. **Winning universal models are narrow and emergent.** The abstractions that conquered computing - files, sockets, functions - were extracted from practice, not designed over it. The record's recurring pattern runs the other way: a strong core domain, an adjacent domain claimed without proportional evidence, and the adjacent domain shipping nothing. A structural difficulty that survives three revision cycles is a boundary awaiting a name, not a bug awaiting a fix.
7. **Output is bounded by absorption.** Twenty-nine implementers gathered at Kona 2025 to say that conformance is falling behind the standard's growth ([P3962R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p3962r0.pdf)<sup>[20]</sup>). A live player treats the people who build the compilers as a constraint on output rate. A dead player treats them as an obstacle to schedule.

**Every program is a hypothesis. The committee never runs the experiment.**

---

## 9. Why Papers, Then

In 1946 George Orwell catalogued the motives for writing in an essay whose title this paper borrows ([Why I Write](https://www.orwellfoundation.com/the-orwell-foundation/orwell/essays-and-other-works/why-i-write/)<sup>[31]</sup>). The one operating here is the third on his list, the historical impulse: the "desire to see things as they are, to find out true facts and store them up for the use of posterity."

The strategy follows from the position, so the position is stated plainly. The author returned to the committee after a long absence, during which his reputation was shaped by others. He holds newcomer standing on a body that prices seniority. He cannot win polls: he commands no bloc, chairs no subgroup, and employs no delegation. Sections 4 and 6 describe the game such a participant loses by playing. There is exactly one game the structure cannot take away from any participant, and it is the record.

So: ask for nothing, document everything. Write the retrospectives the committee never writes for itself. Tabulate the claims against the evidence on file when the claims prevailed. Document how the votes distribute, how the framings were chosen, and what was in the room when consequential decisions were made with no evidence and no notes. Score the predictions, including the author's own, and publish the scorecard.

The papers are addressed to the public record because the institution's own record is closed. SD-4 itself states the rule for subgroup records, wikis, and reflectors: "It is not allowed to quote from these publicly" ([SD-4](https://isocpp.org/std/standing-documents/sd-4-wg21-practices-and-procedures)<sup>[25]</sup>). The standard is public; the deliberation that produces it is not, and the public that lives with the output cannot examine the reasoning behind it. Visibility is the precondition of accountability, and the mailing is the one venue where visibility is guaranteed for as long as the standard exists.

Information-only is the mechanism. An info-paper requests no floor time, so it cannot be denied scheduling. It competes with no proposal, so it cannot lose a poll. It asks for nothing, so there is nothing to refuse. Within the tournament, it is the one move that is not a play - and the structure has no procedure against it. Such a paper can be answered in exactly four ways: it can be refuted, which requires engaging the evidence; it can be dismissed, which requires explaining why the committee's own minutes and poll records say something other than what they say; it can be attacked on volume or provenance, which Section 6 documents and which engages nothing; or it can be ignored, and it does not go away, because it sits in the permanent record of ISO/IEC JTC 1/SC 22/WG21, where every future participant, delegate, and historian of the language will find it.

The corpus attacks three structural conditions at once. The information seal: each paper assembles the sealed institution's public traces into a narrative that anyone can verify without being in the room. The evidentiary asymmetry: each paper meets a uniform citation discipline, which makes the absence of that discipline elsewhere visible by contrast rather than by accusation. The social evaluation of persons: a written record is verifiable by readers who have never met the author, so the question "who are you?" stops being a filter. One fact about production deserves a dry sentence: AI assistance lets the author produce papers at this standard, at this rate, for less than the cost of conference attendance - which is why the rational responses of Section 6 target volume and provenance, and never content.

Thirty-four of the thirty-six papers close their disclosure with the same sentence: "This paper asks for nothing." The sentence is the strategy, performed in the front matter.

### 9.1. "Why not a blog?"

The question has been asked, in good faith, and it deserves a complete answer. In its strongest form: a corpus that asks for nothing does not need the mailing; a blog reaches more readers and burdens no one; filing thirty-six papers is itself a claim on the committee's attention, and the venue is the imposition.

The venue is load-bearing, for four reasons.

The subject is the record, so the examination belongs in the record. The corpus audits decisions whose primary sources - the proposals, the poll outcomes, the revision histories - live in the mailing archive. A correction filed in the same archive as the claims it examines travels with them: the reader who finds P2300R10 and the poll records in the document index finds the retrospectives in the same index, under the same numbering scheme, at the same permanence. A correction filed anywhere else is discoverable only by readers who already know to look for it.

The wager requires immutability and dating. Section 11 declares predictions whose value depends on the trail being dated before the outcome and unalterable after it. A mailing paper is numbered, dated, and immutable once published; the method of [P4047R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4047r0.pdf)<sup>[22]</sup> admits only predictions that are dated, public, and falsifiable. A blog can be edited after the fact, moved, or quietly deleted. The C++29 wager cannot be made from a venue whose contents can change after the outcome is known.

The archive reader exists in only one venue. The mailing is preserved by ISO/IEC JTC 1/SC 22/WG21 for the life of the standard and read by every future participant who researches how a feature came to be. The corpus is written for that reader. The median blog does not survive a platform migration, let alone a decade.

The mailing is the institution's public channel. The deliberation behind the standard is sealed; the mailing is the one public record the institution itself maintains. Work addressed to the institution and its users belongs in the channel the institution owns, where it is part of the proceedings rather than commentary upon them.

One property of the suggestion is recorded without attribution: it concedes the content and contests the venue. It asks for the audit to be relocated out of the records of the audited.

### 9.2. The corpus

What follows is the complete published corpus, February through May 2026, grouped by function. One line each.

#### Retrospectives: how the present was decided

| Paper | What it documents |
| :---- | :---------------- |
| [P4094R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4094r1.pdf)<sup>[2]</sup> | The 2014-2020 unification of executors; one hypothetical snippet as its entire code-level evidence base. |
| [P4095R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4095r1.pdf)<sup>[3]</sup> | The 2019 basis-operation pivot, decided under one of two valid framings, the second never analyzed. |
| [P4096R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4096r1.pdf)<sup>[4]</sup> | The P2464R0 diagnosis that set the Networking TS aside, evaluated under the same inherited framing. |
| [P4097R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4097r1.pdf)<sup>[5]</sup> | The October 2021 networking poll: the full tally, voter comments, and the published evidence behind the word "networking". |
| [P4098R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4098r1.pdf)<sup>[6]</sup> | Twenty years of async claims tabulated against the published evidence for each; the networking cells are empty. |
| [P4099R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4099r1.pdf)<sup>[7]</sup> | The four-decision causal chain, 2014-2021, assembled from the five papers above. |
| [P4048R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4048r0.pdf)<sup>[26]</sup> | The twenty-one-year networking gap as a call to action for C++29. |

#### Evidence and methodology: tools the process lacks

| Paper | What it documents |
| :---- | :---------------- |
| [P4047R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4047r0.pdf)<sup>[22]</sup> | Twenty-seven dated, falsifiable predictions scored against the record; eighteen confirmed. |
| [P4137R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4137r0.pdf)<sup>[32]</sup> | A verification-evidence framework for safety profile claims. |
| [P4207R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4207r0.pdf)<sup>[23]</sup> | Adversarial review of a committee paper for under a dollar in fifteen minutes; the methodology. |
| [P4208R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4208r0.pdf)<sup>[19]</sup> | The methodology demonstrated on the contracts proposal: fifteen candidate charges, one survivor. |
| [P4046R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4046r0.pdf)<sup>[30]</sup> | Capturing veteran judgment before it leaves; structured interviews as committee memory. |

#### The `std::execution` record: claims against text

| Paper | What it documents |
| :---- | :---------------- |
| [P2583R4](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p2583r4.pdf)<sup>[10]</sup> | Symmetric transfer absent from all eleven revisions of P2300; O(N) stack growth where C++20 provides O(1); the fix. |
| [P4007R3](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4007r3.pdf)<sup>[27]</sup> | Sixteen open issues in `std::execution::task`, documented after design approval. |
| [P4014R2](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4014r2.pdf)<sup>[33]</sup> | All thirty sender algorithms taught progressively; what the model demands of a beginner. |
| [P4041R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4041r0.pdf)<sup>[11]</sup> | The universality claim audited: twenty-four post-adoption sender items against zero coroutine items; 1,300 stars against 21. |
| [P4089R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4089r1.pdf)<sup>[34]</sup> | Task-type diversity across the ecosystem; what one mandated shape forecloses. |
| [P4090R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4090r1.pdf)<sup>[35]</sup> | A sender I/O stack constructed for comparison, so the comparison cites running code. |
| [P4091R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4091r1.pdf)<sup>[36]</sup> | The error-model mismatch between regular C++ and the sender sub-language. |
| [P4123R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4123r0.pdf)<sup>[37]</sup> | The measured cost of senders for coroutine I/O. |
| [P4124R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4124r0.pdf)<sup>[38]</sup> | Compound I/O results against the three-channel completion model. |
| [P4166R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4166r0.pdf)<sup>[39]</sup> | What frame-visible coroutines offer senders. |
| [P4178R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4178r0.pdf)<sup>[16]</sup> | Eighteen passages of P2300R10 in tension with each other, each with a charitable reading; seven lines against 125. |

#### The constructive program: coroutine-native I/O

| Paper | What it documents |
| :---- | :---------------- |
| [P4003R3](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4003r3.pdf)<sup>[40]</sup> | The IoAwaitable execution model: executor affinity, stop tokens, frame allocation, in three operations. |
| [P4088R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4088r1.pdf)<sup>[15]</sup> | What C++20 coroutines already provide; the benchmark record. |
| [P4092R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4092r1.pdf)<sup>[13]</sup> | Consuming senders from coroutine-native code; the bridge in one direction. |
| [P4093R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4093r1.pdf)<sup>[14]</sup> | Producing senders from coroutine-native code; the bridge in the other. |
| [P4100R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4100r1.pdf)<sup>[41]</sup> | The Network Endeavor: the staged path to standard networking on coroutines for C++29. |
| [P4125R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4125r1.pdf)<sup>[17]</sup> | Field experience at a derivatives exchange; the tail-latency record. |
| [P4126R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4126r1.pdf)<sup>[42]</sup> | One universal continuation model beneath both worlds. |
| [P4127R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4127r0.pdf)<sup>[43]</sup> | The frame-allocator timing problem and its two possible paths. |
| [P4172R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4172r1.pdf)<sup>[44]</sup> | IoAwaitable for byte-oriented I/O; the design rationale and evidence framework. |

#### API design principles

| Paper | What it documents |
| :---- | :---------------- |
| [P4035R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4035r1.pdf)<sup>[45]</sup> | Escape hatches as a design requirement for standard abstractions. |
| [P4036R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4036r0.pdf)<sup>[46]</sup> | Why `span` does not satisfy the buffer-sequence requirement. |

#### Reference and infrastructure

| Paper | What it documents |
| :---- | :---------------- |
| [P4170R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4170r0.pdf)<sup>[47]</sup> | A reader's guide to the May 2026 corpus. |
| [P4182R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4182r1.pdf)<sup>[48]</sup> | A citable inventory of platforms, operating systems, and toolchains, so claims about them resolve. |

The papers are not lobbying. They are measurement - the measurement the incentive structure of Section 4 suppresses, supplied from the one position the structure cannot reach.

**One paper is a complaint; thirty-six are an audit.**

---

## 10. The Shelf: Work Already Written

This paper touches everything and proves little in its own text; the corpus carries the proofs. The corpus, in turn, is not the whole of the work. The supporting body is complete and sits unpublished as of this writing, because the reception documented in Section 6 prices further publication accordingly. The deterrent is acknowledged as effective. The shelf is described here in outline, with no titles and no paper numbers, so the record reflects that it exists.

A complete coroutine-native networking proposal set: thirteen ask-and-rationale pairs over a foundation execution model, from the task type and buffer vocabulary through TCP, DNS, UDP, and TLS, each ask paired with its design rationale, each design carried by a shipping implementation validated on three platforms.

A governance analysis of the incentive structure WG21's self-authored procedures create, examined point by point against the design philosophy of the wider ISO framework - the analysis Section 5 summarizes.

A failure-mode catalog of large-scale standardization: the recurring patterns the decision record exhibits, paired with a corrective the author calls proportional deliberation - scrutiny that scales with the breadth of the claim being advanced. A universal claim earns a universal examination. A narrow claim proceeds on narrow evidence.

Studies of the machinery itself: how votes distribute and flip with room composition; how appointment chains concentrate; where officer confirmation gaps sit; the information seal on deliberation; the one-direction ratchet of feature removal; the selection effects that shape who remains on consensus bodies; the widening interval between a standard's publication and its conformance; and the independent convergence of four unrelated organizations on the same coroutine async pattern for GPU data movement.

**The corpus is the visible fraction.**

---

## 11. The Wager: C++29

The strategy is declared openly, because declaring a strategy in advance is what makes it falsifiable - the same discipline [P4047R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4047r0.pdf)<sup>[22]</sup> applies to everyone else's predictions applies to the author's program. The papers appear to accomplish little today. Their term is C++29.

The wager, stated as a prediction with its falsification criteria attached. C++29 is the cycle in which the committee's answer on networking comes due. If that answer is sender-based, it will arrive into a world that already contains published benchmarks, a working coroutine-native networking implementation on three platforms, and a dated decision-by-decision record of how the universal claim advanced and on what evidence. Any gap between what ships and what the working implementations deliver will be measurable, and the measurement will have a paper trail that predates the outcome - which decisions, taken under which framings, on which evidence, with which warnings on file. The trail was built in real time, before the result was known, which is the only construction that makes it proof rather than hindsight. If instead the shipped answer matches or exceeds the working implementations, the corpus documents that too, and [P4047R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4047r0.pdf)<sup>[22]</sup>'s method scores the author's predictions with the same table it applies to everyone else's. The record exists either way.

The committee's institutional memory is the one Section 7 describes: tallies without rationale, conclusions without the judgment that produced them, veterans who leave and take the reasoning with them. The corpus is that memory, externalized: every claim dated, every source resolvable, every prediction scored on a schedule, readable by a delegate in 2032 who was never in any of the rooms.

There are early signs the record travels. [P4223R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4223r0.pdf)<sup>[49]</sup> (Petersen, May 2026) - a sender-side proposal - introduces a frame-allocator query it describes as "inspired by Vinnie Falco et al.'s work in P4003", adopting into the sender model a finding the coroutine-native corpus published first. SG14, the study group for low latency, games, finance, and embedded systems, advises in [P4029R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4029r0.pdf)<sup>[50]</sup> that "Networking (SG4) should not be built on top of P2300" - the scoped-domain conclusion, now a study group's filed position. And the Direction Group's [P5000R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p5000r0.pdf)<sup>[51]</sup> suggests that C++29 "be considered a 'maintenance release'", citing the implementer testimony of [P3962R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p3962r0.pdf)<sup>[20]</sup> - the absorption constraint of Section 8, now in the committee's own direction papers.

**The corpus is the committee's externalized memory: it forgets nothing and serves no faction.**

---

## 12. The Refounding

Great Founder Theory<sup>[29]</sup> holds that a dead player revives only by refounding: someone re-derives the institution's purpose from first principles and rebuilds practice around it, working from inside the institution, from outside it, or both at once. The author does both.

Inside: attend, file the record, and practice the doctrine of Section 8 in public. The corpus is the inside track - retrospectives, prediction scoring, adversarial review applied first to the author's own papers, evidence standards demonstrated rather than demanded. Reconciliation is restored the same way: by modeling it, in papers that document the strongest version of every position they examine and credit what each design achieves before measuring what it costs.

Outside: working code only. [Capy](https://github.com/cppalliance/capy) and [Corosio](https://github.com/cppalliance/corosio) are headed for Boost, gathering the field experience the standardization record lacks ([P4125R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4125r1.pdf)<sup>[17]</sup>), on the staged path the Network Endeavor describes ([P4100R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4100r1.pdf)<sup>[41]</sup>). Boost is the venue where adoption is earned: every user is a choice, every retention is a measurement, and the feedback loop that Section 7 notes never reaches a standard library is the daily operating condition of a Boost library. The libraries are the experiment the committee never runs, run anyway.

The two tracks reinforce each other. The inside track files what the outside track learns. If the committee takes up coroutine-native networking for C++29, the work arrives mature, deployed, and documented. If it does not, the work exists anyway, where the users are.

**Do not quit the Room. Build.**

---

## 13. What the Author Wants

The second question from Section 2, answered concretely, so that it does not need to be asked again.

For himself: nothing. Not a chair, not floor time, not adoption by acclaim. The closing sentence of this paper's disclosure is literal.

For the institution, the answer is best expressed as the conditions under which this corpus would stop being necessary. They are conditions, on the record; what to do about them is the committee's own business.

1. **A symmetric evidence bar.** One standard of proof, applied to incumbent and challenger alike. The Networking TS was set aside under questions its replacement was never asked ([P4097R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4097r1.pdf)<sup>[5]</sup>, [P4098R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4098r1.pdf)<sup>[6]</sup>). The retrospective half of the corpus exists because that asymmetry currently has no other corrective.
2. **Records that can be examined.** Decision rationale on file - what was decided, on what evidence, against which alternatives, with what dissent - so that a future committee can check a decision against its reasons when conditions change ([P4046R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4046r0.pdf)<sup>[30]</sup>, [P4095R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4095r1.pdf)<sup>[3]</sup>). The decision-archaeology half of the corpus exists because no one writes these records inside.
3. **Output bounded by absorption.** The twenty-nine implementers of [P3962R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p3962r0.pdf)<sup>[20]</sup> operating as a constraint on the rate of standardization, rather than as commentary upon it.
4. **Reconciliation as the operating mode.** The ideal the ISO Directives define where they operate - conflicting arguments taken into account, sustained opposition addressed ([ISO/IEC Directives, Part 1](https://jtc1info.org/wp-content/uploads/2023/11/ISO-IEC-Consolidated-JTC-1-Supplement-2023.pdf)<sup>[24]</sup>, Clause 2.5.6) - practiced at the level where the design work actually happens, where today nothing requires it. Evidence requirements that bind everyone equally convert papers from weapons back into records, and a tournament with a uniform evidence bar stops being a tournament.

Every one of these is already permitted. That is the quiet consequence of Section 5: a rulebook that imposes nothing at this level forbids nothing at this level. No ISO process, no external approval, and no waiting period stands between the current structure and any better one. The authority that built the tournament is local, it is complete, and it can be exercised tomorrow.

If the conditions arrive, the corpus becomes what it is on its face: historical analysis, benchmarks, methodology, and proposals, useful to the readers they serve. If the conditions do not arrive, the corpus keeps doing the one thing the structure cannot do for itself. The rules as written provide no mechanism for self-examination: no decision records to consult, no outcome reviews to schedule, sealed minutes that cannot be quoted. The function does not vanish because no office holds it. It has to live somewhere.

**The papers are the conscience the rules do not permit.**

---

## Acknowledgments

Thanks to Samo Burja for the *Great Founder Theory* framework, which provides the conceptual lens of Sections 7 and 12.

Thanks to Howard Hinnant and Sean Parent, whose recorded reflections - preserved with permission in [P4046R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4046r0.pdf) - carry the generating principles this paper tries to read back into the record.

Thanks to Eric Niebler, Kirk Shoop, Lewis Baker, and their collaborators for [P2300R10](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html), whose achievements in its domains this paper affirms, and whose presence in the standard the author's bridge papers serve.

---

## References

[1] [P2300R10](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html) - "std::execution" (Micha&lstrok; Dominiak, Georgy Evtushenko, Lewis Baker, Lucian Radu Teodorescu, Lee Howes, Kirk Shoop, Michael Garland, Eric Niebler, Bryce Adelstein Lelbach, 2024).

[2] [P4094R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4094r1.pdf) - "The Unification of Executors and P0443" (Vinnie Falco, 2026).

[3] [P4095R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4095r1.pdf) - "The Basis Operation and P1525" (Vinnie Falco, 2026).

[4] [P4096R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4096r1.pdf) - "Coroutine Executors and P2464R0" (Vinnie Falco, 2026).

[5] [P4097R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4097r1.pdf) - "The Networking Claim and P2453R0" (Vinnie Falco, 2026).

[6] [P4098R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4098r1.pdf) - "Async Claims and Evidence" (Vinnie Falco, 2026).

[7] [P4099R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4099r1.pdf) - "The Twenty-One Year Networking Arc" (Vinnie Falco, 2026).

[8] [P2453R0](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2453r0.html) - "2021 October Library Evolution Poll Outcomes" (Bryce Adelstein Lelbach, 2022).

[9] [P0761R2](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0761r2.pdf) - "Executors Design Document" (Jared Hoberock, Michael Garland, Chris Kohlhoff, Chris Mysen, Carter Edwards, Gordon Brown, Michael Wong, 2018).

[10] [P2583R4](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p2583r4.pdf) - "Symmetric Transfer and Sender Composition" (Mungo Gill, Vinnie Falco, 2026).

[11] [P4041R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4041r0.pdf) - "Is `std::execution` a Universal Async Model?" (Vinnie Falco, 2026).

[12] [libunifex issue #586](https://github.com/facebookexperimental/libunifex/issues/586#issuecomment-1845934903) - Ian Petersen, comment of December 7, 2023.

[13] [P4092R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4092r1.pdf) - "Consuming Senders from Coroutine-Native Code" (Vinnie Falco, Steve Gerbino, 2026).

[14] [P4093R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4093r1.pdf) - "Producing Senders from Coroutine-Native Code" (Vinnie Falco, Steve Gerbino, 2026).

[15] [P4088R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4088r1.pdf) - "What C++20 Coroutines Already Buy The Standard" (Vinnie Falco, 2026).

[16] [P4178R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4178r0.pdf) - "Trade-offs in Asynchronous Abstraction Design" (Vinnie Falco, 2026).

[17] [P4125R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4125r1.pdf) - "Coroutine-Native I/O at a Derivatives Exchange" (Mungo Gill, 2026).

[18] [P2900R14](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p2900r14.pdf) - "Contracts for C++" (Joshua Berne, Timur Doumler, Andrzej Krzemie&nacute;ski, 2025).

[19] [P4208R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4208r0.pdf) - "C++ Contracts on Trial - Does P2900 Survive Cross-Examination?" (Claude Opus 4.6, Vinnie Falco, 2026).

[20] [P3962R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p3962r0.pdf) - "Implementation reality of WG21 standardization" (Nina Ranns, Erich Keane, Vlad Serebrennikov, Aaron Ballman, Iain Sandoe, Jonathan Caves, Cameron DaCamara, Gabriel Dos Reis, Gonzalo Brito, Christof Meerwald, Chuanqi Xu, Shafik Yaghmour, Cody Miller, Wyatt Childers, Waffl3x (Alex), Bruno Cardoso Lopes, Hubert Tong, Louis Dionne, 2026).

[21] [Trip Report: Summer ISO C++ Meeting in St. Louis, USA](https://www.think-cell.com/en/career/devblog/trip-report-summer-iso-cpp-meeting-in-st-louis-usa) - Jonathan M&uuml;ller, July 2024.

[22] [P4047R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4047r0.pdf) - "CRYSTAL BALL: Checking Predictions Against the Record" (Vinnie Falco, 2026).

[23] [P4207R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4207r0.pdf) - "Prosecute Your Paper To Improve It" (Vinnie Falco, 2026).

[24] [ISO/IEC Directives, Part 1 - Consolidated JTC 1 Supplement](https://jtc1info.org/wp-content/uploads/2023/11/ISO-IEC-Consolidated-JTC-1-Supplement-2023.pdf) (ISO/IEC, 2023).

[25] [SD-4: WG21 Practices and Procedures](https://isocpp.org/std/standing-documents/sd-4-wg21-practices-and-procedures) (Standing Document, isocpp.org).

[26] [P4048R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4048r0.pdf) - "Networking for C++29: A Call to Action" (Vinnie Falco, 2026).

[27] [P4007R3](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4007r3.pdf) - "Open Issues in `std::execution::task`" (Vinnie Falco, Mungo Gill, 2026).

[28] [The C++ Ecosystem in 2023](https://blog.jetbrains.com/clion/2024/01/the-cpp-ecosystem-in-2023/) - JetBrains CLion Blog, January 2024.

[29] [Great Founder Theory](https://samoburja.com/gft/) - Samo Burja, 2020.

[30] [P4046R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4046r0.pdf) - "SAGE: Saving All Gathered Expertise" (Vinnie Falco, 2026).

[31] [Why I Write](https://www.orwellfoundation.com/the-orwell-foundation/orwell/essays-and-other-works/why-i-write/) - George Orwell, 1946.

[32] [P4137R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4137r0.pdf) - "Profile Analysis and Verification Evidence (PAVE)" (Vinnie Falco, 2026).

[33] [P4014R2](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4014r2.pdf) - "The Sender Sub-Language For Beginners" (Vinnie Falco, Mungo Gill, 2026).

[34] [P4089R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4089r1.pdf) - "On the Diversity of Coroutine Task Types" (Vinnie Falco, 2026).

[35] [P4090R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4090r1.pdf) - "Sender I/O: A Constructed Comparison" (Vinnie Falco, Steve Gerbino, 2026).

[36] [P4091R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4091r1.pdf) - "Error Models of Regular C++ and the Sender Sub-Language" (Vinnie Falco, 2026).

[37] [P4123R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4123r0.pdf) - "The Cost of Senders for Coroutine I/O" (Vinnie Falco, 2026).

[38] [P4124R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4124r0.pdf) - "Combinators and Compound Results from I/O" (Vinnie Falco, 2026).

[39] [P4166R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4166r0.pdf) - "Benefits of Frame-Visible Coroutines for Senders" (Vinnie Falco, 2026).

[40] [P4003R3](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4003r3.pdf) - "A Minimal Coroutine Execution Model" (Vinnie Falco, Steve Gerbino, Mungo Gill, 2026).

[41] [P4100R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4100r1.pdf) - "Coroutine-Native I/O for C++29 (The Network Endeavor)" (Vinnie Falco, Steve Gerbino, Michael Vandeberg, 2026).

[42] [P4126R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4126r1.pdf) - "A Universal Continuation Model" (Vinnie Falco, Klemens Morgenstern, 2026).

[43] [P4127R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4127r0.pdf) - "The Coroutine Frame Allocator Timing Problem" (Vinnie Falco, 2026).

[44] [P4172R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4172r1.pdf) - "IoAwaitable for Coroutine-Native Byte-Oriented I/O" (Vinnie Falco, Steve Gerbino, Mungo Gill, 2026).

[45] [P4035R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4035r1.pdf) - "The Need for Escape Hatches" (Vinnie Falco, 2026).

[46] [P4036R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4036r0.pdf) - "Why Span Is Not Enough" (Vinnie Falco, 2026).

[47] [P4170R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4170r0.pdf) - "A Reader's Guide to the May 2026 Mailing" (Vinnie Falco, 2026).

[48] [P4182R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4182r1.pdf) - "A Citable Inventory of Platforms, Operating Systems, and Compiler Toolchains" (Mungo Gill, 2026).

[49] [P4223R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4223r0.pdf) - "Towards Senders in Interfaces" (Ian Petersen, 2026).

[50] [P4029R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4029r0.pdf) - "The SG14 Priority List for C++29/32" (Michael Wong, 2026).

[51] [P5000R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p5000r0.pdf) - "Direction for ISO C++29" (Daveed Vandevoorde, Jeff Garland, Paul E. McKenney, Roger Orr, Bjarne Stroustrup, Michael Wong, 2026).
