---
title: "A Behavioral Detection Model for Unchecked Institutional Proposals"
document: P4196R0
date: 2026-08-11
intent: info
audience: WG21
reply-to:
  - "Vinnie Falco <vinnie.falco@gmail.com>"
---

## Abstract

[P4195R0](https://wg21.link/p4195r0) identifies the incentive structures that [SD-4](https://isocpp.org/std/standing-documents/sd-4-wg21-practices-and-procedures)'s consensus mechanism creates for proposal authors. This companion paper derives observable behavioral profiles from those structures and provides a detection criteria table with falsification conditions that can be applied to the documented record of any proposal's passage through WG21. The distinguishing characteristic is the author's relationship to feedback: the system works when feedback modifies the design, and fails when the author's institutional position allows feedback to be neutralized instead. The model identifies three author profiles, describes what structural conditions enable unchecked institutional behavior, and provides a diagnostic checklist that distinguishes normal procedural fluency from behavior that exceeds norms the system has no mechanism to enforce.

## Revision History

### R0

- Initial version.

## 1. The Model

P4195R0 analyzes the game that SD-4's rules produce. Three mechanisms drive outcomes:

1. Consensus is chair judgment, not a formula applied to poll numbers, so the author's optimization target is the chair's willingness to advance;
2. Polls function as state transitions whose reversal cost grows monotonically, creating path dependence that protects early decisions regardless of their quality; and
3. The institutional record preserves the author's papers but not a minutes register of opposing arguments, so the winning side's case survives as a durable artifact while the losing side's case collapses to a vote tally.

These mechanisms interact with procedural fluency, the ability to convert technical merit into institutional action, which simultaneously increases the probability of adoption and decreases its cost. This advantage compounds for funded, repeat players. The result is an advocacy equilibrium in which many motivated advocates cross-examining each other under capable chair judgment approximate the best design, but which fails when review costs are high, opposition is unfunded, and the record preserves only one side of the argument.

Three behavioral profiles cover the range of how authors operate within this system.

### Profile 1: New Author (Technical Correctness Only)

An individual who:

- Has technical merit but limited institutional resources,
- Treats feedback as substantive and engages it at face value, and
- Lacks procedural fluency to convert merit into institutional action

This author engages substantively with objections, welcomes comparison with competing designs, openly discusses weaknesses, and does not think about burden-of-proof management, or about direction polls as state transitions. The author may produce excellent work that cannot advance because it was submitted to the wrong group, lacks required motivation, or arrives after scheduling decisions. Technical merit is necessary but not sufficient.

### Profile 2: Senior Author (Procedural Fluency)

An individual who:

- Has deep technical expertise and accumulated procedural fluency,
- Treats feedback as strategic input and adjusts their design to build consensus, and
- Operates within institutional norms

This author revises tactically to move experts from SA to WA/N, seeks direction polls deliberately, builds coalitions through reciprocal accommodation, and provides the chair with tractable decisions. The author distinguishes between competitors worth accommodating and those worth outlasting. This is the "ideal author" of P4195R0 - the profile the system is designed to reward.

### Profile 3: Unchecked Institutional Author

An individual who:

- Has institutional backing sufficient to sustain a multi-year campaign,
- Treats feedback as adversarial rather than informative, and
- Directs procedural fluency toward protecting the design, not refining it

The Profile 2 author lets feedback change the design, controlling how much it changes. The Profile 3 author changes the institutional environment to protect the design from feedback. This is the distinguishing characteristic, and it is observable: the C2 response to an objection is technical (the design moves), while the C3 response is political (the environment moves). Objections are decomposed into sub-issues without revisiting the premise, competitors are denied agenda time and discussion polls, repeated objections are characterized as "already addressed" even when the core concern was never answered, and persistent opposition is moralized as blocking progress. Whether these moves reflect conviction or calculation is outside the model's scope; the behavioral pattern is the same.

---

## 2. What the Incentive Structure Produces

The following ten moves are what an unchecked institutional author does when the incentives P4195R0 identifies are unconstrained by committee norms.

1. **Capture the consensus-determination function itself.** Consensus is the chair's call. The author whose institution pays the chair controls the call. Technical objections stop mattering because they never change the outcome.

2. **Manufacture the appearance of independent agreement.** The system approximates the best design by counting independent advocates. Create duplicate votes through employer-bloc polling where employees feel pressure to conform. These satisfy the mechanism on paper while corrupting what it measures. The polls count hands and cannot verify independence.

3. **Control the routing topology.** Route your paper to the group where your coalition is strongest. Influence which group gets jurisdiction over the design space before the competitor arrives. Dissolve the study group if the opportunity permits. The competitor prepared to fight in Room A finds the battle has moved to Room B.

4. **Create prerequisites your competitors must overcome.** Author the foundational infrastructure that competing designs must build on. Your competitor's proposal is now dependent on your machinery. You control what the machinery requires, how it evolves, and what architectures it permits.

5. **Seal the pipeline asymmetrically.** Secure the fastest ship vehicle for your proposal while steering competitors toward slower tracks. Your feature ships in C++29. The competition is still writing position papers for C++32.

6. **Raise review costs structurally.** Present large papers not in the review system and made available days before a meeting, containing significant wording and cross-cutting dependencies such that thorough review exceeds the effort unpaid reviewers are willing to exert. Too long to evaluate, too late to prepare for, too informal to cite against. The system under-produces scrutiny on the proposals that need it most.

7. **Decompose architectural challenges so the premise never receives a direct vote.** "The architecture is wrong" becomes "concerns about X," then "concerns about Y," then individual fixable issues. When the chair is aligned with the author, the question "Should this architecture exist?" is never polled.

8. **Shift the burden of proof deliberately.** First, label the competing design as an alternative. Then characterize the alternative as an objection. Then treat the objection as reopening a settled question. The institutional framing changes while the technical content stays the same.

9. **Script the historical record.** Report favorable polls in your paper's history. Omit unfavorable ones. When objections arise, move to a poll, record the tally, and change the subject. The substance of the objection never enters the written record. Twenty years later, your 40 papers are the institutional memory. The opposition is "SA=4" in the minutes.

10. **Moralize continued opposition.** Characterize opposition as blocking progress, refusing to accept the committee's decision, harming C++. Make the social cost of dissent exceed the technical cost of a bad decision. Whether the author acts from conviction or from institutional strategy, the observable behavior is the same: technical disagreement is reframed as a character deficiency.

These ten moves are what the incentive structure produces when institutional backing exceeds what the system's safeguards can detect and correct for.

---

## 3. Detection Criteria Table

The following table turns the three profiles into observable behaviors. Column C1 scores Profile 1, column C2 scores Profile 2, and column C3 scores Profile 3. For each criterion, it describes how each profile would characteristically act - providing a diagnostic checklist that can be applied to the documented record of any proposal's passage through WG21.

| Detection Criterion | C1: New Author (Technical Correctness Only) | C2: Senior Author (Procedural Fluency) | C3: Unchecked Institutional Author |
|---|---|---|---|
| Response to architectural objections | Engages substantively; may redesign if convinced | Acknowledges objection, revises tactically to move expert from SA to WA/N | Decomposes "the architecture is wrong" into sub-issues, addresses each narrowly, never revisits the premise |
| Competing designs | Welcomes comparison; may not know how to get a joint discussion scheduled | Distinguishes between competitors worth accommodating and those worth outlasting | Actively denies competitors agenda time, discussion polls, or framing opportunities |
| Early direction polls | Does not seek them; may not know they exist as a strategic tool | Seeks them deliberately; understands their option value | Seeks them aggressively and uses accumulated state to block late-arriving alternatives |
| Treatment of minority objections | Takes them seriously regardless of vote outcome | Addresses enough to satisfy the chair; stops when consensus is achievable | Characterizes repeated objections as "already answered" or "no new information" without addressing the core concern |
| Written record behavior | Produces a paper; may not produce rebuttals | Produces papers and responses; creates favorable institutional memory naturally | Produces extensive artifacts for own position; avoids faithful restatement of the opposition's case |
| Relationship with chair | Minimal; may not understand what the chair needs to declare consensus | Collaborative; provides the chair with a tractable decision | Ensures the chair's path of least resistance is always "advance" |
| Coalition building | Absent or naive; relies on technical merit alone | Strategic; trades concessions with other repeat players | Leverages institutional backing to assemble coalitions; may trade support on unrelated proposals |
| Moralization of opposition | "They raise a good point" | "We've considered that and made changes" | "They are blocking progress" / "They refuse to accept the committee's decision" - opposition is moralized |
| Reaction when pulled back | Confused; may not understand what happened procedurally | Regroups, revises, returns next meeting with a plan | Treats reversal as illegitimate; escalates procedurally; seeks to restore previous state transitions |
| Burden of proof management | Does not think in these terms | Understands that accumulated polls shift the burden onto competitors | Deliberately engineers or reinforces the four-stage linguistic transformation |
| Use of procedural moves | Unaware of most available moves | Knows the full move set; uses it selectively and within norms | Exceeds norms: blocking discussion polls for rivals, controlling poll wording, exploiting scheduling |
| Transparency about design tradeoffs | Openly discusses weaknesses | Discusses tradeoffs selectively; frames them favorably | Minimizes or conceals known weaknesses; frames any admission as already resolved |
| Response to "investigate the objection thoroughly" | Does it, even at high personal cost | Does it if cost-benefit is favorable; skips if objection can be rendered non-dispositive more cheaply | Refuses or performs superficial investigation; the design is correct by prior conviction |
| Behavior between meetings | Works on the paper; may not engage politically | Maintains relationships; builds support informally | Campaigns actively; may lobby chairs, NB contacts, or employers of opposing participants |
| Observable cost structure | High cost, low fluency, low probability of success | Moderate cost, high fluency, high probability of success | Low cost (funded), high fluency (institutional backing), high probability of success, expanded move set (unchecked) |
| What happens if they win | A technically sound feature enters the standard, possibly with rough edges | A refined feature shaped by negotiation; quality correlates with but is not identical to optimality | A feature reflecting the author's original conviction; objections managed, not resolved; correction requires implementer revolt or senior committee member intervention |

The distinguishing signal for C3 is the combination: the author decomposes objections but never answers them at the architectural level, denies competitors procedurally rather than refuting them technically, and moralizes opposition rather than engaging it. Any one of these in isolation is common. All three together, sustained across multiple meetings, is the detection signature.

---

## 4. Falsification Conditions

An evidence item scores C3 only when the C2 explanation is insufficient. The following list defines, for each criterion, what a competent, well-funded, sincere author operating within norms would do (the C2 baseline) and what specific observation exceeds that baseline (the falsifier). If no falsifier is present, the item scores C2.

**Falsification principle:** The bright line between C2 and C3 is the type of response to feedback. A C2 author responds technically: the design changes. A C3 author responds politically: the institutional environment changes to protect the design. If a reasonable observer could attribute the behavior entirely to procedural competence and strategic design revision - the author adjusting the proposal to build consensus - the item scores C2. A C3 score requires behavior where the author adjusts the environment instead of the design: suppressing legitimate alternatives procedurally, engineering burden-of-proof shifts, or ensuring the chair's path of least resistance is always "advance." These are observable acts, not inferences about motivation.

1. **Response to architectural objections**
   - *C2 baseline:* Responds thoroughly, may disagree after genuine analysis. Volume alone is not diagnostic.
   - *Falsifier:* The study group chair states concerns were not addressed. Multiple independent seniors characterize the response as non-engagement despite its length. The pattern repeats across years without the architectural premise ever being revisited.

2. **Competing designs**
   - *C2 baseline:* Argues own design is superior. May seek favorable scheduling. Does not actively prevent a competing paper from receiving a discussion poll.
   - *Falsifier:* Poll wording embeds the conclusion. Competing designs declared "closed" at subgroup level while a higher group later deadlocks on the same question. Competitors denied comparable scheduled time.

3. **Early direction polls**
   - *C2 baseline:* Seeks direction polls deliberately; cites favorable results to establish priority. Standard committee strategy.
   - *Falsifier:* A direction poll is converted into a permanent "mandate" that forecloses all subsequent deliberation. Omnibus polls bundle unrelated decisions to prevent granular objection.

4. **Treatment of minority objections**
   - *C2 baseline:* Addresses minority objections enough to satisfy the chair. May disagree after genuine engagement. Stops revisiting when consensus is achievable.
   - *Falsifier:* Objections dismissed as "no new information" when the core technical concern was never directly answered in writing. The same dismissal pattern repeats across multiple meetings without the substance of the objection ever being engaged.

5. **Written record behavior**
   - *C2 baseline:* Frames position favorably, cites favorable outcomes. Selective presentation is normal advocacy.
   - *Falsifier:* Specific unfavorable poll results are omitted from self-reported history while favorable results from the same period are reported. The opposition's case is never stated in its strongest form.

6. **Relationship with chair**
   - *C2 baseline:* Good working relationship with the chair. Chair's favorable treatment may reflect genuine assessment.
   - *Falsifier:* The chair co-authors the proposal under their own oversight while receiving undisclosed income from the proposal's institutional sponsor.

7. **Coalition building**
   - *C2 baseline:* Recruits co-authors, assembles broad support. Large co-author lists are standard.
   - *Falsifier:* Internal dissenters are excluded rather than accommodated. The coalition includes undisclosed financial relationships with oversight personnel.

8. **Moralization of opposition**
   - *C2 baseline:* May use sharp language under pressure. Characterizes the argument, not the opponent's conduct.
   - *Falsifier:* The act of submitting an alternative is treated as illegitimate. A competing approach is equated with "halting all forward progress."

9. **Reaction when pulled back**
   - *C2 baseline:* Regroups, revises, returns with a plan. Persistence after rejection is normal and encouraged.
   - *Falsifier:* The unfavorable result is omitted from the paper's history section while only the favorable poll is reported. Committee requirements are overridden rather than satisfied.

10. **Burden of proof management**
    - *C2 baseline:* Cites prior decisions and asks "what's new?" Prevents infinite re-litigation.
    - *Falsifier:* A vote tally is used to dismiss objections that post-date the vote. The four-stage linguistic transformation ("competing design" -> "alternative" -> "objection" -> "reopening settled question") is documented across multiple arcs.

11. **Use of procedural moves**
    - *C2 baseline:* Uses full move set within norms. Short incubation happens under deadline pressure.
    - *Falsifier:* A majority of binding papers polled with under one week's incubation systematically, including self-authored papers. Poll wording drafted privately with leadership while objectors are excluded.

12. **Transparency about design tradeoffs**
    - *C2 baseline:* Frames tradeoffs favorably. Being candid under cross-examination is evidence of integrity.
    - *Falsifier:* Weaknesses are conceded verbally under cross-examination but do not propagate into the written institutional record. Written artifacts omit or neutralize the verbal concession.

13. **Response to "investigate the objection thoroughly"**
    - *C2 baseline:* Investigates when cost-benefit is favorable. May decline if the objection is non-dispositive.
    - *Falsifier:* A strong-consensus recorded committee instruction is overridden without being satisfied or formally reversed.

14. **Behavior between meetings**
    - *C2 baseline:* Maintains relationships, coordinates with co-authors, prepares papers. Employer-funded teams are normal.
    - *Falsifier:* Undisclosed financial relationships with persons exercising oversight authority. Coordinated campaigns designed to present decisions as already made before deliberation occurs.

15. **Observable cost structure**
    - *C2 baseline:* Significant employer backing with funded engineers and coordinated papers. How major facilities get standardized.
    - *Falsifier:* Cost structure includes undisclosed financial relationships with oversight authority AND duplicate vote representation from the same funding source. This weakens the independence of technical review.

16. **What happens if they win**
    - *C2 baseline:* Feature may have rough edges. Author schedules extensions for known gaps. Some dissent persists.
    - *Falsifier:* Co-author dissent, implementer "unusable" finding, major vendor non-implementation, record DIS opposition, AND post-victory acknowledgment that concerns dismissed pre-vote in fact had merit - all simultaneously.

---

## 5. What The Model Cannot Distinguish

The C2/C3 distinction is behavioral, and the tests are bright-line. A C2 author responds to feedback technically: the design changes. A C3 author responds politically: the institutional environment changes to protect the design from feedback. These are different observable behaviors. A C2 author, no matter how capable, well-funded, or convinced, responds by adjusting the design - that is what makes them C2. No amount of skill, resources, or conviction converts a political response into a technical one. The detection criteria in Section 3 and the falsification conditions in Section 4 operationalize this distinction across sixteen criteria.

What the model cannot distinguish is *motivation within C3*. A Profile 3 author who sincerely believes their design is correct and whose political behavior flows from that conviction (the true believer) produces a behavioral record identical to a Profile 3 author who uses institutional position to advance a design for reasons that are not purely technical (the institutional operator). The model does not and cannot determine which. Neither can WG21.

That limitation is irrelevant to the diagnosis. P4195R0's advocacy equilibrium requires many motivated advocates cross-examining each other under capable chair judgment to approximate the best design. C3 behavior breaks this approximation regardless of motivation, because:

- Expert cross-examination is defeated when competitors are denied a hearing
- Chair judgment is captured when advancing becomes the chair's path of least resistance
- The review public-goods problem is exploited when review cost is high and the equilibrium becomes Push/Abstain

The system's only correction mechanisms are implementer revolt (refusing to ship the feature) or a senior committee member absorbing the personal cost of sustained opposition. Both are expensive, unreliable, and activate only after the damage is done.

The Code of Conduct requires an assumption of good faith, but the system has no structural defense against C3 behavior. The model diagnoses the behavior. The motivation is the author's own affair.

---

## 6. Application

This model can be applied to the documented record of any proposal's passage through WG21 by:

1. Collecting evidence items from papers, wiki minutes, reflector posts, and trip reports
2. Scoring each item against the detection criteria table
3. Applying the falsification conditions: an item scores C3 only when the C2 explanation is insufficient
4. Tallying hits per column within each criterion
5. Evaluating the combination signal: all three distinguishing markers (decomposing objections without architectural engagement, denying competitors procedurally, moralizing opposition) present simultaneously across multiple meetings

### Evidence Sources

Not all evidence is equal. The following table classifies source types by what they can establish.

| Source | Classification | Use |
|---|---|---|
| Official minutes and poll records (N-documents) | Primary | Direct evidence of outcomes, vote tallies, and chair determinations |
| Numbered P-papers by principals | Primary | Direct evidence of positions, design rationale, and stated responses to objections |
| SD-4 and standing documents | Primary | Establishes the rules under which behavior is evaluated |
| Reflector posts by participants present in the room | Corroborating | Independent confirmation of what occurred; contemporaneous accounts |
| NB comments filed through the formal process | Corroborating | Independent institutional objections with formal standing |
| Implementation reports from vendors | Corroborating | Independent verification of post-adoption outcomes |
| GitHub issue discussions (e.g. cplusplus/papers) | Corroborating | Contemporaneous records of committee-adjacent deliberation |
| D-papers (drafts without P-numbers) | Indirect | The paper itself is not citable, but minutes recording its presentation are. The existence of large D-papers provided just before meetings is evidence for criterion 6 |
| Trip reports and blog posts by participants | Supporting | Context, pattern recognition, and insight into how participants experienced the process |
| Conference journals and talks | Supporting | Public statements by participants outside the committee record |
| Implementation repository history | Supporting | Timeline of engineering investment and design changes |
| Reflector posts summarizing what others said | Supporting | Hearsay; useful for establishing pattern but not for individual scoring |

### Methodology

A single analyst performs the scoring. Two requirements make the work checkable:

1. **Per-criterion assessment.** Before listing evidence items for a criterion, the analyst states whether C2 does or does not explain the record for that criterion, and why. This forces the analyst to commit to a reading before presenting evidence, making the analytical framework transparent.

2. **Per-item citation.** Every scored item cites the specific source document with URL or P-number and date, states the C2 baseline it was tested against, and explains why C2 was insufficient (for C3 scores) or sufficient (for C2 scores).

A second analyst reading the same sources should be able to verify or dispute each individual score without re-reading the entire record.

### Threshold

A finding exists when any single criterion produces more than 20% C3 hits relative to the total C2 and C3 hits collected for that criterion. This is per-criterion, not aggregate across all 16.

Findings are not equal. The strength of a finding is proportional to the extent that the behavior blocks competitors from receiving a fair hearing or shields the proposal from scrutiny. The analyst's report should make the qualitative weight self-evident from the evidence presented.

The combination signal (all three distinguishing markers present simultaneously across multiple meetings) is a separate, stronger threshold indicating systematic rather than isolated C3 behavior.

### Scope

This model diagnoses process capture. It does not prescribe remedies.

The model does not determine whether a proposal's design is good or bad. A technically excellent design can be adopted through C3 behavior, and a technically poor design can fail despite C2 behavior. The model evaluates the *process* of adoption, not the *quality* of the result. A system that produces good outcomes through captured processes is still captured.

This analysis surfaces structural dynamics that are not discussed in the committee's normal discourse. It is a starting point for conversation. Readers who see a framework should discuss how to formalize one. Readers who want to apply the model to a specific proposal's record should do so. The purpose is to make the invisible visible.
