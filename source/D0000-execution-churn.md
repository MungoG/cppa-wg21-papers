---
title: "The Velocity of Change in `std::execution`" document: D0000R0 date: 2026-02-14 reply-to:
  - "Vinnie Falco <vinnie.falco@gmail.com>" audience: All of WG21
---

## Abstract

Since `std::execution` was approved for the C++26 working draft at Tokyo in March 2024, the committee has processed 31 papers, 11 LWG defects, and 2 national body comments - 44 items modifying a single feature in 22 months. The rate of change has accelerated over that period, not slowed. The subjects of these papers span removals, major design reworks, wording corrections, missing functionality, and safety defects including two Priority 1 issues. This paper presents a systematic survey of the published record - WG21 paper mailings, the LWG issues list, and national body ballot comments - and asks what the trajectory tells us about the feature's readiness for ABI freeze.

---

## 1. The Data

All data in this paper is gathered from the published WG21 paper mailings (open-std.org), the LWG issues list (cplusplus.github.io/LWG), and the C++26 national body ballot comments (github.com/cplusplus/nbballot). The survey identifies every WG21 paper, LWG issue, and NB comment that fixes, reworks, removes, or completes missing functionality in `std::execution` after its approval for C++26. Papers that extend the framework into new domains (networking, for example) are excluded.

### 1.1 Items by Meeting Period

The committee's work on `std::execution` falls naturally into five periods aligned with the plenary meetings since Tokyo:

| Period                          | Months | Removals | Reworks | Wording | Missing | LWG | Total |
|---------------------------------|:------:|:--------:|:-------:|:-------:|:-------:|:---:|:-----:|
| Pre-Wroclaw (Mar-Oct 2024)      |    8   |    1     |    5    |    0    |    1    |  0  |     7 |
| Pre-Hagenberg (Nov 2024-Feb 25) |    4   |    1     |    0    |    2    |    2    |  3  |     8 |
| Pre-Sofia (Mar-Jun 2025)        |    4   |    0     |    2    |    0    |    7    |  1  |    10 |
| Pre-Kona (Jul-Nov 2025)         |    5   |    0     |    3    |    3    |    1    |  7  |    14 |
| Pre-London (Dec 2025-Feb 2026)  |    3   |    0     |    2    |    1    |    0    |  0  |     3 |
| **Total**                       | **24** |  **2**   | **12**  |  **6**  | **11**  |**11**| **42** |

Two NB comments on allocator support (US 253 and US 255) bring the total to 44 items.

### 1.2 Items per Period - Bar Chart

```
  Items
   14  |                                     ##############
       |                                     ##############
   12  |                                     ##############
       |                                     ##############
   10  |                        ##########   ##############
       |                        ##########   ##############
    8  |           ########     ##########   ##############
       |           ########     ##########   ##############
    7  | #######   ########     ##########   ##############
       | #######   ########     ##########   ##############
    5  | #######   ########     ##########   ##############
       | #######   ########     ##########   ##############
    3  | #######   ########     ##########   ##############   ###
       | #######   ########     ##########   ##############   ###
    1  | #######   ########     ##########   ##############   ###
       +---------|----------|-----------|------------|----------->
        Pre-      Pre-        Pre-        Pre-         Pre-
        Wroclaw   Hagenberg   Sofia       Kona         London
        (8 mo)    (4 mo)      (4 mo)      (5 mo)       (3 mo)*

   * Pre-London period is incomplete (3 of ~6 months elapsed)
```

### 1.3 Cumulative Items Over Time

```
  Cumulative
   44  |                                                       oo (+ 2 NB)
   42  |                                                      o
       |
   39  |                                            o--------o
       |                                          /
       |                                        /
       |                                      /
       |                                    /
   25  |                          o--------o
       |                        /
       |                      /
       |                    /
   15  |          o--------o
       |        /
       |      /
    7  | o---o
       |
    0  +-----+----------+---------+----------+---------+------->
         Mar    Nov        Mar       Jul        Dec       Feb
         2024   2024       2025      2025       2025      2026
         Tokyo  Wroclaw    Hagenberg Sofia/Kona London
```

### 1.4 Rate of Change (Items per Month)

Normalizing by period duration reveals the acceleration:

| Period                          | Items | Months | Items/Month |
|---------------------------------|:-----:|:------:|:-----------:|
| Pre-Wroclaw (Mar-Oct 2024)      |     7 |      8 |        0.88 |
| Pre-Hagenberg (Nov 2024-Feb 25) |     8 |      4 |        2.00 |
| Pre-Sofia (Mar-Jun 2025)        |    10 |      4 |        2.50 |
| Pre-Kona (Jul-Nov 2025)         |    14 |      5 |        2.80 |
| Pre-London (Dec 2025-Feb 2026)  |     3 |      3 |        1.00 |

```
  Items/Month
   3.0 |                                  o
       |                        o--------/
   2.5 |                      /
       |            o--------/
   2.0 |          /
       |        /
   1.5 |      /
       |    /                                          ?
   1.0 |  /                                           o
       | o                                          /
   0.5 |                                          /
       |
   0.0 +----+--------+--------+--------+--------+-------->
        Mar   Nov      Mar      Jul      Dec      Feb
        2024  2024     2025     2025     2025     2026
```

The rate rose steadily from 0.88 items/month to 2.80 items/month over the first four periods. The Pre-London period shows 1.0 items/month, but that period is only 3 months old at the time of writing and has not yet reached its meeting. Prior periods accumulated the bulk of their items in the final months before the meeting deadline. Absent evidence of a structural change, the decline is likely an artifact of incomplete data collection.

---

## 2. What the Papers Address

The 44 items are not minor editorial fixes. They span every category of change:

### 2.1 Removals (2 items)

Two features were removed entirely after approval:

- `ensure_started` and `start_detached` - dynamically allocating with no allocator customization, breaking structured concurrency ([P3187R1](https://wg21.link/p3187r1))
- `split` - removed due to incorrect description and problematic semantics ([P3682R0](https://wg21.link/p3682r0))

### 2.2 Major Design Reworks (12 items)

Twelve papers rework fundamental aspects of the approved design:

- `tag_invoke` replaced with member customization points ([P2855R1](https://wg21.link/p2855r1)) - a breaking API change
- Sender algorithm customization rewritten three times: [P2999R3](https://wg21.link/p2999r3), then [P3303R1](https://wg21.link/p3303r1) to fix missing wording from P2999, then [P3718R0](https://wg21.link/p3718r0) for further fixes, then [P3826R3](https://wg21.link/p3826r3) whose title evolved from "Defer...to C++29" to "Fix or Remove..." to "Fix..."
- The `on` algorithm renamed to `starts_on`, `transfer` renamed to `continues_on` because usage revealed a gap between expectations and behavior ([P3175R3](https://wg21.link/p3175r3))
- `get_completion_signatures` reworked from member function to static constexpr function template ([P3557R3](https://wg21.link/p3557r3))
- `task_scheduler` does not parallelize bulk work, requiring a new paper to fix ([P3927R0](https://wg21.link/p3927r0))

### 2.3 Missing Functionality (11 items)

Eleven papers add functionality that was absent from the approved design:

- `system_context` and `system_scheduler` - a basic execution context needed to run code at all ([P2079R10](https://wg21.link/p2079r10))
- `async_scope` - the abstraction needed for safe non-sequential concurrency, replacing the removed `ensure_started`/`start_detached` ([P3149R11](https://wg21.link/p3149r11))
- `task` - the coroutine type that users need to use the framework ([P3552R3](https://wg21.link/p3552r3)), adopted with 29 abstentions and 11 against (77-11-29)
- `prop` and `env` class templates for creating execution environments ([P3325R5](https://wg21.link/p3325r5))
- `write_env` and `unstoppable` sender adaptors ([P3284R4](https://wg21.link/p3284r4))
- Early diagnostics for sender expressions, moving diagnosis from connection time to construction time ([P3164R4](https://wg21.link/p3164r4))

### 2.4 LWG Defects (11 items)

Eleven LWG issues have been filed, including two at Priority 1:

- **LWG 4368** (Priority 1): dangling-reference vulnerability in `transform_sender` - returns xvalue to a dead temporary, potential undefined behavior
- **LWG 4206** (Priority 1): `connect_result_t` unconstrained, causing hard errors instead of SFINAE-friendly failures
- **LWG 4215**: `run_loop::finish` should be `noexcept` - throwing causes `sync_wait` to hang forever
- **LWG 4190**: `completion-signatures-for` specification is recursive - a circular dependency that cannot be satisfied
- **LWG 4356**: `connect()` should use `get_allocator(get_env(rcvr))` - directly relevant to the allocator sequencing gap

### 2.5 NB Comments (2 items)

Two US national body comments on allocator support remain without wording:

- **US 255** (LWG4335): use allocator from receiver's environment
- **US 253** (LWG4333): allow use of arbitrary allocators for the coroutine frame

---

## 3. Comparison with `<ranges>`

`<ranges>` is the closest precedent for a large library feature adopted into the standard. After its adoption for C++20, `<ranges>` accumulated roughly 30 LWG issues in its first two years, most at Priority 2-3.

`std::execution` has accumulated 11 LWG issues in less time, including two Priority 1 safety defects affecting core mechanisms (`connect` and `transform_sender`) that most sender/receiver programs exercise. The defect count may be comparable; the severity is not.

Beyond defects, `std::execution` has required 31 papers to fix, rework, or complete - a volume that `<ranges>` did not require in a comparable period.

---

## 4. The Churn Is Accelerating

The data supports three observations:

1. **The rate of change is increasing, not decreasing.** Items per month rose from 0.88 to 2.80 over the first four complete periods. A feature approaching stability would show the opposite trend.

2. **The subjects are not converging.** Early periods focused on API reworks (replacing `tag_invoke`, renaming algorithms). Later periods introduced new categories: safety defects (Priority 1 LWG issues in mid-2025), allocator concerns (NB comments at Kona in late 2025), and task-type design changes (D3980R0 in January 2026 changing the allocator model relative to the text adopted at Sofia in June 2025). The design surface under active modification is expanding, not contracting.

3. **The severity has not decreased.** The two Priority 1 defects - a dangling-reference vulnerability and an unconstrained alias causing hard errors - were filed in the Pre-Kona period, 16 months after approval. Priority 1 defects appearing more than a year after approval suggest that review has not yet reached the parts of the design where the most serious problems live.

---

## 5. What the Trajectory Implies

A feature that is ready for ABI freeze exhibits a recognizable pattern: the rate of change declines, the severity of discovered issues decreases, and the subjects of remaining work converge toward editorial polish. `std::execution` exhibits the opposite pattern on all three measures.

The question is not whether `std::execution` brings value - it does. The question is whether the committee is confident that the current API surface is stable enough to freeze. Forty-four modifications in 22 months, with the rate accelerating, suggests the design has not yet reached that point.

---

## 6. Conclusion

The evidence does not support the conclusion that `std::execution` has stabilized. The committee may wish to consider whether the current trajectory - accelerating churn, expanding scope of modifications, and undiminished severity of discovered defects - is consistent with freezing the ABI in C++26, or whether allowing more time for the design to converge would produce a stronger, more stable result.

---

## Appendix A - Complete Item Catalogue

The following tables list every item identified in the survey, organized by category.

### Removals

| Paper   | Title                                                   | Date       | Status  |
|---------|---------------------------------------------------------|------------|---------|
| P3187R1 | Remove `ensure_started` and `start_detached` from P2300 | 2024-10-15 | Adopted |
| P3682R0 | Remove `std::execution::split`                          | 2025-02-04 | Adopted |

### Major Design Reworks

| Paper   | Title                                                     | Date       | Status      |
|---------|-----------------------------------------------------------|------------|-------------|
| P2855R1 | Member customization points for Senders and Receivers     | 2024-03-18 | Adopted     |
| P2999R3 | Sender Algorithm Customization                            | 2024-04-16 | Adopted     |
| P3303R1 | Fixing Lazy Sender Algorithm Customization                | 2024-10-15 | Adopted     |
| P3175R3 | Reconsidering the `std::execution::on` algorithm          | 2024-10-15 | Adopted     |
| P3557R3 | High-Quality Sender Diagnostics with Constexpr Exceptions | 2025-06-10 | Adopted     |
| P3570R2 | Optional variants in sender/receiver                      | 2025-06-14 | Adopted     |
| P3718R0 | Fixing Lazy Sender Algorithm Customization, Again         | 2025-07-24 | In Progress |
| P3826R3 | Fix Sender Algorithm Customization                        | 2025-11-14 | In Progress |
| P3927R0 | `task_scheduler` Support for Parallel Bulk Execution      | 2026-01-17 | In Progress |

### Wording Fixes

| Paper   | Title                                                      | Date       | Status      |
|---------|------------------------------------------------------------|------------|-------------|
| P3396R1 | `std::execution` wording fixes                             | 2024-11-22 | Adopted     |
| P3388R3 | When Do You Know `connect` Does Not Throw?                 | 2025-02-14 | Adopted     |
| P3914R0 | Assorted NB comment resolutions for Kona 2025              | 2025-11-07 | In Progress |
| P3887R1 | Make `when_all` a Ronseal Algorithm                        | 2025-11-07 | Adopted     |
| P3940R0 | Rename concept tags for C++26: `sender_t` to `sender_tag` | 2025-12-15 | In Progress |

### Missing Functionality

| Paper    | Title                                                        | Date       | Status      |
|----------|--------------------------------------------------------------|------------|-------------|
| P3425R1  | Reducing operation-state sizes for subobject child operations | 2024-11-19 | Approved    |
| P3284R4  | `write_env` and `unstoppable` Sender Adaptors                | 2025-02-14 | Adopted     |
| P3685R0  | Rename `async_scope_token`                                   | 2025-04-09 | Adopted     |
| P3706R0  | Rename `join` and `nest` in async scope proposal             | 2025-04-09 | Adopted     |
| P3325R5  | A Utility for Creating Execution Environments                | 2025-05-22 | Adopted     |
| P2079R10 | Parallel scheduler                                           | 2025-06-02 | Adopted     |
| P3149R11 | `async_scope`                                                | 2025-06-02 | Adopted     |
| P3164R4  | Early Diagnostics for Sender Expressions                     | 2025-06-02 | Adopted     |
| P3552R3  | Add a Coroutine Task Type                                    | 2025-06-20 | Adopted     |
| P3815R1  | Add `scope_association` concept to P3149                     | 2025-09-12 | Adopted     |

### Post-Adoption Issues

| Paper   | Title                                               | Date       | Status      |
|---------|-----------------------------------------------------|------------|-------------|
| P3433R1 | Allocator Support for Operation States              | 2024-10-17 | Adopted     |
| P3481R5 | `std::execution::bulk()` issues                     | 2024-10-17 | Adopted     |
| P3796R1 | Coroutine Task Issues                               | 2025-07-24 | In Progress |
| P3801R0 | Concerns about the design of `std::execution::task` | 2025-07-24 | In Progress |
| D3980R0 | Task's Allocator Use                                | 2026-01-25 | In Progress |

### LWG Issues

| Issue    | Title                                                            | Date       | Priority/Status     |
|----------|------------------------------------------------------------------|------------|---------------------|
| LWG 4190 | `completion-signatures-for` specification is recursive           | 2025-01-02 | Open                |
| LWG 4206 | `connect_result_t` should be constrained with `sender_to`        | 2025-02-04 | Open - Priority 1   |
| LWG 4215 | `run_loop::finish` should be `noexcept`                          | 2025-02-13 | Open                |
| LWG 4260 | Query objects must be default constructible                      | 2025-05-07 | Resolved            |
| LWG 4355 | `connect-awaitable()` should mandate receiver completion-signals | 2025-08-27 | Open                |
| LWG 4356 | `connect()` should use `get_allocator(get_env(rcvr))`            | 2025-08-27 | Open                |
| LWG 4358 | `[exec.as.awaitable]` uses Preconditions when should be constraint | 2025-08-27 | Resolved          |
| LWG 4360 | `awaitable-sender` concept should qualify `awaitable-receiver`   | 2025-08-27 | Resolved            |
| LWG 4368 | Potential dangling reference from `transform_sender`             | 2025-08-31 | Open - Priority 1   |
| LWG 4369 | `check-types` for `upon_error` and `upon_stopped` is wrong       | 2025-08-31 | Resolved            |
| LWG 4336 | `bulk` vs. `task_scheduler`                                      | 2025-10-23 | Open                |

### Allocator-Related NB Comments

| NB Comment | Title                                                | Status        |
|------------|------------------------------------------------------|---------------|
| US 255     | Use allocator from receiver's environment            | Needs wording |
| US 253     | Allow use of arbitrary allocators for coroutine frame | Needs wording |

---

## References

- [P2079R10](https://wg21.link/p2079r10) Lee Howes. "Parallel scheduler." 2025-06-02.
- [P2855R1](https://wg21.link/p2855r1) Ville Voutilainen. "Member customization points for Senders and Receivers." 2024-03-18.
- [P2999R3](https://wg21.link/p2999r3) Eric Niebler. "Sender Algorithm Customization." 2024-04-16.
- [P3149R11](https://wg21.link/p3149r11) Ian Petersen, Jessica Wong, Kirk Shoop, et al. "async_scope." 2025-06-02.
- [P3164R4](https://wg21.link/p3164r4) Eric Niebler. "Early Diagnostics for Sender Expressions." 2025-06-02.
- [P3175R3](https://wg21.link/p3175r3) Eric Niebler. "Reconsidering the std::execution::on algorithm." 2024-10-15.
- [P3187R1](https://wg21.link/p3187r1) Lewis Baker, Eric Niebler. "Remove ensure_started and start_detached from P2300." 2024-10-15.
- [P3284R4](https://wg21.link/p3284r4) Eric Niebler. "write_env and unstoppable Sender Adaptors." 2025-02-14.
- [P3303R1](https://wg21.link/p3303r1) Eric Niebler. "Fixing Lazy Sender Algorithm Customization." 2024-10-15.
- [P3325R5](https://wg21.link/p3325r5) Eric Niebler. "A Utility for Creating Execution Environments." 2025-05-22.
- [P3388R3](https://wg21.link/p3388r3) Ville Voutilainen. "When Do You Know connect Doesn't Throw?" 2025-02-14.
- [P3396R1](https://wg21.link/p3396r1) Eric Niebler. "std::execution wording fixes." 2024-11-22.
- [P3433R1](https://wg21.link/p3433r1) Dietmar Kuhl. "Allocator Support for Operation States." 2024-10-17.
- [P3481R5](https://wg21.link/p3481r5) Lucian Radu Teodorescu, Lewis Baker, Ruslan Arutyunyan. "std::execution::bulk() issues." 2024-10-17.
- [P3552R3](https://wg21.link/p3552r3) Dietmar Kuhl, Maikel Nadolski. "Add a Coroutine Task Type." 2025-06-20.
- [P3557R3](https://wg21.link/p3557r3) Eric Niebler. "High-Quality Sender Diagnostics with Constexpr Exceptions." 2025-06-10.
- [P3570R2](https://wg21.link/p3570r2) Lewis Baker. "Optional variants in sender/receiver." 2025-06-14.
- [P3682R0](https://wg21.link/p3682r0) Eric Niebler. "Remove std::execution::split." 2025-02-04.
- [P3685R0](https://wg21.link/p3685r0) Ian Petersen, Jessica Wong. "Rename async_scope_token." 2025-04-09.
- [P3706R0](https://wg21.link/p3706r0) Ian Petersen, Jessica Wong. "Rename join and nest in async scope proposal." 2025-04-09.
- [P3718R0](https://wg21.link/p3718r0) Eric Niebler. "Fixing Lazy Sender Algorithm Customization, Again." 2025-07-24.
- [P3796R1](https://wg21.link/p3796r1) Dietmar Kuhl. "Coroutine Task Issues." 2025-07-24.
- [P3801R0](https://wg21.link/p3801r0) Jonathan Wakely. "Concerns about the design of std::execution::task." 2025-07-24.
- [P3815R1](https://wg21.link/p3815r1) Jessica Wong, Ian Petersen. "Add scope_association concept to P3149." 2025-09-12.
- [P3826R3](https://wg21.link/p3826r3) Eric Niebler. "Fix Sender Algorithm Customization." 2025-11-14.
- [P3887R1](https://wg21.link/p3887r1) Robert Leahy. "Make when_all a Ronseal Algorithm." 2025-11-07.
- [P3914R0](https://wg21.link/p3914r0) Various. "Assorted NB comment resolutions for Kona 2025." 2025-11-07.
- [P3927R0](https://wg21.link/p3927r0) Eric Niebler. "task_scheduler Support for Parallel Bulk Execution." 2026-01-17.
- [P3940R0](https://wg21.link/p3940r0) Arthur O'Dwyer, Yi'an Ye. "Rename concept tags for C++26: sender_t to sender_tag." 2025-12-15.
- [D3980R0](https://isocpp.org/files/papers/D3980R0.html) Dietmar Kuhl. "Task's Allocator Use." 2026-01-25.
