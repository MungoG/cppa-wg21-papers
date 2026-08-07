# Concept Manifest - A Young Delegate's Notebook

Binding contract for every writing agent. Read this before drafting any chapter.

## How to use this manifest

- Each chapter OWNS the terms listed under it. A term is defined once (bolded, in plain language) in its owning section.
- Never use a term formally before its owning section. If you need a concept owned by a later chapter, paraphrase it in plain words and do not claim the formal definition.
- When you use a term first defined in an earlier chapter, restate its meaning in a short parenthetical on first use in your chapter.
- Section numbers are decimal. Chapter headings are `## N. Title`, sections `### N.M Title`, subsections `#### N.M.K Title`. Headings carry the number, no section symbol. In prose, references use the symbol (§5.2).

## The anatomy bridge (important)

The chapter order introduces papers (Ch 3) and remote participation (Ch 4) before the full structural map (Ch 5). To avoid forward references, Chapter 2 introduces the committee's basic anatomy at a shallow level: WG21, National Body, convener, plenary, subgroup. Chapter 5 owns the deep structural treatment (the specific rooms, study groups, the Direction Group, chair power) and restates each shared term with a parenthetical. Writers in Ch 3-4 use only the shallow Ch 2 meanings.

## Cross-cutting psychological awareness

Woven in, never its own topic, never jargon. Advice TO the reader about their own judgment. Placed in: Ch 1 (responsibility, not a game), Ch 6 (realistic expectations), Ch 7 (don't get addicted; stay skeptical), Ch 8 (consensus is social pressure; demand evidence), Ch 9 (evidence over enthusiasm), Ch 11 (burnout; boundaries), Ch 13 (participation is not impact).

---

## Chapter 1 - The C++ Standard

Arc: enters curious about "what is this thing," leaves feeling the weight and seriousness of it.

Subheading order:
- 1.1 Standardization Is a Responsibility (not a hobby, resume item, or game)
- 1.2 The Delegate's Oath
- 1.3 Every Feature Is Forever (the standard only grows; std::regex; ABI locks mistakes; nothing can be removed)
- 1.4 The People Who Inherit Your Work (sixteen million users)
- 1.5 The Standard Is Not the Language (the document vs compilers vs what people write)
- 1.6 What "The Standard" Actually Is (the working draft; IS, TS, TR)
- 1.7 How to Read the Standard (clauses, normative text vs notes, stable section labels)
- 1.8 From Standard to Compiler (the path to implementation; the three implementers)

Owns:
- **the standard** (§1.5) - the official ISO document that defines what C++ means; not the compilers, not the code people write.
- **the Delegate's Oath** (§1.2) - the vow: do what's best for the language, make no unnecessary proposals, put users first.
- **the working draft** (§1.6) - the living, in-progress text that becomes the next official version of C++.
- **International Standard (IS)** (§1.6) - the finished, published version of C++, like C++23; the binding standard.
- **Technical Specification (TS)** (§1.6) - an optional, experimental document for trying a feature before it enters the standard.
- **Technical Report (TR)** (§1.6) - an older informational document type, now mostly replaced by the TS.
- **normative text** (§1.7) - the parts that state real requirements, as opposed to notes and examples.
- **stable section labels** (§1.7) - bracketed names like [container.requirements] that stay fixed across drafts, unlike section numbers.
- **the three implementers** (§1.8) - GCC, Clang, and MSVC, the three main compiler teams who turn the standard into something usable.

Prerequisites: none (first chapter). Uses "WG21" plainly (from the intro).
Links: isocpp.org, eel.is/c++draft, github.com/cplusplus/draft. Defer the formal wg21.link explanation to Ch 3. Defer "feature-test macro" to Ch 9.

---

## Chapter 2 - Meet the Committee

Arc: enters intimidated ("can I even join?"), leaves feeling the door is open.

Subheading order (as written):
- 2.1 You Don't Need Permission (the barrier is lower than you think; non-technical help is valued)
- 2.2 The Nesting: ISO, JTC1, SC22, WG21
- 2.3 How the Committee Meets (plenary and subgroups; anatomy bridge)
- 2.4 What a National Body Is (one country, one vote)
- 2.5 The Easiest Start: Attending as a Guest (email the convener, ~one week's notice; guest limit)
- 2.6 Getting Further In: INCITS and the Boost Foundation (INCITS open to anyone, ~$800/yr; Boost free)
- 2.7 Getting a Vote: The ISO Global Directory (accreditation)
- 2.8 The Roles You Can Play (note-taker, reviewer, author, implementer, champion)
- 2.9 Your First Day: Orientation and Introductions (Sunday 6pm orientation; newcomer welcome)
- 2.10 Showing Up Is the Secret (eighty percent of success is showing up)
- 2.11 Where Ideas Start: Conferences (C++Now refines ideas, CppCon reaches the community)

Owns:
- **ISO** (§2.2) - the International Organization for Standardization, the global body that publishes technical standards.
- **JTC1** (§2.2) - the joint ISO/IEC committee for information technology.
- **SC22** (§2.2) - the subcommittee under JTC1 for programming languages.
- **WG21** (§2.2) - Working Group 21, the group inside SC22 that writes the C++ standard (formal nesting; used plainly before this).
- **subgroup** (§2.3, shallow) - one of the smaller groups WG21 splits into to do its work. Ch 5 owns the specific rooms.
- **plenary** (§2.3, shallow) - the session where the whole committee meets and makes final decisions. Ch 7 and Ch 8 deepen.
- **National Body (NB)** (§2.4) - a country's official ISO member; its delegates and votes represent that country.
- **guest** (§2.5) - someone who attends without joining a National Body; can take part but can't vote at plenary.
- **convener** (§2.5, shallow) - the person who runs WG21 and whom you email to attend. Ch 5 deepens.
- **INCITS** (§2.6) - the US National Body path, open to anyone regardless of nationality (about 800 USD/year).
- **ISO Global Directory** (§2.7) - the roster your name must reach to cast a counting vote at plenary.
- **champion** (§2.8, shallow) - someone who presents and pushes a proposal. Ch 11 deepens.

Prerequisites: the standard (§1.5); WG21 (intro).
Links: isocpp.org committee page, INCITS, the Boost Foundation, C++Now, CppCon, ISO Global Directory.

---

## Chapter 3 - How Papers Work

Arc: enters confused about how work happens, leaves understanding the paper is the unit and how to find and read one.

Subheading order:
- 3.1 Nothing Happens Without a Paper
- 3.2 What a Paper Is and Its Types (design, wording, direction, experience reports, standing documents)
- 3.3 The P-Number System (P, N, D; revisions R0 and up)
- 3.4 wg21.link: Finding Any Paper
- 3.5 The Mailing System and Deadlines (papers arrive in batches; pre-meeting deadline weeks ahead)
- 3.6 The Paper Lifecycle (idea to International Standard)
- 3.7 Design Review vs Wording Review (and the handoff between them)
- 3.8 Defect Reports and the Issues Lists (Core and Library Issues Lists)
- 3.9 Reading a Paper Critically (what to look for, red flags, compare against the prior revision)
- 3.10 The Documents Everyone Should Read: SD-4 and P0939

Owns:
- **paper** (§3.1) - a written proposal or report; the unit of all committee work.
- **P-number / N-number / D-number** (§3.3) - a paper's ID: P is a numbered proposal, D is an unpublished draft, N is older or administrative.
- **revision (R0, R1, ...)** (§3.3) - the version of a paper; R0 is first, higher is later.
- **wg21.link** (§3.4) - the link shortener that resolves any paper, like wg21.link/p2300r10 (used and linked earlier; formal explanation here).
- **the mailing** (§3.5) - the batch of papers published before and after each meeting.
- **pre-meeting mailing deadline** (§3.5) - the cutoff, weeks before a meeting, after which papers aren't actionable.
- **the paper lifecycle** (§3.6) - the path a paper takes from idea to International Standard.
- **design review** (§3.7) - judging whether a proposal's idea is good and wanted.
- **wording review** (§3.7) - polishing the exact standard text after the design is accepted.
- **defect report** (§3.8) - a report of a bug in the standard itself.
- **issues lists** (§3.8) - the Core Issues List and Library Issues List that track standard defects.
- **standing document (SD)** (§3.2) - a numbered document holding the committee's own rules and practices.
- **SD-4** (§3.10) - the standing document describing how WG21 works in practice; everyone is expected to know it.
- **Direction Group / P0939** (§3.10, shallow) - the small group that sets direction, and its priorities paper P0939. Ch 5 deepens.

Prerequisites: the standard, IS (§1.5, §1.6); WG21, plenary, subgroup (Ch 2).
Links: wg21.link, SD-4 on isocpp.org, P0939, open-std.org.

---

## Chapter 4 - How to Participate Remotely

Arc: enters thinking you must travel to matter, leaves empowered to contribute today from your desk.

Subheading order:
- 4.1 You Can Take Part From Your Desk (scribing, reviewing, championing without a paper)
- 4.2 The Reflector (what it is, culture, subscription is chair-gated after your first meeting)
- 4.3 The Official Sites: isocpp.org, open-std.org, wg21.org
- 4.4 The Wiki (agendas, schedules, Zoom links, poll pages; don't edit unless told)
- 4.5 Real-Time Chat: Mattermost and Discord
- 4.6 Floating Ideas: std-proposals
- 4.7 Telecons Between Meetings (subgroup telecons, ~30/month; the shared calendar)
- 4.8 Voting From Afar: Electronic Polls
- 4.9 Following a Live Meeting (virtual vs in-person; results, trip reports, minutes; time zones)
- 4.10 Open-Source Help: The Beman Project and Tools (npaperbot, Minutes Sanitizer)

Owns:
- **the reflector** (§4.2) - the committee's email mailing lists, where most discussion happens between meetings.
- **isocpp.org** (§4.3) - the public-facing C++ site, with standing documents and the committee page.
- **open-std.org** (§4.3) - the official archive of papers and drafts.
- **wg21.org** (§4.3) - the enhanced mailing and paper-discovery site.
- **the wiki** (§4.4) - the committee's internal hub for agendas, schedules, Zoom links, and poll pages.
- **Mattermost** (§4.5) - the committee's real-time chat at chat.isocpp.org.
- **std-proposals** (§4.6) - the public Google Group for floating an idea before writing a paper.
- **telecon** (§4.7) - an online meeting a subgroup holds between the big in-person meetings.
- **the Beman Project** (§4.10) - an effort to implement standard-library proposals as open source.

Prerequisites: paper, the mailing (Ch 3); subgroup, plenary (Ch 2).
Note: "straw poll" appears here in plain words only ("quick online votes"). The formal definition is owned by §8.2.
Links: isocpp.org, open-std.org, wg21.org, chat.isocpp.org, std-proposals group, #include C++ Discord, the Beman Project, wg21.link.

---

## Chapter 5 - How WG21 Is Structured

Arc: enters seeing a blur of acronyms, leaves with a clear map of the rooms and who decides what.

Subheading order (as written):
- 5.1 One Committee, Many Rooms (the rooms and the schedule grid)
- 5.2 The Evolution Rooms: EWG and LEWG
- 5.3 The Wording Rooms: CWG and LWG
- 5.4 Study Groups (incubation; subgroup specialties)
- 5.5 How a Paper Moves Between Groups (joint sessions)
- 5.6 The Train Model (ships every three years)
- 5.7 How Features Get Removed (pulling vs removing)
- 5.8 The People Who Run It (convener, project editor, chair, session staff)
- 5.9 Who Sets Priorities: The Direction Group
- 5.10 Chair Power (discretion, paper queues, scheduling as a silent veto)
- 5.11 The Rules: Standing Documents
- 5.12 The Limits of Power: Volunteers and the Implementer Veto

Owns:
- **the rooms** (§5.1) - the nickname for WG21's main working groups.
- **the schedule grid** (§5.1) - the timetable of parallel sessions across the week.
- **EWG (Evolution Working Group)** (§5.2) - the room that decides language design direction.
- **LEWG (Library Evolution Working Group)** (§5.2) - the room that decides library design direction.
- **CWG (Core Working Group)** (§5.3) - the room that finalizes the exact language wording.
- **LWG (Library Working Group)** (§5.3) - the room that finalizes the exact library wording.
- **study group (SG)** (§5.4) - a focused group that incubates ideas in one area before they reach the main rooms.
- **the train model** (§5.6) - the rule that C++ ships on a fixed schedule, every three years, with whatever is ready.
- **convener** (§5.8, deep) - the officer who runs WG21: appoints chairs, creates study groups, sets the schedule (introduced in Ch 2).
- **project editor** (§5.8) - the person who maintains the working draft text.
- **chair** (§5.8) - the person who runs a room: sets its agenda, words its polls, and calls consensus.
- **the Direction Group** (§5.9, deep) - the small senior group that sets priorities, publishing them in P0939 (introduced in Ch 3).
- **the implementer veto** (§5.12) - the reality that if GCC, Clang, and MSVC won't implement something, the standard can't force them.

Prerequisites: convener, plenary, subgroup (Ch 2, restate with parenthetical); design review, wording review (Ch 3); IS (Ch 1).
Links: isocpp.org, P0939, the LEWG GitHub wiki.

---

## Chapter 6 - Getting There

Arc: enters anxious about the practical leap, leaves prepared and braced.

Subheading order:
- 6.1 What a Meeting Week Looks Like (three per year; Monday-Saturday, ~8:30 to 5:30, plus evenings)
- 6.2 Preparing Before You Go (read the mailing, choose subgroups, read SD-4, check the wiki)
- 6.3 The Cost and How to Pay for It (no registration fee; travel, lodging, meals on you; sponsorship, self-funding, grants)
- 6.4 What to Bring (laptop, charger, power adapter for the host country)
- 6.5 Bracing for Your First Meeting (it will be overwhelming, and that's normal)

Owns:
- **meeting week** (§6.1) - the six-day Monday-to-Saturday gathering, held three times a year.

Prerequisites: the mailing, SD-4 (Ch 3); subgroup, the wiki (Ch 2, Ch 4); the rooms (Ch 5).
Psychological note: your first meeting is orientation, not production. Set expectations low and steady.

---

## Chapter 7 - In the Room

Arc: enters nervous about how to behave, leaves confident in the room's rhythm and norms.

Subheading order:
- 7.1 The Opening Plenary
- 7.2 Cycling Between Rooms (follow the paper schedule, not one room)
- 7.3 Meeting Etiquette (queue behavior, laptop etiquette)
- 7.4 How to Speak (raise your hand, two minutes, state your name, use the microphone)
- 7.5 Confidentiality and the Code of Conduct (what you can and cannot quote)
- 7.6 Note-Taking (how minutes work, how to volunteer)
- 7.7 Where Relationships Form (evening sessions, meals are networking)
- 7.8 Get Out of Your Lane (spend time in rooms outside your specialty)

Owns:
- **opening plenary** (§7.1) - the Monday session that introduces officers, approves the agenda, then sends everyone to the rooms.
- **the Code of Conduct** (§7.5) - the rules of behavior covering every WG21 interaction, including meals and social media.
- **confidentiality rule** (§7.5) - the norm against blogging, recording, or quoting people by name without consent.

Prerequisites: plenary (Ch 2), the rooms, chair (Ch 5), paper schedule (Ch 3, Ch 5).
Psychological note: the meeting cadence is seductive; don't let attendance become the reward. Go into every discussion skeptical and question the room's priors.

---

## Chapter 8 - How Decisions Get Made

Arc: enters puzzled by how votes work, leaves able to read the room and vote with conviction.

Subheading order (as written):
- 8.1 The Culture of Consensus (not unanimity, absence of sustained opposition; consensus as social pressure)
- 8.2 Straw Polls and the Five-Point Scale
- 8.3 Reading Poll Results (the 2:1 guideline; strong opposition outweighs the count)
- 8.4 Straw Poll vs Formal Vote
- 8.5 When to Vote and When to Abstain (vote sincerely; demand evidence)
- 8.6 Two Gates: Direction vs Design (and "encouragement is not approval")
- 8.7 A Small Minority Can Block (and the escalation path)
- 8.8 The Plenary: Where Features Enter C++ (closing plenary, unanimous consent, std::byte)
- 8.9 Processing Issues (CWG and LWG)
- 8.10 The Ballot Stages: DIS and FDIS (National Body comments)

Owns:
- **consensus** (§8.1) - general agreement shown by the absence of sustained opposition; not a majority and not unanimity.
- **straw poll** (§8.2) - a non-binding vote on a five-point scale (strongly favor, weakly favor, neutral, weakly against, strongly against) that shows the room's sense.
- **the 2:1 guideline** (§8.3) - the rough rule that a proposal normally advances with about twice as many in favor as against.
- **formal vote** (§8.4) - a binding vote, used at plenary and in ballots, distinct from a straw poll.
- **abstain** (§8.5) - choosing not to vote because you aren't familiar with the issue.
- **direction approval** (§8.6) - an early "we want this" gate.
- **design approval** (§8.6) - a later "this specific design is right" gate.
- **escalation path** (§8.7) - the SD-4 route to bring a disagreement to a higher chair, with a paper.
- **closing plenary** (§8.8) - the Saturday session where subgroups report and adoptions are voted.
- **DIS / FDIS** (§8.10) - the Draft and Final Draft International Standard ballot stages where National Bodies vote.

Prerequisites: plenary, NB (Ch 2); the rooms, CWG, LWG, chair (Ch 5); paper (Ch 3); SD-4 (Ch 3).
Psychological note: consensus is social pressure with a polite name. Keep your wits about you, vote your conviction not the room's momentum, and demand evidence.

---

## Chapter 9 - What Goes Into a Proposal

Arc: enters with an idea, leaves knowing the bar a proposal must clear.

Subheading order:
- 9.1 What Belongs in a Paper (motivation, design, alternatives, implementation experience)
- 9.2 The Three Pillars (example-based, principle-based, shows alternatives)
- 9.3 The Abstract as Elevator Pitch
- 9.4 Tony Tables
- 9.5 Show the Alternatives: The Steel Man
- 9.6 Implementation Experience (and feature-test macros)
- 9.7 Writing Standard Wording (shall vs should; stable labels, not section numbers)
- 9.8 A Short Tutorial
- 9.9 Scope and Dependencies (split large papers; omnibus for tiny changes)
- 9.10 The High Bar (burden on the proposer, default no; core language is hard mode; most papers don't make it)

Owns:
- **the Three Pillars** (§9.2) - the habit that a strong paper is example-based, principle-based, and shows alternatives.
- **Tony Table** (§9.4) - a side-by-side before/after table that makes a proposal's value visible.
- **the Steel Man** (§9.5) - the strongest version of the argument against your proposal, which you then answer with evidence.
- **implementation experience** (§9.6) - proof your proposal has been built and used, ideally in more than one place.
- **feature-test macro** (§9.6) - a predefined symbol that lets code check whether a compiler supports a feature.
- **shall / should** (§9.7) - standard-wording verbs: "shall" is a requirement, "should" is advice.

Prerequisites: paper, design review, wording review, SD-4, P0939 (Ch 3); the standard, stable labels (Ch 1); consensus, design approval (Ch 8).
Psychological note: be convinced by overwhelming evidence and nothing else. Go into your own paper skeptical.

---

## Chapter 10 - Producing and Submitting Your Paper

Arc: enters ready to write, leaves equipped with the tools and the submission path.

Subheading order:
- 10.1 Float the Idea First (std-proposals)
- 10.2 Writing the Paper: Tools and Template (mpark/wg21, Bikeshed, COWEL, LaTeX; the official template)
- 10.3 The Formatting Standard: SD-7
- 10.4 Make It Accessible (contrast, monospace code, logical headings)
- 10.5 Prototyping: Compiler Explorer and GitHub
- 10.6 Co-Authoring and Revision History
- 10.7 Patent Disclosure
- 10.8 Getting on the Agenda (contact the chair; no presenter means no discussion)
- 10.9 The Other 80% (design approval is a start, not the finish)

Owns:
- **SD-7** (§10.3) - the standing document with the paper formatting rules.
- **the official template** (§10.2) - the proposal template at isocpp.org, plus tools like mpark/wg21 and Bikeshed.
- **Compiler Explorer** (§10.5) - the online tool (godbolt.org) for showing code run across compilers.

Prerequisites: std-proposals (Ch 4); paper, P-number, the mailing, the deadline (Ch 3); chair (Ch 5); design approval (Ch 8); implementation experience (Ch 9).
Links: std-proposals, isocpp.org template, mpark/wg21, Bikeshed, godbolt.org, GitHub.

---

## Chapter 11 - Championing Your Paper

Arc: enters hopeful about your paper, leaves resilient for the long social campaign.

Subheading order (as written):
- 11.1 Before the Room: Presocialization
- 11.2 Finding and Being a Champion (finding one when you can't attend)
- 11.3 The First Gate: "Do We Want It at All?" (frame it in Direction Group priorities)
- 11.4 When the Poll Goes Against You (sent back is not rejection; re-litigation)
- 11.5 Negotiating: Back Pocket and Max-Min
- 11.6 Disagreement vs Opposition, and When to Withdraw
- 11.7 How Reputation Works (start small, claim a territory, pick your battles)
- 11.8 The Presenter Is Judged Too (don't cause needless delay; escalation erodes credibility; procedural momentum)
- 11.9 The Long Game (years not months; famous long-running proposals)
- 11.10 The Structural Headwinds (the bandwidth problem; the expert bubble; corporate resource asymmetry)
- 11.11 The Emotional Side (don't take it personally; burnout is real)

Owns:
- **presocialization** (§11.1) - talking to people about your idea in the hallways before you present it.
- **champion** (§11.2, deep) - the person who presents and advocates for a paper, author or not (introduced in Ch 2).
- **back pocket alternative** (§11.5) - a fallback design you keep ready if your first choice is rejected.
- **max-min solution** (§11.5) - the smallest version of a proposal that everyone can accept.
- **re-litigation** (§11.4) - re-debating a question the committee already settled.
- **procedural momentum** (§11.8) - the benefit of the doubt a paper earns from many revisions and prior favorable polls.
- **the bandwidth problem** (§11.10) - there are far more papers than reviewers and time to handle them.
- **the expert bubble** (§11.10) - the tendency for papers to be written by experts for experts.

Prerequisites: paper, design and wording review (Ch 3); the rooms, the Direction Group, chair (Ch 5); consensus, straw poll, direction and design approval (Ch 8); most papers don't make it (Ch 9).
Psychological note: burnout is real, the bandwidth problem is structural, and knowing when to stop is a skill. Set boundaries.

---

## Chapter 12 - C++ Design Principles

Arc: enters wondering why good ideas die, leaves respecting the deep constraints.

Subheading order:
- 12.1 Standardize Existing Practice (the committee's safest path; bless what already works)
- 12.2 Backward Compatibility (the weight of decades of code)
- 12.3 The Zero-Overhead Principle
- 12.4 ABI: The Invisible Constraint (why proposals die on ABI; the Prague ABI vote of 2020)
- 12.5 Freestanding vs Hosted

Owns:
- **standardize existing practice** (§12.1) - the preference for blessing tools that already work over inventing new ones.
- **backward compatibility** (§12.2) - the rule that old code keeps working with new standards.
- **the zero-overhead principle** (§12.3) - you don't pay for what you don't use, and what you use you couldn't hand-code better.
- **ABI (application binary interface)** (§12.4) - the binary contract between compiled pieces of a program; breaking it breaks existing programs.
- **the Prague ABI vote** (§12.4) - the February 2020 decision not to break ABI across the library for C++23, while refusing to promise stability forever. In effect, existing binaries stayed safe and the frozen types stayed frozen.
- **freestanding vs hosted** (§12.5) - two environments: freestanding (no operating system, limited library) and hosted (full library).

Prerequisites: the standard, the three implementers (Ch 1); the implementer veto, the Direction Group (Ch 5).

---

## Chapter 13 - Common Mistakes

Arc: enters confident, leaves humble and careful, able to step around the traps.

Subheading order (one mistake per section, each a recap with the correct behavior):
- 13.1 Showing Up Without Announcing
- 13.2 Proposing Core Language Features Too Early
- 13.3 Writing an Idea-Only Paper
- 13.4 Voting on Things You Haven't Followed
- 13.5 Quoting Reflectors or Notes Publicly
- 13.6 Raising Concerns Too Late
- 13.7 Expecting Majority Rule
- 13.8 Talking Too Much Too Soon
- 13.9 Misreading Encouragement as Approval
- 13.10 Ignoring Subgroup Direction
- 13.11 Declaring Victory at Design Approval

Owns: no new terms. Restate each referenced term with a short parenthetical.

Prerequisites: draws on every prior chapter.
Psychological note: don't confuse committee participation with impact. A proposal that never ships helped nobody.
