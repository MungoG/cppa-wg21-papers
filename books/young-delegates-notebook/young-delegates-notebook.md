# A Young Delegate's Notebook

*A Newcomer's Path Into WG21 and the Standardization of C++*

A practical guide for anyone who wants to help shape C++ but has never set foot in a committee meeting. It accumulates: each chapter stands on the ones before it, and you can stop at any chapter and still have something useful to offer. Every paper number is a live link, and every named resource points somewhere real.

*Assembled 2026-08-07*

## Contents

- Introduction
  - Who This Notebook Is For
  - What This Notebook Is Not
  - What Each Chapter Covers
  - How to Read This Notebook
  - Keep Your Wits About You
- 1. The C++ Standard
  - 1.1 Standardization Is a Responsibility
  - 1.2 The Delegate's Oath
  - 1.3 Every Feature Is Forever
  - 1.4 The People Who Inherit Your Work
  - 1.5 The Standard Is Not the Language
  - 1.6 What "The Standard" Actually Is
  - 1.7 How to Read the Standard
  - 1.8 From Standard to Compiler
- 2. Meet the Committee
  - 2.1 You Don't Need Permission
  - 2.2 The Nesting: ISO, JTC1, SC22, WG21
  - 2.3 How the Committee Meets
  - 2.4 What a National Body Is
  - 2.5 The Easiest Start: Attending as a Guest
  - 2.6 Getting Further In: INCITS and the Boost Foundation
  - 2.7 Getting a Vote: The ISO Global Directory
  - 2.8 The Roles You Can Play
  - 2.9 Your First Day: Orientation and Introductions
  - 2.10 Showing Up Is the Secret
  - 2.11 Where Ideas Start: Conferences
- 3. How Papers Work
  - 3.1 Nothing Happens Without a Paper
  - 3.2 What a Paper Is and Its Types
  - 3.3 The P-Number System
  - 3.4 wg21.link: Finding Any Paper
  - 3.5 The Mailing System and Deadlines
  - 3.6 The Paper Lifecycle
  - 3.7 Design Review vs Wording Review
  - 3.8 Defect Reports and the Issues Lists
  - 3.9 Reading a Paper Critically
  - 3.10 The Documents Everyone Should Read: SD-4 and P0939
- 4. How to Participate Remotely
  - 4.1 You Can Take Part From Your Desk
  - 4.2 The Reflector
  - 4.3 The Official Sites: isocpp.org, open-std.org, wg21.org
  - 4.4 The Wiki
  - 4.5 Real-Time Chat: Mattermost and Discord
  - 4.6 Floating Ideas: std-proposals
  - 4.7 Telecons Between Meetings
  - 4.8 Voting From Afar: Electronic Polls
  - 4.9 Following a Live Meeting
  - 4.10 Open-Source Help: The Beman Project and Tools
- 5. How WG21 Is Structured
  - 5.1 One Committee, Many Rooms
  - 5.2 The Evolution Rooms: EWG and LEWG
  - 5.3 The Wording Rooms: CWG and LWG
  - 5.4 Study Groups
  - 5.5 How a Paper Moves Between Groups
  - 5.6 The Train Model
  - 5.7 How Features Get Removed
  - 5.8 The People Who Run It
  - 5.9 Who Sets Priorities: The Direction Group
  - 5.10 Chair Power
  - 5.11 The Rules: Standing Documents
  - 5.12 The Limits of Power: Volunteers and the Implementer Veto
- 6. Getting There
  - 6.1 What a Meeting Week Looks Like
  - 6.2 Preparing Before You Go
  - 6.3 The Cost and How to Pay for It
  - 6.4 What to Bring
  - 6.5 Bracing for Your First Meeting
- 7. In the Room
  - 7.1 The Opening Plenary
  - 7.2 Cycling Between Rooms
  - 7.3 Meeting Etiquette
  - 7.4 How to Speak
  - 7.5 Confidentiality and the Code of Conduct
  - 7.6 Note-Taking
  - 7.7 Where Relationships Form
  - 7.8 Get Out of Your Lane
- 8. How Decisions Get Made
  - 8.1 The Culture of Consensus
  - 8.2 Straw Polls and the Five-Point Scale
  - 8.3 Reading Poll Results
  - 8.4 Straw Poll vs Formal Vote
  - 8.5 When to Vote and When to Abstain
  - 8.6 Two Gates: Direction vs Design
  - 8.7 A Small Minority Can Block
  - 8.8 The Plenary: Where Features Enter C++
  - 8.9 Processing Issues
  - 8.10 The Ballot Stages: DIS and FDIS
- 9. What Goes Into a Proposal
  - 9.1 What Belongs in a Paper
  - 9.2 The Three Pillars
  - 9.3 The Abstract as Elevator Pitch
  - 9.4 Tony Tables
  - 9.5 Show the Alternatives: The Steel Man
  - 9.6 Implementation Experience
  - 9.7 Writing Standard Wording
  - 9.8 A Short Tutorial
  - 9.9 Scope and Dependencies
  - 9.10 The High Bar
- 10. Producing and Submitting Your Paper
  - 10.1 Float the Idea First
  - 10.2 Writing the Paper: Tools and Template
  - 10.3 The Formatting Standard: SD-7
  - 10.4 Make It Accessible
  - 10.5 Prototyping: Compiler Explorer and GitHub
  - 10.6 Co-Authoring and Revision History
  - 10.7 Patent Disclosure
  - 10.8 Getting on the Agenda
  - 10.9 The Other 80%
- 11. Championing Your Paper
  - 11.1 Before the Room: Presocialization
  - 11.2 Finding and Being a Champion
  - 11.3 The First Gate: "Do We Want It at All?"
  - 11.4 When the Poll Goes Against You
  - 11.5 Negotiating: Back Pocket and Max-Min
  - 11.6 Disagreement vs Opposition, and When to Withdraw
  - 11.7 How Reputation Works
  - 11.8 The Presenter Is Judged Too
  - 11.9 The Long Game
  - 11.10 The Structural Headwinds
  - 11.11 The Emotional Side
- 12. C++ Design Principles
  - 12.1 Standardize Existing Practice
  - 12.2 Backward Compatibility
  - 12.3 The Zero-Overhead Principle
  - 12.4 ABI: The Invisible Constraint
  - 12.5 Freestanding vs Hosted
- 13. Common Mistakes
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

---

## Introduction

> "I have not found among my possessions anything I hold more dear or esteem so highly as my knowledge of the actions of great men, learned through long experience of modern events and continual study of ancient ones - which, having thought about and examined with great diligence, I have now set down in this little book and send to Your Magnificence."
>
> Niccolò Machiavelli, dedicatory letter, [*The Prince*](https://www.gutenberg.org/ebooks/1232)

I wrote this for you. Not for the experts who've sat in these rooms for twenty years. For you, the one who just found out that a group of volunteers decides what goes into C++, and wondered how to join them.

Machiavelli gave a prince the one thing he valued most: what he'd learned watching powerful people up close. This notebook works the same way, smaller and plainer. It's the worn, marked-up notebook I wish someone had pressed into my hands before my first meeting.

The group is real, and you can join it. It's called [WG21](https://isocpp.org/std/the-committee), and it's the committee that owns the C++ standard. Around 350 people show up, argue, vote, and ship a new version of the language about every three years.

These are some of the most careful people you'll ever meet. They'll spend an hour on the placement of a single word. They do it because the stakes are high, and you'll come to understand why.

### Who This Notebook Is For

This is for the newcomer. Maybe you've never been to a meeting, or you've been to a few and still feel lost in the room. Either way, you're in the right place.

You don't need to know every rule in the language. You don't need to have written a compiler. You need curiosity and the patience to keep showing up.

### What This Notebook Is Not

This isn't a textbook. It isn't a manual or an encyclopedia. It's a notebook, and a notebook has a point of view.

This one takes sides. It values stability over novelty, and it puts the people who use C++ ahead of the people who design it. When the evidence is thin, it says so.

### What Each Chapter Covers

The book moves in order, from "what is this thing?" to "how do I take part well?" Each line below is one chapter:

- Chapter 1 shows you why the standard matters and why every change you make is close to permanent.
- Chapter 2 introduces the committee and the many doors you can walk in through.
- Chapter 3 explains how papers work, because the paper is the unit of all the work.
- Chapter 4 shows you how to take part from your desk, long before you ever travel.
- Chapter 5 maps the rooms and groups, so you know who decides what.
- Chapter 6 helps you plan, fund, and survive your first meeting.
- Chapter 7 covers how to behave, speak, and find your footing once you're in the room.
- Chapter 8 explains how decisions get made, and what a vote really means.
- Chapter 9 lays out what a proposal must contain to earn the committee's time.
- Chapter 10 walks through the tools, the format, and the deadlines for your own paper.
- Chapter 11 covers the long work of building support and surviving feedback.
- Chapter 12 names the deep design constraints that decide what can and can't ship.
- Chapter 13 collects the common mistakes, so you can step around them.

### How to Read This Notebook

Read this in order. Each chapter stands on the ones before it. By the end of any chapter, you could stop and still have something useful to offer.

Follow the links. Every paper number points to its source through [wg21.link](https://wg21.link), and every named resource points somewhere real. When you want to go deeper, the path is right there in the text.

### Keep Your Wits About You

One habit matters more than any fact in this notebook: stay skeptical. The committee runs on persuasion, and persuasion can carry you past your own judgment. Hold onto it.

Everything else is detail. Here's the whole notebook in four lines:

> Keep your wits about you. Be skeptical. Demand evidence. Put the users first.

That's the notebook. Welcome. I'm glad you came.

---

## 1. The C++ Standard

Start here, because everything else rests on this. The C++ standard is the thing you'll help build, and it deserves your care.

### 1.1 Standardization Is a Responsibility

The work you're joining is serious. Banks and hospitals run on C++. When you change the language, you change the ground they stand on.

So this isn't a game. It isn't a place to collect a credential or win an argument. It's a place to leave the language better than you found it.

The urge to get your idea adopted is strong, and it can fool you. Passing a proposal was never the goal. The goal is a better language for the people who use it.

The committee has a way to say all of this in one breath.

### 1.2 The Delegate's Oath

The duty fits in a single sentence. Some delegates call it the Delegate's Oath, and it goes like this:

> I vow to do what is best for the language, to make no unnecessary proposals, and to put the needs of sixteen million users ahead of my own.

Each part guards against a real temptation. "Best for the language" outranks best for you or your employer. "No unnecessary proposals" puts the burden on you to show a change earns its place.

The last part is the heart of it. Sixteen million people use C++, and they didn't ask you for anything. You serve them first, even when they'll never know your name.

To feel why the oath is this strict, look at what a single feature costs.

### 1.3 Every Feature Is Forever

The hard rule: the standard almost never shrinks. Features get added. They rarely get removed.

C++ got a [regular-expression library](https://en.cppreference.com/w/cpp/regex) in 2011. It shipped slow, and everyone soon knew it. More than a decade later, it's still slow.

It can't be fixed, because a faster version would break programs already built on the old one. So the slow version stays. The mistake is frozen in place.

That's the pattern for everything you add. Proposing a feature costs you a few months. Maintaining it costs everyone else forever.

### 1.4 The People Who Inherit Your Work

The cost falls on the people who use C++. More than sixteen million developers write it today. Whatever you standardize, they inherit.

They didn't vote for your feature, and most will never read a proposal. They meet your work as a line in compiler release notes. By then it's permanent.

The cost reaches past users, too. Three compiler teams must build and maintain every feature. Every textbook, course, and tutorial has to cover it.

All of this is "the standard." That word means something more exact than you might expect.

### 1.5 The Standard Is Not the Language

You might think "C++" and "the C++ standard" are the same thing. They're not. Three different things wear that name.

**The standard** is a document. It's a long, exact text that says what C++ means. It's words, not a program.

A compiler is a program that tries to follow that document. The three big ones don't follow it perfectly, and they don't all agree. So the C++ you can actually use is whatever your compiler accepts.

Keep three things apart:

- The standard is the text.
- The compilers are the programs that try to follow it.
- The living language is what real code relies on.

That text isn't one fixed thing either. It comes in a few forms worth knowing.

### 1.6 What "The Standard" Actually Is

#### 1.6.1 The Working Draft

Behind every C++ release is a living document. It's called **the working draft**, and it changes all the time. You can [read the current one](https://eel.is/c++draft/) online, and its source lives on [GitHub](https://github.com/cplusplus/draft).

#### 1.6.2 The Published Standard

Every three years, the committee freezes the draft and publishes it. That published version is an **International Standard**, or IS. [C++23](https://en.cppreference.com/w/cpp/23) is an IS, and C++26 will be one when it lands.

#### 1.6.3 Two Smaller Document Types

Two lighter document types show up too. A **Technical Specification**, or TS, is an experimental tryout. Implementers can offer it so people test a feature before it joins the standard.

A **Technical Report**, or TR, is an older, informational kind. You'll rarely meet a new one. The committee mostly uses the TS now.

### 1.7 How to Read the Standard

The standard looks scary, and it is dense. But it's organized, and a few habits make it readable. You don't have to read it front to back.

Most of it is **normative text**, the parts that state actual rules. Mixed in are notes and examples, which never add a rule. They only help you understand the rules around them.

Each part carries a **stable section label** in brackets, like [\[container.requirements\]](https://eel.is/c++draft/container.requirements). Those labels stay fixed between drafts. Section numbers shift, so point to the label, not the number.

Knowing the document is one thing. Getting a feature into your hands is another.

### 1.8 From Standard to Compiler

Publishing the standard doesn't put a feature in your hands. Someone has to build it into a compiler first. Until they do, it stays words on a page.

Three teams do most of that building. They're **the three implementers**: [GCC](https://gcc.gnu.org/), [Clang](https://clang.llvm.org/), and [MSVC](https://learn.microsoft.com/en-us/cpp/). Almost all the C++ in the world runs through one of them.

This makes the three implementers powerful. If all three won't build a feature, the standard can't force them. So a feature isn't real until it ships in a compiler you can download.

Now you know what the standard is and why it's heavy. A feature is forever, and sixteen million people carry it. Keep that weight in mind as you find your way in.

---

## 2. Meet the Committee

You can join this. The door is wider than it looks.

### 2.1 You Don't Need Permission

No one grants you permission to help. You won't find a gate or a gatekeeper. If you want to contribute, you start.

The real barriers are smaller than the rumors. You can attend a meeting as a guest for free. You can review a proposal with nothing but a free login.

You also don't have to be a wizard. Plenty of useful work isn't writing code at all. Taking notes and reviewing other people's proposals both count.

### 2.2 The Nesting: ISO, JTC1, SC22, WG21

The committee has a long official name, and it tells a story. It's **ISO/IEC JTC1/SC22/WG21**. Read it left to right, and it nests from the biggest body down to the smallest.

[ISO](https://www.iso.org) is the International Organization for Standardization, which publishes standards for almost everything. Inside it, **JTC1** handles information technology, and **SC22** handles programming languages. **[WG21](https://isocpp.org/std/the-committee)**, Working Group 21, is the part that owns C++.

The surprising part: WG21 doesn't publish the standard (the official document that defines C++) by itself. It agrees on what C++ should be, and then the bodies above it run the official vote.

### 2.3 How the Committee Meets

The committee works in two modes. Most of the time it splits into **subgroups**, the smaller teams that each handle one slice of the work. Once in a while the whole committee gathers as one.

That full gathering is the **plenary**, where the whole committee makes its final decisions. The subgroups do the building. The plenary signs off.

This happens at a real scale. A recent meeting drew about 200 people from 28 nations. Twenty-five of them were first-time guests, so newcomers are normal here.

Those countries are called **National Bodies**.

### 2.4 What a National Body Is

The official members here are countries, not people. Each participating country is a **National Body**, or NB. Your NB is your country's standards organization.

Each National Body gets one vote on the big decisions. It's one country, one vote, no matter how many people that country sends. A delegation of twenty counts the same as a delegation of one.

You take part by joining a National Body or by visiting as a guest. Joining makes you a full member, with a path to a vote. Visiting as a guest is the quickest way to start.

### 2.5 The Easiest Start: Attending as a Guest

The lowest-cost way in is to attend as a **guest**. A guest takes part in a meeting without joining a National Body. It's free.

To come as a guest, [email the convener](https://isocpp.org/std/meetings-and-participation), the person who runs WG21. Tell them you'll attend, and name the country where you live or work. ISO asks for at least one week's notice.

As a guest you can do almost everything. You join the subgroups, argue, and take part in their polls. The one thing you can't do is vote in the plenary, where final decisions are made.

If you want that plenary vote, you join a National Body. Two paths make that affordable.

### 2.6 Getting Further In: INCITS and the Boost Foundation

You don't have to be American to join through the United States. Its National Body, [INCITS](https://www.incits.org/participation/apply-for-membership), opened its C++ committee to anyone, regardless of nationality or employer. Membership runs about 800 US dollars a year.

If even that fee is a barrier, a free path exists. The [Boost Foundation](https://sites.google.com/boost.org/boost-foundation/c-standardization) sponsors free representative spots for C++ developers who can't afford dues. The access is the same.

Pick by what you need. To start cheap and fast, come as a guest. To get a vote without a big bill, take the INCITS or Boost path.

### 2.7 Getting a Vote: The ISO Global Directory

A vote at the plenary counts only when you're accredited. Accredited means your name sits in the **ISO Global Directory**, the official roster of voting experts. Your National Body puts you there.

A common trap: people attend for years as guests and never get accredited. Then, on the one vote that matters to them, they have no vote to cast.

A vote is only one way to matter. Many roles need no vote at all.

### 2.8 The Roles You Can Play

You can help in more ways than you'd guess. The easiest roles need no membership at all. Here are the main ones:

- Note-taker. Every session needs minutes, and anyone can be asked to take them. It's a clear way to be useful on day one.
- Reviewer. Read a proposal that's coming up and post what you think. A free login is all you need, and giving reviews earns you reviews of your own.
- Author. Write up an idea and submit it for the committee to consider.
- Implementer. Build part of a feature in a compiler and report what you learn.
- Champion. Present a proposal and push it through the rooms. A **champion** can be the author, or someone else who believes in the idea.

Notice that the first two need nothing but your time and attention.

### 2.9 Your First Day: Orientation and Introductions

Your first meeting starts before the work does. The evening before, at 6pm on Sunday, there's a newcomer orientation, usually in the meeting hotel's lobby. It covers how the week works and answers your questions.

Soon after, the newcomers get a welcome. You may be asked to say who you are and where you're from. It's brief and friendly, not a test.

After the welcome, one habit matters more than any other.

### 2.10 Showing Up Is the Secret

The single biggest thing you can do is keep showing up. An old line says eighty percent of success is showing up. In this committee, it's close to true.

Showing up builds two things: familiarity and trust. People learn your name and your judgment over many meetings. The committee runs on relationships, and relationships need time.

The reverse is true too. When a regular voice stops attending, the committee quietly loses what that person carried. Presence matters here more than brilliance.

### 2.11 Where Ideas Start: Conferences

A lot of C++ ideas grow at conferences first. Two matter most for newcomers. They're where you'll meet committee people in a relaxed setting.

[C++Now](https://cppnow.org) is the smaller one, held each spring in Aspen, Colorado. It's intense and expert-heavy, and it's where people push and refine new ideas. Many committee members go.

[CppCon](https://cppcon.org) is the big one, open to everyone from beginners to experts. It's where ideas reach the wider community. Going to either is a low-pressure way to start.

The door isn't locked. You can attend for free, help without a title, and grow from there. Next up: the thing all this work revolves around - the paper.

---

## 3. How Papers Work

All the committee's work flows through one kind of object: the paper. Learn how papers work and the rest of the process makes sense.

### 3.1 Nothing Happens Without a Paper

One rule explains most of the committee. If a proposal doesn't have a **paper**, it doesn't exist. A paper is a written proposal or report, and it's the unit of all the work.

A paper needs three things before a room can act on it. The paper has to exist, it has to arrive on time, and someone has to be there to present it. Miss one and the room moves on.

This is why hallway ideas don't decide anything. A sharp point made out loud, with no paper, rarely changes a vote. If you want to move the committee, you write it down.

### 3.2 What a Paper Is and Its Types

Papers sort first by what they want from you. An ask-paper proposes something and asks for a poll. An inform-paper puts facts on the record and asks for nothing.

Here's a quiet truth: the inform-paper often gets read more. It adds nothing to the chair's crowded schedule. So it slips in where an ask-paper waits in line.

Papers also split by job. Some argue for a design, and others nail down exact standard wording. A separate kind, the **standing document**, holds the committee's own rules instead of proposing anything.

Every paper has an ID, and the ID tells you a lot.

### 3.3 The P-Number System

Every paper carries a number, and the first letter tells its kind. A **P-number** marks a proposal, like P1234. An **N-number** marks an administrative document like an agenda, and a **D** marks a draft that isn't public yet.

After the number comes a **revision**, like R0 for the first version, then R1, R2, and up. Read the newest one you can find.

The revision tells you history, not quality. R0 is brand new and untested in the room. A paper at R5 has been reworked five times, which signals persistence, not correctness.

Once you have a number, the paper is one link away.

### 3.4 wg21.link: Finding Any Paper

You don't hunt for papers by hand. **wg21.link** turns any paper number into a link. Type the number after the slash, in lowercase, and it takes you there.

The pattern stays the same every time. For P2300 revision 10, you write [wg21.link/p2300r10](https://wg21.link/p2300r10). A committee member has run this service for years.

Papers reach you in batches called mailings.

### 3.5 The Mailing System and Deadlines

The committee publishes papers in batches called **the mailing**. A few times a year, hundreds of papers drop at once, before and after each meeting. The committee puts out 300 to 500 papers a year this way.

Two sites hold the same papers. [open-std.org](https://www.open-std.org/jtc1/sc22/wg21/) is the official archive, a plain list in date order. [wg21.org](https://wg21.org) is a friendlier view, searchable and sorted by group.

Timing decides what the committee can act on. Each meeting has a **pre-meeting mailing deadline**, weeks ahead of it. A paper that misses the deadline isn't actionable, so it waits for the next round.

From that first mailing, a paper begins a long climb.

### 3.6 The Paper Lifecycle

A paper that makes it follows a set path called **the paper lifecycle**. It starts as an idea and ends in the published standard. Most papers stop somewhere in between.

The steps, in order:

1. You submit the paper to a mailing.
2. A small group incubates the idea and shapes it.
3. A design group decides whether the direction is right.
4. A wording group turns the approved design into exact standard text.
5. The plenary, the full committee, votes it into the working draft (the living text that becomes the next standard).
6. The national bodies run a formal ballot, and it's published.

Each step is a place to stop. A paper can stall, get sent back, or quietly die at any rung. Reaching the top takes years, and most papers never do.

Two of those rungs, design and wording, ask very different questions.

### 3.7 Design Review vs Wording Review

The committee reviews a paper in two separate stages. **Design review** asks one question: is this the right feature? **Wording review** asks a different one: is the exact text precise and complete?

These take different mindsets. Design review weighs whether C++ wants the feature at all. Wording review assumes the answer is yes and hunts for gaps in the spec text.

The two happen in order. First a design group approves the direction. Then a wording group takes over and writes the standard text.

Not every paper proposes something new. Some just fix mistakes.

### 3.8 Defect Reports and the Issues Lists

The standard ships with bugs, and they get fixed. A **defect report** points out a mistake in the published standard and asks to correct it. It repairs wording rather than adding a feature.

These bugs are tracked in public. The **issues lists**, one for the core language and one for the library, hold every known defect and its status. Editors work through them over time.

Whether a paper proposes or fixes, you'll want to read it well.

### 3.9 Reading a Paper Critically

You can't read 500 papers a year, so don't try. Read the ones in your subgroup (the smaller group you follow) and skim the rest. Read deeply only where you'll vote.

Always read the newest revision. An old one makes you raise a concern the author already fixed. When a paper comes back, compare it against the last version to see what changed.

Run a few questions on any paper you read:

- What problem does it solve, and does it say so?
- Does it weigh other options, or pretend none exist?
- Has anyone built it, or is it only an idea?
- What old code breaks, and how do users move forward?
- Is the wording clear on the hard cases?

The gaps are the red flags. Empty answers, like no implementation and no migration plan, are objections waiting on the floor. A strong paper closes them before you ask.

Two documents will save you more than any single paper.

### 3.10 The Documents Everyone Should Read: SD-4 and P0939

Two documents are worth reading before almost anything else. **[SD-4](https://isocpp.org/std/standing-documents/sd-4-wg21-practices-and-procedures)** is the committee's own rulebook, the practices and procedures everyone is expected to know. It covers polls, consensus, deadlines, and the rules for guests.

The other is **[P0939](https://wg21.link/p0939)**, the priorities paper. A small senior group called the **Direction Group** writes it to say what the committee should focus on. Reading it tells you which way the wind is blowing.

Now you can find a paper, read it, and follow its climb. Everything so far works from your desk. The next chapter is about taking part.

---

## 4. How to Participate Remotely

You don't have to fly anywhere to start. Most of the committee's life happens online, between meetings. All you need is a desk and a browser.

### 4.1 You Can Take Part From Your Desk

Plenty of useful work happens with no plane ticket. You can read, review, scribe, and discuss from home. None of it needs a meeting badge.

The lowest-effort start is reviewing a proposal. You read one that's coming up and post what you think. A free login is all it takes, and giving reviews earns you reviews of your own.

### 4.2 The Reflector

The committee's main online home is email. Its mailing lists are called **the reflector**, where most discussion happens between meetings. Each subgroup (smaller group) keeps its own list.

You join the lists once you start taking part, usually after your first meeting. A committee officer helps you subscribe. Before then, you can still read and learn elsewhere.

One rule matters from day one. Reflector posts are private, so you can't quote them in public. You can share poll numbers freely, but a person's words need that person's consent.

### 4.3 The Official Sites: isocpp.org, open-std.org, wg21.org

Three websites cover most of what you need. **[isocpp.org](https://isocpp.org)** is the public-facing site, home to the standing documents and committee news. Start there for anything official.

**[open-std.org](https://www.open-std.org/jtc1/sc22/wg21/)** is the official archive of every paper and draft. It's plain and sorted by date, with no search. It's the source of truth, even if it's bare.

**[wg21.org](https://wg21.org)** is the friendly front door to the same papers. Volunteers built it to search and sort the mailing by group. The papers still live on open-std.org. This view only makes them easier to find.

### 4.4 The Wiki

The committee runs its meetings off **the wiki**. It holds the agendas, schedules, video-call links, and poll pages for each meeting. If you attend, you'll live in it that week.

Two things to know. Don't edit the wiki unless someone asks you to. And you'll get access once you're a participant, not before.

### 4.5 Real-Time Chat: Mattermost and Discord

When things move fast, people use chat. **[Mattermost](https://chat.isocpp.org)**, at chat.isocpp.org, is the committee's own real-time chat. During meetings it keeps the parallel rooms in touch.

For the wider community, the [#include C++ Discord](https://www.includecpp.org) is open to anyone. It's friendly, public, and full of people who'll answer questions. It's a gentle place to learn the culture.

### 4.6 Floating Ideas: std-proposals

Before you write a formal proposal, float the idea first. **[std-proposals](https://lists.isocpp.org/mailman/listinfo.cgi/std-proposals)** is a public list for early-stage ideas. You describe your idea and see what people think.

This step saves you months. You learn fast whether anyone wants the idea. You also get early feedback that makes the real proposal stronger.

### 4.7 Telecons Between Meetings

The big meetings happen three times a year, but the work never stops. Between them, subgroups hold online meetings called **telecons**. Many run every month across the various groups.

You find telecons on the committee's shared calendar. The wiki lists the schedule and the links. Pick a subgroup you care about and sit in.

### 4.8 Voting From Afar: Electronic Polls

Not every vote waits for a meeting. Some groups, the library side especially, run polls online between meetings. They use these to settle smaller questions without burning floor time.

### 4.9 Following a Live Meeting

Meetings are hybrid, so you can join the rooms by video. You won't catch everything from afar. The hallway talk, where a lot gets settled, is hard to join remotely.

After each meeting, the results come out fast. Several delegates post public trip reports, like [Herb Sutter's blog](https://herbsutter.com). Every session is also minuted, so the record is there to read.

One catch if you're far away is time zones. A meeting on another continent can run through your night. Pick the sessions worth losing sleep over, and read the rest later.

### 4.10 Open-Source Help: The Beman Project and Tools

You can help with code without ever attending. The **[Beman Project](https://github.com/bemanproject)** builds library proposals as open source, so people can try them for real. Contributing there is contributing to the standard's future.

A few small tools make committee life easier. wg21.link resolves any paper number, and a chat bot named npaperbot looks up papers for you. You'll pick up more as you go.

Distance is no excuse. You can read, review, chat, and even vote without leaving home.

---

## 5. How WG21 Is Structured

WG21 looks like a wall of acronyms at first. EWG, LEWG, and a dozen study groups blur together. This chapter turns that wall into a clear map.

### 5.1 One Committee, Many Rooms

WG21 does its work in several groups people call **the rooms**. Each room owns one slice of the job. They meet at the same time, in parallel, all week.

That parallel timetable is the **schedule grid**. Six or seven rooms run at once, so you can't attend them all. You pick one room and follow it.

The rooms split into two jobs. Some decide what to build (design), and others write the exact words (wording).

### 5.2 The Evolution Rooms: EWG and LEWG

Two rooms decide whether a feature is worth building. **EWG**, the Evolution Working Group, handles the core language. It judges proposed language features.

**LEWG**, the Library Evolution Working Group, does the same for the standard library. It decides which library features the committee wants. Both rooms answer the design question, not the wording one.

### 5.3 The Wording Rooms: CWG and LWG

Two more rooms turn approved designs into exact standard text. **CWG**, the Core Working Group, writes the language wording. It hunts for gaps and ambiguity in the spec.

**LWG**, the Library Working Group, writes the library wording. It's careful, detailed work on the precise words. These rooms decide whether the text is right, not whether the feature is wanted.

### 5.4 Study Groups

New ideas usually start in a **study group**, or SG. A study group focuses on one area, like concurrency or networking, and shapes raw ideas. Around a dozen are active at a time, numbered SG1 through the low twenties.

Each study group is a world of its own, deep in its specialty. When an idea is ready, it graduates to a design room. The convener creates and closes study groups as needs change.

### 5.5 How a Paper Moves Between Groups

A paper passes through a chain of groups. A study group shapes it, a design room approves it, a wording room finalizes it, and the plenary (the full committee) adopts it. Each handoff is a vote to send it onward.

Some papers touch more than one room. A feature that changes the language and the library visits both evolution rooms, sometimes together in a joint session. The groups coordinate so nothing falls through the cracks.

### 5.6 The Train Model

C++ ships on a fixed schedule, and that rule is **the train model**. Every three years a new standard leaves the station with whatever features are ready. C++11, 14, 17, 20, 23, and 26 all rode it.

The idea came from Herb Sutter in 2011, written up in [P1000](https://wg21.link/p1000r6). Before it, releases slipped for years. The schedule traded "now or never" for "catch the next train."

The model has a safety valve: pull what isn't ready. The committee did exactly that with contracts, taking them out of C++20 to let them bake longer. The valve works, but pulling a big feature is never cheap.

### 5.7 How Features Get Removed

Taking a feature out is far harder than putting one in. Once a feature ships, real programs depend on it, and breaking them isn't an option. So the standard almost only grows.

There's a difference between pulling and removing. Pulling takes a feature out before it ships, which happens now and then. Removing something already shipped is rare, and it can take many years.

### 5.8 The People Who Run It

A handful of officers keep the committee running. The convener (the person who runs WG21, introduced in §2.5) appoints the chairs, creates study groups, and sets the schedule. ISO appoints the convener for a fixed term, confirmed by the national bodies.

The **project editor** keeps the working draft. This is the person who applies every approved change to the text of the standard. It's quiet, exacting work that holds the whole document together.

Each room has its own small crew. A **chair** runs the session, and someone takes the minutes. Anyone in the room can be asked to scribe, so it might be you.

### 5.9 Who Sets Priorities: The Direction Group

A small senior group steers the committee's focus. It's the Direction Group (introduced in §3.10), and it publishes its priorities in [P0939](https://wg21.link/p0939). Membership is by invitation, not election.

The Direction Group doesn't decide individual papers. It sets the big-picture priorities that chairs lean on when scheduling. Framing your work to match those priorities helps it get attention.

### 5.10 Chair Power

The chair of a room holds more power than it looks. The chair decides which papers reach the agenda, how the polls are worded, and when the room has agreed. None of that is a vote, yet all of it shapes outcomes.

The strongest power is the calendar. A chair who never schedules a paper can stall it for years, without ever voting against it. So getting on the agenda is its own battle.

You can watch the queue yourself. The library group tracks its papers in a public list on [GitHub](https://github.com/cplusplus/papers). It shows you what's waiting and what's moving.

### 5.11 The Rules: Standing Documents

The committee's own rules live in standing documents (the numbered SD files from §3.2). A handful exist, from study-group setup to how to write papers. The main one is [SD-4](https://isocpp.org/std/standing-documents/sd-4-wg21-practices-and-procedures), the practices and procedures.

Standing documents change slowly and carefully. New ones need the full committee's agreement. No rule expires on its own, so old practices tend to stay.

### 5.12 The Limits of Power: Volunteers and the Implementer Veto

Here's the truth under all the structure: nobody can be ordered to do anything. WG21 is volunteers, and it only makes recommendations. A proposal dies the moment its people lose interest.

The compiler teams hold a quiet **implementer veto**. If GCC, Clang, and MSVC (the three implementers from §1.8) won't build a feature, the standard can't make them. So the standard can say one thing while your compiler does another.

This isn't only theory. Eighteen implementers recently wrote [P3962](https://wg21.link/p3962r0) asking the committee to slow down, because features pile up faster than they can build them. The people who turn the standard real are stretched thin.

Now the acronyms have a shape: rooms that design, rooms that word, study groups that feed them, and officers who steer. You can see where a paper goes and who decides. Next, let's get you to an actual meeting.

---

## 6. Getting There

Going to your first meeting is a leap, and this chapter is your running start. None of it is as scary as it feels from the outside.

### 6.1 What a Meeting Week Looks Like

The committee meets in person three times a year. Each one is a **meeting week**, running Monday through Saturday. Days are long, roughly 8:30 in the morning to 5:30 in the evening, with optional evening sessions on top.

Inside the week, the rooms run in parallel all day, six or seven at once. The week builds toward the closing plenary on Saturday, where adopted wording enters the standard. It's a marathon, not a sprint.

A good week starts before you arrive.

### 6.2 Preparing Before You Go

The best preparation is reading. Pick one subgroup (smaller group) to focus on, and read its agenda papers before you arrive. You can't follow six rooms, so don't try.

Two more steps round it out. Read [SD-4](https://isocpp.org/std/standing-documents/sd-4-wg21-practices-and-procedures) so the polls and rules don't surprise you. Check the wiki for the schedule, the rooms, and the video links.

Preparation is free. The trip itself is not.

### 6.3 The Cost and How to Pay for It

Attending has no registration fee. What costs money is the trip: flights, lodging, and meals for a week. The bigger cost is often the week itself, away from your job.

People cover it in a few ways. Most are sponsored by an employer who values the work, and others pay their own way. If cost is the barrier, remember the free guest and Boost paths from §2.5 and §2.6.

Once the trip is booked, packing is quick.

### 6.4 What to Bring

Pack light, but pack a laptop. You'll read papers, follow along, and maybe take notes, all on screen. Bring its charger and a plug adapter for the host country.

Bring patience too. The pace is slow and the debates run long. That's the work, not a sign something's wrong.

Even well-packed and well-read, your first meeting will hit hard.

### 6.5 Bracing for Your First Meeting

Your first meeting will overwhelm you. Six rooms run at once, hundreds of papers are in play, and everyone seems to know each other. That's normal, and it passes.

Set your goal low on purpose. Your first meeting is for watching and learning, not for big wins. Pick one room, follow it, and let the rest wash over you.

You're not the odd one out. About two dozen new guests show up at every meeting, just as you will. The Sunday orientation exists because everyone starts here.

Get there, prepare for one room, and forgive yourself for missing the rest. The hard part isn't getting in the door. Next, let's learn how to act once you're in the room.

---

## 7. In the Room

The first time in the room, you'll worry about doing something wrong. Don't. Once you know the unwritten rules, you can relax and pay attention to the work.

### 7.1 The Opening Plenary

The week opens with the whole committee in one room. This is the **opening plenary**. The officers say hello, the agenda gets set, and then everyone scatters to the subgroups.

Don't expect drama here. The opening plenary is short and mostly logistics. The real work happens in the rooms right after.

Once it breaks up, your day becomes a series of choices.

### 7.2 Cycling Between Rooms

You can be in only one room at a time, and six or seven run at once. So follow the schedule, not a single room. Go where the paper you care about is being discussed.

Whatever room you pick, arrive having read its papers. Walking in cold is the classic rookie mistake. The discussion moves fast and won't recap for you.

Wherever you sit, a few manners keep the room running.

### 7.3 Meeting Etiquette

The room runs on small courtesies. Don't dominate the discussion, and don't reopen a point the room has moved past. Keep your laptop for notes, not for distractions that pull your focus.

As a newcomer, listen more than you talk. Spend your first sessions learning how the room thinks. You'll earn your voice faster by using it sparingly.

When you do speak, a little technique goes a long way.

### 7.4 How to Speak

When you want to speak, raise your hand and wait to be called. Make one point, and keep it to about two minutes. The room has a lot to get through, and short is respected.

Two habits help everyone follow you. Say your name before you start, because the scribe and the remote attendees need it. Use the microphone, because without it the online half of the room can't hear you.

What's said in the room comes with a catch.

### 7.5 Confidentiality and the Code of Conduct

What happens in the room mostly stays in the room. The **confidentiality rule** means you don't blog, tweet, photograph, or record the sessions. Treat everything said as off the public record by default.

Two narrow exceptions exist. You can share poll questions and their numbers freely. You can quote a person by name only with that person's consent, which is rarely asked and rarely given, so paraphrase instead.

A **[Code of Conduct](https://isocpp.org/std/standing-documents)** covers how people treat each other. It applies everywhere the committee gathers, including chat, meals, and social events. The short version is to be decent and assume good faith.

One of the easiest ways to help is to write down what happens.

### 7.6 Note-Taking

Every session has to be minuted, and anyone can be asked to do it. So you might be handed the keyboard on day one. Say yes.

Scribing is the best seat in the house for a newcomer. You follow every word, and you learn how the room really works. People also notice and remember the person who helped.

The room isn't the only place that matters. A lot happens in the hallway.

### 7.7 Where Relationships Form

The hallway is not break time. A lot of the real work happens there, between sessions. People sort out disagreements over coffee that would stall a formal room.

Meals work the same way. Lunch and dinner groups form on their own, so join one. Introducing yourself over a meal is normal and expected, not pushy.

One more habit will set you apart.

### 7.8 Get Out of Your Lane

Spend some time in rooms outside your specialty. The committee's own [direction paper](https://wg21.link/p0939) encourages it. You'll build trust across groups and see how the whole machine fits together.

Two habits earn respect fast. Serve before you push: help onboard newcomers and explain procedure. And when someone says "that's just how it works," ask how other groups like IETF or W3C handle the same thing.

Above all, keep your wits about you. The meeting has a rhythm that can sweep you up, and belonging can quietly bend your judgment. Stay skeptical, and question the room's assumptions, including your own.

Now you can walk in, find your room, speak well, and keep your head. The next mystery is how all this talk turns into decisions - the committee's voting process.

---

## 8. How Decisions Get Made

Voting here doesn't work the way you'd expect. No majority rules, and no simple count wins. This chapter shows you how the committee really decides, so you can read a room and vote with a clear head.

### 8.1 The Culture of Consensus

The committee decides by **consensus**, which isn't what most people think. It doesn't mean everyone agrees, and it doesn't mean the majority wins. It means no strong, sustained opposition is left standing.

The test is whether people can live with a decision. A few mild objections don't block it. A small group of firm, reasoned objections can.

Consensus is social pressure with a polite name. Keep your wits about you, and don't let the room's mood stand in for your own judgment.

The committee measures consensus with one simple tool: the straw poll.

### 8.2 Straw Polls and the Five-Point Scale

To read the room, a subgroup takes a **straw poll**. You pick one of five positions: strongly favor, weakly favor, neutral, weakly against, or strongly against. It's not binding, and it shows where the room leans.

In a subgroup, everyone present can vote, guests included. The chair (the person running the room) reads the result and decides whether it's consensus. No fixed number settles it, so the chair judges.

### 8.3 Reading Poll Results

A rough rule guides the call. A proposal normally advances when those in favor outnumber those against by about two to one. This is the **2:1 guideline**, and "normally" is doing a lot of work.

But the count isn't everything. Because the bar is no sustained opposition, a block of strong-against votes can sink a proposal even when far more people favor it.

This really happens. One proposal for C++23 drew 37 in favor and 17 against, and still failed for "sustained strong opposition." More than two to one wasn't enough.

Straw polls steer the work, but they're not the binding vote.

### 8.4 Straw Poll vs Formal Vote

A straw poll is a temperature check, not a decision. The binding kind is a **formal vote**, used at plenary and in the national ballots. One steers the work, the other commits it.

Still, take straw polls seriously. The committee treats them as honest predictors of the formal vote to come. How you vote in a subgroup signals where you'll land later.

Whichever vote it is, you face the same question: should you vote at all?

### 8.5 When to Vote and When to Abstain

Vote your honest position when you have one. If you favor something, say so, not neutral. Neutral means you studied it and truly land in the middle, which should be rare.

When the topic is outside what you know, **abstain**. Abstaining means not voting at all, and it's the honest move on a paper you haven't read. Hearing a summary isn't the same as knowing the issue.

Whatever you vote, ask for evidence. Don't vote yes on enthusiasm or a friend's nod. Be moved by proof, not by the mood of the room.

Votes also come in two flavors that newcomers mix up.

### 8.6 Two Gates: Direction vs Design

A proposal passes through two different gates. **Direction approval** is early: does the committee want this kind of thing at all? **Design approval** is later: is this specific design the right one?

Here's the trap newcomers fall into. Passing either gate is not a promise to ship. A warm direction poll can sit for years with nothing delivered.

Read encouragement for what it is. "We'd like to see more work" means keep going, not "this will ship." One async proposal won a yes on direction, then failed to ship the very next cycle.

Because the bar is no sustained opposition, a few people hold real power.

### 8.7 A Small Minority Can Block

A small group can stop almost anything. Since consensus means no sustained opposition, a handful of firm objectors can block a proposal the majority wants. This is built into the system, not a loophole.

Find the objections early. Learn who has concerns and talk to them before the poll, not after. A lone objector is weaker than one who's found an ally on the merits.

If you still disagree with a result, a path exists. The **escalation path** runs from the room's chair upward, and it carries weight only with a paper, raised before the deadline. A complaint with no paper rarely moves anything.

All these subgroup votes feed one place: the plenary.

### 8.8 The Plenary: Where Features Enter C++

The week ends with the **closing plenary** on Saturday. Each subgroup reports, and the committee votes to adopt finished wording. A feature formally enters C++ the moment a plenary motion to adopt it carries.

Plenary usually runs on silence. The chair asks if anyone objects, and if no one does, the motion passes. Raising a hand to object takes no explanation.

Passing the subgroups isn't the final word. A proposal can clear every room and still fail at plenary. The type `std::byte` did exactly that, blocked late over its name by one national body.

Plenary handles new features. A quieter process handles old bugs.

### 8.9 Processing Issues

Not every decision is about new features. The wording rooms, CWG and LWG (from §5.3), work through the issues lists steadily. They take defects one at a time and settle the exact fix.

New features and fixes alike still face one last hurdle: the national ballots.

### 8.10 The Ballot Stages: DIS and FDIS

Before the standard is final, the countries vote. The draft goes out as a **DIS** (Draft International Standard) for a few months, where national bodies vote and file comments. Then a near-final **FDIS** (Final Draft International Standard) gets a last, shorter approval vote.

Passing takes a strong majority. About two-thirds of voting countries must favor it, and too many no votes sink it. This is where a national body can force a change, as Canada did with `std::byte`.

Now you can read a poll, weigh consensus, and see how a feature becomes real. Notice how often the answer is "show me the evidence." That's exactly what a good proposal brings, which is where we go next.

---

## 9. What Goes Into a Proposal

You have an idea, and you think it belongs in C++. This chapter is the cold shower and the toolkit: what a proposal must carry to earn the committee's time, and why the bar is so high.

### 9.1 What Belongs in a Paper

Getting a paper heard is the low bar. A paper, on time, with someone to present it, and the room will look at it. But getting a yes is a different game.

A real proposal carries four arguments. It shows the problem is worth solving, lays out the design, weighs the alternatives, and proves the thing works. Miss one and a reviewer will find the hole.

### 9.2 The Three Pillars

Strong papers share three habits, call them the **Three Pillars**. They lead with concrete examples, argue from clear principles, and show the alternatives they weighed. A paper missing a pillar feels thin.

Examples do the heavy lifting. Show real before-and-after code, not abstract claims. A reader who sees the improvement needs less convincing than one who's only told about it.

### 9.3 The Abstract as Elevator Pitch

Most readers decide from the abstract alone. So write it as an elevator pitch a busy stranger can grasp. The first sentence should state your finding, not wind up to it.

Keep it honest, though. Promise only what the paper delivers, because reviewers punish hype. A title that names the result beats a title that names the topic.

### 9.4 Tony Tables

The committee's favorite tool is the **Tony Table**, a side-by-side of old code and new. You put the current way on the left and your way on the right. The reader sees the win without you claiming it.

Make the table honest to earn trust. Include the cases where your design isn't shorter or clearer. A table that hides its weak spots gets caught and loses the room.

### 9.5 Show the Alternatives: The Steel Man

Don't dodge the alternatives. Attack your own idea first. A **Steel Man** is the strongest possible case against your proposal, stated fairly, which you then answer with evidence.

Build two of them. One argues that no standard feature is needed at all, that a library would do. The other argues that a rival design is better, and then you show why yours wins.

### 9.6 Implementation Experience

Nothing beats proof that your idea works. **Implementation experience** means you've built the feature and used it, ideally in more than one place. The committee wants that proof before it freezes anything forever.

The bar is real but not impossible. For a library, ship code that compiles, passes tests, and has real users. For a language feature, a rough compiler fork that shows it working is enough to start.

One small piece rides along with new features: the **feature-test macro**. It's a predefined name that lets code check whether a compiler has your feature yet. Including it lets people adopt the feature safely while support spreads.

### 9.7 Writing Standard Wording

Standard text isn't prose, it's closer to law. Two words carry the weight: **shall** marks a hard requirement, and **should** marks advice. Mixing them up changes what implementers must do.

Point to the right place, too. Cite the stable section labels (the bracketed names from §1.7), not page or section numbers. The numbers shift between drafts. The labels don't.

### 9.8 A Short Tutorial

A great proposal teaches its feature. Include a short tutorial that shows someone how to use it. If you can't explain it in a few clear lines, the design may be too complex.

### 9.9 Scope and Dependencies

Keep each paper to one topic. If your title needs the word "and," split it in two. Big, bundled proposals stall, while small focused ones move.

For a large idea, ship it in stages. Land the useful piece first, let people use it, and build the rest on top. Some features even wait on another paper to land before they can.

### 9.10 The High Bar

The burden of proof sits on you, the proposer. The default is no, because a mistake is forever. Being useful isn't enough, so show why only the standard can deliver what you need.

Language features are the hardest of all. A library lives on GitHub if the committee says no, but a language feature has nowhere else to go. So the committee guards the language even more tightly than the library.

Most papers don't make it, and that's the system working. Big efforts have died over decades, from networking to several number libraries. Go into your own paper skeptical, and let only strong evidence change your mind.

Now you know what a strong proposal carries: examples, alternatives, proof, and a narrow scope. Next, let's actually build and submit the paper.

---

## 10. Producing and Submitting Your Paper

You're ready to write the paper. Here's the practical path - tools, format, and the route from your editor to the committee's agenda.

### 10.1 Float the Idea First

Before you write a full paper, float the idea. Post a short description to [std-proposals](https://lists.isocpp.org/mailman/listinfo.cgi/std-proposals) (the public list from §4.6) and see who bites. The official [submit-a-proposal](https://isocpp.org/std/submit-a-proposal) page walks you through it.

This saves you from writing a paper nobody wants. You'll find prior art, hear objections early, and maybe meet a co-author. A warm reception here is a green light.

Once the idea has interest, you write it up.

### 10.2 Writing the Paper: Tools and Template

You don't format a paper by hand. Most authors use [mpark/wg21](https://github.com/mpark/wg21), which turns Markdown into a committee-styled document, or [Bikeshed](https://github.com/tabatkins/bikeshed), which builds HTML specs. Newer tools like COWEL and plain LaTeX work too.

Start from the **official template**. It's posted on [isocpp.org](https://isocpp.org/std/submit-a-proposal), and the tools above ship their own versions. The template gives you the front matter and the expected sections.

### 10.3 The Formatting Standard: SD-7

**[SD-7](https://isocpp.org/std/standing-documents/sd-7-mailing-procedures-and-how-to-write-papers)** covers the mechanics: how to get a paper number, how to format the document, and how to submit it to a mailing. Read it once before your first paper.

Remember the basics from §3.3 and §3.5. Your paper gets a P-number and a revision, and it has to land before the mailing deadline. A late paper waits for the next round.

However you format it, make sure everyone can read it.

### 10.4 Make It Accessible

Use enough color contrast, a monospace font for code, and clear headings. Don't lean on color alone to make a point.

This isn't busywork. A clear paper reaches the busy delegate skimming on a phone between sessions. Accessibility is reach.

The strongest part of a paper is proof that it runs.

### 10.5 Prototyping: Compiler Explorer and GitHub

Let readers run your code, don't just describe it. **[Compiler Explorer](https://godbolt.org)**, at godbolt.org, runs C++ in the browser across many compilers. Share a live link and the reader sees it work.

Put your implementation on GitHub. A public repo lets people read the code, file issues, and try it themselves. It also tracks your revisions as the design evolves.

Few papers are solo efforts, and none survive unchanged.

### 10.6 Co-Authoring and Revision History

Many papers have more than one author. Co-authors bring expertise, share the load, and lend credibility. List everyone in the front matter, and name one person as the presenter.

Expect to revise, again and again. Each revision (R0, R1, R2) answers the last round of feedback. Reviewers read what changed, so spell out your changes.

Before you present, don't forget one legal step.

### 10.7 Patent Disclosure

ISO has a patent rule, and you must follow it. If you know of a patent that covers your proposal, disclose it before the work advances. The [ISO patent policy](https://www.iso.org/iso-standards-and-patents.html) spells out the duty.

With the paper done and disclosed, you need a slot to present it.

### 10.8 Getting on the Agenda

A paper with no presenter goes nowhere. Email the chair (the person running the room, from §5.10) before the meeting to ask for agenda time. Confirm whether you'll present in person or by video.

When your slot comes, be ready to present and to listen. You'll show the problem, the design, and the proof, then take questions. The room's reaction steers your next revision.

Getting through that first discussion feels like the finish line. It isn't.

### 10.9 The Other 80%

Winning design approval (the "we want this" gate from §8.6) feels like victory. It's the first fifth. Wording review and the national ballots are the other 80%.

After design comes the long part. A wording group polishes the exact text, then the countries vote it through the ballot stages. Plan for years, not months, and don't celebrate too early.

Writing the paper is the start. Getting people to say yes is the real work - and that's the long art of championing, covered next.

---

## 11. Championing Your Paper

A paper doesn't pass itself. Getting to yes is a long, social campaign, and this chapter is your field guide. Newcomers underestimate this part most.

### 11.1 Before the Room: Presocialization

Most papers are won before they're presented. **Presocialization** means talking up your idea in hallways and on the lists, building support ahead of the room. A cold presentation to a surprised room usually stalls.

The decisive moment is the five minutes before a poll. The chair's summary, the first voice to speak, and the room's mood shape the result. Line up a few supporters who'll speak early.

If you can't be in the room, someone else has to carry the paper.

### 11.2 Finding and Being a Champion

Every paper needs a champion (the person who presents and pushes it, from §2.8). Often that's you, the author, but it doesn't have to be. If you can't attend, find someone who knows the work deeply, not just someone to read it aloud.

You can also champion someone else's paper. Carrying a good idea you didn't write builds trust and skill. It's one of the fastest ways for a newcomer to become useful.

### 11.3 The First Gate: "Do We Want It at All?"

The first discussion of your paper is the scary one. The room asks "do we want this at all?" before it touches any detail. A no here kills the idea outright.

Frame your opening as solving a problem the committee already cares about. Tie it to the [Direction Group](https://wg21.link/p0939) priorities from §5.9. A paper that matches the agenda gets heard. One that doesn't waits.

Even a wanted paper will take fire.

### 11.4 When the Poll Goes Against You

A poll against you is not a death sentence. Being sent back for more work is the normal path, not a rejection. You take the feedback, revise, and return with a stronger paper.

Know the one risky move: reopening a settled question. **Re-litigation** is re-debating something the room already decided, and it costs you credibility. Reopen only with genuinely new evidence, not the same argument again.

When you're stuck, the way forward is often a smaller ask.

### 11.5 Negotiating: Back Pocket and Max-Min

Always carry a fallback. A **back pocket alternative** is a second design you're ready to offer if your first choice is rejected. Walking in with one keeps you from leaving empty-handed.

When the room is split, shrink the ask. The **max-min solution** is the smallest version everyone can accept. A small win that ships beats a big proposal that stalls.

### 11.6 Disagreement vs Opposition, and When to Withdraw

Learn to tell two things apart. Disagreement is a technical objection, usually with a paper behind it. Opposition is a person set against your idea, and the two need different responses.

Sometimes the right move is to stop. If the room clearly doesn't want it and won't budge, withdraw the paper with grace. Pouring energy into a dead idea costs you the credibility you'll need later.

### 11.7 How Reputation Works

In a room of volunteers, reputation is what gets your paper read. People can't review every proposal, so they weigh who wrote it. A trusted name gets the benefit of the doubt, while a new one gets extra scrutiny.

You build it in small steps. Start with modest, useful contributions, become the reliable expert in one narrow area, and use your credibility only on the battles that matter. Reputation compounds when you're right over time.

### 11.8 The Presenter Is Judged Too

The committee weighs the presenter, not only the proposal. Are you reasonable, willing to concede a fair point, someone people want to work with? Those signals shape how your paper lands.

Don't be the obstacle. Causing needless delay, or escalating every dispute, marks you as hard to work with. The formal objection tools exist, but reaching for them often erodes your credibility.

Persistence does pay, though. **Procedural momentum** is the benefit a paper earns from many revisions and prior polls. By the fourth or fifth revision, the room tends to assume it'll pass, so steady iteration works in your favor.

### 11.9 The Long Game

Set your clock to years, not months. The train runs every three years, and most features ride more than one. Patience isn't optional here. It's the job.

The famous features all took ages. Coroutines, modules, and the executors work that became [std::execution](https://wg21.link/p2300) each ran many years and many revisions. Stackful coroutines have waited over a decade and still aren't in.

### 11.10 The Structural Headwinds

Some headwinds aren't about you. The **bandwidth problem** is plain: far more papers arrive than anyone can review. The committee puts out 300 to 500 papers a year, and the median delegate reads a few dozen.

Another is the **expert bubble**. Papers are often written by experts for experts, which makes it hard for newcomers and outside ideas to break in. The review system is one crack in that wall, so use it.

Resources tilt the field, too. Proposals with paid committee time and funded implementations move faster than volunteer efforts. It's not a conspiracy, just the weight of who can afford to show up every cycle.

Those forces wear people down, and that brings us to the last thing worth protecting.

### 11.11 The Emotional Side

Having your paper picked apart stings, and that's normal. Don't take it personally. The room argues hard about ideas, not about you.

Burnout is real here, not a weakness. Surveys show most delegates feel the pace is too much, and good people leave because of it. Set boundaries, pace yourself, and know when to step back.

Protecting your energy is protecting your impact. The committee runs for decades, and it needs you for the long haul, not for one heroic burst. Pace beats sprint.

Now you can champion a paper: socialize it, frame it, defend it, and survive the years it takes. Some proposals die for reasons no campaign can fix. Next, the deep design rules that decide what's even possible.

---

## 12. C++ Design Principles

By now you've seen good ideas stall. The reason often isn't politics, it's a handful of deep design rules that decide what C++ can even accept. This chapter names them.

### 12.1 Standardize Existing Practice

The committee's safest path is to **standardize existing practice**. That means blessing a design that already works in the field, instead of inventing something new at the table. A design with years of real use behind it clears the bar far faster.

The pattern shows up again and again. The fmt library became `std::format`, and range-v3 became `std::ranges`, each after proving itself in the wild. Designs invented at the table carry far more risk.

Why so cautious? Because the standard can't easily fix its mistakes.

### 12.2 Backward Compatibility

C++ keeps a hard promise: old code keeps working. This is **backward compatibility**, and it carries the weight of decades of existing programs. A change that breaks working code faces enormous resistance.

This is also why the committee won't ship two versions of the same type. A second, "safer" vector would split the language, breaking code that passes one kind where the other is expected. So one type has to serve everyone.

### 12.3 The Zero-Overhead Principle

C++ lives by the **zero-overhead principle**. It has two halves: you don't pay for what you don't use, and what you do use runs as fast as hand-written code. A feature that taxes people who never touch it tends to fail.

This shapes every proposal. If your feature slows down code that doesn't use it, expect to defend that cost in writing. The committee protects the people who came for speed.

### 12.4 ABI: The Invisible Constraint

The constraint that kills the most proposals is **ABI**, the application binary interface. ABI is the binary contract between compiled pieces of a program: the memory layout of types and the way functions are called. Change it, and programs already built stop working until they're rebuilt.

This is why some slow types never get faster. Speeding up `std::regex` or `std::unordered_map` would change their layout and break every binary that uses them. So the committee keeps the slow version, and the market routes around it.

The committee faced this head-on at the **Prague ABI vote** in early 2020. It chose not to break ABI across the library for C++23, while refusing to promise stability forever. In practice, that left existing binaries safe and the frozen types frozen.

### 12.5 Freestanding vs Hosted

C++ runs in two very different worlds, and the standard names them. A **hosted** environment has an operating system and the full library behind it. A **freestanding** environment has neither, like a microcontroller, a kernel, or a bootloader.

This split shapes what a library feature can assume. Something that needs files or threads won't work freestanding, where no operating system provides them. A proposal that forgets the freestanding world will hear about it from the people who live there.

These constraints explain the quiet noes: an idea can be right and still impossible. Knowing them saves you from proposing the unbuildable.

---

## 13. Common Mistakes

You've made it far enough to be dangerous. This chapter lists the traps that catch newcomers, each with its fix. Read it once now, and again before your first meeting.

### 13.1 Showing Up Without Announcing

The first trap is simply appearing. The convener controls access, so guests give about a week's notice and members register ahead. Email the convener before you go, and catch the Sunday orientation.

### 13.2 Proposing Core Language Features Too Early

Bringing a language feature before you know the room is a hard road. Core changes face the steepest bar of all (the language hard mode from §9.10), and a cold first attempt usually fails. Watch how the room works first, socialize the idea, and bring evidence.

### 13.3 Writing an Idea-Only Paper

An idea with no examples, alternatives, or proof wastes the room's time. The bar is cultural, not enforced, but thin papers still go nowhere. Bring worked examples and an implementation, even on your first paper (see §9.6).

### 13.4 Voting on Things You Haven't Followed

Voting with the room on a paper you didn't read is a quiet mistake. Your passive vote still counts under consensus, so it muddies the signal. When you haven't followed a topic, abstain (the honest non-vote from §8.5) instead of guessing.

### 13.5 Quoting Reflectors or Notes Publicly

Pasting a reflector post or meeting note into a blog breaks the rules. Those are private under the confidentiality rule (from §7.5), and quoting them publicly is a real breach. Paraphrase instead, and share only poll numbers freely.

### 13.6 Raising Concerns Too Late

Saving your objection for the closing plenary gains nothing. Concerns belong in the subgroup, early, where the work happens. Raise it in the room, bring a paper, and beat the deadline the evening before plenary.

### 13.7 Expecting Majority Rule

Counting hands and expecting the bigger number to win is wrong here. The committee runs on consensus, the absence of sustained opposition (from §8.1), not majority rule. A determined minority can block, and a majority alone can't force a yes.

### 13.8 Talking Too Much Too Soon

Speaking constantly at your first meetings works against you. Newcomers build trust by learning the room before filling it with their voice. Listen first, contribute something specific, and let your work speak.

### 13.9 Misreading Encouragement as Approval

Hearing "come back with more" and reading it as "this will ship" sets you up for a fall. An early warm poll is direction to keep going, not adoption (the encouragement-is-not-approval point from §8.6). Real approval comes later, at plenary, after wording.

### 13.10 Ignoring Subgroup Direction

Getting direction from a room and then ignoring it stalls your paper. The design rooms exist to set direction, and working against it keeps the opposition alive. Follow the guidance, or reopen it in that same room with a paper, but don't route around it.

### 13.11 Declaring Victory at Design Approval

Treating design approval as the finish line is the last big trap. It's an early milestone, with wording review and the national ballots still ahead (the other 80% from §10.9). Budget for the long tail, and don't celebrate until it ships.

One mistake underlies them all: confusing being busy with making a difference. Showing up and passing polls can feel like impact, but a feature that never ships helped nobody. Keep your eyes on the people who use C++.

That's the end of the notebook, and the start of your part in it. Keep your wits about you, demand evidence, and put the users first. Now go do it well.
