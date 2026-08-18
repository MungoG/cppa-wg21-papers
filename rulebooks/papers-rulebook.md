# Rulebook: Evaluating Edits to a WG21 Paper

Rules for judging a human editor's changes to a model-generated WG21 paper. Give this document to a model along with the original paper and the editor's diff. The paper under review is the **target**; the editor's changes are the **diff**; your judgment of each change is the **verdict**. Two rules bind everything below:

1. Judge every edit better, worse, or neutral on four axes: grammar, clarity, register, and alignment with the paper's thesis. Alignment outranks the other three: a grammatically perfect edit that weakens the thesis is worse.
2. Always prefer brevity, all else equal. An edit that cuts words without losing content is better; an edit that adds words must add content.

## 1. Protocol

1. Read the whole target before judging anything. Extract its thesis - the finding it states or the ask it makes - and its intent (ask or info) from the front matter.
2. Judge each edit in the diff against the four axes and the rules below, one edit at a time.
3. Quote both versions for every non-neutral verdict, name the axis and the rule, and give one line of reason.
4. The editor's judgment is final. An edit that deliberately breaks a rule is an override: report it, do not fight it.
5. Judge grammar and clarity with your own competence. Mechanical prose rules - dashes, contractions, paragraph length, idiom rates - are another rulebook's domain; do not grade them.

## 2. The Reader

- The reader has two hundred papers and reads three gates: title, abstract, conclusion. A reader who fails a gate never opens the body. An edit that weakens a gate costs more than the same edit in a body section.
- The reader who draws a conclusion owns it. An edit that states a verdict the evidence already implies is worse: a verdict from the author is a claim to be challenged; a verdict from the reader is a conviction.
- The most important reader arrives cold, years later. An edit that assumes insider context - unexpanded abbreviations, undated meetings, bare paper numbers - is worse.

## 3. Invariants

1. NEVER quote or traceably paraphrase private committee records: reflector posts, committee wiki pages, private meeting minutes.
2. NEVER let an edit fabricate or embellish evidence: quoted text stays byte-identical, and every qualifier that keeps a claim true survives the edit.
3. The finding is ALWAYS stated in full on the surface - the abstract and the conclusion - and nowhere else; each body section states only its own local conclusion.

## 4. Protected Elements

Removal or weakening of any of these is presumptively worse; the edit must visibly earn its way:

- "This paper asks for nothing." (info papers). Deleting it is the canonical worse edit.
- The posture line: "The author provides information and serves at the pleasure of the committee."
- The machine-assistance statement.
- The abstract's finding line and each section's closing local conclusion.
- The disclosure's admissions: affiliations, competing work, the genuine limitation.
- Citations and their reference entries.
- The ask, restated so a conclusion-only reader can vote (ask papers).

## 5. Posture

- Serve, do not petition. An edit that adds a request to an info paper - floor time, scheduling, "we hope the committee will" - is worse.
- An edit that adds urgency aimed at the reader is worse: a date is a fact; a deadline pressed on the reader is pressure.
- An edit that adds pleading, anxiety, or credit-seeking ("we hope this is useful", "the first paper to", "deserves careful consideration") is worse.
- No "should", "must", or "ought" aimed at the committee, a subgroup, a chair, or an officer. The replacement is an observation: "the committee should revisit X" becomes "the conditions that produced X have changed".
- Ask papers: the ask stays votable, gains and costs stay named, and objections stay stated in their strongest form. An edit that softens an objection is worse.
- The paper stops when the evidence stops. An edit that adds a summary, a call to action, or an inspirational close is worse.

## 6. Evidence

- An unsourced claim is an opinion. An edit that adds a factual claim without a source is worse; an edit that removes a source is worse.
- An edit that orphans a citation, drops a qualifier, strengthens a claim past its source, or removes the disclosed method behind an absence claim is worse.
- Show, do not tell: side-by-side code, tables whose columns carry the argument, empty cells left empty. An edit that replaces shown evidence with an evaluative adjective ("simpler", "verbose") is worse - the reader can count.
- Evidence escalates from simplest to most complex, each step adding one dimension. An edit that leads with the hardest case is worse.
- Attribute with neutral verbs: "writes", "observed", "noted" - never "admitted", "conceded", "revealed".

## 7. Fairness

- Praise precedes analysis and is specific: the design's named properties, stated in attestation verbs ("provides", "enables"). An edit that removes earned recognition or makes it perfunctory is worse.
- Every cost is attached to a design or a mechanism, never to a person, and paired with what it provides. An edit that names a cost without its benefit, or attaches a cost to a person, is worse.
- Concessions stand plain, with no "however" softening. An edit that hedges a conceded limitation is worse.
- Directions are a menu, not a recommendation: options stated with their properties, unranked.

## 8. Voice

- The author is third person: "the author", never "I" or "we" outside quotations.
- Never name the ghost: an edit that adds a defensive negation or disclaimer ("this is not an attack", "the evidence is public") is worse - the negation plants the accusation it denies.
- Verbs measure, they do not dramatize: "produces", "exhibits", "reduces to" over "breaks", "collapses", "deepens".
- No vague quantifiers: an edit that adds "some", "many", "various", "several", "often", or "widely" without naming the items is worse.
- A word fails by mechanism: side-label, intent-load, motive attribution, diminish, dramatize, conspiratorial frame, patronize, innuendo, delegitimize. An edit that introduces one is worse.

## 9. Structure and Genre

- Headings state each section's point; an edit that flattens a heading into a topic label is worse.
- Every section opens stating what it covers and closes with its own local conclusion.
- Every contributor is thanked with their specific contribution; "helpful feedback" is a form letter.
- The disclosure sits after the conclusion and before acknowledgments and references, in every genre.
- Two genres are legitimate: the formal paper (canonical sections, quoted-objection headings, poll blocks) and the personal posture essay. Do not flag a genre-legitimate deviation.

## 10. Verdict Format

One line per edit, then a summary:

```
[edit location] better|worse|neutral - [axis] - [rule] - [one-line reason]
Summary: the diff is a net improvement|regression|wash - [one sentence]
```

## Checklist

- Every non-neutral verdict quotes both versions and names its axis and rule. (1)
- No verdict grades mechanical prose: dashes, contractions, paragraph length, idiom rates. (1)
- Every protected element survives, or its removal is flagged. (4)
- No worse verdict rests on a genre-legitimate deviation. (9)
- The summary names the diff's net direction in one sentence. (10)

Restated: judge every edit better, worse, or neutral - alignment first, brevity always - and report overrides without fighting them.

*2026-08-18 - Kimi K3 (Cursor agent)*
