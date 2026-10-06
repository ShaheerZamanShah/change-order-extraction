# Change order extraction: how I approached it and where it breaks

## The problem as I understood it

Change orders are the paperwork that modifies a construction contract. Somebody needs the numbers out of them, in a database, with the money adding up. The documents themselves are all over the place. Some are tidy AIA forms. Some are a scan of a fax of a form. Some are an email that says "pricing came back at 48,920 all in" and nothing else. A few are a superintendent's handwriting typed up by whoever could read it.

So the real problem is not parsing a change order. It is parsing a change order when you do not know which kind you were handed, and knowing afterwards whether to trust what you got.

That second half shaped everything. A wrong contract sum that arrives looking confident is worse than no answer, because somebody will pay against it. I decided early that the output had to say how sure it was, per field, and that the confidence had to be earned from something other than the fact that a regex matched.

## The thing change orders give you for free

Change orders are redundant on purpose. The same money shows up three or four times in different roles, and it has to reconcile:

```
contract sum before this change = original sum + net previous changes
new contract sum                = contract sum before + this change
sum of the line items           = this change
revised completion date         = original completion date + days added
```

That redundancy is the most useful thing in the document, and it is free. I do not need a reference database or a second document to tell whether a number was misread. If five numbers are supposed to add up and they do, something is very likely right about all five. If they do not, something is wrong with at least one of them and I can say so without knowing which.

So the pipeline is built around those identities rather than treating them as a validation step bolted on at the end. Validation is where confidence comes from.

## How it runs

Seven stages, each one testable on its own.

**Ingest.** Text files, PDFs with a text layer, scans through OCR. One thing here turned out to matter a lot more than I expected. A PDF text layer has no idea what a column is, so `extract_text` gives you a single space between two words whether they sit next to each other in a sentence or in different columns of a cost breakdown. That one space destroys exactly the signal the table reader needs. I rebuild the page text from word positions instead and insert a wide gap when the distance between two words is large relative to the character width on that line. Before that change, line items came out of my text samples and vanished from the identical PDFs. After it, both paths give the same answer.

Every document also gets a text quality score, and that score caps the confidence of everything read from it. A bad scan should not be able to produce a 0.95.

**Normalise.** Ligatures, hyphenated line breaks, page furniture that repeats on every page, and the OCR confusions. `l,85O.OO` becomes `1,850.00`. `S2,480.00` becomes `$2,480.00`. `Pr0ject Name` becomes `Project Name`. That last one is small and cost me a whole document before I caught it: one digit inside a caption means every pattern for the project name and project number stops matching, and you lose both fields silently.

**Extract.** Three extractors, none of which decide anything. They propose candidates with a local score and the span they came from. There is a rule extractor driven by a field registry, a table reader for line items, and an optional LLM for narrative documents.

**Arbitrate.** Candidates get clustered by value. A cluster that several independent patterns agree on gets a bonus, because independent agreement is the cheapest reliable signal available. Then the gap between the best cluster and the runner up decides how much of that score survives. Two plausible values with near equal support means I genuinely do not know, and the confidence should say that rather than picking one and looking certain.

**Repair.** If four of the five money fields are present, the fifth follows. I fill it in, mark it as derived rather than read, and cap its confidence. A derived value is useful but it is not evidence.

**Validate.** The identities above, plus date ordering, the day count against the completion dates, and plausibility checks like a change that exceeds the contract it modifies.

**Score and route.** Validation feeds back into per field confidence in both directions. A field that took part in a check that balanced goes up. A field named in a check that failed goes down. Then the document is routed to auto accept, review, or reject.

Two details in there that I think are the difference between this working and this looking like it works.

When a check fails I usually cannot tell which of the three numbers is wrong, so every field named in the issue takes the penalty. All three go to review. The honest statement is "one of these is wrong" and sending all three to a human is the correct response to it.

A derived value is never allowed to corroborate the numbers it was derived from. It was computed to satisfy that identity, so of course it satisfies it. Letting it hand confidence back would be circular and would make a half read document look fully verified. That one is easy to get wrong and hard to notice afterwards.

## Where it stands

Twelve sample documents with ground truth I wrote by reading them, not by copying output. 192 stated values graded: 191 correct, 1 missed, 0 wrong, 0 spurious. All twelve line item totals match. Identical results from the text files and from the rendered PDFs. 150 tests.

The number I actually care about is the last row of the results table in the README: zero wrong fields inside the documents that were auto accepted. Precision is easy to inflate by reporting less. Being right about what you let through unchecked is the part that decides whether anyone can use this.

## Failure modes

These are the ones I know about. Some I fixed, some I decided to live with, and some I can only point at.

**No label, no value.** The rule extractor needs a cue. On the email sample it finds the amounts through narrative phrasing like "the contract started at", but it misses the contractor name because the only place it appears is a signature block. That is the one remaining miss in the eval. It is also the whole reason the LLM extractor exists.

**OCR errors that stay plausible.** My repairs only fire when the result is unambiguously a number, which is deliberate. The dangerous OCR failure is not the garbled one, it is the clean one: a 3 read as an 8 produces a perfectly reasonable amount and no check catches it unless it breaks an identity. If the same digit is misread consistently across the document, the identities balance and everything looks fine. This is why OCR output is never auto accepted regardless of score. It is a policy, not a fix, and I do not have a better answer.

**Values that are consistent and wrong.** If a document is internally coherent but somebody typed the wrong number into it in the first place, every check passes and I report it with high confidence. The deliberately inconsistent sample is caught because it contradicts itself. A document that is wrong and consistent is invisible to me, and nothing inside the document can fix that.

**Ambiguous number and date formats.** `1.500` is either 1500 or 1.5 and the document often does not say. `03/04/2024` is March or April depending on who typed it. I pick the more likely reading, tag the value as ambiguous, and drop its confidence so it surfaces in review rather than passing quietly. On a single document mixing both conventions I will get some of them wrong.

**Tables that are not tables.** A change order log looks exactly like a cost breakdown once you split on whitespace: a reference, a description, an amount. My sample with a log in it was happily summing the log and reporting it as the breakdown. It is excluded now by noticing that breakdown rows do not carry dates and that log headers mention status, but this is pattern matching on conventions and a layout I have not seen will get through. The same applies to the contract summary block, which also looks like a list of line items.

**An LLM that invents a quote.** Requiring the model to quote the document is not enough by itself, and my own test caught this. A model that invents an amount invents a quote containing it, so the two agree perfectly, and the invented quote is usually a near copy of a real line so it fuzzy matches that line and arrives with real looking evidence attached. The check now verifies the digits against the actual document text near where the quote resolved, not against the quote. If I had trusted the quote I would have shipped a hole.

**Catastrophic backtracking.** Two of my patterns opened with the money group, which the engine has to attempt at every character in the document. Twelve short samples took forty seconds. Anchoring those patterns on a leading word brought it to 188 milliseconds. There is a test with a loose time bound now, because this is the kind of mistake that looks fine on a sample and falls over on a real batch.

**Tolerance that scales away the error.** My first version compared amounts with a relative tolerance, which on a 1.9 million dollar contract allowed a 9,500 dollar discrepancy. It silently passed a document built to fail. Tolerance is now capped in absolute terms, because a three thousand dollar error is still a three thousand dollar error on a fifty million dollar contract.

**Confidence is conservative, not calibrated.** Every reported value in this corpus was correct, so observed accuracy sits at 1.0 in all five confidence bins while mean confidence runs from 0.35 to 0.95. The expected calibration error is 0.116 and all of it is in the safe direction. I could close that gap by raising the constants, and it would mean nothing, because twelve clean documents cannot tell me where the thresholds belong. The harness prints the reliability table so this can be redone properly on a real corpus, which is the only place the numbers would be worth anything.

## What I would do next

Calibrate on real documents first, a few hundred with human corrections, and refit the thresholds against that instead of my constants. The harness already reports what is needed.

Use the review corrections. Every time a human fixes a field, that is a labelled example of a cue the rules missed, and it should turn into either a new alias or a weight change rather than sitting in a log.

Keep a per project memory. The same owner, contractor and architect appear on every change order for a job, and the numbering is sequential. A new change order that disagrees with the last ten on the same project is worth flagging even when it is internally consistent, and that catches the consistent and wrong case that nothing inside a single document can.

Track field level agreement between the rules and the LLM as a signal in its own right. Right now agreement earns a bonus. With enough volume it would be worth measuring which disagreements actually predict errors.
