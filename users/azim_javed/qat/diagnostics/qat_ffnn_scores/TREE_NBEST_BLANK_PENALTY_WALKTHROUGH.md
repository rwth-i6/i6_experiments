# Tree N-best Blank Penalty: Line-by-Line Code Walkthrough

Read this beside [`tree_nbest_blank_penalty.py`](tree_nbest_blank_penalty.py).
Line numbers refer to its **641-line version inspected on 2026-10-27 after cleanup**. Function
names remain the anchors if later edits move the lines.

This follows source order and explains every executable statement. Multiline
expressions, declarations, and blank lines are grouped where they do not add a
separate operation. The final sections explain the upstream collection choices,
the decisions recorded in OpenCode, and what the result does **not** establish.

## 1. The whole method in simple words

1. Run ordinary tree search and save several alternative hypotheses per utterance.
2. For each saved candidate, recover its cost, its collection penalty, its number
   of blank emissions, and its word errors against the reference.
3. Calculate how that candidate's saved cost would change with the blank penalty.
4. Find exactly where the cheapest candidate changes for each utterance.
5. Combine those changes across the corpus and find the penalty intervals with
   the fewest total word errors.
6. Choose a usable number inside an optimal interval.
7. Test that number with **fresh decoding**, which can discover different paths
   and produces fresh hardware readout noise.

The optimizer in this file performs steps 2–6. It does not run steps 1 or 7.

### Vocabulary and signs

| Name | Meaning |
|---|---|
| `T`, `frames` | Number of encoder frames, not raw audio samples |
| `U`, `tokens` | Number of emitted nonblank **model tokens** on a complete candidate |
| `B`, `blanks` | Blank emissions: `T - U`, under the checked RNA topology |
| `a`, `anchor` | Blank penalty used when collecting that candidate |
| `C(a)` | Candidate's exported cost at that anchor, with the LM scale corrected when needed |
| `b`, `penalty` | Proposed new blank penalty |
| `intercept` | `C(a) - a * B`: the same observation with its additive blank term removed |
| `rank` | Stable candidate order, used only after cost and blank-count ties |
| `edits` | `(substitutions, deletions, insertions)` against the reference |
| `lower envelope` | The cheapest candidate cost at each penalty |
| `oracle WER` | WER if we could independently choose the fewest-error candidate for every utterance |

**Costs are minimized.** A positive blank penalty adds to blank cost; it does not
reward blanks. With unit scorer scale and an additive post-softmax penalty:

```text
C(b) = C(a) + (b - a) * B
     = intercept + b * B
```

Each candidate is therefore a straight line. Its slope is its blank count.
Increasing `b` favors candidates with fewer blanks, hence more model-token
emissions for the same utterance length. This does not guarantee a monotonic
change in word errors: more model tokens is not the same as more correct words.

### A small example used below

Suppose an utterance has two candidates, with saved costs already normalized to
penalty zero:

```text
A: cost(b) = 10 + 8*b; word errors = 2
B: cost(b) = 14 + 6*b; word errors = 1

Crossing: 10 + 8*b = 14 + 6*b
         b = 2

b < 2: A wins
b >= 2: B wins under our fewer-blanks tie rule
```

The reference determines the error counts, **not** which candidate wins a score
tie. The score-selection rule must not peek at the correct answer.

## 2. Lines 1–19: scope and imports

| Lines | What they do and why |
|---|---|
| 1–6 | State the scope: rerank completed standard `nbest.json` output. No model, alignment, decoder, or manager is run. Also disclose that exported costs are not certified complete lattice costs. |
| 8 | `configparser` reads the saved RASR INI configuration. We recover actual collection settings rather than trusting experiment names. |
| 9 | `gzip` allows compressed lexicons. |
| 10 | `json` reads observations and writes reports/candidate files. |
| 11 | `math` checks finite floating-point representatives and steps to neighboring floats. |
| 12 | `ElementTree` reads the XML lexicon incrementally. |
| 13 | `Counter` accumulates rejection reasons; `defaultdict` groups simultaneous corpus switches. |
| 14 | `dataclass` defines compact candidates; `replace` attaches pooled-source identities without modifying frozen instances. |
| 15 | `Fraction` represents decimal costs and crossings as exact rational numbers. |
| 16 | `Path` handles filesystem paths. |
| 17 | `Optional` annotates the possibly absent upper penalty bound in the job interface. |
| 19 | Import Sisyphus jobs, tasks, paths, and output registration. Importing the module does not start a manager. |

## 3. Lines 22–24: `number` — consistent exact arithmetic

| Lines | Explanation |
|---|---|
| 22–23 | Define the shared numeric conversion helper and its purpose. |
| 24 | Convert through `str(value)` into a `Fraction`. For example, the decimal string `"1.44"` becomes exactly `36/25`, not the binary float's full expansion. Let `Fraction` raise its own conversion errors rather than duplicate its validation. |

**Tradeoff:** exact arithmetic makes grouping equal crossings and handling ties
reproducible, but costs more CPU than float arithmetic. It is exact for the
decimal representation supplied to this helper, **not** for an unknowable
original hardware score. The JSON parser below uses ordinary Python numeric
decoding; a JSON floating-point literal has already passed through a float
before this conversion. Thus this is not arbitrary-precision preservation of
every digit in arbitrary JSON input, nor removal of measurement noise.

## 4. Lines 27–91: `iter_nbest` — streaming a standard JSON object

The input is one object keyed by utterance ID:

```json
{
  "utterance-A": {"reference": "...", "encoder_frames": 123, "candidates": []},
  "utterance-B": {"reference": "...", "encoder_frames": 456, "candidates": []}
}
```

The empty candidate lists above only illustrate the outer format; actual records
must contain usable candidates. This is **not** the probe's `nbest.jsonl` format.

| Lines | Explanation |
|---|---|
| 27–32 | Define a generator reading chunks of roughly 1 MiB. Require a complete top-level object rather than accepting an unfinished running-job output. |
| 33–34 | Define an error handler for non-standard JSON constants such as `NaN`. |
| 36 | Construct the JSON decoder with that handler. |
| 37 | Initialize the text buffer, current read position, and end-of-file flag. |
| 39–40 | Define `refill`; `nonlocal` lets it update the outer parser state. |
| 41 | Read another chunk. |
| 42 | Drop the consumed prefix, keep the unconsumed suffix, and append new text. |
| 43–44 | Reset the position and mark EOF only when a read returns no text. |
| 46–47 | Define `peek`, which can advance the shared position. |
| 48–50 | Skip whitespace, repeatedly if necessary. |
| 51–52 | Return the next available non-whitespace character without consuming it. |
| 53–55 | Return `None` at EOF, otherwise refill and try again. |
| 57–61 | `expect` checks and consumes a required punctuation character, failing clearly on invalid or incomplete input. |
| 63–65 | `value` prepares to decode one JSON value at the next non-whitespace position. |
| 66–70 | Decode one value, advance to its end, and return it. |
| 71–74 | A decode failure may mean the value crosses a chunk boundary. Refill until decoding succeeds or EOF proves the input invalid/incomplete. |
| 76–78 | Require `{`, initialize the first-entry flag, and track seen IDs. |
| 79–81 | Read entries until `}`; require commas between them. |
| 82–85 | Decode each utterance ID and reject duplicate IDs rather than silently overwriting records. Trust the recognition output's ID type. |
| 86–88 | Require `:`, decode and yield one record immediately, then require separator handling for subsequent records. Trust the recognition record schema rather than checking its type again. |
| 89–91 | Require the closing brace and reject trailing non-whitespace data. |

**Tradeoff:** standard JSON was explicitly preferred over JSONL/custom formats.
A whole-file `json.load` would retain potentially gigabytes of tracebacks. This
small parser avoids that dependency and memory burden, at the cost of more
parser code. It still holds one complete utterance's raw record, the seen-ID
set, and eventually all compact candidates. It is **not constant-memory**.
Malformed values can also be buffered until EOF before an error is reported.
Records may be yielded before the closing brace is checked, but analysis fails
before returning a final result if the complete file is invalid.

## 5. Lines 94–118: `lexicon_counts` — recover token lengths, not alignments

| Lines | Explanation |
|---|---|
| 94–96 | Define the reader; create a lemma-to-lengths mapping and a sentence-end name set. |
| 97–99 | Open plain or `.gz` XML in binary mode and process completed XML elements incrementally. |
| 100–101 | Ignore end events that are not lemma elements. |
| 102–105 | For every pronunciation, count whitespace-separated model tokens except `<blank>` and `[BLANK]`. Store a **set of lengths**, preserving ambiguity instead of selecting a pronunciation arbitrarily. |
| 106 | Gather orthographic aliases after stripping outer whitespace; empty names are discarded below. |
| 107–109 | Include a lemma's explicit `name` attribute and discard empty names. |
| 110–111 | Associate every alias with all observed pronunciation lengths, including lengths from other lemmas sharing that alias. |
| 112–113 | Recognize explicit sentence-end/boundary special lemmas. |
| 114 | Clear processed lemma contents to reduce retained XML data. |
| 115 | Also recognize conventional sentence-end names if they occur in the lexicon. |
| 116–117 | Require sentence-end names and require all of them to have exactly zero nonblank tokens. Otherwise the completion/count assumption is not established. |
| 118 | Return the length mapping and sentence-end names. |

We need **length**, not token identity. Multiple pronunciations are acceptable
if they all have the same length. Different lengths make the blank count
ambiguous, so affected candidates are rejected later.

Why not count traceback blanks? The inspected word-level tree N-best output had
zero timestamps, and word spans can hide blank emissions. The earlier `MART`
probe had `T=29`, `U=3`, hence `26` blanks, while visible blank spans accounted
for only `19`. Token-level lexicon-free tracebacks are not evidence that tree
N-best tracebacks expose the same internal transitions.

## 6. Lines 121–169: `read_context` — establish when the formula is valid

| Lines | Explanation |
|---|---|
| 121–122 | Define the context reader. Disable INI interpolation and allow repeated declarations, matching saved RASR-style configuration. |
| 123 | Preserve option spelling instead of lowercasing keys automatically. |
| 124–125 | Read the saved configuration. |
| 126–128 | Name the search, scorer, and recognition sections used below. |
| 129–135 | Specify required topology: tree time-synchronous search, RNA tree builder, monophone tying, one state per phone, one state repetition. |
| 136–138 | Check every requirement; reject missing or different settings. |
| 139–145 | Reject enabled repeated-label collapse, pre-softmax blank penalty, and vertical label transitions. Missing options default to disabled. |
| 146–147 | Reject a multiple-label-scorer declaration rather than guessing how its scores combine. |
| 148–149 | Require the fixed-context Python scorer, cached or uncached. |
| 150–152 | Require unit scale in both scorer and recognition sections. |
| 153 | Read external LM scale from `[lib-rasr.lm] scale`, falling back to `[lib-rasr] lm.scale`, then to `1`. |
| 154–156 | Read the lexicon path and require it to be absolute. This avoids resolving it against the optimizer job's unrelated working directory. |
| 157–158 | Parse each distinct lexicon path once and cache its counts. |
| 159 | Read the actual collection blank penalty, defaulting to zero. |
| 160–168 | Record provenance: configuration/lexicon paths, collection penalty, LM/ILM scales, fallback setting, and joint-layer hardware settings. Metadata uses floats/strings for JSON reporting. |
| 169 | Return configuration, exact anchor, cached lexicon information, and metadata. |

### Why these restrictions matter

Under this RNA topology, every encoder frame contributes either one nonblank
emission or a blank. A complete candidate with `U` nonblank model tokens therefore
has exactly `T-U` blanks. Collapsed repeats, other state structures, or vertical
transitions can break this accounting.

A **pre-softmax** penalty would also change the softmax normalization and hence
nonblank costs. The simple additive straight line would no longer describe it.
Nonunit scorer scales could be supported with carefully derived coefficients,
but this implementation deliberately rejects them rather than applying `B`
without checking the scale convention.

The messages saying “V1” mean **version one of this diagnostic**, not OpenCode V1.
LM and fixed ILM are now allowed; earlier zero-LM/zero-ILM restrictions were
relaxed after checking the exported components. These guards establish the
supported configuration, not compatibility with arbitrary RASR setups. Defaults
and explicit options are checked as written, not through a general RASR config
resolution engine.

## 7. Lines 172–192: `check_fallback` — completion must be auditable

| Lines | Explanation |
|---|---|
| 172–174 | Select contexts where sentence-end fallback is enabled. |
| 175–176 | If none enable it, return an audit record saying a log audit was unnecessary. |
| 177–180 | For enabled contexts, require both warning and log channels to point to the supplied log's filename. Otherwise absence of messages in that file would not establish anything. |
| 181–185 | Scan for either known diagnostic indicating fallback use; count diagnostic messages. |
| 186–191 | Reject the source if fallback occurred. An unfinished word can contain emitted tokens omitted from the word traceback, invalidating `T-U`. Recommend collecting with fallback disabled. |
| 192 | Return a successful audit with the log path and zero events. |

**Tradeoff:** require evidence that fallback was unused rather than rejecting
every already-collected run whose configuration merely enabled it. This lets us
reuse valid expensive recognition output. Conversely, even one fallback event
rejects the source: the code does not guess which hidden tokens occurred or
quietly drop the affected utterances. The audit relies on these known log
messages and the saved log being complete; the channel check compares filenames,
not independently verified log contents or provenance.

## 8. Lines 195–210: `edit_counts` — reference supervision

| Lines | Explanation |
|---|---|
| 195–197 | Split reference and hypothesis into whitespace-delimited words. There is no extra case folding, punctuation normalization, or Sclite processing. |
| 198 | Initialize edit costs from an empty reference to every hypothesis prefix: all insertions. Each cell stores `(S,D,I)`, not just a scalar distance. |
| 199–200 | For each reference word, initialize the empty-hypothesis cell: all deletions. |
| 201–203 | Form the diagonal transition: match if equal, otherwise add one substitution. |
| 204–205 | Form deletion of the current reference word from the previous row. |
| 206–207 | Form insertion of the current hypothesis word from the current row. |
| 208 | Choose the smallest total edit count. Python's stable `min` picks the first equal-cost option: diagonal, then deletion, then insertion. |
| 209–210 | Keep only the latest row and return the complete-string edit tuple. |

**Tradeoff:** this is transparent CPU word Levenshtein scoring without another
scoring job/dependency. Its WER is explicitly **not Sclite WER**. Different
normalization or alignment conventions can change results. Equal-distance edit
alignments can change the S/D/I breakdown, though not the total distance.

The reference is used to evaluate candidate errors and tune the corpus penalty.
It does not generate histories, add candidates, or break candidate score ties.

## 9. Lines 213–239: `Candidate` — the compact cost-line representation

| Lines | Explanation |
|---|---|
| 213–214 | Define a frozen dataclass so candidate observations are not changed in place. |
| 215–219 | Store stable rank, transcript, exact zero-penalty intercept, blank count, and word-edit tuple. |
| 220–221 | Reserve source index and original export rank, initially zero until pooling assigns provenance. |
| 223–225 | Expose total word errors as the sum of S/D/I. |
| 227–228 | Evaluate the candidate line at any supplied penalty. |
| 230–239 | Serialize identity, transcript, provenance, blank count, errors, and intercept. Save the intercept both as a convenient float and an exact rational string. |

The huge traceback is deliberately absent. After obtaining `B` and the terminal
cost, it is unnecessary for the analytical sweep. Original recognition outputs
remain the source for tracing how those quantities were extracted.

## 10. Lines 242–275: `prepare_candidates` — turn observations into lines

| Lines | Explanation |
|---|---|
| 242–245 | Define preparation and read encoder length and reference. Trust the schema produced by recognition rather than checking these fields' Python types again. |
| 246–248 | Initialize usable candidates and rejection counts, then enumerate exports starting at rank one. Candidate-specific validation is inside `try`. |
| 249–251 | Require a nonempty traceback ending in a recognized sentence-end lemma. A merely plausible transcript is not enough to establish completion. |
| 252–254 | For nonzero external LM scale, reconstruct the total from the final cumulative fields as `AM + lm_scale * LM`. ILM correction is already in AM. |
| 255–256 | Otherwise convert the saved terminal cost. The LM branch does not redundantly parse a terminal cost it will not use. |
| 257–262 | Look up every traceback item's lemma, require exactly one possible token length, and sum those lengths. Include every trace item; blank and sentence-end items must resolve to zero-length entries. |
| 263 | Read the saved transcript. This function neither rechecks its type nor reconstructs it from lemmas. |
| 264–266 | Infer `blanks = frames - tokens` and reject negative counts: more model-token emissions than encoder frames is impossible under the checked RNA topology. Zero blanks are allowed. |
| 267–270 | Construct the candidate with `intercept = cost - anchor * blanks`, and calculate its word edits against the reference. |
| 271–272 | Count `ValueError` rejection reasons and continue with other alternatives. Missing keys and type errors now fail directly instead of being silently treated as candidate rejections. |
| 273–274 | Fail if no candidate survives, rather than removing that utterance and changing the corpus denominator. |
| 275 | Return surviving candidates and all rejection counts. |

### The LM correction is intentional, not double counting

The working-session source inspection found a distinction:

```text
Direct-best traceback: LM component already scaled
N-best traceback:      raw LM component exported, scale stored separately

Correct N-best cost = final_am_score + configured_lm_scale * final_lm_score
```

The upstream saved `traceback_terminal_cost` simply adds AM and LM. That is not
the intended total for a nonunit LM scale. The minimal fix was to correct it
here, using the separately saved fields, rather than change all recognition
outputs. Fixed ILM must **not** be subtracted a second time.

For zero LM scale, this implementation keeps `traceback_terminal_cost`; it does
not execute the reconstruction branch. That relies on the supported no-LM
export convention, not on a general proof that arbitrary raw LM fields vanish.
Sentence-end scoring was found on the explicit sentence-end trace item in the
inspected setup. The broader epsilon/final-cost export caveat still remains.

## 11. Lines 278–299: `lower_envelope` — find only meaningful crossings

| Lines | Explanation |
|---|---|
| 278–283 | Define the output as `(inclusive_start, candidate)` pairs. `None` means minus infinity for the first interval. State the tie rule: fewer blanks, then earlier rank. |
| 284–288 | Group by blank count. Equal slopes are parallel: retain only the lowest intercept, breaking equal-intercept ties by earlier rank. The others cannot win. |
| 289–290 | Start an empty hull and process lines from largest slope to smallest. As penalty increases, the winning slope can only decrease. |
| 291 | Initialize this line's entry point. |
| 292–294 | Compare with the last retained line and calculate their exact crossing: `(new_intercept - old_intercept) / (old_blanks - new_blanks)`. Slopes differ after grouping, so the denominator is nonzero. |
| 295–296 | Keep that crossing if the previous line has a nonempty winning interval before it; the first hull line begins at minus infinity. |
| 297 | Otherwise remove the previous line: the new line overtakes it before it would ever become the selected winner. Repeat against the shortened hull. |
| 298 | Append the new line and its entry point, or `None` if it is now the first line. |
| 299 | Return the ordered winning intervals. |

In the two-line example, this returns `[(None, A), (2, B)]`.
With three lines meeting at the same point, the `start > previous_start` test
removes zero-width intermediate winners. At the crossing, the fewer-blanks line
wins, so intervals are **right-continuous**: `[start, next_start)`.

**Tradeoff:** a sorted hull avoids enumerating every pairwise intersection and
does not approximate with a hand-chosen grid. For `K` candidates, sorting costs
roughly `K log K` comparisons and hull construction performs a linear number of
push/pop operations; rational arithmetic adds its own cost. Importantly, lines
are discarded based only on costs/counts/rank, never on reference errors.

## 12. Lines 302–309: direct selection and corpus reporting

| Lines | Explanation |
|---|---|
| 302–303 | `winner` directly minimizes `(cost, blanks, rank)` at a specified penalty. This agrees with the hull's tie convention and is used for comparison points and saved hypotheses. |
| 306–309 | `error_record` reports total errors, individual S/D/I, and `100 * errors / reference_words`. The caller establishes a nonzero corpus denominator. |

**Why fewer blanks at a tie?** It is reference-free and agrees with the winner
immediately to the right of a crossing. It makes interval handling simple and
consistent. It is our reranker's deterministic convention, **not** a claim about
RASR's internal equal-score tie breaking. Identical lines use export order,
which ultimately makes pooled input order observable at exact ties.

## 13. Lines 312–330: `representative` — choose a float inside an interval

| Lines | Explanation |
|---|---|
| 312–314 | If the interval is a single allowed point, that point is the ideal representative. |
| 315–316 | Otherwise keep the preferred penalty only if it is strictly inside the interval. |
| 317–318 | Use the midpoint of a finite interval, or `lower + 1` for an unbounded interval. These avoid choosing a switching boundary by default. |
| 319–321 | Convert to a normal float for downstream decoding. Return `None` if it is not finite. |
| 322–323 | If rounding fell below the lower bound, move one floating-point step upward. |
| 324–325 | If rounding exceeded the upper bound or landed on an excluded upper endpoint, move one step downward. |
| 326–329 | Recheck both bounds. A sufficiently narrow interval may contain no representable float; mark it unusable rather than selecting a number outside it. |
| 330 | Return the valid float. |

**Tradeoff:** analysis uses exact crossings, but decoding consumes ordinary
numeric penalties. The representative needs to be usable, not printed with
misleading exactness. This does not model downstream float32 arithmetic or
hardware variability. `lower + 1` is a simple convention, not a robustness
estimate. The preferred penalty is a secondary selection preference, not a
regularization term in the word-error objective.

## 14. Lines 333–397: `optimize` — sweep the corpus without a grid

### Initialize the allowed range and the first winners

| Lines | Explanation |
|---|---|
| 333–334 | Define defaults: nonnegative penalty range, no upper limit, preferred `1.44`, and some report-only comparison penalties. |
| 335–337 | Convert bounds and preference into exact rationals. `None` keeps the upper range unbounded. |
| 338–339 | Reject reversed bounds. Equality is permitted: optimization can be restricted to one point. |
| 340–342 | Count all reference words and require a nonempty corpus with a positive total. |
| 343 | Initialize active winners, threshold-to-switch lists, and corpus S/D/I totals. |
| 344–346 | Build each utterance's lower envelope and begin with its minus-infinity winner. |
| 347–349 | Apply every switch at or below the lower bound to obtain the winner **at** that bound. |
| 350–351 | Schedule later switches within the upper bound, including a switch exactly at a finite upper bound. Ignore switches outside the allowed range. |
| 352–353 | Save the starting winner and add its edit tuple to corpus totals. |
| 355 | Initialize the output curve. |

### Emit intervals and apply grouped switches

| Lines | Explanation |
|---|---|
| 357–364 | Define `region`, recording numeric and exact interval endpoints, endpoint inclusion, a usable representative, and the **current** corpus error totals. This closure observes the updated `edits` variable. |
| 366 | Start at the allowed lower bound. |
| 367–368 | Visit distinct thresholds in exact sorted order. Record the interval before the switch with its upper endpoint excluded. |
| 369–373 | Apply every utterance switch at that exact threshold. Update S/D/I by `new - old`, and replace the active candidate. |
| 374 | The next region starts at that threshold. |
| 375 | Emit the final region; a finite upper bound is inclusive. A switch at that upper bound can therefore produce a final singleton region. |

Grouping simultaneous events is important: we must not evaluate an artificial
state in which only some utterances have switched at an identical penalty.

### Select an optimum and report requested comparisons

| Lines | Explanation |
|---|---|
| 376–377 | Find the smallest total word errors and keep **all** regions attaining it. Since the reference-word denominator is fixed, minimizing errors also minimizes corpus WER. |
| 378–380 | Among those regions, require at least one with a representable float penalty. Do not silently substitute a worse region if every true optimum is unrepresentable. |
| 381–383 | Select the region whose representative is closest to the preferred penalty; break a remaining tie by its exact lower bound. This is selection among equally good regions, not a different primary objective. |
| 384–386 | Initialize comparisons, deduplicate/sort their penalty values, and reset totals for each. |
| 387–390 | Directly choose each utterance's winner at that comparison penalty and sum its edits. These points report context; they do **not** determine the optimum. |
| 391–397 | Return denominator, minimum errors/WER, selected number and interval, all optimal intervals, count of unique scheduled corpus thresholds, full curve, and comparisons. |

Comparison penalties are evaluated even if they lie outside optimization bounds.
`unique_corpus_switches` counts distinct threshold values inside the scheduled
range, not the total number of utterance switches and not the number of
word-error changes. Some winner changes leave the error total unchanged.

**Why not average per-utterance optimal penalties?** There is one shared scalar
for decoding. Different utterances may prefer incompatible intervals. Sweeping
the union minimizes the actual total errors from their simultaneous choices.
The corpus WER can have multiple separated minima and need not be convex or
monotonic, even though individual candidate costs are straight lines.

## 15. Lines 400–415: `compatible_settings` — what may differ across sources

| Lines | Explanation |
|---|---|
| 400–405 | Document comparison of score/topology settings while allowing collection penalty and pruning changes. |
| 406–407 | List four allowed beam/pruning differences explicitly. |
| 408–415 | Build a mapping from `(section, key)` to value. Exclude the wildcard logging section, the collection blank penalty, those four search settings, and channel destinations. Keep the other explicit options, including checkpoint, lexicon, hardware, and LM/ILM settings. |

**Tradeoff:** wider beams or different anchors can expand the candidate pool
without changing the intended path-scoring function. Different hardware, models,
or LM scales cannot be pooled under that same claim. The comparison is deliberately
conservative string equality on saved options, not semantic equivalence checking:
equivalent spellings/paths can be rejected, and identical paths do not prove
their files were never modified. Pruning differences and fresh noise still mean
the *search process* was not identical.

## 16. Lines 418–457: `read_source` — read one completed recognition variant

| Lines | Explanation |
|---|---|
| 418–420 | Initialize compact utterances, this source's contexts, and rejection counts. |
| 421–423 | Stream the N-best file and locate the saved config for each record. |
| 424–425 | Read/cache a configuration the first time its path occurs. |
| 426–432 | If any context was already seen, compare the new config with the first known context and fail with differing keys if incompatible. `contexts` is shared across sources, so this checks both within-source and cross-source compatibility. |
| 433–435 | Cache the context, track it for this source's audit, and unpack its anchor/counts/metadata. |
| 436–441 | Prepare candidates with the source's actual anchor and configured LM scale; annotate utterance-level failures with filename and tag. |
| 442–444 | Keep only reference, frame count, and compact candidates; accumulate rejected-candidate reasons. |
| 445–446 | Reject an empty source. |
| 447–457 | Return compact utterances plus source provenance, anchors, candidate/rejection counts, exact-reference presence, oracle errors, and the fallback audit. |

`reference_present_utterances` means some candidate has **zero word edits**. It
is not a raw byte-for-byte string comparison: splitting ignores whitespace
differences. It does not require that zero-error candidate to be a score winner.
The oracle independently selects the minimum error count per utterance; it
does not require a common penalty that can realize those choices.

Saved config and lexicon paths must still exist and describe the collection.
Passing an N-best file alone does not make this a self-contained archive.

## 17. Lines 460–524: `analyze_nbest` — pool sources, then optimize once

### Require the same corpus and preserve observation identity

| Lines | Explanation |
|---|---|
| 460–464 | Define the analysis interface and initialize shared caches, pooled records, source statistics, and total rejections. |
| 465 | Pair every N-best file with its recognition log using `zip(..., strict=True)`. Unequal list lengths fail instead of silently losing a source. Assign source indices from zero. |
| 466–468 | Read a source and require exactly the same utterance IDs as previous sources. Pooling is a per-utterance union, not an intersection that drops hard examples. |
| 469–473 | Create each pooled record if needed; require matching reference strings and encoder lengths across sources. |
| 474–478 | Append observations with a unique pooled rank, source index, and original export rank. `replace` preserves all costs/counts/errors. |
| 479–480 | Save source-level statistics and accumulate rejections. |

### Deduplicate without changing the fixed-observation winner

| Lines | Explanation |
|---|---|
| 482–484 | Explain parallel-line deduplication and initialize aggregate candidate/oracle/coverage counts. |
| 485–488 | Sort observations by intercept, then rank; keep the first observation for each `(transcript, blank_count)` pair. It has the cheapest intercept and deterministic provenance. |
| 489 | Restore retained candidates to stable rank order. |
| 490–492 | Count unique candidates, sum per-utterance oracle errors, and count utterances with a zero-error candidate. |
| 493 | Collect all actual collection anchors. |
| 494–497 | Optimize the union once; add source anchors to the comparison list so reports show the saved-pool result there too. |

Why key on both transcript and blank count? The same words with a different
valid token count have different slopes and may win at different penalties.
For identical transcript/count pairs, keeping only the lowest intercept does
not change any winner or error obtainable from the union of fixed observations.

**Important statistical tradeoff:** when scores came from independent noisy
runs, that minimum can be a lucky read. We did **not** average duplicate costs.
Averaging would define a different estimator and would require a deliberate
repeated-scoring protocol to interpret as expected hardware performance. Here
pooling is explicitly a candidate-pool heuristic, not mean-score estimation.

### Describe the objective and its limitations in the result itself

| Lines | Explanation |
|---|---|
| 498 | Count usable observations before deduplication. |
| 499–509 | Add objective, input/source provenance, corpus size, unique/observed/duplicate/rejected candidates, reference coverage, and candidate-oracle WER. Coverage's denominator is utterances; WER's is reference words. |
| 510–512 | Record the requested bounds and preferred penalty. |
| 513–518 | State tie rule, whitespace scoring, incomplete-score caveat, rational arithmetic, deduplication rule, and lucky-read limitation. These travel with the output rather than living only in documentation. |
| 519–520 | Attach saved context metadata. |
| 521–523 | Convert each source's coverage and oracle errors into comparable percentages using the same pooled corpus denominators. |
| 524 | Return both the summary and compact utterances for subsequent output writing. |

Utterances whose reference is absent remain in every total. We optimize among
their imperfect candidates too. A candidate can have the fewest word errors
yet never lie on the lower envelope; a blank penalty cannot make it win.
Even when several good candidates can each win, their winning intervals may
not overlap, preventing one shared penalty from realizing the oracle.

## 18. Lines 527–557: `format_results` — the human-readable report

| Lines | Explanation |
|---|---|
| 527–541 | Build a report header with objective, inputs, corpus/candidate sizes, reference coverage, oracle WER, selected penalty, minimum saved-pool WER, and tie rule. |
| 542–545 | Add per-source anchors, coverage, and oracle WER so pooling benefits can be inspected. |
| 546–550 | List every optimal interval, showing infinity or the correct open/closed upper endpoint and its representative. |
| 551–554 | List comparison penalties with WER and S/D/I. |
| 555–556 | Print score-export, word-scoring, and noise limitations; explicitly state no new model scoring/decoding was performed. |
| 557 | Return newline-joined text with a final newline. |

The printed decimal precision is for reproducibility, not a confidence interval.
The JSON summary retains exact interval strings for arithmetic inspection.

## 19. Lines 560–578: `write_candidate_outputs` — compact audit artifacts

| Lines | Explanation |
|---|---|
| 560–562 | Define output writing and convert the chosen float back into the arithmetic representation used by `winner`. |
| 563–564 | Start two standard JSON objects: all retained candidates and selected hypotheses. |
| 565–568 | Iterate utterances and write commas only between records. |
| 569–572 | Write each ID, reference, frame count, and compact candidate list; reject NaN serialization. Do not duplicate raw tracebacks. |
| 573–576 | Select that utterance's winner at the chosen penalty and save its reference, penalty, and candidate/provenance details. |
| 577–578 | Close both JSON objects. |

`candidate_lines.json` contains retained transcript/count pairs, including lines
that never win; it is not just the lower envelope. `best_hypotheses.json`
contains saved-pool predictions, **not** fresh decoder output.

## 20. Lines 581–618: `TreeNBestBlankPenaltyJob` — Sisyphus integration

| Lines | Explanation |
|---|---|
| 581–587 | Define a CPU-only Sisyphus job taking tuples of N-best paths and log paths, bounds, preference, and comparison points. No standalone CLI is added. |
| 588–593 | Store all inputs and analysis parameters on the job. |
| 594–597 | Declare summary JSON, report text, compact candidates, and selected hypotheses as output paths. |
| 598 | Declare a Sisyphus output variable for the selected scalar penalty. |
| 600–601 | Request one CPU, 8 GB memory, and two hours for `run`; no GPU is requested. This is a resource request, not a guarantee for arbitrarily large inputs. |
| 603–609 | Resolve Sisyphus input paths to local `Path` objects and invoke the analysis with all configured parameters. |
| 610–612 | Write the indented machine-readable summary, forbidding NaN, and add a newline. |
| 613–614 | Write the human-readable report. |
| 615–617 | Write compact candidates and chosen hypotheses. |
| 618 | Publish the selected penalty only after output writing succeeds. |

**Tradeoff:** this stays a Sisyphus workflow component, as explicitly requested.
Dependencies point to completed recognition outputs rather than scraped running
temporary files. The optimizer itself neither submits jobs nor starts a manager.

## 21. Lines 621–641: `run_tree_nbest_blank_penalty` — a deliberately small hook

| Lines | Explanation |
|---|---|
| 621–625 | Accept a list of recognition jobs plus output prefix and analysis settings. State that source order breaks exact score ties. |
| 626–631 | Build the optimizer from each source job's `out_files["nbest.json"]` and `out_files["rasr.recog.log"]`. Passing Sisyphus paths preserves dependencies, including merge jobs for shards. |
| 632 | Add a readable optimizer alias. |
| 633–640 | Register the four output files and scalar penalty under the diagnostic prefix. |
| 641 | Return the job so the surrounding workflow can use its outputs. |

Earlier proposals accepted a mixture of one job, lists, directories, and optional
logs. The user asked for simpler code. The final interface intentionally has
one path: **a list of recognition jobs with their expected outputs**. We kept
correctness checks for counts, corpus identity, topology, and fallback rather
than retaining defensive branches for unrelated input forms.

## 22. Upstream collection and experiment wiring: the other tradeoffs

These are not executable lines in the optimizer, but they explain why its
inputs and assumptions look this way. See
[`../../recipe/model_pipelines/common/recog.py`](../../recipe/model_pipelines/common/recog.py)
and [`../../config/tree_nbest_debug.py`](../../config/tree_nbest_debug.py).

### Real candidate histories, not reference histories

Tree search runs with each hypothesis's own prediction history. It uses the
model, lexicon, and configured language model to explore alternatives. The
reference is saved for later error calculation only. This avoids pretending
that local decisions on teacher-forced reference histories simulate actual
tree competition.

The older local decision-risk method uses digital soft alignment probabilities.
This tree method instead evaluates actual retained transcripts against the
reference. It needs neither a digital guide nor forced alignment, soft alignment,
nor monotonic Viterbi rescoring. Those would add another scoring/alignment method
without solving the current missing-candidate problem. Separate digital
experiments in the current config are comparisons, not teachers supplying
alignments to this optimizer.

### Reuse the existing N-best binding; do not decode a second time

The installed LibRASR exposes `get_current_n_best_list` and
`recognize_segment_n_best`. After ordinary recognition, collection uses
`get_current_n_best_list(nbest_size)` on that search's current lattice.
Calling `recognize_segment_n_best` **after** `recognize_segment` would rerun
decoding, adding cost and producing different noisy observations.

An ordinary 1-best CTM does not provide competing candidate lines. The existing
N-best interface supplied alternatives without a C++ rebuild. Other inspected
Python methods did not recover full internal model-token paths. Exposing a word
lattice alone would not restore token timings omitted during tree trace
construction.

### One opt-in setting, standard JSON, and stable existing experiments

The collection choice was a single `nbest_size` setting, with zero disabling
storage, rather than a separate enable flag and size that could disagree.
Collection saves full tracebacks, component scores, reference, encoder length,
and saved configuration path in standard `nbest.json`.

This increases disk usage substantially, but retains evidence needed to inspect
score conventions later—the LM correction is a concrete benefit. The optimizer
then emits compact artifacts rather than duplicating those tracebacks. The
disabled path was intended to preserve existing recognition behavior/job hashes;
enabling N-best collection is a distinct workflow setting.

### Sharding stays; N-best resume support does not

Each shard handles complete utterances and stores up to N candidates **per
utterance**, not N divided among shards. The merge produces one corpus file.
Optimization consumes that merged result and selects one corpus-wide penalty,
not one penalty per shard.

The user explicitly chose to remove **N-best extraction's resumable support**
to simplify its caching/reconstruction code. Ordinary resumable recognition
remains supported; the N-best/resume combination is disallowed. An interrupted
N-best shard must restart rather than resume cached extraction. Sharding still
limits the size of each independently repeated unit of work.

### Keep the diagnostic out of `RecogResult`

The user rejected adding experimental producer/job fields to the shared
`RecogResult` abstraction. `config/tree_nbest_debug.py` instead follows the
already registered N-best output to its creating job. That keeps the shared
result interface clean, at the cost of coupling this diagnostic config to
registered output names and Sisyphus graph details.

### Multiple anchors expand coverage; they are not a full sweep

Collecting at several penalties can discover hypotheses pruned at the original
anchor. Each source is normalized using **its own** anchor before pooling. We
do not average separately optimized penalties, and we do not pretend N-best
enumerates all possible hypotheses.

Wide beams and larger N improve potential coverage but cost runtime, memory,
and disk. A wide search beam is not the same as exhaustive candidate coverage;
exporting 20 alternatives still truncates what the optimizer sees. Allowing beam
differences is useful for pool expansion but does not reconstruct how a fresh
search would prune at each candidate penalty.

### Keep model/hardware/LM score functions separate

The investigation explicitly pivoted to joint L2 `2prec/6range` and its estimated
penalty `1.44`; old probe scores from a different ADC setup are interface
evidence, not compatible scoring observations. The current config uses
`2prec/6range` for **both** joint layers. That later setting must not be confused
with the earlier probe's other-layer settings.

At inspection, the config defines separate no-LM memristor pooling at anchors
`0.9` and `1.44`, a memristor ILM `0.4` / external LM `0.5` experiment at anchor
`1.44`, and separate digital experiments collected at zero. The digital LM
comparison uses ILM `0.2` / external LM `0.7`. These are separate optimizer
groups, **not** a pool mixing different score functions. Check the config for
later changes rather than treating these values as optimizer requirements.

Collect LM-enabled alternatives with the intended LM in the search: adding an
LM afterward cannot recover candidates that its search would have retained.
For a fixed retained observation, however, its LM/ILM contributions are constant
with respect to the blank penalty, so no additional LM run is needed here.

## 23. The central approximations — what we knowingly gave up

### A. Exported terminal cost instead of the authoritative complete lattice cost

The inspected Python N-best export accumulates labeled-arc components. Source
inspection found more authoritative candidate totals on N-best root selection
arcs, including epsilon/final contributions. Matching top-N-best and direct-best
costs in three probe utterances did **not** prove every alternative complete.

We proceeded with the available totals rather than change/rebuild RASR first.
Candidate-dependent omitted costs can shift crossings and the selected optimum.
A constant omitted amount shared by all candidates of one utterance would not
change its ranking; differing omitted amounts would. Reconstructing scaled LM
components fixes the identified scaling issue, not every possible export issue.
If this becomes limiting, exposing the existing authoritative root-arc total is
a smaller targeted change than exporting all internal model-token paths.

### B. Infer counts under restricted topology instead of exporting alignments

We traded generality for a simple, auditable coefficient: `B=T-U`. This needs
complete paths, known pronunciation lengths, no repeated-label collapse, and
the checked RNA setup. Ambiguous candidates are rejected; fallback use can
invalidate the whole source. The strict checks prevent a convincing but wrong
line slope from silently driving optimization.

### C. Fixed noisy observations instead of controlling hardware noise

The user explicitly ruled out fixing/freezing readout noise as unfaithful to
actual hardware. The analytical calculation holds the **collected observations**
fixed only to ask a conditional question: how would these lines rank at `b`?
It does not claim the hardware will reproduce those same costs on a new read.

Fresh decoding keeps active noise. Repeated fresh runs can measure variability
without turning the noise off. Minimum-intercept pooling across runs can mix
candidate discovery with selection of unusually favorable reads; its WER is not
an unbiased prediction of the next hardware run.

### D. Optimize retained candidates instead of rerunning search at every penalty

This removes the cost of a dense decoding sweep and computes exact switches
for the available lines. It cannot recover unseen hypotheses, penalty-dependent
pruning, or fresh noisy scores. Moving far from collection anchors makes the
fixed-pool approximation especially questionable. Pooling anchors helps but
does not prove completeness.

### E. One global scalar instead of per-utterance oracle choices

The method minimizes corpus errors with one deployable penalty. It does not
force the reference to win everywhere. When the reference is absent, use the
available imperfect candidates and report the remaining oracle floor. When a
good candidate never becomes cheapest, or favorable intervals conflict between
utterances, changing a single blank penalty cannot repair that limitation.

### F. Development-set tuning instead of a generalization guarantee

Reference transcripts are legitimate supervision for selecting a penalty on a
development corpus. The optimized WER on that same corpus is a tuning result,
not held-out evidence. Narrow intervals and many decimal places are not proof
of a universally precise optimum. Evaluate the selected region on fresh
decoding and reserve independent evaluation data for claims of generalization.

## 24. What the recorded results taught us

These are **historical observations quoted in the working-session chat**, not
results recomputed for this walkthrough or guaranteed current output values.

| Metric | Original `1.44` pool | Pooled `0.9 + 1.44` |
|---|---:|---:|
| Whole-utterance zero-error coverage | 42.53% | 50.31% |
| Candidate-oracle WER | 8.4066% | 6.6695% |
| Best global-penalty saved-pool WER | 9.4312% | 8.8639% |

The original pool's approximately 43% coverage was **not** 43% recognition
accuracy. A single wrong word makes an entire utterance uncovered. Oracle WER
was more informative: it separated missing-candidate limitations from the
restriction to one global blank penalty.

Pooling lowered the oracle, showing useful candidates were missing. But even
the pooled oracle was below what one penalty could achieve. The chat reported
four separated optimal intervals and a representative around `0.8233`, chosen
nearest to preferred `1.44` among their representatives. In that pool, `0.9`
was only 12 word errors worse than the optimum.

Crucially, the pooled WER at the **unchanged** penalty `1.44` also improved—from
9.5274% to 9.1544%. That improvement cannot be attributed to changing the blank
penalty: the pool and noisy observations changed. The defensible conclusion was
to test a lower region around `0.8–0.9`, not to claim that fresh hardware decoding
would achieve exactly 8.8639% at a precisely calibrated `0.823292694092`.

## 25. Source trail for the design decisions

The main retrieved OpenCode discussion is:

```text
Title:   QAT blank penalty calibration Tree working session
Session: ses_eac8b2724ffeybBKnwkwg2smX7
```

Its earlier related discussion was also retrieved:

```text
Title:   QAT blank penalty calibration Tree WIP (fork #1)
Session: ses_ecf403460ffeVFlvXOXgBWNcKQ
```

Selected message anchors in the working session, useful for locating the
reasoning rather than treating this document as an invented retrospective:

| Decision/discussion | Message ID |
|---|---|
| Do not fix hardware noise | `msg_15374f9aa001x7O8qh6QEPxV0S` |
| Pivot to joint L2 `2prec/6range`, estimated penalty `1.44` | `msg_15374fbec001L2MRFY8DfxSDNE` |
| Prefer standard formats | `msg_1634abd49001QmTfQHzYtsgwo3` |
| Clarification: remove only N-best resumable support | `msg_1636970de001sbJzm0Vj7GwyUf` |
| Existing-lattice getter versus decoding twice | `msg_1637c47fb001tTS59UDnc6O9DK` |
| Require a Sisyphus optimizer consuming completed output | `msg_163ef0f3b001K5Djzzrx9gzHRN` |
| Fallback audit rationale | `msg_163f24e37001QStyXx5MoA1hKL` |
| Keep experimental fields out of `RecogResult` | `msg_164ee1dca001LK8avQTeUdtGT2` |
| Multiple variants and simplified interface | `msg_1653d68eb001MSIgfCoAhxPSrk`, `msg_165405c4c001jEmBLSgRJNviBW` |
| Historical pooled result interpretation | `msg_168694b08001FNgQlwW7yV4sYN` |
| N-best versus direct-best LM scaling investigation | `msg_1687a6469001Q1M7gasSMJWjDi` |
| Minimal LM correction and fixed ILM treatment | `msg_1687f946c0011Vw2RffVElIVOP` |
| Separate ILM `0.4` / external LM `0.5` collection | `msg_1688088e50015DurB31x5mnoDw` |

Some earlier chat proposals were superseded: alignment-based rescoring,
zero-LM-only support, mixed single-job/path interfaces, and N-best resume
handling are **not** descriptions of the current optimizer. Current behavior
above is grounded in the inspected code; additional caveats identified while
reading it are explanations, not claims that every one was explicitly debated
in the chat.

Related local evidence:

- [`TREE_NBEST_APPROACH.md`](TREE_NBEST_APPROACH.md): interface inspection,
  blank-count example, and complete-cost export limitation.
- [`TREE_NBEST_PROBE.md`](TREE_NBEST_PROBE.md): candidate collection probe.
- [`TREE_NBEST_VALIDATION.md`](TREE_NBEST_VALIDATION.md): separate optional
  statistical rescoring investigation; not a prerequisite alignment step in
  this optimizer.
- [`OPTIMAL_BLANK_PENALTY_WALKTHROUGH.md`](OPTIMAL_BLANK_PENALTY_WALKTHROUGH.md):
  the **different**, digital-soft-alignment local decision-risk method.

## 26. How to read an optimizer result responsibly

1. Check rejection reasons, context compatibility, and the fallback audit.
2. Compare per-source and pooled oracle WER and reference coverage.
3. Inspect all optimal intervals, not only the chosen decimal.
4. Compare nearby practical penalties; tiny saved-pool differences may not
   survive readout variability or different search pruning.
5. Remember that comparison WERs use the same saved pool, not fresh decoding.
6. Validate the chosen region against the original anchor with fresh decoding,
   active noise, the same model/hardware/LM settings, and repeated runs where
   feasible. Use the intended external scoring convention for final claims.

**Bottom line:** the calculation is an exact optimizer of a deliberately
restricted, fixed-observation candidate-set objective. Its use as a prediction
of new tree decoding is approximate and must be tested.
