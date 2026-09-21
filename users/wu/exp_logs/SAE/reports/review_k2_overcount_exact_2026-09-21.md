# Review: the exact log-semiring CSR reference, commit `f240391` (2026-09-21)

Reviewed `recipe/2025-10-speech-llm` `f240391` (parent `2ced584`) against the dispatch and against
`impl_k2_overcount_exact_2026-09-21.md`, `SAE_4A_lexlat.md` amendment 9 item 5 + the "Over-count
read, amendment 9.5" results subsection, and `audit_k2_overcount_2026-09-21.md` sections 2-3.
Read-only; nothing changed, nothing launched.

**Verdict: the reference and the job are sound.** Three minor findings below, none of which
changes a reported number as the job stands.

## What I verified myself (not taken from the implementer's report)

**(1) Arc parity and no double counting -- verified independently, including ESCAPE.** The
committed enumeration test (`test_lexlat_k2_overcount_exact.py:68-81`) only covers
`escape=False`, while the registered read and this job both run at `escape=True`
(`overcount.json` `"escape": true`). I wrote my own enumeration of readings -- word spans taken
off the trie with all homophones, escape spans of every length, SIL a free boundary, each
reading priced by a fresh `lm_score` walk plus its own `(n_segments + 1) * sil_lp` -- and
compared it with `lexlat.string_log_sum` on `test_lexlat._STRINGS` (all 10 strings, including the
ambiguous one, SIL at a boundary, SIL inside a word, an out-of-lexicon phone, the pure-escape
string, the repeated phone and the empty string) at `escape` in {True, False} and
`sil_log_prob` in {0.0, log 0.5}. Agreement to 3.6e-15 nats on every combination (up to 70
enumerated readings per string). So the escape word transition, the per-phone escape price, the
per-parse SIL constant and the free escape close are each priced exactly once, and no reading is
reachable by two routes.

On REAL strings I ran the job's own max-semiring self-check: `string_log_sum(semiring="max")`
reproduces the registered `csr_log_prob` with `max_semiring_max_abs_dev = 0.0` (bit-for-bit) on
50 strings per set. Sentence end: neither scorer adds an end score and the log-sum is taken over
all final states, exactly as `string_best_segmentation` maximises over them.

**(2) Per-token denominator.** `lexlat_k2_overcount_exact_jobs.py:178` uses
`n = max(len(ids), 1)` with `ids` re-derived from `step0_strings.json`, identical to
`lexlat_k2_train.py:661`, and `:164` asserts `len(ids) == reg["tokens"]`. Same denominator as
every registered column. Medians are per-string, unweighted, `np.percentile(...,95)` -- the same
convention as `lexlat_k2_train._summarise`.

**(3) Pairing and exclusions.** The job iterates `registered["sets"][key]["rows"]` -- exactly the
rows the finished job scored -- and re-reports `registered_set["skipped"]` verbatim (`:229`). A
string it cannot map or score raises (`KeyError` at `:163`, asserts at `:164` and `:166`); it
never drops one silently. On the real inputs it pairs 2091 gold + 2842 private rows, matching the
registered `summary.txt`.

**(4) Clock and memory, measured by me on the login node.** 7.5-7.9 ms per token (2,746 and
12,031 token samples, resources pre-loaded), peak RSS 1.26 GB. The 276,451 tokens of the two sets
are ~36 min against a `budget_seconds` of 3.6 h and a 4 h clock, in 16 GB. The implementer's
5.8-7.4 ms/token and 1.27 GiB are accurate. Partial write every 200 rows and once per set.

**(5) Units of the graph column.** `k2_log_sum` is `get_tot_scores(log_semiring=True)` on the
exact host intersection (`lexlat_k2_train.py:659`) -- the log-semiring total, not the max-plus --
and the HLG carries the silence model per PATH, while `exact` carries `sil_log_prob` per PARSE
inside the sum (`lexlat.py:719,722,726`). Same side, same units. The registered
`tropical_residual` median of 0.000000 on both sets (job `summary.txt`) is the evidence that H
and G contribute nothing beyond the CSR score on the max side, which is what makes the two
totals comparable at all.

**(6) The enumeration is independent of the DP.** `test_lexlat_k2._parses` (`:546`) is a plain
recursive enumeration of cuts over `TINY_LEXICON`'s pronunciations and `_parse_score` (`:564`)
scores each parse through `lexlat_k2._lookup`, the numpy twin of `lexlat.lm_score`, plus
`sil_model_log_prob`. It shares no code with `string_log_sum`.

**(7) Hash census, run by me** in the setup dir under the sis env, on the working tree as it
stands: `py_overcount()` resolves to exactly two jobs --
`LexlatK2OvercountJob.f5Ljn6twbc4b` (unmoved) and
`LexlatK2OvercountExactJob.4yKkB64ZDvJ4`. The uncommitted `lexlat_k2.py` edit of the concurrent
`#0`-loops worker (a new `backoff_loops` parameter defaulting to `"all"`, no job-class change)
does not move it either.

**(8) Corroboration against the audit.** Driving `exact_overcount` on the first 60 rows per set of
the real inputs: gold `over_count_exact` median **+0.2742**, against the auditor's independently
written DP at **+0.274**; private **+0.1523** against the auditor's +0.117 on a different
60-string sample (the registered `over_count` on my rows is +0.186 vs the auditor's +0.1386, so
the samples differ). `legit_multi_parse` +0.0089 gold / +0.0354 private, the same "small on gold,
larger on private" shape the audit reports. 0 identity violations, 0 log-sums below their own
max-plus.

**(9) Nothing beyond the intended delta.** Four files. The `_string_lm_cache` extraction is
behaviour-identical for `string_best_segmentation` (the only change is `int()` normalisation of a
cache key that already held python ints). The config change adds `overcount_exact` and rewires
`py_overcount`; `overcount()` itself and `py()` are untouched. The job states no experimental
constant of its own: `escape`, `sil_prob`, the strings, the exclusions come from the finished
json, the tolerance from `lexlat_k2_train.OVERCOUNT_TOLERANCE = 0.05`.

## Findings

**F1 (traceability). `lexlat_k2_overcount_exact_jobs.py:338-343` -- `summary.txt` never names the
graph.** The sibling job's summary prints `graph: .../LexlatHLGBuildJob.rtX44PBJFNy1/output/HLG.pt`
on its second line. This one prints only `registered_json` and `registered_name`;
`out["hlg"]` is collected at `:134` and never rendered. The phase is about to build a second graph
(the word-boundary `#0` loops, `lexlat_k2.py` uncommitted in the working tree) and to repeat this
read on it. Two `overcount_exact/summary.txt` files under different job hashes would then be
indistinguishable by graph without opening the json, which is the mis-attribution the
`stale-output-symlinks` rule exists for. Recoverable (the hash is inside `registered_json`), so
this is a defect of the rendered artifact, not of a number.

**F2 (rendered verdict). `lexlat_k2_overcount_exact_jobs.py:141-144` and `:394`.** On a budget
abort `out["pass"]` is forced `False`, so the summary prints `VERDICT: FAIL` even when every
per-set line above it prints `READ: PASS`. The `PARTIAL RUN` line at `:396-399` discloses it, but
the word quoted from such a summary would be FAIL. Identical to the sibling job, and with a
36-minute job inside a 3.6-hour budget it should not fire.

**F3 (scope, for the orchestrator -- not a code defect).** As registered, `overcount_exact()` is
wired only to `overcount()`, whose graph is the banked `rtX44PBJFNy1`. The phase's decision of
2026-09-21 states that the read which DECIDES amendment 9.5 is `over_count_exact` on the REBUILT
graph; nothing in this commit registers that. On the banked graph the job will report FAIL -- my
60-row cross-check gives +0.274 gold / +0.152 private against the 0.05 acceptance -- so what this
run delivers is the full-set baseline and the `legit_multi_parse` split, not the deciding read.
That is what the dispatch asked for and it costs ~36 min of one CPU; it is named here only so the
`VERDICT: FAIL` this job prints is not read as the phase's gate.

## Observation, no failure path today

The committed enumeration identity (`test_lexlat_k2_overcount_exact.py:68`) is STRICT-only, while
the operating point is `escape=True`. The escape branch is covered in the committed battery by
max-semiring parity alone, which cannot detect a duplicated route (two equal-weight routes give
the same max and a log-sum inflated by log 2, in the direction that would understate
`over_count_exact`). I closed that gap myself (item 1 above) and the branch is correct as
written, so no number is at risk now; a future edit to the escape transitions would not be caught
by the committed tests.
