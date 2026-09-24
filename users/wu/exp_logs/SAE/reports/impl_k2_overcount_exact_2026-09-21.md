# Implementer report: the EXACT log-semiring CSR reference for amendment 9.5, 2026-09-21

Commit `f240391` in `recipe/2025-10-speech-llm` (branch as checked out, parent `2ced584`), four
explicit paths, not pushed. **Nothing launched.** No project document touched. The concurrent
implementer's modified `config_sae_1g_v1.py` and untracked `config_sae_3e1_d6_swap_cont_v1.py` were
left alone. `lexlat_k2_arm_jobs.py` and `lexlat_k2_train.py` were read and NOT edited, so the
finished `LexlatK2OvercountJob.f5Ljn6twbc4b` cannot move.

## What the delta is, and why

The registered read compares the graph's LOG-semiring string score with
`lexlat.string_best_segmentation`, a MAX-PLUS scorer, so its `over_count` column holds the epsilon
back-off double count AND every legitimate reading a marginal sums over (alternative word
segmentations, the escape reading of a span). This round adds the reference the amendment's sentence
names for a log-semiring quantity -- the sum over those readings under the CSR automaton's
longest-suffix back-off, which gives each reading exactly one route -- and a job that reports the
double count alone.

### 1. `sae/emc/lexlat.py` (+143 / -18): `string_log_sum`, a sibling of `string_best_segmentation`

`string_log_sum(phone_ids, res, *, escape=True, sil_log_prob=0.0, semiring="log")` is
`string_best_segmentation`'s state space (position, word-LM state, escape-open), its arcs (word
close per homophone, SIL a free boundary, escape open, escape extend, free escape close) and its
prices, with `max` replaced by log-sum-exp. `semiring="max"` switches the combiner back, which is
what makes it a reference rather than a second scorer.

* **Shared pricing path.** The memoised back-off lookup both functions use was lifted into
  `_string_lm_cache(res)` and `string_best_segmentation` now calls it; that function's behaviour is
  unchanged (same `lm_score` calls, same cache, same tie-breaking), and its own tests still pass.
  **Disclosed deviation from the brief:** the TRANSITION code is mirrored, not shared.
  `string_best_segmentation` carries back-pointers and a word read-out that a log-sum has no
  analogue for, and rewriting it would have put the banked E-1 scorer (and its tie-breaking, which
  decides the reported word list) at risk for no measurement gain. Escape parity is therefore
  established by test, not by construction: `semiring="max"` reproduces `string_best_segmentation`
  to 1e-9 on `test_lexlat`'s whole string battery at BOTH conventions, and to 0.0 (bit-for-bit) on
  real strings inside the job.
* **The SIL constant enters per reading.** `sil_log_prob = log(1 - p_sil)` is paid once at the start
  and once per word transition (an opened escape counts as one word, as `sil_model_log_prob`
  documents), so each parse pays its own `(n_words + 1) * log(1 - p_sil)` -- it cannot be added
  outside a sum whose terms have different word counts. `0.0` (the default) leaves the bare CSR
  score, which is what the max-plus equivalence is checked at.

### 2. NEW `sae/emc/lexlat_k2_overcount_exact_jobs.py`: `LexlatK2OvercountExactJob` (+ `exact_overcount`)

CPU, in-process, no k2, no GPU. It takes the finished job's `overcount.json` as an INPUT PATH and
recomputes nothing on the graph side: `k2_log_sum`, `k2_max_plus`, `reference_total`, the tags and
the EXCLUSION COUNTS are read off disk, and it scores exactly the rows that job scored (so the
exclusions -- 772 of 2,864 gold, 22 private, adjacent repeats the run-collapse topology cannot emit
-- are carried over unchanged and re-reported). `escape`, `sil_prob`, the string sets and the
acceptance all come from that json and from `lexlat_k2_train.OVERCOUNT_TOLERANCE`; **this job states
no experimental constant of its own.**

Per set, per token: `over_count_exact = graph log-sum - exact CSR log-sum` (median / p95 / max /
min) -- the column the unchanged acceptance (median <= 0.05 nats per token, PASS/FAIL per set) is
read on; `legit_multi_parse = exact CSR log-sum - the registered max-plus reference`; and
`over_count` as registered, re-printed. `over_count = over_count_exact + legit_multi_parse` is
asserted per row (violations counted), as is `exact >= reference_max_plus`; the first
`MAX_CHECK_STRINGS = 50` strings per set are also scored in the max semiring and asserted equal to
the registered `csr_log_prob` (an internal self-check that changes no reported number). Partial
writes every 200 strings and a budget inside the Slurm clock, as the sibling job does.

**Resource estimate, measured on the login node 2026-09-21** (not an assumption): the DP costs
5.8 ms/token (private sample) to 7.4 ms/token (gold sample) in one log-semiring pass, so the two
sets' 276,451 tokens are about 30 min single-threaded; peak RSS of a process that loads the 396 MB
`lexlat_resources.npz` is 1.27 GiB. Requested: 2 CPUs, 16 GB, 4 h (`budget_seconds` = 0.9 x clock).
The registered job's 77 min on 16 CPUs / 200 GB paid for the k2 host intersection, which this job
does not run.

### 3. `config_sae_4a_lexlat_k2_pack_v1.py`: `overcount_exact()`, registered in `py_overcount()`

Registered on the EXISTING shim `config/sae_4a_lexlat_k2_overcount.py`; outputs under
`{PREFIX}/overcount_exact/` (`overcount_exact.json`, `partial.json`, `summary.txt`), alias
`sae/4a/lexlat_k2_pack/overcount_exact`. It consumes `overcount().out_stats`, `STEP0_STRINGS` and
`RESOURCES` -- the same pins the registered read used.

## Checks run

| check | result |
|---|---|
| new tests, training env (`test_lexlat_k2_overcount_exact.py`) | 7 passed |
| same file, k2 scratch env, with `test_lexlat_k2.py` | 33 passed, 1 skipped, 1 xfailed (the graph still over-counts) |
| `test_lexlat.py` + `test_lexlat_k2.py`, training env | 60 passed, 23 skipped (unchanged) |
| `test_lexlat_train.py`, `test_lexlat_k2_train.py`, `test_lexlat_k2_official.py` | 37 passed, 22 skipped |
| ruff on the four touched files | clean (the pre-existing E731/F841 in `lexlat.py` are untouched) |
| hash census of `py_overcount` (`scripts/sae_4a_lexlat_k2_overcount_census.py`, new) | 2 jobs: `LexlatK2OvercountJob.f5Ljn6twbc4b` **unmoved**, plus `LexlatK2OvercountExactJob.4yKkB64ZDvJ4` |
| hash census of the live k2 probe graph (`scripts/sae_4a_lexlat_k2_census.py live`) | 10 jobs, `LexlatHLGBuildJob.rtX44PBJFNy1` / `LexlatK2ProbeJob.Tkl94t85j4pV` unmoved |
| end-to-end smoke of the reader on the REAL inputs, `max_utterances=3` per set | ran, summary rendered, max-semiring parity deviation 0.0 nats, 0 decomposition violations |

The three verifications the brief asked for, on the fixture of `test_lexlat_k2`'s xfail
log-semiring test: (i) `string_log_sum` equals the brute-force enumeration over parses (the object
the graph exceeds by +0.064 nats) to 1e-9 on all three `_LOGSUM_STRINGS`, (ii) with max it
reproduces `string_best_segmentation` to 1e-9, (iii) on the single-parse strings log-sum equals
max-plus to 1e-9.

## The 6-string smoke is NOT the read (no number from it is a result)

It exercised the plumbing only. For the record of what it exercised, at 3 strings per set the
columns came back `over_count_exact` median +0.26 (gold) / +0.088 (private) and `legit_multi_parse`
median +0.027 / +0.030 against the registered `over_count` +0.287 / +0.100 -- 6 of 4,933 strings,
the first by tag, which is neither the set's median nor a read against the acceptance. The job is
registered and NOT launched, as instructed; the phase's read needs the full job.

## Left undetermined / for the orchestrator

* Nothing in the spec was left unresolved that I had to choose. Two judgement calls are disclosed
  above: the transitions are mirrored rather than shared (parity by test), and `MAX_CHECK_STRINGS
  = 50` / `MAX_CHECK_TOLERANCE = 1e-6` are self-check knobs that enter no reported number.
* The new job is a downstream consumer of `f5Ljn6twbc4b`'s output. A manager on the overcount shim
  will see two jobs, one FINISHED; launching the second is the orchestrator's call.
* Not committed anywhere: `scripts/sae_4a_lexlat_k2_overcount_census.py` (setup dir, not a
  repository) and this report.
