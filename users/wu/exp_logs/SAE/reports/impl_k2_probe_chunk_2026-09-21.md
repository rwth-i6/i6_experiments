# Implementer: chunk the k2 pruned intersection, and give the probe's child a clock (2026-09-21)

Status: DONE.  Committed as `3454206` on `haotian_modality_matching_jupiter` in
`recipe/2025-10-speech-llm` (five explicit paths, not pushed).  Nothing was launched.

## What changed

### 1. Chunking (`lexlat_k2.py`, `lexlat_k2_train.py`)

New in `lexlat_k2`: `CHUNK_SEQS = 16`, `CHUNK_SEQS_ENV = "LEXLAT_K2_CHUNK_SEQS"`,
`K2_PRUNE_WINDOW = 30`, `INT32_MAX`, and the functions `resolve_chunk_seqs`, `chunk_bounds`,
`max_out_degree`, `window_arc_bound`, `chunk_guard`, `chunked_tot_scores`, `lattice_sizes`.

`chunked_tot_scores(graph, dense_in, lens, ..., chunk_seqs, backward, sync)` runs one
`k2.intersect_dense_pruned` + `get_tot_scores(log_semiring=True)` per block of at most
`chunk_seqs` sequences and returns `(tot_scores, chunks, times)`:

* `tot_scores` is the concatenation in the batch's own order;
* `chunks` is `[(first batch index, lattice), ...]` for the readers of the lattice;
* `times` is the SUM over the chunks of the per-stage seconds -- the true cost of the leg,
  because the chunks run one after the other.
* with `backward=True` each chunk's `tot.sum().backward()` runs inside the loop, so every
  chunk's gradient lands in the ONE `dense_in.grad` (each chunk is a slice of that tensor) and
  no chunk's autograd graph outlives it.  `run_probe` uses this; the training runtime uses
  `backward=False` and keeps the graph.

`run_probe` now takes `chunk_seqs` (CLI `--chunk-seqs`), reports `chunk_seqs` / `n_chunks` per
cell and `chunk_seqs` / `chunk_guard` at the top level of both `probe.json` and `partial.json`.
Its peak-memory read is unchanged and is torch's running maximum over the chunk loop, i.e. the
MAX over the chunks, never their sum; lattice sizes are summed over the chunks, which is the
batch's own lattice.  The max-plus sanity pass stays outside the timed region and concatenates
the chunks' scores.

`LexlatK2Runtime` goes through the same function: `log_z_hlg` returns `(log Z_HLG, chunks)`,
`_monitors` sums `lattice_sizes` over the chunks and `_expected_words` concatenates the
per-chunk arc-posterior counts (`_expected_words_one` is the old body, per chunk).
`LexlatK2Spec` gains `chunk_seqs` (resolved by `K.resolve_chunk_seqs`) and `describe()` reports
it.

WHERE THE KNOB LIVES, and why it is not a hashed parameter: an explicit argument wins, then
`$LEXLAT_K2_CHUNK_SEQS`, then `CHUNK_SEQS = 16`.  No sisyphus `Job.__init__` and no config
keyword carries it, so no job hash and no arm hash can move with it; the probe job resolves it
in `run()` and passes it on the child's command line, so the value is in the log, in
`probe.json` and in `summary.txt`.  This was possible without any edit inside a job signature,
so the "unless unavoidable" case did not arise, and the model definition (`sae_blankfree.py`,
outside my files) was NOT touched: an arm that wants another chunk size sets the environment
variable.

### 2. The child cannot idle any more (`lexlat_k2_jobs.py`)

* `_child_env()` sets `K2_ABORT=1` (any value enables it; `k2/version/version.py:65` reads it
  with `getenv(...) is not None`) and forwards `LEXLAT_K2_CHUNK_SEQS` when set.  This env is
  shared with `LexlatHLGBuildJob` and with `lexlat_k2_official_jobs` / `lexlat_k2_arm_jobs`,
  whose child calls already had timeouts -- they now also abort instead of unwinding on a k2
  fatal.  That is the only effect of my change outside the probe job.
* New `_run_with_timeout(cmd, env, timeout)`: `Popen(start_new_session=True)`, then on expiry
  `SIGTERM` and `SIGKILL` to the PROCESS GROUP, each with a bounded wait; returns
  `(returncode or None, note)`.
* The probe's child call gets `budget = max(60, rqmt["time"] * 3600 * 0.90 - elapsed)` -- the
  build job's idiom at a 10 % margin instead of 5 % so the reads and `summary.txt` still fit.
  ASSUMPTION STATED: the dispatch's "a timeout equal to the job's step budget" is read as the
  task's remaining wall clock (the sibling call at `lexlat_k2_jobs.py:201` the debug report
  points at), not `STEP_ABORT_SEC` -- one child measures many cells, so a one-step timeout would
  kill it after the first.
* On a timeout or a non-zero exit the job no longer raises immediately: it reads the child's
  `partial.json`, runs the normal reads on the cells that were measured, and writes
  `summary.txt` with a `FAIL: <cause>` line (and `child_failure` in `probe.json`).  Only when
  NO cell was measured does it write `partial.json`, `probe.json`, `per_batch.json` and a
  `summary.txt` naming the cause (`_write_failure`) and then raise.  JUDGMENT CALL: a child that
  died for an unrelated reason therefore finishes the job with all-FAIL reads; the cause is
  named verbatim in `summary.txt` and `probe.json`, and every rung with no cell says it reads
  FAIL "for WANT of a measurement, not from one".

### 3. The guard

`chunk_guard(hlg, chunk_seqs, max_active_ladder)` runs in `run_probe` after the HLG is on the
device and BEFORE the first intersect call.  It logs the graph's largest out-degree and, per
`max_active` rung, `chunk_seqs * out_degree * 30 * max_active` as a fraction of 2^31, with
`OK` below 0.5, `WARNING: approaching ...` below 1.0 and `WARNING: ABOVE ...` at or above it.
It warns and never raises (the bound is a worst case: every active state assumed to be the
graph's largest-fan-out state on every frame).  The record is in `probe.json` / `partial.json`.

## Checks run

All in the k2 env `/e/scratch/spell/wu24/envs/sae_k2/bin/python` unless stated.

* BEFORE the change, baseline: `test_lexlat_k2.py` 28 passed / 1 skipped; `test_lexlat_k2_train.py`
  18 passed.
* AFTER: `test_lexlat_k2.py` + `test_lexlat_k2_train.py` + `test_lexlat_k2_official.py` +
  `test_lexlat_k2_overcount_exact.py` -> **88 passed, 1 skipped** (the skip is the pre-existing
  `kaldilm` one).  The overcount file is another implementer's; it was only RUN, never edited.
* NEW tests, all passing:
  * `test_lexlat_k2.test_chunking_is_bit_identical_to_one_call[1|2|3]` -- 4 utterances of
    unequal length on the compiled toy ESCAPE HLG, one call vs chunks of 1, 2 and 3:
    `torch.equal` on the per-sequence `tot_scores` AND on `dense_in.grad`.  BIT-IDENTICAL, not
    within a tolerance; no relaxation was needed (CPU, float32 dense / double scores).
  * `test_chunk_bounds_partitions_the_batch`, `test_resolve_chunk_seqs_reads_the_environment`,
    `test_window_arc_bound_is_the_sum_k2_takes_in_an_int32` (includes the failed cell: 114 x
    399,785 x 30 x 1000 > 2^31), `test_max_out_degree_is_the_largest_fan_out` (against a
    brute-force count of the real graph's arcs), `test_chunk_guard_logs_the_fan_out_and_the_bound`.
  * `test_lexlat_k2_train.test_chunking_leaves_the_term_its_gradient_and_its_monitors_unchanged[1|3]`
    -- 4 utterances through `LexlatK2Runtime.step`: the term, `log_q.grad` and every monitor but
    `lexlat_k2_sec` are equal at chunk 1 and 3 to the one-call run.
* Shared training env (`/e/project1/spell/wu24/env/conda/envs/speech_llm/bin/python`):
  `lexlat_k2_jobs`, `lexlat_k2`, `lexlat_k2_train` import without k2; `resolve_chunk_seqs()` =
  16, `$LEXLAT_K2_CHUNK_SEQS=8` -> 8 and appears in `_child_env()` together with `K2_ABORT=1`.
* `_run_with_timeout` exercised directly: `sleep 60` with a 2 s budget returns `(None, 'killed
  after 2 s')` in 2.0 s; `exit 7` returns 7; a self-`SIGABRT` returns -6.
* HASH CENSUS with `scripts/sae_4a_lexlat_k2_census.py`, before and after:
  `live` (config/sae_4a_lexlat_k2.py, 12 jobs) and `official`
  (config/sae_4a_lexlat_k2_official.py, 17 jobs) diff to NOTHING.  `LexlatHLGBuildJob.rtX44PBJFNy1`,
  `LexlatK2ProbeJob.Tkl94t85j4pV`, `LexlatK2ProbeJob.R7QzD6vYBLD3`, `.5dP7W2NMFq4T`,
  `.DtWYToXTPh6w` and the three official builds are unchanged.  No `__sis_version__` was bumped
  (all are hand-bumped integers, so editing these files is hash-neutral).
* `ruff check` on the five touched files: the only two findings are pre-existing `E731`
  lambdas; no new line exceeds 100 characters.

## What this does NOT show

The tests are CPU and toy-scale.  Nothing here measures the real probe on the GPU: whether the
114-sequence batch now completes at 16 sequences per call, and what the leg costs against E1's
bar, is only known after the job runs.  The verified statement is that chunking does not change
per-sequence results and that the child can no longer idle past its budget.

## Left undetermined by the spec

* "a timeout equal to the job's step budget" -- read as the task's remaining wall clock (see
  above).  If the intent was `STEP_ABORT_SEC` per cell, that has to be a per-cell clock inside
  the child, not a `sp.run` timeout, and is not what was implemented.
* The chunk size itself: 16 is the dispatch's value ("default 16, the verified size"), i.e. the
  largest size the debug report's CPU reproduction passed.  The arithmetic ceiling that report
  names for this graph is ~45; nothing between 16 and 45 was tried here.

## Files

* `recipe/2025-10-speech-llm/src/speech_llm/sae/emc/lexlat_k2.py`
* `recipe/2025-10-speech-llm/src/speech_llm/sae/emc/lexlat_k2_train.py`
* `recipe/2025-10-speech-llm/src/speech_llm/sae/emc/lexlat_k2_jobs.py`
* `recipe/2025-10-speech-llm/src/speech_llm/sae/emc/test_lexlat_k2.py`
* `recipe/2025-10-speech-llm/src/speech_llm/sae/emc/test_lexlat_k2_train.py`
