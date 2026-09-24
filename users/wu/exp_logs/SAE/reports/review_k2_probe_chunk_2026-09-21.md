# Review: 3454206 -- chunk the k2 pruned intersection, give the probe's child a clock (2026-09-21)

Status: DONE_WITH_CONCERNS.  Read-only; nothing edited.  Verdict: **safe to rerun the four
probes** (`LexlatK2ProbeJob.Tkl94t85j4pV`, `.R7QzD6vYBLD3`, `.5dP7W2NMFq4T`, `.DtWYToXTPh6w`).
The single intended delta is present and correct; four concerns below, none of which blocks the
launch, the first of which decides how the result may be READ.

## What I verified (and how)

* **Chunking is per-sequence neutral.**  `chunk_bounds` partitions the batch exactly
  (`lexlat_k2.py:920`); each chunk's supervision segment is `arange(n)` against the slice
  `dense_in[start:stop]`, so rows line up; `backward=True` accumulates into the ONE leaf
  `dense_in.grad` through the slice; `torch.cat(parts)` preserves batch order.  Upstream settles
  the framework question: in `intersect_dense_pruned.cu`, `GetPruningCutoffs` builds
  `Array1<float> cutoffs(c_, num_fsas)` and indexes `dynamic_beams_data[i]` PER FSA, so beams,
  `min_active` and `max_active` are per sequence and nothing global leaks across a chunk boundary.
  The same file's `PruneTimeRange` accumulates `int32_t tot_arcs` over the window, which confirms
  the debugger's diagnosis and the direction of the fix.
* **Tests are real k2, not a fixture.**  `test_chunking_is_bit_identical_to_one_call[1|2|3]` calls
  `K.chunked_tot_scores` on a compiled toy escape HLG (`_build_hlg`, real `k2.Fsa`) and asserts
  `torch.equal` on the per-sequence totals AND on `dense_in.grad`; `lens = [7,5,6,4]` against a
  full-T slice, so it does exercise the "a chunk sees a smaller max-T" question.  I reran the four
  k2 test files in `/e/scratch/spell/wu24/envs/sae_k2/bin/python`: **88 passed, 1 skipped**, as
  reported.  CAVEAT: the equality tests run at `max_active=10000` on a 4-utterance toy graph, where
  the dynamic beam never adapts; neutrality at the live operating point (`max_active` 1000, a
  24.7M-state graph, beams adapting on every frame) rests on the upstream per-FSA indexing above,
  not on a measurement.
* **Monitors aggregate as claimed.**  Probe: `stage_sec` summed over chunks (`chunked_tot_scores`),
  `lattice_sizes` summed, peak memory read once after the loop (torch's running max).  Runtime:
  `_monitors` takes `K.lattice_sizes(chunks)` and `_expected_words` concatenates the per-chunk
  counts in chunk order = batch order (`_expected_words_one` scatters with `index_add_` into a
  chunk-local `zeros(b)`); `test_chunking_leaves_the_term_its_gradient_and_its_monitors_unchanged`
  asserts every monitor but `lexlat_k2_sec` is equal at chunk 1, 3 and unchunked.
* **Hash neutrality, checked independently of the implementer's census.**  No `Job.__init__`
  signature changed in the diff; both `__sis_version__` in `lexlat_k2_jobs.py` (167, 416) are
  hand-bumped `2` and untouched.  I ran `scripts/sae_4a_lexlat_k2_census.py live` and `official`
  on the post-change tree: they print `LexlatHLGBuildJob.rtX44PBJFNy1`,
  `LexlatK2ProbeJob.Tkl94t85j4pV` (12 jobs) and `.5dP7W2NMFq4T`, `.R7QzD6vYBLD3`, `.DtWYToXTPh6w`
  (17 jobs) -- the same ids that exist as job dirs under `work/.../lexlat_k2_jobs/`, i.e. the hash
  of the job that actually ran on 2026-09-21 is unchanged.  `LexlatK2Spec.chunk_seqs` and
  `describe()["chunk_seqs"]` reach only a `print` (`sae_blankfree.py:499`) and an output json,
  never a hash.  The pack config the dispatch named cannot be censused because it REFUSES TO BUILD:
  `config_sae_4a_lexlat_k2_pack_v1.py:286` raises `PROBE_SEC_PER_SUBEPOCH is not stated`, so no
  training arm using `lexlat_k2_train` is in any graph yet and nothing there can move.
* **The timeout reads the task's real remaining clock.**  `lexlat_k2_jobs.py:654`
  `budget = max(60, rqmt["time"]*3600*0.90 - (monotonic()-started))` with `started` at the top of
  `run()`; `PROBE_TIME_HOURS = 2.0` for all four probes and `settings.py:120` clamps time to
  `min(11.5, ...)`, so 2.0 h is the real Slurm limit and the budget is ~1.6 h after the batch dump.
  This is the sibling idiom of the HLG build (`:247`, at 0.95).  `_run_with_timeout` starts the
  child with `start_new_session=True` and kills the PROCESS GROUP, SIGTERM then SIGKILL.

## Concerns, most severe first

1. **`lexlat_k2_jobs.py:757` / `:858` -- a FAIL run can print `READ: PASS`, and there is no VERDICT
   line.**  When the child dies after at least one cell, the job now reads `partial.json` and runs
   the normal reads on it.  `per_ma[key]["read"]` is computed from the measured cells only and is
   never forced to FAIL by `child_failure`; `seconds_per_subepoch` is then
   `fmean(seconds) * n_total`, an extrapolation that can rest on ONE cell.  `summary.txt` carries a
   `FAIL: <cause>` header line but no `VERDICT:` line -- unlike the sibling jobs
   (`lexlat_k2_arm_jobs.py:219`, `lexlat_k2_overcount_exact_jobs.py:442`), which both write one.
   The concrete failure: the downstream consumer of this file is a HUMAN transcription into
   `PROBE_SEC_PER_SUBEPOCH` (`config_sae_4a_lexlat_k2_pack_v1.py:286`), which gates the training
   arm; a `READ: PASS` and a seconds figure lifted from a crashed run would fund route A at a cost
   that was never measured.  This is operationally live, not hypothetical: 3 rungs x 9 batches = 27
   cells must fit in ~1.6 h of child budget, so a mean cell above ~210 s ends in exactly this state.
   Minimal fix: set `read`/`time_read`/`memory_read` to FAIL whenever `child_failure is not None`,
   and/or write one `VERDICT: PASS|FAIL` line.
2. **`lexlat_k2.py:938-971` -- the pre-flight guard can never pass on this graph, so it warns on a
   good run.**  `window_arc_bound = chunk_seqs * out_degree * 30 * max_active` with the live escape
   HLG's `out_degree = 399,785` gives, at the configuration the fix believes is safe
   (chunk 16, max_active 1000), `1.92e11 = 89 x (2^31-1)` -> `WARNING: ABOVE k2's int32 window sum
   -- reduce chunk_seqs or max_active`; at max_active 10000 it is 894x, and even at `chunk_seqs=1`
   it is 5.6x.  The bound is a valid worst case but its threshold is unreachable for this graph, so
   the line is printed identically for the chunk size that faulted and the one that passes, carries
   no information, and never refuses or shrinks anything (`chunk_guard` only logs).  Concrete
   failure: a reader acting on the WARNING shrinks `chunk_seqs` and spends another GPU allocation,
   or discounts a genuine PASS.  The debug report holds the empirical figure that would make the
   guard discriminating -- the 57-sequence overflow value `-1565110379` implies ~4.8e7 arcs per
   sequence per 30-frame window, i.e. ~0.36 x 2^31 at 16 sequences -- and it was not used.
3. **`lexlat_k2_jobs.py:511` -- the GPU-check child still has no clock.**  It runs
   `sp.run(check_cmd, env=_child_env(), capture_output=True, text=True)` with no `timeout`, before
   `budget` exists.  `K2_ABORT=1` now makes a k2 FATAL exit there, but any other wedge (a sticky
   CUDA context, a busy device) still idles to the Slurm limit with nothing written -- the exact
   outcome this commit set out to make impossible.  Cheap to close with the same
   `_run_with_timeout`.
4. **`lexlat_k2_jobs.py:654` reads the UNCLAMPED rqmt.**  `settings.py:120` clamps
   `current_rqmt["time"] = min(11.5, ...)` at submit time, while the budget uses
   `self.rqmt["time"]`.  At `PROBE_TIME_HOURS = 2.0` (and `BUILD_TIME_HOURS = 6.0`,
   `TIME_HOURS_4G = 8.0`) the two agree, so no current run is affected; if a probe's `time_rqmt`
   were ever raised above 11.5 the child budget would exceed the real Slurm limit and the timeout
   would never fire.  Same latent shape in the build job at `:247`.  Memory rule
   `constants-trace-to-same-scale-reference` names this trap.

## Minor, no failure path

* `lexlat_k2.py:1040` keeps every chunk's lattice in `chunks`, so the cell's peak is
  max-over-chunks PLUS the accumulated lattices; at ~0.5M arcs per chunk x 8 chunks (~0.1 GB)
  this is immaterial against the 80 GiB bar, but the docstring's "the MAX over them and never
  their sum" is not literally true.
* `LexlatK2Runtime.log_z_hlg` uses `backward=False`, so ALL chunks' autograd graphs coexist until
  `term.backward()`, whereas the probe frees each chunk's graph inside the loop.  The probe's
  memory number is therefore a lower bound for the training step it is meant to price.  The
  retained quantity is the lattice plus its arc map, so the gap is small -- but it is a difference
  between the configuration measured and the configuration that will run.
* `_expected_words`' docstring says the per-chunk counts are "scattered back at that offset"; the
  code concatenates (equivalent, and `start` is unused by that reader).
* `K2_ABORT=1` in the shared `_child_env()` also reaches `LexlatHLGBuildJob`,
  `lexlat_k2_official_jobs` and `lexlat_k2_arm_jobs`.  Disclosed by the implementer; it changes
  fatal-error handling only, no result and no hash, and those jobs' outputs are banked.
* No other difference from the baseline was found in the five touched files: the diff is the
  chunking, the guard, the child clock, K2_ABORT, the FAIL path and their tests, and nothing else.

## What this review does NOT establish

That the 114-sequence batch now completes on the GH200 at 16 sequences per call, or what the leg
costs against E1's bar.  Both are only known after the job runs.  Neutrality of chunking under an
ACTIVELY ADAPTING beam is argued from k2's per-FSA `dynamic_beams_`, not measured.
