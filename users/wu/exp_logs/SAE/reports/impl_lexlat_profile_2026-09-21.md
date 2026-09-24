# Per-COMPONENT time profile of one lexicalised step -- implementation (2026-09-21)

Commit `66ac4e2` on `haotian_modality_matching_jupiter` (`recipe/2025-10-speech-llm`), three NEW
files, nothing existing touched:

| file | what |
| --- | --- |
| `src/speech_llm/sae/emc/lexlat_profile_jobs.py` | `PhaseTimers` + `LexlatStepProfileJob` |
| `src/speech_llm/sae/emc/test_lexlat_profile.py` | six CPU tests |
| `src/.../librispeech/configs/config_sae_4a_lexlat_profile_v1.py` | the registration, `py()` |
| `config/sae_4a_lexlat_profile.py` (setup dir, not in git) | the shim, `py` / `run` |

Job hash: **`LexlatStepProfileJob.ItFHebFGK6nb`**
(`work/speech_llm/sae/emc/lexlat_profile_jobs/LexlatStepProfileJob.ItFHebFGK6nb`), alias
`sae/4a/lexlat_pack/step_profile`, outputs under
`output/exp2025_11_06_speech_llms/librispeech/sae_4a_lexlat_pack/step_profile/`.

## 1. What the job measures

The step being decomposed is the one `LexlatEfficiencyProbeJob.x3LaY6KlB6I4` priced: C = 1024,
`lam_lex = 1`, sub-epoch 10 of `lexlat_20` from the pinned `ctrl_50` checkpoint, the pack's
`batch_size = {"features": 88,000}` / `max_seqs = 128`, 802-912 s per complete update against
11-12 s for the banked `lattice.py` path (its `log.run.1`, the `[lexlat] step N` lines).

**(a) The coarse, CUDA-synchronised phase table.** `PhaseTimers` patches twelve module attributes
of `lexlat` with a call-through wrapper (the idiom the E1 probe's `_dest_block` spy already uses,
with the same restore-in-`finally`):

`lexlat_forward_backward`, `lexlat_log_z`, `_forward`, `_replay`, `_step`, `_frame_slots`
(the trie candidate expansion), `lm_score` (the word-LM CSR lookup), `_slot_u` (the
per-destination-block gather), `_select` (the top-C pruning and its renormalisation),
`_band_reduce` (the forward band reduction), `_logmm` (the log-domain products), `_final_weight`.

Each phase is keyed by the **leg of its caller**: `lexlat_forward_backward` opens `backward`, so
everything it calls directly is the manual backward's own work, and `_forward` / `_replay`
override it with `forward` / `replay`.  The rows therefore read `forward.slot_gather`,
`backward.slot_gather`, `replay.trie_expand`, ... and the manual backward is decomposed by the
same names as the forward.  Every open phase carries a child accumulator, so a phase's
**exclusive** time is its own elapsed time minus its wrapped children's, and

    sum(exclusive over every phase) + UNACCOUNTED = the step's wall clock

with `UNACCOUNTED` the python frame loop, the host-device copies and everything between the
wrapped calls -- the brief's "time not inside any phase".  The four coarse stages E1 already times
(`h2d`, `train_step_fwd`, `loss_total+backward`, `optimizer_step`) are reported beside the table,
and each `lexlat_forward_backward` call of the step is reported separately (`#1` the loss DP,
`#2..` the finite-difference tilted passes of the rate term -- three DP calls per step at this
arm's settings).

**(b) `torch.profiler` over ONE step**, CPU + CUDA activities, `record_shapes=True`, with the phase
timers in COUNT mode so nothing this job does synchronises inside the trace.  Reported: total self
CUDA time over all kernels against the step's wall clock (hence the GPU-busy and the
idle/launch-bound fraction), the CUDA kernel launches divided by the frame steps **counted in the
same step** (so "launches per frame" is a ratio of two measured counts), the top 25 CUDA kernels by
self CUDA time and the top 25 ops by total CUDA time, torch's own tables verbatim including the
shape-resolved one, and the Chrome trace.

**Also reported**: the per-frame time series of the forward (`per_frame.json`: one row per
`_step` call with its seconds, `n_candidates` from `_frame_slots`, `n_contexts_kept` and
`max_multiplicity`, tagged by DP call and leg, so a cost that scales with candidates per frame is
visible as such) and `torch.cuda.max_memory_reserved` / `max_memory_allocated` per step and over
the run.

## 2. The timers change no value, and how that is checked

The wrapper adds a python call, a `torch.cuda.synchronize()` on both sides and a `perf_counter()`.
No chunk width, no dtype, no order of operations moves; `_dest_block`, `_batch_block`,
`DEST_BLOCK_BYTES` and `BATCH_BLOCK_BYTES` are untouched.

* **At the run shape**: the first profiled batch runs ONCE with no wrapper installed at all, and
  the timed step's and the profiled step's total loss (and every named loss) must equal that
  reference within `LOSS_REL_TOL = 1e-6` relative; the report says whether it was bit-identical.
  The band is there because several kernels of a real step accumulate with atomics (`scatter_add_`),
  whose summation order is not reproducible on CUDA -- two identical un-instrumented steps can
  differ in the last bits for the same reason.  This is a sanity assert, not a measurement.
* **On CPU, where the DP is deterministic**, the strong statement is tested:
  `test_the_timers_leave_every_value_bit_identical` asserts every field of `LexlatOutput` is
  `torch.equal` with the timers in `time` mode and in `count` mode, and that every patched
  attribute is restored to the *same object*.

**Disclosed cost**: a synchronised timer serialises the launch queue, so the timed step is slower
than the un-instrumented one.  The job reports both wall clocks and their ratio, and every
percentage in the phase table is a percentage of the **timed** step -- a decomposition of that
step, never a claim about the 802-912 s one.  The un-instrumented reference step is the number to
compare with E1.

## 3. What runs, and the one reading of the brief that had to be fixed

Batch selection is E1's own sampler: `LexlatEfficiencyProbeJob._plan` imported and called UNBOUND
on a namespace carrying `n_steps = 100`, `n_points = 4`, `max_timed_batches = 9`
(`pack.E1_MAX_TIMED_BATCHES`) -- the idiom `test_lexlat_train` already uses -- cut to its first
`n_profile_batches = 2` indices.

The brief says "the first 2 timed batches of the existing sampler (**largest-T batch + one sampled
batch**)".  On the measured sub-epoch the two do not coincide: 57 batches, longest T at index 48,
so E1's plan is `[0, 1, 14, 15, 28, 29, 43, 44, 48]` and the largest-T batch is the LAST of it, at
T = 1137.  The first-two reading is what runs, because the shape the brief itself names
(B about 120, T about 700, m_max about 3,800) is exactly batches 0 and 1 (B = 114 / T = 770,
867.0 s and B = 127 / T = 689, 911.6 s in `x3LaY6KlB6I4`'s log), and those are the seconds being
decomposed.  **If the largest-T batch was wanted instead, `n_profile_batches` does not express it
and the selection would need a new argument -- say so and it is a two-line change.**

Per profiled batch: the phase-timed lexicalised step and the banked trigram-only step (the train
step's own fall-back branch, `model.lexlat` dropped for the duration, same cold parameters).  On
the first batch additionally the un-instrumented reference step and the `torch.profiler` step.
Four lexicalised steps at about 900 s plus two banked ones plus the loader's shape walk (about
96 s in E1) is about 1.1-1.3 h against the 2 h allocation.  The parameters are restored to the
pinned checkpoint before every step, outside the timing, and the RNG is seeded from
`(epoch, index, random_seed)`, so the four runs of batch 0 are the same step.

`summary.txt`, `profile.json`, `per_frame.json` and `profile_top_kernels.txt` are rewritten after
every completed batch (E1's `partial.json` lesson: an OOM or a time-limit kill in the priced path
must not yield zero information); `trace.json.gz` is written when the profiled step completes, and
a failed export leaves a gz holding the error rather than an absent output.

## 4. Operating point, asserted not re-stated

The config states no shape, no checkpoint, no C and no sub-epoch of its own: it calls
`config_sae_4a_lexlat_pack_v1._arm_train_config` / `_flat_init`, `config_sae_4a_attrib_v1._control`
/ `_priorshuf_prior` and `config_sae_4a_lexlat_probes_v1._pin` with exactly the arguments
`efficiency_probe(max_contexts=E1_CONTEXTS_LOWER)` passes.  `MAX_CONTEXTS = pack.E1_CONTEXTS_LOWER`
(= 1024) is the C the measured step ran at, never re-typed.  The job re-asserts, at job start,
E1's own four: `batch_size`, `max_seqs`, `runtime.spec.max_contexts == C` and
`runtime.lam(epoch) == lam_lex`.

`rqmt = {"cpu": 16, "mem": 64, "gpu": 1, "gpu_mem": 96, "time": 2.0}`.  The brief fixes 1 GPU,
64 GB and 2 h; `cpu` and `gpu_mem` are E1's own (the bed's GH200 requirement and its loader
workers) -- the step reserved 41 GiB and a smaller card would not run it at all.

Outputs: `summary.txt`, `profile.json`, `profile_top_kernels.txt`, `trace.json.gz`,
`per_frame.json` and `returnn.config`.  The brief names four; `returnn.config` is the config the
job must write to load it (E1 has the same output), and `profile.json` is the machine-readable
twin of `summary.txt` (E1's `efficiency.json` idiom) so a reader does not have to parse the table.

## 5. Checks

* **Tests, `test_lexlat_profile.py`: 6 passed** (CPU, 3 s).
  1. `test_phase_timer_accounting_sums_to_the_step_wall_clock` -- on one toy
     `lexlat_forward_backward` call, `sum(timers.exclusive)` against a `perf_counter` the TEST
     takes around the call: within **2 %** (the brief's bar).  The check is deliberately not on
     the printed table, whose `UNACCOUNTED` row is the wall clock minus that sum and would close by
     construction; it catches double counting, a lost accumulation and a timer overhead over 2 %.
     It also asserts no negative exclusive time, that all four legs (`step`, `forward`, `replay`,
     `backward`) fired, and that the per-frame series carries the candidate counts.
  2. `test_the_timers_leave_every_value_bit_identical` -- every `LexlatOutput` field
     `torch.equal` in both modes; every patched attribute restored to the same object.
  3. `test_count_mode_counts_the_frames_the_launch_ratio_divides_by` -- `forward.frame_step`
     counts exactly `t_max` frames and count mode times nothing.
  4. `test_the_plan_is_e1s_own_cut_to_the_profiled_batches` -- `_profile_plan` equals
     `LexlatEfficiencyProbeJob._plan` at (57, 48) and the profiled set is `[0, 1]`.
  5. `test_the_phase_table_is_rendered_into_the_summary` -- every phase row of a REAL timer run
     appears in `summary.txt`, with the percent column, the `UNACCOUNTED` row, the
     idle/launch-bound line and the launches-per-frame line (the "computed but never rendered"
     trap).
  6. `test_the_profiler_reader_runs_against_this_torch` -- `key_averages()` field names, both
     sorts, the degenerate no-CUDA case and the `.json.gz` chrome export, on torch 2.7.1.
* **The existing suites are unaffected**: `test_lexlat.py` + `test_lexlat_train.py` **74 passed,
  1 skipped**; with the new file, 79 passed + 1 skipped.
* **Census** (one process per entry point, `tk.sis_graph.jobs()` sorted):
  `config_sae_4a_prepro_pack_v1` **133**, `config_sae_4a_budget_pack_v1` **775**,
  `config_sae_4a_lexlat_probes_v1` **3**, `py_e1` **10** with
  `LexlatEfficiencyProbeJob.x3LaY6KlB6I4` unchanged -- the running probe keeps its hash.  Nothing
  could have moved: this commit adds files and edits none, and the new module is imported by no
  existing config.  The new graph is **9 jobs**, and its diff against `py_e1`'s is exactly
  "`LexlatEfficiencyProbeJob.x3LaY6KlB6I4` + `LexlatWordCountsJob.1ZJy5dFbOAHD` out,
  `LexlatStepProfileJob.ItFHebFGK6nb` in": the eight upstream jobs are byte-identical, so the
  profile runs on E1's own inputs.
* **The shim loads through the loader's own lookup**: `config/sae_4a_lexlat_profile.py` exposes
  `py` (and `run`), returns the same `ItFHebFGK6nb`, the six declared outputs and the rqmt above.
* Not run: the job itself (no launch in this round).  **Loading proves the graph, not the
  measurement**: no phase number exists yet.

## 6. What could not be timed without changing a value, and other limits

* **Nothing inside the wrapped set had to be modified.**  Every phase is timed by patching a module
  attribute; `lexlat.py`, `lexlat_train.py` and `lattice.py` are untouched, so this is hash-neutral
  for `PackedBlankfreeTrainJob` and for both E1 probes.
* **Sub-phases of the manual backward's arc section cannot be separated without editing
  `lexlat.py`.**  The masking, the `_pad_by_k`/`_unpad_by_k` padding, the `torch.exp` accumulations
  into `seg_post_pad` / `post_q` / `acc[*]` and the batch-row loop are straight-line code inside
  `lexlat_forward_backward`, not callables; they appear as the `backward` leg's share of the
  UNACCOUNTED row, not as named phases.  Timing them individually would mean inserting timers in
  the DP itself -- possible and still value-neutral, but it is an edit to a shared file that both
  E1 probes and the pack import at run time, so it was not done.  Say so if the backward's
  remainder turns out to dominate and it is wanted.
* **`_slot_add`, `_compact`, `_group_by_key`, `_band_group`, `_seg_lse`, `_pad_by_k`,
  `_unpad_by_k`, `_scatter_lse` are deliberately NOT wrapped**: the brief asks for a COARSE table
  and every wrapper costs two CUDA syncs at a frame-inner call site (`_slot_u` alone is called
  about 70 times per frame at this C).  They fall into their leg's remainder.
* **The banked leg is timed coarsely only** (its four stages).  It runs `lattice.py`, which shares
  none of the wrapped names, so a phase table of it would be empty by construction.
* **The synchronised timers inflate the step** (serialised launch queue).  Measured as
  `timed / reference` and printed; the reference step is the comparable number.
* **The profiler's own overhead** is not separable from the profiled step either; its wall clock is
  reported beside the reference's so the inflation is visible.
* **Trace size is a real risk**: the step runs roughly 1,300 frame steps (three DP calls x about
  256 frames, plus the replay), each launching hundreds of kernels, so the event count is of order
  a million.  The export is wrapped: a failure records itself in `summary.txt` and leaves a gz with
  the error, and the kernel tables (from the same step) are written either way.  If kineto drops
  events under its buffer limit, the launch count is a lower bound -- the summary reports the raw
  counts so this is visible.

## 7. For the plan, not acted on

* `LexlatStepProfileJob` reads no bar and gates nothing; it is a decomposition of E1's step.  E1
  (`x3LaY6KlB6I4` at C = 1024, `fY18YN7LVdAe` at C = 4096) still owns the funding decision.
* The profile is at **C = 1024**, the C of the measured 802-912 s steps, not the pack's re-declared
  C = 4096.  A profile at 4096 is the same job with `max_contexts=pack.MAX_CONTEXTS` and its own
  tag; per the E1 OOM report's section 5 the per-frame set there is 4-12x larger, so it would need
  its own allocation.
