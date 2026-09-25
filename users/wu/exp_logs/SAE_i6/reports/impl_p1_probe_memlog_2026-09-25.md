# impl: P1 probe memory logging (G1.M) and the CUDA NaN-aware monitor compare (2026-09-25)

Status: DONE. Nothing was launched and nothing was committed. No file under `model/`, `training/` or
`analysis/` was touched. The parity log `log/p1_rt_parity.4361250.out` was read but not modified.
Inputs: `reports/review_p1_arms_chunked_backward_2026-09-25.md` (findings F1 and F4),
`reports/impl_p1_arms_chunked_backward_2026-09-25.md`, and the coordinator's follow-up on Slurm 4361250.
Package path below: `P = recipe/i6_experiments/users/wu/experiments/unsupervised_asr`.

## Outcome

- **F1 fixed (logging only).** Before its `reset_peak_memory_stats()`, the per-chunk k2 step now reads
  `max_memory_allocated()` and `max_memory_reserved()`. These cover everything since the previous reset,
  including the stability read. It also samples the whole device's usage through `mem_get_info`. The new
  values are three new `lexlat_k2_*` monitors. On CPU the fixture shows the term, the total loss, every
  monitor that was already there, and every gradient are bit-identical to the held path; every recorded
  deviation is 0.0.
- **F4 fixed.** rt_r100 now carries `torch_log_memory_usage = True`, as rt_r90 does. No job id changed.
- **The GPU parity failure (Slurm 4361250) is a test defect, not a code defect.** `lexlat_k2_stability`
  is NaN in both paths because the fixture's rung equals the reference rung, so the held stability read
  does not run by design (details below). The CUDA comparison now accepts "both NaN", as the CPU tests
  already do, so the gradient asserts will run. The GPU test has NOT been re-run; the command is below.

## Diff summary

`P/reverse_model/rt_chunked_backward.py` (284 -> 311 lines):
- **Module docstring.** A new paragraph "THREE MEMORY MONITORS ARE ADDED" says what each monitor covers
  and how G1.M combines them.
- **`step`, before the reset (now line 161).** On CUDA, after the existing `synchronize`, the step reads
  `pre_alloc_gib = max_memory_allocated()/2**30` and `pre_reserved_gib = max_memory_reserved()/2**30`.
  Both are 0.0 on CPU, where no CUDA call is made. The reset is unchanged.
- **`step`, after the existing `peak_gib`.** On CUDA, `free, total = torch.cuda.mem_get_info(log_q.device)`
  gives `device_used_gib = (total - free)/2**30`. It is 0.0 on CPU.
- **Monitors.** Three monitors are inserted right after `lexlat_k2_peak_reserved_gib` and before
  `lexlat_k2_stability`:
  - `lexlat_k2_pre_peak_allocated_gib`
  - `lexlat_k2_pre_peak_reserved_gib`
  - `lexlat_k2_device_used_gib`

  They go into the same `monitors` dict. The held `model/train_step.py:209-211` emits that dict with
  `mark_as_loss(..., as_error=True)`, and RETURNN's `total_loss` skips `as_error` entries
  (`recipe/returnn/returnn/frontend/run_ctx.py:423`). So they cannot enter the loss. The existing
  `lexlat_k2_peak_reserved_gib` (the per-k2-leg peak) is kept unchanged.

`P/tests/test_rt_chunked_backward.py` (338 -> 371 lines):
- **`MEMORY`.** A new constant holding the three new keys.
- **`_compare`.** Asserts that the new keys are in the chunked path's monitors and absent from the held
  path's. The order of all other keys must still equal the held order.
- **`test_full_train_step_theta_and_phi`.** The loss-name order is compared with the new keys excluded.
  For each new key it asserts `as_error` is True and the value is 0.0 on CPU. The record now also stores
  each loss's `as_error`.
- **New `test_memory_monitors_emitted_and_inert`.** It asserts:
  - the three keys exist, in order, right after `lexlat_k2_peak_reserved_gib`;
  - each is 0.0 on CPU and absent from the held path;
  - the term and total are exactly equal to the held path's;
  - `g_log_q` and `g_bias` are `torch.equal` to the held path's;
  - the full `_compare` passes.
- **`test_chunked_backward_equals_held_cuda`.** The monitor comparison now passes when both values are
  NaN; otherwise it uses `pytest.approx(rel=1e-9, abs=1e-12)` as before. The gradient asserts are
  unchanged (atol 1e-7), and all tolerances are kept. On CUDA it also asserts the three memory monitors
  are finite and > 0, and prints them as a `[record]` line.

`config/sae_i6_p1_ladder.py` (155 -> 157 lines):
- **`MEMORY_LOG_ARMS = (PROBE_ARM, "rt_r100")`.** `rt_arm` passes `memory_log=(arm in MEMORY_LOG_ARMS)`
  (was `arm == PROBE_ARM`), and the docstring line was updated to match. The second-seed arm stays
  `memory_log=False`, which the dispatch did not ask to change.

## How G1.M reads the probe (what these monitors measure)

The arm config has `log_verbosity = 5`, so RETURNN prints `eval_info`, including every `as_error`
monitor, on each step's line. With `torch_log_memory_usage` it also prints `mem_usage:cuda:0`.

- **`lexlat_k2_pre_peak_allocated_gib` at step n** covers the span from step n-1's reset to step n's
  reset. That is step n-1's k2 leg, backward and optimizer step, then step n's forward and, once per
  sub-epoch, the stability read. At step 1 of a sub-epoch it covers the span since RETURNN's epoch-start
  reset (`engine.py:325`).
- **`mem_usage:cuda:0` at step n** covers the span from step n's reset through its backward and
  optimizer step.
- **The sub-epoch peak allocated** is therefore the max over steps of both values. The pre-reset read
  covers the stability read (step 1 of the sub-epoch, because on-set 1 is the first sub-epoch where the
  leg runs). The reserved-memory counterpart is the max of `lexlat_k2_pre_peak_reserved_gib` and
  `lexlat_k2_peak_reserved_gib`.
- **`lexlat_k2_device_used_gib` is a point sample, not a peak.** It is taken once per step, at the end of
  the k2 leg. It includes the CUDA context and any memory allocated outside the torch caching allocator
  (for example by k2). It stands in for the unavailable `nvidia-smi` (F2).
- **RETURNN's sub-epoch summary averages these columns.** Maxima must be read from the per-step lines.

## Why `lexlat_k2_stability` is NaN in the held path (Slurm 4361250)

It is a by-design "no read" sentinel caused by the fixture. It is not a defect of the stability read on
CUDA.

- **The fixture's rung equals the reference rung.** The fixture runtime uses
  `WIDE = dict(max_active=10000, ...)` (`P/tests/test_model_lexlat_k2.py:35`, applied by `runtime()`,
  `:67-71`). The reference rung is `STABILITY_REFERENCE_MAX_ACTIVE = 10_000`
  (`P/model/lexlat_k2_train.py:119`).
- **The held read then returns NaN on purpose.** `LexlatK2Runtime._stability` sets the value to NaN
  (`:614`). The guard `if ref <= int(self.spec.max_active):` (`:619`) prints "the stability reference
  rung (10000) is not above this arm's rung (10000); lexlat_k2_stability is nan (no read), not 0"
  (`:622-625`) and returns NaN (`:626`) without intersecting. The chunked class inherits this method
  unchanged.
- **The parity log shows exactly that path.** `log/p1_rt_parity.4361250.out` contains this print 4
  times: held and chunked, for chunk sizes 1 and 4.
- **The CPU run gives NaN too.** In this session's CPU run of `tests/test_rt_chunked_backward.py`, the
  same print appears 94 times, and the "stability at sub-epoch" line of a real read appears 0 times. The
  CPU tests passed only because `_compare` (line 104 before, same logic now) and the full-step test
  (line 299 before) treat "both NaN" as equal.
- **Consequence for G0.R2 and G1.M.** The real rt arms run at rung 1000 < 10000, so the read runs there.
  No test in this suite exercises an actual stability read on CPU or CUDA, because the fixture's rung
  equals the reference. Whether the read completes on the L40S at the arms' batch shape is shown only by
  the probe's "stability at sub-epoch 1: median ..." line, which must not say `FAILED`.

## Checks run (CPU, this node; sae python; `PYTHONPATH=recipe:recipe/returnn:sisyphus`)

- **`tests/test_rt_chunked_backward.py`:** 39 passed, 2 skipped. The skips are the two GPU cases, "CUDA is
  not available". Before the fix there were 38 passed and 2 skipped; the new test accounts for the extra
  one. All 31 `[record]` lines show deviation 0.000e+00: log Z, monitors, `g_log_q`, the parameter
  gradient, and theta/phi/other in the full step.
  - A first run failed 4 cases (`test_full_train_step_theta_and_phi`). My new assert read `.as_error`
    from a stored tensor instead of the Loss object. I fixed the test by recording `as_error` per loss,
    and the rerun passed.
- **`tests/test_model_lexlat_k2.py` + `tests/test_model_blankfree.py`:** 117 passed, 2 skipped (gpu),
  4 xfailed.
- **`tests/test_reverse_ladder.py` + `tests/test_training_config.py`:** 22 passed. With the line above,
  that is 139 passed, the same as the review's count.
- **NOT run:** the GPU tests. This node has no GPU, and launching was out of scope. The CUDA path of the
  new monitors (`max_memory_*` before the reset, `mem_get_info`) has therefore not been executed. Only
  the probe shows that the monitors reach the RETURNN log.

## Dry build: job ids (read-only `sis console -s`, sae python, setup dir)

Ids were dumped before any edit and again after, with the same script (first column = job path).

| build | jobs before | jobs after | id diff |
|---|---|---|---|
| `P1_LADDER_STAGE=probe` | 87 | 87 | empty |
| `P1_LADDER_STAGE=arms` | 113 | 113 | empty |
| `STAGE=arms P1_LADDER_R100=1` | 130 | 130 | empty |
| `config/sae_i6_p0.py` | 164 | 164 | empty; also empty against `analysis_out/g0g_p0_jobids_before_2026-09-25.txt`; loaded package module list unchanged |

- **Reviewed lists.** The probe and arms lists, before and after, are identical to the first column of
  the reviewed `analysis_out/p1_ladder_jobs_{probe,arms}_2026-09-25.tsv`. Those files are the full
  durable lists.
- **Arm training jobs, the same before and after:**
  - rt_r70: `ReturnnTrainingJob.9lD2HS2Mzgcl`
  - rt_r80: `ReturnnTrainingJob.5CL3pQyyOvQt`
  - rt_r90: `ReturnnTrainingJob.EexT85vdfx25` (probe and arms stages)
  - rt_r100: `ReturnnTrainingJob.M1jS5YvLGi1O`
- **Probe-stage jobs outside P0 (25):**
  - rt_r90 `EexT85vdfx25`;
  - the fits `ruLnJFWyifwp`, `6IkAAuzcBwBR` and `uO8wkodbR2uh`, with their CorruptSeedGoldJob,
    PhoneTargetHdfJob and SupervisedReverseDataJob (3 each);
  - 4 each of ExtractSubmoduleCheckpointJob, ReturnnForwardJobV2 and BlankfreeGreedyPerJob.
- **Dumped arm configs after the change.** r70, r80 and r90 are byte-identical to the reviewed
  `analysis_out/p1_ladder_rt_{r70,r80,r90}_2026-09-25.returnn.config`. r100 differs from the reviewed
  `p1_ladder_rt_r100_2026-09-25.returnn.config` only by the added line `torch_log_memory_usage = True`
  (post_config, not hashed). The flag is set in r90 and r100 only.

## GPU parity re-run (not submitted; the review's Launch A, unchanged)

```
sbatch -A hlt -p gpu_test_24gb --gres=gpu:1 -c 4 --mem=16G -t 0:30:00 -J p1_rt_parity -o /u/hwu/setups/librispeech-960/2026-09-24-unsupervised/log/p1_rt_parity.%j.out --wrap 'S=/u/hwu/setups/librispeech-960/2026-09-24-unsupervised; export PATH=/work/asr4/hwu/conda/envs/sae/bin:$PATH PYTHONPATH=/work/asr4/hwu/setups$S/i6_core/tools/git/CloneGitRepositoryJob.KQ3NuCaDE6QH/output/returnn:$S/recipe:$S/recipe/returnn:$S/sisyphus; cd $S/recipe/i6_experiments/users/wu/experiments/unsupervised_asr && nvidia-smi && python -c "import sys,torch,k2;sys.exit(not(torch.cuda.is_available() and k2.with_cuda))" && python -m pytest -p no:cacheprovider -m gpu -rA -s tests/test_rt_chunked_backward.py'
```

The new job id gives a new log file, so 4361250's log is not overwritten.

PASS read:
- `2 passed`, with no SKIPPED;
- two `[record] cuda chunk {1,4}: max |dlogZ| ...` lines, within log Z 1e-9 and gradient 1e-7;
- two `[record] cuda chunk {1,4} memory monitors (GiB): {...}` lines, with all three values > 0.

## Assumptions and undetermined

- **"No-ops on CPU"** is read as: no CUDA call, and each of the three monitors is emitted with value 0.0.
  This is the existing `lexlat_k2_peak_reserved_gib` convention, and it is needed for the CPU test to see
  the monitors.
- **Where `mem_get_info` is sampled.** It is sampled once per step at the end of the k2 leg, next to the
  existing peak read. The dispatch did not state a point in the step. Because it is a point sample, a
  transient that is freed before the sample, and not held by torch's cache, is not seen by it.
- **Monitor names** are my choice, following the dispatch's example wording.
- **Proposal for `SAE_i6_P1.md` G1.M (orchestrator's decision).**
  - State the peak read as the max over sub-epoch-1 steps of `mem_usage:cuda:0` and
    `lexlat_k2_pre_peak_allocated_gib`.
  - Replace the `nvidia-smi` field with `lexlat_k2_device_used_gib` (max over steps), disclosed as a
    point sample.
