# Review: P1 probe memory logging (G1.M) and the NaN-aware CUDA compare (2026-09-25)

Reviewer: code-reviewer. Scope: the fix in `reports/impl_p1_probe_memlog_2026-09-25.md` on top of the change
reviewed in `reports/review_p1_arms_chunked_backward_2026-09-25.md` (PASS_WITH_FIXES; its MUST fix was F1, the
peak reset at `rt_chunked_backward.py:142`). Nothing was edited, launched or committed.
Package path below: `P = recipe/i6_experiments/users/wu/experiments/unsupervised_asr`.

## Verdict: PASS

The fix only adds logging. It closes F1: with `torch_log_memory_usage`, the per-step lines now cover every
allocation of the training part of a sub-epoch, including the stability read. F4 is also closed: rt_r100 now
carries the memory flag, and no job id changes. The NaN explanation is correct. The NaN-aware compare still
fails when only one path is NaN. Launch A and Launch B stand unchanged. Neither finding below blocks either
launch.

## Findings (most severe first; neither is a MUST before Launch A or B)

F1 (pre-existing; not introduced by this fix) `P/model/lexlat_k2_train.py:650-652` with the rt config's
`stop_on_nonfinite_train_score = True` (dumped r90 config, line 140) and `recipe/returnn/returnn/torch/engine.py:554-591`.
- What happens: a stability read that fails (a caught OOM or other k2 error) sets `lexlat_k2_stability` to NaN.
  The same happens when no utterance gives two finite totals. The step emits that NaN as an `as_error` monitor
  (`P/model/train_step.py:209-211`). RETURNN's non-finite check scans every loss, including `as_error` ones,
  so it raises `Inf/nan score in step 0` after re-running the step once in `debug_inf_nan`.
- Effect on the probe: a failed read at sub-epoch 1 ends the job at step 0, and no `epoch.001.pt` is written.
  G1.M counts this as a miss either way. It contradicts the previous review's F3 ("leaves it training on").
  Before calling a crash a divergence, the executor or debugger should grep for `stability read at sub-epoch 1 FAILED`.
- Effect on the arms, and on a passed probe that continues as rt_r90: if the reference-rung read fails at any
  later sub-epoch (2-8), the arm dies. The read's docstring says the opposite ("MAY NOT KILL THE ARM",
  `:605-607`). The probe shows only that sub-epoch 1's read fits on the L40S. Decide this before the arms
  launch, not before the probe.

F2 `SAE_i6_P1.md:108-109` (amended G1.M) - the wording "the max ... of the pre-k2 peak ... and the k2 leg's
peak" does not name the fields.
- The correct allocated read, per step: the maximum of `lexlat_k2_pre_peak_allocated_gib` and RETURNN's
  `mem_usage:cuda:0`. The second covers the span from the k2-leg reset through RETURNN's backward and optimizer
  step (`engine.py:531-540`, `:1614-1618`).
- The misread: if "the k2 leg's peak" is taken to be `lexlat_k2_peak_reserved_gib`, two things go wrong. That
  value is RESERVED memory, compared against an allocated bound, so it can give a false miss above 40 GiB and a
  wasted cancel. It also stops at the end of the k2 leg, so the outer backward of the sub-epoch's last step is
  covered only by the epoch-end `Memory usage (cuda:0): ... alloc peak` line (`engine.py:648`).
- Units: both allocated reads are GiB. RETURNN's "GB" is `human_bytes_size` with factor 1024 and one decimal
  (`returnn/util/basic.py:945`). The monitor is printed with 3 decimals (`_format_score_value`).

## Check 1: logging only

- Diffs: I compared against the implementer's pre-edit copies in
  `/var/tmp/claude-2764/.../scratchpad/memlog/orig/`. The copies match the reviewed version: 284 lines, the
  stability read at :138, the reset at :142 and `peak_gib` at :170. The test copy has 338 lines, and its line
  333 is the assert shown in the traceback in `log/p1_rt_parity.4361250.out`.
- `rt_chunked_backward.py` diff: 27 pure additions, with no line changed or removed. They are:
  - the docstring paragraph (lines 45-58);
  - `pre_alloc_gib = pre_reserved_gib = 0.0` (154), plus the two `max_memory_*` reads before the unchanged
    reset (157-160), inside `if cuda:`;
  - `mem_get_info` after `peak_gib` (190-193), inside `if cuda:`;
  - the three monitor insertions (198-201).
- Effect on the loss and timing: no line that feeds `term`, `z_hlg`, `z_h`, `upstream`, `leaf`, `_InjectGrad`
  or any existing monitor is touched. The pre-reads come before `t0`, and `mem_get_info` comes after `sec`, so
  `lexlat_k2_sec` does not move.
- Why the monitors cannot enter the loss: they are marked through `train_step.py:209-211` as fresh
  `torch.tensor([float(v)])` with `as_error=True`. `run_ctx.py:423` skips `as_error` in `total_loss`. The rt
  config has `keep_best_n: 0` and an explicit `learning_rates` list for all 8 sub-epochs, so no error key
  selects checkpoints or learning rates.
- CPU: every CUDA call is behind `cuda = log_q.is_cuda`. The CPU tests ran on this GPU-less node, and the
  monitors came out 0.0.

## Check 2: does the pre-reset peak span the stability read; is any peak still missed

- **The first step of every sub-epoch, including sub-epoch 1 of the real arms.**
  - The start of the span: `train_epoch` resets the peak stats (`engine.py:386` -> `:322-326`) before step 0.
  - Inside the step: `_stability` runs at `rt_chunked_backward.py:152`, and the pre-read at `:159-160`, before
    the reset at `:161`.
  - So the pre-read of step 0 covers everything since the epoch-start reset: the recognizer forward, the bed
    lattice and the finite-difference passes, `dense`, and the stability read.
  - The read's `finally: empty_cache()` (`lexlat_k2_train.py:653-655`) lowers the current reserved memory but
    not the peak stat.
- **The bed lattice runs BEFORE the k2 leg, not after.** The float64 `lattice_loss` (`train_step.py:94-97`) and
  `_fd_passes` (`:124`) both run before the leg (`:153`). Their forward peaks fall in the pre-read. Their
  backward runs in RETURNN's outer backward, which is in `mem_usage:cuda:0` of the same step and in the next
  step's pre-read.
- **The union of reads.** At step n, the pre-read covers the span from reset(n-1) to reset(n), and
  `mem_usage` covers the span from reset(n) through that step's optimizer step. Together they cover the
  training part of the sub-epoch without a gap. The epoch-end `Memory usage` line (`engine.py:648`) covers the
  tail.
- **The only unread span** runs from `:648` to the eval-start reset (`:676`). It holds the model and optimizer
  save, which allocates no GPU memory of note.
- **The dev pass.** Each dev step runs the leg under no_grad, where the stability read is cached. Its per-step
  line carries both the monitors and `mem_usage` (`engine.py:745-755`), so the dev pass can be read in the
  same way if G1.M counts it.
- **k2's allocations.** They go through torch's caching allocator: `libk2context.so` references
  `c10::cuda::CUDACachingAllocator::allocator` (`PytorchCudaContext`). So they are in `max_memory_allocated`.
  `lexlat_k2_device_used_gib` is a point sample and is disclosed as one.

## Check 3: NaN = "no read"; the read runs at sub-epoch 1 of rt_r90

- **The NaN.** `lexlat_k2_train.py:614` sets NaN. The guard `if ref <= max_active` (`:619`) prints "no read"
  (`:623-625`) and returns NaN (`:626`).
- **Why the fixture hits it.** The fixture has `WIDE max_active=10000` (`tests/test_model_lexlat_k2.py:35`,
  `:67-71`), which equals the reference rung `STABILITY_REFERENCE_MAX_ACTIVE = 10_000` (`:119`).
- **The evidence in the logs.** The GPU log shows the print 4 times. My CPU run shows it 94 times, with 0 real
  reads.
- **The real rt_r90 config runs the read at sub-epoch 1:**
  - The config states `lexlat_k2_max_active: 1000`, `lexlat_k2_onset: 1` and `lexlat_k2_ramp: 3`.
  - Neither `stability_reference_max_active` nor `stability_seqs` is passed (`blankfree_model.py:548-553`), so
    the defaults apply: 10000 and 16. `STABILITY_CHUNK_SEQS = 1` is a module constant.
  - `lam(1) = 1/3 > 0`, so the leg is active at sub-epoch 1, and 10000 > 1000, so the read runs.
  - RETURNN trains before it evaluates (`engine.py:269-272`, `:659`), so the read runs on training step 0 of
    sub-epoch 1.
- **Not tested anywhere.** No test exercises a real stability read on CPU or CUDA. Only the probe's line
  `stability at sub-epoch 1: median ...` shows it (see F1 for what a failure does).

## Check 4: the NaN-aware compare does not mask a one-sided NaN

- **CUDA compare** (`tests/test_rt_chunked_backward.py:362`). I checked it with pytest 9.1.1 in the sae env:
  both NaN gives True; NaN in the held path only gives False; NaN in the new path only gives False; values off
  by 1e-6 give False. `pytest.approx` has `nan_ok=False`.
- **CPU compares.** `_compare` (`:107-108`) and the full-step compare (`:325`) accept only "both NaN".
- **Gradients.** `torch.allclose` (`:365`, `:117`) rejects NaN.
- **Which monitor is affected.** In 4361250 every monitor before `lexlat_k2_stability` passed on CUDA, so the
  "both NaN" branch fires only for stability.
- **`_compare` is no weaker.** Besides requiring that the memory keys appear only in the new path, it still
  requires that the remaining keys equal the held keys in the same order.

## Check 5: job ids (independent dry builds, `sis console -s`, sae python)

- **Probe stage:** 87 jobs. **Arms stage:** 113 jobs. The sorted ids equal the first column of the reviewed
  `analysis_out/p1_ladder_jobs_{probe,arms}_2026-09-25.tsv`; the diff is empty.
- **Arm training jobs:**
  - rt_r90 `EexT85vdfx25` (probe and arms)
  - rt_r70 `9lD2HS2Mzgcl`
  - rt_r80 `5CL3pQyyOvQt`
  - rt_r100 `M1jS5YvLGi1O` (arms + R100: 130 jobs)
  - Every arm has `-p gpu_48gb`.
- **P0** (`config/sae_i6_p0.py`): 164 jobs; the diff against `analysis_out/g0g_p0_jobids_before_2026-09-25.txt`
  is empty. `rt_chunked_backward` is not in `sys.modules` after the P0 load.
- **Dumped arm configs.** r70, r80 and r90 are byte-identical to the reviewed
  `analysis_out/p1_ladder_rt_*_2026-09-25.returnn.config`. r100 differs only by `torch_log_memory_usage = True`,
  which is in `post_config` and not hashed (`i6_core/returnn/config.py:314-319`).
- **The config diff.** Against the pre-edit copy, the only changes are `MEMORY_LOG_ARMS` and `arm in
  MEMORY_LOG_ARMS`, plus the docstring line.
- **Other arms.** The three monitors are also emitted by r70 and r80, where they have no loss effect. Those two
  arms do not carry `mem_usage`.

## Check 6: tests; Launch A and Launch B

- **CPU tests.** Run with `PYTHONPATH=recipe:recipe/returnn:sisyphus`:
  - `tests/test_rt_chunked_backward.py`: 39 passed, 2 skipped (gpu). All 31 `[record]` lines show deviation
    0.000e+00 (117 values).
  - `test_model_lexlat_k2` + `test_model_blankfree` + `test_reverse_ladder` + `test_training_config`: 139 passed,
    2 skipped, 4 xfailed.
- **Launch A** (the command in `review_p1_arms_chunked_backward_2026-09-25.md`) holds unchanged. Its pinned
  RETURNN path exists. `-m gpu` now selects 2 of 41 tests. PASS read: `2 passed`, no SKIPPED, two
  `[record] cuda chunk {1,4}: ...` lines within log Z 1e-9 and gradient 1e-7, and two `memory monitors` lines
  with all three values > 0. The CUDA test still does not exercise a stability read.
- **Launch B** (`P1_LADDER_STAGE=probe`, one manager) holds unchanged: the same id `EexT85vdfx25`, and the same
  written config. The project rule applies: commit the reviewed code before launch. Only the P0 manager
  (pid 1646677) runs now.
- **G1.M read** at the end of sub-epoch 1, from `log.run.1`:
  - the maximum over the train `step` lines of max(`lexlat_k2_pre_peak_allocated_gib`, `mem_usage:cuda:0`), and
    the epoch-end `Memory usage ... alloc peak`, must be <= 40 GiB (see F2);
  - `lexlat_k2_device_used_gib` (maximum over steps) stands in for `nvidia-smi`;
  - `stability at sub-epoch 1: median` must be present; if the job crashed, grep for `FAILED` (see F1).
