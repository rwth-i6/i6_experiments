# Review: P1 rt arms, a non-finite stability read no longer stops the arm (2026-09-25)

Reviewer: code-reviewer. Input: `reports/impl_p1_stability_nan_guard_2026-09-25.md`. Baseline: the version that
passed re-review in `reports/review_p1_probe_memlog_2026-09-25.md`. Nothing was edited, launched or committed.
P = `recipe/i6_experiments/users/wu/experiments/unsupervised_asr`, S = the setup dir.

## Verdict: PASS

The change does exactly what the dispatch asked for, and nothing else. The code may be committed and Launch A may
go. There is no MUST. Two SHOULD findings concern how G1.M's stability clause is read and acted on. Both are
needed before Launch B, and neither needs a code change.

## Findings (most severe first; neither blocks the commit or Launch A)

S1 `reports/review_p1_arms_chunked_backward_2026-09-25.md:165` and `reports/review_p1_probe_memlog_2026-09-25.md:161`.
- What breaks: the documented G1.M read is "`stability at sub-epoch 1: median ...` present, and not `FAILED`".
  That read passes an EMPTY read. An empty read prints
  `lexlat_k2: stability at sub-epoch 1: median nan nats per retained frame over 0 of 16 utterances (...)`
  (`P/model/lexlat_k2_train.py:647-649`). The line contains the searched text and does not contain FAILED.
- When: before this change, that NaN stopped the probe at step 0, so "`epoch.001.pt` written" failed and the miss
  was caught anyway. With the guard, sub-epoch 1 completes and every other check on the list can pass.
- Read instead: the median is finite and N >= 1 in "over N of 16". Equivalently, `train_loss_lexlat_k2_stability`
  is present in the epoch-1 entry of the probe's `learning_rates` file. With the guard, the key is present
  exactly when the read produced a number.

S2 `SAE_i6_P1.md:110-111` (G1.M amendment: "A probe over 40 GiB that has not crashed is cancelled at the read").
- What breaks: the cancel rule covers only the memory miss. With the guard, a FAILED or empty stability read at
  sub-epoch 1 is also a miss that does not crash. Under the current text, the probe trains on into sub-epoch 2,
  and the only barrier to it continuing as rt_r90 is that someone reads line 105.
- The memory clause does not catch this case either. When the read's reference rung OOMs, the exception is
  caught. The allocation that failed never enters `max_memory_allocated`, so `lexlat_k2_pre_peak_allocated_gib`
  and `mem_usage` can both read <= 40 GiB even though the reference rung did not fit.
- Fix: extend the cancel rule to a failed or empty sub-epoch-1 read, read as in S1.

## Check 1: the delta is exactly the intended one

- **Baseline copy.** I used the implementer's pre-edit copy (`<scratchpad>/rt_chunked_backward.orig.py`, 311 lines).
  - I checked it independently: its diff against the pre-memlog copy (`<scratchpad>/memlog/orig/`) is exactly the 27
    additions that the memlog review lists (docstring 45-58, 154, 157-160, 190-193, 198-201).
  - The test baseline (371 lines) has the NaN-aware CUDA compare that the memlog review describes.
- **Module diff (+20 / -2).** It has three parts:
  - the docstring paragraph (`rt_chunked_backward.py:59-71`);
  - `import math` (:82);
  - `if math.isfinite(float(stability)):` around the stability monitor (:216-222).
- **Nothing else changed.** No other line differs. The body of my test diff is identical to the implementer's.
  Since the memlog review, only these two files changed under P, `config/`, `analysis/` and `settings.py`
  (`find -newer`).
- **Score and gradient.** Every monitor is marked `as_error` (`P/model/train_step.py:209-211`), and
  `RunCtx.total_loss` skips `as_error` losses (`recipe/returnn/returnn/frontend/run_ctx.py:423-424`). Omitting the
  key therefore moves no total loss and no gradient.
  - `train_step` iterates over the dict. No code indexes `lexlat_k2_stability`: grep finds only
    `lexlat_k2_train.py:583` and `rt_chunked_backward.py:221`, both of which write it.
- **The non-finite stop.** RETURNN scans every accumulated loss (`engine.py:554-555`). Only the stability key is
  conditional. The term `lexlat_k2`, every bed loss and every other monitor are still marked unconditionally.
  - The Engine tests confirm this on CPU: `held_stability_nan`, `total_nan` and `other_monitor_nan` each raise
    `Inf/nan score in step 0`; `failed` and `empty` finish the sub-epoch.
  - The guard cannot hide a real divergence. A NaN posterior makes `log Z_H` non-finite, the assert at
    `rt_chunked_backward.py:187` fires, and the term itself is NaN.
- **No stand-in number.** The held `_stability` (untouched) keeps NaN, and the rt step omits the key. Nothing
  substitutes a value.
- **RETURNN's error key.** `LearningRateControl.get_error_key` picks its key per epoch from that epoch's own dict
  (`learning_rate_control.py:379-419`), so an epoch without the key cannot trip its assert. The rt configs set
  `keep_best_n: 0` and give an explicit `learning_rates` list.

## Check 2: consumers of the absent key

- **Code.** No code reads `lexlat_k2_stability` in `config/`, `analysis/` or P outside the two writers above. The
  ladder only registers `learning_rates` as an output (`config/sae_i6_p1_ladder.py:83`). No reader can crash on
  the missing key, and none can take it as a pass.
- **Gates.** G1.R70 and G1.L read PER only. G1.M's stability clause is the only gate read that is affected
  (S1, S2).
- **How G1.M's "the sub-epoch 1 stability read completes" is read now.**
  - It completes when `log.run.1` of `EexT85vdfx25` has `lexlat_k2: stability at sub-epoch 1: median <finite>
    nats per retained frame over N of 16 utterances (|log Z(1000) - log Z(10000)|)` with N >= 1, and the epoch-1
    entry of `learning_rates` has `train_loss_lexlat_k2_stability`.
  - It is a miss when the log has `lexlat_k2: the stability read at sub-epoch 1 FAILED (...)` or
    `median nan ... over 0 of 16`, together with the missing column. The job keeps running in that case (S2).
- **P0.** G0.R2 (`SAE_i6_P0.md:95`, "`lexlat_k2_stability` <= 0.05 from sub-epoch 11") reads the held path.
  This change does not touch it.

## Check 3: job ids

- **Fresh dry builds** (`sis console -s`, sae python, run from S):
  - probe: 87 jobs; arms: 113 jobs; P0: 164 jobs;
  - each list is identical to `analysis_out/p1_ladder_jobs_{probe,arms}_2026-09-25.tsv` or
    `analysis_out/g0g_p0_jobids_before_2026-09-25.txt`;
  - `EexT85vdfx25` is in the probe and arms builds, not in P0.
  - The arm training jobs are EexT85vdfx25 (r90), 9lD2HS2Mzgcl (r70) and 5CL3pQyyOvQt (r80). All three have
    `-p gpu_48gb`.
- **Written configs.** The r70, r80 and r90 `returnn.config` files (arms stage, and r90 from the probe stage) are
  byte-identical to `analysis_out/p1_ladder_rt_*_2026-09-25.returnn.config`.
- **P0 does not load the module.** After the P0 load, `rt_chunked_backward` is not in `sys.modules`. The running
  k2lat job binds `model.train_step` (`returnn.config:75-76`) and has no epilog.
- **Correction to the implementer's report.** The probe is not running. `work/.../ReturnnTrainingJob.EexT85vdfx25`
  does not exist, no ladder manager runs, and pid 1646677 is the P0 manager. No live process holds the old
  module.

## Check 4: Launch A holds with the edited test

- **The CUDA test** (`tests/test_rt_chunked_backward.py:499-524`):
  - The key-list assert (:510) drops a NaN held stability and expects it absent from the rt path.
  - The loop (:511-516) skips that key, so it can neither raise KeyError nor compare NaN with NaN.
  - The gradient asserts (:517-518) are therefore reached, and they run before the memory assert and the
    `[record]` prints.
- **What the CUDA test still does not cover.** The fixture rung (10000) equals the reference rung, so on CUDA the
  held read is the "no read" NaN. The test exercises neither a real stability read nor the case where a finite
  read is emitted.
- **Collection.** With Launch A's PYTHONPATH, 2 of 47 tests are selected (45 deselected). The Engine tests carry
  no gpu mark.
- **Pinned RETURNN.** The clone exists at commit 00171dfe, the same commit as `recipe/returnn`, and its
  `engine.py` is byte-identical.
- **The command is unchanged:**
  `sbatch -A hlt -p gpu_test_24gb --gres=gpu:1 -c 4 --mem=16G -t 0:30:00 -J p1_rt_parity -o /u/hwu/setups/librispeech-960/2026-09-24-unsupervised/log/p1_rt_parity.%j.out --wrap 'S=/u/hwu/setups/librispeech-960/2026-09-24-unsupervised; export PATH=/work/asr4/hwu/conda/envs/sae/bin:$PATH PYTHONPATH=/work/asr4/hwu/setups$S/i6_core/tools/git/CloneGitRepositoryJob.KQ3NuCaDE6QH/output/returnn:$S/recipe:$S/recipe/returnn:$S/sisyphus; cd $S/recipe/i6_experiments/users/wu/experiments/unsupervised_asr && nvidia-smi && python -c "import sys,torch,k2;sys.exit(not(torch.cuda.is_available() and k2.with_cuda))" && python -m pytest -p no:cacheprovider -m gpu -rA -s tests/test_rt_chunked_backward.py'`
  - Resources: gpu_test_24gb (RTX 3090, sm_86), never gpu_11gb or A100; 1 GPU, 4 CPU, 16 GB, 30 min against
    the 1 h cap. Output goes to S/log, not /var/tmp.
- **PASS read.** All of the following:
  - `collected 47 items / 45 deselected / 2 selected` and `2 passed`, with no SKIPPED or FAILED;
  - two `[record] cuda chunk {1,4}:` lines with max |dlogZ| <= 1e-9 and both gradient maxima <= 1e-7;
  - two `memory monitors` lines, with all three values > 0 in each.

## Check 5: CPU tests (sae python; PYTHONPATH = S/recipe, S/recipe/returnn, S/sisyphus; sae bin first on PATH)

- **`tests/test_rt_chunked_backward.py`:** 45 passed, 2 skipped (the gpu cases).
  - The 31 old `[record]` lines carry 117 deviation values, all 0.000e+00. Every term and total equals the held
    path's.
  - The Engine records match the implementer's report: for failed and empty reads, 14 `train_loss_lexlat_k2*` keys
    and no stability key; for a real read, the value is 0.0.
- **Whole suite, `-m "not gpu" tests`:** 591 passed, 11 skipped, 5 deselected, 12 xfailed. This matches the
  implementer's counts.

## P0 question: does the frozen held path run the stability read in `k2lat_20_ma3000` (jcKXbLMDk4hl)?

Yes. The config states `lexlat_k2_max_active` 3000 and on-set 8, and no reference rung.
- `log.run.1:353` logs `stability_reference_max_active 10000, stability_seqs 16, stability_chunk_seqs 1`.
- 10000 > 3000, so the read runs at training step 0 of every sub-epoch from 8 to 20, on the held path.
- On that path a NaN still stops the run (`stop_on_nonfinite_train_score = True`, config line 180).
- The job is currently in sub-epoch 3.

The log lines of a failed read, in order:
1. The read's own line: `lexlat_k2: the stability read at sub-epoch N FAILED (<Type>: <msg>); lexlat_k2_stability
   is nan for this sub-epoch`. For an empty read it is `lexlat_k2: stability at sub-epoch N: median nan nats per
   retained frame over 0 of 16 utterances (|log Z(3000) - log Z(10000)|)`.
2. Step 0's progress line, showing `lexlat_k2_stability` nan (log_verbosity is 5).
3. `Model seems broken, got inf or nan score.`
4. `Accumulated scores: {...}`, in which only `lexlat_k2_stability` is nan.
5. `Checking for inf/nan in model parameters...` and then `(No inf/nan in model parameters.)`.
6. `Running debug_inf_nan...` and its op trace. It re-runs the step with the cached NaN, so no second read line is
   printed. The first inf the trace flags may be an intentional -inf mask, so the trace is not the diagnosis.
7. `Exception: Inf/nan score in step 0.` The job goes to error, and the last checkpoint is epoch N-1.

The diagnosis rests on line 1 together with line 4. If the terminal traceback is anything else, for example an OOM
in the debug re-run, line 1 still decides. A `K2_CHECK` abort in a pool thread is not caught at all: it ends the
process with no Python traceback.
