# Review: P0 k2lat per-chunk k2 backward, live-job config change and restart (2026-09-25)

Verdict: PASS_WITH_FIXES. The change is exact and hash-neutral, and the restart procedure holds.
There are no code fixes. Two conditions must be met before and after the install (findings 1 and 2).
Nothing was run or touched in J, the job, or the managers.

J = `work/i6_core/returnn/training/ReturnnTrainingJob.jcKXbLMDk4hl` (Slurm 4359832_1, RUNNING on
gpu_32gb). P0 manager pid 1646677, started 12:54:29. Inputs reviewed:
`reports/impl_p0_k2lat_perchunk_2026-09-25.md`, `reports/impl_p0_k2lat_v100_probe_2026-09-25.md` (the
cross-worker check), `config/sae_i6_p0.py`, `analysis_out/k2lat_perchunk_*`,
`reverse_model/rt_chunked_backward.py` (commit 35f676b70), and `tests/test_rt_chunked_backward_k2lat.py`.

## Findings

1. SHOULD (launch condition), `analysis_out/k2lat_perchunk_install.sh:9` and `settings.py:449-450`.
   The install always runs chunk_seqs = 16. The epilog sets no chunk size, and J's config has no
   `lexlat_k2_chunk_seqs`. Sisyphus' worker env (CLEANUP_ENVIRONMENT, KEEP set without
   LEXLAT_K2_CHUNK_SEQS) drops the env override, and the live log line 353 shows chunk_seqs 16.
   The probe has two variants: A (chunk 16) and B (chunk 8, via `LEXLAT_K2_CHUNK_SEQS`, per the
   probe report, addendum). Only an A PASS licenses this install. If A fails and only B passes,
   running this install puts the configuration that failed (possibly an OOM at sub-epoch 8) on the
   V100. Chunk 8 would need a different change, and it must stay hash-neutral: a net-arg
   `lexlat_k2_chunk_seqs` would move the job hash. That change would need its own review.
2. SHOULD (disclosure and gate read), `rt_chunked_backward.py:220` against `SAE_i6_P0.md:97` and `:164`.
   On the per-chunk path, a failed or non-finite stability read is omitted from monitors. The held path,
   and therefore JUPITER, would stop the run on it (`stop_on_nonfinite_train_score`). G0.R2 reads
   "`lexlat_k2_stability` <= 0.05 from sub-epoch 11". If a sub-epoch has no stability column, a max
   taken over the columns that exist would pass a run whose reads failed. Two actions:
   - Record the omission under Deviations. The 3 new memory monitor columns go there too; they carry
     no loss.
   - State in G0.R2 that a missing `lexlat_k2_stability` for any sub-epoch from 11 on fails the clause.
     Because no k2lat read exists yet, this is not a gate rewrite.

No other finding.

## Checks and evidence

1. Config delta (PASS).
   - `analysis_out/k2lat_perchunk_returnn.config` = live `J/output/returnn.config` (sha dff329b6, equal
     to `.orig`) + exactly the 239-byte epilog (import `rt_chunked_backward.train_step as train_step`,
     overriding the held import at config line 75 because it comes later).
   - An independent dry build of `config/sae_i6_p0.py` reproduced the new file byte for byte, with the
     same 164 job ids after normalization. The ctrl_20/s1/rc/supervised configs are byte-identical.
   - `_per_chunk_k2_backward` is applied to k2lat only (`config/sae_i6_p0.py:36-63`). It asserts an
     empty epilog and an unchanged `python_epilog_hash`; `_sis_hash` reads the hash, not the text.
     This is the same mechanism as the P1 ladder.
   - Reads (`analysis/per.py`) use NET_ARGS and checkpoints, not the training config.
   - The epilog comment says "P1 rt arms". That is cosmetic.
2. Exactness at k2lat settings (PASS).
   - Re-run: `test_rt_chunked_backward.py` 45 passed, 2 skipped; `test_rt_chunked_backward_k2lat.py` 6
     passed; all deviations 0.0.
   - The new test uses the held leg as the reference and exercises:
     - max_active 3000 and reference 10000 (the stability read runs);
     - chunk 16 over 3 chunks (16/16/8 of 40 sequences, rows 3/17/33 dropped);
     - epochs 8-11 (lam 1/3, 2/3, 1, 1) at tau 2.0;
     - a no_grad forward.
   - Limits: the fixture is tiny, so pruning barely binds. Sub-epoch 7 is not a case, but at lam 0
     the held `train_step` never calls the leg (`model/train_step.py`). The module asserts the leg is
     active.
   - The ratio in `_InjectGrad` is exactly 1.0: RETURNN's scale is `lam` in float64
     (`lattice_float64=True`).
3. k2lat specifics (no delta issue).
   - The runtime is built in `SaeBlankfreeModelV1.__init__` (`blankfree_model.py:514-548`), so the
     per-step assert in `rt_chunked_backward.train_step` holds at sub-epochs 4-7.
   - The on-set, temperature, prior weight 1.0 and dev no_grad follow the held path.
   - `_abort_on_empty` has the same rule and call as the held path. On a single i6 job, its `os._exit(0)`
     would mark the run task finished without checkpoints. That behaviour predates this change.
   - Checkpoints are unaffected: the runtime is a plain object, and the agg EMA buffers are persistent
     and train-only.
   - `cleanup_old_models` has keep_best_n 0, so the new monitor columns cannot change which
     checkpoints survive.
   - The epilog under MultiProcDataset (6 procs) and DataLoader workers is exercised live by P1
     `EexT85vdfx25` (log line 577, 36 steps, no error).
4. Restart (PASS).
   - The install writes only `J/output/returnn.config` (temporary file, then mv), plus `.orig` if
     it differs. It refuses when `epoch.008.pt` exists.
   - RETURNN reads the config once, and subprocesses get the pickled dict.
   - Resume:
     - `get_epoch_model` picks the newest epoch with both `.pt` and `.opt.pt`;
     - the optimizer is loaded (patched with `weights_only=False`);
     - the model load under torch 2.7.1's default `weights_only=True` was tested read-only on
       `epoch.003.pt`: it loads, with keys epoch 3, step 171 = 57*3, and the agg EMA buffers present;
     - learning-rate multipliers are reapplied;
     - the per-epoch seed is merge(epoch, global step, random_seed), so the redone sub-epoch has the
       same data order.
   - A GPU run is not bitwise-deterministic anyway.
   - Resubmit: the task becomes INTERRUPTED_RESUMABLE, and the one earlier submit has
     completed_fraction None, so there is no retry cap. The in-memory rqmt plus `check_engine_limits`
     gives `-p gpu_32gb`, 72 h, 16 CPUs, 64 GB, 1 GPU.
   - Only the P0 manager holds J. The `job.save`/`info` rewrite is cosmetic.
   - The error marker is covered by the report's step 5. The watcher ignores `*.cancel_backup`.
5. Editing `config/sae_i6_p0.py` under the live manager (safe). The manager loaded it once at 12:54
   and does not reload it. Watchers re-import it read-only (the dry build proves the import works).
   `create_files` is finished, so the file on disk is what the resubmit reads.

No drift: no imported source is newer than 13:00 (`model/`, `training/`, `lm/`, the pinned RETURNN,
`i6_models`). The two commits since then touch only `reverse_model/` (genmarg, ladder,
rt_chunked_backward). The P0 rule "model/ training/ FROZEN until the trainings end" (SAE_i6_P0.md:26)
must hold through the restart.

## Restart sequence (all from S)
1. Only when probe A (chunk 16) PASSES G0.K2M.
2. For some N <= 6: `J/output/models/epoch.00N.opt.pt` exists, and `J/log.run.1` shows
   "Epoch N evaluation: dev:" followed by `ep N+1 train, step` lines.
   - For N = 7, do not wait for epoch-8 steps, because they run the held leg. Cancel once
     `epoch.007.opt.pt` exists; a missing eval is re-run on resume.
   - Check that `find -L recipe/returnn/returnn recipe/i6_experiments/users/wu/experiments/unsupervised_asr/{model,training,lm} -name '*.py' -newermt '2026-09-25 13:00'` is empty.
3. `bash analysis_out/k2lat_perchunk_install.sh` and expect `INSTALLED`.
4. `scancel 4359832`, then confirm `squeue -j 4359832` is empty.
5. If `J/error.run.1` exists: `mv J/error.run.1 J/error.run.1.cancel_backup`. Re-arm the P0 watcher if it
   reports ERROR.
6. No manager restart. Wait for a new line in `J/submit_log.run` with `-p gpu_32gb` and time 72.
7. In `J/log.run.1`, after the RETRY banner:
   - `Load model .../epoch.00N.pt`;
   - "Starting training at epoch N+1, global train step" equal to the `step` stored in
     `epoch.00N.pt`. Do not assume 57*N: sub-epoch 4 is partition 3, which has not been trained yet;
   - `k2 lexicon term: {... 'chunk_seqs': 16}`;
   - one `rt_chunked_backward: per-chunk k2 backward installed (chunk_seqs = 16)`;
   - at sub-epoch 8: the stability line and the `lexlat_k2_pre_peak_*` / `device_used` columns, no OOM,
     and no `J/work/lexlat_k2_ABORT.json`.
