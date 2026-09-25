# Review: V100 routing of P0 trainings and the planned 12:53 migration (2026-09-25)

Verdict: APPROVE_WITH_CONDITIONS. The settings delta is correct and hash-neutral. A resumed run lands
on gpu_32gb. Two things in the launch plan need fixing: the trigger moment, and the record of the
gate and hardware changes. Written 12:26-12:40, read-only; nothing was submitted, killed or edited.

## Conditions (exact actions)
1. Trigger: do not scancel when `epoch.003.pt` appears. For each of DvVfxf1LrCBi and llSFybyKXkbL,
   wait for the line `start epoch 4 global train step` in `work/returnn.log`, then check that both
   `output/models/epoch.003.pt` and `epoch.003.opt.pt` exist.
   - Why. RETURNN's epoch end (pinned engine.py:651-658) runs in this order: `_save_model`,
     `_save_optimizer`, the dev `eval_model`, the learning_rates save, then the cleanup.
   - At epoch 2 of ctrl_20_s1: the .pt was written at 11:53:19.620, the .opt.pt at 11:53:19.776,
     and learning_rates at 11:55:39.
   - A kill inside the .pt/.opt.pt gap breaks the resume: `_load_optimizer(epoch=3)` raises
     FileNotFoundError (updater.py:299). That leaves error.run.1, which `-io` ignores, so the
     run stalls.
   - A kill during the dev eval loses epoch 3's dev scores in learning_rates.
2. Order: kill manager 1629803 first, then scancel. Otherwise the old manager, which still has the
   old settings, resubmits to gpu_48gb.
   - After the jobs leave squeue, check that neither job dir contains `error.run.1`.
   - If one does, delete only that file. Never use `-co`: it would also move the trie job aside.
3. Restart with `-io` (below it is confirmed correct).
   - Within about 2 minutes, check that the new `submit_log.run` line of both jobs carries
     `'sbatch_args': ['-p', 'gpu_32gb']`.
   - Check that `log.run.1` shows "Load optimizer ...epoch.003.opt.pt" and "start epoch 4 global
     train step 171" (57 steps per sub-epoch).
4. Record before any result. Write down the hardware split of each pair against its gate:
   - ctrl_20: all L40S.
   - ctrl_20_s1 and ctrl_20_rc: sub-epochs 1-3 on L40S, 4-20 on V100.
   - k2lat: all V100.
   Three gates are affected:
   - G0.R2 (SAE_i6_P0.md:95) requires "Step 1 identical to ctrl_20's", so it now compares V100
     with L40S. Define the tolerance now.
   - The G0.R1s seed band and the G0.RC paired deltas at ep4/10/20 now carry a hardware term.
   - line 260's own decision kept P0 on L40S for exactly this reason. The user's move supersedes
     it; record the move as an amendment.

## (a) settings delta: PASS
- `diff` of scratchpad `settings.py.orig` against settings.py gives only the 3 stated hunks.
- `is_train_run` is the old test. The new branch runs after the explicit-`-p` check and before
  the gpu_mem rules; everything else is byte-identical.
- ids_before/ids_after: 164 ids each, identical (checked with `diff`). check_engine_limits only
  touches rqmt, which is not hashed.
- Partitions the code can reach: gpu_32gb, gpu_24gb, gpu_48gb and the default. No config or
  package file sets `sbatch_args` (grep), so gpu_11gb and A100 are unreachable.
- Not verifiable: whether settings.py.orig equals what the manager loaded at 09:48. No copy exists
  from that time, and no report since then mentions editing settings.py.
- Side effect: any later resume of ctrl_20 (GiT88bxzoZbZ), for example after a node failure, also
  goes to V100.

## (b) state after scancel: interrupted_resumable
- scancel sends SIGTERM to every process of the job. Sisyphus installs handlers only for
  SIGUSR1/2 (tools.py:433-434), so the worker dies without writing an error file. Error files are
  written only on the Python exception paths, task.py:189-209 and worker.py:163-177.
- State sequence: RUNNING (squeue CG, or log mtime recent); then UNKNOWN and started; then 1
  submit ≤ MAX_SUBMIT_RETRIES gives INTERRUPTED_RESUMABLE (task.py:394-418). resume_jobs then
  resubmits it (manager.py:417-435).
- Residual race: RETURNN could exit first and the worker catch CalledProcessError(-15), which
  writes an error file. This is negligible because SIGTERM is fatal to the worker on delivery.
  Condition 2 covers it anyway.
- `-io` is right. It skips only the startup prompts "Clear jobs in error / not-resumable state?"
  (manager.py:552-565), which would hit EOF when backgrounded. The trie stays in error and nothing
  is cleared.
- rqmt on resubmit (engine.py:86-114):
  - History values are kept only for the current rqmt keys, so the old `-p gpu_48gb` in
    submit_log and usage is dropped.
  - update_engine_rqmt: time is 11.5 h against about 3.1 h used; mem is 64 GB against RSS 11.8 GB;
    sacct CANCELLED gives no OOM or TIMEOUT flag. Nothing doubles.
  - check_engine_limits then sets time 72 and `-p gpu_32gb`.
  - The request fits a gpu_32gb node (96 CPUs, 1.5 TB, 7 d). Free at 12:28: cn-33 has 96 CPUs and
    16 GPUs, cn-32 has 84 CPUs and 10 GPUs.

## (c) RETURNN resume across GPU types: what differs
- Device mapping: the model is loaded with `torch.load(map_location=device)` (engine.py:1793) and
  the optimizer with `map_location=self._device` (updater.py:299). Both are single-GPU cuda:0.
- Start: the start epoch comes from the last model (epoch 3), so the run starts at 4. The optimizer
  is loaded from epoch.003.opt.pt.
- Data order and seeds are the same as without the interruption:
  - Sub-epoch 4's data is a pure function of the epoch (laplace:.1000, partition_epoch 4,
    random_seed_offset 1000, all on the CPU side).
  - The torch seed is `merge_random_seeds([epoch, global_train_step, random_seed])`
    (engine.py:309-314), and the step count is restored.
- Differences for the deviation record:
  - The float64 lattice GEMM uses a different cuBLAS kernel and reduction order (native DGEMM on
    V100), and so do all fp32 kernels. The results differ in the last bits and then diverge.
  - The CUDA RNG, if the model draws any random numbers, can give different draws, because
    PyTorch sizes its Philox grids by SM count (80 on V100 vs 142 on L40S).
  - TF32: matmul TF32 is off by default and is forced off around every lattice GEMM
    (lattice.py:634-640). cudnn.allow_tf32 defaults to True, but grep finds no Conv in
    blankfree_model/emc_model, so TF32 has no effect.
  - No AMP.
  - Bit-reproducibility was already absent on one GPU (atomics in backward).

## (d) Memory on 32 GB: low risk, not measured at the laplace extremes
- None of the three logs has a peak memory: `torch_log_memory_usage` is off, there are no "Memory
  usage" lines, and sacct/sstat have no GPU TRES.
- The only laplace data point is one nvidia-smi reading of 34.3 GiB on L40S. The L40S bench
  reserved 36.8 GiB, so laplace was not above the bench there.
- The bench batches are already at 87.2-87.9k padded frames, against the 88k cap that also binds
  laplace batches.
- The untested part: B up to 128 (the bench went up to 122) and T_max up to the longest utterance
  (the bench went up to 817).
- Allocated 20.28 GiB against about 31.3 GiB usable leaves about 11 GiB. PyTorch frees its cache
  and retries before it raises OOM.
- If OOM happens: RETURNN raises, the task goes to error, and the run stalls; there is no crash
  loop. The epoch.003 checkpoint survives. The first V100 sub-epoch is the test.
- `expandable_segments:True`:
  - It is hash-neutral if added to `DEFAULT_ENVIRONMENT_SET`: environment is not hashed, but check
    it with the id-list diff. KEEP does not pass it through.
  - It is not advisable now: it is an extra delta and untested with k2. Keep it as the first remedy
    if an OOM occurs.

## (e) k2lat at sub-epoch 8: cannot tell
- On-device HLG: about 104 M arcs x about 24 B, roughly 2.5 GB. This sits on top of the ctrl_20
  path's 20.28 GiB.
- The unknown is intersect_dense_pruned's working set per 16-sequence chunk. It expands every
  out-arc before pruning, and there are up to 399,785 out-arcs per state (lexlat_k2.py:893-906).
- Contingency 1: set `LEXLAT_K2_CHUNK_SEQS` to 8 or 4 through DEFAULT_ENVIRONMENT_SET. It is
  hash-neutral and moves no number (lexlat_k2.py:906-909, bit-identity test). The cost is time.
- Contingency 2: a per-job `-p gpu_48gb` exception, resuming from epoch 7.
- Failure cost: an error stall at sub-epoch 8 and at most one partial sub-epoch, plus the time
  until someone notices. Watch sub-epoch 8.
- The alternative is to keep k2lat on gpu_48gb from the start. That removes both this unknown and
  the G0.R2 hardware confound. It is a fork against the user's decision, so it is the user's call.

## (f) run_k2_tests.sh: PASS
- Diff against G0.V's sbatch: header comments, job name, partition gpu_48gb→gpu_32gb, time
  2 h→1 h, output paths, the self-submit block, and the sm_70 guard (exit 2 through FATAL).
- Unchanged: account hlt, 8 CPUs, 48G, PYTHONPATH, the preflight checks and the pytest command.
  RET resolves to the same clone as `work/.../KQ3NuCaDE6QH`.
- Caveat: test_model_lattice.py, test_model_sil_run_collapse.py and test_model_blankfree.py have
  changed since G0.V, so the 506/11/12 counts may shift. Read the result as:
  - 0 failed;
  - no skip with reason gpu: or k2:;
  - T1.8 and T1.19 parity PASSED.

Not checked (time): the implementer's dry_check.py output, which I did not re-run; the
gpu_route_test.py update, which is still in progress; and the k2 kernels on sm_70, which is what
the test wrapper checks.
