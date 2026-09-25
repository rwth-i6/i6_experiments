# Implementation: per-chunk k2 backward for the live P0 k2lat job, prepared and not applied (2026-09-25)

Role: implementer. Setup dir S = `/u/hwu/setups/librispeech-960/2026-09-24-unsupervised`; package
P = `S/recipe/i6_experiments/users/wu/experiments/unsupervised_asr`; live job
J = `S/work/i6_core/returnn/training/ReturnnTrainingJob.jcKXbLMDk4hl` (Slurm 4359832, V100 on cn-32).
No manager or Slurm job was started, stopped or restarted. J was not modified. Nothing was committed.

## Status: DONE_WITH_CONCERNS

- The P0 graph now selects the per-chunk module for k2lat only. All 164 job ids are unchanged.
- `analysis_out/k2lat_perchunk_returnn.config` is J's config plus the epilog and nothing else: 5 added
  lines, 239 bytes. The file was written by the same `ReturnnConfig.write()` + black that create_files
  ran. This method reproduces J's current config byte for byte.
- The CPU exactness tests pass. The existing file did not cover k2lat's settings, so a new test file
  covers them. All deviations are 0.0.
- The concerns are operational, not correctness issues (see "Concerns"). They are: the manager rewrites
  `job.save` with the old config object when it resubmits; a restart costs about 1 h of startup plus
  queue wait; and a possible stray `error.run.1` after `scancel`.

## Files

| file | delta |
|---|---|
| `S/config/sae_i6_p0.py` | +22/-1. Adds `_per_chunk_k2_backward(arm)`, which sets `returnn_config.python_epilog = rt_chunked_backward.EPILOG`. It asserts that the epilog was empty and that `python_epilog_hash` is unchanged, the same mechanism and asserts as `config/sae_i6_p1_ladder.py:_arm_job`. It is applied to `arms.k2lat_20_ma3000(...)` only. The docstring notes the selection. |
| `P/tests/test_rt_chunked_backward_k2lat.py` | new, 6 tests (untracked in git; not committed) |
| `S/analysis_out/k2lat_perchunk_returnn.config` | the new config for J (12165 bytes, sha256 `36e7032e...a8`) |
| `S/analysis_out/k2lat_perchunk_returnn.config.orig` | a copy of J's current config (11926 bytes, sha256 `dff329b6...cdc`) |
| `S/analysis_out/k2lat_perchunk_returnn.config.diff` | `diff -u` of the two files |
| `S/analysis_out/k2lat_perchunk_install.sh` | the install script, NOT run (tested against a mock job dir) |
| `S/analysis_out/k2lat_perchunk_p0_jobids_after_2026-09-25.txt` | 164 P0 job ids after the change |
| `S/analysis_out/k2lat_perchunk_test_{existing,k2lat}_2026-09-25.log` | pytest logs |

## 1. Graph change and the job-id check

Method: a read-only dry build, run twice (before and after the edit), from S:
`PATH=/work/asr4/hwu/conda/envs/sae/bin:$PATH /work/asr4/hwu/conda/envs/sae/bin/python sisyphus/sis --log_level 40 console config/sae_i6_p0.py -s -c "exec(open('<dump.py>').read())"`.
The script lists `tk.sis_graph.jobs(update_graph=False)` and calls `returnn_config.write(<scratch>)` for
every ReturnnTrainingJob, which is the call `ReturnnTrainingJob.create_files` makes
(`recipe/i6_core/returnn/training.py:377`). No manager was started.

- Before the edit: 164 jobs, identical to `analysis_out/g0g_p0_jobids_before_2026-09-25.txt`. All six
  training configs written by the dry build are byte-identical to the `output/returnn.config` on disk:
  jcKXbLMDk4hl (k2lat), GiT88bxzoZbZ (ctrl_20), DvVfxf1LrCBi (ctrl_20_s1), llSFybyKXkbL (ctrl_20_rc),
  Y1vbqR6KeJSx and Ac2eioZbRX7d (supervised). This validates the write method, including black
  (`/work/asr4/hwu/conda/envs/sae/bin/black`, 26.3.1, the one create_files used).
- After the edit: 164 jobs, and the diff against "before" is empty. `jcKXbLMDk4hl` is present. The k2lat
  config has `python_epilog` length 229 and `python_epilog_hash` `''`, unchanged. ctrl_20, ctrl_20_s1,
  ctrl_20_rc and both supervised configs are byte-identical before and after.
- Only `config/sae_i6_p0.py` builds k2lat's training job. `sae_i6_p0_screen.py` builds ctrl_20 only,
  `sae_i6_p0_supinit.py` builds no k2lat training, and `sae_i6_g0g.py` builds no k2lat training.

## 2. The live job's config

`diff J/output/returnn.config analysis_out/k2lat_perchunk_returnn.config`:
```
238a239,243
>
> # P1 rt arms: the k2 lexicon leg with a per-chunk backward (exact; reverse_model/rt_chunked_backward.py)
> from i6_experiments.users.wu.experiments.unsupervised_asr.reverse_model.rt_chunked_backward import (
>     train_step as train_step,
> )
```
The first 11926 bytes are identical to J's file. The addition comes after `locals().update(**config)`, so
it rebinds `train_step`. The layout is the same as the rt_r90 probe's live config.

**Procedure used:** a fresh create_files, in effect. The config was built by the edited P0 graph in a
console and written with `ReturnnConfig.write`, the create_files call. The method is validated above.
**Load check:** exec'ing the old file binds `model.train_step.train_step`. The new file binds
`reverse_model.rt_chunked_backward.train_step`, both by exec and by `returnn.config.Config.load_file`,
with the pinned RETURNN. The new file's `num_epochs` is 20 and its `model` is J's own `output/models/epoch`.
This shows that the file loads. It does not show that the job uses the new path; the install log line
does (step 3).

## 3. Restart procedure (executor)

Facts the steps rely on:
- A sub-epoch is 57 steps (`learning_rates` `global_train_step_end` 57/114/171). Checkpoints so far:
  epoch.001 at 14:41, epoch.002 at 15:39, epoch.003 at 16:21:48, so about 42-58 min per sub-epoch.
  Projection: epoch.007 at about 19:10-19:30, then sub-epoch 8, the k2 on-set. `cleanup_old_models` keeps
  1/4/10/20 and the last one.
- End of sub-epoch N in RETURNN (`returnn/torch/engine.py:636-661`): `learning_rates` (train) ->
  `epoch.00N.pt` -> `epoch.00N.opt.pt` -> dev eval ("Epoch N evaluation: dev: ...") -> `learning_rates`
  (dev) -> cleanup of older checkpoints -> sub-epoch N+1.
- Resume: RETURNN resumes by itself. With `task = "train"` and no `load_epoch`,
  `get_epoch_model` (`returnn/engine/base.py:150-185`) takes the newest epoch that has BOTH `.pt` and
  `.opt.pt` and starts at N+1, with the model and optimizer state restored. If N's dev scores are
  missing, `_check_missing_eval` (`engine.py:1541`) runs N's dev eval first. RNG seeds are derived from
  (epoch, global step, random_seed) (`engine.py:309-314`, `:1238-1240`), so the redone sub-epoch sees
  the same data order and seeds.
- RETURNN reads the config file only at startup. Subprocesses get the parsed dict pickled
  (`returnn/config.py:38-62`), and i6_core's `run()` only calls `rnn.sh` on the file. So a file installed
  while the job runs has no effect until the next start.
- The manager does NOT need a restart. The run task is resumable (`Task("run", resume="run")`). Once the
  Slurm job is gone and `usage.run.1` is older than about 100 s, the task state becomes
  `interrupted_resumable` (`sisyphus/task.py:385-418`). The retry cap does not apply, because the only
  earlier submit has `completed_fraction` None. The P0 manager (pid 1646677) then resubmits with its
  in-memory rqmt: `-p gpu_32gb`, 72 h, the same `rnn.sh`, reading `output/returnn.config` from disk.
  create_files is finished and never re-runs, so the installed file stays.

Steps:
1. Precondition: the V100 memory probe has PASSED. Nothing is applied before that.
2. Wait until, for some N <= 7, `J/output/models/epoch.00N.opt.pt` exists and `J/log.run.1` shows
   "Epoch N evaluation: dev:" followed by the first `ep N+1 train, step` lines. Cutting early in
   sub-epoch N+1 loses the least work. Use the first such boundary after the probe passes.
3. Install: `bash S/analysis_out/k2lat_perchunk_install.sh`. The script checks that the live file is the
   original (sha `dff329b6...`) and that `epoch.008.pt` does not exist. It installs atomically (cp to a
   temporary file, then mv), checks the new sha `36e7032e...`, prints the 5-line diff and `INSTALLED`.
   Installing before the cancel is inert for the running process (see above). It also removes the race
   with the manager's resubmit, which can come about 2 min after the job leaves the queue.
4. `scancel 4359832`. Check with `squeue -j 4359832` that it is gone.
5. If `J/error.run.1` exists after the cancel (a race in which the worker records RETURNN's SIGTERM exit),
   the manager will not resubmit. Rename it aside (`mv J/error.run.1 J/error.run.1.cancel_backup`); the
   task then reads `interrupted_resumable`. The P0 watcher (`sis_watch.sh`) may report `STATUS=ERROR` on
   that marker and need re-arming.
6. The manager resubmits within a few minutes (new line in `J/submit_log.run`, `-p gpu_32gb`). Do not
   start a second manager.
7. Checks in `J/log.run.1`, after the "RETRY OR CONTINUE TASK" banner:
   - `Load model .../epoch.00N.pt` and `Starting training at epoch N+1, global train step 57*N`
     (not epoch 1);
   - `k2 lexicon term: {... 'max_active': 3000 ... 'stability_reference_max_active': 10000 ... 'chunk_seqs': 16}`;
   - exactly one `rt_chunked_backward: per-chunk k2 backward installed (chunk_seqs = 16)`. It is printed
     at the first `train_step` call: step 0 of N+1, or during the re-run dev eval if one runs.
   - At sub-epoch 8: `lexlat_k2: stability at sub-epoch 8: median ...` or `... FAILED (...)` (now
     non-fatal); per-step `lexlat_k2_pre_peak_allocated_gib` / `lexlat_k2_pre_peak_reserved_gib` /
     `lexlat_k2_device_used_gib`; no CUDA OOM; no `J/work/lexlat_k2_ABORT.json`.
   - Measured on the first launch, startup to the first step took about 56 min: job start 13:05,
     "Starting training" 13:38, first step about 14:01. Add the gpu_32gb queue wait.

If the timing is off:
- **Cut in the middle of sub-epochs 4-7:** the partial sub-epoch is discarded. RETURNN redoes it from
  `epoch.00N` with the same seeds. The cost is time only, and nothing is written for the partial
  sub-epoch.
- **Cut between `.pt` and `.opt.pt` (under a second):** the resume starts from N-1, whose files still
  exist until after N's eval, and redoes N.
- **Cut during N's dev eval:** the eval is re-run on resume.
- **Sub-epoch 8 already started on the held path:** if the cut or a crash comes before `epoch.008.pt`,
  the resume from epoch.007 redoes all of sub-epoch 8 on the per-chunk path, and no held-path result
  persists. An OOM on the held path (expected at 35-38 GiB against 32 GB) makes RETURNN exit non-zero.
  The worker writes `error.run.1` and the manager does NOT resubmit. Recovery: step 3, then step 5,
  then steps 6-7. A NaN stability read on the held path would also end the run (see commit 700c81dd0).
  The per-chunk path omits a NaN read.
- **`epoch.008.pt` written on the held path:** the install script refuses. Held and per-chunk are the
  same computation (CPU deviation 0.0, CUDA 1.9e-9), so a later switch is scientifically harmless. It
  would, however, be a mixed-runtime record, and whether to proceed is the orchestrator's decision.

## 4. CPU exactness tests

Command (P dir, sae python, pinned RETURNN first, `CUDA_VISIBLE_DEVICES=`):
`python -m pytest -p no:cacheprovider -rA -s tests/test_rt_chunked_backward.py [tests/test_rt_chunked_backward_k2lat.py]`.

- `test_rt_chunked_backward.py`: 45 passed, 2 skipped (gpu). All 34 `[record]` deviations are 0.000e+00.
- **Coverage of k2lat's settings: NOT covered.** The fixture runs at max_active 10000 with beams 1000.
  Its reference rung is 10000, so the stability read returns nan and never runs. The engine tests use
  reference 20000 against rung 10000. On-set is 1. Chunk 16 is tested, but on a 5-sequence batch, so it
  is a single chunk. Rung 3000 and beams 20/8/30 never occur.
- New file `test_rt_chunked_backward_k2lat.py`, at k2lat's spec as J prints it: rung 3000, search beam 20,
  output beam 8, min active 30, on-set 8, ramp 3, full lam 1, reference 10000, chunk 16. It uses a
  40-sequence batch (chunks 16/16/8, one dropped row per chunk) at tau 2.0 and sub-epochs 8/9/10/11
  (lam 1/3, 2/3, 1, 1). The held leg is forbidden inside the per-chunk runtime. It also includes the
  no_grad pass and a check that the unstated spec entries are the package defaults. Result: **6 passed**.
  At every sub-epoch, max |dlogZ| = max |dmonitor| = max |dgrad log_q| = max |dgrad param| = 0.000e+00,
  and the term is bit-equal (0.0860062307299628). The stability read runs and is emitted equal in both
  paths (0.0). Gradient magnitudes are 1.7e-4 to 5.1e-4 (log_q) and 4.4e-3 to 1.3e-2 (param), far above
  the 1e-7 tolerance.
- Both files together: 51 passed, 2 skipped.
- Limits: on the 4-phone fixture, the k2lat beams move log Z by 1.5e-3 from the wide point, but rung
  3000 against 10000 prunes nothing (stability 0.0). The pruned-rung regime of the real HLG is exercised
  only by the GPU memory probe. The full blank-free `train_step` test was not repeated at these settings.

## Concerns

1. When the manager resubmits, it rewrites `J/job.save` and `J/info` from its in-memory job, which has
   the epilog-less config (`sisyphus/manager.py:425-429`, `job.py:311-313`). `output/returnn.config`
   is untouched and is what runs. The effect is cosmetic unless someone forces create_files to re-run,
   which would overwrite the installed file. A later P0 manager restart loads the edited
   `sae_i6_p0.py`, and its in-memory config then matches the file.
2. The restart costs about 1 h of startup plus queue wait on gpu_32gb, on top of the discarded part of
   the sub-epoch that is running.
3. The epilog's comment line reads "P1 rt arms". It is the module's `EPILOG` constant, used verbatim so
   that P0 and P1 select the path identically. `rt_chunked_backward.py`'s docstring still names only the
   ladder config as a user. Changing either is outside my files. Proposal for the next edit of that
   module: generalise the wording. That would change neither ids nor numbers.
4. P0 k2lat does not set `torch_log_memory_usage` (the P1 probe arm does). The module's own
   `lexlat_k2_pre_peak_*` and `lexlat_k2_device_used_gib` monitors are logged regardless. Setting it was
   not requested and was not added.

## Semantics of the switch for k2lat (hash-neutral, value-neutral)

These are the module's documented differences from the held path; none moves a score or gradient.
- A non-finite stability read is omitted, not emitted. Under `stop_on_nonfinite_train_score = True`, the
  held path would end the arm on such a read.
- Three memory monitors are added from sub-epoch 8 on. They are `as_error`, so they are not in the loss,
  and they appear as new `learning_rates` columns.
- `lexlat_k2_sec` and `lexlat_k2_peak_reserved_gib` now include the per-chunk backwards.
- The phase file should record the switch sub-epoch and restart time as a disclosed runtime change of P0.
