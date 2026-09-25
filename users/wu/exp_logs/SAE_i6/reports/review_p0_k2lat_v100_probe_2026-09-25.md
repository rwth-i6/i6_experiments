# Review: P0 k2lat V100 memory probe (G0.K2M), variants A (chunk 16) and B (chunk 8), 2026-09-25

Role: code-reviewer, read-only. Nothing was submitted and nothing was modified in the setup, in J or in
the probe dirs. The only thing run was `read.py` on a scratch copy of a real rt log (details under item 4).

## Verdict: PASS (DONE_WITH_CONCERNS: SHOULD items only; none blocks the launch)

Both launches can go as written. Each probe trains exactly sub-epoch 8 with J's config plus the per-chunk
epilog and memory logging. Every schedule is at its sub-epoch-8 value, on J's sub-epoch-8 data, into its
own directory, in J's environment, on gpu_32gb. The gate numbers are logged, and read.py parses them
correctly from a real rt log. The SHOULD items concern how the result is read and applied, not the run itself.

Launch commands (verified):
- A: `sbatch -t 3:00:00 /u/hwu/setups/librispeech-960/2026-09-24-unsupervised/analysis/p0_k2lat_v100_probe/probe.sbatch`
- B: `PROBE_DIR=/work/asr4/hwu/sae_i6_probes/p0_k2lat_v100_ep8_2026-09-25_cs8 PROBE_CHUNK_SEQS=8 sbatch -t 3:00:00 -J sae_i6_p0_k2lat_v100_probe_cs8 --chdir=/work/asr4/hwu/sae_i6_probes/p0_k2lat_v100_ep8_2026-09-25_cs8 -o /work/asr4/hwu/sae_i6_probes/p0_k2lat_v100_ep8_2026-09-25_cs8/slurm-%j.out /u/hwu/setups/librispeech-960/2026-09-24-unsupervised/analysis/p0_k2lat_v100_probe/probe.sbatch`

## SHOULD findings (most important first)

1. Cross-worker: there is no reviewed route that runs J at chunk size 8. The live change
   (`reports/impl_p0_k2lat_perchunk_2026-09-25.md`, lines 120-121; `analysis_out/k2lat_perchunk_install.sh`,
   NEW_SHA 36e7032e...) installs chunk 16 only. The Sisyphus worker clears the environment
   (`settings.py:449-462`), so `$LEXLAT_K2_CHUNK_SEQS`, B's mechanism (`probe.sbatch:47`), cannot reach J.
   P1 sets its chunk size through the get_model kwarg `lexlat_k2_chunk_seqs` (`config/sae_i6_p1_ladder.py:53`).
   That kwarg resolves to the same `spec.chunk_seqs`, so B's result would transfer. When: A FAILS and B
   PASSES. Then a new implementer and review round is needed before sub-epoch 8 (about 19:15), which is
   the second round the hedge was meant to avoid. Prepare the chunk-8 install variant in parallel.
2. `read.py:112-117` looks for the stability print line (`lexlat_k2: stability at sub-epoch 8: median X ...
   over n of 16 utterances`, or `... FAILED`) only in `returnn.log`. That line is a plain `print`: it goes
   to stdout, which is `slurm-<id>.out`, never to `returnn.log`. Evidence: in EexT85vdfx25 it appears at
   `log.run.1:621` and nowhere in `work/returnn.log`. The same holds for the runtime line
   `rt_chunked_backward: per-chunk k2 backward installed (chunk_seqs = N)` (`log.run.1:577`), which is the
   run's own proof of the chunk size. read.py never prints either line. The gate stays decidable:
   the `lexlat_k2_stability` monitor is emitted only when the median is finite over at least 1 utterance
   (`rt_chunked_backward.py:220`, `lexlat_k2_train.py:640-645`), and a FAILED read shows up in the
   FAILED scan of the Slurm output. Still, when you read, grep `slurm-*.out` for `stability at sub-epoch 8`
   and `installed (chunk_seqs` to get the utterance count and to confirm A=16 and B=8 in the real run.
3. read.py never combines the two peaks into the registered single number and prints no verdict.
   It prints the per-step max(pre_peak, mem_usage) (`read.py:84-87`) and, separately, the epoch-end
   alloc peak (`read.py:103-106`). The gate value is the max of the two, compared with 28 GiB. Take that
   max by hand.
4. Seeding happens at each job's start (`probe.sbatch:52`, `seed.py:137-141`). If A and B start in
   different J epochs (J writes `epoch.004.pt` at about 17:05), the two variants differ in theta as well
   as in chunk size, and an A-versus-B comparison is confounded. cn-33 had 16 idle V100s at 16:35, so both
   should start at once on epoch 3. Before comparing, check that both `read.txt` files show the same
   `seed: J epoch N` and the same sha256.
5. Only on a relaunch into the same directory: `read.py:120` reads every `slurm-*.out`, and `seed.py --force`
   does not delete an old `lexlat_k2_ABORT.json`. Stale OOM lines or a stale marker from an earlier
   attempt would then count against the new run. Move the old files aside before any relaunch.

## Item-by-item evidence

1. Config. I ran `diff` of J's `output/returnn.config` (sha256 dff329b6...cdc, unchanged since 13:04)
   against both probe configs. The differences are exactly: the `model` line (-> PD/models/epoch),
   `torch_log_memory_usage = True` after line 183, and the 5 epilog lines. A and B differ only in the
   `model` path. The probe A config also equals the live change's new config
   (`analysis_out/k2lat_perchunk_returnn.config`) except for the model path and the memory line. So the
   probe runs the same code path J will run after the install. The memory line only reads allocator
   statistics (RETURNN `engine.py:328-339`, `:1614-1619`).
2. Sub-epoch 8. The epoch comes from the file name: `get_epoch_model` returns 7, and `_load_model`
   (`engine.py:985`) ignores the checkpoint's internal epoch 3 because the file-name epoch is not None.
   The start epoch is 8; `++num_epochs 8` makes the final epoch 8, and the package never reads num_epochs.
   Train loop `engine.py:268-271`. The dry check output shows lr(8) 1e-4 (ConstantLearningRate, the
   default control, `learning_rate_control.py:778`), temperature 2.0, anchor 0.0, prior weight 1.0,
   lam 1/3 (7..11: 0, 1/3, 2/3, 1, 1), and partition [3] of 0..3. The data order is the same as J's
   epoch 8: `_get_random_seed_for_epoch` (`datasets/basic.py:718-732`) depends only on the epoch,
   partition_epoch and the offset. The offset is 0 in both, since `env -i` drops
   RANDOM_SEED_OFFSET_ENV_VAR as Sisyphus does. No online_shuffle_batches is set. Placeholders: the lr
   control is constant, so scores never set an lr. `_check_missing_eval` (`engine.py:1541-1566`) only
   needs a train_/dev_ loss key per epoch. cleanup has keep_best_n=0, so scores choose nothing; it deletes
   only PD/models/epoch.007 after the dev pass. Nothing else reads them.
   Disclosed deviation: the torch RNG seed mixes in global_train_step (`engine.py:309-313`; 171 here,
   399 live), so the dropout masks differ. This has no effect on memory worth mentioning.
   The package code is the one J loaded: since 13:00 only `reverse_model/{ladder,genmarg}.py` changed,
   plus the new `rt_chunked_backward.py`; `reverse_model/__init__.py` imports nothing.
3. Isolation. Every write is inside PD: the config, the model, `learning_rates`, `returnn.log` and the
   ABORT marker (cwd). seed copies with `copyfile` to a tmp file, checks the sha256, then renames; it
   never links or moves. The `--force` wipe is asserted to stay inside PD. J's checkpoints and lr file
   are written atomically (tmp + rename), and seed retries on a cleanup race. The environment is
   `env -i` with settings.py's KEEP/SET, OMP/MKL=16 (i6_core `training.py:402-403`), and PATH with
   `/usr/local/bin/cf`. The RETURNN clone is the same; recipe/returnn is a symlink to it, both at
   00171dfe2 with the same dirty files dated 09-24. The resources match `sacct 4359832` (16 CPU, 64G,
   1 GPU, gpu_32gb). The Slurm default export is ALL (no SBATCH_EXPORT). If PROBE_DIR were not
   exported, B would refuse at the cwd guard; it would not run in A's dir.
4. Chunk size. J sets no `lexlat_k2_chunk_seqs`, so `blankfree_model.py:543-547` passes none and
   `resolve_chunk_seqs` reads `$LEXLAT_K2_CHUNK_SEQS` (`lexlat_k2.py:917-925`). The rnn call uses the same
   ENV array as the dry check. The dry check shows 16 for A and 8 for B. Monitors: the `lexlat_k2_pre_peak_*`,
   `device_used` and `stability` keys are marked as losses (`train_step.py:209-211`); `mem_usage:cuda` and
   the epoch-end `Memory usage (cuda)` line both come from `torch_log_memory_usage`. A step's line is
   printed after its backward and optimizer step (`engine.py:505-540`). The pre-peak at step n+1 covers
   step n's tail plus step n+1's forward and the stability read, so the max of the two per-step values
   covers the whole step. Real-log parse: I relabelled EexT85vdfx25's returnn.log from `ep 1` to `ep 8`
   and ran read.py on it. It found 26 steps and a gate peak of 27.256 GiB at step 6 (pre 27.256,
   mem_usage 26.9); `mem_usage:cuda 27.0GB` was parsed as 27.0 GiB (RETURNN GB means base 1024). The
   stability monitor read 0.083. The error scan gave 0 false hits on both returnn.log and the stdout log.
   `watch_memory` lines start with `MEMORY:` and do not interfere.
5. Live jobs. Both probes run outside Sisyphus's work dir, and the manager never sees them. They do not
   share a GPU with J; cn-33 was idle. `cf` handles concurrent caching itself, and J already holds its
   HDFs open. J's files are only read. The numba and kernel caches under /var/tmp are the ones J
   already uses. I found no lock or shared writable file.
   Note: if `k2lat_perchunk_install.sh` ran before a probe starts, `build_config.py:42` would assert and
   the probe would stop at once. That is safe, and the plan installs only after the probe passes.

Not verified: the GPU run itself (memory, speed, fit in 3 h). The gate premise that an earlier theta
gives larger lattices is part of the registered gate and was not re-examined here.
