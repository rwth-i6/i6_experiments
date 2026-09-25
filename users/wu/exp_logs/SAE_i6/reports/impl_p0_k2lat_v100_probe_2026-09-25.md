# impl: P0 k2lat V100 memory probe, sub-epoch 8 on the per-chunk k2 backward (2026-09-25)

Status: DONE_WITH_CONCERNS. The probe is built, and the full pre-flight (config build, seeding, CPU dry check) ran on the desktop and passed. Nothing was submitted and nothing was committed. Nothing was written into J (`work/i6_core/returnn/training/ReturnnTrainingJob.jcKXbLMDk4hl`), `config/` or `settings.py`, and the package was not edited.

## Files
Scripts, all new, in `S/analysis/p0_k2lat_v100_probe/`:
- `common.py` holds the paths: J, the probe directory PD, the pinned RETURNN, the sae python, target epoch 8 presented from epoch 7.
- `build_config.py` writes `PD/returnn.config`. The file is J's `output/returnn.config` byte for byte, with one line changed and two additions:
  - the `model` line now points to `PD/models/epoch` (changed line);
  - `torch_log_memory_usage = True` is inserted after `torch_dataloader_opts`, which is where ReturnnConfig writes a `post_config` key and where the ladder config `ReturnnTrainingJob.EexT85vdfx25` has it;
  - the per-chunk epilog is appended. Its text is copied verbatim from the tail of that ladder config and checked AST-equal to `rt_chunked_backward.EPILOG`.
  The script asserts that the diff against J is exactly these three edits and writes the diff to `PD/config_delta.diff`.
- `seed.py` seeds the probe from J. It is run at job start, so it takes the checkpoint that is latest when the job starts.
  - It picks the largest N with both `epoch.00N.pt` and `.opt.pt` in J's `output/models`. It refuses if N is 8 or higher.
  - It copies (never links) both files to `PD/models/epoch.007{,.opt}.pt` and checks the sha256 of each copy.
  - It writes `PD/learning_rates` through RETURNN's own `LearningRateControl.save`. Epochs 1..N are J's entries. Epochs N+1..7 get the list lr plus placeholder train and dev scores (`*_loss_probe_placeholder` = 0.0, marked in `:meta:`); the placeholders exist only so that `_check_missing_eval` passes. Epochs 8..20 carry the lr only. Every lr is asserted equal to the config's `learning_rates`.
  - It records source epoch, internal step and sha256 values in `PD/seed_record.json`. A copy of the seeded lr file is kept as `PD/learning_rates.seeded`.
- `dry_check.py` is the CPU check. It uses RETURNN's own code and exits 1 on any mismatch.
- `probe.sbatch` is the V100 job. It runs `build_config`, `seed --force`, the dry check (and aborts on failure), then `rnn.py PD/returnn.config ++num_epochs 8`, then `read.py > PD/read.txt`.
- `read.py` is the gate read. It is read-only; usage is `read.py [PD]`.

Probe directory: PD = `/work/asr4/hwu/sae_i6_probes/p0_k2lat_v100_ep8_2026-09-25`. It currently holds the config, the diff, a seeding from J epoch 3 (the job re-seeds itself at start), and the pre-flight outputs.

## Launch (executor)
```
sbatch /u/hwu/setups/librispeech-960/2026-09-24-unsupervised/analysis/p0_k2lat_v100_probe/probe.sbatch
```
The job header sets `-A hlt -p gpu_32gb --gres=gpu:1 --cpus-per-task=16 --mem=64G --ntasks-per-node=1 --time=02:00:00`, with `--chdir` set to PD and `-o PD/slurm-%j.out`.

The environment copies J's run task:
- CPU, memory and GPU as `sacct 4359832` reports them (cpu=16, mem=64G, gres/gpu=1).
- The same RETURNN (`CloneGitRepositoryJob.KQ3NuCaDE6QH/output/returnn/rnn.py`) and the same python (`/work/asr4/hwu/conda/envs/sae/bin/python`); no apptainer.
- The environment is built with `env -i`, as Sisyphus' worker does. It keeps settings.py's KEEP variables and sets its SET values: PATH is `sae/bin` first, then the system paths, including `/usr/local/bin/cf`. OMP_NUM_THREADS and MKL_NUM_THREADS are 16, as `ReturnnTrainingJob.run` sets them.
- The HDFs are cached through `cf`, because the config's `use_cache_manager=True` is unchanged.

Read: `/work/asr4/hwu/conda/envs/sae/bin/python S/analysis/p0_k2lat_v100_probe/read.py`. The job also writes the read to `PD/read.txt` at its end. The read prints:
- the gate peak, max over step lines of max(`lexlat_k2_pre_peak_allocated_gib`, `mem_usage:cuda`) in GiB, with the step where it occurs;
- the epoch-end `Memory usage` line, the one after `Epoch 8: Total train loss`, and its alloc peak;
- the max `lexlat_k2_device_used_gib`;
- the stability print line(s) and the `lexlat_k2_stability` monitor;
- any OOM, int32, FAILED or ABORT line in returnn.log and the Slurm output, and the ABORT marker file;
- the sec/step median (all steps, and excluding step 0);
- the number of steps completed, and dev-pass memory for information.
The parser was tested on a synthetic log; it has not been run on real probe output.

## Schedule values at sub-epoch 8 (from the dry check)
- RETURNN resumes from `PD/models/epoch.007`. The start epoch is 8 and the final epoch is 8 (through `++num_epochs 8`), so `ctx.epoch` is 8.
- The learning-rate control is `ConstantLearningRate`, and lr(8) = 1e-4.
- temperature = 2.0, anchor_weight = 0.0 and prior_weight = 1.0 (a scalar; there is no schedule).
- k2 is active at sub-epoch 8 with lam_lex = 1/3. Over sub-epochs 7..11 lam_lex runs 0, 1/3, 2/3, 1, 1.
- k2 spec: max_active 3000, search_beam 20, output_beam 8, min_active_states 30. The per-chunk runtime swap works, with chunk_seqs = 16 (the package default; J's config sets no chunk size).
- Train data: feats `partition_epoch` is 4, so epoch 8 takes partition index 3 of 0..3. The laplace seed term is (8-1)//4+1 = 2. This is the same as J's epoch 8.
- The seeded weights load strictly: no missing or unexpected keys.
- `train_step` is `rt_chunked_backward.train_step`, `torch_log_memory_usage` is True, and `_check_missing_eval` finds no gaps.
- Result: `DRY CHECK OK`, 21 of 21 checks OK. The full output is in `PD/dry_check.txt`.

## Deviations from the live sub-epoch 8, and what is unverified
1. The weights and Adam state come from J's latest epoch at job start (epoch 3 now), not epoch 7. The global step is N*57 rather than 399; no schedule reads the step. Lattice density, and therefore the k2 memory, depends on the weights, so the peak may differ from the live sub-epoch 8. This is the main caveat on reading the probe as the live number.
2. `++num_epochs 8` is a command-line override; the config file keeps J's 20. RETURNN reads num_epochs only for the final epoch and for model scanning.
3. Epochs N+1..7 in the lr file hold placeholder scores. The lr control is constant, so scores are unused.
4. Not verified: the GPU run itself, which covers memory, speed, and whether 57 steps plus the dev pass fit in 2 h. J needed about 40 min per sub-epoch without k2, and the per-chunk k2 cost on the V100 is unmeasured. If the time limit hits, the read reports the steps completed.
5. If the job starts after J has written `epoch.008.pt`, `seed.py` refuses and the job exits without training.

## Addendum: variant B, k2 chunk size 8 (coordinator request, same day)
B is A with two differences: the k2 chunk size is 8 instead of 16, and B has its own probe directory, `/work/asr4/hwu/sae_i6_probes/p0_k2lat_v100_ep8_2026-09-25_cs8`. Each variant seeds, keeps its model and writes its output only in its own directory.

What changed, all inside the probe scripts:
- `common.py`: PD is `$PROBE_DIR` if set, otherwise A's directory. `PROBE_CHUNK_SEQS` is read from the environment.
- `dry_check.py` gained one check: `model.lexlat_k2.spec.chunk_seqs` after the per-chunk swap must equal `$PROBE_CHUNK_SEQS`, or 16 when it is unset. That is the value `rt_chunked_backward` iterates over (`K.chunk_bounds(b, self.spec.chunk_seqs)`); the held path reads the same spec.
- `probe.sbatch`:
  - PD is `${PROBE_DIR:-A}`.
  - The job refuses to run if cwd is not PD, and if `PROBE_CHUNK_SEQS` is set without `PROBE_DIR`. The second guard was tested: it refuses.
  - When set, the job forwards `PROBE_DIR` and `PROBE_CHUNK_SEQS` through `env -i`, and exports `LEXLAT_K2_CHUNK_SEQS=$PROBE_CHUNK_SEQS` to RETURNN. This is the runtime knob `lexlat_k2.resolve_chunk_seqs` reads: it is not a config key and carries no hash.
  - `read.py` is called with `$PD`.
- A's behaviour is unchanged when the variables are unset. `env -i` also drops any stray `LEXLAT_K2_CHUNK_SEQS`, so A always runs the default of 16.

Launch commands (both can run at once; submit from any directory):
```
# A (chunk 16)
sbatch /u/hwu/setups/librispeech-960/2026-09-24-unsupervised/analysis/p0_k2lat_v100_probe/probe.sbatch
# B (chunk 8)
PROBE_DIR=/work/asr4/hwu/sae_i6_probes/p0_k2lat_v100_ep8_2026-09-25_cs8 PROBE_CHUNK_SEQS=8 sbatch -J sae_i6_p0_k2lat_v100_probe_cs8 --chdir=/work/asr4/hwu/sae_i6_probes/p0_k2lat_v100_ep8_2026-09-25_cs8 -o /work/asr4/hwu/sae_i6_probes/p0_k2lat_v100_ep8_2026-09-25_cs8/slurm-%j.out /u/hwu/setups/librispeech-960/2026-09-24-unsupervised/analysis/p0_k2lat_v100_probe/probe.sbatch
```
Read: `.../read.py` for A and `.../read.py <B dir>` for B. Each job also writes `<dir>/read.txt` at its end.

Dry checks, both rerun through the sbatch pre-flight on the desktop, both `DRY CHECK OK`:
- A: `per-chunk k2 backward installed (chunk_seqs = 16)`; `per-chunk runtime chunk_seqs = 16 OK`.
- B: `LEXLAT_K2_CHUNK_SEQS = '8'`; `per-chunk k2 backward installed (chunk_seqs = 8)`; `per-chunk runtime chunk_seqs = 8 OK`.
- Every other value is identical between the two: epoch 7 to 8, learning rate 1e-4, temperature 2.0, lam 1/3, prior weight 1.0, partition 3.
- Both are seeded from J epoch 3, with the same sha256 values (f9212055…, a2edc3e7…). The configs differ only in the `model` path.

### Steps and wall time against the 2 h limit
- **Steps:** sub-epoch 8 should have about 57 train steps. J trained exactly 57 steps in each of sub-epochs 1-3 (partitions 0-2). Sub-epoch 8 uses partition 3, which J has not trained yet, so 57 is not verified; batching is random at 88k frames, so expect 57 ± 1.
- **Bed:** at about 51 s/step (the coordinator's figure; J's sub-epoch 3 averaged 43 s/step), 57 steps take about 48 min.
- **Startup overhead:** J's first sub-epoch took 63.6 min against 40.6 min for the later ones, so a fresh process lost about 23 min to `cf` caching and warm-up. Budget 0-23 min, depending on the node cache.
- **Dev pass:** 285 sequences, about 2 min without k2 (from J's timestamps); a few minutes with the k2 forward.
- **Pre-flight:** under 1 min.
- **k2 leg:** unmeasured on the V100. The P1 design review estimated 12-32 s/step on the L40S (80-100 s/step total against a 68 s bed); that is an estimate, not a measurement.
- **Estimate:** A should take about 65-110 min, which fits 2 h but is tight at the top of the range. A finishes within 2 h only if the k2 leg averages under about 45 s/step with a cold node cache, or about 65 s/step with a warm one. B makes twice as many k2 calls per step, so it may be slower, and it is more likely to be cut off.
- **If the time limit hits:** `read.py` still reports the peak over the steps completed and how many steps completed. The epoch-end memory line would then be missing, and the peak would cover only part of the sub-epoch.
