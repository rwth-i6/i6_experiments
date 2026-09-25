# Review: V100 vs L40S ctrl_20 step benchmark (2026-09-25)

Verdict: APPROVE_WITH_CONCERNS. Both launches can be submitted as written. The one concern is in the
`--compare` arithmetic: the cross-shape projection for the 44,000/64 shape, shape (b). It can be
recomputed afterwards from fields the JSONs already record, so no rerun is needed.

Reviewed: `analysis/v100_bench/bench.py` (503 lines) and `run.sh` (47 lines), the implementer report
`reports/impl_v100_bench_2026-09-25.md`, the ctrl_20 `output/returnn.config`, and the pinned RETURNN
(`CloneGitRepositoryJob.KQ3NuCaDE6QH`, 00171dfe plus local diffs to engine.py, updater.py and
task_system.py; none of the diffs touches the paths below). Also read: the model code
(`train_step.py`, `lattice.py`, `emc_model.py`, `blankfree_model.py`; unchanged since 2026-09-24
18:12, before ctrl_20 started at 23:53; git status is clean), the ctrl_20 `log.run.1`, and Slurm
sinfo/scontrol/sacctmgr.

## Findings

1. MEDIUM. `bench.py:461-463` (the compare) and `bench.py:342` (summarize). The shape (b)
   projection is `3237 s x (V100 ms per real frame at (b)) / (L40S ms per real frame at (a))`.
   Real frames are the wrong normaliser here.
   - The DP runs over the padded B x T_max. The ctrl_20 log shows the L40S at 7.87e-4 s per padded
     frame at B = 128, constant from T_max = 257 to 687 (steps 32, 16 and 6 of sub-epoch 1).
   - Random batches carry different padding at the two shapes. By the midpoints of the
     implementer's ranges, real/padded is about 0.706 at (a) and about 0.757 at (b). The laplace
     training runs at 7.2-7.9 % padding at both shapes (the "Epoch N: Trained" lines).
   - Result: the (b) projection comes out about 7 % too optimistic, making the V100 look faster at
     (b) than it would be.
   - Correct form: laplace training at (b) is twice the steps at the same T_max with half the B. The
     per-padded-frame ratio matches this in the compute-bound limit and stays within about 1 % in
     the launch-bound limit. Per-real-frame is off by 7-8 % in both.
   - Fix: use `padded_frames_total` (already recorded) in place of real frames. Equivalently, take
     3237 x [L40S (b) / L40S (a) per padded frame] x the V100/L40S same-shape ratio at (b).
   - The same flaw means the absolute ms per 1k real frames must never be multiplied by training
     frame counts: random batches carry about 29 % padding, the training about 7.5 %.

## The six checks

1. No write, delete or lock in any live job dir or shared input. PASS, checked against the source.
   - The bench calls `init_config`, `init_log`, `init_backend_engine`, `init_engine` and
     `Engine.init_train_from_config` with no data. It never calls `init()`, `init_data()`,
     `train()` or `train_epoch()`.
   - `_save_model`, `_save_optimizer`, `learning_rate_control.save()` and `cleanup_old_models` are
     reachable only from the end of `train_epoch` (engine.py:643-661) and from the
     `cleanup_old_models` task. Neither runs.
   - `model`, `learning_rate_file` and `log` are overridden into `<out>/<tag>/`. `LearningRateControl`
     only reads its file at init, and only if the file exists.
   - `models/` holds symlinks to `<out>/init`. Any `delete_model` would unlink those symlinks,
     never the targets.
   - No DataLoader, dev or eval dataset is built (the config has no `eval_datasets`), so there is
     no cache-manager copy. The bench's own MetaDataset sets `use_cache_manager=False`.
   - The HDFs are opened with h5py "r", which takes an HDF5 shared lock and does not conflict with
     other readers. The model code writes no files on this path. No tensorboard or profiler runs.
   - Snapshot sha256: a374c191... and b8f817d8... in both init/ dirs, identical to the live
     `epoch.011.*` (verified 11:03).
2. The timed computation is the real ctrl_20 step. PASS.
   - The same `train_step`, reached through `Engine._run_step`. It runs l_tau on the float64
     lattice (asserted), agg, and rate with its central-difference passes (`rate_dp_calls 2`).
   - Then `total_loss`, the per-loss `.item()` reads, backward, and `Updater.step`: clip at 5.0 and
     Adam with `emc_param_groups`, both loaded from epoch 11.
   - `set_epoch(12)` feeds `ctx.epoch` = 12: tau 2.0, anchor 0.0, prior weight 1.0, lr 1e-4. The
     model reads no epoch at construction.
   - The step is bracketed by cuda synchronize. Batch building is outside the timed region. The
     host-to-device copy is inside it, as it is in RETURNN's own step time.
3. Batches. PASS.
   - `init_seq_order(seq_list=...)` keeps the given order in both MetaDataset (meta.py:296-298)
     and CachedDataset (cached.py:90-92). No laplace order and no partition is applied.
   - Batching goes through RETURNN's `BatchingIterDataPipe` with the padded rule.
   - The seed-0 permutation over the sorted `train.segments` gives the same batches on both GPUs.
     `--compare` checks this through the seq_tags.
   - The 2 warm-up batches are untimed.
4. OOM path. PASS.
   - An OOM exits with code 3. The driver retries shape (a) once with expandable_segments in a fresh
     process and its own run_dir, keeps both JSONs, then runs (b).
   - Peak memory stats are reset after the model and optimizer load and before the warm-ups.
     `max_memory_allocated` and `max_memory_reserved` are recorded both at OOM and at the end.
5. Slurm. PASS.
   - gpu_32gb is cn-[32-33] with gres V100:16. gpu_48gb is cn-[506-509] with L40S. Both allow
     account hlt, with MaxTime 7 days.
   - torch 2.7.1+cu126 includes sm_70.
   - The out_dir guard and init/ check are sound, and the name guard (`--expect-gpu`) is in place.
   - 2 h is enough: the L40S needs about 68 s per step at (a) (7.8e-4 s x 87.5k padded), so about
     30 min in total. The V100 fits even at 1.5 times the L40S time, including the retry.
6. Projection. At (a) the batches are identical, so per-real-frame and per-step reduce to the same
   quantity, 3237 x (sum of V100 step times) / (sum of L40S step times).
   - That ratio carries over to the laplace training because cost scales with padded B x T_max.
   - It is conservative if the V100 is partly launch-bound: random batches have B of about 115,
     against about 128 in the training.
   - For (b), see finding 1.

## Notes (no change requested)
- 3237 s is train time only. The 57 min wall clock also holds about 3 min of dev evaluation and
  checkpointing per sub-epoch (epoch.001 at 00:52 to epoch.004 at 03:43), which the bench does not
  measure. Compare the projection with 54 min, not 57.
- Running (b) as a training needs accum_grad 2. That is not the same update: grads are summed over
  two half-batch means against a clip of 5.0, and the agg count-EMA updates twice. That launch
  needs its own review.
- The L40S job may share a node with the live runs. It gets its own GPU and cgroup, so this is
  harmless.
