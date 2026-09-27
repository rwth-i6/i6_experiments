# Review -- packed launch of the four S3b arms (PackedEmcTrainJob.SeZzGUScxq4x), 2026-09-15

Verdict: **PASS_WITH_CONCERNS**. The delta is what it says it is -- four byte-identical arms in one
4-GPU allocation, every read wired to the packed checkpoints -- and I found no path to a wrong
number. The three concerns are all failure/rework paths that the packing creates by coupling four
arms into one sisyphus job; none of them blocks the launch.

Scope: `recipe/2025-10-speech-llm` commits 5994bd9 / 3f5c805 (`src/speech_llm/sae/emc/pack_jobs.py`,
`test_pack_jobs.py`) and 69efa16 (`.../librispeech/configs/config_sae_4a_s3b_pack_v1.py`,
`src/speech_llm/sae/emc/test_pack_config.py`), plus the untracked `config/sae_4a_s3b_pack.py`.
`git show --stat` on the three commits: those five files and nothing else.

## What I verified myself (not read off the implementer reports)

**(3) Byte-identity, independently.** I re-ran `speech_llm.sae.emc.test_pack_config` (rc 0, "ALL
pack-config TESTS PASSED") AND wrote my own raw diff (no normalisation) of
`PackedEmcTrainJob.finalized_config(tag).write()` against the source stages'
`ReturnnTrainingJob.returnn_config.write()`, with the single-arm side built by
`config_sae_4a_s3b_rate_v1.build()` (ACTIVE_ARMS temporarily = lam1/lam10) and
`config_sae_4a_s3b_cons_v1.build()`. Result per arm: **exactly one differing line**, the `model =`
post-config line pointing at each job's own output dir --

* lam1 vs `ReturnnTrainingJob.WCh1fMyr88yu`, lam10 vs `axf11lrsap5i`,
  spec vs `fw5O1L5qL2HL`, spec_speed vs `be8qKc7aRj16` -- 5 unified-diff lines each, all the
  `model` line.
* `run_cmd` per arm == `ReturnnTrainingJob._get_run_cmd()` (same conda python, same
  `returnn/rnn.py`, its own config path); epoch key set 1..8 identical; `learning_rate_file` =
  `learning_rates`.
* `test_pack_jobs` re-run: rc 0, 4 tests (epoch keys, CPU smoke, failure/skip-on-finished, lam3
  config equivalence vs the on-disk `DF6blPpto23t`).

**(4) Hash census (current tree, `scripts/sae_4a_cons_census.py`).** s3 **98**, phase **1021**,
rate **91** with `lam3 = ReturnnTrainingJob.DF6blPpto23t`, cons **160** with
`spec = fw5O1L5qL2HL` / `spec_speed = be8qKc7aRj16` -- exactly the reported numbers and the reviewed
ids. A before/after pair is unnecessary here: `grep -rn pack_jobs\|s3b_pack` shows the two new
modules are imported by nothing except the new config and its test, so they cannot reach those four
graphs. The packed graph itself: **285 jobs**, one `PackedEmcTrainJob.SeZzGUScxq4x`, and exactly two
`ReturnnTrainingJob`s (the S0b inits `65NNK8Bwxdtd`, `HcXzd6M2eyVZ`); `DF6blPpto23t` is absent.

**(5) Dry-load.** `config/sae_4a_s3b_pack.py` loaded through sisyphus' own
`config_manager.load_configs` under the sis venv from the setup dir (no manager, no submit): rc 0,
285 jobs, 514 registered outputs (473 under the pack prefix + the upstream S0b / perturbed-dump
outputs registered by their own configs). On-disk state of that graph: everything upstream is
finished (incl. the perturbed L15 dump the speed arm needs); the only unfinished jobs are the pack
itself and its 252 reads, so the allocation is runnable immediately and waits on nothing.

**(1) Runtime.** Four `subprocess.Popen`, `cwd = <job>/work/<arm>` (sisyphus runs tasks in
`work/`, `task.py:177`), `CUDA_VISIBLE_DEVICES` = 0..3 by sorted arm name, `OMP/MKL_NUM_THREADS = 16`
-- the same value `ReturnnTrainingJob.run` sets from its own `rqmt["cpu"]` (`training.py:389-392`).
`env = os.environ.copy()` of the worker, i.e. the `worker_wrapper` env, identical to the single-arm
job's. The only relative paths in the written config are `learning_rates` and `["./returnn.log"]`,
both per-arm because each process has its own cwd; every other path is absolute. Output dirs
(`output/<arm>/models`) are created by sisyphus at setup (`job.py:277-280`).
*Time*: no clamp risk. `DF6blPpto23t` is **finished** (not running, as the dispatch has it): its
`usage.run.1` reads `used_time 0.654 h` for all 8 sub-epochs, `max rss 13.57 GB`, `max cpu 112 %`,
and the 8 checkpoints are on disk. The pack asks 6.6 h against the 11.5 h clamp
(`settings.py:118`), i.e. ~10x the measured need -- no arm can be killed before sub-epoch 8; the
only cost is that a 6.6 h exclusive request backfills worse than a ~1.5 h one.
*Sizing*: booster node = 288 CPUs / 878 GB / `gpu:gh200:4` (`sinfo`), so `--gres=gpu:4`,
`--cpus-per-task=64`, `--mem=256G` all fit, and 4 x 13.6 GB measured rss is a fifth of the request.
No NUMA/core binding per arm (three of four arms may run off their GPU's Grace socket) -- a
throughput effect only, with 10x wall-clock headroom and an SM-98 % compute-bound step.

**(2) Checkpoint / log layout.** `out_checkpoints[arm][ep] = output/<arm>/models/epoch.%03d.pt` as
`PtCheckpoint`, key set equal to the single-arm job's; `subepoch_reads`, the `ExtractSubmoduleCheckpointJob(checkpoint=...path)`
at sub-epochs 4 and 8, `DecodeStatsJob`/`GreedyPerJob` per split and the selection job consume them
unchanged (473 registered outputs, layout asserted by `test_pack_config.test_alias_layout`).
`path_available` mirrors `ReturnnTrainingJob`'s, so per-sub-epoch reads start while the pack runs.
Two extraction paths MOVE and any hand-written path must follow them: the live per-arm CV scores are
`<pack>/work/<arm>/learning_rates` (not `<job>/work/learning_rates`), and the per-arm RETURNN stdout
is `<pack>/output/<arm>/log.run.1` (the single-arm equivalent is `<job>/log.run.1`);
`read_efficiency_gate` takes the path as an argument, so it adapts.

## Findings

**F1 -- `pack_jobs.py:84-88`: the per-arm resume survives a SLURM timeout, not an operator's error
clear.** An arm's non-zero exit raises `ArmFailure` (`pack_jobs.py:251-254`) and the pack goes to
error state. The documented recovery ("on a restart an arm with that marker is SKIPPED") holds only
for sisyphus' automatic resubmit. The standard operator route -- the `sis m` prompt "Clear jobs in
error state?", `gs.CLEAR_ERROR`, `-c` -- runs `Manager.clear_states` -> `job._sis_move()`
(`manager.py:256-261`, `job.py:817-825`), which renames the WHOLE pack directory to
`....cleared.0001` and recreates it empty: all four arms lose their checkpoints and their
`arm.finished` markers and retrain from zero, and any read job in flight loses its input. The
recovery that preserves the work is the manual one (remove the `error.run.1` marker in the job dir,
as in the blank-stream trap). The single-arm baseline has the same property with a blast radius of
one arm; the packing makes it four. Nothing to change in the code -- this belongs in the launch
instructions.

**F2 -- `pack_jobs.py:557-560`: repairing one arm re-funds the other three and all 252 reads.**
`hash` covers the whole `arms` dict, so changing one arm's config (or dropping it from
`PACK_ARMS`, `config_sae_4a_s3b_pack_v1.py:82-87`) yields a new `PackedEmcTrainJob` id, a new job
dir and a full retrain of the three healthy arms; because every read's input path contains the pack
id, all 252 downstream jobs re-hash and re-run too. This is not hypothetical: `spec` and
`spec_speed` have never executed (no job dir, `fw5O1L5qL2HL` / `be8qKc7aRj16` are unrun), and
`spec_speed` is the only arm carrying the perturbed second stream and the per-key `batch_size`. The
cheap insurance is to accept a first-attempt crash as a per-arm restart (F1's manual route) rather
than a config repair, or to expect ~4 x 0.7 h of retraining if a repair is really needed.

**F3 -- `pack_jobs.py:542-555` + `config_sae_4a_s3b_pack_v1.py:271-282`: a failing sibling blocks
the healthy arms' selection.** `path_available` deliberately keeps `learning_rates` at the default
(available only when the JOB is finished), matching `ReturnnTrainingJob`; but the pack's job is
finished only when all four arms are. So if one arm fails permanently, the other three arms'
`learning_rates` output and their `UnsupervisedCheckpointSelectionJob` never become runnable even
though those arms completed and their file is already hard-linked into `output/<arm>/`. The G4a.3b
gate reads at sub-epoch 4 (greedy PER, derangement gap, phone rate) are NOT affected -- they hang
off the checkpoints, which `path_available` releases as soon as they exist.

## Non-findings I checked and cleared

* No train/eval contamination introduced: training and CV are the seed-0 1 % holdout of
  train-clean-100 exactly as in both source stages; all reads are dev-clean/dev-other; gold phones
  enter scoring and the gap's eligible-tag list only, unchanged.
* Constants: `TIME_RQMT_PER_ARM_HOURS = 6.0` is the literal both stages pass to `emc_training`;
  `GPUS_PER_NODE = 4` matches `sinfo` (`gpu:gh200:4`); cpu 16 / mem 64 / gpu_mem 96 per arm are read
  off `emc_training`'s signature by `inspect`, not retyped. `SHARED_NODE_TIME_FACTOR = 1.1` traces
  only to the coordinator's round-2 correction -- it enters no hash and no measured quantity, and at
  a measured 0.65 h against a 6.6 h request it cannot bind.
* `check_blacklisted_parameters(self, cfg)` called with an `EmcArmSpec` as `self` is fine: the
  method never touches `self` (`training.py:521-534`).
* Labels: the selection job's `arm` field and the aliases read `s3b_pack_<tag>` (not
  `s3b_rate_`/`s3b_cons_`) and the outputs land under `.../sae_4a_s3b_pack/<arm>/...`; the gate
  reads must be taken from that prefix, and the per-arm `phone_rate` file (not `rate_in_band`).
* Not exposed, no consumer: `out_plot_se` / `out_plot_lr`, and `info()` / `completed_fraction()`
  (the console shows no per-arm progress; the per-arm logs do).
* Pre-existing, not a delta: the 252 read jobs are GPU jobs and `settings.py:128-131` makes each of
  them hold a node exclusively too; the packing saves the training side only.
* Concurrency: this graph contains S0b, the perturbed dump and the shared `UnitsHdfJob`, so its
  manager must not run beside the S0b / S3 / S3b-R / S3b-C / pert-dump managers (the config says so;
  duplicate workers on one job are the known livelock).

Artifacts: `/tmp/claude-34349/.../scratchpad/{test_pack_config.log,test_pack_jobs.log,census.*.after,dryload.log}`
(session-local; the census output is reproducible with `scripts/sae_4a_cons_census.py {s3,phase,rate,cons}`).
