# Executor report: soft_pack forward jobs "problem state" check — 2026-09-21

## Verdict
Neither job is actually failed. Both are transient in-flight `ReturnnForwardJobV2` runs whose
`create_files` task completed normally and whose `run` task is currently RUNNING on SLURM. No
error/interrupted marker exists in either job dir. The watcher must have caught them mid-submit
(create_files done, run just queued/starting) and reported them as a "problem state" snapshot,
not an actual error/queue_error/interrupted state.

## work/i6_core/returnn/forward/ReturnnForwardJobV2.qDadVLNEdGN5
- Identity: `alias/sae/4a/eval/soft/sf_20/ep1/dev-clean/posteriors` (per log.create_files.1 Start Job line).
- Inputs: BlankfreeVadHdfJob.SAjz8y1cT06g/output/feats.dev-clean.shard0.hdf,
  ExtractSubmoduleCheckpointJob.KXsfZADIavow/output/model.pt.
- Markers: only `finished.create_files.1` present; no error/interrupted marker of any kind.
- create_files task: finished successfully (log.create_files.1: "Job finished successfully"),
  usage RSS 493MB, negligible time.
- run task: submitted to SLURM job 1929509_1 (submit_log.run), no log.run.* written yet (job just
  started). sacct: `1929509_1 RUNNING 00:00:35` on node jpbo-097-15, ExitCode 0:0, Reason None.
  squeue confirms state CF (configuring)->running.

## work/i6_core/returnn/forward/ReturnnForwardJobV2.GddOIzWGIGlV
- Identity: `alias/sae/4a/eval/soft/sf_20/ep1/dev-other/posteriors`.
- Same inputs (dev-other shard), same checkpoint ExtractSubmoduleCheckpointJob.KXsfZADIavow.
- Markers: only `finished.create_files.1`; no error/interrupted marker.
- create_files: finished successfully, usage RSS 493MB.
- run task: submitted to SLURM job 1929507_1. sacct: `1929507_1 RUNNING 00:00:36` on node
  jpbo-088-07, ExitCode 0:0, Reason None.

## Cause assessment
Same for both: not a failure at all — both are dev-clean/dev-other posterior-dump forward jobs for
the same soft/sf_20/ep1 evaluation, launched together, and were simply between task states (run
task freshly submitted) when the watcher snapshotted them. No code/config defect, no
tooling/environment defect, no node/hardware event. No black-missing / TypeError / login-node FS
signature seen in either log.

## Manager
`log/sae_4a_soft_pack.manager.20260921T092744Z.log` has only 1 line total (the WARNING-level
startup line from 11:27:44); at `--log_level 30` no INFO-level per-job lines are logged, so no
manager-side mention of these two jobs is expected. Manager pid 409858 confirmed alive (etime
05:54:28), untouched.

## PackedBlankfreeTrainJob.MXKoywbfon8O (Slurm 1926421)
Job dir: `work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.MXKoywbfon8O`
(NOT under i6_core/returnn/training — it is a custom speech_llm job class). `log.run.1` exists and
is growing; last lines show steady progress (RSS climbing 16GB->46GB over ~9 min, run time
0:09:11 as of last line). sacct: `1926421_1 RUNNING 00:17:10` on jpbo-081-34. Training has started
and is healthy; not blocked by the two forward jobs above (independent branches).

## Gating
Neither forward job gates PackedBlankfreeTrainJob — they are downstream eval/posterior dumps of a
different checkpoint (ExtractSubmoduleCheckpointJob.KXsfZADIavow), not inputs to the pack training.
No action taken: nothing to fix, manager and watcher left untouched.
