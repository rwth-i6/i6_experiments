# exec_budget_resume2_2026-09-21

## Task
Confirm updater.py optimizer-load patch, clear node A's failed run task, and poll for resumption
of PackedBlankfreeTrainJob node A/B/C after the UnpicklingError fix.

## 1. Patch confirmed
`recipe/returnn/returnn/torch/updater.py:295-300` has `weights_only=False` (with TypeError fallback
for older torch), in place, as reported by the implementer.

## 2. Node A marker clear
Job dir: `work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.ks7CbtlvpcIL`
- Verified `error.run.1` traced to the same failure: `ArmFailure: 2 of 4 arm(s) failed:
  ctrl_100 (rc 1), odmprior_100 (rc 1)`, and both arm logs (`output/ctrl_100/log.run.1`,
  `output/odmprior_100/log.run.1`) contain `UnpicklingError`.
- Removed `error.run.1` and `submit_log.run` only. `output/`, `work/` checkpoints, and the 50-arm
  finished markers were not touched.
- Manager pid 2945974 was left alive and untouched (not restarted), per instruction; it picked up
  the cleared task on its own next loop.

## 3. Result: resubmitted but scheduler-blocked
Manager log shows: task went `interrupted_resumable` -> submitted -> `Submitted with job_id:
['1921418']` at 2026-09-21 01:42:16. However:

```
1921418_[1] PENDING ReqNodeNotAvail, Reserved for maintenance   (node A, new resubmit)
1921334_[1] PENDING ReqNodeNotAvail, Reserved for maintenance   (node B)
1921332_[1] PENDING ReqNodeNotAvail, Reserved for maintenance   (node C)
```

All three nodes are now PENDING on the same cluster-wide maintenance reservation
(`ReqNodeNotAvail, Reserved for maintenance`). None started in the observation window, so there is
no run log yet to confirm epoch.067 optimizer resume or arm-skip behavior for ctrl_100/odmprior_100.
This is a scheduler-side wait, not a job/config problem; nothing further to fix on my end.

## Not touched
Managers 4004154, 1355841, 347372 and their jobs were not touched. No code edited.

## Next action
Re-poll `squeue -j 1921418,1921334,1921332` once the maintenance reservation lifts; then tail
node A's new `log.run.1` and each 100-arm's `output/<arm>/log.run.1` to confirm the 50-arms are
skipped and ctrl_100/odmprior_100 load epoch.067 and take a training step.
