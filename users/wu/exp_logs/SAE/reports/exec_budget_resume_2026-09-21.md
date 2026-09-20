# Executor report: budget_pack resume check (2026-09-21)

## Node A — PackedBlankfreeTrainJob.ks7CbtlvpcIL (arms ctrl_50/odmprior_50/ctrl_100/odmprior_100)
- Original attempt Slurm 1907749_1: sacct State=TIMEOUT, Elapsed 11:31:18, End 2026-09-21T01:14:45 — the expected 11.5h wall kill, no error marker written for it (interrupted_resumable).
- Manager auto-resubmitted at same hash (submit_log.run shows 1907749 then 1921159) without my intervention.
- 1921159_1 (sacct: COMPLETED, 00:02:21): log.run.1 shows ctrl_50 and odmprior_50 SKIPPED (arm.finished present) — resume-skip logic (pack_jobs.py:143,195) verified working.
- ctrl_100 and odmprior_100 both launched (resuming from epoch.067) and FAILED rc 1 within ~40s: `UnpicklingError: Weights only load failed ... Unsupported global: functools.partial` in `Engine._load_optimizer` -> `Updater.load_optimizer` -> `torch.load(epoch.067.opt.pt, weights_only=True)` (PyTorch 2.6 default changed). This raised ArmFailure -> job wrote error.run.1 (real Python exception, not a SIGTERM).
- Root cause is NOT the time-limit kill by itself: resume DOES pick the newest checkpoint (epoch.067) as required (part b claim on checkpoint selection holds), but loading that optimizer checkpoint is broken by a torch/RETURNN version incompatibility — a code-level defect. Condition "both (a) skip and (b) resume-from-newest-checkpoint succeed" does NOT hold (load itself crashes), so per instructions I did NOT clear node A. No files touched.

## Node B — .reEI2Nd0S77A (Slurm 1908003_1) and Node C — .4QzmftNlbErt (Slurm 1908002_1)
- Both TIMEOUT at ~01:32-01:33 as expected; no error.run.* marker present on either — manager already auto-resubmitted at same hash BEFORE I looked (submit_log.run: B has 1908003 then 1921334; C has 1908002 then 1921332).
- squeue: 1921332_[1] and 1921334_[1] are PENDING (ReqNodeNotAvail, Reserved for maintenance) — not yet started, so their resume behavior is unobserved.
- Node B log.run.1 tail before timeout: arms bt_50, odmbt_50 already finished (rc 0). Node C: nosched_odmprior_50, nosched_ctrl_50 finished (rc 0). Both were mid-run on their 100-arms when killed.
- No error state exists on B/C, so nothing to clear; left untouched per instructions (only act on actual error states under the stated condition).

## Status: CANNOT_TELL / no action needed
Node A: real bug (torch.load weights_only=True vs optimizer state pickled with functools.partial) blocks the ctrl_100/odmprior_100 resume, independent of the time-limit kill; not cleared, not fixed (code fix required, out of scope). Nodes B/C: healthy auto-resubmit already pending in Slurm queue, no error, nothing done.

Manager 2945974 not restarted; not touched.
