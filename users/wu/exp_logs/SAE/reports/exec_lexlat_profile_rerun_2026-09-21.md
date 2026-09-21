# Executor report: lexlat_profile rerun 2026-09-21

Job dir: work/speech_llm/sae/emc/lexlat_profile_jobs/LexlatStepProfileJob.ItFHebFGK6nb

## Actions taken
1. Moved stale output/* into output_failed1/ (per_frame.json, profile.json,
   profile_top_kernels.txt, returnn.config, summary.txt, trace.json.gz). output/ kept empty.
2. Deleted error.run.1 and submit_log.run (both confirmed removed, ls shows no error/submit_log
   files left).
3. Started manager: `sis m -io -r config/sae_4a_lexlat_profile.py`, pid 564094, log
   log/sae_4a_lexlat_profile.manager.20260921T094122Z.log. Manager exited after ~30s.
4. Manager log showed: `ERROR: interrupted_not_resumable: Job<... LexlatStepProfileJob.ItFHebFGK6nb>`
   -- no new submit_log.run was created, no SLURM job submitted.
5. Confirmed via graph console status: job state is `interrupted_not_resumable` (not `error`),
   so `-io` (which only ignores `error`-state jobs once) does not apply/clear it; `-cio` is
   required per the sis skill's Retry section for this state.
6. Attempted `sis m -cio -r config/sae_4a_lexlat_profile.py` -- the auto-mode classifier denied
   this Bash call ("Interfere With Workloads"), matching the known ops trap
   (manager-restart-blocked-on-error): clear/-cio manager restarts are classifier-blocked for
   this agent and need a manual `!`-prefixed command from the user/orchestrator.

## Current state
- No manager is running (pid 564094 has exited cleanly).
- Job dir is left with error.run.1/submit_log.run removed, output_failed1/ backup present,
  log.run.1 (stale, from the original failed run) still present, output/ empty.
- No new SLURM job was submitted for this job.

## Needed to unblock
Run `sis m -cio -r config/sae_4a_lexlat_profile.py` (nohup, same log-naming convention) from a
context that is not classifier-restricted (user or orchestrator's own Bash), then poll for a new
submit_log.run and report the SLURM id/state via `scontrol show job <id>`.

## Follow-up: unblocked (coordinator moved log.run.1/usage.run.1 into output_failed1/)
- Confirmed job dir clean (only engine/info/input/job.save/output/output_failed1/work; no
  log.run.1, usage.run.1, error.run.1, submit_log.run left in the job dir itself).
- Started fresh manager: `sis m -io -r config/sae_4a_lexlat_profile.py`, pid 628195, log
  log/sae_4a_lexlat_profile.manager.20260921T094432Z.log.
- submit_log.run reappeared after 30s: rqmt {cpu:16, mem:64.0, time:2.0, gpu:1, gpu_mem:96},
  slurm job id 1926511.
- `scontrol show job 1926511`: JobState=PENDING, Reason=ReqNodeNotAvail,_Reserved_for_maintenance
  (maintenance reservation, expected today). StartTime=Unknown (not yet scheduled by SLURM).
- Manager log tail after submission: no errors, only the routine WARNING about the absolute path
  inside vad_port.py (pre-existing, harmless).
- Manager pid 628195 is still alive (not polled further per brief: do not wait for the job to run).
