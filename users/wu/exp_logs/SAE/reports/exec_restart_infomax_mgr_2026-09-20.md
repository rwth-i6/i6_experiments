# Restart sae_4a_infomax_pack manager -- 2026-09-20

Status: DONE

- Confirmed old pid 1230487 was `m -r config/sae_4a_infomax_pack.py` before killing (matches task).
- Killed 1230487 (SIGTERM); exited cleanly within a few seconds, no leftover ps match.
- Started new manager with `sis_env` activated (so `black` on PATH), plain (no -co/-cio), via
  `nohup tools/sisyphus/sis --log_level 20 m -r config/sae_4a_infomax_pack.py`.
  New pid: 1355841. New log: log/sae_4a_infomax_pack.manager.20260920T181107.log
  (log/sae_4a_infomax_pack.manager.log symlink repointed to it).
- No interactive clear-jobs prompt appeared; no manager helper (sis_managers.sh) covers this
  config (IN_SCOPE list only has sae_4a_budget_pack, not sae_4a_infomax_pack), so the manual
  nohup form from the sis skill was used.
- Census confirmed via console: JOBS 315, TARGETS 536 (matches expected).
- node_d training job PackedBlankfreeTrainJob.YtvrSez8z9Wf: manager log shows it recognized as
  `running` immediately on graph load (not resubmitted). Underlying Slurm job 1912407 confirmed
  still RUNNING (squeue) on jpbo-035-48. Only one live job dir for that hash; the
  `.YtvrSez8z9Wf.cleared.0001` sibling dir predates this restart (mtime 17:52:57, before the old
  manager even started at 17:53) -- not caused by this action. No second/duplicate node_d job dir.
- Other manager pid 2945974 (config/sae_4a_budget_pack.py) untouched, still running, unaffected.
- Read jobs: on graph load, 6 new CPU read jobs for ctrl_50 (entropy + usage_null, ep1/ep4/ep10)
  went runnable and were submitted to Slurm (job ids 1912635/1912636/1912638/1912640/1912641/1912642),
  now CONFIGURING/PENDING. ep25 entropy (TFMuFnpcuBF3) also submitted. Remaining 43 read jobs sit
  `waiting`, blocked on their per-epoch eval jobs for other arms/epochs, as expected.
- No errors, no queue_error, no anomaly observed in the new manager log.

Full report path: recipe/i6_experiments/users/wu/exp_logs/SAE/reports/exec_restart_infomax_mgr_2026-09-20.md
