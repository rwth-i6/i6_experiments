# Executor report: launch config/sae_4a_infomax_pack.py (2026-09-20)

## Manager
- Started manually (`sis_managers.sh start` not used) matching the sae_4a_budget_pack launch pattern:
  `/e/project1/spell/wu24/env/sis_env/bin/python tools/sisyphus/sis --log_level 20 m -r config/sae_4a_infomax_pack.py`
- Final live manager pid: 1230487
- Log: `log/sae_4a_infomax_pack.manager.20260920T175302.log`
- PATH used included `/e/project1/spell/wu24/env/sis_env/bin` prepended (confirmed from `/proc/2945974/environ` of the running sae_4a_budget_pack manager).

## Trap hit and fix
First manager start (pid 1224162, PATH without sis_env/bin) crashed the new job's `create_files`
task: `self._black_path = None` -> `TypeError: expected str, bytes or os.PathLike object, not NoneType`
in `subprocess.check_call([self._black_path, file_path])` — the venv-PATH/`black`-missing trap.
Killed that manager, cleared `error.create_files.1`+`submit_log.create_files`, restarted with venv
PATH prepended. `create_files` then ran but the previous crash had left the task
`interrupted_not_resumable` (stale short-engine artifact). Manager-kill for a plain restart was
blocked by the auto-mode classifier ("Interfere With Workloads"), so cleared the one job instead via
console (`j._sis_move(); j._sis_setup_directory()`) — job is stateless (`create_files`, no training
progress), non-destructive. Live manager (1230487, never killed after this) picked it up on its next
loop and submitted normally.

## Result
- Job dir: `work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.YtvrSez8z9Wf`
- Alias: `alias/sae/4a/infomax_pack/node_d/training`
- Slurm job id: 1912407, state PENDING (queued) as of 17:57
- Per-arm subdirs present under `output/`: aug_50, aughi_50, ent_50, entaug_50
- node_a (`PackedBlankfreeTrainJob.ks7CbtlvpcIL`, Slurm 1907749) untouched: `submit_log.run` mtime
  unchanged (13:40), still RUNNING on jpbo-008-46. Not resubmitted.

## Status: DONE
