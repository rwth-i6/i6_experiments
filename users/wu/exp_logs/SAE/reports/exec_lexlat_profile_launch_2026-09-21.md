DONE

LexlatStepProfileJob.ItFHebFGK6nb launched via config/sae_4a_lexlat_profile.py.

Slurm job id: 1925267, state RUNNING (actively profiling sub-epochs, e.g. "sub-epoch 10: 57 batches ... profiling [0, 1]" in log.run.1; RSS ~11GB, 40.9% CPU at last check).

Manager pid: 3972993, log: log/sae_4a_lexlat_profile.manager.20260921T081022Z.log. Manager stayed alive (running, etime ~9min at last check).

Job dir: work/speech_llm/sae/emc/lexlat_profile_jobs/LexlatStepProfileJob.ItFHebFGK6nb (submit_log.run has Slurm id 1925267; no error marker present).

Operational note (not a config/hyperparameter change): first two launch attempts failed.
1. `sis m -io` alone hit the generic interactive startup prompt (EOFError, no `-r`); fixed by adding `-r`.
2. Second attempt submitted (Slurm 1925024) but the job crashed instantly: `ReturnnConfig._write_to_file` called `black` via subprocess and `self._black_path` was None because the manager's PATH lacked sis_env (matches memory note sis-manager-needs-venv-path.md). Fixed by activating sis_env before starting the manager, then clearing the errored, stateless profiling job via console `_sis_move()`/`_sis_setup_directory()` (job.save based rerun, no rqmt/hash change), and restarting the manager plainly.

No hyperparameter, dataset, or config content was changed; only the manager launch environment and one error-marker clear of a stateless profiling job.
