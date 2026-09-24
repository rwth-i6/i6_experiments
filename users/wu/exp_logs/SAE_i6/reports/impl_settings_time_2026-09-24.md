# settings.py: 72 h time floor for ReturnnTrainingJob run (2026-09-24)

Status: DONE. Only `check_engine_limits` in `SETUP/settings.py` changed. Nothing was launched, no manager
was started, git was not used, and the env files were not touched.

## Delta

After the existing `time = min(168, requested)` line, a new branch applies only when `task.name() == "run"`
and the task's job (`task._job`) is exactly of class `ReturnnTrainingJob` from module `i6_core.returnn.training`.
That branch sets `time = min(168, max(time, 72))` and carries a one-line comment giving the reason (the 11.5 h
was tuned on GH200, and the change avoids relying on TIMEOUT -> resume). The match is on the exact class
(module and name), not `isinstance`, so subclasses and all other jobs and tasks keep their old behaviour.
The partition routing is unchanged.

## Verification

Script: scratchpad `rqmt_check.py`, run with the sae python from the setup dir. It loads
`config/sae_i6_p0_screen.py` through `sisyphus.loader.config_manager`. For each task it prints
`gs.engine().get_rqmt(task, 1, update=False)`, which applies the engine defaults and the submit history and
then calls `check_engine_limits`. The graph has 75 jobs, including one ReturnnTrainingJob.

- `ReturnnTrainingJob.GiT88bxzoZbZ` (ctrl_20), run: 16 cpu, 64 GB, 1 gpu, gpu_mem 96, **time 72** (was 11.5), `-p gpu_48gb`
- same job, create_files and plot: engine short, time 1.0, no sbatch_args (unchanged)
- `ReturnnForwardJobV2.5z8bUupJ1f8j` run: 4 cpu, 64 GB, gpu_mem 24, time 8.0, `-p gpu_24gb` (unchanged)
- `BlankfreeGreedyPerJob.6oSHcmvHPHWI` run: 2 cpu, 16 GB, time 1.0, default partition (unchanged)
- `BlankfreeVadHdfJob.RLrgIh6lFv9m` run: 4 cpu, 96 GB, time 8.0, default partition (unchanged)

The ctrl_20 job id is still `GiT88bxzoZbZ`, as the dispatch gives it. This is expected because the time rqmt
is not part of the job hash, and `check_engine_limits` runs only when the engine resolves requirements.
The other 3 ReturnnTrainingJobs of the full P0 graph (k2lat_20_ma3000, gold phi, p0 supervised) will also
get 72 h through the same rule. Those jobs are not in the screen graph, so this run did not print their values.

## Notes

- If the job has a submit history and the recipe rqmt has not changed, `get_rqmt` keeps the last submitted
  time, and `check_engine_limits` then applies the floor to that time. Values above 72 h, for example after
  `update_rqmt` raises the time following a TIMEOUT, are kept, and the 168 h cap still holds.
