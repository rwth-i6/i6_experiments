# Executor report: sae_4a_lexlat_e1_c4096 launch — aborted mid-task

2026-09-21. Task: launch config/sae_4a_lexlat_e1_c4096.py under its own manager.
Aborted by coordinator mid-launch: the C=1024 E1 (r4Iaa72mU27T, Slurm 1923108) died with
CUDA OOM at 83 GiB held, so the C=4096 run would fail the same way.

## Actions taken (in order)
1. Started manager `python tools/sisyphus/sis --log_level 20 m -r config/sae_4a_lexlat_e1_c4096.py -io`,
   pid 2721535, log `log/sae_4a_lexlat_e1_c4096.manager.20260921T050903Z.log`.
2. It submitted the expected NEW job LexlatEfficiencyProbeJob.eHrOFiwUqKAf as Slurm 1923148
   (rqmt cpu=16 mem=64G time=2h gpu=1 gpu_mem=96, booster --exclusive).
3. Coordinator sent abort mid-poll. Killed manager pid 2721535 (`kill 2721535`); confirmed
   gone via `ps` (no matching process).
4. Cancelled Slurm 1923148 (`scancel 1923148`). Final sacct state:
   `1923148_1 CANCELLED by 34349, Elapsed 00:01:20, ExitCode 0:0`
   (batch/step sub-entries show FAILED as side effect of the cancel).
5. Left job dir `work/speech_llm/sae/emc/lexlat_train_jobs/LexlatEfficiencyProbeJob.eHrOFiwUqKAf/`
   untouched (no marker rename, no -co/-cio, no delete).
6. Did not touch managers 2945974 (sae_4a_budget_pack, confirmed still alive) or 2582573
   (E1 C=1024 manager — not found running at check time; not modified by me).

## Status
BLOCKED (task aborted by coordinator instruction, not by an error I hit). No further action
taken on this config.
