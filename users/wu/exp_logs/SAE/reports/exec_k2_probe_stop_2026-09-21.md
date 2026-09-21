# exec_k2_probe_stop 2026-09-21

DONE

## Actions and state
1. Confirmed `LexlatK2ProbeJob.R7QzD6vYBLD3` log.run.1: k2 CUDA fault ("illegal memory access", error 700) at 18:26:13; usage.run.1 frozen RSS/VMS since 18:26 while CPU kept ticking at 20:17 -- matches the hung-child description. `scancel 1932516`: job left squeue within ~10s (cancelled cleanly, no state left in squeue).
2. Killed manager pid 857002 (`kill 857002`); `ps` confirms no `m -r config/sae_4a_lexlat_k2_official.py` process remains.
3. Builds `LexlatOfficialHLGBuildJob.GxCkDk90bQpT` and `.NrghE6fnf5hc` both have `finished.tar.gz` on disk (finished at 18:19 and 18:40, before I acted) -- **the manager had already submitted both dependent probes before I intervened**: `LexlatK2ProbeJob.5dP7W2NMFq4T` (Slurm 1932565, host jpbo-076-33) and `.DtWYToXTPh6w` (Slurm 1933402, host jpbo-018-29) both show the identical k2 error-700 crash + hung child in their log.run.1/usage.run.1. I scancelled both (1932565, 1933402); both left squeue within ~10s. `YJsdBTcEQJz9` build: finished.tar.gz present, untouched. No `WordLmPerplexityJob` directory exists anywhere on disk (graph shows `WordLmPerplexityJob.ddxHefQ0vl2D` waiting, never started) -- nothing to verify finished there; not caused by my actions.
4. Moved log.run.1 + usage.run.1 into `output_failed_k2fault/` for all four affected probe dirs: R7QzD6vYBLD3, Tkl94t85j4pV (as instructed), and additionally 5dP7W2NMFq4T, DtWYToXTPh6w (extended, since they hit the identical bug before I could stop the manager). No error/interrupted markers existed on any of the four (consistent with hung, not-yet-errored child); nothing cleared or resubmitted.

## Note
Console `get_jobs_by_status` misreported NrghE6fnf5hc as "runnable" and the two finished builds/probe DtWYToXTPh6w inconsistently -- trusted on-disk `finished.tar.gz`/log timestamps per the known console-misreport trap.

Manager pid 857002: killed, not restarted (fix still pending). Slurm jobs cancelled: 1932516, 1932565, 1933402.
