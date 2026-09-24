# exec_k2_probe_rerun2_2026-09-21

Task: recover the five k2 probes after the previous executor's managers (2371400, 2371983) died
from a black=None TypeError because they were started without the sis venv on PATH.

## Confirmed dead
`ps -p 2371400 2371983` returned nothing before any action; both old managers were already gone.

## Cleanup
For each of the 5 job dirs under `work/speech_llm/sae/emc/lexlat_k2_jobs/`
(`LexlatK2ProbeJob.{Tkl94t85j4pV,by7UYEjtYdel,R7QzD6vYBLD3,5dP7W2NMFq4T,DtWYToXTPh6w}`) removed
`error.run.1` and `submit_log.run`. Left `job.save` untouched: `job.py::_sis_setup_directory`
(called from `manager.py::run_jobs` for every runnable job on each manager start, since a fresh
process has `_sis_setup_since_restart=False`) rewrites `job.save` unconditionally before submit,
so the stale None-black pickle is replaced automatically once black resolves correctly.

## Managers started (inside `source .../sis_env/bin/activate`, verified `which black` ->
`/e/project1/spell/wu24/env/sis_env/bin/black` before each start; used `tools/sisyphus/sis`
directly since the `sis` alias only exists in interactive shells, not this non-interactive one)
- `config/sae_4a_lexlat_k2.py`: first `-io` start (pid 2421692) found the leftover `usage.run.1`
  from the killed attempt made both jobs read as `interrupted_not_resumable`, not `error`, so
  `-io` didn't unblock them and the manager exited idle (STALLED). Restarted with `-cio`, pid
  **2432123**, log `log/sae_4a_lexlat_k2.manager.20260921T192507Z.log`. Cleared+resubmitted:
  `Tkl94t85j4pV` slurm **1937725**, `by7UYEjtYdel` slurm **1937724**. Both stateless probe jobs
  (not training), so `-cio` clearing is safe per skill.
- `config/sae_4a_lexlat_k2_official.py`: same pattern, restarted with `-cio`, pid **2439016**, log
  `log/sae_4a_lexlat_k2_official.manager.20260921T192602Z.log`. Cleared+resubmitted:
  `R7QzD6vYBLD3` slurm **1937770**, `5dP7W2NMFq4T` slurm **1937769**, `DtWYToXTPh6w` slurm
  **1937771**.
- Confirmed both pids alive via `ps` after start. Did not touch `sae_4a_lexlat_k2_overcount.py`
  (1969011) or `sae_4a_soft_pack.py` (409858).

## Confirmation from disk
All 5 job dirs show fresh `submit_log.run` with the Slurm ids above, and a fresh `log.run.1`
(timestamps ~21:27-21:28 UTC+2, well after manager restart) containing 0 occurrences of
`TypeError`/`expected str` and showing dataset loading in progress (MetaDataset sequence list
reads) -- i.e. the config write succeeded and the k2 child process launched. No error markers
present. No squeue/sacct/background wait used; confirmed via one poll loop under 2 minutes.

Watcher commands for the orchestrator:
```
bash ~/.claude/skills/sis/sis_watch.sh 2432123 config/sae_4a_lexlat_k2.py 600
bash ~/.claude/skills/sis/sis_watch.sh 2439016 config/sae_4a_lexlat_k2_official.py 600
```
