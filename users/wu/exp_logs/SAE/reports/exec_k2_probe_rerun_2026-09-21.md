# exec_k2_probe_rerun_2026-09-21

Task: rerun the four dead k2 probes plus the rebuilt-graph probe by7UYEjtYdel.

## Correction to the brief
`config/sae_4a_lexlat_k2_pack_v1.py` does not exist and the actual shim
`config/sae_4a_lexlat_k2_pack.py` (module `config_sae_4a_lexlat_k2_pack_v1.py`) does NOT register
`LexlatK2ProbeJob.Tkl94t85j4pV`: its `py()` -> `build()` raises `ValueError` immediately
(`PROBE_SEC_PER_SUBEPOCH is not stated`, a downstream-arm gate, module-level `None` constants,
untouched) before any job is registered. Both `Tkl94t85j4pV` (the "all" build) and
`by7UYEjtYdel` (the `word_boundary` build on `LexlatHLGBuildJob.cdcxYJMjiYj5`, confirmed
finished) are registered together by `config_sae_4a_lexlat_k2_v1.py` via `settling_probe()`,
shimmed as `config/sae_4a_lexlat_k2.py`. That config had no live manager (only
`sae_4a_lexlat_k2_overcount.py` pid 1969011 was live, untouched). Confirmed both hashes present
in that config's graph before submitting.

## Cleanup done
All four dead probes already had `error.run.*`/`log.run.*` moved to `output_failed_k2fault/` by
an earlier session; only a stale `submit_log.run` remained in each. Removed
`submit_log.run` from `LexlatK2ProbeJob.{Tkl94t85j4pV,R7QzD6vYBLD3,5dP7W2NMFq4T,DtWYToXTPh6w}`
(all under `work/speech_llm/sae/emc/lexlat_k2_jobs/`). `by7UYEjtYdel` had no on-disk dir (first
submission). Confirmed no live worker for any of the 5 job dirs via `ps` first.

## Managers started
- `config/sae_4a_lexlat_k2.py`: pid 2371400, log
  `log/sae_4a_lexlat_k2.manager.20260921T191749Z.log`, pidfile
  `log/sae_4a_lexlat_k2.manager.pid`. Submitted: `Tkl94t85j4pV` slurm 1937488,
  `by7UYEjtYdel` slurm 1937487.
- `config/sae_4a_lexlat_k2_official.py`: pid 2371983, log
  `log/sae_4a_lexlat_k2_official.manager.20260921T191754Z.log`, pidfile
  `log/sae_4a_lexlat_k2_official.manager.pid`. Submitted: `R7QzD6vYBLD3` slurm 1937489,
  `5dP7W2NMFq4T` slurm 1937491, `DtWYToXTPh6w` slurm 1937490. That config also has two
  `LexlatOfficialHLGBuildJob` runnable (`GxCkDk90bQpT`, `NrghE6fnf5hc`), untouched, part of the
  same graph.

Submission confirmed by fresh `submit_log.run` (newer than manager pidfile) in each of the 5
`LexlatK2ProbeJob.*` dirs under `work/speech_llm/sae/emc/lexlat_k2_jobs/`, each containing a
Slurm id — never via squeue/sacct.

Watcher commands for the orchestrator:
```
bash ~/.claude/skills/sis/sis_watch.sh 2371400 config/sae_4a_lexlat_k2.py 600
bash ~/.claude/skills/sis/sis_watch.sh 2371983 config/sae_4a_lexlat_k2_official.py 600
```
