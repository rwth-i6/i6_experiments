# Executor report: k2 build/probe launch, 2026-09-21

Setup: /e/project1/spell/wu24/2026-07-13_unsupervised
Config: config/sae_4a_lexlat_k2.py

## Action taken
Checked no existing manager for this config was running, then started:
  /e/project1/spell/wu24/env/sis_env/bin/python tools/sisyphus/sis --log_level 30 m -io -r config/sae_4a_lexlat_k2.py
Manager pid: 1704172 (recorded in log/sae_4a_lexlat_k2.manager.pid)
Manager log: log/sae_4a_lexlat_k2.manager.20260921T103524Z.log
Started at 2026-09-21T10:35:24Z.

## State at check (foreground, ~4.5 min after start)
- Manager process 1704172 alive (etime 04:30), no crash.
- Manager log tail shows only one benign WARNING line (absolute-path-inside-workdir for
  vad_port.py); no ERROR lines yet.
- LexlatHLGBuildJob.rtX44PBJFNy1* : job directory NOT YET present under work/ (find returned
  nothing). Graph/config load for this recipe (heavy imports) is apparently still in progress;
  no submit_log to report yet, no Slurm id yet.
- LexlatK2ProbeJob.Tkl94t85j4pV* : job directory NOT YET present under work/ (expected, since it
  depends on the build job).

No errors observed. No scontrol job id available yet since the build job has not been submitted
to Slurm as of this check. Manager must be left running (or the setup's watcher engaged) to catch
submission of the build job and, later, the probe job.

## Not touched
Other live managers (409858 soft pack, 628195 lexlat profile) and their jobs were not touched.
