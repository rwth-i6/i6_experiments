# exec_prior_probe_launch_2026-09-20

DONE

- Config: config/sae_4a_prior_probe.py imports config_sae_4a_prior_probe_v1 (confirmed).
- Manager pid: 3849944 (alive), log: log/sae_4a_prior_probe.manager.20260920T214015Z.log
- Manager loaded graph, logged runnable(3) exactly matching the three requested job hashes, then "Start manager".

Job dirs (all present on disk, submit_log.run written, ~90s after start):
- work/speech_llm/sae/emc/prior_probe/NeighbourhoodProbeJob.ZhYOeCltF4wW (ep4) — submitted
- work/speech_llm/sae/emc/prior_probe/NeighbourhoodProbeJob.6JZvwE7T1UmA (ep10) — submitted
- work/speech_llm/sae/emc/prior_gap/PriorGapAnalysisJob.m6lUxAO65f6A (word-bigram row) — submitted, waits on NeuralPhoneLmTrainJob.xObXEwRpvmzd per dispatch note

No errors in the manager log. sae_4a_prior_gap manager (pid 3594890) and its jobs were not touched.
