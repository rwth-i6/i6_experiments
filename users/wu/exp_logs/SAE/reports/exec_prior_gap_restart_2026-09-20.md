# sae_4a_prior_gap restart check, 2026-09-20

Status: DONE (no restart needed)

## Evidence
- pid 3594890: confirmed dead (`ps -p` empty).
- Manager log `log/sae_4a_prior_gap.manager.20260920T211253Z.log`: 0 Traceback occurrences; ends cleanly at 23:53:22 with `All output calculated`, a normal DONE exit (not a crash).
- All four jobs finished on disk:
  - NeuralPhoneLmTrainJob.xObXEwRpvmzd (3.3M): `finished.tar.gz` present; SLURM 1919330 COMPLETED, ExitCode 0:0, elapsed 00:17:20.
  - NeuralPhoneLmTrainJob.pBozvj6c3l16 (10.9M): `finished.tar.gz` present; SLURM 1919329 COMPLETED, ExitCode 0:0, elapsed 00:23:50.
  - PriorGapAnalysisJob.5wNIQs2lpC5P: `finished.tar.gz` present.
  - PriorGapAnalysisJob.pg14aEYJyiva: `finished` + `finished.run.1` present, no error.* marker.
- Output artifacts exist on disk:
  - output/exp2025_11_06_speech_llms/librispeech/sae_4a_prior_gap/ctrl_50_ep10/dev-other_neural_e30/{prior_gap.json,prior_gap.md,per_utt.json}
  - output/exp2025_11_06_speech_llms/librispeech/sae_4a_prior_gap/ctrl_50_ep10/dev-other_neural_l6w384_e30/{prior_gap.json,prior_gap.md,per_utt.json}
- Console graph read (`-s -c get_jobs_by_status(skip_finished=True)`) returned only one job under key `finished`, no runnable/queue/error entries — consistent with a fully drained graph.

## Classification
(a)-adjacent but actually already complete: the manager exited normally after computing all output; the watcher's STALLED(runnable=1) read appears to be a stale/transient race against that DONE exit (or graph re-evaluation timing), not real remaining work. No error markers anywhere; no host-exclusion or FS-outage evidence needed.

## Action taken
None. No manager restarted (pid 3594890 correctly stayed dead; nothing runnable). Other managers (3849944, 4004154, 2945974, 1355841) untouched. No error markers cleared, no code edited.
