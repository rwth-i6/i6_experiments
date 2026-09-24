# exec_lexlat_probes_launch_2026-09-21

Manager pid 1542733, log log/sae_4a_lexlat_probes.manager.20260921T022058Z.log. Started, then killed
per coordinator's mid-task change of plan (code review found a crash in LexlatCensusJob's unpruned
cell; E0 must not start).

Jobs:
- LexiconTrieBuildJob.rlMsnTBSZXsB (work/speech_llm/sae/emc/lexlat_jobs/) Slurm 1922638: RAN TO
  COMPLETION on jpbo-007-01 before the stop instruction arrived (COMPLETED, ~6 min elapsed,
  finished/finished.run.1 markers present, output/exp2025_11_06_speech_llms/librispeech/sae_4a_lexlat/resource/{summary.txt,build.json}
  written). Not cancelled -- it had already finished. Will be re-hashed/rerun once the jobs file is edited.
- LexlatCensusJob.hJKXcRjKieh3 (E0) Slurm 1922668: was PENDING in queue (never started running).
  Cancelled via scancel.
- LexlatEquivalenceProbeJob.lq61PSAg1DcC (E-1) Slurm 1922669: was PENDING in queue (never started
  running). Cancelled via scancel.

Manager 1542733 killed; confirmed gone via ps. squeue confirms 1922668/1922669 no longer present.
No training/data jobs (PackedBlankfreeTrainJob, BoundedBlankfreeTrainingJob, PriorGapAnalysisJob)
were ever submitted by this manager -- only the three lexlat jobs above ever appeared runnable.
