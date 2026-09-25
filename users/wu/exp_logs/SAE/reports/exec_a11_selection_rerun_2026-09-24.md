# A11 selection rerun 2026-09-24
DONE. HEAD of recipe/2025-10-speech-llm contains 9bada5fb.
Cleared EmTableSelectionJob.RyEwer4kERuw run state (error.run.1, submit_log.run removed; log.run.1, usage.run.1 -> *.backup).
Stopped manager 2099342 (had imported old module); restarted detached with sis_env on PATH: pid 2072482 (PATH check = 1).
Both tasks are mini_task (login short engine), no SLURM node used.
Selection: work/speech_llm/sae/emc/blankfree_emtable_read/EmTableSelectionJob.RyEwer4kERuw finished.tar.gz; output/selection.json, output/report.txt
Diagnostics: work/speech_llm/sae/emc/blankfree_emtable_diag/EmTableDiagnosticsJob.JjdvpYwqYyfH finished; output/diagnostics.json, output/report.txt
Manager 2072482 then exited (graph complete, 0-byte log at level 30 = clean end); no manager is running for this config now.
CONCERN: EmTableDiagnosticsJob ran on booster node jpbo-055-15 via SLURM (not login/gpupack), i.e. held a node alone for its runtime; recipe rqmt should route it (not changed here, job already finished).
