# Executor report: lexlat E1 relaunch (C=1024, C=4096) 2026-09-21

Code checked: recipe/2025-10-speech-llm at d2f6a5e (SAE 4A lexlat: block DP over batch axis; cap and instrument E1) — confirmed via git log --oneline -1.

## config/sae_4a_lexlat_e1.py (C=1024)
- Manager pid 2967198 (also parent bash launcher 2967196), log log/sae_4a_lexlat_e1.manager.20260921T055018Z.log. Manager exited cleanly after submitting (nothing else runnable).
- Submitted: LexlatEfficiencyProbeJob.x3LaY6KlB6I4 -> Slurm 1923286 (gpu=1, mem=64G, time=4h, booster --exclusive), state RUNNING; log.run.1 already shows progress ("sub-epoch 10: 57 batches, longest T at index 48 (T=1137)").
- Also runnable/submitted (expected, allowed): LexlatWordCountsJob.1ZJy5dFbOAHD -> Slurm 1923287 (CPU, mem=16G, time=2h), state RUNNING.
- No other jobs were runnable; old orphans r4Iaa72mU27T, eHrOFiwUqKAf, X2YnVYfqN7aV untouched (not in current graph).

## config/sae_4a_lexlat_e1_c4096.py (C=4096)
- Manager pid 2982979, log log/sae_4a_lexlat_e1_c4096.manager.20260921T055018Z.log. Exited cleanly after submitting.
- Submitted only: LexlatEfficiencyProbeJob.fY18YN7LVdAe -> Slurm 1923324 (gpu=1, mem=64G, time=4h, booster --exclusive), state PENDING (queued behind the other GPU job).

No PackedBlankfreeTrainJob/BoundedBlankfreeTrainingJob/data job/PriorGapAnalysisJob appeared runnable in either graph. Manager 2945974 (budget pack) untouched.

DONE
