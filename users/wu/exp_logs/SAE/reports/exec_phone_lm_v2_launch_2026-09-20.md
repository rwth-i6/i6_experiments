# Launch: sae_4a_phone_lm.py (2026-09-20)

Manager pid: 4004154
Log: log/sae_4a_phone_lm.manager.20260920T214940Z.log

Import sanity: config/sae_4a_phone_lm.py imports
speech_llm.prefix_lm.sis_recipe.exp2025_11_06_speech_llms.librispeech.configs.config_sae_4a_phone_lm_v2:py — confirmed.

Manager submitted the first two runnable jobs (rest waiting on them, as expected: waiting(3)):
- HeldLinesJob.SKxs9aPu2Sha — work/speech_llm/sae/emc/neural_phone_lm_v2/HeldLinesJob.SKxs9aPu2Sha — dir exists, SLURM job 1920042_[1] PENDING
- SampleLinesJob.tHBnfzwuo9ok — work/speech_llm/sae/emc/text_sample/SampleLinesJob.tHBnfzwuo9ok — dir exists, SLURM job 1920043_[1] PENDING

Not yet materialized on disk (waiting on upstream, expected): ExcludeLinesJob.nOMT5eGtLKDV,
NeuralPhoneLmTrainJobV2.vkNGAOeLgNsy, PriorGapAnalysisJob.OO0iAEVLKgOO.

No errors in manager log. Manager alive at ps confirm. Did not wait for CPU text jobs to finish
(10M lines, expected to take a while) per instructions — did not use run_in_background, all checks
done in foreground.
