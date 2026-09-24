# Executor report: sae_4a_phone_lm.py (phone LM instance c) — 2026-09-21

## Manager
- Log: log/sae_4a_phone_lm.manager.20260920T214940Z.log
- pid 4004154: not present in ps (exited).
- Log ends with `Finished output: .../prior_gap.json`, `.../prior_gap.md`, `.../per_utt.json`, then `All output calculated`. Clean success exit, no traceback, no interactive prompt.

## Jobs
### NeuralPhoneLmTrainJobV2.vkNGAOeLgNsy (training, instance c)
- Dir: work/speech_llm/sae/emc/neural_phone_lm_v2/NeuralPhoneLmTrainJobV2.vkNGAOeLgNsy
- Markers: only `finished.tar.gz` present (JOB_AUTO_CLEANUP archived it) — counts as finished.
- train_log.json: 5/5 epochs completed, no early stop (no-improvement abort not triggered). Final epoch 5: held_perplexity=3.9627 (epoch1 was 4.321, monotonically improved).
- Outputs present at output/exp2025_11_06_speech_llms/librispeech/sae_4a_phone_lm_v2/phone_lm_10m_l8w512/{model.pt,train_log.json,heldout.txt} (symlinks into work/.../output/, targets exist).

### PriorGapAnalysisJob.OO0iAEVLKgOO
- Dir: work/speech_llm/sae/emc/prior_gap/PriorGapAnalysisJob.OO0iAEVLKgOO
- Markers: `finished`, `finished.run.1`, `log.run.1`, `usage.run.1` — fully finished, not archived.
- Outputs present: output/exp2025_11_06_speech_llms/librispeech/sae_4a_phone_lm_v2/ctrl_50_ep10/dev-other_neural_10m_l8w512/{prior_gap.json,prior_gap.md,per_utt.json} (symlinks resolve).

## Conclusion
Everything this config registers is finished (graph "finished: 1" matches — training job's descendant, the gap job, is the terminal registered output). This is the same transient-race pattern as the sibling config: manager exited cleanly after "All output calculated" once the last job (prior gap) finished, and the watcher's STALLED report was a race against that exit, not a real failure. No restart performed, no action taken.
