# Executor check: sae_4a_lexlat_probes E0 STALLED verdict, 2026-09-21

## Verdict
DONE — case (a), job FINISHED cleanly; manager exit was not a stall, it was normal completion racing the watcher's poll.

## Evidence
- `ps -p 2378407`: no such process (manager not running).
- Manager log `log/sae_4a_lexlat_probes.manager.20260921T041115Z.log` tail: submitted job_id 1922956 at 06:11:22, running by 06:14:53, then at 06:49:44-45 logged `Finished output: .../summary.txt`, `.../census.json`, `.../per_utt.json`, and `All output calculated`. Clean success exit, no error prompt, no crash.
- Job dir `work/speech_llm/sae/emc/lexlat_jobs/LexlatCensusJob.RDMTvt4ngAPB`: contains `finished` and `finished.run.1` markers.
- `sacct -j 1922956`: all steps `COMPLETED`, exit 0:0, elapsed 00:35:56, node jpbo-009-07.
- `output/exp2025_11_06_speech_llms/librispeech/sae_4a_lexlat/e0_census/`: `census.json`, `per_utt.json`, `summary.txt` all present.

## Summary (verbatim head, see full content in this report's source read)
Pruned-mass table: C1024/C256/C4096/Cunpruned x lam_lex 0.333/0.667/1.0, all pruned medians 0.0000.
FUNDING RULE: C=256 MISSES (logZ gap 0.0913), C=1024 MISSES (logZ gap 0.0512), C=4096 MEETS (reference).
KILL READ (a): augmented vs greedy PER, lam_lex=1.0 C=1024: paired delta -0.0509 (sd 0.0646, 76 better/9 worse) -> PASS (rule <= -0.01). Control lam_lex=0: paired delta -0.0330.
KILL READ (b): E[N] 54.06 -> 48.63, mean drop +9.66% -> PASS (rule: drop >10% fails).
Runtime 2076.3 s.

## Action taken
None — no restart needed; job already complete, output banked.
