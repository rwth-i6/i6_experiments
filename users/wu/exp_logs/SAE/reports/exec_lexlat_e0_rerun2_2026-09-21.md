# Executor report: lexlat e0 census rerun (2026-09-21)

DONE

- Confirmed old manager pid 1858841 was dead (`ps -p 1858841` empty) before touching anything.
- Confirmed fix commit: recipe/2025-10-speech-llm HEAD is 6cbb3f9, which `merge-base --is-ancestor 299922f HEAD` shows IS a descendant of 299922f ("size the destination block to the frame, not to a constant"). `git diff --stat HEAD -- src/speech_llm/sae/emc/lexlat.py` is empty (no local uncommitted changes).
- In job dir work/speech_llm/sae/emc/lexlat_jobs/LexlatCensusJob.RDMTvt4ngAPB:
  - renamed log.run.1 -> log.run.1.guard.backup
  - removed error.run.1
  - removed submit_log.run
  - (pre-existing log.run.1.oom.backup left untouched)
- Started manager: `python tools/sisyphus/sis --log_level 20 m -r config/sae_4a_lexlat_probes.py -io`, pid 2378407 (background nohup launched from bash pid 2378404), log: log/sae_4a_lexlat_probes.manager.20260921T041115Z.log
- Manager log showed only one runnable job: alias/sae/4a/lexlat/e0_census (LexlatCensusJob.RDMTvt4ngAPB). No training/data/prior-gap job appeared as runnable. E-1 (lq61PSAg1DcC) and trie build (rlMsnTBSZXsB) not listed (already FINISHED per instructions).
- Submitted with Slurm job id 1922956. rqmt: cpu=8, mem=64.0GB, time=6.0h, gpu=1, gpu_mem=96, partition booster --exclusive.
- Polled sacct: job reached RUNNING at poll 4 (~3 min after submit). `sacct -j 1922956`: 1922956_1 RUNNING, Elapsed 00:00:58.
- Tail of stripped log.run.1 (5 lines): job started subtask arg id 0, no errors, RSS 37MB, normal startup — consistent with the AssertionError fix taking effect (past the point where it previously crashed in _dest_block).

No blockers. Manager 2945974 and its jobs were not touched.
