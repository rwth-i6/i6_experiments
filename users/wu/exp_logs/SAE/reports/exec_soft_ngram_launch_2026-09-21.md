DONE

Manager: pid 1969214 (`sis m -io -r config/sae_4a_soft_ngram.py`), started 2026-09-21 18:27:53 UTC,
exited cleanly once its graph completed (no error/queue_error markers). Log:
log/sae_4a_soft_ngram.manager.20260921T182753Z.log; pid file log/sae_4a_soft_ngram.manager.pid.

Graph matched the census exactly: 4 NgramModeSeekingJob, no pack job. All four FINISHED on disk
(finished marker present, no error.run.* in any dir):

- w4NlQzcoXIfd (ep1)  -> work/i6_experiments/users/wu/experiments/unsupervised_asr/ngram_mode_seeking/NgramModeSeekingJob.w4NlQzcoXIfd -- FINISHED
- E4gg11FZ1itE (ep4)  -> .../NgramModeSeekingJob.E4gg11FZ1itE -- FINISHED
- EKkpTHPdLKGD (ep10) -> .../NgramModeSeekingJob.EKkpTHPdLKGD -- FINISHED
- KmbcDX5k6aaS (pooled) -> .../NgramModeSeekingJob.KmbcDX5k6aaS -- FINISHED

Slurm ids observed for these tasks were transient (already drained by the time of the disk check);
squeue at report time showed only jobs belonging to the concurrent lexlat_k2_overcount manager
(1926421_1 running, 1936422_1/1936423_1 running), none for sae_4a_soft_ngram.

Output artifacts under output/sae/4a/soft/ (symlinks into work/, all resolve):
- output/sae/4a/soft/ep1/{ngram_mode_seeking.json,summary_ngram.md}
- output/sae/4a/soft/ep4/{ngram_mode_seeking.json,summary_ngram.md}
- output/sae/4a/soft/ep10/{ngram_mode_seeking.json,summary_ngram.md}
- output/sae/4a/soft/{ngram_mode_seeking.json,summary_ngram.md} (pooled, top-level)

No blocker.
