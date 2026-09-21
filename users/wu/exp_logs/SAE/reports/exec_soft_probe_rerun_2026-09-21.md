# Executor report: SoftLamProbeJob rerun (2026-09-21)

DONE. Slurm job id 1925609. Manager pid 4135363, log
log/sae_4a_soft_probe.manager.20260921T082849Z.log (started `-io`).

Cleared job.wAJQ26T7iZzX: removed error.run.1 and submit_log.run, renamed
log.run.1 -> log.run.1.oom1. Manager submitted within seconds; job finished
in ~6 minutes (well inside the 10-20 min window).

## Summary (output/summary.txt, arm ctrl_20_ep4, epoch 4)

- median ratio |grad soft| / |grad l_tau| = 0.306148
- lam_soft for target 0.1 x l_tau: 0.326639
- lam_soft for target 0.3 x l_tau: 0.979918
- per-batch ratios: 0.298342, 0.306148, 0.335182 (3 batches, 88000 padded
  frames, max_seqs 128)
- artefact_gap_mean = 0.013789 (pre-registered ceiling 0.3 nats/token, not
  enforced by this probe since it runs no arm)
- monitors mean: soft_term_mean -0.0471163, soft_reward_mean -0.236283,
  soft_scored 121.333, soft_tokens 134.609
- wall time 77.1s; checkpoint
  work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.5EIGJJ1MkcO9/output/ctrl_20/models/epoch.004.pt

Job dir: work/speech_llm/sae/emc/soft_scorer_jobs/SoftLamProbeJob.wAJQ26T7iZzX
Output: .../output/summary.txt, .../output/probe.json
