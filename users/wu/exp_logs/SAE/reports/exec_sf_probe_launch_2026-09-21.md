# sf_lam probe launch, 2026-09-21

Manager: sae_4a_sf_lam_probe, started via `sis m -io -r config/sae_4a_sf_lam_probe.py` (venv sourced
first), pid 295756, log `log/sae_4a_sf_lam_probe.manager.20260921T091326Z.log`. Manager exited cleanly
after its one job (no live-process residual; nothing else in this graph was runnable).

Job: work/speech_llm/sae/emc/sf_scorer_jobs/SfLamProbeJob.OHrcN9pEuXni
Slurm job id 1926336, JobState=COMPLETED, RunTime 00:02:40 (StartTime 11:18:02, EndTime 11:20:42).
finished + finished.run.1 present on disk.

## output/summary.txt (arm ctrl_20, sub-epoch 4, tau 2.0)
checkpoint PackedBlankfreeTrainJob.5EIGJJ1MkcO9/output/ctrl_20/models/epoch.004.pt
scorer NeuralPhoneLmTrainJob.xObXEwRpvmzd; unigram PhoneNgramPriorJob.RtzbESkOedsT
G=8 exact FFBS draws, 3 batches (first of arm's sub-epoch ordering).

Median ratio |grad sf|/|grad l_tau| = 2.20294
LAM_SF_01 (0.1x l_tau) = 0.0453938
lam_sf (0.3x l_tau) = 0.136181

Per-batch: b0 ratio 2.36821 peak 19.57 GiB 19.3s; b1 ratio 2.18608 peak 23.64 GiB 14.9s;
b2 ratio 2.20294 peak 23.07 GiB 14.2s. Peak over run: 23.64 GiB.

Monitors (mean of 3 batches): sf_reward_mean -0.534895, sf_reward_std_within 9.6777,
sf_unique_strings 8, sf_masked_long 0, sf_adv_absmean 8.34797.

No error markers; log.run.1 ends "Job finished successfully".
