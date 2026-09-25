# P1 arms launch (rt_r70, rt_r80), 2026-09-25

Status: DONE. Executed per reports/review_p1_arms_launch_2026-09-25.md, Launch steps 2, 3 and 5. No config or code edited; no watcher started.

- Probe-stage manager 1700738 killed; ps shows no manager on config/sae_i6_p1_ladder.py afterwards. P0 manager 1646677 untouched.
- rt_r90 Slurm 4362041_1 (ReturnnTrainingJob.EexT85vdfx25) RUNNING on cn-506 before and after; one squeue entry, submit_log.run has a single submission (not resubmitted/duplicated).
- Probe manager log moved to log/sae_i6_p1_ladder.probe.manager.log.
- New arms manager pid 1726329 (log/sae_i6_p1_ladder.manager.pid), started 17:20. environ: P1_LADDER_STAGE=arms, PATH starts with /work/asr4/hwu/conda/envs/sae/bin:. Manager log shows only deprecation warnings.
- rt_r80 ReturnnTrainingJob.5CL3pQyyOvQt: Slurm 4363117_1, gpu_48gb, RUNNING on cn-506. log.run.1 contains the chunk_seqs = 4 k2 backward line.
- rt_r70 ReturnnTrainingJob.9lD2HS2Mzgcl: Slurm 4363118_1, gpu_48gb, RUNNING on cn-507. Briefly PENDING (reason None) before starting. The chunk_seqs line was not yet in log.run.1 at 17:23 (still initialising); needs a recheck.
- Both submit_log.run: ['-p','gpu_48gb'], time 72. Both output/returnn.config identical to analysis_out/p1_ladder_rt_r{70,80}_2026-09-25.returnn.config.
- Nothing else submitted: squeue holds only the 3 P1 trainings, P0 GiT88bxzoZbZ (cn-508) and 3 pre-existing gpu_32gb trainings.
