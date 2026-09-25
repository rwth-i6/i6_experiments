# Launch P1 rt_r90 probe (Launch B), 2026-09-25

Status: DONE_WITH_CONCERNS (minor: memory field is named `mem_usage:cuda`, not `mem_usage:cuda:0`)

Preconditions: only live manager before launch was P0 (pid 1646677, untouched); no ladder/fits manager.
Dry build P1_LADDER_STAGE=probe: 87 jobs; EexT85vdfx25 present (unfinished); fits ruLnJFWyifwp,
6IkAAuzcBwBR, uO8wkodbR2uh finished.

Manager pid 1700738 (env P1_LADDER_STAGE=probe verified), command from setup dir:
P1_LADDER_STAGE=probe PATH=/work/asr4/hwu/conda/envs/sae/bin:$PATH setsid nohup /work/asr4/hwu/conda/envs/sae/bin/python sisyphus/sis --log_level 30 m -r config/sae_i6_p1_ladder.py > log/sae_i6_p1_ladder.manager.log 2>&1 < /dev/null &
Manager log: log/sae_i6_p1_ladder.manager.log; pid file log/sae_i6_p1_ladder.manager.pid.

Job: work/i6_core/returnn/training/ReturnnTrainingJob.EexT85vdfx25
Slurm 4362041_1 RUNNING, gpu_48gb, node cn-506, gres/gpu:1; submit_log shows gpu_48gb.
log.run.1: "rt_chunked_backward: per-chunk k2 backward installed (chunk_seqs = 4)" present.
Stability line present: median 0.0826 nats per retained frame over 16/16 utterances (not FAILED).
Step 1: lexlat_k2_pre_peak_allocated_gib 24.648, lexlat_k2_peak_reserved_gib 32.117,
lexlat_k2_device_used_gib 32.639, mem_usage:cuda 26.5GB, 74.7 sec/step (early step, incl. warmup).
No FAILED / OOM lines. G1.M not yet read (needs end of sub-epoch 1). No watcher started.
