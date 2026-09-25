# L2-0 k2 pre-flight rerun read (LadderK2PreflightJob.Pl51viCk4CVP, Slurm 1975698), 2026-09-23
Verdict: PASS under A7.
Job dir: work/speech_llm/sae/emc/blankfree_ladder_jobs/LadderK2PreflightJob.Pl51viCk4CVP (finished; sacct COMPLETED 0:0, 28:38)
Source: output/preflight.json, output/preflight.txt, output/rt_r0/log.run.1
- returncode -15, stop_reason "step budget reached (100 train steps)" (wrapper stop of healthy child); error_lines [].
- ABORT / "EMPTY pruned" / overflow / OOM in rt_r0/log.run.1: 0 matches.
- Stability read sub-epoch 1: median 0.0809 nats/retained frame, 16/16 utts, 2.4 s after HLG load (not failed). Sub-epoch 2: 0.0042, 16/16.
- Steps: 100/100 logged, epochs_seen [1,2]; first step 83 s after launch; after first: sec/step median 16.12, mean 15.56, max 23.56. lexlat_k2_sec median 4.87.
- Peak GPU: nvidia-smi 78495 MiB = 76.7 GiB (<= 80); k2 peak reserved 75.86 GiB.
- Projection: sub-epoch 1 = 57 steps, train 937 s + dev/ckpt 67 s -> 8 x = 2.23 h vs pack TIME_RQMT 11.5 h (config_sae_4a_lexlat_v2_ladder_v1.py:181 TIME_RQMT = gp.TIME_RQMT, chain resolves to config_sae_4a_budget_v1.py:147 = 11.5). No resume needed.
- Warnings: empty-lattice frac max 0.008 (n_empty total 2 over 100 steps, no abort); 15x benign "settings.py does not exist" RETURNN warning. Pruned lattice collapses fast (arcs/frame 16360 -> median 39).
