# Launch P0 prior T3 (2026-09-25)
DONE. Both parts submitted; nothing edited.
- Part A: Slurm 4365697, cpu_modern, RUNNING on cn-614. Out dir (partA_summary.json and prior/ go here):
  /work/asr4/hwu/setups/librispeech-960/2026-09-24-unsupervised/analysis_out/prior_t3/partA
  Log: .../partA/slurm-4365697.out. First lines: params (n_out 1010000, sil_prob 0.5); g2p emulation 773672 -> 384893 entries;
  lexicons union_emulated 584893; "scan 5000000 lines, emulation drops 103274, 33 s". No error/traceback
  (one benign warning: no settings.py in the job cwd).
- Part B: Slurm 4365700, -p gpu_32gb, --dependency=afterok:4365697, PENDING (Dependency).
  Log: /work/asr4/hwu/sae_i6_probes/p0_prior_t3_2026-09-25/slurm-4365700.out.
  Step-0 logs per arm: /work/asr4/hwu/sae_i6_probes/p0_prior_t3_2026-09-25/{i_i6prior,ii_partA}/returnn.log (and rnn.out, step0_found).
