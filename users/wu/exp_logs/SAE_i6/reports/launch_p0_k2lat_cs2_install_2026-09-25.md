# P0 k2lat chunk-2 install (restart steps 1-3), 2026-09-25

J = work/i6_core/returnn/training/ReturnnTrainingJob.jcKXbLMDk4hl (Slurm 4359832, gpu_32gb V100)

1. Probe check: slurm-4363361.out in /work/asr4/hwu/sae_i6_probes/p0_k2lat_v100_ep8_2026-09-25_cs2 has `installed (chunk_seqs = 2)` once and `rnn exit 0` once.
2. Drift check, 14a8042d7..HEAD on users/wu/experiments/unsupervised_asr: 15 files, all newly added (status A), 5146 insertions and 0 deletions. They are all w2vu2 files, from commits 68b39418a, 692d6e55a and 0dbe9cf7f. The w2vu2 files under model/ and training/ are new files and do not touch the k2 path. `git status --short` on the path is clean. Verdict: no drift.
3. The script k2lat_perchunk_returnn_cs2_install.sh exited with rc 0 and printed `INSTALLED 2026-09-25 18:45:15`. Its diff adds only epilog lines 239-246: the rt_chunked_backward train_step import, and a get_model partial with lexlat_k2_chunk_seqs=2. The full output is in analysis_out/cs2_install_run.log. The patch k2lat_perchunk_cs2_sae_i6_p0.patch printed "patching file config/sae_i6_p0.py" and exited with rc 0. No .rej files were produced.
4. Verification: the sha256 of J/output/returnn.config is 3a53ad6c0ad127c66cbed724a37063e439ea7de287b523e24b92f2ca0a3792a2, which matches the expected value. Slurm 4359832 is RUNNING. J/hold exists. The newest checkpoints are epoch.006.pt and epoch.006.opt.pt, and there is no epoch.008.pt. I did not cancel J, remove the hold or touch the manager.
