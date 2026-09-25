# Launch: w2vu env rebuild (2026-09-25)

DONE. Slurm job 4365689 (sae_i6_w2vu_env_build), partition gpu_32gb, node cn-32, RUNNING since 19:51:26.
Log: log/w2vu_env_build.4365689.out

Pre-checks:
- build_w2vu_env.sh last commit 45c20acef; env/ has no uncommitted changes.
- squeue for sae_i6_w2vu_env_build was empty.
- /work/asr4/hwu/conda/envs/w2vu existed without bin/w2vu-python; it was removed (rm -rf) and its absence confirmed.

Preflight (log):
- line 34: `host cn-32  job 4365689  start 2026-09-25 19:51:26  CUDA_VISIBLE_DEVICES=0`
- line 36: `Tesla V100-SXM3-32GB, 7.0, 32768 MiB, 580.173.02`
- The mamba create ran through to its activation hint (lines 97-136). At 226 lines there was no error.
The full build was not waited on; the orchestrator watches it.
