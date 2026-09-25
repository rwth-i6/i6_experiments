# P0 k2lat cs2 resume of J (2026-09-25)
J = work/i6_core/returnn/training/ReturnnTrainingJob.jcKXbLMDk4hl

Step 1: config sha 3a53ad6c...92a2 confirmed; squeue -j 4359832 empty; no error.run.1 present (nothing renamed); hold removed 19:12:48; manager pid 1646677 alive (not restarted).
Step 2: manager resubmitted at ~19:14. New submit_log.run line: -p gpu_32gb, time 72, gpu_mem 96 (routing tag), Slurm 4365260_1. RUNNING since 19:14:21 on cn-32 (V100-SXM3-32GB, cap 7.0).
Step 3: NOT VERIFIED. By 19:36 (22 min after start) RETURNN had not yet started training. The job is still staging its train HDFs through the cache manager (cf): log.run.1 shows 12 "ERROR: cannot receive: timed out [_readAll] / no connection to master" pairs, each ~2 min, then falls back to "parsing file" of the /work HDFs (units/feats train shards). No Load model, Starting training, chunk_seqs, rt_chunked_backward, stability or ep-8 step lines yet. No Traceback, no CUDA OOM, no work/lexlat_k2_ABORT.json.
Note: log.run.1 was restarted by the new attempt (1913 lines before -> fresh 157 lines); work/returnn.log still holds the prior runs.
Step 4: learning_rates epoch 7 has train scores and global_train_step_end = 399 but NO dev score (cancel hit the dev eval). Expected resume step X = 399.
Next check: grep work/returnn.log for "Load model", "Starting training at epoch 8", "chunk_seqs", "rt_chunked_backward", "stabil", "^ep 8 train".
