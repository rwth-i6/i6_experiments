# Launch: P0 k2lat relaunch with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (2026-09-25)

J = work/i6_core/returnn/training/ReturnnTrainingJob.jcKXbLMDk4hl

Status: DONE. J is running on V100 gpu_32gb.

## Procedure
1. Created J/hold.
2. analysis_out/k2lat_alloc_expandable_install.sh printed `INSTALLED 2026-09-25 21:24:43`. The sha256 of J/output/returnn.config is aaf07df44339500181d159b273ac2423aabdc2625695b6c2aa599f0de4973a8d, which matches the pinned value.
3. Removed J/error.run.1 and nothing else.
4. Removed J/hold at about 22:04. The live manager (pid 1646677) resubmitted J within minutes as Slurm 4368174_1. submit_log.run shows `-p gpu_32gb` with completed_fraction 0.35. The job runs on cn-32. No other manager was started.

## Which log is which
The task has tries=1, so no log rotation happens. The new run rewrote J/log.run.1, which is a hardlink of engine/...run.4368174.1. The OOM run's log survives only as engine/i6_core.returnn.training.ReturnnTrainingJob.jcKXbLMDk4hl.run.4365260.1.

## Checks on the new log.run.1
- `alloc_conf: expandable_segments:True` appears once (line 63): PASS.
- The env dump contains PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (line 77): PASS.
- epoch.007.pt and epoch.007.opt.pt are loaded: PASS.
- "Starting training at epoch 8, global train step 399" is present: PASS.
- `installed (chunk_seqs = 2)` appears exactly once: PASS.

## Steps 0-3, new run vs OOM run 4365260
- l_tau: 1.902, 1.912, 1.898, 1.911 in both runs.
- lexlat_k2: 0.424, 0.402, 0.401, 0.411 in both runs.
- Pre-k2 peak allocated (GiB), new run: 19.907, 27.220, 26.690, 26.047.
- Pre-k2 peak allocated (GiB), old run: 19.908, 27.224, 26.694, 26.051.

The losses are identical at the logged precision. Pre-k2 peak allocated memory is 1-4 MiB lower in the new run, which fits a different allocator mode. Step 17 has not been reached yet; the orchestrator's loop is watching for it.
