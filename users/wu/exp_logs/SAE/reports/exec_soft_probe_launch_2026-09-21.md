# Launch: sae_4a_soft_probe (2026-09-21)

Manager: started `sis m -io config/sae_4a_soft_probe.py` (had to pipe `y` into the initial
"Start manager (y)..." confirmation prompt, not an old-error prompt). Manager pid 3986319,
log `log/sae_4a_soft_probe.manager.20260921T081115Z.log`. Manager exited on its own once the
single job reached error state (no runnable/waiting work left) -- this is not a manager crash.

Job: `work/speech_llm/sae/emc/soft_scorer_jobs/SoftLamProbeJob.wAJQ26T7iZzX`
- submit_log.run: SLURM id 1925281, rqmt cpu16/mem64GB/gpu1/gpu_mem96/time1h, engine slurm.
- squeue progression: PENDING -> CONFIGURING -> RUNNING -> (gone, job ended) within ~4 min.
- State: error (error.run.1 present, empty; real traceback in log.run.1).

Root cause (from log.run.1 tail): `torch.autograd.backward` inside the scorer's
gradient step raised `CUDA out of memory. Tried to allocate 224.61 GiB` (GPU has 95GiB total,
89GiB free at the time) while backprop-ing through a parameter tensor of shape ~[...,17...].
This looks like an unbounded/broadcast autograd graph in SoftLamProbeJob's backward call
(224GiB single allocation), not a resource-rqmt shortfall -- gpu_mem is already at the 96GB max
this cluster offers, so raising rqmt would not fix it. This is a job/algorithm defect
(memory guard on the allocated tensor shape), which is out of executor scope to patch.

No output artifact was produced (job never finished).
