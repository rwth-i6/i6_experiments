# A16 (a2) launch, priorscale_s -- 2026-09-24
Status: DONE (launched; not waited for completion)
- recipe/2025-10-speech-llm HEAD contains 2186ad57.
- Precondition: manager 3023733 gone. Graph read (config/sae_4a_lexlat_v2_priorscale_s.py): 38 ReturnnForwardJobV2 runnable + 1 PriorScaleSReadJob waiting = 39 unfinished (1 finished PhiEmissionMapJob, short-circuit listing). None of the 39 appeared in squeue or had submit logs before launch (runnable = not claimed by em 4111121, a14 4152192, phicontent 721657; exhaustive cross-graph load not done).
- Manager: `setsid nohup python ... m -io -r config/sae_4a_lexlat_v2_priorscale_s.py`, sis_env on PATH. Real pid 943255 (ppid 1); /proc environ PATH contains sis_env (1). Pid file log/sae_4a_lexlat_v2_priorscale_s.manager.pid corrected to 943255. settings.py untouched (JOB_AUTO_CLEANUP=True).
- Routing: 3 gpupack packs, all 38 forwards, srun --gres=gpu:1 per task, gres/gpu:4 per node:
  - 1990141 pack_052113_943255_73229 (16 forwards)
  - 1990142 pack_052124_943255_84824 (16 forwards)
  - 1990144 pack_052232_943255_52171 (6 forwards)
- Markers: /e/project1/spell/wu24/2026-07-13_unsupervised/log/gpupack/2026-09-24/pack_*_943255_*.{sh,members,*.batch}; per-job /e/project1/spell/wu24/2026-07-13_unsupervised/log/gpupack/jobs/<slurm id>
- sacct at report time: 1990141 COMPLETED, 1990142 RUNNING, 1990144 COMPLETED. Per-job finish not verified (executor does not wait).
