# A15-F launch (exec, 2026-09-24)
DONE. Launched config/sae_4a_lexlat_v2_phicontent.py.
- Recipe HEAD (recipe/2025-10-speech-llm) contains 8972c436 and 48ca9bc9.
- Pre-launch graph read: 18 unfinished jobs, all in speech_llm/sae/emc/phi_content_controls/ (15 PhiContentJob, 1 TableContentJob, 1 PartitionBoundsJob runnable; PhiContentReadJob.vySKYIh5RAaB waiting). No other work; the module is new and not used by the em/a14/relabel_s configs. Those managers (4111121, 4152192, 3023733) were left alone.
- Manager: pid 721657, started with setsid nohup, `m -io -r`, on login node jpbl-s02-03. Its PATH contains sis_env (check returned 1). The pid file log/sae_4a_lexlat_v2_phicontent.manager.pid is corrected to 721657; the first pgrep hit, 721648, was the launching shell.
- settings.py not touched (JOB_AUTO_CLEANUP=True). priorscale_s not started.
- Routing: all 16 gpu-0 tasks are in ONE gpupack pack, Slurm 1990124 (gpupack.le1h.16t). It has 4 columns x 4 tasks, each srun uses 72 cpus. Pack walltime is 4:00:00, although each task asked for 1 h. At report time the pack was PENDING (Priority).
  Pack script: /e/project1/spell/wu24/2026-07-13_unsupervised/log/gpupack/2026-09-24/pack_051556_721657_56563.sh (plus the .members file)
- PartitionBoundsJob.0CHOXvZNOC9d already finished on engine short, host jpbl-s02-03 (login).
- The reader is still waiting on the pack, so its login routing is not yet observed.
