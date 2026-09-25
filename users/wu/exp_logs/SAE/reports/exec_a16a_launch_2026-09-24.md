# A16 (a) launch, config/sae_4a_lexlat_v2_relabel_s.py (2026-09-24)
DONE
- recipe/2025-10-speech-llm HEAD contains 1e637fe3. JOB_AUTO_CLEANUP=True, unchanged. No prior manager on this config.
- Pre-start graph read (console, skip_finished): unfinished = 8 PhiRelabelJob (runnable), 8 ReturnnForwardJobV2 + 1 RelabelSReadJob.Y9RU2PSbVfuu (waiting). 17 in total. PhiFirstProbeTrainingJob.KWQGSa1BfXon and PhiEmissionMapJob.agYgeieCNYdE were listed under "finished" and have finished.tar.gz on disk. The forwards consume the new relabel outputs, so none can be shared with em (4111121) or a14 (4152192). The graph showed 8 forwards, not the 9 the review mentioned.
- Manager pid 3023733, started with setsid nohup < /dev/null. Its environ PATH contains sis_env (count 1). Pid file: log/sae_4a_lexlat_v2_relabel_s.manager.pid (corrected, because setsid forked).
- PhiRelabelJob x8 ran on the login node (short engine). All 8 are finished (finished marker or tar present).
- Forwards: ReturnnForwardJobV2 .8QnbPbvQOPSs .myDyOizw1S3m .NIB4nXPzrjV8 .eIUm8wJwks1o .cBGB69OA6ZQN .T5OprslZXDaO .dp3J1fx77Tbd .yyS8j8oADvX7, all in ONE gpupack SLURM job 1990055 (gpupack.le1h.8t, 1 node, 4 GPUs, PENDING at report time). Each task runs as srun --gres=gpu:1 with 2 tasks queued in sequence per GPU column. That is packing, not one exclusive node per forward.
- Pack marker: log/gpupack/2026-09-24/pack_045833_3023733_13884.{sh,members}; run marker dir log/gpupack/jobs/1990055
- Reader: login mini task, waiting on the forwards.
- The priorscale_s config was not started.
