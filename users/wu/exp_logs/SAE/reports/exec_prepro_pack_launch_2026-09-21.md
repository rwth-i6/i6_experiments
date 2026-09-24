# Executor report: sae_4a_prepro_pack launch (2026-09-21)

DONE

- Manager: pid 347372, started with `sis --log_level 20 m -r config/sae_4a_prepro_pack.py -io`
  Log: log/sae_4a_prepro_pack.manager.20260920T230546Z.log (alive, no error lines)
- FlatRecognizerInitJob.DMSwTLXT9MWG: submitted (slurm 1921037), finished this run
- FlatRecognizerInitJob.0J9d6wjrkRYH: already present/finished, not resubmitted
- TrimmedAudioBlankfreeDataJob (qb4o6dlW3urA, hxIx0ItTvx15, 0IOLr6hZnYWj): untouched, manager only logged "clean up" (post-finish housekeeping), no rerun
- PackedBlankfreeTrainJob.5EIGJJ1MkcO9 (matches expected hash): job dir
  work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.5EIGJJ1MkcO9
  create_files finished; run task submitted as Slurm 1921103, rqmt cpu64/mem256GB/gpu4/gpu_mem96/time11.5h,
  partition booster --exclusive (matches expected 4 GPU / 11.5h one node)
  sacct: 1921103_1 RUNNING on node jpbo-021-06 (confirmed via squeue too)
- Arms started: YES, all three confirmed training with evidence from per-arm logs
  (output/<arm>/log.run.1): ctrl_20, ctrl_20_s1, prepro_20 each show
  "ep 1 train, step 2, ..." lines around 01:13, no error text in any log.run.1
- No other jobs submitted by this manager beyond the FlatRecognizerInitJob and the pack job
- Blocker: none

Full report path: /e/project1/spell/wu24/2026-07-13_unsupervised/recipe/i6_experiments/users/wu/exp_logs/SAE/reports/exec_prepro_pack_launch_2026-09-21.md
