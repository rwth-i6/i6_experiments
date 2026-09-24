# Executor report 2026-09-21: pause lexlat E1 + kill budget-pack N=100

## A. Lexlat E1 probe (config/sae_4a_lexlat_e1.py)
- Job: work/speech_llm/sae/emc/lexlat_train_jobs/LexlatEfficiencyProbeJob.x3LaY6KlB6I4
- Slurm 1923286: scancel issued, went to CG (completing) then gone.
- Manager pid 2967198: already exited (confirmed by dispatch, unchanged).
- Timer lines seen in log.run.1 ([lexlat] step N: T B padded m_max time alloc reserved):
  - step 0: T=770 B=114 padded=87780 m_max=3840 867.031s alloc 38.71GiB reserved 41.10GiB
  - step 1: T=689 B=127 padded=87503 m_max=3520 911.564s alloc 37.91GiB reserved 41.13GiB
  - step 14: T=687 B=128 padded=87936 m_max=2977 874.700s alloc 38.05GiB reserved 41.13GiB
  - step 15: T=764 B=113 padded=86332 m_max=2711 826.150s alloc 36.50GiB reserved 41.13GiB
  - step 28: T=615 B=128 padded=78720 m_max=4775 802.304s alloc 39.87GiB reserved 43.08GiB
  (only 5 timer lines present, all reported; abort=null in partial.json, arm=lexlat_20/ctrl_50_ep10)
- LexlatWordCountsJob.1ZJy5dFbOAHD (Slurm 1923287, CPU): FINISHED (finished.tar.gz present on disk). Left untouched.

## B. Budget-pack N=100 kill (config/sae_4a_budget_pack.py, manager pid 2945974, SAE_4A_budget.md)
Live Slurm ids confirmed via submit_log.run + squeue (all PENDING, ReqNodeNotAvail/maintenance) before cancel:
- Node A: work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.ks7CbtlvpcIL, Slurm 1921418 cancelled.
  Arms: ctrl_100 (last epoch 067), ctrl_50 (050), odmprior_100 (067), odmprior_50 (050)
- Node B: PackedBlankfreeTrainJob.reEI2Nd0S77A, Slurm 1921334 cancelled.
  Arms: bt_100 (066), bt_50 (050), odmbt_100 (065), odmbt_50 (050)
- Node C: PackedBlankfreeTrainJob.4QzmftNlbErt, Slurm 1921332 cancelled.
  Arms: nosched_ctrl_100 (067), nosched_ctrl_50 (050), nosched_odmprior_100 (067), nosched_odmprior_50 (050)

Post-cancel squeue -u wu24 shows none of 1921418/1921334/1921332.
Manager 2945974: killed; `ps -p 2945974` confirms dead (no process).

No job dir, output, checkpoint or error marker was deleted.
