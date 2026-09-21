# Executor report: relaunch sae_4a_lexlat_probes (2026-09-21)

## Manager
- pid 1858841 (parent bash wrapper pid 1858838, ignore)
- log: log/sae_4a_lexlat_probes.manager.20260921T030507Z.log
- Started with `-io` from setup dir after activating sis_env.

## Cleanup performed
- `work/speech_llm/sae/emc/lexlat_jobs/LexlatEquivalenceProbeJob.lq61PSAg1DcC/` had a stale
  `submit_log.run` referencing Slurm 1922669 (scancelled while PENDING) and NO log.run/error/finished
  markers. Deleted `submit_log.run` only, per instructions, before starting the manager.
- `LexiconTrieBuildJob.rlMsnTBSZXsB` untouched (already FINISHED). Orphan
  `LexlatCensusJob.hJKXcRjKieh3` left alone.

## Submissions confirmed (manager log)
Only two runnable jobs at manager start: E0 (LexlatCensusJob.RDMTvt4ngAPB) and E-1
(LexlatEquivalenceProbeJob.lq61PSAg1DcC). No PackedBlankfreeTrainJob / BoundedBlankfreeTrainingJob /
PriorGapAnalysisJob / data job appeared as runnable; manager was not killed.

- E0 LexlatCensusJob.RDMTvt4ngAPB: Slurm 1922782, rqmt cpu=8 mem=64GB time=6h gpu=1 gpu_mem=96,
  partition booster --exclusive.
- E-1 LexlatEquivalenceProbeJob.lq61PSAg1DcC: Slurm 1922783, rqmt cpu=4 mem=32GB time=4h,
  partition booster.

## State at report time (~20 min after submit)
- E-1 (1922783_1): RUNNING on jpbo-037-23, ~18 min elapsed, log.run.1 actively growing
  (loaded lexicon trie / gold phones / greedy PER inputs, no finished/error marker yet). NOT
  terminal within the poll window.
- E0 (1922782): FAILED — `error.run.1` present. Root cause: CUDA OutOfMemoryError, tried to
  allocate 37.44 GiB, process already held 80.37 GiB of a 95 GiB device. This is a batch/config
  size issue, not a scheduler `rqmt` shortfall (gpu_mem=96 already requests the full device) — I
  did not resubmit or alter anything; flagging as a blocker per scope (fixing it would change the
  experiment's batch/config).

## Blocker
E0 (LexlatCensusJob.RDMTvt4ngAPB) OOMs at current config; needs owner decision on batch size /
memory strategy before relaunch, not an executor-fixable rqmt.
