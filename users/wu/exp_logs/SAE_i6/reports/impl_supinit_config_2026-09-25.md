# impl: config/sae_i6_p0_supinit.py (supervised-init subgraph under its own manager), 2026-09-25

**Status: BLOCKED.** The i6 phone prior is upstream of p0. `PhoneNgramPriorJob.qJxXHgXLe31S`
(`prior.npz`) is a direct input of p0's training `ReturnnTrainingJob.Y1vbqR6KeJSx`, so p0's hash,
its selection, its exported recognizer and its dev-other PER read all depend on which prior is chosen.
Gold phi (`ReturnnTrainingJob.Ac2eioZbRX7d`) does NOT depend on the prior.

## File written (not committed; nothing under recipe/ edited)
- `config/sae_i6_p0_supinit.py` (new, 22 lines): py() returns `{"supervised_init": supervised_init.py()}`.
  It does not call `base.register_inputs`, because that function registers `sae/4a/lm/prior.npz` as an output.
  Docstring states that it is a subset of `config/sae_i6_p0.py` with the same hashes.

## Checks (console only, read-only; scripts in the session scratchpad)
1. Graph: 65 jobs (53 finished, 1 runnable, 11 waiting). All 65 ids are in `config/sae_i6_p0.py`'s
   graph (164 jobs); none is missing. Registered outputs: the supervised_init outputs plus the inputs'
   own feats/units stats outputs, which the package registers while building the inputs.
2. Forbidden jobs present, all finished: `PhoneNgramPriorJob.qJxXHgXLe31S`, `SampleLinesJob.CrPgeKXsOosb`,
   `PhonemizeWithSilJob.NpoY1pGJWNUJ`, and `FlatRecognizerInitJob.0J9d6wjrkRYH`.
   Dependency path (reverse edges, traced in the console):
   PhonemizeWithSil -> SampleLines (text.phn.gz) -> PhoneNgramPrior (text.phn.gz) -> p0 training
   Y1vbqR6KeJSx (prior.npz) -> GetBestPtCheckpointJob.rpORzC4ZhN0p -> ExtractSubmoduleCheckpointJob.v1bPX23lPGsH
   -> ReturnnForwardJobV2.HkvgYqWPtkCP (dev-other posterior dump) -> BlankfreeGreedyPerJob.TVPzxTiROAkk.
   Mechanism: `reverse_model/p0.py` `p0_train_config` starts from `build_train_config(**data)`,
   which is ctrl_20's config. That config sets the model arg `prior_npz_path` (`training/config.py:312`), and
   the model loads it at construction (`model/emc_model.py:395`). `reverse_model/p0_steps.py` contains
   no reference to the prior, so the supervised loss probably does not use it numerically. This is NOT verified.
   The dependency in the hash is certain. The only ReturnnTrainingJobs are gold phi and p0. There is no k2/HLG job.
3. Non-finished jobs, with their run-task rqmt and the partition that settings.check_engine_limits picks
   (called in the console on the task; no submission). Sub-tasks create_files/plot have empty rqmt (time 2 h, default partition):
   - ReturnnTrainingJob.Ac2eioZbRX7d (gold phi), waiting: gpu 1, gpu_mem 96, cpu 16, mem 64, time 4 -> time raised to 72 h, `-p gpu_48gb`.
     gpu_mem 96 is the GH200 value; on L40S it is only a routing key.
   - ReturnnTrainingJob.Y1vbqR6KeJSx (p0), waiting: gpu 1, gpu_mem 96, cpu 16, mem 64, time 3 -> 72 h, `-p gpu_48gb`.
   - ReturnnForwardJobV2.HkvgYqWPtkCP (p0 posterior dump), waiting: gpu 1, gpu_mem 24, cpu 4, mem 24, time 2 -> flex route,
     at the time of the check `-p gpu_48gb --comment=flex24` (this depends on the queue state at submit time).
   - SeedGoldPhonesJob.BUWKmEsK2zTk, waiting: cpu 4, mem 32, 4 h, default partition.
   - PhoneTargetHdfJob.foScplXsRIMc, waiting: cpu 2, mem 16, 2 h, default.
   - BlankfreeSeedSupportJob.cMzBOdHlXVZN, waiting: cpu 2, mem 16, 0.5 h, default.
   - CvHoldoutSplitJob.zYcc8EJsvdfV, SupervisedReverseDataJob.onjA2xZUQBdx, waiting: cpu 1, mem 8, 1 h, default.
   - ExtractSubmoduleCheckpointJob.v1bPX23lPGsH, BlankfreeGreedyPerJob.TVPzxTiROAkk, waiting: cpu 2, mem 16, 1 h, default.
   - GetBestPtCheckpointJob.rpORzC4ZhN0p, waiting: empty rqmt (2 h, default).
   - DownloadHuggingFaceSnapshotJob.7tLASYdh10dO, runnable: empty rqmt (2 h, default). It has no work dir yet.
     It is probably the train-clean-100 MFA download (`data/gold.py` get_mfa_alignments). This is inferred, not checked.
4. Overlap with `config/sae_i6_p0_screen.py` (75 jobs): all 53 finished jobs are shared. None of the
   12 non-finished jobs is in the screen graph, so the two managers would not both submit any job.

## Undetermined / for the orchestrator
- p0 cannot be made prior-free by config: its hash includes the prior. Removing that input requires a recipe change,
  which would give a new hash and depart from the banked p0.
- Gold phi could run now without the prior. This would need a config that registers only gold phi's outputs (not done: outside the brief).
