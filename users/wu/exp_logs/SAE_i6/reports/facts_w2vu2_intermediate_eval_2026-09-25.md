# w2vu2 GAN: intermediate checkpoints and intermediate evaluation (facts, 2026-09-25)

Read-only investigation for SAE_i6 P0, gate G0.GAN. No job, env or config was touched. The live
manager (pid 1786381, `config/sae_i6_w2vu2.py`) was not disturbed.

Path prefixes used below:

- `P` = `recipe/i6_experiments/users/wu/experiments/unsupervised_asr` (the package; `P/config/w2vu2.py`
  is the entry that `config/sae_i6_w2vu2.py` calls, and nothing else).
- `FS` = `/work/asr4/hwu/conda/envs/w2vu/lib/python3.9/site-packages` (fairseq 0.12.2).
- `EX` = `FS/fairseq/examples/wav2vec/unsupervised`.

State at 21:06 CEST: no `FairseqHydraTrainingJob` directory exists yet under `work/i6_core/fairseq/training/`,
so no GAN seed has started. The p_sil 0.5 text data job has finished. The `TextToPhonemeJob` is still running.

## 1. Checkpoint retention and size

**Every checkpoint is retained.**

- GAN yaml (`P/training/w2vu2_gan.py:91-96`): `save_interval: 1000`, `save_interval_updates: 1000`,
  `no_epoch_checkpoints: true`, `best_checkpoint_metric: weighted_lm_ppl`. No `keep_*` key is set.
- fairseq defaults (`FS/fairseq/dataclass/configs.py:691-740`): `keep_interval_updates`,
  `keep_interval_updates_pattern`, `keep_last_epochs` and `keep_best_checkpoints` are all -1.
- Deletion (`FS/fairseq/checkpoint_utils.py:136-185`) happens only when a `keep_*` value is > 0.
- The i6_core job (`recipe/i6_core/fairseq/training.py`) deletes nothing and has no `path_available`
  override. `keep_epochs=[]` (`P/training/w2vu2_gan.py:377-386`) only affects which files are exposed
  in `out_models`.

**All 150 update checkpoints are written.**

- The name is `checkpoint_{E}_{U}.pt` when `U % 1000 == 0` and the update is not at an epoch end
  (`checkpoint_utils.py:74`).
- With N_train = 28,539 and batch 160 there are 179 updates per epoch. No multiple of 1000 is a
  multiple of 179, so no save coincides with an epoch end.
- Files run from `checkpoint_6_1000.pt` to `checkpoint_838_150000.pt`, with E = ceil(U/179). Each seed also
  has `checkpoint_best.pt` and `checkpoint_last.pt`.
- Caveat: 179 updates per epoch holds only for N in [28,481, 28,640].

**Size.**

- Per checkpoint, about 34 MB. This is an estimate from the parameter count, not a measurement: the
  model has 2,841,409 parameters (generator plus discriminator), stored as fp32 weights plus two Adam
  moments, which gives 34.1 MB.
- Per seed, 152 files, about 5.2 GB. For 5 seeds, about 26 GB.
- The JUPITER review measured 4.4 GB per seed
  (`exp_logs/SAE/reports/review_gan_port_2026-09-25.md:100`), so the estimate is an upper bound.
- `/work/asr4/hwu` has 940 GB free (df, 21:06).

**Atomicity.** The first file in the save list is `checkpoint_{E}_{U}.pt`. It is written by
`trainer.save_checkpoint` (`checkpoint_utils.py:115`) through `torch_persistent_save`, which writes
`.tmp` and then renames (`checkpoint_utils.py:549-558`). `best` and `last` are copies of it
(`:125-127`). A reader that tests for the final name therefore never sees a partial update checkpoint.

## 2. In-training UER

**Logged, but meaningless: it is 0.0.**

- The task validates every 1000 updates on `valid_subset: valid` (`P/training/w2vu2_gan.py:101-119`, `labels: phn`).
- `valid` = dev-clean + dev-other concatenated (`P/data/w2vu2_features.py:70`, `FAIRSEQ_SPLITS`).
- The feature job writes only npy / lengths / km / ids / stats.txt (`P/data/w2vu2_features.py:283-293`, `:398`).
  So `valid.phn` does not exist, and `ExtractedFeaturesDataset` sets labels to None when
  `<split>.<labels>` is missing (`EX/data/extracted_features_dataset.py:56-57`).
- `P/lm/w2vu2_text.py:33` states this design explicitly: "Only the train split exists: the GAN's valid split is label-free".
- With no target, `valid_step` sets `c_len = pred_c_len` and makes no edit-distance errors
  (`EX/tasks/unpaired_audio_text.py:174-290`). `reduce_metrics` then logs
  `uer = errors*100/_num_chars = 0.0` (`:340-416`).

**If labels existed:**

- The UER would be pooled over dev-clean + dev-other, as a single `valid` subset.
- Decoding would be: JOIN segmenter in eval mode (runs of equal argmax averaged, `EX/models/wav2vec_u.py:146-196`),
  then argmax, pad masking, specials removed, `<SIL>` dropped, and `unique_consecutive` only if `ctc_eval`
  (default False, `unpaired_audio_text.py:64`), scored by editdistance.
- It could not give per-set PER.
- Supplying `valid.phn` would change the `W2vu2FeatureDataJob` output, and with it every GAN training hash. Not usable.

**Where it appears.** The `valid` JSON line of `<train job>/work/outputs/*/*/hydra_train.log`
(`P/config/w2vu2.py:119`) and `log.run.1`. The same line carries `weighted_lm_ppl`, the real label-free
selection metric, every 1000 updates.

## 3. Coverage of the post-hoc eval

It evaluates `checkpoint_best.pt` only.

- Per seed, there is one generator conversion, a forward and a PER on dev-clean and on dev-other
  (`P/config/w2vu2.py:238-253`).
- The checkpoint input is `train.out_checkpoint_dir.join_right("checkpoint_best.pt")` (`:241`).
- The selected seed's train-clean-100 pseudo-labels come from the same kind of forward (`:256-265`).
- The Path carries its creator, so availability falls back to `Job.path_available`, which is "creator
  finished" (`sisyphus/sisyphus/job_path.py:146-161`, `sisyphus/sisyphus/job.py:1101-1109`). Nothing
  runs until the training has finished.
- No intermediate checkpoint is evaluated.

## 4. Recognition on CPU or gpu_11gb

**No k2 and no CUDA needed.**

- The forward model is `P/model/w2vu2_generator.py` (`greedy_decode`: argmax, collapse, drop `<SIL>`)
  with `P/model/recognizer_only.py` / `P/model/recognizer.py`. `ConvRecognizer` casts features to the
  parameter dtype. No k2 import appears on this path, and `P/model/__init__.py` imports nothing.
- Conversion (`W2vu2GeneratorCheckpointJob`, `P/analysis/w2vu2_gan_eval.py:134-201`) is a mini_task.
  It uses `torch.load(weights_only)` without fairseq.
- PER (`W2vu2GanPerJob`, `:353-401`) is a mini_task. Mini tasks run on the local 'short' engine
  (`settings.py:5-12`, `LocalEngine(cpus=4, mem=8)`), that is, on this desktop.

**Setting device="cpu".**

- `w2vu2_forward_job(device=...)` (`P/analysis/w2vu2_gan_eval.py:302-347`, default "gpu") sets `gpu_mem`
  only for gpu.
- `ReturnnForwardJobV2` sets rqmt `gpu = 1 if device == "gpu" else 0` (`recipe/i6_core/returnn/forward.py:223-288`).
  `device` sits in `post_config` (`:375`), which the hash excludes (`recipe/i6_core/returnn/config.py:314-323`).
- The hashed part of the forward config is the batch size of 1.6M frames and max_seqs 200 (`w2vu2_gan_eval.py:254-299`).
- With gpu=0, `check_engine_limits` (`settings.py:386-420`) leaves the job on the default `cpu_modern`,
  with a memory cap of 180 GB (`:36`). `FORWARD_RQMT` (`P/config/w2vu2.py:194`) asks for 4 cpu, 24 GB and 2 h.

**Precedent.** The CPU RETURNN chain reproduced the fairseq PER on JUPITER
(`exp_logs/SAE/reports/impl_gan_port_B_2026-09-25.md:96-104`).

**Runtime estimate.** This is not a measured i6 run.

- About 1 min of compute for both dev sets at 4 threads.
- About 2-3 min wall per split forward, including RETURNN start-up.
- The dev-other HDF (1.6 GB) read in 15.3 s from this host.
- Slurm queueing comes on top.

**sae env.**

- torch 2.7.1+cu126, compiled arch flags sm_50 60 61 70 75 80 86 89 90.
- So GTX 1080 (sm_61) and RTX 2080 (sm_75) would run this k2-free forward.
- `get_arch_list()` is empty on the desktop because there is no driver there; the list comes from `_cuda_getArchFlags`.

**Routing.**

- GPU jobs are routed only to `GPU_ROUTE_FLEX` (gpu_24gb, gpu_48gb) or `GPU_ROUTE_TRAIN` (gpu_32gb)
  (`settings.py:69-77`). gpu_11gb is never picked automatically and would need an explicit
  `sbatch_args -p gpu_11gb`.
- The project rule forbids gpu_11gb for GPU jobs because k2 lacks sm_61/75. This forward has no k2,
  but CPU is the simpler route and loses nothing at about 1 min of compute.

## 5. g2p lexicon completeness (i6)

**Chunks.** All 16 `ApplyG2PModelJob` chunks are non-empty and match their word lists
(`P/lm/lexicon.py:141-155`, `filter_empty_words=True`, concurrent=16).

**Empty pronunciations.** Exactly one: `HHH` (chunk 7). Sequitur produced no pronunciation for it,
and the merge/filter step, which keeps only 4-field lines (`recipe/i6_core/g2p/apply.py:103-125`), dropped it.
`_load_g2p_lexicon` also requires 4 tab fields (`P/lm/lexicon.py:67-82`). The merged g2p lexicon has no
empty pronunciation.

**Counts.**

| Quantity | i6 | JUPITER |
|---|---|---|
| Merged entries | 773,672 | |
| g2p word types with a pronunciation | 388,779 | 388,780 requested |
| Defective g2p lexicon entries | | 384,893 |

- The single missing type on i6 is HHH.
- The earlier `reports/extract_i6_g2p_counts_2026-09-25.md` says 773,673 merged; that is off by one.
- The one-word difference left open in `SAE_i6_P0.md:265-301` / `SAE_i6_ref.md:155-165` is HHH.

**Dropped lines (measured).**

- `PhonemizeWithSilJob.NpoY1pGJWNUJ/output/stats.txt` reads `lines_in=40418261`, `lines_out=40418258`,
  `dropped_oov=3`.
- The 3 dropped lines are the 2 lines containing HHH (8465501, 8987220) and one empty line.
- `FairseqTextDataJob.ZoHEQDEMVSbS/log.run.1` (the GAN's text data, alias `sae/1c/text_data_sil0.5`,
  finished 21:05) reads "text_data: 40418258 lines, 3382467116 tokens (incl. </s>), 44 symbols (40 in
  dict.txt), 0 <unk>".

**Not yet measured.** `TextToPhonemeJob.YralP5FWlGjA` (alias `w2vu2/sae/0b/phonemize_lm_corpus`, Slurm
4366222) is still running and has no output vars yet. The expected values are `num_sentences_in`
40,418,261, `num_sentences_out` 40,418,258, `num_unresolved` 1, and `unresolved_words.txt` = HHH
(`P/lm/w2vu2_text.py:213-280`).

**How to count dropped lines.**

- PhonemizeWithSilJob: `stats.txt` `lines_in - lines_out`.
- TextToPhonemeJob: the output vars `num_sentences_in - num_sentences_out`, with `num_unresolved`
  and `unresolved_words.txt`.
- FairseqTextDataJob: the "text_data: N lines ... U <unk>" log line (`P/lm/w2vu2_text.py:119-175`),
  with N compared to the input line count.

## 6. Minimal change: per-set PER every K updates, on CPU, with no training-hash change

**Mechanism.** The training job and its arguments stay untouched. New jobs consume its output through
a sisyphus Path with a custom availability check.

- `Path(path, creator=None, ..., available=None)` (`sisyphus/sisyphus/job_path.py:59-83`) with
  `available` set short-circuits the creator-finished rule (`:146-147`).
- The Path hash is (creator, path) (`:129-161`, `:241-243`), so the availability callable is not hashed.
- The callable *is* pickled with the Path (`__getstate__`, `:260-273`). It must therefore be a
  module-level function, not a lambda.
- `ReturnnTrainingJob.path_available` (`recipe/i6_core/returnn/training.py:345`) is the standard
  i6 precedent for evaluating while training runs.

**Delta** (one file in the package, `P/config/w2vu2.py`, plus one module-level helper; no new job class):

```python
def _file_exists(path):                      # module level: picklable
    return os.path.isfile(path.get_path())

UPDATES_PER_EPOCH = 179                      # N_train 28,539 / batch 160
for u in range(K, 150_001, K):
    ck = tk.Path(f"checkpoints/checkpoint_{math.ceil(u / UPDATES_PER_EPOCH)}_{u}.pt",
                 creator=train, available=_file_exists)
    gen = W2vu2GeneratorCheckpointJob(fairseq_checkpoint=ck, text_dict=..., net_args=...)  # unchanged class
    for split in ("dev-clean", "dev-other"):
        fwd = w2vu2_forward_job(..., device="cpu", alias suffix f"u{u}/{split}", **cpu rqmt without gpu_mem)
        per = W2vu2GanPerJob(hyps=fwd.out_files["hyps.json"], gold=inputs.gold, split=split)
        tk.register_output(f".../intermediate/seed{s}/u{u}/{split}.per", per.out_...)
```

- Everything else is the existing per-seed block at `:238-253`, with `checkpoint_best.pt` replaced by
  the per-U file.
- The training hash does not change, because `FairseqHydraTrainingJob` arguments are unchanged.
- The existing checkpoint_best eval hashes do not change either. `device` is not hashed, but keep GPU
  as the default there and pass `device="cpu"` only on the new jobs.

**Robustness to the epoch index.** If 179 updates per epoch were wrong, the named file would never
appear. The eval would then stall safely rather than evaluate a wrong checkpoint.

- Check it once, when `checkpoint_6_1000.pt` appears in seed 1's `output/checkpoints/`.
- A fix changes only the eval hashes.
- The alternative is a resolver job that globs `checkpoint_*_{U}.pt`. It is more robust but needs a new
  job class, and the implementer and code-reviewer would then have to handle two changes.

**K = 5000**, giving 30 points per seed.

- Cost: 5 seeds x 30 x 2 splits = 300 CPU forwards at about 12 core-min each, about 60-80 core-hours.
  There are also 150 conversion and 300 PER mini tasks, each taking seconds locally.
- K = 1000 would be 1,500 forwards, about 300 core-hours.
- All 150 checkpoints are kept (item 1), so the curve can be made denser later, for example around the
  `weighted_lm_ppl` minimum, without touching training.
- `weighted_lm_ppl` is already logged every 1000 updates, so the PER points can be paired with it.

**Adding while the manager runs.**

- Sisyphus does not hot-reload the config. The edited config takes effect only after pid 1786381 is
  stopped and the same command is restarted
  (`python sisyphus/sis --log_level 30 m -r config/sae_i6_w2vu2.py`).
- This is a restart, not a second manager. Running Slurm jobs are unaffected and are re-attached.
- No GAN training directory exists yet, so doing this now costs nothing.

**No dependence on the finished training.** The intermediate jobs' only input from the training is the
custom-available per-U Path, which becomes available when the atomic rename of `checkpoint_{E}_{U}.pt`
happens (item 1). Each (seed, U) point is therefore evaluated as soon as that checkpoint exists.

**Required process.** This is recipe code under `P/config/`, so it needs the implementer, then the
code-reviewer before the restart, then a commit with the review report (project rules).
