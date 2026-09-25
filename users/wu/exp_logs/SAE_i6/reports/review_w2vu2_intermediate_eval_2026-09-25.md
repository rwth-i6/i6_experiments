# Review: w2vu2 GAN intermediate evaluation on CPU (2026-09-25)

**FAIL (DONE_WITH_CONCERNS).** The graph change is hash-safe, and the sisyphus availability mechanism is
sound. The checkpoint file name is wrong for 26 of the 30 points per seed, so those points never run.

Reviewed: the uncommitted diff of `P/config/w2vu2.py` and `P/tests/test_w2vu2_config.py` in
`recipe/i6_experiments` (branch `haotian_cycle_consistency_unsupervised`), the setup-local
`config/sae_i6_w2vu2.py`, the facts report and the implementer report (rounds 1 and 2).
`P` = `recipe/i6_experiments/users/wu/experiments/unsupervised_asr`, `FS` = the w2vu env's site-packages
(fairseq 0.12.2). Nothing was edited, submitted or restarted. No manager was started.

## Findings

### F1 (blocking): 180 updates per epoch, not 179. 26 of 30 points per seed wait forever, silently.

`P/config/w2vu2.py:229` sets `GAN_UPDATES_PER_EPOCH = ceil(28539/160) = 179`. The comment at `:226` says
that 179 holds for any N_train in [28,481, 28,640]. That is false. fairseq's `required_batch_size_multiple`
defaults to 8 (`FS/fairseq/dataclass/configs.py:482-483`), and the GAN yaml does not override it. The
trainer passes it to `batch_by_size` (`FS/fairseq/trainer.py:705`). As a result, the 59-utterance remainder
(28,539 - 178 x 160) is split into two batches, of 56 and 3 utterances.

**Evidence 1: fairseq's own batching code.** I ran `fairseq.data.data_utils.batch_by_size` in the w2vu
env, with N = 28,539, max_sentences 160 and required_batch_size_multiple 8. Both the `vec` and the `fn`
Cython paths give **180 batches**, and the last three batches hold 160, 56 and 3 utterances. For N =
28,488 the result is 179 batches and for N = 28,641 it is 180, so the comment's range is wrong in both
directions.

- No size filter removes any utterances. The task's `max_positions()` returns None, and `min_length=3` is
  already applied by the feature job.
- The update frequency is 1 and the world size is 1, so one batch is one update.

**Evidence 2: production.** Production's s0 checkpoint_best is byte-identical to `checkpoint_823_148000.pt`
(`exp_logs/SAE/reports/sae_attrib_step1_audit_2026-09-19.md:32`). The port's converted checkpoint reports
epoch 823 at step 148000 (`impl_gan_port_B_2026-09-25.md:52`). The production train split also has
28,539 utterances (`impl_gan_port_A:82`).

- ceil(148000/180) = 823, which matches.
- ceil(148000/179) = 827, which does not.

**What breaks.** Of the 30 points, U = 25,000 to 150,000 (26 points) get epoch indices that fairseq never
writes. For example, the code expects `checkpoint_140_25000.pt`, but fairseq writes
`checkpoint_139_25000.pt`. I confirmed on the built graph that s0 u150000 expects
`checkpoint_838_150000.pt`, whereas fairseq will write `checkpoint_834_150000.pt`.

- Affected jobs: 130 of the 150 conversions, and the 260 forwards and 260 PER jobs that follow them.
- Only U = 5,000 to 20,000 evaluate. This is 20k of the 150k updates.
- The failure is silent. The Path has a creator, so sisyphus never reports it as input_missing
  (`sisyphus/graph.py:506-509`). The custom `available` also skips the "finished but not available" warning
  (`job_path.py:146-147`). WAITING is logged only at debug level, and the manager runs at `--log_level 30`.
- The facts report's proposed check does not catch this. At `checkpoint_6_1000.pt`, 179 and 180 give the
  same epoch index. The first save that tells them apart is U = 7,000: fairseq writes
  `checkpoint_39_7000.pt`, while 179 predicts `checkpoint_40_7000.pt`.

### F2 (blocks the naive fix): three points fall on epoch ends and never have an update-named file.

With 180 updates per epoch, U = 45,000, 90,000 and 135,000 are exact epoch ends. At an epoch end, fairseq
writes neither file:

- no `checkpoint_E_U.pt`, because its condition is `not end_of_epoch` (`FS/fairseq/checkpoint_utils.py:74-78`);
- no `checkpointE.pt`, because `no_epoch_checkpoints: true` is set.

Only best and last are written. Changing `:229` to 180 alone would make the assert at `:257` fire for those
three U at graph time, and the manager would fail to load the config. The fix must drop or shift those
three points, and the test at `P/tests/test_w2vu2_config.py:180` (`range(5000, 150_001, 5000)`) must change
with it.

This also affects the facts report. Its claim of "all 150 update checkpoints" (facts `:33-35`) is wrong: the
16 saves at multiples of 9,000 are epoch ends, so 134 update-named files exist per seed, and the last is
`checkpoint_834_150000.pt`. That matters if the curve is later made denser.

No test covers the checkpoint name or the epoch mapping, so F1 passes all 12 tests.

## Checks that passed

**(a) Hashes.** I built the graph in-process (no manager, `scratchpad/rev/dump_graph.py`) twice: with
HEAD's `w2vu2.py` (from `git show HEAD:`, executed as the same module) and with the working tree.

- The graph has 101 jobs before the change and 851 after. All 101 old ids are present, and none is missing.
- Unchanged attributes: class and aliases for every old job, and rqmt and device for every old job except
  the 10 checkpoint_best dev forwards. Those 10 changed from gpu 1 / gpu_mem 40 to gpu 0 with device cpu, as
  intended.
- The select/train forward is unchanged (gpu 1, gpu_mem 40, device gpu). The training ids and rqmt are
  unchanged.
- There are 750 new jobs (150 conversions, 300 forwards, 300 PER jobs), all with `/intermediate/` aliases.
  The new forwards all have `{cpu 4, gpu 0, mem 24, time 2}` and device cpu.

**(b) Availability.** Custom `available` short-circuits the creator-finished rule (`job_path.py:146-147`).
It adds no output to the training and does not change the training's hash, `_sis_finished` or run.

- fairseq writes the update checkpoint first, through `torch_persistent_save`, which writes `.tmp` and then
  calls `os.rename` (`checkpoint_utils.py:549-558`).
- `supports_rename` is True because iopath is not installed in the w2vu env, so `IOPathManager` is None.
  Asynchronous writes are off by default.
- best and last are copies of that file, made after it is written.
- The pickle round trip works in a fresh interpreter. Per the worker's check (`worker.py:208`), the job is
  not runnable while the file is missing. `_available` resolves to `...config.w2vu2._checkpoint_file_exists`.
- Poll cost is 150 stat calls per 30 s loop, plus the usual finished-marker checks. There is no log noise
  at level 30.
- `CACHE_FINISHED_RESULTS` is off, and it caches only True in any case.
- The 1d chain does not depend on the intermediate jobs, so no deadlock is possible.

**(d) CPU.** `device` is set in `post_config` and is not hashed. No AMP or autocast is configured, and the
model casts to float32. CPU and GPU can therefore differ only by near-tie argmax flips: on JUPITER this was
3 of 177k dev-other errors (port B), far inside the G0.GAN band of +-0.03. All dev PERs now come from one
device. My unmeasured estimate is that a 4-cpu / 24 GB / 2 h job fits easily: a batch of at most 200
sequences is about 1-4 GB, and the compute is seconds.

**(e) Tests.** 12 of 12 pass (my own run). Only the counts and rqmt were changed; no assertion was weakened.

**(f) Restart.** The same command, `config/sae_i6_w2vu2.py`, is the only manager on that config. pid
1786381 had no child processes when I checked, so no local mini task would be orphaned. The setup file
still just calls `w2vu2.py()`. A restart is safe, but only after F1 and F2 are fixed. Otherwise the 26 stuck
points per seed never evaluate.

---

## Re-review, round 3 (2026-09-25): PASS

Scope: the round-3 uncommitted diff of `config/w2vu2.py` and `tests/test_w2vu2_config.py` against HEAD
(package code unchanged since bc68b9fb7), plus the coordinator's N_train question. F1 and F2 are resolved;
no new finding.

**F1 resolved (180 updates per epoch).** `w2vu2.py:243-246` derives
`N//160 + (N%160 >= 8) + (N%160%8 > 0)` and asserts 180. I ran fairseq 0.12.2's own
`data_utils.batch_by_size` (w2vu env; max_sentences 160, multiple 8, max_tokens None, random sizes) for
every N in 28,400-28,699: the formula matches fairseq for all 300 values. The count depends only on N.
The comment's validity claims are right: 180 holds for N 28,537-28,543, and for 160 values in
[28,489, 28,800] overall, with 28,536 and 28,544 giving 179. `gan_update_checkpoint_name(148000)` =
`checkpoint_823_148000.pt`, which is production s0's best. The GAN yaml's `dataset` section sets
`batch_size: 160` and leaves `required_batch_size_multiple` unset. No override in `w2vu2_gan_config`
touches `dataset.*`, and `update_freq` is not set, so its default of 1 applies. The graph asserts at
`w2vu2.py:320-324` therefore check the values fairseq actually uses.

**F2 resolved (epoch-end points).** `intermediate_eval_updates` moves 45000, 90000 and 135000 to 44000,
89000 and 134000. Each of these is 80 mod 180 and a multiple of `save_interval_updates` 1000, so fairseq
writes it (`checkpoint_utils.py:74-78`, not end_of_epoch). All 30 points per seed are non-multiples of 180.
`gan_update_checkpoint_name` asserts this condition, and the tests check it.

**Graph.** I rebuilt the graph in-process, without a manager, and got 851 jobs. All 101 HEAD ids are
present. The only changes to them are the 10 intended checkpoint_best dev forwards, which move from gpu
1/gpu_mem 40 to gpu 0 with device cpu. The graph adds 750 jobs: 150 conversions, 300 forwards on CPU
with gpu 0, and 300 PER jobs. Per seed there are 30 points, and the only ones off the 5000 grid are
44000/89000/134000. 650 of the round-1 intermediate ids changed because the names changed. None of them
was ever created on disk: there are no `analysis/w2vu2_gan_eval` or fairseq training job dirs. pytest
passes 13 of 13 (my run). The new test asserts the constant 180, production's name, the first
179/180-discriminating save (`checkpoint_39_7000.pt`), the update list, and that an epoch-end name raises.

**N_train on i6: 28,539. This is established from existing i6 artifacts, before the feature job runs.**
- The i6 VAD job `data/vad/BlankfreeVadHdfJob.RLrgIh6lFv9m` has finished. Its
  `output/counts_vs_expected.json` gives train utterances 28,539 against 28,539 expected, and original
  frames equal to the banked value. Kept frames are 15,427,887 against 15,427,853 banked. That is a
  report-only mode, but the utterance count is exact.
- I read the four `feats.train.shard*.hdf` directly: 28,539 sequences, minimum kept length 46 frames,
  0 below 3.
- `W2vu2FeatureDataJob.5Do5oigXsUKq` (`data/w2vu2_features.py:245-262`) drops only frames at or beyond
  `min(original, len(mfcc))`, then utterances under `min_length` 3. The original frame totals equal the
  banked ones, so the MFCC cut removes at most a frame or two per utterance. An utterance with 46 frames
  cannot fall below 3.
- fairseq then applies only `length >= 3` with max_length null (`extracted_features_dataset.py:65`). The
  task's and the model's `max_positions` are both None, so `get_batch_iterator` skips
  `filter_indices_by_size` (`fairseq_task.py:289`).

N_train therefore equals the line count of `train.lengths`, and the evidence says 28,539. Even if that
were wrong, only drop counts d = 3, 11, 19, ... or d >= 51 would change the per-epoch count. In that case
the intermediate points would wait forever, and no wrong checkpoint would be evaluated, because U is in
the file name.

**Graph-time assertion: not worth it.** A job-based check that gates the chain would add an input and a
new class argument. A guarded file read inside `gan_1c` is hash-neutral, but it runs only when the config
loads. The manager loads the config once, and the feature file does not exist yet, so the read would not
fire unless the manager is restarted after the feature job. Note also that `w2vu2.py:320` compares the
constant with the raw corpus count `EXPECTED_UTTS`, not with the filtered N. It encodes "zero drops"
rather than checking it.

**Manual checks before seed 0 reaches 7000 updates:**
1. When `work/i6_experiments/users/wu/experiments/unsupervised_asr/data/w2vu2_features/W2vu2FeatureDataJob.5Do5oigXsUKq`
   finishes, which happens before any GAN training can start:
   - `wc -l < output/data/train.lengths` must print 28539;
   - `output/data/train.stats.txt` must show `utts=28539` and `utts_dropped_short=0`.
2. After seed 0's first epoch, `log.run.1` of `i6_core/fairseq/training/FairseqHydraTrainingJob.gY9AdQ2aYl3B`
   (also `work/outputs/*/*/hydra_train.log`) must show:
   - `loaded 28539, skipped 0 samples` for the train split;
   - the train JSON line after `end of epoch 1` with `"train_num_updates": "180"`.

   The second item verifies the per-epoch count directly.
3. At 7000 updates, `output/checkpoints/checkpoint_39_7000.pt` must exist. With 179 updates per epoch the
   file would instead be `checkpoint_40_7000.pt`.
