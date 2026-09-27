# S3b-C consistency regularization -- implementer report (2026-09-15)

Status: DONE_WITH_CONCERNS (the term, the container keys, the job wiring, the two arms and the
tests are in; two things the spec does not fix are named under "Undetermined" and one of them --
the SpecAugment strength -- is a real knob of the experiment).

Delta as dispatched: `L_cons = mean_t KL(p_clean(t) || p_aug(t))` on the recognizer's untempered
per-frame posterior, teacher = the clean view under stop-gradient, added to the existing objective
as `lam_cons * L_cons`; two selectable views (SpecAugment on the L15 features, and the
speed-perturbed L15 dump warped back onto the clean clock); everything OFF by default so no
existing hash moves; two arms at `lam_rate = 3`, `lam_cons = 1`, views {specaug} and
{specaug, speed}, on the S3b-R schedule/seed/reads.

## Files

New, in `recipe/2025-10-speech-llm` (branch `haotian_modality_matching_jupiter`):

* `src/speech_llm/sae/emc/consistency.py` -- the term itself, plain torch, no RETURNN import.
* `src/speech_llm/sae/emc/test_consistency.py` -- 13 tests (list below).
* `src/speech_llm/prefix_lm/sis_recipe/exp2025_11_06_speech_llms/librispeech/configs/config_sae_4a_s3b_cons_v1.py`
  -- the two arms.

Modified in the same checkout:

* `src/speech_llm/prefix_lm/model/train_steps/sae_emc.py` -- the consistency block + monitors.
* `src/speech_llm/prefix_lm/model/definitions/sae_emc.py` -- four container keys.
* `src/speech_llm/sae/emc/emc_train_jobs.py` -- builder keys, the second data stream, the second
  extern_data key, the omission rules.

Outside any git checkout (the setup dir is not a repo), reported as untracked exactly as the
rate-term round did:

* `config/sae_4a_s3b_cons.py` -- the sisyphus entry point (`sis m config/sae_4a_s3b_cons.py`).
* `scripts/sae_4a_cons_census.py` -- the hash census script (one argument: `s3 | phase | rate`).

## What the term computes

`consistency.py` public surface:

    VIEWS = ("specaug", "speed");  DEFAULT_VIEWS = ("specaug",)
    PERT_FEATURES_KEY = "features_pert";  SEED_BASE = 0x5AE4A2C
    SpecAugmentOpts(...)  / DEFAULT_SPECAUG
    step_generator(device, *, epoch, step, seed_base=SEED_BASE) -> torch.Generator
    specaugment(feats, lens, *, generator, opts=DEFAULT_SPECAUG) -> Tensor
    warp_posterior(log_p, src_lens, tgt_lens, *, tgt_max=None, sum_tol=1e-3) -> (log_p, lens)
    consistency_loss(*, log_p_clean, clean_lens, aug_log_p) -> (loss, ConsistencyStats)
    check_views(views) -> Tuple[str, ...]        # ValueError, not assert

Conventions pre-registered in the module docstring (the "pre-registration lives with the code"
rule), because SAE_4A.md fixes none of them:

* teacher = the CLEAN view, detached; the direction is KL(clean || aug) and is not symmetrized.
* the reduction is a mean over VALID CLEAN frames pooled over views, i.e. the two-view arm is not
  weighted twice as heavily as the one-view arm; with both views compared at the clean frame count
  this equals the mean of the per-view means (asserted in the test).
* every utterance enters, including those the lattice finds infeasible -- the term needs no
  lattice.
* the warp interpolates in PROBABILITY space (a convex combination of two rows of the simplex is
  on the simplex), align-corners mapping `x = i * (L_src - 1) / (L_tgt - 1)` clamped so no padded
  source frame is ever read, with a no-grad assertion that the valid warped rows sum to 1 within
  `sum_tol` at the production dtype (measured max |row sum - 1| = 2.4e-07 at fp32).
* the draw is seeded from `(SEED_BASE, epoch, step)`, so a resumed run repeats the masks of the
  step it resumes at instead of drawing from the global torch stream.

Train step (`train_steps/sae_emc.py`), only when `lam_cons != 0`:
specaug view = `model.recognizer(specaugment(feats, ...), feat_lens)`; speed view =
`model.recognizer(pert, pert_lens)` followed by `warp_posterior(..., tgt_max=T_clean)` and an
explicit `torch.equal(warped_lens, feat_lens)` check (a ValueError, not a silent broadcast).
The loss is marked as `mark_as_loss(..., "cons", scale=lam_cons)`; the monitors are
`emc/cons_kl_<view>` (one per view) and `emc/cons_frames`, added to the monitor loop only when the
term is on -- a scale-0 loss would still be REPORTED by RETURNN, which is why nothing is marked at
`lam_cons = 0`.

## Hash neutrality

Everything of this delta is hash-neutral BY OMISSION: at `lam_cons = 0` the builder writes none of
`lam_cons`, `cons_views`, `cons_specaug`, declares no `features_pert` extern_data key and adds no
`feats_pert` dataset, so `model_args` and the dataset dicts are bit-identical to what they were.
The `features_pert` key and the perturbed HDFs enter only with the `speed` view; the builder
refuses the two inconsistent combinations (hdfs without the view, `cons_specaug` without the term).

Census (`scripts/sae_4a_cons_census.py`, run BEFORE and AFTER the whole change, and once more after
the final docstring edit; outputs compared with `cmp`):

| graph | jobs | result |
|---|---|---|
| `config_sae_4a_s3_v1` (`s3`) | 98 | byte-identical |
| `config/sae_4a_phase.py` (`phase`) | 1021 | byte-identical |
| `config_sae_4a_s3b_rate_v1` (`rate`) | 219 | byte-identical |

The three S3b-R arms are unmoved: lam1 `ReturnnTrainingJob.WCh1fMyr88yu`, lam3
`ReturnnTrainingJob.DF6blPpto23t`, lam10 `ReturnnTrainingJob.axf11lrsap5i`.

## The two arms

`config_sae_4a_s3b_cons_v1.py` is the S3b-R graph with the term added and nothing else changed:
S3 temperature schedule `[8.0, 5.04, 3.17, 2.0, 2.0, 2.0, 2.0, 2.0]`, alpha = 0, `lam_agg = 0.1`,
the flat init `FlatRecognizerInitJob.21Kxgr5JLR3k`, 8 sub-epochs, keep all, the same per-sub-epoch
reads (PER, phone rate, distinct strings, derangement gap at epochs 4 and 8) and the same
label-free `UnsupervisedCheckpointSelectionJob`. `LAM_RATE = 3.0` is asserted to be one of
`config_sae_4a_s3b_rate_v1.LAM_RATE_ARMS`; `RATE_RHO_HZ` is imported from that config, not retyped.

    arm spec        (views ("specaug",))          i6_core/returnn/training/ReturnnTrainingJob.fw5O1L5qL2HL
    arm spec_speed  (views ("specaug","speed"))   i6_core/returnn/training/ReturnnTrainingJob.KsVX6rV1ic03

Graph: 160 jobs, containing the perturbed dump by import (`SpeedPerturbFrameCheckJob.oqcrbnKKCd2H`,
`AvStatesJob.pBRPNg7F2C8c`, `L15FeatureHdfJob.OTO33H7dWkfy` for the tc100 train split), so the two
graphs must not be managed at the same time.

The serialized RETURNN config of both arms was read back with `ReturnnConfig._serialize()`:

* `spec`: `model_args` carries `'lam_rate': 3.0, ..., 'lam_cons': 1.0, 'cons_views': ['specaug']`;
  zero occurrences of `features_pert`, `feats_pert` or `cons_specaug` anywhere in the config.
* `spec_speed`: the same with `'cons_views': ['specaug', 'speed']`, plus `extern_data`'s
  `'features_pert'`, the MetaDataset `data_map` entry
  `'features_pert': ('feats_pert', 'data')` and the `'feats_pert': {'class': 'HDFDataset', ...}`
  stream -- on BOTH the train and the CV dataset, because RETURNN runs the same `train_step` on the
  dev set.

## Tests (all green)

`python -m speech_llm.sae.emc.test_consistency` -- 13 PASS, "ALL consistency TESTS PASSED":

* KL = 0.0 when the views coincide, 0.7676 when they differ, padding excluded, and the two-view
  pooled mean 0.3838 equals the mean of the per-view means.
* gradient: the teacher leaf gets `None`, the student gets 0.8835 (the flow is through p_aug only).
* warp 0.9x (112 -> 101 frames) max |p - p_clean| = 3.69e-04, warp 1.1x (92 -> 101) = 5.54e-04,
  max |row sum - 1| = 2.4e-07 / 1.2e-07 at fp32; identity at equal lengths; a mixed batch warps to
  each utterance's own clock; a source length past the tensor is refused.
* specaug: T unchanged, padding untouched, reproducible from (epoch, step).
* specaug policy re-measured at the production shape (T = 634, D = 1024): 19.0 % of the frames and
  28.0 % of the channels, held in a band around the documented 18.7 % / 28.1 %.
* `lam_cons = 0`: a full `train_step` produces 15 losses/monitors bit-identical to the pre-change
  step, no `cons` key, and `consistency_loss` is never called (spied).
* `lam_cons = 1`, both views, through the real `train_step` at production dtype: cons 0.25632 =
  mean(specaug 0.12084, speed 0.39180) over 36 frame-view pairs, theta grad 1.049e+00, phi grad 0.
* the speed view refuses a job whose extern_data lacks `features_pert`; the container refuses an
  unknown view, an empty view set, a repeated view, a mistyped mask key and a mistyped `model_args`
  key; the builder's three literals equal the module's.

Also re-run unchanged: `test_sae_emc` (6 PASS), `test_rate_term` (15 PASS), `test_emc_train_jobs`
(all pass).

## Undetermined / assumptions

1. **The SpecAugment strength is not fixed by SAE_4A.md.** The defaults in `SpecAugmentOpts` mirror
   RETURNN's `rf.audio.specaugment` policy at the final stage of its `steps=(0, 1000, 2000)` ramp
   (the ramp itself is deliberately not reproduced: an S3b run is ~470 steps, so a ramped policy
   would spend the whole run at its weakest setting). Measured at the production shape that is
   18.7 % of the frames (14.6-22.9 % over 50 draws) and 28.1 % of the channels (22.3-34.7 %). This
   is the one number of this delta that materially changes what the experiment measures, and it is
   a choice, not a trace; it is reachable at config level through
   `emc_train_jobs.build_emc_train_config(cons_specaug=...)` without touching the module.
2. **WER.** The brief lists WER among the unchanged reads, but the reference rate config registers
   `word_wer=False` (no word decode per sub-epoch). I followed "reads unchanged from the rate
   config", i.e. this config registers no word decode either. If the S3b-C gate is to be read on
   WER, one flag flip in the config turns it on (and moves this config's hashes only).
3. Nothing was launched, and the perturbed dump's train HDF (`L15FeatureHdfJob.OTO33H7dWkfy`) is
   reached by IMPORT rather than by a frozen path, so the two configs must not be managed
   concurrently.

## Commit

`recipe/2025-10-speech-llm`, branch `haotian_modality_matching_jupiter`, explicit paths only
(6 files), no push, nothing launched: commit `33ea43b`.

---

# Round 2 (2026-09-15) -- the three review findings

Commit `1f05ca9` on the same branch, four files:
`sae/emc/consistency.py`, `sae/emc/test_consistency.py`,
`prefix_lm/model/train_steps/sae_emc.py`, `sae/emc/emc_train_jobs.py`.
Setup-dir `scripts/sae_4a_cons_census.py` (untracked) gained a `cons` mode and tolerates the
upstream `ACTIVE_ARMS` change.

## F1 -- the augmented passes no longer write the BatchNorm running statistics

`consistency.frozen_batch_norm_stats(module)` is a context manager that switches
`track_running_stats` off on every `_BatchNorm` of the recognizer for the duration of the augmented
forwards and restores it afterwards; the train step wraps the whole `specaug` + `speed` block in it.
Why this switch rather than `eval()` or `momentum = 0`: in train mode torch then hands
`F.batch_norm` `None` for the buffers, i.e. it still normalizes with the BATCH's own statistics --
the forward value is exactly what it was before the guard -- and writes nothing back, with
`num_batches_tracked` not incremented either (an `eval()` guard would have changed the augmented
pass's normalization, and `momentum = 0` would still have moved `num_batches_tracked`). The clean
teacher pass runs OUTSIDE the block and keeps feeding the buffers exactly as every stage before
S3b-C did, so the eval-mode decode every gate quantity is read from is normalized by clean data
only.

Tests: (a) on the REAL `ConvRecognizer` (dropout 0 so the two passes are comparable bit for bit) the
guarded forward equals the un-guarded one while `running_mean`/`running_var`/`num_batches_tracked`
stay bit-identical -- the un-guarded pass moves the mean by 1.428e-02, so the check is not vacuous
-- and the flag is restored; (b) through a real `train_step` with a BatchNorm-carrying stub
recognizer, the three buffers after a `lam_cons = 1` step with BOTH views are bit-identical to those
after a `lam_cons = 0` step, while the clean pass still moves them (`num_batches_tracked` 0 -> 1).

## F2 -- a real pairing guard for the speed view

`consistency.check_speed_lengths(pert_lens, clean_lens)` applies the dump's own rule per row,
before the recognizer runs on the perturbed stream:

    round(T_clean / max(SPEED_FACTORS)) - tol  <=  T_pert  <=  round(T_clean / min(SPEED_FACTORS)) + tol

with `SPEED_FACTORS = (0.9, 1.1)` (`sae/perturb.py MIXES["speedmix_v1"]`) and
`SPEED_LEN_TOL_FRAMES = 2`, the tolerance `SpeedPerturbFrameCheckJob.oqcrbnKKCd2H` itself checked at
(`rule: T_pert == round(T_clean / alpha)`, `tolerance = +-2 frames`; it measured max |dev| = 1 frame
over 28,539 utterances, 0 outside). Which of the two factors an utterance got is decided by its seq
tag inside the dump and is not visible in the step, hence the union band. The error names the
offending rows, their two lengths and the band. The old post-warp equality is kept as a plain
`assert` and is documented as the tautology it is. Tested at the production frame count (634 clean
accepts 576..704 at +-2, refuses 573 and 707) and on a swapped pairing (`[1000, 111]` against
`[100, 900]`).

## F3 -- the speed arm's batch is sized on the clean key only

RETURNN's torch batching applies an int `batch_size` to every data key it sees
(`torch/data/pipeline.py:306-313`, `NumbersDict.any_compare`), and `features_pert` is ~11 % longer
than `features` for the speed0.9 half, so the frame cap closed the `spec_speed` arm's batches
earlier. A dict `batch_size` has no broadcast value, so any key it does not name is unconstrained
(`util/basic.py:2182-2197`): the builder now writes `{"features": 88000}` when the speed view is on
and the plain int otherwise, i.e. the clean key -- the one the lattice's own budget is about --
decides the batch, and an arm without the second stream batches bit-identically to before.

Measured against RETURNN's real `BatchingIterDataPipe` (600 sequences, lengths U{200, 1400},
`max_seqs = 128`, 88,000 frames): 10 batches without the second stream, 10 with it under the per-key
limit (identical grouping), 11 under an int limit -- i.e. the fix removes a **+10.0 % steps per
sub-epoch** difference (60.0 -> 54.5 utterances per batch) that would otherwise have separated
`spec_speed` from `spec` and from the rate arms.

## Checks

* `test_consistency` 17 tests, "ALL consistency TESTS PASSED" (the 13 of round 1 plus the four
  above); `test_sae_emc` 6 PASS; `test_rate_term` 15 PASS; `test_emc_train_jobs` pass.
* Census: `config_sae_4a_s3_v1` 98 jobs and `config/sae_4a_phase.py` 1021 jobs byte-identical to the
  round-1 PRE-change census. The rate graph shrank upstream (`ACTIVE_ARMS = ("lam3",)`, commit
  8ff23fe): it now has 91 jobs, every one of which was already in the 219-id round-1 census -- no
  hash moved, and the launched arm is unmoved at `ReturnnTrainingJob.DF6blPpto23t` (its job dir is
  on disk). `lam1` / `lam10` are no longer built.
* S3b-C arms: `spec` keeps `ReturnnTrainingJob.fw5O1L5qL2HL`; `spec_speed` MOVED with its
  `batch_size` to `ReturnnTrainingJob.be8qKc7aRj16`. Neither the old (`KsVX6rV1ic03`) nor the new
  hash has a job directory, so nothing is orphaned or re-funded.
* Serialized configs re-read with `ReturnnConfig._serialize()`: `spec` has `lam_cons: 1.0`,
  `cons_views: ['specaug']`, `batch_size = 88000` and no `features_pert` / `feats_pert` anywhere;
  `spec_speed` has `cons_views: ['specaug', 'speed']`, `batch_size = {'features': 88000}`, the
  `features_pert` extern_data entry and the `feats_pert` stream on both train and dev.

## Notes

* The checkout moved under this round: `8ff23fe` (rate `ACTIVE_ARMS`), `5994bd9` / `3f5c805`
  (`sae/emc/pack_jobs.py`, four EMC arms per allocation). The S3b-C config registers its two arms as
  plain `ReturnnTrainingJob`s and is NOT packed; if the coordinator wants these two arms to share an
  allocation, that is a config-level change in `config_sae_4a_s3b_cons_v1.py` and was not part of
  this dispatch.
* Nothing was launched. `sae/emc/recognizer.py` and `sae/emc/content_term.py` are being edited by
  another implementer in the same checkout; neither was touched here, and only the four files above
  were staged.
