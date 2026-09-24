# S3b-C consistency term -- code review (2026-09-15, commit 33ea43b)

Status: **PASS_WITH_CONCERNS**. The term is real, optimized, label-free and hash-neutral; the two
arms' resolved configs differ from the rate lam3 arm by the cons keys only. Three findings, one of
which touches every number these arms report.

## Findings

**F1 (must fix or disclose before launch). The augmented forwards update the recognizer's
BatchNorm running statistics.** `sae/emc/recognizer.py:127-129` builds `nn.BatchNorm1d(in_dim)`
(default `track_running_stats=True`, momentum 0.1) and `recognizer.py:180-185` applies it over the
valid frames of every forward. `train_steps/sae_emc.py:296-303` and `:317` call
`model.recognizer(...)` a second (spec) and third (speed) time per step, in train mode. So the
running mean/var that the EVAL-mode decode uses are estimated from 1/2 (spec) or 1/3 (spec_speed)
clean batches, the rest from SpecAugment-masked features (~19 % of frames and 28 % of channels set
to 0.0, `consistency.py:198,219`) and from speed-perturbed features. Every gate quantity is read
off those checkpoints (`DecodeStatsJob` PER, phone rate, distinct strings, the derangement gap), so
the arms differ from the rate lam3 baseline in a second respect beyond `+ lam_cons * L_cons`, and
the spec and spec_speed arms differ from each other in it too. It is also confounded with the term
itself: an arm that moves off blank may have moved because its BN statistics moved. Fix is local
(run the augmented passes with the BN module in eval / momentum 0, or with running-stat updates
disabled) and does not change the loss.

**F2. The length guard for the speed view is a tautology.** `consistency.py:280` returns
`tgt_lens`, i.e. exactly the `feat_lens` the caller passed in, so the
`torch.equal(warped_lens, feat_lens)` check at `train_steps/sae_emc.py:331-336` can never fire; it
is not a check that the perturbed row belongs to the clean row. Nothing else in the step compares
the two streams (`assert pert.shape[0] == b` at `:315` only checks the batch size). Pairing
therefore rests entirely on RETURNN's MetaDataset tag lookup (`returnn/datasets/meta.py:293-300,
455-460`: with no `seq_list_file` the control dataset's tags are handed to every other sub-dataset,
so the streams are matched BY TAG, not by index), plus the fact that the perturbed dump carries the
identical tag set -- which is established, `SpeedPerturbFrameCheckJob.oqcrbnKKCd2H`
(`output/frame_check.txt`): 28,539 utts, missing/extra = 0, mean T_pert/T_clean 1.1110 (speed0.9)
and 0.9088 (speed1.1), 0 utterances out of the +-2-frame tolerance, i.e. the dump covers the CV
holdout as well. A real guard would cost one line (`pert_lens/feat_lens` inside [0.85, 1.15]).
Risk is low, but the claimed safeguard does not exist.

**F3. The perturbed stream shrinks the spec_speed arm's batches.** `emc_train_jobs.py:764` sets
`batch_size = 88_000` and RETURNN's torch batching applies that limit PER DATA KEY
(`returnn/torch/data/pipeline.py:312`, `batch_size_if_included.any_compare(max_batch_size, >)`).
`features_pert` is ~11 % longer than `features` for the speed0.9 half, so frame-capped batches in
the spec_speed arm close ~10 % earlier: fewer utterances per step and more steps per sub-epoch than
the spec arm and than the rate arms. The utterance set per sub-epoch is unchanged (partition by
segments), so the gate reads are not biased, but "S3 schedule unchanged" holds only approximately
for that arm, and per-step monitors are not comparable across the two arms.

## Checks

1. **Resolved config diff -- DONE.** My own graph build (scratch script, not the implementer's):
   rate lam3 `DF6blPpto23t` vs cons spec `fw5O1L5qL2HL` differs in exactly
   `'lam_cons': 1.0, 'cons_views': ['specaug']` plus the job-id-derived `model` path -- nothing
   else. spec_speed `KsVX6rV1ic03` adds only `cons_views: ['specaug','speed']`, the
   `features_pert` extern_data entry, the `feats_pert` HDFDataset and its `data_map` entry on both
   train and dev. No `cons_specaug` key is written (the module default runs).
2. **lam_cons reaches the step and is optimized -- DONE.** `model_args['lam_cons']` ->
   `definitions/sae_emc.py:147,313` -> `train_steps/sae_emc.py:285` `getattr(model,"lam_cons")`;
   `ctx.mark_as_loss(cons_loss, "cons", scale=lam_cons, use_normalized_loss=False)` at `:339`,
   i.e. an optimized loss, not `as_error`. Monitors (`emc/cons_kl_*`, `emc/cons_frames`) are marked
   `as_error=True` and only when the term is on.
3. **Teacher / gradient -- DONE by reading.** `consistency.py:311` detaches the teacher;
   `log_p_clean` is the untempered `log_q` (temperature enters only the DP, `train_steps:203,206`);
   the student is a fresh recognizer forward, so theta gets gradient and phi appears nowhere in the
   term. Production dtype is fp32 (`feats_.raw_tensor.float()`), and the warp's simplex assertion
   runs at that dtype (`consistency.py:270-276`). Not re-run by me: the implementer's
   `test_consistency` numbers (theta grad 1.049, phi grad 0) were not independently executed.
4. **Speed pairing / warp direction / padding -- DONE, with F2.** Direction is right:
   `warp_posterior(log_p_pert, pert_lens, feat_lens)` maps the perturbed clock onto the clean one
   (src = perturbed, tgt = clean), align-corners with `lo`/`hi` clamped to `src_lens - 1`
   (`consistency.py:256-267`), so no padded source frame is read; the KL masks to the valid clean
   frames (`:308`). Both train and CV datasets get the stream (`config:171-172`), and the CV
   holdout is inside the dumped tag set (see F2). Tag pairing is by tag, not by order.
5. **Clean view genuinely clean -- DONE.** `specaugment` is out-of-place (`masked_fill`,
   `consistency.py:198,219`); `log_q` is computed from the untouched `feats` at
   `train_steps:176` before the term runs. Caveat: the clean pass is a train-mode pass (dropout
   0.1), so the teacher is "clean features, train-mode net", not an eval-mode teacher.
6. **No labels -- DONE.** `consistency.py` reads no gold; the perturbed dump carries
   `units_store=None`; the config's `gold_phones` reaches only the reads and the gap job, as in the
   rate arm; selection is `CV_LOSS_KEY = "dev_loss_l_tau"` (`emc_train_jobs.py:219`), i.e. the
   cons term does not enter checkpoint selection either.
7. **Hash census -- DONE, replicated independently.** I extracted the parent tree (`33ea43b^`) with
   `git archive` and ran my own census against both trees: `config_sae_4a_s3_v1` 98 jobs IDENTICAL,
   `config/sae_4a_phase.py` 1021 jobs IDENTICAL, `config_sae_4a_s3b_rate_v1` 219 jobs IDENTICAL
   with `WCh1fMyr88yu / DF6blPpto23t / axf11lrsap5i` unmoved. Cons graph: 160 jobs, arms
   `fw5O1L5qL2HL` (spec) and `KsVX6rV1ic03` (spec_speed) as reported.
8. **Caveat holds -- DONE.** Both arms pass `lam_rate = 3.0` (asserted against
   `rate.LAM_RATE_ARMS`, `config:116`) and inherit `lam_agg = 0.1` from the builder default, both
   visible in the resolved `model_args`. L_cons never runs alone in these two arms. Nothing in the
   code forbids a future `lam_cons`-only arm; the pre-registration is in the docstring only.
9. **The two undetermined items -- DONE.** (a) SpecAugment strength: it is a free choice, not a
   trace -- the numbers are RETURNN's `rf.audio.specaugment` opts at the final stage of its ramp
   (`consistency.py:46-67`), and SAE_4A.md fixes none of them. It does not change what the gate
   READS (PER, phone rate, gap are all decode-side), but it is the arm's main hyperparameter: if
   both arms fail, the failure is of this strength, not of consistency regularization. Record it
   with the gate. (b) `word_wer=False`: G4a.3b-C is defined as the G4a.3b-R clauses, none of which
   is WER, so no gate clause becomes unreadable; the brief's "reads unchanged" is satisfied by
   matching the rate config. No action.

## Not reached

* No smoke run and no re-run of `test_consistency` (13 tests) -- claims 3 and the warp round-trip
  errors are read from the code and the implementer's report, not re-executed.
* The nested alias `sae/4a/pert/feats_train_speedmix/frame_check` is created under an alias symlink
  that already points at the L15 job dir (`config_sae_4a_pert_dump_v1:290`, commit 72a30f8, out of
  this review's scope); cosmetic, the job's outputs are registered and readable.
