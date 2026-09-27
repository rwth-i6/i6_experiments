# Code review: S3b-BT-aux (commits d40f708, 325e47c) -- 2026-09-15

Verdict **PASS_WITH_CONCERNS**. Read-only review of `recipe/2025-10-speech-llm` @ 325e47c against
baseline fb97abb, the spec `SAE_4A.md` "S3b-BT-aux" (line 1005ff) and the RETURNN checkout at
`/e/project1/spell/wu24/2026-07-13_unsupervised/returnn` (00171dfe). Nothing launched, nothing
edited. Review ended early on the coordinator's instruction; two items below are explicitly
NOT CHECKED.

## The 11 checks

**1. Separate optimizer step -- VERIFIED.** `returnn/torch/data/pipeline.py:421-457`: a callable
`torch_batching` is called as `cls(dataset, train=..., **fwd_compat)` and batch_size/max_seqs are
NOT passed, so the `PartialImport` carrying them (`emc_train_jobs.py:610-622`) is the right
mechanism and cannot disagree with the config dict. `returnn/torch/engine.py:494-515`: one backward
+ one `optimizer.step()` per batch at `accum_grad_multiple_step: 1` (written at
`emc_train_jobs.py:1040`), so one marker batch == one BT optimizer step; the EMC and BT graphs never
share a run context (`train_step` returns right after `bt_train_step`, `emc_train_jobs.py:422-427`).
Marker completeness: `bt_marker_seq` (`bt_aux.py:596-615`) writes a length-1 zero array for EVERY
key of the preceding seq dict and copies `seq_idx/epoch/num_seqs/complete_frac`, which is what
`extern_data.py:53ff` (every template key required) and `engine.py:453-455` read. No marker while
the ramped weight is 0 (`bt_aux.py:641`), so the scale-0 drop at `frontend/run_ctx.py:416-425`
(confirmed) is never reached with an empty optimized total.

**2. No labels -- VERIFIED.** `bt_text_path` resolves in the generated config to
`work/.../TextToPhonemeJob.THKMON3k9LJQ/output/phon.txt.gz` (file present, 1.59 GB). The BT path
reads only that corpus, the live phi, and the EMC batches' own features/units. No transcript,
alignment or gold artifact appears anywhere in `bt_aux.py` or in the arm wiring.

**3. Argmax units, sampled nuisance, phi ungradiented -- VERIFIED by mechanism.**
`build_bt_batch` is `@torch.no_grad()` (`bt_aux.py:422`), `reverse.sample` gains `unit_draw`
(insertion-only, default `"sample"`; argmax at `reverse.py:456-459`) while durations and the eta row
stay sampled (`bt_aux.py:444-449`). Everything theta sees is created inside that no-grad block, so
phi's parameters cannot receive gradient; the test's `grad is None` is a consequence, not the
evidence.

**4. Frame pool -- VERIFIED with a CONCERN.** Filled after each EMC step from that batch's own
features and its frozen-quantizer unit stream (`emc_train_jobs.py:441-455`), which the EMC step
already asserts is frame-aligned with the features. Ring buffer, plain attribute, never a
state-dict key (`bt_aux.py:280-391`), 500x64x1024 fp16 = 65 MB. A unit with no frame falls back to
the pool mean (`bt_aux.py:377-378`).
CONCERN: `real_frac` -- the share of synthetic frames that came from a REAL frame of their unit --
is computed (`bt_aux.py:459`, returned at `:517`) and then **never marked as a monitor**
(`bt_aux.py:565-570` has only the four keys). The pool is process-local and empty after any
resubmit, so the first BT steps of a restarted run collage the mean frame for most units and
nothing in the log distinguishes that from a healthy step; the disclosed deviation from the probe's
fixed pool is exactly the thing that cannot be read back. Recommend `emc/bt_real_frac` before the
launch.

**5. Loss, frozen BN, ramp -- VERIFIED except the ramp table.** CTC blank 0, target id+1,
`zero_infinity`, normalized `per_utt.sum()/out_lens.sum()` (`bt_aux.py:503-512`); forward inside
`consistency.frozen_batch_norm_stats` (`bt_aux.py:494`), which flips `track_running_stats` and not
`eval()`, i.e. the same mechanism the consistency term uses. Ramp folded into the VALUE at
`scale=1.0` (`bt_aux.py:562`), epoch from `int(rf.get_run_ctx().epoch)` (`bt_aux.py:533`), same as
the entropy term.
NOT RESOLVED: `test_bt_aux` prints the ramp as `0.0000, 0.0750, 0.2250, 0.3000, 0.3000` at
lam_bt 0.3, while lam_bt x (e-1)/4 gives .075 at e2, .15 at e3, .225 at e4. Most likely the test
prints a non-consecutive epoch list (1,2,4,5,6); one read of `entropy_term.lam_ent_effective`
settles it and should be done before launch.

**6. Depth attenuation -- VERIFIED.** `out_logit_layer` is a property returning `self.conv`
(`recognizer.py:216`) and the forward calls it as a module (`recognizer.py:284`, `x = self.conv(x)`),
so the forward pre-hook at `bt_aux.py:492` does fire; its input is the transposed content tap
(`pre_conv`, the conv's input after BN + in_proj + dropout + mask), so under `output_only` only the
final conv gets BT gradient. `"full"` installs no hook at all and the recognizer forward is the EMC
one. `output_only` at `n_layers != 1` is refused at job start (`definitions/sae_emc.py:463-472`).

**7. Hash neutrality and control -- VERIFIED by re-running the census.** Before = `git archive
fb97abb`, after = the working tree, `scripts/sae_4a_cons_census.py`:
`s3` and `phase` diff to NOTHING (99 / 1022 output lines); `rate` differs only by three
`# arm bt_*: not built` comment lines, job ids 91 -> 91, `ReturnnTrainingJob.DF6blPpto23t` present
and unmoved. Rebuilding `config/sae_4a_s3b_pack.py` gives `PackedEmcTrainJob.SeZzGUScxq4x` unmoved
(285 jobs). A scratch tree with `ACTIVE_ARMS = ("lam3","bt_a","bt_b","bt_c")` reproduces the
reported hashes exactly -- bt_a `6G8CpcswqlTR`, bt_b `NSxRSjyBWp5g`, bt_c `UjJpp4iQ4kkT` -- with
lam3 still `DF6blPpto23t`, 192 ids added and 0 removed. The `lam_bt = 0` bit-identity test (18
losses/monitors) passes. CV pipeline: the config defines no `batch_size_dev` / `batch_size_train`,
so the `train=False` branch (`bt_aux.py:664-665`) constructs exactly the `BatchingIterDataPipe`
RETURNN's default path would (`pipeline.py:429-436`) -- the CV pass, and therefore the selection
quantity, is untouched.
(The census numbers in d40f708's message, 99/1022/98, are the same runs counted with the comment
header lines; no discrepancy.)

**8. Monitors -- PARTIAL.** All four required keys are marked, `as_error=True`, named with the
project's `key.replace("/", "_")` convention (`bt_aux.py:571-575`), matching
`train_steps/sae_emc.py:520-530`. But `SAE_4A.md:1017` pre-registers `emc/bt_units_per_s` and the
code emits `emc/bt_synth_phones_per_s`; the gate line in the plan should be amended so the read
does not look for a column that does not exist. `real_frac` missing, see check 4.

**9. Held arms -- VERIFIED by diffing the generated RETURNN configs.** `bt_a` vs `lam3` differ in
exactly: the five `model_args` keys, the `bt_interleaved_batching` partial
(`batch_size 88000, max_seqs 128, lam_bt 0.1, bt_ramp_epochs 4` -- the config's own values),
`torch_batching = bt_interleaved_batching`, and the arm's own model output path. `bt_c` vs `bt_b`
differ in `bt_depth` alone. `ACTIVE_ARMS` is `("lam3",)` in the live tree.

**10. Live re-import safety -- VERIFIED.** The five defaults in `definitions/sae_emc.py:80-86` are
literals; `bt_aux` is imported only inside the `lam_bt != 0` guard in `train_step`
(`emc_train_jobs.py:422-427`, `:438-455`), and the marker-tag literal is duplicated in
`emc_train_jobs.py:216` so the test needs no import. No new parameter or buffer enters the model, so
a running job's checkpoint and state dict are unaffected; `train_steps/sae_emc.py` itself is
untouched by both commits.

**11. Other differences / cost.**
* NOT CHECKED: `engine.py:527` accumulates `losses_dict` into a `NumbersDict` across steps, and with
  the auxiliary on the loss-key set ALTERNATES (EMC keys vs `bt` + four BT monitors). I read
  `NumbersDict.keys_union` but did not finish `__add__`'s disjoint-key behaviour. If it raises, it
  raises at the first BT step of sub-epoch 2; if it broadcasts, the per-key epoch means are still
  each divided by their own accumulated inv-norm factor and stay correct. One targeted read closes
  this.
* Cost for the pack rqmt: the BT step pays NO lattice DP, and the DP is 91 % of the 1.67 s EMC step
  at this shape (`analysis/out/emc_profile_step.b128.txt`). A BT batch is 128 synthetic utterances
  of ~570 frames mean (20-200 phones at ~5.2 frames/phone), i.e. roughly the EMC batch's frame
  count, so the BT step is about one recognizer forward/backward plus phi's emission gather:
  expect 0.15-0.4 s, i.e. +10-25 % wall time, not +100 %. Two one-off costs to budget: the ~95 s
  T_phi reservoir read inside the FIRST BT step of each process, and a padded BT batch (uniform
  20-200 phone draw, so ~1.8x padding waste -- the EMC batches are laplace-ordered, these are not).
* All three arms inherit `prior_order: 2` (bigram) from the lam3 control, visible in the generated
  config. The standing rule is trigram-or-better in every new EMC stage; matching the control
  forbids changing it here. Coordinator's call, not a code defect.

## Checks run
`speech_llm.sae.emc.test_bt_aux` 7/7 PASS; `speech_llm.sae.emc.test_emc_train_jobs` all pass
(incl. the BT hash-neutrality test); `speech_llm.prefix_lm.model.train_steps.test_sae_emc` 11/11
PASS (incl. `lam_bt = 0` 18-loss bit-identity and `lam_bt = 0.3` separate-step). Census and scratch
build as in check 7.
