# Implementation report: the InfoMax arms of SAE_4A_infomax.md (node_d)

Date 2026-09-20. Implementer round. Spec: `SAE_4A_infomax.md` "Design" plus the design-review
amendments of 2026-09-20 (entropy units and normalization, the ent schedule, the statswap floors,
the per-view consistency diagnostics). Arm shape from `SAE_4A_budget.md` lines 50-111.

## What runs

`PackedBlankfreeTrainJob.lQXR4mpLqVPC` -- node_d, four arms on one 4-GPU node,
`work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.lQXR4mpLqVPC`
(not on disk yet: no manager was started, as the brief requires).
rqmt `{'gpu': 4, 'cpu': 64, 'mem': 256.0, 'time': 11.5, 'gpu_mem': 96}`, `N = 50` sub-epochs,
bed `ctrl`, kept checkpoints `(1, 4, 10, 25, 50)` -- all four taken from
`config_sae_4a_budget_v1` (`TIME_RQMT`, `KEEP_EPOCHS[50]`, `_schedules("ctrl", 50)`), so the arms
sit at the budget round's operating point and differ from it only in the InfoMax keys.

Arms: `ent_50`, `enthold_50`, `entaug_50`, `entaughold_50` (the 2 x 2: decayed vs held entropy
weight x consistency off vs on).

The baseline, `ctrl_50` on node_a, is rebuilt inside the config with the non-registering idiom
(`tk.register_output` temporarily replaced by a no-op while `config_sae_4a_budget_pack_v1.build`
runs) and the resulting job id is asserted to end in `PackedBlankfreeTrainJob.ks7CbtlvpcIL` -- the
id of the running node_a. Nothing of the budget round is re-registered and no budget file was
edited.

## The terms

`speech_llm/sae/emc/infomax.py` (new, 334 lines) holds all the new logic; the modules the three
running nodes re-import at their resume were touched only where the brief allows (see below).

* `frame_entropy_loss(log_q, feat_lens, retained)` -- H_t = -sum_c q_t(c) ln q_t(c) on the CLEAN,
  untempered `log_q` the lattice just consumed. NO division by ln C. Per utterance the sum of H_t
  over its valid OUTPUT frames is divided by that utterance's RETAINED unit-frame count
  (`lengths`, as `sae_blankfree.py:88` does for `cycle`), then averaged over utterances -- so the
  term is in the same units as `l_tau` (nats per retained unit frame; at stride 3 about H/3 per
  unit frame). Padded frames enter neither the loss nor its gradient (asserted in the test).
  Monitors: `blankfree_entropy_per_frame` (nats per valid OUTPUT frame, pooled, detached) and
  `blankfree_lambda_ent`, both `as_error`.
* `statswap_view(feats, lengths, generator)` -- the new augmented view: each utterance is
  standardized with its own valid-frame mean/std and re-scaled with a partner's. The partner map is
  a random CYCLE (a derangement; the code asserts no fixed point), the std is floored at
  `STATSWAP_STD_EPS = 1e-3` before dividing, an utterance is swapped only when it and its partner
  both have at least `STATSWAP_MIN_FRAMES = 200` valid frames (4 s at 50 Hz) and otherwise keeps
  its own features, padding is untouched and the result is asserted finite. A batch of one passes
  through unchanged.
* `infomax_terms(model, feats, lengths, log_q_clean, feat_lens, epoch, *, step=0)` -- returns the
  named terms to mark: `ent` at this sub-epoch's `lambda_ent`
  (`sae_emc.schedule_value`: 1-based, last entry held), and at `lam_cons != 0` the registered
  S3b-C `cons` plus `emc/cons_kl_specaug`, `emc/cons_kl_statswap`, `emc/cons_frames` as `as_error`.
  The augmented forwards run inside `consistency.frozen_batch_norm_stats(model.recognizer)` with
  `consistency.step_generator(device, epoch, step)`; the specaug view is
  `consistency.specaugment(..., opts=consistency.DEFAULT_SPECAUG)`.
* `check_views` -- `ValueError` (not `assert`) on an empty, unknown or duplicated view, at job
  start, over `VIEWS = ("specaug", "statswap")`.

`consistency.consistency_loss(log_p_clean=, clean_lens=, aug_log_p=)` is called unchanged and
returns `(loss, ConsistencyStats)`: `loss` is the MEAN OVER VIEWS of each view's mean
`KL(p_clean || p_aug)` over the valid clean frames (`total / (frames * n_views)`), the teacher
`log_p_clean` is detached inside it, `stats.per_view[name]` is that view's own per-frame mean KL
and `stats.frames` is valid clean frames x n_views. It consumes augmented LOG POSTERIORS, which is
what `infomax_terms` hands it.

### Signature deviation

`infomax_terms` carries one keyword-only parameter the spec did not list, `step` (default 0),
because `consistency.step_generator` seeds from `(epoch, step)`; without it every step of a
sub-epoch would draw the same SpecAugment masks and the same derangement. The train step passes
`_scalar_int(ctx.step)`.

## The schedules (as amended 2026-09-20)

`ent_50` / `entaug_50`: `lambda_ent = 0.1` held over sub-epochs 1..10 (the tau 8 -> 2 anneal
window), geometric 0.1 -> 0.001 over 11..20 with entries 11 and 20 written exactly, then 0.0 for
21..50. Measured first entries `[0.1, 0.1, 0.05995] ... [0.00167, 0.001, 0.0, 0.0]`.
`enthold_50` / `entaughold_50`: `[0.1] * 50`.
`entaug_50` / `entaughold_50` additionally carry `lam_cons = 1.0` and
`cons_views = ("specaug", "statswap")`.

## The frozen modules

* `prefix_lm/model/definitions/sae_blankfree.py`: `__init__` gains `ent_schedule=None`,
  `lam_cons=0.0`, `cons_views=()` and, after `super().__init__`, a guarded block that stores them.
  At the defaults it adds no parameter, no buffer and no forward path and does not import
  `infomax`, so every banked checkpoint and every running arm is bit for bit what it was. The
  `lam_cons` / `cons_views` OVERWRITE is deliberate: the base `SaeEmcModelV1` already sets these two
  but validates them with `consistency.check_views`, whose registered set is
  `("specaug", "speed")` and would reject `"statswap"`; `consistency.py` is frozen by the brief, so
  the blank-free model re-validates with `infomax.check_views` after the base has run.
  `ent_schedule`, when stated, must have one entry per sub-epoch (asserted against
  `temperature_schedule`).
* `prefix_lm/model/train_steps/sae_blankfree.py`: ONE guarded block, after the `rate` mark and
  before the monitor block, that imports `infomax_terms` inside the guard (the BT-guard idiom) and
  marks whatever it returns with `use_normalized_loss=False`. The guard reads the two attributes
  through `getattr`, so a model built before they existed does not enter it.
* No other module of the brief's frozen list was touched. `settings.py` untouched.

## Tests

`speech_llm/sae/emc/test_infomax.py` (new, 385 lines), run from the setup dir with the
`speech_llm` conda env on PATH (`black` must be reachable or `ReturnnConfig.write` dies):

```
WS=$PWD; PATH=/e/project1/spell/wu24/env/conda/envs/speech_llm/bin:$PATH \
PYTHONPATH=$WS/tools/sisyphus:$WS/recipe:$WS/recipe/i6_models:$WS/recipe/returnn:$WS/recipe/2025-10-speech-llm/src \
/e/project1/spell/wu24/env/conda/envs/speech_llm/bin/python -m speech_llm.sae.emc.test_infomax
```

All pass (exit 0):

* `frame_entropy_loss` against a hand-computed 2 x 3 x 40 posterior of closed-form rows, in BOTH
  normalizations (loss 0.639046 per unit frame, monitor 1.752811 nats per output frame), with the
  padded frame proved to enter neither value nor gradient;
* statswap: every utterance wears its partner's mean and std to 1e-4 with padding intact; a batch
  of one is unchanged; an utterance below 200 frames AND the utterance whose partner it is keep
  their own features; partner = self reproduces the input to 1e-5 (max deviation 1.91e-06);
* the REAL blank-free train step on the REAL model (the `test_blankfree_permute` fixture): at
  `ent_schedule = None` / `lam_cons = 0` the marked losses are exactly the previous 33 keys, value
  for value (`l_tau` 1.517274); with the arms' keys on it is that set plus exactly the InfoMax
  names -- `ent` 0.077114 at `lambda` 0.100 (0.262 nats per output frame) and `cons` 11.003740 =
  (specaug 18.011061 + statswap 3.996420) / 2 over 14 frames, which is the mean-over-views
  convention above;
* the four written RETURNN configs against the budget round's `ctrl_50`: each differs, and after
  popping exactly the InfoMax keys out of `get_model`'s hashed arguments each is byte-identical to
  `ctrl_50`'s written config (`black` reformats the dict, so a line-level diff rule was replaced by
  this pop-and-compare rule); plus the schedules, `LAM_CONS == 1.0`, the view tuple, the rqmt, the
  kept-checkpoint set, `node_d._sis_id() != node_a._sis_id()` and node_a's id.

Graph load (sisyphus venv, script mode, no manager):
`sis --config config/sae_4a_infomax_pack.py console --script -c ...` -> exit 0, 265 jobs, 436
targets, and exactly two `PackedBlankfreeTrainJob`s in the graph: the new `lQXR4mpLqVPC` and the
running `ks7CbtlvpcIL`. On disk the packed-job dirs are still the three running nodes
(`ks7CbtlvpcIL`, `4QzmftNlbErt`, `reEI2Nd0S77A`); no dir and no alias was created by the load.

## Reads registered

Per arm, per kept checkpoint (1, 4, 10, 25, 50), on both dev splits, named
`infomax_pack/<arm>/ep<epoch>/<split>`, plus the paired contrasts
(`PairedPerDeltaJob`, `per_a` = baseline, `per_b` = candidate, so a NEGATIVE delta means the
candidate is better): each of the four arms vs `ctrl_50`, `entaug_50` vs `ent_50`,
`entaughold_50` vs `enthold_50`, and `enthold_50` vs `ent_50`.

## Not done here

No manager was started and nothing was launched; node_d is graph-only. Nothing was pushed.
