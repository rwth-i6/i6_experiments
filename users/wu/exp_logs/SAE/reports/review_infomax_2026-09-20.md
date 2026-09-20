# Code review: the InfoMax arms (commit 35bcf39, node_d) -- 2026-09-20

Verdict **PASS_WITH_CONCERNS**. The change is inert at the defaults (running nodes are safe), the
four configs are ctrl_50 plus the three InfoMax keys and nothing else, and every mechanism the
spec names is implemented as written. ONE sizing finding blocks the two aug arms: at `lam_cons`
= 1.0 the consistency term's gradient is ~6x the WHOLE current objective's, measured on the
running control's own checkpoint and real features.

## 1. Running-node safety -- PASS

* No top-level import of `infomax` anywhere in the two re-imported modules. Verified by importing
  `definitions/sae_blankfree.py` and `train_steps/sae_blankfree.py` fresh:
  `"speech_llm.sae.emc.infomax" in sys.modules` is **False** after import AND after constructing a
  DEFAULT model with the running arm's own get_model arguments; **True** only when `lam_cons` is
  stated. Both guards are the BT-guard idiom (`definitions/sae_blankfree.py:297`,
  `train_steps/sae_blankfree.py:122-123`).
* state_dict: the RUNNING ctrl_50 checkpoint
  (`PackedBlankfreeTrainJob.ks7CbtlvpcIL/output/ctrl_50/models/epoch.010.pt`) loads
  `strict=True` into the POST-commit default model: *All keys matched successfully*. Default vs
  aug-arm key list identical -> no parameter, no buffer.
* Train step at defaults: guard is `ent_schedule is None and lam_cons == 0` -> block skipped, no
  mark. The implementer's `test_the_step_is_unchanged_at_the_defaults` re-runs the real step and
  gets the same 33 keys value for value; re-run here, exit 0.
* The base's `lam_cons` handling (`definitions/sae_emc.py:419-426`) is reached at the base's own
  default 0.0 (the subclass intercepts the kwarg and does not forward it), so
  `consistency.check_views` never runs and the base does nothing conditional. NOTE, harmless: the
  base's default `cons_views` is `("specaug",)`, not `()`, so an ent-only arm carries
  `self.cons_views = ("specaug",)` with `lam_cons = 0`; nothing reads it at 0.
* The base train step (`train_steps/sae_emc.py`) is NOT the step these arms run; the blank-free
  step is, and it is the only file that marks the new terms.

## 2. Magnitude of `cons` -- CONCERN (blocking for entaug_50 / entaughold_50)

Measured on the real bed: VAD-masked L15 train features (`BlankfreeVadHdfJob.SAjz8y1cT06g`), the
RUNNING ctrl_50 recognizer at sub-epoch 10, tau = 2, train mode, 8 utterances, full model
(prior.npz + eta.npz of the arm's own config), reproducing the step's own terms.

| term | value | `|grad|` over theta |
|---|---|---|
| l_tau (lam_tau 1) | 1.985 / unit frame | 7.92 |
| 0.1 * agg | 0.904 | 0.11 |
| 3.0 * rate | 0.0168 (9.29 Hz) | 5.64 |
| **whole current objective** | | **9.71** (9.71 over ALL params: phi's share is negligible) |
| 0.1 * ent | H 0.303 nats/output frame, ent 0.101 / unit frame | 0.98 (**10 %** -- correctly sized) |
| **1.0 * cons** | **0.852** nats/output frame (specaug 1.411, statswap 0.293) | **61.8 (6.4x the whole objective)** |

Per view at weight 1.0: specaug-only 106.5, statswap-only 55.2. Reproduced at B = 4 and B = 12
(cons/l_tau 6.8 and 8.5), and on dev-other at ep1/4/10 (cons 0.07 / 1.74 / 0.91, specaug always
the large one: 0.14 / 3.43 / 1.79 against statswap 0.002 / 0.05 / 0.03).

The config runs `gradient_clip_global_norm = 5.0`, and the current objective already exceeds it
(9.71). Adding cons at 1.0 makes ~86 % of the pre-clip gradient the invariance direction, i.e.
after clipping the lattice + rate signal is attenuated ~7x. The two aug arms would not be
"ctrl_50 + entropy + invariance": they would be invariance training with a throttled objective,
and the pre-registered `entaug_50 vs ent_50` contrast would not isolate the invariance effect.
S3b-C's own low-inventory collapse (SAE_4A.md:984, phase amendment 5) happened at lam_cons = 1.

**Recommended lam_cons 0.03** (band 0.016-0.047 for 10-30 % of the whole objective's gradient;
0.013-0.038 for 10-30 % of l_tau alone). The later S3b-C trigram value 0.3
(`config_sae_4a_s3b_rate_v1.LAM_CONS_TRI`) is still ~10x too large on this bed.

Two contributing causes, both real:
* SpecAugment is the large view, and it is the `DEFAULT_SPECAUG` channel rule `D // 5 = 204`
  channels x 2-5 masks (~28 % of the 1024 wav2vec2 channels) plus ~19 % of frames
  (`consistency.py:58-79`). It is reachable per arm without touching `consistency.py`.
* UNIT MISMATCH: `ent` is normalized per RETAINED UNIT frame (l_tau's units, as the spec asks),
  but `consistency_loss` normalizes per OUTPUT frame. At stride 3 a nominal `lam_cons` is
  therefore ~3x heavier than the same number in l_tau's units. Not a bug, but `lam_cons` is not
  in the units the design's sizing paragraph is written in.

## 3. statswap -- PASS

`_derangement` (`infomax.py:165-178`) draws one random cycle (`partner[order] = order.roll(-1)`),
so no fixed point for B >= 2, asserted. B = 1 returns the input. The >= 200-frame rule is
`long_enough & long_enough[partner]` (both sides), stats are taken under `no_grad` on valid frames
only, sd floored at 1e-3, padding copied through, `isfinite` asserted. Real-data checks: the floor
never binds (min per-utterance per-dimension std over 30 utts = 4.33, ~4000x the floor); only
6.05 % of the 28,539 train utterances are below 200 frames, so ~88 % of rows are swapped (with
`laplace:.1000` ordering the short ones cluster into the same batches, where the view degenerates
to the identity -- acceptable, worth knowing when reading `emc/cons_kl_statswap`).
Determinism: the generator is `consistency.step_generator(device, epoch, step)`, a pure function of
(epoch, step); the specaug draw precedes it but is itself deterministic given the batch, so a
resumed step reproduces. `torch.randperm(n, generator=g, device=cuda)` honours a CUDA generator in
torch 2.7 (`aten/src/ATen/native/cuda/Randperm.cu`, `get_generator_or_default<CUDAGeneratorImpl>`),
no CPU fallback.

## 4. Gradient routing -- PASS

`ent` is computed on the CLEAN `log_q` the lattice consumed -> theta only (phi does not appear).
`cons` detaches the teacher inside `consistency.consistency_loss:415` and the student is
`model.recognizer(aug, lengths)` -> theta only. Both augmented forwards sit inside
`consistency.frozen_batch_norm_stats(model.recognizer)` -- the same module object the clean pass
ran (`train_steps/sae_blankfree.py:59`) -- and the recognizer has exactly one BatchNorm1d with
`track_running_stats = True`, which the guard flips off and restores. `feats` does not require
grad, so neither view builds an autograd graph for the transform itself.

## 5. Schedules -- PASS

`ent_decay_schedule()` printed from the config module: e1 0.1, e10 0.1, e11 0.1, e20 0.001,
e21 0.0, e50 0.0, 50 entries, geometric between 11 and 20 (constant ratio to 1e-12).
`ent_hold_schedule()` = [0.1] x 50. Read through the same `sae_emc.schedule_value` the tau anneal
uses: (1, 10, 11, 20, 21, 50) -> (0.1, 0.1, 0.1, 0.001, 0.0, 0.0). `_model_args_delta`: ent_50
decay, enthold_50 hold, entaug_50 decay + lam_cons 1.0 + ("specaug","statswap"), entaughold_50
hold + the same. lam_cons only on the two aug arms. The length of `ent_schedule` is asserted
against `temperature_schedule` at construction (a 10-entry list is refused).

## 6. Config identity and the reads -- PASS

`test_the_arm_configs_differ_from_ctrl_only_in_the_infomax_arguments` takes the baseline from
`config_sae_4a_budget_pack_v1.build("node_a")` via `INFOMAX._baseline_node()` (the real builder,
not a local copy) and diffs the WRITTEN configs; re-run here, exit 0, all tests pass. The baseline
resolves to `PackedBlankfreeTrainJob.ks7CbtlvpcIL`, which is the job dir
`alias/sae/4a/budget_pack/node_a/training` points at on disk; node_d is the new
`PackedBlankfreeTrainJob.lQXR4mpLqVPC`, rqmt `{'gpu': 4, 'time': 11.5, ...}` = budget's TIME_RQMT.
Kept checkpoints (1, 4, 10, 25, 50) per arm, asserted against the pack's offer. Paired reads:
seven contrasts x 5 kept epochs x 2 dev splits, each pairing the SAME epoch on both sides,
`per_a` = baseline. `_register_epoch_reads` is byte-identical to the budget pack's except the
`name=` stamp (`infomax_pack/` instead of `budget_pack/`).

## 7. Second differences -- none found

The commit touches 5 files, none of them `consistency.py`, `blankfree_train_jobs.py`,
`blankfree_pack_jobs.py` or `settings.py`. `_arm_train_config` omits `**bt_kwargs`, which is `{}`
for the ctrl kind, and the byte-identity test covers the result. No seed, batching, dropout or
rqmt change: the aug pass reuses the arm's own dropout 0.1 (the registered S3b-C convention, so
the KL carries a dropout component -- measured identity-pass KL 0.087 in train mode, ~10 % of
cons, 0.000 in eval mode). Compute cost of the two extra forwards is negligible against the
10.6 s/step float64 lattice.

## Outside the diff, but it changes a pre-registered read

The phase's early read declares symmetry breaking when the arm's eval-mode entropy is below
1.0 nat "while ctrl_50's at the same checkpoint is not". Measured on ctrl_50's own checkpoints,
dev-other, eval mode, 16 utterances: **ep1 3.25, ep4 2.71, ep10 0.30, ep21 0.19** nats per output
frame. The control is already at 0.30 nats at sub-epoch 10, so that clause cannot fire as written
and the ep10 read would be uninformative. The control is CONFIDENT and content-free, which also
weakens the premise that the band is a flat-posterior stationary point; the entropy penalty is
being added to a recognizer whose entropy is already 8 % of ln 40. The gate should be restated
against these numbers before the ep10 read is taken.

## What was run

`/e/project1/spell/wu24/env/conda/envs/speech_llm/bin/python -m speech_llm.sae.emc.test_infomax`
from the setup dir (exit 0, all 9 tests), and scratch measurement scripts under
`/tmp/claude-34349/.../scratchpad/` (`meas2.py` recognizer-only entropy/KL per checkpoint,
`meas3.py` full model l_tau/agg/rate/ent/cons gradients, `importsafe.py` import and state_dict
checks, `misc.py` length and std statistics). CPU only, no GPU, nothing launched, nothing edited.
