# S3b-BT-aux round 2: closing the code review's three open items

Date 2026-09-16 (round dispatched 2026-09-15). Setup `/e/project1/spell/wu24/2026-07-13_unsupervised`,
checkout `recipe/2025-10-speech-llm` on `haotian_modality_matching_jupiter`, base `806ea11`.
Commit **`9c7c1af`** (three files, staged by explicit path, NOT pushed). Nothing launched;
`ACTIVE_ARMS` is still `("lam3",)`.

Input: `reports/review_bt_aux_2026-09-15.md` (PASS_WITH_CONCERNS) and `reports/impl_bt_aux_2026-09-15.md`.

## 1. `emc/bt_real_frac` (review check 4)

`BT_MONITOR_KEYS` gains `"emc/bt_real_frac"` and `bt_train_step` marks
`float(stats["real_frac"])` under it, beside `emc/bt_ctc`, on every BT step (`as_error=True`,
name `emc_bt_real_frac`, the project's `key.replace("/", "_")` convention). The value is the share
of the batch's synthesized frames that were collaged from a REAL pooled frame of their unit, as
opposed to the pool-mean fallback used while that unit's ring is still empty -- the quantity that
distinguishes the first BT steps of a RESUMED process (empty process-local pool) from a healthy
step. `1.0` = every frame real.

Test `test_bt_real_frac_is_0_before_any_pool_feed_and_1_once_every_unit_is_fed` (test_bt_aux):
a cold `FramePool` gives `real_frac == 0.0` (and every collaged frame is the empty pool's all-zero
mean frame), a pool fed until `units_seen() == n_units` gives `real_frac == 1.0`; both are asserted
again on `bt_loss`'s `stats`. The end-to-end marking is covered by
`train_steps/test_sae_emc.py`, whose BT test loops `BT_MONITOR_KEYS` (now 5 keys, "the BT step
reports 6 losses") and now also bounds `emc_bt_real_frac` to [0, 1].

## 2. The ramp (review check 5): the code was right, the PRINT was misleading

`lam_bt_effective` delegates to `entropy_term.lam_ent_effective` and gives, at `lam_bt = 0.3`,
**0 / 0.075 / 0.15 / 0.225 / 0.3** at sub-epochs 1, 2, 3, 4, 5. The `0, .075, .225, .3, .3` the
review read came from the test's PRINT, which listed the non-consecutive sub-epochs
`(1, 2, 4, 5, 8)` (`test_bt_aux.py`, old line 105); the assertion beside it always covered
`{1, 2, 3, 4, 5, 8}` correctly. NO ramp fix was needed.

The print now walks 1, 2, 3, 4, 5 and says so, and a new test
`test_the_ramp_over_CONSECUTIVE_sub_epochs_is_the_entropy_companions` asserts the five consecutive
values explicitly, the hold from `ramp_epochs + 1` on, and equality with
`entropy_term.lam_ent_effective` at every `(lam, sub-epoch, ramp length)` in a grid -- the same
function, not merely the same shape.

## 3. RETURNN's accumulation across alternating loss-key sets (review check 11): ACCEPTED, no fix

READING, in the checkout the training job uses (`/e/.../2026-07-13_unsupervised/returnn` @ `00171dfe`):

* `torch/engine.py:527-528` -- `accumulated_losses_dict += losses_dict` and
  `accumulated_inv_norm_factors_dict += inv_norm_factors_dict`, both `NumbersDict`.
* `util/basic.py:2086` `NumbersDict.__iadd__` -> `bin_op` (`util/basic.py:2057`), which iterates
  `keys_union` (`util/basic.py:1945`) and calls `bin_op_scalar_optional`
  (`util/basic.py:2040`) with `zero=0`: the side that LACKS a key contributes that zero.
  So a step whose key set differs from the previous step's is **accepted** -- no KeyError, no
  assertion -- and a step that does not report a key adds 0 to its summed loss AND 0 to its
  inv-norm factor.
* `torch/engine.py:638` -- `accumulated_losses_dict / accumulated_inv_norm_factors_dict`
  (`__truediv__`, same `bin_op`): each key is divided by ITS OWN accumulated inv-norm, so every
  key's epoch mean is over exactly the steps that reported it, never over all steps.
* `torch/engine.py:640-641` -- `set_epoch_error(epoch, {f"train_loss_{k}": v})` over the union of
  the keys; `engine.py:646` prints the same dict as the epoch-end "Total train loss" summary.
  The per-step console line (`engine.py:529`) divides the step's own two dicts, whose keys always
  agree.

SETTLED BY A TEST that drives the real path (`test_the_engine_accumulates_the_two_alternating_loss_key_sets`,
test_bt_aux): a real `returnn.torch.engine.Engine`, `Task12AXDataset`, 2 sub-epochs, with the arm's
own `bt_interleaved_batching` as the config's `torch_batching` (test-local `bt_ramp_epochs = 1`, so
the ramp is open at sub-epoch 2 -- the earliest a BT step can run at any ramp length;
`torch_dataloader_opts num_workers = 0` so no worker process pickles the test module's globals).
The stand-in train step dispatches on `A.is_bt_batch` and marks DISJOINT key sets: `l_tau` + error
`emc_agg` (2.0 / 7.0) on an EMC batch, `bt` + error `emc_bt_real_frac` (5.0 / 0.25) on a marker
batch. Result: the engine ran 4 EMC steps at sub-epoch 1 (no marker, ramp 0) and `emc, bt` x 4 at
sub-epoch 2, and the `learning_rates` file holds

    epoch 1: train_loss_l_tau 2.0, train_loss_emc_agg 7.0            (no BT key at all)
    epoch 2: train_loss_l_tau 2.0, train_loss_emc_agg 7.0, train_loss_bt 5.0,
             train_loss_emc_bt_real_frac 0.25

i.e. each key at its own per-step value; a pooled denominator (8 steps) would have halved all four.
NO code fix was required, so the question of a fix's hash-neutrality does not arise.

## 4. Checks run

| check | result |
|---|---|
| `speech_llm.sae.emc.test_bt_aux` | 10/10 PASS (7 before; +ramp, +real_frac, +engine accumulation) |
| `speech_llm.sae.emc.test_emc_train_jobs` | all PASS ("all emc_train_jobs tests passed") |
| `speech_llm.prefix_lm.model.train_steps.test_sae_emc` | 11/11 PASS (BT step now reports 6 losses) |
| `speech_llm.sae.emc.test_pack_config2` | EXIT 0, `PackedEmcTrainJob.wK3hCW0JdJ6G`, arms byte-identical to `6G8CpcswqlTR` / `NSxRSjyBWp5g` / `UjJpp4iQ4kkT` / `DhsiFoHwBP5l` |
| `speech_llm.sae.emc.test_pack_config` | EXIT 0, `PackedEmcTrainJob.SeZzGUScxq4x` unmoved |

`test_pack_config*` need `black` on PATH (`ReturnnConfig.write` shells out to it); without it the
config-equivalence test dies in `subprocess` with `TypeError: expected str ... not NoneType`.
Run them with `PATH=/e/project1/spell/wu24/env/conda/envs/speech_llm/bin:$PATH`.

CENSUS (`scripts/sae_4a_cons_census.py`, before = `git archive 806ea11 src` via `SAE_CENSUS_SRC`,
after = the working tree):

| graph | before | after | diff |
|---|---|---|---|
| s3    | 98   | 98   | none (byte-identical) |
| phase | 1021 | 1021 | none (byte-identical) |
| rate  | 91   | 91   | none (byte-identical) |

`ReturnnTrainingJob.DF6blPpto23t` (the launched lam3 arm) present and unmoved; `ACTIVE_ARMS`
still `('lam3',)`, so the three BT arms print "not built" on both sides. The BT-ON hashes are
covered by `test_pack_config2`'s `REFERENCE_IDS`, which still match after the change. The round is
HASH-NEUTRAL: it adds a monitor value and tests, and touches no config-written key.

## 5. Files touched (commit `9c7c1af`)

* `src/speech_llm/sae/emc/bt_aux.py` -- `emc/bt_real_frac` in `BT_MONITOR_KEYS` and in
  `bt_train_step`'s values dict; two comment/docstring passages saying what it reads back.
* `src/speech_llm/sae/emc/test_bt_aux.py` -- the consecutive-sub-epoch ramp test, the
  `bt_real_frac` test, the real-engine accumulation test (with the code reading above as its
  header comment), the corrected ramp print, the header checklist and the `__main__` runner.
* `src/speech_llm/prefix_lm/model/train_steps/test_sae_emc.py` -- one line bounding
  `emc_bt_real_frac` in the existing `lam_bt = 0.3` end-to-end test.

## 6. Notes for the coordinator

* CONCURRENT EDITS: while this round ran, another session was editing
  `configs/config_sae_4a_s3b_pack_v2.py` and `sae/emc/test_pack_config2.py` (the pack2 time budget:
  the ask moved from 9.9 h to 3.0 h with a "derived need x headroom" rule). Those files are NOT in
  this commit and were not touched here; `test_pack_config2` was re-run after their edit landed and
  still passes with `wK3hCW0JdJ6G` unmoved. `configs/config_sae_1g_v1.py` and the untracked
  `config_sae_3e1_d6_swap_cont_v1.py` are likewise someone else's and were left alone.
* Still open from the review, NOT part of this round: `SAE_4A.md:1017` pre-registers
  `emc/bt_units_per_s` while the code emits `emc/bt_synth_phones_per_s` (the plan line needs the
  amendment, a coordinator edit); the arms inherit `prior_order: 2` from the lam3 control; and the
  BT step's GPU cost is still an expectation, to be read off the first arm's own log.
