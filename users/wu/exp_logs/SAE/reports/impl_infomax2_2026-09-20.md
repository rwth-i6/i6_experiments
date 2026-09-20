# Implementation report: the amended InfoMax arm set (node_d), second round

Date 2026-09-20. Implementer round, bounded change to the config the first round wrote
(`reports/impl_infomax_2026-09-20.md`). Spec: `SAE_4A_infomax.md` "Design", paragraph
"Amendment after code review" -- its table IS the arm set below. `emc/infomax.py`, the model
definition, the train step, `consistency.py` and every budget config are untouched.

## What changed

`config_sae_4a_infomax_pack_v1.py` (the only experimental file):

| arm | ent_schedule | lam_cons | cons_views | vs the first round |
|---|---|---|---|---|
| ent_50 | `ent_decay_schedule()` | -- (key absent) | -- | unchanged |
| entaug_50 | `ent_decay_schedule()` | 0.03 | ("specaug", "statswap") | was 1.0 |
| aug_50 | absent (model default `None`) | 0.03 | ("specaug", "statswap") | NEW |
| aughi_50 | absent (model default `None`) | 0.1 | ("specaug", "statswap") | NEW |
| enthold_50, entaughold_50 | | | | REMOVED |

`NODES = {"node_d": ("ent_50", "entaug_50", "aug_50", "aughi_50")}`. N = 50, bed `ctrl`,
`KEEP_EPOCHS[50]`, `TIME_RQMT`, the tau/LR schedules and every registered read are the budget
round's own, as before; only the InfoMax model arguments moved.

* `LAM_CONS` 1.0 -> 0.03, new `LAM_CONS_HI` = 0.1; `CONS_VIEWS` unchanged.
* `HOLD_ARMS` and `ent_hold_schedule()` deleted -- no arm consumes them any more (the held-penalty
  arms are gone). `ENT_HOLD_SUBEPOCHS` stays: it is the plateau of `ent_decay_schedule`.
* `HOLD_ARMS` / `AUG_ARMS` replaced by one table `ARM_SPEC = {arm: (with_ent, lam_cons)}`, asserted
  at import to cover exactly the arms of `NODES`; `AUG_ARMS` is derived from it. `_model_args_delta`
  writes ONLY the keys the row turns on, so `aug_50` / `aughi_50` state no `ent_schedule` at all and
  are built at the model's default `None` (no entropy term), and `ent_50` states no `lam_cons`.
* `paired_arm_pairs()`: `ent_50`, `entaug_50`, `aug_50`, `aughi_50` each vs `ctrl_50`; plus
  `entaug_50` vs `ent_50`, `entaug_50` vs `aug_50`, `aughi_50` vs `aug_50`. The pairs of the two
  deleted arms are gone. 7 contrasts, as before, at every kept checkpoint on both dev splits.
* Module docstring: arm table replaced, with the amendment's own reasons (ctrl_50's measured entropy
  3.25 / 2.71 / 0.30 / 0.19 nats per output frame at sub-epochs 1 / 4 / 10 / 21, so a held penalty
  binds on nothing; lam_cons 1.0 measured at 6.4 x the objective's gradient norm), and the paired-read
  paragraph updated.

`test_infomax.py`: the arm table is typed out in the test as `EXPECTED_ARMS` (not read back out of
the config), `NODES["node_d"]` must equal it in order, and per arm the test asserts the exact key
set, the decayed schedule or the ABSENCE of `ent_schedule`, the exact `lam_cons` and the view tuple,
on top of the unchanged byte-identity rule (each written config equals `ctrl_50`'s after popping
that arm's own InfoMax keys). The pair set is asserted against the amendment's list. The real
train-step test gained the `aug_50` / `aughi_50` shape: at `ent_schedule` unset and `lam_cons` > 0
the step marks exactly `cons`, `emc/cons_kl_specaug`, `emc/cons_kl_statswap`, `emc/cons_frames` on
top of the 33 baseline keys and NONE of `ent`, `blankfree_entropy_per_frame`,
`blankfree_lambda_ent`. The hold-schedule test is replaced by an assertion that
`ent_hold_schedule` no longer exists.

## Checks

Both run from the setup dir, `speech_llm` conda env (black on PATH) and the sisyphus venv.

1. `python -m speech_llm.sae.emc.test_infomax` -- exit 0, all 11 checks pass, including
   `PASS aug_50 marks cons (11.003740 over 14 frames) and no entropy term at lam_cons 0.03`,
   the same for `aughi_50` at 0.1, `PASS the defaults leave the step bit for bit (33 loss keys,
   l_tau 1.517274)`, `ent 0.077114 (lambda 0.100, 0.262 nats/output frame)` and
   `PASS the four arms differ from ctrl_50 only in the InfoMax model arguments`.
2. `sis --config config/sae_4a_infomax_pack.py console --script -c <census>` -- exit 0, no
   traceback, **265 jobs / 436 targets** (the first round's counts) and exactly two
   `PackedBlankfreeTrainJob`s in the graph:
   * `PackedBlankfreeTrainJob.YtvrSez8z9Wf` -- the NEW node_d (was `lQXR4mpLqVPC`, which was never
     on disk),
   * `PackedBlankfreeTrainJob.ks7CbtlvpcIL` -- node_a, running, unmoved (asserted by
     `_baseline_node` and again in the test).
   On disk the packed-job dirs are still only the three running nodes (`ks7CbtlvpcIL`,
   `4QzmftNlbErt`, `reEI2Nd0S77A`) and no `alias/sae/4a/infomax_pack` was created: the load funded
   and launched nothing.

A passing load and a passing test are not a result: nothing here says the arms train or that the
weights 0.03 / 0.1 are the right ones -- they are the amendment's numbers, taken as given.

## Commit

`recipe/2025-10-speech-llm`, branch `haotian_modality_matching_jupiter`, `68549f9`, two files staged
by explicit path (`config_sae_4a_infomax_pack_v1.py`, `sae/emc/test_infomax.py`). Not pushed. Other
people's uncommitted edits in that checkout (`config_sae_1g_v1.py`, an untracked
`config_sae_3e1_d6_swap_cont_v1.py`) were left alone.

No manager was started and no job was launched; node_d is graph-only.
