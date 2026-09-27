# S3b pack v4: implementer report, 2026-09-16

Checkout `recipe/2025-10-speech-llm`, branch `haotian_modality_matching_jupiter`, commit `f8c3749`
(NOT pushed).  This is item 5b of `impl_trigram_pack_2026-09-16.md`, which that round left NOT
STARTED.  Nothing was launched; `settings.py` was not touched; the rate config was not edited.

## Files

| file | delta |
|---|---|
| `configs/config_sae_4a_s3b_pack_v4.py` | NEW, committed |
| `sae/emc/test_pack_config4.py` | NEW, committed |
| `config/sae_4a_s3b_pack4.py` | NEW (workspace entry; the workspace is NOT a git repo, so it is not and cannot be committed -- same as `sae_4a_s3b_pack3.py`) |

## The node

`PackedEmcTrainJob.cgeWUCSGt7xf`, arms `{bt_b_tri_cr, bt_b_tri_s2, lam3_tri_cr, lam3_tri_s2}`
(`rate.TRIGRAM_VARIANT_ARMS`, all of it and nothing else -- asserted at build time, and asserted
disjoint from `pack3.PACK_ARMS`).  rqmt **gpu 4 / cpu 64 / mem 256 / gpu_mem 96 / time 8.0 h**,
pack v3's.  Structure mirrors pack v3 line for line: the same `arm_config` block, the same
per-arm kwargs `dict(lam_rate, rate_rho_hz, **bt, **rate.TRIGRAM_DP, **rate.TRIGRAM_VARIANT_ARMS[tag])`,
the same `EmcArmSpec` / read / selection loop, prefix `sae_4a_s3b_pack4`, **473** registered outputs
under `output/exp2025_11_06_speech_llms/librispeech/sae_4a_s3b_pack4/<arm>/ep<k>/<split>/` (pack v3
has 473 too).  Every constant is imported from `config_sae_4a_s3b_rate_v1`; no arm carries a content
head, so the graph borrows nothing from the source stage's prefix (asserted).

## Byte-identity

`test_config_equivalence_per_arm`, arm by arm, the pack's `finalized_config` against the config the
SINGLE-ARM job writes (built by the source config itself with `ACTIVE_ARMS` temporarily set), up to
each job's own output dir:

* `lam3_tri_cr` == `ReturnnTrainingJob.7X38hg3FYoJs`
* `bt_b_tri_cr` == `ReturnnTrainingJob.WnrqZ9Ny6M7E`
* `lam3_tri_s2` == `ReturnnTrainingJob.leZu891PYYm6`
* `bt_b_tri_s2` == `ReturnnTrainingJob.vWCDsWIJV1Su`

i.e. the four ids the dispatch names, unmoved.  `SECOND_MODEL_SEED = 43` stands untouched.

## The budget (the one thing this file states itself)

Stated as a comment in the config and asserted in `test_time_budget`, every input imported from the
node that measured or derived it:

    0.69 h x 6.9 = 4.761 h  (pack3.MEASURED_ARM_HOURS x the trigram step factor)
         + 0.345 h          (the BT optimizer step, no lattice DP)
         = 5.106 h          (pack3.TIME_RQMT_ARM_NEED_HOURS)
         x 1.1              CONS_STEP_FACTOR: the CR arms' second AUGMENTED recognizer forward.
                            Priced at the only measurement of it there is -- the FIRST pack's
                            per-sub-epoch spread over its four arms (two of which are the
                            lam_cons = 1 consistency arms), 269-296 s, i.e. 296/269 = 1.10x.
                            Applied to every arm, since one number sets the allocation.
         = 5.617 h  x 1.1 (SHARED_NODE_TIME_FACTOR) = 6.178 h derived need; ASK 8.0 h (1.3x),
                            pack v3's own ask, under the 11.5 h settings clamp.

Caveat carried in the comment: the 1.10 was measured at `lam_cons = 1` WITH the speed view, not at
this node's 0.3 on {SpecAugment} alone, so it is an upper-ish bound rather than this arm's own
number; the 6.9 is a lattice-only bench and the 0.345 has never been timed on a GPU.

## Tests (setup dir, `black` on PATH, conda python)

| module | result | checks |
|---|---|---|
| `sae.emc.test_pack_config4` | PASS | 7 tests / 10 `ok` lines (new module) |
| `sae.emc.test_pack_config3` | PASS | 9 |
| `sae.emc.test_emc_train_jobs` | PASS | 13 |

`test_pack_config4` is `test_pack_config3`'s suite (arm set, one-node rqmt, time budget, alias
layout, per-arm byte-identity, the full-trigram read off the written config) plus ONE node-specific
test, `test_each_arm_carries_its_one_variant_key`: the CR arms carry `"lam_cons": 0.3` and
`"cons_views": ["specaug"]` with no `"speed"` view (which would declare a second stream and change
the batching, confounding the paired read) and no seed; the S2 arms carry `random_seed = 43` in the
CONFIG and not in `model_args`, and no consistency key.

## Census

`scripts/sae_4a_cons_census.py`, after the change (the change ADDS files only and edits nothing, so
the before side is the committed tree):

* `s3` **98** jobs, `phase` **1021**, `rate` **91** -- the dispatch's counts, unmoved.
* `ReturnnTrainingJob.DF6blPpto23t` (the launched lam3) present in the rate graph.
* Rebuilt at their banked ids: `PackedEmcTrainJob.SeZzGUScxq4x` (pack v1, 285 jobs),
  `.wK3hCW0JdJ6G` (pack v2, 284), `.byYMQmBNEpLZ` (pack v3, 280).
* Pack v4's graph: **280** jobs, of which **253** are in neither the rate graph nor pack v3's (the
  pack itself plus the four arms' per-sub-epoch reads, gaps and selections); the 27 shared are S0b
  and the common `UnitsHdfJob`.

## Nothing undetermined

Every constant traces: the arm set and its keys to `rate.TRIGRAM_VARIANT_ARMS`, the ids to the
dispatch, the rqmt and `CONS_STEP_FACTOR` to the dispatch's own derivation (pack v1's 269-296 s).
`config_sae_1g_v1.py` and `config_sae_3e1_d6_swap_cont_v1.py` are another implementer's live
working-tree changes in the same checkout; they were left alone and not staged.
