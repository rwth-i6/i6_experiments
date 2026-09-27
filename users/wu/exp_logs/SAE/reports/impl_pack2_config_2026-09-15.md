# SAE 4a S3b-PACK2 -- the second packed-node config (implementer report, 2026-09-15)

Status: DONE. One `PackedEmcTrainJob` over the four held arms `bt_a` / `bt_b` / `bt_c` / `lam3_ct`
is wired, nothing is launched, no existing hash moved, and each arm's RETURNN config is
BYTE-IDENTICAL to its single-arm counterpart's.

## Files

New, in `recipe/2025-10-speech-llm` (branch `haotian_modality_matching_jupiter`, commit `806ea11`,
base `325e47c`; staged by explicit path, NOT pushed):

* `src/speech_llm/prefix_lm/sis_recipe/exp2025_11_06_speech_llms/librispeech/configs/config_sae_4a_s3b_pack_v2.py`
* `src/speech_llm/sae/emc/test_pack_config2.py`

New, untracked (the setup dir is not a repository):

* `config/sae_4a_s3b_pack2.py` -- the sisyphus entry point (`sis m config/sae_4a_s3b_pack2.py`),
  mirroring `config/sae_4a_s3b_pack.py`.

`config_sae_4a_s3b_rate_v1.py` (incl. `ACTIVE_ARMS`), `config_sae_4a_s3b_pack_v1.py`,
`pack_jobs.py`, `test_pack_config.py` and everything else were READ ONLY. No edit was needed in
`sae/emc/` outside the new test file. The concurrent code review of the BT-aux change touched
nothing here.

## The four arms

| arm | GPU | source table | term keys on top of `lam_rate = 3` / rho = 9.6619373279 |
|---|---|---|---|
| `bt_a` | 0 | `rate.BT_ARMS` | `lam_bt 0.1`, `bt_depth "full"`, `bt_text_phn` = T_phi |
| `bt_b` | 1 | same | `lam_bt 0.3`, `bt_depth "full"`, same text |
| `bt_c` | 2 | same | `lam_bt 0.3`, `bt_depth "output_only"`, same text |
| `lam3_ct` | 3 | `rate.CONTENT_ARMS` | `lam_content 0.3`, `content_k 64`, `content_layer 1`, the MFCC code stream on train AND dev |

GPU index is `pack_jobs`' own sorted-name assignment. `lam3` (`ReturnnTrainingJob.DF6blPpto23t`)
and the first pack's four arms (`PackedEmcTrainJob.SeZzGUScxq4x`) are NOT in this graph; the config
and the test assert both out.

Every constant is IMPORTED from `config_sae_4a_s3b_rate_v1` (rho, the lam grid, `BT_ARMS` and
`BT_TEXT_PHN`, `CONTENT_ARMS` and `CONTENT_K`, the splits, the frozen prior / eta / phone LM), the
tau schedule via `s3_jobs` and the flat init via `s0b`. `lam3_ct`'s label-free MFCC -> k-means ->
codes chain is built by S3b-R's OWN helper `rate._content_codes(...)` rather than re-typed here, so
the three jobs are the single-arm ones at their own ids
(`MfccFeatureJob.Y8INRgLo8Avo` / `MfccKMeansJob.wejLUflqBxxo` / `MfccCodesHdfJob.ovaQXsbT8s3E`,
the ids `impl_content_wiring_2026-09-15.md` section 5 reports). Their three stats outputs stay
registered under S3b-R's prefix, exactly as the S0b inits' outputs stay under S0b's in v1; no ARM
read of this pack lands outside `sae_4a_s3b_pack2/` (asserted).

Reads per arm are the single-arm set, registered off `pack.arm(tag)` with the same helpers: per
sub-epoch (1..8) greedy PER, decode stats (phone rate, distinct strings) and the theta slice on
dev-clean AND dev-other; the derangement gap at sub-epochs 4 and 8 on both splits; then the
unsupervised checkpoint selection over the tau = 2 checkpoints. `word_wer=False`. `lam3_ct`'s reads
carry `net_args = {**RECOGNIZER_NET_ARGS, content_k, content_layer}` (its theta slice has
`content_head.*`); the other three pass `None`, i.e. the read as it always was. Outputs are
`exp2025_11_06_speech_llms/librispeech/sae_4a_s3b_pack2/<arm>/ep<k>/<split>/<file>` -- 473
registered outputs, exactly the expected set.

## The time requirement and its derivation

`rqmt = {gpu: 4, cpu: 64, mem: 256.0, time: 9.9, gpu_mem: 96}` -- 4 x one arm's cpu (16) and mem
(64 GB) as in v1, and the LONGEST arm's time x `SHARED_NODE_TIME_FACTOR = 1.1`.

Per arm (`PackedEmcTrainJob(time_rqmt={...})`, a mapping, not one number):

* `lam3_ct` **6.0 h** -- the literal S3b-R passes to `emc_training`, v1's per-arm number. The
  content term is one linear head on an activation the forward already computes: no extra DP call,
  no extra optimizer step.
* `bt_a` / `bt_b` / `bt_c` **9.0 h** = 6.0 x **1.5**. The BT arms run TWO optimizer steps per EMC
  batch once the ramp opens at sub-epoch 2; the added step is a CTC forward/backward of the
  recognizer on 128 collage sentences with NO lattice DP, against an EMC step that pays three DP
  calls at `rate_fd_mode = "central"`, so the pair costs less than two EMC steps. 1.5 x is the
  dispatch's estimate for that pair (EMC half unchanged + BT half priced at half an EMC step).

Allocation = max(9.0, 6.0) x 1.1 = **9.9 h**, inside the 11.5 h clamp of
`settings.check_engine_limits` (`settings.py:120`), asserted by the test.

READING OF THE DISPATCH (stated, because it is a factor-of-1.1 ambiguity): "1.5x the v1 per-arm
time" is applied to the per-arm literal 6.0 h, not to v1's 6.6 h allocation -- 6.6 already contains
the shared-node factor, which `pack_jobs` applies once itself. The other reading would give
6.6 x 1.5 x 1.1 = 10.89 h, also under the clamp.

This budget is an ESTIMATE, not a measurement: no BT step has been timed on a GPU
(`impl_bt_aux_2026-09-15.md` section 10). It should be read off the first `bt_a` log
(`emc_sec_per_step` spans an EMC/BT pair there, so the BT cost is that value minus lam3's). For
scale, the finished `lam3` arm used 0.654 h of its 6 h for all 8 sub-epochs.

## Verification

**(1) Per-arm config equivalence.** For each arm the file `PackedEmcTrainJob.create_files` would
write was diffed against the file the single-arm `ReturnnTrainingJob.create_files` would write,
normalising only each job's OWN output prefix (`model = "<ARM_OUT>/models/epoch"`). The single-arm
side is built by `config_sae_4a_s3b_rate_v1.build()` with its four HELD arms temporarily active,
never by a hand-built copy of its arguments.

| arm | single-arm job | diff |
|---|---|---|
| `bt_a` | `ReturnnTrainingJob.6G8CpcswqlTR` | **byte-identical** |
| `bt_b` | `ReturnnTrainingJob.NSxRSjyBWp5g` | **byte-identical** |
| `bt_c` | `ReturnnTrainingJob.UjJpp4iQ4kkT` | **byte-identical** |
| `lam3_ct` | `ReturnnTrainingJob.DhsiFoHwBP5l` | **byte-identical** |

Each reconstructed single-arm job carries the dispatched id (asserted before the diff), so the
reference is the reviewed arm and not something the pack invented. Also equal per arm: the kept
epoch key set (1..8) and the checkpoint file names the reads consume.

**(2) Census** (`scripts/sae_4a_cons_census.py`, current tree = as committed):

| graph | jobs | result |
|---|---|---|
| `s3` | 98 | as before |
| `phase` | 1021 | as before |
| `rate` | 91 | as before, `lam3` unmoved at `ReturnnTrainingJob.DF6blPpto23t` |
| pack v1 graph (`config_sae_4a_s3b_pack_v1`) | 285 | `PackedEmcTrainJob.SeZzGUScxq4x` unmoved, rqmt time 6.6 h |

A before/after pair is unnecessary: the two new files are imported by nothing except the new
workspace entry config and the new test, so they cannot reach those graphs (and `ACTIVE_ARMS` is
untouched).

**The new graph**: `config/sae_4a_s3b_pack2.py` loads to **284 jobs** / 508 registered outputs
through sisyphus' own `config_manager.load_configs` (no manager, no submit, rc 0), containing
`PackedEmcTrainJob.wK3hCW0JdJ6G` and no `ReturnnTrainingJob` of any S3b arm. Against the union of
the s3 / phase / rate / pack-v1 graphs (1343 ids) it adds **256 jobs**: the pack itself (1), the
three MFCC-chain jobs `lam3_ct` needs, and 252 reads (4 arms x 63; by class: 104
`emc_train_jobs`, 64 `eval_jobs`, 64 `i6_core/returnn/forward`, 16 `s3_jobs`, 4 `selection`) --
the same 252 reads per pack as v1. Nothing is dropped.

**(3) Tests** (conda `speech_llm` env, `black` on PATH, all rc 0):

* `speech_llm.sae.emc.test_pack_config2` -- NEW, 4 tests, "ALL pack2-config TESTS PASSED": the arm
  set (the four, one per GPU, sorted-name GPU bijection, `DF6blPpto23t` and `SeZzGUScxq4x` absent,
  cpu/mem/gpu_mem = 4 x one arm's), the time budget (1.5 x for the BT arms, 1.0 x for `lam3_ct`,
  allocation = longest x 1.1 = 9.9 h, under the 11.5 h clamp), the alias layout (473 outputs, the
  per-sub-epoch file names per split, the gap at sub-epochs 4 and 8 and nowhere else, exactly the
  three borrowed S3b-R MFCC outputs and nothing under the BT / cons / pack-v1 prefixes) and the
  per-arm config equivalence of (1).
* Re-run unchanged and still green: `test_pack_config` (3 tests, v1's four arms still
  byte-identical), `test_pack_jobs` (4 tests), `test_emc_train_jobs` (11 tests).

Nothing was launched, no manager was started, `settings.py` was not touched.

## Assumptions and open points

1. **The 1.5 x base** is the per-arm 6.0 h, not v1's 6.6 h allocation (see above). Both clear the
   clamp; if the orchestrator meant the other reading, only `BT_STEP_TIME_FACTOR`'s base changes.
2. **`rate._content_codes` is a private helper of another config module**, called here so the MFCC
   chain cannot drift from the single-arm arm's. The alternative -- rebuilding the three jobs in
   this file -- would retype constants and risk a silent hash split; the byte-identity test would
   catch it, but only after the fact.
3. **Arm names** are the source stage's own tags; the read-job names and the selection `arm` field
   are `s3b_pack2_<tag>`, as v1 uses `s3b_pack_<tag>`.
4. **Concurrency**: this graph contains S0b, the shared `UnitsHdfJob` and the S3b-F MFCC chain, so
   its manager must not run beside the S0b / S3 / S3b-R / S3b-C / pack-v1 managers.
5. The review findings on v1 carry over unchanged, as they are properties of `PackedEmcTrainJob`:
   an operator's error-clear destroys all four arms' work (F1 -- use the manual error-marker
   route), repairing one arm re-funds the other three and all 252 reads (F2), and a permanently
   failing arm blocks its siblings' `learning_rates` output and therefore their selection job (F3).
   The per-sub-epoch gate reads are not affected by F3.

## Amendment 2026-09-16 -- the time ask comes off the measurement, not the literal

Coordinator correction: the first pack's review measured 269-296 s per sub-epoch on its SLOWEST arm,
i.e. **0.69 h for all 8 sub-epochs**, so the 9.9 h above was ~14x the observed time and only cost
backfill wait. The budget block in `config_sae_4a_s3b_pack_v2.py` was replaced by that derivation
(commit `abff24a`):

    MEASURED_ARM_HOURS = 0.69                                    # v1's slowest arm, 8 sub-epochs
    BT_STEP_TIME_FACTOR = 1.5                                    # the BT arms' second optimizer step
    TIME_RQMT_BT_ARM_NEED_HOURS = 0.69 x 1.5 = 1.035 h
    derived allocation need = 1.035 x 1.1 (SHARED_NODE_TIME_FACTOR) = 1.139 h
    TIME_RQMT_ALLOC_HOURS = 3.0                                  # rounded up with headroom, 2.6x

`PackedEmcTrainJob` applies the 1.1 itself, so the per-arm number handed to it is
`3.0 / 1.1 = 2.727 h` (ONE number for all four arms now: the headroom, not any arm's own need, sets
the ask) and the request is exactly **`time: 3.0` h** -- `{gpu: 4, cpu: 64, mem: 256.0, time: 3.0,
gpu_mem: 96}`, well inside the 11.5 h clamp. `build()` asserts the ask covers the derived need. The
1.5 remains an ESTIMATE (no BT step has been timed on a GPU); at 3.0 h an arm that overruns resumes
on its own `arm.finished` marker.

HASH: unchanged. `PackedEmcTrainJob.hash` reads only `arms`, so every rqmt is outside it -- the pack
is still **`PackedEmcTrainJob.wK3hCW0JdJ6G`** and the graph's 284 job ids are id-for-id identical to
the pre-amendment census (`diff` empty).

TESTS: `test_pack_config2` re-run, rc 0, 4 tests, "ALL pack2-config TESTS PASSED"; the rewritten
`test_time_budget` asserts the derivation (need = measured x 1.5 x 1.1), one budget for all four
arms, `rqmt["time"] == TIME_RQMT_ALLOC_HOURS`, and the clamp. The four byte-identity diffs are
unaffected (the rqmt is not in the RETURNN config).
