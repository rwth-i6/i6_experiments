# S3b-F: the label-free content term reaches the live train step (2026-09-15)

Status: **DONE**. Commit `fb97abb` on `haotian_modality_matching_jupiter` (base `e74d561`, the
entropy companion; the term itself is `6ec6b8a`). Defaults OFF, hash-neutral, nothing launched.
The wiring diff of `impl_content_term_2026-09-15.wiring.diff` was written against an older base and
was rebased by hand onto the live `train_steps/sae_emc.py`, `definitions/sae_emc.py` and
`emc_train_jobs.py` (changed since by `1f05ca9` and `e74d561`); every hunk is an insertion, and the
content block is placed AFTER the consistency block, i.e. newest term last, as the live file orders
the others.

## (1) The tap: `content_layer = 1`

`recognizer.py:105 CONTENT_TAPS = {0: "post_bn", 1: "pre_conv", 2: "mid"}`. At the stage's
`n_layers = 1` the conv IS the output projection (`recognizer.py:175`, `in_dim -> n_out`), so the
LAST hidden representation before it is the conv's input: tap **1**, `pre_conv`, taken at
`recognizer.py:287` after batch norm, after the residual `in_proj` + dropout, after the pad mask,
at the 50 Hz input clock (`_forward`). Tap 0 (`post_bn`) is shallower -- it precedes the residual
projection -- and tap 2 (`mid`) exists only at `n_layers = 2` and is refused here
(`recognizer.py:199-205`). This matches Liu et al. 2022 (wav2vec-U 2.0), where the auxiliary
label-free head sits on an INTERMEDIATE generator representation rather than on the phone logits;
a head on the output projection would price the emissions directly and is not that term.

`content_k = 64` is pinned by that paper's Table 2 and by `mfcc_codes.NUM_CODES`; the config asserts
`CONTENT_K == mfcc_codes.NUM_CODES == emc_train_jobs.EMC_CONTENT_K`.

## (2) What the wiring does

* `train_steps/sae_emc.py`: `lam_content = float(getattr(model, "lam_content", 0.0))`; only when it
  is non-zero does the step call `recognizer.forward_with_content(...)` instead of the plain
  forward. The block refuses a missing head, a missing `content_codes` key and a code/feature frame
  mismatch, then `mark_as_loss(content_ce, "content", scale=lam_content)` and reports the two
  MONITORS `emc/content_ce` (nats per frame, unscaled) and `emc/content_frame_acc`. Under
  `freeze_recognizer` the logits are detached with the rest.
* `definitions/sae_emc.py`: container kwargs `lam_content = 0.0` and `content_codes_key`; the init
  load is now `_load_state(..., allow_missing_prefix="content_head." if the head exists else None)`
  -- missing keys UNDER that prefix are left at their fresh init and printed, every other missing
  key and EVERY unexpected key stays a hard error. A scale without a head, or a head without a
  scale, is refused (an untrained readout would otherwise ride in every checkpoint).
* `emc_train_jobs.py`: `lam_content`, `content_k`, `content_layer`, `train/dev_content_code_hdfs`.
  HASH-NEUTRAL BY OMISSION, a fourth time: at the default nothing is written into `model_args`, the
  `recognizer_kwargs` copy, `extern_data` or either MetaDataset, so no banked job moves.
  `subepoch_reads(..., net_args=...)` forwards to `eval_jobs.posterior_dump`, which is how a read
  rebuilds the recognizer in the shape its checkpoint slice was saved in.

## (3) The held arm `lam3_ct`

`config_sae_4a_s3b_rate_v1.py`: `LAM_RATE_ARMS` gains `(3.0, "lam3_ct")` and
`CONTENT_ARMS["lam3_ct"] = dict(lam_content=0.3, content_k=64, content_layer=1)`. `ACTIVE_ARMS`
stays `("lam3",)`. `MfccFeatureJob` / `MfccKMeansJob` (2000 tc100-train utterances, the job's own
default fit) / `MfccCodesHdfJob` are built LAZILY, inside the arm loop, only when a content arm is
active, so the held graph registers no MFCC job at all. The CV stream reuses the train codes: the
stage's CV set is a segment split of the same tc100 train feature store, and the dev-clean /
dev-other reads are forward-only.

Generated config of `lam3_ct` vs `lam3` (arm temporarily activated, NOT committed): 52 diff lines,
all accounted for -- `'lam_content': 0.3`, `'content_k': 64`, `'content_layer': 1` in `model_args`,
the `content_codes` extern_data entry and the `content` HDF stream + `data_map` on the train AND dev
MetaDataset (train and dev config blocks), plus the job-id-derived `model` path. `lam3`'s own
serialized config carries none of the four keys.

## (4) The census script (`scripts/sae_4a_cons_census.py`)

The `SAE_CENSUS_SRC` override was a silent no-op (`review_consistency_r2_2026-09-15.md` finding 1):
`settings.py:191` re-inserts the LIVE `recipe/2025-10-speech-llm/src` at `sys.path[0]` while
`from sisyphus import tk` runs. Reproduced again here: with the archive path inserted BEFORE that
import, `speech_llm.__file__` resolves to the live tree.

Fixed in place (`settings.py` untouched): the sisyphus import comes FIRST, then the live src is
dropped from `sys.path` outright and the archive is put in front, and every imported module is
asserted to resolve under the tree that was asked for. The tree used (`# src:`) and each graph
module (`# module:`) are printed on STDERR, so the stdout census of an archived tree still diffs
against the live one line for line. `--self-test` unpacks HEAD twice, changes ONE arm tuple
(`(3.0, "lam3") -> (3.5, "lam3")`) in the second copy, censuses both as subprocesses and requires
(a) each run to report the tree it was given and (b) the changed tree to report a MOVED job id --
with the defeated override both sides are the live tree and the test fails. It passes:
`plain -> ReturnnTrainingJob.DF6blPpto23t`, `moved -> ReturnnTrainingJob.sJNHnbiyS7YN`, 128 ids
differ. NOTE: the setup dir is not a git repository, so this script cannot be committed; it lives
at `/e/project1/spell/wu24/2026-07-13_unsupervised/scripts/sae_4a_cons_census.py`.

## (5) Census with the fixed script (before = `git archive e74d561`, after = the live tree)

| graph | before | after | verdict |
|---|---|---|---|
| s3 | 98 | 98 | id-for-id IDENTICAL |
| phase | 1021 | 1021 | id-for-id IDENTICAL |
| rate | 91 | 91 | id-for-id IDENTICAL (only the new `# arm lam3_ct: not built` comment line differs) |

Arms (all four temporarily activated in a scratch process, both trees):
`lam3 ReturnnTrainingJob.DF6blPpto23t`, `lam3_ent_a TWh8GPa8ItGi`, `lam3_ent_b HdP6u4MSWRpk` --
all three UNMOVED; **`lam3_ct` = `ReturnnTrainingJob.DhsiFoHwBP5l`** (new). With the four arms
active the rate graph goes 219 -> 287 jobs: 68 added (the lam3_ct chain plus
`MfccFeatureJob.Y8INRgLo8Avo` / `MfccKMeansJob.wejLUflqBxxo` / `MfccCodesHdfJob.ovaQXsbT8s3E`),
0 dropped.

## (6) Tests (conda `speech_llm` env, `black` on PATH, all exit 0)

* `prefix_lm.model.train_steps.test_sae_emc` -- 9 PASS, three new:
  * `lam_content = 0`: 15 losses/monitors identical BIT FOR BIT against the pre-change step, no
    `content` loss key, 4337 parameters either way, no `content_head` in the state dict.
  * `lam_content = 0.3` on a synthetic code tensor: CE 47.6529 nats/frame at scale 0.30, frame acc
    0.000 (random head), gradient reaches head 1.076e+02 AND the tapped `in_proj` 4.077e+01 and the
    BN 4.736e-01 -- so not only the head -- while the output projection is untouched.
  * checkpoints both ways.
* `sae.emc.test_emc_train_jobs` -- 10 ok, one new: constants agreement, hash EQUALITY at the
  defaults, the four keys absent from the default prolog, hash change + all four keys + the
  extern_data entry + both dataset streams when on, and four refusal cases.
* Unchanged suites re-run green: `test_content_term`, `test_recognizer`, `test_entropy_term` (7),
  `test_rate_term` (20), `test_consistency` (17), `test_pack_jobs` (4, re-derives
  `ReturnnTrainingJob.DF6blPpto23t`).
* Graph load: the rate config builds at `ACTIVE_ARMS = ("lam3",)` in 2.5 s, 91 jobs.

None of this is a run: nothing was launched, and no content-term training step has executed on real
MFCC codes or on a GPU.

## Assumptions named

1. **The "head-carrying checkpoint at `lam_content = 0`" direction.** The container REFUSES a head
   without a scale by design, so that direction cannot be tested on the container. It is tested
   where it actually occurs -- the READ path: a recognizer built from `subepoch_reads`' `net_args`
   (`RECOGNIZER_NET_ARGS` + `content_k` / `content_layer`) loads the head-carrying theta slice, while
   a BARE recognizer refuses it as an unexpected key. That refusal is exactly why `net_args` exists.
2. **`dev_content_code_hdfs = train_content_code_hdfs`** for `lam3_ct`, on the CV-is-a-segment-split
   argument above. If a future arm puts a genuinely different corpus in the CV stream, it needs its
   own codes job.
3. The `black` binary must be on `PATH` (sis_env) for any run that serializes a `ReturnnConfig`.

## Not done / out of scope

* `consistency.py`, `pack_jobs.py`, `config_sae_4a_s3b_pack_v1.py` untouched, as instructed;
  `lam3_ct` is not in the pack config, so the second packed node still has to be wired when it is
  funded.
* Another implementer's uncommitted `config_sae_1g_v1.py` and the untracked
  `config_sae_3e1_d6_swap_cont_v1.py` in the same checkout were left alone and are NOT in `fb97abb`.
