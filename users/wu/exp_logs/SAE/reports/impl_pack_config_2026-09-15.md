# SAE 4a S3b-PACK -- the packed-node config for four S3b arms (implementer report, 2026-09-15)

Status: DONE. One `PackedEmcTrainJob` with the four dispatched arms is wired, nothing is launched,
no existing hash moved, and each arm's RETURNN config is BYTE-IDENTICAL to its single-arm
counterpart's.

## Files

New, in `recipe/2025-10-speech-llm` (branch `haotian_modality_matching_jupiter`, commit `69efa16`):

* `src/speech_llm/prefix_lm/sis_recipe/exp2025_11_06_speech_llms/librispeech/configs/config_sae_4a_s3b_pack_v1.py`
* `src/speech_llm/sae/emc/test_pack_config.py`

New, untracked (the setup dir is not a repository, as for the other stage wrappers):

* `config/sae_4a_s3b_pack.py` -- the sisyphus entry point (`sis m config/sae_4a_s3b_pack.py`).

`config_sae_4a_s3b_rate_v1.py`, `config_sae_4a_s3b_cons_v1.py`, `pack_jobs.py` and every other file
were READ ONLY. No edit was needed in `pack_jobs.py`: the config-level surface it exposes
(`EmcArmSpec`, `PackedEmcTrainJob`, `pack.arm(tag)`) carries the whole wiring.

## The four arms

| arm | GPU | source | term keys |
|---|---|---|---|
| `lam1` | 0 | `config_sae_4a_s3b_rate_v1.LAM_RATE_ARMS` | `lam_rate 1.0`, `rate_rho_hz 9.6619373279` |
| `lam10` | 1 | same | `lam_rate 10.0`, same rho |
| `spec` | 2 | `config_sae_4a_s3b_cons_v1.CONS_ARMS` | `lam_rate 3.0`, `lam_cons 1.0`, `cons_views ["specaug"]` |
| `spec_speed` | 3 | same | the above with `cons_views ["specaug","speed"]`, the `features_pert` stream on train AND dev, and `batch_size = {"features": 88000}` |

GPU index is `pack_jobs`' own sorted-name assignment. `lam3` (`ReturnnTrainingJob.DF6blPpto23t`,
running as a single-arm job) is NOT packed, and the config asserts it out; the packed graph does not
contain that id.

Everything an arm needs is IMPORTED from the two stage configs (rho, the lam grid, the view sets,
the splits, the frozen prior / eta / phone LM, the tau schedule via `s3_jobs`, the flat init via
`s0b`), so no literal of theirs is retyped. The only number this config states itself is the per-arm
`time_rqmt = 6.0 h`, which is the literal both stages pass to `emc_train_jobs.emc_training`.

Reads per arm are the single-arm set, registered off `pack.arm(tag)` with the same helpers: per
sub-epoch (1..8) greedy PER, decode stats (phone rate, distinct strings) and the theta slice on
dev-clean AND dev-other; the speaker-matched derangement gap at sub-epochs 4 and 8 on both splits;
then the unsupervised checkpoint selection over the tau = 2 checkpoints. `word_wer=False`, as in
both sources. Outputs are `exp2025_11_06_speech_llms/librispeech/sae_4a_s3b_pack/<arm>/ep<k>/<split>/<file>`
with the single-arm file names (473 registered outputs, exactly the expected set).

## Verification

**(1) Per-arm config equivalence (the check that matters).** For each arm the file
`PackedEmcTrainJob.create_files` would write was diffed against the file the single-arm
`ReturnnTrainingJob.create_files` would write, normalising only each job's OWN output prefix
(`model = "<ARM_OUT>/models/epoch"`, the one line that must differ). The single-arm side is built by
the SOURCE configs themselves -- `config_sae_4a_s3b_rate_v1.build()` with its two HELD arms
temporarily active, and `config_sae_4a_s3b_cons_v1.build()` -- never by a hand-built copy of their
arguments.

| arm | single-arm job | diff |
|---|---|---|
| `lam1` | `ReturnnTrainingJob.WCh1fMyr88yu` | **byte-identical** |
| `lam10` | `ReturnnTrainingJob.axf11lrsap5i` | **byte-identical** |
| `spec` | `ReturnnTrainingJob.fw5O1L5qL2HL` | **byte-identical** |
| `spec_speed` | `ReturnnTrainingJob.be8qKc7aRj16` | **byte-identical** |

Each reconstructed single-arm job carries the reviewed id in the table, i.e. the reference the diff
is taken against is the reviewed arm and not something the pack invented. The reconstruction method
itself was validated against disk: rebuilding the LAUNCHED `lam3` arm from
`config_sae_4a_s3b_rate_v1` reproduces
`work/i6_core/returnn/training/ReturnnTrainingJob.DF6blPpto23t/output/returnn.config` **byte for
byte** (the other three reference arms have no job directory, as none of them ever ran).

Also equal per arm: the kept epoch key set and the checkpoint file names the reads consume.

**(2) Census** (`scripts/sae_4a_cons_census.py`, BEFORE = the tree without the two new files, AFTER
= as committed; sorted id lists compared with `diff`, all empty):

| graph | jobs | result |
|---|---|---|
| `config_sae_4a_s3_v1` (`s3`) | 98 | identical |
| `config/sae_4a_phase.py` (`phase`) | 1021 | identical |
| `config_sae_4a_s3b_rate_v1` (`rate`) | 91 | identical, `lam3` unmoved at `ReturnnTrainingJob.DF6blPpto23t` |
| `config_sae_4a_s3b_cons_v1` (`cons`) | 160 | identical, `spec` `fw5O1L5qL2HL` / `spec_speed` `be8qKc7aRj16` unmoved |

The new graph itself: `config/sae_4a_s3b_pack.py` loads to **285 jobs**, containing
`speech_llm/sae/emc/pack_jobs/PackedEmcTrainJob.SeZzGUScxq4x` and exactly two
`ReturnnTrainingJob`s, both S0b inits shared with the rate graph
(`65NNK8Bwxdtd`, `HcXzd6M2eyVZ`) -- no single-arm S3b training job is in it.

**Pack job and its rqmt**: `PackedEmcTrainJob.SeZzGUScxq4x`,
`{gpu: 4, cpu: 64, mem: 256.0, time: 6.6, gpu_mem: 96}` -- 4 x one arm's cpu (16) and mem (64 GB),
and the LONGEST arm's 6.0 h x `SHARED_NODE_TIME_FACTOR = 1.1`, under the 11.5 h engine clamp.

**(3) Tests.** `python -m speech_llm.sae.emc.test_pack_config` -- 3 tests, "ALL pack-config TESTS
PASSED": the arm set (exactly the four, one per GPU, sorted-name GPU bijection, `lam3`'s id absent,
the rqmt above), the alias/output layout (473 outputs, the per-sub-epoch file names per split, the
gap registered at sub-epochs 4 and 8 and nowhere else, nothing registered under the rate/cons
prefixes) and the per-arm config equivalence of (1). Re-run unchanged and still green:
`python -m speech_llm.sae.emc.test_pack_jobs` (4 tests, rc 0).

Nothing was launched, no manager was started, `settings.py` was not touched, and no polling loop
was run.

## Assumptions and open points

1. **Where the config lives.** The dispatch places the new config under `recipe/i6_experiments/...`;
   the two configs it names actually live in
   `recipe/2025-10-speech-llm/src/speech_llm/prefix_lm/sis_recipe/exp2025_11_06_speech_llms/librispeech/configs/`,
   and the new one was written beside them (the only place the imports resolve).
2. **Arm names** are the source stages' own tags (`lam1` / `lam10` / `spec` / `spec_speed`), which
   are unique across the two stages; the dispatch names the arms but not the directory names.
3. **No arm-building function existed to import.** Both source configs build their arm inside a
   monolithic `build()` loop and must not be edited, so the shared `build_emc_train_config` keyword
   block (the S3 call both stages make) is written once in this config, with every VALUE imported
   from those modules; the per-arm equivalence diff in (1) is the guard against drift, and
   `test_pack_config` pins the four reviewed single-arm ids so a later divergence fails loudly.
4. **The selection job is registered per arm**, as in both single-arm configs (the dispatch's file
   list enumerates the per-sub-epoch layout only). It consumes `learning_rates`, which a packed arm
   only publishes when that arm's process exits.
5. **Concurrency**: this graph contains S0b, the perturbed dump and the shared `UnitsHdfJob`, so it
   must not be managed at the same time as S0b / S3 / S3b-R / S3b-C / the perturbed dump.
6. `config_sae_4a_s3b_rate_v1.py` is being edited live by another implementer (its working tree adds
   the held companion arms). This config reads its `COMPANION_ARMS` table defensively
   (`getattr(..., {})`) so a packed rate arm automatically carries whatever that stage defines for
   it; at the two packed tags the table is empty today, which is why the diffs are byte-identical.
