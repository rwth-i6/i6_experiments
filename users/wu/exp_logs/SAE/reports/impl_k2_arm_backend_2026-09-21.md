# Implementer report: the k2 ARM backend (`SAE_4A_lexlat.md` design amendment 9), 2026-09-21

Commit `b19f2ba` on `haotian_modality_matching_jupiter` (`recipe/2025-10-speech-llm`), six explicit
paths. Nothing launched. No project document touched. The four files of the concurrent implementer
(`lexlat_k2.py`, `lexlat_k2_jobs.py`, `config_sae_4a_lexlat_k2_v1.py`,
`config_sae_4a_lexlat_k2_official_v1.py`) were imported from and never edited.

## What was built

* **NEW `sae/emc/lexlat_k2_train.py`** -- `LexlatK2Runtime`, a PLAIN object (never an `nn.Module`;
  nothing in `state_dict`), caching `HLG.pt` per device and `H` from `lexlat_k2.h_topology`. Per
  step `L_lex = log Z_HLG(e / temperature) - log Z_H(e / temperature)`: HLG through
  `k2.intersect_dense_pruned` (search 20, output 8, min_active 30, `max_active` from the config),
  H unpruned through `k2.intersect_dense` (segments re-sorted by decreasing duration and restored),
  both `get_tot_scores(log_semiring=True, use_double_scores=True)`, the DenseFsaVec built as
  `run_probe` builds it. BOTH the emissions and the graph's arc scores are divided by the
  temperature (the bed's DP divides every arc weight; the graph's scores are rescaled in place when
  the anneal moves). The term is `mean_kept[(-L_lex) / n_unit_frames]` -- the lattice loss's own
  reduction and its `keep.sum()` denominator -- handed to `mark_as_loss` at `scale = lam_lex`.
  `lam_lex` is `lexlat_train.lexlat_lambda`, re-exported, unchanged (9.2). Empty pruned lattices are
  dropped from the numerator and counted; above `EMPTY_RAISE_FRAC = 0.10` of a batch the step
  raises. Monitors under `lexlat_k2_*`: `_lam`, `_term_mean`, `_n_empty`, `_empty_frac`,
  `_lattice_arcs_per_frame`, `_lattice_states_per_frame`, `_expected_words`,
  `_expected_escape_words` (from `get_arc_post` over the ragged aux labels) and a CUDA-synchronised
  `_sec`. The module also carries amendment 9.5's child (`overcount_strings`) and its CLI.
* **NEW `sae/emc/test_lexlat_k2_train.py`** -- 18 tests, all passing.
* **NEW `sae/emc/lexlat_k2_arm_jobs.py`** -- `LexlatK2OvercountJob` (CPU 16 / 200 GB / 11 h,
  `__sis_version__ = 1`), the shared-env worker handing the k2 leg to the scratch python
  (`lexlat_k2_jobs._child_env`). Exact `k2.intersect` (host, no beam, so no search constant enters
  the measurement) against `lexlat.string_best_segmentation` + `sil_model_log_prob`. THREE per-token
  columns, because the amendment compares a log-sum with a max-plus reference:
  `over_count = tropical_residual + sum_minus_max`; the pre-registered median <= 0.05 nats/token is
  read on `over_count` as written. Adjacent-repeat and no-path strings are excluded and counted;
  `partial.json` every 50 strings; the summary discloses a partial run.
* **`prefix_lm/model/definitions/sae_blankfree.py`** -- a default-off `lexlat_k2_*` keyword block
  (10 keywords), guarded import, `assert_ramp` against the temperature schedule, mutual exclusion
  with `lexlat_resources`, and a REFUSAL when `lexlat_k2_max_active` is absent.
* **`prefix_lm/model/train_steps/sae_blankfree.py`** -- the guarded term after the rate term and its
  monitors in the `no_grad` block. The k2 arm states no `lexlat_resources`, so the bed's banked
  `lattice.py` carries `l_tau`, the rate term and every `blankfree_*` column and the lexicon enters
  as the added loss alone.
* **NEW `configs/config_sae_4a_lexlat_k2_pack_v1.py`** + setup shim `config/sae_4a_lexlat_k2_pack.py`
  -- the DP pack's arms, on-sets, seeds, mixing, pairing, prior-gap rerun and selection statistic
  with the term swapped; arms `k2lat_20`, `k2shuf_20`, `k2lat_20_e5`, `k2lat_20_s1`. The null's graph
  is `LexlatHLGBuildJob(shuffled=True)` (job unmodified), registered here. `py_overcount()` registers
  the over-count job alone. `returnn_exe` is the k2 env python for the TRAINING job only.

## Checks run

| check | result |
|---|---|
| new suite `test_lexlat_k2_train.py` (k2 env, CPU) | 18 passed |
| `test_lexlat_k2` + `test_lexlat_train` + `test_lexlat` + new suite | 118 passed, 2 skipped, 1 xfailed |
| hash census `config_sae_4a_prepro_pack_v1` before/after the two model edits | 133 job ids, byte-identical (incl. `PackedBlankfreeTrainJob` and 24 `ReturnnForwardJobV2`) |
| hash census `config_sae_4a_lexlat_k2_v1` before/after | 10 job ids, byte-identical |
| both edited model sources imported under the shared AND the k2 python | 10 `lexlat_k2_*` keywords, the train step's block present |
| new pack config: `py()` and `py_overcount()` before the constants are filled | both refuse, naming what to fill and where it is read off |
| new pack config under TEST-ONLY constant overrides | 201 jobs build; `LexlatK2OvercountJob.f5Ljn6twbc4b`, `LexlatHLGBuildJob.zUCMVmtrEigZ` (shuffled), `PackedBlankfreeTrainJob.WbNATSBGkXjG`, `LexlatSelectionStatJob.kwdfnCbZ8cxF` |

The 18 tests cover: `L_lex` against an exact enumeration over frame strings on a back-off-free toy
graph at temperature 1.0 and 2.0 (1e-6); `log Z_H` against the closed form; a finite-difference
backward; the gradient reaching the emissions alone; bit-identity (`torch.equal` on the loss and
both gradients) with the REAL `lattice.lattice_loss` before the on-set and a moved loss at it; the
curriculum equal to `lexlat_train`'s; `max_active = 0` refused; an empty lattice counted and dropped
(the term exactly halves on a 2-utterance batch) and >10 % raising; monitors finite; `check_bed`
refusing another stride or topology; the three over-count columns and the adjacent-repeat exclusion.

## Not covered, and what is undetermined

1. **No executed train step.** The two `prefix_lm/model` edits are covered by the import check, the
   hash census and the runtime's own tests. No test in this tree constructs `SaeBlankfreeModelV1`
   and runs `train_step` (the DP arm's own `test_lexlat_train.py` replicates the step's two lines
   rather than importing them), and building one needs the bed's frozen artifacts and a GPU. The
   first job of the arm is therefore the first execution of that block.
2. **`MAX_ACTIVE`, `HLG_PATH`, `HLG_STATS_PATH` are unfilled by design.** They are measurements of
   `LexlatK2ProbeJob` / `LexlatHLGBuildJob` and the config refuses to build until they are stated.
   `LexlatHLGBuildJob.rtX44PBJFNy1/output/{HLG.pt,build.json}` is on disk and was used ONLY as the
   load test's override.
3. **`_step0_phone_lm_4gram` sources.** All three copies the DP pack lists have been auto-cleaned, so
   `config_sae_4a_lexlat_pack_v1` itself no longer builds. My config appends the surviving
   byte-identical copy (`PriorGapAnalysisJob.l0p0srBryKrs/work/lms/phones_o4.bin`, md5
   `61eae5635495b66cec12ef638bafa6d4`, the digest that config's own comment records) and, because
   the pin is a content sha, the statistic's hash is where any copy would put it. **The DP pack
   config needs the same one-line addition; it is outside my file scope and was not edited.**
4. **The two-environment hazard (point 10), disclosed.** The training job's launcher becomes the k2
   scratch python, so THAT interpreter resolves RETURNN, torch and every i6 package for the arm but
   not for the controls it pairs against, and `/e/scratch` is a 90-day lease. The decodes are
   unaffected (`epoch_reads` forwards the extracted recognizer under `NET_ARGS`). `ep4` of the
   paired table is the check that the two environments agree where the term is still off.
   `test_sae_emc.py` fails 6 tests IDENTICALLY under both interpreters (`'Backend' object has no
   attribute 'RawTensorType'`), i.e. a pre-existing condition of that suite in this checkout, not a
   difference between the environments and not caused by these edits.
5. **Deviations from the dispatch, all minor.** The efficiency probe and the frequency-stratified
   read were NOT copied into the pack config (the dispatch's list names arms, seeds, mixing,
   pairing, selection statistic and prior gap; the k2 route's cost read is the settling probe, and
   the freq-strat read would re-register the DP pack's `word_counts` outputs). The allocation assert
   reuses the DP pack's `E1_TIME_FACTOR_BAR = 2.00`: the k2 route is priced at the SAME Design 7
   bar, and what it actually costs is the settling probe's number, read before launch.
   `overcount_strings` takes an optional `phone2id` (default `prior.PHONE2ID`) so the toy tests can
   pass their own map.

## Code-review round 2 (`reports/review_k2_arm_backend_2026-09-21.md`), 2026-09-21

All five items applied.

1. **The allocation assert no longer prices the arm.** `PROBE_SEC_PER_SUBEPOCH` is a THIRD unfilled
   constant -- the settling probe's MEASURED seconds per sub-epoch for the lexicon leg at the rung
   `MAX_ACTIVE` names -- and `py()` refuses to build without it. `build()` now checks
   `PROBE_SEC_PER_SUBEPOCH <= _e1_bar_sec()` FIRST, where the bar is
   `LexlatEfficiencyProbeJob.TIME_FACTOR_BAR * .SEC_PER_SUBEPOCH` read off the class (2.00 x 601 =
   1202.0 s, E1's own bar, imported and not retyped). The DP-side factor clause is kept below it,
   documented as the weaker one that guards the Slurm clock and passes at any k2 cost.
2. **`HLG_PATH` / `HLG_STATS_PATH` filled** with `LexlatHLGBuildJob.rtX44PBJFNy1/output/{HLG.pt,
   build.json}` -- the finished banked-trigram build (ladder rung 0, ESCAPE, seed-0 pronunciations)
   the live settling probe prices -- as frozen content-carrying pins. The docstring states that an
   official-LM graph is a different objective, not another rung: taking that route REPLACES these
   two constants and re-runs both the over-count read and the settling cells. `MAX_ACTIVE` and
   `PROBE_SEC_PER_SUBEPOCH` stay unfilled.
3. **NEW setup shim `config/sae_4a_lexlat_k2_overcount.py`** exporting `py_overcount` as `py`.
   Loaded: the graph holds ONE job, `LexlatK2OvercountJob.f5Ljn6twbc4b`, and nothing of the pack.
4. **`config_sae_4a_lexlat_pack_v1.py` `_STEP0_PHONE_LM_SOURCES`** gains the surviving
   `PriorGapAnalysisJob.l0p0srBryKrs` copy (the content pin makes which copy is read hash-neutral).
   That config builds again: 202 jobs, and its pack is `PackedBlankfreeTrainJob.DPiivOfTWAdM`, the
   hash the review names -- unchanged. My config now imports that list instead of re-typing it, so
   the two routes score under one fit by construction.
5. **The monitor docstring** now states that the escape word IS a word of `G`, so
   `lexlat_k2_expected_words` INCLUDES the `<unk>` transitions `lexlat_k2_expected_escape_words`
   counts (the two are not disjoint and must never be added), and that the Gate read is the RATIO
   escape / words, at 1.0 meaning the lexicon is inactive.

Re-checks: `test_lexlat_k2_train` + `test_lexlat_k2` + `test_lexlat_train` + `test_lexlat` = 118
passed, 2 skipped, 1 xfailed (k2 env). Census `config_sae_4a_prepro_pack_v1` (133 ids) and
`config_sae_4a_lexlat_k2_v1` (10 ids) byte-identical to the pre-edit run. New pack config: refuses
on `MAX_ACTIVE`, then on `PROBE_SEC_PER_SUBEPOCH`, then FAILS the bar at a test value of 1300 s and
builds at 900 s -- 201 jobs, `PackedBlankfreeTrainJob.WbNATSBGkXjG` unchanged by the fills.
