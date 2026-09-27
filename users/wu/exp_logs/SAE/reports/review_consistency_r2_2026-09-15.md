# S3b-C round-2 fixes -- code review (2026-09-15, commit 1f05ca9 vs 33ea43b / 3f5c805)

Status: **PASS_WITH_CONCERNS**. All three intended deltas (F1 BN guard, F2 speed-length band,
F3 per-key batch_size) are implemented, reach the model / the generated config, and move nothing
else. Compute may be spent. Two findings, neither blocking; both are about verification artifacts
and a docstring claim, not about the arms' numbers.

## Verdict per delta

**F1 -- BN guard. SOUND.** `consistency.py:359-384` walks `module.modules()` and flips
`track_running_stats` False on every `_BatchNorm`, restoring it in a `finally`. The recognizer has
exactly one BN (`recognizer.py:168 nn.BatchNorm1d(in_dim)`, applied over valid frames at
`recognizer._forward`), so `model.recognizer` covers it; `SyncBatchNorm` would also be caught
(subclass of `_BatchNorm`). Torch semantics verified in the installed torch 2.7
(`nn/modules/batchnorm.py:170-177,764-770`): in train mode with the flag off, `F.batch_norm` gets
`None` buffers (batch statistics, forward value unchanged), no buffer write, and
`num_batches_tracked` is NOT incremented; momentum is 0.1 (not None) so the cumulative-average path
is irrelevant. In eval mode the block is a no-op. The three call sites are right:
`train_steps/sae_emc.py:183` clean teacher OUTSIDE, `:315` (specaug) and `:335` (speed) inside the
`with` at `:306`. At `lam_cons = 0` the whole block is unreachable (`:295 if lam_cons:`), so the
step is bit-identical to before -- `test_sae_emc` 6 PASS covers this. The two new tests are
non-vacuous (the un-guarded pass is asserted to move the mean; the buffer equality is checked
against the lam_cons=0 step).

**F2 -- pairing guard. SOUND, as specified.** `consistency.py:343-344` implements the per-row band
`round(T_clean/1.1) - 2 .. round(T_clean/0.9) + 2`; `SPEED_FACTORS = (0.9, 1.1)` matches
`sae/perturb.py:52 MIXES["speedmix_v1"] = (("speed0.9", 0.5), ("speed1.1", 0.5))`, and the ratios
the frame-check job measured (1.1110 / 0.9088) are 1/0.9 and 1/1.1, inside the +-2 tolerance. It is
raised BEFORE the perturbed forward (`train_steps/sae_emc.py:334` precedes `:335`). The ex-tautology
is kept as a plain `assert` (`:342`); under `python -O` it would vanish, which is harmless here
(RETURNN is not run with -O) but it is no longer an error path.

**F3 -- per-key batch_size. SOUND, traced end to end.** `emc_train_jobs.py:774-776` writes
`{"features": 88000}` only when the speed view is on. Traced into RETURNN: `torch/engine.py` ->
`data/pipeline.py:433-437 config.typed_value("batch_size")` (dict passes through untouched) ->
`BatchingIterDataPipe._parse_batch_size:253 NumbersDict(dict)` (broadcast `value = None`) ->
`:298 min_value() = 88000 > 0` (`util/basic.py:2255`, `values()` excludes a None broadcast) ->
`:313 any_compare` (`util/basic.py:2182-2197`): a key absent from the dict is skipped because
`other.value is None`, and the final `self.value/other.value` branch is skipped too. So only
`features` constrains; `units` has the same per-utterance length as `features`, so the realized
batching equals the spec/rate arms'. Confirmed in the generated config of the moved arm:
`ReturnnTrainingJob.be8qKc7aRj16` serializes `batch_size = {'features': 88000}` (line 47 of its
serialized config), `fw5O1L5qL2HL` keeps `batch_size = 88000`. The dict also applies to the dev
set, so dev batching (hence `dev_loss_l_tau`, the selection key) is now comparable across arms.
Side effect, intended and benched: the spec_speed arm now carries ~11 % more total padded frames
per step than under the int limit, because the clean stream is back at the full 88,000.

## Findings

1. **`scripts/sae_4a_cons_census.py:33-34` -- the `SAE_CENSUS_SRC` override is a silent no-op.**
   The script inserts the alternate `src` at `sys.path[0]`, but `from sisyphus import tk` loads the
   setup's `settings.py`, whose `:191 sys.path.insert(0, .../recipe/2025-10-speech-llm/src)` puts
   the LIVE checkout back in front. Reproduced: with `SAE_CENSUS_SRC` pointing at a `git archive` of
   `33ea43b^`, `speech_llm.__file__` and the rate config resolve to the live tree, and the "before"
   run printed the live `ACTIVE_ARMS = ('lam3',)`, which does not exist in that commit. Concrete
   failure: any BEFORE/AFTER census produced with that flag compares the live tree with itself and
   reports "byte-identical" whatever the change did. The round-2 numbers themselves are NOT affected
   (they were compared against stored round-1 files), and I re-verified them independently with a
   corrected script that re-inserts the archive path AFTER `import tk` and asserts `G.__file__`.
   Fix (implementer, not me): move the insert after the sisyphus import and assert the module path.

2. **`consistency.py:325` -- the docstring overclaims the band.** It says the band is "exactly one
   of the two copies, +-tol frames"; the code is the closed interval between the two, which also
   admits `T_pert == T_clean`. Concrete failure mode the guard is advertised to catch but does not:
   if the second stream were wired to an un-perturbed dump, every row passes and the speed view
   degenerates into a dropout-only self-consistency term. Checked and currently fine: the arm's
   `feats_pert` is `L15FeatureHdfJob.OTO33H7dWkfy` (the speedmix dump, dir on disk) while `feats` is
   `L15FeatureHdfJob.e2athsQ218Og`. The band is exactly what the dispatch prescribed, so this is a
   wording/robustness note, not a deviation.

## Beyond the three deltas

Nothing. `git diff 33ea43b 1f05ca9` over the four files of this commit is: the two new functions +
constants + `__all__` + `import contextlib` in `consistency.py`, the guard/indentation and the two
new asserts in `train_steps/sae_emc.py`, the batch_size ternary in `emc_train_jobs.py`, and tests.
The intervening upstream commits (8ff23fe rate `ACTIVE_ARMS`, 5994bd9/3f5c805 `pack_jobs.py`) are
not this round's and touch no file of it. Serialized-config check: `spec` at `3f5c805` vs `1f05ca9`
is byte-identical; `spec_speed` differs in `batch_size` only (plus the job-id-derived `model` path).
`spec` vs `spec_speed` at `1f05ca9` differ only in `cons_views`, the `features_pert` extern_data
entry, the `feats_pert` stream + data_map on train AND dev, and `batch_size` -- 32 diff lines, all
accounted for.

## Census (my own runs, one process per graph, archive trees of d69f938 = 33ea43b^ and 3f5c805)

| graph | pre0 (33ea43b^) | pre1 (3f5c805) | post (1f05ca9) | verdict |
|---|---|---|---|---|
| s3 | 98 | 98 | 98 | id-for-id IDENTICAL |
| phase | 1021 | 1021 | 1021 | id-for-id IDENTICAL |
| rate | 219 | 91 | 91 | pre1 == post; all 91 post ids are in the pre0 219 set |
| cons | - | 160 | 160 | 64 ids moved, 96 unchanged |

`ReturnnTrainingJob.DF6blPpto23t` (launched lam3) unmoved, job dir on disk.
`ReturnnTrainingJob.fw5O1L5qL2HL` (spec) unchanged. `KsVX6rV1ic03 -> be8qKc7aRj16` (spec_speed).
None of the 64 moved cons ids, and neither spec_speed hash, has a directory under `work/`, so
nothing is orphaned or re-funded.

## Tests (conda env `/e/project1/spell/wu24/env/conda/envs/speech_llm/bin/python`, all exit 0)

`speech_llm.sae.emc.test_consistency` 17 PASS ("ALL consistency TESTS PASSED");
`speech_llm.prefix_lm.model.train_steps.test_sae_emc` 6 PASS;
`speech_llm.sae.emc.test_rate_term` 15 PASS; `speech_llm.sae.emc.test_emc_train_jobs` 9 ok, "all
emc_train_jobs tests passed". (The report calls the second one `test_sae_emc`; it lives under
`prefix_lm/model/train_steps/`, not under `sae/emc/`.)

## Other checks

* KL unchanged from round 1: `consistency_loss` has no diff hunk in this commit; teacher detached at
  `consistency.py:415`, fed `log_q` = the untempered recognizer output (`train_steps:183`;
  `temperature` is built at `:180` and enters only the lattice call at `:212`).
* The working tree carries another implementer's uncommitted `recognizer.py` / `config_sae_1g_v1.py`
  edits. My censuses ran off `git archive` trees, so they are free of that; the live-tree census
  gave the same ids.
* Not done: no smoke run of an actual training step on GPU, and no measurement of the realized
  utterances-per-batch on the real dataset (the batching claim rests on the RETURNN source trace
  plus the implementer's test against the real `BatchingIterDataPipe`, which I re-ran).
