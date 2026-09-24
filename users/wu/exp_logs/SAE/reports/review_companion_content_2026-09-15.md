# Review: entropy companion (A, e74d561) + content term (B, 6ec6b8a + fb97abb), 2026-09-15

Read-only. Nothing launched, no manager touched. Verdict: **A PASS**, **B PASS_WITH_CONCERNS**
(one failing test in the data-side suite, one unverified read path -- see 8). Several dispatched
checks were cut by the turn limit and are marked NOT CHECKED.

## 1. Census (item 6) -- VERIFIED, with the FIXED script

`scripts/sae_4a_cons_census.py` (the repaired override: sisyphus first, LIVE_SRC dropped, `# src:`
printed and asserted). before = `git archive 1f05ca9` (the parent of 6ec6b8a, i.e. before BOTH
changes); after = the live tree (= fb97abb plus two unrelated uncommitted files,
`config_sae_1g_v1.py` and an untracked `config_sae_3e1_d6_swap_cont_v1.py`, neither in these graphs).

| graph | before | after | verdict |
|---|---|---|---|
| s3 | 98 | 98 | id-for-id identical |
| phase | 1021 | 1021 | id-for-id identical |
| rate (ACTIVE_ARMS = lam3) | 91 | 91 | id-for-id identical |

`# arm lam3: ReturnnTrainingJob.DF6blPpto23t` on both sides.

All six arms activated in a scratch process (live tree, 415 jobs):
`lam1 WCh1fMyr88yu`, `lam3 DF6blPpto23t`, `lam10 axf11lrsap5i`, `lam3_ent_a TWh8GPa8ItGi`,
`lam3_ent_b HdP6u4MSWRpk`, `lam3_ct DhsiFoHwBP5l`; `MfccFeatureJob.Y8INRgLo8Avo` (4 shards, split
train), `MfccKMeansJob.wejLUflqBxxo` (fit_utts 2000, seed 42, K 64), `MfccCodesHdfJob.ovaQXsbT8s3E`.
Every id the two implementer reports claim is confirmed.

## 2. The running arm is bit-identical (item 3) -- VERIFIED at the config level

Rebuilt `lam3` config (`ReturnnConfig.write`, black on PATH) vs
`work/i6_core/returnn/training/ReturnnTrainingJob.DF6blPpto23t/output/returnn.config`:
**byte-identical, diff empty**. So the live packed run is untouched by both commits.

Deltas of the held arms against that same file, complete:
* `lam3_ent_a`: `'rate_hinge': True`, `'lam_ent': 0.1`, `'ent_ramp_epochs': 4`, and the job's own
  `model =` path. Nothing else.
* `lam3_ct`: `extern_data['content_codes'] = {dim 64, shape (None,), sparse, int32}`,
  `recognizer_kwargs` + `content_k 64` / `content_layer 1`, `'lam_content': 0.3`, the `content`
  HDFDataset stream + `data_map` entry on the TRAIN and the DEV MetaDataset (both point at
  `MfccCodesHdfJob.ovaQXsbT8s3E/output/codes.train-clean-100.shard{0..3}.hdf`), and the `model =`
  path. Nothing else.

NOT CHECKED: the parameter-set / checkpoint-layout claims at the defaults rest on
`prefix_lm.model.train_steps.test_sae_emc` and `sae.emc.test_emc_train_jobs`, which I did not get to
run (turn limit). The container/builder omission logic was read and is sound
(`emc_train_jobs.py:700-726` for rate_hinge/lam_ent, `:788-820` for lam_content), and
`test_entropy_term`'s own container test (which asserts `sis_hash_helper` equality at the defaults)
passes.

## 3. Label leakage (item 1) -- VERIFIED, none found

* `mfcc_codes.py:372-379` `MfccFeatureJob._iter_audio` reads `ex["id"]` and `ex["audio"]["array"]`
  and nothing else off the on-disk HF DatasetDict.
* The dir it is given resolves (graph build, printed) to
  `work/i6_core/datasets/huggingface/TransformAndMapHuggingFaceDatasetJob.OYvh9012Pgkb/output/dataset`,
  which is exactly the `hf_data_dir` in `work/speech_llm/sae/av_states/AvStatesJob.Dsynh5MqmgjY/info`
  -- the same waveforms the frozen L15 store was tapped from. The docstring's claim checks out.
* `MfccKMeansJob`: fit on 2000 tc100 TRAIN utterances at seed 42, MFCC vectors only.
* `MfccCodesHdfJob.run` takes its tag list and lengths from the L15 feature HDF and writes codes in
  that order; no text input anywhere in the module (grep for transcript/lexicon/gold: nothing).
* `entropy_term.entropy_penalty` reads `log_q` and `feat_lens` only.
* The content stream reaches the model only as `extern_data['content_codes']`.

Minor note (not a defect): the k-means fit subset is drawn from the full 28,539-utt tc100 tag list,
so ~20 of the ~285 CV-split utterances are expected to sit in it. The codebook is label-free and the
gate reads PER / rate / derangement, not the CV content CE, so nothing gated is affected; worth one
sentence if the CV content CE is ever quoted.

## 4. Entropy on the untempered posterior, ramp source, hinge in both FD paths (item 2) -- VERIFIED

* `train_steps/sae_emc.py:231-236` assigns `log_q` from the recognizer; `:273` hands that same
  tensor to `lattice_loss`; `:356-358` hands the same tensor to `entropy_penalty`. The DP applies
  `1/tau` to its own arc weights inside `lattice.py`, never to this tensor.
  `test_entropy_term.test_the_entropy_is_taken_on_the_untempered_posterior` passes (3.5196 nats vs
  3.6234 for the tau-tempered copy).
* The sub-epoch is `epoch = int(ctx.epoch)` (`:196`), the same local that feeds
  `model.temperature(epoch)` (`:224`) -- one counter, no state.
* `entropy_term.lam_ent_effective` = `lam * min(1, max(0, (e-1)/r))` -> 0, .25, .5, .75, 1.0 at
  sub-epochs 1-5, which is the dispatched ramp; test passes.
* Hinge: `rate_term.py:700-712` is the ONLY place the loss shape enters, and it sits AFTER `g`
  (= d post_q / d b) is formed, so the stacked and the sequential FD path share it by construction.
  `test_rate_term` passes, including the hinged-gradient-vs-autograd oracle at fp64 and fp32 x
  stacked and sequential (cosine 1.000000, rel. 0.09 %).
* Note, harmless here: under `freeze_recognizer` (`:243`) `log_q` is detached, so the entropy term
  would have no gradient in a phi warm-up sub-epoch. The S3b arms run `freeze_recognizer: False`
  (visible in the serialized config), so this never fires for lam3_ent_a/b.

## 5. Gradient path and the head (item 4) -- VERIFIED BY READING, tests NOT RUN

`recognizer.py:287` takes tap 1 AFTER the batch norm, the residual `in_proj`, the dropout and the
pad mask, and `:302-306` applies `content_head` to it, so the gradient of the CE reaches `in_proj`
and the BN, not only the head. The n_layers=2 rewrite (`mid_out`) is arithmetically the same
statements as before, one dropout call, same RNG order. `forward()` returns `_forward()[0]`, so the
head never enters the decode/posterior path. The implementer's grad-magnitude test
(`test_sae_emc`: head 1.08e2, in_proj 4.08e1, BN 4.74e-1, output projection untouched) was NOT
re-run here.

## 6. Masking and the frame-count assert (item 5) -- VERIFIED BY READING, one gap

`content_term.content_cross_entropy` masks with `valid_frame_mask(lens, T)` and divides by the valid
frame count (`content_term.py:139-150`); `train_steps/sae_emc.py:459-470` additionally refuses a
per-utterance `code_lens != feat_lens`. `MfccCodesHdfJob.run:637-651` asserts, unconditionally and
before writing, that every tag of the feature HDF has an MFCC and that its code count equals the L15
frame count at tolerance 0; `collect:711-715` re-asserts `tokens == frames` per split. Nothing skips
these. `test_mfcc_codes` exercises the frame rule end to end (0 mismatches over the synthetic set).
NOT CHECKED: `check_expected_utts` default and whether the config leaves it on.

## 7. FINDING (B): `test_mfcc_codes` fails in this env

`src/speech_llm/sae/emc/test_mfcc_codes.py:185`
(`test_kmeans_is_deterministic_at_a_fixed_seed_and_assign_matches_sklearn`):

    assert np.array_equal(c1, c2) and i1 == i2      -> AssertionError

i.e. two `fit_kmeans(x, num_clusters=8, seed=42)` calls in the SAME process returned different
centroids. Both implementer reports state this suite passes ("k-means determinism at seed 42").
Run: conda `speech_llm` python, PYTHONPATH = src:recipe:i6_models:returnn:tools/sisyphus, no
thread-count pinning. I did not diagnose the cause (MiniBatchKMeans is thread-sensitive; the
implementer may have run under a different `OMP_NUM_THREADS`), so this may be environment-dependent
rather than a code defect -- but as it stands the claim "deterministic at a fixed seed" is not
reproducible, and `MfccKMeansJob` has no recorded fallback: if the job dir is ever cleared and
re-run, `lam3_ct`'s codes change under an unchanged job hash, and the arm's monitors stop being
comparable to the first run's. Failure path: a re-run codebook, silently different, at the same id.
Everything else in the suite (frame rule, dump, codes-job agreement) ran clean up to that point.

## 8. NOT CHECKED (turn limit)

1. `prefix_lm.model.train_steps.test_sae_emc` and `sae.emc.test_emc_train_jobs` were not executed;
   the bit-identity, parameter-count and checkpoint-crossing claims for `lam_content = 0` /
   `lam_ent = 0` rest on the config diff of section 2 and on reading, not on those tests.
2. `eval_jobs.posterior_dump` was NOT read: whether it actually accepts and uses the new `net_args`
   (`emc_train_jobs.subepoch_reads:1537-1575`) is unverified. If it ignored `net_args`, every
   sub-epoch read of `lam3_ct` would die on `content_head.*` as unexpected keys -- worth one grep
   before that arm is funded. The `lam3` reads are unaffected (`net_args=None`).
3. `_load_state(allow_missing_prefix="content_head.")` was read, not exercised.
4. The test files themselves (863 + 541 new lines) were not reviewed line by line.
5. `config_sae_4a_s3b_pack_v1.py` / `pack_jobs.py` (commit 69efa16) were out of scope and not read;
   `lam3_ct` is not in the pack config, so the second packed node is still unwired.

## 9. Beyond the intended delta (item 7)

Nothing found beyond the dispatched deltas in the six wiring files: the config diff of section 2 is
complete and every key traces to the dispatch (lam_ent 0.1/0.3, ent_ramp_epochs 4 = the dispatched
ramp, rate_hinge, lam_content 0.3, content_k 64 = Liu Table 2, content_layer 1 = the stated tap,
MFCC_SHARDS_TC100 = 4 = the clean tc100 AvStatesJob's shard count). The `lam_ent` UNIT choice
(normalized entropy H/ln C rather than nats) is disclosed in the implementer report and matches the
dispatch's own wording; it is the difference between 8 % and 30 % of dev_loss_emc 1.23 at a diffuse
posterior, so the arms' effective strength is the ~10 % calibration, not 3x it.
