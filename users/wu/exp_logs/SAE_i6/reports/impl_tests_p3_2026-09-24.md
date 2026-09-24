# Priority-3 (data-side) tests, T3.1-T3.7: implementer report, 2026-09-24

Status: DONE_WITH_CONCERNS. Every requested test is written and passes. Nothing had to be marked
xfail, and no package code was changed. The concern is T3.7's rho check, which cannot be built as
the dispatch specified (see S8 below). I replaced it with checks that can be traced to banked
numbers, and those need a decision.

PKG = recipe/i6_experiments/users/wu/experiments/unsupervised_asr. Spec: reports/test_plan_2026-09-24.md,
sections 0, 2 (S8, S11) and 5.

## Files touched (tests only)

- `tests/test_data_vad.py`: 4 tests added (T3.1). The 2 existing tests are unchanged.
  - `test_rvad_silence_rule_subframes_2`: at 50 Hz, a frame is silence only when both of its 10 ms
    labels are non-speech. A partial tail is dropped, and the waveform reaches rVADfast as float32
    at 16 kHz. Integer and float labels are both checked.
  - `test_rvad_silence_rule_subframes_4`: a frame is silence when more than 50 % of its labels are
    non-speech, so 3 of 4 is silence and 2 of 4 is speech. The tail is dropped and the default
    number of subframes is 4.
  - `test_rvad_silence_shorter_than_one_frame`: returns an empty bool mask.
  - `test_prepare_blankfree_data_silence_rule_stubbed`: `rVADfast` is replaced in `sys.modules` by
    a fake. Three utterances have different lengths, and their masks are 3 frames shorter than,
    4 frames longer than, and equal to the feature stream. The test checks:
    - the fake is constructed with threshold 0.4;
    - a speech frame is labelled (1, 0), which also tests that one non-speech label is not enough
      for silence;
    - raw_index = `flatnonzero(~pad(mask, True)[:orig])`;
    - features and units equal the source rows at those indices;
    - the manifest's kept and original totals.
- `tests/test_w2v2_features.py`: 3 tests and one module fixture added (T3.2, S11). The fixture
  builds a tiny random `Wav2Vec2Model` (stable layer norm, 4 layers, hidden 16, conv_dim 8, all
  dropout and layerdrop 0, `mask_time_prob` 0.5), saved with `save_pretrained` to tmp. Nothing is
  downloaded.
  - `test_l15_tap_is_hidden_state_k`: `_build_l15_module(encoder_layer=2)` keeps 3 blocks, is in
    eval mode, and has no parameter with requires_grad. Its states equal `hidden_states[2]` of the
    full 4-layer model with `apply_spec_augment=False`, to within 1e-6 absolute. Two eval forwards
    are bit-identical. The output lengths equal the explicit conv-stack formula. The S11 power
    check: `module.train()` changes the states by more than 1e-3, so SpecAugment is on and only
    eval mode keeps it off.
  - `test_l15_encoder_layer_range`: `encoder_layer=5` raises ValueError, and `encoder_layer=4`
    keeps all 4 blocks.
  - `test_zero_mean_unit_var_matches_feature_extractor`: equals
    `Wav2Vec2FeatureExtractor(do_normalize=True, return_attention_mask=True)` per utterance and
    batched, to 1e-5 (the float32 tolerance in §0). Padding comes out as 0.
- `tests/test_data_speaker.py`: new, 8 tests (T3.3, T3.4).
  - T3.3, `fit_speaker_pca` on data of 200 x 12 with dim 3, checked against the SVD:
    - the components are orthonormal and match the top right singular vectors up to sign;
    - the sign rule holds and the components equal the sign-fixed SVD matrix (1e-9);
    - the scales, the explained-variance ratio and the mean are checked;
    - row order and a global sign flip do not change the components;
    - `apply()` gives float32 with mean 0, std 1 and identity covariance (1e-5);
    - save and load round-trip.
  - T3.3, `SpeakerEtaJob` on tiny pickles (2 fit shards with unsorted keys, a dev store with one
    non-gold tag, and gold.json):
    - the tags are sorted and unique, and cover exactly fit plus gold dev;
    - each row equals `apply()` of an oracle PCA (1e-6);
    - the saved PCA and eta.json are checked;
    - the assertion that the fit set contains no dev utterance fires;
    - the fit-count guard fires.
  - T3.4: `lookup_eta` returns rows in the order of the batch tags, repeats included, and an
    unknown tag raises KeyError. Inside `train_step`, `build_segment_table` is wrapped to record its
    input and then stop the step.
    - Swapping the tags on the same units swaps the eta rows, and the tables change.
    - Permuting the whole batch permutes the tables (rtol 1e-5).
    - Each row equals `build_segment_table` on its own units and its own tag's eta.
- `tests/test_lm_phone_text.py`: 2 tests and a fixture added (T3.5); `import math` added.
  `PhonemizeWithSilJob` runs at its defaults (sil_prob 0.5, surround, seed 0) on 20,000 random
  lines of 1-phone words.
  - `test_sil_insertion_statistics`: the line edges are always SIL, there is no double SIL, and the
    non-SIL tokens are exactly the words. The internal-boundary SIL count is within 4 sigma of 0.5
    (n > 100k boundaries).
  - `test_count_ngrams_maps_sil_token`: `unigram[SIL_ID]` equals the number of `<SIL>` tokens, and
    bigram(BOS, SIL) and trigram(BOS, BOS, SIL) equal the number of lines. `[SIL]`, `sil` and `SIL`
    map to the same id, and an unknown token raises KeyError.
- `tests/test_lm_official_lexicon.py`: new, 7 tests; the first two run on .txt and .gz (T3.6).
  - `read_official_lexicon`: stress digits are stripped. Entries with an out-of-inventory phone,
    with SIL, with no pronunciation, or duplicated after stripping are dropped, and each is counted
    exactly once.
  - `official_prons`: restricted to G's vocabulary. `<s>` is dropped as a marker. Several
    pronunciations per word are kept, and there is no pronunciation for `<s>`, `</s>` or `<UNK>`.
  - `derange_official_prons`:
    - every new block is one old block, and the pronunciation multiset is unchanged;
    - there is no fixed point, and the permutation is one Sattolo cycle;
    - the result is deterministic, and seed 1 is also a derangement;
    - `kept_by_homophony` counts correctly on a homophone example.
- `tests/test_artefacts.py`: new (T3.7).
  - 2 unmarked tests:
    - the rho literal equals 2.7 x 2,784,159,269 / 778,025,128 (1e-9 relative);
    - `build_train_config`'s model args carry 9.6619373279.
  - 7 tests marked `artefact`, 2 of them also `slow`:
    - prior tables: shapes, phone order, finite values, rows summing to 1 (1e-9),
      P(SIL | BOS, BOS) > 0.9, and the lines_read and held values in meta;
    - prior.json `held_ppl_order3` rounds to 9.561, the value banked in PhoneNgramPriorJob.RtzbESkOedsT;
    - (slow) the held-out ppl recomputed from the npz and the held lines of the window equals 9.561
      at 3 decimals and equals prior.json to 1e-9;
    - the rho numerator on the full phone corpus;
    - (slow) rho estimated on the window;
    - the VAD manifest equals BANKED_VAD_COUNTS, with no short utterances;
    - eta.npz: dim 16, finite, sorted, 28,539 + 5,567 tags, covering every VAD raw_index tag of
      train, dev-clean and dev-other.

  Files that are not registered are reached from registered outputs:
  - prior.json and eta.npz through the job output directory that the registered symlink points into;
  - the window through the prior's `meta["corpus"]`;
  - the phone corpus through the first line of the window job's stats.txt.

## FINDINGS

- **S8, spec conflict (not a code defect).** The dispatch asked to recompute
  2.7 x phones-per-word from the prior window and match 9.6619373279 to 1e-9. That cannot be done:
  - The banked provenance of the literal is the full boundary-free T_phi: 2,784,159,269 phones over
    778,025,128 words (exp_logs/SAE/reports/review_rate_term_2026-09-15.md V7), not the
    1,010,000-line window.
  - The window is SIL-augmented with p = 0.5 and records no word count.

  What I implemented instead:
  - the literal against the banked counts (passes);
  - an exact artefact check of the numerator from the phone corpus job's stats.txt: tokens minus
    sil_tokens = 2,784,159,269, with lines 40,418,261 in and 39,630,169 out;
  - a window estimate whose word count is lines + 2 x internal SILs, which is unbiased under
    p = 0.5, compared with the literal within 4 delta-method standard errors.

  The 4-sigma bound is my choice, following T3.5's convention. On a synthetic corpus the estimator
  came out 1.43 SE from the true ratio. The words denominator (778,025,128) stays unverified,
  because it needs the LM text plus a replay of which lines were dropped for OOV.
- **S11, confirmed and characterised.** On the tiny model, a train-mode forward of the tap applies
  time masking (the states change by more than 1e-3). Eval mode is the only guard.
  `_build_l15_module` calls `eval()`, and RETURNN's `forward_with_callback` calls
  `self._pt_model.eval()` (returnn/torch/engine.py:1541), so the dump path is safe as long as
  nothing calls `train()` on this module.
- Context, not a finding of mine: `data/vad.py` gained `counts_report_only` and
  `write_counts_report` during this session, edited by someone else. The T3.1 tests pass against
  the current file. Under an ffmpeg accept label the VAD counts may legitimately differ, and the
  artefact test `test_vad_manifest_is_the_banked_counts` will then fail by design.

## How to run the artefact tests

After the P0 input graph has produced `output/sae/4a/...`:

    cd $PKG; PATH=/work/asr4/hwu/conda/envs/sae/bin:$PATH PYTHONPATH=<setup>/recipe:<setup>/recipe/returnn:<setup>/sisyphus \
      SAE_ARTEFACT_DIR=/u/hwu/setups/librispeech-960/2026-09-24-unsupervised/output python -m pytest -m artefact tests/test_artefacts.py

Add `-m "artefact and not slow"` to skip the two tests that read the 1.01 M-line window.

- If an output is missing, its test skips and names the missing file.
- If `SAE_ARTEFACT_DIR` is unset, the artefact tests skip.

On a fake artefact tree in scratch, every path resolved and each test failed only on its
comparison with banked numbers, as expected. This shows the plumbing works; it is not a result.

## Checks run

- My six files together: 38 passed and 7 skipped (the artefact tests) in 18 s. The largest cost
  is 11.7 s of fixture setup, mostly importing transformers.
- Full suite (`pytest tests`, sae interpreter): 451 passed, 14 skipped, 12 xfailed, 0 failed in
  62 s (log: scratchpad p3tests/full_suite.log). All 12 xfails are in files owned by other
  implementers (prior.py:246, lexlat_k2 S3/S4, lattice `_logmm`, prior_history, null_log_q).

## Undetermined / for the orchestrator

- Whether the S8 replacement above is acceptable, or whether rho should get a re-derivation job
  that counts words from the LM text.
- The 4-sigma tolerance of the window estimate is my choice, following T3.5's convention.
- The perplexity value 9.561 was taken from the banked PhoneNgramPriorJob.RtzbESkOedsT prior.stats.txt
  (exp_logs), which is more precise than the plan's "approximately 9.56".
