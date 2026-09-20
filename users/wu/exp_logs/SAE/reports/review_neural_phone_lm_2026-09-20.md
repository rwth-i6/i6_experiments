# Review: SAE 4a Step 0b, the neural phone LM row (739d9ed, e971603) -- 2026-09-20

Read-only review, fresh context. Verdict: **DONE_WITH_CONCERNS**. Nothing found that makes the
submitted jobs produce a wrong number; two items change how the numbers may be read, one of them
a pre-registration mismatch that bears on the Step 0b decision.

Files read: `sae/emc/neural_phone_lm.py` (whole), the two `prior_gap.py` diffs (whole), the config
diff, `SAE_4A_prior.md` "Step 0b", the implementer report, `sae/emc/prior.py`
(`count_ngrams`/`fit_prior`/`per_token_log_probs`/`perplexity`), the banked
`PriorGapAnalysisJob.2RkbKYl0v1XK` json + per_utt, `PhoneNgramPriorJob.RtzbESkOedsT/output/prior.stats.txt`.

## 1. Leakage -- clean, with one disclosure

* The two halves come from ONE call of `prior_gap.window_line_flags` and are asserted disjoint by
  index before either text is written (`neural_phone_lm.py:715-720`). The flag rule
  (`prior_gap.py`, `window_line_flags`: `is_held = bool(toks) and held_stride and i % held_stride == 0`,
  counted = non-empty and not held) is byte-for-byte the rule of `prior.count_ngrams`
  (`prior.py:251-257`), including that an EMPTY line still consumes an index. So the neural LM's
  training lines are exactly the lines `lmplz` is fed, and its validation lines are exactly the
  lines the live trigram holds out.
* Confirmed against the live prior's own job: `PhoneNgramPriorJob.RtzbESkOedsT` was fitted on the
  same file the config pins as `WINDOW_PHN`
  (`text_sample/SampleLinesJob.orN768ARKwlt/output/text.phn.gz`), `lines_read = 1010000`,
  `lines_counted = 1000000`, `lines_held = 10000`, `held_stride = 101`. Both jobs resolve
  `n_window_lines = 1_010_000`, `held_stride = 101`, `sample_seed = 0` from the same defaults, so
  the split is the same object in all three places.
* No shared tokenizer statistic: the vocabulary is the fixed 40-phone `prior.PHONE2ID` plus BOS and
  PAD; nothing is estimated from text.
* Exact TEXT repeats between the halves are counted (`leakage_check`, `neural_phone_lm.py:524-547`)
  and banked, not repaired. That is the right call: the n-gram estimator it is benchmarked against
  holds out the same lines, so the comparison stays like-for-like whatever the count.
* DISCLOSURE (not a defect, but it must be stated wherever the number is quoted): best-epoch
  selection reads the SAME held lines the benchmark reports on
  (`neural_phone_lm.py:476-499` selects `argmin held_nats_per_token`; `prior_gap.py` benchmark
  scores the kept model on those lines). The reported held-out perplexity is therefore
  best-of-at-most-3-epochs on its own report set. Magnitude: ~10,000 lines / ~810 k tokens, three
  highly correlated checkpoints, and an epoch only counts as an improvement at >= 1e-4 nats, so the
  optimistic bias is of order 1e-3 nats -- well under 1 % in perplexity. It does not threaten a
  comparison against a trigram at ~9.56. A separate selection split is NOT needed for this to be a
  benchmark; the number must simply be labelled "best of <= 3 epochs, selected on these same
  lines". No label is involved anywhere, so `no-labels-in-training` is not touched.

## 2. Convention parity -- sound

* Position 0's input is BOS and its target is the line's first token
  (`neural_phone_lm.py:230-233`); there is no EOS target anywhere and no EOS term in scoring.
* Causal mask: `mask = torch.ones((t,t), dtype=torch.bool).triu(1)` (`neural_phone_lm.py:287`),
  True = may-not-attend, so position i sees 0..i only; `is_causal=True` agrees with it. No future
  leakage; the position-0 prediction sees only BOS. Correct as written.
* Padding: rows are left-aligned, so PAD is a right tail that no real position can attend to under
  the causal mask; PAD targets are dropped by `ignore_index=PAD_ID` and by `keep = (tgt != PAD_ID)`
  (`neural_phone_lm.py:309-315`). PAD_ID 41 is never a real phone id, so `keep` is exactly the real
  tokens: the sum runs over exactly the string's tokens.
* Loss/score computed in fp32 (`logits.float()`, line 310) even under bf16 autocast, and the scorer
  turns TF32 off, so a GPU score equals a CPU score.
* Denominator: `scores[NEURAL_LM_KEY][s][t] = (log p, len(sets[s][t]))` in `PriorGapAnalysisJob.run`
  -- the same denominator every other row uses.
* Length: the SCORER refuses a string over 512 tokens rather than truncating. Verified against the
  banked `per_utt.json` of `2RkbKYl0v1XK`: the longest scored string over all six sets is 315
  tokens (`private_sil`, `7601-291468-0006`), so the assert cannot fire and no row is truncated.
  Training lines over 512 ARE truncated and counted (`encode_lines:164-166`).

## 3. The held-line benchmark

* All four models score the SAME list `cut` (`prior_gap.py`, `held_line_benchmark`), truncated once
  at the neural model's 512 positions, pooled as total nats over total tokens, perplexity =
  exp(nats). `truncated_lines` is banked and rendered. The train job truncates its held lines at
  512 too, so `train_job_minus_recomputed_nats` is a real cross-check, not a definitional identity.
* The trigram row is the live Witten-Bell prior loaded from `prior_npz`, the same estimator whose
  own job reports the held perplexity -- so the row is a genuine reproduction.
* READ TRAP: the anchor to compare it against is **9.5611**, not 9.469.
  `PhoneNgramPriorJob.RtzbESkOedsT/output/prior.stats.txt` gives `held_ppl_order3 = 9.561056`
  (`held_bits_per_phone_order3 = 3.2572`) on exactly these 10,000 held lines of exactly this
  window. The 9.469 quoted in `SAE_4A_prior.md:57` is the 2026-09-15 estimate, a different fit on a
  different held set. A benchmark `order3` row near 9.56 is CORRECT and must not be read as a
  defect; a small residual difference from 9.5611 is expected from the 512-token truncation.

## 4. Training soundness

* FINDING (parameter count): the model is ~3.31 M parameters, not the pre-registered "5 to 8 M".
  With `vocab 42, d_model 256, 4 layers, FFN 1024, 512 positions` the count is
  10,752 (tok) + 131,072 (pos) + 4 x 789,760 (layers) + 512 (final norm) + 10,794 (output) =
  **3,312,170**. `SAE_4A_prior.md` "Step 0b" fixes both "about 4 layers, width 256" AND "5 to 8 M
  parameters"; those two clauses are mutually inconsistent at a 42-symbol vocabulary, and the
  implementer resolved the conflict silently in favour of the architecture. `neural_phone_lm.py:611`
  additionally states "a 5 M-parameter model", which is wrong by 1.7 M. Effect on the read: if
  Step 0b comes back `partial_proxy` or `not_the_scorer`, that verdict is about a model 40-60 %
  below the registered size band and does not discharge the pre-registered question; the
  `not_the_scorer` branch funds a GPU trie DP, so the mis-sizing has a spend consequence. The fit
  is minutes of GPU (see below), so re-running at the registered size is cheap. An
  `adequate_lexical_scorer` verdict is unaffected (a smaller model clearing the bar is a stronger
  result).
* lr schedule: linear warmup 500 steps then cosine to zero over the planned 3-epoch step count, laid
  out over the full 3 epochs whatever the time cap does -- a capped run is a prefix, not a different
  schedule. Correct.
* Token batching: shuffle, length-sort inside 8192-line windows, greedy fill to 32768 PADDED tokens,
  shuffle batch order; every line lands in exactly one batch (`token_batches`). Deterministic from
  `numpy.default_rng(seed)`.
* Abort / best epoch: `improved = held < best - 1e-4`; `best` starts at inf so epoch 1 always
  counts; a non-improving epoch sets `stopped_reason`, prints `ABORT: ...`, and the kept model is
  reloaded from the best epoch's CPU state dict. The time-cap branch is checked BEFORE the
  no-improvement branch, so a capped run is not mislabelled as an abort. Sound.
* Seed/dtype: `torch.manual_seed(0)`, cuDNN deterministic, benchmark off; bf16 autocast only on
  CUDA and only for the forward, with every REPORTED number recomputed in eval mode at fp32.
* Time cap: 81,559,944 counted tokens (from the live prior's stats) x 3 epochs / 32768 tokens per
  batch ~ 7.5 k steps for a 3.3 M-parameter model -- minutes on a GH200, far inside the 2 h training
  cap; the 4 h job rqmt also covers the two full `replay_window_texts` passes over
  `librispeech-lm-norm.txt.gz` (the banked v2 job did one replay plus three lmplz builds in 303 s).
  The analysis instance pays a second replay plus CPU scoring of ~810 k held tokens and ~690 k
  utterance tokens on 4 cores; that is minutes, inside its own 4 h.

## 5. The read rule, the row, the hash

* `neural_read` implements the spec's three branches with the bar recomputed from the run's own rows
  (`gap_3 + 2/3 (gap_escape - gap_3)`) and `bar_preregistered = 2.01` banked beside it. Checked
  against the banked v2 numbers: like_for_like `order3` 1.3928, `lexicon_escape` 2.3213 ->
  recomputed bar 2.0118, i.e. the spec's "1.39 + 0.62 = 2.01" reproduces. Clause 2 compares the
  neural gap on the `lexicon_strict` PAIRED SUBSET (609 utterances) against the strict row's gap on
  that same subset (2.5057), tolerance 0.3 -- the right pairing.
* The documented middle-branch assumption (clears the bar, does not track the lexicon -> partial
  proxy) is the only reading that covers that case and is recorded in the producing code, as the
  project rule requires.
* `STEP0B_KEYS` removes the neural row from `decision()`'s `any_more`, so Step 0's pre-registered
  table cannot move. Correct.
* v2 hash: `__sis_hash_exclude__ = {"neural_lm": None}` is the hash-NEUTRAL direction (a new
  parameter excluded at its default); `prior_gap.py` carries no `__sis_version__`. Indirect on-disk
  evidence agrees: after the manager was restarted at ~22:28-22:32 (the train job dir
  `NeuralPhoneLmTrainJob.Iv6P6YVPNWmB` was created at 22:32) the only `prior_gap` job dirs are still
  `2RkbKYl0v1XK` (v2, finished, `finished.tar.gz`) and `l0p0srBryKrs` (v1). A moved v2 hash would
  have produced a third, immediately-runnable dir. Not independently recomputed -- see UNCHECKED.

## UNCHECKED

* `test_neural_phone_lm.py` and `test_prior_gap.py` were not read; the implementer's "36 passed" is
  taken at face value and is not evidence for any claim above.
* `per_utterance_record` (`prior_gap.py:1054-1098`) was not read: whether the `neural_lm` row
  actually reaches `per_utt.json` is UNCHECKED (the module's own "every banked field is rendered"
  test is claimed to cover it).
* The render helpers `_fmt` / `_ordered_fields` / `_detail_table` were not read: "every banked field
  rendered" for the Step 0b block and the benchmark block is UNCHECKED.
* The v2 instance's hash was not recomputed from a sisyphus graph load; the evidence above is
  on-disk and indirect.
* `witten_bell_utt_log_prob` and `kenlm_utt_log_prob` were read and match the BOS / no-EOS / nats
  convention; the KenLM binaries and the `lmplz` invocation were NOT re-verified (unchanged code).

## Summary of defects, most severe first

1. `neural_phone_lm.py:116-122` (and the wrong claim at `:611`): 3.31 M parameters against the
   pre-registered 5-8 M band. A negative or middling Step 0b verdict would not discharge the
   registered question and would fund a GPU trie DP on an undersized model's failure.
2. `SAE_4A_prior.md:57` vs the live prior's own stats: the benchmark's trigram row will reproduce
   9.5611, not 9.469; quoting 9.469 as the expected value would manufacture a false discrepancy.
3. `neural_phone_lm.py:476-499`: held-out perplexity is best-of-<= 3 epochs selected on the same
   lines it is reported on. Bias ~1e-3 nats, < 1 % in perplexity; label the number, do not rerun.
