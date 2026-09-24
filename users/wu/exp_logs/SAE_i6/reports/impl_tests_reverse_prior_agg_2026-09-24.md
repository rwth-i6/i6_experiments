# Oracle tests T1.9-T1.13, T1.16, T1.17: implementation report (2026-09-24)

Status: DONE_WITH_CONCERNS. All the planned tests are written. One package defect was found
(FINDING 1), pinned by two strict xfails. The same defect also breaks the other implementer's T1.6
oracle. No package code was changed.

PKG = recipe/i6_experiments/users/wu/experiments/unsupervised_asr. Spec: reports/test_plan_2026-09-24.md sections 0, 2 and 3.
Invocation (the same one the existing tests use): from PKG,
`PYTHONPATH=<setup>/recipe:<setup>/sisyphus /work/asr4/hwu/conda/envs/sae/bin/python -m pytest -q tests/...`

## Files
- `tests/test_model_reverse.py` (new): 12 tests, 12 passed, about 1 s.
  - T1.9 (a): position_bounds partition and `floor(r n/d)`.
  - T1.9 (b): segment_scores against the explicit sum, for n_pos in {1, 3, 4}.
  - T1.9 (c): duration support and values against an independent masked log-softmax.
  - T1.9 (d): build_segment_table.
  - T1.10: 1320 (y, S) cases for each of n_pos 1 and 3, against the explicit enumeration of
    compositions, plus the feasible() rule.
  - T1.10: padded batch equals single runs; gradients on every parameter against the oracle's
    autograd.
  - T1.10: the infeasible case raises in assert_finite_sufficient_stats; evaluate() rows are checked.
  - T1.11: lattice versus reverse versus prior identity, plus the tau = 2 record.
- `tests/wb_oracle.py` (new): a dict/Counter interpolated Witten-Bell with two-BOS padding, written
  without numpy or package code.
- `tests/test_lm_phone_prior.py`: tests appended only; the existing 5 are unchanged. The file now
  has 11 passed and 2 xfailed (the new tests account for 6 passed and 2 xfailed).
  - T1.12: every entry of the log_uni, log_bi and log_tri tables at 1e-12.
  - T1.12: perplexity at orders 1 and 2 passes. Order 3 is an xfail. A separate order-3 check on
    held lines of two or more tokens passes.
  - T1.13: the history walk for the trigram and bigram on 500 strings, with lengths 1-8 at the
    bigram and 2-8 at the trigram. The one-token trigram case is split into an xfail.
- `tests/test_model_agg.py` (new): 4 tests, 4 passed, about 1.8 s.
  - T1.16: for K in {3, 4}, values at 1e-12, the diagonal, padding, and gradients at 1e-10.
  - T1.17 (a)-(c): three train steps with decay 0.99. Checked: the values, the EMA buffers, that
    buffers do not move in eval mode or with update_ema=False, and the gradient: step 1 is the full
    KL gradient, steps 2 and 3 are (1-d) times it.
  - T1.17 (d): the targets that SaeBlankfreeModelV1 itself builds.

Tolerances are the plan's. Reading of "1e-4 rel" for the T1.10 gradients: per parameter tensor,
max|g - g_oracle| <= 1e-4 * max|g_oracle|. T1.17 (a)-(c) use the module's float32 target buffers
cast to float64 as the oracle's targets, and 1e-10 on the value.

## FINDING 1: model/prior.py:246
`per_token_log_probs(seq, order=3)` with len(seq) == 1 returns TWO terms. The line
`h2 = [BOS, BOS] + ids[:-2]` has length 2 and broadcasts against `h1` and `ids`, which have length 1.
As a result, `log_prob` and `perplexity` count log P(y0 | BOS, BOS) twice for any one-token sequence.

Hand check: prior fitted on [[0, 1], [2]]. `per_token_log_probs([5], 3)` returns
[-5.768321, -5.768321], and `log_prob` = 2 * log_tri[BOS, BOS, 5]. Orders 1 and 2 are correct, and
so are sequences of two or more tokens.

Pinned by strict xfails:
- `test_perplexity_matches_oracle_on_held_lines[3]`: code ppl 1.72423, oracle 1.64457.
- `test_prior_table_layout_one_token_strings_trigram`.

Reach: the lattice reads tables, not this scorer, so the training loss is unaffected. Affected are
`lm/phone_prior.py:137` (the held_ppl_order3 statistic) and `lm/prior_gap.py:46` (log_prob of a
decoded string), both only on one-token sequences.

The defect also breaks the other implementer's T1.6. `tests/test_model_lattice.py:553` builds its
oracle with `per_token_log_probs(..., order=3)`, and its latents include 40 one-token strings. On an
unchanged file (md5 3ea4021e...), T1.6 fails with the package scorer (gap 0.0029 nats) and passes
when the scorer is patched in memory to return one term per token (scratch script, no file edited).
The lattice therefore agrees with a correct oracle at T1.6; the discrepancy comes from the oracle's
scorer.

## Recorded measurements
- T1.11 at tau = 2 (S 10, T 4, W 60, trigram, beta 1):
  - Code log Z = -24.414381; the log Z of NOTE l.155's form is -24.476614. Gap (code - NOTE) =
    +0.062233 nats. The sign agrees with sum_seg p^(1/2) >= (sum_seg p)^(1/2).
  - The tau = 1 identity holds to 1.6e-6 against a 1e-4 tolerance.
- T1.17 (d) (toy Witten-Bell prior; the test corpus has repeats and SIL runs):
  - The bigram target's projection removes a diagonal mass of 0.158.
  - The unigram target is NOT projected. Against the run-collapse unigram uni[k](1 - p(k|k))
    renormalised: L1 = 0.2005, KL = 0.0457 nats.
  - The row marginal of the projected bigram differs from text_uni by the same L1 = 0.2005. The two
    targets are therefore mutually inconsistent by exactly this amount.
  - A production-scale value needs the banked prior.npz. It was not computed here.

## Checks
- Mutation checks (scratch scripts, not committed). Each of these made the tests fail, so the
  oracles are not vacuous:
  - wrong position bucketing;
  - duration log-probabilities scaled by 0.9;
  - a +1 on the Witten-Bell type count;
  - EMA decay swapped;
  - full-weight EMA gradient;
  - diagonal kept in expected_run_counts.
- T1.11 cannot catch a defect that the lattice and the reverse model share, such as the scaled
  durations. That is inherent to a cross-implementation identity.
- Full suite (log /work/asr4/hwu/tmp_probe/tests/full_suite_reverse_prior_agg.log): 295 passed,
  5 skipped, 2 xfailed, 3 failed, in 35 s. All 3 failures are in the other implementer's
  tests/test_model_lattice.py: test_t1_4c_sil_only_is_one_token, test_t1_5_z_zero_and_padding_invariance
  and test_t1_6_production_config_through_train_step (explained above).
- My non-slow tests total about 4 s. No slow marker was needed.

## Undetermined / notes
- A production-scale number for the T1.17 unigram-projection record needs the banked prior.
- T1.9 (b) checks only the reachable cells (s + d <= S). By its docstring, the code fills the other
  cells with 0.
