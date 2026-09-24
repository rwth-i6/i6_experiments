# Review: independence, semantics and strength of the P0 oracle tests (G0.V), 2026-09-24

Verdict: PASS_WITH_NOTES (DONE_WITH_CONCERNS). No oracle imports the function it checks, and no
oracle reuses a helper that shares that function's logic. For the loss terms, every oracle encodes
the objective note, including the deviations recorded in section 10: S1 (joint tempering), S2 (the
band holds at every state), the SIL split and S7 (the k2 term).

Three constants and one piece of plumbing are not pinned: mutations of them pass the whole suite.
One code-vs-note difference (S9, the agg targets) is asserted as correct without being recorded in
section 10. None of these gives a wrong number today, because the current code values are right by
reading. The runs are what a later regression of them would slip through.

Scope. PKG = recipe/i6_experiments/users/wu/experiments/unsupervised_asr. Read: the objective note
(including section 10), SAE_i6_P0.md, test_plan_2026-09-24.md, the five impl_tests_* reports and
impl_p0_amendments, every oracle module, and the loss-term tests. The repo was not edited.

## What I ran

All runs used the sae python and the setup PYTHONPATH; logs are in the session scratchpad.

- **Full suite:** 495 passed, 14 skipped, 12 xfailed, 0 failed (68 s). This matches SAE_i6_P0.md.
- **`--runxfail` on the 12 strict xfails:** all 12 fail at their stated assertion, with the stated
  magnitude. None is xfailing for an incidental reason such as an import or type error (table below).
- **Extra check, SIL split at binding bands.** The train step's lattice uses the history built for
  "ctc". I ran it on the T1.1 instance (W 1/2/3/40, bigram and trigram, tau 1/2/8, matmul) against
  `lattice_oracle` with `sil_split=True`, over 84 utterances.
  - log Z agrees to 4.4e-16 relative, and the gradients to 2.9e-15.
  - Why this was needed: T2.1 compares the SIL-split set only at T = 2 with a band that never binds.
    This check shows the step's lattice is exactly the SIL-split lattice when the band binds too.
- **Mutations** of a copy of the package in the scratchpad, one at a time. First the loss-term
  files were run, non-slow; then the full suite for the three survivors. The results are in the
  mutation table below.

### Mutation results

| mutation | failing tests | caught? |
|---|---|---|
| prior not divided by tau (`lattice._prior_term`) | 91 | caught |
| seg surrogate without 1/tau (`lattice_loss`) | 25, incl. T1.2 and T2.1 phi grads | caught |
| band offset off by one (`_band_index d_idx`) | 145 | caught |
| d_min read as 1 | 102 | caught |
| D_phone and D_SIL swapped | 124 | caught |
| SIL split also under blankfree (`same_nonsil`) | 92 | caught |
| rate divided by retained instead of original length | 2 (T2.1) | caught |
| E[N] includes SIL | 15 | caught |
| rate coefficient missing its factor 2 | 2 (T2.1) | caught |
| l_tau / T_rec instead of / S | 4 | caught |
| mean over all rows instead of kept rows | 4 | caught |
| tilt also on SIL | 9 | caught |
| agg: EMA blended at step 1 | 3 | caught |
| **frame_rate_hz 50 -> 50/3 (`lattice.py:209`)** | 0 (494 passed, full non-slow suite) | **missed** |
| **agg_bigram_weight default 1.0 -> 0.5 (`emc_model.py:280`)** | 0 (full suite) | **missed** |
| **train step passes `retained=original` to the k2 step (`train_step.py:155`)** | 0 (full suite) | **missed** |

## Findings

1. **The rate clock is read from the code.**
   - Where: `tests/test_model_blankfree.py:419`, where the T2.1 oracle sets
     `rho = RHO_HZ / cfg.frame_rate_hz`.
   - The plan says rho = 9.6619373279 / 50. T2.2 (`:629-631`) does not pin `frame_rate_hz`.
   - Consequence: a wrong clock rescales the rate target by the same factor in the step and in the
     oracle, so all 494 non-slow tests pass (mutation M12). Today the value is 50.0, which is right.
2. **The agg weights are read from the code.**
   - Where: `tests/test_model_blankfree.py:391`, which uses `w.unigram_weight` and `w.bigram_weight`
     from `model.agg.cfg`.
   - T1.17 pins its own `AggConfig` at weight 1. T2.2 does not check the bed's `agg_*_weight`.
   - Consequence: a changed default passes the whole suite (M13). Today both weights are 1.0, which
     matches note §4.3 (an unweighted sum of the unigram and bigram KLs).
3. **The train step -> k2 plumbing is untested.**
   - Where: T2.3's fake step (`tests/test_model_blankfree.py:681-682`) records only epoch,
     temperature and dtype.
   - T1.20 exercises `LexlatK2Runtime.step` directly. Nothing checks the
     `feat_lens` / `retained` / `keep` that `train_step.py:154-159` passes to it.
   - Consequence: M14 normalises L_lex by the original length instead of S, and it passes the suite.
     That would change G0.R2's lexicon-term read (the k2lat_20_ma3000 arm) with every test green.
     The current code passes the right arguments.
4. **S9 is asserted as correct, not recorded.**
   - Where: `tests/test_model_agg.py:208-243` (T1.17 d).
   - The test asserts the code's targets: Witten-Bell smoothed tables, the bigram with its diagonal
     removed and renormalised, and the unigram not projected.
   - Note §4.3 defines c_text as "the unigram and bigram frequencies of the phonemised text".
     Section 10 does not list S9.
   - The resulting inconsistency (the projected bigram's row marginal against text_uni: L1 0.20 on
     the toy prior) is only printed. Its size at production is unmeasured.
   - The G0.V statement "agg ... targets match" is therefore a statement about the code's
     construction, not the note's. Either record S9 in section 10 or measure it on the P0 prior.npz.
5. **Minor: adjacent escape spans.**
   - Where: `tests/k2_oracle.py:400-409`.
   - The oracle lets a new `<unk>` span open directly after an escape exits. This mirrors the graph.
     The design (SAE_i6_ref_lexicon.md l.33) prices `<unk>` once per span.
   - Only the "S5" label in P0 Results covers this; the text of section 10 records only the sign and
     the tempering.
   - Size on the fixture: at most 3.4 % of W_1(y), and 0.01 % of Z_HLG.

## Per loss term

- **Lattice l_tau and its gradient** (T1.1-T1.7).
  - Independent: yes. `lattice_oracle` imports nothing from PKG. The history rows, the band, the
    collapse and the SIL readings are all its own.
  - Semantic basis: note §4.1 with §10 S1 and S2. The SIL split is pinned by a strict xfail at T1.6.
  - Strength: the grid binds the band (W 1-3) and D (3 against 5). It covers tau 1/2/8, bigram and
    trigram up to 4 tokens, padding, and fp32. The tolerances are 1e-10 and 1e-9. All six structural
    mutations were caught.
  - Weakest spot: the production width (K 40, |h| 1681) is checked against the oracle only at T = 2,
    where the band and D never bind. At production size there is only self-consistency (T1.7).
- **Reverse DP** (T1.9-T1.11).
  - Independent: yes. It uses explicit compositions, its own masked duration softmax and its own
    buckets.
  - Semantic basis: note §1 (d_min 2, D_k, bucketed emissions).
  - Weakest spot: the emission head `emission_log_probs` is read, not re-derived. The tolerance is
    1e-4 relative (fp32).
- **Witten-Bell prior** (T1.12-T1.13).
  - Independent: yes, a dict/Counter oracle.
  - Every table entry, including the BOS and unseen rows, matches to 1e-12. The history walk is
    checked against h2*41+h1.
  - Weakest spot: the fixture sees only 4 of the 40 types.
- **Rate term and its FD surrogate** (T1.14, T1.15, T2.1).
  - Independent: yes. E[N] and the exact dE[N]/dlog_q come from the enumeration.
  - Semantic basis: note §4.2.
  - Weakest spot: the clock (finding 1). The bound at the bed's eps is loose (5e-2 against a measured
    1.9e-3), but T2.1 pins the step to the oracle's own FD at eps 0.25 to 1e-6.
- **Agg term** (T1.16, T1.17).
  - Independent: yes. Run counts come from path enumeration, and the EMA from a hand recursion.
  - Weakest spot: the targets are the code's own (finding 4), and the bed's weights are not pinned
    (finding 2).
- **Train-step assembly** (T2.1).
  - Independent: yes, for l_tau, rate and agg over the SIL-split latent set.
  - Pinned: the scales 1/3/0.1, the total, the kept-row means, the theta gradient (logit projection
    and pull-back) and phi's gradient (l_tau only).
  - Weakest spot: the instance is tiny (T = 2). The constants are read from code (findings 1 and 2).
- **k2 term** (T1.18-T1.22).
  - Independent: yes. The oracle is an explicit L∘G recursion plus a textbook ARPA scorer. The G
    tables come from the production parser, but T1.23a checks them against the ArpaScorer.
  - Semantic basis: SAE_i6_ref_lexicon.md B4 plus §10 S7 (a)-(d).
  - Weakest spot: the step-level plumbing (finding 3).

Shared helpers used only to build inputs are harmless:
- `L.build_prior_history` (DP input);
- `build_segment_table`, whose seg input is pinned by T1.9 d and T1.6;
- the npz built by `parse_arpa_word_lm`, validated by T1.23a;
- `_lg_stages`, which replays compile_hlg for T1.22, with "before" compared against the oracle.

## Strict xfails (12), each failing for its stated reason

| # | test | pinned code behaviour | measured | recorded |
|---|---|---|---|---|
| 1 | test_model_lattice T1.4c `[matmul]` | `_logmm` floor at `lattice.py:673` | Z = 0 row not flagged | P0 Results |
| 2 | test_model_lattice T1.5 `z_zero_row[matmul]` | same | same | P0 Results |
| 3 | T1.6 `log_z_vs_run_collapse_oracle` | SIL split (`emc_model.py:367`, `blankfree_model.py:481`) | 3.9e-8 against a tolerance of 2.2e-9 | §10 and P0 |
| 4 | T1.6 `model_history_is_the_blankfree_one` | same | table mismatch | §10 and P0 |
| 5 | test_lm_phone_prior ppl `[3]` | one-token double count at `prior.py:246` | ppl 1.724 against 1.645 | P0 Results |
| 6 | test_lm_phone_prior one-token walk | same | two terms instead of one | P0 Results |
| 7-8 | test_model_lexlat_k2 T1.19 `[all-1, all-2]` | S4, tropical remove_epsilon | down 0.26-0.75 nats per utterance | §10 (d) |
| 9 | T1.22 `[all]` | same | down 2.717 nats per string | §10 (d) |
| 10-11 | test_lm_prune T1.23b `[3-0.5, 4-5.0]` | S3, pruned G unnormalised | 5.1e-3 and 1.3e-4 | §10 (c) |
| 12 | T2.4 null `exact_log_40` | float32 -log 40 at `blankfree_model.py:617` | 2.6e-9 relative | P0 Results p2 |
