# Oracle tests for the k2 lexicon term (T1.18-T1.23): implementer report, 2026-09-24

Status: DONE_WITH_CONCERNS. The tests are complete and the suite is green. S3 is confirmed, S4 is confirmed only for the "all" placement, S5 is confirmed, and S6 is refuted. No package code was changed.

PKG = recipe/i6_experiments/users/wu/experiments/unsupervised_asr. Specification: reports/test_plan_2026-09-24.md, sections 0 and 2 (items S3-S7) and tests T1.18-T1.23.
Interpreter: /work/asr4/hwu/conda/envs/sae/bin/python (k2 1.24.4.dev20260924, CPU), run from PKG with PYTHONPATH=SETUP/recipe:SETUP/sisyphus.

## Files (all new, under tests/ only)
- `tests/k2_oracle.py`. This holds the fixture: phones A, B, C, SIL; words a, ab, bc, x, y; escape probabilities [0.3, 0.3, 0.2, 0.2]; sil_prob 0.5. It writes normalised order-3 and order-4 ARPA files whose back-off weights are derived exactly. It also provides:
  - `ArpaScorer`: reads an ARPA into dicts and applies the textbook back-off recursion.
  - `csr_walk`: an independent walk of the CSR tables.
  - `enumerate_paths`: the explicit L∘G path enumerator of T1.19. It supports the word_boundary and "all" placements, escape and strict lexicons, and any lexicon mapping. Each path records its words, `<unk>` count, whether two escapes were adjacent, and its trailing back-off moves.
  - `proper_paths`: the same parse space priced by the exact back-off LM, one route per word.
  - `brute_force`: sums over all a in 4^T in float64 torch, so autograd gives the exact gradient.

  The oracle uses the package only to build the fixture's input files through the production path: `parse_arpa_word_lm`, `build_trie`, `derange_pronunciations`, `save_resources` and `lexlat_k2.main(["build-hlg", ...])`.
- `tests/test_model_lexlat_k2.py`: T1.18-T1.22, with 32 passed and 3 xfailed.
- `tests/test_lm_prune.py`: T1.23, with 34 passed and 2 xfailed.

## Verdicts
- **S3: CONFIRMED.** After `prune_lm_tables`, contexts of order 1 and 2 still sum to 1 within 1e-6. Contexts of order 3 and above do not. The largest deviation |Σ−1| was:

  | order | theta 0.5 | theta 2 | theta 5 |
  |---|---|---|---|
  | 3 | 5.13e-3 | 1.95e-2 | 1.26e-4 |
  | 4 | 5.77e-3 | 1.95e-2 | 1.26e-4 |

  The cause is at lexlat_k2.py:715-725: the back-off weight is re-derived against the unpruned lower-order lookup, while lower-order arcs are pruned in the same pass. The docstring claim at lexlat_k2.py:679-685 is therefore false. The value 5.13e-3 was checked by hand.
  - The kept-arc criterion itself matches an independent recomputation at every theta.
  - theta 0 returns the tables unchanged.
  - Magnitudes are for the fixture only, not for the banked LMs.
- **S4: REFUTED for word_boundary, the placement of the default arm; CONFIRMED for "all".** I replayed compile_hlg's stages over all 484 strings y with |y| ≤ 5:
  - word_boundary: the totals before and after `k2.remove_epsilon` agree with each other and with the oracle to within 2.7e-7.
  - "all": the total before epsilon removal matches the oracle to 5.8e-8. After removal it drops for 474 of 484 strings, by up to 2.717 nats per string. The cause is at lexlat_k2.py:874: the removal is tropical, so it merges parallel epsilon routes (back off then exit, versus exit then back off).
  - Consequence: the compiled "all" HLG undercounts its own path semantics by 0.26-0.75 nats per utterance on the T1.19 batch (tau 1 and 2). Every graph banked with the "all" placement had this.
- **S5: CONFIRMED.** L_lex ≤ 0 is not enforced.
  - The largest log W_tau(y) over |y| ≤ 5 was −0.182 at tau 1, +0.685 at tau 2 (17 of 484 strings are positive) and +4.85 at tau 8 (471 of 484 are positive).
  - The runtime's own L_lex on a posterior peaked on those strings was +0.685 at tau 2 and +4.848 at tau 8.
  - Adjacent escape spans carry at most 3.4% of W_1(y) (median 0.07%), and 0.01% of Z_HLG on a random posterior with T = 5.
- **S6: REFUTED.** With unsorted lens [2, 5, 3, 4]:
  - Totals agree within 1e-9 and gradients within 1e-7 for chunk_seqs 1, 2 and 16, for a permuted batch and for a length-sorted batch, at tau 1 and 2.
  - All of them equal the oracle.

## Other findings
- **Over-count against the proper LM (T1.19e), recorded.** Per token, the over-count has a median of 0.28 nats and a maximum of 1.03 nats (fixture back-off weights of 0.4-0.8). About half of it (median 0.148 nats per token) comes from trailing back-off moves before acceptance. Every G state is final at 0 and the #0 loop sits at LOOP, so each utterance's total gains a factor of (1 + α(s) + α(s)α(bo(s)) + …). This is structural in g_fsa / lexicon_to_fst (lexlat_k2.py:560, 790-794). Each utterance pays it once, so on long utterances it is small per token.
- **`lexlat.lm_score` with levels=3 on an order-4 table (T1.23a).** It returns log p = 0 for a word that has only a unigram arc (true value −4.08). This is the behaviour the docstring at lexlat.py:512-515 documents, pinned as passing.
- **The two order-3 parsers agree.** `parse_arpa_word_lm` and `parse_arpa_lm` produce bit-identical tables, and both match the scorer.

## Verified by the tests (1e-5 unless noted)
- **T1.18**
  - log Z_H on unsorted lens equals Σ_t logsumexp(e_t/tau), and the gradient equals softmax/tau (1e-6).
  - With min_frames 2 the total equals the brute force over frame strings whose runs are at least 2 frames long.
- **T1.19**
  - Z_HLG and its gradient equal the oracle at T ∈ {3, 4, 5} and tau ∈ {1, 2} for the word_boundary, strict and shuffled graphs.
  - "all" ≥ word_boundary holds in both the graph and the oracle.
- **T1.20**
  - The step's term, its gradient, `term_mean`, `expected_words` and `expected_escape_words` equal the oracle.
  - An empty row is counted. It is left out of the numerator, the denominator stays keep.sum(), and the row's gradient is 0.
  - With more than 10% of rows empty and no `learning_rates` file, the step raises RuntimeError and writes the marker JSON.
  - `check_bed` refuses d_min 3, stride 1, ctc and a determinized build. `check_expected_build` refuses theta 5 against a theta-0 build. `lam` and `active` equal `lexlat_lambda`.
- **T1.23c**
  - Compaction leaves every reachable lookup and its successor unchanged. The null state stays 0 and the keys stay sorted.

## Assumptions
- **T1.22 "before" graph.** "Before" means LG with the disambiguation tokens relabelled to 0 but with its epsilon arcs kept, intersected with `treat_epsilons_specially=True`. Otherwise a linear token FSA could not match it.
- **Beams in the T1.20 empty-lattice tests.** These tests use the wide operating point (beams 1e3, max_active 10000), so the scored rows can be held to the unpruned oracle. At production beams, pruning moves the term by 2.6e-4 relative.
- **"Per token" in the T1.19e over-count** means |y| phone tokens, SIL included.

## Checks
- New files: 66 passed and 5 xfailed (strict) in 3.9 s.
- Full suite, run once: 394 passed, 5 skipped, 11 xfailed, 0 failed (log at /work/asr4/hwu/tmp_probe/tests/full_suite_k2impl.log).
- The 6 xfails outside my files belong to the other implementers: test_lm_phone_prior ×2, test_model_lattice ×4.

## Proposals for the project documents
- **SAE_i6_ref_objective.md §10.** Record S3 CONFIRMED and S4 CONFIRMED only for "all" (the default arm's word_boundary graph is exact). Record that L_lex > 0 is reachable at tau ≥ 2.
- **Possible S3 fix, left to the owner.** Re-derive the back-off weights bottom-up from the pruned lower order. The default arm (official 4-gram at theta 5) is affected.
