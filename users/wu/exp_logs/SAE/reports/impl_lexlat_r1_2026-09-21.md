# Implementation report -- SAE 4a lexlat, ROUND 1 (2026-09-21)

Scope: the lexicon-inside-the-lattice DP core, its resource build job, its tests, and the two
pre-funding probes (E-1 CPU, E0 GPU census with the two kill reads).  Training integration, the
efficiency probe E1 and the pack config are round 2 and are **not** in this round.

Spec: `SAE_4A_lexlat.md` Designs 1, 2 (incl. amendment 3), 4, 6, 8, the design-review amendments,
and the orchestrator note on the probe checkpoint (`ctrl_50` ep10 wherever the Design writes
`ctrl_20 ep10` for a probe).  Supporting: `reports/survey_lexicon_scorer_2026-09-20.md`,
`reports/design_review_lexlat_2026-09-21.md`.

## Files

All NEW modules; no banked module was edited (`lattice.py`, `blankfree*.py`, `prior_gap.py` are
untouched -- verified by `git status`, which shows them unmodified).

| file | what it is |
|---|---|
| `recipe/2025-10-speech-llm/src/speech_llm/sae/emc/lexlat.py` (1712 lines) | the DP core |
| `.../sae/emc/lexlat_jobs.py`  (1124 lines) | `LexiconTrieBuildJob`, `LexlatEquivalenceProbeJob` (E-1), `LexlatCensusJob` (E0) |
| `.../sae/emc/test_lexlat.py` (591 lines) | the five Design 8 tests + the gradient check |
| `.../prefix_lm/sis_recipe/exp2025_11_06_speech_llms/librispeech/configs/config_sae_4a_lexlat_probes_v1.py` | registers the three jobs |
| `/e/project1/spell/wu24/2026-07-13_unsupervised/config/sae_4a_lexlat_probes.py` | the workspace shim (`run = py`) |

Commit: **`319ce90`** on `haotian_modality_matching_jupiter` in the `2025-10-speech-llm`
checkout (the four tracked files, staged by explicit path; NOT pushed).  Two unrelated working-tree
entries of another session (`config_sae_1g_v1.py`, `config_sae_3e1_d6_swap_cont_v1.py`) were left
untouched and unstaged.  The workspace shim `config/sae_4a_lexlat_probes.py` lives outside both
checkouts and is therefore not under version control; this report is untracked in the
`i6_experiments` checkout (the dispatch named only the speech-llm checkout for commits).

## `lexlat.py` -- what it contains

* `build_trie` -- flat int32 `child[node, phone]`, `word_start` / `word_id` (homophones are a RANGE,
  not a single id), `is_word_end`.  Nodes are created over the SORTED SET OF PRONUNCIATIONS, not in
  word order, so the node numbering is a function of the pronunciation multiset alone (this is what
  makes the derangement leave the arrays bit-identical -- see the deviations).
* `derange_pronunciations` -- Sattolo single cycle under `default_rng(seed)`, fixed-point-free by
  construction and asserted; reports `kept_by_homophony`, the honest residual.
* `parse_arpa_word_lm` -- the KenLM ARPA as a CSR back-off automaton with integer state ids
  (state 0 = null, `1 + w` = unigram context, `1 + V + j` = the j-th 2-gram), sorted int64 key
  `state * V + word`, `searchsorted` lookup, three back-off levels.
* `LexResources` (+ `save_resources` / `load_resources`) -- the device-side resource: trie, CSR
  automaton, escape price vector (order-1 Witten-Bell phone log prob + `log 0.5`, SIL floored to
  NEG_INF), `unk_logp` / `unk_next` precomputed for every state.  One npz carries everything,
  including the null's `shuffled_word_id` column, so arm and null read the same file.
* `string_best_segmentation` -- `prior_gap.best_segmentation` through THESE resources (the
  instrument of E-1 (b) and of test (b)).
* `lexlat_log_z` (the autograd oracle), `lexlat_forward_backward` (the manual backward, checkpoint
  stride 32, D4 replay), `lexlat_viterbi` (the augmented max-plus decode, blank-free only).
* Monitors on `LexlatOutput`, beside the seven duck-compatible `LatticeOutput` fields:
  `expected_lex` (the term mean), `expected_words`, `expected_escape_words`,
  `expected_escape_phones`, `expected_word_phones`, `pruned_median/p95/max`, `max_multiplicity`,
  `contexts_per_frame`.

Numerics mirror `lattice.py` by IMPORT, not by copy: `NEG_INF`, `_logmm` (float64 accumulation
whatever the caller's dtype), `_exact_fp32_matmul`, `_band_matrix`, `_band_index`, `scaled_seg_pad`,
`_arc_weights`, `_prior_term`, `trigram_history`.

Two implementation choices worth recording, both documented in the module docstring:

1. **Arcs are materialised as SLOTS and deduped by destination key**, not written with an injective
   `index_copy_`: several arcs of one frame reach the same `(trie node, LM state, escape, phone
   history)` destination and `log P3(k | h)` needs the full history, so the destination is a packed
   int64 key, the slots are sorted by it and a segment-logsumexp closes the group.
2. **Top-C ranking is exact and PRE-BAND**: `T[o, k] = logsumexp_d band(o, d, k)` and the
   contraction over the source offset commute, so the mass a frame gives a destination is computed
   without the band reduction, ranked, and only then reduced on the C survivors.  `C_esc = 64`
   escape slots are reserved by boosting the top-`C_esc` escape candidates before the global topk.

## Tests -- `test_lexlat.py`, all run

```
31 passed, 1 skipped in 39.6 s
```
(`/e/project1/spell/wu24/env/conda/envs/speech_llm/bin/python -m pytest -q
recipe/2025-10-speech-llm/src/speech_llm/sae/emc/test_lexlat.py`, from the setup dir)

Measured residuals (re-measured after the final edit):

| check | result |
|---|---|
| (a) `lam_lex = 0`, pruning off vs `lattice.forward_log_z`, float64, s_len 6-9 x tau 1,2 | worst \|delta log Z\| = **4.44e-16** (bar 1e-9) |
| (a) the same DP on an AMBIGUOUS lexicon | log Z excess **+0.0195 .. +0.4474 nats** -- the lexical axis is a sum, not a relabelling |
| (b) `string_best_segmentation` vs `prior_gap.best_segmentation`, TINY_LEXICON, real `lmplz` trigram + real `kenlm`, 10 strings x ESCAPE/STRICT | worst **2.86e-07 nats per token** (bar 1e-4) |
| (b) `lexlat_viterbi` vs a brute-force enumeration of every path x duration x analysis, 4 configs | score exact to **< 1e-9**, phone string identical; log Z equals the logsumexp over ALL analyses to **< 1e-9** |
| (b) 50 banked Step 0 strings through the REAL trie | **SKIPPED** (see deviations) -- the check of record is E-1 |
| (c) escape identities (one `<unk>` per contiguous span, per-phone term, SIL never escaped and closing an escape, dear escape not taken, dominant escape covering the whole string) | all hold to 1e-6 |
| (d) production shape (40 phones, band 25, stride 3, d_min 2), float32 tables | rows-sum-to-1 <= **7.2e-07**, all fields finite, no z_zero; C=256: pruned max 9.96e-02, 229.5 ctx/frame; unpruned: pruned max 0, 1692.5 ctx/frame |
| (d) the same in float64 | rows-sum-to-1 <= **6.7e-16** |
| (e) derangement | 0 fixed points, `child` / `word_start` / `is_word_end` bit-identical, `word_id` moved, reproducible from the seed |
| manual backward vs autograd (float64, tau 1 and 2, checkpoint 2/3/32, ESCAPE and STRICT) | \|post_q - tau grad\| <= **3.3e-16**, \|seg_post - tau grad\| <= **1.1e-16** |
| lam_lex = 0 vs `lattice_forward_backward`'s own tables | post_q, seg_post, expected_tokens/prior/reverse all < 1e-9 |

## Local smokes (CPU, login node)

* **E0 census path, 3 utterances, end to end** (`work/.../LexlatCensusJob` run in-process, job dir
  deleted afterwards): model rebuilt from the banked `ctrl_50` `returnn.config`, ep10 checkpoint
  loaded, recognizer forward, plain `lattice_forward_backward`, 12 lexlat cells, the backward
  monitors, the Viterbi decode, PER against gold, both kill reads and the full render -- **21.8 s**.
* **E-1 path, 5 real dev-other utterances** with the tiny resource: the two scorers agreed to
  **1.31e-08 nats per token** on real gold strings; the bliss-only branch, the render and the
  strings dump all ran -- **1.5 s**.
* `LexiconTrieBuildJob._bigram_types` on a hand-checked 4-line text: the three conventions come out
  3 / 5 / 8 as computed by hand.
* **Both smokes used a TINY 5-word synthetic lexicon, not the phase's resource.  Every number they
  printed (pruned mass, log Z, PER, E[N], the monitors) is a smoke artefact and is NOT a
  measurement of anything.**  The real resource does not exist yet: the Step 0 job built its word
  LM in its own work directory and that directory has been cleaned (only `output/` survives).

## Graph load

`config_sae_4a_lexlat_probes_v1.py` loads and registers exactly three NEW jobs and eight outputs;
no banked job is reconstructed (every input is a `_pin`ned frozen path):

```
speech_llm/sae/emc/lexlat_jobs/LexiconTrieBuildJob.rlMsnTBSZXsB        cpu 4  mem 48  time 4
speech_llm/sae/emc/lexlat_jobs/LexlatEquivalenceProbeJob.lq61PSAg1DcC  cpu 4  mem 32  time 4
speech_llm/sae/emc/lexlat_jobs/LexlatCensusJob.hJKXcRjKieh3            gpu 1  gpu_mem 96  mem 64  time 6.0
```

The shim is `config/sae_4a_lexlat_probes.py` and defines `py()` (imported as `run`).

## Deviations from the spec, and what was done instead

1. **Test (a) as written is false in general.**  At `lam_lex = 0` the lexical WEIGHTS vanish but the
   lexical AXIS does not: the augmented DP still sums over the decompositions of each phone string,
   so log Z equals the bed's only when every string decomposes uniquely.  The test therefore runs
   on a one-phone-word lexicon under STRICT (exact to 4.4e-16) AND asserts the other half -- that
   an ambiguous lexicon at `lam_lex = 0` gives a strictly larger log Z (+0.02 to +0.45 nats here).
   Consequence for Design 5's pre-registered check (a) (step-1 loss of `lexlat_20` matching
   `ctrl_20`'s to 1e-4): that check is fine as written, because the arms run the BANKED `lattice.py`
   path before the on-set (amendment 2), not the augmented DP at `lam_lex = 0`.
2. **The real-resource half of test (b) cannot be a unit test.**  The 151,731-word trie and the word
   trigram are not on disk anywhere (`PriorGapAnalysisJob.2RkbKYl0v1XK/output` holds only
   `per_utt.json`, `prior_gap.json`, `prior_gap.md`; the LM lived in the cleaned work directory).
   The test is skip-guarded on `LEXLAT_TEST_RESOURCES` (a `LexiconTrieBuildJob` output directory)
   plus `LEXLAT_TEST_STRINGS`, and **the check of record is `LexlatEquivalenceProbeJob` (E-1)**,
   which builds the resource itself, runs the comparison on every gold and private string, and
   RAISES when the worst per-token disagreement exceeds 1e-4 -- a failure stops the phase, as
   Design 6 requires.
3. **"3,302,936 bigram types" is a property of the TEXT, not of the ARPA header** (survey s1
   measured it on `window.words.txt`), and KenLM's `<s>` / `</s>` contexts make the header count
   different.  The build job computes three counting conventions (in-line pairs, with `<s>`, with
   `<s>` and `</s>`) and asserts that the banked number equals EXACTLY one of them, recording which.
   `|V| = 151,731` is asserted twice: on the trie word set and on the text's word-type count.
4. **"Unpruned on the 20 shortest" is realised as a declared ceiling plus a measured exactness
   flag.**  An unbounded context axis has no bound on the real resource, so the cell runs at
   `unpruned_ceiling = 16384` (a memory bound, not an experimental constant: it enters no rule) and
   the job reports per utterance whether the measured pruned mass is exactly 0, in which case no
   context was discarded and the cell IS the exact DP.  A cell that hits the ceiling is reported as
   NOT exact.  **This value is an implementation bound I chose; if the phase wants a different one,
   it is a one-line constructor argument.**
5. **Aggregation of the funding rule's clauses was not pre-registered; it is now fixed in the job's
   docstring**: clause 1 reads the median over ALL frames of all utterances of a cell (per-utterance
   medians reported beside it); clause 2 reads the POOLED `sum |delta log Z| / sum retained`, with
   the per-utterance mean and max reported beside it; `retained` is the utterance's unit-frame
   count, the divisor `l_tau` itself uses.  Kill read (a)'s delta of record is the MEAN OVER
   UTTERANCES of the per-utterance paired PER difference (memory `paired-data-for-model-eval`), with
   the pooled delta reported beside it.  Kill read (b) reads `expected_tokens` (SIL included, the
   module's own `E[N]`), with the non-SIL count the rate term uses reported beside it.
6. **The census runs at B = 1.**  "Seconds per utterance" is one of the axes Design 6 asks the curve
   to be read on, and it is only clean at batch 1.  The batched run shape belongs to E1 (round 2).

## What round 2 needs from this round

* `lexlat_forward_backward` has the bed's exact interface: `post_q` and `seg_post` are the same
  quantities `lattice_forward_backward` returns, so the training step's surrogate
  `-(1/tau)(sum post_q log q + sum seg_post G)` is unchanged and `rate_term` works off `seg_post`.
  The call site needs `res` (from `LexiconTrieBuildJob.out_resources`, loaded with
  `lexlat.load_resources`) and a `LexlatParams`; `seg_pad` can be computed once and shared.
* The null arm is `load_resources(..., shuffled=True)`: one argument, the same file.
* The Design 4 monitor band needs the build job's token-weighted mean pronunciation length, which
  the build job prints into `output/summary.txt` and `build.json` (`phones_per_word_band`).

## Not done, and not attempted

E1 (the efficiency probe), the training-step integration, the pack config and the four arms.  No
job was launched; no manager was touched.
