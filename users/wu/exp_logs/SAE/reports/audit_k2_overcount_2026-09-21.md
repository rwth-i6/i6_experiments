# Audit: what `LexlatK2OvercountJob` actually measured (amendment 9.5 read), 2026-09-21

Fresh-context audit of one reading. Nothing was edited, re-run or repaired; the only compute is a
read-only CPU probe under `/e/scratch/spell/wu24/envs/sae_k2/bin/python` (scripts left in the
session scratchpad, not in the project).

Artifact under test:
`/e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/lexlat_k2_arm_jobs/LexlatK2OvercountJob.f5Ljn6twbc4b/output/{summary.txt,overcount.json}`
(job dir resolves to `/e/scratch/spell/wu24/2026-07-13_unsupervised/work/...`, `finished` present,
`partial: false`, `pass: false`).

## 1. The numbers re-derived

From `overcount.json` (`rows` arrays, medians recomputed independently of the summary):

| set | strings scored | tokens | median `over_count`/token | median `tropical_residual` | median `sum_minus_max` |
|---|---|---|---|---|---|
| gold | 2091 | 110919 | **+0.2871** | +0.000000 (max +0.0576) | +0.2863 |
| private | 2842 | 165532 | **+0.1621** | +0.000000 (max +0.2600) | +0.1430 |

These reproduce `summary.txt` exactly. The identity `over_count = tropical_residual +
sum_minus_max` holds to 1e-16. The aggregation choice does not matter: token-weighted means are
+0.2833 / +0.1611, and 100 % of gold and 91.4 % of private strings individually exceed 0.05.
The pre-registered criterion (`SAE_4A_lexlat.md` amendment 9, item 5: median over-count <= 0.05
nats per token, both string sets) is missed by 5.7x (gold) and 3.2x (private): **the FAIL read is
arithmetically correct as the gate is worded.**

Frame checks that hold: the graph priced is the escape, undeterminized, unshuffled
`LexlatHLGBuildJob.rtX44PBJFNy1/output/HLG.pt` (`build.json`: `escape true`, `determinized false`,
`shuffled false`, `sil_prob 0.5`, theta = 0, i.e. the unpruned 1M-line trigram); the CSR side is the
same `LexiconTrieBuildJob.rlMsnTBSZXsB` npz the graph was compiled from; the strings are E-1's own
`LexlatEquivalenceProbeJob.lq61PSAg1DcC/output/step0_strings.json`, both sets, no cap
(`max_utterances: null`), intersection exact (`k2.intersect`, no beam).

## 2. What the registered column is

`lexlat.string_best_segmentation` (lexlat.py:515) is a **max-plus** Viterbi (`if total > old[0]`)
over (position, word-LM state, escape-open). The graph side is `get_tot_scores(log_semiring=True)`.
So `over_count = log-sum over every reading of the graph - the single best reading of the CSR`, and
`sum_minus_max` is the graph's whole multi-reading mass, not the back-off double count.

Which readings a linear phone FSA intersected with this HLG admits (from the construction in
`lexlat_k2.py`):

* **alternative word segmentations** - yes. `lexicon_to_fst` branches at `loop_state` for every
  pronunciation; all cuts of the string are separate paths and are summed.
* **homophones / several words per pronunciation** - yes. 14427 trie nodes carry more than one word
  id (up to 13); each is a separate L branch.
* **escape readings of spans** - yes. The escape phone loop reads any non-SIL phone in parallel with
  the lexical branch, and a span may also be split into consecutive escape spans (one `<unk>` each).
* **optional-SIL readings - no, here they contribute nothing.** No string in either set contains
  SIL (0 of 2864 in each) and no pronunciation contains SIL (`child[:, sil_id] >= 0` nowhere), so
  L's silence arcs can never fire; with `sil_prob = 0.5` the sil and no-sil boundary weights are
  equal anyway, so the silence model is a deterministic `(n_words + 1) * log 0.5` on every path.
* **back-off routes of G and their insertion positions** - yes, and they dominate. `g_fsa` gives
  every state a `#0` back-off arc; `_add_self_loops` puts a `#0` self-loop on every L state with a
  non-epsilon outgoing token, so one G route can be consumed at several positions inside a word.
  `compile_hlg` deliberately skips determinization and `k2.remove_epsilon` is tropical-only
  (k2 fsa_algo.py:718: "equivalent to the input `fsa` under the tropical semiring"), so the copies
  survive and are summed. This is exactly what the module's own xfail reason states
  (`test_lexlat_k2.py:627`: "over-counts the back-off routes of G **and their insertion
  positions**").

## 3. Decomposition measured (read-only CPU probe, 60 strings per set)

I transcribed `string_best_segmentation` arc for arc with the semiring switchable and the icefall
silence constant paid per parse (no_sil at the start and at each word / escape close). Validation:
the max-plus version reproduces the graph's own `k2_max_plus` to a median 9e-9 nats per token
(max 4.2e-3, the float32 arc scores of the graph) on all 120 strings - so this parse set IS the
graph's parse set on the max side, and the E-1 equivalence holds at full scale.

Three-way split of the per-string, per-token over-count (medians, n = 30 per set for the third
column, n = 60 for the first two):

| set | `over_count` | legitimate multi-parse (exact CSR log-sum - CSR max) | plain back-off route sum | remainder (insertion positions) |
|---|---|---|---|---|
| gold | +0.2894 | **+0.0125 (4.3 %)** | +0.0325 | +0.2404 |
| private | +0.1386 | **+0.0194 (14 %)** | +0.0173 | +0.1098 |

"Plain back-off route sum" replaces each word transition by the log-sum of its up to three routes
(explicit arc; back off then the lower-order arc) with their own successor states, plus G's
end-of-string back-off closure; it is the double count as usually understood. The remainder is the
positional multiplicity of the `#0` loops.

So: the legitimate mass the design's objective should sum over (segmentations, homophones, escape
spans; SIL contributes nothing) is a **minor** part of the measured column - about 4-5 % (gold) and
13-15 % (private) - and the rest is graph-construction multiplicity, of which the plain back-off
double count is itself only ~10 %.

## 4. Verdict on the three questions asked

* **(a) / (b) / (c)?** The column is **(c) something else, which happens to be an upper bound**: it
  is `graph log-sum - best single parse` = back-off double count + `#0` insertion-position
  multiplicity + legitimate multi-parse mass + (occasionally) slack in the reference itself. Every
  added term is non-negative, so it does bound the double count from above, but it is not tight and
  it is not the quantity the amendment's cited fixture number measured.
* **Does `sum_minus_max` isolate the double count?** No. It is the whole multi-reading mass. The
  summary says as much in its own words ("the multi-reading mass ... where the epsilon back-off
  double count lives"); the number is nonetheless read against a tolerance that was anchored to a
  differently-defined fixture measurement (below).
* **Did the amendment fix the reference as max-plus or as the exact log-semiring CSR sum?** As
  written, **max-plus**: "the exact CSR score of `lexlat.py` (E-1's path)" is
  `string_best_segmentation`, a Viterbi; "exact" qualifies the Viterbi, not a semiring. The job
  implemented the sentence as written. But the sentence's own justification cites the k2 fixture's
  "+0.064 nats", and that fixture number
  (`test_hlg_log_semiring_exceeds_the_sum_over_parses`, lexlat_k2 test docstring) is
  `graph log-sum - log-SUM over parses`, a sum-vs-sum quantity. The amendment therefore names one
  quantity in its motivation and defines a different one in its comparison, and the pre-registered
  0.05 was never tied to either scale (the fixture's 0.064 is per string, ~0.011 per token, on a
  toy LM).

## 5. Does the conclusion survive the correction?

Yes. Under the fixture-comparable reference (exact log-semiring CSR sum over the same parse set)
the measured over-count is a median **+0.274 (gold) / +0.117 (private) nats per token**, still
5.5x / 2.3x the 0.05 tolerance. The FAIL and what it licenses ("not funding the k2 arm without a
further amendment; the term would be an approximate word constraint") stand under either reading of
the reference. The mis-specification changes the attribution, not the decision.

## 6. Concerns to record

1. **The summary's reading of `tropical_residual` is wrong in its explanation** (not in its number).
   It says a non-zero value "is a defect of the graph, not a double count". It is neither: the
   banked `reference_total` is `max over parses of (LM + escape score)` plus
   `sil_model_log_prob(n_words)` of *that* argmax, while the graph maximises the sum including the
   silence constant. When the two argmaxes differ the graph's max-plus is legitimately larger. With
   the constant inside the DP my max reproduces `k2_max_plus` to 1e-8 per token, while it exceeds
   the banked reference by up to +0.081 nats per token (22 of 60 private strings, 5 of 60 gold).
   `tropical_residual` is reference slack, and it is one-sided (positive up to float noise: 2719 of
   2842 private, 1686 of 2091 gold strictly positive). Its median is ~0, so it moves nothing here.
2. **Gold coverage.** 773 of 2864 gold strings (27.0 %) are excluded, 772 of them for adjacent equal
   phones; private loses 22 (0.8 %). Disclosed in the summary, but the gold median is over a
   non-random 73 % of that set, and the two sets are therefore not equally covered.
3. **Frame item not checkable from these files.** Amendment 9 item 4 fixes the arm's graph as "the
   graph the probe passed on". The HLG here is named `lexlat_20/hlg` and is the theta = 0 escape
   build; whether that is the rung the E0/E1 probe passed is not determinable from this job's
   artifacts. It would need the probe job's own summary. If the arm would run a pruned LM, this
   over-count is measured on a different graph from the one funded.
4. The k2 log-sum is measured on a graph whose epsilons were removed by a tropical-only algorithm,
   so it is not a marginal in any exact sense even before the double count - it is what the training
   term would actually compute, which is the right thing to price, but it should not be described as
   "the marginal over word-decomposable strings".

## 7. What would be needed to close what I could not

* The exact split between "back-off double count" and "insertion-position multiplicity" at full
  scale would need a registered reader (the two probe scripts here are a scratch check on 60 + 30
  strings per set, not a banked job). The direction and order of magnitude are robust: on every one
  of the 120 strings the graph's log-sum exceeds the exact CSR log-sum, and the legitimate
  multi-parse mass never reaches 0.05 nats per token (max observed +0.0486 gold, +0.1215 private).
