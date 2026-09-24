# Design review: SAE_4A_prior.md training arm (sf_50 / soft_50, gate G4a.7) — 2026-09-20

Read-only review before the first node of the arm. Files read: `SAE_4A_prior.md` (State, Objective,
Bed, Design Step 0 / 0b / Training arm, Gate, Results incl. the new Step 0b section), `SAE.md` index
lines 79-92, `SAE_4A_budget.md` Objective/Bed/Design/G4a.4 (lines 33-130), `SAE_4A_infomax.md`
Private-code analysis and ctrl_50 band table (166-210, 251-320), `SAE_4A_objective.md` (whole),
`SAE_4A_blankfree.md` sampled-group result (337-395), `reports/survey_sampled_prior_term_2026-09-20.md`,
`reports/audit_prior_gap_2026-09-20.md`, Step 0b job output
`work/speech_llm/sae/emc/prior_gap/PriorGapAnalysisJob.Gct95xZHe0zt/output/prior_gap.md` (per-set
table lines 164-211, neural rows 31-64), code: `sae/emc/lattice.py:1268-1310`, `sae/emc/candidates.py:16-23`,
`sae/emc/prior_gap.py` (lexicon docstring 104-160, `lexicon_viterbi` 402-461), `sae/emc/neural_phone_lm.py`
(60-80, 241-292, 553-608), `sae/emc/rate_term.py:385-394`, `train_steps/sae_blankfree.py:55-115`.
Nothing on disk was changed apart from this file.

## Verdict: STOP (as written). Approvable after amendments A1-A4 and the probe in section 5.

The arm cannot launch yet in any case (Step 0b read partial_proxy; the two-instance rerun is
pre-registered before any branch). Independently of the scorer filling, the reward as written has
the wrong sign on the string set the sampler visits, so a node funded on the current design would
train a term that weakens the trigram prior — the opposite of the phase's lever.

## 1. The reward log p_strong(y) − log p_3(y) is anti-trigram on non-lexical strings (both fillings)

Design line 160-163 fixes r(y) = log p_strong(y) − log p_3(y). From the banked per-set table of
`PriorGapAnalysisJob.Gct95xZHe0zt/output/prior_gap.md` (nats per token, dev-other, same strings for
every row; the shuffled nulls keep length and unigram and destroy order):

| string set | trigram WB | neural LM | lexicon ESCAPE | unigram | r_nn = nn − tri | r_lex = lex − tri | nn − uni | lex − uni |
|---|---|---|---|---|---|---|---|---|
| gold | −3.199 | −2.555 | −2.194 | −3.504 | +0.64 | +1.01 | +0.95 | +1.31 |
| gold shuffled | −7.191 | −5.654 | −4.479 | −3.504 | **+1.54** | **+2.71** | −2.15 | −0.98 |
| private (ctrl_50 ep10) | −4.629 | −4.335 | −4.461 | −3.669 | +0.29 | +0.17 | −0.67 | −0.79 |
| private shuffled | −7.591 | −5.876 | −4.662 | −3.669 | **+1.71** | **+2.93** | −2.21 | −0.99 |

Under the reward as designed, a SHUFFLED gold string earns 0.9 (neural) / 1.7 (lexicon) nats per
token MORE than gold, and a shuffled private string 1.4 / 2.8 more than the private string. The
cause is structural, not a smoothing accident: the Witten-Bell trigram penalises order destruction
by 3-4 nats/token (backed-off unseen trigrams compound), while the escape lexicon's floor is the
order-insensitive unigram (audit §3: the decode's best path is essentially the all-escape path,
margin −0.17/token) and the neural LM is smoother on nonsense (−5.65 vs −7.19). So on every string
that does not route through real words — three quarters of the private strings under STRICT, and
essentially all local edits of them — r ≈ const − log p_3(y) + (order-insensitive terms): the
score-function term pushes q toward strings the trigram dislikes. With the lattice target
∝ (A·p_3)^{1/tau} and the term adding lam_sf·r, the effective prior off the lexical set is
p_3^{1/tau − lam_sf}·p_1^{lam_sf} — strictly weaker than the live prior. A PASS would then require
the sampler to stumble on lexical routings against a gradient that points away from trigram-typical
text; a FAIL would be read as "the prior lever fails" (Gate, line 209-210) when the lever was never
applied. That is the confound that makes G4a.7 pass or fail for the wrong reason.

The "double counting" worry that motivated subtracting p_3 is not real: the lattice's p_3 (exponent
1/tau) and the reward's p_strong (exponent lam_sf) are two priors at two weights; overlapping content
only reweights. What must NOT be the reward's baseline is anything order-sensitive.

**A1 (required).** Reward baseline = an order-insensitive scorer sharing p_strong's smoothing
family: for the lexicon filling the all-escape floor (r = lexical margin: gold +1.99, decode −0.17,
nulls ≈ 0 per token; audit §3), for the neural filling the unigram (r = nn − uni; column above:
gold +0.95 > private −0.67 > nulls −2.2, the right order). Or reward = log p_strong alone. Re-derive
the monitor targets accordingly: Design line 189 "toward gold's value, +2.3 nats/token" is wrong
for every filling — 2.3 is the gold-minus-private GAP under the lexicon, not gold's reward; gold's
per-token reward is +1.01 (lex − tri), +1.31 (lex − uni), +0.95 (nn − uni), +0.64 (nn − tri). As
written, an arm that reached gold's level would read "monitor not moved".

## 2. Does the term test the mechanism or the sampler? (Q1)

Mostly the sampler, at this bed, even after A1. Score-function gradient = Cov_q(r, grad log A) over
the posterior's support. Two facts about that support:
- At the confident code (ep10: eval-mode entropy 0.38 nats/output frame, `SAE_4A_infomax.md` 251-259)
  the tau = 2 posterior puts its mass on local edits of the private string. The lexicon reward is
  flat there (all-escape path; a single-token edit creates a vocabulary word rarely and a
  word-trigram-plausible one almost never), so Cov is ≈ 0 for the lexical component; the within-group
  variance that remains is the baseline's local structure (with A1: ≈ 0 for the margin, unigram
  noise for nn − uni). This is the "dead band" regime, and it is where the private code lives.
- At the soft code (ep1: entropy 3.04; tau 8, target post^(1/7)) samples are near-random strings;
  strict-segmentable fraction of order-destroyed strings is 1-2 % (nulls 0.024 / 0.011), so the
  lexical bonus is a rare event among G = 8; lam_sf calibrated to 0.1-0.3 of the l_tau gradient at
  step 1 (line 166-167) is calibrated at the point where reward variance is largest and LEAST
  lexical, and then held fixed while the variance collapses.
So the arm measures "can 8 posterior draws per utterance find lexical routings before the code
commits", which is the user's question verbatim; it does not measure "does a lexicalised prior
identify the code" — that question is answered by construction only by a lexicon INSIDE the lattice
(the fallback the Gate names), or by the soft arm if its gradient is trustworthy (section 3).

Instance details: G = 8 centre-only is fine (std-division pitfall correctly avoided); per-string
sum is right; "active from sub-epoch 1" is right (only regime with a diffuse posterior). Cheaper
baseline with lower variance: r(greedy decode) as the per-utterance control variate (valid, any
function of x; the decode is free from argmax log_q) instead of, or added to, the group mean.
Higher-temperature proposal T' > tau: legitimate (the centred estimator is exactly grad
E_{q_T'}[r] up to 1/T'; the −grad log Z term cancels under sum_g A_g = 0), costs one extra forward
table (tempering sits inside the per-frame logsumexp; a table cannot be re-tempered post hoc), so
~+300 s/sub-epoch. Only useful after the anneal (tau is already 8 early). Mixed proposal with
text-side samples: invalid without importance weights (the weight-sd problem Step 0 closed) and,
unweighted, is a BT-shaped pressure paid regardless of x — reject. Trigram-DP Viterbi as a group
member: not a sample of q; use it only as the baseline.

## 3. Soft-input arm (Q2)

Not fatal, but as written it is not like-for-like and can be paid by mixing. (i) A frozen
transformer's value at an interpolated embedding is unconstrained: a mixture can score above every
one-hot string, so the dense gradient rewards KEEPING positions soft; at ep1 (entropy 3.04) the
inputs are far off-manifold and the gradient is not a lexical signal in any guaranteed sense
(wav2vec-U avoids this by training the discriminator on the soft inputs jointly). (ii) "log p_3(soft
y)" is undefined in the Design: E_post[log p_3] (multilinear in posteriors) and log E_post[p_3]
(a trigram DP over soft emissions) differ by a Jensen gap of the opposite sign to the transformer's
mixing artefact, so the difference term inherits an uncontrolled bias. (iii) Argmax segmentation
fixes the length; the arm can relabel, never merge/split (disclosed limitation, fine).
Trie DP with posteriors as arc weights computes log E_post[p_lex(y)] exactly (an honest marginal
likelihood under the factorised posterior) — the right form for the soft arm IF the trie filling is
built; note it has no mixing exploit because the value is a mixture of one-hot scores, not a score
of a mixture.
**A2.** Neural filling: straight-through (forward on the one-hot argmax string, gradient through
the soft vector); the forward value is then always on-manifold and equals the decode reward. Both
fillings: log both the soft-input value and the hard-decode value per sub-epoch; their difference
is the artefact monitor, pre-register a ceiling (e.g. 0.2 nats/token) above which the arm's monitor
gain is not read. Apply A1's baseline here too. The read rule "only with sf_50 and the derangement
gap" adds nothing: the gap is positive in every content-free arm (`SAE_4A_budget.md` G4a.4 health
clause) and PER < 0.50 already protects a PASS; what it does not protect is the FAIL reading.

## 4. Gate G4a.7 and the UNINFORMATIVE clause (Q3)

PER < 0.50, rate band, matched-completion paired reads: sound (copied from G4a.4, effect size
0.85 -> < 0.50 needs no seed band to detect; n = 1 negatives read "not funded", standing rule).
The dead-band clause cannot fire as written: `sf_reward_std_within < 1.0 nats/utt` for 5 sub-epochs
requires near-identical samples, and the only sampled-group measurement on this project's lattices
(`SAE_4A_blankfree.md` 373-378: 494-510 unique strings of 512 at tau 2 on the final checkpoints)
says posterior samples of 60-token strings are almost all distinct; 8 distinct strings differ by
several nats under any scorer. So the sampler failure that actually threatens the arm (section 2:
diverse samples whose reward differences are non-lexical) reads FAIL, not UNINFORMATIVE, and the
FAIL clause licenses the wrong conclusion. Thresholds were not calibrated from any measurement
(1.0 = "half a phone's lexicon price" is a per-token figure applied to a per-utterance std).
**A3.** Replace the dead band by two monitors read together, thresholds set from the probe below:
(a) within-group std of the A1 reward (label-free), and (b) disclosed label-using per kept
checkpoint: fraction of utterances where r(gold) > max_g r(y_g). If (b) stays ≈ 1 while (a) is
above its floor, the sampler explores but never reaches strings as lexical as gold — the exploration
statement, now falsifiable. Add (c) label-free lexical-content monitor: fraction of sampled tokens
covered by strict-lexicon words of length >= 2 (trie filling) or the A1 margin itself.

## 5. Cheapest pre-launch probe (falsifier)

Zero cost, already done above: the null-vs-original contrast falsifies the reward as written.
Next, before the FFBS exists (CPU, one PriorGapAnalysisJob-class job on the banked decodes, ctrl_50
ep4 and ep10): for each private string, K = 8 single-token substitutions / deletions drawn from the
frame posterior's second choice (the neighbourhood a confident posterior samples), score under
the A1 rewards and under the trigram; report within-neighbourhood std, correlation of the reward
delta with the trigram delta (anti-trigram check at the local scale), and fraction of edits that
gain a strict-lexicon word. Then, once the blank-free FFBS is written (needed for the arm anyway):
G = 8 draws on 300 dev-other utterances at ctrl_50 ep1 / ep4 / ep10 at the schedule's tau, report
unique strings, within-group std, r(gold) > max_g, and the gradient-norm ratio that fixes lam_sf at
ep4 and ep10 (record both; prefer ep4 over step 1). If r(gold) > max_g in > 95 % of utterances at
every checkpoint, do not fund sf_50; fund the lattice-internal lexicon design instead.

## 6. Implementation route and cost (Q5)

- Fixed-string blank-free marginal is unnecessary on the gradient path. Fisher identity:
  grad log q(y|x) = E_{path ~ q(path | y, x)}[grad log w(path)], and the FFBS returns exactly one
  such path per sample (`candidates.py:16-23` fields; the blank-free version will too). So
  A_g · [sum_t log_q[t, a_t] + sum_seg seg_table[k, s, d]] (prior term constant in theta/phi; the
  −log Z term cancels under centred advantages) is an unbiased estimator of the designed term at
  the cost of a gather; no per-string DP, no autograd through a DP (memory trap
  `lattice-dp-autograd-memory`). Keep `fixed_phone_log_marginals` as a diagnostic only.
  **A4.** Use the path-level score; drop the differentiable fixed-string marginal from the plan.
- Neural scorer length cap: `neural_phone_lm.py:60-80, 575-580` ASSERTS len <= 512 and the position
  table ends at 512. A 35 s utterance has ~583 output frames at stride 3; at tau 8 in sub-epoch 1 a
  sampled blank-free string is ~0.97 T tokens > 512. Predicts an assertion crash at the first long
  utterance under the neural filling. Pre-register the policy (mask the utterance from the sf term
  with a counter, or window the scorer); never silent truncation.
- SIL: rate term counts non-SIL tokens (`rate_term.py:385-394`); with the trigram baseline SIL
  removal raised r by 0.4 nats/token under both fillings (private_sil vs private rows), i.e. the term
  would push SIL out unopposed. Under the A1 unigram baseline the SIL effect is small (−0.5 vs −0.67).
  Fix the convention explicitly: score the SIL-dropped string under both scorers (Step 0's primary
  pairing), or keep SIL and disclose the direction. BOS / no-EOS: consistent across `prior.py`,
  `neural_phone_lm.py`, `prior_gap.py`; the lattice's BOS-BOS history matches.
- FFBS inside the checkpointed backward: consistent with what the backward has (survey s1); sampler
  state per draw is index-sized, so G = 8 adds little to the recomputed 32-frame chunk. Measure, as
  the Design says.
- LM loading via hashed model_args with a sha256 pin: fine (precedent survey s6).
- Cost: the 12.5 h > 11.5 h line depends on the pack resume path, verified by reading only; its
  first real kill is tonight (~01:14, `SAE_4A_budget.md` State). Do not fund before that resume is
  seen to work, or pre-register the ep25 read (ctrl_50 keeps 25) as the fallback now.
- Trie filling: Step 0's lexicon row uses a WORD TRIGRAM state (`prior_gap.py:154`, KenLM back-off
  state); a batchable GPU trie DP will realistically carry a word unigram or bigram. Step 0 has no
  word-unigram lexicon row, so the trie filling's reward would not be the quantity Step 0 measured.
  Bank a word-unigram (and bigram) ESCAPE row in the same job before building the DP; it is a CPU
  rerun with one changed input.

## 7. Scorer slot (Step 0b, read against the new Results section)

Partial proxy, 1.71 vs bar 2.01, strict subset 1.52 vs 2.51; the schedule-bound reading and the
two-instance rerun are appropriate. Note for the read: perplexity does not predict the gap (6-gram
5.42 < neural 5.77 in perplexity, yet 1.62 < 1.71 in gap; the audit's 6-gram reading is that extra
context buys the decode as much as gold), so a 10 M model reaching the 6-gram's perplexity is not
expected to clear 2.01 on that account. Either way A1-A4 apply to both fillings.

## 8. What cannot be falsified as written

- "sampler does not explore" via the dead band (section 4): the threshold never triggers.
- "the monitor should rise toward +2.3 nats/token" (line 189): unreachable target.
- soft_50's "embedding artefact" caveat (line 172-173): no monitor separates artefact from gain.

## Amendments in priority order

A1 reward baseline order-insensitive (unigram / escape floor) and corrected monitor targets;
A3 replace the dead band by the gold-above-max fraction plus calibrated within-group std;
A2 straight-through for the neural soft arm, artefact monitor with a ceiling, pin log p_3(soft);
A4 path-level score (Fisher identity), no differentiable fixed-string DP; length-cap policy;
SIL convention; resume path verified or ep25 fallback pre-registered; word-unigram lexicon row
banked before any trie DP.
