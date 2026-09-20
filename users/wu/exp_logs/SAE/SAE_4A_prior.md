# SAE §4a — Prior strength: shrinking the family of codes the prior accepts

## State

Phase opened 2026-09-20 on the user's approval, replacing the context-dependent reverse model
(`SAE_4A_cdrev.md`, deferred without limit by the user the same day, nothing built).
Step 0 (CPU prior-gap diagnostic, `PriorGapAnalysisJob.2RkbKYl0v1XK`, speech-llm 291dab1,
`reports/impl_prior_gap_2026-09-20.md`, `reports/review_prior_gap_2026-09-20.md`) is finished,
read in Results and audited (CONFIRMED). Its v1 (`.l0p0srBryKrs`, 1431dbf) failed on the KenLM
order cap and is superseded. Conventions fixed before the numbers stand: like-for-like pairing
(SIL dropped from the decode) primary, SIL-kept disclosed; BOS, no end-of-sentence term, identical
denominators, every prior from the priorshuf uniform window (`SampleLinesJob.orN768ARKwlt`).
Step 0b is running: `NeuralPhoneLmTrainJob.Iv6P6YVPNWmB` (GPU, 4 h cap, Slurm 1918176) then
`PriorGapAnalysisJob.Gct95xZHe0zt` (neural row + held-line perplexity benchmark), speech-llm
739d9ed, e971603, `reports/impl_neural_phone_lm_2026-09-20.md`; code review
`reports/review_neural_phone_lm_2026-09-20.md`. Manager pid 3231512,
`log/sae_4a_prior_gap.manager.20260920T202826Z.log`; watcher
`bash ~/.claude/skills/sis/sis_watch.sh 3231512 config/sae_4a_prior_gap.py 600` (re-arm first
after any resume, with the budget and infomax watchers).
Training-arm design and G4a.7 are written (Design "Training arm", Gate); the scorer slot is
filled by Step 0b's rule. Code survey banked (`reports/survey_sampled_prior_term_2026-09-20.md`).
Open user question (SIL vs rVAD): prior text has 13.8 % SIL tokens (sil_prob 0.5, surround;
the local wav2vec-U pipeline uses 0.25 with rVAD); the gold SIL share on retained frames is being
computed (`reports/extract_sil_rate_2026-09-20.md`); a sil_prob arm is a bed change, own arm.
NEXT: watcher verdict on Step 0b; read the neural row against the Step 0b rule and the perplexity
benchmark (trigram 9.561 expected); fill the scorer slot; design review of the training arm; then
implementer (blank-free FFBS, fixed-string marginal, LM loading via hashed model_args, soft-input
term) before any node is funded.

## Objective

The budget control settles into a confident frame-level acoustic code that satisfies the live
trigram prior nearly as well as phones do (`SAE_4A_infomax.md` Results: prior per token −3.98 on
the dev-other decode against −3.20 for the gold phones; training monitor −3.44 flat from
sub-epoch 12). A trigram over 40 symbols admits a large family of codes with English-like trigram
statistics, and the joint optimum picked one of them. Raising the prior's weight beta would make
that code more trigram-typical, not more phone-like, so weight is not the lever (user, 2026-09-20).
The lever is a prior whose accepted family is smaller: higher n-gram order, or the lexicon (phone
strings must decompose into dictionary words, which an acoustic clustering code does not). The
phase asks, first, which of these enlarges the gold-minus-private gap, and second, whether that
prior can be brought into training so that a cold blank-free arm leaves the content-free band.

Literature anchor (`reports/lit_cdrev_2026-09-20.md`): the reproduced gains in unsupervised phone
recognition come from prior strength and lexicalisation (Klejch et al. 2022: bigram -> 5-gram ->
word trigram with a lexicon, the prior the most important factor), with a context-free emission.

## Bed and constants

Inherited from `SAE_4A_budget.md` ctrl_50 for any training arm (priorshuf bed, N = 50, the budget
round's schedule and gate thresholds). The live prior: interpolated Witten-Bell trigram over the
39 ARPAbet phones + SIL, held-out perplexity 9.561 on the live fit's own 10,000 held lines
(`PhoneNgramPriorJob.RtzbESkOedsT` prior.stats.txt; the September 15 estimate's 9.469 was a
different held set, `reports/estimate_prior_order_2026-09-15.md`); the order-4 KenLM
(`output/sae/1a/phoneme_lm_o4.{arpa.gz,bin}`) was bracketed at about 8.5 (0.15 bits/phone more)
on a different held set. The September 15 estimate rejected 4-gram rescoring of a BIGRAM DP as an
importance-sampling estimator (per-phone gap 0.41-0.51 nats, per-utterance log-weight spread
25-30 nats, effective sample size near zero); with the trigram now in the DP the residual to the
4-gram is about 0.1 nats per phone, which reopens that estimator for the 4-gram and closes it for
any strongly discriminating prior (the lexicon), where a score-function term with a per-utterance
baseline is the estimator instead.

## Design

### Step 0: prior-gap diagnostic (pre-registered 2026-09-20, before the job was written)

A disclosed label-using analysis, in the class of the Hungarian oracle and the private-code
table: nothing from it enters any training, checkpoint choice or selection. Strings scored, on
dev-other, per utterance: (a) the MFA gold phone strings (`s0b.GOLD_PHONES`, the PER reference);
(b) the ctrl_50 ep10 collapsed greedy decode (the private code, the same banked decode the
private-code analysis read); (c) a null per string: the same tokens shuffled within the
utterance (seed fixed), which keeps length and unigram and destroys order. Priors, all estimated
from the same uniform-sample text window the live prior uses (never the alphabetical head, the
n-gram rule), with the same phonemisation and SIL convention: phone n-grams of order 1, 2, 3 (the
live estimator, `prior.py` Witten-Bell), 4, 6 and 8 (KenLM, modified Kneser-Ney, the 6- and
8-gram as the table-lookup proxy for a lexicalised prior), and the exact lexicalised prior: a word
trigram over the lexicon, scoring a phone string by its best segmentation into lexicon words
(Viterbi over the lexicon trie with the word-LM state, SIL treated as an optional word boundary,
an out-of-lexicon segment impossible), reported with the fraction of strings that have any
segmentation at all. Reported per prior and per string set: mean log-probability per token and
per utterance, and the paired gap gold minus private per token with its even/odd-utterance halves
(spread = |gap_even − gap_odd|). Also reported, as the importance-sampling bracket: the
per-utterance standard deviation over utterances of log p_order − log p_3 for gold and for the
private code, at orders 4, 6, 8 and the lexicon.

Read rule, in the job's docstring: a prior "discriminates more than the trigram" if its
gold-minus-private gap exceeds the trigram's by more than the larger of the two priors' spreads.
Decision table, fixed now: 4-gram discriminates more and its per-utterance log-weight sd is below
3 nats -> the next arm is the 4-gram importance-sampled correction inside the trigram DP; the
lexicon (or its 6/8-gram proxy) discriminates more but the 4-gram does not -> the next arm is the
lexicon score-function term (sampled strings from the lattice posterior, reward log p_lex −
log p_3, per-utterance mean baseline, from sub-epoch 1, within-group reward variance logged as
the engagement monitor); neither discriminates more than the trigram -> the prior family is not
where the private code is identified, recorded as a negative and the phase closes without a node.
The prediction, written before the numbers: the 4-gram gap grows little (the private code is
locally English-like), the lexicon gap is large and the null strings have no segmentation.

Amendments after the code review (2026-09-20, `reports/review_prior_gap_2026-09-20.md`, before
any number was produced; the v1 job had failed on the KenLM order cap): (i) phone n-gram orders
are 1-4 and 6; the 8-gram is dropped (the installed KenLM supports order 6 at most). (ii) The
lexicalised prior had two defects: its paired gap was taken over the segmentable subset (about 70
% of the private strings) while the trigram gap it was compared with covered every utterance, and
the dropped utterances are exactly where a lexicon separates hardest; and a lexicon word unseen by
the word trigram cost one `<unk>` rather than being excluded, which lets a non-English string buy
a cheap score from long rare words. v2 reports two conventions on the full paired set: STRICT
(lexicon restricted to the word LM's vocabulary, no `<unk>` path; gap on the subset where both
strings segment, with the trigram gap recomputed on that same subset, and the segmentable
fraction per string set) and ESCAPE (the same trie plus an escape word for any phone span, costing
the word LM's `<unk>` transition plus the order-1 phone score and log 0.5 per phone, one fixed
convention). ESCAPE is the row that enters the decision table's lexicon clause, because a
training reward must be finite on every string; STRICT is disclosed beside it. (iii) The nulls do
segment (34 of the 39 phones are single-phone words), so the prediction's last clause is replaced
by: the nulls' segmentable fraction and lexicon score fall well below gold's. (iv) Per-utterance
scores are dumped so any subset read is recoverable; every banked aggregate is rendered.

### Step 0b: does a neural phone LM learn the lexicon? (pre-registered 2026-09-20, after Step 0's numbers, before the job was written)

Question: can a scorer that runs on a GPU batch of sampled strings inside a training step carry
the lexicon-level discrimination of Step 0? Candidate: a small causal transformer phone LM (about
4 layers, width 256, 5 to 8 M parameters, seed 0) trained on the same priorshuf uniform window as
every other prior, with the window's own SIL convention, held-out perplexity reported on the
window's held lines; scored in the same PriorGapAnalysisJob (a new job instance with the neural
LM as an added input; every existing row recomputed unchanged) with the same conventions
(sentence-start context, no end-of-sentence term, same denominators, both pairings). Two rows:
the neural LM alone, and the neural LM's gap read against the lexicon ESCAPE row. Read rule,
fixed now: the neural LM is an adequate lexical scorer if its like-for-like gap exceeds the
trigram's by at least two thirds of the lexicon ESCAPE row's excess (i.e. gap >= 1.39 + 0.62 =
2.01) and its strict-subset behaviour tracks the lexicon (gap on the strict subset within 0.3 of
the lexicon STRICT row); if it lands between the 4-gram and that bar it is a partial proxy and
the arm's design must say what it loses; if it does not beat the 4-gram, a neural phone LM is not
the scorer and the exact lexicon must be made batchable (a GPU trie DP) before an arm exists.
Prediction: the neural LM lands near the lexicon (a phone LM with a receptive field of tens of
tokens learns word forms; the null strings will score as far below as under the lexicon).

Code review (2026-09-20, `reports/review_neural_phone_lm_2026-09-20.md`, before any number):
leakage clean (counted and held lines from the one window split, asserted disjoint, the live
prior's own 1,000,000 / 10,000 split), causal mask and no-EOS convention correct, denominators and
truncation handling sound. Three points, resolved before the numbers: (i) the built model has
3.31 M parameters, below the pre-registered 5 to 8 M. Rule fixed now: a PASS of the read rule by
the 3.3 M model stands (a smaller model clearing the bar is the stronger result); a "partial
proxy" or "not the scorer" outcome from it does not discharge Step 0b and is re-read after one
rerun at 6 layers / width 384 (about 10 M parameters), same data, same conventions, before any
branch is taken. (ii) The benchmark's expected live-trigram perplexity on these exact held lines
is 9.561 (`PhoneNgramPriorJob.RtzbESkOedsT` prior.stats.txt), not the 9.469 of the September 15
estimate, which was a different held set; the Bed section is corrected. (iii) Best-epoch
selection reads the same held lines the benchmark reports, so the neural held-out perplexity is a
best-of-at-most-3-epochs number on its own report set; the bias is below 1 % in perplexity and
the number is labelled so, no separate split.

### Training arm (pre-registered 2026-09-20, before Step 0b's read; the scorer slot is filled by
Step 0b's rule, nothing else changes with it)

Mechanism under test: Step 0 showed the trigram inside the lattice prices the private code only
1.39 nats/token below gold while a lexicalised prior prices it 2.32 below (Results). A strong,
non-decomposable scorer p_strong (the Step 0b neural LM if it PASSES, else the lexicon trie DP on
GPU) cannot sit in the DP, so it enters as a correction term outside it:
- **Score-function arm `sf_50`.** Per utterance, draw G = 8 strings y_1..y_G from the lattice
  posterior q_theta(y | x) by forward-filtering backward-sampling in the blank-free trigram DP,
  vectorised over G inside the existing checkpointed backward recomputation (survey s1: same
  table, `cx`, `seg_pad`; the CTC/stride-1 `sample_joint_paths` is NOT reused). Reward
  r(y) = log p_strong(y) − log p_3(y), summed over the string (never per-token: the per-token mean
  pays for length, `reward.py:115-146`; both LMs score the same y so the length cost cancels in the
  difference). Advantage A_g = r(y_g) − mean_G r (centre only, no std division). Term
  lam_sf · mean_G[ A_g · log q_theta(y_g | x) ], log q_theta(y | x) = log A_tau(y) − log Z from the
  fixed-string blank-free marginal (to be written; the CTC `fixed_phone_log_marginals` is the
  template), differentiable through theta and phi. Active from sub-epoch 1 with the budget
  schedule (`blankfree_budget_jobs`), lam_sf chosen so the term's gradient norm at step 1 is
  0.1–0.3 of the l_tau gradient norm (measured in the 100-step probe, recorded as a choice).
- **Soft-input arm `soft_50`** (the user's exploration concern, 2026-09-20: sampling may never
  leave the mode). The wav2vec-U route: argmax segmentation, soft symbol vectors (the per-segment
  posterior over 40 symbols under the lattice), fed to the same frozen p_strong through its
  embedding matrix; term lam_soft · [log p_strong(soft y) − log p_3(soft y)], dense gradient, no
  sampling. Caveat pre-registered: p_strong was trained on one-hot strings; a gain here may be an
  embedding artefact, so the arm is read only together with sf_50 and the derangement gap.
- Control: ctrl_50 (budget round, banked; same bed, schedule and step count). Seeds n = 1; a PASS
  needs a second seed before it is claimed (standing rule).
- Cost: one extra DP pass per step for sf_50 (~3 passes vs ~2 today, survey s7), so ~900 s per
  sub-epoch and 50 sub-epochs = 12.5 h > the 11.5 h clamp: the arm runs on the resume path the
  budget round is establishing, or it is read at the sub-epoch the clamp reaches, recorded before
  launch. Memory: G = 8 strings x per-frame recomputation stays inside the 96 GiB node if the
  sampled paths are drawn chunk-wise; measured at a long-S batch in the probe, never assumed.

Monitors (per sub-epoch, label-free, in the training log):
- `sf_reward_std_within`: std over G of r(y_g), median over utterances. The dead band: if it is
  below 1.0 nats/utterance (about half a phone's lexicon price) for 5 consecutive sub-epochs after
  the anneal, the term received no signal; the arm is read UNINFORMATIVE ("sampler does not
  explore"), which is the user's concern made measurable, not a mechanism null.
- `sf_unique_strings`: distinct strings among the G samples, mean over utterances (1.0 = mode
  only); `sf_reward_mean`: the reward on the decode, i.e. log p_strong − log p_3 per utterance
  (the label-free proxy for the prior gap: it should rise toward gold's value, +2.3 nats/token in
  Step 0, if the term moves the code); expected phone rate; the rate FD check under its
  existing tolerance with the term on (code-review pass condition, as in the cdrev review).
- Disclosed label-using read at ep10 / ep25 / ep50 (nothing enters training or selection): the
  Step 0 prior-gap table rerun on the arm's decode (`PriorGapAnalysisJob` with the new
  private-code input), like-for-like pairing: the gap under p_strong should shrink from 2.0–2.3.

Pre-registered prediction: sf_50 moves `sf_reward_mean` up within the first 10 sub-epochs while
the code is still soft (frame NMI jumps between ep4 and ep10 in ctrl_50); if the code sharpens
first, `sf_reward_std_within` collapses and the arm reads UNINFORMATIVE. soft_50 is expected to
move the monitor regardless; whether it moves PER is the open question.

## Gate

**G4a.7** (per arm, dev-other, final sub-epoch or the clamp-reached sub-epoch recorded before
launch; never best-PER over the kept set; same form as G4a.4): greedy PER < 0.50 AND emitted rate
in [5.80, 14.49]/s; health: speaker-matched derangement gap > 0. Paired reads (PairedPerDeltaJob):
sf_50 vs ctrl_50, soft_50 vs ctrl_50, sf_50 vs soft_50. Read at the same sub-epoch count as
ctrl_50 (matched completion, not matched wall time). UNINFORMATIVE clause: the dead band above;
an UNINFORMATIVE arm licenses "this sampler at this bed does not explore" and a larger G or a
higher sampling temperature as the next arm, not "the prior lever fails". FAIL (PER >= 0.50 with
the monitor engaged) licenses not funding the outside-the-DP correction further; a lexicon inside
a new lattice is then the remaining route. Abort rule as G4a.4. A PASS is audited from a fresh
context and needs a second seed. Step 0 and 0b have read rules, not gates.

## Results

### Step 0: prior gap on dev-other, ctrl_50 ep10 (2026-09-20; job `PriorGapAnalysisJob.2RkbKYl0v1XK`; audited, `reports/audit_prior_gap_2026-09-20.md`, all readings CONFIRMED)

2864 utterances, nats per token, gold-minus-private paired gap; like-for-like pairing (gold and
the SIL-dropped decode) primary; SIL-kept pairing (the string the prior sees in training)
disclosed. Spread = |even-half − odd-half|.

| prior | gold | private | gap like-for-like | spread | gap SIL-kept | IS log-weight sd per utt (gold / private) | discriminates more than trigram |
|---|---|---|---|---|---|---|---|
| unigram WB (live) | −3.50 | −3.67 | 0.16 | 0.002 | 0.07 | – | no |
| bigram WB (live) | −3.25 | −3.78 | 0.52 | 0.005 | −0.01 | – | no |
| trigram WB (live) | −3.20 | −4.63 | 1.39 | 0.017 | 0.49 | – | reference |
| 4-gram KenLM MKN | −2.79 | −4.50 | 1.67 | 0.008 | 1.18 | 12.8 / 17.9 | yes |
| 6-gram KenLM MKN | −2.57 | −4.26 | 1.62 | 0.014 | 1.25 | 22.8 / 20.0 | yes (below the 4-gram) |
| lexicon STRICT (subset n = 609) | −2.15 | −4.66 | 2.51 (trigram on same subset 1.09) | 0.060 | 2.23 | 31.0 / 20.9 | yes (subset, disclosed) |
| lexicon ESCAPE (full set) | −2.19 | −4.46 | 2.32 | 0.016 | 2.18 | 31.7 / 24.3 | yes |

Segmentable under the strict lexicon: gold 0.876, private 0.237, gold null 0.024, private null
0.011. Null strings score 2 to 4 nats per token below their originals under every prior of order
2 and above.

Audit additions (independent re-derivation from the per-utterance dump, every banked aggregate
reproduced; KenLM re-scoring matches to 4e-6 nats): a modified Kneser-Ney TRIGRAM built from the
job's own window text gives gap 1.19, so the smoothing change alone lowers the gap by 0.20 and
the order-4 effect within one estimator is +0.48 (t = 85); the mixed-estimator table understates
the 4-gram. The 6-minus-4 gap difference is −0.044 with standard error 0.005, a real effect. The
escape row's fixed prices contribute 0.16 of its gap and lexical routing 2.17; with a free escape
the gap still exceeds the 4-gram's. The order-only log-weight sd is 12.7 / 17.6 nats per
utterance (0.18 / 0.25 per token), so the importance-sampling clause fails under either estimator
when read per utterance, which is the operative reading for a per-utterance weight.

Readings (audited):
1. The lexicon is where the private code is identified. The lexicon gap is 0.93 nats per token
   above the trigram's, against 0.27 for the 4-gram; three quarters of the private strings have no
   segmentation into vocabulary words at all, against one eighth of gold. Prediction held on the
   lexicon and on the nulls; the 4-gram gap grew more than "little" (about 20 %), with the caveat
   that the 4-gram and trigram rows use different smoothing estimators (KenLM modified Kneser-Ney
   vs live Witten-Bell), so part of that increase may be smoothing, a point given to the audit.
2. Phone n-gram order is not a proxy for lexical structure: the 6-gram discriminates less than the
   4-gram. The pre-registered "6/8-gram proxy" idea is dropped; a training reward has to score the
   lexicon itself or something that has learned it.
3. The importance-sampled 4-gram correction is closed: the per-utterance log-weight sd is 13 to 18
   nats against the pre-registered 3-nat ceiling (and 24 to 32 for the lexicon). Only a
   score-function estimator can carry a lexicalised prior into training.

Decision table outcome as printed: no row fires (the 4-gram discriminates more, so the lexicon
clause's "but the 4-gram does not" is unmet, while the 4-gram's own estimability condition fails).
Amendment, made after the numbers and recorded as such: the table did not anticipate "4-gram
discriminates more but is not estimable"; by the table's own logic the 4-gram route is conditioned
on estimability and is closed, and the lexicon row discriminates more, so the outcome is the
lexicon score-function term. The audit confirmed readings 1-3 and called the table silent as
written, with the amended reading a restatement of its own logic; it stands.

Consequence for the training arm: the reward must be a lexicon-level score that can be evaluated
on sampled strings inside a training step. The exact trie Viterbi costs about 1e5 extensions per
utterance in Python and is not step-rate compatible; Step 0b measures whether a small neural
phone-level LM trained on the same text learns the lexical constraint (its gap row against the
lexicon's 2.32) before the arm is designed around it.
