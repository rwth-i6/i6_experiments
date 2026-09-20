# SAE §4a — Prior strength: shrinking the family of codes the prior accepts

## State

Phase opened 2026-09-20 on the user's approval ("go ahead. approved."), replacing the
context-dependent reverse model (`SAE_4A_cdrev.md`, deferred without limit by the user the same
day, nothing built). Step 0 is a CPU diagnostic, no node: the gold-minus-private-code prior gap as
a function of prior order and of lexicalisation, on dev-other. Job built (`sae/emc/prior_gap.py`,
`PriorGapAnalysisJob.l0p0srBryKrs`, config `config/sae_4a_prior_gap.py`, speech-llm commit
1431dbf; `reports/impl_prior_gap_2026-09-20.md`), launched under its own manager (pid 2571413,
`log/sae_4a_prior_gap.manager.20260920T185700Z.log`, Slurm 1916086; watcher
`bash ~/.claude/skills/sis/sis_watch.sh 2571413 config/sae_4a_prior_gap.py 300`, re-arm first
after any session resume, together with the budget and infomax watchers named in their phase
files). The v1 job failed before producing numbers (KenLM compiled with max order 6, the 8-gram
build aborted; manager exited, watcher done) and the code review
(`reports/review_prior_gap_2026-09-20.md`) found the lexicon row biased (see Design, "Step 0",
amendments), so v2 runs under a new hash (`PriorGapAnalysisJob.2RkbKYl0v1XK`, speech-llm commit
291dab1; orders 4 and 6 only, two lexicon conventions, per-utterance dump): manager pid 2790865,
`log/sae_4a_prior_gap.manager.20260920T192919Z.log`, Slurm 1916711; watcher
`bash ~/.claude/skills/sis/sis_watch.sh 2790865 config/sae_4a_prior_gap.py 300` (re-arm first
after any session resume). The v1 pointers above are superseded. Convention fixed before the numbers: the gold reference is
SIL-free and the decode carries SIL, so the job scores both pairings; the like-for-like pairing
(SIL dropped from the decode) is primary and the SIL-kept pairing is disclosed beside it. Sound
per the review and kept: trigram anchors reproduce the decipherment's −3.20 / −3.99 / −4.63, BOS
with no end-of-sentence term in every scorer, identical token denominators, the priorshuf
uniform window (`SampleLinesJob.orN768ARKwlt`) for every prior.
Step 0 finished (5 min) and is read in Results; audit dispatched
(`reports/audit_prior_gap_2026-09-20.md` when it lands); Step 0b (neural phone LM row) is
pre-registered in Design and its implementer dispatched.
NEXT: audit verdict on the Step 0 readings and the decision-table amendment; Step 0b job (train
the phone LM on GPU, rescore in a new PriorGapAnalysisJob instance) launched under the prior_gap
config's manager; read its rule; then decide between the 4-gram
importance-sampled correction and the lexicon score-function term and write that arm's design and
gate here before any node is funded.

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
39 ARPAbet phones + SIL, held-out perplexity 9.469 (3.243 bits/phone) on 10,000 held lines of the
LibriSpeech LM text (`reports/estimate_prior_order_2026-09-15.md`); the order-4 KenLM
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

### Training arm

Written after Step 0b; gate G4a.7 to be defined then, from the budget round's thresholds. Sketch
fixed by Step 0: a score-function term on strings sampled from the lattice posterior
(forward-filtering backward-sampling in the trigram DP), reward log p_strong(y) − log p_3(y)
with a per-utterance mean baseline, active from sub-epoch 1, within-group reward variance logged
as the engagement monitor; the strong scorer is what Step 0b selects.

## Gate

G4a.7: not yet defined (see Design, "Training arm"). Step 0 has a read rule, not a gate.

## Results

### Step 0: prior gap on dev-other, ctrl_50 ep10 (2026-09-20; job `PriorGapAnalysisJob.2RkbKYl0v1XK`, audit pending)

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

Readings (pending audit):
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
lexicon score-function term. The amended reading stands only if the audit confirms readings 1-3.

Consequence for the training arm: the reward must be a lexicon-level score that can be evaluated
on sampled strings inside a training step. The exact trie Viterbi costs about 1e5 extensions per
utterance in Python and is not step-rate compatible; Step 0b measures whether a small neural
phone-level LM trained on the same text learns the lexical constraint (its gap row against the
lexicon's 2.32) before the arm is designed around it.
