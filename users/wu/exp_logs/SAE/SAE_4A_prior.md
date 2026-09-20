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
amendments), so v2 is being written under a new hash: orders 4 and 6 only, two lexicon
conventions, per-utterance dump. Convention fixed before the numbers: the gold reference is
SIL-free and the decode carries SIL, so the job scores both pairings; the like-for-like pairing
(SIL dropped from the decode) is primary and the SIL-kept pairing is disclosed beside it. Sound
per the review and kept: trigram anchors reproduce the decipherment's −3.20 / −3.99 / −4.63, BOS
with no end-of-sentence term in every scorer, identical token denominators, the priorshuf
uniform window (`SampleLinesJob.orN768ARKwlt`) for every prior.
NEXT: implementer v2 report; executor restarts the manager on the v2 hash (the live manager holds
the v1 graph); re-arm the watcher; read the pre-registered table; then decide between the 4-gram
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

### Training arm

Written after Step 0; gate G4a.7 to be defined then, from the budget round's thresholds.

## Gate

G4a.7: not yet defined (see Design, "Training arm"). Step 0 has a read rule, not a gate.

## Results

(none yet)
