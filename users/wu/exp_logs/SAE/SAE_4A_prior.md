# SAE §4a — Prior strength: shrinking the family of codes the prior accepts

## State

Phase opened 2026-09-20 on the user's approval, replacing the context-dependent reverse model
(`SAE_4A_cdrev.md`, deferred without limit). Active: the soft / sf pack of the reopened training arm (section "Training arm, reopened by user
ruling"). Scorer: the approved 3.3 M instance (a), `NeuralPhoneLmTrainJob.xObXEwRpvmzd` selected
epoch 10. Arms sf_20 / soft_20 / soft_20_s1 / softshuf_20 against the frozen prepro ctrl_20 /
ctrl_20_s1; LAM_01 = 0.326639, LAM_SF_01 = 0.0453938 (Results, lam probes). Earlier steps are
closed and read in Results: Step 0 audited CONFIRMED; Step 0b closed on its third clause
(overturned for funding by the user ruling); falsifier (ii) stands as a prediction for sf_20. The
lexlat successor is PAUSED by the user after its E1 cost failure (`SAE_4A_lexlat.md`); survey and
literature for it are banked (Results, falsifier (ii)).

Run pointer: pack hash **`PackedBlankfreeTrainJob.MXKoywbfon8O`** (196 jobs;
`reports/exec_soft_pack_launch_2026-09-21.md`): pack manager pid 409858
(`log/sae_4a_soft_pack.manager.20260921T092744Z.log`, must stay alive for the 196-job graph),
Slurm 1926421 (jpbo-078-23, exclusive 4 GPU, 11.5 h); watcher `bash ~/.claude/skills/sis/sis_watch.sh 409858
config/sae_4a_soft_pack.py 600` (re-armed 2026-09-21 as background id b5adkh067; re-arm first
after any resume). The ep1 efficiency clause passed with all four arms kept (Results "Soft pack,
ep1 read"); interim ep10 read: no arm beats its control (Results "Soft pack, interim read"). G4a.7
unread. Open from the code reviews: census counts not re-derived, tests not executed by the
reviewer (Training arm, reopened: "Build and review notes").
NEXT: when the pack starts: executor reads step-1 loss per arm and sec per sub-epoch at ep1, check sec per sub-epoch at ep1 per arm against 2.00 × 601 s (sf_20 dropped from the
pack rather than delaying it if it alone exceeds); reads at kept epochs 1 / 4 / 10 / 20 against
G4a.7 with the paired margin rule; sf_20's UNINFORMATIVE read at ep10 / ep20.

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

SIL convention of the bed (user question 2026-09-20, "why SIL if rVAD trims silence; did Meta use
both?"): the prior text is phonemised with sil_prob 0.5 and surrounding SIL
(`PhonemizeWithSilJob.DbFgvZOGZQ8F`), which makes 13.8 % of its tokens SIL (0.160 per phone;
first 200k window lines). wav2vec-U used rVAD AND text-side SIL insertion; the local port runs
sil_prob 0.25 with `--surround` (`users/enrique/.../wav2vec_u/full_pipeline.py:97`). Gold on
dev-other after the bed's rVAD (label-using diagnostic, enters nothing;
`analysis/gold_sil_share_retained.py`, `reports/impl_gold_sil_retained_2026-09-20.md`; the
retained 50 Hz count 781,130 and the 60 ms output count 261,295 reproduce the banked totals):

| quantity | value |
| --- | --- |
| gold SIL frames before VAD | 20.3 % of 919,980 |
| gold SIL frames after VAD (retained) | 7.9 % of 781,130 |
| gold SIL runs on retained frames | 5.7 % of 186,083 runs (0.060 per phone run) |
| gold SIL run length on retained frames | median 100 ms, 28 % are 1–2 frames, 13 % ≥ 200 ms |
| decode SIL tokens, ctrl_50 ep10 | 4.4 % |
| prior text SIL tokens | 13.8 % |

Reading: rVAD removes 60 % of gold silence but 7.9 % of retained frames are still silence, so
the SIL symbol is needed; the prior expects SIL 2.4x more often than gold shows it and the decode
sits closer to gold than the prior does. Paper check (`reports/lit_w2vu2_preprocessing_2026-09-20.md`,
full text of arXiv 2204.02492v2 §4.1): wav2vec-U 2.0 ALSO removes audio silence with rVAD before
feature extraction and inserts SIL at word boundaries with probability 0.5 plus sentence-edge
SIL, with no justification and no sweep; "no audio pre-processing" in its abstract refers to
segmentation, k-means, PCA and pooling only. So the bed's rVAD + 0.5 is the reference
combination, not a mix-up; the only unmeasured departure is that we mask features after
full-waveform SSL extraction while the paper cuts the waveform first (already disclosed in
`SAE_4A_blankfree.md`). wav2vec-U 1.0's sweep (its Fig. 5) picked 0.25 and found 0.5 worse at
1.0's operating point, and removing audio silence altogether cost 8 PER there. A sil_prob 0.25
text (about 7–8 % SIL tokens) would match gold's token share; it is a departure from the
reference, a bed change (new prior, new hashes), a candidate arm for this phase's queue, and not
ahead of the training arm.

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

**Design review amendments (2026-09-20, `reports/design_review_prior_arm_2026-09-20.md`, STOP as
written, approvable with A1–A4; all applied before any job, the text above is kept as the
original):**
- A1, reward. The trigram baseline is a sign trap: from the Step 0b per-set table (nats/token)
  the trigram scores shuffled gold −7.19 against gold −3.20 while the neural LM scores them −5.65
  / −2.56 and the escape lexicon −4.48 / −2.19, so log p_strong − log p_3 pays a shuffled gold
  string 0.9 (neural) / 1.7 (lexicon) nats per token MORE than gold and a shuffled private string
  1.4 / 2.8 more than the private string: on every non-lexical string the term is −log p_3 and
  weakens the trigram. Amended reward: r(y) = log p_strong(y) − log p_uni(y) with the window's
  unigram as the order-insensitive baseline (neural − unigram: gold +0.95, private −0.67, nulls
  −2.2 nats/token; lexicon − unigram: +1.31 / −0.79 / −1.0). Monitor target for `sf_reward_mean`
  corrected: the decode's value should rise from −0.67 toward +0.95 (neural filling), not "+2.3".
- A3, dead band replaced. ctrl_50's sampler already draws 494–510 distinct strings of 512 at
  tau 2 (`SAE_4A_blankfree.md`), so a within-group std floor never fires and a real sampler
  failure would read FAIL. The UNINFORMATIVE clause is now: the disclosed, label-using fraction of
  dev-other utterances in which r(gold) exceeds max_g r(y_g) (the samples never reach a string
  the reward prefers), read in the pre-launch probe and at ep10 / ep25 / ep50, together with the
  within-group std calibrated by the same probe (not the 1.0 nats/utt guess).
- A2, soft arm. Straight-through (hard argmax string forward, soft gradient), the same frozen
  scorer; artefact monitor = scorer(soft) − scorer(hard) per token with a pre-registered ceiling
  of 0.3 nats/token above which the arm's reads are void; the unigram baseline is linear in the
  soft vector, so it needs no straight-through and no soft trigram is defined.
- A4, implementation. No differentiable fixed-string DP: by the Fisher identity the sampled
  path's own log weight (frame log-q gathers plus segment scores; log Z cancels under centred
  advantages) is an unbiased estimator of grad log q(y | x), so the term is A_g times that path
  score. Length policy for the neural filling: a sampled string longer than the scorer's 512
  positions is masked out of the term and counted (`sf_masked_long`), never truncated silently.
  SIL: both scorers see the SIL-dropped string (Step 0's primary pairing), because with SIL kept
  the term pushes SIL out unopposed by the rate term, which counts non-SIL tokens only. Resume:
  the arm is not funded before the budget pack's 11.5 h resume is seen to work; the fallback read
  at ep25 (ctrl_50 keeps it) is pre-registered now. Trie filling: Step 0's lexicon row carries a
  word-trigram state that a batchable GPU trie DP would not; word-unigram and word-bigram ESCAPE
  rows are banked in the same job (CPU) before any DP is built.
- Pre-launch falsifier (replaces "measured in the 100-step probe"): (i) CPU neighbourhood probe
  on the banked ctrl_50 ep4 and ep10 decodes, K = 8 single-token substitutions / deletions per
  utterance drawn from the frame posterior's second choice, scored under the A1 rewards and the
  trigram: within-neighbourhood std, correlation of the reward delta with the trigram delta, and
  the fraction of edits that gain a strict-lexicon word; (ii) once the blank-free FFBS exists,
  G = 8 draws on 300 dev-other utterances at ctrl_50 ep1 / ep4 / ep10 at the schedule's tau:
  distinct strings, within-group std, the r(gold) > max_g fraction, and the gradient-norm ratio
  that fixes lam_sf at ep4 and at ep10 (recorded, ep4 preferred over step 1). Rule: if r(gold)
  exceeds max_g in more than 95 % of utterances at every checkpoint, sf_50 is not funded and the
  lattice-internal lexicon design is the next arm.
- A5 (2026-09-21, pinned before probe (ii) runs; `reports/impl_blankfree_sampler_2026-09-20.md`
  item 2): the sf term is normalised PER RETAINED FRAME, i.e. each utterance's score-function term
  is divided by its retained unit count, exactly as the lattice term l_tau = mean_b(−log Z_b /
  retained_b) is, then averaged over the batch. Rationale: the unit count is constant across the
  G draws of an utterance, so it cannot pay for length (the lm_prior_norm = "units" sign
  guarantee), and lam_sf then scales two per-frame quantities. The probe's `utterance_mean`
  variant is disclosed only; lam_sf is read from the `per_frame` gradient-norm ratio (median over
  batches, 0.1x preferred, 0.3x the ceiling). The lexicon ESCAPE reward is not in probe (ii) (its
  word LM is fitted inside the analysis job); its neighbourhood behaviour is read from probe (i).

## Gate

**G4a.7** (per arm, dev-other, final sub-epoch or the clamp-reached sub-epoch recorded before
launch; never best-PER over the kept set; same form as G4a.4): greedy PER < 0.50 AND emitted rate
in [5.80, 14.49]/s; health: speaker-matched derangement gap > 0. Paired reads (PairedPerDeltaJob):
sf_50 vs ctrl_50, soft_50 vs ctrl_50, sf_50 vs soft_50. Read at the same sub-epoch count as
ctrl_50 (matched completion, not matched wall time). UNINFORMATIVE clause (original: the dead
band; amended by the design review A3 before any job): the r(gold) > max_g fraction and the
probe-calibrated within-group std, as in the amendments; an UNINFORMATIVE arm licenses "this
sampler at this bed does not reach the strings the reward prefers" and a larger G or a higher
sampling temperature as the next arm, not "the prior lever fails". Reward as amended (A1):
strong minus unigram; the pre-launch falsifier's 95 % rule decides whether sf_50 is funded at all. FAIL (PER >= 0.50 with
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

### Step 0b: the 3.3 M neural phone LM (2026-09-20; `NeuralPhoneLmTrainJob.Iv6P6YVPNWmB`, scored in `PriorGapAnalysisJob.Gct95xZHe0zt`; not yet audited)

Held-line benchmark (10,000 held lines, 806,207 tokens, every model on the same strings, BOS
context, no end-of-sentence term, pooled nats per token):

| model | perplexity |
| --- | --- |
| trigram (Witten-Bell, live) | 9.557 (expected 9.561, reproduced) |
| 4-gram KenLM modified Kneser-Ney | 7.271 |
| 6-gram KenLM modified Kneser-Ney | 5.424 |
| neural phone LM, 3.3 M, epoch 3 of 3 | 5.774 |

Prior gap (gold − private, nats per token, dev-other, ctrl_50 ep10): like-for-like 1.71 (spread
0.009, n = 2863), i.e. above the 4-gram's 1.67 and the 6-gram's 1.62 but below the 2.01 bar;
strict subset 1.52 against the lexicon STRICT row's 2.51 (bar: within 0.3). SIL-kept 1.40.
Read by the pre-registered rule: **partial proxy**, not an adequate lexical scorer. Under the size
rerun rule it does not discharge Step 0b.

Training-log reading, made after the numbers: the job's default schedule was 3 epochs of cosine
decay to zero (8,289 steps, 125 s of GPU; the 4 h cap was irrelevant), the held curve was still
falling at the last epoch (6.47, 5.87, 5.77) and the train-minus-held gap is 0.001 nats, so the
model is schedule-bound, not capacity-bound: the 3.3 M model is worse than the 6-gram in
perplexity while it has seen the counted lines three times. Amendment to the size rerun rule
(recorded before the rerun is launched): the rerun is two instances, same data, split, held
lines, optimiser and conventions, 30 epochs each under the same 4 h cap and the existing
best-held-epoch / no-improvement abort: (a) 4 layers / width 256 (3.3 M, the schedule delta
alone) and (b) 6 layers / width 384 (about 10 M, the pre-registered size delta). Both are scored
in the same prior-gap job. Read rule unchanged: 2.01 bar and strict-subset tracking. Expected
if the lexicon is learnable from this window: (b) below the 6-gram's perplexity and a gap above
2.01; if both instances plateau near the 6-gram's gap (1.6–1.7), a phone-level LM of this size
does not learn the lexical constraint from 1 M lines and the GPU trie DP is the scorer.
Instance (c), added under the user's directive to reach the gate (recorded before its launch;
`reports/impl_phone_lm_v2_2026-09-20.md`, `reports/review_phone_lm_v2_2026-09-20.md`,
APPROVE_WITH_AMENDMENTS = disclosures only): 6 layers / width 768-class, 25.5 M params, trained on
10.09 M lines (`SampleLinesJob.tHBnfzwuo9ok`, seed 1) with the benchmark's 10,000 held lines removed
by content (`ExcludeLinesJob.nOMT5eGtLKDV`) and used as the fit's own held-out set
(`HeldLinesJob.SKxs9aPu2Sha`, same order, respelling and 512 truncation as the banked benchmark),
5 epochs, same optimiser and conventions, `NeuralPhoneLmTrainJobV2.vkNGAOeLgNsy`,
`PriorGapAnalysisJob.OO0iAEVLKgOO` (differs from `.Gct95xZHe0zt` in name and neural_lm only).
Disclosures: (c) moves text, capacity and epochs at once, so its row answers "does more of
everything reach the bar", never what the extra text alone bought; the benchmark's n-gram rows
remain fitted on the 1 M-line window, so neural-vs-n-gram is not like-for-like in training data;
the bar (2.01) and the strict reference (2.51) come from the trigram / lexicon rows and are
unchanged. Reported held perplexity is the min over epochs on the same held set, as for (a)/(b).

**Rerun result (instances (a) and (b), read 2026-09-20 from
`output/.../sae_4a_prior_gap/ctrl_50_ep10/dev-other_neural_e30/prior_gap.md` =
`PriorGapAnalysisJob.5wNIQs2lpC5P` and `.../dev-other_neural_l6w384_e30/prior_gap.md` =
`.pg14aEYJyiva`; manager exited cleanly, `reports/exec_prior_gap_restart_2026-09-20.md`):**

| instance | params | epochs run / selected | held ppl (10,000 lines) | gap like-for-like | strict subset (609) vs 2.5057 | 4-gram gap | verdict |
|---|---|---|---|---|---|---|---|
| (a) 4L / w256, 30 ep | 3.3 M | 11 / 10 (no-improvement abort) | 5.091 | 1.861 | 1.646 (distance 0.86) | 1.666 | partial proxy |
| (b) 6L / w384, 30 ep | 10.9 M | 11 / 10 (no-improvement abort) | 4.603 | 1.849 | 1.663 (distance 0.84) | 1.666 | partial proxy |
| first pass, 3 ep | 3.3 M | 3 / 3 | 5.774 | 1.71 | 1.52 | 1.666 | partial proxy |
| (c) 8L / w512, 5 ep, 10.09 M lines | 25.5 M | 5 / 5 (no abort; held ppl still falling 4.321 -> 3.963) | 3.963 | 1.809 (bootstrap 1.801–1.816) | 1.693 (distance 0.81) | 1.666 | partial proxy |

Instance (c) read 2026-09-21 (`NeuralPhoneLmTrainJobV2.vkNGAOeLgNsy`, `PriorGapAnalysisJob.OO0iAEVLKgOO`,
`output/.../sae_4a_phone_lm_v2/ctrl_50_ep10/dev-other_neural_10m_l8w512/prior_gap.md`; manager
exited cleanly, `reports/exec_phone_lm_c_finish_2026-09-21.md`). Ten times the text and 2.3x the
parameters of (b) lower the held perplexity from 4.60 to 3.96 and LOWER the gap from 1.849 to 1.809:
the stronger phone LM scores gold better (−2.40 vs −2.57 nats/token for the 6-gram) but scores the
decode better still (−4.22 vs −4.26), i.e. the decode's errors are phonotactically fluent and a
phone-sequence model, however strong, rewards that fluency; the lexicon's excess discrimination
(escape row 2.32, strict 2.51) comes from the word constraint, which no n-gram or neural phone LM in
this family approaches (all between 1.62 and 1.86). **Step 0b closes on its pre-registered third
clause in its data-and-capacity limit: no phone LM trained on this text reaches the bar, and the
exact lexicon must be made batchable (GPU trie DP) before a strong-scorer arm exists.** The soft arm
with a phone-LM scorer is therefore not reconsidered; the score-function arm stays unfunded
(falsifier (ii)). Not audited: the read is a mechanical comparison against a pre-registered
threshold on a registered job's output; the direction it selects was pre-registered as the fallback.

User directive 2026-09-20 (`SAE.md`) item 3 is met in its negative branch: the phone LM was
trained to the data-and-capacity limit and does not reach the gate (moved from State).

Bar recomputed from the run's own rows: 2.012 (pre-registered 2.01). Neither instance meets it;
neither tracks the lexicon on the strict subset (tolerance 0.30). Reading against the pre-registered
expectation: both now beat the 6-gram in perplexity (5.42) and the 6-gram in gap (1.62), but the gap
plateaus at 1.85 for a 3.3x parameter increase and a 0.5 nat perplexity gain, and both fits ended
by the no-improvement abort, so the model is no longer schedule-bound: the ceiling is the 1 M-line
window. Instance (c) (10 M lines, 25.5 M params, running) is the last phone-LM test of the data
axis; if it also lands below 2.01 or off the strict track, the pre-registered fallback holds and
the GPU trie lexicon DP is the scorer (survey of the lexicon scoring path started now so that the
fallback is specifiable the moment (c) reads).

### Pre-launch falsifier (i): neighbourhood probe on the ctrl_50 decodes (read 2026-09-21)

`NeighbourhoodProbeJob.ZhYOeCltF4wW` (ep4) / `.6JZvwE7T1UmA` (ep10), speech-llm d22d81d, 2864
dev-other utterances, K = 8 single-token edits per utterance (second-choice substitutions and
deletions on the frame sequence, collapsed, SIL dropped), conventions in the module docstring
(`reports/impl_prior_probe_2026-09-20.md`, `reports/extract_prior_probe_2026-09-21.md`). Rewards
in nats per utterance; "gold above max" = fraction of utterances with r(gold) > max over
{decode} U edits (disclosed, label-using, enters nothing).

| reward | ckpt | std over neighbourhood (median) | corr of reward delta with trigram delta (Pearson) | gold above max | mean edit delta | decode / gold nats per utt |
|---|---|---|---|---|---|---|
| A1 neural − unigram | ep4 | 4.00 | +0.65 | 0.9993 | +1.64 | −95.1 / +58.7 |
| A1 neural − unigram | ep10 | 4.23 | +0.68 | 0.9920 | −1.32 | −39.0 / +58.7 |
| A1 lexicon ESCAPE − unigram | ep4 | 0.50 | +0.42 | 0.9997 | +0.46 | −42.6 / +81.0 |
| A1 lexicon ESCAPE − unigram | ep10 | 2.41 | +0.36 | 0.9972 | −0.33 | −46.4 / +81.0 |
| original neural − trigram | ep4 | 5.03 | −0.73 | 0.0552 | −1.36 | +85.9 / +39.9 |
| original neural − trigram | ep10 | 5.01 | −0.75 | 0.6390 | +1.81 | +17.2 / +39.9 |

Lexical content: the decode segments under the strict lexicon in 5.2 % (ep4) / 23.7 % (ep10) of
utterances against 87.6 % for gold; single-token edits raise the strict word count in 1.5 % /
4.2 % of edits and the escape-row real-word count in 3.2 % / 10.1 %.

Reading (probe (i) only; the funding rule is read on (ii), the FFBS draws):
- The sign trap is real and A1 removes it: the original reward (neural minus trigram) scores the
  decode ABOVE gold at ep4 and its local deltas anti-correlate with the trigram (−0.73); both A1
  rewards put gold far above the decode and correlate positively with the trigram locally.
- Under A1 the neighbourhood is not a dead band (std 4 nats per utterance for the neural reward),
  but gold exceeds every string of the neighbourhood in > 99 % of utterances at both checkpoints,
  above the 95 % line at the single-token scale. The local signal of the neural reward tracks the
  trigram (+0.65 / +0.68): at the edit scale the strong scorer mostly restates the prior already in
  the DP. The lexicon reward is nearly flat around the ep4 decode (std 0.5) because almost no
  single edit creates a dictionary word; it only wakes up at ep10 where a quarter of the decodes
  segment.
- Expectation for (ii): at ep10 (tau 2) the FFBS draws are local edits and will reproduce these
  numbers; at ep1 / ep4 (tau 8 / 5) the draws are diverse and the r(gold) > max_g fraction is the
  open measurement. If it also exceeds 95 % at every checkpoint, the rule stands: sf is not funded
  and the lattice-internal lexicon design is the next arm, which coincides with the Step 0b
  fallback (GPU trie lexicon DP, `reports/survey_lexicon_scorer_2026-09-20.md`).
- Word-bigram ESCAPE row (`PriorGapAnalysisJob.m6lUxAO65f6A`): gap 2.287 against the trigram
  row's 2.321, strict 2.512 vs 2.506; the word-LM order above bigram buys 0.03 nats per token, the
  lexicon itself carries the discrimination. The word-unigram row stays deferred (KenLM order 1).

### Pre-launch falsifier (ii): FFBS draws on the blank-free lattice (read 2026-09-21)

`SampledRewardProbeJob.d1NoUQJ2EXN5` (ep1, tau 8) / `.Y81PrZ6fWWKu` (ep4, tau 5.04) /
`.HVHIlaUkIkVi` (ep10, tau 2), speech-llm 2ae6cc5, 300 fixed dev-other utterances, G = 8 exact
posterior draws at the schedule's tau (rebuild check: greedy decode of the rebuilt model matches
the banked decode on 292 / 292 / 299 of 300; max log w − log Z ≤ 0 everywhere; no Z = 0). Reward
r = neural (first-pass 3.3 M, `Iv6P6YVPNWmB`) minus unigram, SIL dropped, nats per utterance.
Audit from a fresh context (`reports/audit_sf_probe_2026-09-21.md`): CONFIRMED; fractions
recomputed from per_utt.json 300/300, 300/300, 291/300; strict >, max over the 8 draws only, same
SIL and tokenisation on both sides; draws genuine at the asserted tau; the 8 / 1 rebuild mismatches
are single argmax near-ties and cannot move the fraction; ep10 not fragile (next positive margins
+1.5 / +1.9 / +6.7 nats); the scorer is the weakest banked LM (first pass), a stronger one raises
r(gold); one empty-gold utterance counts as a pass (290 / 299 = 0.970 without it).

| ckpt | distinct strings of 8 | tokens per string sample / greedy / gold | r per token gold / greedy / sample | within-group std (median) | r(gold) > max_g fraction | r(gold) > r(greedy) |
|---|---|---|---|---|---|---|
| ep1 | 8.00 | 80.1 / 21.5 / 62.9 | +0.84 / −4.12 / −1.42 | 14.9 | **1.0000** | 1.0000 |
| ep4 | 8.00 | 69.9 / 36.2 / 62.9 | +0.84 / −2.76 / −1.09 | 12.0 | **1.0000** | 1.0000 |
| ep10 | 7.36 | 57.5 / 59.8 / 62.9 | +0.84 / −0.72 / −0.48 | 5.1 | **0.9700** | 0.9900 |

Gradient-norm ratio (sf over l_tau, A5 per-frame convention, median over 3 batches): 5.0 at ep4
(lam_sf 0.020 for 0.1x, 0.060 for 0.3x), 1.8 at ep10 (0.055 / 0.166); the utterance_mean
convention is disclosed in the reports and differs by the 1 / retained factor.

**Verdict under the pre-registered rule: r(gold) exceeds every draw in more than 95 % of
utterances at EVERY checkpoint (100 %, 100 %, 97 %), so the score-function arm (sf_20) is NOT
funded and the lattice-internal lexicon design is the next arm.** What the numbers say beyond the
rule: the draws are diverse (8 of 8 distinct at tau ≥ 5) and the reward is not a dead band (std
5–15 nats per utterance), so the term would have moved parameters; but even the best of 8 draws
sits 45–75 nats per utterance below gold at every checkpoint, the sampled strings are scored
better than the greedy decode at every checkpoint (−1.4 vs −4.1 per token at ep1), and probe (i)
showed the neural reward's local ordering tracks the trigram already in the DP. A reward whose
gradient never sees a string near the lexical region cannot supply the lexical constraint; it can
only re-weight trigram-typicality. The soft (straight-through) arm shares the scorer and the
same neighbourhood, and is not funded either while the scorer is a partial proxy; it is
reconsidered only if instance (c) meets the Step 0b bar. Gate G4a.7 stands unread (no arm ran).

Literature for the next arm (`reports/lit_lexicon_in_objective_2026-09-21.md`, read 2026-09-21;
only findings that constrain the design): the lexicon INSIDE the marginalised objective has
precedent, Klejch, Wallington, Bell (Interspeech 2022: Baum-Welch on the composed
acoustic-lexicon-word-LM transducer, 100k-word LM, but a grapheme lexicon, 20 min of speech, and a
mandatory char-LM curriculum first) and Nuhn and Ney (ACL 2014: exact forward-backward over a
word LM is intractable above about 200 word types; V = 3,661 with a trigram needs beam
preselection). wav2vec-U 1.0 / 2.0, EURO, REBORN and Chen 2019 use the lexicon and word LM in
decoding and self-training only; Yeh 2019 matches phone 5-gram statistics without a lexicon; Ni
2025's word-level attempt forces a closed vocabulary ≤ 4,096 and fails on rare words. Yang,
Schlüter, Ney (2026, arXiv 2603.02285) propose exactly the loss −log Σ_c p_LM(c) q(x|c) but only in
theory and simulation. Failure modes reproduced across groups: frequent-pattern collapse,
length / deletion exploits (deletion penalty, unnormalised LM sums, length rewards), unbounded
OOV. Design constraints taken from this: the lexicon term is switched on LATE (after the
trigram-only warm phase), the trie/word-LM state space is pruned under a declared budget reported
as a cost / quality curve, per-retained-frame normalisation with an anti-deletion guard, a
frequency-stratified word-error read, and a shuffled-pronunciation null (same trie topology,
pronunciations permuted across words) as the destroyed-structure control. Nobody has published the
trigram-only vs trigram-plus-lexicon ablation in this setting.

Lexicon-scorer survey for the successor (`reports/survey_lexicon_scorer_2026-09-20.md`): exact
Viterbi, 23–38 live states per position, trie 151,731 words, about 0.1 s per utterance in pure
Python.

## Training arm, reopened by user ruling (2026-09-21)

User ruling 2026-09-21 (verbatim intent): "I approve the previous 3.3M transformer LM scoring,
chance still much better than 3gram." Read as: the 3.3 M neural phone LM is approved as the
scorer p_strong for the training arm although it missed the Step 0b bar (gap 1.861 against the
bar 2.01; the trigram in the DP prices the private code 1.39 below gold). Step 0b's third clause
is overturned for funding purposes by the user; the phase's other rules stand. Falsifier (ii)'s
rule is about the sampler, not the scorer (a stronger scorer raises r(gold) further), so the
score-function arm stays unfunded; the funded arm is the **soft (straight-through) arm** of the
pre-registered design (A2, A4, A5), which is also the user's own mechanism (2026-09-20).

Pack spec, fixed before the build (amendments to "Training arm"; original text kept above):
- Schedule N = 20 (user item 1, `SAE_4A_prepro.md`), kept epochs 1 / 4 / 10 / 20; controls are
  the frozen ctrl_20 / ctrl_20_s1 of the prepro pack (checkpoints and dev-other decodes), never
  retrained. Term active from sub-epoch 1 as pre-registered, constant lam_soft.
- Scorer: instance (a) 4L / w256, selected epoch 10, held ppl 5.091 (the `dev-other_neural_e30`
  row, `NeuralPhoneLmTrainJobV2`), frozen, consumed as a path. Reward per utterance
  r(y) = log p_strong(y) − log p_uni(y) (A1; unigram of the 1 M-line window), SIL dropped on both
  sides (A4), strings beyond the scorer's 512 positions masked out and counted (`soft_masked_long`).
- Straight-through (A2): segmentation and symbols from the max-plus (Viterbi) path of the
  blank-free lattice; per segment the soft vector is the lattice's conditional posterior over the
  40 symbols with the neighbouring Viterbi symbols held fixed (its argmax is the Viterbi symbol,
  since the joint maximum is a conditional maximum); forward on the hard string, gradient through
  y_soft = onehot + (p − stop_grad(p)) into the scorer's embedding; the segmentation itself gets
  no gradient (disclosed). Term = −lam_soft × mean_b[ r(ŷ_b) / retained_b ] (A5, per retained
  frame; minus because the loss is minimised).
- lam_soft from the pre-registered gradient-norm rule: probe of 3 batches at ctrl_20 ep4 (seed 0),
  median ratio of the soft term's gradient norm to l_tau's; arms take 0.1× (preferred) and 0.3×
  (ceiling). Recorded before the pack launches.
- Arms, one exclusive 4-GPU node: `soft_20` (0.1×, seed 0), `soft_20_s1` (0.1×, seed 1),
  `soft_20_r03` (0.3×, seed 0), `softshuf_20` (0.1×, seed 0, destroyed-structure null: the
  scorer sees the 39 non-SIL symbols through a fixed recorded derangement, so the reward keeps its
  statistics but not the phone identities). Pairings: soft_20 − ctrl_20, soft_20_s1 − ctrl_20_s1,
  soft_20_r03 − ctrl_20, softshuf_20 − ctrl_20, and soft_20 − softshuf_20; seed band
  ctrl_20_s1 − ctrl_20 (banked, −0.001 [−0.004, +0.001] at ep20).
- Monitors per sub-epoch (label-free): `soft_reward_mean` per token on the hard string (target:
  rise from about −0.67 toward +0.95), artefact gap scorer(soft) − scorer(hard) per token with the
  0.3 nats/token ceiling above which the arm's reads are void, `soft_masked_long`, expected rate,
  the rate FD check. Disclosed label-using read at ep10 and ep20: the Step 0 prior-gap table on
  the arm's decode (like-for-like pairing).
- Gate: G4a.7 as written, read at ep20 (matched completion): greedy PER < 0.50, rate in
  [5.80, 14.49]/s, derangement gap > 0; paired deltas against the frozen controls with the margin
  rule of `SAE_4A_lexlat.md` G4a.9 (delta ≤ −M, M = max(|seed band|, |null delta|, 0.010)); the
  null arm must not beat its control by more than the seed band or the read is void. Abort rule as
  G4a.4. Efficiency: sec per sub-epoch at ep1 read off the pack log; the pack is stopped if it
  exceeds 2.00 × 601 s (the scorer forward on ≤ 128 strings × ≤ 512 positions should cost well
  under 1 s per step).
- Cost: one exclusive node for about 3.4 h (13.6 GPU-h charged) plus reads (about 10 GPU-h) plus
  the lam probe (minutes).

**User override 2026-09-21 ("do it, try 8 samples as well"):** the score-function arm is funded
after all, as a hedge against the one-best basin concern the user raised, in the slot of the 0.3×
strength point. Falsifier (ii)'s rule stands as a prediction, not a block: the arm's UNINFORMATIVE
clause (A3) is what it is read against. Pack becomes: `soft_20` (0.1×, seed 0), `soft_20_s1`
(0.1×, seed 1), `sf_20` (G = 8 exact FFBS draws from the blank-free lattice posterior at the
schedule's tau, reward r(y) = log p_strong(y) − log p_uni(y) with the SAME approved 3.3 M scorer,
SIL dropped, > 512 masked and counted, centred advantages over G, Fisher-identity path score
(A4), per-retained-frame normalisation (A5), lam_sf at 0.1× of the l_tau gradient norm measured by
its own probe at ctrl_20 ep4 with THIS scorer, not the first-pass ratio of falsifier (ii)),
`softshuf_20` (0.1×, seed 0, null). `soft_20_r03` is dropped. Added pairings: sf_20 − ctrl_20,
sf_20 − soft_20. sf monitors per sub-epoch: `sf_reward_mean` (hard decode and sample mean),
`sf_reward_std_within` (median over utterances), `sf_unique_strings`, `sf_masked_long`; disclosed
label-using read at ep10 / ep20: the r(gold) > max_g fraction on the 300 falsifier utterances
(UNINFORMATIVE if ≥ 0.95 at both, as A3) and the prior-gap rerun. Gate G4a.7 unchanged. Cost
per step: one extra DP pass (the checkpointed backward recomputation carries the G draws), read
at ep1 against the same 2.00 × 601 s stop rule as the soft arms; if sf_20 alone exceeds it, the
pack runs without it rather than delaying the three soft arms.

Build and review notes (moved from State). Round 1 (soft arm) built in speech-llm f99f9f6
(`reports/impl_soft_arm_r1_2026-09-21.md`: `sae/emc/soft_scorer.py`, `soft_scorer_jobs.py`,
default-off `soft_*` block, `configs/config_sae_4a_soft_pack_v1.py` + shims
`config/sae_4a_soft_{probe,pack}.py`, 22 tests; census unchanged). **Scorer correction:** the scorer
resolves on disk to `NeuralPhoneLmTrainJob.xObXEwRpvmzd` selected epoch 10 (ppl 5.0906, 3.31 M
params), the checkpoint `PriorGapAnalysisJob.5wNIQs2lpC5P` scored; the spec's "V2" class name
above is wrong (`NeuralPhoneLmTrainJobV2` is the 25.5 M instance (c)). Code review r1
PASS_WITH_CONCERNS (`reports/review_soft_arm_r1_2026-09-21.md`) confirmed scorer, reward
conventions (unigram = the bed's own prior file, SIL dropped, > 512 masked, BOS/no-EOS as
prior_gap), straight-through gradient path, sign, default-off plumbing, arms, pairings and labels.
Adopted reading rules: (i) the null arm permutes BOTH halves of the reward (scorer and unigram), so
r_null(y) = r(sigma^-1 y) keeps the reward's statistics, and softshuf_20's `soft_reward_mean` is a
different statistic from the real arms' and is never read in the same column; (ii) the
"−0.67 → +0.95" band is the dev-other greedy statistic of Step 0b while the monitor is on the
tempered max-plus TRAIN string: direction only, the like-for-like read is the prior-gap rerun.
Round 2 (sf arm) built in speech-llm ff4f005 (`reports/impl_sf_arm_r2_2026-09-21.md`,
DONE_WITH_CONCERNS: `sf_scorer.py`, G = 8 FFBS draws, A1 reward via the soft-arm primitives,
centred advantages, A4 Fisher path score on un-expanded tables, A5 per frame;
`SfLamProbeJob.OHrcN9pEuXni`, shim `config/sae_4a_sf_lam_probe.py` (the older
`sae_4a_sf_probe.py` is the finished falsifier graph); default-off `sf_*` block, soft / sf mutually
exclusive; 34 tests; census unchanged, `SoftLamProbeJob.wAJQ26T7iZzX` unmoved). Monitor
`sf_reward_mean` is the decode string's value, `sf_reward_mean_sample` the draws' mean. Code review
r2 DONE_WITH_CONCERNS (`reports/review_sf_arm_r2_2026-09-21.md`) confirmed the sampler (banked
`sample_blankfree_paths` at the step's own tau, seeded per (epoch, step)), the path score (banked
`path_score`, both gathers on un-expanded tables), detached reward, per-utterance centring, the
same `retained` divisor as l_tau, sign, unedited reward primitives, structurally unmovable hashes
and pairing orientation. Adopted reading rule: `sf_reward_mean` scores the greedy-argmax collapse
while `soft_reward_mean` scores the max-plus string, so the two monitor columns are never read side
by side for the sf_20 − soft_20 contrast (the gate reads PER). **Not checked by either review:**
census counts re-derived, tests executed (GPU memory / time of the sampler pass was read by the
probe). Constants filled in speech-llm ecf846c (`reports/impl_soft_pack_fill_2026-09-21.md`;
LAM_03 = 0.979918 recorded, no arm): pack hash `PackedBlankfreeTrainJob.MXKoywbfon8O`, 196 jobs,
slots sorted alphabetically sf_20 / soft_20 / soft_20_s1 / softshuf_20 (GPU assignment only),
census unchanged, both probes unmoved.

### Soft-arm lam probe (`SoftLamProbeJob.wAJQ26T7iZzX`, FINISHED 2026-09-21, Slurm 1925609, 77 s; `output/summary.txt`, `probe.json`; `reports/exec_soft_probe_rerun_2026-09-21.md`)

ctrl_20 ep4 seed 0 (`PackedBlankfreeTrainJob.5EIGJJ1MkcO9/output/ctrl_20/models/epoch.004.pt`), 3 batches at 88,000 padded frames / max_seqs 128, A5 per-frame convention.

| quantity | value |
|---|---|
| ratio |grad soft| / |grad l_tau| per batch | 0.298, 0.306, 0.335 (median 0.306148) |
| **lam_soft for 0.1× (LAM_01, the pack constant)** | **0.326639** |
| lam_soft for 0.3× (ceiling, no arm) | 0.979918 |
| artefact gap scorer(soft) − scorer(hard), nats per token | 0.0138 (ceiling 0.3) |
| soft_reward_mean per token on the train max-plus string | −0.236 (direction read only; dev greedy band −0.67 → +0.95) |
| strings scored per batch / tokens per string | 121.3 / 134.6 |

Read: the term's raw gradient is about 0.3 of l_tau's, so the 0.1× arm runs at lam 0.327; the straight-through artefact is 20× under its ceiling at the start. The first (OOM) attempt is recorded in State; the rerun is the same hash after a value-identical fix.

First (OOM) attempt, moved from State: Slurm 1925281 died in the term's backward asking for
224.61 GiB (`reports/exec_soft_probe_launch_2026-09-21.md`); cause a gather over an expanded view
of the per-segment table, whose backward allocates the expanded [B, U_max, K·d_cap·(S+1)] float64
shape (`reports/debug_soft_probe_oom_2026-09-21.md`); fixed value-identically in speech-llm 0474d7f
(23 tests; backward peak 1.14× the table on the GH200 against 65× before; probe hash unchanged).

### Sampled-arm lam probe (`SfLamProbeJob.OHrcN9pEuXni`, FINISHED 2026-09-21, Slurm 1926336, 2 min 40 s; `output/summary.txt`; `reports/exec_sf_probe_launch_2026-09-21.md`)

Same protocol as the soft probe: ctrl_20 ep4, tau = 2, 3 batches at 88,000 padded frames / max_seqs 128, G = 8 FFBS draws, A5 per-frame convention, two real backwards per batch.

| quantity | value |
|---|---|
| ratio |grad sf| / |grad l_tau| per batch | 2.368, 2.186, 2.203 (median 2.20294) |
| **lam_sf for 0.1× (LAM_SF_01, the pack constant)** | **0.0453938** |
| lam_sf for 0.3× (recorded, no arm) | 0.136181 |
| peak GPU memory per batch | 19.6 / 23.6 / 23.1 GiB |
| probe time per batch (two backwards) | 19.3 / 14.9 / 14.2 s |
| sf_reward_mean (greedy string, per token) | −0.535 |
| sf_reward_std_within (nats per utterance, median) | 9.68 (dead band 1.0) |
| sf_unique_strings (of 8) | 8.0 |
| sf_masked_long | 0 |
| sf_adv_absmean | 8.35 |

Read: the score-function term's raw gradient is 7× that of the soft term at the same target
(2.20 against 0.31 of l_tau), as expected for a sampled estimator; the constant is set so the pack
arms run at the same 0.1× target. The sampler explores at ep4: eight distinct strings per
utterance and a within-group reward spread of 9.7 nats, far above the dead band, so the
"sampler does not explore" UNINFORMATIVE clause is not triggered at the start. Memory 23.6 GiB
against the 96 GiB device; step time in the pack is read at ep1 against the 2.00 × 601 s clause.

### Soft pack, ep1 read (`PackedBlankfreeTrainJob.MXKoywbfon8O`, RUNNING since 2026-09-21 15:06 UTC; `reports/extract_soft_pack_ep1_2026-09-21.md`; moved from State)

All four arms past sub-epoch 4 by 16:08 UTC. Wall seconds per sub-epoch sf_20 717 / soft_20 849 /
soft_20_s1 852 / softshuf_20 849 (1.19–1.42 × 601 s, all under the 2.00 bar, so sf_20 stays in the
pack). ep1 dev loss sf 1.517 / soft 1.430 / soft_s1 1.435 / softshuf 1.520; ep4 dev reward_mean
sf −0.510 / soft −0.451 / soft_s1 −0.408 / softshuf −1.799 (the shuffled null prices its own
strings far below the structured arms, as a destroyed-structure control should); retained rate
10.2–10.9 Hz, tau 2.0, lam soft 0.327 / sf 0.0454; peak RSS 52 GB.

### Soft pack, interim read at kept epoch 10 of 20 (`PackedBlankfreeTrainJob.MXKoywbfon8O`, RUNNING, sub-epoch 12–14 at 2026-09-21 18:25 UTC; `reports/extract_soft_pack_gaps_2026-09-21.md`, every file traced to this pack by its checkpoint chain)

Not a gate read (G4a.7 reads the final sub-epoch); recorded because it answers whether the scorer term moves the decoded strings toward the text.

| arm | trigram gap gold − decoded, nats per token (dev-other, ep10) | 6-gram gap | lexicon-escape gap | paired PER delta vs its control, dev-other, ep1 / ep4 / ep10 (negative = arm better) | dev reward_mean, latest |
|---|---|---|---|---|---|
| sf_20 | 1.147 | 1.491 | 2.237 | +0.0075 / +0.0351 / +0.0306 | −0.515 |
| soft_20 | 1.295 | 1.600 | 2.313 | +0.0206 / −0.0004 / +0.0101 | −0.590 |
| soft_20_s1 | 1.055 | 1.537 | 2.294 | +0.0209 / +0.0099 / +0.0099 | −0.498 |
| softshuf_20 (shuffled-pronunciation null) | 1.066 | 1.541 | 2.269 | +0.0710 / +0.0007 / +0.0121 | −2.025 |

Reading: at ep10 the gap between decoded and gold is of the Step 0 magnitude (trigram 1.39, lexicon 2.32 at ctrl_50 ep10; the matched ctrl_20 prior-gap rows are not registered, so this is against the pre-training number), and no arm is better than its control on paired PER at any kept epoch; the scorer separates the real arms from the shuffled null by 1.5 nats of reward, so it sees structure that the arms have not converted into a smaller gap. Decision table: no row fires for any arm. Final read at ep20 under G4a.7.

**n-gram JSD of the decoded private code against the text (user request 2026-09-21; `NgramModeSeekingJob` w4NlQzcoXIfd / E4gg11FZ1itE / EKkpTHPdLKGD per epoch, pooled KmbcDX5k6aaS; `config/sae_4a_soft_ngram.py`, `output/sae/4a/soft/ep{1,4,10}/summary_ngram.md`; conventions of the 2026-09-19 priorshuf read: unbiased window `orN768ARKwlt`, trigram `RtzbESkOedsT`, SIL stripped, rows subsampled to the epoch's smallest row, 1000 utterance-block resamples; descriptive, no gate).** Dev-other greedy decodes; JSD in bits against the text n-gram distribution, mean SIL-free trigram log-prob per phone in nats.

| row | ep4: log P3 / JSD1 / JSD4 | ep10: log P3 / JSD1 / JSD2 / JSD3 / JSD4 |
|---|---|---|
| sf_20 | −3.628 / 0.078 / 0.798 | −3.708 / 0.047 / 0.298 / 0.555 / 0.789 |
| soft_20 | −3.679 / 0.046 / 0.746 | −3.814 / 0.032 / 0.272 / 0.531 / 0.772 |
| soft_20_s1 | −3.470 / 0.028 / 0.741 | −3.639 / 0.027 / 0.249 / 0.498 / 0.748 |
| softshuf_20 (null) | −3.555 / 0.054 / 0.723 | −3.641 / 0.037 / 0.252 / 0.498 / 0.746 |
| ctrl_20 | −3.459 / 0.068 / 0.771 | −3.524 / 0.025 / 0.250 / 0.500 / 0.739 |
| ctrl_20_s1 | −3.478 / 0.041 / 0.739 | −3.721 / 0.032 / 0.253 / 0.501 / 0.748 |
| gold | −2.534 / 0.002 / 0.229 | −2.534 / 0.001 / 0.016 / 0.068 / 0.230 |

Reading: at ep10 no scorer-trained arm is closer to the text than its control at any order; soft_20 and sf_20 are slightly farther (JSD4 +0.033 and +0.050 against ctrl_20, CIs exclude zero) and their decoded strings are LESS likely under the text trigram (−3.81 / −3.71 against −3.52 nats per phone), while the shuffled-pronunciation null sits at the control's level. At ep4 soft_20 had been marginally closer than ctrl_20 (JSD4 −0.025); the direction reverses by ep10. All rows stay near 0.75 bits at n = 4 against gold's 0.23, the same sequence-structure gap the attribution phase measured on the earlier bed (0.71). So the transformer scorer, which prices the real arms 1.5 nats above the null, does not move the decoded n-gram distribution toward the text; the scorer's preference and the text's n-gram structure are being satisfied on different axes. Final read at ep20 (a second registration when the ep20 decodes exist).

**Examples read, ep10 dev-other decodes (descriptive audit from a fresh context, `reports/audit_soft_pack_examples_2026-09-21.md`; its own Levenshtein reproduces every arm's per.txt S/D/I exactly).** (1) Every arm, controls included, decodes fluent English-shaped phone nonsense: 94-96 % of reference length, right inventory, word-like 4-grams of which only 36-47 % occur in gold at all, zero adjacent repeats, and an utterance-initial HH artefact (ctrl_20 91 %, ctrl_20_s1 81 %, sf_20 83 %, soft_20 31 %, soft_20_s1 10 %). (2) The code is not a consistent relabelling: normalised mutual information between aligned gold and hypothesis phones 0.066-0.093 nats of H(gold) = 3.34, modal share per gold phone 0.19-0.23 against a 0.08-0.12 chance baseline; ctrl_20 is the most consistent code (0.093), soft_20 0.084, sf_20 0.066. (3) No error-pattern statistic (length ratio, 4-gram diversity, 4-gram-in-gold share, unigram JS, NMI, sub/del/ins mix, paired per-utterance delta) separates the soft arms from the controls beyond the seed spread; the two controls sit at both ends of the six-arm range on several of them (4-gram diversity 2749 / 3148 per 10k, N production +2.4 / -1.5 pp). The shuffled null sits between soft_20 and soft_20_s1 on essentially every statistic. (4) sf_20 alone is outside the control band and worse: PER 0.899, substitutions 74 % of N, worse than ctrl_20 on 65.5 % of utterances, R over-produced by 5 pp, and repeated R-AH/AO-R-AH-S 4-grams at 0.3-0.4 % each. Reading: the scorer's phone identities are not being used by the policy; the score-function estimator degrades the code, the straight-through one changes nothing the seed does not change more. Descriptive, no gate; seed spread is n = 2.
