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
Step 0b first pass finished (`NeuralPhoneLmTrainJob.Iv6P6YVPNWmB`, `PriorGapAnalysisJob.Gct95xZHe0zt`,
speech-llm 739d9ed, e971603, `reports/impl_neural_phone_lm_2026-09-20.md`,
`reports/review_neural_phone_lm_2026-09-20.md`): partial proxy, schedule-bound (Results, Step 0b).
The pre-registered rerun (speech-llm bf49fb7, `reports/impl_neural_phone_lm_rerun_2026-09-20.md`)
is FINISHED and read (Results, "Rerun result"): (a) 3.3 M gap 1.861, (b) 10.9 M gap 1.849, both
partial proxies, both ended by the no-improvement abort at epoch 11, so the 1 M-line window is the
ceiling; the manager (3594890) exited cleanly, its graph is complete. The survey for the
pre-registered fallback (GPU trie lexicon DP) runs in parallel with instance (c)
(`reports/survey_lexicon_scorer_2026-09-20.md`); no fallback code before (c) reads.
Training-arm design and G4a.7 are written (Design "Training arm", Gate); the design review
(`reports/design_review_prior_arm_2026-09-20.md`) returned STOP as written, approvable with A1–A4,
all applied (reward = strong minus unigram, r(gold) > max_g fraction replaces the dead band,
straight-through soft arm, path-level Fisher estimator, length / SIL / resume policies, pre-launch
falsifier). The scorer slot is filled by Step 0b's rule. Code survey banked
(`reports/survey_sampled_prior_term_2026-09-20.md`). The wav2vec-U 2.0 preprocessing question (does
2.0 apply rVAD at all; our bed has rVAD AND sil_prob 0.5) is with the literature agent
(`reports/lit_w2vu2_preprocessing_2026-09-20.md`); its answer goes to `SAE_ref.md`.
Open user question (SIL vs rVAD): prior text has 13.8 % SIL tokens (sil_prob 0.5, surround;
the local wav2vec-U pipeline uses 0.25 with rVAD); the gold SIL share on retained frames is being
computed (`reports/extract_sil_rate_2026-09-20.md`); a sil_prob arm is a bed change, own arm.
User directive 2026-09-20 (`SAE.md`): reach the Step 0b gate and launch the strong-scorer arm,
autonomously. N = 20 sub-epochs for the arm (`SAE_4A_budget.md` "Sub-epoch count"; replaces the
N = 50 cost line above: 3.3 h per arm, no resume path needed, ctrl_20 in the same pack). In
flight: instance (c) of the phone LM built (speech-llm d1c14cf: 25.5 M params, 10.1 M-line sample
seed 1, the original 10,000 held lines removed by content and used as the fit's held-out set;
`NeuralPhoneLmTrainJobV2.vkNGAOeLgNsy`, `PriorGapAnalysisJob.OO0iAEVLKgOO`,
`config/sae_4a_phone_lm.py`, `reports/impl_phone_lm_v2_2026-09-20.md`), reviewed
(`reports/review_phone_lm_v2_2026-09-20.md`, disclosures in Design) and running under its own
manager: pid 4004154, `log/sae_4a_phone_lm.manager.20260920T214940Z.log`, watcher
`bash ~/.claude/skills/sis/sis_watch.sh 4004154 config/sae_4a_phone_lm.py 600`
(`reports/exec_phone_lm_v2_launch_2026-09-20.md`; text jobs Slurm 1920042/1920043 first, then the
GPU fit, ~2 h estimated). Falsifier probe (i) built (speech-llm d22d81d:
`NeighbourhoodProbeJob.ZhYOeCltF4wW` ep4 / `.6JZvwE7T1UmA` ep10, word-BIGRAM row
`PriorGapAnalysisJob.m6lUxAO65f6A`; `reports/impl_prior_probe_2026-09-20.md`) is FINISHED and read
(Results, "Pre-launch falsifier (i)"): A1 removes the sign trap, but gold beats the whole
single-token neighbourhood in > 99 % of utterances at ep4 and ep10 and the neural reward's local
deltas track the trigram (+0.65); the funding rule is read on (ii). Its manager (3849944) exited
cleanly (note the venv is `/e/project1/spell/wu24/env/sis_env`).
The word-UNIGRAM row is deferred: KenLM has no order 1, it needs an edit inside prior_gap.py, which
stays frozen while the rerun's analysis jobs are pending; it is not on the launch path. Blank-free
FFBS sampler + probe (ii) built (speech-llm 2ae6cc5, new modules only, brute-force checked against
full enumeration, reproduces the banked greedy decode; `reports/impl_blankfree_sampler_2026-09-20.md`)
and running: `SampledRewardProbeJob.d1NoUQJ2EXN5` (ep1, tau 8) / `.Y81PrZ6fWWKu` (ep4, tau 5.04) /
`.HVHIlaUkIkVi` (ep10, tau 2), 300 dev-other utterances, G = 8; FINISHED and read (Results,
"Pre-launch falsifier (ii)"): r(gold) > max_g in 100 / 100 / 97 % at ep1 / ep4 / ep10, above 95 %
at every checkpoint, so under the pre-registered rule the sf arm is NOT funded; the soft arm is
not funded either while the scorer is a partial proxy. The manager (4178489) exited cleanly. The sf
term's normalisation was pinned per retained frame (A5) before the read. Audit of the reading
dispatched (`reports/audit_sf_probe_2026-09-21.md`); literature check for the lexicon-in-the-
objective design dispatched (`reports/lit_lexicon_in_objective_2026-09-21.md`).
NEXT: read the audit; read instance (c) when its watcher fires (bar 2.01 + strict tracking): if
(c) is adequate, reconsider the soft arm with (c) as scorer (design amendment, review); otherwise
Step 0b closes on "no phone LM reaches the bar from this text" and the next arm is the
lattice-internal lexicon (GPU trie DP; survey banked, literature pending), which needs its own
pre-registered design, a design review before its first job, and probably its own phase file.
No training arm of this phase runs before that.

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
