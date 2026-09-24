# SAE_i6 reference: the blank-free cycle bed and its cold rounds (attribution, budget, InfoMax, waveform cut, cdrev)

Scope: the blank-free era of phase 4A. `SAE_i6_ref_objective.md` holds the reverse-KL derivation (and why
tau ends at 2). The ported package's `ctrl_20` preset is the bed of section 1.

**Headline.** No cold blank-free arm has left the content-free band: all 25 read cold arms end
at dev-other greedy PER 0.86-0.94. Budget (up to 67 sub-epochs), schedule, prior window, coverage
terms, back-translation, entropy penalty, invariance and the waveform cut all failed to move it. The
binding problem is a confident, input-dependent private code shared by the recognizer and the
reverse model, not a diffuse stationary point.

## 1. The blank-free bed as run

| item | value |
|---|---|
| training speech | LibriSpeech train-clean-100, 28,539 utterances, unlabelled. The training partition is 28,254; the rest are CV segments. |
| evaluation | dev-other (2864 utterances, 33 speakers, 177,275 reference phones) and dev-clean (2703). MFA phone references, used for scoring only. |
| features | wav2vec 2.0 large-lv60 `hidden_states[15]`, 1024-dim, 50 Hz, fp16. Extracted from the full (untrimmed) waveform; frozen. |
| VAD | rVADfast (distribution 0.0.5; the module declares 0.0.3), threshold 0.4, 25 ms window / 10 ms shift. The 10 ms labels are OR-aggregated pairwise to 50 Hz (a tie counts as speech), with truncate / tail-pad-as-silence reconciliation. The mask is applied to features and units after SSL extraction ("feature mask"). No gold enters it. |
| retained frames | train 15,427,853 of 18,088,388; dev-clean 831,372 of 968,057; dev-other 781,130 of 919,980. No utterance is dropped; the original frame index and length are kept. |
| reverse units z | "enc50 K500": PCA 96 + k-means K = 500 on the same L15 features; frozen. |
| speaker vector eta | 16-dim fixed projection of the utterance-mean features; frozen. |
| text | LibriSpeech LM corpus (39,630,169 lines), phonemised by lexicon. SIL is inserted at word boundaries with probability 0.5, plus at sentence edges. |
| prior P_psi | Interpolated Witten-Bell trigram over 40 symbols, BOS context, no EOS factor. Fit on a seeded (seed 0) uniform sample of 1,010,000 lines, with every 101st line held out; held-out perplexity 9.56. This is the "priorshuf" prior. |
| rate target | rho = 9.6619373279 phones per original-audio second (phones per word in the text x a disclosed 2.7 words/s). |

**Recognizer theta** (the wav2vec-U 2.0 generator shape). Input 1024; valid-frame batch norm
initialised at scale 30; dropout 0.1; a residual 1024-to-1024 linear; one bias-free convolution
(kernel 9, stride 3, padding 4, dilation 1). It has 40 outputs (39 ARPAbet phones plus SIL) and **no
CTC blank**. S retained frames give T = ceil(S/3) output frames. Initialisation is a flat
recognizer: logits exactly zero, so the per-frame posterior is uniform (no checkpoint transplant).
The output string is y = B(a): adjacent identical outputs, SIL included, collapse to one token. The
support therefore excludes adjacent identical phones; this is a support restriction, not a
normalisation defect. **Reverse model phi** is a segmental HSMM, p(z | y, eta). Each token occupies
one segment of duration d in [2, D_k], with D_k = 25 for phones and 50 for SIL, on the retained 50
Hz clock. The emission is a categorical over 500 units, conditioned on type, 2 duration buckets, 3
position buckets and eta. This is effectively a 240 x 500 table, produced by a 0.37 M-parameter MLP.
It starts from an independent random initialisation and has no cross-token context. **Lattice**
(exact, nothing sampled). The state F_t(s, h) combines the recognizer step t, the retained frame s
and the trigram history h, starting from F_0(0, BOS) = 0. A prediction equal to last(h) consumes one
step and scores log q / tau. Any other prediction, or any prediction from BOS (SIL included), starts
exactly one segment: it advances the trigram, consumes a legal d, and adds [log q + beta log P3(k |
h) + G(k, s, d)] / tau. The band is |s - 3t| <= 25, and the final gather is at s = S over all
stride-3 padding residues, with no EOS factor. beta (prior_weight) = 1. The DP runs in float64 with
matmul reduction and checkpoint stride 32. The sum covers only this band and duration support; it is
not the unrestricted q pushforward.

**Loss.** L = l_tau + lam_rate L_rate + lam_agg L_agg, averaged over the batch.

| term | definition and constants |
|---|---|
| l_tau | -log Z / S, normalised by the retained S (never the stride-3 T); lam_tau = 1 |
| L_rate | ((E[N] / T_audio - rho) / rho)^2. E[N] is the expected number of non-SIL tokens under the tempered posterior; T_audio is the **original** duration. lam_rate = 3; finite-difference tilt eps 0.25 |
| L_agg | KL(text \|\| expected counts) over run unigrams and off-diagonal run bigrams. Counts are q_0(k) + sum_{t>0} q_t(k)(1 - q_{t-1}(k)) and sum_{t>0} q_{t-1}(j) q_t(k); the text bigram diagonal is zeroed and renormalised. EMA decay 0.99; lam_agg = 0.1 |
| tau | Set per sub-epoch. The original 4-sub-epoch bed used (8, 5.04, 3.17, 2). At N sub-epochs: geometric 8 -> 2 over ceil(0.2 N), then held at 2 |

**Optimisation.** Adam (0.5, 0.98), eps 1e-6, weight decay 0, global clip 5. The measured gradient
norm is about 0.1, so the clip is inert. theta LR peaks at 1e-4; phi uses 3e-3 (multiplier 30,
traced to a batch-8 reference). Batches: 88,000 padded frames, max_seqs 128, in laplace-sorted order
seeded by the full-epoch index plus `random_seed_offset`. `partition_epoch` 4: a sub-epoch is a
quarter of train-clean-100 (57 updates), so 4 sub-epochs make one pass (228 updates). LR schedule:
warmup from 0.1x, hold at 1x until floor(0.6 N), then linear decay to 0.1x at N. At N = 20 the
warmup is 2 sub-epochs, because the 5 % rule gives 1, which the builder rejects. **ctrl_20** runs N
= 20 (1140 updates). tau is 8, 5.04, 3.17, then 2 from sub-epoch 4. The LR warms up to sub-epoch 2,
holds to 12 and decays to 20. Checkpoints 1, 4, 10 and 20 are kept. It takes about 601 s per
sub-epoch on one GH200 96 GB GPU (about 3.3 h per arm), so three arms fit one 11.5 h node
allocation. Preset `ctrl_20_x60` adds 40 held sub-epochs (tau 2, LR 1e-4). The seed replicate
ctrl_20_s1 moves flat_seed, random_seed and random_seed_offset (to 1000; offset 1 only shifts the
shuffles by one epoch). It pairs at -0.001 [-0.004, +0.001] at sub-epoch 20, so the seed spread is
about 0.01 PER. GPU training is not bit-reproducible.

**Evaluation** (read-side only):

| read | definition |
|---|---|
| greedy PER | per-frame argmax; collapse repeats **including SIL**; remove SIL **without a second collapse** (A SIL A stays A A); label 0 is not a blank; scored against SIL-stripped references with S/D/I and the greedy rate `phone_rate_original_hz` |
| paired delta | corpus-ratio PER delta, 95 % speaker-clustered bootstrap (2000 resamples); near zero it can disagree in sign with the macro per-utterance delta |
| derangement gap | reverse log-likelihood per frame given the utterance's own decode minus given a speaker-matched donor's decode, 500 utterances, no CI; positive in every content-free arm after sub-epoch 1, so it does not discriminate |
| chance null | strings matched to the arm's own per-utterance length and unigram distribution (5 draws, seed 0); E/N = null PER - arm PER; positive controls S3b-OR frozen-oracle +0.409, GAN +0.696; band exit needs PER more than 0.05 below the own null |
| checkpoint rule | final sub-epoch or a label-free-selected checkpoint, **never best-PER**. The wav2vec-U 2.0 selector (4-gram phone-LM perplexity over squared vocabulary-seen fraction, SIL stripped) was never registered on a blank-free pack, so every gate here is a final read |
| content-free band | PER 0.83-0.91; length-faithful random strings score 0.84-0.87 |

**Standing constraints.** Label quarantine: unpaired speech and text, frozen SSL features only. Gold
is used only for evaluation and disclosed diagnostics. No GAN component in new cycle arms. Every
n-gram uses the seeded uniform sample. N = 20 for new arms, each pack with its own ctrl_20. Never
compare with ctrl_50 at sub-epoch 20, which is in a different LR phase. The masked-feature bed is
the default (orchestrator ruling; the user may overturn it). The supervised 10 h branch feeds
nothing to cold arms, and no 100 h adaptation follows it.

**Defects and pitfalls:**

| defect | effect | fix |
|---|---|---|
| Alphabetical prior window: the first prior used the first 1,010,000 lines of the sorted corpus (73.5 % AH-initial, 20.7 % AE, 5.0 % AA; P(AH \| BOS SIL) about 0.74) | AH-first collapse (58.1 % at ep4); confounds absolute statistics and GAN comparisons (the GAN's text is the full corpus). Not the cause of the PER collapse (§5 priorshuf) | Seeded uniform sample. norev, k64, agg1 and agg10 stay on the biased window and are compared within bed only |
| KL-form L_agg gradient damped 0.01x by the EMA blend | lam_agg 0.1 acts as about 1e-3; the bed is prior-driven | Ratio form, with the EMA only in the denominator (§5 step 5) |
| Coverage KL read from the training EMA | Biased low (0.45 vs 0.83), because the EMA blends model states | Read on the checkpoint: a full pass or at least 100 batches |
| `PhoneNgramPrior.per_token_log_probs` double-counts length-1 sequences | Wrong per-token scores | Worked around in the n-gram reader |
| float32 lattice | Violates posterior conservation at the flat initialisation | float64 DP |

(src: SAE_4A_blankfree.md § Registered first model; SAE_4A_budget.md § Bed and constants, § Design,
§ Sub-epoch count for future arms; SAE_4A_prepro.md § Design, § G4a.8 read of the pack;
SAE_4A_attrib.md § Prior-window defect, § Step 6 item 1 result, § Hyperparameter review; SAE_ref.md
§ Training-prior text window is alphabetically biased, § Cold-initial numerical validity;
SAE_4A_objective.md § 1, § 4)

## 2. First cold blank-free model (gate G4a.3)

**Question.** Can the user's bundled simpler-alignment model take off from cold? It bundles VAD,
stride-3, no blank and an exact trigram marginal, so no effect can be attributed to one of them.
**Design.** The §1 bed on the biased prior window, 4 sub-epochs (228 updates), checkpoints 1 and 4.
Release required exhaustive-enumeration agreement of logZ and gradients (tau 1, 2, 8) and an
all-finite actual-data profile (maximum cold update 18.24 s; 456 x 18.24 s x 1.25 = 2.889 h, under
the 8 h cap). **Gate G4a.3 (pre-registered).** Ep4 dev-other PER < 0.50 AND a positive own-phi gap.
**Outcome: FAIL** (43m59s on one GPU).

| dev-other | ep1 | ep4 |
|---|---:|---:|
| PER | 0.834602 | 0.864877 |
| distinct / empty strings | 2864 / 0 | 2864 / 0 |
| non-SIL phones emitted (per original s) | 120187 (6.53204) | 169036 (9.18694) |
| types used; modal phone | 39; AH 14.08 % | 38; N 12.94 % |

dev-clean PER is 0.826548 at ep1 and 0.848407 at ep4. The ep4 gap is +2.4808968 per frame on
dev-other (own -4.4247514, donor -6.9056483) and +2.6842944 on dev-clean. 58.1 % of ep4 outputs
start with AH. Example: REF `P R AH D UW S IH M`, HYP `T AH L IH NG HH ER D Z`. Against the
sampled-path M512 arms (descriptive): they also had 2864 distinct non-empty outputs; inventory P3 36
/ P6 35 / new 38; unigram entropy 4.3638 / 4.2363 / 4.389 bits; modal first phone P3 AH 90.85 %, P6
AA 77.65 %, new AH 58.1 %. **Conclusion.** Output diversity coexists with chance-level phone error.
The positive gap is not recognition.

(src: SAE_4A_blankfree.md § Actual-data cost profile, § First cold four-subepoch result)

## 3. Supervised 10 h separate initialisation (disclosed reference, not cold progress)

**Design.** theta and phi are fit independently on the labelled 10 h (2849 utterances, 2821 / 28
held). theta: exact blank-free log Q(y | x) at tau 1, with no LM, reverse factor or band. Targets
are SIL-free phones with adjacent runs collapsed. LR 5e-5, 24 passes. phi: exact reverse marginal on
the uncollapsed phones, with SIL at both ends. LR 3e-3, 8 passes, batch 8. **Gate.** A support
census, with no silent filtering. theta needs U_collapsed <= ceil(S/3); phi needs 2U <= S <= sum
D(y_i). **Outcome.** Theta fails on exactly one item, `8629-261139-0016` (108 CNN outputs vs 131
collapsed phones). All 2849 phi targets pass. So **neither fit ran**. Open user decision: exclude
the item from theta training only, or supervise on MFA frames.

(src: SAE_4A_blankfree.md § Supervised 10 h separate initialization only, § Actual-data support
result)

## 4. Side task: diversity within sampled path groups (historical M512 arms)

The sampled-path M512 arms P3 (trigram) and P6 (six-gram) draw 512 paths per utterance (384
target-temperature, 128 hot). **Question:** is their importance weight concentrated? **Design:**
replay 4 real batches (411 groups) at cold (tau 8) and at the ep8 checkpoint (tau 2), with no
training. ESS < 2 and a maximum weight > 0.9 are descriptive scales. **Outcome:** every group has
512 unique paths. P3 mean path ESS is 395.00 cold and 384.14 final, with no group under either
scale. P6 mean path ESS is 3.74 cold and 3.50 final; ESS < 2 in 106 then 165 of 411 groups, maximum
weight > 0.9 in 35 then 76. **Conclusion:** P6 concentrates weight from the cold state. This does
not explain the greedy pattern, and the exact-sum lattice has no such estimator.

(src: SAE_4A_blankfree.md § Sampled-group result)

## 5. Attribution: why the GAN takes off and the cycle does not

**Question.** The blank-free recognizer equals the wav2vec-U 2.0 generator. The local GAN
reproduction (150k updates, 37552 s) reaches dev-other PER 0.214 (seeds 0-3: 0.214 / 0.205 / 0.215 /
0.168); every cold cycle arm stays at 0.86-0.90. On the earlier bed, a gold-derived *frozen* phi
(S3b-OR) reached 0.454 at ep4 and 0.368 at ep8 from a flat theta. Two hypotheses. H1: the fixed
trigram is mode-seeking where a discriminator is mode-covering. H2: the jointly trained phi degrades
the recognizer. **Design.** A 2x2 of {GAN term, trigram lattice} x {with, without reverse term},
plus side arms. Each cold arm runs 4 sub-epochs with a single delta. Decision rule: steps 1, 2 and 4
as predicted mean a mode-covering term inside the cycle; step 4 collapsing at every weight means
freeze phi; step 3 taking off means an unsupervised route. **It fired on no branch.**

**Step 1: training-free mode-seeking check (audited).** SIL-free n-gram JSD of decodes vs text and
mean trigram log-prob per phone, count-matched, with a 1000-resample utterance-block bootstrap. The
gate (amended to paired CIs) needs all three: (i) ep4 log P3 >= GAN - 0.10; (ii) JSD4 at least 0.05
above gold and above the GAN; (iii) GAN within 0.10 of gold. Reads (log P3 / JSD4): ep1 -6.2027 /
0.9359; ep4 -3.3424 / 0.7134; GAN s0 (update 148000) -3.0087 / 0.3021; gold -2.8898 / 0.2749. Cell
(i) fails at -0.334 [-0.347, -0.290]; (ii) and (iii) pass. **Conclusion:** the cold output has
*lower* trigram likelihood than the GAN or gold. It is not a high-likelihood mode, so H1 in its
mode-seeking form is not supported.

**Step 2: no reverse term, and more aggregate weight** (biased-prior bed, audited). norev: zero
emission, duration frozen at initialisation. This was amended; zeroing both leaves an unnormalised
segment count. agg1 / agg10: lam_agg 1 and 10. The rule: norev worse by more than 0.05 means the
reverse term carries content.

| arm | ep4 PER | paired vs reference 0.8649 | ep4 gap | first phone |
|---|---|---|---|---|
| norev | 0.9357 | +0.071 [+0.056, +0.088] | -0.026 | AH 0.950 |
| agg1 | 0.9202 | +0.055 [+0.051, +0.060] | 3.44 | AH 0.919 |
| agg10 | 0.8986 | +0.034 [+0.027, +0.040] | 3.39 | AE 0.778 |

**Conclusion:** the cold reverse term carries something the trigram lacks, so H2 in its "degrades"
form fails. Stronger low-order matching does not help. The initial phone follows the prior window.
**Step 3: MFCC K = 64 reverse observations.** Ep4 PER 0.9147, paired +0.050 [+0.039, +0.061], gap
+0.75. The route gate (PER < 0.50 with a positive gap) fails. An informative reverse target does not
make theta take off. Every arm worsens from ep1 to ep4 while its losses fall (step 6: the null's own
rise).

**Prior-window test (priorshuf).** The single delta is the uniform-sample prior. Rule: the window is
the cause if ep4 AH-first < 20 % and PER improves by more than 0.05. It is not the cause if PER
stays within ±0.03 and AH-first stays above 40 %. Outcome: ep1 0.8206, ep4 0.8829, paired +0.018
[+0.015, +0.022]. First phone: AH 0.003, **HH 0.886**. **Conclusion:** the window is **not** the
cause. A single-phone sentence start belongs to the objective; the window only picks the phone. All
later arms use this bed. Not audited: JSD1 falls from 0.067 to 0.026, but JSD4 stays at 0.692 (gold
0.225). The decoded trigram log-prob is -3.611, while the training `prior_per_token` reaches -2.55.
The prior is satisfied inside the lattice posterior, not on the emitted string.

**Steps 5 and 5b: corpus-level coverage term** (Empirical-ODM direction). Basis (Liu, Chen, Deng
2017; Yeh et al. 2019): an LM score of the model's own output is mode-seeking; forward cross-entropy
from text n-grams to corpus-level expected output n-grams is coverage-seeking; the estimator matters
as much as the direction. Design: `prior_weight` 0, with the term skipped (never 0 x -inf). Order-3
L_agg by the exact (last, previous run label) recursion. Ratio loss -sum_w p_text(w) c_batch(w) /
c_EMA(w).detach(), floored per cell at max(1e-6, 0.01 p_text). lam_agg is priced by rule: the median
ratio of the removed prior term's gradient norm to the order-3 term's, at initialisation, gives
**0.009**. At initialisation the cycle term without the prior moves theta by 6e-3 of what the prior
does. Step 5b adds lam 0.1 and 1.0, and the prior-plus-coverage corner. Take-off gate, read at ep4
only: PER < 0.50, gap > 0 (dropped for odm3_norev), and greedy rate in [5.80, 14.49] /s. The term
"descends" if KL3 at ep4 is below 1.2 and monotone. **Outcome: FAIL everywhere** (5b audited). KL3 =
CE3 - 8.242 at ep4, read from the training EMA:

| arm (priorshuf bed) | ep4 PER | rate /s | paired vs priorshuf 0.8829 | KL3 |
|---|---|---|---|---|
| prior0 (prior_weight 0 only) | 0.8814 | 7.35 | -0.0015 [-0.0091, +0.0052] | - |
| odm3 (lam 0.009) | 0.9008 | 7.76 | +0.0178 [+0.0112, +0.0241] | 1.70 |
| odm3_norev | 0.8735 | 5.16 (fails) | -0.0095 [-0.0202, +0.0009] | 1.72 |
| odm3_lam0.1 | 0.8829 | 8.95 | -0.0001 [-0.0039, +0.0044] | 0.64 |
| odm3_lam1 | 0.9069 | 9.49 | +0.0240 [+0.0194, +0.0296] | 0.45 (0.830 at checkpoint) |
| odm3_prior | 0.8756 | 9.26 | -0.0073 [-0.0108, -0.0036] | 1.43 |

**Conclusion:** Removing the prior alone changes nothing. At 0.009 the coverage term is inert,
because it was priced against an inert term. At 0.1 and 1.0 it optimises, yet PER stays content-free
(the "satisfied content-free" outcome). The {prior} x {coverage} 2x2 spans 0.876-0.901, less than
any arm's own ep1-to-ep4 drift. The gap clause's "CI excludes 0" sub-clause was never evaluable.

**Step 6, item 1: estimator bias (audited).** Order-3 KL of the odm3_lam1 ep4 checkpoint: one batch
2.016 ± 0.341; 10 batches 1.042; 100 batches 0.852; full pass **0.830**; training EMA 0.450. That is
branch (C), unresolved (between 0.60 and 1.2), so the full-batch / SPDG arm is not funded. The EMA
blends model states; the step-5b phrase "within half a nat of its floor" is **overturned**; the
"optimises" verdict survives.

**Step 6, item 2: destroyed structure** (odm3_lam1_perm, one per-utterance frame permutation of all
training streams). The primary read was amended before the result to checkpoint-level KL3: <= 0.93
means structure-free, >= 2.2 means real structure. Outcome: 1.270 vs 0.830 (EMA 0.644 vs 0.450),
**unresolved** on both reads. Audited DONE_WITH_CONCERNS (every KL reproduced; eval permutation, tau, prior, batches and utterances verified identical). The permuted
arm is also worse at order 1 (+0.18) and order 2 (+0.27), which points to an optimisation handicap;
the net order-3 residual is about 0.26 nats. Most of the descent needs no temporal structure, and
what the term extracts is not phone content.

**Step 6, item 4: chance null (audited).** The positive controls pass. 20 of 24 cold decodes have an
absolute E/N below 0.02; the maximum is +0.025, under 4 % of the S3b-OR signal. **The ep1-to-ep4 PER
rise is the null's:** the reference's null rises 0.836 -> 0.889 while its excess hits grow. The
between-arm movements of steps 2, 3 and 5b are length and unigram skew, not content. E/N correlates
with decode length (Spearman 0.62), so the 0.02 bound is not scale-free. For future cold arms, read
E/N on the S3b-OR / GAN scale; the take-off gate stays at PER < 0.50, which the null cannot reach.

**Hyperparameter review.** No scalar retune is worth funding: no BN mismatch, no NaN, and both
parameter groups move. It scoped every cold FAIL to "no take-off within 228 updates" (hence §6) and
found the L_agg damping defect. Not funded: the GAN generator auxiliaries (smoothness, code penalty,
aux k-means), tau ending at 1, and the phi/theta LR ratio.

**Step 4: reverse term inside the GAN** (fairseq stack, audited). The generator loss gets + lam_rev
L_rev, the blank-free exact reverse marginal at beta 0 and tau 1 (amended from 2) on enc50 K500. phi
is trained jointly (Adam (0.5, 0.98), LR 3e-3) on sub-batches of b = 16, at 0.439 s/update against
0.258 for the plain GAN. Each arm runs 150k updates and is read at its own
`weighted_lm_ppl`-selected checkpoint. H1 predicts both lam 0.01 seeds within +0.03 of the control;
H2 predicts all six arms >= control + 0.10.

| arm | pick | dev-other PER | paired vs seed-matched control |
|---|---|---|---|
| control s0 / s1 (banked) | 148k / 77k | 0.214 / 0.205 | reference |
| w0_s0 (port no-op) | 129k | 0.240 | +0.026 [+0.023, +0.029] |
| lam0.01 s0 / s1 | 150k / 5k | 0.225 / 0.930 | +0.011 / +0.725 |
| lam0.1 s0 / s1 | 24k / 60k | 0.872 / 0.885 | +0.657 / +0.680 |
| lam1.0 s0 | 103k | 0.204 | -0.011 [-0.013, -0.009] |
| frozen_lam0.1 s0 (phi = blank-free ep4) | 109k | 0.217 | +0.003 [-0.000, +0.005] |

H1 and H2 both FAIL; the result depends on the weight. Three of the six joint-phi arms trained
normally to 18k-34k updates and then collapsed; their picks are pre-collapse checkpoints. The
collapses are not ordered by weight and, at n = 3, cannot be separated from GAN seed instability.
**Conclusion:** the reverse term does not degrade a GAN run that stays on its manifold. With step 2,
this places the cold failure in the alignment-sum-times-prior objective or in the budget. A
lam1.0_s0 diagnostic (not audited): own logZ -3.354 vs deranged -4.909 per frame. Its 22 % top-1
unit accuracy is phi decoding over a free path, not a read of theta's argmax. **Phase conclusion:**
no tested component moves cold PER off the null.

(src: SAE_4A_attrib.md § Question, § Design, § Design-review amendments, § Results (steps 1-6,
prior-window defect, step 4 diagnostic, step 6 synthesis))

## 6. Training budget and the N = 20 decision (gate G4a.4)

**Question.** Does a cold arm leave the band at 12.5x / 25x the budget (N = 50 / 100) with an LR and
tau schedule? **Design.** priorshuf bed, n = 1, four arms per 4-GPU node. ctrl_N: the schedule only.
bt_N: adds back-translation (lam_bt 0.1, ramped over the first 20 % of N). This is a separate theta
step on units synthesised by the live phi (collage from the retained-frame pool), with the
run-collapse full-sum loss and no gradient to phi. odmprior_N: order-3 coverage at 0.009 with the
prior (= odm3_prior). odmbt_N: both. nosched_ctrl_N / nosched_odmprior_N: constant LR, with the
4-sub-epoch anneal held at 2. The schedule is tau 8 -> 2 over ceil(0.2 N); LR warmup 5 %, hold to 60
%, decay to 0.1x. Kept checkpoints: 1, 4, 10, 25, 50. max_seqs stays at 128, because coverage is a
per-batch n-gram estimate. **Gate G4a.4 (pre-registered).** Final PER < 0.50, rate in [5.80, 14.49]
/s, gap > 0. The prediction was PER >= 0.80 throughout. **Outcome: FAIL for all six N = 50 arms.**
Rates were 10.7-10.9 /s and gaps 4.24-4.63.

| arm | ep1 | ep4 | ep10 | ep25 | ep50 | paired at ep50 |
|---|---|---|---|---|---|---|
| ctrl_50 | 0.855 | 0.869 | 0.897 | 0.888 | 0.897 | — |
| odmprior_50 | 0.793 | 0.849 | 0.908 | 0.895 | 0.905 | +0.008 |
| bt_50 | 0.855 | 0.903 | 0.896 | 0.890 | 0.893 | -0.003 |
| odmbt_50 | 0.793 | 0.865 | 0.896 | 0.893 | 0.893 | -0.004 |
| nosched_ctrl_50 | 0.822 | 0.893 | 0.893 | 0.886 | 0.895 | -0.001 |
| nosched_odmprior_50 | 0.815 | 0.881 | 0.883 | 0.877 | 0.889 | -0.016 (vs odmprior_50) |

The user killed the N = 100 arms at sub-epochs 65-67 ("not progressing"); no ep100 read exists.
Their ep20 PER was 0.875-0.910; past ep50 the objective drifted down 0.4-2.2 % (non-odm) or stayed
flat (odm), and the same drift from ep20 to ep50 bought no PER. **Conclusion.** The band is
stationary. Neither coverage + prior nor joint BT moves PER by more than 0.008. Constant LR equals
the schedule, so nothing is schedule-bound. PER rises as the anneal ends: the N = 100 arms sit 0.05
lower at ep10 and rejoin at ep20. **PER tracks confidence, not learning.** After the anneal every
term plateaus. The ctrl_50 lattice term goes 1.85 -> 1.73 over sub-epochs 10-50 with the expected
rate flat at 8.7-8.9 /s. The coverage KL *worsens* from 0.89 (ep8) to 1.33 (ep24).

**N = 20 decision** (from label-free curves only). The phone rate stalls within a few sub-epochs of
the anneal's end (8-11 with the 10-sub-epoch anneal, 5-7 with the 4-sub-epoch one), and the losses
stall within about ten. The stall follows the anneal, not the update count. Decision: N = 20 with
the §1 schedule. The N = 50 arms stay the G4a.4 reference.

(src: SAE_4A_budget.md § Objective, § Design, § Results: G4a.4 read, Interim kept-checkpoint reads,
Sub-epoch count for future arms, Training objective of the six N = 100 arms)

## 7. InfoMax: conditional-entropy penalty and augmentation invariance (gate G4a.5)

**Question.** Can a symmetry breaker move the model off the content-free stationary point (objective
note section 7)? The user proposed min D(Q^Y, P_Y) + lambda H(Y | X) (a deliberate mode-seeking
cold-start device) plus augmentation invariance (IMSAT). **Design.** N = 50, one node, ctrl_50's
bed, paired against ctrl_50. Entropy term: per-output-frame entropy over the 40 outputs, normalised
like l_tau. Invariance term: per-frame KL(clean || augmented), teacher detached, BatchNorm frozen.
Two views: SpecAugment, and a swap of the per-dimension feature mean and std with another utterance
(a speaker/loudness proxy). Amended before launch: the control turned out confident by itself, so
the held-penalty arms were dropped. lam_cons 1.0 was 6.4x the objective's gradient norm and was
reduced. Arms: ent_50: lambda_ent 0.1 over sub-epochs 1-10, geometric to 0.001 by 20, then 0.
entaug_50: ent_50 + lam_cons 0.03. aug_50: lam_cons 0.03. aughi_50: lam_cons 0.1. **Gate G4a.5.**
The G4a.4 thresholds at sub-epoch 50. Band exit needs PER more than 0.05 below the arm's own null.
Symbol-usage entropy below 3 bits from ep10 on is a FAIL.

| ctrl_50 band reference, dev-other | ep1 | ep4 | ep10 | ep25 |
|---|---|---|---|---|
| eval-mode entropy (nats / output frame) | 3.04 | 2.41 | 0.38 | 0.23 |
| symbol-usage entropy (bits of 40) | 2.82 | 4.56 | 4.99 | 4.99 |
| PER minus own null | +0.005 | +0.002 | -0.010 | -0.012 |

These replace the code-review values 3.25 / 2.71 / 0.30 / 0.19 still quoted in the objective note
section 7. **Outcome: FAIL for all four arms.** Ep50 PER, paired against ctrl_50 at 0.897 (* = CI
excludes 0): ent_50 0.916 (+0.019*), entaug_50 0.891 (-0.006*), aug_50 0.898 (+0.002), aughi_50
0.891 (-0.006*). No arm leaves its null by more than 0.021; symbol usage stays at 5.0 bits; no abort
fired. (Unresolved: this log gives ctrl_50's ep50 rate as 9.16 /s, the budget log 10.79 /s.)
**Conclusion.** ent_50 is sharper at ep1 (1.21 vs 3.04 nats), but ctrl_50 reaches the same floor by
ep10: sharpening does not change *which* partition is reached. Invariance moves PER by 0.006, a
third of the control's own checkpoint-to-checkpoint movement.

**Private-code analysis of ctrl_50** (audited; a label-using diagnostic, dev-other):

| | ep1 | ep4 | ep10 |
|---|---|---|---|
| PER as scored | 0.855 | 0.869 | 0.897 |
| best many-to-one relabeling / one-to-one with drop | 0.820 / 0.822 | 0.827 / 0.862 | 0.855 / 0.841 |
| label-free decipherment (HMM + EM under the prior, held half) | - | 0.842 | 0.855 |
| NMI(symbol, phone): tokens / frames | 0.115 / 0.087 | 0.063 / 0.057 | 0.056 / 0.256 |
| frame error, best many-to-one (identity 0.92) | 0.898 | 0.869 | 0.699 |
| NMI(symbol, unit) (gold phone 0.461) | 0.148 | 0.092 | 0.377 |
| NMI(symbol, speaker) (gold 0.0035) | 0.018 | 0.041 | 0.0065 |

A random many-to-one map with the same target multiset already scores 0.868 at ep10. **Conclusion.**
It is **not a relabeling of phones**: every relabeling stays in the band, and token NMI falls.
Label-free decipherment matches the oracles, so there is nothing phone-like to recover, and output
permutation is not a lever. It is **not a speaker code**. It **is a confident frame-level acoustic
code** (about 1.3 of 5 bits of frame-level phone information) whose token sequence is not
phone-like. It tracks the units less than gold phones do. **Withdrawn:** "the labeling is already
prior-best"; no relabeling search was run. From about sub-epoch 10 this reading **supersedes** the
diffuse stationary-point account.

(src: SAE_4A_infomax.md § Objective, § Design, § Gate, § Results: ctrl_50 band reference, G4a.5
read, Private-code analysis of ctrl_50)

## 8. Waveform cut versus feature mask (gate G4a.8)

**Question.** wav2vec-U 2.0 cuts silence from the waveform before SSL extraction, while the bed
masks features after it. Does this matter at N = 20? **Design.** A data job applies fairseq's cut
rule to the bed's own 10 ms rVAD labels (no margin, minimum length or merging). It re-extracts L15
on the trimmed audio and applies the frozen PCA-96 / K500 quantizer. Pack: ctrl_20, ctrl_20_s1,
prepro_20; funded on a non-empty treatment and an exact extraction-path null. The data check found
trimmed frames 0.56-0.7 % fewer than retained frames and unit agreement with the bed of
0.7055-0.7377 (0.22-0.27 within one frame of a splice, 0.72-0.75 far from one); untrimmed audio
agrees 1.0000 with bit-identical states. So the cut changes layer-15 states across the whole
utterance. **Gate G4a.8.** The G4a.4 thresholds. Primary read: prepro_20 - ctrl_20 against the seed
band.

| dev-other PER | ep1 | ep4 | ep10 | ep20 |
|---|---|---|---|---|
| ctrl_20 / ctrl_20_s1 / prepro_20 | 0.855 / 0.852 / 0.851 | 0.875 / 0.873 / 0.881 | 0.869 / 0.880 / 0.891 | 0.875 / 0.876 / 0.889 |
| prepro_20 - ctrl_20 | -0.004 [-0.007, -0.001] | +0.006 | +0.022 | +0.014 [+0.011, +0.017] |
| ctrl_20 - ctrl_20_s1 | +0.004 | +0.002 | -0.011 | -0.001 [-0.004, +0.001] |

**Outcome: FAIL.** Rates are 9.03-9.26 /s from ep4 (3.3-3.6 /s at ep1 under LR warmup, gap -0.007).
**Conclusion.** The cut is slightly worse from ep4 on; the silence convention is not what stalls the
bed. This supports only "not the missing ingredient at N = 20", not equivalence. The wav2vec-U 2.0
selection statistic was not produced, because no reader was registered.

(src: SAE_4A_prepro.md § Objective, § Design, § Results)

## 9. Context-dependent reverse model (deferred; design and literature only)

**Status.** Deferred without limit by the user; revisit only when a success needs an ablation or
after every other route has failed. Nothing was built. **Question.** Does p_phi(x_seg | k, previous
phone h) make phones a cheaper code than the private code? This targets the bound's slack (objective
note section 6 item 2). **Literature** (against): Yeh et al. 2019 (triphone emissions 44.7 -> 44.9
PER with non-matching text); Chorowski et al. 2019 (the code stops encoding what the decoder is
conditioned on); Ondel and Burget 2019 (constrain the generative model); wav2vec-U 2.0 (64 clusters
beat 128). The reproduced gains come from prior strength and lexicalisation. **Design**
(pre-registered, reviewed). A context head p_ctx(u | h, k, eta) emits the first 2 unit frames of
each segment (about 355k parameters). It is exact in the existing DP, whose group axis already
indexes h. Arms at N = 50: cdrev_50, cdrevci_50 (h forced to BOS), cdrevodm_50, cdrev_s2_50. A
pre-launch falsifier (label-using, discarded afterwards): fit context-free and context models on
even dev-other utterances and score the odd ones, for gold and for ctrl_50's ep10 code. Fund only if
gain(gold) - gain(private) > 0 in both fold directions, by more than their spread. Gate G4a.6 uses
the G4a.4 thresholds; the prediction is a null. **Constraints for any revisit.** Width is not a
lever: the MLP already over-parameterises the 240 x 500 table. A transducer-style phi is exact only
with finite-state phone context (up to trigram, 41^2 states); a full-context phone encoder forces
sampled y. Unit-side depth needs a context limit or unit-history dropout. A 4-gram prior (41^3
states) does not fit this DP.

(src: SAE_4A_cdrev.md § State, § Objective, § Literature, § Design, § Design review, § Gate)

## 10. What this era rules out on the cold blank-free bed (n = 1 seed each; FAIL = no take-off here)


| lever | result | where |
|---|---|---|
| removing the reverse emission | worse (+0.071) | §5 step 2 |
| unigram/bigram matching at lam_agg 1 or 10 | worse | §5 step 2 |
| MFCC K = 64 reverse observations | worse (+0.050) | §5 step 3 |
| unbiased prior window | not the cause; initial phone AH -> HH | §5 priorshuf |
| removing the lattice prior | no change | §5 step 5 |
| order-3 coverage at 0.009 / 0.1 / 1.0, with or without the prior | content-free even when optimised | §5 steps 5-6 |
| 50 / 100 sub-epochs, with or without an LR schedule | band stationary | §6 |
| jointly trained back-translation | within ±0.004 | §6 |
| entropy penalty; augmentation invariance | within ±0.02 | §7 |
| output-layer permutation by decipherment | nothing phone-like to recover | §7 |
| waveform cut before SSL | slightly worse | §8 |

Untested on this bed: more updates per epoch by a smaller batch (max_seqs 128 -> 32; the working GAN
ran 150k updates, the bed 1,140; named in the budget round but not run), an explicit end-of-word symbol
in the phone set (user idea 2026-09-21: a hard EOW the audio never realises is a free symbol for the
private code, so it needs a training arm read by the derangement gap, not a CPU screen), a smaller
reverse inventory, the GAN
generator auxiliaries, and tau ending at 1 (which the objective note argues against). Stronger priors
were tested afterwards: `SAE_i6_ref_lexicon.md`.
