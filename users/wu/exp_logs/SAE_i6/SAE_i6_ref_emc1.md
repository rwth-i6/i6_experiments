# SAE_i6 reference: EMC round 1 — the CTC-blank exact-marginal cycle (S1–S3b, S2d)

This part covers the first round of phase 4A: the exact-marginal cycle (EMC) with a 50 Hz CTC recognizer that has a
blank symbol. It records the seeded refinement reads (S1-S2d), the cold-start reads (S3, S3b) and the degradation
analysis. `SAE_i6_ref_emc2.md` covers the second round (S2e-S2g, S3c/S3d, numerics, six-gram, M512, higher-context pilot,
recognizer-factor diagnostic). The loss derivation is in `SAE_i6_ref_objective.md`. The later main line (VAD plus a
blank-free stride-3 recognizer) is a different model; see the package README. The ported package keeps this round's
lattice (`model/lattice.py`), aggregate term (`model/agg.py`), rate term (`model/rate_term.py`) and init anchor
(`anchor_weight`). It cuts the entropy, consistency, content and BT terms, which raise `ValueError` at non-default
values. No preset reproduces any arm named here; all presets are blank-free.

## 1. Method as run

**Data.** Speech: LibriSpeech train-clean-100 ("tc100", 28,539 utterances, about 100 h), unlabelled, with no VAD in
this round. Text: the phonemized LibriSpeech LM text "T_phi" (39 ARPAbet phones plus SIL; SIL inserted at word
boundaries with p_sil = 0.5, as in the phase-1c GAN text). Reads use dev-clean (2703 utterances) and dev-other (2864
utterances, 33 speakers). Greedy PER takes the per-frame argmax, collapses repeats, drops blank and SIL, and is scored
against MFA gold phones. Labels enter reads only.
**Components.** Recognizer theta: frozen wav2vec 2.0 large-lv60 layer-15 (L15) features (50 Hz, 1024-d), then a conv
network with kernel 9 and stride 1 (T = S = 50 Hz), dropout 0.1 twice and BatchNorm. It has 1.43 M params and 41
outputs (blank, 39 phones, SIL). Stride 1 was chosen over w2v-U 2.0's stride 3 because 16.7 Hz leaves 1.7 frames per
phone at about 10 phones/s, which makes CTC with blank infeasible on fast utterances with repeats. Stride 2 was never
run. Observation z: frozen 50 Hz k-means units of the same features, K = 500. A categorical target keeps the acoustic
and prior scales within one order of magnitude. Reverse model phi: a segmental model with one state per token, derived
from phase 3a's psi_align. It has an explicit duration p(d | k) with d_min = 2 <= d <= D_k (D = 25 for phones, D_sil =
50, SIL may repeat), stored as one free categorical per type (945 params) and initialised uniform. Its emissions
nu_phi(k, d, r, eta) depend on the duration bucket, the within-segment position r and the speaker vector eta. It has
no left context and never sees the recognizer. Speaker vector eta: PCA-16 of utterance-mean L15 features, fitted on
tc100 and frozen. Prior P_psi: a phone n-gram with interpolated Witten-Bell smoothing on the SIL-inserted text. The
bigram came first for cost only; the trigram is the full 41 x 41 history. Standing constraint: no EMC stage is
launched with a bigram prior.
**Objective.** L_tau(x) = -log sum_(pi, sigma) [ q_theta(pi | x) * P_psi(B(pi))^beta * p_phi(z, sigma | B(pi), eta)
]^(1/tau), with beta = 1. The sum runs over the joint latent: the CTC path pi and the segmentation sigma. A banded
log-semiring forward pass with a manual backward computes it exactly over (t, s, h, f). The band W = 25 frames is the
allowed offset between the recognizer clock t and the segment clock s. The joint tempering equals the string-level
formula only at tau = 1. The per-frame target is the tempered arc posterior post_q, roughly (q_theta R)^(1/tau), where
R is the product of the prior and the reverse model. Added terms: L_agg = KL(c_text || c_hat) over unigrams plus
bigrams, with c_hat the expected counts under q. lambda_agg = 0.1 and the EMA decay is 0.99. It is a collapse guard,
not an anchor. Init anchor (arms B and D): a factor q_init(pi)^alpha inside the summand. Self-distillation (control
E): the per-frame KL(q_init || q_theta), with lam_tau = 0. Rate term (S3b-R): lam_rate * ((E_q[N]/T - rho)/rho)^2 per
utterance, where N is the expected number of emitted non-SIL tokens. rho is label-free: T_phi phones per word times
2.7 words/s, which gives 9.6619 phones/s. The gradient reaches theta only, through a central finite difference in a
non-SIL tilt (2-3 DP calls per step). The code's lam equals the user's (r - rho)^2 weight divided by 26.8.
Temperature: fixed tau = 2 in the seeded arms. The cold anneal is 8 / 5.04 / 3.17 / 2 over sub-epochs 1-4, then 2. tau
= 1 was never run.
**Training.** Adam betas (0.5, 0.98), clip 5, theta lr 1e-4 (w2v-U 2.0's 5e-5 scaled linearly with the doubled batch),
phi lr 3e-3. Batch: 128 utterances or 88,000 padded frames. A sub-epoch is a quarter of tc100 (about 7.1k utterances,
56-58 steps), with a checkpoint at each. The seeded warm-up trains phi alone for 1 sub-epoch with theta frozen.
Label-free selection is argmin weighted_lm_ppl (Baevski) over the dev-clean plus dev-other decodes; it is label-free
but not out-of-sample. Revert rule: a sub-epoch whose emitted rate leaves [0.6, 1.5] x the text rate is reverted.
**Cost (one GH200).** Bigram: 1.48-1.648 s per step at B 125-128, about 76 utt/s and about 123 s per cold sub-epoch.
The DP takes 89-91 % of the step as a per-frame loop of about 86.6k tiny kernels. It is device-bound at B >= 128, so
only a fused DP kernel would help. Class trigram (C = 8, |h| = 369): 7.9x the bigram, so the cost is linear in |h|.
With the log-matmul history reduction "D3" and forward-table checkpointing "D4" (stride 32) it takes 4.33 s/step and
16.84 GiB. Full trigram: 10.31 s/step and 9.90 GiB (6.9x). In runs a sub-epoch took 1734 s against 277 s for the
bigram. Held-out phone perplexity: bigram 14.23, trigram 9.47, 4-gram 7.03, class trigram C = 8 10.67 and C = 16 9.94.
A 4-gram pruned to about 2000 states gives 7.94. Unfunded routes: pruned 4-gram states lack group structure, so D3
does not apply (about 40x the bigram). A two-pass pruned-then-exact lattice scores z at the wrong frame in its
single-clock pass and adds a peakiness ratchet.
(src: SAE_4A.md § Objective; § Design decisions; § Stages; § Standing constraints carried; § Prior order: cost estimate; § S3b: cold-start remedies)

## 2. Gates (originals; amendments marked)

**G4a.1 (S1a, spend control):** held-out log p_phi(z | y, eta) per frame, with phi refit per condition. PASS requires
monotonicity in kappa and gold ahead of every null (paired). *Amended before the read:* kappa became a per-token
substitution rate, because a bijective permutation is invariant under refit. **G4a.2 (seeded, WER speaks):** dev-other
sclite WER (lexicon plus official 4-gram flashlight decode, pinned decoder) at the last and the selected checkpoint.
It is a paired per-utterance delta against the arm's own init with a speaker-clustered bootstrap. "Refines" means the
CI excludes 0 in the arm's favour; "usable" means matching phase 1d (17.96 / 21.87). *Amended:* for the 10 h and 1 h
seeds the bar is the arm's own init. **G4a.3 (cold):** at sub-epoch 4 (tau = 2), dev-other PER < 0.50 AND a positive
speaker-matched derangement gap. **G4a.3b-R / -C / -BT / -CT:** PER < 0.50 at sub-epoch 4, and also at sub-epoch 8 for
BT and CT. R and C also need a gap > 0 with the CI excluding 0, and a greedy rate in [0.6, 1.5] x rho = [5.80,
14.49]/s. BT and CT are read paired against lam3_tri. CT also needs its content monitor above 1/K. A PER inside
0.83-0.90 at both reads closes CT. **G4a.S2d:** paired dev-other arm-minus-init delta. HOLD: CI95 upper bound < +0.010
at sub-epoch 8. IMPROVE: upper bound < 0 at both the selected checkpoint and sub-epoch 8. An arm above 0.150 at
sub-epoch 4 reads as anti-aligned. BT helps only if its CI excludes 0 at sub-epoch 8, with the same sign on dev-clean.
(src: SAE_4A.md § Gates; § Stages; § S3b: cold-start remedies)

## 3. S1a: the refit reverse term is content-specific (G4a.1 PASS)

Question: does log p_phi separate true from corrupted transcripts here? Phase 1a's HSMM and phase 1f's matcher did
not. Setup: 998 paired dev utterances (799 fit, 199 held out), phi refit from scratch per condition (0.37 M params, 10
epochs, lr 3e-3), CI clustered by speaker.

| condition | log p_phi / frame | gold minus condition [95 % CI] |
|---|---|---|
| gold | -3.644 | — |
| kappa 0.1 / 0.25 / 0.5 / 1.0 | -4.012 / -4.439 / -4.928 / -5.218 | +0.368 / +0.796 / +1.285 / +1.575 |
| same-speaker derangement | -5.207 | +1.588 [+1.525, +1.652] |
| unigram draws (matched length) | -5.205 | +1.602 [+1.538, +1.663] |
| constant fluent string | -5.088 | +1.450 [+1.376, +1.523] |

The score is monotone in kappa and gold leads every null. Refit noise is about 0.03 nats/frame. A wrong same-speaker
transcript scores no better than random phones. Caveat: the derangement is nearest-length (exact on 14 % of pairs);
the exact subset still gives +1.42. This licensed S2, but it scores GIVEN strings with a REFIT phi; a jointly trained
phi can reward content-free strings (§7). (src: SAE_4A.md § S1a training-free alignment read)

## 4. Seeded refinement: S1b, S2, S2b, S2c and the WER read

**Inits** (greedy PER, dev-clean / dev-other): 10 h seed (iii): CTC on the 2849-utterance supervised seed, epoch 24,
0.058 / 0.116. 1 h seed: 287 utterances, epoch 240, matched in updates, 0.068 / 0.130. GAN-lineage init (i):
CTC-distilled on phase-1d pseudo-labels, 0.166 / 0.218. It is worse than phase 1d's own 0.138 / 0.172 and emits
near-one-hot posteriors (entropy 0.13 nats). The seed supervision is disclosed.
**S1b and S2.** S1b ((iii) plus L_tau) went 0.058 -> 0.230 -> 0.250 dev-clean over 2 sub-epochs. It is CONFOUNDED
(random phi, unfrozen theta) and superseded by S2b arm A. The S2 GAN-init arms after 6 sub-epochs reached 0.353 /
0.384 (A, unanchored) and 0.345 / 0.374 (B, alpha 1 -> 0). Held-out L_tau fell in every run while PER rose. The GAN
init was retired as a start: it shares the distribution-matching signal with L_tau, so its failure is not diagnostic.
**S2b** (seeds, warm-up phi, tau 2, bigram, 6 sub-epochs). Arm A is plain L_tau. Arm B anchors with alpha
1/0.75/0.5/0.25/0/0. Arm C is A with lambda_agg = 0. 10 h greedy PER, dev-clean / dev-other:

| sub-ep | A | B | C |
|---|---|---|---|
| 1 | 0.129 / 0.174 | 0.054 / 0.106 | 0.147 / 0.192 |
| 3 | 0.177 / 0.218 | **0.053 / 0.103** | 0.177 / 0.219 |
| 4 | 0.183 / 0.223 | 0.060 / 0.107 | 0.180 / 0.222 |
| 6 | 0.184 / 0.225 | 0.140 / 0.186 | (ep5 0.185 / 0.225) |

Rates stayed at 9.1-10.1/s, with no collapse. The 1 h track replicates the pattern: A goes 0.119/0.170 -> 0.177/0.218;
B is best at 0.067/0.121 at sub-epoch 4 and reaches 0.124/0.173 at sub-epoch 6. Readings: the plain objective triples
the seed's PER while L_tau falls (1.88 -> 1.75). C tracks A within 0.02, so the aggregate term is not the driver. The
anchor holds only while alpha > 0. Selection picks the least-trained checkpoint on the degrading arms.
**S2c** (8 sub-epochs, tau 2). Arm D holds alpha 0.25 throughout. Arm E is self-distillation toward the frozen init,
with lam_tau = 0 and no phi gradient. Dev-other PER stays flat below the init: 10 h (init 0.1157) D 0.1068-0.1120, E
0.1082-0.1112; 1 h (init 0.1303) D 0.1185-0.1235, E 0.1253-0.1279.
**Decoder defect (voids every earlier WER).** Flashlight-text 0.0.7's LexiconDecoder does not merge CTC repeats. The
recognizer's 3-7-frame argmax runs were walked as repeated tokens ("LECTURES" -> "LECTURER ZZZ"). Two earlier
diagnoses, beam pruning and then the lexicon/trie wiring, are overturned. Fix: `decoder_version = 2` keeps one full
row per maximal argmax run. The v2 temperature sweep on init (i) gave WER 36.47 / 33.10 / 32.30 / 35.23 / 41.74 at T =
1 / 1.5 / 2 / 3 / 4. The chain below runs at T = 1.0 (beam 500, lm_weight 2.0, word_score -1.0), so absolute levels
are off-optimum. The deltas share one T.
**G4a.2 read (dev-other WER %, delta vs own init [CI]; audited).** 10 h, init 26.50. Refining rows: B ep1 25.48 (-1.02
[-1.44, -0.61]); B ep3 = selected 25.87 (-0.63 [-1.15, -0.15]); E last 25.90 (-0.60 [-1.07, -0.16]). Flat row: E
selected 26.65 (+0.14 [-0.27, 0.55]). Degrading rows: A ep1 = selected 56.01 (+29.5); A ep6 72.75; C ep1 = selected
59.80; B ep6 59.90; D last 29.68 (+3.17 [2.18, 4.13]); D selected 29.09 (+2.59). 1 h, init 36.93. The init row is
inferred from the paired rows; the audit traced it. Refining rows: B ep4 = selected 35.18 (-1.75 [-2.40, -1.11]); D
last 34.91 (-2.02); D selected 34.90 (-2.03 [-2.76, -1.32]); E selected 35.91 (-1.03 [-1.40, -0.67]); E last 36.54
(-0.39). Degrading rows: A ep1 = selected 53.76 (+16.8); C ep1 = selected 54.96 (+18.0); B ep6 53.22.
**Paired greedy PER vs init (selected checkpoints).** 10 h: B -0.0131 [-0.0165, -0.0101], D -0.0090, E -0.0060, D
minus E -0.0020 [-0.0054, +0.0013] (not separable). 1 h: B -0.0090, D -0.0117 [-0.0150, -0.0088], E -0.0050, D minus E
-0.0084 [-0.0110, -0.0060]. The plain arms (A and C) lose 0.04-0.11.
**Conclusion.** Nothing is usable. The anchor buys about 0.01 PER, or 1-2 WER. Self-distillation, which never sees the
objective, matches about half of D's 1 h gain and all of the 10 h effect.
**Unresolved.** At 10 h, D is WER-worse than its init (+2.59 to +3.17) while its paired PER is better (-0.0090). The
source does not reconcile the two.
**Later qualification.** Round 2's (`SAE_i6_ref_emc2.md`) stabilized seeded recipe S2f U_joint (fixed tau 2, seed-KL 1, no rate term, FP64
lattice, phi trainable) does meet IMPROVE and REFINE.
(src: SAE_4A.md § Init (i) and S1b oracle-drift reads; § S2 GAN-init arms and the oracle start point; § S2b, 10 h seed track; § S2c; § G4a.2 read)

## 5. Degradation investigation: why plain L_tau at tau 2 drifts

**Bugs excluded first.** At step 0 the exact target equals the recognizer: no symbol shift, no blank/SIL swap, and no
sign error (finite differences 24/24, ratio 0.989-1.006). The offline path reproduces the logged dev loss exactly.
**Pre-registered rule.** Fixed-point drift calls for a model-side remedy. A flat fixed point with a degrading E points
to the optimiser. A flat fixed point with E holding points to phi co-adaptation.
**Read (a): sequence errors (audited).** The degradation is the merging of SHORT gold segments. Dev-other deletion
rate by gold length, init -> A ep1 -> A ep6: 1 frame 6.6 -> 25.0 -> 31.3 %; 2 frames 4.2 -> 17.3 -> 23.7 %; 3 frames
2.2 -> 9.0 -> 13.4 %; 6-8 frames 1.1 -> 1.4 -> 2.1 %. Deletions explain +10.4k of the +10.3k error rise at ep1. 81 %
of utterances are worse, so the effect is population-wide. Gold "X X" pairs merge 25.6 -> 45.9 -> 55.6 %. CH absorbs
insertions. Arm B at ep3 keeps the init pattern.
**Read (a2): phi was unfitted.** After the 56-step warm-up the durations were near uniform: E[d] 13.44 frames (gold
4.68), P(d <= 3) 0.103 (gold 0.471), KL(gold || model) 1.03 nats. Every segment therefore paid about 3 nats regardless
of length. The emissions did fit (per-type entropy 3.25 against log K 6.22). Durations moved slowly toward gold during
the arms (E[d] 12.5 at A ep1, 8.5 at A ep6, 7.6 at D ep8).
**Defect.** Adam steps cannot fit a 945-parameter categorical in time. A count-based init (hard EM from the seed's
argmax runs, label-free) is needed.
**Reference: MFA durations.** Phones: mean about 4.1-4.2 frames, p99.9 18-19, with 0.011-0.019 % above 25 frames and
3.4-4.9 % one-frame. SIL runs above 50 frames: 1.26 % (dev-other) and 4.33 % (tc100). D = 25 therefore truncates
almost nothing, while d_min = 2 excludes one-frame segments.
**Read (b): network-free fixed point (audited).** Iterate q_{k+1} = target(q_k) from the seed's posteriors (bigram,
warm phi, 300 dev-other utterances):

| k | 0 | 1 | 2 | 5 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|---|
| PER | 0.1100 | 0.0962 | 0.1048 | 0.1400 | 0.1898 | 0.2421 | 0.2573 |

One step improves the seed. The gain is an insertion trim, and its sign depends on the checkpoint. Repetition drifts
monotonically. Neighbouring phones absorb short segments; they do not become blank. Single-component ablations move
the one-step target by < 0.003, because q_theta dominates at tau = 2.
**S2d launch check (pre-registered):** PER(k = 10) <= PER(k = 0).
**Read (b2): remedies**, PER at k = 1 / 10: uniform prior 0.1052 / 0.2399 (the prior restrains the drift); uniform
reverse model 0.1007 / 0.5842 (phi's evidence ties the target to the audio); label-free argmax-run durations 0.0961 /
0.1705 (as good as gold durations, 0.1720); full trigram 0.0936 / 0.1641; trigram plus durations 0.0936 / 0.1491
(0.2015 at k = 30). No remedy passes. Each useful one delays the curve by about 2 iterations without changing its
shape.
**Read (b3): anchored fixed points** (clean bigram with fitted durations), PER at k = 10: alpha 1.0 holds at 0.0981;
alpha 0.5 drifts slowly to 0.1015; alpha 0.25 passes the seed by k = 6 and reaches 0.1116. A held anchor is a product
of experts with a fixed q_init, so it is NOT a cold-start mechanism. The first anchored trigram run was void because
of the row-sum defect (§9).
**Overturned.** The claim after (b2), that the fixed point is R's posterior with a ceiling of about 0.16-0.19, is
wrong. The iterate is a factorised per-frame (mean-field) projection. No un-anchored line plateaus by k = 30, though a
later plateau is not excluded. A later amendment adds that finite trajectories establish no optimum or fixed point.
**Literature.** The design's ESPUM "bigram suffices" citation was wrong (ESPUM uses 5-grams). Mode-seeking text terms
fall to the LM mode (Liu 2017). w2v-U 2.0 fails at 25-28 Hz segment rates (PER > 100); this reverse clock is 50 Hz.
Merialdo 1994: EM drifts from a good init; an anchor halves the added errors. w2v-U 2.0's cold fix is a forward
per-frame CE onto MFCC k-means K = 64 codes (15.9 -> 13.6 PER). A gameable LM score anti-aligns with PER (REBORN). The
only executed exact-marginal mapping had a supervised acoustic side (Klejch 2022).
**Decision:** a model-side remedy (S2d), not a knob sweep.
(src: SAE_4A.md § Degradation investigation; § Read (a); § Read (a2); § Read (b); § Read (b2); § Read (b3); § Literature on the deletion mechanism; § Literature on cold-start collapse)

## 6. S3 flat cold start (G4a.3 FAIL) and the cold network-free reads

**S3 setup.** Flat theta (zero logits), random phi, the tau anneal, bigram, 8 sub-epochs of about 123 s each.
**S3 result.** Dev L_tau fell 1.74 -> 1.21. Dev-other PER was 0.8955 at sub-epoch 4 (best 0.8387 at sub-epoch 5), with
deletions at 84 % of N. The greedy rate per sub-epoch was 3.71 / 1.98 / 0.97 / 1.58 / 4.75 / 6.87 / 4.29 / 7.73 per s,
outside the band at every read. The gap was -0.3218/frame. Its CI fell to a pairing bug (§9); the sign stands. The
FAIL rests on PER alone. It licenses "not funded further", not "cannot work" (one schedule, one seed).
**Design review F1.** The expected rate under the posterior was 21.6 -> 8.9/s while the greedy rate was 1-2/s: a
DIFFUSE posterior whose argmax is blank.
**Cold EM reads** (300 dev-other utterances, tau 2, PER by iteration k; audited): c2 (flat theta, phi refit): all
blank for k = 1..30. A tau = 8 pass is identical, so the anneal is not the cause. Under the trigram, 1-3 phone types
appear from k = 14 (PER 0.985). c3 (durations pinned): unchanged. c5 (a rate tilt solved to E[N]/T = rho): met by
diffuse mass (about 0.19 non-blank per frame) while the argmax stays blank (PER 1.0). An expected count prices the
expectation, not the mode. c1 (flat theta, warm seed phi held): PER 1.000 -> 0.431 (k 10) -> 0.279 (k 30) under the
bigram, and 0.238 at k 30 under the trigram. A content-carrying phi lets a flat recognizer take off. This phi is
GOLD-DERIVED (fit on the supervised seed's decodes), so it is a ceiling, not a route. c4 (seed theta, phi refit every
iteration): PER at k = 10 is 0.1380 (bigram) and 0.1218 (trigram), against 0.1898 with a frozen phi. The drift
direction is fewer emissions.
(src: SAE_4A.md § S3 cold start: G4a.3 read; § S2b, 10 h seed track; § S3b: cold-start remedies)

## 7. S3b: cold-start remedies (all CLOSED FAIL)

**Rate term (bigram).** lam3 (lam_rate 3) ran first as F1's falsifier. F1 was refuted on its rule: the greedy rate at
sub-epoch 4 was 6.95/s, although the mode still collapsed during the anneal (0.29/s at sub-epoch 3). Sub-epoch 4: PER
0.847. The gap was +1.910/frame (macro +1.730, ci95 [1.606, 1.861]; 488/500 positive; audited), the campaign's first
positive gap. Sub-epoch 8: gap +3.289 at PER 0.886. The gap grows as a private per-utterance code sharpens, decoupled
from content.
**Packed grid, dev-other PER at sub-epoch 4 / 8:** lam1 0.829 / 0.858, lam10 0.914 / 0.898, spec 0.848 / 0.865,
spec_speed 0.849 / 0.874. All rates were in band, so G4a.3b-R and G4a.3b-C both FAIL. The spec arms add consistency
KL(p_clean stopgrad || p_aug) at lam_cons 1: SpecAugment, and SpecAugment plus 0.9/1.1 speed. Weight 1 collapses the
output onto 3-4 symbols (K/IH/T 57 % of spec's tokens). Its "never alone" guard is void, because diffuse or
content-free output is also view-consistent.
**Chance floor.** Random strings at hypothesis length drawn from each arm's own unigram score 0.840-0.922; the arms
beat that by only 0.003-0.019. The 0.83-0.91 band is therefore chance for a length-faithful string, and the gap
measures theta/phi self-consistency on a private code.
**Retracted later.** The claim "not a relabelled phone code" (Hungarian gain <= 0.003) is unestablished: the mapping
was fitted on one alignment against gold.
**P-BT probe** (standalone back-translation loop, 2000 utterances, rounds 0-3). Each round decodes, fits phi, samples
units for real text, renders them to features and trains theta by CTC. Cold phi: content-free every round (collage PER
0.867-0.886, rate never in band, gap about 0). Gold-derived warm phi: takes off (collage 0.363 -> 0.314). Renderings:
collage > speaker-run collage > centroid > learned renderer. The last two end above PER 1.0 and were dropped.
Literature on back-translation: MT back-translation without denoising yields 0.0 BLEU. Synthetic 1-best used as plain
CE destroys ASR (Hori 2019). Attenuating the synthetic branch is the largest lever (Baskar 2021). w2v-U 2.0's seed
spread is 0.9-2.2 PER. Constraint (user): BT enters only as an auxiliary interleaved with EMC, as separate optimizer
steps without gradient accumulation, with units from the live phi under no_grad. It never trains a recognizer
standalone.
**BT auxiliary (trigram cold bed).** Base recipe: lam_rate 3, the anneal, 8 sub-epochs. After every EMC step one BT
step runs on 128 T_phi sentences: argmax units from phi with sampled duration and speaker, collage rendering, and CTC
with frozen BN statistics. lam_bt ramps up over sub-epochs 1-4. Arms: bt_a_tri: lam_bt 0.1, full depth. bt_b_tri:
lam_bt 0.3, full depth. bt_c_tri: lam_bt 0.3, output projection only. _cr arms: add consistency at 0.3, SpecAugment
only. _s2 arms: seed 43. Dev-other PER at sub-epoch 4 / 8:

| arm | 4 | 8 | arm | 4 | 8 |
|---|---|---|---|---|---|
| lam3_tri (control) | 0.845 | 0.877 | lam3_tri_cr | 0.845 | 0.862 |
| bt_a_tri | 0.840 | 0.868 | bt_b_tri_cr | 0.887 | 0.894 |
| bt_b_tri | 0.857 | 0.869 | lam3_tri_s2 | 0.844 | 0.873 |
| bt_c_tri | 0.874 | 0.892 | bt_b_tri_s2 | 0.866 | 0.881 |

G4a.3b-BT FAILs; the minimum over all 64 reads is 0.836. Paired contrasts at sub-epoch 8: bt_a_tri minus lam3_tri:
-0.93 pt [-1.65, -0.13], under the 1-point floor. bt_a_tri against the seed-43 control: -0.55 [-1.32, +0.32]. BT on
top of consistency (bt_b_tri_cr minus lam3_tri_cr): +3.16 pt, i.e. worse. The trigram bought shape, not identity. The
output is syllable-shaped (EH R IY, P R IY). It scores about 2 nats/phone better than the nulls under the trigram, and
1.5 worse than gold. It is substitution-dominated (S 52-70 % of N) with flat deletion across length bins. Correct
phones are isolated (longest correct run 5). LM plausibility is anti-aligned with content across arms.
**S3b-CT: forward content term** (Liu 2022 form: MFCC k-means K 64, tap layer 1, lam 0.3) on the trigram cold bed.
Dev-other PER at sub-epoch 8, with the paired delta against lam3_tri (0.877185): content 0.887926 (+0.010740
[+0.006754, +0.015096]); BT + content 0.869733 (-0.007452 [-0.014924, -0.000033], sign flips on dev-clean); entropy-a
/ -b (rate hinge plus lam_ent 0.1 / 0.3, so confounded) 0.905441 / 0.927249 (+0.028256 / +0.050063). CLOSED FAIL. The
content term was active (code accuracy 0.345 -> 0.528 against chance 1/64). At tap 1, however, its gradient bypasses
the final phone convolution, so it trains an acoustic auxiliary and adds no identity to the output.
**Alignment-mass profile, cold lam3_tri** (300 dev-other utterances; audited). The target knows WHERE but not WHAT.
Timing is found. SIL takes 74-96 % of the mass on gold-SIL frames. The matched boundary F1 at ±2 frames is 0.75 at
sub-epoch 8, against 0.58 for random boundaries and 0.85 for the seeded reference. E[d] reaches the gold mean (4.7 vs
4.14) by sub-epoch 4. Identity is not. The gold-phone mass on speech is 1 % during the anneal and about 5 % after it.
That is 2-3x chance and no more than the recognizer's own share. The target copies the recognizer (TV 0.10). Seeded
arm A shows the same relation (TV 0.012) at 64 % gold-phone mass. The objective neither creates identity from cold nor
defends it when seeded. The row-sum tolerance was widened post hoc from 0.1 % to 0.7 %; this is disclosed and lies
within the audited fp32 range.
(src: SAE_4A.md § S3b: cold-start remedies)

## 8. S2d (seeded, trigram) and S3b-OR (frozen gold-derived phi)

**S2d.** Does the full trigram stop the seeded drift? Setup: theta from the 10 h seed, phi warmed for 1 sub-epoch
under the trigram at tau 8, then the lam3_tri recipe (rate 3, tau 8 -> 2, 8 sub-epochs) plus three BT arms. Dev-other
PER:

| arm | ep1 | ep2 | ep4 | ep5 | ep8 | selected |
|---|---|---|---|---|---|---|
| lam3_tri | 0.329 | 0.337 | 0.261 | 0.252 | 0.254 | ep7 0.255 |
| bt_a_tri | 0.329 | 0.367 | 0.272 | 0.271 | 0.265 | ep5 0.271 |
| bt_b_tri | 0.329 | 0.358 | 0.283 | 0.279 | 0.299 | ep7 0.280 |
| bt_c_tri | 0.329 | 0.407 | 0.283 | 0.283 | 0.280 | ep7 0.278 |

**G4a.S2d FAIL in all arms** (audited). The ep8 delta against the init is +0.139 [+0.129, +0.148] for lam3_tri and
+0.149 to +0.183 for the BT arms. BT is worse than the control on both splits (bt_b +0.045). Rates stay at 9.1-9.8/s
and every string is distinct. The loss occurs in sub-epoch 1 at tau 8 (TV 0.11). PER then partly recovers as tau
reaches 2 and stalls about 0.14 above the seed. The mass profile settles where bigram arm A did (gold-phone mass 63 %,
F1 0.85, TV 0.02), so the trigram did not move the attractor. The target's argmax beats the recognizer's by 0.02-0.08
PER but sits at about 0.23. Conclusion as recorded: this objective cannot refine a seed. Later qualified: "settles at
an optimum" was downgraded to "degrades and partly recovers", and round 2's stabilized S2f does refine.
**S3b-OR** (reporting-only oracle reference: flat theta, the S2d warm phi, trigram cold bed):

| arm | dev-other PER ep4 | ep8 | paired delta vs frozen, ep8 |
|---|---|---|---|
| phi frozen | 0.453832 | 0.367813 | ref |
| phi jointly trained | 0.642978 | 0.624076 | +0.256263 [+0.239735, +0.274029] |
| frozen + BT (bt_a) | 0.423280 | 0.350416 | -0.017397 [-0.032549, -0.002260] |
| frozen, seed 43 | 0.435617 | 0.383213 | +0.015400 |

A content-carrying frozen phi lets the network take off below 0.50. Joint phi updates hurt (dev-clean +0.272). The
registered reading (ii) had this sign reversed; it is corrected here. BT helps once its reverse side carries content
(dev-clean -0.040).
**OR diagnostic** (theta held fixed, phi snapshots crossed, sub-epoch 8, 300 utterances, tau 2). Updated minus frozen
phi, gold-phone mass delta [ci95] / target PER delta: with the frozen-arm theta -0.092855 [-0.096375, -0.089227] /
+0.173611; with the joint-arm theta -0.022239 [-0.024275, -0.020231] / +0.050472. Joint training degrades phi's
evidence itself, not only theta's path. This covers the whole phi (durations included) and is not shown for S2d. The
frozen-phi S2d arm proposed here was withdrawn at the time; round 2's S2e arm B ran it.
(src: SAE_4A.md § S2d result; § S3b: cold-start remedies; § Current mechanism assessment and partial follow-ups)

## 9. Defects and pitfalls

**CTC decoder:** the decoder does not merge repeats (flashlight-text 0.0.7). Fixed by decoder v2; every v1 WER is
void. **D3 underflow:** `_logmm` floored fp32 underflow per operand, which raised entries. Non-bigram posterior rows
then summed to [3.3e-9, 3891.9]. Fixed with fp64 accumulation plus an fp32 row-sum test, and every trigram fixed-point
line was rerun. An fp64 review check cannot catch this. Round 2 later moved the whole lattice to FP64. **Row order:**
`reverse.evaluate()` returns rows in length-bucket order, and the gap job paired them by position. Sums and means
stand; per-utterance CIs fell. Fixed with index keys. **Train/eval asymmetry:** the training target uses the
train-mode forward (dropout and batch BN), while offline diagnostics run in eval mode. **Rate band:**
`DecodeStatsJob.rate_band` uses the gold 9.8/s, so read gate bands off `phone_rate`. **Stale monitor name:**
`emc_agg_kl_bigram` keeps that name under the trigram. **BT frame pool:** the pool is process-local, so after a
resubmit the first BT steps collage the pool mean. **Random phi with unfrozen theta:** this confounds the first
sub-epoch; warm phi first. **Gold-derived phi:** the warm phi of c1, c4, P-BT warm and OR comes from the supervised
seed; it is a ceiling only.
(src: SAE_4A.md § S2 GAN-init arms and the oracle start point; § Read (b3); § S3 cold start: G4a.3 read; § S3b: cold-start remedies; § Prior order: cost estimate)

## 10. Mechanism assessment at the end of this round

**Main suspect.** The unanchored cycle supplies a self-reinforcing phone target, and joint phi updates can erode its
phonetic grounding. Reconstruction, rate and phonotactic plausibility improve without recovering the spoken phones.
**What the evidence does and does not establish.** The cold profile keeps timing and weak identity. It establishes
neither zero acoustic information nor a coherent private code. A good initializer is not protected. The prior helps in
ablations, but a higher order alone did not stop the drift.
**Open.** Emission, duration, LM and CTC-path effects are not isolated. Round 2's (`SAE_i6_ref_emc2.md`) recognizer-factor diagnostic showed
the recognizer ADDS identity mass to the epoch-4 posterior. This weakens the "path preferences suppress a better
target" explanation; earlier q/phi co-adaptation stays unresolved.
**Do not repeat without a new ingredient:** plain L_tau at tau 2 from a seed; a decaying anchor; a higher prior order
alone; rate, consistency, entropy, forward-content or BT-auxiliary terms on a cold phi; standalone BT rounds;
continuous unit-to-feature renderers.
(src: SAE_4A.md § Current mechanism assessment and partial follow-ups)
