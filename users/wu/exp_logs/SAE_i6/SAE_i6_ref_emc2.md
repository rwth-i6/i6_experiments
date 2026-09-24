# SAE_i6 reference: EMC round 2 — seeded refinement, cold interventions, numerics, six-gram prior (M512)
This part covers the second half of the phase-4A exact-marginal-cycle (EMC) work: the reopened 10 h-seed refinement
round (S2e, S2f, S2g), the cold interventions S3c and S3d, a float32 lattice defect and its repair, a recognizer-factor
diagnostic, and the six-gram prior line that ended with 512-path importance training (M512). The loss derivation is in
`SAE_i6_ref_objective.md`; stages S1 to S3b and S2d are in `SAE_i6_ref_emc1.md`. Numbers are copied as the source gives them.

## 0. Common terms, beds and gates
- **Cold bed, canonical control `lam3_tri`:** train-clean-100 (tc100, speech only), frozen wav2vec 2.0 large L15
  features at 50 Hz, K=500 enc50 unit targets, flat recognizer theta, random reverse model phi, full phone trigram,
  tau annealed 8 to 2 over sub-epochs 1 to 4, rate weight 3, aggregate weight 0.1, band 25, beta 1, seed 42,
  8 sub-epochs (a sub-epoch is a quarter of tc100; 8 sub-epochs = 477 updates at 128 utterances / 88,000 padded frames).
- **Seeded bed:** theta from the supervised 10 h seed recognizer (greedy PER 0.058003 dev-clean / 0.115741 dev-other),
  phi from a one-sub-epoch trigram warm-up at tau 8 (theta frozen), then speech-only EMC on tc100 for 8 sub-epochs.
- **Evaluation:** dev-other 2864 utterances / 33 speakers / 177275 phones / 50948 words; dev-clean 2703 / 40 / 193644
  phones. PER is greedy (collapse, drop blank and SIL, no LM) against MFA gold. Paired deltas use a speaker-clustered
  bootstrap: PER 2000 resamples seed 0, WER 10000 seed 42. WER uses **fixed decoder v2**: lexicon + official 4-gram
  flashlight decode, beam 500, LM weight 2, word score -1, acoustic temperature 1, sclite.
- **Label-free selector:** seeded: argmin decode weighted-LM perplexity over sub-epochs 4 to 8; cold: the phase's
  existing selector. Gold never selects.
- **G4a.3 (cold, never amended):** sub-epoch-4 dev-other PER < 0.50 and a positive own-phi speaker-matched
  derangement gap. **G4a.S2d (seeded):** dev-other arm-minus-init PER; HOLD if the CI95 upper bound at sub-epoch 8 is
  < +0.010; IMPROVE if the upper bound is < 0 at both the selected checkpoint and sub-epoch 8. **G4a.2 (WER):** v2
  WER against the same initializer at the last and selected checkpoints; "refines" = interval excludes zero in the
  favourable direction. The original "usable" bar (17.96 / 21.87) used another decoder: no usability verdict exists.

Intervals condition on fixed trained models and reused dev sets, not on training-seed variability. PER and WER are
reported separately; a PER gain does not establish a WER gain. (src: SAE_4A.md § Reopened seeded refinement)

## 1. Seeded refinement (10 h supervised seed, 100 h speech-only adaptation)
### 1.1 What went wrong in S2d
S2d (the cold recipe on the seeded bed) ends at dev-other PER 0.254441 (S/D/I 14229/16962/13915) versus the seed's
0.115741 (8171/4192/8155); paired deltas +14.913 points [14.336, 15.460] dev-clean, +13.870 [12.936, 14.775] dev-other.
Of 12770 added deletions, 11393 fall on MFA segments of 1 to 3 frames; one-/two-frame deletion rates rise from
6.59%/4.17% to 30.30%/20.39% (6 to 8 frames: 1.11% to 1.18%). Other drifts: IH->AH 366->1756, D deletions 374->2769, JH
insertions 36->3513; v2 WER 72.46% at epoch 1, 67.62% at epoch 8. The pattern does not identify the cause. Direct
self-distillation to the seed earlier reached 25.9029% WER vs 26.5035%, so remedies need a matched self-distillation
control. (src: SAE_4A.md § Reopened seeded refinement)

### 1.2 S2e: four-arm causal round
Question: are reverse-model updates the cause, and does the cycle add anything beyond self-distillation? All arms: seed
theta and own warm phi, S2d's data, seed 42, 8 sub-epochs, theta lr 1e-4, phi lr 0.003 when trained, beta 1, trigram,
band 25, aggregate 0.1, **whole-lattice FP64 in all arms** (section 5), posterior tilt alpha 0, no BT. Seed KL is
framewise `KL(q_init || q_theta)` to a frozen eval-mode copy of the seed. B-A isolates reverse updates; C-D isolates the
cycle term. One four-GPU pack, 8 h cap.

| Arm | phi | tau | cycle | seed KL | rate | sel. ep | PER ep8 | PER sel | WER ep8 | WER sel |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 h seed | - | - | - | - | - | - | 11.5741 | 11.5741 | 26.50 | 26.50 |
| A joint control | trained | 8->2 | 1 | 0 | 3 | 7 | 26.3229 | 25.3375 | 68.74 | 66.14 |
| B freeze-only | frozen | 8->2 | 1 | 0 | 3 | 7 | 24.5348 | 23.6988 | 61.78 | 59.93 |
| C stabilized cycle | frozen | 2 | 1 | 1 | 0 | 4 | 10.8859 | 10.7889 | 25.23 | 25.91 |
| D self-distillation | frozen | 2 | 0 | 1 | 0 | 4 | 11.0540 | 10.9784 | 25.90 | 26.65 |

| Dev-other contrast (points) | PER delta [CI95] | WER delta [CI95] |
| --- | ---: | ---: |
| C - init, ep8 | -0.6882 [-0.8779, -0.5197] | -1.2699 [-1.7587, -0.8245] |
| C - init, sel 4 | -0.7852 [-0.9872, -0.6000] | -0.5967 [-1.1027, -0.0925] |
| C - D, ep8 | -0.1681 [-0.2412, -0.0977] | -0.6693 [-0.8646, -0.4741] |
| C - D, sel 4 | -0.1895 [-0.2638, -0.1255] | -0.7400 [-1.0126, -0.5041] |
| B - A, ep8 | -1.7882 [-2.2608, -1.3176] | -6.9541 [-8.1009, -5.8411] |
| B - A, sel 7 | -1.6387 [-2.0783, -1.1945] | -6.2103 [-7.2616, -5.2091] |

Outcome: C meets G4a.S2d IMPROVE and G4a.2 REFINE at both endpoints; D meets PER IMPROVE but its selected WER delta is
+0.1433 [-0.2697, +0.5526]; A and B fail HOLD. Freezing phi helps but leaves severe degradation; the FP64 joint control
also degrades, so precision is not the cause; C-D shows a positive cycle contribution; C versus A bundles four changes,
so no single remedy is isolated. Dev-clean C WER: final 13.95% vs 14.65% (-0.7003 [-0.9125, -0.4869]), selected 14.77%
(+0.1176 [-0.1114, +0.3420]). Pack runtime 4h43m19s (the log gives the identical figure for the separate cold FP64
control, possibly a copy error). (src: SAE_4A.md § Reopened seeded refinement)

### 1.3 S2f: trainable reverse model and LM ablations
Question: does C's stabilized recipe (tau 2, seed KL 1, rate 0) tolerate phi updates, and what do the in-objective LM
terms contribute? All arms restart from seed theta and warm phi, FP64, phi lr 0.003, everything else as C.

| Arm | sequence-LM beta | aggregate | PER ep8 | PER sel 4 | WER ep8 | WER sel 4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| U_joint | 1 | 0.1 | 11.2385 | 10.7172 | 25.30 | 25.72 |
| U_no_lattice_lm | 0 | 0.1 | 11.6322 | 11.4038 | 25.44 | 26.25 |
| U_no_text_prior | 0 | 0 | 11.6266 | 11.3344 | 25.46 | 26.19 |

| Dev-other WER contrast (points) | ep8 [CI95] | selected [CI95] |
| --- | ---: | ---: |
| U_joint - init | -1.2071 [-1.6873, -0.7528] | -0.7832 [-1.2667, -0.2906] |
| U_no_lattice_lm - init | -1.0678 [-1.5313, -0.6125] | -0.2532 [-0.7289, +0.2136] |
| U_no_text_prior - init | -1.0481 [-1.5152, -0.5894] | -0.3140 [-0.7771, +0.1371] |
| U_joint - C | +0.0628 [-0.1364, +0.2565] | -0.1865 [-0.3299, -0.0371] |
| U_no_lattice_lm - U_joint | +0.1394 [-0.0495, +0.3368] | +0.5300 [+0.2917, +0.7476] |
| U_no_text_prior - U_no_lattice_lm | +0.0196 [-0.1383, +0.1768] | -0.0608 [-0.1788, +0.0591] |

Outcome: **the stabilized recipe refines the seed with phi trainable.** U_joint meets G4a.S2d IMPROVE (PER vs init
-0.3356 [-0.5696, -0.1143] ep8, -0.8569 [-1.0527, -0.6793] selected) and G4a.2 REFINE at both endpoints; versus C, final
PER +0.3526 [+0.2547, +0.4535], selected -0.0716 [-0.1123, -0.0311]. Both ablations reach only PER HOLD. Removing the
sequence-LM factor hurts selected WER and both PER endpoints; also removing the aggregate term changes nothing clearly.
All arms share an LM-exposed warm phi, LM-based selector and 4-gram decoder: this is not an LM-free model. Dev-clean is
mixed (U_joint final PER +0.196 [+0.077, +0.321] vs init). Runtime 1h37m56s.
(src: SAE_4A.md § Reopened seeded refinement)

### 1.4 S2g: independently supervised reverse initialization
Question: does a phi fitted on the seed transcripts beat the warm phi? Fit on the seed's gold phone strings (2,821 train
/ 28 held-out utterances, 9.998 h), same K500 units and eta, no timestamps or LM: maximize
`SegmentalReverseModel.log_likelihood(z, y, eta)` over durations with one SIL at each utterance edge, d_min 2, phone
d_max 25, SIL d_max 50; 8 epochs, batch 8, Adam lr 0.003, clip 5, loss `-sum(logp)/sum(frames)`, seed 42, fixed epoch-8
export (8m25s; train NLL/frame 4.46945 -> 3.29034, held-out 3.51912 -> 3.38357, lowest 3.36134 at epoch 6). Then
U_joint with only phi swapped (1h38m17s). Primary criterion: epoch-8 dev-other WER vs init and vs U_joint, CI upper < 0.

| Split | PER ep8 | PER sel 4 | WER ep8 | WER sel 4 |
| --- | ---: | ---: | ---: | ---: |
| dev-clean | 5.3361 | 5.3743 | 13.77 | 14.58 |
| dev-other | 10.6642 | 10.6783 | 24.91 | 25.79 |

| S2g minus baseline (points) | ep8 [CI95] | selected [CI95] |
| --- | ---: | ---: |
| dev-other WER vs init | -1.5977 [-2.0550, -1.1583] | -0.7086 [-1.1633, -0.2504] |
| dev-other WER vs U_joint | -0.3906 [-0.5501, -0.2314] | +0.0746 [-0.0435, +0.1988] |
| dev-other WER vs C | -0.3278 [-0.5062, -0.1514] | -0.1119 [-0.2442, +0.0221] |
| dev-other PER vs init | -0.9099 [-1.0931, -0.7443] | -0.8958 [-1.0906, -0.7193] |
| dev-clean WER vs U_joint | -0.3989 [-0.5648, -0.2392] | -0.0533 [-0.1369, +0.0351] |

Outcome: primary criterion met; G4a.S2d IMPROVE and G4a.2 REFINE at both endpoints; final dev-other PER vs U_joint
-0.5742 [-0.6991, -0.4583]. Selected checkpoints show no advantage over U_joint. This compares initialization
procedures, not a single ingredient, and says nothing about cold start. S2g is the best seeded WER here (24.91% vs seed
26.50%). The package's analysis-only `config/supervised_init.py` ports a blank-free variant of this fit (held-out NLL
3.2888 at epoch 8), not the S2g number. (src: SAE_4A.md § Reopened seeded refinement)

### 1.5 Causal reading and specified-but-unrun seeded follow-ups
No single culprit among temperature, rate term and missing seed anchor is isolated, and the short-phone pattern is
descriptive. S2d already tested BT from the seed: all three BT packages worsened PER on both splits (895 optimizer
steps vs 477). A crossed theta/phi checkpoint read was proposed but not run.

The next seeded decision is BT or alternation, each alone, with its own cost screen and allocation first. Compare
against U_joint (or S2g's U_joint if supervised phi is used; never change initialization only in the intervention),
with fixed epoch-8 paired WER against init and the matched control; improvement needs the CI upper bound < 0.
- **BT alone:** U_joint plus the existing detached text -> phi units -> real L15-frame collage -> theta CTC package at
  its smallest reference setting (weight 0.1, four-sub-epoch ramp, 128 sentences per real batch, full theta depth,
  argmax units with sampled durations/speaker/frame nuisance); phi gets no BT gradient; the extra step raises theta
  exposure, so gains belong to the package.
- **Odd/even alternation alone:** theta trains only in sub-epochs 1, 3, 5, 7 (phi frozen), phi only in 2, 4, 6, 8;
  one pass per sub-epoch, Adam state kept, inactive gradients None, inactive BatchNorm buffers frozen. Block-coordinate
  training, not EM: no monotonicity guarantee.
- **Later options:** reciprocal BT (specify targets, refresh, SIL/duration and normalization first); smaller phi
  updates judged by change in predicted distributions; a reverse-only auxiliary lattice with the frozen seed
  recognizer; EMA targets. One intervention per comparison, with a matched self-distillation reference.
(src: SAE_4A.md § Reopened seeded refinement)

## 2. Withdrawn and scope-corrected cold proposals
A SylCipher-style standalone initializer (shared masked Transformer over speech and text syllables) was **withdrawn by
the user before implementation**: work stays inside the cycle model, with no external initializer. Unsupervised-MT
evidence is conditional (Lample et al. 2018: en-fr BLEU 25.1 with denoising, 0.0 without, but with shared embeddings
absent here; Kim et al. 2020: denoising and BT do not rescue weak initialization). Since EMC already trains q through
detached exact posteriors and BT already exists, input corruption of the main objective was the untested piece: S3c.
(src: SAE_4A.md § Withdrawn standalone-initializer proposal; SAE_4A.md § Scope-corrected cycle proposal)

## 3. S3c: input denoising in the cold cycle
Design: `lam3_tri` plus SpecAugment on the recognizer input of real training batches in sub-epochs 1 to 4 only, clean
afterwards; the lattice uses q_theta(pi | C(x)) with the clean unit stream and eta. Masks: time width 1 to 8
(`time_max_width=8`), count uniform in 2 to min(max(T//100,2)*4, T); 2 to 5 channel masks of width 1 to 204; zero fill;
separate RNG. Aggregate/rate terms and train-mode BatchNorm also see masked input. One GPU, 6 h; no sweep.

| Epoch | Split | control PER | S3c PER | S3c - control [CI95] |
|---|---|---:|---:|---|
| 4 | dev-other | 0.845398 | 0.883881 | +0.038483 [+0.031037, +0.046189] |
| 4 | dev-clean | 0.830090 | 0.863270 | +0.033179 [+0.029421, +0.037128] |
| 8 | dev-other | 0.877185 | 0.879921 | +0.002736 [-0.001650, +0.007320] |
| 8 | dev-clean | 0.860910 | 0.870412 | +0.009502 [+0.006063, +0.013491] |

**G4a.3 FAIL:** gap positive (+2.187896 per frame at epoch 4) but PER far above 0.50. At epoch 4 S3c trades deletions
(24,456 vs 45,528) for substitutions (120,960 vs 99,188) and insertions (11,274 vs 5,152). A positive gap shows the
reverse model prefers its own strings, not phone accuracy. **Amendment to earlier S3b reads:** a Hungarian 1:1
relabeling fitted and scored on the same gold (control epoch 4: 0.845393) does not show outputs are "not a relabeled
phone code"; it bounds only that fitted mapping. (src: SAE_4A.md § S3c denoising result)

## 4. What cold outputs look like
S3c epoch 8 maps gold `N OW` to `DH AH T ER Z DH`. Among 485 dev-other groups of equal frame count (8,530 pairs), no
pair shares a hypothesis at any endpoint. S3c epoch 4 emits `AH L IY` 8,350 times in 2,526 utterances (5.2726% of
trigram positions vs 0.0816% in gold); control epoch 4 emits `EH R IY` 2,014 times (196 in gold). Outputs are recurring
fragments with utterance-specific variation and poor phone identity, not duration-only templates or coherent language.
(src: SAE_4A.md § Literal hypothesis inspection)

## 5. Lattice numerics: float32 defect and the float64 DP repair
**Defect.** At the actual cold initialization (uniform q: entropy 3.713572 nats, every class 0.024390245; epoch 1,
tau 8; 300 dev-other utterances, 96,676 frames, batches 128x233, 128x497, 44x1602), float32 DP breaks posterior mass
conservation on the long batch: 7,201/32,185 valid rows outside the 0.007 tolerance, row sums 0.987968 to 1.025847.
Float64 promotion of the same inputs passes (max error 1.3878e-13); partitions are finite in both. A constant q input
in the recognizer-factor probe (section 7) triggers it more strongly: row error 0.067930, versus 0.000766 with zero
acoustic weights.

**Repair (`lattice_float64`, default off; kept in the ported package).** Neural nets stay float32; copies of q,
segment scores, the stored prior and any anchor are promoted at the DP boundary for the main loss and every rate
finite-difference call; nothing else changes. One-step check on the first tc100 batch (128 utterances, 32,615 frames):
all float64 calls conserve mass (worst 5.1736e-14); loss 3.0786665023 vs float32 3.0838878155; gradient cosines
0.9999967795 / 0.9999999820 (theta/phi); peak GPU memory 17.14 GiB vs 9.43 GiB. Float32 passes this short batch.

**Precision-only cold control** (`lam3_tri` + `lattice_float64=True`, nothing else):

| Sub-epoch | Split | FP32 PER | DP64 PER | DP64 - FP32 [CI95] |
|---|---|---:|---:|---|
| 4 | dev-other | 0.845398 | 0.828177 | -0.017222 [-0.021240, -0.013307] |
| 4 | dev-clean | 0.830090 | 0.813689 | -0.016401 [-0.018782, -0.013999] |
| 8 | dev-other | 0.877185 | 0.860640 | -0.016545 [-0.021145, -0.011431] |
| 8 | dev-clean | 0.860910 | 0.835952 | -0.024958 [-0.027082, -0.022878] |

Own-phi gap: pooled +1.911260, utterance mean +1.749149 [1.656558, 1.850135] at 4; +3.227839 / +3.095246 at 8.
**G4a.3 FAIL.** 4 h 43 m 19 s allocation, 16,079 s training (33.71 s/update). The repair improves PER at this seed but
does not explain the collapse; this DP64 run is the matched control for all later cold arms. (src: SAE_4A.md §
Cold-initialization numerical result; § DP-only training-step result; § Precision-only control result; § Six-gram
cold-training authorization)

## 6. S3d: categorical phone-output content auxiliary
Question: does forcing acoustic information through the final categorical phone output help, unlike the failed
hidden-layer content auxiliary (S3b-CT)? With q_t the final 41-class CTC probabilities and c_t the fixed 50 Hz K64
MFCC codes of S3b-CT: `L_phone_content = mean_valid_t sum_p q_t(p) * [-log R(c_t | p)]`, R a phone-to-code table
(linear decoder on one-hot phones, row softmax), exact over all 41 classes, constant weight 0.3, updating q and R, not
phi; otherwise the DP64 control. Risks: blank absorption, a private acoustic code (which the user accepts as progress).

Result at sub-epoch 4 (then stopped by the user; no epoch 8): dev-other PER 0.8301678184 vs DP64 control 0.8281765618,
paired +0.001991 [-0.002240, +0.005766]; dev-clean +0.003047 [-0.000053, +0.005964]; gap +1.342968 [+1.235555,
+1.456392]. **G4a.3 FAIL.** The auxiliary adds acoustic information but no phone identity; the cause is not isolated.
(src: SAE_4A.md § S3d preparation: categorical phone-output content; SAE_4A.md § S3d epoch-4 result and early stop)

## 7. Recognizer-factor diagnostic
At frozen cold `lam3_tri` epoch 4 (tau 2, float64 DP, 300 dev-other utterances, 96,676 frames, 77,333 gold non-SIL
speech frames), the lattice was read with the real `log_q` and with a constant `-log(41)` input, which removes q's
path preferences but keeps CTC path multiplicity (the untilted main L_tau posterior only).

| Read | original q | neutral q |
|---|---:|---:|
| mean per-utterance phone-identity mass | 0.076275624 | 0.044485215 |
| gold-phone / all-real-phone mass on gold speech | 3823.109 / 46186.118 | 2080.066 / 45440.102 |
| target-marginal greedy PER (18,743 phones) | 0.824521 | 0.999627 |
| target-marginal phones/second | 7.238611 | 0.003620 |

Paired identity-mass difference (neutral - original): -0.031790408 [-0.036698462, -0.026740672]. The current q carries
information the frozen phi/LM calculation lacks; without it the targets nearly empty. This does not establish acoustic
identifiability or support a neutral-q training remedy. (src: SAE_4A.md § Recognizer-factor diagnostic; SAE_4A.md §
Recognizer-factor result)

## 8. Six-gram phone prior in the cycle
### 8.1 Formulation and LM
Dense six-gram states in the exact lattice are too costly, so the prior is applied over sampled candidates. For a
complete legal path h with string y(h), frame labels pi(h) and segmentation sigma(h):
`A_tau(y) = sum_{h:y(h)=y} exp((log q_theta(pi(h)|x) + log p_phi(z,sigma(h)|y,eta))/tau)`.
The **finite-candidate objective** is `L = mean_utt[-log Z_n(Y)/T]`, `Z_n(Y) = sum_{y in Y} A_tau(y) P_n(y)^(beta/tau)`,
n = 3 or 6, with Y the deduplicated strings (SIL included) of complete paths drawn by forward-filtering/backward-
sampling, 3/4 from the P3 joint posterior at tau and 1/4 at 1.5 tau, and A_tau computed exactly by a constrained
conditional DP. It is a truncated partition, not an estimate of the full P6 partition. (Reweighting trigram-posterior
samples to a new prior would instead need the ratio `exp((beta/tau)(log P_context(y) - log P_trigram(y)))`.) The LM
uses sparse counts over 1,000,000 phonemized LM-corpus lines (91,100,286 phones), recursive Witten-Bell, BOS, no EOS.
Held-out (10,000 lines / 912,142 phones) NLL per phone P3/P4/P6 2.248017 / 1.950871 / 1.666830, perplexity 9.468940
/ 7.034812 / 5.295356. (src: SAE_4A.md § Long-context prior option; SAE_4A.md § Six-gram cold-training authorization;
SAE_4A.md § Higher-context pilot result)

### 8.2 Frozen candidate pilot
DP64 control at epoch 4 (tau 2, hot 3), eight dev-other items (all speaker 116; a cost/coverage pilot only):

| Draws | unique strings (8 items) | retained full-P3 mass | P3 ESS | P4 ESS | P6 ESS | P6 max weight |
|---|---:|---:|---:|---:|---:|---:|
| 64 | 437 | 0.27903078 | 7.80479 | 3.58900 | 2.58345 | 0.67698036 |
| 128 | 824 | 0.34203705 | 11.54036 | 3.89546 | 3.42870 | 0.60125984 |
| 256 | 1525 | 0.39339895 | 17.28447 | 5.16134 | 3.87624 | 0.54186986 |

At 256 draws the longest item (532 frames) retains 3.49797e-8 of P3 mass and a 271-frame item 0.000394086 (others
0.163976 to 0.821112); P6 and P3 pick different top strings on 8/8 items. The higher-order prior changes rankings, but
256 whole-string draws are not adequate on long utterances; P6 coverage cannot be measured. (src: SAE_4A.md §
Higher-context pilot result)

### 8.3 Cost of finite-candidate training
One-batch cost, cold first training batch (128 utterances, max T 392, tau 8 / hot 12), forward + neural backward:

| Implementation | draws / prior | total s | sampling | LM | conditional DP | peak GiB |
|---|---|---:|---:|---:|---:|---:|
| serial, per-frame autograd | 256 / P3 | 1,703.300 | 272.930 | 1.040 | 1,421.618 | 5.187 |
| batched explicit log-domain F-B | 512 / P6 | 260.004830 | 14.613 | 28.427 | 208.713 | 22.9543 |
| same | 256 / P6 | 143.953577 | 14.601397 | 14.104947 | 108.114553 | 12.754991 |
| same, larger execution groups | 256 / P6 | 161.934754 | 8.478892 | 13.999245 | 126.659622 | 79.870245 |

A matched 256-draw P3/P6 run was started on a 19.1 h projection (143.953577 s x 477). **It failed on cost:** later
batches of padded length 577 / 654 / 697 took 420 / 655 / 789 s, median steps were 897.085 s (P3) and 1001.968 s (P6),
and after an 11 h 30 m allocation neither arm had finished sub-epoch 1 (P3 step 52, P6 step 46); no checkpoint exists.
**Never extrapolate training cost from the first batch.** Smaller counts (K16 = 12+4, K4 = 3+1) against a registered
<= 48 s/step gate (477 x 48 x 1.25 = 7.95 h) failed: on fixed first/median/longest batches K16 took 47.648 to 130.825 s
and K4 45.005 to 125.077 s; with conditional scoring grouped across the batch, on three random real batches plus the
longest (1226 frames), K16 took 88.909 to 100.487 s and K4 49.539 to 56.638 s (0/8 pass). Proposal sampling (29 to
40 s at K4) then dominated. The user then **vetoed K4/few-string training** as conflicting with the intended
exploration. Compiling the proposal FFBS cells (preserving FP64 sums) remains an unmeasured option. (src: SAE_4A.md §
Candidate training cost read and 512-draw amendment)

### 8.4 Complete-path importance estimator (M512)
This replaces the finite-string objective: train directly on M >= 512 sampled complete paths (CTC frame symbols, SIL
decisions and reverse durations) with no per-string conditional DP. Draw 3M/4 paths from the full-P3 posterior at tau
and M/4 at 1.5 tau, keep repeats, and with s_n(h) the live recognizer and reverse log probabilities plus frozen
beta log P_n(y) and `m(h) = 0.75 Q3,tau(h) + 0.25 Q3,1.5tau(h)`:
`log Zhat_n = logsumexp_h[s_n(h)/tau - stopgrad(log m(h))] - log M`, loss `mean_utt(-log Zhat_n / T)`, training theta
and phi. Proposal densities and paths are detached; target scores keep all parameter-dependent normalizers. Zhat is
unbiased for the full banded-path partition under support conditions; log Zhat and its gradients are biased at finite
M. The rate term uses the same sampled P_n weights for the non-SIL token mean, the existing squared relative-rate
penalty and +/-0.25 central finite differences with theta-only gradient (its value depends on phi, its gradient not).

Cost gate: all 16 cases (3 random real batches + longest; cold tau 8 and trained DP64 epoch-8 tau 2; P3 and P6; full
update including optimizer) <= 144 s, since 477 x 144 x 1.25 = 23 h 51 m fits the 24 h cap. M512 passed at 22.232 to
74.310 s/update, peak 45.787 GiB, proxy 12.308 h. An adaptive search later qualified **M1200** as the largest fully
measured count (31.948 to 143.444 s; M1216 fails one P6 case at 144.629 s; bracket open). Overlap warning: one
trained-state utterance had path ESS 1.0000016/512 and max weight 0.9999992. The user fixed M = 512 for the first run.
(src: SAE_4A.md § Candidate training cost read and 512-draw amendment)

### 8.5 M512 P3/P6 cold result
Matched cold arms (DP64 bed, 8 sub-epochs / 477 updates, tau 8->2), each sampling from its own model; both finished in
one 4 h 23 m 19 s allocation on two parallel GPUs.

| Split | P3 ep4 | P6 ep4 | P6 - P3 ep4 [CI95] | P3 ep8 | P6 ep8 | P6 - P3 ep8 [CI95] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev-other | 84.4868 | 86.3799 | +1.8931 [+0.8970, +3.1280] | 86.7663 | 87.9989 | +1.2325 [0.3414, 2.2830] |
| dev-clean | 82.7606 | 82.7601 | -0.0005 [-0.3078, +0.3109] | 85.3876 | 84.4793 | -0.9084 [-1.1859, -0.6395] |

Selectors pick P3 epoch 4 and P6 epoch 5 (dev-other 84.4868% / 85.2529%). Epoch-4 gaps are positive (+2.213992 /
+2.322235 per frame). **G4a.3 FAIL in both arms; the six-gram prior does not improve dev-other.** No WER was registered.

| Dev-other epoch 8 output | reference | P3 | P6 |
|---|---:|---:|---:|
| effective unigram vocabulary | 28.50 | 27.05 | 22.87 |
| modal first phone share | DH 15.33% | AH 69.17% | AA 93.51% |
| repeated trigram occurrences | 3.488% | 4.266% | 5.779% |
| repeated sixgram occurrences | 0.398% | 0.122% | 0.136% |
| emitted/reference phone count | 1 | 0.933 | 0.968 |

All outputs are distinct strings; length correlation 0.960 / 0.961. Collapse type: strong initial-phone bias, skewed
phone frequencies (P6 AH 14.05% vs 9.53%) and excess short-motif reuse; dev-clean is the same. The trajectory is
nonmonotone (P6 AA starts 77.65% / 45.71% / 93.51% at epochs 4 / 5 / 8). First eight phones of 116-288045-0000:
reference `EH Z AY AH P R OW CH`, P3 `AH V R AH L AE T AY`, P6 `AA AH AE F AH M P IH`. After this read the campaign
moved to the VAD + stride-3 blank-free cycle model (`SAE_i6_ref_blankfree.md`), which the ported package implements. (src:
SAE_4A.md § Candidate training cost read and 512-draw amendment; SAE_4A.md § State)

## 9. Constraints carried forward
- Cost screens use uniformly sampled real training minibatches over the whole schedule (seed-42 reservoir) plus the
  global longest batch, with actual loader order and no cropping; never fixed first/median/max batches alone.
- Candidate training uses at least 512 complete paths; K4/few-string training is vetoed; a complete matched P3/P6 run
  is capped at 24 h wall clock. The M512 result releases no further M512 run; the timed-out 256-draw pack and the
  stopped S3d run are not restarted.
- Seeded packs: 8 h training cap; no gold-best checkpoint, unregistered sweep or extra 100 h labels; report fixed
  epoch 8 and label-free-selected results separately.
- EMC comparisons use whole-lattice float64 in arm and control alike; posteriors are never renormalized to pass the
  0.007 conservation check.
- A positive own-phi gap, a lower reconstruction loss or a fitted 1:1 relabeling is not evidence of phone learning;
  only the PER/WER gates count. (src: SAE_4A.md § Candidate training cost read and 512-draw amendment; SAE_4A.md §
  Reopened seeded refinement; SAE_4A.md § State)

**Port status.** The ported package keeps `lattice_float64` and an analysis-only supervised phi fit, but cuts
primary-input SpecAugment (S3c), the content auxiliaries (S3b/S3d), the BT auxiliary, candidate/complete-path LM
training (`candidate_path_draws` accepted only at its default), seed self-distillation KL and posterior anchoring.
Rerunning S2e/S2f/S2g, S3c, S3d or M512 needs re-implementation. (src: ported package, `model/emc_model.py` docstring)

