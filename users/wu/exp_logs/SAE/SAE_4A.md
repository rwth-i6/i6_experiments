# SAE §4a — Exact-marginal cycle (EMC): tempered posterior-marginalized training with a semi-Markov reverse model

## State

Active question: can the existing exact-marginal cycle learn useful phone content from cold start?
User clarification keeps work within §4a, without GANs or supervised-derived models; seeded refinement
and standalone SylCipher initialization are withdrawn. Constraints and the canonical control are in `SAE_ref.md`.

S3c denoising, S3d categorical content and matched DP64 control are CLOSED FAIL; see their result
sections below. S3d `ReturnnTrainingJob.Pso7oeIpqYjY` was stopped after its failed epoch-4 gate;
checkpoints 1–6 remain. Do not restart it. Control `ReturnnTrainingJob.IoCQmrlJbCC0` and the frozen
higher-context pilot are complete and audited; see "Precision-only control result" and
"Higher-context pilot result" for their operating points and evidence.

The old serial profile `ContextTrainProfileJob.VglVxZtlgpI0` was intentionally stopped; do not retry it.
Rewritten P6 profiles completed successfully but failed the cost screen: 512 draws took 260.005s/batch;
256 draws took 143.954s/batch (`ContextTrainProfileJob.1y34VroYbhyf` / SLURM `1857433_1`). Both had
finite gradients. Details, operating points and artifacts are under "Candidate training cost read".
Active: one otherwise identical 256-draw profile allowing the existing memory bounds to choose execution
batch sizes, replacing the profile's hard upper limits of 16 sampling/four conditional utterances.
Use the actual neural batch size as the upper bound. The remaining diagnostic budget is 9m38s;
cap this measurement at nine minutes. No optimizer or statistical-objective change is authorized.
Submitted profile: `ContextTrainProfileJob.2ZXMPEhUGLz8` / SLURM `1857703_1`, wrapper
`config/sae_4a_context_train_profile.py`. At the watcher wake-up verify finished state, both outputs,
measured cost, actual execution sizes, gradients and peak memory before any training decision.
The prepared 512-draw pack `PackedEmcTrainJob.CNcm2jEmCouC` remains unsubmitted, with its 6.6h
bound and G4a.3 unchanged. Details and artifacts are under "Candidate training cost read".
Sparse acoustic-code work is registered in `SAE_4B.md`, with no execution authorized.

## Objective

**Priority amendment (user 2026-09-16):** the active objective is pure unsupervised cold start without GANs.
**Scope clarification (same day):** work stays within this phase's cycle-consistency model. External
initializer reproduction is not the next experiment; unsupervised MT may inform changes inside the cycle.
The original sequence below is retained as provenance; its GAN/seeded refinement stages are not the active
research objective. Existing supervised results are diagnostic controls only.

Decide, with plain PER and WER, whether the exact-marginal cycle objective

    L_tau(x) = -log sum_pi [ q_theta(pi | x) * P_psi(B(pi))^beta * p_phi(z | B(pi), eta) ]^(1/tau)

(CTC recognizer q over 50 Hz wav2vec2-L15 frames; frozen phone m-gram P_psi over phonemized text; segmental
reverse model p_phi predicting the 50 Hz K=500 unit stream z from the transcript with per-token duration and a
frozen speaker embedding eta; everything marginalized exactly by DP) does three things this project's earlier
objectives did not. **Implementation note (2026-09-15):** the code tempers the JOINT latent (CTC path pi,
segmentation sigma), i.e. the sum runs over (pi, sigma) with p_phi(z, sigma | B(pi), eta) inside the bracket;
this is what the (t, s, h, f) lattice computes exactly and what the tau = 2 fixed-point argument covers. It
coincides with the formula above only at tau = 1. The recognizer has 41 outputs: blank + 39 ARPAbet + SIL.
It should show, in order, that it:

1. **Its reverse term aligns with phonetic truth on this bed.** §1a's Gaussian HSMM from an oracle init raised
   its likelihood while PER went 0.275 -> 0.392 (`SAE_1a.md:102-106`); §1f's statistics matcher preferred a
   content-free decoy on all five configurations (`SAE_1f.md:682-687`). S1 measures whether log p_phi and PER
   move together here, with phi refit per condition.
2. **Refines the best label-free start instead of degrading it.** The GRPO loop from theta_0^G ended 4.9/6.2 WER
   worse than its own init after eight legs (`SAE_3E1.md:1300-1311`). S2 reads PER/WER per sub-epoch from a
   GAN-lineage init under L_tau, with and without an anchor to that init.
3. **Bootstraps from a flat start** with deterministic annealing (S3). Literature ceiling for the non-adversarial
   family is 33-49 TIMIT PER against 12-22 for GANs (`reports/lit_method_v2_2026-09-15.md`, Q1); nobody has
   executed the exact-lattice form (Q8; Yang/Schlueter/Ney ICASSP 2026 propose it without experiments).

Not an objective of this phase: excluding private codes. User ruling (2026-09-15): avoid collapse to a useful
degree; WER speaks; analysis gates control spend only.

## Design decisions (deltas from the method note, with the evidence behind each)

- **Phone-level, no LLM anywhere.** Text side = T_phi (`TextToPhonemeJob.THKMON3k9LJQ`, 39 ARPAbet + SIL), the
  same text side as the §1c GAN. The "no G2P anywhere" ruling (`SAE_ref.md:159-172`) scopes the orthographic
  LLM reward and lists the §1c/§1d phone pipeline as allowed; §4a is that kind of track.
- **Observation = 50 Hz enc50 units, K=500** (codebook `QuantizeStatesJob.FWpGhC941JMi`; frozen raw store
  `PackUnitsJob.I0uzRMfUrKWC`, the store the §3g scorer consumed, `SAE_3E1.md:109`). Categorical targets keep
  acoustic and prior scales within one order (§1a's PCA-48 Gaussian density drowned the prior: "O(few) nats
  against O(d*frames)", `SAE_1a.md:76-79`). The 500-way codebook's discriminability is load-bearing
  (`SAE_1f.md:537-539`); K=4000 is a later ablation.
  Reverse clock: 50 Hz, so that d_min = 2 means 40 ms. The §3a finding that the 50 Hz store never won an eta
  comparison against subsampled stores (`SAE_3A.md:260-262`) argues for a 25 Hz reverse clock at a quarter of
  the DP cost, but d_min = 2 at 25 Hz is 80 ms, above the median phone; 25 Hz with d_min = 2 is therefore an
  ablation, not the primary, and the d_min rule is not relaxed.
- **Reverse model = the §3a `psi_align` forward-sum machinery** (one state per token, exact log-space forward,
  K=500 categorical, d_min=2, ~11 M params, `SAE_3A.md:31-37`) modified to the method: explicit categorical
  duration p(d | k) with 2 <= d <= D_k (D=25 frames for phones, D_sil=50, SIL may repeat), emissions
  nu_phi(k, d, r, eta) that depend on within-segment position r and a frozen speaker embedding eta, no left
  phonetic context in the first build. It sees no recognizer posterior, hidden state or CTC path.
- **Speaker embedding eta** = PCA-16 of the utterance-mean L15 features, fitted once on train-clean-100, frozen.
  Label-free, cannot be trained into a content channel; carries at most a bag-of-units summary, disclosed.
  §3g found duration, then speaker, were what its scorer learned instead of content (`SAE_3G.md:472-478`);
  giving the reverse model eta removes the incentive to encode speaker in the symbols.
- **Recognizer** = frozen wav2vec2-lv60 L15 features (50 Hz, 1024-d) + a small trainable network (the w2v-U 2.0
  generator shape: 1-2 conv layers over the raw 1024-d features, blank + 41 outputs), T = S = 50 Hz.
  **Amended 2026-09-15 (user question on stride):** the §1c w2v-U 2.0 run (`FairseqW2vu2TrainJob.HOb2GgtYT7Bc`
  train.log resolved config) uses kernel 9, stride 3, input_dim 1024, no PCA, so its generator outputs at 16.7
  Hz. §4a keeps kernel 9 (180 ms context) and the 1024-d input but stride 1: the recognizer is CTC with blank,
  and at 16.7 Hz the ~10 phones/s dev rate leaves 1.7 frames per phone, so fast utterances with repeated
  phones become CTC-infeasible. Stride 2 (25 Hz output, halves lattice T) is the registered cost ablation,
  paired with the 25 Hz reverse-clock ablation. "PCA-512" was a w2v-U 1.0 detail and is dropped. The GAN-lineage
  init is built by CTC on the frozen §1d pseudo-labels (`GanPseudoLabelJob.xjn6QnNqwEEH`, 28,539 utts).
  **Registered decision (2026-09-15):** the "GAN/§1d output as initialization only" carve-out (`SAE.md:32-36`)
  is G-track-scoped (`config_sae_3d_gtrack_v1.py:17-20`); §4a extends it to this phase under the same
  restriction (never in-loop teacher, reward or selection signal) and discloses it as the init's provenance.
- **Prior order**: bigram first (M = 84 states, ~2e9 ops per 10 s utterance, 9 MB forward table), trigram as
  the first S2 ablation. This is a cost ordering, not a claim that a bigram suffices: Nuhn 2013 reports the
  trigram optimum 38 % wrong at cipher length 32 and 5 % at 128, with no bigram row; our utterances average
  ~125 phones, and the acoustic term does most of the work. SIL inserted at word boundaries with p_sil = 0.5,
  the §1c value (`FairseqPreprocessTextJob.bi2B89fES77z`), so the GAN comparison shares its text side.
- **Band W = 25 frames (0.5 s)**; single-clock variant is an ablation only.
- **Temperature and anchoring**: S2 arm A runs at fixed tau = 2 (exact-posterior point). Plain deterministic
  annealing discards a good initializer (Smith & Eisner 2004), and soft EM from a good init is exactly the §1a
  setting that drifted, so S2 arm B is init-anchored: the summand carries an extra factor q_init(pi)^alpha
  (the skew posterior of skewed annealing; per-frame, so it is one added log-prob per arc), alpha decaying
  1 -> 0 over sub-epochs 1-4. tau = 1.5 is the sharpening ablation. S3 (flat start) anneals tau 8 -> 2.
  tau = 1 (mode / Viterbi training) is not run.
- **Corpus-level term** L_agg = KL(c_text || c_hat) over phone unigrams and bigrams under q, one small
  lambda_agg fixed for both S2 arms (swept only as an ablation, 100 h bed only, `SAE.md:47-50`). It is a
  collapse guard, not an anchor: §1f showed low-order matching cannot separate truth from a matched-length
  decoy (`SAE_1f.md:682-687`).
- **Constants fixed before the first EMC training job (2026-09-15, orchestrator; no reference setup exists
  for them, so they are pre-registered here and swept only as ablations):** lambda_agg = 0.1 on
  KL(c_text || c_hat) summed over unigram and bigram terms (a KL of 0.1 nat then costs 0.01 nats/frame against a
  ~3.6 nats/frame reverse term: a guard, not an anchor); expected-count EMA decay 0.99 (~100 steps of 64
  utterances, about one sub-epoch window); batch = 64 utterances by frames (lattice bench: launch-bound,
  B = 64 costs the same 0.9 s as B = 16); sub-epoch = train-clean-100 / 4 (~7.1k utts), checkpoints kept at
  every sub-epoch; S2 warm-up = 1 sub-epoch of phi only from init (i). S1b = init (iii) + L_tau for 2
  sub-epochs with greedy PER per sub-epoch; it doubles as the pipeline smoke and the first-100-steps
  efficiency read. Accepted from the config build (`reports/impl_s1b_s2_configs_2026-09-15.md`): theta lr 5e-5
  (the w2v-U 2.0 generator lr, `FairseqW2vu2TrainJob.HOb2GgtYT7Bc` config) and phi lr 3e-3 (the S1a refit lr);
  warm-up at tau = 2; unsupervised selection pooled over dev-clean + dev-other with held-out L_tau as the
  tie-break; data via MultiProcDataset (6 workers, buffer 128) after the loader-bound init job.
- **Amendment after the S1b efficiency read (2026-09-15):** the first S1b job ran at 37 utt/s against the 75 utt/s
  gate. Profile on a GH200 (`reports/debug_s1b_step_time_2026-09-15.md`, `reports/exec_profile_emc_step_2026-09-15.md`,
  `analysis/profile_emc_step.py`): the lattice DP is a per-frame Python loop of unfused log-semiring elementwise
  kernels (elementwise/sum/add/sub/amax = 90 % of GPU time; cudaLaunchKernel 29 % of CPU), so the step is
  ~1.0 s at B <= 64 regardless of B and only sublinear above it (post-fix: 31 / 59 / 76 / 82 utt/s at max_seqs
  32 / 64 / 128 / 256; 1.7 / 3.2 / 6.4 / 12.8 GiB). Fixes taken (commits dce4046, ce78446, f59814f, 2fed7d6):
  collation moved off the main process (num_workers 1; 0.39 s/step of loader wait), the agg loss's second
  autograd T-loop and host sync removed (bit-equal on the toy shapes), and **batch = 128 utterances /
  88,000 padded frames with theta lr scaled linearly 5e-5 -> 1e-4** (phi lr 3e-3 unchanged; sub-epoch = ~56
  steps). The forward-dump extern_data key bug (`KeyError: 'features'`, all four S1b evals) is fixed in the same
  round. Decision: 76 utt/s at B = 128 meets the gate as measured; kernel fusion of the frame body
  (torch.compile / Triton) is queued as the efficiency item that must land before any run beyond tc100 scale,
  not before S2 (S2 on tc100 costs ~1.6 min per sub-epoch at this rate). Review (`reports/review_step_fixes_2026-09-15.md`,
  all four items PASS) notes the 75 utt/s number was derived from the B = 64 bench; the rule behind it (loader wait
  at most a quarter of the step) re-derived at the run shape and batch is 0.75 x 75.9 = 57 utt/s. The original
  number stands as written; both are reported at the re-read. Re-read on the S2b / S3 launch (2026-09-15, `reports/exec_s2b_s3_efficiency_2026-09-15.md`): S2b-10h warm-up phi (`ReturnnTrainingJob.htxT2f9FHvWw`, 58 steps) median 75.3 utt/s, loader wait 0.3 %; S3 cold start (`sBlPYBA1YcIQ`, steps 10-100) 73.8 utt/s, loader wait 0.4 %, ~123 s per sub-epoch (8 sub-epochs ~16 min). Same reading as S1b: number marginal (S3 1.6 % under 75), rule passed, compute-bound in the lattice kernel; kernel fusion remains queued before any run beyond tc100.
- **Rate**: d_min = 2 and D_sil = 50 bound the token count only weakly. The emitted phone rate is logged per
  sub-epoch; a sub-epoch whose rate leaves [0.6, 1.5] x the text phone rate (9.8/9.4 per s on dev,
  `SAE_1f.md:533-536`) is reverted, not compounded (wav2vec-U 2.0 reports PER > 100 whenever the rate drifts).
- **Not built in this phase**: back-translation (a permutation code is bidirectionally consistent; the
  unit-collage inputs are off-distribution), Wasserstein-Procrustes init (§1f: observable-graph PMI rank
  correlation 0.373/0.370 between a 0.214/0.216 floor and a 0.413/0.398 ceiling, `SAE_1f.md:524-525`, predicts
  a near-random map), unsupervised-boundary restriction (REBORN Table 4: external boundaries hurt). Each is a
  named follow-up, funded only if S2 or S3 shows a live objective.

## Gates (loose by ruling; originals kept, amendments marked)

- **G4a.1 S1a reverse-term alignment (analysis, spend control only).** Quantity: held-out log p_phi(z | y, eta)
  per frame, with phi REFIT on each condition's transcripts (the prior is reported beside it, never inside it,
  because permuting phone rows makes -log P_psi worse by construction). Conditions: gold; kappa-corrupted maps
  for kappa in {0.1, 0.25, 0.5, 1.0} [ORIGINAL; **amended 2026-09-15 before any gate read**: a bijective type
  permutation is a relabeling, and phi refit from scratch is invariant to it (implementer smoke on 200 utts:
  gold -4.786 vs kappa=1.0 -4.803 per frame, `reports/impl_s0a_2026-09-15.md`); kappa is now the per-token
  substitution rate, each token independently replaced with probability kappa by a unigram draw, nested across
  kappa, so kappa=1.0 coincides with the unigram null below and serves as a consistency check];
  speaker-and-length-matched derangement; text-unigram draws at matched
  length; one constant fluent string. Pass = monotone in kappa and gold ahead of every null on the paired
  per-utterance read. If it fails AND S1b loses more than 0.05 absolute PER within two sub-epochs, S2 waits for a
  reverse-model redesign; otherwise S2 runs regardless and WER decides.
- **G4a.2 S2 refinement (WER speaks).** Gated quantity: dev-other sclite WER of the reported checkpoint (last,
  and the unsupervised-selected one), lexicon + official 4-gram flashlight decode with the decoder parameters
  pinned to the §1d decode's values (read from `Wav2Vec2KenlmDecodeJob.AQw3EcUo6rks`), as a paired per-utterance
  delta against the arm's own init with a clustered-bootstrap CI. "Refines" = CI excludes 0 in the arm's favour.
  "Usable" = matches or beats §1d (17.96 / 21.87). PER (per-frame argmax, collapse, drop SIL, no LM) reported
  beside it. The selection formula (held-out L_2 + LM NLL of decodes + vocabulary usage + un-normalized length,
  Baevski's full recipe) is pre-registered in the selection job's docstring before the first checkpoint exists.
  A degrading run is reverted, not compounded (`SAE.md:173-175`). Never best-PER.
- **G4a.3 S3 cold start (bet).** Read at the END of the anneal (sub-epoch 4, tau = 2), not earlier: continue only
  if dev-other PER < 0.50 with a positive speaker-matched derangement gap; otherwise the cold-start route closes
  with the family behaviour recorded.

## Stages

**S0a Reverse side (implementer; code-reviewer before S1a).** `speech_llm/sae/emc/reverse.py` (psi_align-derived
segmental model with duration and position-dependent emissions and eta; test: forward-sum equals brute-force
enumeration on 8 small (S, U) shapes, d_min = 2 respected, likelihood finite on the sufficient statistics),
`prior.py` (phone m-gram from T_phi with SIL insertion; bigram and trigram; built 2026-09-15 on the §1c file
that carries the boundary SIL, `PhonemizeWithSilJob.DbFgvZOGZQ8F`, since `THKMON3k9LJQ` has no word
boundaries; interpolated Witten-Bell; S1a prior fit on a 1M-line budget, the S0b training prior must use the
full corpus), `speaker.py` (PCA-16 of
utterance-mean L15), and the S1a read job (`S1aReverseLadderJob`, CPU, 500 dev-clean + 500 dev-other MFA-gold
utterances, refit per condition, paired output json, convention printed by the job).

**S1a Training-free alignment read.** Runs as soon as S0a passes review. Numbers go to Results before S0b is
reviewed; they do not block S0b's implementation.

**S0b Lattice and training (implementer; code-reviewer before any GPU).** `lattice.py`: banded forward AND
manual backward in the log semiring over (t, s, h, f) with the three transitions of the method note; autograd
is the test oracle on tiny shapes only (the expanded per-step transition tensor is ~4.4e6 floats; autograd
would save it T times, ~9 GB per utterance). `agg.py` (expected m-gram counts over the CTC topology),
`train_steps/sae_emc.py`, recognizer init jobs (i) CTC-distill on §1d pseudo-labels with its PER read,
(ii) flat, (iii) quarantined oracle init (CTC on the 2849-utt seed dir
`TransformAndMapHuggingFaceDatasetJob.mQmb6aW1IDH5`, S1b only). Eval path: pure-torch greedy PER job; posterior
dump -> lexicon + 4-gram flashlight decode -> sclite, matching `Wav2Vec2KenlmDecodeJob.AQw3EcUo6rks` (all 2703 /
2864 utterances, zero empty). Wall time projected from the first 100 measured steps of the real job.

**S1b Oracle drift (quarantined diagnostic).** Init (iii) + L_2 training on train-clean-100, two sub-epochs, PER
per sub-epoch. §1a's row is the comparison. Artifacts never feed the ladder.

**S2 GAN-lineage refinement (main, 100 h bed, train-clean-100, 6 sub-epochs of ~7.1k utts).** Reverse-model
warm-up: one sub-epoch of phi on the init recognizer's tempered posterior, theta frozen. Then arm A (tau = 2,
unanchored) and arm B (tau = 2, init-anchored, alpha 1 -> 0), bigram, W = 25, beta = 1, one lambda_agg. Per
sub-epoch: PER, WER, phone rate, distinct-string count, derangement gap. Ablations only if an arm refines:
trigram; tau = 1.5; single-clock; lambda_agg sweep; 25 Hz reverse clock; K = 4000.

**S3 Flat start (one or two runs).** Random phi, zero logits, tau annealed 8 -> 2 over sub-epochs 1-4,
lambda_agg from S2, same reads, gate read at sub-epoch 4.

**Amendment 2026-09-15 (user direction, after the S1b read):** the GAN-distilled init (i) is retired as the main
S2 start. Reason: the GAN init and L_tau learn from the same distribution-matching signal, so "L_tau does not
refine (i)" cannot separate a bad objective from a shared blind spot. The finished S2 (GAN-init) arms are read
and banked as a diagnostic row only; no further GAN-init work. **S2b (main refinement read)** = the S2 machinery
(phi warm-up, arm A unanchored, arm B init-anchored, 6 sub-epochs, same constants and reads) started from small
real-label inits: init (iii), the 10 h seed recognizer (`ReturnnTrainingJob.65NNK8Bwxdtd`), and a 1 h variant
trained on a deterministic 1 h subset of the same seed. Supervision cost of the seed is disclosed on every claim
from this track; the usability bar for S2b is a supervised fine-tune on the same seed and features (to be located
in `SAE_ref.md` or registered before the read), not §1d. Checked 2026-09-15 (`reports/impl_s2b_configs_2026-09-15.md`):
no supervised 10 h / 1 h WER row exists in the project logs, and the seed recognizer itself IS the supervised
fine-tune of this model class on the seed, so the S2b bar is the paired delta against the arm.s own init;
published 10 h wav2vec 2.0 fine-tunes are context, not a bar (different model class). Init (iii) thereby loses its
quarantine wording: it is the disclosed 10 h seed init of S2b-10h. The 1 h seed is the banked
`TransformAndMapHuggingFaceDatasetJob.TAZO5vh3T7X2` (seed 42); its init matches (iii).s update count, not its
epoch count. G4a.2.s paired rule (WER delta vs the arm.s own init,
clustered-bootstrap CI, selected checkpoint) is unchanged. S1b is the unanchored, un-warmed precursor of S2b-10h
and drifted (+0.02 PER per sub-epoch); the prior for S2b is therefore that arm A drifts and arm B is the question.
**S3 starts now**, in parallel and no longer conditional on S2: the unsupervised case is the claim the campaign is
after. S3 runs 8 sub-epochs (anneal over 1-4, tau = 2 for 5-8); G4a.3 is read at sub-epoch 4 as pre-registered
and decides whether sub-epochs 5-8 are read at all.

## Standing constraints carried

d_min >= 2 (`SAE_3E1.md:161`); labels never train or select in S2/S3, S1 is a quarantined diagnostic; paired
reads and plain sclite WER (`SAE.md:40-45`); lambdas per bed, sweep at 100 h; no replay data in any phi refit;
unsupervised checkpoint selection by the pre-registered formula; token-rate revert rule above. ADDED 2026-09-15 (user): the LM term inside the objective must carry a higher-order dependency than a bigram — no further EMC stage launches with `prior_order: 2`; the mechanism (trigram/class-trigram in the DP, or higher-order rescoring of the target) is chosen from `reports/estimate_prior_order_2026-09-15.md` with its measured step cost.

## Results

### S1a training-free alignment read (2026-09-15) — G4a.1 PASS (audited CONFIRMED_WITH_CAVEATS,
`reports/audit_s1a_2026-09-15.md`: independent recompute matches to 5 decimals; fit/held disjoint; strings
reconstructed from inputs with 0 mismatches. Caveats: the "length-matched" derangement is nearest-length
within speaker, exact match on 14 % of pairs, mean |diff| 6.7 tokens; on the exactly matched subset the gap is
still +1.42 and the exactly length-matched unigram null gives more; kappa 1.0 vs unigram = refit noise.)

Run `work/speech_llm/sae/emc/s1a_job/S1aReverseLadderJob.TuHHK47CQwhl` (CPU, 998 paired utts = 799 fit / 199
held-out, 67 speaker clusters, 3 dropped; reverse model 0.37 M params, 10 epochs, lr 3e-3, seed 0, refit from
scratch per condition; eta from `SpeakerEtaJob.U4etvcSpsQi4` (PCA-16 fit on clean train-clean-100); prior
`PhoneNgramPriorJob.TRPE0D5nF3bh`). Gated quantity: held-out log p_phi per frame, paired per utterance.

| condition | log p_phi / frame (paired per-utt mean) | gold minus condition [95 % CI, speaker-clustered] |
|---|---|---|
| gold | -3.644 | — |
| kappa 0.1 | -4.012 | +0.368 [+0.333, +0.404] |
| kappa 0.25 | -4.439 | +0.796 [+0.751, +0.843] |
| kappa 0.5 | -4.928 | +1.285 [+1.232, +1.340] |
| kappa 1.0 | -5.218 | +1.575 [+1.504, +1.646] |
| derangement (same speaker, length-matched) | -5.207 (pooled) | +1.588 [+1.525, +1.652] |
| unigram draws (matched length) | -5.205 (pooled) | +1.602 [+1.538, +1.663] |
| constant fluent string | -5.088 (pooled) | +1.450 [+1.376, +1.523] |

Monotone non-increasing in kappa on the paired per-utterance means: yes. Gold ahead of every null: yes. The
prior column (reported beside, never inside) degrades with kappa as expected and is flat for derangement and
constant (real text). Consistency check kappa 1.0 vs unigram null: +0.028 [+0.0003, +0.055], CI marginally
excludes 0; the two conditions coincide in distribution by construction, so this is read as the refit noise
floor (~0.03 nats/frame), 50x below the gold-vs-null gaps. Reading: a real-but-wrong same-speaker transcript
scores no better than random phones (derangement gap = unigram gap), so the refit reverse term is
content-specific on this bed, the opposite of §1a's Gaussian HSMM. Consequence: S2 runs; S1b becomes a
pipeline smoke and drift diagnostic only, no longer decision-relevant for S2.

### Init (i) and S1b oracle-drift reads (2026-09-15; `reports/extract_4a_status_2026-09-15b.md`)

Greedy PER (argmax, collapse, drop blank and SIL, vs `GoldPhonesJob.ZGSp0hxyd2YP`), the recognizer of §"Recognizer"
(conv over frozen L15, 1.43 M params):

| checkpoint | dev-clean PER | dev-other PER | jobs |
|---|---|---|---|
| init (i): CTC distilled on §1d pseudo-labels, epoch 20 (`ReturnnTrainingJob.HcXzd6M2eyVZ`, dev CTC loss 0.109) | 0.166 | 0.218 | GreedyPerJob.iwsZ1KYIfMsS / M6WMMELxG5i7 |
| S1b: oracle init (iii) + L_tau, sub-epoch 1 (`ReturnnTrainingJob.93UGC7HzGC2P`; CONFOUNDED: phi random at step 0, see train-time review) | 0.230 | 0.257 | 2n7guvyHYuLk / x9GWHcVAN9jb |
| S1b: sub-epoch 2 | 0.250 | 0.277 | OetKCUla1mp4 / hF5YS6dUP65f |

Init (i) sits above §1d's full fine-tune (0.138 / 0.172) as the design review predicted for a conv head on frozen
features; it is the S2 paired baseline (its lexicon + 4-gram WER decodes are pending). S1b: held-out L_tau per frame
2.004 -> 1.921 (train 2.699 -> 1.908), phone rate 8.6 / 8.5 per s on dev (inside the revert band), no Z = 0
utterances; PER rises ~0.02 per sub-epoch on both sets while the objective falls. Same direction as §1a's
oracle-EM drift (0.275 -> 0.392), smaller so far. The oracle init's own PER was not registered; its eval is being
added (hash-neutral) so the drift has a starting point. S1b is diagnostic only (G4a.1 passed); the decision read is
S2, where arm B carries the init anchor S1b lacks.

### S2 GAN-init arms and the oracle start point (2026-09-15; `reports/extract_s2_gan_results_2026-09-15.md`,
`reports/extract_s2_gan_per_2026-09-15.md`; diagnostic row per the Stages amendment, NOT audited yet)

| checkpoint | dev-clean PER | dev-other PER |
|---|---|---|
| oracle init (iii), 10 h seed, epoch 24 (`65NNK8Bwxdtd`; GreedyPerJob.3TcOOObr1IrW / RMkjOs4czduh) | 0.058 | 0.116 |
| S1b = (iii) + L_tau, sub-epoch 1 / 2 (from the table above; confounded by the random phi start) | 0.230 / 0.250 | 0.257 / 0.277 |
| init (i), epoch 20 | 0.166 | 0.218 |
| S2 arm A (unanchored), sub-epoch 6 (`3EVuGpAEAn8m`) | 0.353 | 0.384 |
| S2 arm B (anchored, alpha 1 -> 0), sub-epoch 6 (`HUSP5F9GBUVr`) | 0.345 | 0.374 |

Both arms' unsupervised selection (`B8xyZzBQnZvz` / `THl07BoM4t2Q`, Baevski weighted_lm_ppl) picks sub-epoch 1,
the least-trained checkpoint. Held-out L_tau falls in every run (S2: 2.00 -> ~1.6; S1b: 2.00 -> 1.92); phone
rates 8.1-8.4 / s, all 2703 decodes distinct (no collapse). Reading: at tau = 2, on this bed, L_tau training
degrades every recognizer it is given, by 4x PER from the 10 h oracle within one sub-epoch and 2x from init (i)
over six; the anchor (alpha annealed to 0) only delays it. Before this is read as "objective anti-aligned", two
implementation checks are pending: (a) `analysis/emc_target_diag.py` — the per-frame q-target's own PER and its
confusion against the recognizer (a symbol-id shift between the 41 recognizer outputs and the 40 prior/reverse
symbols, or a surrogate-gradient sign error, would produce exactly this picture; a finite-difference check of the
surrogate is included); (b) the init (i) decode chain: sclite WER 51.7 / 58.3 % (dev-clean / dev-other) from a
recognizer with greedy PER 0.166 / 0.218 is not a plausible lexicon + 4-gram outcome (§1d: PER 0.138 -> WER 17.96),
diagnosed (`reports/debug_init_wer_2026-09-15.md`): the chain is correct (symbol order == ARPABET_39, blank 0 / sil 2 /
vocab 43 asserted, 0 OOV, in-job WER == sclite); the cause is the decode operating point. The distilled recognizer
emits near-one-hot posteriors (mean entropy 0.13 nats, median top1-top2 log gap 7.6; SIL column dead), so with the
pinned beamthreshold 50 the gold CTC path falls > 50 nats behind the argmax prefix and is pruned before the word-end
LM reward (sclite S 40.7 / D 6.8 / I 4.2). Remedy (in implementation): an acoustic temperature T on the log-posteriors
(log_softmax(log p / T)) in KenlmPosteriorDecodeJob, hash-neutral at T = 1, swept T in {1, 1.5, 2, 3, 4} on init (i);
T is picked ONCE as the dev-clean argmin of init (i) and frozen for every §4a decode (inits, arms, S3), disclosed
beside every WER as a dev-clean-tuned decoder constant like lm_weight; dev-other never picks T. No §4a WER is
quotable until T is fixed; the 51.7 / 58.3 % numbers are the T = 1 row of that sweep. G4a.2 is therefore NOT read yet.
Launch of S2b / S3 is held until (a) returns: if a bug, every
run so far is void and reruns at the fixed hashes; if the target is genuinely anti-aligned, S2b would only
repeat the S1b picture and S3's G4a.3 becomes the phase's only remaining question.

Decode temperature sweep on init (i) (2026-09-15, `config_sae_4a_decode_temp_v1.py`, pinned decoder otherwise; sclite
plain WER %, S / D / I %, hyp/ref words; `alias/sae/4a/decode_temp/init_i/T*/…/counts/output/decode_counts.json`):

| T | dev-clean WER (S/D/I) | hyp/ref | dev-other WER (S/D/I) | hyp/ref |
|---|---|---|---|---|
| 1.0 | 51.7 (40.7/6.8/4.2) | 53031/54402 | 58.3 (45.6/8.3/4.4) | 48966/50948 |
| 1.5 | 46.0 (35.4/7.9/2.7) | 51534/54402 | 52.4 (40.1/9.6/2.7) | 47455/50948 |
| 2.0 | 42.9 (32.1/8.8/1.9) | 50649/54402 | 49.7 (36.9/10.8/2.0) | 46468/50948 |
| 3.0 | **40.5** (28.6/10.7/1.2) | 49214/54402 | 47.9 (33.3/13.3/1.3) | 44826/50948 |
| 4.0 | 41.3 (27.2/13.3/0.9) | 47666/54402 | 48.6 (31.6/16.2/0.9) | 43179/50948 |

Pre-registered pick: T = 3.0 (dev-clean argmin). Flattening the posteriors helps monotonically to T = 3 and trades
substitutions for deletions, but 40.5 % WER at greedy PER 0.166 is still far from the §1d operating point (PER 0.138
-> 17.96 %), so beam pruning is at most part of the cause. The T decision is HELD: the 10 h seed init (greedy PER
0.058, 39 symbol types, never SIL) decodes to in-job WER 0.985 on dev-clean under the same chain
(`reports/extract_seed_init_sil_2026-09-15.md`), which a better recognizer cannot produce by pruning alone; a
debugger is on it (`reports/debug_seed_init_wer_2026-09-15.md`). No §4a WER is quotable until it returns.

DECODE CHAIN BUG FOUND (2026-09-15, `reports/debug_seed_init_wer_2026-09-15.md`; OVERTURNS the beam-pruning reading
above): the 10 h seed init (greedy PER 0.058) decodes to in-job WER 0.985 under the same chain, and the debugger
showed that `KenlmPosteriorDecodeJob`'s flashlight worker returns word labels inconsistent with the token path it
scored: with lm_weight 0 / word_score 0 the 1-best score equals the free frame-argmax acoustic score of the token
path L EH K CH ER Z (LECTURES) while the returned words are "LECTURER ZZZ", whose spellings score -39.4 on the same
dump; the wrong words are neighbouring lexicon entries (LECTURES -> LECTURER, DEAR -> DEARIE). Ruled out: tag
alignment, column order and frame rate (forward configs differ only in the checkpoint), beam pruning (bt 50 / 200 /
1000 byte-identical; at 200 the 1-best score DROPS, impossible for a sound max-search), a -inf sentinel, and the
temperature code (identity at T = 1). Sharpness only sets how far the LM repairs the mislabelling (seed median
top1-top2 gap 46.9 nats -> 98.5 %; init (i) 7.6 nats -> 51.7 %). Fix layer: the worker's lexicon / word-dict /
trie wiring (`eval_jobs.py:541-615`, `_scatter_map` :723). Consequences: EVERY §4a WER so far (init (i) 51.7 /
58.3, the temperature sweep table, the S2 GAN-init decodes) is VOID; the decode jobs must re-hash after the fix and
the temperature sweep is re-run under the pre-registered rule (the beam-pruning mechanism may vanish with the
bug); PER reads are unaffected (GreedyPerJob does not use the worker).
Root cause and fix (2026-09-15 evening, `reports/debug_decoder_wiring_2026-09-15.md`, `reports/impl_decoder_collapse_fix_2026-09-15.md`,
review `reports/review_decoder_collapse_fix_2026-09-15.md` APPROVE_WITH_CONCERNS): the wiring reading above is
OVERTURNED — lexicon, word dictionary and trie were verified correct (trie.search(L EH K CH ER Z) = LECTURES, both
word maps identical). The flashlight-text 0.0.7 LexiconDecoder in the w2vu env does not merge CTC repeats: a run of
identical non-blank argmax frames without a blank between them is walked as repeated tokens through the trie
(minimal repro without LM: "- AAA BBB -" with lexicon {AB, ABB, AAB} decodes ABB). The recognizer emits 50 Hz runs
of 3-7 frames (utt 251-136532-0022: L3 EH4 K5 CH5 ER7 blank1 Z7 -> LECTURER ZZZ); the reference w2vu2 decode is
insulated by one-frame peaks. Fix (commit aab3ed2): `collapse_frame_runs` (eval_jobs.py:155-197) keeps, per
utterance, one full log-probability row per maximal argmax run (the row with the highest argmax log-prob), applied
after the temperature and before the decoder; `decoder_version = 2` (a real constructor argument) re-hashes every
KenlmPosteriorDecodeJob and everything downstream (120 decodes, 120 CTM, 120 sclite, 96 PairedWerDelta, 10
DecodeCounts moved; trainings, dumps, GreedyPerJob, PairedPerDeltaJob, DecodeStats, selection unchanged: 1010 ids).
Evidence on the real utterance at the pinned point (lm_weight 2, word_score -1): "LECTURER ZZZ" -> "LECTURES".
v1 and v2 WERs are different operating points; every v1 WER in this file is void and is not compared with v2. The
temperature sweep is re-run under the pre-registered rule; the argmax runs also multiplied the top1-top2 gap past
the beam threshold, so the mechanism that motivated the sweep may vanish with the collapse. The
`config_sae_4a_decode_temp_v1.py` docstring still names the v1 T = 1.0 decode ids (stale text, the cell re-decodes).

Q-target diagnostic (2026-09-15, `analysis/emc_target_diag.py`, result `analysis/out/emc_target_diag.txt`, report
`reports/exec_target_diag_2026-09-15.md`; 300 dev-clean utts = first 300 of the HDF order, 109,284 frames, 5 speakers;
units from the frozen enc50 store, L15 from the feature dump, gathered by tag; SLURM 1804704, 3 min on one GH200).
Layout check on the full 2703-utt split reproduces the banked greedy PERs exactly (oracle (iii) 0.058003, init (i)
0.166223). Per combo, recognizer PER / q-target (post_q argmax) PER / frame agreement / top confusion
recognizer -> target: oracle theta + S1b ep1 phi: 0.0553 / 0.0576 / 98.69 % / blank->SIL 0.42 %; oracle theta +
S2 warm-up phi: 0.0553 / 0.0633 / 98.83 % / blank->T 0.19 %; init (i) theta + warm-up phi: 0.1667 / 0.1621 /
96.93 % / AH->blank 0.35 %. Finite differences: moving theta toward the target lowers L_tau in 24/24 probes,
measured/predicted 0.989-1.006. Reading: at the start of training the target is the recognizer (no symbol shift, no
SIL/blank swap, no sign error); the exact target does not by itself point away from the truth. The degradation
therefore comes from the trajectory (joint drift with phi at lr 3e-3, the aggregate term, or a train-time
discrepancy the offline script does not exercise: data pairing, padding, train-mode forward). A code review of the
training job's data pipeline against the offline path is pending (`reports/review_train_alignment_2026-09-15.md`).

Train-time review (2026-09-15, `reports/review_train_alignment_2026-09-15.md`, read on `ReturnnTrainingJob.93UGC7HzGC2P`'s
resolved returnn.config and learning_rates): no bug in data pairing, the T == S check, masking or the optimizer.
Three by-design facts change the reading of the runs so far. (1) S1b started phi RANDOM with theta unfrozen
(`config_sae_4a_s1b_v1.py:126`): all 59 theta updates of sub-epoch 1 were taken against an untrained reverse model
(reverse term per frame -6.86 at step 0 -> -3.62 at the ep1 dev read), so the S1b "oracle drift" row (0.058 -> 0.230)
is CONFOUNDED and is not a read of L_tau at a warm phi; it is superseded by the S2b 10 h arms, which start from a
warm-up phi fitted at frozen theta. (2) The applied gradient is l_tau + 0.1 * agg; the aggregate term fell 0.56 ->
0.11 while l_tau fell 3.38 -> 2.82, i.e. ~45 % of sub-epoch 1's loss drop, and the offline finite-difference check
covered l_tau alone. lambda_agg = 0 (the pre-registered ablation) is therefore funded now as arm C of S2b for both
inits, read as a paired delta vs arm A on the same checkpoints. (3) The training-time target is computed from the
train-mode forward (dropout 0.1 x2, BN on batch statistics, `recognizer.py:180-189`); the offline agreement numbers
are eval-mode. Queued cross-check (not launch-blocking): the diagnostic with theta AND phi from S1b `epoch.001.pt`
on the job's cv segments (285 utts) must reproduce the logged dev L_tau 2.00415. DONE (2026-09-15, `reports/exec_target_diag_replay_2026-09-15.md`): the offline path with theta AND phi from epoch.001.pt on the job's 285 cv utterances in the job's own batching reproduces dev_loss_l_tau 2.0041547616322837 and dev_loss_agg 0.7693092823 to zero difference; the offline diagnostic and the training job compute the same objective.



## S2b, 10 h seed track: per-sub-epoch reads (2026-09-15; `reports/extract_s2b_10h_reads_2026-09-15.md`)

Seed init (iii) `ReturnnTrainingJob.65NNK8Bwxdtd` ep 24: greedy PER 0.058 / 0.116 (dev-clean / dev-other), phone rate
10.1 / 9.9 per s. Warm-up phi `htxT2f9FHvWw` (1 sub-epoch, theta frozen). Arms (6 sub-epochs each, warm phi):
A plain L_tau `4BRumKQFcXim`; B init anchor alpha 1.0/0.75/0.5/0.25/0/0 `k33xnlSDACIv`; C = A with lambda_agg 0
`HNg9s1aBuD56`. Greedy PER dev-clean / dev-other per sub-epoch (dev L_tau in brackets):

| sub-ep | A: PER (L_tau) | B: PER (L_tau) | C: PER (L_tau) |
|---|---|---|---|
| 1 | 0.129 / 0.174 (1.877) | 0.054 / 0.106 (2.077) | 0.147 / 0.192 (1.876) |
| 2 | 0.151 / 0.192 (1.872) | 0.059 / 0.109 (2.079) | 0.167 / 0.213 (1.874) |
| 3 | 0.177 / 0.218 (1.798) | **0.053 / 0.103** (1.984) | 0.177 / 0.219 (1.798) |
| 4 | 0.183 / 0.223 (1.788) | 0.060 / 0.107 (1.938) | 0.180 / 0.222 (1.787) |
| 5 | 0.184 / 0.225 (1.754) | 0.115 / 0.163 (1.793) | 0.185 / 0.225 (1.755) |
| 6 | 0.184 / 0.225 (1.829) | 0.140 / 0.186 (1.860) | not read (1 row missing) |

Phone rates 9.1-10.1 / s in every arm and sub-epoch (rate rule never fires); 2702-2703 distinct decodes.
Unsupervised selection (argmin weighted_lm_ppl): A -> sub-epoch 1 (22.6; the ppl rises monotonically 22.6 -> 36),
B -> sub-epoch 3 (14.2; 14.6/14.8/14.2/14.8/20.2/23.8), C -> sub-epoch 1. Readings: (1) with a warm phi and no
aggregate term the plain objective still degrades the oracle recognizer 3x in PER while L_tau falls 1.88 -> 1.75;
arm C tracks arm A within 0.02 PER at every sub-epoch, so the aggregate term is NOT the driver — L_tau at tau = 2
is itself anti-aligned with PER on this bed beyond the first step (consistent with the offline target check: the
target equals the recognizer at step 0, the trajectory moves away). (2) Arm B holds the oracle while the anchor is
on and reads slightly BELOW the init at sub-epochs 1-4 (best 0.053 / 0.103 vs 0.058 / 0.116 at sub-epoch 3, which
the unsupervised selection also picks); once alpha reaches 0 (sub-epochs 5-6) it degrades like A. Whether the
anchored improvement is real is a PAIRED question (PairedPerDeltaJob in implementation: arm B ep3 and the selected
checkpoint vs init, per utterance, speaker-clustered bootstrap, on both dev sets; audit before any claim). (3) The
unsupervised selection rule acts as a guard: on the degrading arms it picks the least-trained checkpoint.
S1b's confounded oracle row is superseded by arm A here.

1 h seed track (2026-09-15; `reports/extract_s2b_1h_reads_2026-09-15.md`; init `ReturnnTrainingJob.t4K6Z6gHK56e` ep 240,
287 train utts: greedy PER 0.068 / 0.130, phone rate 10.1 / 9.8 per s) replicates the 10 h picture. Greedy PER
dev-clean / dev-other per sub-epoch: arm A 0.119/0.170, 0.143/0.190, 0.158/0.202, 0.169/0.212, 0.173/0.215,
0.177/0.218 (dev L_tau 1.889 -> 1.766 -> 1.840); arm B 0.068/0.126, 0.070/0.128, 0.068/0.124, **0.067/0.121**,
0.104/0.156, 0.124/0.173 (L_tau 2.103 -> 1.956 at sub-epoch 4, 1.813 at 5); arm C 0.126-0.175 dev-clean, like A.
Phone rates 9.1-10.1 / s throughout; 2703 distinct decodes. Selection: A -> 1 (ppl 22.0), B -> 4 (15.6, the
best-PER checkpoint), C -> 1 (24.1). Same three readings as the 10 h track; the anchored gain at alpha = 0.25 is
0.0014 / 0.0090 absolute PER here (0.005 / 0.013 on 10 h) and is only a claim after the paired read. Follow-up
funded and LAUNCH-READY (S2c, `config_sae_4a_s2c_v1.py`, commits af0d7f2 + 0c547f5, both tracks, 8 sub-epochs,
tau 2, same data/lr/dropout as S2b; `reports/impl_s2c_2026-09-15.md`): arm D = held anchor alpha 0.25 on every
sub-epoch (10 h `ReturnnTrainingJob.zvXHrxR2Kwdj`, 1 h `fjwkj6kVXc8W`); arm E = self-distillation control, theta
trained toward the frozen init recognizer's posteriors (per-frame KL(q_init || q_theta) on valid frames,
`train_steps/sae_emc.py:149-166`, `lam_selfdistill` 1.0) with `lam_tau` 0 (L_tau reported, not optimised) and
lam_agg 0.1 as in D; phi receives no gradient (10 h `DNWDPWsJODis`, 1 h `381lj8zEBm12`). Resolved configs differ
only in `lam_tau` / `lam_selfdistill`. E is the standard semi-supervised bar an anchored objective must beat, so
D - E reads "L_tau's tilted target vs plainly pulling toward the init"; an earlier E (lam_tau 0 with no
distillation term) was replaced before launch because it trained theta on the aggregate term alone. Pre-registered
read (docstring of the config) = paired D - E dev-other PER delta per sub-epoch and at the unsupervised-selected
checkpoint (PairedPerDeltaJob, clustered bootstrap), CI excluding 0 in D's favour; PER of each arm vs its own init
reported beside it. Caveat from the implementer: the unit test's stub recognizer is deterministic, so the
dropout/BatchNorm teacher-student asymmetry of the real net is not exercised.

S3 audit (2026-09-15, `reports/audit_s3_g4a3_2026-09-15.md`): CONFIRMED FAIL on both clauses; sub-epoch 4 dev-other
PER 0.8955 with deletions 84 % of N (near-empty decodes), gap -0.3218 (macro -0.342) with the CI wholly below 0;
checkpoint, split, gold and tau = 2 verified; the rate rule fired at sub-epochs 1-5 and 7, so the read point itself
would have been reverted. S3 is not funded further at this schedule.
## S2c, held anchor (D) vs self-distillation control (E): per-sub-epoch reads (2026-09-15; `reports/extract_s2c_reads_2026-09-15.md` + .full.md)

All four trainings finished 8 sub-epochs. Greedy dev-other PER (init -> ep1..ep8; selected = argmin weighted_lm_ppl):

| track | arm | init | ep1 | ep2 | ep3 | ep4 | ep5 | ep6 | ep7 | ep8 | selected |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 h | D (alpha 0.25 held) | 0.1157 | 0.1075 | 0.1079 | 0.1076 | 0.1072 | 0.1087 | 0.1120 | 0.1068 | 0.1082 | 7 |
| 10 h | E (self-distill) | 0.1157 | 0.1101 | 0.1112 | 0.1088 | 0.1098 | 0.1082 | 0.1104 | 0.1088 | 0.1105 | 4 |
| 1 h | D | 0.1303 | 0.1235 | 0.1225 | 0.1229 | 0.1216 | 0.1227 | 0.1185 | 0.1202 | 0.1189 | 6 |
| 1 h | E | 0.1303 | 0.1276 | 0.1279 | 0.1268 | 0.1275 | 0.1253 | 0.1269 | 0.1260 | 0.1275 | 5 |

dev l_tau: D 2.004 -> 1.903 (10 h), 2.018 -> 1.922 (1 h) while PER stays flat, i.e. the tilt holds theta where the
unanchored arm drifted; E's l_tau is flat (2.17 / 2.20, not optimised). Phone rates 9.67-10.08/s, distinct
2702-2704 for every checkpoint (no rate drift in either arm). Provisional pooled reading (NO claim until the 96
PairedPerDeltaJob reads and an audit): both arms improve on their init; D sits 0.002-0.003 (10 h) and 0.005-0.009
(1 h) absolute PER below E at matched sub-epochs and at the selected checkpoints (10 h 0.1068 vs 0.1098; 1 h 0.1185
vs 0.1253). The gain of E over init (about 0.005 / 0.003) is the self-training floor an anchored objective has to
beat; the pre-registered read is the paired D - E delta.

## G4a.2 read: fixed-decoder (v2) dev-other WER, paired against each arm's own init (2026-09-15, `reports/extract_v2_wer_2026-09-15.md` + .full.md; audited CONFIRMED_WITH_CAVEATS, `reports/audit_g4a2_wer_2026-09-15.md`)

Decoder: KenlmPosteriorDecodeJob `decoder_version = 2` (run-collapsed emissions), lexicon + official 4-gram, beam 500, lm_weight 2.0,
word_score -1.0, acoustic temperature 1.0 for every chain row; 57 decodes, all verified v2; paired delta = clustered bootstrap
(10k resamples, seed 42), negative = arm better. Plain sclite WER, 2864 utts.
10 h seed init 26.50 %. S2b 10 h: arm B ep1 25.48 (-1.02 [-1.44, -0.61]), ep3 = selected 25.87 (-0.63 [-1.15, -0.15]), ep4 28.39
(+1.89), ep5 51.37, ep6 = last 59.90 (+33.4); arm A ep1 = selected 56.01 (+29.5), ep6 72.75; arm C ep1 = selected 59.80 (+33.3),
ep6 72.93. S2c 10 h: D ep8 = last 29.68 (+3.17 [2.18, 4.13]), selected 29.09 (+2.59 [1.70, 3.48]); E ep8 = last 25.90 (-0.60
[-1.07, -0.16]), selected 26.65 (+0.14 [-0.27, 0.55]).
1 h seed init 36.93 % (implied from the paired rows; its own alias row missing, audit traces the init decode). S2b 1 h: arm B ep4 =
selected 35.18 (-1.75 [-2.40, -1.11]), ep1 35.46 (-1.47), ep5 47.44, ep6 = last 53.22 (+16.3); arm A ep1 = selected 53.76 (+16.8);
arm C ep1 = selected 54.96 (+18.0). S2c 1 h: D ep8 = last 34.91 (-2.02 [-2.79, -1.23]), selected 34.90 (-2.03 [-2.76, -1.32]);
E ep8 = last 36.54 (-0.39 [-0.64, -0.14]), selected 35.91 (-1.03 [-1.40, -0.67]).
Temperature sweep on ctc_init (i), same decoder: T 1.0 36.47, 1.5 33.10, 2.0 32.30, 3.0 35.23, 4.0 41.74 — T = 2.0 is the argmin
(-4.2 absolute vs T = 1.0); every chain row above is at T = 1.0, so absolute levels are not at the decoder's best operating point,
paired deltas are read at a common T.
Audit: six WERs re-derived from sclite.pra (2864 utts, 50,948 words; one 0.01 rounding difference where the table quoted the in-job number instead of the sclite one); the 1 h init is ScliteJob.XYeXkOKYd4VJ from the 1 h seed t4K6Z6gHK56e ep240 = 36.93 %, every delta is against the arm's own init; selection is label-free argmin weighted_lm_ppl and matches SelectPosteriorsJob for all 10 arms; the arm B ep1 delta and its interval reproduce from the per-utterance rows with a speaker-clustered bootstrap (33 speakers; clustered interval wider than unclustered, as it should be). Gate reading: "refines" (interval excludes 0 in the arm's favour, at the reported checkpoint) holds for
arm B at both seed sizes while the init tilt is active (selected checkpoints: -0.63 at 10 h, -1.75 at 1 h) and for the held-anchor
arm D at 1 h only (-2.03); the self-distillation control E, which never sees the objective, refines by -1.03 at 1 h (selected) and
-0.60 at 10 h (last), so roughly half of D's 1 h gain and all of the 10 h effect is available without the EMC objective. Every plain-
objective row (A, C, B after alpha reaches 0, D at 10 h) degrades, by up to +46 absolute. Nothing is "usable" (21.87 %). Read with
the investigation: the bigram-prior objective with an unfitted duration model has no refinement to offer beyond the anchor; S2d is
funded on the model-side remedy, not on a repeat of this design.

Paired greedy PER beside the WER (PairedPerDeltaJob, utterance-paired, speaker-clustered bootstrap, dev-other, 2864 utts; jobs under
`alias/sae/4a/{s2b_10h,s2b_1h}/paired_per/{arm}/{ep,selected}/dev-other` and `alias/sae/4a/{s2c_10h,s2c_1h}/{armD,armE}/ep*/dev-other/
paired_per_vs_{init,armE}`; `reports/extract_paired_per_inputs_2026-09-15.md`; read off the jsons by the orchestrator, one line each;
negative = arm better). 10 h (init 0.1157): arm A +0.058 (ep1 = selected) .. +0.109 (ep6); arm C +0.076 .. +0.112; C vs A +0.018 (ep1),
within 0.003 from ep3; arm B ep1 -0.010, ep2 -0.007, ep3 = selected -0.0131 [-0.0165, -0.0101] (20 % of utts worse), ep4 -0.008,
ep5 +0.047, ep6 +0.070; D ep7 = selected -0.0090 [-0.0138, -0.0046], E ep4 = selected -0.0060 [-0.0075, -0.0045], D vs E at the
selected epochs -0.0020 [-0.0054, +0.0013] (not separable). 1 h (init 0.1303): arm A +0.040 .. +0.087; arm C +0.044 ..; arm B ep4 =
selected -0.0090 [-0.0119, -0.0064], ep5 +0.026, ep6 +0.043; D ep6 = selected -0.0117 [-0.0150, -0.0088], E ep5 = selected -0.0050
[-0.0059, -0.0040], D vs E -0.0084 [-0.0110, -0.0060]. Reading: the PER picture is the WER picture — the anchor buys about 0.01 PER,
the plain objective costs 0.04-0.11, and the objective's own contribution over the self-distillation control is 0.008 at 1 h and
nothing separable at 10 h. (An earlier extractor table, `reports/extract_paired_per_2026-09-15.md`, labelled the C-vs-A contrast as
"armC"; superseded by the rows above.)

## Degradation investigation (opened 2026-09-15 evening on the user's instruction; reads pre-registered here)

Question: why does plain L_tau at tau = 2 triple the PER of the 10 h seed recognizer (arm A dev-other 0.1157 ->
0.1740 after one sub-epoch -> 0.2248 after six; phone rate 9.85 -> 9.13/s; dev l_tau 2.15 with theta frozen at the
end of the phi warm-up -> 1.88 -> 1.75 while PER rises; decode weighted_lm_ppl 22.6 -> 35.9; arm C without the
aggregate term tracks arm A). Established before this section: the exact target at step 0 is the recognizer to
within +0.002 PER (q-target diagnostic above), the pipeline reproduces the logged loss exactly, arm A starts from
seed theta + warm-up phi (`reports/extract_armA_dynamics_2026-09-15.md`: theta lr 1e-4, phi lr 3e-3, Adam betas
(0.5, 0.98), clip 5), and the train loss falls 2.20 -> 1.90 inside the first 30 steps, so the shift is fast and
directional, not optimiser noise.

Working hypothesis (to be tested, not assumed): at tau = 2 the target is q* ∝ (q_theta R)^(1/2); its fixed point
under repeated fitting is R's own posterior (prior x reverse model), so the recognizer's information decays
geometrically and the objective's optimum is wherever the generative model's preference lies — the Merialdo
mechanism the design review named (item 2). The per-step target shift is small (+0.002 PER) but accumulates; the
init tilt of arm B is what stops it, and drift resumes when alpha reaches 0. What R prefers (which phones and
durations are deleted or merged; d_min = 2 frames vs one-frame gold segments; blank/SIL mass) is the pattern to read.

Reads (scripts by the implementer, run by the executor; both report patterns, the reading is written here after):
(a) `analysis/per_error_pattern.py` — sequence-level: S/D/I split and per-phone / per-class error growth vs init,
top confusions, deletions by MFA gold-segment duration bin (1, 2, 3, 4-5, 6-8, 9+ frames at 50 Hz), gold repeat
merges, per-utterance degradation distribution and worst utterances, hyp phones/s; checkpoints init, arm A ep1/3/6,
arm B ep3/6, arm C ep1/6, both dev splits. (b) `analysis/emc_target_vs_gold.py` — frame-level against the MFA
alignment: recognizer argmax vs target argmax vs gold; where following the target helps/hurts by phone and duration
bin; mass movement (to blank / SIL / same-class / other); segment keep/delete/substitute by duration bin;
component ablations (uniform prior, uniform reverse, arm-B tilt) where the lattice call allows; and the
network-free fixed-point iteration q_{k+1} = target(q_k), k = 1..10, reporting PER and frame accuracy per k (if the
PER rises monotonically with k toward a plateau, the objective's optimum is away from the truth independently of
training dynamics; if it stays flat, the degradation is a training-dynamics effect). (c) S2c arm E (self-
distillation toward the init, running) is the training-dynamics control: a recognizer trained with the same
optimiser, data and dropout but a truth-preserving target.

Decision rule written in advance: if (b) shows the fixed point drifts from the truth with a specific pattern (e.g.
short-segment deletions from d_min = 2, or prior-driven substitutions), the remedy is on the model side (duration
floor, prior, reverse-model class) and is a new stage, not a knob sweep; if (b) is flat and (c) shows E degrades
too, the remedy is on the optimiser side (theta lr, phi/theta lr ratio, target smoothing); if (b) is flat and E
holds, the remaining suspect is phi co-adaptation (phi lr 3e-3 chasing theta), tested by a phi-frozen arm.

### Read (a): sequence-level error pattern (2026-09-15, `reports/exec_per_error_pattern_2026-09-15.md` + .full.md;
`analysis/out/per_error_pattern.{txt,json}`; all 16 recomputed S/D/I/PER match the banked per.json exactly; MFA join
2863/2864 dev-other utts). AUDITED CONFIRMED_WITH_CAVEATS (`reports/audit_short_segment_deletion_2026-09-15.md`: independent DP and MFA rasterisation reproduce S/D/I exactly and every bin within 0.15 pp; deletions give +10,386 of the +10,322 error rise at ep1; uniform thinning rejected, ep6/init deletion ratio 4.8-6.2 in bins 1-3 vs 1.5-1.9 in 6+; caveat: bin membership is boundary-convention sensitive, ordering and the short/long gap survive, exact digits do not).

The degradation is a deletion of SHORT segments. dev-other, 10 h track, deletion rate by MFA gold-segment length
(50 Hz frames; gold_n = segments in the bin; half of all gold phones are <= 3 frames):

| gold length | gold_n | mean ms | init | arm A ep1 | arm A ep6 | arm B ep3 |
|---|---|---|---|---|---|---|
| 1 frame | 8,761 | 30 | 6.6 % | 25.0 % | 31.3 % | 7.9 % |
| 2 frames (= d_min) | 37,581 | 40 | 4.2 % | 17.3 % | 23.7 % | 4.9 % |
| 3 frames | 39,513 | 60 | 2.2 % | 9.0 % | 13.4 % | 2.3 % |
| 4-5 frames | 53,327 | 87 | 1.4 % | 3.3 % | 5.5 % | 1.4 % |
| 6-8 frames | 28,702 | 133 | 1.1 % | 1.4 % | 2.1 % | 1.1 % |
| 9+ frames | 9,391 | 220 | 1.2 % | 1.5 % | 1.8 % | 1.2 % |

S/D/I dev-other: init 8,171 / 4,192 / 8,155 -> arm A ep1 9,701 / 14,578 / 6,561 (N 177,275): +10.4k deletions,
+1.5k substitutions, -1.6k insertions. Phones most hit: AH, D, IH, T, CH, HH, R, N, DH (D deletion 2-4 % -> 30-36 %
by ep6); vowels/diphthongs and stops carry most of the growth. Gold repeat pairs X X are merged 19.5 % -> 41.8 % ->
57.0 % (dev-clean; 25.6 -> 45.9 -> 55.6 dev-other). Hyp phones/s: gold 9.63, init 9.85, ep1 9.20, ep6 8.89. The
degradation is population-wide, not a tail: 81 % of utterances worse at ep1, 91 % at ep6. One artifact: CH
insertions grow by 1.4-1.6k (an absorbing symbol). Arm B ep3 (alpha 0.5) is at the init pattern in every bin; arm C
matches arm A (0.1916 / 0.2280).

Reading (provisional until (b) and the audit): the objective's preference is against segments of 1-3 frames — the
2- and 3-frame bins, which d_min = 2 allows, are hit almost as hard as the 1-frame bin, so this is not only the
structural floor but the segmental model's pricing of short segments (duration distribution p_phi(d | k), the
per-segment cost, and the CTC-side requirement of a blank between repeated phones, which the merge statistic shows
being traded away). Next reads: p_phi(d | k) of the warm-up, arm A ep1/ep6 and arm B ep3 phi against the MFA gold
duration histogram per phone; frame-level target vs gold and the network-free fixed-point iteration (read (b)).

### Read (a2): the reverse model's duration distribution is still uniform at arm start (2026-09-15,
`reports/impl_reverse_duration_check_2026-09-15.md`; full run `reports/exec_reverse_duration_check_2026-09-15.md` pending)

p_phi(d | k) is one free categorical per type (dur_logits [40, 50], masked to [2, D_k], log-softmax; 945 free
parameters; initialised at zero = uniform). After the phi warm-up (`htxT2f9FHvWw`: 1 sub-epoch = 56 Adam steps,
phi lr 3e-3, theta frozen) on dev-other gold segments, gold-weighted totals: gold mean duration 4.68 frames,
gold P(d <= 3) 0.471 | model E[d] 13.44 frames, P(d = 2) 0.051, P(d = 3) 0.052, P(d <= 3) 0.103 | log p(d = 2)
-3.00, log p(d = 6) -3.02 | KL(gold || model) 1.03 nats. The uniform over 2..25 has mean 13.5 and log p = -3.18:
the warm-up moved the duration model by a few hundredths of a nat. Arm E ep8 phi is bit-identical to the warm-up
(lam_tau 0, no gradient), which confirms the E control. Gold mass at d < 2 (impossible under d_min = 2): 5.1 %.

Reading: every segment pays a flat ~3 nat duration cost independent of its length, i.e. the duration model is a
pure per-segment penalty; together with the bigram's ~2-3 nats per phone, a phone costs ~5-6 nats of per-segment
constants against 2-3 frames of evidence when it is short. That is the short-segment deletion mechanism of read (a)
in numbers, and it says the arms started from an essentially UNFITTED reverse model (the emission tables p(z | k),
20k parameters after the same 56 steps, are checked next). Design defect, not a knob: 56 SGD steps cannot fit a
categorical duration model; phi needs an expected-count (EM) fit or an initialisation from unit-run statistics
before theta is unfrozen. Full run (`reports/exec_reverse_duration_check_2026-09-15.md`): the duration model DOES move during the arms, slowly and toward the gold — model E[d] 13.4 (warm-up) -> 12.5 (arm A ep1) -> 8.5 (arm A ep6); 11.1 (arm B ep3); 7.6 (arm D ep8, the held-anchor arm, closest to the gold 4.7), P(d <= 3) 0.10 -> 0.25 (A ep6) / 0.34 (D ep8); no explicit per-segment constant exists in the lattice, only log p(d | k) and beta log P_psi enter the emit weight. So phi learns in the right direction at lr 3e-3 but needs thousands of steps, and theta, unfrozen at step 0, drifts long before phi is fitted; arm D shows phi fitting while theta is held. The read of the drift is therefore
from the full run.
Emission tables (same script, item 6; `reports/impl_reverse_duration_check_2026-09-15.md`): unlike the durations, the
emission rows p(z | k) ARE fitted after the warm-up — per-type entropy 3.25 nats vs log K 6.22, KL from uniform 2.96,
p_max 0.156, 42 of 500 units above 1/K per row on average, pairwise symmetric KL between types mean 10.3 (min 0.69 CH-JH,
max 22.2 AW-S); eta shifts the entropy by at most 0.09. phi did move in the warm-up (dur_logits max deviation from the
zero init 0.24, arm A ep1 a further 0.22, all in dur_logits) but the only fitter is the Adam per-frame likelihood step
(`reverse.py:519`); no count-based or EM fit exists in the code, so the categorical duration model stays near uniform
for the ~60 warm-up steps while the 20k emission parameters, which see every frame, do fit. Consequence for the next
stage: the duration model needs a count-based initialisation (hard-EM from the seed recognizer's argmax runs, label-
free) rather than more warm-up steps.

### Prior order: cost estimate (2026-09-15, `reports/estimate_prior_order_2026-09-15.md` + .full.md; profile at the run shape B 125 / T_pad 704)

Measured base step 1.648 s, 6.43 GiB peak; the DP is 89 % of the step, 37 % of it history-dependent, ~15 % of HBM peak.
Held-out perplexity of the banked prior text: bigram 14.23, trigram 9.47, 4-gram ~8.5 (legacy split), so a trigram
buys ~0.59 of the ~0.74 bits/phone on offer. The band is orthogonal to the history axis and CTC already carries the
last symbol, so order 2 is free and order 3 multiplies the history-dependent part by 41 with no reuse. Options costed:
naive trigram (|h| 1681) 14x step, ~70 GiB, dead; trigram with matmul h-reduction plus forward-table checkpointing
(D3 + D4) 1.5x, ~12 GiB, ~1000 lines; bigram DP plus 4-gram importance rescoring 1.3x but degenerate weights (~30
nats/utt); argmax-context tilt 1.1x but a self-confirming prior; class trigram h = (class(p_-2), p_-1) with C = 8
(|h| 328) 3.3x, ~30 GiB, ~500 lines on today's code path, table by marginalising the banked trigram counts. Decision:
(1) free CPU read of the class-trigram held-out perplexity at C = 4..16 against the trigram and 4-gram on the same
held lines (running, `analysis/prior_order_ppl.py`); (2) build the general history axis once (`h' = f(h, k)`, |h| a
knob; bigram path bit-identical; running, `reports/impl_prior_history_axis_2026-09-15.md`), operating point chosen
by (1); the full trigram becomes a knob change once D3 + D4 land. The efficiency read (step time, peak memory at the
run shape) is part of the implementation report and precedes any launch.

Held-out perplexity read (2026-09-15, `analysis/prior_order_ppl.py`, `reports/impl_prior_order_ppl_2026-09-15.md` + .full.md;
banked text and split of `config_sae_4a_s1a_v1.py:67-78`, 1 M counted / 10 k held lines, 912,142 held phones, Witten-Bell
as in prior.py; bigram 14.2314 and trigram 9.4689 reproduce the banked values exactly, class C = 40 equals the trigram):
order 2 14.23 (3.831 bits/phone), order 3 9.47 (3.243), order 4 7.03 (2.815); class trigram h = (class(p_-2), p_-1)
with classes from the training text by exchange clustering: C = 4 11.58, C = 6 11.03, C = 8 10.67, C = 10 10.43,
C = 12 10.23, C = 16 9.94; the hand manner/place partition with 8 classes 11.62 (worse than the text-derived C = 4).
Share of the bigram-to-trigram gain kept: C = 8 70.7 %, C = 16 88 %; the trigram itself holds only 57.8 % of the
bigram-to-4-gram headroom (1.017 bits/phone), not the ~80 % the estimate assumed from the legacy 4-gram figure.
Top-1 prediction flips vs the bigram: trigram 49 %, C = 8 46 % of held phones. Reading: the class trigram at C = 8
(|h| = 9 x 41 = 369 with the BOS class, step ~4x by the estimate's cost model) is the first affordable operating
point on today's code path; the full trigram is worth the D3 + D4 work (1.5x) and is the intended steady state;
a 4-gram cannot sit in the DP (|h| = 41^3) and stays in the decoder.

History axis landed (2026-09-15, speech-llm commit 9edd836, `reports/impl_prior_history_axis_2026-09-15.md` + .full.md): the lattice's
LM history is a knob (`PriorHistory`: bigram |h| = 41, class_trigram (class(p_-2), p_-1) with the BOS class separate so C = 8 gives
|h| = 369, trigram 41 x 41); only emit arcs advance the history, blank and repeat arcs do not, SIL is an ordinary symbol (the bigram
rule, in the docstring); the class table marginalises trigram COUNTS per class and re-runs the same Witten-Bell against the bigram.
Bit-identity: all 20 banked pre-change digests reproduce; brute force matches class_trigram C = 2 and trigram (< 2e-4); posteriors
sum to one; 136 tests pass; census of `config/sae_4a_phase.py` unchanged (1021 jobs, every class count and decode id as banked).
Efficiency read, one GH200, B = 125, T_pad = 704: bigram 1.48 s/step, 5.84 GiB; class trigram |h| = 369 11.65 s/step, 19.8 GiB —
memory under the estimate, step 7.9x the bigram (estimate said 3.6x): the DP cost is ~linear in |h|, the "37 % history-dependent"
split of the estimate was wrong. Consequence: the class trigram is launchable in absolute terms (a 10 h sub-epoch of ~58 steps is
~11 min, 8 sub-epochs ~1.5 h) but is the launch-bound Python loop paying 8x; the matmul reduction over the history (D3) plus
forward-table checkpointing (D4) is now on the critical path before any S2d launch (`reports/impl_lattice_d3d4_2026-09-15.md`,
dispatched), and the full trigram is reachable only through it. Open items: the banked `prior.npz` carries no raw counts, so the
class table needs a small new count job over the same banked text (not a re-run of the finished prior job); the peak-memory
estimator and MAX_BATCH_FRAMES remain bigram fits.
D3 + D4 landed (speech-llm commit bb2f1cb; `reports/impl_lattice_d3d4_2026-09-15.md`): log-matmul reduction over
history, band and group (D3) plus forward-table checkpointing with stride S (D4). GH200, B = 125, T_pad = 704,
s/step | peak GiB: bigram elementwise 1.50 | 5.87 (unchanged, `reduction="auto"` keeps the elementwise path at the
bigram and all 20 banked bigram digests reproduce exactly); class trigram |h| = 369: 11.74 | 19.64 -> 3.40 | 16.84
with D3; full trigram |h| = 1681 with D3 + D4 (S = 32): 7.89 | 9.55. The trigram is 5.3x the bigram step, not the
~1.5x target, and memory no longer binds; the profile is 7 % GEMM, ~60 % shift/exp/log/cat glue, 19 % launch-queue
stall, so the next factor is kernel fusion, not algorithm. matmul vs elementwise agree to 1.3e-4 abs (rel 4.8e-6) on
GPU; D4 bit-identical across S; 122 emc tests pass; census 1021 and the 34 decode-job ids unchanged. Open: the S to
train with (only S = 32 benched at B = 125).
Follow-up commit 9b59812 (`reports/impl_lattice_knobs_2026-09-15.md`, after the code review
`reports/review_lattice_d3d4_2026-09-15.md` and the row-sum diagnosis below): `_logmm` accumulates in float64 (fp32
interface kept; an fp32 production-width row-sum test at |h| = 369, log Z = -172, fails before and passes after);
backward GEMMs and the D4 recompute run under the same fp32/no-autocast guard; `lattice_reduction` / `lattice_checkpoint`
reach the DP from the model definition with a guard that refuses a non-bigram history without checkpointing; unknown
config keys now raise. Suite 142 passed; census 1021 and the 34 decode-job ids unchanged. Cost after the fix at B = 125,
T_pad = 704: class trigram 4.33 s/step, 16.84 GiB; full trigram (S = 32) 10.31 s/step, 9.90 GiB — 6.9x the bigram
step. Not yet settable from a config (build_emc_train_config lacks the hash-neutral-by-omission lines); S unpinned.

**Two-pass lattice (user proposal 2026-09-15; design review `reports/design_two_pass_lattice_2026-09-15.md`,
PROCEED_WITH_CHANGES as a 4-gram vehicle only, STOP as "the S2d lattice").** Proposal: pass 1 a frame-synchronous
exact DP with the n-gram prior and a single-clock acoustic term, pruned by arc posterior to ~100 states/frame; pass 2
the banded semi-Markov DP exact within that lattice, its marginals the training target. Review findings: (1) pass 1
differs from the full weight in four factors, not one — the duration table (per-segment log-ratios of order 10 nats
over the segments inside the band, not absorbed by a beam of 5-10), d_min (superset, harmless), the band itself
(`lattice.py:39-63`: W is the offset between the recognizer's frame clock t and the reverse model's segment clock s,
so a single-clock pass scores z at the wrong frame), and the emission term (`reverse.py:255-266`: nu_phi is
conditioned on duration bucket, position bucket and eta; no per-frame 40 x 500 table exists), so the capture condition
does not hold as stated; (2) pruning is neutral on the drift (the neighbour merges of read (a) sit at high posterior
and survive any beam) and adds a peakiness ratchet; (3) the acceptance checks are re-specified in the review
(per-utterance median <= 0.05 nats, p99 <= 0.5; beam = inf must reproduce the exact log Z to 1e-4 at bigram and
trigram; paired k = 1, 2, 10 with bootstrap CI); (4) a pruned backoff-FST history breaks the group property D3 relies
on, so at trigram the exact D3 + D4 single pass dominates; at 4-gram both routes need a pruned LM, and the single pass
is then exact for the pruned LM with no beam bias — first check the held-out ppl of the 4-gram pruned to 500-4000
contexts (CPU); (5) the trigram's k = 1 value is 0.0026 PER without a CI, so fund neither lattice until read (b3)
sizes the anchored one-step design. Decision: two-pass not funded now; trigram inside the objective goes through the
exact D3 + D4 pass; two-pass revisited only for the 4-gram. Cheap next read: paired bootstrap of trigram-vs-bigram
k = 1 PER on the banked 300 utterances.
Literature on the pruning question (`reports/lit_lattice_pruning_2026-09-15.md`): lattice MMI generates denominator
lattices once under a unigram LM at 10-25 arcs/frame and regenerates at most once (Vesely et al. 2013); frames whose
truth falls outside the lattice get outsized gradients and cause the over-training that motivated frame rejection
(same paper); a posterior TARGET has an interior beam optimum (Manohar et al. 2018: best-path 54 % -> beam 4 62-64 %
-> worse at 8), so "~100 states/frame" is 4-10x the depth that worked, not a safe margin; seed-model bias in a
generated target is fixed at the source (a deletion penalty applied only when generating supervision, -13 % rel WER,
Fainberg et al. 2019) — relevant to the deletion pattern of read (a). LF-MMI keeps the sum exact with a 24k-state
denominator (trigram + 2000 selected 4-gram histories, Povey et al. 2016) and forward-backward under 20 % of the
step; with our 25-slot duration axis that state space is ~5.5 M arcs/frame, about 8x the graph that already took
85 % of the step in Michel et al. 2019 (agent arithmetic, flagged). Net: the pruned-4-gram single pass is not
affordable with the duration axis either; the 4-gram inside the objective has no funded route today, and the
trigram single pass (D3 + D4) is the operating point.
Pruned-LM state spaces (`reports/impl_prior_pruned_ppl_2026-09-15.md`, `analysis/out/prior_order_ppl.pruned.txt`; same
text and held-out split as the banked ppl read, 912,142 held-out tokens; top-N histories kept as explicit states, the
rest scored from the longest kept suffix; a kept row is a proper Witten-Bell distribution): trigram N = 1000 -> 1042
states, ppl 9.49 (unpruned 1566 seen states, 9.47); 4-gram N = 1000 -> 2042 states, ppl 8.42; N = 2000 -> 3566
states, 7.94; N = 4000 -> 5566 states, 7.50; unpruned 36,575 states, 7.03. So a 4-gram at ~2000 states already beats
the full trigram by 1 ppl point at 1.3x its state count. Caveat (design review Q4): a pruned backoff state space has
no group structure in its successor map (the next state depends on whether (h, k) is kept), so the D3 matmul
reduction does not apply and the history axis costs linear in |h| again (7.9x bigram at 369 states); the pruned
4-gram inside the exact single pass is therefore priced at ~40x the bigram step until a successor-structured kernel
exists. Not funded; recorded as the cheapest 4-gram route if the objective is ever worth it.

### Read (b): frame-level target vs gold and the network-free fixed point (2026-09-15, `analysis/emc_target_vs_gold.py`,
`reports/exec_target_vs_gold_2026-09-15.md` + .full.md; 14 GPU runs, dev-other 18,660 gold segments; audited CONFIRMED_WITH_CAVEATS, `reports/audit_fixed_point_2026-09-15.md`)

Fixed-point iteration q_{k+1} = target(q_k) with the network removed, starting from the seed recognizer's dev-other
posteriors (arm A step 0, phi = warm-up), PER by k: 0 0.1100, 1 0.0962, 2 0.1048, 3 0.1179, 4 0.1290, 5 0.1400, 6 0.1489,
7 0.1580, 8 0.1680, 9 0.1791, 10 0.1898; frame accuracy 0.645 -> 0.647 (k = 1) -> 0.600 (k = 10); total variation from
q_0 grows monotonically to 0.30. From arm A ep6: PER 0.2226 -> 0.2178 (k = 1) -> 0.2439 (k = 10). ONE application of
the target improves the recognizer (-0.014 PER at step 0); repeated application walks away from the truth monotonically
with no plateau in 10 steps. This is the pre-registered signature of "the objective's optimum is away from the truth
independently of training dynamics", and it matches the training curve (arm A ep1 0.174 sits between k = 6 and 7).
Frame-level pattern at arm A ep6 (gold segments by length, share of frames kept / taken by another phone / blank):
1 frame 0.49 / 0.46 / 0.05, 2 frames 0.66 / 0.31 / 0.03, 3 frames 0.80 / 0.19 / 0.01, 4-5 0.89 / 0.10 / 0.00, 6+ 0.92 /
0.08 / 0.00 (step 0: all 0.91 / 0.08 / 0.01). Short gold segments are absorbed by a NEIGHBOURING phone, not turned into
blank: the sequence-level deletions of read (a) are merges. Single-component ablations at step 0 (uniform prior,
uniform reverse model, arm-B tilt) move the step-0 target by < 0.003 in every statistic — at tau = 2 the one-step target
is dominated by q_theta, so which component of R drives the fixed point cannot be read from one step; the fixed-point
iteration under each ablation (and under a count-fitted duration model) is the next read, queued below.
Decision (rule above, branch 1): model-side remedy, new stage S2d, not a knob sweep. Pre-registered launch check for
S2d, costing one analysis run and no training: under the S2d model (higher-order prior, count-initialised duration
model, d_min = 2), the network-free fixed point from the seed recognizer must satisfy PER(k = 10) <= PER(k = 0) on
dev-other; a model that fails this is not launched. Note the diagnostic uses gold for the READ only; no training or
selection touches it.
Audit (fresh context): every figure re-derived from the outputs; the iterated quantity is the real training target
(same lattice call, tau 2, prior weight 1, W 25, bigram prior TRPE0D5nF3bh, eta U4etvcSpsQi4, phi = warm-up
htxT2f9FHvWw ep1), per-frame renormalisation <= 0.7 %, PER via the GreedyPerJob collapse against GoldPhonesJob
.ZGSp0hxyd2YP on a 300-of-2864 stride subset (k = 0 0.1100 vs 0.1157 on the full split, k-deltas paired on the same
utterances), no gold in the DP. Caveats to carry: the k = 1 gain is an insertion trim (I 819 -> 483, D 421 -> 497,
phone rate 9.90 -> 9.69 vs gold 9.69), frame accuracy +0.002 only, no per-utterance interval, and the earlier target
diagnostic at another checkpoint on dev-clean had the one-step target 0.0024 WORSE, so the sign of the one-step
effect is checkpoint-specific; the iteration freezes phi and eta (training moves phi at lr 3e-3), drops lam_agg and
train-mode BN, runs on dev-other not the 10 h split, and k = 10 is not converged (TV 0.30, rising). The read
evidences drift under repeated exact fitting, not the location of the objective's optimum; the S2d launch check is
worded accordingly (PER(k = 10) <= PER(k = 0), a no-drift requirement).

### Read (b2): the fixed point under each remedy (2026-09-15, SLURM 1811809, `reports/exec_fixed_point_ablations_2026-09-15.md`
+ .full.md; same subset, band and seed as read (b); lattice at commit 9edd836 for the trigram line)

PER by k = 0, 1, 2, 5, 10 (un-ablated: 0.1100, 0.0962, 0.1048, 0.1400, 0.1898):
uniform prior 0.1100, 0.1052, 0.1211, 0.1720, 0.2399 (drift faster: the prior is holding the target back, not driving it);
uniform reverse model 0.1100, 0.1007, 0.1010, 0.1641, 0.5842 (diverges: the reverse model's acoustic evidence is what ties the
target to the audio); duration table from the seed recognizer's argmax runs (label-free, E[d] 4.0 vs 12.7 uniform) 0.1100, 0.0961,
0.1017, 0.1308, 0.1705; duration table from the gold histogram (diagnostic, E[d] 5.3) 0.1100, 0.0966, 0.1015, 0.1310, 0.1720 —
the count-based init is as good as gold durations; argmax durations + uniform prior 0.2096 at k = 10; full trigram prior
0.1100, 0.0936, 0.0992, 0.1251, 0.1641 (best one-step target and slowest drift, still monotone from k = 2).
Reading against the pre-registered launch check (PER(k = 10) <= PER(k = 0) = 0.1100): NO single remedy passes, and the two useful
ones (trigram, fitted durations) each remove only a quarter of the drift. The drift is not a defect of one component: under
q_{k+1} ∝ sqrt(q_k R) the stationary point is R's own posterior, so the recognizer's information decays geometrically whatever R is,
and the fixed point can only be as good as decoding with the prior and the 500-unit categorical reverse model alone. What the
objective does offer is the ONE-STEP product of experts: seed 0.1100 -> 0.0936 with the trigram (-0.016 PER on this subset), which is
the gain the anchored arms realise partially (B ep1 -0.010 PER / -1.0 WER at 10 h; D held at alpha 0.25 -0.012 PER / -2.0 WER at
1 h). Queued (`reports/impl_fixed_point_anchor_2026-09-15.md`): the anchored fixed points at alpha = 1.0 / 0.5 / 0.25 under trigram +
fitted durations, the combined un-anchored line to k = 30, and the current objective to k = 30 (where the drift ends). Decision
pending on those: S2d as an anchored one-step refinement (bounded gain, ~1-2 WER absolute at these operating points) vs a
redesign of the reverse model's acoustic side (the only lever that moves the fixed point itself) vs closing the mechanism.

### Read (b3): the anchored fixed point and the K = 30 limits (2026-09-15, SLURM 1813551 / 1813554, `reports/exec_fixed_point_anchor_2026-09-15.md`; audit pending)

Same bed as (b)/(b2): arm A step 0 (seed theta + warm-up phi), dev-other 300-utt stride subset, W = 25, tau = 2, greedy
PER vs gold, network removed. "TRI" = prior_trigram + dur_from_argmax_runs. Anchor alpha = the arm-B/D tilt exactly as
the recipe applies it (log q_k + alpha log q_init into the same lattice call, q_init = this checkpoint's own recognizer).

| line | k=0 | k=1 | k=2 | k=5 | k=10 | k=20 | k=30 |
|---|---|---|---|---|---|---|---|
| (e) un-ablated, no anchor | 0.1100 | 0.0962 | 0.1048 | 0.1400 | 0.1898 | 0.2421 | 0.2573 |
| (a) TRI, no anchor (rerun on the fixed lattice, SLURM 1817267, rows [0.9968, 1.0014]) | 0.1100 | 0.0936 | 0.0964 | 0.1166 | 0.1491 | 0.1890 | 0.2015 |
| (b) TRI, alpha 1.0 | 0.1100 | 0.0981 | 0.0959 | 0.0949 | 0.0944 | | |
| (c) TRI, alpha 0.5 | 0.1100 | 0.0961 | 0.0939 | 0.0949 | 0.0952 | | |
| (d) TRI, alpha 0.25 | 0.1100 | 0.0947 | 0.0937 | 0.0991 | 0.1018 | | |

Reading. (1) The un-anchored iteration does NOT plateau at the reverse model's posterior: both K = 30 lines rise
monotonically past k = 1 and are still rising at k = 30 (0.257 un-ablated, 0.201 with trigram + durations on the fixed lattice; the first run of that line, 0.243, was on the broken matmul path and is kept as *.broken_rowsum). The
"fixed point = R's posterior, ceiling ~0.16-0.19" reading given after (b2) was wrong: the iterate is the per-frame
FACTORISED projection of the target's marginals, not the path distribution, so the limit is not R's posterior and no
finite ceiling is visible within 30 iterations. The trigram + durations shift the curve by ~2 iterations, they do not
change its shape. (2) The anchor tilt holds: alpha >= 0.5 keeps the iterate at 0.094-0.095 through k = 10 (a held
-0.015 PER vs the seed), alpha 0.25 lets it drift (0.102 at k = 10). The held anchor is a product of experts with a
fixed q_init, i.e. it requires a seed recognizer and is therefore NOT a cold-start mechanism. (3) For the user's
cold-start direction this makes the drift the central defect of the objective as it stands, ahead of the take-off
problem: from any start the recognizer's information decays and nothing in the current model (prior order, duration
table) stops it. Consequential; auditor dispatched with the five jsons, the gold PER read and the criterion "does the
un-anchored curve plateau by k = 30".
Audit (`reports/audit_fixed_point_anchor_2026-09-15.md`, CONFIRMED_WITH_CAVEATS, re-derived from the jsons): no plateau
or turn-down by k = 30 on either un-anchored line (strictly increasing at every k; decelerating, so a later plateau is
not excluded — K ~ 60 would be needed to claim a limit); alpha 1.0 and 0.5 hold within 0.0015 of their k = 2 value
through k = 10, alpha 0.25 breaches at k = 5; un-ablated K = 30 reproduces the earlier K = 10 run to 6 d.p.; iterate =
`post_q` per-frame arc marginal renormalised (a mean-field loop), q_init = q_0 exactly, no gold in the loop. CAVEAT
that suspends the anchor rows: in all four runs that combine prior_trigram with dur_from_argmax_runs (line (a) and
the three anchored lines) the post_q row sums are in [3.3e-9, 3891.9] against the script's own rows-sum-to-1
invariant (un-ablated and single-ablation runs: [0.995, 1.002]); argmax reads may survive a per-row scale but float32
rows at 3e-9 risk underflow. Debugger dispatched on the combined-ablation path; the anchored lines are being rerun
on the un-ablated (clean) model, which is the read that matters for the anchor claim. Also noted: alpha = 1 at
tau = 2 pins the iterate to q_0 by construction (TV 0.015 vs 0.474 un-anchored), and the arm-B schedule has alpha = 0
by this epoch, so the training arms never ran a held anchor at these values except arm D (0.25).
Anchor rerun on the un-ablated (bigram, fitted-duration) model, SLURM 1816641
(`reports/exec_fixed_point_anchor_unablated_2026-09-15.md`; row sums [0.9971, 1.0020], 0 NaN frames — the invariant
holds here, so these rows replace (b)-(d) for the anchor claim): alpha 1.0: k = 1 0.1005, k = 2 0.0994, k = 5 0.0982,
k = 10 0.0981 (held); alpha 0.5: 0.0994, 0.0980, 0.1000, 0.1015 (slow drift, still below k = 0); alpha 0.25: 0.0984,
0.0992, 0.1081, 0.1116 (above the seed by k = 6). Un-anchored reference k = 10 0.1898. Reading: on the clean model
only alpha = 1 holds through k = 10, and the held gain is -0.012 PER on this subset (vs -0.015 for the one-step
target); alpha = 0.5 buys most of the hold at k <= 5. Whether the combined trigram + duration model holds better at
alpha 0.5 (as lines (b)-(d) suggested) is unknown until the row-sum defect is fixed and those lines are rerun.
Row-sum defect resolved (`reports/debug_post_q_rowsum_2026-09-15.md`): the cause is not the ablations but the D3
log-matmul path of commit bb2f1cb — `_logmm` (lattice.py:627-631) shifts per operand and floors fp32 underflow at
a_max + b_max - 87.3, RAISING those entries; forward and backward inflate unequally, so rows come out as
exp(flow_t - log Z). Every non-bigram history runs through that path since the commit; the single-ablation trigram run
was clean only because it finished before the edit, on the elementwise path. CPU reproduction at |h| = 1681: fp32
matmul row sums [0.836, 1.045], elementwise / fp64 / exact log-sum-exp all 1.000. Consequence: the PER of lines (a)-(d)
of read (b3) must be RERUN after the fix (the defect is per entry, so the argmax changes; k = 0 and the un-ablated
K = 30 curve survive); the D3 code review's fp64 agreement check (7e-15) could not see it. Fix (float64 accumulation
inside `_logmm` + an fp32 non-bigram row-sum test) is with the implementer together with the review's two changes;
blocks any order-3 line, including the trigram lines of reads (c1)-(c4).

### Literature on the deletion mechanism (2026-09-15, `reports/lit_length_bias_2026-09-15.md`; full texts read)

- CORRECTION of a design citation: ESPUM (Yeh et al. ICLR 2019) trains against N = 5 (top-10k 5-grams, App. B),
  not "unigram + bigram"; no LM-order ablation exists there, and its segment-level formulation (one output per
  segment) cannot delete at all. The "bigram first, supported by ESPUM" line in Design decisions is void; the bigram
  was a compute choice only.
- Empirical-ODM (Liu et al. NeurIPS 2017): the mode-seeking cross-entropy form "easily converges to predicting the
  output with largest p_LM"; the coverage-seeking (corpus-frequency) form fixes it; 1/2/3-gram error 71.8 / 10.9 /
  10.2 %. Implication: the order gain from 2 to 3 was modest there; the form of the objective mattered more.
- wav2vec-U (Baevski et al. 2021): no LM in the GAN loss, 4-gram only in decoding; label-free rate control via
  silence insertion in text (0.25), a diversity entropy term, and a decode blank bonus tuned in [-3, 8].
  wav2vec-U 2.0 (Liu et al. 2022): the UNIT RATE is the constant — 16 Hz segments (gold ~10 phones/s) PER 19.0;
  25-28 Hz PER > 100 (Tab. 1, dev-other greedy). Our reverse clock is 50 Hz.
- Merialdo 1994: EM from a good init 97.0 -> 96.8 (it. 1) -> 95.2 (it. 10); with > 5k sentences the first
  iteration already hurts; an init anchor cut the added errors 818 -> 419, freezing marginals only 767 -> 712.
  Johnson 2007: sparse Dirichlet priors help; transition priors and annealing are nulls.
- Tang et al. 2017 (segmental): segment score = frame AVERAGE of log-posteriors + duration weight + bias (eqs. 6,
  10, 11) — per-frame normalisation of the evidence is the standard segmental remedy (the weights there are
  supervised). Kamper et al. 2017: a 250 ms minimum duration cut Xitsonga WER 116.2 -> 78.9.
- Not read: Ondel, Glarner 2018, Ebbers 2017, Wang 2018.

Implication for the next stage: (b) per-frame normalisation of the segment evidence is the cheapest label-free
remedy for the per-segment-constant imbalance; (a) a label-free insertion bonus calibrated to the text phone rate is
the second lever and the tilt stays; (c) a higher LM order is required by the user (standing constraint) — the
literature predicts a modest gain from order alone under a mode-seeking objective, so it is carried together with
(a)/(b), and the coverage-seeking form of the text term is the literature's own fix for LM-mode collapse.

### Literature on cold-start collapse (2026-09-15, `reports/lit_cold_start_collapse_2026-09-15.md`)

Bearing on the user's direction (cold start is the target, S3 CLOSED FAIL is the problem). (1) wav2vec-U 2.0 names the
S3 failure ("satisfy the criterion by producing the most common n-grams regardless of input") and fixes it with a
FORWARD per-frame cross-entropy from an intermediate recognizer layer onto MFCC k-means codes with K = 64: dev-other
greedy PER 15.9 -> 13.6 over 8 seeds; K = 128 and a 320 x 2 VQ target were worse than no term (Liu et al. 2022, SLT).
Our K = 500 generative reverse model sits on the wrong side of all three axes Liu measured (direction, granularity,
K). (2) The LM score anti-correlates with PER once the model can game it: REBORN's PPL-only reward gave the best PPL
and the worst PER (10.0 / 14.7) vs 11.2 / 12.9 with edit-distance and length trust-region terms anchored to the
previous transcript (Tseng et al. 2024, NeurIPS) — never gate on the objective or prior score alone; gate on phone
rate, vocabulary usage and the derangement margin (the G4a.3 clauses were right). (3) Trainability is a phase
transition in the 4-gram JSD between audio-side and text-side phone n-grams (threshold ~0.27; beyond it PER 70-95;
Lin et al. 2022, ICASSP). (4) Rate/silence machinery is a convergence precondition: without clustering / PCA /
mean-pooling 0 % of seeds converge vs 100 %; without a SIL symbol a phone is repurposed as silence (Baevski et al.
2021, NeurIPS, Table 8). (5) The only executed exact-marginalisation unsupervised mapping (WFST Baum-Welch + n-gram)
needed a supervised multilingual phone recognizer as the acoustic side, an LM-order curriculum bigram -> 5-gram ->
word trigram, 50 restarts picked by train likelihood, and failed outright on 2 of 6 languages (Klejch et al. 2022,
Interspeech). (6) Bayesian acoustic-unit discovery never learns durations from flat: presegmentation (Lee & Glass
2012) or a 3-state minimum-duration topology plus a cross-lingual prior (Ondel et al. 2019, PER 65.4 -> 49.2). (7) No
published non-GAN cold-start system reaches comparable PER (EURO, Gao et al. 2023, is a GAN re-implementation).
Implication for S3b: flat-start exact marginalisation has never taken off without an informative acoustic side; the
first remedy to test is a forward per-frame content term (low K, intermediate layer), with take-off read on phone
rate, vocabulary usage and the derangement margin, never on L_tau. Reads (c1)-(c3) size the ceiling and the collapse
network-free before any of this is built.

## S3 cold start: G4a.3 read (2026-09-15; audited CONFIRMED FAIL, see the audit note above)

Run `ReturnnTrainingJob.sBlPYBA1YcIQ` (flat init `FlatRecognizerInitJob.21Kxgr5JLR3k`, 8 sub-epochs, tau 8 / 5.04 /
3.17 / 2 / 2 / 2 / 2 / 2, ~123 s per sub-epoch; extractor report `reports/extract_s3_reads_2026-09-15.md`). Dev L_tau
1.74 (sub-ep 1) -> 1.21 (sub-ep 3) -> 1.21 (sub-ep 8); reverse term per frame -7.22 -> -3.32. Dev-other greedy PER:
0.8955 at sub-epoch 4 (gate read), best 0.8387 at sub-epoch 5. Emitted phone rate per sub-epoch 3.71, 1.98, 0.97,
1.58, 4.75, 6.87, 4.29, 7.73 per s: every sub-epoch is outside [0.6, 1.5] x 9.8/s = [5.9, 14.7] (the rate rule
fires on all eight; nothing to revert to). Speaker-matched derangement gap at sub-epoch 4 on dev-other
(`S3DerangementGapJob.Ez8bGMB7LnPX`): gap_per_frame -0.3218, ci95 [-0.6446, -0.0170], n_selected 314/500,
identical_donor 12 — negative with the CI excluding 0: the held-out reverse model scores a deranged donor's decode
ABOVE the utterance's own. AMENDMENT (code review 2026-09-15, `reports/review_bt_probe_2026-09-15.md`): the gap job
paired `reverse.evaluate()` rows with tags by position, but evaluate returns rows in length-bucket order (each row
carries its item index). gap_per_frame -0.3218, gap_macro_mean, n_selected and the drop counters STAND (sums and
complete-set means are permutation-invariant); the ci95 [-0.6446, -0.0170] and every per_utterance.json row FALL.
The sign conclusion (negative gap) stands; "CI excludes 0" is undecided from this job. G4a.3 FAIL is unaffected (the
PER clause fails on its own). Fix (index-keyed pairing, hash-neutral run()-body change) queued before any S3b gate
read; the BT probe already pairs by index. G4a.3 as pre-registered (dev-other PER < 0.50 AND positive gap at
sub-epoch 4): FAIL on both clauses. Reading (pending audit): from a flat start the objective is optimised (L_tau falls 1.74 -> 1.21) by a
recognizer whose output carries no utterance-specific content (PER ~ 0.9, negative gap, phone rate collapsing to
1-2 / s during the anneal); the cold-start bootstrap does not take off on this bed. Per the gate this licenses "not
funding S3 further", not "it could not work" (a single schedule and seed were run).

## S3b: cold-start remedies (registered 2026-09-15, user decisions)

**Cold EM reads c2/c3, first read** (bigram, 300 dev-other utts, `analysis/out/cold.c2_flat_fitphi.bigram.dev-other.txt`,
`cold.c3_flat_fitphi_pindur.bigram.dev-other.txt`; executor report and audit pending): from the flat theta the
fixed point is all blank at every k = 1..30 — blank share 0.994, emitted phones 0.000 / s, 0 distinct phones, no
feasible item for any phi refit. Pinning the duration table (c3, gamma mean 5 frames) changes nothing. Reading: the
collapse is a blank collapse, nothing in the objective prices an empty output (d_min / D_sil price segments that
exist). **c1 (flat theta, warm-up phi HELD, executor `reports/exec_cold_fixed_point_bigram_2026-09-15.full.md`,
SLURM 1816775, banked check PASS 0.1100 / 0.0962):** PER 1.000 (k 0) -> 0.978 (k 5) -> 0.431 (k 10) -> 0.293
(k 20) -> 0.279 (k 30), phones / s 0 -> 8.27. Reading (first read, audit with the full set): under a phi that
carries content, the objective pulls a FLAT recognizer out of the blank solution network-free to 0.279, above
the seed's own limit (0.190) but far from content-free — take-off from flat is possible once phi carries content,
so the cold problem sits on phi's side (from cold, phi has no feasible item to fit and stays uninformative). This
is what the P-BT warm-phi arms and the S3b-C consistency arms bear on. Trigram (SLURM 1817476, executor
`reports/exec_cold_fixed_point_trigram_2026-09-15.md`): c1 PER 0.367 (k 10) -> 0.248 (k 20) -> 0.238 (k 30), 8.7
phones / s, 39 distinct phones, E[d] 5.1; c2 stays blank through k 10 and reaches only 0.985 / 0.16 phones / s /
1 distinct phone at k 20 (the trigram prior alone starts to price the empty output, but at a content-free
solution); c3 pinned durations: same as bigram. c4 (seed theta, phi refitted) running in both chains.

**c4, seeded theta with phi refit (bigram SLURM 1816775, 2:44 h; trigram chain 1817476; `analysis/out/cold.c4_seed_fitphi.{bigram,trigram}.dev-other.txt`; `reports/exec_cold_fixed_point_bigram_2026-09-15.md`).** theta = the seeded recognizer (dev-other greedy PER 0.1100 at k = 0, 9.9 ph/s), phi refit every iteration. PER by k (0/1/5/10): bigram 0.1100 / 0.0962 / 0.1178 / 0.1380; trigram 0.1100 / 0.0936 / 0.1064 / 0.1218; banked frozen-phi bigram read 0.1100 / 0.0962 / 0.1400 / 0.1898. Phone rate drifts 9.9 -> 9.1 (bigram) / 9.3 (trigram) per s, blank share 0.31 -> 0.35 / 0.34; every k feasible for the refit; drift still monotone at k = 10 (no fixed point reached). Reading: from a content-carrying start the objective's fixed point drifts away from the seed slowly, the refit of phi roughly halves the drift of the frozen-phi read, and the trigram prior halves it again; the drift direction is toward fewer emissions (rate down, blank up), the same direction as the cold collapse, only much slower. The cold-read set c1-c5 is complete; audit dispatched.

**Step profile (SLURM 1818385, 2026-09-15; `analysis/emc_profile_step.py`, `analysis/out/emc_profile_step.{b32,b128,b256}.txt`).** One S3 EMC step at the config's own shape (max_seqs 128, 88k padded frames, pad/real 1.002), theta+phi from sBlPYBA1YcIQ epoch 4, batches pre-loaded (no loader wait), 30 timed steps after 5 warm-up, SM utilisation from nvidia-smi dmon against an idle control (0.4 %). b128: 1671 ms/step (p10-p90 1669-1677), 76.0 utt/s, 52k real frames/s; stages: train_step forward 91.0 % (the lattice DP alone 1474 ms), loss+backward 8.8 %, optimizer 0.1 %; SM 97.8 % mean over the steps, 99.9 % during the DP; 86.6k CUDA kernels per step (mean 17.5 us; 82.9k of them in the DP, elementwise 34 % + reduce 27 % of kernel time), launch-bound ratio 0.91 (device busy, not CPU-launch-bound); peak 6.5 GiB allocated / 10.4 GiB reserved. b32: 1358 ms/step, 22.8 utt/s, SM 41 % (launch-bound below ~128). b256: 3092 ms/step, 81.8 utt/s, SM 99 % (saturated: doubling B doubles the time). Conclusions: one GH200 is fully used at the S3 shape by a DP that is a chain of tiny kernels; larger batches and the rate term's stacked tilt cannot raise per-GPU throughput; the only per-GPU win is a fused DP step (queued); the cluster-side waste is the exclusive-node flag (3 of 4 GPUs idle per arm), fixed by packing 4 arms per node. The rate term's finite difference costs 3 DP calls per step (~4.6 s/step, 28 utt/s) whichever batching is chosen; `rate_fd_mode="forward"` (2 calls) is the cheap option if the arms must be shortened.

**c5, rate tilt on the cold fixed point (SLURM 1818371, 2026-09-15, bigram chain; `analysis/out/cold.c5a_flat_fitphi_ratetilt.bigram.dev-other.txt`, `cold.c5b_flat_fitphi_pindur_ratetilt.bigram.dev-other.txt`).** c2 and c3 rerun with a per-token tilt b on the non-SIL segment table, solved every iteration so that the EXPECTED non-SIL emission rate equals the label-free rho (held within 1 %, b between -5 and +0.6, 1-9 DP passes). Result: the expectation constraint is met by a DIFFUSE posterior, not by emissions. At every k = 1..10 in both c5a and c5b the per-frame argmax stays blank (blank share 0.99, greedy ph/s 0.000, 0 distinct phones, PER 1.0), E[N] under the posterior is 10.5-13.5 segments/s, and no decode is feasible for a phi refit (300/300 items rejected, refit skipped at every k). Reading: with theta flat, an expected-count term has a trivial fixed point in which every frame keeps ~0.19 of its mass on non-blank symbols spread over the 39 phones and the argmax never leaves blank; the term prices the expectation, not the mode. Implication for S3b-R (informative, no gate): the training arms can satisfy L_rate this way, so the gate rightly reads the GREEDY phone rate (rate_in_band off the posterior would pass); and the design review is asked whether the rate term needs a sharpness companion (the tau anneal already in the schedule, or a per-frame entropy / mode-count variant) before it can move the argmax. c5 with the trigram chain not run (chain ended before the tilt existed); queued behind the profile.

**Audit of the cold-read set c1-c5 (2026-09-15, `reports/audit_cold_reads_2026-09-15.md`, DONE_WITH_CONCERNS).** All numbers reproduce from the json. Amendments: (i) the c1 "warm phi" htxT2f9FHvWw is fit on decodes of recognizer 65NNK8Bwxdtd ep024, which trains on PhoneTargetHdfJob.DXDTg3VoP47A <- SeedGoldPhonesJob.zii9E9tvr51e (alias init_oracle_quarantined): c1 is a GOLD-DERIVED CEILING, and so is the c4 seed; "the cold problem sits on phi's side" stands as a decomposition (a content-carrying phi lets a flat theta take off) but licenses no label-free route, and the P-BT warm-phi arms (phi htxT2f9FHvWw) inherit this: they are reporting-only oracle references, never a funded or selected arm. (ii) c2 trigram is not literally all-blank: from k = 14 it emits 0.04-0.16 ph/s of ONE phone type with 2 kept items, c3 trigram up to 3 types; the collapse reading is unchanged (PER 0.985, distinct 1-3). (iii) Every cold read ran at tau = 2 while the S3 run collapsed during its anneal tau 8.0 / 5.04 / 3.17 / 2.0 (config:21): the regime S3 actually collapsed in is unprobed; a tau = 8 pass of c2 DONE (SLURM 1818817, 18 min, `analysis/out/cold.c2_flat_fitphi_tau8.{bigram,trigram}.dev-other.txt`, `reports/exec_cold_c2_tau8_2026-09-15.md`): identical to tau = 2 -- all blank at every k (blank share 1.000, 0 phones/s, PER 1.000, argmax agreement 100 %) in both chains at k = 1, 2, 5, 30 on the same 300 dev-other utterances; the flat start's fixed point does not depend on the temperature, so the S3 anneal is not what makes the null, the objective is. (iv) c5 is bigram-only, K = 10, and a tilt is not a trained rate loss; no spread or paired deltas on any PER (single 300-utterance stride subset).

**S3b-R, rate term (user's form).** L = L_tau + lam_rate * ((E_q[N]/T - rho)/rho)^2 per utterance, mean over kept
utterances; N = expected EMITTED non-SIL tokens under the tempered posterior (sum of seg_post over non-SIL types),
T = frames. rho is LABEL-FREE: phones per word of the phonemized text T_phi times a disclosed read-speech constant
2.7 words / s, divided by 50 Hz; the dev-transcript rate 9.8 / 9.4 per s (SAE_1f.md:533-536, gold) never enters the
loss and is reporting-only. Gradient to theta only (the user's E_{q_theta}), through a central finite difference in
a per-token tilt b added to the non-SIL segment table (d post_q / d b), two extra DP passes per step; tiny-shape
autograd oracle. Reads: (c5) network-free — the cold harness with b solved per iteration so E[N]/T = rho (the
penalty's fixed point), durations free and pinned: does the rate-constrained fixed point carry content (PER, distinct
phones, reverse LL; informative, not a gate — S3b-R is funded by the user)? Training arms lam_rate in {1, 3, 10}, S3
schedule, seed and reads otherwise unchanged (`config/sae_4a_s3b_rate.py`), after the design review.
Gate **G4a.3b-R** (pre-registered): in any arm at sub-epoch 4, dev-other greedy PER < 0.50 AND speaker-matched
derangement gap > 0 with the CI excluding 0 AND emitted phone rate within [0.6, 1.5] x rho. PASS: the cold start
takes off with the rate term; continue on the cold track (content term and trigram as ablations). FAIL: the rate
term is a precondition, not the take-off mechanism; the forward per-frame content term (Liu 2022) is the next lever
and carries the rate term. Expected in advance: S3 sub-epochs 5-8 already emitted 4-8 phones / s at PER 0.84-0.90
with a negative gap, so a non-zero rate alone may still be content-free.
Code review 2026-09-15 (`reports/review_rate_term_2026-09-15.md`, PASS_WITH_CONCERNS; commits dddbb56 + 629d5ae): loss and
selection label-free, gradient reaches theta through the tilt, SIL excluded, census 98/98 and 1021/1021 unmoved, resolved
config differs from S3 by the four rate keys only. Reading rules fixed by the review: (i) the code's ((r - rho)/rho)^2 equals
the user's (r - rho)^2 with lam scaled by 1/rho_per_frame^2 = 26.8, so lam {1, 3, 10} = {0.037, 0.112, 0.37} in the
user's units; (ii) DecodeStatsJob's rate_band uses the gold 9.8 / s (pre-existing job, emc_train_jobs.py:201, hash-bound):
the gate band [0.6, 1.5] x rho = [5.80, 14.49] / s is read off `phone_rate`, never off `rate_in_band`; (iii) at max_seqs 128
the stacked tilt call exceeds the lattice caps, so every arm runs 3 DP calls per step (monitor emc/rate_dp_calls = 2 expected);
raise caps or halve max_seqs after the step profile, not before.

**S3b-C, consistency regularization (user 2026-09-15: speed perturbation and SpecAugment at least).** Term:
L_cons = mean over frames of KL(p_clean(t) || p_aug(t)), p = the recognizer's untempered per-frame posterior,
p_clean stop-gradient (teacher = clean view), two views: (a) SpecAugment on the L15 features (time masks and
channel masks, no length change, frame-wise KL); (b) a speed-perturbed copy of the utterance (Kaldi convention
speed0.9 / speed1.1, one kind per utterance by its tag, `sae/perturb.py`, features from a perturbed
AvStatesJob -> L15FeatureHdfJob dump of tc100, a second extern_data key), the perturbed posterior linearly
time-warped to the clean frame count before the KL. Why speed: the private code §3g measured was DURATION
(corr 0.856 with the frame count, `SAE_3G.md:221-230`); a speed view moves the frame count 10 % and leaves the
content, so a duration code cannot explain both views. Caveat pre-registered: a constant or all-blank output is
perfectly consistent, so L_cons never runs alone — it runs WITH the rate term (forbids empty output) and L_agg
(forbids a constant phone distribution). Arms (fixed in advance, not tuned on the rate arms' results): lam_rate = 3
(the middle rung), lam_cons = 1, views {SpecAugment; SpecAugment + speed}; S3 schedule, seed and reads otherwise
unchanged. Gate **G4a.3b-C** = the G4a.3b-R clauses. Prerequisite: the perturbed tc100 dump (label-free, units not
needed for the perturbed view). Code after the rate-term implementer releases sae_emc.py.

**Packed node read, sub-epoch 4 (2026-09-15, PackedEmcTrainJob.SeZzGUScxq4x, SLURM 1819395, dev-other, `output/.../sae_4a_s3b_pack/<arm>/ep<k>/dev-other/`).** Greedy PER (S+D+I)/N at sub-epochs 1-4 and emitted rate per s: lam1 0.945 / 0.933 / 0.992 / **0.829** at 0.79 / 0.88 / 0.10 / 6.29; lam10 0.855 / 0.875 / 0.879 / **0.914** at 3.99 / 2.82 / 2.48 / 9.08; spec 0.971 / 0.974 / 0.964 / **0.848** at 2.10 / 1.66 / 1.88 / 8.35; spec_speed 0.935 / 0.996 / 0.979 / **0.849** at 2.19 / 0.05 / 0.30 / 8.44. lam3 (single-arm run) for comparison 0.847 at 6.95. All four arms: rate clause in band [5.80, 14.49] at sub-epoch 4, PER clause FAIL (no arm below 0.50; all inside or above the content-free band), so **G4a.3b-R FAIL in every rate arm (lam 1, 3, 10)** and **G4a.3b-C FAIL in both consistency arms**, as pre-registered for S3b-C ("no take-off expected"). Pattern shared by all five arms: the mode collapses during the anneal (sub-epochs 2-3 near 0-3 per s) and reappears at tau = 2 with content-free output; lam10 emits above rho with 18k insertions (PER 0.914), lam1 gives the lowest PER of the campaign (0.829) but still content-free. Speaker-matched derangement gap at sub-epoch 4 (per frame, 500 utterances): lam1 +1.658, lam10 +2.279, spec +1.281, spec_speed +1.637 (lam3 +1.910): positive in every arm at content-free PER, so the gap clause measures theta/phi self-consistency on a private code, not phone content; it cannot separate the arms. Sub-epochs 5-8 (PER; rate per s): lam1 0.831 / 0.841 / 0.855 / 0.858 (5.9 / 6.9 / 4.6 / 8.3), lam10 0.883 / 0.881 / 0.879 / 0.898 (7.5 / 7.5 / 6.6 / 8.6), spec 0.850 / 0.848 / 0.864 / 0.865 (8.2 / 7.8 / 8.2 / 8.4), spec_speed 0.859 / 0.847 / 0.875 / 0.874 (8.3 / 7.9 / 8.8 / 8.7); every arm stays in the content-free band with the rate in band; label-free selection picked sub-epochs 8 / 4 / 6 / 7. The node finished 8 sub-epochs for all four arms in one 4-GPU job (packing works; nothing crashed). Sub-epoch 8 gaps per frame: lam1 +3.245, lam10 +2.983, spec +1.886, spec_speed +2.673 (lam3 +3.289): larger than at sub-epoch 4 in every arm while PER did not improve; the gap tracks the sharpening of the private code. **S3b-R and S3b-C CLOSED FAIL on G4a.3b-R / G4a.3b-C** (all five arms, sub-epochs 4 and 8, PER clause; graph complete). Hypothesis inspection at sub-epoch 4 (user question; `reports/extract_hyp_inspect_2026-09-15.md`, `analysis/out/emc_hyp_inspect.<arm>.ep4.dev-other.txt`, stored greedy decodes, counts re-verified against per.json): the content-free output is NOT a relabeled phone code (best bijective relabeling gains at most 0.003 PER: lam1 0.829 -> 0.827, lam3 0.847 -> 0.846, lam10 0.914 -> 0.910), NOT a repeat pathology (runs >= 3 are 0.1-0.3 % of tokens, max run 4-5, like gold), and length-faithful (hyp/gold length r 0.89-0.97); the rate arms spread over the inventory nearly as evenly as gold (entropy 4.3-4.5 vs 4.8 bits), the consistency arms collapse onto 3-4 symbols (spec: K/IH/T = 57 % of tokens, gold IY and EH never produced). Reading: the private code is a per-utterance, length-faithful symbol sequence without stable phone identity; the positive gap measures exactly that. Chance-level null (same script, section 7; 5 draws, seed 0, draw sd <= 0.0006): random strings at the hyp length drawn from each arm's own unigram distribution score PER lam1 0.840, lam3 0.850, spec 0.859, spec_speed 0.867, lam10 0.922 against the arms' 0.829 / 0.847 / 0.848 / 0.849 / 0.914, i.e. every arm beats its own length-and-unigram-matched chance floor by 0.003-0.019 PER; uniform-symbol strings score 0.89-0.97; the Hungarian relabeling lifts the null by at most 0.0012, so the arms' relabeling gain is chance-scale; the null drawn at gold length scores 0.888-0.900, and lam10 (0.914) is WORSE than that null. Reading: the 0.83-0.91 PER band IS chance for a length-faithful string with the arm's symbol frequencies; no phone content is present.

**Design review of S3b (2026-09-15, `reports/design_review_s3b_2026-09-15.md`, DONE_WITH_CONCERNS) and the decisions taken.**
F1 (decisive): the S3 run's own monitor `dev_loss_emc_phone_rate` (expected emitted tokens/s under the posterior) reads 21.6 / 20.4 / 16.9 / 8.9 / 9.0 / 9.2 / 8.9 / 9.3 at sub-epochs 1-8 while the greedy rate was 3.71 / 1.98 / 0.97 / 1.58 / 4.75 / 6.87 / 4.29 / 7.73: the c5 diffuse fixed point is already what training does. At the gate sub-epoch the expectation sat at 8.9/s (rho 9.66) with the mode at 1.58/s; during the anneal E[N] is ~2x rho, so the symmetric rate term pushes emissions DOWN at tau 8-3 and is nearly inactive at sub-epoch 4. Pre-registered prediction for the lam3 arm: expected_phone_rate_hz pinned near rho by sub-epoch 2-3, greedy phone_rate < 0.6 x rho at sub-epoch 4, PER 0.85-0.95; lam 1/3/10 indistinguishable. Decisions: (1) LAUNCH lam3 (DF6blPpto23t) ALONE as the falsifier (read at sub-epoch 1 and 4); hold lam1 / lam10 and both consistency arms until its read. (2) If F1 confirmed: replace lam1 / lam10 by lam3 + a mode-pricing companion, per-frame entropy penalty on the untempered q ramped 0 -> lam_ent over sub-epochs 1-4 (no DP cost), one-sided hinge max(0, rho - r)^2 as the disclosed deviation from the user's form; build queued behind the consistency round-2 fixes (same file). If refuted (greedy rate in band at sub-epoch 4): launch the grid as specified, packed. (3) F2: no S3b arm carries a LABEL-FREE content-carrying acoustic side, the one thing c1 shows take-off needs (c1/c4/warm-phi are oracle-derived, audit (b)); the forward per-frame content term (Liu 2022: cross-entropy from an intermediate recognizer layer onto MFCC k-means codes, K = 64) is built NOW in parallel, not left to the FAIL branch. (4) Consistency: the "never alone" guard is void (the rate term forbids only an empty EXPECTED output, and L_cons ~ 0 for diffuse or content-free-but-non-constant output since both views keep speaker and duration); lam_cons 1 is not small (KL 0.12 / 0.39 vs L_tau 1.7-2.2); arms launch only after the lam3 read, with "no take-off expected" pre-registered. (5) Probe: the pipeline transfers content only from an oracle phi (first three arms read by the reviewer: warm_phi/collage PER 0.363 -> 0.314 with gap +2.9 -> +3.6, warm/centroid 0.695 -> 1.058, cold phi_0/collage 0.867-0.886 with rate 3.7 -> 2.7/s and gap ~ 0); the any-round "takes off" reading stays as pre-registered and a FINAL-round reading is added as a secondary, both reported; the full S3b-BT design must carry a label-free content term, mode rate control, collage rendering, an explicit trigram statement, a GPU-hour budget rule, the full dev-other read and 2 phi seeds; no differentiable path from the recognizer's loss into phi (EMC's soft-EM gradient is the principled coupling). (6) Gates G4a.3b-R/C measure the right thing (greedy rate off phone_rate, PER < 0.50 above the 0.84-0.90 content-free band), decidable (~36 min per arm), n = 1 seed disclosed. (7) Sequencing: lam3 now; probe finishes; then one packed node; BT full only with the content term.

lam3 sub-epoch 1 read (`reports/extract_rate_lam3_2026-09-15.md`, DF6blPpto23t, dev-other): expected rate emc/expected_phone_rate_hz 10.91/s (rho 9.66, inside the band at the first read; the term acts), monitor phone_rate 15.86 tokens/s, greedy emitted rate 4.66/s (below 0.6 rho = 5.80), greedy PER 0.868 (content-free band), FD check 4.4e-5, 1.67 DP calls/step. Consistent with F1 (expectation priced, mode blank); not a gate read, decision at sub-epoch 4.

lam3 sub-epoch 4 read (DF6blPpto23t, dev-other, `output/.../sae_4a_s3b_rate/lam3/ep{1..4}/dev-other/{per.txt,phone_rate}` and work/learning_rates). Sub-epochs 1-5 expected rate (emc/expected_phone_rate_hz) 10.91 / 11.05 / 9.83 / 9.28 / 9.26 per s; monitor phone_rate 15.86 / 15.63 / 13.26 / 9.95 / 9.88; greedy emitted rate (sub-epochs 1-4) 4.66 / 3.03 / 0.29 / 6.95 per s; greedy PER 0.868 / 0.859 / 0.980 / 0.847 (S+D+I)/N; D 91.8k / 121.6k / 171.9k / 53.2k of N 177.3k. **F1 decision (pre-registered rule): REFUTED on the rule** -- greedy rate 6.95/s at sub-epoch 4 is inside the gate band [5.80, 14.49] (S3 had 1.58/s at the same point), so the term does move the mode by the gate sub-epoch. Caveat recorded, not a rule change: during the anneal the mode still collapsed (0.29/s at sub-epoch 3 with the expectation at 9.8/s), i.e. the expectation/mode split of c5 and F1 is real inside the anneal and only closes at tau = 2; PER 0.847 stays in the content-free band 0.84-0.90, as expected without a content side (F2). G4a.3b-R for lam3 at sub-epoch 4: PER clause FAIL (0.847 vs < 0.50); gap clause PASS: speaker-matched derangement gap +1.910 per frame (macro +1.730, ci95 [1.606, 1.861], 500 utterances, 33 speakers, 500 distinct decoded strings, live phi, own decodes; `.../lam3/ep4/dev-other/derangement_gap.json`), the first positive gap of the campaign (S3 read -0.322 at the same point). Audit (`reports/audit_lam3_gate_2026-09-15.md`): gate NOT met (PER 0.847 recomputed from per.json against GoldPhonesJob.ZGSp0hxyd2YP; rate clause passes); gap CONFIRMED (+1.910 / macro +1.730 recomputed from per_utterance.json, 488/500 positive, index-keyed pairing with the assert, same checkpoint epoch 4 for phi and theta, frames identical on both sides by construction, exact-token-match subset n 123 +1.48 and donor-longer subset n 170 +1.59, so neither a length nor an emission-count artifact); caveat: the S3 baseline -0.322 was scored on 314/500 utterances with different phi units, so the sign flip is not a paired contrast; "in any arm" cannot be closed because lam1/lam10 have not run. Reading: the decodes are utterance-specific and phi predicts the units from them, but they are not phones (PER in the content-free band, one distinct string per utterance) -- the private-code symmetry the S3b-BT paragraph names, now measured. lam3 sub-epochs 5-8 (dev-other): greedy rate 6.70 / 6.66 / 5.47 / 8.50 per s, PER 0.850 / 0.853 / 0.857 / 0.886, expected rate 9.26 / 9.32 / 9.29 / 9.68; dev-clean PER 0.879 at sub-epoch 8; label-free selection (weighted LM perplexity) picked sub-epoch 4; sub-epoch 8 gap +3.289 per frame (macro +3.184, ci95 [3.044, 3.330], 500 utterances, 500 distinct strings) while PER rose to 0.886: the gap grows as the private code sharpens, decoupled from phone content. G4a.3b-R for lam3: FAIL on the PER clause at sub-epochs 4 and 8; the arm and its graph finished (SLURM 1818726, manager exited). Consequence per the rule: launch the grid as specified, packed: lam1, lam10, spec (fw5O1L5qL2HL), spec_speed (be8qKc7aRj16) on one node (pack config being wired, `reports/impl_pack_config_2026-09-15.md`); the entropy companion arms (lam3_ent_a / lam3_ent_b, `reports/impl_entropy_companion_2026-09-15.md`) are built but HELD, since the rule did not select them. Consistency round-2 code review PASS_WITH_CONCERNS (`reports/review_consistency_r2_2026-09-15.md`: BN guard, KL and per-key batch verified; census-script "before tree" flag is a no-op, numbers re-verified independently; speed-band docstring wording), compute may be spent.

**P-BT, back-translation probe** (user 2026-09-15: full training funded regardless; a content-free probe does not
reject the idea, it shapes the full design). Mechanism argument under test: phi is fit on the recognizer's decodes
of real audio; from cold those are content-free, so phi ignores its text input and a recognizer trained on
(phi-synthesized units, real text) learns the text prior only. The probe: label-free 2000-utterance train subset of
tc100 (features, units, eta), 2000 real sentences of T_phi; rounds r = 0..3 of (decode real audio with theta_r ->
fit phi on (real units, pseudo-text) -> sample units from phi for the real sentences with a real speaker eta ->
map units to L15 features -> train theta by CTC on (synthetic features, real text)). Eight arms: phi start
{cold phi_0 (S3's constructed init), warm-up phi `htxT2f9FHvWw` (seed-track, diagnostic only: does the loop transfer
content from a content-carrying generator at all)} x units-to-features {collage of real frames per unit, per-unit
centroid (the cluster mean), speaker-matched run-level collage, learned label-free unit-to-feature renderer (unit
context + eta -> L15, L2 on real frames; a continuous decoder in Hori 2019's sense but fit on OBSERVED units, so
immune to the cold problem; added on the user's question 2026-09-15 — a direct phone-to-feature regressor was not
taken because from cold it is fit on pseudo-labels and regresses to the global mean)}. Reads per round on the 300 dev-other bed: greedy PER, emitted phones / s, distinct phones, distinct
strings, speaker-matched derangement gap under the round's phi; train-side feasible fraction and pseudo-text
distinct strings. Pre-registered reading: "takes off" = dev-other PER < 0.80 AND phone rate in [0.6, 1.5] x rho AND
gap > 0 (CI excluding 0) in any round; otherwise content-free. Not a funding gate. If the warm-phi arms are also
content-free, the synthesis pipeline (not the cold start) is the suspect and the full design must fix that first.

**P-BT result (2026-09-15, all eight BtProbeJob arms finished, `reports/extract_bt_probe_2026-09-15.md`; arms 1E6mfTahsKLZ, 3BFG7tATDxqI, 5owmxhkPXkdf, 91VDNBrLup2B, 96K3LFZ7brhJ, rNJbMa2sG9ia, uXVX60ASzhKI, ZXlljMDUea5m; rho 9.6619/s, every read on 300 dev-other utterances).** Pre-registered any-round "takes off" (PER < 0.80 AND rate in band AND gap > 0) and the secondary final-round reading, per arm (best PER): warm_phi/collage YES / YES (0.314); warm_phi/collage_spk_run YES / YES (0.334); warm_phi/centroid YES / no (0.695); warm_phi/renderer no / no (1.371); phi_0/collage no / no (0.867); phi_0/collage_spk_run no / no (0.897); phi_0/centroid no / no (1.506); phi_0/renderer no / no (1.246). Cold collage arm by round 0-3: PER 0.867 / 0.871 / 0.886 / 0.875, emitted 3.70 / 3.21 / 2.56 / 2.68 per s (never in band), gap -0.000 / +0.006 / -0.020 / -0.031 (round 3 CI [-0.056, -0.006]), train feasible fraction 0 / 0.93 / 0.95 / 0.93, CTC loss on synthetic data 3.30 -> 0.31 (the recognizer learns the synthetic data, which carries no content). Warm collage arm: round 0 PER 0.363 at 7.74/s with gap +2.74 [2.59, 2.90], improving to 0.314. Reading: the cold loop is content-free at every round exactly as the mechanism argument predicted (phi fit on content-free decodes ignores its text input; theta then learns the text prior only), and the loop transfers content only from a phi that already carries it; the warm phi is oracle-derived (audit (b)), so the warm arms are ceilings, not a route. Renderings: collage > collage_spk_run > centroid > learned renderer in both phi conditions; centroid and the learned renderer end above PER 1.0 (worse than an empty hypothesis), so the continuous renderings are dropped from the full design. Per the user's rule the failed probe does not reject the idea; per the design review the full S3b-BT design carries a label-free content term (being built) and mode rate control before it is launched. USER DIRECTION (2026-09-15, after the probe): a randomly initialised reverse model not helping is expected and says nothing about BT in general; BT is to be integrated into the current cycle-consistency (EMC) training as an AUXILIARY term, never used to train a recognizer directly. Consequence for the S3b-BT design: no standalone decode -> fit phi -> render -> CTC rounds; instead the EMC-trained reverse model phi (the one in the lattice, updated jointly) synthesizes units for real text from T_phi during EMC training, and a BT loss on theta for those (synthesized units, text) pairs is trained INTERLEAVED with the EMC batches (user 2026-09-15: the two are different forwards on different data, so not a summed loss in one graph): SEPARATE optimizer steps alternating (user 2026-09-15: no gradient accumulation either): an EMC step on a real-feature batch through the lattice, then a BT step on a batch of (units from the live phi under no_grad for real T_phi sentences, collage-rendered from real frames; text as target) with its loss weighted by lam_bt (ramped, label-free), each step its own update. Phi receives no gradient from the BT pass; BatchNorm running statistics are frozen on the synthetic batch as in the consistency term. Alongside the rate term and the content term; the S3b-BT paragraph below is superseded on this point.
Literature for the auxiliary form (`reports/lit_bt_auxiliary_2026-09-15.md`): BT-as-auxiliary exists only in semi-supervised ASR, never inside a zero-label objective, and always with a separate frozen text-to-speech side (our jointly trained phi is outside every measured regime). What it fixes in the design: (1) interleaving is mandatory: a BT-only stream gives 28.3 WER against 25.2 baseline and 23.6 interleaved with real audio (Hayashi 2018, LibriSpeech 100 h paired + 360 h text); (2) synthetic pairs used as plain CE targets destroy the recognizer (25.2 -> 89.3 with the 1-best, even at weight 0.1; Hori 2019), the guard is scoring or attenuation, not the weight; (3) the largest effect in the field is attenuating the synthetic branch's acoustic context (EAT, Baskar 2021: 24.1 -> 15.9 test-other; alpha 0 degenerates to a pure LM), so the sweep axis is the attenuation, not lam_bt; (4) sampling vs argmax of the content is a null (Tjandra 2019); argmax the units, randomise the nuisance (duration, speaker); (5) nobody tuned the weight (0.5 / 1 / w2v-U 2.0's 0.3-0.5); (6) the only clean main-vs-main+aux read is 16.8 -> 16.6 WER, below the w2v-U 2.0 seed spread of 0.9-2.2 PER over 8 runs, so the read needs >= 4 seeds and a label-free selection rule (LM NLL + vocabulary usage, w2v-U). Design consequences: lam_bt in {0, 0.1, 0.3} pre-registered, attenuation swept, argmax units with sampled durations and speaker, interleaved 1:1 with EMC batches, >= 4 seeds per cell, gate on the paired dev-other read.

**S3b-BT, full back-translation training**: funded by the user; designed after the probe and the literature read
(`reports/lit_backtranslation_cold_start_2026-09-15.md`), reviewed by the design-reviewer before its first job.
Gate form fixed now: the G4a.3 clauses (dev-other PER < 0.50 AND positive gap) at a matched budget of 8 tc100
sub-epoch equivalents; the operating point and any init are named in the design.
Literature (2026-09-15, `reports/lit_backtranslation_cold_start_2026-09-15.md`): this IS tried-and-failed evidence,
not absence. Unsupervised MT: back translation alone with a good cross-lingual init but no denoising autoencoder
gives 0.0 / 0.0 BLEU against 25.1 / 24.2 with it (Lample et al. EMNLP 2018, Table 4); without the pretrained init
8.8 / 9.2 against 27.5 / 28.1 (Lample et al. ICLR 2018, Table 4). Artetxe 2018 states the mechanism: BT without
denoising lets the model ignore its input and learn a target-side LM. ASR<->TTS dual transformation (Ren et al.
ICML 2019) is a threshold: 100 paired utterances -> PER 64.2, 200 -> 11.7; zero paired was not tried. Liu 2020 and
Chen 2019 are not cold-start back translation. Implication for S3b-BT (amended after the user's objection: the
ingredient does not port literally, our directions share no encoder, decoder or latent space): the MT denoising
autoencoder works through SHARED parameters that place both sides in one latent space and stop the decoder ignoring
its input; the full design must supply that FUNCTION in our architecture, the design review decides the form.
Native analogs: (i) the shared unit codebook plus collage synthesis (their shared-BPE init alone: 0.0 -> 10.5 BLEU),
already in the probe; (ii) a real-versus-synthetic consistency term on theta (theta on an utterance's real features
against theta on the collage rebuilt from its own corrupted units), candidate only, closes the synthetic-to-real gap
and forces theta to read its input but does not by itself break the private-code symmetry; (iii) the rate term.
Back translation alone from the flat init is predicted to stay at the all-blank null; the probe's cold arms test
that prediction, its warm-phi arms test the pipeline.

**S3b-BT-aux, back translation as an auxiliary inside EMC** (spec, 2026-09-15; replaces the standalone S3b-BT loop per the
user direction above; built after the content-term wiring, reviewed before its first job). Training = the lam3 EMC arm
(rate term lam 3, tau anneal, 8 tc100 sub-epochs) with, after every EMC optimizer step, ONE separate BT optimizer
step: a batch of 128 sentences from T_phi (the unpaired text corpus phi already uses; never the audio's transcripts),
converted to phones, units taken by ARGMAX from the live phi (no_grad, phi gets no gradient) with durations and the
speaker / frame-pool draw SAMPLED (literature: argmax the content, randomise the nuisance), rendered by collage from
real L15 frames (the probe's best rendering; centroid and the learned renderer dropped), then the recognizer with
BatchNorm running stats frozen and CTC on the phone sequence, loss weight lam_bt ramped 0 -> lam_bt over sub-epochs
1-4 (BT is content-free while phi is; the ramp keeps it small until EMC has moved phi), held from sub-epoch 5.
Attenuation axis (the field's largest lever, EAT): how deep the BT gradient reaches, two levels: full depth, and
output projection only (hidden detached in the BT step). Pre-registered arms (one packed node, n = 1 seed disclosed):
bt_a lam_bt 0.1 full; bt_b lam_bt 0.3 full; bt_c lam_bt 0.3 output-only; control = the running lam3 arm
(DF6blPpto23t, lam_bt 0, same schedule); fourth slot lam3_ct (content term, lam_content 0.3) so the node is full.
Cost: the BT step is one CTC forward-backward per EMC step (no lattice DP), expected < 15 % wall-time on the profile.
Gate **G4a.3b-BT** (pre-registered, unchanged clauses): at sub-epoch 4 and 8 on dev-other, greedy PER < 0.50 AND
positive speaker-matched derangement gap, read PAIRED against the lam3 control on the same utterances (clustered
bootstrap), label-free selection only; monitors emc/bt_ctc, emc/bt_synth_phones_per_s (AMENDED 2026-09-16: the code emits this name, the plan said bt_units_per_s), emc/bt_distinct_units, emc/lam_bt_effective, emc/bt_real_frac (share of synthetic frames drawn from a real pooled frame; the frame pool is a process-local rolling ring buffer, so after any resubmit the first BT steps collage the pool mean until every unit is refilled). Prediction
Prior order (design review point 4, stated 2026-09-16): the three BT arms inherit the bigram prior of the lam3 control (prior_history "bigram", the only history the config builder exposes); theta's BT loss is a plain CTC with no LM term (the sampled sentence carries the full text prior), phi's fit is unchanged. This is a bigram EMC launch and conflicts with the standing rule (SAE_4A.md:220, "no further EMC stage launches with prior_order: 2"); the paired read against the bigram lam3 control needs matched priors, and the class-trigram history costs 7.9x the step (line 633) and is not config-settable. USER DECISION 2026-09-16: trigram ("I sincerely do not think bigram would work"); the bigram pack2 node (wK3hCW0JdJ6G) is NOT launched. Amended design: full trigram prior (|h| = 1681) through the exact D3 + D4 pass, S = 32, measured 10.31 s/step and 9.90 GiB at B 125 / T_pad 704 (6.9x the bigram step; line 656); arms lam3_tri (control: lam_rate 3, lam_bt 0, trigram), bt_a_tri (0.1 full), bt_b_tri (0.3 full), bt_c_tri (0.3 output_only), one packed node; lam3_ct moves to a later node. Gate G4a.3b-BT unchanged, read paired against lam3_tri. Expected node time: 0.69 h x 6.9 for the EMC steps plus the BT steps, ask about 8 h. USER ADDITION 2026-09-16: arm bt_b_tri_cr = bt_b_tri plus the consistency term on the recognizer (S3b-C mechanism, commit 1f05ca9: KL(p_clean stopgrad || p_aug), SpecAugment views only, augmented passes under frozen BatchNorm stats, lam_cons 0.3 (user 2026-09-16; the weight-1 S3b-C arms were the ones that collapsed onto 3-4 symbols); the speed view is left out so batching stays identical to bt_b_tri). It needs a second 4-GPU node; to not idle three GPUs that node carries lam3_tri_cr (consistency without BT: the CR control), and a second model seed of bt_b_tri and lam3_tri (bt_b_tri_s2, lam3_tri_s2) for the seed spread the literature demands. Contrasts: bt_b_tri_cr vs bt_b_tri (CR on top of BT), lam3_tri_cr vs lam3_tri (CR alone), seed pairs (spread). Same gate, same reads. (pre-registered): the probe's mechanism says BT cannot create content that EMC does not put into phi; the arms test
whether the coupling ACCELERATES leaving the content-free band (0.84-0.90), not whether it creates it. A gain below
1 PER at n = 1 is noise (w2v-U 2.0 seed spread 0.9-2.2); seeds (>= 4) are funded only for an arm that passes the
gate. Reads: `output/.../sae_4a_s3b_bt/<arm>/ep<k>/<split>/` in the single-arm layout.

**Trigram packed nodes, sub-epoch 4 read (2026-09-16; pack v3 PackedEmcTrainJob.byYMQmBNEpLZ SLURM 1821649 node jpbo-081-27,
pack v4 PackedEmcTrainJob.cgeWUCSGt7xf SLURM 1821650 node jpbo-086-16; greedy PER from `alias/sae/4a/s3b_pack{3,4}_<arm>/ep4/<split>/per`
(GreedyPerJob: per-frame argmax, repeats collapsed, blank and SIL dropped; `decoder_version` is a KenlmPosteriorDecodeJob
knob and does not apply to a greedy number), rate from the sibling `decode_stats`; `reports/extract_ep4_reads_2026-09-16.md`;
AUDITED CONFIRMED_WITH_CAVEATS `reports/audit_trigram_ep4_2026-09-16.md`: independent Levenshtein reproduces lam3_tri and
bt_a_tri exactly, gold = GoldPhonesJob.ZGSp0hxyd2YP in all 16 reads, every checkpoint epoch.004.pt of the right pack job,
trigram prior in every resolved returnn.config, no empty hypotheses).**
Trigram cost confirmed in the run itself: 1734 s per sub-epoch vs 277 s for the bigram lam3 run DF6blPpto23t (6.3x); the
learning_rates monitor is still named `emc_agg_kl_bigram`, a stale label, not the prior.

| arm | PER dev-clean | PER dev-other | phones/s dev-clean / dev-other |
|---|---|---|---|
| lam3_tri (control) | 0.830 | 0.845 | 7.99 / 7.44 |
| bt_a_tri | 0.835 | 0.840 | 7.81 / 7.24 |
| bt_b_tri | 0.845 | 0.857 | 7.35 / 6.91 |
| bt_c_tri | 0.863 | 0.874 | 8.84 / 8.44 |
| lam3_tri_cr | 0.823 | 0.845 | 8.02 / 8.06 |
| bt_b_tri_cr | 0.865 | 0.887 | 9.28 / 9.25 |
| lam3_tri_s2 (seed 43) | 0.827 | 0.844 | 7.70 / 7.44 |
| bt_b_tri_s2 (seed 43) | 0.859 | 0.866 | 7.96 / 7.38 |

Reading (gate G4a.3b-BT, sub-epoch 4 clause): PER clause FAIL in every arm (all 0.84-0.89 on dev-other against the
< 0.50 threshold), i.e. the content-free band of the bigram packed read (0.83-0.91) again; the trigram prior did not move
it at sub-epoch 4. BT vs control (audit correction): bt_a_tri is 0.0057 below lam3_tri on dev-other (paired
speaker-clustered bootstrap ci95 [-0.0112, -0.0004]) but 0.0047 ABOVE it on dev-clean (ci95 [+0.0002, +0.0102]), and both
are under the pre-registered 1-PER-point noise floor (line 1023), so no BT effect is claimed; bt_b / bt_c sit 0.01-0.04
above the control. Caveat: bt_ramp_epochs = 4, so lam_bt reaches full weight only AT this read; sub-epoch 4 decides the
PER clause but is a weak test of BT, sub-epoch 8 carries that question. The consistency arm is unchanged vs its control
on dev-other. Error pattern (`reports/exec_per_error_pattern_trigram_2026-09-16.md`, all 9 arms reproduce banked S/D/I):
NOT the short-segment-deletion signature of read (a); every arm is substitution-dominated (S 52-70 % of N = 177.3k)
with a flat deletion rate across gold-length bins (lam3 bigram 32/32/31/30/27/21 %, lam3_tri 27/27/27/26/23/18 %,
bt_b_tri_cr 13/12/12/12/10/8 % with S+I exploding). Qualitative read of the same 12 aligned utterances across bigram lam3,
lam3_tri, bt_a_tri, bt_b_tri_cr (`reports/qual_errors_trigram_ep4_2026-09-16.md`, model judgement): correct phones are
isolated in every arm (~75 % stand alone, longest correct run 5 in 2864 utts, no syllable recovered); each arm collapses
onto its own favourite phones (top-10 share 63-74 % vs gold 55 %), neither a consistent relabelling nor random. Bigram:
0.72x gold length, consonant strings, V/C alternation 57 % vs gold 73 %. Trigram: 0.77x, alternation 66 %, syllable-shaped
templates (EH R IY, P R IY, K R IY) repeated thousands of times: phonotactics, not words. +BT: same structure, filler
rebuilt from function words (AE N D 3208x, DH AH), fewer distinct trigram types. +consistency: 0.96x length, no dropouts,
wall-to-wall substitution with open loops (DH AH x4; 10 gold phones -> 34). What improves across the arms is the shape
of the output, never phone identity. Trigram-prior score of the hypotheses (`analysis/emc_hyp_inspect.py` block 8, the DP's
own log_tri table via PhoneNgramPrior.per_token_log_probs, two BOS, no EOS, Witten-Bell so no floor; hyp / gold carry no
SIL while the table was fit on SIL-inserted text, so 2.25 nats for real text is a scale reference only;
`reports/exec_hyp_inspect_trigram_2026-09-16.md`, summary `analysis/out/emc_hyp_inspect.summary.ep4.dev-other.txt`),
nats per phone on dev-other, gold -3.22 in every row:

| arm | PER | chance-null PER | Hungarian PER | distinct | H_hyp bits (gold 4.83) | LM hyp | LM chance null | LM shuffled hyp |
|---|---|---|---|---|---|---|---|---|
| lam3 (bigram) | 0.847 | 0.850 | 0.846 | 37 | 4.33 | -5.05 | -6.77 | -6.73 |
| lam1 / lam10 / spec / spec_speed (bigram) | 0.829 / 0.914 / 0.848 / 0.849 | 0.840 / 0.922 / 0.859 / 0.867 | 0.827 / 0.910 / 0.848 / 0.848 | 37 / 39 / 22 / 32 | 4.47 / 4.52 / 3.19 / 3.56 | -5.42 / -5.01 / -3.78 / -4.34 | -7.48 / -6.95 / -6.77 / -7.59 | |
| lam3_tri | 0.845 | 0.863 | 0.845 | 38 | 4.51 | -4.79 | -6.95 | -6.91 |
| lam3_tri_s2 | 0.844 | 0.859 | 0.840 | 36 | 4.23 | -4.42 | -6.74 | -6.67 |
| bt_a_tri / bt_b_tri / bt_c_tri | 0.840 / 0.857 / 0.874 | 0.844 / 0.870 / 0.889 | 0.840 / 0.857 / 0.874 | 38 / 38 / 37 | 4.24 / 4.63 / 4.48 | -4.73 / -4.99 / -4.63 | -6.77 / -7.44 / -7.29 | |
| lam3_tri_cr / bt_b_tri_cr | 0.845 / 0.887 | 0.863 / 0.894 | 0.823 / 0.882 | 34 / 37 | 4.05 / 4.27 | -4.05 / -4.33 | -6.61 / -6.67 | |
| bt_b_tri_s2 | 0.866 | 0.869 | 0.866 | 38 | 4.62 | -5.12 | -7.83 | -7.80 |

Reading: (1) every arm beats its chance null by 0.003-0.02 PER only and a Hungarian relabelling recovers nothing (no
private code); (2) every arm's output is ~2 nats/phone more trigram-plausible than its nulls and ~1.5 nats less than
gold, and the trigram arms score 0.3-0.6 nats better than the bigram lam3 (the "more English-like" of the qualitative
read is real and measurable); (3) LM plausibility is anti-aligned with content across arms: the best LM scores belong to
the consistency arms (-4.05 / -4.33, PER 0.845 / 0.887) and to spec (-3.78 with 22 distinct phones), i.e. the objective
buys prior-likeness of the output, which the trigram makes cheaper to buy, without phone identity; (4) variety does not
rise with the trigram or BT (H_hyp 4.2-4.6 bits vs bigram lam3 4.33, gold 4.83). Dev-loss trajectories to sub-epoch 2 track the bigram reference (dev agg
1.31 -> 1.27 in lam3_tri vs 1.23 -> 1.18 bigram), so the objective again is no content signal. The gap clause is not
read here (gate is AND; sub-epoch 4 fails on PER alone).

**Sub-epoch 8 read (2026-09-16; both pack graphs COMPLETE, SLURM 1821649 / 1821650 COMPLETED, managers exited at end of
graph; `reports/extract_pack3_ep8_2026-09-16.md`; AUDIT RUNNING `reports/audit_trigram_ep8_2026-09-16.md`).** Greedy PER
dev-other at sub-epochs 1-8, then dev-clean at 8 and phones/s at 8:

| arm | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | dev-clean 8 | phones/s 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| lam3_tri | 0.857 | 0.878 | 0.941 | 0.845 | 0.837 | 0.849 | 0.851 | 0.877 | 0.861 | 8.80 |
| bt_a_tri | 0.859 | 0.888 | 0.920 | 0.840 | 0.839 | 0.889 | 0.865 | 0.868 | 0.854 | 7.81 |
| bt_b_tri | 0.845 | 0.912 | 0.907 | 0.857 | 0.855 | 0.871 | 0.854 | 0.869 | 0.862 | 8.19 |
| bt_c_tri | 0.877 | 0.910 | 0.899 | 0.874 | 0.863 | 0.885 | 0.889 | 0.892 | 0.871 | 8.83 |
| lam3_tri_cr | 0.924 | 0.984 | 0.937 | 0.845 | 0.844 | 0.841 | 0.843 | 0.862 | 0.847 | 8.35 |
| bt_b_tri_cr | 0.908 | 0.967 | 0.936 | 0.887 | 0.867 | 0.908 | 0.868 | 0.894 | 0.877 | 8.81 |
| lam3_tri_s2 | 0.864 | 0.865 | 0.938 | 0.844 | 0.836 | 0.852 | 0.856 | 0.873 | 0.859 | 8.75 |
| bt_b_tri_s2 | 0.861 | 0.927 | 0.920 | 0.866 | 0.856 | 0.891 | 0.868 | 0.881 | 0.875 | 8.28 |

(dev-other rows at sub-epochs 1-3, 5-8 from the extract, sub-epoch 4 from the audited read above; no label-free selection
job is registered in the pack3/pack4 graphs.) Dev agg loss of lam3_tri 1.31 / 1.27 / 1.13 / 0.76 / 0.73 / 0.80 at sub-epochs
1-6: the same drop at sub-epoch 4 as the bigram lam3 run (0.71 / 0.65 / 0.72), rate 9.2-9.5 / s after sub-epoch 4.
Reading: **G4a.3b-BT FAIL at sub-epoch 8 in every arm** (0.86-0.89 on dev-other, every arm WORSE than its own sub-epoch 4
or 5); the minimum over all 64 arm x sub-epoch reads is 0.836. At full BT weight (sub-epoch 8) bt_a_tri / bt_b_tri sit
0.009 / 0.008 below lam3_tri and 0.005 / 0.004 below lam3_tri_s2, inside the 1-point floor; bt_c_tri and both _cr arms are
above their controls. **S3b-BT-aux CLOSED FAIL on the cold bed** (AUDITED 2026-09-16 CONFIRMED_WITH_CAVEATS, `reports/audit_trigram_ep8_2026-09-16.md`: all 16 sub-epoch 8 reads re-derive exactly from per.json and trace to epoch.008.pt of the right pack/arm, cold, trigram; caveat: bt_a_tri - lam3_tri = -0.93 pt with speaker-clustered bootstrap CI [-1.65, -0.13], P(delta<0) 0.991, reliably signed but under the pre-registered 1-point floor, and -0.55 [-1.32, +0.32] vs lam3_tri_s2; the matched control for bt_b_tri_cr is lam3_tri_cr, BT worse by 3.16 pt): back translation as an auxiliary does not
move the cold start out of the content-free band with a trigram prior, and the consistency term does not either. What the
trigram bought is recorded above: LM-plausible, syllable-shaped filler. Per the design review (F2, point 7: "BT full only
with the content term") the one S3b remedy not yet run cold is the label-free forward content term, which the trigram
launch deferred ("lam3_ct moves to a later node"); it goes on the next node (S3b-CT below) while S2d runs.

**S3b-CT: content term on the trigram cold bed (registered 2026-09-16, before any result).** One packed node, four arms,
control = the banked lam3_tri above (same node type, same schedule): lam3_ct_tri = lam3_tri + the forward content term
(content_term.py, MFCC k-means K 64 codes, tap content_layer 1, lam_content 0.3, as the held bigram arm
ReturnnTrainingJob.DhsiFoHwBP5l); bt_ct_tri = bt_a_tri + the same content term; lam3_ent_a_tri / lam3_ent_b_tri = the
held entropy companions (COMPANION_ARMS in config_sae_4a_s3b_rate_v1.py: rate hinge + lam_ent 0.1 / 0.3, ramp 4; a paired win there is not attributable to the entropy term alone) on the trigram control. Reads as pack v3. WIRED commit 9af2329 (`reports/impl_pack5_config_2026-09-16.md`; review PASS_WITH_CONCERNS `reports/review_pack5_config_2026-09-16.md`: per-arm diff vs the pack v3 twins is exactly the registered delta, frozen lam3_tri decodes consumed as paths, checkpoint selection keys on dev_loss_l_tau so the content CE never enters it), LAUNCHED 2026-09-16: pack PackedEmcTrainJob.nineWO8G0tFD (rqmt as pack v3), 16 PairedPerDeltaJob rows vs lam3_tri at sub-epochs 4 and 8 on both dev sets; content monitor keys {train,dev}_loss_emc_content_frame_acc / _content_ce in learning_rates.
Gate **G4a.3b-CT**: at sub-epochs 4 and 8 on dev-other, greedy PER < 0.50 (the standing clause) AND, secondary, a paired
delta vs lam3_tri at the same sub-epoch with ci95 excluding 0 in the arm's favour on both dev sets; the content monitor
(emc/content_acc or its registered name) must rise above its chance level 1/K = 0.016 or the term was never active.
A PER inside 0.83-0.90 at both reads closes S3b-CT FAIL and with it the S3b remedy set on the cold bed.

**Alignment-mass profile of the cold trigram control (2026-09-16, first read, AUDITED `reports/audit_mass_profile_lam3_tri_2026-09-16.md`: every number reproduces, see the amendment after the table; user's question "where does
the alignment mass lie and how does it distribute in training"; registered reader `analysis/emc_mass_profile.py`, executor
`reports/exec_mass_profile_lam3_tri_2026-09-16.md` + .full.md, sbatch 1829978; lam3_tri epoch.001-008.pt of pack v3
byYMQmBNEpLZ, 300-utt stride subset of dev-other against the MFA gold frame alignment; `--full-per-check` reproduces the
banked 0.845398 / 0.877185 at sub-epochs 4 / 8 exactly; post_q row sums within 0.6 % of 1, see amendment).** Gold
is used for the read only. Headline trajectory (tau 8 / 5.04 / 3.17 / 2 x5):

| statistic (dev-other, 300 utts) | ep1 | ep2 | ep3 | ep4 | ep6 | ep8 |
|---|---|---|---|---|---|---|
| target mass on the GOLD phone, gold speech frames | 1.0 % | 1.3 % | 1.5 % | 4.9 % | 4.5 % | 5.2 % |
| target mass on another phone, gold speech frames | 28.9 | 30.2 | 30.2 | 54.8 | 57.3 | 67.1 |
| target mass on blank / SIL, gold speech frames | 30.8 / 39.3 | 33.6 / 34.9 | 48.0 / 20.2 | 36.2 / 4.1 | 35.0 / 3.3 | 24.7 / 3.1 |
| target mass on SIL, gold SIL frames | 96.2 | 93.1 | 89.0 | 84.3 | 79.1 | 74.5 |
| target minus recognizer on the gold phone | +0.4 | +0.2 | +0.5 | +0.1 | +0.1 | -0.2 |
| TV(target, recognizer) per frame | 0.19 | 0.22 | 0.17 | 0.11 | 0.11 | 0.10 |
| segment start mass within +-2 frames of a gold boundary (uniform null 74.3; SUPERSEDED by the matched row below, user objection 2026-09-16: precision-like, no one-to-one matching, no recall, null nearly saturated) | 76.2 | 80.0 | 83.8 | 86.3 | 87.2 | 87.4 |
| boundary F1, one-to-one matched at +-2 frames, argmax(target) (random null with the same count 0.58, every-frame null 0.35) | 0.46 | 0.41 | 0.29 | 0.69 | 0.70 | 0.75 |
| same, R-value / segment count ratio hyp:gold excl. SIL | .52 / 0.16 | .48 / 0.03 | .42 / 0.03 | .72 / 0.75 | .72 / 0.74 | .78 / 0.88 |
| E[d] under seg_post, phones (gold 4.14) / SIL | 3.2 / 3.4 | 3.3 / 3.6 | 3.7 / 5.1 | 4.7 / 8.1 | 4.9 / 7.7 | 4.7 / 7.3 |
| subset PER argmax(recognizer) / argmax(target) | .861 / .905 | .878 / .974 | .942 / .972 | .841 / .825 | .847 / .834 | .878 / .856 |

Audit amendment (2026-09-16): (i) the reader's row-sum gate (0.1 %) FAILED at sub-epochs 3-8 (worst 0.594 % at
sub-epoch 4) and was widened to 0.7 % AFTER the run; the auditor refuted the precedent I named (armA ep1 dev-other,
0.37 %), but the full 2026-09-15 set of 12 runs reaches 0.611 % (armA ep3 dev-clean, `analysis/out/emc_target_vs_gold.*.txt`),
so 0.594 % is inside the fp32 range that read was audited at, and the 0.7 % ruling stands; the post-hoc widening is
recorded here as such. Worst-case effect on a mass share is 0.03 points, immaterial to every row except "target minus
recognizer", whose values (<= 0.5 points) are within that uncertainty: read them as zero, which is the reading anyway.
(ii) Chance level: 4.94 / 5.16 % on the gold phone is 3.2x / 2.8x the uniform null (1.53 / 1.85 %) and 2.0x / 1.9x a
frequency-matched null (2.42 / 2.67 %); the recognizer's own share is the same or higher (3.0x / 2.7x uniform), so the
target adds nothing the recognizer does not already have. (iii) The boundary null is fair (same offsets array, same
tie rule). (iv) No gold enters the DP call. Checkpoints were read from output/lam3_tri (job cleaned).
Matched-boundary read (`reports/exec_mass_profile_v2_boundaries_2026-09-16.md`, reader section C(iv), pooled counts over the 300 utts, greedy one-to-one matching, R-value per Räsänen; A/B/D/F rows unchanged from the first run): at sub-epoch 8 the target's argmax segmentation scores P 0.80 / R 0.71 / F1 0.75 / R-value 0.78 at +-2 frames and F1 0.60 at +-1 frame, against 0.58 / 0.46 for a random segmentation with the same count and 0.35 for a boundary on every frame; OS -0.10 (under-segmentation, 0.88 segments per gold segment), so the number is not bought by over-segmentation. The seeded reference scores F1 0.85 / R-value 0.87 at +-2 frames (null 0.58), ratio 0.94. During the anneal (sub-epochs 1-3) the cold target emits almost no phone segments (ratio 0.03-0.16); boundaries appear when tau reaches 2. AUDITED `reports/audit_mass_profile_boundaries_2026-09-16.md` CONFIRMED_WITH_CAVEATS: matching is one-to-one and Räsänen's formulas; greedy matching can only deflate P/R vs optimal (0 inflations in 4000 random cases) and scores the nulls with the same code; every number re-derives from the json counts; the free utterance-end matches are 3 % of boundaries and cost 0.008 F1 when removed. Caveat: the target path is conditioned on the label-free acoustic units through phi, so the audio-only number is the recognizer's own F1 0.748, which is the same reading.
Reading (first read): the mass knows WHERE but not WHAT. Silence is found (75-96 % of the mass on gold-silence frames
sits on SIL), phone boundaries are found at F1 0.75 against the seeded reference's 0.85 and a random 0.58, and the duration law
converges on the gold mean by sub-epoch 4. But on gold speech frames the gold phone carries 1 % of the mass during the
anneal and 5 % from sub-epoch 4 on, against 1.4-1.7 % expected if the 55-67 % "other phone" mass were spread evenly:
identity sits at 2-3x chance and stays there while everything else sharpens. The target is a copy of the recognizer at tau 2
(TV 0.10, gold-phone delta <= 0.5 points, target PER 2 points under the recognizer's): the objective supplies NO pull
toward phone identity, so the lattice components that see the audio only through phi (emission) do not discriminate
identities, while the components that see duration and position (duration table, prior, rate) do their job. This is the
frame-level form of "content-free, LM-plausible filler" and points at phi's emission, which S3b-OR (frozen content phi)
and S2d (content phi jointly trained) now test directly; the same reader on those arms is the planned read.
Scale from the seeded reference (same reader, S2b arm A ReturnnTrainingJob.4BRumKQFcXim: 10 h seed recognizer, warm phi,
bigram, tau 2; `reports/exec_mass_profile_armA_2026-09-16.md`, sbatch 1833219, full-split PER 0.173967 reproduced at
sub-epoch 1, row sums within 0.37 %), sub-epochs 1 / 3 / 6: target mass on the gold phone on gold speech frames 63.8 /
63.8 / 64.2 % (cold control 1-5 %), other phone 21.4 / 22.5 / 23.7 %, blank 14.6 / 13.4 / 11.7 %; entropy 0.035 nats,
TV(target, recognizer) 0.012, gold-phone delta <= 0.05 points; start mass within +-2 frames 90 % (null 74 %); E[d] 5.0 /
5.2 / 5.2 (gold 4.14); subset PER 0.168 / 0.215 / 0.223. So a content-carrying alignment puts two thirds of its mass on
the right phone, and in BOTH regimes the target is a copy of the recognizer with no pull toward the gold identity: the
objective neither creates identity from cold nor defends it when seeded (arm A's PER drift shows up as "other phone"
+2.3 points and over-long segments, not as a change in the target-recognizer relation).

**S3b-OR: frozen content-carrying reverse model, cold recognizer (registered 2026-09-16 on the user's question, before
any result; REPORTING-ONLY ORACLE REFERENCE, never a funded or selected arm).** The EM fixed-point read c1 above (flat
theta, warm phi HELD, network-free) reached dev-other PER 0.238 (trigram, k 30, 300 utts) against the seed's own 0.190,
so a content-carrying phi lets a flat recognizer take off in EM; whether the TRAINED recognizer network does the same
under the full EMC objective has not been run. phi = the S2d warm-up slice ExtractSubmoduleCheckpointJob.u3GoBYOWr741
(one theta-frozen sub-epoch at tau 8 with the trigram prior on the supervised 10 h seed recognizer 65NNK8Bwxdtd ep024:
GOLD-DERIVED, as c1's htxT2f9FHvWw). One packed node, four arms, every arm a pack v3 twin (flat recognizer init,
trigram, lam_rate 3, S3_TAU_SCHEDULE, 8 sub-epochs) plus the phi checkpoint: or_frz_tri = phi loaded and FROZEN
(requires_grad False, out of the optimizer, as freeze_recognizer does for theta); or_free_tri = phi loaded and trained
jointly (the S2d analogue from a flat theta: does joint training erode phi's content); or_frz_bt_a_tri = or_frz_tri +
bt_a (back translation against a frozen content side, the regime the literature measured); or_frz_tri_s2 = or_frz_tri
at seed 43. Reads as pack v3; paired rows at sub-epochs 4 and 8 on both dev sets: or_free vs or_frz, or_frz_bt_a vs
or_frz, or_frz_s2 vs or_frz, and every arm vs the banked lam3_tri. Pre-registered readings (no funding gate, the phi is
gold-derived): (i) take-off = dev-other greedy PER < 0.50 at sub-epoch 4 or 8 in or_frz_tri; if it stays in the
0.83-0.90 band the recognizer NETWORK cannot learn from the objective even with a content-carrying phi, contradicting
c1 and pointing at the training loop rather than at phi; (ii) or_free below or_frz by a paired ci95 excluding 0 means
joint training erodes phi's content, which bears directly on S2d; (iii) or_frz_bt_a vs or_frz reads whether BT helps
once its reverse side carries content. Anchors: c1 0.238 / 0.279 (trigram / bigram EM, 300 utts), seed limit 0.190,
cold band 0.836-0.89. WIRED commit 792684f (`freeze_reverse` flag, written into the config only when True; `reports/impl_pack6_config_2026-09-16.md`; review PASS `reports/review_pack6_config_2026-09-16.md`: reverse model has no BatchNorm/Dropout, agg holds no parameters, strict 12-key phi load before the optimizer on every resubmit, census clean), LAUNCHED 2026-09-16: PackedEmcTrainJob.go0lRvkvA6Kq (rqmt as pack v3), 28 PairedPerDeltaJob rows.

**S2d: the main trigram arms from the supervised 10 h seed init (USER DECISION 2026-09-16, "for the main arms, still test
supervised 10 h for init + unsupervised 100 h").** Bed = S2b's: theta init `ReturnnTrainingJob.65NNK8Bwxdtd` ep 24 (greedy
PER 0.058 / 0.116 dev-clean / dev-other, the supervision cost disclosed: 10 h of transcripts), phi warm-up 1 sub-epoch with
theta frozen, REBUILT under the trigram prior (S2b's `htxT2f9FHvWw` was fit under bigram); EMC on tc100, 8 sub-epochs.
Arms = pack v3's lam3_tri (control), bt_a_tri, bt_b_tri, bt_c_tri with only the two init paths changed, one packed node
(`config/sae_4a_s2d_pack.py`, being wired, `reports/impl_s2d_pack_2026-09-16.md`). Reads per sub-epoch on both dev sets
(GreedyPerJob + decode_stats), PairedPerDeltaJob vs the banked init rows (GreedyPerJob.3TcOOObr1IrW / RMkjOs4czduh) at
sub-epochs 4 and 8, and BT arm vs lam3_tri at the same sub-epochs; label-free selection (argmin weighted_lm_ppl) read.
Baseline: S2b arm A (bigram plain L_tau, warm phi) degraded the init 3x, dev-other 0.174 / 0.192 / 0.218 / 0.223 / 0.225 /
0.225 at sub-epochs 1-6, and the selection rule picked sub-epoch 1.
Gate **G4a.S2d** (pre-registered before any result): on dev-other, paired per-utterance delta arm - init with a
speaker-clustered bootstrap. HOLD = ci95 upper bound < +0.010 at sub-epoch 8 (non-inferiority margin one PER point; the
usability rule); IMPROVE = ci95 upper bound < 0 at the label-free-selected checkpoint AND at sub-epoch 8; FAIL otherwise.
BT clause: a BT arm counts as helping only if its paired delta vs lam3_tri on dev-other has ci95 excluding 0 in its favour
at sub-epoch 8 AND the same sign on dev-clean; sub-epoch 4 is a weak BT read (bt_ramp_epochs = 4). Any arm that degrades
like S2b arm A (dev-other > 0.150 at sub-epoch 4) is read as the objective still being anti-aligned with PER on this
bed, trigram or not. Plain greedy PER only; no rescored variant.

### S2d result (2026-09-16, pack PackedEmcTrainJob.tLyvlNT8Zyxq, SLURM 1824989, 8 sub-epochs, tc100, greedy PER)

Numbers: `reports/extract_s2d_read_2026-09-16.md` (per-sub-epoch tables) and the 36 PairedPerDeltaJob summaries under
`output/exp2025_11_06_speech_llms/librispeech/sae_4a_s2d_pack/paired_per/` (speaker-clustered bootstrap, 2000 resamples,
dev-other 2864 utts / 33 speakers, dev-clean 2703 / 40). Init = the 10 h seed at epoch 24: dev-clean 0.0580, dev-other
0.1157 (paired rows say "init10h/ep24").

Dev-other greedy PER by sub-epoch (dev-clean in the report; same shape, 0.05 lower):

| arm | ep1 | ep2 | ep3 | ep4 | ep5 | ep6 | ep7 | ep8 | selected |
|---|---|---|---|---|---|---|---|---|---|
| lam3_tri | 0.329 | 0.337 | 0.288 | 0.261 | 0.252 | 0.263 | 0.255 | 0.254 | ep7 0.255 |
| bt_a_tri | 0.329 | 0.367 | 0.304 | 0.272 | 0.271 | 0.273 | 0.256 | 0.265 | ep5 0.271 |
| bt_b_tri | 0.329 | 0.358 | 0.321 | 0.283 | 0.279 | 0.285 | 0.280 | 0.299 | ep7 0.280 |
| bt_c_tri | 0.329 | 0.407 | 0.305 | 0.283 | 0.283 | 0.292 | 0.278 | 0.280 | ep7 0.278 |

Paired delta arm minus init on dev-other, mean [ci95]: lam3_tri ep4 +0.146 [+0.137, +0.155], ep8 +0.139 [+0.129, +0.148],
selected +0.139 [+0.130, +0.149]; bt_a_tri ep8 +0.149 [+0.140, +0.158]; bt_b_tri ep8 +0.183 [+0.172, +0.194]; bt_c_tri
ep8 +0.164 [+0.154, +0.174]. Dev-clean deltas are the same size (+0.149 .. +0.191 at ep8). BT arm minus lam3_tri on
dev-other at ep8: bt_a +0.011 [+0.008, +0.013], bt_b +0.045 [+0.042, +0.048], bt_c +0.026 [+0.022, +0.029]; same sign on
dev-clean at ep4 and ep8 in every arm. Greedy rate 9.1-9.8 phones/s on both dev sets at every sub-epoch (rho 9.66), every
utterance a distinct string (2864/2864): not the blank or content-free regime.

**G4a.S2d: FAIL in all four arms** (ci95 lower bound >= +0.129 against a HOLD bound of +0.010; IMPROVE impossible). The
degradation clause fires in every arm (dev-other 0.26-0.28 > 0.150 at sub-epoch 4): on the seeded bed the objective is
anti-aligned with PER with the full trigram as it was with the bigram (S2b arm A). **BT clause: no BT arm helps**; all
three are significantly worse than the control on both dev sets, bt_b (the largest lam_bt) worst. Label-free selection
picked ep7 / ep5 / ep7 / ep7 out of the pool sub-epochs 4-8 (S3_SELECTION_EPOCHS; ep1 was never a candidate), not the
best epoch (ep5 / ep7 / ep5 / ep7 on dev-other); irrelevant here since every checkpoint is far above the init.

Audit (`reports/audit_s2d_read_2026-09-16.md`): CONFIRMED. per_a of every paired row is the frozen init GreedyPerJob
(RMkjOs4czduh / 3TcOOObr1IrW, PER 0.115741 / 0.058003, identical tag sets asserted in the job), every arm config inits
theta from 65NNK8Bwxdtd ep24 and phi from MI27gurZzzeG ep1, plain greedy decode both sides, no blank or empty outputs.
Two frame concerns, neither able to produce the FAIL: the label-free selector is scored on dev-clean+dev-other themselves
(label-free, not out-of-sample, so optimistic for the arms), and the selection pool is sub-epochs 4-8.

Shape, compared with S2b arm A (bigram, warm phi: 0.174 -> 0.225, monotone drift): the trigram arms lose 0.21 PER in the
FIRST sub-epoch (tau 8, the flattest temperature) and then recover monotonically to 0.25 as tau anneals to 2 (sub-epoch
4 on), yet the recovery stops 0.14 above the init and the label-free selector points at the late epochs. So the objective's
optimum on this bed lies at a recognizer that is 2.2x worse than the seed on dev-other and 3.6x worse on dev-clean; with the
full trigram it is no longer content-free (cf. the cold band 0.84-0.89) but it is not the seed either. Together with the
cold read (target copies the recognizer, gold-phone mass 5 %) this locates the defect in what the cycle rewards, not in
the search or the init: the pending S3b-OR (frozen oracle phi) and S3b-CT (content term) are the two funded probes of that.
Consequence for the campaign: an EMC stage that starts from a supervised seed cannot be used as a "refinement" step under
this objective; the seed itself remains the best recognizer we have (G4a.S2d FAIL, recorded; no rewrite of the gate).

Alignment-mass profile of the S2d control (analysis/emc_mass_profile.py, same protocol as the cold read: dev-other, 300
stride utterances, `analysis/out/emc_mass_profile.s2d.lam3_tri.ep1-4-8.dev-other.txt`, SLURM 1839487,
`reports/exec_mass_profile_s2d_2026-09-16.md`; row-sum sanity 1.7e-4 .. 3.0e-3, under the 7e-3 tolerance):

| statistic, post_q | S2d ep1 (tau 8) | S2d ep4 | S2d ep8 | cold lam3_tri ep8 | seeded bigram arm A ep6 |
|---|---|---|---|---|---|
| gold-phone mass in gold speech | 54.5 % | 61.9 % | 63.3 % | 5 % | 64 % |
| other-phone mass | 27.8 % | 23.5 % | 23.8 % | | |
| blank / SIL mass in gold speech | 14.8 / 2.9 % | 13.3 / 1.2 % | 11.7 / 1.3 % | | |
| TV(target, recognizer) | 0.114 | 0.022 | 0.020 | 0.10 | 0.012 |
| matched boundary F1 (tol 2) / R-value | 0.804 / 0.821 | 0.848 / 0.869 | 0.852 / 0.873 | 0.752 / 0.783 | 0.839 / 0.853 |
| segment ratio excl SIL | 0.88 | 0.97 | 0.98 | 0.88 | 0.92 |
| E[d] phones (gold 4.14) | 4.15 | 4.69 | 4.68 | 4.75 | |
| subset PER argmax recognizer / target | 0.321 / 0.245 | 0.257 / 0.233 | 0.252 / 0.233 | 0.878 / 0.856 | |

Reading: from the seed the objective settles within 4 sub-epochs into the SAME profile the seeded bigram arm A reached
(gold-phone mass 63 % vs 64 %, boundary F1 0.85 vs 0.84, target a copy of the recognizer at TV 0.02), at PER 0.25 vs
0.225, while the seed it started from is at 0.116. So the trigram did not move the objective's fixed point on this bed;
the recognizer is pulled from the seed to that point in the first sub-epoch (at tau 8, TV 0.11, PER 0.32) and then
converges to it. One new fact: on this bed the target's argmax is better than the recognizer's by 0.02-0.08 PER at
every sub-epoch (0.245 vs 0.321 at ep1), unlike the cold bed where both are content-free, so the cycle target carries
some identity here, but its own optimum is at 0.23, not at the seed. The 24 % of mass on wrong phones in gold speech is the
substitution error the greedy read shows.
## Current mechanism assessment and partial follow-ups (2026-09-16)

Fresh audit: `reports/codex_mechanism_audit_2026-09-16.md`; numerical extraction:
`reports/codex_current_evidence_2026-09-16.md`. Main suspect: the unanchored cycle supplies a self-reinforcing
phone target, and joint reverse-model updates can worsen its phonetic grounding. Reconstruction, rate and
phonotactic plausibility can improve without recovering the spoken phones. The cold mass profile retains timing
information and weak identity; it does not establish zero acoustic information or a coherent private code.
The seeded failures and repeated network-free marginal fitting show that a good initializer is not protected.
The prior helps in the registered ablations; stronger prior order alone did not stop the drift.

**2026-09-17 qualification:** the audited fixed-checkpoint recognizer-factor contrast below shows that
the current recognizer adds conditional phone-identity mass to the epoch-4 reconstruction posterior.
It therefore weakens the specific explanation that its current path preferences suppress a better target
from the frozen reverse model and LM. Earlier q/phi coadaptation remains unresolved. The comparison
does not isolate weak reverse emissions from duration, LM or CTC path effects; those remain the next
mechanistic distinction, rather than an established attribution to phi alone.

The S3b-OR **sub-epoch-4 and -8** reads are complete, on the registered tc100/trigram/rate-3/tau-8-to-2 bed;
endpoint audit: `reports/codex_pack6_endpoint_audit_2026-09-16.md`.
Every OR arm starts with flat theta and the same S2d warm phi slice `u3GoBYOWr741` (one tau-8 warm-up sub-epoch
from the supervised 10 h seed). These are GOLD-DERIVED, REPORTING-ONLY diagnostics. Full-split greedy PER:

| OR arm | sub-epoch | dev-clean | dev-other | paired dev-other delta vs frozen phi [speaker ci95] |
|---|---|---|---|---|
| frozen phi | 4 | 0.383141 | 0.453832 | reference |
| jointly trained phi | 4 | 0.614282 | 0.642978 | +0.189147 [+0.167703, +0.211536] |
| frozen phi + BT | 4 | 0.357940 | 0.423280 | -0.030551 [-0.045059, -0.013737] |
| frozen phi, seed 43 | 4 | 0.364710 | 0.435617 | -0.018215 [-0.028536, -0.007747] |
| frozen phi | 8 | 0.321105 | 0.367813 | reference |
| jointly trained phi | 8 | 0.593543 | 0.624076 | +0.256263 [+0.239735, +0.274029] |
| frozen phi + BT | 8 | 0.280721 | 0.350416 | -0.017397 [-0.032549, -0.002260] |
| frozen phi, seed 43 | 8 | 0.314691 | 0.383213 | +0.015400 [+0.006677, +0.024070] |

Sources: `PackedEmcTrainJob.go0lRvkvA6Kq`; registered
`output/exp2025_11_06_speech_llms/librispeech/sae_4a_s3b_pack6/` per-arm `ep{4,8}/<split>/per.json` and
`paired_per/<arm>_vs_or_frz_tri/ep{4,8}/<split>/paired_per.json`. Frozen/joint sub-epoch-4 dev-other PER resolves to
`GreedyPerJob.toupXJO215uo` / `UGaAtpqf07ah`. Both sides use the same 2,864 dev-other items, 33 speakers,
gold, greedy collapse and SIL removal; the checked config delta is `freeze_reverse=True` alone.
The frozen arm meets the pre-registered reporting-only take-off threshold at both reads; all four label-free
selectors choose sub-epoch 8. The joint-training penalty persists on dev-clean too: at 8,
+0.272438 [+0.253559, +0.290691]; BT versus frozen is -0.040383 [-0.051789, -0.028165] there.
BT's improvement survives the full-weight endpoint on both splits. This supports the value of an informative
reverse side without establishing an unsupervised route. The seed-43 frozen arm also takes off, but its
relative dev-other advantage at 4 reverses at 8; it is not a uniformly better seed.

**Interpretation amendment:** S3b-OR reading (ii) above has its PER sign reversed. Joint **above** frozen,
with positive free-minus-frozen paired CI, is the harmful direction. The original wording is retained as provenance;
no numerical threshold or funding gate is changed. This contrast identifies harm from allowing phi updates,
not specifically loss of phi's emission information: joint dynamics also change theta's trajectory.
There is no matched frozen-phi experiment starting from S2d's seeded theta, so the OR contrast cannot alone
attribute S2d's degradation to phi updates. S2d's high initial temperature remains a plausible contributor to
the first-sub-epoch loss, while the older tau-2 drift rules it out as the whole explanation.

**S3b-CT endpoint: CLOSED FAIL** (`PackedEmcTrainJob.nineWO8G0tFD`, same registered cold bed;
fresh audit `reports/codex_pack5_endpoint_audit_2026-09-16.md`). Every arm fails the original dev-other
PER < 0.50 clause at both sub-epochs 4 and 8. Final full-split greedy PER and paired arm-minus-control deltas:

| arm | dev-other PER, sub-epoch 8 | paired delta vs lam3_tri [speaker ci95] |
|---|---|---|
| content | 0.887926 | +0.010740 [+0.006754, +0.015096] |
| BT + content | 0.869733 | -0.007452 [-0.014924, -0.000033] |
| entropy-a | 0.905441 | +0.028256 [+0.022435, +0.034685] |
| entropy-b | 0.927249 | +0.050063 [+0.041965, +0.058618] |

Baseline = banked lam3_tri at the same sub-epoch (0.877185); identical 2,864 dev-other items / 33 speakers,
GoldPhonesJob.ZGSp0hxyd2YP, greedy SIL-drop, 177,275 reference phones and 2,000 speaker resamples.
BT+content's dev-other gain has the opposite sign on dev-clean, so it also fails the paired cross-split clause.
Entropy-a's paired gain on both splits at sub-epoch 4 reverses by 8; its rate hinge remains a confound for
entropy-only attribution. All four label-free selectors choose sub-epoch 4, which also fails take-off.

The auxiliary loss was active: content-arm dev code accuracy rises 0.345 -> 0.528 (chance 1/64), and CE
falls 9.668 -> 1.730; BT+content reaches 0.527 accuracy / 1.738 CE. At the registered n_layers=1/tap1,
the head reads the pre-convolution representation. Its gradient reaches the head and shared BatchNorm affine /
residual input projection, but bypasses the final phone convolution (`recognizer.py:282-308`). Thus this is
a trained acoustic auxiliary, not evidence that the phone output acquired the corresponding identity.
The original CT gate and its cold-remedy closure stand; this result leaves the OR mechanism question open.
Artifacts: `output/exp2025_11_06_speech_llms/librispeech/sae_4a_s3b_pack5/`, per-arm
`ep{4,8}/<split>/per.json`, `paired_per/<arm>_vs_lam3_tri/ep{4,8}/<split>/paired_per.json`,
and the pack's per-arm `learning_rates`; comparison provenance and constants are checked in the audit.

**Interpretation amendment to S2d and the index:** the finite training and marginal-fitting trajectories do
not establish an attained objective optimum or a fixed point equal to the reverse-model posterior.
The statements above that S2d "settles" at a measured optimum are superseded by the weaker observation:
the tested eight-sub-epoch run degrades its seed and only partly recovers. Factorized frame-marginal projection
does not preserve the full structured posterior. Existing gate failures and measured PER remain unchanged.

The planned alignment-mass profiles and fixed-recognizer comparisons are complete and audited below.
They directly test the reverse-evidence interpretation of the persistent OR gap. Its transfer to the
seeded S2d failure remains the next experimental question; no new training sweep is launched.

**OR diagnostic read, specified before execution (2026-09-16):** use the existing mass reader on the
frozen/joint arms at sub-epochs 4 and 8, with the same 300 stride-selected dev-other utterances, frame gold,
feature/unit inputs, tau=2 and row-sum tolerance as the registered S2d read. At sub-epoch 8 additionally
cross the two reverse snapshots with the two recognizer snapshots; the native reads supply the diagonal,
so only two extra checkpoint combinations are needed. No model is trained or selected by this read.
Holding theta fixed makes its frame probabilities identical across the phi contrast. Read the paired change
in target gold-phone mass on gold speech and target greedy PER; retain utterance identities and use the
existing 2,000 speaker-resample convention for uncertainty. A consistent gold-mass loss under updated phi
for both fixed recognizers supports loss of useful reverse-model evidence. A joint-training PER penalty
without that loss leaves changes in the recognizer trajectory as an alternative. This compares the whole
reverse model (including durations), not its emissions alone, and does not add or change a phase gate.
The measured target is the untilted cycle posterior, as in the reference read; rate/aggregate gradients
are outside this diagnostic's scope.
Run: `OrMassDiagnosticJob.SWqFadg0m8X3`, launched 2026-09-16 via
`config/sae_4a_or_mass_diagnostic.py`; six checkpoint evaluations, all native/cross epochs must pass
the existing lattice sanity before the paired read is accepted. Both component checkpoint paths and
the exact utterance-level counts are retained in its outputs.

**OR diagnostic result (2026-09-16):** complete; fresh audit
`reports/codex_or_mass_endpoint_audit_2026-09-16.md`. The same 300 distinct dev-other utterances cover
33 speakers, 77,333 gold-speech frames and 18,743 reference phones. Gold, feature/unit paths, tags and
reference counts match within both pairs and the registered S2d reader inputs. Strict component loading
holds theta fixed; all six checkpoint reads pass the original lattice sanity (worst row-sum deviation
0.006836, tolerance 0.007). Sub-epoch-8 updated-minus-frozen phi contrasts:

| fixed recognizer | gold-phone mass delta [speaker ci95] | target greedy PER delta [speaker ci95] |
|---|---|---|
| frozen-arm theta | -0.092855 [-0.096375, -0.089227] | +0.173611 [+0.165066, +0.181720] |
| joint-arm theta | -0.022239 [-0.024275, -0.020231] | +0.050472 [+0.043362, +0.058360] |

Source: the run's `output/paired_comparison.json` and `output/profiles/*.json`; paired speaker bootstrap,
2,000 resamples, seed 0. The pre-specified directional mass criterion is met: updated whole phi loses useful
reverse evidence under both fixed recognizers, and target PER worsens too. Thus the OR penalty is not
explained solely by comparing different recognizer trajectories. This concerns the untilted cycle target
at tau=2 on the 300-utterance diagnostic, not full-split recognizer PER or the complete training gradient.
It does not isolate emissions from durations, explain how the deterioration develops, or establish the
same mechanism in seeded S2d. OR's supervised origin and reporting-only status remain unchanged.

**Next training proposal (2026-09-16, in response to the user's next-step question):** repeat the S2d
control from the same 10 h recognizer seed and its same warmed phi, changing only `freeze_reverse=True`.
Keep the trigram, tc100 data, eight-sub-epoch schedule including tau 8 -> 2, rate term and selection rule
identical. Read it against the original seed with G4a.S2d's existing HOLD rule (paired dev-other CI95 upper
bound below +0.010 at sub-epoch 8), and against the banked joint S2d control. This tests whether freezing
protects an already useful recognizer; the OR cold-start result alone cannot answer that. If it holds, pursue
a stable reverse teacher; if it fails, isolate the high-temperature start next. This is a proposed additional
training arm, with no launch or additional training budget committed. Obtaining an informative phi without
paired labels remains the unresolved cold-start problem.

**Superseded by the user's subsequent priority (2026-09-16):** the seeded-S2d proposal above is withdrawn
from the active queue. Research now targets fully unpaired, non-GAN cold initialization; improving the
supervised seed is not an experimental objective. Original results and gates above remain unchanged.

### Withdrawn standalone-initializer proposal (2026-09-16)

**User scope correction:** the SylCipher-inspired experiment below is withdrawn before any implementation
or launch. It is retained as proposal provenance only. The active direction stays within §4a cycle consistency.

The existing failure inventory rules out treating a new count fit as a new idea: §1f already tried K500
positional unigrams, skip-1..6 bigrams and tri-skipgrams (best selected dev-other PER 0.8580). Its G9
identifiability read favored an audio-free null on all five configurations, including after rate repair.
§1g's selected ESPUM seed followed by matched-4gram Gaussian/table EM remained at 0.8271/0.8165 PER on
that phase's 890-utterance selection role; those are not full-split cold-EMC results. No closed gate is reopened.
Evidence and exact operating points: `reports/codex_cold_prior_inventory_2026-09-16.md`, `SAE_1f.md`, `SAE_1g.md`.

The distinct literature lead is [Wang et al., SylCipher (2026)](https://arxiv.org/pdf/2608.22907): separate
acoustic/text syllable embeddings and reconstruction heads share a Transformer and quantizer. Each modality
reconstructs its own masked sequences; unpaired contexts couple the modalities through shared capacity.
Reported unmatched-English CER moves from 48.6 after this initialization to 35.9 after later stages, on a
different 460-hour speech bed. These are gold-selected CER results, not a pure label-free-selected SAE baseline:
Appendix Table 8 uses paired-validation SER/WER. The advertised code was unavailable at review time.
Full-text/provenance review: `reports/codex_cold_representation_literature_2026-09-16.md`.

Other reviewed methods do not provide a ready initializer: the count-matching recipe overlaps our failed
family; bare geometry transport lacks useful demonstrated cold ASR; the reviewed articulatory model uses
supervised phone/feature targets. Sources and limitations: `reports/codex_cold_mapping_literature_2026-09-16.md`.
This is a reason to test a different source of alignment information, not evidence that shared modeling will work.

**Proposed next experiment, not launched or a new funded gate:** test the shared masked-model initialization
alone, before boundary self-training, moment matching or joint cycle training. The hypothesis is that repeated
syllable contexts across unpaired speech and text provide naming information that local phone counts miss.
Preserve tc100 and the existing unpaired text scope for the local pilot; explicitly report this reduced speech
scale relative to the paper. Use syllable-level observations and text tokens, with all model/segmentation
choices fixed without paired labels. Replacing them with K500 frames and phone text would be a separate
hypothesis; a negative result there would not test the syllable mechanism.

The text head must use the upstream **orthographic** corpus, not invert `T_phi` phone strings. Syllabification
must retain explicit word-boundary markers so its predicted pieces deterministically concatenate back to
word text; no G2P inversion or gold-assisted word reconstruction is allowed. Unknown outputs remain scored
errors rather than being dropped. Pin the corpus path, vocabulary/OOV rule and existing reference/hypothesis
normalization together before interpreting CER/WER. Training discards utterance IDs and never joins text
strings to the speech examples.
The verified upstream source is `DownloadJob.g4jClO48cAvP`'s `librispeech-lm-norm.txt.gz`, shared by
`TextToPhonemeJob.THKMON3k9LJQ` and `PhonemizeWithSilJob.DbFgvZOGZQ8F`; the latter uses all lines.
Use this orthographic source directly. Held-out word references are
`LibriSpeechWordRefsJob.1EsLvSbyl06D/output/word_refs.json`; the existing WER normalization uppercases and
whitespace-splits both sides. Reuse the references and normalization for native rendered hypotheses;
the historical recognizer and its beam/LM settings are not inputs to this pilot. No existing CER scorer was
found, so its exact normalization and implementation remain a launch-specification item. Input provenance:
`reports/codex_cold_prior_inventory_2026-09-16.md`.

Before implementation, pin an admissible speech-only front end and its segmentation rule, account for any
gold-calibrated shifts/thresholds or checkpoint selection in its upstream recipe, and specify the update
budget and a fixed endpoint or held-out unimodal reconstruction selector. No public checkpoint is accepted
merely because it is called self-supervised. The shared-model reference and all new constants must be traced
to the paper or existing setup; this is an adaptation with new selection, not a reproduction of its reported score.

Freeze the initializer and selector before paired evaluation. Read native speech-to-text CER/WER and
utterance-specific correctness against an audio-free control and speaker/length-matched audio derangement,
with the same output/decoding rules. Also compare to a control with the same acoustic/text tokenization but
separate sequence-model parameters, isolating the proposed cross-modal sharing. Lower reconstruction loss,
shared codebook use or improved boundaries alone do not establish recognition. Only an actual, content-sensitive
cold recognition gain would motivate converting predictions into an EMC initialization and testing that against
the existing cold baseline. This proposal does not use or seek to preserve the supervised seed.

The shared/separate comparison uses the same hidden dimension and text head architecture; both inference
paths apply the learned text head to acoustic hidden states, with no fitted cross-modal or gold permutation.
Its update budget and native decoding rule must be identical. Fresh direction audit:
`reports/codex_cold_direction_audit_2026-09-16.md`; the lead is distinct, but admissible front-end provenance,
exact text rendering/scoring, stopping and budget remain launch-specification requirements. No cold improvement
is claimed from this review and no new job is running.

### Scope-corrected cycle proposal (user clarification 2026-09-16)

Keep the current phone-level exact-marginal cycle, speech-only features/K500 observations, trigram and flat
initialization. The user requests inspiration from unsupervised MT for improving this cycle's cold start;
the standalone syllable/shared-MLM proposal above is withdrawn, not a replacement for GAN initialization.

Verified UMT evidence is conditional, not a recipe transfer. [Lample et al., EMNLP 2018](https://arxiv.org/pdf/1804.07755)
report Transformer en-to-fr BLEU 25.1 with denoising versus 0.0 without it despite backtranslation; their shared
BPE/pretrained embeddings supply alignment absent from the SAE cold bed. [Their ICLR 2018 study](https://arxiv.org/pdf/1711.00043)
finds a smaller denoising ablation effect, while removing input noise changes Multi30k en-to-fr BLEU from
27.48 to 16.76. [Kim et al., EAMT 2020](https://aclanthology.org/2020.eamt-1.5.pdf) find that more denoising/BT
does not rescue weak crosslingual initialization. Thus denoising may help a cycle retain content but does not
establish a way out of this project's flat-start basin. Full-text review:
`reports/codex_4a_umt_cycle_literature_2026-09-16.md`.

Code evidence (`reports/codex_4a_cycle_update_code_2026-09-16.md`): EMC already trains the final phone
probabilities from clean-unit reconstruction via detached, freshly computed exact Gibbs posteriors. BT already
uses separate interleaved optimizer steps and a detached live reverse generator. The real recognizer also uses
dropout 0.1, so denoising is not entirely absent. Existing structured SpecAugment is confined to the auxiliary
clean-teacher KL view comparison; the content auxiliary bypasses the final phone convolution.

**S3c experiment, approved by the user before execution (2026-09-16):** compare canonical cold `lam3_tri` with the same run using the existing
SpecAugment corruption C on the recognizer input of real training batches in **sub-epochs 1-4 only**, the
reference annealing window; sub-epochs 5-8 return to ordinary inputs with the baseline dropout retained.
This fixes the schedule discrepancy found in `reports/codex_4a_incycle_proposal_audit_2026-09-16.md` before
any implementation or result. The lattice uses q_theta(pi | C(x))
and the same p_phi(z_clean | pi, eta) and phone trigram; the observed unit stream, frame count and eta stay
from the original utterance. Both directions remain the existing models. There is no new tokenizer, pretrained
ASR teacher, standalone initialization or direct supervision. This asks whether structured corruption inside
reconstruction provides more useful pressure on phone content than the tested prediction-consistency term.
It does not claim reconstruction was previously absent or that corruption resolves label identity.

Use the existing corruption implementation with `SpecAugmentOpts(time_max_width=8)`; all other settings
retain the helper defaults. The width cap is the recognizer's 9-frame convolutional receptive field minus
one, not a value selected on gold; overlapping masks can still erase a wider neighborhood. Time-mask count
is uniform from 2 to min(max(T//100,2)*4,T), clipped for tiny inputs, and each width is uniform from 1 to 8.
Channel count is uniform from 2 to 5, width from 1 to 204 for the 1024-dimensional L15 features; masks use
zero fill and clip at the input boundary. The existing independently seeded `step_generator` preserves
the global RNG stream used by baseline dropout. No mask-strength sweep is authorized.
The intervention changes the real training input of the q-based objective, including its aggregate/rate
terms; describing only the main L_tau as changed would be incorrect. Normal train-mode BN sees the masked
input, an explicit consequence to report; do not borrow the auxiliary branch's BN-freeze guard without a
clean pass. Apply masks only during training: validation/selection and greedy evaluation stay clean.
The single-arm reference allocation is one GPU, 6 hours, 64 GB host RAM, 16 CPUs and `gpu_mem=96`;
it comes from the existing rate-arm `emc_training` registration, not Pack3's four-GPU pack multiplication.
One new arm is authorized, with eight sub-epochs and all checkpoints retained; reuse the completed control.
Independent code review must verify the opt-in configuration/hash, training/epoch gating, clean targets,
unchanged default path and exact registration before launch. Source/resource evidence:
`reports/codex_4a_mask_launch_scope_2026-09-16.md`.

Implemented in `2025-10-speech-llm` commits `656e5e4` and `3f4de8f`.
The rendered training configuration matches the saved `lam3_tri` control apart from the two
opt-in masking arguments and output path; the concrete run is linked in State.
Implementation evidence: `reports/codex_4a_s3c_denoise_impl_2026-09-16.md`; independent pre-compute review:
`reports/codex_4a_s3c_denoise_code_review_2026-09-16.md`. These checks establish wiring, not efficacy.

The comparison retains the same tc100 data, seed, parameter initialization, trigram, tau/rate schedules,
training length and label-free checkpoint selector. BT stays at the canonical control's setting; no old BT
or consistency gate is silently reopened. Read paired dev-other PER against the cold control at sub-epochs
4 and 8, with dev-clean alongside. The original G4a.3 take-off criterion remains PER <0.50 plus positive
speaker-matched derangement gap at anneal end. A lower reconstruction loss or larger reverse gap without
phone-error improvement would not count as successful cold-start learning. This is a limited hypothesis
test inside the cycle, not a claim that UMT has solved the speech/text grounding problem.

Relevant literature qualifications, verified in `reports/codex_mechanism_literature_2026-09-16.md`:
[Liu et al., SLT 2022, section 3.3](https://arxiv.org/pdf/2204.02492) demonstrate that phone-distribution
plausibility need not preserve speech content; [Smith and Eisner, ACL 2004, sections 4.3–5](https://aclanthology.org/P04-1062.pdf)
show annealing can damage a labeled initializer in a different latent-language model. Neither establishes SAE's
specific cause, and a fixed named-phone LM means an arbitrary symbol permutation is not an exact symmetry here.

### S3c denoising result (2026-09-17)

The run linked in State completed all eight sub-epochs and all registered reads. Independent endpoint
audit: `reports/codex_4a_s3c_denoise_endpoint_audit_2026-09-17.md`; extraction:
`reports/codex_4a_s3c_denoise_extract_2026-09-17.md`. Same tc100 cold flat-theta/random-phi bed, full
trigram, seed, optimizer and fixed endpoints as the banked `lam3_tri` control; only the preregistered
training-input corruption differs. These are full-split greedy SIL-drop PER fractions.

| Epoch | Split | Control PER | S3c PER | Paired S3c minus control [speaker 95% CI] |
|---|---|---:|---:|---|
| 4 | dev-other | 0.845398 | 0.883881 | +0.038483 [+0.031037, +0.046189] |
| 4 | dev-clean | 0.830090 | 0.863270 | +0.033179 [+0.029421, +0.037128] |
| 8 | dev-other | 0.877185 | 0.879921 | +0.002736 [-0.001650, +0.007320] |
| 8 | dev-clean | 0.860910 | 0.870412 | +0.009502 [+0.006063, +0.013491] |

Pairing covers identical 2,864 dev-other utterances / 33 speakers / 177,275 reference phones and
2,703 dev-clean utterances / 40 speakers / 193,644 reference phones; speaker bootstrap 2,000 draws,
seed 0. At the gate, dev-other errors are 156,690 versus control 149,868. The candidate's own-phi
speaker-matched derangement gap is +2.187896 log likelihood/frame at epoch 4 and +2.740964 at epoch 8
(500/500 dev-other items). **G4a.3 FAIL:** the positive-gap clause passes, but epoch-4 PER is above 0.50.
The two existing label-free selectors both choose epoch 4; this is also the selected-checkpoint comparison.
Epoch 8 does not show a significant dev-other advantage or disadvantage; dev-clean remains worse.

This scheduled corruption did not produce cold take-off. Its effect on primary-input masking and
train-mode BatchNorm is joint; these data cannot attribute the degradation to either separately.
The gap establishes a reverse-model preference for its own strings, not phone accuracy. The result
does not establish that a different noise schedule would work; it does not justify a mask-strength sweep.
Main mechanism hypothesis remains an ungrounded, self-reinforcing cycle target; this experiment alone
does not identify its cause or establish absence of all phonetic information.

Bounded follow-up, fixed before its diagnostic outputs: reuse `analysis/emc_hyp_inspect.py` for
`lam3_tri`, `s3c_denoise`, `lam3_tri_ep8`, `s3c_denoise_ep8`, on the full dev-other set, with five matched
null draws at seed 0 and the same trigram. Outputs go to `analysis/out/s3c_denoise_endpoint_2026-09-17/`.
The gold-dependent relabeling is a diagnostic only; it never supplies a training map or selects a checkpoint.
This read retains the existing statistic and does not replace the failed take-off gate.

The read is complete; audit: `reports/codex_4a_s3c_hyp_inspect_audit_2026-09-17.md`, execution:
`reports/codex_4a_s3c_hyp_inspect_exec_2026-09-17.md`. All four outputs reproduce their banked S/D/I/PER;
the control's epoch-4 diagnostic also reproduces its earlier reference. Additional descriptive reads:

| Checkpoint | Own length/unigram-matched null PER | Alignment-derived remapped PER | Hypothesis phones/s | Length correlation |
|---|---:|---:|---:|---:|
| Control ep4 | 0.862811 | 0.845393 | 7.4403 | 0.918565 |
| S3c ep4 | 0.895228 | 0.880570 | 8.9183 | 0.953042 |
| Control ep8 | 0.894780 | 0.875053 | 8.7987 | 0.955298 |
| S3c ep8 | 0.886195 | 0.879763 | 8.4469 | 0.936982 |

Each decode beats its own limited random-string construction, with a smaller descriptive margin for S3c
at both endpoints. These arm-specific null margins are not paired treatment estimates; five-draw SDs are
not speaker confidence intervals. At epoch 4, S3c reduces deletions (24,456 versus 45,528) but increases
substitutions (120,960 versus 99,188) and insertions (11,274 versus 5,152). Better length tracking is
therefore insufficient. Its raw trigram score is higher (-4.1866 versus -4.7923 nats/phone), but differing
token counts and unigram compositions prevent interpreting this as an isolated sequence-quality gain.

**Interpretation amendment:** the older S3b hypothesis-inspection claim that these outputs are "not a
relabeled phone code" is not established by its mapping statistic. Hungarian matching optimizes one
existing edit-alignment confusion matrix, then recomputes PER; it is fitted and evaluated on the same
gold, not optimized globally over realigned sequences. Small gains establish only the performance of
that fitted mapping, not absence of phonetic information or of another useful relabeling. The failed
greedy-PER gates stand unchanged. This limitation is also recorded in `SAE_ref.md`.

### Literal hypothesis inspection (2026-09-17)

Read the same eight utterances (four shortest and four median by gold phone count, selected by the existing
reader) across control/S3c and epochs 4/8: 32 actual hypothesis/reference pairs. In S3c epoch 8, gold
`N OW` for `1686-142278-0068` gives `DH AH T ER Z DH`; two `Y EH S` items,
`1686-142278-0041` and `4515-11057-0053`, give `DH AH T ER G OW DH` and
`DH AH T ER Z G OW DH`. S3c epoch 4 repeatedly emits fragments such as `AH L IY` and `AH L ER`
in the medium-length examples; control outputs also reuse fragments such as `B AA N TH`.

These strings support recurring multi-phone fragments with utterance-specific variation and poor named
phone identity. They do not establish coherent language, audio irrelevance, or duration-only templates.
The four medium examples share 50 gold phones, not necessarily the same audio duration; their differing
outputs do not settle the duration-only explanation. The fitted 1:1 mapping's limitations remain as stated
above. Examples: `analysis/out/s3c_denoise_endpoint_2026-09-17/emc_hyp_inspect.*.dev-other.txt`, section 1;
fresh qualitative audit: `reports/codex_4a_hypothesis_read_audit_2026-09-17.md`.

The subsequent saved-output read covers all 2,864 dev-other items with exact acoustic frame counts from
the common 50-Hz L15 feature HDF. It finds 485 repeated-duration groups, covering 2,645 utterances and
8,530 unordered pairs. Each of the four endpoints has zero identical complete hypotheses among these
pairs. Thus the saved outputs are not a single deterministic string per acoustic frame count; this does
not exclude several stereotyped variants or establish phone-content sensitivity.

S3c epoch 4 emits `AH L IY` 8,350 times across 2,526 utterances: 5.2726% of its 158,365 available
trigram positions, versus 140/171,549 = 0.0816% in gold. `AH L ER` appears 4,344 times across 2,049
utterances, versus 44 gold occurrences. The control also repeats fragments: its epoch-4 `EH R IY`
occurs 2,014 times across 1,274 utterances, versus 196 in gold. These are overlapping within-utterance
trigram counts with each stream's own denominator; no inferential interval or causal LM attribution is
claimed. The result supports widespread recurring fragments with variation, beyond the initial examples.
Full strings, deterministic duration-group examples and counts:
`analysis/out/cold_pattern_read_2026-09-17/{summary.json,summary.txt}`; execution:
`reports/codex_4a_pattern_read_exec_2026-09-17.md`; independent result/direction audit:
`reports/codex_4a_pattern_s3d_direction_audit_2026-09-17.md`.

### S3d preparation: categorical phone-output content

Question: does requiring acoustic information to pass through the final categorical phone output improve
cold learning, compared with the failed hidden-layer content auxiliary? This is a hypothesis test within
the existing cycle. Core reconstruction already traverses q; the proposed difference is a direct acoustic
code objective that does not obtain its targets from the co-trained reverse model or read hidden features.
The literal output motifs motivate the question but do not establish its cause.

Use the existing, fixed 50-Hz K64 MFCC cluster targets from S3b-CT, with no refitting or paired labels.
For final 41-class CTC probabilities q_t and observed code c_t, add
`L_phone_content = mean_valid_t sum_p q_t(p) * [-log R(c_t | p)]`.
R is a categorical phone-to-code table, parameterized with a standard linear decoder on one-hot phone
categories and row-wise code softmax. It receives no continuous q vector, hidden features, speaker vector
or audio side input. Compute the expectation exactly; no sampling, straight-through estimator or Gumbel
temperature is introduced. All 41 categories, including blank and SIL, contribute on all valid frames.
There is no blank-conditioned normalization, gold mask or new rate term. Blank absorption and a private
acoustic code remain possible; the original phone-error gate, rather than auxiliary success, decides take-off.

The weight is the existing CT value 0.3, constant with no ramp. Replace the hidden content route with this
route, retaining canonical `lam3_tri` for all other losses, data, batching, optimizer, seed, eight-subepoch
schedule, dropout and trigram. Primary S3c masking, BT and entropy remain at canonical baseline settings.
Use the existing content target pipeline; its K64/seed42/fit-set constants are reference inputs, not new
tuning choices. Preserve baseline theta/phi initialization and RNG state when creating the extra decoder;
its initialization uses the existing linear-layer default. The new loss updates q and R, with no direct
gradient into phi. Defaults and all previously registered arms must remain unchanged.

Measure the same full dev-other/dev-clean greedy PER and paired speaker intervals at epochs 4 and 8,
against the banked canonical control; the existing hidden-CT arm is a secondary diagnostic reference.
Keep the label-free checkpoint selector and G4a.3 unchanged: epoch-4 dev-other PER <0.50 and a positive
own-phi speaker-matched derangement gap. Log expected code CE and code accuracy using the argmax phone
category, alongside existing blank/rate metrics. These establish only acoustic code information, never
named-phone success, and must not select mainline hyperparameters or checkpoints using gold.

Preparation requires an independent direction audit, review of the exact implementation/config delta,
proof that the auxiliary gradient reaches the final phone convolution and R, common-parameter/RNG
equality with baseline, and a full graph with only the intended new work. A numerical preflight must
check the real cold-initialized lattice before training, given the earlier uniform-input float32 defect;
it may not normalize posteriors or relax the existing 0.007 conservation tolerance. Any required change
to core numerical computation must be separated from the content intervention before a result is read.
The planned single-arm resource envelope is the existing one-GPU/16-CPU/64-GB/6-hour/96-GB-GPU-memory
reference, with the same exclusive-node policy; no grid or new training allocation is committed yet.

Code evidence: `reports/codex_4a_cycle_coverage_code_2026-09-17.md`. Verified literature and limitations:
`reports/codex_4a_template_literature_2026-09-17.md`; [Chorowski et al., TASLP 2019](https://arxiv.org/pdf/1901.08810)
supports discrete speech bottlenecks carrying phonetic information, but its phone accuracy used a gold-fitted
mapping. [HuBERT](https://arxiv.org/pdf/2106.07447) and [BEST-RQ](https://proceedings.mlr.press/v162/chiu22a/chiu22a.pdf)
support acoustic-code prediction as representation learning, with labeled ASR fine-tuning. None evaluates
this expected categorical loss or establishes cold recovery of the intended phone names.
Independent direction audit found the single-arm contrast coherent, with the above private-code, blank
and numerical-validity limits: `reports/codex_4a_pattern_s3d_direction_audit_2026-09-17.md`.
The prepared candidate is `work/i6_core/returnn/training/ReturnnTrainingJob.fDLNbtpqVnbG`, wrapper
`config/sae_4a_s3d_phone_content.py`, with 134 intended output roots. Independent implementation review
confirmed the stated delta and exact common-parameter/RNG equality under the actual canonical Engine
initialization and flat checkpoint: `reports/codex_4a_s3d_phone_content_review_2026-09-17.md`.
Implementation evidence: `reports/codex_4a_s3d_phone_content_impl_2026-09-17.md`; recipe commit
`39675aa7fe09765361454f97fb212e1dec910bb6`. No training result exists.

**Launch amendment (user 2026-09-17, before S3d training).** The user explicitly authorizes starting
this arm and regards an acoustically informative private code as useful intermediate progress. Code
prediction alone does not establish phonetic content or the intended phone identities; report those
distinctions while retaining the original G4a.3 ASR gate and label-free checkpoint selection. No new
gold-fitted training target, initializer or selection criterion is introduced.

Enable `lattice_float64=True`, matching `ReturnnTrainingJob.IoCQmrlJbCC0`; relative to that control,
the only training change is the content stream/head and the specified weight-0.3 categorical loss.
Primary epoch-4/8 paired PER comparisons now use this DP64 control. The old hidden-CT comparison
remains a disclosed historical float32 diagnostic. Preserve the original control's independent graph;
reference its evaluation outputs without registering its training under a second manager.
The earlier requirement to finish the control before starting S3d is superseded by this explicit start
instruction; both completed endpoints are required before interpreting the content effect. Allocate
one eight-subepoch S3d arm in the already specified GPU1/CPU16/mem64/time6h/gpu_mem96 envelope,
with unchanged whole-node exclusivity and existing evaluation resources. No grid is authorized here.
The earlier float32 S3d candidate remains unlaunched; review the precise config delta before launch.

The matched arm is now `ReturnnTrainingJob.Pso7oeIpqYjY`, with 134 output roots and one new training
job; recipe commit `8d2280810464c34d64e4434aa0c19823c25e2585`. Source/config review passed:
`reports/codex_4a_s3d_dp64_review_2026-09-17.md`; complete configuration, endpoint paths and input
provenance: `reports/codex_4a_s3d_dp64_impl_2026-09-17.md`. All training inputs exist. The two
precision-control epoch-8 comparison inputs are pending their separately managed producer and do not
block treatment training. Launch: `reports/codex_4a_s3d_dp64_launch_2026-09-17.md`; SLURM `1853056_1`
is CONFIGURING at handoff. No S3d training result is available yet.

**Cold-initialization numerical preflight (specified before execution).** Initialize the canonical
`PackedEmcTrainJob.byYMQmBNEpLZ/output/lam3_tri/returnn.config` through RETURNN's
`Engine._create_model(epoch=1, step=0)`: its default seed is 42, and its saved flat-recognizer initialization
is retained; no trained epoch checkpoint is loaded. Use clean/eval mode, epoch-1 tau=8 and alpha=0.
Reuse the completed recognizer-factor diagnostic's deterministic 300-utterance dev-other selection,
96,676 frames, three batches, max_seqs=128 and padded-frame limit 88,000. Gold/MFA only reproduce that
reader's membership, with no phone scoring or training target derived from them. No MFCC labels are needed.
Compute q and reverse segment scores once and compare native float32 DP with float64 promotion of those
same DP inputs, including the prior. Preserve the complete trigram, eta, durations, band and implementation
settings; do not neutralize q, change offsets or renormalize posteriors. Both precisions must have finite
log partitions and finite valid posterior rows within the existing 0.007 conservation tolerance. Report
all batch shapes, row errors, raw precision differences and actual initial q flatness; introduce no new
cross-precision closeness threshold. This checks the initial clean forward only, not the full training
trajectory or the auxiliary loss. A failure blocks the content-arm launch until numerical validity is
resolved separately from the intervention. It uses the remaining original diagnostic envelope: 3,164
exclusive-node seconds after the two completed attempts, with a 52-minute job limit. Source/seed trace:
`reports/codex_4a_cold_numerical_preflight_scope_2026-09-17.md`.
Submitted as `ColdInitNumericsJob.h6DXjOs9sKQS`, SLURM 1848113, with only two result roots:
`output/cold_init_numerics.{json,txt}`. Code review and source fingerprints:
`reports/codex_4a_cold_init_numerics_review_2026-09-17.md`; launch and resource verification:
`reports/codex_4a_cold_init_numerics_launch_2026-09-17.md`. A complete diagnostic can report numerical
FAIL; that is a result to audit, not a reason to discard outputs or retry unchanged.

### Cold-initialization numerical result (audited 2026-09-17)

`ColdInitNumericsJob.h6DXjOs9sKQS`, SLURM 1848113, completed both outputs and all registered work;
the manager exited and scheduler drained. The preregistered numerical gate is **FAIL**. At canonical
cold Engine epoch 1/step 0, seed 42, required flat checkpoint, clean/eval, tau=8 and alpha=0, all 300
stride-selected dev-other utterances and 96,676 frames were retained. The batches are 128×233,
128×497 and 44×1602 padded shapes; full trigram, eta, durations, band and other settings match the reference.

Float32 passes the first two batches, but the last has 7,201/32,185 valid rows beyond the existing 0.007
conservation tolerance. Its row sums span 0.987968–1.025847, maximum error 0.025847. Float64 passes
all valid rows, maximum error 1.3878e-13. Both precisions have 300 finite partitions, no nonfinite rows
and no zero-partition flags. The last batch's maximum raw posterior difference is 0.007411 and partition
difference 0.014165 nats; neither had a new closeness cutoff. Initial q is uniform to the reported
precision: entropy 3.713572 nats, maximum and blank probability both 0.024390245.

Thus the conservation defect occurs at the actual cold initialization, not only under the earlier
synthetic q replacement. Boundary promotion of the same neural inputs removes this observed forward
defect. This is an initial clean-forward result, not evidence that precision caused historical PER failure
or that backward, the rate finite differences, optimizer or training trajectory are valid. S3d stays held
while a DP-only precision repair is checked through the complete training path. The old PER observations
stand; a repaired content-arm comparison will require a control with matching numerical computation.

Actual run use is 156 exclusive-node seconds; cumulative diagnostic use is 592/3,600, leaving 3,008
seconds. Artifacts: `work/analysis/cold_init_numerics/ColdInitNumericsJob.h6DXjOs9sKQS/output/cold_init_numerics.{json,txt}`.
Terminal evidence: `reports/codex_4a_cold_init_numerics_terminal_2026-09-17.md`; independent audit:
`reports/codex_4a_cold_init_numerics_endpoint_audit_2026-09-17.md`.

**DP-only training repair and validation, specified before execution.** Add default-off
`lattice_float64`, omitted from old generated configurations. Preserve neural float32 and all original
non-DP loss inputs. Promote differentiable copies of q and segment scores, the existing represented prior,
and any frozen anchor at the DP boundary; use these same aliases for the main loss and all rate
finite-difference calls. Retain the existing posterior-weighted surrogates, gradient routing, normalization,
epsilon=0.25, central-difference policy, schedules and topology. No recurrence, offset or posterior
normalization change is proposed. Source trace: `reports/codex_4a_dp64_training_scope_2026-09-17.md`.

The paired witness uses the first raw batch from the saved canonical tc100 training dataset and loader:
epoch 1, partition_epoch=4, laplace:.1000 ordering, max_seqs=128 and 88,000 padded frames. It preserves
the actual Engine cold constructor and training epoch RNG, caches the batch once, then executes the
configured training step and backward once per precision from identical parameters, BN/aggregate buffers
and pre-forward CPU/CUDA RNG states. Use train mode, epoch 1/step 0, clear gradients and a fresh run
context for each cell. No optimizer, learned-parameter update, new data, gold, content head or checkpoint.

Repair acceptance requires identical nonprecision inputs/state/RNG; float64 in the main and every rate
DP call; finite partitions and valid posterior rows within the unchanged 0.007 tolerance in every float64
call; finite total loss and all produced gradients, with nonzero theta/phi group gradient norms; and no
learned-parameter change. Record all call shapes/settings, scalar losses, gradient norms and descriptive
cross-precision differences, actual runtime and GPU peak memory. Introduce no gradient-closeness cutoff.
Float32 need not fail on this first training batch. The original both-precision preflight remains FAIL;
this new check validates the repair at one training step, not its accuracy or complete training trajectory.
The witness uses the remaining 3,008 node-seconds of the original diagnostic budget, with a 50-minute
request under the unchanged GPU1/CPU4/32-GB exclusive-node policy. Code review precedes execution.
The opt-in production repair is committed as `ab94453fcf881d864ca0c1d3f7c5f9c8de7c0c18`;
implementation and independent review: `reports/codex_4a_dp64_training_impl_2026-09-17.md` and
`reports/codex_4a_dp64_training_review_2026-09-17.md`. Default configurations and banked job identities
are unchanged. This source review does not certify the pending full-step numerical result.
The reviewed witness is submitted as `Dp64TrainStepJob.TWT6VGrjjT2f`, SLURM 1848370, with two
expected artifacts `output/dp64_train_step.{json,txt}`. Implementation and independent harness review:
`reports/codex_4a_dp64_train_step_impl_2026-09-17.md` and
`reports/codex_4a_dp64_train_step_review_2026-09-17.md`. Launch, source fingerprints and verified
allocation: `reports/codex_4a_dp64_train_step_launch_2026-09-17.md`. The completed result follows.

### DP-only training-step result (audited 2026-09-17)

`Dp64TrainStepJob.TWT6VGrjjT2f`, SLURM 1848370, completed with **PASS** on all preregistered
repair checks. The actual first tc100 training batch has 128 utterances, 32,615 valid frames and padded
feature shape 128×392×1024. Both cells share the saved partition/order, cold initialization, epoch-1/step-0
training context, seed 42, epoch RNG seed 3671162496, state, random draws and represented neural inputs.
There is no optimizer or learned-parameter update. Main and sequential ±0.25 rate calls each cover the
same 32,615 valid frames; every float64 call conserves mass, worst error 5.1736e-14, with all partitions
finite and no nonfinite rows or zero partitions. Float32 also passes on this batch, worst error 0.0009415;
the earlier long-batch failure remains unchanged.

The float64 total loss is 3.0786665023 versus float32 3.0838878155. All produced gradients are finite;
float64 theta/phi group norms are 12.59786336/0.02787474, with live main-q and segment-score gradient
paths. Paired theta/phi gradient cosines are 0.9999967795/0.9999999820, descriptive only. This validates
the repair at one initial training step; it establishes neither full-trajectory stability, PER improvement
nor the cause of historical collapse.

Peak allocated GPU memory is 17.14 GiB for float64 versus 9.43 GiB for float32. The recorded forward
times include diagnostic capture/hash/transfer overhead and are not a training-throughput benchmark.
This run used 170 exclusive-node seconds; the completed numerical diagnostics total 762/3,600 seconds.
Terminal evidence: `reports/codex_4a_dp64_train_step_terminal_2026-09-17.md`; independent audit:
`reports/codex_4a_dp64_train_step_endpoint_audit_2026-09-17.md`; complete scalar, gradient and provenance
data: `work/analysis/dp64_train_step/Dp64TrainStepJob.TWT6VGrjjT2f/output/dp64_train_step.{json,txt}`.

### S3d epoch-4 result and early stop (audited 2026-09-17)

New user instruction: stop S3d after subepoch 4 if it does not work. Its completed fixed-epoch read
fails the unchanged G4a.3 PER clause. Training had already reached epoch 7 when this instruction
arrived; the exact training allocation `1853056_1` was cancelled after 3h40m07s and its manager stopped.
Checkpoints 1–6 and completed evaluations remain in `ReturnnTrainingJob.Pso7oeIpqYjY`. There is no
epoch 8 or eight-epoch-selected result; this user-directed schedule amendment does not alter the gate.

The stored full-split dev-other PER is 0.8301678184 (2864 utterances, S84713/D59509/I2946/N177275),
versus DP64 control 0.8281765618. Paired pooled delta is +0.001991, with 95% speaker-clustered interval
[-0.002240,+0.005766] across 33 speakers. Dev-clean PER is 0.816736 versus 0.813689;
paired delta +0.003047, interval [-0.000053,+0.005964] across 2703 utterances/40 speakers.
The own-phi dev-other gap is positive: pooled +1.544069/frame; utterance mean +1.342968 with 95%
speaker-clustered interval [+1.235555,+1.456392], 500/500 items and 33 speakers. The gap clause passes,
but neither it nor the small PER differences establish cold take-off.

Concrete results: `output/exp2025_11_06_speech_llms/librispeech/sae_4a_s3d_phone_content/ep4/`
contains both splits' `per.json`, `per.txt` and derangement reads; sibling
`paired_per/ep4/{dev-other,dev-clean}/summary.txt` contains the matched DP64 contrasts.
Independent audit `reports/codex_4a_s3d_epoch4_result_audit_2026-09-17.md` re-derives the counts,
verifies identical cohorts/decoding and the matched config, and confirms G4a.3 FAIL. The opposite-sign
utterance-macro delta is a separate estimand and does not change the predefined pooled-PER gate.
The auxiliary is a
frame-aligned reconstruction channel through the final categorical phone outputs. It can encourage
acoustic information but supplies no phoneme identity; the failed result does not establish which
degeneracy caused failure. No replacement auxiliary or §4b experiment is launched on this reading.

### Precision-only cold control (specified before training)

Under the user's standing instruction to adapt and execute autonomously within cold-start §4a, the next
single-arm slot is a precision-only canonical control. Its only change from banked `lam3_tri` is
`lattice_float64=True`, plus the output path. Retain the exact tc100 data, splits/order, K500 observations,
features, eta, full trigram, flat recognizer/random reverse initialization, seed, optimizer, all loss weights,
rate finite differences, tau schedule, eight subepochs and checkpoints 1–8. No content auxiliary or
S3c input masking is enabled. Use the same clean decoding/evaluation procedures as the banked control.

Register full dev-other/dev-clean greedy PER and statistics, epoch-4/8 own-phi speaker derangement and
paired PER against the banked float32 control (2,000 seed-0 speaker resamples). Keep the label-free
selector unchanged and report fixed epoch 4, selected checkpoint and epoch 8 separately. G4a.3 remains
epoch-4 dev-other PER <0.50 and positive own-phi speaker-matched derangement. No gradient agreement,
auxiliary loss or numerical validity check substitutes for this ASR gate. The contrast measures the effect
of the numerical repair through training at one reference seed. Its float64 endpoint can subsequently
serve as S3d's matched control; no content-arm allocation is included in this run.

Allocate one reference training arm: GPU1, CPU16, host memory 64 GB, gpu_mem 96 GB, six-hour limit,
unchanged whole-node exclusivity, and the existing evaluation-job resources. This is separate from the
closed one-node-hour numerical diagnostic budget, with no grid or additional training arms. Resource
constants come from the existing S3c/rate helper; the phase documents contain no overall cumulative cap
(`reports/codex_4a_budget_scope_2026-09-17.md`). Independent direction and exact-config reviews precede
launch; old configurations/defaults must remain unchanged and the full graph must contain only this new
training and intended reads.
The prepared control is `ReturnnTrainingJob.IoCQmrlJbCC0`, wrapper
`config/sae_4a_dp64_control.py`. Independent scientific and exact-config reviews passed:
`reports/codex_4a_dp64_control_direction_audit_2026-09-17.md` and
`reports/codex_4a_dp64_control_review_2026-09-17.md`; inputs, generated config and endpoint pointers:
`reports/codex_4a_dp64_control_impl_2026-09-17.md`. Paired intervals are registered for epochs 4/8;
if label-free selection picks epoch 5–7, report its PER separately without attributing an epoch-4/8
paired interval to that selected checkpoint.
Recipe commit: `6f16d2fa42dd2f163775f490b53338290c953c41`. The manager is launched for this exact
126-root graph; initial handoff found local create_files, with scheduler submission not yet observed.
Launch/source/resource evidence: `reports/codex_4a_dp64_control_launch_2026-09-17.md`.

### Precision-only control result (audited 2026-09-17)

`ReturnnTrainingJob.IoCQmrlJbCC0` completed all eight subepochs and all 126 registered outputs.
The actual training allocation used 16,999 node-seconds (4 h 43 m 19 s) within its six-hour envelope.
Terminal artifact verification: `reports/codex_4a_dp64_control_terminal_2026-09-17.md`; numerical
extraction and full curves: `reports/codex_4a_dp64_control_extract_2026-09-17.md`; independent endpoint
audit: `reports/codex_4a_dp64_control_endpoint_audit_2026-09-17.md`.
The comparison preserves the specified tc100 inputs, flat/random initialization, reference seed,
trigram and losses/schedule. The complete rendered-config delta is only `lattice_float64=True`
and the model output path. Scores below are full-split greedy SIL-dropped PER fractions; paired
intervals use the unchanged 2,000 seed-0 speaker resamples.

| Subepoch | Split | Banked FP32 PER | DP64 PER | DP64 minus FP32 [95% speaker CI] |
|---|---|---:|---:|---|
| 4 | dev-other | 0.845398 | 0.828177 | -0.017222 [-0.021240, -0.013307] |
| 4 | dev-clean | 0.830090 | 0.813689 | -0.016401 [-0.018782, -0.013999] |
| 8 | dev-other | 0.877185 | 0.860640 | -0.016545 [-0.021145, -0.011431] |
| 8 | dev-clean | 0.860910 | 0.835952 | -0.024958 [-0.027082, -0.022878] |

Pairing covers 2,864 dev-other utterances/33 speakers/177,275 reference phones and 2,703 dev-clean
utterances/40 speakers/193,644 phones. The unchanged label-free selector
`UnsupervisedCheckpointSelectionJob.3N1a52Q3F9fX` selects subepoch 4; the selected result therefore
has the same registered paired interval as that fixed endpoint. Subepochs 5–7 are descriptive reads,
not gold-selected alternatives. Concrete evaluation-job paths are in the extraction report.

**G4a.3 FAIL:** fixed subepoch-4 dev-other PER remains above 0.50. Its own-phi same-speaker gap is
positive on 500 utterances/33 speakers: pooled frame-weighted +1.911260, and separate utterance-mean
+1.749149 with speaker-bootstrap CI [1.656558, 1.850135]. At subepoch 8 these are +3.227839 and
+3.095246 [2.990628, 3.208321]. These gap intervals use 10,000 seed-42 resamples and belong to the
utterance-mean statistic, not the pooled frame-weighted point; do not interchange the estimands.

The precision intervention improves PER at this one reference seed, but does not produce cold take-off.
In the user's stated interpretation, small movements within this high-error regime are diagnostic rather
than evidence of successful initialization. The defect warranted correction; this result does not make
precision a sufficient explanation of collapse or isolate LM weakness from acoustic/reverse-model weakness.
S3d continues unchanged against this completed matched control; its previously pending epoch-8 control
hypotheses now exist. No additional training arm is authorized by this endpoint alone.

### Long-context prior option (frozen diagnostic authorized 2026-09-17)

The user asks how to add linguistic context without the dense history-state growth of higher n-grams.
The completed DP64 control and active S3d run remain the comparison. A candidate-based extension could use the current lattice
to propose phone strings, score them with a longer-context text LM, and sum alignments/segmentations
exactly within each retained string. The outer sum over strings would be approximate. The user has
authorized implementing and executing the frozen diagnostic; no approximate training arm is launched.
For a complete valid lattice path h, write y(h) for its emitted phone string, pi(h) for its frame
labels, sigma(h) for its reverse segmentation, and
`A_tau(y) = sum_{h:y(h)=y} exp((log q_theta(pi(h)|x) + log p_phi(z,sigma(h)|y,eta))/tau)`.
A finite-candidate objective is `-log sum_{y in Y} A_tau(y) * P_context(y)^(beta/tau)`.
If instead sampling from the original trigram posterior, replacement of the prior requires the ratio
`exp((beta/tau) * (log P_context(y) - log P_trigram(y)))`, with the corresponding sampling estimator;
simply multiplying by the new LM is a different objective. Candidate enumeration and Monte Carlo
estimation are distinct approximations. The existing SIL/band/temperature semantics must be preserved
(see `SAE_ref.md`); ordinary CTC likelihood times a reverse-only likelihood is not this joint marginal.

Initial feasibility read: `reports/codex_4a_context_rescoring_code_2026-09-17.md` identified the missing
complete-path sampler and constrained joint scorer, now implemented for this frozen pilot. Before any
training proposal, a frozen-checkpoint diagnostic must measure candidate coverage/diversity, weight
concentration and cost. A stronger LM cannot restore sequences absent from its proposal, and any
training comparison must control the inference approximation as well as LM choice. Full-text evidence
in `reports/codex_4a_lm_identification_literature_2026-09-17.md` motivates this coverage-first question:
[Nuhn et al., ACL 2013](https://aclanthology.org/P13-1154.pdf) show capacity and search interacting in
fixed-symbol decipherment; [Kumar et al., ASRU 2017](https://arxiv.org/pdf/1711.05448) demonstrate a
candidate-coverage limitation in supervised LM rescoring. Neither validates this cold-ASR training route.
Existing BT is a complementary way to use full unpaired sentences while retaining the main exact DP;
it is not insertion of a full-context prior into the real-speech posterior.

Provisional response to the user's candidate-budget question: at the fixed epoch-4 diagnostic point
(target tau=2), draw at most 256 complete legal joint paths, 192 from the current joint trigram posterior
Q_tau and 64 from Q_(1.5*tau). The 256/3:1/1.5 choices are proposed engineering defaults, not measured
optima or reference-setup constants. Sample via forward-filtering/backward-sampling, tempering all
joint terms in the hotter proposal; retain the target tau for rescoring. Deduplicate exact emitted
strings including SIL. This gives at most 256 unique strings, not a guarantee of that count. Use the
same set for trigram, reconstructed 4-gram and proposed 6-gram scoring (five previous phones), preserving
the frozen unpaired corpus, split, SIL convention, recursive Witten-Bell smoothing, BOS and no-EOS
semantics. P4=7.03 was an on-the-fly analysis; no fitted P4/P6 artifact existed at registration. The pilot
implements sparse count/scoring code, not a dense six-gram tensor. Provenance:
`reports/codex_4a_context_lm_assets_2026-09-17.md`.

Enumerate the unique finite set with prior-free A_tau(y) and the replacement LM above; neither draw
multiplicity nor an importance correction belongs in this finite-set enumeration. Nested 64/128/256
draw sets must each preserve the 3:1 mixture. Report unique count/diversity, target weight concentration,
runtime/memory, and exact retained trigram mass Z3(Y)/Z3(full). P4/P6 full denominators are unavailable:
ESS or stable doubling is not evidence of their coverage. Training would additionally require
constrained conditional forward-backward/derivatives and candidate-weighted frame/segment marginals;
sampled alignments alone do not implement the proposed exact inner sum. Astra review:
`reports/codex_4a_candidate_sampling_spec_review_2026-09-17.md`. Training adequacy is not established by
these defaults. The user has now authorized their frozen diagnostic implementation and execution.

**Execution registration (before results).** Freeze the completed DP64 control's own epoch-4 theta,
phi and inputs in `ReturnnTrainingJob.IoCQmrlJbCC0`; this is an explicit operating-point choice for the
new diagnostic, rather than the earlier float32 checkpoint. Read no gold. Reuse the existing 300-tag
dev-other strided diagnostic selection, derived solely from the 2,864 HDF sequence tags; process its
first eight tags in fixed order. Eight is a new root-chosen cost-pilot cap, not a corpus-wide sample
claim. Record IDs before scoring. Seed42 follows the reference convention; nested budgets and the
3:1/hot1.5 choices above follow the proposal approved by the user. Use DP64, tau2, beta1, band25 and
all remaining saved model/input constants. Keep the production prior table and exact casting order
for P3/full-partition comparisons; quantify reconstructed-prior parity separately.

One pilot allocation is capped at the existing GPU1/CPU16/mem64/time6h/gpu_mem96 resource envelope
(an exclusive four-GH200 node). The new job registers metadata, pinned sparse LM counts, per-item
candidates/scores and summary/timing outputs. No optimizer, training update, gold-scored selection,
full-300 expansion or sparse-autoencoder work is part of this pilot. Astra owns all complex code.

Pilot completion gate: all eight items produce the registered finite scores, candidate records,
trigram retained-mass and per-order weight/convergence diagnostics within that allocation, after
small exhaustive topology/partition checks and frozen-P3 parity pass. Numerical tolerances are fixed
in the implementation/review before the run and must reflect DP64 and the stored-prior precision.
An incomplete, invalid or over-budget pilot cannot license a training comparison. A completed pilot
is a feasibility measurement, not an ASR or coverage PASS: inspect actual coverage, concentration,
hypotheses and cost, then preregister any further comparison. Neither this pilot nor small numerical
tests close G4a.3. The same-candidate trigram is the approximation control for any future LM comparison.

Implementation and preregistered checks: Astra's new `sae/emc/candidates.py` and exhaustive tests are
committed as `a5fe6d5` in the speech checkout. The frozen driver/LM/job are
`analysis/context_rescore.py`, `analysis/context_lm.py`, `analysis/context_rescore_diagnostic.py`, with
wrapper `config/sae_4a_context_rescore.py`. Reports:
`reports/codex_4a_context_lattice_impl_2026-09-17.md`,
`reports/codex_4a_context_rescore_impl_2026-09-17.md` and its referenced source hashes. Tiny exhaustive
joint/fixed-string partitions use tolerance 1e-11; full-corpus P3 parity requires maximum absolute
float64 log error <=1e-12 and equality after the production float32 cast, before any candidate scoring.
Retained P3 log mass must be <=1e-8, a numerical bound check rather than an adequacy threshold.
Core support/partition and 24,000 sampled-path checks passed; they establish only the exercised small
lattices. Corpus parity, coverage and cost remain unmeasured until execution. Fresh protocol audit:
`reports/codex_4a_context_pilot_protocol_audit_2026-09-17.md`; final Astra code review:
`reports/codex_4a_context_rescore_code_review_2026-09-17.md`, no unresolved findings.

Pilot submitted as `analysis/context_rescore_diagnostic/ContextRescoreDiagnosticJob.gIPIR0c49NfS`,
SLURM `1855413_1`. Registered job outputs under `output/pilot/` are `summary.json`, `per_item.jsonl`,
`candidates.jsonl`, `metadata.json` and `lm_counts.npz`. Wrapper SHA256 is
`448845acffd1913174d090081583ed1c00d941eb4a36c33e96a55f8f4043dd44`.
Launch evidence: `reports/codex_4a_context_rescore_launch_2026-09-17.md`. The completed measurement is below.

### Six-gram cold-training authorization (2026-09-17; implementation in progress)

The user explicitly authorizes launching six-gram reweighting without waiting for S3d. This amends
the post-pilot proposal to improve candidate generation before training; the measured coverage limits
remain limitations of the new experiment. No claim that the pilot established training adequacy is made.
Stay within the existing cycle, with a fresh cold initialization and no S3d content term, back-translation,
denoising, supervised initializer or §4b execution.

Compare a finite-candidate P6 arm with a finite-candidate P3 control and the completed exact-P3 DP64
reference `ReturnnTrainingJob.IoCQmrlJbCC0`. Both new arms inherit its data, initialization, seed,
optimizer, eight-subepoch schedule, temperature anneal, band, beta=1 and loss normalization. For each
utterance at the current temperature tau, draw 192 complete paths from the current joint P3 posterior
and 64 at 1.5*tau, then deduplicate exact emitted strings including SIL. These counts and temperatures
extend the user-approved pilot settings to training; candidate generation is detached. Each arm uses
its own evolving theta/phi; identical sampling rules do not imply identical candidate sets after updates.
Reuse the pilot's frozen sparse counts, corpus/split and Witten-Bell/BOS/no-EOS convention, without refit.

For the resulting set Y, replace the cycle partition with
`Z_n(Y) = sum_{y in Y} A_tau(y) * P_n(y)^(beta/tau)`, n=3 or 6, and retain
`L_cycle = mean_utt[-log Z_n(Y)/T]`. Exact conditional alignment/segmentation derivatives train both
theta and phi; there is no outer tau multiplier, sampled-alignment surrogate, duplicate-frequency
weight or importance-sampling correction. This is a truncated partition, not an unbiased estimate of
the full P6 partition. The weight-0.1 aggregate penalty remains unchanged. The weight-3 rate term uses
the same candidate posterior and non-SIL token count, preserving rho and the existing theta-only
central finite-difference rule (step 0.25, phi detached); its precise implementation must be reviewed.

Before a full launch, tiny exhaustive partition/gradient checks and an actual forward/backward timing
read must establish executable cost and finite gradients. Use the existing resource envelope, with
matched arms packed if supported; no parameter grid. The original G4a.3 gate is unchanged: fixed
subepoch 4 dev-other PER<0.50 and positive own-phi speaker-matched gap. Report paired comparisons,
fixed epoch 4, epoch 8 and the unchanged label-free checkpoint selector separately. Candidate coverage
and weight concentration explain approximation behavior; they do not substitute for the ASR gate.
Run pointers, measured cost and the final bounded allocation are recorded before submission below.

Cost pre-registration: the completed DP64 control used 477 updates, 16,079s of training
(33.71s/update), and 16,999s of exclusive-node allocation. First run one source-reviewed, real-batch
forward/backward measurement, capped at one node-hour using GPU1/CPU16/mem64/gpu_mem96.
The existing two-arm packed job requests GPU4/CPU64/mem256 and 6.6h (the reference 6h multiplied by
its existing 1.1 packing allowance). Launch that comparison only if measured cost supports this bound;
do not silently change batch size, candidate count, training data or schedule to fit it. The allocation
policy reserves the full four-GH200 node even for the one-GPU measurement. Source of measured baseline
and settings policy: `reports/codex_4a_context_train_preflight_2026-09-17.md`.

The conditional backward implementation is speech-repo commit `ae2e965`; its exact objective,
gradient normalization, finite-difference semantics and tiny exhaustive checks are documented in
`reports/codex_4a_context_train_lattice_2026-09-17.md`. These checks establish their tested mathematical
cases, not production speed or a cold-start result. Actual-batch cost remains the launch prerequisite.

The integrated implementation is speech-repo commit `72d1ac50`; rendered-config comparisons are in
`reports/codex_4a_context_train_integration_2026-09-17.md`. Astra source review
`reports/codex_4a_context_training_review_2026-09-17.md` releases the bounded measurement only;
fresh protocol audit `reports/codex_4a_context_training_protocol_audit_2026-09-17.md` limits any
subsequent inference to the candidate-training method, not an exact P6 partition. The prepared
two-arm pack is `PackedEmcTrainJob.l36lBHeAK40X`, wrapper `config/sae_4a_context_train.py`,
with eight-subepoch P3/P6 arms and 260 registered roots; it has not been submitted.

Actual-batch measurement submitted as `ContextTrainProfileJob.VglVxZtlgpI0` / SLURM `1856748_1`,
wrapper `config/sae_4a_context_train_profile.py`. Each order uses the canonical first epoch-1 training
batch and cold model/RNG at tau8 (hot proposal tau12), with full forward and neural backward,
no optimizer update and no trained checkpoint. It checks finite gradients, the rate's phi stop-gradient,
initialization/RNG identity, and records stage times and memory. Both `output/profile.json` and
`output/profile.txt` plus successful Sisyphus completion are required. Its time projection excludes
optimizer/CV overhead and does not certify all-epoch throughput. Launch evidence:
`reports/codex_4a_context_train_profile_launch_2026-09-17.md`.

### Candidate training cost read and 512-draw amendment (2026-09-17)

The 256-draw profile was stopped deliberately after its first completed batch established that the
implementation cannot support the registered training allocation. Its overall completion gate did
not pass: P3 completed, P6 remained partial, and no final `profile.txt` exists. The exact job
`ContextTrainProfileJob.VglVxZtlgpI0` / SLURM `1856748_1` was cancelled after 39m40s. Evidence was
preserved before cancellation in `reports/codex_4a_context_train_profile_partial_2026-09-17.json`;
terminal state and literal extraction: `reports/codex_4a_context_train_profile_early_stop_2026-09-17.md`.

Measured operating point: canonical cold epoch 1/step 0, tau 8/hot 12, 128 utterances, 32,615 real frames,
50,176 padded frames and max T=392. All 128 P3 utterances retained 256 unique strings. Full forward plus
neural backward took 1,703.300s: proposal FFBS 272.930s, LM scoring 1.040s, conditional lattice forward/
backward 1,421.618s; the final neural backward took 0.837s. Stage sums exclude other forward overhead.
Both theta and phi had finite nonzero gradients; the rate term had no phi gradient; peak allocated
GPU memory was 5.187 GiB. These are one-batch implementation results, not ASR or optimization progress.
The preserved P6 snapshot covers 71/128 utterances and has no completed step time or gradient verdict.

The exact-P3 control's 33.71s/update is its eight-epoch average, not a paired measurement of this same
batch. Multiplying the measured candidate first-batch time by 477 gives 812,474s; this is a screening
projection, not measured training duration, and greatly exceeds the 23,760s packed-node cap.
The old conditional recurrence processes utterances serially, replays checkpoints, and rebuilds
per-frame autograd for three terminal adjoints. Its extra exact per-string marginalization also does
real additional work. LM lookup accounts for little of the measured cost. No claim that batching alone
will recover the required speed is made.

The user's subsequent "start 512 directly" changes the intended candidate budget to 512 per utterance:
384 target-temperature plus 128 hot 1.5*tau draws, with exact string deduplication and no cross-string
recombination. Apply this to both matched training arms. Keep all other data, initialization, schedule,
finite-set objective and ASR gates unchanged; there is no new candidate-count grid.

The per-frame autograd construction is replaced by explicit log-domain conditional forward–backward,
and both sampling and conditional inference now batch utterances. The full neural batch and loss
normalization remain unchanged. FFBS execution batches start at most 16 and conditional batches at
most 4, shrinking before allocation from actual tensor shapes. Each uses 48 GiB (half the 96 GiB reference
GPU envelope) for its main forward or checkpoint/replay buffers and reserves the other half for
transients and current resident tensors. No OOM retry, filtering or candidate reduction is introduced.
This conservative bound is not measured all-length memory certification; skewed single items may be
rejected before allocation. A metadata-only read finds 28,254 canonical
training tags spanning 70–1,226 frames, with no max-sequence-length filter in the saved config;
`reports/codex_4a_context_batch_input_lengths_2026-09-17.txt`. Execution sizing must account for
that range without filtering data. Batched inverse-CDF sampling must preserve the
proposal distribution and private per-tag RNG; same-seed paths may differ from the old multinomial
implementation, and this change must be disclosed and tested. Astra owns implementation and review.
After parity checks, one actual P6 batch is sufficient for the next cost/gradient screen; do not repeat
P3. The original one-node-hour diagnostic envelope has 20m20s left, so cap that job at 20 minutes.
The full 6.6h training allocation remains conditional on executable cost; no optimizer run is launched.

Implementation: speech-repo commits `c209489` (analytic/batched kernels) and `9a408de3` (512-draw
integration). Reports `reports/codex_4a_context_kernel_cost_2026-09-17.md` and
`reports/codex_4a_context_batch_integration_2026-09-17.md` specify the exact memory formulas and
test scope. CPU comparison against the old kernel reaches max absolute difference 1.776e-15 on the
tested cases. Execution chunk changes preserve sampled Y, losses, gradients and private RNG in the
new sampler's tests. Astra review `reports/codex_4a_context_batched_review_2026-09-17.md` releases
only the bounded profile; no throughput or ASR result follows from these checks.

Submitted corrected profile: `ContextTrainProfileJob.yjsJGPFL3M1M` / SLURM `1857340_1`, one actual
cold P6 batch with 512 draws and the same canonical epoch 1 input, tau 8/hot 12. It requests GPU1,
CPU16, mem64, gpu_mem96 and 20 minutes. Expected outputs are its `output/profile.json` and
`output/profile.txt`; inspect status and gradient/cost fields as well as successful job completion.
Launch evidence: `reports/codex_4a_context_512_profile_launch_2026-09-17.md`. The prepared
matched 512-draw P3/P6 training pack is now `PackedEmcTrainJob.CNcm2jEmCouC` (260 roots), superseding
the unsubmitted 256-draw pack. It remains unsubmitted pending the actual cost/memory read.

The corrected 512-draw profile subsequently completed successfully, with both registered outputs and
SLURM COMPLETED/0:0, before the requested stop could act. Its allocation lasted 6m18s. Actual P6
forward plus neural backward was 260.004830s (259.116458s forward, 0.888372s backward), with finite
nonzero theta/phi gradients and peak allocated memory 22.9543 GiB. Batch, temperature and data match
the operating point above; the old measured arm was P3 with 256 draws, so the two timings do not isolate
a single implementation delta. The new stage timers are 14.613s for proposal FFBS, 28.427s for P6 LM
scoring and 208.713s for conditional inference; the conditional stage still dominates.
Multiplying the new first-batch time by 477 gives 124,022s, still over
the 23,760s packed-node cap; this is a cost screen, not a measured training runtime. Result evidence:
`reports/codex_4a_context_512_profile_result_2026-09-17.md` and the job's `output/profile.json`.

The user required checking 256-draw viability first; this superseded the earlier direct-512 launch
order. At that decision no rewritten 256 timing had been measured. The next profile was one actual P6 batch at
256 draws (192 target, 64 hot), without an optimizer update. Only the profile's proposal budget and
allocation cap change; the prepared training default remains 512 and unsubmitted. The two completed
profiles consumed 45m58s of the original one-node-hour diagnostic allocation, leaving 14m02s; cap
this measurement at 14 minutes. No full training is released by correctness checks or finite gradients.
Submitted job: `ContextTrainProfileJob.1y34VroYbhyf` / SLURM `1857433_1`, same profile wrapper, two registered outputs
`output/profile.json` and `output/profile.txt`. The isolated profile override and proposal-count check
are documented in `reports/codex_4a_context_256_profile_amendment_2026-09-17.md`; actual timing is
reported below. No model, data, optimizer, temperature, kernel or execution-chunk setting changed.
Independent release: `reports/codex_4a_context_256_profile_review_2026-09-17.md`; launch evidence:
`reports/codex_4a_context_256_profile_launch_2026-09-17.md`.

The rewritten 256-draw profile completed with SLURM COMPLETED/0:0, Sisyphus finished markers and both
registered outputs. The same cold P6 batch (B128, max T392, 32,615 real/50,176 padded frames, tau 8/hot 12)
retained 256 unique strings for every utterance. Measured forward/backward was 143.953577s (142.969254s
forward, 0.984323s neural backward), peak allocation 12.754991 GiB. Stage timers: sampling 14.601397s,
P6 LM scoring 14.104947s, conditional inference 108.114553s. Theta/phi gradients were finite and
nonzero, rate phi gradient absent, and parameters unchanged; no optimizer step ran. Result and exact
artifacts: `reports/codex_4a_context_256_profile_result_2026-09-17.md`.

Multiplying this first-batch time by 477 gives 68,665.856s (19.074h), above the unchanged 6.6h allocation.
This is a cost-screen failure, not a measured whole-training duration or an ASR result. Its 4m24s
allocation brings diagnostic use to 50m22s, leaving 9m38s of the original node-hour.

The next bounded implementation check keeps the same 256-draw P6 objective and data. Current execution
caps are 16 utterances for sampling and four for conditional inference. In the profile only, set both
upper bounds to the actual loaded neural batch size; the existing shape/resident-memory checks still
shrink each execution group before allocation, within the same 48+48 GiB reference envelope. This tests
execution fragmentation without reducing candidates or changing the learner batch. It does not assume
speedup or all-length memory safety. Require sampled-Y/loss/gradient parity and independent code review
before one actual batch, capped at nine minutes. Tracked training defaults remain unchanged and full
training stays unsubmitted.

Submitted job `ContextTrainProfileJob.2ZXMPEhUGLz8` / SLURM `1857703_1` retains the same two profile output names and
GPU1/CPU16/mem64/gpu_mem96 request, with a nine-minute cap. A 17-tag synthetic fixture crossed both
previous execution limits and matched sampled Y exactly and losses/gradients within 1e-11, including
forced preallocation splitting and restoration of RNG/module defaults. This verifies the tested wiring,
not GPU performance. The source delta, frozen prechange script, hashes and graph are in
`reports/codex_4a_context_memory_batched_profile_2026-09-17.md`.
Independent code review released this profile only:
`reports/codex_4a_context_memory_batched_review_2026-09-17.md`. Larger CUDA execution groups still
require the measured cost/memory read; this release does not approve full training.
Launch evidence: `reports/codex_4a_context_memory_batched_launch_2026-09-17.md`.

### Higher-context pilot result (audited 2026-09-17)

The registered eight-item pilot is complete: `ContextRescoreDiagnosticJob.gIPIR0c49NfS`, SLURM
`1855413_1` COMPLETED/0:0, with finished markers and all five registered outputs. Completion evidence:
`reports/codex_4a_context_rescore_handoff_diagnosis_2026-09-17.md`; literal extraction:
`reports/codex_4a_context_pilot_extract_2026-09-17.md`; fresh result audit:
`reports/codex_4a_context_pilot_result_audit_2026-09-17.md`. The pilot completion gate passes unchanged.

Measured operating point is the registered DP64 control epoch4/step238, beta1, target tau2, hot tau3,
band25, seed42, fixed first eight HDF-stride tags. All eight belong to speaker116; this is a cost and
coverage pilot, not a representative speaker or full-split evaluation. Sparse counts use exactly
1,000,000 train lines/91,100,286 phones and 10,000 held lines/912,142 phones. Reconstructed P3 has
zero maximum float64 log error; production cast bit equality passes. Corpus parity is therefore
measured, rather than inferred from the small fixtures. The audit reconstructed all72 per-item,
budget/order finite log partitions from the saved candidates to maximum error 6.83e-13.

Equal-utterance means (ESS and maximum weight are normalized within each utterance's retained set):

| Draws per item | Unique strings, summed over8 | Retained full-P3 mass | P3 ESS | P4 ESS | P6 ESS | P6 maximum weight |
|---|---:|---:|---:|---:|---:|---:|
|64|437|0.27903078|7.80479|3.58900|2.58345|0.67698036|
|128|824|0.34203705|11.54036|3.89546|3.42870|0.60125984|
|256|1525|0.39339895|17.28447|5.16134|3.87624|0.54186986|

The mean masks a major coverage difference. At256 draws, `116-288045-0000` (532 frames) retains
3.49797e-8 of known P3 mass; `116-288047-0012` (271 frames) retains0.000394086. The other six retain
0.163976–0.821112. On the longest item, P6 assigns0.964417 of its retained-set weight to one string
(ESS1.07421). These are exact finite-set/P3-denominator measurements, not P4/P6 coverage estimates.

Actual strings were read for all eight items at256. Winners differ on6/8 for P4 versus P3, on8/8 for
P6 versus P3 and on7/8 for P6 versus P4, through phone substitutions/insertions and SIL changes.
For `116-288045-0009`, the P3 fragment
`R AE N G EH` becomes `R AE N D G EH` under P6. Both P3 and P6 retain `K AE N D Z K AE N D Z` in
`116-288045-0019`. No gold was opened: these changes show a material ranking effect, not correctness
or escape from the cold-start failure. Full strings and per-item scores are in the extraction report.

Worker runtime was about90 seconds, allocation2m31s. LM counting/parity/pinning took49.17s; peak
process RSS was about2.0GB and GPU allocation0.83GB. This establishes feasibility at these eight
observed lengths, not training throughput. Sampled latent paths were not saved individually; their
legality is supported by the preregistered implementation tests, not replayable from result files.

Decision: the higher-order prior affects ranking, but256 whole-string draws are not established as an
adequate training approximation. Known baseline coverage is particularly poor on the two longer items,
and stronger-prior coverage remains unknown. Preserve this finite-set diagnostic as the baseline;
prepare an LM-guided candidate-generation comparison, using the new prior during proposal construction
and retaining the exact final fixed-string scorer and same-candidate P3 control. Freeze its method,
comparison and spend gate before execution; no training or further sampling job is launched by this
result. Existing full-text search/LM evidence remains in
`reports/codex_4a_lm_identification_literature_2026-09-17.md`. G4a.3 and S3d's matched comparison remain
unchanged, and §4b stays registered without execution.

Preparation for that follow-up: the minimal proposed direction is a pruned search over legal joint
transitions with P6 used when extending phone prefixes, followed by the existing exact A2 scorer for
completed distinct strings. A P3-guided search under the same pruning rule is the proposal control;
score each candidate set and their union with the same P3/P4/P6 models. Report actual search cost as
well as output count; equal beam width and equal path-draw count do not imply equal computation.
The beam/state bookkeeping, budget and gate are still to be fixed before a new job.

A union supplies a useful diagnostic despite the unavailable full P6 partition: the fraction of full
P6 mass in the old set is at most `Z6(Y_old) / Z6(Y_old union Y_new)`. A small ratio can establish that
the old set misses much P6 mass; a ratio near one cannot establish adequate coverage. New candidates
must be generated without gold. Full-text follow-up:
`reports/codex_4a_context_proposal_followup_literature_2026-09-17.md`. The verified Nuhn et al. beam
result motivates using context during search while warning against tight pruning; the Kumar et al.
lattice result supports broader candidate structures. [Lindsten et al., JMLR2014](https://jmlr.org/papers/volume15/lindsten14a/lindsten14a.pdf) adds particle-history
degeneracy as a reason to defer particle/block machinery until its conditioning and weights are specified.
These analogies do not establish efficacy for this cold-ASR cycle.

The additional text-only read is complete and included in the result audit above. It uses frozen
pilot counts on the same10,000 held lines/912,142 phones, raw float64 WB and the existing BOS/no-EOS
convention, with no refit, tuning, audio or gold. P3/P4/P6 per-phone NLLs are2.248017/1.950871/1.666830;
perplexities are9.468940/7.034812/5.295356. P3/P4 match the banked text analysis within1.1e-13.
Thus P6 improves measured held-text prediction, separately from its candidate-ranking effect;
this establishes neither acoustic recognition quality nor coverage of its posterior.
Helper `analysis/context_lm_heldout.py` was reviewed in
`reports/codex_4a_context_lm_heldout_review_2026-09-17.md`; outputs are
`reports/codex_4a_context_lm_heldout_2026-09-17.json` and
`reports/codex_4a_context_lm_heldout_run_2026-09-17.md`. CPU-only runtime19.46s, peak RSS544,128KiB;
no additional scheduler or GPU allocation was used.

### Sparse acoustic-code follow-up

By explicit user request, the preliminary plan and literature are now owned by the separate phase
`SAE_4B.md`. It is registered with no execution; S3d retains its MFCC target and registered gate.

### Recognizer-factor diagnostic (preregistered 2026-09-17)

Question: at the fixed canonical cold `lam3_tri` epoch-4 checkpoint, how does the recognizer's current
relative path weighting affect the reconstruction term's phone targets? Read the same checkpoint twice:
the original lattice input `log_q` and a same-shaped normalized constant `-log(41)` over all CTC classes.
Keep original recognizer outputs for every recognizer metric. Freeze phi, the full trigram, eta, observations,
lengths, blank/repeat support, duration model, tau=2, alpha=0, band=25, batching and all other inputs.
The latter input adds the same `-T*log(41)/tau` to every valid fixed-length path, removing q preferences
while retaining CTC path multiplicity. This is the **untilted main L_tau posterior**, not the aggregate/rate
terms or the complete optimizer gradient. Neither checkpoint is retrained.

Use the existing cold mass-profile protocol: 300 deterministically stride-selected dev-other utterances,
Torch seed 0, MFA frame centers `(t+0.5)/50`, row-sum tolerance 0.007 without posterior renormalization,
the existing boundary tolerances, and 20 boundary-null draws at seed 12345. The banked protocol measured
96,676 frames and 33 speakers; verify the new read's actual
counts rather than assume them. Source, sampling and resource references:
`reports/codex_4a_qneutral_scope_2026-09-17.md`. This sampled diagnostic is separate from full-split PER.
The two passes share one analysis job with the existing `OrMassDiagnosticJob` allocation: requested
GPU=1, CPU=4, host memory=32 GB, time=1 hour. Unchanged booster exclusivity reserves one four-GPU node;
only one GPU is used. The bounded diagnostic budget is therefore at most one exclusive node-hour,
with no new training. Reference resource provenance: `analysis/or_mass_diagnostic.py`.

Primary descriptive contrast: per-utterance correct-phone target mass divided by total target mass on
the 39 real phones, summed over gold non-SIL speech frames before dividing. Exclude blank and SIL from
the denominator. Compare neutral minus original using paired speaker bootstrap, 2,000 draws, seed 0;
also report pooled raw gold/blank/SIL/other mass, scored-frame counts, target-mode PER/rate and the
existing duration/boundary profiles. Original recognizer metrics must agree between passes. Gold remains
quarantined to this diagnostic; it provides no initializer, training map or checkpoint selection.

Interpretation is deliberately limited to this fixed checkpoint and combined phi/LM evidence. Higher raw
gold mass alone can arise from changed emission/blank behavior; even an identity-mass increase does not
separate acoustic evidence from phone-prior effects or prove that retraining would improve ASR. A null
cannot exclude earlier q/phi coadaptation. This read has no new take-off gate and does not amend G4a.3.
The literature distinguishes a recognition proposal from a separately defined generative posterior
([Bornschein and Bengio, ICLR 2015](https://arxiv.org/pdf/1406.2751)); SAE's tempered cycle is not that model.
Actually training from neutral-q targets would change the objective and require a separately reviewed
student loss. Verified equations and limits: `reports/codex_4a_qfactor_literature_2026-09-17.md`.
The reviewed implementation and source fingerprints are recorded in
`reports/codex_4a_qneutral_impl_2026-09-17.md`; pre-compute review:
`reports/codex_4a_qneutral_code_review_2026-09-17.md`.

The initial normalized-uniform read is numerically invalid: across the same 300 utterances and 96,676
frames, its posterior row sums range from 0.932070 to 1.039491, outside the preregistered 1±0.007 tolerance.
The original-q pass ranges from 0.994061 to 1.002298 and reproduces the banked profile. Neither pass has
NaN frames or zero-partition utterances. The paired contrast is withheld; these numbers diagnose
evaluation validity, not the effect of removing q. Evidence:
`reports/codex_4a_qneutral_debug_2026-09-17.md` and preserved `output/profiles/` under
`QNeutralDiagnosticJob.LkSVqPhvffwL`. Numerical validation must preserve the posterior mathematics and
tolerance, without post-hoc renormalization. The original one-node-hour diagnostic budget still applies.

The reviewed correction keeps neural evaluation unchanged and promotes all floating lattice inputs to
float64 for both diagnostic arms; the neutral input remains normalized uniform. Before the full profiles,
the first failing existing batch supplies a numerical witness, selected only by probability conservation.
Compare normalized and zero acoustic constants in float32 and float64, with the known partition shift
restored. Both float64 variants must conserve mass and agree on posteriors, expected tokens and restored
log partition to 1e-8; failure stops the read. These are engineering equivalence checks, not a new scientific
gate. The original control is recomputed at the same precision and any difference from its banked float32
profile is disclosed. Exact test order, checks and mathematical derivation:
`reports/codex_4a_qneutral_numerical_design_2026-09-17.md`. The witness and two profiles share the
remaining allocation, capped at 56 minutes after the initial 190 seconds of node usage.
Implementation and source fingerprints: `reports/codex_4a_qneutral_fp64_impl_2026-09-17.md`;
independent pre-compute review: `reports/codex_4a_qneutral_fp64_code_review_2026-09-17.md`.
The new run preserves the failed predecessor and places its numerical witness in
`output/profiles/numerical_witness.json`; only an accepted witness permits the two full profiles and
`output/paired_comparison.{json,txt}`. The completed read is reported below.

### Recognizer-factor result (audited 2026-09-17)

`QNeutralDiagnosticJob.P1EBNHvA8Rxi` completed with all three registered roots. Both arms use the same
300 dev-other utterances, 33 speakers, 96,676 frames (77,333 gold non-SIL speech frames), frozen
`lam3_tri` epoch 4, tau 2, alpha 0, phi, full trigram, eta, support, band 25 and float64 DP. Original
recognizer reports are identical between arms. No model was updated.

| Sampled posterior read | Original q | Neutral q |
|---|---:|---:|
| Mean per-utterance conditional phone-identity mass | 0.076275624 | 0.044485215 |
| Gold-phone mass / all-real-phone mass on gold speech | 3823.109 / 46186.118 | 2080.066 / 45440.102 |
| Target-marginal greedy PER, 18,743 reference phones | 0.824521 | 0.999627 |
| Target-marginal greedy phones/second | 7.238611 | 0.003620 |

The primary neutral-minus-original paired difference is **−0.031790408**, speaker-bootstrap CI95
**[−0.036698462, −0.026740672]**, using the preregistered 2,000 seed-0 resamples. The audit re-summed
the per-utterance masses and checked the CI's code and inputs; it did not independently regenerate
bootstrap draws. Full class-mass, duration and boundary profiles are retained in
`output/profiles/{original,uniform}.json`; paired results are in `output/paired_comparison.{json,txt}`.
These sampled posterior-mode PERs are distinct from full-split recognizer PER and from a MAP path decode.

Neutralizing q loses conditional phone identity and nearly empties the greedy read of the target
marginals at this checkpoint. Thus the current q supplies useful information relative to this frozen
combined phi/LM calculation. This result does not establish acoustic identifiability, identify which
remaining factor causes the weak targets, exclude earlier coadaptation or predict a retraining result.
It provides no evidence for a neutral-q training remedy. The original G4a.3 gate is unchanged.

The numerical witness reproduced the first float32 failure in batch 2: maximum row error 0.067930
with the normalized constant versus 0.000766 with zero acoustic weights. Both float64 representations
conserve mass to 3.71e-12 or better and their required equivalence differences are at most 4.44e-10.
This supports constant-offset-triggered float32 drift in this probe, without attributing historical
training failure to it. Both final profiles pass the unchanged 0.007 tolerance with no NaN frames or
zero partitions. The original arm's mean identity changes by only −4.83e-8 from the preserved float32
read; its numerical differences are disclosed in the paired artifact. No posterior was renormalized.

Terminal verification and budget: `reports/codex_4a_qneutral_fp64_terminal_2026-09-17.md`;
independent scientific audit: `reports/codex_4a_qneutral_fp64_endpoint_audit_2026-09-17.md`.
The two allocations used 436 exclusive-node seconds in total against the 3,600-second diagnostic cap;
the scheduler is drained and the manager exited. No additional experiment is allocated by this result.

## Artifacts

Path-prefix key: `T/ = work/i6_core/returnn/training/`, `S/ = work/speech_llm/sae/`,
`W/ = work/i6_experiments/users/wu/experiments/unsupervised_asr/w2vu2/`.

| item | path |
|---|---|
| units codebook (enc50, K=500) | `S/quantize_states/QuantizeStatesJob.FWpGhC941JMi` |
| frozen raw 50 Hz unit store | `S/quantize_states/PackUnitsJob.I0uzRMfUrKWC` |
| §1d pseudo-labels (init i) | `W/selftrain/GanPseudoLabelJob.xjn6QnNqwEEH` |
| quarantined seed dir (init iii) | `TransformAndMapHuggingFaceDatasetJob.mQmb6aW1IDH5` |
| phonemized text T_phi | `TextToPhonemeJob.THKMON3k9LJQ` |
| MFA gold phones (S1 reads) | `GoldPhonesJob.ZGSp0hxyd2YP` |
| WER decode convention | `W/word_decode/Wav2Vec2KenlmDecodeJob.AQw3EcUo6rks` |

## Verifier feedback

- 2026-09-15 design review (`reports/design_review_4a_2026-09-15.md`): G4a.1 moved from -log R to the refit
  reverse term; S2 arm B anchored; manual backward required; S0 split so S1a precedes the lattice; G4a.2
  quantity, selection pre-registration, G4a.3 timing and the rate revert added; four citations corrected.
