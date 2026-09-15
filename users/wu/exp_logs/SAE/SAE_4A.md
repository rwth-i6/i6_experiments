# SAE §4a — Exact-marginal cycle (EMC): tempered posterior-marginalized training with a semi-Markov reverse model

## State

History (details in the sections below and the named reports): phase registered 2026-09-15 (`reports/lit_method_v2_2026-09-15.md`,
`reports/design_review_4a_2026-09-15.md`); S1a banked, G4a.1 PASS audited; S0b inits built (ctc_init (i) `ReturnnTrainingJob.HcXzd6M2eyVZ`,
seed inits 10 h `65NNK8Bwxdtd` / 1 h `t4K6Z6gHK56e`); S1b/S2 (GAN-init track) retired by the user's direction change, their reads kept in
Results; efficiency rule = loader wait at most a quarter of the step (measured 1.3 % at 1.54 s/step, B 128; the kernel is the cost);
S2b (seed inits, arms A/B/C) and S2c (held anchor D vs self-distillation control E) trained and read; S3 cold start CLOSED FAIL (G4a.3);
decoder fixed at `decoder_version = 2` (run-collapsed emissions; every v1 WER void).
NEXT (2026-09-15, late; USER DIRECTION: cold start is the target — no GAN init, no seed init as the deliverable; S2b/S2c seed tracks are diagnostic beds only; S3 CLOSED FAIL is the problem to solve): (1) Cold-start framing from reads (b)/(b2): from a flat start theta converges to the objective's fixed point = the reverse model's posterior, so the fixed-point PER is the cold-start CEILING (bigram 0.190, trigram 0.164, count durations 0.171, k = 30 limits pending in read (b3)); S3 failed to REACH it (content-free solution, phone rate 1-2/s, negative derangement gap) — take-off is a separate problem from the ceiling. (2) Reads (c1)-(c3), network-free on the banked 300 dev-other utts (`reports/impl_cold_fixed_point_2026-09-15.md`, implementer running -> executor GPU): (c1) fixed point from the FLAT recognizer with the warm-up phi held (does the flat start reach the seed's limit?); (c2) cold EM: flat theta, phi refitted every iteration from its S3 init (reproduces the collapse network-free, or not); (c3) (c2) with the duration table pinned to a label-free gamma prior (mean 5 frames) — does pinning durations stop the rate collapse? Report PER, phone rate, E[d], reverse LL per k. (3) Read (b3) running (anchored fixed points; K = 30 limits = ceiling under a good phi). (4) Literature: cold-start collapse and remedies (`reports/lit_cold_start_collapse_2026-09-15.md`, running); lattice pruning bias (`reports/lit_lattice_pruning_2026-09-15.md`, running). (5) Prior order: trigram inside the objective via the exact D3 + D4 single pass (bench running); two-pass lattice not funded (design review, see "Two-pass lattice" under the prior-order section); 4-gram needs a pruned-LM state space — first read the held-out ppl of the 4-gram pruned to 500-4000 contexts (CPU, queued). (6) Then: S3b design (cold start with the remedies the reads select) -> design-reviewer before the first job. (7) GAN-init cleanup: 228 dirs in `reports/exec_gan_cleanup_2026-09-15.list` (4.9 G), needs the user's one-line command. (8) Queued: temperature sweep rerun (T = 2.0), stale docstring config_sae_4a_decode_temp_v1.py:16-23, kernel fusion.
Live pids / watcher: no manager needed — `config/sae_4a_phase.py` is COMPLETE (1021/1021 finished, 0 errors). Next manager only for S2d; then re-arm `bash ~/.claude/skills/sis/sis_watch.sh <pid> <config> 300` (run_in_background, this session). Running analyses (SLURM, not sisyphus): read (b3) executor; D3+D4 implementer bench.

## Objective

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

## S3 cold start: G4a.3 read (2026-09-15; audited CONFIRMED FAIL, see the audit note above)

Run `ReturnnTrainingJob.sBlPYBA1YcIQ` (flat init `FlatRecognizerInitJob.21Kxgr5JLR3k`, 8 sub-epochs, tau 8 / 5.04 /
3.17 / 2 / 2 / 2 / 2 / 2, ~123 s per sub-epoch; extractor report `reports/extract_s3_reads_2026-09-15.md`). Dev L_tau
1.74 (sub-ep 1) -> 1.21 (sub-ep 3) -> 1.21 (sub-ep 8); reverse term per frame -7.22 -> -3.32. Dev-other greedy PER:
0.8955 at sub-epoch 4 (gate read), best 0.8387 at sub-epoch 5. Emitted phone rate per sub-epoch 3.71, 1.98, 0.97,
1.58, 4.75, 6.87, 4.29, 7.73 per s: every sub-epoch is outside [0.6, 1.5] x 9.8/s = [5.9, 14.7] (the rate rule
fires on all eight; nothing to revert to). Speaker-matched derangement gap at sub-epoch 4 on dev-other
(`S3DerangementGapJob.Ez8bGMB7LnPX`): gap_per_frame -0.3218, ci95 [-0.6446, -0.0170], n_selected 314/500,
identical_donor 12 — negative with the CI excluding 0: the held-out reverse model scores a deranged donor's decode
ABOVE the utterance's own. G4a.3 as pre-registered (dev-other PER < 0.50 AND positive gap at sub-epoch 4): FAIL on
both clauses. Reading (pending audit): from a flat start the objective is optimised (L_tau falls 1.74 -> 1.21) by a
recognizer whose output carries no utterance-specific content (PER ~ 0.9, negative gap, phone rate collapsing to
1-2 / s during the anneal); the cold-start bootstrap does not take off on this bed. Per the gate this licenses "not
funding S3 further", not "it could not work" (a single schedule and seed were run).
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
