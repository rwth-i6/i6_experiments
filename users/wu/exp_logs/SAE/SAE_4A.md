# SAE §4a — Exact-marginal cycle (EMC): tempered posterior-marginalized training with a semi-Markov reverse model

## State

Phase registered 2026-09-15 from the revised method note; literature in `reports/lit_method_v2_2026-09-15.md`,
reusable artifacts in `reports/SAE_setup_report_2026-09-15.md`, project evidence in
`reports/sae_failure_evidence_for_method_v2.md`, design review in `reports/design_review_4a_2026-09-15.md`.
S1a done and banked (Results; G4a.1 pass, audited). S0b built and reviewed (`reports/review_s0b_core_2026-09-15.md`,
`reports/review_s0b_inits_2026-09-15.md`; commits up to bc73083 / 73b9f64 on haotian_modality_matching_jupiter;
lattice bench 98 utt/s at B = 64 on a GH200). S0b INITS LAUNCHED 2026-09-15 (config/sae_4a_s0b_inits.py):
ctc_init (i) `work/i6_core/returnn/training/ReturnnTrainingJob.HcXzd6M2eyVZ` (SLURM 1799887; compute 0.063
s/step but wall ~1.6 s/step of 157 utts, GPU 0-6 %: HDF loader-bound, accepted for this one-off), oracle_init
(iii) 65NNK8Bwxdtd (FINISHED), feature/units HDF repack, greedy PER + WER of (i) at epoch 20, all in that graph.
S1b/S2 configs reviewed and fixed (`reports/review_s1b_s2_configs_2026-09-15.md`, commit d9905df); accepted
constants and their efficiency amendment below. First S1b job (c4WZmlzJzAbw, 2026-09-15) ran at 37 utt/s -> gate
FAIL -> profiled and fixed (loader collation, agg loop, batch 128 / theta lr 1e-4, forward extern_data key;
commits dce4046..2fed7d6, review PASS). S1b RELAUNCHED at the new hash
`work/i6_core/returnn/training/ReturnnTrainingJob.93UGC7HzGC2P` (SLURM 1800557;
`reports/exec_s1b_relaunch_2026-09-15.md`). Efficiency re-read, log-order steps 11..100: median emc_utts_per_sec
73.2 (min 33.9, max 192.2), median num_seqs 122, loader wait 0.02 s of a 1.54 s step (was 0.39), l_tau 3.20 ->
1.88, Z=0 fraction 0, no nan/inf. Verdict: FAIL on the number as written (75), PASS on the rule behind it (loader
<= a quarter of the step; measured 1.3 %; profiler compute ceiling 75.9 utt/s at this batch). Orchestrator decision:
S2 is funded on the rule (the remaining cost is the DP kernel itself, ~1.6 min per tc100 sub-epoch; fusion queued
before any scale-up); the miss on the absolute number is recorded, not amended away. S1b is a pipeline/drift read
only (G4a.1 passed). S2 LAUNCH in progress under one combined manager (config/sae_4a_phase.py = S1b + S2 graphs;
`reports/exec_s2_launch_2026-09-15.md`); its warm-up waits on ctc_init (i).
NEXT: (1) when (i) finishes, bank its greedy PER and dev-other WER as the S2 paired baseline in Results; check its
DecodeStats phone rate. (2) Bank S1b PER per sub-epoch (2 sub-epochs) against §1a's oracle-EM drift row
(0.275 -> 0.392). (3) S2: per-sub-epoch phone rate against [0.6, 1.5] x 9.8/s (revert rule); at the end read G4a.2
(paired dev-other WER delta vs init (i), usability vs 17.96 / 21.87) on the unsupervised-selected checkpoint;
audit before acting on it. (4) Efficiency: fuse the lattice frame body (torch.compile / Triton) before any run
beyond tc100.
Live pids / watcher: ONE manager pid 2643163 on config/sae_4a_phase.py (S1b + S2 graphs, 259 jobs incl. the oracle-init PER evals GreedyPerJob.3TcOOObr1IrW / RMkjOs4czduh;
log/sae_4a_phase.manager.log; log_level 30, judge from job dirs); watcher `bash ~/.claude/skills/sis/sis_watch.sh
2643163 config/sae_4a_phase.py 300` from the setup dir. Arms A/B trainings FINISHED (SLURM 1802597/1802596); their evals, decodes and selections run. S2 job dirs at the current hashes: warm-up
ReturnnTrainingJob.q6BUlBXVfA1t, arm A 3EVuGpAEAn8m, arm B HUSP5F9GBUVr, selections
UnsupervisedCheckpointSelectionJob.B8xyZzBQnZvz / THl07BoM4t2Q. Earlier managers 1167500, 1453067, 1697904, 1724233 stopped.

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
  number stands as written; both are reported at the re-read.
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

## Standing constraints carried

d_min >= 2 (`SAE_3E1.md:161`); labels never train or select in S2/S3, S1 is a quarantined diagnostic; paired
reads and plain sclite WER (`SAE.md:40-45`); lambdas per bed, sweep at 100 h; no replay data in any phi refit;
unsupervised checkpoint selection by the pre-registered formula; token-rate revert rule above.

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
| S1b: oracle init (iii) + L_tau, sub-epoch 1 (`ReturnnTrainingJob.93UGC7HzGC2P`) | 0.230 | 0.257 | 2n7guvyHYuLk / x9GWHcVAN9jb |
| S1b: sub-epoch 2 | 0.250 | 0.277 | OetKCUla1mp4 / hF5YS6dUP65f |

Init (i) sits above §1d's full fine-tune (0.138 / 0.172) as the design review predicted for a conv head on frozen
features; it is the S2 paired baseline (its lexicon + 4-gram WER decodes are pending). S1b: held-out L_tau per frame
2.004 -> 1.921 (train 2.699 -> 1.908), phone rate 8.6 / 8.5 per s on dev (inside the revert band), no Z = 0
utterances; PER rises ~0.02 per sub-epoch on both sets while the objective falls. Same direction as §1a's
oracle-EM drift (0.275 -> 0.392), smaller so far. The oracle init's own PER was not registered; its eval is being
added (hash-neutral) so the drift has a starting point. S1b is diagnostic only (G4a.1 passed); the decision read is
S2, where arm B carries the init anchor S1b lacks.

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
