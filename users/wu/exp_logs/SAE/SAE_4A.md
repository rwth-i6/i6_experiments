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
NEXT (direction change 2026-09-15, see Stages amendment; S2b + S3 LAUNCHED 2026-09-15 under the single phase manager): (0) Decode sweep on init (i): when the 10 count jobs finish, extractor reads dev-clean / dev-other sclite WER, S/D/I and hyp/ref word counts per T in {1, 1.5, 2, 3, 4}; T = argmin dev-clean WER (tie -> smaller T) is frozen for every §4a decode; implementer wires it into `config_sae_4a_s2b_v1.py` / `config_sae_4a_s3_v1.py` (decode jobs re-hash, trainings do not), manager restarted by the executor. (1) Efficiency read on the first ~100 steps of the S2b-10h warm-up and S3 trainings (median emc_utts_per_sec, loader wait = emc_sec_per_step - sec/step, GPU memory) against the rule "loader <= 1/4 of the step" (number 57 at B = 128). (2) S2b per-sub-epoch PER and phone rate vs [0.6, 1.5] x 9.8/s (revert rule) for arms A / B / C x {10 h, 1 h}; the arm C - arm A paired PER delta is the lambda_agg read; S1b's confounded oracle row is superseded by S2b-10h arm A. (3) G4a.2 on S2b: paired dev-other WER delta vs the arm's own seed init at the frozen T, unsupervised-selected checkpoint; audit before acting. NOTE: `emc_train_jobs.py:1084` labels every S2b paired delta "init_i"; the baseline is the track's own seed init (verified in `reports/review_s2b_armC_2026-09-15.md`); read hashes, not the label. (4) S3: rate rule per sub-epoch; G4a.3 read at sub-epoch 4 (`S3DerangementGapJob.Ez8bGMB7LnPX`, dev-other PER < 0.50 with a positive gap); audit before acting. (5) Queued, not launch-blocking: kernel fusion of the lattice frame body before any run beyond tc100. (6) GAN-init cleanup (user: kill and delete allowed): after the decode sweep is banked, drop S1b/S2 from the phase launcher, restart the manager, delete the GAN-init job dirs and posterior dumps.
Live pids / watcher: NONE — manager 3504494 on config/sae_4a_phase.py EXITED 2026-09-15 ~14:00 with 56 KenlmPosteriorDecodeJob errors (collect-step assert on empty hypotheses; hash-neutral fix in implementation); watcher stopped. Restart = executor clears the 56 error markers, `./sis_managers.sh start sae_4a_phase` (graph now + 108 PairedPerDeltaJobs, + S2c when committed), then re-arm `bash ~/.claude/skills/sis/sis_watch.sh <pid> config/sae_4a_phase.py 300`. Previous: manager 3504494 on config/sae_4a_phase.py (S1b + S2 + S2b arms A/B/C both tracks + S3 + init (i) decode temperature sweep; 1040 jobs, 779 unfinished at start; log/sae_4a_phase.manager.log); watcher `bash ~/.claude/skills/sis/sis_watch.sh 3504494 config/sae_4a_phase.py 300` from the setup dir, plus a 5-min poll on `alias/sae/4a/decode_temp/init_i/T*/dev-*/counts/finished.tar.gz` (10 = sweep done). First SLURM submissions: S2b-10h warm-up ReturnnTrainingJob.htxT2f9FHvWw (1805347) and S3 sBlPYBA1YcIQ (1805348); arms wait on the warm-ups. Handover report `reports/exec_phase_manager_handover_2026-09-15b.md`. Previous manager 2643163 on config/sae_4a_phase.py (S1b + S2 graphs, all finished; GAN-init S2 job dirs: warm-up
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
