# PLAN_3E1 — Scorer trainability without collapse (sub-plan of PLAN.md §3e.1)

Opened 2026-08-07 (planner) after the 14-agent design fan-out (2 grounding readers, 6 literature
lenses, 3 designs each red-teamed). Question: what update rule / pair construction / scheduling
lets psi_align repair its contaminated prior in the G-track without collapse, catastrophic
co-adaptation, or training on badly sampled policy text. Collapses back into §3e.1 when closed.

## Verdict on the driving hypothesis (2026-08-07)

User hypothesis: "the GAN-inited scorer cannot rank the samples well enough; co-training is
unavoidable." Refined, evidence-anchored:

- Ranking NOISE is refuted for the psi_align family: the G-track `recon` arm kept usable
  within-group recon std (0.1112 -> 0.0276 over ep1-4, SAE_3A.md) — roughly 10x the AR-era
  flatness that killed the 2S sweeps (0.0074-0.0142 against 0.27 nats total range).
- The live mechanism is DIRECTIONAL BIAS: scorer and init share byte-identical §1d pseudo-text,
  so the scorer REWARDS the inherited filler ("to" = 61.8% of the recon arm's insertions; 66.8%
  of theta_0^G's own insertions before any loop step). A noisy scorer slows the loop; a
  correlated-bias scorer steers it into the shared defect. D0 splits the bias (2026-08-07,
  `SAE_3E1.md` c4): ~70% is a psi_align FAMILY property — the never-contaminated gold-text
  control also pays for "to" at matched WER (beta 0.167 vs the loop scorer's 0.243) — and only
  the ~+0.075 differential is attributable to the shared text; the IBM-Model-1
  garbage-collector account, confirmed on our own bed.
- Group blindness is MEASURED, partial, and binding (2026-08-07, replaces "untested";
  `SAE_3E1.md` c6): only 23.3% of "to"-carrying groups hold a "to"-free member (9.2%
  suspect-wide), so ~77% of live groups are unsteerable on that axis for ANY scorer — not the
  coverage~0 fork, but the cap on every scorer-side repair.
- "Co-training unavoidable": the scorer must MOVE (a frozen scorer provably keeps the inherited
  prior; a scorer that must start good imports the bootstrap problem), but continuous in-loop
  co-training is the known-collapsed form (2S replay `freeze_ar=False`: 18.79 -> 46.71).
  Re-diagnosed 2026-08-07 (replaces "text-blind"; `SAE_3E1.md` c1-2): the co-trained scorer
  KEPT and GREW its text contrast (usage gate 0.333 -> 0.621) while drifting off the gold
  domain entirely (CE_true 5.744 -> 6.294, past the unit marginal 6.007 and uniform ln 500 =
  6.215) — the failure mode is DRIFT, not blindness, so any trainability gate must read
  absolute conditional fit, not a contrast statistic (launch-time grad-norm parity measured a
  different quantity and stands). The admissible shape is DISCRETE, GATED, OFFLINE refresh
  rounds — and possibly a round-0 text repair with no co-training at all, which has never been
  tried (exactly one pseudo-text decode exists on record).
- ATTRIBUTION REOPENED 2026-08-09 (user question; replaces "known-collapsed form" as a settled
  premise — the record supports only "collapsed once, unattributed"): the 18.79 -> 46.71 run is
  the only three-term arm on its bed and NO frozen-scorer control has ever run on the 100 h bed
  (`config_sae_2s_grpo_loop_100h_v1` defaults `freeze_ar=False`; every 100 h arm on record is
  jointAR — the alias template supports frozenAR but no such run exists), while the one matched
  frozen-vs-joint pair on record (10 h seed-audio bed, 4 epochs) went the OTHER way: frozen
  Goodharted 13.07/15.89 -> 14.47/17.09, joint won 14.74/16.87 -> 13.15/16.13. And two sibling
  jointAR arms on the SAME 100 h bed did not show the collapse signature (recon-only blew up
  via the insertion exploit with binding intact 77.9 -> 74.2; hinge-only recovered monotonically
  83.38 -> ~41 plateau). So co-training is not collapse per se; the live hypothesis is
  bed-dependent scorer drift — safe while the policy's samples stay near gold, lethal when they
  do not. D5 below funds the missing control.

## Standing facts that bound any design (verified 2026-08-07, file:line)

1. NO in-loop trainable path for psi_align exists: `grpo/psi_scorer.py:153-156` raises on
   ce_loss; `train_steps/sae_grpo.py:153` forces the frozen path under `reward_mode="psi_align"`
   (`freeze_ar` refers to the retired token-LM scorer only). Any refresh is a sisyphus outer
   loop: dump -> build HF text dir -> `PsiAlignTrainJob` -> gates -> new loop job. The current
   job interface has an `init_model` input, added after this standing fact was first written;
   D7 nevertheless uses the exact round-1 from-scratch recipe for its matched loss ablation.
   psi lives inside the loop job's
   state_dict (`definitions/sae_grpo.py:203-215`), so a swap is stop/relaunch with explicit
   checkpoint import; an import override that carries the policy but NOT the old psi is a build
   item.
2. The G1 gate as instrumented is GOLD-CONDITIONED: `config_sae_3a_enc50_units_v1.py:233-243`
   wires `PsiAlignInfoGateJob` with seed_hf (real 10 h text). Admissible as evaluation on the
   supervision axis; NOT as a G-track acceptance test (labels would select what trains). A
   label-free twin on pseudo/anchor text is a build item.
3. `text_explained_loo` has the WRONG SIGN against the filler mode: a scorer that lets
   G2P("to") absorb frames RAISES text-explained. Gate v1 catches text-blindness only and
   passes — even favors — the filler mode. A filler-directed side is mandatory.
4. `held_nll` is a random 5% split of the scorer's OWN training corpus
   (`psi_align_jobs.py:363-371`): corpus-level defects sit in train and held alike, and each
   refresh re-draws the split, so verdicts are not comparable across rounds. Acceptance needs
   ONE frozen external held pair set scored by a standalone job.
5. Units are k-means over theta_0^G's own post-adapter states — a measurement through a
   contaminated model. They stay PINNED for this whole program; never re-extract units from an
   updated policy (that would close the loop on the target side too).
6. "Pin emissions/transitions, refresh the text side" is not a real partition: emit/arc/start
   are heads on one shared trunk (`psi_align.py:409-430`). Withdrawn from the candidate list.
7. lam_2 KL anchors the policy to theta_0^G, which carries the filler at 66.8% — in the G-track
   the KL term actively DEFENDS the defect. Any accepted refresh must revisit the anchor
   (re-point or reset the policy; overoptimized policies are unrecoverable — Wolf & Kirk).

## What the literature says (six lenses, 2026-08-07)

Every field that ran this experiment converged on the same stabilizer set; the G-track broke all
of them at once:

- ANCHOR on genuinely uncorrelated data with a floored share; accumulate, never replace (UNMT
  denoising anchor — Lample; positive-real-fraction — Ferbach; Gerstgrasser).
- LAG / separate clocks: stale caches or EMA teachers, two-timescale updates (slimIPL, momentum
  pseudo-labeling, kaizen, TTUR). Continuous same-step co-training collapses (slimIPL without
  cache, MPL at momentum 0 — and our 2S replay arm).
- HARD, FILTERED targets selected by a signal the trained model does not own (hard-EM, ReST-EM;
  West-of-N warning: best-of-group under the scorer's own reward amplifies its bias).
  Soft/posterior-weighted targets are a published collapse mode — the old "posterior-weighted
  CE" candidate is withdrawn.
- VIEW INDEPENDENCE (Blum-Mitchell): agreement filtering is worthless when the views share the
  defect; byte-identical text is the maximal violation. Seed twins (psi_g_seed) detect nothing
  on a shared defect — kept only as a noise null.
- IDENTIFIABILITY (Yang et al., sequence-level unsupervised training theory): contaminated
  conditioning text does not add noise, it voids the guarantee that low held unit NLL implies
  low error (a rank condition on the conditioning distribution). Gates measured on the defect's
  own distribution cannot see it — the theory twin of facts 2-4.
- MAKE TEXT-BLINDNESS UNREACHABLE rather than gated: matching-aware contrastive terms (GAN-CLS),
  contrastive-estimation neighborhoods (Smith-Eisner), MMI denominators — a text-blind scorer
  gets zero gradient by construction. The IBM-Model-1 "garbage collector" literature is our
  filler under another name, with tested counters (null-word down-weighting, smoothing,
  early-stopped EM).
- The published twin of our exact bug: wav2vec-U's silence-token episode; its fix — match the
  suspect-token RATE between pseudo-text and real text — transfers directly.

## The ladder (pre-registered 2026-08-07; each step decides the next; nothing past D1 is funded
without the user's call)

**D0 — mechanism discriminator** (CPU + <=2 GPU-h; labels EVALUATE only, never select). On
existing recon-arm rollout dumps (or one fresh G=8 `ReturnnForwardJobV2` of theta_0^G on ~2k
utterances):
  (a) oracle best-of-8 WER and group-contrast coverage = fraction of groups with >=1 "to"-free
      member — the group-blindness test;
  (b) within-group spearman(reward, -WER) on spread-bearing groups — the noise test;
  (c) partial effect of "to"-count on reward rank at matched WER — the bias test;
  (d) suspect-vocab derivation, label-free: EXCESS MASS (rate difference) of pseudo-text tokens
      vs the LM corpus, threshold pre-registered before looking; recovering "to" is a reported
      sanity check, never the calibration target (a ratio test top-ranks rare words and misses a
      high-frequency filler);
  (e) cross-view covariance: cov(each candidate selector signal, -WER) — the admissibility
      precondition for any curated refresh (adversarial-curation Lemma 3.3).
  Forks: bias-dominant -> D1/D2; noise-dominant -> ensemble/contrastive escalation;
  coverage ~ 0 -> policy-side diversity first (temperature / novelty term) and scorer work is
  pointless until groups carry contrast.
  VERDICT 2026-08-07 (`SAE_3E1.md`, audit clean): (a) coverage 23.3% "to" / 9.2% suspect-wide;
  (b) noise refuted in-group at the loop's operating point (spearman 0.496 recon / 0.556
  shaped); (c) bias confirmed, family-dominant (~70% gold-control-shared, ~30% differential);
  (d) suspect set {to, of, buy} at the pre-registered 0.002; (e) lm_prior_units 0.502
  admissible, neg_n_suspect 0.186 weakly, psi's duration channel not a selector. Fork taken:
  bias-dominant -> D1/D2 with the amendments below; the coverage sweep below runs as a
  CO-REQUIREMENT beside D2 (replaces "coverage ~ 0 -> first", 2026-08-07: partial blindness
  caps any repair at 23% of live groups, so contrast restoration cannot wait on that fork).
  Coverage contingency (amended 2026-08-07, replaces the temperature-and-G sweep — because the
  D0 dump already contains T = {0.3, 0.5, 0.7, 0.9, 1.0} at G=12, and the G axis is DEAD: the
  blindness is per-utterance habit, not sampling odds — at T=0.7, 77% of "to"-live groups have
  ZERO free members in 12 draws, so plug-in coverage at G=16/24 stays at ~22-23% vs 23% at
  G=12, while at fixed rows = max_seqs x group_size a larger G cuts max_seqs on a launch-bound
  loop). The sweep is a temperature-only READ of existing artifacts (CPU + at most one small
  rerank per candidate scorer). Planner scratch read 2026-08-07 (implementer to reproduce as a
  logged table): cov_to 0.14/0.16/0.23/0.53/0.82 over the T grid; T=1.0 is garbage diversity
  (mean WER 0.52, oracle 0.28) and coverage alone is gameable by it, so the selection statistic
  is STEERABLE coverage — a "to"-free member exists AND its recon under the scorer that will
  actually run beats the group mean (positive-advantage test; label-free: token counts + recon
  only). Under the CURRENT scorer: steerable 0.19 @ T=0.7 -> 0.31 @ T=0.9; mixed-temperature
  compositions (e.g. 8@0.7+4@1.0) buy coverage 0.58 but convert poorly (steer|cov 0.38) and
  need a per-member-T sampler build — parked unless the repaired scorer still under-steers at
  uniform T. Presumptive point WITHDRAWN 2026-08-07 (replaces "T=0.9, G and max_seqs unchanged":
  the implementer's logged read, verified — `SAE_3E1.md` c11 — shows T=0.9 buys steerable
  0.19 -> 0.34 but degrades the ORACLE 0.107 -> 0.150 and conversion 0.835 -> 0.642, so it is
  not the free move the scratch read treated it as). T stays 0.7 unless the joint
  (scorer, lambda, T) re-read with the D2 winner on the same dump shows the repaired scorer
  still under-steering at 0.7. If no temperature yields steerable coverage, the init itself
  never emits the filler-free variant and the lever moves to the init (§1e ep50 pins / §3d
  round-2) — a fork reported to the user, not absorbed.

**D1 — probe build + power check** (~2 GPU-h). Extend `PsiAlignInfoGateJob` to text-side
pairings (per-pairing (states, obs) and per-pairing feasibility; injection shifts the U <= 2T
set, so contrasts use length-matched SUBSTITUTION, not deletion). Probes: filler-injection /
filler-substitution contrast (delta_filler on ce_loo) and a corruption ladder (DEL1, LM-word
substitution, filler at k = 1, 2, 4; statistic = spearman(severity, ce_loo increase)). Power
check: probes must separate psi_align^G from the 10 h-true scorer — the labeled contrast
VALIDATES the instrument offline; floors are then pinned to the pre-loop frozen scorer's own
values (labeled runs may report, never set a threshold). Build the frozen external held pair set
(fact 4) in the same step.
  VERDICT 2026-08-07 (`SAE_3E1.md` c8-c10, audited): the power check FAILED — no filler statistic
  separates psi_align^G from the gold-text control (insertion discount, substitution discount and
  suspect state mass agree across all three arms within CIs). The audit sharpens the reading: the
  headline insertion discount is majority a state-count artifact (the LM control is drawn
  frequency-proportional with no length matching; "to" is one emitting BPE state vs ~2.7 for the
  mean draw), the token-specific residual is 0.011-0.027 nats/frame and scorer-dependent, and
  what IS scorer-invariant is the lattice's ~0.03 nats/frame price per inserted emitting state —
  the exploit class is every minimal-state word, not the suspect set; contamination chose which
  word, not whether. Consequences: (a) the battery is demoted from contamination discriminator to
  MECHANISM METER, and gate v2 (iii)/(iv) keep it only as regression arms; (b) a length-matched
  probe variant (control pool restricted to 1-emitting-state words, per-item records dumped) is a
  build item required before any D2 admission read; (c) a round-0 text repair cannot be
  load-bearing — the D0 family split is confirmed by an independent route.

**D2 — round-0 repair, no co-training** (~6-10 GPU-h; the scorer is ~11M). Retrain psi from
scratch on repaired pseudo-text: excess-mass suspect set from D0(d), suspect rates matched to
the LM corpus (the wav2vec-U silence fix), per-utterance multiplicity cap. Options decided by
D0: a second, decorrelated decode of the same student (no-LM greedy — D0 says whether the filler
lives in the student or in the word-LM pass); the GAN-CLS-style matched/mismatched contrastive
term (text-blindness unreachable by construction; ~2x scorer-step cost) — CO-PRIMARY as of
2026-08-07 (replaces "kept only if its own gate read pays": D0 shows ~70% of the filler payment
is family-level and unreachable by any text repair, `SAE_3E1.md` c4, so a mechanism-level
counter — the contrastive term and/or null-word down-weighting — must ship in D2, with
rate-matching addressing only the ~30% differential). Admission: D1 battery +
`PsiScorerParityJob`.
  AMENDED 2026-08-07 (replaces the admission line above and the co-primary framing, because D1
  failed its power check and the audit widened the exploit to every minimal-state word):
  rate-matching is HYGIENE for the ~30 % differential; the PRIMARY lever is mechanism-level
  insertion pricing — the contrastive term, plus the zero-training knob below. Candidate status:
  d2_rate / d2_contrast / d2_both are mid-training (ep 5-6 of 30; the contrastive arms' held NLL
  degraded ~2.90 -> 3.23 / ~2.95 -> 3.36 when the alignment prior annealed off — watch the gate
  (i) floor). FOURTH ARM registered 2026-08-07 (implementer-added, `SAE_3E1.md` approach 8 +
  c12; wiring AUDIT-VERIFIED same day): d2_states (chars_per_state 1.5 -> 0.5) — the only arm
  that moves the structural frames-per-state factor (2 T/U 9.77 -> 3.92 on the held set), i.e.
  the direct response to the audited per-state pricing finding; the prediction itself is
  verified on the probe dumps (orphaned frames chars_per_state-invariant, inserted-word state
  counts scale ~2.5x). Feasibility caveat RESOLVED for the held/probe sets (1500/1500 feasible
  at cps 0.5, probe-common 1442 with identical membership) — still to be reported for the
  ROLLOUT set before any d2_states rerank read. Status correction 2026-08-07 (replaces the
  ep 5-6 line above): d2_rate at ep ~9/30 (held best 2.8843; pre-resume code — a preemption
  loses it), d2_contrast / d2_both RESTARTED and at ep1, d2_states launched 0/30 (~2.5x epoch
  cost vs time_rqmt 11 h — completion at risk without working resume). Admission (replaces "D1 battery + parity"): length-matched probe battery not worse
  AND per-inserted-state price raised vs the incumbent; gate v2 (i) floor-only + (ii)-(v); bias
  beta and steerable coverage re-read on the D0 dump; `PsiScorerParityJob`.
  PRIOR REPRICING (planner scratch read 2026-08-07 on the D0 dump — free, shaped recomputed per
  lambda as recon + lambda * prior_sum / n_units; implementer to reproduce as a logged table): at
  T=0.7, beta_to falls 0.228 (lambda=1) -> 0.111 (8) -> 0.032 (16) -> -0.006 (24) while in-group
  spearman RISES 0.556 -> 0.678 (peak at lambda=8, still >= 0.61 at 24; prior-alone limit 0.502);
  BUT the prior's share of within-group reward variance goes 1.3 % (lambda=1) -> 46 % (8) ->
  77 % (16) -> 88 % (24) — the beta-zero point is a nearly audio-free reward and inadmissible
  under the audio-free-null principle. Registered as a BOUNDED secondary knob (lambda in [4,8]:
  halves the filler payment at the ranking peak, prior share <= ~46 %); the final
  (scorer, lambda, T) operating point is selected JOINTLY on the same dump with the D2 winner's
  rerank, before D3.
  CANDIDATE STATUS CLOSED 2026-08-08 (replaces the two status lines above): all four arms
  finished 30 epochs and are read at their best-held epochs; resume was never needed.
  D2 VERDICT (2026-08-08, planner; full audit in `SAE_3E1.md` Verifier feedback): every logged
  number reproduces; the winner is d2_contrast CONDITIONAL on the clause pins under D3 below
  (under the unpinned literal point reading it would be d2_states) — the pins need the user's
  blessing. The pre-registered joint (scorer, lambda, T) read is DONE on the D0 dump
  (planner scratch; implementer to reproduce as a logged table): lambda dominates the scorer
  axis — the incumbent at lambda=8 reaches spearman 0.6778 / beta_to 0.1112 / sel_wer 0.1222
  against 0.5558 / 0.2284 / 0.1316 at the live lambda=1, and at matched operating points no D2
  candidate beats the incumbent on any rollout statistic; the optimum is arm-invariant at prior
  share ~0.45 of within-group reward variance, i.e. lambda 7.9 / 8.2 / 9.5 for incumbent /
  d2_contrast / d2_states (lambda does not transfer across scorers — match prior share, not the
  scalar). T stays 0.7: steerable coverage at the repriced point is 0.20-0.21 for every arm, so
  the dead-band, not the scorer, is binding. Selected operating point: prior share ~0.45
  (lambda=8 for a cps-1.5 scorer), carried into the D3 design fork below.

**D3 — frozen-repaired control arm** (~9-18 GPU-h): rerun G-track `recon` + `shaped` 2-4
sub-epochs with the D2 winner FROZEN. Pre-registered bar: dev not worse than theta_0^G
13.89/18.34 AND a falling "to" insertion share. This is the control every co-training claim must
beat; if it converges, "co-training unavoidable" is refuted at round-0 cost.
  COST CORRECTED 2026-08-07 (replaces "~9-18 GPU-h", because that was the planner's 100 h-bed
  assumption while the pre-registered bar 13.89/18.34 pins D3 to the 960 h bed): as wired
  (`config/sae_3e1_d3.py` through the 960 h loop graph) it is ~85 GPU-h for two arms at four
  sub-epochs. Recommendation: fund 2 arms x 2 sub-epochs (~42 GPU-h, within the pre-registered
  2-4 range), per-sub-epoch monitors carry kill/extend authority. Scheduling constraint: the D3
  config shares the running loop manager's graph — run exactly one of the two managers at a time.
  WINNER-RULE AMENDMENT 2026-08-07 (before any D2 read exists; amends the implementer's
  pre-registered rule in `SAE_3E1.md` approach 9, whose eligibility floors and no-winner clause
  stand unchanged): the selection statistic must be the LENGTH-MATCHED insertion discount — the
  control pool state-matched to the filler under EACH candidate's own segmenter — not the
  frequency-drawn discount 0.0584, because the audit shows 53-81 % of that number is
  state-count artifact, and d2_states changes state counts mechanically, making the unmatched
  statistic incomparable across arms (it would move without any change in scorer behavior).
  Corruption-ladder and discount comparisons across arms are read on the INTERSECTION of the
  arms' feasible sets, reported per arm. EXTENDED same day after the approach-9 audit
  (`SAE_3E1.md` verifier feedback): the no-winner threshold must be the PAIRED cross-arm
  difference CI on the shared probe utterances — the logged ~0.005 is the incumbent's own level
  CI half-width and bounds the wrong quantity (the audit confirms the state-count confound is
  preserved almost exactly under cps 0.5: filler-vs-control ratio 1.855 vs 1.866); and the
  eligibility ladder floor applies to each of the five ladders separately (filler_sub / lmsub /
  del / filler_ins / lmins), read paired on the shared set. The rule is currently prose-only
  (`WINNER = None`, nothing combines the clauses) — the selection read is a build item.
  CLAUSE PINS (2026-08-08, planner, after the D2 read was audited — both pins are DECISIVE for
  the winner and therefore NEED THE USER'S BLESSING; the D3 config's hard-coded
  WINNER='d2_contrast' is provisional until blessed):
  (1) the ladder floor's "not below" is pinned as "paired 95 % CI vs the incumbent wholly below
  zero counts as a violation" — the convention the logged 3/1/3/0/0/0 column already uses; under
  the literal point reading only d2_states is eligible and the winner flips to d2_states, so
  this choice IS the winner selection. Planner's case for the CI reading: a point test on a
  noisy per-utterance mean rejects arms on -0.003-level draw noise.
  (2) clause (ii) is STRUCK for changed-text candidates (replaces its unconditional form,
  2026-08-08, because H_uni is bit-identical across arms on the frozen set, making the clause
  algebraically the gate v2 (i) improvement comparison the 2026-08-07 amendment already ruled
  inadmissible for them — as instrumented it silently re-imposed that comparison and was the
  sole eliminator of d2_both; admitting d2_both changes no argmax at k=1/k=4).
  (3) the selection statistic is pinned at k=1 with k=2/k=4 reported (approach 9's 0.0584
  reference is the k=1 value; the d2_contrast-over-d2_states argmax holds at every k).
  (4) for d2_states (unchanged text, changed segmenter) the improvement halves of clauses
  (i)/(ii) are void — its ce_loo is cps-incomparable — and only the absolute floors bind.
  Under these pins the winner is d2_contrast; under the point reading it is d2_states; both
  reduce the matched discount with solid CIs at k=2/k=4 (d2_states' k=1 interval is
  seed-fragile), so the no-winner clause does not fire either way.
  DESIGN FORK (2026-08-08, planner — the user's call, together with the funding ask): the joint
  read shows repricing (lambda 1 -> 8) is the dominant lever while the scorer axis is invisible
  on rollout statistics and visible only on the controlled insertion probes; as wired D3 runs
  recon + shaped with the winner at the live lambda=1, a now-known-suboptimal operating point,
  whereas the PRIOR REPRICING paragraph pre-registered exactly this joint selection before D3.
  Recommended 2-arm x 2-sub-epoch package (~42 GPU-h): (A) shaped at lambda=8 with the INCUMBENT
  scorer — the pure-repricing arm, directly testing whether the units-normalized prior at its
  admissible ceiling saves the 960 h loop; (B) shaped at lambda=8 with d2_contrast — the same
  operating point with the winner, isolating the scorer axis. The as-wired alternative
  (recon + shaped, d2_contrast, lambda=1) tests the original scorer-repair question at the wrong
  lambda; a recon arm under d2_contrast is the purest mechanism test but highest-risk (the
  matched discount is halved, 0.0172 -> 0.0082, not closed — gold sits at 0.0031). The
  pre-registered bar (dev not worse than 13.89/18.34 AND a falling "to" insertion share) applies
  unchanged to every variant; the lambda=8 point is measured on the D0 dump (theta_0^G rollouts,
  tc100 audio — the loop's own early distribution), and reward shape is per-bed, so the
  per-sub-epoch monitors keep kill authority. Selection hygiene: the operating point is chosen
  by the LABEL-FREE statistic — largest lambda keeping the prior a minority (<0.5) of
  within-group reward variance, inside the pre-registered [4,8] bracket — and the WER columns
  CONFIRM rather than select (they peak at the same point); lambda=1 remains the derived
  posterior-identity point, so lambda=8 is a swept departure from it and the arms carry that as
  a counted cost, not a hidden one.
  STATUS CORRECTED 2026-08-17 (planner, verified from the job dirs — replaces the
  "parked-unstarted" reading my own 2026-08-17 registration note repeated): D3 DID
  LAUNCH as wired (WINNER='d2_contrast' frozen, live lambda=1) and is HELD part-way,
  not unstarted — `ReturnnTrainingJob.rJWSC5xOsrf2` (shaped) and `.L6FwOOpffNL4`
  (recon) each hold 3 banked sub-epochs with dev recogs; verified 2026-08-17 from
  rJWSC5xOsrf2's work/learning_rates (epochs 1-3 complete, trained 2026-08-08 22:10
  through 2026-08-09 09:41 — in flight when the 2026-08-09 park landed, held rather
  than deleted) and its empty `hold` marker. PARKED stands as a funding state (no
  further legs); the banked legs serve as the frozen-REPAIRED-scorer control at
  matched sub-epochs 1-3 for D6-PERIODIC/GAN.

**D4 — gated discrete outer refresh** (TRIGGER OVERRIDDEN BY THE USER 2026-08-08, replaces
"only if D3 plateaus": D4 starts NOW — the user's rationale, which D2's own read supports, is
that rate-matching is a targeted heuristic rather than a general strategy (d2_rate is the arm
that failed the paired read), so the general levers — mechanism-level pricing and refresh on
the policy's own decodes — take priority over the frozen-control ladder step. D3 as a separate
phase is PARKED, not cancelled: its control claim is preserved by folding a FROZEN comparator
arm into every D4 relaunch (see round-1 spec below); nothing else of D3 is funded.
PREREQS (2026-08-08, all cheap, all BEFORE any acceptance verdict): (a) the psi-preserving
checkpoint-import override (fact 1), (b) the selector filler-affinity row for lm_prior_units on
the existing D0 dump (the registered admissibility condition, one CPU row), (c) the
clause-table job from the D2 audit, reused as the per-round acceptance readout — and the
CI-convention pin above still needs the user's blessing, because it now decides D4
accept/reject verdicts, not just the D2 winner.):

BED REDIRECTED BY THE USER 2026-08-09 (replaces the 2026-08-08 G-track round-1 spec as the
ACTIVE work — that spec is PARKED below verbatim, not deleted, and with it the bad-init
self-repair read: the new bed's psi is GOLD-seed-trained, so this D4' advances the best
system, not the fully-unsupervised north star; the G-track round returns only on the user's
word): the refresh rounds run on the best arm — the theta_0' lbs 960 h 1-pass shaped loop
(`vhyvv2waeU16`, the D5 bed) — as ITERATIVE retraining of its psi on the policy's own
decodes, not a one-shot repair rooted in the gan-lineage corpus. The user's rationale: the
scorer's entire training set today is the 2849 gold seed pairs while the loop decodes 281 241
utts — the refresh grows the scorer on its own bed, with the seed as the natural anchor. The
mechanics bullets below are the SHARED machinery (both instantiations); the two round-1
specs follow them, D4' active, G-track parked. One fork checkpoint feeds THREE update-rule
arms on this bed — frozen continuation (the running arm, free) / continuous joint (D5(b)) /
gated discrete refresh (D4') — the complete update-rule comparison §3e.1 exists to make.
- The loop always runs on the last ACCEPTED frozen scorer; candidates train offline; rejection
  costs nothing (nothing enters the loop).
- Pairs: anchor = repaired round-0 text, share floored at 50%, accumulate never replace; plus
  curated best-of-group with a selector the scorer does not own AND that is not audio-free
  LM-prior-alone — the pseudo-text IS a student+word-LM decode, so that view authored the
  contamination, and an external LM favors filler function words. Two-view agreement +
  both-signs-advantage group contrast + per-utterance cap; selector admissibility = positive
  covariance in D0(e) AND a non-positive filler-affinity partial effect (suspect count on the
  selector signal at matched WER; added 2026-08-07 — this row is NOT yet computed for
  lm_prior_units, one arm-invariant addition to the existing D0 dump, required before any D4
  admission).
- Cadence: on a KL(policy||init) increment, not wall-clock (Gao); hard budget 3-4 rounds per
  bed; each round refits from scratch on the accumulated pool (the job has no resume; matches
  refit-from-base — Wolf & Kirk).
- On acceptance: re-point or reset the policy anchor (fact 7). Integration prerequisite: the
  psi-preserving checkpoint-import override (fact 1).
- G-TRACK ROUND-1 SPEC (2026-08-08, planner; PARKED 2026-08-09 by the user's bed redirect —
  revive only on the user's word; the implementer surfaces any deviation before training):
  PARK-TIME STATUS 2026-08-09: the round was already in flight — dump and curation are DONE
  (`SAE_3E1.md` approach 12, conclusions 16-19: an admissible external selector exists in
  `lm_prior_units`, the AR's own reward is inadmissible and worse-than-random as a selector,
  curation reaches 79 % of the bed BUT the curated picks are dirtier than the anchor they
  join — "to" rate 0.0462 vs 0.0275 — so on this bed refresh can only launder the policy's
  habits back into the scorer, which independently supports the park); the refit
  (`PsiAlignTrainJob.cRIigmxPtt75`) finishes on its own, its gate verdict is recorded as the
  round's closing datum, and NOTHING relaunches from it.
  (1) Dump: one fresh `ReturnnForwardJobV2` of theta_0^G at T=0.7 over the full tc100
  pseudo-text utterance set (G=8-12) — the D0 dump's 512 utterances are a read, not a corpus.
  (2) Curation exactly as pre-registered above (two-view agreement, both-signs-advantage,
  per-utterance cap; lm_prior_units enters only after prereq (b) shows a non-positive
  filler-affinity partial effect). (3) Anchor: the rate-REPAIRED round-0 corpus at the floored
  50 % share — kept deliberately: under the user's ruling rate-matching is HYGIENE, and hygiene
  on the stability anchor is free since the corpus is built; the user may swap to the raw
  corpus by saying so. (4) Candidate refit recipe: the d2_contrast recipe (contrastive term ON,
  cps 1.5) refit from scratch on anchor + curated pool — the winner mechanism carried into the
  refresh; d2_states' cps 0.5 is the registered fallback if the round-1 candidate fails gates.
  (5) Acceptance: gate v2 under the clause pins ((i) floor-only — every refresh candidate is
  changed-text; (ii) struck; (iii)/(iv) regression arms under the CI convention; (v) paired
  rank stability) plus `PsiScorerParityJob`. (6) Relaunch: TWO arms at the repriced operating
  point (prior share ~0.45, lambda=8 at cps 1.5) — the accepted refresh scorer AND the frozen
  incumbent-family comparator (the folded-in D3 control; without it a refresh gain is
  unattributable); per-sub-epoch monitors keep kill authority; one manager at a time on the
  960 h graph.
- D4' ROUND-1 SPEC (2026-08-09, planner; ACTIVE — the best-bed instantiation; planner-chosen
  constants are marked and the user may override any of them; the implementer surfaces any
  deviation before training):
  (1) FORK POINT, shared with D5(b): the last `vhyvv2waeU16` checkpoint passing the standing
  label-free selection rule plus a rate/length health screen — dev reward + LM score select,
  text_len/speaking-rate within band (the live exploit is and/but/i minimal-state insertions,
  `SAE_0d.md` approach 12); WER confirms, never selects. All three update-rule arms leave
  from this one checkpoint at matched remaining schedule.
  (2) Dump: one `ReturnnForwardJobV2` of the fork policy at T=0.7, G=8-12, over the tc100
  third of the bed (28 539 utts — the corpus size the psi_align^G reference setup used;
  planner constant, resize on the user's word).
  (3) Curation — AMENDED 2026-08-10 (planner; replaces two-view curation FOR ROUND 1 on
  this bed, because the pre-registered machinery ran and came back empty on both required
  inputs, `SAE_3E1.md` c26/c30, both spot-verified from the job outputs: the label-free
  suspect derivation returns the EMPTY set at the pre-registered min_excess 0.002 (largest
  excess "and" 0.00135 — the instrument is out of range on a 5.34/9.50 policy, not broken),
  and the admissibility table disqualifies every audio-conditioned view — psi's own score is
  filler-POSITIVE at matched WER (partial beta +0.2254 recon / +0.2029 shaped, CIs excluding
  zero) while `lm_prior_units`, the only view clearing both clauses, is the audio-free one
  the rule forbids to curate alone): ROUND 1 RUNS UNCURATED — no per-group selection at all.
  The refit corpus is the gold anchor at the floored 50 % share plus ONE GREEDY decode per
  tc100 utterance from the fork policy (one new `ReturnnForwardJobV2`; deterministic,
  label-free, and neither reward nor scorer touches corpus construction, which also retires
  the self-amplification concern for this round). Why this loses little: on the G-track
  round the same recipe on the UNCURATED corpus already achieved most of the discount
  reduction (0.0082 vs r1's 0.0064) at a hair better held ce_loo (c28), and this bed's fork
  decodes are 5.34/9.50-grade, so the refresh premise — policy text upgrades the scorer's
  2849-pair corpus by VOLUME at near-anchor quality — holds here without selection. The
  curation machinery and admissibility instruments stay wired and inert (`CURATION_VIEWS`
  unset) and are re-read on round 2's dump; user may override toward blocking round 1
  instead. Unchanged from the old (3): the filler watch reads the minimal-state class
  (and/but/i), and every insertion monitor here and in (5) reads absolute COUNTS, never
  shares (2026-08-09, D5(a)-1): the 100 h collapse grew insertions 6.4x while the suspect
  SHARE stayed flat at 0.05-0.07 because the added mass is generic function words — a
  share-normalized bar is structurally blind to this failure.
  (4) Refit: psi from scratch on anchor + the round-1 corpus of (3). ANCHOR = the 2849 gold seed pairs at
  the floored 50 % share, accumulate-never-replace across rounds — sanctioned on this track
  (they trained round-0 psi), gold-quality, and no gan-lineage text anywhere, per the user's
  direction. Recipe: the incumbent gold-psi recipe with the matching-aware contrastive term
  ON (the D2 winner mechanism; curated text is policy-authored — the same contamination class
  the term exists to price); contrastive OFF is the registered fallback.
  (5) Acceptance: the gate v2 machinery ported to this bed — held-NLL floor on the seed-dev
  gold split, the state-matched probe battery rebuilt on this bed's decodes, clause table
  under the CI pins (blessing still pending — it decides these verdicts too),
  `PsiScorerParityJob`.
  (6) Relaunch on acceptance: psi-preserving import; continue the fork policy with psi_1 for
  the remaining schedule, read against the frozen continuation at matched sub-epochs. The
  reward stays THIS arm's own — shaped lam=1 units-norm at T=0.7; the lambda=8 / prior-share
  0.45 repricing is a G-track-bed read and does NOT carry (per-bed rule); repricing this bed
  is a separate read if the user asks. Cadence for rounds 2+: KL increments, budget 2-3
  rounds inside the current 1-pass schedule — more passes is a new funding decision.
  REGISTERED ROUND-2 AXIS (2026-08-11, planner; announced in conversation 2026-08-10,
  recorded now): if round 1 ACCEPTS, round 2's single variable is corpus SCALE — one greedy
  decode per utterance over the FULL 960 h bed (~281 k utterances, ~10x round 1's 28 539),
  same recipe, cadence and gates. Prerequisites before launch: (i) an explicit anchor rule
  replacing the floored 50 % share — at ~100:1 repetition of the 2 849 gold pairs it is a
  different regime than round 1's ~10:1 and needs its own spec; (ii) per-subset held reads
  (the tc360/to500 decodes are noisier than tc100's 5.34/9.50 grade — scale must not
  silently trade text quality). Round 1 stays pinned at 28 539 (single-variable
  discipline).
  REGISTERED FOLLOW-ON, unfunded (2026-08-09, user raised psi capacity): a psi WIDTH probe —
  refit the accepted round's recipe at 2x d_model (384 -> 768) on the SAME corpus and read
  the existing offline battery only (held-NLL, state-matched probes, fixed-dump re-rank
  spearman/eta) — capacity is tested AFTER refresh grows the data, never confounded into a
  refresh round (round-1 stays single-variable); fund only if the user says so, and only if
  round-1's held-NLL suggests the corpus has outgrown 11M rather than the reverse.
  The 2026-08-10 SKIP-ARC PRICE STEERING probe is FOLDED INTO D6 rung 1 (2026-08-11,
  planner — the user directed a general insertion repair, not only the word-specific
  discount; spec, cautions and gates now live in D6 below).

**D5 — does co-training really cause the collapse? (USER-directed 2026-08-09).** The premise behind D4's offline-only shape rests on one uncontrolled run;
see the ATTRIBUTION REOPENED bullet in the verdict section for the full evidence ledger
(no frozen arm on the 100 h bed; the 10 h matched pair went the other way; two 100 h jointAR
siblings without the signature). Evidence FOR causality already on record (cite, do not
re-measure): CE_true crossed the unit marginal after ONE sub-epoch (5.7444 -> 6.2045) while dev
WER was still 18.79 — the scorer died before the WER collapsed (`SAE_3E1.md` c1-2).
SCOPE REDIRECTED BY THE USER 2026-08-09 (replaces the same-day scope note that pinned D5 to
the retired token-LM system, and the unexercised (b) 100 h `freeze_ar=True` replica + its
gate below — because the user's call is that the decision-relevant question is whether
co-training collapses the CURRENT best system, not the retired one): (b) becomes a JOINT-psi
control arm on the best model — the theta_0' lbs-adapted shaped 960 h 1-pass bed
(`ReturnnTrainingJob.vhyvv2waeU16`, RUNNING, checkpoints through sub-ep 4; dev-clean 5.34 at
sub-ep 2, insertion regression 6.56 at sub-ep 3, `SAE_0d.md` approach 12) — whose running
frozen arm IS the matched control, for free. The 100 h replica is DROPPED, not deferred; the
historical attribution of the 2S collapse therefore stays formally open and is carried by (a)
alone (decisive only if the allegiance grid is).
- (a) Collapse forensics FIRST (user-directed 2026-08-09; cheap, on the EXISTING ep0-ep6
  scorer checkpoints approach 1 already extracted — implementer asserts they still exist
  before building, keep_epochs trap): (1) policy anatomy per epoch from the existing sclite
  reports (CPU): ins/del/sub, %Corr, hyp/ref length, filler share of insertions — is the
  terminal state the insertion exploit again or a new error mode; (2) scorer allegiance grid
  (one `ReturnnForwardJobV2` sweep, fixed 512-utt dev subset): CE of {gold text, each epoch's
  own recog decodes} under each scorer checkpoint ep0-ep6 — the discriminating read: if
  CE(own decodes | ep-k scorer) stays low or falls while CE_true rises, the scorer FOLLOWS the
  policy and the reward became self-preference (the ratchet); if both rise, the scorer
  degenerates independently of the policy; (3) within-group ranking-vs-oracle of the ep-k
  scorer on one fixed rollout set per k — did the reward invert before WER moved. Descriptive,
  no gate; the result defines what "collapse" concretely is, which the D5 verdict and any
  future trainability alarm then reference.
- (b) JOINT-psi control arm (the user-directed treatment; the running frozen arm is the
  control): single-knob flip of `vhyvv2waeU16` — psi trains in-loop, everything else
  verbatim (same lbs prior at lam=1 / T=0.7, same partition_epoch so sub-epoch indices
  compare 1:1). AMENDED 2026-08-09 (same day, planner, replaces the restart-at-theta_0'
  form because the D4 redirect created the shared fork): (b) FORKS from the D4' fork
  checkpoint rather than restarting — one fork point, three update rules (frozen / joint /
  refresh) at matched remaining schedule, and half the cost. The 1.7x joint slowdown was the
  7B-class token-LM AR; psi is ~11M so the overhead should be small — the walltime assert
  stays regardless. Update rule RE-AMENDED 2026-08-11 (planner; replaces the 2026-08-10
  4-of-12 psi-CE subsample re-spec, because the implementer surfaced a fourth fix that
  rescues the FAITHFUL form the 08-10 amendment had judged infeasible after the first launch
  OOMed at step ~72 with ~2.6x step time — `SAE_3E1.md` c25 and its 08-11 correction): psi
  NLL on ALL G=12 sampled texts, shared optimizer, ce scale 1.0, NO in-loop contrastive
  term, run at `batch_size` 1e6 / `accum_grad_multiple_step` 2 against the parent's 2e6 / 1
  — effective batch, GRPO group, update count, `max_seqs` and the 1:1 sub-epoch indexing all
  preserved; the only delta is that each update's gradient is the mean of two half-batches.
  Measured fit: memory flat at 48.1 of 95 GiB, sub-epochs 35 982 s and 33 252 s (2.29x /
  2.12x the parent, inside the 11.5 h cap). The arm therefore ran at FULL coupling strength
  and the 08-10 lower-bound caveat is VOID. Retired unexercised: the 4-of-12 subsample
  (psi CE on a fixed seeded random 4 of G=12, detached ids) and its DP-checkpoint fallback. Budget: 4-6 sub-epochs, then STOP regardless of
  trajectory — the point is to observe, not to win. BUILD PREREQS, all ADDITIVE-ONLY because
  the running frozen arm re-imports the live recipe tree on every walltime resubmit (no
  executed line of the frozen path may change): (i) a trainable-psi channel behind a new
  default-OFF flag — a real `ce_loss` on the psi scorer plus a `train_psi` branch in the
  train step (fact 1 becomes "no path EXCEPT behind this flag"); (ii) a NEW config/graph
  sharing only the data/model builders, never the running arm's `baseline()` (the abandoned
  `..._960h_joint_v1` lesson — two managers must never own one train job), with the launch
  assert steps_per_sub_epoch / measured_steps_per_hour < 11.5; (iii) per-sub-epoch scorer
  forensics as first-class reads, instrumented DURING the run this time: CE_true on a fixed
  gold-pair dev subset, derangement contrast, and the (a)-style allegiance read (CE of the
  arm's own decodes under its own current scorer) — labels gate/evaluate only, never train
  or select. AMENDED 2026-08-09 (adds one read; because of the D5(a)-3 result): the
  pinned-policy re-rank probe joins the per-sub-epoch set — re-rank ONE fixed dump of
  fork-policy rollouts under each sub-epoch's scorer and read eta and sel-WER. On the 100 h
  bed this instrument flipped negative after ONE sub-epoch (eta +0.2246 -> -0.1185) while
  dev WER still looked survivable, so it — not end-of-arm WER — is the arm's real clock,
  and one sub-epoch is the number to beat (WER-derived, so gate/evaluate only).
- Gate (pre-registered, on (b); replaces the unexercised 100 h replica gate, user redirect,
  see SCOPE): read joint-minus-frozen dev-clean at matched points over the budget. COLLAPSE
  CONFIRMED if the gap exceeds +2 abs WER and grows for two consecutive read points with
  CE_true rising monotonically — then D4's offline-only premise is proven ON THE BEST BED and
  CE_true (absolute conditional fit) is the mandatory alarm on any trainable form.
  Co-training SURVIVABLE on this bed if the gap stays within the frozen arm's own largest
  sub-epoch-to-sub-epoch swing (currently 1.22, sub-ep 2 -> 3) through the budget with
  CE_true within +0.1 nats of its ep0 value. Anything between is PARTIAL: report both
  trajectories and stop for the planner. Either way D4 CONTINUES — the joint arm is a bounded
  control, never the mainline update rule; if SURVIVABLE, what returns is a design discussion
  with the user, not a silent switch.
- Status: QUEUED 2026-08-09 (user-directed; (a) unchanged, (b) redirected to the best bed
  same day). 2026-08-09 later: (a) COMPLETE (`SAE_3E1.md` approaches 13-15, conclusions
  20-22) — collapse is pure over-generation (%Corr RISES while insertions grow 6.4x; any
  share-normalized bar is blind), the scorer's preference migrates from gold to its own
  padded decodes on top of a dead conditional, and with the policy pinned the reward's
  ranking utility (eta) flips NEGATIVE after one sub-epoch. (b) remains the causal test, but
  its question sharpens from "does joint psi collapse" to "how fast on this bed" — reads at
  every sub-epoch, not only at the gate's matched points. 2026-08-10: the fork is PINNED at
  `vhyvv2waeU16` sub-ep 2 by the label-free screen (`SAE_3E1.md` c23, spot-verified; dev
  reward's own argmax picked the vetoed sub-ep 3 — the minimal-state COUNT screen is what
  carried the fork, first live confirmation of the counts-not-shares ruling). (b)'s faithful
  single-knob form is INFEASIBLE on the node (c25) — update rule amended above to the
  4-of-12 psi-CE subsample with the pre-registered lower-bound caveat; relaunch gated on the
  trial-measured walltime assert. D4' round-1: dump, incumbent battery, and admissibility
  reads DONE (c26-c27, c29-c30); the (3) amendment above unblocks the refit, uncurated.
  2026-08-11: (b) RAN, faithfully, under the implementer's batch-halving fix (re-amendment
  above; the 08-10 subsample re-spec never ran). Result (`SAE_3E1.md` approach 17, c32):
  sub-ep 1 is the best WER anywhere on this bed — 5.12 / 9.27 against the matched frozen
  control's 6.56 / 11.15 and the parent's all-time best 5.34 / 9.50, with insertions
  385 / 630, about a third of the frozen band — then sub-ep 2 collapses to 17.35 / 21.97
  with insertions 6114 / 6450 (~16x). Gate verdict PENDING by the gate's own terms: the
  +2-and-growing clause needs two consecutive read points and monotone CE_true; the CE_true
  forensics are unread and sub-ep 3 is in flight. Planner reading either way: the
  cliff-after-one-good-step shape — with the good step's mechanism being the insertion
  channel falling — is the strongest evidence yet FOR the gated discrete refresh and against
  any continuous update rule; whether the joint arm's sub-ep-1 artifacts (policy and/or its
  extracted psi) become production candidates is a USER decision — D5 is a measurement arm.
  2026-08-12 (planner, read from job outputs ahead of the log): VERDICT — COLLAPSE
  CONFIRMED by the gate's own terms. Sub-ep 3 is 41.8 / 50.9 (`ScliteJob.yVyM2WLvkXxG` /
  `.4mnMvy9mUVI7`, traced to `jQmmGy2yGtGR` epoch 3): the joint-minus-frozen dev-clean gap
  grows from ~+10.5 to >= ~+33 over two consecutive read points, and CE_true on the frozen
  gold pairs rises monotonically across them — 2.7614 (ep0) -> 2.6343 -> 2.7928 -> 2.9771
  (extracted psi at sub-eps 1/2/3). D4's offline-only premise is proven on the best bed;
  D5 CLOSES as a measurement arm. One refinement the (iii) forensics add: the CE_true
  LEVEL is a LAGGING alarm — at sub-ep 2, with WER already +10.5, it sat only +0.031 over
  ep0, inside the survivable +0.1 — while the allegiance gap (CE_true minus the CE of the
  arm's own decodes under the same checkpoint's psi: 2.6270 / 2.4726 / 2.2994) led the
  collapse: +0.007 / +0.320 / +0.678 at sub-eps 1/2/3. Any future trainable form carries
  the GAP as its in-loop tripwire, the level staying as the registered backstop. The
  sub-ep-1 artifact question remains with the user. D4' round-1's refit also read: the
  uncurated refit improves held fit and both insertion ladders but its clause table
  returns NO WINNER under both readings on one ladder (`lmsub` -0.0058, CI excluding
  zero; c33) — the gate-design question c33 poses is MOOT for production: the swap
  candidate is D6's d_min=2 refit on the same corpus, which repairs that ladder outright
  (0.9572, paired +0.0093 vs the comparator) and is point-wise not below psi0_gold on any
  ladder; r1_uncurated stands as D6's comparator, not a production candidate.

**D6 — structural insertion repair (USER-directed 2026-08-11).**
- Purpose: remove the topology half of the insertion under-pricing. Inserting any word is
  ~6x cheaper than deleting one (k=1 +0.0693 vs +0.4295 on the best bed, `SAE_3E1.md` c27)
  because the skip arc crosses an inserted state in ~half a frame; the refresh lever moves
  the word-specific discount only (c28: asymmetry -7.7 % vs discount -63 %). Goal, the
  user's words: a scorer that ranks as well as the incumbent but without the insertion
  cheapness. Measured NON-fixes, for the record: upweighting the external LM prior (the
  filler payment reaches zero only at prior share ~88 %, inadmissible) and corpus repair
  alone (the 7.7 %).
- Approach — three rungs, cheap to expensive; rungs 2-3 are parallel recipe candidates on
  the IDENTICAL corpus as the D4' round-1 refit, so the corpus axis (D4' rounds) and the
  recipe axis (D6) each stay single-variable. (1) PRICE STEERING, offline, no training
  (folds in the 08-10 probe): re-score the existing fixed dumps under (a) a flat arc-softmax
  bias against the skip arc and (b) a per-state minimum-duration cost (charge d_min minus
  frames held, when positive); sweep both; the same job computes rung-3's feasibility
  statistic — the fraction of pairs with T < 2*U_content. Caution carried over: the skip arc
  is also the phone-deletion arc and what makes optional silence optional
  (`psi_align.py:13-14`), so the substitution/deletion probe rows and held NLL check that
  insertion pricing is not bought at their expense. (2) CORRUPTION-TRAINED ARC PRICES,
  refit from scratch: the round-1 recipe and corpus plus synthetic insertion negatives —
  LM-corpus-drawn words inserted into training text (no annotations touched) — under a
  margin term charging the corrupted pair at least the clean pair plus margin*n_inserted.
  Rationale: the arc softmax is text-conditioned, so it CAN learn skip-is-expensive across
  content states while staying cheap where real silence or deletion needs it — pressure the
  clean-pair NLL objective never supplies, which is why today's skip transition price is
  near zero (+0.0065, c12). (3) MIN-DURATION TOPOLOGY, refit from scratch: content states
  must hold >= 2 frames (state splitting; skip arcs only across designated optional/SIL
  states), so an inserted word must absorb >= 2 real frames per state with wrong emissions —
  the same currency deletion pays in; gated on rung 1's feasibility statistic.
- Pre-registered ceiling, so the verdict cannot overclaim: at feasible minimum durations
  (mean T/U ~4.9 caps d_min ~2 for the tail) topology alone lifts insertion to roughly a
  quarter of deletion's price, not parity — rung 2's learned prices are what can carry it
  further, so rungs 2+3 together are the expected shape, not a fallback.
- Gate (pre-registered; numeric margins land with the pending CI-convention pin, which
  decides these verdicts too): on the same fixed dump and the frozen gold seed-dev —
  (i) within-group spearman and picked-vs-random WER not worse than the incumbent-recipe
  refit on the same corpus; (ii) held NLL within +0.05 nats; (iii) k=1 insertion price at
  least 2x the incumbent's (>= +0.14) with the gap growing in k; (iv) insertion no longer
  the least monotone edit class in the corruption ladder (now 0.658 vs 0.735-0.784). A
  candidate passing all four becomes the scorer-swap candidate for the fork continuation
  under the same protocol as a D4' acceptance; the in-loop confirmation read is the frozen
  arm's sub-epoch-3 insertion regression shrinking at matched points.
- Status: REGISTERED 2026-08-11. Nothing launched; rung 1 is implementer-ready (offline,
  existing dumps), rungs 2-3 wait on the round-1 corpus (greedy-decode job).
  2026-08-12 (planner, from `PsiGateClauseTableJob.JdrWdaCm7UeG` + the rerank jobs; log
  c34-c36): all three rungs RAN on the round-1 corpus. Rung 1 FAILS — the best of twelve
  settings lifts the k=1 insertion price 1.09x against the 2x bar; the carried caution was
  the mechanism (a charge both sides pay cancels in the paired read). Its feasibility
  statistic clears far wider than the ceiling assumed: 6.64 frames per content state, 3
  infeasible rows of 59 878 at d_min=2 and 7 at d_min=3 — d_min=3 is a live dial, and the
  ceiling under-predicted (ins/del already 0.345 at d_min=2). Rung 2 as-designed is
  REFUTED: the hinge learned its own negative distribution (LM-drawn controls repriced
  2.9x vs the filler's 2.4x), fails clause (iv) alone, and costs three ladders at CIs
  excluding zero when combined with rung 3 — the ceiling paragraph's "rungs 2+3 together
  are the expected shape" is WRONG; topology alone is the shape. Rung 3 (d_min=2) passes
  ALL FOUR clauses, including the (i) picked-WER half absent from the log's table
  (planner-read: sel_wer 0.05028 vs comparator 0.05219, random 0.05477/0.05613, oracle
  ~0.0412; 7 of 28 538 groups unscorable under the topology — verifier note in the log).
  Clause (iv) is determinate only under the CI reading (0 ladders CI-worse; the point
  reading counts 2, both CIs spanning zero) — the pending user CI-vs-point pin decides
  this verdict too. The v2 winner test returns NO WINNER on the level-currency matched
  discount (recorded verdict stands unedited); planner pin 2026-08-12 for cross-arm reads
  going forward: the matched discount is a magnitude in the arm's own units, so cross-arm
  comparison uses its SHARE of the arm's own k=1 insertion price (the standing
  per-arm-magnitude principle) — psi0_gold 11.3 %, comparator 12.3 %, d_min=2 7.1 %, a
  reduction. Per the gate's own sentence d6_mindur is the scorer-swap candidate for the
  fork continuation, funding gated on the user's convention pin; the d_min=3 refit is the
  one cheap adjacent probe (same recipe, same clause table); the rung-2 redesign is PARKED
  until the swap-in continuation reads.
  USER 2026-08-12: swap-in APPROVED, and the min-duration test extends to BOTH beds. (a)
  Best bed (10h-init 960h loop): the registered swap-in continuation, funded. (b) G-track
  (gan-init 960h loop): the TOPOLOGY transfers, checkpoints do not — refit the G-track's
  own round-1 refresh recipe with the min-duration topology on the G-track round-1 corpus
  (single variable vs the G-track round-1 refit `cRIigmxPtt75`, whose own clause-table
  verdict is read first as the comparator baseline), run the same four D6 clauses on
  G-track instruments, and on pass swap into the G-track loop against its control at that
  bed's registered operating point (per-bed reward rule; no cross-bed checkpoint reuse).
  CI-vs-point: the user approved the planner's recommendation in the same message but
  asked for the convention in plain words before it binds — PENDING CONFIRMATION; every
  clause table stays dual-reported (both readings) until then.
  USER 2026-08-14: START NOW — the best-bed swap-in continuation ("D4' with min duration")
  launches on the user's word without waiting for the convention pin: the pin words the
  acceptance verdicts, it does not shape the run, and the user's directive is the funding
  authority (noted for the record: d6_mindur's clause-(iv) eligibility holds under the CI
  reading only; dual-reporting continues). Spec unchanged from the registered sentence:
  fork checkpoint + d6_mindur FROZEN (`PsiAlignTrainJob.wlruSpBK1EDP`) vs the frozen
  control at matched sub-epochs, confirmation read = the sub-epoch-3 insertion regression
  shrinking at matched points, bed-registered reward settings. The d_min=3 refit stays the
  cheap parallel probe through the same clause table. The G-track half is NOT covered by
  this start — its prerequisite (read `cRIigmxPtt75`'s own clause verdict first) stands.
  2026-08-15 (planner; final rows verified from `PolicyAnatomyJob.pxqfrYx23Rth`): the
  best-bed swap-in continuation is COMPLETE and PASSES its pre-registered confirmation
  outright — no sub-epoch-3 regression at all (fork 5.34/9.50 -> 4.68/8.64 one sub-epoch
  after the swap), 4.73/9.31 at sub-epoch 10 vs the control's 6.46/11.41, dev-other
  insertions 933 vs 1964 (log c39). Planner read of the SHAPE: the entire gain lands in
  the first post-swap sub-epoch and the frozen scorer buys nothing after it (dev-other
  never again under 8.64, mild upward drift to the end) — the staleness signature of a
  scorer fit on the fork policy's decodes, though the parent's cosine tail and the
  control's own flatness leave schedule exhaustion unseparated as a cause. The round-2
  proposal that stood here is SUPERSEDED same day by the user-directed periodic arm
  (D6-PERIODIC below), which subsumes it — round 2 is that arm's first refresh. The
  G-track refit read (c37, NO WINNER on the substitution
  ladder) awaits planner verification before its plan verdict is recorded here.
  2026-08-17 (planner): c37 VERIFIED bit-for-bit from `PsiGateClauseTableJob.qYRE7JWyUcJQ` /
  `.H9QbX4VgXAwf` (verifier bullet in `SAE_3E1.md` same date; vs `psi_g_tc100` the refit is
  CI-worse on del too, while its matched discount is CI-lower at k=4). PLAN VERDICT: the
  one-shot G-track swap-in registered by USER 2026-08-12 (b) does NOT proceed — `r1_mindur`
  fits the held set 0.34 nats better and wins both insertion ladders but is ineligible on the
  substitution/deletion side under both readings. SUPERSEDED by the USER-directed
  D6-PERIODIC/GAN arm (below), whose per-round from-scratch refits carry the topology under
  the 2026-08-15 standing min-duration rule; `r1_mindur` retires to a comparator role (its
  corpus — curated, rate-repaired anchor — is not the periodic arm's pool recipe).

**D6-PERIODIC — per-sub-epoch scorer refresh on the best bed (USER-directed 2026-08-15).**
- Purpose: the one-refit continuation measured a fresh scorer's useful lifetime at about
  one sub-epoch — the entire gain landed in the first post-swap sub-epoch and the frozen
  scorer bought nothing after it. This arm asks whether refreshing at every sub-epoch
  boundary keeps the steps coming, and it separates scorer staleness from schedule
  exhaustion by construction: identical schedule to the finished arm, only the scorer's
  recency changes.
- STANDING RULE (user, 2026-08-15): every scorer fit under any new plan, on any bed or
  track, carries the min-duration topology (d_min=2 today; the parked d_min=3 probe may
  raise the dial, never remove it). The Z-track already complies — its psi is a
  min-duration psi_align from birth (`SAE_3G.md`).
- Cadence pin: ONCE PER SUB-EPOCH BOUNDARY. Grounds: the measured one-sub-epoch useful
  lifetime means refreshing less often provably wastes sub-epochs at this operating
  point; the CUDA refit (c38, ~1 h) is small against a sub-epoch of training; the
  cadence matches Z4's registered refresh so the two tracks read on each other; and
  nothing measured supports refreshing inside a sub-epoch, which also has no natural
  decode point.
- Approach: re-fork from the SAME parent sub-epoch-2 checkpoint the finished arm used
  and re-run sub-epoch 3 with the round-1 scorer unchanged — that leg replicates the
  finished arm's first leg by construction (its dev read doubles as a free replication
  check of 4.68/8.64) and avoids the cold-optimizer mismatch a mid-arm restart would
  introduce (the kept `epoch.001.pt` carries no optimizer state). From the 3->4 boundary
  on, at every boundary: (1) decode the refresh corpus with the round-1 recipe UNCHANGED
  (gold anchor 50 % + one greedy decode per utterance; the only per-round variable is
  the decoding checkpoint); (2) refit the d_min=2 recipe FROM SCRATCH on it, CUDA
  forward-backward path; (3) run the per-round acceptance gate below against the current
  incumbent on the standing frozen instruments (same corruption draw, frozen held set,
  fork rerank dump); (4) swap on pass, keep the incumbent on fail, log either way.
  Sub-epochs 3-10 run on the parent's own cosine tail — matched point for point to the
  finished one-refit arm, which is the control for free. Extension past sub-epoch 10 is
  a separate decision after the read.
- Per-round acceptance gate (pre-registered; adapted from the four D6 clauses because
  clause (iii)'s 2x-insertion bar is a one-time repair bar that would compound absurdly
  round-over-round): (i) within-group spearman and picked WER not worse than the
  incumbent's on the fixed dump; (ii) held NLL within +0.05 nats of the incumbent's;
  (iii') k=1 insertion price not below the incumbent's at a CI excluding zero; (iv') no
  corruption ladder worse at a CI excluding zero. Dual-reported CI/point until the
  convention pin. TWO CONSECUTIVE FAILED ROUNDS = the refresh ladder is dry: stop
  refitting, run the arm out frozen, read as exhaustion.
- Arm gate (pre-registered): PRIMARY — some refreshed round sets a new best dev-other
  below 8.64 (the finished arm's best, never re-touched after its first step), and the
  arm ends at sub-epoch 10 not above the finished arm's 4.73/9.31. SECONDARY — dev-other
  insertions at or under the finished arm's at every matched point (the D6 repair must
  survive refits). Registered failure reading: rounds keep passing acceptance yet none
  breaks 8.64 = the plateau was never scorer staleness but schedule/reward exhaustion at
  this operating point; verdict "not funding higher refresh frequency", and the cadence
  question closes. A failed gate licenses not-funding, never "refresh would not have
  worked elsewhere".
- Status: REGISTERED AND FUNDED 2026-08-15 on the user's word ("new version of D6 with
  more often refit"; the min-duration standing rule from the same message). Build order:
  (i) the per-round unit as one sisyphus chain (boundary decode -> from-scratch refit ->
  clause table -> conditional swap), (ii) the continuation wiring that unit at every
  boundary, (iii) launch. Planner constants the user may override: the per-round clause
  set, the two-consecutive-failures stop rule, the re-fork start (vs the one-sub-epoch-
  cheaper mid-arm restart), and the cadence itself.

- Status 2026-08-18 (D6-PERIODIC-WARM sub-arm registered; planner rulings). USER-directed
  2026-08-18: the sibling's from-scratch refits make it a clean read on corpus recency and
  a structurally blind one on scorer accumulation; the warm sub-arm changes exactly one
  argument — the boundary refit CONTINUES from the previous round's scorer (commit
  5773910; single-argument diff at the refit site verified; warm-start test re-run by the
  planner, passes; hash neutrality of the init_model exclusion proven from the new job's
  own input list, which resolves the pre-change hash of the round-1 incumbent's fit).
  Leg 1 precedes the first warm start and is the sibling's own finished leg, shared by
  hash — the arms part at the boundary that produces leg 2's scorer.
- RULING (warm-source reading, 2026-08-18): the warm start is the INCUMBENT's model — the
  scorer actually live in the reward during the previous leg — never a gate-rejected
  fit's raw output. Accumulation is only meaningful along the chain that shaped the
  reward; the readings coincide on accepted rounds and the incumbent reading stays
  monotone with the gate's own verdict after a rejection. Disclosed mechanism: inherited
  weights carry earlier rounds' corpora forward — that is the arm's question, and it does
  not touch the no-replay rule, whose subject is refit DATA (still current-policy decodes
  only, verified at the pool input).
- RULING (gate and evidence reading, 2026-08-18): the four-clause acceptance gate is KEPT
  so the arm stays single-variable against the sibling; because the candidate is a
  continuation of its own comparator, clause passes are structurally easier here.
  REGISTERED NON-READ: accept counts are never compared across the warm arm and the
  sibling. BINDING READ: plain dev-clean/dev-other WER at matched parent sub-epochs
  against the sibling, legs 2-10; the minimum meaningful difference is the sibling's
  round-1 replication spread at the same read points (submitted 0.29/0.24, planner
  verification due with the batch report); differences inside it read as NO EFFECT. If
  the warm arm also stays inside the floor, the periodic question closes as: neither
  refresh frequency nor scorer inheritance moves the loop at this operating point.
- RULING (economy, 2026-08-18): the online/offline parity check is not re-spent (a
  property of code and topology, both identical). The matched-point policy-anatomy job is
  NOT funded now — the offline sclite read is the gating read; anatomy is priced only if
  the trajectory separates beyond the floor.

- Status 2026-08-18 (GATE TRACE COMPLETE THROUGH ROUND 5 -- verdict on the arm's
  question; artifacts planner-verified: PsiRefreshAcceptJob .H6AbNweDfzTF / .lWmT0OpDXfSp
  / .9NV2smxymaZX / .G7zxNoiYrTke). The arm has never deployed a refit: rounds 2 and 3
  fail clause (iii') -- the k=1 insertion price falls below the incumbent's at a CI
  excluding zero -- under the binding CI reading; the two-consecutive-failure stop rule
  entered force at round 4, where the candidate passed ALL FOUR clauses under CI and was
  overridden per the registration; round 5 fails (iii') and (iv') on its own. Legs 2-8
  therefore all run the round-1 scorer. Disclosed deviation, cost-only: refits and
  clause tables for rounds 4-5 were still measured after the registered "stop refitting"
  point (never deployed) -- that extra measurement is what surfaced the round-4
  counter-cell. VERDICT per the pre-registered dry reading: the refresh ladder is dry --
  from-scratch refits on the policy's later decodes could not beat the round-1 scorer on
  the clauses in three of four rounds, failing precisely the insertion-price guarantee
  the topology exists to hold -- so scorer staleness was not the binding constraint at
  this operating point; not funding higher refresh frequency. The arm gate's PRIMARY
  read (a refreshed round setting a new best below 8.64) is VACUOUS -- no refreshed
  round exists -- and the matched-sub-epoch comparison against the one-shot arm is void
  AS a refresh read: under a frozen incumbent the two arms differ only in the shard rule
  and the Adam restarts, so that comparison is re-labeled a replication read and its
  spread (implementer-measured 0.29 dev-clean / 0.24 dev-other at round 1, rounds 2-5 at
  or inside it; planner verification due with the batch report) becomes the bed's
  run-to-run noise floor. A failed gate licenses not-funding, never "refresh would not
  have worked elsewhere".
- RULING 2026-08-18 (WARM arm: amendment by replacement + pre-registered source fork;
  registered while the round-2 warm verdict does NOT exist -- its refit was mid-training
  at epoch 26/30 at check time). Replaces the BINDING READ clause of the 2026-08-18
  gate-and-evidence ruling above, because the sibling froze at the round-1 scorer: the
  warm arm's comparison now reads "does an accumulating scorer chain beat the FROZEN
  round-1 scorer at matched parent sub-epochs" -- a cleaner question than refresh
  frequency; the replication floor stays the minimum meaningful difference, legs 2 on.
  WARM-SOURCE FORK: HOLD the incumbent source (no change now; rounds 3+ are unbuilt).
  If the round-2 warm candidate is ACCEPTED under the registered gate, the incumbent
  ruling stands -- the chain forms. If it is REJECTED, the incumbent source is
  degenerate in practice (the warm start would freeze at the round-1 scorer and the
  legs would duplicate the sibling's job for job), and the arm switches its warm-start
  source for rounds 3+ to the previous round's refit output -- the chain the arm exists
  to test then always forms while the gate keeps controlling what goes LIVE in the
  reward; the switch is surfaced to the user as taken-unless-overridden. Grounds: on an
  accepted round the two readings coincide, so switching early buys nothing; and a gate
  rejection is a deployment decision, not evidence a continuation chain is worthless.
- USER RULING 2026-08-18 (relayed by the implementer; the user chose between the two
  readings in plain terms and answered "B is correct, yes" -- B being the scorer actually
  grading the next pass, i.e. the gate-controlled incumbent). REPLACES the pre-registered
  source fork above, same day: the warm-start source stays the incumbent under EVERY
  verdict and the fork does not fire on a round-2 rejection; no code change was ever
  made, so nothing unwinds. Disclosed consequence the user accepted: if this arm's own
  gate also rejects twice, the stop rule freezes the grading scorer at the round-1 one,
  every later boundary fits from that same file, and legs 3-8 replicate the sibling
  rather than test accumulation.
- DECISION RULE registered for that contingency (planner, resource-conservative, before
  the round-2 verdict exists): at WARM DRY (two consecutive rejections) the refits stop
  per the standing rule and the implementer lets NO further leg start -- the remaining
  legs (~5 GPU-h each) are a pure second replication sample under the frozen scorer, and
  whether to buy that sample is the user's resource call; planner recommendation on
  file: stop and bank the refit verdicts. A single round-2 rejection alone changes
  nothing: round 3's refit still runs (cheap) and can still form the chain.
- CORRECTION 2026-08-18 (gate bookkeeping; implementer-found, planner-verified at
  source): the per-round gate's recorded "label-free subset" overstates by one clause --
  refresh_gate.py names (ii)-(iv) as touching no annotation, but clause (ii)'s held set
  is d5b's gold_held_pairs, which is gold dev text by its own docstring. The plan
  already carries the correct fact (the GAN arm's no-gate rationale names the held set
  as gold text), so no plan text moves; on this gold-seeded track the gold read is
  sanctioned, disclosed supervision. No decision moves: (i) and (ii) passed in all four
  rounds and every rejection was carried by (iii), with (iv) joining at round 5
  (verified from the four verdict artifacts). Code fix -- constant, docstring, and the
  report's "read no annotation" line narrowed to (iii)-(iv) -- is the implementer's,
  AFTER the pending warm round-2 verdict job lands (live-source-import trap on a queued
  job).
- USER RULING 2026-08-18 (relayed; verbatim in substance: "The quality should NOT be
  decided by these metrics you come up with but not necessarily reflect performance!
  For all periodic, remove the gate completely. Delete all those jobs and restart
  running."). Executed by the implementer, commit 3257edc; planner-verified: acceptance
  step removed from both gold-seeded periodic configs, the gated run's jobs deleted
  (user-directed), both arms relaunched gate-less -- boundary = decode, pool, refit,
  hand the fit on, the label-free arm's shape, with that arm's priced cost now carried
  here too (a bad refit is carried, never rejected; the arms read as trajectories).
  SUPERSEDED by this ruling, kept above as dated history: the per-round acceptance-gate
  registration and its dry-stop rule; the forward-looking clause of the gate-trace
  verdict (the "not funding higher refresh frequency" verdict binds the GATED design
  only -- the user re-funded per-boundary refresh WITHOUT a gate, and the restarted
  arms ask the recency and accumulation questions directly); every warm-source ruling
  and fork (when every fit is handed on, the fitted scorer and the serving scorer are
  the same object and the readings coincide); and the wait-for-the-verdict-job
  constraint on the refresh_gate.py fix (no periodic arm imports the module now; fix at
  leisure -- it remains the D4-prime machinery's). CARRIED OVER unchanged to the
  restarted arms (planner constants, user may override): the arm-level WER gate --
  PRIMARY, some round sets a new best dev-other below 8.64 and the arm ends at
  sub-epoch 10 not above 4.73/9.31; SECONDARY, dev-other insertions at or under the
  one-shot arm's at every matched point -- it reads recognition performance, which the
  ruling endorses, not scorer statistics. Pointer repair: the four PsiRefreshAcceptJob
  hashes cited in the gate-trace Status above are deleted (two dirs lingered at planner
  check time); the numbers are preserved in SAE_3E1.md approach 22, planner-verified
  against pre-deletion first-hand reads.
- REPLICATION FLOOR registered 2026-08-18 (replaces BOTH earlier floor clauses -- the
  0.29/0.24 two-point spread and the five-leg range 0.73/2.13 -- because the five gated
  legs are successive segments of ONE trajectory at different schedule positions, leg k
  training from leg k-1's checkpoint on a decaying cosine, so their across-leg spread
  conflates schedule evolution with noise and is not a matched-point floor). The
  run-to-run measure on this bed is the five PAIRED matched-point deltas between the
  gated periodic legs and the one-shot arm at the same global sub-epoch, both running
  the identical round-1 scorer (planner-computed from the two logged tables): dev-clean
  0.29 / 0.03 / 0.32 / 0.36 / 0.19 (max 0.36, median 0.29); dev-other 0.24 / 0.30 /
  0.32 / 1.30 / 0.11 (max 1.30, median 0.30). Reading rule: a claim from a SINGLE
  matched point must clear the max paired delta (0.36 / 1.30); a consistent-sign
  difference across four or more matched points reads against the paired median
  (0.29 / 0.30). The five-leg range (0.73 / 2.13) is banked as within-trajectory wobble
  across schedule positions, nothing more. Cross-check: the one-shot-vs-control
  separation (SAE_3E1.md conclusion 39) clears both rules at every matched point.
- RESTARTED-ARM READS (registered 2026-08-18, before any ungated leg-2+ number
  exists): ungated periodic vs the one-shot arm at matched sub-epochs = scorer RECENCY;
  ungated WARM vs ungated periodic = ACCUMULATION vs fresh fit, single variable, the
  chain now forming by construction. Both read on plain WER at matched parent
  sub-epochs under the paired floor above; leg 1 stays shared across all three arms
  (ReturnnTrainingJob.5FqdnhWTOf1f).
- Status 2026-08-20 (ungated prefixes, planner-verified from concrete training and sclite
  work directories; neither arm has ended). Fresh-from-scratch periodic has completed legs
  1-5 at dev-clean/dev-other 4.97/8.88, 4.65/9.02, 5.28/9.27, 6.05/10.56 and 7.42/12.68;
  WARM has completed 1-5 at 4.97/8.88, 5.07/9.19, 4.85/9.04, 6.39/11.19 and
  12.18/19.33. Both leg-6 jobs are running (`ReturnnTrainingJob.n8abYvLR4IP5` fresh,
  `.kkh0u4rI7I6D` warm) with live managers and no error markers; legs 7-8 remain
  dependency-unbuilt. The fresh arm never
  beats the one-shot scorer's registered 8.64 best, and by matched leg 5 is worse than
  one-shot by +2.72/+3.56 and the frozen control by +0.73/+1.65. Its late dev-other loss is
  almost entirely insertions (479 -> 2,489, substitutions 3,572 -> 3,597); WARM compounds
  this to 5,874 insertions and is +4.76/+6.65 worse than fresh at leg 5. Provisional verdict:
  scorer inheritance is harmful here, while periodic fresh refits do not beat a good frozen
  scorer. The closest continuously trainable-scorer control, D5(b)-b, reads 5.12/9.27 ->
  17.35/21.97 -> 41.78/50.88 over its three banked sub-epochs, dominated by insertions
  (630 -> 21,406). Thus freezing the scorer within each sub-epoch avoids same-step joint
  collapse, but this is not a single-variable causal comparison: D5(b)-b uses the gold d_min=1
  scorer, continuous full-dataset partitioning and 1e6/accum-2 batching, whereas periodic uses
  d_min=2 outer refits, round-robin shard jobs and Adam restarts.
  TWO PRIMARY 10 h-INIT ANCHORS, stated explicitly: the adapted-donor AV-SFT checkpoint
  theta_0' before any loop is 11.43/15.54 dev-clean/dev-other; the best previous frozen-scorer
  loop checkpoint is the D6 one-shot d_min=2 swap at 4.68/8.64. The original incumbent-scorer
  loop's 5.34/9.50 fork remains a secondary historical anchor, not the best frozen result.

**D6-PERIODIC/GAN — the periodic-refresh arm on the label-free init (USER-directed
2026-08-17; launched by the implementer same day; planner registration from the launched
config, verified at source).**
- Purpose: the §3d standing question under the update rule that worked on the best bed.
  Both frozen-scorer G-track arms held because init and scorer share the §1d pseudo-text
  defect; the admissible fix §3d names is an outer re-estimation step between passes, and
  a from-scratch refit on the policy's own decodes at every sub-epoch boundary is that
  step. This is also the bed where the incumbent scorer ranks WORSE than random on the
  full bed (ar_recon eta -0.1103) against a ~6-point oracle-random gap — maximal measured
  headroom for a scorer-side fix, and the strongest a-priori case any refresh arm has had.
- As launched (`config_sae_3e1_d6periodic_gan_v1`, own manager; planner read the config
  source and the implementer verified the emitted returnn.configs): init theta_0^G
  (`ReturnnTrainingJob.2fb02hGUdHNj` ep10, 13.89/18.34), shaped arm only, 8 rounds. Per
  boundary: full-tc100 dump at T=0.7 -> anchor-free `GreedyPoolJob` (the pool text is a
  separate greedy argmax pass; the scorer never reaches pool construction, it only fills
  the dump's diagnostic reward columns) -> d_min=2 from-scratch refit (contrastive term
  ON, CUDA fast-bw path, c38) -> handed STRAIGHT to the next leg. Round 1 refreshes too
  (round-1 scorer = a fresh refit on theta_0^G's own greedy decodes,
  `PsiAlignTrainJob.dsMKgPHQApyR`). Reward = the held shaped arm's kwargs verbatim
  (lam_lm 1.0, lm_prior_norm "units", no lam_len); one cosine over the held arms' 30
  sub-epochs walked at epoch_offset k-1, partition_epoch 10 asserted against the held
  arms' own builder — leg k sits exactly at the held arms' sub-epoch k, so they are
  matched-point controls by construction.
- TWO LABEL-HYGIENE DELETIONS vs the D-track sibling (USER-directed in the implementer's
  session 2026-08-17, recorded here from the implementer's report): (1) NO gold anchor in
  the refit pool (the sibling's 50% gold floor is sanctioned on the gold-seeded track,
  inadmissible on the label-free claim); (2) NO per-round acceptance gate — the gate
  battery's held pair set is gold text and a gate SELECTS what trains the next leg, i.e.
  annotation-derived selection; and the only label-free held text on this bed is §1d
  pseudo-text whose domain IS the contamination (the measured gate v2 (i) confound), so a
  pseudo-text gate would select FOR the defect. Priced cost, stated not hidden: a bad
  refit is carried, never rejected — the arm reads as a trajectory (Z4's trade, same
  reason).
- Planner ratifications and corrections (2026-08-17, pre-registered before any leg
  output is read): (i) the fresh round-1 refit is RATIFIED (uniform recipe, label-free);
  the launch note's reason "no G-track d_min=2 scorer exists" is corrected for the
  record — `r1_mindur` exists and is c37's NO WINNER; different corpus recipe, retired to
  comparator. (ii) lam=1 is RATIFIED as a departure from the parked round-1 spec's
  repriced point (prior share ~0.45): the lambda=8 scalar was derived for cps-1.5
  d_min=1 scorers and does not transfer across scorers, no prior-share read exists under
  a scorer that did not exist pre-launch, and the verbatim held-arm reward buys the
  single-variable read against the held shaped control. REGISTERED FOLLOW-UP (CPU,
  label-free, never edits this arm mid-flight): prior-share and in-group-spearman lambda
  grid on the round-1 dump under the round-1 refit — informs whether a repriced variant
  is ever worth funding. (iii) Kill authority is label-free ONLY: per-leg insertion
  COUNTS (minimal-state class, counts never shares), within-group reward std, dev reward
  + LM score. Every gold read (per-leg WER, any CE_true/allegiance forensics) reports
  and can never gate, select, or kill. (iv) Attribution, fixed before any read: against
  the held shaped arm the contrast carries scorer topology AND recency TOGETHER (no
  G-track arm ever froze a d_min=2 scorer) — a win belongs to the refresh package. The
  topology- and schedule-matched frozen control registered below now separates scorer recency
  from that package.
- Arm read (pre-registered; planner constants, user may override): PRIMARY — some leg's
  unsupervised-selected checkpoint clears theta_0^G's 18.34 dev-other by >= 0.5 abs (the
  level the held shaped arm never cleared), and from leg 2 on the trajectory sits below
  the held shaped arm at every matched sub-epoch it has (held arm ran to sub-ep 4; past
  that the theta_0^G bar is the only anchor). SECONDARY (mechanism confirmation) —
  dev-other insertion counts fall across legs and stay below the held arm's at matched
  points. Registered failure reading: a trajectory inside the held arm's own band means
  the binding defect on this bed is not scorer staleness/topology — the lever returns to
  the init (§1e ep50 pins / §3d round-2); the verdict licenses "not funding more refresh
  here", never "refresh would not work elsewhere".
- Status: LAUNCHED 2026-08-17 (round-1 dump queued at launch; 8 rounds registered).
  D3 question RESOLVED 2026-08-17 (implementer evidence, planner-verified from
  rJWSC5xOsrf2's learning_rates + hold marker): D3 ran and is HELD at 3 banked
  sub-epochs per twin — my "no G-track D3 arm ran" was wrong, corrected in the D3
  block above. MATCHED-CONTROL COVERAGE, fixed before any leg is read: legs 1-3 have a
  three-way ladder at matched sub-epochs — held shaped (frozen CONTAMINATED psi_g) /
  D3 (frozen REPAIRED d2_contrast, d_min=1) / this arm (per-leg-refreshed d_min=2) —
  which separates "any scorer repair" from the refresh package, though topology and
  recency stay joint between D3 and this arm; leg 4 has the held shaped arm only;
  legs 5-8 have NO matched control and read against the theta_0^G bar alone — later
  legs are weaker evidence by construction (implementer-flagged, recorded). The
  pre-registered arm read is unchanged: its clause already binds only "at every
  matched sub-epoch it has".
- Status 2026-08-20 (six-leg prefix, planner-verified from concrete work directories;
  arm not ended). Dev-clean/dev-other is 14.45/19.69, 12.85/17.89, 13.20/18.20,
  17.76/23.17, 17.92/23.27 and 18.38/24.01. Leg 7
  `ReturnnTrainingJob.QTQuYQnppmSs` is running; leg 8 is dependency-unbuilt, the manager is
  live, and no error marker is
  present. Only leg 2 improves the no-loop theta_0^G dev-other 18.34, by 0.45, missing the
  registered >=0.5 bar by 0.05; the improvement is transient. At the matched first three
  points it does not beat the frozen repaired scorer decisively, and by leg 6 it is
  +4.49/+5.67 worse than its init. Unlike the D-track periodic failure, the late loss is
  chiefly substitutions (4,110 at the best leg 2 -> 7,331 at leg 6), while insertions stay
  near 4.3-4.5k. There is no same-init continuously trainable-scorer arm, so the D5(b)-b
  contrast is qualitative only and cannot attribute this trajectory to within-leg freezing.
  TWO PRIMARY GAN-INIT ANCHORS, stated explicitly: theta_0^G AV SFT before any loop is
  13.89/18.34; the best previous frozen-scorer loop checkpoint is the shaped arm with the
  repaired d2_contrast scorer at 12.68/17.57. GAN+HOM has its own 16.67/21.45 AV-SFT anchor
  but no same-init frozen-scorer loop; 12.68/17.57 is therefore cross-init context for it,
  not a controlled baseline. CLARIFICATION 2026-08-20: even for plain GAN, 12.68/17.57 is a
  reference baseline, not a schedule-only control. It freezes a d_min=1 d2_contrast scorer
  trained once under the D2 recipe and carries Adam through one multi-sub-epoch job; periodic
  uses from-scratch d_min=2 scorers trained on each policy's anchor-free greedy pool and one
  training job with a fresh Adam state per leg. A schedule-only test requires freezing the
  periodic arm's own d_min=2 round-1 scorer inside the same segmented-leg graph.
  WHY d_min=1, source/history audit 2026-08-20: it was inherited, not selected against
  d_min=2. D2 was committed 2026-08-07 as a single-variable contrastive-objective repair whose
  topology stayed byte-identical to psi_g_tc100; PsiAlignTrainJob did not yet expose min_dur.
  D6 introduced the minimum-duration topology on 2026-08-11 after the structural insertion-price
  diagnosis, and the later standing d_min=2 rule applied to new refits rather than retroactively
  rebuilding D2/D3. Scorer topology is therefore the main scientific confound in this comparison.

**D6-PERIODIC/GAN-FROZEN — topology- and schedule-matched frozen control (USER-directed
2026-08-20).**
- Purpose: isolate the value of refreshing the scorer after round 1. Rebuilding the historical
  D2/D3 frozen scorer with `d_min=2` would remove only the topology mismatch while retaining a
  different scorer corpus and one continuous policy-training job, so it would not answer the
  schedule question completely.
- Approach: start from the exact theta_0^G checkpoint and freeze the existing periodic arm's
  exact round-1 scorer, `PsiAlignTrainJob.dsMKgPHQApyR`, for all eight legs. Reuse the
  D6-PERIODIC/GAN segmented policy graph verbatim: the same 960 h round-robin shard at each leg,
  shaped reward and T=0.7, cosine epoch offsets, batching, checkpoint imports and fresh Adam
  state per leg. There are no scorer dumps or refits after round 1. Thus scorer recency is the
  only experimental difference from D6-PERIODIC/GAN.
- Experiments: verify leg 1 reproduces the periodic arm, then compare frozen and periodic policy
  checkpoints at every matched leg through leg 8. Plain dev-clean/dev-other WER is primary;
  substitution/deletion/insertion counts explain any separation but do not select checkpoints.
- Gate: this control does not change the periodic arm's absolute gate. A durable/actionable recency
  benefit requires periodic leg 8 to beat frozen leg 8 on both dev-clean and dev-other.
  Intermediate matched-leg deltas may establish a transient effect at those operating points but do
  not select an endpoint or license continued refresh; a one-leg crossing is inconclusive as a
  durable benefit.
  A split trade-off or a frozen final-leg win means scorer refresh has no established durable
  benefit here. The historical D3 comparison remains evidence for scorer repair as a package, not
  for recency.
- Status: IMPLEMENTED AND LAUNCHED 2026-08-20; verifier pass, no result yet. Leg 1 resolves to the
  banked periodic job `ReturnnTrainingJob.kr1foUV6lecx`; every leg reads the exact round-1 scorer
  `PsiAlignTrainJob.dsMKgPHQApyR`, and no dump, pool or refit exists after round 1. Leg 2 is queued
  under cluster maintenance. The gate above is unchanged.
  2026-08-22: COMPLETE, gate DECIDED, VERIFIER-CONFIRMED (all 32 cells traced to their own
  scoring jobs; frozen scorer verified in all eight legs' on-disk configs; `SAE_3E1.md` approach
  36, verdicts 68-69). The result is the registered "frozen final-leg win" case: periodic leg 8
  reads 18.82/24.56 against frozen 17.61/22.66, so scorer refresh has NO established durable
  benefit at this operating point; the periodic lead at legs 2-4 (best 0.55/0.55) is the
  registered transient that licenses nothing. Standing fact for any successor arm (verdict 69):
  both arms degrade after leg 3 and neither leg-8 endpoint beats the no-loop init 13.89/18.34 —
  the recency question was answered inside a regime where the loop itself loses ground, which no
  frozen-versus-periodic contrast can address.

**D6-PERIODIC/GAN960-FROZEN — frozen-scorer loop from the theta_0^G960 init (USER-directed
2026-08-21).**
- Purpose: the A5 scale arm produced a better label-free start (theta_0^G960, 13.11/16.82 against
  theta_0^G's 13.89/18.34, `SAE_3D_GTRACK.md` A5), and the user funds one reconstruction loop on
  it with a frozen scorer only — no periodic refresh, no refit machinery. Two questions, one arm:
  does the GRPO loop improve on the stronger init at all (both prior loop trajectories peaked by
  leg 2 and then degraded), and how much of the loop's behavior was owed to the weaker init
  (paired against GAN-FROZEN, which differs from this arm only in its starting checkpoint).
- Approach: the D6-PERIODIC/GAN-FROZEN recipe verbatim — the segmented eight-leg policy graph,
  the same 960 h round-robin shard at each leg, shaped reward and T=0.7, cosine epoch offsets,
  batching, checkpoint imports and fresh Adam state per leg, and the same frozen scorer
  `PsiAlignTrainJob.dsMKgPHQApyR` at every leg — with exactly one experimental change: the
  starting checkpoint is theta_0^G960 (`ReturnnTrainingJob.HuSkdbuVRg6d`
  `output/models/epoch.010.pt`). Any reward or monitor anchor defined relative to the arm's own
  init (a KL snapshot, if wired on this bed) follows the init; the implementer flags any knob
  where recipe reuse and the init swap conflict rather than resolving it silently. Disclosed
  asymmetry, accepted: the frozen scorer was refit on round-1 decodes of the theta_0^G
  trajectory, so it is native to the OLD init's error distribution; a fresh refit on
  theta_0^G960's own decodes would confound the init comparison with a scorer change and is NOT
  funded — it is a possible follow-up decision, not part of this arm.
- Experiments: run the same leg schedule as GAN-FROZEN with per-sub-epoch recogs; report plain
  dev-clean/dev-other WER and S/D/I per leg, the full trajectory, and the paired per-leg deltas
  against GAN-FROZEN at matched legs. No scorer dumps, pools or refits exist anywhere in this
  graph.
- Gate (pre-registered before any leg result exists): the loop is useful on this init only if the
  fixed leg-8 endpoint beats the arm's own init 13.11/16.82 on BOTH dev splits; intermediate
  matched-leg deltas may establish transient effects but select no endpoint and license nothing.
  The paired init read is leg 8 vs GAN-FROZEN leg 8, reported on both splits. Per-sub-epoch
  monitors (filler mass, per-term within-group std, insertion histogram) carry kill authority
  between legs per the standing residual-risk rule; a degrading run is reverted, not compounded;
  no dev-selected checkpoint feeds anything downstream.
- Status: REGISTERED AND AUTHORIZED 2026-08-21 (the user's launch word, given on the verified A5
  read); not yet implemented. Implementation must not displace any running funded job.
  2026-08-21 later: IMPLEMENTED AND VERIFIED, awaiting the user's manager start
  (`./sis_managers.sh start sae_3e1_d6periodic_gan960_frozen`; classifier-blocked for both
  sessions). Planner verification basis: the config diff against the sibling
  (`config_sae_3e1_d6periodic_gan960_frozen_v1.py` vs `..._gan_frozen_v1.py`) read directly —
  the only non-cosmetic change is the leg-1 init `ReturnnTrainingJob.HuSkdbuVRg6d`
  `output/models/epoch.010.pt` plus the arm's own alias namespace, so no refit path exists
  structurally. Implementer graph census (accepted on that basis): leg-1 training is the new
  `ReturnnTrainingJob.ohmLWWmr6Kxe` (sibling `kr1foUV6lecx`, no collision); the 64 jobs a launch
  funds contain zero psi_align/curate/scorer_diag work; frozen `dsMKgPHQApyR` at every leg.
  Two recorded resolutions of flagged points: (i) the "anchors follow the init" instruction above
  is VACUOUS on this bed — `loop_config` takes only `psi_checkpoint` and `av_checkpoint`, no KL
  snapshot or init-relative anchor exists, nothing to re-point; (ii) leg 1 inherits the sibling's
  `round1_artifacts()` bookkeeping, so its record's dump/pool/refit fields cite the sibling's
  finished round-1 jobs. Ruling: keep the recipe verbatim (no compute is funded by those
  references) and read them as PROVENANCE OF THE SHARED FROZEN SCORER, built from theta_0^G
  decodes — this arm refreshed NOTHING at leg 1 or anywhere; any audit of this arm's round 1 must
  use this line, not the record fields.

**D6-PERIODIC/GAN+HOM — homophone-diversity SFT arm on the same bed (USER-directed
2026-08-17).**
- Purpose (the user's mechanism, planner-formalized): make the AV's conditional diverse
  over homophone spellings at SFT so sampling exposes the variants and the reward's
  contextual term — the LM prior — can steer spelling; a committed policy has zero
  within-group spelling variance and GRPO cannot steer what sampling never exposes. On
  THIS bed the diversity also propagates into the scorer: per-boundary refits train psi
  on the policy's own decodes, so spelling variants over identical acoustics equalize
  psi's spelling-specific emissions round over round — a mechanism no frozen-scorer arm
  has.
- GROUND-TRUTH CORRECTION THE ARM IS BUILT ON (2026-08-17 planner audit, at source): the
  live psi conditions on GRAPHEMIC BPE — there is NO G2P anywhere in the reward (PLAN.md
  §3e formula corrected same day). Homophone spellings are therefore NOT
  reward-invariant: the reconstruction term carries a structural per-state price on
  orthographic length (the minimal-state exploit's substrate) plus learned
  spelling-specific emissions. The arm's claim is "diversity lets the LM prior plus
  refresh equalization pick spellings", never "spelling is reward-free"; HOM-0b below
  measures the actual spelling sensitivity before any SFT is funded.
- Class construction (REPINNED 2026-08-17, replaces the first-pronunciation /
  vocabulary-membership form — because HOM-0a measured both original rules broken: the
  LM-vocabulary restriction is vacuous at 973,673 types (it admits "tha"/"thaa"/"waz"
  as spellings of the most frequent words — the top class by mass was the/tha/thaa at
  5.8 % of tokens), and the first-pronunciation map picks REDUCED forms for function
  words, e.g. TO -> T AH while TOO/TWO -> T UW, splitting the intended reference
  class): (i) two words are homophones iff their FULL pronunciation SETS in the
  lexicon are equal — strict on purpose: it kills reduced-form splits' false joins
  (firm/from share only a first pronunciation) at the cost of splitting words that
  carry an extra variant pronunciation, and the conservative direction is the safe
  one; a consequence accepted as a feature: "to" (two pronunciations) does not join
  too/two, so the filler word is never resampled and the augmentation cannot touch
  the filler pathology. (ii) every member must have LM-corpus frequency >= 1e-5 of
  tokens (~8,000 occurrences) — the smallest floor in HOM-0a's sweep that removes the
  typo members (at 1e-6, "tha" at 1,333 occurrences still survives); chosen on that
  junk criterion, explicitly NOT on the funding verdict it induces. (iii) no
  single-character members ("i", "u", "r", "b" are frequent as spelled-out letters
  but register-wrong as substitutes). One-to-one word substitutions only, lowercase,
  classes of >= 2 surviving members; the FINAL class list is dumped and read by eye
  before any SFT funding (a frequency floor provably does not clean everything). The
  reduced-first-pronunciation property of the probe map phi is recorded as a caveat
  for probe interpretation (phi("to") is the reduced form).
- Augmentation (pinned semantics; job mechanics the implementer's): per epoch, per
  occurrence of a class word, resample the spelling from
  (1-p) * keep-original + p * Uniform(full class); p = 0.3 planner constant. Rng seeded
  on (seed, epoch, utt-id, position) — deterministic and inspectable; per-epoch
  substitution counts dumped. Applied ONLY to the init SFT targets (the §1d pseudo-label
  corpus -> theta_0^G_hom); everything downstream is D6-PERIODIC/GAN verbatim — loop,
  refresh recipe, reward, schedule. Quarantine: the pronunciation lexicon is allowed
  prior knowledge, the targets are pseudo-labels; no transcript is touched.
- ADMISSION READS, before the SFT is funded (label-free, on existing artifacts, CPU +
  <= 1 GPU-h): (HOM-0a) class statistics on the §1d pseudo-text corpus — token share in
  multi-member classes (funding floor: >= 5 % of tokens; below it the ceiling is too
  small and the arm is reported unfunded, not absorbed), class size/frequency table,
  the to/too/two share. (HOM-0b) psi spelling sensitivity on the round-1 dump: swap
  spellings within class in homophone-bearing sampled texts and re-score BOTH reward
  terms under the round-1 refit, deltas measured in the arm's own reward units
  (post-weight, post-normalization); admission bar: median |delta lm_prior| >
  median |delta recon| — the contextual term must be the dominant spelling signal. If
  psi's spelling bias dominates, the arm is NOT funded as designed and the finding goes
  to the user as its own result (a spelling-sensitive scorer is a distinct defect,
  related to the minimal-state family). Report the sign structure too (does psi favor
  short spellings). (HOM-0c) spelling coverage at T=0.7 on the init's own full-bed G=12 dump (replaces "the same dump", 2026-08-18: the round-1 dump samples one candidate per utterance -- DUMP_GROUP_SIZE=1 by the periodic arm's design -- so a group of one cannot hold two spellings and the registered artifact cannot answer the registered question by construction; init, corpus and temperature held fixed, and G=12 is the loop's own group structure): fraction of
  homophone-bearing groups already holding >= 2 spellings of one class — if already
  high, SFT augmentation solves a non-problem and only the reward-side question
  remains (reported, arm re-scoped).
- Arm read (pre-registered, if admitted; planner constants, user may override):
  one-argument A/B vs D6-PERIODIC/GAN at matched legs. PRIMARY — dev-other (plain WER
  as scored) within 0.3 abs of the plain arm at every matched leg AND better at the
  final leg. MECHANISM — within-class spelling entropy of group samples above the plain
  arm's at matched legs (label-free); class-internal substitution counts in the sclite
  reports lower at the final leg (gold read, reports only, never selects). GUARD — the
  audio-free-null discipline: the prior's share of within-group reward variance stays a
  minority (< 0.5) at matched legs; the augmentation deliberately enlarges the LM
  term's role, and an arm whose reward became mostly audio-free is inadmissible
  regardless of WER. KILL (label-free): class-aggregated insertion counts above the
  plain arm's at two consecutive legs.
- MONITOR AGGREGATION RULE (pinned): on this arm, every filler / minimal-state /
  suspect monitor aggregates counts BY HOMOPHONE CLASS (too+two is one count; "to"
  forms no class under the corrected map and stays under the standing
  minimal-state watch) —
  spelling resampling would otherwise mechanically dilute any per-spelling watch and
  read as a fake improvement.
- Cost: one SFT (theta_0^G_hom) + one 8-leg loop arm with refits; admission reads are
  CPU + <= 1 GPU-h.
- Status: REGISTERED 2026-08-17. Next step: HOM-0a/0b/0c (implementer, cheap, on the
  round-1 dump and the §1d corpus); the SFT is funded only after 0a and 0b clear their
  floors.
  2026-08-17 (later; HOM-0a first pass, `HomophoneClassStatsJob.HrAzOCnfGuLw`): the
  raw 0.7189 token share is NOT read as a pass — the implementer showed the
  instrument broken (vacuous vocabulary restriction; reduced-first-pronunciation
  split) and correctly declined to report a verdict. Instrument repaired by the
  REPINNED class-construction rules above, BEFORE any verdict is read; the 5 %
  funding floor itself is UNTOUCHED and the verdict comes from a rerun on the final
  class set (sensitivity anchors from the broken sweep: 0.1277 at the 1e-5 member
  floor alone, 0.0476 at 1e-4 — the pinned rules land near the former minus the
  set-equality and single-character removals, so the gate is genuinely live, not
  rigged). DRAW RULE DECISION, same date: the implementer's frequency-floor
  recommendation is ADOPTED; the frequency-PROPORTIONAL draw recommendation is
  REJECTED and uniform stands — a proportional draw puts near-zero mass on the
  minority spelling of a skewed class (in/inn: ~0.06 % of resamples), producing
  near-zero within-group variance exactly where steering is needed, i.e. it
  recreates the commitment the arm exists to break; junk exclusion is the class
  hygiene's job, not the sampler's, and the user's standing view (added entropy is
  good for GRPO where the ranking signal is sound) backs the uniform form. HOM-0b
  runs on the FINAL pinned class set only (implementer-flagged, agreed). Rerun is a
  one-line config change; the job gains a member-floor argument and re-hashes.
  2026-08-17 (rerun on the pinned rule; `HomophoneClassStatsJob.our76yheSD0c`):
  HOM-0a VERDICT: PASS — 7.68 % of pseudo-corpus tokens sit in a multi-member class,
  against the untouched pre-registered 5 % floor (the share a uniform draw actually
  rewrites, 4.02 %, is reported beside the gated share, never in its place). 142
  classes / 292 members (131 pairs, 8 triples); the discriminating cases behave as
  pinned ("to" joins nothing, "from" does not join "firm"). EYEBALL STEP (planner,
  full 142-class list read): PASS, no strikes — every member is a real word above the
  floor (weakest: "ad" 8,084 vs the 8,033 cutoff), no typos survived. Named
  observations, no rule change: a spelling-convention subfamily (~16 classes,
  honor/honour, recognise/recognize, gray/grey...) where context never disambiguates
  — small mass, kept, watched per class; day/dey register-odd but rule-clean; 3
  classes (ann/anne, lo/low, oh/owe) never fire because NO member appears in the
  corpus — consistent with the known missing-word pathology (LOW/OH), and out of
  HOM's reach by construction (it resamples occurrences; it cannot create absent
  words). REPAIR-CHANNEL FINDING (implementer-raised, planner-confirmed from the
  class list): for several classes the corpus systematically uses only a WRONG or
  minority member while a dominant-LM-mass member has ZERO corpus occurrences — by
  (0 vs buy 2,984), sea (0 vs see 1,364), right (0 vs write 292), side (0 vs sighed
  279), air (0, vs ere/heir), fair (0 vs fare 56), they're (0) — roughly 0.5-0.6 %
  of corpus tokens, each a guaranteed substitution error wherever the reference
  wants the absent spelling. Consequence: HOM is partly a REPAIR channel (uniform
  draw introduces the correct spelling at 1/k of class occurrences), not only a
  diversity channel — and this retro-confirms the proportional-draw rejection, since
  a corpus-frequency-proportional draw would introduce these members with
  probability exactly zero. Apostrophe check done: the corpus produces apostrophe
  spellings elsewhere (it's 309, i'll 247, there's 285), so they're=0 is a decoder
  commitment, not an alphabet artifact. HOM-0b/0c READING AMENDED PRE-RUN (bars
  untouched; 0b has not run — blocked on the round-1 dump): 0b's admission bar stays
  the registered aggregate median comparison, and the report additionally splits
  every swap into (i) swaps INTO a corpus-zero member (repair direction — a spelling
  psi's refit corpus never contained, so recon's penalty there is an extrapolation)
  vs (ii) swaps between corpus-attested members (diversity direction), plus
  per-class medians for the top-8 classes (60.5 % of rewrites). If the bar passes on
  only one of the two swap types, the arm read must name which channel carries it.
  0c unchanged, annotated by the same split. FUNDING STATE: 0a and the eyeball are
  cleared; SFT training of theta_0^G_hom stays gated on 0b as registered;
  augmentation machinery may be built meanwhile (cheap, reversible). Log: SAE_3E1.md
  approach 23 / conclusion 40; superseded first-pass numbers (HrAzOCnfGuLw) are not
  to be quoted.
  2026-08-17 (pre-run, augmentation artifact built and measured; bars untouched): the
  corpus the HOM arm would be SFT'd on exists (HomophoneAugmentJob.k2OwZiTcKpEG,
  commit 3024058; planner-verified by an independent token diff against the source
  corpus — 38923 of 963857 tokens rewritten = 4.04%, repair 5627 / diversity 33296,
  zero out-of-class or case violations, dev splits byte-identical, and the realized
  draw sits within sampling noise of the analytic expectation from the ratified
  classes.json, so the sampler IS the draw 0a's arithmetic assumed; class list read
  from the ratified artifact, stats-job hash unchanged, the SFT deliberately not
  wired). Measured concentration, implementer-flagged: in/inn carries 7386 rewrites =
  19% of all rewrites, because a uniform draw flips about half of a high-frequency
  function word's occurrences to a rare member context never disambiguates. This was
  priced when the proportional draw was rejected and the draw stands; the consequence
  is registered for the reads: the 0b/0c report names in/inn's own per-class median
  explicitly and reports the aggregate medians WITHOUT in/inn beside — never in place
  of — the gated aggregate, so a pass carried largely by one class is visible on the
  page; day/dey (557 rewrites, rank 14, register-odd from the eyeball) joins the
  named per-class watch. Admission bars unchanged. The augmentation is machinery, not
  an experiment: no new Approach item — the Catalog row and the verifier bullet are
  its record, and its numbers enter the log with 0b.
  2026-08-18 (0c RUN AND VERIFIED; 0b pending): HomophoneCoverageJob.F76iJ8j0AQi1 on
  ReturnnForwardJobV2.lQMOR5n2ntcS (theta_0^G full bed, 28,539 groups all G=12, T=0.7),
  planner-verified against the artifact: 26,584 groups homophone-bearing; 6,228 = 23.4 %
  of those already hold two spellings of one class; only 217 = 0.8 % ever contain a
  spelling absent from the scorer's refit corpus. SCOPING READING (planner): the
  DIVERSITY direction is already proposed by sampling in about a quarter of
  homophone-bearing groups -- within-group variance exists, so the reward can already
  steer on it and diversity-type augmentation feeds a channel that is not starved; the
  REPAIR direction is essentially never proposed (0.8 %), i.e. it sits in the dead band
  -- no reward-side weight can steer toward a spelling the policy never samples -- so a
  support change at the init (the SFT) is the ONLY lever that reaches the repair
  channel. If 0b admits the arm, the mechanism read weights the repair-type split
  accordingly. Funding fork unchanged: the SFT stays gated on 0b as registered.
  2026-08-18 (0b RUN AND VERIFIED -- GATE VERDICT): HomophoneSensitivityJob.xB5RvcgLVgtD,
  planner-verified from the artifact (scorer PsiAlignTrainJob.dsMKgPHQApyR at epoch 30;
  23,085 single-word swaps; the recomputed prior reproduces the dump's own column to
  median |delta| 0.0053 nats/token after the terminator fix, commit 9fa9ecc). THE
  PRE-REGISTERED AGGREGATE BAR PASSES: median |delta lm_prior| 0.0134 vs median
  |delta recon| 0.0106 per unit frame at lam_lm 1.0, ratio 1.264 -- the arm is ADMITTED
  and SFT training of theta_0^G_hom is licensed per the registered funding rule; the
  build proceeds. MECHANISM READ (the registered repair/diversity split, now measured):
  the split REVERSES -- diversity swaps (n=22,584) pass at 1.274, repair swaps (n=501)
  FAIL at 0.816, and both reward terms penalize repair swaps on signed median (recon
  -0.0179, lm_prior -0.015) -- so read with 0c, the arm's unique lever (repair: sampled
  in 0.8 % of groups, only reachable by SFT) is exactly where the reward's contextual
  term is NOT dominant and the net reward resists the swap, while the direction the
  reward handles well (diversity) is the one sampling already supplies. Sign question
  answered in the negative: shorter spellings are penalized MORE (-0.0135 vs -0.0073),
  the opposite of the watched-for orthographic-length price. Per-class structure is
  real, not uniform (knot/not 3.70 and too/two 3.88 against wood/would 0.49 and
  their/there/they're 0.54). ORDERED (standing dead-band rule, before any steerability
  claim): std_within_group of the shaped reward and its lm term on this bed's own dump
  -- implementer wires it; it runs in parallel and blocks no build. SURFACED TO THE
  USER: the gate the user pre-authorized has passed and the arm proceeds per its
  registration, but the mechanism reversal is on the record -- if the repair channel
  was the point of the arm, stopping the loop A/B before launch is the user's open
  option.
  2026-08-18 (later -- SFT LAUNCH HELD, decision moved to the user): hours after the 0b
  license, the user ruled (periodic-gate removal, quoted in the D6-PERIODIC section)
  that quality must not be decided by constructed scorer-side metrics that do not
  necessarily reflect performance. The implementer declined to build theta_0^G_hom on
  the planner's license alone -- the 0b admission bar is exactly such a scorer-side
  statistic -- and put the choice to the user with both recommendations attached. The
  PLANNER ENDORSES the hold, same day: the funding rule pre-dates that ruling, the
  mechanism read (repair swaps net-penalized by both terms) independently argues for a
  user look, and an SFT plus an eight-leg loop is the kind of spend the ruling
  reserves. The std_within_group read proceeds regardless (ordered above); if the user
  says go, the registered arm runs unchanged.
  2026-08-18 (later still — HOLD RELEASED): the user greenlit the arm in the
  implementer's session ("I greenlight Homophone-diversity SFT arm in 3e1", relayed by
  the implementer); theta_0^G_hom SFT LAUNCHED and planner-verified the same day
  (ReturnnTrainingJob.EabxlDlT0oji — single-moving-argument construction off
  theta0g_av_sft, corpus = the ratified augment artifact byte-checked in full at zero
  mismatches, theta_0^G hash unmoved; detail in SAE_3E1.md fourth-round bullet). The
  registered arm proceeds unchanged: the loop A/B vs D6-PERIODIC/GAN at matched legs
  once the SFT finishes; the std_within_group read stays ordered; the 0b
  mechanism-reversal record stands unweakened.
  2026-08-18 (INIT READ IN, planner verdict and funding recommendation; the pre-registered
  arm read above is UNTOUCHED and remains UNREAD -- no leg has run, so nothing here is a gate
  verdict). theta_0^G_hom is 16.67 / 21.45 against theta_0^G's 13.89 / 18.34, +3.11 dev-other
  on plain WER as scored, verified first-hand to the concrete sclite dirs with a four-line
  config diff between the arms and no path by which dev WER could have selected anything
  (SAE_3E1.md approach 27, fifth-round verifier bullets). The damage is the augmentation
  reproducing itself, not a broken model: 96.7 % of the extra errors are within-class
  substitutions and everything outside the classes sits inside a bootstrap CI straddling zero.
  RECOMMENDATION TO THE USER: do NOT fund the eight legs. Three grounds, the first two in the
  evidence class the 2026-08-18 ruling endorses (plain recognition performance, not constructed
  scorer statistics). (1) ARITHMETIC OF THE REGISTERED PRIMARY: it requires dev-other within
  0.3 abs of D6-PERIODIC/GAN at every matched leg; that arm's own leg 1 went 18.34 -> 19.69, so
  the hom arm must land at or below 19.99 from 21.45 -- a 1.46 absolute gain in ONE leg, from a
  deficit, where no leg on this bed has produced anything of the kind. (2) THE ONLY ROUTE BACK
  IS SPELLING: since the deficit is 96.7 % class-internal, the arm recovers only if the loop
  re-picks spellings; nothing else is broken to repair. (3) THE LOOP CANNOT RE-PICK THEM AT THE
  REGISTERED OPERATING POINT: on the swaps that would repair a wrong spelling, the LM prior
  points at the reference about nine times in ten while the reconstruction term points at it
  fewer than two times in ten, and the arm's own composed reward at lam_lm 1.0 lands at a coin
  flip (planner join on the 0b dump, artifacts already on disk; MUST be re-emitted as a job
  before any of those numbers is cited -- ordered below). The measurement also closes the
  obvious rescue: raising lam_lm until the prior dominates is exactly what this arm's own GUARD
  clause forbids (prior share of within-group reward variance must stay below 0.5), so the one
  lever that would make the mechanism work is inadmissible by the registration it would be
  serving. WHAT THE ARM ALREADY BOUGHT, bankable either way: a clean single-argument
  demonstration that an SFT can install spelling diversity at will -- 40.96 % of class-bearing
  dev-other reference tokens now carry a within-class substitution against the plain arm's
  6.57 % -- and that the composed reward cannot exploit it. That is a reward-side finding about
  the orthographic channel, not a failure of the SFT. ORDERED BUILDS, both CPU-cheap and
  neither blocking anything: (a) the correctness-direction read as a real job over the existing
  0b dump (position-aligned if feasible, bag-of-words disclosed if not; gold read, reports
  only, never selects) -- until it exists the finding stays in the verifier section and out of
  every conclusion; (b) the still-unwired std_within_group read, whose lm-term half needs
  either the hardcoded scorer_diag tuple widened or the per-token/per-unit conversion applied
  (SAE_3E1.md fifth-round bullet). IF THE USER FUNDS THE LEGS ANYWAY the arm runs exactly as
  registered, with one reporting amendment: leg 1 reports the class-internal substitution count
  beside plain WER, so the mechanism claim is falsifiable at the first leg rather than the
  eighth. A recommendation not to fund licenses "not funding this", never "spelling diversity
  could not work" -- at a lam_lm the guard permitted, or under a scorer without the
  orthographic price, the same mechanism is untested.
  2026-08-18 (USER OVERRIDE, relayed by the implementer -- "I mean this is just sft errors
  right? Maybe it's better in loop"; arm LAUNCHED, commit 282eeb3, leg 1
  ReturnnTrainingJob.JocWKAmYroFJ). The planner's not-fund recommendation above is
  SUPERSEDED as a funding decision and kept as dated history; the user's hypothesis -- that
  the deficit is fine-tuning error the loop repairs -- is now the arm's question. My verdict
  is not retracted as evidence and its instrument stands: the composed reward pointed at the
  reference on about half of the repairing swaps.
  PRIMARY READ, AMENDED AND REGISTERED BEFORE LEG 1 EXISTS (replaces the "within 0.3 abs at
  every matched leg AND better at the final leg" clause, which is unsatisfiable by
  construction once the init starts 3.11 behind -- a clause that can only be failed cannot
  decide anything. The MECHANISM, GUARD and KILL clauses are UNTOUCHED and read as written).
  All numbers are plain dev-other WER as scored, at matched legs against D6-PERIODIC/GAN.
  Two clauses, both required to be reported, each with its own verdict:
  (a) RECOVERY -- the arm's own init deficit is 3.11. Bar: gap(hom minus plain) <= 1.55 at
      leg 4 AND <= 0.30 at leg 8, where BOTH readings additionally require the hom arm to sit
      strictly below its OWN init 21.45 at that leg. The second half is not decoration: the
      gap can close because the plain arm degrades, and a gap that closes without the hom arm
      improving is not recovery. 1.55 is half the measured deficit at the schedule midpoint --
      derived from the deficit, not chosen -- and 0.30 is the ORIGINAL registered margin,
      preserved and relocated to the one leg where an init handicap makes it readable.
  (b) SUPERIORITY -- hom dev-other strictly below the plain arm's at leg 8. This is the
      arm's original ambition and the only clause that can make the augmentation worth its
      cost.
  VERDICTS, pre-registered: (a)+(b) = the arm wins and homophone augmentation is funded
  onward. (a) only = the user's hypothesis is CONFIRMED (the loop does repair fine-tuning
  spelling damage) while the augmentation buys nothing over the plain init -- report as
  recovery-only, do not fund further. Neither = the deficit is not loop-repairable at this
  operating point. A THIRD outcome is registered here because the measurement above predicts
  it: gap shrinking monotonically but missing the leg-4 bar = PARTIAL recovery at a rate
  consistent with a weak per-swap reward edge; that licenses not funding an extension and
  does NOT license "the loop cannot repair spelling".
  WHY A RATE BAR AND NOT A DIRECTION BAR: a 53 % per-swap edge is not a prediction of zero
  recovery. GRPO aggregates over groups and steps, so even a small systematic edge accumulates
  -- slowly. The informative quantity is therefore the RATE at which the gap closes, which is
  what (a)'s midpoint bar measures and what separates a 90 %-edge mechanism (the prior acting
  alone) from a 53 %-edge one (the composed reward as registered). This arm is consequently a
  direct test of my own finding, and both outcomes are informative.
  REPORTING AMENDMENT (adds visibility, changes NO verdict condition): the class-internal
  substitution count on dev-other is reported at EVERY leg beside plain WER, not only at the
  final leg where the MECHANISM clause reads it. It starts at 1,827 against the plain init's
  293 and is the mechanism's most direct instrument, so it makes the arm falsifiable at leg 1
  rather than leg 8. Pre-registered interpretation, an indicator and NOT a kill: a leg-1 count
  that has not moved off 1,827 means the mechanism is not engaging; the plain arm's own leg 1
  went 18.34 -> 19.69, so one leg is too noisy to stop an arm the user directed.
  IMPLEMENTER CORRECTION ACCEPTED, AND ITS LIMIT: the correction direction (inn -> in,
  knot -> not, bee -> be) is a swap between corpus-attested spellings, i.e. HOM-0b's DIVERSITY
  class which passed at 1.274, not the REPAIR class which failed at 0.816 -- correct, and my
  "mechanism reversal" framing did not describe this direction. It does not, however, move the
  composed-reward finding: that was measured INSIDE the diversity subset (n=1,523 of the 1,550
  repairing swaps), where the prior points at the reference 90.7 % of the time and the composed
  reward at 53.7 %. The unresolved quantity is the reconstruction term's opposition, not the
  0b class.
  STD_WITHIN_GROUP: the ordered dead-band read moves to THIS arm's round-1 dump -- approved and
  now the better home, since it is the first policy on this bed carrying real within-class
  spelling variance. The lm-term half still needs the per-token to per-unit conversion or the
  widened scorer_diag tuple (SAE_3E1.md fifth-round bullet); it reports and gates nothing.
  2026-08-18 (WEIGHTING DEFECT in the direction read, planner-found; demotes BOTH earlier
  readings of it). The direction read is built from swaps on theta_0^G's samples, so its class
  mix is the PLAIN policy's error profile: buy/by/bye is 67.6 % of the measurement because
  190 of that policy's 293 class-internal substitutions are by->buy -- one fact seen twice.
  The HOM arm's damage has a different shape: buy/by/bye is about 157 of its 1,827
  class-internal substitutions (~8.6 %), while in->inn alone is 329 (18 %) and the mass sits
  in classes the measurement barely observes PRECISELY BECAUSE the plain policy spells them
  correctly. CONSEQUENCE: neither the aggregate 0.529 (my not-fund grounds) nor the 0.702
  everything-else slice (the implementer's counter) is the damage-weighted quantity; both are
  weighted by the wrong policy and neither answers the funding question. The corrected
  instrument is the same read on the HOM arm's own round-1 dump, where toward-reference swaps
  carry the distribution the loop must actually repair.
  AMENDED SAME DAY, after verification against the finished artifact (replaces "no number from
  the plain-policy read may be quoted as a prediction of this arm's recovery", which was too
  strong -- I had not considered the away cell). Quantified: the measurement's class mix
  overlaps the plain arm's error profile at total variation 0.880 and the hom arm's at 0.197;
  78.8 % of the hom damage sits outside the eight reported classes and 31.1 % in classes with
  ZERO toward-reference swaps (be/bee and knot/not, 155 each); in/inn, 18 % of the damage, is
  measured on four swaps. Damage-weighted, the toward cell gives 0.555-0.637 on thin data. BUT
  the AWAY cell (n=19,328) covers 99.7 % of the hom damage mass and sign-reverses to 0.900 --
  a different conditional over a different utterance population, and arguably the LESS biased
  estimator here, since the hom arm's errors are augmentation-induced and land on a near-random
  subset of class-bearing utterances while the toward cell is selected for utterances the plain
  policy itself found confusing. STANDING RULE, replacing the blanket ban: the plain dump
  BRACKETS this arm's per-swap edge at roughly 0.51 to 0.90 and any quotation of it must carry
  that bracket and its two named biases; no single point from it may stand as the prediction.
  A bracket spanning chance to strong is uninformative for funding, which strengthens rather
  than weakens the case for the read on the arm's own dump.
  2026-08-18 (EQUALIZATION vs ENTRENCHMENT -- pre-registered BEFORE the arm's own refit
  exists; implementer-raised, planner-ratified, and it puts a SIGN on this arm's registered
  mechanism that nobody has checked). The arm's Purpose claims per-boundary refits "equalize
  psi's spelling-specific emissions round over round". The refit corpus is the policy's OWN
  decodes, so a policy over-producing "inn" yields a scorer trained on "inn" -- which would
  make the reconstruction term oppose inn->in MORE, not less. Equalization and entrenchment
  are the same mechanism with opposite signs. PRE-REGISTERED READ, on the direction job run
  under the arm's own refit (ACP3LqKDUSQ0) against the same job under dsMKgPHQApyR: recon's
  toward-reference rate FALLING = ENTRENCHMENT, and the registered mechanism claim has the
  wrong sign; RISING = equalization as registered. This is why the read waits for the arm's
  own refit rather than using the scorer that exists -- the reconstruction term is the one
  doing the cancelling and the only scorer-dependent one, so the available scorer would answer
  about a scorer that will never grade leg 1, on exactly the axis under test.
  MANDATORY CONTROL, registered with it: the comparison must be WITHIN-CLASS paired, because
  the toward and away populations differ in class composition -- the same weighting defect as
  above, and it is the reason the tempting inference from the existing table (recon favours
  whatever the policy sampled at 0.821 toward / 0.746 away, i.e. agrees with the policy rather
  than the truth) is NOT yet supportable. Register it as a hypothesis with that control, never
  as a finding.
  ONE STRUCTURAL FACT that needs no control and bounds every reading here: class members are
  defined by EQUAL FULL PRONUNCIATION SETS, so homophone spellings are acoustically
  indistinguishable by construction. Any reconstruction-term preference between them is
  therefore a text-side artifact -- the per-state orthographic length price plus learned
  spelling-specific emissions -- and can never be acoustic evidence about which spelling is
  right. A reward whose spelling decision rests on that term is deciding on an artifact.
  SCOPE, and why this read is worth having whatever the funding decision: an entrenchment
  result is NOT specific to the homophone arm. It would be a quantified instance of the
  standing G-track diagnosis that a scorer refit on the policy's own output rewards correlated
  errors rather than catching them, and it would bear on every arm in the D6-PERIODIC family,
  which all refit on their own decodes. Registered as a reported diagnostic; it gates nothing
  and selects nothing, and its gold reads report only.
  2026-08-18 (BOTH READS IN AND PLANNER-VERIFIED against the artifacts;
  HomophoneDirectionJob.deNc7xXnCfSu and HomophoneScorerDeltaJob.JKbbRWimojlI, both on the
  arm's own round-1 dump with its own refit ACP3LqKDUSQ0). Every number reproduces bit-for-bit
  under independent reimplementation, provenance traces to the hom SFT not the plain one, and
  the weighting defect IS repaired: overlap with the hom damage profile rises 0.197 -> 0.891,
  damage mass measured at n>=20 rises 11.3 % -> 95.1 %, zero-coverage mass falls 31.1 % ->
  0.1 %, and the damage-weighted composed rate (0.822) now matches the aggregate (0.825).
  (1) SIGN TEST: ENTRENCHMENT CONFIRMED and robust. The registered within-class paired control
      is satisfied by construction (one shared key list scored under both scorers); the two
      swap files differ in exactly three fields with the language-model column byte-identical,
      so it is single-variable at the data level. It also survives two confounds I had not
      registered: per-utterance memorization (held-out utterances -0.351 vs -0.326 on training
      ones) and foreign-scorer edit preference (over all 23,085 swaps the scorers endorse at
      0.464 vs 0.479, and the delta REVERSES to +0.250 on away-swaps; difference-in-differences
      -0.577). The registration's equalization claim has the wrong sign on its own arm.
  (2) THE HEADLINE 0.825 IS THE LANGUAGE MODEL, NOT THE SCORER, and may never be relayed
      without the audio-free null beside it -- this is exactly the case that principle exists
      for. Pooled over 19,765 directional swaps: lm_prior alone 0.9702, composed 0.8444, recon
      alone 0.4038 at z = -27. Adding the audio-grounded term COSTS 12.6 points of reference
      accuracy. The lm term is Qwen3-1.7B-Base -- free English carrying no audio.
  (3) RECON'S APPARENT DIRECTION IS A CHARACTER-LENGTH PRICE. Split by length change its
      endorsement is near-identical across the two cells (longer 0.704/0.697, same 0.469/0.429,
      shorter 0.236/0.230); the 0.357-vs-0.559 split is an artifact of toward-swaps being
      mostly shorter and away-swaps mostly longer. State-matched at equal character count recon
      reads 0.5279 -- detectable, practically chance. This CONFIRMS this section's own
      ground-truth correction (a structural per-state orthographic length price, the
      minimal-state substrate) and shows the learned spelling-emissions half is close to nil.
  (4) OPERATING POINT, binding on how (1) is quoted: -0.327 is the reconstruction term ALONE.
      Under the reward the loop applies (lam_lm 1.0, units-normalized) the same swaps give
      0.825 vs 0.895, delta -0.069. Entrenchment costs ~7 points in the deployed reward, not
      33; the larger figure applies only to a reconstruction-only arm.
  ORDERED, CPU-ONLY, AND IT DECIDES WHAT (1) MEANS: re-slice the sign test at EQUAL CHARACTER
  COUNT -- `d_chars` is already on every swap row, so it is a free re-cut. (3)'s confound
  applies to (1) UNTESTED: the repairs are predominantly shortening (inn->in, knot->not,
  bee->be, know->no), so a refit that merely prices length upward -- which a corpus full of the
  augmentation's longer spellings would produce -- reproduces the entrenchment signature with
  no spelling-specific learning at all. Both readings are entrenchment and both are real, but
  they differ in mechanism and generality, and only the length-matched cut separates them.
  Until it runs the verdict reads "entrenchment confirmed, mechanism unresolved between an
  orthographic-length price and spelling-specific emissions".
  SYNTHESIS, less alarming than (2) sounds. Homophones are acoustically identical by class
  construction, so the reconstruction term CANNOT discriminate them from audio; recon at chance
  is the structurally correct behaviour, not a defect. The division of labour is what the arm
  registered: its stated thesis was always that the LM prior steers spelling, and at 0.970 that
  thesis is CONFIRMED. What is refuted is the secondary claim that per-boundary refits equalize
  the scorer's spelling preferences -- they entrench, and the composed reward stays correct
  despite its scorer half rather than because of it.
  TWO CONSEQUENCES. (a) DISCLOSURE, a Phase-4 item and not a footnote: if this arm recovers
  WER, the repair is attributable to a 1.7B English language model whose pretraining almost
  surely contains the Gutenberg books underlying LibriSpeech -- the exact contamination the
  north star registers as disclosed and controlled. Such a gain reports as supervision cost,
  never as loop progress. (b) The GUARD's literal statistic is NOT this read -- it is the
  prior's share of within-group reward VARIANCE, which only the outstanding std_within_group
  read measures. That read is ELEVATED from housekeeping to this arm's admissibility
  instrument: a reward whose directional content on this axis is 97 % text prior is precisely
  what the guard was written to catch, and nothing currently measures the guard as written.
  REPORTING DEFECTS in the artifacts, none touching a verdict: the class-move count is 77
  negative / 18 positive / 26 tied, not "95 moved"; both jobs truncate at top_k_classes=8, so
  the 121-class tables are planner re-derivations and not citable until a rerun raises it; the
  corpus-zero repair direction is unmeasured (11 of 8,806) and this read says nothing about it;
  the measurement is T=0.7 sampled rollouts on tc100 train while the damage profile is greedy
  dev-other, so the 0.891 overlap spans split AND decoding mode and the read must not be
  relayed as measuring dev-other; the refit saw every base_text (bias runs against the
  reference so it does not inflate the headline, but it is disclosed wherever recon is quoted);
  and the dump consumed psi_g_tc100 for its own reward columns while the swap job used the
  arm's refit, so both scorers must be named.
  2026-08-20 (live loop prefix, planner-verified from concrete work directories; arm not
  ended). Legs 1-3 read 14.84/19.99, 13.94/18.77 and 12.80/18.08 dev-clean/dev-other;
  leg 4 `ReturnnTrainingJob.JBaqJExxDKGz` is running, legs 5-8 remain dependency-unbuilt,
  the manager is live, and no error marker is present. Against its own 16.67/21.45 init this
  is a 3.87/3.37 gain by leg 3. Against plain D6-PERIODIC/GAN the hom-minus-plain deltas are
  +0.39/+0.30, +1.09/+0.88 and -0.40/-0.12: it misses the old every-leg 0.3 condition at leg
  2 but catches and slightly beats plain by leg 3. The class-internal dev-other substitutions
  fall from 1,827 at init to 130, 110 and 105 in legs 1-3 (plain init: 293); total improvement
  is substitution-led. This reverses the practical implication one might draw from the fixed-
  dump scorer-entrenchment diagnostic: the deployed loop rapidly removes the augmentation's
  spelling damage even though the refitted reconstruction scorer alone prefers it. The result
  remains attributable mainly to the Qwen3 language-model prior and is provisional until the
  registered midpoint/final reads exist.

**D7-GAN-SEQDISC — full-960 h online negative sampling (USER-corrected 2026-08-21).**

**Purpose.** Test the reverse audio-negative sequence objective after removing the offline graph
formalism that prevented the first D7 specification from reaching scorer training. This is the
active, corrected D7 method. The executed D7-v2/D7.0b structural failure remains evidence, not an
active specification; it is preserved in `SAE_3E1.md` approaches 30--31 and conclusion 57. D7 asks
whether ordinary dynamic
negative sampling over the actual 960 h loop population improves utterance-specific correspondence
without damaging absolute fit, and whether a scorer with that change helps one matched policy leg.

The operating population is all 281,241 LibriSpeech train-clean-100, train-clean-360 and
train-other-500 utterances in the existing unlabeled 960 h HF/Ogg bed, with the frozen enc50 K=500
raw 50 Hz unit store. Generate exactly one deterministic greedy pseudo-text per utterance from the
fixed theta_0^G checkpoint using the existing G-track argmax decoder; neither a scorer, stochastic
rollout, reference transcript nor the blocked §3d.A packed CTC decoder may choose these texts.
LibriSpeech speaker ID and raw unit length are the only donor-conditioning metadata. Chapter,
silence share, text density, unit-run rate, histogram distance, percentile bands, global donor
capacities and graph regularity are absent.

This follows the ordinary negative-sampling pattern rather than claiming a novel mining result:
CPC makes the contrastive objective tractable through sampled negatives
([van den Oord et al.](https://arxiv.org/abs/1807.03748)), and wav2vec 2.0 samples speech
distractors uniformly during training rather than solving a corpus-wide assignment
([Baevski et al.](https://arxiv.org/abs/2006.11477)). Those methods do not validate this scorer or
its pseudo-text; they justify making the negative sampler simple and testing the loss itself.

**Approach.** For pseudo-text `z_i`, own units `U_i`, raw unit length `T_i`, and dynamically sampled
donor `j`, retain the per-frame score

    s_psi(z_i,U_j) = log p_psi(U_j | z_i) / T_j

and add one binary sampled-softmax term per eligible anchor visit,

    L_online = mean_i softplus(s_psi(z_i,U_j) - s_psi(z_i,U_i)).

Use K=1, temperature 1 in the score's native nats/frame currency, and coefficient 1. The exact
control is `L_NLL + L_U->z`; the candidate is `L_NLL + L_U->z + L_online`. (Definition pinned
2026-08-21 post-implementation-verification, no operative change: `L_NLL` is the per-frame
forward-sum unit NLL and `L_U->z` is the D2 matching-aware in-batch text-negative term
(`matching_contrastive_loss`) at weight 1.0 with 1 negative — exactly the objective and operating
point the D6 round-1 `d_min=2` refit `PsiAlignTrainJob.dsMKgPHQApyR` trained under, whose recipe
constants (batch cells 24M, max batch 256, lr 1e-3, warmup 500, default architecture, CUDA
forward-backward) both arms reuse verbatim. One forced deviation from that refit's 30-epoch
curriculum: the alignment-prior anneal is inactive (prior weight 0 from step 0), because the
contrastive term is defined only at prior weight 0 and the registered control objective carries it
across the whole single pass.) Both use the existing
graphemic-BPE psi_align architecture, `d_min=2`, random initialization, seed, optimizer, batch order
and one complete 960 h corpus pass split into the bed's existing ten 96 h shards (eight of the ten
interleaved shard datasets pre-existed from the D6-PERIODIC campaign and are reused at identical
hash; shards 8-9 are fresh instantiations of the same rule). Fixed final is the
only checkpoint; loss, donor gaps and internal-held metrics never select an epoch. No temperature,
K, coefficient or duration-window sweep is admitted.

Apply the existing deterministic seed-42 5% ID holdout to the full pseudo-pair list and hash both
roles. Training anchors draw donors only from the training role; internal-held anchors draw only
from internal held. For each anchor, its ordinary candidate pool is every other utterance from the
same speaker with `0.8 <= T_j/T_i <= 1.25`. Draw uniformly with a stateless RNG keyed by
`(seed=42, corpus_pass, global_step, anchor_id)`, so reruns reproduce exactly while repeated visits
resample donors. If that window is empty, use the closest-duration other utterance from the same
speaker, with donor ID as the tie break; only a true single-utterance speaker skips `L_online` for
that anchor while retaining the two control losses. Report ordinary-window, nearest-fallback and
singleton rates plus realized length ratios and donor reuse. They are diagnostics, not admission
filters: no row is removed from NLL, no donor is padded, and frequency equalization is left to
sampling in expectation.

**Experiments.** Run in order:

1. **D7.0 full-bed pool and one-step preflight:** generate/hash all 281,241 theta_0^G greedy texts;
   bind the existing unit store; materialize only a speaker/duration index, never an edge table; and
   census the three sampling cases above. On one frozen 96 h shard, verify deterministic resampling,
   exact own/donor path finiteness, control parity when `L_online=0`, candidate-only gradient flow,
   and measured time/RSS for one update. This is an interface/resource check, not a support gate or
   hyperparameter search.
2. **D7.1 full-bed scorer A/B:** train the exact control and candidate for one 960 h pass with the
   same ten-shard order. Persist the fixed-final scorers, role hashes, sampler seed/state contract,
   loss curves, internal-held NLL and online loss, and sampling diagnostics. No policy trains.
3. **D7.2 label-free admission:** at both fixed-final scorers, evaluate internal-held NLL and
   `L_online` with 32 stateless donor draws per eligible held anchor, paired across scorers. Then run
   the existing all-1,500-row Acceptance gate v2 and `PsiScorerParityJob`. The exact D7 full-bed
   control is the paired comparator; the gate's incumbent absolute floors, population and weights
   remain unchanged. Reference text, rollout WER and intermediate checkpoints remain sealed.
4. **D7.3 matched one-leg policy assay:** only if D7.2 passes, request launch authorization. Freeze
   the scorer and substitute it for the exact full-bed control in one otherwise-identical
   theta_0^G, 960 h G-track leg. Submit candidate and control before opening gold; then report paired
   dev-clean/dev-other WER and S/D/I. No periodic refresh or full loop is authorized.

**Gate.** D7.0 fails only for an interface error, non-finite loss/gradient, missing 281,241-row
coverage or resource infeasibility; donor frequency or fallback prevalence cannot recreate D7-style
row filtering. D7.1 reaches admission only at its fixed final endpoint. D7.2 passes only if all hold:

1. candidate minus control mean internal-held `L_online` has a speaker-cluster-bootstrap 95% upper
   bound below zero under the paired 32-draw estimator;
2. candidate internal-held per-frame NLL is no greater than the exact control's point value;
3. every cumulative Acceptance-gate-v2 clause passes on the unchanged 1,500 external rows; and
4. scorer parity passes before any live reward read.

Failure closes D7 without a policy leg; no sampler/temperature rescue is selected from the result.
After an authorized D7.3 submission, its local scientific pass is candidate WER strictly below the
matched control on both dev splits. A label-free scorer win without a two-split policy win establishes
only scorer-side conditional discrimination, not useful ASR progress.

**Status.** IMPLEMENTED AND LAUNCHED 2026-08-21; VERIFIER-CONFIRMED same day (replaces "not
implemented or launched", because the implementer built and launched the graph under the standing
authorization). The user's 2026-08-21 correction
supersedes the offline D7-v2 design, not its recorded result: that graph admitted 56 rows/two
speakers against the frozen 6,778/201 floor, with an independently verified necessary-core upper
bound of 120/four; it stopped before loss, scorer or policy work (phase log conclusion 57).
Ten theta_0^G greedy decode shards (frozen `ReturnnTrainingJob.2fb02hGUdHNj` epoch-10 checkpoint,
the pre-existing G-track argmax decoder reused, GPU `ReturnnForwardJobV2`s) run since 16:37; the
pool, preflight barrier, and both fixed-final train jobs are registered and waiting behind them
(hashes in `SAE_3E1.md` Catalog/State). D7.1 auto-submits only after the D7.0 preflight PASS
artifact exists, per the standing authorization; no funded GPU job was displaced at launch
(verified against the scheduler and every running arm's logs). Verification 2026-08-21 also
confirmed: pseudo-text path replaces the train text with two-sided coverage asserts and no empty
rows; frozen raw-50 Hz K=500 store bound; seed-42 5% holdout reproduces the D7-v2/PsiAlign
row-order convention exactly (toy re-run); every training constant traces to the round-1 refit's
job record; the fixed-final checkpoint format satisfies `PsiScorerParityJob`. Two implementer
actions are requested before the D7.1 jobs first run (both in not-yet-created jobs, so hash-free):
carry the torch CPU/CUDA RNG state through the shard-resume checkpoint so the registered
same-dropout-stream parity survives a wall-clock resume, and count donor-side
structurally-infeasible pairs (donor shorter than the anchor text's minimum feasible frames; the
softplus term saturates to exactly 0 there) in the sampling diagnostics.
D7.3 still requires a fresh launch word.
2026-08-21 later (VERIFIER-CONFIRMED same day: diff read, all five CPU tests reproduced, GH200
parity log verified; replaces "implementer-reported"): both requested fixes landed
(speech-llm commit 1d10945; resume payload carries torch CPU/CUDA RNG state with a
device-change refusal; donor structural-infeasibility counters in sampling/held/preflight
diagnostics; all four D7 hashes plus the merge unmoved). All ten decode shards FINISHED and
merged 18:11; the pool stopped on the source-identity guard as designed and its error marker is
cleared, so the d7 manager restart -- a user `!` line, classifier-blocked for both sessions --
is the only step between here and D7.0/D7.1 running with the fixes in.
2026-08-21 latest (planner ruling, operational-parity amendment): the pool PASSED on its own
artifact (281,241 rows, 2,338 speakers, 0 empty; train 266,138 ordinary / 1,041
nearest_fallback / 0 singleton; held 11,855 / 2,153 / 54), then the preflight FAILED its
control-parity step because the implementation asserted BIT-EXACT gradient equality
(`torch.equal`) between the two `L_online=0` copies. A read-only diagnostic
(`scripts/d7_parity_diag.py`, `log/d7_parity_diag.1446568.out`, GH200) shows the assertion is
unsatisfiable by construction on this backend: losses are exactly equal (9.983121871948242
both), cross-copy max abs gradient difference is 2.623e-06, and the SAME model object re-run on
the identical batch/RNG differs by MORE (5.484e-06) -- CUDA atomics in the FastBaumWelch
backward, `deterministic_algorithms` False. The registered clause is "control parity when
`L_online=0`"; bit-exactness was an implementation over-strengthening, and the
tolerance-not-equality principle was already pinned for D8 the same day BEFORE this failure was
observed. This amendment therefore specifies the clause's operational form; it does not relax
what the clause detects (a real state/wiring difference between control and parity paths).
Operational rule, pre-registered here before any PASS artifact exists: (1) losses must be
exactly equal; (2) in the same preflight job, same batch, same restored-RNG protocol, measure
the backend noise floor F = the maximum over at least 2 additional re-runs of ONE model of the
max abs gradient difference against that model's first run; (3) PASS requires cross-copy max
abs gradient difference <= 3*F AND F <= 1e-4. F = 0 (a deterministic backend) degenerates to
exact equality, which is then the correct demand; F > 1e-4 at this loss scale (~10) indicates a
different defect and fails the barrier. The recorded FAIL stands as a fact about the old
instrument, not about the wiring. Cost disclosed: `d7_online.py` is inside the source-identity
pin, so this one edit requires one further user-run manager restart. First live counter read,
same run: infeasible donors 0/256 pairs in shard 0's first batch (209 ordinary_window / 47
nearest_fallback).
2026-08-21 latest+1 (verifier ruling, own-infeasible-anchor amendment; pre-registered before
any re-run exists). After the user's restart the preflight PASSED under the operational parity
rule exactly as amended (losses exactly equal at 9.983121871948242; noise floor F = 7.391e-06
<= 1e-4; cross-copy delta 4.053e-06 <= 3F; `D7OnlinePreflightJob.ZxfANwBZYpaI/output/
preflight.json`, verdict PASS), and then BOTH D7.1 trainings failed closed at data load, at
21:32, on the same row: `_make_items` raises when an anchor's OWN pseudo-text cannot align to
its OWN audio under d_min=2 (`3889-130125-0028`: 481 states, minimum feasible 400 frames, T =
356 units). The preflight could not see this because it loads shard 0 only. Verifier census
over the whole pool with the production `_encode`/`_min_frames` law and the pool index's unit
lengths (`scripts/d7_own_infeasible_census.{py,json}`): exactly 4 of 281,241 rows are
own-infeasible, ALL train-role, ZERO internal-held — `3488-85273-0024`, `3889-130125-0028`,
`4492-8904-0032`, `8424-284526-0028` — and all four texts are the greedy decoder's known
runaway-repetition tail ("and not at shanghai and not at shanghai ...", "tickety tickety ...",
"hahaha..."), i.e. the established ~0.035% degenerate tail, not an interface error. The
registered contract is the D7 exact-control recipe VERBATIM, and the incumbent recipe defines
this exact case: `_load_pairs` DROPS an own-infeasible row with a counted diagnostic — the
round-1 refit `PsiAlignTrainJob.dsMKgPHQApyR` itself trained after "1 dropped as U > 2T" on
its own 28,538-row pool. Raising was therefore an implementation over-strengthening, the same
shape as the parity defect, and this amendment specifies the clause's operational form without
relaxing what it detects. Operational rule: (1) both arms drop own-infeasible rows (empty
states/units, or minimum feasible frames > T) from training as ANCHORS only — no unit NLL, no
`L_U->z`, no `L_online` visit — deterministically and identically in both arms; (2) dropped
rows remain donor-eligible unchanged, because the donor path draws from the index/unit store
and never reads text feasibility, so the registered donor law is untouched; (3) the realized
dropped set is counted AND NAMED per role in the training diagnostics, and any realized set
other than exactly these four train-role rows (a held row, a fifth row, an arm asymmetry)
FAILS the run as a different defect — this named-set bound is what keeps the drop from ever
becoming D7-style row filtering; (4) the pool artifact and its 281,241-row coverage clause are
untouched. Consequence for D8, dated here because its spec is already registered: the D8
parenthetical "(the preflight asserts greedy feasibility bed-wide)" is unsatisfiable as
written on the real bed; the D8.1a preflight instead asserts greedy feasibility on all rows
OUTSIDE this named four-row set, and the four rows drop from D8 training as anchors under the
same named-set bound, so an all-infeasible support cannot arise elsewhere. Cost disclosed: the
fix lives in `_make_items`, inside the source-identity pin and outside every job hash, so one
further user-run d7 manager restart (clearing the two train error markers) is required.
2026-08-22 (verifier): the user ran that restart 2026-08-21 22:50 and **D7.1 is COMPLETE AND
VERIFIER-CONFIRMED on both arms** (finished 23:05, one 14-minute ten-shard pass each, no
resubmit). The named-four-row drop confirmation deferred at relaunch is closed from each arm's
own `monitors.json`, digit-identical to the offline dropcheck and arm to arm; an exhaustive
cross-arm diff of the training diagnostics finds `online_weight` 0 vs 1 as the ONLY non-metric
difference, so the A/B is single-variable at the artifact level. Both fixed-final scorers exist
(`SAE_3E1.md` approach 32, verdicts 64-65, Catalog). D7.2 is AUTHORIZED to build and run as
registered above — it was never gated on new word — and the implementer launched it 23:16
(speech-llm `c40655d`; ten jobs covering all four clauses, the four D7.0/D7.1 hashes unmoved,
no policy leg in any graph; launch verification follows with its round). The implementer's
clause-2 flag is acknowledged and changes nothing: D7.1's banked one-draw point values order
clause 1's way on `L_online` (candidate 26% lower) and against clause 2 on held NLL (candidate
2.5319 vs control 2.5259); the gate is pre-registered and does not move, the D7.2 estimator is
the registered 32-draw paired read, and if clause 2 fails there D7 closes without a policy leg
per the registered failure law — that outcome would be a decision fact, not a measurement that
the candidate loss term cannot work (gate-decision-vs-measurement rule). The donor-diversity
proposal is RATIFIED AS A DISCLOSURE: on this population 32 draws with replacement reach a mean
~3.7 distinct same-speaker donors per held anchor (exactly 1 for `nearest_fallback` anchors by
construction); the registered clause is unchanged — its precision comes from the
speaker-cluster bootstrap, not the draw count — and the per-anchor distinct-donor count is a
MANDATORY reported diagnostic in the admission artifact, never a gate input, so the estimate is
never read as carrying 32 draws' worth of donor variation.
2026-08-22 later (planner): **D7 IS CLOSED.** D7.2 ran all four registered clauses and FAILED
clause 2 (candidate internal-held per-frame NLL 2.531898 against the control's 2.525882 over
8,642,253 frames; clause 1 passed decisively, clause 4 exactly, clause 3 returned no winner
under both eligibility readings) — verifier-confirmed against the artifacts, with the gate
table independently recomputed bit-exactly (`SAE_3E1.md` Verifier feedback 2026-08-22, verdicts
66-67). Per the registered gate the failure closes D7 without a policy leg and no sampler or
temperature rescue is selected from the result; D7.3 is closed by the gate, not merely
unauthorized. The closure licenses NOT FUNDING the policy leg at this operating point and is
not evidence that an online same-speaker negative cannot work; verdict 67's trade (decisive
same-speaker discrimination bought a significantly larger insertion discount, and the external
usage-gate widening is almost entirely the null side) is the recorded legacy a future design
must break rather than re-tune. The clause-3 point-versus-CI eligibility convention remains
DUAL-REPORTED per the user's pending blessing (PLAN.md queue item 2) and is not pinned here.

**D8 -- posterior-weighted multi-hypothesis scorer refit (soft EM over sampled rollouts;
USER-proposed 2026-08-21, registered same day; UNFUNDED).**

**Purpose.** The incumbent refit recipe trains the scorer on exactly one deterministic greedy
pseudo-text per utterance, so every 1-best decoding error becomes a fully-trusted target. D8 asks
whether spreading each utterance's training target over the frozen policy's own sampled hypothesis
group -- weighted by a detached, tempered posterior built from the frozen incumbent scorer and the
loop's registered text-LM prior -- improves scorer discrimination without damaging absolute fit.
The statistical grounding is one approximate EM step on the latent transcript: the sampled group is
the support, the LM prior approximates the transcript prior, the frozen scorer is the likelihood,
and the temperature tau tempers the posterior. At tau -> 0 the weights collapse to one-hot on the
best-scoring hypothesis and the method nests the incumbent 1-best refit, so the A/B is well posed.

Precedent cuts both ways and justifies the controls, not the method: lattice/soft pseudo-label
supervision and LM-decoded self-training are standard
([wav2vec-U](https://arxiv.org/abs/2105.11084) self-trains on LM-decoded output;
[IPL](https://arxiv.org/abs/2005.09267)), but
[slimIPL](https://arxiv.org/abs/2010.11524) found that keeping the LM inside the pseudo-label loop
teaches the model the LM instead of the acoustics. In-project, D0(e) measured `lm_prior_units` as
the only admissible selector view (cov 0.502), while the d4p notes record best-of-group selection
under the scorer's own score as the known self-amplification family -- and a soft posterior over
the scorer's own score is a softened form of exactly that selection. The acoustic-only arm, the
LM-dominance no-go rule and the audio-free attribution reads below exist because of these two
facts.

**Approach.** Bed and frame identical to D7: the 960 h HF/Ogg bed (281,241 utterances), frozen
enc50 K=500 raw 50 Hz unit store, frozen theta_0^G, seed-42 5% ID holdout with role-local reads,
`d_min=2`, reference text nowhere. For utterance i with units U_i, the candidate support is the
deduplicated union -- after the registered READER normalization (amended 2026-08-21, implementer
flag: the pool stores merge text verbatim, so the normalizer is the D8 reader, not the pool;
lowercase, then exactly `SearchOutHypsJob`'s fold, NFKD with combining marks dropped and remaining
non-ASCII DELETED, `scorer_diag.py:1255`; on the 281,241 pool texts the re-application is an exact
identity by construction, since the merge already applied it) -- of the D7
pool's deterministic greedy 1-best (reused at identical hash) and 12 rollouts sampled from
theta_0^G at the loop's registered operating point (group size 12, temperature 0.7 -- the group
the deployed GRPO loop actually ranks), generated by the existing rollout-dump machinery in ten
shards. (Two reader rules added 2026-08-21 after the implementer's artifact review: (1) the
rollout files also carry `kind="true"` reference-transcript rows and a gold-derived `wer` column
-- every D8 reader selects rows by WHITELISTING `kind=="rollout"` (D8.0's self-consistent read may
additionally whitelist the artifact's own `kind=="greedy"` rows), never by subtracting kinds, and
never reads `wer`; (2) the operative support's greedy member comes from the D7 pool artifact at
identical hash, never from a dump's own greedy re-decode -- argmax ties under different batching
make re-decoded greedy non-identical (0.459% word distance measured on tc100, `SAE_3E1.md`
approach 32), and the candidate's one-hot special case must coincide exactly with the control's
targets.) Score every candidate y in the loop's own per-unit shaped currency, with no new scorer,
LM, normalization or coefficient:

    s(y) = recon(y) + lam_lm * lm_prior_units(y),    lam_lm = 1.0

where `recon` is the per-unit forward-sum log-likelihood of U_i given y under the PINNED weight
scorer -- the D6 round-1 `d_min=2` refit `PsiAlignTrainJob.dsMKgPHQApyR` fixed final, chosen to
match the theta_0^G/round-1 frame the whole D7/D8 campaign runs at, not the deployed loop's moving
refit -- and `lm_prior_units` is the registered units-normalized Qwen3-1.7B-Base prior
(`base_lm_logprob_sum`, adapters disabled). (Amended 2026-08-21 on the implementer's dump-schema
flag, no operative change: the rollout dumps' raw `lm_prior` column is per generated TEXT TOKEN
while `recon` is per unit frame, so every D8 reader forms the prior term as
`lm_prior * n_tokens / n_units` with `n_units` joined from the unit store -- the existing
`curate.py` view -- before any weight or ESS statistic. tau_star is fixed by an ESS rule computed
on these weights, so a mixed-currency read would silently move tau_star and every D8.0 statistic;
the conversion is pinned in the reader and weight job docstrings.) The user's alpha and beta are
therefore not free:
alpha=1, beta=lam_lm=1.0 is the arm's deployed reward shape, and the acoustic-only variant sets
the prior coefficient to 0. Weights are w(y) proportional to exp(s(y)/tau_star) over the support,
computed once, detached, and frozen as an artifact before any training exists. A structurally
infeasible candidate (text minimum feasible frames > T_i), an empty text, or a text the scorer's
text side cannot encode after the registered normalization (out-of-inventory characters; added
2026-08-21) gets weight exactly 0 before weight normalization and is counted; such candidates
also do not count toward clause (a)'s distinct-support statistic. (Mechanism corrected
2026-08-21, implementer-measured on the binding artifact, replacing "priced automatically by
clause (a) through exclusion": the exclusion arm cannot fire there -- the fold deletes non-Latin
glyphs BEFORE the text side sees the string, the ASCII residue is always encodable because an
unmatched word maps to UNK, which is in the inventory, and all 236 non-ASCII T=0.7 texts survive
the fold as non-empty scoreable English. Leakage is therefore priced by three mechanisms, in this
order, not by exclusion: COLLAPSE -- glyph-only variants dedup to one string and thin clause
(a)'s distinct support; WEIGHT -- a partially-deleted rollout becomes a shorter English candidate
that competes at the price of its normalized string, legitimate by the same symmetry that lets
the control train on folded greedy text, since the merge applies the identical fold; and
exclusion only as the backstop for texts that fold to empty, live at higher temperatures where
whole-string leakage occurs. To make the WEIGHT price honest, one same-string rule binds
everywhere: the dedup key, the weight-side scoring -- pinned-scorer forward and LM prior -- and
the training target are the SAME normalized string; the per-group repair rate, candidates whose
normalized string differs from the raw generation, is a reported diagnostic. Implementation
pinned 2026-08-21, implementer proposal: the WEIGHT job owns the fold, the dedup key, the
re-scoring and the reported rate, co-located with the tau_star rule in one docstring, and the
dump machinery keeps scoring what it generates, unchanged; the weight job MAY reuse a stored
column verbatim wherever normalized == raw -- the same string under the same pinned scorer --
and re-scores only the differing minority, in the dump pass's exact forward CONFIGURATION --
same checkpoint, precision, prior settings and normalization, but not bit-exact numerics, since
batch composition necessarily differs (a ~1e-6-nat perturbation, ~2e-5 relative on a weight at
the grid's smallest tau, below anything the ESS rule or the arm-selection spearman can resolve;
any parity assertion against this contract compares within tolerance, never for equality); the
differing-string predicate and the repair rate are one computation. D8.0's provisional
statistics read the dumps' STORED columns, computed at dump time on raw pre-fold text --
negligible on the fork dump, 3 of 342,468 raw texts non-ASCII, and one more reason clauses (b)
and (c) only report at D8.0.) Context, measured as character-class shares only, no gate
statistic touched: theta_0^G sampled text leaks multilingual vocabulary -- non-ASCII in 3.8% of
T=0.7 hypotheses on the 512-utterance dump (planner-verified 236/6,144), 41.1%/76.2% at
T=0.9/1.0, ~0 for the fork policy and the greedy rows. A
group whose whole support is infeasible or unscoreable collapses to one-hot greedy and is
counted (the preflight asserts greedy feasibility bed-wide). tau_star comes
from a pre-registered deterministic rule, never a sweep of trainings: on the operative weight
artifact, tau_star is the grid point in {0.05, 0.1, 0.2, 0.5, 1.0} (per-unit nats) minimizing
|median-over-groups effective sample size - 3|, ESS = 1/sum_y w(y)^2; the rule lives in the weight
job's docstring per the standing pre-registration rule.

Each candidate arm trains under the D7 exact-control recipe verbatim -- `L_NLL + L_U->z`, every
round-1 refit constant, one 960 h pass in the same ten shards, prior weight 0, random
initialization, fixed final as the only checkpoint -- with a single change: at each anchor visit
the training text is drawn from the anchor's frozen weight vector with the D7-verified stateless
RNG keyed by `(seed=42, corpus_pass, global_step, anchor_id)`, instead of always the greedy text.
The expected objective is sum_y w(y) log p_psi(U_i|y) plus the matched contrastive term, and the
per-step cost equals the control's. The CONTROL is the D7.1 exact control itself, reused at
identical hash (under this recipe it is exactly the one-hot-greedy special case); no new control
trains. Training draws only train-role anchors; internal-held anchors appear only in evaluation.
No temperature, group-size, tau or coefficient sweep is admitted.

**Experiments.** Run in order:

1. **D8.0 feasibility read (CPU only, frozen artifact):** on the existing frozen fork-epoch tc100
   group-12 rollout dump `ReturnnForwardJobV2.QbIYruVEI0fF` (verified on disk 2026-08-21: 28,539
   tc100 utterances x 12 rollouts, all T=0.7, plus per-utterance greedy rows; recon column scored
   by `psi_g_tc100`; replaces "all stored temperatures reported, the T=0.7 slice decides" --
   vacuous on this single-temperature artifact. The only multi-temperature dump,
   `ReturnnForwardJobV2.J9yA1eYnxwYA`, 512 theta_0^G utterances at T={0.3,0.5,0.7,0.9,1.0} G=12,
   carries the OPERATIVE policy and now decides clause (a) -- replaces "decides nothing", see the
   Gate re-scope of 2026-08-21), compute per group:
   distinct-hypothesis count after dedup; ESS across the tau grid for the shaped and acoustic-only
   scores; the within-group fraction of weight variance explained by token count alone; and the
   median-across-groups spearman between shaped weights and each single-term weight vector
   (LM-only, acoustic-only). A pure read. Its scorer, bed AND SAMPLING POLICY all differ from
   D8.1a's (fork epoch vs theta_0^G), so its verdict is the clause-(a) go/no-go on the theta_0^G
   slice plus a provisional tau; every other statistic is reported, not binding (re-scoped
   2026-08-21, see Gate).
2. **D8.1a candidate generation and weight artifact:** one group-12 T=0.7 sampled dump pass of
   theta_0^G over all 281,241 utterances (ten shards, existing dump machinery, recon under the
   pinned weight scorer, `lm_prior_units` under the registered prior); then a deterministic weight
   job recomputes every D8.0 statistic on the operative bed, fixes tau_star by the registered
   rule, and freezes the per-utterance supports and weight vectors. The no-go clauses re-apply
   verbatim; a no-go here closes D8 before any training.
3. **D8.1b scorer arms:** train candidate-shaped (weights from s) and candidate-acoustic (weights
   from recon alone), both full-bed, subject to the arm-selection rule in the Gate. Persist
   fixed-final scorers, role hashes, the sampler seed/state contract, loss curves, internal-held
   reads and sampling diagnostics. No policy trains.
4. **D8.2 label-free admission:** paired against the D7.1 exact control at fixed finals --
   internal-held per-frame NLL on the held greedy targets; the paired corruption-ladder read; then
   the unchanged all-1,500-row Acceptance gate v2 and `PsiScorerParityJob`. Reference text,
   rollout WER and intermediate checkpoints remain sealed.
5. **D8.3 matched one-leg policy assay:** only if D8.2 passes, request launch authorization; same
   design as D7.3 (freeze the scorer, one otherwise-identical theta_0^G 960 h G-track leg against
   the exact control leg, paired dev-clean/dev-other plain WER and S/D/I). The proposal's eventual
   once-per-leg periodic refresh inside the deployed loop is a separate arm that would be
   registered only after a D8.3 pass; nothing here authorizes it.
6. **D8.4 paired ranking-quality (eta) read (added 2026-08-23 on the USER's reopening
   directive; ordered before any D8.3 authorization despite its number):** both fixed-final
   scorers -- candidate-acoustic (`D8ScorerRefitJob.2bQzhz6U1yHp`) and the exact control
   (`D7OnlineTrainJob.j16rTskXF1QU`) -- rerank the SAME banked rollout groups, and the read is
   the paired delta eta with the registered bootstrap. Full spec and pre-declared reading in
   the Status entry of 2026-08-23 later; the reporting rule goes verbatim into the producing
   job's docstring before any result exists.

**Gate.** (Scope amended 2026-08-21, before any clause statistic was computed anywhere -- the
implementer flagged, and deliberately declined to measure, that clauses (a) and (b) are
sampling-policy properties while the fork-epoch dump is the wrong policy; D5(a) established the
fork policy over-generates, so an unscoped D8.0 could close D8 on diversity theta_0^G would not
produce. At D8.0, only clause (a) -- the scorer-free statistic -- BINDS, and it is read on the
theta_0^G T=0.7 slice of `J9yA1eYnxwYA` (512 groups; a median is robust at that n); clauses (b)
and (c) are computed and reported on both artifacts, since neither carries the pinned weight
scorer, and bind only at D8.1a, where all three clauses re-apply verbatim on the operative bed,
policy and scorer.) No-go clauses, each closing D8 with no rescue selected from the result:
(a) median distinct hypotheses per group < 3 of 12 -- the support is too thin and the method
degenerates to the incumbent at extra cost; (b) at every grid tau the median ESS falls outside
[1.5, 8] -- the posterior is one-hot or near-uniform at every registered temperature; (c) token
count alone explains >= 50% of within-group weight variance at tau_star -- the weights are a
length artifact, and the standing per-unit normalization is the only admitted counter-measure.
Arm selection, decided from D8.1a statistics before any training: if spearman(shaped weights,
LM-only weights) > 0.95, candidate-shaped is NOT funded (its target is free English by
construction) and only candidate-acoustic trains; if spearman(shaped, acoustic-only) > 0.95 the
two arms are operationally identical and only candidate-acoustic trains; otherwise both train.

D8.2 passes for a candidate only if all four hold:

1. paired candidate-minus-control internal-held per-frame NLL on the held greedy targets has a
   speaker-cluster-bootstrap 95% upper bound below delta_NI, where delta_NI is the
   speaker-cluster bootstrap standard deviation of the CONTROL's own held per-frame NLL, computed
   and recorded before any candidate number is read. This is non-inferiority with a data-defined
   margin rather than D7's zero margin, and deliberately so: the candidate trains on soft targets,
   so demanding it match the control on the control's own training distribution at margin zero
   would structurally reject the method under its success mode -- the same trap the gate-v2 (i)
   amendment closed for changed training text;
2. candidate-minus-control corruption-ladder spearman on the unchanged 1,500 external rows has a
   row-bootstrap 95% lower bound above zero -- the mechanism's claimed win, required to clear
   noise per the standing spread rule;
3. every cumulative Acceptance-gate-v2 clause passes at unchanged floors and population; and
4. scorer parity passes before any live reward read.

If both candidates pass, candidate-acoustic is preferred unless candidate-shaped beats it on
clause 2 at the same bootstrap resolution: circularity toward the deployed reward and LM
distillation count as costs at parity. Failure at any rung closes D8 without a policy leg; no
tau, temperature, support or coefficient rescue is selected from results. After an authorized
D8.3 submission, the local scientific pass is candidate WER strictly below the matched control on
both dev splits; a scorer-side pass without the two-split policy win establishes conditional
discrimination only, not useful ASR progress.

**Status.** REGISTERED 2026-08-21 on the user's drafting instruction; NOT IMPLEMENTED, NOT
FUNDED. D8.0 is a CPU read of a frozen artifact and may be implemented on the user's word at any
time. D8.1a-b additionally wait for the D7.2 verdict: the two campaigns share the control arm,
the admission machinery and the GPU budget, and a D7.2 admission would move gate v2's
last-accepted reference. The weight scorer and the paired control REMAIN PINNED as written
(`dsMKgPHQApyR`; the D7.1 exact control) unless a dated amendment re-pins them; a combined D7+D8
objective is not specified here and would be a separately registered arm. Estimated envelope
beyond D8.0: one group-12 sampled dump pass over the 960 h bed (3.37M decodes, priced 2026-08-21
at about 12x the D7 greedy pass, which measured ~1.5 h on ten GPUs, plus one pinned-scorer
forward per candidate; replaces the vaguer d6periodic-envelope wording), one CPU weight job, and
up to two D7.1-sized fixed-final trainings;
the policy leg is priced separately at D8.3 authorization. Labels appear nowhere: rollouts read
audio only, every gate statistic is label-free, and speaker IDs enter only the evaluation-side
cluster bootstrap, as in D7.2.
2026-08-21 later (USER): D8 FUNDED — "I approve starting D8". This is the launch word for D8.0
(implement and run the CPU read of the frozen fork-epoch artifact now). It does NOT lift the
registered ordering: D8.1a-b still wait for the D7.2 admission verdict (shared control arm,
admission machinery and reference), and D8.3 still requires its own authorization after a D8.2
pass. No gate clause, pin or statistic changes with this funding line.
2026-08-22 (planner ruling on the D8.0 UNRESOLVED read; resolves the implementer's proposal 1,
ratifies proposal 2, answers proposal 3 — pre-registered before any corrected statistic exists).
Verdicts 59-60 are accepted: clause (a) cannot be read on the frozen theta_0^G dump as
instrumented, because the exclusion was evaluated against the dump's own joined ~12.5 Hz pooled
unit store while every D8.1b training aligns to the raw 50 Hz store — the 5,096/5,730
"infeasible" members are an instrument property, the v2 law-conflict guard fired correctly, and
the v1 NO-GO is superseded evidence. NEITHER offered reading binds: the dedup-only count ignores
the registered exclusion and would change what the clause detects, and the as-run exclusion is
the wrong frame. Operational form: a support member is structurally infeasible for clause (a)
iff its READER-NORMALIZED text's minimum feasible frames at d_min=2 under the registered text
side exceeds T_i read from the frozen raw 50 Hz store `PackUnitsJob.I0uzRMfUrKWC` — the
operative D8.1a/D8.1b frame. This is text-plus-store arithmetic, the same law the D7 drop rule
uses; it needs no scorer forward and NO new dump, so the proposal's third option is declined as
unnecessary. The binding statistic is the median over the 512 binding-slice groups of the
distinct FEASIBLE support count with the greedy member included (max 13;
`distinct_rollouts_only` stays a reported column); threshold < 3 means no-go, unchanged.
Execution: extend the registered read job (v3) changing ONLY the feasibility join to the raw
50 Hz store, with a full-coverage assert over the slice's 512 ids. The v2 law-conflict guard
retires for v3 by construction — the exclusion is now deliberately the operative law, not the
artifact's own, so a stored-finite-but-operative-infeasible member is an expected, counted
category (it is exactly D8.1a's weight-0 set), not a contradiction. Safety valve of the same
kind as the parity noise floor: if the operative-law exclusion removes more than 5% of scored
members on the binding slice, the read returns UNRESOLVED for planner review instead of feeding
clause (a) — the matched-law fork read prices the genuine rate at 0.018% and the D7 greedy
census at 4/281,241, so 5% fires only on another frame error. The clause verdict first exists
in the v3 job's own output, per the standing pre-registration discipline. Proposal 2 RATIFIED:
the dedup survivor rule (the member already in normalized form, else the earliest stored row)
is the reader's tie law; for clause (a) only the count matters, and at D8.1a the weight job's
same-string rule governs scoring, so the tie rule affects report columns only — pin it beside
the tau_star rule in the weight-job docstring. Proposal 3 needs no plan change: the leg-1
scorer provenance is exactly the registered "Disclosed asymmetry" paragraph of
D6-PERIODIC/GAN960-FROZEN; audits read the plan, and the job record adds no new claim.
2026-08-22 later: the ruling is EXECUTED AND VERIFIER-CONFIRMED (speech-llm `3843918`;
operative v3 jobs `D8FeasibilityReadJob.mv2d0vkWN93a` / `.W7TWfwoZtkaC`; independent recompute
of the binding slice matches exactly). Clause (a) reads GO in the operative frame — exclusion
0 of 5,730, median distinct feasible support 12 of 13 against the threshold 3, safety valve
not fired, fork read unchanged to the last digit. Two conservative implementation deviations
ratified as the operational form (valve denominator = all excluded scored members; coverage
assert over the whole dump). This GO discharges D8.0 ONLY: D8.1a-b remain gated behind the
D7.2 admission verdict and D8.3 behind its own authorization, exactly as registered. Noted
for D8.1a, selecting nothing now: the binding slice's provisional rho(shaped, LM-only) is
0.9790, above the registered 0.95 arm-selection bar.
2026-08-22 later (planner): **D8.1a-b ARE RELEASED.** The D7.2 verdict is in (FAIL on clause 2,
verifier-confirmed; D7 Status above), which is the registered release condition — the ordering
waited on the VERDICT, not on a pass. Consequences, all as already written: the D7.1 exact
control (`D7OnlineTrainJob.j16rTskXF1QU`) remains the pinned comparator and gate v2's
last-accepted reference does NOT move; the weight scorer stays `dsMKgPHQApyR`; the admission
machinery and GPU budget are free (the whole D7 config is finished). The implementer may build
and launch D8.1a — the group-12 T=0.7 sampled dump pass over all 281,241 utterances plus the
deterministic weight job — under the user's standing D8 funding and the priced envelope, with
no displacement of the running D6-periodic loops. The three no-go clauses re-apply verbatim at
D8.1a and the arm-selection rule reads D8.1a statistics alone (the banked D8.0 rho 0.9790 > 0.95
forewarns that candidate-shaped is likely not funded, but the D8.1a value decides). D8.1b
trains only per that rule; D8.2 follows the registered design; D8.3 still requires its own
authorization after a D8.2 pass. One operational requirement added for D8.2, registered now
before that job exists (from the D7.2 verification, `SAE_3E1.md` Verifier feedback 2026-08-22):
the D8.2 admission job MUST persist its per-anchor paired deltas and speaker-cluster ids beside
the aggregate artifact, so the paired mean, negative share and bootstrap bound are re-derivable
from disk — in D7.2 those three rested solely on the job's own arithmetic, harmless only
because the gate closed on clause 2's deterministic comparison.
2026-08-22 later (planner, on the launched D8.1a build; verifier round in `SAE_3E1.md`): the
launch is sound at the frame level (constants trace, shards genuinely reused, weight job
label-free, interface equivalence independently confirmed over the whole shared population),
and two operational rulings bind before any D8.1a number is read. (1) GREEDY PROVENANCE: the
registered "greedy 1-best reused at identical hash" is implemented as an in-dump regeneration
through a different code path at the same checkpoint; this operational form is ACCEPTED IF AND
ONLY IF a zero-mismatch normalized-text equivalence read against the D7 pool's greedy texts
over all 281,241 utterances exists once the dump finishes — any mismatch makes the D8.1a read
UNRESOLVED for planner review before D8.1b, and no verdict is accepted before the read exists.
(2) The 5 % SAFETY VALVE ruled for D8.0's clause-(a) read applies at D8.1a identically:
exclusion above 5 % of scored members on the binding slice returns UNRESOLVED instead of
feeding clause (a). Five weight-job fixes (dead temperature filter -> fail-closed T==0.7
assert; valve enforcement; tau_star undefined-ESS fallback to match D8.0; the RATIFIED dedup
survivor rule with the score-differing-collapse diagnostic; loud counting of
feasible-but-non-finite-recon members and de-double-counted exclusion counters) are required
to land before `D8WeightJob` first executes — the dump's ~8 h remaining runtime is the window,
and if any fix moves the job hash the implementer states it and the planner requests the one
manager restart.
2026-08-22 latest: ALL FIVE FIXES ARE IN AND VERIFIER-CONFIRMED (speech-llm `3af12bd`),
hash-neutrality independently reproduced by a fresh graph build (weight/merge/shard hashes
unmoved), and the greedy-equivalence read is in the graph as
`D8GreedyEquivalenceJob.XTdRp3OO3LNf`, whose EQUIVALENT verdict requires full 281,241-tag
coverage by construction. The equivalence read remains a sibling output rather than a
`D8WeightJob` input; this is ACCEPTED as a process gate because the planner is the verdict's
only consumer and acceptance requires the read — no hash-moving rewiring. Both rulings above
stand unchanged; next planner action is reading the weight artifact, the equivalence read and
the valve together when the dump finishes.
2026-08-22 (relaunch amendment): the first dump launch was cancelled at 49 % on a verified
wall-clock projection (two shards finishing past the unraisable 11.5 h clamp; no forward
resume) and relaunched at the corpus-scale batching `max_seqs=8`, which traces to the
`config_sae_3e1_d4p_v1` full-corpus dump — batching moves cost, not the registered
distribution or any deterministic column, and no first-launch number was ever banked. ALL
downstream hashes moved and are verifier-reconfirmed by an independent rebuild: weight
`D8WeightJob.1G2lPRnRmPks`, merge `D8MergeRolloutsJob.gXDwFsfvraDS`, equivalence
`D8GreedyEquivalenceJob.xR1RduqgjFKe` (the `XTdRp3OO3LNf` pin above is superseded — that
registration is now an orphan). Both rulings and the process gate carry over to the new
hashes unchanged.
2026-08-22 latest+1 (planner ruling on the failed equivalence read; resolves the verdict-70
fork). The read is in and is NOT EQUIVALENT — 31,562 of 281,241 utterances differ at exact
coverage (0 only-in-dump, 0 only-in-pool, 0 duplicates; planner-verified directly against
`greedy_equivalence.json` and sampled `mismatches.jsonl` rows: single rare-word lexical flips
inside otherwise identical sentences). Per ruling (1) above, the in-dump regeneration's
conditional acceptance is void by its own condition and D8.1a is UNRESOLVED on the launched
support. RESOLUTION — restore the registered support; nothing in the registration moves: the
Approach's own reader rule (2) already says the operative support's greedy member comes from
the D7 pool artifact at identical hash, never from a dump's own greedy re-decode. The weight
job is rewired to take the D7 pool artifact as an explicit hash-carried input; each utterance's
support is the registered dedup of the POOL greedy member plus the dump's twelve
`kind=="rollout"` members; the dump's regenerated `kind=="greedy"` rows are QUARANTINED as
support for every D8 reader and remain in the merge artifact only as the divergence record,
together with the equivalence read. Scoring law unchanged — the registered same-string rule
already covers this case: stored dump columns are reused verbatim only where the normalized
string is identical, and the pool greedy member is scored for the differing utterances in the
dump pass's exact forward configuration under the pinned scorer and registered prior,
within-tolerance parity never bit equality, exactly as the weight-job contract reads. The
corrected weight job must assert full coverage on both sides (281,241 pool members; twelve
whitelisted rollouts per group; zero duplicates) — safe by measurement, since the equivalence
read proved exact ID alignment. Its hash moves; the implementer states the new hash and the
planner requests the one manager restart if needed. The quarantined
`D8WeightJob.1G2lPRnRmPks` output is not a D8.1a result and feeds nothing. The standing
three-together read then applies to the CORRECTED weight artifact: no-go clauses (a)/(b)/(c)
verbatim, the 5% valve, and the arm-selection rule on D8.1a statistics alone; D8.1b waits on
that read; D8.2/D8.3 unchanged. Banked as a descriptive fact with the artifact as its record:
two argmax implementations of the same checkpoint at different batching disagree on 11.22% of
utterances (lexically, on rare words) on this bed — consistent with the 0.459% tc100 word
distance already in approach 32 — so decode-path identity is never again assumed without a
read of exactly this kind.

2026-08-22 latest+2 (planner ruling on the collapse-diagnostic proposal; execution notes for
ruling piece 3). Pieces 1 and 2 of the latest+1 ruling are verified as built (line review of
speech-llm `54929cf` plus the 21/21 support suite re-run by the planner); the hash-sequencing
position — no weight hash statable until piece 3's producing job exists — is correct and is
what the ruling intended. COLLAPSE PROPOSAL, ruled in three parts. (1) The substance is
ACCEPTED and planner-verified from the merged dump: the dedup diagnostic flags only
`round(recon,12)` spread, but lm_prior is tokenization-dependent while recon is
text-determined — over rollout-only classes the planner measures 17,874 of 155,890 collapse
classes with differing lm_prior and 17,713 with shaped-numerator spread above 0.01 nats
(max 65.0); including the quarantined dump-greedy rows, 84,649 and 74,410 of 198,172. The
proposal's own printed figures (197,825/135/77,077/73,902/0.881/35.2) sit in neither of those
two populations, so when the diagnostic is banked its population rule must be stated in the
output; no verdict rests on the ad hoc figures and nothing blocks. (2) The SURVIVOR RULE DOES
NOT CHANGE for D8.1a: it is the registered computation, it is deterministic, and both arms of
the A/B read the same support builder over the same merged dump, so the tokenization
arbitrariness is arm-shared and cannot bias the comparison; changing the statistic between
registration and read would unregister the gate. Any survivor-rule change (for example scoring
the canonical tokenization) is a NEW registration for a later round. (3) The DIAGNOSTIC IS
EXTENDED inside the already-moving weight-job hash, report-only, feeding no clause and no
valve: per-class lm_prior-differ count and shaped-numerator spread statistics (count above
0.01 nats, p50/p90/max) over the OPERATIVE post-exclusion support population, banked in the
weight artifact as the authoritative measurement superseding all ad hoc figures. PIECE 3
EXECUTION NOTES, binding on the build: (a) the pool member's score is DEFINED as the score of
the pool STRING through the dataset text pipeline — the same path the D7.1 control consumed
its targets through — never a reconstruction of the D7 decode's internal token path;
control-consumption parity is the operative meaning of "the pool member's score". (b) The
implementer's own flagged check is required: the pass must produce non-degenerate columns
where the banked dump's `kind=="true"` rows are empty. (c) An OVERLAP PROBE is required
before the 31,562 differing rows are trusted: score a sample (at least 64) of AGREEING tags
through the same true-row hook; recon must agree with the dump's stored greedy columns within
tolerance (this validates forward-configuration parity end to end), and the probe reports the
lm_prior and n_tokens deltas as the measured decode-path-versus-text-path tokenization gap —
the same phenomenon the collapse proposal surfaced — so the size of the mixed convention
(reused dump columns on agreeing rows, text-path scores on differing rows) is bounded by
measurement inside the artifact before any weight read.

2026-08-22 latest+3 (planner ruling on the held mixed-convention question; resolves the
verdict-71 hold). The probe did exactly what it was registered to do: the bound came back, and
it is too large and one-sided to admit. The verifier's signed read of
`D8PoolOverlapProbeJob.GerShND5ibtT` (all headline statistics independently reproduced first)
shows the shaped-score NUMERATOR (`lm_prior * n_tokens`, the quantity the weights consume) is
HIGHER through the text path on 64 of 64 tags — median +9.17 nats, mean +9.53, range +6.94 to
+17.69 — because the decode path always spends more tokens on the same normalized string
(`n_tokens` delta negative 64 of 64, median -1, minimum -3; the per-token column alone, median
+0.097, understates this and is sign-mixed 61/3). RULING, in five parts. (1) The MIXED
CONVENTION IS REJECTED for D8.1a. Binding note (a) DEFINES the member's score as the text-path
score of the pool string; the latest+1 reuse clause was an efficiency shortcut resting on the
assumption that the score is a function of the string. The probe refutes that assumption for
the prior columns while confirming it for `recon` (4.77e-07 maximum). Reusing decode-path
prior columns on agreeing tags would score the member ~9.5 nats below its defined value in
88.8 % of classes and at its defined value in 11.2 % — a one-sided offset, three orders above
the collapse diagnostic's own 0.01-nat materiality line, concentrated exactly on the
divergence classes, in the column the shaped arm consumes, with the arm-selection bar
(rho 0.9790 vs 0.95) sitting 0.03 from its threshold. (2) CORRECTED LAW: the pool member's
`lm_prior` and `n_tokens` come from the text path on ALL 281,241 tags; `recon` may be reused
on agreeing tags (validated string-determined). Two admissible implementations, implementer's
choice: (i) extend the probe-validated pass machinery to the full bed, ten shards at the
dump's own measured rate and sharding, zero new code; or (ii) the built four differing shards
plus a text-only prior scorer over the 249,679 agreeing strings — admissible ONLY if the
text-only scorer reproduces the audio-pass text-path values on ALL 31,626 reference rows
(31,562 differing + 64 probe: `n_tokens` identical, per-token `lm_prior` maximum absolute
difference <= 1e-4); failing that bar falls back to (i). (3) MECHANISM NAMING, disclosure not
gate: one log line naming which token(s) the decode path spends that the text path does not
(the constant -1 suggests a boundary token), from stored token data or a three-tag
inspection. (4) PRE-REGISTERED CONVENTION-SENSITIVITY LINE in the three-together read: the
no-go clauses, the 5 % valve, and the arm selection are computed under the corrected
convention AND recomputed under the legacy mixed convention (both computable from the same
artifacts on agreeing tags, no new compute); any flip of any verdict renders the read
UNRESOLVED and returns it to the planner; no flip makes the convention choice immaterial to
the decision by measurement. (5) The weight job's prior-column reuse branch retires and its
asserts move to full member coverage; `D8WeightJob.qBb5teJvluqB` was correctly stated under
the standing requirement and is superseded by construction — the requirement stands
unchanged: state the successor hash in State before any manager restart, after which the
implementer may start the D8.1a manager. This is the last D8.1a-scoring spend the planner
will order; any further convention issue goes to the user.

2026-08-23 (planner verdict acceptance and D8.1b authorization). D8.1a is COMPLETE and its GO
is ACCEPTED: the verifier independently recomputed every clause statistic from the frozen
`D8WeightJob.juRpzTNHKCSq/output/supports.jsonl` over all 281,241 groups with a fresh
implementation of the registered definitions and reproduced the artifact to the last digit
(median distinct 13; median shaped ESS 2.982409 / 5.32854 in the [1.5, 8] band at tau 0.05 /
0.1 and outside it elsewhere; tau_star 0.05; median per-group spearman shaped-versus-LM-only
0.34615384615384615 and shaped-versus-acoustic-only 0.9835164835164836, average-rank ties,
exactly as the code states; median token-count R-squared 0.06203486419535191 on 278,215
defined groups; member accounting 3,170,658 live = 3,170,676 scored minus 18 excluded). All
three no-go clauses pass, the valve is idle at 5.7e-06 against 0.05, and the arm-selection
rule fires on its REDUNDANCY clause: shaped and acoustic-only weights are operationally
identical (0.9835 > 0.95), so `candidate_shaped` is not funded and `candidate_acoustic` is the
one funded arm. The pre-registered convention-sensitivity line finds no flip, so the latest+3
convention question is closed by measurement. Reconciliation for the record: the banked D8.0
forewarning rho(shaped, LM-only) 0.9790 was measured on the fork-epoch policy's 512-group
slice and was explicitly non-binding; on the operative theta_0^G bed the same statistic is
0.3462, and the registered rule reads D8.1a statistics alone — there is no contradiction,
the two beds' rollout populations differ exactly the way D5(a) said they would. RULING:
D8.1b is AUTHORIZED per registered phase item 3, for `candidate_acoustic` ONLY — weights from
`recon` alone at tau_star 0.05 read from the frozen artifact, full-bed, fixed-final scorer
persisted with role hashes, the sampler seed/state contract, loss curves, internal-held reads
and sampling diagnostics; no policy trains; D8.2's already-registered persistence requirement
(per-anchor paired deltas and speaker-cluster ids beside the aggregate) binds its admission
job. The shaped arm may not be revived from this read; a future shaped arm is a new
registration.

2026-08-23 later (D8.1b acceptance and D8.2 authorization). D8.1b is COMPLETE and ACCEPTED
(verifier-confirmed): the realized greedy-draw fraction 0.25312 reproduces the frozen
artifact's 0.2527 mean greedy weight to 4.6e-04 on a quantity nothing tuned, zero drawn
members were infeasible, per-step cost parity with the control held (13:52 vs 13:59 over an
identical 10 shards / 2,361 batches / 267,175 + 14,062 anchors), and the persistence set is
banked content-bound (fixed-final checkpoint, role hashes, code identity, input sha256s
including the frozen weight artifact, sampler RNG contract, ten per-shard loss records). The
internal-held per-frame NLL banked by the training job is DESCRIPTIVE ONLY, exactly as verdict
77 fences it. D8.2 IS AUTHORIZED, with one convention pinned by this ruling: the authorized
persistence set necessarily exposes the candidate's held aggregate before the admission job
exists, so the registered "delta_NI computed and recorded before any candidate number is read"
is discharged by CONVENTION PINNING rather than by literal sequencing — delta_NI is the
speaker-cluster bootstrap standard deviation of the control's own held per-frame NLL computed
by the D7.2 admission implementation VERBATIM (its resample count, seed rule, and
speaker-cluster construction, all fixed before any D8 candidate existed), leaving zero free
choices; any deviation from that convention is a new registration and returns to the planner.
The admission job must persist the per-anchor paired deltas and speaker-cluster ids beside the
aggregate (registered 2026-08-22), and the rest of the registered D8.2 battery applies
unchanged: the paired corruption-ladder read, the all-1,500-row Acceptance gate v2, and
`PsiScorerParityJob`, with reference text, rollout WER and intermediate checkpoints sealed.

2026-08-23 latest (D8.2 verdict and D8 CLOSURE). D8.2 DOES NOT PASS and, per the registered
gate ("D8.2 passes for a candidate only if all four hold"; "failure at any rung closes D8
without a policy leg; no tau, temperature, support or coefficient rescue is selected from
results"), **D8 IS CLOSED WITHOUT A POLICY LEG**. D8.3 never runs and the proposal's eventual
periodic-refresh arm, registered only behind a D8.3 pass, dies with it. Verifier-confirmed
from the artifacts: clause 1 PASSES decisively — paired mean -0.012475, speaker-cluster
bootstrap one-sided 95 % upper bound -0.011800 against delta_NI 0.004826, and the bound is
below ZERO, so the data-defined margin never became load-bearing; the verifier reproduced
mean, delta_NI, upper bound and negative share BIT-EXACTLY from the persisted
`per_anchor.jsonl`, which is the per-anchor persistence registration doing its job. Clause 2
FAILS — no corruption ladder's lower bound clears zero and filler-insertion discrimination is
significantly degraded (-0.00333 [-0.00638, -0.00028]); clause 3 FAILS — gate v2 returns NO
WINNER under both the point and CI readings because the candidate is ineligible on the
ladder-not-below clause, despite passing the floors, both improvement clauses, ce_loo 2.23428
vs 2.25883, and significantly improving the matched insertion discount at every k. Clause 4
(`PsiScorerParityJob.sRJ7LUmF4nMw`) completes and is read FOR THE RECORD; it cannot change
the outcome. The registered legacy to carry: the acoustic-weighted refit learned the target
distribution better (held fit better than the control at the strict zero margin) and priced
insertions better, without learning to discriminate corruption better — the failure mode the
candidate-acoustic arm was registered to expose. What this licenses: soft multi-hypothesis
targets are NOT funded to a policy leg at this operating point (group 12, T=0.7, tau_star
0.05, acoustic-only weights); it is not a finding that they cannot work, and no operating
point may be selected from this table.

2026-08-23 later (USER overrules the closure: D8 REOPENED; two standing rules; D8.4
registered). The USER's directive: paired data for model evaluation is now a real rule; D8 is
not closed; the phase question must be answered by measuring eta -- the real ranking quality
-- to say whether the candidate is better or worse, in a fair comparison; and a
planner-constructed kill gate must not kill a phase. Ruling: the closure entry above stands as
an accurate record of the registered gate firing and is SUPERSEDED as a phase verdict by this
directive -- the USER is the one authority above a registration. What survives unchanged: no
tau, temperature, support or coefficient is selected from the failed D8.2 tables; both
operating points stay frozen exactly as trained (candidate-acoustic at tau_star 0.05, fixed
final; the D7 exact control, fixed final). The reopening adds a measurement, not a rescue.
Standing rules (added same day to `PLAN.md` North star & hard constraints): (1) every
model-evaluation comparison is PAIRED -- same items, per-item deltas, resampled CI, never two
pooled numbers; (2) proxy clause batteries (corruption ladders, constructed discrimination
statistics) may gate spend inside a phase but never close one -- a phase-closing
better-or-worse verdict requires the direct measurement of the real target quantity in a fair
paired comparison, and the closure decision then rests with the user. D8.4 -- paired
ranking-quality (eta) read, pre-declared before any statistic exists:
- VEHICLE. The psi rerank/gate-table machinery (`PsiAlignRerankJob` family), whose eta is the
  registered convention: eta = (mean_WER - selected_WER)/(mean_WER - oracle_WER) as a ratio of
  corpus means over rollout groups, first-max argmax ties, groups containing an unalignable
  candidate dropped whole with the rank-columns sensitivity table, produced by the registered
  reader that prints its own convention. Step zero: `PsiScorerParityJob.sRJ7LUmF4nMw` (clause
  4, still running) is read FIRST when it lands -- if it already carries the paired
  candidate-vs-control rerank on an operative-policy bed with the registered nulls, D8.4
  discharges by reading it; otherwise fund exactly the missing rerank probe, no new scorer or
  dump machinery.
- FAIRNESS PINS. Same-bed/same-n/same-G/same-draw (absolute-eta bars are withdrawn; only the
  paired delta is read): both scorers rerank the SAME banked rollout groups from the operative
  theta_0^G-family policy at T=0.7, G=12, n >= 512 utterances, reusing an existing banked dump
  where role hygiene admits it before any new decode spend; the two arms differ in the scorer
  only. Transcripts enter as evaluation measurement only (the label quarantine's allowed use);
  both scorers are fixed-final, so the read selects nothing.
- READING (a measurement, not a kill gate). Primary: paired delta eta (candidate minus
  control) with the registered `bootstrap_delta_eta` (utterance resampling on shared groups;
  the shared mean/oracle denominators cancel), 95 % CI. On shared groups the delta eta equals
  the paired selection-WER delta divided by the shared positive oracle headroom, so this IS
  the plain-WER form of the same question. BETTER if the CI excludes zero in the candidate's
  favor; WORSE if it excludes zero against; otherwise INDISTINGUISHABLE, resolving to the
  control per the standing incumbent-tie rule. Context columns, reported never gating:
  spearman, the AR-free null, the length-only null, the OOV-count null (the standing null
  battery), beside the D8.2 clause results above.
- CONSEQUENCE. The verdict goes to the USER with the D8.3 authorization question attached:
  BETTER makes D8.3 the natural next ask; WORSE or INDISTINGUISHABLE makes non-funding the
  natural ask -- but per standing rule (2) the phase closes only on the USER's word over this
  measured number, never automatically.

2026-08-23 latest (planner ruling on the launched D8.4 build; verifier round in `SAE_3E1.md`).
The build is ACCEPTED IN STRUCTURE with ONE REQUIRED CORRECTION before the verdict may be read.
Ratified: step zero's answer (verified on the class -- `PsiScorerParityJob` re-scores ONE arm's
own rerank through the online path against its own `recon` column; no second arm, no eta, so it
cannot discharge D8.4); the reuse of PLAN_3A's paired instrument at the pinned `n_boot=10000`,
`seed=42`; the three-way reading in the producing module's docstring, which carries the
registered rule verbatim; the refusal tests; the manager replacement. REQUIRED: the launched
rerank pair consumes `ReturnnForwardJobV2.QbIYruVEI0fF` (alias `forkep2_tc100full_g12_T0.7`) --
the FORK-EPOCH-2 policy's rollouts, which the registration's own fairness pin excludes: the
D8.0 gate re-scope already classified that dump as "fork epoch vs theta_0^G" and moved its
binding clause to `ReturnnForwardJobV2.J9yA1eYnxwYA` (alias `gtrack_p10_tc100_n512_g12_parts`)
precisely because it "carries the OPERATIVE policy". The State's claim that the fairness pins
were already satisfied by the D8.2 graph is therefore wrong on the policy pin, and the module
docstring (operative theta_0^G-family) contradicts the wiring. Ruling: (i) the PRIMARY D8.4
verdict is read from a second rerank pair on `J9yA1eYnxwYA`'s T=0.7 slice -- n=512 (exactly the
registered floor), G=12, per-rollout WER banked in the dump, same fixed-final scorers, same
paired instrument; cheap (about 2 % of the running rerank's bed). (ii) The launched fork-policy
pair COMPLETES and its paired delta prints as the same-bed-as-precedent CONTEXT column -- it is
directly comparable to the banked D7 rerank etas (candidate 0.258 / control 0.250 on this very
dump) and so carries the D7-vs-D8 continuity story -- but it is never the verdict. (iii) The
`D8EtaReadJob` verdict key is re-registered to the operative pair with the fork pair as
context; transcripts of the 512 tc100 utterances enter as evaluation measurement only, the same
use every reward-rank probe of this dump family has made, disclosed.

2026-08-23 closing (D8.4 FAILED CLOSED on the operative bed as registered; bed re-pin ruling).
`D8EtaReadJob.S3NTCZAOfSnZ` refused per its own registered guard: 46 of 512 groups survive,
because 25,867 of 31,744 rollout rows (81.5 pct) are unalignable by the psi family on the
registered units join -- identically in BOTH arms (verified on disk: both rerank JSONs report the
same 25,867/498 counts), so infeasibility belongs to the text-to-unit alignment, not to either
scorer's weights. The guard behaved exactly as the standing rule wants: it gated a broken
measurement and closed nothing. Mechanism verified by the planner's own read of both stores: the
operative sae3d store (`MergeUnitsPklJob.hJmZtbPDa2hd`) and the 50 Hz enc50 tc100 pkl
(`MergeUnitsPklJob.ncxcd3vouD5E`) carry the IDENTICAL 34,106 utterances at mean 146.8 vs 585.7
unit frames per utterance, per-utterance ratio inside [3.79, 4.00] (median 3.99) -- the same
stream at a quarter of the frame rate -- and under the standing minimum-duration topology
(d_min >= 2 frames per symbol state) most operative-bed rollout texts need more states than the
frames can host. The frame error was the PLANNER'S at registration: D8.0's clause (a) ("each
dump joins ITS OWN unit store") governs reading a dump's STORED per-frame columns, and I carried
it into FRESH alignment scoring, where the instrument's native evidence stream is what matters --
both fixed-final scorers train against the same frozen 50 Hz enc50 store
(`PackUnitsJob.I0uzRMfUrKWC`, verified in both training jobs' info files), so the quarter-rate
join handed the instrument a bed it structurally cannot read.

RULING (adopts implementer proposal 1; replaces the units join of the D8.4 primary pair,
2026-08-23, because the registered join is structurally unreadable by the instrument): the
primary pair re-runs on the SAME `J9yA1eYnxwYA` T=0.7 slice -- same policy, same draw, same
banked per-rollout WERs -- joined to the 50 Hz enc50 tc100 pkl `MergeUnitsPklJob.ncxcd3vouD5E`
(verified to cover all 512 dump utterances). Everything else is unchanged: same fixed-final
scorers differing in `model_pt` alone, `n_boot=10000`, `seed=42`, the three-way reading with the
incumbent-tie rule, and the reader guard (shared groups at or just below 512) -- which is now
expected to PASS; if it refuses again that is again a planner matter, never a fallback. Clause
(a) is SCOPED, not revoked: stored-column reads keep the own-store join; fresh alignment scoring
joins the scorer's native-rate stream. Proposal 2 (re-decode) is DECLINED: the rollout texts do
not depend on the store join, so a fresh decode changes the draw, loses the banked per-rollout
WERs, costs a GPU pass, and buys nothing. Proposal 3 (full-set rank-only as the primary) is
DECLINED: its measurand is dominated by a feasibility artifact common to both arms and
confounded with candidate length; it stays the descriptive column it already is. The
failed-closed artifacts stay banked as the record (log verdict 83). REQUIRED before the verdict
is read: the bed-feasibility statistics quoted in verdict 83 / approach 38 (146.8 / 585.7,
ratio 3.99, characters per frame, the crude bound) have no registered producer -- register a
small reader job that prints its convention and point the approach-38 feasibility paragraph at
it (the planner's verification read reproduced the headline numbers, but verdict-quoted
aggregates need a reproducible producer; standing rule of 2026-08-22). Context that MUST print
beside the eventual verdict for the USER: under the standing d_min >= 2 topology NEITHER psi arm
can score the operative quarter-rate G-track stream (81.5 pct of rows at exactly zero
probability), so psi-family selection reading that stream directly is structurally unavailable
as pinned and any deployment reading of D8.3 must have the scorer read the 50 Hz stream, as this
measurement now does. d_min >= 2 itself is standing by the user's ruling and is not revisited
here; the interaction is recorded, not reopened.

2026-08-23 verdict (D8.4 READ COMPLETE on the re-pinned bed: INDISTINGUISHABLE, resolving to the
CONTROL; the phase-closure question goes to the USER). The re-pinned primary pair
(`PsiAlignRerankJob.GNOktIsG251m` / `.JSZvokFxjNkJ` on the 50 Hz enc50 join) came back with ZERO
infeasible rows and ZERO dropped groups in both arms, the guard passed at 512 of 512 shared
groups, and `D8EtaReadJob.KwmHTXqiJMGr` finished. Operating point: operative theta_0^G rollouts
(`J9yA1eYnxwYA`), 512 utterances, G=12, T=0.7, both scorers fixed-final differing in `model_pt`
alone, `bootstrap_delta_eta` at `n_boot=10000`/`seed=42`. THE MEASURED NUMBER: paired delta eta
-0.0293 with 95 pct CI [-0.0697, +0.0085] (candidate eta +0.4220, control +0.4513); in the
plain-WER identity, selection WER 0.1417 candidate vs 0.1400 control over shared oracle headroom
0.0600. The CI includes zero, so per the pre-registered three-way reading the verdict is
INDISTINGUISHABLE and resolves to the CONTROL under the incumbent-tie rule. Context, reported and
never gating: only 6.4 pct of bootstrap replicates favour the candidate; the paired spearman
delta is -0.0114 [-0.0234, -0.0004] (a context statistic, not the verdict statistic -- the
registered reading is delta eta and is not revisited after the result); the fork-bed context
column is likewise indistinguishable (-0.0033 [-0.0164, +0.0096]); arm-internal nulls are sane in
both arms (length-only null eta strongly negative, audio-free null margin positive). The
verifier recomputed the eta identity from the reader's own JSON and the reader's printed text
carries the registered rule verbatim. Deployment context printed beside the verdict as required:
psi-family scoring of the operative policy reads the 50 Hz stream; the quarter-rate G-track
stream is structurally unscoreable under the standing topology. PLANNER RECOMMENDATION TO THE
USER (decision is the user's under the standing closure rule): do NOT fund the D8.3 policy leg
and close D8 with the control retained -- the direct ranking measurement finds no candidate
advantage at full coverage on the operative policy's own rollouts (point estimate worse, 94 pct
of bootstrap mass negative), consistent with D8.2's localization that the refit improved fit and
insertion pricing but not discrimination; a 960 h policy leg would spend GPU on a scorer with no
measured ranking edge. One bookkeeping item stays open and does not block the verdict: the
feasibility producer's printed figures and the approach-38 paragraph quotes need a true-up
(Verifier feedback, same date).

2026-08-23 CLOSED on the user's word (the message funding D9 accepted the planner's bundled
recommendation: "register D9 into plan and let implementer work on it"). Control retained,
D8.3 not funded. The verdict stands as scoped: INDISTINGUISHABLE at the cold operating point,
full coverage, incumbent kept by the tie rule. The evolved-operating-point question is a NEW
registration (D9, below) and not a reopening -- reopening would re-scope a discharged gate.
The bookkeeping true-up stays tracked in Verifier feedback and does not block closure.

**D9 -- scorer refit at an evolved operating point (three-arm paired eta read;
USER-funded 2026-08-23).**

**Purpose.** D8 measured refit value only at the cold operating point (frozen theta_0^G decodes,
before any loop improvement) and returned INDISTINGUISHABLE; every failure reading in this
family is registered as operating-point-scoped. D9 asks the same question where the policy has
demonstrably improved through the loop: do a hard-label 1-best refit and a posterior-weighted
soft-EM refit, each trained on the evolved policy's OWN decodes, rank the evolved policy's own
rollouts better than the incumbent scorer that shaped its rewards? Context that motivates but
does not decide: D6-PERIODIC's acceptance gate kept the incumbent every round (`SAE_3E1.md`
verdict 41 -- with one round-4 CI pass overridden by the stop rule, so the evolved-point
evidence has a crack), GAN-family refresh was transient-then-worse (verdicts 68-69), and the
paired eta currency on one shared draw has never been read at an evolved point. The 17.61/22.66
GAN-FROZEN endpoint is a DEGRADATION terminus, not a plateau, and the scorer-mismatch
explanation of it is currently disfavored by verdicts 68-69 -- D9 is registered as a refit-value
measurement in eta currency, never as a rescue of the degrading loop family.

**Approach.**
- PINNED POLICY: the D3 shaped control arm's sub-epoch-2 endpoint --
  `work/i6_core/returnn/training/ReturnnTrainingJob.rJWSC5xOsrf2` `output/models/epoch.002.pt`
  (on disk, verified 2026-08-23), the point whose banked recog is 12.68/17.57 (`SAE_3E1.md`
  approach 9 table). Pinned by SCHEDULE POSITION (the arm's improving phase; its later
  degradation is disclosed), never re-selected by scanning WERs. Before any spend the
  implementer asserts the epoch-to-recog provenance from the recog job's own inputs, not from
  a label list.
- ARM 1 (incumbent control): the frozen d2_contrast scorer that actually shaped this arm's
  rewards, `PsiAlignTrainJob.DnBJxqz4sNQZ`, d_min=1 AS TRAINED. Disclosed asymmetry against
  the refits (standing d_min>=2 rule applies to every NEW scorer): the incumbent rides as it
  rode the loop, because the question is "beat the scorer that made this policy".
- REFIT CORPUS: current decodes of the pinned checkpoint ONLY -- never init-round or replay
  data (standing no-replay rule). Transcripts appear nowhere in refit or selection; labels
  enter only the eta evaluation.
- ARM 2 (1-best refit): the incumbent refit recipe, d_min=2, from scratch on the pinned
  checkpoint's greedy 1-best decodes.
- ARM 3 (soft-EM refit): D8's registered recipe transplanted, constants BY REFERENCE to the
  D8 Approach above (support = deduplicated union of the pinned checkpoint's greedy 1-best
  and 12 rollouts at T=0.7 after the registered reader normalization; detached tempered
  posterior from the arm-1 incumbent and the registered text-LM prior at D8's registered
  tau). No constant is re-derived here.
- BED AND FRAME otherwise identical to D8: 960 h HF/Ogg bed (281,241 utterances), frozen
  enc50 K=500 raw 50 Hz unit store, seed-42 5% ID holdout with role-local reads; fresh
  alignment scoring joins each scorer's native-rate stream (D8.0 clause (a) as scoped
  2026-08-23). The implementer flags any knob where the D8 frame and the new checkpoint
  conflict rather than resolving it silently.

**Experiments.** D9.0 -- feasibility LAUNCH GATE before any refit spend (scope amended by
replacement 2026-08-23, implementer flag: arms 2-3 are built by D9.1, so a three-arm
finite-score census cannot exist at gate time): one rollout dump from the pinned checkpoint at
the registered operating point (group 12, T=0.7) on a read bed sized to D8.4's read for
comparability; a `D8BedFeasibilityJob`-analog prints, naming its populations, (a) the
INCUMBENT's finite-score census under its own d_min=1 topology and (b) the STRUCTURAL
d_min>=2 alignability census of every group member -- a property of rollout length vs
phone-sequence length under min-duration and of the units join, independent of any refit's
learned weights and therefore identical for arms 2 and 3 by construction. The gate passes or
fails on (a) and (b); this is what catches a D8.4-style bed failure before refit GPU is
spent. The read-set rule stays pre-registered here and is APPLIED at D9.2, where all three
arms exist: the read set is the groups where all three arms score every member finitely,
per-arm drop counts printed (impossible scores poison means) -- and a row that is
structurally alignable under (b) yet scored non-finite by a refit is a STOP surfaced to the
planner, never a silent drop, because (b) predicts the refit census by construction. D9.1 --
the two refits on the shared corpus. D9.2 -- the three-arm read on the ONE shared draw:
per-group eta, paired per-group delta eta, `bootstrap_delta_eta` n_boot 10000 seed 42 --
D8.4 machinery verbatim. Contrasts: each refit vs arm 1 (decisional); soft-EM vs 1-best
(attribution only, never adoption). The reporting rule lives in the producing job's docstring
before any result exists.

**Gate (pre-registered).** A refit arm is adopted only if its paired delta eta against the
incumbent has a 95 pct CI excluding zero in the refit's favor; INDISTINGUISHABLE resolves to
the incumbent under the standing tie rule. SUCCESS funds a policy-leg follow-up as a separate
decision -- this read gates spend and closes nothing at deployment level; plain WER on a
funded leg is the deployment currency. FAILURE licenses "scorer refitting is not funded on
this loop family at cold or evolved operating points" (jointly with D8.4 and D6-PERIODIC),
never "refitting could not work elsewhere". A GAN960-FROZEN replication is conditional on
that arm passing its own leg-8 gate and is a separate decision.

**Status.** REGISTERED AND FUNDED 2026-08-23 on the user's word ("register D9 into plan and
let implementer work on it"); with the implementer to build, D9.0 first. Planner constants
the user may override: the schedule-pinned checkpoint, the incumbent-as-trained d_min
asymmetry, the read-bed sizing at D8.4 parity.
2026-08-23 later: PRE-SPEND PROVENANCE DISCHARGED AND ACCEPTED (planner spot-checked the
chain's endpoints on disk): `epoch.002.pt` is the exact `grpo_checkpoint` of
`ExtractAvSubmodelJob.FSYsyEJm5VHX` (the recog consumes the extracted AV submodel, not the
raw training checkpoint), and `ScliteJob.paK5JVk5SckU` / `.KTVFso7HriMn` print 12.68 / 17.57
from their own `output/wer`. Arm identity confirmed from the training job's own inputs
(`PsiAlignTrainJob.DnBJxqz4sNQZ` in the INPUT list; alias
`..._shaped_T0.7_lr2e5_psid2_contrast/training`) -- the pin is the d2_contrast-shaped arm,
distinct from the adjacent incumbent-shaped arm at 13.91/18.91. The D9.0 scoping conflict the
implementer flagged is RULED in the Experiments clause above (amended by replacement): the
gate decides on the incumbent census plus the scorer-independent structural d_min>=2 census;
the three-arm read-set rule applies at D9.2 with the structural census as its by-construction
predictor and any violation a STOP. The pinned training job's `hold` file is noted and is not
a blocker (D9 reads a written checkpoint only). D9.0 build may proceed.
2026-08-23 latest: D9.0 COMPLETE, GATE PASS, VERIFIED -- D9.1 REFIT SPEND IS AUTHORIZED
(`D9FeasibilityJob.oabVIcp22cy1`; dump `ReturnnForwardJobV2.t4sIOlpGVDcY` at D8.4's exact read
size, incumbent rerank `PsiAlignRerankJob.cysJQBiP9iW1`). Census (a): 7,168 rows, 0 infeasible,
0 groups dropped, taken from the rerank job's own report (which also prints 0 of 512 groups
inside psi_align's own training set). Census (b): structural d_min>=2 alignability 6,144 of
6,144 rollout rows, 512 of 512 groups, exact DP `_min_frames` bound at the refit topology
(median 695 frames against 210 needed) -- the opposite of D8.4's bed, which lost 81.5 pct of
rows. Three implementer constants RATIFIED: (i) the numeric PASS bar (rollout share >= 0.95,
every group >= 2 alignable members, full units coverage) as the operative reading of "passes
or fails on (a) and (b)" -- at share 1.0000 every admissible bar agrees, so the choice is
immaterial by measurement; (ii) the STOCK Qwen donor, traced to the pinned checkpoint's own
INPUT list (no `ExportHfLmDirJob`), not the D4' adapted donor; (iii) the tc100 read bed as
D8.4's established read frame -- the registration's 960 h bed names the refit frame. The gate
says the bed can carry the read and nothing about eta. D9.1 builds the two refits per the
registration; D9.2 constants unchanged.

Replaces the §3e.1 two-sided gate (2026-08-07) BEFORE any verdict was read against v1, because
v1's second arm is gold-conditioned as instrumented (fact 2), has the wrong sign against the
filler mode (fact 3), and its first arm is not comparable across rounds (fact 4). All statistics
label-free, on the FROZEN external held pair set, on utterances outside the candidate's curated
pairs:

  (i)   held unit NLL improves vs the last ACCEPTED scorer (same frozen set every round), AND
        stays below the unit-marginal floor on that set (floor clause added 2026-08-07, before
        any verdict against v2: the replay trajectory shows a contrast statistic RISING while
        CE_true crossed the unit marginal and uniform — drift, not blindness, is the observed
        failure mode, `SAE_3E1.md` c1-2, and only an absolute fit floor catches it).
        AMENDED 2026-08-07 (after the D1 held table was read, before any accept/reject decision
        under v2 — flagged for the user's blessing): the IMPROVEMENT clause binds only between
        candidates sharing their training-text domain. The frozen held text is the §1d decoder's
        own dev output, so held ce_loo orders scorers by domain match, not quality (2.72 / 3.02 /
        3.14 across tc100 / seed / gold at matched family) — as written, (i) would structurally
        reject every repaired-text candidate against the unrepaired incumbent. For candidates
        whose training text changed (all of D2), (i) is FLOOR-ONLY; their quality comparison
        moves to (iii)-(v) plus the D0-dump rerank reads (bias beta, steerable coverage, paired
        rank stability);
  (ii)  text_explained_loo >= the pre-loop floor (still necessary: catches text-blindness);
  (iii) filler probes not worse vs last accepted: delta_filler and the posterior frame mass
        claimed by G2P(suspect set) must not degrade;
  (iv)  corruption-ladder spearman not decreased;
  (v)   rank stability: `PsiAlignPairedCompareJob` paired spearman vs the last accepted scorer
        on a fixed rollout dump, above a pre-registered floor (no silent reward flips at equal
        NLL);

plus `PsiScorerParityJob` before any accepted state scores a live reward. The gold-text G1
remains a REPORTED diagnostic and can never flip a decision in the G-track.

## Residual risks (accepted, monitored)

- Positional/collocational contamination passes unigram rate-matching; probes cover only
  enumerated suspects. Monitor: per-sub-epoch insertion-histogram drift.
- No clean selector view exists in the G-track (audio units are psi's own currency; external LM
  text is audio-free and filler-positive; other decodes share the student). Two-view agreement
  is mitigation, not proof; D0(e) decides whether it clears the covariance bar.
- Zero-variance shared defects stay unsteerable by GRPO under ANY scorer (dead-band); measured
  2026-08-07: coverage 23.3% ("to") / 9.2% (suspect-wide) — a partial dead-band, so the
  sampling-only sweep runs as a co-requirement beside D2, and only its failure escalates to
  the init.
- A within-sub-epoch exploit can outrun round-boundary gates (shaped's first launch died in
  4.5 h); per-sub-epoch monitors (filler mass, per-term within-group std, insertion histogram)
  carry kill authority between rounds.
