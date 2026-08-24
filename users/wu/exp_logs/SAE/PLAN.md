# SAE — Speech AutoEncoder: Unsupervised ASR via a Text Bottleneck

Reconstruction-through-text unsupervised ASR, structurally following the NLA training loop
(transformer-circuits.pub/2026/nla): an **AV** (audio verbalizer, speech→text policy) and an **AR**
(audio reconstructor, text→speech-unit channel model) trained jointly — AR by supervised CE on AV
samples, AV by GRPO against the AR's reconstruction likelihood. The pair is an autoencoder over
speech whose bottleneck is a grapheme transcript; the reconstruction score is an exact discrete
likelihood, so the AV-optimal policy is amortized noisy-channel decoding. In the adopted live
`psi_align` system the channel conditions on the scorer's own orthographic BPE states, not G2P:
z_hat = argmax_z p_LM(z) * p_psi(u | BPE_states(z)). G2P survives only in evaluation and probes.

> Restructured 2026-08-07 (planner): this file holds live decisions, gates, and one status
> snapshot; results and history live in the SAE_*.md logs; `PLAN_3A.md` is the normative psi_align
> sub-plan. Every sub-phase carries the same five fields — Purpose / Approach / Experiments /
> Gate / Status.

## North star & hard constraints

- **North star (user ruling 2026-08-01).** Real unsupervised ASR with the **autoencoder as the
  single main mechanism**. GAN-based initialization is not the goal: an adversarial init as the
  load-bearing mechanism would demote the autoencoder to a refiner. The mainline initialization
  question is §1e (pairing-free); GAN/§1d is the working label-free fallback init (§3d hierarchy).
- **Label quarantine.** True transcripts appear in exactly three quarantined places: evaluation
  metrics (PER/WER, probes, gate measurements on dev), the §0c architecture toplines, and the §2S
  anchor arm (1 h/10 h paired seeds; its artifacts never feed the unsupervised ladder). In the
  unsupervised arm no training signal, checkpoint selection, or hyperparameter choice may depend
  on them; checkpoint selection uses dev reward + LM score only. Disclosed exception (NLA-style):
  loop mechanics and lambda ranges may be developed on §2S and reused. Amendment (USER
  2026-08-14, strengthened USER 2026-08-16 — replaces the trigger-gated form): speaker IDs,
  previously never-train (2026-07-16 ruling), MAY train and may be tried first-line;
  disclosed as supervision cost; transcripts and alignments stay absolute. Tier menu:
  `PLAN_3G.md` Z3.
- **Independence rule (GAN is not a teacher).** Admissible AR targets are *measurements of the
  audio* (deterministic transforms of encoder states), never another model's hypotheses. Passing
  the label rule does not make a target admissible. One explicit, bounded carve-out (user,
  2026-08-03): GAN/§1d output as *initialization only* in the G-track (§3d) — never as in-loop
  teacher, reward, or selection signal.
- **Framing: usability, not superiority.** Matching at lower supervision cost is the win;
  pre-register non-inferiority margins; count circularity as a cost. "Unpaired" = no paired
  audio–text; Qwen3's pretraining almost surely contains the Gutenberg books underlying
  LibriSpeech — disclosed, controlled (§4), never hidden.
- **Evaluation discipline (USER rulings 2026-08-18 and 2026-08-23).** Every model-evaluation
  comparison uses PAIRED data: both arms score the same items and the read is per-item paired
  deltas with a resampled/clustered CI, never two pooled numbers. Constructed clause batteries
  (corruption ladders, proxy discrimination statistics) may gate spend inside a phase but
  never close a phase: a phase-closing better-or-worse verdict requires the direct measurement
  of the real target quantity -- ranking quality eta (equal, on shared groups, to the paired
  selection-WER delta over the shared oracle headroom) for scorers, plain WER for policies --
  in a fair paired comparison, and the closure decision then rests with the user.

## Status & priority queue (current read 2026-08-20)

**Where we are.** psi_align (§3a, `PLAN_3A.md`) is the adopted reconstruction scorer — frozen within
each policy leg (periodic arms refit it only between legs; sha-verified, `SAE_3A.md` §6.10). Loop
checkpoints below use fixed or label-free pins, but the strongest arms are semi-supervised: their AV
initialization and scorer use the 2,849-pair 10 h seed. The adapted donor was chosen by transcript-dev
perplexity, observationally equal to its fixed final checkpoint because the curve was monotone; it is
therefore disclosed and treated as fixed-final, not evidence that the whole stack is label-free-selected.

- **10 h seed bed** (§6.5/§6.9): psi_align beats the token-LM AR on all four sets at equal
  supervision (last-epoch −3.4/−3.2/−2.8/−2.2); extended run final: shaped 9.59/12.84 dev.
- **100 h bed is the BEST bed** (§6.9/§6.10): shaped monotone to 8/8, final **6.06/10.31 dev,
  6.33/10.84 test** — same init/scorer as 10 h, nothing added but unlabeled audio. The 2S-era
  off-seed collapse was a *reward* problem, not a bed problem. Insertions FALL with no length
  term (the topology prices length natively). `recon` turns at ep2: off-seed the LM prior is the
  difference between converging and turning, and its share grows with bed size — lam values are
  per-bed, never carried over.
- **G-track (train-clean-100 pseudo-SFT init; 960 h loop bed): no durable loop gain yet.** The
  original `recon` arm diverged through an inherited filler and `shaped` plateaued then slipped
  (§6.7/§6.10): init and scorer share the §1d pseudo-text, so correlated errors are rewarded rather
  than caught. The repaired frozen-scorer arm's best matched point is 12.68/17.57 from the
  13.89/18.34 no-loop init, but it returns to 13.54/18.56 one sub-epoch later. 2026-08-22 (replaces the
  six-leg-prefix and control-running sentences — both arms completed): the D6-PERIODIC/GAN
  recency A/B is DECIDED against refresh: the frozen-scorer control wins the final leg on both
  splits (17.61/22.66 vs periodic 18.82/24.56, verifier-confirmed), the periodic lead at legs
  2-4 is the registered non-licensing transient, and — the larger fact — BOTH arms degrade
  after leg 3 with neither leg-8 endpoint beating the no-loop init 13.89/18.34
  (`PLAN_3E1.md` GAN-FROZEN Status, `SAE_3E1.md` verdicts 68-69). Both §3d.A reads are in (2026-08-21, replacing "the
  decisive read is now §3d.A"): one-generation own-label self-training FAILED both starts, while
  the 960 h pseudo-label scale arm PASSED its gate — theta_0^G960 reads 13.11/16.82 against the
  13.89/18.34 init, the project's best label-free AV start (verifier-confirmed;
  `SAE_3D_GTRACK.md` A5). The superseded offline D7-v2 specification CLOSED
  at its prospective D7.0b structural gate. The label-free feature read found 136,966 Q2 training
  edges, but the exact
  common same/different-chapter 2-in/2-out construction admitted only 56 rows/two speakers against
  the frozen 6,778/201 floor. An independent necessary-core replay upper-bounds every feasible
  solution by 120 rows/four speakers. No external assignment, loss calibration, scorer or policy
  ran. This is a failure of the registered K=4/Q2/common-regular operating point, not a measurement
  of the reverse loss; no graph amendment or retry is authorized. USER 2026-08-21 corrects active
  D7-GAN-SEQDISC to one full-960 h scorer A/B using K=1 online random
  same-speaker, duration-windowed donors, with no graph, chapter/Q2 band, nuisance filters or donor
  capacity. D7.0/D7.1 are IMPLEMENTED AND LAUNCHED 2026-08-21 (verifier-confirmed same day: decode
  shards running, scorer A/B gated behind the preflight barrier, no funded job displaced); a policy
  leg still needs authorization. Same day latest: preflight PASSED under the amended parity rule;
  the first D7.1 run failed closed on four degenerate own-infeasible greedy anchors — drop-and-count
  amendment registered (`PLAN_3E1.md` D7 Status); 2026-08-22 the edit is implemented and
  verifier-confirmed (speech-llm `e2a421b`, hashes unmoved), the user ran the restart 22:50, and
  D7.1 COMPLETED 23:05 on both arms and is verifier-confirmed — four named drops closed per-arm,
  arms single-variable at the artifact level, fixed-final scorers banked. 2026-08-22 later:
  D7.2 COMPLETED and FAILED clause 2 (candidate held NLL 2.531898 vs control 2.525882; clauses
  1 and 4 pass, clause 3 no winner) — **D7 IS CLOSED** per the registered gate,
  verifier-confirmed against the artifacts: no policy leg, no rescue selected from the result;
  the recorded legacy is verdict 67's trade (decisive same-speaker discrimination, but a
  significantly larger insertion discount). The D8.1a-b release this verdict unlocks is ruled
  in `PLAN_3E1.md` D8 Status. 2026-08-23: D8.1a read GO on the corrected support
  (verifier-recomputed to the last digit; convention immaterial by the pre-registered
  sensitivity read) with `candidate_acoustic` the ONLY funded arm — shaped and acoustic-only
  weights are operationally identical (spearman 0.9835 > 0.95) — and D8.1b is authorized for
  that arm alone (`PLAN_3E1.md` D8 Status 2026-08-23). 2026-08-23 later: D8.1b trained at
  measured cost parity and D8.2 read its gate — clause 1 passes below even the strict zero
  margin, but clauses 2-3 fail (no discrimination gain on any corruption ladder,
  filler-insertion significantly degraded, gate v2 NO WINNER) — so **D8 IS CLOSED without a
  policy leg** per the registered no-rescue gate, verifier-confirmed bit-exactly from the
  per-anchor evidence. The banked legacy: posterior-weighted refitting improves fit and
  insertion pricing, not discrimination (`PLAN_3E1.md` D8 Status 2026-08-23 latest).
  2026-08-23 later: the USER OVERRULES the closure -- **D8 IS REOPENED** for the D8.4 paired
  ranking-quality (eta) read: both fixed-final scorers rerank the same banked rollout groups
  and the verdict is the paired delta eta with the registered bootstrap (spec in `PLAN_3E1.md`
  D8 Status; two new standing evaluation rules under North star & hard constraints). No
  operating point is selected from the failed tables; the still-running parity job is read
  first as the possible vehicle. 2026-08-23 closing: D8.4 FAILED CLOSED on the operative bed
  as registered -- the quarter-rate G-track units join leaves 81.5 pct of rollout rows
  unalignable under d_min >= 2, identically in both arms -- so the primary pair's units join is
  re-pinned to the 50 Hz enc50 stream both scorers train against (same dump, same draw; ruling
  and mechanism in `PLAN_3E1.md` D8 Status 2026-08-23 closing; fork-bed context column is
  banked and INDISTINGUISHABLE at -0.0033 [-0.0164, +0.0096]). 2026-08-23 verdict: the
  re-pinned read COMPLETED at full 512-group coverage -- paired delta eta -0.0293
  [-0.0697, +0.0085], INDISTINGUISHABLE, resolving to the CONTROL under the incumbent-tie
  rule. 2026-08-23 latest: **D8 IS CLOSED on the user's word** (control retained, D8.3 not
  funded; the D9-funding message accepted the bundled recommendation). **D9 IS REGISTERED AND
  FUNDED** (`PLAN_3E1.md` D9): the same refit question at an EVOLVED operating point -- three
  arms (frozen d2_contrast incumbent `PsiAlignTrainJob.DnBJxqz4sNQZ` / 1-best refit / soft-EM
  refit, both refits on the pinned checkpoint's own decodes only) ranking one shared rollout
  draw from the D3 shaped arm's sub-epoch-2 endpoint `ReturnnTrainingJob.rJWSC5xOsrf2`
  epoch.002.pt (banked 12.68/17.57), paired delta eta with the D8.4 machinery verbatim;
  feasibility gate D9.0 before any refit spend; with the implementer to build. 2026-08-23
  latest: D9.0 PASS, verified (`D9FeasibilityJob.oabVIcp22cy1`: incumbent census 7,168 rows /
  0 infeasible; structural d_min>=2 alignability 6,144/6,144 rollout rows, 512/512 groups,
  median 695 frames vs 210 needed -- the opposite of D8.4's bed) -- D9.1 refits AUTHORIZED,
  with the implementer. 2026-08-24 (verified; ruling in `PLAN_3E1.md` D9 Status): ARM 3 (soft-EM)
  IS NOT FUNDED -- `D9WeightJob.uyKXr4ZiGj9R` rules NO-GO on clause (a), median 2.0 distinct
  support strings of 13 candidates (33.1 pct of the 281,241 groups collapse to ONE), a measured
  mode-collapse finding about the pinned policy's sampling, not an instrument artifact. D9.2 is
  amended by replacement to the TWO-ARM read (1-best refit vs incumbent, D8.4 machinery
  verbatim); the threshold edit and a diversity re-dump are both rejected/not funded. Arm 2 is
  training; D9.2 registration next. 2026-08-24 later (verified; `PLAN_3E1.md` D9 Status): D9.2
  COMPLETE -- paired delta eta -0.0310 [-0.1545, +0.0923], INDISTINGUISHABLE resolving to the
  incumbent, arm 2 NOT adopted (verdict 86; power caveat: this bed's headroom is a fifth of
  D8.4's, so the interval is three times wider). D9's registered reads are all banked and
  nothing runs. AWAITS THE USER'S WORD: close D9 under the joint license ("scorer refitting is
  not funded on this loop family at cold or evolved operating points") and decline the
  D8.3-style policy-leg assay -- the planner's recommendation, with the evolved policy's
  sampling collapse (verdict 85) banked as the family's newest fact.
- **960 h stock-donor supervision-axis endpoint is ABSENT**: the theta_0 + gold-scorer arm ran
  only through sub-epoch 4, was stopped and deleted 2026-08-08, and never produced the listed
  3-pass endpoint; `ReturnnTrainingJob.22Ntu7y0O6iW` does not exist. Its observed collapse is
  retained in `SAE_3A.md`, but any restart is a new decision and is not the current critical path.
- **§0d donor swap VERIFIED 2026-08-08**: the LBS-adapted donor's theta_0' re-SFT alone is
  11.43/15.54 dev, 11.99/14.34 test vs stock 16.91/20.64, 15.28/20.78 (one-argument A/B) —
  better than anything any loop earned from stock theta_0. Both `_lbslm` loop arms are FINISHED:
  the 100 h fixed-last pin is 5.22/9.67 dev; the 960 h one-pass arm ends 6.46/11.41 dev and
  6.94/12.16 test after a transient 5.34/9.50 at sub-epoch 2. Gate (ii) still awaits the user's
  blessing and its small reward-margin claims require the registered EOS-consistent re-score;
  these caveats do not affect the direct SFT or WER endpoints.
- 2S incumbent reward program: superseded by psi_align; history in `SAE_2S*.md`.
- Reward hygiene: the LM-prior per-token mean pays for length (§6.6) — `lm_prior_norm="units"`
  is the standing fix; `len_eps` 0.4 leaves a 49 % free band if the hinge is ever load-bearing.

**Priority queue (current revision, 2026-08-21):**
1. **§1g H4 — TOP NEW SPEND.** Preserve the verified 821-job prerequisite graph: all 85-by-4 channel
   tables, direct-`Q` starts, role-local selection donors, and update/selection decoder resource
   contracts are complete. No selection decode, normalized score, selector, final refit, or evaluation
   has run. Add the bounded global-beam stability extension and deterministic Section-4 aggregation/
   selector boundary. Before controlled labels open, persist the prospective reference/four-H3
   provisional maxima and winner audits; then validate own-minus-donor and freeze those maxima
   unchanged. Final-refit the H3 rows on 7,304 construction IDs and the reference on 4,455 dev IDs,
   pass the release checks, and open the 1,112-ID evaluation once. Likelihood is update-
   health evidence only, never a selector, fallback, or tiebreaker. The workspace has
   `JOB_AUTO_CLEANUP = True`; verify the effective imported value before any new manager.
   2026-08-22: the global-beam boundary ruled ALL 12 grid points ineligible (baseline surface is
   local-decoder-only), the pre-label selection surfaces are COMPLETE and verifier-confirmed
   (340 local decodes, 3,400 donor scores, 85 provisional maxima persisted, every winner local —
   the winner beam audit is discharged by the registered local-winner exemption), and the
   planner has ruled the controlled reference labels OPEN (`PLAN_1G.md` Status 2026-08-22, with
   two banked observations the validation read must respect: 76 effective independent controls
   of 81, and the pre-label cross-start ordering ranks the random-map null above the reference).
   2026-08-22 (user): the PER of this approach is wanted. No label PER exists yet by design (the
   E5 rehearsal endpoints are engineering-only); the 1g.2 controlled validation read is clear to
   START NOW — D8.1a babysitting does not block it — and the one-shot 1,112-ID evaluation PER
   follows the refits and release checks in the registered order.
   2026-08-22 LATER — THE 1g.2 GATE FIRED NEGATIVE (verified; `PLAN_1G.md` 1g.2 Status): the
   own-minus-donor selector is inverted (reference loses to the strongest content-free control
   by 5.02, all correlation upper bounds below zero), while the count safety read passes (repair
   itself is safe; the choosing score is what failed). H4 is unresolved, maxima frozen, refits
   and the 1,112-ID evaluation CLOSED — no PER of the approach can exist on this route. Phase 1g
   HOLDS for the user's direction word: close the phone-repair route / fund a new selector
   science with fresh controls / amend to open the lexicon-free character route.
   2026-08-22 (user): a DESCRIPTIVE dev PER read on the four real seeds is FUNDED
   (labels-as-evaluation-only over the closed gate; ruling in `PLAN_1G.md` 1g.2 Status): plain
   per-split PER, all four repair counts, on the frozen 432/458 selection-role decodes; the
   1,112-ID held-out evaluation stays sealed; selects and funds nothing.
   2026-08-22 LATER (user): trigram/4-gram fitting context is MANDATORY — H4-LM (1g.2a) is
   FUNDED at D scope (engine + matched 2/3/4 artifacts + resource gate + five-start diagnostic;
   implementation ruling in `PLAN_1G.md` 1g.2a Status). The selector route stays closed. SAE
   init from the best-PER pseudo-pair row is recommended AGAINST (output audit in `PLAN_1G.md`
   1g.2 Status: deletion-dominated collapsed outputs, margins are unigram-level only).
   2026-08-22 LATEST (user): the 1g.9 anti-collapse constrained-repair probe is GREENLIT at
   HIGHEST priority — the locate-the-collapse diagnostic runs first and alone, the constrained
   refits only past its clause-0 off-ramp; spec and pre-registered gate in `PLAN_1G.md` 1g.9.
   The D8.1a ruling execution (piece 3, `PLAN_3E1.md` D8 Status) continues as the next spend.
   2026-08-22 LATER: 1g.9 CLOSED by its own clause-0 off-ramp (verifier-confirmed; ruling in
   `PLAN_1G.md` 1g.9 Status): every start's training posterior already satisfies both proposed
   constraints (total variation 0.012-0.074 vs 0.15, rate within 5.5 % vs 20 %), so no
   constrained arm runs; the audited babble is decode-resident and specific to the pseudo-pair
   start under the LM-blind frozen local decoder. The direction fork — close the phone-repair
   route (planner's recommendation) or fund a bounded descriptive full-model-decode follow-up —
   awaits the USER's word.
   2026-08-23 (USER): the fork is RESOLVED for the decode route — the user, surprised the LM
   was never in the production decode, directs that it be used: 1g.10 (bounded descriptive
   full-model sequence decode of the audited count-4 channels with the LM and duration law in
   the decoder; beam instability reported and explained by measurement, not used as an
   eligibility bar) is REGISTERED in `PLAN_1G.md`; the label-free selection surface stays
   local-only and closed. 2026-08-23 later: the espum start's count-4 cells are promoted into
   1g.10 experiment (1) on the USER's PUSM question; the fairseq-side companion is
   `PLAN_1F.md` entry 8. 2026-08-23 result: 1g.10 COMPLETED and its table is BLOCKED by its own
   pre-registered explanation duty -- adjacent beams disagree (median agreement 0.61 of 1) while
   score margins are wide, the decoder-defect branch, so no cell is read. The 1g.10a cross-beam
   defect diagnostic is REGISTERED on banked data (scoring-determinism and pruning-monotonicity
   tests, pre-registered consequences, `PLAN_1G.md` 1g.10 Status 2026-08-23 result); the route
   question stays with the USER. 2026-08-23 discharge: 1g.10a ran under the re-ruled invariants
   (the banked score is a pruned path-sum, so the original equality test was void) and
   DISCHARGED the suspicion -- zero violations in the determinism and exact-upper-bound tests,
   and the wider beam finds the genuinely better sequence in 352 of 384 disagreements. The
   beam-512 table is now READABLE AS DESCRIPTIVE with per-cell 256-vs-512 agreement disclosed;
   channel-vs-channel comparisons wait for the registered 1g.10b beam-1024 probe (26-of-27
   agreement bar). Route decision remains the USER's. 2026-08-23 (USER: "insertion bonus makes
   sense, please try that"): 1g.10c REGISTERED -- positive insertion-bonus cells (lm_scale
   {1,2} x beta {+1,+2}) on the two content-bearing channels, paired within-channel reading
   against the beta 0 boundary cells, option-(b) mechanism pre-approved (`PLAN_1G.md` 1g.10
   Status 2026-08-23 extension); with the implementer to build. 2026-08-23 (1g.10b result):
   parity PASS but the quoting bar is NOT cleared (0 of 36 cells at 26-of-27; median 512-vs-1024
   agreement 0.704, up from 0.611 at the previous doubling) -- cross-channel rankings from the
   LM-aware grid stay unquotable, further beam escalation is NOT funded (about three more
   doublings away at doubling cost), and within-channel paired reads remain the grid's standing
   currency. 2026-08-23 (1g.10c result): parity PASS, the two rows SPLIT BY SIGN -- the positive
   control recovers phones at every extension point (+0.0222 to +0.0555 paired correct-phone
   delta, intervals excluding zero) while the real ESPUM arm loses them at three of four and
   straddles zero at the fourth; the bonus buys length on both rows, content only on the control.
   1g.10c CLOSES (`PLAN_1G.md` 1g.10 Status 2026-08-23 1g.10c result): the deletion mechanism is
   confirmed causal, the truncated-grid concern is discharged, no further decode-parameter probes
   on this harness. 2026-08-23 (entry 8 cells 1-2 result, `PLAN_1F.md` entry 8 Status): the LM
   decode kills the insertion flood as predicted (full loss 1.6828 -> 0.8444 at the label-oracle
   cell) but delivers NO usable decode -- both arms land at 0.82-0.85 by emitting half the
   reference length, deletion-dominated; the sil_weight axis measured inert and is retired; the
   registered label-free selector ANTI-selects (per-token perplexity pays for length), so every
   entry-8 quote is a (pick, oracle-best, range) triple. No null margin until cell 4 re-banks
   the nulls in this currency -- cells 3-4 remain THE USER'S WORD, stakes raised.
   2026-08-23 (USER: "I greenlight 1g 11"): 1g.11 REGISTERED AND FUNDED (`PLAN_1G.md` 1g.11) --
   the continuous-emission twin of the table channel: same topology, duration, LMs, repair and
   local readout, categorical `B(unit|phone)` swapped for tied diagonal Gaussians on
   segment-mean frozen-PCA layer-15 features (leading 128 components, variance-floored), five
   1g.2a starts at counts 0/4, paired per-utterance attribution read against each start's own
   banked table cell, babble null plus continuous observation null; claim under test is
   geometric inductive bias, not information loss; with the implementer to build.
   2026-08-24 (1g.11 gate verdict, verified; ruling in `PLAN_1G.md` 1g.11 Status): CLAUSE 3
   FAILS ON THE CONTROL -- the selected real start's paired gain over its own banked table
   cell (+0.0098 [+0.0058, +0.0137]) is exceeded by the content-free random-map Gaussian
   control (+0.0251 [+0.0202, +0.0302], non-overlapping intervals), and the positive-control
   reference start LOSES phones under the swap (-0.0208). Continuous emissions are NOT funded
   at this operating point; evidence toward the training paradigm as the binding constraint;
   the wav2vec-U-faithful follow-up is not funded. 1g.11's question is ANSWERED; the phone
   route's direction is the USER's call, alongside entry 8 cells 3-4.
   2026-08-24 (USER: "run 4gram training with 4gram LM decoding also for 1g11"): 1g.12
   REGISTERED AND FUNDED (`PLAN_1G.md` 1g.12) -- 1g.11's question re-asked at the strongest LM
   operating point the campaign owns: emission model (table / Gaussian) crossed with fitting
   order (bigram / matched 4-gram) at repair count 4 on five starts, every corner decoded by a
   NEW exact beam-free order-4 one-best readout (the beam harness is closed, an exact decode
   needs no stability duty, measured at minutes per fold over the 68-token-per-utterance
   retained stream) with each cell's banked LM-blind decode as the no-LM leg. Three of four
   corners cost a decode only -- the order-4 table channels persist as fitted artifacts -- and
   the Gaussian bigram corner re-runs 1g.11's EM (minutes) because its parameters were never
   persisted, with exact reproduction as the acceptance check. Gate has three paired contrasts
   (readout / fitting order / emission model), each against the random-map control and the
   observation null, the 1g.11 "comparable gain" ruling carried over verbatim; 1g.11's fired
   gate is not reopened. With the implementer to build, resource read first. 2026-08-24
   experiment 1 VERIFIED: resource gate PASS for one curve (4 h / 4 GiB vs the 11.5 h clamp),
   infeasible single-process, ten fitting cells in flight one job per (start, order).
   2026-08-24 (USER: same as 1g.12 but with wav2vec-U v1-equivalent segmentation, parallel):
   1g.13 REGISTERED AND FUNDED (`PLAN_1G.md` 1g.13) -- the 1g.12 factorial transported onto a
   stream segmented the wav2vec-U v1 way (banked rVAD-trimmed layer-15 features, K=128 k-means
   on raw features, segments = cluster-ID runs at their natural ~28/s rate, plain unwhitened
   PCA-512 run means; run cluster ID as the discrete twin), five starts re-derived by their
   registered procedures, gate = 1g.12's clauses plus paired contrast (d) SEGMENTATION against
   1g.12's Gaussian 4-gram cell over the shared 890. Constants traced from the real fairseq
   scripts and the banked dumps before registration; experiments 1-3 first, resource read
   before any cell. 2026-08-24 experiments 1-3 VERIFIED (stream anchors the ~28/s published
   rate; five starts transport, all numbers recomputed); 1g.12 observation-null readout seam
   RULED (`PLAN_1G.md` 1g.12 Status: the null persists its redrawn selection-fold vectors,
   readout module untouched), unblocking 1g.12 experiment 5. Experiment-4 first run superseded
   by a NaN-posterior engine defect it exposed (fixed `41127e8`, verified; banked 1g.12
   unaffected, anchored by registered `G12EngineEquivalenceJob`); re-measured gate VERIFIED
   PASS (9 h vs 11.5 clamp, one job per start) -- experiment 5 builds at that shape, first
   cell's wall clock read before the rest. 2026-08-24: 1g.12 experiment-5 null seam VERIFIED
   end to end on the finished bigram pair, both-orders null and contrast-(c) exclusion
   RATIFIED, experiment-6 reader build reviewed with the bootstrap convention RULED (all in
   `PLAN_1G.md` 1g.12 Status; reader waits on the 4-gram null cell); 1g.13 table-arm port
   verified hash-neutral and its gate RATIFIED as completing experiment 4 -- read PASS at 10 h
   vs the 11.5 clamp (verdict 69: six E-steps vs the Gaussian's five, so LESS headroom than
   the Gaussian arm) -- both arms funded, one cell of EACH arm launches first. 2026-08-24
   evening: 1g.12 experiment 5 CLOSED VERIFIED (one-bed certified in the strong form --
   count-0 decodes byte-identical across the two fitting orders, count-4 separates), reader
   `G12EvaluateJob.yJgxKex9peLp` IN FLIGHT (the first phone error rate in 1g.12, gate reads
   when it lands); 1g.13 factorial pilots VERIFIED and IN FLIGHT (requests read from each
   arm's own gate, same fold asserted, hash-neutral by census); 20:55 USER DIRECTION: the
   WHOLE factorial launched immediately, replacing the pilot-first clause -- all 20 fitting
   cells queued, verified on disk; recovery for a table-cell clamp overrun is the gate-passed
   sharded shape, not a bigger request. 21:00 1g.12 GATE READ AND RULED (verifier recomputed
   every decision number from raw hypotheses; `PLAN_1G.md` 1g.12 Status): clause 2 fails for
   every real start, clause 3 NOT POSITIVE on all three contrasts -- on (a) the observation
   null's readout gain exceeds the arm's with non-overlapping intervals -- clause 4 passes;
   the registered failure license fires verbatim, continuous emissions NOT funded at this
   operating point, nothing reopens 1g.11. All six 1g.12 experiments COMPLETE; closing the
   subphase is the USER's word; 1g.13 (in flight) carries the segmentation question.
2. **§3e.1 — D4+D5 on the best bed (USER REDIRECTS 2026-08-08/09)**: the user overrode the
   D3-plateau trigger (2026-08-08, rationale: rate-matching is a targeted heuristic, which
   D2's own read supports — d2_rate failed the paired read, d2_contrast is the conditional
   winner), then redirected BOTH phases to the best arm — the theta_0' lbs 960 h 1-pass
   shaped loop `vhyvv2waeU16` (finished; gold-seed psi, 2,849-pair train set, 281k-utt bed).
   One label-free fork checkpoint feeds THREE update-rule arms at matched remaining
   schedule: FROZEN continuation (the running arm, free) / CONTINUOUS JOINT psi (D5(b), the
   collapsed form, 4-6 sub-epochs, stop regardless) / GATED DISCRETE REFRESH (D4' —
   iterative psi refit on curated own-decodes anchored by the gold seed at 50 % floor).
   Specs + pre-registered gates in `PLAN_3E1.md` D4'/D5; D5(a) collapse forensics on the §3c
   run's existing checkpoints go first (cheap). Planner read 2026-08-09: D5(a) COMPLETE —
   the collapse is pure over-generation with the scorer's preference migrating to its own
   padded decodes, and pinned-policy eta flips negative after ONE sub-epoch (share-based
   monitors are blind; counts + in-run eta/CE_true are the mandatory instruments — folded
   into D4'/D5(b) as dated amendments); the in-flight G-track D4 round-1 curation read
   independently supports the park (curated picks dirtier than the anchor; refit finishes
   for the record only). Planner read 2026-08-10: fork PINNED at sub-ep 2 (the count screen
   vetoed the reward's own argmax pick); D5(b)'s faithful single-knob form is INFEASIBLE on
   the node (OOM + 2.6x step time) — re-specced to a 4-of-12 psi-CE subsample with a
   pre-registered lower-bound caveat; D4' round-1 has NO admissible curation view on this
   bed by measurement (psi filler-positive at matched WER, suspect derivation empty) —
   round 1 re-specced UNCURATED (gold anchor 50 % + one greedy decode per utterance); both
   amendments dated in `PLAN_3E1.md`, user may override. Planner read 2026-08-11: the
   4-of-12 re-spec is SUPERSEDED — the implementer's batch-halving fix ran the FAITHFUL
   joint arm; one sub-epoch of co-training is the bed's best-ever WER (5.12/9.27, beating
   the matched frozen control 6.56/11.15) and the next destroys it (17.35/21.97, insertions
   ~16x) — gate verdict pending CE_true forensics and sub-ep 3, but the shape (one good step
   then a cliff) is the strongest case yet for the gated discrete refresh (D4' round 1)
   over any continuous update rule. User-directed 2026-08-11: NEW TRACK D6 (`PLAN_3E1.md`)
   — general insertion repair, three rungs (offline price steering; corruption-trained arc
   prices; min-duration topology), goal a scorer ranking as well as the incumbent without
   the insertion cheapness. Planner read 2026-08-12: D5 CLOSED — collapse confirmed by its
   gate (sub-ep 3 = 41.8/50.9, CE_true monotone rising; new finding: the allegiance GAP,
   not the CE_true level, is the leading in-loop alarm). D6 read: rung 3 (min-duration
   topology, d_min=2) passes all four pre-registered clauses and is the scorer-swap
   candidate; rung 1 failed its bar, rung 2 refuted (learned its own negative
   distribution). D4' round-1 clause table NO WINNER (c33) — moot for production, the swap
   candidate is the D6 refit on the same corpus. NEXT FUNDED STEP, gated on the user's
   CI-vs-point pin: swap-in continuation (d6_mindur frozen) vs the frozen control at
   matched sub-epochs, confirmation read = the insertion regression shrinking; cheap
   parallel: d_min=3 refit through the same clause table. USER 2026-08-12: swap-in
   approved and extended to BOTH beds — best bed as registered; G-track via a
   min-duration refit of its own round-1 refresh recipe on its own corpus (topology
   transfers, checkpoints don't; spec in `PLAN_3E1.md` D6 Status). CI-vs-point blessing
   PENDING CONFIRMATION (user asked for the plain-words definition first); clause tables
   stay dual-reported until confirmed. USER 2026-08-12, new parallel front — real
   unsupervised without GAN: (a) §3g Z-track, from-scratch joint loop (LBS-SFT text donor +
   min-duration psi co-trained from zero, full D5 forensics; deliverable = failure-mode
   classification, taxonomy pre-registered in §3g); (b) §1f, statistics-matching init
   revisited (1b was never run — superseded, not refuted; two kill-condition prerequisites
   registered before any matching arm). Z-track (now `PLAN_3G.md`): base arm CLOSED (A)
   2026-08-13; Z2 actually completed all six sub-epochs despite the earlier stop directive —
   coupling ladder duration -> density, no phone-content evidence. Z3 also completed all six
   and FAILED its primary: the duration-matched gap stayed negative (-0.0137...-0.0093), the
   same duration code rebuilt more purely, and final WER was 94.87/96.14. The formal B/C
   taxonomy remains incomplete because no unit-emission purity/PER read was produced. LM-prior
   demotion was WITHDRAWN 2026-08-15 (user pushback + mechanics review — prior
   is the posterior's own term, and the code is recon-funded). Z4 (discrete psi refresh +
   within-seq repetition price + lam_len activation; lam_lm kept) REGISTERED AND FUNDED
   2026-08-15 on the user's word — build order and pre-registered gate in `PLAN_3G.md` 3g.4;
   Z3 runs untouched to its registered end as the like-for-like comparison. USER
   2026-08-14: best-bed swap-in continuation ("D4' with min duration") GREENLIT to start
   now, CI pin still pending and non-blocking (spec in `PLAN_3E1.md` D6 Status). Planner
   read 2026-08-15: that continuation is COMPLETE and passes its confirmation outright
   (4.73/9.31 vs control 6.46/11.41 at sub-epoch 10, dev-other insertions halved, 933 vs
   1964; log c39) — but the whole gain lands in the first post-swap sub-epoch and then
   plateaus. USER 2026-08-15: the periodic version REGISTERED AND FUNDED — refit the
   min-duration scorer from scratch at EVERY sub-epoch boundary on the current policy's
   decodes, per-round acceptance gate, re-forked from the same parent checkpoint so the
   finished one-refit arm is the matched control; spec in `PLAN_3E1.md` D6-PERIODIC.
   Same message, STANDING RULE: every new scorer plan carries the min-duration topology
   (d_min>=2). The
   G-track full-bed read closes
   the reward question there: ar_recon eta -0.1103 (argmax worse than random) while the
   oracle-random gap is ~6 WER points — the scorer, not group degeneracy, is the G-track
   binding defect. Requires the additive-only trainable-psi
   build (the running arm re-imports the recipe tree on resume — no executed frozen-path
   line may change). PARKED by these redirects: the G-track D4 round-1 (and with it the
   bad-init self-repair read — revive on the user's word); D3 stays parked. Still needs
   blessing: the CI-convention pin (now decides D4' round acceptance too) + gate v2 (i)
   floor-only. USER 2026-08-17: D6-PERIODIC extends to the gan-init bed —
   **D6-PERIODIC/GAN launched** (theta_0^G init, 8 rounds, per-boundary from-scratch
   d_min=2 refits on the policy's own greedy decodes; anchor-free pool and NO acceptance
   gate — both gold touchpoints deleted for label hygiene, user-directed in the
   implementer session; c37 planner-verified same day, so the one-shot G-track swap-in
   does NOT proceed and is superseded by this arm), plus a **homophone-diversity SFT
   arm** on the same bed as the one-argument A/B against it (specs, ratifications and
   pre-registered reads in `PLAN_3E1.md` D6-PERIODIC/GAN and /GAN+HOM).
   2026-08-20 VERIFIER: D6-PERIODIC/GAN-FROZEN is IMPLEMENTED AND LAUNCHED with periodic
   round 1's exact `d_min=2` scorer and segmented policy graph. Leg 1 reuses the banked periodic
   job and legs 2–8 contain no later scorer refit; leg 2 was verified running at 15:37 CEST, and
   there is no endpoint yet.
   USER 2026-08-17: 1f fork resolved — entry 5 (ESPUM statistics-matching init) FUNDED
   as one contained simplicity-constrained batch; spec pre-registered in `PLAN_1F.md`;
   BPE-level ESPUM registered as conditional follow-up on a phone-level pass.
   2026-08-17: entry 5 RAN AND FAILED THE GATE, both clauses (label-free pick 0.8580
   dev-other PER vs the 0.8446 bar; audio-swap rise 0.0466 vs 0.05, close). Health
   passed (no collapse); failure is identity, not rate. Best 1f arm to date (unary
   solve 0.8809; margins tripled) but 0.44 above the memoryless ceiling. Entry 5
   CLOSED per its gate; verdict in `PLAN_1F.md` entry-5 Status; table in SAE_1f.md.
   USER 2026-08-17 (later): ruling 6 — "try your best to make a PUSM-like approach
   work; reproduction accepted" — 1f does NOT close. Reproduce-then-bridge registered
   as entry 7 in `PLAN_1F.md`: stage A reproduces the released ESPUM stack on TIMIT
   unmatched (anchor 0.473); stage B swaps one component at a time toward our setup
   (frozen segmentation / our 500-way units / LibriSpeech bed) to localize the killer;
   stage C transplants the fix and takes the unchanged arm gate. Ruling 4's TIMIT ban
   lifted for reproduction only; simplicity yields to fidelity inside entry 7.
   Implementer: TIMIT availability check (step zero), then stage A build.
   2026-08-17 (later): step zero found NO TIMIT on the cluster; USER clarifies ruling 6
   — reproduction is APPROACH-wise, no TIMIT at all. Entry 7 amendment 1: reference
   pipeline verbatim (wav2vec2 features, k-means-128, learned segmenter, relabeling)
   on our LibriSpeech seed bed, full + bigram-only arms, signature-based read
   (bigram-only worse by >= 0.10); swaps then localize on the same bed. TIMIT returns
   only as a user option if the signature is absent.
   2026-08-19: stage A RAN AND CLOSED NOT ANSWERABLE — the signature is reversed (-0.44
   vs the +0.10 bar) but both arms sit far above the interpretability margin with flat
   audio-swap controls, a contrast between two uninformative decodes (`PLAN_1F.md` entry
   7 Status); stages B/C never built.
   2026-08-23 (USER): "maybe even old PUSM approach should be decoded with LM? I never
   saw PER from it as well" — verified: every banked 1f PER is a greedy/argmax decode
   (the phone 4-gram only ever selected checkpoints, never decoded), while the released
   wav2vec-U-family protocol's headline numbers are LM-decoded. Entry 8 (LM-decoded PER
   of the PUSM/ESPUM arms: flashlight KenLM unit-LM decode of the stage-A arms +
   CTC-student sanity control + nulls/ceiling RE-BANKED under the same decode +
   published-anchor decode pin) is REGISTERED in `PLAN_1F.md` and AWAITS THE USER'S
   LAUNCH WORD (planner recommends funding); the espum channel cells are already funded
   inside 1g.10 experiment (1).
3. **LM-prior domain adaptation (§0d) — RUN AND VERIFIED 2026-08-08** (`SAE_0d.md`; replaces
   the pre-run item because the phase executed): pre-check (i) PASSED; gate (ii) read — planner
   verdict in §0d Status **awaits the user's blessing** (margin over the audio-free null is
   statistic/lam/bed-dependent; pass proposed for theta0-bed lam=1 only, no lam=0.3, no G-track).
   theta_0' re-SFT alone beats every stock-theta_0 loop result. Both donor-axis loop reads are
   finished; §2a-rescorer and lam_1/lam_2 recalibration remain deferred behind the bootstrap-critical
   §1g assay. The adapted reward sweeps mix EOS conventions, so their small margin claims await the
   cheap registered correction; direct SFT and WER comparisons are unaffected.
4. **Pseudo-label scale and one-generation self-training (§3d.A) — BOTH QUESTIONS ANSWERED
   2026-08-21 (replaces the 2026-08-20 BLOCKED/in-flight text; decision now with the user).**
   The scale gate PASSES, verifier-confirmed end to end: theta_0^G960 (from-scratch AV SFT on §1d
   pseudo-labels for all 281,241 960 h utterances, one pass, fixed sub-epoch-10 endpoint) reads
   13.11/16.82 dev-clean/dev-other against theta_0^G's 13.89/18.34 — both splits improve, the
   project's best label-free AV start; against the 10 h-PAIRED self-training operator (13.05/17.74)
   it is a split trade-off (+0.06 clean / -0.92 other) at strictly lower supervision cost —
   usability, not superiority. The blocking packed-input preflight was root-caused (waveform/PCM16
   mismatch) and re-passed exactly (298/298) before any decode spend — never waived. The
   one-generation fresh-label gate FAILED both starts (own labels worse than the fixed §1d labels
   from either start; verdict 10); no second generation. USER 2026-08-21 (resolves the open
   decision in part): fund ONE frozen-scorer reconstruction loop from theta_0^G960 — registered
   as D6-PERIODIC/GAN960-FROZEN in `PLAN_3E1.md` (the GAN-FROZEN recipe verbatim, init swapped,
   same frozen round-1 scorer; leg-8-vs-init both-splits gate pre-registered). Same day later:
   IMPLEMENTED AND VERIFIED (planner config-diff read + implementer graph census, zero scorer
   work funded); launch awaits the user's manager start, detail in `PLAN_3E1.md`. Rebasing anything
   else (D7/D8, refits, other arms) on theta_0^G960 remains undecided; the running D7 A/B stays
   on theta_0^G as registered. Detail: `SAE_3D_GTRACK.md` A5 and `PLAN.md` §3d.A Status.
   The superseded offline D7-v2 design CLOSED at
   D7.0b: its exact K=4/Q2
   common-regular training graph admits 56 rows/two speakers and the independent necessary core has
   at most 120/four, far below the frozen 6,778/201 floor. No external assignment, loss preflight,
   scorer A/B or policy leg ran. Do not retry the optimizer or relax the floor. USER 2026-08-21
   replaces that design inside active D7-GAN-SEQDISC: generate theta_0^G greedy pseudo-text for all
   281,241 960 h utterances, then train a one-pass matched scorer control/candidate with one online
   same-speaker donor per anchor and only a reciprocal 0.8--1.25 duration window. No offline edge
   table, matching, chapter balance, nuisance quartile or capacity law survives. D7.0/D7.1 may be
   implemented now without another planner round; do not displace running funded GPU jobs.
   2026-08-21 (later, verifier): D7.0/D7.1 implemented and launched — decode shards running, the
   scorer A/B gated behind the D7.0 preflight barrier, no funded job displaced; detail in
   `PLAN_3E1.md` D7 Status and `SAE_3E1.md` Verifier feedback. Decode shards FINISHED 18:11;
   both pre-run fixes verifier-confirmed; the user restarted the d7 manager, the pool PASSED on
   its own artifact, and the preflight FAILED its control-parity step on an unsatisfiable
   bit-exact gradient assertion (CUDA-atomics noise exceeds the cross-copy difference; losses
   exactly equal). Planner amendment same day pins the operational parity rule
   (self-calibrating noise floor, `PLAN_3E1.md` D7 Status); one implementer edit plus one more
   user-run d7 manager restart are pending. Same day latest: parity fix verified, restart done,
   preflight PASSED under the amended rule; both D7.1 trainings then failed closed at data load on
   the first of four own-infeasible degenerate greedy anchors (verifier census 4/281,241, all
   train-role, none held; runaway-repetition texts). The incumbent recipe drops such rows with a
   counted diagnostic, so a drop-and-count amendment with a named-four-row fail-closed bound is
   registered in `PLAN_3E1.md` D7 Status (with its D8 preflight consequence); 2026-08-22 the
   edit is implemented and verifier-confirmed (four named train-role drops, hashes unmoved) —
   only the user-run d7 restart pends. D8.0 ran and its binding clause read UNRESOLVED on a
   frame contradiction; the planner's clause-(a) ruling (raw 50 Hz feasibility join, v3 read,
   no new dump) is registered in `PLAN_3E1.md` D8 Status. 2026-08-22 later: the v3 read ran
   and is verifier-confirmed — clause (a) GO (exclusion 0/5,730, median distinct feasible 12
   vs threshold 3); D8.0 is discharged, D8.1a-b stay gated behind the D7.2 verdict as
   registered. 2026-08-22 latest: the user's 22:50 d7 restart worked and D7.1 is COMPLETE AND
   VERIFIER-CONFIRMED on both arms (one 14-minute pass each; four named drops confirmed from
   each arm's own artifact; `online_weight` the only non-metric cross-arm difference); D7.2
   then ran all four clauses and FAILED clause 2 exactly as flagged — **D7 CLOSED 2026-08-22**,
   verifier-confirmed, no policy leg, no rescue (`PLAN_3E1.md` D7 Status). The D7.2 verdict
   releases D8.1a-b under the user's standing D8 funding: the D7.1 exact control stays the
   pinned comparator, the no-go clauses and arm-selection rule bind at D8.1a, and the D8.2
   admission job must persist per-anchor bootstrap inputs. D8.1a LAUNCHED 02:32,
   build-verified, five weight-job fixes in and verified; RELAUNCHED 08:10 at the
   corpus-scale batching after a verified wall-clock projection showed shards missing the
   unraisable 11.5 h clamp — all downstream hashes moved and are verifier-reconfirmed, the
   rulings carry over, and the verdict is accepted only with a zero-mismatch
   greedy-equivalence read and the 5 % safety valve clear (`PLAN_3E1.md` D8 Status
   2026-08-22).
   2026-08-21 (planner, on the user's instruction): D8 REGISTERED in `PLAN_3E1.md` —
   posterior-weighted multi-hypothesis scorer refit (soft EM over theta_0^G sampled rollouts,
   weights from the arm's own shaped score at pinned lam_lm=1.0, D7 control reused as the
   comparator). USER-FUNDED 2026-08-21 ("I approve starting D8"): D8.0 — the CPU read of the
   frozen group-12 dump — starts now; D8.1 still waits for the D7.2 verdict as registered; the
   policy leg still needs its own launch word.
5. **PLAN_3A matrix wrap-up**: M4 contingency call; collapse the sub-plan when closed.
6. **§1e §2.5(d)+usage gates on the ep50 pins** — the §3d init upgrade path.
7. **G2P-equivalence ceiling** on existing rollouts.jsonl (CPU): phone-reachable vs
   orthography-only oracle-gap split.
8. **Rung repair** (Rung S 1 h/10 min): first attempt VOID (budget artifacts, `SAE_2S.md` approaches 3-4);
   extend AV budgets through the phase transition, ARs get full budget, then per-rung §2.5(d).
9. **Rung 0 is COMPLETE; §2a is unblocked but deferred.** The fixed CTC-student + lexicon/4-gram
   word decode is 17.96/21.87 dev WER with full 2,703/2,864 coverage. Qwen rescoring can now run,
   but it is behind §1g/H4 because it cannot resolve the north-star initialization question.
10. **B0 gate table** (§3b) — role shrunk by the PLAN_3A closures; read under psi_align only if
    the target axis reopens.
11. **§1g simple weak initialization — detailed handover for priority 1 (rewritten 2026-08-19
    after the USER clarified Phase 1's role).** H1 is accepted: the construction-only topology read
    selected two states for both live routes and fixed the phone duration at `p=0.23560298`; do not
    rerun it. H3 calibration is also complete and valid: the corrected 715,099-run stream selected
    full-loss ESPUM seed 0/update 30,000 on the exact 6,414/890 roles, its strict update-population
    `Q`/`B` projection is materialized, and the GH200 resume trajectory is bit-exact. H2's numerical
    engine, actual wired start, strict input parsing, evidence, and shard merge now pass. H2 is now
    corrected and verified: repair consumes the same explicit deleted-silence boundary vector as
    scoring/decoding, fails closed when it is absent, and passes 23/23 channel tests including exact
    enumeration. Eight persisted alternatives remain an output-only cap because one-best and
    confidence use the complete beam. H3's three simple 7,304-ID final initializers are finished;
    the selected ESPUM seed-0/update-30,000 final refit and strict projection are finished.
    H4's 821-job prerequisite graph is complete and verified; it binds all corrected starts/counts,
    the selection donors, and both resource contracts. Preserve it. The remaining boundary is the
    bounded beam-stability extension, full-role decode/raw scoring, and deterministic sole-selector
    aggregate specified in `PLAN_1G.md`; no selector, final refit, or evaluation result exists yet.
    Reuse 1g.4's spectral and hard-descriptor failures; the unrun six-factor product is corrected to
    not answerable and stays parked. Reuse the fixed 1f recipes and original artifacts as provenance,
    but not as held-out inputs: both banked seeds saw the evaluation audio. The first E5 job remains an
    engineering rehearsal and cannot fire a gate. Complete the remaining H4 full-role decode,
    selector, final-refit, and score assay, then test policy-side and scorer-side SAE handoffs
    separately.
    Characters are the first lexicon-free candidate. Use the loop's exact BPE only for a demonstrated
    scorer-interface need. Resegmentation, repeated-speech mining, synthetic speech, and adaptive
    restart searches remain deferred. Prospective admission compares uncertainty-aware gains over identically
    treated content-free controls and then measures downstream usefulness; it does not reuse the
    historical absolute 0.05/0.05 cliff. The implementer-facing corrective package is Phase 1g.H in
    the canonical specification, `PLAN_1G.md`.

*Read 2026-08-07 (planner): the §3c 100 h replay arm FAILED its matched-compute read — ep2
23.94/29.22 vs the 10 h arm's 13.15/16.13, never beat its init, killed at ep4 (18.79→46.71). It
survives only as the artifact-backed 2S bar for the 100 h bed (§6.8).*

*Read 2026-08-07 (planner): §3e.1 fan-out closed — ranking noise refuted (recon within-group std
0.1112→0.0276), correlated bias live (the scorer rewards the inherited filler), group blindness
untested; gate v1 found gold-conditioned as instrumented and sign-blind to the filler mode →
gate v2 registered pre-verdict; ladder D0–D4 pre-registered in `PLAN_3E1.md`.*

*Read 2026-08-07 (planner, post-diagnostics): `SAE_3E1.md` verified clean against the job
outputs. Noise refuted again in-group at the operating point; bias ~70% psi_align-family /
~30% shared text (gold control pays 0.167 vs the loop's 0.243); group blindness partial and
binding (23%/9% contrast coverage) — the sampling sweep is a co-requirement, not a
contingency. The replay collapse is re-diagnosed as scorer DRIFT off the gold domain (contrast
rose 86% while CE_true crossed uniform) — gate v2 gains an absolute unit-marginal floor.
Fork presented to the user: sweep + D1/D2 (contrastive co-primary) + D3, ~20-35 GPU-h.*

*Read 2026-08-07 (planner, D1 read): D1's power check FAILED as pre-registered, and the audit
shows why no filler statistic could have separated the arms — the probe's headline insertion
discount was majority a state-count artifact (LM control unmatched in length), while the real
invariant is the lattice's ~0.03 nats/frame price per inserted emitting state, in the gold
control too. The cheap-insertion exploit is a property of the alignment lattice open to every
minimal-state word; contamination chose which word, not whether. Text repair demoted to hygiene;
insertion pricing (contrastive term; lambda reprice bounded by the audio-free share) is the
load-bearing lever. Gate v2 (i)'s improvement clause is domain-confounded by the held set's
provenance — floor-only amendment for changed-text candidates, user's blessing pending. D3 cost
corrected 9-18 -> ~85 GPU-h (planner's 100 h-bed assumption; the bar pins the 960 h bed). Two
numeric slips found, direction-neutral.*

## Resources, notation, anchors

| Item     | Value |
|----------|-------|
| Audio    | LibriSpeech 960 h (no transcripts in the unsupervised arm) |
| Text     | LibriSpeech LM corpus (`get_librispeech_normalized_lm_data()`) |
| Prior knowledge | Pronunciation lexicon + G2P (allowed); MFA gold alignments (evaluation only) |
| Encoder  | **wav2vec2-Large-lv60, layer 15** (SSL-only ckpt; decided 2026-07-18, §1c). 1024-d @ 50 Hz, per-utterance norm; units = k-means K=500 on 50→25 Hz pooled states; AV adapter stride ×4 → 12.5 Hz. Frozen for unit dumps and the GAN; AV SFT trains the transformer (conv extractor frozen); frozen inside the GRPO loop. BEST-RQ = documented negative (`SAE_1c.md`). lv60 pretrains on 60 kh LibriLight audio, zero transcripts. |
| LLM      | Qwen3-1.7B (Phases 0–4), Qwen3-8B (Phase 5 only) |
| Compute  | 4×GH200 96 GB per experiment |

**Storage placement (user decision 2026-08-22).** Any job that creates many small files (high
inode usage) must put that payload on `$SCRATCH`, and future jobs of that character are designed
to land there; trained model checkpoints and every durable/decision-bearing artifact stay in the
project fileset. Constraints that bind this: the project fileset is at ~3.58M of 4.0M inodes and
~47 of 54 TB (jutil 2026-08-19/22 reading, cached); `$SCRATCH` auto-purges untouched files on a
90-day lease starting 2026-09-01, so scratch may hold only regenerable payloads; when relocating
an existing job dir, move only the payload subdir and symlink it back, never the `finished`
marker, which stays in project. The inode test for a job design is outputs per cell PLUS
upstream fan-in (one `input/` symlink per upstream job, duplicated in full by every cleared
rerun dir, which sisyphus keeps forever), times the reruns the job is likely to need (replaces
the outputs-only reading, 2026-08-22, because the fan-in dominates for wide-input jobs:
implementer-measured and planner-verified, `H4ContextResourceGateJob` carries 342 input
symlinks against 2-3 outputs, and its three cleared rerun dirs hold 1,026 debris symlinks —
removable, pending the user's word). Measured 2026-08-22: D8.1a dump shards write ~10-23 files
each (the payload is one `rollouts.jsonl`) and the H4/H4-LM family writes 2-3 files per cell
plus the fan-in above — still not relocation-scale; no kaldi-style many-small-file archives
exist on this route. Inode
census 2026-08-22 (du over the fileset): ~3.75M scanned inodes split wu24 985k / xu34 948k /
struver1 873k / zeyer1 610k — three quarters of the pressure is other members' trees. Within
wu24: this setup 268k (largest block `work/speech_llm/sae/h4_decode_jobs`, 136k across 4,180
finished tarball-cleaned job dirs — frozen 1g.2 gate evidence, exempt from scratch relocation
under the purge lease), the 2026-05-20 and 2026-07-28 setups 251k + 196k, conda/venv envs
240k. No current SAE job needs relocation; the policy binds future many-small-file designs.

**Notation.** x waveform; h = E_l(x) encoder features; u = dedup(kmeans_K(h)) unit sequence;
z grapheme transcript; phi = G2P(z), stress-free ARPAbet, one canonical pronunciation per word, no
word-boundary symbols in AR inputs. AV: p_theta(z|x) = base LLM + LoRA-A + conv
downsampler/projector. AR/scorer: p_psi(u|phi). AV-U: p(z|u), unit-token-input verbalizer
(LoRA-A'), the §3B vehicle. p_base(z): frozen adapterless base LLM as grapheme prior. T: text
corpus; T_phi = G2P(T).

**Code anchors** (relative to `recipe/`; `ssl/` = `i6_experiments/users/wu/experiments/ssl/`,
fixed 2026-08-17 — the bare `ssl/` base does not exist under `recipe/`): AV SFT recipe
`2025-10-speech-llm/src/speech_llm/prefix_lm/sis_recipe/exp2025_11_06_speech_llms/librispeech/configs/`
(w2v2 variant `config_sae_2s_av_sft_w2v2_v1.py`); GRPO loop `train_steps/sae_grpo.py` + configs
`config_sae_3a_*`; psi_align `sae/psi_align.py` + `sae/psi_align_jobs.py`; HF downloads
`hf_models.py`; k-means `ssl/experiments/pretrain_two_level/kmeans.py`; LM corpus / lexicon / G2P
`i6_experiments/common/datasets/librispeech/{language_model,lexicon}.py`; gold alignments
`ssl/analysis/seg_diag.py` (eval only); external references: fairseq
`examples/wav2vec/unsupervised`, ESPUM arXiv:2310.02382, Hori et al. arXiv:1811.01690; survey
numbers in `ssl/LITERATURE_REVIEW.md`, `ssl/SPEECH_UNIT_BPE_REVIEW.md`.

---

## Phase 0 — Foundations

### 0a. Representation audit

**Purpose.** Measure what the frozen encoder and unit inventory can support — the information
ceiling for any downstream mapper, and the calibration for Phase 1.
**Approach.** k-means per layer/K on ~100 h scored against MFA gold (eval-only): PNMI/purity,
CTC-probe PER, oracle-assignment PER, H(phi|u), plus a label-free utterance-separability probe.
**Experiments.** Layer × K sweep + probes; freeze the winning (layer, K, centroids) tuple.
**Gate.** Proceed regardless — values calibrate rather than block; CTC-probe PER > ~25 % or
oracle-map PER > ~45 % means the unit inventory is the constraint (shorten §1a, expect §1c).
**Status: CLOSED.** Tuple frozen; linear probe 0.145 vs oracle-map ~0.53–0.60 — the *units*, not
the encoder, cap hard assignment, which is the bound that closed §1a. Log: `SAE_0.md`.

### 0b. Phoneme/grapheme-adapted LLM (CPT)

**Purpose.** A phoneme-aware LLM for decipherment LMs, neural P2G, and phoneme priors.
**Approach.** Extend Qwen3-1.7B-Base with ARPAbet tokens; CPT on mixed phonemized / grapheme /
synthetic-P2G streams rendered from the text corpus.
**Experiments.** None run.
**Gate.** Phoneme-LM ppl stabilized; grapheme ppl regression ≤ 5 %; P2G robustness curve
(theta_P2G = max input PER with output WER ≤ 40 %).
**Status: DEFERRED (2026-07-18), never run — consumers dissolved.** Revival triggers: §2a shows
lexicon/word-LM-limited headroom, or the Phase-4 pure-phoneme arm runs. If revived, drop `<wb>`.

### 0c. Supervised topline of the exact AV architecture

**Purpose.** The architecture-gap denominator for every rung, and Delta_input = WER(AV-U) −
WER(AV), which decides whether the token-only AV-U can carry mainline experiments.
**Approach.** SFT on true LS960 transcripts through the AV path (quarantined), feature-input and
unit-input twins.
**Experiments.** The two SFTs.
**Gate.** Healthy: dev-other ≤ ~10 %. Blocker: > 14.33 % — worse than the LS100 CTC baseline
means the architecture, not unsupervision, is broken.
**Status: PENDING, unscheduled** — not run on the wav2vec2 stack; the 2S/G-track SFTs have
served the calibration role in the meantime.

### 0d. LM-prior domain adaptation to LibriSpeech text (USER-proposed 2026-08-06)

**Purpose.** Close the domain gap of the lam_1 prior (and §2a rescorer): stock Qwen3-1.7B-Base is
multilingual web text, the candidates are 19th-century prose under LibriSpeech normalization —
genre and surface convention both off-distribution.
**Approach.** Full fp32 finetune of Qwen3-1.7B-Base on lowercased `librispeech-lm-norm` at
one-pass volume, then re-run of the 10 h AV SFT on the adapted donor (theta_0') and `_lbslm`
shaped loop arms on both beds at unchanged lam=1 / T=0.7. (Replaces "short finetune, text-only
swap", 2026-08-08, because the AV checkpoint carries the 2.03 B donor weights over anything
`av_args()` names — a donor swap only lands through re-running the AV SFT, so the phase is a
retraining and its main effect appears there; `SAE_0d.md` concl. 1.)
**Experiments.** (i) Blocking pre-check: dev/test disjointness of the LM corpus by 8-gram
content scan with train-clean-100 as positive control and a >=20 % control-power clause.
(Replaces "book-level disjointness", 2026-08-08, because the norm corpus carries no book
boundaries; the positive control is what gives the scan power.) (ii) Offline re-rank of the
existing n=512 rollout dumps with lam_1 under base vs finetuned, same bed/n/G (~1 GPU-h); loop
use only after that read.
**Gate — NOT perplexity.** gap_true + spearman + the **audio margin over the audio-free null**;
if the margin shrinks, reject regardless of perplexity (a better English prior raises the null
too, and the over-generation exploit gets stronger, not weaker). After any swap: re-sweep lam_lm
per bed, keep `lm_prior_norm="units"`, recalibrate the lam_1/lam_2 balance.
**Status: EXPERIMENTS COMPLETE; interpretation partly open — verified 2026-08-08** (`SAE_0d.md`; planner audit same
day: all numbers reproduce; the "one exact pass" claim is false as run — rank-dependent shuffle
seeds gave one-pass volume over 68.4 % of distinct lines; no number invalidated; pin a common
`random_seed_offset` on any future donor iteration).
- 2026-08-08, (i) **PASSED**: dev/test 8-gram overlap 1.59–3.36 % vs the train-clean-100
  positive control's 42.98 % (13–27x separation); the eval books are not in the corpus.
- 2026-08-08, (ii) **READ by the planner** (the sweeps existed, finished, unlogged:
  `RewardShapeSweepJob.{yCfwZSr3huv7,GUbnTUM2ggiv}` stock vs `.{oaOqWCrd3ZPO,FPIKAU6TkEK4}`
  adapted): absolute ranking improves everywhere (theta0 bed, T=0.7, lam=1: spearman
  0.6684 -> 0.7132, sel_wer 0.1578 -> 0.1187, eta 0.1583 -> 0.5644) AND the audio-free null
  strengthens everywhere too (null spearman 0.4668 -> 0.5258 theta0, 0.5216 -> 0.5952 gtrack) —
  the gate's feared mechanism is real. The gate text pinned neither statistic nor lam: at the
  loops' operating point (theta0 bed, lam=1) the sel_wer and eta margins over the null WIDEN and
  spearman's shrinks by 0.014; at each column's best lam (0.3) and on the gtrack bed at every lam
  the margins SHRINK. Planner verdict, **needs the user's blessing**: pass for theta0-bed loop
  use at lam=1; do NOT chase the re-swept peak lam=0.3 (that is where the free-English share
  grows); no G-track use of the adapted prior is licensed by this read.
- 2026-08-08, the unregistered but verified main effect: theta_0' re-SFT alone is
  **11.43 / 15.54 dev, 11.99 / 14.34 test** vs stock 16.91 / 20.64, 15.28 / 20.78 — a clean
  one-argument A/B at ep50 both, psi-view deltas <= 0.06 — better than anything any loop earned
  from stock theta_0. Loop use began before the (ii) read (procedural violation, noted); at
  unchanged lam=1 / T=0.7 the now-finished `_lbslm` arms (100 h `fFp8sXTA5Wug`, fixed-last
  5.22/9.67 dev; 960 h 1-pass `vhyvv2waeU16`, 6.46/11.41 dev and 6.94/12.16 test) are the
  controlled donor-axis A/B against the stock shaped arms. The donor checkpoint was selected with
  transcript-dev perplexity, contrary to the unsupervised-arm selection rule, but the curve was
  monotone and the selected checkpoint is the fixed final endpoint; future donor runs use a fixed or
  label-free pin. The 2026-08-18 EOS-token scoring mismatch leaves only the small gate-(ii) reward
  margins unresolved pending the registered re-score; it does not change these WER endpoints.

---

## Phase 1 — Bootstrap

Policy: decipherment first (novelty-preserving), PUSM and GAN as gated fallbacks; resolved
2026-07-18 with the GAN passing on wav2vec2 and the encoder decision falling out of it.

### 1a. Decipherment program

**Purpose.** Classical LM-guided decipherment of unit streams — the primary, adversarial-free
bootstrap the paper's story wants.
**Approach.** Hard CDF/ICM assignment; fertility-HMM channel model trained by Baum–Welch; OT
embedding-cloud init; unsupervised LL selection with a restart-agreement diagnostic.
**Experiments.** Ran on the BEST-RQ-era units with multiple inits/restarts.
**Gate.** dev-other PER ≤ 50 % under the §1.0 unsupervised ppl-selection metric.
**Status: CLOSED permanently, on a bound** — decipherment LL anti-aligned with PER, and hard-unit
decipherment is capped by §0a's oracle-map ceiling on *either* encoder; do not revisit. `SAE_1a.md`.
AMENDED IN SCOPE 2026-08-18 (planner, `PLAN_1G.md`; replaces the unqualified "do not revisit",
because both legs were read back to `SAE_1a.md` and neither covers the discrete case): the closure
stands as written for CONTINUOUS generative maximum likelihood over features — the configuration that
produced the anti-alignment — while the DISCRETE channel decoded through a language model on the
pooled stream is reopened as §1g, since the ceiling leg bounds memoryless lookup decodes only and
`SAE_1a.md` approach 4 measured the discrete objective WELL-aligned and init-limited. The only
real-data discrete evidence is one row — Gromov-Wasserstein init collapsing on the RAW stream — and
§1g re-runs 1a's own anti-alignment test on the pooled stream before funding any fit.

### 1b. Fallback A — PUSM

**Purpose.** Positional/skipgram distribution matching — the escalation aimed at unbroken
permutation symmetries, decipherment's known failure mode.
**Approach.** ESPUM recipe: frame one-hots → CNN generator + boundary segmenter, unigram +
skipgram L1 objectives, length-matched text batches.
**Experiments.** None run.
**Gate.** Same §1.0-metric selection, PER ≤ 50 %.
**Status: NOT EXERCISED** — superseded by 1c's pass; no spec beyond the Approach line
survives (pre-restructure backup deleted) — re-derive from the ESPUM paper if ever reopened.

### 1c. Fallback B — wav2vec-U 2.0 GAN

**Purpose.** Feature-level distribution matching that bypasses the unit inventory — the fallback
matched to §0a's "units are the constraint" verdict, and the encoder-discrimination instrument.
**Approach.** Faithful fairseq w2vu2 reproduction (rVAD trim, batch-normed features, CNN
generator to ~12.5 Hz, discriminator + gp/sp/pd/ss terms); selection by the §1.0 unsupervised
metric only.
**Experiments.** Full grid × seeds on both encoders, identical pipeline.
**Gate.** §1.0 metric with a non-empty converged filter set.
**Status: PASSED on wav2vec2** — the honest perplexity-selected seed is 0.173/0.214
dev-clean/dev-other PER; 0.137/0.168 is the oracle-best seed and is diagnostic, not reportable.
BEST-RQ stays flat at 0.75–0.92; this run decided the encoder. Tables: `SAE_1c.md`.

### 1d. Rung 0 self-training

**Purpose.** The standard-recipe baseline (Rung 0) from the winning bootstrap — the number the
loop must beat.
**Approach.** WFST pseudo-label decode → CTC finetune of wav2vec2 on pseudo-labels (HMM-GMM stage
skipped, no-Kaldi route), last checkpoint, lexicon + 4-gram word decode.
**Experiments.** CTC student and word decode complete.
**Gate.** §1.0-metric selection throughout; Rung 0 = word WER of the final system.
**Status: CLOSED; Rung 0 complete** — 0.172 dev-other phone PER; the fixed lexicon/4-gram word
decode is 17.96/21.87 dev WER with 2,703/2,864 utterances and zero empty hypotheses
(`Wav2Vec2KenlmDecodeJob.AQw3EcUo6rks`). `SAE_1d.md`.

### 1e. Pairing-free initialization (mainline; USER priority 2026-08-01)

**Purpose.** Initialize AV and AR from unpaired audio + text only — no GAN, no decipherment; if
it clears its gates it replaces the G-track's GAN init and restores full independence.
**Approach.** SFT on length-paired / random-paired / audio-continuation pseudo-pairs; any 1e loop
runs joint AR with the lam_1 + lam_2 anchors mandatory (no seed pins the text side).
**Experiments.** SFT screens done at gold budget; §2.5(d) + usage gates on the ep50 pins are
queue 6. Kill-switch if all arms gate flat: non-adversarial output-distribution matching, then
§1c/§1d stays the init of record.
**Gate.** §2.5(d) on vanilla-unit rollouts before any loop compute.
**Status: UNDECIDED** — the length-vs-random contrast is real (209.8 vs 307.0 dev-clean) but AR
CE is length ≈ random (Δ0.019 nats); whether the reward carries rank is exactly the pending
§2.5(d) question. `SAE_1e.md`.

### 1f. Statistics-matching initialization, revisited (USER-directed 2026-08-12)

**Purpose.** Answer whether FIXED-statistic (frequency / n-gram) distribution matching can
replace the GAN as the bootstrap. Honest history: 1a closed DECIPHERMENT (generative ML/EM over
features) on measured evidence — LL anti-aligned with PER, §0a unit-information ceiling — but 1b,
the moment-matching form the user is asking about, was NEVER RUN: it was superseded when the GAN
passed, i.e. dismissed by supersession, not by evidence. Reopening 1b's question is legitimate;
1a's "do not revisit" covers EM-decipherment only and stands.
**Approach.** Two cheap prerequisites BEFORE any matching run, each a registered kill condition:
(i) the §0a information audit re-run on the CURRENT enc50 unit inventory (oracle-map PER,
H(phone|unit)) — the old inventories capped ANY unit-level token mapping at PER 0.53-0.63, and a
static matching init cannot beat the oracle map by construction; if the current units are
similarly capped, the arm moves to FEATURE level (the 1b/ESPUM shape: low-capacity segmental
generator over features, unigram + skipgram / n-gram objectives) or dies; (ii) the
channel-structure read from 1a c6, measured not assumed: correlation of the unit co-occurrence
graph with the phone-bigram graph (on the old units, real co-occurrence was acoustic-flicker- and
coarticulation-dominated — the phone-LM signal the matcher needs was swamped; simulated units
whose co-occurrence mirrored the bigram recovered 0.97 of the map, real units 0.146).
**Experiments.** Prerequisites first (CPU-cheap, existing MFA/probe machinery); the matching arm
itself only if both clear. Literature-pinned modern form (planner scan 2026-08-12): ESPUM-style
positional-unigram + n-skipgram L1 matching (Wang/Hasegawa-Johnson/Yoo, ICASSP 2024,
arXiv:2310.02382 — small-batch-stable where the GAN diverges; positional unigrams are
load-bearing, bigrams-only collapses; 4/5-grams HURT), not Empirical-ODM's coverage-KL (its
corpus-frequency-inside-the-log needs ~50k-token batches, arXiv:1812.09323). Two theory anchors
map onto our prerequisites: identifiability needs (a) the channel to factorize as the model
assumes — prerequisite (ii) tests exactly this, and 1a's flicker/coarticulation finding is that
condition failing — and (b) spectral genericity of the text statistics (Wang et al. ACL 2023,
arXiv:2306.07926; Yang/Schlueter/Ney 2026, arXiv:2603.02285). Reference verdicts (2026-08-16,
replaces the verify-FLAG of 2026-08-12; all three flagged references verified first-hand):
coarse/syllable granularity ADOPTED as the default target unit (the 2510.03639 ablation
stands) while that paper's pipeline and its bootstrap claim are REJECTED; the 2306.07926
closed-form estimator enters only ridge-regularized and sigma_min-gated (never run on real
speech); 2603.02285's rank condition is kept, its training loss REJECTED as 1a's decipherment
likelihood in gradient form. Details and candidate ladder: `PLAN_1F.md`.
**Gate.** REPLACED 2026-08-16 (was: 1b's dev-other PER <= 50 % bar, registered 2026-08-12 —
no matcher result existed, so replaceable) because the USER re-set the criterion: the single
requirement is that the init be BETTER THAN RANDOM/UNPAIRED initialization. Operational form:
dominate the strongest content-free nulls — a marginal-matched random unit-to-phone map, and
the 1e pseudo-pair init — on plain PER as scored (labels eval-only) AND the audio-swap
content-dependence control; margin pre-registered in `PLAN_1F.md` before the first matcher
read. The prerequisite kill conditions stand as registered; verdicts in Status.
**Status: REGISTERED 2026-08-12.** Awaiting prerequisite runs; literature scan DONE (planner,
same day — citations inline above). 2026-08-16 planner fan-out (five-agent workflow, 28
candidates screened): design space pinned in `PLAN_1F.md` — the prerequisite screen becomes a
per-representation battery (raw / deduped / segment-pooled / Brown-K100 / unit-BPE; adds
sigma_min(P_X), Laplacian eigen-similarity, spectrum overlay; calibrated on the simulated-unit
generator), plus a six-entry candidate ladder each with a pre-investment kill-test — new
front-runners are fixed-core tri-factorization (fit only the emission matrix against a
text-pinned phone-bigram core; method-of-moments, not EM) and ridge positional-unigram least
squares (first real-speech run of the 2306.07926 estimator). The screen battery remains the
first fundable step; nothing above relaxes the registered gate or kill conditions.
2026-08-16 (later): prerequisites RAN (`SAE_1f.md`). Kill (i) FIRED — oracle-map PER 0.832
dev-other vs the 0.50 bar — but localized to over-segmentation (ins 0.692; subs 0.132 and
PNMI 0.682 both program-best), so the fork is staged: the battery's pooled rows on the
current codebook decide the representation, feature-level ESPUM is the fallback, death only
if all cap. Kill (ii) measured: the observable graph carries bigram signal (PMI spearman
0.373/0.370 vs floor ~0.215, ceiling ~0.41) but the matcher's own TV objective separates
truth from no-correspondence by only 9-11 % relative — a separability bar for
transition-consuming matchers is now pre-registered in `PLAN_1F.md`; transition-free entries
are unaffected and lead the queue. Same day the USER ruled: simplest-possible init (ladder
re-ranked — pooled-rows screen first, then fingerprint assignment, then the ridge solve;
ESPUM last) and the gate replacement recorded above. Detail: `PLAN_1F.md`.
2026-08-16 (battery): kill (i) CLEARED at the unit level — data-driven segment pooling
passes the bar on every rung (`seg12.5` 0.414 / `seg16` 0.452 / `seg9` 0.481 dev-other vs
0.50, program-best ceilings), the feature-level fallback is not exercised, and inventory
coarsening at fixed rate is catastrophic (`brown100` 1.152) so the 500-way codebook stays.
Entry 2 (ridge positional-unigram) CLOSED by its own sigma_min gate, structurally (0 on all
pooled rows; a simulated perfect channel also reads 0). The kill-(ii) separability bar is
VOID AS MEASURED (the real stream beats its seg_swap ceiling on pooled rungs — coarticulation
inverts the control), so entries 1/4 stay parked behind the transition-free entries with no
post-hoc replacement bar. Arm-gate margin pre-registered in `PLAN_1F.md` before any matcher
run: beat min(random-map, 1e-pseudo-pair) by >= 0.05 dev-other PER AND degrade >= 0.05 under
audio-swap. NEXT FUNDABLE STEP: entry 3 (fingerprint assignment) + its two nulls on
`seg16`/`seg12.5`/`seg9`, entry 6 kill-test on `ubpe12.5` — CPU-cheap; a funded init later
needs the pooling pass on the assign-side shards. Verdicts: `PLAN_1F.md`; rows: `SAE_1f.md`.
2026-08-16 (USER ruling 3): the screens run TWO text-side arms per the 3a section-5c
pattern — phone-level reference (statistics from T_phi) vs lexicon-free (text-BPE-512 /
frequent-word statistics from the raw corpus; entry 6's function-word kill-test is that
arm's precondition) — and the gap is reported as the measured price of the lexicon. The
phone arm's extra lexicon touchpoint (pseudo-labels need lexicon + word decode to become
SFT text; the lexicon-free arm outputs text directly) is disclosed in its supervision
cost. Gate and margins unchanged, applied per arm. Detail: `PLAN_1F.md` ruling 3.
2026-08-16 (entry 3): the fingerprint assignment FAILS the arm gate on every
representation — best margin over the stronger null +0.015 vs the registered 0.05, and
audio-swap movement at the random null's own level, i.e. content-free by the control;
both of its own kill-tests also fail. Measured cause: the matchable transition-free
statistics rank the true phone only ~2x above chance — too diffuse for a 39-way
assignment. NOT FUNDED (licenses not funding, not "could never work"). Remaining
simple-family step: entry 6's function-word kill-test + ruling-3's lexicon-free arm;
entries 1/4 stay parked; entry 5 (ESPUM, GPU training, the one entry with published
real-speech evidence) stays last — funding it after entry 6 is the USER's call.
Verdict detail: `PLAN_1F.md`; table: `SAE_1f.md` approach 4.
2026-08-16 (USER ruling 4): no TIMIT bed — the staged TIMIT reproduction proposed for
entry 5 is declined; entry 5, if ever funded, is judged directly on LibriSpeech against
the arm gate. NEXT STEP (funded, dispatched to the implementer): entry 6's function-word
kill-test on `ubpe12.5` + the ruling-3 lexicon-free text-side screens — CPU-cheap, on
existing artifacts; gate and margins as registered.
2026-08-16 (later): entry 6's kill-test CLEARS (verified) — the lexicon-free arm keeps
its precondition, with the recorded scope that the signature is positional only and an
utterance-onset acoustic confound remains (eval-only oracle read on the hitting units
green-lit to resolve it). Ruling-3 screens launched (4 LexFreeMatchJob, one per
representation); frame ratified except the oracle ceiling, overturned to the candidate's
own restricted map space — re-run required, gate reads unaffected. Verdicts and both
frame rulings: `PLAN_1F.md`; numbers: `SAE_1f.md` approach 5, conclusions 19-23.
2026-08-16 (later still): onset control DONE and verified — the confound resolves
LINGUISTIC on `seg12.5` (unit 403 is a genuine THE-like unit) while `ubpe12.5`'s
headline hit was a missed all-silence unit (proxy defect recorded; one genuine YOU-like
hit remains), so the lexicon-free arm's precondition stands on direct evidence; hit
counts corrected under `SAE_1f.md` conclusions 19/22, amended verdict and the proxy-
defect consequence for the running `ubpe12.5` screen in `PLAN_1F.md`.
2026-08-17 (ruling-3 batch amendments, planner-verified): the `ubpe12.5` open-ceiling
screen died at wall clock unwritten — the restricted re-runs (queued) carry BOTH
ceilings from one pass, with a pre-registered bit-for-bit reproduction check against
the finished seg runs; the `ubpe12.5` STREAM itself was found budget-stopped at a
default (8000 merges, measured 14.08 tok/s vs the 12.5 target) — no rebuild this
batch, the matched-rate contrast with `seg12.5` is retired, a true-12.5 rebuild is a
conditional follow-up; the words text side is unreachable at 2.8 words/s on every rung
(screened at each rung's floor with the mismatch printed, pre-registered as a frame
limitation). Rulings and the resume-change ratification: `PLAN_1F.md` 2026-08-17.
2026-08-17 (ruling-3 batch close, planner-verified): the screens FAIL the arm gate
in all twelve cells (best dev-other M2 0.0252 vs 0.05; M1 negative in 10 of 12) —
the lexicon-free arm is NOT FUNDED and the phone-reference side fails the same
gate, so NO 1f init is fundable from the screens run to date. The words cells price
a rate-mismatched arm (no rung reaches 2.8 words/s, pre-registered), but the
rate-matched cells fail equally, so no retry is proposed. All pre-registered
determinism checks passed bit-for-bit across three job generations; one hash-label
swap in the log Catalog (seg12.5/seg16 audio jobs) is being corrected — labels
only, no numbers. USER FORK NOW OPEN: fund entry 5 (last unkilled ladder entry,
LibriSpeech-direct per ruling 4, raised bar), register a new screen for parked
entries 1/4, or close 1f. Detail: `PLAN_1F.md` amendment (7).
2026-08-17 (USER ruling 5): the fork resolves — entry 5 FUNDED, with a second
instruction that the whole process stay as simple as possible. Spec registered same day
pre-run (`PLAN_1F.md` entry-5 funded batch): the ESPUM reference mechanism (ICASSP 2024,
verified first-hand including its released code) with seven traceable deviations — fixed
measured boundaries instead of the learned segmenter, our 500-way one-hot units, the
ruling-3 silence convention, LABEL-FREE selection (the released config selects by error
rate against test references — quarantine-incompatible, so the deviation is mandatory),
the screens' eval protocol so the banked seg12.5 nulls price the candidate directly (M1
bar: dev-other PER <= 0.8446), 3 seeds plus the bigram-only collapse control as the
health pair — one contained batch on the 20.5 h seed stream. Honest anchor: the paper's
UNMATCHED-text TIMIT column (PER 0.451-0.473); LibriSpeech is unanchored, the
research-bet framing stands. A failed gate closes entry 5 and returns 1f with no
unkilled entry. Post-close defect disclosed (`PLAN_1F.md` (7b)): the ruling-3/entry-3
text statistics sampled only the first 60.6% of the alphabetically sorted corpus —
nulls and candidates shared the sample so the verdicts stand; a standing full-coverage
sampling rule is registered and entry 5 pins the proven full-coverage sample.

### 1g. A simple weak starting point for the SAE loop (rewritten 2026-08-19; sub-plan `PLAN_1G.md`)

**Purpose.** Produce a label-free, audio-dependent seed that gives the speech autoencoder loop a
better starting point than an identically treated content-free control. Phase 1g does not need to
solve ASR by itself.

**Approach.** Estimate `P(audio unit | text symbol)` and decode it jointly with a
text language model. Test phones first because two real phone seeds, two controls, and an oracle
already exist; this is a reference and mechanics check that pays for a pronunciation lexicon.
Characters are the first primary lexicon-free route. Use the loop's exact BPE vocabulary only when a
direct scorer handoff requires it. The channel may seed either the audio-to-text policy through
pseudo-transcript cross-entropy training or the reconstruction scorer directly; test those paths separately before
combining them.

**Experiments.** Reuse 1g.0's label-free one-segment rejection; keep its old full-dev and
gold-duration cells diagnostic. Fit each prospective shared duration and recompute the live
dependence read on update audio only; choose the smaller admissible one-state or two-state form.
Reuse the 1g.4 spectral/hard-descriptor not-funded verdicts. The unrun six-factor product is not
answerable and stays parked. First run the corrected phone assay on construction-only
rebuilds of the proxy-silence masks, ESPUM, fingerprint, and both controls, with common held-out data, preprocessing,
empirical channels, full text coverage, nested local decoder, and fixed repair counts. Keep the
original full-bed seeds as transductive provenance rows only. Validate that the fitting and selection
scores follow speech content. Then run separate phone policy-side and scorer-side handoffs and start
the character route once one is valid; a combined phone loop is optional and must not delay
characters. The lexicon-free candidate receives the fixed combined test. Preserve one-best text,
alternatives, posteriors, confidence, per-utterance gate statistics, donor tables, and uncertainty
inputs. Full corrective handoff: `PLAN_1G.md` Phase 1g.H.

**Gate.** From now on, separate two questions. A seed is content-bearing when paired, uncertainty-
aware comparisons show that it beats treated content-free controls under both plain error and
same-speaker audio-swap dependence. A separate policy or scorer handoff can identify a promising
component; a usable Phase-1 initialization requires the fixed combined path to beat its matched
controls without materially degrading from its start. A failed path-specific positive control makes
that assay unresolved, not the seed negative.
The historical 1f (0.05/0.05) failure remains recorded but is not the future admission cliff.
Phone results validate mechanics. The phone-versus-character difference bundles several design
changes, including pronunciation-lexicon cost; only a lexicon-free result supports the main claim.

**Status.** **1g.2a FUNDED 2026-08-22, route fork otherwise open (replaces the bare HOLDING
line, because the user funded H4-LM execution at D scope): the 1g.2 selector gate fired
NEGATIVE 2026-08-22 — H4 is unresolved with no selector, no likelihood fallback, and the
evaluation closed (details and pre-registered consequences in `PLAN_1G.md` 1g.2 Status;
evidence SAE_1g.md verdicts 18-20) — while the user directs matched trigram/4-gram fitting
context (1g.2a) to run at D scope under the implementation ruling in `PLAN_1G.md` 1g.2a
Status.** The first E5 job
remains exploratory and non-decisive. H1 freezes the split, masks, two-state topology, and phone
`p=0.23560298`; no further H1 run is required. H2's deleted-silence boundary law is now identical in
repair, scoring, and decoding and passes the exact-enumeration suite; its timing preflight must not be
rerun. H3's fingerprint, random-map, and pseudo-pair final initializers are finished on all 7,304
construction utterances; the selected ESPUM final refit and strict projection are also finished.
H4's verified 821-job prerequisite graph exposes every corrected channel table and passes both
resource contracts. 2026-08-22 (replaces "no full-role decode ... has run", because the pre-label
boundary completed): the beam-stability extension ruled every sequence setting ineligible (baseline
surface local-only), the selection surfaces and all 85 provisional maxima are persisted, hash-bound
and verifier-confirmed, every winner is local (winner beam audit discharged by the registered
exemption), and the controlled reference labels are ruled open for selector validation
(`PLAN_1G.md` Status 2026-08-22). Final refit and evaluation still have not run;
H5--H6 remain gated on H4's scientific result. 2026-08-22 latest (USER): the anti-collapse
constrained-repair probe (1g.9 — coverage-direction unigram matching plus rate regularization on
the repair objective, corrected and pre-gated by the planner) is GREENLIT at HIGHEST priority;
spec and gate in `PLAN_1G.md` 1g.9. 2026-08-22 later: 1g.9 CLOSED by its clause-0 off-ramp —
the training posterior already meets both targets on every start, the babble is decoder-resident
and start-specific, no constrained arm runs; direction fork with the user (`PLAN_1G.md` 1g.9
Status). Details: `PLAN_1G.md`; evidence:
`SAE_1g.md`.

---

## Phase 2 — Warm start (SFT)

### 2a. Offline decode → Rung 1

**Purpose.** The LLM-decoding claim with no CPT: Rung 1.
**Approach.** Frozen-base Qwen3 n-best rescoring of §1d's WFST lattices (kappa from a
dev-disjoint unsupervised sweep), with a 4-gram-prior-only control separating "better prior"
from Gutenberg memorization.
**Experiments.** Rescoring vs same-lattice 4-gram 1-best; Rung 0 is complete, so this is unblocked.
**Gate.** Rung 1 ≤ WER of the 4-gram WFST decode of the *same* lattices.
**Status: PENDING, lower priority than §1g/H4.**

### 2b. AV SFT → Rung 2

**Purpose.** Distill Rung 1 pseudo-labels into the feature AV (Rung 2); the AV-U twin is the §3B
init.
**Approach.** §0c recipe on (audio → pseudo-transcript) pairs, LoRA-A; AV-U twin with LoRA-A'.
**Experiments.** The two SFTs; pending on Rung 1.
**Gate.** Rung 2 ≤ Rung 1 + 1 abs AND dev insertion rate ≤ 1.5× the teacher's.
**Status: PENDING** — the G-track AV^G (13.89/18.34 from 28,539 train-clean-100 utterances with
§1d pseudo-labels; its later loop bed is 960 h) is evidence the distillation step itself works.

### 2c. AR SFT

**Purpose.** Warm-start a text→units channel model for loop use.
**Approach.** Boundary-free phonemes → deduped units CE (LoRA-B), no speaker/F0 conditioning.
**Experiments.** Pending in-phase; in practice superseded by psi_align training (§3a) for the
reward role.
**Gate.** The old ΔCE usage screen is superseded — the binding screen is §2.5(c)/(d) (measured
2026-07-17: full-history ΔCE ≈ +0.005, a target wall, which started the scorer program).
**Status: SUPERSEDED for the reward; retained only if an LLM-AR is ever revived in-loop.**

---

## Phase 2S — Semi-supervised anchor arm (quarantined)

**Purpose.** Loop-validity control with a known-good init, validation of the validators, the
Rung-S hedge, and the supervision-equivalence framing. Seed artifacts never enter Rungs 0–4;
mechanics and lambda ranges transfer, disclosed.
**Approach.** Paired-seed SFTs (1 h/10 h) → full GRPO loop vs self-training from the *identical*
seed, plus the shuffled-reward and frozen-vs-joint controls.
**Experiments.** 10 h loop + controls done; the 1 h/10 min rung repair is queue 8 (first attempt
VOID — budget artifacts, not seed-size verdicts).
**Gate.** Loop beats identical-seed self-training by ≥ 0.5 dev-other, unsupervised-selected.
**Status: role complete at 10 h.** The reported +1.24 was INVALID for this gate because it compared
independently dev-WER-selected minima (15.89 vs 17.13). Under the fixed four-epoch endpoint, joint AR
beats identical-start self-training by 1.61 dev-other (16.13 vs 17.74), clearing 0.5 without label-based
checkpoint choice; no learned unsupervised selector was measured. Shuffled reward is DECISIVE (ep1
207.59 vs 16.87 dev-other — the reward is load-bearing); joint AR beats frozen run-to-completion; the measured
0.27-nat information cap of the token-LM reward drove the §3a escalation. Logs: `SAE_2S*.md`.

---

## Phase 2.5 — Go/no-go instruments

**Purpose.** Cheap verdicts before any RL compute; instrument (d) is decisive for every new
scorer, target, or init.
**Approach.** (a) rerank test and (c) graded-corruption ladder (SNR ≥ 1) as pre-screens — (c) is
a known-optimistic synthetic proxy; (b) deadlock test superseded by (d); **(d) reward-RANK
probe**: replay the loop step on real theta_0 rollouts (G≈12, T ∈ {0.3, 0.5, 0.7}; T=1.0 logged,
never evidence).
**Experiments.** Run (d) for every candidate before loop compute; calibrate any new diagnostic on
the §2S paired-init models first (a failure there indicts the instrument, not the signal).
**Gate.** Within-group spearman with CI > 0, gap_true = r(z_true) − mean r(z_i) > 0,
reward-selected WER ≤ group mean. Read discipline (2026-08-05): **absolute-eta bars withdrawn** —
same-bed/same-n/same-G, gap_true + spearman lead, plus the audio margin over the audio-free null.
**Status: in active service** — governed the §3a adoption, the §3d funding decision, and gates
§1e next.

---

## Phase 3 — Joint RL loop (Rung 3, the central claim)

### 3a. Reconstruction scorer — psi_align

**Purpose.** A reward whose score moves with transcript quality on real rollouts — where the
teacher-forced AR (text-blind on fine units) and the CI-given-text LLM scorer (family-capped)
failed; alignment is the missing expressivity.
**Approach.** Conditional neural HMM over the text symbol string: ~11 M from-scratch
bidirectional text encoder, per-state categorical emissions over the 500-unit inventory, 3-way
{self-loop, advance, skip} transitions, exact forward-sum p(units, T | symbols); CI given (text,
alignment) with the alignment marginalized, unit history structurally absent, length priced
natively. Frozen in-loop by construction (`train_steps/sae_grpo.py:153` forces it).
**Experiments.** G0–G3 gates + the scorer×target matrix (§5b) and text-side axis (§5c), all per
`PLAN_3A.md`; remaining: M4 contingency call (queue 5).
**Gate.** G1 usage gate and G3 re-rank as pre-registered in PLAN_3A §6 (same-bed/same-n/same-G,
audio margin over the audio-free null).
**Status: ADOPTED.** G1 + G3 passed decisively 2026-08-05; §5c BPE 12/12 cells — carry-forward
text side `bpe512_cps15` (lexicon-free, zero OOV); M2 CLOSED (discrete k-means-500 stands);
substrate CLOSED (post-adapter 12.5 Hz); frozen-scorer state sha-verified across all six arms.
Normative: `PLAN_3A.md`; log: `SAE_3A.md`.

### 3b. Reconstruction target

**Purpose.** Choose what the scorer reconstructs; the target's information content bounds every
reward (the 2S collapse root cause was a 0.27-nat target-information cap).
**Approach.** Candidate ledger gated by §2.5(d), select the finest that passes; admissible
targets are measurements of the audio only (independence rule — the withdrawn GAN-phone stream
is the precedent).
**Experiments.** The remaining B0 gate table is queue 10, read under psi_align only if the target
axis reopens.
**Gate.** Same-set §2.5(d) comparisons against the incumbent stream.
**Status: SETTLED at avunits k500** by the PLAN_3A M2/substrate closures. History:
`SAE_2S.md` approach 13 (conclusions 23-25).

### 3c. Seed-replay

**Purpose.** Stabilize the quarantined 2S arm by keeping seed supervision in the objective (the
Hori/TTE mechanism).
**Approach.** Mix seed paired CE into AV/AR objectives; lambdas calibrated against within-group
monitors (measured lambda_av 0.466 / lambda_ar 0.996 — the AR term near-duplicates in-loop
ar_ce, so supervision enters via AV).
**Experiments.** The 100 h replay arm ran to ep4.
**Gate.** Matched-compute read: ep2 vs the 10 h arm's final 13.15/16.13.
**Status: FAILED its read (2026-08-07)** — ep2 23.94/29.22, never beat its init, 46.71 by ep4;
survives as the artifact-backed 2S bar (§6.8). Replay anchors to a good seed — the wrong anchor
for the bad-init regime (§3e.1); admissible in the 2S arm only.

### 3d. G-track — GAN-init label-free track (train-clean-100 SFT init; 960 h loop bed)

**Purpose.** The operator question: does the autoencoder loop beat plain self-training as the
refinement operator, from the same label-free init?
**Approach.** Init = ten-pass SFT on §1d-student pseudo-labels for the 28,539 train-clean-100
utterances under the **init-only carve-out** (AV: audio→pseudo-text; scorer conditioning only —
targets stay audio-derived units). The existing theta_0^G is not a 960 h pseudo-label SFT; 960 h
enters only as the later unlabeled loop bed. Arms: (1) real
reward, (2) shuffled reward (= iterated pseudo-labeling, the built-in pivot result), (3) no-loop
baselines; (1)−(2) = reward contribution, (2)−(3) = distillation contribution. Init hierarchy:
§1e (goal) → GAN/§1d (working fallback) → 10 h seed (2S only).
**Experiments.** AV^G and psi_align^G built and gated; both 960 h loop arms ran to sub-ep4;
the missing full-960 h pseudo-SFT and one-generation own-label controls are §3d.A / queue 4.
**Gate.** §2.5(d) at the init before loop compute (passed under psi_align^G; the earlier AR_G
attempt declined funding on the same read).
**Status: both loop arms HELD at sub-ep4** (§6.7/§6.10) — `recon` diverges through an
*inherited* `to` filler (init and scorer share byte-identical §1d pseudo-text, so correlated
defects are rewarded); `shaped` plateaus then slips. Standing suspicion: the frozen scorer
cannot leave the shared bad prior; the admissible fix is outer-EM re-estimation between passes,
gated on the §3e.1 rule — deferred, user's call.
2026-08-17: the deferral ENDS — the outer re-estimation runs as `PLAN_3E1.md` D6-PERIODIC/GAN
(per-boundary from-scratch d_min=2 refits on the policy's own greedy decodes; the §3e.1
acceptance-gate clause is DELETED on this track by the user's label-hygiene ruling — a gold-read
gate selects what trains the next leg, and no annotation may train or select here). The
homophone-diversity SFT arm rides the same bed as its one-argument A/B.
2026-08-20 planner read: the outer-refresh hypothesis has not produced a durable gain. The repaired
frozen scorer reaches 12.68/17.57 only transiently, while D6-PERIODIC/GAN reaches 12.85/17.89 at leg
2 and degrades to 18.38/24.01 by leg 6. Finish the already-funded trajectories and frozen schedule
control, but open no further GAN-loop mechanism arm before the §3d.A same-start self-training read.
USER 2026-08-20 registered D7-GAN-SEQDISC. Its first, offline-graph specification CLOSED at the
prospectively frozen D7.0b support gate: the feature job passed, but the exact K=4/Q2 common-set
optimizer admitted 56 rows/two speakers versus
6,778/201 required; a separate necessary-core replay bounds the whole feasible surface by 120/four.
The assignment failed closed before external matching or loss calibration, and no offline-graph
scorer, admission or policy work ran. No solver retry, support-floor relaxation or graph amendment
is authorized. Historical failure evidence: `SAE_3E1.md` approach 31 and conclusion 57.
USER 2026-08-21 corrects the active
D7-GAN-SEQDISC method to full-960 h pseudo-pairs, K=1 online uniform donors from
same-speaker 0.8--1.25 duration pools, and no graph/nuisance matching. Its full specification is in
`PLAN_3E1.md`; D7.0/D7.1 are implemented and launched 2026-08-21 (verifier-confirmed), while D7.3
policy compute still needs a launch word.
2026-08-21: both §3d.A reads now exist -- one-generation self-training FAILED both starts, the
960 h scale arm PASSED (theta_0^G960 13.11/16.82; §3d.A Status) -- so the 2026-08-20 hold's
condition is met; whether any new arm opens on theta_0^G960 is a fresh user decision.

#### 3d.A. Pseudo-label scale and one-generation self-training (USER-directed 2026-08-20)

**Purpose.** Answer two separate questions: whether pseudo-labeling all 960 h gives a better
label-free AV starting point than the existing 100 h start, and whether one fresh generation of a
model's own labels improves either the 10 h supervised init or GAN-init beyond simply continuing
on the §1d teacher's labels. The scale arm is a practical matched-exposure comparison, not a pure
data-volume ablation: unique coverage, clean/other domain mix and out-of-sample pseudo-label quality
all change together.

**Approach.** For the scale arm, retain the banked §1d word hypotheses for all 28,539
train-clean-100 utterances and decode train-clean-360 plus train-other-500 with the exact same frozen
§1d CTC student, lexicon, word LM and decoder configuration. Merge to the exact 281,241-utterance
960 h bed; every training transcript must be pseudo-text, never the audio bed's gold/blank text.
The completed HF/Ogg Arrow bed is only the audio source, not the FLAC+manifest input expected by the
legacy word-decoder job. USER 2026-08-20 selects a new packed/sharded input path that reads those
existing shards directly; do not materialize an intermediate FLAC tree. Preserve the legacy
`Wav2Vec2KenlmDecodeJob` constructor, behavior and completed hashes, including the canonical tc100
artifact. The packed reader is a separate job/interface with new hashes only for its new decode
work, while reusing the identical §1d acoustic model and lexicon/LM search. Before the 860 h spend,
run it on one fixed tc100 shard and report ordered-ID/coverage and hypothesis agreement with the
banked decoder output; this is an input-path check, not a new model-selection gate.
Train theta_0^G960 from scratch with the theta_0^G AV recipe and approximately matched exposure:
one 960 h corpus pass, partitioned into ten sub-epochs, versus ten 100 h passes for theta_0^G; walk
the same learning-rate curve by optimizer update, not by nominal corpus epoch.

For one-generation self-training, freeze theta_0 at ep50 and theta_0^G at ep10 as two teachers.
Here “10 h” means theta_0 itself, the paired 10 h AV SFT, rather than the later xCh self-training
checkpoint.
Each generates hard labels for all train-clean-100 audio with the project's canonical AV beam-4
decoder. Continue its own checkpoint with the same four-epoch CE recipe already used by the
theta_0 self-training control. Read the theta_0 own-label arm against the existing matched
comparator `ReturnnTrainingJob.xChfzEkd4CGE`, which starts from theta_0 and trains on fixed §1d
teacher labels. Add the corresponding theta_0^G continuation on its original §1d labels so the
theta_0^G own-label arm changes targets only. Keep
the current augmentation, label smoothing, batching and learning-rate schedule fixed; no confidence
filter or threshold sweep in this first diagnostic.

Iterative pseudo-labeling is established ASR practice: [IPL](https://arxiv.org/abs/2005.09267)
uses repeated LM-assisted decoding plus augmented student training, and
[Noisy Student](https://arxiv.org/abs/2005.09629) likewise separates a clean teacher from an
augmented student. The concern about a peaky fixed point is real: cache-free online relabeling can
collapse to empty transcripts in [slimIPL](https://arxiv.org/abs/2010.11524), while wav2vec-U's
same-model fine-tune→fine-tune step was essentially flat (12.1→12.0 PER) and attributed to
overfitting ([§6.5](https://arxiv.org/abs/2105.11084)). Therefore this plan funds one new label
generation with matched controls, not an automatic multi-round loop.

**Experiments.** (A) theta_0^G960 at the fixed final sub-epoch versus theta_0^G's 13.89/18.34
dev-clean/dev-other anchor. (B) theta_0-own-label and theta_0^G-own-label continuations versus their
same-start §1d-label comparators at every epoch and at the fixed final epoch. Report plain WER and
S/D/I, pseudo-label coverage/empty rate, old-versus-new transcript edit rate, length/token
distribution, and the inherited `to` rate. Training loss or confidence is diagnostic only.

**Gate.** Call theta_0^G960 a better AV start only if its fixed final checkpoint improves both dev
splits over theta_0^G; a split trade-off is reported but does not replace the start. Call fresh-label
self-training useful only if its fixed final checkpoint beats both its teacher start and its
same-start §1d-label comparator on both dev splits. Lower loss or sharper confidence without a
WER gain is self-confirmation, not progress. A further generation is not authorized by this plan;
it becomes a new decision only after a fresh-label arm passes this gate.

**Status.** IMPLEMENTED AND LAUNCHED 2026-08-20; no experimental result yet. The verifier confirms
that the legacy tc100 decoder still resolves to `Wav2Vec2KenlmDecodeJob.AQw3EcUo6rks`, the new path
reads packed HF/Ogg audio without a per-utterance FLAC tree, and the real bed has exactly 281,241
unique IDs with the banked 28,539 tc100 IDs first in identical order. Runtime acceptance required
`PackedDecodeAgreementJob.xEBbTHwTJScE` to report exact hypothesis agreement before the full decode's
281,241-ID coverage assertion could run. The one-generation graph is also
verifier-approved: both teachers and all four same-start arms are pinned as specified, with no
further generation wired. At 15:21 CEST, the packed-input acceptance condition has FAILED: shard 0
has correct ordered coverage (298/298) but only 289 hypotheses exactly match the banked decoder; the
nine differences include real word/segmentation changes rather than case or ordering differences.
`PackedDecodeAgreementJob.xEBbTHwTJScE` therefore failed closed, and the full 860 h decode and 960 h
SFT correctly remain waiting pending root-cause diagnosis. This does not block the separate
one-generation graph: its theta_0^G fixed-§1d-label continuation is running, teacher decodes are
partly finished/partly running, and both own-label continuations wait on their teacher outputs. At
15:37 CEST `JOB_AUTO_CLEANUP=True` is effective and the self-training manager is running again; no
self-training endpoint exists yet. Funding
still ends at the fixed-final theta_0^G960 AV-SFT read: no GRPO/autoencoder loop, scorer refit or D6
branch from theta_0^G960 is authorized without a new preregistered decision.
2026-08-20 later: the packed-input failure was root-caused and FIXED, never waived -- the packed
reader now reconstructs PCM16 to byte-match the legacy decoder's waveform (i6_experiments
3d3918698 + b2d98a5b1), and the re-run acceptance on the new decode hashes reads 298/298 exact
hypothesis agreement (`PackedDecodeAgreementJob.FATi7mwI43o7`; the invalid 289/298 attempt
`xEBbTHwTJScE` is preserved as history). The 860 h decode and 960 h SFT proceeded behind the
passing gate.
2026-08-20/21 one-generation verdict: the fresh-label gate FAILED for both starts at the fixed
epoch-4 endpoint (own labels worse than fixed §1d labels by 2.82/2.52 from theta_0 and 1.38/1.11
from theta_0^G; `SAE_3D_GTRACK.md` verdict 10). No second generation is authorized.
2026-08-21 scale verdict: the scale gate PASSES, verifier-confirmed same day end to end (sclite
outputs and raw counts, CTM/search chains, checkpoint identity, training frame): theta_0^G960 =
13.11/16.82 dev-clean/dev-other at the fixed sub-epoch-10 endpoint against theta_0^G's
13.89/18.34 -- both splits improve, so theta_0^G960 is the project's best label-free AV start
(`SAE_3D_GTRACK.md` A5). Two disclosures: "random init" means fresh adapter/LoRA over the public
foundation weights with no project checkpoint imported (the comparator recipe's own convention,
config-verified), and all four data-parallel ranks each make one full corpus pass (the comparator
shares this convention verbatim, so the registered exposure match holds). The funding boundary
above is UNCHANGED: nothing downstream of theta_0^G960 -- loop, scorer refit, D6 branch, or
adopting it as an init elsewhere -- is authorized without a new preregistered decision, which is
now with the user.
2026-08-21 later: the user gives that decision in part -- one frozen-scorer reconstruction loop
from theta_0^G960 is funded, registered as D6-PERIODIC/GAN960-FROZEN in `PLAN_3E1.md`. Everything
else downstream of theta_0^G960 remains unauthorized.

### 3e. Reward and update protocol

**Purpose.** The loop itself: reward composition, decoupled updates, monitoring, selection.
**Approach.** Reward per sampled transcript z (utterance units u, duration D):

    r(z) = (1/|u|) log p_psi(u | BPE_states(z))   reconstruction (psi_align forward-sum;
                                                   graphemic bpe512 sub-states, 1.5 chars/state,
                                                   SIL at word boundaries)
         + lam_1 * lm_prior(z)                     LM prior, p_base; lm_prior_norm="units"
         - lam_2 * KL_hat(z)                       anchor to theta_0 (frozen SFT snapshot)
         - lam_3 * length_hinge(n_chars(z), D)     chars/s hinge (nu 14.55, len_eps 0.4);
                                                   lam_len 0 in the G-track arms, 0.5 in Z4

(Formula corrected 2026-08-17, replaces the G2P form — because the live reward contains NO G2P
anywhere, verified at source: psi re-encodes the decoded string under its own graphemic BPE
(`psi_scorer.py:141-146`, `psi_align_jobs.py:87-104`; the "phones" branch exists but no live arm
sets it), the hinge is `len(decoded_string)` (`train_steps/sae_grpo.py:205-212`; `reward.py:14-15`
documents the deviation), and the old lam_4 OOV term is unwired dead code that raises if enabled.
The G2P map — first pronunciation, stress-free — survives in probes and analyses as phi = G2P(z),
NOT in the reward. Consequence, load-bearing for §3e.1 D6-PERIODIC/GAN+HOM and queue 7: the
orthographic channel is LIVE — homophone spellings are NOT reward-invariant; the scorer carries a
per-state price on orthographic length (the minimal-state exploit's substrate) plus any
spelling-specific emissions it learned.)

`lm_prior_norm="units"` because the per-token mean pays for length (22:1 trade measured, §6.6).
Updates are decoupled
(NLA shape): sample G=8–12 at the bed's T; scorer frozen under psi_align (any update goes
through §3e.1); AV by GRPO with group-normalized advantages (the group shares one utterance, so
speaker/prosody cancel). As built: RETURNN `train_steps/sae_grpo.py`, one ReturnnTrainingJob per
arm, per-sub-epoch recogs.
**Experiments.** Lambda sweeps per bed at ≤ 100 h, never 960 h; lambdas are bed-size-dependent
(§6.9/§6.10: off-seed the prior decides converge-vs-turn and its share grows with bed size) —
recalibrate against the within-group-std monitors, never carry values across beds.
**Gate.** Checkpoint selection by dev reward + LM score only; monitor reward components, ins/del,
and within-group std separately; a degrading run is reverted, not compounded.
**Status: LIVE only through the §3e.1 update-rule families.** The 10 h and 100 h rounds are
complete; the stock-donor 960 h 3-pass arm stopped at sub-epoch 4 and was deleted, while the adapted-
donor 960 h one-pass arm completed all ten sub-epochs. The fresh/warm and GAN periodic trajectories,
HOM arm, and exact GAN-frozen schedule control remain in flight; no endpoint verdict yet.

### 3e.1 Scorer trainability without collapse (USER-directed 2026-08-06; sub-plan `PLAN_3E1.md`)

**Purpose.** The bad-init north star needs a scorer that repairs itself in-loop; a scorer that
must start good imports the bootstrap problem into the reward. Both endpoints fail: frozen
Goodharts (2S) or cannot leave a contaminated prior (G-track), and training on the policy's whole
sample set collapses the scorer — the trainable 100 h replay arm (`freeze_ar=False`) went
18.79 → 46.71 by DRIFT off the gold domain, not text-blindness (re-diagnosed 2026-08-07,
`SAE_3E1.md` c1-2: its text contrast rose 86% while CE_true crossed the unit marginal and
uniform) — so the update *rule*, not trainability, is the question. Attribution of that
collapse to co-training itself is REOPENED 2026-08-09 (user question): no frozen-scorer
control has ever run on that bed and the 10 h matched pair went the other way — `PLAN_3E1.md`
D5 carried it: (a) forensics on the collapsed run's own checkpoints, (b) a USER-redirected
joint-psi control arm on the current best 960 h bed (the now-finished frozen arm is its matched
control).
**Approach.** The evidence splits the failure three ways (`PLAN_3E1.md`): ranking NOISE is
refuted (twice — recon within-group std, and in-group spearman ~0.50/0.56 at the loop's own
operating point); correlated BIAS is confirmed but ~70% is a psi_align FAMILY property (the
gold-text control also pays for the filler, beta 0.167 vs 0.243 — only the differential is
contamination, `SAE_3E1.md` c4); GROUP BLINDNESS is measured partial and binding (23%/9%
contrast coverage — ~77% of "to"-groups unsteerable for ANY scorer, c6). Admissible shape: discrete gated OFFLINE refresh rounds at sisyphus-job granularity
— no in-loop psi channel exists (`grpo/psi_scorer.py:153`), the loop always runs on the last
accepted frozen scorer, so rollback is free; ladder D0–D4 (discriminator → probes → round-0 text
repair without co-training → frozen-repaired control arm → gated outer refresh) pre-registered in
`PLAN_3E1.md`. Old candidates posterior-weighted CE and emissions-pinned text refresh are
withdrawn (published collapse mode; not a real parameter partition — one trunk feeds all heads).
**Experiments.** D0--D4 diagnostics and historical repairs are complete; D1's power check failed
and localized the insertion price as a scorer-family issue. D5 closed continuous co-training as
catastrophic on the best bed. D6's offline minimum-duration screen and one-shot frozen continuation
are complete; the fresh/warm periodic, GAN periodic, exact GAN-frozen schedule control, and GAN+HOM
endpoint trajectories are the only live experiments. Full configurations and gates are maintained in
`PLAN_3E1.md`; no old D2/D3 launch item survives. The superseded offline D7-v2 design closed before
scorer fitting.
Its feature census passed, but the registered K=4/Q2 common-regular training construction
failed its prospective support floor by two orders of magnitude (56 admitted rows versus 6,778;
necessary-core upper bound 120). The external instrument, loss calibration, scorer A/B and policy
leg were never reached. Active D7 drops graph matching entirely and runs the
matched scorer A/B on all 281,241 pseudo-labeled 960 h utterances with dynamic K=1 same-speaker,
duration-windowed donors. D7.0/D7.1 are implemented and launched (2026-08-21); D7.3 is not
authorized.
**Gate v2 (replaces the two-sided gate, 2026-08-07 — amended BEFORE any verdict was read against
v1, because v1's `text_explained_loo` arm is gold-conditioned as instrumented
(`config_sae_3a_enc50_units_v1.py:233-243`) and has the wrong sign against the filler mode, and
its held-NLL arm is a per-round redraw, not comparable across rounds).** Accept a scorer update
only if, label-free on a frozen external held pair set outside the candidate's curated pairs:
held unit NLL improves vs the last accepted scorer AND `text_explained_loo` ≥ the pre-loop floor
AND filler-contrast probes do not degrade AND paired rank stability vs the last accepted scorer
holds; `PsiScorerParityJob` before any live use. Full battery in `PLAN_3E1.md`; the gold-text G1
stays a reported diagnostic that can never flip a G-track decision.
**Status: ACTIVE; D5/D6 decisions supersede the old D0--D3 queue.** Continuous joint psi is closed
after 5.12/9.27 -> 17.35/21.97 -> 41.78/50.88. The one-shot `d_min=2` scorer-repair package passed
its matched continuation read, ending 4.73/9.31 against 6.46/11.41 with dev-other insertions 933 vs
1,964. Repeated fresh and warm scorer refits deteriorate through their current prefixes; the
label-free GAN periodic, HOM, and exact schedule-matched frozen-control arms are in flight; their
endpoints remain pending. Finish that GPU work and the §3d.A operator read without displacement.
D7-v2 closed at D7.0b without scorer or policy compute: exact admission was 56 rows/two
speakers and the necessary upper bound is 120/four versus the 6,778/201 gate. Do not retry the
offline graph. Corrected D7-GAN-SEQDISC is implemented and launched (2026-08-21,
verifier-confirmed) through its full-960 h matched scorer A/B, gated behind the D7.0 preflight
barrier, with no displacement of the already-funded trajectories or §3d.A operator read. Its
policy leg is held. 2026-08-21 latest: preflight PASSED under the amended parity rule; both
D7.1 trainings failed closed on four own-infeasible degenerate greedy anchors — drop-and-count
amendment registered in `PLAN_3E1.md` D7 Status; 2026-08-22 the edit is implemented and
verifier-confirmed, the user restarted 22:50, and D7.1 COMPLETED 23:05 on both arms
(verifier-confirmed; four named drops closed per-arm, fixed finals banked). D7.2 completed
2026-08-22 and FAILED clause 2: **D7 is CLOSED** per the registered gate (verifier-confirmed;
no policy leg, no rescue; verdicts 66-67). The verdict releases D8.1a-b as registered.
Normative details and exact operating points: `PLAN_3E1.md`; evidence: `SAE_3E1.md`.

### 3f. Exit gate (Rung 3)

**Purpose.** The central claim's acceptance test.
**Approach/Experiments.** Read on the loop's final arms from the identical bootstrap; the
Phase-4 probes and the 3-BT head-to-head are inputs to it.
**Gate (pre-registered, unchanged).** All of: (1) dev-other ≤ min(Rung 0, Rung 2) − 0.5 abs;
(2) the winning checkpoint is the one the **unsupervised** criterion selects; (3) sign reproduced
by a second RL seed; (4) stable over the last third, ins/del within 1.5× SFT, §4 probes clean;
(5) reported head-to-head vs Rung 3-BT — if RL loses, BT becomes the headline and RL the
reported negative arm.
**Status: NOT FIRED.** If (1) fails with §2.5 passed, the failure localizes to the loop (lambda
balance, scorer drift, anchor) — iterate there, not in Phase 1.

### 3g. Z-track — from-scratch fully-unsupervised joint loop (USER-directed 2026-08-12)

(Moved to `PLAN_3G.md` 2026-08-14 — replaces the inline block; all four registered arms are now
closed. Gate text was carried verbatim there.)
**Purpose.** Real unsupervised ASR without GAN: run the joint loop from zero paired data and
classify the failure mode against the pre-registered (A)/(B)/(C) taxonomy; labels evaluate
only.
**Approach / Experiments / Gate.** `PLAN_3G.md`; log `SAE_3G.md`.
**Status.** 3g.1 base arm CLOSED 2026-08-13, outcome (A) — mode collapse to one constant
sentence by step 346; the per-utterance joint objective's optimum sits at zero coupling.
3g.2 (Z2: diversity price + pseudo-pair init + derangement hinge) completed all six sub-epochs
despite the earlier stop directive — verdict: it escaped zero coupling via a nuisance-channel
ladder, duration then speech density, with no phone-content evidence. 3g.3 (Z3:
tempo/noise/pitch perturbation consistency + hardened negatives + raised lam_div) also completed
all six and FAILED its primary: the duration-matched gap remained negative throughout and final
WER was 94.87/96.14. Neither Z2 nor Z3 has the registered purity/PER read, so the formal B/C
taxonomy letter remains incomplete even though the controlled nulls strongly support nuisance/private
coding. 3g.4 (Z4: discrete psi refresh replacing co-training +
within-seq repetition price + lam_len activation; lam_lm kept at 1.0 units-norm by the
2026-08-15 ruling) REGISTERED AND FUNDED 2026-08-15. Z4 GATE VERDICT 2026-08-16 (six
rounds, planner-verified): FAILS — primary above bar only at round 1, speaker-meter
secondary fails as written, repetition price binds; the registered exhaustion reading does
NOT fire (within-group spread recovers past start), so this is a gate failure with
earnable variance remaining, not a loop that ran dry. No Z5 is funded: the planner recommendation is
to require a content-bearing §1g seed before any further no-pairs loop, rather than add another reward
term to the same content-free initialization. Detail: `PLAN_3G.md` 3g.4 Status.

---

## Phase 3B — Backtranslation branch (no RL; parallel)

**Purpose.** An RL-independent second shot at beating Rung 2 — upgrades the program's worst case;
independent of the §2.5 gate.
**Approach.** Unit-level iterative backtranslation between AV-U and the AR (shared token space).
Invariant: each model always trains toward a REAL target (real text for AV-U, real units for AR);
only sources are synthetic; ~50 % previous-round data retained; unsupervised stopping.
**Experiments.** 2–4 rounds; headline = distill-back (final AV-U decodes → feature-AV SFT =
**Rung 3-BT**).
**Gate.** ≥ 1 round of positive unsupervised-score gain, and Rung 3-BT ≤ Rung 2 − 0.5 abs,
unsupervised-selected.
**Status: NOT STARTED** (pending Phase 2).

---

## Phase 4 — Controls and ablations

**Purpose.** The paper's credibility section: leakage probes, attribution controls, ablations.
**Approach.** Probes (dev, frequent during Phase 3):

| Probe | Transform | Interpretation |
|-------|-----------|----------------|
| Orthographic | homophone swap and case/punctuation jitter | Report reconstruction, LM-prior, and composed-reward deltas separately; the live BPE scorer is not homophone-invariant, so this is an attribution diagnostic, not a pass condition |
| Word-boundary | resegmentation with the same lexical content | Report the delta; BPE states and explicit SIL boundaries may legitimately change |
| Content sensitivity | random BPE-distinct word substitution | More-negative reward is better; report scale relative to within-group reward std |
| Speaker leakage | linear speaker-ID probe on AV states, pre vs post RL | accuracy gain ≤ 2 abs |

**Experiments.** Remaining ablations at 100 h scale: scorer-frozen-vs-updated (now the §3e.1
program), lam_1 = 0, lam_2 = 0, pure-phoneme Option A, warm-start degradation sweep; plus the
confabulation check and the contamination control (log p_base of true dev transcripts vs
length-matched LM-corpus sentences; 4-gram-only prior deltas).
**Gate.** All probes reported; no numeric gate.
**Status.** Shuffled-reward control DONE and DECISIVE (2026-08-04, `SAE_2S.md` approach 19): the
reward is load-bearing. Probes and ablations pending on the Phase-3 endgame.

---

## Phase 5 — Refinement (gated on Rung 3 > Rung 0)

**Purpose.** Scale and side channels as measured deltas, then the headline decode.
**Approach.** (a) Qwen3-8B warm-started from the winning branch's pseudo-labels, shortened
rerun; (b) label-free speaker embedding (mean-pooled frozen states, else crop-contrastive
InfoNCE; usage-gated) + quantized F0/energy streams conditioning the AR; (c) 8B n-best
noisy-channel rescoring tuned on dev by reward.
**Experiments.** (a) then (b) then (c), each reported as a delta over Rung 3.
**Gate.** Rung 4 dominates Rung 3 with the side-channel delta isolated.
**Status: NOT STARTED** (gated).

---

## Deliverables ladder

| Rung | Claim | Must dominate |
|------|-------|---------------|
| 0    | bootstrap + self-training + WFST decode (standard recipe) | — |
| 1    | LLM rescoring of the same lattices | same-lattice 4-gram decode |
| 2    | AV SFT distillation of Rung 1 | Rung 1 |
| 3-BT | iterative backtranslation, distilled back, no RL | Rung 2 |
| 3    | reconstruction-reward GRPO, identical bootstrap | Rung 0, Rung 2; head-to-head vs 3-BT |
| 4    | 8B + side channels | best of 3 / 3-BT |
| S    | anchor arm: RL from {1 h, 10 h} seed vs self-training from the identical seed | separate supervision axis |

Publish from the highest rung that holds; the BT branch and Rung S hedge the RL and bootstrap
axes respectively. The SAE story survives either head-to-head outcome — both branches
instantiate the text-bottleneck autoencoder.
