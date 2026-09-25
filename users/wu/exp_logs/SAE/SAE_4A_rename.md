# SAE 4A -- rename: what phi training needs so that EM can correct wrongly named phones

## State

Created 2026-09-25 as a proposal phase (user request 2026-09-24: "carefully plan with literature review: how
should we adjust our phi training to make it possible. The rule for no new training arms stays, just propose in a
next_step phase file"; new analyses are allowed, including GPU). Nothing here is funded training.
- Every TP arm needs the user's OK, because each trains phi. TP0 was funded on 2026-09-25.

No manager or watcher is live. TP0 finished and is recorded (Results: EM LOCKED on the right partition, audited).
AN-1 V3 is recorded (NOT GOLD FIRST, NOT VISIBLE; no TP-C). AN-6 `InventoryCeilingJob.CmkLGGuJFZyg` is fixed but
still errored, and needs the user's `-co`.

NEXT:
1. The Proposal (TP-D, then TP-B'; audited, `reports/audit_rename_proposal_2026-09-25.md`) now carries TP0's
   registered consequence: item 4 brings back TP-A1, the sparse channel prior and TP-B''s name step. It waits for
   the user's OK and four rulings (secondary acceptance read, broad-class allocation, V2 occupancy term, TP-B''s name
   step).
2. AN-6 runs after the user's `-co`; then extract, audit and record it.

## Objective

The user's premise: phi EM cannot correct wrongly named phones. Find the change to phi training that would let it,
decide it with analyses on existing phis first, and propose training only as pre-registered arms for the user.
Ceiling of every proposal: the basin (phi generative PER 0.35-0.50; recognizer lift to 0.20-0.36).

## Constraints (inherited, not revisitable here)

- Pure unsupervised, GAN-free. Supervised or gold-derived phis are disclosed analysis inputs only, never an init,
  option or fallback of a proposed arm. Labels never train, select, or set a hyperparameter of a label-free arm;
  label-using analyses only decide which proposal goes to the user.
- Every LM term is trigram or higher, fitted on the uniform-sample window. d_min 2. Durations: general knowledge
  only (A9 durinit); MFA rates and durations are never priors.
- Final objective of any proposed arm: S (held-out tau = 1 NLL per frame on the 260 set) or a change the reads here
  justify. Always report generative PER (direct, Hungarian, NMI).

## Evidence ledger (all audited in `SAE_4A_lexlat_v2.md` unless marked)

| # | Finding | Source | Bearing on the premise |
|---|---|---|---|
| E1 | S's lower basin is phonetic: gold-EM 3.216, gold-key 3.207, r30 3.210, r70 3.271 against random-init EM 3.30-3.40; r100 3.378 does not reach it | A14 (ii), gold-key control | S prefers the right basin between basins |
| E2 | The basin is reachable from type-level labels with durinit durations and cell-free emissions (the key itself induces boundary F1 0.813 at sub-epoch 0) | gold-key control | no supervised segmentation is needed given the right key |
| E3 | Random-init EM finals are mislabelled and merged: own-label agreement at chance, 11-19 phones unclaimed, SIL/N/T/W/AY/Z/M take 3-4 symbols | A15, A15-F | the state the premise describes |
| E4 | Along A10 trajectories (one seed per setting) the partition's emission-map accuracy (R4 emis 0.29-0.31) is flat from sub-epoch 4 to 48, while own-name agreement T1 falls (0.150 -> 0.077, 0.230 -> 0.115) from 4 to 12 | A15 | own names get worse after sub-epoch 4 while the partition holds |
| E5 | Relabelling an EM final through its emission map, without refit, raises S by 0.12-0.31 | A16 (a) | S rejects a pure rename of a co-adapted phi; refit untested |
| E6 | Up-weighting the trigram in scoring (S_lambda, lambda <= 3) does not rank gold above EM finals; gold was not refit there | A16 (a2) | scoring only; lambda in the E-step during training is untested |
| E6b | The S_lambda slope puts the LM cost at about 0.58 nats per frame for gold and the finals alike; the gap lies in the channel | A16 (a2) audit | scaling the LM cannot re-rank existing phis; says nothing about the trajectory from random init |
| E7 | Key level: all 124 stage-1 finals beat gold on J; J penalises the gold-matching 1:1 names by 0.42-0.57 per frame; the found keys are swap-optimal | stage 1, A20 | at key level the objective, not the search, rejects right names |
| E8 | On the gold partition J penalises scrambled names by 0.69-1.09 | A20 audit, unregistered | the trigram carries a strong name signal on a clean partition |
| E9 | J's name-free margin over gold (0.20-0.23) is mostly less d_min absorption (0.10-0.17), then symbol entropy (0.06-0.10) | A20 audit, unregistered | J's defect is a hard-key artefact that phi's soft channel may not share |
| E10 | Found keys fit the trigram better than gold, per frame (-0.948 at 9.20 Hz against -0.990 at 9.47 Hz) and per phone (-5.15 against -5.23); a hard key pushes unit noise (purity 0.644) into gold's strings | A20 | raising the LM weight at key level favours wrong keys; H3's "lose on LM per phone" is false at key level |
| E11 | Inside the basin EM drifts: gold-init PER 0.19 -> 0.35 while S falls; tau = 1 from the start removes about a quarter | A14 (ii), A17 (ii) | S does not reward accuracy inside the basin |
| E12 | r70 (random label noise, per-unit argmax right) improves under EM, PER 0.61 -> 0.39-0.50 | A14 (ii), A17 (ii)/(iii) | denoising, not renaming |
| E13 | From the gold-key partition with deranged names, 12 sub-epochs of EM keep the wrong names (DELTA +0.024 / +0.013 / -0.005) and break the partition (best 1:1 identity 0.52-0.65, control 0.78) | TP0 (Results) | the premise holds on the right partition |
| E14 | Rates on two conventions: A10 finals 6.8-7.6 Hz (A8 greedy emitted rate); basin arms 10.0-10.5 Hz and r100-EM 9.13 Hz (Viterbi segment rate, `SegmentationBoundaryJob.g52gIGtUHpkr`). Key level: found 9.2-9.7 against gold 9.47 | A10, keyinit audit, A20 | a phi-level rate gap is not established on one convention (AN-4); none at key level |
| E15 | Gold phi under a fixed name permutation (permphi): decode-based Hungarian PER 0.819, its decode map recovers 16 of 40 labels (A15-E registration's build test); the emission map recovers 40 of 40 | A15, A15-E | renaming must be measured through the emission map or key identity, never decode PER |
| E16 | Basin phis lift the recognizer (A17 (i) 0.20); EM finals do not (A14 (i) 0.82-0.84; A18 (a) 0.842) | A17 (i), A14 (i), A18 (a) | fixing names is worth the work |
| E17 | Count-table EM from Dirichlet random tables stops at iteration 2-3 (gain < 0.01), S 3.536-3.608 on the 285 holdout; its frame-permuted nulls fall out of the rate band, so the family read CANNOT_TELL | A11/A12 read | the table operator locks in almost at once |

Status of the premise: tested directly on the right partition by TP0 (E13), where it holds; indirect elsewhere (E3,
E4, E17). Between basins S is right (E1); within J it is
wrong (E7). The phi-level question is therefore search (EM cannot leave a wrong-name optimum that S ranks higher)
or coupling (the partitions EM reaches are ones on which the right names do not win under S).

## Hypotheses (phi level)

- **H1 lock-in.** The per-frame channel dominates the E-step once emissions sharpen (about 4-5 dependent frames per
  segment, each counted as independent evidence), so the posterior cannot move a whole symbol to another name.
  EM is local; a rename is a coordinated permutation (a QAP under a bigram or higher, Nuhn & Ney 2013). Predicts:
  EM from permuted gold stays permuted at lambda = 1; an LM-weighted E-step moves a swapped symbol back when its
  context is mostly right.
- **H2 weak name signal under S.** S barely separates right from wrong names even on a clean partition. E8
  (key level, 0.69-1.09) argues against; AN-2 measures it at phi level.
- **H3 coupling.** The name-free channel shapes partitions that duplicate frequent phones, starve rare ones and
  possibly under-segment (E3; E14 unverified); on those partitions no naming is near right, so neither a rename nor
  a stronger LM helps until the partition changes. Variant H3' (private code): the found partitions also win on
  the LM per phone (as at key level, E10), so every LM-side lever pulls toward them.

H1 and H3 can both hold; the analyses separate their sizes.

## Analyses (registered 2026-09-25 before any job, amended by the design review; label-using where marked; no training)

Shared inputs: the 260 set and the S reader of A14 (ii); the gold-key phi `PhiFromKeyInitJob.f0jaGuiJVe6A`; the
stage-1 selected keys and their A20 variants (`KeyOracleNameReadJob.QD3S2xcmg0Wt/output/keys`); the six A10
sub-epoch-48 phis; the A14 (ii) and keyinit (ge1MKcAPmZIV) arms at 0/4/12/48. Permutations are derangements of the
non-SIL symbols (seeds 1-3), asserted fixed-point-free: 1-pair, 5-pair and full. Key identity: the key is
argmax_s m(u|s) pi(s) with pi the expected symbol frame share from the same E-step (or forward); identity is its
frame agreement with the gold key, weighted by train unit counts as in A20 (decode PER cannot see renaming, E15).
Constants: 0.01 is A7's floor; 0.05 is 5 x A7's floor.

**Amendment R1 (2026-09-25, at the AN-0 launch review `reports/review_rename_an0_launch_2026-09-25.md`, before
any AN-0, AN-2 or AN-3 result).** The reviewer's 24-utterance probe on the undeployed gold-key phi (not a
registered row) showed that `prior_weight` scales the per-token prior cost and so acts as a token penalty:
expected tokens per frame 0.236 / 0.215 / 0.159 / 0.067 and gold identity after one step 0.980 / 0.973 / 0.955 /
0.518 at lambda 1 / 2 / 4.4 / 10. A read at lambda = 10 is therefore decided by segmentation collapse, not by
names, and the plain P_LM^lambda form confounds names with rate in AN-2 and AN-3 as well. Amended:
- Two lambda forms everywhere lambda appears: *plain* (P_LM^lambda) and *rate-neutral* (P_LM^lambda times
  exp((lambda - 1) H_LM) per token, i.e. the LM log-probability scaled about its mean; H_LM is the frozen trigram's
  mean per-token log-loss on its own uniform-window text, text-only and label-free, printed by the job). This is
  the ASR convention of an LM scale paired with an insertion term. It is built by modifying the prior table passed
  to the lattice, not the lattice code.
- Gold-row guard: each (lambda, form) of AN-0 and AN-3 also runs the undeployed gold-key phi through the same step.
  The (lambda, form) is VALID only if the gold row's key identity after the step is at least 0.95 (a step that
  damages right names by more than the 0.05 restoring bar cannot show restoration). Tokens per frame and SIL share
  are printed for every row.
- Primary AN-0/AN-3 statistic: restore = the train-unit-weighted frame share whose key moves from a wrong name to
  its correct name. rho (net identity change) and many-to-one are reported beside.
- Gold-key derangements draw only from non-SIL symbols that hold units in the gold key (OY and ZH hold none; a swap
  onto an empty uniform row is not a rename).
- S_1 before and after a step use the same smoothing.

- **AN-0 (GPU, a few minutes; label-using) Pre-check for AN-3.** The gold-key phi under the 5-pair derangement,
  seed 1; one E-step on 300 fixed train utterances at lambda in {1, 2, 4.4, 10} in both forms (A12's smoothing,
  0.9 table + 0.1 uniform), one closed-form M-step, table discarded; restore, rho and the gold-row guard as in
  R1. **AN-0 OPEN** if restore >= 0.05 at some VALID (lambda <= 4.4, either form); **AN-0 DEAD** if restore < 0.05
  at every VALID lambda in {2, 4.4} and at least one of them is VALID: no LM-weighted E-step moves a whole symbol
  even in a right context, TP-A1's mechanism is dead, AN-3's grid is dropped, and AN-2 and AN-4 run alone;
  **AN-0 CANNOT TELL** if neither 2 nor 4.4 is VALID in either form (then TP-A1 is not dropped on AN-0 and AN-3
  runs with the guard). lambda = 10 is reported only.
- **AN-0b (GPU, minutes; label-using; registered 2026-09-25 after AN-0's result and before its own run)
  Where the LM pressure goes.** This follows AN-0's audit (Results): the LM moves swapped units' mass to SIL and
  to the empty uniform rows, not to the right name.
  - Hypothesis H1b (escape): in the E-step the LM pressure is relieved through SIL and through symbols that hold
    no units in the key, because those are cheaper in the channel than the right name's populated row.
  - Inputs as in AN-0: gold-key phi; the seed-1 5-pair derangement; the same 300 utterances; one E-step and one
    M-step; the table is discarded.
  - Cells: lambda in {1, 4.4} x {plain, rate-neutral}, with lambda 1 shared. Each cell runs two operators on the
    deranged row and the gold row:
    - as-is (AN-0's step);
    - no-escape: symbols holding no units in the phi's key (OY, ZH) are masked out of the E-step lattice.
  - A seed-1 1-pair row is added as a descriptive row. It carries no reading and does not reopen AN-3.
  - Printed per cell, row and operator: restore, rho, the gold-row guard, SIL share, tokens per frame, and the
    split of swapped units' posterior frame mass into right name / partner name / SIL / masked or empty rows /
    other. The job also saves N(s,u).
  - MAP paths of 5 fixed utterances in the as-is rate-neutral 4.4 cell, for both rows, each re-scored by an
    independent re-add of its terms.
  - Readings:
    - **E-STEP CHECK FAILS** if any re-added path score differs from the lattice's by more than 1e-4 relative.
      Then the E-step is debugged before any other reading. Launch review
      (`reports/review_rename_an0b_launch_2026-09-25.md`), read-time rules fixed before the run:
      - the check passes only if all 10 paths were extracted and each has bridge_rel_diff <= 1e-4 (the check
        runs at tau 1e-7 elementwise, the E-step at tau 1 by matmul);
      - otherwise ESCAPE is not read.
      - The best path is taken through the posterior support (mass > 0.01) because exact posterior ties occur.
      - A pass shows correct path scoring, not a verified N(s,u) at full size.
    - **ESCAPE BLOCKS THE STEP** if no-escape restore >= 0.05 at a VALID lambda 4.4 cell (either form).
    - **ESCAPE NOT THE BLOCK** if no-escape restore < 0.05 at every VALID lambda 4.4 cell.
    - **SIL ESCAPE** (descriptive) if, under no-escape at rate-neutral 4.4, SIL takes more of the swapped units'
      mass than the right name does.
  - Consequence: AN-0 DEAD and its registered consequence stand. ESCAPE BLOCKS THE STEP is brought to the user as
    a mechanism finding, disclosed as found after AN-0, with its training counterpart as a candidate. ESCAPE NOT
    THE BLOCK records H1 at the sharp-channel operating point.
- **AN-1 (CPU, key level) Objective screen.** J variants on all stage-1 finals, the A20 (a)/(b)/(c) keys, gold,
  K30/K70/K100, and 3 random non-SIL derangements of each selected key's (a) names:
  - V1: emission without d_min absorption (on the unabsorbed symbol string).
  - V2: J plus Kearns, Mansour and Ng's per-frame class-prior term over non-SIL frames, (1/T) sum_t log
    pi_LM(s_t | non-SIL), pi_LM the trigram's unigram restricted to non-SIL and renormalised. SIL is excluded,
    because the stage-0 amendment found the SIL occupancy prior puts 0.487 on SIL against a 0.058 frame share.
    Analysis only; its use in training needs a ruling under the trigram-or-higher rule.
  - V12 = V1 + V2 (E9: removing one part cannot put gold first if the other exceeds the remaining margin).
  - V3 (4-gram and 5-gram priors) needs a new prior fit (`prior.py` fits orders 1-3) and a new LM term in
    `unit_key.py`; built only if the implementer's estimate is under one day, else dropped.
  - Readings: **GOLD FIRST under V** if J_V(gold) exceeds every final and every A20 variant by more than 0.01 and
    gold > K30 > K70 > K100. **NAMES VISIBLE under V** (V2 and V12 only; V1's emission is name-free, so its rename
    difference equals J's) if J_V(b) - J_V(a) > 0.01 on at least 3 of 4 selected keys; the random-rename rows are
    reported beside. The stage-1 null keys are scored on their own corpus, as in stage 1, and enter no reading.
  - Also registered: the train-side split of J's emission gap into H(S) - H(U) and absorption (E9), printed by the
    reader with its convention.
- **AN-2 (GPU forwards; label-using) Does S see names, and does lambda move the ranking?** S_lambda =
  -(1/T) log sum_y P_LM(y)^lambda p_phi(x | y), lambda in {1, 2, 4.4, 10} (`prior_weight`; a new reader module).
  Phis: the gold-key phi with gold names and each derangement; the stage-1 key phis under (a) found names,
  (b) oracle 1:1 names, and 3 random derangements of (a); the six A10 finals; the durinit basin set (gold-key, G-dur
  and r30-dur at 48), with r70-dur and the A14 (ii) MFA-duration arms reported beside. Readings:
  - **H2 REFUTED** if S_1(gold-key) < S_1(full derangement) - 0.1 on all 3 seeds; **H2 SUPPORTED** otherwise, which
    contraindicates TP-A1 and TP-B.
  - **S SEES NAMES ON FOUND PARTITIONS** if S_1(b) - S_1(a) is below the minimum over the 3 random renames minus
    0.01 on at least 3 of 4 keys; **S PREFERS FOUND NAMES** if S_1(b) > S_1(a) + 0.01 on at least 3 of 4;
    **S NAME-BLIND ON FOUND PARTITIONS** if S_1(b) lies inside the random-rename band on at least 3 of 4; else MIXED.
  - The basin's lead at lambda = min over the six finals minus max over the durinit basin set. **LAMBDA KEEPS THE
    BASIN** if the lead is positive at every lambda in the grid; **LAMBDA FAVOURS FINALS** if it shrinks
    monotonically with lambda and turns negative by 10. Read on the rate-neutral form (R1); the plain form is
    reported beside, since its token penalty favours the lower-rate phis whatever their names.
- **AN-3 (GPU E-steps plus CPU M-step; label-using) Does the EM operator restore names?** From each deranged
  gold-key phi and each stage-1 key phi (a): one exact E-step (`blankfree_emtable.e_step_batch`, 1,000 fixed train
  utterances, tau = 1, A12's smoothing, lambda in {1, 2, 4.4, 10}), one closed-form M-step to a type-level table,
  then discard. Measure rho (the change in key identity), the change in many-to-one agreement, and S_1 after the
  step. Readings on the 5-pair derangements (3 seeds); the full derangements are the many-wrong-names bound; the
  1-pair rows and stage-1 keys are reported beside:
  - Both lambda forms and the gold-row guard of R1; only VALID (lambda, form) cells are read.
  - **LOCKED AT 1** if restore < 0.01 at lambda = 1 on all 3 seeds.
  - **LAMBDA OPENS at L (form)** if restore >= 0.05 on all 3 seeds at lambda = L in that form with many-to-one
    falling by at most 0.05. Opening only at 10 is reported and brings no proposal (no label-free lambda0 above
    4.4).
  - **MANY-WRONG LOCKED** if restore < 0.05 at every VALID cell on the full derangements (then no lambda helps the
    stage-2 regime).
- **AN-4 (GPU forwards; label-free statistics, PER reported) Term anatomy of S.** Per utterance on the 260 set:
  E_q[log P_LM] per frame as -dS_lambda/dlambda by a symmetric difference at lambda = 1 +/- 0.05, and per phone;
  E_q[log p_phi(x | y)] per frame from seg_post and the segment scores; H(q) per frame by subtraction from -S;
  expected non-SIL rate, SIL share and E[d] from seg_post (one convention for every phi); decoded unigram and bigram
  KL to the prior's marginals; I(U; S) of the posterior-weighted unit-symbol table. Phis: the A10 finals and their
  trajectories at 4/12/48, the A14 (ii) and keyinit arms at 0/4/12/48, stage 2's arms when finished. Predictions
  fixed now, for the six finals against the durinit basin set at 48:
  - P1a: every final has a higher channel term per frame. P1b: every final has a lower LM term per phone.
    P1c: every final has a lower expected non-SIL rate.
  - P1a and P1c with P1b failing (the finals win the LM per phone) is the private-code outcome H3'; it
    contraindicates TP-A1 and TP-B swaps.
  - P2: the LM term per frame differs between finals and basin by less than half the paired S gap of the same phis.
  - P3 (E11): along the basin trajectories, the channel term rises and the LM per phone falls as PER rises.
  - R1 scope (2026-09-25, at the launch review `reports/review_rename_an24_launch_2026-09-25.md`, before any AN-4
    result): AN-4's lambda is a measuring device at lambda = 1, not an operating point. Its registered quantity
    is E_q[log P_LM], which only the plain form's derivative gives, so AN-4 uses the plain form only. The
    rate-neutral derivative would add H_LM times the token rate: a constant per phone, so P1b is unchanged; per
    frame it is recoverable from the printed rate columns. P2 is read on the plain term as registered, and the
    rate difference is reported through P1c.
- **AN-5 (GPU forwards; label-using; owed by the keyarms review) Name tracking in stage 2.** A15-F measures
  (own-label agreement, claimed and unclaimed phones, duplicates, R4) and key identity on the gold-key arm and the
  four stage-2 key arms at 0/4/12/48. **EM RENAMES** if identity at 48 minus identity at 0 is at least 0.2 (K70's
  identity 0.30, the lowest from which EM is known to reach the basin, minus the found keys' 0.07-0.14) on at least
  2 of 4 key arms; **EM LOCKED** if at most 0.05 on at least 3 of 4; else PARTIAL. EM RENAMES licenses "stage 2 is
  the route" only with KEY BASIN (S at 48 < 3.289) on the same arm.

Cost: AN-2 and AN-4 are about 180 gpupack forwards at about 20 s of GPU each, AN-3 is 28 E-steps on 1,000
utterances (about 16 GPU-minutes); 1-2 GPU-h in all, limited by job count. AN-1 is one login-node job
(10-20 minutes); AN-5 is the A15-F reader on 16 checkpoints.

**Pre-decision analyses (added 2026-09-25 after the proposal audit, on the user's question whether every needed
analysis has run; registered before any job; CPU, no training):**
- **AN-1 V3 (as registered above; it was never estimated or built).** Delta from `KeyObjectiveScreenJob.erCoRAmZID5a`:
  J with its trigram LM term replaced by a 4-gram (V3a) and a 5-gram (V3b). Both are fit on the same uniform-sample
  window with the trigram's estimator extended in order. Everything else is unchanged: keys, corpus, split, emission,
  duration and absorption. Check: order 3 through the new path reproduces V0 (A20's J) to 1e-9. The readings are
  AN-1's GOLD FIRST and NAMES VISIBLE; NAMES VISIBLE applies here, because the LM term carries names. The consequence
  is decision-table row "AN-1 GOLD FIRST under some V": TP-C (the V3 term in J and S) beside the proposal.
- **AN-6 (label-using, report only) TP-D ceiling.** Second-level k-means on the 500 centroids to K in {100, 200, 300}
  (seeds 1-2, unweighted as TP-D specifies; frame-weighted reported beside), plus the 500 units themselves. For each
  inventory, against the gold frame alignments: many-to-one frame accuracy (the majority map fit on train, read on
  train and on the 260 set), PNMI, phone purity, and the number of phones that are some class's majority. It is shown
  with the TP-D request and decides nothing. It measures how much phone information each coarse inventory keeps
  before TP-D is funded.
- **TP0, funded by the user on 2026-09-25 (analysis only, label-built init; amended before any job).** The user was
  asked whether to run it now and chose "Run TP0 now". Reason: AN-5 starts from found partitions, so it cannot
  separate partition from names, which is TP0's registered trigger.
  - Arms, one 4-GPU pack, A17 (ii)'s recipe (tau = 1, 12 sub-epochs), each starting from the gold-key phi (stage 2's
    key-to-phi conversion, durinit):
    - full derangement, seed 1;
    - full derangement, seed 2;
    - the 5-pair derangement of AN-0 (seed 1);
    - a no-derangement control.
    Derangements are built exactly as in AN-0.
  - Measured at 0/4/8/12: key identity (AN-5's rule); the best 1:1 renaming's identity, a partition-only measure;
    A15-F; S on the 260 set; generative PER.
  - **EM RENAMES** if identity(12) - identity(0) >= 0.2 on at least 2 of the 3 deranged arms. **EM LOCKED** if it is
    <= 0.05 on at least 2 of 3. Otherwise PARTIAL. The control's identity drift and each arm's best-1:1 identity are
    reported beside, so that name changes can be told apart from partition drift.
  - Consequence:
    - EM LOCKED: EM keeps wrong names even on the right partition. Fixing the partition alone is then not enough.
      The naming levers (TP-A1, the sparse channel prior) and TP-B''s name step go back to the user beside the
      proposal; row 2 dropped them on the evidence of found partitions only.
    - EM RENAMES: the found partitions' lock is a partition fault, and the proposal stands as written.
    - PARTIAL: reported beside, with no change.

## Proposed training changes (not funded; each needs the user's OK)

- **TP0 (label-using diagnostic, supervised init, analysis only) Direct test of the premise.** phi EM from the
  gold-key phi under a full derangement (seeds 1-2) and a 5-pair derangement, A17 (ii)'s recipe (tau = 1,
  12 sub-epochs). EM RENAMES / EM LOCKED as in AN-5, S and genPER at 0/4/8/12. If AN-3 reads LAMBDA OPENS at
  L <= 4.4, the same arms under TP-A1's schedule show whether the opening survives iteration. Worth running only if
  AN-5 reads PARTIAL or cannot separate partition from names.
- **TP-A1 (label-free) LM-led E-step continuation.** The E-step's LM exponent (`prior_weight`) is 4.4 at the start
  and falls linearly to 1 over the first 12 iterations (Form 1) or sub-epochs (Form 2), then stays at 1; the final
  objective is S unchanged. The form (plain or rate-neutral, R1) is the one AN-0/AN-3 open in; with the plain form
  the schedule also lowers the segment rate early (R1's probe: tokens per frame -33 % at 4.4), which the read must
  report. The duration term is not scaled: under durinit it carries no name information, and the
  lever is the Knight/Yin P_LM^lambda form. lambda0 = 4.4 is A9's general-knowledge mean phone length in frames,
  from the frame-dependence argument (supervised ASR compensates dependent frames with an LM scale of about 16,
  Wegmann and Gillick 2010; cipher EM succeeds when the channel starts flat and the LM leads, Ravi and Knight 2008,
  2011). It is never tuned on labels; the finals' own E[d] of 6.0-6.2 means it compensates only in part.
  - Form 1 (table, first; about 1-2 GPU-h): A12's stage A/B protocol, arm (c) (durinit, tau 1), with the single
    delta prior_weight = lambda(iteration) and no convergence stop before iteration 12 (E17: the table EM otherwise
    stops at 2-3). Read against A12's banked arm (c) finishers: S on the 285 holdout, key identity (disclosed),
    generative PER, and the rise over one seed of the same schedule on A12's frame-permuted corpus (A7's form),
    reported with its rate. **TABLE LM-LED LOWER** if the best finisher's S is below real_c_s16's 3.536 by more
    than 0.01 and its key identity exceeds A12's finishers'.
  - Form 2 (neural), only if Form 1 reads TABLE LM-LED LOWER: the A10 durinit recipe with the tau = 4 sub-epoch
    removed (with it the effective LM exponent at sub-epoch 1 is 4.4 / 4 = 1.1), seeds 1-3, plus a lambda = 1 arm
    with the same tau = 1 schedule (seed 1) and one seed on the frame-permuted corpus. **LM-LED BASIN** if S at 48
    < 3.289 on at least 2 of 3 seeds; this licenses the A14 (i)-form lift pack only; a claim about names comes from
    the lift read and the AN-5 measures, not from S. Report genPER at 0/4/12/48, rate, and the phi_c comparison
    ("not beyond private code").
  - Contraindicated if AN-2 reads H2 SUPPORTED or LAMBDA FAVOURS FINALS, or AN-4 reads H3'.
- **TP-A2 (label-free) Rate floor.** A penalty on expected non-SIL phone rate below a general-knowledge band
  (source to be verified by literature, never MFA's 11.9 Hz). Only if AN-4 P1a and P1c hold.
- **TP-B (label-free) Relocation moves under S with refit.** Every 4 sub-epochs, move a duplicated symbol (two
  symbols with near-identical emission rows) to the region of a starved LM symbol, and accept on held-out S after
  2 sub-epochs of refit (Jain and Neal 2004; SMEM, Ueda et al. 2000; random swap with 2 iterations, Franti). The
  swap half is dropped: a renaming under a trigram is a QAP, and the stage-1 swap climb moved away from gold (A20
  audit). Relocation is the literature-supported move for duplicated and starved states.
- **TP-C (label-free) Key route under a repaired J.** Stage-1 search under a J_V that reads GOLD FIRST in AN-1,
  then stage-2 phi EM from its keys. GOLD FIRST is necessary, not sufficient: TP-C carries stage 0 and 1's own
  gates again under J_V (J_V sees the key; whether any J_V final beats gold).
- Handoff items registered by A12 on NO SIGNAL, not proposed here yet:
  - coarse-to-fine unit inventory (500 codebook vectors clustered to about 100 classes, label-free): attacks H3
    directly; brought with TP-A2/TP-B when AN-2 reads the partition as the lever;
  - a word-level LM in the E-step with a wider beam: allowed by the order rule, but the k2 lexicon term worsened
    the bridge (A18 (a) +0.092) and ESPUM found higher orders hurt; not before an objective screen shows it
    ranks gold first;
  - a sparse channel prior (Ravi and Knight 2011: Bayesian 95.2 % against EM 32.6 % on a homophonic cipher): the
    strongest published fix for EM stuck on a cipher, untested at 500 noisy units; a candidate beside TP-A1 in
    Form 1's table.
- Needs a ruling before any use in training: the occupancy term (V2), a unigram-level term beside the trigram.

## Decision table (fixed before any result)

| Reads | Proposal brought to the user |
|---|---|
| AN-3 LOCKED AT 1 and LAMBDA OPENS at L <= 4.4 on the 5-pair row; AN-2 H2 REFUTED and LAMBDA KEEPS THE BASIN; AN-4 not H3' | TP-A1, Form 1 first (TP0's lambda arms as its mechanism check) |
| AN-2 S PREFERS FOUND NAMES or S NAME-BLIND ON FOUND PARTITIONS, with H2 REFUTED | the premise holds through the objective: S, like J, ranks wrong names first on the partitions EM reaches. Rename levers dropped (TP-A1, TP0's lambda arms). Bring TP-A2 if AN-4 P1a and P1c hold, TP-B, and the coarse-to-fine inventory |
| AN-4 P1a and P1c hold | TP-A2 (AN-2's lambda read reported beside) |
| AN-1 GOLD FIRST under some V | TP-C beside the above |
| AN-3 LAMBDA OPENS only at lambda = 10 | reported; TP-A1 not brought |
| AN-5 EM RENAMES with KEY BASIN | the premise fails for found partitions: stage 2 is the route; no TP |
| AN-2 H2 SUPPORTED, or none of the above | report NO LEVER SUPPORTED; name what would settle it |

Every proposal's best case is the basin (phi PER 0.35-0.50; lift to 0.20-0.36), not gold.

Precedence amendment (2026-09-25, at the AN-2/AN-4 audit, before any AN-5 job and before stage 2's KEY BASIN
read; source `reports/audit_rename_an24_2026-09-25.md`, finding 8). Rows 2 and 6 can both fire, and the table
had no precedence rule. Row 6 takes precedence, because it measures what EM does from the found partitions, while
row 2 reads the objective on those partitions before EM.
- If row 6 fires, no TP is brought. Row 2's read is then reported as a property of the pre-EM key partitions.
- If AN-5 reads EM LOCKED, PARTIAL, or EM RENAMES without KEY BASIN on the same arm, row 2's proposal stands. On
  PARTIAL, TP0 is listed beside it, as TP0's bullet says.

## Proposal to the user (2026-09-25, after AN-0 to AN-5; decision-table row 2; nothing funded)

Audit: `reports/audit_rename_proposal_2026-09-25.md`, CONFIRMED_WITH_CORRECTIONS; corrections applied below.

**Answer.** In every regime measured, EM keeps the names it starts with: one LM-weighted E-step on the right
partition with swapped names (AN-0, AN-0b), and 48 sub-epochs from the four found partitions (AN-5, one seed each).
On those found partitions S itself prefers the found names to the best 1:1 naming (AN-2). The registered table
(row 2) therefore does not fund the naming levers, and this proposal targets the partition, the only place where S
has been shown to see names. Evidence (Results, all audited):
- On the right partition, S sees names. Deranging the gold key costs 0.51-0.57 nats per frame (AN-2, H2 REFUTED).
- On the four partitions the label-free key search finds, scored as key-built phis before EM, S prefers the found
  names over the best 1:1 oracle naming by 0.25-0.35 (AN-2). This largely re-expresses A20: the found names were
  climbed on J, and the keys were selected on the same 260 set. These partitions duplicate 9-11 gold phone types and
  leave 12-13 unclaimed as built, and 7-11 and 8-12 after 48 sub-epochs of stage-2 EM; the gold-key arm has 1 and 1
  at 48 (A15-F in AN-5, label-using, report only). Their best 1:1 renaming reaches identity 0.43-0.49 against
  0.05-0.17 as named, so the names are a large part of the fault.
- EM does not rename in the regimes measured:
  - One LM-weighted E-step (lambda <= 4.4, either form) from the gold-key phi's sharp channel with 5 swapped pairs
    (seed 1, 300 utterances) restores at most 0.0053 of train frames (0.013 of the 0.41 that were swapped) (AN-0).
    Masking the empty OY/ZH rows does not change this (AN-0b); SIL was not masked. Iterated EM, a flat-channel start
    and other seeds were not tested.
  - 48 sub-epochs from the four found keys leave key identity flat, with changes of -0.040 to -0.001 (AN-5, EM
    LOCKED). Keys still change on 28-51 % of frames.
  - 12 sub-epochs from the right partition with deranged names (TP0, EM LOCKED; added after this proposal was
    audited): identity changes by +0.024 and +0.013 (full derangement, 2 seeds) and -0.005 (5 pairs). EM breaks the
    partition instead: best 1:1 identity falls from 1.0 to 0.52-0.65, against 0.78 for the underanged control.
  - The best arm meanwhile reaches the S basin bar at PER 0.86 (KEY BASIN, on S only; its 0.016 margin is below the
    A10 seed spread 0.025-0.071, one seed). Bridged into joint training it does not lift (dev-other PER 0.843 at
    ep8; A18 (c) in `SAE_4A_lexlat_v2.md`, audited).
- The finals are not a private code the LM favours. They lose to the basin on both the channel term and the LM term
  (AN-4, P1a False, P1b True). Under the registered table TP-A1 is not funded: AN-0 DEAD dropped AN-3, so row 1
  cannot fire, and row 2 drops the rename levers. TP-A2 is not brought, because its condition needs P1a, which is
  False. P1c is True (6.85-7.56 Hz against 8.42-8.55 Hz). Neither lever has been shown not to work.
- The split-merge and mixture papers read for this phase treat duplicated and starved states as a search error when
  the global optimum is right, and fix them with partition moves or restarts (`reports/lit_partition_levers_2026-09-25.md`:
  Ueda et al. 2000; Zhao et al. 2012; Jain and Neal 2004; Hughes et al. 2015; Berg-Kirkpatrick and Klein 2013; Jin
  et al. 2016). Objective-side fixes exist elsewhere with mixed results (sparse priors: Johnson 2007; Ravi and Knight
  2011; posterior regularisation did not help, Ganchev et al. 2010). Whether S's optimum is right is shown only
  between basins (E1), not within them (E11). Gaps: no published move works on a named, fixed inventory; none
  accepts on held-out likelihood; SMEM's mean LL was below repeated EM on 6 of 7 sets (Zhao et al. 2012); the one
  held-out read found went the wrong way (SMEM 2000 toy); no published work shows that coarsening makes naming
  recoverable.

**Proposed (each needs the user's OK; gates are fixed at registration, before any job):**
1. **TP-B' Named merge + split moves under held-out S.** This amends TP-B after the literature, before any TP-B job.
   It stays inside TP-B's relocation-only lever (fixed M). Disclosed: its name choice is a naming decision accepted
   under S, and on found partitions S ranks the wrong names first (AN-2).
   - Starts: the two durinit A10 finals and the four stage-2 key arms at 48 (all durinit, label-free). The A10
     uniform-duration restarts (A9: reported only, never chosen) and durfrz restarts are excluded.
   - Move: SMEM's fixed-M structure with a naming step added.
     - Merge a duplicate pair, ranked by emission-row similarity, into one symbol. This frees a name. (Ueda eq. 8
       is a posterior inner product; row similarity is an analogue.)
     - Split a symbol whose units fall into two groups by their decoded left and right symbol contexts. SMEM's
       local-KL split is degenerate for free multinomial rows (the literature agent's inference).
     - Give the new half an unclaimed name, ranked by how well its decoded trigram contexts fit under the LM, and
       try the top 2-3. "Unclaimed" is defined from the phi's own decoded occupancy against the uniform-window LM
       unigram (threshold fixed at registration), never from A15-F's gold counts.
   - Refit: the affected rows and their durations first (Ueda eq. 7), then full EM. Before fixing the refit length,
     check on the first moves that S has flattened by sub-epoch 2.
   - Acceptance: paired per-utterance held-out S on the 260 set, above the measured same-config S spread.
     - Each round evaluates 3-5 candidate triples, packed on one node, and accepts the best that clears the bar.
     - Stop when none clears. The published schedules try far more than one candidate per 4 sub-epochs (Zhao et
       al.: more than M^2 trials).
   - Reported beside, never used for acceptance:
     - label-free: the duplicate-pair count, occupancy KL to the uniform-window LM unigram (SIL excluded, AN-1's
       lesson), the unclaimed-name count, LM log-prob per decoded phone (AN-4's P1b separator), the non-SIL rate,
       cross-seed key agreement. LM per phone and rate stay report-only: they are in the battery because they
       separated label-built from label-free phis (AN-4), so gating on them would be label-informed selection;
     - label-using, report only: generative PER and key identity (AN-5's rule).
   - Needed at registration (the S-only draft read was implied by acceptance, had no control and could not show
     that names are corrected):
     - a no-move continuation control per start (the same EM continuation without moves);
     - a held-out read set disjoint from the 260 acceptance set, since acceptance makes 36-60 selections on it;
     - which spread sets the bar: the A10 exact-rerun band (3.4e-5) or the seed spread (0.025-0.071). The choice
       decides the outcome and is fixed before any job;
     - a label-using name read as the decision read: key identity (AN-5's rule) of the chain's final against its
       no-move control. It decides what goes to the user and selects nothing inside the arm.
   - Cost, from A10 (2.80 h for 48 sub-epochs on one GPU, about 3.5 min per sub-epoch): at 2 refit sub-epochs,
     about 7 min per candidate plus the S forward, so 10-15 min per round for at most 4 candidates (5 need a second
     wave, about 20 min). Up to 12 rounds is about 2-3 node-hours per start, 12-18 node-hours (48-72 GPU-h) over the
     six starts, plus the controls. Excluded: job startup, the separate ReturnnForwardJobV2 S jobs, the queue wait
     between the 12 sequential rounds; a 4-sub-epoch refit doubles the cost.
2. **TP-D Coarse-to-fine unit inventory.** This amends the registered item ("about 100 classes") to a sweep used
   for coarse-to-fine training, after the literature, before any job.
   - Cluster the 500 k-means centroids to {100, 200, 300} classes, label-free, by a second-level k-means on the
     centroids. The counts span the published range: Sicherman and Adi 2023 (second k-means on centroids; V-measure
     peak at 100-200) and Liu et al. 2018 (GAN, oracle boundaries, background only; best near 300). Coarsening costs
     purity (HuBERT PNMI 0.686 at 500 against 0.575 at 100). Labels never pick the count.
   - Fit phi on the classes with the A10 durinit recipe, 2 seeds per count. Then expand each class row to its units,
     m(u|s) = m(c(u)|s) p(u|c(u)), and continue EM on the 500 units (by analogy to Petrov et al. 2006, who split
     latent subsymbols under observed categories; TP-D coarsens the observations instead).
   - Read:
     - decision read, label-using, selects nothing inside the arm: key identity (AN-5's rule) and generative PER of
       the expanded phi against the two durinit A10 finals, margin fixed at registration. S alone cannot decide: the
       3.289 bar was cleared by a chance-PER arm;
     - S on the expanded phi against the A10 finals (3.299) and the bar (3.289);
     - TP-B's label-free battery;
     - each count's oracle many-to-one purity, as a label-using analysis only.
   - Cost, estimated: the coarse fit is about 2.8 h; the 500-unit continuation length (1.2-2.2 h would give 4-5
     GPU-h per arm) is fixed at registration. 6 arms is about two 4-GPU packs if both stages run in one pack job,
     four if the expansion is a separate job.
3. **Order (recommended).** TP-D first: with its name read it tests the partition hypothesis directly, at a fixed
   cost. Then TP-B' on TP-D's S-best finals, or on the six starts above if TP-D is not funded.
- The best case of either is the basin (phi PER 0.35-0.50; lift to 0.20-0.36), not gold. No published result covers a
  named inventory, so neither is expected to reach it; each tests the partition hypothesis.
4. **Naming levers brought back by TP0 (registered consequence; beside the proposal, not ranked, each needs the
   user's OK).** TP0 read EM LOCKED on the right partition, so a partition fix alone does not correct names; the
   names must already be right when the channel sharpens, or be moved by a lever that is not plain EM.
   - TP-A1, the LM-led E-step continuation (spec under "Proposed training changes"; Form 1, the table version, costs
     about 1-2 GPU-h). Its contraindications do not fire (AN-2 H2 REFUTED and LAMBDA KEEPS THE BASIN; AN-4 not H3').
     AN-0 found one LM-weighted E-step from a sharp channel DEAD; TP-A1 differs in starting from a flat channel.
   - The sparse channel prior (handoff item; Ravi and Knight 2011), a candidate beside TP-A1 in Form 1's table.
   - TP-B''s name step goes to the user as a separate decision (below): after TP0, a name the move assigns wrongly is
     not expected to be corrected by the refit EM.

**Not tested (the user should know):**
- TP0 ran iterated EM from the right partition for 12 sub-epochs only (A17 (ii)'s recipe, tau 1), with one seed for
  the 5-pair arm. Whether the 5-pair arm's frames moved for the five phones that the emission map renames is not
  measured.
- AN-3 was dropped after AN-0 DEAD, so LAMBDA OPENS and an iterated or flat-start LM-led E-step were never measured.
- AN-1 read NOT GOLD FIRST under V1, V2 and V12, and V3 (4- and 5-gram, run 2026-09-25 after this list) also reads
  NOT GOLD FIRST and NOT VISIBLE (gold 136th of 188 at every order), so TP-C is not brought. The V12 near miss at the non-SIL divisor was +0.0076, below the 0.01 floor,
  with the divisor chosen after the result.
- The sparse channel prior (the handoff item beside TP-A1; Johnson 2007, Ravi and Knight 2011) is untested.
- One-seed margins: AN-0 seed 1; AN-5 one seed per key; KEY BASIN rank1's margin. Also unmeasured: m without the
  duration weighting (AN-5), SIL escape (AN-0b), N(s,u) at batched size.

**Rulings needed from the user:**
- May a label-free secondary read gate a move's acceptance in TP-B'? Near the top, S alone is a weak judge
  (Berg-Kirkpatrick and Klein 2013; our PER-0.86 arm at the S bar). The project's candidates (LM per phone, rate)
  are label-informed (AN-4), so such a read would have to be fixed from general knowledge before any result.
- Is a broad-class name allocation (vowel / stop / fricative / nasal / SIL counts) admissible as general phonetic
  knowledge? It is the direct analogue of Jin et al.'s group counts, and is not proposed without a ruling.
- Still open: the occupancy term (V2), a unigram-level term beside the trigram.
- After TP0: is TP-B''s name step (an unclaimed name ranked by LM fit of its decoded contexts) acceptable as the
  only naming mechanism, given that EM does not correct a wrong name afterwards?

## Ruled out by current evidence (not proposed)

- Rename or name search under the current J (E7, A20: J ranks the right names lower; keys are swap-optimal).
- Up-weighting the LM at key level (E10).
- More random restarts or init hunting (A10, the wave, stage 1, A11/A12).
- Joint-temperature annealing as implemented (A17 (ii)): tau divides LM and channel alike and leaves their ratio,
  unlike TP-A1.
- Segmentation-first inits (E2).
- Unigram frequency matching as the main lever (Liu, Chen and Deng 2017: 71 % error alone against 10 % with
  bigrams; Nuhn and Ney 2013: unigram decipherment is rank sorting).

## Literature (full reports: `reports/lit_phi_name_signal_2026-09-24.md`, `reports/lit_phi_rename_moves_2026-09-24.md`)

- A better search helps only where the objective's optimum is right; otherwise accuracy falls (Dhavare 2013;
  Franti and Sieranoja 2019; Liang et al. 2010; Liang and Klein 2008). This is why J search stops (E7) and why
  every rename lever waits for AN-2.
- Hard-assignment clustering carries a partition-entropy reward (Kearns, Mansour and Ng 1997); EM balances states
  (Johnson 2007). Our audit finds absorption larger than entropy in J's margin (E9).
- Refit before accepting is established for splits and relocations (Jain and Neal 2004; SMEM; random swap), not
  for merges; a 1:1 rename cannot fix merged states.
- Dependent frames overweight the channel; supervised ASR compensates with an LM scale of about 16 (Wegmann and
  Gillick 2010). Text decipherment raises the channel weight instead (Knight et al. 2006), consistent because one
  text symbol is one observation. Untested for unsupervised phone naming.
- Higher-order LMs help ciphers only with a wide enough search (Nuhn, Schamper and Ney 2013); the speech moment-
  matching result goes the other way (ESPUM, Wang et al. 2024). Naming is identifiable in principle (Wang et al.
  2023) without a bound under noisy clusters and free segmentation.
- Corrections to the name-signal report: it read A20's oracle rename as a cost decrease (search error), but J is a
  log-likelihood, so the rename is a model error at key level; and its rate-confound argument compares rates on two
  conventions (E14), so it stays a hypothesis until AN-4.

## Results

### AN-0 (2026-09-25): AN-0 DEAD (audited CONFIRMED_WITH_CORRECTIONS)

Job `RenameEmStepJob.sXyLgUqfPBPv` (d2bbd4c7, 300 train utterances, derangement AH-T, AO-N, K-OW, M-Y, P-S
moving 0.407 of frames, H_LM 2.2571). Extraction `reports/extract_rename_an0_2026-09-25.md`; audit
`reports/audit_rename_an0_2026-09-25.md`, which re-scored every cell from the saved keys with independently
recomputed unit weights. All 16 cells match.
- restore (the bar is 0.05):
  - plain: 0 at lambda 1, 0.0019 at 2, 0.0001 at 4.4
  - rate-neutral: 0.0032 at 2, 0.0053 at 4.4
  - lambda 10, report only: 0 plain, 0.021 rate-neutral; both cells fail the guard.
  - The gold-row guard is 0.983-0.996 at lambda <= 4.4, so every one of those cells is VALID.
- Not a silent no-op:
  - 8-35 units change key per cell.
  - The posterior mass that swapped units send to their correct name rises with lambda (0.019 / 0.031 / 0.045) but never exceeds the wrong name's.
  - S_1 falls by 0.31-0.39.
  - The LM prior is not permuted along with phi: log Z, tokens per frame and S_1 differ between rows.
- Consequence applied as registered: TP-A1 is not brought on AN-0, and AN-3's grid is dropped.
- Correction (audit): the registered wording "TP-A1's mechanism is dead" states more than was measured. What is licensed: one LM-weighted E-step at lambda <= 4.4, in either form, does not flip the keys of a material share of swapped units of the deranged gold-key phi. Not licensed: anything about iterated EM, a flat-channel start (TP-A1's own regime), 1-pair contexts or other seeds.
  - The operating point is the sharpest possible channel. The margin against the swap partner averages about 4.4 nats per frame, and 58 % of moved-phone tokens in the LM text have a moved neighbour.
- Unexplained magnitude (audit estimate, not a registered statistic):
  - The LM gain from flipping one swapped token back has a median of 6.6 nats at lambda 1, which should outweigh the channel margin at 4.4. Yet only 0.4-11 % of swapped units' mass reaches the right name, and 24-65 % goes elsewhere.
  - Part of it goes to SIL, whose share rises from 0.079 to 0.118. Part goes to the empty OY and ZH rows, which take 5.7 % of frames at plain/4.4 against 0.45 % at lambda 1.
  - Under the smoothing, an empty row's emission (1/500) is 1.66 nats per frame above an off-key entry of a populated row, which is where the right name's row sits for a swapped unit.
  - Lattice behaviour or an undetected E-step issue: these files cannot tell. AN-0b is registered to settle it.

### AN-0b (2026-09-25): E-STEP CHECK passes; ESCAPE NOT THE BLOCK (audited CONFIRMED_WITH_CORRECTIONS)

Job `RenameEscapeJob.cw4BeJzrQ3U9` (0822791c, 3.5 min, same 300 utterances and derangement as AN-0). Extraction
`reports/extract_rename_an0b_2026-09-25.md`; audit `reports/audit_rename_an0b_2026-09-25.md`. The audit re-added
the paths with its own code (channel to 4e-5 nats, LM to 2e-12), rebuilt the keys from `n_su.npz` (500/500 units
match) and recomputed restore, rho and the split to 1e-15. The as-is cells equal AN-0 exactly.
- E-step check: all 10 MAP paths extracted, max relative difference 6.4e-7, max bridge_rel_diff 4.2e-16. The
  re-add uses no lattice code. This settles AN-0's open question: path scoring is correct.
- No-escape masks only the OY and ZH next-token columns, the same way in every row. N(OY, ZH) is 0 in every
  no-escape cell. Restore on the 5-pair row: 0.0000 at plain 4.4 and 0.0053 at rate-neutral 4.4, both VALID (gold
  0.9888, 0.9834). **ESCAPE NOT THE BLOCK.**
- Swapped units' mass at rate-neutral 4.4, no-escape: right name 0.060, partner 0.489, SIL 0.071, other names 0.379.
  SIL ESCAPE fires (0.071 > 0.060). The gold row sends 0.050 of the same units to SIL, so the excess is about 0.021.
  At plain 4.4, masking freed 0.069 of the mass; the right name gained 0.0024 of it.
- Consequence as registered: H1 (lock-in) is recorded at the sharp-channel operating point. No escape finding goes
  to the user.
- Corrections (audit):
  - Only the empty-row half of H1b was tested, since SIL was not masked. Whether SIL blocks the step is not measured.
  - "The pressure escapes through SIL" is not licensed. The wrong name keeps most of the mass.
  - Licensed: one step from the sharp gold-key channel (5 pairs, seed 1, 300 utterances, lambda <= 4.4) restores no
    material share of keys, even without the empty rows. Not licensed: iterated or flat-start EM, other seeds, or
    N(s,u) at batched size.

### AN-1 (2026-09-25): NOT GOLD FIRST under V1, V2, V12; NOT VISIBLE under V2, V12 (audited CONFIRMED_WITH_CORRECTIONS)

Job `KeyObjectiveScreenJob.erCoRAmZID5a` (login node, 198 s). Extraction `reports/extract_rename_an1_2026-09-25.md`;
audit `reports/audit_rename_an1_2026-09-25.md` re-scored all 218 real-corpus keys with independent code (terms
match to 4e-15; its V0 equals A20's J and stage 1's J_final). Held-out, nats per frame, 187 competitors.
- GOLD FIRST: gold's margin over the best competitor and its rank of 188 are V1 -0.139 (134th), V2 -0.123 (48th),
  V12 -0.058 (6th). The K30 > K70 > K100 ladder holds on seed means under every variant.
- NAMES VISIBLE: J_V(b) - J_V(a) is -0.56 to -0.83 under V2 and V12, so 0 of 4 keys pass. The class-prior term does
  not reverse the trigram's preference for the found names on the found partitions (A20).
- V3 (4- and 5-gram) was built later: see "AN-1 V3" below. It is also NOT GOLD FIRST and NOT VISIBLE.
- Correction (audit): the registered V2 formula did not fix its divisor. As built, V2's occupancy sum is divided
  by all frames, so a SIL frame scores 0 against about -3.1 to -3.5 for a non-SIL frame, and the term rewards SIL
  share. All 4 keys that beat gold under V12 carry SIL share 0.083-0.093 against gold's 0.074. Under the other
  admissible reading (divide by non-SIL frames), V12 puts gold 1st of 188 by +0.0076. That is below the 0.01
  floor, so the reading is still NOT GOLD FIRST, by 0.0024. On the unabsorbed string, V12's margin is -0.082 (all
  frames) or -0.019 (non-SIL frames).
- Decision table: no row fires, so TP-C is not brought. Any later use of V2 or V12 must first fix the divisor to
  non-SIL frames, or a key search gains by inflating SIL. The near miss is a measurement at one divisor chosen
  after the result, not a pass. Even there, names stay invisible, so a V12 search would still not rename toward
  gold on found partitions.

### AN-1 V3 (2026-09-25): NOT GOLD FIRST and NOT VISIBLE under a 4-gram and a 5-gram (audited CONFIRMED_WITH_CORRECTIONS)
- Job `KeyObjectiveV3Job.N7q9hgRBDrFw` (login node, 6:52, 2.65 GB). Implementation `reports/impl_rename_an1v3_2026-09-25.md`;
  audit `reports/audit_rename_an1v3_2026-09-25.md`. The auditor refit the n-grams independently on the uniform window
  `orN768ARKwlt`, reproduced orders 1-3 of the banked prior exactly, and matched lm3/lm4/lm5 on all 218 real rows to
  8.9e-16. Order 3 through the new path reproduces V0 (= AN-1's J) to 8.9e-16. The competitor set is AN-1's.
- Estimator: interpolated Witten-Bell, the trigram's own, extended in order. Held-out perplexity on the window is
  9.56 / 7.20 / 5.98 at orders 3 / 4 / 5, so there is no over-fit.
- GOLD FIRST: gold's margin to the best competitor is -0.2656 (V0), -0.2570 (V3a) and -0.2453 (V3b), and gold ranks
  136th of 188 in each. The K30 > K70 > K100 ladder holds.
- NAMES VISIBLE: J_V(b) - J_V(a) is -0.63 to -0.46 on all 4 selected keys under both orders, so 0 of 4 pass. The gap
  is only the LM term, because (b) renames (a)'s symbols one to one with SIL kept, so emission and duration are
  unchanged.
- The audit adds: gold's deficit is mostly emission (0.20-0.22), with only 0.03-0.04 from the LM. On gold's
  key-decoded string the LM cost rises with order (4.26 / 4.98 / 5.55 nats per token, against 3.69 for a uniform
  LM), and 51 % of its 5-grams never occur in the text (1 % for real text). Key-decoded strings are not phone text
  even under the gold key, so a higher-order LM penalises them more.
- Decision table: no row fires, so TP-C is not brought. The read licenses only that, not "no higher-order LM term
  can see names".
- Correction (audit): the job ran on the login node, not gpupack as the executor report says.

### AN-2 and AN-4 (2026-09-25): H2 REFUTED; S SEES NAMES and S PREFERS FOUND NAMES; LAMBDA KEEPS THE BASIN; P1a False, P1b and P1c True, not H3' (audited CONFIRMED_WITH_CORRECTIONS)

Jobs `An2ReadJob.91eeK6mtb6UH` and `An4ReadJob.c75KVY92wtjP` (941e392d, 996a6aad). They score the 260 CV-holdout
utterances, none excluded, with the same settings in all 44 columns. Extraction
`reports/extract_rename_an24_2026-09-25.md`; audit `reports/audit_rename_an24_2026-09-25.md`, which recomputed every
value from `per_utterance.tsv` (AN-2 exactly, AN-4 to 2e-15). Held-out S_1, nats per frame.
- H2 REFUTED. The gold-key phi with gold names has S_1 = 4.573. The 3 full derangements (fixed-point-free over 37
  symbols) are worse by +0.513 to +0.567, against the 0.1 bar. S sees names on the gold partition.
- Found partitions: the 4 stage-1 keys as key-built phis at sub-epoch 0; (b) against (a) is a pure row permutation.
  - S_1(b) - S_1(a) is +0.351 / +0.251 / +0.292 / +0.278. The 3 random renames of (a) are worse by +0.587 to +0.744.
  - Both SEES NAMES (4 of 4: the oracle names beat random renames) and PREFERS FOUND NAMES (4 of 4: the found names
    beat the oracle names) fire. S orders the namings found, then oracle 1:1, then random.
- Basin lead (min over the six A10 finals minus max over the durinit basin set at 48), rate-neutral: +0.075 /
  +0.166 / +0.260 / +0.272 at lambda 1 / 2 / 4.4 / 10, so LAMBDA KEEPS THE BASIN. The plain form beside: +0.075 /
  +0.081 / -0.053 / -0.481.
- AN-4 (plain form, R1 scope), six finals against the basin set at 48:
  - P1a False. The finals' channel term per frame is lower (-3.049 to -2.918 against -2.877 to -2.860).
  - P1b True. LM per phone is -3.987 to -3.809 against -3.163 to -3.124.
  - P1c True. The non-SIL rate is 6.85-7.56 Hz against 8.42-8.55 Hz.
  - Not H3'.
  - P2 True. The LM term per frame differs by -0.022 against the paired S gap of +0.137 (group means). So the finals
    lose to the basin on both terms, and most of the S gap lies outside the LM term.
  - P3 HOLDS on the three basin trajectories: from 0 to 48, PER rises, the channel term rises and LM per phone falls.
- Generative PER, dev-other, Hungarian: the finals 0.838-0.862; the basin set at 48 0.345-0.391.
- Decision table: row 2 fires.
  - The rename levers are dropped (TP-A1, TP0's lambda arms).
  - TP-A2 is not brought, because P1a fails.
  - TP-B and the coarse-to-fine inventory are brought.
  - Rows 1, 3, 4, 5 and 7 do not fire. Row 6 waits on AN-5, under the precedence amendment below the table.
- Corrections (audit):
  - Row 2's wording "on the partitions EM reaches" overstates the evidence. The read concerns the four stage-1 key
    partitions as key-built phis before EM. The claim is that S, like J, ranks the found names above the oracle
    1:1 names on those partitions. The found names were climbed on J, and S is J's objective with a soft channel,
    so PREFERS largely re-expresses A20. What EM does from those phis is AN-5's question.
  - The oracle names are the best 1:1 naming of a many-to-one partition (identity 0.39-0.47), not gold.
  - The four keys were selected by held-out J on the same 260 set, which favours (a). Flipping PREFERS would need a
    bias above 0.24 per frame on 2 of 4 keys, more than the whole J range over 124 finals.
  - P2 holds on group means and on 17 of 18 pairs (durinit_s01 against g_dur fails). It fails under the
    unregistered rate-neutral derivative.
  - P3's gold-key endpoint rise is +0.006 on a non-monotone path, so P3 rests on g_dur and r30_dur.
  - Neither P2 nor P3 gates anything.
  - The AN-4 job's printed rules omit the R1 scope bullet. Its behaviour conforms.

### AN-5 (2026-09-25): EM LOCKED; stage 2 is not the route (audited CONFIRMED_WITH_CORRECTIONS)

Job `An5ReadJob.2q262c8lhmdT` (40cb0854, guard 5e4df526; `output/sae/4a/rename/an5/report.txt`). It reads the
stage-2 key arms of pack `G0Vzzokj5PQC` and, beside them, the gold-key arm of `ge1MKcAPmZIV`, at 0/4/12/48, with
no checkpoint substituted. m and pi come from the same checkpoint in every cell. Audit
`reports/audit_rename_an5_2026-09-25.md`, which reproduced all 20 cells exactly with its own code (weights
15,275,716, equal to A20's).
- Key identity from epoch 0 to 48, with DELTA:
  - rank1 0.115 to 0.075 (-0.040)
  - rank2 0.143 to 0.139 (-0.005)
  - rank3 0.124 to 0.123 (-0.001)
  - rank4 0.065 to 0.052 (-0.013)
  - gold-key arm 1.000 to 0.766 (not counted)
- 0 of 4 arms rename and 4 of 4 are locked: **EM LOCKED**. STAGE 2 IS THE ROUTE is not licensed. rank1 and rank3
  have KEY BASIN S (3.273, 3.285; `SAE_4A_lexlat_v2.md`), but their DELTAs are negative.
- Robustness (audit): every pi variant and the posterior-key variant keep all 4 arms locked (highest DELTA +0.017).
  m without the duration weighting is not measured.
- A15-F at 48, key arms against the gold-key arm:
  - 7-11 duplicated gold types (1 for gold) and 8-12 unclaimed (1 for gold);
  - own-label agreement 2-6 of 40 (40 of 40 for gold);
  - R4 emission accuracy 0.34-0.35 (0.54 for gold).
- Generative PER at 48, Hungarian: 0.80-0.87 for the key arms, 0.39 for the gold-key arm.
- Decision table under the precedence amendment: row 6 does not fire, so row 2's proposal stands. That is TP-B
  and the coarse-to-fine inventory, without TP-A2 (P1a False). TP0 is not triggered. The best 1:1 renaming of each
  arm's partition stays at 0.43-0.49 while identity stays at 0.05-0.14, so partition and names are separable.
- Corrections (audit):
  - Identity ranges over 0.05-0.17 across epochs (rank2 at epoch 4 is 0.165).
  - LOCKED is not frozen. Keys change on 28-51 % of frames, with no net move toward gold.
  - The gold-key arm's drop is a partition change, not a renaming.

### TP0 (2026-09-25): EM LOCKED on the right partition (audited CONFIRMED_WITH_CORRECTIONS)

Pack `PackedBlankfreeTrainJob.21tww6QQK0tH` (SLURM 2011293, 12 sub-epochs per arm, A17 (ii)'s recipe with the init
phi as the only delta), reader `Tp0ReadJob.t7gzZhB2fWtn` (`output/report.txt`, `output/tp0.json`). Extraction
`reports/extract_tp0_read_2026-09-25.md`; finish check `reports/exec_tp0_check_2026-09-25.md`; audit
`reports/audit_tp0_read_2026-09-25.md`. The audit reproduced all 16 cells with its own code and confirmed three things:
the inits and derangements match AN-0's, the configs equal A17 (ii)'s except the init path, and epoch 0 is measured on
the init phis. Identity uses AN-5's code.

| arm | key identity 0 / 4 / 8 / 12 | DELTA | best 1:1 at 12 | S (260 set) 0 / 12 | genPER direct 0 / 12 | dup / unclaimed at 12 |
|---|---|---|---|---|---|---|
| full_s1 | 0.078 / 0.106 / 0.100 / 0.102 | +0.024 | 0.554 | 5.113 / 3.387 | 0.953 / 0.854 | 7 / 8 |
| full_s2 | 0.078 / 0.093 / 0.092 / 0.090 | +0.013 | 0.518 | 5.086 / 3.370 | 0.949 / 0.855 | 7 / 8 |
| 5pair_s1 | 0.593 / 0.627 / 0.598 / 0.588 | -0.005 | 0.650 | 4.933 / 3.292 | 0.660 / 0.502 | 6 / 6 |
| control | 1.000 / 0.817 / 0.793 / 0.780 | -0.220 (not counted) | 0.780 | 4.573 / 3.252 | 0.327 / 0.336 | 3 / 3 |

Best 1:1 identity is 1.000 at epoch 0 in every arm. Duplicated / unclaimed gold types are 1 / 2 at epoch 0 in every
arm (A15-F, label-using, report only).
- All 3 deranged arms read DELTA <= 0.05: **EM LOCKED**.
- Corrections (audit):
  - The two full arms carry the verdict; they had room to pass. The 5-pair arm could not reach 0.2: a perfect
    rename plus the control's drift gives about 0.19.
  - In 5pair_s1 the emission map puts AH, T, N, M and Y back on their own names while key identity stays flat.
    Whether their frames moved is not measured; a per-phone key breakdown would show it.
  - The partition does not stay right: best 1:1 identity falls to 0.52-0.65, against the control's 0.78.
- Descriptive, not registered (full: 2 seeds; 5-pair and control: 1 seed):
  - Rather than renaming, EM re-partitions around the wrong names. Duplicated gold types rise from 1 to 6-7 and
    unclaimed ones from 2 to 6-8, against 3 / 3 for the control: the state of the found finals (E3), now reached
    from the right partition.
  - S falls in every arm. At sub-epoch 12 the control is lowest (3.252), then 5-pair (3.292), then full (3.370,
    3.387). These are unpaired means on the common 260 set, not a registered comparison.
- Consequence (registered; applied in the Proposal): EM keeps wrong names even on the right partition, so fixing the
  partition alone is not enough. The naming levers (TP-A1, the sparse channel prior) and TP-B''s name step go back
  to the user beside the proposal.
