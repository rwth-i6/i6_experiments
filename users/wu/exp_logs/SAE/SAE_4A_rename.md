# SAE 4A -- rename: what phi training needs so that EM can correct wrongly named phones

## State

Created 2026-09-25 as a proposal phase (user request 2026-09-24: "carefully plan with literature review: how
should we adjust our phi training to make it possible. The rule for no new training arms stays, just propose in a
next_step phase file"; new analyses are allowed, including GPU). Nothing here is funded training.
- Allowed now: forwards, readers, CPU analyses and one-step operator evaluations whose updated tables are discarded
  (AN-0 to AN-5). Every TP arm needs the user's OK, because each trains phi.
- Design review `reports/design_review_rename_2026-09-25.md`: APPROVE_WITH_AMENDMENTS; amendments A1-A12 applied
  below before any job.
- Live dependency in `SAE_4A_lexlat_v2.md`: the stage-2 key arms (pack G0Vzzokj5PQC) feed AN-5.

NEXT:
1. AN-0 pre-check under Amendment R1 (lambda grid 1/2/4.4/10, plain and rate-neutral forms, gold-row guard,
   restore statistic; made at the launch review, before any result): implementer re-build
   (`reports/impl_rename_an0_r1_2026-09-25.md`), then a bounded re-review, then executor through gpupack.
2. AN-1 read recorded (Results): NOT GOLD FIRST, NOT VISIBLE, so no TP-C. AN-2/AN-4 are built (`reports/impl_rename_an24_2026-09-25.md`) and are being amended for R1; then code review and launch.
   AN-3 as re-scoped by AN-0; AN-5 when stage 2 finishes.
3. After the reads are audited, bring the decision table's proposal to the user.

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
| E13 | No phi EM run has started from a clean partition with systematically wrong names; stage 2 starts from found partitions with wrong names (identity 0.07-0.14, many-to-one 0.49-0.58) | registrations | the premise has no direct test yet |
| E14 | Rates on two conventions: A10 finals 6.8-7.6 Hz (A8 greedy emitted rate); basin arms 10.0-10.5 Hz and r100-EM 9.13 Hz (Viterbi segment rate, `SegmentationBoundaryJob.g52gIGtUHpkr`). Key level: found 9.2-9.7 against gold 9.47 | A10, keyinit audit, A20 | a phi-level rate gap is not established on one convention (AN-4); none at key level |
| E15 | Gold phi under a fixed name permutation (permphi): decode-based Hungarian PER 0.819, its decode map recovers 16 of 40 labels (A15-E registration's build test); the emission map recovers 40 of 40 | A15, A15-E | renaming must be measured through the emission map or key identity, never decode PER |
| E16 | Basin phis lift the recognizer (A17 (i) 0.20); EM finals do not (A14 (i) 0.82-0.84; A18 (a) 0.842) | A17 (i), A14 (i), A18 (a) | fixing names is worth the work |
| E17 | Count-table EM from Dirichlet random tables stops at iteration 2-3 (gain < 0.01), S 3.536-3.608 on the 285 holdout; its frame-permuted nulls fall out of the rate band, so the family read CANNOT_TELL | A11/A12 read | the table operator locks in almost at once |

Status of the premise: plausible and indirect (E3, E4, E13, E17). Between basins S is right (E1); within J it is
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
- **AN-5 (GPU forwards; label-using; owed by the keyarms review) Name tracking in stage 2.** A15-F measures
  (own-label agreement, claimed and unclaimed phones, duplicates, R4) and key identity on the gold-key arm and the
  four stage-2 key arms at 0/4/12/48. **EM RENAMES** if identity at 48 minus identity at 0 is at least 0.2 (K70's
  identity 0.30, the lowest from which EM is known to reach the basin, minus the found keys' 0.07-0.14) on at least
  2 of 4 key arms; **EM LOCKED** if at most 0.05 on at least 3 of 4; else PARTIAL. EM RENAMES licenses "stage 2 is
  the route" only with KEY BASIN (S at 48 < 3.289) on the same arm.

Cost: AN-2 and AN-4 are about 180 gpupack forwards at about 20 s of GPU each, AN-3 is 28 E-steps on 1,000
utterances (about 16 GPU-minutes); 1-2 GPU-h in all, limited by job count. AN-1 is one login-node job
(10-20 minutes); AN-5 is the A15-F reader on 16 checkpoints.

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

### AN-1 (2026-09-25): NOT GOLD FIRST under V1, V2, V12; NOT VISIBLE under V2, V12 (audited CONFIRMED_WITH_CORRECTIONS)

Job `KeyObjectiveScreenJob.erCoRAmZID5a` (login node, 198 s). Extraction `reports/extract_rename_an1_2026-09-25.md`;
audit `reports/audit_rename_an1_2026-09-25.md` re-scored all 218 real-corpus keys with independent code (terms
match to 4e-15; its V0 equals A20's J and stage 1's J_final). Held-out, nats per frame, 187 competitors.
- GOLD FIRST: gold's margin over the best competitor and its rank of 188 are V1 -0.139 (134th), V2 -0.123 (48th),
  V12 -0.058 (6th). The K30 > K70 > K100 ladder holds on seed means under every variant.
- NAMES VISIBLE: J_V(b) - J_V(a) is -0.56 to -0.83 under V2 and V12, so 0 of 4 keys pass. The class-prior term does
  not reverse the trigram's preference for the found names on the found partitions (A20).
- V3 (4- and 5-gram) was not built, so the negative covers V1, V2 and V12 only.
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
