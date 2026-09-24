# SAE 4A -- rename: what phi training needs so that EM can correct wrongly named phones

## State

Created 2026-09-25 as a proposal phase (user request 2026-09-24: "carefully plan with literature review: how
should we adjust our phi training to make it possible. The rule for no new training arms stays, just propose in a
next_step phase file"; new analyses are allowed, including GPU). Nothing here is funded training.
- Allowed now: forwards, readers, CPU analyses and one-step operator evaluations whose updated tables are discarded
  (AN-1 to AN-5). They need a design review before the first job, then implementer, code review and executor.
- Needs the user's OK: every TP arm below, because each trains phi.
- Live dependencies in `SAE_4A_lexlat_v2.md`: stage 2 key arms (pack G0Vzzokj5PQC) feed AN-5; A18 (b) is unrelated.

NEXT:
1. Design review (fable) of this file before any AN job.
2. After review: implement AN-1 to AN-4 (one GPU pack for AN-2/3/4, AN-1 on the login node); AN-5 when stage 2
   finishes.
3. After the reads are audited, bring the decision table's recommended TP to the user.

## Objective

The user's premise: phi EM cannot correct wrongly named phones. Find the change to phi training that would let it,
decide it with analyses on existing phis first, and propose training only as pre-registered arms for the user.

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
| E2 | The basin is reachable from type-level labels with durinit durations and cell-free emissions | gold-key control | segmentation init is not the bottleneck |
| E3 | Random-init EM finals are mislabelled and merged: own-label agreement at chance, 11-19 phones unclaimed, SIL/N/T/W/AY/Z/M take 3-4 symbols | A15, A15-F | the state the premise describes |
| E4 | Along A10 trajectories the emission-map accuracy (R4 emis) is flat from sub-epoch 4 to 48; own-label T1 falls from 4 to 12 | A15 | names stop improving early (indirect) |
| E5 | Relabelling an EM final through its emission map, without refit, raises S by 0.12-0.31 | A16 (a) | S rejects a pure rename of a co-adapted phi; refit untested |
| E6 | Up-weighting the trigram in scoring (S_lambda, lambda <= 3) does not rank gold above EM finals; gold was not refit there | A16 (a2) | scoring only; lambda in the E-step during training is untested |
| E7 | Key level: all 124 stage-1 finals beat gold on J; J penalises the gold-matching 1:1 names by 0.42-0.57 per frame; the found keys are swap-optimal | stage 1, A20 | at key level the objective, not the search, rejects right names |
| E8 | On the gold partition J penalises scrambled names by 0.69-1.09 | A20 audit, unregistered | the trigram carries a strong name signal on a clean partition |
| E9 | J's name-free margin over gold (0.20-0.23) is mostly less d_min absorption (0.10-0.17), then symbol entropy (0.06-0.10) | A20 audit, unregistered | J's defect is a hard-key artefact that phi's soft channel may not share |
| E10 | Found keys fit the trigram better than gold (lm -0.948 at 9.20 Hz against -0.990 at 9.47 Hz): a hard key pushes unit noise (purity 0.644) into gold's strings | A20 | raising the LM weight at key level favours wrong keys |
| E11 | Inside the basin EM drifts: gold-init PER 0.19 -> 0.35 while S falls; tau = 1 from the start removes about a quarter | A14 (ii), A17 (ii) | S does not reward accuracy inside the basin |
| E12 | r70 (random label noise, per-unit argmax right) improves under EM, PER 0.61 -> 0.44-0.50 | A14 (ii), A17 (iii) | denoising, not renaming |
| E13 | No phi EM run has started from a clean partition with systematically wrong names; stage 2 starts from found partitions with wrong names (identity 0.07-0.14, many-to-one 0.49-0.58) | registrations | the premise has no direct test yet |
| E14 | Rates on two conventions: A10 finals 6.8-7.6 Hz (A8 greedy emitted rate); basin arms 10.0-10.5 Hz and r100-EM 9.13 Hz (Viterbi segment rate, `SegmentationBoundaryJob.g52gIGtUHpkr`). Key level: found 9.2-9.7 against gold 9.47 | A10, keyinit audit, A20 | a phi-level rate gap is not established on one convention (AN-4); none at key level |
| E15 | Gold phi under a fixed name permutation (permphi): decode-based Hungarian PER 0.819, its map recovers 16 of 40 labels; the emission map recovers 40 of 40 | A15, A15-E | renaming must be measured through the emission map or key identity, never decode PER |
| E16 | Basin phis lift the recognizer (A17 (i) 0.20); EM finals do not (A14 (i) 0.82-0.84; A18 (a) 0.842) | A17 (i), A14 (i), A18 (a) | fixing names is worth the work |

Status of the premise: plausible and indirect (E3, E4, E13). Between basins S is right (E1); within J it is wrong
(E7). The phi-level question is therefore search (EM cannot leave a wrong-name optimum that S ranks higher) or
coupling (the partitions EM reaches are ones on which the right names do not win under S).

## Hypotheses (phi level)

- **H1 lock-in.** The per-frame channel dominates the E-step once emissions sharpen (about 4-5 dependent frames per
  segment, each counted as independent evidence), so the posterior cannot move a whole symbol to another name.
  EM is local; a rename is a coordinated permutation (a QAP under a bigram or higher, Nuhn & Ney 2013). Predicts:
  EM from permuted gold stays permuted at lambda = 1; an LM-weighted E-step restores names.
- **H2 weak name signal under S.** S barely separates right from wrong names even on a clean partition. E8
  (key level, 0.69-1.09) argues against; AN-2 measures it at phi level.
- **H3 coupling.** The name-free channel shapes partitions that duplicate frequent phones, starve rare ones and
  possibly under-segment (E3; E14 unverified); on those partitions no naming is near right, so neither a rename nor a stronger LM
  helps until the partition changes. Predicts: EM finals win S on the channel term and lose on LM per phone.

H1 and H3 can both hold; the analyses separate their sizes.

## Analyses (registered 2026-09-25 before any job; disclosed label-using where marked; no training)

Shared inputs: the 260 set and S reader of A14 (ii); the gold-key phi `PhiFromKeyInitJob.f0jaGuiJVe6A`; the
stage-1 selected keys and their A20 variants (`KeyOracleNameReadJob.QD3S2xcmg0Wt/output/keys`); the six A10
sub-epoch-48 phis; the A14 (ii) and keyinit arms at 0/4/12/48. Permutations: non-SIL symbols, seeds 1-3,
1-pair, 5-pair and full.

- **AN-1 (CPU, key level) Objective screen.** J variants on all stage-1 finals, the A20 (a)/(b)/(c) keys, gold,
  K30/K70/K100 and the stage-1 nulls: V1 emission without d_min absorption (on the unabsorbed symbol string);
  V2 J plus a frame-occupancy term against the trigram's own unigram (Kearns, Mansour and Ng 1997; analysis only,
  its use in training needs a ruling under the trigram-or-higher rule); V3 4-gram and 5-gram priors from the
  uniform window, if the prior job supports the order. **GOLD FIRST under V** if J_V(gold) exceeds every final and
  every A20 variant by more than 0.01 and gold > K30 > K70 > K100; **NAMES VISIBLE under V** if J_V(b) - J_V(a)
  > 0.01 on at least 3 of 4 selected keys. Also registered here: the train-side split of the emission gap into
  H(S) - H(U) and absorption (E9), printed by the reader with its convention.
- **AN-2 (GPU forward; label-using) Does S see names, and does lambda move the ranking?** S_lambda =
  -(1/T) log sum_y P_LM(y)^lambda p_phi(x | y) for lambda in {1, 2, 4.4, 10} on: the gold-key phi with gold names
  and with each permutation; the stage-1 key phis under (a) found and (b) oracle 1:1 names; the six A10 finals; the
  basin phis (A14 (ii) gold/r30/r70 and gold-key at 48). Readings:
  - VOID unless S_1(gold-key) < S_1(full permutation) - 0.1 on all 3 seeds (S must see names on a clean partition).
  - **S SEES NAMES ON FOUND PARTITIONS** if S_1(b) < S_1(a) - 0.01 on at least 3 of 4 keys; **S PREFERS FOUND
    NAMES** if S_1(b) > S_1(a) + 0.01 on at least 3 of 4; else MIXED.
  - **LAMBDA KEEPS THE BASIN** if, at every lambda in the grid, every basin phi's S_lambda is below every A10
    final's; **LAMBDA FAVOURS FINALS** if the basin's lead shrinks monotonically with lambda and reverses by 10.
- **AN-3 (GPU forward plus CPU M-step; label-using) Does the EM operator restore names?** From each permuted
  gold-key phi and each stage-1 key phi (a), one exact E-step (A11's count-table forward-backward, 1,000 fixed train
  utterances, tau = 1, LM exponent lambda in {1, 2, 4.4, 10}), one closed-form M-step to a type-level table, then
  discard. Measure the change in frame-weighted identity of the unit-to-symbol argmax key (rho), the change in
  many-to-one agreement, and S_1 after the step. Readings on the full permutations (3 seeds):
  - **LOCKED AT 1** if rho < 0.01 at lambda = 1 on all 3 seeds.
  - **LAMBDA OPENS at L** if rho >= 0.05 on all 3 seeds at lambda = L with many-to-one falling by at most 0.05.
  - The 1-pair and 5-pair rows and the stage-1 keys are reported beside.
- **AN-4 (GPU forward; label-free statistics, PER reported) Term anatomy of S.** Per utterance on the 260 set:
  E_q[log P_LM] per frame and per phone, E_q[log p_phi(x | y)] per frame, H(q) per frame (the three sum to -S),
  expected non-SIL rate, SIL share, E[d], decoded unigram and bigram KL to the prior's marginals, and I(U; S) of
  the posterior-weighted unit-symbol table. Phis: A10 finals and their trajectories at 4/12/48, the A14 (ii) and
  keyinit arms at 0/4/12/48, stage 2's arms when finished. Predictions fixed now:
  - P1 (H3): every A10 final has a higher channel term per frame and a lower LM term per phone than every basin
    phi at 48, at a lower rate.
  - P2: the LM term per frame differs by less than 0.05 between finals and basin (the rate confound).
  - P3 (E11): along the basin trajectories, the channel term rises and the LM per phone falls as PER rises.
- **AN-5 (GPU forward; label-using; owed by the keyarms review) Name tracking in stage 2.** A15-F measures
  (own-label agreement, claimed and unclaimed phones, duplicates, R4) and identity on the gold-key arm and the four
  stage-2 key arms at 0/4/12/48. Identity here and in AN-3 is key identity: the frame-weighted agreement of the
  phi's unit-to-symbol argmax key with the gold key, as in A20 (decode PER cannot see renaming, E15). **EM RENAMES** if identity at 48 minus identity at 0 is at least 0.2 on at least 2
  of 4 key arms; **EM LOCKED** if at most 0.05 on at least 3 of 4; else PARTIAL.

Cost: AN-2 to AN-4 fit one four-GPU pack (about 60 forwards on 260 utterances and 28 E-steps on 1,000); AN-5 is
the A15-F reader on stage 2's checkpoints; AN-1 is one login-node job. All results go to Results below.

## Proposed training changes (not funded; each needs the user's OK)

- **TP0 (label-using diagnostic, supervised init, analysis only) Direct test of the premise.** phi EM from the
  gold-key phi under a full permutation (seeds 1-2) and a 5-pair permutation, A17 (ii)'s recipe (tau = 1,
  12 sub-epochs). EM RENAMES / EM LOCKED as in AN-5 (from init identity), S and genPER at 0/4/8/12. If AN-3 reads
  LAMBDA OPENS, the same arms under TP-A1's schedule show whether the opening survives iteration. Worth running
  only if AN-5 reads PARTIAL or cannot separate partition from names.
- **TP-A1 (label-free) LM-led E-step continuation.** The E-step uses P_LM^lambda with lambda = 4.4 at sub-epoch 1,
  falling linearly to 1 by sub-epoch 12, then 1 to 48; the final objective is S unchanged. lambda0 = 4.4 is the
  general-knowledge mean phone length in frames (A9 durinit), from the frame-dependence argument: LM scale 16
  compensates dependent acoustic frames in supervised ASR (Wegmann and Gillick 2010), and cipher EM succeeds when
  the channel starts flat so the LM leads (Ravi and Knight 2008, 2011). lambda0 is never tuned on labels. Arms: the
  A10 durinit seeds 1-3 from random init, plus r100 as the negative control. Gate, fixed now: **LM-LED BASIN** if
  S at 48 < 3.289 on at least 2 of 3 seeds; report genPER at 0/4/12/48, rate, the AN-5 measures, and the rise over
  the G4a.L2.2 null; an A14 (i)-form lift pack only on LM-LED BASIN. Contraindicated if AN-2 reads LAMBDA FAVOURS
  FINALS or AN-4 shows the finals winning on the LM term.
- **TP-A2 (label-free) Rate floor.** A penalty on expected non-SIL phone rate below a general-knowledge band
  (source to be verified by literature, never MFA's 11.9 Hz). Only if AN-4 P1 and P2 hold.
- **TP-B (label-free) S-level moves with refit.** Every 4 sub-epochs, propose a global name swap set (from the
  E-step counts, solved as an assignment) and relocations of a duplicated symbol to a starved LM region; accept on
  held-out S after 2 sub-epochs of refit (Jain and Neal 2004; SMEM, Ueda et al. 2000; random swap with 2
  iterations, Franti). Only if AN-2 reads S SEES NAMES ON FOUND PARTITIONS; otherwise S rejects the right move too.
- **TP-C (label-free) Key route under a repaired J.** Stage-1 search under a J_V that reads GOLD FIRST in AN-1,
  then stage-2 phi EM from its keys. Only on GOLD FIRST.
- Needs a ruling before any use in training: the occupancy term (V2), a unigram-level term beside the trigram.

## Decision table (fixed before any result)

| Reads | Proposal brought to the user |
|---|---|
| AN-3 LOCKED AT 1 and LAMBDA OPENS; AN-2 LAMBDA KEEPS THE BASIN | TP-A1 (with TP0's lambda arms as its mechanism check) |
| AN-2 S SEES NAMES ON FOUND PARTITIONS; AN-3 no lambda opens | TP-B |
| AN-4 P1 and P2 hold; AN-2 not LAMBDA KEEPS THE BASIN | TP-A2 |
| AN-1 GOLD FIRST under some V | TP-C beside the above |
| AN-5 EM RENAMES | the premise fails for found partitions: stage 2 is the route; no TP |
| AN-2 VOID, or none of the above | report NO LEVER SUPPORTED; name what would settle it |

## Ruled out by current evidence (not proposed)

- Rename or name search under the current J (E7, A20: J ranks the right names lower; keys are swap-optimal).
- Up-weighting the LM at key level (E10).
- More random restarts or init hunting (A10, the wave, stage 1).
- Joint-temperature annealing as implemented (A17 (ii); tau scales LM and channel together and does not change
  their ratio, unlike TP-A1).
- Segmentation-first inits (E2).
- Unigram frequency matching as the main lever (Liu, Chen and Deng 2017: 71 % error alone against 10 % with
  bigrams; Nuhn and Ney 2013: unigram decipherment is rank sorting).

## Literature (full reports: `reports/lit_phi_name_signal_2026-09-24.md`, `reports/lit_phi_rename_moves_2026-09-24.md`)

- A better search helps only where the objective's optimum is right; otherwise accuracy falls (Dhavare 2013;
  Franti and Sieranoja 2019; Liang et al. 2010; Liang and Klein 2008). This is why J search stops (E7) and why
  TP-B waits for AN-2.
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

(none yet)
