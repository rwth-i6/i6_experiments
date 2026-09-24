# SAE_i6 reference: lexlat v2 — phi-first decipherment on the cold line (last phase before the move)

This was the last phase of the cold line before the campaign moved clusters. It asks whether a competent reverse model phi can be obtained without labels while the recognizer theta is out of the loop, and whether such a phi then anchors a random theta. It builds on the blank-free bed (`SAE_i6_ref_blankfree.md`) and the k2 lexicon arms (lexlat, `SAE_i6_ref_lexicon.md`); their objective, prior, graph and model classes are used here unchanged.

## Objective and premise

The lexlat record gives the premise:
- A competent reverse model anchors a recognizer under the joint objective. With the gold phi, the k2 arm holds p0 at PER 0.180 (lexlat D10e). A random-init phi destroys p0 within 57 steps.
- The cold phi is a contract on the recognizer's own decode. Its own decode is preferred over gold by 800-1190 nats, while gold and deranged gold are separated by only 18-79 (lexlat D10b).
- The objective prefers the phonetic solution: the gold-init run is lower on every term (l_tau 1.749 vs 1.873, k2 0.247 vs 0.306, agg 0.211 vs 1.264; lexlat D13).
- The joint cold start lets theta fix a length-faithful private code before phi carries any structure.

So the label-free problem is to obtain a competent phi with theta out of the loop. The phase has three parts:
- L2-1 fits phi by EM on the unit stream under the existing prior, with likelihood-selected restarts (Berg-Kirkpatrick & Klein 2013; Klejch et al. 2022).
- L2-0 measures beforehand how competent phi must be, and which label-free statistic tracks that.
- L2-2 brings the recognizer in afterwards.

(src: SAE_4A_lexlat_v2.md § Objective)

## Constraints that still bind

- The work is pure unsupervised and GAN-free. Labels may appear only in disclosed diagnostics (L2-0's ladder, supervised inits, label-using reports). No label enters a selection or a gate. Plain dev-other PER may gate a lift read, as G4a.9 did.
- Supervised inits (gold phi, corrupted-gold phis, MFA durations, the emission map) are analysis only. They never become a route or a fallback.
- Selection statistics are held out and label-free. dev-other enters only reports and lift reads.
- The objective, the phone trigram prior, the treatment graph at max_active 1000, both model classes, tau = 2 for joint runs and the D10e pack constants are unchanged from lexlat.
- These are not funded:
  - further full joint cold restarts, since every cold arm of the campaign sits in the chance band 0.83-0.91 at every seed (the cold_ctl control is exempt);
  - a recognizer-neutral (alpha = 0) arm from a cold start, which hands the posterior to an untrained phi, the collapse driver;
  - a neural-refit split-merge search. Count-table repair moves on phi are the only split-merge form allowed.
- The duration prior may encode general knowledge of phone length. A duration model learned from supervised data may not initialise or restrict phi.
- The phone-prior weight lambda is never chosen from a gold-rank read. If a weighted prior is used, it takes the literature value lambda = 3, fixed in advance.

(src: SAE_4A_lexlat_v2.md § Constraints; § Design review amendments, A9, A16)

## Methods

### L2-1 phi-first EM (the main, label-free line)

- **Objective.** This is the lattice term with the recognizer removed. theta is a null recognizer (zero logits, frozen), so the posterior is the generative one, proportional to (P_psi(B(a)) p_phi(z | B(a), eta))^(1/tau). phi trains by gradient on this term (generalised EM). Rate and agg depend only on theta, so they are constant. With q absent, tau = 1 is the exact E-step. Pruning at max_active 1000 sees only prior and phi scores.
- **Prior.** Only the frozen phone trigram is used. The registered stage 2 (the k2 word graph) was removed by A1: the k2 term reads only the recognizer's emissions, so under a null theta it is a constant. The word graph enters at L2-2.
- **Optimiser (A3).** phi uses Adam at a constant lr 3e-3 with clip 5, as in the supervised reverse fit.
- **Data (A3).** phi trains on the bed's train stream in its own sub-epochs, so 4 sub-epochs make one pass over train-clean-100. tau per sub-epoch is [4, 1, 1, 1], then 1.
- **Score.** S is the held-out tau = 1 NLL per frame on the train stream's CV holdout (285 utterances; lower is better).
- **Duration settings (A9).** All three use SegmentalReverseModel's topology (d_min 2, D 25, D_sil 50).
  - `uniform` is the bed's init. It is reported only and never chosen.
  - `durinit` initialises every phone type's duration logits from one rate-matched maximum-entropy law on [2, D_k] and leaves them trainable. The law's mean is (50 / 9.66) x (retained / original frames) = 4.41 retained frames. 9.66 phones/s is the label-free rate. SIL stays uniform and trainable.
  - `durfrz` uses the same law with the phone rows frozen.
- **Rate (A2, A8).** The rate is the greedy emitted rate of phi's genmarg decode (the tau = 1 generative posterior), pooled over the CV holdout: 50 x (sum of non-SIL tokens) / (sum of original 50 Hz frames). It is never an expected count. A restart outside the bed's band [5.80, 14.49] Hz is VOID for selection. The random init sits at 3.7 Hz (expected rate) and 0.98-1.41 Hz emitted at sub-epoch 1.
- **Seeds (A8).** A seed sets both the init and the data order. phi has no dropout, so without the order seed two phi_c-initialised runs would be one run.
- **The wave (A7, A10).** It runs 16 random-init restarts, 4 nulls, 2 phi_c-initialised restarts and exact reruns of seeds 1 and 2.
  - The nulls use a structure-destroyed corpus (units permuted within each utterance, length and speaker embedding kept) and are gated on their permuted holdout.
  - phi_c is the reverse block of the cold arm `k2lat_20_x60` at sub-epoch 60. The phi_c arms keep phi_c's own durations.
  - Selection takes the lowest S among the eligible restarts.
- **Report-only diagnostics, every 4 sub-epochs (A10).** phi's genmarg decode of the 500-utterance D4 dev-other set gives direct PER, Hungarian PER, NMI(symbol, phone) and E[d].
- **Port.** `config/lexlat_v2.py` holds this line: `py()`/`probe()` is the probe, and `wave(phi_c_source=, duration=, num_subepochs=)` is the wave with its label-free selection.

### L2-0 competence ladder (disclosed label-using diagnostic)

- **Phis.** phi_rho is the gold phi's recipe (the 10 h supervised reverse init) fitted to seed gold strings in which a fraction rho of tokens is substituted.
  - Each substitute is drawn from the seed's gold unigram without the original symbol. Length is kept. Realised rates are 0.300, 0.500, 0.700 and 1.000.
  - permphi is the gold recipe on gold strings under one fixed phone permutation (seed 0).
- **Node R, the bridge condition (A4).** theta starts at the cold zero-logit flat init, with the D10e pack constants and the `supphi_k2lat` k2 block. Runs last 8 sub-epochs (kept 1/2/4/8) with both models trainable.
  - Arms: rt_r0 (gold phi, two seeds), rt_r30, rt_r50, rt_r70, rt_r100, rt_perm and cold_ctl (random phi).
  - Classes use dev-other greedy PER at ep8: LIFT < 0.50, PARTIAL < 0.8164 (0.83 minus 0.0136), else NO LIFT.
  - rho*_lift is the largest rho that LIFTs with every smaller rung lifting.
- **Node P.** Here theta = p0, the anchor question. Classes use D10e's bands: HOLD <= 0.2894, COLLAPSE >= 0.70.
- **Competence statistics.** (a) is S. (b) is phi's own decode minus the same-speaker deranged own decode (D10b's form without gold).
- **Port.** The ladder is `config/lexlat_v2.ladder()`, analysis only.

### L2-2 bridge into the joint run

- **Arms.** Each runs D10e pack constants, tau 2.0 from the start, the `supphi_k2lat` k2 block and 8 sub-epochs (kept 1/2/4/8).
  - dec_joint: theta at the cold random init and phi from L2-1, both trainable from step 1.
  - dec_distil: theta first trains one sub-epoch by cross-entropy to the stopped-gradient generative posterior with phi frozen, at tau 2.0 and lr 1e-5, with target the plain l_tau posterior and no k2 term. Then it trains jointly.
  - dec_frz: phi is frozen throughout.
  - dec_joint_s2: a second seed, for the identity band.
- **Baseline (A6).** cold_ctl at L2-2 constants replaces the registered `k2lat_20` baseline. It reuses L2-0's cold_ctl, whose random phi has uniform duration logits. That is accepted as part of phi's init.
- **Port.** L2-2 has no preset in the port.

(src: SAE_4A_lexlat_v2.md § L2-1; § L2-0; § L2-2; § Design review amendments, A1-A10, A18)

## Gates

The original clauses were registered before any job. The amendments listed supersede them.

| Gate | Question | Rule in force |
|---|---|---|
| G4a.L2.1 feasibility | Does a restart run within budget? | A3: PASS if a restart projects to <= 1.5 h, the held-out gain from sub-epoch 3 to 4 is < 0.01 nats/frame for both seeds, and both rates are in band. A9: the wave takes the duration prior that passes with the lower mean S. A10 re-derivation: K*(setting) is the first k >= 4 with S(k-1) - S(k) < 0.01 for both seeds and both rates in band. Among the settings with a K*, pick the lowest mean S at the largest K*. The wave length is that K*, rounded up to a multiple of 4. |
| G4a.L2.2 signal (a spend gate, A7) | Does EM find more than the shuffled corpus keeps? | SIGNAL if the selected S minus the best null S < -max(null range, identity band, 0.01). The identity band is the larger \|S\| difference between seeds 1 and 2 and their reruns. NO SIGNAL means L2-2 is not funded, not that the route fails. If every restart is VOID on rate, the read is CANNOT_TELL. BEYOND PRIVATE CODE (the selected restart against the best phi_c restart) is report only. |
| G4a.L2.3 competence (A5) | Does the selected phi reach the competence bar? | The bar is the statistic in (a) or (b) that is monotone in rho, separates every LIFT arm from every NO LIFT arm, places phi_c on the non-lifting side and places permphi on the side its own arm reads. If no statistic does, the read is CANNOT_TELL. SIGNAL + BELOW funds count-table repair (merge, split and swap on the 40 x 500 rows, accepted only if held-out likelihood rises by more than the null spread). |
| G4a.L2.4 bridge (A6) | Does dec_joint lower the objective and break the code? | Per-frame dev total (l_tau + lexlat_k2 + 3 rate, CV holdout, agg as a point contrast) at ep8, dec_joint paired per utterance against cold_ctl with a speaker-clustered bootstrap. LOWER if the interval is below 0 and \|delta\| > max(B, 0.01), with B = \|dec_joint - dec_joint_s2\|. CODE BROKEN needs LOWER and dev-other greedy PER < 0.50 at a kept epoch. LOWER alone is OBJECTIVE ONLY. |
| L2-2 funding (A7, A18) | | Originally SIGNAL AND at least one rt_r0 seed reading LIFT or PARTIAL. A18 funds the bridge whatever G4a.L2.2 reads. |

(src: SAE_4A_lexlat_v2.md § Gates; § Design review amendments)

## Results

### Feasibility and the k2 pre-flight

- **L2-1 probe (A3/A9): NO WAVE SETTING, failed on the gain clause only.** A restart takes 0.246 h (57 steps per sub-epoch at 3.44 s) against the 1.5 h bar. Rates enter the band from sub-epoch 2-3. The gain from sub-epoch 3 to 4 is still 0.28-0.33:

  | Setting | seed 1: S / Hz / gain | seed 2: S / Hz / gain |
  |---|---|---|
  | durinit | 3.683 / 7.83 / 0.285 | 3.709 / 7.47 / 0.279 |
  | durfrz | 3.717 / 7.59 / 0.299 | 3.739 / 7.42 / 0.296 |
  | uniform | 3.760 / 6.53 / 0.314 | 3.709 / 6.74 / 0.333 |

  E[d] for the phone types: durinit drifts from 4.09 to 4.86-4.94, and uniform falls from 12.8 to 9.95. SIL falls from 24.9 to about 18.4 in every setting.
- **k2 pre-flight for L2-0 node R and L2-2 (pitfall).** With a flat theta and the k2 term on from sub-epoch 1, the first k2 call runs on a uniform posterior. The pre-training stability read (16 utterances in one k2 call at max_active 10000) overflowed k2's int32 arc count, with 2.41e9 arcs in a 20-frame window.
  - Fix, implementation only: score 1 utterance per call and train on small chunks. k2's per-sequence max_active makes chunking exact.
  - The rerun passed: 100/100 steps at a median 16.1 s, peak 76.7 GiB, and 8 sub-epochs projected at 2.23 h.
  - Rejected alternatives change registered constants: a smaller beam, or on-set 8.
- **Memory (A13).** Near-uniform lattices peak at steps 1-2 and then stay flat. Of 95 GB: cold_ctl 83.7 GB, rt_r100 81.7 GB, rt_r70 80.3 GB. Arcs per frame fall from 37,200 at step 0 to about 500-1,400 by step 12. If an arm OOMs, the planned fix is per-chunk backward with gradient accumulation.
- **CV-holdout overlap (A13, pitfall).** 25 of the 285 CV-holdout utterances lie in the fit set of every ladder phi (gold, r30-r100, permphi). Any CV statistic that compares a ladder phi with a non-ladder phi is read on the 260 disjoint utterances, with 285 beside it.

(src: SAE_4A_lexlat_v2.md § Results; § Design review amendments, A13)

### A10: extended EM from random init. The objective prefers non-phonetic phis

- **Design.** The six probe restarts were rerun to 48 sub-epochs (12 passes).
- **Wave decision.** K* is durinit 12, durfrz 10 and uniform 10 (uniform report only). At sub-epoch 12, mean S is 3.4217 for durinit and 3.4571 for durfrz, so **wave = durinit, 12 sub-epochs**. The identity band against the probe is 3.4e-5. Every rate at 48 is 6.8-7.6 Hz.
- **S at sub-epoch 48** (260 set; 285 set in parentheses):

  | Setting | seed 1 | seed 2 |
  |---|---|---|
  | durinit | 3.2990 (3.3040) | 3.3703 (3.3760) |
  | durfrz | 3.3667 (3.3720) | 3.3996 (3.4058) |
  | uniform | 3.3476 (3.3505) | 3.3224 (3.3277) |

  gold phi 3.4735 (3.4702); r100 phi 4.6901 (4.6861). Every restart is below gold from sub-epoch 12 on.
- **Decodes** (D4 dev-other, report only). Direct PER is 0.832-0.856 and Hungarian PER 0.838-0.861, inside the chance band at every checkpoint and flat after sub-epoch 4. NMI(symbol, phone) is 0.079-0.113. E[d] is 6.0-6.3 for durinit and uniform; durfrz is held at 4.41.
- **Conclusion.** Random-init EM finds phis that S prefers to the gold phi, and whose decodes look content-free. A15 later showed that the decodes hide real, mislabelled content.

(src: SAE_4A_lexlat_v2.md § Results)

### A11/A12: exact-EM count-table phi, a second L2-1 family

- **Model.** A free emission table maps (type, duration bucket, position bucket) to 500 units, with no eta, plus a per-type duration table. The E-step is L2-1's posterior. The M-step normalises counts plus a 1e-3 pseudo-count.
- **Recipe, after the literature pass (A12).**
  - The E-step emission is smoothed as 0.9 x table + 0.1 x uniform (Nuhn & Ney 2014).
  - The init is Dirichlet(1). Frequency-rank init hurts at this homophony (Kambhatla et al. 2018).
  - Stage A: 16 restarts per arm (halved from 32 on the measured 33.6 ms per utterance E-step), on a fixed 300-utterance subset, run to a gain < 1e-3 (at most 100 iterations) and ranked by S. The arms are (a) durfrz at tau 1, (b) durfrz with a slow anneal (tau 4, 2, 1.5, each held 5 iterations), and (c) durinit at tau 1.
  - Stage B: the top 4 per arm continue on the 7.1k-utterance sub-epoch until the gain is < 0.01 (at most 40).
  - Nulls are frame-permuted. A run-preserving null is report only.
- **Read: G4a.L2.2 = CANNOT_TELL (NO ELIGIBLE NULL).** All 12 real finishers are in band (6.88-7.38 Hz). None of the 12 frame-permuted null finishers is (1.63-4.09 Hz).
  - Selected: real_c_s16 (durinit, tau 1), S 3.536 (285 set). The real finishers span 3.536-3.608, and the identity band is 0.
  - Every finisher stopped at iteration 2 or 3.
  - The selected S minus the best run-preserving null is -1.696.
  - The stage-A Spearman between iteration 10 and the end is 0.28 / 0.10 / 0.59 (real arms a/b/c).
- **Decodes.** They sit at chance (direct PER 0.844-0.863, NMI(symbol, phone) 0.070-0.094, NMI(symbol, unit) 0.458-0.490). A15-F later found the A10 content profile in the selected table.
- **Pitfall.** Frame permutation destroys unit runs, so null decodes fall below the rate band and cannot be selected. A16 (b) switched to a run-level null.
- **Port.** The count-table family is not in the ported package.

(src: SAE_4A_lexlat_v2.md § Design review amendments, A11, A12; § Results)

### L2-0 ladder: a fitted phi lifts a random theta, rho*_lift = 0.7, G4a.L2.3 = CANNOT_TELL (audited)

| Arm | ep1 | ep8 | Class |
|---|---|---|---|
| rt_r0 / rt_r0_s2 | 0.1807 / 0.1809 | 0.1776 / 0.1786 | LIFT |
| rt_r30 | 0.1861 | 0.1826 | LIFT |
| rt_r50 | 0.1902 | 0.1863 | LIFT |
| rt_r70 | 0.2533 | 0.1909 | LIFT |
| rt_r100 | 0.8445 | 0.8552 | NO LIFT |
| cold_ctl | 0.8881 | 0.8487 | NO LIFT |
| rt_perm | 0.8891 | 0.8655 | NO LIFT |

- **Node P** (theta = p0), at ep8: r30 0.1817, r50 0.1824 and r70 0.1887 read HOLD (PRESERVES by the D14 rule). r100 reads 0.8530, COLLAPSE / DEGRADES (+0.6635).
- **Statistics on the 260 set.**
  - (a) S: gold 3.474, r30 3.719, r50 3.959, r70 4.411, r100 4.690, phi_c 3.499, permphi 4.075.
  - (b) own minus deranged: +4.30, +2.53, +1.91, +1.08, +0.64, phi_c +4.56, permphi +4.10.
  - Both are monotone in rho. But both put phi_c and permphi on the lifting side, although rt_perm does not lift. So no statistic separates, and G4a.L2.3 = CANNOT_TELL.
- **Conclusions.**
  - A gold phi, and one with up to 70 % random label noise, lifts a random theta to about gold level by ep8. Wrong labels on a fully phonetic phi (permphi) block the lift.
  - Withdrawn by A4: "r100 HOLDs on p0, so a fitted phi suffices". A content-free phi holds p0 by exerting no pull.
- **Caveat (audit N1).** The corruption is independent of the acoustics, so each phone's most likely unit stays right below rho 1. rho*_lift therefore does not transfer to an EM phi, whose errors are structured.

(src: SAE_4A_lexlat_v2.md § Results; § L2-0; § Design review amendments, A4, A5)

### A15/A15-E/A15-F: the EM phis are mislabelled and merged, with sub-r70 phone-level content

- **Design (descriptive, label-using, no gate).** Phis are compared directly on D4 dev-other (500 utterances), with speaker-clustered intervals.
  - T1: gold string minus a same-speaker deranged string, in per-frame log p_phi.
  - T2: gold minus gold under a phone permutation.
  - T3: the same map M applied to both gold and deranged.
  - R1/R2: genmarg decode Hungarian PER, under the trigram and under a uniform prior.
  - R3: per-phone JS against gold's mean unit distribution.
  - R4: frame unit-to-phone accuracy against MFA. The majority-unit oracle reaches 0.615.
- **A15-E emission map.** A one-to-one Hungarian assignment of symbols to gold-phi phones on the JS cost. It was needed because the decode-based Hungarian map aligns by identity-label edit distance and recovered only 16 of 40 of permphi's labels.

| phi | T1 own | T2 full | T3 emis | R1 / R2 | R3 JS direct / emis | R4 direct / emis | phones unclaimed |
|---|---|---|---|---|---|---|---|
| gold | 3.93 | 4.47 | 3.93 | 0.193 / 0.276 | 0 / 0 | 0.589 / 0.589 | 0 |
| r30 | 2.22 | 2.44 | 2.22 | 0.240 / 0.363 | 0.116 / 0.116 | 0.571 / 0.571 | 0 |
| r50 | 1.58 | 1.77 | 1.58 | 0.313 / 0.437 | 0.232 / 0.232 | 0.548 / 0.548 | 0 |
| r70 | 0.57 | 0.64 | 0.59 | 0.628 / 0.648 | 0.532 / 0.522 | 0.371 / 0.342 | 15 |
| r100 | 0.00 | 0.06 | 0.18 | 0.830 / 0.862 | 0.728 / 0.652 | 0.092 / 0.136 | 37 |
| A10 EM x6 | 0.08-0.34 | 0.35-0.71 | 1.99-2.20 | 0.838-0.861 / 0.846-0.869 | 0.79-0.87 / 0.50-0.55 | 0.07-0.12 / 0.29-0.31 | 11-19 |
| phi_c | 0.16 | 0.52 | 1.75 | 0.868 / 0.859 | 0.827 / 0.563 | 0.099 / 0.284 | 15 |
| permphi | -0.07 | -0.24 | 3.92 (40/40) | 0.819 / 0.605 | 0.868 / 0.007 | 0.071 / 0.591 | 0 |
| random init | 0.00 | 0.00 | 0.00 | 0.903 / 0.921 | 0.717 / 0.716 | 0.080 / 0.080 | 39 |
| rt_r70 ep8 | 4.04 | 4.33 | 4.04 | 0.228 / 0.326 | 0.111 / 0.111 | 0.564 / 0.564 | 0 |

- **Under their own labels, the EM phis are near content-free.** T1 is 0.08-0.34. R4 is 0.07-0.12, and on non-SIL frames 0.029-0.082.
- **Through the emission map they carry real structure.** T3 relabel (emis minus identity) is +1.2 to +1.7, against r70's -0.04. The map agrees with the own labels on only 1-6 of 40 symbols.
- **They are merged.** The symbols claim only 21-29 gold phones. The unclaimed phones hold 19-40 % of non-SIL frames (e.g. AO, AW, OY and UH in 6/6), while SIL, N, T, W, AY, Z and M each take 3-4 symbols.
- **A15-F content resolution.**
  - Oracle partitions give: random 40-way R4 0.118 (max 0.129) and within-class 0.370 (max 0.429); 7 manner classes R4 0.287, class share 0.767, within-class 0.374; 13 classes R4 0.377.
  - The EM phis read class share 0.50-0.56 and within-class 0.54-0.62. r70 reads 0.49 / 0.70 and gold 0.73 / 0.81.
  - Every EM gain beats its own 5 sharp permuted-unit nulls: T3 1.99-2.20 against 0.21-0.26, R4 emis 0.29-0.31 against 0.10-0.12.
  - A11's selected table has the same profile (R4 emis 0.283, within-class 0.565).
  - The rule "ABOVE MANNER LEVEL" was amended at build to beat the maximum of the 7-class oracle, the random-partition maximum and the phi's own nulls, because within-class accuracy is inflated for weak partitions. r100 also passes it narrowly (0.433), so without an own null the flag is weak.
- **Conclusion (audited).**
  - The EM phis are mislabelled and merged. Their content is phone-level within classes, with the errors crossing classes. It is close to, and somewhat below, r70's.
  - Chance-level decode PER says nothing about a phi's content in either direction.
  - phi_c, the cold line's own phi, has the same profile, slightly weaker. It is a mislabelled partition with structure, not a content-free one.
  - r70 lifts because its labels are right.
- **Overturned.** Two earlier readings are overturned by A15-F: that the EM content is "at about the manner-class level", and that a relabelled gold fit worse than r100's shows missing content. That fit comparison is confounded by sharpness: a sharp content-free phi scores -10, and the EM phis score -6.0 to -6.7.
- **Port.** The battery is not in the ported package.

(src: SAE_4A_lexlat_v2.md § A15 and A15-E read; § A15-F read; § Design review amendments, A15)

### A16 (a) and (a2): does S prefer the right labelling, and does a weighted prior fix it?

- **(a) OBJECTIVE LABEL-BLIND OR WRONG.**
  - dS = S(emission-map labelling) - S(identity) is positive for 6/6 EM phis: +0.123 to +0.308, on 250-259 of 260 utterances.
  - The control is valid: permphi's true inverse gives dS -0.598, reaching gold's level.
  - Audit correction: relabelling keeps the emission and duration terms, so dS measures only the trigram's reaction to a renaming of a co-adapted phi. A positive dS is expected by construction, and it shows no preference for wrong content. No right-labelled negative control was run.
- **(a2) NO LAMBDA <= 3.** S_lambda = -(1/T) log sum_y P_LM(y)^lambda P_phi(x|y).
  - Gold never ranks first. S(gold) - S(EM) is +0.074 to +0.175 at lambda 1 and +0.091 to +0.202 at lambda 3.
  - The ladder stays monotone (gold/r30/r70/r100: 3.474/3.719/4.411/4.690 at lambda 1; 4.601/4.731/5.072/5.243 at lambda 3).
  - The dS of (a) grows with lambda. phi_c drops below gold at lambda 3 (4.559 against 4.601).
- **Conclusion.** Re-weighting the phone trigram is ruled out, and a permutation-only label search on S is unsupported.
- **Withdrawn by audit.** The reading "model error" in Yin et al.'s sense is withdrawn. Gold was fitted on 2821 utterances under a different criterion, and the gap sits in the channel terms (the prior term costs about 0.58 nats/frame for both), so gold is not the same-data model. A14 (ii) settles the matched comparison.

(src: SAE_4A_lexlat_v2.md § A16 (a) read; § A16 (a2) read)

### A14 (ii): PHONETIC BASIN LOWER (analysis only, supervised init; audited)

- **Design.** The A10 durinit recipe (tau 4 then 1, 48 sub-epochs) runs with phi initialised from the L2-0 fits (gold, r30, r70, r100), one GPU each.
- **Gate.** PHONETIC BASIN LOWER needs S_g < S_min - 0.01 and gold-init Hungarian PER at 48 < 0.50. NON-PHONETIC PREFERRED needs S_g > S_min + 0.01 or PER >= 0.83. The reference S_min = 3.2990 is the best A10 restart.

| init | S at 0 | S at 48 | PER at 0 | PER at 48 |
|---|---|---|---|---|
| gold | 3.474 | 3.216 | 0.193 | 0.353 |
| r30 | 3.719 | 3.210 | 0.240 | 0.394 |
| r70 | 4.411 | 3.271 | 0.608 | 0.495 |
| r100 | 4.690 | 3.378 | 0.826 | 0.860 |
| A10 random (6) | | 3.30-3.40 | | chance band |

- **Result.** S_g - S_min = -0.083 [-0.094, -0.072], with 222/260 utterances lower. The margin is about the six-restart seed range (0.10), and the gold arm has one seed.
- **Audit correction.** The gold (and r30, r70, r100) inits kept their MFA-fitted durations instead of durinit. So every arm that reached the basin carried supervised segmentation. r100 carried the same segments with random labels and did not reach it.
- **Conclusion.**
  - At matched data and criterion, S's lower basin is phonetic, and random-init EM is search-limited.
  - Not licensed: that S's global minimum is phonetic, that search reaches the basin without labels, or that S rewards accuracy inside the basin.
  - EM degrades gold: PER rises 0.193 -> 0.300 (sub-epoch 4) -> 0.328 (12) -> 0.353 (48) while S falls. Most of the damage comes in the tau = 4 sub-epoch, which sends every init to S 4.71-4.92 (gold 3.474 -> 4.814).
  - A perfect search would therefore stop at about PER 0.35-0.5.
  - There are two defects: the search, and the objective inside the basin.

(src: SAE_4A_lexlat_v2.md § A14 (ii) read; § Design review amendments, A14, A17)

### A16 (b) stage 0: J SEES THE KEY, fragile (audited)

- **Key objective J(k).** This is CPU work on count tables.
  - A key maps each of the 500 units to one of 40 symbols. The train-stream units are mapped and their runs collapsed. Overlong runs are split into ceil(d/D_k) segments.
  - J sums the trigram log P of the symbol string, the ML emission log-likelihood of units given symbols (train-side counts) and the general-knowledge duration log-prior, per frame. It is read on the 260 held-out utterances.
  - Keys outside the rate band are void.
  - An EM phi's key is argmax_s m(u|s) pi(s), with pi fitted label-free to the train-side unit counts. The prior-occupancy alternative put 0.487 on SIL against a gold share of 0.058 and was rejected.
- **Gate.** J SEES THE KEY needs J monotone on the reassignment ladder (gold > K30 > K70 > K100) and J(gold) above the best EM/A11 key and the best random key by more than max(0.01, the range over 20 random keys).
- **Result.** gold -4.818, K30 -5.610, K70 -6.334, K100 -6.560. The best EM key (a10_durfrz_s01) reads -4.971, a gap of 0.153 against a margin of 0.144.
- **Fragility.**
  - The verdict clears by 0.009, and only on the trigram term (+0.239). The emission (-0.080) and duration (-0.005) terms favour the EM key.
  - The train-side margin clause fails (0.1416 against 0.1466).
  - A fresh draw of random keys would reverse the verdict about 40 % of the time.
  - J does not separate gold from the r30 key (-4.822).
- **Key agreement with the gold key.** The A10 argmax keys read identity agreement 0.07-0.13 and many-to-one agreement 0.46-0.50. They are good partitions with wrong names, and their label-independent emission term beats gold's (-3.27 against -3.35). The Spearman of J against identity agreement is 0.75 over all 56 keys and 0.32 without the random keys. So J ranks mislabelled clean partitions above 70 %-correct noisy ones.
- **Conclusion.** Stage 1 is funded. It is not shown that J's maximum lies at or near gold.

(src: SAE_4A_lexlat_v2.md § A16 (b) stage 0 read; § Design review amendments, A16)

### Adjacent-repeat handling in the objective (code review)

- **No double counting.** No term reads a run of identical frames as several phones, and no path count is inflated.
  - In l_tau, the repeat arc keeps the trigram history and adds no prior or segment. A token equal to its predecessor, SIL included, is masked.
  - In k2, H has one path per frame string.
  - Both terms admit collapsed runs, SIL as an ordinary symbol, and never two equal phones in a row.
- **Disclosed costs.**
  - A true repeat can surface only as X SIL X. Dev-other has 989 repeat pairs in 177,275 reference phones (0.56 %), which sets a PER floor of about 0.56 % without SIL.
  - The trigram was fitted on uncollapsed text (0.265 % repeat tokens). The masked mass is not renormalised, costing 0.0027 nats/token.
  - 533 of 151,731 lexicon words have an internal repeat and are unreachable in k2. How strongly k2 pushes cross-word repeats toward X SIL X is unmeasured.
  - phi's HSMM on its own allows same-symbol neighbours, but never inside l_tau.

(src: SAE_4A_lexlat_v2.md § Adjacent-repeat handling in the objective)

## Open at the move

At the move, no result exists for the wave, A14 (i), A17, A18, A19 or the A16 (b) stage-1 key search. Their designs and gates below are registered and unread. **Several were still running on JUPITER when the logs were frozen (2026-09-24):** the L2-1 wave (six 4-GPU packs), the A14 (i) pack (pending), A17 (i) and A17 (ii), the keyinit pack (gold-key control plus A17 (iii); pending) with the A18 (b) lift pack waiting on it, and the stage-1 key search. Their outputs may exist on JUPITER: ask the user before rerunning any of them here. A19 was built and in review, not launched.

**Planned experiments and registered gates**

| Experiment | Design | Gate |
|---|---|---|
| L2-1 wave | durinit, 12 sub-epochs: 16 restarts, 4 nulls, 2 phi_c restarts, 2 reruns | G4a.L2.2 (A7), G4a.L2.3 (A5). BEYOND PRIVATE CODE is reported. |
| A14 (i) rt_em | L2-0 node-R recipe, with phi from each of the 4 durinit/durfrz A10 restarts at sub-epoch 48; one 4-arm pack, no selection | EM PHI LIFTS if any arm reads LIFT or PARTIAL (A4 bands) at ep8; DOES NOT LIFT if all 4 read NO LIFT. A paired row against cold_ctl is reported. |
| A17 (i) | A14 (i)'s recipe with the A14 (ii) sub-epoch-48 phis: gold-EM (0.353), r30-EM (0.394), r70-EM (0.495), r100-EM (0.860, negative control) | BASIN SUFFICIENT if gold-EM or r30-EM lifts and r100-EM does not; INSUFFICIENT if gold-, r30- and r70-EM all read NO LIFT (this withdraws A16 (b) stages 1-2); VOID if r100-EM lifts. |
| A17 (ii) | A14 (ii) recipe at tau = 1 from sub-epoch 1 (no tau = 4), 12 sub-epochs, gold and r70 inits | Gold drift PER(12) - 0.193: ANNEALING-DOMINATED < 0.05, which amends A16 (b) stage 2 to tau = 1; OBJECTIVE DRIFT >= 0.10; MIXED otherwise. |
| A17 (iii) | A10 durinit recipe with emission rows from gold, r30 and r70 and durinit durations (G-dur, r30-dur, r70-dur) | S at 48 against 3.289 (unrounded 3.28903): SEGMENTATION-CARRIED if G-dur >= bar; LABELS SUFFICE if G-dur and r70-dur are both < bar; PARTIAL-LABELS NEED SEGMENTS if only G-dur < bar. Either of the first and third holds the stage-2 key arms. Boundary F1, segment rate and R-value are reported. |
| A16 (b) stage 1 | Unit-reassignment and symbol-swap moves by annealed ICM/Gibbs on train J. Starts: 64 random, 7 EM argmax, 16 cluster-then-decipher, 7 relabel; informed starts run a full schedule (T0) and a warm one (T0/100). Null: run-level segment permutation. Rate repair has a stall rule; starts still out of band are VOID. | Top 4 by held-out J, label-free. Agreement with the gold key is reported. |
| A16 (b) stage 2 | A10 recipe with emission rows from the key's smoothed counts, durinit. Gold-key control first. | Control: GOLD KEY REACHES BASIN if S at 48 < 3.289, else the key arms are held. Key arms: KEY BASIN if the best S < 3.289. The key-to-phi pre-activation scale PREACT_ON = 2.0 is an untraced constant, and a failed control must name it. |
| A18 bridges | (a) the wave's selected restart -> L2-2 (G4a.L2.4); (b) the four keyinit-pack phis -> a lift pack; (c) the S-best key arm -> L2-2; (d) the A17 (ii) phis only if A17 (i) gold-EM reads NO LIFT and A17 (ii) reads ANNEALING-DOMINATED | Each runs whatever its S-gate reads. Hold only if a completed same-line lift read was NO LIFT on every arm and the candidate's S is not lower by > 0.01. (b): DURINIT BASIN LIFTS if G-dur or r30-dur lifts; DOES NOT LIFT if all three do not; MIXED if only r70-dur lifts. |
| A19 trigram-only ladder | The L2-0 node-R recipe without the k2 block: tri_r30, tri_r50, tri_r70, tri_r100 | rho*_tri from LIFT only: SAME LADDER if rho*_tri = 0.7; K2 NEEDED ABOVE rho*_tri if it is lower; VOID if tri_r100 reads LIFT or PARTIAL. Paired rows against rt_rX use M = 0.010. |

**Unresolved questions**
- Can a label-free search reach the phonetic basin (A16 (b))? And does J's maximum sit at gold, given that J ranks mislabelled clean partitions above noisy correct ones?
- Did the A14 (ii) basin depend on supervised MFA durations (A17 (iii))? The literature suggests resegmentation under the trigram, which the HSMM EM already does, with a self-supervised segmenter only as its init. The key's own run segmentation is the weakest fallback, since run merging overcounts boundaries 2.6-3.3x.
- Does an EM-degraded phonetic phi (PER 0.35-0.5) still lift a random theta (A17 (i))? Does an A10 phi lift at all (A14 (i))?
- Objective drift inside the basin: is the tau = 4 sub-epoch or tau = 1 EM responsible (A17 (ii))?
- The candidate mechanism "a 500-way per-frame emission term dwarfs the phone prior" is untested.
- Does A15-E-style content exist in the A11 tables beyond the selected one?
- Handoff items needing new training: a label-free segmentation init, a duration-shape run, and count-table repair (not built). Also the literature's list if the families read NO SIGNAL: more restarts and iterations; a coarse-to-fine unit inventory (about 100 classes); a word-level LM in the E-step with a wider beam; a sparse channel prior.

**Values the ported package leaves open, against the README's "Open values"**
1. `WAVE_DURATION_SETTING` and 2. `WAVE_NUM_SUBEPOCHS` are open in the README. The A10 read decided them before the move: **durinit** and **12** sub-epochs. They should be set from that read, not re-derived.
3. phi_c's source (`k2lat_20_x60` at sub-epoch 60) has no preset. It is needed for the wave's 2 phi_c restarts and for phi_c's reference rows.
4. The L2-0 random-init phi rung is None in `ladder.ladder_phis`. In the source, cold_ctl had no standalone reverse checkpoint either.
5. These have no entry point in the port: the A11 count-table family, the A14-A19 arms, the A15 battery, the A16 key search and L2-2 (including dec_distil's tau 2.0 and lr 1e-5).

(src: SAE_4A_lexlat_v2.md § State; § Design review amendments, A14-A19; § Constraints)
