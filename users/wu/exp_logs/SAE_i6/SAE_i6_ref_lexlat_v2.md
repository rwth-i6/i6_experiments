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

**Input defect (amended 2026-09-25, src: SAE_ref.md @7e7c38aee § Input defect).** The phone prior, the lexlat word set, trie and HLG (151,731 words) and the rate target in this file come from JUPITER's truncated phone text; `SAE_i6_ref.md` section 6 holds the defect. Comparisons within JUPITER are unaffected: all arms share the prior.

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
- **Conclusion.** Stage 1 is funded. It is not shown that J's maximum lies at or near gold. **Amended by the stage-1 read (7e7c38aee):** gold is not J's maximiser; see "A16 (b) stage 1" below.

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


## Final reads after the move (JUPITER, 2026-09-24 to 2026-09-25)

Every registered run of the phase has been read and audited on JUPITER. JUPITER's State: "Every registered run of this phase has now been read. The next training changes wait for the user's decision on the proposal in `SAE_4A_rename.md`." A18 (d) was not triggered (below), so no registered gate of this phase remains unread. S is on the 260 set unless marked; PER is dev-other greedy PER on all 2,864 utterances unless marked "generative" (D4, 500 utterances). (src: SAE_4A_lexlat_v2.md @7e7c38aee § State)

### G4a.L2.2 wave: SIGNAL, NOT BEYOND PRIVATE CODE (audited CONFIRMED_WITH_CORRECTIONS)

- **Question.** Does EM find more than the shuffled corpus keeps (A7)? BEYOND PRIVATE CODE is report only.
- **Numbers** (S at sub-epoch 12, 285 set). Selected em_s13, S 3.3863 (next em_s01 3.3981), 7.58 Hz; all 16 restarts in band, none VOID. Best null null_s02 5.7133; null range 0.0027, identity band 2.7e-5, margin 0.01. Gap 2.327, and em_s13 beats the best null on all 285 utterances: **SIGNAL**. Best phi_c restart phic_s01 3.3441 is 0.042 below em_s13: **NOT BEYOND PRIVATE CODE**.
- **Reading.** A7 expects SIGNAL from any segmental fit, so it shows no phonetic content. em_s13's generative PER (from A18 (a)'s dec_frz) is direct 0.857, Hungarian 0.859, NMI 0.076, the worst edge of the A10 band.
- **Audit corrections** (verdict unchanged): report.txt quoted A7's funding rule as LIFT only (registered: LIFT or PARTIAL); the bridge checks neither the verdict nor the rt_r0 condition, which A18 (a) makes moot.
- **G4a.L2.3.** The log records no separate read on the selected phi. L2-0 found no separating statistic, for which A5 registers CANNOT_TELL. The count-table repair that SIGNAL + BELOW would fund is new training and was not built; the bridge ran without it (A18 (a)).
- Checkpoint: `KCj5mptWgBqb/output/em_s13/models/epoch.012.pt`.

(src: SAE_4A_lexlat_v2.md @7e7c38aee § G4a.L2.2 wave read; `reports/audit_wave_read_2026-09-24.md`)

### A14 (i) rt_em: EM PHI DOES NOT LIFT (audited CONFIRMED_WITH_CORRECTIONS)

- **Question.** Does an A10 EM phi (sub-epoch 48) lift a random theta under L2-0's node-R recipe (configs equal rt_r70 except the phi path)?
- **Numbers.** PER ep1 / ep2 / ep4 / ep8, and ep8 minus cold_ctl [95 % CI]:

  | Arm | PER | ep8 band | minus cold_ctl |
  |---|---|---|---|
  | cold_ctl | 0.888 / 0.886 / 0.855 / 0.849 | NO LIFT | |
  | durinit s1 | 0.864 / 0.852 / 0.834 / 0.840 | NO LIFT | -0.009 [-0.013, -0.005] |
  | durinit s2 | 0.839 / 0.831 / 0.824 / 0.820 | NO LIFT | -0.029 [-0.032, -0.026] |
  | durfrz s1 | 0.866 / 0.864 / 0.841 / 0.842 | NO LIFT | -0.006 [-0.010, -0.003] |
  | durfrz s2 | 0.845 / 0.829 / 0.820 / 0.821 | NO LIFT | -0.028 [-0.032, -0.025] |

- **Reading.** No arm leaves the chance band; the s2 arms sit 0.004 above the PARTIAL bar (0.8164). An EM phi moves a random theta slightly, far from the 0.18-0.19 of L2-0's lifting phis. The job reports no generative PER.
- **A18 hold rule applied to (a)** (after (a) was launched): the registered clause assumes an S at 48, which em_s13 lacks, so it reads CANNOT_TELL and gives no ground for withholding. (em_s13 at 12 against the arms at 48: +0.082 [+0.072, +0.092], an auditor calculation; at matched sub-epoch 12, em_s13 is 0.0107 below durinit s1.)

(src: SAE_4A_lexlat_v2.md @7e7c38aee § A14 (i) read; `reports/audit_a14i_hold_2026-09-24.md`)

### A19 trigram-only ladder: SAME LADDER (audited CONFIRMED)

- **Question.** Does a partly wrong phi still lift a random theta without the k2 block? Single delta against each rt_rX: the 17 k2 config lines (verified: tri logs have no k2 lines, rt logs 496; step 0 identical, same phi fits).
- **Numbers.** PER ep1 / ep2 / ep4 / ep8:

  | Arm | PER | ep8 band | tri minus rt, ep1 / ep8 | phi generative PER ep8 (direct / Hungarian / NMI) |
  |---|---|---|---|---|
  | tri_r30 | 0.203 / 0.282 / 0.298 / 0.268 | LIFT | +0.017 / +0.085 | 0.260 / 0.260 / 0.788 |
  | tri_r50 | 0.210 / 0.277 / 0.300 / 0.263 | LIFT | +0.020 / +0.077 | 0.257 / 0.257 / 0.784 |
  | tri_r70 | 0.335 / 0.368 / 0.354 / 0.329 | LIFT | +0.081 / +0.138 | 0.316 / 0.357 / 0.725 |
  | tri_r100 | 0.831 / 0.899 / 0.903 / 0.898 | NO LIFT | -0.014 / +0.043 | 0.874 / 0.876 / 0.068 |

- **Verdict.** rho*_tri = 0.7, monotone, not VOID; no alternative reading changes it. Seven of eight paired rows read K2 HELPS (M = 0.010); r100 at ep1 reads TRIGRAM ENOUGH. The ep1 labels for r30 and r50 are the least robust (M comes from the k2 runs' seed pair; report only).
- **Reading.** The phone trigram alone lets a phi with up to 70 % of its fit tokens substituted lift a random theta. k2 is not needed for the lift but improves ep8 by 0.08-0.14 PER. Without k2, r30 and r50 are best at ep1 (about 0.20), worsen to about 0.30 by ep4 and recover to 0.26-0.27 by ep8, matching D10e's trigram-only drift from p0.
- **Ladder rungs run on JUPITER.** With or without k2, only rho 0, 0.3, 0.5, 0.7 and 1.0 were run; no rung lies between 0.7 and 1.0 (L2-0; A19).

(src: SAE_4A_lexlat_v2.md @7e7c38aee § A19 read; § Design review amendments, A19; `reports/audit_a19_read_2026-09-24.md`)

### A17 (i): BASIN SUFFICIENT; A17 (ii): OBJECTIVE DRIFT, at the bar (audited CONFIRMED_WITH_CORRECTIONS)

- **(i) Question.** Does an EM-degraded phonetic phi (A14 (ii)'s sub-epoch-48 checkpoints) still lift a random theta? Configs differ from A14 (i) only in the phi path. PER ep1 / ep2 / ep4 / ep8, and phi generative PER init -> ep8 (direct / Hungarian):

  | Arm | PER | ep8 band | phi generative PER |
  |---|---|---|---|
  | gold-EM | 0.284 / 0.255 / 0.253 / 0.200 | LIFT | 0.353 -> 0.232 / 0.232 |
  | r30-EM | 0.323 / 0.283 / 0.224 / 0.201 | LIFT | 0.394 -> 0.239 / 0.239 |
  | r70-EM | 0.434 / 0.432 / 0.412 / 0.362 | LIFT | 0.495 -> 0.393 / 0.432 |
  | r100-EM | 0.869 / 0.859 / 0.836 / 0.839 | NO LIFT | 0.860 -> 0.858 / 0.858 |

  **BASIN SUFFICIENT.** r100-EM clears the NO LIFT bar by only 0.0225. Where the phi sits decides whether it lifts; joint training repairs the basin phis' generative PER by 0.10-0.16. These phis kept MFA durations (settled by A17 (iii) and A18 (b)).
- **(ii) Question.** Is the in-basin drift from the tau = 4 sub-epoch or from tau = 1 EM? Generative PER at 0 / 4 / 8 / 12:
  - gold 0.193 / 0.266 / 0.283 / 0.295 (A14 (ii): 0.193 / 0.300 / 0.317 / 0.328); S 3.474 / 3.264 / 3.249 / 3.241.
  - r70 (direct / Hungarian) 0.608/0.628, 0.385/0.413, 0.382/0.412, 0.393/0.421 (A14 (ii): 0.608/0.628, 0.477, 0.478, 0.482); S 4.411 / 3.400 / 3.304 / 3.266.
  - **OBJECTIVE DRIFT**: gold drift +0.1019, 0.0019 (57 phones) over the 0.10 bar. The audit's bootstrap SE is 0.0024-0.0031, so MIXED lies within sampling error; ANNEALING-DOMINATED is excluded.
  - Reading: dropping tau = 4 removes about a quarter of the gold damage at 12 (0.295 against 0.328). The rest comes at tau = 1 while S falls by 0.23, so S's optimum near gold is not gold.
- **Audit correction.** No same-config PER repeat exists for (ii); the 0.01-0.03 spread cited under A17 is the joint bed's, not phi EM's.
- **Consequences as registered.** A16 (b) stage 2 not withdrawn; no tau = 1 amendment to stage 2; A18 (d) not triggered (gold-EM lifts, and the drift is not below 0.05).

(src: SAE_4A_lexlat_v2.md @7e7c38aee § A17 (i)/(ii) read; `reports/audit_a17_reads_2026-09-24.md`)

### Gold-key control: GOLD KEY REACHES BASIN; A17 (iii): LABELS SUFFICE (audited CONFIRMED_WITH_CORRECTIONS)

- **Question.** Does a key-to-phi init reach the basin (S at 48 < 3.28903), and did the A14 (ii) basin need MFA-fitted durations?
- **Durations.** All four inits hold exactly the durinit values (phones 4.4138, SIL 26.0); the gold key's phi takes only `key.json` and unsupervised unit frame counts.
- **S at 48.** gold_key 3.20704 (-0.082): **GOLD KEY REACHES BASIN**. G-dur 3.22387 (-0.065) and r70-dur 3.26568 (-0.023): **LABELS SUFFICE**. r30-dur 3.21151, report only.
- **Generative PER** (direct / Hungarian / NMI) at 0 / 48: gold_key .327/.385/.740 -> .346/.391/.695; G-dur .197/.197/.836 -> .345/.345/.710 (A14 (ii) twin .353); r30-dur .248 -> .368 (twin .394); r70-dur .605/.622/.452 -> .442/.466/.606 (twin .495).
- **Boundaries** against MFA (20 ms): F1 at 48 is 0.776-0.799 for all four (A14 twins 0.777-0.800; A14 r100 0.660), at 10.0-10.5 Hz against MFA's 11.90 Hz. gold_key starts at F1 0.813.
- **Audit corrections.** G-dur's and r70-dur's emission heads still carry MFA-fitted segment structure (JS 0.21 and 0.16 nats per phone type; G-dur init F1 0.834), so LABELS SUFFICE licenses only "the A14 (ii) basin did not need MFA-fitted durations". The clean no-segmentation evidence is the gold-key control. Not holding the key arms rests on r70-dur's 0.023 margin, one seed; the A10 restarts span 0.10 in S.
- **Reading.** A key-to-phi conversion from type-level counts reaches the basin with no supervised segmentation and ends at generative PER 0.35. S and PER agree between basins but not inside one (G-dur's PER rises 0.148 while S falls). Neither gate holds the stage-2 key arms; the PREACT_ON caveat is moot.

(src: SAE_4A_lexlat_v2.md @7e7c38aee § Gold-key control and A17 (iii) read; `reports/audit_keyinit_reads_2026-09-24.md`)

### A16 (b) stage 1 key search: J's optima are clusters with wrong names; gold is not J's maximiser (audited CONFIRMED_WITH_CORRECTIONS)

- **Question.** Label-free top 4 keys by held-out J; agreement with the gold key reported.
- **Status.** real ok 196, failed 0; null ok 172, VOID-ON-RATE 8; loop_error null; 15 null runs truncated. Stage 1's J matches stage 0 term by term on 27 shared keys.
- **Numbers.** Top 4, all cluster starts under the warm schedule: J -4.5525 / -4.5647 / -4.5671 / -4.5755. References: gold -4.8180, best A10 argmax start -4.9714, K30 -5.6097, best null -5.8543. All 124 real finals beat gold; every start family reaches -4.55 to -4.75. The 0.24-0.27 margin over gold splits into emission 0.20-0.22, trigram 0.01-0.04, duration < 0.01; over all finals the trigram term ties gold (mean -0.002).
- **Agreement with the gold key** (identity / many-to-one): selected 0.07-0.14 / 0.49-0.58; A10/A11 argmax 0.110 / 0.486 before, 0.10-0.12 / 0.50-0.51 after; random 0.026 / 0.223 before, 0.098 / 0.487 after; K30 0.70/0.71, K70 0.30/0.35, K100 0/0.22.
- **Reading.** The search sharpens clusters (many-to-one above K70's) but does not find names (identity below K70's 0.30). J prefers these keys through the name-free emission term.
- **Licensed:** gold is not J's maximiser, and J's found optima are clusters with wrong names. **Not licensed:** that no near-gold key beats -4.55 (no search started from gold, r30 or K30); anything about stage-2 S or PER.

(src: SAE_4A_lexlat_v2.md @7e7c38aee § A16 (b) stage 1 read; `reports/audit_keysearch_s1_2026-09-24.md`)

### A20 oracle names on the found keys: NAME-BLIND by the registered rule; J penalises gold-matching names (audited CONFIRMED_WITH_CORRECTIONS)

- **Question** (registered after stage 1, before any job; label-using, CPU, gates nothing). Does J reward the right names on the found partitions? NAMES VISIBLE if J(oracle 1:1 rename) - J(found) > 0.01 on at least 3 of the 4 selected keys; NAME-BLIND if <= 0.01 on at least 3; MIXED otherwise.
- **Numbers.** J(b) - J(a): -0.573, -0.436, -0.422, -0.435, entirely in the trigram term (partition unchanged). The A10/A11 argmax finals give -0.39 to -0.62. Found keys fit the trigram better than gold at a matched rate (lm -0.948 at 9.20 Hz against -0.990 at 9.47 Hz). 0 of 4 above 0.01: **NAME-BLIND**.
- **Audit correction that changes the reading.** The registered gloss "the trigram term does not reward the right names" is wrong in direction: the trigram term *penalises* the gold-matching 1:1 names by 0.42-0.57 on these partitions, and on the gold partition penalises scrambled names by 0.69-1.09 (unregistered control), so it does see names. The found keys are swap-optimal, and a label-free name-swap climb from (b) moves away from gold (identity 0.39-0.47 to 0.13-0.26). No 1:1 naming is near right: 10-15 gold phones each hold 2-4 symbols, and 14-18 phones are no symbol's majority.
- **Unregistered (audit).** The emission term's 0.20-0.23 margin over gold splits into 0.10-0.17 from less d_min absorption and 0.06-0.10 from higher symbol entropy.
- **Licensed:** a rename step under J would not help; at key level the objective, not the search, rejects gold-agreeing names, and the name-free emission term (mostly absorption) selects partitions that split and merge phones. **Not licensed:** that the cost work belongs to the name signal alone ((b) cannot separate a name deficit from a partition deficit); anything about S or phi-level renaming.

(src: SAE_4A_lexlat_v2.md @7e7c38aee § Design review amendments, A20; § A20 read; `reports/audit_a20_oracle_names_2026-09-24.md`)

### A16 (b) stage 2 key arms: KEY BASIN, on S only (audited CONFIRMED_WITH_CORRECTIONS)

- **Question.** Does the best key arm (A10 recipe with the key init, durinit, one seed per key) reach S at 48 < 3.289?
- **Numbers.** S at 48: rank1 3.27275, rank2 3.29550, rank3 3.28529, rank4 3.33540. rank1 against S_min 3.29903: -0.026 [-0.037, -0.015]: **KEY BASIN**. Generative PER at 48 (direct / Hungarian / NMI): rank1 0.858 / 0.861 / 0.077, rank2 0.847 / 0.798 / 0.099, rank3 0.849 / 0.842 / 0.096, rank4 0.872 / 0.873 / 0.068: the chance band.
- **Audit corrections that change the reading.** KEY BASIN licenses S only: not the phonetic basin (A14 (ii) also needed PER < 0.50), not names, not lift. rank1's margin (0.016) is below the A10 same-recipe seed spread of S (0.025-0.071); the four arms average 3.297, above the bar. rank3's 0.004 is within jitter.
- Name tracking on these arms (AN-5 in `SAE_4A_rename.md`): EM LOCKED, key identity 0.05-0.17 throughout.

(src: SAE_4A_lexlat_v2.md @7e7c38aee § A16 (b) stage 2 read; `reports/audit_keyarms_read_2026-09-25.md`)

### A18 bridges (a)-(d)

- **(a) wave phi em_s13 -> L2-2, G4a.L2.4: LOWER, OBJECTIVE ONLY (audited CONFIRMED_WITH_CORRECTIONS).** dec_joint minus cold_ctl at ep8: -0.1158 nats/frame [-0.1257, -0.1035], B = 0.0093, margin 0.01: LOWER. Terms: l_tau -0.240 (the frozen phi, dec_frz, gives 85 %), lexlat_k2 +0.092, 3 x rate +0.033. PER ep1 / 2 / 4 / 8: dec_joint 0.862 / 0.854 / 0.846 / 0.842; dec_distil 0.854 / 0.853 / 0.847 / 0.846; dec_frz 0.862 / 0.859 / 0.843 / 0.846; dec_joint_s2 0.863 / 0.855 / 0.843 / 0.846; cold_ctl 0.888 / 0.886 / 0.855 / 0.849. Correction: cold_ctl lacks A9's durinit (a build choice made before any result); its share of the drop is unmeasured, but PER alone rules out CODE BROKEN. (src: § A18 (a) bridge read; `reports/audit_a18a_bridge_read_2026-09-24.md`)
- **(b) keyinit phis -> lift pack: DURINIT BASIN LIFTS (analysis only, label-built inits; audited CONFIRMED_WITH_CORRECTIONS).** PER ep1 / 2 / 4 / 8, all LIFT: gold-key 0.275 / 0.258 / 0.235 / 0.208; G-dur 0.275 / 0.237 / 0.232 / 0.193; r30-dur 0.290 / 0.264 / 0.228 / 0.206; r70-dur 0.355 / 0.345 / 0.278 / 0.228. Jointly trained phi generative PER at ep8 (direct / Hungarian / NMI): 0.258/0.310/0.771, 0.221/0.221/0.810, 0.243/0.243/0.787, 0.284/0.335/0.741. Paired rows against A17 (i)'s MFA-duration arms: G-dur -0.0073 [-0.0093, -0.0054], r30-dur +0.0051 [+0.0034, +0.0070] (ties within the bed's seed spread 0.004-0.010; neither help nor harm from MFA durations); r70-dur -0.1344 [-0.1397, -0.1295] (differs in more than durations; one seed). Corrections: the gold-key and r70-dur ep8 forwards ran on a faulty GPU and are not bit-verified (the verdict does not depend on them; the r70 row rests on one); every init is built from labels, so the read says nothing about label-free phis. (src: § A18 (b) lift read; `reports/audit_a18b_lift_2026-09-25.md`)
- **(c) S-best key arm (rank1) -> L2-2, G4a.L2.4: LOWER, OBJECTIVE ONLY; NO LIFT (audited CONFIRMED_WITH_CORRECTIONS).** The hold rule did not fire (the gold-key arm lifts). dec_joint minus cold_ctl at ep8: -0.1609 [-0.1710, -0.1488], B = 0.0038: LOWER (also under bootstrap seed 7 and frame weighting). l_tau -0.284 (97 % from the frozen phi), lexlat_k2 +0.081, 3 x rate +0.042. PER ep1 / 2 / 4 / 8: dec_joint 0.862 / 0.852 / 0.840 / 0.843; dec_distil 0.857 / 0.854 / 0.844 / 0.843; dec_frz 0.861 / 0.857 / 0.850 / 0.846; dec_joint_s2 0.863 / 0.850 / 0.843 / 0.843; cold_ctl as in (a). Lowest of all 20 cells 0.840; lift test (A4 bands): **NO LIFT**. Phi generative PER at ep8 stays at chance (0.858 / 0.855 / 0.077). Corrections: the baseline's uniform durations against rank1's durinit plus EM durations have an unmeasured share of the drop; LOWER alone licenses no claim; the read licenses not funding this phi, not that the key line cannot lift. (src: § A18 (c) bridge read; `reports/audit_l22_keyarm_bridge_2026-09-25.md`)
- **(d) A17 (ii) phis: not triggered.** Its condition (A17 (i) gold-EM NO LIFT and A17 (ii) ANNEALING-DOMINATED) did not hold. (src: § A17 (i)/(ii) read)

(src: SAE_4A_lexlat_v2.md @7e7c38aee § A18 (a), (b), (c) reads; § Design review amendments, A18)

## Open after the final reads

- **No label-free phi lifts.** Every label-free phi read so far reads NO LIFT (A10 phis 0.820-0.842; wave em_s13 0.842; key arm rank1 0.843), while every label-built phi in the basin lifts (A17 (i) 0.200-0.362, A18 (b) 0.193-0.228). G4a.L2.4 read LOWER, OBJECTIVE ONLY on both bridges; CODE BROKEN was never reached.
- **Names.** Stage 1 and A20: J's optima are sharp partitions with wrong names, and J (trigram term) penalises gold-matching 1:1 names on them. Open: whether a near-gold key beats the found optima (no search started from gold, r30 or K30), and whether the deficit is in names or in the partition (A20 cannot separate them).
- **Objective inside the basin.** A17 (ii) and the keyinit read: S keeps falling while generative PER rises from gold (to about 0.30-0.35), so S's optimum near gold is not gold.
- **Unmeasured.** The share of the bridges' objective drop due to cold_ctl's uniform durations; seed spread for KEY BASIN (one seed per key) and for LABELS SUFFICE (r70-dur, one seed); the bit-exactness of two A18 (b) ep8 forwards.
- **Carried from the move, not addressed by these reads.** G4a.L2.3 (CANNOT_TELL; count-table repair not built); the untested mechanism "a 500-way per-frame emission term dwarfs the phone prior"; A15-E-style content in the A11 tables beyond the selected one; the handoff items needing new training (a label-free segmentation init, a duration-shape run, count-table repair) and the literature's list (more restarts and iterations; a coarse-to-fine unit inventory of about 100 classes; a word-level LM in the E-step with a wider beam; a sparse channel prior).
- **Named next by JUPITER's logs** (all unfunded, each training arm needing the user's OK):
  - `SAE_4A_rename.md` (user request 2026-09-24): what phi training needs so EM can correct wrong phone names. Analyses AN-0 to AN-5 and TP0 are read there (AN-5 and TP0: EM LOCKED). Its proposal (TP-D, then TP-B') waits for the user's OK and four rulings; AN-6 waits for the user's `-co`.
  - `SAE_4A_phiinit.md` (user request 2026-09-25; runs on i6, unfunded): a coarse-to-fine named alphabet with binary splits and a SIL anchor; oracle split arm first. It waits for the user's OK on round 1.

(src: SAE_4A_lexlat_v2.md @7e7c38aee § State; § Results; SAE.md @7e7c38aee; SAE_4A_rename.md and SAE_4A_phiinit.md @7e7c38aee § State)

**Values the ported package leaves open, against the README's "Open values"**
1. `WAVE_DURATION_SETTING` and 2. `WAVE_NUM_SUBEPOCHS` are open in the README. The A10 read decided them before the move: **durinit** and **12** sub-epochs. They should be set from that read, not re-derived.
3. phi_c's source (`k2lat_20_x60` at sub-epoch 60) has no preset. It is needed for the wave's 2 phi_c restarts and for phi_c's reference rows.
4. The L2-0 random-init phi rung is None in `ladder.ladder_phis`. In the source, cold_ctl had no standalone reverse checkpoint either.
5. These have no entry point in the port: the A11 count-table family, the A14-A19 arms, the A15 battery, the A16 key search and L2-2 (including dec_distil's tau 2.0 and lr 1e-5).

(src: SAE_4A_lexlat_v2.md § State; § Design review amendments, A14-A19; § Constraints)
