# SAE 4A -- lexlat v2: phi-first decipherment on the cold line

## State

Watcher: `bash ~/.claude/skills/sis/sis_watch.sh <pid> <config> 600`; re-arm first on resume. LIVE:
- The keyinit manager finished on 2026-09-25. The r30 genmarg decode was rerun after a node fault (`reports/debug_keyinit_genmarg_2026-09-25.md`).
- No live manager. Keyarms manager 3019650 exited normally on 2026-09-25 at 08:42 ("All output calculated"). The key-arm pack G0Vzzokj5PQC and the L2-2 bridge SCtv4DjzFQ50 are finished and read.
- Finished: em, keysearch_s1 (top 4 in `KeySearchSelectJob.g9wsznNnqmyO`), A20 (recorded).

Reads 2026-09-24 (Results, all audited): A10 EM phis decode at chance; A15/A15-F mislabelled and merged. A14 (ii) PHONETIC BASIN LOWER (3.216 against 3.299). Stage 0: J SEES THE KEY, fragile. G4a.L2.2 SIGNAL, NOT BEYOND PRIVATE CODE. A14 (i): EM PHI DOES NOT LIFT. A19 SAME LADDER. A17 (i) BASIN SUFFICIENT (0.20); A17 (ii) OBJECTIVE DRIFT. GOLD KEY REACHES BASIN (3.207); LABELS SUFFICE. Stage 1: finals beat gold on J through the emission term, wrong names (identity 0.07-0.14). A18 (a): LOWER, OBJECTIVE ONLY. A20: NAME-BLIND by rule; J penalises gold-matching names by 0.42-0.57 on the found partitions. A18 (b): DURINIT BASIN LIFTS, all 4 label-built arms at 0.19-0.23. Next-step proposals: `SAE_4A_rename.md`.

Rulings: pure unsupervised, GAN-free. Last training round: A14, A17, A16 (b), A18 joint runs on selected phis, A19 (Constraints).

NEXT:
1. Stage 2 recorded (Results): KEY BASIN on S only; the arms decode at chance. The A18 (c) bridge is recorded (Results): LOWER, OBJECTIVE ONLY, NO LIFT. Every registered run of this phase has now been read. The next training changes wait for the user's decision on the proposal in `SAE_4A_rename.md`.
2. The owed A15-F measures on the key arms ran as AN-5 (`SAE_4A_rename.md`).
3. At each watcher wake the executor checks; every read is audited, then recorded.

## Objective

Break the cold line's private code without labels and without a GAN. The record: a competent reverse model anchors a recognizer under the joint objective (`SAE_4A_lexlat.md` D10e, audited: gold phi HOLD 0.180 with k2; random-init phi destroys p0 in 57 steps); the cold phi is a contract on the recognizer's own decode (D10b: own decode preferred over gold by 800-1190 nats, gold vs deranged gold separated by 18-79); the objective prefers the phonetic solution (D13 logged read: gold-init lower on every term, l_tau 1.749 vs 1.873, k2 0.247 vs 0.306, agg 0.211 vs 1.264); the joint cold start lets theta fix a length-faithful code before phi carries structure. So the label-free problem is obtaining a competent phi with theta out of the loop. L2-1 fits phi by EM on the unit stream under the existing prior and lexicon, with likelihood-selected restarts (Berg-Kirkpatrick & Klein 2013; Klejch et al. 2022) and a prior-order curriculum; L2-2 brings the recognizer in afterwards. L2-0 measures beforehand how competent phi must be and which label-free statistic tracks that.

## Constraints

- Pure unsupervised, GAN-free (ruling 2026-09-23). Labels appear only in L2-0's ladder construction and in reports; no label enters a selection or a gate.
- Objective, prior `RtzbESkOedsT`, treatment graph `cdcxYJMjiYj5` at rung 1000, both model classes, tau = 2 for the joint runs, D10e pack constants: unchanged from `SAE_4A_lexlat.md`.
- No full joint cold restarts (every cold arm of the campaign sits in the chance band 0.83-0.91 at every seed); no recognizer-neutral (alpha = 0) arm from a cold start (it hands the posterior to an untrained phi, the collapse driver); no neural-refit split-merge search. Count-table repair moves on phi are the only split-merge form funded, and only by amendment on L2-1 SIGNAL + BELOW.
- Selection statistics are held-out and label-free; dev-other enters only reports.
- **Last training round here (user ruling 2026-09-24, quota; the project continues elsewhere through the porting work).**
  - Training is limited to the work already in progress:
    - the wave (4111121), A14 (i), and A17 (i)/(ii);
    - A17 (iii) with the gold-key control;
    - A16 (b) as registered: the stage-1 key search, the stage-2 key arms, and the conditional tau = 1 form of the gold-key control, each under its own gate.
  - The user clarified that "working on" includes work being implemented. The first reading, which dropped A16 (b) stages 1-2, is withdrawn.
  - Analysis on these runs is allowed without limit: forwards, readers, CPU analyses and audits, including new analyses registered here.
  - Only new training is excluded, such as a segmentation-init or duration-shape run. Such runs go to the continuation as handoff items.
  - Bridge and joint runs on the phis these lines select are permitted (user, 2026-09-24), withheld only when surely non-lifting: A18.
  - The trigram-only lift ladder, one four-GPU pack, is authorised by the user (2026-09-24): A19.

## Gates (pre-registered 2026-09-23, before any job)

| Gate | Clause | PASS | FAIL / other |
|---|---|---|---|
| G4a.L2.1 feasibility (L2-1 probe) | one restart, 1 pass, one GPU: step time, peak memory, the E-step at max_active 1000 under the null recognizer completes | a 4-pass restart projects to <= 1.5 h wall | otherwise the restart budget is re-derived from the measured rate and recorded before any wave launches |
| G4a.L2.2 signal (L2-1 wave) | selected stage-1 restart's held-out tau = 1 marginal likelihood minus the best null's | > null spread (max minus min over the 4 nulls) = SIGNAL | else NO SIGNAL: EM under this prior finds nothing beyond the unit statistics the shuffled corpus keeps; L2-2 not funded; the phi-first route closes at this scale |
| G4a.L2.3 competence (L2-1 stage 2) | selected phi's L2-0 statistic against the bar at rho* | at or beyond = CLEARS | BELOW; CANNOT_TELL if L2-0 found no separating statistic. SIGNAL + BELOW funds the count-table repair amendment before L2-2 |
| G4a.L2.4 bridge (L2-2) | ep8 per-frame dev total (l_tau + lexlat_k2 + agg + 3 rate, CV holdout) paired per utterance against the cold baseline at ep8, speaker-clustered bootstrap; identity band B = abs(dec_joint - dec_joint_s2) | LOWER: interval below zero and abs(delta) > max(B, 0.01 nats/frame) | HIGHER symmetric; TIE otherwise. Reported beside, never gating: dev-other greedy PER at ep1/2/4/8 against the chance band and D10e's bands. "Code broken" needs LOWER and PER below the chance band; LOWER with PER in the band is recorded as the reviewer's warning (lower objective, no phonetic content) |

The original clauses above stay as registered text; amendments A3 (G4a.L2.1), A7 (G4a.L2.2), A5 (G4a.L2.3) and A6 (G4a.L2.4) below supersede them.

## Design review amendments (2026-09-23, before any job)

Source: `reports/design_review_lexlat_v2_2026-09-23.md` (B1-B6, N1-N7). Orchestrator decisions; nothing changes the objective, lexicon, LM or model classes.

- **A1 (B1) Stage 2 of L2-1 removed.** The k2 term reads only the recognizer's emissions (`lexlat_k2_train.py`), so under a null recognizer it is a constant that neither moves phi nor re-ranks restarts. The word graph enters at L2-2. G4a.L2.3 reads the stage-1 selected phi. "Pruning sees prior and phi scores only" holds for the lattice term alone.
- **A2 (B2) Rate anchor.** Rate and agg reach log_q only, so nothing prices phi's token count. Every restart logs held-out expected non-SIL rate. A restart whose final held-out rate lies outside the bed's [5.80, 14.49] Hz is VOID for selection. If either probe restart ends outside the band, the wave freezes phi's duration logits at a rate-matched init (mean 50 / 9.66 frames per token). The rate term's tilted DP passes are skipped under a null theta.
- **A3 (B6, N4) EM budget and G4a.L2.1.** phi: Adam, constant lr 3e-3, clip 5 (the supervised reverse fit, the only phi-alone precedent). Data: the bed's train stream in its own sub-epochs, so 4 sub-epochs = one pass over train-clean-100 (replaces one quarter repeated). tau per sub-epoch [4, 1, 1, 1]. Probe = seeds 1 and 2, 4 sub-epochs each, every sub-epoch checkpoint scored (held-out tau = 1 marginal, expected rate). PASS = a restart projects to <= 1.5 h AND the held-out gain from sub-epoch 3 to 4 is < 0.01 nats per frame for both seeds AND both rates in band. Gain >= 0.01: the probe restarts continue and the wave's sub-epoch count is re-derived, recorded before the wave. Rate out of band: A2's freeze.
- **A4 (B3) L2-0 at the bridge's condition.** A p0 HOLD measures preservation; L2-2 needs a phi that lifts a random theta. Primary ladder: theta at the cold random init, D10e pack constants. Node R1: rt_r0 (gold phi), rt_r30, rt_r50, rt_r100. Node R2: cold_ctl (random phi), rt_perm (permphi), rt_r70, rt_r0_s2. LIFT = dev-other greedy PER < 0.50 at ep8, PARTIAL < 0.8164 (0.83 minus G4a.9's 0.0136), else NO LIFT. rho*_lift = the largest rho that LIFTs with every smaller rung lifting; non-monotone = CANNOT_TELL. Both rt_r0 seeds NO LIFT = L2-2 not funded (a gold phi does not lift a random theta here). The p0 ladder (node P, built) runs as registered for the anchor question only. Read 2's "`_r100` HOLD, a fitted phi suffices" is withdrawn: a content-free phi holds p0 by exerting no pull.
- **A5 (B4, N6) Statistics and G4a.L2.3.** Statistic (b) rewards any sharp code (the cold private code reads 3.04-3.61 nats per frame against gold's 3.9-4.4). Two sharp-wrong references join the set: phi_c (the reverse block of `k2lat_20_x60` at ep60) and permphi (the gold-phi recipe on gold strings under one fixed phone permutation, seed 0). The bar is the statistic among (a), (b) that is monotone in rho on the random-theta ladder, separates every LIFT arm from every NO LIFT arm, places phi_c on the non-lifting side and permphi on the side its own arm reads. Else G4a.L2.3 = CANNOT_TELL. Sets: all 285 CV-holdout utterances, the 500-utterance D4 dev-other set. (c) runs on dev-other only (no banked p0 decode on the CV holdout).
- **A6 (B5) G4a.L2.4.** Pack C has no ep8 and differs in rung, on-set and tau. Baseline = cold_ctl at L2-2 constants (only phi's init differs from dec_joint; L2-0's cold_ctl is reused if its config is identical). Paired per utterance: l_tau + lexlat_k2 + 3 rate via the D16 eval path at one setting on the CV holdout; agg as a point contrast. CODE BROKEN = LOWER AND dev-other greedy PER < 0.50 at a kept epoch; LOWER alone = OBJECTIVE ONLY, no claim. Constraints now read "no label enters selection" (plain PER may gate, as G4a.9). "No full joint cold restarts" exempts cold_ctl, a control.
- **A7 (N1, N2) G4a.L2.2 is a spend gate.** Any segmental fit beats the within-utterance shuffle, so NO SIGNAL means L2-2 is not funded, not that the route fails. Margin = max(null range, identity band, 0.01 nats per frame); identity band = the larger |S| difference of seeds 1 and 2 against their exact reruns. Nulls are gated on their permuted holdout (E4 convention), real holdout reported. Wave = 16 restarts + 4 nulls + 2 phi_c-initialised restarts + reruns of seeds 1 and 2 (6 nodes). Report only: selected restart minus best phi_c restart beyond the margin = BEYOND PRIVATE CODE, else NOT BEYOND. L2-2 is funded on SIGNAL AND at least one rt_r0 seed reading LIFT or PARTIAL (ruling after code review: PARTIAL means the gold phi moves a random theta out of the chance band; A4's "both NO LIFT = not funded" is the complement). R1 / R2 wait for a one-GPU k2 pre-flight at rt_r0's config: with the flat theta and on-set 1, the first k2 call runs on a uniform posterior (D9: about 500x a trained lattice; 5o9P5uidnoWv died there on a k2 int32 error) and packs cannot resume. Pre-flight (`LadderK2PreflightJob.ZM3MD9viV7sM`, rt_r0's written config, stability read + about 100 timed steps) PASS = the stability read and every step complete with no k2 overflow or OOM, peak GPU memory <= 80 GiB (E1's bar), and 8 sub-epochs at the measured step rate project inside the pack's TIME_RQMT with no resume; else R1 / R2 are held and the cause diagnosed. Every restart VOID on rate = NO ELIGIBLE RESTART: G4a.L2.2 reads CANNOT_TELL, and the wave reruns once under A2's duration freeze; if the freeze was already applied, NO SIGNAL. Rate is pooled per set as 50 x (sum of expected non-SIL tokens) / (sum of unit frames). The random init sits at 3.7 Hz (uniform durations), below the band.
- **A8 (2026-09-23, after implementation, before any L2-1 job) Rate reads and seeds.** Supersedes A2's "expected rate" and A7's pooling sentence, which disagreed with each other and with the band's own definition: [5.80, 14.49] Hz is a greedy emitted rate per original second, "never a posterior expectation" (`SAE_4A_attrib.md` S3b-R; c5 met an expected count with a diffuse posterior). Under the null recognizer the emitted sequence is phi's decode of the tau = 1 generative posterior (the genmarg decode). Rate = 50 x (sum of non-SIL tokens in that decode) / (sum of original 50 Hz frames), pooled over the CV holdout; A2's VOID, A3's probe PASS and NO ELIGIBLE RESTART read it, so the probe's scoring reads decode. The expected rate (both denominators) and the training monitor `blankfree_nonsil_rate_retained_hz` (retained frames, the sub-epoch's tau) are reported only; the monitor's 14.5 Hz at tau 8 and genmarg's 3.7 Hz (expected, random phi, tau 1, original frames) are different quantities. A2's freeze mean is converted into retained unit frames: (50 / 9.66) x (sum retained / sum original frames) on the train stream. Every L2-1 run's seed also sets the training data order (reruns keep their original's seed): phi has no dropout and phi_c a fixed init, so phic_s01 / s02 were one run; restarts become independent draws of init and order, and the identity band still measures nondeterminism.
- **A9 (user request 2026-09-23, before any L2-1 job) General-knowledge duration prior.** General knowledge of phone length may initialise or restrict phi; a duration model learned from supervised data may not (user clarification). Prior and modes `durinit` / `durfrz` as `SAE_4A_lexlat.md` D17: one rate-matched maximum-entropy law for every phone type, from the label-free rho; SIL uniform and trainable. Probe = 3 duration settings x seeds 1, 2: uniform (as registered), durinit, durfrz, each read by A3's PASS with A8's rate. The wave uses the prior setting that passes PASS with the higher mean held-out tau = 1 marginal at sub-epoch 4 (label-free); if neither passes, the wave waits for a re-derivation under A3, recorded before it. Uniform is reported only (does the prior matter under phi-first EM). Nulls use the wave's setting; phi_c arms keep phi_c's own durations. durfrz replaces A2's max-entropy freeze. Report only: E[d] per type and the emitted rate per probe sub-epoch. L2-0 is unchanged (its fitted phis carry fitted durations); L2-2's cold_ctl baseline takes the wave's duration setting, so its single delta stays phi's emissions.
- **N7.** dec_distil's lr and steps are stated, with a source, before L2-2 is built.
- **A10 (2026-09-23, after the probe read, before any wave job) The re-derivation that A3 and A9 prescribe.** The probe read NO WAVE SETTING (Results): every restart passes time and rate, and fails only the gain clause, with gains of 0.28-0.33 nats per frame from sub-epoch 3 to 4. No gate threshold changes.
  - Extension: the six probe restarts rerun from scratch with the same seeds and settings to 48 sub-epochs (12 passes, about 3 h each; tau 4 at sub-epoch 1, then 1). Every sub-epoch checkpoint is scored as in the probe. Sub-epochs 1-4 against the probe are an identity-band sample, reported.
  - K*(setting) = the first sub-epoch k >= 4 at which the held-out gain S(k-1) - S(k) < 0.01 for both seeds, with both emitted rates in band.
  - Wave setting and length: among the settings with a K*, the lowest mean S over seeds at the largest K* among them. The wave's sub-epoch count is that setting's K*, rounded up to a whole pass (a multiple of 4).
    - If no setting reaches K* by 48: the setting with the lowest mean S at 48 among those in band, run for 48 sub-epochs, and the wave is reported as NOT CONVERGED.
    - Nulls and the phi_c arms run the same count.
    - Uniform is still reported only and never chosen (A9).
    - The reader's readings, recorded after code review (`reports/review_l21_a10_2026-09-23.md`; none changes a verdict A10 decides):
      - uniform stays out of "the largest K* among them";
      - a clause unreadable before K* reads CANNOT_TELL;
      - no setting in band at 48 reads NO WAVE SETTING;
      - an exact tie reads TIE.
  - Report only, never a selection or a gate: every 4 sub-epochs, phi's genmarg decode of the 500-utterance D4 dev-other set is read for direct PER against gold (the symbols are the prior's phones), Hungarian PER, NMI(symbol, phone) and E[d]. The same held-out S is also computed for the gold phi and L2-0's `_r100` phi, as reference points on the label-free scale. These say whether EM is finding phonetic structure, and they are what the adjustment decisions after the wave will read.
- **A11 (user ruling 2026-09-23 night, before any A11 job) Exact-EM count-table phi, a second L2-1 family in parallel with A10.** A10 stays as registered.
  - Why: phi's emission head sees only (type, duration bucket, position bucket, eta) (`reverse.py` SegmentalReverseModel), so without eta its class is a 240 x 500 emission table plus the duration table. Under A3, a pass fits it with 228 Adam steps on the marginal, each a partial M-step; the probe's 0.28-0.33 nats per frame gains at sub-epoch 4 fit that. The decipherment recipe L2-1 cites (Berg-Kirkpatrick & Klein 2013; Klejch et al. 2022) fits the table by closed-form EM. EM reaches a fixed point in tens of iterations and makes many restarts affordable.
  - Model: a free emission table (type, duration bucket, position bucket) -> 500 units, no eta; the duration table per type on [d_min, D_k]; A9's durinit and durfrz (uniform not run). SegmentalReverseModel's topology (d_min 2, D 25, D_sil 50).
  - E-step: L2-1's posterior (null recognizer, stage-1 frozen trigram, max_active 1000, the same lattice code). Expected counts of (type, bucket, position, unit) and (type, duration) from its forward-backward; counts asserted finite and summing to the frame and token totals. Deterministic annealing: tau 4 -> 1 linearly over iterations 1-4, then 1.
  - M-step: normalised counts plus a 1e-3 pseudo-count per cell, a positivity floor only. The job prints the smallest expected row mass, so that the floor's share can be checked.
  - Stage A (screen): 32 restarts per duration setting, Dirichlet(1) emission tables (seeds 1-32), 10 iterations on a fixed 1,000-utterance subset of the train stream. Selected by S (A3's held-out tau = 1 NLL per frame, the 285-utterance CV holdout).
  - Stage B: the top 4 per setting continue on the whole train stream (one iteration = one pass over train-clean-100) until S(i-1) - S(i) < 0.01, at most 40 iterations. Selected restart: the lowest S among both settings' finishers.
  - Budget: the first stage-A restart's measured iteration time is recorded. If stage A projects above 12 GPU-h per corpus, the restart count halves, recorded before launch.
  - Nulls: the identical pipeline (stage A seeds 1-32, stage B top 4, both settings) on the structure-destroyed corpus (units permuted within each utterance). Nulls are gated on their permuted holdout (A7). Null spread = max - min of S over the null stage-B finishers.
  - Identity band: stage-B restarts 1 and 2 of the real pipeline rerun exactly (A7's definition).
  - Gate: G4a.L2.2 by A7's clause, per family. SIGNAL if the selected S minus the best null stage-B S is below minus max(null spread, identity band, 0.01). A10 and A11 are read against their own nulls and both reads are reported; either family's SIGNAL funds the next step at the same bar.
  - Report only, never a selection or a gate: for every stage-B finisher at iterations 10, 20 and 40 and at its end, the D4 dev-other genmarg decode, with direct PER, Hungarian PER, NMI(symbol, phone), E[d] and A8's emitted rate.
  - Bridge (on SIGNAL): the selected table is distilled into SegmentalReverseModel's head, fit to the table's categoricals with eta from the data (no labels). Its S is re-read, and it enters L2-2 as phi.
  - Efficiency: one restart per GPU, four per node.
- **A12 (2026-09-23 night, after the literature read, before any A11 job) The A11 recipe brought in line with published EM decipherment.** Source: `reports/lit_em_decipherment_2026-09-23.md`. No published EM decipherment works on 50 Hz units, so the literature cannot say whether A11 will work. The working recipes do share three elements that A11 lacked:
  - restarts run to near-convergence before they are ranked (200 iterations in Berg-Kirkpatrick & Klein 2013; 20 per stage in Klejch et al. 2022);
  - the E-step model is smoothed toward uniform when the lattice is pruned (Nuhn & Ney 2014: pruning zeros counts that never recover);
  - any annealing holds each temperature for several steps, and the evidence that it helps is weak (Smith & Eisner 2004; Johnson 2007).
  Frequency-rank initialisation hurts at this homophony (Kambhatla et al. 2018), so the init stays Dirichlet(1). These changes supersede the corresponding A11 items. The gate, selection statistic and nulls of A11 are unchanged.
  - Stage A:
    - a fixed, seeded 300-utterance subset of the train stream;
    - iterations run until the subset's training log-likelihood gain is < 1e-3 nats per frame, at most 100;
    - ranking by held-out S at the end;
    - S is also logged at iterations 10, 20 and 40, and the rank agreement between iteration 10 and the end is reported.
  - Arms (32 restarts each, Dirichlet seeds 1-32 shared across arms): (a) durfrz, tau 1 throughout; (b) durfrz, slow anneal (tau 4, 2, 1.5, each held 5 iterations, then 1); (c) durinit, tau 1 throughout. A11's 4-step anneal is dropped.
  - E-step smoothing: the emission model used in every E-step and in S is 0.9 x table + 0.1 x uniform over the 500 units (Nuhn & Ney 2014; Klejch et al. 2022). The M-step keeps the 1e-3 pseudo-count. Durations are not interpolated.
  - Stage B: the top 4 per arm continue on L2-1's fixed 7.1k-utterance sub-epoch until S(i-1) - S(i) < 0.01, at most 40 iterations; the literature puts data length well below the bottleneck. Selected restart: the lowest S among all arms' finishers.
  - Frame-permuted nulls (the gate): the identical pipeline, all three arms.
  - Report only, never a selection or a gate:
    - A run-preserving null: whole runs of identical units shuffled within each utterance, 8 seeds per arm, the top 2 per arm to stage B; reported as the selected S minus its best S. Frame shuffling also destroys unit runs, so a content-free duration model can beat the gate's null. This null keeps the runs and removes only the order.
    - For every finisher: the phone-trigram NLL per token of its genmarg decode, and the number of phone types used.
  - Budget: the measured stage-A time is recorded. If stage A projects above 24 GPU-h per corpus, the restart count halves, recorded before launch.
    - Reading at build (before any A11 job): the E-step measures 33.6 ms per utterance. At the 100-iteration cap, 32 restarts per arm project to 27.45 GPU-h per corpus. That is above 24, so the count halves to 16 restarts per arm (seeds 1-16, shared across arms), about 13.7 GPU-h. Source: `reports/impl_l21_a11_2026-09-23.md`.
  - Rate VOID (a clarification at build, before any A11 job): A7's clause carries A2/A8's VOID, so a finisher whose A8 emitted rate falls outside [5.80, 14.49] Hz cannot be selected. This holds in the real pipeline and in the nulls, including the set that defines the null spread. If no finisher is eligible, G4a.L2.2 reads CANNOT_TELL for the family. A7's duration-freeze rerun does not apply, because arms (a) and (b) already freeze durations. Stage A ranks by S only.
  - Planned next, if the families read NO SIGNAL or report no phonetic content, in the literature's order:
    1. more restarts and iterations;
    2. a coarse-to-fine unit inventory (the 500 codebook vectors clustered into about 100 classes, label-free; its own nulls; bridged by P(unit | class));
    3. a word-level LM in the E-step with a wider beam (Ravi & Knight 2009: phonetic decipherment error 73.6 with a trigram against 57.2 with a word LM);
    4. a sparse channel prior.
- **A13 (2026-09-23 night, at the ladder launch review, before any ladder statistic is read) CV-holdout overlap and diffuse-arm memory.** Source: `reports/review_l20_ladder_launch_2026-09-23.md`.
  - Overlap:
    - 25 of the 285 CV-holdout utterances (8.8 %) are in every ladder phi's fit set: gold `16v7R6ztSq1u`, r30-r100 and permphi, all fitted on the 2821-utterance split.
    - phi_c and every L2-1 phi exclude the CV holdout.
    - So A5's CV statistics, and the gold and r100 S references of A10 and A11, include memorised items for the ladder phis only.
  - Repair: every CV-holdout statistic that compares a ladder phi with a non-ladder phi is read on the 260 utterances disjoint from the ladder fit set. That covers A5's bar and the S references. The 285-utterance value is reported beside it.
  - dev-other is unaffected, and so is L2-1's own selection. Its S compares L2-1 phis only, which all exclude the holdout.
  - Memory: the pre-flight's 76.7 GiB peak was set at steps 0-1, on the two shortest batches, while rt_r0's lattice collapsed (37,210 -> 1,007 arcs per frame by step 7). A near-uniform lattice on a full-length batch was never measured, and chunking does not bound held memory.
  - Exposed arms: cold_ctl, rt_r100 and possibly rt_r70.
  - Decision: all three packs launch now, since an OOM would stop only that arm. The executor reads those arms' arcs per frame and peak memory through step 12.
  - An arm that OOMs is rerun alone after an implementation-only fix: per-chunk backward with gradient accumulation, which bounds held memory. The fix is reviewed, and no registered constant changes.
  - Watch result: every arm passed step 12 by 22:23, including the longest batches at steps 7-8 (maxlen 784-803). No arm OOMed. Peak reserved memory, of 95 GB:
    - cold_ctl 83.7 GB, at step 3;
    - rt_r100 81.7 GB;
    - rt_r70 80.3 GB;
    - other arms at most 80.9 GB.
  - Peaks were set at steps 1-2 and stay flat after them. Arcs per frame at step 12 are 1,403 (cold_ctl), 1,434 (rt_r100) and 526 (rt_r70), against 37,200 at step 0.
  - The pre-flight's 80 GiB clause was a pre-flight bar, not an abort rule. The running peaks exceed it by up to 4 GB and stay under the card.
- **A14 (2026-09-24, after the A10 and L2-0 reads, before any A14 job) Does an EM phi lift a random theta, and does the stage-1 objective prefer phonetic phis?** Trigger: every A10 restart beats the gold phi on held-out S (3.30-3.40 against 3.47 on the 260 set), while its genmarg decode stays in the chance band (Results). L2-0's rt arms show that a fitted phi lifts a random theta at this recipe. No registered gate or constant changes, and the wave runs as registered.
  - **(i) rt_em, pure unsupervised: the direct lift test.**
    - Recipe: the L2-0 R-node recipe verbatim, with theta at the cold random init, D10e pack constants, `supphi_k2lat`'s k2 block, 8 sub-epochs, and both models trainable. phi is initialised from an A10 restart's sub-epoch-48 checkpoint.
    - Arms: all four candidate restarts, durinit s1/s2 and durfrz s1/s2, so no selection is made. They run as one 4-arm pack.
    - Read: dev-other greedy PER at ep8 in A4's bands (LIFT < 0.50, PARTIAL < 0.8164, else NO LIFT).
    - Verdict: **EM PHI LIFTS** if any arm reads LIFT or PARTIAL; **EM PHI DOES NOT LIFT** if all four read NO LIFT. Reported beside: ep1/2/4/8 PER, and a paired row rt_em minus cold_ctl at ep8.
    - PER is a read here and never selects a phi. The L2-2 funding rule (A7) is unchanged.
  - **(ii) Objective floor, analysis only (supervised init, disclosed; never a route or a fallback).**
    - Recipe: the A10 recipe verbatim (durinit, tau 4 then 1, 48 sub-epochs, the same sub-epoch and scoring), with phi initialised from the L2-0 fits: gold `16v7R6ztSq1u`, r30, r70 and r100. These are 4 one-GPU restarts. Diagnostics run as in A10, every 4 sub-epochs and also at sub-epoch 0, the init phi itself.
    - Read: S_g = the gold-init restart's S at 48 on the 260 set, against S_min = 3.2990, the lowest of the six A10 restarts at 48.
      - **PHONETIC BASIN LOWER** if S_g < S_min - 0.01 and the gold-init Hungarian PER at 48 is < 0.50. The objective prefers the phonetic basin, and random-init EM is search-limited.
      - **NON-PHONETIC PREFERRED** if S_g > S_min + 0.01, or the gold-init Hungarian PER at 48 is >= 0.83 (drift into the chance band). Here the stage-1 objective, not the search, stops EM, and the next cost work goes to the objective.
      - **TIE** otherwise.
    - The 0.01 is A7's floor. The r30, r70 and r100 inits are reported only.
  - Cost: 8 GPUs on two nodes, about 2.2 h for (i) and about 3 h for (ii). (i) runs as a 4-GPU pack, and (ii) runs through gpupack.
- **A15 (user request 2026-09-24, "results analysis of L2-1 result vs gold vs gold 70 noise, how do they perform wrt text preference (gold vs swap) and raw accuracy"; registered before any job; disclosed label-using analysis, descriptive, no gate) Phi competence battery.** It compares the reverse models directly, apart from any recognizer.
  - **Phis:**
    - the six A10 restarts at sub-epoch 48, plus durinit s1 and durfrz s1 at sub-epochs 4 and 12, which gives a trajectory;
    - the L2-0 fits: gold `16v7R6ztSq1u`, r30, r50, r70, r100 and permphi;
    - phi_c, D14's decphi, and one random-init phi as the floor;
    - the reverse blocks of rt_r0 and rt_r70 at ep8, i.e. phi after joint training with a lifted theta.
  - **Set:** the 500-utterance D4 dev-other set, which is disjoint from every phi's fit.
  - **Text preference.** Each is the per-frame log p_phi(z | string, eta), marginalised over segmentations: gold minus the alternative. Speaker-clustered 95 % intervals.
    - (T1) Utterance swap: gold against a same-speaker deranged gold string (D10b's form).
    - (T2) Phone-identity swap: gold against gold with phone labels permuted. Three levels: 1 random pair, 5 random pairs, and a full random permutation of the 39 non-SIL phones. 5 permutation seeds each, reported as a mean and a range.
    - (T3) Relabelled gold: the phi's Hungarian symbol-to-phone map (from R1's decode) is applied to gold, and the result is set against T1's deranged string. This asks whether the phi is phonetic up to relabelling. For permphi, its own permutation is also scored. Amended 2026-09-24, before any result, because that contrast mixes the relabelling with the utterance swap. The primary T3 is now M(gold) against M(deranged), the same map applied to both strings, so it isolates content. Also reported: M(gold) against gold, which reads the relabelling alone, and the original mixed contrast.
  - **Raw accuracy.**
    - (R1) The genmarg decode under the phone trigram, as in A10's diagnostics: direct PER, Hungarian PER, NMI(symbol, phone), E[d].
    - (R2) The same decode under a uniform phone prior, which is phi's content without the text prior's help.
    - (R3) Emissions: per phone, the JS divergence between phi's mean unit distribution and the gold phi's (marginalised over position, duration and eta on the set), both directly and after the Hungarian match.
    - (R4) Frame-level unit-to-phone accuracy against the MFA dev-other alignment, if one exists, with the majority-unit oracle as the ceiling. If no alignment exists, this is reported as not built.
  - Reporting rules sit in the job docstring. The battery explains the lift and non-lift reads (L2-0, A14 (i)) and says what EM is missing. It never selects or gates.
  - **A15-E, emission-matched map** (added 2026-09-24, before any battery result).
    - Why: the build test found that R1's decode-based Hungarian map recovers only 16 of permphi's 40 true labels. permphi's Hungarian PER reads 0.819, inside the chance band, although its emissions are gold's under a fixed permutation. Matched R4 through that map reads 0.18, against 0.59 through the true permutation. So a decode-based map can miss a phi that is phonetic up to relabelling.
    - The emission map: a one-to-one Hungarian assignment of the phi's 40 symbols to gold-phi phones, on the per-phone JS cost between R3's mean unit distributions. Also reported, per symbol, the many-to-one nearest gold phone with its JS, and how many gold phones are claimed by no symbol.
    - Through the emission map: T3 (primary, relabel), R3 matched and R4 matched.
    - Positive control: the emission map recovers permphi's true permutation (count of correct labels). The gold phi maps to the identity.
    - Descriptive, no gate, never selects. Rules go in the job docstring.
  - **A15-F, content resolution and a sharp null** (added 2026-09-24, after the A15-E audit, before any A15-F job; source `reports/audit_a15_read_2026-09-24.md`).
    - Why: the audit found that R4 emis (0.29-0.31) equals a 7-manner-class oracle partition (0.287), that r100 is no null for a sharp phi's T3, and that the relabelled gold fit is worse than r100's. Whether the EM content is phone-level or broad-class decides whether a label search (A16 (b)) can be enough.
    - Inputs: the six A10 sub-epoch-48 phis, the two trajectory phis, phi_c, gold, r70, r100 and permphi, on A15's D4 set and eta. It uses a new module and leaves the A15-E code untouched.
    - (i) Save each phi's per-symbol mean unit distribution m_phi (40 x 500), computed the way A15-E does.
    - (ii) Split R4 emis by class. Report the share of frames whose matched phone is in the gold phone's manner class (7 fixed classes: vowel, stop, fricative/affricate, nasal, liquid, glide, SIL), and accuracy within class given the class is right. Also report per-phone R4 emis.
    - (iii) Oracle partition bounds, as a registered reader:
      - Random 40-way unit partitions, 200 draws, mean and max.
      - Units assigned to their majority MFA class under 7 manner classes and under 13 place-and-voicing classes.
      - In both, groups map to phones one-to-one, with the map chosen from the MFA labels.
    - (iv) A sharp destroyed-structure null. Each A10 sub-epoch-48 phi gets its unit axis randomly permuted, 5 seeds, keeping its sharpness and symbol group sizes. It runs through the same emission map procedure, with T3 (primary, relabel), R4 emis and the relabelled gold fit.
    - (v) The relabelled gold-string fit per gold phone, split by claimed and unclaimed phones (nearest-phone map).
    - (vi) If the A11 selected table (real_c_s16) yields a per-symbol mean unit distribution (bucket-averaged under its own duration table), it gets the A15-E map and R4 emis. Otherwise it is reported as not built.
    - Reading rules (descriptive, never a gate or a selection; they go in the job docstring):
      - A phi's content counts as ABOVE MANNER LEVEL if its within-class accuracy exceeds the within-class accuracy of the 7-class oracle.
      - An EM T3 or R4 gain counts only if it exceeds the maximum over its 5 permuted-unit nulls.
    - Amendment (2026-09-24, at build, before any A15-F job; source `reports/impl_a15f_content_2026-09-24.md`):
      - The build found the 7-class oracle's within-class accuracy (0.374) below the random 40-way partitions' maximum (0.429). The sharp permuted-unit nulls reach 0.34-0.40. Within-class accuracy conditions on class-correct frames, so it is inflated for weak partitions. The rule above could therefore fire on a null.
      - Replacement: ABOVE MANNER LEVEL only if the phi's within-class accuracy exceeds the maximum of three references: the 7-class oracle's, the random-partition maximum, and the maximum over the phi's own 5 permuted-unit nulls. The class-correct share and R4 emis are reported next to it, against the same three references.
      - Class assignment: the primary oracle assigns each unit the class of its majority MFA phone, which reproduces the audit's 0.287 and 0.377. The per-unit majority-class variant (0.279 and 0.367) is printed as VARIANT.
- **A16 (2026-09-24, after the A15/A15-E read, before any A16 job) Is the labelling the missing piece, and can it be found without labels?** The A15-E read (Results) finds the A10 EM phis phonetic at about r70's emission-matched frame accuracy, but mislabelled and merged. An L2-1 extension under the user's latitude ruling.
  - Premise corrected by the audit (2026-09-24, before any A16 result): even relabelled, the EM phis sit below r70, at about the manner-class level, and fit gold worse than r100 (Results, A15 read). A16 (a) still reads as registered. Its dS asks whether S prefers the emission-map labelling, which stays informative. A16 (b)'s design waits on A15-F as well.
  - The emission map is chosen against the transcript-fitted gold phi. So any use of it, A16 (a) included, stays analysis-only and never relabels a phi that trains or is selected.
  - **(a) Does the label-free objective prefer the right labelling?** Analysis only: gold enters through the emission map, never a selection.
    - Held-out S (tau = 1, A13's 260-utterance set, the A10 S reader path) for each of the six A10 sub-epoch-48 phis, under two labellings: identity, and relabelled through its A15-E emission map (the phi's symbol axis permuted so that symbol h^-1(k) is read as phone k).
    - Positive control: permphi under identity and under its true inverse permutation. Its identity S is 4.075, and gold's is 3.474.
    - Read per phi, as dS = S(emis) - S(identity), with a speaker-clustered interval:
      - **OBJECTIVE PREFERS THE RIGHT LABELLING** if dS < -0.01 for at least 4 of 6. EM stops at a labelling local optimum, so a discrete label search on S is the extension, (b).
      - **OBJECTIVE LABEL-BLIND OR WRONG** if dS > -0.01 for at least 4 of 6. The label signal must come from the prior term, and the next cost work goes there.
      - Otherwise **MIXED**.
    - The positive control must read dS < -0.3; otherwise the read is VOID.
  - **(a2) Model error: does a weighted prior make the objective rank gold top?** (Added 2026-09-24, after the literature read, before any A16 result.) Analysis only, like (a).
    - Why: the closest published analogue is Yin et al. 2019 (https://arxiv.org/pdf/1810.04297v3), a joint LM-GMM over unsupervised glyph clusters. It found "model error": the gold model does not get the highest score, because the strong emission term dominates the LM, and the fix was P(E)^3. That matches A10, where the EM phis beat gold on S (3.30-3.40 against 3.474).
    - Statistic: S_lambda = -(1/T) log sum_y P_LM(y)^lambda P_phi(x | y), tau = 1, on (a)'s 260 set, for lambda in {1, 2, 3}. lambda = 1 is S and reuses the banked values.
    - Phis:
      - gold, r30, r70 and r100;
      - the six A10 sub-epoch-48 phis under identity and under the emission relabelling;
      - permphi under identity and under its true inverse;
      - phi_c.
    - Reading (descriptive, label-using, never a gate or a selection):
      - MODEL ERROR REMOVED AT lambda: S_lambda(gold) is below min over the six EM phis (identity) of S_lambda by more than 0.01, AND S_lambda is monotone on the ladder (gold < r30 < r70 < r100). Report the smallest such lambda, or NO LAMBDA <= 3.
      - Also report (a)'s dS at each lambda.
    - lambda is never chosen from this read, because choosing it by gold's rank would be label-based selection. If (b) or a restart uses a weighted prior, it takes the literature value lambda = 3 fixed in advance. This read only says whether that value removes the model error seen at lambda = 1.
  - **Literature read** (`reports/lit_decipherment_relabel_2026-09-24.md`, 2026-09-24). These points shape (b):
    - Yin et al. 2019 separate model error from search error. The joint EM missed the gold score after 5,000 random restarts, and restarting it from a pipeline decipherment fixed that (Borg 0.35 -> 0.20 NED; 0.16 with a trigram, below the 0.22 optimal-mapping ceiling).
    - Moves over whole symbol types beat point-wise EM on homophonic ciphers: Ravi & Knight 2011 get 95.2 % against 32.6 % for 3-gram EM (https://aclanthology.org/P11-1025.pdf). Nuhn, Schamper & Ney 2014 run a count-based beam search (https://aclanthology.org/D14-1184.pdf).
    - An objective that scores only the LM of the outputs collapses to the majority guess (Liu, Chen & Deng 2017). So any many-to-one or unit-level key keeps the channel term, as S does.
    - Merges cannot be repaired by relabelling states, which is capped at the optimal-relabelling accuracy. Going further needs unit-type moves (reassign a unit type, or split and merge), each accepted only if S improves.
    - No published result is at our noise level.
  - **(b) An unsupervised label search**, designed after (a) and a literature read on substitution-cipher decipherment (homophonic, noisy, under an n-gram LM). It is registered as its own amendment before any job. It must be label-free and GAN-free, and select only on held-out S.
  - **(b) registered (2026-09-24, after the A14 (ii) read and its audit, before any A16 (b) job; an L2-1 extension under the 2026-09-23 latitude; pure unsupervised, GAN-free): unit-type key search, then S-EM from the key (pipeline decipherment).**
    - Premise:
      - A14 (ii) (audited) found that at matched data and criterion, S's lower basin is phonetic. At sub-epoch 48, S is gold-init 3.216, r30 3.210 and r70 3.271, against random-init 3.30-3.40.
      - The r70 init reaches that basin (PER 0.61 -> 0.49); r100 does not (0.83 -> 0.86, S 3.378). So random-init EM is search-limited.
      - r70's corruption is random, so its per-unit argmax phone stays right (L2-0 N1). The missing piece is therefore a label-free unit-to-symbol key.
      - The literature's fix for this failure is a discrete key search on count tables, followed by the joint EM from the key (literature bullet above).
      - Not licensed by A14 (ii), and so tested here: that search can reach the basin from no labels. Also untested: whether the gold arm's supervised durations carried part of the gap. The audit found that the A14 (ii) gold init replaced durinit.
    - **Key objective J(k)**, CPU, count tables.
      - A key k maps each of the 500 units to one of the 40 symbols. The unit sequences of the A10 train stream are mapped through k, and runs are collapsed into segments.
      - J(k), per frame, sums:
        - log P(collapsed symbol string) under the same trigram `RtzbESkOedsT`;
        - the maximum-likelihood emission log-likelihood of each unit given its symbol, with counts from the train side;
        - the general-knowledge duration log-prior of the segment lengths (A9; d_min >= 2 is standing).
      - A key whose emitted rate lies outside [5.80, 14.49] Hz is void (A2).
      - J is read held out, on the CV holdout of the train stream.
    - **Stage 0, J's floor (CPU; disclosed label-using analysis; gates stage 1).** Keys:
      - the gold key: each unit's majority phone on the train-side alignment that the L2-0 fits used, never D4;
      - a key ladder: the gold key with 30, 70 and 100 % of units reassigned uniformly at random, 5 seeds each;
      - the argmax keys of the six A10 sub-epoch-48 phis and of A11's selected table phi;
      - 20 random keys.
      - Every key is also scored on the structure-destroyed corpus (units permuted within each utterance), and the rise of each over its destroyed score is reported.
      - **J SEES THE KEY** if held-out J is strictly monotone on the ladder (gold > K30 > K70 > K100, seed means), AND J(gold) exceeds the best A10/A11 and random key by more than max(0.01 nats per frame, the range over the 20 random keys).
      - Otherwise **J BLIND**. Stage 1 is not funded, and J's terms are reported for the objective work.
      - Amended before any stage-0 result (2026-09-24, `reports/impl_a16b_stage0_2026-09-24.md`):
        - Overlong runs are split into ceil(d/D_k) same-symbol segments in both the LM term and the duration term, following S's topology.
        - An A10/A11 phi's key is the argmax of m(u|s)·pi(s). pi is fitted label-free as the mixture weights that maximise the likelihood of the train-side unit counts under a fixed m. The trigram-unigram-times-mean-duration prior occupancy is rejected: it put 0.487 on SIL against a gold frame share of 0.058, and inflated the SIL units in the keys gold must beat.
        - Conservative comparison: each phi contributes the better of its posterior key and its likelihood-only key.
    - **Stage 1, key search (CPU, one node, restarts in parallel; funded on J SEES THE KEY).**
      - Moves: reassign one unit type's symbol, via annealed ICM/Gibbs sweeps on train-side J. Split and merge happen only as unit-set moves.
      - Starts: 64 random keys and the 7 A10/A11 argmax keys.
        - Added before any stage-0 result (2026-09-24): 16 cluster-then-decipher keys.
          - The 500 units are clustered into 40 classes by label-free features: the distance between unit centroids, and each unit's left and right context distributions on the train side. Each feature set gives one clustering.
          - Each clustering is deciphered into symbols by class-level moves on train-side J, from 8 random class keys.
          - The best 16 keys by train J enter the unit-level search.
          - Each selected key's family is reported.
        - Added before any stage-1 result (2026-09-24, `reports/impl_a16b_stage1_2026-09-24.md`):
          - The search adds symbol-swap moves and 7 relabel starts (the A10/A11 partitions with deciphered maps; stage-0 agreement read).
          - Every informed start (argmax, relabel, cluster) runs under two schedules. The full schedule starts at T0, the data-calibrated median |best move|; the warm one starts at T0/100. Random starts run only the full schedule.
          - The reason: the full T0 scrambled an argmax start in timing (J -5.005 -> -5.78).
          - Selection pools both schedules.
      - Selection: the top 4 by held-out J, label-free.
      - Null: the same search on the destroyed corpus. The real and null J rises are both reported.
        - Amended before any stage-1 result: the stage-1 null corpus permutes the run-collapsed unit segments within each utterance, each segment keeping its length.
        - The reason: the frame-level permutation leaves only 1-frame runs. Its keys sit at 2.3-2.5 Hz, and repair reached only 2.52 Hz against the 5.80 floor, so in-band null keys barely exist. The run-level null destroys only the order the trigram exploits, and keeps the hard band.
        - The frame-level corpus stays as stage 0's report-only statistic.
      - Amended before any stage-1 result (review `reports/review_a16b_stage1_launch_2026-09-24.md`):
        - The null's context clustering is degenerate (one class holds 321 of 500 units). Its 8 cluster starts sit at 5.665-5.683 Hz, and their rate repair stalls near 5.69 Hz.
        - Rate repair is now bounded by a stall rule, a proposal cap and the job deadline. A start still outside the band is VOID-ON-RATE, for real and null starts alike.
        - VOID runs never fail the job. Selection takes eligible real runs only and removes duplicate keys, and the VOID counts are reported.
      - Reported, never gating or selecting: each selected key's unit agreement with the gold key.
    - **Stage 2, S-EM from a key (A10 recipe verbatim except the init; durinit for every arm).** phi's emission rows start from the key's smoothed unit counts.
      - Control, launched as soon as it is built (disclosed analysis only): the gold key as the init. It calibrates the conversion from key to phi, and it also asks whether the A14 (ii) gap survives without supervised durations.
        - GOLD KEY REACHES BASIN if S at 48 < S_min - 0.01 = 3.289. If not, stage 2's key arms are held until a conversion that reaches the basin is found.
        - Launch notes (review `reports/review_keyinit_launch_2026-09-24.md`):
          - The reader uses the unrounded bar, 3.29903 - 0.01 = 3.28903, and the A17 (iii) reader uses the same bar.
          - The key-to-phi construction's hidden pre-activation scale (PREACT_ON = 2.0) is an untraced free constant. The built emission does not depend on it, but it sets the weight scale EM starts from. It affects only this control arm, and a failed control must name it as a possible cause.
      - Key arms: the stage-1 top 4, as one four-GPU pack.
        - **KEY BASIN** if the best key arm by S at 48 (260 set, paired as in A14 (ii)) has S < 3.289; else **NO KEY BASIN**.
        - Reported, never gating: direct and Hungarian PER, NMI, and A15-F's measures at 0-48.
      - KEY BASIN sends the selected phi to the lift test (A14 (i)'s form) and to L2-2. A18 (c) amends this: the S-best key arm goes to L2-2 whatever KEY BASIN reads.
    - Cost: stage 0 takes minutes on CPU; stage 1 is one node for about 1-3 h; stage 2 is one GPU for about 3 h (the control) and one four-GPU pack for about 3 h.

- **A17 (2026-09-24, after the A14 (ii) read, before any A16 (b) or A17 result; disclosed analysis only, supervised inits) Is the phonetic basin worth reaching, and what degrades gold under EM?**
  - Trigger (user challenge, 2026-09-24): "the EM degrades the gold phi a lot, why is our conclusion still that EM could find an optimum".
    - In A14 (ii), gold-init PER goes 0.193 -> 0.300 (sub-epoch 4) -> 0.328 (12) -> 0.353 (48) while S falls. So S's optimum near gold is not gold, and A14 (ii) licenses only the ranking between basins.
    - Most of the damage comes in sub-epochs 1-4. The recipe's tau = 4 sub-epoch sends every init to S 4.71-4.92 at sub-epoch 1 (gold 3.474 -> 4.814). After that, at tau = 1, PER drifts a further +0.05 (gold) and +0.04 (r30) over 44 sub-epochs while S falls by about 0.05.
    - So there are two defects: the search (random init never reaches the basin) and the objective inside the basin. A16 (b) addresses only the first, and its ceiling is a phi at about PER 0.35-0.5.
  - **(i) Does an EM-degraded phonetic phi lift a random theta?**
    - Recipe: A14 (i)'s, verbatim, with 8 sub-epochs and both models trainable.
    - phi comes from the A14 (ii) sub-epoch-48 checkpoints: gold-EM (PER 0.353), r30-EM (0.394), r70-EM (0.495), and r100-EM (0.860) as the negative control. One four-GPU pack.
    - Read per arm: dev-other greedy PER at ep8 in A4's bands (LIFT < 0.50, PARTIAL < 0.8164, else NO LIFT).
    - Reported beside, never gating (user request 2026-09-24): the jointly trained phi's generative PER (genmarg decode of D4; direct, Hungarian, NMI) at ep8.
    - **BASIN SUFFICIENT** if gold-EM or r30-EM reads LIFT or PARTIAL and r100-EM reads NO LIFT. **BASIN INSUFFICIENT** if gold-EM, r30-EM and r70-EM all read NO LIFT. VOID if r100-EM lifts.
    - BASIN INSUFFICIENT withdraws A16 (b) stages 1-2 before any funding, and the next cost work goes to the objective.
  - **(ii) Annealing damage against objective drift.**
    - Recipe: A14 (ii)'s, but at tau = 1 from sub-epoch 1 (no tau = 4 sub-epoch), 12 sub-epochs, diagnostics at 0, 4, 8 and 12.
    - Arms: gold and r70 inits, with each init's own durations, as in A14 (ii). One GPU each, through gpupack.
    - Read the gold arm's drift, PER(12) - PER(0), against 0.193:
      - **ANNEALING-DOMINATED** if the drift is < 0.05 (A14 (ii) shows +0.135 at 12);
      - **OBJECTIVE DRIFT** if it is >= 0.10;
      - **MIXED** otherwise.
    - Reported beside: S at matched sub-epochs against A14 (ii), and the r70 arm.
    - ANNEALING-DOMINATED amends A16 (b) stage 2 to tau = 1 for key inits before its key arms run. The gold-key control then runs in both forms.
  - The 0.05 and 0.10 thresholds exceed the bed's same-config PER spread (0.01-0.03).
  - **(iii) Does the basin need supervised segmentation?** (registered 2026-09-24, before any A17 or A16 (b) result)
    - Trigger: the user reported (2026-09-24) that a colleague finds a similar approach works well under gold segmentation but not without it. The A14 config shows that every A14 (ii) arm kept its init phi's durations, which were fitted on the MFA segments. So the three arms that reached the basin (gold, r30, r70) all carried supervised segmentation. r100 carried the same segments with random labels and did not reach it, so segments alone do not suffice. Label information without segments is untested.
    - Arms: the A10 durinit recipe, verbatim except the init (seed 1, 48 sub-epochs, tau 4 then 1). Emission rows come from the gold, r30 or r70 phi; the duration logits are replaced by the general-knowledge durinit values. These are G-dur, r30-dur and r70-dur. They share one four-GPU pack with A16 (b)'s gold-key control, which takes type-level key counts plus durinit.
    - Read S at 48 on the 260 set, paired as in A14 (ii), against 3.289:
      - **SEGMENTATION-CARRIED** if G-dur S >= 3.289. The A14 (ii) basin then depended on supervised durations.
      - **LABELS SUFFICE** if G-dur and r70-dur are both < 3.289.
      - **PARTIAL-LABELS NEED SEGMENTS** if G-dur < 3.289 and r70-dur >= 3.289.
    - Reported beside, never gating: generative PER (direct, Hungarian, NMI) at 0, 4, 12 and 48. Also the boundary precision, recall and F1 (20 ms tolerance) of each phi's Viterbi segmentation against MFA on the 260 set, for these arms and the four A14 (ii) arms. Added before any result: the segment rate (Hz), the over-segmentation and the strict R-value, beside the MFA reference rate.
    - SEGMENTATION-CARRIED or PARTIAL-LABELS NEED SEGMENTS holds A16 (b) stage-2 key arms. The next cost work then goes to a label-free segmentation init, such as a self-supervised phone segmenter or the key's own run segmentation, before any key arm is funded.
    - Literature (`reports/lit_segmentation_gap_2026-09-24.md`), which changes the handoff recommendation:
      - Oracle boundaries beat unsupervised ones by 14-22 PER at the first iteration on TIMIT (Chen et al. 2019; Yeh et al. 2019).
      - Swapping only the boundaries costs 5.5 PER on LibriSpeech (Tseng et al. 2024).
      - Resegmenting with the text model in the loop closes most of the gap (Tseng 8.9 vs 6.4; Wang 2026). Resegmenting without text does not (Ondel et al. 2016).
      - The output rate predicts failure better than boundary F1: runs at 25-28 Hz fail, against a true rate of 10-12 Hz (Liu et al. 2022). A2's rate band already guards this.
      - Merging runs of units overcounts boundaries by 2.6-3.3 times (Kamper 2022). So the key's own run segmentation is the weakest fallback.
      - The best-supported label-free replacement is resegmentation under the trigram, which our HSMM EM already does, with a self-supervised segmenter only as its init.
      - No published work compares MFA and rate-matched durations in an EM HSMM, and every published segmenter tuned its settings on labels.

- **A18 (2026-09-24, user permission, before any wave, keyinit or key-arm result) Joint runs on the selected phis.**
  - The user (2026-09-24): "for all your currently planned phi EM, the bridging/joint training on selected phi is also permitted, otherwise the phi EM means nothing to us. Unless you are really sure it cannot lift".
  - Rule: each phi-EM line of the last round sends its label-free selected phi to a joint run.
    - A spend gate on S (A7's SIGNAL, KEY BASIN) no longer withholds that joint run, because none of them measures lift. The verdicts are still read and reported as registered.
    - Hold rule, the only ground for withholding a joint run ("really sure it cannot lift"): a completed lift read on phis of the same line read NO LIFT on every arm, and the candidate's S at 48 (260 set) is not below the best of those arms' S by more than 0.01 (A7's floor).
  - (a) The wave: G4a.L2.2's selected restart goes to L2-2 as registered (4 arms, A6's cold_ctl baseline, read by G4a.L2.4), whatever G4a.L2.2 reads.
    - Same-line lift read for the hold rule: A14 (i) (A10 restarts, the wave's recipe).
    - The count-table repair that SIGNAL + BELOW would fund first is new training and is not built; the bridge runs without it.
  - (b) The keyinit pack (ge1MKcAPmZIV: gold-key control, G-dur, r30-dur, r70-dur), disclosed analysis only.
    - All four sub-epoch-48 phis go to one four-arm lift pack in A14 (i)'s form, as A17 (i): no selection, launched when the keyinit pack finishes.
    - Read per arm in A4's bands. **DURINIT BASIN LIFTS** if G-dur or r30-dur reads LIFT or PARTIAL; **DURINIT BASIN DOES NOT LIFT** if G-dur, r30-dur and r70-dur all read NO LIFT.
    - Reported beside: the jointly trained phi's generative PER at ep8, and paired rows against A17 (i)'s same-init arm with MFA durations (G-dur against gold-EM, r30-dur against r30-EM, r70-dur against r70-EM). These rows ask the segmentation question at the lift level.
  - (c) Stage-2 key arms: the S-best key arm at 48 (260 set) goes to L2-2, whatever KEY BASIN reads.
    - dec_joint is A14 (i)'s arm form, so L2-2 carries the registered lift test.
    - Same-line lift read for the hold rule: (b)'s gold-key arm (the same key-to-phi conversion from the gold key).
    - The key arms' own gates (A17 (i), A17 (iii), GOLD KEY REACHES BASIN) are unchanged.
  - (d) A17 (ii)'s tau = 1 phis are not bridged by default. Their gold arm lies between L2-0's gold phi, which lifts, and A17 (i)'s gold-EM.
    - They get a lift pack (gold and r70, seeds 1 and 2) only if A17 (i) reads NO LIFT for gold-EM and A17 (ii) reads ANNEALING-DOMINATED. Only then does their lift read tell whether the tau = 4 sub-epoch is what destroys liftability.
  - Every joint arm reports its phi's generative PER (direct, Hungarian, NMI) at the kept epochs, report only.
  - Build choices, fixed before any result (`reports/impl_a18_bridge_2026-09-24.md`):
    - (b) reads **MIXED** in the one case left open: G-dur and r30-dur read NO LIFT and r70-dur lifts. MIXED is reported and has no consequence.
    - G4a.L2.4 is decided by dec_joint against cold_ctl, with B = |dec_joint - dec_joint_s2|. The other arms are reported.
    - cold_ctl reuses L2-0's `UdhhxiGIMBob`.
      - Its random phi has uniform duration logits instead of A9's durinit values. A9's prior is init-only, so this is part of phi's init, which A6 lets differ.
      - A new one-GPU baseline would have idled a whole node.
    - dec_distil trains at tau 2.0 (the arms' tau) and lr 1e-5 (the pack schedule's sub-epoch-1 value, since the D10e constants apply verbatim). Its target is the plain l_tau posterior, with no k2 term.
  - Cost: one four-GPU pack each for (a), (b) and (c), about 11.5 h each (A14 (i)'s request); (d) adds one pack if its condition holds.

- **A19 (2026-09-24, user request, before any job) Trigram-only lift ladder: does a partly wrong phi still lift a random theta without the k2 word-lexicon term?** Disclosed label-using diagnostic, as L2-0.
  - Trigger: the user asked whether, under the phone trigram alone, r30, r50 and r70 still lift, and ruled "do it here with a gpu pack".
    - L2-0's random-theta ladder ran only with the k2 block on.
    - The only trigram-only evidence starts from p0: D10e's supphi_plain drifted from 0.193 to 0.256 by ep8, against 0.180 with k2.
  - Recipe: L2-0's R-node recipe verbatim (theta at the zero-logit flat init, D10e pack constants, tau 2.0, 8 sub-epochs, kept 1/2/4/8, both models trainable, the same corrupted phi fits), with the k2 block removed. That removal is the single delta against each rt_rX arm.
  - Arms, one four-GPU pack: tri_r30, tri_r50, tri_r70, and tri_r100 as the content-free negative control.
  - Read per arm: dev-other greedy PER at ep8 in A4's bands (LIFT < 0.50, PARTIAL < 0.8164, else NO LIFT).
    - rho*_tri is the largest rho that LIFTs with every smaller rung lifting; a non-monotone ladder reads CANNOT_TELL.
    - **SAME LADDER** if rho*_tri = 0.7. **K2 NEEDED ABOVE rho*_tri** if rho*_tri < 0.7, including no lifting rung. **VOID** if tri_r100 lifts.
    - Reading fixed before any result (`reports/impl_a19_triladder_2026-09-24.md`): a rung lifts only on LIFT; tri_r100 voids the read on LIFT or PARTIAL, as in A17 (i). The job prints READING-SENSITIVE if another reading would change the verdict.
  - Reported beside, never gating:
    - The paired rows tri_rX minus rt_rX at ep1 and ep8 (`PairedPerDeltaJob`, speaker-clustered bootstrap), with M = max(0.010, |rt_r0 - rt_r0_s2| at ep8 = 0.001) = 0.010. K2 HELPS at a rung if the delta exceeds M with the interval above zero, and TRIGRAM ENOUGH otherwise.
    - The jointly trained phi's generative PER (direct, Hungarian, NMI) at ep8.
    - The ep1/2/4 PER.
  - Cost: one four-GPU pack, at most L2-0's request; steps are cheaper without k2.
- **A20 (2026-09-24, after the stage-1 read, before any job; disclosed label-using analysis, CPU, no training) Does J reward the right names on the found partitions?**
  - Trigger: the user asked whether EM has the freedom to rename. Stage 1 found keys whose clusters are sharper than K70's but whose names are wrong: identity 0.07-0.14 against K70's 0.30. J prefers them over gold through the name-free emission term.
  - The question: is the name gap a search failure (J's trigram term would reward the right names, but no local move reaches them) or an objective failure (the trigram term does not see names on these partitions)?
  - Keys: the 4 selected stage-1 keys and the 6 A10/A11 argmax finals. Gold and the K30/K70 ladder give the scale.
  - For each key, three variants:
    - (a) as found;
    - (b) oracle 1:1 rename: symbol names permuted by the Hungarian assignment that maximises frame agreement with the gold key, with the partition unchanged;
    - (c) oracle many-to-one rename: each symbol takes its majority gold phone.
  - Read held-out J and its terms on the 260 set, computed as in stages 0-1.
  - Reading, fixed before any result, on the 4 selected keys:
    - **NAMES VISIBLE** if J(b) - J(a) > 0.01 nats per frame (A7's floor) on at least 3 of the 4. J then rewards the right names on the found partitions, and a global rename move (an assignment over names under J, or S for phi) is the cost work.
    - **NAME-BLIND** if J(b) - J(a) <= 0.01 on at least 3 of the 4. The trigram term then does not reward the right names on these partitions, a rename step would not help, and the cost work goes to the objective's name signal.
    - **MIXED** otherwise.
  - Reported beside: each variant's term decomposition, J(c), the 1:1 identity agreement after (b) (it separates wrong names from merges), and the same rows for the A10/A11 finals.
  - It gates nothing registered, and stage 2 runs regardless. Cost: one CPU job on the login node.

## L2-0: the reverse model's competence ladder (disclosed label-using diagnostic)

Question: how competent must phi be to anchor a recognizer, and which label-free statistic tracks it? D10e gives rho = 0 (gold strings, HOLD 0.180); D14 gives rho about 0.19 (p0's decodes, pending). Nothing between is measured, and L2-1's phi needs a competence criterion without labels.

**Inputs.** theta = p0 (`SupervisedRecognizerExportJob.YtuVg6ZYuK0P`, dev-other 0.1894, as D14). phi_rho = `BlankfreeSupervisedReverseInitJob` with the gold phi's recipe and constants on the same 10 h seed utterances and split (the D10e / D14 fit), whose targets are the seed gold strings with a fraction rho of tokens substituted, each by a symbol drawn from the seed's gold unigram renormalised without the original symbol (seed 0; length kept; run-collapse may shorten a string). The corruption job prints the realised substitution rate and the PER of the corrupted strings against gold per rho. The label source is the only delta from `16v7R6ztSq1u`. Ladder: rho in {0.3, 0.5, 0.7, 1.0}.

**Arms.** One exclusive node, D10e pack constants verbatim (tau 2.0 held, lr schedule, 8 sub-epochs, kept 1 / 2 / 4 / 8, seed, batches; k2 block = `supphi_k2lat`'s), both models trainable: `corrphi_k2lat_r30`, `_r50`, `_r70`, `_r100`.

**Competence statistics per phi, before the pack** (one forward job per phi; CV holdout of the train stream and dev-other, 500 utterances each): (a) label-free: per-frame lattice marginal under a uniform recognizer, -log sum_a exp(log P_psi(B(a)) + log p_phi(z | B(a), eta)), at tau = 1 and the tau = 2 free energy; (b) label-free: phi's own posterior decode minus the same-speaker deranged own decode, per utterance with interval (D10b's form without gold); (c) label-using reference: gold minus same-speaker deranged gold (`BlankfreeDecodeGapJob`). The same three for D10e's gold phi and D14's decphi.

**Reads.**
1. Dev-other greedy PER at ep8 against p0 in D10e's bands (HOLD <= 0.2894, COLLAPSE >= 0.70, PARTIAL between) and the D14 rule (REFINES / PRESERVES / DEGRADES, M_w from W2's identity band, `PairedPerDeltaJob`).
2. rho* = the largest rho on {0, 0.19, 0.3, 0.5, 0.7, 1.0} that reads HOLD at ep8. `_r100` HOLD: rho* is above the ladder, a content-free but fitted phi anchors, the collapse driver is the random-init phi's gradient and not its content; then G4a.L2.3 is moot and L2-2 needs only a fitted phi. `_r30` COLLAPSE: rho* < 0.3, L2-1 must deliver a nearly phonetic phi.
3. The bar for G4a.L2.3: the label-free statistic among (a), (b) that is monotone in rho across the six phis and separates every HOLD arm from every COLLAPSE arm; its value at rho*. Neither separates: NO SEPARATING STATISTIC, and G4a.L2.3 reads CANNOT_TELL.

## L2-1: phi-first EM decipherment from a random reverse model (pure cold, label-free; the escape experiment)

**Objective.** The lattice term with the recognizer removed: theta is a null recognizer (zero logits, frozen with `freeze_recognizer`), so the posterior is the generative one, proportional to (P_psi(B(a)) p_phi(z | B(a), eta))^(1/tau); phi trains by gradient on this term (generalised EM). Rate and agg depend on theta only and are constant. Stage 1 prior: the frozen phone trigram alone. Stage 2 adds the k2 word graph (`supphi_k2lat`'s block: on-set at the first stage-2 pass, ramp 3). Temperature: deterministic-annealing EM, tau 4 -> 1 linearly over passes 1-2, held at 1 after; the tau = 1 objection in `SAE_4A_objective.md` concerns q inside the lattice, and with q absent tau = 1 is the exact E-step. Pruning at max_active 1000 sees prior and phi scores only, so no q-driven pruning restricts the search.

**Data.** Restarts train on one fixed sub-epoch of train-clean-100 (the packs' partition, about 7.1k utterances), repeated for 4 passes; selection on the CV holdout of the train stream.

**Restarts and nulls.** Wave 1: 16 restarts, phi at the bed's random init with seeds 1-16, stage 1 only, 4 passes each, 4 per node. Null: 4 restarts (seeds 1-4) of the same recipe on the structure-destroyed corpus, units permuted within each utterance (length and speaker embedding kept). Reference points on the label-free scale: L2-0's `_r100` phi and the gold phi.

**Selection.** Best per-frame held-out tau = 1 marginal likelihood among the 16. Stage 2: the top 4 continue 4 more passes with the word graph on; re-selected by the held-out total including the k2 term (D13's sum).

**Report, entering no gate:** the selected phi's posterior decode on dev-other: NMI(symbol, phone), NMI(symbol, unit), Hungarian PER against the chance band 0.83-0.91, the D10b gold-vs-deranged gap (D10a's script).

**Amendment slot (SIGNAL + BELOW):** count-table repair moves on phi, merge / split / swap on the 40 x 500 emission rows rescored by the held-out E-step likelihood, accepted only when the held-out likelihood rises by more than the null spread; never a neural refit per move.

## L2-2: the bridge into the joint run (funded on G4a.L2.2 SIGNAL; A18 (a) funds it whatever G4a.L2.2 reads)

**Arms** (one node, D10e pack constants, tau 2.0 held from the start as the supphi arms, k2 block as `supphi_k2lat`, 8 sub-epochs, kept 1 / 2 / 4 / 8): `dec_joint` (theta at the cold packs' random init, phi = L2-1's selected phi, both trainable from step 1); `dec_distil` (theta first trained one sub-epoch by cross-entropy to the stopped-gradient generative posterior with phi frozen, the reviewer's alpha = 0 update, then joint); `dec_frz` (phi frozen throughout); `dec_joint_s2` (second seed, the identity band). Cold baseline: pack C's cold k2 arm at matched sub-epochs (`k2lat_20`). Read by G4a.L2.4.

## Results

**k2 pre-flight at rt_r0's config: FAIL, R1 / R2 held (A7).** Source: `LadderK2PreflightJob.ZM3MD9viV7sM`, Slurm 1971051; `output/rt_r0/log.run.1`; read in `reports/exec_l20_preflight_fail_2026-09-23.md`.
- The failure is the one A7 predicted: a k2 int32 overflow in the first training step, with 0 steps completed. The exact error is `Array1<char>::Init Check failed: size >= 0 (-1885666895 vs. 0)` in `MultiGraphDenseIntersectPruned::PruneTimeRange`. The HLG has 23.9M states and 98.6M arcs. nvidia-smi peaked at 61.8 GiB, and the crash came 1m46 after startup.
- The wrapper missed the child's exit and held the GPU until the 2 h limit.
- Not infrastructure, so an unchanged resubmit is not valid.
- Cause (`reports/debug_l20_preflight_overflow_2026-09-23.md`): the crash is in the pre-training stability read, not in training. That check scores 16 utterances in one k2 call at max_active 10000. With a uniform posterior, all 16 are identical, and k2's 20-frame window holds 2.41e9 arcs, above the int32 limit. The rung-1000 call on the same 16 utterances completed. The cold k2lat_20 passed the same check only because its theta was trained by its on-set at 8.
- Remedy chosen: implementation-only, with no constant changed. The stability read scores 1 utterance per call (about 2.1e8 arcs at rung 10000; the sample stays 16), and training uses a small chunk. k2's per-sequence max_active makes chunking exact. The rerun pre-flight measures the resulting step time.
- The rejected alternatives change registered constants: a smaller search beam (9.3) or on-set 8 (A4).
- L2-1 runs no k2. L2-2's dec_joint carries the same exposure and takes the same fix.

**k2 pre-flight rerun at rt_r0's config (chunked): PASS, R1 / R2 released (A7).** Source: `LadderK2PreflightJob.Pl51viCk4CVP`, Slurm 1975698, COMPLETED; read in `reports/exec_l20_preflight_rerun_read_2026-09-23.md`.
- The wrapper stopped the run at its 100-step budget (returncode -15), not at the time limit. The logs contain no ABORT, overflow or OOM.
- Stability read, sub-epoch 1: median 0.0809 nats per retained frame, 16/16 utterances, 2.4 s.
- Steps: 100/100, median 16.1 s per step.
- Peak GPU memory: 76.7 GiB, against the 80 GiB bar.
- Projection: 8 sub-epochs take 2.23 h (1,004 s per sub-epoch including dev) against TIME_RQMT 11.5 h, with no resume.
- 2 empty lattices in 100 steps, no abort.

**L2-1 probe (A3 / A9): NO WAVE SETTING, gain clause only.** Source: `PhiFirstProbeReadJob.RuFm51PHSz4q` (`output/report.txt`, `probe_read.json`); restarts (seed 1 / 2) uniform TVCw6EcU5ahd / F4WuEU8vqmeI, durinit zWqS49iSFTdV / G270UY24rWaM, durfrz ivlkWhhMJd53 / OAlQmq7yNkQh (review `reports/review_l21_probe_2026-09-23.md`), Slurm 1971921-1971926, all COMPLETED; read in `reports/exec_l21_probe_stall_2026-09-23.md`.
- Time passes: 57 steps per sub-epoch at 3.44 s per step and 95% GPU. A restart takes 0.246 h, against the 1.5 h bar.
- Rate passes: the emitted rate is 0.98-1.41 Hz at sub-epoch 1 (tau 4) and in band from sub-epoch 2-3 on.
- Gain fails for every restart. At sub-epoch 4, S is the held-out tau = 1 NLL per frame (lower is better; 285 utterances):

  | Setting | Seed 1: S / Hz / gain 3-4 | Seed 2: S / Hz / gain 3-4 |
  |---|---|---|
  | durinit | 3.683 / 7.83 / 0.285 | 3.709 / 7.47 / 0.279 |
  | durfrz | 3.717 / 7.59 / 0.299 | 3.739 / 7.42 / 0.296 |
  | uniform | 3.760 / 6.53 / 0.314 | 3.709 / 6.74 / 0.333 |

- S falls from 5.77 at sub-epoch 1, through 4.91 and 3.97-4.07.
- E[d] for the phone types: durinit drifts from 4.09 to 4.86-4.94; durfrz is held at 4.41; uniform falls from 12.8 to 9.95. SIL falls from 24.9 to about 18.4 in every setting.
- Consequence: A10's re-derivation.

**A10 extension read: WAVE SETTING durinit, 12 sub-epochs.** Source: `PhiFirstA10ReadJob.DcfCsZNq1ucr` and diagnostics `PhiFirstA10DiagnosticsDisjointJob.G2NeV8oNr4tO`, all six restarts COMPLETED at sub-epoch 48; read in `reports/extract_a10_read_2026-09-24.md`.
- **K\*:** durinit 12, durfrz 10, uniform 10 (report only). Every rate is in band at 48, between 6.8 and 7.6 Hz.
- **Choice:** at sub-epoch 12, the largest K\* among the candidates, mean S is 3.4217 for durinit and 3.4571 for durfrz. So WAVE_DURATION_SETTING = durinit and WAVE_NUM_SUBEPOCHS = 12.
- **Identity band:** the largest difference between sub-epochs 1-4 and the probe is 3.4e-5.
- **S at sub-epoch 48, 260-utterance disjoint set, with the 285 set in parentheses:**

  | Setting | Seed 1 | Seed 2 |
  |---|---|---|
  | durinit | 3.2990 (3.3040) | 3.3703 (3.3760) |
  | durfrz | 3.3667 (3.3720) | 3.3996 (3.4058) |
  | uniform | 3.3476 (3.3505) | 3.3224 (3.3277) |

  The gold phi reads 3.4735 (3.4702) and L2-0's r100 phi 4.6901 (4.6861). Every restart is below gold from sub-epoch 12 on.
- **Report only (dev-other, 500 utterances, phi's genmarg decode):**
  - At 48, direct PER is 0.832-0.856 and Hungarian PER 0.838-0.861. Both are inside the chance band 0.83-0.91 at every checkpoint, and neither improves after sub-epoch 4.
  - NMI(symbol, phone) is 0.079-0.113.
  - E[d]: durinit 6.0-6.2, uniform 6.2-6.3, durfrz held at 4.41.
- **What it says, untested interpretation:** EM from random init finds phis that the stage-1 held-out likelihood prefers to the gold phi, and whose decodes carry no phonetic content. A5's bar, which needs a statistic that separates competent from sharp-wrong phis, is the same question. The A14 analysis tests whether the objective or the search is responsible.

**A11 / A12 read (EM count table): G4a.L2.2 = CANNOT_TELL (NO ELIGIBLE NULL).** Source: `EmTableSelectionJob.RyEwer4kERuw` (`output/report.txt`, `selection.json`) and diagnostics `EmTableDiagnosticsJob.JjdvpYwqYyfH`, both finished; `reports/exec_a11_selection_rerun_2026-09-24.md`.
- Rate clause: all 12 real stage-B finishers are eligible (6.88-7.38 Hz). None of the 12 frame-permuted null finishers is (1.63-4.09 Hz, against the band [5.80, 14.49]). So the family reads CANNOT_TELL, as A12 provides.
- Selected: real_c_s16 (durinit, tau 1), S 3.536 on the 285 CV holdout, paired over 285 of 285 utterances. The real finishers span S 3.536-3.608. The exact reruns of b_s04 and b_s07 reproduce S, so the identity band is 0. Every finisher stopped at iteration 2 or 3 (gain < 0.01).
- Report only:
  - The nulls' S is 5.80-5.81 on their permuted holdout and 5.66-5.81 on the real holdout.
  - The selected S minus the best run-preserving null (runnull_a_s03) is -1.696 on its own holdout and -1.648 on the real holdout.
  - The stage-A Spearman between S at iteration 10 and at the end is 0.28 / 0.10 / 0.59 for real arms a / b / c, against 0.94 / 0.72 / 0.93 for the nulls.
- Report only, label-using (D4 dev-other, 500 decoded, none impossible):
  - Real: direct PER 0.844-0.863, Hungarian 0.850-0.872, NMI(symbol, phone) 0.070-0.094, NMI(symbol, unit) 0.458-0.490.
  - Nulls: direct PER 0.848-0.900, NMI(symbol, phone) 0.128-0.318, NMI(symbol, unit) 0.013-0.042.
  - The decodes sit at chance, as A10's do. The symbols track the units but not the phones under their own labels. A15-E was not run on the tables, so whether they carry A10's mislabelled phonetic content is open.

**L2-0 ladder read: a fitted phi lifts a random theta; rho*_lift = 0.7; G4a.L2.3 = CANNOT_TELL (audit CONFIRMED_WITH_NOTES, `reports/audit_l20_ladder_2026-09-24.md`).** Sources: packs P `WX41NC734WLo`, R1 `mZaZk7Ptt5Sg` and R2 `UdhhxiGIMBob`, and the reader `LadderCompetenceDisjointReadJob.EktNvNSRrXKj`, all COMPLETED; read in `reports/extract_l20_ladder_read_2026-09-24.md`.
- **Audit.**
  - Theta in R1 and R2 is the zero-logit flat init, never p0.
  - Realised substitution rates are 0.300, 0.500, 0.700 and 1.000. permphi changes every token, and cold_ctl has no reverse checkpoint.
  - PER uses the same 2864 utterances and scorer as p0 and the chance band.
  - No transcripts are seen beyond the disclosed 10 h seed labels. The prior, graph and lexicon come from the LibriSpeech LM corpus and the official lexicon.
- **Dev-other PER, ep1 / ep8, and the A4 class:**

  | Arm | ep1 | ep8 | Class |
  |---|---|---|---|
  | rt_r0 | 0.1807 | 0.1776 | LIFT |
  | rt_r0_s2 | 0.1809 | 0.1786 | LIFT |
  | rt_r30 | 0.1861 | 0.1826 | LIFT |
  | rt_r50 | 0.1902 | 0.1863 | LIFT |
  | rt_r70 | 0.2533 | 0.1909 | LIFT |
  | rt_r100 | 0.8445 | 0.8552 | NO LIFT |
  | cold_ctl | 0.8881 | 0.8487 | NO LIFT |
  | rt_perm | 0.8891 | 0.8655 | NO LIFT |

  rho*_lift = 0.7, and the ladder is monotone. A7's rt_r0 condition for L2-2 is met; SIGNAL is pending. The wave is not held.
- **Node P** (theta = p0), at ep8: r30 0.1817, r50 0.1824 and r70 0.1887 read HOLD; r100 0.8530 reads COLLAPSE. By the D14 rule with M_w = 0.010, r30 to r70 read PRESERVES (r30 -0.0078 [-0.0132, -0.0026]) and r100 reads DEGRADES (+0.6635).
- **A5 bar on the 260 set** (the 285 set agrees):
  - (a), tau = 1 NLL per frame: gold 3.474, r30 3.719, r50 3.959, r70 4.411, r100 4.690, phi_c 3.499, permphi 4.075.
  - (b), own minus deranged: gold +4.30, r30 +2.53, r50 +1.91, r70 +1.08, r100 +0.64, phi_c +4.56, permphi +4.10.
  - Both are monotone in rho and separate r70 from r100. Both put phi_c and permphi on the lifting side, although permphi's own arm reads NO LIFT. So G4a.L2.3 = **CANNOT_TELL**: no separating statistic.
  - (c) was not built, because its inputs do not exist.
- **Audit notes.**
  - N1: the corruption is independent of the acoustics, so each phone's most likely unit stays right below rho 1. rho*_lift therefore does not carry over to an EM phi, whose errors are structured. A14 (i) tests an EM phi directly.
  - N2: the label-free statistics rate the sharp-wrong phi_c as competent as gold.

### A15 and A15-E read (2026-09-24; descriptive, label-using, no gate; audited CONFIRMED_WITH_CORRECTIONS)

Sources:
- `reports/extract_a15_read_2026-09-24.md`: reader `PhiCompetenceBatteryReadJob.HWlC9YZ1U2GI`.
- `reports/extract_a15e_read_2026-09-24.md`: reader `PhiEmissionMapReadJob.mZWCsgaj92GS`.

Conventions:
- Set: the D4 dev-other set of 500 utterances. The item set is identical for every phi and contrast, and nothing was dropped.
- T columns are per-frame nats, with speaker-clustered 95% intervals (listed in the extracts).
- R4 is frame accuracy against MFA, with the majority-unit oracle 0.615.
- "emis" means through the A15-E emission map. The EM rows cover the six A10 restarts at sub-epoch 48.

| phi | T1, own labels | T2 full | T3 primary, emis | R1 / R2 Hungarian PER | R3 JS direct / emis | R4 direct / emis | phones unclaimed |
|---|---|---|---|---|---|---|---|
| gold | 3.93 | 4.47 | 3.93 | 0.193 / 0.276 | 0 / 0 | 0.589 / 0.589 | 0 |
| r30 | 2.22 | 2.44 | 2.22 | 0.240 / 0.363 | 0.116 / 0.116 | 0.571 / 0.571 | 0 |
| r50 | 1.58 | 1.77 | 1.58 | 0.313 / 0.437 | 0.232 / 0.232 | 0.548 / 0.548 | 0 |
| r70 | 0.57 | 0.64 | 0.59 | 0.628 / 0.648 | 0.532 / 0.522 | 0.371 / 0.342 | 15 |
| r100 | 0.00 | 0.06 | 0.18 | 0.830 / 0.862 | 0.728 / 0.652 | 0.092 / 0.136 | 37 |
| A10 EM ×6 | 0.08-0.34 | 0.35-0.71 | 1.99-2.20 | 0.838-0.861 / 0.846-0.869 | 0.79-0.87 / 0.50-0.55 | 0.07-0.12 / 0.29-0.31 | 11-19 |
| phi_c | 0.16 | 0.52 | 1.75 | 0.868 / 0.859 | 0.827 / 0.563 | 0.099 / 0.284 | 15 |
| permphi | -0.07 | -0.24 | 3.92 (40/40 labels) | 0.819 / 0.605 | 0.868 / 0.007 | 0.071 / 0.591 | 0 |
| random_init | 0.00 | 0.00 | 0.00 | 0.903 / 0.921 | 0.717 / 0.716 | 0.080 / 0.080 | 39 |
| rt_r70_ep8 | 4.04 | 4.33 | 4.04 | 0.228 / 0.326 | 0.111 / 0.111 | 0.564 / 0.564 | 0 |

- **Under their own labels, the EM phis sit near the content-free end.**
  - T1 is 0.08-0.34. Every interval excludes 0, but it is 12-50 times below gold and below r70's 0.57 for all six.
  - R4 direct is 0.07-0.12, against r100's 0.092 and random_init's 0.080. The R1 and R2 Hungarian PERs are at r100's level.
- **Through the emission map, they carry real structure, but below r70 and at about the manner-class level** (corrected by the audit). [The manner-class part was overturned by the A15-F read below: within-class accuracy is 0.54-0.62, and the errors cross classes.]
  - R4 emis is 0.29-0.31 (non-SIL 0.26-0.29). That is below r70 through the same map: 0.342, non-SIL 0.310 (r70's 0.371 is its direct R4). JS emis is 0.50-0.55, against r70's 0.52.
  - SIL is identity-mapped in all six, which lifts both R4 columns. On non-SIL frames, own-label R4 is 0.029-0.082, at chance or the majority level.
  - The audit ran a scratch bound that is not banked. It assigns units to groups and maps groups to phones one-to-one, choosing the map from the MFA labels, which bounds any label-free map from above.
    - Random 40-way unit partitions (200 draws) reach mean 0.118 and max 0.129. So the EM R4 reflects real structure, not the map's selection.
    - A partition that knows only 7 manner classes reaches 0.287, and one that knows 13 place and voicing classes reaches 0.377. So R4 cannot separate phone-level from broad-class content.
    - Against a pure broad-class reading: the nearest-to-second-nearest gold phone gap is 0.12-0.17 bits (r70 0.036), and only 38-50 % of a symbol's 2nd and 3rd nearest phones share its nearest phone's manner class (gold 70 %). The data do not decide between the readings.
  - r100 is flat, so it bounds the map's selection gain only for flat phis. It is not a null for a sharp phi's T3 gain, because T scales with sharpness. No sharp destroyed-structure control was run (A15-F).
  - **Relabelled fit of the gold string** (pooled log p per frame, `PhiEmissionMapJob.*/output/per_utterance.json`):
    - The EM phis score -6.03 to -6.74 under the emission labelling. That is worse than content-free r100 (-5.26) and r70 (-4.82), and near random_init (-6.54).
    - permphi's relabelling restores -3.67, against gold's -3.65.
    - The EM phis' T3 of about 2.0 comes from penalising the deranged string (-7.94 to -8.66), not from fitting the right one. T3 therefore ranks phis within a family only.
- **The EM phis are mislabelled, and also merged.**
  - T3 relabel (emis against identity) is positive for every EM phi (+1.2 to +1.7), against r70's -0.04 and r100's -0.01. Only the sign transfers across families.
  - The emission map agrees with the own labels on only 1-6 of 40 symbols, with SIL among them in all six.
  - Under nearest-phone matching, the 40 symbols claim only 21-29 gold phones. The 11-19 unclaimed phones hold 19-40 % of non-SIL frames: AO, AW, OY and UH in 6/6; AE, G, JH, SH, TH and ZH in 5/6; IH and EH in 4/6; AH, the most frequent non-SIL phone, in 3/6. SIL, N, T, W, AY, Z and M each take 3-4 symbols.
  - T2 full is 0.35-0.71, against r70's 0.64. Own labels still beat a random relabelling, so some symbols sit on or near their phones.
- **Along the trajectory (durinit s1 and durfrz s1 at sub-epochs 4, 12 and 48), T3 and the relabelled fit rise, but R4 emis is flat.**
  - T3 primary emis: 1.77, 2.08, 2.20 and 1.77, 1.92, 2.03 (sharpness-confounded).
  - Relabelled gold fit: -7.36, -6.74, -6.32 and -7.51, -6.59, -6.43. JS emis falls slightly (0.549, 0.518, 0.511 and 0.567, 0.543, 0.531).
  - R4 emis: 0.291, 0.288, 0.312 and 0.291, 0.286, 0.294.
  - T1 under own labels falls from sub-epoch 4 to 12, then stays flat: 0.150, 0.077, 0.078 and 0.230, 0.115, 0.116.
  - There is one seed per setting.
- **Decode-based detectors cannot see this.** GenDecodeReportJob aligns decoded tokens to gold by identity-label edit alignment before its Hungarian step, so its map and NMI depend on the labels. permphi shows this: R1 0.819 and R2 0.605, while the emission map gets 40/40 and R4 emis reads 0.591. So chance-level decode PER says nothing about the EM phis' content, in either direction.
- **phi_c, the cold line's phi, has the same profile, slightly weaker on every emission-map column**: R4 emis 0.284, JS 0.563, T3 1.75, relabel 1.07, 15 unclaimed, relabelled fit -6.80. This qualifies audit note N2: phi_c is a mislabelled partition with real structure, not a content-free one.
- **Interpretation (audited, `reports/audit_a15_read_2026-09-24.md`, CONFIRMED_WITH_CORRECTIONS; the first-draft reading "not on content, permphi's case" was contradicted).**
  - r70 lifts because its labels are right (identity is its best labelling, T3 relabel -0.04) and its argmax is right on 0.371 of frames. Joint training then repairs it toward gold: rt_r70_ep8 reads T1 4.04, R4 0.564, R1 0.228.
  - The EM phis are mislabelled and carry more phone structure than any structureless partition. But they are not in permphi's case. Even relabelled, they sit below r70 on R4, at the manner-class level, with 19-40 % of non-SIL frames on unclaimed phones, and they fit gold worse than content-free r100.
  - So L2-1 fails at least on labelling and is also short of r70 on content. rt_perm's NO LIFT (0.87) shows that wrong labels suffice to block even a fully phonetic phi. It does not show that the EM content would suffice with the right labels.
  - Whether relabelling alone would help is untested: A16 (a) for the objective's preference, A15-F for the content resolution. A14 (i) tests the direct use.

### A16 (a) read (2026-09-24): OBJECTIVE LABEL-BLIND OR WRONG (analysis only, label-using)

Source: `RelabelSReadJob.Y9RU2PSbVfuu` (`output/table.txt`); launch `reports/exec_a16a_launch_2026-09-24.md`. Set: A13's 260 utterances, paired over all 260. Intervals: speaker-clustered bootstrap, 95 %.
- dS = S(emis) - S(identity) is positive for all six A10 sub-epoch-48 phis, with every interval above 0:

  | phi | dS | utterances with dS > 0 |
  |---|---|---|
  | uniform s1 | +0.219 | 257 / 260 |
  | uniform s2 | +0.308 | 259 / 260 |
  | durinit s1 | +0.242 | 258 / 260 |
  | durinit s2 | +0.202 | 258 / 260 |
  | durfrz s1 | +0.278 | 259 / 260 |
  | durfrz s2 | +0.123 | 250 / 260 |

  0/6 fall below -0.01. S(emis) is 3.52-3.64, against S(identity) 3.30-3.40 and gold 3.474.
- The control is valid. permphi's true inverse moves S from 4.075 to 3.477, a dS of -0.598 [-0.615, -0.577], which is gold's level.
- Identity S reproduces the banked values within 2.6e-4.
- Reading: at lambda = 1, S prefers each EM phi's own (wrong) labelling over the emission-map labelling, by 0.12-0.31 nats per frame, on 250-259 of 260 utterances. By the registered branch, the label signal must come from the prior term; A16 (a2) (the prior weight) tests that first.
- Caveat: the emission-map labelling is one-to-one over a merged, manner-level partition (A15 read), and it is not refit. So it is a weak stand-in for the right labelling, and A15-F bears on that. The read does not show that S would reject a phone-level phi with the right labels. permphi shows the opposite for a gold-quality phi.
- Audit correction (2026-09-24, `reports/audit_a16_combined_2026-09-24.md`, CONFIRMED_WITH_CORRECTIONS): the verdict stands as registered, but it does not show a preference for wrong content. Relabelling keeps the emission and duration terms, so dS measures only the trigram's reaction to a renaming. After co-adaptation, a positive dS is expected by construction. The one-to-one map also forces 12-20 of 40 symbols onto phones they do not resemble. No right-labelled negative control (r70 or r30 under their own maps) was run.

### A16 (a2) read (2026-09-24): NO LAMBDA <= 3 (analysis only, label-using)

Source: `PriorScaleSReadJob.ttn9rEAZ1VQd` (`output/table.txt`, `per_utterance.tsv`); launch `reports/exec_a16a2_launch_2026-09-24.md`. Set: A13's 260 utterances, paired over all 260 at every lambda. At lambda = 1 the banked S reproduces within 2e-6.
- **Gold never ranks first.** S(gold) - S(EM phi), identity labels, over the six A10 sub-epoch-48 phis: +0.074 to +0.175 at lambda 1, +0.071 to +0.158 at lambda 2, +0.091 to +0.202 at lambda 3. Every interval is above 0. The gap does not close as lambda rises.
- **The ladder stays strictly monotone at every lambda.** Gold, r30, r70, r100:

  | lambda | gold | r30 | r70 | r100 |
  |---|---|---|---|---|
  | 1 | 3.474 | 3.719 | 4.411 | 4.690 |
  | 2 | 4.055 | 4.255 | 4.782 | 5.016 |
  | 3 | 4.601 | 4.731 | 5.072 | 5.243 |

  The ladder steps shrink with lambda (r30 - gold: 0.246, 0.201, 0.130).
- **phi_c, the cold line's phi,** is 3.499 at lambda 1 and 4.062 at lambda 2. At lambda 3 it is 4.559, below gold's 4.601.
- **A weighted prior makes S prefer the EM phis' own labels more, not less.** The (a) dS (emis minus identity) is +0.12 to +0.31 at lambda 1, +0.15 to +0.41 at lambda 2 and +0.15 to +0.44 at lambda 3, and 6/6 are above -0.01 at each. permphi's control strengthens: -0.60, -0.79, -0.85.
- **Reading.** Yin et al.'s fix, the literature value lambda = 3, does not remove the model error seen at lambda = 1. S at any lambda up to 3 ranks the EM phis above gold, and above their own emission relabellings.
  - With an up-weighted trigram, the EM phis' own symbol sequences fit it better than the relabelled ones. So EM has matched phone-trigram statistics with a labelling that is acoustically wrong.
  - lambda is not tuned further from this read (A16 (a2) rule).
- **Open, for the audit:**
  - The gold phi was fitted on the 2821-utterance split, while the EM phis trained on the train stream. Part of their S advantage may therefore be density fit from more data, not a preference for wrong content. No same-data gold phi exists to separate the two.
  - The per-frame emission term (500-way, about 3.3 nats per frame) dwarfs the phone prior (about 7 phones per second against 50 frames). That scale mismatch is a candidate mechanism, not established.
- **Audit correction (2026-09-24, `reports/audit_a16_combined_2026-09-24.md`, CONFIRMED_WITH_CORRECTIONS).** The numbers and the verdict stand. The Reading's "model error" does not follow, and it is withdrawn.
  - Gold and the EM phis differ in data (2821 utterances, supervised, against about 10x the utterances under S itself) and in criterion. So gold is not the same-data gold model of Yin et al.'s model-error test. The L2-0 ladder does hold data and criterion fixed, and there S rewards correct labels at every lambda.
  - The lambda slopes (audit-derived, not banked) put the prior term's cost per frame at about 0.58 for gold and for the EM phis alike. The gold-EM gap therefore sits in the channel (emission, duration, entropy), where data and criterion act. The trigram does not score gold's posterior better. Up-weighting it cannot close the gap, and between lambda 2 and 3 it widens it.
  - Whether S's minimum is phonetic is the registered question of A14 (ii) (same data, same criterion, gold init), which is live and unread.
  - Licensed now: re-weighting the phone trigram is ruled out, and a permutation-only label search on S is unsupported. Search moves versus an objective change stays open until A14 (ii) reads.

### A15-F read (2026-09-24): content resolution and sharp nulls (descriptive, label-using, no gate; audited CONFIRMED, `reports/audit_a16_combined_2026-09-24.md`)

Source: `PhiContentReadJob.vySKYIh5RAaB` (`output/table.txt`, `table.json`); launch `reports/exec_a15f_launch_2026-09-24.md`; build `reports/impl_a15f_content_2026-09-24.md`. Set: D4 dev-other, 500 utterances. Rules as amended at build.
- **Oracle bounds (iii), primary class assignment.** Groups map to phones one-to-one by the label oracle.
  - Random 40-way partitions: R4 0.118 (max 0.129), within-class 0.370 (max 0.429).
  - The 7-manner-class partition: R4 0.287, class share 0.767, within-class 0.374.
  - The 13-class partition: R4 0.377, within-class 0.491.
- **Class share and within-class accuracy, through the emission map:**

  | phi | class share | within-class | R4 emis |
  |---|---|---|---|
  | A10 EM ×6 (sub-epoch 48) | 0.50-0.56 | 0.54-0.62 | 0.29-0.31 |
  | trajectory phis (sub-epochs 4 and 12) | 0.53-0.56 | 0.52-0.55 | 0.29 |
  | phi_c | 0.56 | 0.51 | 0.28 |
  | A11 table real_c_s16 | 0.50 | 0.57 | 0.28 |
  | r70 | 0.49 | 0.70 | 0.34 |
  | r100 | 0.31 | 0.43 | 0.14 |
  | gold | 0.73 | 0.81 | 0.59 |

  On non-SIL frames, the EM phis read R4 0.26-0.29 and within-class 0.50-0.58.
- **ABOVE MANNER LEVEL (amended rule): yes for every EM phi.** Within-class 0.54-0.62 exceeds all three references: the 7-class oracle 0.374, the random maximum 0.429, and each phi's own null maximum 0.35-0.39.
  - The flag also reads yes for r100 (0.433, just above 0.429, no own null). So without an own null the flag is weak. The EM margins are wide.
- **Sharp permuted-unit nulls (iv), the EM phi's value against the maximum over its 5 nulls:**
  - T3 primary: 1.99-2.20 against 0.21-0.26.
  - T3 relabel: 1.23-1.70 against -0.09 to +0.18.
  - R4 emis: 0.29-0.31 against 0.10-0.12.
  - Relabelled gold fit: -6.0 to -6.7 against -10.0 to -10.6.
  Every EM gain exceeds its sharpness-matched null. The nulls leave 14-24 phones unclaimed.
- **Relabelled fit per gold phone (v).** Claimed phones score -5.4 to -6.1, and unclaimed phones -7.7 to -8.8, which hold 16-36 % of frames. Gold scores -3.70.
- **A11's selected table (vi) has the A10 profile.** 10 phones unclaimed, R4 emis 0.283, class share 0.50, within-class 0.565, ABOVE MANNER yes.
- **Reading.** The EM phis are not a manner-class partition.
  - Within a class, they pick the right phone at 0.54-0.62, well above the 7-class oracle's 0.37 and their own sharp nulls.
  - Their errors cross classes: class share is 0.50-0.56, against gold 0.73 and the oracle 0.77. This is r70's class share (0.49) with lower within-class accuracy (r70 0.70). So their content is close to, and somewhat below, r70's.
  - Their poor relabelled fit is set by sharpness. A sharp phi without content scores -10, so the comparison with the flat r100 (-5.26) does not measure content.
  - This matches the audit's alternative (f): the symbols are partly phone-specific and partly cross-class mixtures.
- **Overturned (2026-09-24).** The A15 read's corrected bullet "at about the manner-class level", and the audit's broad-class reading, do not survive A15-F. R4 had matched the 7-class oracle, but the class split shows phone-level distinctions within classes, with the errors across classes. Also overturned: the relabelled-fit comparison with r100 as evidence of missing content, which was sharpness-confounded. Still standing: mislabelled, merged, and below r70 on R4.

### A14 (ii) read (2026-09-24): PHONETIC BASIN LOWER (analysis only, supervised init; audited CONFIRMED_WITH_CORRECTIONS)

Source: `A14ObjectiveFloorReadJob.PiYQ1OCFD4ot` (`output/report.txt`, `a14_floor.json`). Extraction: `reports/extract_a14ii_read_2026-09-24.md`. Audit: `reports/audit_a14ii_read_2026-09-24.md`. The four restarts ran the A10 recipe on the same stream, one GPU each.
- **Verdict.** S_g = 3.216 (gold init, sub-epoch 48, 260 set, paired 260/260) against S_min = 3.299 (durinit s1). The difference is -0.083, with a speaker-paired interval of [-0.094, -0.072], and 222 of 260 utterances are lower. The gold-init Hungarian PER at 48 is 0.353 (the map is the identity, so the direct PER is the same). Both clauses of PHONETIC BASIN LOWER hold.
- **By init** (reported only):

  | init | S at 0 | S at 48 | PER at 0 | PER at 48 |
  |---|---|---|---|---|
  | gold | 3.474 | 3.216 | 0.193 | 0.353 |
  | r30 | 3.719 | 3.210 | 0.240 | 0.394 |
  | r70 | 4.411 | 3.271 | 0.608 | 0.495 |
  | r100 | 4.690 | 3.378 | 0.826 | 0.860 |
  | A10 random (6) | | 3.30-3.40 | | chance band |

- **Audit.**
  - Provenance holds: the gold sub-epoch-48 forward loads `fTBdXD0SwBaA` epoch 48, which starts from gold `16v7R6ztSq1u`.
  - S is computed identically for all arms.
  - The PER is computed as in A10.
  - Correction: the gold init replaced durinit, so the gold arm kept its supervised durations. The implementer disclosed this; the A14 text did not. A16 (b)'s gold-key control reruns the comparison with durinit.
  - The margin is about the six-restart seed range (0.10), and the gold init has one seed. Three phonetic inits (gold, r30, r70) all end below S_min.
- **Reading.**
  - At matched data and criterion, S's lower basin is phonetic, so the random-init restarts are search-limited. This settles the gold-versus-EM confound left open by A16 (a2).
  - An init with 70 % random label noise reaches the basin; one with 100 % does not.
  - Not licensed:
    - that S's global minimum is phonetic;
    - that search can reach the basin from no labels;
    - that S rewards accuracy inside the basin. Gold-init PER rose from 0.19 to 0.35 while S fell, and r30 ends 0.006 below gold.
  - So a perfect search would stop at about PER 0.35-0.5 before the joint run.

### A16 (b) stage 0 read (2026-09-24): J SEES THE KEY (audited CONFIRMED_WITH_CORRECTIONS)

Source: `KeyFloorReadJob.DgDbciHlq2wY` (`output/table.txt`, `table.json`); 260 held-out utterances, train side 28254 utterances. Launch: `reports/exec_a16b_stage0_2026-09-24.md`.
- Ladder, held-out J (seed means): gold -4.818, K30 -5.610, K70 -6.334, K100 -6.560; monotone.
- Best A10/A11 comparison key: a10_durfrz_s01, -4.971, a gap of 0.153 against a margin of 0.144 (the random-key range). The best random key's gap is 1.64. No verdict key is VOID. The decisive comparison clears the margin by 0.009.
- The A10/A11 argmax keys span -4.97 to -5.18, above K30 (70 % of units correct) and below gold. The r30 argmax key reads -4.822 and r70 -5.548 (report only). Fitted pi(SIL) is 0.061-0.074 for the A10 phis.
- Audit (`reports/audit_a16b_stage0_2026-09-24.md`): the verdict is what the registered rule gives. J is computed the same way for every key. The gold key uses only the 2821 fit utterances. The held-out 260 are disjoint from them, and no labels reach any non-gold key.
  - Fragile:
    - Gold beats the best EM key only on the trigram term (+0.239); the emission term (-0.080) and the duration term (-0.005) favour the EM key.
    - On the train side, which is report-only, the margin clause fails: 0.1416 against 0.1466.
    - A fresh draw of 20 random keys would reverse it about 40 % of the time.
    - J does not separate gold from r30, whose keys differ in 91 units (-0.004 held out).
  - Correction: the A10 keys are not chance-level. They are clusters with wrong names, direct agreement 0.06-0.11 against 0.03 for random keys.
  - What the verdict licenses: the stage-1 search is funded, as registered. It does not license that J's maximum is at or near gold. Whether a non-gold key beats J(gold) cannot be determined from stage 0; stage 1 and the relabel starts answer it.
- Key agreement with the gold key (`KeyAgreementReportJob.HZxcl9qgT1im`, report only, label-using):

  | key | held-out J | identity agreement, frame | many-to-one, frame |
  |---|---|---|---|
  | gold | -4.818 | 1.00 | 1.00 |
  | r30 | -4.822 | 0.86 | 0.86 |
  | A10 argmax | -4.97 to -5.06 | 0.07-0.13 | 0.46-0.50 |
  | r70 | -5.548 | 0.45 | 0.48 |
  | K30_s1 | -5.602 | 0.69 | 0.71 |
  | K70_s1 | -6.368 | 0.30 | 0.36 |
  | K100_s1 | -6.520 | 0.00 | 0.23 |

  - J against identity agreement, Spearman: 0.75 over all 56 keys, 0.32 without the random keys.
  - The A10 keys are good partitions carrying wrong labels. Their emission term, which does not depend on the labels, beats gold's (-3.27 against -3.35). Only the LM and duration terms see the labels.
  - So J is monotone in accuracy along the random-reassignment ladder, but not across families. It ranks mislabelled clean partitions above 70 %-correct noisy ones.
  - Consequence for stage 1, a pre-result amendment:
    - add symbol-swap moves, which exchange two symbols' unit sets and change only the LM and duration terms;
    - add relabel starts, from each A10/A11 partition with its class-to-symbol map deciphered under J.
  - Deciphering the A10 partitions also answers a question for the handoff: whether J at the best relabelling of an EM partition reaches J(gold).

### G4a.L2.2 wave read (2026-09-24): SIGNAL, NOT BEYOND PRIVATE CODE (audited CONFIRMED_WITH_CORRECTIONS)
- Source: `GenMargSelectionJob.dPjElzqvLUYt`, over all 16 restarts, 4 nulls, 2 phi_c restarts and the seed 1/2 reruns. The wave ran durinit for 12 sub-epochs, and S was read at sub-epoch 12 on the 285-utterance CV holdout (the nulls on its permuted copy). Extraction: `reports/extract_wave_read_2026-09-24.md`. Audit: `reports/audit_wave_read_2026-09-24.md`, which recomputed the read from the per-restart files.
- Selected restart: em_s13, S 3.3863 (next best em_s01, 3.3981). Its emitted rate is 7.58 Hz, and all 16 restarts are in band, so none is VOID.
- Best null: null_s02, S 5.7133. Null range 0.0027, identity band 2.7e-5, so the margin is 0.01. The gap of 2.327 is SIGNAL, and em_s13 beats the best null on all 285 utterances.
- Report only: the best phi_c restart (phic_s01, S 3.3441) is 0.042 below em_s13, so the read is NOT BEYOND PRIVATE CODE.
- Reading: A7 expects SIGNAL from any segmental fit, so it shows no phonetic content. The wave's reads compute no generative PER. The selected phi's PER will come from A18 (a)'s dec_frz arm, whose phi stays frozen.
- Audit corrections, none changing the verdict:
  - report.txt quotes A7's funding rule as LIFT only; the registered rule is LIFT or PARTIAL.
  - The bridge checks neither the verdict nor the rt_r0 condition. A18 (a) funds it whatever G4a.L2.2 reads, and rt_r0 read LIFT.
- A18 (a) takes `KCj5mptWgBqb/output/em_s13/models/epoch.012.pt` (flag commit b9f676f5). Launched 19:20; pack SLURM 1998206 running since 21:00.

### A14 (i) read (2026-09-24): EM PHI DOES NOT LIFT (audited CONFIRMED_WITH_CORRECTIONS)
- Source: `A14LiftReadJob.GAKp6IJ5pA3l`, pack `PackedBlankfreeTrainJob.e9xZa5ElF16P` (SLURM 1989249, 1 h 55 min). All 8 sub-epochs ran with no NaN. Configs equal L2-0's rt_r70 except the phi path. Extraction: `reports/extract_a14i_read_2026-09-24.md`. Audit: `reports/audit_a14i_hold_2026-09-24.md`, which recomputed PER on all 2,864 dev-other utterances.
- Dev-other greedy PER, ep1 / ep2 / ep4 / ep8:

  | Arm (A10 phi at sub-epoch 48) | PER | ep8 band | ep8 minus cold_ctl [95 % CI] |
  |---|---|---|---|
  | cold_ctl (random phi) | 0.888 / 0.886 / 0.855 / 0.849 | NO LIFT | |
  | durinit s1 | 0.864 / 0.852 / 0.834 / 0.840 | NO LIFT | -0.009 [-0.013, -0.005] |
  | durinit s2 | 0.839 / 0.831 / 0.824 / 0.820 | NO LIFT | -0.029 [-0.032, -0.026] |
  | durfrz s1 | 0.866 / 0.864 / 0.841 / 0.842 | NO LIFT | -0.006 [-0.010, -0.003] |
  | durfrz s2 | 0.845 / 0.829 / 0.820 / 0.821 | NO LIFT | -0.028 [-0.032, -0.025] |

- Reading: no arm leaves the chance band. The s2 arms sit 0.004 above the PARTIAL bar (0.8164), and every arm beats cold_ctl by a small margin with its interval excluding zero. So an EM phi moves a random theta slightly, but far from the 0.18-0.19 a gold or 30-70 %-corrupted phi reaches (L2-0).
- The job reports no generative PER.
- A18 hold rule for (a), applied after (a) was launched:
  - A14 (i) reads NO LIFT on every arm.
  - Comparing the arms' S at 48 with em_s13's S at its final sub-epoch 12: 3.2990 against 3.3810 on the 260 set, +0.082 with paired CI [+0.072, +0.092]. On the 285 set it is 3.3040 against 3.3863. The clause is met. em_s13's 260-set value is the auditor's calculation from per-utterance files; no job prints it.
  - At matched sub-epoch 12, em_s13 is 0.0107 below durinit s1, so the clause fails narrowly.
  - The registered clause assumes the candidate has an S at 48, and em_s13 has none. The rule reads CANNOT_TELL, so it gives no ground for withholding ("really sure it cannot lift").
  - (a) continues. It had started at 21:00, before this read was taken.

### A19 read (2026-09-24): SAME LADDER (audited CONFIRMED)
- Source: `A19TriLadderReadJob.5ny4LPLFrIXw`, pack `PackedBlankfreeTrainJob.DzrmcjOQ4I3r` (SLURM 1995148, 1 h 25 min), and 8 `PairedPerDeltaJob`s. Extraction: `reports/extract_a19_read_2026-09-24.md`. Audit: `reports/audit_a19_read_2026-09-24.md`, which re-scored every arm on all 2,864 dev-other utterances.
- Single delta verified: each tri arm's config differs from its rt twin only in the 17 k2 lines and the output path. The tri logs have no k2 lines, while each rt log has 496. Step 0 is identical, with the same phi fits.
- Dev-other greedy PER, ep1 / ep2 / ep4 / ep8:

  | Arm | PER | ep8 band | tri minus rt, ep1 / ep8 | phi genPER ep8 (direct / Hungarian / NMI) |
  |---|---|---|---|---|
  | tri_r30 | 0.203 / 0.282 / 0.298 / 0.268 | LIFT | +0.017 / +0.085 | 0.260 / 0.260 / 0.788 |
  | tri_r50 | 0.210 / 0.277 / 0.300 / 0.263 | LIFT | +0.020 / +0.077 | 0.257 / 0.257 / 0.784 |
  | tri_r70 | 0.335 / 0.368 / 0.354 / 0.329 | LIFT | +0.081 / +0.138 | 0.316 / 0.357 / 0.725 |
  | tri_r100 | 0.831 / 0.899 / 0.903 / 0.898 | NO LIFT | -0.014 / +0.043 | 0.874 / 0.876 / 0.068 |

- rho*_tri = 0.7, the ladder is monotone and not VOID, and no reading changes the verdict. Seven of the eight paired rows read K2 HELPS; r100 at ep1 reads TRIGRAM ENOUGH.
- Reading: the phone trigram alone lets a phi with up to 70 % of its fit tokens substituted lift a random theta. The k2 term is not needed for the lift, but it improves the ep8 result by 0.08-0.14 PER.
- Without k2, r30 and r50 are best at ep1 (about 0.20), worsen to about 0.30 by ep4, and recover partly to 0.26-0.27 by ep8. This matches D10e's trigram-only drift from p0.
- M is taken from the k2 runs' seed pair. The ep1 labels for r30 and r50 are therefore the least robust (report only).

### A17 (i)/(ii) read (2026-09-24): BASIN SUFFICIENT; OBJECTIVE DRIFT, at the bar (audited CONFIRMED_WITH_CORRECTIONS)
- Source: `A17BasinLiftReadJob.q0C0J9o54cwW` (pack `PackedBlankfreeTrainJob.T18RrTNTdg65`) and `A17AnnealDriftReadJob.jcIFDGis10WJ` (runs `PhiFirstProbeTrainingJob.nVpD2O3xpfcJ`, `.YtsRkvAl7Kl8`). Extraction: `reports/extract_a17_reads_2026-09-24.md`. Audit: `reports/audit_a17_reads_2026-09-24.md`, which re-scored every (i) arm on all 2,864 dev-other utterances.
- (i): the phi inits are A14 (ii)'s sub-epoch-48 checkpoints, and the configs differ from A14 (i) only in the phi path. All 8 sub-epochs ran without NaN. Dev-other greedy PER, ep1 / ep2 / ep4 / ep8:

  | Arm | PER | ep8 band | phi genPER, init -> ep8 (direct / Hungarian) |
  |---|---|---|---|
  | gold-EM | 0.284 / 0.255 / 0.253 / 0.200 | LIFT | 0.353 -> 0.232 / 0.232 |
  | r30-EM | 0.323 / 0.283 / 0.224 / 0.201 | LIFT | 0.394 -> 0.239 / 0.239 |
  | r70-EM | 0.434 / 0.432 / 0.412 / 0.362 | LIFT | 0.495 -> 0.393 / 0.432 |
  | r100-EM | 0.869 / 0.859 / 0.836 / 0.839 | NO LIFT | 0.860 -> 0.858 / 0.858 |

  - **BASIN SUFFICIENT.** r100-EM clears the NO LIFT bar by only 0.0225, but it does not lift. The A10 EM phis read 0.820-0.842 in A14 (i), so where the phi sits decides whether it lifts. Joint training also repairs the basin phis' generative PER, by 0.10-0.16.
  - The basin phis kept their MFA durations, so A17 (iii) and A18 (b)'s paired rows decide whether the lift needs supervised segmentation.
- (ii): tau = 1 from sub-epoch 1, otherwise A14 (ii)'s recipe; PER(0) is the init phi. Generative PER at 0 / 4 / 8 / 12, against A14 (ii) at the same sub-epochs:
  - gold: 0.193 / 0.266 / 0.283 / 0.295 (A14 (ii): 0.193 / 0.300 / 0.317 / 0.328). S on the 260 set: 3.474 / 3.264 / 3.249 / 3.241.
  - r70 (direct / Hungarian): 0.608/0.628, 0.385/0.413, 0.382/0.412, 0.393/0.421 (A14 (ii): 0.608/0.628, 0.477, 0.478, 0.482). S on the 260 set: 4.411 / 3.400 / 3.304 / 3.266.
  - **OBJECTIVE DRIFT**: gold drift +0.1019 against the registered 0.193 (+0.1017 against the measured 0.1933). That is 0.0019 over the 0.10 bar, or 57 phones.
  - The audit's bootstrap over the 500 utterances gives an SE of 0.0024-0.0031. MIXED therefore lies within sampling error, while ANNEALING-DOMINATED is excluded. Direct and Hungarian PER agree for gold.
  - Reading: dropping the tau = 4 sub-epoch removes about a quarter of the gold damage at 12 (0.295 against 0.328). The rest comes at tau = 1 while S falls by 0.23, so S's optimum near gold is not gold.
- Correction: no same-config PER repeat exists for (ii), because the sub-epoch-0 value comes from the same job. The 0.01-0.03 spread cited under A17 is the joint bed's, not phi EM's.
- Consequences as registered:
  - Stage 2 of A16 (b) is not withdrawn.
  - No tau = 1 amendment to stage 2.
  - A18 (d) is not triggered: gold-EM lifts, and the drift is not below 0.05.

### Gold-key control and A17 (iii) read (2026-09-24): GOLD KEY REACHES BASIN; LABELS SUFFICE (audited CONFIRMED_WITH_CORRECTIONS)
- Source: pack `PackedBlankfreeTrainJob.ge1MKcAPmZIV` (4 arms, 48 sub-epochs, no NaN), `KeyInitControlReadJob.JH7zzrX3Egfm`, `A17SegmentationReadJob.lg4fTnb7deS8`, `SegmentationBoundaryJob.g52gIGtUHpkr`. Extraction: `reports/extract_keyinit_reads_2026-09-24.md`. Audit: `reports/audit_keyinit_reads_2026-09-24.md`.
  - The audit recomputed S from the per-utterance genmarg files (260/260 paired), with the same code and A10 files as A14 (ii).
  - The readers and the boundary job each wrote once across the 22:16 manager restart.
- Durations: all four inits hold exactly the general-knowledge durinit values (phones 4.4138, SIL 26.0). No MFA-fitted duration leaks. The gold key's phi (`PhiFromKeyInitJob.f0jaGuiJVe6A`) takes only `key.json` and unsupervised unit frame counts.
- S at 48 (260 set), against the bar 3.28903:
  - gold_key 3.20704 (-0.082): **GOLD KEY REACHES BASIN**;
  - G-dur 3.22387 (-0.065) and r70-dur 3.26568 (-0.023): **LABELS SUFFICE**;
  - r30-dur 3.21151, report only.
- Generative PER on D4 (500 utterances), direct / Hungarian / NMI at 0 / 4 / 12 / 48:

  | Arm | ep0 | ep4 | ep12 | ep48 | A14 (ii) twin at 48 (MFA durations) |
  |---|---|---|---|---|---|
  | gold_key | .327/.385/.740 | .284/.334/.747 | .322/.367/.717 | .346/.391/.695 | |
  | G-dur | .197/.197/.836 | .297/.297/.751 | .324/.324/.730 | .345/.345/.710 | .353/.353/.703 |
  | r30-dur | .248/.248/.803 | .327/.327/.725 | .350/.350/.705 | .368/.368/.685 | .394/.394/.671 |
  | r70-dur | .605/.622/.452 | .430/.453/.609 | .435/.459/.606 | .442/.466/.606 | .495/.508/.566 |

- Boundaries against MFA (20 ms, MFA 11.90 Hz): F1 at 48 is 0.776-0.799 for all four arms, as for their A14 twins (0.777-0.800; A14 r100 0.660). All run at 10.0-10.5 Hz, under-segmenting by 0.12-0.16. At 0, gold_key already has F1 0.813.
- Corrections:
  - The G-dur and r70-dur emission heads still carry the MFA-fitted segment structure (JS 0.21 and 0.16 nats per phone type against the type marginal; G-dur's init F1 is 0.834). So LABELS SUFFICE licenses only "the A14 (ii) basin did not need MFA-fitted durations".
  - The clean no-segmentation evidence is the gold-key control: durinit durations and cell-independent emissions, and a larger margin.
  - Not holding the key arms rests on r70-dur's 0.023 margin, one seed and no seed spread. The A10 restarts span 0.10 in S.
- Reading:
  - A key-to-phi conversion from type-level counts reaches the basin with no supervised segmentation. It ends at generative PER 0.35, about where the MFA-duration gold arm ends.
  - S and PER agree between basins (0.34-0.47 against 0.86 for r100) but not inside one. G-dur's PER rises by 0.148 while S falls, as in A17 (ii).
  - The durinit arms end 0.008-0.053 below their A14 twins in PER, on one seed and descriptive only.
- Consequences as registered: neither the gold-key control nor A17 (iii) holds the stage-2 key arms, which wait only on stage 1. The PREACT_ON caveat is moot, since the control passed.

### A16 (b) stage 1 read (2026-09-24): J's optima are clusters with wrong names; gold is not J's maximiser (audited CONFIRMED_WITH_CORRECTIONS)
- Source: `KeySearchJob.AzM1NoHpOnFJ` (status real ok 196, failed 0; null ok 172, VOID-ON-RATE 8; loop_error null; 15 null runs truncated), `KeySearchSelectJob.g9wsznNnqmyO`, and `KeyAgreementReportJob.Xk3Wxu9kJ38t` (report only, label-using). Extraction: `reports/extract_keysearch_s1_2026-09-24.md`. Audit: `reports/audit_keysearch_s1_2026-09-24.md`.
- Stage 1's J matches stage 0 term by term on the 27 shared keys, so the values are comparable. The printed gold row is copied from stage 0, not recomputed.
- Selected top 4, all cluster starts under the warm schedule: cluster_centroid_s01, cluster_centroid_s04, cluster_context_s01 and cluster_context_s04. Held-out J:
  - selected: -4.5525 / -4.5647 / -4.5671 / -4.5755;
  - references: gold -4.8180; best A10 argmax start -4.9714; K30 -5.6097; best null -5.8543.
  - All 124 real finals beat gold. Every start family reaches -4.55 to -4.75. The real rise over the null is 1.29-1.30.
- Decomposition of the 0.24-0.27 margin over gold: emission 0.20-0.22, trigram 0.01-0.04, duration < 0.01. Over all 124 finals the trigram term ties gold (mean -0.002). The emission term does not depend on symbol names.
- Agreement with the gold key, frame-weighted, identity / many-to-one:

  | Keys | Before search | After search |
  |---|---|---|
  | selected 4 | | 0.07-0.14 / 0.49-0.58 |
  | A10/A11 argmax | 0.110 / 0.486 | 0.10-0.12 / 0.50-0.51 |
  | random | 0.026 / 0.223 | 0.098 / 0.487 |
  | K30 / K70 / K100 | 0.70/0.71, 0.30/0.35, 0/0.22 | |

- The final keys are distinct: an unregistered audit check gives identity about 0.11 between finals, and 0.22 for same-start twins.
- Reading:
  - The search sharpens clusters: many-to-one agreement rises from 0.22 to 0.49 from random starts, above K70's 0.35. It does not find names: identity stays at 0.07-0.14, below K70's 0.30.
  - The selected keys are no better than what the search reaches from random starts. The argmax starts gain 0.33-0.43 in J with no gain in agreement.
  - J prefers these keys over gold through the name-free emission term. The name-dependent trigram term only ties gold.
- Licensed: gold is not J's maximiser, and J's found optima are clusters with wrong names. Not licensed: that no near-gold key beats -4.55, since no search started from gold, r30 or K30; and nothing about stage-2 S or PER.
- Stage 2 runs as registered (the agreement is label-using, report only). A20 asks whether J's trigram term rewards the right names on these partitions.

### A20 read (2026-09-24): NAME-BLIND by the registered rule; J penalises the gold-matching names on the found partitions (audited CONFIRMED_WITH_CORRECTIONS)
- Source: `KeyOracleNameReadJob.QD3S2xcmg0Wt` (login node, 70 s; code 6e6156ca after registration 59121ec0a). Extraction: `reports/extract_a20_oracle_names_2026-09-24.md`. Audit: `reports/audit_a20_oracle_names_2026-09-24.md` (independent J implementation; every row matched to 5 decimals; the in-job same-scale check gives |dJ| = 0 on 22 keys).
- Held-out J (260 set, 137,933 frames, nats per frame):

  | key | J(a) found | J(b) oracle 1:1 | J(b)-J(a) | J(c)-J(a) | identity (a) / (b) / m2o |
  |---|---|---|---|---|---|
  | selected_1 | -4.5525 | -5.1258 | -0.573 | -0.577 | 0.114 / 0.465 / 0.564 |
  | selected_2 | -4.5647 | -5.0006 | -0.436 | -0.474 | 0.143 / 0.473 / 0.579 |
  | selected_3 | -4.5671 | -4.9889 | -0.422 | -0.574 | 0.124 / 0.394 / 0.489 |
  | selected_4 | -4.5755 | -5.0104 | -0.435 | -0.489 | 0.065 / 0.428 / 0.515 |
  | gold | -4.8180 | | | | |

  - J(b)-J(a) sits entirely in the trigram term; emission, duration, absorption and the partition are unchanged (SIL maps to SIL in all four). The A10/A11 argmax finals (7 rows, one more than registered; report only) give -0.39 to -0.62.
  - Found keys fit the trigram better than gold at a matched rate: lm -0.948 at 9.20 Hz against gold's -0.990 at 9.47 Hz.
- Verdict: 0 of 4 above 0.01, so **NAME-BLIND** as registered. Every difference is at least 0.42 below the margin.
- Correction (audit): the registered gloss "the trigram term does not reward the right names" is wrong in direction. The trigram term *penalises* the gold-matching 1:1 names on these partitions by 0.42-0.57, and on the gold partition it penalises scrambled names by 0.69-1.09 (unregistered control), so it does see names. The found keys are swap-optimal (the final sweep over all 780 name pairs moved nothing), and a label-free name-swap climb from (b) moves away from gold (identity 0.39-0.47 to 0.13-0.26).
- Why no 1:1 naming is near right (audit): 10-15 gold phones each hold 2-4 symbols, 14-18 phones are no symbol's majority (B, CH, G, JH, M, OY, P, TH, UH, ZH in all four), and after (b) 15-22 symbols holding 25-47 % of frames carry leftover names.
- Unregistered (audit, train side): the emission term's 0.20-0.23 margin over gold splits into 0.10-0.17 from less d_min absorption (relabelled frames 9.3-10.6 % against gold's 13.6 %) and 0.06-0.10 from higher symbol entropy; H(S)-H(U) alone separates found keys from gold by only 0.03-0.07.
- Licensed: a rename step under J would not help, because J ranks the right names lower. At key level the objective, not the search, rejects gold-agreeing names, and the name-free emission term (mostly absorption) selects partitions that split and merge phones. Not licensed: that the cost work belongs to the name signal alone, since (b) cannot separate a name deficit from a partition deficit; and nothing about S or phi-level renaming (stage 2 and `SAE_4A_rename.md`).

### A18 (a) bridge read, G4a.L2.4 (2026-09-24): LOWER, OBJECTIVE ONLY (audited CONFIRMED_WITH_CORRECTIONS)
- Source: pack `PackedBlankfreeTrainJob.63nbj6Jgiegj` (SLURM 1998206; 4 arms, kept 1/2/4/8, no NaN) and `BridgeReadJob.v6SONiAZL7rH`. cold_ctl is `UdhhxiGIMBob`, which has the same config and code as dec_joint apart from `reverse_checkpoint_path`. Extraction: `reports/extract_a18a_bridge_read_2026-09-24.md`. Audit: `reports/audit_a18a_bridge_read_2026-09-24.md`.
  - The audit recomputed the read from the per-utterance crosseval files (285/285 utterances, 164 speakers).
  - It checked dec_joint's phi against em_s13 `epoch.012.pt`: identical. dec_frz's phi is unchanged at every kept epoch.
- Verdict: dec_joint minus cold_ctl at ep8 is -0.1158 nats per frame, CI [-0.1257, -0.1035]. With B = 0.0093 the margin is 0.01, so the read is **LOWER**. PER stays in the chance band, so this is **OBJECTIVE ONLY**, not CODE BROKEN.
- Terms against cold_ctl:
  - l_tau -0.240, which carries the whole drop; the frozen phi (dec_frz) gives 85 % of it;
  - lexlat_k2 +0.092, so the lexicon term worsens;
  - 3 x rate +0.033.
- Dev-other greedy PER, ep1 / ep2 / ep4 / ep8, recomputed on all 2,864 utterances:
  - dec_joint 0.862 / 0.854 / 0.846 / 0.842;
  - dec_distil 0.854 / 0.853 / 0.847 / 0.846;
  - dec_frz 0.862 / 0.859 / 0.843 / 0.846;
  - dec_joint_s2 0.863 / 0.855 / 0.843 / 0.846;
  - cold_ctl 0.888 / 0.886 / 0.855 / 0.849.
  - At ep8, dec_joint minus cold_ctl is -0.007, the size of the seed spread (0.004).
- em_s13's own generative PER (dec_frz, phi frozen): direct 0.857, Hungarian 0.859, NMI 0.076. That is at the worst edge of the A10 band.
- Correction: cold_ctl lacks A9's durinit, a relaxation made before any result, when A18 was built. Its effect on the drop is unmeasured, but PER alone rules out CODE BROKEN.
- Reading: the label-free wave phi behaves like the A10 phis under joint training (A14 (i)). It lowers the objective through l_tau and carries no phonetic content. The basin phis of A17 (i) reach 0.20 under the same kind of run.

### A18 (b) lift read (2026-09-25): DURINIT BASIN LIFTS (analysis only, label-built inits; audited CONFIRMED_WITH_CORRECTIONS)
- Job `DurinitBasinLiftReadJob.uRzbIh0EfQXG` on pack `ZUZypSQn7qc0`; extraction `reports/extract_a18b_lift_2026-09-25.md`; audit `reports/audit_a18b_lift_2026-09-25.md`. The auditor recomputed PER from each posteriors.hdf over all 2864 utterances, and the values match. Each arm starts from its own keyinit `ge1MKcAPmZIV` epoch.048 phi; the arms differ only in phi path and model dir.
- Dev-other greedy PER at ep1/2/4/8 (ep8 class):
  - gold-key 0.275/0.258/0.235/0.208 (LIFT)
  - G-dur 0.275/0.237/0.232/0.193 (LIFT)
  - r30-dur 0.290/0.264/0.228/0.206 (LIFT)
  - r70-dur 0.355/0.345/0.278/0.228 (LIFT)
- Jointly trained phi generative PER at ep8 (direct/Hungarian/NMI):
  - gold-key 0.258/0.310/0.771
  - G-dur 0.221/0.221/0.810
  - r30-dur 0.243/0.243/0.787
  - r70-dur 0.284/0.335/0.741
  - The init phis (ep0) read 0.346, 0.345, 0.368 and 0.442 direct.
- Paired rows at ep8, PER(durinit arm) - PER(A17 (i) MFA arm), 95 % speaker-clustered CI:
  - G-dur vs gold-EM: -0.0073 [-0.0093, -0.0054]
  - r30-dur vs r30-EM: +0.0051 [+0.0034, +0.0070]
  - r70-dur vs r70-EM: -0.1344 [-0.1397, -0.1295]
  - The first two have opposite signs and sit below the bed's seed spread (0.004-0.010), so they are ties. They show neither help nor harm from MFA durations. The r70 pair differs in more than durations: different EM endpoints, init genPER 0.442 against 0.495, one seed each.
- A18 (c) hold rule: it does not fire, because the gold-key arm reads LIFT. BRIDGE_KEYARMS stays True.
- Corrections (audit):
  - The ep8 forwards of gold-key (`e9LrruCo4AdR`) and r70-dur (`9AUpBk28Ak7F`) ran on the faulty jpbo-028-30 GPU1, seconds after the fault that broke the r30 genmarg decode. No corruption was found, but they are not bit-verified. The verdict does not depend on them, since every arm sits 0.27-0.31 under the LIFT bar. The r70 paired row rests on `9AUpBk28Ak7F`.
  - Scope: every init is built from labels. G-dur and r70-dur also carry MFA-fitted segment structure in their emission heads. The read shows that general-knowledge durations suffice once the init is near right. It says nothing about label-free phis, which so far all read NO LIFT (0.820-0.842).

### A16 (b) stage 2 read (2026-09-25): KEY BASIN, on S only (audited CONFIRMED_WITH_CORRECTIONS)

Read job `KeyArmsReadJob.STcxhF0w4kpq` on pack `PackedBlankfreeTrainJob.G0Vzzokj5PQC` (A10 recipe verbatim except
the key init; durinit; one seed per key). Extraction `reports/extract_keyarms_read_2026-09-25.md`; audit
`reports/audit_keyarms_read_2026-09-25.md`, which recomputed S from the per-utterance files (260 of 260 paired, as
in A14 (ii)).
- S at 48 on the 260 set: rank1 3.27275, rank2 3.29550, rank3 3.28529, rank4 3.33540. The bar is 3.289 (S_min
  3.29903, A10 durinit_s01). The best arm, rank1, is below it, so the verdict is **KEY BASIN**. rank1 against S_min:
  -0.026 [-0.037, -0.015].
- Generative PER at 48 (`GenDecodeReportJob`), direct / Hungarian / NMI: rank1 0.858 / 0.861 / 0.077, rank2
  0.847 / 0.798 / 0.099, rank3 0.849 / 0.842 / 0.096, rank4 0.872 / 0.873 / 0.068. This is the chance band, as for A10.
- Name tracking on these arms is AN-5 in `SAE_4A_rename.md`: EM LOCKED, key identity 0.05-0.17 throughout.
- A18 (c): rank1, the S-best arm, goes to L2-2. The bridge is under manager 3019650.
- Corrections (audit):
  - KEY BASIN licenses S only. It licenses neither the phonetic basin (A14 (ii) also needed PER below 0.50), nor
    names, nor lift.
  - rank1's margin (0.016) is below the A10 same-recipe seed spread of S (0.025-0.071), with one seed per key. The
    four arms average 3.297, above the bar, so only the best of four clears it. rank3's 0.004 is within jitter
    (below the bar only at sub-epochs 40, 45 and 48).
  - log.run.1 is inside finished.tar.gz; it has no warnings.

### A18 (c) bridge read, G4a.L2.4 (2026-09-25): LOWER, OBJECTIVE ONLY; the key arm does not lift (audited CONFIRMED_WITH_CORRECTIONS)
- Source: pack `PackedBlankfreeTrainJob.SCtv4DjzFQ50` (SLURM 2007225) and `BridgeReadJob.HWuOTb0Tefo1`. Extraction
  `reports/extract_l22_keyarm_bridge_2026-09-25.md`; audit `reports/audit_l22_keyarm_bridge_2026-09-25.md` (recomputed from
  the five per_utterance.json files and from the posteriors on all 2864 utterances; every number matches).
- The bridged phi is rank1 `G0Vzzokj5PQC` epoch.048, verified identical. dec_frz's phi is unchanged at every kept epoch.
  cold_ctl is A18 (a)'s own run `UdhhxiGIMBob`, which differs from dec_joint only in reverse_checkpoint_path.
- Verdict: dec_joint minus cold_ctl at ep8 is -0.1609 nats per frame, CI [-0.1710, -0.1488], 285 utterances, 164
  speakers. B = 0.0038, so the margin is 0.01 and the read is **LOWER**. The read is also LOWER under bootstrap seed 7
  and frame weighting. PER stays in the chance band, so it is **OBJECTIVE ONLY**, not CODE BROKEN.
- The whole drop is l_tau (-0.284), and the frozen phi alone gives 97 % of it. lexlat_k2 (+0.081) and 3 x rate (+0.042)
  get worse. The other arms, reported only: dec_distil -0.165, dec_frz -0.106, dec_joint_s2 -0.165, all LOWER, OBJECTIVE
  ONLY.
- Dev-other greedy PER, ep1 / ep2 / ep4 / ep8:
  - dec_joint 0.862 / 0.852 / 0.840 / 0.843;
  - dec_distil 0.857 / 0.854 / 0.844 / 0.843;
  - dec_frz 0.861 / 0.857 / 0.850 / 0.846;
  - dec_joint_s2 0.863 / 0.850 / 0.843 / 0.843;
  - cold_ctl 0.888 / 0.886 / 0.855 / 0.849.
  - The lowest of all 20 cells is 0.840. Every arm is NO LIFT at ep8 under A4's bands.
- Lift test (A14 (i) form; the reader prints no verdict, so it is taken from A4's bands): dec_joint 0.843 = **NO LIFT**
  (one phi, two theta seeds). The same-line gold-key arm lifts to 0.208 (A18 (b)).
- The phi's generative PER at ep8 (direct / Hungarian / NMI): dec_joint 0.858 / 0.855 / 0.077, dec_frz 0.858 / 0.861 /
  0.077, dec_joint_s2 0.859 / 0.851 / 0.078. Joint training leaves it at chance.
- Corrections (audit):
  - The baseline relaxes A9, a build choice made before the result: cold_ctl has uniform durations, while rank1 has
    durinit plus EM durations. Their share of the -0.161 is unmeasured.
  - Under A6, LOWER alone licenses no claim.
  - The read licenses not funding this phi, not that the key line cannot lift.
- Reading: rank1 behaves like the A10 and wave phis (A14 (i), A18 (a)). It lowers the objective through l_tau and
  carries no phonetic content. Next steps for phi training are the proposal in `SAE_4A_rename.md`.

### Adjacent-repeat handling in the objective (2026-09-24; code review `reports/review_repeat_handling_2026-09-24.md`, user question)
- No term reads a run of identical frames as several phones, and no path count is inflated.
  - l_tau: the repeat arc keeps the trigram history and adds no prior or segment. A token equal to its predecessor, SIL included, is masked.
  - k2: H has one path per frame string.
  - l_tau and k2 admit the same support: collapsed runs, SIL as an ordinary symbol, and never two equal phones in a row.
- Costs of the blank-free support, small and disclosed:
  - A true repeat can only surface as X SIL X. Dev-other has 989 repeat pairs in 177,275 reference phones (0.56 %, in 772 of 2,864 utterances; 962 cross-word, 6 with a real pause). That sets a PER floor of about 0.56 % without SIL.
  - The trigram was fitted on uncollapsed text (0.265 % repeat tokens). The masked mass is not renormalised: 0.0027 nats per token.
  - The k2 lexicon has 533 of 151,731 words with an internal repeat, and these can never be reached. A cross-word repeat is reachable only through SIL, so k2 pushes such spans toward X SIL X or another parse. The size of that pressure is unmeasured.
  - phi's HSMM allows same-symbol neighbours on its own, but never inside l_tau.
