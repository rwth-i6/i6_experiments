# SAE 4A -- lexlat v2: phi-first decipherment on the cold line

## State

Watcher: `bash ~/.claude/skills/sis/sis_watch.sh <pid> <config> 600`; re-arm first on resume. LIVE:
- 2099342 `config/sae_4a_lexlat_v2_em_table.py`: A11 stage B, 5 four-GPU jobs.
- 4111121 `config/sae_4a_lexlat_v2_em.py`: the wave, 6 four-GPU packs, Slurm 1989235-38/40/41, 1.06 h.
- 4152192 `config/sae_4a_lexlat_v2_a14.py`: A14 (i) as one four-GPU pack, 1989249, 11.5 h; A14 (ii) as one gpupack node, 1989248, 4 h.

Launch record: `reports/exec_a14_wave_launch_2026-09-24.md`. The em_ext, ladder, decphi and D18 graphs are complete.

Submission layer (user: "change now", 2026-09-23): sub-4-GPU tasks are packed 4 per exclusive node by `gpupack_engine.py`; light CPU classes run on the login node. Live-verified: concurrent members get distinct CUDA_VISIBLE_DEVICES, and 541 members ran with rc 0 (`reports/exec_first_pack_verify_2026-09-24.md`). Isolation relies on the variable alone; nvidia-smi shows all 4 GPUs.

Reads 2026-09-24 (Results):
- L2-0: rho*_lift 0.7, G4a.L2.3 CANNOT_TELL, audited.
- A10: wave durinit, 12 sub-epochs. EM phis beat gold on S, but their decodes are at chance.
- A14 and A15 registered.
- D18 (`SAE_4A_lexlat.md`): on one common objective, the frozen-phi arms are worse by 0.11-0.12 despite better PER. So training phi lowers the objective by co-adaptation, the same direction as A10.

Rulings (2026-09-23): pure unsupervised, GAN-free, supervised inits analysis-only; parallelise everything; L2-1 extensible, user: "try hard enough on L2-1 in case the initial round is not successful"; extensions are amendments from A11 on.

BUILT: ladder; L2-1 to A9 (3dc01561); A10 (c49559ce); A11+A12 (e8bf63a8, 93c06db5); A13 (22602b90); wave and A14 (9974004e, tests af912d60; review `reports/review_a14_wave_launch_2026-09-24.md`); A15 (a6392d82, T3 89713d1a; `reports/impl_a15_phibattery_2026-09-24.md`), in launch review (`reports/review_a15_launch_2026-09-24.md`).

NEXT:
1. A15 review returns, then the executor launches `config/sae_4a_lexlat_v2_phibattery.py` (21 battery jobs, gpupack 1 slot; 25 one-GPU decodes). Then the extractor, my analysis, an auditor, and the report to the user (their request: L2-1 against gold and r70 on text preference and raw accuracy).
2. At the wave and A14 wakes, the executor checks. Also check that gpupack members get distinct CUDA_VISIBLE_DEVICES, and where the 53 CPU report jobs route (login expected).
3. A11: the executor reads stage B at the em_table wake; G4a.L2.2 reads after the nulls.
4. The A14 (i) read decides whether an EM phi lifts; A14 (ii) decides objective against search. Audit both before any direction change.

## Objective

Break the cold line's private code without labels and without a GAN. The record: a competent reverse model anchors a recognizer under the joint objective (`SAE_4A_lexlat.md` D10e, audited: gold phi HOLD 0.180 with k2; random-init phi destroys p0 in 57 steps); the cold phi is a contract on the recognizer's own decode (D10b: own decode preferred over gold by 800-1190 nats, gold vs deranged gold separated by 18-79); the objective prefers the phonetic solution (D13 logged read: gold-init lower on every term, l_tau 1.749 vs 1.873, k2 0.247 vs 0.306, agg 0.211 vs 1.264); the joint cold start lets theta fix a length-faithful code before phi carries structure. So the label-free problem is obtaining a competent phi with theta out of the loop. L2-1 fits phi by EM on the unit stream under the existing prior and lexicon, with likelihood-selected restarts (Berg-Kirkpatrick & Klein 2013; Klejch et al. 2022) and a prior-order curriculum; L2-2 brings the recognizer in afterwards. L2-0 measures beforehand how competent phi must be and which label-free statistic tracks that.

## Constraints

- Pure unsupervised, GAN-free (ruling 2026-09-23). Labels appear only in L2-0's ladder construction and in reports; no label enters a selection or a gate.
- Objective, prior `RtzbESkOedsT`, treatment graph `cdcxYJMjiYj5` at rung 1000, both model classes, tau = 2 for the joint runs, D10e pack constants: unchanged from `SAE_4A_lexlat.md`.
- No full joint cold restarts (every cold arm of the campaign sits in the chance band 0.83-0.91 at every seed); no recognizer-neutral (alpha = 0) arm from a cold start (it hands the posterior to an untrained phi, the collapse driver); no neural-refit split-merge search. Count-table repair moves on phi are the only split-merge form funded, and only by amendment on L2-1 SIGNAL + BELOW.
- Selection statistics are held-out and label-free; dev-other enters only reports.

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

## L2-2: the bridge into the joint run (funded only on G4a.L2.2 SIGNAL)

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
