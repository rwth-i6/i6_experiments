# SAE 4A -- lexlat v2: phi-first decipherment on the cold line

## State

Watcher command `bash ~/.claude/skills/sis/sis_watch.sh <pid> <config> 600`; re-arm all watchers first after any resume.

LIVE managers: none of this phase yet. The lexlat phase's D14 / D15 / D16 managers are listed in `SAE_4A_lexlat.md` State and stay watched from there.

Standing rulings (2026-09-23): every main-line method is pure unsupervised and GAN-free; GAN-lineage and supervised inits are analysis only, enter no gate and initialise no main-line arm. The lexlat objective, lexicon, LM and both models are unchanged in this phase; it changes the order in which the two models are fitted.

REGISTERED 2026-09-23, before any job: L2-0 (competence ladder, label-using diagnostic), L2-1 (phi-first EM decipherment, pure cold, the escape experiment), L2-2 (bridge into the joint run, funded only on L2-1 SIGNAL). Gate table below, defined before any number. Background: `reports/assessment_reviewer_escape_plan_2026-09-23.md`, `reports/lit_escape_private_code_2026-09-23.md`.

User ruling 2026-09-23: start everything parallelisable; the step order above need not be followed. So the L2-1 wave launches on the G4a.L2.1 read without waiting for D14 (D14 only fills L2-0's 0.19 rung); design review runs alongside implementation, before any launch.

Design review done; amendments A1-A7 applied below (before any job). Nothing launched yet.

IN FLIGHT (implementers): L2-0 p0 ladder built and code-reviewed clean (commit 10880dbe, pack WX41NC734WLo, `reports/review_l20_ladder_2026-09-23.md`); its extension to the random-theta nodes R1 / R2, permphi and phi_c -> `reports/impl_l20_ladder_2026-09-23.md`; L2-1 null-recognizer EM, probe, wave -> `reports/impl_l21_phifirst_2026-09-23.md`; genmarg eval / decode / selection jobs -> `reports/impl_l2_genmarg_2026-09-23.md`.

NEXT: no ladder launch until the L2-1 edits to `sae_blankfree.py` are committed (the packs import the live checkout). Then wire genmarg into both configs, code-review, launch L2-0 (fits, nodes P / R1 / R2) and the probe; the wave launches on amended G4a.L2.1, held if both rt_r0 seeds have read NO LIFT by then.

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
- **A7 (N1, N2) G4a.L2.2 is a spend gate.** Any segmental fit beats the within-utterance shuffle, so NO SIGNAL means L2-2 is not funded, not that the route fails. Margin = max(null range, identity band, 0.01 nats per frame); identity band = the larger |S| difference of seeds 1 and 2 against their exact reruns. Nulls are gated on their permuted holdout (E4 convention), real holdout reported. Wave = 16 restarts + 4 nulls + 2 phi_c-initialised restarts + reruns of seeds 1 and 2 (6 nodes). Report only: selected restart minus best phi_c restart beyond the margin = BEYOND PRIVATE CODE, else NOT BEYOND. L2-2 is funded on SIGNAL AND at least one rt_r0 seed reading LIFT. Every restart VOID on rate = NO ELIGIBLE RESTART: G4a.L2.2 reads CANNOT_TELL, and the wave reruns once under A2's duration freeze; if the freeze was already applied, NO SIGNAL. Rate is pooled per set as 50 x (sum of expected non-SIL tokens) / (sum of unit frames). The random init sits at 3.7 Hz (uniform durations), below the band.
- **N7.** dec_distil's lr and steps are stated, with a source, before L2-2 is built.

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

None yet.
