# SAE 4A -- lexlat v2: phi-first decipherment on the cold line

## State

Watcher command `bash ~/.claude/skills/sis/sis_watch.sh <pid> <config> 600`; re-arm all watchers first after any resume.

LIVE managers: none of this phase yet. The lexlat phase's D14 / D15 / D16 managers are listed in `SAE_4A_lexlat.md` State and stay watched from there.

Standing rulings (2026-09-23): every main-line method is pure unsupervised and GAN-free; GAN-lineage and supervised inits are analysis only, enter no gate and initialise no main-line arm. The lexlat objective, lexicon, LM and both models are unchanged in this phase; it changes the order in which the two models are fitted.

REGISTERED 2026-09-23, before any job: L2-0 (competence ladder, label-using diagnostic), L2-1 (phi-first EM decipherment, pure cold, the escape experiment), L2-2 (bridge into the joint run, funded only on L2-1 SIGNAL). Gate table below, defined before any number. Background: `reports/assessment_reviewer_escape_plan_2026-09-23.md`, `reports/lit_escape_private_code_2026-09-23.md`.

NEXT: design-reviewer on this file (first job of a phase), then implementer for L2-0 (corruption job, competence forward job, pack) and the L2-1 feasibility probe (null recognizer, tau 4 -> 1, shuffled-unit corpus); code-reviewer before each launch. L2-0 and the L2-1.1 probe may launch before the lexlat D14 read; the L2-1 wave waits for L2-1.1 and for D14 (whether a decode-fitted phi anchors p0 fixes the ladder's 0.19 rung).

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
