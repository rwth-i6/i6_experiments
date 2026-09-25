# Design review: SAE_4A_lexlat_v2 (phi-first decipherment), before any job

Reviewer: design-reviewer, 2026-09-23. Read-only. Object: `SAE_4A_lexlat_v2.md` (73 lines, as updated with the
user's "start everything parallelisable" ruling). Also read: `SAE.md` north star / live reward / queue,
`SAE_ref.md` constraints and bed constants, `SAE_4A_lexlat.md` State, D10b, D10e, D13, D14, D15, D16,
`SAE_4A_objective.md` 4.1-4.2, the reviewer assessment and the literature report, and the code that the
phase's quantities rest on (`sae/emc/lattice.py`, `train_steps/sae_blankfree.py`, `lexlat_k2_train.py`,
`rate_term.py`, `reverse.py`, `blankfree_permute.py`, `blankfree_train_jobs.py`, `crosseval_jobs.py`,
`init_jobs.py`, the D10e config builder). Standing rulings were taken as given and are not reviewed.

Verdict: DONE_WITH_CONCERNS. The phase is the right shape (phi first, likelihood-selected restarts, a
competence ladder, a bridge, nulls before numbers), but six items would make a gate mean something other
than what the phase says, or waste the cohort. Each has a cheap amendment. Nothing here requires a
route change; every fix keeps the objective, lexicon, LM and both model classes.

## Blocking defects (most material first)

### B1. L2-1 stage 2 cannot influence phi; the "pruning sees phi scores" claim is false as the code stands
- `lexlat_k2_train.py:1-30`: `L_lex = log Z_HLG(e / temperature) - log Z_H(e / temperature)` on `e`, the
  recognizer's dense per-frame phone log-probabilities; "GRADIENT. To the recognizer's emissions alone".
  `dense()` (line 456) consumes `log_q` only; `step()` (line 505) takes `log_q`, `feat_lens`, `retained`,
  `keep`; no segment table, no reverse model. D16's registration says the same: "lexlat_k2 and agg depend on
  theta only" (`SAE_4A_lexlat.md` D16 consistency check).
- Under a null recognizer `e` is the constant `-log 40`, so the k2 term is a constant per utterance
  length, contributes nothing to phi's E-step, adds the same constant to every restart's "held-out total
  including the k2 term", and its pruning at max_active 1000 is driven by the HLG's own scores alone, never
  by phi. The phase's sentence "Pruning at max_active 1000 sees prior and phi scores only" (L2-1 Objective)
  is not what the code does.
- Predicted failure: stage 2 spends 4 GPUs x 4 passes and re-selects the stage-1 ranking; G4a.L2.3 is read
  on a phi that never saw the lexicon, while the phase text says it did.
- Amendment: remove stage 2. The word graph enters where theta exists (L2-2). G4a.L2.3 reads the stage-1
  selected phi. Reassign the stage-2 budget to the reference restarts of N1 and the cold control of B5.
  (A phi-visible lexical E-step exists in principle, the round-2 trie DP `lexlat_train.LexlatRuntime`,
  which takes the segment table; it is a different DP that failed E1 on cost and would be a new
  objective, outside this phase's ruling.)

### B2. The phi-first objective has no rate anchor; EM can lengthen segments to cut prior cost
- `train_steps/sae_blankfree.py:119-148`: the rate surrogate is `coef * (g * dp_log_q).sum()`; its gradient
  reaches `log_q` only. `_fd_passes` (`rate_term.py:599-639`) tilts the segment table under `no_grad`;
  its posteriors are constants. The agg term reads `log_q` alone (line 115). With theta null and frozen
  the rate and agg terms have zero gradient; only `-log Z` trains phi.
- `SAE_4A_objective.md` 4.1-4.2: the rate term exists because the lattice alone does not price the
  token count. In L2-1 the token count is set by phi's trainable duration logits (`reverse.py:188`),
  the prior and the path multiplicity. Each token costs about 2-3 nats of `-log P_psi(k | h)`; fewer,
  longer segments (up to D_k = 25, D_sil = 50) with position-bucketed emissions cut that cost at little
  emission cost.
- Predicted failure: every restart drifts to a low token rate, held-out S rises, SIGNAL fires, the
  selected phi's decode is deletion-dominated, and L2-2 inherits a phi that prices 10 Hz output as
  unlikely.
- Amendment: (i) the probe logs the expected non-SIL rate (`blankfree_phone_rate_retained_hz` is already
  marked) and the held-out S per pass; (ii) register a label-free eligibility band on the held-out
  expected rate, the bed's own [5.80, 14.49] Hz (the G4a.9 rate clause, `SAE_4A_lexlat.md:533`); a restart
  outside it is VOID for selection; (iii) if the probe drifts, freeze `dur_logits` at a rate-matched
  label-free init (mean duration 50 / 9.66 frames from the bed's rho) before the wave. Also skip the two
  tilted DP passes when the recognizer is frozen (`sae_blankfree.py:128-134` runs them unconditionally):
  otherwise G4a.L2.1 measures a step three times its real cost.

### B3. L2-0 measures competence for the wrong task
- L2-0 arms: theta = p0 (supervised), read HOLD / COLLAPSE = "does phi preserve a phonetic theta". L2-2
  `dec_joint`: theta at the cold packs' random init = "does phi lift a random theta". These are different
  axes. A content-free fitted phi HOLDs p0 because it exerts no pull; it cannot lift a random theta. The
  phase's `_r100` HOLD branch ("then G4a.L2.3 is moot and L2-2 needs only a fitted phi") is therefore a
  wrong inference by construction. No arm on the record pairs a competent phi with a random theta
  (D10e, D14 use p0; the assessment's D16b-theta uses the trained cold theta).
- Predicted failure: rho* and the bar are calibrated on preservation; G4a.L2.3 CLEARS while L2-2 does not
  lift; or the gold phi itself cannot lift a random theta under this objective, in which case the bridge
  is untestable and every L2-1 node is wasted.
- Amendment: run the ladder at the bridge's condition: theta = cold random init, L2-2 constants (tau 2
  held, k2 on-set 1 ramp 3, 8 sub-epochs, kept 1/2/4/8), phi_rho for rho in {0 (gold), 0.3, 0.5, 1.0}
  (four arms, one node; 0.7 or the 0.19 decphi rung in a fifth slot if one frees). Read LIFT = dev-other
  greedy PER < 0.50 at ep8 (G4a.9's standing bar), PARTIAL = below the chance band's 0.83 by more than
  0.0136 (the G4a.9 margin), else NO LIFT; rho* = the largest rho that LIFTs with every smaller rung
  lifting (non-monotone: CANNOT_TELL). The rho = 0 arm (gold phi + random theta) is the cheapest
  falsifier of the whole phase: NO LIFT there means the bridge cannot work and L2-1 is moot. It is a
  disclosed supervised-init analysis arm (no gate, initialises no main-line arm), as D10e is. Keep the
  p0-HOLD ladder only if a spare node exists; it answers the user's "anchor" question, not the bridge's.

### B4. The competence statistics cannot separate a sharp wrong code from competence; the ladder lacks that rung
- Statistic (b) (own posterior decode minus deranged own decode) is a sharpness / self-consistency
  measure. The cold private-code phi already reads 3.04-3.61 nats per frame on exactly this contrast
  (dev-other ep20, `SAE_4A_lexlat.md:533`) against the gold phi's 3.9-4.4 (D10e, gold minus deranged
  gold); `SAE_4A.md:984` recorded "the gap clause measures theta/phi self-consistency on a private code,
  not phone content". The iid-substitution ladder smears emissions and is monotone in rho on (b) by
  construction, so a bar set at rho* on (b) is cleared by any sharp code, unit-tracking included, which
  is what EM under this objective may find. Statistic (a) is untested on such a phi.
- Predicted failure: G4a.L2.3 CLEARS on (b) for a private-code phi.
- Amendment: add two sharp-wrong-code reference phis to the statistics set before the bar is chosen:
  phi_c = the cold end point's reverse block (`k2lat_20_x60` ep60; it loads strictly per D16; no fit) and
  permphi = the gold-phi recipe fitted on gold strings under one fixed random phone permutation (one
  `BlankfreeSupervisedReverseInitJob`, 4 h GPU budget). Read 3's separating statistic must place both with
  the non-lifting arms; a statistic that only orders the smear ladder is not a competence statistic.
  If neither (a) nor (b) separates them, G4a.L2.3 is CANNOT_TELL, learned for about 2 GPU-hours instead
  of after the wave. Optional pack arm: permphi + random theta (does the lexicon term correct a
  consistent mislabelling; informative either way).

### B5. G4a.L2.4's baseline does not exist at ep8, differs in three constants, and LOWER is the proxy the phase itself calls a warning
- Cold N = 20 arms keep checkpoints 1/4/10/20 (`SAE_ref.md:17`; `SAE_4A_lexlat.md:78`): there is no ep8
  of `k2lat_20`. Pack C is D15's rung-3000, on-set-8, beta-scheduled pack (`SAE_4A_lexlat.md` D15); L2-2
  runs rung 1000, on-set 1, tau 2 held from step 1 (the cold packs anneal 8 -> 2 over four sub-epochs).
  The total names agg "paired per utterance", but agg is batch-level (`crosseval_jobs.py:49-50`; D16
  amendment: the per-utterance sum is l_tau + lexlat_k2 + 3 rate).
- Predicted failure: the read is uncomputable at ep8, or compares objective values at different lam_lex,
  max_active, beta and tau schedule, so LOWER measures the schedule and not the phi init. And LOWER with
  PER in the band is the outcome the cold line has produced twelve times.
- Amendment: add `cold_ctl` (theta random init, phi random init, L2-2 constants verbatim) as the paired
  baseline, so the only delta from `dec_joint` is phi's init. Evaluate every arm and the baseline through
  the D16 eval path (`crosseval_jobs`, per-utterance l_tau + lexlat_k2 + 3 rate, agg as a point contrast)
  at one setting on one set (the 285-utterance CV holdout for the gate; dev-other for the report); keep
  C20 / C60 as descriptive references. Put the plain PER clause in the gate as G4a.9 does (`SAE.md` label
  quarantine allows "gate measurements on dev"): CODE BROKEN = dev-other greedy PER < 0.50 at a kept epoch
  AND LOWER; LOWER alone = OBJECTIVE ONLY, no claim. Amend the phase's "no label enters a selection or a
  gate" to "no label enters selection", as the campaign's gates already read.

### B6. The wave's optimisation budget is inherited from the pack's sub-epoch, not derived from what EM needs
- One sub-epoch = 57 updates (`SAE_4A_lexlat.md:559`); 4 passes = 228 updates. The phase states no
  learning rate for phi; the pack's list is theta's (peak 1e-4, ep1 at 1e-5, `blankfree_budget_jobs.py:114`,
  `SAE_4A_lexlat.md:559`). The only phi-alone precedent, the supervised reverse fit, runs lr 3e-3 for about
  2,800 updates (`reverse.py:96-98`, 8 epochs of 2.8k utterances at batch 8). BK&K ran 200 exact EM
  iterations; Klejch ran Baum-Welch to convergence per restart.
- Predicted failure: sixteen restarts that have not left their inits; the restart spread is init noise;
  selection is meaningless; the 1.5 h bar passes.
- Also: "one fixed sub-epoch repeated for 4 passes" is not what the bed's `partition_epoch = 4` with
  `laplace:.1000` gives (`emc_train_jobs.py:254, 1072`; `blankfree_train_jobs.py:244`): the quarter changes
  each full epoch. A fixed subset needs its own segment file with partition 1.
- Amendment: state phi's optimiser (lr, betas, steps) with its source. Make G4a.L2.1 a convergence probe
  as well as a cost probe: two restarts (seeds 1, 2), held-out S per pass; the wave launches only when the
  pass-to-pass gain at the last pass is below the margin, or the pass budget is re-derived. If more passes
  are needed, use the four quarters of the bed (one full pass) rather than one quarter four times: more
  evidence at the same cost.

## Non-blocking (improvements and text corrections)

- N1. G4a.L2.2's null cannot fail for a content-related reason. The within-utterance unit permutation
  (E4's `permute_frames_seed`, `blankfree_permute.py`) destroys run and segment structure; any segmental
  model, the private code included (NMI(symbol, unit) 0.36; derangement gap above 3 nats per frame), beats
  it by O(1) nat per frame. SIGNAL is expected regardless of phonetic content; the FAIL clause ("the
  phi-first route closes") is unreachable. The spec does not say which held-out stream the nulls are
  scored on: with `permute_frames_seed`, RETURNN's CV pass sees the permutation (E4's convention), so the
  nulls are scored on the permuted holdout; state that, and report them on the real holdout too. Write
  into the gate that SIGNAL is a spend gate expected for any segmental fit; route closure sits with
  G4a.L2.4 (amended) and with a label-free reference scale: add restarts initialised at phi_c (two seeds:
  the private-code reference; if the best random restart does not beat them by more than |S_a - S_b|, EM
  found nothing the cold line lacked) and one exact rerun of seed 1 (the identity band of S).
- N2. Order statistics. Best-of-16 against best-of-4 with the 4-null range as margin has a false-positive
  rate of about 0.20 under exchangeability (1e5 simulated draws: 16 vs 4 = 0.197; 4 vs 4 = 0.089; 16 vs 16
  = 0.001). Immaterial against the shuffled null (the effect is O(1) nat per frame) but material for the
  phi_c reference and for the count-table amendment: the shuffled nulls will agree closely, so "rises by
  more than the null spread" accepts every move. Margins = max(null range, identity band, 0.01 nats per
  frame); count-table moves accepted on the selection holdout need confirmation on a second disjoint
  holdout (the CV set is 285 utterances, the only one).
- N3. Statistic (a). With uniform log_q, -log Z_1 = T log 40 - log sum_y N(y) P_psi(y) p_phi(z | y): the
  E-step is exact for the prior P'(y) proportional to N(y) P_psi(y), N the path multiplicity of
  `SAE_4A_objective.md` 4.1 (about 0.01 nats per frame here). The constant cancels across phis on a fixed
  set. The DP at temperature 2 returns F_2 / 2 (`lattice.py` docstring "L_tau = F_tau / tau"); state which
  quantity is banked. Statistic (a) also rewards a good density model of z; a phi that ignores y can score
  well (the `_r100` reference is there for that).
- N4. tau schedule. `temperature_schedule` is a per-sub-epoch list, last entry held (definitions
  `sae_blankfree.py:229, 299`); "linearly over passes 1-2" is [4, 1, 1, 1], a step. State the list. With
  q absent, tau = 1 is the exact E-step under P'; the objection does not apply. Agreed.
- N5. Seeds and the null recognizer. The flat checkpoint carries theta only
  (`blankfree_train_jobs.py:170`); phi is built by `reset_parameters()` under the engine's `random_seed`,
  so seeds 1-16 do vary phi's init. `freeze_recognizer` exists on the base class (`sae_emc.py:254`). The
  code reviewer should assert that the sixteen initial reverse state dicts differ and that the null
  recognizer's log-probs are exactly -log 40 on every frame.
- N6. Spec numbers. The CV holdout has 285 utterances (`init_jobs.py:137`, cv.segments); L2-0's "500
  utterances each" cannot be met on it. D14 dependence: only the 0.19 rung and Read 1's M_w; no L2-1 or
  L2-2 gate depends on D14. The wave may launch on G4a.L2.1 (amended: convergence and rate reads) without
  D14, but not before B2 and B6 are read; G4a.L2.3 waits for the amended L2-0; L2-2 waits for L2-0's
  rho = 0 arm reading LIFT (B3). That is the one ordering the parallel plan must keep.
- N7. `dec_distil`: one sub-epoch of CE (57 steps) at the pack's ep1 lr 1e-5 will barely move theta;
  state lr and steps for the distillation, traced to a source.

## What is already known that changes the baseline
- D10e (audited): a gold phi HOLDs p0; the untrained phi drove the collapse. This is preservation, not
  lifting (B3).
- D10b and `SAE_4A.md:984`: the derangement / own-decode gap is a self-consistency statistic that a
  private code maximises (B4).
- D13 / D16 machinery: per-utterance objective terms at matched settings exist (`crosseval_jobs`); the
  gate should use it rather than training-log columns from packs at different constants (B5).
- The k2 term's gradient path is theta-only by construction and test (`test_lexlat_k2_train`) (B1).

## Cheapest check
Run, beside the amended G4a.L2.1 probe (two restarts x 4 passes, per-pass held-out S and expected rate),
one GPU for about 1.8 h: the rho = 0 arm of the amended ladder, gold phi + random theta at the L2-2
constants. NO LIFT there falsifies the bridge before any L2-1 node runs; LIFT fixes the bar's task.
