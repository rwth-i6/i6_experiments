# SAE_i6 P1 — corrupted-phi lift ladder: reproduce rt_r70, extend to rt_r80 and rt_r90

## State

OPEN (2026-09-25 14:45, user request). No job yet. The phase file and gates are registered; the design review
comes next, then the implementer (entry point, memory probe), then code review, then the phi fits and the probe.
NEXT: design-reviewer on this file; then the implementer brief, after the P0 G0.G implementer releases
`reverse_model/ladder.py` (one writer per file).

## Objective

JUPITER's L2-0 node R showed that a supervised phi fitted to corrupted gold strings lifts a cold theta to near-gold
PER by sub-epoch 8. It LIFTs up to rho = 0.7 and does NOT LIFT at 1.0 (rho*_lift = 0.7; `SAE_i6_ref_lexlat_v2.md`,
"L2-0 ladder"). Where the boundary lies inside (0.7, 1.0) is unknown. P1 reruns rt_r70 on i6 to check that the
regime reproduces here, then runs rt_r80 and rt_r90 to place the boundary to 0.1.

This is a disclosed LABEL-USING diagnostic (the phis are fitted to seed gold strings): analysis only. It is never
cold-start progress, a route or a fallback (`SAE_i6_ref.md` section 2).

## Design (JUPITER node R exactly; `reverse_model/ladder.py` docstring; source `SAE_4A_lexlat_v2.md` L2-0)

- phi_rho: the gold phi's fit (`supervised.blankfree_supervised_reverse_init`, final epoch 8) on the 10 h seed's gold
  strings, with exactly round(rho n_u) tokens per utterance substituted.
  - Each substitute is drawn from the seed gold unigram without the original symbol; `CorruptSeedGoldJob`, seed 0.
  - The draw depends on seed and tag only, so the ladder is nested: r70's substitutions are the first ones of r80's,
    and r80's the first ones of r90's.
- Arms rt_r70, rt_r80, rt_r90:
  - D10e's `supphi_k2lat` with tau held at 2.0 and N = 20 learning rates truncated to 8 sub-epochs, kept 1/2/4/8;
  - the k2 block at rung 1000, on-set 1, ramp 3, lam 1; both models trainable;
  - theta at the cold flat init (`FlatRecognizerInitJob`, seed 0); phi from phi_rho; `lexlat_k2_chunk_seqs` 4.
  - rt_r70 is JUPITER's arm unchanged. rt_r80 and rt_r90 differ from it only in rho.
- Reads: dev-other greedy PER at ep1/2/4/8 per arm; the corruption report of each fit (`corruption.json`).
- Known deviations inherited from P0 (Deviations there): the i6 prior and i6 phone text, the i6 HLG (182,215 words,
  not 151,731), the audio label, i6 hardware, and the i6 gold phi recipe refit (G0.R3).

## Gates (pre-registered 2026-09-25, before any job)

- **G1.F phi fits (A).** The realised substitution rate of r70 / r80 / r90 on all seed utterances is 0.700 / 0.800 /
  0.900 +-0.005. The nesting holds. Each fit ends at epoch 8 without error. Its final dev NLL is report-only (B).
- **G1.M memory and time (report; a deviation if anything moves).** Record peak allocated and reserved memory over
  the first sub-epoch's steps 0-3, where near-uniform lattices peak, and the step time. A change of batch shape,
  or of anything that moves a score or gradient, is a new operating point: it needs the design review and voids
  the G1.R70 comparison. `lexlat_k2_chunk_seqs` below 4 is a launch granularity (k2 prunes per sequence) and moves
  nothing.
- **G1.R70 reproduction (A).** rt_r70's dev-other greedy PER at ep8 reads LIFT (< 0.50) and lies within 0.1909
  +-0.02. (B): ep1 0.2533 +-0.03.
- **G1.L boundary (the question; read only if G1.R70 passes).** Class at ep8 by JUPITER's A4 bands: LIFT < 0.50,
  PARTIAL < 0.8164, NO LIFT otherwise. rho*_lift on i6 is the largest rho in {0.7, 0.8, 0.9} that LIFTs with every
  smaller one lifting. r30 and r50 LIFT on JUPITER and are not rerun here; r100 NO LIFTs on JUPITER.
  - r80 not LIFT: boundary in (0.7, 0.8].
  - r80 LIFT, r90 not: boundary in (0.8, 0.9].
  - Both LIFT: boundary in (0.9, 1.0], resting on JUPITER's r100.
  - PARTIAL at ep8 is reported with its ep1-8 trajectory. It is not read as LIFT.
  - If G1.R70 fails, the r80 / r90 classes are reported but no boundary is claimed; the miss goes to the debugger.
- **Caveat (audit N1, carried).** The corruption is independent of the acoustics, so each phone's most likely unit
  stays right below rho 1. rho*_lift does not transfer to an EM phi, whose errors are structured.

## Runs

(none yet)

## Deviations from the reference (filled as they are made)

## Results
