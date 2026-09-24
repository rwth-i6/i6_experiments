# Design review -- S3b cold-start remedies (rate term, consistency, back-translation), 2026-09-15

Verdict: DONE_WITH_CONCERNS. Read-only; nothing edited, nothing launched. Every number below was
re-read from the artifact named beside it, not from a report.

Files read: SAE_4A.md (Objective, Design decisions, Gates, S3 result :865-885, S3b :887-989, reads
(a)-(b3) and the two literature sections), SAE.md :17-76, SAE_ref.md (grep only),
reports/{review_rate_term, review_bt_probe, impl_consistency, audit_cold_reads}_2026-09-15.md,
reports/lit_backtranslation_cold_start_2026-09-15.md (setup-dir reports/), sae/emc/{rate_term.py,
consistency.py, lattice.py docstring, agg.py docstring, bt_probe.py :1-260 and the sampler/render
sites, reverse.py :220-236 and :415-460, s3_jobs.py :237-248}, train_steps/sae_emc.py,
config_sae_4a_s3b_{rate,cons}_v1.py, analysis/out/cold.c5a_*.txt, cold.c1/c2 tables, the S3 job's
learning_rates, and the three finished BtProbeJob jsons.

## 1. Do the remedies address the mechanism the cold reads expose?

### F1 (decisive). The c5 fixed point is already visible in S3's own training monitors, so the rate
term as built has nothing to pull on at the gate point and pushes the wrong way during the anneal.

`ReturnnTrainingJob.sBlPYBA1YcIQ/output/learning_rates`, key `dev_loss_emc_phone_rate` = the
EXPECTED token count under the lattice posterior per audio second (train_steps/sae_emc.py:341-342;
SIL included; CV split), by sub-epoch 1..8:

    expected tokens/s   21.6  20.4  16.9   8.9   9.0   9.2   8.9   9.3
    greedy phones/s      3.71  1.98  0.97  1.58  4.75  6.87  4.29  7.73   (SAE_4A.md:870, dev-other)
    tau                  8     5.04  3.17  2     2     2     2     2

At the gate sub-epoch the expectation was 8.9/s against rho 9.66/s while the mode emitted 1.58/s:
that is c5's "expectation met by a diffuse posterior with a blank argmax", in the training run
itself, at every tau, largest at tau 8. Consequences for `rate_term.rate_loss` (rate_term.py:668-673,
value ((r-rho)/rho)^2, gradient coef x d post_q/d b):
* sub-epochs 1-3: E[N] is ~2x rho (even after removing SIL's share), so the term is active at
  value ~1 x lam with the sign "fewer emissions" -- the direction of the collapse it is meant to
  prevent. It is symmetric; nothing in the spec says so.
* sub-epoch 4: relative error <= ~0.3, value <= 0.1, gradient ~0; and the gradient it does carry is
  spread over 39 phone types x T frames (c5: the tilt moves mass uniformly, argmax sequence
  bit-identical to the untilted run, audit_cold_reads (d)), so it cannot concentrate mass on one
  symbol per frame. The gap between expectation and mode is closed only by phi's soft-EM
  symmetry breaking (S3 sub-epochs 5-8: greedy 4.8-7.7/s, reverse term -4.9 -> -3.3), which the
  hard-refit harness cannot show (audit (a): phi never refit in c2/c3/c5) and which the term does
  not touch (phi gets no gradient from it, rate_term.py:596).
Prediction, falsifiable in one arm: `emc/expected_phone_rate_hz` pinned within ~10 % of rho by
sub-epoch 2-3 in all three arms; DecodeStatsJob `phone_rate` at sub-epoch 4 < 0.6 rho; PER
0.85-0.95; lam 1/3/10 differ only in how fast the expectation is pinned. G4a.3b-R FAIL on the rate
and PER clauses, which must then NOT be read as "rate control does not help take-off".

Caveat on the evidence: emc_phone_rate includes SIL and is measured on the 1 % CV holdout, the
greedy rates on dev-other; the comparison is in kind, not paired. It is still a factor 5 at the
gate sub-epoch and a factor 6-17 during the anneal.

Minimal change (in order of preference; all label-free, hash-neutral by omission):
(a) a mode-pricing companion: per-frame entropy penalty on the UNTEMPERED q, weight ramped 0 ->
    lam_ent over sub-epochs 1-4 with the tau anneal (no DP cost). With the expectation held at rho
    the only sharp solution has ~rho T confident non-blank frames, so greedy rate = expected rate.
    Risk: early lock-in of arbitrary phone identities -- hence the ramp, and hence still no content
    claim (S3 sub-epochs 5-8 already show confident content-free output).
(b) one-sided hinge max(0, rho - r)^2 so the term never pushes emissions down during the anneal
    (half a line; deviates from the user's stated form, disclose).
Not recommended: tau < 2 (changes the objective the campaign is about) or a straight-through /
Gumbel count on the argmax (a new estimator for a term that is a precondition, not the mechanism).
Launch: lam3 alone first as the falsifier (F1's read is available at sub-epoch 1, ~10 min in, and at
sub-epoch 4); hold lam1 / lam10 and the two consistency arms until that read. If F1 is confirmed,
replace lam1 / lam10 by lam3 + companion (two ramps) before the packed launch. If F1 is refuted
(greedy rate in band at sub-epoch 4), launch the grid as specified.

### F2. No S3b arm carries a label-free content-carrying acoustic side, which is the one thing c1
shows take-off needs. audit_cold_reads (b): c1's warm phi descends from the gold-seeded recognizer
(65NNK8Bwxdtd ep24 <- SeedGoldPhonesJob), c4 is oracle on both sides, and the probe's warm_phi arms
inherit it. So c1 licenses "a content-carrying phi lets a flat theta take off network-free", not "a
label-free route to such a phi exists". The rate term (precondition), the consistency term
(regularizer, see 3) and BT-from-cold (content-free at the probe scale, see 4) are all nulls or
preconditions with respect to that mechanism; the literature's remedy (forward per-frame content
term onto low-K k-means codes, Liu 2022, SAE_4A.md:842-845, :860-862) is only the FAIL branch of
G4a.3b-R. Recommend building it now, in parallel, so the FAIL branch does not cost a round trip.

### F3 (coordinator's tau point). Every cold read ran at tau = 2 while S3 collapsed at tau 8 / 5 /
3.17. The S3 monitors above show the expectation/mode gap at every tau, largest at tau 8 (21.6 vs
3.7), so c5's tau = 2 read is the FAVOURABLE regime for a mode read; the anneal regime is worse.
The tau caveat strengthens F1, it does not weaken it.

## 2. Gates G4a.3b-R / G4a.3b-C

Right quantity: greedy PER < 0.50 on full dev-other (2864 utts) separates from the content-free band
0.84-0.90 (S3 sub-epochs 5-8; probe cold arm 0.867-0.886), so the "any arm of 5" rule adds no
false-positive risk; the greedy rate clause is exactly the clause that catches F1 (read off
`{base}/{split}/phone_rate` against [5.797, 14.493], never `rate_in_band`, whose band is the gold
9.8/s, emc_train_jobs.py:201 -- review_rate_term F1). Gap pairing: d69f938 is HEAD of
haotian_modality_matching_jupiter; s3_jobs.py:247-248 keys rows by evaluate's index; the change is
run()-body, so every NEW S3DerangementGapJob runs the fixed code. S3's own gap job (Ez8bGMB7LnPX)
never reruns, so the S3 baseline CI stays void -- immaterial for an absolute gate.
Budget: ~470 steps x ~4.6 s (3 DP calls, rate_dp_calls = 2 expected) ~= 36 min + reads per arm;
decidable. n = 1 seed per arm; acceptable for a take-off gate (effect size 0.4 PER), say so.
Amend before the first read (reading rule, no hash): record `emc/expected_phone_rate_hz` (sub-epoch
4 mean) beside `phone_rate` as the named diagnostic that attributes a FAIL to F1 vs to phi.

## 3. Consistency arms (lam_rate 3 / lam_cons 1 / SpecAugment 19 % frames, 28 % channels)

The pre-registered caveat's guard is void as built: the rate term forbids an empty EXPECTED output,
not an empty decode (F1), and KL(p_clean || p_aug) (consistency.py:318) is ~0 for any diffuse
posterior. Further, a content-free but non-constant output (S3 sub-epochs 5-8: prior-like strings
plus speaker/duration) is consistent under BOTH views, since masking and speed change preserve
speaker and duration. The term prices neither the mode nor content; it is a regularizer whose value
appears once a content signal exists. Magnitude: measured KL 0.12 (specaug) / 0.39 (speed) at a
random model (impl_consistency tests) against L_tau/frame 1.7-2.2 at tau 2 -- lam_cons 1 is not
small. The SpecAugment strength is an implementer default with no trace (consistency.py:46-67,
204-channel masks on 1024 dims). As a single shot it is defensible only with the pre-registered
expectation "no take-off; read = gap and cons_kl per view", and only after the lam3 read, because
the arms share F1's failure mode and would otherwise be read as a second null of the same kind.
The speed warp (linear, align-corners, probability space) is exact for a global speed factor.

## 4. Back-translation

Probe status on disk (3 of 8 arms finished; `work/speech_llm/sae/emc/bt_probe/*/output/bt_probe.json`):

    warm_phi/collage   PER 0.363 -> 0.327 -> 0.324 -> 0.314; rate 7.7-8.5/s; gap +2.88 -> +3.58, CI excl. 0
    warm_phi/centroid  PER 0.695 -> 0.814 -> 0.962 -> 1.058; rate 9.6 -> 14.2/s; gap +1.6 -> +1.2
    phi_0/collage      PER 0.867-0.886; rate 3.7 -> 2.7/s (out of band); gap -0.002/+0.015/+0.009/-0.005,
                       CI covers 0 at rounds 0-2, [-0.056, -0.006] at round 3

Reading: the collage pipeline transfers content from a content-carrying phi (0.314 vs c1's
network-free 0.279), the centroid rendering does not survive rounds, and BT from the cold phi_0 is
content-free with a declining rate -- Artetxe's "ignore the input, learn the target LM" attractor
(lit report 3a). Since the warm phi is oracle-derived, `takes_off = True` on warm arms is a pipeline
check and must be banked as such, not as a take-off. The PER < 0.80 clause is 0.07 from the cold
arm's own 0.87 and S3 reached 0.839 content-free; the gap clause carries the reading, and "in ANY
round" over 8 x 4 = 32 gap reads inflates a nominal 95 % CI (the cold arm's round-3 CI already
excludes 0 on the negative side by 0.006). Register now, before the remaining 5 arms are read:
take-off = the clauses at the FINAL round (or two consecutive rounds), not any round.

What the full S3b-BT design must carry that the probe does not:
1. A label-free content signal that plays the DAE's role (F2): first candidate the forward
   per-frame content term (Liu 2022); second a real-vs-synthetic consistency on theta; the EMC
   term on real audio is the principled coupling of the two directions (phi's soft-EM gradient).
   Without one of these the design's only cold mechanism is BT, predicted null by the literature
   and now by the probe; do not fund that run.
2. Rate control on the MODE (F1): the cold arm's greedy rate fell 3.7 -> 2.7/s.
3. Collage synthesis (speaker-run or frame), not centroid; the renderer and spk_run arms decide
   between the collages when they finish.
4. The trigram constraint: theta's CTC loss has no LM term (the sentence sample carries the full
   LM); phi's fit on pseudo-text has none. State this explicitly; the EMC leg stays on trigram.
5. The budget in GPU-hours at the packed rate: "8 tc100 sub-epoch equivalents" = 8 x 123 s of S3
   training; a BT round at tc100 scale (28.5k decodes, 10-epoch phi fit, 2000 CTC steps) exceeds it
   several-fold -- the probe spent 1378-1734 s per arm on 2000 utts. Name the equivalence rule.
6. The gate read on full dev-other (2864) as S3, index-keyed gap, gold gate, and at least two phi_0
   seeds (the failing side is one draw, seed 42; audit (e).2).

Differentiable path from theta's CTC loss into phi (coordinator's question): NO for the first full
design. What it buys: a joint text -> units -> text gradient, so phi's samples become decodable by
theta. What it risks, given the cold reads: theta's gradient pulls phi away from p(z | y) fit on real
units toward units theta can decode -- the pair closes the cycle by a private code with no
constraint from real audio (the §3g finding; lit report 3c-3d), and the cold arm shows the loop
already has a content-free fixed point to fall into. Expected-feature rendering regresses to the
mean (the phase's own reason for rejecting the direct regressor); Gumbel/ST through two discrete
draws (duration, unit) at K = 500 is high-variance. The principled coupling exists already: L_tau on
real audio trains phi against real units; alternate EMC (theta, phi on real) with BT (theta on
samples). If a differentiable path is ever added, pair it with the real-unit likelihood as a
constraint and read it with the derangement gap plus a private-code control (phi samples for
deranged text decoded by theta).

## 5. Sequencing (one GH200 saturated per arm, 4 per node once packed)

1. Now: lam3 rate arm alone (one exclusive node for ~1 h is the cheapest falsifier of F1).
2. Meanwhile: the remaining 5 probe arms finish; register the final-round reading rule first.
3. After the lam3 sub-epoch-4 read, one packed node: {lam1, lam10} or {lam3 + companion ramps} per
   F1, plus cons spec and cons spec_speed with the "no take-off expected" pre-registration.
4. S3b-BT full training only with item 4.1 in the design; no warm-phi (oracle) arm at full scale;
   no centroid mode.
5. Build the forward content term in parallel now (F2); it is the next lever on every branch.

## Cheapest check

Launch lam3 alone; at sub-epoch 1 (~10 min) read `emc/expected_phone_rate_hz` from its
learning_rates against DecodeStatsJob `phone_rate` on dev-other: expectation within ~20 % of 9.66/s
with greedy < 3/s confirms c5-in-training and F1 before any other S3b arm is funded.
