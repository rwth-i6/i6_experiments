# Design review: SAE 4A InfoMax (entropy penalty + augmentation invariance), 2026-09-20

Verdict: APPROVE_WITH_AMENDMENTS. Read-only review before the first job. Files read: SAE_4A_infomax.md
(whole), SAE_4A_objective.md (whole), SAE_4A_budget.md:33-126, emc/consistency.py:1-110, emc/lattice.py:1-40,
emc/agg.py:1-40; bounded greps beyond the brief (disclosed): train_steps/sae_blankfree.py:60-112,
train_steps/sae_emc.py:376-410, emc/entropy_term.py:1-60,100-160, SAE_4A.md lines matching "S3b-R" (940-984),
reports/extract_nodeA_ep1_2026-09-20.md (no entropy column), config tree for `lam_ent` (never set outside defaults).

The design CAN answer the campaign question: a jump from the 0.83-0.91 band to PER < 0.50 is not producible by
seed noise, so n = 1 suffices for the gate. What is not yet sound is (1) the entropy read, which has no baseline,
(2) the lambda sizing, which is wrong in two places, (3) the decayed schedule's confound with the tau anneal,
(4) the statswap view's numerics on short utterances, (5) a missing read for the failure mode the S3b consistency
arms actually showed (inventory collapse with rate in band).

## Findings, most material first

F1. The entropy read has no baseline (SAE_4A_infomax.md:69 "near 3 in the band"; :109-112 early read
"entropy < 1.0 nat"). ctrl_50's per-output-frame entropy is not banked anywhere (extract_nodeA_ep1 has no such
column). The tau 8 -> 2 anneal alone sharpens q: entropy_term.py's docstring records that the S3 arms are
"confident and content-free" at tau = 2 from sub-epoch 5. So ctrl_50 may already sit below 1.0 nat at sub-epoch
10 and the early read's entropy clause, the prediction "leaves 2.5-3 nats", and the sizing paragraph are all
unmeasured. Failure predicted: "symmetry breaking" declared on an entropy drop the anneal produced.
Amendment 1: before the ep10 read (ideally before launch; node_a is at sub-epoch 19, so ctrl_50 ep1/ep4/ep10 exist
now), bank ctrl_50's eval-mode dev-other mean per-output-frame entropy in nats at ep1/4/10 through a registered
read, and set the early-read clause as "entropy below ctrl_50's at the same sub-epoch by > 0.5 nat AND below 1.0".

F2. Sizing (SAE_4A_infomax.md:67-73) is wrong twice. (a) The per-logit entropy gradient q_k |log q_k + H| is
0.02-0.1 at a diffuse 40-way posterior (q_k ~ 0.025, |log q_k + H| ~ 0.7), not "order 1"; order 1 is reached only
near a two-way peaked posterior. (b) Denominators differ: sae_blankfree.py:88
`cycle = ((loss_utt / retained) * keep).sum() / n_keep` is a per-UTTERANCE mean of a per-UNIT-frame (50 Hz) loss;
the spec's `ent` is a pooled mean over OUTPUT frames (stride 3, asserted sae_blankfree.py:60). Per logit the ent
term therefore carries about 3 * lambda * (S_i / S_mean) relative to the lattice: effective lambda ~0.3, larger on
long utterances. (c) A registered entropy term already exists (emc/entropy_term.py, `entropy_penalty`): units are
H / ln C (dimensionless), ramp 0 -> lam over 4 sub-epochs, wired only into train_steps/sae_emc.py:407 (S3 bed),
never run in any config. If the implementer reuses it, lambda 0.1 means 0.027 nats and the schedule direction
is inverted. Failure predicted: an unintended operating point, undetectable from the loss column name.
Amendment 2: pin in the implementer brief and phase file: `ent` in NATS per output frame (no / ln C), normalized
like the lattice (per-utterance sum of H_t divided by the utterance's `retained` unit frames, mean over utterances)
so lambda is in the lattice's units; if the pooled output-frame mean is kept instead, write "effective per-logit
weight 3 x lambda" into the phase file. Report `emc/entropy_per_frame` in nats as as_error in all four arms.

F3. Schedule confound (SAE_4A_infomax.md:56, :58). The decayed arms' penalty (geometric 0.1 -> 0.001 over 1..10,
< 0.013 from sub-epoch 5) lives entirely inside the tau 8 -> 2 anneal, during which the lattice targets
post^(1/7) and pushes q FLAT (SAE_4A_objective.md:93-96); the 2026-09-15 design review's remedy (a), quoted in
entropy_term.py, warned that full entropy pressure at tau = 8 locks in arbitrary identities. The decayed schedule
applies maximum lambda at maximum tau and withdraws it exactly when the lattice reaches the temperature at which
the penalty could act. Failure predicted: ent_50 / entaug_50 track ctrl_50 from sub-epoch ~5; the pre-registered
prediction "entropy < 1 within 10 sub-epochs in all four arms" fails for reasons unrelated to the mechanism, and
"band survives once withdrawn" cannot be separated from "anneal completed". The window is the user's; keep it but
Amendment 3: rewrite the prediction for the decayed arms as "entropy tracks ctrl_50 after sub-epoch 5; any
sustained gap at ep25/ep50 is hysteresis" and add to the phase file that the decayed arms test the penalty at
tau in [8, 2.2] only. Offer the user a decay window 11..20 (after the anneal) as the version that tests the
penalty at the principled temperature.

F4. Stationary-point argument. Correct in substance, but (a) it is NOT in SAE_4A_objective.md sections 3/5 as
SAE_4A_infomax.md:15 claims: those sections discuss temperature and looseness, and section 5 item 3 argues AGAINST
an explicit entropy term with the bound's sign (an entropy BONUS). The phase adds the OPPOSITE sign (+lambda H,
minimized), a mode-seeking pressure of the same family as tau < 2, which the note ranks as its item-1 objection.
State it as a deliberate departure from the bound (IMSAT), not a consequence. (b) The argument itself: with
p_phi(x | y) independent of y, post(y | x) = P_Y(y) up to a length-only factor, the tau-2 lattice's theta-optimum
is input-independent (frame-factorized q then only needs t and T), and phi's MLE on (x, y) with y independent of x
is the y-independent model; hence a joint fixed point. Hole: at that exact point the entropy gradient
-q_k (log q_k + H) is the SAME at every frame, so it pushes toward the constant output, not out of the symmetry;
the symmetry breaker is the actual recognizer's residual input dependence (gap > 0), which the penalty amplifies.
(c) Joint dynamics: once theta sharpens on any partition, phi fits it and the lattice becomes self-consistent
there (SAE_4A.md:984: "the gap tracks the sharpening of the private code"), so withdrawal will not restore the
band; hysteresis in ent_50 is expected by construction, not a finding. (d) Minor: sum_t H(q_t) is the PATH
entropy and also prices alignment boundaries, which the lattice sums out.
Amendment 4: write (b) in three lines into the phase file's Objective and remove the attribution to the note;
add "the entropy penalty is a deliberate mode-seeking departure from the bound".

F5. What the invariance arms can and cannot remove. Statswap is a per-utterance per-dimension affine map: every
partition on WITHIN-utterance relative features (energy, spectral tilt, the length-faithful private code S3b
found) survives it exactly, and specaug does not forbid it either. The arms remove only utterance-level
(speaker/channel) partitions. Prior evidence the phase does not cite: the S3b-C consistency arms collapsed onto
3-4 symbols (spec: K/IH/T = 57 % of tokens, SAE_4A.md:984) while the rate arms spread over the inventory at
4.3-4.5 bits (gold 4.8). Consistency + entropy is the standard route to a low-inventory confident partition, and
the abort rule (entropy < 0.05 AND rate outside [4, 20]) does not catch it when the rate stays in band.
Amendment 5: add the greedy output's symbol-usage entropy (bits over the 40 outputs; emc_hyp_inspect already
computes it) to the registered reads at every kept checkpoint in all four arms and ctrl_50, and pre-register
"below 3.0 bits with rate in band reads low-inventory confident partition (FAIL-type), never symmetry breaking".
Report the consistency KL per view (specaug, statswap) as separate as_error columns so a dead view is visible.

F6. Statswap numerics and content (SAE_4A_infomax.md:81-86). Per-dimension sd over a short utterance is noisy and
can be ~0 for near-constant dimensions, so (x - mu_i) / sd_i * sd_j explodes; for a short utterance the mean IS the
content, so the swap injects the partner's content. Laplace length-sorting makes partners similar in length
(good) but they may share a speaker (swap near identity; disclose).
Amendment 6: floor sd_i at 1e-3 of the batch-pooled per-dimension sd, apply the swap only when BOTH partners have
>= 200 valid frames (shorter utterances keep their own statistics and count as an unswapped view), compute the
statistics over valid frames only, leave padding at 0, assert finiteness, and unit-test that the swapped view
of an utterance with itself is the identity at production dtype.

F7. "Outside the band" threshold (SAE_4A_infomax.md:110-111). The band is the chance floor of a length-faithful
string: the null at the arm's own length and unigram scores 0.840-0.867 and 0.888-0.900 at gold length
(SAE_4A.md:984). With n = 1, PER 0.80 vs 0.83 is inside the null's spread across symbol distributions.
Amendment 7: at the ep10 early read and at every kept checkpoint, run the existing chance-null script
(emc_hyp_inspect section 7) on the arm's own decode and declare "outside the band" only when PER is below the arm's
own length-and-unigram-matched null by more than 0.05; keep 0.80 as a screening number only.

F8. Which entropy. Amendment 8: the 1.0-nat clause is the EVAL-mode dev-other entropy from the registered read
(dropout off, BatchNorm running statistics), not the train-mode as_error mean, which is taken under dropout 0.1 on
training batches; both are reported.

F9. 2 x 2 at n = 1. Adequate for the gate (see above). The schedule contrast (enthold_50 vs ent_50) and the
invariance contrast between two in-band arms are uninterpretable at n = 1 (within-band paired deltas are not
content; the file already says so). The omitted invariance-only arm is justified, but by data rather than theory:
S3b-C (spec, spec_speed, with rate and agg) already ran it and failed with inventory collapse (SAE_4A.md:984);
cite that. Abort rule: the expected phone rate is pinned by the rate term, so the [4, 20] window will not fire on
an anneal transient; fine as written.

## Cheapest falsifying check (do it now)
A registered eval-mode forward of ctrl_50's kept checkpoints ep1, ep4, ep10 (all exist) on dev-other reporting
(i) mean per-output-frame entropy in nats and (ii) greedy symbol-usage entropy in bits. If (i) at ep10 is already
below 1.0, the early-read entropy clause is void and lambda must be re-sized before the four arms are funded; if
(i) is 2.5-3 nats the sizing paragraph's premise holds and only the denominator fix (F2) remains.

## What I could not verify
The exact tensor and denominator the implementer will use for `ent` (no code exists yet); the utterance-length
distribution of the priorshuf bed (the 200-frame guard is a stated choice); whether PairedPerDeltaJob emits an
interval (it changes nothing at n = 1 across seeds).
