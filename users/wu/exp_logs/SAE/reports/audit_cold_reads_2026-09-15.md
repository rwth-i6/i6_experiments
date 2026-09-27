# Audit -- network-free cold-start fixed-point reads of the EMC objective (2026-09-15)

Fresh-context, read-only. Every number below re-derived from the `.json` under
`/e/project1/spell/wu24/2026-07-13_unsupervised/analysis/out/`, not from the `.txt` or the
executor reports. Nothing was edited or rerun.

Provenance control that holds: `cold.check_banked.dev-other.json` reproduces the banked
frozen-phi fixed point at k=0 (0.1100) and k=1 (0.0962), |diff| 0.00e+00, tol 1e-3, PASS.
Every cold line shares tau=2, alpha=0, band W=25, 300 dev-other utts `--select stride`,
same gold (`GoldPhonesJob.ZGSp0hxyd2YP`), same prior npz, same eta table. The cross-line
comparisons are therefore on one axis each.

## (a) c2/c3 blank collapse -- CONFIRMED on the bigram chain, REFUTED as literally stated on trigram

Bigram, `cold.c2_flat_fitphi.bigram.dev-other.json` and
`cold.c3_flat_fitphi_pindur.bigram.dev-other.json`: at every k = 1..30,
`per` = 1.0000, `phones_per_sec` = 0.0000, `distinct_phones` = 0,
`blank_share` in [0.9907, 0.9938], `fit_items.kept` = 0 with `infeasible_long` = 300,
and `fit_history` = null at every k. Pinning durations (c3) changes nothing: identical
PER / ph/s / blank-share envelope. Claim holds exactly.

Trigram, however, is NOT "0 ph/s at every k" and NOT "no feasible refit at every k":
* `cold.c2_flat_fitphi.trigram...json`: k=14..30 carry ph/s 0.0434-0.1552 with
  `distinct_phones` = 1; `kept` = 2 at k=14..21, 1 at k=22 (so the refit DID fire, on 2 of
  300 utterances). Min PER 0.9847.
* `cold.c3_flat_fitphi_pindur.trigram...json`: k=7..30 carry ph/s up to 0.1552,
  `distinct_phones` rising 1 -> 3 by k=30, `kept` 1-3 at k=7..17 and k=25..30.
These are quantitatively negligible (PER 0.985-0.996, blank >= 0.9876) so the *substance*
-- a blank collapse that durations do not rescue -- survives; the universal quantifier does
not. Report it as "0 ph/s on bigram; <=0.16 ph/s and 1-3 phone types on trigram".

Second precision point: `phones_per_sec` = 0 does not mean the argmax is all blank.
`runs_per_sec` = 0.1552 and `distinct_phones_with_sil` = 1 at every k in c2 -- the argmax
emits SIL runs and no phone. "Blank collapse" should read "blank+SIL collapse".

Third, and material for how (a) is used downstream: because `kept` = 0 on every bigram k,
`reverse.fit` never ran. c2/c3/c5 bigram are therefore NOT "phi refit each k" runs; phi
stayed at the single RETURNN-seeded random construct for all 30 iterations. They read the
target map under a fixed random phi, not joint EM from cold.

## (b) c1 take-off -- numbers CONFIRMED; the label question is answered YES, labels entered

`cold.c1_flat_warmphi.bigram.dev-other.json` k=30: PER 0.2794, ph/s 8.2699,
`distinct_phones` 39. Trigram k=30: PER 0.2379, ph/s 8.7261, `distinct_phones` 39.
Both start at k=0 PER 1.0000. Matches "~0.28 / ~0.24, 8.3-8.7 ph/s, 39 phones".

Caveat on the "1.0 ->" framing: the k=0 row is an argmax tie-break, not an objective
property. The txt header states it: flat theta has final-layer max|W| = 0.000e+00, so every
frame's posterior is uniform over 41 outputs and argmax returns index 0 = blank. The PER 1.0
start is free.

**What the warm phi was fit on.** `ReturnnTrainingJob.htxT2f9FHvWw/output/returnn.config`
line 43 sets `recognizer_checkpoint_path` =
`work/i6_core/returnn/training/ReturnnTrainingJob.65NNK8Bwxdtd/output/models/epoch.024.pt`
with line 42 `freeze_recognizer: True`. That recognizer's own config
(`ReturnnTrainingJob.65NNK8Bwxdtd/output/returnn.config` lines 27-30, 95-98) trains on
`PhoneTargetHdfJob.DXDTg3VoP47A/output/targets.hdf`, whose `info` file records
`INPUT: SeedGoldPhonesJob.zii9E9tvr51e/output/seed_gold_phones.json` under
`ALIAS: sae/4a/s0b/init_oracle_quarantined/targets`.

So c1's phi was fit against the frozen posteriors of a GOLD-PHONE-SUPERVISED recognizer
(the same 65NNK8Bwxdtd ep24 that is the "seed" theta reading PER 0.1100). Labels reach c1's
phi indirectly but decisively. The harness's own gold-freedom assertion
(`assert_items_are_gold_free`) covers only the refit item builder inside this script; it says
nothing about the provenance of a checkpoint handed in via `--phi warmup`. c1 is a
label-informed ceiling on what a good phi can do, not a label-free cold start.

## (c) c4 drift -- CONFIRMED, comparison is like-for-like

`cold.c4_seed_fitphi.bigram.dev-other.json` k=0/1/5/10: 0.1100 / 0.0962 / 0.1178 / 0.1380.
Trigram: 0.1100 / 0.0936 / 0.1064 / 0.1218. Both match the claim.

Frozen-phi reference located: `analysis/out/emc_target_vs_gold.fp.armA_step0.dev-other.json`
(label `armA_step0_fp`, job `ReturnnTrainingJob.4BRumKQFcXim`), k=0/1/5/10 =
0.1100 / 0.0962 / 0.1400 / 0.1898 -- the claimed row. Read from that file directly, not from
the copy embedded in c4's json. Its `checkpoint` block names theta = 65NNK8Bwxdtd ep024 and
phi = htxT2f9FHvWw ep001; header n_utts 300, select stride, split dev-other, tau 2.0,
band 25, ablations null (= bigram). Same theta, same phi, same subset, same metric, same
greedy decode, no ablation on either side. Comparable.

"Drifts slowly toward fewer emissions" is supported: ph/s 9.8996 -> 9.0958 over k=0..10 with
the refit, against 9.8996 -> 8.6955 frozen; 39 phone types retained throughout; refit
degrades PER strictly more slowly than frozen phi. Note the reverse LL rises monotonically
(-4.152 -> -2.540) while PER worsens, i.e. the fitted quantity and the target quantity move
in opposite directions here.

Frame note: c4 is oracle-seeded on both sides (theta = the gold-trained seed, phi = the
oracle-derived warm-up). It is a stability read of the target around a good point, not a
cold-start read, and carries nothing about cold start on its own.

## (d) c5 rate tilt -- CONFIRMED, and the constraint was genuinely met

`cold.c5a_flat_fitphi_ratetilt.bigram...json` and
`cold.c5b_flat_fitphi_pindur_ratetilt.bigram...json`: at every k = 0..10,
`per` = 1.0000, `phones_per_sec` = 0.0000, `distinct_phones` = 0, `kept` = 0 with
`infeasible_long` = 300, `fit_history` = null.

The constraint was actually satisfied, so the claim is not vacuous: every k carries
`rate_tilt.converged` = true, `rate_hz` in [9.603, 9.741] against `rho_hz` 9.6619373279
(`analysis/out/rho.rate_term.txt`), `rel_err` <= 0.0082, inside the 1% tolerance. Per-frame:
E[N_nonSIL]/T = 0.1920-0.1948 vs rho_per_frame 0.193239. No `conv = NO` row anywhere.

Strongest single number for "expectation, not mode": `runs_per_sec` = 0.1552 with
`distinct_phones_with_sil` = 1 at k=1/5/10 in c5a, c5b AND untilted c2 alike -- the tilt
moved the expected non-SIL count from ~0.31/frame (b=0) onto rho while leaving the argmax
sequence bit-identical to the untilted run.

rho itself is label-free as documented: 3.5785 phones/word from
`TextToPhonemeJob.THKMON3k9LJQ` x a disclosed 2.7 words/s, not the dev gold rate.

## (e) The two inferences

"The cold problem sits on phi's side" -- supported as a decomposition, with a serious frame
limit. c1 vs c2 is a genuinely single-axis contrast: identical flat theta, prior, tau, alpha,
band and subset, and since c2's refit never fired both lines are effectively frozen-phi runs
differing only in which phi. So which phi you hold does decide whether the target map runs
away to blank or to 8+ ph/s at 39 phone types. But:
 1. the only phi shown to work is oracle-derived (section b), so nothing here shows a
    label-free route to such a phi, and the inference must not be restated as "fix phi and
    cold start works";
 2. the failing side is one random draw (RETURNN seed rule, epoch 1 / step 0,
    random_seed 42) -- n = 1, no spread;
 3. **operating point.** Every read is at tau = 2. The S3 run's own config
    (`ReturnnTrainingJob.sBlPYBA1YcIQ/output/returnn.config:21`) sets
    `temperature_schedule = [8.0, 5.04, 3.17, 2.0, 2.0, 2.0, 2.0, 2.0]` over 8 sub-epochs, and
    the cold reads' header states they take the FIRST sub-epoch at the schedule's final value.
    The real cold-start collapse (emitted rate 1-2 /s "during the anneal") happens at tau
    8 / 5.04 / 3.17. These reads do not probe that regime at all. The claim explains a
    fixed point the real run reaches only after the anneal it failed in.

"The rate term prices the expectation, not the mode" -- CONFIRMED by (d), with the mechanism
caveat below.

## Confounds, named

* **Oracle-derived phi in c1** (decisive, section b). Also inherited by c4 on both sides.
* **tau/alpha vs the training run**: reads at tau=2, alpha=0; S3 cold-starts at tau=8 and
  anneals. alpha=0 matches (`anchor_weight_schedule: 0.0`), tau does not.
* **Tilt != a trained rate term**: c5 solves one scalar b by bisection to hit the expectation
  EXACTLY at each k, applied to non-SIL segment-table entries with blank and SIL untilted.
  A trained rate loss is a weighted gradient penalty on theta with no exactness guarantee and
  a different blank/SIL treatment. c5 licenses "an exactly-enforced expectation constraint
  does not move the mode"; it does not license a prediction about a tuned rate loss in
  training.
* **c5 is bigram-only and K=10**; no trigram c5 exists, so the prior-order rung is untested
  for the rate tilt, and the trigram c2/c3 lines are precisely the ones that leak a little
  non-blank mass (section a) -- i.e. the chain most likely to behave differently under a tilt
  is the one not run.
* **"Refit each k" is vacuous on c2/c3/c5 bigram** (`fit_history` null at every k), and near
  vacuous on trigram (1-3 items of 300).
* **Subset / denominator**: 300 of 2864 dev-other, stride-selected, identical on both sides of
  every contrast, so comparability is fine. But every PER is a single corpus-level number over
  that subset with no spread and no per-item paired deltas; differences like c4's 0.1380 vs
  0.1898 have no reported uncertainty.
* **eval() vs train()**: reads run the recognizer in eval(); the train step runs train().
  Immaterial for the flat-theta lines (zero logits give a uniform posterior in either mode) and
  reaches only the k=0 row of c4 / check_banked.
* **Not re-derivable here**: the S3 baseline "dev-other greedy PER 0.8955 at its gate read".
  It appears in none of these artifacts. It is context, not one of (a)-(e), but it should not
  be quoted as if these files support it.

## Verdict

DONE_WITH_CONCERNS. All the quoted numbers reproduce from the json. Two things stop this
being a clean confirmation: the trigram c2/c3 lines are not literally "0 ph/s, no feasible
refit at every k", and -- the decisive one -- the c1 phi that produces the take-off descends
from gold phone targets through a frozen supervised recognizer, so the evidence carrying
"the cold problem sits on phi's side" is label-informed and cannot be read as a label-free
cold-start result.
