# SAE §1e — pairing-free initialization

## Approach

**1. Length-banded pseudo-pairing (1e.1) against a random-permutation control (1e.2).** For each audio
utterance, target `n_hat = duration / 0.1044 s-per-phone` and sample uniformly without replacement
from texts whose phoneme count lies in `[n_hat(1-band), n_hat(1+band)]` with `band = 0.169`, the
measured speaking-rate CV; `band -> inf` degenerates to a uniform random permutation, so the control
is the same code path with one different hashed argument. Rank-matching was rejected on measurement:
sorting both sides and zipping gives pairs **3x more length-consistent than real speech** (median
relative length error 0.035 against the true 0.104), which teaches AV and AR a deterministic
duration -> length map that reality does not obey. Text pool = train-clean-100 minus the seed's own
2849 utterances (25,690 in-domain gold transcripts, disjointness asserted), so no seed sentence is
reachable; train split only, dev/test pass through gold.

| quantity | length-paired (1e.1) | random (1e.2) | TRUE pairing (eval-only) |
|---|---|---|---|
| median abs relative length error | **0.0795** | **0.2258** | 0.1016 |
| mean | 0.0812 | 0.5004 | -- |
| assigned phoneme count (mean) | 124.2 | 124.6 | 124.4 |

Speaking rate estimated from marginals alone is 0.10123 s/phone against the true paired 0.1044 —
within 3 %, so the estimator consults no true pair. 1e.1 sits at 0.78x the true length spread, 1e.2
at 2.2x, matching the independently measured random floor; the text marginal is preserved to 0.2
phonemes in both arms, and the band never had to widen.

**2. Budget-identical SFT screens against the gold 10 h AV.** Six SFT jobs (AV + two AR arms per mode)
on the pairing dirs, units from the pairing-free `KMeansUnitsJob` (vanilla `wav2vec2-large-lv60`,
no checkpoint and no transcript column as inputs, so it is pairing-free by construction). The schedule
is the 10 h track's verbatim — same 2849 utterances, same durations, 189 steps/epoch, 50 epochs,
9450 steps — so only the pairing differs.

| pairing | ep40 dev-clean | **ep50 dev-clean** | **ep50 dev-other** |
|---|---|---|---|
| gold (10 h theta_0) | 20.45 | **16.91** | **20.64** |
| length band 0.169 | 209.76 | **191.14** | **206.26** |
| random | 307.00 | **301.80** | **349.12** |

Pin rule (user, 2026-08-01): **last epoch by train-loss plateau**, not the dev-CE minimum. Train loss
is flat to ~0.005/ep by ep50 on both arms; the rule is label-free and avoids the CE/WER anti-alignment
that makes a dev-CE pin select a 470 %-WER checkpoint on the 1 h rung.

## Conclusion

1. (1) The designed contrast is real and correctly calibrated — length-banded pairing reproduces the
   true length statistics rather than an idealized one, and the random arm lands on the independently
   measured random floor.
2. (2) **Neither permuted arm reaches the gold's phase transition on the gold's own step budget.** The
   10 h gold goes 119.47 -> 20.45 -> 16.91 dev-clean between ep30 and ep50; the length arm is still
   improving monotonically (399 -> 333 -> 234 -> 210 -> 191) but at 9450 steps has not reached where
   the gold stood at ep20.
3. (2) The duration-band signal is nonetheless real: length separates cleanly from random (209.76 vs
   307.00 at ep40). It is far from enough to trigger the transition.
4. This is the honest supervision comparison the seed-size rungs cannot offer — the 1 h and 10 min
   rungs are under-stepped (4066 / 1664 steps) and sit below the transition window, so their >100 %
   WERs say nothing about seed size.
5. Open and unanswered by this screen: whether the reward *ranks* under this init. AR CE is
   length ~ random (0.019 nats apart) while the WER contrast is large, which is exactly the §2.5(d)
   question — no §1e loop compute is funded before it is read.

## Catalog

`T/` = `work/i6_core/returnn/training/`, `S/` = `work/speech_llm/sae/`.

| artifact | path |
|---|---|
| pairing builder + tests | `speech_llm/sae/pairing.py`, `test_pairing.py` (9/9) |
| pairing dirs (length band 0.169 / random) | `S/pairing/BuildTextPairingJob.*` |
| pairing-free units (0/500 dead, usage entropy 0.9396, dedup rho 0.83) | `S/units/KMeansUnitsJob.Yd8s22imQZ91` |
| disjoint text pool source (tc100, finished) | `TransformAndMapHuggingFaceDatasetJob.OYvh9012Pgkb` |
| gold comparison AV (theta_0, ep50) | `T/ReturnnTrainingJob.OLzy9Q2oC3mU` |
| arm configs | `config_sae_2s_1e_init_v1.py`, `config/sae_2s_1e_init.py` (commit `de1dae8`) |

Quarantine note, wider than PLAN spells out: both the p10 **avunits** and the pre50/pre125 "encoder
tap" units are seed-derived — every `AvStatesJob` pins `av_checkpoint=OLzy9Q2oC3mU` and theta_0 was
trained with `encoder_trainable=True`, so even the pre-adapter tap reads SFT-fine-tuned wav2vec2
weights. Only `KMeansUnitsJob` output is legal for §1e.

Deferred: **1e.3 (audio-continuation)** — no train step in the repo predicts units *from* audio, and
an AV emitting unit tokens needs a vocab resize `SpeechLmV1` does not do (`install_sentinel_overlay`
asserts `first_id + num_ids <= num_labels`; Qwen3's ~290 spare rows cannot hold K+2 = 502). Budget it
separately.

## Verifier feedback

**2026-08-01.**
- Reproduced the pairing statistics independently to three decimals.
- Useful asymmetry in the screen anchors: `devtrain` is built from the *train* split of the same dir,
  so `devtrain_loss_ce` is measured on the PERMUTED pairing (label-free, legal to read and select on)
  while `dev_loss_ce` is gold (label-derived, eval only, must not select).
- The §2.5(d) anchor is `devtrain_loss_ce`, not `dev_loss_ce`: the rollout set is 128 utterances of
  the AR's own train split, so `recon + gap_true` must equal `-devtrain_loss_ce`. Measured on the
  validated p10 arm: R_true -5.7226 at all four temperatures against devtrain 5.70746 (delta 0.015);
  anchoring on `dev_loss_ce` 5.7371 instead would have manufactured a false 0.2-nat failure.
- `band=0.216` would match the true length spread exactly; 0.169 is kept because it is the disclosed
  2S-measured constant rather than a number fitted to the gold pairing.
