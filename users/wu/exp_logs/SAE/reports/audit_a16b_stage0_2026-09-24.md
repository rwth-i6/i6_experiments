# Audit: A16 (b) stage-0 key-floor read (KeyFloorReadJob.DgDbciHlq2wY), 2026-09-24

CONFIRMED_WITH_CORRECTIONS

The printed verdict, J SEES THE KEY, follows from table.json under the registered rule and its
pre-result amendments. The decisive clause clears its margin by 0.009 nats per frame. That excess is
smaller than the train-side versus held-out difference of the decisive key itself, and it is not
robust. J is monotone in key accuracy along the random-reassignment ladder only. Across key families
it is not monotone, under direct, one-to-one (Hungarian) and many-to-one accuracy alike. Near gold, J
cannot tell gold apart from the r30 key, which differs from gold on 91 units.

Scope: read-only. I re-derived everything from
`/e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/unit_key_jobs/KeyFloorReadJob.DgDbciHlq2wY/output/table.json`
and from `GoldUnitKeyJob.sLnMRRd2qO0t/output/{key.json,counts.npz}`, using short numpy scripts on the
login node. I did not recompute J from the corpora. The key-accuracy numbers below are audit-derived
and label-using. They are a report only and must never be used for selection.

## 1. Verdict arithmetic (held-out J, real corpus, 260 utterances, 137,933 frames)

- J(gold) = -4.8180.
- Ladder seed means:
  - K30 = -5.6097 (seeds span -5.641 to -5.587);
  - K70 = -6.3335 (-6.395 to -6.282);
  - K100 = -6.5597 (-6.629 to -6.509).
- The ladder is strictly monotone. The steps are 0.792, 0.724 and 0.226, and no seeds overlap between rungs.
- Random keys: 20 keys, all distinct (seeds 1-20).
  - Maximum -6.4596 (random_s01), minimum -6.6039 (random_s13).
  - Range 0.1443, standard deviation 0.0397.
  - Margin = max(0.01, 0.1443) = 0.1443.
- Comparison key per phi, the better of its posterior and likelihood keys: the posterior key won for all seven phis. Amendment (2) therefore did not change any comparison key.
- The best comparison key is a10_durfrz_s01_ep48 at -4.9714.
  - Gap 0.1533 against margin 0.1443, so the excess is 0.0090 (the gap is 1.063 times the margin).
  - The gap splits into LM +0.2386, emission -0.0801 and duration -0.0052. Gold wins only through the trigram term. The emission and duration terms prefer the EM key.
- The best random key's gap is 1.6415, far above the margin.
- VOID: none of the 50 keys the rule reads is VOID. Held-out rates are 7.84-9.82 Hz against the band [5.80, 14.49]. The VOID handling (flag and name, do not remove) therefore could not affect the verdict.
- The printed verdict matches the json and the rule text in the job docstring (commit 6ae84c59, 09:01:19). The manager started at 09:02:17 and the read ran at 09:13.

How close the decisive comparison is:
- On the train side (report-only, 15.3 M frames), the same clause would fail. The gap there is gold -4.8179 minus durfrz_s01 -4.9595 = 0.1416, against a train-side random range of 0.1466.
- The decisive key's own held-out and train J differ by 0.012, which is more than the 0.009 excess.
- The margin is the range of 20 random draws. If the 20 random J values are treated as normal with sd 0.0397, a fresh set of 20 seeds would give a range above 0.1533 with probability of about 0.40. This is a model-based estimate, not a banked statistic.
- The pass is therefore within the noise of the criterion itself. That does not void it: the rule, seeds and side were fixed in advance, and the read applies them correctly.

## 2. J is computed identically for every key

`KeyFloorReadJob.score` loops over all 56 keys with one `Tables` object:
- the same trigram RtzbESkOedsT;
- the A9 durinit law;
- d_min = 2 and D_k = 25/50 by symbol;
- alpha = 1e-3.

It also uses the same train and held-out `Corpus` objects and calls `corpus_counts` and `j_terms` identically for every key. Each key's held-out emission is the train-side table of its own assignment, add-alpha smoothed. That is the same rule for every key, and no key gets an extra term. The overlong split applies to all keys, as amended.

## 3. Gold key, held-out set and leakage

- Gold key: the majority label per unit from gilkeyio MFA train_clean_100.
  - It uses the 2821 tags of CvHoldoutSplitJob.sD7U6CYs8ACM (1,517,529 frames, frame purity 0.644, 38 symbols used, 66 units mapped to SIL).
  - It has no dev-other input.
  - It is byte-identical to the key the read used.
- Held-out set: disjoint.segments contains 260 tags. All of them lie in the 285-tag CV holdout. Their overlap is 0 with the 2821 fit set and 0 with the 28254 train segments, and the job asserts the train-side part. The 25 CV-holdout utterances that are in the fit set are excluded.
- Registration wording: the phase doc says "the CV holdout of the train stream". The standing A13 repair (the 260 disjoint subset) is the right reading, because the gold key is ladder-derived.
- Argmax keys:
  - Inputs are only a phi checkpoint or the A11 table, the train.segments of CvHoldoutSplitJob.PfpCPQRCfIAk (28254 tags) and SpeakerEtaJob eta.
  - pi is fitted to the train-side unit frame counts. All ten fits converged.
  - No D4 input and no held-out labels.
- The ladder keys derive from gold by design: exactly 150, 350 and 500 units are moved.
- Random keys use only an RNG.

## 4. What the verdict licenses

It licenses the registered consequence: stage 1 is funded. On held-out J, gold beats the seven EM/A11 keys and 20 random keys by more than the registered margin, and the ladder is monotone.

It does not license the following.

(a) That J is monotone in key accuracy.

Accuracy here is frame accuracy against MFA on the 2821 fit set. Group means (direct / one-to-one Hungarian / many-to-one) with held-out J:

| Keys | Direct | Hungarian | Many-to-one | Held-out J |
|---|---|---|---|---|
| gold | 0.644 | 0.644 | 0.644 | -4.818 |
| A10/A11 posterior keys | 0.088 (0.059-0.113) | 0.297 (0.288-0.317) | 0.359 | -4.97 to -5.18 |
| A10/A11 likelihood keys | 0.078 | 0.281 | 0.359 | -5.11 to -5.31 |
| K30 | 0.453 | 0.454 | 0.466 | -5.59 to -5.64 |
| K70 | 0.198 | 0.208 | 0.239 | -6.28 to -6.40 |
| random | 0.026 | 0.122 | 0.162 | -6.46 to -6.60 |
| K100 | 0.009 | 0.123 | 0.161 | -6.51 to -6.63 |

- Every A10/A11 key, in both variants, has higher J than every K30 key, although K30 is more accurate on all three measures.
- The reported r70 posterior key (0.346) and r70 likelihood key (0.421) also outscore K30.
- Using direct accuracy, 355 of the 1540 key pairs are discordant.
- The ladder confounds accuracy with temporal incoherence. The absorbed fraction is 0.14 for gold, 0.27-0.29 for K30, 0.38-0.39 for K70 and 0.41 for K100, against 0.15-0.21 for the EM keys. So ladder monotonicity shows only that J penalises random unit reassignment.

(b) That gold is J's maximum, or that J resolves accuracy near gold.
- The r30 posterior key (report-only) differs from gold on 91 units and has frame accuracy 0.610 against 0.644.
- It scores -4.8224 held-out (0.0044 below gold) and -4.8145 train-side (0.0034 above gold).
- Train-side J is the objective the stage-1 moves climb.

(c) That stage 1 will select accurate keys.
- EM keys with 6-11 % direct accuracy sit 0.15 below gold.
- Their emission term already beats gold's.
- Whether a local search from them, or from random or cluster starts, ends above gold at a wrong key cannot be determined from these artifacts.

(d) The destroyed-structure null.
- All 56 destroyed-corpus scores are VOID: rate about 2.4 Hz, absorbed fraction 0.90, overlong fraction 0.62.
- The reported "rise" is therefore measured against a degenerate null.
- Stage 1's registered null (the same search on the destroyed corpus) will face the same degeneracy.

Correction to the phase doc's stage-0 result text: "The A10 keys decode at chance" is not right for the keys.
- Their direct accuracy is 2-4 times the random keys' (0.059-0.113 against 0.026).
- Their Hungarian accuracy, about 0.29, lies between K70 and K30.
- They are partial clusterings with the wrong symbol names, not chance keys.
- The open question it raises is answered: J is not monotone in accuracy across families.

## 5. Not determinable from the existing artifacts

- Per-utterance paired J for gold against durfrz_s01. The read banks no per-utterance values.
- The spread of the margin under other random seeds.
- Held-out-set key accuracy. MFA labels exist only for the fit set.
- Whether gold is a local maximum of train-side J under single-unit moves.
- Whether any non-gold key exceeds J(gold).

Each of these needs a new registered reader. For example: `reassign_delta` from the gold key over all 500 x 39 moves, and from the r30 and A10 keys; per-utterance held-out J; and more random seeds, reported descriptively.
