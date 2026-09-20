# Audit: private-code analysis of ctrl_50 (ep1/4/10), fresh context, 2026-09-20

Read-only. Artifacts: `work/speech_llm/sae/emc/private_code/{PrivateCodeAnalysisJob.*,SymbolDeciphermentJob.*}`,
code `recipe/2025-10-speech-llm/src/speech_llm/sae/emc/private_code.py`, pre-registration
`SAE_4A_infomax.md` section "Private-code analysis". Re-derivations run with
`/e/project1/spell/wu24/env/conda/envs/speech_llm/bin/python`, scratch in `/tmp/claude-34349`.

Frame check: the three posterior dumps trace to `PackedBlankfreeTrainJob.ks7CbtlvpcIL/output/ctrl_50/models/epoch.{001,004,010}.pt`
(via `ExtractSubmoduleCheckpointJob.{lTNO6CVoEo1B,tlthmI5gMGrz,Pxbk3AVmFsm2}`), i.e. the epochs claimed.
All six analysis jobs read the uniform-window prior `PhoneNgramPriorJob.RtzbESkOedsT` (the standing
"n-grams from the unbiased window" rule) and gold `GoldPhonesJob.ZGSp0hxyd2YP`.

## 1. Internal consistency — CONFIRMED

`per_as_scored` equals the banked `per.json` to the last digit in all six reads
(dev-other ep1/4/10 0.855428 / 0.868775 / 0.896793 against `BlankfreeGreedyPerJob.{XN6vhGGyKQu5,Vz6QOYliPU40,dG4n46xTRSl0}`;
dev-clean 0.846393 / 0.849745 / 0.881008 against `{VhipueEiV9cf,gAmloZ1hqFfq,GYhOaiWJF2Yv}`). Counts agree too
(ep10 dev-other sub 130396 / del 19174 / ins 9409 / ref 177275). Coverage is complete: 2864 dev-other
utterances (2703 dev-clean), 261295 output frames = `per.json:output_frames`, 781130 unit frames =
`retained_frames`, MFA frame gold on 2864/2864 with 0 dropped and 0 missing. dev-clean moves the same way
on every read (token NMI 0.1312 -> 0.0731 -> 0.0571; frame NMI 0.0930 -> 0.0647 -> 0.2713;
NMI(sym,unit) 0.1593 -> 0.1013 -> 0.3957; NMI(sym,spk) 0.0146 -> 0.0302 -> 0.0052).

## 2. Oracles — computed as documented, but one convention bug and one missing null

Code matches the docstrings: `hungarian_labelling` is a 40x40 `linear_sum_assignment(maximize)` with a single
zero-gain DELETE column; `hungarian_drop_labelling` adds one private drop column per symbol at gain `ins(s)`,
with an assert against cross-assignment; `majority_labelling` is per-row argmax. I reproduced the banked
many-to-one and with-drop PERs exactly from `greedy_raw.json` + the maps stored in the json.

The with-drop *pricing* is a correct account **under the fixed identity alignment**: keeping `s` at `c` costs
`n_tok(s) - conf[s,c]` (substitutions + retained insertions), dropping `s` costs `aligned(s)` (each aligned token
becomes a deletion, the insertions vanish). Since the reported PER is re-scored with a *fresh* alignment, the
assignment only bounds each option, so the map is not provably optimal: the reported with-drop PER is an
**upper bound** on the best achievable with-drop PER (pessimistic, direction stated in the docstring).

Bug: `PrivateCodeAnalysisJob.score()` applies `relabel()` **without** `collapse_runs`, although the module's own
`collapse_runs` docstring says a many-to-one map must be re-collapsed (and `SymbolDeciphermentJob` E5 does
re-collapse). The part-A many-to-one row is therefore pessimistic by up to 0.019 PER:

| dev-other | many-to-one as banked | re-collapsed (recomputed here) |
|---|---|---|
| ep1 | 0.821469 | 0.820144 |
| ep4 | 0.835871 | 0.826958 |
| ep10 | 0.874376 | **0.855191** |

(with-drop is essentially unaffected: 0.841179 -> 0.840288 at ep10.) The corrected 0.855 is still inside the
0.85-0.91 chance band, so the reading survives; the number quoted in the phase file should be the collapsed one.

Missing null: at ep10 the with-drop oracle gains 0.897 -> 0.841 while dropping 12 of 40 symbols (18 % of tokens).
The only banked relabeling null (`SAE_4A.md` line 984: a Hungarian relabeling lifts a length-and-unigram-matched
chance string by <= 0.0012 PER) is a *no-drop bijection* null and does not price that freedom. Nothing on disk
says how much of the 0.056 a content-free string would also get.

## 3. Decipherment (part E)

* **No gold in the fit — CONFIRMED by code path.** `SymbolDeciphermentJob.run` calls `em_decipher` at line 1180 with
  only `ids` (from `greedy_raw.json`) and `prior.log_bi`; `gold` is first touched at line 1193 building `conf`,
  after the fit. `labels_used_in_fit: false` is recorded. The label-using rows (Hungarian PER, E3 agreement) are
  oracles computed over *both* halves, so "held" is held out for the cipher only, not for those rows.
* **Halves disjoint — CONFIRMED.** `tags[0::2]` / `tags[1::2]`, 1432 + 1432 = 2864, no overlap by construction.
* **E5 null is NOT family-matched.** The hard relabeling is many-to-one (24 distinct target phones at ep10, 23 at ep4;
  SIL takes 5 symbols) and is re-collapsed; the 20 null draws are *bijections*, which preserve the symbol
  distribution and cannot collapse anything. A multiset-matched null (50 random permutations of the deciphered map's
  own target vector, i.e. the same many-to-one shape, random assignment) is far stronger than the published one:

  | ep10 dev-other | hard | published perm null mean/sd/max | matched m2o null mean/sd/max |
  |---|---|---|---|
  | fit half | -3.8123 | -8.4257 / 0.3918 / -7.8346 | -6.9628 / 0.4081 / -6.1773 |
  | held half | -3.8373 | -8.4245 / 0.3908 / -7.8242 | -6.9671 / 0.4062 / -6.1883 |

  About 1.6 nats of the published 4.6-nat margin is bought by the many-to-one shape alone. The verdict survives
  (hard still exceeds the matched max by ~2.4 nats), so the gate's answer does not flip — but the margin is inflated.
* **Worse: the pre-registered rule is passed by doing nothing.** The *identity* labeling scores -3.9783 / -3.9949 at
  ep10, above both null maxima. "Prior score exceeds the null max" therefore does not establish "a relabeling the
  prior prefers exists" in the sense the phase file reads it; the informative contrast (hard minus identity) has no
  null at all. At ep4 the rule is informative: identity -7.4261 sits *inside* the permutation null spread
  (mean -8.4408, sd 0.6108, max -7.3536) and the hard map buys 1.69 nats.
* **Does 0.17 vs 1.7 nats support "by sub-epoch 10 the labeling is already near the prior-best hard relabeling"? NO,
  not as stated.** (i) Nothing searched for the prior-best relabeling; the hard map is the EM cipher's argmax.
  (ii) The prior-best hard relabeling is degenerate and unbounded: a constant map collapses each utterance to one
  token, whose per-token prior score is far above anything here, so "near the best" has no defined target.
  (iii) The gap shrank because the *identity* baseline moved (-7.43 -> -3.98), not because a ceiling was approached.
  (iv) The two means use different denominators (88168 vs 87437 tokens); that confound is not decisive here, since
  the totals move the same way (-350754.6 vs -333339.4). Supported instead: at ep10 the raw output is already far
  more trigram-plausible than any relabeling of it, and relabeling buys little.

## 4. Missing reference: the prior's score on GOLD dev-other

Recomputed with `PhoneNgramPrior.load` on the job's own prior `PhoneNgramPriorJob.RtzbESkOedsT`, order 3, the
`log_prob` convention the job uses (two BOS, no EOS):

**gold dev-other = -3.1992 nats/token** (177275 tokens, 2864 utts; fit half -3.1898, held half -3.2087;
dev-clean -3.1227). The banked -3.22 in `SAE_4A.md` line 1100 is the same quantity under the *older head-window*
prior `PhoneNgramPriorJob.TRPE0D5nF3bh`; I reproduce it as -3.2195, so the two priors agree to 0.02 nats.

**The quoted comparison is not like-for-like.** Gold (`GoldPhonesJob`) is SIL-free; E2/E5 score token streams that
**keep SIL** (7639 SIL tokens = 4.4 % at ep10, 17 % at ep4). Scoring the same streams SIL-free (ep10 fit / held):
identity **-4.6174 / -4.6402**, hard relabeling **-4.5447 / -4.5746**. So the honest gap to gold is ~1.42 / ~1.35
nats, not the ~0.78 / ~0.61 the with-SIL numbers suggest. Separately, the Viterbi row -2.12 is **better than gold
(-3.20)**, i.e. the deciphered string is more trigram-plausible than real speech — the clearest confirmation of the
report's own warning that E2's deciphered row is prior-optimized by construction and carries no evidence.

## 5. Trend ep1 -> ep4 -> ep10 (dev-other)

| read | ep1 | ep4 | ep10 | direction |
|---|---|---|---|---|
| token NMI(symbol, phone) | 0.1149 | 0.0627 | 0.0562 | **down** |
| many-to-one token PER (as banked / re-collapsed) | 0.8215 / 0.8201 | 0.8359 / 0.8270 | 0.8744 / **0.8552** | **up (worse)** |
| frame NMI(symbol, phone) | 0.0868 | 0.0569 | **0.2559** | down then sharply up |
| frame error, many-to-one | 0.8984 | 0.8694 | **0.6987** | **down (better)** |
| NMI(symbol, unit) | 0.1481 | 0.0922 | **0.3770** | down then sharply up |
| NMI(symbol, speaker) | 0.0177 | 0.0405 | **0.0065** | up then down to near gold (0.0035) |
| symbol usage entropy, frames (bits) | 1.667 | 3.759 | 4.985 | up |
| mean run length | 2.64 fr / 158 ms | 2.11 fr / 126 ms | 1.49 fr / 89.5 ms | down toward gold 84 ms |

Frame-level chance reference (not in the tables; estimated here from gold token counts x the banked per-phone mean
durations): the best content-free constant map gives frame error ~0.930 (max gold frame class ~7.0 %). So 0.6987 is
0.23 absolute below the content-free floor — real frame-level phone information.

* (a) **a relabeling of phones — CONTRADICTED at the token level.** Best one-to-one-with-drop 0.8412, many-to-one
  0.8552 (collapsed), both inside the 0.85-0.91 chance band, while token NMI *falls* to 0.0562 (I = 0.276 bits).
  It holds only at the frame level.
* (b) **a speaker or session code — CONTRADICTED.** NMI(symbol, speaker) = 0.0065 at ep10 (I = 0.033 bits), lower
  than at ep1/ep4 and within a factor of 2 of the gold-phone reference 0.0035. The code moves *away* from speaker.
* (c) **a coarse clustering of the reverse units — PARTIALLY, and the pre-registered clause fails as written.**
  NMI(symbol, unit) rises to 0.3770, the biggest mover, but the symbol is not "nearly a function of the unit"
  (H(symbol|unit) = 2.503 bits of H(symbol) = 4.987), and the clause's second half is contradicted: gold phones are
  *more* unit-determined than symbols (H(phone|unit) = 1.920 of 4.944, NMI 0.4610). The "easy reconstruction code"
  reading cannot fire on these numbers.
* (d) **a frame-level acoustic code whose token sequence is not phone-like — SUPPORTED, and it is the only
  description all the numbers fit.** Decisive pair: frame I(symbol; phone) = 1.270 bits with frame error 0.699 (floor
  0.930), against token I = 0.276 bits with every token oracle at chance. The code tracks phone identity in time but
  its collapsed token string cannot be aligned to gold by edit distance (59.6 % one-frame runs; 175149 emitted tokens
  against 177275 gold phones, so it is length-faithful, not degenerate).

## 6. What undercuts the pre-registered readings

1. **No null for the with-drop / many-to-one oracles** at the one epoch where they gain (ep10, 0.056 PER,
   12 symbols dropped). The banked null is a no-drop bijection null; it does not price a drop option.
2. **Part A's many-to-one string is not re-collapsed** while part E's is — one convention, two jobs, 0.019 PER.
3. **E5's null is a bijection null for a many-to-one statistic** (margin inflated ~1.6 nats), and the identity
   labeling also clears the null max at ep10, so the pre-registered rule is passed without any relabeling.
4. **SIL convention mismatch** between the prior-score rows (SIL kept) and any gold reference (SIL-free): -3.98 vs
   gold -3.20 compares different token streams; like-for-like is -4.62 vs -3.20.
5. **Part B's pre-registered clause is contradicted, not merely unmet** (gold is more unit-determined than the symbol).
6. **Part D's KL rows use the no-drop Hungarian map**, the map the report itself says is not the one to quote.
7. **Frame-level rows have no chance reference** in the tables, and the frame error rate is a different denominator
   from the token PER band, so the two must not be read on one scale.

None of 1-7 overturns the headline reading (the ep10 output is not a token-level relabeling of phones; it is a
frame-level acoustic code with real but non-segmental phone information, not a speaker code). Items 2, 3 and 4
change quoted numbers, and item 1 leaves the 0.841 with-drop read unlicensed.
