# Audit -- pre-launch falsifier (ii), sampled-reward probe (read-only, fresh context, 2026-09-21)

**Verdict: CONFIRMED.** The rule fires as banked: r(gold) > max_g r(y_g) in > 95 % of utterances at
every checkpoint, so sf_50 is not funded. Nothing was edited, rerun or resubmitted.

## 1. Recomputed fractions (from `output/per_utt.json`, not from probe.md)

| ckpt | job | recomputed | banked (probe.md / probe.json) |
|---|---|---|---|
| ep1, tau 8 | `SampledRewardProbeJob.d1NoUQJ2EXN5` | 300/300 = **1.0000** | 1.0000 |
| ep4, tau 5.039684 | `.Y81PrZ6fWWKu` | 300/300 = **1.0000** | 1.0000 |
| ep10, tau 2 | `.HVHIlaUkIkVi` | 291/300 = **0.9700** | 0.9700 |

Exact agreement. The comparison is the rule's:
* strict `>`, max over the **8 draws only** (greedy excluded) -- `blankfree_probe_jobs.py:172`
  `gold_gt_max = [1.0 if r[key]["gold"] > max(r[key]["samples"]) else 0.0 ...]`;
* per utterance, denominator `n_scored` = 300 at each checkpoint, `n_masked` = 0 (no over-cap
  string anywhere; A4 masks, never truncates -- `:790-802`);
* SIL dropped by the same `drop_sil` on gold, greedy and every draw (`:130-132, 514-515, 761-766`);
* identical tokenisation on both sides: every string is mapped through the same `PHONE2ID` and
  handed to one `scorer.log_probs` batch and one `prior.log_prob(..., order=1)` (`:791-801`).
Recomputing with greedy inside the max gives the same 1.0000 / 1.0000 / 0.9700.

## 2. Are the draws genuine posterior samples at the stated tau?

* tau is asserted against the arm's own anneal (`:452-456`, `abs(tau - self.tau) < 1e-9`): 8 /
  5.039684 / 2 = `budget_temperature_schedule` at sub-epochs 1 / 4 / 10, the schedule's tau.
* sampler validated against full enumeration before the run: chi-square 61.11 on 59 dof (bound
  113.3), sampler `log Z` equals the enumerated `log Z`, decomposition identity over s_len 6-9 x
  tau 1/8 (`reports/impl_blankfree_sampler_2026-09-20.md:41-52`, `test_blankfree_sampler.py`).
* `max log w(path) - log Z` = -46.15 (ep1) / -40.0 (ep4) / -5.535 (ep10), all <= 0; 0 utterances
  with Z = 0.
* distinct strings 8.000 / 8.000 / 7.360 (min 1 at ep10) -- consistent with a peaked posterior at
  tau 2 on short utterances, not with a degenerate sampler.
* **Rebuild check, why not 300/300.** I diffed the probe's greedy against the banked
  `BlankfreeGreedyPerJob.XN6vhGGyKQu5/output/greedy_phones.json`: all 8 ep1 mismatches are a
  SINGLE edit (SequenceMatcher ratio 0.917-0.989; 6 x replace, 1 delete, 1 insert; lengths equal
  or +-1). Both sides use collapse-then-drop-SIL (`blankfree_eval_jobs.py:80-86` argmax +
  run-collapse + SIL filter; probe `_greedy` `:748-757` + `drop_sil`), and the banked json contains
  0 SIL in 60842 tokens, so the comparison is like-for-like. The pattern (8/300 at the flat tau 8
  and tau 5 checkpoints, 1/300 at the sharp tau 2 one) is what argmax near-ties between a live
  float32 forward and a dumped posterior produce. It does not touch the rule: greedy is not in the
  max, and the draws come from the same rebuilt `log_q` whose decode the check validates.

## 3. Scorer identity

`probe.md:11` and `info:7,17` of all three jobs: `NeuralPhoneLmTrainJob.Iv6P6YVPNWmB/output/model.pt`
-- the banked **first-pass, 3-epoch, 3.3 M** phone LM, gap 1.71 (`SAE_4A_prior.md:~455` table).
The two retrained instances ((a) 4L/w256 gap 1.861, (b) 6L/w384 gap 1.849) are separate finished
jobs (`NeuralPhoneLmTrainJob.{pBozvj6c3l16, xObXEwRpvmzd}`) and were NOT used. They are different
fits, not reruns of the same config. Effect on the rule: a scorer that separates gold from non-gold
MORE can only raise r(gold) relative to the draws, so the fraction would not be expected to fall
below 0.97 -- but that is an argument, not a measurement; the probe was never run with (a)/(b).
Note the probe's gold row (+0.837 nats/token) sits below A1's quoted +0.95, i.e. the falsifier ran
on the weakest banked phone LM. The lexicon ESCAPE reward is explicitly out of probe (ii) (A5).

## 4. Sample length -- not a sampling or SIL artefact

* Adjacent-repeat rate inside the sampled strings is 0.001 (ep1 raw mean 80.1 vs run-collapsed
  80.0; ep10 57.5 vs 57.5). The draws are therefore NOT inflated by uncollapsed runs that the
  greedy/gold convention would merge; the 80.1 vs greedy 21.5 gap is the flat tau-8 posterior
  itself, at the schedule's stated tau.
* `GoldPhonesJob.ZGSp0hxyd2YP/output/gold.json` contains **no SIL at all** on these 300 tags (raw
  mean 62.9 = SIL-free mean 62.9), so `drop_sil` is a no-op on gold and cannot shorten it
  asymmetrically. Dropping SIL from the draws, whose per-token r is negative, RAISES their reward,
  i.e. works against the rule firing.
* Length does enter the totals (per-token r: gold +0.837, sample -1.423 at ep1), but see 5: the
  per-token variant still clears the bar.

## 5. Fragility of the 0.97 at ep10, and A5

Margins r(gold) - max_g over the 300: 9 negative (min -10.48, all short utterances, gold 3-32
tokens), median +53.63. Only **2 utterances lie within 1 nat of the threshold, and both are already
on the fail side**; the smallest positive margins are +1.47, +1.86, then +6.71, +7.09, +7.57.
Falling to <= 285/300 (<= 95 %) needs 6 more flips, i.e. ~8 nats of movement. Not fragile.
A5 normalisation: the sf term's divisor is the utterance's retained unit count, a positive constant
across that utterance's strings, so dividing by it is order-preserving and the fraction is
unchanged (0.97) -- confirmed. Even the stronger per-TOKEN variant (a divisor that does vary by
string) gives 0.9900 / 0.9933 / 0.9667: above 0.95 at every checkpoint.

## 6. Was the pre-registered rule what was measured?

Yes. A1's r(y) = log p_neural(y) - log p_unigram(y) as a per-string sum, SIL-free, gold vs the max
over the G = 8 draws, 300 dev-other utterances, ctrl_50 ep1/ep4/ep10 at the schedule's tau, G = 8
-- every constant traces to the falsifier bullet (`SAE_4A_prior.md:303-312`) or A1/A4/A5. The
300-utterance subset is a seeded random sample of the sorted dev-other tag pool (`:92-103`), not an
alphabetical head. `gold_gt_greedy_fraction` is banked as a separate column (0.99 at ep10) and is
not the one the rule is read on.

## Caveats (do not change the verdict)

1. **One empty gold string.** Tag `1651-136854-0012` has `gold: []` in `gold.json` itself (an
   upstream gold defect, not a probe bug); by the empty-sum convention r(gold) = 0.0 and it counts
   as a pass with margin +7.57 at ep10. Dropping it: ep10 = 290/299 = 0.9699, ep1/ep4 = 299/299.
   Verdict unchanged at every checkpoint.
2. The rule does not itself pin totals vs per-token; the probe's totals reading is the faithful one
   (A1 states r on a string) and both variants fire.
3. Scope: this falsifies THIS reward (first-pass neural LM minus unigram), not "any scorer". Per
   the standing distinction, the failed gate licenses "not funding sf_50", not "it could not work"
   under the lexicon scorer, which probe (ii) never scored.
