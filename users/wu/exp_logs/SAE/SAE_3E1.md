# SAE §3e.1 — scorer trainability without collapse (ladder D0–D4, `PLAN_3E1.md`)

## State
<!-- Overwritten in place, never appended; deleted at phase close. In-flight runs (job dir + the
question each answers), blockers, next action, proposals for the planner. -->

In flight 2026-08-22:

- **D7.2, the registered label-free admission** (`config/sae_3e1_d7_gan_seqdisc.py`, speech-llm
  `c40655d`), launched 23:16 under manager pid 936497 with a watcher. Ten jobs, all D7.2, verified
  against the built graph before launch: 227 jobs total, exactly 10 unfinished, and the four
  D7.0/D7.1 hashes unmoved, so nothing finished reruns and no policy leg is funded anywhere.
  Clauses 1-2 are `S/d7_admission/D7OnlineAdmissionJob.h0LsMi9zt5aI` (paired 32-draw internal-held
  read); clause 3 is `S/gate_table/PsiGateClauseTableJob.4Z0gb5GgtD2u` over the unchanged frozen
  1,500 external gold-dev rows with the exact D7 control as incumbent, fed by two `PsiTextProbeJob`
  and two `PsiHeldNllJob`; clause 4 is two `PsiScorerParityJob` behind two `PsiAlignRerankJob`.
  The admission job is past its checkpoint asserts and into the GPU loop.
- **D6-PERIODIC/GAN-FROZEN**: leg 7 of 8 running (`T/ReturnnTrainingJob.ZgRzUxDRhajE`), legs 1-6
  finished. Manager pid 1991977, watcher attached.
- **D6-PERIODIC/GAN960-FROZEN** (approach 33): leg 1 `T/ReturnnTrainingJob.ohmLWWmr6Kxe` running,
  manager pid 3514914, watcher attached. Its gate is leg 8 beating this arm's own init 13.11/16.82
  on both splits; matched-leg deltas against GAN-FROZEN are reported and select nothing.
- `sae_3e1_hom` manager also alive (pid 1992923).

**D7-GAN-SEQDISC IS COMPLETE THROUGH D7.1** (approach 32, verdicts 64-65; config
`config/sae_3e1_d7_gan_seqdisc.py`, which registers d7_0 and d7_1 only). The own-infeasible-anchor
amendment (speech-llm `e2a421b`) held: both arms name exactly the four registered train-role rows in
their own `monitors.json` and `train.txt`, 267,175 trained / 14,062 held, digit-identical to the
offline dropcheck and to each other -- the confirmation deferred at relaunch is now closed from the
artifacts, not inferred. Each arm is a single ~14-minute job, 2,361 batches over ten round-robin
shards; the 11.5 h wall cap and shard-boundary resume never came into play, so the earlier
expectation of several resubmits per arm was wrong. Both fixed-final scorers exist
(`.../output/model_final.pt`). Held statistics and the matched-bed evidence are in approach 32.

Correction to an earlier State claim: this manager, like `sae_1g_h4_prelabel_surfaces`, exits on
sisyphus's interactive "All calculations are done" prompt once its graph finishes, and a one-shot
console status then calls several finished jobs `runnable`/`waiting`. A watcher verdict of
`STATUS=STALLED ... work remains` on this config is that artifact; all 49 jobs carry `finished` or
`finished.tar.gz` on disk.

**D8.0 IS COMPLETE AND ITS BINDING CLAUSE PASSES** (approach 34, verdicts 62-63; speech-llm
`889750c`, `a3dd6c7`, operative v3 `3843918`; manager `sae_3e1_d8_0`, all reads finished). The planner's
2026-08-22 clause-(a) ruling is implemented: the structural-infeasibility exclusion now joins `T_i`
from the frozen raw 50 Hz store `S/quantize_states/PackUnitsJob.I0uzRMfUrKWC`, the operative
D8.1a/D8.1b frame, with coverage over the slice's ids asserted; the per-unit prior currency still
divides by each dump's own store. The v2 law-conflict guard retired by construction, its count
survives as a reported diagnostic, and the ruled 5 % safety valve is implemented and did not fire.
On the binding theta_0^G T=0.7 slice: median distinct feasible support **12 of 13** against a
threshold of 3, exclusion **0 of 5,730** scored members, verdict **GO**. The exclusion is empty on
all five slices, so the v2 UNRESOLVED was entirely the pooled-store frame. D8 does not close at
D8.0; D8.1a-b remain gated behind the D7.2 admission verdict and D8.3 still needs its own word.

All three earlier proposals are ruled and discharged: clause (a) reads in the operative frame
(implemented, no new dump needed); the dedup survivor tie rule is ratified and is to be pinned in
the D8.1a weight job's docstring when that job is built; the GAN960 leg-1 scorer provenance is
covered by the registered Disclosed-asymmetry paragraph and needs no plan change.

Verifier follow-ups from the 2026-08-22 round, all applied: `scripts/d7_parity_diag.py` unpacks the
5-tuple again (it would have crashed on a rerun); the D8 code is cited at branch head `a3dd6c7`, not
its pre-amend twin; the collapse tally is labelled CLASSES not groups in both the job docstring and
this log, and the artifact now reports classes and groups separately; approach 34's v2 table carries
the omitted T=0.5 row.

Next action: hold the D7.2 watcher to DONE, then read the four clause artifacts and append the
admission verdict. D7.3, the policy leg, needs its own launch authorization and is in no graph.
D8 has nothing left to build until D7.2 returns a verdict.
§1g H4's pre-label surfaces are COMPLETE (`SAE_1g.md` State, approach 11, verdict 17).

Proposal for the planner, measured while building D7.2 and now reported inside the admission
artifact: the registered "32 stateless donor draws" does not buy 32 independent donors on this
population. A speaker's internal-held rows inside the duration window are few, so an anchor's draws
are sampled with replacement from a pool of typically about four -- mean 3.67 distinct donors over
32 draws on the first 400 held anchors, and exactly 1 for every `nearest_fallback` anchor by
construction. Nothing about the clause changes; the precision comes from the speaker-cluster
resampling and not from the draw count, and the artifact now reports the distinct-donor count per
anchor so the estimate is never read as carrying 32 draws' worth of donor variation.

Proposal for the planner, on D7.2's second gate clause: it requires the candidate's internal-held
per-frame NLL to be no greater than the control's point value, and the two point values D7.1 banked
already order the wrong way (candidate 2.5319 against control 2.5259). Those are one-draw values
from the training jobs, not the D7.2 estimator, and clause 1 orders the other way (candidate mean
`L_online` 26 % lower), so nothing is decided -- but if D7.2 reproduces the NLL ordering, D7 closes
without a policy leg on clause 2 while clause 1 passes. Flagging it now rather than after the run,
since the gate is pre-registered and must not move once the number is in.

Proposal for the planner: none outstanding.

## Approach

**1. AR text-usage gate along the co-trained trajectory** (PLAN.md §3e.1 queue item 2, first half).
The §3c 100 h replay arm (`freeze_ar=False`, the only trainable-scorer run on record) is re-read with
the §2.5 usage gate at each of its own checkpoints: `gate = ln(ppl_shuffled) - ln(ppl_true)` on the
10 h seed dev subset (5000 utts), avunits k500 stream, within-dev derangement at seed 42, p=1.0
history masking — one protocol, only the checkpoint moves. `SaeGrpoModelV1` nests the scorer as
`self.ar`, so each epoch's `ar.`-stripped sub-state is a standalone `SaeTokenLmV1` checkpoint
(`ExtractAvSubmodelJob`, `submodel_prefix="ar."`). ep0 is the frozen AR every loop arm starts from and
is the already-finished `p10` cell of `config_sae_2s_ar_usage_gate_avunits_v1`, reused by asserted job
id, so the anchor predates the question; its CE 5.7444 reproduces that AR's logged dev CE 5.7371.
Interpretation floors on this stream: unit marginal 6.0072, uniform ln 500 = 6.2146.

| point | CE_true | CE_shuffled | gate | vs ep0 |
|---|---|---|---|---|
| ep0 (frozen AR, `ExCoQDKtXAGH` ep50) | 5.7444 | 6.0774 | 0.3331 | 0 |
| ep1 | 6.2045 | 6.5704 | 0.3659 | +0.033 |
| ep2 | 6.1985 | 6.5478 | 0.3493 | +0.016 |
| ep3 | 6.2235 | 6.6268 | 0.4033 | +0.070 |
| ep4 | 6.2226 | 6.7405 | 0.5179 | +0.185 |
| ep5 | 6.2385 | 6.8034 | 0.5650 | +0.232 |
| ep6 | 6.2938 | 6.9149 | 0.6210 | +0.288 |
| ep7 | 6.0770 | 6.6032 | 0.5261 | +0.193 |

**2. Excess-mass suspect vocabulary, label-free** (D0(d)). Rate of each token in the §1d pseudo-text
(28 539 utts, 963 857 tokens) minus its rate in the LibriSpeech LM corpus (803 M tokens); threshold
pre-registered from the mechanism before any table was read: `min_excess = 0.002`, one extra
occurrence per 500 tokens, i.e. about one per utterance at this bed's length — the smallest rate at
which a token can act as a per-utterance filler. A ratio test was rejected in the plan because it
top-ranks rare words; sensitivity: 4 tokens at 0.001, 1 at 0.005.

| word | n_pseudo | rate_pseudo | rate_lm | excess |
|---|---|---|---|---|
| to | 45 561 | 0.047269 | 0.027452 | **0.019817** |
| of | 34 445 | 0.035737 | 0.030868 | 0.004869 |
| buy | 2 984 | 0.003096 | 0.000061 | 0.003035 |
| vary (below threshold) | 1 044 | 0.001083 | 0.000005 | 0.001078 |

**3. D0 mechanism discriminator** (queue item 2, second half). Bias vs noise vs group blindness on
finished artifacts only — 512 tc100 utterances sampled from theta_0^G at G=12/T=0.7, the loop's own
operating point, re-ranked by three psi_align scorers that share the rollout set
(`ReturnnForwardJobV2.J9yA1eYnxwYA`) and the unit stream (`AssignUnitsJob.X8DBup0jQlhR`) and differ
only in which text they were fitted to: `psi_g_tc100` (the loop's own scorer, fitted to the same §1d
pseudo-text theta_0^G was initialized from), `psi_g_seed` (same recipe, 10x less of that text),
`gold_enc50` (10 h gold text — the never-contaminated control that localizes any effect to the
training text rather than to the bed). Labels enter as evaluation only. Both live reward variants are
read: `recon`, and `shaped` = recon + 1.0 * prior/n_units — the dumps normalize the prior per text
token, so the job rebuilds the sum and divides by the utterance's own unit count to restore the live
`lm_prior_norm="units"` term. Bias statistic = group-centred partial effect of the suspect count on
reward with WER as covariate (`beta_ols`; `beta_rank` is its nonparametric twin), positive meaning the
scorer PAYS for the filler at matched WER. Arm-invariant rows (sampling headroom, coverage, and every
selector taken from the shared dump) come out identical across arms, which is the wiring check.

Shared across arms: mean_wer 0.1670, oracle 0.1071 (G=12) / 0.1125 (G=8), greedy 0.1345, 512 groups.
Group contrast — fraction of groups carrying the token that also hold a token-free member: **"to"
0.2334** (467 live groups, mean within-group count std 0.4824), "of" 0.1089, any suspect 0.0922.

| arm | spearman recon | spearman shaped | beta_ols "to" | beta_rank "to" | beta_ols any |
|---|---|---|---|---|---|
| gold_enc50 (control) | 0.5801 | 0.6300 | 0.1673 | 0.1792 | 0.1887 |
| psi_g_tc100 (the loop's) | 0.4959 | 0.5558 | 0.2425 | 0.2634 | 0.2514 |
| psi_g_seed | 0.4696 | 0.5404 | 0.2664 | 0.3219 | 0.2753 |

Selectors, within-group spearman(signal, -WER) with 95 % CI over groups (arm-invariant):
`lm_prior_units` 0.5020 [0.4737, 0.5308], `neg_n_suspect` 0.1855 [0.1510, 0.2186], `n_tokens` -0.0354
[-0.0710, 0.0004], `psi_len_only` 0.0125–0.0354 with the CI straddling zero, `neg_n_oov` undefined
(every row on this bed has n_oov = 0).

**4. Frozen external held pair set, and gate v2 (i)+(ii) read on it** (D1). The §1d student decoded
LibriSpeech dev as well as tc100, so (pseudo-text, enc50 units) pairs exist on 5567 utterances no
scorer in this program trains on; 1500 are taken by a seeded permutation of the id-sorted pool and
never move again, which is what `PsiAlignTrainJob`'s per-candidate 5 % split of its own corpus cannot
be. 1493 are feasible under both the true and the length-matched deranged pairing. All three D0
scorers are read on it, unrepeated and label-free.

| arm | ce_loo (true) | H_uni on these frames | text_explained_loo | usage gate (len-matched) |
|---|---|---|---|---|
| psi_g_tc100 | 2.7198 | 6.0324 | +3.3126 | +3.6853 |
| psi_g_seed | 3.0235 | 6.0324 | +3.0089 | +3.7036 |
| gold_enc50 | 3.1274 | 6.0324 | +2.8939 | +4.2821 |

**5. D1 filler probe battery** (D1). Paired text-side corruptions on the same 1442 held pairs that
survive every pairing's U <= 2T bound: at k = 1, 2, 4 randomly drawn slots, the filler and an LM-drawn
frequent word are written into the SAME slots (substitution) and inserted at the SAME slots
(insertion), and the same slots are deleted; the statistic is the per-utterance increase in `ce_loo`
over the untouched text, bootstrapped over utterances. Substitution asks what the filler costs to
write over a word, insertion what it costs to ADD — and the G-track's degradation is made of
insertions.

| arm | del_1 | sub filler_1 | sub LM_1 | ins filler_1 | ins LM_1 | insertion discount k=1 / 2 / 4 | suspect state mass |
|---|---|---|---|---|---|---|---|
| psi_g_tc100 | 0.3336 | 0.3183 | 0.3219 | **0.0274** | 0.0859 | **0.0584** / 0.1129 / 0.2201 | 1.90 % |
| psi_g_seed | 0.3315 | 0.3010 | 0.3054 | 0.0172 | 0.0735 | 0.0563 / 0.1084 / 0.2199 | 2.19 % |
| gold_enc50 | 0.3941 | 0.3595 | 0.3570 | 0.0261 | 0.0851 | 0.0590 / 0.1056 / 0.2351 | 2.08 % |

Insertion-discount CIs at k=1: psi_g_tc100 [0.0537, 0.0634], psi_g_seed [0.0520, 0.0604], gold_enc50
[0.0539, 0.0639]. The substitution discount is ~0 in every arm (+0.0036 / +0.0044 / −0.0025). Ladder
spearman (severity vs `ce_loo` increase) 0.94 for substitution and deletion, 0.66–0.86 for insertion.

**6. Sampling-side contingency: contrast coverage and steerability vs temperature** (D0 coverage
co-requirement, reproduced as a logged table). The D0 dump already carries T = {0.3, 0.5, 0.7, 0.9,
1.0} at G=12, so this is a re-read: coverage is the fraction of "to"-carrying groups holding a
"to"-free member, steerable additionally requires that member's live shaped reward to beat the group
mean. Coverage is arm-invariant; steerability is not, and WER enters as evaluation only.

| T | coverage "to" | steerable (psi_g_tc100) | steerable / coverage | mean WER | oracle WER |
|---|---|---|---|---|---|
| 0.3 | 0.1359 | 0.1091 | 0.803 | 0.1386 | 0.1074 |
| 0.5 | 0.1645 | 0.1360 | 0.827 | 0.1467 | 0.1039 |
| 0.7 | 0.2334 | 0.1949 | 0.835 | 0.1670 | 0.1071 |
| 0.9 | 0.5270 | 0.3382 | 0.642 | 0.2994 | 0.1496 |
| 1.0 | 0.8212 | 0.5343 | 0.651 | 0.5153 | 0.2829 |

**7. D2 round-0 pseudo-text repair** (D2, corpus side). Rates of the three excess-mass suspects are
matched to the LibriSpeech LM corpus by removal only, with a per-utterance multiplicity cap read off
the LM corpus at matched utterance length (q99: 3 for "to" in the 20–30-token bucket). 60.6 % of
utterances are edited and the corpus loses 2.94 % of its tokens; no utterance is emptied, and the
repaired corpus differs from the contaminated one only where a token was removed.

| word | rate before | rate after | rate in LM corpus | removed by cap / by rate |
|---|---|---|---|---|
| to | 0.047269 | 0.027452 | 0.027452 | 1001 / 18 879 |
| of | 0.035737 | 0.030868 | 0.030868 | 63 / 5 506 |
| buy | 0.003096 | 0.000061 | 0.000061 | 18 / 2 909 |

**8. D2 matching-aware contrastive term** (D2, scorer side). A GAN-CLS/MMI denominator over in-batch
negatives, `-log p(u_i|z_i) / sum_j p(u_i|z_j)`, where the negatives are other rows' texts scored
against the same audio, so text-blindness is unreachable by construction; the encoder output is
reused and only the DP repeats, and the term activates once the alignment prior has annealed off
(epoch 5 of 30). Four arms separate corpus from mechanism at otherwise identical hyperparameters,
one variable each against psi_g_tc100: `d2_rate` (repaired corpus), `d2_contrast` (weight 1),
`d2_both`, and `d2_states` (chars_per_state 1.5 -> 0.5, the frames-per-state term of conclusion 12).
The control corpus is asserted byte-equal to the one psi_g_tc100 was fitted to, id order included,
so "the only difference is the repair" is checked rather than claimed. All four arms ran the full 30
epochs and each is read at its own best-held epoch; every ce_loo-derived column is
segmenter-dependent, so `d2_states`' entries in those columns are NOT comparable to the cps-1.5 rows
and are marked (*).

| arm (corpus / weight / cps) | best ep | held ce_loo | ins. disc. k1 | ladder filler_ins | beta_to | spearman | steerable | susp. mass % |
|---|---|---|---|---|---|---|---|---|
| psi_g_tc100 — contaminated / 0 / 1.5 (incumbent) | 9 | 2.7198 | +0.0584 | 0.6552 | 0.2425 | **0.4959** | 0.1949 | 1.902 |
| d2_rate — repaired / 0 / 1.5 | 11 | 2.7195 | +0.0595 | 0.6180 | **0.2232** | 0.4931 | 0.1906 | **1.724** |
| d2_contrast — contaminated / 1 / 1.5 | 28 | **2.7139** | **+0.0558** | 0.6820 | 0.2469 | 0.4808 | **0.2013** | 1.881 |
| d2_both — repaired / 1 / 1.5 | 28 | 2.7332 | +0.0619 | 0.6652 | 0.2437 | 0.4801 | **0.2013** | 2.053 |
| d2_states — contaminated / 0 / 0.5 | 11 | 2.1278 (*) | +0.1475 (*) | **0.8540** | 0.2389 | 0.4878 | 0.1906 | 3.268 (*) |

The `ins. disc. k1` column is the FREQUENCY-DRAWN discount and is kept only because approach 9's
original rule names it; it is state-count-confounded and approach 10 replaces it. 2 T/U on the held
set is 9.77 at cps 1.5 and 3.92 at cps 0.5. `beta_to`, `spearman` and `steerable`
are the D0-dump re-reads at T=0.7 with every arm re-ranking the SAME rollouts, so those three columns
are cross-arm comparable even for `d2_states`; contrast coverage itself is arm-invariant at 0.2334,
so `steerable` moves only through the scorer. All four candidates PASS `PsiScorerParityJob` at
max |online - offline| = 0, and all four clear the three G3 bars of `PLAN_3A` §6 (gap_true >= 0.0248,
spearman >= 0.17, audio-margin CI excluding zero; margins +0.146 to +0.154, all CIs overlapping).

**9. D3 frozen-repaired G-track control arm** (D3). The winner's scorer is frozen into the same
`config_sae_3a_gan_loop_960h_v1.baseline` that builds the arms it controls, so bed, data and
schedule differ in one input only; bar 2 is the suspect share of sclite insertions, read off the
recogniser's own alignment at four sub-epochs. The winner is selected by a rule fixed before the D2
read: an arm is eligible only if its held CE_loo is below the unit-marginal floor 6.03, its
text_explained_loo is not below the pre-loop floor, and its corruption-ladder spearman is not below
psi_g_tc100's; among eligible arms the winner is the one that most reduces the insertion discount
(psi_g_tc100: 0.0584), ties broken by the D0 rollout beta at matched WER. If no arm reduces the
discount by more than the bootstrap CI half-width (~0.005) there is no winner and D3 is not funded
from D2 — the fallback is the planner's call. Gate v2 (i)'s round-to-round improvement clause is not
used for eligibility because the held text is unrepaired pseudo-text, which asks a repair arm to
model the defect it removed; it is reported alongside. Three of four sub-epochs are in (`psid2_contrast`
= the frozen `d2_contrast` scorer, against the same arm on the incumbent `psi_g_tc100`); the
insertion columns are dev-clean, read off the recogniser's own alignment.

| arm | sub-ep | psi | dev-clean | dev-other | ins | `to` ins | suspect share |
|---|---|---|---|---|---|---|---|
| shaped | 1 | incumbent | 13.42 | 18.75 | 3927 | 3234 | 0.844 |
| shaped | 1 | d2_contrast | 13.57 | 19.69 | 4109 | 3418 | 0.850 |
| shaped | 2 | incumbent | 13.91 | 18.91 | 4151 | 3539 | 0.871 |
| **shaped** | **2** | **d2_contrast** | **12.68** | **17.57** | **3616** | **3043** | **0.862** |
| recon | 1 | incumbent | 23.91 | 30.55 | 7105 | 4870 | 0.698 |
| recon | 1 | d2_contrast | 24.24 | 29.75 | 5184 | 2390 | 0.478 |
| recon | 2 | incumbent | 31.46 | 36.89 | 7021 | 3817 | 0.556 |
| recon | 2 | d2_contrast | 27.04 | 32.89 | 5343 | 2096 | 0.408 |
| shaped | 3 | incumbent | 13.49 | 18.81 | 4154 | 3484 | 0.857 |
| shaped | 3 | d2_contrast | 13.54 | 18.56 | 3929 | 3346 | 0.871 |
| recon | 3 | incumbent | 33.54 | 39.74 | 8724 | 4627 | 0.542 |
| recon | 3 | d2_contrast | 32.94 | 37.91 | 5623 | 1834 | 0.341 |

**10. State-matched control pool, and the D2 selection read on it** (D1 build item (b), D2 admission).
The LM control is redrawn from a pool holding the filler's own emitting-state count under each arm's
own segmenter — 57 one-state words at cps 1.5 and 51 four-state words at cps 0.5, of the same 6 472
above the rate floor — where the
frequency-drawn pool averages 2.70 states against the filler's one (8.16 against four at cps 0.5), and
per-utterance `ce_loo` is dumped so every cross-arm number below is a PAIRED difference on the 1442
utterances all seven arms could score. The frequency-drawn pairings are drawn from their own generator
and reproduce the pre-extension jobs to the last digit (0 of 7 arms differ on any statistic), so this
is a control added beside the old one, not a re-measurement of it.

| arm | matched ins. disc. k1 | paired vs incumbent | k4 | paired vs incumbent | ladders worse (of 5) |
|---|---|---|---|---|---|
| psi_g_tc100 (incumbent) | +0.0172 | — | +0.0561 | — | — |
| gold_enc50 (10 h-true control) | **+0.0031** | **-0.0141 [-0.0174, -0.0108]** | +0.0094 | -0.0468 [-0.0536, -0.0401] | 3 |
| psi_g_seed | +0.0078 | -0.0094 [-0.0123, -0.0067] | +0.0299 | -0.0262 [-0.0318, -0.0207] | 1 |
| d2_rate | +0.0154 | -0.0018 [-0.0048, +0.0011] | +0.0555 | -0.0007 [-0.0057, +0.0044] | 3 |
| **d2_contrast** | **+0.0082** | **-0.0090 [-0.0119, -0.0062]** | **+0.0323** | **-0.0238 [-0.0288, -0.0189]** | **0** |
| d2_both | +0.0097 | -0.0075 [-0.0101, -0.0050] | +0.0351 | -0.0210 [-0.0260, -0.0163] | 0 |
| d2_states | +0.0125 | -0.0047 [-0.0095, -0.0001] | +0.0419 | -0.0142 [-0.0230, -0.0049] | 0 |

**11. The acceptance rule as a job, and the D4 admissibility instruments** (D4 prereqs a/b/c). The
gate v2 clauses now compute from the per-arm `items.json` + `held.json` at a pinned seed and resample
count (`PsiGateClauseTableJob`), printing both readings of the ladder floor rather than choosing one;
the selector block of the D0 discriminator gained a filler-affinity twin (partial beta of the suspect
count on each CURATION view at matched WER, group-bootstrap CI), opt-in and hash-excluded so the
audited D0/D2 tables keep their ids, and it scores `ar_recon` -- the G-track AR's own reward, carried
in every re-rank dump as `recon_incumbent` and never read as a selector before -- as the one
audio-conditioned view on offer. `SaeGrpoModelV1` gained `av_checkpoint_prefix`, which imports a
previous round's policy out of the loop's own state_dict and leaves psi at the newly accepted scorer.

The clause table on the seven finished arms reproduces the audited D2 verdict (ladders-worse
3/1/3/0/0/0; point reading -> `d2_states`, paired-CI reading -> `d2_contrast`), and the import
override carries 719 `av.*` keys out of a live 960 h loop checkpoint -- name- and shape-identical to
theta_0^G's own AV SFT checkpoint -- while dropping its 82 `psi.*` keys.

Candidate curation views under both pre-registered bars (arm-invariant rows; `to` for (f)):

| view | (e) spearman(signal, -WER) | (f) beta on suspect count at matched WER | admissible |
|---|---|---|---|
| `lm_prior_units` | **0.5020** [0.4737, 0.5308] | **-0.0937** [-0.1405, -0.0449] | yes |
| `neg_n_suspect` | 0.1855 [0.1510, 0.2186] | -0.9232 (by construction) | yes |
| `ar_recon` (the G-track AR's own reward) | 0.0944 [0.0571, 0.1321] | **+0.0716** [0.0102, 0.1214] | no -- (f) |
| `psi_len_only` | 0.0354 [-0.0018, 0.0722] | -0.0031 [-0.0690, 0.0465] | no -- (e) |
| `n_tokens` | -0.0354 [-0.0710, 0.0004] | +0.3180 [0.2610, 0.3835] | no -- both |

**12. Refresh round 1: a curated pool from theta_0^G's own rollouts, and a scorer refit on it** (D4).
One fresh dump of theta_0^G over all 28539 pseudo-text utterances at the loop's own T=0.7 and G=12
(a whole-bed pass at the finished probes' batching measured 9.5 h against the 11.5 h cap and the
forward job has no resume; the step is latency-bound in the decode, not throughput-bound in the
batch, so 8 utterances per step rather than 4 buys 1.68x and G stays at the loop's value), curated
by two-view agreement with both advantages positive and one member per utterance, on top of the rate-repaired round-0 corpus as the anchor at a floored 50 %
share. A curated pool holds an utterance twice against one unit stream, and both rows are pairs the
NLL term maximizes, so the matching-aware term drops a same-utterance negative exactly as it drops a
structurally impossible one — without that mask 5.48 % of rows per epoch contrast a reading of their
own audio, and since `_batches` sorts by (T, U) the twins land adjacent and the shorter always wins,
which is the length detector the (T, U) bucketing exists to rule out. The candidate is the
`d2_contrast` recipe refit from scratch on anchor + curated, judged by
the same frozen held set, state-matched probe battery and clause table the D2 arms were judged by,
with gate v2 (i) floor-only because a refresh candidate is changed-text by construction.

`d2_contrast` is the same recipe on the uncurated round-0 corpus, so its column is what curation is
worth. Paired on the 1442 utterances every arm scores; the discount reduction is against the
incumbent, ladder deltas likewise.

| frozen held set, pinned ep28 | incumbent `psi_g_tc100` | `d2_contrast` (uncurated) | round-1 `r1` |
|---|---|---|---|
| state-matched insertion discount k=1 | 0.0172 | 0.0082 (-0.0090 [-0.0118, -0.0062]) | **0.0064** (-0.0108 [-0.0140, -0.0077]) |
| k=2 | 0.0323 | 0.0180 (-0.0143) | 0.0122 (-0.0200) |
| k=4 | 0.0561 | 0.0323 (-0.0238) | 0.0140 (-0.0422) |
| held ce_loo (H_uni 6.0324) | 2.7198 | 2.7139 | 2.7168 |
| filler_ins ladder spearman | 0.6552 | 0.6820 (+0.0268 [+0.0065, +0.0472]) | 0.6771 (+0.0219 [-0.0006, +0.0440]) |
| ladders nominally / significantly worse | -- | 2 / 0 | 4 / 0 |
| eligible, point / CI reading of the floor | -- | no / yes | no / yes |

**13. Error anatomy along the collapsing trajectory** (D5(a)-1). The four rates are recomputed from
sclite's own counts against the reference length for both dev sets at ep0-ep4 of the 100 h
seed-replay joint-AR arm, alongside the hypothesis/reference length ratio, the top inserted words and
the suspect set's share of all insertions. No new decoding: the ten finished `ScliteJob` report dirs
are pinned by absolute path.

| point | set | WER | %Corr | %Del | %Ins | hyp/ref | n_ins | susp share | top inserted |
|---|---|---|---|---|---|---|---|---|---|
| ep0 | dev-clean | 16.91 | 89.07 | 4.46 | 5.98 | 1.015 | 3255 | 0.066 | the:222, and:221, to:117 |
| ep0 | dev-other | 20.64 | 87.31 | 3.49 | 7.96 | 1.045 | 4054 | 0.063 | the:209, a:190, and:169 |
| ep1 | dev-clean | 18.79 | 90.42 | 2.86 | 9.21 | 1.064 | 5010 | 0.047 | the:243, and:185, to:120 |
| ep1 | dev-other | 25.59 | 87.89 | 2.52 | 13.48 | 1.110 | 6868 | 0.045 | the:299, and:224, a:185 |
| ep2 | dev-clean | 23.94 | 90.49 | 2.88 | 14.42 | 1.115 | 7846 | 0.056 | the:449, and:352, of:229 |
| ep2 | dev-other | 29.22 | 88.04 | 2.57 | 17.26 | 1.147 | 8793 | 0.051 | the:481, and:320, to:242 |
| ep3 | dev-clean | 42.17 | 90.77 | 2.69 | 32.94 | 1.302 | 17919 | 0.056 | the:1011, and:843, of:541 |
| ep3 | dev-other | 50.00 | 88.36 | 2.27 | 38.36 | 1.361 | 19546 | 0.057 | the:1080, and:789, of:553 |
| ep4 | dev-clean | 46.71 | 91.55 | 2.14 | 38.26 | 1.361 | 20814 | 0.057 | the:1183, and:1031, of:612 |
| ep4 | dev-other | 51.42 | 88.92 | 1.96 | 40.34 | 1.384 | 20552 | 0.056 | the:1111, and:994, to:595 |

**14. Scorer allegiance grid** (D5(a)-2). CE(units | conditioning text) in nats/unit under each ep-k
scorer for six texts on one 5000-utterance seed dev subset — gold, theta_0's decodes (`dec0`) and the
arm's own decodes at ep1-ep4 — so only the scorer and the conditioning text move and the gold column
is approach 1's `CE_true` column by construction. The policy decodes are merged from both dev sets'
`search_out`, lowercased and NFKD-folded, three utterances carrying accents that the corpus reader
cannot decode.

| scorer | dec0 | gold | dec1 | dec2 | dec3 | dec4 | self_pref | follow |
|---|---|---|---|---|---|---|---|---|
| ep0 | 5.7583 | **5.7444** | 5.7520 | 5.7516 | 5.7643 | 5.7659 | -0.0139 | 0.0000 |
| ep1 | 6.2131 | 6.2045 | 6.1926 | 6.1788 | 6.1490 | **6.1426** | +0.0119 | 0.4406 |
| ep2 | 6.1946 | 6.1985 | 6.1721 | 6.1579 | 6.1187 | **6.1141** | +0.0407 | 0.4063 |
| ep3 | 6.2331 | 6.2235 | 6.2017 | 6.1845 | 6.1374 | **6.1316** | +0.0861 | 0.3731 |
| ep4 | 6.2380 | 6.2226 | 6.1906 | 6.1703 | 6.1052 | **6.0959** | +0.1267 | 0.3300 |

`self_pref` = CE(gold) - CE(own decodes) under one scorer; `follow` = CE(own decodes | this scorer) -
CE(own decodes | ep0). Floors on this stream: unit marginal 6.0072, uniform ln 500 = 6.2146.

**15. Ranking-vs-oracle with the policy pinned** (D5(a)-3). The reward-rank probe re-run five times
with theta_0 as the sampling policy in every cell and only the scorer swapped to the ep-k extraction,
so all five cells re-rank the same rollouts and eta is attributable to the scorer alone; the ep0 cell
is the finished `theta0_avunits_p10` job, asserted by job id as a free wiring anchor. Shared across
all five at T=0.7: mean WER 0.1076, oracle 0.0316, greedy 0.0525.

| scorer | recon @0.7 | std_wg @0.7 | spearman @0.7 | sel_wer @0.7 | eta @0.3 | eta @0.5 | eta @0.7 | eta @1.0 |
|---|---|---|---|---|---|---|---|---|
| ep0 | -5.7331 | 0.01806 | 0.3138 | 0.0905 | +0.0195 | +0.1272 | **+0.2246** | 0.7594 |
| ep1 | -5.9792 | 0.01936 | 0.1754 | 0.1166 | -0.3288 | -0.1108 | **-0.1185** | 0.3557 |
| ep2 | -5.9721 | 0.02078 | 0.1905 | 0.1139 | -0.2300 | -0.1110 | **-0.0831** | 0.3260 |
| ep3 | -5.9888 | 0.02254 | 0.1903 | 0.1183 | -0.0792 | -0.1873 | **-0.1418** | 0.3141 |
| ep4 | -5.9863 | 0.02593 | 0.1203 | 0.1288 | -0.2374 | -0.2877 | **-0.2792** | 0.2837 |

**16. The fork point for the three update-rule arms, label-free** (D4'/D5(b) step 1). The standing
selection rule (dev reward = recon + 1.0 * lm_prior from the arm's own `learning_rates`) is combined
with a health screen computed on the arm's own dev hypotheses and nothing else: words per utterance,
and the ABSOLUTE count of the minimal-state class {and, but, i}, each required to sit within a
pre-registered 10 % of the window minimum over the four sub-epochs that existed at fork time. The
two dev sets are pooled (5567 utterances, fixed across sub-epochs, so words/utt is the speaking-rate
screen up to the constant total duration); no WER enters the job, and the fork epoch is a config
constant the job asserts against.

| ep | dev reward | words/utt | d_len | and | but | i | min-state | d_cls | screen | dev WER (confirm) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -3.28110 | 18.920 | +0.00 % | 3596 | 646 | 1466 | 5708 | +0.00 % | pass | 6.28 / 10.25 |
| 2 | -3.25726 | 19.049 | +0.68 % | 3623 | 645 | 1496 | 5764 | +0.98 % | **pass** | **5.34 / 9.50** |
| 3 | **-3.24568** | 19.320 | +2.11 % | 4239 | 1179 | 2128 | 7546 | +32.20 % | VETO | 6.56 / 11.15 |
| 4 | -3.25140 | 19.357 | +2.31 % | 4698 | 1380 | 1826 | 7904 | +38.47 % | VETO | 6.89 / -- |

**17. The joint-psi control arm on the best bed** (D5(b)-b). One knob off the fork checkpoint:
`train_psi=True`, so psi's per-frame NLL on all G sampled texts joins the shared optimizer at ce
scale 1.0 with no in-loop contrastive term, and everything else is the parent's (shaped lam_lm 1.0
units-normed, T=0.7, `partition_epoch` 10, `keep_epochs` all). The learning rate continues the
parent's cosine rather than restarting it — `epoch_offset` evaluates the parent's curve at
epoch + 2 with the parent's 10-epoch span, reproducing its ep3-8 values exactly — and the arm runs
6 sub-epochs, then stops regardless of trajectory. Forensics are instrumented during the run, one
row per sub-epoch plus the fork as ep0: CE_true, the length-matched derangement contrast and the
unit-marginal floor on a frozen 1500-pair gold dev set, then the same job with the conditioning text
replaced by that sub-epoch's own decode of the same frames. The joint backward does not fit at the
parent's batching, so this arm runs `batch_size` 1e6 / `accum_grad_multiple_step` 2 against the
parent's 2e6 / 1: the 2e6-frame effective batch and the updates per sub-epoch are preserved and
`max_seqs`, `group_size`, `max_seq_length`, schedule and `partition_epoch` are untouched, but the
per-update gradient is the mean of two half-batches rather than one full batch.

| sub-ep | arm | dev-clean / dev-other | insertions dc / do | substitutions dc / do |
|---|---|---|---|---|
| 1 | joint | **5.12 / 9.27** | 385 / 630 | 2029 / 3641 |
| 1 | frozen control (parent sub-ep 3) | 6.56 / 11.15 | — | — |
| 2 | joint | 17.35 / 21.97 | 6114 / 6450 | 2952 / 4426 |
| 2 | frozen control (parent sub-ep 4) | 6.89 / 11.31 | — | — |

The frozen arm's insertions over its whole post-peak stretch are 1182-1415 dc / 1592-1794 do
(`SAE_0d.md` c13), which is the band both joint rows are read against.

**18. Refresh round 1 on the best bed** (D4', steps 2-3). The rollout source is the fork policy
itself at the loop's own operating point (T=0.7, G=12) over all of tc100, and it is ranked by this
bed's own psi rather than by a token-LM AR, which does not exist here — the scorer swap is the only
way `recon` in the dump means what it means everywhere else in this log. The suspect set is
re-derived on this bed instead of carried over: excess mass of the fork policy's own dev decodes
against the same LM corpus at D0's pre-registered `min_excess` 0.002, so whether the minimal-state
class {and, but, i} falls out of a label-free derivation is a reported check rather than the target.
That class stays a MONITOR and never enters the curation views, because it was found by counting
insertions against references. Steps 4-5 are wired and inert behind an unset `CURATION_VIEWS`: the
views follow this bed's own admissibility table, not the G-track's. The incumbent half of the gate
battery (held-NLL and the state-matched probes on the frozen 1500-pair gold seed-dev set that D5(b)
also reads, plus the online/offline parity check on the dump's own psi column) runs ahead of it.

The label-free derivation returns an EMPTY set here (largest excess "and" 0.00135 against the
pre-registered 0.002; 0.001 admits that word alone, 0.005 none), so the table below is the incumbent
battery's instead: the round-0 gold psi on this bed, 1443 of 1500 frozen gold seed-dev pairs, ce_loo
2.7560 on the untouched text against a unit marginal of 6.0332, held ce_loo 2.7614.

| edit at k=1, delta ce_loo (>0 = the scorer charges) | minimal-state word | frequency-matched LM word | discount (LM - minimal-state), 95% CI |
|---|---|---|---|
| INSERTION | +0.0693 | +0.0902 | **+0.0209** [+0.0164, +0.0250] |
| substitution | +0.4051 | +0.3892 | -0.0159 [-0.0228, -0.0091] |
| deletion (no filler twin) | +0.4295 | -- | -- |
| position-matched substitution | +0.4051 | +0.3913 | -0.0138 |
| ladder monotonicity (k = 0,1,2,4) | ins 0.658 | sub 0.735-0.784 | -- |
| suspect-state alignment mass (gate v2 iii) | 2.6838 % of 497 767 frames | -- | -- |

**19. Round 1 refits on the whole pool, selection removed** (D4', steps 4-5). No admissible curation
view exists on this bed (approach 18, c30), so the planner amended round 1 to refit on the anchor plus
EVERY greedy decode instead of a selected subset: 2 849 gold seed pairs repeated 11x beside 28 539
one-per-utterance argmax decodes of the fork policy, 59 878 rows at a 52.3 % anchor share, split by
utterance so a repeated anchor cannot land on both sides of the internal held-out. The recipe is the
incumbent gold-psi one with the D2-winner contrastive term on; the greedy rows come from the existing
dump rather than a second forward job, since its `greedy` kind already is one deterministic argmax
decode per utterance. Batching is widened 8x (`max_batch` 32 -> 256, cells 3e6 -> 24e6) after
measuring the DP to be launch-bound rather than FLOP-bound, which is a SECOND difference from the
loop's frozen scorer and is recorded as a confound, not absorbed.

| statistic | set | psi0_gold (ep 9) | round-1 uncurated (ep 30) |
|---|---|---|---|
| held ce_loo | frozen gold seed-dev, 1 493 pairs | 2.7614 | **2.6432** |
| text_explained_loo (gate v2 ii) | same | +3.2718 | +3.3900 |
| usage gate, length-matched derangement | same | +4.6548 | +5.2952 |
| unit marginal on those frames | same | 6.0332 | 6.0332 |
| within-group spearman | fork dump, 28 538 groups at T=0.7 | +0.3399 | **+0.3621** |
| eta | same | +0.2599 | +0.2663 |
| internal held NLL/frame at the pin | own corpus, not cross-comparable | -- | 2.8122 |

**20. D6 -- structural insertion repair, three rungs on one corpus.** Insertion is under-priced by the
topology, not by the vocabulary (c27, and c28's 7.7 % says corpus repair cannot reach it), so the
three rungs go at the arcs: (1) OFFLINE PRICE STEERING re-scores the frozen incumbent on the same
corruption draw the D1 battery uses, sweeping a renormalized bias against the skip arc and a
minimum-duration charge of `dur_cost` nats per frame a state falls short of `d_min` (silence exempt,
carried exactly in the DP as a frames-held axis); (2) CORRUPTION-TRAINED ARC PRICES refits with a
hinge demanding an inserted LM-drawn word cost at least half of what deleting a word costs on the SAME
row -- scale-free, so no nats constant is chosen, and the deletion side is detached so the term can
only make insertion dearer; (3) MIN-DURATION TOPOLOGY refits with every content symbol split into
`d_min` states and the skip arc masked wherever it would cross one, which lives in the model config so
a checkpoint carries its own topology and every scorer follows it. Rungs 2-3 train on the IDENTICAL
corpus as approach 19 and are read against it, which keeps the corpus axis and the recipe axis each
single-variable; rung 3 waits on rung 1's feasibility statistic (the share of pairs with fewer frames
than `d_min` times their content states), reported there on both the held set and the refit corpus.


| arm (all on the round-1 corpus) | held ce_loo | ins_1 | del_1 | ins/del | mono(ins) | matched ins discount k1 | spearman | eta |
|---|---|---|---|---|---|---|---|---|
| psi0_gold, round-0 incumbent | 2.7614 | +0.0693 | +0.4295 | 0.161 | 0.658 | +0.0078 | +0.3399 | +0.2599 |
| r1_uncurated, the comparator | 2.6432 | +0.0763 | +0.4948 | 0.154 | 0.760 | +0.0094 | +0.3621 | +0.2663 |
| rung 2, corruption margin | 2.5943 | +0.1675 | +0.5454 | 0.307 | 0.692 | +0.0578 | +0.4016 | +0.3482 |
| rung 3, min-duration d_min=2 | **2.1620** | **+0.1985** | +0.5759 | 0.345 | **0.853** | +0.0141 | **+0.4357** | +0.3296 |
| rungs 2+3 combined | 2.1768 | +0.4964 | +0.7106 | 0.699 | 0.785 | +0.3150 | +0.4441 | +0.3392 |

`mono(ins)` is the `filler_ins` ladder's monotone fraction, the clause (iv) statistic (the sub/del band
runs 0.71-0.80 across these arms); the D6 bars are spearman/eta not below the comparator's, held ce_loo
within +0.05 of it, ins_1 >= +0.14 growing in k, and mono(ins) out of last place.

**21. D6 swap-in -- the min-duration scorer as the live reward, on both beds.** The rung-3 checkpoint
replaces the incumbent psi in the reward and nothing else moves. (a) BEST BED: the fork policy
continues from the same sub-epoch 2 state on the parent's own cosine tail for the remaining 8
sub-epochs, same shaped reward at T=0.7, same 960 h slices, same batching -- so the free frozen
continuation that already ran those sub-epochs with the incumbent psi is the control at matched
points. Read as dev WER plus both arms' sclite error decomposition in absolute insertion counts, the
gate's pre-registered in-loop confirmation being that the control's sub-epoch-3 regression (5.34/9.50
at the fork to 6.56/11.15 one sub-epoch later, never recovered through sub-epoch 10) shrinks.
(b) G-TRACK: the topology transfers, checkpoints do not, so the G-track round-1 refresh recipe is
refit at `min_dur=2` on that bed's own round-1 curated corpus, single-variable against the
`min_dur=1` refit, and read with the four D6 clauses on G-track instruments. Both arms there gain the
G3 re-rank and the online/offline parity check the G-track round 1 never carried -- clause (i)'s
within-group ranking half has no other instrument, so the comparator had to be re-ranked too.

Arm (a), all eight sub-epochs (arm sub-ep k = parent global sub-ep k+2), plain sclite WER, with
the control read at every matched point and the dev-other insertion counts both arms are separated by:

| arm sub-ep | global sub-ep | swap-in dev-clean / dev-other | frozen control at the same point | dev-other insertions, swap-in / control |
|---|---|---|---|---|
| 1 | 3 | **4.68 / 8.64** | 6.56 / 11.15 | — |
| 2 | 4 | **4.61 / 8.98** | 6.89 / 11.31 | — |
| 3 | 5 | **4.61 / 9.03** | 6.32 / 11.03 | — |
| 4 | 6 | 5.01 / 9.51 | 6.54 / 11.16 | — |
| 5 | 7 | **4.70 / 9.12** | 6.69 / 11.03 | — |
| 6 | 8 | 5.08 / 9.73 | 5.97 / 10.66 | 1098 / 1640 |
| 7 | 9 | **4.80 / 9.39** | 6.49 / 11.31 | 952 / 1954 |
| 8 | 10 | **4.73 / 9.31** | 6.46 / 11.41 | 933 / 1964 |

**22. D6-PERIODIC -- the min-duration scorer refit at every sub-epoch boundary, best bed.** Approach
21a's whole gain landed in its first post-swap sub-epoch and the frozen scorer bought nothing after
it, so this arm re-forks from the same parent sub-epoch-2 checkpoint and repeats one unit at every
boundary from 3->4 on: decode the tc100 refresh corpus with the round-1 recipe unchanged (gold anchor
at its 50 % floor plus one greedy decode per utterance, only the decoding checkpoint varying), refit
`d_min=2` from scratch on the CUDA path, read the four pre-registered clauses against the last
ACCEPTED scorer on the standing frozen instruments (same corruption draw, same held pair set, same
fork re-rank dump), and swap on pass or keep on fail. Everything else is 21a point for point -- same
fork, same cosine tail evaluated at the parent's own epoch index (verified equal to the control's
logged rates at parent sub-epochs 8 and 10), same shaped reward at T=0.7, same 960 h bed at the
parent's partition size, and the control's own two batching regimes (2e6 through parent sub-epoch 7,
then 1e6 with `accum` 2) -- so 21a is the control for free and the scorer's recency is the only
variable. Two differences it cannot avoid, both forced by one sisyphus job per sub-epoch: the bed
partition moves into the graph as round-robin shards (the RETURNN epoch counter resets in every leg,
which would otherwise train all eight on the same tenth of the bed), and Adam restarts at every
boundary against the control's twice. The acceptance clauses below are the arm AS FIRST RUN; the
verdicts are recorded here because the user removed the acceptance step from every gold-seeded
periodic arm on 2026-08-18 and its jobs were deleted, so this table is the only surviving record of
what it decided.

| leg / boundary | dev-clean / dev-other | (i) rank quality | (ii) held likelihood | (iii) insertion price | (iv) corruption ladders | verdict |
|---|---|---|---|---|---|---|
| 1 | 4.97 / 8.88 | -- no boundary before leg 1 -- | | | | |
| 2 | 4.64 / 8.68 | pass | pass | **fail** | pass | keep the scorer in use |
| 3 | 4.93 / 8.71 | pass | pass | **fail** | pass | keep |
| 4 | 5.37 / 10.81 | pass | pass | pass | pass | keep -- the two-consecutive-failure stop rule had already fired |
| 5 | 4.89 / 9.23 | pass | pass | **fail** | **fail** | keep |

Because every verdict was KEEP, all five legs ran the SAME round-1 scorer as the one-shot swap arm
did, which makes each leg a PAIRED replicate of that arm at its own global sub-epoch -- and the
paired difference, not the spread across legs, is the run-to-run measure: the legs are successive
segments of one trajectory on a decaying schedule, so 4.64-5.37 and 8.68-10.81 across them is mostly
trajectory shape (the one-shot arm's own dev-other walks 8.64, 8.98, 9.03, 9.51, 9.12 over the same
positions and BOTH arms bump at global sub-epoch 6). Matched-point absolute differences against it:
dev-clean 0.29 / 0.03 / 0.32 / 0.36 / 0.19 and dev-other 0.24 / 0.30 / 0.32 / 1.30 / 0.11, i.e. a
maximum of 0.36 / 1.30 and a median of 0.29 / 0.30 over five paired points. A single matched-point
claim on this bed has to clear the maximum; a consistent-sign difference over four or more matched
points reads against the median.

The user then removed the scorer-statistic gate and relaunched. The completed ungated prefix is:

The two primary 10 h-init anchors are separated from the trajectory. “Best” selects the checkpoint
with lowest dev-other WER and reports its paired dev-clean value:

| 10 h-init anchor | dev-clean / dev-other | operating point |
|---|---|---|
| AV SFT, no loop: adapted-donor theta_0' | 11.43 / 15.54 | 10 h AV SFT, epoch 50 |
| best previous frozen-scorer loop | **4.68 / 8.64** | D6 one-shot d_min=2 scorer swap, scorer then frozen, global sub-epoch 3 |

The older incumbent-scorer loop's 5.34 / 9.50 fork is retained as a secondary historical anchor,
but it is not the best previous frozen-loop result.

| ungated leg / global sub-ep | fresh periodic dev-clean / dev-other | one-shot frozen scorer | frozen control | dev-other S / D / I, fresh |
|---|---|---|---|---|
| 1 / 3 | 4.97 / 8.88 | 4.68 / 8.64 | 6.56 / 11.15 | 3572 / 471 / 479 |
| 2 / 4 | 4.65 / 9.02 | 4.61 / 8.98 | 6.89 / 11.31 | 3593 / 395 / 607 |
| 3 / 5 | 5.28 / 9.27 | 4.61 / 9.03 | 6.32 / 11.03 | 3551 / 561 / 610 |
| 4 / 6 | 6.05 / 10.56 | 5.01 / 9.51 | 6.54 / 11.16 | 3553 / 476 / 1350 |
| 5 / 7 | 7.42 / 12.68 | 4.70 / 9.12 | 6.69 / 11.03 | 3597 / 376 / 2489 |

These are a prefix, not an endpoint: leg 6 is submitted but pending for maintenance and legs 7-8
do not yet exist. `S / D / I` are sclite substitution, deletion and insertion counts on dev-other;
the late WER loss is almost entirely insertion growth, not a drift in substitutions.

**23. HOM-0a -- how much of the pseudo-label corpus a homophone substitution could reach.** The
D6-PERIODIC/GAN+HOM arm resamples homophone spellings in the init's SFT targets, so the admission
read asks whether enough corpus mass sits in a homophone class to be worth funding, against a
pre-registered floor of 5 % of tokens. A class is one distinct FULL pronunciation set over the
39-ARPAbet lexicon, members must reach 1e-5 of LM tokens (8,033 occurrences) and two characters, and
the in-class draw is uniform; the read is label-free, on the §1d student's own word decode with the
lexicon as allowed prior knowledge.

| quantity | value |
|---|---|
| classes (>= 2 members, after filtering) | 142 |
| class sizes seen in corpus | 131 of size 2, 8 of size 3 |
| corpus tokens | 963,857 |
| in a multi-member class | 74,037 = **7.68 %** |
| funding floor | 5 % -- **PASS** |
| share a uniform draw actually rewrites | 4.02 % |
| top 8 classes' share of all rewrites | 60.5 % |

**24. D6-PERIODIC-WARM -- the same per-boundary refit, CONTINUED from the previous round's scorer.**
Approach 22's seven refits each discarded the previous scorer and re-fit from the random
initialization at a fixed seed, so the only thing carrying across a boundary was the corpus; this arm
changes that one argument and nothing else -- same fork, same cosine offsets, same shard rule
(it calls approach 22's own `train_bed`, so leg k trains on the identical utterances), same refresh
corpus, pool recipe, reward, batching regimes and four-clause gate against the last accepted scorer.
Leg 1 precedes the first warm start and is therefore approach 22's own finished leg, shared by hash
rather than recomputed, so the arms are identical through parent sub-epoch 3 and the changed argument
first lands in the boundary producing leg 2's scorer. Relaunched 2026-08-18 with no acceptance step,
as the sibling was; the one verdict the gated run produced before that is banked here because its job
was deleted with the rest -- the warm-started round-2 candidate was REJECTED under the binding
confidence-interval reading, failing the corruption ladder (worse on filler substitution and on
language-model substitution) while passing the insertion price it was the sibling's habitual failure,
so a continuation of the incumbent did NOT find the clause table easier as registered.

The ungated relaunch has completed the same five-leg prefix as approach 22:

| leg | fresh periodic dev-clean / dev-other | warm periodic dev-clean / dev-other | dev-other S / D / I, warm |
|---|---|---|---|
| 1 | 4.97 / 8.88 | 4.97 / 8.88 | 3572 / 471 / 479 |
| 2 | 4.65 / 9.02 | 5.07 / 9.19 | 3576 / 456 / 680 |
| 3 | 5.28 / 9.27 | 4.85 / 9.04 | 3570 / 386 / 649 |
| 4 | 6.05 / 10.56 | 6.39 / 11.19 | 3593 / 522 / 1593 |
| 5 | 7.42 / 12.68 | 12.18 / 19.33 | 3685 / 441 / 5874 |

Warm inheritance is inside the fresh arm's range through leg 3, then separates in the harmful
direction; the leg-5 gap is +4.76 / +6.65 WER and is an insertion explosion. Leg 6 is submitted but
pending for maintenance, so this is not the final registered read.

**25. HOM-0b and HOM-0c -- whether the reward can act on a spelling, and whether sampling already
varies one.** 0b takes the label-free arm's own round-1 samples at T=0.7, substitutes ONE in-class
spelling per variant leaving the rest of the text untouched, and re-scores both reward terms under
that arm's round-1 refit and the same language model the loop's prior reads, at the arm's own weight
(lam_lm 1.0) and the arm's own per-unit-frame normalization -- so the pre-registered bar, median
|delta lm_prior| against median |delta recon|, is a direct comparison; the swaps split into the
repair direction (a spelling the refit corpus never contained) and the diversity direction (both
attested), and the sign of delta recon is read against the change in spelling length. The prior
column is anchored rather than trusted: the dump carries the prior the loop itself banked for every
base text, rows whose text does not re-tokenize to the length the loop scored are dropped first, and
the recomputed column has to reproduce the banked one or the job fails. 0c counts how often the
init's sampled groups already hold two spellings of one class, and runs on the label-free init's
full-bed G=12 dump at the same temperature rather than the arm's round-1 dump, which samples one
candidate per utterance and therefore cannot express within-group coverage at all. Measured:
0b on 8,000 utterances (25,541 of 28,539 sampled texts round-trip through the tokenizer exactly,
89.5 %) giving 23,085 swaps with 5,162 dropped by a 4-per-text cap, the prior column reproducing the
dump's own to a median 0.0053 nats/token against a 0.05 bar; 0c on all 28,539 groups of 12, of which
26,584 are homophone-bearing, 6,228 = 23.43 % already hold two spellings of one class, and 217 =
0.82 % ever contain a spelling absent from the scorer's own training corpus.

| medians over 23,085 single-word swaps, per unit frame at lam_lm 1.0 | abs delta lm_prior | abs delta recon | ratio |
|---|---|---|---|
| all swaps | 0.0134 | 0.0106 | **1.26** |
| diversity (both spellings attested in the refit corpus) | 0.0134 | 0.0105 | **1.27** |
| repair (into a spelling the refit corpus never contained) | 0.0159 | 0.0194 | **0.82** |

**26. D6-PERIODIC/GAN -- the same per-boundary refit on the label-free init** (logged after the fact;
launched 2026-08-17). Approach 22's refresh unit with theta_0^G in place of the gold-seeded fork, on
the same 960 h bed, same shard rule, same shaped reward at T=0.7, same `d_min=2` topology -- with the
two parts that read gold text dropped rather than ported, so the pool is an anchor-free greedy decode
and the refit's own model goes straight to the next leg with no acceptance gate at all. Leg k sits at
the schedule position the two held frozen-scorer arms occupied at their sub-epoch k, so those arms and
the frozen-repaired-scorer control are read at matched points; the no-loop init theta_0^G is
13.89 / 18.34 and is the level every G-track loop arm has so far failed to clear.

The two primary GAN-init anchors are:

| GAN-init anchor | dev-clean / dev-other | operating point |
|---|---|---|
| AV SFT, no loop: theta_0^G | 13.89 / 18.34 | pseudo-label AV SFT, epoch 10 |
| best previous frozen-scorer loop (reference, not schedule-only control) | **12.68 / 17.57** | shaped arm, repaired d2_contrast scorer frozen, sub-epoch 2 |

The frozen contaminated-scorer arm is shown below as a diagnostic control, but it is not the best
previous frozen-loop result. The best frozen row is also not a single-variable control for periodic:
both start from theta_0^G and match the 960 h bed, shaped reward, T=0.7 and nominal cosine position,
but the frozen row uses one d_min=1 d2_contrast scorer trained under the D2 recipe and one continuous
multi-sub-epoch training job. Periodic fits d_min=2 from scratch on each policy's anchor-free greedy
pool and runs one training job per leg, restarting Adam. Isolating scorer schedule requires the
periodic graph with its own round-1 d_min=2 scorer held fixed across otherwise identical legs; that
arm does not exist.

The d_min=1 setting was historical, not a winning hyperparameter. D2 was committed on 2026-08-07
to change only the contrastive objective relative to psi_g_tc100, and its PsiAlignTrainJob call had
no `min_dur` argument because that interface did not yet exist. D6 added the minimum-duration
topology on 2026-08-11 in response to the later insertion-price diagnosis. D3 then froze the already
finished D2 winner, inheriting d_min=1; it never compared d_min=1 against d_min=2.

| dev-clean / dev-other, plain WER as scored | sub-ep 1 | sub-ep 2 | sub-ep 3 | sub-ep 4 | sub-ep 5 | sub-ep 6 |
|---|---|---|---|---|---|---|
| frozen contaminated psi_align^G, `shaped` (held) | 13.42 / 18.75 | 13.91 / 18.91 | 13.49 / 18.81 | 17.99 / 23.33 | -- | -- |
| frozen repaired scorer, `shaped` (held) | 13.57 / 19.69 | 12.68 / 17.57 | 13.54 / 18.56 | -- | -- | -- |
| refit at every boundary (this arm) | 14.45 / 19.69 | 12.85 / 17.89 | 13.20 / 18.20 | 17.76 / 23.17 | 17.92 / 23.27 | 18.38 / 24.01 |

Only sub-epoch 2 improves the no-loop init's 18.34 dev-other, by 0.45; the later loss is mainly
substitutions (4,110 at sub-epoch 2 to 7,331 at sub-epoch 6), not insertions. Sub-epoch 7 is submitted
but pending for maintenance and sub-epoch 8 is dependency-unbuilt.

The GAN+HOM variant changes the policy initialization through homophone-resampled SFT and then runs
the same loop with its own downstream refits:

| dev-clean / dev-other, plain WER as scored | init | loop leg 1 | loop leg 2 | loop leg 3 |
|---|---|---|---|---|
| plain GAN init / periodic | 13.89 / 18.34 | 14.45 / 19.69 | 12.85 / 17.89 | 13.20 / 18.20 |
| GAN+HOM init / periodic | 16.67 / 21.45 | 14.84 / 19.99 | 13.94 / 18.77 | 12.80 / 18.08 |
| class-internal substitutions, GAN+HOM dev-other | 1827 | 130 | 110 | 105 |

The hom arm loses at legs 1-2 but catches the plain trajectory at leg 3, while removing nearly all
augmentation-specific class-internal substitutions in its first leg. Its leg 4 is submitted but
pending for maintenance; later legs do not yet exist.

**27. theta_0^G_hom -- the homophone arm's policy init** (launched 2026-08-18 on the user's
greenlight, after HOM-0b admitted the arm). theta_0^G's own builder with the resampled pseudo-label
corpus as targets and every other argument shared, so the two inits differ in the training text and
nothing else; 10 epochs, last-epoch pin, no dev-WER selection. The dev recogs it runs are the arm's
no-loop baseline and the level its eight loop legs will be read against, exactly as theta_0^G's
13.89 / 18.34 serves approach 26.

| dev-clean / dev-other, plain WER as scored | ep 2 | ep 4 | ep 6 | ep 8 | ep 10 (pinned) |
|---|---|---|---|---|---|
| theta_0^G (plain corpus) | 175.25 / 180.54 | 28.27 / 33.04 | 14.46 / 19.09 | 13.91 / 18.74 | 13.89 / 18.34 |
| theta_0^G_hom (resampled corpus) | 226.53 / 217.88 | 20.57 / 24.06 | 17.25 / 22.37 | 16.84 / 21.45 | 16.67 / 21.45 |

The homophone arm's dev-other does not move between ep 8 and ep 10, so the 3.11 gap at the pin is
carried by the plain arm's own late gain.

Where the extra errors sit, at the pinned epoch on dev-other (the registered class-internal
substitution read; gold, reported only, selecting nothing): the homophone init makes 1,587 more
errors NET than the plain one, and 1,534 of that net -- 96.7 % -- are substitutions WITHIN a
homophone class (of extra SUBSTITUTIONS alone the share is 92.2 %: 130 non-class substitutions were
also added, offset by 20 fewer deletions and 57 fewer insertions). Class-internal substitutions are
25.2 % of all its substitutions against 5.2 % of the plain arm's -- and that 5.2 % baseline is 65 %
one pair, `by -> buy`, 190 of 293. Its top confusions after the shared `with -> of` are `in -> inn`
(329), `not -> knot` (155), `be -> bee` (155), `by -> buy` (91), `no -> know` (81). Per dev-other
REFERENCE token the class-internal substitution rate is 3.59 % against the plain arm's 0.58 %; over
class-bearing reference tokens only, the same counts read 40.96 % against 6.57 %. The like-for-like
expectation is 4.58 % of reference tokens if the SFT reproduced the uniform draw in full, so the
realized 3.59 % is 78 % of it -- the policy under-reproduces the draw by about a fifth. The
damage also SPREADS: it lands in 82 distinct classes against the plain init's 33. Neither
figure is an artifact of the arm's 292-word filtered class list -- recounted against the full
pronunciation lexicon's 52,969 in-class words the same two decodes read 12.87 % and 31.38 %
of substitutions.

**28. Which SPELLING the reward points at** (`HomophoneDirectionJob.Uo4UAJp5Ue42`, on HOM-0b's own
23,085 swaps and the round-1 dump's reference rows). HOM-0b's bar compares the two terms' absolute
movement, so it cannot separate a term that swings toward the right spelling from one that swings
toward the wrong one; the reference text sits unused in the same dump the swaps were built from.
This joins it back and reports, per direction, the share of swaps each term PREFERS. Position-aligned
(reference word aligned to the swapped position) is the primary read; bag-of-words is reported
beside it and agrees. Gold read: reports only, selects nothing.

| share of swaps the term prefers | n | reconstruction | language-model prior | composed, lam_lm 1.0 |
|---|---|---|---|---|
| TOWARD the reference spelling | 1421 | 0.179 | 0.906 | 0.529 |
| AWAY from the reference spelling | 19328 | 0.254 | 0.016 | 0.063 |

The composed column is concentrated in one class: `buy`/`by`/`bye` supplies 961 of the 1,421
toward-reference swaps (67.6 %) and reads 0.446, while the remaining 460 read 0.702. Per class,
`air`/`ere`/`heir` 1.000, `side`/`sighed` 0.949, `knew`/`new` 0.884, `sea`/`see` 0.800,
`right`/`write` 0.726, `their`/`there`/`they're` 0.588, `war`/`wore` 0.333 (n=15). Splitting by
whether the reference spelling is one the scorer's refit corpus holds does NOT explain it: only 24
toward-reference swaps are the repair direction at all, and the attested-spelling subset still
reads 0.534. Measured on the PLAIN arm's round-1 dump, i.e. on a policy without the augmentation.

**29. The same two reads on the arm's OWN dump and OWN scorer** (`HomophoneDirectionJob.deNc7xXnCfSu`
and `HomophoneScorerDeltaJob.JKbbRWimojlI`, on theta_0^G_hom's round-1 dump). Approach 28 is
weighted by theta_0^G's error profile, which overlaps this arm's damage barely; here the swaps come
from the policy whose errors the loop must actually repair, and the scorer is this arm's own round-1
refit -- the reward leg 1 is graded by. The second job holds the dump and the swaps fixed and moves
ONLY the scorer, so the refit's effect is separable from the policy's.

| share of swaps the term prefers, position-aligned | n | reconstruction | language-model prior | composed |
|---|---|---|---|---|
| TOWARD the reference spelling | 8806 | 0.357 | 0.970 | 0.825 |
| AWAY from the reference spelling | 10959 | 0.559 | 0.030 | 0.140 |

Coverage is now the damage distribution: `in`/`inn` n=1755 (composed 0.833), `their`/`there`/`they're`
819 (0.896), `knot`/`not` 807 (0.927), `buy`/`by`/`bye` 700 (0.661), `be`/`bee` 632 (0.728),
`know`/`no` 517 (0.660), `wood`/`would` 368 (0.948), `too`/`two` 273 (0.905) -- every class above
chance, against approach 28's 1,421 toward-swaps of which 961 were one class.

Read beside the audio-free null, as the standing principle requires: the language-model prior ALONE
reads 0.9701 on the same swaps, so the composed 0.8255 means adding the audio-grounded term COSTS
14.5 points of reference accuracy. Homophone class members are acoustically identical by
construction, so this is expected rather than a defect -- there is no audio evidence to use -- but
the headline is the prior's number, not the scorer's, and is not quotable without it.

The scorer contrast, same swaps, scorer the only thing that moves: the reconstruction term's
toward-reference rate is 0.357 under this arm's own refit against 0.684 under the plain arm's
(-0.327); at the OPERATING POINT, i.e. under the composed reward the loop actually applies, the same
swaps read 0.825 against 0.895, so entrenchment costs -0.069 in the deployed reward. Paired per swap
2547 both / 598 own only / 3480 plain only; of 121 classes 77 move down, 18 up, 26 tie, median
per-class delta -0.172.

Length-matched, because these repairs are predominantly SHORTENING and a refit that merely priced
character length upward would reproduce the signature with no spelling-specific learning: the
entrenchment survives at EQUAL character count, -0.284 on n=1889 (own 0.469 against plain 0.752),
beside -0.388 shortening and -0.137 lengthening. So it is not a length price. The sharper reading is
that the plain scorer holds real spelling discrimination at equal length (0.752) and the arm's own
refit collapses it to near chance (0.469).

Operating point of the measurement, named rather than assumed: T=0.7 sampled rollouts over
train-clean-100, whereas the damage profile it is weighted against is a greedy dev-other decode --
the two are not the same population. The refit saw every base text in its own training corpus (a
bias that runs against the reference, so it cannot inflate the headline). The dump's own reward
columns were written under psi_g_tc100 while every swap number here is under the named refit. The
corpus-zero repair direction is unmeasured at 11 of 8806 swaps and nothing here speaks to it.

**30. D7.0a raw donor-support census** (`D7RawDonorCensusJob.zsnx1p9nLyV3`). This is the
standalone, label-free feasibility read authorized before D7-v2: it enumerates every directed edge
from each of the immutable 1,500 external source utterances to the disjoint 4,067-utterance dev
complement, and separately every directed edge within the intended 28,539-utterance scorer corpus.
An edge requires a different utterance from the same speaker and inclusive raw-unit duration match
`20 * abs(L_d - L_s) <= max(L_s, 1)`, where `L_s` and `L_d` are the 50 Hz unit-array lengths before
deduplication. “Same chapter” means equality of the middle LibriSpeech utterance-ID field; the other
stratum is “different chapter”. No tokenization, dynamic-programming feasibility, duplicate
filtering, nuisance ranking, capacity, assignment, scorer, reference text, WER, or training enters.

| population | sources | candidates | raw edges | same / different chapter | sources with >=2 in both | sources with >=8 in both |
|---|---:|---:|---:|---:|---:|---:|
| external held/complement | 1,500 | 4,067 | 4,911 | 2,553 / 2,358 | 276 (18.4 %) | 0 |
| intended scorer corpus | 28,539 | 28,539 | 632,913 | 327,169 / 305,744 | 18,843 (66.0 %) | 11,711 (41.0 %) |

On the external graph, 1,331 sources have any donor and 169 are isolated; 2,571 candidates are used
and 1,496 have zero load. The edgeful bipartite graph has 583 weak components. Complete sorted edge
tables and per-source degrees, donor loads, component membership, split/speaker/chapter coverage and
signed/absolute duration shifts are retained in the cited artifact. Their semantic tuple hashes are
`7855557c...d2f3` externally and `3a6038ab...4376` on the scorer corpus.

**31. D7-v2 / D7.0b frozen donor and loss preflight.** The 2026-08-21 amendment is implemented as
three serial, label-free jobs. The first binds the accepted D7.0a edges, exact pseudo-pairs, raw
units, BPE/lexicon inventory and round-1 scorer, then applies the registered feasibility, duplicate
and two-stage ordinal nuisance law. The second solves the common-set training construction (ten
K=4, 2+2, exact 2-in/2-out Q2 tables) and the external construction (one K=1 cap-three matching,
fixed chapter balance and split floors). Only after both structural floors pass, the third recreates
the common epoch-4 training point, freezes MAD temperature and gradient-norm coefficient, reports
the K1/K4/K8 diagnostics on one common K8-eligible population, and measures one K=4 update. The graph
contains no D7.1 scorer, policy, reference text or WER consumer.

**32. D7-GAN-SEQDISC full-bed online-negative A/B.** This is the corrected active D7 and shares no
construction with Approaches 30--31. Ten deterministic theta_0^G argmax-decode shards cover the
281,241-utterance unlabeled 960 h bed. A D7.0 barrier binds those texts to the frozen enc50 K=500 raw
50 Hz unit store, reproduces the established ordered seed-42 5% holdout, persists only a
speaker/duration/role index, and runs the registered one-update finite/resource check on frozen shard
0. Only after that PASS artifact exists do the matched D7.1 control (`L_NLL + L_U->z`) and candidate
(`L_NLL + L_U->z + softplus(s_donor-s_own)`) run for one ten-shard corpus pass. Both jobs preserve the
same initialization, batch order and dropout RNG stream; the candidate's extra forward contributes
gradient without advancing the next positive batch's RNG. D7.2 and D7.3 are absent from this graph.

Decoder equivalence of the merged shards, 2026-08-21 (verification-round check, label-free): on the
28,539 train-clean-100 utterances the ten-shard merge shares with the banked greedy decode
`ReturnnForwardJobV2.66pIzBzffnK2`, the two texts agree on 25,426 utterances exactly (89.09 %) and
differ by 4,667 word edits against 1,016,991 reference words, i.e. 0.459 %. Of the 3,113 differing
utterances 67.9 % differ by a single word edit (mean 1.50), and the net length drift is +7 words over
the whole set. That is the signature of argmax ties resolving differently under a different batching
of the same model and decoder, not of a different decode: no systematic length or content bias.

D7.0's registered parity clause cannot pass on this backend, 2026-08-21. The preflight asserts that
two deepcopies of one model, given the same restored RNG state and the same batch at
``online_weight=0``, produce byte-equal loss AND byte-equal gradients
(``torch.equal(g_control, g_parity)``). On its first ever run it raised "D7 control parity failed
when L_online=0". Reproduced read-only on one GH200 through the job's own code path
(`scripts/d7_parity_diag.py`, `log/d7_parity_diag.1446568.out`, shard 0's first batch, 256 rows):

| comparison | loss | max abs gradient delta | gradients equal |
|---|---|---:|---|
| the two deepcopies, i.e. what the clause asserts | 9.983121871948242 both, EQUAL | 2.623e-06 | no |
| the SAME model object, run twice -- the control | identical again | 5.484e-06 | no |

The same-object repeat is decisive: rerunning one model on one batch perturbs its gradients MORE
than the two copies differ from each other, so the difference is the backend's own run-to-run noise
(``deterministic_algorithms`` False, ``fast_bw`` True, atomics in the FastBaumWelch backward), not a
state difference between the arms and not an effect of the resume-RNG or counter commit. Exact
gradient equality is therefore unreachable here and the barrier can never emit its PASS artifact as
written, while loss equality holds exactly. This is the configuration-exact versus bit-exact
distinction already pinned for D8, firing inside D7's own barrier. The clause is pre-registered, so
its form is the planner's to rule on and no repair has been made; the implementer's proposal is to
keep exact equality on the loss and make the gradient arm self-calibrating -- measure the run-to-run
floor in the same job and require the control-versus-parity delta not to exceed it -- which needs no
constant and still fails loudly on a real state difference. First live exercise of the registered
infeasible-donor counter in the same run: 0 infeasible of 256 donor pairs, cases 209
ordinary_window / 47 nearest_fallback.


**D7.1 completed 2026-08-21 23:05, both arms, one ten-shard corpus pass each.** The pass is a single
14-minute job per arm, not a multi-resubmit run: 2,361 batches over 10 round-robin shards at ~70 s
per shard, peak resident set 4.89 / 4.85 GiB. Both arms report the identical bed -- 281,241 rows,
267,179 train and 14,062 held before filtering, then the SAME four own-infeasible train anchors
dropped by name (`3488-85273-0024`, `3889-130125-0028`, `4492-8904-0032`, `8424-284526-0028`),
267,175 trained and 14,062 held. Shard row and frame counts agree arm to arm at every one of the ten
shards, and the internal-held donor draw is identical (case counts 11,855 ordinary_window / 2,153
nearest_fallback / 54 singleton, 14 infeasible donor pairs, 9,825 unique donors), so the two arms
differ only in the loss term.

| arm | objective | internal-held NLL per frame | internal-held mean `L_online` | job |
|---|---|---:|---:|---|
| control | `L_NLL + L_U->z` | 2.5259 | 0.010225 | `S/d7_online/D7OnlineTrainJob.j16rTskXF1QU` |
| candidate | control `+ softplus(s_donor - s_own)` | 2.5319 | 0.007541 | `S/d7_online/D7OnlineTrainJob.WA1bqjXQtzeZ` |

Train-side sampling over the 267,175 anchors, identical in both arms: 266,134 ordinary_window /
1,041 nearest_fallback, 1 infeasible donor pair, 170,443 unique donors, maximum donor reuse 8,
own/donor duration ratio mean 1.0144 (min 0.4615, max 3.7246). D7.2 and D7.3 remain absent from this
graph; the config registers d7_0 and d7_1 only and is now fully finished.


**33. D6-PERIODIC/GAN960-FROZEN: the frozen-scorer loop restarted from theta_0^G960.** User-funded
2026-08-21 on the §3d.A scale read (`SAE_3D_GTRACK.md` approach 5, verdict 11), registered by the
planner in `PLAN_3E1.md`. The arm is `config_sae_3e1_d6periodic_gan_frozen_v1`'s recipe verbatim --
eight segmented policy legs, the same round-robin 960 h shard per leg, shaped reward, temperature
0.7, cosine offsets, fresh optimizer state per leg, and round 1's completed `d_min=2` scorer
`S/psi_align_jobs/PsiAlignTrainJob.dsMKgPHQApyR` held frozen at EVERY leg -- with exactly one
experimental change: the policy init is theta_0^G960
(`T/ReturnnTrainingJob.HuSkdbuVRg6d` sub-epoch 10, dev 13.11/16.82) instead of theta_0^G
(`.2fb02hGUdHNj` ep10, dev 13.89/18.34). A fresh scorer refit on theta_0^G960's own decodes is
deliberately NOT funded: holding the sibling's frozen scorer is what keeps this a one-argument init
A/B readable against it at matched legs.

Verified against the built graph rather than the diff, before any launch: the frozen scorer resolves
to `dsMKgPHQApyR` at all eight legs; leg 1 is a new job `T/ReturnnTrainingJob.ohmLWWmr6Kxe` against
the sibling's `.kr1foUV6lecx`; the alias namespace is `..._8se_gan960frozen/rK`, no collision; legs
2-8 carry dump/pool/refit `None`; and of the 64 unfinished jobs a launch would fund the classes are 8
ReturnnTrainingJob, 8 ExtractAvSubmodelJob, 17 ReturnnForwardJobV2, 16 SearchWordsDummyTimesToCTMJob
and 16 ScliteJob, with ZERO psi_align/curate/scorer_diag work -- no refit is funded anywhere. The
planner's anchor instruction has nothing to bind on this bed: `loop_config` takes only
`psi_checkpoint` and `av_checkpoint`, so no KL snapshot or reference anchor exists to follow the
init. One inherited-bookkeeping conflict is flagged to the planner rather than resolved: `build()`
reuses `round1_artifacts()`, so leg 1's record carries dump/pool/refit that are the SCORER's
provenance from theta_0^G decodes, which in this arm is not its own round 1. Those jobs are finished
and fund nothing, but a downstream audit could misread them.

**34. D8.0 feasibility read of the two frozen rollout dumps.** The CPU-only registered read
(`config/sae_3e1_d8_0.py`, `speech_llm/sae/d8_feasibility.py`) that can close D8 before any rollout
is generated on the 960 h bed. Per group it dedups the support on the D8 reader-normalized string,
zeroes and counts candidates that are empty after the fold, structurally infeasible on their own
audio at `d_min=2`, or unencodable, then reports distinct support, median effective sample size
across the registered tau grid for the shaped, acoustic-only and LM-only scores, the provisional
`tau_star` from the `|median ESS - 3|` rule, the within-group weight variance token count alone
explains at `tau_star`, and the median spearman between shaped weights and each single-term weight
vector. Rows are whitelisted by `kind`, so the reference rows and the gold-derived `wer` column are
never read; the LM term is converted to per-unit currency (`lm_prior * n_tokens / n_units`) against
each dump's OWN unit store before any weight. Only the theta_0^G artifact binds, on its T=0.7 slice.

The v1 read returned NO-GO on that slice. The v2 reader adds one guard, pre-registered in the job
docstring before it was run: a member the reader calls structurally infeasible whose STORED `recon`
is finite is a contradiction, because the artifact's own scorer aligned that pair. When that count
is nonzero on the binding slice the verdict is UNRESOLVED, never a no-go. Every slice also reports
`distinct_support_scorer_free`, which applies only dedup and the empty-after-fold drop.

| artifact | units store, median length | slice | law conflicts | distinct: exclusion / scorer-free | tau* | verdict | job |
|---|---|---|---:|---|---:|---|---|
| theta_0^G, 512 utts | `MergeUnitsPklJob.hJmZtbPDa2hd`, 169 | T=0.3 | 2,872/3,182 | 0 / 6 | 1.0 | reported | `S/d8_feasibility/D8FeasibilityReadJob.mDQ2LoAzrMTE` |
| theta_0^G, 512 utts | same | T=0.5 | 4,199/4,693 | 0 / 10 | 0.05 | reported | same |
| theta_0^G, 512 utts | same | **T=0.7 (binds)** | 5,096/5,730 | **0 / 12** | 0.05 | **UNRESOLVED** | same |
| theta_0^G, 512 utts | same | T=0.9 | 5,211/6,457 | 1 / 13 | 1.0 | reported | same |
| theta_0^G, 512 utts | same | T=1.0 | 4,228/6,604 | 4 / 13 | 0.5 | reported | same |
| fork epoch, 28,539 utts | `MergeUnitsPklJob.ncxcd3vouD5E`, 674 | T=0.7 | 18/101,190 | 3 / 3 | 1.0 | reported-only | `S/d8_feasibility/D8FeasibilityReadJob.ulUbBcxIiJtf` |

The superseded v1 reads are kept as the evidence that motivated the guard:
`D8FeasibilityReadJob.iCuYuvkL6bwr` (theta_0^G, verdict NO-GO) and `.onK5ekDuoLLA` (fork epoch).

**v3, the ruled operative-frame read** (clause-(a) ruling, `PLAN_3E1.md` D8 Status 2026-08-22).
Exactly one thing changes: the structural-infeasibility exclusion is evaluated against `T_i` from
the frozen raw 50 Hz store `S/quantize_states/PackUnitsJob.I0uzRMfUrKWC` -- the frame every
D8.1a/D8.1b training aligns to -- instead of each dump's own joined store, with coverage over the
slice's ids asserted rather than assumed. The per-unit prior currency still divides by the DUMP's
own store, because that is the stream its stored `recon` is per frame of; the two joins are
deliberately separate. The v2 law-conflict guard retires by construction and its count survives as
the reported `operative_law_finite_score_exclusions`. A 5 % safety valve on the binding slice's
exclusion rate returns UNRESOLVED instead of feeding clause (a).

| artifact | slice | excluded / scored | distinct (feasible, greedy incl.) | tau* | median ESS at tau* | token R2 | rho(shaped, LM-only) | rho(shaped, acoustic-only) | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| theta_0^G, 512 utts | T=0.3 | 0 / 3,182 | 6 | 0.05 | 3.685 | 0.4133 | 1.0000 | 0.3000 | reported |
| theta_0^G, 512 utts | T=0.5 | 0 / 4,693 | 10 | 0.05 | 5.308 | 0.2837 | 0.9833 | 0.2857 | reported |
| theta_0^G, 512 utts | **T=0.7 (binds)** | **0 / 5,730** | **12 of 13** | 0.05 | 5.433 | 0.1741 | 0.9790 | 0.3132 | **GO** |
| theta_0^G, 512 utts | T=0.9 | 0 / 6,457 | 13 | 0.05 | 2.392 | 0.0483 | 0.9785 | 0.5497 | reported |
| theta_0^G, 512 utts | T=1.0 | 0 / 6,604 | 13 | 0.2 | 3.976 | 0.0354 | 0.9785 | 0.6593 | reported |
| fork epoch, 28,539 utts | T=0.7 | 18 / 101,190 | 3 | 1.0 | 2.976 | 0.3928 | 0.5000 | 1.0000 | reported-only |

Jobs: `S/d8_feasibility/D8FeasibilityReadJob.mv2d0vkWN93a` (theta_0^G, binding) and
`.W7TWfwoZtkaC` (fork epoch). The fork read's numbers are unchanged from v2 to the last digit,
because that dump already joined the raw 50 Hz store; its 18 exclusions are the genuine rate the
ruling prices the safety valve against.

## Verdicts

1. **The co-trained scorer did NOT go text-blind — the hypothesis is refuted by its own instrument.**
   The usage gate RISES from 0.3331 at ep0 to a peak 0.6210 at ep6 (+86 %) and never falls below ep0
   at any epoch, so the replay arm's 18.79 -> 46.71 needs a different explanation.
2. **What it did instead is lose its conditional entirely**: CE_true jumps 5.7444 -> 6.2045 after ONE
   sub-epoch and reaches 6.2938 at ep6, i.e. past the unit marginal 6.0072 and past uniform
   ln 500 = 6.2146 — by ep6 the scorer is worse than a coin on gold pairs while its text-contrast
   grows. Co-training damage is scorer DRIFT off the gold domain, not text-blindness; a future gate
   on trainability must read CE_true, which the usage gate alone would have passed.
3. **Ranking noise is refuted a second time, now within-group and at the loop's operating point**:
   psi_g_tc100 spearman 0.4959 (recon) / 0.5558 (shaped), frac_pos 0.93/0.95 over 512 groups, with no
   difference between all groups and the WER-spread-bearing subset.
4. **Directional bias is confirmed but is mostly NOT contamination.** All three scorers pay for the
   filler at matched WER, including the never-contaminated gold-text control (beta 0.1673 on "to").
   Only the differential — psi_g_tc100 minus gold_enc50, +0.075 on "to", +0.063 pooled — is
   attributable to the shared pseudo-text; roughly 70 % of the effect is a psi_align FAMILY property,
   so a round-0 text repair (D2) can address at most the smaller part and the family term needs the
   plan's other lever (null-word down-weighting / matching-aware contrastive term).
5. **Corpus size is not the axis**: psi_g_seed, with 10x less of the same pseudo-text, has the LARGER
   bias (0.2664 vs 0.2425) and the worse ranking (0.4696 vs 0.4959) — more pseudo-text helps slightly
   in both directions.
6. **Group blindness is real but partial, and is the binding constraint on "to"**: only 23 % of the
   groups that carry "to" hold a "to"-free member (9 % for the suspect set as a whole), so in ~77 %
   of live groups no scorer of any quality can steer off the filler. This is not the plan's
   "coverage ~ 0" fork, but it caps what any scorer-side repair can buy and makes the sampling-side
   contingency (temperature / G sweep) a co-requirement rather than an alternative.
7. **A curated refresh has an admissible external selector**: the base-LM prior in its live
   units-normalized form clears the D0(e) covariance bar at 0.5020 [0.4737, 0.5308], and the suspect
   count itself is weakly admissible at 0.1855 [0.1510, 0.2186]. psi's own duration channel is not a
   selector (CI straddles zero), which also confirms psi's ranking is not a length artifact.
8. **The filler is cheap to INSERT, not cheap to write — and the effect is token-specific but
   scorer-invariant** (5). Inserting "to" costs 0.0274 nats/frame against 0.0859 for an LM-drawn word
   in the same slot (discount 0.0584 [0.0537, 0.0634], growing to 0.2201 at k=4), while writing it
   OVER a word costs the same as any frequent word (discount +0.0036). The never-contaminated
   gold-text scorer shows the same insertion discount (0.0590 [0.0539, 0.0639]) and the same
   substitution non-effect, on the same pairs.
   WRONG in part (2026-08-07 verifier): "token-specific but scorer-invariant" does not survive the
   audit — the LM control is drawn with no length matching (~2.7 emitting BPE states per draw vs 1
   for "to"), 53–81 % of the discount is state-count-attributable, the surviving residual
   (0.011–0.027 nats/frame) is scorer-DEPENDENT, and what is scorer-invariant is the lattice's
   ~0.03 nats/frame price per inserted emitting state.
9. **D1's pre-registered power check therefore FAILS: no filler statistic separates psi_align^G from
   the 10 h-true scorer** (4, 5). All three arms agree on the insertion discount, the substitution
   discount and the suspect state mass (1.90–2.19 %) to within their CIs; `ce_loo` does separate them
   (2.72 / 3.02 / 3.13) but in the direction of the pseudo-text DOMAIN, since the held text is that
   decoder's own output. The direction of §3e.1 has to change: with the family share measured at ~70 %
   by D0(c) and at ~100 % here, a round-0 text repair cannot be the load-bearing fix, and gate v2 (i)'s
   improvement clause is not neutral between a text-repair candidate and the incumbent.
   - **WRONG (2026-08-08, approach 10):** the power check fails only for the frequency-drawn control —
     on the state-matched one psi_align^G separates from the gold-text control decisively and in the
     expected direction (0.0172 vs 0.0031, paired -0.0141 [-0.0174, -0.0108] on the shared 1442), so
     the instrument has power and it was the CONTROL that was blunt; the family-share reading from
     D0(c) is untouched.
10. **The mechanism is an insertion/deletion asymmetry of the lattice, not a text defect** (5).
   Deleting a word costs 0.3336 nats/frame and inserting the filler 0.0274 — a factor 12 — so against
   any real alternative the policy has (commit to a word, or leave it out) padding is nearly free,
   which is exactly the insertion-driven degradation the G-track `recon` arm showed. The only existing
   counterweight is the LM prior at `lm_prior_norm="units"` — of order 0.01 nats/frame for this token
   at the bed's 338 frames per utterance, estimated from the LM-corpus unigram rate rather than
   measured contextually — which is consistent with `recon` diverging where `shaped` plateaued.
11. **Raising the sampling temperature buys contrast but degrades the oracle** (6). T=0.7 -> 0.9 lifts
   steerable coverage 0.1949 -> 0.3382, but conversion falls (steerable/coverage 0.835 -> 0.642) and
   the ORACLE WER rises 0.1071 -> 0.1496, i.e. the best of twelve samples gets worse, so T=0.9 is not
   the free operating-point move the presumptive read treated it as.
12. **The asymmetry is arithmetic and its size is frames per state, which no D2 arm but `d2_states`
   can reach** (5, 8). Deleting a word orphans T/U = 4.88 frames per state removed and the emission
   term charges every one of them (+0.2558 of del_1's +0.2461 nats/frame), while an inserted word is
   absorbed in half a frame by the skip arc (+0.0184 emission, +0.0065 transition), so the price
   ratio carries a structural factor 2 T/U = 9.77 — the filler sits at that floor (12.16) and an
   average LM word 3.1x above it (3.89). Neither a repaired corpus nor an in-batch contrast against
   whole other texts changes T/U, which is why `d2_states` (2 T/U 9.77 -> 3.92) is the arm that tests
   the diagnosis rather than a symptom.
   WRONG in part (2026-08-07 verifier): each number reproduces, but the sentence switches statistic
   bases (+0.2461 is the pooled per-frame NLL delta, not the table's per-utterance del_1 0.3336, and
   the 12.16/3.89 ratios are on the ce_loo basis where the NLL basis gives 9.88/4.24), "3.1x above
   it" inverts the comparison (3.89 is BELOW the 9.77 floor; 3.1 is filler/LM), "charges every one"
   over-counts (the emission delta +0.2558 exceeds the deletion total, the NLL-minus-emission
   residual being transitions PLUS alignment entropy and negative for deletion), and per
   removed/added STATE the del/ins ratio is 5.3–7.4 — below the claimed floor — so "the filler sits
   at the floor" is a state-count artifact (2 states vs the mean deleted word's 3.76); what
   survives, verified numerically, is the d2_states prediction itself: deletion orphans a
   chars_per_state-INVARIANT frame count (18.3 vs 18.4 across cps 1.5/0.5) while an inserted word's
   state count scales ~2.5x, so the price ratio falls ~2.5x under the arm and no corpus or
   contrastive arm moves it.
13. **All four D2 arms finished and NONE separates from the incumbent — on any statistic, not just the
    one the winner rule names** (8). The insertion discount moves 0.0584 -> 0.0595 / 0.0558 / 0.0619,
    every step smaller than the 0.005 bootstrap half-width; the filler payment itself barely moves
    (beta_to at matched WER 0.2425 -> 0.2232 for the repaired corpus, a 8 % cut, and UP for both
    contrastive arms); steerable coverage moves by at most three of 467 live groups; and the incumbent
    has the HIGHEST in-group spearman (0.4959) and the lowest sel_wer (0.1380) of the five. Read
    literally the approach-9 rule returns NO WINNER, and the length-matched statistic is unlikely to
    overturn that: beta_to and steerable coverage need no length matching, and neither orders any
    candidate ahead of the incumbent (`RolloutMechanismJob` reports beta as a point estimate with no
    interval, so this is a statement about which way the estimates point, not a significance claim).
    - Follows from this, and it is the phase's direction: D0 predicted ~70 % of the filler payment is
      family-level and unreachable by text repair, and D2 now shows the remaining ~30 % is not reached
      either — not by rate-matching, not by an in-batch contrastive term, and not by a 2.5x finer state
      rate. What D2 refutes is "round-0 repair is enough"; the D3 control it was meant to select for
      has no candidate to freeze that differs from the incumbent in any measured way.
    - **WRONG in part (2026-08-08, approach 10):** "no candidate differs in any measured way" holds
      only for the confounded discount and for the rollout statistics — on the state-matched discount
      three of four arms reduce the filler's insertion advantage with paired CIs excluding zero
      (`d2_contrast` -0.0090, `d2_both` -0.0075, `d2_states` -0.0047 at k=1), and the mechanism-level
      arms do it while `d2_rate`, the pure text repair, does NOT (-0.0018, n.s.). The refutation of
      "round-0 repair is enough" stands and is now sharper — the corpus arm is the one that fails —
      but "no winner" does not: see conclusion 15.
14. **`d2_states` splits the two insertion statistics in exactly the direction the state-count confound
    predicts** (8, 12). It is the only arm to improve ALL FIVE corruption ladders (filler_ins
    0.6552 -> 0.8540, lmins 0.8620 -> 0.9178) — insertion severity becomes far better ordered — yet
    its frequency-drawn insertion discount is 2.5x the incumbent's (0.1475 vs 0.0584), which is what a
    control pool of ~2.7-state words must do once every word's state count is cut. Neither reading can
    be checked against the other from the shipped outputs, so the length-matched control pool
    (`PLAN_3E1` D1 build item (b)) was built and read the same day — approach 10.
    - **Resolved (2026-08-08, approach 10):** the ladder was right and the discount was artifact. Under
      the state-matched control `d2_states` charges +0.0125 against the incumbent's +0.0172 — a real
      reduction, not a 2.5x blow-up — so the entire apparent regression was the frequency-drawn pool
      averaging 8.16 states against its four-state filler.
15. **On the statistic the amended rule actually names, D2 HAS a winner, and it is the mechanism arm**
    (9, 10). Eligibility on the pre-registered clauses leaves `d2_contrast` and `d2_states`
    (`d2_rate` is worse on three of the five ladders paired; `d2_both` sits 0.0135 below the
    pre-loop `text_explained_loo` floor); both reduce the state-matched insertion discount with paired
    CIs excluding zero, so the no-winner clause does not fire, and the larger reduction is
    `d2_contrast`'s — 0.0172 -> 0.0082 at k=1 and 0.0561 -> 0.0323 at k=4, roughly half the incumbent's
    filler advantage and within reach of `psi_g_seed`'s 0.0078. Its edge over `d2_states` is itself
    significant only at k=4 (-0.0096 [-0.0187, -0.0007]), so the rule's argmax picks it while the two
    are close; and the rollout tiebreaker disagrees mildly (beta_to 0.2469 vs the incumbent's 0.2425),
    which is the one tension in the read — a controlled text-side edit on held pairs and an
    observational partial effect on policy rollouts are not the same measurement, and only the first
    is what the rule selects on. The winner does not depend on reading the `text_explained_loo` floor
    as psi_g_tc100's own value: `d2_both` reduces less than `d2_contrast` either way, so admitting it
    changes nothing.

16. **The external LM prior is filler-NEGATIVE at matched WER, and the only audio-conditioned view is
    the one that pays** (11). `lm_prior_units` clears both D4 bars — it ranks (0.5020) and it charges
    for the suspect count rather than paying for it (-0.0937, CI excluding zero), refuting the plan's
    premise that an external LM would favour the filler; `ar_recon`, the G-track AR's own reward and
    the only view here that conditions on audio, ranks barely at all (0.0944) and PAYS for "to"
    (+0.0716, CI excluding zero), so it is inadmissible and the two views a refresh round may curate
    with — `lm_prior_units` and `neg_n_suspect` — are both text-side, which is a residual the plan's
    "not audio-free" clause anticipated and no measurement on this bed can currently remove.
17. **The frozen repaired scorer moves the filler on the G-track and, at sub-epoch 2, the WER with
    it — but bar 2's SHARE normalization is blind to the move** (9). At sub-ep 2 `d2_contrast` beats
    the incumbent on both arms (shaped 13.91 -> 12.68 / 18.91 -> 17.57; recon 31.46 -> 27.04 /
    36.89 -> 32.89) while cutting dev-clean "to" insertions 3539 -> 3043 and 3817 -> 2096, yet the
    shaped arm's suspect share barely moves (0.871 -> 0.862) because total insertions fall in
    proportion, and the recon arm's share falls (0.556 -> 0.408) only because a non-word fragment
    ("st", 856) takes the vacated mass — two of four sub-epochs, so this is a read, not the verdict.
    - Sub-ep 3 supplies the verdict this deferred: see (31).
18. **The incumbent AR reward's ranking replicates from the D0 sample to the whole bed, and its
    argmax pick is worse than a random one** (12). On all 28 539 utterances at T=0.7, G=12 its
    within-group spearman is 0.0959 against the D0 512-utterance `ar_recon` read of 0.0944, but
    eta = -0.1103 — picking by the reward gives 17.72 % WER against a random pick's 17.06 % and an
    oracle's 11.08 % — while greedy decoding scores 13.86 % and the gold text earns a margin of
    0.0001 over the samples, so no refresh round may curate with the reward itself.
19. **Two-view curation reaches 79 % of the bed, but its picks are dirtier than the anchor they join**
    (12). 22 667 of 28 539 groups yield a candidate with both advantages positive (83 219 members
    qualify, one kept per utterance) for a 51 206-row pool at a 55.7 % anchor share, yet the curated
    half's suspect rates run above the anchor's ("to" 0.0462 vs 0.0275, "buy" 0.0031 vs 0.0001):
    within-group selection can only take the least-bad of twelve samples, and on this bed all twelve
    are worse than the repaired round-0 text it is added to.
20. **The collapse is pure over-generation, and a suspect-SHARE bar is blind to it** (13). Recall does
    not fall — %Corr RISES 89.07 -> 91.55 (dev-clean) and 87.31 -> 88.92 (dev-other) while %Del falls
    4.46 -> 2.14 and %Ins goes 5.98 -> 38.26 at hyp/ref 1.015 -> 1.361 — so the policy still finds the
    reference words and pads around them; the suspect set's share of insertions is flat at 0.045-0.066
    across the whole trajectory because the added mass is generic function words (the, and, of, to,
    a), which is why any bar normalized by total insertions cannot see a 6.4x insertion blow-up and
    D4' must read insertion COUNTS instead.
21. **The scorer's preference migrates from gold to its own padded output, on top of an already-dead
    conditional** (14). The gold column reproduces conclusion 2 exactly (5.7444 -> 6.2045 after one
    sub-epoch), and on those same rows `self_pref` goes -0.0139 -> +0.1267 with the best-scoring
    column moving from gold at ep0 to the arm's own ep4 decodes at ep4 — the co-collapse the phase
    asked about — but every ep >= 1 row sits at or above BOTH floors, so the within-row spread
    (<= 0.14 nats) orders texts under a scorer that carries essentially no information about the units.
    - WRONG in part (2026-08-09 verifier): "at or above BOTH floors" holds only for the unit
      marginal — 20 of the 24 ep >= 1 cells sit BELOW uniform ln 500 (max shortfall 0.1187) and the
      ep4 within-row spread is 0.14205, not <= 0.14; the inference survives in the weaker form
      "above the unit marginal and within 0.12 nats of uniform".
22. **With the rollouts held identical, the reward's ranking utility goes NEGATIVE after one
    sub-epoch** (15). Pinning the policy at theta_0 so every cell re-ranks the same samples, eta at
    the arm's operating T=0.7 falls +0.2246 -> -0.1185 at ep1 and -0.2792 at ep4 (sel_wer
    0.0905 -> 0.1288 against an unchanged mean 0.1076), while `std_wg` GROWS 0.018 -> 0.026 — so this
    is not a dead band but a reward that actively prefers the padded sample, and the loop's own
    gradient condition (spearman > 0 and sel_wer < mean_wer) fails from ep1 on.
    - Follows from this, and it is the phase's direction: this bed's update rule IS D5(b)'s continuous
      joint psi, so D5(b) is no longer asking whether continuous joint psi collapses but how fast it
      does on the 960 h theta_0' bed — one sub-epoch is the number to beat, and such an arm needs a
      within-loop read of eta or CE_true, not an end-of-arm WER, to be informative at all.
23. **The label-free selection rule and the health screen disagree on this arm, and the screen is
    what makes the fork defensible** (16). Dev reward ranks sub-ep 3 first and sub-ep 2 third, i.e.
    it is anti-correlated with WER over the four available sub-epochs; the minimal-state count vetoes
    3 and 4 at +32 % and +38 %, leaving sub-ep 2 — which WER then confirms at 5.34 / 9.50, the arm's
    best. **All three update-rule arms therefore fork from `vhyvv2waeU16` sub-epoch 2.**
    - The plan's own named statistic would have missed it: words per utterance moves only
      +2.11 % / +2.31 % on the same checkpoints, inside any band that is not itself post-hoc. Reading
      the exploit CLASS by absolute count is what carries the screen, which extends D5(a)-1's
      counts-not-shares ruling — length is blind here too, not just the share.
24. **The joint arm's psi CE and its reward agree numerically from step 1** (17), so the trainable
    channel is scoring the same quantity the frozen reward reads: `psi_ce` 2.94-3.48 against
    `reward_recon` -2.95 to -3.48 on the same steps. This is a wiring check, not a result.
25. **The joint rule does not fit this bed's node at the parent's settings, on either axis, so
    D5(b)-b cannot be a single-knob arm here** (17). Measured over 76 steps: **12.39 s/step against
    the parent's 4.81, a 2.57x slowdown**, which projects a sub-epoch at **11.2 h against the 11.5 h
    SLURM cap** — no room for the recog, and `ReturnnTrainingJob` resumes per sub-epoch, so a
    sub-epoch that misses the cap never completes at all. It did not get that far: **CUDA OOM at
    step ~72**, GPU 1 at 95 GiB, because the parent already peaks at 80.9 of 95 GiB at
    `max_seqs` 8 / `group_size` 12 and the CE channel's alignment-DP autograd graph over all 96
    rollouts has nowhere to live.
    - Every fix trades against the property the control depends on: `partition_epoch` buys the time
      but ends the 1:1 sub-epoch comparison the plan pinned; `max_seqs`/`group_size` buy the memory
      but change the optimizer trajectory or the GRPO group; checkpointing the DP recurrence buys the
      memory at the price of one more DP forward, i.e. it spends the axis that is already binding.
      Reported, not chosen.
    - **WRONG (2026-08-11)**: halving `batch_size` to 1e6 with `accum_grad_multiple_step` 2 — a fix
      this list did not consider, and which preserves the effective batch, the group, the update count
      and the 1:1 sub-epoch comparison — makes the arm fit on both axes: memory flat at 48.1 of 95 GiB
      and sub-epochs of 35982 s and 33252 s against the parent's 15693 s, i.e. 2.29x and 2.12x, inside
      the 11.5 h cap.
26. **The excess-mass suspect derivation is empty on this bed, so the two-view curation rule has no
    second view here without a plan amendment** (18). At D0's pre-registered `min_excess` 0.002 no
    word qualifies; the largest excess is "and" at 0.00135, and 0.001 admits that one word alone.
    The instrument is not broken, it is out of range: it prices a *rate* difference against the LM
    corpus, and this policy decodes at 5.34/9.50, so the ~140 extra "and" tokens the minimal-state
    exploit contributes are 0.13 % of a 106 k-token corpus. `neg_n_suspect` — the weaker of the two
    G-track views at 0.186 — therefore cannot be one, and the only other cleared view,
    `lm_prior_units`, is the audio-free one the rule forbids to curate alone.
27. **The round-0 gold scorer on the best bed under-charges INSERTIONS by ~6x and discounts the
    minimal-state class inside that, before any refresh** (18). An inserted word costs +0.069 against
    +0.405 for a substitution and +0.430 for a deletion, and within insertions the minimal-state word
    is +0.021 cheaper than a frequency-matched LM word at the same slot (CI [+0.0164, +0.0250],
    frac>0 0.563). The sign FLIPS for substitutions (-0.016, CI excluding zero), so this is not a
    generic filler affinity: the scorer prices writing a filler over a word correctly and prices
    adding one almost not at all. Insertions are also where the corruption ladder is least monotone
    (0.658 against 0.735-0.784).
    - Bears on D5's attribution question: the price is wrong at ep0 of the loop, so this bed's
      insertion exploit needs no scorer drift to explain it. What the frozen arm contributes on top
      of that is what D5(a)/(b) still have to separate.
28. **The G-track refresh round completes and moves the gate statistic further than any D2 arm, but
    only under the CI reading of the ladder floor** (12). The state-matched insertion discount falls
    0.0172 -> 0.0064 at k=1 (paired -0.0108 [-0.0140, -0.0077], p 0.000) and the gap widens with k
    (-0.0422 at k=4) at no held cost (ce_loo 2.7198 -> 2.7168 against H_uni 6.0324); no ladder is
    significantly worse, but four are nominally lower, so the point reading elects nobody and the CI
    reading elects `r1`. Recording the numbers only -- the verdict is the planner's.
    - Curation is worth about a fifth of the reduction: the same recipe on the uncurated corpus
      already reaches 0.0082, and it buys that at a hair better held ce_loo (2.7139) than `r1`.
    - The contrastive term is not what did it. contrast/utt is 0.8971 in its first active epoch
      (ep5, once the alignment prior has annealed off), 0.0452 by ep6 and exactly 0.0000 from ep21
      on, so the last third of the refit is pure NLL on the curated corpus.
    - **It moves the text-side preference and not the topology.** On the same held set the insertion
      ASYMMETRY goes 0.3062 (incumbent) -> 0.2968 (`d2_contrast`) -> 0.2827 (`r1`), a 7.7 % shift
      against the discount's 63 %, so after the refresh the filler is no longer specially cheap to
      insert while inserting anything at all is still ~7x cheaper than deleting. The degradation
      this ladder exists to stop is over-generation of generic words (20), which the asymmetry
      prices and the discount does not -- the gate's pre-registered statistic is the one the refresh
      can move. (Point estimates: the clause table pairs the discount, not the asymmetry.)
29. **On the best bed psi ranks its own rollouts well, but its advantage over the audio-free null
    does not clear the pre-registered margin** (18). Within-group spearman is +0.3407 against the
    0.17 bar and the length-only null is -0.0074, so the ranking is not length; the audio margin is
    +0.0229 [-0.0036, +0.0522] (P(>0) 0.953) and gap_true +0.0089 against +0.0248, so two of the
    three G3 bars FAIL on 28538 groups.
    - Bears on the second-view problem (26): the one audio-conditioned signal this bed offers is
      itself unproven against an audio-free null at 95 %, which is a stronger reason than
      self-amplification to keep it out of a curation view.
    - The reward the loop optimizes IS the number these bars measure: the row-level parity check is
      exact, max |online - offline| 0.000e+00 on 512 of 512 round-tripped rollouts with no row
      floored, so nothing in the id round trip, the batching or the flooring has drifted.
30. **This bed has no admissible audio-conditioned curation view, by measurement rather than by
    exhaustion** (18). On 647 live groups psi's own score is filler-POSITIVE at matched WER --
    partial beta +0.2254 [+0.0817, +0.3239] for the watch class, +0.2029 [+0.0558, +0.3000] once
    shaped -- so it fails clause (f) outright: at equal WER it prefers the member carrying more of
    them. `lm_prior_units` is the only signal that clears both clauses (spearman 0.5171, beta
    -0.3018 [-0.4703, -0.1661]) and it is the audio-free one the rule forbids to curate alone;
    `n_tokens` ranks WER negatively (-0.1589) and `lm_prior_tokens`' affinity CI spans zero.
    - Supersedes the reading in (26) that the second view was merely missing: the audio-conditioned
      candidate exists and is disqualified, which is a stronger result and a worse one.
    - The (f) block reads the label-derived watch class, which is its monitor role -- it evaluates a
      view, it does not select with one. `neg_n_suspect` scores 0.4229 in (e) on that same class and
      is barred from selecting for exactly that reason.
31. **A refit scorer's WER advantage is one sub-epoch wide and does not survive the next one** (9).
    Against the incumbent's own trajectory on the shaped arm `d2_contrast` runs 13.57 / 19.69 ->
    12.68 / 17.57 -> 13.54 / 18.56 where the incumbent runs 13.42 / 18.75 -> 13.91 / 18.91 -> 13.49 /
    18.81: WORSE at sub-ep 1, better by 1.23 / 1.34 at sub-ep 2 -- which is the incumbent's own worst
    sub-epoch -- and level on dev-clean by sub-ep 3. Both arms wobble inside a 13.4-13.9 band, so
    (17)'s sub-ep 2 read was taken against a bump and the repair has not bought a durable WER point.
    - The recon arm keeps a real gap (32.94 / 37.91 against 33.54 / 39.74, on 5623 insertions against
      8724) but both arms there are diverging, so it prices a slower divergence, not a fix.
    - What the cancellation left unmeasured is the only thing that would settle it: the incumbent
      COLLAPSES at sub-ep 4 (13.49 -> 17.99, substitutions 2894 -> 5301) and the repaired arm is held
      at sub-ep 3, so whether the repair delays that collapse is unknown.
    - The repaired arm's own sub-ep 3 regression is insertions and deletions, not substitutions
      (3616 -> 3929, 412 -> 582, 2868 -> 2857) -- the same shape the best bed shows (`SAE_0d.md` c13).
32. **One sub-epoch of co-training is the largest WER gain anywhere on this bed, and the next one
    destroys it** (17). The joint arm runs 5.12 / 9.27 at sub-ep 1 where its matched frozen control
    runs 6.56 / 11.15 -- better by 1.44 / 1.88, and better than the parent's all-time best 5.34 / 9.50
    -- then 17.35 / 21.97 at sub-ep 2 where the control runs 6.89 / 11.31. Both moves are the insertion
    channel: 385 / 630 at sub-ep 1, far BELOW the frozen band's 1182-1415 / 1592-1794, then 6114 / 6450
    at about 16x that, while substitutions move far less (2029 -> 2952 dc). The collapse D4's
    offline-only shape was built around is therefore real and now carries the frozen control it was
    missing, but it is a cliff after one good step rather than a decay, which argues for GATING a
    discrete refresh rather than against refitting.
    - The pre-registered CE_true alarm (+0.1 nats) is unread: psi is extracted at both sub-epochs but
      the held-NLL forensics have not run, so the scorer-side mechanism is still open.
    - Sub-ep 3 is in flight, so whether the arm recovers or stays collapsed is unmeasured.

**33. Round 1's uncurated refit fits the frozen held set materially better than the incumbent but
fails the gate on one ladder** (19). Held ce_loo 2.7614 -> 2.6432 with text_explained_loo up +0.1182
and the two INSERTION ladders' paired spearman up +0.0434 [+0.0340, +0.0531] (filler) and +0.0288
[+0.0194, +0.0384] (LM word) and the within-group spearman on the fork dump up +0.3399 -> +0.3621,
while `lmsub` falls -0.0058 [-0.0091, -0.0025] -- the single clause v2 (iv) reads as "not decreased"
-- so the clause table returns NO WINNER under both the point and the CI reading, and whether an unweighted ladder rule should let a 0.006 substitution loss outweigh a
0.043 insertion gain is a gate-design question for the planner, not one an executor may resolve.

**34. Re-pricing a trained scorer's arcs cannot move insertion, and the topology it was meant to gate
is free** (20). The strongest of twelve settings lifts the k=1 insertion price 1.09x (+0.0693 ->
+0.0755) against the gate's 2x and a 4-nat skip bias only 1.03x, because a duration charge falls on
the clean text's own short states as heavily as on an inserted word's and the paired delta keeps only
the difference -- which is the argument for rung 2's TEXT-CONDITIONED prices and rung 3's hard
constraint rather than a price both sides pay -- while the feasibility statistic that gates rung 3
clears by a wide margin: 6.64 frames per content state, 3 infeasible rows in 59 878 at d_min=2 and 7
at d_min=3, so the plan's ceiling ("mean T/U ~4.9 caps d_min ~2 for the tail") was derived on a
tighter number than this corpus shows and d_min=3 is a live dial.

**35. The minimum-duration topology passes every D6 clause and repairs round 1's failing one, but the
acceptance rule's winner test asks a question the phase deliberately rescales** (20). d_min=2 clears
all four bars -- spearman +0.3621 -> +0.4357, held ce_loo 2.6432 -> 2.1620, k=1 insertion price 2.86x
the incumbent's growing in k, and `filler_ins` monotonicity 0.658 -> 0.853, from last place to first --
and it also lifts the `lmsub` ladder to 0.9572, above both the comparator's 0.9479 (the single clause
that made c33 a no-winner) and psi0_gold's 0.9538, while online/offline parity on the wider graph holds
to 5e-07; yet the clause table returns NO WINNER because its winner test wants the state-matched
insertion discount to FALL and this arm is neutral on it (+0.0047 [-0.0009, +0.0101], p=0.096) -- a
delta-ce_loo LEVEL, and every edit price on this arm is ~2.8x the incumbent's, so the same measurement
read as a share of what an insertion costs falls 11.3 % -> 7.1 %. Which reading binds is the planner's
to pin; it is also what decides eligibility, since the arm is eligible under the CI reading (0 ladders
worse) and not under the point one (2 worse, both CIs spanning zero).

**36. The corruption margin hurts on its own and drags the topology down with it, which refutes the
registered expectation that rungs 2+3 together are the shape** (20). Alone it fails clause (iv)
(`filler_ins` monotonicity 0.692, still under the sub/del band) and multiplies the matched insertion
discount six-fold (+0.0094 -> +0.0578, CI excluding zero); added to the topology arm it costs three
ladders at CIs excluding zero and drives the discount to +0.3150. The mechanism is in the probe rows:
the term raised the price of exactly the LM-drawn control words it trains against (`lmins_m` 2.9x)
more than the filler's (2.4x), so it learned its own negative distribution rather than insertion in
general -- and the combined arm's ins/del ratio of 0.699 is the closest to parity anywhere in this log.
  CORRECTION 2026-08-12: the last clause originally read that the 0.699 ratio "is indiscriminate
  inflation rather than discrimination", which the combined arm's re-rank (measured after this
  conclusion was written) does not support -- it ranks rollouts BEST of every arm here, spearman
  +0.4441 and eta +0.3392 against d_min=2's +0.4357 / +0.3296, so rung 2 mis-prices the filler against
  a matched control while still improving the statistic the loop actually consumes.

**37. The min-duration topology transfers as a FIT but fails the G-track gate on the substitution
ladder** (21b). `r1_mindur` fits the frozen held set far better than either comparator (ce_loo 2.3774
against `r1`'s 2.7168 and `psi_g_tc100`'s 2.7198) and wins both insertion ladders by a wide margin
(filler_ins +0.0849, lmins +0.0488 against `r1`, CIs excluding zero), but it is significantly worse on
filler substitution (-0.0136, CI -0.0194 to -0.0077) and so is ineligible on both readings of the
"not worse on any ladder" clause; the round closes with NO WINNER against either incumbent, which
separates fitting the held set from being safe to hand the loop.

**38. Running psi's alignment recursion on the GPU instead of in python costs nothing in fit and
makes a per-round refit affordable** (21b, replicated). The same d_min=2 refit -- same corpus, same
hyperparameters, only the forward-backward recursion moved into RETURNN's CUDA fast-Baum-Welch
kernel -- reaches best held_nll 2.3160 at epoch 23 against the python path's 2.3186 at epoch 23, in
0.78 h against 5.94 h (94 s against 713 s per epoch, 7.6x), which is what turns "refit the scorer
once per loop round" from a half-day into an hour and is the premise Z4's schedule is built on.

**39. The min-duration scorer as the live reward PASSES its pre-registered confirmation outright, and
the separation widens to the end of the run** (21a, complete at eight of eight sub-epochs). The
control's sub-epoch-3 regression does not merely shrink -- the swap-in arm never regresses at all,
improving past the fork point (5.34/9.50) to 4.68/8.64 one sub-epoch later and holding 4.73/9.31 at
sub-epoch 10 against the control's 6.46/11.41, with dev-other insertions less than half the control's
(933 against 1964). The insertion exploit the whole D6 ladder was built to close is closed in the
live loop, on the reward side alone, with no change to the policy, the data or the schedule.

**40. The homophone arm clears its admission floor, but the reachable mass is thin and concentrated**
(23). 7.68 % of corpus tokens sit in a class against the 5 % floor, so the arm is admissible; a
uniform draw rewrites only 4.02 %, eight classes carry 60.5 % of all rewrites, and 131 of the 139
classes the corpus uses have just two members, so what HOM can vary is close to a handful of frequent
function-word pairs rather than broad spelling diversity.
41. **The periodic arm has never once refreshed its scorer** (22). Its gate returned KEEP at rounds
    2, 3 and 5 and the two-consecutive-failure stop rule fired at round 4 -- where the binding
    confidence-interval reading of the clauses actually PASSED and was overridden -- so legs 2 to 8
    all run the round-1 scorer, and the arm's comparison against the one-shot swap-in measures the
    shard rule and the Adam restarts, not refresh frequency.
42. **WRONG after the six-leg read: on the label-free init a refit at every boundary is the first loop arm to go below the no-loop
    init at a matched sub-epoch, and it is still not the best scorer there** (26). At sub-epoch 2 it
    reaches 12.85 / 17.89 against theta_0^G's 13.89 / 18.34, where both frozen-scorer arms sat at or
    above the init -- but the frozen REPAIRED scorer reaches 12.68 / 17.57 at the same point, so
    what the two scored legs support is "a scorer that is not the contaminated one helps", not yet
    "recency helps"; six legs are outstanding and no replication spread exists on this bed.
    CORRECTION 2026-08-20: only two legs were available here. The six-leg prefix makes that gain
    transient and is replaced by conclusion 54.
43. **Sampling already proposes spelling variety, but almost never proposes the spelling that is
    missing** (25). 23.43 % of homophone-bearing groups already hold two spellings of one class, so
    the reward has within-group variance to steer on there today, while only 0.82 % ever contain a
    spelling the scorer's training corpus does not have -- so the direction an SFT support change
    uniquely reaches is the repair direction, not the diversity one.
44. **The homophone arm clears its admission bar overall and in the diversity direction, and fails
    it in the repair direction** (25). The contextual term outweighs the reconstruction term at
    ratio 1.26 over all swaps and 1.27 on swaps between attested spellings, but only 0.82 on swaps
    into a spelling the scorer never trained on, so the bar passes where sampling already supplies
    the variance and fails where the augmentation would be the only lever; per-class the split is
    real rather than uniform (knot/not 3.70 and too/two 3.88 against wood/would 0.49 and
    their/there/they're 0.54), and every delta is ~0.01 nats/unit in absolute size, which is small
    enough that the within-group reward spread has to be read before any of it is called steerable.
45. **No short-spelling bias in the reconstruction term** (25). Every in-class substitution is
    penalized, and swaps to a SHORTER spelling are penalized MORE than swaps to a longer one
    (median -0.0135 against -0.0073), which is the opposite ordering from the per-state
    orthographic-length price the arm was registered to watch for.

46. **The homophone init costs 3.11 dev-other WER, and essentially all of it is the augmentation
    reproducing itself rather than a degraded model** (27). theta_0^G_hom reads 16.67 / 21.45
    against theta_0^G's 13.89 / 18.34, and 96.7 % of the extra errors are substitutions within a
    homophone class -- the SFT learned the resampling distribution, at a plain-WER price of 78 % of
    what full reproduction of the uniform draw would cost. Outside the classes the two inits are
    within noise of each other (+53 errors, confidence interval -70 to +173).
    CORRECTION 2026-08-18: as first written this compared the realized rate to the 4.04 % of TRAIN
    tokens the augmentation rewrote, which is not like-for-like against a dev reference-token rate;
    the comparable figure is 4.58 % of reference tokens, and "96.7 %" is the share of NET extra
    errors (92.2 % of extra substitutions). The conclusion itself is unchanged.
47. **That price is not accommodated by the arm's pre-registered primary read** (27). The arm is
    registered to stay within 0.3 dev-other WER of D6-PERIODIC/GAN at every matched leg; its init
    starts 3.11 behind, so leg 1 fails that clause unless one GRPO leg closes ten times the margin.
    Whether the arm still runs is the planner's and the user's call, not a fact this log settles.

48. **The reward's language-model prior knows which spelling is right; the reconstruction term is a
    near-direction-blind preference for the text already sampled, and at lam_lm 1.0 the two cancel
    exactly where correction is needed** (28). The prior prefers the reference spelling on 90.6 % of
    swaps toward it and only 1.6 % of swaps away from it — sharply direction-sensitive and correct.
    Reconstruction prefers the swap on 17.9 % toward and 25.4 % away, i.e. it mostly opposes changing
    the sampled text whichever way the change runs. Their sum therefore rejects wrong spellings well
    (0.063) and is a coin flip on right ones (0.529). Raising lam_lm is the mechanical fix and is
    what the arm's own audio-free-null GUARD forbids.
49. **That coin flip is one homophone class, not a property of the reward** (28). `buy`/`by`/`bye`
    is 67.6 % of the toward-reference swaps at composed 0.446; the rest read 0.702. Why that class
    behaves so differently is UNEXPLAINED — the corpus-coverage hypothesis was tested and refuted —
    and it is 68 % of the sample any inference from this read rests on. Dropping it is a post-hoc
    slice and licenses no claim about the arm; the registered leg-4 rate bar is what discriminates a
    0.70-edge mechanism from a 0.45-edge one.

50. **On the distribution that matters the composed reward is NOT at chance -- it points at the
    correct spelling 82.5 % of the time** (29). Measured on the homophone policy's own errors rather
    than the plain policy's, the toward-reference rate is 0.825 with every one of the eight
    damage-carrying classes above chance, and the away direction is correctly rejected at 0.140.
    Conclusions 48-49's 0.529 is superseded as a prediction for this arm: it was measured on a
    near-orthogonal class distribution, and the correctly-weighted number sits near the top of the
    0.51-0.90 bracket rather than at its floor.
51. **Refitting the scorer on the policy's own decodes ENTRENCHES the spelling error rather than
    equalizing it, and the registration's mechanism claim has the wrong sign** (29). Holding dump
    and swaps fixed, the reconstruction term's toward-reference rate falls 0.684 -> 0.357 when the
    scorer is this arm's own refit instead of the plain arm's; the paired counts are 6:1 against
    (3480 vs 598) and 95 of 121 classes move. The composed reward survives at 0.825 only because
    the language-model prior at 0.970 outweighs it -- and that prior is an audio-free reader, so the
    composed rate sits 14.5 points BELOW the text-only null. Length-matching refutes the cheaper
    explanation: the entrenchment is -0.284 at equal character count, so it is spelling-specific
    learning and not an orthographic-length price, and the plain scorer's genuine equal-length
    discrimination (0.752) is what the refit collapses (0.469). In the deployed reward the cost is
    -0.069, not -0.327. This is a quantified instance of the standing
    G-track diagnosis -- a scorer refit on its own policy's output rewards that policy's correlated
    errors -- and it is not specific to the homophone arm, since every arm in the D6-PERIODIC family
    refits the same way. The open risk it names is compounding: each round refits on decodes the
    previous round's entrenched scorer helped produce.
52. **Periodic outer updates avoid the catastrophic continuous-joint failure, but they do not beat
    a good frozen scorer on the D-track** (17, 21, 22). The closest continuously trainable-scorer
    arm reads 5.12/9.27, 17.35/21.97 and 41.78/50.88 over three banked sub-epochs, with dev-other
    insertions growing 630 -> 21,406. Fresh periodic is 4.97/8.88, 4.65/9.02 and 5.28/9.27 at its
    first three legs, so holding the scorer fixed within each leg avoids same-step collapse. But it
    then worsens to 7.42/12.68 by leg 5, never beats the one-shot scorer's 8.64 best, and trails
    both the matched one-shot scorer and the original frozen control at that point. This comparison
    establishes a useful timescale, not a single-variable causal effect: the joint arm also differs
    in scorer topology, data partitioning, batching and optimizer continuity.
53. **Carrying scorer weights across periodic refits is harmful on this bed** (24). Fresh and warm
    are close through leg 3, but warm reaches 12.18/19.33 at leg 5 against fresh 7.42/12.68. The
    separation is an insertion failure: warm dev-other insertions grow from 479 to 5,874 while
    substitutions stay near 3.6k. The trajectories are not ended, but the completed prefix rejects
    warm inheritance as a stabilizer at this operating point.
54. **The plain GAN periodic gain is small, transient, and does not establish a recency benefit**
    (26). Leg 2 improves the no-loop init by only 0.45 dev-other, missing the registered 0.5 bar,
    and the arm worsens to 18.38/24.01 by leg 6. At the three matched points it is not decisively
    better than the frozen repaired scorer; later deterioration is substitution-led. No same-init
    continuously trainable-scorer arm exists, so D5(b)-b is not a causal control for this variant.
55. **The live GAN+HOM loop rapidly removes the augmentation's spelling damage despite the fixed-dump
    scorer-entrenchment diagnostic** (26, 27, 29). Its class-internal dev-other substitutions fall
    from 1,827 at init to 130 after one leg and 105 after three; total WER improves from 16.67/21.45
    to 12.80/18.08 and catches the plain periodic trajectory at leg 3. Thus conclusion 51 remains a
    valid statement about the reconstruction scorer on controlled swaps, but it does not predict the
    realized policy direction under the composed reward, whose Qwen3 language-model term dominates
    homophone spelling. The midpoint and final registered reads remain outstanding.
56. **WRONG in its description of the registered surface (2026-08-20 verifier): “D7's registered
    eight-donors-per-chapter-stratum construction is impossible.”** The external band and donor-
    capacity law were not registered; K=4 means two donors per chapter stratum. **Correction:**
    D7.0a proves that the original external donor statistic is not executable as written. Only
    276/1,500 immutable sources meet even the all-eligible raw K=4 degree requirement, so full-
    `E_all` K=4 is impossible; zero meet the conservative eight-donors-per-stratum diagnostic. The
    latter does not establish exact second-quartile support because the rank, boundary and tie laws
    are themselves unregistered. Later filters can only shrink a chosen surface. This triggered the
    prospective D7-v2 amendment, now frozen on 2026-08-21: training retains K=4 balanced Q2
    negatives, while the external donor-gap instrument uses one no-band nuisance-minimized donor,
    one table and donor load at most three on a coverage-gated `E_D`. Conclusion 57 records the
    prospective D7.0b read of that surface; D7.1 was never authorized before a pass.
57. **D7-v2 / D7.0b fails its preregistered training-support floor and is structurally unresolved**
    (31). The feature census passed every frozen checksum and found 28,538 feasible scorer rows,
    136,966 Q2 edges and 17,748 rows with at least two raw outgoing donors in both chapter strata.
    The exact common-set 2-in/2-out optimizer could admit only 56 rows from two speakers, versus the
    required 6,778 rows and 201 speakers. An independent iterative necessary-core calculation leaves
    at most 120 rows from four speakers, proving that the floor cannot be met by this registered graph
    rather than merely exposing a poor optimizer solution. The assignment job therefore stopped
    before external matching, and the loss preflight did not run. Per the frozen gate, this result
    permanently closes the offline-graph branch; it does not constrain the corrected online D7.
58. **USER-DIRECTED correction of active D7: retire offline donor graphs and test the reverse loss
    with online negatives on the full 960 h bed.** This does not reinterpret conclusion 57 or claim
    an experimental win. The corrected D7-GAN-SEQDISC uses all 281,241 theta_0^G-greedy pseudo-pairs,
    one dynamically resampled same-speaker duration-windowed donor per anchor, and no chapter/Q2,
    nuisance, capacity or regularity constraint. The matched full-bed scorer A/B is the next method
    read; a policy leg becomes eligible for separate launch authorization only after its label-free
    fixed-final gate.
59. **The D8.0 binding clause cannot be read on either frozen dump: its exclusion rule is not the
    law those dumps were scored under** (34). Clause (a) excludes structurally infeasible
    candidates at `d_min=2`, but both dumps predate the standing min-duration topology, and the
    theta_0^G artifact additionally joins a ~12.5 Hz pooled unit store (median length 169) rather
    than the raw 50 Hz store the pinned weight scorer uses (median 674). On its binding T=0.7 slice
    the reader calls 5,096 of 5,730 scored members infeasible while the artifact's own scorer
    returned finite scores for them — including all 512 greedy rows — so the exclusion is a property
    of the instrument, not of the policy. The same law costs the fork-epoch dump 18 of 101,190
    members. The registered read therefore returns UNRESOLVED, not the NO-GO the exclusion alone
    would produce.
    - **SUPERSEDED IN SCOPE (2026-08-22 ruling, verdict 62):** the frame diagnosis is confirmed,
      but "cannot be read on either frozen dump" is too strong -- the clause is readable on the
      existing dumps once the exclusion is joined to the operative raw 50 Hz store, with no new
      dump and no scorer forward.
60. **Which reading clause (a) takes decides it outright, in opposite directions** (34). On the
    binding slice the median distinct support is 0 of 13 with the exclusion applied and 12 of 13
    without it, against a threshold of 3. No intermediate outcome exists, so the clause cannot be
    reported as a measurement until the plan fixes the reading; this is a specification question,
    not a noisy statistic.
    - **SUPERSEDED (2026-08-22 ruling, verdict 62):** neither offered reading was accepted -- the
      dedup-only count ignores the registered exclusion, and the as-run exclusion is the wrong
      frame. Under the ruled third reading the two collapse into one number, 12, because the
      operative-frame exclusion is empty on this slice.
61. **On the one artifact whose scorer law the reader nearly matches, no D8 clause fires** (34).
    The fork-epoch dump gives median distinct support 3 under both readings, every grid tau inside
    the [1.5, 8] ESS band, token count explaining 0.39 of within-group weight variance at
    `tau_star = 1.0`, and shaped-versus-acoustic-only spearman 1.0 against shaped-versus-LM-only
    0.5. Its policy, bed and scorer are all wrong for D8.1a, so this reports and binds nothing;
    the arm-selection rule reads only D8.1a statistics.
    - **CONFIRMED at v3 (2026-08-22):** unchanged to the last digit, because this dump already
      joined the raw 50 Hz store. Its 18 exclusions in 101,190 members are the genuine rate the
      ruling prices the 5 % safety valve against.
62. **Read in the operative frame, the D8.0 binding clause PASSES with room** (34). On the
    theta_0^G T=0.7 slice the median distinct feasible support with the greedy member included is
    **12 of 13** against a threshold of 3, and the operative-law exclusion removes **0 of 5,730**
    scored members -- zero on every one of the five slices, against 5,096 under the pooled-store
    join. That is the frame diagnosis closed by construction: the entire v2 exclusion was the
    instrument. Verdict GO; D8 does not close at D8.0.
63. **Reported at D8.0, binding nowhere: the shaped weights track the LM-only weights closely on
    the operative policy** (34). Median spearman between shaped and LM-only weight vectors is
    0.9790 on the binding slice and 0.978-1.000 across all five, while shaped versus
    acoustic-only runs 0.2857-0.6593; both are now columns of the v3 table in approach 34. The
    registered arm-selection rule reads only D8.1a statistics on the operative bed and scorer, so
    this selects nothing and funds nothing; it is logged because a value above the rule's 0.95 line
    would, if it survived to D8.1a, leave only candidate-acoustic funded. Clauses (b) and (c) fire
    nowhere at v3: at least one grid tau sits inside the [1.5, 8] ESS band on every slice, and
    token count explains 0.035-0.413 of within-group weight variance at `tau_star`.
    (Correction 2026-08-22: the shaped-versus-acoustic-only low end was first transcribed as 0.30;
    the T=0.5 slice reads 0.2857. Direction-neutral -- the verdict binds nowhere either way.)

64. **The D7 own-infeasible drop set is exactly the four registered train-role rows, confirmed
    per arm from each arm's own artifact** (32). Both `monitors.json` files carry
    `own_infeasible_dropped = {"train": ["3488-85273-0024", "3889-130125-0028", "4492-8904-0032",
    "8424-284526-0028"]}` with `anchor_rows = {"train": 267175, "internal_held": 14062}`, digit-identical
    to the offline end-to-end dropcheck on the same pool inputs and to each other. This closes the
    check that was deferred at relaunch; nothing was inferred from the runs merely starting.

65. **D7.1 reached its fixed final endpoint on both arms, and its two banked held statistics point
    in opposite directions** (32). The candidate's internal-held mean `L_online` is 0.007541 against
    the control's 0.010225, i.e. 26 % lower, which is the direction the online same-speaker negative
    is meant to produce; its internal-held per-frame NLL is 2.5319 against the control's 2.5259, i.e.
    0.0060 higher. Both are single point values from the arms' own `monitors.json`, computed on one
    donor draw per held anchor, so neither is the D7.2 statistic: the registered admission recomputes
    `L_online` with 32 stateless donor draws per eligible held anchor under a paired speaker-cluster
    bootstrap, and adds the 1,500-row Acceptance gate v2 and scorer parity. What D7.1 establishes is
    only that the A/B ran matched to its endpoint and produced two fixed-final scorers.


## Catalog

`T/` = `work/i6_core/returnn/training/`, `F/` = `work/i6_core/returnn/forward/`,
`S/` = `work/speech_llm/sae/`.

| artifact | path |
|---|---|
| code | `sae/scorer_diag.py`, `sae/text_repair.py`, `sae/psi_align_jobs.py`, `sae/psi_align.py`, `sae/curate.py`, `sae/gate_table.py`, `sae/refresh_gate.py`, `sae/d7_census.py`, `sae/d7_v2.py`, `sae/d7_online.py` (+ focused tests; D7.0a commit `a0a22b4`, D7-v2 commit `7b2069d`, D7 resume-RNG and infeasible-donor counter `1d10945` on speech-llm `haotian_modality_matching_jupiter`). `test_psi_align.py`'s CUDA/python lattice parity test now also carries two `d_min=2` skip_ok cases, so the topology D7 trains in is pinned; executed on a GH200 2026-08-21 (`log/parity_test.1445759.out`, passed, not skipped) since the login node has no GPU. |
| entry points | `config/sae_3e1_d0.py`, `config/sae_3e1_usage.py`, `config/sae_3e1_d1d2.py`, `config/sae_3e1_d3.py`, `config/sae_3e1_d4.py`, `config/sae_3e1_d4p.py`, `config/sae_3e1_d5b.py`, `config/sae_3e1_d6.py` (builds D4' and the swap-in too), `config/sae_3e1_d6periodic.py`, `config/sae_3e1_d6periodic_warm.py`, `config/sae_3e1_hom.py`; D7 tracked canonical configs `src/speech_llm/prefix_lm/sis_recipe/exp2025_11_06_speech_llms/librispeech/configs/config_sae_3e1_d7_0a_v1.py` at `a0a22b4` and `config_sae_3e1_d7_v2_v1.py` at `7b2069d` (workspace wrappers only delegate) |
| D8.0 registered feasibility reads (approach 34) | **operative v3** `S/d8_feasibility/D8FeasibilityReadJob.mv2d0vkWN93a` (theta_0^G, binding, GO) and `.W7TWfwoZtkaC` (fork epoch); superseded v2 `.mDQ2LoAzrMTE` / `.ulUbBcxIiJtf` and v1 `.iCuYuvkL6bwr` / `.onK5ekDuoLLA`, kept as the evidence that motivated the guard and then the ruling |
| D8.0 code and entry point | `sae/d8_feasibility.py`, `configs/config_sae_3e1_d8_0_v1.py`, `config/sae_3e1_d8_0.py`, `scripts/d8_0_mechanics_test.py`, 47 synthetic-only checks at v3, all passing (the `889750c` commit message says 32, which was the v1 count) (speech-llm `889750c`, v2 guard `a3dd6c7`, operative v3 `3843918`) |
| D7 own-infeasible-anchor drop law and its verification | speech-llm `e2a421b`; `scripts/d7_make_items_dropcheck.{py,json}` |
| D7.0a complete raw external/scorer edge tables and census (approach 30) | `S/d7_census/D7RawDonorCensusJob.zsnx1p9nLyV3` |
| D7-v2 / D7.0b feature and fail-closed assignment jobs (approach 31); the downstream loss preflight never materialized | `S/d7_v2/D7V2FeatureJob.hnReOv8t9UWg`, `S/d7_v2/D7V2AssignmentJob.aSOMkw3hSc0K` |
| D7.0 parity diagnostic (approach 32): the read-only GPU reproduction of the preflight's parity clause | `scripts/d7_parity_diag.py`, output `log/d7_parity_diag.1446568.out` |
| D6-PERIODIC/GAN960-FROZEN graph (approach 33) | `config/sae_3e1_d6periodic_gan960_frozen.py` -> `configs/config_sae_3e1_d6periodic_gan960_frozen_v1.py`; init `T/ReturnnTrainingJob.HuSkdbuVRg6d` ep10; frozen scorer `S/psi_align_jobs/PsiAlignTrainJob.dsMKgPHQApyR`; legs 1/2/8 `T/ReturnnTrainingJob.ohmLWWmr6Kxe`, `.liehXoiGoRI0`, `.V1WEV1giQXZA` |
| corrected D7-GAN-SEQDISC graph (approach 32) | `config/sae_3e1_d7_gan_seqdisc.py`; pool `S/d7_online/D7OnlinePoolJob.XLjSgTzHfwAu`; preflight `S/d7_online/D7OnlinePreflightJob.ZxfANwBZYpaI`; fixed-final control/candidate `S/d7_online/D7OnlineTrainJob.j16rTskXF1QU`, `.WA1bqjXQtzeZ` |
| D7.1 fixed-final scorers, both arms complete | control `S/d7_online/D7OnlineTrainJob.j16rTskXF1QU/output/model_final.pt`; candidate `S/d7_online/D7OnlineTrainJob.WA1bqjXQtzeZ/output/model_final.pt`; per-arm `monitors.json`, `sampling.json`, `train.txt` beside each |
| D6-PERIODIC legs 1-8 (approach 22), parent sub-ep 3-10 | `T/ReturnnTrainingJob.5FqdnhWTOf1f`, `.BTnU1gSuMG0i`, `.ZKCbq529Hgp8`, `.gFNpNmXwvrsc`, `.nQtnPdKCuJ0m`, `.n8abYvLR4IP5`, `.jGj7TTbW5DTm`, `.wWqYY7iOCw1s` |
| its per-boundary refits (rounds 2-8) | `S/psi_align_jobs/PsiAlignTrainJob.JWV3InILYF5v`, `.yUUSN2Hx96E0`, `.QMO8VcAtZ6Gi`, `.DzhBWCy61tiN`, `.Vha8vvKu9lWk`, `.RGTtwlQHt3HY`, `.Ls0TQGiyhQbf` |
| D6-PERIODIC-WARM legs 1-8 (approach 24), parent sub-ep 3-10; leg 1 is approach 22's, shared | `T/ReturnnTrainingJob.5FqdnhWTOf1f`, `.OOr3UybqUEHD`, `.X3biCvDKgQ7N`, `.7dANeLqxFFbq`, `.nd92xaRDY0uw`, `.kkh0u4rI7I6D`, `.kQRZtXc1ubTV`, `.oRbUsmYR6fRT` |
| its warm-started refits (rounds 2-8) | `S/psi_align_jobs/PsiAlignTrainJob.2TDm8VwIZzjv`, `.frtMcQ6wvR4s`, `.ENcr81sGwHfp`, `.3tMeo1Meuceg`, `.ZeEsJq6JOdNx`, `.34mTYfJioAsm`, `.3JLOhu5PSKwj` |
| BOTH ARMS RELAUNCHED 2026-08-18 with no acceptance step (user ruling). The ids above are the ungated arms; the gated run's legs 2-5, its boundary fits for rounds 3-6, and every clause table and verdict job were deleted the same day, so approach 22's table is the only record of what that acceptance step decided. Surviving from the gated run: leg 1 and both arms' round-2 fits, whose hashes the change did not move. | -- |
| all three arms' error anatomy at matched points | `S/scorer_diag/PolicyAnatomyJob.Cda1gPFxLM2V` |
| D6-PERIODIC/GAN legs 1-8 (approach 26), G-track sub-ep 1-8 | `T/ReturnnTrainingJob.kr1foUV6lecx`, `.AuzMGgyskdJT`, `.KD73Hc4eGDfW`, `.E6s3lUUaodzw`, `.J9m38fxEwXl4`, `.AS1g33qDo28i`, `.QTQuYQnppmSs`, `.cR8Q29Pmfuhy` |
| its per-boundary refits (rounds 1-8; no gate on this track, each one serves the next leg) | `S/psi_align_jobs/PsiAlignTrainJob.dsMKgPHQApyR`, `.7jHYVGToyWPR`, `.M2Z0M9UpKW98`, `.rdkbJsLOLEJW`, `.YPyCrmgjglsj`, `.jMaYmBUAffMb`, `.NM6sQa0D9uQM`, `.wPujQSh4PLSd` |
| D6-PERIODIC/GAN+HOM loop legs 1-4 currently materialized (approach 26; first three finished) | `T/ReturnnTrainingJob.JocWKAmYroFJ`, `.dp0XmU5Mm9V5`, `.tpby6E3kTeSE`, `.JBaqJExxDKGz` |
| its comparators at matched sub-epochs, both held: frozen contaminated psi_align^G and the frozen repaired scorer | `T/ReturnnTrainingJob.2fb02hGUdHNj` is the init they all start from; the two arms' own WER rows are in `SAE_3A.md` approach 10 and this log's approach 9 |
| HOM-0a class statistics (approach 23); its `classes.json` carries every class with per-member LM and corpus counts | `S/homophone/HomophoneClassStatsJob.our76yheSD0c` |
| HOM augmented corpus (uniform in-class resampling, seed 0, train split only); `word_hyps.json` is the SFT-ready drop-in | `S/homophone/HomophoneAugmentJob.k2OwZiTcKpEG` |
| theta_0^G_hom, the HOM arm's policy init: theta_0^G's own builder with that corpus as targets and every other argument shared (launched 2026-08-18 on the user's greenlight) | `T/ReturnnTrainingJob.EabxlDlT0oji` (corpus dataset `work/i6_core/datasets/huggingface/TransformAndMapHuggingFaceDatasetJob.157IDJgBOv9H`) |
| HOM-0b swap measurement (approach 25), 8000 utterances, <= 4 single-word swaps each | `S/homophone_probe/HomophoneSwapScoreJob.gN7mZ0EcPhsS` |
| HOM-0b read against the pre-registered bar | `S/homophone_probe/HomophoneSensitivityJob.xB5RvcgLVgtD` |
| HOM direction read (approach 28): which spelling each reward term points at, reference joined back to the same swaps | `S/homophone_probe/HomophoneDirectionJob.Uo4UAJp5Ue42` |
| HOM round-1 diagnostic on the arm's own dump (approach 29): refit `S/psi_align_jobs/PsiAlignTrainJob.ACP3LqKDUSQ0`, swaps under own/plain scorer `S/homophone_probe/HomophoneSwapScoreJob.IG6wFl5QWnld` / `.iRCxGqNRxQha`, direction `S/homophone_probe/HomophoneDirectionJob.deNc7xXnCfSu`, sign test `S/homophone_probe/HomophoneScorerDeltaJob.JKbbRWimojlI` |
| HOM-0c coverage on the label-free init's G=12 T=0.7 full-bed dump (`F/ReturnnForwardJobV2.lQMOR5n2ntcS`) | `S/homophone_probe/HomophoneCoverageJob.F76iJ8j0AQi1` |
| the round-1 artifacts both reads run on: dump, refit corpus, scorer | `F/ReturnnForwardJobV2.66pIzBzffnK2`, `S/curate/GreedyPoolJob.Yv6qBpz0UC0U`, `S/psi_align_jobs/PsiAlignTrainJob.dsMKgPHQApyR` |
| D6 swap-in arm, best bed (approach 21a) | `T/ReturnnTrainingJob.YUh6Gzvavctf` |
| its last three sub-epochs, half micro-batch | `T/ReturnnTrainingJob.qQeSijpUKP2k` |
| both arms' error anatomy at every matched point | `S/scorer_diag/PolicyAnatomyJob.pxqfrYx23Rth` |
| its frozen control, same fork, sub-ep 3–10 | `T/ReturnnTrainingJob.vhyvv2waeU16` |
| D6 G-track min-duration refit (approach 21b), best epoch 23 | `S/psi_align_jobs/PsiAlignTrainJob.TicugJYx52p2` |
| its clause tables (vs the `r1` incumbent / vs `psi_g_tc100`) | `S/gate_table/PsiGateClauseTableJob.qYRE7JWyUcJQ`, `.H9QbX4VgXAwf` |
| its re-rank and text probe | `S/psi_align_jobs/PsiAlignRerankJob.BRfnFlMK1job`, `S/psi_align_jobs/PsiTextProbeJob.mSJzvpTBW0Y3` |
| its comparator, the G-track round-1 refit | `S/psi_align_jobs/PsiAlignTrainJob.cRIigmxPtt75` |
| round-1 uncurated corpus (59 878 rows) | `S/curate/UncuratedPoolJob.1RgS3KEtkdEy` |
| round-1 uncurated refit (approach 19) | `S/psi_align_jobs/PsiAlignTrainJob.Be8yVs7MaLrS` |
| D6 rung 1 price-steering sweep | `S/psi_align_jobs/PsiPriceSteerJob.4Eqth3bY2Zc2` |
| D6 rung 2 corruption-margin refit | `S/psi_align_jobs/PsiAlignTrainJob.zjUitbvGbDg3` |
| D6 rung 3 min-duration refit (d_min=2) | `S/psi_align_jobs/PsiAlignTrainJob.wlruSpBK1EDP` |
| the same refit on the CUDA forward-backward backend (conclusion 38) | `S/psi_align_jobs/PsiAlignTrainJob.QhaW4lUpbkl6` |
| D6 rungs 2+3 combined refit | `S/psi_align_jobs/PsiAlignTrainJob.HVjMgYBlJ4tp` |
| D4' round-1 clause table (no winner, c33) | `S/gate_table/PsiGateClauseTableJob.hRgVjm5bYRKI` |
| D4' round-1 re-rank on the fork dump | `S/psi_align_jobs/PsiAlignRerankJob.DU9JY7WG9b0y` |
| D6 clause table (c35, c36) | `S/gate_table/PsiGateClauseTableJob.JdrWdaCm7UeG` |
| co-trained replay arm (on hold, ep1–7 kept) | `T/ReturnnTrainingJob.KBTADeS7Qp1G` |
| frozen AR every loop arm starts from (ep0) | `T/ReturnnTrainingJob.ExCoQDKtXAGH/output/models/epoch.050.pt` |
| ep0 usage cells (reused, finished) | `F/ReturnnForwardJobV2.GBuKgHp3GNlz` (true) / `.HKsuKQJdUwGA` (shuffled) |
| usage trajectory | `S/scorer_diag/ArUsageTrajectoryJob.9Ughq5htDaXx` |
| D0 rollout set (theta_0^G, 512 utts, G=12, 5 T) | `F/ReturnnForwardJobV2.J9yA1eYnxwYA` |
| D0 bed units (enc50_raw) | `S/quantize_states/AssignUnitsJob.X8DBup0jQlhR` |
| D0 re-rankings (scorer: `PsiAlignTrainJob`) | `S/psi_align_jobs/PsiAlignRerankJob.QdHRXsev2Txh` (psi_g_tc100 ← `.kSYy0ADBgPGo`), `.2AUBSd8Y0oq0` (psi_g_seed ← `.SUAAuCS2o3pz`), `.bZCAVAKWQq3I` (gold_enc50 ← `.IN3zmmGpH4Bv`) |
| LM-corpus rate reference | `S/scorer_diag/LmWordCountsJob.SqAFPqiRBD9k` over `i6_core/tools/download/DownloadJob.g4jClO48cAvP` |
| suspect vocabulary | `S/scorer_diag/SuspectVocabJob.7LSZhTXKculV` |
| D0 table | `S/scorer_diag/RolloutMechanismJob.vsl00qaCHQbP` |
| D0 coverage/steerability vs T | `S/scorer_diag/CoverageTemperatureJob.JAP5gJQE0PwP` |
| frozen external held pair set (1500 dev pairs) | `S/scorer_diag/FrozenHeldPairsJob.E8UaEwRF65HW` |
| D1 held NLL (gate v2 i/ii) | `S/psi_align_jobs/PsiHeldNllJob.J1A028bt3Faw` (psi_g_tc100), `.WrmDwFU9dVvV` (psi_g_seed), `.ag5DZ3A2Gd1K` (gold_enc50) |
| D1 probe battery | `S/psi_align_jobs/PsiTextProbeJob.eNVc8JTbm7n8` (psi_g_tc100), `.qo8IB8MLA8ES` (psi_g_seed), `.rY39iGv8bhhi` (gold_enc50) |
| D2 LM per-line multiplicity reference | `S/text_repair/LmLineStatsJob.l9ZJSEj8tP0S` |
| D2 repaired pseudo-text | `S/text_repair/RepairPseudoTextJob.o086K9a8uXDa` |
| D2 corpora (control / repaired) | `S/text_repair/TextHfDirJob.UEAxxdGitOHu` / `.7Msi4BxlykgV` |
| D2 candidate scorers | `S/psi_align_jobs/PsiAlignTrainJob.HTy12IMDmYdB` (d2_rate), `.DnBJxqz4sNQZ` (d2_contrast), `.9pTbjjx29yVc` (d2_both), `.hxK0HTBZQSJa` (d2_states) |
| D2 held NLL, same arm order | `S/psi_align_jobs/PsiHeldNllJob.XvvciDyN3LyS`, `.Z8quArGjzAj3`, `.1okicjOpTszW`, `.9D2ywKhnL5ZH` |
| D2 probe battery, same arm order | `S/psi_align_jobs/PsiTextProbeJob.wkAV3KfAUwW9`, `.g3p5aA7nBONQ`, `.8LGrp6IuVyzD`, `.eRBqqPfUtf6k` |
| D2 re-rankings (carry `rollouts.jsonl`), same arm order | `S/psi_align_jobs/PsiAlignRerankJob.jRvegq7Bf7lu`, `.zAzQGZbtxrw9`, `.DQQLmfIhPTOe`, `.DVcQhryzLU2j` |
| D2 parity, same arm order | `S/psi_align_jobs/PsiScorerParityJob.bBjvefspGS4L`, `.0U0yG8pdt6fB`, `.O7WiXL0OfmvA`, `.g5gIUiLRMqLg` |
| D2 cross-arm D0-dump re-read (beta, spearman, sel_wer) | `S/scorer_diag/RolloutMechanismJob.uDTs6ZlhOFQa` |
| state-matched probe battery (approach 10; carries `items.json`) | `S/psi_align_jobs/PsiTextProbeJob.WBXWwmZIK7HY` (psi_g_tc100), `.4KpANAZV864A` (psi_g_seed), `.a8WsW4jjddcq` (gold_enc50), `.jQPGx36tCccz` (d2_rate), `.cMO136SC9uUu` (d2_contrast), `.rNCJA9Y987bY` (d2_both), `.lcbBuAIimK11` (d2_states) |
| D2 steerable coverage vs T, per candidate | `S/scorer_diag/CoverageTemperatureJob.Ku9zNUNDK12D` |
| D3 control arms (frozen `d2_contrast` psi) | `T/ReturnnTrainingJob.rJWSC5xOsrf2` (shaped), `.L6FwOOpffNL4` (recon) |
| D6-PERIODIC/GAN-FROZEN schedule-only control | round-1 scorer `S/psi_align_jobs/PsiAlignTrainJob.dsMKgPHQApyR`; policy legs `T/ReturnnTrainingJob.kr1foUV6lecx` (reused), `.JVfEDCPIPWkq`, `.o2GFVkZZPNRT`, `.fEvotypkqDao`, `.91wIJ5JpsdIW`, `.2p2hpz7nk5vd`, `.ZgRzUxDRhajE`, `.ycoJLypxisD7` |
| D4 entry point | `config/sae_3e1_d4.py` |
| D4 (b) selector affinity, 5 arms | `S/scorer_diag/RolloutMechanismJob.jYDxg98sWJIj` |
| D4 (c) clause table, 7 finished arms | `S/gate_table/PsiGateClauseTableJob.x0d7dYpOdilI` |
| D4 round-1 dump (theta_0^G, 28539 utts, G=12, T=0.7) | `F/ReturnnForwardJobV2.lQMOR5n2ntcS` |
| D4 round-1 chain (curate -> refit -> gates -> clauses) | `S/curate/CuratePairsJob.0Xs8AhGwRn80`, `S/psi_align_jobs/PsiAlignTrainJob.cRIigmxPtt75`, `.PsiHeldNllJob.Q24MX1AhUGFK`, `.PsiTextProbeJob.UEVRWgPseI16`, `S/gate_table/PsiGateClauseTableJob.5oMRtYKrhE3C` |
| D5(a) entry point | `config/sae_3e1_d5a.py` |
| D5(a)-1 anatomy | `S/scorer_diag/PolicyAnatomyJob.eMeWgTsMWSRM` |
| D5(a)-2 allegiance grid | `S/scorer_diag/AllegianceGridJob.kR0YA9kfUd4s` |
| D5(a)-3 rerank sweep, ep0-ep4 (each carries `rollouts.jsonl`) | `F/ReturnnForwardJobV2.p9y6xUfCZ4sW`, `.4nTgyBPY2SlM`, `.dEOPiBW4ADQM`, `.aoynOYDHBqLs`, `.9px8IEReJyUG` |
| fork/D5(b) code | `sae/fork_screen.py`, `sae/psi_forensics.py` (+ their two test modules, 11 tests) |
| fork/D5(b) entry points | `config/sae_3e1_fork.py`, `config/sae_3e1_d5b.py` |
| fork parent (the frozen control, running) | `T/ReturnnTrainingJob.vhyvv2waeU16`, sub-ep 2 = `output/models/epoch.002.pt` |
| fork screen | `S/fork_screen/ForkPointScreenJob.avOkAB1TUN3d` |
| joint-psi arm, first launch (OOM in sub-ep 1; superseded, conclusion 25) | `T/ReturnnTrainingJob.eYhb6alu9OIQ` |
| joint-psi arm (D5(b)-b, at `batch_size` 1e6 / accum 2; stopped on hold after sub-ep 3, ep1–3 kept) | `T/ReturnnTrainingJob.jQmmGy2yGtGR` |
| — its sub-ep 1 / sub-ep 2 WERs (dev-clean, dev-other) | `ScliteJob.{onJeeX0UOiRy,RYa3OTRBO2Uf}` / `.{1qm9kIUcj2y6,49zgvrMKwznh}` |
| frozen gold-pair forensics set (1500 seed-dev pairs, enc50 units) | `S/psi_forensics/HfSplitTextJob.hITA2tWgTklY` -> `S/scorer_diag/FrozenHeldPairsJob` |
| bed psi (ep0 of the forensics, frozen in the parent) | `S/psi_align_jobs/PsiAlignTrainJob.IN3zmmGpH4Bv` |
| D4' entry point | `config/sae_3e1_d4p.py` |
| D4' suspect set, re-derived on this bed | `S/scorer_diag/SuspectVocabJob.UG1VLQjflE7G` |
| D4' round-1 dump (fork policy, tc100, G=12, T=0.7, psi-ranked) | `F/ReturnnForwardJobV2.QbIYruVEI0fF` |
| D4' filler watch (minimal-state, monitor-only) | `S/scorer_diag/FillerWatchJob.3x3IRoxcQSha` |
| D4' selector admissibility on this bed's dump | `S/scorer_diag/RolloutMechanismJob.UJ0DfPXTH8Cq` |
| D4' incumbent battery (psi0_gold: held, probes, rerank, parity) | `S/psi_align_jobs/PsiHeldNllJob.yMQGlcL3OVVj`, `.PsiTextProbeJob.pBrTx11FPZvS`, `.PsiAlignRerankJob.pJONTykQhQaS`, `.PsiScorerParityJob.gRkOlabxfLVY` |

Both configs pin their inputs by absolute path + `hash_overwrite` instead of importing
`config_sae_3a_enc50_units_v1` / the replay arm's graph: those graphs belong to the running
`sae_3a_gan_loop` manager and to a training job on hold, and sisyphus has no cross-process lock, so
an import would put a second manager over jobs another one owns.

`config/sae_3e1_d1d2.py` follows the same pinning rule and shares no job with the running loop
manager. `config/sae_3e1_d3.py` does NOT: it builds its arms through
`config_sae_3a_gan_loop_960h_v1.baseline` so the control differs from the arms it controls in the
scorer alone, which pulls in the finished 960 h unit and dataset graph — run exactly one of the two
managers. D3's cost on that bed is ~5.3 h per sub-epoch on 4 GPUs, i.e. ~85 GPU-h for two arms at four
sub-epochs, against the ladder's "~9–18 GPU-h" estimate.

Two measurement caveats a later reader needs. The OOV-count null is INERT on this bed — `n_oov` is 0
for all 6144 rows because the psi inventory carries no UNK state, so `neg_n_oov` is undefined rather
than uninformative. And the (c) covariate is the rollout's own WER, which controls the filler's
direct insertion cost but not the composition of the remaining errors; the gold-text control arm, not
the absolute beta, is what carries the contamination claim.

## Verifier feedback

- 2026-08-07: full audit clean — every logged number in all three tables reproduces from the
  cited job outputs (usage trajectory row-by-row; all 15 scorer-arm statistics; selector CIs;
  suspect vocab to five decimals), and the statistics code is verified: the group-centred
  partial beta recovers planted effects and stays zero when the count acts only through WER;
  the shaped rebuild (sum / n_units) exactly matches live `lm_prior_norm="units"` semantics
  including the dump's per-token norm; the derangement is asserted fixed-point-free and the ep0
  reuse is hash-asserted.
- 2026-08-07: conclusion 3's frac_pos 0.95 (shaped) is 0.9450 on the all-groups convention used
  everywhere else (0.9452 is the spread-subset value) — transcription slip, direction-neutral.
- 2026-08-07: the "(arm-invariant)" label on the selector block overreaches: `psi_len_only` and
  `neg_n_oov` are recomputed per arm by each `PsiAlignRerankJob` (each arm's own zero-emission
  forward / own lexicon encoding); only lm_prior_units / neg_n_suspect / n_tokens come from the
  shared dump by construction — n_oov coincides because the arms share the lexicon config, and
  psi_len_only genuinely differs across arms (hence its logged range). Conclusion 7's
  length-artifact reading survives (the CI straddles zero in every arm). Minor: the selector CIs
  run on 509/505/438 groups after degenerate-group filtering, vs 512 in the ranking block.
- 2026-08-07: conclusion 7 is necessary-not-sufficient for a curated refresh — no filler-affinity
  statistic exists for the selector itself (the partial effect of suspect count on
  lm_prior_units at matched WER; `_bias` runs only on the two reward keys). An external word LM
  plausibly PAYS for a high-frequency function word, so this one arm-invariant row from the same
  dump is required before lm_prior_units is admitted as a curation view; registered as a D4
  admissibility condition in `PLAN_3E1.md`.
- 2026-08-07 (D1/D2 audit, numbers): approaches 4–7 reproduce from the cited job outputs to the
  printed precision (probe battery incl. paired CIs; coverage_T to <1e-12; the repair's exact
  joint rate solve; the frozen held draw re-executed byte-identically, disjointness from all
  three training corpora confirmed) — except two cells: gold_enc50 held ce_loo is 3.1385
  (`PsiHeldNllJob.ag5DZ3A2Gd1K`; the logged 3.1274 is the probe job's 1442-pair value
  transcribed into the 1493-pair table — the row's derived columns already use 3.1385, so
  ordering is unaffected and conclusion 9's "3.13" reads 3.14), and the insertion-ladder
  spearman range "0.66–0.86" is psi_g_tc100 only — across arms it is 0.55–0.87 (psi_g_seed
  filler_ins 0.5516), which strengthens, not weakens, the ins-vs-sub/del contrast.
- 2026-08-07 (D1/D2 audit, code): conclusion 8 flipped — correction under it. The LM insertion
  control is drawn frequency-proportional with NO state-length matching
  (`psi_align_jobs.py:1804-1812`); "to" is one emitting BPE state vs ~2.7 for the mean draw, and
  53–81 % of the 0.0584 discount is state-count-attributable. The lattice reading (c10) survives
  SHARPENED: all three scorers charge the same ~0.031–0.035 nats/frame per inserted emitting
  state, so the cheap-insertion exploit is open to EVERY minimal-state word — contamination chose
  which word, not whether. The decisive equal-state contrast ("to" vs IN/IT/HE) is not computable
  from the dumps (`per_item` stripped before `probes.json`, `:1887`); a length-matched control
  pool (1-emitting-state words, ~18 % of current pool mass) with per-item dump is registered as a
  D1 amendment in `PLAN_3E1.md` and is required for any D2 admission read. c9's power-check
  failure itself stands on the paired design.
- 2026-08-07 (D1/D2 audit, gate): the held set's provenance (the §1d decoder's own dev output)
  domain-confounds gate v2 (i)'s improvement clause — held ce_loo orders the three arms by
  training-text domain match (2.72/3.02/3.14), not quality, so a repaired-text candidate would be
  structurally rejected against the unrepaired incumbent; amendment registered in `PLAN_3E1.md`
  (floor-only for changed-text candidates), flagged for the user's blessing.
- 2026-08-07 (approach 8/9 + c12 audit): c12's numbers all reproduce from `probes.json`
  (`ce_emis` + `nll` per corruption; there is no transition field — the "transition" is the
  NLL-minus-emission residual, i.e. transitions plus alignment entropy) but the sentence
  misreads them — correction under c12; the surviving core is the d2_states prediction,
  verified: orphaned frames are chars_per_state-invariant while inserted-word state counts
  scale ~2.5x. Approach 8 is verified: d2_states wiring correct (cps 0.5, contaminated control
  corpus, weight 0), 2 T/U 9.7652/3.9180 reproduce (states are per BPE token,
  min(8, max(1, round(chars/cps))), plus n_words+1 SIL; pooled sum T / sum U), the byte-equal
  claim is text-column + id-order equality (`text_repair.py:376-381`, asserted, 0 diffs), and
  there is NO feasibility shift at cps 0.5 — 1500/1500 held pairs feasible (worst U/2T 0.365),
  probe-common 1442 with identical membership (the 58 exclusions are <=4-word utterances where
  k=4 is undefined, not lattice-infeasible); only the rollout set's feasibility under 0.5
  remains unchecked. Approach 9's rule is PROSE-ONLY (`WINNER = None`; `PsiHeldNllJob` reports
  clause i/ii but nothing combines the rule), its ~0.005 threshold is the incumbent's own LEVEL
  CI half-width (0.00485) while the rule thresholds a between-arm REDUCTION — the right
  instrument is the paired cross-arm difference CI on the shared 1442 utterances, computed
  nowhere — "corruption-ladder spearman" names none of the five ladders, and under cps 0.5 the
  filler-vs-control state-count ratio is preserved (1.855 vs 1.866) so the unmatched discount
  moves mechanically with the per-state price; the plan's winner-rule amendment (state-matched
  pool, intersection reads, paired difference CI) is the operative rule.
- 2026-08-08 (approach 10 audit, numbers): every published cell reproduces — matched-discount
  levels and paired point estimates bit-for-bit from items.json (which reproduces every
  probes.json aggregate to <1e-12), the ladders-worse column 3/1/3/0/0/0 under the paired-CI
  convention, pool facts 6472/57/51 and mean states 2.6985/8.1603 re-derived independently from
  bpe.codes + the LM counts, and the frequency-drawn side of all seven arms bit-identical to the
  pre-extension jobs (0 differing leaves — the rng split is proven, not claimed).
- 2026-08-08: the logged paired-CI ENDPOINTS are not reproducible from any pinned seed (17 of 24
  differ in the 4th decimal, max 0.0007) — verdict-neutral except two boundary calls: d2_states'
  k=1 zero-exclusion flips with the bootstrap seed (t-test p=0.046; c15's "both ... CIs excluding
  zero" overstates it at k=1, solid at k=2/k=4), and d2_contrast-over-d2_states at k=4 excludes
  zero by only 0.0003-0.0006. No seed or resample count is pinned for statistics computed outside
  any job — the clause-table job below fixes this.
- 2026-08-08 (winner-rule application): the arithmetic reproduces but the winner turns on two
  UNPINNED clauses — (a) clause (ii) is algebraically clause (i)'s improvement comparison
  sign-flipped (H_uni bit-identical across arms), so d2_both, a changed-text candidate, is
  eliminated by exactly the comparison the gate v2 (i) floor-only amendment ruled inadmissible,
  and it is the only thing removing d2_both (argmax unchanged at k=1/k=4 if admitted; at the
  omitted k=2 d2_both out-reduces d2_contrast, n.s.); (b) the ladder floor's "not below" is
  CI-read in the log but point-read in the rule text, and under the point reading only d2_states
  is eligible — the winner flips to d2_states. Pins proposed in `PLAN_3E1.md` (need the user's
  blessing); the hard-coded WINNER='d2_contrast' in `config_sae_3e1_d3_v1.py:37-38` is
  provisional until then. Also: d2_states is admitted through the improvement halves of (i)/(ii)
  on ce_loo numbers approach 8 itself marks (*) cps-incomparable — only the absolute floors bind
  for it.
- 2026-08-08 (rollout columns): approach-8 D2 table verified exact incl. sel_wer (incumbent
  0.1380 strictly lowest), parity 0.0 x4, G3 bars (margins 0.1464-0.1540, CIs overlapping), and
  suspect state mass — but the table is COLUMN-MIXED: beta_to and spearman are the lambda=0
  recon reads while steerable is the lambda=1 shaped read (incumbent shaped beta_to at lambda=1
  is 0.2284); implementer: relabel the columns. d2_both's floor shortfall is 0.013413 exact (the
  logged 0.0135 is rounded-cell arithmetic).
- 2026-08-08: the d2_states rollout-feasibility caveat is RESOLVED — all three rerank dumps
  (incumbent, d2_contrast, d2_states) carry an identical 31744-row census (512 groups at every
  T, zero missing/non-finite; lm_prior bit-identical across arms), so cross-arm differences are
  the scorer alone. Carry-over caution: d2_states' recon scale differs (within-group var ratio
  k = 0.0091 vs the incumbent's 0.0131), so a scalar lambda is NOT comparable across scorers —
  match operating points on prior share.
- 2026-08-08 (joint repricing read, planner scratch on the D0 dump — implementer to reproduce as
  a logged table with the clause-table job): at T=0.7 the live lambda=1 is far below every
  scorer's ranking optimum — incumbent at lambda=8: spearman 0.5558 -> 0.6778, beta_to 0.2284 ->
  0.1112, sel_wer 0.1316 -> 0.1222, steerable 0.1949 -> 0.2034, prior share ~46 %; the optimum is
  arm-invariant at prior share ~0.45 (lambda 7.9 / 8.2 / 9.5 for incumbent / d2_contrast /
  d2_states); at matched operating points NO D2 candidate beats the incumbent on any rollout
  statistic; beta_to reaches zero only at lambda ~22-27 at prior share ~88 % (inadmissible).
- 2026-08-08 (build items to the implementer): a small CPU job reading the seven items.json plus
  the four held.json that prints the eligibility clause table and the paired cross-arm discount
  CIs with pinned seed/resamples (the winner rule is still computed nowhere); register
  items.json as an output; note in the log that the matched draw is frequency-weighted inside a
  concentrated pool (top-10 words = 73 % of draw mass) and that the cps-0.5 pool excludes
  single-letter words (23.9 % of the cps-1.5 control mass) — d2_states is state-matched but not
  lexicon-matched.
- 2026-08-09: approach 1's parenthetical "the only trainable-scorer run on record" is
  inaccurate — the 100 h recon-only and hinge-only arms were also jointAR (config default
  `freeze_ar=False`, aliases `grpo_100h_seed10h_jointAR_*`); the replay arm is the only one
  with per-epoch scorer forensics, not the only trainable one. Conclusions c1-c2 stand (they
  describe that run's mechanism), but the causal generalization "co-training causes the
  collapse" is UNPROVEN: no frozen-scorer control ever ran on the 100 h bed, the 10 h matched
  pair went the other way (frozen Goodharted 14.47/17.09, joint won 13.15/16.13), and the two
  100 h jointAR siblings lack the collapse signature. Temporal order (CE_true crossed the unit
  marginal after one sub-epoch while dev WER was still 18.79) supports scorer-first but is not
  attribution. D5 (the freeze_ar=True control, `PLAN_3E1.md`) is registered to settle it.
- 2026-08-09 (approaches 11-15 + c16-c22 full audit; five independent recomputes from raw
  artifacts): every logged table cell reproduces to the last digit. Approach 11: seed=42 /
  n_boot=10000 are hashed job inputs, both ladder-floor readings printed, winners
  d2_states(point)/d2_contrast(CI) recompute; all 20 selector cells recompute at the pinned
  seed; the 719 av.* / 82 psi.* import claim verified by key-level name+shape bijection
  against theta_0^G's AV SFT checkpoint. Approach 13: all 80 anatomy cells recompute from the
  sclite.dtl counts (convention C/ref, D/ref, I/ref confirmed); all ten report dirs pinned,
  finished 08-04/05, checkpoint lineage ep0 = the replay arm's own av_checkpoint_path, ep1-4 =
  its epochs 1-4. Approach 14: all 30 grid cells + unrounded self_pref/follow recompute; the
  gold column is the SAME five forward jobs as approach 1's CE_true (one instrument, not two
  agreeing); every cell scored an identical 428,064-unit stream. Approach 15: all 45 cells
  recompute; the five rollouts.jsonl carry a byte-identical 6400-row census (policy-pinned is
  literal); ep0 is the finished Jul-31 job reused by hash; eta = ratio of across-group means
  (not a mean of per-group ratios). Approach 12: dump/curation numbers exact incl. the 5.48%
  mask figure bit-exact (2664/48646); refit confirmed from-scratch d2_contrast (info differs
  from the audited d2_contrast job in hf_data_dir alone; mid-run ep10/30, downstream gate jobs
  not yet in existence).
- Prose defects, none flipping a conclusion's direction: c21's "at or above BOTH floors" is
  false for 20/24 cells (correction under c21); approach 14's "three utterances carrying
  accents that the corpus reader cannot decode" is wrong as written — nothing is dropped,
  accents are ASCII-folded in place, 12 fold events over 7 distinct utterances (ep3 alone has
  3); approach 11's table caption "arm-invariant rows" is false for psi_len_only — its (e) is
  psi_g_tc100's value and its (f) is d2_both's (every arm's (e) CI straddles zero so the
  verdict stands; implementer: relabel the row); approach 10's D2 paired-CI endpoints remain
  the old unpinned draws and differ from the pinned clause-table job in the last digit on
  several endpoints (verdict-invariant; the 08-08 finding, pinned values now on record).
- Reading qualifications: c20's 6.4x is dev-clean only (dev-other 5.07x, pooled 5.66x); the
  ep0 insertion baseline is 56.5% ten runaway repetition-loop utterances vs 2.1% at ep4 — the
  collapse trades rare loops for broad diffuse padding; the five named function words carry
  only ~18% of insertion mass, so monitors must read TOTAL insertion counts (as the D4'
  amendment requires). c19's "all twelve are worse than the repaired round-0 text" is an
  aggregate reading, literally true in ~56-72% of groups. c16's psi_len_only "no -- (e)" is
  the one selector verdict that depends on the unblessed CI-convention pin; the other four are
  convention-independent.
- Unverifiable (the measuring trial jobs were deleted): the ~9.5 h whole-bed estimate and the
  1.68x max_seqs-8 gain survive only as config-comment claims; the 11.5 h cap, the 4 -> 8
  max_seqs change, the no-resume property and the actual 5:17:30 runtime all verify and are
  consistent with both.
- 2026-08-10 (spot-check of the decision-critical cells of approaches 16-18 — NOT the full
  audit, which remains open): the fork-screen table reproduces exactly (reward argmax =
  sub-ep 3, d_cls +32.20 %/+38.47 % vetoes, FORK sub-ep 2); the suspect derivation is empty
  as logged (largest excess "and" 0.001345, 0.001 admits one word); the D4' admissibility
  table reproduces cell-for-cell ((c)/(f) betas incl. psi +0.2254/+0.2029 with CIs, (e)
  spearmans, n_groups 342/24/292/647); the joint arm's OOM is in the job log (step ~72-75,
  95 GiB card, sampled step times 12-21 s). c31 and the c28/c29 statistics are unaudited.
- 2026-08-10, new information from the D4' dump's own (a) block (in the job output, not yet
  in the log — implementer may want it in approach 18's table): on this bed at T=0.7
  within-group suspect-count contrast is nearly ABSENT — coverage 0.0037 (any suspect, vs
  the G-track's 0.092-0.233), mean within-group count std 0.0116, while the ranking prize is
  real (mean_wer 0.0562, oracle 0.0414, greedy 0.0541 over 28539 groups). The minimal-state
  exploit therefore sits in a near-total GRPO dead band at the fork's operating point: no
  in-loop reward term can steer it, which is the quantitative case for the offline refresh
  path and against adding reward terms on this bed.
- 2026-08-12: D6 clause table spot-verified from `PsiGateClauseTableJob.JdrWdaCm7UeG` —
  c35's cells reproduce (d_min=2 matched discount paired +0.0047 [-0.0009, +0.0101]
  p=0.096; lmsub 0.9572; both verdict rows NO WINNER as logged). Clause (i)'s picked-WER
  half is missing from approach 20's table — read from the rerank jobs: sel_wer d6_mindur
  0.05028 / combined 0.05015 / margin 0.05097 / r1_uncurated 0.05219 / psi0_gold 0.05228;
  d_min=2 PASSES that half. Caveat: the min-duration arms score 28531 of 28538 groups (7
  unscorable under the topology), so their random/oracle baselines differ slightly
  (0.05477/0.04116 vs 0.05613/0.04133) — cross-arm sel_wer is not perfectly paired,
  ordering unaffected.
- 2026-08-12: c32's two open bullets are now stale — the reads finished: sub-ep 3 is
  41.8 / 50.9 (`ScliteJob.yVyM2WLvkXxG` / `.4mnMvy9mUVI7`, epoch 3 via
  `ExtractAvSubmodelJob.a1d9LlyUDSED`), and the forensics give gold-pair ce_loo
  2.6343 / 2.7928 / 2.9771 at sub-eps 1/2/3 (`PsiHeldNllJob.LTg9xnjtl8Zs` / `.SFOP6DaI3Zpv`
  / `.vJnzFU0eRSyl`) against own-decode ce_loo 2.6270 / 2.4726 / 2.2994
  (`.8DHdEHY7HZ2b` / `.2WmXVQYlCjnF` / `.uEc3jigALnmE`) — the D5 gate verdict these decide
  is recorded in `PLAN_3E1.md`.
- 2026-08-17: c37 VERIFIED from `PsiGateClauseTableJob.qYRE7JWyUcJQ` / `.H9QbX4VgXAwf`
  (clauses.txt, paired n=1442): every quoted number reproduces exactly — ce_loo
  2.3774 / 2.7168 / 2.7198, filler_ins +0.0849 [0.0635, 0.1058] and lmins +0.0488
  [0.0362, 0.0616] vs `r1`, filler_sub -0.0136 [-0.0194, -0.0077], NO WINNER under both
  readings against both comparators. New information from the vs-`psi_g_tc100` table:
  `r1_mindur` is CI-worse on the del ladder too (-0.0097 [-0.0158, -0.0039]) and on
  filler_sub -0.0146, while its matched insertion discount is CI-LOWER at k=4 (-0.0182
  [-0.0276, -0.0085], p=0.000; k=1 n.s.) — the topology's insertion-pricing gain grows
  with k while eligibility fails on the substitution/deletion side. Plan verdict recorded
  in `PLAN_3E1.md` D6 Status same day.
- 2026-08-17: HOM-0a rerun (`HomophoneClassStatsJob.our76yheSD0c`) verified — share
  7.68 % against the untouched 5 % floor, PASS; planner eyeball of the full 142-class
  list: no strikes (weakest admitted member "ad" at 8,084 LM occurrences vs the 8,033
  floor; no typos, no single-char, "to" classless as pinned). New information from the
  class list: the repair-channel reading is confirmed and quantified — corpus-zero
  members carrying dominant LM mass (by, sea, right, side, air, fair, they're) total
  ~0.5-0.6 % of corpus tokens, and they're=0 is a decoder commitment, not an alphabet
  artifact (apostrophe forms it's/i'll/there's all attested). HOM-0b/0c reading
  amended pre-run in `PLAN_3E1.md` (bars untouched): swaps reported split repair-type
  vs diversity-type, top-8 per-class medians beside the aggregate.
- 2026-08-17 (HOM augmentation machinery): HomophoneAugmentJob.k2OwZiTcKpEG verified by
  an independent token-level diff of the augmented corpus against the source — every
  reported number reproduces exactly (38923 rewrites; repair 5627 / diversity 33296;
  top-8 classes; day/dey 557), zero out-of-class or case violations, dev splits
  byte-identical, and the realized draw is within ~1.6 sd of the analytic expectation
  from the ratified classes.json (E[rewrites] 38706, E[repair] 5560) — the sampler is
  the draw 0a's arithmetic assumed. Class list consumed from the ratified artifact
  (stats-job hash unchanged); the SFT stays unwired, gated on 0b. New fact for 0b:
  in/inn alone carries 19% of rewrites — 0b/0c reporting amended pre-run in
  PLAN_3E1.md (in/inn's median named explicitly; aggregate-without-in/inn beside the
  gated aggregate; day/dey added to the named watch); admission bars untouched.
- 2026-08-18 (D6-PERIODIC-WARM, approach 24, code verification): every submitted claim
  verified at source and artifact. Commit 5773910 on haotian_modality_matching_jupiter
  carries the three files; `PsiAlignTrainJob.run()` loads the warm state dict
  (psi_align_jobs.py:615) BEFORE the [UNK] unigram pin (:621), the guards assert the
  inventory plus six topology keys against the checkpoint's own cfg dict, and any
  mismatch the guards miss fails loudly in strict `load_state_dict`. Hash neutrality
  proven from the artifact itself: the new round-2 warm refit
  (`PsiAlignTrainJob.2TDm8VwIZzjv`, alias sae/3e1/d6periodic_warm/r2/refit) lists
  `PsiAlignTrainJob.wlruSpBK1EDP/output/model.pt` as its `init_model` INPUT — the graph
  built under the NEW code resolved the PRE-change hash; wlruSpBK1EDP and JWV3InILYF5v
  finished on disk, no respawned refit dirs, shared leg 1
  (`ReturnnTrainingJob.5FqdnhWTOf1f`) finished. Planner re-ran the warm-start test:
  passes, warm held NLL 1.1869 vs cold 2.2609 after one epoch, source best 1.4341,
  output unigram re-pinned on the fit corpus (matches all submitted numbers).
- 2026-08-18 (same submission, rulings — normative text in PLAN_3E1.md D6-PERIODIC
  Status): warm source = the INCUMBENT's model, ratified; four-clause gate KEPT with a
  registered non-read — accept counts are never compared across the warm arm and the
  sibling; binding read = plain WER trajectory at matched parent sub-epochs vs the
  sibling, legs 2 on, with the sibling's round-1 replication spread as the minimum
  meaningful difference; leg-1 hash sharing and the skipped parity re-run ratified; the
  matched-point anatomy job NOT funded unless the trajectory separates beyond the floor.
- 2026-08-18 (bookkeeping): the periodic-vs-one-shot rounds 2-5 differences and the
  "refresh buys nothing over the one-shot" reading are not yet a logged conclusion, so
  the implementer's tightened reading needs no correction marker here; when that
  conclusion is written it must carry the round-1 replication floor (submitted
  0.29 dev-clean / 0.24 dev-other, to be verified with the batch report) beside the
  per-round differences.
- 2026-08-18 (second submission same day: HOM-0b/0c code, the 0c read, the periodic
  gate trace, the dump-column finding). Commits 4537617 and cb6a9a8 verified on
  haotian_modality_matching_jupiter; all 19 homophone-probe tests re-run by the
  planner, pass. 0c artifact verified: HomophoneCoverageJob.F76iJ8j0AQi1 (alias
  sae/3e1/hom/hom_0c) reads exactly the named dump and pool, and every submitted
  number reproduces (26,584 bearing; 6,228 = 23.43 % covered; 217 = 0.82 % including
  a pool-zero member). The 0c artifact substitution is RATIFIED as a frame repair --
  the registered dump has DUMP_GROUP_SIZE=1 (config verified) and cannot express
  within-group coverage by construction; plan definition amended by replacement and
  the scoping reading recorded in PLAN_3E1.md (diversity already reachable by
  sampling; repair in the dead band, reachable only by an SFT-side support change).
- 2026-08-18 (finding 1 verified from the four verdict jsons): r2 and r3 fail (iii')
  at CI, the dry rule enters force at r4 (dry_started_here true), r4 passes all four
  clauses under CI and is overridden with accepted=False, r5 fails (iii') and (iv');
  conclusions 41 and 42 check against the artifacts, both accurate as written. The
  periodic-arm verdict, the warm-read amendment, and a pre-registered warm-source
  fork (registered while the round-2 warm verdict does not exist; its refit was at
  epoch 26/30 at check time) are in PLAN_3E1.md D6-PERIODIC Status, 2026-08-18.
- 2026-08-18 (finding 2, handed audit closed): SAE_3A approach 9's lam_lm sweep rows
  are NOT invalidated -- the parts dump (config_sae_2s_rewardrank_parts_v1 sets no
  reward_kwargs at all) and the arms live at sweep time both ran the legacy per-token
  norm, so the sweep was internally consistent at its own operating point, and the
  standing shaped setting (lam_lm 1.0, per-unit) was derived, not swept. The live
  mismatch is forward-looking only: dumps still emit per-token lm_prior columns while
  shaped arms train per-unit-frame (verified at rewardrank_avunits:130, reward.py:67,
  psi_loop:113); the 0b numerator-anchored comparison is the correct fix. Implementer
  follow-ups, their lane: audit any post-per-unit-switch consumer of dump lm_prior
  columns beyond 0b; thread the arm's reward_kwargs into future dumps or document the
  column as per-token. Same-day bullet in SAE_3A.md; trap saved to planner memory.
- 2026-08-18 (third round: 0b results, consumer audit, user warm ruling). 0b verified
  from HomophoneSensitivityJob.xB5RvcgLVgtD/output/hom_0b.json: every submitted number
  reproduces (aggregate 0.0134/0.0106 ratio 1.2636 PASS; diversity 1.274 n=22,584;
  repair 0.816 n=501; all six per-class ratios; sign structure -0.0135/-0.0073), plus
  one banked fact beyond the message: the SIGNED medians are negative for both terms in
  both directions -- even lm_prior penalizes repair swaps on median -- which hardens the
  mechanism-reversal reading. Conclusions 43-45 check against the artifacts as written;
  gate verdict (aggregate PASS, arm admitted, SFT licensed), the ordered
  std_within_group read, and the user surfacing are in PLAN_3E1.md HOM Status. Commits
  9fa9ecc (terminator fix; diagnosis matches the artifact distribution 1724/2000 at -1,
  2000/2000 at 0) and 1216064 (dump-norm documentation at both production sites)
  verified.
- 2026-08-18 (finding-2 consumer audit spot-verified at all five cited lines):
  curate.py:43 converts to per-unit; scorer_diag.py:291 names the raw view
  lm_prior_tokens, :363 and :853 convert with the n_units-constant rationale in
  comments; PsiAlignRerankJob shaped_weight lam_lm 0.075 is a per-token weight on the
  per-token column -- internally consistent throughout, matching the audit conclusion.
- 2026-08-18 (user ruling on the warm source, relayed): reading B -- the gate-controlled
  incumbent -- stands under every verdict; my pre-registered rejection fork is REPLACED
  in PLAN_3E1.md, and a dry-contingency decision rule (no further legs at two
  consecutive rejections; user's resource call; planner recommends stop) is registered
  in its place. Label-free-clause correction recorded there too: (ii) reads gold held
  text; code fix directed after the pending round-2 verdict job lands.
- 2026-08-18 (gate-removal teardown verified): commit 3257edc on the branch; kept
  hashes finished on disk (ReturnnTrainingJob.5FqdnhWTOf1f, PsiAlignTrainJob.JWV3InILYF5v,
  .2TDm8VwIZzjv); the approach-22 table matches the planner's PRE-deletion first-hand
  artifact reads clause for clause and verdict for verdict, and its leg-1 row is
  consistent with the earlier 0.29/0.24 replication submission; the Catalog's relaunch
  note and new id rows are in place. All superseded rulings annotated in PLAN_3E1.md.
- 2026-08-18 (two dirs survived the teardown; one holds a bankable verdict):
  PsiRefreshAcceptJob.lWmT0OpDXfSp and .uXG53BObiW55 were still on disk at planner
  check, contrary to the teardown report. uXG53BObiW55 is FINISHED: the warm round-2
  candidate was ALSO REJECTED under the binding CI reading -- (i) pass, (ii) pass,
  (iii) point fail / CI PASS, (iv) point fail / CI FAIL -- i.e. a warm-started
  continuation failed the clause table too, on the corruption ladder rather than the
  insertion price. Worth one banked line in approach 24 before these dirs are
  re-deleted (the deletion itself stays user-directed).
- 2026-08-18 (correction to the approach-22 closing sentence): "the spread across them
  is this bed's run-to-run noise and nothing else" overstates -- the five legs are
  successive segments of one trajectory at different schedule positions (leg k trains
  from leg k-1's checkpoint on a decaying cosine), so the across-leg range conflates
  schedule evolution with noise. The run-to-run measure is the five PAIRED matched-
  point deltas against the one-shot arm at the same global sub-epoch (planner-computed
  from the two logged tables): dev-clean 0.29/0.03/0.32/0.36/0.19, dev-other
  0.24/0.30/0.32/1.30/0.11; floor and reading rule registered in PLAN_3E1.md. The
  table's numbers are untouched; the sentence is the implementer's to amend.
- 2026-08-18 (answers): refresh_gate.py constant/docstring fix -- yes, at leisure, the
  module is still the D4-prime machinery's. HOM SFT hold ENDORSED; recorded with the
  user surfacing in PLAN_3E1.md HOM Status.
- 2026-08-18 (fourth round: HOM SFT launch verified; the user's greenlight was given in
  the implementer's session and is recorded as relayed in PLAN_3E1.md HOM Status).
  Commit b44952e verified: theta_0^G_hom is theta0g_av_sft called with hf_data_dir as
  the single moving argument (plain-function kwarg, no job-ctor change; theta_0^G hash
  2fb02hGUdHNj unmoved — finished marker and every checkpoint mtime untouched since
  2026-08-04). The running job (ReturnnTrainingJob.EabxlDlT0oji, alias
  ...seed10h_layer15_gtrack_pseudo_tc100_hom/training, SLURM 1405194) trains on
  TransformAndMapHuggingFaceDatasetJob.157IDJgBOv9H (alias sae/2s/data/
  hom_worddecode_tc100_q3), which attaches HomophoneAugmentJob.k2OwZiTcKpEG's
  word_hyps.json to the base train-clean-100 ogg dataset — one derivation step
  downstream of the augment job (the submission's phrasing compressed this; substance
  holds). Byte-verified independently with a planner-seeded sample and then the FULL
  corpus: zero text mismatches over all 28,539 train utterances vs the augmented json,
  no changed utterance serves source text, dev splits byte-identical; the dataset
  builder hard-asserts bidirectional uid coverage (data.py:88-125). Checkpoint pinned
  at the last epoch, dev recogs select nothing — quarantine holds. The implementer's
  disclosed sis_env/create_files trap (markers renamed, resubmitted) left no live
  error marker and moved no hash.
- 2026-08-18 (fifth round, theta_0^G_hom read): approach 27's numbers all reproduce from the
  concrete job dirs -- 13.89/18.34 vs 16.67/21.45 at ep10 (ScliteJob.4xgsEBkQtPsg/.KKjjg7A3vT52
  and the dev-clean pair), +3.11 dev-other, both arms scored on the identical dev-other stm
  (2,864 utts, 50,948 ref words), runtimes 2:12:25 vs 2:11:17, and a config diff of the two
  training jobs that moves FOUR lines only (the three dataset dirs plus the model path). No
  dev-WER selection can have entered: learning_rate_control constant, keep_best_n ranks
  pseudo-label dev CE, ep10 = num_epochs, and ep10 is also each arm's best scored epoch, so
  last-epoch and best-epoch rules give the same gap. Three reading notes: ep2 was also scored
  for both arms (degenerate, above 100 % WER) and is silently absent from the logged curve;
  the hom arm's ep10 dev-other EQUALS its ep8 (21.45), so the +3.11 is carried by the
  baseline's own ep8->ep10 gain and is not an epoch-10 effect; and a parallel non-sclite
  scorer exists on this arm (JoinRobustMetricsJob.6il1r3BMTMEj, with normalized WER_clean /
  WER_cap columns) whose numbers must never be quoted -- plain sclite only, standing rule.
- 2026-08-18 (same round, class-internal read INDEPENDENTLY RECOMPUTED from the two arms'
  sclite.pra alignments, parser validated against sclite's own per-utterance (#C #S #D #I)
  headers with zero mismatches over 2,864 utterances): every claimed figure is exact --
  +1,587 errors, 1,534 class-internal (96.7 %), 25.18 % vs 5.24 % of substitutions, all seven
  top confusions to the unit, 3.586 %/0.575 %. Four framing corrections, none flipping
  conclusion 46's direction: (i) the "within-class substitution rate 3.59 %/0.58 %" uses ALL
  dev-other reference tokens as denominator -- over the class-bearing tokens the name implies
  (4,461) the same counts read 40.96 % vs 6.57 %, an 11x difference a reader will mis-assume;
  (ii) "close to the 4.04 % the augmentation rewrote" compares train pseudo-label tokens with
  dev-other reference tokens -- the like-for-like expectation under full reproduction of the
  uniform draw is 4.58 % of reference tokens (2,331 substitutions), so the realized 3.59 % is
  78 % of it and the SFT UNDER-reproduces the draw by about a fifth; (iii) 96.7 % is the share
  of NET extra errors -- against extra substitutions it is 92.2 %, with +130 non-class
  substitutions added and offset by 20 fewer deletions and 57 fewer insertions; (iv) the plain
  arm's 5.2 % baseline is 65 % a single pair (by->buy 190 of 293), so it is not a broad
  background rate. "Outside the classes within noise" is quantitatively upheld: +53 errors,
  paired bootstrap CI [-70, +173], straddling zero against a total-error CI of [+1,438, +1,735].
- 2026-08-18 (same round -- THE ARM'S MECHANISM QUESTION, answered on existing artifacts;
  planner join, NOT yet a job, so no number below may be cited anywhere until it is emitted
  from one, per the standing scratchpad rule). HOM-0b is reference-BLIND by construction: its
  verdict is `lm_dominates` = median|delta lm_prior| > median|delta recon|
  (homophone_probe.py:411-417, :454), abs() on both sides, so a reward whose prior term moved
  more while pointing at the WRONG spelling returns the identical PASS; its signed medians are
  signed against the policy's own sampled spelling, not against truth. The gold text sits
  INSIDE the very dump 0b consumed (the 28,539 `kind:"true"` rows of
  ReturnnForwardJobV2.66pIzBzffnK2) and is discarded by the kind filter at :260, so the
  correctness read is a pure CPU join, zero GPU. Joined (crude bag-of-words reference
  membership, not position-aligned; 0 missing references over the 23,085 swaps): in the one
  informative cell -- 1,550 swaps where the sampled spelling is ABSENT from the reference and
  the swap target is PRESENT, i.e. the swap repairs it -- the LM prior points at the reference
  in 89.5 % of cases (median +0.0127) while recon points at it in 18.2 % (median -0.0120), and
  the ARM'S OWN composed reward at its registered lam_lm 1.0 points at the reference in 53.2 %:
  chance. Repair-type inside that cell is n=27 (25.9 %), i.e. untestable at this sample size.
  Standing caveats for whoever builds the job: gold read, so it reports and can never select;
  the base texts are theta_0^G's samples, not theta_0^G_hom's; 8.8 % of swaps fall in a
  neither-spelling-in-reference cell the test cannot classify.
- 2026-08-18 (ordered std_within_group read -- status and one trap). Still NOT wired: no job,
  config entry or alias exists. Two existing routes, one of them a trap. RolloutMechanismJob
  (scorer_diag.py:412-420) already emits std_within_group but over a HARDCODED
  ("recon", "shaped") tuple at :540-541, so the lm term ALONE -- the half the order names -- is
  missing even though `_lm_prior_units` is already computed at :364; a banked full-bed instance
  (RolloutMechanismJob.UJ0DfPXTH8Cq) reads recon 0.0218 / shaped 0.0239 over 28,539 groups at
  0.041 h CPU-only. RewardShapeSweepJob would answer it as a mini_task and carries an
  audio-free (w_recon 0, lam_lm 1) cell, BUT its compose() reads the dump's RAW lm_prior column,
  which is per generated TEXT TOKEN while the shaped arms train per UNIT FRAME -- and n_tokens
  varies within a group while n_units does not, so this is NOT a within-group constant rescale
  and the sweep route needs the scorer_diag conversion (:361-364) or it answers in the wrong
  units. This is the standing dump-column trap, now with a second consumer.
- 2026-08-18 (sixth round: HOM loop arm launched on the user's override; commit 282eeb3).
  Hash-neutrality for the LIVE D6-PERIODIC/GAN arm CONFIRMED, and for the right reason:
  neither file the commit touches contains a `class` statement, so the standing
  "defaulted ctor kwarg moves every instance's hash" trap cannot fire -- `build` at
  config_sae_3e1_d6periodic_gan_v1.py:261 and the four other functions that gained `tag` are
  plain module-level builders, and all 19 occurrences of `tag` terminate in add_alias,
  tk.register_output, or an f-string consumed only by those. Behavioural confirmation: the
  post-commit whole-graph alias pass at 17:32:41 rewrote 48 symlinks, every one of them the
  new arm's; all 16 live-arm symlinks keep their pre-commit 2026-08-17 12:17:55 mtime and
  resolve to the same hashes, and the untagged round-1 unit re-derives identically under
  post-commit code (transitive, via hom_0b_swaps' own input list). TWO SCOPE CORRECTIONS:
  (i) only SIX of the sixteen named artifacts exist on disk (legs kr1foUV6lecx / AuzMGgyskdJT
  / KD73Hc4eGDfW, refits dsMKgPHQApyR / 7jHYVGToyWPR / M2Z0M9UpKW98) -- for the other ten
  "unmoved" means the graph still predicts the id, not that finished compute was preserved;
  (ii) the neutrality has NOT been road-tested, because the live arm's manager started 3h41m
  before the commit and is driving its pre-commit in-memory graph -- the first real test is
  the next restart of config/sae_3e1_d6periodic_gan.py, which is a watch item for whoever
  restarts it. Also for the leg-1 A/B: hom leg 1 will carry a different psi_checkpoint
  (ACP3LqKDUSQ0 vs dsMKgPHQApyR) as well as a different init, since its refit is downstream
  of its own decodes -- a consequence of the one swapped argument, not a second design
  choice, but it cannot be stated as a single differing input.
- 2026-08-18 (ops, found during the same round and handed to the implementer): the HOM arm's
  manager (log/sae_3e1_hom.manager.pid 822401) is DEAD while all four other live-arm managers
  run. Not a crash -- manager.log is 0 bytes, which at --log_level 30 is a clean run, and the
  graph was built correctly first (171 -> 292 jobs, aliases written) -- the signature is a
  manager killed with its parent. Consequence: GreedyPoolJob.g37LQXi9ABH3,
  PsiAlignTrainJob.ACP3LqKDUSQ0 and ReturnnTrainingJob.JocWKAmYroFJ have no job dir, so when
  the running round-1 dump (ReturnnForwardJobV2.95rhVVVmPWlo, slurm 1406706_1) lands nothing
  submits them and the arm stalls with no error marker. Recorded because a silently stalled
  arm reads exactly like a slow one.
- 2026-08-18 (CORRECTION to the bullet immediately above, same day, implementer-supplied and
  accepted): the forensics were right and the CAUSE AND CONSEQUENCE WERE WRONG. The manager
  did not die -- the implementer killed it deliberately at 17:38, four minutes after starting
  it, to stop an eight-leg spend from dispatching while the user's funding decision was
  reopened by the composed-reward finding. A deliberate stop and a parent-kill produce a
  BYTE-IDENTICAL signature (clean exit, 0-byte log at --log_level 30, graph already built),
  so that signature can never establish cause -- ask whose hold it is. My "deadline" was also
  wrong: job dirs are content-addressed, so a manager started after the running dump lands
  picks up the refit and loses no compute; the only cost of waiting is delay. What I framed
  as an outage to repair was a correctly-held spend, and restarting it would have started the
  two days of GPU the user was being asked to authorize. The dump was deliberately left
  running because that artifact is wanted under either decision.
- 2026-08-18 (funding state made explicit, ratified): `config_sae_3e1_hom_v1.LOOP_FUNDED`
  now carries the decision -- at False the graph still builds the admission reads, the init
  and the direction read, and leg 1 cannot be submitted by ANY manager. RATIFIED as the
  standing pattern for a held spend: funding state belongs in the config where it is readable
  and reviewable, never implied by which manager process happens to be alive.
- 2026-08-18 (ordered build (a) DELIVERED and independently reproducing; HomophoneDirectionJob,
  commit 67da952, hash Uo4UAJp5Ue42): the planner scratch read is reproduced EXACTLY at the
  bag-of-words join (n=1550, lm_prior 0.895, recon 0.182, composed 0.532) AND the
  position-aligned join I asked for as a stretch is feasible -- the swap rows already record
  `position`, so a minimal-edit alignment yields the reference word at the swapped position --
  and it agrees (n=1421, lm_prior 0.906, recon 0.179, composed 0.529). The finding therefore
  stands on the better join, and its numbers are now job-emitted rather than scratch, so they
  may enter a conclusion once the artifact is read off disk.
- 2026-08-18 (weighting defect QUANTIFIED against the artifacts; HomophoneDirectionJob
  .Uo4UAJp5Ue42 finished 18:01 and its per-class table was re-derived independently through the
  job's own alignment logic with the top-8 truncation removed -- 81 classes carry at least one
  toward-reference swap). The mismatch is larger than I stated: the measurement's per-class
  share vector overlaps the PLAIN arm's dev-other class-internal profile at total variation
  0.880 and the HOM arm's at 0.197 -- near-orthogonal to the distribution it is being used to
  predict. buy/by/bye is 67.63 % of the measurement, 66.55 % of the plain arm's class-internal
  substitutions (195/293) and 8.65 % of the hom arm's (158/1827). COVERAGE, the decisive
  column: 78.8 % of the hom arm's damage sits outside the eight classes the job reports, and
  31.1 % sits in classes with ZERO toward-reference swaps in the measurement -- be/bee (155)
  and knot/not (155) among them. The single largest hom class, in/inn at 18.0 % of the damage,
  is measured on FOUR swaps. Damage-weighted composed rates: 0.555 over the reported eight
  (21.2 % of the damage mass), 0.637 over all 81 (68.9 % mass) -- the latter driven by in/inn
  reading 1.000 off those four observations. So the aggregate 0.529 is, to a good
  approximation, one confusion pair's number.
- 2026-08-18 (CORRECTION to my own claim in the bullet above and in PLAN_3E1: "no number from
  the plain-policy read may be quoted as a prediction of this arm's recovery" was too strong,
  and I had not considered the slice that refutes it). The same job's AWAY-from-reference
  cell (n=19,328) covers 99.7 % of the hom damage mass -- because the plain policy spells
  those classes correctly, every class is represented in the corrupting direction -- and
  sign-reversed it reads composed 0.900. It answers a different conditional (does the reward
  resist corrupting a correct spelling, rather than prefer repairing a wrong one) over a
  different utterance population, so it is not interchangeable with the toward cell; but it is
  arguably the LESS biased estimator for this arm, because the hom arm's errors are
  augmentation-induced and therefore land on a near-random subset of class-bearing utterances,
  whereas the toward cell is selected for utterances where the plain policy itself was
  confused. NET STATE: the plain dump BRACKETS the hom arm's per-swap edge between about 0.51
  and 0.90, biased in known and opposite directions at the two ends. That bracket spans chance
  to strong, so it is uninformative for the funding question -- which is a stronger reason to
  run the read on the hom arm's own dump than the one I gave, not a weaker one.
- 2026-08-20 (periodic-family on-disk audit): read every available WER from the concrete sclite
  work artifacts and independently reconstructed each completed dev-other S/D/I count from
  `sclite.pra`, checking it against the reported WER. Completed prefixes are fresh D-track 5/8,
  warm D-track 5/8, plain GAN 6/8 and GAN+HOM 3/8; the next leg of each exists in Slurm but is
  pending because nodes are reserved for maintenance. All four managers are live and none of the
  four next jobs has an error marker, so no arm has an endpoint yet. The fresh periodic prefix
  avoids D5(b)-b's catastrophic continuous-joint insertion collapse but later loses to both the
  one-shot scorer and matched frozen control through insertions; warm inheritance compounds that
  failure. Plain GAN's one improvement is transient and later loss is substitution-led. GAN+HOM
  removes its induced class-internal errors in one leg and catches plain GAN by leg 3. The D5(b)-b
  comparison is only a timescale control because scorer topology, partitioning, batching and
  optimizer continuity also differ; there is no continuously trained scorer with the GAN or HOM
  initialization.
- 2026-08-20 (baseline presentation correction, user-directed): the periodic tables now expose
  exactly two primary anchors per initialization family before the live trajectories. For the 10 h
  adapted-donor init these are theta_0' AV SFT 11.43/15.54 and the best prior frozen-scorer loop
  4.68/8.64; for GAN init they are theta_0^G AV SFT 13.89/18.34 and the best prior frozen-scorer
  loop 12.68/17.57. GAN+HOM has no same-init frozen loop, so its 16.67/21.45 AV-SFT checkpoint is
  reported as its own anchor and the plain-GAN frozen result is labeled cross-init context only.
  Clarification after source-level comparison: the plain-GAN frozen result is itself a reference,
  not a scorer-schedule-only control, because scorer topology and corpus plus policy-optimizer
  continuity differ. The missing controlled arm freezes periodic round 1's own d_min=2 scorer while
  retaining periodic's segmented-leg graph.
- 2026-08-20 (why the frozen G-track reference is d_min=1, source/history audit): git history and
  the original D2 builder establish chronology rather than selection. D2 landed 2026-08-07 with
  topology intentionally identical to psi_g_tc100 so d2_contrast changed one objective term;
  PsiAlignTrainJob had no min_dur interface in that revision. D6 introduced the structural
  minimum-duration repair on 2026-08-11, after D2 was complete, and D3 inherited its frozen winner.
  Therefore d_min=1 has no empirical superiority claim over d_min=2 here, and topology is the main
  scientific confound in the frozen-versus-periodic WER comparison; corpus and Adam continuity
  remain additional confounds.
- 2026-08-20 (matched frozen control): merely rebuilding D2/D3 with d_min=2 would leave the
  scorer-corpus and policy-optimizer-continuity confounds. Source, resolved-config and graph checks
  pass for the isolated schedule-only control: leg 1 reuses
  `ReturnnTrainingJob.kr1foUV6lecx`, all eight legs read periodic round 1's exact d_min=2 scorer
  `PsiAlignTrainJob.dsMKgPHQApyR`, and no dump, pool or refit exists after round 1. Thus scorer
  recency is the only intended difference from D6-PERIODIC/GAN. Leg 2 was verified running at
  15:37 CEST; no endpoint exists yet. Normative gate and interpretation are in `PLAN_3E1.md`
  D6-PERIODIC/GAN-FROZEN.
- 2026-08-20 (D7.3 gate correction): the former absolute 13.89/17.84 clause was unsupported for a
  one-leg causal read. Conditional on required scorer parity, the exact matched control is
  `ReturnnTrainingJob.kr1foUV6lecx` at 14.45/19.69; the GAN initialization (13.89/18.34) and prior
  frozen-loop result (12.68/17.57) are report-only utility anchors. The prospective scientific gate
  is improvement over that exact matched control on both dev splits; any durability or absolute
  utility decision remains a separately preregistered next stage, with gold sealed until then.
- 2026-08-21 (D7.0a independently verified; D7-v2 frozen): PASS. The reconstructed Sisyphus graph contains exactly
  one finished label-free mini-task and no live scorer, assignment, training, reference or WER graph
  dependency. Code commit `a0a22b4` predates execution and contains only the tracked canonical
  config, census implementation and focused tests; the artifact source hash matches it and all 9
  tests pass. Independently regenerating both graphs from the four pinned inputs without importing
  D7 code matches every emitted edge and all input, population,
  tuple and file hashes: 4,911 external edges (`7855557c...d2f3`) and 632,913 intended-scorer edges
  (`3a6038ab...4376`). Representative boundary, just-outside-duration, different-speaker, self-edge
  and directed-asymmetry examples all behave as specified. The artifact contains no downstream
  filter, band, capacity, assignment, scorer or gold field.

  The scientific verdict is narrower than Conclusion 56 originally claimed. The external band was
  unregistered: 276/1,500 is the optimistic all-eligible raw K=4 support, while zero rows meet the
  conservative eight-donors-per-stratum diagnostic. Exact second-quartile support was undefined
  before the amendment, so D7.0a establishes that full-`E_all` K=4 cannot be measured rather than an
  eight-per-stratum theorem.

  A planner-side maximum-matching replay of the immutable external raw edge table gives 1,267,
  1,328 and 1,331 admitted edgeful sources at donor capacities one, two and three; a deterministic
  cap-three raw matching contains 669 same-chapter and 662 different-chapter edges. Capacity three
  is therefore the smallest tested load cap that preserves every raw edgeful source. Before any D7
  scorer read, D7-v2 freezes: the original K=4 (2+2), row-local Q2, ten-table regular construction
  for training with an executable ordinal rank/tie law; and a separate external K=1, M=1, no-band,
  max-cardinality/minimum-nuisance cap-three matching. External admission requires at least 435/725
  dev-clean and 465/775 dev-other sources plus 32/40 and 27/33 source speakers, retains fixed
  725/1500 and 775/1500 split weights, and never shrinks the all-1,500-row Acceptance gate. This
  prospectively authorized the D7.0b read whose closed verdict follows.
- 2026-08-21 (D7-v2 / D7.0b structural verdict, independently verified): FAIL. The label-free
  feature job completed and its manifest reproduces all frozen role sizes, hashes and checksums:
  28,538 feasible scorer rows, 569,785 hard training edges, 136,966 Q2 edges and 17,748 rows with at
  least two raw outgoing donors in both chapter strata. The assignment code encodes, for every
  admitted vertex and each chapter stratum, both incoming and outgoing edge counts exactly equal to
  `2*a_i`; its zero-gap MILP returned 56 admitted rows from two speakers before the registered
  6,778-row/201-speaker check raised.

  Independently streaming the emitted Q2 table and repeatedly removing every vertex with fewer than
  two incoming or two outgoing edges in either stratum leaves 120 rows from four speakers. Every
  feasible common 2-in/2-out solution must lie inside that necessary core, so even a different exact
  optimizer is bounded far below the floor. This is the intended fail-closed scientific gate, not a
  scheduler, timeout or convergence failure. The assignment stopped before external matching; no
  loss-preflight directory, scorer training, policy training, reference-text or WER consumer exists.
  The registered K=4/Q2/common-regular operating point is structurally closed, while the reverse
  matching loss itself remains unmeasured. Per the prospective rule, no solver retry, floor
  relaxation or third graph amendment is authorized; that offline-graph branch remains closed.
- 2026-08-21 (post-D7-v2 user correction; active D7 registration): the user rejects the offline
  graph as the
  wrong abstraction and directs ordinary online random negatives on the full loop population. Local
  interface verification supports that separation: the existing enc50 960 h chain already binds
  exactly 281,241 train-clean-100/train-clean-360/train-other-500 utterances and their packed K=500
  raw-50 Hz units, whereas every D6 periodic scorer refresh decodes/refits only the 28,539-row
  train-clean-100 pool. Corrected D7 therefore generates one scorer-independent theta_0^G greedy
  pseudo-text
  per 960 h utterance and samples K=1 donors dynamically from role-local same-speaker pools with
  reciprocal duration ratio 0.8--1.25, using closest-duration fallback only when that window is
  empty. It does not depend on §3d.A's currently blocked packed CTC decoder. The executed offline
  D7-v2 result remains closed as conclusion 57, while its active specification is superseded.
  D7.3 policy compute remains held.
- 2026-08-21 (corrected D7 implementation verified; approach 32 / Catalog / State audited): PASS —
  build faithful to the registered specification; no experimental number exists yet. Confirmed
  against code, the live manager graph, on-disk artifacts and the scheduler: all four Catalog job
  ids match the manager's own graph (aliases the manager wrote at 16:37; work dirs legitimately
  absent while the decode chain runs); the ten decode shards are GPU `ReturnnForwardJobV2`s on the
  frozen theta_0^G epoch-10 checkpoint, with `d7_greedy` a thin wrapper that calls the pre-existing
  argmax stepping (`forward_step.py` `_greedy_argmax_decode`, the same function behind the D6
  greedy dumps; it never reads the dataset's text stream) and the same tokenizer/lowercase/
  ascii-fold conventions; the pseudo-text transform replaces the train text with two-sided
  coverage asserts and a zero-empty assert; the bound unit store is the frozen raw-50 Hz K=500
  `PackUnitsJob.I0uzRMfUrKWC`; the seed-42 5% holdout reproduces the D7-v2/PsiAlign row-order
  convention exactly (verbatim toy re-run, identical held sets); every training constant
  (contrastive weight 1.0 / 1 negative, batch cells 24M, max batch 256, lr/decay/warmup, default
  architecture, `d_min=2`, CUDA backend, bpe512 codes and lexicon artifacts) is byte-identical to
  the round-1 refit `PsiAlignTrainJob.dsMKgPHQApyR` job record; the loss algebra implements the
  registered per-frame softplus donor-minus-own term with the stateless keyed draw (three CPU unit
  tests pass); D7.1 is hard-gated on the D7.0 preflight PASS artifact plus index-hash binding;
  fixed-final only, D7.2/D7.3 absent from the graph; the fixed-final checkpoint dict satisfies
  `PsiScorerParityJob.from_checkpoint`; no funded GPU job was cancelled or displaced at launch.
  Caveats a reader of D7.1 numbers must know (rewritten 2026-08-21 after the fix verification
  below; the resolved resume-RNG, donor-infeasibility and parity-gap caveats are absorbed):
  (i) prior weight is 0 from step 0, a forced deviation from the refit's 4-epoch prior anneal,
  entailed by carrying `L_U->z` across a single pass (definition pinned in `PLAN_3E1.md`);
  (ii) the ~0.035% max-generation-length truncation tail of the argmax decoder is the decoder's
  established operating point, not a D7 deviation — and the equivalence check ran once the shards
  finished: the D7 merge agrees with `ReturnnForwardJobV2.66pIzBzffnK2` on the 28,539 shared
  tc100 utterances at 0.459% word distance, argmax ties under different batching, no length or
  content bias (approach 32).
  Fix verification 2026-08-21 (speech-llm commit 1d10945, verifier-reproduced): the shard-resume
  payload now carries torch CPU+CUDA RNG state, restored after the start-of-run reseed with
  refusals on a changed dropout device or CUDA device count — diff-verified, and the
  stream-continuity test carries a negative control that fails without the restore; donor
  structural infeasibility is counted via `_min_frames` in the train sampling diagnostics, the
  fixed-final held diagnostics and the preflight `candidate_stats`, pairing kept (conservative
  direction unchanged); the CUDA/python lattice parity test gained two `d_min=2` skip-arc cases
  and was EXECUTED on a GH200 (`log/parity_test.1445759.out`: the test's own ok line, 1 passed —
  exit 0 alone cannot distinguish passed from skipped there), pinning backend agreement at 1e-3
  on scores, gamma and both gradients in D7's own topology; all five CPU unit tests pass in the
  verifier's own run under the project env; all four D7 job hashes and the merge are unmoved by
  the fix commit.
- 2026-08-21 (parity fix verified; first D7.1 run root-caused; amendment registered): commit
  `91c437a` is diff-verified faithful to the registered operational parity rule — losses keep
  exact equality; F is the max over 2 extra re-runs of the CONTROL model of the max abs gradient
  difference against its own first run, each `grads` call restoring the saved CPU+CUDA RNG state
  and zeroing grads on the fixed first batch, gradients snapshotted as fresh flattened CPU
  copies; PASS iff cross <= 3F and F <= 1e-4, with the F > 1e-4 case failing as the distinct
  backend-too-noisy defect; the report carries F and cross. Hash-neutral by construction:
  `source_identity` is an instance attribute, not a constructor argument, so no job hash can
  move. After the user's restart the preflight PASSED on its own artifact
  (`D7OnlinePreflightJob.ZxfANwBZYpaI/output/preflight.json`: verdict PASS, losses exactly equal
  at 9.983121871948242, F 7.391e-06, cross 4.053e-06 — same order as the diagnostic's 5.5e-06/
  2.6e-06, confirming noise-floor calibration was the right form; candidate gradient delta
  0.02046 confirms candidate-only gradient flow). BOTH D7.1 trainings
  (`D7OnlineTrainJob.j16rTskXF1QU` / `.WA1bqjXQtzeZ`) then failed closed at 21:32 in
  `_make_items` on the SAME row — own pseudo-text infeasible on own audio under d_min=2
  (`3889-130125-0028`: 481 states, min feasible 400 frames, T=356) — identical raise in both
  arms, so the matched-arm property held even in failure. Verifier census over all 281,241 pool
  rows with the production law (`scripts/d7_own_infeasible_census.{py,json}`): exactly 4
  own-infeasible rows, all train-role, zero internal-held; all four are runaway-repetition
  greedy texts. The incumbent recipe's own artifact defines the handling —
  `PsiAlignTrainJob.dsMKgPHQApyR` trained after "pairs: 28538 (27111 train / 1427 held out),
  1 dropped as U > 2T" — so the raise was an implementation over-strengthening of the
  exact-control-recipe-verbatim contract. Drop-and-count amendment with a named-four-row bound
  registered in `PLAN_3E1.md` D7 Status (including the D8 bed-wide-greedy-feasibility
  consequence); one implementer edit in `_make_items` plus one further user-run d7 manager
  restart (clearing both train error markers) are pending.
- 2026-08-22 (round verification: D7 drop law, D8.0, both VERIFIED; clause-(a) ruling issued).
  D7 drop law (`e2a421b`) verified end to end against the amendment, by diff read and agent
  reproduction: named four-row constant with a train-role pre-assert (d7_online.py:32-42,
  352-354), drop-and-continue keeps rows out of shards/held so no unit NLL, `L_U->z` or
  `L_online` visit exists for them, realized-set-equality raise (:378-383), naming per role in
  the train.txt report line and in monitors.json (`own_infeasible_dropped`, counts,
  `anchor_rows`), donor path untouched (reads index/store only; all four rows present in their
  speaker lists). The dropcheck artifact reproduces independently: full load drops exactly the
  four, 267,175 train / 14,062 held (= index 267,179 - 4 / 14,062; ten shard sizes sum to
  267,175); the preflight's shard-0 load keeps 26,743 rows and drops none. Hash neutrality
  proven at the sisyphus level: no `__init__` signature or config change, `source_identity` is
  an instance attribute and not hashed, `j16rTskXF1QU`'s info lists only the unchanged ctor
  args. The proposed `-co` restart is safe: the only error-state jobs in the graph are the two
  train dirs, both with empty work/ and output/ (no checkpoint to discard); pool and preflight
  are finished and untouched by `-co`. All five D7 unit tests pass in the verifier's own run.
  Two implementer follow-ups, their lane: `scripts/d7_parity_diag.py:25` still unpacks the old
  4-tuple and crashes if rerun; the cited commit `4dc65a3` is the pre-amend twin of branch head
  `a3dd6c7` (same message, D8 files byte-identical; the amend dropped an out-of-scope
  config_sae_1g_v1 hunk) — cite `a3dd6c7`.
  D8.0 verified: the v2 guard's docstring (d8_feasibility.py:51-61) and logic (:325-327,
  :499-509) match; a full offline re-run of the v2 reader reproduces EVERY field of all five
  slices of the binding artifact exactly (group counts, distinct histograms, law conflicts,
  collapse counts, tau_star, clause booleans, median ESS to 1e-9) and re-derives all three
  verdict branches (theta v2 UNRESOLVED / theta v1 NO-GO / fork REPORTED-ONLY); whitelist
  discipline confirmed at source (rows enter only via kind in {rollout, greedy}, `wer` never
  read, no reference input); the v1 jobs are preserved as superseded evidence; the store joins
  and medians confirm verdict 59's frame diagnosis (dump-joined pooled store median 169 vs raw
  50 Hz 674/695), and the raw store covers all 512 binding-slice tags — so the registered v3
  read (clause-(a) ruling, `PLAN_3E1.md` D8 Status 2026-08-22) is executable offline with no
  new dump. ONE DISCREPANCY, label only: "exercised on 256 groups / 25 groups" counts collapse
  CLASSES, not groups — 256 classes across 237 groups at T=0.7 and 25 across 24 at T=1.0; the
  job docstring at d8_feasibility.py:45 shares the mislabel (code :274-279 increments per
  class); numbers real, unit label wrong — implementer to fix both wordings. Reading notes:
  `a3dd6c7` is timestamped seconds AFTER the v2 jobs finished — pre-registration is carried by
  the hashed `reader_revision=v2` job parameter and the in-job verdict rule, not by commit
  order; approach 34's table omits the artifact's T=0.5 slice (conflicts 4,199/4,693, distinct
  0 / scorer-free 10, tau* 0.05) — add the row; the commit message's "5,096 of 6,656" is the
  pre-dedup member count, the log's 5,730 post-dedup denominator is the correct one.
- 2026-08-22 (D8.0 v3 round VERIFIED; clause-(a) GO confirmed; two conservative deviations
  ratified). The v3 reader (speech-llm `3843918`) implements the 2026-08-22 ruling clause by
  clause, confirmed in code and by a fully independent recompute of the binding slice from the
  raw dump rows and the raw 50 Hz store (own join/dedup/median logic, project primitives only):
  512 groups, 5,730 distinct scored classes, ZERO empty/unencodable/infeasible members against
  the operative T_i, with-greedy median 12.0 (rollouts-only also 12.0) — exactly the artifact,
  so clause (a) is GO at threshold 3, and the verdict first existed in the job's own output as
  required. The margin is structural, not marginal: raw-store median 695 frames over the slice
  vs the pooled 169/174 that drove v1's 88.9% exclusion, tightest single-utterance margin 65
  frames. The fork v3 read matches v2 field for field, and its operative exclusion 18/101,190 =
  0.0178% is digit-identical to v2's law-conflict count — the measured genuine rate the 5%
  valve was priced against. All four v1/v2 job dirs untouched; only the two v3 jobs are new;
  aliases repointed. Verdict hygiene checked: 59/60 originals unchanged with accurate
  corrections below, 61's confirmation true, 62 rests on the tables. TWO RATIFICATIONS of
  implementation-over-ruling deviations, both conservative: the safety-valve denominator counts
  ALL excluded scored members (superset of operative-infeasible — trips earlier), and the
  coverage assert spans all 34,106 dump ids (superset of the slice — fails closed sooner);
  both stand as the operational form. Hand-backs, implementer's lane: verdict 63's
  "shaped-versus-acoustic-only runs 0.30-0.66" — the T=0.5 value is 0.2857, so the range reads
  ~0.29-0.66, and that clause rests on the JSON rather than a table column (add the
  acoustic-spearman column or cite the artifact in the verdict); the State's "operative v3
  below" pointer needs the commit `3843918` (the Catalog copy was completed by the verifier as
  an objectively dangling reference); trivia: the commit message says 32 mechanics checks, the
  script now prints 47, all passing. Consequence recorded for D8.1a, no action now: the binding
  slice's rho(shaped, LM-only) 0.9790 sits above the registered 0.95 arm-selection bar — if
  D8.1a reproduces it, candidate-shaped is not funded and only candidate-acoustic trains, per
  the registered rule; the D8.0 value is provisional and selects nothing.

- 2026-08-22 (D7.1 completion round VERIFIED; hand-backs closed). Every approach-32 D7.1 claim
  confirmed against the raw job artifacts: both arms' `monitors.json` carry
  `own_infeasible_dropped` = exactly the four registered train-role rows (no held drops, key
  absent), `anchor_rows` 267,175 / 14,062, digit-identical to the offline dropcheck and to each
  other; NLL 2.52588->2.5259 (control) / 2.53190->2.5319 (candidate), mean `L_online`
  0.0102246->0.010225 / 0.0075412->0.007541; the train-side sampling files are byte-equal across
  arms and all ten per-shard row/frame counts and step boundaries are bit-identical arm to arm.
  An exhaustive recursive diff of the two arms' monitors finds ONE non-metric difference:
  `online_weight` 0.0 vs 1.0 — the A/B is single-variable at the artifact level, not just by
  intent. Single 14-minute run each (13:59/13:58), no resubmit, `.cleared.0001` failure dirs
  preserved; both `model_final.pt` present; the four dropped anchors were all ordinary_window
  donor cases (census 266,138 -> 266,134, fallback untouched), so the donor law was untouched by
  the drop. Both manager logs end at sisyphus's "All calculations are done" EOFError, confirming
  the corrected STALLED-artifact reading in State. Verdicts 64-65 rest on the tables and are
  accurate; the 26% claim recomputes (0.7376x). Precision notes, no action needed: held
  `L_online` averages over 14,008 rows — the 54 singleton anchors contribute no online term —
  matching the gate's own "per eligible held anchor" wording, same denominator both arms;
  control shards 6/9 and candidate shard 9 report `u_to_z` exactly 0.0 (fine for a satisfied
  hinge, worth knowing if `u_to_z` is ever read as a live signal). Hand-backs from the previous
  round verified closed: the rho(shaped, acoustic-only) column matches the artifacts at every
  digit (0.3000/0.2857/0.3132/0.5497/0.6593; fork 1.0000), verdict 63's corrected range
  0.2857-0.6593 is the exact min/max, the State pin names `3843918`, and the mechanics test
  reproduces 47/47 PASS live at branch head. The D7.2 clause-2 flag is acknowledged in
  `PLAN_3E1.md` D7 Status: the gate does not move, and a failure closes D7 without a policy leg
  per the registered law — D7.2 is authorized to build and run as registered, no new word needed.
