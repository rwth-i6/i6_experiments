# SAE §3e.1 — scorer trainability without collapse (ladder D0–D4, `archive/SAE_3e1_spec_legacy.md`)

## State

D9 IS COMPLETE AND NOTHING OF THIS PHASE IS QUEUED OR RUNNING. D8 closed on verdict 84; D9's
registered reads are all banked (approach 39, verdicts 85-86).

Run pointers. Arm 2, the 1-best refit D9.2 reads: `S/d9_refit/D9OnlineTrainJob.nJQy199AQZQu`
(2,421 steps, held NLL/frame 2.2550). The read: `S/d9_refit/D9EtaReadJob.A7QvXl7VR7wl`. Arm 3,
`S/d8_train/D8ScorerRefitJob.XvPF118rphQP`, stays in its guard-fired error state by planner ruling
(2026-08-24): DO NOT clear, retry or delete it while D9 is open, and a manager exiting on it is not
a stall to repair.

NEXT EXPERIMENTAL ACTION: none of the implementer's own. The phase closes only on the USER's word
over verdict 86, with the D8.3 authorization question attached.

Pending decisions carried into that word (details under Open findings): the clause-3
point-versus-CI eligibility convention is still dual-reported and unpinned; the D2 winner rule's
ladder-floor reading is likewise unpinned and is what decides `d2_states` against `d2_contrast`,
while `config_sae_3e1_d3_v1.py` hard-codes `WINNER='d2_contrast'` provisionally.

## Gates

Predefined gates, as registered. Where a gate was amended or a conclusion overturned, both readings
are kept and the current one is named.

**D0.** Suspect-vocabulary threshold pre-registered from the mechanism before any table was read:
`min_excess = 0.002` (one extra occurrence per 500 tokens).

**D2 winner rule (approach 9), fixed before the D2 read.** An arm is eligible only if its held
CE_loo is below the unit-marginal floor 6.03, its `text_explained_loo` is not below the pre-loop
floor, and its corruption-ladder spearman is not below `psi_g_tc100`'s; among eligible arms the
winner is the one that most reduces the insertion discount (`psi_g_tc100`: 0.0584), ties broken by
the D0 rollout beta at matched WER. If no arm reduces the discount by more than the bootstrap CI
half-width (~0.005) there is no winner and D3 is not funded from D2 -- the fallback is the
planner's call.
- AMENDMENT (operative): the rule reads the STATE-MATCHED control pool, intersection reads and the
  paired cross-arm difference CI (`archive/SAE_3e1_spec_legacy.md` D1 build item (b)), because the
  frequency-drawn control is state-count-confounded (verdict 8 correction).
- AMENDMENT (operative): gate v2 (i)'s round-to-round improvement clause is FLOOR-ONLY for
  changed-text candidates, since the held text is unrepaired pseudo-text and would ask a repair arm
  to model the defect it removed.
- UNPINNED: the ladder floor's "not below" is CI-read in this log and point-read in the rule text.
  Point reading elects `d2_states`; CI reading elects `d2_contrast`. See Open findings.

**D4 curation-view admissibility.** A candidate view is admissible only if (e) it ranks -- within-group
spearman(signal, -WER) with a CI excluding zero -- and (f) its partial beta on the suspect count at
matched WER does not pay for the filler. A refresh round may not curate with an audio-free view alone.

**D6 (approach 20) bars.** spearman/eta not below the comparator's, held ce_loo within +0.05 of it,
ins_1 >= +0.14 growing in k, and mono(ins) out of last place.

**D6 swap-in (approach 21a) pre-registered in-loop confirmation.** The control's sub-epoch-3
regression (5.34/9.50 at the fork to 6.56/11.15 one sub-epoch later, never recovered through
sub-epoch 10) shrinks.

**D6-PERIODIC acceptance gate (as first run).** Four clauses -- (i) rank quality, (ii) held
likelihood, (iii) insertion price, (iv) corruption ladders -- read against the last ACCEPTED
scorer, swap on pass and keep on fail, with a two-consecutive-failure stop rule.
- AMENDED BY THE USER 2026-08-18: the acceptance step was REMOVED from every gold-seeded periodic
  arm and the arms relaunched ungated. Approach 22's table is the only surviving record of what the
  gated run decided.

**D6-PERIODIC/GAN-FROZEN (user-directed 2026-08-20), verbatim:** "a durable/actionable recency
benefit requires periodic leg 8 to beat frozen leg 8 on both dev-clean and dev-other". A frozen
final-leg win is the gate's named "refresh has no established durable benefit" case; an early-leg
transient is pre-registered as non-licensing.

**D6-PERIODIC/GAN960-FROZEN.** Leg 8 beating this arm's own init 13.11/16.82 on both splits;
matched-leg deltas against GAN-FROZEN are reported and select nothing.

**HOM.** 0a admission floor: >= 5 % of corpus tokens sit in a homophone class. 0b bar: median
|delta lm_prior| against median |delta recon| at the arm's own lam_lm 1.0 and per-unit-frame
normalization. Arm primary read: stay within 0.3 dev-other WER of D6-PERIODIC/GAN at every matched
leg.

**D7-v2 / D7.0b structural floors (frozen 2026-08-21, before any scorer read).** Training: K=4
(2+2) row-local Q2, ten-table regular construction, requiring 6,778 rows and 201 speakers. External:
K=1, M=1, no-band, max-cardinality/minimum-nuisance cap-three matching admitting at least 435/725
dev-clean and 465/775 dev-other sources plus 32/40 and 27/33 source speakers, retaining fixed
725/1500 and 775/1500 split weights, never shrinking the all-1,500-row Acceptance gate. Fail-closed:
no solver retry, floor relaxation or third graph amendment is authorized.

**D7.0 parity clause (original).** Two deepcopies of one model, same restored RNG state, same batch
at `online_weight=0`, must produce byte-equal loss AND byte-equal gradients
(`torch.equal(g_control, g_parity)`).
- AMENDED (operational parity rule, commit `91c437a`, CURRENT): losses keep exact equality; F is the
  max over 2 extra re-runs of the CONTROL model of the max abs gradient difference against its own
  first run; PASS iff cross <= 3F and F <= 1e-4, with F > 1e-4 failing as a distinct
  backend-too-noisy defect. Reason: exact gradient equality is unreachable on this backend
  (approach 32).

**D7.2 gate.** Four clauses -- (1) paired internal-held `L_online` admission, (2) internal-held
per-frame NLL no greater than the control's, (3) the 1,500-row external Acceptance gate v2,
(4) scorer parity. The gate passes only if all four hold; failure closes D7 without a policy leg and
no sampler or temperature rescue may be selected from the result.

**D7.3 (corrected 2026-08-20).** The prospective scientific gate is improvement over the exact
matched control `ReturnnTrainingJob.kr1foUV6lecx` (14.45/19.69) on both dev splits, conditional on
required scorer parity. The former absolute 13.89/17.84 clause is RETIRED as unsupported for a
one-leg causal read; the GAN init 13.89/18.34 and the prior frozen-loop 12.68/17.57 are report-only
anchors.

**D8.0 / D8.1a no-go clauses** (the same three, re-applied verbatim at D8.1a): (a) median distinct
support >= 3; (b) at least one grid tau inside the [1.5, 8.0] ESS band; (c) median token R2 below
the 0.5 ceiling. A binding-slice exclusion rate above the ruled 5 % safety valve returns UNRESOLVED
instead of feeding clause (a).
- CLAUSE-(a) RULING 2026-08-22 (CURRENT): the structural-infeasibility exclusion is evaluated
  against `T_i` from the frozen raw 50 Hz store `S/quantize_states/PackUnitsJob.I0uzRMfUrKWC` with
  coverage asserted; the per-unit prior currency still divides by each dump's OWN store. The two
  earlier readings -- dedup-only, and the as-run pooled-store exclusion -- are both rejected.

**D8.1a arm-selection rule** (pre-registered, reads only D8.1a statistics): if
spearman(shaped, acoustic-only) > 0.95 the two arms are operationally identical and only
`candidate_acoustic` is funded; a spearman(shaped, LM-only) above the same bar would strike the
shaped arm out as free English.

**D8.1a support deviation ruling (2026-08-22).** The registration reuses the D7 pool's greedy 1-best
at identical hash; a regenerated greedy is admissible ONLY against a zero-mismatch normalized-text
equivalence read over all 281,241 utterances.
- RULED latest+1 after that read failed: support restored to the registration's own reader rule --
  the D7 pool greedy at identical hash as an explicit weight-job input, dump `kind=="rollout"`
  whitelist, the dump's regenerated greedy quarantined as the divergence record, same-string scoring
  law for the differing minority, both-sides coverage asserts.
- RULED latest+3: the mixed convention (dump columns on agreeing tags, text path on differing ones)
  is REJECTED; every column of the pool member comes from the text path on all 281,241 tags, with a
  pre-registered convention-sensitivity line.

**D8.2 gate, verbatim:** "D8.2 passes for a candidate only if all four hold" and "failure at any
rung closes D8 without a policy leg; no tau, temperature, support or coefficient rescue is selected
from results." `delta_NI` must be computed from the CONTROL's own held spread before any candidate
number is read, at D7.2's convention and resample count.

**D8.4 (registered after the user reopened D8).** A constructed clause battery gates spend but never
closes a phase, so the phase question is answered by ranking quality in a fair paired comparison.
Three-way verdict, owned by the reader: BETTER when the 95 percent interval excludes zero for the
candidate, WORSE when it excludes zero against it, otherwise INDISTINGUISHABLE, resolving to the
control under the standing incumbent-tie rule. Pins: `n_boot=10000`, `seed=42`, two arms differing in
`model_pt` alone on the SAME frozen dump at the same temperature, fairness floor of 512 shared
groups (the reader refuses any other bed). Nulls (length-only, OOV-count, audio-free margin) and the
D8.2 clause verdict print as context and cannot move it; every null is arm-internal and is never
differenced.
- AMENDED 2026-08-23 closing (CURRENT): the primary pair's units join is re-pinned from the sae3d
  quarter-rate store to `MergeUnitsPklJob.ncxcd3vouD5E` (50 Hz enc50), same dump and same draw, after
  the registered join failed closed on the bed guard (verdict 83). Clause (a) is scoped to
  stored-column reads. The failed-closed artifacts stay banked and in the graph at their finished
  hashes.

**D9.0 (amended 2026-08-23, CURRENT).** Gate on the incumbent census plus the structural d_min>=2
census; the read-set rule applies at D9.2; a structurally-alignable row scored non-finite by a refit
is a STOP. The original three-arm census is withdrawn (it cannot precede D9.1).

**D9.2.** Registered as a three-arm read; AMENDED BY REPLACEMENT 2026-08-24 (CURRENT) to a TWO-ARM
read -- arm 2 (1-best refit) against arm 1 (incumbent), per-group eta, paired per-group delta eta,
`bootstrap_delta_eta` at n_boot 10000 seed 42, D8.4 machinery and constants verbatim; the read set is
the groups where BOTH arms score every member finitely with per-arm drop counts printed; the
structural-census STOP clause is unchanged. The soft-EM-versus-1-best attribution contrast is STRUCK
as unreadable. A refit arm is adopted only on an interval excluding zero in its favour. Option (ii)
(re-dump for diversity) not funded; option (iii) (threshold edit) rejected as a post-hoc gate edit.

**Standing constraints.** Minimum-duration topology d_min>=2 on every new scorer plan (user
2026-08-15). Plain sclite WER only, never a rescored or normalized variant. Labels never train or
select; gold reads report and select nothing. G3 bars (`archive/SAE_3a_spec_legacy.md` §6):
gap_true >= 0.0248, spearman >= 0.17, audio-margin CI excluding zero.

## Results

Path prefixes: `T/` = `work/i6_core/returnn/training/`, `F/` = `work/i6_core/returnn/forward/`,
`S/` = `work/speech_llm/sae/`.

**1. AR text-usage gate along the co-trained trajectory** (SAE.md §3e.1 queue item 2). The §3c 100 h
replay arm (`freeze_ar=False`) re-read with the §2.5 usage gate at each of its own checkpoints:
`gate = ln(ppl_shuffled) - ln(ppl_true)` on the 10 h seed dev subset (5000 utts), avunits k500
stream, within-dev derangement at seed 42, p=1.0 history masking -- one protocol, only the
checkpoint moves. Each epoch's `ar.`-stripped sub-state is a standalone `SaeTokenLmV1` checkpoint
(`ExtractAvSubmodelJob`, `submodel_prefix="ar."`). ep0 is the frozen AR every loop arm starts from,
the finished `p10` cell of `config_sae_2s_ar_usage_gate_avunits_v1` reused by asserted job id, so
the anchor predates the question; its CE 5.7444 reproduces that AR's logged dev CE 5.7371.
Interpretation floors on this stream: unit marginal 6.0072, uniform ln 500 = 6.2146.
Artifacts: `S/scorer_diag/ArUsageTrajectoryJob.9Ughq5htDaXx`; replay arm
`T/ReturnnTrainingJob.KBTADeS7Qp1G`; ep0 `T/ReturnnTrainingJob.ExCoQDKtXAGH/output/models/epoch.050.pt`;
ep0 usage cells `F/ReturnnForwardJobV2.GBuKgHp3GNlz` (true) / `.HKsuKQJdUwGA` (shuffled).

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
(28 539 utts, 963 857 tokens) minus its rate in the LibriSpeech LM corpus (803 M tokens), at the
pre-registered `min_excess = 0.002`. A ratio test was rejected in the plan because it top-ranks rare
words; sensitivity: 4 tokens at 0.001, 1 at 0.005. `S/scorer_diag/SuspectVocabJob.7LSZhTXKculV` over
`S/scorer_diag/LmWordCountsJob.SqAFPqiRBD9k` (`i6_core/tools/download/DownloadJob.g4jClO48cAvP`).

| word | n_pseudo | rate_pseudo | rate_lm | excess |
|---|---|---|---|---|
| to | 45 561 | 0.047269 | 0.027452 | **0.019817** |
| of | 34 445 | 0.035737 | 0.030868 | 0.004869 |
| buy | 2 984 | 0.003096 | 0.000061 | 0.003035 |
| vary (below threshold) | 1 044 | 0.001083 | 0.000005 | 0.001078 |

**3. D0 mechanism discriminator** (queue item 2, second half). Bias vs noise vs group blindness on
finished artifacts only -- 512 tc100 utterances sampled from theta_0^G at G=12/T=0.7, the loop's own
operating point, re-ranked by three psi_align scorers that share the rollout set
(`F/ReturnnForwardJobV2.J9yA1eYnxwYA`) and the unit stream (`S/quantize_states/AssignUnitsJob.X8DBup0jQlhR`)
and differ only in the text they were fitted to: `psi_g_tc100` (the loop's own scorer), `psi_g_seed`
(same recipe, 10x less of that text), `gold_enc50` (10 h gold text -- the never-contaminated control
that localizes any effect to the training text rather than to the bed). Labels enter as evaluation
only. Both live reward variants are read: `recon`, and `shaped` = recon + 1.0 * prior/n_units (the
dumps normalize the prior per text token, so the job rebuilds the sum and divides by the utterance's
own unit count to restore the live `lm_prior_norm="units"` term). Bias statistic = group-centred
partial effect of the suspect count on reward with WER as covariate (`beta_ols`; `beta_rank` its
nonparametric twin), positive meaning the scorer PAYS for the filler at matched WER. Arm-invariant
rows come out identical across arms, which is the wiring check. Table
`S/scorer_diag/RolloutMechanismJob.vsl00qaCHQbP`; re-rankings `S/psi_align_jobs/PsiAlignRerankJob.QdHRXsev2Txh`
(psi_g_tc100 <- `PsiAlignTrainJob.kSYy0ADBgPGo`), `.2AUBSd8Y0oq0` (psi_g_seed <- `.SUAAuCS2o3pz`),
`.bZCAVAKWQq3I` (gold_enc50 <- `.IN3zmmGpH4Bv`).

Shared across arms: mean_wer 0.1670, oracle 0.1071 (G=12) / 0.1125 (G=8), greedy 0.1345, 512 groups.
Group contrast -- fraction of groups carrying the token that also hold a token-free member: **"to"
0.2334** (467 live groups, mean within-group count std 0.4824), "of" 0.1089, any suspect 0.0922.

| arm | spearman recon | spearman shaped | beta_ols "to" | beta_rank "to" | beta_ols any |
|---|---|---|---|---|---|
| gold_enc50 (control) | 0.5801 | 0.6300 | 0.1673 | 0.1792 | 0.1887 |
| psi_g_tc100 (the loop's) | 0.4959 | 0.5558 | 0.2425 | 0.2634 | 0.2514 |
| psi_g_seed | 0.4696 | 0.5404 | 0.2664 | 0.3219 | 0.2753 |

Selectors, within-group spearman(signal, -WER) with 95 % CI over groups: `lm_prior_units` 0.5020
[0.4737, 0.5308], `neg_n_suspect` 0.1855 [0.1510, 0.2186], `n_tokens` -0.0354 [-0.0710, 0.0004],
`psi_len_only` 0.0125-0.0354 with the CI straddling zero, `neg_n_oov` undefined (every row on this
bed has n_oov = 0). The "(arm-invariant)" label these were first logged under is corrected under
Open findings.

**4. Frozen external held pair set, and gate v2 (i)+(ii) read on it** (D1). The §1d student decoded
LibriSpeech dev as well as tc100, so (pseudo-text, enc50 units) pairs exist on 5567 utterances no
scorer in this program trains on; 1500 are taken by a seeded permutation of the id-sorted pool and
never move again, which is what `PsiAlignTrainJob`'s per-candidate 5 % split of its own corpus cannot
be. 1493 are feasible under both the true and the length-matched deranged pairing. All three D0
scorers are read on it, unrepeated and label-free. `S/scorer_diag/FrozenHeldPairsJob.E8UaEwRF65HW`;
held NLL `S/psi_align_jobs/PsiHeldNllJob.J1A028bt3Faw` / `.WrmDwFU9dVvV` / `.ag5DZ3A2Gd1K`.

| arm | ce_loo (true) | H_uni on these frames | text_explained_loo | usage gate (len-matched) |
|---|---|---|---|---|
| psi_g_tc100 | 2.7198 | 6.0324 | +3.3126 | +3.6853 |
| psi_g_seed | 3.0235 | 6.0324 | +3.0089 | +3.7036 |
| gold_enc50 | 3.1274 | 6.0324 | +2.8939 | +4.2821 |

**5. D1 filler probe battery** (D1). Paired text-side corruptions on the same 1442 held pairs that
survive every pairing's U <= 2T bound: at k = 1, 2, 4 randomly drawn slots, the filler and an
LM-drawn frequent word are written into the SAME slots (substitution) and inserted at the SAME slots
(insertion), and the same slots are deleted; the statistic is the per-utterance increase in `ce_loo`
over the untouched text, bootstrapped over utterances. Substitution asks what the filler costs to
write over a word, insertion what it costs to ADD -- and the G-track's degradation is made of
insertions. `S/psi_align_jobs/PsiTextProbeJob.eNVc8JTbm7n8` / `.qo8IB8MLA8ES` / `.rY39iGv8bhhi`.

| arm | del_1 | sub filler_1 | sub LM_1 | ins filler_1 | ins LM_1 | insertion discount k=1 / 2 / 4 | suspect state mass |
|---|---|---|---|---|---|---|---|
| psi_g_tc100 | 0.3336 | 0.3183 | 0.3219 | **0.0274** | 0.0859 | **0.0584** / 0.1129 / 0.2201 | 1.90 % |
| psi_g_seed | 0.3315 | 0.3010 | 0.3054 | 0.0172 | 0.0735 | 0.0563 / 0.1084 / 0.2199 | 2.19 % |
| gold_enc50 | 0.3941 | 0.3595 | 0.3570 | 0.0261 | 0.0851 | 0.0590 / 0.1056 / 0.2351 | 2.08 % |

Insertion-discount CIs at k=1: psi_g_tc100 [0.0537, 0.0634], psi_g_seed [0.0520, 0.0604], gold_enc50
[0.0539, 0.0639]. The substitution discount is ~0 in every arm (+0.0036 / +0.0044 / -0.0025). Ladder
spearman (severity vs `ce_loo` increase) 0.94 for substitution and deletion, 0.66-0.86 for insertion
(psi_g_tc100 only; across arms 0.55-0.87, see Open findings).

**6. Sampling-side contingency: contrast coverage and steerability vs temperature** (D0 coverage
co-requirement). The D0 dump already carries T = {0.3, 0.5, 0.7, 0.9, 1.0} at G=12, so this is a
re-read: coverage is the fraction of "to"-carrying groups holding a "to"-free member, steerable
additionally requires that member's live shaped reward to beat the group mean. Coverage is
arm-invariant; steerability is not, and WER enters as evaluation only.
`S/scorer_diag/CoverageTemperatureJob.JAP5gJQE0PwP`.

| T | coverage "to" | steerable (psi_g_tc100) | steerable / coverage | mean WER | oracle WER |
|---|---|---|---|---|---|
| 0.3 | 0.1359 | 0.1091 | 0.803 | 0.1386 | 0.1074 |
| 0.5 | 0.1645 | 0.1360 | 0.827 | 0.1467 | 0.1039 |
| 0.7 | 0.2334 | 0.1949 | 0.835 | 0.1670 | 0.1071 |
| 0.9 | 0.5270 | 0.3382 | 0.642 | 0.2994 | 0.1496 |
| 1.0 | 0.8212 | 0.5343 | 0.651 | 0.5153 | 0.2829 |

**7. D2 round-0 pseudo-text repair** (D2, corpus side). Rates of the three excess-mass suspects are
matched to the LibriSpeech LM corpus by removal only, with a per-utterance multiplicity cap read off
the LM corpus at matched utterance length (q99: 3 for "to" in the 20-30-token bucket). 60.6 % of
utterances are edited and the corpus loses 2.94 % of its tokens; no utterance is emptied, and the
repaired corpus differs from the contaminated one only where a token was removed.
`S/text_repair/RepairPseudoTextJob.o086K9a8uXDa`, corpora `S/text_repair/TextHfDirJob.UEAxxdGitOHu`
(control) / `.7Msi4BxlykgV` (repaired), LM reference `S/text_repair/LmLineStatsJob.l9ZJSEj8tP0S`.

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
The control corpus is asserted byte-equal to the one psi_g_tc100 was fitted to, id order included.
All four arms ran the full 30 epochs and each is read at its own best-held epoch; every
ce_loo-derived column is segmenter-dependent, so `d2_states`' entries in those columns are NOT
comparable to the cps-1.5 rows and are marked (*). Scorers
`S/psi_align_jobs/PsiAlignTrainJob.HTy12IMDmYdB` (d2_rate), `.DnBJxqz4sNQZ` (d2_contrast),
`.9pTbjjx29yVc` (d2_both), `.hxK0HTBZQSJa` (d2_states); held NLL `.PsiHeldNllJob.XvvciDyN3LyS`,
`.Z8quArGjzAj3`, `.1okicjOpTszW`, `.9D2ywKhnL5ZH`; probes `.PsiTextProbeJob.wkAV3KfAUwW9`,
`.g3p5aA7nBONQ`, `.8LGrp6IuVyzD`, `.eRBqqPfUtf6k`; re-rankings `.PsiAlignRerankJob.jRvegq7Bf7lu`,
`.zAzQGZbtxrw9`, `.DQQLmfIhPTOe`, `.DVcQhryzLU2j`; parity `.PsiScorerParityJob.bBjvefspGS4L`,
`.0U0yG8pdt6fB`, `.O7WiXL0OfmvA`, `.g5gIUiLRMqLg`; cross-arm D0-dump re-read
`S/scorer_diag/RolloutMechanismJob.uDTs6ZlhOFQa`; steerable coverage vs T
`S/scorer_diag/CoverageTemperatureJob.Ku9zNUNDK12D`.

| arm (corpus / weight / cps) | best ep | held ce_loo | ins. disc. k1 | ladder filler_ins | beta_to | spearman | steerable | susp. mass % |
|---|---|---|---|---|---|---|---|---|
| psi_g_tc100 — contaminated / 0 / 1.5 (incumbent) | 9 | 2.7198 | +0.0584 | 0.6552 | 0.2425 | **0.4959** | 0.1949 | 1.902 |
| d2_rate — repaired / 0 / 1.5 | 11 | 2.7195 | +0.0595 | 0.6180 | **0.2232** | 0.4931 | 0.1906 | **1.724** |
| d2_contrast — contaminated / 1 / 1.5 | 28 | **2.7139** | **+0.0558** | 0.6820 | 0.2469 | 0.4808 | **0.2013** | 1.881 |
| d2_both — repaired / 1 / 1.5 | 28 | 2.7332 | +0.0619 | 0.6652 | 0.2437 | 0.4801 | **0.2013** | 2.053 |
| d2_states — contaminated / 0 / 0.5 | 11 | 2.1278 (*) | +0.1475 (*) | **0.8540** | 0.2389 | 0.4878 | 0.1906 | 3.268 (*) |

The `ins. disc. k1` column is the FREQUENCY-DRAWN discount and is kept only because approach 9's
original rule names it; it is state-count-confounded and approach 10 replaces it. 2 T/U on the held
set is 9.77 at cps 1.5 and 3.92 at cps 0.5. `beta_to`, `spearman` and `steerable` are the D0-dump
re-reads at T=0.7 with every arm re-ranking the SAME rollouts, so those three columns are cross-arm
comparable even for `d2_states`; contrast coverage itself is arm-invariant at 0.2334, so `steerable`
moves only through the scorer. All four candidates PASS `PsiScorerParityJob` at
max |online - offline| = 0, and all four clear the three G3 bars (margins +0.146 to +0.154, all CIs
overlapping). COLUMN-MIXING CAVEAT under Open findings.

**9. D3 frozen-repaired G-track control arm** (D3). The winner's scorer is frozen into the same
`config_sae_3a_gan_loop_960h_v1.baseline` that builds the arms it controls, so bed, data and
schedule differ in one input only; bar 2 is the suspect share of sclite insertions, read off the
recogniser's own alignment at four sub-epochs. Winner rule and its amendments are in Gates. Gate v2
(i)'s improvement clause is not used for eligibility here (the held text is unrepaired pseudo-text)
and is reported alongside. Three of four sub-epochs are in (`psid2_contrast` = the frozen
`d2_contrast` scorer, against the same arm on the incumbent `psi_g_tc100`); the insertion columns are
dev-clean. Arms `T/ReturnnTrainingJob.rJWSC5xOsrf2` (shaped) and `.L6FwOOpffNL4` (recon).

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

The shaped/d2_contrast sub-epoch 2 row (12.68 / 17.57) is the best previous frozen-scorer loop result
on the GAN init and is the anchor D9 pins against; provenance chain in the D9 entry (approach 39).

**10. State-matched control pool, and the D2 selection read on it** (D1 build item (b), D2
admission). The LM control is redrawn from a pool holding the filler's own emitting-state count under
each arm's own segmenter -- 57 one-state words at cps 1.5 and 51 four-state words at cps 0.5, of the
same 6 472 above the rate floor -- where the frequency-drawn pool averages 2.70 states against the
filler's one (8.16 against four at cps 0.5), and per-utterance `ce_loo` is dumped so every cross-arm
number below is a PAIRED difference on the 1442 utterances all seven arms could score. The
frequency-drawn pairings are drawn from their own generator and reproduce the pre-extension jobs to
the last digit (0 of 7 arms differ on any statistic), so this is a control added beside the old one,
not a re-measurement of it. Batteries `S/psi_align_jobs/PsiTextProbeJob.WBXWwmZIK7HY` (psi_g_tc100),
`.4KpANAZV864A` (psi_g_seed), `.a8WsW4jjddcq` (gold_enc50), `.jQPGx36tCccz` (d2_rate), `.cMO136SC9uUu`
(d2_contrast), `.rNCJA9Y987bY` (d2_both), `.lcbBuAIimK11` (d2_states); each carries `items.json`.

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
count (`S/gate_table/PsiGateClauseTableJob.x0d7dYpOdilI`, 7 finished arms), printing both readings of
the ladder floor rather than choosing one; the selector block of the D0 discriminator gained a
filler-affinity twin (partial beta of the suspect count on each CURATION view at matched WER,
group-bootstrap CI), opt-in and hash-excluded so the audited D0/D2 tables keep their ids
(`S/scorer_diag/RolloutMechanismJob.jYDxg98sWJIj`), and it scores `ar_recon` -- the G-track AR's own
reward, carried in every re-rank dump as `recon_incumbent` and never read as a selector before -- as
the one audio-conditioned view on offer. `SaeGrpoModelV1` gained `av_checkpoint_prefix`, which imports
a previous round's policy out of the loop's own state_dict and leaves psi at the newly accepted
scorer. The clause table on the seven finished arms reproduces the audited D2 verdict (ladders-worse
3/1/3/0/0/0; point reading -> `d2_states`, paired-CI reading -> `d2_contrast`), and the import
override carries 719 `av.*` keys out of a live 960 h loop checkpoint -- name- and shape-identical to
theta_0^G's own AV SFT checkpoint -- while dropping its 82 `psi.*` keys.

Candidate curation views under both pre-registered bars (`to` for (f)):

| view | (e) spearman(signal, -WER) | (f) beta on suspect count at matched WER | admissible |
|---|---|---|---|
| `lm_prior_units` | **0.5020** [0.4737, 0.5308] | **-0.0937** [-0.1405, -0.0449] | yes |
| `neg_n_suspect` | 0.1855 [0.1510, 0.2186] | -0.9232 (by construction) | yes |
| `ar_recon` (the G-track AR's own reward) | 0.0944 [0.0571, 0.1321] | **+0.0716** [0.0102, 0.1214] | no -- (f) |
| `psi_len_only` | 0.0354 [-0.0018, 0.0722] | -0.0031 [-0.0690, 0.0465] | no -- (e) |
| `n_tokens` | -0.0354 [-0.0710, 0.0004] | +0.3180 [0.2610, 0.3835] | no -- both |

**12. Refresh round 1: a curated pool from theta_0^G's own rollouts, and a scorer refit on it** (D4).
One fresh dump of theta_0^G over all 28539 pseudo-text utterances at the loop's own T=0.7 and G=12
(`F/ReturnnForwardJobV2.lQMOR5n2ntcS`; the step is latency-bound in the decode, so 8 utterances per
step rather than 4, G at the loop's value), curated by two-view agreement with both advantages
positive and one member per utterance, on top of the rate-repaired round-0 corpus as the anchor at a
floored 50 % share. A curated pool holds an utterance twice against one unit stream, and both rows
are pairs the NLL term maximizes, so the matching-aware term drops a same-utterance negative exactly
as it drops a structurally impossible one -- without that mask 5.48 % of rows per epoch contrast a
reading of their own audio, and since `_batches` sorts by (T, U) the twins land adjacent and the
shorter always wins, which is the length detector the (T, U) bucketing exists to rule out. The
candidate is the `d2_contrast` recipe refit from scratch on anchor + curated, judged by the same
frozen held set, state-matched probe battery and clause table the D2 arms were judged by, with gate
v2 (i) floor-only. Chain `S/curate/CuratePairsJob.0Xs8AhGwRn80` -> `S/psi_align_jobs/PsiAlignTrainJob.cRIigmxPtt75`
-> `.PsiHeldNllJob.Q24MX1AhUGFK`, `.PsiTextProbeJob.UEVRWgPseI16`, `S/gate_table/PsiGateClauseTableJob.5oMRtYKrhE3C`.
`d2_contrast` is the same recipe on the uncurated round-0 corpus, so its column is what curation is
worth. Paired on the 1442 utterances every arm scores.

| frozen held set, pinned ep28 | incumbent `psi_g_tc100` | `d2_contrast` (uncurated) | round-1 `r1` |
|---|---|---|---|
| state-matched insertion discount k=1 | 0.0172 | 0.0082 (-0.0090 [-0.0118, -0.0062]) | **0.0064** (-0.0108 [-0.0140, -0.0077]) |
| k=2 | 0.0323 | 0.0180 (-0.0143) | 0.0122 (-0.0200) |
| k=4 | 0.0561 | 0.0323 (-0.0238) | 0.0140 (-0.0422) |
| held ce_loo (H_uni 6.0324) | 2.7198 | 2.7139 | 2.7168 |
| filler_ins ladder spearman | 0.6552 | 0.6820 (+0.0268 [+0.0065, +0.0472]) | 0.6771 (+0.0219 [-0.0006, +0.0440]) |
| ladders nominally / significantly worse | -- | 2 / 0 | 4 / 0 |
| eligible, point / CI reading of the floor | -- | no / yes | no / yes |

**13. Error anatomy along the collapsing trajectory** (D5(a)-1). The four rates recomputed from
sclite's own counts against the reference length for both dev sets at ep0-ep4 of the 100 h
seed-replay joint-AR arm, alongside the hypothesis/reference length ratio, the top inserted words and
the suspect set's share of all insertions. No new decoding: the ten finished `ScliteJob` report dirs
are pinned by absolute path. `S/scorer_diag/PolicyAnatomyJob.eMeWgTsMWSRM`.

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
scorer for six texts on one 5000-utterance seed dev subset -- gold, theta_0's decodes (`dec0`) and the
arm's own decodes at ep1-ep4 -- so only the scorer and the conditioning text move and the gold column
is approach 1's `CE_true` column by construction (the same five forward jobs, one instrument, not two
agreeing). The policy decodes are merged from both dev sets' `search_out`, lowercased and NFKD-folded
in place (12 fold events over 7 utterances; nothing is dropped).
`S/scorer_diag/AllegianceGridJob.kR0YA9kfUd4s`.

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
so all five cells re-rank the same rollouts (byte-identical 6400-row census) and eta is attributable
to the scorer alone; the ep0 cell is the finished `theta0_avunits_p10` job, asserted by job id as a
free wiring anchor. eta is the ratio of across-group means, not a mean of per-group ratios. Shared
across all five at T=0.7: mean WER 0.1076, oracle 0.0316, greedy 0.0525. Dumps
`F/ReturnnForwardJobV2.p9y6xUfCZ4sW`, `.4nTgyBPY2SlM`, `.dEOPiBW4ADQM`, `.aoynOYDHBqLs`, `.9px8IEReJyUG`.

| scorer | recon @0.7 | std_wg @0.7 | spearman @0.7 | sel_wer @0.7 | eta @0.3 | eta @0.5 | eta @0.7 | eta @1.0 |
|---|---|---|---|---|---|---|---|---|
| ep0 | -5.7331 | 0.01806 | 0.3138 | 0.0905 | +0.0195 | +0.1272 | **+0.2246** | 0.7594 |
| ep1 | -5.9792 | 0.01936 | 0.1754 | 0.1166 | -0.3288 | -0.1108 | **-0.1185** | 0.3557 |
| ep2 | -5.9721 | 0.02078 | 0.1905 | 0.1139 | -0.2300 | -0.1110 | **-0.0831** | 0.3260 |
| ep3 | -5.9888 | 0.02254 | 0.1903 | 0.1183 | -0.0792 | -0.1873 | **-0.1418** | 0.3141 |
| ep4 | -5.9863 | 0.02593 | 0.1203 | 0.1288 | -0.2374 | -0.2877 | **-0.2792** | 0.2837 |

**16. The fork point for the three update-rule arms, label-free** (D4'/D5(b) step 1). The standing
selection rule (dev reward = recon + 1.0 * lm_prior from the arm's own `learning_rates`) combined
with a health screen computed on the arm's own dev hypotheses and nothing else: words per utterance,
and the ABSOLUTE count of the minimal-state class {and, but, i}, each required to sit within a
pre-registered 10 % of the window minimum over the four sub-epochs that existed at fork time. The two
dev sets are pooled (5567 utterances, fixed across sub-epochs); no WER enters the job, and the fork
epoch is a config constant the job asserts against. `S/fork_screen/ForkPointScreenJob.avOkAB1TUN3d`;
fork parent `T/ReturnnTrainingJob.vhyvv2waeU16`, sub-ep 2 = `output/models/epoch.002.pt`.

| ep | dev reward | words/utt | d_len | and | but | i | min-state | d_cls | screen | dev WER (confirm) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -3.28110 | 18.920 | +0.00 % | 3596 | 646 | 1466 | 5708 | +0.00 % | pass | 6.28 / 10.25 |
| 2 | -3.25726 | 19.049 | +0.68 % | 3623 | 645 | 1496 | 5764 | +0.98 % | **pass** | **5.34 / 9.50** |
| 3 | **-3.24568** | 19.320 | +2.11 % | 4239 | 1179 | 2128 | 7546 | +32.20 % | VETO | 6.56 / 11.15 |
| 4 | -3.25140 | 19.357 | +2.31 % | 4698 | 1380 | 1826 | 7904 | +38.47 % | VETO | 6.89 / -- |

**17. The joint-psi control arm on the best bed** (D5(b)-b). One knob off the fork checkpoint:
`train_psi=True`, so psi's per-frame NLL on all G sampled texts joins the shared optimizer at ce
scale 1.0 with no in-loop contrastive term, everything else the parent's (shaped lam_lm 1.0
units-normed, T=0.7, `partition_epoch` 10, `keep_epochs` all). The learning rate continues the
parent's cosine rather than restarting it (`epoch_offset` evaluates the parent's curve at epoch + 2
with the parent's 10-epoch span, reproducing its ep3-8 values exactly) and the arm runs 6 sub-epochs,
then stops regardless of trajectory. Forensics are instrumented during the run, one row per
sub-epoch plus the fork as ep0. The joint backward does not fit at the parent's batching, so this arm
runs `batch_size` 1e6 / `accum_grad_multiple_step` 2 against the parent's 2e6 / 1: the 2e6-frame
effective batch and the updates per sub-epoch are preserved and `max_seqs`, `group_size`,
`max_seq_length`, schedule and `partition_epoch` are untouched, but the per-update gradient is the
mean of two half-batches rather than one full batch. Arm `T/ReturnnTrainingJob.jQmmGy2yGtGR`
(first launch `.eYhb6alu9OIQ` OOMed, superseded); WERs `ScliteJob.{onJeeX0UOiRy,RYa3OTRBO2Uf}` /
`.{1qm9kIUcj2y6,49zgvrMKwznh}`.

| sub-ep | arm | dev-clean / dev-other | insertions dc / do | substitutions dc / do |
|---|---|---|---|---|
| 1 | joint | **5.12 / 9.27** | 385 / 630 | 2029 / 3641 |
| 1 | frozen control (parent sub-ep 3) | 6.56 / 11.15 | — | — |
| 2 | joint | 17.35 / 21.97 | 6114 / 6450 | 2952 / 4426 |
| 2 | frozen control (parent sub-ep 4) | 6.89 / 11.31 | — | — |

The frozen arm's insertions over its whole post-peak stretch are 1182-1415 dc / 1592-1794 do
(`SAE_0d.md` c13), which is the band both joint rows are read against. Sub-ep 3 completed at
**41.8 / 50.9** (`ScliteJob.yVyM2WLvkXxG` / `.4mnMvy9mUVI7`, epoch 3 via
`ExtractAvSubmodelJob.a1d9LlyUDSED`), with dev-other insertions 21,406; the psi forensics give
gold-pair ce_loo 2.6343 / 2.7928 / 2.9771 at sub-eps 1/2/3 (`PsiHeldNllJob.LTg9xnjtl8Zs` /
`.SFOP6DaI3Zpv` / `.vJnzFU0eRSyl`) against own-decode ce_loo 2.6270 / 2.4726 / 2.2994
(`.8DHdEHY7HZ2b` / `.2WmXVQYlCjnF` / `.uEc3jigALnmE`) -- the scorer's fit to gold degrades while its
fit to its own output improves, monotonically, over the three sub-epochs.

**18. Refresh round 1 on the best bed** (D4', steps 2-3). The rollout source is the fork policy
itself at the loop's own operating point (T=0.7, G=12) over all of tc100
(`F/ReturnnForwardJobV2.QbIYruVEI0fF`), ranked by this bed's own psi rather than by a token-LM AR,
which does not exist here. The suspect set is re-derived on this bed instead of carried over
(`S/scorer_diag/SuspectVocabJob.UG1VLQjflE7G`): excess mass of the fork policy's own dev decodes
against the same LM corpus at D0's pre-registered `min_excess` 0.002. The minimal-state class
{and, but, i} stays a MONITOR (`S/scorer_diag/FillerWatchJob.3x3IRoxcQSha`) and never enters the
curation views, because it was found by counting insertions against references. The label-free
derivation returns an EMPTY set here (largest excess "and" 0.00135; 0.001 admits that word alone,
0.005 none), so the table below is the incumbent battery's instead: the round-0 gold psi
(`S/psi_align_jobs/PsiAlignTrainJob.IN3zmmGpH4Bv`) on this bed, 1443 of 1500 frozen gold seed-dev
pairs, ce_loo 2.7560 on the untouched text against a unit marginal of 6.0332, held ce_loo 2.7614.
The frozen 1500-pair gold seed-dev set is
`S/psi_forensics/HfSplitTextJob.hITA2tWgTklY` -> `S/scorer_diag/FrozenHeldPairsJob`. Battery
`S/psi_align_jobs/PsiHeldNllJob.yMQGlcL3OVVj`, `.PsiTextProbeJob.pBrTx11FPZvS`,
`.PsiAlignRerankJob.pJONTykQhQaS`, `.PsiScorerParityJob.gRkOlabxfLVY`; selector admissibility
`S/scorer_diag/RolloutMechanismJob.UJ0DfPXTH8Cq`.

| edit at k=1, delta ce_loo (>0 = the scorer charges) | minimal-state word | frequency-matched LM word | discount (LM - minimal-state), 95% CI |
|---|---|---|---|
| INSERTION | +0.0693 | +0.0902 | **+0.0209** [+0.0164, +0.0250] |
| substitution | +0.4051 | +0.3892 | -0.0159 [-0.0228, -0.0091] |
| deletion (no filler twin) | +0.4295 | -- | -- |
| position-matched substitution | +0.4051 | +0.3913 | -0.0138 |
| ladder monotonicity (k = 0,1,2,4) | ins 0.658 | sub 0.735-0.784 | -- |
| suspect-state alignment mass (gate v2 iii) | 2.6838 % of 497 767 frames | -- | -- |

**19. Round 1 refits on the whole pool, selection removed** (D4', steps 4-5). No admissible curation
view exists on this bed (approach 18, verdict 30), so the planner amended round 1 to refit on the
anchor plus EVERY greedy decode instead of a selected subset: 2 849 gold seed pairs repeated 11x
beside 28 539 one-per-utterance argmax decodes of the fork policy, 59 878 rows at a 52.3 % anchor
share, split by utterance so a repeated anchor cannot land on both sides of the internal held-out.
The recipe is the incumbent gold-psi one with the D2-winner contrastive term on; the greedy rows come
from the existing dump rather than a second forward job. Batching is widened 8x (`max_batch` 32 ->
256, cells 3e6 -> 24e6) after measuring the DP to be launch-bound rather than FLOP-bound, which is a
SECOND difference from the loop's frozen scorer and is recorded as a confound, not absorbed. Corpus
`S/curate/UncuratedPoolJob.1RgS3KEtkdEy`, refit `S/psi_align_jobs/PsiAlignTrainJob.Be8yVs7MaLrS`,
clause table `S/gate_table/PsiGateClauseTableJob.hRgVjm5bYRKI`, re-rank `.PsiAlignRerankJob.DU9JY7WG9b0y`.

| statistic | set | psi0_gold (ep 9) | round-1 uncurated (ep 30) |
|---|---|---|---|
| held ce_loo | frozen gold seed-dev, 1 493 pairs | 2.7614 | **2.6432** |
| text_explained_loo (gate v2 ii) | same | +3.2718 | +3.3900 |
| usage gate, length-matched derangement | same | +4.6548 | +5.2952 |
| unit marginal on those frames | same | 6.0332 | 6.0332 |
| within-group spearman | fork dump, 28 538 groups at T=0.7 | +0.3399 | **+0.3621** |
| eta | same | +0.2599 | +0.2663 |
| internal held NLL/frame at the pin | own corpus, not cross-comparable | -- | 2.8122 |

**20. D6 -- structural insertion repair, three rungs on one corpus.** Insertion is under-priced by
the topology, not by the vocabulary (verdicts 27-28), so the three rungs go at the arcs: (1) OFFLINE
PRICE STEERING re-scores the frozen incumbent on the same corruption draw the D1 battery uses,
sweeping a renormalized bias against the skip arc and a minimum-duration charge of `dur_cost` nats
per frame a state falls short of `d_min` (silence exempt, carried exactly in the DP as a frames-held
axis; `S/psi_align_jobs/PsiPriceSteerJob.4Eqth3bY2Zc2`); (2) CORRUPTION-TRAINED ARC PRICES refits
with a hinge demanding an inserted LM-drawn word cost at least half of what deleting a word costs on
the SAME row -- scale-free, deletion side detached so the term can only make insertion dearer
(`S/psi_align_jobs/PsiAlignTrainJob.zjUitbvGbDg3`); (3) MIN-DURATION TOPOLOGY refits with every
content symbol split into `d_min` states and the skip arc masked wherever it would cross one, which
lives in the model config so a checkpoint carries its own topology (`.wlruSpBK1EDP`; the CUDA
forward-backward twin is `.QhaW4lUpbkl6`, rungs 2+3 combined `.HVjMgYBlJ4tp`). Rungs 2-3 train on
the IDENTICAL corpus as approach 19 and are read against it; rung 3 waits on rung 1's feasibility
statistic. Clause table `S/gate_table/PsiGateClauseTableJob.JdrWdaCm7UeG`. Bars in Gates.

| arm (all on the round-1 corpus) | held ce_loo | ins_1 | del_1 | ins/del | mono(ins) | matched ins discount k1 | spearman | eta |
|---|---|---|---|---|---|---|---|---|
| psi0_gold, round-0 incumbent | 2.7614 | +0.0693 | +0.4295 | 0.161 | 0.658 | +0.0078 | +0.3399 | +0.2599 |
| r1_uncurated, the comparator | 2.6432 | +0.0763 | +0.4948 | 0.154 | 0.760 | +0.0094 | +0.3621 | +0.2663 |
| rung 2, corruption margin | 2.5943 | +0.1675 | +0.5454 | 0.307 | 0.692 | +0.0578 | +0.4016 | +0.3482 |
| rung 3, min-duration d_min=2 | **2.1620** | **+0.1985** | +0.5759 | 0.345 | **0.853** | +0.0141 | **+0.4357** | +0.3296 |
| rungs 2+3 combined | 2.1768 | +0.4964 | +0.7106 | 0.699 | 0.785 | +0.3150 | +0.4441 | +0.3392 |

`mono(ins)` is the `filler_ins` ladder's monotone fraction, the clause (iv) statistic (the sub/del
band runs 0.71-0.80 across these arms). Clause (i)'s picked-WER half, missing from the table above
and read from the rerank jobs: sel_wer d6_mindur 0.05028 / combined 0.05015 / margin 0.05097 /
r1_uncurated 0.05219 / psi0_gold 0.05228 -- d_min=2 PASSES that half. Caveat: the min-duration arms
score 28531 of 28538 groups (7 unscorable under the topology), so their random/oracle baselines
differ slightly (0.05477/0.04116 vs 0.05613/0.04133) and cross-arm sel_wer is not perfectly paired;
ordering unaffected.

**21. D6 swap-in -- the min-duration scorer as the live reward, on both beds.** The rung-3 checkpoint
replaces the incumbent psi in the reward and nothing else moves. (a) BEST BED: the fork policy
continues from the same sub-epoch 2 state on the parent's own cosine tail for the remaining 8
sub-epochs, same shaped reward at T=0.7, same 960 h slices, same batching -- so the free frozen
continuation that already ran those sub-epochs with the incumbent psi is the control at matched
points. Read as dev WER plus both arms' sclite error decomposition in absolute insertion counts;
pre-registered confirmation in Gates. (b) G-TRACK: the topology transfers, checkpoints do not, so the
G-track round-1 refresh recipe is refit at `min_dur=2` on that bed's own round-1 curated corpus
(`S/psi_align_jobs/PsiAlignTrainJob.TicugJYx52p2`, best epoch 23), single-variable against the
`min_dur=1` refit (`.cRIigmxPtt75`), and read with the four D6 clauses on G-track instruments
(`S/gate_table/PsiGateClauseTableJob.qYRE7JWyUcJQ` vs `r1`, `.H9QbX4VgXAwf` vs `psi_g_tc100`;
re-rank `.PsiAlignRerankJob.BRfnFlMK1job`, probe `.PsiTextProbeJob.mSJzvpTBW0Y3`). Both arms there
gain the G3 re-rank and the online/offline parity check the G-track round 1 never carried.
Arm (a): `T/ReturnnTrainingJob.YUh6Gzvavctf` (last three sub-epochs at half micro-batch,
`.qQeSijpUKP2k`); control `T/ReturnnTrainingJob.vhyvv2waeU16`; anatomy
`S/scorer_diag/PolicyAnatomyJob.pxqfrYx23Rth`.

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
21a's whole gain landed in its first post-swap sub-epoch, so this arm re-forks from the same parent
sub-epoch-2 checkpoint and repeats one unit at every boundary from 3->4 on: decode the tc100 refresh
corpus with the round-1 recipe unchanged (gold anchor at its 50 % floor plus one greedy decode per
utterance, only the decoding checkpoint varying), refit `d_min=2` from scratch on the CUDA path, read
the four pre-registered clauses against the last ACCEPTED scorer on the standing frozen instruments,
and swap on pass or keep on fail. Everything else is 21a point for point -- same fork, same cosine
tail at the parent's own epoch index, same shaped reward at T=0.7, same 960 h bed at the parent's
partition size, and the control's own two batching regimes (2e6 through parent sub-epoch 7, then 1e6
with `accum` 2) -- so 21a is the control for free and the scorer's recency is the only variable. Two
differences it cannot avoid, both forced by one sisyphus job per sub-epoch: the bed partition moves
into the graph as round-robin shards, and Adam restarts at every boundary against the control's twice.
Legs `T/ReturnnTrainingJob.5FqdnhWTOf1f`, `.BTnU1gSuMG0i`, `.ZKCbq529Hgp8`, `.gFNpNmXwvrsc`,
`.nQtnPdKCuJ0m`, `.n8abYvLR4IP5`, `.jGj7TTbW5DTm`, `.wWqYY7iOCw1s`; per-boundary refits
`S/psi_align_jobs/PsiAlignTrainJob.JWV3InILYF5v`, `.yUUSN2Hx96E0`, `.QMO8VcAtZ6Gi`, `.DzhBWCy61tiN`,
`.Vha8vvKu9lWk`, `.RGTtwlQHt3HY`, `.Ls0TQGiyhQbf`.

The acceptance clauses below are the arm AS FIRST RUN; the verdicts are recorded here because the
user removed the acceptance step on 2026-08-18 and those jobs were deleted, so this table is the only
surviving record of what it decided (planner-verified against the pre-deletion artifacts, clause for
clause).

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
segments of one trajectory on a decaying schedule (leg k trains from leg k-1's checkpoint), so the
across-leg range conflates schedule evolution with noise. Matched-point absolute differences against
the one-shot arm: dev-clean 0.29 / 0.03 / 0.32 / 0.36 / 0.19 and dev-other 0.24 / 0.30 / 0.32 / 1.30
/ 0.11, i.e. a maximum of 0.36 / 1.30 and a median of 0.29 / 0.30 over five paired points. A single
matched-point claim on this bed has to clear the maximum; a consistent-sign difference over four or
more matched points reads against the median.

The user then removed the scorer-statistic gate and relaunched. The two primary 10 h-init anchors are
separated from the trajectory; "best" selects the checkpoint with lowest dev-other WER and reports
its paired dev-clean value:

| 10 h-init anchor | dev-clean / dev-other | operating point |
|---|---|---|
| AV SFT, no loop: adapted-donor theta_0' | 11.43 / 15.54 | 10 h AV SFT, epoch 50 |
| best previous frozen-scorer loop | **4.68 / 8.64** | D6 one-shot d_min=2 scorer swap, scorer then frozen, global sub-epoch 3 |

The older incumbent-scorer loop's 5.34 / 9.50 fork is retained as a secondary historical anchor, but
it is not the best previous frozen-loop result.

| ungated leg / global sub-ep | fresh periodic dev-clean / dev-other | one-shot frozen scorer | frozen control | dev-other S / D / I, fresh |
|---|---|---|---|---|
| 1 / 3 | 4.97 / 8.88 | 4.68 / 8.64 | 6.56 / 11.15 | 3572 / 471 / 479 |
| 2 / 4 | 4.65 / 9.02 | 4.61 / 8.98 | 6.89 / 11.31 | 3593 / 395 / 607 |
| 3 / 5 | 5.28 / 9.27 | 4.61 / 9.03 | 6.32 / 11.03 | 3551 / 561 / 610 |
| 4 / 6 | 6.05 / 10.56 | 5.01 / 9.51 | 6.54 / 11.16 | 3553 / 476 / 1350 |
| 5 / 7 | 7.42 / 12.68 | 4.70 / 9.12 | 6.69 / 11.03 | 3597 / 376 / 2489 |

These are a prefix, not an endpoint: leg 6 was submitted but pending for maintenance and legs 7-8 did
not exist at the last read. `S / D / I` are sclite substitution, deletion and insertion counts on
dev-other; the late WER loss is almost entirely insertion growth, not a drift in substitutions.

**23. HOM-0a -- how much of the pseudo-label corpus a homophone substitution could reach.** The
D6-PERIODIC/GAN+HOM arm resamples homophone spellings in the init's SFT targets, so the admission
read asks whether enough corpus mass sits in a homophone class to be worth funding, against the
pre-registered 5 % floor. A class is one distinct FULL pronunciation set over the 39-ARPAbet lexicon,
members must reach 1e-5 of LM tokens (8,033 occurrences) and two characters, and the in-class draw is
uniform; the read is label-free, on the §1d student's own word decode with the lexicon as allowed
prior knowledge. `S/homophone/HomophoneClassStatsJob.our76yheSD0c` (`classes.json` carries every class
with per-member LM and corpus counts); augmented corpus `S/homophone/HomophoneAugmentJob.k2OwZiTcKpEG`.

| quantity | value |
|---|---|
| classes (>= 2 members, after filtering) | 142 |
| class sizes seen in corpus | 131 of size 2, 8 of size 3 |
| corpus tokens | 963,857 |
| in a multi-member class | 74,037 = **7.68 %** |
| funding floor | 5 % -- **PASS** |
| share a uniform draw actually rewrites | 4.02 % |
| top 8 classes' share of all rewrites | 60.5 % |

From the ratified class list: corpus-zero members carrying dominant LM mass (by, sea, right, side,
air, fair, they're) total ~0.5-0.6 % of corpus tokens, so the repair channel is real and small;
`they're` = 0 is a decoder commitment, not an alphabet artifact. `in`/`inn` alone carries 19 % of
rewrites.

**24. D6-PERIODIC-WARM -- the same per-boundary refit, CONTINUED from the previous round's scorer.**
Approach 22's refits each discarded the previous scorer and re-fit from the random initialization at a
fixed seed; this arm changes that one argument and nothing else -- same fork, same cosine offsets,
same shard rule (it calls approach 22's own `train_bed`, so leg k trains on the identical utterances),
same refresh corpus, pool recipe, reward, batching regimes and four-clause gate. Leg 1 precedes the
first warm start and is therefore approach 22's own finished leg, shared by hash, so the arms are
identical through parent sub-epoch 3. Relaunched 2026-08-18 with no acceptance step, as the sibling
was. The one verdict the gated run produced is banked here because its job was deleted: the
warm-started round-2 candidate was REJECTED under the binding confidence-interval reading -- (i) pass,
(ii) pass, (iii) point fail / CI PASS, (iv) point fail / CI FAIL -- failing the corruption ladder
(worse on filler substitution and on LM substitution) while passing the insertion price that was the
sibling's habitual failure, so a continuation of the incumbent did NOT find the clause table easier as
registered (`S/refresh_gate/PsiRefreshAcceptJob.uXG53BObiW55`, read off disk before deletion). Legs `T/ReturnnTrainingJob.5FqdnhWTOf1f`, `.OOr3UybqUEHD`, `.X3biCvDKgQ7N`, `.7dANeLqxFFbq`,
`.nd92xaRDY0uw`, `.kkh0u4rI7I6D`, `.kQRZtXc1ubTV`, `.oRbUsmYR6fRT`; warm-started refits
`S/psi_align_jobs/PsiAlignTrainJob.2TDm8VwIZzjv`, `.frtMcQ6wvR4s`, `.ENcr81sGwHfp`, `.3tMeo1Meuceg`,
`.ZeEsJq6JOdNx`, `.34mTYfJioAsm`, `.3JLOhu5PSKwj`. Warm-start mechanics were verified to load the
state dict before the [UNK] unigram pin, with inventory and six topology keys asserted against the
checkpoint (warm held NLL 1.1869 vs cold 2.2609 after one epoch, source best 1.4341).

| leg | fresh periodic dev-clean / dev-other | warm periodic dev-clean / dev-other | dev-other S / D / I, warm |
|---|---|---|---|
| 1 | 4.97 / 8.88 | 4.97 / 8.88 | 3572 / 471 / 479 |
| 2 | 4.65 / 9.02 | 5.07 / 9.19 | 3576 / 456 / 680 |
| 3 | 5.28 / 9.27 | 4.85 / 9.04 | 3570 / 386 / 649 |
| 4 | 6.05 / 10.56 | 6.39 / 11.19 | 3593 / 522 / 1593 |
| 5 | 7.42 / 12.68 | 12.18 / 19.33 | 3685 / 441 / 5874 |

Warm inheritance is inside the fresh arm's range through leg 3, then separates in the harmful
direction; the leg-5 gap is +4.76 / +6.65 WER and is an insertion explosion. Leg 6 was submitted but
pending for maintenance, so this is not the final registered read.

**25. HOM-0b and HOM-0c -- whether the reward can act on a spelling, and whether sampling already
varies one.** 0b takes the label-free arm's own round-1 samples at T=0.7, substitutes ONE in-class
spelling per variant leaving the rest of the text untouched, and re-scores both reward terms under
that arm's round-1 refit and the same language model the loop's prior reads, at the arm's own weight
(lam_lm 1.0) and per-unit-frame normalization -- so the pre-registered bar is a direct comparison; the
swaps split into the repair direction (a spelling the refit corpus never contained) and the diversity
direction (both attested), and the sign of delta recon is read against the change in spelling length.
The prior column is anchored rather than trusted: rows whose text does not re-tokenize to the length
the loop scored are dropped first, and the recomputed column has to reproduce the banked one or the
job fails. 0c counts how often the init's sampled groups already hold two spellings of one class, on
the label-free init's full-bed G=12 dump at the same temperature (the round-1 dump has
DUMP_GROUP_SIZE=1 and cannot express within-group coverage at all -- an artifact substitution ratified
as a frame repair). Measured: 0b on 8,000 utterances (25,541 of 28,539 sampled texts round-trip
through the tokenizer exactly, 89.5 %) giving 23,085 swaps with 5,162 dropped by a 4-per-text cap, the
prior column reproducing the dump's own to a median 0.0053 nats/token against a 0.05 bar; 0c on all
28,539 groups of 12, of which 26,584 are homophone-bearing, 6,228 = 23.43 % already hold two spellings
of one class, and 217 = 0.82 % ever contain a spelling absent from the scorer's own training corpus.
`S/homophone_probe/HomophoneSwapScoreJob.gN7mZ0EcPhsS`, read
`S/homophone_probe/HomophoneSensitivityJob.xB5RvcgLVgtD`, coverage
`S/homophone_probe/HomophoneCoverageJob.F76iJ8j0AQi1` on `F/ReturnnForwardJobV2.lQMOR5n2ntcS`.

| medians over 23,085 single-word swaps, per unit frame at lam_lm 1.0 | abs delta lm_prior | abs delta recon | ratio |
|---|---|---|---|
| all swaps | 0.0134 | 0.0106 | **1.26** |
| diversity (both spellings attested in the refit corpus) | 0.0134 | 0.0105 | **1.27** |
| repair (into a spelling the refit corpus never contained) | 0.0159 | 0.0194 | **0.82** |

Diversity n=22,584, repair n=501. The SIGNED medians are negative for both terms in both directions:
even `lm_prior` penalizes repair swaps on median. HOM-0b is reference-BLIND by construction (its
verdict compares absolute movement), which is what approach 28 joins back.

**26. D6-PERIODIC/GAN -- the same per-boundary refit on the label-free init** (launched 2026-08-17).
Approach 22's refresh unit with theta_0^G in place of the gold-seeded fork, on the same 960 h bed,
same shard rule, same shaped reward at T=0.7, same `d_min=2` topology -- with the two parts that read
gold text dropped rather than ported, so the pool is an anchor-free greedy decode and the refit's own
model goes straight to the next leg with no acceptance gate at all. Leg k sits at the schedule
position the two held frozen-scorer arms occupied at their sub-epoch k. Legs
`T/ReturnnTrainingJob.kr1foUV6lecx`, `.AuzMGgyskdJT`, `.KD73Hc4eGDfW`, `.E6s3lUUaodzw`, `.J9m38fxEwXl4`,
`.AS1g33qDo28i`, `.QTQuYQnppmSs`, `.cR8Q29Pmfuhy`; refits `S/psi_align_jobs/PsiAlignTrainJob.dsMKgPHQApyR`,
`.7jHYVGToyWPR`, `.M2Z0M9UpKW98`, `.rdkbJsLOLEJW`, `.YPyCrmgjglsj`, `.jMaYmBUAffMb`, `.NM6sQa0D9uQM`,
`.wPujQSh4PLSd`; init `T/ReturnnTrainingJob.2fb02hGUdHNj`.

| GAN-init anchor | dev-clean / dev-other | operating point |
|---|---|---|
| AV SFT, no loop: theta_0^G | 13.89 / 18.34 | pseudo-label AV SFT, epoch 10 |
| best previous frozen-scorer loop (reference, not schedule-only control) | **12.68 / 17.57** | shaped arm, repaired d2_contrast scorer frozen, sub-epoch 2 |

The frozen contaminated-scorer arm below is a diagnostic control, not the best previous frozen-loop
result, and the best frozen row is NOT a single-variable control for periodic: both start from
theta_0^G and match bed, reward, T and nominal cosine position, but the frozen row uses one d_min=1
d2_contrast scorer under the D2 recipe in one continuous multi-sub-epoch training job, while periodic
fits d_min=2 from scratch on each policy's anchor-free greedy pool and runs one training job per leg,
restarting Adam. Isolating scorer schedule requires the periodic graph with its own round-1 d_min=2
scorer held fixed across otherwise identical legs; that arm does not exist. The d_min=1 setting was
historical, not a winning hyperparameter: D2 landed 2026-08-07 with topology intentionally identical
to psi_g_tc100 and `PsiAlignTrainJob` had no `min_dur` interface then; D6 added the minimum-duration
topology 2026-08-11; D3 froze the already-finished D2 winner and inherited d_min=1. It never compared
d_min=1 against d_min=2, so topology is a standing confound in every frozen-versus-periodic contrast,
beside corpus and Adam continuity.

| dev-clean / dev-other, plain WER as scored | sub-ep 1 | sub-ep 2 | sub-ep 3 | sub-ep 4 | sub-ep 5 | sub-ep 6 |
|---|---|---|---|---|---|---|
| frozen contaminated psi_align^G, `shaped` (held) | 13.42 / 18.75 | 13.91 / 18.91 | 13.49 / 18.81 | 17.99 / 23.33 | -- | -- |
| frozen repaired scorer, `shaped` (held) | 13.57 / 19.69 | 12.68 / 17.57 | 13.54 / 18.56 | -- | -- | -- |
| refit at every boundary (this arm) | 14.45 / 19.69 | 12.85 / 17.89 | 13.20 / 18.20 | 17.76 / 23.17 | 17.92 / 23.27 | 18.38 / 24.01 |

Only sub-epoch 2 improves the no-loop init's 18.34 dev-other, by 0.45; the later loss is mainly
substitutions (4,110 at sub-epoch 2 to 7,331 at sub-epoch 6), not insertions. The GAN+HOM variant
changes the policy initialization through homophone-resampled SFT and then runs the same loop with
its own downstream refits (legs `T/ReturnnTrainingJob.JocWKAmYroFJ`, `.dp0XmU5Mm9V5`, `.tpby6E3kTeSE`,
`.JBaqJExxDKGz`):

| dev-clean / dev-other, plain WER as scored | init | loop leg 1 | loop leg 2 | loop leg 3 |
|---|---|---|---|---|
| plain GAN init / periodic | 13.89 / 18.34 | 14.45 / 19.69 | 12.85 / 17.89 | 13.20 / 18.20 |
| GAN+HOM init / periodic | 16.67 / 21.45 | 14.84 / 19.99 | 13.94 / 18.77 | 12.80 / 18.08 |
| class-internal substitutions, GAN+HOM dev-other | 1827 | 130 | 110 | 105 |

The hom arm loses at legs 1-2 but catches the plain trajectory at leg 3, while removing nearly all
augmentation-specific class-internal substitutions in its first leg. The PLAIN arm's own eight-leg
trajectory is in approach 36 -- legs 4-8 are far worse than legs 2-3, so the three legs above are its
best three and not a representative sample. HOM leg 1 also carries a different psi_checkpoint
(`ACP3LqKDUSQ0` vs `dsMKgPHQApyR`) as well as a different init, since its refit is downstream of its
own decodes, so the A/B cannot be stated as a single differing input.

**27. theta_0^G_hom -- the homophone arm's policy init** (launched 2026-08-18 on the user's
greenlight, after HOM-0b admitted the arm). theta_0^G's own builder with the resampled pseudo-label
corpus as targets and every other argument shared (config diff moves four lines: three dataset dirs
plus the model path); 10 epochs, last-epoch pin, no dev-WER selection (learning_rate_control constant,
`keep_best_n` ranks pseudo-label dev CE, ep10 = num_epochs and is also each arm's best scored epoch).
`T/ReturnnTrainingJob.EabxlDlT0oji` on `TransformAndMapHuggingFaceDatasetJob.157IDJgBOv9H`; pinned-epoch
dev-other scores `ScliteJob.4xgsEBkQtPsg` (plain) and `.KKjjg7A3vT52` (hom). A parallel NON-SCLITE
scorer exists on this arm (`JoinRobustMetricsJob.6il1r3BMTMEj`, normalized WER_clean / WER_cap
columns) whose numbers must NEVER be quoted -- plain sclite only, standing rule.

| dev-clean / dev-other, plain WER as scored | ep 2 | ep 4 | ep 6 | ep 8 | ep 10 (pinned) |
|---|---|---|---|---|---|
| theta_0^G (plain corpus) | 175.25 / 180.54 | 28.27 / 33.04 | 14.46 / 19.09 | 13.91 / 18.74 | 13.89 / 18.34 |
| theta_0^G_hom (resampled corpus) | 226.53 / 217.88 | 20.57 / 24.06 | 17.25 / 22.37 | 16.84 / 21.45 | 16.67 / 21.45 |

ep2 was scored for both arms (degenerate, above 100 % WER) and is omitted from the curve above. The
homophone arm's dev-other does not move between ep 8 and ep 10, so the 3.11 gap at the pin is carried
by the plain arm's own late gain.

Where the extra errors sit, at the pinned epoch on dev-other (registered class-internal substitution
read; gold, reported only, selecting nothing; independently recomputed from both arms' `sclite.pra`):
the homophone init makes 1,587 more errors NET than the plain one, and 1,534 of that net -- 96.7 % --
are substitutions WITHIN a homophone class (of extra SUBSTITUTIONS alone the share is 92.2 %: 130
non-class substitutions were also added, offset by 20 fewer deletions and 57 fewer insertions).
Class-internal substitutions are 25.2 % of all its substitutions against 5.2 % of the plain arm's --
and that 5.2 % baseline is 65 % one pair, `by -> buy`, 190 of 293. Its top confusions after the shared
`with -> of` are `in -> inn` (329), `not -> knot` (155), `be -> bee` (155), `by -> buy` (91),
`no -> know` (81). Per dev-other REFERENCE token the class-internal substitution rate is 3.59 %
against the plain arm's 0.58 %; over class-bearing reference tokens only (4,461), the same counts read
40.96 % against 6.57 %. The like-for-like expectation is 4.58 % of reference tokens (2,331
substitutions) if the SFT reproduced the uniform draw in full, so the realized 3.59 % is 78 % of it --
the policy under-reproduces the draw by about a fifth. The damage also SPREADS: 82 distinct classes
against the plain init's 33. Not an artifact of the arm's 292-word filtered class list -- recounted
against the full pronunciation lexicon's 52,969 in-class words the same two decodes read 12.87 % and
31.38 % of substitutions. Outside the classes the two inits are within noise: +53 errors, paired
bootstrap CI [-70, +173], against a total-error CI of [+1,438, +1,735].

**28. Which SPELLING the reward points at** (`S/homophone_probe/HomophoneDirectionJob.Uo4UAJp5Ue42`,
on HOM-0b's own 23,085 swaps and the round-1 dump's reference rows; the round-1
artifacts both HOM reads run on are dump `F/ReturnnForwardJobV2.66pIzBzffnK2`, refit corpus
`S/curate/GreedyPoolJob.Yv6qBpz0UC0U`, scorer `S/psi_align_jobs/PsiAlignTrainJob.dsMKgPHQApyR`).
HOM-0b's bar compares the two
terms' absolute movement, so it cannot separate a term that swings toward the right spelling from one
that swings toward the wrong one; the reference text sits unused in the same dump the swaps were built
from. This joins it back and reports, per direction, the share of swaps each term PREFERS.
Position-aligned is the primary read; bag-of-words is reported beside it and agrees (n=1550, lm_prior
0.895, recon 0.182, composed 0.532). Gold read: reports only, selects nothing. Measured on the PLAIN
arm's round-1 dump, i.e. on a policy without the augmentation.

| share of swaps the term prefers | n | reconstruction | language-model prior | composed, lam_lm 1.0 |
|---|---|---|---|---|
| TOWARD the reference spelling | 1421 | 0.179 | 0.906 | 0.529 |
| AWAY from the reference spelling | 19328 | 0.254 | 0.016 | 0.063 |

The composed column is concentrated in one class: `buy`/`by`/`bye` supplies 961 of the 1,421
toward-reference swaps (67.6 %) and reads 0.446, while the remaining 460 read 0.702. Per class,
`air`/`ere`/`heir` 1.000, `side`/`sighed` 0.949, `knew`/`new` 0.884, `sea`/`see` 0.800,
`right`/`write` 0.726, `their`/`there`/`they're` 0.588, `war`/`wore` 0.333 (n=15). Splitting by
whether the reference spelling is one the scorer's refit corpus holds does NOT explain it: only 24
toward-reference swaps are the repair direction at all, and the attested-spelling subset still reads
0.534. WEIGHTING DEFECT, quantified against the artifact: this measurement's per-class share vector
overlaps the PLAIN arm's dev-other class-internal profile at total variation 0.880 and the HOM arm's
at 0.197 -- near-orthogonal to the distribution it was being used to predict; 78.8 % of the hom arm's
damage sits outside the eight classes the job reports and 31.1 % in classes with ZERO toward-reference
swaps here. The AWAY cell (n=19,328) covers 99.7 % of the hom damage mass and sign-reversed reads
composed 0.900, so the plain dump BRACKETS the hom arm's per-swap edge between about 0.51 and 0.90 --
uninformative for the funding question, which is why approach 29 was run on the arm's own dump.

**29. The same two reads on the arm's OWN dump and OWN scorer**
(`S/homophone_probe/HomophoneDirectionJob.deNc7xXnCfSu` and `.HomophoneScorerDeltaJob.JKbbRWimojlI`,
on theta_0^G_hom's round-1 dump; refit `S/psi_align_jobs/PsiAlignTrainJob.ACP3LqKDUSQ0`, swaps under
own/plain scorer `S/homophone_probe/HomophoneSwapScoreJob.IG6wFl5QWnld` / `.iRCxGqNRxQha`). Here the
swaps come from the policy whose errors the loop must actually repair, and the scorer is this arm's own
round-1 refit; the second job holds dump and swaps fixed and moves ONLY the scorer.

| share of swaps the term prefers, position-aligned | n | reconstruction | language-model prior | composed |
|---|---|---|---|---|
| TOWARD the reference spelling | 8806 | 0.357 | 0.970 | 0.825 |
| AWAY from the reference spelling | 10959 | 0.559 | 0.030 | 0.140 |

Coverage is now the damage distribution: `in`/`inn` n=1755 (composed 0.833),
`their`/`there`/`they're` 819 (0.896), `knot`/`not` 807 (0.927), `buy`/`by`/`bye` 700 (0.661),
`be`/`bee` 632 (0.728), `know`/`no` 517 (0.660), `wood`/`would` 368 (0.948), `too`/`two` 273 (0.905)
-- every class above chance. Read beside the audio-free null, as the standing principle requires: the
language-model prior ALONE reads 0.9701 on the same swaps, so the composed 0.8255 means adding the
audio-grounded term COSTS 14.5 points of reference accuracy; homophone class members are acoustically
identical by construction, so this is expected rather than a defect, but the headline is the prior's
number and is not quotable without it. The scorer contrast, same swaps: the reconstruction term's
toward-reference rate is 0.357 under this arm's own refit against 0.684 under the plain arm's
(-0.327); at the OPERATING POINT (composed reward) the same swaps read 0.825 against 0.895, so
entrenchment costs -0.069 in the deployed reward. Paired per swap 2547 both / 598 own only / 3480 plain
only; of 121 classes 77 move down, 18 up, 26 tie, median per-class delta -0.172. Length-matched, the
entrenchment survives at EQUAL character count, -0.284 on n=1889 (own 0.469 against plain 0.752),
beside -0.388 shortening and -0.137 lengthening, so it is not a length price: the plain scorer holds
real spelling discrimination at equal length (0.752) and the arm's own refit collapses it to near
chance (0.469). Operating point, named rather than assumed: T=0.7 sampled rollouts over
train-clean-100, whereas the damage profile it is weighted against is a greedy dev-other decode -- not
the same population. The refit saw every base text in its own training corpus (a bias running against
the reference). The dump's own reward columns were written under psi_g_tc100 while every swap number
here is under the named refit. The corpus-zero repair direction is unmeasured at 11 of 8806 swaps.

**30. D7.0a raw donor-support census** (`S/d7_census/D7RawDonorCensusJob.zsnx1p9nLyV3`). The
standalone, label-free feasibility read authorized before D7-v2: every directed edge from each of the
immutable 1,500 external source utterances to the disjoint 4,067-utterance dev complement, and
separately every directed edge within the intended 28,539-utterance scorer corpus. An edge requires a
different utterance from the same speaker and inclusive raw-unit duration match
`20 * abs(L_d - L_s) <= max(L_s, 1)` on 50 Hz unit-array lengths before deduplication. "Same chapter"
is equality of the middle LibriSpeech utterance-ID field. No tokenization, DP feasibility, duplicate
filtering, nuisance ranking, capacity, assignment, scorer, reference text, WER or training enters.

| population | sources | candidates | raw edges | same / different chapter | sources with >=2 in both | sources with >=8 in both |
|---|---:|---:|---:|---:|---:|---:|
| external held/complement | 1,500 | 4,067 | 4,911 | 2,553 / 2,358 | 276 (18.4 %) | 0 |
| intended scorer corpus | 28,539 | 28,539 | 632,913 | 327,169 / 305,744 | 18,843 (66.0 %) | 11,711 (41.0 %) |

On the external graph, 1,331 sources have any donor and 169 are isolated; 2,571 candidates are used
and 1,496 have zero load; the edgeful bipartite graph has 583 weak components. Semantic tuple hashes
`7855557c...d2f3` externally and `3a6038ab...4376` on the scorer corpus (both independently
reproduced). A planner-side maximum-matching replay of the immutable external raw edge table gives
1,267, 1,328 and 1,331 admitted edgeful sources at donor capacities one, two and three; a
deterministic cap-three raw matching contains 669 same-chapter and 662 different-chapter edges, so
capacity three is the smallest tested load cap preserving every raw edgeful source.

**31. D7-v2 / D7.0b frozen donor and loss preflight** (`S/d7_v2/D7V2FeatureJob.hnReOv8t9UWg`,
`S/d7_v2/D7V2AssignmentJob.aSOMkw3hSc0K`). Three serial, label-free jobs: bind the accepted D7.0a
edges, exact pseudo-pairs, raw units, BPE/lexicon inventory and round-1 scorer under the registered
feasibility, duplicate and two-stage ordinal nuisance law; solve the common-set training construction
(ten K=4, 2+2, exact 2-in/2-out Q2 tables) and the external construction (one K=1 cap-three matching,
fixed chapter balance and split floors); only after both structural floors pass, recreate the common
epoch-4 training point, freeze MAD temperature and gradient-norm coefficient, report K1/K4/K8
diagnostics on one common K8-eligible population and measure one K=4 update. The graph contains no
D7.1 scorer, policy, reference text or WER consumer. The feature census found 28,538 feasible scorer
rows, 569,785 hard training edges, 136,966 Q2 edges and 17,748 rows with at least two raw outgoing
donors in both chapter strata; the zero-gap MILP returned 56 admitted rows from two speakers against
the registered 6,778-row / 201-speaker floor. An independent iterative necessary-core calculation
(repeatedly removing every vertex with fewer than two incoming or two outgoing edges in either
stratum) leaves at most 120 rows from four speakers, so every feasible common 2-in/2-out solution lies
inside that core and no optimizer can reach the floor. The assignment stopped before external
matching; the loss preflight never ran.

**32. D7-GAN-SEQDISC full-bed online-negative A/B** (`config/sae_3e1_d7_gan_seqdisc.py`). The
corrected active D7, sharing no construction with approaches 30-31. Ten deterministic theta_0^G
argmax-decode shards cover the 281,241-utterance unlabeled 960 h bed. A D7.0 barrier binds those
texts to the frozen enc50 K=500 raw 50 Hz unit store (`S/quantize_states/PackUnitsJob.I0uzRMfUrKWC`),
reproduces the established ordered seed-42 5 % holdout, persists only a speaker/duration/role index,
and runs the registered one-update finite/resource check on frozen shard 0
(`S/d7_online/D7OnlinePoolJob.XLjSgTzHfwAu`, `.D7OnlinePreflightJob.ZxfANwBZYpaI`). Only after that
PASS artifact exists do the matched D7.1 control (`L_NLL + L_U->z`) and candidate
(`L_NLL + L_U->z + softplus(s_donor-s_own)`) run for one ten-shard corpus pass, preserving the same
initialization, batch order and dropout RNG stream; the candidate's extra forward contributes gradient
without advancing the next positive batch's RNG. Deviation disclosed: prior weight is 0 from step 0
against the refit's 4-epoch prior anneal, entailed by carrying `L_U->z` across a single pass.

Decoder equivalence of the merged shards (label-free): on the 28,539 tc100 utterances the merge shares
with the banked greedy decode `F/ReturnnForwardJobV2.66pIzBzffnK2`, the two texts agree on 25,426
utterances exactly (89.09 %) and differ by 4,667 word edits against 1,016,991 reference words, i.e.
0.459 %. Of the 3,113 differing utterances 67.9 % differ by a single word edit (mean 1.50) and the net
length drift is +7 words -- the signature of argmax ties resolving differently under a different
batching, not of a different decode.

D7.0's registered parity clause cannot pass on this backend (reproduced read-only on one GH200 through
the job's own code path, `scripts/d7_parity_diag.py`, `log/d7_parity_diag.1446568.out`, shard 0's
first batch, 256 rows):

| comparison | loss | max abs gradient delta | gradients equal |
|---|---|---:|---|
| the two deepcopies, i.e. what the clause asserts | 9.983121871948242 both, EQUAL | 2.623e-06 | no |
| the SAME model object, run twice -- the control | identical again | 5.484e-06 | no |

The same-object repeat is decisive: rerunning one model on one batch perturbs its gradients MORE than
the two copies differ from each other, so the difference is the backend's own run-to-run noise
(`deterministic_algorithms` False, `fast_bw` True, atomics in the FastBaumWelch backward), not a state
difference between the arms. Loss equality holds exactly. The amended self-calibrating clause (Gates)
then PASSED on its own artifact: losses exactly equal at 9.983121871948242, F 7.391e-06, cross
4.053e-06, candidate gradient delta 0.02046 confirming candidate-only gradient flow. First live
exercise of the registered infeasible-donor counter: 0 infeasible of 256 donor pairs, 209
ordinary_window / 47 nearest_fallback.

D7.1 completed 2026-08-21, both arms, one ten-shard corpus pass each: a single 14-minute job per arm
(13:59 control / 13:58 candidate), 2,361 batches over 10 round-robin shards at ~70 s per shard, peak
resident 4.89 / 4.85 GiB. Both arms report the identical bed -- 281,241 rows, 267,179 train and 14,062
held before filtering, then the SAME four own-infeasible train anchors dropped by name
(`3488-85273-0024`, `3889-130125-0028`, `4492-8904-0032`, `8424-284526-0028`), 267,175 trained and
14,062 held, digit-identical to the offline dropcheck and to each other. Shard row and frame counts
agree arm to arm at every shard and the internal-held donor draw is identical (11,855 ordinary_window
/ 2,153 nearest_fallback / 54 singleton, 14 infeasible donor pairs, 9,825 unique donors), so the two
arms differ only in the loss term; an exhaustive recursive diff of the two arms' monitors finds ONE
non-metric difference, `online_weight` 0.0 vs 1.0.

| arm | objective | internal-held NLL per frame | internal-held mean `L_online` | job |
|---|---|---:|---:|---|
| control | `L_NLL + L_U->z` | 2.5259 | 0.010225 | `S/d7_online/D7OnlineTrainJob.j16rTskXF1QU` |
| candidate | control `+ softplus(s_donor - s_own)` | 2.5319 | 0.007541 | `S/d7_online/D7OnlineTrainJob.WA1bqjXQtzeZ` |

Train-side sampling over the 267,175 anchors, identical in both arms: 266,134 ordinary_window / 1,041
nearest_fallback, 1 infeasible donor pair, 170,443 unique donors, maximum donor reuse 8, own/donor
duration ratio mean 1.0144 (min 0.4615, max 3.7246). The four dropped anchors were all
ordinary_window donor cases, so the donor law was untouched by the drop.

D7.2, all four registered clauses (ten jobs, all finished). Clauses 1-2,
`S/d7_admission/D7OnlineAdmissionJob.h0LsMi9zt5aI`: 14,008 of the 14,062 internal held anchors are
eligible (the 54 `singleton` anchors have no donor and are excluded identically in both arms), in
2,274 speaker clusters, 32 paired draws each.

| statistic | control | candidate | clause |
|---|---:|---:|---|
| mean internal-held `L_online` | 0.0102249 | 0.0075263 | -- |
| paired candidate-minus-control mean | — | -0.00269867 | -- |
| speaker-cluster bootstrap, one-sided 95 % upper bound | — | -0.0026589 | **1 PASS** |
| two-sided bootstrap interval | — | [-0.0027472, -0.0026505] | -- |
| share of anchors with a negative difference | — | 99.25 % | -- |
| internal-held NLL per frame (8,642,253 frames) | 2.525882 | 2.531898 | **2 FAIL** |

The recomputed per-frame NLLs reproduce the values D7.1 banked to 3.62e-9 on both arms. Ratified
donor-diversity diagnostic: 32 draws land on a median 3.0 distinct donors per anchor (mean 3.42, min
1, max 12) and 4,022 of the 14,008 anchors -- 28.7 % -- see exactly ONE distinct donor across all 32
draws, so the precision comes from the speaker-cluster resampling and not from the draw count. 448 of
448,256 donor draws are structurally impossible and contribute exactly 0 to both arms by documented
design.

Clause 3, external half -- both `PsiHeldNllJob`s (text probes `S/psi_align_jobs/PsiTextProbeJob.o6d4bN6EBB2O`, `.C3vgGM2guvS0`) on the unchanged frozen 1,500-row external gold-dev
set, 1,493 of 1,500 pairs scored in both arms, 0.47 % impossible under the length-matched null in both:

| arm | nll/frame, true | held `ce_loo` | `text_explained_loo` | usage gate | job |
|---|---:|---:|---:|---:|---|
| control | 2.4595 | 2.2588 | +3.7744 | +5.0509 | `S/psi_align_jobs/PsiHeldNllJob.6bf5GyPGHuAi` |
| candidate | 2.4595 | 2.2581 | +3.7751 | +5.3587 | `S/psi_align_jobs/PsiHeldNllJob.QDKOlbGXdEOA` |

The candidate's usage-gate widening (+0.3078) decomposes to +0.30712 from the length-matched deranged
NULL against +0.00071 from the true side -- the candidate mostly prices the null worse, not the truth
better.

Clause 3, gate table (`S/gate_table/PsiGateClauseTableJob.4Z0gb5GgtD2u`, incumbent = the exact D7
control, 1,443 utterances scored by both arms, seed 42, 10,000 resamples, `improvement_void` empty).
Both arms clear the (i) absolute floor against H_uni 6.0332 and pass (ii); the candidate also passes
the (i) improvement half. The five corruption ladders (clause iv) are unchanged -- every paired
difference straddles zero. The one clear movement is the wrong way:

| matched insertion discount | control | candidate | paired difference | 95 % CI | p |
|---|---:|---:|---:|---|---:|
| k=1 | 0.0037 | 0.0109 | +0.0072 | [0.0039, 0.0106] | 0.000 |
| k=2 | 0.0156 | 0.0238 | +0.0082 | [0.0031, 0.0133] | 0.001 |
| k=4 | 0.0238 | 0.0443 | +0.0205 | [0.0125, 0.0287] | 0.000 |

The substitution discount (computed, registered to decide nothing) is also worse for the candidate at
every k with CIs excluding zero. The table's eligibility columns split on the unpinned convention:
`elig_pt False` / `elig_CI True` for the candidate (`worse_pt 2`, `worse_CI 0`), and the table returns
NO WINNER under BOTH readings -- but for two DIFFERENT reasons, neither of which is the one the
finished artifact prints (see Open findings). Clause 4, scorer parity -- PASS on both arms exactly:
512 of 512 rollouts round-tripped, `max |online - offline| = 0.000e+00` against a 2e-3 tolerance,
0.00 % rows floored (`S/psi_align_jobs/PsiScorerParityJob.sZPxS9hlIGKa`, `.vTOLyrLz4Kl6`).

Label-as-evaluation ranking tables computed by the clause-4 parity vehicles for BOTH D7 fixed finals
over the fork-epoch dump (28,531 groups, T=0.7; WER never a gate input): candidate
(`PsiAlignRerankJob.kkEEVosPO80P`) spearman 0.3889, selection WER 5.126 %, eta 0.258; control
(`.OiRBghBiTriv`) 0.3878, 5.136 %, eta 0.250; shared mean WER 5.477 %, oracle 4.116 %. The two arms
rank near-identically; the tiny candidate lead has no null spread behind it and selects nothing. This
bed is the BEST-BED fork policy's tc100 rollouts, not the operative theta_0^G policy, and the tables'
G3 bar lines are cross-bed diagnostics (gap_true is per-arm units). Policy-leg performance -- the WER
of a leg trained on the candidate reward -- remains the one genuinely unmeasured quantity.

**33. D6-PERIODIC/GAN960-FROZEN: the frozen-scorer loop restarted from theta_0^G960** (user-funded
2026-08-21 on the §3d.A scale read; `config/sae_3e1_d6periodic_gan960_frozen.py`). Approach 26's
frozen sibling verbatim -- eight segmented policy legs, same round-robin 960 h shard per leg, shaped
reward, T=0.7, cosine offsets, fresh optimizer state per leg, round 1's completed `d_min=2` scorer
`S/psi_align_jobs/PsiAlignTrainJob.dsMKgPHQApyR` frozen at EVERY leg -- with exactly one experimental
change: the policy init is theta_0^G960 (`T/ReturnnTrainingJob.HuSkdbuVRg6d` sub-epoch 10, dev
13.11/16.82) instead of theta_0^G (`.2fb02hGUdHNj` ep10, dev 13.89/18.34). A fresh scorer refit on
theta_0^G960's own decodes is deliberately NOT funded: holding the sibling's frozen scorer is what
keeps this a one-argument init A/B. Verified against the built graph before launch: the frozen scorer
resolves to `dsMKgPHQApyR` at all eight legs; legs 1/2/8 are `T/ReturnnTrainingJob.ohmLWWmr6Kxe`,
`.liehXoiGoRI0`, `.V1WEV1giQXZA`; legs 2-8 carry dump/pool/refit `None`; of the 64 unfinished jobs a
launch would fund there is ZERO psi_align/curate/scorer_diag work. One inherited-bookkeeping conflict
flagged rather than resolved: `build()` reuses `round1_artifacts()`, so leg 1's record carries
dump/pool/refit that are the SCORER's provenance from theta_0^G decodes, which in this arm is not its
own round 1 -- those jobs are finished and fund nothing, but a downstream audit could misread them.
Legs 1-3 had finished/were running at the last read; no endpoint exists in this log.

**34. D8.0 feasibility read of the two frozen rollout dumps** (`config/sae_3e1_d8_0.py`,
`sae/d8_feasibility.py`; speech-llm `889750c`, v2 guard `a3dd6c7`, operative v3 `3843918`). The
CPU-only registered read that can close D8 before any rollout is generated on the 960 h bed. Per group
it dedups the support on the D8 reader-normalized string, zeroes and counts candidates that are empty
after the fold, structurally infeasible on their own audio at `d_min=2`, or unencodable, then reports
distinct support, median ESS across the registered tau grid for the shaped, acoustic-only and LM-only
scores, the provisional `tau_star` from the `|median ESS - 3|` rule, the within-group weight variance
token count alone explains at `tau_star`, and the median spearman between shaped weights and each
single-term weight vector. Rows are whitelisted by `kind`, so reference rows and the gold-derived
`wer` column are never read; the LM term is converted to per-unit currency
(`lm_prior * n_tokens / n_units`) against each dump's OWN unit store before any weight. Only the
theta_0^G artifact binds, on its T=0.7 slice.

The v1 read returned NO-GO on that slice (`S/d8_feasibility/D8FeasibilityReadJob.iCuYuvkL6bwr`; fork
`.onK5ekDuoLLA`). The v2 reader added one guard, pre-registered in the job docstring before it ran: a
member the reader calls structurally infeasible whose STORED `recon` is finite is a contradiction, so
a nonzero count on the binding slice returns UNRESOLVED, never a no-go -- which is what it returned
(`.mDQ2LoAzrMTE`, fork `.ulUbBcxIiJtf`; binding slice 5,096 law conflicts of 5,730 scored, distinct 0
with the exclusion applied and 12 without). Both are kept as the evidence that motivated the ruling.

**v3, the ruled operative-frame read** (clause-(a) ruling; Gates). Exactly one thing changes: the
structural-infeasibility exclusion is evaluated against `T_i` from
`S/quantize_states/PackUnitsJob.I0uzRMfUrKWC` with coverage over the slice's ids asserted; the
per-unit prior currency still divides by the DUMP's own store, the two joins deliberately separate.
The v2 law-conflict guard retires by construction and its count survives as the reported
`operative_law_finite_score_exclusions`. Two implementation-over-ruling deviations ratified, both
conservative: the safety-valve denominator counts ALL excluded scored members, and the coverage assert
spans all 34,106 dump ids.

| artifact | slice | excluded / scored | distinct (feasible, greedy incl.) | tau* | median ESS at tau* | token R2 | rho(shaped, LM-only) | rho(shaped, acoustic-only) | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| theta_0^G, 512 utts | T=0.3 | 0 / 3,182 | 6 | 0.05 | 3.685 | 0.4133 | 1.0000 | 0.3000 | reported |
| theta_0^G, 512 utts | T=0.5 | 0 / 4,693 | 10 | 0.05 | 5.308 | 0.2837 | 0.9833 | 0.2857 | reported |
| theta_0^G, 512 utts | **T=0.7 (binds)** | **0 / 5,730** | **12 of 13** | 0.05 | 5.433 | 0.1741 | 0.9790 | 0.3132 | **GO** |
| theta_0^G, 512 utts | T=0.9 | 0 / 6,457 | 13 | 0.05 | 2.392 | 0.0483 | 0.9785 | 0.5497 | reported |
| theta_0^G, 512 utts | T=1.0 | 0 / 6,604 | 13 | 0.2 | 3.976 | 0.0354 | 0.9785 | 0.6593 | reported |
| fork epoch, 28,539 utts | T=0.7 | 18 / 101,190 | 3 | 1.0 | 2.976 | 0.3928 | 0.5000 | 1.0000 | reported-only |

Jobs: `S/d8_feasibility/D8FeasibilityReadJob.mv2d0vkWN93a` (theta_0^G, binding) and `.W7TWfwoZtkaC`
(fork epoch). The fork read's numbers are unchanged from v2 to the last digit, because that dump
already joined the raw 50 Hz store; its 18 exclusions in 101,190 members are the genuine rate the
ruling prices the 5 % valve against. Independent recompute of the binding slice from the raw dump rows
and the raw 50 Hz store reproduces it exactly: 512 groups, 5,730 distinct scored classes, ZERO
empty/unencodable/infeasible members, with-greedy median 12.0 and rollouts-only 12.0. The margin is
structural: raw-store median 695 frames over the slice against the pooled 169/174 that drove v1's
88.9 % exclusion, tightest single-utterance margin 65 frames.

**35. D8.1a: the operative-bed candidate generation pass and the frozen weight artifact.** Released by
the planner 2026-08-22 once the D7.2 verdict existed -- the registered condition was the verdict, not
a pass. One group-12, T=0.7 sampled dump of theta_0^G over all 281,241 utterances of the 960 h bed, in
the same ten round-robin shards D7.0 decoded in, `recon` under the pinned weight scorer
`S/psi_align_jobs/PsiAlignTrainJob.dsMKgPHQApyR` and `lm_prior` under the registered prior; then
`D8MergeRolloutsJob.gXDwFsfvraDS` (asserts the shards are a partition -- no tag in two shards, union
exactly 281,241) and `D8WeightJob`. The merged dump is complete: 281,241 utterances, 3,937,374 rows =
greedy 281,241 + rollouts 3,374,892 + true 281,241, and 281,241 x 12 = 3,374,892 exactly.

Two construction facts, each a place a silent error would have survived. The dump does not call the
existing `_reward_rank` (that builder attaches its reconstruction target from a whole
`{seq_tag: units}` pickle, which the 960 h raw 50 Hz stream cannot be), so `forward_step._units_by_tag`
also accepts the packed store and the dump passes `units_store_path`; the two interfaces were checked
against each other on the real artifacts BEFORE the config was written -- over the whole 34,106-tag
shared population the tc100 pickle `MergeUnitsPklJob.ncxcd3vouD5E` and the operative store
`PackUnitsJob.I0uzRMfUrKWC` return byte-equal unit sequences with zero mismatches (population median
length 674), an independent re-confirmation of the premise behind D8.0's v3 ruling. And because this
dump is GENERATED against the operative store, the per-unit currency denominator and the
structural-feasibility frame are the same object, which `D8WeightJob` asserts rather than assumes.
`D8WeightJob` imports every statistic from `d8_feasibility` (`build_support` / `slice_statistics` are
module-level and the D8.0 read job delegates to them unchanged), so "the same statistic as D8.0" is
the same code; the binding-slice filter is a fail-closed assert on each row's T, the 5 % valve is
enforced through the shared `valve_verdict` ordering, the ratified dedup-survivor rule (already-
normalized member, else earliest stored row) is implemented with a score-differing collapse
diagnostic, and a structurally feasible member with non-finite `recon` is its own loud category
feeding the valve. Both D8.0 dumps carry ZERO feasible-but-non-finite rows (31,232 and 371,007
whitelisted rows, all finite), so that guard is pure there.

`S/d8_pool_scores/D8GreedyEquivalenceJob.xR1RduqgjFKe` implements the planner's ruling on the
registration deviation (Gates): it compares on the D8 reader's own fold, reports coverage in its own
buckets so zero mismatches cannot be reached on a subset, and reports rather than raises. Its verdict
is NOT EQUIVALENT (verdict 70). The pre-relaunch pin `XTdRp3OO3LNf` is superseded and never ran.
Pool scoring under the corrected convention: `S/d8_pool_scores/D8PoolScoresJob.1ivehCZ5q5ON`
(281,241 of 281,241 members through the text path, 0 dump columns reused, 0 degenerate rows, 249,679
tags where the dump's regenerated greedy agrees anyway); overlap probe
`S/d8_pool_scores/D8PoolOverlapProbeJob.GerShND5ibtT` (verdict 71), token mechanism
`.D8PoolTokenMechanismJob.rVkoJpPoBGG8` (verdict 72).

RESULT TABLE (one row per read of the frozen weight artifact; `rho_ac` is spearman(shaped,
acoustic-only) and `rho_lm` spearman(shaped, LM-only), both at `tau_star`).

| convention | job | excluded / scored | median distinct | taus in ESS band | token R2 | tau_star | rho_lm | rho_ac | verdict / arms |
|---|---|---|---|---|---|---|---|---|---|
| corrected text path (operative) | `D8WeightJob.juRpzTNHKCSq` | 18 / 3,170,676 | 13.0 | 0.05, 0.1 | 0.0620 | 0.05 | 0.3462 | 0.9835 | GO / candidate_acoustic |
| legacy mixed (sensitivity only, same job) | `D8WeightJob.juRpzTNHKCSq` | 18 / 3,170,676 | 13.0 | 0.05, 0.1 | 0.0620 | 0.05 | 0.2747 | 0.9835 | GO / candidate_acoustic |

Independent recompute over all 281,241 frozen groups of `supports.jsonl` reproduces every gate
statistic to the last digit: median distinct 13, median shaped ESS at all five taus (2.982409 and
5.32854 inside the band, the rest outside), tau_star 0.05 by the nearest-to-target rule, median
per-group spearman 0.34615384615384615 (shaped-LM) and 0.9835164835164836 (shaped-acoustic), median
token R-squared 0.06203486419535191 with 278,215 defined groups, live members 3,170,658 = 3,170,676
scored minus the 18 exclusions (7 `empty_after_fold` + 11 `infeasible`), valve idle at 5.7e-06. The
spearman convention is per-group rho with average-rank ties aggregated by median.

**37. D8.1b: the candidate-acoustic scorer refit** (`S/d8_train/D8ScorerRefitJob.2bQzhz6U1yHp`;
speech-llm `aadf92b`). The D7 exact-control recipe by import, with the one registered change -- each
anchor visit draws its target from the anchor's frozen `acoustic_only` weight vector at `tau_star`
0.05 instead of always the greedy 1-best. Batches are the control's own, formed by the control's code
before any draw. No control trains: the D7.1 exact control (`S/d7_online/D7OnlineTrainJob.j16rTskXF1QU`)
is reused as the one-hot special case. The job refuses to run unless the weight artifact reads GO,
funds exactly `candidate_acoustic` and reports no sensitivity flips, plus a source-drift guard after
graph construction.

| arm | job | wall clock | shards / batches | anchors trained / held | realized greedy-draw fraction | internal-held per-frame NLL |
|---|---|---|---|---|---|---|
| D8.1b candidate-acoustic | `D8ScorerRefitJob.2bQzhz6U1yHp` | 13:52 | 10 / 2,361 | 267,175 / 14,062 | 0.25312 | 2.51389 |
| D7.1 exact control (reused) | `D7OnlineTrainJob.j16rTskXF1QU` | 13:59 | 10 / 2,361 | 267,175 / 14,062 | n/a (always greedy) | 2.52588 |

Draw diagnostics banked in `sampling.json`: 267,175 draws (one per anchor visit, one pass), mean
11.273 members available, 67,628 greedy draws, 68,164 drawn targets that encode identically to the
control's, mean state length delta +1.379 against the control, 0 infeasible drawn members, donor cases
`ordinary_window` 266,134 / `nearest_fallback` 1,041 with 0 infeasible donor pairs. Artifact-reading
caveat: `sampling.json`'s `target` field says "own greedy pseudo-text, never a draw" and describes the
INTERNAL-HELD evaluation only, not training -- it is inherited from the D7 schema's held block and
reads as a contradiction of the arm's purpose; the successor schema names `held_target` explicitly, and
the finished artifact was not re-hashed for a wording fix.

D8.2 clause 1 (`S/d8_admission/D8AdmissionJob.C2HUHUtUjfhN`), on the 14,062 internal-held anchors over
2,328 speaker clusters, 10,000 resamples at seed 42; `per_anchor.jsonl` persists the paired deltas and
cluster ids, and the verifier recomputed mean, `delta_NI`, bound and negative share bit-exactly from it:

| quantity | value |
|---|---|
| control pooled per-frame NLL | 2.525882 |
| candidate pooled per-frame NLL | 2.513888 |
| paired per-anchor delta, mean | -0.012475 |
| paired delta, one-sided 95 percent upper bound | -0.011800 |
| `delta_NI` (control-only, D7.2 convention verbatim) | 0.004826 |
| clause 1 | PASSES |

D8.2 clauses 2-3 (`S/gate_table/PsiGateClauseTableJob.xFSaHcqvUR2S`), paired on the 1,443 of 1,500
external rows scored by every arm, seed 42, 10,000 row-bootstrap resamples. Corruption-ladder
spearman, candidate minus control:

| ladder | candidate level | paired delta | 95 percent CI |
|---|---|---|---|
| `filler_sub` | 0.9493 | +0.0027 | [-0.0001, +0.0053] |
| `lmsub` | 0.9617 | +0.0019 | [-0.0007, +0.0044] |
| `del` | 0.9520 | -0.0012 | [-0.0046, +0.0021] |
| `filler_ins` | 0.9769 | -0.0033 | [-0.0064, -0.0003] |
| `lmins` | 0.9258 | -0.0028 | [-0.0078, +0.0022] |

Matched insertion discount, candidate minus control: k=1 +0.0078 [0.0044, 0.0113] p 0.000; k=2 +0.0060
[0.0006, 0.0114] p 0.029; k=4 +0.0097 [0.0019, 0.0172] p 0.012. Leave-one-out cross entropy: candidate
2.2343 against control 2.2588 (reference `PsiHeldNllJob.BhUn7Sa3CW67`). Gate v2 clause row: candidate
(i)floor PASS, (i)improvement PASS, (ii) PASS, ladders worse on 3 of 5 by point estimate and 1 of 5 by
CI, ELIGIBLE False under both readings; verdict NO WINNER (no eligible arm) under both.

**36. D6-PERIODIC/GAN-FROZEN completes: the frozen-scorer control against the periodic arm, all eight
legs.** The registered schedule-only control (user-directed 2026-08-20): the D6-PERIODIC/GAN policy
graph verbatim from theta_0^G, with round 1's scorer `S/psi_align_jobs/PsiAlignTrainJob.dsMKgPHQApyR`
frozen at all eight legs, so scorer recency is the only intended experimental difference. Both arms
finished 2026-08-22. Policy legs `T/ReturnnTrainingJob.kr1foUV6lecx` (reused), `.JVfEDCPIPWkq`,
`.o2GFVkZZPNRT`, `.fEvotypkqDao`, `.91wIJ5JpsdIW`, `.2p2hpz7nk5vd`, `.ZgRzUxDRhajE`, `.ycoJLypxisD7`.
The registered reproduction check passes by construction: frozen leg 1 and periodic leg 1 resolve to
the SAME scoring job (`R/scoring/ScliteJob.LzKRDl102Jaf`), because leg 1 IS the banked periodic job.
All 32 WER cells were traced to their own `ScliteJob` artifacts (reference word counts constant at
54,402 / 50,948 across every cell, both arms on the same two reference STM jobs), the frozen arm reads
`dsMKgPHQApyR` in all eight legs' on-disk configs with no per-leg refit anywhere in its 127-job
closure, and every cell is plain sclite with no rescoring job in either closure.

| leg | periodic, dev-clean / dev-other | frozen, dev-clean / dev-other | ahead |
|---|---|---|---|
| 1 | 14.45 / 19.69 | 14.45 / 19.69 | identical by construction |
| 2 | **12.85 / 17.89** | 13.40 / 18.44 | periodic, 0.55 / 0.55 |
| 3 | 13.20 / 18.20 | 13.55 / 18.74 | periodic, 0.35 / 0.54 |
| 4 | 17.76 / 23.17 | 17.80 / 23.27 | periodic, 0.04 / 0.10 |
| 5 | 17.92 / 23.27 | **16.01 / 21.57** | frozen, 1.91 / 1.70 |
| 6 | 18.38 / 24.01 | 17.66 / 23.02 | frozen, 0.72 / 0.99 |
| 7 | 18.28 / 23.70 | 17.50 / 22.82 | frozen, 0.78 / 0.88 |
| 8 | 18.82 / 24.56 | 17.61 / 22.66 | frozen, 1.21 / 1.90 |

Reference level on the same reading: the no-loop init theta_0^G is 13.89 / 18.34.

**38. D8.4: the paired ranking-quality (eta) read on the operative theta_0^G bed.** Built after the
user reopened D8. One instrument (`PsiAlignPairedCompareJob`, `archive/SAE_3a_spec_legacy.md`'s,
reused unchanged except that its per-temperature cell now also carries the shared
`mean_wer`/`oracle_wer`/`sel_wer` it already computed -- hash-neutral, and it lets the reader restate
delta eta in its plain-WER form without recomputing the pairing on a second, differently-dropped bed),
two arms differing in `model_pt` alone, both reranking the SAME frozen rollout dump at the SAME
temperature; the reader (`D8EtaReadJob`) restates delta eta in its plain-WER form, refuses any bed that
is not the registered one, and fails closed if the two forms disagree. Step zero answered on the code:
`PsiScorerParityJob` does NOT discharge D8.4 -- it re-scores ONE arm's own rerank dump against that same
dump's `recon` column, with no second arm, no selection, no eta and no null. Two beds are instantiated:
the OPERATIVE theta_0^G dump (`F/ReturnnForwardJobV2.J9yA1eYnxwYA`, 512 utterances, G=12, T=0.7) as the
primary, and the fork-epoch-2 dump (`.QbIYruVEI0fF`, 28,539 groups) as context only. Each dump joins its
own unit store, since that is the stream its stored `recon` column is per frame of.

The first build reused the D8.2 graph's rerank pair, which consumes the FORK-EPOCH-2 dump; D8.0 had
already moved its binding clause to `J9yA1eYnxwYA` precisely because that one carries the OPERATIVE
policy, so the registration's own pin excluded the reused bed. The correction added two reranks on the
operative dump's T=0.7 slice, and the primary verdict reads from that pair alone.

| bed | rerank pair | groups offered | groups surviving | rows infeasible | candidate eta | control eta | paired delta eta [95 pct CI] |
|---|---|---|---|---|---|---|---|
| operative theta_0^G, 50 Hz enc50 join (THE VERDICT) | `GNOktIsG251m` / `JSZvokFxjNkJ` | 512 | 512 | 0 of 31,744 (0.000 pct) | +0.4220 | +0.4513 | -0.0293 [-0.0697, +0.0085] |
| operative theta_0^G, sae3d quarter-rate join (failed closed) | `8oYpO4IBeqHb` / `sQGYUL22Kpg6` | 512 | 46 | 25,867 of 31,744 (81.5 pct) | -0.5688 | -0.5731 | +0.0043 [-0.1020, +0.1257] |
| operative, full-set rank-only column | same | 512 | 512 | (infeasible ranked last) | -0.1680 | -0.1548 | not computed as a pair |
| fork epoch 2 (context) | `qVTVrRvyOjZ9` / `OiRBghBiTriv` | 28,539 | 28,531 | 38 of 399,546 (0.010 pct) | +0.2471 | +0.2503 | -0.0033 [-0.0164, +0.0096] |

Jobs: primary (re-pinned 50 Hz enc50 join) `S/psi_align_jobs/PsiAlignRerankJob.GNOktIsG251m` /
`.JSZvokFxjNkJ` -> `S/psi_align_compare/PsiAlignPairedCompareJob.ACR10RHlnsop` ->
`S/d8_eta/D8EtaReadJob.KwmHTXqiJMGr` (THE VERDICT, verdict 84). Failed-closed quarter-rate join
`.8oYpO4IBeqHb` / `.sQGYUL22Kpg6` -> `PsiAlignPairedCompareJob.ffqCTOA3qssf` ->
`S/d8_eta/D8EtaReadJob.S3NTCZAOfSnZ` (`error.run.1`; the refusal message is the result, verdict 83) --
banked as the record and left in the graph at their finished hashes. Fork context
`PsiAlignRerankJob.qVTVrRvyOjZ9` / `.OiRBghBiTriv` -> `PsiAlignPairedCompareJob.yrEq1ogcluJF`. Same
dump, same draw, same banked per-rollout WERs across the two joins -- only the frame stream moved.

Bed feasibility, quoted from the REGISTERED PRODUCER `S/d8_bed_feasibility/D8BedFeasibilityJob.9fCCv5HAPg4a`
(v2, superseding `QTlLFcnka0Hy`) in the producer's own figures and populations. Its conventions print
ahead of every number: per-utterance frame ratios summarized on the shared key set and never taken as a
ratio of corpus means, printed as right / left in the order the label names the two beds so a
quarter-rate store reads 0.25 and NOT its reciprocal 4.00; nearest-rank quantiles with no interpolation;
the crude length bound stated as OVER-predicting (dividing characters by the nominal 1.5 characters per
state overestimates the state count, because BPE merges pull the realized count below it) and offered
only as a mechanism check; observed infeasibility taken from each arm's own rerank report with the two
arms ASSERTED equal; and the shared-key-set corpus mean distinguished from the per-bed mean. THREE
POPULATIONS APPEAR BELOW AND THEY ARE NOT THE SAME SET: the crude bound is on the T=0.7 rollout slice
(6,144 rows operative, 342,468 fork), observed infeasibility is the rerank's own count over ALL rows at
every temperature (31,744 operative, 399,546 fork), and the frame ratios and shared-key-set means are
over the 34,106 utterances every store holds.

| statistic | operative, sae3d quarter-rate join | operative, 50 Hz enc50 join | fork ep2, 50 Hz enc50 join |
|---|---|---|---|
| shared-key-set corpus mean frames/utt (34,106 utterances) | 146.8 | 585.7 | 585.7 (same store) |
| per-bed mean frames/utt (that bed's own utterances) | 158.2 | 631.2 | 633.8 |
| characters per unit frame, p05 / p50 / p95 | 0.88 / 1.16 / 1.42 | 0.22 / 0.29 / 0.36 | 0.22 / 0.29 / 0.36 |
| crude bound, T=0.7 slice | 6,068 of 6,144 (98.76 pct) | 0 of 6,144 (0.00 pct) | 54 of 342,468 (0.02 pct) |
| observed infeasible, all rerank rows | 25,867 of 31,744 (81.486 pct) | 0 of 31,744 (0.000 pct) | 38 of 399,546 (0.010 pct) |
| groups dropped | 498 | 0 | 8 |

Per-utterance frame ratio, quarter-rate store over the 50 Hz store, on all 34,106 shared utterances:
p05 0.25, p50 0.25, p95 0.25, min 0.25, max 0.26 -- the same stream decimated by four, with no
utterance escaping the pattern. The two 50 Hz beds ratio to 1.00 throughout, the job's own check that
they share a store. Every infeasibility count is IDENTICAL across the two arms, asserted by the job,
which is what carries the cause from either scorer's weights to the text-to-unit alignment. Both
scorers train against the same frozen 50 Hz store `PackUnitsJob.I0uzRMfUrKWC` (read from both training
jobs' info files), which grounds the re-pin. The producer's revision bump re-hashes only the
feasibility leaf: `D8EtaReadJob.KwmHTXqiJMGr` is byte-identical in hash before and after, and the
producer has no dependents.

D8.2 clause 4 (`S/psi_align_jobs/PsiScorerParityJob.sRJ7LUmF4nMw`), read for the record after the
outcome was already reached: online (loop) against offline (G3) per-frame reconstruction on the
candidate arm -- 512 of 512 rollouts round-tripped, max absolute difference 2.384e-07, mean 4.657e-10
against a 2.0e-03 tolerance, 0.00 pct floored. PASS.

**39. D9: the evolved-point refit of the D8 recipe, its support census and the two-arm read.**

PRE-SPEND PROVENANCE, traced end to end 2026-08-23, every link read from a job's own `info`:
`T/ReturnnTrainingJob.rJWSC5xOsrf2/output/models/epoch.002.pt` -> `ExtractAvSubmodelJob.FSYsyEJm5VHX`
(its `grpo_checkpoint` PARAMETER is that exact file, `submodel_prefix` `av.`) ->
`F/ReturnnForwardJobV2.SgTOBGwxO6nF` (dev-clean) and `.9GwKJ97FtuG6` (dev-other) ->
`SearchWordsDummyTimesToCTMJob.e3rgP2fFMD8f` / `.lyBmKhRT3pUZ` -> `ScliteJob.paK5JVk5SckU` = **12.68**
and `ScliteJob.KTVFso7HriMn` = **17.57**, each read from its own `output/wer`. The recog does NOT
consume the training checkpoint directly -- an `ExtractAvSubmodelJob` sits between -- so a check
stopping at "the training job has an epoch.002.pt" would have proved nothing. ARM IDENTITY from the
same source: the training job's INPUT list contains `PsiAlignTrainJob.DnBJxqz4sNQZ/output/model.pt`
and its alias is `..._shaped_T0.7_lr2e5_psid2_contrast/training`, so the pin is the d2_contrast-shaped
arm, not the incumbent-shaped one at the same schedule position -- the two sit adjacent in approach 9's
table at 12.68/17.57 and 13.91/18.91, a real confusion risk. The pinned training job carries a `hold`
file, so that arm is paused; D9 only reads a written checkpoint from it.

D9.0 FRAME, settled by reading the checkpoint's own inputs rather than by choosing: the donor is STOCK
(the pinned training job's INPUT list carries `DownloadHuggingFaceRepoJob.JcEANaYZr2oe/output/hub_cache`
and no `ExportHfLmDirJob` output, so the dump must NOT pass `qwen_hub_dir=lbslm_donor()`); and the read
bed is tc100, not the 960 h bed, which is the established frame (D6-PERIODIC's own refresh dumps a
960 h-trained policy over tc100 with `tc100_units()`; D8.4's beds are tc100 on both sides), the
registration's "960 h HF/Ogg bed" naming the refit frame, with "sized to D8.4's read" fixing the read at
512 utterances x 12 rollouts at T=0.7. Machinery reused rather than rebuilt: the dump is `_reward_rank`
with `psi_model_args` and `av_checkpoint_prefix="av."`; the incumbent census is a `PsiAlignRerankJob` on
that dump with `DnBJxqz4sNQZ` at d_min=1 as trained; the one genuinely new piece is the STRUCTURAL
census, exact rather than D8.4's crude character bound (feasible iff the realized state count times
d_min=2 fits the frame count), because the D9.2 STOP clause rests on it predicting the refit census by
construction.

D9.0 GATE PASSES (`S/d9_feasibility/D9FeasibilityJob.oabVIcp22cy1`; dump
`F/ReturnnForwardJobV2.t4sIOlpGVDcY`, incumbent rerank `S/psi_align_jobs/PsiAlignRerankJob.cysJQBiP9iW1`).
The dump is 512 utterances x 12 rollouts at T=0.7 plus 512 greedy and 512 reference rows = 7,168, which
is D8.4's read size exactly. The rerank reports 0 of 512 groups dropped at T=0.7 and 0 of 512 groups
inside psi_align's own training set (a leakage guard worth having in print).

| census | result |
|---|---|
| (a) incumbent finite scores, d_min=1 as trained | 7,168 rows, 0 infeasible, 0 groups dropped |
| (b) structural alignability, d_min>=2, rollout rows | 6,144 of 6,144 (share 1.0000), 512 of 512 groups retained |

Median row: 695 unit frames against 210 needed under the refit topology, so the bed clears the
minimum-duration bound by better than a factor of three -- the opposite of D8.4's operative
quarter-rate join, which lost 81.5 percent of rows and 498 of 512 groups, and the point of running the
gate first was that D8.4 could not make this call until after its refits had been trained. WHAT THE
GATE DOES NOT SAY: nothing about eta, nothing about whether a refit beats the incumbent, and nothing
about D9.2's read set. It says the bed can carry the read.

D9.1 (speech-llm `a42fa37`, `8df2580`): a ten-shard 960 h rollout dump from the pinned checkpoint
(`F/ReturnnForwardJobV2` x10: `7pn7wCdqQ7Wb pfmSXPmED4Ov hhbi9TmvyRnc lzKuXw4bkFRF 5tXsjYhVMIMF
0ZcqQ8rhgO0N UMJyglLRiKkf AHhdYqA5ukeE yVygb8dep7HF ueKzHy0j1OjW`, G=12, T=0.7, `max_seqs=8`), merged at
`S/d8_weights/D8MergeRolloutsJob.C4G6qGzjEIrx`, with the pseudo-text pool built from THAT dump's own
greedy rows (`S/d9_refit/D9PoolFromDumpJob.RhwBlgMhqHbA`), tau PINNED at D8.1a's 0.05 rather than
re-solved. Two refit arms were registered: arm 2, the 1-best refit
(`S/d9_refit/D9OnlineTrainJob.nJQy199AQZQu`, `online_weight=0`), and arm 3, the soft-EM refit
(`S/d8_train/D8ScorerRefitJob.XvPF118rphQP`) on the weight artifact.

TWO BUILD DECISIONS, both ratified by the planner. (1) THE POOL IS BUILT FROM THE DUMP'S OWN GREEDY
ROWS. In D8 the support's greedy member was the D7 pool 1-best decoded through a different decoder
path, and the dump's regenerated greedy disagreed with it on 31,562 of 281,241 utterances (verdict 70);
D9 has no pre-existing pool, so building it from the dump makes the two agree BY CONSTRUCTION, makes
the same string arm 2's target and arm 3's greedy support member, and saves the ten-shard greedy decode
(~115 GPU-hours). The price, stated rather than enjoyed: D8's convention-sensitivity line is VACUOUS
here -- both conventions read the same row for every tag -- so its empty flip list is a tautology and
not a passed check, and the weight job's report says so in print. (2) ARM 2 IS `D7OnlineTrainJob` AT
WEIGHT 0, i.e. D8's control recipe, on the reading that "the incumbent refit recipe" means the recipe
D8's own A/B calls the one-hot special case of the drawn target; under the other reading (a fresh
`PsiAlignTrainJob`) arms 2 and 3 would differ in recipe as well as target.

TAU IS PINNED, NOT SOLVED: `D8WeightJob` solves the registered |median ESS - 3| rule on whatever bed it
is given, so running it unchanged here would re-derive the constant the registration pins.
`D9WeightJob` pins it at D8.1a's `tau_star` = 0.05 and prints the bed's OWN solution beside it (1.0), so
a bed that disagrees with the pin is visible in the artifact instead of hidden by it.

`S/d9_refit/D9WeightJob.uyKXr4ZiGj9R` is the gate between the dump and arm 3 and it rules NO-GO on
clause (a), so the table below is a census of the dump's support rather than a refit result. Every
number is the weight job's own.

| statistic | value | registered bar | reading |
|---|---|---|---|
| groups with live support | 281,241 of 281,241 | -- | the bed carries the read |
| members excluded by the safety valve | 0 of 877,334 (0.0000 pct) | 5 pct | valve idle |
| clause (a): median distinct support | 2.0 | >= 3.0 | **NO-GO** |
| clause (a), scorer-free variant | 2.0 | >= 3.0 | fires identically |
| distinct support, rollouts only | 2.0 | -- | not the greedy member's inclusion |
| clause (b): taus in the [1.5, 8.0] ESS band | 0.05, 0.1, 0.2, 0.5, 1.0 | >= 1 | pass |
| clause (c): median token R2 at tau_star | 0.4403 | < 0.5 | pass |
| tau_star | 0.05 (pinned) | -- | this bed's own ESS rule would pick 1.0 |

Distinct-support distribution over the 281,241 groups, which is the finding rather than the median:
1 member 92,995 (33.1 pct); 2 members 61,989 (22.0 pct); 3 members 37,445; 4 members 26,418; then
18,194 / 12,931 / 9,517 / 7,308 / 5,145 / 3,759 / 2,696 / 1,808 / 1,036 at 5 through 13. Mean 3.12,
max 13, against 13 candidates offered per group. Arm 3 refused to start on the registered guard
("D8.1b requires a GO weight artifact; this one reads 'NO-GO'") and stays in that error state by
ruling.

Arm 2's own-infeasible drop set is EMPTY, which is the STRICTEST reading of D7's bound, not a loosened
one: D7's constant names four rows of D7's OWN pseudo-text bed, and `D9OnlineTrainJob` registers this
bed's set after MEASURING it with the production text side and feasibility law before the value was
fixed -- 0 of 281,241 rows own-infeasible, tightest row `6065-109178-0010` at 77.9 pct of its frame
budget (612 of 786), D7's four named rows at 23-32 pct here. The audio side is literally the same store
(`PackUnitsJob.I0uzRMfUrKWC`) in both beds, so the whole difference is the text -- a fact about the
pinned checkpoint's decode, not about the recipe. D7's constant and every D7/D8 hash are untouched
(`D7OnlineTrainJob.WA1bqjXQtzeZ`/`j16rTskXF1QU`, `D8WeightJob.juRpzTNHKCSq`,
`D8ScorerRefitJob.2bQzhz6U1yHp`, `D9WeightJob.uyKXr4ZiGj9R`, re-read from the loaded graphs after the
edit).

D9.2, the two-arm read (`S/d9_refit/D9EtaReadJob.A7QvXl7VR7wl`; refit rerank
`S/psi_align_jobs/PsiAlignRerankJob.X7sDGLPgDFWm`, incumbent rerank `.cysJQBiP9iW1` (D9.0's own),
compare `S/psi_align_compare/PsiAlignPairedCompareJob.dMSa5z0knjLI`; speech-llm `c147014`): arm 2
against arm 1 on the ONE shared D9.0 draw, the two arms differing in `model_pt` alone, D8.4 machinery
and constants verbatim. Arm 2 finished at 2,421 steps with held NLL/frame 2.2550. The STOP clause
passed on its own terms: D9.0's structural census predicted 6,144 of 6,144 rollout rows alignable, and
BOTH arms scored 7,168 of 7,168 rows finite with 0 groups dropped.

| quantity | refit_1best (arm 2) | incumbent (arm 1) | paired |
|---|---|---|---|
| eta at T=0.7 | +0.1993 | +0.2303 | delta -0.0310 [-0.1545, +0.0923] |
| selection WER on shared groups | 0.1484 | 0.1481 | delta +0.0004 |
| spearman (context, never gating) | +0.3293 | +0.3238 | delta +0.0055 [-0.0763, +0.0873] |
| audio-free null margin (arm-internal, never differenced) | -0.0844 | -0.0456 | not a pair |
| rows scored / non-finite / groups dropped | 7,168 / 0 / 0 | 7,168 / 0 / 0 | 512 shared groups |

Shared mean WER 0.1508, shared oracle 0.1391, headroom 0.0116; the reader recomputes delta eta from the
WER identity and gets the same -0.0310. ONE BUILD DECISION, cheap to reverse and ratified: D9.2's config
PINS arm 2's finished model by path instead of importing D9.1's build, because that build constructs
arm 3 and an error job in the graph makes the manager hit sisyphus's interactive "Clear jobs in error
state?" prompt and exit -- importing it would have made D9.2 unrunnable without clearing the very job
the ruling protects. The pin is checked at graph-build time against D9.1's OWN alias
(`alias/sae/3e1/d9_1/refit_1best`), so a re-hashed arm 2 fails loudly rather than being read stale, and
it points at the `work/` job directory, never an `output/` alias.

WALL CLOCK FROM MEASUREMENT: D8.1a's finished shards ran 3,516 steps in 7:00 h at `max_seqs=8`, 41 GB
resident against 64 requested; D9.0's 512-utterance dump of THIS policy ran 256 steps in 21:23 at
`max_seqs=2`, i.e. 2.51 s per utterance against D8.1a's 0.895 s at four times the batch, the ratio the
launch-bound cost model predicts -- so the shards project near 7 h against the 11.5 h cap at D8.1a's
proven batching.

## Conclusions

Numbering is the log's own and is cited by the approach entries; corrections and overturns are kept
under the conclusion they correct, with the current reading named.

1. **The co-trained scorer did NOT go text-blind -- the hypothesis is refuted by its own instrument**
   (1). The usage gate RISES from 0.3331 at ep0 to a peak 0.6210 at ep6 (+86 %) and never falls below
   ep0, so the replay arm's 18.79 -> 46.71 needs a different explanation.
2. **What it did instead is lose its conditional entirely** (1). CE_true jumps 5.7444 -> 6.2045 after
   ONE sub-epoch and reaches 6.2938 at ep6, past the unit marginal 6.0072 and past uniform
   ln 500 = 6.2146 -- by ep6 the scorer is worse than a coin on gold pairs while its text-contrast
   grows. Co-training damage is scorer DRIFT off the gold domain, not text-blindness; a future
   trainability gate must read CE_true, which the usage gate alone would have passed.
   - UNPROVEN CAUSAL STEP (2026-08-09 audit): "co-training causes the collapse" is not established.
     No frozen-scorer control ever ran on the 100 h bed; the 10 h matched pair went the other way
     (frozen Goodharted 14.47/17.09, joint won 13.15/16.13); the two 100 h jointAR siblings lack the
     collapse signature. Temporal order (CE_true crossed the unit marginal after one sub-epoch while
     dev WER was still 18.79) supports scorer-first but is not attribution. Also, approach 1's "the
     only trainable-scorer run on record" is inaccurate -- the 100 h recon-only and hinge-only arms
     were also jointAR; the replay arm is the only one with per-epoch scorer forensics.
3. **Ranking noise is refuted a second time, within-group and at the loop's operating point** (3).
   psi_g_tc100 spearman 0.4959 (recon) / 0.5558 (shaped), frac_pos 0.93/0.95 over 512 groups, with no
   difference between all groups and the WER-spread-bearing subset. (Transcription: the shaped
   frac_pos is 0.9450 on the all-groups convention used elsewhere; 0.9452 is the spread subset.)
4. **Directional bias is confirmed but is mostly NOT contamination** (3). All three scorers pay for
   the filler at matched WER, including the never-contaminated gold-text control (beta 0.1673 on
   "to"). Only the differential -- psi_g_tc100 minus gold_enc50, +0.075 on "to", +0.063 pooled -- is
   attributable to the shared pseudo-text; roughly 70 % of the effect is a psi_align FAMILY property,
   so a round-0 text repair can address at most the smaller part.
5. **Corpus size is not the axis** (3). psi_g_seed, with 10x less of the same pseudo-text, has the
   LARGER bias (0.2664 vs 0.2425) and the worse ranking (0.4696 vs 0.4959).
6. **Group blindness is real but partial, and is the binding constraint on "to"** (3). Only 23 % of
   the groups carrying "to" hold a "to"-free member (9 % for the suspect set as a whole), so in ~77 %
   of live groups no scorer of any quality can steer off the filler. Not the plan's "coverage ~ 0"
   fork, but it caps what any scorer-side repair can buy and makes the sampling-side contingency a
   co-requirement rather than an alternative.
7. **A curated refresh has an admissible external selector** (3). The base-LM prior in its live
   units-normalized form clears the D0(e) bar at 0.5020 [0.4737, 0.5308]; the suspect count is weakly
   admissible at 0.1855. psi's own duration channel is not a selector (CI straddles zero), which also
   confirms psi's ranking is not a length artifact.
8. **The filler is cheap to INSERT, not cheap to write -- token-specific but scorer-invariant** (5).
   Inserting "to" costs 0.0274 nats/frame against 0.0859 for an LM-drawn word in the same slot
   (discount 0.0584 [0.0537, 0.0634], 0.2201 at k=4), while writing it OVER a word costs the same as
   any frequent word (+0.0036).
   - WRONG IN PART (2026-08-07 audit; CURRENT): "scorer-invariant" does not survive -- the LM control
     is drawn with no length matching (~2.7 emitting BPE states per draw vs 1 for "to"), 53-81 % of
     the discount is state-count-attributable, the surviving residual (0.011-0.027 nats/frame) is
     scorer-DEPENDENT, and what is scorer-invariant is the lattice's ~0.031-0.035 nats/frame price per
     inserted emitting state. The cheap-insertion exploit is therefore open to EVERY minimal-state
     word -- contamination chose which word, not whether.
9. **D1's pre-registered power check FAILS: no filler statistic separates psi_align^G from the 10 h
   true scorer** (4, 5). All three arms agree on the insertion discount, the substitution discount and
   the suspect state mass (1.90-2.19 %) within their CIs; `ce_loo` separates them (2.72 / 3.02 / 3.13)
   but in the direction of the pseudo-text DOMAIN.
   - WRONG (2026-08-08, approach 10; CURRENT): the power check fails only for the frequency-drawn
     control -- on the state-matched one psi_align^G separates from the gold-text control decisively
     and in the expected direction (0.0172 vs 0.0031, paired -0.0141 [-0.0174, -0.0108] on the shared
     1442), so the instrument has power and it was the CONTROL that was blunt. The ~70 % family-share
     reading from D0(c) is untouched.
   - NUMBER CORRECTION: gold_enc50's held ce_loo is **3.1385** (`PsiHeldNllJob.ag5DZ3A2Gd1K`); the
     3.1274 in approach 4's table is the probe job's 1442-pair value transcribed into the 1493-pair
     table. The row's derived columns already use 3.1385, so ordering is unaffected and this
     conclusion's "3.13" reads 3.14. UNRESOLVED: the table cell was never corrected.
10. **The mechanism is an insertion/deletion asymmetry of the lattice, not a text defect** (5).
   Deleting a word costs 0.3336 nats/frame and inserting the filler 0.0274 -- a factor 12 -- so
   against any real alternative the policy has, padding is nearly free. The only existing counterweight
   is the LM prior at `lm_prior_norm="units"`, of order 0.01 nats/frame for this token at the bed's 338
   frames per utterance, estimated from the LM-corpus unigram rate rather than measured contextually.
11. **Raising the sampling temperature buys contrast but degrades the oracle** (6). T=0.7 -> 0.9 lifts
   steerable coverage 0.1949 -> 0.3382 but conversion falls (0.835 -> 0.642) and the ORACLE WER rises
   0.1071 -> 0.1496, so T=0.9 is not the free operating-point move the presumptive read treated it as.
12. **The asymmetry is arithmetic and its size is frames per state, which no D2 arm but `d2_states`
   can reach** (5, 8). Deleting a word orphans T/U = 4.88 frames per state removed while an inserted
   word is absorbed in half a frame by the skip arc, so the price ratio carries a structural factor
   2 T/U = 9.77.
   - WRONG IN PART (2026-08-07 audit): each number reproduces but the sentence switches statistic
     bases (+0.2461 is the pooled per-frame NLL delta, not the table's per-utterance del_1 0.3336; the
     12.16/3.89 ratios are on the ce_loo basis where the NLL basis gives 9.88/4.24), "3.1x above it"
     inverts the comparison (3.89 is BELOW the 9.77 floor; 3.1 is filler/LM), "charges every one of
     them" over-counts (the emission delta +0.2558 exceeds the deletion total; the NLL-minus-emission
     residual is transitions PLUS alignment entropy and is negative for deletion), and per
     removed/added STATE the del/ins ratio is 5.3-7.4, below the claimed floor -- so "the filler sits
     at the floor" is a state-count artifact (2 states vs the mean deleted word's 3.76). WHAT SURVIVES,
     verified numerically: deletion orphans a chars_per_state-INVARIANT frame count (18.3 vs 18.4
     across cps 1.5/0.5) while an inserted word's state count scales ~2.5x, so the price ratio falls
     ~2.5x under `d2_states` and no corpus or contrastive arm moves it.
13. **All four D2 arms finished and NONE separates from the incumbent on the frequency-drawn
   statistics** (8). Every discount step is smaller than the 0.005 bootstrap half-width; beta_to moves
   0.2425 -> 0.2232 for the repaired corpus and UP for both contrastive arms; steerable coverage moves
   by at most three of 467 live groups; the incumbent has the HIGHEST in-group spearman (0.4959) and
   the lowest sel_wer (0.1380). Read literally, approach 9's rule returns NO WINNER.
   - WRONG IN PART (2026-08-08, approach 10; CURRENT): "no candidate differs in any measured way"
     holds only for the confounded discount and the rollout statistics -- on the state-matched discount
     three of four arms reduce the filler's insertion advantage with paired CIs excluding zero
     (`d2_contrast` -0.0090, `d2_both` -0.0075, `d2_states` -0.0047 at k=1), and the mechanism-level
     arms do it while `d2_rate`, the pure text repair, does NOT (-0.0018, n.s.). The refutation of
     "round-0 repair is enough" stands and is sharper -- the corpus arm is the one that fails -- but
     "no winner" does not: see conclusion 15.
14. **`d2_states` splits the two insertion statistics exactly as the state-count confound predicts**
   (8, 12). It is the only arm to improve ALL FIVE corruption ladders (filler_ins 0.6552 -> 0.8540,
   lmins 0.8620 -> 0.9178) yet its frequency-drawn insertion discount is 2.5x the incumbent's.
   - RESOLVED (2026-08-08, approach 10): the ladder was right and the discount was artifact. Under the
     state-matched control `d2_states` charges +0.0125 against the incumbent's +0.0172 -- a real
     reduction -- so the entire apparent regression was the frequency-drawn pool averaging 8.16 states
     against its four-state filler.
15. **On the statistic the amended rule names, D2 HAS a winner, and it is the mechanism arm** (9, 10).
   Eligibility leaves `d2_contrast` and `d2_states` (`d2_rate` is worse on three of five ladders
   paired; `d2_both` sits 0.013413 below the pre-loop `text_explained_loo` floor); both reduce the
   state-matched insertion discount with paired CIs excluding zero, and the larger reduction is
   `d2_contrast`'s (0.0172 -> 0.0082 at k=1, 0.0561 -> 0.0323 at k=4). Its edge over `d2_states` is
   significant only at k=4 (-0.0096 [-0.0187, -0.0007]). The rollout tiebreaker disagrees mildly
   (beta_to 0.2469 vs the incumbent's 0.2425), which is the one tension: a controlled text-side edit on
   held pairs and an observational partial effect on policy rollouts are not the same measurement, and
   only the first is what the rule selects on. The winner does not depend on reading the
   `text_explained_loo` floor as psi_g_tc100's own value.
   - OPEN (2026-08-08 audit): `d2_states`' k=1 zero-exclusion FLIPS with the bootstrap seed (t-test
     p=0.046), so "both CIs excluding zero" overstates at k=1 (solid at k=2/k=4), and
     d2_contrast-over-d2_states at k=4 excludes zero by only 0.0003-0.0006. The winner also turns on
     two unpinned clauses -- see Open findings.
16. **The external LM prior is filler-NEGATIVE at matched WER, and the only audio-conditioned view is
   the one that pays** (11). `lm_prior_units` clears both D4 bars (0.5020; -0.0937 CI excluding zero),
   refuting the plan's premise that an external LM would favour the filler; `ar_recon` ranks barely at
   all (0.0944) and PAYS for "to" (+0.0716, CI excluding zero), so the two admissible views are both
   text-side -- a residual the plan's "not audio-free" clause anticipated and no measurement on this
   bed can remove.
17. **The frozen repaired scorer moves the filler on the G-track and, at sub-epoch 2, the WER with it
   -- but bar 2's SHARE normalization is blind to the move** (9). At sub-ep 2 `d2_contrast` beats the
   incumbent on both arms while cutting "to" insertions 3539 -> 3043 and 3817 -> 2096, yet the shaped
   arm's suspect share barely moves (0.871 -> 0.862) because total insertions fall in proportion, and
   the recon arm's share falls (0.556 -> 0.408) only because a non-word fragment ("st", 856) takes the
   vacated mass. Superseded as a verdict by conclusion 31.
18. **The incumbent AR reward's ranking replicates from the D0 sample to the whole bed, and its argmax
   pick is worse than a random one** (12). On all 28 539 utterances at T=0.7, G=12 its within-group
   spearman is 0.0959 against the D0 512-utterance read of 0.0944, but eta = -0.1103 -- picking by the
   reward gives 17.72 % WER against a random pick's 17.06 % and an oracle's 11.08 % -- while greedy
   decoding scores 13.86 % and the gold text earns a margin of 0.0001 over the samples. No refresh
   round may curate with the reward itself.
19. **Two-view curation reaches 79 % of the bed, but its picks are dirtier than the anchor they join**
   (12). 22 667 of 28 539 groups yield a candidate with both advantages positive (83 219 members
   qualify, one kept per utterance) for a 51 206-row pool at a 55.7 % anchor share, yet the curated
   half's suspect rates run above the anchor's ("to" 0.0462 vs 0.0275, "buy" 0.0031 vs 0.0001).
   Qualification: "all twelve are worse than the repaired round-0 text" is an aggregate reading,
   literally true in ~56-72 % of groups.
20. **The collapse is pure over-generation, and a suspect-SHARE bar is blind to it** (13). %Corr RISES
   89.07 -> 91.55 (dev-clean) and 87.31 -> 88.92 (dev-other) while %Del falls 4.46 -> 2.14 and %Ins
   goes 5.98 -> 38.26 at hyp/ref 1.015 -> 1.361; the suspect set's share of insertions is flat at
   0.045-0.066 because the added mass is generic function words. Any bar normalized by total insertions
   cannot see the insertion blow-up, so D4' must read insertion COUNTS. Qualifications: the 6.4x is
   dev-clean only (dev-other 5.07x, pooled 5.66x); the ep0 insertion baseline is 56.5 % ten runaway
   repetition-loop utterances against 2.1 % at ep4, so the collapse trades rare loops for broad diffuse
   padding; the five named function words carry only ~18 % of insertion mass.
21. **The scorer's preference migrates from gold to its own padded output, on top of an already-dead
   conditional** (14). `self_pref` goes -0.0139 -> +0.1267 with the best-scoring column moving from
   gold at ep0 to the arm's own ep4 decodes at ep4 -- the co-collapse the phase asked about.
   - WRONG IN PART (2026-08-09 audit; CURRENT): "at or above BOTH floors" holds only for the unit
     marginal -- 20 of the 24 ep >= 1 cells sit BELOW uniform ln 500 (max shortfall 0.1187) and the ep4
     within-row spread is 0.14205, not <= 0.14. The inference survives in the weaker form "above the
     unit marginal and within 0.12 nats of uniform".
22. **With the rollouts held identical, the reward's ranking utility goes NEGATIVE after one
   sub-epoch** (15). eta at T=0.7 falls +0.2246 -> -0.1185 at ep1 and -0.2792 at ep4 (sel_wer
   0.0905 -> 0.1288 against an unchanged mean 0.1076) while `std_wg` GROWS 0.018 -> 0.026 -- not a dead
   band but a reward that actively prefers the padded sample; the loop's own gradient condition
   (spearman > 0 and sel_wer < mean_wer) fails from ep1 on. Direction: this bed's update rule IS
   D5(b)'s continuous joint psi, so D5(b) asks how fast it collapses, not whether -- one sub-epoch is
   the number to beat, and such an arm needs a within-loop read of eta or CE_true to be informative.
23. **The label-free selection rule and the health screen disagree, and the screen is what makes the
   fork defensible** (16). Dev reward ranks sub-ep 3 first and sub-ep 2 third, i.e. anti-correlated
   with WER over the four sub-epochs; the minimal-state count vetoes 3 and 4 at +32 % and +38 %,
   leaving sub-ep 2, which WER confirms at 5.34 / 9.50. **All three update-rule arms fork from
   `vhyvv2waeU16` sub-epoch 2.** The plan's own named statistic would have missed it: words per
   utterance moves only +2.11 % / +2.31 %. Reading the exploit CLASS by absolute count is what carries
   the screen.
24. **The joint arm's psi CE and its reward agree numerically from step 1** (17): `psi_ce` 2.94-3.48
   against `reward_recon` -2.95 to -3.48 on the same steps. A wiring check, not a result.
25. **The joint rule does not fit this bed's node at the parent's settings** (17). Measured over 76
   steps: 12.39 s/step against the parent's 4.81 (2.57x), projecting a sub-epoch at 11.2 h against the
   11.5 h SLURM cap; it did not get that far -- CUDA OOM at step ~72, GPU at 95 GiB, because the parent
   already peaks at 80.9 of 95 GiB at `max_seqs` 8 / `group_size` 12 and the CE channel's alignment-DP
   autograd graph over all 96 rollouts has nowhere to live. Every considered fix trades against the
   property the control depends on.
   - WRONG (2026-08-11; CURRENT): halving `batch_size` to 1e6 with `accum_grad_multiple_step` 2 -- a
     fix this list did not consider, preserving the effective batch, the group, the update count and
     the 1:1 sub-epoch comparison -- makes the arm fit on both axes: memory flat at 48.1 of 95 GiB and
     sub-epochs of 35982 s and 33252 s against the parent's 15693 s (2.29x, 2.12x), inside the cap.
26. **The excess-mass suspect derivation is empty on this bed** (18). At `min_excess` 0.002 no word
   qualifies (largest "and" 0.001345). The instrument is not broken, it is out of range: it prices a
   rate difference against the LM corpus, and this policy decodes at 5.34/9.50, so the ~140 extra "and"
   tokens are 0.13 % of a 106 k-token corpus. `neg_n_suspect` therefore cannot be a curation view here.
   Superseded in scope by conclusion 30.
27. **The round-0 gold scorer on the best bed under-charges INSERTIONS by ~6x and discounts the
   minimal-state class inside that, before any refresh** (18). An inserted word costs +0.069 against
   +0.405 for a substitution and +0.430 for a deletion, and within insertions the minimal-state word is
   +0.021 cheaper than a frequency-matched LM word at the same slot (CI [+0.0164, +0.0250], frac>0
   0.563). The sign FLIPS for substitutions (-0.016, CI excluding zero), so this is not generic filler
   affinity. Insertions are also where the ladder is least monotone (0.658 against 0.735-0.784). Bears
   on D5's attribution question: the price is wrong at ep0 of the loop, so this bed's insertion exploit
   needs no scorer drift to explain it.
28. **The G-track refresh round moves the gate statistic further than any D2 arm, but only under the CI
   reading of the ladder floor** (12). The state-matched discount falls 0.0172 -> 0.0064 at k=1 (paired
   -0.0108 [-0.0140, -0.0077], p 0.000), the gap widens with k (-0.0422 at k=4) at no held cost; no
   ladder is significantly worse but four are nominally lower, so the point reading elects nobody and
   the CI reading elects `r1`. Curation is worth about a fifth of the reduction (the uncurated recipe
   already reaches 0.0082, at a hair better held ce_loo 2.7139). The contrastive term is not what did
   it: contrast/utt is 0.8971 in its first active epoch (ep5), 0.0452 by ep6 and exactly 0.0000 from
   ep21 on, so the last third of the refit is pure NLL. **It moves the text-side preference and not the
   topology**: the insertion ASYMMETRY goes 0.3062 -> 0.2968 -> 0.2827, a 7.7 % shift against the
   discount's 63 %, so after the refresh the filler is no longer specially cheap to insert while
   inserting anything at all is still ~7x cheaper than deleting. The degradation this ladder exists to
   stop is over-generation of generic words (20), which the asymmetry prices and the discount does not.
29. **On the best bed psi ranks its own rollouts well, but its advantage over the audio-free null does
   not clear the pre-registered margin** (18). Within-group spearman +0.3407 against the 0.17 bar with
   the length-only null at -0.0074, but the audio margin is +0.0229 [-0.0036, +0.0522] (P(>0) 0.953)
   and gap_true +0.0089 against +0.0248 -- two of three G3 bars FAIL on 28538 groups. The row-level
   parity check is exact (max |online - offline| 0.000e+00 on 512 of 512), so nothing in the id round
   trip, the batching or the flooring has drifted.
30. **This bed has no admissible audio-conditioned curation view, by measurement rather than
   exhaustion** (18). On 647 live groups psi's own score is filler-POSITIVE at matched WER -- partial
   beta +0.2254 [+0.0817, +0.3239], +0.2029 [+0.0558, +0.3000] once shaped -- so it fails clause (f)
   outright. `lm_prior_units` is the only signal clearing both clauses (spearman 0.5171, beta -0.3018
   [-0.4703, -0.1661]) and it is the audio-free one the rule forbids to curate alone; `n_tokens` ranks
   WER negatively (-0.1589) and `lm_prior_tokens`' affinity CI spans zero. Supersedes conclusion 26's
   reading that the second view was merely missing: the audio-conditioned candidate exists and is
   disqualified. The (f) block reads the label-derived watch class in its monitor role only.
31. **A refit scorer's WER advantage is one sub-epoch wide and does not survive the next one** (9).
   `d2_contrast` runs 13.57/19.69 -> 12.68/17.57 -> 13.54/18.56 where the incumbent runs
   13.42/18.75 -> 13.91/18.91 -> 13.49/18.81: WORSE at sub-ep 1, better by 1.23/1.34 at sub-ep 2 --
   the incumbent's own worst sub-epoch -- and level on dev-clean by sub-ep 3. Both arms wobble inside a
   13.4-13.9 band, so conclusion 17's sub-ep 2 read was taken against a bump. The recon arm keeps a
   real gap (32.94/37.91 against 33.54/39.74, 5623 insertions against 8724) but both arms there are
   diverging, so it prices a slower divergence, not a fix. What the cancellation left unmeasured is the
   only thing that would settle it: the incumbent COLLAPSES at sub-ep 4 (13.49 -> 17.99, substitutions
   2894 -> 5301) and the repaired arm is held at sub-ep 3.
32. **One sub-epoch of co-training is the largest WER gain anywhere on this bed, and the next one
   destroys it** (17). 5.12/9.27 at sub-ep 1 against the matched frozen control's 6.56/11.15 -- better
   than the parent's all-time best 5.34/9.50 -- then 17.35/21.97 at sub-ep 2 and 41.8/50.9 at sub-ep 3.
   Both moves are the insertion channel: 385/630 at sub-ep 1, far BELOW the frozen band's
   1182-1415/1592-1794, then 6114/6450, then 21,406 on dev-other, while substitutions move far less.
   The collapse D4's offline-only shape was built around is real and now carries the frozen control it
   was missing, but it is a cliff after one good step rather than a decay, which argues for GATING a
   discrete refresh rather than against refitting. The scorer-side mechanism is now measured too
   (approach 17): gold-pair ce_loo degrades 2.6343 -> 2.7928 -> 2.9771 while own-decode ce_loo improves
   2.6270 -> 2.4726 -> 2.2994 over the same three sub-epochs.
33. **Round 1's uncurated refit fits the frozen held set materially better than the incumbent but
   fails the gate on one ladder** (19). Held ce_loo 2.7614 -> 2.6432, `text_explained_loo` up +0.1182,
   the two INSERTION ladders' paired spearman up +0.0434 [+0.0340, +0.0531] (filler) and +0.0288
   [+0.0194, +0.0384] (LM word), within-group spearman on the fork dump +0.3399 -> +0.3621, while
   `lmsub` falls -0.0058 [-0.0091, -0.0025] -- the single clause v2 (iv) reads as "not decreased" -- so
   the clause table returns NO WINNER under both readings. Whether an unweighted ladder rule should let
   a 0.006 substitution loss outweigh a 0.043 insertion gain is a gate-design question for the planner.
34. **Re-pricing a trained scorer's arcs cannot move insertion, and the topology it was meant to gate
   is free** (20). The strongest of twelve settings lifts the k=1 insertion price 1.09x (+0.0693 ->
   +0.0755) against the gate's 2x and a 4-nat skip bias only 1.03x, because a duration charge falls on
   the clean text's own short states as heavily as on an inserted word's. The feasibility statistic
   gating rung 3 clears by a wide margin: 6.64 frames per content state, 3 infeasible rows in 59 878 at
   d_min=2 and 7 at d_min=3, so the plan's ceiling ("mean T/U ~4.9 caps d_min ~2 for the tail") was
   derived on a tighter number than this corpus shows and d_min=3 is a live dial.
35. **The minimum-duration topology passes every D6 clause and repairs round 1's failing one, but the
   acceptance rule's winner test asks a question the phase deliberately rescales** (20). d_min=2 clears
   all four bars -- spearman +0.3621 -> +0.4357, held ce_loo 2.6432 -> 2.1620, k=1 insertion price
   2.86x the incumbent's growing in k, `filler_ins` monotonicity 0.658 -> 0.853 from last place to
   first -- and lifts `lmsub` to 0.9572, above both the comparator's 0.9479 (the single clause that made
   c33 a no-winner) and psi0_gold's 0.9538, while online/offline parity holds to 5e-07. Yet the clause
   table returns NO WINNER because its winner test wants the state-matched insertion discount to FALL
   and this arm is neutral on it (+0.0047 [-0.0009, +0.0101], p=0.096) -- a delta-ce_loo LEVEL, and
   every edit price on this arm is ~2.8x the incumbent's, so read as a share of what an insertion costs
   the same measurement falls 11.3 % -> 7.1 %. Which reading binds is the planner's to pin; it also
   decides eligibility (eligible under CI with 0 ladders worse, not under point with 2 worse, both CIs
   spanning zero).
36. **The corruption margin hurts on its own and drags the topology down with it, refuting the
   registered expectation that rungs 2+3 together are the shape** (20). Alone it fails clause (iv)
   (`filler_ins` monotonicity 0.692) and multiplies the matched insertion discount six-fold
   (+0.0094 -> +0.0578, CI excluding zero); added to the topology arm it costs three ladders at CIs
   excluding zero and drives the discount to +0.3150. The mechanism is in the probe rows: the term
   raised the price of exactly the LM-drawn control words it trains against (`lmins_m` 2.9x) more than
   the filler's (2.4x), so it learned its own negative distribution rather than insertion in general.
   - CORRECTION 2026-08-12: the original closing clause called the combined arm's 0.699 ins/del ratio
     "indiscriminate inflation rather than discrimination", which its re-rank (measured after the
     conclusion was written) does not support -- it ranks rollouts BEST of every arm here, spearman
     +0.4441 and eta +0.3392 against d_min=2's +0.4357 / +0.3296. Rung 2 mis-prices the filler against
     a matched control while still improving the statistic the loop consumes.
37. **The min-duration topology transfers as a FIT but fails the G-track gate on the substitution
   ladder** (21b). `r1_mindur` fits the frozen held set far better than either comparator (ce_loo
   2.3774 against `r1`'s 2.7168 and `psi_g_tc100`'s 2.7198) and wins both insertion ladders by a wide
   margin (filler_ins +0.0849 [0.0635, 0.1058], lmins +0.0488 [0.0362, 0.0616] against `r1`), but it is
   significantly worse on filler substitution (-0.0136 [-0.0194, -0.0077]) and so is ineligible on both
   readings; NO WINNER against either incumbent, which separates fitting the held set from being safe
   to hand the loop. Against `psi_g_tc100` it is also CI-worse on the del ladder (-0.0097 [-0.0158,
   -0.0039]) and filler_sub (-0.0146), while its matched insertion discount is CI-LOWER at k=4 (-0.0182
   [-0.0276, -0.0085], p=0.000; k=1 n.s.) -- the topology's insertion-pricing gain grows with k while
   eligibility fails on the substitution/deletion side.
38. **Running psi's alignment recursion on the GPU instead of in python costs nothing in fit and makes
   a per-round refit affordable** (21b, replicated). The same d_min=2 refit -- same corpus, same
   hyperparameters, only the forward-backward moved into RETURNN's CUDA fast-Baum-Welch kernel --
   reaches best held_nll 2.3160 at epoch 23 against the python path's 2.3186 at epoch 23, in 0.78 h
   against 5.94 h (94 s against 713 s per epoch, 7.6x).
39. **The min-duration scorer as the live reward PASSES its pre-registered confirmation outright, and
   the separation widens to the end of the run** (21a, eight of eight sub-epochs). The control's
   sub-epoch-3 regression does not merely shrink -- the swap-in arm never regresses at all, improving
   past the fork point (5.34/9.50) to 4.68/8.64 one sub-epoch later and holding 4.73/9.31 at sub-epoch
   10 against the control's 6.46/11.41, with dev-other insertions less than half the control's (933
   against 1964). The insertion exploit the whole D6 ladder was built to close is closed in the live
   loop, on the reward side alone, with no change to the policy, the data or the schedule.
40. **The homophone arm clears its admission floor, but the reachable mass is thin and concentrated**
   (23). 7.68 % against the 5 % floor; a uniform draw rewrites only 4.02 %, eight classes carry 60.5 %
   of all rewrites, and 131 of the 139 classes the corpus uses have just two members.
41. **The periodic arm has never once refreshed its scorer** (22). Its gate returned KEEP at rounds 2,
   3 and 5 and the two-consecutive-failure stop rule fired at round 4 -- where the binding CI reading of
   the clauses actually PASSED and was overridden -- so legs 2 to 8 all run the round-1 scorer, and the
   arm's comparison against the one-shot swap-in measures the shard rule and the Adam restarts, not
   refresh frequency.
42. **WRONG after the six-leg read** (26). As written: "on the label-free init a refit at every
   boundary is the first loop arm to go below the no-loop init at a matched sub-epoch, and it is still
   not the best scorer there" -- at sub-epoch 2 it reaches 12.85/17.89 against theta_0^G's 13.89/18.34,
   where both frozen-scorer arms sat at or above the init, but the frozen REPAIRED scorer reaches
   12.68/17.57 at the same point.
   - CORRECTION 2026-08-20: only two legs were available when this was written. The six-leg prefix
     makes that gain transient; REPLACED by conclusion 54.
43. **Sampling already proposes spelling variety, but almost never proposes the spelling that is
   missing** (25). 23.43 % of homophone-bearing groups already hold two spellings of one class, so the
   reward has within-group variance to steer on there today, while only 0.82 % ever contain a spelling
   the scorer's training corpus does not have -- so the direction an SFT support change uniquely
   reaches is the repair direction, not the diversity one.
44. **The homophone arm clears its admission bar overall and in the diversity direction, and fails it
   in the repair direction** (25). Ratio 1.26 over all swaps and 1.27 on attested-spelling swaps, but
   0.82 on swaps into a spelling the scorer never trained on; per class the split is real rather than
   uniform (knot/not 3.70 and too/two 3.88 against wood/would 0.49 and their/there/they're 0.54), and
   every delta is ~0.01 nats/unit in absolute size, small enough that the within-group reward spread has
   to be read before any of it is called steerable.
45. **No short-spelling bias in the reconstruction term** (25). Every in-class substitution is
   penalized, and swaps to a SHORTER spelling are penalized MORE than swaps to a longer one (median
   -0.0135 against -0.0073) -- the opposite ordering from the per-state orthographic-length price the
   arm was registered to watch for.
46. **The homophone init costs 3.11 dev-other WER, and essentially all of it is the augmentation
   reproducing itself rather than a degraded model** (27). 16.67/21.45 against 13.89/18.34, with 96.7 %
   of the extra NET errors substitutions within a homophone class, at a plain-WER price of 78 % of what
   full reproduction of the uniform draw would cost. Outside the classes the two inits are within noise
   (+53 errors, CI [-70, +173]).
   - CORRECTION 2026-08-18: as first written this compared the realized rate to the 4.04 % of TRAIN
     tokens the augmentation rewrote, which is not like-for-like against a dev reference-token rate; the
     comparable figure is 4.58 % of reference tokens, and "96.7 %" is the share of NET extra errors
     (92.2 % of extra substitutions). The conclusion itself is unchanged.
47. **That price is not accommodated by the arm's pre-registered primary read** (27). The arm must stay
   within 0.3 dev-other WER of D6-PERIODIC/GAN at every matched leg; its init starts 3.11 behind, so leg
   1 fails that clause unless one GRPO leg closes ten times the margin. Whether the arm still runs is
   the planner's and the user's call.
48. **The reward's LM prior knows which spelling is right; the reconstruction term is a
   near-direction-blind preference for the text already sampled, and at lam_lm 1.0 the two cancel
   exactly where correction is needed** (28). The prior prefers the reference spelling on 90.6 % of
   swaps toward it and only 1.6 % of swaps away; reconstruction prefers the swap on 17.9 % toward and
   25.4 % away, i.e. it mostly opposes changing the sampled text whichever way the change runs. Their
   sum rejects wrong spellings well (0.063) and is a coin flip on right ones (0.529). Raising lam_lm is
   the mechanical fix and is what the arm's own audio-free-null GUARD forbids.
49. **That coin flip is one homophone class, not a property of the reward** (28). `buy`/`by`/`bye` is
   67.6 % of the toward-reference swaps at composed 0.446; the rest read 0.702. Why that class behaves
   so differently is UNEXPLAINED -- the corpus-coverage hypothesis was tested and refuted -- and it is
   68 % of the sample any inference from this read rests on. Dropping it is a post-hoc slice.
50. **On the distribution that matters the composed reward is NOT at chance -- it points at the correct
   spelling 82.5 % of the time** (29). Measured on the homophone policy's own errors rather than the
   plain policy's, with every one of the eight damage-carrying classes above chance and the away
   direction correctly rejected at 0.140. Conclusions 48-49's 0.529 is SUPERSEDED as a prediction for
   this arm: it was measured on a near-orthogonal class distribution, and the correctly-weighted number
   sits near the top of the 0.51-0.90 bracket rather than at its floor.
51. **Refitting the scorer on the policy's own decodes ENTRENCHES the spelling error rather than
   equalizing it, and the registration's mechanism claim has the wrong sign** (29). Holding dump and
   swaps fixed, the reconstruction term's toward-reference rate falls 0.684 -> 0.357 under this arm's
   own refit; paired counts 6:1 against (3480 vs 598) and 95 of 121 classes move. The composed reward
   survives at 0.825 only because the LM prior at 0.970 outweighs it -- and that prior is an audio-free
   reader, so the composed rate sits 14.5 points BELOW the text-only null. Length-matching refutes the
   cheaper explanation (-0.284 at equal character count), so it is spelling-specific learning and not an
   orthographic-length price. In the deployed reward the cost is -0.069, not -0.327. A quantified
   instance of the standing G-track diagnosis -- a scorer refit on its own policy's output rewards that
   policy's correlated errors -- and not specific to the homophone arm, since every arm in the
   D6-PERIODIC family refits the same way. The open risk it names is compounding: each round refits on
   decodes the previous round's entrenched scorer helped produce.
52. **Periodic outer updates avoid the catastrophic continuous-joint failure, but they do not beat a
   good frozen scorer on the D-track** (17, 21, 22). The closest continuously trainable-scorer arm reads
   5.12/9.27, 17.35/21.97 and 41.78/50.88 over three banked sub-epochs, with dev-other insertions
   growing 630 -> 21,406. Fresh periodic is 4.97/8.88, 4.65/9.02 and 5.28/9.27 at its first three legs,
   so holding the scorer fixed within each leg avoids same-step collapse; it then worsens to 7.42/12.68
   by leg 5, never beats the one-shot scorer's 8.64 best, and trails both the matched one-shot scorer
   and the original frozen control at that point. A useful timescale, not a single-variable causal
   effect: the joint arm also differs in scorer topology, data partitioning, batching and optimizer
   continuity.
53. **Carrying scorer weights across periodic refits is harmful on this bed** (24). Fresh and warm are
   close through leg 3, but warm reaches 12.18/19.33 at leg 5 against fresh 7.42/12.68. The separation
   is an insertion failure: warm dev-other insertions grow from 479 to 5,874 while substitutions stay
   near 3.6k. The trajectories are not ended, but the completed prefix rejects warm inheritance as a
   stabilizer at this operating point.
54. **The plain GAN periodic gain is small, transient, and does not establish a recency benefit** (26).
   Leg 2 improves the no-loop init by only 0.45 dev-other, missing the registered 0.5 bar, and the arm
   worsens to 18.38/24.01 by leg 6. At the three matched points it is not decisively better than the
   frozen repaired scorer; later deterioration is substitution-led. No same-init continuously
   trainable-scorer arm exists, so D5(b)-b is not a causal control for this variant.
55. **The live GAN+HOM loop rapidly removes the augmentation's spelling damage despite the fixed-dump
   scorer-entrenchment diagnostic** (26, 27, 29). Its class-internal dev-other substitutions fall from
   1,827 at init to 130 after one leg and 105 after three; total WER improves from 16.67/21.45 to
   12.80/18.08 and catches the plain periodic trajectory at leg 3. Conclusion 51 remains a valid
   statement about the reconstruction scorer on controlled swaps but does not predict the realized
   policy direction under the composed reward, whose Qwen3 LM term dominates homophone spelling. The
   midpoint and final registered reads remain outstanding.
56. **WRONG in its description of the registered surface (2026-08-20 audit)**: "D7's registered
   eight-donors-per-chapter-stratum construction is impossible." The external band and donor-capacity
   law were not registered; K=4 means two donors per chapter stratum.
   - CORRECTION (CURRENT): D7.0a proves that the original external donor statistic is not executable as
     written. Only 276/1,500 immutable sources meet even the all-eligible raw K=4 degree requirement, so
     full-`E_all` K=4 is impossible; zero meet the conservative eight-donors-per-stratum diagnostic. The
     latter does not establish exact second-quartile support, because the rank, boundary and tie laws
     are themselves unregistered. Later filters can only shrink a chosen surface. This triggered the
     prospective D7-v2 amendment (frozen 2026-08-21).
57. **D7-v2 / D7.0b fails its preregistered training-support floor and is structurally unresolved**
   (31). 56 admitted rows from two speakers against the required 6,778 rows and 201 speakers, with an
   independent necessary-core calculation bounding any exact solution at 120 rows from four speakers --
   so the floor cannot be met by this registered graph, rather than merely exposing a poor optimizer
   solution. The intended fail-closed scientific gate, not a scheduler or convergence failure. Per the
   frozen gate this permanently closes the offline-graph branch; it does not constrain the corrected
   online D7.
58. **USER-DIRECTED correction of active D7: retire offline donor graphs and test the reverse loss with
   online negatives on the full 960 h bed.** This does not reinterpret conclusion 57 or claim an
   experimental win. The corrected D7-GAN-SEQDISC uses all 281,241 theta_0^G-greedy pseudo-pairs, one
   dynamically resampled same-speaker duration-windowed donor per anchor (reciprocal duration ratio
   0.8-1.25, closest-duration fallback only when that window is empty), and no chapter/Q2, nuisance,
   capacity or regularity constraint. A policy leg becomes eligible for separate launch authorization
   only after its label-free fixed-final gate.
59. **The D8.0 binding clause cannot be read on either frozen dump: its exclusion rule is not the law
   those dumps were scored under** (34). Clause (a) excludes structurally infeasible candidates at
   `d_min=2`, but both dumps predate the standing min-duration topology, and the theta_0^G artifact
   additionally joins the ~12.5 Hz pooled unit store `MergeUnitsPklJob.hJmZtbPDa2hd` (median length 169)
   rather than the raw 50 Hz store
   the pinned weight scorer uses (median 674/695). On its binding T=0.7 slice the reader calls 5,096 of
   5,730 scored members infeasible while the artifact's own scorer returned finite scores for them --
   including all 512 greedy rows -- so the exclusion is a property of the instrument. The same law costs
   the fork-epoch dump 18 of 101,190 members. The registered read returns UNRESOLVED, not the NO-GO the
   exclusion alone would produce.
   - SUPERSEDED IN SCOPE (2026-08-22 ruling, verdict 62): the frame diagnosis is confirmed, but "cannot
     be read on either frozen dump" is too strong -- the clause is readable on the existing dumps once
     the exclusion is joined to the operative raw 50 Hz store, with no new dump and no scorer forward.
60. **Which reading clause (a) takes decides it outright, in opposite directions** (34). On the binding
   slice the median distinct support is 0 of 13 with the exclusion applied and 12 of 13 without it,
   against a threshold of 3. No intermediate outcome exists.
   - SUPERSEDED (2026-08-22 ruling): neither offered reading was accepted -- the dedup-only count
     ignores the registered exclusion and the as-run exclusion is the wrong frame. Under the ruled third
     reading the two collapse into one number, 12, because the operative-frame exclusion is empty here.
61. **On the one artifact whose scorer law the reader nearly matches, no D8 clause fires** (34). The
   fork-epoch dump gives median distinct support 3 under both readings, every grid tau inside the
   [1.5, 8] ESS band, token count explaining 0.39 of within-group weight variance at `tau_star` = 1.0,
   and shaped-versus-acoustic-only spearman 1.0 against shaped-versus-LM-only 0.5. Its policy, bed and
   scorer are all wrong for D8.1a, so this reports and binds nothing. CONFIRMED at v3 unchanged to the
   last digit, because that dump already joined the raw 50 Hz store.
62. **Read in the operative frame, the D8.0 binding clause PASSES with room** (34). Median distinct
   feasible support with the greedy member included is **12 of 13** against a threshold of 3, and the
   operative-law exclusion removes **0 of 5,730** scored members -- zero on every one of the five
   slices, against 5,096 under the pooled-store join. The entire v2 exclusion was the instrument.
   Verdict GO; D8 does not close at D8.0.
63. **Reported at D8.0, binding nowhere: the shaped weights track the LM-only weights closely on the
   operative policy** (34). Median spearman between shaped and LM-only weight vectors is 0.9790 on the
   binding slice and 0.978-1.000 across all five, while shaped versus acoustic-only runs 0.2857-0.6593.
   The registered arm-selection rule reads only D8.1a statistics, so this selects and funds nothing; it
   is logged because a value above the 0.95 line would, if it survived to D8.1a, leave only
   candidate_acoustic funded. Clauses (b) and (c) fire nowhere at v3. (Correction 2026-08-22: the
   shaped-versus-acoustic-only low end was first transcribed as 0.30; the T=0.5 slice reads 0.2857.)
   The D8.0 forewarning rho 0.9790 against D8.1a's 0.3462 is a bed/policy difference, not a
   contradiction.
64. **The D7 own-infeasible drop set is exactly the four registered train-role rows, confirmed per arm
   from each arm's own artifact** (32), digit-identical to the offline dropcheck and to each other.
65. **D7.1 reached its fixed final endpoint on both arms, and its two banked held statistics point in
   opposite directions** (32). Candidate internal-held mean `L_online` 0.007541 against the control's
   0.010225 (26 % lower, the direction the online same-speaker negative is meant to produce); its
   internal-held per-frame NLL is 2.5319 against 2.5259, i.e. 0.0060 higher. Both are single point
   values from one donor draw per held anchor, so neither is the D7.2 statistic. What D7.1 establishes
   is only that the A/B ran matched to its endpoint and produced two fixed-final scorers.
66. **D7.2 FAILS on clause 2, so D7 closes without a policy leg** (32). Clause 1 passes decisively,
   clause 4 passes exactly on both arms, clause 3 shows both arms clearing the external floor -- but the
   candidate's internal-held per-frame NLL is 2.531898 against the control's 2.525882 over 8,642,253
   frames, and clause 2 requires it to be no greater. Not a surprise from a new instrument: it
   reproduces what D7.1 banked to 3.62e-9 and was flagged to the planner as the standing risk before
   D7.2 ran, with the gate ruled unmoved. The outcome does not depend on the one convention still open
   (clause 3's point-versus-CI eligibility reading), because clause 2 fails under either. Per the
   registered gate no sampler or temperature rescue may be selected. This licenses not funding the D7.3
   policy leg at this operating point; it is not evidence that an online same-speaker negative cannot
   work.
67. **The online same-speaker negative did exactly what it was built to do, and the cost landed on the
   insertion channel** (32). Paired candidate-minus-control mean `L_online` -0.00269867, two-sided
   bootstrap [-0.0027472, -0.0026505] over 2,274 speaker clusters, 99.25 % of 14,008 eligible anchors
   moving the right way -- a population-wide shift, not a tail. It transfers off its own bed: the
   candidate's usage gate on the frozen external gold-dev rows is +5.3587 against +5.0509 while both
   arms' plain per-frame NLL there is 2.4595 to four digits (though the widening decomposes to +0.30712
   from the deranged NULL against +0.00071 from the true side -- the candidate mostly prices the null
   worse, not the truth better). What it costs is the length/insertion channel: the matched insertion
   discount is significantly LARGER for the candidate at every k. Sharper same-speaker discrimination
   and a worse insertion exploit are the same trade here.
68. **Scorer refresh has no established durable benefit: the frozen control WINS the final leg on both
   splits** (36). The registered requirement is explicit, and at leg 8 periodic is 18.82/24.56 against
   frozen's 17.61/22.66 (worse by 1.21 and 1.90) -- the gate's named "frozen final-leg win" case. The
   early legs do show the transient the gate anticipates and refuses to fund: periodic leads at legs 2,
   3 and 4 (by 0.55/0.55 at its best), then loses from leg 5 onward and never recovers. This licenses
   not funding scorer refresh at this operating point; it is not evidence that a refreshed scorer cannot
   help.
69. **The bigger fact both arms share: the eight-leg loop degrades badly after leg 3, and NEITHER arm
   ends better than its own no-loop init** (36). Periodic runs 12.85 -> 18.82 dev-clean and
   17.89 -> 24.56 dev-other from its best leg to its last; frozen runs 13.40 -> 17.61 and
   18.44 -> 22.66. Against theta_0^G's 13.89/18.34, leg 8 is worse by 4.93/6.22 (periodic) and
   3.72/4.32 (frozen). Only legs 2 and 3 of either arm ever beat the init on dev-clean, and no leg of
   either beats it on dev-other by more than 0.45. The recency question is therefore settled inside a
   regime where the loop itself is losing ground after leg 3.
70. **D8.1a's regenerated greedy is NOT the D7 pool's 1-best: 31,562 of 281,241 utterances differ
   (11.2 %), so the registered deviation is NOT admissible** (35). Coverage is exact and rules out a
   subset artifact: 281,241 of 281,241 compared, 0 only-in-dump, 0 only-in-pool, 0 duplicate greedy
   rows. The differences are lexical rather than formatting -- already on the D8 reader's own normalized
   fold -- and fall on hard or rare words ("barny to unless" against "barnett unless"; "sowing wood"
   against "saucing wood"). Per the registration the D8.1a verdict is NOT ACCEPTED on that support.
   Nothing was auto-escalated; any number `D8WeightJob.1G2lPRnRmPks` produced rests on a support that
   failed its admissibility read and must not be read as a D8.1a result.
71. **D8.1a piece 3: the pool scoring pass reproduces the dump's forward configuration exactly, and the
   mixed convention it licenses is NOT small** (35). BINDING half: `recon` reproduces the dump's stored
   greedy column to 4.77e-07 maximum absolute difference against a 1e-3 tolerance, median exactly 0, no
   degenerate row and no text mismatch -- so the text-path pass IS the dump pass, and verdict PARITY
   licenses scoring the 31,562 differing utterances. MEASURED half: `lm_prior` differs on 64 of 64 tags
   (median absolute 0.0967, maximum 0.5310) and `n_tokens` differs on 64 of 64 (maximum 3) -- for the
   SAME string the dataset text pipeline and the decode's own token path never agree on the
   tokenization, not rarely but always. Signed, the shaped numerator `lm_prior * n_tokens` is higher
   through the text path on 64 of 64 tags (median +9.17 nats, mean +9.53, range +6.94..+17.69). So the
   mixed convention is a systematic, one-sided difference in exactly the column the shaped score depends
   on. RESOLVED 2026-08-22 (ruling latest+3): the mixed convention is rejected; every column of the pool
   member now comes from the text path on all 281,241 tags.
72. **The decode path's extra token is the generation's terminal token, and it explains 58 of 64 tags
   but not all of them** (35). The text path's `n_tokens` equals the plain tokenization of the pool
   string on 64 of 64 tags -- an exact anchor -- so the decode path's surplus is a clean subtraction:
   exactly one token on 58 tags, `<|endoftext|>` (id 151643), appended by the generation and never by
   the text path; two on 3 tags and three on 3 tags, reported unexplained rather than absorbed. It feeds
   no clause, no weight and no verdict. What it makes legible: under the corrected convention all twelve
   rollouts pay the terminal token's prior cost while the pool member does not -- the member-versus-
   rollout gap definition (a) always implied, now with a measured size (~+9.5 nats) and a named cause.
73. **D8.1a is COMPLETE and its verdict is GO, funding ONE arm: `candidate_acoustic`** (35, RESULT
   TABLE). All three no-go clauses pass with margin and the exclusion rate is 18 of 3,170,676 scored
   members against the 5 percent valve, with no feasible-but-non-finite `recon`; all 281,241 groups are
   frozen to `supports.jsonl`. The arm-selection rule fires on its second clause: spearman(shaped,
   acoustic-only) 0.9835 > 0.95, so `candidate_shaped` is NOT funded -- not because it failed but
   because at this operating point it is not a different experiment. spearman(shaped, LM-only) is
   0.3462, far below the same bar, so the shaped score is NOT free English; that was the other way the
   shaped arm could have been struck out and it was not.
74. **The pool-member scoring convention is IMMATERIAL to the D8.1a decision, by measurement rather
   than by argument** (35). The pre-registered sensitivity line recomputed the whole read under the
   superseded mixed convention from the same artifacts: no no-go clause flips, the valve does not flip,
   the verdict does not flip, the funded-arm set does not flip. The only moving statistic is
   spearman(shaped, LM-only); spearman(shaped, acoustic-only) is identical under both to four decimals
   (ranks cannot move under the probe's <=4.77e-07 recon deltas). What it does NOT license: the
   correction was still necessary, because the ~9.5-nat one-sided offset in the shaped numerator
   (verdicts 71-72) was real and its immateriality could only be established by making the measurement.
75. **D8.1b candidate-acoustic is COMPLETE, and the realized draw reproduces the frozen weights** (37).
   Realized greedy-draw fraction 0.25312 against the 0.25266 mean weight the frozen artifact places on
   the greedy member -- agreement to 5e-04 on a quantity nothing tuned. Three quarters of visits trained
   on a non-greedy target, so the arm is not the control in disguise, and 0 drawn members were
   infeasible, so the training bed and the weight artifact agree about the store.
76. **The registered per-step-cost parity with the control HOLDS, measured** (37). Forming batches from
   the control's items before any draw makes the shard membership, the batch partition and the step
   count the control's by construction, and the wall clock (13:52 against 13:59 and the D7 online
   candidate's 13:58 over identical shards and batches) confirms the drawn targets did not move the cost.
77. **DESCRIPTIVE, NOT AN ADMISSION READ: the candidate's fixed-final internal-held per-frame NLL is
   below the control's** (37). One deterministic read on the held greedy targets, scored the same way
   for both arms and reported because the authorization asks for it. It decides NOTHING: D8.2 owns the
   registered admission, a PAIRED estimator with a speaker-cluster bootstrap and a control-defined
   `delta_NI`. A raw difference of 0.012 between two unpaired aggregates is not that statistic and must
   not be quoted as evidence of non-inferiority in either direction.
78. **D8.2 clause 1 PASSES, and by a margin that does not depend on the margin** (37). The one-sided
   bound is not merely below `delta_NI`, it is below ZERO -- so the clause would also pass at D7's
   stricter zero margin and the data-defined margin never became load-bearing. Both arms' pooled
   aggregates reproduced their banked values before the clause was read. SCOPE: this is per-frame NLL on
   the held GREEDY targets, i.e. absolute fit on the incumbent's own distribution; it is one of FOUR
   clauses and decides D8.2 with none of them.
79. **D8.2 clause 2 FAILS -- the mechanism's claimed win is absent** (37). No ladder has a bootstrap 95
   percent lower bound above zero: the two positive point estimates straddle zero and `filler_ins` is
   significantly WORSE. Spreading the training target over the sampled group did not improve
   discrimination on any registered corruption family, and degraded filler-insertion discrimination.
80. **D8.2 clause 3 FAILS -- gate v2 returns NO WINNER because the candidate is ineligible** (37). It
   passes the (i) floor, (i) improvement and (ii) clauses and improves the matched insertion discount
   significantly at every k and leave-one-out cross entropy, yet is INELIGIBLE under both the point and
   the CI reading on the ladder-not-below clause verdict 79 measures. The incumbent control is eligible
   but cannot improve on itself, so the table returns NO WINNER with no arm admitted.
81. **D8.2 does not pass, and the registered consequence is that D8 CLOSES WITHOUT A POLICY LEG** (37).
   Clause 1 passes, clauses 2 and 3 fail, and clause 4 was not needed to reach the outcome but was read
   rather than skipped, because no clause is decided on another's expected result. WHAT THIS LICENSES
   AND WHAT IT DOES NOT: the posterior-weighted refit is not funded to a policy leg AT THIS OPERATING
   POINT (group 12, T=0.7, tau_star 0.05, acoustic-only weights). It is NOT a finding that soft
   multi-hypothesis targets cannot work, and the registered no-rescue rule exists precisely so that the
   tau, group size or weight view that happens to look better here cannot be selected from this table.
   Localization worth carrying: the arm improved absolute fit and insertion pricing while failing to
   improve ranking -- it learned the target distribution better without learning to discriminate
   corruption better, the failure mode the acoustic-only arm was registered to expose.
82. **D8.2 clause 4 PASSES -- the online and offline scorer paths are the same function** (38). Read
   after clauses 2 and 3 had already closed D8.2; an implementation-identity check, not a quality one,
   and it changes nothing about verdict 81.
83. **D8.4 CANNOT BE READ ON THE REGISTERED OPERATIVE BED AS PINNED -- 91 percent of that bed is
   unscoreable by the psi alignment family, and this is a property of the bed, not a wiring error**
   (38). The reader fails closed on its own guard at 46 shared groups against the pin of 512. The cause
   is measured, not inferred: both arms independently report identical infeasibility counts, so it is a
   property of the text-to-unit alignment and not of either scorer's weights, and the quarter-rate store
   carries a quarter of the frames per utterance on the IDENTICAL 34,106 utterances, so under the
   standing d_min >= 2 topology most operative rollouts have more symbol states than the available
   frames can host and score exactly zero probability. The 46 surviving groups give delta eta +0.0043
   [-0.1020, +0.1257], INDISTINGUISHABLE on 9 percent of the registered bed, which discharges the read
   in neither direction. The wiring was verified against the registration before this verdict was
   written (`config_sae_3e1_d8_0_v1.py:54-61` pins `GTRACK_DUMP` = `ReturnnForwardJobV2.J9yA1eYnxwYA` and
   `GTRACK_UNITS` = `MergeUnitsPklJob.hJmZtbPDa2hd`, exactly the pair D8.4 consumed, and the two arms
   differ in `model_pt` alone). Descriptive and NOT a verdict on ranking: the full-set
   rank-only column reports eta -0.1680 candidate against -0.1548 control, against the same candidate's
   +0.3086 on the fork bed. This licenses "the registered D8.4 comparison cannot be made on this bed as
   pinned" and nothing about either arm's ranking quality. Stands as written; the bed was re-pinned by
   ruling and re-read as verdict 84.
84. **D8.4 ANSWERS THE REOPENED D8 QUESTION -- the candidate-acoustic scorer and the exact D7 control
   are INDISTINGUISHABLE at ranking, and the tie resolves to the control** (38). Paired delta eta
   -0.0293 [-0.0697, +0.0085] straddles zero, so under the pre-registered three-way rule the verdict is
   INDISTINGUISHABLE and resolves to the control under the standing incumbent-tie rule; the guard passes
   at 512 of 512 shared groups. The plain-WER form recomputes the same number and the reader refuses if
   the two forms disagree. Both scorers are fixed-final, so this read selects nothing. CONTEXT, never
   gating: the paired delta spearman, the arm-internal nulls (never differenced) and the fork-bed pair,
   which agrees in direction, also straddles zero, and is a different policy. WHAT THIS LICENSES: "the
   posterior-weighted refit does not rank better than the control at this operating point" -- a
   MEASUREMENT of the real target quantity and not a constructed clause battery, the distinction the
   user's reopening rested on. It does NOT license "the candidate is worse": the interval contains zero
   and the tie resolves to the incumbent by rule, not by evidence of inferiority.
85. **The evolved policy's within-group sampling has largely collapsed, and that is what closed D9.1's
   arm 3** (39). The median group carries 2.0 distinct-scoring support members against 13 candidates
   offered, with a third of groups at exactly one. Clause (a) reads NO-GO and arm 3, the soft-EM refit,
   is not funded -- a tempered posterior over a median-two-string support would be the 1-best refit at
   extra cost, which is the degeneration the clause names. THE THINNESS IS THE POLICY'S, not an
   instrument artifact: the scorer-free variant and the rollouts-only variant both read 2.0, so it is
   neither the scorer nor the greedy member's inclusion, and both convention readings report the clause
   fired. THE BED IS SOUND: every group carries live support, the valve is idle, and clauses (b) and (c)
   pass. DESCRIPTIVE, ADOPTING NOTHING -- one dump from one checkpoint at one temperature; it licenses
   no claim about the loop family without a temperature sweep that is neither registered nor run.
86. **D9.2 answers the evolved-point question -- the 1-best refit and the incumbent are
   INDISTINGUISHABLE, and the tie resolves to the incumbent by rule** (39). The interval straddles zero
   and the plain-WER form recomputes the same delta eta; both scorers are fixed-final, so this read
   selects nothing, and under the registered gate a refit arm is adopted only on an interval excluding
   zero in its favour -- arm 2 is NOT ADOPTED. THE STOP CLAUSE PASSED ON ITS OWN TERMS, worth recording
   as a pass rather than a silence: D9.0's structural census predicted every rollout row alignable and
   both arms scored 7,168 of 7,168 rows finite with 0 groups dropped. WHAT LIMITS THIS READ, stated
   because a tie is exactly where power matters: the shared oracle headroom is 0.0116 against D8.4's
   0.0600, five times smaller, and eta divides by it -- so the interval is 0.247 wide against D8.4's
   0.078. The verdict is "not distinguishable on this bed"; it licenses "the refit is not adopted",
   never "the two scorers rank equally well", and a future arm wanting to be distinguished here would
   need a much larger effect than D8.4 needed. WHAT IT LICENSES per the registration: jointly with D8.4
   and D6-PERIODIC, "scorer refitting is not funded on this loop family at cold or evolved operating
   points" -- never "refitting could not work elsewhere". The phase closes only on the USER's word.
## Open findings and unresolved verifier feedback

Conventions and pins still open, then measured caveats that qualify a banked number. Resolved
hand-backs are not repeated here.

**A. Unpinned conventions that change a winner or an eligibility call.**
- CLAUSE-3 POINT-VERSUS-CI ELIGIBILITY (D7.2, and every acceptance round after it). The gate table
  deliberately leaves the reading to the planner and prints both (`elig_pt False` / `elig_CI True` for
  the D7 candidate). It decided nothing for D7, because clause 2 fails under either, but it recurs at
  the next acceptance round. The user's blessing is still pending (SAE.md queue item 2).
- THE D2 WINNER RULE TURNS ON TWO UNPINNED CLAUSES (2026-08-08 audit, never closed). (a) Clause (ii) is
  algebraically clause (i)'s improvement comparison sign-flipped (H_uni is bit-identical across arms),
  so `d2_both`, a changed-text candidate, is eliminated by exactly the comparison the gate v2 floor-only
  amendment ruled inadmissible -- and that is the only thing removing it (the argmax is unchanged at
  k=1/k=4 if it is admitted; at the omitted k=2 `d2_both` out-reduces `d2_contrast`, n.s.). (b) The
  ladder floor's "not below" is CI-read in this log and point-read in the rule text; under the point
  reading only `d2_states` is eligible and THE WINNER FLIPS to `d2_states`. Pins were proposed and need
  the user's blessing; `config_sae_3e1_d3_v1.py:37-38` hard-codes `WINNER='d2_contrast'` provisionally.
  Also: `d2_states` is admitted through the improvement halves of (i)/(ii) on ce_loo numbers approach 8
  itself marks (*) cps-incomparable -- only the absolute floors bind for it.
- CONCLUSION 16's `psi_len_only` verdict ("no -- (e)") is the one selector verdict that depends on the
  unblessed CI-convention pin; the other four are convention-independent.

**B. Statistics whose reproducibility or labelling is still off.**
- APPROACH 10's D2 PAIRED-CI ENDPOINTS are not reproducible from any pinned seed (17 of 24 differ in
  the 4th decimal, max 0.0007); no seed or resample count was pinned for statistics computed outside
  any job. Verdict-neutral except the two boundary calls recorded under conclusion 15. The clause-table
  job fixed this going forward; the logged endpoints were never re-issued.
- APPROACH 4's gold_enc50 held ce_loo cell reads 3.1274; the artifact
  (`PsiHeldNllJob.ag5DZ3A2Gd1K`) says **3.1385**. UNRESOLVED CONTRADICTION in the table; the row's
  derived columns already use 3.1385 and the ordering is unaffected.
- APPROACH 8's TABLE IS COLUMN-MIXED: `beta_to` and `spearman` are the lambda=0 `recon` reads while
  `steerable` is the lambda=1 `shaped` read (the incumbent's shaped `beta_to` at lambda=1 is 0.2284).
  The relabel was handed back and is not reflected above.
- "ARM-INVARIANT" LABELS OVERREACH in two places. Approach 3's selector block: `psi_len_only` and
  `neg_n_oov` are recomputed per arm by each `PsiAlignRerankJob`; only `lm_prior_units`,
  `neg_n_suspect` and `n_tokens` come from the shared dump by construction (n_oov coincides because the
  arms share the lexicon config, and `psi_len_only` genuinely differs, hence its logged range); the
  selector CIs run on 509/505/438 groups after degenerate-group filtering against 512 in the ranking
  block. Approach 11's caption is false for `psi_len_only` -- its (e) is psi_g_tc100's value and its (f)
  is d2_both's. Every arm's (e) CI straddles zero, so conclusion 7 stands either way.
- `S/gate_table/PsiGateClauseTableJob.4Z0gb5GgtD2u` (D7.2 clause 3) PRINTS A FALSE REASON:
  `clauses.txt` says "NO WINNER (reduction CI includes zero)" in both places because that string was
  hard-coded rather than derived. The OUTCOME is correct under both readings but for two different
  reasons: on the point reading no arm is eligible at all, and on the CI reading the candidate is the
  argmax and its k=1 paired CI [+0.0039, +0.0106] excludes zero on the WRONG side. The numbers in the
  same file and in `clauses.json` are correct and were verified bit-exactly. Fixed in code (speech-llm
  `fc30dc1`) but the FINISHED artifact still carries the false line -- do not quote it.
- CROSS-SCORER LAMBDA IS NOT COMPARABLE: `d2_states`' recon scale differs (within-group variance ratio
  k = 0.0091 against the incumbent's 0.0131), so a scalar lambda cannot be carried across scorers --
  match operating points on prior share instead.

**C. Reads that exist only as scratch or are not wired.**
- THE JOINT REPRICING READ (2026-08-08, planner scratch on the D0 dump, never emitted from a job and
  therefore not citable): at T=0.7 the live lambda=1 sits far below every scorer's ranking optimum
  (incumbent at lambda=8: spearman 0.5558 -> 0.6778, beta_to 0.2284 -> 0.1112, sel_wer 0.1316 ->
  0.1222, steerable 0.1949 -> 0.2034, prior share ~46 %), the optimum is arm-invariant at prior share
  ~0.45, at matched operating points NO D2 candidate beats the incumbent on any rollout statistic, and
  beta_to reaches zero only at lambda ~22-27 at prior share ~88 % (inadmissible). Must be reproduced as
  a logged table by the clause-table job before any of it is used.
- THE ORDERED `std_within_group` READ IS STILL NOT WIRED (no job, config entry or alias). Two existing
  routes, one a trap: `RolloutMechanismJob` emits `std_within_group` over a hardcoded
  ("recon", "shaped") tuple, so the lm term ALONE -- the half the order names -- is missing (a banked
  full-bed instance, `RolloutMechanismJob.UJ0DfPXTH8Cq`, reads recon 0.0218 / shaped 0.0239 over 28,539
  groups); `RewardShapeSweepJob`'s `compose()` reads the dump's RAW `lm_prior` column, which is per
  generated TEXT TOKEN while the shaped arms train per UNIT FRAME, and `n_tokens` varies within a group
  while `n_units` does not, so it is NOT a within-group constant rescale and that route answers in the
  wrong units without the `scorer_diag` conversion. Standing dump-column trap, now with two consumers.
- RECORDED BUT NOT IN APPROACH 18's TABLE (from the D4' dump's own (a) block): on the best bed at
  T=0.7 the within-group suspect-count contrast is nearly ABSENT -- coverage 0.0037 for any suspect
  against the G-track's 0.092-0.233, mean within-group count std 0.0116 -- while the ranking prize is
  real (mean_wer 0.0562, oracle 0.0414, greedy 0.0541 over 28539 groups). The minimal-state exploit
  therefore sits in a near-total GRPO dead band at the fork's operating point: no in-loop reward term
  can steer it, which is the quantitative case for the offline refresh path and against adding reward
  terms on this bed.

**D. Measured caveats that qualify a banked number.**
- THE OOV-COUNT NULL IS INERT on the D0 bed: `n_oov` is 0 for all 6144 rows because the psi inventory
  carries no UNK state, so `neg_n_oov` is undefined rather than uninformative.
- THE (c) COVARIATE is the rollout's own WER, which controls the filler's direct insertion cost but not
  the composition of the remaining errors; the gold-text control arm, not the absolute beta, is what
  carries the contamination claim.
- D8.1a's SAMPLING SEED IS UNPINNED, as in the reference machinery, so artifact reproducibility rests
  on the frozen `supports.jsonl` -- a disclosed property, not a defect. Batching moves the wall clock
  and the deterministic columns not at all, but the SAMPLED rollouts are a fresh draw from the
  registered distribution; no number from D8.1a's first launch was ever banked.
- D7.1 PRECISION NOTES: held `L_online` averages over 14,008 rows (the 54 singleton anchors contribute
  no online term), the same denominator in both arms; control shards 6/9 and candidate shard 9 report
  `u_to_z` exactly 0.0, which is fine for a satisfied hinge but matters if `u_to_z` is ever read as a
  live signal.
- D6-PERIODIC/GAN960-FROZEN carries an inherited-bookkeeping conflict, flagged and not resolved: leg 1's
  record carries dump/pool/refit entries that are the SCORER's provenance from theta_0^G decodes, which
  in this arm is not its own round 1. Those jobs are finished and fund nothing, but a downstream audit
  could misread them.
- THE FROZEN-VERSUS-PERIODIC CONTRAST IS NOT SINGLE-VARIABLE against the best frozen G-track row:
  topology (d_min=1 vs 2), scorer corpus and policy-optimizer continuity all differ, and d_min=1 was
  historical rather than a winning hyperparameter. THE MISSING CONTROLLED ARM freezes periodic round
  1's own d_min=2 scorer across otherwise identical periodic legs; it does not exist.
- NO ENDPOINT EXISTS in this log for D6-PERIODIC (fresh D-track, 5 of 8 legs), D6-PERIODIC-WARM (5 of
  8), D6-PERIODIC/GAN+HOM (3 of 8) or D6-PERIODIC/GAN960-FROZEN; the next leg of each was pending for
  node maintenance at the last read, with no error markers.
- UNVERIFIABLE (the measuring trial jobs were deleted): approach 12's ~9.5 h whole-bed estimate and the
  1.68x `max_seqs`-8 gain survive only as config-comment claims. The 11.5 h cap, the 4 -> 8 `max_seqs`
  change, the no-resume property and the actual 5:17:30 runtime all verify and are consistent with both.
- D8.1a's ten `all_bed*` jobs carry `error.run.1` markers from duplicate workers; all ten are FINISHED
  with complete outputs (`finished.tar.gz` present) and every downstream job read them successfully.
  Recorded only so a later reader does not treat them as failures.

## Entry points and shared code

`config/sae_3e1_d0.py`, `_usage.py`, `_d1d2.py`, `_d3.py`, `_d4.py`, `_d4p.py`, `_d5a.py`, `_d5b.py`,
`_fork.py`, `_d6.py` (builds D4' and the swap-in too), `_d6periodic.py`, `_d6periodic_warm.py`,
`_d6periodic_gan960_frozen.py`, `_hom.py`, `_d7_gan_seqdisc.py`, `_d8_0.py`, `_d8_1a.py`, `_d8_1b.py`,
`_d8_2.py`, `_d8_4.py`, `_d9_1.py`, `_d9_2.py`. D7 tracked canonical configs
`config_sae_3e1_d7_0a_v1.py` at `a0a22b4` and `config_sae_3e1_d7_v2_v1.py` at `7b2069d` (workspace
wrappers only delegate).

Code: `sae/scorer_diag.py`, `text_repair.py`, `psi_align_jobs.py`, `psi_align.py`, `curate.py`,
`gate_table.py`, `refresh_gate.py`, `d7_census.py`, `d7_v2.py`, `d7_online.py`, `d8_feasibility.py`,
`d8_weights.py`, `d8_pool_scores.py`, `d8_train.py`, `d8_admission.py`, `d8_eta.py`,
`d8_bed_feasibility.py`, `psi_align_compare.py`, `d9_refit.py`, `fork_screen.py`, `psi_forensics.py`,
`homophone_probe.py`, plus focused tests. `test_psi_align.py`'s CUDA/python lattice parity test carries
two `d_min=2` skip_ok cases, so the topology D7 trains in is pinned; executed on a GH200 2026-08-21
(`log/parity_test.1445759.out`).

Cross-arm error anatomy at matched points: `S/scorer_diag/PolicyAnatomyJob.Cda1gPFxLM2V` (periodic
family), `.pxqfrYx23Rth` (swap-in vs control), `.eMeWgTsMWSRM` (D5(a)-1).

Manager hygiene, kept because it constrains how these graphs may be run: both D0/D1-D2 configs pin
their inputs by absolute path + `hash_overwrite` instead of importing the running loop's graph, since
sisyphus has no cross-process lock. `config/sae_3e1_d3.py` does NOT: it builds its arms through
`config_sae_3a_gan_loop_960h_v1.baseline` so the control differs from the arms it controls in the
scorer alone, which pulls in the finished 960 h unit and dataset graph -- run exactly one of the two
managers. D3's cost on that bed is ~5.3 h per sub-epoch on 4 GPUs, i.e. ~85 GPU-h for two arms at four
sub-epochs, against the ladder's "~9-18 GPU-h" estimate. `sae_3e1_d8_4` REPLACES `sae_3e1_d8_2` (its
graph is a strict superset); two managers over the shared reranks would double-submit them.
