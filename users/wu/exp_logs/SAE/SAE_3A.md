# SAE §3a — psi_align, the monotonic-alignment scorer, and the loops it rewards

## Approach

**1. P0 — feasibility of the text side before building anything.** `PhoneStatsJob` on two beds (the
10 h seed where psi_align trains, and the exact n=512 tc100 utterances the G3 dumps contain), both
inventories plus rVAD silence correction; the pre-registered stop rule is >10 % of utterances with
T < U. A BPE cell expands each token into one state per `chars_per_state` characters, so the state
count lands on the phone arm's scale without a lexicon and becomes nearly independent of the merge
count — the vocabulary buys *context per label*, not length.

| cell (tc100 G3 bed, n=512) | base symbols | U median | T/U median | T_speech/U median | frac T < U | verdict |
|---|---|---|---|---|---|---|
| phones | 41 | 128 | 1.284 | **1.100** | 0.033 | on the stop line |
| chars | -- | 157 | 0.859 | 0.74 | **0.815** | STOP |
| bpe512, 1 state/token | 952 | 69 | 2.387 | 2.039 | 0.000 | GO |
| **bpe512, cps 1.5** | 952 | 102 | 1.622 | **1.391** | 0.000 | GO |
| bpe128 / bpe2048, cps 1.5 | 264 / 3790 | 108 / 103 | 1.539 / 1.599 | 1.316 / 1.372 | 0.000 | GO |
| any vocab, cps 1.0 | -- | 157 | 1.052 | 0.905 | **0.350** | collapses to chars |

The same instrument found the OOV rate that changed the G3 design: gold seed text is 0.26 % OOV but
*rollout* text is 4.21 %, with **51.4 %** of candidates carrying at least one OOV word — mostly
misspellings, and misspellings correlate with WER, so psi could outrank the incumbent purely by
counting words the lexicon lacks.

**2. P1 — the model and the correctness gate.** One state per text symbol, per-state 3-way arc
softmax {self-loop, advance, skip}, per-state categorical over K=500 only (no shared-vocab
denominator, so the 2S mass leak does not exist), exact forward-sum in fp32 log space, 6-layer
bidirectional RoPE text encoder, ~11 M parameters. The gate is that the forward-sum equals brute-force
enumeration of every monotonic path on 8 (T,U) shapes and both start distributions, and that the
likelihood **sums to 1 over all lengths** — the second is what licenses "length is priced inside the
score" as a measurement rather than an assertion.

**3. P2 — where silence gets a state: `edges` (one skippable state per utterance edge) vs `words`
(one at every word boundary too), +26.6 % states.** The second arm exists because rVAD puts 14.6 %
of frames in silence and the silence-corrected budget is 1.09 against a raw 1.27, i.e. essentially
all of the aligner's slack IS silence.

| arm | pinned ep | held NLL/frame | align entropy | p_skip | silence occupancy (rVAD: 0.146) |
|---|---|---|---|---|---|
| edges | 11 of 30 | 2.5524 | 0.438 -> 0.121 | 0.10 | 0.038 (under-absorbs) |
| **words** | 9 of 30 | **2.2542** | -- | 0.25 | 0.24 (over-absorbs) |

**4. G1 — the information gate, on the incumbent's own instrument** (same 5000-utterance tc100-dev
subset, same `ShuffleUnitsPickleJob` seed-42 within-dev derangement the 2S AR usage gate and the FER
job used, plus a length-matched derangement and a unit-unigram length-only null). The plan mandated
`ce_emis` — the emission component under the alignment posterior — but that posterior is inferred
from the very frame it scores, so G1 also reports `ce_loo`, the CE of p(u_t | u_-t, text) with frame
t's own emission removed from alpha; that is the number that belongs next to the incumbent's masked
CE, and it is what the planner adopted as the gated statistic.

| arm (H_uni 5.9764) | NLL/frame | ce_loo | loo top-1 | usage gate (loo, length-matched) | text-explained |
|---|---|---|---|---|---|
| psi_align `words` | 2.3675 | **2.0834** | **39.42 %** | **+6.64** | **+3.68 (62 % of the stream's entropy)** |
| psi_align `edges` | 2.5816 | 2.2945 | 36.71 % | +6.45 | -- |
| incumbent p10 AR (bar) | -- | -- | 3.17 % | 0.564 | 0.419 (7.0 %) |
| family-wins bar | -- | -- | -- | 0.67 | 0.53 |

**5. G3 — the reward-rank gate, re-scoring the existing dumps with no new sampling, plus two nulls
the plan did not have.** The **OOV-count null** ranks candidates by `-n_oov` alone (forced by P0's
51 % measurement); the **length-only null** is psi's own lattice with the emissions switched off,
i.e. log p(terminate at T | text) — without it "psi beats the audio-free null" is consistent with psi
being nothing more than a better duration model. Every table carries an anchor: the incumbent's own
`recon` column recomputed from the same file must reproduce `reward_rank.txt` on all 50 column values
x 5 temperatures or the job asserts.

| theta_0 bed, T=0.7 | gap_true | spearman | eta | grad_align | audio margin [95 % CI] |
|---|---|---|---|---|---|
| **psi_align `words`** | +0.7723 | **+0.7778** | **+0.8257** | +0.0747 | **+0.764 [+0.633, +0.906]** |
| psi_align `edges` | +0.7512 | +0.7504 | +0.8214 | +0.0747 | +0.734 [+0.608, +0.870] |
| psi length-only null | +0.0069 | +0.1481 | -0.0076 | +0.0381 | -- |
| incumbent AR | +0.0123 | +0.1692 | -0.0311 | +0.0335 | -- |
| OOV-count null | +0.5362 | +0.2281 | +0.1208 | +0.0219 | -- |

On the G-track bed, where the incumbent family's audio margin was *negative* (-0.074), psi reads
+0.338 (`words`). Seen/unseen split on the 39 of 512 utterances inside psi's own training set is flat
(0.804/0.746 spearman), so nothing here is leak.

**6. §5c — the BPE text side, all cells on `words`.** BPE is not a formatting variant of the phone
arm: no lexicon, no OOV at all, and it moves the id layout, the silence symbol and the embedding
shape at once. The 10 h budget forbids learning each (token, sub-index) pair from scratch, so the
embedding is factorized — ids are `base_token * max_sub + sub_index` with the token vector shared
across its sub-states. Codes come from `ReturnnTrainBpeJob` on the LibriSpeech **LM corpus**, the same
unpaired-text resource the phone side's G2P already draws on; no audio transcript enters.

| arm | ce_loo | loo_top1 | usage gate | theta_0 spearman / eta | G-track spearman / eta | OOV-null spearman |
|---|---|---|---|---|---|---|
| phones_words | **2.0834** | **39.42 %** | +6.64 | 0.7778 / 0.8257 | 0.6507 / 0.6805 | **+0.2230** |
| **bpe512_cps15** | 2.3426 | 34.57 % | +6.09 | 0.7533 / 0.8162 | 0.6122 / 0.6540 | none (closed inventory) |
| bpe128_cps15 | 2.2638 | 36.20 % | +6.06 | 0.7506 / 0.8160 | 0.6009 / 0.6140 | none |
| bpe512_1tok | 2.5488 | 28.91 % | +5.74 | 0.7528 / 0.8161 | 0.6221 / 0.6755 | none |

**7. §5b.1 (cell M2) — continuous vs discrete emissions, single-variable.** M2's frames are *the very
vectors k-means quantizes*: corpus-standardize the post-adapter state, then project onto the
ALREADY-FITTED PCA basis read verbatim out of the finished `QuantizeStatesJob`'s `quantizer.pkl`, so
the two arms see an identical 96-d representation and differ only in whether it passes through a
500-way argmax. Emissions become per-state diagonal Gaussians with a variance floor
(`logvar = 2*min_log_std + softplus(raw)`, `min_log_std = log 0.1`) — without one a state shrinks its
variance around a single frame and ranks candidates by variance collapse. Absolute CE/NLL are void
across the arms (a log-mass and a 96-d log-density are different currencies), so the comparator is a
**paired** bootstrap of eta on shared groups, with the pre-registered rule that ties, overlapping CIs
and "matches within noise" all resolve to the incumbent.

| bed / T | d_eta (cont - disc) [95 % CI] | d_spearman [95 % CI] |
|---|---|---|
| theta_0, T=0.3 | +0.0363 [-0.016, +0.091] | **+0.0458 [+0.012, +0.083]** |
| theta_0, **T=0.7 (decision)** | +0.0027 [-0.027, +0.035] | +0.0009 [-0.014, +0.017] |
| theta_0, T=1.0 | **-0.0266 [-0.043, -0.010]** | **-0.0702 [-0.080, -0.061]** |
| gtrack, T=0.7 | +0.0129 [-0.039, +0.067] | +0.0056 [-0.015, +0.027] |
| gtrack, T=1.0 | +0.0003 [-0.018, +0.021] | **-0.0397 [-0.048, -0.032]** |

**8. §5b.2 — the substrate axis: five cells, one variable per edge**, all discrete K=500 / PCA-96,
all bpe512 cps 1.5, `words` silence, same beds. The incumbent scores the post-adapter soft prompt
(2048-d @12.5 Hz); "simply wav2vec2" is the encoder output the adapter consumes (1024-d @50 Hz), and
the AV SFT fine-tuned that encoder — two variables, so going straight between them answers neither.
No-SFT is `av_checkpoint=None` (encoder tap only, asserted in the job), not a separate model build.

| within-design edge, T=0.7 | what it prices | theta_0 d_eta | theta_0 d_sp [95 % CI] | gtrack d_sp [95 % CI] |
|---|---|---|---|---|
| enc50_sft vs enc125_sft | frame rate, adapter absent both sides | +0.0038 | **-0.0177 [-0.031, -0.004]** | **-0.0390 [-0.056, -0.022]** |
| enc50_raw vs enc50_sft | the AV SFT @ 50 Hz | +0.0004 | -0.0105 [-0.025, +0.004] | -0.0129 [-0.034, +0.007] |
| enc125_raw vs enc125_sft | the AV SFT @ 12.5 Hz | -0.0081 | -0.0094 [-0.023, +0.004] | **-0.0393 [-0.058, -0.021]** |
| enc125_raw vs incumbent | adapter + SFT together | -0.0227 | -0.0007 [-0.011, +0.010] | -0.0195 [-0.039, +0.002] |

**9. §6.1 — psi_align as the online loop reward.** psi's recon spread is ~50x the incumbent's
(gap_true +0.617 vs +0.012), so 2S's absolute `lam_lm` 0.01 / `lam_len` 0.2 do not transfer;
`RewardShapeSweepJob` re-ranks the finished dumps over a grid extended by that factor, choosing by
split-half selection (both weights picked on one half, scored on the other, swapped) **per bed**.
`PsiAlignScorer` implements the existing `ArScorer` protocol and takes its text side FROM the
checkpoint, so the online reward cannot score under a different segmentation than G1/G3 measured;
`reward_mode="psi_align"` drops the 1.7 B token-LM outright, which is what makes a 960 h checkpoint
small enough to bank. A `PsiScorerParityJob` requires the online path to reproduce a finished
rerank's own `recon` column with rollouts entering as token ids.

| bed, T=0.7 | unshaped (lm 0) spearman / eta | split-half pick | picked spearman / eta | parity |
|---|---|---|---|---|
| theta_0 (10 h, 100 h arms) | 0.7346 / 0.8132 | **lam_lm 0.1** | 0.752 / 0.8252 | max abs diff **0.000e+00** |
| G-track (960 h arms) | 0.5801 / 0.6545 | **lam_lm 0.3** | 0.6453 / 0.6924 | max abs diff 1e-6 |

`lam_len` is measured inert on both beds (len 0.2 is indistinguishable from len 0 to four decimals)
and is carried only as insurance. Both shaped weights were picked by WER-derived statistics, so
`shaped` is oracle-tuned shaping on top of an unsupervised reward — **`recon` is the arm that carries
the label-free claim.**

**10. §6.2-6.10 — the loops, one decoder and one table.** Every row including the two no-loop inits
is re-decoded through `config_sae_3a_psi_loop_v1.recog_av` (SpeechLmV3, beam 4, lowercased refs,
literally the same four eval datasets), because a beam or batching difference between "baseline" and
"arm" would read as progress; test epochs are declared before any dev WER exists, and missing cells
print `--` so a partially-decoded row cannot be compared on fewer sets than its neighbours. The bed
axis moves the loop audio at a fixed init (theta_0) and a fixed gold-seed scorer; the G-track arms
move init and scorer together to stay label-free.

| arm (dev-clean / dev-other, test-clean / test-other at the last epoch) | dev | test |
|---|---|---|
| theta_0 (10 h AV SFT, no loop) | 16.91 / 20.64 | 15.28 / 20.78 |
| theta_0^G (pseudo AV SFT, no loop) | 13.89 / 18.34 | 14.25 / 18.34 |
| 2S loop, token-LM AR, ep3 (its best) | 12.99 / 16.20 | 14.09 / 17.00 |
| psi 10 h `recon`, ep4 | 11.06 / 13.93 | 12.20 / 15.98 |
| psi 10 h `shaped`, ep6 (final) | 9.59 / 12.84 | -- |
| psi **100 h `recon`**, sub-ep 8 (best sub-ep2 9.00 / 13.63) | 10.76 / 15.31 | 11.61 / 16.03 |
| psi **100 h `shaped`**, sub-ep 8 | **6.06 / 10.31** | **6.33 / 10.84** |
| psi **960 h `shaped`**, stock donor, sub-ep 4 -- last, stopped (as scored / scorer view) | 19.67 / 24.92, **6.77 / 11.51** | -- |
| psi 960 h G-track `recon` / `shaped`, sub-ep 4 (held) | 38.09 / 44.64, 17.99 / 23.33 | -- |

**11. Where the loop's step goes, and the `max_seqs` ladder.** Measured on a running 960 h arm:
29-32 % GPU utilization, 5 % memory-bandwidth utilization, 194-347 W of 680, 30-36 GB of 97 — while
each rank's main python thread sits at 92-99 % of one core. Audio and units are collinear at
r = 0.9999, so the fit holds audio fixed: `sec/step = 0.888 + 0.04905 * mean_generated_tokens`, i.e. at
the modal length 55 about **76-82 % of a step is the autoregressive rollout**. The root cause upstream
of both slow paths is python-driven kernel launches over tiny tensors — `av_policy.sample` runs up to
100 sequential single-token decodes over 24 rows at ~3,800 aten dispatches per generated token, and
`psi_align.forward_logsum` loops at 30.00 dispatches per frame identical at B=2 and B=24 — and the
roofline agrees (a batch-24 decode token reads 3.4 GB of bf16 weights, ~0.85 ms at 4 TB/s inside a
~30 ms token). The ladder below is one-epoch trainings on the same 10 h bed, `recon` only: instruments,
not arms, since `num_epochs=1` compresses the cosine schedule and no WER may be read off any of them.

| arm | max_seqs | batch_size | psi cells/max | GPU util | mem BW | peak cuda | train wall |
|---|---|---|---|---|---|---|---|
| production | 2 | 1e6 | 3e6 / 32 | 29-32 % | 5 % | 26.0 GB | 70.4 min |
| `control_ms2` | 2 | 1e6 | 3e6 / 32 | 36.1 % | 9.1 % | 26.0 GB | 66.4 min proj. |
| **`ms8`** | 8 | 2e6 | 3e6 / 32 | **61.9 %** | 22.4 % | 80.9 GB | **35.6 min** |
| `ms8_psiwide` | 8 | 2e6 | 3e7 / 256 | 62.1 % | 24.1 % | 80.7 GB | 36.0 min |
| `ms12_bs3m` | 12 | 3e6 | 3e6 / 32 | -- | -- | 87.7 GB | **OOM** step 27/220 |
| `ms16_bs2m` | 16 | 2e6 | 3e6 / 32 | -- | -- | 88.6 GB | **OOM** step 11/157 |
| `ms16_psiwide` | 16 | 4e6 | 6e7 / 512 | -- | -- | -- | **OOM** 92 GB |

Data equivalence is what makes the wall times comparable (`control_ms2` 1414 steps x 2 = 2828 seq-slots
against `ms8` 361 x 8 = 2888). The memory model, fitted on two points and reproducing all seven arms, is
`mem_GB ~ 6.0 + 0.00264 * rows * (A + T_pad)`, because the differentiable re-forward in
`av_policy.logprobs` runs over `rows * (A + T_pad)` decoder positions with the audio prefix replicated
per sample; `batch_size` caps `rows*A` (confirmed to the sample) and `max_seqs` caps `rows*T_pad`, since
`av_policy.sample` breaks only on `ended.all()` so one non-terminating rollout pins `T_pad` at the cap
for every row in the step.

**12. Why the shaped 960 h arm exploded: the LM prior's length gradient, and the fix.** GRPO advantages
are group-normalized, so only a term's WITHIN-GROUP spread can move the policy — and `len_hinge`'s
`reward_<term>_std_within_group` runs 0.0028 -> 0.0004 over ep1-4 against `recon`'s 0.1112 -> 0.0276.
The prior itself is the driver: `lm_prior = base_logprob_sum / n_text_tokens` is a per-token *mean* and
`m_{n+1} - m_n = (s_{n+1} - m_n)/(n+1)`, so the mean **rises** whenever the marginal token is cheaper
than the running average, the normal case past the opening tokens. Measured within groups on 2560 groups
from two independent dumps, `corr(n_tokens, lm_prior)` is positive at every temperature (+0.19 to +0.36)
and at T=0.7 `d(mean-prior)/d(token)` is **+0.053..+0.060 nats** while `d(SUM-prior)/d(token)` is
**-3.6 nats** — that sign flip is the exploit, and it gets cheaper as it is practised (-3.6 nats/token
cross-sectionally at init, -1.0 longitudinally after 500 steps). The fix is a denominator the policy
**cannot move at all**: `n_units` is constant across a group, so `d(prior)/d(token) = s_n / n_units < 0`
for every LM, bed and length — a sign guarantee, not a calibration — and at `lam_lm = 1.0` it also makes
the objective exact, `total = (1/|u|) log p(u,z)`, whose group-standardized advantage is exactly that of
the posterior `log p(z|u)` with no weight chosen on the data.

| on the killed arm's own measured drift (`n_units` = 642.4) | prior | hinge | recon destroyed | net |
|---|---|---|---|---|
| as shipped (0.3 x per-token mean) | +0.576 | -0.026 | -0.070 | **+0.480** |
| per-unit at `lam_lm` 1.0 | -0.058 | -0.026 | -0.070 | **-0.154** |

## Conclusion

1. (1) The character arm is stopped on the rule and the phone arm sits **exactly on the line** (1.091
   seed / 1.100 tc100 against a 1.1 threshold) — the honest reading is a straddle, not "one bed passed".
2. (1, 3) The stop criterion applies only to **mandatory** states: `words` has T/U median 0.986 and 53 %
   of utterances with fewer frames than states — worse than the char arm on the statistic that stopped
   it — yet trains to a better likelihood and wins every G3 bar, because the skip arc removes an
   optional state for free, so the statistic must never be recomputed over a graph containing optional
   states.
3. (2) The skip arc **leaked probability mass at the last state** (skip from U-1 lands one beyond the
   absorbing end); left in, the score is not a distribution over (length, units) and the whole "length
   is taxed inside the likelihood" argument is false — the sum-to-one test is what proves the fix.
4. (3) The held-out pin earned its keep: `edges` bottoms at epoch 11 and rises monotonically to 3.246 by
   epoch 30 while train NLL keeps falling to 1.225, so `checkpoint_last` would have pinned a model 0.69
   nats/frame worse and the gates would have run on it.
5. (4) **G1 passes by roughly ten times the family-wins bar** — gold text explains 62 % of the unit
   stream's entropy against the incumbent p10's 4.4 %, and 39.4 % loo top-1 against 3.17 % — and the
   mechanism is not a bigger model: the incumbent must solve the alignment from position alone, which at
   12.5 Hz with real speaking-rate variation is a very weak proxy for "which phone am I in", while psi
   marginalizes the alignment instead.
6. (4) With the wrong transcript psi is worse than the unigram (9.05 vs 5.98 nats, 1.95 % top-1), so the
   alignment-inference channel cannot track units without the right text — the text-conditioned state
   bottleneck is the only route from u_-t to u_t.
7. (5) **The length-only null kills the live alternative explanation**: psi's own learned per-phone
   duration model with the emissions switched off ranks at spearman 0.148 / eta -0.008, the incumbent's
   level, so the duration channel accounts for essentially none of psi's advantage.
8. (5) The OOV-count null is beaten (+0.121 vs +0.821 eta) but is worth noting on its own account —
   counting out-of-lexicon words ranks rollouts *better than the incumbent reconstruction reward does*.
9. (6) The BPE arms reach the **same audio margin with one confound structurally removed**: the margin
   is flat across all four arms (0.740-0.764 on theta_0, 0.294-0.338 on G-track, overlapping CIs) while
   the phone arm's lexicon-coverage channel does not exist on the BPE side at all.
10. (6) `gap_true` is **not comparable across arms** — nats for psi, avoided-OOV-words for the null,
    reward for the incumbent — so the cross-arm reads are spearman, eta and the audio margin only.
11. (6) The vocabulary axis is flat (bpe128 vs bpe512 under 0.02 spearman) and the sub-state axis is not
    (`1tok` loses 0.21 nats and 5.7 points of loo_top1), which is the sub-states earning their place: a
    single categorical per token cannot say which unit comes where inside it.
12. (7) **The discrete k-means-500 target stands** — at the decision temperature the paired CI includes
    zero on both beds — and the structure at the ends is consistent: continuous wins where candidates are
    near-duplicates (T=0.3) and loses where they are wildly different (T=1.0), as a bounded log-mass
    versus an unbounded log-density should.
13. (8) **Subsampled beats unsubsampled**: 50 Hz never wins an eta comparison and its spearman deficit
    excludes zero on both beds, widening with temperature — more frames per phone gives the self-loop
    more to explain and does not help an explicit duration model.
14. (8) The AV SFT is worth ~nothing to the scorer at the decision temperature, so **psi_align does not
    need a seed-SFT checkpoint**: pretrained lv60 mean-pooled to 12.5 Hz is G3-equivalent, which removes
    two of the three paired dependencies from the reward side and breaks a circularity no gate measures.
15. (8) The 2S-era ruling "post-adapter WINS, substrate CLOSED" (2026-07-31) is **WRONG** under this
    scorer family — `enc125_sft` is the only arm with spearman >= incumbent on both beds, so the adapter
    is free but not useful, and what that ruling measured was mostly the rate change.
16. (8) PLAN_3A §5b's premise that "the AV's w2v2 encoder is frozen during SFT" is **WRONG** —
    `config_sae_2s_av_sft_w2v2_v1` passes `encoder_trainable=True` (conv front end frozen, transformer
    fine-tuned); the flag differs by phase and era, and the 2026-07-18 wav2vec2 pivot flipped SFT to
    trainable.
17. (8) **G1 information does not predict G3 ranking** — the SFT makes the unit stream 1.1-1.4 nats more
    text-predictable at both rates and buys essentially no ranking ability — the third independent
    instance of the dissociation, so the remaining leverage is on the candidate/target side, not the
    scorer's input representation.
18. (9) Shaping weight is **per bed, and that was a finding**: theta_0's own rollouts pick lam_lm 0.1
    under eta (0.3 is already past the peak) while the G-track bed picks 0.3 at every temperature under
    both metrics.
19. (10) **psi_align beats the token-LM AR on all four sets from the same init, same 2849 gold seed
    utterances, same RL knobs**: at each arm's last epoch -3.41 / -3.16 / -2.79 / -2.19, and still
    -1.93 / -1.96 / -1.89 / -1.02 after granting the incumbent its best epoch per column — the size the
    G3 re-rank predicted for replacing a reward at chance with one at eta 0.81.
20. (10) **No length exploit with no length term**: `recon` ran lam_len 0 for four passes with
    `grpo_text_len` median flat at 42.1-42.4, and on the 100 h bed dev-other insertions FALL (2.9 % ->
    1.4 %) where the 2S loop's ran 98.6 -> 216.3 % — length priced inside the likelihood does the work
    the hinge was insurance for.
21. (10) **`grpo-loop-breaks-off-seed-bed` is refuted for this reward**: at matched samples seen (100 h
    ep2 = 14,270 utts vs 10 h ep5 = 14,245) the bigger bed wins on both arms and both sets, and the 100 h
    `shaped` arm finishes at 6.06 / 10.31 dev, the best number the program has — the 2S collapse on this
    exact bed was a REWARD problem.
22. (10) Off the seed bed the LM prior is not a bonus term but **the difference between a loop that
    converges and one that turns**: `recon` bottoms at sub-ep2 and walks back up by substituting more
    while its insertions keep falling (a drift into confident wrong words, not a length exploit), and the
    prior's share grows with training — so lam_lm's value is a function of bed size, not a constant.
23. (10, 12) The 960 h G-track `shaped` arm was killed 4.5 h in on an **LM-prior length exploit it could
    not outrun**: doubling the length bought +1.92 nats of prior against 0.026 of hinge, a 22:1 trade,
    and the decisive number is that the audio-grounded term FELL 0.07 over the same span where it ROSE
    0.09 in `recon` on the identical scorer, init, data and step range — the audio-free-null failure mode
    running live, which an offline split-half sweep over fixed dumps is blind to by construction.
24. (10) The 960 h G-track `recon` degradation is an **inherited filler token, not the loop**: 61.8 % of
    insertions are the single word `to` (9.15 of 30.55 WER) and theta_0^G already had it at 66.8 % of its
    own; the root cause is structural, since theta_0^G and psi_align^G train on byte-identical
    pseudo-transcripts, so a defect both carry is not merely invisible to GRPO but rewarded. A prior
    revision naming the raw GAN as the source is **WRONG** — the pseudo-text is the §1d self-trained CTC
    student read out with a word LM (17.96 / 21.87), and no word-level decode of the raw GAN exists.
25. (10) theta_0's dev-other is **20.64, not 20.71** — the 20.71 figure appears at no epoch of that arm
    and two independent decoders both say 20.64; fixed in the configs, changes no conclusion.
26. (10) The scorer is frozen in all six psi_align arms, verified against ground truth rather than code
    reading: the `psi.*` tensors in the trained checkpoints hash bit-identical to the `PsiAlignTrainJob`
    output after 8 sub-epochs (sha256 over 82 tensors / 11,210,348 params).
27. (11) 30 % utilization is real and it is a launcher problem — four GPUs starved by four
    single-threaded launchers, neither slow path arithmetic-bound — so the only cheap lever is putting
    more rows behind each launch.
28. (11) **1.98x measured speedup** (70.4 -> 35.6 min per sub-epoch train) against the production job,
    1.87x against the same-round control; eval is excluded and is not sped up by this.
29. (11) psi's DP batching is **inert** — `ms8` and `ms8_psiwide` are indistinguishable on utilization,
    memory and wall — refuting the hypothesis that the default `batch_cells` splits a 96-row step into
    ~4 DP re-entries and eats the batching win.
30. (11) **`max_seqs`, not `batch_size`, is the memory governor**, the reverse of what an earlier
    within-arm fit suggested (that fit does not transfer across `max_seqs` and is superseded); both knobs
    set the ceiling, one term each, and `ms8` is a *saturated* peak, which is why 80.9 GB transfers to the
    960 h bed unchanged despite its 30 s utterances.
31. (11) "Long audio cannot cause the OOM" is **WRONG** — it is the dominant term, and what is true is
    that `batch_size` caps it so it self-limits; the OOM arms died in the short-audio ramp because
    `laplace` ordering makes each sub-epoch a hill and they were still climbing.
32. (12) **`lam_len` was never the free parameter**: `len_eps` 0.4 is a +-49 % dead band and 96.5 % of
    12,288 real rollouts sit inside it, so all twelve rollouts of a group are in-band and the term is
    constant across the group — it is a fence, not a slope, and cannot oppose a drift, only mark one that
    already happened.
33. (12) The naive reading "long text escapes a penalty" is **WRONG** and the correction changes the fix:
    the prior does not merely fail to charge for length, **it pays for it**, and the killed arm's +1.92
    nats is fully accounted for with no second mechanism needed.
34. (12) Under the per-unit norm the policy is charged 0.154 where it was paid 0.480 — about 5x the arm's
    own per-step reward noise and systematic where the noise is not — so `lm_prior_norm="units"` is the
    standing fix, opt-in with the legacy default so no already-run arm can change.
35. (12) The hinge was **not inert, only rare and weak** (29 %/14 % of steps carry a nonzero hinge on the
    batched arms against 4.24 % at `max_seqs` 2, which is the batching, not a policy change) and was
    removed from both `shaped` arms anyway for three reasons in order of weight: its constants are
    **gold-derived** (`nu_chars_per_sec` 14.55 and `len_eps` 0.4 are tc100 gold-transcript statistics, and
    annotations may gate but never train); it **voids the posterior identity**, since `recon + prior` is
    `log p(u,z)/|u|` only while nothing else is added; and its one useful direction is now structural via
    the per-unit sign guarantee.
36. (12) The surviving 960 h `recon` arm drifts too (median rollout 53.75 -> 56.54 over 6300 steps,
    monotone in all five duration bands) but **not with this disease** — its `recon` reward is *rising* in
    every band where `shaped`'s fell, on identical scorer, init, data and step range — which reads as
    convergence toward the speaking rate and argues for a tokens-per-unit monitor rather than for
    intervening.
37. (10, 12) **The per-unit norm closed the length exploit, not the blindness under it**: the 960 h
    `shaped` arm's sub-ep2 jump to 18.22 / 22.21 is 11.61 / 12.25 points of pure PUNCTUATION (0.00 % of
    sub-ep1 hypothesis words carry a mark against 12.77 / 14.21 % at sub-ep2), and read through the
    scorer's own view of the text the arm went 6.87 / 10.98 -> **6.61 / 9.96**, deletions 1.57 -> 0.69,
    i.e. it improved rather than diverged. `PsiAlignScorer._texts` lowercases and `phones._normalize`
    strips `[^A-Z' ]` before the state graph exists, so recon is EXACTLY invariant to punctuation while
    the base-LM prior pays for it and `lam_len`/`lam_oov` are 0 — a reward direction with identically
    zero within-group recon variance opposing it, found once recon's own steerable spread had collapsed
    0.167 -> 0.037 while the prior's fell only 0.0155 -> 0.0073 (prior share of within-group reward std
    8.5 % -> 16.5 %, ep1 to ep3).
38. (10) The bed axis therefore still reads at matched samples and it is **mixed, not a win**: 960 h
    sub-ep2 6.61 / 9.96 against 100 h sub-ep8's 6.06 / 10.31 (0.02 / 0.01 % punctuated, so that row
    needs no correction), and the pre-registered R6 insertion check rose on dev-clean under the same
    view, 0.68 -> 1.59 %.
39. (10) **The 960 h arm turns under the scorer's own view too, so punctuation was only half of it**:
    6.87 / 10.98 -> 6.61 / 9.96 -> 6.28 / 10.45 -> 6.77 / 11.51 over sub-ep1-4, i.e. dev-other peaks
    at sub-ep2 and is worse at sub-ep4 than at init-plus-one, and no sub-epoch reaches the 100 h arm's
    6.06 / 10.31. The turn is *consistent with* the stock prior taking the policy over -- its share of
    within-group reward std rose 8.5 -> 16.5 % across exactly these sub-epochs (concl. 37) while
    `recon`'s steerable spread collapsed -- which is a mechanism §0d's LibriSpeech-adapted prior
    removes, but that is a hypothesis this arm cannot settle. Read it on the 100 h `_lbslm` arm, where
    the stock twin has a full 8-sub-epoch trajectory and a round costs half a day.

## Catalog

`T/` = `work/i6_core/returnn/training/`, `S/` = `work/speech_llm/sae/`.

| artifact | path |
|---|---|
| code | `sae/{phones,psi_align,psi_align_jobs,phone_stats}.py`, `sae/grpo/psi_scorer.py` |
| entry points | `config/sae_3a_p0.py`, `config/sae_3a_psi.py`, `config/sae_3a_gan_loop.py`, `config_sae_3a_speed_probe_v1.py` |
| **the carry-forward scorer (gold seed, enc50_raw)** | `S/psi_align_jobs/PsiAlignTrainJob.IN3zmmGpH4Bv` |
| psi_align^G (pseudo-text, G-track) | `S/psi_align_jobs/PsiAlignTrainJob.kSYy0ADBgPGo`; control twin `psi_g_seed` |
| enc50_raw codebook (labels every bed and both tracks) | `S/quantize_states/QuantizeStatesJob.FWpGhC941JMi`; 960 h store `PackUnitsJob.I0uzRMfUrKWC` |
| **10 h loop arms** | `T/ReturnnTrainingJob.<recon>` / `.<shaped>`, alias `psi_enc50_theta0_{recon,shaped}_T0.7_lr2e5`; relaunched at `max_seqs` 8: `.lDYXKfZaHHwH` (recon), `.P5KPZKGnD8lT` (shaped) |
| **100 h loop arms** | `T/ReturnnTrainingJob.B5FX6Ze32HGe` (recon), `.iFe57G4QdmdR` (**shaped, the headline**) |
| 960 h G-track arms (held, checkpoints intact) | `T/ReturnnTrainingJob.QgZYaiUrVVYY` (recon), `.x8xsz3wv8wkL` (shaped) |
| 960 h theta_0 + gold scorer, stock donor (stopped sub-ep4, checkpoints deleted) | concl. 39 rests on `work/i6_core/recognition/scoring/ScliteJob.{IrhQcEwRpPcd,SivN7MS4Wj2D,0iDEH3iZpWfP,H974yEIR9j4V,RoETLtK9BmQR,NBvhzl6u6r82,trYPSsqnOa4s,oaDMHVRN32ac}` (sub-ep1-4, dev-clean/dev-other) |
| scorer-view rescoring of any finished ScliteJob | `scripts/wer_scorer_view.py` (anchored: reproduces concl. 37's 6.61 / 9.96) |
| inits | theta_0 `T/ReturnnTrainingJob.OLzy9Q2oC3mU` ep50; theta_0^G `.2fb02hGUdHNj` ep10 |
| head-to-head decoder + tables | `config_sae_3a_headtohead_v1` -> `output/.../sae_3a/headtohead/wer_{10h,960h,100h}.md` |
| reward change | `sae/grpo/reward.py` — `RewardConfig.lm_prior_norm` ("tokens" legacy default \| "units"); `test_reward.py` 17 -> 22, asserting the defect under "tokens" and the guarantee under "units" |
| standing shaped setting | `SHAPED = {"lam_lm": 1.0, "lm_prior_norm": "units"}` in `config_sae_3a_psi_loop_v1.py`, imported by the 960 h config so the beds cannot drift apart |
| commits | `90e2936` (round-1 hardening), `f468754` (scorer wiring, 11 tests) |

Provenance for the headline cell (100 h shaped sub-ep4 dev-clean) traced end to end: `ScliteJob.YsKcBQlkL4Rg`
<- `SearchWordsDummyTimesToCTMJob.G5wnrZnnEZIh` <- `ReturnnForwardJobV2.b332odZYxYfu` <-
`ExtractAvSubmodelJob.6QaRoZ59wbWY` <- `ReturnnTrainingJob.iFe57G4QdmdR/output/models/epoch.004.pt`.
Label quarantine verified by grep rather than asserted: `train_steps/sae_grpo.py:72` is the only
`extern_data["text"]` read and it sits in the `reward_mode == "oracle_neg_wer"` branch; `lam_replay`
appears zero times in both generated configs.

Relaunch decisions taken with the `max_seqs` 8 batching, all reversible: `max_lr` 1e-5 -> 2e-5,
sqrt-scaled for the 4x effective batch (batching cuts optimizer steps per pass 4x; the noise-dominated
regime argues for linear scaling and nothing has measured the critical batch size, so sqrt is the
compromise); warmup stays at 0.5 seed-epochs, i.e. matched in **samples**; 6 and 30 epochs for +50 %
compute, both still less wall-clock than the runs they replace. `lam_lm` must be re-derived in the new
units before any further launch — the reward changes units (nats per audio frame, not per text token),
so the offline sweep's 0.3 pick does not carry over. `max_seqs=2` was never a measured limit; it appears
with no comment in four 2S-era configs from the era when the model also carried a second 1.7 B token-LM.

Two standing traps. `ReturnnTrainingJob`'s hash covers neither `keep_epochs` nor the alias, so a relaunch
resolves to the same job dir and a `keep_epochs` edit made after `create_files` finished is silently
inert. And four superseded pre-hardening Rerank dirs (`hucgV0uvla4u`, `cJ0cHRKFiH7G`, `oddx93x1dvNw`,
`HehK0ZGqwROt`) carry inverted audio-margin signs — read through `alias/sae/3a/*`, never the raw job dirs.

## Verifier feedback

**2026-08-05, round 1 (build audit).** 13-agent adversarial review plus an independent DP re-read;
DP core, G1/G3 instruments, label quarantine and git hygiene sound, 7 of 8 serious claims refuted on
ground truth. CONFIRMED and latent: the G3 anchor compare is **nan-blind** (`abs(nan - want) > tol`
is False), so an empty or mis-keyed bed PASSES the anchor while every bar prints FAIL — a measurement
failure wearing a gate failure's clothes. Same class: the gate jobs accept `sil_mode` as a free ctor
arg and never check it against the checkpoint; `_length_matched`'s `assert ... or True` is a
tautology; `lenmatch_seed`'s rng is created, unused and deleted. Two refutations kept as upgrades —
the incumbent DOES price termination (EOU is inside its scored span), so NLL/frame is legitimately
comparable across scorers; and tc100 k500 units are the only valid choice on the G-track bed, needing
a caption rather than a fix. Planner rulings: running P2-P4 past the knife-edge ENDORSED and §4.4's
stop rule amended to read a straddle as "proceed with the words-arm hedge"; char arm STOPPED stands;
`ce_loo` deltas adopted as G1's gated statistic; the OOV-count null made normative.

**2026-08-05, round 2.** The encoder-freeze finding CONFIRMED and cell M6 REINSTATED — M1 (theta_0
encoder + adapter) / M6 (theta_0 encoder + mean pool x4) / M3 (off-the-shelf lv60 + mean pool x4) at
a common 12.5 Hz, M1->M6 pricing the adapter and M6->M3 the fine-tuning.

**2026-08-05, round 3 (result audit).** Six-lane audit, 127 checks, every number traced to a job
artifact rather than log prose: **the results are real** — every headline table reproduces at quoted
precision, suite 48/48 green, quarantine chain clean (BPE codes trace to the openslr LM-norm
download, train/dev id intersection 0, pinning reads only held-out NLL, G3 dumps predate every psi
job). CONFIRMED and result-neutral: the round-1 dead-assert fix is **still dead** — `units.pkl` holds
Python lists and `np.asarray(list)` is always a fresh object, so the identity check can never fire;
it is a check-shaped no-op, the exact defect class round 1 flagged. Minors: "1tok loses 0.28 nats" is
the delta vs bpe128 (0.206 vs bpe512_cps15); the §5c G1 held-NLL column is the 142 held-out train
pairs, not the 5000-utt dev set; `max_sub` does cost parameters (sub_embed is a real 8x384 table) —
only the `n_sub_states=1` form of the claim is exact; the "exactly 1 of 512 dropped at T=0.7" figure
is the edges cell, words/theta_0 drops 8; the words-arm "+26.6 % states / T/U 0.986 / 53 %" numbers
have no located artifact and should be emitted from a job.

**2026-08-06 (throughput).** Independently reproduced the step-time decomposition on all 24,916 steps:
`sec/step = 0.3078 + 0.05210*text_len + 0.10303*(audio/1e5)`, R^2 0.873, length term 81.7 % at the
mean; and out-of-regression by step-bucketing at 0.0584-0.0587 s/token.

**2026-08-18 (lam_lm sweep vs dump-column normalization).** Audited, on the implementer's
flag, whether approach 9's sweep rows are affected by the newly measured mismatch (reward-
parts dumps emit lm_prior per generated text token; live shaped arms train per unit frame).
NOT affected: the parts dump sets no reward_kwargs, and at sweep time the live arms also ran
the legacy per-token norm (RewardConfig's default, kept "so no already-run arm changes
meaning"), so the sweep was internally consistent at its own operating point -- and the
standing shaped setting (lam_lm 1.0, per-unit) was derived analytically, never read from
this sweep. Standing trap: comparing a dump's lm_prior column to the live per-unit reward
compares different units (measured -4.90 nats/token vs -0.34 per unit frame on the same
utterances); compare numerators or renormalize.
