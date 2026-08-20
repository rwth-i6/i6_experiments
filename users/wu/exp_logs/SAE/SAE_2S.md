# SAE Phase 2S — the text-bottleneck autoencoder and its token-LM reconstruction reward

## Approach

**1. Build the autoencoder: audio -> AV -> text tokens -> AR -> discrete units, the AV trained by GRPO
against the AR's reconstruction likelihood.** The text bottleneck *is* the interface, so 2S is
phoneme-free; the only vocab extension is the K+2 unit tokens, shared between AR output and AV-U input,
and the reward is
`r(z) = (1/|u|) log p_psi(u|z) + lam_lm*(1/|z|) log p_base(z) - lam_kl*KL(z) - lam_len*hinge - lam_oov`
with the KL anchored to the frozen SFT snapshot theta_0 and GRPO group-normalized (G, clip 0.2, one AR
CE step per batch, AR gradient never reaching the AV). Two design fixes came out of the tests and are
load-bearing: **the sampling temperature is part of the policy** (`sample()` and `logprobs()` must apply
it identically or the PPO ratio is not 1 at the sampling point), and unit tokens are a *trainable
overlay* rather than mean-initialized rows, because under LoRA-only the tied embed/lm_head are frozen
and appended rows would keep their init forever — a failure that looks like a capacity verdict.

**2. Audit what the deduped unit stream actually contains, before pinning it as the AR target.**
Measured on held-out frames with rVAD silence used eval-only and gold MFA silence alongside.

| quantity | value |
|---|---|
| deduped rate overall | 20.04 tok/s (raw 25 -> dedup buys only 1.25x) |
| gold phone-segment rate (incl. SIL) | 10.22 tok/s |
| decomposition of 20.04 | 10.22 true segments + 3.57 phone-level over-segmentation + **6.25 within-phone cluster alternation** |
| cluster IDs covering 90 % of silence frames | 90 (rVAD) / 74 (gold) |
| silence share | 13.7 % of frames -> ~9 % of deduped tokens |
| silence CE under a context-only bigram | 4.19 nats/token vs 4.39 for speech |

**3. The paired-init supervision ladder, from 10 min to 100 h.** Rungs are strictly nested (10 min in
1 h in 10 h, `_take_hours` filling a budget after a fixed-seed shuffle) so the curve carries no sampling
noise between points, and everything downstream of each theta_0 (`AvStatesJob` -> `QuantizeStatesJob`
K=500 -> AR SFT {p10, none}) is rebuilt per rung because the codebook is fit on that policy's own states.
The BEST-RQ rows are the earlier encoder era, read for trajectory shape only; both halves of the
autoencoder show the same transition.

| half / rung | utts | steps | convergence signature | dev-clean WER |
|---|---|---|---|---|
| BEST-RQ AV, 10 h | -- | -- | dev token-error stuck at 0.739 through all 50 epochs | 218-432 (345 % ins) |
| BEST-RQ AV, 100 h | -- | -- | 0.657 (ep6) -> **0.551 -> 0.236 -> 0.182** (ep10) | 21.4 |
| BEST-RQ AR, 100 h / 10 h | -- | -- | CE 6.10 -> 3.77 still falling / flat ~6.10 to ep12 then 4.21 | -- |
| w2v2 AV, 10 h (shipped theta_0) | 2849 | 9450 | -- | ep40 20.45 -> ep50 **16.91** |
| w2v2 AV, 1 h (100 ep / 200 ep rebuild) | 290 | 4066 / 8130 | train loss plateaued, no transition | 121.4 / 127.6 |
| w2v2 AV, 10 min | 49 | 2593 | train flat from ~ep109, dev CE rising from ep10 | ep100 150.79 |

**4. Paired-minus-shuffled %Corr, the instrument that works where WER does not.** Above 100 % WER the
number is saturated by insertions and says nothing, while re-scoring each hypothesis against a randomly
permuted reference asks directly whether the output depends on its own audio. Label-gated: evaluation
only, never selection.

| arm | steps | %Corr paired | %Corr shuffled | **binding gain** | WER |
|---|---|---|---|---|---|
| 1 h ep80 | 3252 | 7.81 | 7.34 | **0.47** | 121.4 |
| 1 h ep200 | 8130 | 8.39 | 7.84 | **0.55** | 127.6 |
| 10 h ep20 | 1890 | 12.25 | 6.73 | **5.52** | 133.3 |
| 10 h ep30 | 5670 | 49.45 | 5.86 | **43.59** | 119.5 |
| 10 h ep40 | 7560 | 86.19 | 5.26 | **80.93** | 20.4 |

**5. Unfreeze the BEST-RQ encoder at SFT** — exactly one variable against the frozen 100 h baseline
(same data, LR, schedule, LoRA-A, adapter, eval sets), with the conv front end left frozen and `no_grad`
(the wav2vec2/HuBERT split). A CPU test asserts gradient reaches the conformer and not the front end,
because a left-over `no_grad` would train identically to the frozen arm and make the comparison a lie.

| set | frozen ep10 | **unfrozen ep10** | delta |
|---|---|---|---|
| dev-clean | 21.37 | **10.64** | -10.7 (-50 %) |
| dev-other | 38.75 | **21.10** | -17.7 (-46 %) |
| test-clean / test-other | 23.46 / 38.55 | **10.56 / 22.53** | -12.9 / -16.0 |

**6. Pivot the whole autoencoder onto the §1c-winning wav2vec2 encoder, at a 10 h seed.**
`facebook/wav2vec2-large-lv60`, `hidden_states[15]` (block-14 residual, pre-final-norm), 1024-d @ 50 Hz,
`do_normalize=True` per utterance; AV soft prompt via a x4 adapter to 12.5 Hz, AR target = k-means over
the same layer pooled 50->25 Hz at K=500 rVAD-trimmed, encoder **trainable** at SFT.

| | BEST-RQ (100 h) | **wav2vec2-L15 (10 h)** |
|---|---|---|
| AV dev-clean / dev-other WER | 21.4 / 38.8 | **16.9 / 20.6** |
| mask=none AR usage gate | +0.0036 | +0.0153 (4.2x) |
| p=1.0 AR usage gate | 0.266 | **0.182** (12x the arm's own mask=none) |
| §2.5(d) arbiter at T=0.7 | FAIL (sel > mean everywhere) | **PASS** (spearman 0.286, rank_acc 0.63, sel < mean) |

**7. The AR text-usage gate: CE with true text minus CE with text deranged across utterances**, as a
sisyphus GPU forward reusing the PPL machinery. Two anchors make the number trustworthy: `ln ppl_true`
must reproduce the checkpoint's logged dev CE, and the derangement must be length-matched *within dev*.

| protocol | CE_true | CE_shuffled | gate |
|---|---|---|---|
| global derangement (confounded, withdrawn) | 3.823 | 3.696 | -0.13 (length artifact) |
| **within-dev, length-matched (`val_elements` exactly equal)** | 3.8232 | 3.8268 | **+0.0036 nats/token** |

Against a unit-marginal entropy of 6.140 nats/token, the AR extracts 2.43 nats of reconstruction and
0.004 of it from text.

**8. Break the unit-LM shortcut by starving the teacher-forced unit history.** Masking is a permanent
model property applied in BOTH training and gate scoring, so masked-train + masked-score is a
self-consistent pseudo-likelihood, with a per-arm anchor requiring each masked gate's true CE to
reproduce that arm's logged masked dev CE.

| arm | CE_true | gate | below unigram (6.140) | older-units |
|---|---|---|---|---|
| control (full history) | 3.706 | 0.0053 | 2.434 | 2.43 |
| uniform p=0.5 / 0.7 / 0.9 | 4.618 / 5.171 / 5.920 | 0.021 / 0.072 / **0.242** | 1.522 / 0.970 / **0.220** | -- |
| blind_shift L1 / L2 / L4 | 4.926 / 5.424 / 5.704 | 0.019 / 0.061 / **0.096** | 1.214 / 0.716 / 0.436 | 1.195 / 0.655 / 0.340 |
| blind_window L2 / L4 (leaky, superseded) | 3.796 / 3.814 | 0.0044 / 0.0055 | -- | -- |

**9. Coarsen the target so text determines more of it, and take masking to its conditional-independence
limit.** K in {50, 100, 500} x mean-pool P in {2, 4} on the same pinned features, each with its own AR
re-SFT, screened by the §2.5(c) graded-corruption ladder (word sub/del/swap at eps in {0, .1, .2, .4,
.8}, 500 dev utts). The selection rule was corrected under stress-test before any pick: nats/second is
monotone-increasing in coarseness and is a headroom **floor** only, so the SNR `dR(0->0.2)/per-utt-std`
is the selector, biased to the FINEST candidate that clears it, with a length-preserving lexical-only
variant as a Goodhart guard. The CI scorer is deterministic uniform p=1.0, zeroing every unit input
embedding so each unit is predicted from text + RoPE position alone.

| target | scorer | units/s | ep10 CE | SNR standard (bar 1.0) | SNR lexical | usage gate |
|---|---|---|---|---|---|---|
| K500 (fine ref, anchor reproduces 0.238) | blind_shift-L2 | 20.6 | 5.423 | 0.238 | 0.376 | 0.061 |
| k500_p2 / k100_p2 / k50_p2 | blind_shift-L2 | ~10 / ~11 / ~10 | -- | 0.249 / 0.198 / 0.319 | 0.354 / 0.367 / 0.437 | -- |
| k100_p4 | blind_shift-L2 | ~5.8 | -- | 0.337 | 0.404 | -- |
| **k50_p4 (best)** | blind_shift-L2 | ~5.6 | 3.636 | **0.479** | **0.626** | -- |
| K500 | **p=1.0 (CI)** | 20.6 | 6.003 | 0.350 | 0.490 | **0.266** |
| k50_p4 | p=1.0 (CI) | ~5.6 | 3.732 | 0.331 | **0.630** | 0.283 |

**10. Top-k frame accuracy, true-unit rank and the shared-vocabulary mass leak.** Every AR statement in
the program had been a cross-entropy in nats, and CE 5.61 is equally consistent with "the right unit is
usually rank 2" and "usually rank 200"; same 5000-utterance dev subset, same checkpoints and same seed-42
derangement the usage gate used, CE and ranks from one forward. The leak is that `log_softmax` runs over
all 152,438 logits, so 151,936 frozen text logits compete for normalization and the reward carries a
`log M_t` offset, measured by adding `logsumexp` over the unit slice of the log-softmax already computed.

| arm | pairing | CE | top-1 | mean rank | unit mass | leaked nats |
|---|---|---|---|---|---|---|
| **p10_10h** (incumbent scorer) | true / shuffled | 5.7444 / 6.0774 | **3.17 %** / 2.38 % | 134.7 / 162.4 | 0.9863 / 0.9689 | 0.0151 / 0.0376 |
| **gtrack** (AR_G) | true / shuffled | 5.6213 / 6.1946 | **4.18 %** / 3.26 % | 129.0 / 171.7 | 0.9848 / 0.9476 | 0.0165 / 0.0621 |
| **none_10h** (full unit history) | true / shuffled | 3.2847 / 3.3109 | **27.16 %** / 26.90 % | 23.5 / 23.9 | 0.9949 / 0.9949 | 0.0057 / 0.0057 |

Uniform floor over K=500 is 0.20 %; text-attributable accuracy (true minus shuffled) is p10 **+0.79 pt**,
gtrack **+0.92 pt**, none **+0.26 pt**; renormalizing the softmax to the unit slice would delete 6.8 % /
8.1 % / -0.2 % of each arm's total text discrimination.

**11. The discrete avunits-k500 bake-off, plus a pre-adapter substrate challenger.** Units are k-means
over theta_0's post-adapter states (standardize -> PCA-96 -> K, no dedup at 12.5 Hz), audited against MFA
gold before any SFT spend (K=500 dominates K=100 on purity 0.644 vs 0.497 and oracle-PER 0.453 vs 0.618).
`pre125` is the same protocol on the SFT'd L15 **pre**-adapter states pooled x4 — the user's hypothesis
that the adapter map word-ifies the target, which the unit audit supports (substitutions -40 %).

| arm | ep50 dev CE | marginal | text channel | usage gate | arbiter T=0.7: spearman / rank_acc / eta / gap_true |
|---|---|---|---|---|---|
| **post-adapter p10** | 5.7371 | 6.0072 | 0.270 | **+0.333** | **0.314 / 0.644 / +0.225 / +0.011** |
| post-adapter none | 3.2830 | 6.0072 | -- | +0.026 | weakly positive (eta 0.26/0.05/0.18) |
| pre125 p10 | 5.7075 | 5.9350 | 0.228 | +0.3142 | 0.313 / -- / **+0.145** / -- |
| pre125 none | 3.0867 | 5.9350 | -- | +0.0166 | mixed/negative |

**12. The TTE / continuous-scorer family (Hori-style), three arms.** Target = a continuous frame sequence
rather than cluster ids, removing the quantization bottleneck: first the frozen w2v2-L15 features, then —
after the user's correction that a *teacher-bounded* target is the wrong axis — the seed-SFT'd AV's own
post-adapter state sequence, text-shaped by construction and frozen at dump time. Scorer arms: CI (every
frame slot holds a constant learned query) and prenet (Hori's Prenet-bottlenecked autoregressive frame
feedback, dropout ON at scoring).

| arm | dev recon | usage gate (shuffled - true) | arbiter eta at T=.3/.5/.7 |
|---|---|---|---|
| frozen w2v2 feats, CI | 1.657 (MSE 0.908; mean-predictor = 1.0) | +0.027 (1.7 % rel) | -1.06 / -1.08 / -0.60 |
| avstates, CI | 1.6708 | **+0.0611** (3.7 % rel) | -1.23 / -0.96 / -0.51 |
| avstates, prenet | **1.3046** (-22 % rel vs CI) | +0.0115 (0.9 % rel) | -0.87 / -0.83 / -0.50 |

**13. §3b/B0 — stream candidates through one fixed chain.** Every candidate is a different bet on raising
the 0.26 nats/unit gold text explains, and each runs the identical chain before any loop compute (unigram
entropy -> text-only p10 seed-AR CE -> usage gate -> §2.5(d) eta), sharing the theta_0 rollout policy, the
fixed 128-utterance rollout set and the seed dev subset. Arms: (a) `k100`; (b) `perutt_k500`,
per-utterance state standardization; (c) `brown_k100`, bigram-context merge of the k500 inventory;
(e) `ceiling_k500`, the incumbent stream with the AR trained on 100 h gold text, eval-only. Pre-registered
bar **eta(T=0.7) >= 0.35 AND gap_true > 0 at T <= 0.7**, with (a) vs (c) the designed contrast — same
inventory size, one coarsening in state geometry and the other in distributional identity.

| arm | K | H_uni | H_bi | CE_true | usage gate | text-expl (% of H) | **eta(0.7)** [95 % CI] |
|---|---|---|---|---|---|---|---|
| incumbent k500 (10 h AR) | 500 | 6.0072 | 3.0979 | 5.7444 | +0.3331 | 0.2628 (4.4 %) | **0.2246** [0.077, 0.359] |
| (a) k100 | 100 | 4.4122 | 2.6513 | 4.1788 | +0.3373 | 0.2334 (5.3 %) | **-0.1255** [-0.387, 0.098] |
| (b) perutt_k500 | 500 | 6.0220 | 3.1478 | 5.7723 | +0.2967 | 0.2497 (4.1 %) | **0.0069** [-0.184, 0.176] |
| (c) brown_k100 | 100 | 4.5038 | 2.5677 | 4.2791 | +0.3047 | 0.2247 (5.0 %) | **-0.0370** [-0.303, 0.191] |
| (e) ceiling_k500 (100 h gold AR) | 500 | 6.0072 | -- | 5.5985 | **+0.6101** | **0.4087 (6.8 %)** | **0.0223** [-0.178, 0.205] |

`gap_true` is positive on every arm (+0.0059 to +0.0120). Arms (a) and (c) sit on a different rollout draw
(`num_units=100` builds a differently-sized AR and consumes a different amount of RNG) so their eta is
unpaired; (b) and (e) share the incumbent's draw exactly and admit a paired bootstrap on
`mean_wer - sel_wer`, which the incumbent wins against both at P = 0.989.

**14. The §2.5(d) reward-rank arbiter, on REAL theta_0 rollouts rather than synthetic corruptions.**
Replays `grpo_step`'s sample->score path (byte-identical reward) on 128 seed-train utterances x G=12 x
four temperatures, decodes each sample, computes its WER against truth (eval-only) and reports spearman,
pairwise rank accuracy, selected WER against group mean / oracle / **theta_0 greedy**, and `gap_true`. Two
probes fixed the sampling regime first: T=1.0 gives `n_uniq` 11.9/12 with every sample opening correctly
and then derailing into multilingual token soup (degenerate diversity, not near-duplication), and mean
reward is flat across T (-5.387 at T=0.3 vs -5.443 at T=1.0), i.e. coherent near-correct transcripts
reconstruct the units no better than garbage.

| arm (T=0.7 unless noted) | spearman | rank_acc | sel_wer vs mean / greedy 0.0525 | gap_true |
|---|---|---|---|---|
| BEST-RQ blind_shift-L2 K500 | -0.028 | 0.49 | sel > mean at all T | negative |
| BEST-RQ p=1.0 k50_p4 | 0.080 | 0.54 | sel > mean at all T | negative |
| oracle phone screen (lexicon-G2P edit distance, no training) | **0.794** | **0.879** | 0.0430 < mean 0.1076, **< greedy** | >0 in 98 % of groups |
| w2v2 p=1.0 K500 | 0.286 | 0.63 | 0.092 < mean 0.108, **> greedy** | +0.0066 |

**15. Audit the arbiter itself: precision, G-dependence, bed confound.** Bootstrap over the existing n=128
dumps (2000 resamples) at zero GPU cost, then an n=512 re-run on ONE bed with G in {4,12,24} and five
temperatures, both arms on the SAME gold tc100 draw, because sub-sampling assumes the only thing G changes
is how many iid rollouts land in a group. The two n=128 sets overlap by one utterance, so "0.225 at the
paired init versus -0.046 here" confounds init with bed.

| read | theta_0 / anchor | gtrack / §3d gate |
|---|---|---|
| eta(T=0.7), n=128 | +0.2246 [+0.082, +0.369] | -0.0462 [-0.202, +0.096] |
| **eta(T=0.7), n=512, same bed** | **-0.023** | **-0.172** |
| gap_true, n=512 | **+0.0124** | +0.0020 (6.2x apart) |
| reward std within group | 0.0324 | 0.0124 (2.6x) |
| spearman at G=4 / 12 / 24 | 0.139 / 0.170 / 0.180 | 0.065 / 0.094 / 0.116 |
| eta at G=4 / 12 / 24 | +0.146 / -0.023 / +0.011 | -0.002 / -0.172 / -0.204 |
| greedy WER / bed oracle WER at T=0.7 | 0.1196 / 0.0316 (10 h seed = theta_0's own SFT data) | 0.1345 / 0.1113 (tc100) |

CI half-width scales +-0.15 at n=128, +-0.075 at n=512, +-0.05 at n=1152. Read by eye, the best-pick group
has all 12 rollouts sharing one skeleton and differing almost only in proper-noun spelling (`cosett`,
`coxett`, `cashett`, `kosetz`, ...), 11 of 12 carrying one systematic error no reranking can reach, and
spanning 0.021 nats of reward across candidates whose WER spans 0.212 to 0.515; in the worst-pick group the
gold transcript itself scored below 10 of the 12 machine samples.

**16. Reward shape: an offline weight sweep, an audio-free null, and the statistic GRPO follows.**
`dump_reward_parts` adds `recon` / `lm_prior` / `len_hinge` columns to `rollouts.jsonl`, so every candidate
weight is a CPU re-rank (92 weights x 5 temperatures x 512 groups per arm, ~40 s) with the
`lam_lm=0, lam_len=0` cell asserted to reproduce `reward_rank.txt` before any swept cell is written. The
**audio-free null** scores `lm_prior - c*len_hinge` with no acoustic model at all (the hinge divides by
duration, so it knows how long the utterance is and nothing about its content), and **audio margin** =
eta(shaped) - eta(best audio-free), selected honestly by a split over utterances. `grad_align =
-mean_i(A_i * WER_i)` with `A` standardized within the group is the first-order expected WER change per
unit of policy movement, i.e. the training-relevant quantity a top-1 statistic is not.

| theta_0 at T=0.7 (mean_wer 0.1731, oracle 0.0768) | spearman | rank_acc | sel_wer | eta | grad_align |
|---|---|---|---|---|---|
| `recon` (live) | 0.170 | 0.571 | 0.1753 | **-0.023** | +0.0347 |
| `recon - 1.0*hinge` | 0.200 | 0.584 | 0.1656 | +0.078 | -- |
| `recon + 0.075*prior` | 0.414 | 0.687 | 0.1788 | -0.059 | -- |
| **`recon + 0.075*prior - 1.0*hinge`** | **0.454** | **0.704** | **0.1477** | **+0.263** | **+0.0500** |
| audio-free null (best) | **0.466** | -- | 0.1621 | +0.115 | +0.0420 |

The optimum is broad, not a spike. The hinge fires on **5.4 %** of rollouts at a mean penalty of 0.235
against a within-group reward std of 0.032 — roughly 7 sigma, a veto rather than a preference — and what it
vetoes has mean WER 0.4762 against 0.1558 for the rest. Honest audio margin averages **+0.148** across five
temperatures for theta_0, all positive, against -0.022 to +0.033 for AR_G; `lam_lm` is well determined
(0.03-0.15 in every fold) while `lam_len` is **not identified** (folds pick 0.25, 1, 2, 8).

**17. Is the reward blind off the seed bed?** The finished reward-rank probe re-run verbatim on
train-clean-100 minus the seed — same theta_0, same AR, same k500 units, same G and temperatures, only the
utterances change. At T=0.7 eta goes 0.2246 on-bed -> 0.1623 off-bed and spearman 0.3138 -> 0.2437 with
`sel_wer` still below `mean_wer`; at T=0.3/0.5 off-bed eta is *higher*, and the text-blind control AR sits
at eta ~0 off-bed, so text conditioning is intact.

**18. The loop at the 10 h seed: one bed, one init, six rewards.** All arms start from theta_0 ep50 and run
the oracle-calibrated knobs (LR 1e-5 cosine, T=0.7, G=12, 4 passes, per-epoch dev recogs); only the reward
and the AR's frozen/joint status differ, verified from each generated config. The oracle arm (reward =
-WER against gold, legal inside the quarantined 2S arm) is the bring-up positive control — if the policy
does not move under a *perfect* reward the failure is loop mechanics and no target swap helps — and the
self-training control is theta_0 continued with CE on the §1d word-decode pseudo-transcripts (17.96 /
21.87 WER, 28,539 utterances, 10x the loop's audio). A drift arbiter re-runs the reward-rank probe per
epoch on *fixed* theta_0 rollouts with only `ar_checkpoint` swapped, identity-checked at 6400/6400 texts.

| arm (dev-clean / dev-other), theta_0 = 16.91 / 20.64 | ep1 | ep2 | ep3 | ep4 |
|---|---|---|---|---|
| oracle, -WER vs gold (eta = 1) | 13.37 / 14.83 | 12.10 / 14.47 | 11.56 / **14.02** | **11.05** / 14.82 |
| frozen AR, recon | **13.07 / 15.89** | 13.87 / 16.51 | **12.99** / 16.20 | 14.47 / 17.09 |
| joint AR, recon | 14.74 / 16.87 | 13.68 / 16.23 | 13.33 / **15.93** | **13.15** / 16.13 |
| frozen AR, anchored (lm 0.01 / kl 0.02), dev-clean only | 16.20 | **12.91** | 13.91 | 13.99 |
| self-training control (10x audio) | 13.99 / 17.62 | 13.16 / **17.13** | 13.12 / 17.87 | **13.05** / 17.74 |
| **shuffled-reward null** | **202.54 / 207.59** | -- | -- | -- |
| joint-AR drift on fixed theta_0 rollouts (spearman / eta) | 0.075 / -0.66 | 0.169 / -0.30 | 0.186 / -0.36 | 0.161 / -0.38 |

**19. The shuffled-reward null: within-group reward permutation.** Rewards are permuted uniformly at random
inside each GRPO group, immediately after the reward is composed and immediately before the advantage is
normalized; nothing else changes. This is the right null because a group's mean and std are
permutation-invariant, so the *multiset* of advantages the optimizer sees is exactly the one the real arm
produced — same magnitudes, same gradient norms, same clipping — and only the pairing of advantage to
rollout is destroyed. The AR CE channel is untouched, so the control isolates what the *policy gradient*
gets out of the ranking and leaves self-distillation running.

**20. Stage B: the same loop over 100 h of unlabeled tc100 and over 960 h.** No 10 h-seed/100 h-audio point
existed (the validated track looped on 10 h of audio, the scale-up on 960 h), so without this control the
rungs have no fixed-loop-audio reference. Joint AR, `partition_epoch=4` / `num_epochs=8`, warmup
step-matched to the 10 h arm.

| 10 h-seed arm, dev-clean | ep1 | ep2 | ep3 | ep4 | ep5 | ep6 |
|---|---|---|---|---|---|---|
| 100 h audio, recon only | 112.45 | 207.89 | 232.61 | 156.99 | -- | -- |
| 100 h audio, + `lam_len` 0.2 | 83.38 | 74.67 | 63.87 | 70.21 | 40.75 | 42.40 |
| 960 h audio, frozen AR + `lam_len` | 84.23 | 92.29 | -- | -- | -- | -- |
| 10 h audio, frozen AR (the bed that works) | 13.07 | 13.87 | 12.99 | 14.47 | -- | -- |

Diagnosis on the same arms: %Corr stays at bed level (86.1 at 100 h ep1 against theta_0's 89.1) and binding
gain 72-78 against theta_0's 83.8, so the transcript is intact and the WER is pure insertion (Ins 98.6);
under `lam_len` the dev hyp/ref word ratio is 1.613 with length correlation still +0.70, i.e. the policy
emits a *training-typical* length (32.5 words, near tc100's own gold 34.8) regardless of the 7.2 s dev
utterance in front of it.

**21. The speaking-rate hinge, calibrated on the real corpora before being switched on.** tc100 is 14.552
chars/s (dev-clean 14.872), so the shipped `nu_chars_per_sec` 15.0 is within 3 %, but `len_eps` 0.2
penalizes 18.1 % of *gold* transcripts asymmetrically toward "too short" — the term as configured pushed
toward the over-generation it exists to stop.

| len_eps | % gold penalized | mean gold hinge | hinge on a 2.5x exploit | ratio |
|---|---|---|---|---|
| 0.2 | 18.1 | 0.0180 | 0.716 | 40:1 |
| 0.3 | 5.9 | 0.0072 | 0.616 | 86:1 |
| **0.4 (adopted, with nu 14.55)** | ~1.6 | 0.0024 | 0.516 | **215:1** |

**22. The 100 h reward sweep: four arms identical but for the reward.** The three terms had never been
combined anywhere — the 100 h/960 h arms ran hinge-only, the best 10 h arm ran anchors-only. Read at ep1
dev; the sweep was closed before ep2 on the user's call.

| arm | dev-clean | dev-other | %Corr (do) | hyp/ref (do) | in-train text_len (tok) | lm_prior |
|---|---|---|---|---|---|---|
| `lm0.05` (prior only, 5x) | 693.21 | 702.40 | 14.06 | 7.16 | **100.0000** (cap, never emits EOS) | -0.4168 |
| `lm0.01` (prior only) | 362.67 | 361.02 | 74.51 | 4.35 | 91.62 | -2.3494 |
| `lm0.01_len0.2` | 221.41 | 236.21 | 35.13 | 2.54 | 51.85 | -2.7147 |
| `lm0.01_kl0.02_len0.2` | 81.69 | 97.89 | 89.76 | 1.69 | 54.17 | -4.3732 |
| hinge only, no prior | 83.38 | 98.24 | -- | 1.61 | 42.1 | -- |

`recon` is identical across all four (-5.717 to -5.766) while the policies differ 2x in length and 7x in
fluency, and `reward_std_within_group` is 0.0074-0.0142 on a reward of magnitude 5.7. Information
accounting on the loop's own stream: unigram entropy 6.0072, AR given gold text with unit history masked
5.7371, AR given gold text and unit history 3.2830 — gold text explains **0.27 nats/unit** where the unit
history explains 2.72, and that 0.27 is an upper bound measured on the seed's own dev.

**23. §3c — mix the sanctioned 10 h paired CE back into both loop objectives (Hori/TTE).** AV term = CE of
the gold transcript given the audio through the policy's own teacher-forced scorer; AR term = CE of that
utterance's units given the gold transcript under the same p10 masking the reward uses. The quarantine is
**structural, not procedural** — the arm trains on a dir where `text` is kept only on the 2849 seed
utterances and blanked on the other 25,690, so "this row has text" *is* "this row is a sanctioned pair",
checked against the real arrow data by set equality of ids rather than a count match. The mixing weight is
measured, not guessed: loss *values* cannot set it because the GRPO surrogate is ~0 at the sampling point
by construction, so a probe runs one real GRPO step and one replay step on the same 128 utterances and
compares gradient norms on the same parameters — `replay_av` 13.84 and `replay_ar` 16.99 against `pg` 6.45
and rollout `ar_ce` 16.93, giving lambda_av 0.466 and lambda_ar 0.996. Read at the pre-registered
matched-compute point: the 10 h arm's entire run was 5700 optimizer steps, which this arm passes during ep2.

| sub-epoch | dev-clean | dev-other |
|---|---|---|
| theta_0 (no loop) | 16.91 | 20.64 |
| ep1 | 18.79 | 25.59 |
| **ep2 (matched-compute point)** | **23.94** | **29.22** |
| ep3 / ep4 | 42.17 / 46.71 | 50.00 / 51.42 |

**24. Does the k500 codebook need refitting on the full corpus?** A refit invalidates the AR (units are its
target vocabulary) and forces a retrain, so it was screened first: a sharded streaming fit (12 shards /
~185 GB, vectorized Algorithm-R reservoir) against a **reseeded 10 h codebook as the null**, both scored on
the same 980,195 held-out frames with the seed utterances dropped.

| vs the shipped cb10h | NMI | 1-1 Hungarian | MSE | var expl | vs A |
|---|---|---|---|---|---|
| cb960h (38.97 M frames / 253,038 utts) | 0.7609 | 0.5053 | 44.736 | 0.3722 | **+0.07 %** |
| cb10h reseeded (seed 43, the null) | 0.7605 | 0.4943 | 44.786 | 0.3715 | **-0.04 %** |

## Conclusion

1. (2) "Silence collapses to a couple of tokens" is **WRONG** — a 1 s pause costs ~13 deduped tokens over
   ~90 cluster IDs and silence is as expensive as speech under context-only prediction while being
   unrecoverable from text, a ~0.36 nats/unit z-inert floor in every AR target; user ruling: trim.
2. (2) The larger finding is **global flicker** — the stream runs at 2x the true phone rate and ~1/3 of AR
   target tokens carry no transcript information, so every per-unit threshold is diluted by the alternation
   share and every rate calibration must use the measured ~20 tok/s.
3. (3) The 10 h BEST-RQ AV failure was **data scale, not the adapter** (no implementation defect, the
   feature-scale concern refuted, the 100 h arm groks at ep7-8 on the identical recipe), so the earlier
   "the thin adapter is the bottleneck" lean is WRONG.
4. (3, 4) **The 1 h rung fails mechanistically, not gradually**: at 8130 steps, past the 10 h gold's
   transition window with train loss plateaued, its binding gain is 0.55 against the 10 h AV's 5.52 at 1890
   steps — fluent, length-correlated LibriSpeech prose carrying no phonetic content.
5. (3) "More steps is the lever" is **WRONG** for both small rungs: the 200-epoch 1 h rebuild reached 8130
   steps and never transitioned, and 10 min is data-limited (train loss flat from ~ep109 while dev CE climbs
   from ep10, i.e. 49 utterances memorized).
6. (4) **The 10 h phase transition is not the moment acoustic binding forms** — binding is already at 43.6
   gain by ep30 where WER 119.5 looks like ep20's 133.3, and what changes at ep30->ep40 is *stopping*
   (Ins 69.1 -> 6.5) — so any >100 % WER reading must be resolved by binding gain first.
7. (3) Pinning theta_0 at the **dev-CE minimum selects the worst checkpoint by WER** in the pre-transition
   regime (1 h: ep7 470.18 against ep100's 121.4 at a *higher* CE), where the two metrics are anti-aligned;
   a train-loss-plateau pin has no such failure mode.
8. (3) The rung AR numbers are void: `num_epochs` was hardcoded to the **AV's** pin epoch, so both rung ARs
   were cut far short, sit *above* the uniform ceiling ln 500 = 6.2146 (6.379 / 9.979) and were still
   falling steeply — which the built-but-skipped per-rung reward-rank gate would have caught for one forward.
9. (5) Unfreezing the encoder at SFT **~halves WER on all four sets**, so the frozen BEST-RQ was a bigger AV
   bottleneck than the data-scale read implied; targets `u` stay pinned to the frozen-encoder dump
   regardless, since a drifting target makes the reward non-stationary.
10. (6) The encoder swap moved the wall in the right direction at 1/10 the data — half the dev-other error
    of the 100 h BEST-RQ AV, a 4.2x larger mask=none gate, and the first arbiter PASS — with the mask=none
    control isolating the cause: same encoder and units, only the shortcut differs.
11. (7) **The AR ignores the text bottleneck** (+0.0036 nats/token against 2.43 nats of reconstruction), and
    that is a design consequence rather than a bug but not physics either: given unit history the gold
    current phone still buys >= 0.39 nats/token the AR takes none of, a source-ignoring shortcut.
12. (8) **blind_window never blocked u_{t-1}** and its verdicts are **WRONG** — the band was expressed in
    input-position space so it hid lags 2..L+1, and no attention mask can hide lag 1 because u_{t-1} is the
    query's own input embedding reaching the head through the residual stream.
13. (8) With the leak-free `blind_shift` head **a clean local block moves the gate only to O(0.1)** and even
    full starvation tops at 0.242 on a near-degenerate model, refuting the shortcut story and
    re-establishing the target wall on a valid instrument.
14. (8) Good reconstruction and text-sensitivity are in **hard opposition** for this target — masking only
    trades one for the other — so it is a diagnostic, never a recipe.
15. (9) Coarsening roughly doubles the discrimination SNR (0.238 -> 0.479) but the best candidate is still
    ~half the bar and the trend points into the saturation corner, where many-good-texts -> one-u makes the
    reward saturate among good rollouts and GRPO plateau at a lexical-discriminability floor.
16. (9) The CI scorer **maximizes the usage gate and still fails §2.5**, re-confirming the true-vs-deranged
    gate as necessary-not-sufficient; its lexical-over-standard SNR signature says position-only RoPE
    alignment is the binding weakness.
17. (10) **The incumbent's frame error rate is 96.8 %** — given the *gold* transcript it puts the correct
    unit first 3.17 % of the time and ranks it 135th of 500, of which the transcript accounts for 0.79
    points — while the mask=none control shows over 99 % of its own 27.2-point accuracy is unit-to-unit
    continuity, i.e. a reward built on it would be a smoothness detector in a reconstruction reward's
    clothes.
18. (10) An earlier claim that the off-vocabulary softmax mass is a defect costing resolution is **WRONG on
    both counts**: a scoring-time restriction is rank-preserving, and the leak is signal, carrying 6.8-8.1 %
    of each working arm's text discrimination and exactly zero for the arm that barely reads text.
19. (10, 14) AR_G is the better scorer on frame accuracy (top-1 4.18 vs 3.17, text-attributable +0.92 vs
    +0.79) while ranking rollouts *worse* by eta — a second independent instrument showing that
    **information content and ranking ability are separate axes**.
20. (11) **avunits-k500 p10 is the first reward in the program to pass the arbiter at coherent temperatures**
    (spearman 0.314, rank_acc 0.644, eta +0.225, gap_true positive everywhere), with a gate large enough
    that deranged text scores worse than the unigram marginal.
21. (11) The pre-adapter substrate hypothesis fails — the adapter map does discard phone identity (-40 %
    substitutions without it) but pre125 ties on spearman and loses selection at every temperature.
    (Re-decided under psi_align — see `SAE_3A.md`.)
22. (12) The continuous family is **conclusively out**: at the loop's calibrated T=0.7 both CI and prenet
    select worse than greedy, and prenet reconstructs 22 % better while being *less* text-sensitive, the
    known Hori frame-history dilution.
23. (13) **Every B0 stream candidate fails the pre-registered bar**, and not narrowly — no arm reaches the
    incumbent's own 0.2246, three of four sit at or below zero, and paired the incumbent beats both
    same-draw arms at P = 0.989.
24. (13) **The ceiling instrument falsifies its own premise**: the strictly better text-conditioned model of
    the identical stream (CE 5.5985, usage gate nearly double, text-explained +56 %) returns eta 0.0223,
    outside both pre-registered branches and *below* the AR it was built to upper-bound.
25. (13) So **eta is not a monotone readout of text-explained information**, the assumption every rung of
    that chain was built on — +56 % information bought -90 % eta — which voids CE / gate / text-expl as
    search directions for the stream.
26. (14) On real rollouts the BEST-RQ rewards were **anti-useful** (sel_wer > mean_wer at every coherent
    temperature, rank_acc at chance, gap_true negative), so GRPO would have pushed the policy away from
    truth; the strong T=1.0 numbers are soup-vs-coherent, not near-miss ranking.
27. (14) The §2.5(c) graded-corruption ladder was the optimistic proxy and the rollout probe is loop truth —
    its 20 %-word corruption is a clear degradation where real groups are tight ~10-15 % WER near-misses on
    which the reward is flat.
28. (14) The **oracle phone screen clears every bar the K500 reward fails** (spearman 0.73-0.79, rank_acc
    0.86-0.88, sel below greedy at all coherent T) with noisier teacher targets, but the user closed that arm
    on principle: ranking toward a pseudo-transcript teacher is the wrong axis for exceeding it.
29. (14) Against the honest comparator the K500 reward has little practical value — **greedy_wer 0.0525
    beats its selected 0.058-0.092 at every coherent temperature** — and a reward that only beats a random
    rollout is worse than free argmax decoding.
30. (14) **`gap_true = -0.41` is WRONG and was a probe bug**: the data path bypassed the lowercase
    post-processing and scored UPPERCASE-tokenized true text against a lowercase-trained AR, so corrected it
    is +0.0025 / +0.0037 / +0.0066 and the criterion passed the first time it was measured correctly.
31. (15) **eta(T=0.7) >= 0.2 is WITHDRAWN as a gate** — on the bed the §3d gate actually used, the incumbent
    whose loop demonstrably runs reads -0.023 and fails as decisively as the arm the gate rejected, and at
    n=128 the only supportable claim was the weaker one it was used for (upper bound +0.096 excludes 0.20).
32. (15) **eta is a top-1 statistic and GRPO is not a top-1 algorithm**: measured at n=512 the reward's
    ordering improves with G on both arms while its argmax pick does not, so the gate read the one statistic
    that gets worse with the sample size the loop uses.
33. (15) The sampler produces twelve variants of a single hypothesis and **the variation lives exactly where
    WER is most sensitive and an audio-derived unit stream is least** — proper-noun orthography — which
    explains a near-zero eta without the AR needing to be blind, and there may be no temperature at which
    this policy emits semantically distinct plausible hypotheses.
34. (16) Neither shaping term works alone — the prior more than doubles spearman while making the argmax pick
    *worse* (a language model's favourite mistake is fluent text of the wrong length), the hinge alone barely
    moves the ordering — but together they take selected WER -15.7 % relative and eta from below zero to
    +0.26 at every temperature.
35. (16) **Most of the fix is free English**: the best audio-free ranker reaches eta +0.115 and is *ahead* of
    the shaped reward on spearman, so the language model orders the group and the AR only breaks the tie at
    the top, which is also why spearman and eta disagreed throughout.
36. (16) On `grad_align`, the statistic GRPO actually follows, the AR-free null alone reaches 84 % of the
    shaped reward's alignment, so the acoustic model is a sixth of the improvement where eta made it look
    like a large share.
37. (16) theta_0's AR contributes and AR_G's does not (honest margin +0.148 versus -0.022 to +0.033 even when
    free to choose its own weights), so AR_G's extra information is **redundant with what a language model
    already knows** — stronger than "not worth funding", since it survives the composition being fixed. An
    earlier "AR_G significantly negative at T=0.7" reading was **WRONG**, an artifact of applying theta_0's
    weight to AR_G.
38. (17) **The reward does not go blind off the bed** — it loses about a quarter of its selection efficiency
    and stays clearly informative, which cannot carry a 13.07 -> 83.38 collapse — so every "distribution
    shift / off-bed" account in earlier revisions is **WRONG**: the 10 h seed is a strict subset of tc100
    (2849/2849 ids, same source job).
39. (18) **Loop mechanics: GO** — under a perfect reward the fixed loop moves -35 % / -32 % relative from
    theta_0 with train rollout WER saturating at zero, so any later flatness indicts the reward, not the
    plumbing.
40. (18) **RUN-1/RUN-2 "the reward magnitude is the bottleneck" is WRONG and withdrawn**: the autocast
    cast-cache bug severed the policy gradient — RETURNN wraps the whole `train_step` in one bf16 autocast
    region, the no-grad sample/scoring phase cast every fp32 trainable to bf16, and PyTorch stored those
    casts as detached constants the grad-enabled re-forward silently reused — so the AV LoRA-A and adapter
    received exactly zero gradient and those runs measured a structurally frozen policy; advantages are
    std-normalized anyway, so reward *scale* was never the axis.
41. (18, 19) **The reward is load-bearing for stability**: with the ranking destroyed and everything else
    bit-identical — same rollout draws, same advantage mean and std, same `ar_ce` — the loop runs away to
    >200 % WER inside one epoch, so the 10 h arm's gain is not self-distillation and not an anchor artifact.
42. (18) **But the 10 h loop was never reward-*driven***: a perfect ranker buys 5.86 points and the
    near-noise reward buys 3.76, which ranking quality cannot explain — theta_0 was SFT'd on exactly the
    utterances the loop trains on (greedy WER 5.25 % there against 13.90 % on the complement), so GRPO
    sharpening onto its own confident samples is sharpening onto near-correct text.
43. (18) **For run-to-completion unsupervised training, joint-AR is the better recipe** — it loses ep1 to
    co-adaptation shock, crosses over at ep2 and wins the endpoint decisively (13.15 / 16.13 vs
    14.47 / 17.09) — and frozen's cherry-picked bests need dev WER, which is a supervised signal.
44. (18) The frozen loop's gains are **front-loaded and then over-optimized** while the oracle run improved
    monotonically on the same schedule, so this is Goodhart against a fixed reward on a tiny set, not the LR
    schedule.
45. (18) **The theta_0-anchored drift probe underestimates an adapting reward** — the joint AR's rank quality
    on fixed theta_0 rollouts collapses monotonically (eta +0.225 -> -0.66) and never recovers while the
    policy keeps improving — so trajectory WER is the decisive instrument.
46. (18) **The loop beats pseudo-label self-training at matched start and compute** with the control holding
    10x the audio, and has no teacher ceiling (loop dev-other 15.89 against the pseudo-teacher's 21.87); it
    trained on the **10 h seed audio only** (2849 utterances), so the comparison was compute-matched but
    data-favored the control.
47. (20) The 100 h collapse is **pure insertion on an intact transcript** — %Corr and binding gain stay at
    bed level while Ins runs to 98.6 — and the arm is broken after 3567 steps where the 10 h loop runs 5696
    with Ins <= 1.8.
48. (20) 100 h vs 960 h is a **step artifact, not a data-scale result**: matched by steps they coincide (gain
    72.5 at 10,701 vs 72.4 at 12,784), and comparing them "at ep3" compares 38,352 steps against 10,701.
49. (21) `lam_len` converts divergence into a plateau at ~41 dev-clean at no cost in binding, which is still
    2.4x worse than the arm's own init — damage control, not a fix.
50. (21) **A hinge evaluated only on training audio cannot teach length tracking on durations it never sees
    penalized, and its own monitor cannot detect the failure**; citing in-train rollout length (41.7 healthy
    vs 42.1 exploiting) as evidence the exploit was suppressed is **WRONG**, since both runs are ~10 % long
    on training audio and diverge only on dev.
51. (22) The LM prior **is** the over-generation driver — unbraced `lam_lm` gives a clean monotone
    dose-response on WER, binding and length, and at 0.05 saturates the cap at exactly 100.0000 because
    maximizing a per-token mean logprob suppresses termination.
52. (22) The KL anchor is load-bearing but only **buys back what the prior costs** (97.89 dev-other fully
    braced against the hinge-only arm's 98.24), refuting "the prior alone would work".
53. (22) The reconstruction reward's entire dynamic range is **0.27 nats/unit** and the spread GRPO sees
    across its 12 candidates (std 0.0074-0.0142) is 3-5 % of that, so no lambda in the sweep should have
    crossed theta_0 = 16.91 — the lever is the unit stream's identity space, not the term weights.
54. (22) "The 0.27 nats is what separates 100 h from 10 h" is **WRONG** — both beds ran the same reward, and
    it explains why *neither* bed is reward-driven.
55. (23) Seed replay **fails its own pre-registered comparison** (ep2 23.94 / 29.22 against the 10 h arm's
    final 13.15 / 16.13, and worse than doing nothing from ep1 onward), so it does what Hori/TTE promised —
    it slows the 100 h collapse that killed every prior attempt on this bed — and never beats doing nothing.
56. (23) **lambda_ar ~ 1.0 is a finding, not bookkeeping**: on the AR side, conditioning on the *gold*
    transcript is nearly indistinguishable from conditioning on a *sampled* one (grad norms 16.99 vs 16.93,
    losses 5.677 vs 5.688), so supervision enters through the AV term only.
57. (23) The first lambda measurement is **WRONG and withdrawn** — it scored the dataset's UPPERCASE target
    stream while the loop's training data goes through `LowerCaseTextAndApplyVocab`, voiding both lambdas
    and both replay CEs; the arm launched on them was killed at ~40 min by its own monitors, which read
    `replay_av_ce` 0.086-0.306 against the probe's 2.723.
58. (24) **The codebook axis is null**: a 960 h refit is, if anything, marginally *more* similar to the
    shipped 10 h codebook than a reseeded 10 h codebook is (NMI 0.7609 vs 0.7605), so 96x more audio buys
    +0.07 % distortion where a different random seed costs -0.04 % — which says the 960 h codebook is not a
    better *description*, not that the units are good.

## Catalog

`T/` = `work/i6_core/returnn/training/`, `F/` = `work/i6_core/returnn/forward/`,
`S/` = `work/speech_llm/sae/`.

| artifact | path |
|---|---|
| code | `sae/grpo/{reward,grpo,trainer,fused_ce,psi_scorer}.py`, `sae/{vocab,data,build_units,feats}.py`, `definitions/{sae_token_lm,sae_continuous_scorer}.py`, `encoders/{bestrq,wav2vec2}.py` |
| **theta_0 (w2v2 10 h AV SFT, trainable encoder), ep50** | `T/ReturnnTrainingJob.OLzy9Q2oC3mU` |
| **loop reward: p10 AR on avunits-k500**, ep50 | `T/ReturnnTrainingJob.ExCoQDKtXAGH`; mask=none control `.STEMNbqQUNn2`; AR_G `.cGl2KHUclIlP` |
| units / states the reward scores | `S/quantize_states/QuantizeStatesJob.Zp4b9U9L3gSQ` <- `S/av_states/AvStatesJob.rB2v0ymrzJBR` |
| **10 h loop: frozen / joint / anchored** | `T/ReturnnTrainingJob.{qmkzvAX3gOVW, MquKQUTRgZj9, iNZtd4CRrSLR}` |
| oracle-reward control / shuffled-reward null | `T/ReturnnTrainingJob.zAzwF2KbdMmB` / `.FBOVGb4QZwzW` (`config/sae_2s_grpo_shuffled.py`) |
| self-training control (word-decode pseudo-labels) | `T/ReturnnTrainingJob.xChfzEkd4CGE`; labels from `Wav2Vec2KenlmDecodeJob.AQw3EcUo6rks` (8-shard) |
| §2.5(d) arbiter (v2, casing-fixed) | `F/ReturnnForwardJobV2.{2muM9qHQhqxX, 0ZRE9RsTI8Zq}`; avunits anchor `p9y6xUfCZ4sW` (the drift series' anchor); §3d gate bed `faxctn9Uzcn6`; n=512 power probe `config_sae_2s_rewardrank_power_v1` |
| reward-component dumps (31,744 rows each) + offline sweep | theta_0 `yHGmUzeStu13`, AR_G `J9yA1eYnxwYA` (`config_sae_2s_rewardrank_parts_v1`); `config_sae_2s_rewardrank_sweep_v1` -> `RewardShapeSweepJob` |
| frame-accuracy probe (6 GPU forwards, ~25 min) | `config_sae_2s_ar_fer_v1`; `ar_batch.ar_masked_frame_stats` -> `sae_ar_fer_forward_step_v1` -> `ArFerCallbackV1` -> `fer.txt` |
| §3b/B0 graph (40 jobs) + streams | `config/sae_2s_b0.py`; (b) `S/quantize_states/QuantizeStatesJob.CRgBYjWDbAdQ`, (c) `S/unit_streams/BrownMergeUnitsJob.hneMgEmTsroC`, withdrawn (d) `S/unit_streams/PhoneUnitsJob.kydNe4nXVwwm`; CI reproduction `scripts/b0_eta_ci.py` |
| §3c replay: quarantined dir (2849 transcribed of 28,539), lambda probe, arm | `TransformAndMapHuggingFaceDatasetJob.mQmb6aW1IDH5` (`config/sae_2s_replay_data.py`); `F/ReturnnForwardJobV2.9UBX9QocKb1q`; `T/ReturnnTrainingJob.Z75ybCfEzNjS` (killed) and its successor, held 2026-08-05 at ep4 |
| Stage A rung chain | `config/sae_2s_seed_rungs.py`; AV `T/ReturnnTrainingJob.0KDpBlRUqdXq` (1 h 200-ep rebuild) |
| rung seed dirs (nested) | `TransformAndMapHuggingFaceDatasetJob.TAZO5vh3T7X2` (1 h) from `.OYvh9012Pgkb` (tc100) |
| Stage B 100 h / 960 h loop | `config/sae_2s_grpo_loop_100h.py`; `T/ReturnnTrainingJob.O1r2sJhRJ3cW` (recon only, deleted), `.XoZIrup8WRQK` (+`lam_len`, deleted at ep6), `.o4hSsPrvw3z4` (960 h, kept) |
| 100 h reward sweep (all four cancelled and deleted 2026-08-03) | was `T/ReturnnTrainingJob.{SF3axUKnMuh2, JBmChbW076eE, XGVyzCiIp9PG, AGUN11j0kppQ}`; `BUILD_SWEEP = False` |
| off-bed reward-rank probe | `config/sae_2s_rewardrank_offbed.py` (per-rung gate built, never launched: `config/sae_2s_rewardrank_rungs.py`) |
| codebook probe | `config/sae_2s_codebook_960h_probe.py` -> `FitQuantizerShardedJob.{ahYLt7n9kLwe (960 h), 55FbsynFCJyK (10 h seed 43)}` |
| pre125 challenger / continuous scorers | `T/ReturnnTrainingJob.{bhzWIOAFV14x, hjflRzypLEQM}`, audit `AuditAvUnitsJob.zzZk9wq8vBfe` / `.{uslATgd598ZP (frozen feats CI), X0s6aXKhuRhb (avstates CI), ZiGphrMXI4Ri (prenet)}` |
| history-masking arms | `T/ReturnnTrainingJob.{PcM3MDZXiXC2 (p05), R62J4G37TXMV (p09), wlZcS8OuAuwE (p07), Rzs2Jf7sQmUN/l5oMBhbFCD1h/E5hXYIsNPzWj (blind_shift L1/L2/L4)}` |
| BEST-RQ era (dead) | AV frozen `.F1rZdbZUnP8e` / unfrozen `.fN1LzuJhqEPC` |
| commits | `b6944ab` (autocast fix), `343efff` / `97d39df` / `8849275` (TTE build), `e07d1c7` (bf16 head cast), `f3a1aa4` (`lam_len` wiring), `17d7156` (codebook probe), `df985ee` (sweep off) |

**Vocabulary layout**, checked because an earlier draft got it wrong: `resize_decoder_for_units`
(`sae/vocab.py:63`) grows the table 151936 -> 152438 with `mean_resizing=False` and those 502 rows are
**inert** — the real embeddings live in `install_sentinel_overlay` (`decoders/qwen.py:136`), a separate
trainable `nn.Embedding(502, 2048)` tied through one matrix on input and output. A dedicated K+2 head
therefore already exists; only the softmax *denominator* is unrestricted, and conclusion 18 says leave it
that way.

**GRPO memory is activation-bound by rollouts x transcript length** through the differentiable re-forward —
the OOM threshold is ~1600 rollout-tokens (~75 GB), so per-GPU rollouts must stay around 12 without fused
CE, and first-batch memory under `laplace` ordering *underestimates* the peak. External
`torch.utils.checkpoint` is unusable on `decode_seq` (it mutates `self.current_output`, so the recompute
takes a different path), and the AR re-forward must run with `grad_checkpoint=False` because checkpointing
over params shared with the AV through the PeftModel/overlay wrapper breaks the backward.
`_tie_frozen_base` points the AR decoder's frozen params at the AV's (309 tied, 6.88 -> 4.07 GB, AR forward
bit-identical) and asserts `n_tied > 0`. Truncating the encoder to `encoder_layer+1` = 16 blocks is
bit-identical because LV60 is `do_stable_layer_norm`, and the fused CE computes per-token logprobs in row
chunks with logits recomputed in backward, so peak is `[chunk, V]` not `[B*G*T, V]`.

**B0 arm (d) — the §1c GAN / §1d CTC-student phone decode — was WITHDRAWN before any GPU ran, on the
independence constraint** (planner, 2026-08-03): it is unsupervised in the *label* sense, but if the AR's
target is the GAN's phone decode then the reward becomes "how well does this text explain the GAN's
output", the loop's ceiling is the GAN (already a working ASR at 0.172 dev-other), and the SAE contribution
becomes unmeasurable rather than merely bounded. The correct replacement, if one is wanted, is a
lexicon-derived gold phone stream, eval-only under the same quarantine as (e).

Two standing traps this phase paid for. `partition_epoch` is capped by the **arrow file count**, not by
utterance arithmetic — tc100's Ogg dir has 4 files against 960 h's 84, so `partition_epoch=10` dies in 29 s.
And SLURM-id to arm mapping must be read from `usage.run.1` host x `squeue`, never inferred from launch
order.

Standing caveat on the replay lambda: the probe ran `max_seqs=1`, so both norms are **per-utterance** while
the loop averages `pg` over 8 utterances per optimizer step and the replay terms over the ~1 seed row
present; the correction is bounded between 1x and sqrt(8) and its direction depends on how correlated the
two gradient families are, so it is not applied and lambda is *defined* as per-utterance parity. Monitors on
the surviving arm behaved as designed (replay fired on 18.8 % of steps against a predicted 19.0 %;
`replay_ar_ce` 5.658 vs `ar_ce` 5.677 over 15,772 lines).

Unused and unresolved: the BEST-RQ-era AR on full tc100 (dev CE 3.7074) is **not** a
drop-in better scorer — it models a different unit stream (k-means on encoder layer 5, no PCA) than the loop
emits (k-means on AV hidden states, PCA-96), and two CEs sharing the ln 500 ceiling are not comparable when
they are CEs of different random variables.

## Verifier feedback

**2026-07-15 (build audit).** All 63 tests reproduced; GRPO math matches the plan (k1 KL estimator,
within-group normalization, tie -> 0, gradient-tested decoupling both directions). Findings: an
**undeclared deviation** — the plan's AR capacity ladder (wider LoRA -> full-FT AR -> small from-scratch AR)
was silently replaced by "escalate the base size", also the costliest escalation at RL-sampling time since
the AV samples G continuations while the AR runs once; the lambda-transfer caveat needs stating (loop
mechanics and diagnostics transfer, lambda *values* only as starting ranges); decisions #3 and #4(b) are
internally inconsistent about whether the K+2 rows are shared.

**2026-07-16 (round 2).** All round-1 findings folded in; graph build reproduced at exactly 107 jobs; the
overlay chain verified end to end (resize -> LoRA/freeze -> overlay created after the freeze so it stays
trainable, lm_head hook mirroring the weight tie). Finding: `sae_token_lm.py` is untested though the status
row reads as if covered, and `target_logprobs` feeds both the reward channel and the usage gate, so a silent
off-by-one there poisons every downstream number.

**2026-07-17 (masking).** The blind_window arms measured a different intervention than designed, and the
count-model corroboration is decisive: observed blind CE = control + 0.09/0.11 is the signature of a
lag-1-intact model where a true lags-1..2 block leaves count models at ~5.9, so the "routes around the
window via slightly-older units" explanation is refuted — lags >= 3 carry ~0.1-0.2 nats total, so the
recovery IS the leak. **The ill-posed spec was the planner's, not the implementer's.** The leak-free rebuild
was audited sound, with the new leak test (invariance to perturbing each blocked lag singly and jointly,
sensitivity to the newest visible lag) the decisive instrument; a bounded-upside check shows +10 epochs
could move the gate at most to ~0.1-0.15.

**2026-07-17 (rollout diversity).** One error: "advantages ~ 0 -> no gradient" is wrong —
`group_advantages` std-normalizes, so the live arms take O(1)-advantage steps in garbage-vs-garbage noise
directions, which makes pausing the comparison *more* urgent. Consistency check: the live `reward_recon`
band sits exactly at each scorer's own text-blind level.

**2026-07-19 (continuous scorer).** Code, jobs and config sound (geometry parity with `build_units` by
verbatim import, fit-split-only standardization, KeyError hard-fail attach, no_grad reward channel). Two
notes: the claimed CPU unit test is not in the repo (reproduced independently, all pass); and per-dim global
standardization is not per-utterance hygiene, so speaker/channel offsets stay in the target and the
candidate's win must come from de-quantization.

**2026-07-30 (gap audit).** Three conclusions failed verification and are recorded as corrections above: the
"reward magnitude" verdict is void because advantages are scale-free; "RUNs 1-2 FLAT" was never actually
measured (RUN 1 scancelled mid-epoch with an empty models dir, RUN 2's dir deleted, so zero WER reads exist
on any GRPO checkpoint, and the reward-mean drifts ~0.11 nats from `laplace` length-sort ordering against a
maximum possible policy effect of ~0.02); and `gap_true = -0.41` is the casing bug. Bookkeeping: what
trained as "candidate 3" is the raw-w2v2-feature CI regressor — the plan's unsupervised-arm variant (phone
posteriors) is untested and stays live.

**2026-08-01 (rung Stage-B arms).** The two 100 h loop arms launched from under-trained rung inits are in the
degenerate attractor an anchor-less loop predicts: `grpo_text_len` inflates monotonically until it pins at
exactly 100.0 = `max_gen_len` while within-group reward std collapses 0.17 -> 0.007 and 0.94 -> 0.006.
Meanwhile `ar_ce` "heals" toward the healthy control's level — **in-loop AR CE self-repair is real but
repairs the unit-LM term only; it cannot invent content-conditioned ranking from babble rollouts.**

**2026-08-03 (dedup).** Dedup is not the lever. True run-collapse on the loop's own `units.pkl` is
rho = 0.8598 (924,607 -> 794,931 units; 12.5 -> 10.7 units/s, about the 10.22/s gold phone rate), so dedup
buys a 14 % trim and bounds its effect at roughly +16 % of the 0.27 nats/unit; the stats file's `rho=1.0000`
was never a measurement (with `dedup=False`, post = pre by construction), and the x4 pooling that built this
stream already replaced symbol dedup as the rate compressor. Dilution is therefore realization detail in the
unit IDs plus intrinsic phone-vs-BPE granularity (~2.9 units/token even after collapse), which re-aims the
next phase at identity space.

**2026-08-05 (softmax denominator).** Both claims in the frame-accuracy log's original §5 were wrong as
stated — a scoring-time restriction cannot change any number in the results table (rank-preserving offset),
and `argmax_in_unit_range` does not bound the leaked mass in either direction. Folded into conclusion 18;
the follow-up measurement was run in response and reversed the recommendation.
