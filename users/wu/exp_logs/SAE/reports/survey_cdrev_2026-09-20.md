# Survey for a context-dependent reverse model — 2026-09-20

Read-only code survey (Explore agent) of `recipe/2025-10-speech-llm/src/speech_llm/` for making
`p_phi(x_segment | k)` into `p_phi(x_segment | k, h)`, h = previous phone. Paths relative to
`recipe/2025-10-speech-llm/src/speech_llm/` unless prefixed.

## Brief

1. x IS DISCRETE. `units` = enc50 codebook ids, K=500, one per 50 Hz frame (sae/emc/blankfree_train_jobs.py:158), `[B, S]` int64 (train_steps/sae_blankfree.py:39). Recognizer stride 3 -> `log_q [B, T=ceil(S/3), 40]` (:60). Band `|s - 3t| <= W=25`.
2. p_phi = SMALL NETWORK + DURATION TABLE, no acoustics, no context: `reverse.SegmentalReverseModel` (sae/emc/reverse.py:175), 369,604 params: `dur_logits [40,50]`, `emb_type [40,192]`, `emb_dur [2,192]`, `emb_pos [3,192]`, `eta_proj [192,16]`, `norm`, `lin1 [512,192]`, `lin2 [500,512]`. Emission `nu(k, dur bucket(2), pos bucket(3), eta) -> [B,40,2,3,500]` (:255). LR multiplier 30 (emc_train_jobs.py:278-279; blankfree_train_jobs.py:229-230). Trained ONLY through the lattice surrogate `-(1/tau) * (seg_post * seg_table).sum((1,2,3))` (lattice.py:1303-1305); `seg_post` is the detached manual forward-backward arc posterior. BT gives phi no gradient.
3. G = `lattice.build_segment_table` (lattice.py:419-432): `G[b, k, d-1, s]`, shape `[B, 40, 50, S+1]` fp32 (fp64 copy in the step, `lattice_float64=True`). d_min=2, D_k = 25 / 50 (SIL), d_cap=50, d_pad=53. Built by cumsum per (k, dur bucket, pos bucket) (reverse.py:284-296). Memory at 88,000 padded frames / B=128: G 0.66 GiB fp32 + 1.31 GiB fp64; `scaled_seg_pad [B,S+1+2W,53,40]` 1.49 GiB fp64; `seg_post_pad` 1.41 GiB fp64.
4. STATE = `(t, s, h, f)`, band offset `o = s - 3t + W in [0,50]`, f = repeat flag. `h = outer*41 + last`; blank-free arm runs `prior_history="trigram"` -> |h| = 1681 (definitions/sae_blankfree.py:244-245). Forward table `[B, T, 51, 1681, 2]`, checkpoint 32. G is HISTORY-FREE: `gwin = seg_pad[:, 3t:3t+51]` is `[B,51,53,40]` with no h axis (lattice.py:870, 876, 883, 1162). FB manual under `@torch.no_grad` (lattice.py:1023); arc posteriors at :1174-1179 / :1208-1214, accumulated into `seg_post_pad`, read back at :1255. Matmul path: 2-3 `_logmm` of `[B*K, G, O] x [B*K, O, O]` per frame, fp64.
5. KEY: at the trigram history `succ = arange(41)`, so `group(h) = last(h) = p_-1` (lattice.py:338, 376-383) and `src[b, o, g, k]` (lattice.py:572 elementwise / :732 matmul) is indexed by (previous phone g, emitted phone k) exactly where the segment table is applied; `o` fixes the segment start `s = 3t + o - W`. Hence:
   - FACTORISED `C[h,k,s]` (no d dependence) is almost free: `src += C_pad[:, 3t:3t+51]` (`[B,51,41,40]`) BEFORE the band GEMM; no new GEMM axis, state or band indexing. Memory `[B,S+1+2W,41,40]` = 0.54 GiB fp32 / 1.08 GiB fp64; a companion context posterior `[B,41,40,S+1]` comes from the same `mass` tensor before its group sum (lattice.py:1175 / :1205).
   - FULL `G[h,k,s,d]` is NOT affordable: 26.9 GiB fp32 / 53.8 GiB fp64, band matrix `[B*K*41,51,51]` fp64 = 4.07 GiB per frame.
   - The h axis must be 41 (40 phones + BOS = 40).
6. NO context knobs or dead code in phi (reverse.py:17-18, :182). All "context" hits are the PRIOR's `PriorHistory`.
7. Inventory 40 outputs, no blank (definitions/sae_blankfree.py:240-243; lattice.py:241-243). SIL = id 39, D_sil = 50, SIL-after-SIL forbidden under blankfree (lattice.py:347-350). `BOS_ID = 40` (prior.py:69), start history 1680 for the trigram, `fwd[:, W, hist.start, 0] = 0` (lattice.py:1002). No end-of-sentence term. phi sees no previous phone anywhere today; the first segment's context is BOS.

## Sites that assume G's shape or k-only indexing

Core: lattice.py:419-432 `build_segment_table`; :435-449 `_legality_mask`; :903-927 `scaled_seg_pad`; :870 / 1162 `gwin`; :876-884, :1168-1214 emit reductions; :735-745 `_band_matrix`; :1138 / :1176 / :1211 / :1255 `seg_post_pad`; :402-411 `LatticeOutput.seg_post`; :952 / 974 `forward_log_z` asserts; :1303-1305 surrogate; :1342+ `brute_force_log_z`.
Consumers: train_steps/sae_blankfree.py:68-106; train_steps/sae_emc.py:280-366 (CTC arm); sae/emc/blankfree.py:19-30 `zero_emission_segment_table`; sae/emc/rate_term.py:370-394 (`tilt_segment_table` pins axis 1 = K; `expected_nonsil_tokens` sums seg_post over (2,3)), :598-638 finite-difference passes; sae/emc/candidates.py:316-419 (off by default); sae/emc/path_training.py:10-14, :74; blankfree_train_jobs.py:519-581 (grad-norm profile job); w2vu2 port rev_term.py:38-56, 163-190 (bigram history, out of scope).
Reverse model used outside the lattice (would need the same conditioning): `forward_logsum(z, y, eta, ...)` reverse.py:299-350 (line 332 `seg[rows, k_i]` -> `seg[rows, y[:, i-1], k_i]`), used by fit/evaluate (:586-681), s1a_job.py, s3_jobs.py, emc_train_jobs.py:1471-1475, blankfree_eval_jobs.py:160-163, supervised_reverse_init.py, blankfree_seed_jobs.py:205-212; `sample(y, eta, ...)` reverse.py:367-465 for BT (bt_aux.py:458, bt_blankfree.py:280); oracles `segmentation_log_prob` (:523-553), `brute_force_log_likelihood` (:556-572).
Monitors: train_steps/sae_blankfree.py:139-146 (`reverse_per_frame` from `out.expected_reverse`, lattice.py:1179 / :1214 / :1256; `prior_per_token`; `expected_phone_rate_hz`; `rate_fd_check`); lattice.py:1311-1328 `assert_finite_lattice_stats`.
Tests pinning shapes / values: test_lattice.py:70, 161-176 (`seg_post == tau * dlogZ/dG`), 184-185, 214-216, 233, 247, 336-365 + 555-570 (20 BANKED BIGRAM DIGESTS incl. seg_post, must not move), 424-432, 573-600; test_blankfree_lattice.py:14-40 (oracle arc score `prior[h,k] + seg[k, d-1, s]` at :23 becomes `+ C[last(h), k, s]`), :55-75; test_blankfree_attrib.py:86, 112, 129-130; test_rate_term.py:123, 164-176, 259-295; test_reverse.py:147-148; test_candidates.py:188-223; test_bt_probe.py; train_steps/test_sae_emc.py:167-168, 905.

## Scoping notes (derived)

(a) At the trigram history the DP's group axis IS the previous phone; `_last_logsumexp` with one member per group is the identity (lattice.py:551-555).
(b) Factorised variant: `C` independent of d, added to `src` before the band contraction; context posterior from `mass` before the group sum.
(c) Full table infeasible at B=128 / 88,000 frames.
(d) The w2vu2 port runs the bigram history; out of scope.
(e) Contracts: the 20 banked bigram digests and the checkpoint bit-identity test must not move for existing arms; the added axis must be default-off and take the identical code path at width 1 (as `prior_weight = 0` / `n_outer = 1` do, lattice.py:539, 551, 763-767, 789-791).
(f) The batch-budget estimator (lattice.py:152-157, 290-292) knows nothing of |h| or a context axis; re-fit, do not extrapolate.
(g) Duration-bucket coupling: the existing emission of a segment's first frames depends on the duration bucket of d (2 buckets) and the position bucket, so a "first m frames scored by a context head" model is `G'[h,k,s,d] = (G[k,s,d] - S_old_first_m[k,s,d]) + S_ctx[h,k,s]`: the subtraction stays in the d-indexed table, the context score goes to `src`. m must not exceed d_min = 2 for the context score to be d-independent.
