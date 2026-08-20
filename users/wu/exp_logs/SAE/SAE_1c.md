# SAE §1c — wav2vec-U 2.0 GAN, the distribution-matching bootstrap

## Approach

**1. CPU scaffold (retired as a training path).** A minimal w2v-U-2.0-style pipeline — frozen features
-> linear generator -> softmax -> collapse consecutive argmax -> 1-D-conv discriminator against real
T_phi lines, with gradient penalty, smoothness and diversity terms. Its plumbing gate passes
(`--mode sup` reproduces the §0a linear probe exactly, PER 0.145) but GAN mode never converges on
CPU batch-1 (PER ~0.95-1.0), because the untrained generator's argmax flips every frame so the
collapsed fake is frame-length against ~3x shorter real text and the discriminator wins on length
alone.

**2. The fairseq reference trainer on frozen features.** fairseq 0.12.2 ships the complete w2v-U 2.0
source, so §1c runs it rather than reimplementing it, inheriting all documented paper-vs-code
divergences. Deviations from the reference, each with a reason: `input_dim` 1024 -> 512 (BEST-RQ is
512-d), `generator_stride` 3 -> 2 (output rate governs convergence; 25/2 = 12.5 Hz against our
measured ~11.9 Hz phone rate, and stride 1 = 25 Hz sits on the paper's divergent config),
`generator_kernel` 9 -> 5 (time-matched at 25 Hz), `target_downsample_rate` 2 -> 1, and the §1a
SIL-free 4-gram as the selection LM (fairseq strips `<SIL>` from hypotheses before scoring).

**3. p_sil measured rather than inherited.** The published `p_sil=0.5` is tuned to fairseq's rVAD
residue, so the comparable quantity — a **token** rate, because `segmentation.type: JOIN` applies
`logit_segment` before the discriminator — was measured on full dev (5567 utts, gold used only to
measure), and run as a 2-arm comparison instead of a silent override.

| side | residual SIL token rate |
|---|---|
| audio, dev-other / dev-clean after rVAD | 0.0544 / 0.0444 |
| text at p_sil 0.10 | 0.0522 (matches) |
| text at p_sil 0.25 | 0.0869 |
| text at p_sil 0.50 (plan) | 0.1357 (2.6x the audio) |

**4. BEST-RQ pilot: batch-norm {20,30,40} x p_sil {0.10,0.50}, 150k updates, full dev greedy PER.**
Anchors: chance ~0.90, §0a oracle-map k-means 0.63, §0a supervised linear probe 0.145, paper GAN-only
greedy dev-other 0.136.

| p_sil | bn | PER-clean | PER-other | best weighted_lm_ppl |
|---|---|---|---|---|
| 0.5 | **20** | **0.738** | **0.754** | **63.8** |
| 0.5 | 40 / 30 | 0.829 / 0.835 | 0.833 / 0.839 | 75.3 / 75.7 |
| 0.1 | 40 / 30 / 20 | 0.838 / 0.839 / 0.847 | 0.846 / 0.850 / 0.857 | 76.0 / 69.5 / 67.0 |

**5. Intermediate-checkpoint PER curves.** `W2vu2PerCurveJob` scores greedy PER at every ~10th
save-interval checkpoint and marks the `weighted_lm_ppl`-selected one, so within-run selection is
measurable rather than assumed; validated by reproducing the standalone eval job to every digit.
On the BEST-RQ arm every checkpoint across 150k updates sits at 0.82-0.92, with a selection gap of
+0.014 (ppl-best 0.833 @51k vs PER-min 0.819 @102k).

**6. wav2vec2-Large-lv60 L15 pilot, 5 seeds, p_sil 0.5.** Identical pipeline, LM, greedy eval and
audio; only the encoder changes. Selection-honest reporting: the reported system is the one the
`weighted_lm_ppl` criterion picks, never the best-PER seed.

| seed | weighted_lm_ppl | best_upd | dev-clean PER | dev-other PER |
|---|---|---|---|---|
| **s0** | **15.85** (min -> SELECTED) | 148000 | **0.173** | **0.214** |
| s1 | 16.24 | 77000 | 0.162 | 0.205 |
| s2 | 17.45 | 140000 | 0.175 | 0.215 |
| s3 | 17.76 | 66000 | 0.137 | 0.168 (oracle-best, not reportable) |
| s4 | 63.52 | 29000 | 0.851 | 0.862 (collapsed) |

## Conclusion

1. (1) The scaffold proved the generator capable and the eval path correct, and its failure mode —
   the discriminator winning on length statistics — is the same finickiness the published recipe
   handles with batching, exact penalties and unsupervised model selection; it is not a training path.
2. (3) p_sil 0.10 matches the measured audio SIL token rate, so the plan's 0.5 over-inserts silence
   ~2.6x. **WRONG as a recommendation** — PER contradicts it: 0.5 wins, decisively at bn20 (0.754 vs
   0.857); the marginal SIL-rate match was the wrong predictor and the plan's 0.5 stands.
3. (4) On frozen BEST-RQ features the GAN does not reach a useful phone recognizer: the best config is
   dev-other 0.754, a real signal over chance but *worse than the §0a oracle-map ceiling* and 5.5x the
   paper's 0.136.
4. (4) The §1.0 unsupervised metric selects correctly at **config** level — the lowest
   `weighted_lm_ppl` picks the best-PER config — even though within a single run the proxy falls 8x
   while PER stays pinned.
5. (5) The BEST-RQ failure is global, not a checkpoint artifact, and unsupervised checkpoint selection
   is essentially aligned (+0.014 of the trajectory minimum) — there is no hidden good checkpoint.
6. (6) **The encoder is the cause.** Same pipeline, LM, eval and audio; swapping BEST-RQ-L5 for
   wav2vec2-L15 moves dev-other 0.75 -> 0.21 selection-honest (0.168 oracle-best), with textbook
   convergence (0.904 -> 0.168 by 66k) instead of a flat 0.82-0.92 floor. This decided the encoder for
   the whole program. Mechanism (inferred, not isolated): BEST-RQ's random-projection-quantizer target
   needs only a linear readout to separate random bins, so nothing pressures it to form
   Euclidean-compact phone blobs, whereas wav2vec2 is trained on a quantized codebook.
7. (6) **The selector is now the bottleneck, not the model.** The four converged seeds' ppl
   (15.85/16.24/17.45/17.76) ranks them anti-correlated to PER (0.214/0.205/0.215/0.168), so
   `weighted_lm_ppl` cannot resolve the ~0.05 PER spread among good seeds — a +0.046
   selection-honesty gap, and the same objective-anti-alignment theme as decipherment.
8. (6) A mid-training write-up headlined s3's 0.168 as "the result" — **WRONG**: that is the
   oracle-best seed and is not reportable; the honest number is s0's 0.214 dev-other / 0.173 dev-clean.
9. **Metric discipline**: 0.214 is a phone error rate against the paper's Table-1 GAN-only greedy
   dev-other PER 13.6, not against its ~5 % WER headline, which adds a 4-gram WFST decode and up to
   three self-training stages.

## Catalog

`W/` = `work/i6_experiments/users/wu/experiments/unsupervised_asr/w2vu2/`.

| artifact | path |
|---|---|
| CPU scaffold (retired) | `scripts/unsupervised_asr/continuous_gan.py` |
| selected wav2vec2 GAN, seed 0 (the §1d teacher) | `W/gan/FairseqW2vu2TrainJob.HOb2GgtYT7Bc` — anchored by `W/eval/W2vu2PerEvalJob.ptwMk3TuPPYb` (0.173 / 0.214) |
| BEST-RQ pilot arms | `W/gan/FairseqW2vu2TrainJob.{Oc0pM2YaHs65,j74MEVIWFmxB,5IPlLRrakdij,...}` |
| p_sil measurement | `W/silence_stats/ResidualSilenceStatsJob.{WNWI1jK22xey,EhXVdhGZK0vP}` |
| per-checkpoint PER curves | `W/eval/W2vu2PerCurveJob.*` |
| final per-seed evals | `W/eval/W2vu2PerEvalJob.*` |
| w2v2 repr audit (the GPU-light discriminator) | `W/eval/W2v2ReprAuditJob.BFLCTjxcvrNE` |
| dedicated conda env (rebuild script) | `w2vu2/build_w2vu_env.sh` |

## Verifier feedback

**2026-07-18 (3-agent audit + ground truth).**
- The right paper anchor is **0.136**, not ~0.05: the memorable ~5 % is Table 3 WER with 4-gram WFST
  decoding and 3-stage self-training. Their "greedy" (merge consecutive same-argmax frames) is exactly
  our all-zero-transition viterbi, so 0.754 vs 0.136 is apples-to-apples and essentially none of the
  gap is a measurement artifact.
- The "missing PCA" worry is a v1-vs-v2 confusion: `prepare_audio_v2.sh` has no PCA / mean-pool /
  faiss step (grep-confirmed) and `wav2vec_u.py:460` sets `pca_A = pca_b = None`. The only input
  normalization is the generator BatchNorm. **No missing or wrong preprocessing step.**
- BEST-RQ causation was inferred, not isolated: the GAN ran on layer 5 while the 0.145 probe anchor is
  layer 6; the paper uses a deep layer (L15) against our shallow L5; the winning bn=20 sits at the
  swept edge; LS-100 only, 6 configs.
- The cleanest discriminator is the identical pipeline on wav2vec2-L15 features — run and decisive
  (conclusion 6).
- 2026-08-17 (planner, Catalog audit during the 1f entry-5 grounding sweep): the Catalog
  row "selected wav2vec2 GAN, seed 0 (the 1d teacher)" cites FairseqW2vu2TrainJob.5IPlLRrakdij,
  whose own alias is sae/1c/gan_l5_sil0.5/bn40 — a BEST-RQ layer-5 arm, not the wav2vec2 arm.
  The wav2vec2-L15 seed-0 run behind the reported 0.214 dev-other PER and the 1d teacher is
  FairseqW2vu2TrainJob.HOb2GgtYT7Bc (alias sae/1c/w2v2_lv60_l15/gan_l15_sil0.5/s0; text side
  FairseqPreprocessTextJob.bi2B89fES77z = the p_sil-0.5 binarization). Labels-from-the-
  artifact's-own-field rule again; sent to the implementer for the Catalog fix — numbers and
  conclusions untouched.
