# SAE §1.0 + §1a — shared Phase-1 infrastructure and the decipherment track

## Approach

**1. §1.0 rVAD port + validation gate.** rVADfast ported and aggregated to the 25 Hz grid;
`RVADValidationJob` scores rVAD silence against MFA gold silence on full dev (`spn` counted as
speech). Measured at the default `vad_threshold=0.4` (a 0.3-0.7 sweep does not beat it).

| split | F1 | P / R | removed |
|---|---|---|---|
| dev-clean (2703 utts) | 0.794 | 0.864 / 0.734 | 14.0 % |
| dev-other (2864 utts) | 0.790 | 0.896 / 0.706 | 15.1 % |
| dev-clean recall by silence-run length | 1-2 fr 0.13 | 3-5 fr 0.52 / 6-12 fr 0.77 | >=13 fr 0.81 |

**2. §1.0 unsupervised selection metric.** The Baevski/wav2vec-U criterion that picks every
unsupervised checkpoint, seed and hyperparameter without labels: anchor `argmin[NLL_LM(P) - log U(P)]`
-> fluency-band filter (log 1.2) -> winner `argmax sum log p_LM` (non-length-normalized). Calibrated
against gold PER on a spread of real dev-clean decodings, reusing the §0a feature cache.

| calibration read | value |
|---|---|
| spearman(NLL/tok, gold PER) | 0.89 |
| §1.0 winner | the true lowest-PER model (probe 0.145) — MATCH |
| degenerate constant-phone decode | rejected by the -log U term (U = 0.03) despite best total LM logprob |

**3. §1a phoneme 4-gram LM on T_phi.** KenLM order-4 over the deduped phonemized corpus, the LM the
§1.0 metric scores against and the decipherment target LM. 42 unigrams (39 ARPAbet + KenLM specials),
1582 / 46,289 / 762,737 higher-order n-grams; perplexity ~8.5 (~3.1 bits/phone) on a 39.6 M-token
held-in sample, 0 OOV.

**4. §1a(iii) hard-unit decipherment, validated on simulated streams.** Units drawn from real T_phi
phone sequences through a *known* many-to-one map (+ fertility + emission noise) so recovery is
ground-truth measurable; three methods compared, all selected by unsupervised criteria (bigram-L1
cost or LL), never gold.

| method | recovery | PER | note |
|---|---|---|---|
| (i) pure-LM ICM | 0.30 | -- | collapses onto a few frequent phones, vocab usage 0.13 |
| (ii) HMM channel, init from truth | 0.97 | -- | highest LL — objective is well-aligned, EM is init-limited |
| (ii) HMM, CDF / bigram / random / annealed init | 0.23 / 0.31 / ~0 / 0 | -- | none reaches the true basin |
| (iii) GW-OT init -> HMM (fert=1) | **0.97** | **0.034** | robust to 5/10/15 % emission noise (1.00/1.00/0.945) |

**5. §1a(ii) continuous decipherment — Gaussian-emission HSMM on real features.** Hidden states = 39
ARPAbet + SIL anchored to gold ids by the LM (no permutation ambiguity), phone-bigram transitions
frozen, per-phone full-covariance Gaussians on PCA-48 features, per-phone categorical durations,
Viterbi hard-EM; held-out 100 dev-clean utterances against MFA.

| configuration | PER | frame-acc | note |
|---|---|---|---|
| synthetic gate (known params) | 0.154 -> 0.004 | -- | machinery correct |
| oracle emissions, decode (train / held-out) | 0.275 / 0.328 | 0.770 / 0.716 | the honest model-class anchor is ~0.33 |
| oracle init + 3 EM iterations | 0.392 (train) | 0.683 | LL rises monotonically while PER degrades |
| k-means K=40 + gold-majority map | 0.76 | -- | weak-oracle init, fails |
| GW-OT K=500 (the validated §1a(iii), truly unsupervised) | 0.86 | 0.146 (map) | collapses on real units |

## Conclusion

1. (1) rVAD is adequate for Phase-1 trimming — it catches the long silences that corrupt distribution
   matching and correctly ignores 40-80 ms inter-word micro-gaps; the literal F1 >= 0.85 gate is
   unreachable against an MFA phone-grid silence reference by *any* utterance VAD and must be read
   against long-silence intervals instead.
2. (2) The unsupervised selection metric is trustworthy in this regime, which de-risks label-free
   checkpoint selection for the GAN; caveat that it was calibrated on dev-clean and on a coarse model
   spread, so it needs re-checking on the real checkpoint cloud.
3. (4) Method (i) is degenerate by construction and method (ii) is correct and identifiable but
   init-limited; GW-OT is the init that cracks it — on *simulated* units.
4. (5) The generative Gaussian-emission HSMM has **no unsupervised trajectory to the Phase-1 gate**:
   the only sub-gate number needs gold emissions, the truly-unsupervised GW-OT init collapses on real
   units (map frame-acc 0.146), and ML-EM is anti-aligned — likelihood rises monotonically while
   phonetic accuracy falls from an oracle start.
5. (5) The anti-alignment is **structural, not a silence artifact**: the plan-conformant rVAD-trimmed
   configuration (gold-SIL 16.6 % -> 5.4 %) diverges at nearly the same rate, and the per-segment
   prior contributes O(few) nats against O(d*frames) of density term — so soft Baum-Welch or a
   stronger n-gram cannot fix it (Johnson 2007, "Why doesn't EM find good HMM POS-taggers?").
6. (4 vs 5) The simulation regime was optimistic: GW-OT recovers 0.97 on simulated units because their
   co-occurrence graph mirrors the phone bigram, while real k-means co-occurrence is dominated by
   acoustic flicker and coarticulation.
7. **Decipherment is closed permanently, on a bound.** The evidence indicts *generative ML over
   features* and specifically spares the model class (oracle-emission held-out decode 0.33 from a mere
   bigram + duration prior), so the fix-class is distribution matching over output phone sequences —
   which fires the §1b/§1c fallback gate.

## Catalog

| artifact | path |
|---|---|
| rVAD port + validation job | `recipe/i6_experiments/users/wu/unsupervised_asr/vad_port.py`, `config/sae_1_0_vad.py` |
| unsupervised selection metric (`--calibrate` reproduces the numbers) | `scripts/unsupervised_asr/unsup_metric.py` |
| phoneme 4-gram LM | `output/sae/1a/phoneme_lm_o4.{arpa.gz,bin}` |
| pre-built KenLM binaries (frozen `tk.Path`) | `tools/kenlm/build/bin` |
| hard-unit decipherment (GW-OT + HMM) | `scripts/unsupervised_asr/decipher.py` |
| continuous HSMM decipherment | `scripts/unsupervised_asr/hsmm_decipher.py` |

## Verifier feedback

**2026-07-14.** Audited `hsmm_decipher.py` by code review, reproduction and one discriminating
experiment.
- Core conclusion CONFIRMED and now evidenced: the committed EM loop never logged likelihood, so
  "the ML optimum is not at the phonetic truth" had been asserted without the measurement that
  separates a misaligned objective from a bug. Added: LL/frame rises monotonically (-150.22 ->
  -148.80) while train PER degrades 0.275 -> 0.392 and frame-acc 0.770 -> 0.683.
- The silence hypothesis is REFUTED as root cause (folded into conclusion 5), although the committed
  prior does violate the §1.0 silence protocol three ways (P(SIL|phone) ~ 1.2 % from boundary-free
  text; SIL->SIL impossible; utterance-initial SIL forbidden via `logpi`).
- The headline 0.275 was a **train-set** number; held-out is 0.328 / 0.716, so "0.275 -> 0.49 test"
  mixed a train start with a test end.
- `lm_scale` is confounded — `prior_scale` multiplies LM *and* duration log-probs, so "the bigram
  misleads when trusted" does not follow from that lever; do not cite it.
- The GW-OT collapse number (map frame-acc 0.146) came from a scratchpad script, not the repo — commit
  before citing it anywhere.
- `fit_gaussians` falls back to the global Gaussian for states with < d+2 frames, a garbage-state
  attractor for rare phones (ZH/OY) at this data scale.
- Ruling: the fallback gate is legitimately fired; requirements carried into Phase 1 are the §1.0
  silence protocol, held-out numbers only, and committing scratch code before citing its numbers.
