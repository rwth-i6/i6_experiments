# RESULTS

_Status snapshot: 2026-08-04 (§§1–5 unchanged from 2026-07-02; §§6–9 and the conclusions add the new
adversarial-alignment, text-upsampling, perplexity, and silence-in-input results). Metric everywhere is on LibriSpeech **dev-other**, scored with sclite.
All targets are **phonemes** (~41 units), so every "WER" below is really a **phoneme error rate (PER)**.
"recog" = the actual task (audio clusters → phonemes). "recon_*" = same-modality reconstruction
(denoising) probes. "a-t cosine" = mean pairwise cosine similarity between the audio and text
encoder states of the same utterance (shared-space alignment), from the encoder-PCA analysis._

Goal (reminder): **unsupervised** speech recognition — no paired (audio, text) supervision — by
making a shared encoder map corresponding audio and text to **similar representations** in a shared
embedding space, so a decoder trained on one modality transfers to the other.

---

## 1. Supervised topline (sanity ceiling)

Supervised AED on aligned (cluster, phoneme) pairs — establishes that the units + model are fine.

| experiment | enc/dec | notes | PER ep1000 |
|---|---|---|---|
| `config_960_v1/baseline` | 3/3 | with silence | 16.40 |
| `config_960_wo_sil/baseline` | 3/3 | no silence | 16.23 |
| `…/baseline_enc-6_dec-6_bs-10000` | 6/6 | | 15.38 |
| `…/baseline_enc-6_dec-6_bs-10k_ls-0.1` | 6/6 | label smoothing 0.1 | **15.10** |
| `…/baseline_enc-6_dec-6_bs-10k_ls-0.2` | 6/6 | label smoothing 0.2 | 15.14 |

**Takeaway:** cluster→phoneme is solidly learnable (~15–16 PER); deeper encoder + label smoothing
helps a little. Model capacity, decoder, and the discrete units are **not** the bottleneck.

---

## 2. Unsupervised ASR — the real goal (audio → text) — **FAILS**

Shared denoising AE (masked-denoising per modality, shared encoder + shared decoder), then decode
audio with the text decoder.

| experiment | PER 250 | 500 | 750 | 1000 |
|---|---|---|---|---|
| `baseline` (3/3) | 173.4 | 173.1 | 155.9 | **144.7** |
| `baseline_codebook` | 85.8 | 148.0 | 145.3 | **138.8** |
| `baseline_audio-and-text-mask-p-0.1` | 133.5 | 112.7 | 101.9 | **114.3** |
| `baseline_audio-and-text-mask-p-0.2` | 139.9 | 139.3 | 100.5 | **115.9** |
| `baseline_audio-and-text-mask-p-0.3` | 142.5 | 141.7 | 137.6 | **147.9** |
| `baseline_codebook_audio-and-text-mask-p-0.1` | 100.6 | 136.9 | 104.2 | **105.9** |

**PER > 100 everywhere** (worse than emitting nothing). The cross-modal path never works, at any
epoch, for any variant. This still holds across every intervention tried since — the adversarial
GAN (§6), silence-in-input (§9), and all but one text-upsampling run — the sole exception being
`text-upsample-1-3` at **97** PER (§7), the first (barely) sub-100 recog, but still a failed system.

### Why it fails (diagnosed from the hypotheses)
The audio→text output is **fluent, grammatical-looking phoneme text that ignores the audio and
loops**, e.g. (baseline, ep1000):

```
B AH T SH IY HH AE D B IH N S OW F AA R F R AH M DH EH M S EH L V Z
F AO R DH EH M S EH L V Z  F AO R DH EH M S EH L V Z  F AO R DH EH M S EH L V Z …
```

i.e. the **shared decoder learned to be a good phoneme language model** (from the text-denoising
task) but the **cross-attention to the audio-encoder states carries no usable signal**, so it
hallucinates + repeats. This is decoder-ignores-encoder / posterior-collapse on the cross-modal
path, and it is a direct consequence of §4 (the two modalities are not aligned).

---

## 3. Reconstruction / denoising probes (same-modality)

Each modality is asked to reconstruct its own (masked) input. This works — the shared model is a
competent per-modality autoencoder/denoiser.

**Comparable set — text→text at the base masking (mask_prob 0.3, span 2–10):**

| experiment | text-recon PER |
|---|---|
| `baseline_text-only` (single-task) | **37.12** |
| `baseline_codebook` | 38.45 |
| `baseline_audio-mask-p-0.4` (text mask still 0.3/2–10) | 42.05 |
| `baseline` (multi-task) | 44.58 |
| `baseline_enc-6_dec-6` (6/6 layers) | 51.17 |

- **Single-task text denoising (37.1) beats the multi-task baseline (44.6)** at identical masking →
  the joint text+audio objective **costs ~7.5 PER abs** on text denoising (the multi-task hurts, as
  suspected).
- The deeper 6/6 model is **worse** here (51.2) — likely under-trained / unstable (it needed
  `stop_on_nonfinite_train_score=False` and hit NaNs early).

**Not comparable (different masking) — reported for completeness:**

| experiment (eval masking) | recon PER |
|---|---|
| `audio-and-text-mask-p-0.1` text (0.1/span1) | 15.89 |
| `audio-and-text-mask-p-0.2` text (0.2/span1) | 27.61 |
| `text-mask-p-0.4` text (0.4/2–10) | 49.66 |
| `baseline` **audio→audio** (0.3/span4–20) | 38.17 |

Lower mask-prob / single-token spans are easier, so these small numbers reflect **task difficulty,
not model quality** — which is exactly why a fixed-masking sweep was added (see §5).

**Takeaway:** within a modality the model reconstructs fine (PER ≈ 37–45 at 30 % span masking, ≈ 16
at 10 %). Denoising is not broken. The gap between "can denoise text" and "cannot do ASR" is
entirely the **cross-modal transfer**.

---

## 4. Shared-space alignment (the crux) — **audio and text are NOT aligned**

Mean cosine similarity of encoder states within/between modalities (higher a-t = better alignment):

| experiment (ep1000) | a–a (audio) | t–t (text) | **a–t (cross)** |
|---|---|---|---|
| `baseline` | 0.250 | 0.267 | **0.155** |
| `baseline_audio-and-text-mask-p-0.1` | 0.193 | 0.223 | **0.126** |
| `baseline_codebook` | 0.078 | 0.084 | **0.024** |

Cross-modal similarity is **consistently below** within-modal similarity — the two modalities live
in different regions of the "shared" space. Over training it never crosses over and even
**degrades** late:

| baseline, a–t cosine | ep250 | ep500 | ep750 | ep1000 |
|---|---|---|---|---|
| | 0.130 | 0.200 | 0.209 | **0.155** |

| codebook, a–t cosine | ep250 | ep500 | ep750 | ep1000 |
|---|---|---|---|---|
| | 0.189 | 0.038 | 0.014 | **0.024** |

**Takeaways:**
- **Weight sharing alone does not create a shared space.** Nothing in masked per-modality denoising
  pushes *corresponding* audio and text to the same point, so the encoder settles into two
  weight-tied-but-separate sub-spaces.
- The **codebook makes it worse** — it collapses representations (a-a/t-t → ~0.08) and drives a-t
  toward 0. As currently used (a quantizer + a diversity loss, with no term tying the two modalities
  to the *same* codes) it hurts.
- Alignment **peaks mid-training then declines** — consistent with the model increasingly
  specializing each modality's reconstruction as the LR decays.

---

## 5. Fixed-masking text-recon sweep (copy ceiling)

To characterize the denoiser properly and compare single- vs multi-task **fairly**, a fixed-masking
text-recon sweep (span 2–10, mask_prob ∈ {0.0, 0.1, 0.5}; 0.3 already existed) was added to both
`baseline` and `baseline_text-only`:

- `…/{baseline,baseline_text-only}/recon_text_mask-{0.0,0.1,0.5}/1000/dev-other`

`mask_prob=0.0` = **copy ceiling** (pure pass-through of the enc+dec): if this is not near-zero, the
shared encoder/decoder itself loses information (a prerequisite problem to fix before cross-modal).
The 0.0/0.1/0.3/0.5 points give a **denoising-difficulty curve** per model.

| model | mask 0.0 (copy) | 0.1 | 0.3 | 0.5 |
|---|---|---|---|---|
| `baseline_text-only` | 37.59 | 36.84 | 37.12 | 41.16 |
| `baseline` (multi-task) | 53.03 | 46.35 | 44.58 | 46.47 |

**This is the most important result in the whole report, and it is a red flag:**

- The **copy ceiling is terrible** — with **zero** masking the model still gets **37.6 PER**
  (text-only) / **53.0 PER** (multi-task). A working text autoencoder should copy a clean phoneme
  sequence at ≈0 PER. It cannot.
- The curve is **almost flat** in the mask level (text-only 36.8→41.2 as masking goes 10 %→50 %).
  If the decoder were using the encoder, copy (0.0) would be ≈perfect and heavy masking would hurt
  a lot. Instead the output is **largely independent of how much input is actually present.**
- Interpretation: the autoregressive decoder has **collapsed toward an unconditional phoneme LM**
  and only weakly attends to the encoder — **even within a single modality.** The ~37–53 PER is
  essentially the decoder's LM guessing, lightly nudged by the encoder.
  (Caveat: `mask_prob=0.0` is slightly out-of-distribution — the model always saw ≥1 mask token in
  training — but 0.1/0.3/0.5 are in-distribution and equally flat, so the conclusion holds.)

This is the same disease as the cross-modal hallucination in §2, seen in its milder same-modality
form. **The decoder does not rely on the encoder.** Cross-modal ASR cannot possibly work until the
decoder is forced to depend on the encoder representation — which is exactly the "make same-modality
reconstruction actually work first" prerequisite.

---

## 6. Adversarial alignment (GAN) — aligns the space to a-t cosine ≈ 0.99, **but ASR still fails**

The §4 diagnosis said the modalities are not aligned, so a domain-adversarial discriminator was
wired on the shared encoder output (config `…_wo_sil_gan_v1`, `adv_loss_scale=0.1`, all at
mask_prob 0.1 / span 1). It works **spectacularly on the alignment metric** — and that turns out to
be the story's twist.

| variant (ep1000) | disc | **a–t cosine** | (a–a / t–t) | recog PER ep1000 (min over ckpts) | copy PER (mask 0.0) |
|---|---|---|---|---|---|
| `baseline` (§4, no GAN) | — | 0.155 | 0.250 / 0.267 | 144.7 | 53.0 |
| `baseline_gan-adv-0.1` | mlp 1-gram | **0.986** | 0.990 / 0.988 | 128.6 (110.6) | 21.3 |
| `…_disc-mlp_2gram` | mlp 2-gram | **0.994** | 0.995 / 0.995 | 179.5 (159.6) | 27.5 |
| `…_disc-mlp_3gram` | mlp 3-gram | **0.992** | 0.994 / 0.993 | 196.7 (140.6) | 29.1 |
| `…_disc-mlp_4gram` | mlp 4-gram | **0.991** | 0.993 / 0.992 | 151.0 (113.8) | 24.5 |
| `…_disc-lstm` | biLSTM | **0.948** | 0.957 / 0.956 | 186.2 (111.6) | 27.3 |

**Two things happened, and the second kills it:**

1. **The adversary made same-modality reconstruction much better.** The copy ceiling (mask 0.0)
   dropped from **53 → 21–29 PER**, and the denoising curve became **properly monotonic** in the
   mask level (e.g. 1-gram: 21.3 → 23.5 → 40.4 as masking goes 0.0 → 0.1 → 0.5). Compare the
   baseline's flat/inverted curve (§5). So the §5 "decoder ignores encoder" problem is **largely
   fixed** here — within a modality the decoder now depends on the encoder.
2. **…yet audio→text recog is still PER > 100.** The adversary drives a-t cosine to **0.95–0.99**
   (a–a and t–t rise to the *same* 0.99), but this is **not** representational collapse — the states
   are *not* squashed to a point. Text→text recon actually *improves* (point 1), and the
   mean-centered PCA clouds are clearly spread and structured (in fact more spread than the
   baseline's). What the adversary inflates instead is a large **shared common-mode component**: one
   dominant direction that (nearly) all frames of both modalities share. Cosine is blind to offset
   and scale, so such a component pins mean cosine near 1 regardless of the rest — if the
   frame-specific residuals are roughly uncorrelated, mean cosine `c` implies the shared component
   carries ≈ `c/(1−c)` × the residual's energy (so `c=0.99` ⇒ ~100×, `c=0.95` ⇒ ~19×). The phonetic
   information lives entirely in that small residual: enough for the decoder to reconstruct *within*
   a modality, but in the residual space audio and text are **not** phonetically aligned, so
   cross-modal transfer gains nothing. Perplexity confirms it: the GAN model's conditional
   audio→phoneme PPL is **19.2**, *worse* than the plain LM's 4.96 and worse than the non-GAN unsup
   baseline's 7.74 (§8). (Sanity check on the saved PCA: raw a-t cosine 0.155→0.95–0.99 GAN, yet the
   mean-centered per-seq modality gap is still ≈0.7–1.5 spread-units — the modalities remain
   separated in the *informative* subspace despite cosine ≈ 1.)

**Takeaway — the single most important update to the report:** **mean cosine similarity is a broken
alignment metric.** A discriminator can max it out (0.15 → 0.99) while making ASR *worse* — not by
collapsing the space but by inflating a shared common-mode direction that cosine over-weights and
that carries *no* cross-modal correspondence information. §4's low cosine was a real symptom, but
simply maximizing cosine is not the fix. What is needed is *phonetically structured* alignment
(same-phoneme audio and text nearby, different phonemes apart) in the mean-removed, informative
subspace — precisely the analysis §12.1/§12.2 proposes and which averaged raw cosine cannot see. The GAN runs are also **unstable**: recog usually peaks around
ep250 and degrades, while the a-t cosine keeps climbing (i.e. the shared common-mode component keeps
growing) over training (lstm: 0.92 at ep500 → 0.95 at ep1000; the mlp variants climb from ~0.8 at
ep250 to ~0.99).

---

## 7. Text upsampling — best copy ceiling so far, and the only sub-100 recog

Duplicating each text token `[min,max]`× at the encoder input (`text_expansion_opts`) forces
many-to-one cross-attention (simulating the audio>text length ratio). Both at mask_prob 0.1 / span 1,
no GAN.

| variant (ep1000) | recog ep1000 (min) | copy PER (0.0) | 0.1 | 0.5 | a–t cosine |
|---|---|---|---|---|---|
| `baseline` (§5) | 144.7 | 53.0 | 46.4 | 46.5 | 0.155 |
| `baseline_text-upsample-1-2_bs-12000` | 144.2 (137.0) | 21.2 | 21.4 | 38.6 | 0.004 |
| `baseline_text-upsample-1-3_bs-10000` | **97.2 (96.6)** | **13.4** | 16.0 | 37.3 | 0.012 |

- **`text-upsample-1-3` is the best unsupervised recog in the whole project (≈97 PER)** — the first
  and only run to dip below the "worse-than-empty" 100 line — and has the **best copy ceiling
  anywhere (13.4)** with a clean denoising curve. Forcing the decoder to spread attention over a
  longer encoder input evidently makes it *use* that input.
- Crucially it achieves this with **a–t cosine ≈ 0** (0.012), the opposite regime from the GAN
  (§6). So low averaged cosine clearly does **not** preclude the best cross-modal result — more
  evidence that averaged cosine is not the quantity that matters.
- Still, 97 PER is a failed recognizer. Upsampling is the most promising single lever so far but not
  a solution; pure duplication is a partly trivial (dedup) task — adding per-copy noise/substitution
  (as noted in the config) is the obvious next step.
- Combining upsampling **with** the GAN was worse: `gan…text-upsample-1-2` degenerated (recon PER
  stuck at **83** at every mask level — a dead model), and `gan…text-upsample-1-3` gave recon 35 /
  recog 174. The two levers do not compose as wired.

---

## 8. Perplexity — a three-number proof that the encoder is (un)used

A perplexity forward job scores the frozen decoder as a language model (`phoneme_lm` config) and,
for the AED models, as a *conditional* LM (run the encoder, teacher-force the phoneme decoder, PPL
over the phoneme vocab only). All on dev-other, **wo-silence** reference (169,670 tokens).

| model | what it measures | perplexity ep1000 |
|---|---|---|
| `phoneme_lm/baseline` | unconditional phoneme LM (topline) | **4.96** |
| `sup…_wo_sil/baseline` | supervised audio→phoneme *conditional* PPL | **1.49** |
| `unsup…_wo_sil/baseline` | unsup audio→phoneme *conditional* PPL | **7.74** |
| `unsup…_gan…_disc-lstm` | GAN unsup audio→phoneme *conditional* PPL | **19.23** |

This is the §2/§5 story stated in four numbers:
- The **supervised** decoder *uses* the audio: conditional PPL **1.49 ≪ 4.96** (the LM's own
  guess). Conditioning on audio removes almost all uncertainty.
- The **unsupervised** decoder is *hurt* by the audio: conditional PPL **7.74 > 4.96**. It would be
  better off ignoring the encoder entirely — the audio states are effectively noise to it. This is
  the cleanest single confirmation of the cross-modal failure in §2.
- The **GAN** makes it *worse still* (**19.23**) — consistent with §6: the adversary spends encoder
  capacity on a shared common-mode direction, degrading even the little cross-modal signal the plain
  unsup encoder carried.
- (The phoneme LM itself trains fine: PPL 6.11 → 5.31 → 5.02 → 4.96 over ep250→1000.)

---

## 9. Silence in the input (`…_w_sil_in_input_v1`) — exploratory, recog not yet scored

A variant that keeps silence in the audio-cluster input (`max_num_sil` ∈ {3,5,7}), to test whether
shared silence frames help the encoder align. Completed so far: encoder-PCA cosine and perplexity;
**recog/recon scoring is still running** (search outputs exist, no WER yet).

- **Silence raises the (averaged) a-t cosine of the plain baseline**: `w_sil baseline` a-t = **0.28**
  (a–a 0.36 / t–t 0.40) vs the wo-sil baseline's 0.155 — the shared silence regions are an easy thing
  for both modalities to agree on. The GAN sil variants again inflate cosine to ≈0.98 via the shared
  common-mode component (same caveat as §6).
- **Perplexity is very high** (`w_sil baseline` 1991; `gan…sil-3` 2426; `sil-5` 6189) — but these are
  scored on a **with-silence** reference (≈197–209k tokens incl. `<SIL>`), not comparable to the
  wo-sil PPLs in §8; the wo-sil models assign ~0 prob to `<SIL>` so CE blows up (the known §"Phoneme
  LM" silence caveat). Treat these as internal-only until the recog is scored.

---

## 10. Conclusions — current status

1. **The supervised path works (~15 PER); the unsupervised path is still broken (PER > 100)** — the
   sole exception a single `text-upsample-1-3` run at **97** (§7), i.e. barely sub-100 but still a
   failed recognizer. The blocker is not capacity, units, or the decoder architecture per se.
2. **Prerequisite (a) — "make the decoder depend on the encoder" — is now largely solvable.** §5
   diagnosed a posterior-collapsed decoder (copy ceiling 37.6 / 53.0 PER, flat in the mask level).
   Three independent interventions **fix the copy ceiling and restore a proper denoising curve**:
   the adversarial GAN (53 → 21–29, §6), text upsampling (53 → **13.4**, §7), and codebook+masking
   (53 → **14.7**, §2/§3 variant). So same-modality encoder-dependence is no longer the wall.
3. **Prerequisite (b) — align the two modalities — is the real, still-open problem, and averaged
   cosine turned out to be a trap.** The GAN drives a-t cosine 0.155 → **0.99** yet recog stays
   PER > 100 and conditional PPL gets *worse* (7.74 → 19.2, §8), because it "aligns" by inflating a
   **shared common-mode direction** (a–a and t–t rise to the same 0.99), *not* by collapsing the
   states (recon still works) and *not* by making the informative residuals correspond. Meanwhile
   the *best* recog (§7) has a-t cosine ≈ **0**. **Mean cosine similarity neither guarantees nor is
   required for cross-modal transfer** — it is a broken objective and a broken metric.
4. So the reframed prerequisite is **(b′) phonetically-structured alignment**: same-phoneme audio
   and text nearby, *different* phonemes apart — which averaged raw cosine (and a discriminator that
   games it via a shared common-mode direction) cannot express. Building the metric for this (§12.1/§12.2: phoneme-colored joint
   embedding, cross-modal retrieval@k) is now the highest-value next step, because we currently
   **cannot even measure** whether an intervention aligns the space in the way that matters.
5. **Perplexity gives a clean scalar for "does the model use the audio":** supervised conditional
   PPL 1.49 ≪ LM 4.96 (uses it); unsup 7.74 > 4.96 (audio hurts); GAN 19.2 (the common-mode
   inflation hurts more).
6. **The codebook alone still hurts** (§4: collapses representations), but **codebook + masking**
   gives the best copy ceiling (14.7) — as an encoder-dependence lever it helps; as an *alignment*
   lever it does not (a-t cosine 0.08). **Multi-task still modestly hurts each task** vs single-task
   (§3/§5).

---

## 11. Recommended next experiments (priority order)

**P0 — Build a *structured* alignment metric before running more alignment experiments.** This is
now the top priority: §6 showed we can drive averaged a-t cosine to 0.99 and make ASR *worse*, so we
are currently flying blind on the one quantity that matters. Implement §12.1 (phoneme-colored joint
embedding via the paired `MetaDataset`, analysis-only) and §12.2 (**cross-modal retrieval@k**: for
each text token, is the nearest audio frame the corresponding phoneme?) as a first-class per-checkpoint
scalar. Sanity check it against the known cases: it must score the common-mode-inflated GAN (§6)
*low* despite cosine 0.99, and rank the supervised model high. Without this we cannot tell a real
alignment from a cosine-gaming artifact.

**P1 — Alignment mechanisms, now judged by the P0 metric (not cosine).** Encoder-dependence is
largely handled (§6/§7 fixed the copy ceiling), so the effort moves entirely to making the two
modalities' *informative* (mean-removed) residuals correspond, without the common-mode shortcut:
- **Stop the adversary from gaming a shared direction** before adding more discriminators: the GAN
  "aligns" by inflating a common-mode component (§6). Mean-center / whiten the encoder states before
  the discriminator (so a shared offset can't fool it), and/or add a VICReg-style
  variance–covariance term to keep the residual dimensions used and decorrelated; or cap
  `adv_loss_scale` / stop early (recog peaks ~ep250). Track retrieval@k in the mean-removed space,
  not raw cosine.
- **Push text upsampling further** — it gave the best recog (97) and copy ceiling (13.4, §7) with
  *no* cosine alignment, so it is attacking a real axis. Add per-copy noise/substitution (pure
  duplication is a partly trivial dedup task), sweep the ratio, and try it **without** the GAN
  (the two did not compose, §7).
- **Backtranslation / cycle-consistency** (`aed_denoising_discrete_shared_backtranslation.py` exists
  but is **not wired into `main()`**): audio→text→audio and text→audio→text — the standard engine of
  unsupervised translation, and the one method that directly optimizes cross-modal *correspondence*
  rather than distribution overlap. Highest-upside once the P0 metric exists to judge it.

**P1b — Encoder-dependence levers (now secondary, but cheap):** the **CTC aux head** on the encoder
(exists, off) is still worth turning on to further force the encoder to be label-predictive; combine
with the codebook+masking recipe that already gives copy PER 14.7.

**P2 — Ablations to stop wasting compute:**
- **Redesign the codebook.** Alone it hurts alignment (§4), but **codebook + masking** gives the
  best copy ceiling (14.7, §2) — so keep it as an encoder-dependence lever, and add a loss mapping
  *corresponding* audio and text to the *same* codes (not just a diversity loss) to make it also
  serve alignment.
- **`share_decoder` on/off** and **length mismatch**: collapsed audio-cluster sequences vs phoneme
  sequences differ in length/rate; cross-attention alignment may be hard — quantify the length-ratio
  distribution and consider rate matching.
- Revisit the **LR schedule** — alignment peaks ~ep750 then decays; a longer high-LR phase or early
  stopping on a-t cosine may help.
- **6/6-layer model** is currently worse and unstable (NaNs); de-prioritize until P0/P1 land.

---

## 12. Analyses / visualizations that would help understand the models

The current encoder-PCA + a-t cosine analysis is already the most informative artifact — it is what
revealed the misalignment. Extensions, roughly by value:

1. **Joint embedding colored by phoneme identity** (not just by modality). Using the paired
   `MetaDataset` alignment *for analysis only*, project audio frames and text tokens together and
   color by phoneme. The key question: do same-phoneme audio and text land in the same cluster? The
   current plots only color by modality, which can't show phonetic organization.
2. **Cross-modal retrieval metric (precision@k):** for each text token, is the nearest audio frame
   (by the encoder states) the corresponding one? A single scalar per checkpoint that directly
   measures "shared space" quality and can be tracked over training / across experiments.
3. **Distribution-overlap / discriminator-accuracy metric** between the audio and text encoder-state
   clouds (e.g. a quick linear probe or MMD): quantifies §4 beyond averaged cosine, and doubles as
   the training signal if adversarial alignment is used.
4. **Decoder cross-attention maps** on the failing audio→text recog: confirm quantitatively that the
   decoder's cross-attention is diffuse/ignored (explaining the hallucination), and use it to check
   whether an alignment method actually makes the decoder *attend* to the audio.
5. **Per-position recon PER (masked vs unmasked):** decompose reconstruction into "copy" vs
   "in-fill" to see whether the denoiser truly reconstructs masked spans or mostly copies context.
6. **Track a-t cosine (and retrieval@k) across epochs for every experiment**, as a first-class
   curve — it is a better early indicator of unsupervised-ASR viability than reconstruction PER.

---

# Part II — cluster→phoneme map (cheat-seg) plan: diagnostics, reweighting, unsegmented port

_Started 2026-09-11. Companion to `OBJECTIVES.md` / `OBJECTIVE_VARIANTS.md` (notation) and the plan
in the session. Numbers here are frame accuracy of the **hardened** (argmax) 512→40 table on the
cheat-seg data (ceiling 0.740 = 26.0% PER, unigram chance 0.103), unless stated otherwise. Runs live
under `/work/asr4/schmitt/experiments/2026_04_09_unsupervised_asr/plan_runs/`._

## II.0 Code ↔ notation map (`calc_soft_map_search.py`, `calc_cheat_seg_identifiability.py`)

| notation (plan) | code | notes |
|---|---|---|
| `M[x,c] = q(c|x)`, soft | `M = torch.softmax(logits / tau, dim=1)` in `optimize` | `logits` = `theta`, `tau` geomspace 1.0→0.02 over `--steps` |
| `M`, hard | `SoftMapLoss.hard(assign, m)` builds one-hot from `assign = M.argmax(1)` | all reported acc/loss are of this |
| `p_A` (`P_A[1]`) | `c1` from `unigram_bigram(audio seqs)` | normalized over all tokens |
| `P_A[2]` | `C2` (dense `[512,512]`) | pairs **within** utterances only (`s[:-1], s[1:]`), normalized over all pairs |
| `P_A[3]` | `C3 = trigram_sparse(...)` → `torch.sparse_coo_tensor [n*n, n]` | distinct observed triples with weights summing to 1 |
| `P_A[4]` | `C4 = fourgram_sparse(...)` (prefix-grouped sparse) | idem, 4-grams |
| `p[1..4]` | `t1, t2, t3, t4` (dense) | **each order normalized independently** over its own token count |
| `q[1] = M^T p_A`, `q[2] = M^T P_A[2] M` | `SoftMapLoss.induced` | |
| `q[3]`, `q[4]` (+ back-off) | `induced_trigram`, `induced_fourgram` | `q~3 = (1-b3) q3 + b3 q2(a,b) q1(c)`, `q~4 = (1-b4) q4 + b4 q3(a,b,c) q3(b,c,d)/q2(b,c)` |
| `b3 = 0.05` | `--tri-backoff` default 0.05 | matches |
| `b4 = 0.2` | `--four-backoff` **default 0.05**; the reported runs passed `0.2` explicitly | doc/code mismatch: always pass `--four-backoff 0.2` |
| `s = (s1,s2,s3,s4) = (5,1,10,10)` | `lam`, fixed `1`, `--tri-scale`, `--four-scale` in `total_of` | s2 is hard-coded 1 in the s-form |
| `L = sum_n s_n KL(p[n] || q~[n])` | `total_of(*terms(M))`; `kl_n = -(t_n * log(q_n + 1e-9)).sum() + const_n` | forward KL, EPS only inside the log of q |
| `w_k = sum_{n>=k} s_n` | **new** `--w w1,w2,w3,w4` → `SoftMapLoss(cond_weights=...)` | loss = `sum_k w_k C_k` with **exact** conditional KLs, see II.2 |
| `Search.climb` (variant 2b) | `calc_cheat_seg_identifiability.py::Search.climb` | coordinate descent over `rng.permutation(512)` × 40 candidates, exact objective, ≤25 sweeps |
| Adam loop (variants 3–4b) | `optimize()` | full batch, `lr 0.05`, no minibatching; only randomness = init noise |

Disagreements / caveats found:
- **Vocabulary axis:** the identifiability script keeps `NUM_PHON = 42` (41 + unused slot, zero rows),
  the soft-map script restricts to the 40 phonemes with `t1 > 0` and renormalizes. Same oracle loss
  to 3 digits (0.2125 vs 0.213), so harmless, but the tables are not literally the same shape.
- **Marginal consistency:** `t2`, `t3`, `t4` are normalized separately over `T-1`, `T-2`, `T-3`
  positions per utterance, so `sum_c t3[a,b,c] != t2[a,b]` exactly (last-position effect, ~0.6%/order
  at ~123 phonemes/utt; the audio subset is shorter, ~96 tokens). The back-off mixing breaks
  consistency of the `q~` further. The plan's chain-rule identity `L = sum_k w_k C_k` therefore holds
  only approximately for the s-form loss; the new `--w` mode computes `C_k` exactly instead (II.2).
- **Utterance boundaries:** no n-gram crosses an utterance boundary, on either side (matches the doc).
- **Audio subset is length-biased:** the audio side uses the first 6000 *equal-length* utterances,
  which are short (mean 96 cluster tokens vs 123 phonemes for the text side) — see A2.
- `--four-backoff` default (0.05) differs from the value every reported 4-gram run used (0.2).

## II.A Corpus diagnostics (`calc_corpus_diagnostics.py`, 2026-09-11; log `plan_runs/taskA.out`)

Same split as the searches: 6000 cheat-seg audio utts (576k tokens), 33,654 disjoint text utts (4.16M
tokens). Real clus128 numbers from 2 of the 10 train-960 shards (53k utts with text; audio/text halves
disjoint).

**A1 — geminate mass.** `delta = sum_c p2[c,c] = 0.00606` (0.61% of bigram tokens; 0.00571 on all
266,927 train-960 utts). Top cells: `T T` 0.00174, `D D` 0.00087, `AH AH` 0.00065, `S S` 0.00062,
`DH DH` 0.00048, `N N` 0.00038, `L L` 0.00026, `M M` 0.00024, `R R` 0.00022, `K K` 0.00012 (word
boundaries: "at two", "had done", "the the"...). **Nonzero ⇒ the collapse-repeats pushforward has
`KL = +inf` for every table; Task C needs a diagonal back-off on `p[2]`** (~0.6% of mass, so it is a
small correction, not a structural problem).

**A2 — compression ratio.**
| audio tokens | E[T] (audio utts) | E[S] (disjoint text utts) | R = E[T]/E[S] | paired per-utt T/S mean ± std (p5 / p50 / p95) |
|---|---|---|---|---|
| cheat-seg k512 | 96.08 | 123.47 | 0.778 | 0.983 ± 0.025 (0.942 / 0.989 / 1.007) |
| real clus128 (collapsed) | 314.16 | 119.55 | **2.628** | 2.696 ± 0.333 (2.263 / 2.656 / 3.250) |
The cheat-seg unpaired `R` is *wrong* (0.78 vs the true 0.98) because the equal-length audio subset is
short — a warning that an unpaired `R` needs both sides drawn from the same length distribution. For
the real clusters both halves are random splits and `R = 2.63` is clean: **collapse-model target
`Z = E[S]/E[T] = 0.381`**, with a wide per-utterance spread (std 0.33 on 2.7).

**A3 / A4 — conditional entropies `H(x_k | x_1^{k-1})` (nats/token) and their drops.**
In-sample ML (memorizes at high order) vs 2-fold held-out with Witten-Bell (WB) / fixed Jelinek-Mercer (JM) interpolation:

| side | estimate | H(1) | H(2) | H(3) | H(4) | H(5) | drop 1→2 | drop 2→3 | drop 3→4 | drop 4→5 |
|---|---|---|---|---|---|---|---|---|---|---|
| cheat-seg audio, 6000 utts | in-sample | 6.142 | 3.588 | 2.236 | 0.892 | 0.281 | 2.554 | 1.352 | 1.344 | 0.611 |
| | held-out WB | 6.144 | 3.843 | 3.830 | 3.978 | 4.056 | 2.301 | **0.013** | −0.148 | −0.078 |
| | held-out JM 0.3 | 6.192 | 4.513 | 4.124 | 4.095 | 4.112 | 1.679 | 0.389 | 0.029 | −0.017 |
| cheat-seg audio, all 39,654 utts | held-out WB | 6.147 | 3.730 | 3.509 | 3.614 | 3.743 | 2.417 | 0.221 | −0.105 | −0.129 |
| | held-out JM 0.3 | 6.194 | 4.508 | 3.916 | 3.755 | 3.751 | 1.686 | 0.592 | 0.160 | 0.005 |
| text, 33,654 utts | in-sample | 3.350 | 2.768 | 2.367 | 2.008 | 1.606 | 0.582 | 0.401 | 0.360 | 0.401 |
| | held-out WB | 3.350 | 2.768 | 2.378 | 2.100 | 1.979 | 0.581 | **0.390** | **0.278** | 0.122 |
| | held-out JM 0.3 | 3.550 | 3.144 | 2.752 | 2.409 | 2.152 | 0.406 | 0.392 | 0.343 | 0.258 |
| real clus128 audio, 26,708 utts | held-out WB | 4.721 | 2.764 | 2.465 | 2.479 | 2.613 | 1.956 | 0.299 | −0.014 | −0.134 |
| | held-out JM 0.3 | 4.784 | 3.511 | 2.960 | 2.716 | 2.630 | 1.274 | 0.551 | 0.244 | 0.086 |

`KL3 = drop 2→3`, `KL4 = KL3 + drop 3→4` (identity `KL(P_A[3] || Markov-1) = H(x3|x2) − H(x3|x1,x2)`).

**Decision gate.** In-sample, the audio looks strongly non-Markov (KL3 1.35, KL4 2.70 nats) — but
that is memorization: 452k distinct 4-grams over 558k tokens. Held-out, the generalizable
higher-order audio structure is **KL3 ≈ 0.01–0.39 nats** (WB vs JM-0.3; 0.22–0.59 with 6.6× more
audio) and **nothing measurable at order 4**, while the text robustly gains **0.39 (order 3) + 0.28
(order 4) + 0.12 (order 5)** nats across smoothers. The result is smoothing-sensitive at 6000 audio
utts (the 512² = 262k bigram contexts are barely covered by 565k tokens), so the honest statement is:
a first-order `p_A` discards at most ~0.4 nats/token of *generalizable* audio structure, i.e. ≤ what
the text side gains at order 3 alone and less than orders 3+4 together (0.67); an HMM keeps *all*
text orders for free. **→ Task D (frozen-transition HMM) is worth building.** Caveat in the other
direction: the n-gram route uses the empirical `P_A[3], P_A[4]` *pushed through `M`* (40⁴ cells),
which averages away much of the 512⁴-cell sampling noise these held-out numbers penalize, so the
audio n-gram terms are less noisy than raw KL3/KL4 suggest — the 4-gram route did work (0.727 / 0.656).
Both routes are therefore defensible; the HMM is the one with more of the joint to gain.

## II.B Reweighting the conditional orders (done 2026-09-12; 20 SLURM jobs, `plan_runs/taskB/*.log`)

Sweep of `w = (w1,w2,w3,w4)`: current `(26,21,20,10)`, flat `(10,10,10,10)`, high-order
`(5,5,10,20)` and `(2,4,8,16)`; each with oracle start + uniform seeds 1–4; 3000 steps, τ 1.0→0.02,
`--four-backoff 0.2`. Reference (s-form, earlier session, same data): oracle-start **0.7265**
(hard 11.32); uniform seeds 1–4 **0.656 / 0.116 / 0.115 / 0.308** with hard losses 13.73 / 33.21 /
30.04 / 26.04 → loss-selected 0.656, success 1/4.

Loss = `sum_k w_k C_k` with exact conditional KLs (`--w`), so hardened losses are comparable only within a setting. `C_k` columns are the hardened map's conditional KLs (comparable across settings).

| setting w | start | hard loss | C4 | C3 | C2 | C1 | acc | PER % |
|---|---|---|---|---|---|---|---|---|
| (26,21,20,10) | oracle | 11.39 | 0.493 | 0.224 | 0.083 | 0.009 | **0.7278** | 27.2 |
| (26,21,20,10) | uniform seed 1 **(loss-selected)** | 14.06 | 0.603 | 0.282 | 0.102 | 0.010 | **0.6521** | 34.8 |
| (26,21,20,10) | uniform seed 2 | 29.86 | 1.141 | 0.604 | 0.246 | 0.046 | **0.1401** | 86.0 |
| (26,21,20,10) | uniform seed 3 | 31.12 | 1.211 | 0.654 | 0.243 | 0.032 | **0.1275** | 87.3 |
| (26,21,20,10) | uniform seed 4 | 23.96 | 1.011 | 0.496 | 0.166 | 0.018 | **0.3745** | 62.5 |
| (26,21,20,10) | _summary_ | | | | | | loss-selected 0.6521, success 1/4, oracle-start 0.7278 | |
| (10,10,10,10) | oracle | 8.11 | 0.488 | 0.222 | 0.093 | 0.009 | **0.7288** | 27.1 |
| (10,10,10,10) | uniform seed 1 **(loss-selected)** | 9.12 | 0.545 | 0.257 | 0.100 | 0.010 | **0.6918** | 30.8 |
| (10,10,10,10) | uniform seed 2 | 19.63 | 1.161 | 0.579 | 0.194 | 0.029 | **0.1066** | 89.3 |
| (10,10,10,10) | uniform seed 3 | 22.70 | 1.260 | 0.696 | 0.287 | 0.027 | **0.0585** | 94.1 |
| (10,10,10,10) | uniform seed 4 | 19.41 | 1.139 | 0.575 | 0.194 | 0.033 | **0.1062** | 89.4 |
| (10,10,10,10) | _summary_ | | | | | | loss-selected 0.6918, success 1/4, oracle-start 0.7288 | |
| (5,5,10,20) | oracle | 12.36 | 0.481 | 0.221 | 0.095 | 0.009 | **0.7335** | 26.7 |
| (5,5,10,20) | uniform seed 1 **(loss-selected)** | 12.69 | 0.495 | 0.229 | 0.090 | 0.009 | **0.7203** | 28.0 |
| (5,5,10,20) | uniform seed 2 | 32.75 | 1.205 | 0.664 | 0.303 | 0.099 | **0.0772** | 92.3 |
| (5,5,10,20) | uniform seed 3 | 33.60 | 1.222 | 0.698 | 0.317 | 0.121 | **0.0384** | 96.2 |
| (5,5,10,20) | uniform seed 4 | 34.91 | 1.294 | 0.732 | 0.294 | 0.048 | **0.0476** | 95.2 |
| (5,5,10,20) | _summary_ | | | | | | loss-selected 0.7203, success 1/4, oracle-start 0.7335 | |
| (2,4,8,16) | oracle | 9.97 | 0.486 | 0.227 | 0.091 | 0.009 | **0.7332** | 26.7 |
| (2,4,8,16) | uniform seed 1 **(loss-selected)** | 12.53 | 0.601 | 0.296 | 0.127 | 0.012 | **0.6625** | 33.7 |
| (2,4,8,16) | uniform seed 2 | 26.54 | 1.235 | 0.683 | 0.308 | 0.042 | **0.0486** | 95.1 |
| (2,4,8,16) | uniform seed 3 | 26.73 | 1.249 | 0.678 | 0.304 | 0.049 | **0.0308** | 96.9 |
| (2,4,8,16) | uniform seed 4 | 27.11 | 1.258 | 0.712 | 0.297 | 0.050 | **0.0391** | 96.1 |
| (2,4,8,16) | _summary_ | | | | | | loss-selected 0.6625, success 1/4, oracle-start 0.7332 | |

**Reading.**
- **Reformulation check:** `(26,21,20,10)` in the exact-conditional form reproduces the s-form reference
  run for run (oracle 0.7278 vs 0.7265; seeds 0.652/0.140/0.128/0.375 vs 0.656/0.116/0.115/0.308).
- **Identifiability** (oracle-start optimum, ceiling 0.740): high-order `(5,5,10,20)` **0.7335** ≈
  `(2,4,8,16)` 0.7332 > flat 0.7288 > current 0.7278. The plan's hypothesis holds — the order-4
  conditional was under-weighted — but the gain is small (27.2 → 26.7% PER; floor 26.0). The
  hardened `C_k` of the settled maps are nearly identical across settings (C4 0.48–0.49, C3 0.22,
  C2 0.08–0.09), i.e. all four weightings find essentially the same optimum region.
- **Basin** (the number that decides): loss-selected cold start **0.652 → 0.692 (flat) → 0.7203
  (`5,5,10,20`)**, i.e. **28.0% PER label-free**, within 0.013 of that setting's own oracle-start
  optimum (was 0.076 under the current weights) and 2 points off the memoryless ceiling. `(2,4,8,16)`
  gives 0.663 — more extreme is not better; the ratio, not the monotone increase, seems to matter.
- **Success rate is unchanged at 1/4 in every setting, and it is always seed 1.** The seed only sets
  the 1e-3 symmetry-breaking noise, so the basin is decided in the first, high-τ steps where the
  loss landscape is nearly weight-independent; the weights then decide how far a run in the right
  basin gets. High-order weights also remove partial credit: failed seeds land deeper at chance
  (0.03–0.08) than under the current weights (0.11–0.37; seed 4's intermediate basin at 0.37 is gone).
- **Loss-based selection is exact in all 4 settings** (the lowest hardened loss is the most accurate
  seed every time, by a wide margin: 12.7 vs 32.7–34.9 for `(5,5,10,20)`).
- **Recommendation:** make `--w 5,5,10,20` the default for label-free runs. The remaining gap is the
  1/4 success rate of the uniform start — an early-phase / annealing question, not a weighting one.
  Since the basin is fixed within the first few hundred steps and the failed seeds are separable by
  loss early, a cheap scheme is N short seeds (≤500 steps), finish only the lowest-loss one.

## II.C Unsegmented port at n ≤ 2 (Task C, 2026-09-14)

Adds the parameter-free **collapse-repeats map `B`** to the output of the induced statistics, so the
criterion compares the *collapsed* induced sequence with the text. The model is unchanged; only

    Z        = 1 - sum_c q2y[c,c]                     (collapsed length / frame length)
    q1B[c]   = (q1y[c] - q2y[c,c]) / Z
    q2B[c,c']= q2y[c,c'] * (c != c') / Z

is appended, plus an optional label-free length term `lam_z * (Z - z*)^2`.

**Implementation.** `SoftMapLoss._collapse` (variant 3, soft/Adam, `calc_soft_map_search.py
--collapse --lam-z --z-target`) and `Search._score` / `Search.z_of` (variant 2b, hard/coordinate,
the same three flags on `calc_cheat_seg_identifiability.py`). Both default off; the no-collapse path
is byte-identical in behaviour (checked: the oracle reference loss is still 0.2125 in the soft script
and 0.2125 / climb-to-0.0777 in the hard one). Three details:
- **Verified against a direct count.** Mapping synthetic cluster sequences through a hard map,
  collapsing adjacent repeats and counting gives exactly `Z`, `q1B`, `q2B` (max abs error 3e-10,
  `q2B` diagonal exactly 0). This needed one correction the plan's closed form does not mention:
  `c1` is normalized over *tokens* and `C2` over adjacent *pairs*, i.e. different denominators, so
  the diagonal has to be rescaled by `bigram_ratio = #pairs/#tokens` before it is subtracted —
  without it `Z` is off by ~1/mean-utt-length (1.7e-3 here), which matters because `Z` is exactly
  what the length term targets.
- **Diagonal back-off (A1).** `q2B[c,c] = 0` structurally, so the text bigram's geminate mass
  (A1: **0.00606**) is dropped and the table renormalized — i.e. the bigram KL is conditioned on
  "adjacent symbols differ". The unigram target is left alone (the geminates shift it by ~0.3%
  relative). No blank symbol, as required.
- **`z*` is label-free**: `min(1, E[S]/E[T])` from the two unpaired corpora. On cheat-seg that is
  `min(1, 123.47/96.08) = 1.0` by either estimate (A2's unpaired 0.78 and the true paired 0.98 are
  both < 1). The *oracle* map's own `Z` is **0.9652** — 3.5% of adjacent oracle segments carry the
  same phoneme — so the label-free target is 3.5% too high. An extra arm with `z* = 0.9652` isolates
  that (oracle-informed, sensitivity only).

### II.C.1 What the collapse costs — cheat-seg data (the measurement the plan asks for)

Variant 3 (soft/Adam), 3000 steps, τ 1.0→0.02, `lam = 5`, oracle start + 4 uniform seeds per arm.
Ceiling 0.7400, unigram chance 0.1030. Selection across seeds by hardened loss only.

| arm | start | hard loss | Z | acc | PER % |
|---|---|---|---|---|---|
| none (control) | oracle | 0.1199 | -- | **0.6332** | 36.7 |
| none (control) | uniform seed 1 | 0.1755 | -- | **0.1009** | 89.9 |
| none (control) | uniform seed 2 **(loss-selected)** | 0.1629 | -- | **0.1120** | 88.8 |
| none (control) | uniform seed 3 | 0.1753 | -- | **0.1144** | 88.6 |
| none (control) | uniform seed 4 | 0.1775 | -- | **0.1335** | 86.7 |
| B, lam_z 0 | oracle | 0.0965 | 0.9652 | **0.6600** | 34.0 |
| B, lam_z 0 | uniform seed 1 **(loss-selected)** | 0.1577 | 0.9586 | **0.1158** | 88.4 |
| B, lam_z 0 | uniform seed 2 | 0.1681 | 0.9585 | **0.1123** | 88.8 |
| B, lam_z 0 | uniform seed 3 | 0.2047 | 0.9568 | **0.1165** | 88.3 |
| B, lam_z 0 | uniform seed 4 | 0.1635 | 0.9583 | **0.1278** | 87.2 |
| B, lam_z 1 | oracle | 0.0952 | 0.9649 | **0.6532** | 34.7 |
| B, lam_z 1 | uniform seed 1 | 0.1919 | 0.9613 | **0.1190** | 88.1 |
| B, lam_z 1 | uniform seed 2 **(loss-selected)** | 0.1221 | 0.9610 | **0.1122** | 88.8 |
| B, lam_z 1 | uniform seed 3 | 0.1781 | 0.9578 | **0.1171** | 88.3 |
| B, lam_z 1 | uniform seed 4 | 0.1561 | 0.9609 | **0.1167** | 88.3 |
| B, lam_z 10 | oracle | 0.1076 | 0.9696 | **0.6480** | 35.2 |
| B, lam_z 10 | uniform seed 1 | 0.1733 | 0.9674 | **0.1267** | 87.3 |
| B, lam_z 10 | uniform seed 2 | 0.1673 | 0.9676 | **0.1113** | 88.9 |
| B, lam_z 10 | uniform seed 3 **(loss-selected)** | 0.1623 | 0.9648 | **0.1078** | 89.2 |
| B, lam_z 10 | uniform seed 4 | 0.1794 | 0.9655 | **0.1201** | 88.0 |
| B, lam_z 100 | oracle | 0.1533 | 0.9750 | **0.5358** | 46.4 |
| B, lam_z 100 | uniform seed 1 | 0.2472 | 0.9740 | **0.1088** | 89.1 |
| B, lam_z 100 | uniform seed 2 | 0.2573 | 0.9733 | **0.0815** | 91.8 |
| B, lam_z 100 | uniform seed 3 | 0.2682 | 0.9731 | **0.0685** | 93.2 |
| B, lam_z 100 | uniform seed 4 **(loss-selected)** | 0.2317 | 0.9743 | **0.1143** | 88.6 |
| B, lam_z 1000 | oracle | 0.7591 | 0.9777 | **0.3442** | 65.6 |
| B, lam_z 1000 | uniform seed 1 | 1.3350 | 0.9768 | **0.0265** | 97.4 |
| B, lam_z 1000 | uniform seed 2 **(loss-selected)** | 0.8985 | 0.9774 | **0.0427** | 95.7 |
| B, lam_z 1000 | uniform seed 3 | 0.9356 | 0.9775 | **0.0236** | 97.6 |
| B, lam_z 1000 | uniform seed 4 | 0.9184 | 0.9774 | **0.0363** | 96.4 |
| B, lam_z 100, z*=0.9652 | oracle | 0.0930 | 0.9637 | **0.6590** | 34.1 |
| B, lam_z 100, z*=0.9652 | uniform seed 1 | 0.1733 | 0.9627 | **0.1221** | 87.8 |
| B, lam_z 100, z*=0.9652 | uniform seed 2 | 0.1635 | 0.9623 | **0.1135** | 88.6 |
| B, lam_z 100, z*=0.9652 | uniform seed 3 | 0.1731 | 0.9630 | **0.1120** | 88.8 |
| B, lam_z 100, z*=0.9652 | uniform seed 4 **(loss-selected)** | 0.1566 | 0.9629 | **0.1171** | 88.3 |

Variant 2b (hard / coordinate hill-climb, `--objective kl --restarts 4`), same data and split:

| arm | oracle map loss | best random restart | climb from the oracle | acc | moved |
|---|---|---|---|---|---|
| none (control) | 0.2125 | 0.1821 @ acc 0.0826 | 0.0777 | **0.5934** | 153/512 |
| B, lam_z 0 | 0.1850 | 0.1472 @ acc 0.0708 | 0.0539 | **0.6107** | 146/512 |
| B, lam_z 10 | 0.1971 | 0.1569 @ acc 0.0951 | 0.0644 | **0.6110** | 149/512 |
| B, lam_z 100 | 0.3064 | 0.2392 @ acc 0.0934 | 0.1270 | **0.5583** | 187/512 |

**Reading.**
- **The collapse costs nothing at n ≤ 2; it is a small net gain.** Oracle-start optimum
  0.6332 → **0.6600** in the soft arm and 0.5934 → **0.6107** in the hard arm. The plan's expectation
  was that this arm would measure a *cost*; there is none to measure. Mechanically the pushforward
  removes the induced diagonal, which is mass the segmented criterion had to match against text
  geminates it cannot produce anyway, and it makes the bigram comparison conditional on "the symbol
  changed" — a cleaner target than the raw frame bigram when the audio is already one token per
  phoneme.
- **The `lam_Z` term does not move it, and at weight it hurts.** With the label-free `z* = 1.0`:
  0.6600 (0) → 0.6532 (1) → 0.6480 (10) → 0.5358 (100) → 0.3442 (1000), monotone down. That is the
  target being wrong by 3.5%, not the term being useless: with the oracle's own `z* = 0.9652` even
  `lam_z = 100` is neutral (**0.6590**, and the lowest hardened loss of any arm, 0.0930). So on
  cheat-seg the length constraint is inert-at-best — the audio is already the right length, and the
  only thing a strong `lam_Z` can do is enforce a slightly wrong length. It has to be re-measured
  where `R = 2.63` (II.C.2), which is the setting it exists for.
- **Cold starts stay at chance**, 0.068–0.134 in every arm, exactly as the plan predicted for n ≤ 2
  ("this is a measurement, not a result"). The collapse changes nothing there: the basin problem at
  n ≤ 2 is the criterion's order, not its output map. The loss-selected seed is at chance in all
  seven arms (0.043–0.117), and unlike the 4-gram setting the hardened loss does **not** rank the
  seeds usefully — at n ≤ 2 every cold optimum scores *below* the oracle map's own loss (e.g. 0.1577
  vs 0.1850 at `lam_z 0`), so there is nothing for loss-based selection to select.
- **Both search modes agree** on every conclusion, with the hard climb ~0.05 below the soft one
  throughout (as in the segmented case).

### II.C.2 The same criterion on the REAL unsegmented data (`calc_unsegmented_map.py`)

II.C.1 runs on the oracle segmentation, where the audio is already one token per phoneme (`R ≈ 1`)
and `B` has almost nothing to do. The setting `B` exists for is the real **clus128** tokens:
`R = 2.659`, so `z* = 0.3761`. There is no oracle map there, so the criterion is scored the only way
it can be — decode a held-out **paired** set (map every cluster, collapse, edit distance) → PER.
20,000 audio utts / 60,000 disjoint text utts / 1000 eval utts / 3000 for the supervised references,
all disjoint; reference 118.8 phonemes per utt.

**Supervised references** (labels used throughout — these bound what the model class can do, they are
not results):

| supervised route | PER % | hyp tokens |
|---|---|---|
| random map | 228.57 | 307.7 |
| position-proportional alignment → count map | 87.15 | 143.7 |
| + 1 Viterbi realignment | 73.55 | 93.4 |
| maximum-likelihood fit under the collapse model | 94.35 | 7.0 |
| ML + Viterbi realignment | 91.59 | 11.0 |
| **direct PER hill-climb (converged, 4/128 moved on the last sweep)** | **71.14** | 88.1 |

Two things about this table. First, **maximum likelihood is the wrong supervised reference here.**
Mapping every frame through `M` and collapsing is exactly CTC without a blank, so the marginal over
alignments is a forward pass (`collapse_nll`) and it optimizes cleanly (NLL/token 6.65 → 4.12) — but
the resulting table decodes to **7 tokens per utterance**. The likelihood is happy to spread mass
over alignments; the greedy decode is not. Second, **the machinery is correct**: on synthetic
unsegmented data with a known map and the same `R = 3.0`, all three routes recover it exactly (frame
accuracy 1.000, PER 0.16%). So 71% is a property of clus128, not of the fitting. For reference, a
tied map that ignores the audio entirely scores NLL/token 4.361 against the per-cluster map's 4.106 —
a clus128 id is worth **0.26 nats** about the phoneme under this model.

**Unsupervised arms** (uniform init, 4 seeds, 3000 steps, selection by hardened loss only):

| arm | seed | hard loss | Z | hyp len | PER % |
|---|---|---|---|---|---|
| no collapse (ablation) | 1 | 1.366 | 1.0000 | 307.1 | **214.68** |
| no collapse (ablation) | 2 | 0.805 | 1.0000 | 309.2 | **216.54** |
| no collapse (ablation) | 3 **(loss-selected)** | 0.804 | 1.0000 | 307.7 | **214.82** |
| no collapse (ablation) | 4 | 3.694 | 1.0000 | 306.7 | **213.70** |
| B, lam_z 0 | 1 | 2.099 | 0.9500 | 299.7 | **208.89** |
| B, lam_z 0 | 2 | 1.833 | 0.9414 | 296.9 | **206.72** |
| B, lam_z 0 | 3 **(loss-selected)** | 0.944 | 0.9401 | 296.9 | **207.40** |
| B, lam_z 0 | 4 | 2.370 | 0.9397 | 296.8 | **206.66** |
| B, lam_z 1 | 1 **(loss-selected)** | 2.020 | 0.8370 | 264.1 | **183.78** |
| B, lam_z 1 | 2 | 2.404 | 0.8374 | 264.3 | **183.30** |
| B, lam_z 1 | 3 | 2.180 | 0.8513 | 268.1 | **185.92** |
| B, lam_z 1 | 4 | 2.195 | 0.8333 | 262.6 | **181.72** |
| B, lam_z 10 | 1 | 12.886 | 0.3673 | 115.8 | **88.87** |
| B, lam_z 10 | 2 **(loss-selected)** | 7.348 | 0.4324 | 136.3 | **98.18** |
| B, lam_z 10 | 3 | 8.691 | 0.3583 | 113.0 | **88.27** |
| B, lam_z 10 | 4 | 9.820 | 0.4169 | 131.7 | **96.20** |
| B, lam_z 100 | 1 | 14.511 | 0.3742 | 118.0 | **92.02** |
| B, lam_z 100 | 2 | 12.123 | 0.3894 | 122.7 | **92.67** |
| B, lam_z 100 | 3 **(loss-selected)** | 6.094 | 0.3899 | 122.9 | **95.17** |
| B, lam_z 100 | 4 | 11.043 | 0.3785 | 119.3 | **92.49** |

**Shuffle control** (`calc_shuffle_control.py`'s test in PER form: score each hypothesis against a
different, length-matched utterance's reference). For the `lam_z 100` loss-selected map:
**matched 91.96% vs shuffled 92.01%, gap +0.05 points.** At chance, like every unsupervised model
in this project outside the cheat-seg n ≥ 3 runs.

**Reading.**
- **The length term is essential here and works exactly as designed.** `Z` goes 1.000 (no collapse)
  → 0.940 (`lam_z 0`) → 0.837 (1) → 0.38 ≈ `z*` (10 and 100), and the decoded length follows:
  **307.7 → 296.9 → 264.1 → 119** against a reference of **118.8**. PER follows it down, 214.8 →
  207.4 → 183.8 → 92.0. On cheat-seg the same term was inert-to-harmful (II.C.1); here it is the
  only thing that closes a 2.6x length mismatch. So `lam_Z` earns its place exactly where `R != 1`.
- **`B` alone does *not* fix the length.** At `lam_z 0` the criterion settles at `Z = 0.94`, i.e. it
  prefers a map that almost never repeats — matching a text bigram whose diagonal has been removed
  pushes *away* from repeats, so the pushforward on its own is nearly a no-op. The closed form needs
  the length term to be worth anything; that is a genuine addition to the plan's construction.
- **But the recognition result is at chance** (92% PER, shuffle gap +0.05), and hardened-loss
  selection is not usable: at `lam_z` 10 and 100 it picks the *worst* of the four seeds (98.2 of
  88.3–98.2, and 95.2 of 92.0–95.2). With every seed at chance the loss ranks fit-to-statistics, not
  accuracy.
- **The binding constraint is the model class, not the criterion.** With labels the same table
  reaches **71.14%** PER. Even a perfect unsupervised recovery of the best memoryless 128→40 map
  would therefore give ~71% PER — no usable recognition. The same model class reaches **26.0%** on
  the cheat-seg tokens. The difference is entirely the audio tokenization: 512 oracle *segment*
  clusters (one per phoneme, features averaged over the segment) against 128 *frame* clusters worth
  0.26 nats each. So **the unsegmented n ≤ 2 setting cannot discriminate between criteria** — any
  objective, however good, is capped 45 points above the cheat-seg floor.
- Consequence for the plan: Task D (frozen-transition HMM) and Task E (duration model) both keep
  more of the *text-side* joint, which is the right direction on cheat-seg, but neither enlarges the
  **emission** model, which is what the real unsegmented data is short of. Before n ≥ 3 on real
  clusters is worth running, the audio side needs either more clusters or context — i.e. a generator
  like wav2vec-U's conv net rather than a lookup table. The cheat-seg track (II.B) remains the place
  where criterion work pays off.
- **Reproducibility note:** the anneal is chaotic — the same seed under a different BLAS thread count
  lands on a different optimum (`lam_z 100`, seed 3: hardened 6.09 @ 95.2% vs 14.34 @ 91.9%). On the
  cheat-seg oracle start the spread is small (control 0.6332–0.6388, collapse 0.6557–0.6600 over
  three thread counts, so II.C.1's +0.02 gain is well outside it), but cross-run comparisons must fix
  the thread count.

## II.D Frozen-transition HMM — the master objective (Task D, 2026-09-15)

Replaces the order-4 truncation by a full-length contraction. With the audio side made first-order
Markov (`pi[x] = p_A(x)`, `A[x,x'] = p_A(x'|x)`, `D_c = diag(M[:,c])`),

    q(c_1^T) = pi' D_{c1} A D_{c2} A ... A D_{cT} 1

and the objective is `max_M sum_{c_1^T} p(c_1^T) log q(c_1^T)` over the disjoint text corpus. Since
`p` is fixed this is `min_M KL(p || q) + H(p)` — **the same forward direction** as the n-gram
criterion, at full sequence length. Script: `calc_hmm_map_search.py`.

### II.D.0 Correctness gate (`--self-test`)

The plan's mandatory test: under a Markov `p_A`, truncating the forward product to 2/3/4 factors
must reproduce the existing multilinear contractions.

| truncation | reference | max abs diff |
|---|---|---|
| 2 factors | `SoftMapLoss.induced` `q[2]` | 3.5e-13 |
| 3 factors | `induced_trigram` `q[3]` (backoff 0) | 1.1e-13 |
| 4 factors | `induced_fourgram` `q[4]` (backoff 0) | 6.9e-10 |
| 3 factors | float64 `einsum` on the dense `P3` | 1.1e-13 |
| 4 factors | float64 `einsum` on the dense `P4` | 3.2e-14 |

The 6.9e-10 at order 4 is `induced_fourgram`'s deliberate float32 stages (documented there as
~1.7e-10 against a direct count), not a mismatch — against a float64 reference the same comparison is
3.2e-14. Also checked: the scaled recursion matches a naive dense product on random sequences
(3.7e-12), and `sum_{c_1^T} q = 1` to 4e-12 for T = 2,3,4. **Gate passed.**

Implementation notes. The forward pass is batched over length-bucketed padded sequences and
**scaled** per step (normalize `alpha`, accumulate `log` of the scale) rather than run in log-space,
so the recursion stays a dense matmul. Transitions are smoothed toward the marginal
(`A <- (1-eps)A + eps*pi`, `eps = 1e-3`) so no text sequence is unreachable — the plan's point that
`b3`/`b4` have no analogue here. The hardened map is scored with a 1e-6 emission floor, so a phoneme
no cluster maps to does not give `-inf`. **The gradient needs no autograd**: `log q` is linear in each
emission along a path, so `d log q / d M[x,c]` is the Baum-Welch expected count of `(x,c)` divided by
`M[x,c]` — one E-step per gradient step, the same cost as an EM iteration.

### II.D.1 Both search modes (cheat-seg, 3000 disjoint text utts = 379,559 tokens)

Ceiling 0.7400, unigram chance 0.1030. The oracle map's own hardened objective is 2.62999.
Selection across seeds by hardened objective only.

**EM / Baum-Welch, `A` and `pi` frozen, 400 iterations** (~85 min per run):

| start | hardened objective | acc | PER % |
|---|---|---|---|
| oracle | 2.58813 | **0.7266** | 27.3 |
| uniform seed 1 **(loss-selected)** | 2.59400 | **0.7213** | 27.9 |
| uniform seed 2 | 2.60483 | **0.6994** | 30.1 |
| uniform seed 4 | 2.68303 | **0.6252** | 37.5 |
| uniform seed 3 | 2.68849 | **0.6247** | 37.5 |

**Gradient on `theta` with the same anneal, 300 steps** (converged — seed 1 plateaus from step 125):

| start | hardened objective | acc | PER % |
|---|---|---|---|
| oracle | 2.58039 | **0.7207** | 27.9 |
| uniform seed 1 **(loss-selected)** | 2.82300 | **0.3563** | 64.4 |
| uniform seed 2 | 2.86844 | **0.2781** | 72.2 |
| uniform seed 4 | 2.87538 | **0.2073** | 79.3 |
| uniform seed 3 | 2.91717 | **0.1749** | 82.5 |

### II.D.2 Basin — corrupt-the-oracle sweep (EM, 80 iterations, 2 seeds each)

| rows corrupted | init acc | **HMM-EM** | n-gram trigram arm | n-gram bigram arm |
|---|---|---|---|---|
| 0.2 | 0.583 / 0.592 | **0.7270 / 0.7267** | 0.713 | 0.620 |
| 0.4 | 0.421 / 0.451 | **0.7244 / 0.7261** | 0.714 | 0.590 |
| 0.6 | 0.294 / 0.325 | **0.7246 / 0.7261** | 0.702 | 0.545 |
| 0.8 | 0.153 / 0.158 | **0.7231 / 0.7166** | 0.668 | 0.365 |

**Reading.**
- **The HMM buys basin, not identifiability.** Oracle-start optimum **0.7266** against the order-4
  n-gram criterion's 0.7335 (ceiling 0.740) — slightly *worse*. Keeping all text orders does not
  pin the map down better than truncating at 4; that question was already nearly saturated (II.B).
- **The success rate is the win: 4/4 against 1/4.** Every uniform cold seed lands at 0.62–0.72,
  where the n-gram criterion put three of four seeds at 0.04–0.08, i.e. at chance. The
  loss-selected cold start, **0.7213 (27.9% PER)**, ties the n-gram criterion's 0.7203 — but it is
  now the *typical* outcome rather than the one seed in four that worked.
- **The corrupt-the-oracle basin is flat.** Every corruption level returns to the objective's own
  optimum: a start at acc 0.153 recovers to 0.723, where the trigram arm reached 0.668 and the
  bigram arm 0.365. Combined with the cold-start result, "search, not objective, is the bottleneck"
  is now a fixed problem on cheat-seg, exactly as the plan predicted EM would fix it.
- **EM beats the annealed gradient decisively** (0.7213 vs 0.3563 loss-selected), and the gradient
  runs are converged, not starved — seed 1 plateaus at soft objective 2.740 where EM reaches 2.563.
  A monotone fixed-point iteration in the natural parameterization is simply the better optimizer
  here; the anneal schedule was tuned for the n-gram loss and does not transfer. Use `--mode em`.
- **Iteration count matters more than expected.** At 80 iterations the same seeds gave
  0.695 / 0.616 / 0.532 / 0.496; at 400 they give 0.721 / 0.699 / 0.625 / 0.625, and seeds 3–4 were
  *still* climbing at step 399. Do not read an unconverged EM run.
- **The objective's optimum is still displaced from the truth**, as in the n-gram case: more
  optimization lowers it and lowers accuracy with it (EM oracle start 2.58842 @ 0.7273 at 80
  iterations → 2.58813 @ 0.7266 at 400; the gradient mode reaches a *lower* 2.58039 at a *lower*
  0.7207). So the hardened objective orders seeds correctly **within** a mode — which is all
  label-free selection needs — but is not a reliable comparator across modes.
- **It is also ~5x cheaper** than the 4-gram criterion (85 min vs ~7 h per run), with no sparse
  4-gram tables and no back-off constants to choose.

### II.D.3 Combining the two criteria — and a correction to how II.B was read

The two criteria have complementary failures: the HMM finds the basin (4/4 seeds) but its optimum
sits at 0.7266; the order-4 n-gram criterion has the better optimum but finds it once in four seeds.
So chain them. Both directions, plus a `tau_start` sweep on the refinement stage:

| procedure | acc | PER % | note |
|---|---|---|---|
| order-4 n-gram alone, uniform seed 1 | 0.7209 | 27.9 | 1/4 seeds; reproduces II.B's 0.7203 |
| HMM-EM alone, uniform, loss-selected | 0.7213 | 27.9 | 4/4 seeds |
| n-gram → HMM-EM | 0.7242 | 27.6 | converged by EM iteration 100 |
| HMM-EM → n-gram, `--tau-start 0.2` | 0.7228 | 27.7 | refinement barely moves |
| **HMM-EM → n-gram, `--tau-start 1.0`** | **0.7362** | **26.4** | best hardened step; 0.7308 at the final step |
| — the n-gram criterion's own oracle-start optimum | 0.7356 | 26.4 | best hardened step (0.7335 final) |
| — memoryless ceiling | 0.7400 | 26.0 | |

**A fully label-free procedure now reaches the criterion's own optimum.** Stage 1 (HMM-EM from
uniform, 4 seeds, 400 iterations, pick by hardened objective) lands at 0.7213; stage 2 (order-4
n-gram, `--init assign --tau-start 1.0`) takes it to hardened loss **12.3184 @ acc 0.7362**, which
is the same plateau the *oracle-start* run of that criterion reaches (12.2920 @ 0.7356). The
remaining gap to the memoryless ceiling is **0.4 PER points**, and it is the objective's displaced
optimum, not the search. Previous best label-free was 0.7203 / 28.0%.

`tau_start` is the whole story of stage 2: at 0.2 the refinement is frozen in the basin it was
handed (0.7213 → 0.7228), at 1.0 the high-temperature phase lets it re-explore (→ 0.7362). The
reverse order (n-gram → EM) gives only 0.7242 and its first stage is the unreliable one.

**Correction to II.B.** `optimize()` reported the map at the *final* annealing step, but the anneal
tail drifts past the optimum, and the hardened loss sees it. Reading the logged steps of the same
runs by lowest hardened loss — a legitimate label-free criterion — gives:

| II.B run (`w = 5,5,10,20`) | final step | best hardened step |
|---|---|---|
| oracle start | 12.3632 @ **0.7335** | 12.2920 @ **0.7356** (step 2000) |
| uniform seed 1 (loss-selected) | 12.6876 @ **0.7203** | 12.6443 @ **0.7263** (step 1500) |

So II.B's headline is **0.7263 = 27.4% PER**, not 0.7203 / 28.0%; the II.B tables are final-step
values throughout and are all slightly pessimistic. `--keep-best` (opt-in, off by default so the
recorded runs are unchanged) now returns the lowest-hardened-loss logged step.

**Verification of the code change.** The pre-Task-C and current `calc_soft_map_search.py` produce
bit-identical results at matched thread counts (2 threads: hard 0.1747 @ acc 0.1077; 8 threads:
0.1920 @ 0.1116) — and that pair also re-confirms the BLAS-thread sensitivity noted in II.C.2, which
is why the check has to be run at matched thread counts.

## II.E Emission model (Task E, re-scoped 2026-09-17) — plan in `PLAN_TASK_E.md`, not yet run

The original Task E (run-length / duration model on the unsegmented setting) is **dropped**: II.C.2
measured the *supervised* ceiling of that model class at **71.1% PER**, and a duration term sits
inside that cap. Re-scoped to the emission model, since that is now the binding constraint —
the criterion reaches 26.4% against a 26.0% ceiling on cheat-seg (II.D.3).

Enabling finding (verified 2026-09-17, no new jobs): the featurize job kept **every** stage of
fairseq's `prepare_audio.sh` for all 278,400 train-other-960 utts, giving a ready-made resolution
ladder — `precompute_pca512` (514.4 tokens/utt), `..._cls128_mean` (317.5), `..._cls128_mean_pooled`
(159.0), against 119.55 phonemes/utt. Two consequences:
- The stream the wav2vec-U setup calls "the features" is the `_mean_pooled` one, i.e. a **2× pooled**
  version of the collapsed cluster sequence — *not* frame-synchronous with the clus128 ids
  (`clus_len == 2*feat_len`, or `2*feat_len-1` when odd; checked on 2794 paired seqs).
- At 159 tokens/utt it sits at **`R = 1.33`**, much closer to one-token-per-phoneme than clus128's
  2.65, **without any oracle segmentation**. II.C.2 measured only the 317-token discrete stream.

Task E is therefore a supervised-ceiling ladder over that data (E.1, gate G1 at ≤ 45% PER), then the
unchanged II.D.3 two-stage criterion on whichever rung clears it (E.2), and only then a learned
generator with the n-gram objective in place of wav2vec-U's GAN (E.3). Details, protocol, decision
gates and carried-over guardrails: `PLAN_TASK_E.md`.

### II.E.1 The supervised-ceiling ladder (`calc_emission_ladder.py`, 2026-09-17) — **G1 cleared**

Three SLURM jobs, one per feature stage, ~1 h each on 16 cores (`plan_runs/taskE/{pooled,cls_mean,
pca512}.log`, launcher `launchers/taskE/launch.sh`). Splits as in II.C.2 but on the tag universe of
the feature dump: 20,000 k-means-fit utts / 60,000 text / **1000 paired eval** / 3000 paired for the
fits, disjoint, 266,927 common; 40 phonemes; reference **120.06 phonemes/utt**.

**Harness validation.** The clus128 control reproduces II.C.2 on a different split: +1 Viterbi
**73.57%** (II.C.2: 73.55%), PER hill-climb **72.09%** (II.C.2: 71.14%).

**LABELS ARE USED THROUGHOUT.** E.1 selects a representation, not a model. It is a diagnostic; any
later label-free number inherits a label-dependent choice of representation.

| rung | symbols | tok/utt | **PER %** | hyp len | I nats | via |
|---|---|---|---|---|---|---|
| a clus128 *(control)* | 128 | 319.2 | **72.09** | 96.8 | 0.483 | PER hill-climb |
| pca512 b k128 | 128 | 516.5 | 67.80 | 98.0 | 1.066 | PER hill-climb |
| pca512 c k512 | 512 | 516.5 | 75.58 | 177.5 | 1.737 | PER hill-climb |
| pca512 c k2048 | 2048 | 516.5 | 78.64 | 185.2 | 1.915 | +1 Viterbi |
| pca512 d linear probe | — | 516.5 | 75.62 | 169.8 | 2.520 | +1 Viterbi |
| pca512 d MLP probe | — | 516.5 | 99.32 | 207.4 | 2.458 | +1 Viterbi |
| pca512 f seg x1.25 k512 | 512 | 150.6 | 53.85 | 114.5 | 0.933 | PER hill-climb |
| pca512 f seg x1.25 probe | — | 150.6 | 47.92 | 102.0 | 1.569 | +1 Viterbi |
| pca512 f seg x1.00 k512 | 512 | 120.5 | 59.39 | 94.6 | 0.442 | PER hill-climb |
| pca512 f seg x1.00 probe | — | 120.5 | 54.84 | 86.6 | 0.784 | +1 Viterbi |
| cls_mean b k128 | 128 | 319.2 | 66.20 | 100.7 | 0.911 | PER hill-climb |
| cls_mean c k256 | 256 | 319.2 | 65.83 | 144.2 | 1.417 | PER hill-climb |
| cls_mean c k512 | 512 | 319.2 | 62.86 | 158.1 | 1.702 | PER hill-climb |
| cls_mean c k1024 | 1024 | 319.2 | 63.81 | 161.9 | 1.758 | +1 Viterbi |
| cls_mean c k2048 | 2048 | 319.2 | 61.45 | 161.2 | 1.829 | +1 Viterbi |
| cls_mean d linear probe | — | 319.2 | 53.56 | 133.9 | 2.468 | +1 Viterbi |
| cls_mean d MLP probe | — | 319.2 | 69.69 | 156.5 | 2.360 | +1 Viterbi |
| cls_mean e context ±1 | 384 | 319.2 | 61.71 | 130.6 | 1.760 | +1 Viterbi |
| cls_mean f seg x1.25 k512 | 512 | 150.9 | 51.88 | 117.3 | 1.114 | PER hill-climb |
| **cls_mean f seg x1.25 probe** | — | 150.9 | **42.74** | 99.9 | 1.789 | +1 Viterbi |
| cls_mean f seg x1.00 k512 | 512 | 120.7 | 56.84 | 97.0 | 0.435 | PER hill-climb |
| cls_mean f seg x1.00 probe | — | 120.7 | 53.14 | 83.9 | 0.771 | +1 Viterbi |
| pooled b k128 | 128 | 159.9 | 56.64 | 112.2 | 1.109 | PER hill-climb |
| pooled c k256 | 256 | 159.9 | 51.05 | 116.7 | 1.347 | PER hill-climb |
| pooled c k512 | 512 | 159.9 | 46.58 | 115.3 | 1.485 | PER hill-climb |
| **pooled c k1024** | 1024 | 159.9 | **44.56** | 115.4 | 1.560 | +1 Viterbi |
| **pooled c k2048** | 2048 | 159.9 | **41.89** | 116.1 | 1.648 | +1 Viterbi |
| **pooled d linear probe** | — | 159.9 | **34.74** | 94.9 | 2.264 | +1 Viterbi |
| pooled d MLP probe | — | 159.9 | 48.57 | 104.2 | 2.226 | +1 Viterbi |
| pooled e context ±1 | 384 | 159.9 | 55.04 | 92.7 | 1.503 | +1 Viterbi |
| pooled f seg x1.00 k512 | 512 | 120.7 | 50.19 | 99.6 | 0.706 | PER hill-climb |
| **pooled f seg x1.00 probe** | — | 120.7 | **41.67** | 80.1 | 1.242 | +1 Viterbi |
| pooled f seg x0.80 k512 | 512 | 96.6 | 60.74 | 82.2 | 0.892 | PER hill-climb |
| pooled f seg x0.80 probe | — | 96.6 | 63.27 | 72.4 | 1.130 | +1 Viterbi |

`I` = I(token; phoneme) in nats under each rung's own +1-Viterbi joint. It is **not** II.C.2's 0.26
nats, which came from the collapse-marginal NLL of a tied vs per-cluster map — a different estimator.
Compare `I` within this table only, and with care across rungs of very different token counts.

**Readings.**
- **Granularity dominates every other factor.** At a fixed alphabet of 128 symbols the ceiling is
  67.8 (514 tok/utt) → 66.2 (317) → **56.6** (159), and the original clus128 ids are the worst point
  of all at 72.1. The optimum is around **120–160 tokens/utt** (`R` ≈ 1.0–1.33): pushing below it
  reverses the gain (pooled `f seg x0.80`, 96.6 tok/utt, is 60.7 / 63.3). The single most valuable
  thing in the pipeline turns out to be fairseq's unremarkable `mean_pool --subsample-rate 0.5`.
- **Resolution helps only at the right granularity, and hurts at the wrong one.** k 128→2048 moves
  the ceiling −14.7 points on `pooled` (56.6 → 41.9), −4.7 on `cls_mean`, and **+10.8 on `pca512`**
  (67.8 → 78.6). The mechanism is visible in the hypothesis length: on the fine-grained stream more
  clusters mean more argmax flicker, so the collapse decode emits 98 → 185 tokens against a
  120-token reference. More symbols only pay once each token is long enough to be worth naming.
- **Segmentation is what rescues a fine-grained stream, and does nothing for an already-pooled one.**
  Agglomerative merge to ~150 segments/utt takes `pca512` from 75.6 to **47.9** and `cls_mean` from
  53.6 to **42.7** — the largest single improvement in the table. On `pooled` it does not help
  (41.7 vs 34.7 unsegmented). Mild **over**-segmentation is right: x1.25 beats x1.00 beats x0.80 on
  every stage, i.e. leave the collapse decode something to merge rather than committing boundaries.
- **Quantization costs ~7 points, not 40.** Best discrete rung `pooled k2048` 41.9 against the
  `pooled` linear probe 34.7. The lookup table is not the main problem; the token stream was.
- **Context (±1 one-hots) is a weak substitute for resolution.** 55.0 vs 56.6 at k=128 on `pooled`
  (−1.6) and 61.7 vs 66.2 on `cls_mean` (−4.5), where simply raising k buys 3–9x that.
- **A better frame classifier can decode worse.** The MLP reaches a much lower frame CE than the
  linear probe (1.08 vs ~1.9 on `pca512`) and a comparable `I`, yet decodes at 99.3% PER with 207
  tokens against a 120-token reference. Frame accuracy and collapse-decode PER are different
  objectives: the higher-capacity model flickers, and every flicker survives the collapse as an
  insertion. This is the same pathology wav2vec-U's segmenter and smoothness penalty exist to fix,
  and it is a warning for E.3.

**Follow-up sweep** (`plan_runs/taskE/seg_sweep.log`): E.1 ran rung f only at k=512, and only at
x1.00/x0.80 on the `pooled` stage. Sweeping the segmented stream properly:

| rung | symbols | tok/utt | R | PER % | I nats |
|---|---|---|---|---|---|
| pooled f seg x1.25 k512 | 512 | 150.9 | 1.257 | 46.15 | 1.429 |
| pooled f seg x1.25 k1024 | 1024 | 150.9 | 1.257 | **44.43** | 1.510 |
| pooled f seg x1.25 k2048 | 2048 | 150.9 | 1.257 | **41.74** | 1.600 |
| pooled f seg x1.25 probe | — | 150.9 | 1.257 | **32.27** | 2.209 |
| pooled f seg x1.10 k512 | 512 | 132.8 | 1.106 | 47.43 | 0.996 |
| pooled f seg x1.10 k1024 | 1024 | 132.8 | 1.106 | **44.45** | 1.073 |
| pooled f seg x1.10 k2048 | 2048 | 132.8 | 1.106 | **43.44** | 1.107 |
| pooled f seg x1.10 probe | — | 132.8 | 1.106 | **36.06** | 1.637 |
| pooled f seg x1.00 k512 | 512 | 120.7 | 1.005 | 50.19 | 0.706 |
| pooled f seg x1.00 k1024 | 1024 | 120.7 | 1.005 | 48.39 | 0.754 |
| pooled f seg x1.00 k2048 | 2048 | 120.7 | 1.005 | 46.33 | 0.803 |
| pooled f seg x1.00 probe | — | 120.7 | 1.005 | **41.67** | 1.242 |

Segmentation + resolution compose: the best continuous rung is now **32.27%** (x1.25 probe) and the
best discrete **41.74%** (x1.25, k=2048), against clus128's 71.1%. The over-segmentation factor and
the alphabet size trade off smoothly, and every step of `x` costs ~2–4 points at fixed k.

### G1 verdict — **proceed to E.2**

Five rungs clear the ≤ 45% gate: `pooled d linear probe` **34.74**, `pooled f seg x1.00 probe`
41.67, `pooled c k2048` **41.89**, `cls_mean f seg x1.25 probe` 42.74, `pooled c k1024` **44.56**.
Against the 71.1% that II.C.2 was capped at, the ladder moves the ceiling by **29–36 points**.

E.2 needs a **finite symbol alphabet** (the II.B / II.D count tables are built over it), so the
continuous probes are not directly usable — they belong to E.3. The discrete candidates are:

| candidate | ceiling | map size | note |
|---|---|---|---|
| `pooled c k2048` | 41.89 | 2048 x 40 = 81,920 | best discrete rung; 2048² = 4.2M bigram cells |
| `pooled c k1024` | 44.56 | 40,960 | clears G1 |
| `pooled c k512` | 46.58 | 20,480 | **in the 45–55 band**, but exactly the map size the criterion was validated at on cheat-seg |

Note the tension: the rung that clears G1 best is the one the criterion will find hardest, since its
higher-order count tables grow as kⁿ while the text side is unchanged. The plan's own reasoning
(II.C.2, "the criterion lands within 0.4 points of its ceiling") sets the expectation for E.2 at
**~35–47% PER if the criterion performs as it does on cheat-seg** — a usable result, but no rung
here reaches cheat-seg's 26.0%, so E.2 must not be quoted against that number.

### II.E.2 The validated criterion on a real (segmentation-free) stream — **negative, and it says why**

`calc_e2_unsup_map.py` (a driver only: the criterion code is imported unchanged from
`calc_soft_map_search.py` / `calc_hmm_map_search.py`). Two segmentation arms of the E.1 rung
`pooled f seg xN k512`, both with the II.D.3 recipe: HMM-EM from uniform (4 seeds, 400 iterations,
selected by hardened objective) → order-4 n-gram refinement (`w = 5,5,10,20`, `--tau-start 1.0
--keep-best`). 6000 audio utts / 30,000 disjoint text utts (3000 for the HMM) / **1000 paired eval**,
all disjoint; reference 120.06 phonemes/utt. 18 SLURM jobs, `plan_runs/taskE/E2*.log`.

| arm | tok/utt | R | supervised ceiling | ceiling shuffle gap |
|---|---|---|---|---|
| x1.00 | 121.3 | 1.011 | 49.38% | **+33.01** |
| x1.25 | 151.7 | 1.263 | 46.54% | **+39.38** |
| *(random map, x1.00)* | | | 93.30% | +0.16 |

**Stage 1 — HMM-EM from uniform, all 8 seeds (no labels):**

| arm | seed | hardened | PER % | shuffle gap |
|---|---|---|---|---|
| x1.00 | 4 **(loss-selected)** | 3.07852 | 87.19 | −0.05 |
| x1.00 | 2 | 3.09284 | 86.86 | +0.56 |
| x1.00 | 3 | 3.09397 | 86.25 | +0.99 |
| x1.00 | 1 | 3.17675 | 87.37 | −0.11 |
| x1.25 | 1 **(loss-selected)** | 3.11739 | 98.71 | −0.07 |
| x1.25 | 2 | 3.11833 | 98.55 | −0.20 |
| x1.25 | 3 | 3.13455 | 98.25 | +0.19 |
| x1.25 | 4 | 3.21369 | 98.70 | −0.21 |

**Stage 2 — order-4, from three different starts:**

| arm | start | hardened | PER % | shuffle gap |
|---|---|---|---|---|
| x1.00 | uniform seed 3 **(loss-selected)** | 32.6199 | 86.37 | +1.13 |
| x1.00 | uniform seed 2 | 33.2621 | 86.51 | +0.65 |
| x1.00 | uniform seed 4 | 33.7487 | 86.53 | +0.78 |
| x1.00 | uniform seed 1 | 34.9111 | 87.70 | +0.01 |
| x1.00 | the loss-selected EM map | 35.0234 | 87.28 | +0.01 |
| **x1.00** | **the SUPERVISED map (labels — probe)** | **28.8029** | **60.18** | **+26.39** |
| x1.25 | the loss-selected EM map | 35.0250 | 98.60 | −0.24 |
| **x1.25** | **the SUPERVISED map (labels — probe)** | **29.4600** | **66.02** | **+30.84** |

**HMM-EM from the supervised map (labels — the stage-1 probe):** x1.00 hardened **2.96047**, PER
65.85, gap +20.90; x1.25 hardened **3.00688**, PER 72.89, gap +23.69.

**Reading.**
- **Every label-free run is at chance.** 8 HMM-EM seeds and 4 cold order-4 seeds, shuffle gap −0.24
  to +1.13, against +33.0 / +39.4 for the supervised map on the same stream and +0.16 for a random
  one. The control is sensitive; the criterion found nothing. Loss-selected label-free result:
  **86.37% PER, gap +1.13** — chance.
- **Identifiability holds; the basin fails.** This is the decisive split the plan asks for, and both
  criteria agree. The supervised-map start scores **strictly better than every cold optimum** under
  both objectives and in both arms — HMM 2.96047 vs cold 3.07852–3.21369; order-4 28.8029 vs
  32.6199–35.0250. So the objective is *not* ranking wrong maps above good ones (the `ce` failure
  shape of `calc_cheat_seg_identifiability.py`). What fails is the search: **0/8 cold seeds**, where
  cheat-seg HMM-EM gets **4/4**.
- **But the optimum is displaced far more than on cheat-seg, and that caps the whole setting.**
  Optimizing *from* the supervised map degrades it: 49.38 → 60.18 PER (order-4) and 49.38 → 65.85
  (HMM-EM), while the objective falls. On cheat-seg the same move costs ~1 PER point (0.740 → 0.727
  accuracy). So even a perfect initializer would land near **60% PER** here, 11 points below this
  stream's own 49.4% ceiling. Fixing the search alone would not produce a usable system.
- **Over-segmentation hurts the criterion exactly as predicted.** x1.25 has the better supervised
  ceiling (46.54 vs 49.38) and twice the token-phoneme information (I = 1.453 vs 0.726 nats), yet is
  worse on every criterion axis: cold runs 98–99% PER against 86–87%, and a bigger ceiling-start
  degradation (46.54 → 66.02 vs 49.38 → 60.18). The induced bigram carries ~20% diagonal mass
  against a text bigram with 0.6% geminates, and the criterion spends itself fighting that instead of
  finding the labelling. The n ≤ 2 collapse pushforward (Task C) is the correction, but it is not
  derived at order 3–4 — where all the identifying power is.
- **Why cheat-seg worked and this does not.** The emission is much weaker: I(token; phoneme) is
  0.726 nats (x1.00) against ~2.3 for the cheat-seg oracle segment clusters, and the audio n-grams
  are correspondingly noisier because the segmentation is estimated rather than given. The criterion
  was validated where one audio token *is* one phoneme; here it is a noisy guess at one.
- **A hard constraint found while setting this up:** stage 2's dense `[k², m, m]` intermediate is
  1.7 GB at k=512, ~6.7 GB at k=1024 and ~27 GB at k=2048. The best discrete rung of II.E.1
  (`x1.25 k2048`, 41.74%) is therefore **out of reach** for the order-4 criterion as implemented.
  E.2 had to run at k=512, i.e. at a 46–49% ceiling rather than 41.7%.

**Verdict.** E.2 is a negative result with a clear diagnosis: the objective still identifies the
right map on real audio, but (a) its optimum is displaced ~11 PER points from the ceiling, and (b)
the cold-start basin that cheat-seg's HMM-EM found 4 times out of 4 is not found at all. Both trace
to the same cause — the emission is far less informative per token than an oracle segment cluster,
so the statistics being matched carry proportionally less signal than the noise. The lever is not a
better search: it is a stronger emission model, which is E.3 (a learned generator on the continuous
features, whose supervised ceiling is **32.27%**, with the n-gram objective in place of the GAN).

### II.E.3 A learned generator with the n-gram objective instead of a GAN (`calc_e3_generator.py`, 2026-09-24) — **identifiability strong, basin 0/12**

Generator = standardize → `Conv1d(512, 40)` softmax over E.1's label-free agglomerative segments of
the `pooled` stage. Loss = II.B's `sum_k w_k C_k`, `--w 5,5,10,20`, back-offs 0.05 / 0.2, unchanged;
the induced q_n are soft n-grams of the posteriors, and the loss is evaluated at an EMA (0.95) of the
corpus statistics with the gradient taken through the current batch (256 utts). τ 1 → 0.02 over 6000
Adam steps (lr 1e-3), keep-best by **hardened loss** (exact n-grams of the argmax stream over all
19,998 audio utts). Same splits/eval set as E.1/E.2; 30k text utts (= E.2's targets). One GPU
(RTX 2080 Ti / GTX 1080) per run, ~3–3.8 h. Logs `plan_runs/taskE/e3_*.log`, launcher `launch_e3.sh`.

**Gates.** `selftest`: on a hard map of the E.2 stream both the exact and the soft n-gram path
reproduce `SoftMapLoss.hard` to 1e-6 (3 random maps), and the EMA surrogate's gradient equals the
full gradient when EMA = batch (rel err 0). Supervised generator (labels; E.1's protocol) with
`--kernel 1` reproduces E.1: **42.21%** at x1.00 (E.1 41.67), **32.10%** at x1.25 (E.1 32.27).
wav2vec-U's `--kernel 4` is *worse* under the same protocol (52.30 / 40.78), so all runs use kernel 1.
Data: 2 of 20,000 audio utts had NaN / |x| > 1e5 features (the wav2vec-U dump's `max_abs_value=1e5`
filter) and are dropped; eval/ceiling sets are clean.

**Arms.** `plain` = the criterion as validated. `no-repeat` = every n-gram cell with an adjacent
repeat dropped on both sides + Task C's length term `lam_z (Z − z*)²`, `lam_z` 100, z* = unpaired
mean phonemes / mean segments. Exact for n ≤ 2 (Task C's pushforward); for n = 3, 4 an approximation
(only windows whose interior tokens have run length 1 are counted). The x1.00 `no-repeat` arm was
added after the first `plain` probe reading (see readings) — a label-informed choice of arm.

| arm | init | hardened (start → kept) | PER % | hyp | shuffled | **gap** |
|---|---|---|---|---|---|---|
| x1.00 plain | uniform s1 | 69.74 → 18.4260 | 88.22 | 113.2 | 87.66 | −0.55 |
| x1.00 plain | uniform s2 | 69.65 → **17.8669** ← loss-sel. | 87.58 | 114.0 | 87.62 | +0.04 |
| x1.00 plain | uniform s3 | 70.15 → 18.1831 | 86.77 | 113.3 | 87.53 | +0.76 |
| x1.00 plain | uniform s4 | 69.07 → 18.0168 | 87.82 | 113.5 | 87.75 | −0.06 |
| x1.00 plain | *supervised (labels)* | 33.50 → **12.9904** | *37.52* | 108.4 | 86.02 | *+48.51* |
| x1.00 no-repeat | uniform s1 | 64.41 → 16.6703 | 87.06 | 109.4 | 86.56 | −0.50 |
| x1.00 no-repeat | uniform s2 | 64.30 → **16.2745** ← loss-sel. | 85.99 | 110.2 | 86.49 | +0.50 |
| x1.00 no-repeat | uniform s3 | 64.93 → 16.6047 | 85.74 | 110.0 | 86.50 | +0.76 |
| x1.00 no-repeat | uniform s4 | 63.82 → 16.6089 | 86.59 | 109.2 | 86.49 | −0.10 |
| x1.00 no-repeat | *supervised (labels)* | 29.72 → **10.2551** | *32.85* | 104.5 | 84.87 | *+52.02* |
| x1.25 no-repeat | uniform s1 | 61.38 → **14.9421** ← loss-sel. | 90.77 | 120.5 | 90.56 | −0.21 |
| x1.25 no-repeat | uniform s2 | 60.80 → 15.5312 | 89.93 | 122.3 | 90.45 | +0.52 |
| x1.25 no-repeat | uniform s3 | 61.65 → 15.4767 | 89.87 | 122.2 | 90.45 | +0.58 |
| x1.25 no-repeat | uniform s4 | 61.11 → 14.9698 | 90.29 | 122.1 | 90.46 | +0.17 |
| x1.25 no-repeat | *supervised (labels)* | 16.36 → **6.5912** | ***23.53*** | 114.9 | 86.99 | ***+63.46*** |

Supervised starts: x1.00 generator at 42.21% PER, x1.25 at 32.10%.

**Readings.**
- **Identifiability: holds, and strongly — the reverse of the worry after E.2.** In every arm the
  run started from the supervised generator settles at a hardened loss far below every cold optimum
  (6.59 vs 14.94–15.53; 10.26 vs 16.27–16.67; 12.99 vs 17.87–18.43), i.e. 2.3–8.4 units, where on E.2's
  discrete stream the margin was ~3.8 on a similar scale and the optimum sat 11–16 PER points *worse*
  than the start.
- **The optimum is no longer displaced away from the truth — optimizing the label-free objective
  from the supervised generator IMPROVES on it**: 32.10 → **23.53%** PER at x1.25 (gap +63.5, the
  largest measured in this project), 42.21 → 32.85 (x1.00 no-repeat), 42.21 → 37.52 (x1.00 plain).
  So E.1's "ceiling" here is a ceiling of its *supervised fitting protocol* (proportional alignment +
  one Viterbi pass, frame CE), not of the model class: the n-gram objective is a better training
  signal for the linear generator than that protocol's alignments. Not a label-free result — the start
  uses labels — and not comparable to cheat-seg's 26.0% (different ceiling definition; same 40-phoneme
  scoring, same eval set as E.1/E.2). The x1.25 trajectory reaches **20.81%** at step 250 and then
  drifts up to 24.5 while the loss keeps falling (kept step 3000 → 23.53): a residual displacement of
  ~3 points, the cheat-seg pattern (II.B), not E.2's.
- **Basin: fails completely.** 0/12 cold seeds above chance (gap −0.55…+0.76), in all three arms,
  every seed ending at a nearly identical hardened loss (spread ≤ 0.6). The uniform start does not
  pick a basin here the way it did on cheat-seg (1/4 there for the order-4 loss) — the cold runs all
  converge to the same wrong basin, which the objective itself ranks as clearly worse.
- **`no-repeat` is the right form of the criterion for a collapse decode**, at both segmentations: the
  supervised x1.00 generator emits Z = 0.66 (34% adjacent repeats, which the decode merges) against
  0.6% geminates in text, and the plain criterion charges all of it (start 33.50, *above* the cold
  optima's ~18 — which initially looked like an identifiability failure and is not: after
  optimization it lands at 12.99, below them). `no-repeat` lowers every row and gives the better probe
  optimum (32.85 vs 37.52).
- **Over-segmentation now helps**, unlike E.2: x1.25 has the best probe (23.53 vs 32.85) and the best
  loss-selected cold loss. With repeats removed from the statistics, the reason it hurt the discrete
  criterion (diagonal mass) is gone.

**Verdict.** E.3 fixes the two things E.2 diagnosed as the emission's fault: with a continuous
generator the criterion identifies the map with a wide margin and its optimum sits *better* than the
supervised reference (23.5% PER at x1.25). What remains is exactly one problem, the **basin**: no cold
start finds it. That is the problem the frozen-transition HMM solved on cheat-seg (II.D, 4/4 vs 1/4),
and the next step is to measure the basin's width (a corrupt-the-supervised-generator sweep, as
`--basin` on cheat-seg) before choosing a label-free initializer for it.

#### II.E.3b Basin sweep (2026-09-29, `--init corrupt`, x1.25 no-repeat, kernel 1)

Start = the supervised generator (32.10% PER; **labels used** — a diagnostic of the basin, not a
result), damaged along two axes: `perm f` relabels a fraction f of the 40 output phonemes among
themselves (a derangement: sharp emission, wrong labels on that subset); `mix a` sets
`W <- (1-a) W + a N` with equal-norm Gaussian N (all labels right, signal weakened). Otherwise the
II.E.3 settings; keep-best by hardened loss. Logs `plan_runs/taskE/e3_basin_*.log`.

| start | start PER | start gap | → hardened (kept) | **final PER** | final gap |
|---|---|---|---|---|---|
| perm 0.2 s1 | 46.75 | +38.64 | 6.7387 | **23.58** | +63.71 |
| perm 0.4 s1 | 61.85 | +24.75 | 6.9008 | **23.93** | +63.39 |
| perm 0.6 s1 | 76.01 | +13.24 | 7.1878 | **24.52** | +62.94 |
| perm 0.6 s2 | 70.95 | +14.73 | 14.7392 | 68.35 | +20.86 |
| perm 0.8 s1 | 84.33 | +3.72 | 16.3030 | 89.14 | +0.60 |
| perm 1.0 s1 | 88.16 | −0.14 | 16.0049 | 89.37 | −0.29 |
| perm 1.0 s2 | 89.26 | −0.40 | 17.1776 | 89.81 | −0.31 |
| mix 0.5 s1 | 59.15 | +26.74 | 6.7146 | **23.66** | +63.59 |
| mix 0.8 s1 | 89.58 | **+2.95** | 6.8501 | **23.61** | +63.56 |
| mix 0.95 s1 | 93.19 | +0.45 | 16.7701 | 90.23 | +0.74 |

References: undamaged supervised start → 6.5912 / 23.53%; cold uniform seeds 14.94–15.53 / chance.

**Readings.**
- **The basin is all-or-nothing, and its bottom is one point.** Every start that recovers lands at
  23.5–24.5% PER and hardened 6.59–7.19, whatever the damage — the analogue of cheat-seg's flat basin
  under the HMM (II.D). The failures land at chance (16.0–17.2) or, once, in an intermediate basin
  (perm 0.6 s2: 68.35%, hardened 14.74).
- **The hardened loss ranks all ten outcomes in PER order** (6.59–7.19 < 14.74 < 16.0–17.2), so
  loss-based selection among restarts stays valid. The margin between the partial basin (14.74) and
  the best cold optimum (14.94) is thin, though — do not over-read the ordering *within* the ≥ 14 band.
- **Along the relabeling axis the basin reaches ~60% wrong labels** (0.6: 1 of 2 seeds; 0.8 and 1.0
  fail). A fully relabeled but otherwise perfect emission does *not* recover: the criterion cannot
  undo a global permutation from a sharp start, which is the relabeling ambiguity in its local form.
- **Along the noise axis it is much wider: a start at 89.6% PER with a shuffle gap of only +2.95
  recovers fully (mix 0.8 → 23.61%).** mix 0.95 (gap +0.45) does not. So a *weak but consistent*
  signal on every phoneme is worth far more than a *strong but partly wrong* one (perm 0.8 starts at a
  better PER, 84.3, with a similar gap, +3.7, and fails).
- **What a label-free initializer has to deliver is therefore modest**: something like a +3 shuffle
  gap, spread over all phonemes, rather than a good PER. The uniform start delivers 0 (by
  construction), and the best label-free discrete maps of E.2 reached only +1.13, so neither is enough
  as it stands, but the target is within an order of magnitude of what has been measured.
