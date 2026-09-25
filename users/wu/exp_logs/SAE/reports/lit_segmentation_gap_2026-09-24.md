# Literature: how much unsupervised phone recognition depends on oracle segmentation, and which label-free methods close the gap

Date: 2026-09-24. Role: literature. The question comes from A17 (iii) in SAE_4A_lexlat_v2.md. A colleague says the approach "works with gold segmentation, fails without it". Every A14 (ii) arm that reached the basin carried MFA-fitted durations.

Status: DONE. Every number below was read from the paper's full text, extracted to text with pypdf. Anything that was not read that way is marked UNVERIFIED.

## Verdict

1. **Segmentation carries a large share of reported success. This holds across groups.** In systems that match segment-level (type-level) distributions, oracle boundaries beat unsupervised boundaries at the first mapping iteration:
   - by 14 to 22 PER points in Chen 2019 and Yeh 2019 (TIMIT, MFCC features);
   - by 5.5 points in REBORN, when only the boundaries are swapped under the same predictor (LibriSpeech, wav2vec 2.0).

   Iterative or joint resegmentation that uses the text prior shrinks the gap:
   - to about 9 to 10 points in the MFCC era (Chen, Yeh);
   - to about 2.5 points with wav2vec 2.0 features (REBORN);
   - to about 0.4 to 0.7 CER points at the syllable level (SylCipher, LibriSpeech).

   The one text-free joint model read here does not close it: Ondel 2016's VB acoustic unit discovery (AUD) drops from 47% to 36% of phone information learnt when boundaries are no longer known.
2. **What predicts failure is the output rate, not boundary F1.** This is shown by at least two groups:
   - Segment-level matching fails completely (PER > 100, 0% converged) when the segment stream runs at about 2.4x the phone rate (25 to 28.5 Hz against about 10 to 12 Hz). It works at 14 to 16 Hz.
   - Better boundary F1 at the correct rate did not give better PER in REBORN, nor in the ESPUM/wav2vec-U comparison.
3. **Run-collapse of k-means/VQ units is a poor proxy segmentation.**
   - Over-segmentation is 164% to 233%: the stream has 2.6x to 3.3x the true number of boundaries (two Kamper papers, Buckeye).
   - REBORN measures 28.5 Hz on LibriSpeech.
   - Even a gold-derived unit-to-phone assignment on run-collapsed HuBERT units gives PER of about 97% to 127%, dominated by insertions (DiscoPhon, single paper, preprint).
   - Published decipherment of speech (Klejch 2022) runs on phone-rate symbol streams and allows only one insertion or deletion in a row.
4. **Nothing read tests our case.** No paper compares oracle, general-knowledge and label-free-learnt durations inside an EM-trained HSMM over discrete SSL units with a frozen phone LM. The literature does not settle whether our basin needs MFA durations. A17 (iii) settles it.

## Findings by source, with conditions

### Topic 1: oracle-vs-unsupervised PER gaps

**Liu et al. 2018, Interspeech. Verified from https://arxiv.org/pdf/1804.00316**
- Setup:
  - TIMIT, with oracle phone boundaries only; silence is also removed by oracle.
  - MFCC seq2seq segment embeddings, clustered to K; a GAN learns a cluster-to-phone lookup table, which is a type-level key.
- Results:
  - Accuracy 36.05% with unrelated text (ensemble) and 50.33% with matched text.
  - The upper bound is cluster purity: 59.02% at K = 1000.
  - For K ≥ 500 (about 10 or more clusters per phone), the mapping was hard to learn with unrelated text.
- There is no unsupervised-boundary row. This is the origin of the oracle-segmentation dependence.

**Chen et al. 2019, Interspeech. Verified from https://arxiv.org/pdf/1904.04100**

TIMIT, MFCC-based, PER for matched / nonmatched text:

| Setting | Matched | Nonmatched |
|---|---|---|
| Oracle-boundary GAN | 28.5 | 34.3 |
| Unsupervised GAS boundaries, GAN, iteration 1 | 48.6 | 50.0 |
| GAN at iteration 3, after HMM resegmentation | 38.4 | 44.2 |
| GAN/HMM at iteration 3 | 26.1 | 33.1 |

- The gap at the mapping stage is 20.1 / 15.7 points at iteration 1 and 9.9 / 9.9 after two resegmentations.
- 26.1 is below the oracle GAN's 28.5, but that comparison is not like-for-like: there is no oracle + HMM row.
- Caveat from wav2vec-U (a citation of a personal communication; UNVERIFIED in Chen's own text): Chen's cross-validation used labelled data.

**Yeh et al. 2019, ICLR. GAN-free (segmental empirical ODM). Verified from https://arxiv.org/pdf/1812.09323**

TIMIT, MFCC, Kaldi-decoded PER for matched / nonmatched LM:

| Setting | Matched | Nonmatched |
|---|---|---|
| Oracle boundaries | 32.5 | 40.1 |
| Iteration 1: GRNN boundaries (F 80.1, R-val 82.6, lenient) | 47.0 | 61.7 |
| Iteration 2: MAP-refined boundaries (F 82.6, R-val 84.8) | 42.6 | 49.1 |
| Iteration 2 + HMM tri+SAT | 36.5 | 41.6 |

- The gap is 14.5 / 21.6 points at iteration 1 and 10.1 / 9.0 at iteration 2.
- The authors: "With unsupervised boundary estimation, we still see a big performance loss. Therefore, it is critical to improve the boundary estimation."
- They also report that setting the smoothness weight lambda = 0 "degrades significantly".

**wav2vec-U (Baevski et al.), NeurIPS 2021. Verified from https://arxiv.org/pdf/2105.11084**
- TIMIT, 20 ms tolerance. k-means-128 run segmentation has P .935, R .379, F1 .539. The GAN's own Viterbi output has F1 .629.
- DISAGREEMENT, internal: the text calls the k-means segmentation "very granular". High precision with low recall instead implies under-segmentation. REBORN measures the same k-means-only segmentation at 28.5 Hz on LibriSpeech, which is over-segmentation. I cannot resolve this from the paper.
- Table 7 (TIMIT all-test PER): "+HMM resegment + GAN" gives 13.8, against 13.5 for plain HMM self-training.
- The text: "We also explored HMMs to refine segmentation boundaries but did not find it helpful in our setting." This is a negative result for resegmentation in a continuous-feature GAN.
- There is no oracle-boundary row.

**wav2vec-U 2.0 (Liu et al.), SLT 2022. Verified from https://arxiv.org/pdf/2204.02492**

Table 1: LibriSpeech dev-other, greedy decoding, mean of 8 runs.

| Output rate | PER |
|---|---|
| 28 Hz | > 100 |
| 25 Hz | > 100 |
| 14 Hz (stride 2) | 18.5 |
| 16 Hz (stride 3), no segmentation pre-processing | 19.0 |
| 16 Hz without batch norm | > 100 |
| Full wav2vec-U 2.0 | 13.6 |

- The paper gives the ground-truth phone rate as about 10 Hz. Its conclusion: "having output frequency close to ground truth (around 10 Hz) is important", and segmentation pre-processing is not necessary.
- The batch-norm row shows that the correct rate is necessary, not sufficient.

**REBORN (Tseng et al.), NeurIPS 2024. Verified from https://arxiv.org/pdf/2402.03988**. The oracle is forced alignment.
- Table 1, LibriSpeech 100 h, test-clean PER:

  | System | PER |
  |---|---|
  | Oracle-boundary GAN | 6.4 |
  | wav2vec-U (reproduced) | 19.3 |
  | wav2vec-U 2.0 | 12.6 |
  | REBORN | 8.9 |
  | REBORN + HMM ST | 5.4 |

- Table 2, TIMIT all-test, greedy: oracle 9.2, wav2vec-U 20.3, REBORN 12.4.
- Table 4 swaps the boundaries under one fixed predictor trained on k-means segments (LibriSpeech test-clean, strict scoring):

  | Boundaries | F1 | Rate (Hz) | PER |
  |---|---|---|---|
  | Oracle | 1.0 | 11.89 | 13.8 |
  | k-means + adjacent pooling | .70 | 14.27 | 19.3 |
  | Strgar & Harwath | .75 | 12.26 | 23.2 |
  | REBORN | .65 | 16.29 | 12.9 |

  - The better-F1 SSL segmenter, at nearly the correct rate, gave the worst PER.
  - Confound: the predictor was trained on k-means segments.
- Table 7, 5 runs each:
  - k-means-only at 28.5 Hz: PER > 100, 0% converged. The authors attribute this to "length mismatch".
  - k-means + adjacent pooling at 14.3 Hz: 20.1 ± 0.6.
  - Without boundary merging: 13.9 ± 2.1; with merging: 12.0 ± 0.1.
  - Oracle: 6.64 ± 0.2.
- Moderate over-segmentation (about 1.4x) is tolerated when adjacent identical predictions are deduplicated.

**SylCipher (Wang et al.), arXiv 2608.22907 (2026). Non-adversarial, syllable level. Verified from https://arxiv.org/pdf/2608.22907**
- LibriSpeech CER, matched / unmatched: forced-align boundaries 38.5 / 46.4; Sylber 43.5 / 48.6; Sylber + JE2E (joint refinement using unpaired text) 39.2 / 46.8.
- AISHELL-3 PER without / with tone: forced-align 38.1 / 38.9; Sylber 44.6 / 48.3; Sylber + JE2E 41.7 / 45.1.
- "SylBoost often over-segments, which hurts cross-modal alignment and destabilizes SylCipher, whereas Sylber predicts syllable counts closer to ground truth and is therefore a better initializer." A third group finds the init's rate matters.

**ESPUM (Wang, Hasegawa-Johnson, Yoo), arXiv 2310.02382. GAN-free. Verified from https://arxiv.org/pdf/2310.02382**
- TIMIT test PER, matched / unmatched: iteration 1 gives 39.1 / 45.1; with HMM ST, 33.7 / 42.9.
- Its own segmentation reaches harsh F1 86.2 (matched, iteration 1). wav2vec-U + HMM ST has harsh F1 71.0, yet a far lower PER (12.0 matched).
- Table 3: at a nearly fixed boundary F1 of 86.0 to 87.6, validation PER ranges from 38.4 to 77.9 depending on the n-gram set.
- Boundary F1 does not predict PER. There is no oracle-boundary row.

**Wang et al. 2023, "A Theory of Unsupervised Speech Recognition", arXiv 2306.07926. Verified from https://arxiv.org/pdf/2306.07926**
- Limitations section: the theory "assumes that sufficiently reliable phoneme boundaries are fed to the ASR-U system, and kept fixed during training."

**Yang et al. 2026, RWTH, arXiv 2603.02285. Verified from https://arxiv.org/pdf/2603.02285**
- The theory ignores alignment and assumes equal lengths.
- Neither theory paper covers segmentation.

### Topic 2: label-free segmenter quality, and whether joint resegmentation recovers the gap

Boundary quality, 20 ms tolerance:

| Source | Setting | F1 | Other |
|---|---|---|---|
| Kreuk et al. 2020, Interspeech (arXiv 2007.13465) | CPC peak detection, TIMIT (lenient) | 83.71 | R-val 86.02 |
| Kreuk et al. 2020 | Buckeye | 76.31 | |
| Strgar & Harwath 2022, SLT (arXiv 2211.01461) | HuBERT readout trained on Kreuk pseudo-labels, TIMIT | 89.71 lenient, 81.81 strict | |
| Strgar & Harwath 2022 | same, Buckeye | 84.01 lenient, 77.28 strict | |
| Strgar & Harwath 2022 | supervised readout, TIMIT | 96.2 lenient | about 93 strict |
| Yang & Tang 2024 (arXiv 2409.09646) | HuBERT peak detection, TIMIT, strict | 62.3 | |
| Yang & Tang 2024 | joint HMM-DP (centroids learnt jointly with a duration-constrained segmentation) | 75.5 | |
| Yang & Tang 2024 | joint HMM-DP-BF | 82.1 | |
| Yang & Tang 2024 | two-stage VQ-DP | 69.3 | |
| Lee & Glass 2012, ACL (P12-1005) | joint Dirichlet-process HMM, TIMIT train | 76.3 | |
| Lee & Glass 2012 | heuristic pre-segmentation | 64.0 | P 87.0, R 50.6 |
| Ondel et al. 2019 (arXiv 1904.03876) | Bayesian HMM AUD, TIMIT, MFCC, no cross-lingual prior | 61.84 | eq. PER 64.92 |

Notes on these rows:
- Kreuk: the peak threshold delta is tuned by cross-validation.
- Strgar: lenient scoring inflates F1 by 5 to 8 points.
- Yang & Tang:
  - Strict scoring.
  - Joint beats two-stage by 4 to 6 points without boundary features.
  - Phone purity rises from 47.3 (k-means) to 51.6 (joint HMM), with K = 50.
  - Average duration L = 8.1 frames (81 ms), lambda and gamma were all tuned on a validation split with labels.
- Lee & Glass: the pre-segmentation "allows the model to capture proper phone durations, which compensates the fact that we do not include any explicit duration modeling". The pre-segmentation's P 87.0 caps recall at about 87%.
- Ondel 2019: its better SHMM rows use a phonetic subspace trained on transcribed other languages, which is not label-free for us.

**Ondel et al. 2016, SLTU (Procedia CS 81). Verified from https://www.fit.vut.cz/research/group/speech/public/publi/2016/ondel_sltu2016_17-8037.pdf**

TIMIT, MFCC, VB non-parametric HMM AUD. R is mutual information normalised by phone entropy.

| Setting | R | Units |
|---|---|---|
| Known boundaries | 47% | 85 |
| Unknown boundaries, joint | 36% | 197 |

- Joint segmentation without text loses a quarter of the information and more than doubles the unit count. This is a negative result.

**Lee, O'Donnell & Glass 2015, TACL. Verified from https://aclanthology.org/Q15-1028.pdf**
- Joint word and phone models with a noisy channel (substitute, split, delete) improve word segmentation F1 over no-channel ablations. Examples: FullDP 16.1 to 20.0 against -NC 13.8 to 16.2.
- When the initial segmentation is at a local optimum, the joint model improves it.

Answer to topic 2:
- Joint or iterative resegmentation recovers most of the gap **when the text or LM enters the boundary decision**. Five sources support this: Chen, Yeh, REBORN, SylCipher JE2E and ESPUM relabelling.
- It does not recover the gap without text: Ondel 2016.
- One group found it not helpful beyond self-training: wav2vec-U.
- No label-free segmenter reaches oracle quality. The best is about 82 strict F1 on TIMIT, with hyperparameters tuned on labels.

### Topic 3: decipherment and type-level keys under over-segmentation; run-collapse as a proxy

**Kamper & van Niekerk 2021, Interspeech. Verified from https://arxiv.org/pdf/2012.07551**
- Buckeye dev, 512-code VQ.
- Merging runs ("Merged") gives OS 232.6% (VQ-CPC) and 207.2% (VQ-VAE), with F 45.7 and 48.3.
- DP with a duration penalty gives F 63.8 / 70.8 and OS 34.0 / 14.1.
- VQ-VAE with the DP penalty reaches F 77.5 on Buckeye test.

**Kamper 2023, TASLP (DPDP). Verified from https://arxiv.org/pdf/2202.11929**
- Buckeye test.
- Merged CPC + k-means-50 (no duration penalty) gives OS 164.5%, F1 53.5, R-val −40.5.
- DPDP with the same units gives OS 6.2%, F1 75.4, R-val 78.3.
- DPDP is a fixed duration penalty on segments, in effect an HSMM with a fixed linear duration score.
- lambda was tuned on labelled Buckeye dev (λ = 2 to 400, depending on the codebook).

**DiscoPhon (Poli et al.), arXiv 2603.18612 (2026). Verified from https://arxiv.org/pdf/2603.18612**
- Setup: 12 languages, 10 h each. Units are mapped to phones frame by frame through an assignment derived from **gold** alignments, which is an oracle type-level key. PER is taken on the resulting string.
- Many-to-one, 256 units:
  - HuBERT PER 97% to 127%;
  - SpidR 60% to 85%.
- One-to-one: PER 108% to 273%.
- Insertions are 58% (zero-shot) and 69% (finetuned) of SpidR VP-20 errors. Finetuning "has little effect on insertions".
- The paper attributes HuBERT's weaker scores to over-segmentation, "producing unit spans with spurious insertions".
- This is a single paper and a preprint, with conditions unlike ours (multilingual, 256 units, not LibriSpeech).

**Klejch et al. 2022, Interspeech. Verified from https://arxiv.org/pdf/2111.06799**
- Decipherment in a noisy-channel model (Nuhn's parameterisation) runs on output from a universal phone recogniser, a supervised cross-lingual model, so the input is already at phone rate.
- The alignment model allows "one insertion or one deletion in a row".

**Liu 2018** (above): a type-level key over K ≥ 500 clusters was hard to learn with unrelated text, even on oracle segments.

Answer to topic 3:
- Run-collapse over-segments by a factor of 1.6 to 3.3 (two groups).
- Segment-level mapping fails at about 2.4x the phone rate (two groups).
- An oracle key on run-collapsed units is insertion-dominated (one paper).
- I found no published decipherment or EM key search that tolerates 2x to 3x over-segmentation. Published keys assume near-1:1 token alignment. Run-collapse is **not** a reasonable proxy segmentation for a type-level key unless a duration penalty or a merge step follows it (DPDP; REBORN's merge).

### Topic 4: which label-free replacement for supervised durations is best supported

1. **Joint resegmentation inside the objective, with the text prior in the loop.** Best supported, across 5 or more groups (sources under Topic 2). Our phi already does this: HSMM segmentation inside EM under a frozen trigram. The literature supports keeping it, but gives no evidence on whether it suffices with a general-knowledge duration law.
2. **A rate-matched duration prior.** Well supported for rate (wav2vec-U 2.0; REBORN Table 7; SylCipher's note on the initialiser; DPDP's effect on over-segmentation). Two limits:
   - Every published duration weight or average duration (Kamper lambda, Yang & Tang's L = 81 ms, Kreuk's delta) was tuned on labelled dev data. Our durinit takes its rate from the label-free rho, which is stricter than anything published.
   - No paper isolates per-phone duration *shape* (MFA-fitted per-type laws) against a single rate-matched law. That is exactly A17 (iii)'s G-dur contrast.
3. **A self-supervised segmenter as init.** Weakest direct support:
   - REBORN Table 4 (Strgar boundaries give the worst PER under a fixed predictor, confounded);
   - wav2vec-U found HMM resegmentation unhelpful.
   - Iterative systems do use one as a starting point (Yeh GRNN, Chen GAS, SylCipher Sylber, ESPUM Strgar-teacher), always followed by joint refinement. No paper shows an init alone closing the gap.
4. **The key's own run segmentation.** Least supported: see Topic 3.

## Established versus single-paper

- **Established (3 or more groups):**
  - Oracle boundaries beat unsupervised boundaries at the first iteration (Chen, Yeh, REBORN, SylCipher, Ondel).
  - Text-informed iterative resegmentation narrows the gap (Chen, Yeh, REBORN, SylCipher, ESPUM).
  - The output rate must be near the phone rate (wav2vec-U 2.0, REBORN, SylCipher).
- **Two groups:**
  - Run-collapse over-segments heavily (Kamper twice; REBORN's 28.5 Hz). Contradicted by wav2vec-U's own Table 1 P/R.
  - Boundary F1 does not predict PER (REBORN, ESPUM).
- **Single paper:**
  - Joint HMM beats two-stage VQ for segmentation (Yang & Tang).
  - Text-free joint AUD loses against known boundaries (Ondel 2016).
  - An oracle key on run units is insertion-dominated (DiscoPhon).
  - HMM resegmentation not helpful (wav2vec-U).

## Disagreements

- Chen 2019 and Yeh 2019 find HMM or MAP resegmentation large; wav2vec-U finds it not helpful at 13.8 against 13.5. Features and baseline strength differ: MFCC against wav2vec 2.0, with a 26 to 48 PER baseline against 17.
- wav2vec-U describes its k-means boundaries as granular, but its P/R implies under-segmentation, while REBORN measures 28.5 Hz.
- REBORN Table 4 shows an SSL segmenter hurting relative to k-means boundaries. ESPUM and SylCipher use SSL segmenters as teachers and inits successfully after joint refinement.

## Label-leak caveats in the "unsupervised" numbers

These hyperparameters were set with labels:
- Kreuk's delta (cross-validation);
- Strgar's readout (trained on pseudo-labels from Kreuk, whose delta was label-tuned);
- Kamper's lambda (labelled Buckeye dev);
- Yang & Tang's L, lambda and gamma (validation split with labels);
- Chen's cross-validation (per wav2vec-U; UNVERIFIED in Chen).

Throughout, the "oracle" is forced alignment (REBORN) or TIMIT hand labels.

## Comparability with our setup

Every result above is for one of:
- continuous features (MFCC, wav2vec 2.0) with a GAN or distribution-matching mapper;
- Bayesian AUD without text;
- segmentation-only evaluation.

None uses an EM-trained HSMM over 500 discrete HuBERT units with a frozen phone trigram and held-out likelihood S. Only the qualitative conclusions carry over:
- segmentation dependence is real;
- the rate near the phone rate matters most;
- joint refinement under the text prior is what closes the gap;
- run-collapse is a poor proxy.

The PER magnitudes do not transfer.

## What this changes

- **A17 (iii)'s design and gate stand.** The literature makes SEGMENTATION-CARRIED a plausible outcome but cannot predict it. No paper tests MFA per-phone durations against one rate-matched law in a joint HSMM.
- **Cheap addition to the reported-beside boundary read:**
  - report each phi's Viterbi segment rate (segments per second) against the MFA rate on the 260 set, plus OS and R-value;
  - use strict scoring.

  The rate and OS are what predicted failure in the literature; F1 did not.
- **Fallback ordering if SEGMENTATION-CARRIED or PARTIAL-LABELS NEED SEGMENTS:**
  - The registration's named fallback "the key's own run segmentation" is the least supported option (run-collapse over-segments 1.6 to 3.3x; an oracle key on run units is insertion-dominated).
  - A self-supervised phone segmenter is better supported, but only as an init to joint resegmentation under the trigram, never alone. Its published hyperparameters were label-tuned, so a label-free re-tuning would be needed.
  - A16 (b) stage 0's gold-key floor under run segmentation should be read with DiscoPhon's result in mind. A poor floor there is expected from insertions alone, and would not show the key is wrong.

## What would settle it

- A17 (iii) as registered: G-dur, r30-dur and r70-dur against 3.289 on the 260 set.
- If SEGMENTATION-CARRIED, one further same-bed contrast would separate duration *shape* from segmentation *search*: gold emissions with a per-type duration law whose mean is set from general phonetic knowledge (not MFA), against G-dur. The literature has no such contrast.

## Not reached or not cited

- Chen 2019's use of labelled cross-validation: stated only by wav2vec-U.
- HSMM/segmental EM work with explicit duration models for AUD beyond Ondel and Lee & Glass was not read; nothing is claimed about it.
