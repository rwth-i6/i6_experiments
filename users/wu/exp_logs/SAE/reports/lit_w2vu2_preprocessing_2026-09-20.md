# wav2vec-U 2.0 preprocessing: what the paper actually says

Date: 2026-09-20. Sources read in full text (not abstracts):
- **w2vu 2.0**: Liu, Hsu, Auli, Baevski, "Towards End-to-end Unsupervised Speech Recognition",
  arXiv:2204.02492**v2** (15 Jun 2022), 12 pages. PDF <https://arxiv.org/pdf/2204.02492v2>, abs page
  <https://arxiv.org/abs/2204.02492>. **Only v1 and v2 exist; both are labelled "Preprint. Under
  review."; v2 has NO appendix and NO supplementary tables** beyond Tables 1-6 in the body. Every
  page was read; all quotes below are verbatim from that text.
- **w2vu 1.0**: Baevski, Hsu, Conneau, Auli, "Unsupervised Speech Recognition", arXiv:2105.11084v3
  (35 pages incl. appendix), <https://arxiv.org/pdf/2105.11084v3>.
- Local ground truth (read-only): fairseq wheel at
  `/e/project1/spell/wu24/env/conda/envs/w2vu/lib/python3.9/site-packages/fairseq/examples/wav2vec/unsupervised/`.

---

## 1. Does 2.0 remove silence from the AUDIO with rVAD?

**Yes. 2.0 still removes silence from the waveform with rVAD, exactly as 1.0 did.** The abstract's
"does away with all audio-side pre-processing" refers only to the *segmentation / k-means / PCA /
adjacent-pooling* chain, not to silence removal.

Verbatim, 2.0 Section 4.1, "Implementation Details" (p.5):

> "For English speech from LibriSpeech, the input features are extracted from the 15th layer of a
> pre-trained wav2vec2.0 Large model [15] **encoded from the audio of which silences have been
> removed using an unsupervised voice activity detection tool [30]**. For non-English speech,
> features are extracted from the XLSR-53 [31] model, a multi-lingual version of wav2vec 2.0."

Reference [30] of 2.0 is exactly rVAD: "Zheng-Hua Tan, Achintya Kumar Sarkar, and Najim Dehak. rvad:
An unsupervised segment-based robust voice activity detection method. Computer speech & language,
59:1-21, 2020." So the tool is the same one 1.0 used.

What 2.0 removes is stated in Sec. 3.2 ("Removing Audio Pre-processing"), and silence is not in the
list:

> "Without the need of speech segmentation, a generator taking raw speech representation sequence as
> the input itself can be sufficient for unsupervised ASR with proper output frequency ... we found
> that PCA-based dimensionality reduction can be replaced by a simple batch normalization [20] along
> the time axis over the wav2vec 2.0 features."

**Ablation with/without silence removal in 2.0: NONE.** Table 1 (the interpolation table from
wav2vec-U to wav2vec-U 2.0) has columns *Adjacent pooling / Cluster pooling / PCA reduction / Batch
norm. / Linear proj. / Auxiliary loss / Stride / Freq. / Average PER*; there is no silence row, and
no other table or sentence in v2 touches it. Silence removal is held fixed ON in every 2.0 number.

The only published silence ablation is **1.0, Figure 5 (left), Section 3.4** — PER on LibriSpeech
dev-other, GAN only (no self-training), mean +/- std over **20 random seeds**:

| condition | PER |
|---|---|
| Baseline (rVAD silence removal + begin/end SIL + p=0.25 insertion) | 21.4 +/- 1.2 |
| w/o begin/end SIL tokens | 25.8 +/- 0.7 |
| **w/o audio silence removal** | **29.3 +/- 2.0** |

1.0 Sec. 3.2 verbatim:

> "Before that, we also remove silences from the speech audio using an off-the-shelf unsupervised
> method. One exception is the TIMIT benchmark, where silences are part of the transcription.
> **Silence removal is important to learning a better mapping between speech audio representations
> and transcriptions.**" ... "To remove silences, we apply rVAD, an unsupervised voice activity
> detection (VAD) model which determines the segments in the audio data corresponding to silences,
> and we remove these sections (Tan et al., 2020). We ablate this choice in Sec. 3.4."

So: +7.9 PER absolute (21.4 -> 29.3, a 37% relative degradation) from leaving silence in the audio,
at 1.0's operating point (segmented/pooled features, sil_prob 0.25, greedy/Viterbi PER on dev-other).
**No such number exists for the 2.0 architecture**, and the 2.0 operating point differs in every way
that could matter (no pooling, stride-3 conv, batch norm, MFCC aux loss, sil_prob 0.5). The
literature does not settle what silence-in-audio costs a 2.0-style generator. What would settle it:
the Table-1 final row rerun on with-silence features, 8 seeds, same dev-other greedy PER.

**How the removal is done matters and differs from our setup.** fairseq
`scripts/remove_silence.py` loads the waveform, *concatenates the rVAD speech intervals into a new
wav file* (`torch.cat([data[0][it[0]:it[1]] for it in intervals])`, saved with `torchaudio.save`),
and the README then runs `wav2vec_manifest.py` over that new directory before
`prepare_audio_v2.sh`. The wav2vec 2.0 encoder therefore **never sees the silence**: self-attention
and the conv feature extractor operate on the silence-free signal. Our setup extracts SSL features
from the full waveform and masks frames afterwards; the receptive field and the attention context
are not the same object. This is a methodological difference, not necessarily a defect, but it is
unmeasured in both papers.

## 2. Silence-token insertion probability in the TEXT

**0.5, stated in the paper**, 2.0 Section 4.1 (p.5), verbatim:

> "Unpaired text data is phonemized with an off-the-shelf phonemizer for English [24] and other
> languages with silence token padded on both sides of each sentence and inserted at word boundaries
> **with a probability of 0.5**."

(Footnote marker after "other languages" points to <https://github.com/bootphon/phonemizer>.)

**The paper gives no justification for 0.5 and does not tie it to keeping silence in the audio** —
indeed it cannot, because the same paragraph-set says the audio silence *is* removed. There is no
sweep over the rate in 2.0 and no discussion of why it doubled from 1.0's 0.25. The fairseq README
is equally bare: "The last argument is the probability to introduce silence (`<SIL>`) between the
word boundaries. We found the value `0.25`/`0.5` works in general for wav2vec-U and the 2.0 version
respectively, but you might want to vary for languages that are never tested."

1.0's corresponding rate is a measured optimum, not a guess: Sec. 3.4, "Figure 5 (right) shows that
inserting the silence token at a rate of 0.25 yields the best end accuracy", swept over
{0%, 15%, 25%, 50%, 75%} with PER between ~20 and ~26 on dev-other. Note that in 1.0's own sweep,
**0.5 was measured and was worse than 0.25** at 1.0's operating point. 2.0 moved to 0.5 without
publishing the evidence. The most likely (but unstated) reason is the change in output frequency:
1.0's generator emits ~14 Hz on pooled/segmented features, 2.0 emits 16 Hz on unpooled frames and
then merges, so the target token sequence must be longer; a higher SIL rate lengthens the real-text
side to match. That is inference, not a claim of the paper.

Also standing, from 1.0 Sec. 3.4 and 4.1 (still in force in 2.0, whose text pipeline is the same
`prepare_text.sh`): a SIL is added at the **beginning and end of every sentence** ("First, we add a
SIL token to the beginning and the end of all phonemized unlabeled text sentences"), and SIL is a
member of the output phoneme set: "The phoneme set O includes a silence label SIL to enable labeling
silences in the speech audio as such. Without a silence label, we noticed that the model was
repurposing a particular phoneme to label silences which resulted in much lower performance."

## 3. The 2.0 generator

From the paper (Sec. 3.2, 3.3, 4.1), with code-only facts marked.

- **Input features**: 15th layer of pre-trained **wav2vec 2.0 Large** (LV-60k, frozen) for English
  LibriSpeech; **XLSR-53** for non-English. 1024-dim, 50 Hz. (`prepare_audio_v2.sh` default
  `layer=14`, 0-based = the 15th layer; README passes `14`. `input_dim: 1024` in `w2vu2.yaml`.)
  No PCA, no k-means pooling, no adjacent mean-pooling.
- **Stride 3**: "With the input wav2vec 2.0 features sampled at 50 Hz, we found that outputting a
  phone candidate at around 16 Hz by using a stride of 3 worked consistently well across all 9
  languages tested in this work."
- **Kernel 9**: **not stated in the paper**. From the shipped `config/gan/w2vu2.yaml`:
  `generator_kernel: 9`, `generator_stride: 3`, `generator_bias: false`,
  `generator_dropout: 0.1`, padding = kernel//2 = 4.
- **Batch norm**: replaces PCA, "a simple batch normalization [20] along the time axis over the
  wav2vec 2.0 features". The **scale init is load-bearing**: "We found that initializing the scaling
  factor of batch normalization to **30 for wav2vec 2.0 Large and 35 for XLSR-53** features is
  critical for convergence." (`generator_batch_norm: 30`; code does
  `self.bn.weight.data.fill_(cfg.generator_batch_norm)` and applies BN only over non-padded frames.)
- **Residual**: **not mentioned in the paper**; present in code and enabled in the config
  (`generator_residual: true`). It is a linear self-residual on the *input*:
  `inter_x = in_proj(dropout(dense_x)); dense_x = dense_x + inter_x` with
  `in_proj = nn.Linear(1024, 1024)`. The paper's "Linear proj." column in Table 1 (step vii) is this
  residual linear layer: it alone bought 16.4 +/- 0.7 -> 15.9 +/- 1.1 PER.
- **"2-layer CNN"**: paper says "the generator is a 2-layer CNN with batch normalization over the
  input features". In code that is BN -> residual Linear(1024->1024) -> dropout ->
  Conv1d(1024 -> |O|, k=9, s=3, bias=False). Only one convolution.
- **Output frequency**: 50/3 ~= 16 Hz, versus "ground truth phone sequence ~10 Hz" (Table 1, last
  row). Table 1 is explicit that frequency, not the pre-processing, is what matters: steps (i) and
  (iii) at 28 Hz and 25 Hz both give "> 100" PER, i.e. total failure; matched ~14-16 Hz recovers.
- **Output vocabulary**: `output_size = len(target_dict)` — 39 phones + `<SIL>` + fairseq's 4
  specials. There is no CTC blank; `blank_index` defaults to 0 (a fairseq special) and
  `blank_weight` is 0 in `w2vu2.yaml`, so no blank is used in GAN training.
- **Auxiliary k-means loss**: cross-entropy to **k-means on MFCC with 64 clusters** (Sec. 3.3,
  Eq. 5). Head is a **Linear(1024 -> 64) applied to `inter_x`, i.e. the residual branch BEFORE the
  strided conv**, so the aux prediction is at the full 50 Hz, not 16 Hz. The paper's phrasing is
  "obtained by an additional linear transformation with the softmax function from the intermediate
  layer of the generator". MFCC km labels are dumped at 100 Hz (kaldi default 10 ms shift, via
  `examples/hubert/simple_kmeans/dump_mfcc_feature.py`) and the model subsamples them:
  `target_downsample_rate: int = 2` -> `aux_target[:, ::2]`.
- **Aux loss weight (delta)**: paper gives "1.0 / 1.5, 1.5 / 2.5, 0 / 3, **0.3 / 0.5** for lambda,
  gamma, eta, delta respectively" (two alternative settings per weight). Shipped `w2vu2.yaml`:
  `gradient_penalty: 1.0` (lambda), `smoothness_weight: 1.5` (gamma), `code_penalty: 3.0` (eta),
  `mmi_weight: 0.5` (delta). **Note eta = 0 is one of the two published options** — the phoneme
  diversity term is optional in 2.0.
  Effect of the aux loss, Table 2 (dev-other PER, greedy, mean over 8 runs): None 15.9 +/- 1.1;
  wav2vec2.0 VQ indices (320x2) 16.6 +/- 2.2 (**worse than no aux loss**); k-means on wav2vec2.0
  features 32/64/128 -> 16.4 +/- 1.4 / 15.5 +/- 1.8 / 15.9 +/- 0.9; k-means on **MFCC**
  50/64/100/128 -> 15.2 +/- 0.9 / **13.6 +/- 0.9** / 14.8 +/- 1.3 / 16.8 +/- 1.7. The negative result
  is explicit: "VQ indices perform less well than the baseline method where no pseudo labels are
  used." Note every one of these differences is within ~1 std of 15.9 except MFCC-64.
- **Merging identical consecutive outputs**: "During adversarial training, when obtaining the output
  probability distribution from the generator, consecutive frames that share the same most probable
  phone are merged into one by random selection." (Sec. 4.1; also Sec. 3.2: "consecutive outputs
  sharing the identical most possible phone can be treated as a single output by random sampling as
  done in wav2vec-U".) In code this is `segmentation.type: JOIN` with `mean_pool_join: false`,
  `remove_zeros: false` — applied to the *logits*, after the conv (`logit_segment`), not to features.
- **Decode stride at test time**: "While the generator must be trained with a stride of 3, we found
  decoding using a language model with a stride of 2 (higher output frequency) leads to a more
  accurate output." (Sec. 4.1, last paragraph). README: "While the generator of wav2vec-U 2.0 is
  trained with an output frequency of 16hz, we found decoding at a higher frequency produces better
  results. This can be done by adding `decode_stride=1` or `2`." Mechanically, `w2vu_generate.py`
  does `overrides["model"]["generator_stride"] = cfg.decode_stride` — **the same conv weights are
  re-applied at a different stride**, no retraining, no interpolation. Decoding is via PyKaldi
  ("Pykaldi [32] is used to decode the phone-based ASR outputs into words").
- **Discriminator**: "a 2-layer CNN that uses the output of the generator or a one-hot representation
  of the real phone sequence as input. A single scalar ... obtained by taking the average over output
  sequence followed by the sigmoid activation." Config: dim 384, depth 2, kernel 8, causal, no
  max-pool, no weight norm, dropout 0.
- **Schedule**: paper says "The generator and discriminator are trained for 100k steps in total
  interleaving the updates of each (50k updates for each) with a fixed learning rate of 5e-5 and
  3e-4 respectively. Each batch contains 160 audio samples and 160 randomly selected text samples."
  **The shipped config disagrees**: `optimization.max_update: 150000`, batch_size 160, generator lr
  5e-5, discriminator lr 3e-4. (1.0's paper said 150k total / 75k each, generator 1e-4,
  discriminator 1e-5 — the roles of the two learning rates are swapped between the two papers;
  the 2.0 config matches the 2.0 paper's lr assignment.)
- **Seeds / convergence**: "we run each model with 4 different random seeds and we observe that over
  80% of the models will converge during training." Table 1/2 PERs are averaged over 8 runs.

## 4. Unsupervised model selection in 2.0

The paper delegates entirely: "**Models are selected with the Unsupervised Cross-Validation Metric
proposed by Baevski et al. [6].**" (Sec. 4.1). No labeled data is used for checkpoint selection, in
either paper.

1.0's definition, Sec. 4.3, uses two quantities: phone-LM negative log-likelihood and vocabulary
usage. "We use the metric for early stopping, selecting a random seed, and hyper-parameter selection
(lambda, gamma, eta)." Vocabulary usage U(P) = fraction of the phoneme inventory appearing in the
Viterbi output; it exists to "identify degenerate models which output fluent but trivial
transcriptions". The published procedure is three steps: (a) anchor
`P_hat = argmin NLL_LM(P) - log U(P)`; (b) keep only configs with
`NLL_LM(P) < NLL_LM(P_hat) + log(U(P)/U(P_hat)) + log 1.2`; (c) among survivors take the one with the
highest **un-length-normalised** sum of log probability, "This selects model configurations which
produce phoneme sequences that score high under the language model but are not too long."
Footnote 2: "We remove SIL labels from P when computing the NLL because SIL is not used in pLM."
Footnote 3: "In practice, we used language model perplexity which is equivalent to NLL after taking
the log." Calibration, 1.0 Fig. 7 (TIMIT core-dev/test, GAN and GAN+ST): the unsupervised metric
costs "only between 0.3-1.2 higher PER compared to using a labeled development set".

**The shipped 2.0 code implements a simpler scalar, not the three-step procedure.**
`config/gan/w2vu2.yaml` sets `checkpoint.best_checkpoint_metric: weighted_lm_ppl`, and
`tasks/unpaired_audio_text.py` defines
`weighted_lm_ppl = lm_ppl / vocab_seen_pct ** vocab_usage_power` with `vocab_usage_power: float = 2`,
where `lm_ppl` is KenLM's score over the greedy hypothesis and `vocab_seen_pct` is the fraction of
the phone inventory emitted. Validation runs every 1000 updates. Two details that bite:
`if self.sil_id >= 0: x = x[x != self.sil_id]` strips `<SIL>` from the hypothesis **before both the
LM score and the vocabulary-usage count** (so SIL never counts toward vocabulary usage, and the
4-gram phone LM is queried on SIL-free strings even though it was trained on SIL-inserted text); and
`uppercase` (default False) controls case-matching for the KenLM query. UER against `valid.phn` is
computed and logged when targets are present, but it is not the selection metric — selection is
label-free.

## 5. Other preprocessing our setup could have missed

All from `scripts/prepare_text.sh` (unchanged between 1.0 and 2.0; 2.0 ships no separate text script)
plus 1.0 Sec. 5.2.

- **Text normalisation**: `normalize_and_filter_text.py` applies `regex.compile(r"[^\p{L}\p{N}\p{M}\'
  \-]")` -> replaces anything that is not a letter/number/combining-mark/apostrophe/hyphen with a
  space, collapses whitespace, and runs **fastText LID filtering** (`lid.187.bin`, default
  `--lid-threshold 0.4`) to drop lines not in the target language. Output is `lm.upper.lid.txt`
  (upper-cased). Lines containing `---` are then dropped (`grep -v '\-\-\-'`).
- **Word list filtering**: `fairseq_cli/preprocess.py --thresholdsrc 2` builds the word dict (words
  seen >= 2 times), then `cut -f1 -d' ' dict.txt | grep -v -x '[[:punct:]]*' | grep -Pv '\d\d\d\d\d+'`
  drops pure-punctuation tokens and anything with 5+ consecutive digits.
- **G2P**: for English LibriSpeech, **g2pE** (Park & Kim 2019, ref [24] of 2.0), CMUdict lookup with
  a neural fallback: 1.0 Sec. 5.2 "we use the G2P phonemizer (Park and Kim, 2019) which uses the CMU
  dictionary to look up English word pronunciations, falling back to a neural network trained to
  output a phoneme sequence given a word." For everything else (including **English in MLS**)
  `phonemizer`/espeak with `--language-switch remove-flags`; 1.0 notes espeak "is less accurate than
  G2P for English", and Sec. 7 flags that "G2P vs Phonemizer on the same English data results in a
  performance difference."
- **Phone set**: **39 phones** for LibriSpeech — "We convert the full phoneme set to a reduced set
  containing 39 phonemes by **removing the numerical stress markers from the vowels**" — plus
  `<SIL>`, appended explicitly (`echo "<SIL> 0" >> dict.phn.txt`). TIMIT is the exception: 60 -> 39
  ARPAbet inventory, silence is a real transcribed phone, and per the README "TIMIT transcripts
  include silence. Therefore VAD is not used for audio preprocessing, and we do not wrap transcripts
  with silences or insert random silence in between words."
- **Phone pruning + lexicon filtering**: the phone dict is built with `--thresholdsrc $min_phones`
  (README's example and 1.0 Sec. 5.2: "prune phonemes that appear fewer than **1000** times in the
  text corpus"), and then `filter_lexicon.py -d phones/dict.txt` **drops every lexicon entry whose
  pronunciation contains a pruned phone** before the text is phonemized. So the rare-phone pruning
  silently removes whole words from the unpaired text.
- **SIL insertion mechanics**: `phonemize_with_sil.py -s $sil_prob --surround --lexicon
  lexicon_filtered.lst` — insertion at word boundaries at rate `sil_prob` **and** surrounding SIL,
  both in one pass.
- **LM orders**: the **phone LM used for the unsupervised criterion is a KenLM 4-gram** trained on
  the SIL-inserted phonemized text (`lmplz -o 4 < phones/lm.phones.filtered.txt` ->
  `lm.phones.filtered.04.bin`; README: "KENLM_PATH=.../kenlm.phn.o4.bin # KenLM 4-gram phoneme
  language model (LM data = GAN data here)"). A **6-gram** phone LM is built for the
  `phn_to_phn_sil` decoding FST, and the **word** LM is a 4-gram with `--prune 0 0 0 3`
  (`kenlm.wrd.o40003.bin`), used for the phn->word FSTs. The `phn_to_words_sil` and `phn_to_phn_sil`
  FSTs are built with `blank_symbol='<SIL>'`.
- **Self-training**: 2.0 reuses 1.0's 3-stage pipeline verbatim (GAN -> HMM on pseudo-labels with a
  phone-to-phone WFST -> relabel, fine-tune wav2vec 2.0 with CTC on letters, letter-to-word WFST).
  2.0 observes the pipeline, not the GAN, is the bottleneck on LibriSpeech: WER 9.9/13.9
  (test-clean/other) GAN-only vs 3.7/6.3 with self-training, against 1.0's 13.8/18.0 and 3.8/6.5 —
  i.e. **the 4.0/4.1 WER advantage of 2.0 before self-training collapses to 0.1/0.2 after**
  (Table 3). "This suggested that unsupervised ASR on LibriSpeech might be bottle-necked by the
  self-training pipeline."

## Differences vs our setup (stated, not judged)

1. **rVAD placement.** Paper/fairseq cut the waveform and re-save silence-free wavs *before* SSL
   feature extraction; ours extracts from the full waveform and masks frames after. Unmeasured in
   both papers.
2. **sil_prob 0.5 with surrounding SIL** — matches 2.0 exactly (paper Sec. 4.1 + README), and 2.0
   keeps rVAD as well, so our combination (rVAD + 0.5) is the paper's combination.
3. **Stride 3, kernel 9** — matches (stride from the paper, kernel from the shipped config only;
   the paper never states a kernel size).
4. **40 outputs, no blank** — consistent: 39 phones + `<SIL>`, no CTC blank in GAN training. fairseq
   carries 4 extra fairseq specials in the logit dimension, which are filtered out at eval.
5. **Layer-15 large features** — matches ("15th layer of a pre-trained wav2vec2.0 Large model";
   `prepare_audio_v2.sh` default `layer=14`, 0-based).
6. Items the paper has that are easy to omit: **batch-norm scale initialised to 30** ("critical for
   convergence"), the **residual linear branch** (`generator_residual: true`, worth ~0.5 PER as
   Table 1 step vii), the **aux head reading the pre-conv residual branch at 50 Hz** with km targets
   downsampled 2x from 100 Hz MFCC, **eta = 0 being a sanctioned setting**, **decode stride 2 at
   test time**, the **4-gram phone LM for the selection criterion**, **SIL stripped before the LM
   score and the vocabulary-usage count**, and **min_phones=1000 pruning that also deletes lexicon
   entries**.

## What is UNVERIFIED

- Whether 2.0 also applied rVAD to the non-English (MLS/CommonVoice) audio: the sentence quoted in
  Sec. 1 is scoped to "English speech from LibriSpeech", and the paper never says either way for
  XLSR-53 inputs. The fairseq README's silence-removal step is not language-conditioned, which
  suggests yes, but the paper does not state it.
- Any 2.0-architecture measurement of the cost of leaving silence in the audio. Does not exist in
  v1/v2, and there is no appendix or supplement to check.
- The 100k-step (paper) vs 150k-step (`w2vu2.yaml`) discrepancy is unresolved; the paper's Table 1/2
  numbers cannot be attributed to one or the other from the published text.
