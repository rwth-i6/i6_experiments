# Back-translation as an AUXILIARY term inside a cycle-consistency / EM-style objective

Literature review, 2026-09-15. Extends (does not repeat) `reports/lit_backtranslation_cold_start_2026-09-15.md`,
which asked whether BT can IGNITE a cold cross-modal map (answer there: no, it is a refiner).
This report asks the narrower engineering question the project owner's directive raises: given that BT
is added as an auxiliary term on theta inside an existing objective, what do the papers that actually
did this report about (a) weight/schedule, (b) whose reverse model, (c) sampling vs argmax,
(d) cold/weak reverse model failure, (e) the measured gain over the main objective alone.

All numbers below were read from the full text (arXiv PDF -> text locally), never an abstract.
Converted texts: `/tmp/claude-34349/.../scratchpad/bt/*.txt` (session-scoped, will be purged).

**Terminology fixed for this report.** Two directions get confused in this literature:
- **SO / "cycle" / ASR->TTS(TTE)**: unpaired SPEECH; recognizer emits text, reverse model reconstructs
  the speech side; loss back-propagates into theta through a sampled discrete text. This is the analogue
  of our EMC main objective.
- **TO / "BT" / TTS->ASR**: unpaired TEXT; reverse model synthesizes the speech/unit side, recognizer is
  trained with plain cross-entropy/CTC against the real text. **This is the auxiliary term we mean.**
  It is non-differentiable through the reverse model by construction and by choice: no paper reached
  back-propagates the BT term into the reverse model.

---

## 1. The papers that actually combine a cycle/main term with a BT auxiliary

### 1a. Hori, Astudillo, Hayashi, Zhang, Watanabe, Le Roux, ICASSP 2019, "Cycle-consistency training for end-to-end speech recognition"
https://arxiv.org/abs/1811.01690 (full text read, v2)

This is the **main-objective-only** reference point; it has NO BT term (its BT counterpart is 1b, its own
prior work). Its value here is three-fold.

- **(b) The reverse model is SEPARATE and FROZEN.** The Tacotron2-based text-to-encoder (TTE) is trained
  on the 100 h paired set (Eq. 18, five-term MSE+L1+BCE loss) and then held fixed; theta alone is updated
  by the cycle loss. Verbatim from the conclusion: *"Future work includes joint training of ASR and TTE
  model using both sides of the cycle-consistency loss."* So in the founding cycle-consistency ASR paper,
  joint training of the reverse model was **not done and was named as future work**.
- **(a) Schedule.** No lambda. Verbatim: *"During training, we also used the 100-hour paired data to
  regularize the model parameters in a teacher-forcing manner, i.e., the parameters were updated
  **alternately** by cross-entropy loss with paired data and the cycle-consistency loss with unpaired
  data."* Effective weight = 1 by alternation. Model selection was by validation accuracy at **epoch 6**
  (*"the validation accuracy did not improve smoothly"*) - an early-stopped, non-monotone curve.
- **(c) Sampling.** N = 5 sequences drawn from the ASR softmax per utterance (character-by-character
  until `<eos>`), REINFORCE gradient (Eq. 20) with baseline B = the mean reward over the N samples.
  argmax is explicitly ruled out as non-differentiable.

**(e) Numbers.** LibriSpeech, 100 h paired init + 360 h audio-only, attention encoder-decoder, beam 20,
no lexicon/word LM. Test CER/WER %:

| | valid | eval |
|---|---|---|
| Baseline (100 h paired) | 11.2 / 24.9 | 11.1 / 25.2 |
| + cycle-consistency loss | 9.5 / 21.5 | 9.4 / 21.5 |
| + CE loss on 1-best hypothesis (weight 0.1) | 47.8 / 86.8 | 48.8 / 89.3 |
| + CE loss on the same 5 samples (weight 0.1) | 13.3 / 28.2 | 12.3 / 27.7 |
| Oracle (460 h paired) | 4.7 / 11.4 | 4.6 / 11.8 |

**This is the strongest negative result for us in the whole review.** The *same samples*, used as
cross-entropy targets for theta (i.e. self-training / pseudo-labelling on the recognizer's own decodes,
which is what our failed P-BT probe did in spirit), **destroy the model: 25.2 -> 89.3 WER on 1-best, and
still 27.7 (worse than baseline) with 5 samples, even down-weighted by 0.1**. The identical samples,
*weighted by the reverse model's reconstruction loss*, give 21.5. The reverse model is not a data source;
it is a **scorer**. A down-weight of 0.1 was not enough to make the data-source route safe.

With character LM shallow fusion: baseline 22.9 -> cycle 19.5 WER (eval).

### 1b. Hayashi, Watanabe, Zhang, Toda, Hori, Astudillo, Takeda, SLT 2018, "Back-translation-style data augmentation for end-to-end ASR"
https://arxiv.org/abs/1807.10893 (full text read)

The pure BT term, same group, same corpus/architecture/baseline as 1a - so the two are directly comparable.
TTE (text -> ASR encoder states) trained on 100 h paired; unpaired text (the 360 h set's transcripts) is
converted to synthetic encoder states; **only the ASR decoder is retrained** (the encoder is bypassed
because the synthetic data lives at the encoder output).

**(a) Weight/schedule.** No lambda either: the paired and unpaired *datasets are concatenated* and
retraining runs 15 epochs (vs 30 for initial training), AdaDelta lr 1.0, batch 50. Mixture ratio is
data-proportional (360 h of text : 100 h of speech).

**(c) Sampling.** Generation is the TTE's own auto-regressive decode (a deterministic point estimate) with
**dropout kept ON at the Prenet during generation** (as in Tacotron2) and an end-of-sequence threshold of
0.75. So: argmax-style generation, with the only stochasticity injected as generator-internal dropout.

**(d/e) The decisive measured result - the auxiliary MUST be mixed with real data.** CER/WER %:

| | valid | eval |
|---|---|---|
| Baseline (100 h) | 11.2 / 24.9 | 11.1 / 25.2 |
| Retrain-State (BT states + real states only) | 12.0 / 27.6 | 12.4 / **28.3** |
| Retrain-State-Frozen (same, attention frozen) | 11.9 / 27.1 | 11.9 / 27.6 |
| **Retrain-Joint (BT states for unpaired text + real ACOUSTIC features for paired)** | 10.3 / 23.5 | 10.3 / **23.6** |
| Oracle-State (460 h of REAL extracted states) | 7.6 / 16.8 | 8.7 / 18.4 |
| Oracle-Feature (460 h real features) | 4.7 / 11.4 | 4.6 / 11.8 |

Training theta on the BT stream **without a co-present real-audio stream is worse than not doing it at
all** (28.3 vs 25.2); the only configuration that wins is the mixture (23.6). Authors' mechanism, verbatim:
*"Using both the generated hidden states and the original acoustic features produces a regularization
effect which prevents overfitting to the generated states"* and *"using only the hidden states resulted
in overfitting."*

Second measured caveat: **Oracle-State (18.4) is 6.6 WER worse than Oracle-Feature (11.8) using entirely
REAL encoder states.** So a large part of the BT gap is not synthesis quality at all - it is that the
encoder-state domain carries less usable variability. Their diagnosis: *"the use of data of various
speakers is more important than the use of various text ... there is not enough speaker variation among
the generated hidden states."*

### 1c. Baskar, Watanabe, Astudillo, Hori, Burget, Cernocky, Interspeech 2019, "Semi-supervised sequence-to-sequence ASR using unpaired speech and text"
https://arxiv.org/abs/1905.01152 (full text read)

**The paper that literally writes the combined objective.** Eq. 13:

    L_both = alpha * L_{ASR->TTS} + (1 - alpha) * L_{TTS->ASR},   alpha = 0.5 by default

- **(a) Weight.** alpha = 0.5, i.e. the BT auxiliary and the cycle main term carry **equal weight**, set as a
  default and (in the text read) not swept. **Schedule:** *"Unsupervised training was performed after
  conventional supervised training of each model"* and *"A small amount of paired data was also used to
  regularize the model during the unsupervised stage."* Both directions therefore start warm.
- **(b) Reverse model: separate, pre-trained on paired data, and NOT updated by the BT term.** Verbatim:
  *"Since our target in this work is to increase ASR performance, the resulting computation graph is
  simpler than in the ASR->TTS case. There is no need to backpropagate into the TTS module and thus the
  chain can be realized by forming a non-differentiable TTS->ASR pipeline."*
- **(c) SAMPLING vs ARGMAX is split by direction, explicitly.** The cycle term samples (N = 5, policy
  gradient, Eq. 10, baseline = mean over samples). The **BT term uses ARGMAX** (Eq. 12:
  `X_hat = argmax_X p_TTS(X|Y)`), *"a point estimate"*, with the diversity injected instead through
  **randomly chosen x-vectors** (speaker embeddings). So: deterministic content, randomised nuisance
  variable.

**(e) Numbers.** LibriSpeech 100 h paired + 360 h unpaired; supervised 100 h baseline = 21.0 WER
(test-clean), oracle 11.8. All "this work" rows at alpha = 0.5:

| unpaired data used | no RNNLM | with RNNLM |
|---|---|---|
| Text only (BT alone) | 17.9 | 17.0 |
| Speech only (cycle alone) | - | 16.8 |
| **Both (Eq. 13)** | 17.5 | **16.6** |

**The BT auxiliary on top of the cycle objective is worth 16.8 -> 16.6 WER, i.e. 0.2 absolute (~1 %
relative), on LibriSpeech.** Authors' own words: *"Jointly training unpaired speech and text provided
modest gains."* Their comparison row for Hayashi's BT-only method (1b), re-run in their setup, is
10.0 CER / 22.0 WER vs their 8.0 / 17.9.

WSJ (eval92), where the seed is much weaker, the auxiliary matters much more. %WER:

| unpaired | type | 2 h paired | 5 h | 10 h | 14 h |
|---|---|---|---|---|---|
| 14 h | speech (cycle) | 49.8 | 39.9 | 29.8 | - |
| 14 h | text (BT) | 63.0 | 43.6 | 34.6 | - |
| 14 h | both | **43.7** | **35.5** | **28.3** | - |
| 67 h | speech | 51.9 | 38.8 | 28.4 | 28.0 |
| 67 h | text | **39.6** | 36.8 | 29.6 | 27.1 |
| 67 h | both | 41.4 | **34.2** | **27.7** | **26.2** |

Paired-only baselines for reference: 2 h = 68.2, 5 h = 41.5, 10 h = 33.7, 14 h = 31.5 WER.
Two regime findings: (i) the BT auxiliary buys 1-6 WER absolute over the cycle term alone in every
column except 2 h/67 h, where BT *alone* (39.6) beats the combination (41.4); (ii) *"Overfitting is
observed in the case of unpaired speech"* - at 2 h paired, going from 14 h to 67 h of unpaired speech
made the cycle term **worse** (49.8 -> 51.9). More unpaired data through a weak recognizer is not free.

### 1d. Baskar, Burget, Watanabe, Astudillo, Cernocky, ICASSP 2021, "EAT: Enhanced ASR-TTS for self-supervised speech recognition"
https://arxiv.org/abs/2104.07474 (full text read)

**The single most relevant paper for our cold-phi problem.** It is the sequel to 1c and is entirely about
the failure modes of the two terms. Final objective: `L_ST = L_SO + L_TO` - note the **weights revert to
1 and 1** (the alpha of 1c is repurposed to mean something else entirely, see below).

- **(d) COLD/MISMATCHED REVERSE MODEL: the named, measured failure.** Verbatim on BABEL-Pashto:
  *"Experiments on Pashto using our previous model did not lead to gains and hence are not included in
  this paper. The reason behind the difficulty is that building a multi-speaker TTS model for Pashto is
  harder and hence our previous work failed to provide reasonable TTS scores."* I.e. **a bad reverse model
  turns the whole ASR-TTS objective into a no-op.** Their diagnosis of *why* the BT direction is hit
  harder than the cycle direction is precise and transfers to us verbatim: *"The primary reason behind
  this is that the ground truth is available in ASR->TTS to perform teacher-forcing. Whereas in
  TTS->ASR, this is not available for TTS and thus the reconstructed output deviates from the ground
  truth. Even the segments of speech and silence are wrongly predicted in the TTS->ASR pipeline."*
  (Fig. 2 shows the free-running synthetic fbanks next to the teacher-forced ones and the ground truth.)
- **THE COLD-START GUARD, with a number.** They scale the attention context vector fed to the ASR decoder
  in the BT branch only, by a hyper-parameter alpha (Eq. 6): `c_l = alpha * sum_t a_lt h_t`. Verbatim:
  *"If alpha = 0, no encoder features are used and the ASR model just behaves as a language model. This
  prevents the erroneous TTS generated features to provide a misleading signal, while still allowing to
  backpropagate into the ASR decoder. The value of alpha is chosen heuristically based on the difference
  in domains between data used to train TTS and the TO data."*
  **alpha = 0.7 for LibriSpeech (mild mismatch), alpha = 0.3 for Pashto (severe mismatch).**
- **(a) Schedule - "data annealing".** Instead of a loss weight they anneal *which supervised samples are
  released*, gated on model confidence: release when `p_ASR(y_hat|x) > gamma_t`, with
  `gamma_t = eta_t (1 - 1/K) + eta_t/K`, and eta_t following a log
  (`1 - exp(5t/T)`), linear, or exp (`exp(5(t/T - 1))`) schedule. Their reading, verbatim:
  *"linear and exp schedules are better over log, as release of supervision is initially high and reduces
  at the end of training. The performance of EAT trained using exp schedule outperforms linear as the
  supervised data is mostly released at the final stage of training paving smoother way for training with
  unsupervision."* (Note the internal inconsistency in that sentence - the tables are the evidence.)
  Table 1, 360-ST, %WER: log 7.7 / 23.5 / 6.9 / 24.3; linear 7.1 / 22.7 / 6.9 / 23.6;
  **exp 6.9 / 22.5 / 6.9 / 22.1** (dev-clean / dev-other / test-clean / test-other).
  Motivation, verbatim: *"alternating between large amounts of unsupervised data and little supervised
  data is difficult. The supervised training of certain labels can result in over-fitting, which hinders
  the effect of unsupervised training"* - with a worked example where the cycle training corrupts a word
  the supervised baseline got right ("general" -> "deneral").
- **(b) Reverse model: pre-trained separately, then UPDATED by the cycle term, then reused by the BT term.**
  Verbatim (Pashto): *"the pre-trained TTS is retrained during SO with RNNLM penalizer and later used for
  TO training with alpha = 0.3."* This is the closest thing reached to our design (a jointly-adapted phi
  feeding the BT term) - and note the ordering: **phi is adapted by the main objective first, and only then
  consumed by the auxiliary.**
- The cycle term also gets an LM regulariser: `L_SO = E_{p_ASR(Y|X)}[L_TTS(X|Y) + beta L_LM(Y)]`, motivated
  as *"a regularization role similar to the Kullback-Leibler term in VAEs"*; beta's value is not given in
  the text read (**UNVERIFIED**).

**(e) Numbers.** LibriSpeech 100 h paired + 360 h unpaired, ASR and TTS pre-trained on WSJ-si84 (so an
out-of-domain reverse model by construction), RNNLM at test in all rows. %WER dev-clean / dev-other /
test-clean / test-other:

| | dev-clean | dev-other | test-clean | test-other |
|---|---|---|---|---|
| Baseline (100 h paired) | 14.3 | 36.4 | 14.4 | 36.9 |
| 360-SO (cycle) | 11.0 | 32.4 | 10.6 | 33.6 |
| 360-SO +aug | 9.1 | 24.2 | 8.9 | 25.0 |
| 360-SO +LMP | 6.0 | 18.6 | 5.8 | 19.0 |
| 360-TO (BT alone) | 8.9 | 23.0 | 8.6 | 24.1 |
| **360-TO +alpha (0.7)** | **4.5** | **15.8** | **4.7** | **15.9** |
| 360-ST | 6.9 | 22.5 | 6.9 | 23.6 |
| 360-ST +aug | 6.0 | 18.6 | 5.8 | 19.0 |
| 360-ST +alpha | 5.2 | 19.5 | 5.3 | 20.4 |
| 360-ST +alpha +LMP | 4.3 | 14.9 | 4.3 | 15.2 |
| oracle (460 h paired) | 3.7 | 12.3 | 3.5 | 12.6 |

**The context-scaling guard on the BT branch is worth 24.1 -> 15.9 WER on test-other (8.2 absolute).**
That is far larger than any weight-tuning effect anywhere in this literature.
*Caveat, flagged:* the 360-ST block is not monotone (+alpha row is worse than +aug on the -other sets) and
the "360-ST +aug" row is byte-identical to the "360-SO +LMP" row; one of them is likely a typo in the
published table. Treat the ST decomposition as indicative; the TO rows are unambiguous.

BABEL-Pashto, %WER (oracle = 39.75 h supervised):

| supervision | Baseline | SO | TO | ST |
|---|---|---|---|---|
| 5 h | 71.4 | 63.9 | 62.2 | 63.1 |
| 5 h +aug | 63.1 | 61.4 | 61.7 | 60.1 |
| 5 h +alpha | - | - | 58.8 | 58.5 |
| 5 h +LMP | - | - | - | 58.4 |
| 10 h | 64.7 | 61.1 | 60.9 | 60.4 |
| 10 h +aug | 55.5 | 55.0 | 54.0 | 53.6 |
| 10 h +alpha | - | - | 53.7 | 52.5 |
| 10 h +LMP | - | - | - | 51.6 |
| oracle / oracle+aug | 56.0 / 48.9 | - | - | - |

Note at 5 h **ST (63.1) is worse than TO alone (62.2)**: combining the two terms is not automatically
better when the recognizer is weak. Authors: *"with 5 hours of paired data, the effect of TO is higher
compared to SO, but with 10 hours the TO obtains comparable gains as SO."*

### 1e. Tjandra, Sakti, Nakamura, ASRU 2017, "Listening while speaking: Speech chain by deep learning"
https://arxiv.org/abs/1707.04879 (full text read)

**(a) The only paper reached that sweeps the weight, and it goes the opposite way to intuition.** Eq. 5:

    L = alpha * (L_TTS_P + L_ASR_P) + beta * (L_TTS_U + L_ASR_U)

with **beta = 1 always and alpha in {0.25, 0.5}** - i.e. the *unsupervised* terms (including the BT term
`L_ASR_U`, Algorithm 1 line 13, CE of the real text against the ASR run on TTS-synthesised speech) are at
full weight and the **supervised/grounded term is down-weighted**. Their finding on when to raise alpha,
verbatim: *"we found that a different ASR performance alpha = 0.5 produced a larger improvement than
alpha = 0.25. We hypothesize that because the baseline model is not as good as the previous single
speaker experiment, we should put a larger coefficient on the loss and the gradient provided by the
paired training set."* **Weaker seed => more weight on the grounded term, less on the chain terms.**

**(b)** Both models are updated by the combined loss (Eqs. 6-9), the only paper reached in which the
reverse model is trained by the joint objective. But note that the BT term `L_ASR_U` still only updates
theta - the TTS is updated by the *other* branch (`L_TTS_U`), not through the BT gradient.

**(c)** Algorithm 1 line 11: `x_hat_U ~ P_TTS(.|y_U)`, sampled; line 17, the reverse direction uses
argmax with teacher-forcing. The "gen. mode" column (greedy vs beam-5) is the ASR generation.

**(e)** CER %, single-speaker synthesized corpus (10k paired + 40k unpaired): 10.06 baseline ->
5.83 (alpha 0.25, greedy) / 5.75 (0.5, greedy) / **5.44 (0.25, beam 5)** / 5.77 (0.5, beam 5).
Multi-speaker BTEC ATR-EDB, 80 utt/speaker paired: 26.47 baseline -> 23.03 / 20.91 / 22.55 / **19.99**
(same order). No LM, no lexicon.

### 1f. Tjandra, Sakti, Nakamura, 2019, "End-to-end feedback loss in speech chain framework via straight-through estimator"
https://arxiv.org/abs/1810.13107 (full text read)

**(c) The cleanest argmax-vs-sampling measurement in the family, and it is a NULL.** Loss is a plain sum,
`L_ASR + L_rec_TTS` (weight 1 each). WSJ eval92, CER %, no LM, beam-5 decoding, temperature tuned per
scenario over tau in {0.25, 0.5, 1, 2} on dev CER:

| generation | straight-through | CER % |
|---|---|---|
| teacher-forcing | ST-argmax | 5.75 |
| teacher-forcing | ST-Gumbel (sampling) | **5.70** |
| greedy | ST-argmax | 5.84 |
| greedy | ST-Gumbel (sampling) | 5.88 |
| baseline (L_ASR only, their Att MLP-MA) | - | 6.43 |

**Sampling vs argmax is worth at most 0.05 CER and flips sign between the two rows.** The paper's abstract
credits "sampling from ST-Gumbel-Softmax" with the 11 % relative gain, but the gain over ST-argmax is 0.05
CER; the 11 % is against the *non-chain* baseline. This is a case where the headline claim and the table
disagree about what the sampling choice bought.

---

## 2. The unit-level (discrete-token) family

### 2a. Zhang et al., EMNLP 2022, "SpeechUT: Bridging speech and text with hidden-unit for encoder-decoder based speech-text pre-training"
https://arxiv.org/abs/2210.03730 (full text read, relevant sections)

The closest architectural analogue to our units: a text-to-unit (T2U) generator converts unpaired text
into HuBERT k-means units (500 classes, adjacent-frame repeats removed) which then train a unit-to-text
model. **(b) The T2U generator is a SEPARATE, OFF-LINE, FROZEN seq2seq model trained on PAIRED ASR data**
(6+6 Transformer layers, d=768): *"As the units generated from text should have the same style as the
units generated from speech, we leverage a small amount of paired ASR data to train the T2U generator."*
The unit "style match" requirement is stated as the reason paired data is used at all. Loss weights
(lambda, gamma) = (0.1, 0.5) for ASR pre-training; 3 task losses are accumulated before each optimizer
step. **Not applicable to us as a method** (it spends labels), but it is the field's answer to
"where does a text-to-unit model come from": from labels, offline, and frozen.

### 2b. Liu, Chang, Hsu, Auli, Glass, 2022, "Towards end-to-end unsupervised speech recognition" (wav2vec-U 2.0)
https://arxiv.org/abs/2204.02492 (full text read, relevant sections)

**Not back-translation**, but it is the only paper reached that adds an auxiliary term to an unsupervised
objective **in our exact input regime** (raw wav2vec 2.0 50 Hz features, unpaired phonemized text, zero
labels), so its auxiliary-term engineering is the transferable part.

- **The motivation is our exact failure mode**, verbatim: *"The generator can possibly learn to satisfy the
  adversarial criterion without learning the correct mapping - for example, by consistently producing the
  most common n-grams regardless of the input speech."* The auxiliary's stated job: *"it provides a
  content-based regularization to ensure the output of the generator is closely related to the input
  speech."*
- **(a) Weight.** Eq. 6, `... - lambda L_gp + gamma L_sp + eta L_pd + delta L_ss`, with the text giving
  *"1.0 / 1.5, 1.5 / 2.5, 0 / 3, 0.3 / 0.5 for lambda, gamma, eta, delta respectively"* - i.e. the
  auxiliary weight **delta = 0.3 or 0.5**, an order of magnitude below the main term. The two values per
  weight are not labelled in the text read (two dataset/language settings, presumably) - **UNVERIFIED as
  to which is which**. No annealing: fixed weight, fixed lr (5e-5 / 3e-4), 100k steps interleaving
  generator and discriminator updates 50k/50k.
- **Where the auxiliary attaches matters**: `L_ss = -sum_t log P_G(z_t|X)` is read off *"an additional
  linear transformation with the softmax function from the intermediate layer of the generator"* -
  **not the output layer**. It regularises the representation, not the emitted phone posterior.
- **(e) Value, with the spread.** LibriSpeech dev-other, **greedy-decoded PER, averaged over 8 runs**:
  15.9 +/- 1.1 without the auxiliary -> **13.6 +/- 0.9 with it** (Table 1, steps vii -> viii); 14 % relative.
- **The negative half of the same table (Table 2): a badly-chosen auxiliary is WORSE THAN NONE.**
  None 15.9 +/- 1.1; wav2vec2.0 VQ indices 16.6 +/- 2.2; k-means on w2v2 features 16.4/15.5/15.9 (32/64/128
  clusters); k-means on MFCC 15.2/**13.6**/14.8/16.8 (50/64/100/128). The auxiliary's *source* is worth
  3.2 PER between best and worst choice, and two of the eight choices are net-harmful.
- **Seed spread and convergence rate, which bound any claim we make:** *"we run each model with 4 different
  random seeds and we observe that over 80% of the models will converge during training"*; std over 8 runs
  is 0.9-2.2 PER. In wav2vec-U (https://arxiv.org/abs/2105.11084, full text read for these sections) the
  penalty weights are *searched*: *"the gradient penalty weight lambda is selected from the range
  [1.5, 2.0], the smoothness penalty weight gamma from [0.5, 0.75], the phoneme diversity loss weight eta
  from [2, 4], and we train 5 seeds for each configuration for a total of 40 models"*, with mean and std
  over **20 random seeds** reported for the silence-insertion study.
- **The label-free selection rule we should copy.** wav2vec-U 4.3: *"we developed the following
  cross-validation metric which does not require labeled data. We use the metric for early stopping,
  selecting a random seed, and hyper-parameter selection (lambda, gamma, eta) ... We consider two
  quantities: LM negative log-likelihood (NLL) and vocabulary usage ... Measuring vocabulary usage
  identifies degenerate models which output fluent but trivial transcriptions."* The second quantity is
  precisely a detector for our all-blank / content-free attractor.

---

## 3. Direct answers to (a)-(e)

**(a) Weight and schedule.** There is no consensus value and **nobody reached tuned it carefully**:
- 0.5 / 0.5 interpolation of BT against the cycle term, set as a default (Baskar 2019 Eq. 13).
- 1 / 1 plain sum (EAT `L_ST = L_SO + L_TO`; Tjandra 2019 `L_ASR + L_rec_TTS`).
- Alternating batches, effective weight 1 (Hori 2019); concatenated datasets, data-proportional (Hayashi 2018).
- Grounded term down-weighted instead of the auxiliary: alpha = 0.25-0.5 on the *supervised* loss, beta = 1
  on the chain losses (Tjandra 2017) - **and raised to 0.5 when the seed model was worse.**
- An auxiliary in a genuinely unsupervised objective: delta = 0.3-0.5 against a main term of 1 (wav2vec-U 2.0).
The **schedules** that exist are not on the weight: EAT anneals *which supervised data is released*
(exp best: 22.1 vs 23.6 linear vs 24.3 log, test-other), and Hori/Hayashi early-stop hard (epoch 6 of many;
15 retraining epochs vs 30 initial). The one thing several papers do agree on is **the BT batches must be
interleaved with real-audio batches**, and that this is not a tuning detail but the difference between a
gain and a loss (Hayashi: 28.3 alone vs 23.6 mixed vs 25.2 baseline).

**(b) Whose text-to-unit model.** In every paper reached the reverse model is **pre-trained on paired data
outside the loop and is never updated by the BT term's gradient**. Hori 2019 names joint training as
future work. EAT is the only one where phi is adapted at all, and the order is explicit: *pre-train, adapt
via the cycle (SO) term, then consume in the BT (TO) term.* Our design (phi trained jointly by the lattice
EM objective and then used for BT) is therefore **beyond what any of these papers did**, and the
literature offers no evidence either for or against it. What it does offer: nobody has found it necessary
or safe to let the BT loss push the reverse model, and SpeechUT states the *reason* a paired seed is used
for T2U at all - the synthesised units must match the **style** of real units, or the recognizer learns the
synthesis artefact instead of the content.

**(c) Sampling vs argmax.** Split by direction and consistent across papers: the **recognizer's output is
sampled** (needed for REINFORCE: N=5 in Hori and Baskar; ST-Gumbel or ST-argmax in Tjandra 2019), while the
**text->speech/unit direction uses argmax / a point estimate** (Baskar Eq. 12 verbatim; Hayashi's TTE
decode; Tjandra 2017 line 11 is the one exception, a sample). Diversity in the BT direction is injected
through a **nuisance variable, not the content**: randomly chosen x-vectors (Baskar 2019), Prenet dropout
kept on at generation (Hayashi 2018), SpecAugment on the synthetic side (EAT). Where argmax and sampling
were measured head to head (Tjandra 2019), the difference is <= 0.05 CER and sign-unstable - **a null**.

**(d) What fails when the text-to-unit model starts cold or mismatched.** Three distinct, measured
failure modes:
1. **The whole objective becomes a no-op.** EAT on Pashto: the predecessor model *"did not lead to gains"*
   because the multi-speaker TTS was too poor.
2. **The BT branch degrades theta below baseline.** Hayashi: BT stream alone 28.3 vs 25.2 baseline. Hori's
   adjacent measurement: theta trained on its own (bad) hypotheses as CE targets goes to 89.3 WER even at
   weight 0.1.
3. **The named mechanism.** EAT: free-running synthesis has no teacher forcing, so *"the reconstructed
   output deviates from the ground truth. Even the segments of speech and silence are wrongly predicted"* -
   i.e. the **durations/segmentation are wrong first**, which for us is the HSMM's d and the SIL structure.
The **guards that worked**, in decreasing measured value: scale down the synthetic branch's acoustic
contribution (EAT alpha, 24.1 -> 15.9 test-other; alpha = 0.3 when the mismatch is severe, 0.7 when mild,
alpha = 0 degenerating the branch to pure LM training); keep real-audio batches co-present (Hayashi,
28.3 -> 23.6); down-weight the auxiliary relative to the main term (wav2vec-U 2.0 delta = 0.3-0.5) and
raise the grounded term's weight when the seed is weak (Tjandra 2017).

**(e) Where the auxiliary beat the main objective alone.** Only one paper reached measures exactly this
contrast (main objective alone vs main + BT auxiliary, same setup):
- **Baskar 2019, LibriSpeech 100 h paired + 360 h unpaired, WER with RNNLM: cycle alone 16.8 -> cycle + BT
  at alpha = 0.5: 16.6.** 0.2 absolute. Supervised baseline 21.0, oracle 11.8.
- Same paper, WSJ, weaker seeds (%WER): 10 h paired + 67 h unpaired, 28.4 -> 27.7; 5 h + 67 h, 38.8 -> 34.2;
  2 h + 14 h, 49.8 -> 43.7. **The auxiliary's value grows as the recognizer gets worse - up to a point:
  at 2 h + 67 h the combination (41.4) is worse than BT alone (39.6).**
- EAT, LibriSpeech: SO + LMP 19.0 vs ST + alpha + LMP 15.2 (test-other) - but the ST rows carry the table
  inconsistency flagged in 1d, and specaugment/LMP/alpha are entangled, so this is not a clean
  main-vs-main+aux contrast.
- BABEL Pashto 5 h: SO 63.9 vs ST 63.1, and with augmentation SO 61.4 vs ST 60.1 - but ST (63.1) is worse
  than TO alone (62.2).
**Every one of these is semi-supervised.** No paper reached adds a BT auxiliary to a fully unsupervised
(zero-label) speech objective. Our setting has no paired seed for phi at all, which is strictly outside
the regime in which all of the above gains were measured.

---

## 4. What this changes for our design

1. **Never let the BT term be theta's only data in a step.** Interleave BT batches with EMC batches on real
   audio, or the Hayashi regime applies (BT-only retraining was 3.1 WER *worse* than doing nothing). This
   is the one thing the literature is unanimous and quantitative about.
2. **Start the auxiliary weight low, not at parity.** Parity (0.5/0.5, or 1/1) is what the semi-supervised
   papers use, but all of them have a paired-data-trained phi. The only auxiliary reached that was added to
   a zero-label objective runs at delta = 0.3-0.5 against a main term of 1 (wav2vec-U 2.0), and Tjandra
   2017's one sweep says: the worse the seed, the more weight on the grounded term. Pre-register
   lambda_BT in {0, 0.1, 0.3} with 0 as the registered null.
3. **Copy EAT's guard, not just its loss.** The highest-value single intervention in this entire literature
   is not a weight - it is **attenuating the synthetic branch's acoustic pathway** (24.1 -> 15.9 WER), with
   the attenuation chosen by how mismatched the synthesizer is. Our unit-level analogue is a per-branch
   scale on theta's use of the sampled units (or, equivalently, a temperature/interpolation toward a
   content-free input). alpha = 0 is a defined, meaningful operating point there: the BT term becomes a
   pure LM/prior term on theta's output and cannot corrupt the acoustic mapping. **That gives us a safe
   floor for a cold phi and a single knob to open as phi improves.**
4. **Use argmax for synthesis; put the randomness in the nuisance dimensions.** The field's split is
   consistent and the one head-to-head measurement is a null (<= 0.05 CER). Concretely: argmax the unit
   identity, sample the duration d and the speaker/eta row (which is what our probe already does), rather
   than sampling the unit per frame. This is a cheap change with literature support and no measured cost.
5. **Order the coupling: phi adapted by EMC first, then consumed by BT.** That is EAT's ordering, and it is
   the only ordering anyone has run. Gate the BT term's activation on a phi-quality criterion rather than
   switching it on at step 0.
6. **Pre-registrable read (label-free, and it fits our no-labels rule).** Adopt wav2vec-U's unsupervised
   cross-validation metric as the selection and kill criterion: **phonemized-text LM NLL of the Viterbi
   decode, plus vocabulary usage** (fraction of the phone inventory actually emitted). Vocabulary usage is
   a direct detector of the all-blank/content-free attractor and costs nothing. Register the BT-auxiliary
   read as: *lambda_BT in {0, 0.1, 0.3} x >= 4 seeds each; primary read = per-item paired delta in PER on a
   fixed bed at a fixed decode; the arm is funded only if the paired delta clears the seed spread.*
   The field's own spread bounds what is claimable: wav2vec-U 2.0 reports +/- 0.9 to 2.2 PER over 8 runs and
   ~80 % convergence over seeds in our exact feature regime. **A single-seed improvement below ~1 PER in
   our setting is indistinguishable from seed noise**, and the one measured BT-auxiliary gain in the
   literature (Baskar's 16.8 -> 16.6 WER) is smaller than that spread.
7. **What would settle the question the literature does not answer.** Nobody has run a BT auxiliary with a
   jointly-trained, label-free reverse model. The decisive experiment is ours to run, and the cheapest
   informative version is the EAT alpha-sweep, not a weight sweep: at lambda_BT fixed, sweep the synthetic
   branch's attenuation from 0 (pure prior term) upward and find where the auxiliary stops helping. If it
   only helps at alpha near 0, the auxiliary is contributing an LM/prior signal and not cycle content -
   which is a publishable negative and tells us to spend the compute on the prior instead.

---

## 5. Reading status

| Paper | Status | Used for numbers |
|---|---|---|
| Hori et al. ICASSP 2019 (1811.01690) | FULL TEXT READ | yes (Tables 1-2, Sec 2.4, 4.1) |
| Hayashi et al. SLT 2018 (1807.10893) | FULL TEXT READ | yes (Tables 1-4, Sec 2.4, 3.2) |
| Baskar et al. Interspeech 2019 (1905.01152) | FULL TEXT READ | yes (Eqs. 10-13, Tables 1-3) |
| Baskar et al. ICASSP 2021 EAT (2104.07474) | FULL TEXT READ | yes (Eqs. 1-9, Tables 1-4) |
| Tjandra et al. ASRU 2017 (1707.04879) | FULL TEXT READ | yes (Alg. 1, Eq. 5, Tables 1-2) |
| Tjandra et al. 2019 ST (1810.13107) | FULL TEXT READ | yes (Eq. 13, Table 1, Sec 5.2) |
| Liu et al. 2022 wav2vec-U 2.0 (2204.02492) | FULL TEXT READ (Sec 3-4) | yes (Eq. 6, Tables 1-2, Sec 4.1) |
| Baevski et al. 2021 wav2vec-U (2105.11084) | READ Sec 4.3, 5.3 only | yes (CV metric, weight ranges, seed counts) |
| Zhang et al. EMNLP 2022 SpeechUT (2210.03730) | FULL TEXT READ (Sec 3-5) | yes (Sec 3.3, 4.3) |
| Ao et al. 2021 SpeechT5 (2110.07205) | DOWNLOADED, NOT READ | no - no BT auxiliary; shared-codebook pre-training, out of scope |
| Xu et al. 2020 IPL | NOT READ | no - standalone iterative pseudo-labelling, excluded by the dispatch's own framing |
| Ren et al. ICML 2019 | covered in the prior report | see lit_backtranslation_cold_start_2026-09-15.md Sec 2b |
