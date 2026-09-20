# Lexicon inside the training objective of unsupervised / weakly supervised ASR

Literature review, 2026-09-21. Question: has published unsupervised or weakly supervised ASR
put a LEXICON (phone strings must decompose into dictionary words, scored by a word n-gram)
into the TRAINING OBJECTIVE of a phone/unit-level recogniser, as opposed to only into decoding?
Our setting: blank-free lattice DP over 39 phones + SIL, trigram phone prior inside the DP,
forward algorithm, min duration 2, batched on GH200, no transcripts. Measured discriminating
power of the exact lexicon score is 2.3 nats/phone (gold minus private-code) vs 1.4 for the
phone trigram and 1.85 for the best neural phone LM; single-token neighbourhood probes show
the neural phone LM's local signal tracks the trigram.

All statements below verified against full text (PDF downloaded and converted), not abstracts.

---

## 1. Per source: where the lexicon enters, and by what mechanism

### 1.1 Klejch, Wallington, Bell (Interspeech 2022), "Deciphering Speech: a Zero-Resource
Approach to Cross-Lingual Transfer in ASR", pp. 2288-2292.
https://www.isca-archive.org/interspeech_2022/klejch22_interspeech.pdf

**WHERE: inside the training objective, via Baum-Welch over a composed WFST. The only
clean precedent found.**

Sec. 2.2, p. 2289. The model is a noisy-channel decipherment, parameterised following
Nuhn (2019):

> "ˆY = arg max_Y P(Y|X) = arg max_Y P_lex(X|Y,A) P_lm(Y) P_ali(A)"   (Eq. 3)

and expressed as a WFST composition (p. 2289, Eq. 4):

> "ˆY = shortest path(X ◦ (L ◦ A ◦ G)), where X is an input phone acceptor, L is the lexicon
> model transducer, A is the alignment model transducer and G is the language model acceptor."

Training is EM/Baum-Welch on the composed machine, i.e. the alignment variable A and the
label sequence are **marginalised** while the phone->grapheme substitution table is re-estimated:

> "the lexical model P_lex(X|Y) can be trained in an unsupervised fashion with the Baum Welch
> algorithm" (p. 2289, Sec. 2.2).

Crucially the word LM and the (grapheme) lexicon are in the **training** composition, not only
in the final decode (p. 2289, Sec. 2.2, last paragraph):

> "Hence we start with a simple bigram model and move to using up to 5-gram grapheme models as
> training progresses. Subsequently, we use a word trigram language model together with a
> grapheme lexicon for the final round of training. Since the composition of the lexical model,
> alignment model and the word language model L◦A◦G is slow, and is required after every
> training epoch, we reimplemented the standard composition X◦(L◦A◦G) with a three-way
> composition X◦(L◦A)◦G [Allauzen & Mohri, CIAA 2008]. We also use pruning to speed up training
> and inference with the word language models."

Caveat on terminology: their "lexical model" P_lex(X|Y) is NOT a pronunciation dictionary; it is
a one-state flower transducer of phone<->grapheme substitution/insertion/deletion probabilities
(the learned parameters). The **lexicon proper** is the *grapheme* (spelling) lexicon word ->
letter string that turns the word trigram G into a letter-level acceptor. Structurally this is
exactly the word-trie x word-LM-history state space our design proposes, only with graphemes as
the terminal alphabet instead of phones. The phone side enters through P_lex.

Scale actually handled (Sec. 3.1, p. 2290):
- input alphabet: multilingual shared phone set; target: target-language graphemes.
- **training word LM: 100k most frequent words; inference word LM: 300k most frequent words.**
  ("during training we used a word language model containing only the 100k most frequent words
  but during inference we used the language model containing the 300k most frequent words")
- word trigram, SRILM, up to 1B CommonCrawl tokens, pruned to top-300k vocabulary.
- training data for decipherment: **20 minutes of the shortest development-set utterances** per
  language; 20 full-batch iterations per LM stage.
- lexical model pruned to "the top 20 phones for each grapheme" before moving to word LMs.
- 50 random restarts at the bigram stage, best training likelihood kept (Berg-Kirkpatrick &
  Klein, EMNLP 2013).

Tricks that were *necessary*, reported as such:
- **Curriculum on the prior**: character bigram -> up to character 5-gram -> word trigram +
  grapheme lexicon. Reason given (p. 2289): "using n-gram models with large contexts results in
  a big composition when composed with an unpruned lexical model; therefore it is not feasible
  to use them from the beginning."
- **Smoothing of the learned table against uniform**, alpha = 0.9 (p. 2289, Eq. 5):
  P^s_lex(x|y) = alpha P_lex(x|y) + (1-alpha)/|X|. Applied "at various stages of training",
  and again before each of the two transitions into the word-LM stage. Motivation: pruning of
  unseen substitutions "can be detrimental" because important arcs are lost.
- **Silence as a word-boundary anchor** (p. 2289): "the lexical model always maps silence phones
  to silence or a word boundary in the output. This results in faster training/inference -
  because silence prunes the space of possible word segmentation - and more accurate
  decipherment." Directly relevant: our DP already has an explicit SIL unit.

Results, GlobalPhone, WER (Table 1, p. 2290), decipherment / +semi-supervised training:
BUL 31.0/10.7, CES 49.0/16.5, HAU 70.8/30.3, POR 53.8/21.2, SWA 105.3/98.3, SWE 93.5/55.8,
UKR 34.5/10.3. Oracle supervised with CommonCrawl LM: 7.9 / 12.9 / 12.2 / 17.0 / 10.2 / 23.3 / 8.4.
Hand-crafted phone-mapping baseline: 35.0/12.6, 44.3/16.8, 43.4/25.3, 52.8/21.8, 82.5/52.8,
66.8/37.5, 35.7/12.1.

**Failure modes reported explicitly:**
- Total failure on Swahili (WER > 100 even after cleaning beeps/leading silence); Hausa and
  Swedish much worse than the phone-mapping baseline. Their conclusion (p. 2291): "These Swahili
  results suggest that in order to be able to decipher speech from a new language we need to
  find speech amenable to decipherment." So the objective is **not robust**: it worked for 4 of
  7 languages and was clearly beaten by hand phone-mapping on 3.
- **Length exploit, explicitly**: "Since the decipherment tends to produce a lot of deletion
  errors we used a deletion penalty during decoding to allow the model to learn to fix them"
  (p. 2290, Sec. 3.1). The model buys LM score by deleting phones. Our P_lex has an explicit
  deletion arc P_lex(x|epsilon) and the alignment transducer A permits one insertion or one
  deletion in a row (Fig. 2, p. 2289) - the deletion bias arises even with that restriction.
- Sensitivity to initialisation: 50 random restarts were needed at the bigram stage.

### 1.2 Baevski, Hsu, Conneau, Auli, "Unsupervised Speech Recognition" (wav2vec-U), NeurIPS 2021
/ arXiv 2105.11084. https://arxiv.org/pdf/2105.11084

**WHERE: NOT in the objective. Lexicon enters only (i) offline, as the G2P used to phonemize
the unpaired text corpus, and (ii) at decoding, as a phone->word WFST with a word 4-gram; and
(iii) in self-training.**

Objective (Sec. 4.2, Eq. 6):

> min_G max_C E_{Pr~Pr}[log C(Pr)] + E_{S~S}[log(1 - C(G(S)))] - lambda L_gp + gamma L_sp + eta L_pd

with L_gp gradient penalty (Eq. 7), L_sp segment smoothness (Eq. 8), L_pd phoneme diversity
(Eq. 9). No lexicon, no word LM, no marginalisation - the discriminator sees *phonemized* real
text, so all lexical knowledge is implicit in the empirical distribution of real phone strings.

Decoding (Sec. 5.4): "we use the same phonemizer with which we pre-processed the unlabeled text
data to build a mapping between phonemes to words. The WFST is composed with a 4-gram language
model (Heafield, 2011) pruned to keep only 4-grams occurring more than 3 times."

Unsupervised model selection (Sec. 4.3) uses a **phoneme** LM NLL plus vocabulary usage, not a
word LM.

**Failure modes explicitly named and guarded against:**
- Collapse to fluent-but-trivial output: "Vocabulary usage is the proportion of the phoneme
  vocabulary being output by the model via Viterbi decoding. **Measuring vocabulary usage
  identifies degenerate models which output fluent but trivial transcriptions.**" (Sec. 4.3).
  The whole L_pd term (Eq. 9) exists to stop this.
- **Length exploit, explicitly**: the final model-selection step (Eq. 11) takes the highest
  **unnormalised** sum of log probability, and the paper says why: "This selects model
  configurations which produce phoneme sequences that score high under the language model but
  are not too long."
- Trivial low-entropy solutions in decoder tuning (Eq. 12 and following): "In practice, trivial
  solutions may achieve very low entropy. We counteract this by replacing H_LM(Pbar_j) by the
  average entropy of the language model training data if H_LM(Pbar_j) is lower than the entropy
  of the training data."
- Silence must be an explicit label or the model repurposes a phone for it (Sec. 4.1): "Without
  a silence label, we noticed that the model was repurposing a particular phoneme to label
  silences which resulted in much lower performance since it interfered with subsequent language
  model decoding." (We already have SIL.)

Numbers: Librispeech test-other WER 15.1 -> 18.0 (Large, LL-60k, 4-gram) and 6.0 -> 6.5 with
self-training. TIMIT PER 32.5-59.8 unsupervised, 11.3-26.3 with ST.

### 1.3 Liu, Hsu, Auli, Baevski, "Towards End-to-end Unsupervised Speech Recognition"
(wav2vec-U 2.0), SLT 2022 / arXiv 2204.02492. https://arxiv.org/pdf/2204.02492

**WHERE: NOT in the objective. Identical situation to wav2vec-U 1.0 - word LM only in decoding
(PyKaldi WFST) and self-training.**

Full objective, Eq. (6), Sec. 3.3: GAN terms + L_gp + L_sp + L_pd + delta * L_ss, where
L_ss = - sum_t log P_G(z_t | X) (Eq. 5) reconstructs **MFCC k-means pseudo-labels (64 clusters)**
from an intermediate generator layer. Purely audio-side. No lexicon anywhere in training.

**The most important quote in this review for our design** (Sec. 3.3, p. 4):

> "The generator can possibly learn to satisfy the adversarial criterion without learning the
> correct mapping - for example, **by consistently producing the most common n-grams regardless
> of the input speech.** To address this issue, we introduce an auxiliary objective function
> that aids self-supervised learning by reconstructing a pseudo label sequence Z ... derived
> from the input audio."

i.e. the field's diagnosis of exactly the disease our lexicon term is meant to cure ("the text
prior is satisfied by frequent patterns, not by content"), and its chosen fix is **not a
lexicon**: it is an audio-side reconstruction anchor that ties output to input. They state the
two benefits as "(a) it provides a content-based regularization to ensure the output of the
generator is closely related to the input speech and (b) it guides the generator with a more
explicit signal". Sec. 4.2 step (vii)->(viii) reports "introducing the auxiliary self-supervised
objective improves PER significantly".

Decoding, Sec. 4.1/4.2: "Pykaldi is used to decode the phone-based ASR outputs into words."

### 1.4 Yeh, Chen, Yu, Yu, "Unsupervised Speech Recognition via Segmental Empirical Output
Distribution Matching", ICLR 2019 / arXiv 1812.09323. https://arxiv.org/pdf/1812.09323

**WHERE: NOT in the objective; lexicon-free. A *phone* n-gram LM is in the objective, but as a
corpus-level marginal-matching term, not a per-utterance marginalisation.**

Objective (Sec. 2.2, Eq. 1-3):

> J_ODM(theta) = - sum_{tau} sum_{z in Y^N} p_LM(z) ln p^tau_theta(z),
> with p^tau_theta(z) = (1/K) sum_{i=1}^K p_theta(y_{tau_i} = z | x_{tau_i})
> J(theta) = J_ODM(theta) + lambda J_FS(theta)

"the cost function measures the cross-entropy between the pretrained N-gram LM p_LM(z) and
p^tau_theta(z)". Note the **empirical average over segments sits inside the log** - the
mode-covering direction. J_FS is a frame-wise smoothness term (Eq. 2).

Scale and cost (Appendix, p. 13): N = 5; "for computational issues we only consider the most
frequent 10000 5-grams. ... Among the 5-gram language model P_LM, 69553 5-grams (out of 48^5
possible 5-grams) are non-zero, and the top 10000 5-grams account for almost half of the
probability." Batch size schedule 500 -> 20,000; the paper notes SGD on this objective is
"intrinsically biased if we also sample this empirical average by a mini-batch average. To
alleviate this effect, we use a large mini-batch size to estimate the stochastic gradients."
39 phone classes (61 -> 48 -> 39). PER decoded with Kaldi using per-frame softmax + LM.

The word "lexicon" in this paper means the phone inventory ("a lexicon of 61 distinct phoneme
classes"), not a pronunciation dictionary. No word-level term at all.

### 1.5 Chen, Tsai, Liu, Lee, Lee, "Completely Unsupervised Phoneme Recognition by a GAN
Harmonized with Iteratively Refined HMMs", Interspeech 2019 / arXiv 1904.04100.
https://arxiv.org/pdf/1904.04100

**WHERE: NOT in the objective. Phone-level GAN + HMM refinement; lexicon and word LM appear
only as an aside about decoders.**

Their statement (p. ~205 of the extracted text, discussion section):

> "available decoders such as WFST, which includes lexicon and language model"

The training loop is: GAN maps segment embeddings to phones; an HMM is trained on the GAN's
output and used to re-generate better labels; iterate. The iterative HMM refinement is the
interesting part for us as a *relaxation strategy* (see 3(c)); the prior in the objective is a
phone n-gram / adversarial phone-sequence match, never a word-level object. Reported PER better
than Segmental Empirical-ODM rows (e) 33.3/32.5/40.0/40.1 and (g) 36.5/41.6; their own
matched-case PER 34.3 is cited in the text.

### 1.6 Gao, Shi, Chuang, Garcia, Lee, Watanabe, Khudanpur, "EURO: ESPnet Unsupervised ASR
Open-source Toolkit", ICASSP 2023 / arXiv 2211.17196. https://arxiv.org/pdf/2211.17196

**WHERE: NOT in the objective. Lexicon is a k2 decoding graph only.**

Sec. 3.2 (training): "The network is mainly trained with a GAN-based loss and some [auxiliary
losses] ... To stabilize the training, three auxiliary losses are also proposed" - i.e. the
wav2vec-U objective, reimplemented.

Sec. 3.3 (decoding): "the graph-based search can utilize word-level LMs in the search [via] a
... graph H, a lexicon graph L, and a grammar graph G", with Fig. 4 showing "L topology of
lexicon {cat: k, ae, t; act: ae, k, t} in k2" and "The lattice is generated during decoding".

So the toolkit that *does* have an HLG machine keeps it strictly on the decoding side. This is
a useful negative: a group with k2 and full WFST infrastructure in hand chose not to compose L
and G into training.

### 1.7 Ni, Wang, Zhang, Qian, Gao, Hasegawa-Johnson, Yoo, "Towards Unsupervised Speech
Recognition Without Pronunciation Models", Speech Communication 2025 / arXiv 2406.08380 (v2,
8 Jan 2025). https://arxiv.org/pdf/2406.08380

**WHERE: the objective is word-level, but the lexicon is *deleted*, not added. Whole-word units
with no phone decomposition; the point of the paper is to remove the G2P/lexicon requirement.**

Abstract: "we tackle the challenge of developing ASR systems without paired speech and text
corpora by proposing the removal of reliance on a phoneme lexicon. We explore a new research
direction: word-level unsupervised ASR ... achieves a word error rate of between 20-23%,
depending on the vocabulary size, without parallel transcripts, oracle word boundaries, or a
pronunciation lexicon."

General UASR objective they state (Sec. 1, Eq. 1-2):
min_theta D_f[ P'(Y) || E_X[ Q_theta(Y|X) ] ] subject to D_g[P(Y) || P'(Y)] < epsilon.
Their instantiation is JSTTI: joint speech-text masked token infilling in a shared Transformer.

**Scale, and this is the decisive number for (a):** they could not run on open vocabulary at
all. Sec. 4.1: "Due to the extended tail distribution of English words in the LibriSpeech
corpus, we simplify the problem by **pruning the corpus so that only the top-K high-frequency
words exist in both the unpaired speech and text**", K in {1024, 2048, 4096}, with force-aligned
word segments concatenated to build the synthetic corpus. OOV is therefore not handled; it is
defined away.

WER (Tables 4, 5, Sec. 7): K=2048, best unsupervised-boundary 26.47, +self-training 20.21,
forced-alignment topline 16.57 (12.01 with ST). K=4096: 31.60 unsupervised, forced-alignment
topline degrades to 19.45. Their reading: "as we further increase the vocabulary size beyond
K = 2048, optimizing Eq. (1) on the whole-word level becomes more challenging for the JSTTI
model." Scaling to K=4096 needed **progressive vocabulary expansion**: copy encoder and the
top-1024 text pre/post-net rows from the K=1024 model, initialise C1025..C4096 from the top-3072
cluster means (Algorithm, line 7), then continue.

**The frequent-word collapse, measured** (Sec. 7 error analysis, Figures 8-10):

> "the error distribution is very uneven among groups of words with different occurrence
> frequencies. Word groups with a higher occurrence frequency usually have a lower overall error
> rate for all three vocabulary sizes. In contrast, **low-frequency words barely get recognized
> correctly**, especially for those in the 2048-word and the 4096-word corpora, **where the error
> counts for words in the later word bins even exceed the occurrence counts.**"

Error counts exceeding occurrence counts means the model is *inserting* frequent words in place
of rare ones. And self-training makes it worse in relative terms: "This simple pseudo-text
self-training routine generally reduces word error rates across all frequency bins; however,
the effect is more pronounced for words with high occurrence frequencies."

One mildly encouraging counter-observation: going 1024 -> 2048 did not hurt, "likely because
the inclusion of additional words makes the utterances contextually more natural", and the error
rate on the original top-1024 words dropped when the extra 1024 were added - richer context
helps the words you already had.

### 1.8 Wang, Hasegawa-Johnson, Yoo, "Unsupervised Speech Recognition with N-Skipgram and
Positional Unigram Matching" (ESPUM), ICASSP 2024, pp. 10936-10940.
https://sls.csail.mit.edu/publications/2024/LimingWang_ICASSP-24.pdf

**WHERE: NOT in the objective; lexicon-free, phone-level. Important because it is a *negative*
result about the value of longer-range statistics.**

Objective (Sec. 3.4, Eqs. 5-8):
L_skipgram = sum_{k in K} || Phat_{Y,k} - Phat^theta_{Xbar,k} o G ||_1 ;
L_unigram = sum_l || Phat_{Y_l} - Phat_{X_l} G ||_1 ;
total = L_unigram + L_skipgram + L_segment + lambda_smooth L_smooth.

Their critique of ODM (Sec. 1): "straightforward N-gram matching encounters difficulties: **the
quantity of unique N-grams quickly grows to intractable number and the accuracy of approximating
the N-gram distribution diminishes as N increases.** Notably, [Yeh 2019] revealed the necessity
of 5-gram and large batch size (50,000 tokens per batch) for EODM to yield optimal ASR-U
outcomes. Due to memory restrictions, they consider only the top 10,000 5-gram distributions."

Their ablation (Sec. on Table 3): "we found that **lower-order skipgrams play bigger roles than
higher-order skipgrams** for ASR-U ... Also the information in the positional unigrams is
crucial ... While adding tri-skipgrams [gives less]". They use bi-skipgrams with skip sizes up
to 6, tri-skipgrams with skip sizes up to 2, top 5000.

So within the ODM family, when you push the prior towards longer-range structure you get
*diminishing* returns, and the cheap positional-unigram statistic is what carries the signal.
This is a caution against the assumption that a longer-range/lexical term must pay.

### 1.9 Wang, Ni, Chang, Bhati, Harwath, Hasegawa-Johnson, Glass, "Towards Unsupervised Speech
Recognition at the Syllable-Level" / "Unsupervised Speech Recognition at the Syllable Level"
(SylCipher), arXiv 2510.03639 and 2608.22907. https://arxiv.org/pdf/2510.03639 ,
https://arxiv.org/pdf/2608.22907

**WHERE: NOT in the objective; the lexicon is replaced by a *syllable* inventory. This is the
field's current answer to the tractability problem at the word level.**

Their stated reason for not going word-level (2608.22907, Sec. 1):

> "A word-level alternative avoids G2P, but introduces a different problem: **the vocabulary of
> rare words is effectively unbounded, making coverage and generalization much harder.** Word
> segmentation also depends on longer-range context, which can destabilize the segmentation
> mechanisms used in current UASR systems."

and for syllables:

> "First, unlike words, **the number of syllables is finite, which reduces the long-tail token
> problem.** Second, in many languages, speech and text align more naturally at the syllable
> level than at the phone or word level. ... Third, recent progress in syllable boundary
> detection and unit discovery makes it possible to segment speech into syllable-like units
> without text supervision."

Objective (Eq. 8): L = lambda_1 L_wMLM + lambda_2 L_JE2E + lambda_3 L_PUSM - weighted masked LM
on both modalities, end-to-end boundary refinement, and positional-unigram/skipgram matching.
Text side is syllabified with Pyphen (a hyphenation tool), explicitly "without using a G2P".
Results: 21.8% CER matched on LibriSpeech (Pyphen+JE2E+PUSM), up to 40%/17% relative CER
reduction over prior G2P-free UASR; outperforms word-level JSTTI and character-level wav2vec-U
and REBORN; 12.2% PER on Mandarin.

### 1.10 Tseng et al., "REBORN: Reinforcement-Learned Boundary Segmentation with Iterative
Training for Unsupervised ASR", arXiv 2402.03988. https://arxiv.org/pdf/2402.03988

**WHERE: NOT in the objective. A *phone* 4-gram LM enters as an RL reward for the segmenter;
the lexicon is decoding-only.**

Sec. 3.1.3: the utterance reward R is a weighted sum of a perplexity-difference reward R_ppl
(computed with "a 4-gram phoneme LM"), an edit-distance reward R_edit and a length-difference
reward R_len. Sec. 4: "For decoding word-level outputs, we perform WFST decoding using PyKaldi."
Sec. 2 notes "a lexicon is required to transform the [phoneme transcriptions into words]".

**Length exploit again, named**: the two regularisation rewards exist because the perplexity
reward alone produces degenerate output - "To prevent such undesirable behaviors, we design two
regularization rewards, the edit distance reward and length difference reward, to ensure that
the policy learned by the segmentation model does not drastically alter the phoneme predictions
between iterations." So a third independent group needed an explicit length/edit anchor against
an LM-score-maximising objective.

### 1.11 Yang, Barkoczi, Schlueter, Ney, "Sequence-Level Unsupervised Training in Speech
Recognition: A Theoretical Study", arXiv 2603.02285 (2 Mar 2026).
https://arxiv.org/pdf/2603.02285

**WHERE: they propose exactly our objective - a single-stage sequence-level loss that
marginalises the label sequence under an LM prior. Theory only; no ASR experiments.**

Proposed criterion (Sec. 3.3):

> L(theta) = -(1/S) sum_{s=1}^S log q_theta(x^N_{s,1})
>          = -(1/S) sum_{s=1}^S log sum_{c^N_1} p_LM(c^N_1) q_theta(x^N_{s,1} | c^N_1)

> "The summation over c^N_1 can be computed efficiently via dynamic programming for a limited
> context LM [Povey et al., LF-MMI]; for a full-context LM, one can restrict the summation to a
> hypothesis space obtained by search."

This is the same shape as our lattice DP with the prior inside. Two conditions are given for
unsupervised ASR to be possible at all, and shown to be necessary absent further constraints
(Sec. 4):
1. **Structure constraint**: pr(x^N|c^N) = prod_n pr(x_n|c_n) - emissions are local/conditionally
   independent given labels. They argue it holds in practice because "[wav2vec-U, w2vu2.0, ESPUM]
   achieved successful unsupervised training with a localized mapping from segment
   representations to phonemes". Note a lexicon/word-LM term lives in p_LM(c), **not** in the
   emission, so adding it does not violate this condition.
2. **Full column rank of the LM matrix** P_C, (P_C)_{n,c} = pr_n(c), the position-dependent
   *unigram* marginals: "the labels cannot replace each other, even not possible for linear
   combinations, in terms of position-dependent unigram probabilities."

Theorem 1: D_q <= N^2 ||P_C^+||_1 sum_{x^N_1} |pr(x^N_1) - q(x^N_1)|, and with Pinsker,
(Delta_q)^2 <= beta D_KL(pr(x)||q(x)) with beta = 2 N^4 ||P_C^+||_1^2.

Measured constant, and it is a caution: "We computed the smallest singular value sigma_min of
P_C on LibriSpeech transcriptions and obtained sigma_min approx 3e-4. While small, this value is
clearly non-zero, suggesting that P_C is numerically full rank on the evaluated data."
||P_C^+||_1 therefore of order 1e3-1e4 and the bound is extremely loose. Identifiability is
technically satisfied by unigram statistics alone but is very ill-conditioned; the bound does
not tell you that a richer (lexical) prior helps, because P_C is defined by the LM's *unigram*
marginals and a lexicon does not change it.

Status: simulation validation only (|X|=4, |C|=3, N=3), single group, 2026 preprint, no speech
experiments. Do not treat as established.

### 1.12 Nuhn & Ney, "EM Decipherment for Large Vocabularies", ACL 2014 (short), pp. 759-764.
https://aclanthology.org/P14-2123.pdf

**Not speech, but this is the scaling law for the exact object we want to build, and Klejch 2022
takes its parameterisation directly.**

Model (Sec. 3, Table 2): word-level HMM, p(f^N_1, e^N_1 | theta) = p(e^N_1) prod_n p_lex(f_n|e_n)
with p(e^N_1) an n-gram LM; training maximises the marginal
theta = argmax sum_{[e^N_1]} p(f^N_1, e^N_1 | theta) (Eq. 4) by forward-backward EM (Eqs. 5-6).
That is our DP with the word LM inside, on a word-typed state space.

**Complexity table (Table 1) and the cost of exactness:**
- EM Full (Knight et al. 2006; Ravi & Knight 2011): O(N V^n), n = LM order.
- EM Fixed Candidates (Nuhn et al. 2012): O(N).
- EM Beam (this work): O(N V) - "we only keep the B most promising hypotheses ... exploring all
  V_e possible extensions for each active hypothesis, and thus scaling linearly with the
  vocabulary size. Due to that, standard beam search EM training is too slow to be used in the
  decipherment setting."
- EM Preselection (this work): O(N) - "for a hypothesis ending in a language model state sigma,
  we only look at B_LM many successor words e_{c+1} with the highest LM probability
  p_LM(e_{c+1}|sigma) and at B_lex many successor words e_{c+1} with the highest lexical
  probability p_lex(f_{c+1}|e_{c+1})". Settings: B = 100, B_lex = 5, B_LM = 50. Two lookup
  tables; the LM table is built once, the lexicon table is rebuilt each iteration.

**Measured scale (Table 4, VERBMOBIL, 27,862 sentences / 294,902 words):**

| Vocab | LM order | Method       | Acc. [%] | Time [h] |
|-------|----------|--------------|----------|----------|
| 200   | 2        | exact        | 97.19    | 224.88   |
| 200   | 2        | beam         | 98.87    |   9.04   |
| 200   | 2        | preselection | 98.50    |   4.14   |
| 500   | 2        | beam         | 92.12    |  24.27   |
| 500   | 2        | preselection | 92.16    |   4.70   |
| 3,661 | 3        | beam         | 91.16    | 302.81   |
| 3,661 | 3        | preselection | 90.92    |  19.68   |
| 3,661 | 4        | preselection | 92.14    |  23.72   |

with the flat statement: "**Exact EM is not tractable for vocabulary sizes above 200.**" For the
V=200 and V=500 rows they had to *create* the small vocabularies by mapping words to word
classes, because exact EM could not run on the natural 3,661-type vocabulary.

Same smoothing device as Klejch, and the same reason: "the new parameter estimates in Equation 5
can become zero. This is a critical issue, since pairs (e,f) with p(f|e) = 0 cannot recover from
acquiring zero probability in some early iteration. In order to allow the lexicon to recover from
these zeros, we use a smoothed lexicon phat_lex(f|e) = lambda p_lex(f|e) + (1-lambda)/|V_f| with
lambda = 0.9 when conducting the E-Step." (Identical alpha = 0.9.)

MT results: 3-gram preselection EM reaches BLEU 19.5 in 1.9 h vs Ravi & Knight's exact EM with a
whole-segment LM at 19.3 in 850 h.

### 1.13 Shinozaki, Watanabe, Mochihashi, Neubig, "Semi-Supervised Learning of a Pronunciation
Dictionary from Disjoint Phonemic Transcripts and Text", Interspeech 2017, pp. 2546-2550.
https://www.isca-archive.org/interspeech_2017/shinozaki17_interspeech.pdf

**WHERE: the lexicon is *inside* the objective, as a latent variable, in a joint Bayesian model
of word LM + pronunciation dictionary over unaligned phone transcripts and text. But the phone
side is GOLD phone transcripts, not acoustics.**

Model (Sec. 3.1): a single Bayesian model with nodes "Word sequence (w)" -> "Segmented phone
sequence (psi)" -> "Phone sequence (phi)", a hierarchical Pitman-Yor word LM (Theta) and a
pronunciation dictionary (Delta) with a Dirichlet-process prior; word segmentation of the phone
string is latent. Inference is collapsed Gibbs sampling over a WFST composition with a modified
composition that accumulates input labels.

Setup and caveats (Sec. 4): "Experiments were performed using the WSJ corpus and the CMU
dictionary. **As the phone transcript, true phone labels were used.** The number of phones was
39 and lexical stress was not used." "During the sampling, the vocabulary was fixed so that no
new word was generated with unknown spelling." A partial dictionary is assumed given.

Scale and cost, small setting (Sec. 5): 100 phone utterances + 100 disjoint word-text utterances,
**vocabulary 849**, pronunciations given for 70% (594 words), word bigram, ppl 30.8. WER after
40 epochs and per-epoch time on a Xeon X5650: GIBBS (exact) 9.6% at 37 min/epoch; VitBeam500
16.3% at 2.1 min; VitBeam1k 15.0% at 4 min; VitBeam1kP6 14.6% at 1.3 min; VitBeam10k 9.4% at
99 min. Baseline (G2P-augmented dictionary, no learning) 23.0%.
Large setting: full WSJ, 8000 word-transcript utterances + 2796 disjoint phone-transcript
utterances, **vocabulary 12.5k**, word trigram, ppl 200.8, OOV rate 2.4%. WERs (Table 3), 15% /
30% of pronunciations missing: G2P baseline 16.7 / 21.1; VitBeam 14.5 / 24.1 at epoch 5;
VitBeam+G2P 8.9 / 13.5.

Two things to carry: (i) the beam width is the entire quality/cost trade-off - beam 500 loses
~7 points WER against exact sampling, beam 10k recovers it at 27x the cost; (ii) the method
needed the G2P-initialised dictionary to beat the G2P baseline at 12.5k vocabulary, and plain
VitBeam at 30% missing was *worse* than the G2P baseline (24.1 vs 21.1).

### NOT READ (time; listed for completeness)
- Liu, Chen, Lee, Lee, "Completely Unsupervised Phoneme Recognition by Adversarially Learning
  Mapping Relationships from Audio Embeddings", Interspeech 2018 - cited by every source above
  as the phone-level GAN origin; from the secondary descriptions in 1.4/1.7/1.11 it is
  phone-level with no lexicon, but NOT verified from full text.
- Chung, Weng, Tong, Glass, "Unsupervised Cross-Modal Alignment of Speech and Text Embedding
  Spaces", NeurIPS 2018 - NOT READ.
- Ni et al. 2022 (character/letter-level unsupervised ASR) - NOT READ; cited in 1.7 as
  "letter-based unsupervised training [Liu et al. 2022; Ni et al. 2022]".
- Wang, Hasegawa-Johnson, Yoo, "A Theory of Unsupervised Speech Recognition", ACL 2023 - NOT
  READ; cited by 1.11 as the prior GAN-based theory with global-optimum conditions.
- "Zhu et al. 2024+ on decipherment with word priors" - NOT FOUND. Two targeted searches
  returned no paper matching this description. Either the reference is misremembered or it is
  not indexed under these terms. UNVERIFIED.
- Aldarmaki et al., "Unsupervised Automatic Speech Recognition: A Review", Speech Communication
  2022 - NOT READ (would be the place to check the "nobody does this" claim systematically).
- Chu, Fang, Knight, "Learning to Pronounce Chinese Without a Pronunciation Dictionary",
  EMNLP 2020 - NOT READ; cited by Klejch as a decipherment application.

---

## 2. Direct answers

### (a) Is there precedent for the lexicon inside the marginalised objective, and at what scale?

**Yes, but thinly, and only outside the wav2vec-U mainstream.**

Three precedents, in decreasing closeness to our setting:

1. **Klejch et al. 2022** is the only published *speech* system that composes a lexicon and a
   word trigram into the machine over which the training objective is marginalised
   (Baum-Welch on X ◦ (L ◦ A) ◦ G). Scale: **100k-word training LM / 300k-word inference LM,
   word trigram**, but only after a character-LM curriculum, with pruning at every stage, and on
   only 20 minutes of speech. Caveat: the terminal alphabet is *graphemes*, so their "lexicon" is
   a spelling lexicon, and the phone->grapheme table is what is learned. Structurally it is our
   word-trie x word-LM-history composition.
2. **Nuhn & Ney 2014** is the scaling law. Exact forward-backward EM over a word-typed state
   space with an n-gram LM is **intractable above ~200 word types** (224.88 h at V=200, bigram).
   Beam EM scales O(NV) and is "too slow to be used"; the only thing that worked at V=3,661 with
   a 3-gram was **preselection search**, O(N), with B=100, B_LM=50, B_lex=5, and it cost 19.68 h
   and lost essentially nothing against beam (90.92 vs 91.16 accuracy) at 15x the speed.
3. **Shinozaki et al. 2017** puts the lexicon in the objective as a latent variable (joint
   Bayesian LM + dictionary, collapsed Gibbs over a WFST). Scale: 849-word and 12.5k-word
   vocabularies, word bigram/trigram, 39 phones. But the phone side is **gold phone transcripts**,
   and a partial dictionary (70%/85%/70% of entries) is given, so it is not evidence that the
   objective survives acoustic-model-grade phone uncertainty.

**No one has composed a pronunciation lexicon into an unsupervised *acoustic* training lattice
at open vocabulary.** The closest infrastructure (EURO's k2 HLG, wav2vec-U's PyKaldi WFST) is
deliberately kept on the decoding side. This is a genuine gap, not a settled negative - but see
(b) before reading the gap as an opportunity.

### (b) Known failure modes

Four, each reproduced by more than one group:

1. **Collapse to frequent patterns.** Named independently by wav2vec-U 2.0 ("consistently
   producing the most common n-grams regardless of the input speech"), by wav2vec-U 1.0 (the
   vocabulary-usage term exists to detect "degenerate models which output fluent but trivial
   transcriptions"; L_pd exists to prevent it), and **measured at the word level** by Ni et al.
   2025, where for K=2048 and K=4096 "the error counts for words in the later word bins even
   exceed the occurrence counts" - the model inserts frequent words in place of rare ones. A
   stronger word-level prior is the thing that *causes* this, not the thing that cures it. Note
   that wav2vec-U 2.0's chosen antidote was an **audio-side** reconstruction anchor, not a
   text-side term.
2. **Length exploits.** Klejch: "decipherment tends to produce a lot of deletion errors", fixed
   with a deletion penalty at decode. wav2vec-U: model selection deliberately uses the
   *unnormalised* LM log-probability sum, "so that sequences score high under the language model
   but are not too long", plus an entropy clamp because "trivial solutions may achieve very low
   entropy". REBORN: R_len and R_edit exist solely to stop the perplexity reward from rewriting
   the transcript. Three groups, three different guards, same disease. Our lam_lm length-exploit
   rule and lm_prior_norm="units" are the same family of fix; a word-LM term in the DP will need
   its own per-unit-frame normalisation and an explicit anti-deletion guard.
3. **OOV / the unbounded rare-word tail.** Ni et al. 2025 had to prune LibriSpeech to a closed
   top-K vocabulary and still degraded at K=4096. Wang et al. 2025/2026: "the vocabulary of rare
   words is effectively unbounded, making coverage and generalization much harder." Klejch
   handled it only by pruning to the 100k/300k most frequent words and mapping the rest to <unk>
   during LM training. No one has a principled OOV path in a word-constrained objective.
4. **Brittleness / initialisation sensitivity.** Klejch failed outright on 3 of 7 languages
   (Swahili WER 105.3, Swedish 93.5, Hausa 70.8) and needed 50 random restarts at the bigram
   stage. Nuhn & Ney needed lambda=0.9 smoothing every E-step or zero-probability pairs could
   never recover. Both are consequences of a sharp prior on a badly initialised mapping - exactly
   the regime a lexicon term creates.

### (c) Tractable relaxations actually used

1. **Prior curriculum, weak to strong.** Klejch: character bigram -> character 5-gram -> word
   trigram + lexicon, with re-smoothing (alpha = 0.9) at each transition, because the full
   composition "is not feasible to use from the beginning". This is the single most transferable
   recipe for us: turn the lexicon on late, and only after the phone mapping is already roughly
   right under the trigram.
2. **Preselection / beam pruning of the forward-backward lattice.** Nuhn & Ney: B=100 hypotheses
   per position, expanded only by the B_LM=50 best LM successors and the B_lex=5 best lexical
   successors. Turns O(N V^n) into O(N). Shinozaki: Viterbi beam 500/1k/10k inside Gibbs, with
   the whole quality/cost curve reported. A GPU-batched analogue (top-k over trie states per
   frame) is the obvious port; exact marginalisation over a 100k-word trie x trigram history is
   not on the table.
3. **Three-way composition** X ◦ (L ◦ A) ◦ G (Allauzen & Mohri, CIAA 2008), because rebuilding
   L◦A◦G "after every training epoch" is the bottleneck when the learned table changes each epoch.
4. **Smaller units instead of words**: syllables (SylCipher, 2025/2026, "the number of syllables
   is finite, which reduces the long-tail token problem", 21.8% CER matched on LibriSpeech,
   beating word-level JSTTI); word pieces / characters (wav2vec-U self-training, Ni et al. 2022);
   whole-word closed vocabulary (Ni et al. 2025, K<=2048).
5. **Iterative HMM refinement** as a surrogate for a hard constraint: Chen et al. 2019
   (GAN -> HMM -> relabel -> GAN), wav2vec-U's "+ HMM + HMM" self-training rows (Librispeech
   dev-other 17.8 -> 14.6 -> 14.1). The word LM enters through the pseudo-labels, one generation
   behind, instead of through the gradient.
6. **Corpus-level statistic matching instead of per-utterance marginalisation**: ODM (Yeh 2019,
   top-10k 5-grams, average inside the log, batch 20k) and skipgram/positional-unigram matching
   (Wang 2024). Cheaper, but the ESPUM ablation says the higher-order statistics were the *least*
   useful part.
7. **Silence as a segmentation anchor.** Klejch maps silence phones only to silence or a word
   boundary, explicitly "because silence prunes the space of possible word segmentation". We have
   SIL in the DP already; using it to force word-boundary states would cut the trie search space
   at no modelling cost.

---

## 3. What this changes for the pre-registration

The design is not unprecedented, but the one speech precedent (Klejch 2022) succeeded only with
a prior curriculum, aggressive pruning, re-smoothing, silence-anchored word boundaries, 50
restarts, and an explicit deletion penalty - and still failed outright on 3 of 7 languages. The
scaling evidence (Nuhn & Ney 2014) says exact marginalisation over a word-trie x word-LM-history
state space dies above ~200-500 word types; anything at our vocabulary needs preselection-style
pruning as a first-class part of the design, not an optimisation added later.

The strongest argument *against* is that the field's measurement of the disease we are treating
(wav2vec-U 2.0: the generator satisfies the text prior "by consistently producing the most common
n-grams") led two independent groups away from a stronger text-side prior: wav2vec-U 2.0 added an
audio-side reconstruction anchor, and Ni et al. 2025 showed that a word-level prior actively
concentrates error onto rare words (error counts exceeding occurrence counts in the low-frequency
bins). Our 2.3 vs 1.4 nats/phone gap is a *discrimination* measurement on gold vs private-code;
it establishes the lexicon carries information the trigram does not, and says nothing about
whether the DP can be steered by it without collapsing onto frequent words. Those are different
quantities and the literature does not let us convert one into the other.

Concretely, the pre-registration should carry:
- the lexicon term switched on only after a trigram-only warm start (Klejch's curriculum), not
  from step 0;
- a declared pruning scheme with its own budget (B, B_LM, B_lex analogues), reported as a
  quality/cost curve like Shinozaki's beam table, not a single operating point;
- re-smoothing of any learned table against uniform at the transition (alpha = 0.9 is what two
  independent groups used);
- a per-unit-frame normalisation of the word-LM term and an explicit anti-deletion guard, because
  three groups independently needed one;
- a **frequency-stratified error read** as a pre-registered output, binned by word frequency, so
  the frequent-word collapse is visible in the first result and not after the phase closes;
- a null: the same DP with the lexicon term on a structure-destroyed lexicon (shuffled
  pronunciations), since a rising objective under a sharp prior is not evidence of content.

What would settle the open question: nobody has run the ablation "same acoustic model, same DP,
trigram-only prior vs trigram + marginalised lexicon, plain WER on the same decode". Klejch has
no trigram-only arm; wav2vec-U has no in-objective-lexicon arm; Yang et al. 2026 has no
experiments at all. That ablation is ours to run and it is the thing the literature is missing.

---

## Sources (all verified from full text unless marked NOT READ)
- https://www.isca-archive.org/interspeech_2022/klejch22_interspeech.pdf
- https://arxiv.org/pdf/2105.11084
- https://arxiv.org/pdf/2204.02492
- https://arxiv.org/pdf/1812.09323
- https://arxiv.org/pdf/1904.04100
- https://arxiv.org/pdf/2211.17196
- https://arxiv.org/pdf/2406.08380
- https://sls.csail.mit.edu/publications/2024/LimingWang_ICASSP-24.pdf
- https://arxiv.org/pdf/2510.03639 , https://arxiv.org/pdf/2608.22907
- https://arxiv.org/pdf/2402.03988
- https://arxiv.org/pdf/2603.02285
- https://aclanthology.org/P14-2123.pdf
- https://www.isca-archive.org/interspeech_2017/shinozaki17_interspeech.pdf
