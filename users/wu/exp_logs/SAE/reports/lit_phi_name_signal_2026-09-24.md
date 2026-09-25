# Literature: strengthening the name-sensitive signal for phi without labels (2026-09-24)

Role: literature. Question from the dispatch: how can the signal that tells phi *which phone name* a
cluster of SSL units carries be strengthened without labels, for the explicit-duration HSMM phi
(500 k-means units at 50 Hz to 40 phone symbols, emission m(u | s, duration bucket, position),
d_min 2, D 25 / SIL 50, frozen phone trigram prior, EM on marginal likelihood, S = held-out NLL/frame)?

Conventions. "Verified" means read in the full text fetched this session (PDF converted to text).
"Prior report" means the claim was verified in an earlier campaign report and is only cited here,
not re-read. "(mine)" marks my own inference. UNVERIFIED marks sources I could not read.
No GAN method is proposed; GAN papers appear only as background for a theory or a number.

## Verdict

The literature does not settle how to strengthen the naming signal in this regime (noisy clusters,
free segmentation, frame-level emission, fixed trigram prior). What it does establish, and what it
changes for the plan:

1. **Rate is a confound in the project's own "trigram ties gold" reading (mine, from project findings
   3 and 6).** A per-frame trigram cost equals (segments/s) x (nats/phone) / 50. The random-init phis
   emit 6.8-7.6 seg/s against 10.0-10.5 for basin phis and 11.9 for the reference. If their per-frame
   trigram cost ties gold, their per-phone trigram cost is about 1.4-1.7 times gold's. For
   illustration, a trigram per-phone cost of about 2.2 nats (perplexity about 9.5) at 11.9 phones/s
   gives 0.53 nats/frame; a near-uniform 3.7 nats/phone (log 40) at 7.2 phones/s gives 0.53
   nats/frame as well. So "the trigram cannot tell gold from EM" may really be "under-segmentation
   buys back the trigram penalty by emitting fewer LM factors". This is checkable from existing
   dumps without any training (see "Analyses first").
2. **At fixed clusters the objective already prefers correct names (project finding 4: the oracle 1:1
   rename lowers the key objective by 0.42-0.57/frame); the key search does not find that rename.**
   In the Yin et al. 2019 model-error versus search-error framing (prior report), that is search error.
   The decipherment literature solves exactly this subproblem (renaming a substitution cipher under
   an n-gram LM) reliably for noise-free text, with beam search whose width must grow with LM order
   (Nuhn, Schamper & Ney 2013). It does not reach our noise level, and a 1:1 rename cannot undo the
   merges (11-19 phones with no symbol).
3. **Unigram/occupancy matching is the weakest lever on the evidence**: character-unigram matching
   alone left 71% error where bigram matching gave 10% (Liu, Chen & Deng 2017); a generative model with
   its label prior fixed already carries proportion information and still underperforms (Mann &
   McCallum 2010); unigram decipherment is solved by frequency sorting (Nuhn & Ney 2013), so it
   carries no information beyond rank order. With our frozen trigram prior, symbol occupancy is likely
   already near the LM unigram, so an occupancy constraint is likely inactive (mine; checkable).
4. **Higher-order LMs and word knowledge help ciphers when search is wide enough (reproduced across
   groups, text only); the one speech result with moment matching finds higher orders hurt (ESPUM,
   single paper); the project's own lambda up-weighting of the trigram was null.** These are
   compatible: higher order helps only when the search/clustering can exploit it.
5. **No published work uses phone-specific durations (Klatt; Crystal & House) or speaking-rate priors
   to fix unsupervised phone naming** (search negative). Universal knowledge does break the coarse
   consonant/vowel label symmetry in text (Knight et al. 2006 syllable model), and nothing finer.
6. **Emission-versus-prior weighting has published evidence in both directions**: text decipherment
   raises the channel weight (P(c|p)^3, Knight et al. 2006), while supervised speech lowers the
   acoustic weight because frames are dependent (Wegmann & Gillick 2010), and in unsupervised
   segmentation the frame-count exponent sets over- versus under-segmentation (Kamper et al. 2016/2017).
   Nobody has tested it for EM naming of phones.

Ranked levers (strength of the evidence that they would raise the naming signal *here*):
(1) rate-controlled diagnostics before any spend: strong case, zero risk; (2) rename search over the
n-gram at fixed clusters (beam/QAP-style, trigram then 4-5-gram, wide beam): strong in text, fits
project finding 4, but blocked by merges and untested at our noise; (3) expected-segment-rate
constraint from a general-knowledge speaking rate: moderate and indirect; (4) higher-order phone LM or
word lexicon inside EM, as a curriculum: strong in ciphers, negative in ESPUM and in the project's
lambda test; (5) posterior-regularised bigram/trigram-statistic matching: plausible and cheap, but
single-paper support and possibly inactive under a fixed prior; (6) occupancy matching: evidence
against; (7) emission scaling or segmental emissions: contradictory; (8) phone-duration or C/V
universals: no speech evidence, coarse classes only in text.

## Q1. Is the naming identifiable, and what governs accuracy?

**Wang, Hasegawa-Johnson & Yoo, ACL 2023, "A Theory of Unsupervised Speech Recognition"** — verified
from full text at https://arxiv.org/pdf/2306.07926.
- Setup: a binary unit-to-phone map O = P(Y|X), many-to-one allowed; phones follow an N-gram Markov
  source. Theorem 1: O is unique iff the matrix of stacked positional unigram statistics (positions
  0, N, 2N, ...) has full column rank |X|. The assumptions require at least |X| distinct eigenvalues of
  the N-gram transition matrix and a start distribution that projects onto at least |X| eigen-blocks.
  Identification therefore comes from transient/positional statistics; with a stationary start it
  fails (mine, from the assumption). The smallest singular value grows with sequence length L, and
  Theorem 3 gives a sample complexity that scales with it.
- Their synthetic experiments (GAN/MMD training, background only) show a phase transition: recovery
  succeeds once the number of distinct eigenvalues exceeds the number of speech units, and the
  number required grows with the unit count.
- Limits stated by the authors: quantized units are assumed to preserve all linguistic information,
  and "sufficiently reliable phoneme boundaries" are assumed and kept fixed. There is no channel noise
  and no free segmentation, which are the two things that differ in our setup.
- For us (mine): theory says the naming is identifiable in principle; it does not bound accuracy when
  clusters are impure and segmentation is learned jointly.

**Ravi & Knight, EMNLP 2008, "Attacking Decipherment Problems Optimally with Low-Order N-gram
Models"** — verified at https://aclanthology.org/D08-1085.pdf.
- Exact integer programming (no search error), so the remaining errors are model errors. English
  letter entropies 4.19/3.51/2.93 bits/char for 1/2/3-gram. Shannon unicity U = H(K)/(A - B) with
  H(K) = 88.4 bits gives 173/74/50 letters.
- Empirically worse than Shannon ("a bit too rosy"): a 1-gram model does not solve ciphers at 173
  letters; a 3-gram model works from about 64 letters. At 414 letters, EM with a 2-gram LM had 10%
  error against 0.5% for IP; at 52 letters EM 85% against IP 21%.
- "Uncertainty is only loosely related to accuracy — even if we are quite certain about a solution,
  it may still be wrong." IP runtime at length 32: 0.01 s / 50 s / 450 s for 1/2/3-gram.

**Nuhn & Ney, ACL 2013, "Decipherment Complexity in 1:1 Substitution Ciphers"** — verified at
https://aclanthology.org/P13-1060.pdf. Unigram decipherment is a linear sum assignment problem,
solved by sorting frequencies (O(n log n)). Bigram decipherment is a quadratic assignment problem and
NP-hard (reduction from TSP). Consequence for us (mine): renaming at fixed clusters under a trigram is
a hard combinatorial search, and local moves (single swaps, EM) are expected to stall.

**Nuhn, Schamper & Ney, ACL 2013, "Beam Search for Solving Substitution Ciphers"** — verified at
https://aclanthology.org/P13-1154.pdf.
- Table 1 (1:1 letter ciphers, length 128, space symbol fixed), SER: 3-gram 1.31% exact (A*, 19,700 s)
  and also 1.31% at beam 100k (47.7 s); 4-gram 0.06% and 5-gram 0.06% at beam 100k; 6-gram 0.05% at 100k.
- At beam 10: 3-gram 25.27% against 6-gram 64.77%. "If the beam size is not large enough, the
  decipherment accuracy decreases when increasing the language model order."
- Zodiac-408 (homophonic, Table 2): 6-gram SER 1.96% at beam 10M (167,181 s on 128 cores); 4-gram
  16.18% at 10M; 5-gram 3.19% at 1M.

**Hauer, Hayward & Kondrak, COLING 2014, "Solving Substitution Ciphers with Combined Language
Models"** — verified at https://aclanthology.org/C14-1218.pdf. Score = chi log P_char +
(1 - chi) log P_word, with chi tuned on development data. NYT, length 64: 0.03% error with spaces,
7.85% without spaces (MCTS); a noisy condition (about log2 n corrupted letters, roughly 9% at 64 and 3%
at 256) gives 16.66 / 5.20 / 2.73% (beam, lengths 64/128/256). Word boundaries matter a lot. The noise
tested is far below ours. Monoalphabetic ciphers only.

**Kambhatla, Mansouri Bigvand & Sarkar, EMNLP 2018** — verified at
https://aclanthology.org/D18-1102.pdf. A neural char LM in beam search: Zodiac-408 SER 1.47%, and
1.22% with frequency-matching heuristic, at beam 1M, against 2% for a 6-gram at 10M; 1:1 length 128
0.01% at beam 100k. Gains are modest and mostly in beam efficiency. The frequency-matching heuristic
*hurts* on Beale (4.98 to 5.50%; 41.67 to 48.33% at beam 10k).

Prior reports, not re-read: Ravi & Knight 2011 (EM on a homophonic cipher 32.6% accuracy, Bayesian
95.2%, word + 3-gram 100%); Ravi & Knight 2009 phonetic decipherment (2-gram, then 3-gram, then word
LM: error 100, 73.6, 57.2 NED); Nuhn et al. 2014 (observations per cipher symbol 4.18 solved, 1.74 and
2.34 failed); Yin et al. 2019 (cluster noise 0.22-0.44).

Unicity for our problem (mine): a 40-symbol permutation has log2(40!) = about 159 bits. With a phone
trigram per-phone perplexity of 8-15 (the redundancy is log2 40 minus log2 perplexity, 1.4-2.3
bits/phone), U = 70-115 phones. A many-to-one 500-to-40 key (500 log2 40 = about 2,660 bits) needs
about 1,150-1,900 phones, i.e. 1.5-3 minutes of speech. **Corpus length is not the constraint;
cluster noise and the free segmentation are**, and neither is covered by unicity or by Wang 2023.

## Q2. A stronger prior inside EM

For (text ciphers, reproduced across groups):
- Higher order helps if the search is wide enough: Nuhn/Schamper/Ney Table 1 (above); Ravi & Knight
  2008 (3-gram solves at 64 letters where 1-gram fails at 173); Kambhatla 2018.
- Word knowledge helps: Hauer 2014 (combined char + word LM, with a tuned weight); Ravi & Knight 2011
  and 2009 (prior report).
- Curricula that move up in order: Knight et al. 2006, verified at https://aclanthology.org/P06-2065.pdf
  (bootstrapping C/V then S/N/V: "bootstrapping is good for dealing with too many parameters");
  Klejch 2022 goes 2-gram to 5-gram and then a word trigram, *swapping* priors rather than adding them
  (prior report).

Against, or costs:
- Knight et al. 2006 (verified): "EM initially locks onto the correct theory, but task performance
  degrades as it tries to make the ciphertext decoding fit the expected bigram frequencies. Better
  source models do not suffer much." This is the text analogue of our in-basin drift (project finding
  5: PER 0.19 to 0.35 while S falls).
- ESPUM (Wang, Hasegawa-Johnson & Yoo, ICASSP 2024), verified at https://arxiv.org/pdf/2310.02382,
  Table 3 (TIMIT validation PER; K=128 wav2vec2 layer-14 units; moment matching of positional unigram
  and skip-gram statistics): bigrams only 71.6; uni+bi 39.2; uni+bi+tri 38.4; uni+bi+4-gram 40.0;
  uni+4-gram 45.0; uni+5-gram 77.9; uni+bi+tri+5-gram 41.8. Under moment matching, orders above 3
  degrade. Single paper, different objective (matching, not likelihood).
- Beam width must grow with order (Nuhn Table 1); a 6-gram Zodiac solve needed a 10M beam and 46 CPU-h.
- Project evidence (finding 3): trigram up-weighting (lambda 1-3) does not make gold rank first.
  The prior report lit_phone_lm_with_word_lm (2026-09-23) found that the project's word-term gain was
  matched by a shuffled-pronunciation null, and that no published objective adds a phone and a word LM
  at full weight; Hori-style weight cancellation and Klejch's swap are the precedents.
- (mine) If the rate confound above holds, a stronger LM raises the per-phone penalty, but the phi can
  still escape it by emitting fewer segments. A stronger prior without rate control may therefore
  deepen under-segmentation rather than fix names.

## Q3. Posterior regularization, generalized expectation, EODM, ESPUM-style matching

**Ganchev, Graça, Gillenwater & Taskar, JMLR 2010, "Posterior Regularization"** — verified at
https://www.jmlr.org/papers/volume11/ganchev10a/ganchev10a.pdf.
- PR changes only the E-step (q proportional to p_theta exp(-lambda . phi)). Updates stay monotone,
  and constraints that factorize over the model's cliques keep the DP cost.
- For: a bijectivity constraint fixed the "garbage collector" failure in word alignment
  (translation-table entropy 2.0-2.6 bits dropped to about 0.6; Hansard, 100k sentences; gains in 6
  languages).
- Against: l1/l-infinity posterior sparsity for unsupervised POS (PTB17, PT-CoNLL, BulTree) beat EM on
  1-many and MI, but "does not always outperform VEM" on 1-1, and "Sparse tends to spread the nouns
  over 4 different hidden states". That is exactly our frequent-phone split, and PR did not fix it.
  Numbers appear only in figures.
- Cost: GE-style gradients need feature covariances, which squares the DP cost; PR does not.

**Mann & McCallum, JMLR 2010, "Generalized Expectation Criteria"** — verified at
https://www.jmlr.org/papers/volume11/mann10a/mann10a.pdf. Label regularization matches the model's
expected label proportions to a target. It is robust to noisy proportions. Its degenerate solution
is every instance predicting the prior distribution. Key point against: "the naive Bayes classifier
has its label prior fixed to the input distribution as it is subsequently trained with EM, yet it
typically fails to reach the same level of performance as achieved with the GE methods". A
generative model with a fixed prior already carries the proportions and still fails. Their setting is
semi-supervised (at least one labelled example per class), so this is not a pure-unsupervised test.

**Liu, Chen & Deng, NIPS 2017 (EODM/SPDG)** — verified at https://arxiv.org/pdf/1702.07817.
- OCR, 29 classes, 153,221 characters, Tesseract segmentation with verified 0.9% error. Supervised
  4.63%, unsupervised 9.59%.
- LM order (Fused-LM, Table 2): 1-gram 71.25%, 2-gram 10.33%, 3-gram 9.21%. Unigram matching alone
  fails and bigram does most of the work.
- A mode-seeking cost collapses to the majority guess (83.37%); SGD with 10k batches reaches 56.48%;
  an out-of-domain 3-gram gives 10.17%.
- Their segmentation is essentially given, so free segmentation is not tested.

**ESPUM** (above): the positional unigram is "crucial"; bigrams alone give 71.6 PER; uni+bi gives
39.2. Boundary F1 is about 86-87.6 with a soft aligner distilled from a teacher, so segmentation is
not free.

Prior reports: Yeh et al. 2019 EODM (5-gram, with a segment rate term, 41.6 PER unmatched TIMIT;
smoothness term essential).

What this means for phi (mine):
- A PR occupancy constraint (expected count of symbol s = LM unigram of s) is a per-symbol segment
  bias found by dual ascent. It is cheap. It is probably inactive: under a frozen trigram prior, symbol
  usage is already pulled toward the LM unigram, and our failure (a frequent phone spread over 3-4
  symbols whose summed prior mass matches) can satisfy occupancy with wrong names. That is the Mann &
  McCallum fixed-prior case.
- A PR constraint on symbol bigram expectations factorizes over the trigram DP transitions and is also
  cheap. It pushes only where the posterior bigram statistics deviate from the LM's.
- Whether either constraint would be active is a measurement on existing phis, not a literature
  question.

## Q4. General phonetic knowledge as a symmetry breaker

- **Knight, Nair, Rathod & Yamada, COLING/ACL 2006** — verified at https://aclanthology.org/P06-2065.pdf.
  Knowledge-free C/V EM (uniform trigram over C, V, SPACE, learned source and channel) "correctly
  clusters letters into vowels and consonants, but assigns exactly the wrong labels!" A universal
  syllable-type inventory (V, CV, CVC, ...; onset maximization) fixed the labels. To work it also
  needed the syllables-per-word count "fixed and uniform, rather than learned. This prevents the
  system from making analyses that are too short", plus random restarts selected by likelihood.
  Sonority then split C into sonorous and non-sonorous, but only when bootstrapped from the C/V model
  ("EM hijacking the extra symbols" otherwise). Text, Spanish/Latin, 2-3 classes.
- **Kim & Snyder, ACL 2013** — verified at https://aclanthology.org/P13-1150.pdf. Letters in 503
  languages. C/V accuracy: EM baseline (Knight 2006 trigram HMM, 10 restarts, accuracy under a
  gold-selected 1-1 mapping) 93.37; SYMM (type-level, one tag per letter type) 95.99; MERGE 97.14;
  CLUST (uses labelled other languages) 98.85. C/V/N: 74.59 / 80.72 / 86.13 / 89.37. Text only; coarse
  classes; naming is done by a gold mapping or by priors from labelled languages.
- **Thaine & Penn, 2017** — verified at https://aclanthology.org/W17-4112.pdf. The second singular
  vector of a letter co-occurrence matrix splits V/C: 97.45% type (macro) and 99.39% token accuracy.
  Sukhotin's algorithm 94.34% type. Text only.
- **Phone-specific durations** (Klatt 1976; Crystal & House 1988) and **speaking-rate priors**: I
  found no paper that uses them to name or disambiguate unsupervised speech units. Web searches this
  session were negative. The primary duration sources were not read: UNVERIFIED.
- (mine) The general-knowledge route is supported only at the level of 2-3 classes, and only through
  sequence structure (syllable templates), not through durations. Duration classes (for example, long
  vowels and diphthongs against short stops and flaps) could at best split symbols into coarse groups;
  within-group naming would still depend on the n-gram. Knight's "fixed syllable count prevents
  analyses that are too short" is the one published instance of a length/rate constraint rescuing
  label assignment, and it is text.

## Q5. Emission versus prior scale; frame-rate generative training

- **Johnson, EMNLP 2007, "Why doesn't EM find good HMM POS-taggers?"** — verified at
  https://aclanthology.org/D07-1031.pdf. EM HMMs "assign a roughly equal number of word tokens to each
  hidden state", while gold tags are skewed. EM with 50 states: 1-to-1 0.40, many-to-1 0.62; VB
  (0.1, 0.1): 0.47 / 0.50. Annealing never reached higher likelihood. Transitions were learned (no
  fixed LM), and the task is POS, not speech. The pattern matches project finding 2 (frequent phones
  take 3-4 symbols; 11-19 phones get none).
- **Wegmann & Gillick, arXiv 2010 (1003.0206), "Why has (reasonably accurate) ASR been so hard to
  achieve?"** — verified at https://arxiv.org/pdf/1003.0206. WSJ 5k, word-internal triphones,
  1,500 tied single-Gaussian states. Real test data: 18% WER. Frames resampled to be independent given
  the alignment: 0.5%. Data simulated from the model: 0.2%. LM scale 16 compensates for dependent
  acoustic scores; on model-simulated data the correct scale is 1, and MMI "compensate[d] for the
  over-weighting of the LM scores". Table 2: repeating the first real frame of each state region
  (average 3.4 frames) raises WER only 17.7 to 23.8, against 0.6 to 4.4 for resampled frames. The
  extra real frames within a region carry little independent information. Supervised decoding,
  Gaussian features, not discrete units.
- **Kamper, Jansen & Goldwater, TASLP 2016** (https://arxiv.org/pdf/1603.02845) and **CSL 2017**
  (https://arxiv.org/pdf/1606.06950), both verified. The segment embedding likelihood is raised to the
  number of frames j; "another interpretation is to see j as a language model scaling factor ...
  without this factor, severe over-segmentation occurred". With it, the CSL 2017 systems
  under-segment relative to syllable boundaries (more deletions, lower boundary recall) while cluster
  purity is better. Word-level, one group.
- **Kamper, Livescu & Goldwater 2017, ES-KMeans** (https://arxiv.org/pdf/1703.08135, verified): the
  objective is the sum of len(x) ||x - mu||^2, so segment scores are weighted by duration.
- **Lee & Glass, ACL 2012** (https://aclanthology.org/P12-1005.pdf, verified): joint DP-HMM
  segmentation, boundary P/R/F 76.4 / 76.2 / 76.3 (not under-segmented). Pre-segmentation constrains
  boundaries and "allows the model to capture proper phone durations, which compensates the fact that
  we do not include any explicit duration modeling".
- **Knight et al. 2006** (verified): decoding with P(p) P(c|p)^3 (channel *up*-weighted) cut errors
  from 62 to 42 in a 417-letter cipher, because channel probabilities are "too bunched up". Yin 2019
  used P(E)^3 (prior report).
- Prior reports: in supervised MMI (Woodland & Povey; Michel 2019) the acoustic scale is the live
  knob and the LM scale is inert; wav2vec-U gives one vote per segment and a silence insertion rate.
- UNVERIFIED (not reached): Gillick, Gillick & Wegmann, ASRU 2011 ("Don't multiply lightly");
  Ostendorf, Digalakis & Kimball 1996 (segment models); Glass 2003 (segment-based recognition).

Disagreement to report as such: text decipherment strengthens the channel relative to the source,
while speech weakens frame-level acoustics relative to the LM. The two are consistent if one symbol
is one observation in text, whereas a 50 Hz frame stream counts each segment about 4-5 times with
highly dependent evidence (mine). Our frame-level emission already weights the channel by roughly the
mean duration, at or beyond Knight's cube. Neither literature tests EM phone naming. Down-weighting
emissions also stops EM from maximizing a likelihood, so S would no longer be comparable across arms
(mine). A prior report found deterministic annealing weak, and Johnson 2007 found annealing never
improved the likelihood.

## Established versus single-paper

- Established across groups, text only: higher-order n-grams reduce substitution-cipher error given
  enough search (Ravi & Knight 2008; Nuhn et al. 2013; Kambhatla et al. 2018; Hauer et al. 2014); word
  knowledge helps (Ravi & Knight 2009/2011; Hauer 2014); unigram statistics alone are insufficient
  (Nuhn & Ney 2013; Liu et al. 2017; Ravi & Knight 2008); EM balances or mislabels latent states
  (Johnson 2007; Knight et al. 2006; the Kim & Snyder EM baseline).
- Single paper: higher orders hurt under moment matching (ESPUM); the identifiability theorem (Wang
  2023, fixed segmentation); the frame-count exponent governs segmentation (Kamper et al., one group);
  frame dependence as the dominant ASR error source (Wegmann & Gillick).
- No paper found: phone durations or speaking rate used to name units; PR/GE tried inside a
  generative model with a frozen n-gram prior; rename search at our noise level.

## Analyses first (cheap, from existing phis; decide whether any lever can bite)

All of these use existing dumps or small readers; no labels enter training or selection, and gold is
used for analysis only.
1. For gold-init, basin and random-init phis, report the trigram NLL per *phone* and per *frame*,
   plus the segment rate. If the per-phone NLL separates gold from EM phis while the per-frame NLL
   ties, the name signal exists and the rate is what cancels it; the lever is then rate control, not
   a stronger LM.
2. For the same phis, report the decoded symbol unigram and bigram statistics against the LM's
   (KL/L1). If they match, occupancy and bigram PR would be inactive, and that settles Q3 for us
   without a training run.
3. At the finals of the stage-1 key search: rename search with the trigram, then a 4/5-gram, at a wide
   beam, measuring whether the objective's optimum coincides with the oracle rename. This separates
   search error from model error (Yin 2019 framing) and the effect of LM order. Merges limit any 1:1
   rename, so also report the identity reachable by the best many-to-one rename.

## What would settle it

- For rate: EM with the expected segment count constrained to a general-knowledge speaking rate (the
  campaign already uses 4.4 frames per phone, i.e. about 11 phones/s, which is within published
  read-speech ranges; the primary sources are UNVERIFIED here). Compare merge counts and identity
  against unconstrained EM from the same random starts, with a paired per-item read.
- For LM strength: the same comparison with the trigram against a 5-gram prior, at a controlled rate.
- For PR: only if analysis 2 shows a deviation from the LM's statistics.
- Background agents on rename moves and the relabel/decipherment reports (lit_decipherment_relabel,
  2026-09-24; lit_em_decipherment, 2026-09-23) already cover search moves, restarts and smoothing,
  which are not repeated here.
