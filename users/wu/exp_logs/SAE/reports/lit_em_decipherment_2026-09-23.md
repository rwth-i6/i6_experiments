# Literature read: what makes EM decipherment of speech units into phones work (no GAN, no labels)

Literature agent, 2026-09-23. This read feeds the L2-1 A11 amendment (exact-EM count-table phi) in
`SAE_4A_lexlat_v2.md`. Every numbered claim below was checked against the paper's full text at the URL
given. Anything I could not reach is marked **UNVERIFIED**. Arithmetic and inferences of my own are
marked **(mine)**. Project results are marked **(project record)** and are not literature.

## Verdict

No published work does what A11 does: closed-form EM on 50 Hz SSL k-means units into phones, with an
explicit-duration HSMM and a frozen phone LM. Every published EM decipherment success works at symbol
rate, meaning roughly one cipher token per plaintext token. The closest speech case, Klejch et al.
2022, deciphers the 1-best output of a **supervised** multilingual phone recogniser. The literature
therefore cannot say whether A11 will work. It does agree on what the working recipes share, and A11
departs from three of those shared elements.

1. **Iterations per restart.**
   - Published restarts run to near-convergence before they are ranked: 200 iterations in BK&K, 20
     per LM stage in Klejch, and Johnson 2007 reports accuracy moving by 5% after several hundred.
   - A11 stage A ranks its restarts after 10 iterations.
2. **Smoothing mass.**
   - Every working EM recipe smooths toward uniform by 10-26% of row mass (BK&K, Nuhn & Ney 2014,
     Klejch, Knight 2006).
   - Nuhn & Ney 2014 explain why: a pruned E-step can zero counts that then never recover.
   - A11 prunes (max_active 1000) and adds a 1e-3 pseudo-count per cell, which is about 0.02% of row
     mass (mine).
3. **Annealing schedule.**
   - Every published deterministic-annealing schedule converges, or runs several EM steps, at each
     temperature.
   - A11 moves tau 4 -> 1 in 4 single iterations.
   - The evidence that DA helps at all is weak or negative (Smith & Eisner 2004; Johnson 2007).

Within the constraints (no labels, no GAN, trigram or higher only, general-knowledge durations), the
best-supported next steps if the first round fails are:
- more iterations and restarts on a smaller subset;
- E-step smoothing;
- fixed durations (already an arm);
- coarse-to-fine bootstrapping, which means fewer units or fewer classes first;
- a stronger LM (word-level) only together with a wider beam.

Neural emissions, speaker conditioning and frequency-rank initialisation have no supporting evidence
in this regime.

## Per-source findings

### Speech decipherment

**Klejch, Wallington & Bell, Interspeech 2022** (https://arxiv.org/pdf/2111.06799)

(a) Setup:
- The cipher is the 1-best phone string of a supervised universal phone recogniser: a TDNN trained with
  LF-MMI on 110 h of six languages, decoded with a phone bigram. So the cipher runs at phone rate,
  roughly 1:1 with phones plus insertions and deletions. It is not frame-level k-means.
- The plaintext is graphemes. The model is a WFST, X o (L o A o G).
  - L is a one-state phone-to-grapheme substitution and deletion table, initialised to all
    substitutions, with unseen ones pruned.
  - A allows one insertion or deletion in a row.
- Training is full-batch Baum-Welch.
- There are 50 random restarts, all with a grapheme bigram LM; the best **training** likelihood is
  kept.
- LM order then rises 2 -> 5-gram at 20 iterations per order. L is pruned to the top 20 phones per
  grapheme after the bigram stage, then a word trigram LM is used (100k words; 300k at inference).
- The table is smoothed toward uniform (alpha 0.9) between stages.
- Data: 20 minutes of the **shortest** dev utterances per language; LMs from CommonCrawl.
- The bigram start is for compute, not optimisation. Quote: "using n-gram models with large contexts
  results in a big composition ... therefore it is not feasible to use them from the beginning.
  Hence we start with a simple bigram model."

(b) Numbers (GlobalPhone WER, Table 1):

| Language | Decipherment | + semi-supervised training | Oracle |
|---|---|---|---|
| BUL | 31.0 | 10.7 | 7.9 |
| CES | 49.0 | 16.5 | 12.9 |
| HAU | 70.8 | 30.3 | 12.2 |
| POR | 53.8 | 21.2 | 17.0 |
| SWA | 105.3 | 98.3 | 10.2 |
| SWE | 93.5 | 55.8 | 23.3 |
| UKR | 34.5 | 10.3 | 8.4 |

(c) Failures:
- GlobalPhone Swahili failed. It has beeps and long leading/trailing silence, and results stayed bad
  after both were removed. ALFFA Swahili reached 41.8 (wav2vec-U: 32.2). The authors conclude one
  needs "speech amenable to decipherment".
- Decipherment produces many deletions.
- I found no follow-up decipherment paper by these authors (arXiv and web search). The UK Speech 2022
  abstract is the same work.

Transfer to us: the optimisation recipe transfers. The cipher does not: theirs is supervised-derived
and runs at phone rate.

### Classical EM decipherment

**Berg-Kirkpatrick & Klein, EMNLP 2013** (https://aclanthology.org/D13-1087.pdf)

(a) Setup:
- A fixed character trigram LM backbone. It is uniformly interpolated over 1/2/3-grams, from Google
  N-grams plus about 2K words of the Zodiac killer's own letters, with the corpus mix weighted 0.9
  toward the Zodiac corpus.
- Only the emissions are learned, by EM, for 200 iterations per restart.
- Smoothing: +0.1 added to expected emission counts before each M-step. It was "important", and tuned
  on held-out synthetic ciphers.
- Initialisation: U[0,1], then normalised. A Dirichlet had "little effect so long as the distribution
  did not favor the corners of the simplex"; favouring the corners "deteriorated sharply".
- The restart with the best model score is selected; decoding is posterior decoding.

(b) Numbers:
- Zodiac-408 (408 tokens, 54 symbols): 1 restart gives 18% mean accuracy; 100 restarts give 90%.
  There is no gain beyond 100.
- Gold-key initialisation reaches log-likelihood -1467.4 at 92% accuracy. The best of 1M restarts
  reaches -1466.5 at 89%.
- Synth-340 (340 tokens, 63 symbols): +9% accuracy from 100 to 100K restarts.
- 100 synthetic ciphers at 10K restarts each: mean 75%, minimum 36%.

(c) Failures:
- Zodiac-340 gave nonsense even at 1M restarts. The authors argued it is not a row-order homophonic
  cipher.
- Oranchak, Blake & Van Eycke (arXiv 2403.17350, read) confirm Z340 was a transposition plus
  homophonic cipher, solved 51 years on.
- The authors warn that likelihood tracks accuracy only across large gaps: "small improvements to high
  likelihoods ... may not" track accuracy.
- Smoothing mass, (mine): 408 tokens over 26x54 = 1404 cells is about 0.29 counts per cell, so 0.1
  per cell is about 26% of row mass.

**Ravi & Knight, ACL 2011** (https://aclanthology.org/P11-1025.pdf)

(a) Setup:
- A Bayesian CRP channel with type sampling and a word+3-gram LM, with the source fixed.
- Sparse channel prior beta 0.01.
- The initial key maps frequent letters to frequent symbols.
- 5000 sampling passes, with the temperature annealed linearly from 10 to 1.
- The EM baseline decodes Viterbi with the channel cubed.

(b) Accuracy:

| Method | LM | Simple (414) | Homophonic with spaces (414) | Zodiac-408 |
|---|---|---|---|---|
| EM | 2-gram | 83.6 | 30.9 | - |
| EM | 3-gram | 99.3 | 32.6 | 0.3 (28.8 with 100 restarts) |
| Bayesian | 3-gram | 100 | 95.2 | 23.0 |
| Bayesian | word+3-gram | 100 | 100 | 97.8 |

- Sparse prior beta 0.01 vs 1.0: 97.8 vs 24.0.
- Type vs point-wise sampling: 97.8 vs 14.5.

(c) Failures: EM with a letter trigram collapses on homophonic ciphers (32.6) even though it solves
simple ones (99.3). A 98-letter simple cipher gives EM only 41%.

**Ravi & Knight, NAACL 2009, phoneme decipherment for transliteration** (https://aclanthology.org/N09-1005.pdf)

This is the closest classical analogue to phone decipherment.

(a) Setup:
- English phonemes (about 40) are the plaintext; Japanese phonemes are the cipher.
- The channel allows 1-to-1 and 1-to-2 mappings, is uniformly initialised, and is trained by EM
  (Carmel, at most 200 iterations). The channel is cubed at decode.

(b) U.S. Senator names, normalised edit distance:
- English phoneme 2-gram: **100** (every name wrong).
- Plus consonant-parity pruning of the table: 89.8.
- Phoneme 3-gram: 73.6.
- Word-based phoneme LM (76k CMUdict words): 57.2.
- Plus initialising consonant matches higher: 54.2.
- The parallel-trained system: 25.9.

(c) Failures: the authors call it "substantially more difficult" than letter ciphers because the table
is non-deterministic and fertility is uncertain. Both of those hold for us.

**Knight, Nair, Rathod & Yamada, COLING/ACL 2006** (https://aclanthology.org/P06-2065.pdf)

(a) and (b):
- Letter substitution: errors fell from 68 to 10 (2.4%), through a trigram LM, cubing P(c|p) at
  decode, and source smoothing. Cubing corrects a channel that is "too bunched up" by LM/cipher n-gram
  mismatch.
- Hindi code: a "fixed, uniform fertility model", with EM only on substitutions, cut edit distance
  from 127 to 93.
- Word mapping (7x7): a uniform initialisation "locks up after two iterations". A random
  initialisation got 4/7 correct, 5 restarts got 5/7, and 40 or more restarts got 7/7, selected by
  post-EM P(c), not by accuracy.
- Syllable model: the S/N/V model failed with "EM hijacking the extra symbols" until it was
  bootstrapped from the C/V model. The authors: "bootstrapping is good for dealing with too many
  parameters".

(c) Failures: with the weaker (bigram) source model, "EM initially locks onto the correct theory, but
task performance degrades", so stopping early was better. Likelihood rose while accuracy fell.

**Ravi & Knight, EMNLP 2008** (https://aclanthology.org/D08-1085.pdf)
- Exact ILP decipherment. A 3-gram LM works at length 64 or more.
- EM with a 2-gram: 10% error vs 0.5% for ILP at 414 letters; 85% vs 21% at 52 letters.
- The unicity distance U = H(K)/(A-B) is 173, 74 and 50 letters for 1/2/3-gram models. Empirical
  requirements are worse than these values.

**Nuhn, Schamper & Ney, ACL 2013** (https://aclanthology.org/P13-1154.pdf)
- Beam search over keys. A higher LM order needs a far larger beam.
  - Zodiac-408, 6-gram: 94.61% symbol error at beam 100k vs 1.96% at 10M.
  - Zodiac-408, 5-gram: 3.19% at 1M.
- Length-32 ciphers: the optimal 3-gram solution has 38.3% error vs 4.1% with a 6-gram.

**Nuhn & Ney, ACL 2013** (https://aclanthology.org/P13-1060.pdf)
- Unigram decipherment is a linear assignment problem.
- Bigram decipherment is a quadratic assignment problem, which is NP-hard.

**Nuhn, Schamper & Ney, EMNLP 2014** (https://aclanthology.org/D14-1184.pdf)
- Zodiac-408 is solved with an 8-gram at beam 26.
- Beale part 2 (762 tokens, 182 symbols, 4.18 observations per symbol) is solved at beam 10M
  (157/185 symbols correct).
- Beale parts 1 and 3 (1.74 and 2.34 observations per symbol) fail.

**Nuhn & Ney, ACL 2014, EM for probabilistic substitution** (https://aclanthology.org/P14-2123.pdf)

(a) Setup:
- An HMM with a zero-order lexicon, trained by exact, beam (B=100) and preselection EM.
- Pruning can make estimates "become zero ... cannot recover". So the E-step uses
  0.9*p + 0.1/|V_f|, i.e. 10% uniform.

(b) Numbers:
- Vocabulary 200, bigram: exact 97.19, beam 98.87, preselection 98.50. The pruned versions "find
  sparser distributions" and do better.
- Vocabulary 500: 92.1. Vocabulary 3661, 4-gram: 92.1.

(c) Failure: exact EM was intractable above vocabulary 200. Words were mapped to 200 or 500 classes
to make it tractable.

**Copiale cipher, Knight, Megyesi & Schaefer, BUCC 2011** (https://aclanthology.org/W11-1202.pdf)

(a) Setup:
- About 90 cipher symbols.
- Symbols were clustered by the cosine similarity of their preceding and following co-occurrence
  vectors.

(b) The cluster map was "of great help": once one symbol in a cluster was mapped, its cluster-mates
were mapped to the same letter, and "these leaps were frequently correct".

(c) Failures:
- Knight 2006's automatic homophonic attack, run on the Copiale cipher, gave nonsense in all 40-plus
  candidate languages, with only "a very slight numerical preference for German".
- The cause was a wrong cipher model: symbols with multi-letter values, and spaces encoded as letters.
- This is a warning about a likelihood margin under a mis-specified model.

### Neural decipherment

**Kambhatla, Mansouri Bigvand & Sarkar, EMNLP 2018** (https://aclanthology.org/D18-1102.pdf)
- A neural character LM scores beam hypotheses, with an optional frequency-matching heuristic (FMH).
- Zodiac-408: FMH helps slightly (symbol error 1.47 -> 1.22 at beam 1M).
- Beale part 2 (about 180 symbols): FMH **hurts**, 4.98 -> 5.50 at beam 1M and 41.67 -> 48.33 at
  beam 10k.

**Aldarrab & May, ACL 2021** (https://aclanthology.org/2021.acl-long.561.pdf)
- A seq2seq model trained on synthetic 1:1 ciphers.
- Frequency-rank encoding takes symbol error from 71.75% to 0.00% at length 256.
- It covers 1:1 ciphers only, not homophonic ones, so it does not apply to 500 -> 40.

### Deterministic annealing and EM behaviour

**Smith & Eisner, ACL 2004** (https://aclanthology.org/P04-1062.pdf)

(a) Setup: the full joint is raised to beta, starting at 1e-4 and multiplied by 1.2 per stage up to 1,
with convergence at each stage. That took 1200 E-steps vs 279 for EM.

(b) Numbers:
- POS tagging with a dictionary: DA 70.25 vs EM 66.63 on ambiguous tokens, with no likelihood gain.
  Across 10 splits, DA won only 3/10.
- Grammar induction (CCM): DA from a uniform init was **worse** than EM (47.09 vs 52.27 F).
- Skewed DA from a good initialiser gave 68.03 vs 67.10.

(c) The authors: DA "does not guarantee convergence to a global maximum, or even to a better local
maximum than EM", and fast schedules risk local maxima.

**Ueda & Nakano, NIPS 1994** (https://proceedings.neurips.cc/paper_files/paper/1994/file/92262bf907af914b95a0fc33c3f33bf6-Paper.pdf)
- DAEM: beta multiplied by 1.4 per stage, iterated to convergence at each stage.
- Shown only on a 1-D two-component GMM with 100 samples.
- The 1998 Neural Networks journal version is **UNVERIFIED** (paywalled).

**Itaya et al., Interspeech 2004** (https://www.isca-archive.org/interspeech_2004/itaya04_interspeech.pdf)
- DAEM with beta = sqrt(i/I); for HMMs, I=10 stages at 5 EM steps each.
- Advice: "should proceed as slow as possible, particularly at early stage".
- This is flat-start training **with transcripts**, i.e. supervised.
- Monophone gains: +24.8% relative speaker-dependent, only +3.9% speaker-independent.
- Speaker-independent triphones fell short of the baseline: "variations in speaker characteristics
  affect estimating the phoneme boundaries".

**Johnson, EMNLP 2007** (https://aclanthology.org/D07-1031.pdf)
- EM for unsupervised HMM POS tagging, with transitions learned (unlike ours).
- Likelihood is "still increasing after several hundred iterations". Accuracy moves by about 5% after
  several hundred iterations; it drops early and recovers after 100.
- Final 1-to-1 accuracy across 10 seeds spans 0.38-0.45.
- Annealing the parameters by 1/T, "with a variety of starting temperatures and annealing schedules",
  never produced significantly higher likelihood.
- Reducing the number of hidden states lets EM nearly match VB with sparse priors.

**Salakhutdinov, Roweis & Ghahramani, ICML 2003** (https://www.cs.toronto.edu/~rsalakhu/papers/emecg.pdf)
- EM is near-Newton when missing information is small, and "extremely slow" when posteriors are
  ambiguous.
- On ambiguous data, conjugate-gradient (ECG) beat EM: "substantially" for aliased HMMs and DNA
  HMMs, and "by at least a factor of two" for aggregate Markov models.

### Table vs neural emissions

**Berg-Kirkpatrick et al., NAACL 2010** (https://aclanthology.org/N10-1083.pdf)
- Multinomials are replaced by locally normalised log-linear feature models.
- Direct-gradient (LBFGS) training beat EM "in all cases where the algorithm was sufficiently fast
  to run". Examples:
  - POS many-to-1: 75.5 vs 68.1; the basic HMM gets 63.1.
  - Word segmentation F1: 88.0 vs 84.5.

**Tran et al., SPNLP 2016** (https://aclanthology.org/W16-5907.pdf)
- A neural HMM with the same information as a table HMM performs "almost identically": M-1 59.8 vs
  62.5.
- Gains came only from **added inputs**: character convolutions plus an LSTM context gave 79.1.
- Direct marginal-likelihood gradients beat EM, which was "slower to converge and perform slightly
  weaker".
- Initialisation mattered: V-measure 61.7 vs 71.7.

### Non-GAN unsupervised ASR

**Yeh et al., ICLR 2019** (https://arxiv.org/pdf/1812.09323)

(a) Setup:
- Segmental empirical output distribution matching (ODM) against a 5-gram phone LM, using the top
  10,000 5-grams.
- LM statistics are taken **at segment rate**, one sampled frame per segment. A frame-smoothness term
  is added.
- Boundaries are alternately refined by MAP. Model selection uses the held-out ODM loss, which
  "aligns well" with validation FER.

(b) TIMIT, non-matching LM:
- 49.1 PER after iteration 2.
- 41.6 after HMM self-training with SAT/fMLLR, which is worth about 3 points.
- Oracle boundaries give 40.1.

(c) Failures:
- Without the smoothness term (lambda = 0), performance "degrades significantly".
- The authors note that cluster-then-map approaches are capped by cluster purity: 41.0 FER* at 1000
  clusters.

**Baevski et al., wav2vec-U, NeurIPS 2021** (https://arxiv.org/pdf/2105.11084) (GAN; read for
preprocessing and selection only)
- Table 8 (LibriSpeech dev-other, 20 seeds):
  - Without the k-means repeat-merge segmentation, or without PCA, or without the second mean-pool:
    **0% of runs converge**.
  - 64, 128 and 256 clusters: 23.1, 21.4 and 22.3 PER.
- K-means boundaries have precision 0.935 and recall 0.379 (Table 1), i.e. about 2.5x
  over-segmentation.
- Unsupervised selection uses LM NLL of the Viterbi transcripts plus vocabulary usage. It lands within
  0.3-1.2 PER of labelled selection (TIMIT).

**Wang, Hasegawa-Johnson & Yoo, ESPUM** (https://arxiv.org/pdf/2310.02382; ICASSP 2024 per the
prior project report, venue not re-checked)
- K128 wav2vec2 layer-14 units with a CNN, trained by positional-unigram and skipgram L1 matching.
- TIMIT unmatched test: 45.1 PER, then 42.9 with HMM self-training.
- Table 3 (validation): bigrams only 71.6; uni+bi 39.2; uni+bi+tri 38.4; uni+5-gram **77.9**. Higher
  order alone fails under moment matching.
- The boundary teacher is Strgar & Harwath's readout (https://arxiv.org/pdf/2211.01461, read). Its
  checkpoints were selected on the **gold** validation R-value*. Whether ESPUM reused such a
  checkpoint is **UNVERIFIED**.

**Liu et al., Interspeech 2018** (https://arxiv.org/pdf/1804.00316) (GAN, oracle segments; read for
unit count only)
- Segment k-means with K from 50 to 1000.
- With unrelated text, "for 500 or more clusters (roughly 10 or more clusters for each phoneme ...)
  the mapping relationship became difficult to learn even though the cluster purity was higher". The
  best K was about 300.
- With matched text, accuracy rose with K up to 1000, tracking purity.

**Wang et al., ACL 2023, theory** (https://arxiv.org/pdf/2306.07926)
- Identifiability with a deterministic unit-to-phone map requires P_X to have full column rank.
- In synthetic experiments, the number of distinct LM-transition eigenvalues "required for perfect
  ASR-U" grows "as the number of speech units increases".
- These are GAN/MMD experiments on synthetic languages.

**Yang, Barkoczi, Schlueter & Ney, ICASSP 2026, theory** (https://arxiv.org/pdf/2603.02285)
- The paper motivates our exact objective: the marginal likelihood of a generative q(x|c) under a
  fixed LM.
- The bound needs (i) a per-token factorised emission and (ii) a full-rank positional-unigram LM
  matrix. Their LibriSpeech sigma_min is about 3e-4.
- There are no speech experiments, only a simulation with |X|=4 and |C|=3.
- Note (mine): 50 Hz frames within one phone are not conditionally independent, so assumption (i)
  holds at segment level, not at frame level.

**Ni et al., JSTTI 2024/25** (https://arxiv.org/pdf/2406.08380) and **Wang et al., SylCipher 2026**
(https://arxiv.org/pdf/2608.22907)
- Both are non-GAN, masked-LM, word- or syllable-level systems.
- Both select or stop on **paired validation WER/SER**, so neither is label-free.
- SylCipher sets the speech codebook size equal to the number of text tokens.
- SylCipher seeds give 33.0-34.1 CER, while a syllable-level wav2vec-U GAN gives 115.7-126.5 SER.

**Yang & Tang, ASRU 2023** (https://arxiv.org/pdf/2310.17558)
- K50 clusters, with the speaker direction estimated label-free from PCA of utterance means.
- Collapsing that direction changes type PER from 93.1 to 76.8 for APC, but only 73.0 to 72.6 for
  CPC.
- Frame PER is unchanged (APC 50.1 -> 50.1).

## Established vs single-paper

Established, meaning two or more independent groups agree:
1. **Restarts decide success, and are chosen by likelihood.** BK&K, Knight 2006, Klejch, and Johnson
   2007 (variance across seeds).
2. **Smoothing toward uniform matters.** BK&K, Knight 2006, Nuhn & Ney 2014, Klejch.
3. **A stronger LM (higher order, or word-level) helps when search is adequate.** Ravi & Knight
   2008/2009/2011, Knight 2006, Nuhn 2013, Klejch.
4. **Frame-rate input must be segmented or merged before distribution matching.** wav2vec-U, Yeh,
   ESPUM, SylCipher. These are all matching methods; no EM paper tests this.
5. **DA is not reliably helpful.** Smith & Eisner and Johnson 2007. The positive results are on a toy
   GMM (Ueda) or supervised flat start (Itaya).

Single paper only:
- A sparse channel prior (Ravi & Knight 2011).
- Fixed fertility or duration (Knight 2006).
- Pruned EM beating exact EM (Nuhn & Ney 2014).
- Frequency heuristics hurting under heavy homophony (Kambhatla 2018).
- Many units hurting the mapping (Liu 2018; theory in Wang 2023).
- Speaker-direction removal (Yang & Tang, encoder-dependent).
- Direct gradient at least as good as EM on POS: BK 2010 and Tran 2016 agree, but both are POS
  tagging.

Disagreements:
1. **Zodiac-408 EM at 100 restarts.** Ravi & Knight 2011 get 28.8%; BK&K get 90%.
   - BK&K weight an author-matched 2K-word LM at 0.9 and smooth by 0.1.
   - Ravi & Knight decode Viterbi with a cubed channel.
   - Which factor explains the gap is untested.
2. **Higher LM order.** It helps likelihood and beam decipherment (Nuhn, Klejch, Ravi & Knight). It
   fails alone under moment matching (ESPUM 77.9), and it fails with too small a beam (Nuhn 2013:
   94.61% at 100k).
3. **DA.** It helps POS on one split (Smith & Eisner) but wins only 3/10 splits and hurts CCM from a
   uniform init. Johnson found no schedule that helped.

## Ranked implications for A11 and for a failed first round

Question 1, iterations and restarts (strongest evidence).
- Published EM restarts run 200 iterations (BK&K) or 20 per stage (Klejch) before selection.
- Johnson 2007 shows accuracy moving after hundreds of iterations.
- Salakhutdinov 2003 shows EM is slowest exactly when posteriors are ambiguous, as at a Dirichlet
  initialisation.
- Consequence: ranking 32 restarts at iteration 10 may rank early speed, not the basin.
- Restart counts that saturate on published ciphers: about 100 for a 54x26 key (BK&K), 50 in Klejch,
  40 or more for a 7x7 key (Knight 2006). Our 500x240 table is far larger.
- Cheap changes:
  - more iterations per restart on fewer utterances. Klejch used 20 minutes of the shortest
    utterances; 200-300 utterances is about 0.7-1 h (mine);
  - log S at iterations 10, 20 and 40 for all stage-B runs, to check whether the stage-A rank
    predicts the final one.
- A failure at 32x10 does not show that the objective fails.

Question 2, smoothing.
- Working recipes smooth by 10-26% of row mass: Nuhn and Klejch 10% uniform interpolation; BK&K about
  26% (mine).
- A11's 1e-3 pseudo-count is about 0.02% of stage-A row mass: roughly 2.6k frames per row (mine).
- A11 also prunes (max_active 1000), which is exactly the condition under which Nuhn & Ney 2014 needed
  E-step interpolation.
- Proposal: add 0.9*p + 0.1/500 in the E-step, at least through the annealing iterations.

Question 3, deterministic annealing. A11's tau 4 -> 1 over 4 single iterations is far faster than any
published DA schedule, and the evidence for DA is weak or negative. Either:
- keep it as is, but add one tau=1 (no annealing) arm so that DA is not a silent confound; or
- give it several iterations per temperature.
I found no published test of annealing only phi while keeping the LM at full weight. That question is
open.

Question 4, 500 units vs 50-100.
- The direct evidence favours fewer units: Liu 2018 (best K about 300, with 500 or more becoming hard)
  and the Wang 2023 theory.
- That evidence comes from GAN or oracle-segment setups. Purity rises with K, so fewer units lowers
  the ceiling.
- Coarse-to-fine bootstrapping helps (Knight 2006; Klejch pruning). Copiale-style context clustering
  of cipher symbols is label-free.
- Not settled. It would take an A11 run at about K100 from the same encoder, with its own null.

Question 5, data and cipher length.
- Not the bottleneck (mine). A deterministic 500->40 key has H(K) about 500*log2(40), roughly
  2.7 kbit. At about 2 bits of redundancy per phone, that needs about 1.3k phones, around 100 s of
  speech.
- Nuhn 2014's failures sit at 1.7-2.3 observations per symbol; A11 stage A has about 1.3k frames per
  unit.
- The real limits are channel noise and frame correlation.

Question 6, LM order and a beta>1 LM weight.
- A stronger LM is the best-established lever: Ravi & Knight 2009 phonetic decipherment went 2-gram
  100 -> 3-gram 73.6 -> word LM 57.2.
- Klejch's bigram stage was a compute device, so the trigram-only rule costs nothing demonstrated.
- A word-level or 4-5-gram LM needs a wider beam than max_active 1000 (Nuhn 2013).
- No paper tests beta>1 during EM.
  - Published decipherment cubes the channel at decode, at symbol rate.
  - At 50 Hz each phone contributes several frame emissions, so the channel is already over-weighted
    relative to the LM (mine).
  - Open question; it would take a beta sweep with its null.

Question 7, initialisation.
- Keep Dirichlet(1). BK&K saw little effect from flat initialisations and harm from corner-heavy ones.
- Do not use frequency-rank initialisation under 12.5 homophones per phone. FMH hurt on the
  180-symbol Beale cipher (Kambhatla), and frequency-rank encoding is validated only for 1:1 ciphers
  (Aldarrab).
- Context co-occurrence clustering (Copiale) is untested as an automatic initialiser.
- (Project record) 1g.12: an ESPUM-seeded EM reached PER 0.83.

Question 8, speaker normalisation. The evidence is weak.
- Label-free speaker-direction collapse helped one encoder (APC 93.1 -> 76.8) but not CPC, and left
  frame PER unchanged.
- SAT helped only in supervised-style HMM self-training (Yeh), and Itaya's speaker results are
  supervised.
- A11's dropping of eta has no literature against it.

Question 9, table vs neural emissions.
- Neural emissions equal tables when given the same inputs (Tran 2016). Gains need added information.
- BK 2010 and Tran 2016 find direct gradient training at least as good as EM. So Attempt 1's slow
  progress means too few steps, not a wrong family.
- Keep the table for A11.

Question 10, label-free success detection.
- Likelihood selection is reliable only across large gaps (BK&K: -1466.5 at 89% beat gold's -1467.4
  at 92%).
- Likelihood can keep rising while accuracy falls (Knight 2006; Johnson 2007).
- A likelihood margin under a wrong model can be real but tiny (Copiale).
- Recent non-GAN systems (JSTTI, SylCipher, and Strgar's teacher) quietly select on paired
  validation.
- Frame shuffling within an utterance destroys unit runs as well as phonotactics. A content-free
  duration or persistence model can therefore beat that null (mine).
- Add, as report-only diagnostics:
  - a run-preserving null that shuffles whole runs of identical units;
  - wav2vec-U's label-free LM NLL of the Viterbi phones plus vocabulary usage.

## UNVERIFIED or not read

- Ueda & Nakano 1998 (journal; paywalled).
- Vobbilisetty et al. 2017 Cryptologia, on HMM restarts vs ciphertext length for homophonic ciphers
  (paywalled; the SJSU copy is behind Cloudflare).
- Dhavare, Low & Stamp 2013; Zhong 2016; Kopal 2019. Known only through one-line summaries in the
  Z340 paper.
- ESPUM's venue and its boundary-checkpoint provenance.
- Johnson & Willsky's HDP-HSMM.
- Lin et al. 2022 on UASR robustness.
- Knight & Yamada 1999.
- Snyder 2010 and Dou & Knight 2012 were downloaded but not used. Hauer 2014 was skimmed; it covers
  monoalphabetic ciphers only.

## What this changes for the decision

Three literature-driven changes to register in A11 before launch:
1. More EM iterations per stage-A restart on a smaller subset, plus a check of whether the rank at
   iteration 10 predicts the final rank.
2. E-step interpolation of 0.9 toward uniform, because the lattice is pruned.
3. Either a no-anneal arm or a slower anneal.

Add a run-preserving null as a report-only diagnostic. The gate stays as registered.

If A11 fails, the ordered next steps are:
1. More restarts and iterations.
2. A coarse-to-fine or fewer-unit variant.
3. A word-level or higher-order LM with a wider beam.
4. A sparse channel prior.
