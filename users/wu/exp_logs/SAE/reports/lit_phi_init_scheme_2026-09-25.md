# Literature: a new phi initialisation (A coarse-to-fine alphabet, B Ravi & Knight 2011 sampler, C LM-led flat-channel E-step)

Date: 2026-09-25. Role: literature. Every numbered claim below was read from the full text at the URL given,
unless it is marked "prior report" (verified from full text in the named earlier report, not re-read here) or
UNVERIFIED. Statements marked "(mine)" are my inferences, not published results.

## Verdict

- (A) has no published precedent in the form proposed: a named channel under a fixed class-level LM, mapping
  into noisy acoustic units, split class by class with children inheriting the parent's emission row. What exists
  is: (i) C/V -> S/N/V bootstrapping on letter text (Knight et al. 2006; qualitative only, needed fixed syllable
  counts and restarts); (ii) general consonant-class knowledge constraining a phoneme channel (Ravi & Knight 2009;
  NED 100.0 -> 89.8); (iii) hierarchical split-merge where the coarse categories are observed (Petrov 2006, prior
  report) or unnamed (Rose 1998 DA; Varadarajan 2008 SSS; Franti & Sieranoja 2019 Split). None measures naming
  accuracy of a coarse-to-fine scheme against flat EM. Text-only distributional methods recover English V/C
  reliably (up to 100% by type), but below V/C the distributional classes are phonotactic/positional, not manner
  classes (Hulden 2017: 25% of English splits fall along a single distinctive feature; Mayer 2020 qualitative).
- (B) Ravi & Knight 2011's gains rest on conditions we do not have: a cipher that is deterministic in the
  decipherment direction (their type-sampling justification), a word LM, and word spaces. With a letter 3-gram only,
  Bayesian type sampling reached 23.0% on Zodiac-408 (97.8% needs the word+3-gram LM). No noise study, no speech
  application found. The project's own E10 (wrong hard keys fit the trigram better than gold) predicts the R&K
  posterior would also prefer wrong keys (mine).
- (C) A flat channel under a fixed LM is not a neutral start: Knight 2006's 7x7 word mapping with uniform P(c|p)
  "locks up after two iterations, and every English word learns the same distribution"; the uniform-channel letter
  cipher worked only with P(space|SPACE)=1 fixed. Wang et al. 2023's identifiability condition makes the same point
  formally: identification comes from non-stationary positional marginals (Assumption 2). No speech result exists
  for an LM-led flat start.
- The literature does not settle whether (A) beats flat EM on naming for our problem. It would take the in-project
  measurement listed at the end.

## What was read (full text)

Knight, Nair, Rathod & Yamada 2006; Ravi & Knight 2009, 2011; Kim & Snyder 2013; Hulden 2017; Mayer 2020;
Goldsmith & Xanthos 2008 tech report (not the 2009 Language article); Spitkovsky, Alshawi & Jurafsky 2010;
Smith & Eisner 2006; Rose 1998; Varadarajan, Khudanpur & Dupoux 2008; Wang, Hasegawa-Johnson & Yoo 2023 (ESPUM);
Wang, Hasegawa-Johnson & Yoo 2023 (theory, ACL); Klejch, Wallington & Bell 2022; Aldarrab & May 2021;
Franti & Sieranoja 2019; Wang et al. 2026 (SylCipher). Reused from prior reports without re-reading:
lit_phi_name_signal_2026-09-24, lit_phi_rename_moves_2026-09-24, lit_partition_levers_2026-09-25,
lit_em_decipherment_2026-09-23, lit_decipherment_relabel_2026-09-24, lit_method_v2_2026-09-15.

## Q1. Is coarse-to-fine established, and does it beat flat EM?

### 1a. Precedents, with what is split and whether it carries names

| Work | What is coarse-to-fine | Named? | Numbers (conditions) |
|---|---|---|---|
| Knight et al. 2006, COLING/ACL | Source alphabet: C,V,SPACE model run first, then fed into S,N,V,SPACE | Yes, but labels by a fixed "syllable theory" | Qualitative only (figures, word lists). Without bootstrapping, "EM hijacking the extra symbols"; "bootstrapping is good for dealing with too many parameters". |
| Ravi & Knight 2009, NAACL | General consonant-parity knowledge prunes a phoneme->phoneme channel (English->Japanese katakana) | Yes | Whole-name error / NED: 2-gram 100/100.0; + parity 98/89.8; 3-gram+parity 94/73.6; word LM+parity 77/57.2; + consonant-match init at 100x weight 73/54.2; + more English 66/49.3; parallel-trained 40/25.9. EM up to 200 iterations, 15,320 uniform-weight channel arcs, channel cubed at decode. |
| Petrov et al. 2006 (prior report) | Split-merge of observed treebank categories | Coarse categories observed; subcategories unnamed | Hierarchical 88.4 vs direct 87.3 F1 at 16 subsymbols. |
| Rose 1998, Proc. IEEE | Deterministic annealing: clusters split at phase transitions as temperature falls | No (squared-error clusters) | Theorem 1: a cluster splits at T_c = 2 x lambda_max (variance on its principal axis), along that axis. 6-Gaussian example: DA MSE 5.7 vs GLA best-of-25 random inits 6.4 (reached once; 80% of runs at 12.5). 9-Gaussian: DA 32.854 vs GLA 33.5-40.3. Synthetic 2-D data. |
| Varadarajan, Khudanpur & Dupoux 2008, ACL short | Successive state splitting of an acoustic HMM from 1 state | No; names assigned by a transducer fitted on 5-24 min of transcribed speech | 87.2% phone accuracy at 376 states (Japanese, 1 speaker, 24 min). No comparison against flat or random init. |
| Franti & Sieranoja 2019, Pattern Recognition | Split initialisation of k-means | No | Split CI 1.2, success rate 40% vs random centroids CI 4.5, 5%; fails on Birch1 and Unbalance. "In case of high overlap, k-means reaches almost the same result (about 88% success rate) regardless of how it was initialized." |
| Goldsmith & Xanthos 2008 (UChicago TR-2008-08) | 2-state then 3-state HMM on phone text | By inspection | 2-state EM: C/V "100 percent successful" on English (58,156 CMU-derived types). 3-state on French: third state became "onset cluster last element", not the expected onset/coda split. |

(A) is not in this table: nobody splits a named phone alphabet under a fixed class-level LM into noisy units.

### 1b. How well do text-only methods find V/C and manner classes in English phone text?

- V/C is established across groups and methods:
  - Hulden 2017, English phonemic child-directed text (37 phonemes): OCP alternation 100.0% by type; Moler &
    Morrison SVD 94.59%; Sukhotin 21.62% (13/37 labelled syllabic).
  - Goldsmith & Xanthos 2008, CMU-derived English: 2-state HMM EM 100%; spectral method misclassifies 4 consonants
    (Y, W, R, Z), fixed by two-sided context; Sukhotin fails on English (they attribute this to the stress encoding).
  - Thaine & Penn 2017 (prior report): second singular vector 97.45% type / 99.39% token.
  - Kim & Snyder 2013, 503 languages (Bible, letters, token accuracy, gold 1-1 label mapping): EM 93.37, SYMM 95.99.
- Sukhotin disagrees across papers on English phonemes: 21.62% (Hulden), poor (Goldsmith & Xanthos), 94.34% by
  type (Thaine & Penn, prior report). The difference is the transcription and inventory. Do not use Sukhotin.
- Below V/C there is no established result, and the published ones disagree with manner classes:
  - Kim & Snyder: C/V/Nasal token accuracy EM 74.59, SYMM 80.72 (cross-lingual variants 86.13 / 89.37).
  - Hulden 2017: only 3 of 12 English splits (25%) align with a single distinctive feature (non-tier), 15/25 (60%)
    with the tier variant, which tends toward coronal/non-coronal and front/back. Quote: "the only really robust
    pattern reliably discovered ... is the distinction between consonants and vowels".
  - Mayer 2020 (CMU dict, 26,552 word types): V and C retrieved; the top consonant split is /w r/ vs /p b f h j l/ vs
    the rest, interpreted as onset-position preference; /k g/ and /p b f/ recovered lower down; vowels "more
    difficult to interpret". No accuracy numbers.
- So a distributional hierarchy gives V/C at the top. Below that it gives phonotactic/positional classes, which
  group phones that sound different (/p b f h j l/). (mine) A class-level channel must then map acoustically
  mixed unit sets into one class. That is allowed, but it means the class emission row a child inherits is a
  mixture, not a coarse version of the child.

### 1c. Does splitting beat flat EM on permutation or local optima?

- For unnamed clusters, yes, on synthetic data: Rose 1998 (DA against GLA, numbers above); Franti & Sieranoja 2019
  (Split success 40% vs 5%, but equal to random init under high overlap).
- For observed coarse categories, a little: Petrov 2006 (+1.1 F1, prior report).
- For names: the only evidence is Knight 2006, qualitative. No paper reports naming accuracy (a permutation or key
  error) for coarse-to-fine against flat EM. The literature is silent on the quantity (A) is meant to improve.

### Q1 implications

- (A) No precedent in the proposed form. V/C is the one level where a text-only class is reliable. A general
  phonetic-class allocation (manner) would need the user's ruling, and the distributional alternative does not give
  manner classes below V/C. (mine) At class level the name search is small (2-6 classes give at most 720
  permutations), so exhaustive relabel+refit under S is feasible. At each split, though, the children start
  symmetric, with identical inherited rows. The within-class naming problem then reappears, only with smaller
  permutation groups. Top-level errors propagate, and no paper quantifies how far.
- (B) Nothing in Q1 bears on B.
- (C) Rose's DA gives coarse-to-fine without a hand-made hierarchy. As temperature falls, splits appear by
  themselves. But the splits are unnamed, and the DA-for-EM evidence on naming is negative or neutral (Johnson 2007:
  annealing never raised likelihood; prior report).

## Q2. Ravi & Knight 2011 in detail, follow-ups, noise

- Model (ACL 2011, full text):
  - A CRP cache model for both source and channel. Source alpha = 10^4, which effectively fixes the LM. Channel
    beta = 0.01 (sparse), with a uniform base distribution P0.
  - Point-wise and type-level Gibbs sampling. Type sampling draws one plaintext letter for a cipher TYPE and updates
    all its positions, "similar to Liang et al. (2010)". Its justification is determinism in the decipherment
    direction; a footnote says this "does not strictly apply to the Zodiac-408 cipher where a few cipher symbols
    exhibit non-determinism".
  - Rescoring is incremental, through exchangeability within a context window. Ciphers without spaces get a second
    operator that samples word boundaries.
  - The initial key maps frequent letters to frequent symbols. 5000 iterations; temperature annealed linearly from
    10 to 1.
  - LM: 0.9 word / 0.1 letter n-gram interpolation; 9,881-word dictionary; letter n-gram trained on 50M words.
- Ciphers: a 414-letter simple cipher; a 414-letter homophonic cipher with spaces (Simon Singh key); Zodiac-408
  (408 tokens, 54 types, last 18 characters nonsense, some misspellings).
- Accuracy (simple / homophonic with spaces / Zodiac):

  | Method | Simple | Homophonic with spaces | Zodiac-408 |
  |---|---|---|---|
  | EM 2-gram | 83.6 | 30.9 | - |
  | EM 3-gram | 99.3 | 32.6 | 0.3 (28.8 with 100 restarts) |
  | Bayesian 3-gram | 100 | 95.2 | 23.0 |
  | Bayesian word+2-gram | 100 | 100 | - |
  | Bayesian word+3-gram | 100 | 100 | 97.8 |

- Zodiac ablations: beta 0.01 vs 1.0 gives 97.8 vs 24.0. Type vs point-wise sampling gives 97.8 vs 14.5.
- 98-letter simple cipher: EM 3-gram 41%; Ravi & Knight 2009 method 84%; Bayesian word+3-gram 100%.
- Not in the paper: a noise study (beyond Zodiac's own errors), a speech application, and any homophony level
  beyond these ciphers. I found no speech application of this sampler.
- Beam search under noise: Hauer et al. 2014 (prior report) is the only published noise test for a search decipherer
  (noise of about log2 n corrupted letters: 16.66 / 5.20 / 2.73% at lengths 64 / 128 / 256, simple substitution).
- Aldarrab & May 2021 test noise only on their own seq2seq decipherer, which is trained on synthetic keyed ciphers
  (1:1 substitution, length 256): TER 1.10-17.63% for 5-25% substitution noise, 2.87-27.43% for mixed
  insertion/deletion/substitution noise. They report the Kambhatla 2018 beam search only without noise (SER 26.80 at
  length 16, 0.00 at 256).
- How Nuhn-style beam search on homophonic ciphers degrades with noise is not published. The literature is silent.
- Nuhn 2014 Beale (prior report): solved at 4.18 observations per cipher symbol, failed at 1.74 and 2.34.

### Q2 implications

- (B) The conditions that made R&K work are all absent: determinism in the decipherment direction (our gold key has
  frame purity 0.644, SAE_2S.md line 147), a word LM, and word spaces. The closest matching row, letter 3-gram only
  without word structure on Zodiac, is 23.0%.
- (B) The two ablations that carry R&K are the sparse channel prior and type moves. The project already ran annealed
  type-level ICM/Gibbs under J, and J ranks gold 136th of 188. Liang, Jordan & Klein 2010 report type-greedy moves
  "disastrous for the HMM" (prior report).
- (B) (mine) E10 in SAE_4A_rename.md finds that wrong hard keys fit the trigram better than gold per frame and per
  phone. The R&K posterior is that same LM term plus a sparsity term, so it would share the defect. What B would add
  is the sparsity term alone; whether that flips the ranking is testable by scoring gold vs found keys under the
  beta = 0.01 CRP channel.
- (A) and (C): R&K's frequency-matched initial key is a published instance of "LM-led" initialisation, but it was
  used together with 5000 annealed sampling iterations, not tested alone.

## Q3. When does flat-channel LM-led EM find the key?

- It succeeds when an anchor breaks the symmetry. Knight et al. 2006, 417-letter cipher, uniform P(c|p) with
  P(space|SPACE)=1 fixed:
  - EM gives 68 errors; bigger LM 64; smoothing 62; cubed channel 42.
  - Trigram gives 57 (small source data) or 32 (large); trigram plus cube 15; plus source smoothing 10 (2.4%).
- It fails when there is no anchor. Knight 2006, 7x7 word mapping, uniform P(c|p): "EM locks up after two
  iterations, and every English word learns the same distribution". Random init gives 4/7; 5 restarts 5/7; 40 or more
  restarts always 7/7, selected by post-EM P(c).
- It degrades with a weak LM. Knight 2006: "EM initially locks onto the correct theory, but task performance degrades
  as it tries to make the ciphertext decoding fit the expected bigram frequencies"; early stopping helps.
- Init shape matters (BK&K 2013, prior report): U[0,1] init is fine, a corner-favouring Dirichlet init "deteriorated
  sharply", and 1 restart gives 18% against 90% with 100.
- Homophony and length, for EM 3-gram (R&K 2011): 99.3 on a simple cipher, 32.6 on a homophonic one of the same
  length.
- Speech (Klejch, Wallington & Bell 2022, Interspeech):
  - Setup: the input is the output of a universal phone recogniser trained with supervision on other languages. The
    channel is a flower transducer that allows all substitutions. Silence is always mapped to silence or a word
    boundary (an anchor). The LM grows from bigram to 5-gram graphemes (20 full-batch iterations per order), then a
    word trigram. 50 random restarts with the bigram, selected by training likelihood. Smoothing
    0.9 p + 0.1 uniform.
  - Decipherment WER before semi-supervised training is 31.0-105.3 across 7 GlobalPhone languages.
  - It is worse than the knowledge-based phone-mapping baseline on 5 of 7 languages: CES 49.0 vs 44.3, HAU 70.8 vs
    43.4, POR 53.8 vs 52.8, SWA 105.3 vs 82.5, SWE 93.5 vs 66.8. It is better on BUL (31.0 vs 35.0) and UKR (34.5 vs
    35.7).
  - There is no ablation of the order curriculum or the restarts.

### Q3 implications

- (C) (mine, grounded in Knight 2006 and Wang 2023 Assumption 2) With a uniform channel, the phone posterior at each
  frame equals the LM's positional marginal. Where that marginal is stationary, every phone's emission row becomes
  the unit frequency, and EM sits at a fixed point. Only SIL, utterance edges and the duration topology break this.
  So C is LM-led only through those anchors, and a tempered channel lengthens the time spent near the fixed point.
- (C) No published speech result tests a flat start without a supervised front end. Klejch's front end names phones
  already, so its success does not transfer.
- (A) Knight's 7x7 result, where restarts selected by P(c) always win at 40 or more, is the regime where exhaustive
  class-level search is known to work: a few symbols and a likelihood that ranks the right key first. That last
  condition is what the project has not shown for S at phone level.

## Q4. Non-GAN recognisers that anchor names with classes, landmarks or a hierarchy; ESPUM; Wang 2023

- No non-GAN unsupervised phone recogniser in the literature anchors names with phonetic classes, landmarks or a
  class hierarchy. Searches found none. The GAN papers are background only and are not listed. Ravi & Knight 2009
  (text-to-text phoneme channel, class knowledge) is the nearest instance.
- ESPUM (Wang, Hasegawa-Johnson & Yoo 2023, arXiv 2310.02382):
  - What it learns: a 1-layer CNN generator (kernel 4) from K-means units (K=128, wav2vec 2.0 layer 14,
    LibriLight-pretrained) to phone posteriors.
  - Objective: positional-unigram L1 matching plus N-skipgram L1 matching (bi-skipgrams with skip up to 6,
    tri-skipgrams with skip up to 2), a soft monotonic aligner from a CNN segmenter, and a smoothness loss
    (lambda 16). The segmenter's noisy boundary labels come from the Strgar & Harwath wav2vec 2.0 readout model.
  - Initialisation of the generator is not stated.
  - Results are on TIMIT only; no LibriSpeech result. Matched / unmatched test PER: uni+bi+tri 43.3 / 47.3;
    iter 1 39.1 / 45.1; + HMM self-training 33.7 / 42.9.
  - Table 3 (val PER): bigrams only 71.6 against uni+bi-grams 39.2, and uni+5-grams 77.9. Adding the positional
    unigram is the largest single effect in the table.
- Wang, Hasegawa-Johnson & Yoo 2023 theory (ACL):
  - Theorem 1: the binary mapping O is recoverable iff the stacked positional marginals P_X have full column rank.
    Under the assumptions, it is O = P_X^+ P_Y.
  - Assumption 1: the N-gram transition matrix needs at least |X| distinct nonzero eigenvalues.
  - Assumption 2: the initial distribution pi must excite at least |X| eigenspaces. (mine) A stationary pi violates
    this.
  - Lemma 1: sigma_min(P_X) grows with the minimum eigen-gap and with length L.
  - Experiments are synthetic, with GAN or MMD generators: PER shows a phase transition once the number of distinct
    eigenvalues exceeds the number of units.
  - The theory says nothing about alphabet curricula or class structure. The matrix is |X| x |Y| and binary, so it
    assumes a deterministic unit-to-phone map, which our 0.644-purity units are not.
- SylCipher (Wang et al. 2026, arXiv 2608.22907) is syllable level and uses masked-LM matching. Its seed spread is
  small (CER 33.0-34.1 over 4 seeds, LibriSpeech). It stops training on paired validation SER, which our rules forbid.
  It does not bear on A/B/C.

### Q4 implications

- (A) (mine) Wang's identifiability rests on positional (non-stationary) statistics, and ESPUM's largest gain comes
  from the positional unigram. In a class-level stage, names would therefore come from how classes are distributed
  near SIL and utterance edges, plus the class trigram. With fewer symbols the rank condition is easier to meet
  (fewer distinct eigenvalues needed), but this is not shown in the paper.
- (B) Nothing here.
- (C) Same anchor argument as in Q3.

## Q5. Name lock-in and curricula

- Lock-in and drift are established across groups:
  - Knight 2006: early lock-in to the correct theory, then drift under a weak LM; the 7x7 lock-up.
  - Spitkovsky et al. 2010 (DMV, WSJ45): EM from the oracle initialisation drifts from 69.8% to 50.6%.
  - Liang & Klein 2008: prior report.
  - The project: names lock in within about 4 sub-epochs.
- Curriculum over the output alphabet: only Knight 2006's C/V -> S/N/V bootstrapping (qualitative) and Petrov's
  split-merge (observed categories). No numbers on naming.
- Curriculum over LM order:
  - Knight 2006: moving from bigram to trigram helps, 68 -> 57/32 errors, but these are separate runs, not a
    curriculum.
  - Klejch 2022 uses a bigram -> 5-gram -> word curriculum without ablating it.
  - Our rules forbid bigram terms anyway.
- Curriculum over data (Spitkovsky, Alshawi & Jurafsky 2010): Baby Steps 39.7% directed accuracy on WSJ45 vs
  uninformed EM 19.1%; Leapfrog 45.0% on Section 23.
- Structural annealing (Smith & Eisner 2006, directed accuracy):

  | Language | EM | Fixed delta | Annealed delta |
  |---|---|---|---|
  | English | 41.6 | 61.8 | 53.8 |
  | German | 54.4 | 61.3 | 70.0 |
  | Bulgarian | 45.6 | 49.2 | 58.3 |

  - Annealing does not always win: English is best with a fixed delta.
  - All conditions, including EM, used supervised model selection, and the stopping delta "matters tremendously". The
    gains are therefore not attainable under our no-label selection rule.

### Q5 implications

- (A) A class-first stage is a curriculum over the output alphabet, and it has no quantitative precedent. The
  lock-in evidence cuts both ways (mine): a correct class-level lock-in would carry over, but so would a wrong one.
- (C) Tempering the channel is the DA/annealing family. The evidence is mixed (Rose positive, unnamed; Johnson 2007
  neutral; Smith & Eisner 2006 positive only with supervised selection).

## Established vs single-paper

- Established across groups:
  - Text-only V/C discovery in English phone text is near-perfect with the right method (Hulden; Goldsmith & Xanthos;
    Thaine & Penn).
  - EM decipherment needs restarts or anchors: Knight 2006; BK&K 2013; Klejch 2022; R&K 2011 (Zodiac 0.3 vs 28.8).
  - EM drifts from good initialisations: Knight 2006; Spitkovsky 2010.
  - DA and splitting beat random init on unnamed clustering: Rose 1998; Franti & Sieranoja 2019.
- Single-paper:
  - The type-sampling and sparse-prior gains (R&K 2011, one cipher per condition).
  - Bootstrapping small to large alphabets (Knight 2006, qualitative).
  - The consonant-parity constraint (R&K 2009).
  - Positional unigram as the main naming signal in a speech system (ESPUM Table 3, TIMIT).
- Disagreements:
  - Sukhotin on English phonemes (21.62% vs 94.34%).
  - Whether annealing helps: Smith & Eisner 2006 yes, with supervised selection; Johnson 2007 no.

## UNVERIFIED

- Goldsmith & Xanthos 2009, Language 85(1):4-38 is paywalled. I read the 2008 technical report that precedes it.
- Ueda & Nakano 1998 (DAEM, Neural Networks) is paywalled and was not read. The prior report cites the 1994 version.
- Whether ESPUM's boundary teacher checkpoints were selected on gold (carried from the prior report).
- Luo, Hartmann & Barzilay 2021 (phonetic prior for undersegmented scripts) was not read.

## What would settle it (in-project, no training arm needed for the first two)

1. Under S, with gold units and the current trigram, does the gold CLASS-level key (V/C, then a 4-6 class cut) rank
   first among all class permutations and among nearby partitions? If it does not, (A) fails at its first step, in
   the same way J fails at phone level.
2. At each binary split with gold parents, does S rank the gold child assignment first? This tests whether the
   within-class naming problem is easier than the flat one.
3. For (B): score gold against the found keys under the CRP channel term with beta = 0.01 plus the trigram (E10's
   quantities plus one sparsity term).

## Sources

- Knight, Nair, Rathod & Yamada 2006, COLING/ACL: https://aclanthology.org/P06-2065.pdf
- Ravi & Knight 2009, NAACL: https://aclanthology.org/N09-1005.pdf
- Ravi & Knight 2011, ACL: https://aclanthology.org/P11-1025.pdf
- Kim & Snyder 2013, ACL: https://aclanthology.org/P13-1150.pdf
- Hulden 2017, CoNLL: https://aclanthology.org/K17-1030.pdf
- Mayer 2020, Phonology: https://linguistics.ucla.edu/wp-content/uploads/2020/01/an-algorithm-for-learning-phonological-classes-from-distributional-similarity.pdf
- Goldsmith & Xanthos 2008, UChicago TR-2008-08: https://newtraell.cs.uchicago.edu/files/tr_authentic/TR-2008-08.pdf
- Spitkovsky, Alshawi & Jurafsky 2010, NAACL: https://aclanthology.org/N10-1116.pdf
- Smith & Eisner 2006, ACL: https://aclanthology.org/P06-1072.pdf
- Rose 1998, Proc. IEEE 86(11): https://web.ece.ucsb.edu/publications/rose/pubs/pub42-Proc11-98.pdf
- Varadarajan, Khudanpur & Dupoux 2008, ACL: https://aclanthology.org/P08-2042.pdf
- Wang, Hasegawa-Johnson & Yoo 2023, ESPUM: https://arxiv.org/abs/2310.02382
- Wang, Hasegawa-Johnson & Yoo 2023, theory, ACL: https://aclanthology.org/2023.acl-long.67.pdf
- Klejch, Wallington & Bell 2022, Interspeech: https://arxiv.org/pdf/2111.06799
- Aldarrab & May 2021, ACL: https://aclanthology.org/2021.acl-long.561.pdf
- Franti & Sieranoja 2019, Pattern Recognition 93: https://cs.uef.fi/sipu/pub/KM-Init-PR-2019.pdf
- Wang et al. 2026, SylCipher: https://arxiv.org/abs/2608.22907
