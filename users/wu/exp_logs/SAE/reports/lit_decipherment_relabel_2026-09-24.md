# Literature read: label-free recovery of the symbol-to-phone mapping (decipherment, relabelling, split/merge)

Literature agent, 2026-09-24. Question: our reverse explicit-duration HSMM EM (500 units, 40 phone
states, fixed phone trigram) yields a phonetic partition whose states carry the wrong phones: a
permutation plus merges, with 11-19 phones unused. The optimally relabelled frame accuracy is about
half the oracle. Continuous EM does not repair the labels: they drift while log S rises, and a
permuted but otherwise perfect phi does not bootstrap under the trigram (project record). What
discrete, label-free step can recover the mapping from the phone LM and the unit stream alone?

Every numbered claim below was checked against the full text at the URL given. **UNVERIFIED** marks
anything not read. **(mine)** marks my own arithmetic and inference. **(project record)** marks facts
from the dispatch brief, which are not literature. This report complements
`reports/lit_em_decipherment_2026-09-23.md`, which covers the EM recipe (restarts, smoothing,
annealing, Klejch, BK&K 2013, ESPUM, wav2vec-U), and does not repeat it.

## Verdict

The literature supports a separate discrete key search followed by joint re-estimation. It does not
show that such a step works at our noise level.

1. **Stage 1: a discrete key search, then S-EM from the result.**
   - Search the key over whole symbol types, scored from n-gram count tables, with many restarts.
     Candidate methods are Nuhn beam search, Ravi & Knight type sampling, or swap local search.
   - Then initialise the full-marginal S-EM from the relabelled phi.
   - The closest published analogue is Yin et al. 2019: deciphering unsupervised clusters of
     manuscript glyphs under a character LM. There, joint LM-GMM EM could not reach the gold model's
     score even after 5,000 random restarts, and was fixed by initialising it from the pipeline
     decipherment.
2. **Objective.**
   - For a pure 40-state permutation, the choice hardly matters. The channel term is invariant under
     relabelling, so S and the LM score of the relabelled stream rank permutations nearly identically
     (mine).
   - For any many-to-one or unit-level key, the objective must keep the channel term, as in the full
     marginal S or the homophonic-cipher likelihood.
   - The LM score of decoded outputs alone is mode-seeking and collapses to a trivial solution. On
     OCR, Liu et al. 2017 got 83.37% error for that objective, equal to always guessing the majority
     class, against 9.59% for the coverage-seeking objective.
3. **Merges cannot be repaired by relabelling states.** A state relabel is capped at the
   optimal-relabelling accuracy, which is about half the oracle (project record). Going beyond it
   needs unit-level moves: reassign a whole unit type to another phone, as in type sampling, or split
   and merge candidates, each accepted only if S improves (SMEM; Hughes et al. 2015).
4. **Not settled.** No paper works at our noise level, so the literature cannot say what accuracy a
   trigram key search reaches when the relabelled symbol stream is only about 50% as accurate as the
   oracle. The nearest points are Yin's cluster streams with an optimal-mapping error of 0.22-0.44.

## Per-source findings

### Classical substitution decipherment under n-gram LMs

**Knight & Yamada, WVLC 1999** (https://aclanthology.org/W99-0906.pdf)
- Method: EM on a sound-to-character channel under a fixed phoneme n-gram model, with a
  one-for-one channel for Spanish.
- LM order:
  - Spanish phoneme trigram: 96% of sounds correct.
  - Phoneme pairs (bigram): 92%.
- Decoding: maximising P(s)·P(c|s)^3 beats P(s)·P(c|s), because "the phonetic model [may] overrule
  them incorrectly".
- Japanese kana, by amount of text:

  | Sentences | Phoneme accuracy |
  |---|---|
  | 200 | 99% |
  | 100 | 97.5% |
  | 50 | 96.2% |
  | 20 | 82.2% |
  | 5 | 48.5% |

- Chinese (2,113 distinct characters): 22% syllable accuracy.
- Compute: runtime is "roughly cubic in the number of known sound triples".
- Transfer: this deciphers script into sounds, not speech into phones.

**Knight, Nair, Rathod & Yamada, COLING/ACL 2006** (https://aclanthology.org/P06-2065.pdf)
- Worse source model:
  - With the worse (bigram) source model, "EM initially locks onto the correct theory, but task
    performance degrades as it tries to make the ciphertext decoding fit the expected bigram
    frequencies".
  - This is likelihood up and accuracy down: the same symptom as our label drift.
- Learned source model (C/V/SPACE):
  - With both source and channel probabilities floating, EM "correctly clusters letters into vowels
    and consonants, but assigns exactly the wrong labels!"
  - This is textbook label switching, but only because the LM was learned. Our fixed trigram breaks
    that symmetry.
- Extra symbols:
  - With extra source symbols, EM was "hijacking the extra symbols".
  - Bootstrapping from the smaller model fixed it: "bootstrapping is good for dealing with too many
    parameters".
- Other numbers are in the 09-23 report: letter errors 68 -> 10 (2.4%), and 40 or more restarts
  selected by P(c).

**Ravi & Knight, EMNLP 2008** (https://aclanthology.org/D08-1085.pdf)
- Method: exact integer-programming decipherment of 1:1 ciphers.
- Results:
  - A 3-gram LM works from 64 letters up.
  - At 414 letters with a 2-gram, EM gives 10% error against 0.5% for the exact search.
  - Unicity distances for 1-, 2- and 3-gram models: 173, 74 and 50 letters.
- Compute: 3-gram IP takes 450 s at length 32 and grows fast.

**Ravi & Knight, ACL 2011** (https://aclanthology.org/P11-1025.pdf)
- Method: Bayesian decipherment with **type sampling**. Each move samples one new plaintext letter
  for a cipher-symbol type and rewrites every position of that type, which exploits "determinism in
  the deciphering direction".
- Ablations:
  - Type sampling: 97.8% vs 14.5% for point-wise sampling (Zodiac-408, word+3-gram, 5,000 passes).
  - Sparse prior beta 0.01 vs 1.0: 97.8% vs 24.0%.
- 414-letter homophonic cipher with spaces:

  | Method | LM | Accuracy |
  |---|---|---|
  | EM | 2-gram | 30.9% |
  | EM | 3-gram | 32.6% |
  | Bayesian | 3-gram | 95.2% |
  | Bayesian | word+3-gram | 100% |

- Noise: Zodiac-408 has a few symbols that are non-deterministic in the deciphering direction; these
  were tolerated.

**Corlett & Penn, ACL 2010** (https://aclanthology.org/P10-1106.pdf)
- Method: exact A* search over 1:1 keys under a character trigram.
- Results:
  - 100% on the runs that finished.
  - 80% of runs finished within the 12 h cutoff.
  - A 1,000-character cipher took 40:06 h and reached 92.59%.
  - A 3,500-character cipher took 0:38 h and reached 96.30%.
  - Runtime tends to fall as the cipher gets longer.
- Pitfall: "we must use add-one smoothing ... because even one unseen plain-text letter sequence is
  enough to knock out the correct solution".

**Nuhn & Ney, ACL 2013** (https://aclanthology.org/P13-1060.pdf)
- Unigram decipherment is a linear assignment problem, solved by the Hungarian algorithm.
- Bigram decipherment is a quadratic assignment problem, which is NP-hard.
- The objective is the LM score of the mapped cipher, computed from cipher n-gram counts.

**Nuhn, Schamper & Ney, ACL 2013** (https://aclanthology.org/P13-1154.pdf)
- Method: beam search over partial keys. Hypotheses are scored with "heuristics that use the n-gram
  counts rather than the original ciphertext", so the cost is independent of cipher length once the
  counts exist.
- Length-32 ciphers: 4.1% symbol error with a 6-gram, against 38.3% for the *optimal* 3-gram
  solution.
- Beam size and LM order interact: "If the beam size is not large enough, the decipherment accuracy
  decreases when increasing the language model order."
- Zodiac-408: 6-gram at beam 100k gives 94.61% error, against 1.96% at beam 10M.
- Noise: Zodiac-408 has three polyphonic symbols and 17 garbage letters at the end, and was still
  solved.

**Nuhn, Schamper & Ney, EMNLP 2014** (https://aclanthology.org/D14-1184.pdf)
- Method: an improved rest cost and a count-driven extension order.
- Zodiac-408 (54 symbols, 7.55 observations per symbol): solved at beam 26 with an 8-gram, in
  "less than 10s on a single CPU", against 48 h before.
- Beale cipher, by observations per symbol:

  | Part | Symbols | Observations per symbol | Beam | Result |
  |---|---|---|---|---|
  | 2 | 182 | 4.18 | 10M | 157/185 symbols correct |
  | 1 | 299 | 1.74 | - | failed |
  | 3 | 264 | 2.34 | - | failed |

- The extension-order weights moved results between 3/54 and 52/54 correct mappings.

**Nuhn & Ney, ACL 2014** (https://aclanthology.org/P14-2123.pdf)
- Pruned EM uses E-step smoothing of 0.9p + 0.1/|V|.
- At vocabulary 200, exact EM took 224.88 h (97.19%) against 4.14 h for preselection (98.50%).

**Berg-Kirkpatrick, Durrett & Klein, EMNLP 2011** (https://aclanthology.org/D11-1029.pdf)
- Method: block coordinate descent. The alphabet matching is updated by Hungarian or LP under
  "restricted-many-to-many": each character takes part in at most two mappings, and the total number
  of mappings is max(I,J).
- Why the restriction: without it "this objective can always be driven to zero by either mapping all
  characters to all characters, or matching none of the words".
- Restarts: "the block coordinate descent procedure can get stuck at poor local optima", so random
  restarts are used and the best objective is kept.
- Result: Ugaritic-Hebrew, 28 of 33 gold letter mappings recovered.
- Caveat: the objective is edit distance over lexicons, not an n-gram LM.

**Berg-Kirkpatrick & Klein, EMNLP 2013** (https://aclanthology.org/D13-1087.pdf): see the 09-23
report. In brief:
- EM with 100 restarts reaches 90% on Zodiac-408.
- Likelihood tracks accuracy only across large gaps: the best of 1M restarts reaches -1466.5 at 89%
  accuracy, above the gold key's -1467.4 at 92%.

### Decipherment of noisy cluster streams, and speech

**Yin, Aldarrab, Megyesi & Knight, ICDAR 2019** (https://arxiv.org/pdf/1810.04297v3). This is the
closest analogue to our setup.
- Setup:
  - Glyph images are clustered without supervision.
  - "Transcription" error, NEDoA, is measured after the *optimal many-to-one mapping* of clusters to
    gold symbols. That is the analogue of our optimal-relabelling ceiling.
  - Data: Borg is 1,054 characters, 23 symbols, Latin. Copiale is 6,491 characters, 79 symbols,
    German.
- Pipeline ("3-stage": cluster, then noisy-channel EM decipherment, then Viterbi):

  | Cipher | Transcription error (NEDoA) | Pipeline decipherment (NED) | From gold transcription |
  |---|---|---|---|
  | Borg | 0.22 | 0.35 bigram, 0.24 trigram | 0.01 |
  | Copiale | 0.37 gold segmentation, 0.44 auto | 0.46, 0.51 (bigram) | 0.20 |

- On Copiale, the fully connected channel "overcome[s] transcription mistakes by mapping the same
  cluster ID onto different plaintext symbols, depending on context".
- Joint model ("2-stage" LM-GMM, the analogue of our S), P(G) = sum_E P(E) sum_C P(C|E) P(G|C).
  It found two distinct failures:
  - **Model error.** "The gold model does not receive the highest model score" because the strong
    GMM dominates the LM. The fix was P(E)^3.
  - **Search error.** "Even after many EM restarts, we cannot reach a model that scores as well as
    the gold model" (5,000 random restarts). The fix was to restart EM from the pipeline result:
    Borg went 0.35 -> 0.20 with 50 restarts, against 0.15 for a model retrained on gold.
- Joint model results:
  - Borg: 0.20 with a bigram, 0.16 with a trigram. The trigram result is below the 0.22 ceiling,
    because the joint model refits the emissions.
  - Copiale: 0.41 with gold segmentation, 0.50 with auto segmentation.

**Klejch, Wallington & Bell, Interspeech 2022** (https://arxiv.org/pdf/2111.06799)
- The cipher is the 1-best output of a supervised phone recogniser, whose PER is not reported.
- Decipherment WER: 31.0 BUL, 34.5 UKR, and up to 105.3 SWA.
- Flat-start semi-supervised retraining on the deciphered pseudo-labels: 10.7 BUL, 10.3 UKR.
- This is a second instance of "discrete decipherment, then retrain the acoustic side". Details are
  in the 09-23 report.

**Aldarrab & May, ACL 2021**: a supervised neural model on 1:1 ciphers; 20% substitution noise gives
11.48 TER. It does not apply to us (supervised, 1:1 only).

**Hauer et al. 2014** (skimmed): its noise experiments cover only monoalphabetic ciphers with a few
noisy letters. It does not apply.

### Label-free distribution matching (not GAN)

**Chen, Huang, Wang & Deng, 2016** (https://arxiv.org/pdf/1606.04646)
- The unsupervised cost, the expected LM NLL of the outputs, has "many local optima that are badly
  behaved".
- The weight matrix at those optima is "almost rank one", so the output "totally ignores the inputs,
  although its cost is close to that of the global optimal solution".

**Liu, Chen & Deng, NIPS 2017** (https://arxiv.org/pdf/1702.07817)
- Method: swap the cross-entropy direction, so that p_LM weights the model's output n-gram
  probabilities.
  - The Chen 2016 form penalises outputs the LM finds improbable but not missing LM-probable
    outputs. It is **mode-seeking** and converges to "predicting the same output".
  - The swapped form is **coverage-seeking**.
- OCR (29 classes), 2-gram LM:

  | Method | Error |
  |---|---|
  | Primal-dual, coverage-seeking | 9.59% |
  | Chen 2016 cost | 83.37% (= majority guess) |
  | SGD, batch 10k | 56.48% |
  | Supervised | 4.63% |

- LM order (Fused LM): 1-gram 71.25%, 2-gram 10.33%, 3-gram 9.21%.

**Yeh et al., ICLR 2019** (https://arxiv.org/pdf/1812.09323): segmental empirical output distribution
matching at segment rate reaches 41.6 PER on TIMIT. Details are in the 09-23 report.

### Split/merge moves and label switching

**Ueda, Nakano, Ghahramani & Hinton, NIPS 1998, SMEM** (https://proceedings.neurips.cc/paper_files/paper/1998/file/253f7b5d921338af34da817c00f42753-Paper.pdf)
- Method: merge two components that overpopulate one region and split one in an underpopulated
  region.
  - A partial EM re-estimates only the three affected components, with the other posteriors frozen.
  - Full EM follows. The move is accepted only if Q improves.
- Candidates:
  - Merges are ranked by the posterior inner product.
  - Splits are ranked by local KL divergence.
  - "Cmax ≈ 5 may be enough"; the average rank of the accepted candidate was 1.8.
- Cost: about 6x slower than EM.
- Result: on 20-dimensional face data (M = 5), the worst SMEM solution beat the best EM and DAEM
  solutions.
- Limit: it requires the Q function to decompose into a sum over components (eq. 4), as in a
  mixture.

**Hughes, Stephenson & Sudderth, NIPS 2015** (http://www.mit.edu/~wtstephe/papers/2015_NIPS_HughesStephensonSudderth.pdf)
- Setting: sticky HDP-HMM with memoized variational inference.
- Moves:
  - Birth, merge and delete proposals are accepted only if they improve the *whole-dataset*
    objective.
  - A merge sums cached sufficient statistics.
  - To control computation, deletes are proposed only for "states used in 10 or fewer sequences".
- Split-merge samplers "often have low acceptance rates".
- Result: speaker diarization in minutes, against hours for the sampler.
- Limit: states are exchangeable here, with learned transitions.

**Stephens, JRSS-B 2000** (https://www2.stat.duke.edu/~scs/Courses/Stat376/Papers/Mixtures/LabelSwitchingStephensJRSSB.pdf)
- "The root of the label switching problem is that the likelihood is the same for all permutations."
- Its relabelling algorithms choose one of k! *equivalent* modes.
- **This does not apply to us.** The fixed trigram makes the likelihood permutation-dependent, so a
  wrongly labelled phi is a genuinely different, worse local optimum, not an equivalent mode (mine).

## Answers

**(i) Best-fitting method, and which objective.**

Stage 1a, label-free: search over the 40-state permutation.
- The cipher is the phi-decoded state stream with repeats collapsed. Decode it without the LM, so the
  cipher does not already encode the current labelling (mine).
- The score is the trigram log-probability computed from a 40^3 count table (the Nuhn 2013 count
  score).
- Candidate searches:
  - swap local search, or type sampling over state types (Ravi & Knight 2011);
  - beam search (Nuhn 2014);
  - Hungarian on the unigram term for initialisation only (Nuhn & Ney 2013).
- Run many restarts and select the top candidates by the actual log S on a fixed subset.
- Objective choice: under a bijection, the channel term Σ log phi(u|s) is invariant to the
  relabelling. The LM-only score and S therefore rank candidates alike, and the cheap count score is
  safe for generating candidates (mine).

Stage 1b: unit-level homophonic key (500 -> 40), initialised from phi's partition composed with the
stage-1a permutation.
- Moves are type moves: reassign all frames of one unit at once. Point-wise moves gave 14.5% against
  97.8% for type moves (Ravi & Knight 2011).
- The objective must include the channel term: the homophonic likelihood LM + log q(u|p), or S
  itself.
- The LM score of decoded outputs alone rewards degenerate maps (Liu 2017; Chen 2016; BK&K 2011).
- With repeat-collapse it is worse: mapping many units to one phone shortens the string and raises
  its log-probability (mine).
- Guard against this with the channel term, or with a restricted-many-to-many constraint (BK&K 2011).
- SMEM-style candidate ranking gives cheap proposals: overused phones by local KL, paired with unused
  phones.

Stage 2: S-EM from the relabelled phi (Yin 2019). Report whether log S ends above the log S of the
unrelabelled run.

**(ii) Evidence that EM with the LM in the loop leaves permutations unresolved, and that a separate
step fixes them.**

Evidence for:
- Yin 2019: joint EM with 5,000 restarts could not reach the gold score, and a pipeline
  initialisation fixed it (0.35 -> 0.20).
- Ravi & Knight 2011: EM with a 3-gram gets 32.6% on a homophonic cipher; discrete type sampling gets
  95.2%.
- Ravi & Knight 2008: EM 10% error vs exact search 0.5%.
- Knight 2006: likelihood rises while accuracy degrades under a weaker LM.

Against:
- BK&K 2013: plain EM with 100 or more restarts reaches 90% on Zodiac-408.

No paper tests SSL units or 50 Hz frames. The literature therefore does not settle this for us.

**(iii) Expected accuracy for a trigram on a noisy 40-symbol cipher at about 50%-of-oracle
accuracy.** No published point exists.
- Nearest points (Yin, optimal-mapping error -> pipeline decipherment error):
  - Borg: 0.22 -> 0.24 trigram, 0.35 bigram.
  - Copiale: 0.37-0.44 -> 0.46-0.51 bigram.
- In those cases the pipeline landed 0.02-0.13 above the ceiling. The joint re-fit went below it:
  Borg 0.16.
- Our stream is far noisier, so extrapolation is unsupported.
- The deciding measurement is whether log S, or the stage-1 score, ranks the optimal relabelling
  above the current labelling. Take it as an analysis-only diagnostic, never for selection.
  - If it does, the problem is search, and stage 1 applies.
  - If it does not, the problem is model error. The fix is then an LM weight (Yin's P(E)^3; the
    cubed channel in Knight & Yamada 1999 and Knight 2006 is the same lever at decode) or a better
    objective, not search.

**(iv) Pitfalls.**

1. **Length is not binding.**
   - For a 26-letter key, unicity is 50 letters (3-gram) and 64 letters or more empirically (Ravi &
     Knight 2008).
   - Beam decipherment failed at 1.7-2.3 observations per symbol and succeeded at 4.2 (Nuhn 2014).
   - We have thousands of observations per state and per unit.
   - Noise and model error bind instead (mine).
2. **Unseen trigrams.** One unseen n-gram kills the correct key (Corlett & Penn 2010). Use a smoothed
   LM and smoothed channel rows (Nuhn & Ney 2014).
3. **Higher LM order needs a bigger beam** (Nuhn 2013). At 40 symbols a trigram is cheap: about 64k
   counts, and each swap delta costs O(40^2) (mine).
4. **Mode-seeking collapse** of output-LM objectives (Chen 2016; Liu 2017), and degenerate matchings
   (BK&K 2011).
5. **Likelihood tracks accuracy only across large gaps** (BK&K 2013). The gold key may not score best
   (Yin 2019; Knight 2006).
6. **Frequency-rank initialisation hurts under homophony** (Kambhatla 2018, see the 09-23 report).
7. **Restarts are required** (BK&K 2011/2013; Knight 2006).
8. **Pipeline errors propagate** (Yin: "substantial degradation along the pipeline"), so return to
   joint S-EM.
9. **Split/merge moves are costly.** SMEM ran about 6x EM, and needs a per-component Q decomposition
   that our fixed-LM HSMM lacks. Evaluate moves on log S, with memoized statistics as in Hughes 2015.

## Established vs single-paper

Established (two or more independent groups):
- A higher LM order helps when search is adequate: Knight & Yamada 1999, Ravi & Knight 2011, Nuhn
  2013, Liu 2017, Yin 2019.
- Restarts are needed: BK&K 2011/2013, Knight 2006, Klejch.
- Decipher, then refit the acoustic or emission side: Yin 2019 and Klejch 2022.

Single group or single paper:
- Output-LM collapse: Chen 2016 and Liu 2017 share authors.
- Pipeline initialisation fixing a joint-EM search error: Yin 2019.
- Type sampling beating point-wise sampling: Ravi & Knight 2011.

Disagreement:
- Whether EM alone solves homophonic ciphers. Ravi & Knight 2011 get 32.6% (homophonic, 3-gram) and
  28.8% (Zodiac, 100 restarts); BK&K 2013 get 90% at 100 restarts.

## UNVERIFIED or not read

- Ueda et al. 2000 (SMEM journal version): downloaded, not read.
- Petrov et al. 2006 (split-merge latent grammar): downloaded, not read. It refines categories and is
  off-question.
- Frühwirth-Schnatter 2001; Jakobsen 1995; Dhavare, Low & Stamp 2013; Lasry: not reached.
- Chen et al. 2019 and wav2vec-U are GAN methods, read for the problem statement only.

## What this changes for the decision

- A16(a), the S at the optimal relabelling, is the right next measurement. It separates search error
  from model error, as Yin did.
- If S prefers the optimal relabelling, fund a count-based discrete permutation search with restarts,
  then S-EM from its result.
- Merges need unit-level type moves scored with the channel term. Never use a decoded-stream LM score
  alone.
