# Literature: non-local rename/re-partition moves for phi, refit-before-accept, merges, and J's name-free term (2026-09-24)

Role: literature. Dispatch: Q1-Q4 on moves that rename or re-partition phi's 40 symbols under a fixed
trigram prior. Rules applied: GAN work is background only; no transcripts in training or selection;
LM terms trigram or higher. "(mine)" marks my inference. "UNVERIFIED" marks what I could not read.
This report does not repeat lit_em_decipherment_2026-09-23.md, lit_decipherment_relabel_2026-09-24.md,
lit_escape_private_code_2026-09-23.md or lit_length_bias_2026-09-15.md. Where it reuses their
verified facts, it says so.

## Verdict

1. **Moves cannot fix J, because J's optimum is wrong.** The literature has many working non-local
   moves: random swap, SMEM, split-merge MCMC, type-based block moves, and the homophonic nested hill
   climb. Every one was shown only where the objective's optimum is the correct structure. Where the
   objective prefers a wrong structure, a better search makes accuracy worse. Four independent examples:
   - Dhavare 2013, the Zodiac challenge with mismatched English statistics: the slow, better-fitting
     climb scored 4% against 13% for the fast climb.
   - Fränti & Sieranoja 2019: unbalanced clusters are an SSE problem that "even an optimal algorithm"
     inherits.
   - Liang et al. 2010: on the PTSG, likelihood rose while F1 fell.
   - Liang & Klein 2008: EM started from gold loses accuracy.

   Our stage-1 result has the same shape. All 124 finals beat gold on J, and oracle renaming lowers J.
   Funding a better J search is not supported. The J objective has to change first.

2. **J's name-free H(S) reward is a documented pathology.** Kearns, Mansour & Ng 1997 show that
   hard-assignment clustering carries a partition-entropy (balancing) term, and that learned mixing
   weights remove it. Johnson 2007 found that EM's HMM states are "relatively uniform" while gold tags
   are skewed. KMN 1997 also derive a bias toward components "as different as possible", which is our
   sharp-cluster finals.

   The textbook fix is a frame-rate class-prior term. With ML weights it only cancels the bias and
   carries no names. With weights fixed from general phone frequencies it becomes a name-bearing
   -KL(occupancy || prior) term (mine). That is a unigram occupancy term, which may conflict with the
   rule that LM terms be trigram or higher; the orchestrator must rule on it.

   Against this, see E1-E2: the project's own ad hoc decomposition says H(S) is the minor part of the
   gap, and the one published test of an occupancy constraint was weak.

3. **Refit before accepting a move is established for splits and relocations, not for merges.**
   - Refit needed: Jain & Neal ("extremely poorly" without refit), SMEM (partial then full EM), random
     swap (2 k-means iterations; 2 is best, "the exact value is not critical"), STACS (local
     Baum-Welch), and Dhavare (each count change scored only after the inner key search with 40
     restarts).
   - No refit needed: Petrov's merges.
   - A pure relabel of a fixed partition cannot pass merged states: a 1:1 map scores duplicates as
     errors (Johnson 1-to-1 vs many-to-1; our A20).

4. **Duplicated and starved states are the textbook k-means "missing/extra centroid" error.**
   Relocation moves with a short refit fix it when the objective is right: Fränti's random swap reached
   CI 0 on every ground-truth set. The one precedent for a fixed LM with a many-to-one key is Dhavare's
   outer climb on symbol counts. It matched oracle counts, but only in noiseless ciphers of up to 65
   symbols, and it was "likely poor" at 100 or more symbols at any length. We have 500 noisy units.

   No paper compares split-merge criteria head-to-head with occupancy-driven reallocation. The
   literature does not settle which fixes merges first.

## What this changes for the decision (mine)

- **Do not fund more J search.** Change the objective first.
  - The candidate literature-backed change is a frame-rate occupancy term. It removes the H(S) reward
    (KMN). With a fixed phone-frequency prior it also adds a name signal. It needs a ruling under the
    trigram-or-higher rule.
  - Pre-register it as a stage-0 read: does J(gold) rank first among the 124 finals and the A20
    oracle renames?
  - Also give it a destroyed-structure null. Johnson's own fix, fewer states, is closed to us, because
    we need 40 named symbols.
- **If moves are proposed, put them under S, not J.** S already ranks the basin (PER 0.35-0.47) above
  random-init EM.
  - The literature-shaped move: take one symbol from a duplicated phone region (occupancy or
    posterior-correlation criterion as in SMEM), reinitialise it on a starved region, run 2 EM
    iterations, and accept only if held-out S improves.
  - Limits: its ceiling is the basin, not gold, because S does not rank gold first (finding 5). It
    does not address drift inside the basin.
- **Analyses first, CPU-cheap and both from existing phis** (mine):
  - (a) Move the J(gold)-versus-finals ranking under an occupancy-corrected J before any new search.
  - (b) On the A15 phis, count "missing/extra" symbols per phone (a CI analogue). Test whether a single
    relocation plus 2 EM iterations lowers S. This is Fränti's success test transplanted.

## Evidence AGAINST our hypotheses (stated first on purpose)

- **E1. H(S) may be the minor part of J's gap.** The key-search audit's ad hoc decomposition
  (audit_keysearch_s1_2026-09-24.md; flagged there as not for citation and not a registered reader)
  attributes only about +0.03 to +0.07 of the +0.20 to +0.23 emission advantage to the deterministic
  H(S)-H(U) term. The rest comes from d_min run absorption: gold relabels 14.3% of held-out frames, the
  selected keys 9.9-11.3%.
  - If that holds, an occupancy fix addresses a minority of the gap. The larger part is a smoothness
    reward for keys whose unit runs fit the d_min >= 2 topology.
  - No published analysis of this second term exists that I found (mine). A registered reader must
    confirm the split before any objective change is funded.
- **E2. The published occupancy constraint was weak.** Merialdo 1994 froze the marginal p(t) in the
  supervised-init drift regime. It cut added errors by only about 7% (2141 vs 2196 errors at iteration
  10), against about 49% for anchoring p(t|w) to the init (verified in lit_length_bias_2026-09-15.md).
  - This bears on drift inside the basin (finding 5), not directly on random-init merges.
  - Johnson 2007 reports that Wang & Schuurmans 2005's frequency-preserving constraint helped in the
    lexicon-supervised setting. The primary paper is UNVERIFIED (IEEE paywall).
- **E3. Better search can hurt under a mismatched prior (Dhavare 2013, Table 13).**
  - Setup: MysteryTwister Zodiac challenge, English digram statistics.
  - Result: the slow outer climb scored 4.00% against 13.00% for the fast climb.
  - The authors' explanation: the slow climb matched the English matrix better, "it match[es] the true
    plaintext less".
  - With Zodiac-408 statistics the order reversed: 84% against 70%. Our trigram is a LibriSpeech-LM
    phone trigram against read speech, probably a closer match than English vs Zodiac, but untested (mine).
- **E4. Scale.** Dhavare: "for homophonic substitutions with 100 or more cipher symbols, our results
  are likely to be poor, regardless of the length of the ciphertext".
  - Our key has 500 units into 40 symbols, and the channel is noisy (gold-key frame purity 0.644 per
    the audit). The ciphers are deterministic.
  - Nothing in the decipherment literature covers our regime (mine).
- **E5. Greedy block moves collapse HMMs** (Liang, Jordan & Klein 2010). Deterministic TYPE-greedy was
  "disastrous for the HMM", and token annealing hurt it. Unit-level best-improvement moves on a key are
  greedy type moves (mine).
- **E6. Random swap finds a near-optimum, not a unique one.** On real image data it lands on plateaus
  with SSE within 0.3% but 9% cluster-level differences (Fränti 2018). Acceptance is rare: 8 of 5000
  trial swaps were accepted, and 3 of those fixed a cluster error.
- **E7. 1:1 structure matching cannot represent duplicates.** GW OT uses uniform marginals and is a
  QAP. It fails on distant pairs: EN-FI P@1 6.8 without normalisation, 18.3 with (Alvarez-Melis &
  Jaakkola 2018). Our problem is many-to-one (mine).
- **E8. Posterior sparsity does not stop frequent classes from being duplicated.** Under L1/Linf PR,
  "Sparse tends to spread the nouns over 4 different hidden states" (Ganchev et al. 2010, PTB17), and
  so loses to VEM on 1-1 accuracy.
- **E9. Gold-init drift is generic.** EM "re-purposes under-utilized states to better capture
  distributional similarities" and "reinforces the systematic mistakes" of the initializer on
  iteration 1 (Liang & Klein 2008, WSJ HMM/PCFG/DMV).
  - A short refit after a correct rename can therefore also move away from it. A 2-iteration refit
    limits this, but the literature gives no bound (mine).

## Q1. Non-local moves: what, where, gain over EM, scaling to 40 states x 500 unit types

Label switching (Stephens 2000) does not apply. The fixed trigram breaks the permutation symmetry
(verified in lit_decipherment_relabel_2026-09-24.md). What we face is a relocation and naming problem
under a symmetry-breaking prior, which is decipherment plus clustering relocation.

**Random swap (Fränti, J Big Data 2018)**
- Move: remove a random prototype, place it at a random data vector, repartition locally, run 2
  k-means iterations, accept only if SSE improves.
- Key observation: "it is not even necessary to swap one of the redundant prototypes but simply
  removing any prototype in their immediate neighborhood is enough".
- Gain, measured in CI (clusters in error):

  | Set | single k-means | repeated k-means, 20,000 repeats | random swap reaches CI 0 after |
  |---|---|---|---|
  | Birch1 | 7 | 3 | about 606 swaps |
  | Birch2 | 18 | 9 | about 2067 swaps |
  | Unbalance | 3 | 1 | about 98 swaps |

  Random swap takes under a minute. Repeated k-means succeeds only on Unbalance, in 60% of trials,
  averaging 17,538 repeats.
- Cost: O(N) in the data and O(k^2) in clusters.
- Scaling (mine): k = 40 suggests hundreds to low thousands of trial moves, each with 2 EM iterations
  on the HSMM. That is feasible only on a subset or with memoized sufficient statistics. Placing a
  symbol "at a data vector" has no direct HSMM analogue. The natural analogue is reinitialising the
  symbol's emission from one unit cluster, or from a region where duplicated symbols overlap.

**K-means initialisation study (Fränti & Sieranoja, Pattern Recognition 2019)**
- K-means "fails to relocate the centroids globally".
- Errors grow linearly with k, at 13-16% of centroids. Repeats help: 100 repeats cut erroneous
  clusters from 15% to 1%.
- Unbalance is the worst case, and that is an objective property: "Even an optimal algorithm
  minimizing SSE would end up with the same incorrect result".
- Our random-init EM leaves 11-19 of 40 phones without a symbol, roughly 28-48% by the same count
  (mine). That is far beyond their regime.

**SMEM (Ueda et al. 2000; verified in lit_escape_private_code_2026-09-23.md)**
- Merge plus split, partial EM on the affected components, then full EM; accepted if Q improves.
- Evaluated on mixtures only. Cost 6-8.7x EM, with Cmax about 5 candidates.
- On face data (M = 5) the worst SMEM run beat the best EM run.
- Scaling (mine): 780 merge pairs and 40 split candidates, ranked cheaply by posterior correlation and
  local KL. Only the top 5 get a refit.
- No sequence-model evidence exists. The Minagawa 2002 critique of the Q-based acceptance test is
  UNVERIFIED beyond its abstract.

**Split-merge MCMC (Jain & Neal; verified in the prior report)**
- Conjugate DP mixture, toy data.
- Random splits without refit perform "extremely poorly". 4-6 restricted Gibbs scans are recommended.
- Acceptance was 1.5-4.3% even with refit.
- Chang & Fisher 2013 (NIPS, full text read) keep two persistent sub-clusters per cluster, so a split
  proposal is always ready. They argue this avoids Jain & Neal's scan-count trade-off. It was shown on
  DPMMs (synthetic data, MNIST); results are in figures only.

**STACS (Siddiqi, Gordon & Moore, AISTATS 2007)**
- HMM state splitting. Each candidate is designed by Baum-Welch/Viterbi on the split state's data
  only, and chosen by Viterbi likelihood, then BIC.
- Test log-likelihood at N = 40:

  | Dataset | STACS | best of 5 BW restarts |
  |---|---|---|
  | Robot | -1.75 | -1.78 |
  | Mocap | -4.32 | -4.37 |
  | Mlog | -8.25 | -8.38 |
  | AUSL | -2.89 | -2.99 |
  | Vowel | -4.34 | -4.33 (BW better) |

- The authors: it is "a good way to learn HMMs even if the desired number of states is known".
- Transfer is limited: exchangeable states, learned transitions, continuous observations, and no
  fixed prior (mine).

**Type-based block MCMC (Liang, Jordan & Klein, NAACL 2010)**
- Collapsed Dirichlet models: HMM with K = 45 on WSJ, USM segmentation, PTSG.
- TYPE beats TOKEN on likelihood and accuracy for all three. Results are in figures; no EM baseline.
- For the HMM, the type unit is (previous state, next state, word).
- Scaling (mine): a unit-to-symbol key move is a type move over 500 unit types x 40 symbols, which the
  J search already does. The collapsed, count-based rescoring is what makes it cheap. Stochastic, not
  greedy, acceptance is the published lesson (E5).

**Homophonic nested hill climb (Dhavare, Low & Stamp, Cryptologia 2013; full text read)**
- Three layers:
  - outer climb over the counts (n_a, ..., n_z), i.e. how many cipher symbols each letter owns;
  - 40 random initial keys consistent with those counts;
  - inner Jakobsen swap climb scored against a fixed English digram matrix.
- The outer climb starts from counts proportional to letter frequency, with every count at least 1.
  It moves one unit of count between letter pairs.
- Results, English, 27 plaintext symbols, 300-10,000 characters, 28-100 cipher symbols:
  - the full algorithm (Case 4) roughly matched the oracle-count Case 2;
  - at least 80% of characters recovered for 42 or fewer symbols at 1000 or more characters, and for
    75 or fewer at 3000 or more.
- The authors say trigram scoring would make "swapping and scoring ... extremely complex".
- This is the closest published analogue to "reallocate duplicated symbols to starved ones under a
  fixed LM" (mine). Its limits are E3 and E4.

**Beam/A\* decipherment (Corlett & Penn 2010; Nuhn & Ney 2013; Nuhn et al. 2013/2014)**
- Verified in lit_decipherment_relabel_2026-09-24.md: unigram = LSAP, bigram = QAP, and a beam solves
  Zodiac-408 homophonic (54 symbols into 26).
- Our 500-into-40 noisy-channel problem is outside the tested range.

**GW OT (Alvarez-Melis & Jaakkola, EMNLP 2018)**
- The discrete form is a QAP, solved as a sequence of Sinkhorn problems at O(N1^2 N2 + N1 N2^2).
- P@1 EN-ES 81.7 and EN-RU 45.1/43.7 on the Conneau set. It fails on distant pairs (EN-FI).

**Wasserstein-Procrustes (Grave, Joulin & Berthet, AISTATS 2019)**
- Naive alternation "quickly converges to bad local minima".
- Toy accuracy, random init 0.70/0.62/0.39/0.59 against convex-relaxation init 1.00/1.00/0.987/0.985.
- A convex relaxation for initialisation is the transferable idea for a 40x40 symbol-to-phone
  matching (mine). It is still 1:1 (E7).

**LNS/ILS (Shaw 1998; Lourenço et al.)**
- Primary texts not read: UNVERIFIED.
- Random swap is an ILS-type perturb-refine-accept loop in clustering, and it is the tested instance
  (mine).

## Q2. Must the partition be re-estimated after a rename, and do pipelines evaluate before or after a short refit?

- **Split and relocation proposals: after a refit, established across groups.**
  - Jain & Neal: restricted Gibbs scans, then a final full scan "a necessity".
  - SMEM: partial plus full EM before the Q test.
  - Random swap: local repartition plus 2 k-means iterations. 2 is "slightly better" than 1, and
    "three or more iterations do not provide enough additional benefit" (Fig. 14).
  - STACS: local split-state Baum-Welch.
  - Hughes 2015: acceptance on the whole-data objective (prior report).
  - Dhavare: a count change is scored only after 40 re-seeded inner key climbs.
- **Merges: no refit needed** (Petrov 2006/2007; the exact no-refit criterion was no better, per the
  prior report).
- **Pure permutations under a fixed LM and a deterministic channel** (Jakobsen swaps, Ravi & Knight,
  Nuhn) are scored directly. There is nothing to re-adapt, because the channel is the key itself.
- For us (mine):
  - J recomputes ML emissions from counts, so a J move is already implicitly re-adapted, like a
    collapsed sampler. Its failure is not missing refit.
  - Under S, phi's emissions m(u | s, d, pos) are tied to durations and positions. A relabel without
    refit carries mismatched duration and position tables, so a raw-relabel S penalty is expected
    whatever the rename's merit.
  - Our relabelled EM phis raise S by 0.12-0.31. That is consistent with the literature and is not
    evidence against renaming. The comparison that would count is S after a matched 2-iteration refit
    of the renamed and unrenamed phi.
- Counter-risk: E9. The first EM iterations reinforce initializer biases and repurpose under-used
  states. A short refit may partly undo a correct rename.

## Q3. Do merged or duplicated states defeat a rename, and what fixes merges first?

- **Yes, a 1:1 rename cannot score duplicates, and this is established.**
  - Johnson 2007: EM(50) 1-to-1 0.40 against many-to-1 0.62 on PTB WSJ, 1000 iterations, 10 runs,
    "presumably several hidden states are being mapped onto a single POS tag".
  - Ganchev et al. 2010: PR still spreads nouns over 4 states.
  - Christodoulopoulos et al. 2010 (EMNLP): 1-to-1 and VI peak at about 25-30 clusters for 45 gold
    tags (Brown clustering, WSJ).
  - Our A20 (fixed partitions, Hungarian 1:1 and argmax many-to-one renames both lower J) is the
    same phenomenon under J.
- **What fixes duplication.**
  - Relocation (random swap) fixes missing/extra centroids with a refit, and need not pick the exact
    redundant member.
  - A count-level outer climb (Dhavare) fixes allocation under a fixed LM.
  - SMEM merges the most correlated pair and splits the worst-fit component.
  - Hughes 2015 deletes states used in 10 or fewer sequences.
  - Łańcucki et al. 2020 re-initialise dead VQ codes with k-means++ from a reservoir. On supervised
    WSJ, PER went from 16.9 to 9.8 as code perplexity rose from 16 to 574. That design maximises usage,
    the opposite of the skew we need, so transfer is weak.
- **Which fixes merges first: not settled.** No head-to-head exists. What would settle it (mine): on
  the A15 EM phis, a paired comparison of two move classes, each with a 2-iteration refit and
  acceptance on held-out S:
  - (A) SMEM-criterion merge plus split;
  - (B) occupancy-driven relocation from duplicated to starved phones, with the starved phones found
    by comparing decoded symbol occupancy with the trigram's unigram marginal, not with gold.

  Report S, PER, within-class accuracy and the missing/extra symbol count per class.

## Q4. Is J's name-free H(S) term a known pathology?

- **Yes.** Kearns, Mansour & Ng 1997 (UAI) decompose the hard-assignment loss into a weighted KL
  modelling term plus a partition-entropy term. K-means is "strongly influenced by the entropy of the
  partition"; "in EM this influence is entirely absent". Adding a learned weight "removed the bias
  towards finding an 'informative' partition". They also derive a bias toward components "as
  'different' as possible". The toy setting is two components.
- For a hard key with ML emissions, J's emission term is -H(U|S) = H(S) - H(U) per frame. That is
  exactly unweighted classification likelihood (mine). The trigram in J acts per segment on the
  collapsed string, so it does not supply the frame-rate log-weight that would cancel H(S) (mine).
- **Empirical support in sequence models.** Johnson 2007: EM "tends to assign relatively equal numbers
  of tokens to each hidden state" while 6 tags cover more than 55% of tokens. Fränti & Sieranoja 2019:
  SSE splits large clusters and merges small ones, at the optimum.
- **Disagreement.** KMN say the balancing influence is absent in (mixture) EM. Johnson finds flat
  state usage under soft EM in HMMs. So the balancing can also arise from optimisation or
  model-structure effects, not only from hard assignment. Our random-init soft EM merges too, which
  fits Johnson (mine).
- **Frame versus segment weighting is a known design choice.** ES-KMeans (Kamper et al. 2017)
  weighs each segment's score by its length, because per-segment scoring favours fewer segments. No
  paper analyses a frame-rate channel term against a segment-rate LM term (mine; not found).
- **Hard EM in HMMs gives sparser, faster-converging but biased solutions.** Allahverdyan & Galstyan,
  NIPS 2011: exactly solvable toy only; background.
- **Caveat: E1.** The name-free advantage in J may come mostly from d_min absorption, not H(S).

## Established versus single-paper

Established (several groups):
- Search helps only where the objective prefers the truth (BK&K 2013, Knight 2006, Dhavare 2013,
  Fränti 2019, Liang & Klein 2008).
- Splits and relocations need a refit before acceptance (Jain & Neal, Ueda et al., Fränti, Siddiqi
  et al., Hughes et al.).
- EM duplicates frequent classes and 1:1 scoring punishes it (Johnson 2007, Ganchev et al. 2010,
  Christodoulopoulos et al. 2010).
- Gold-init EM drifts (Merialdo 1994, Elworthy 1994, Liang & Klein 2008).

Single paper or single group:
- the homophonic count climb (Dhavare);
- random swap's CI results (Fränti's group, on their own benchmark sets);
- STACS gains;
- SMEM gains (small mixtures);
- the occupancy-constraint effect size (Merialdo about 7%; Wang & Schuurmans UNVERIFIED).

## UNVERIFIED or not read

- Wang & Schuurmans 2005 (IEEE, paywalled): only Johnson's description.
- Minagawa 2002: abstract only (prior report).
- Stolcke & Omohundro model merging: URL not found.
- Graca et al. 2009: the NeurIPS URL returned a stub. Ganchev et al. 2010 JMLR covers the same PR
  POS experiments and was read.
- Brand 1999 entropic prior, Xiong et al. 2009 (k-means uniform effect), Fritzke LBG-U/ELBG,
  Jakobsen 1995, and the LNS/ILS primaries: not read.
- Liang et al. 2010 and Chang & Fisher 2013 results are in figures, so only qualitative claims are
  taken from them.

## What would settle the open questions (mine)

1. A registered reader decomposing J(key) - J(gold) into the H(S)-H(U) part and the d_min-absorption
   part over the 124 finals and the A20 renames. This decides whether an occupancy term can move the
   ranking at all.
2. J recomputed with a frame-rate occupancy term (ML weights, and separately fixed weights from the
   trigram's unigram marginal times the general-knowledge mean durations), with the ranking of gold
   among the finals reported. Include a destroyed-structure null.
3. The Q3 paired move-class trial under S on the A15 phis.

## Sources (full text read this round unless noted)

- Johnson 2007, EMNLP: https://aclanthology.org/D07-1031.pdf
- Kearns, Mansour & Ng 1997, UAI: https://arxiv.org/pdf/1302.1552
- Liang, Jordan & Klein 2010, NAACL: https://aclanthology.org/N10-1082.pdf
- Liang & Klein 2008, ACL: https://aclanthology.org/P08-1100.pdf
- Alvarez-Melis & Jaakkola 2018, EMNLP: https://aclanthology.org/D18-1214.pdf
- Grave, Joulin & Berthet 2019, AISTATS: https://arxiv.org/pdf/1805.11222
- Fränti & Sieranoja 2019, Pattern Recognition: https://cs.uef.fi/sipu/pub/KM-Init-PR-2019.pdf
- Fränti 2018, J Big Data (random swap): https://link.springer.com/content/pdf/10.1186/s40537-018-0122-y.pdf
- Siddiqi, Gordon & Moore 2007, AISTATS: http://proceedings.mlr.press/v2/siddiqi07a/siddiqi07a.pdf
- Dhavare, Low & Stamp 2013, Cryptologia (author copy): https://www.cs.sjsu.edu/faculty/stamp/papers/topics/topic20/homophonic2.pdf
- Ganchev, Graca, Gillenwater & Taskar 2010, JMLR: https://www.jmlr.org/papers/volume11/ganchev10a/ganchev10a.pdf
- Christodoulopoulos, Goldwater & Steedman 2010, EMNLP: https://aclanthology.org/D10-1056.pdf
- Chang & Fisher 2013, NIPS: https://proceedings.neurips.cc/paper/2013/file/bca82e41ee7b0833588399b1fcd177c7-Paper.pdf
- Łańcucki et al. 2020, IJCNN: https://arxiv.org/pdf/2005.08520
- Kamper, Livescu & Goldwater 2017, ES-KMeans: https://arxiv.org/pdf/1703.08135
- Allahverdyan & Galstyan 2011, NIPS: https://arxiv.org/pdf/1312.4551
- Elworthy 1994, ANLP: https://aclanthology.org/A94-1009.pdf
- Reused from prior reports (verified there): Ueda et al. SMEM, Jain & Neal, Petrov 2006/2007,
  Hughes 2012/2015, Stephens 2000, Merialdo 1994, BK&K 2013, Corlett & Penn 2010, Nuhn & Ney 2013,
  Nuhn et al. 2013/2014.
- Project context (not literature): reports/audit_keysearch_s1_2026-09-24.md,
  reports/impl_a20_oracle_names_2026-09-24.md.
