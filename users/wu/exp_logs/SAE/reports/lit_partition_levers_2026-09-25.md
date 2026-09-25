# Literature: partition levers for phi (TP-B relocation moves, coarse-to-fine unit inventory), 2026-09-25

Role: literature. Dispatch: Q1-Q4 on split-merge/relocation moves and coarse-to-fine inventories, to ground TP-B and the
coarse-to-fine inventory (SAE_4A_rename.md, "Proposed training changes") before they go to the user. Rules applied:
pure unsupervised, GAN work background only, no labels in training or selection, LM terms trigram or higher.
"(mine)" marks my inference. Everything cited was read in full text in this session unless marked UNVERIFIED.
Not repeated here: lit_phi_name_signal_2026-09-24.md, lit_phi_rename_moves_2026-09-24.md,
lit_escape_private_code_2026-09-23.md, lit_em_decipherment_2026-09-23.md. Where I reuse their facts I say so.

## Verdict

1. The failure the project measured (EM locked on partitions with 7-12 duplicated and 8-13 unclaimed phone types, while
   S ranks the basin first between basins) is the textbook "too many components here, too few there" local optimum.
   The literature treats it as a SEARCH error, with the global optimum correct, and fixes it by changing the partition
   search, never the objective: SMEM (Ueda et al. 1998/2000), random swap EM (Zhao et al. 2012), split-merge MCMC
   (Jain & Neal), birth/merge/delete (Hughes et al. 2015), restarts (Berg-Kirkpatrick & Klein 2013), over-seed-then-
   prune (Dasgupta & Schulman 2007), model merging (Stolcke & Omohundro 1994). Theory: Jin et al. 2016. This supports
   TP-B's premise, with the project's own caveat that S is only right BETWEEN basins (E1, E11).
2. No published move operates on a NAMED, fixed inventory. Every method above has exchangeable components (or latent
   substates under observed categories, Petrov 2006). In ours, the 40 slots carry LM names. A relocation therefore has
   to choose a name for the relocated mass, and that is a naming decision that S scores wrongly on found partitions
   (AN-2). The move must be designed as named merge + split + name choice (below). This is (mine), not published.
3. No paper accepts moves on HELD-OUT likelihood. SMEM accepts on training Q, RSEM and Hughes on the training
   likelihood/bound, Jain & Neal by MH. The one within-paper held-out read I found went the wrong way (SMEM 2000 toy:
   training LL up, test LL down as printed). Held-out S acceptance is a project design choice without precedent.
4. No published work shows that coarsening a unit inventory makes naming recoverable under a fixed LM. The support is
   one theory result (Wang et al. 2023, purity-1 units), one GAN count sweep (Liu et al. 2018, background) and one
   untested suggestion (DiscoPhon 2026). Coarsening measurably lowers PNMI (HuBERT: 0.686 at K=500 to 0.575 at K=100).
   The literature does not settle the inventory; a count sweep with its purity ceiling reported would.
5. Nothing found directly contradicts funding TP-B or the inventory. Four findings constrain them: (a) likelihood near
   the top does not track accuracy (BK&K 2013; Stolcke 1994; RSEM 2012; our E11 and the PER-0.86 basin-bar arm), so
   S acceptance needs a label-free second read; (b) SMEM's mean LL was below repeated EM on 6 of 7 sets in Zhao et al.
   2012 (by 0.01-0.09 per sample); (c) the SMEM split criterion (local KL) is degenerate for free multinomial emission rows (mine); (d) random
   relocation needs t > M^2 trials (Zhao et al. 2012), so one candidate every 4 sub-epochs is far sparser than any
   published schedule.

## Q1. Split-merge and relocation moves: acceptance, candidates, refit, fixed-inventory evidence

**SMEM, Ueda, Nakano, Ghahramani & Hinton, NIPS 1998; J. VLSI Signal Processing 26:133-140, 2000**
(https://proceedings.neurips.cc/paper/1998/file/253f7b5d921338af34da817c00f42753-Paper.pdf;
https://mlg.eng.cam.ac.uk/pub/pdf/UedNakGha00b.pdf)
- Move: simultaneous merge(i,j) + split(k), so M stays fixed. Merge init is the posterior-weighted average (eq. 5);
  split init is a perturbation (eq. 6).
- Refit: partial EM on i', j', k' only, with posteriors renormalised so their mass is preserved (eq. 7), then full EM.
- Acceptance: "if Q is improved, accept", training Q, not held-out. Up to Cmax (about 5) candidates are tried in rank
  order, and the first improving one is accepted.
- Candidates: Jmerge = inner product of the two posterior vectors (eq. 8); Jsplit = local KL between the data density
  near k and model k (eqs. 9-10). Average rank of the accepted candidate 1.8 (SD 0.9). 4.7 (SD 0.8) accepted
  operations per run at M=5.
- Cost: 155 EM steps against 47 (VLSI Table 2). The authors give about 155 x 1.8 / 47, roughly 6 times EM.
- Results: face data (20-dim, M=5, 103 train / 103 test, 10 k-means inits). The worst SMEM run beat the best EM run:
  train -145.1 vs -148.2, test -155.9 vs -159.8 per sample.
- Negative detail: 2-D synthetic example (VLSI 2000, Sec. 4). Per-sample LL is -2.210 (EM) vs -2.162 (SMEM) on
  training data, but -2.198 (EM) vs -2.223 (SMEM) on test data. The text nonetheless says "improved for both". As
  printed, the training-accepted solution was worse held-out. It is either a misprint or a real held-out loss; I
  cannot tell which.
- Scope: mixtures whose Q decomposes over components. Components are exchangeable; there are no names.

**Minagawa, Tagawa & Tanaka, Neural Computation 14(6), 2002.** UNVERIFIED beyond the abstract (MIT Press, no full
text). The abstract says Q-based acceptance "may pick up a distribution with smaller likelihood". The critique is
echoed in full text by Zhao et al. 2012: "by doing so the global maximum might be accidentally rejected. ... we
therefore use improvement of the log-likelihood as the acceptance rule."

**Random swap EM (RSEM), Zhao, Hautamäki, Kärkkäinen & Fränti, Pattern Recognition Letters 33:2120-2126, 2012**
(http://www.sipu.uef.fi/pub/RandomSwapEM-PRL2012.pdf)
- Move (Alg. 3): remove a uniformly random component, and add one with its mean at a uniformly random data point
  (eq. 7). This is a pure relocation.
- Refit: EM to convergence (threshold 1.53e-5).
- Acceptance: the training log-likelihood improves (line 8).
- Schedule: "the probability of a single good swap is at least 1/M^2, and t > M^2". For M=40 that is more than 1,600
  random trials (mine).
- Negative result for SMEM: in Table 3 (mean LL over 50 runs, diagonal GMMs), SMEM's mean LL is below repeated EM
  on 6 of 7 sets, by 0.01-0.09 per sample (S1 -26.25 vs -26.20, R15 -6.57 vs -6.48). S3 is the only gain. SMEM
  used Cmax=20.
- Fig. 3: "in terms of parameter estimation, likelihood is not always a good proxy". The closest to ground truth is
  not always the best LL (2 of 4 sets).
- Setting: 2-16 dims, 15-20 components, all unnamed.

**Jain & Neal, TR 2000 / JCGS 2004** (https://glizen.com/radfordneal/ftp/mixsplit.ps; PostScript text extracted)
- Move: pick two observations i and j at random. Build a launch state with intermediate restricted Gibbs scans;
  accept by MH (eq. 5).
- Acceptance rate (Table 7): 1.5% with 1 scan, 3.1-4.3% with 3-100 scans.
- Sec. 4.4.4: 4-6 intermediate scans are "the best compromise". "For data sets in which the number of observations
  per mixture component is large, more intermediate Gibbs sampling scans may be required." A final Gibbs scan after
  the MH updates "has been shown to be a necessity". Simple random split "performs extremely poorly".
- Discussion (Sec. 5), a labelling problem: the correct split is proposed with probability "as low as 25%", and i and
  j can "end up in the 'wrong' mixture components" because their labels are fixed during the scans. This is the
  closest published statement to our naming issue. It is not solved there.
- Setting: conjugate DP mixture of binary data, K variable, no names.

**Hughes, Stephenson & Sudderth, NIPS 2015 (HDP-HMM)**
(https://proceedings.neurips.cc/paper/2015/file/2e65f2f2fdaf6c699b223c61b1b5ab89-Paper.pdf)
- Births: a random interval in one sequence is cut into two contiguous blocks with new states, at the cut that
  maximises L_data. Merges: pairs for which every term except entropy improves; they argue this beats a
  correlation heuristic. Deletes: states used in 10 or fewer sequences, refit on those sequences.
- Acceptance: the whole-dataset variational bound improves (training).
- Toy (8 true states, 50-state redundant init): delete/merge reached K=8 and Hamming distance 0 in 100 laps. The
  sampler still had K=10 after 2,000 laps.
- Mocap: Hamming 0.30 (K=13) against 0.49 for the sampler.
- K variable. Evaluation maps states to labels with an alignment, so names are external.

**Petrov, Barrett, Thibaux & Klein, ACL 2006** (https://aclanthology.org/P06-1055.pdf)
- Split every symbol in two with 1% noise and retrain by EM. Then merge back 50% by the approximate likelihood loss
  Delta computed from inside/outside statistics, without refit.
- Hierarchical estimation beats direct: 83.7 vs 83.2 F1 at 4 subsymbols, 88.4 vs 87.3 at 16. Sec. 2.1: "even
  restarting may not be sufficient".
- Subsymbols sit under OBSERVED treebank categories, so there is no naming problem.

**Stolcke & Omohundro, ICSI TR-94-003, 1994** (https://arxiv.org/pdf/cmp-lg/9405017)
- Fine-to-coarse: start from an HMM that memorises the data and merge states best-first (5-step lookahead) by the
  Bayesian structure posterior.
- Case study I (ac*a | bc*b): Baum-Welch "vary wildly with the choice of initial conditions". Perfect structures in
  2/10 runs with 6 states and 3/10 with 10 states (minimal sample); 3/10 and 7/10 with 20 random strings. Failures
  "redundantly allocated states" and "wasted" states. Merging found the target on both samples.
- "The log likelihood alone can be deceptive ... it may appear close to optimal even though the model structure
  represents poor generalization."

**Theory of the duplicated/starved optimum**
- Jin, Zhang, Balakrishnan, Wainwright & Jordan, NIPS 2016 (https://arxiv.org/pdf/1609.00978). Thm 1: for M >= 3
  well-separated spherical Gaussians the population likelihood has bad local maxima, arbitrarily worse than the
  global one. The configuration: one center between two true centers, two centers on one. Thm 2: random-init EM
  reaches such points with probability at least 1 - e^{-cM}. Centers stay trapped in the group where they start. The
  global optimum is the truth, so this is search error.
- Dasgupta & Schulman, JMLR 8, 2007 (https://www.jmlr.org/papers/volume8/dasgupta07a/dasgupta07a.pdf). Start with at
  least (1/w_min) ln k centers, run one EM round, prune starved centers, then prune to k by a Gonzalez-style distance
  criterion, then run a second EM round. Pruning only starved clusters "is not enough": two estimates can "share the
  same cluster, each with relatively high mixing weight", which "frequently occurs in simulations".

**How a move could be accepted on held-out S without labels.** No precedent. Published practice is training
objective after refit (SMEM, RSEM, Hughes) or MH (Jain & Neal). Zhao et al. and the Minagawa abstract favour the
actual likelihood over Q. For us (mine): S is a held-out marginal likelihood, so it satisfies the "likelihood, not Q"
point. The acceptance statistic should be the paired per-utterance S delta on the 260 set against the measured
same-config S spread. It must be read AFTER refit: SMEM Fig. 2 shows LL drops right after a move and recovers during
refit.

## Q2. Coarse-to-fine inventories: class count, criterion, and whether naming becomes recoverable

| Source | Classes | Criterion | Naming recoverable? |
|---|---|---|---|
| Petrov 2006 | 2^k latent substates per observed category | split in two + EM; merge by approx. likelihood loss | not applicable (categories observed); coarse-to-fine appears only as parse pruning |
| Lee & Glass 2012 (https://aclanthology.org/P12-1005.pdf) | DP-HMM, many units | Bayesian nonparametric | no naming; clusters are context-dependent variants (19/20/21 all /ae/) |
| Ondel et al. SLTU 2016 | truncation 200; 85-197 units found | VB/Gibbs DP-HMM | no naming; units "drastically increase" without boundaries |
| Sicherman & Adi, ICASSP 2023 (https://arxiv.org/pdf/2301.00591) | k-means 2000 merged down to 50-200 | 2nd k-means on centroids (K-K), agglomerative (K-H), CR-weighted (needs a vocoder) | not tested; phoneme V-measure on TIMIT is HuBERT 42.49 / 45.48 / 46.64 / 43.32 at K = 50 / 100 / 200 / 2000 (CPC peak 48.45 at 100) |
| HuBERT, Hsu et al. 2021 (https://arxiv.org/pdf/2106.07447) | direct k-means 100 vs 500 | k-means | not tested; PNMI on BASE-it1 L6 features 0.575 (K=100) vs 0.686 (K=500), 100 h fit (Table IV) |
| Liu et al., Interspeech 2018 (GAN, background; https://arxiv.org/pdf/1804.00316) | 50-1000 segment clusters, oracle boundaries | k-means | with unrelated text, best "around 300"; "for 500 or more clusters (roughly 10 or more clusters for each phoneme) the mapping relationship became difficult to learn even though the cluster purity was higher" |
| Wang, Hasegawa-Johnson & Yoo, ACL 2023 (https://arxiv.org/pdf/2306.07926) | synthetic | theory: many-to-one binary map | fewer units need fewer distinct eigenvalues of the unit N-gram matrix; assumes purity 1 |
| DiscoPhon, Poli et al. 2026 (https://arxiv.org/pdf/2603.18612) | 256 (many-to-one) vs |P|+1 (1:1) | gold-label mappings | one-to-one PER 108-273%; "A possible direction would be to hierarchically group units from a larger vocabulary down to the target size". Untested |
| Copiale, Knight, Megyesi & Schaefer 2011 (https://aclanthology.org/W11-1202.pdf) | about 90 cipher symbols | context vectors (preceding/following symbol counts), cosine, agglomerative | homophones clustered (y, !, Y = I; circumflexed vowels = E). The automatic homophonic attack gave "nonsense"; the manual decipherment used the clusters |
| Nuhn, Schamper & Ney, EMNLP 2014 (https://aclanthology.org/D14-1184.pdf) | 54 to 299 cipher symbols | none (beam search on the full inventory) | solved at 7.55 and 4.18 observations per symbol; failed at 1.74 and 2.34. Our units have many thousands of tokens each, so this limit does not bind (mine) |

Summary:
- Homophonic decipherment names the fine inventory directly: Zodiac-408 maps 54 symbols to 26 letters with restarts
  (BK&K 2013) or beam search. It does not coarsen first.
- The only clustering-before-naming precedent (Copiale) groups by CONTEXT, not by acoustic/visual similarity, and its
  automatic naming failed.
- Petrov's evidence is for coarse-to-fine TRAINING (initialise fine from coarse), not for a coarse final inventory.

## Q3. An objective that is right on the right partition but prefers wrong names on wrong partitions

No paper diagnoses this exact asymmetry. The closest cases:
- Berg-Kirkpatrick & Klein, EMNLP 2013 (https://aclanthology.org/D13-1087.pdf). Homophonic, many-to-one, a fixed
  trigram LM backbone, EM on the emissions only.
  - Zodiac-408: 1 restart gives 18% accuracy, 100 give 90%.
  - The likelihood ranks basins correctly at coarse resolution. Large LL gains track accuracy.
  - Near the top it does not: the gold-init EM reaches LL -1467.4 at 92%, the best of 1M restarts -1466.5 at 89%.
  - The fix is search (restarts, selected by model score), with the objective unchanged. Synth-340 (63 symbols)
    still found better optima after tens of thousands of restarts.
- Jin 2016 and Dasgupta & Schulman 2007: the global optimum is right; the duplicated/starved configuration is a
  search trap; the fix is initialisation, over-seeding and pruning.
- Stolcke 1994: Baum-Welch lands on redundant or wasted states, and structure search (merging) fixes it.

(mine) AN-2's "S prefers found names over oracle 1:1 names on found partitions" is what this literature predicts
without any defect in S. On a many-to-one partition that is wrong, the best naming is not the 1:1 oracle naming. The
audit already notes that PREFERS largely re-expresses A20. The literature therefore supports a partition-search fix,
and it gives no reason to expect a rename step to work before the partition is repaired.

## Q4. Findings that contradict or constrain TP-B and the inventory

- Likelihood is a weak judge near the top: BK&K (above); Stolcke ("can be deceptive"); RSEM Fig. 3; the SMEM 2000
  toy's held-out drop. Our own record agrees: gold-init EM loses identity while S improves (E11), and an arm reached
  the S basin bar at PER 0.86. Acceptance on S alone can accept moves that do not help. This constrains; it does not
  contradict.
- Split-merge does not reliably beat restarts: SMEM below repeated EM on 6 of 7 sets (Zhao et al. 2012, Table 3).
- Count-allocation climbs under a fixed LM worked only on noiseless ciphers of up to 65 symbols (Dhavare 2013, reused
  from lit_phi_rename_moves_2026-09-24.md).
- Coarsening costs purity: HuBERT PNMI falls from 0.686 to 0.575 going from K=500 to K=100. Liu 2018's best count is
  about 300, not 100. A 100-class inventory may lower the basin's own ceiling (mine; our 500-unit purity is 0.644).
- No paper contradicts TP-B's premise (a search error with the right global optimum). Our E1 and the basin lead
  (+0.075 at lambda 1) are consistent with it.

## Established vs single-paper; disagreements

- Established (several groups): EM on mixtures and HMMs gets stuck with duplicated and starved components, and
  non-local moves or restarts escape when the objective's optimum is right (Ueda; Jain & Neal; Hughes; Zhao/Fränti;
  BK&K; Stolcke; Jin; Dasgupta & Schulman). Refit before accept is needed for splits (Ueda; Jain & Neal; Hughes;
  Zhao).
- Established: likelihood tracks accuracy between basins but not near the top (BK&K; Stolcke; Zhao; project E11).
- Disagreement: SMEM beats EM clearly in Ueda et al. (face data), but not repeated EM in Zhao et al. (Table 3).
- Disagreement: acceptance on Q (Ueda) against the likelihood (Minagawa, UNVERIFIED beyond the abstract; Zhao).
- Single paper: context-vector clustering groups homophones (Copiale); the count-sweep optimum near 300 (Liu 2018,
  GAN); hierarchical split beats direct estimation (Petrov 2006); fewer units are easier to identify (Wang 2023,
  theory).

## Design implications for TP-B (mine unless cited)

1. **Move type: a named fixed-M merge + split + name choice.** This is SMEM's structure with a naming step added.
   - Merge a duplicate pair (i, j) into i, which frees name j. Rank pairs by emission-row similarity (Jmerge-like,
     Ueda eq. 8) or by Hughes's rule (all terms except entropy improve).
   - Split a symbol k that covers two regions. SMEM's Jsplit (local KL) is degenerate here, because a free multinomial
     row matches its own posterior-weighted histogram at the M-step. Use a label-free CONTEXT criterion instead: split
     k's units into two groups by their preceding/following decoded-symbol context vectors (Copiale's criterion), or
     by the cut that maximises S (Hughes's births).
   - Give the new half a name from the unclaimed set (j plus the 8-13 already-starved names). Rank the names by how
     well the half's decoded trigram contexts fit each name under the LM. Try the top 2-3 through the refit.
   - Dropping the swap half is consistent with the literature: no paper renames a fixed partition successfully in a
     many-to-one setting.
2. **Refit.** Partial refit of the three affected rows (and their durations) with mass-preserving posteriors (Ueda
   eq. 7), then full EM. 2 sub-epochs is at the short end. RSEM and SMEM refit to convergence; only k-means random
   swap uses 2 iterations (Fränti, reused), and Jain & Neal ask for more scans when components have many
   observations. Check on the first moves that S has flattened by sub-epoch 2 before fixing 2.
3. **Acceptance.** Held-out S after refit, as a paired per-utterance delta on the 260 set, above the measured
   same-config S spread. No precedent: flag it as a design choice.
4. **Schedule.** Several candidates per round, not one. SMEM tries Cmax about 5 in rank order (accepted rank 1.8);
   RSEM needs t > M^2 random trials. With 7-12 duplicated types, one candidate every 4 sub-epochs gives at most 12
   tries in 48 sub-epochs. Evaluate the top 3-5 (pair, split, name) triples in parallel per round, packed on one
   node, and accept the best that clears the bar.
5. **Stop rule.** Stop when no candidate in a round clears the bar (SMEM, Hughes).
6. **Label-free success read, reported beside S and never used alone for acceptance:**
   - duplicate-pair count by emission-row similarity (label-free analogue of the 7-12 duplicated gold types);
   - vocabulary usage: decoded occupancy against the LM unigram and the number of unclaimed names (wav2vec-U's U(P),
     Baevski et al. 2021; the metric is label-free, the paper is GAN background);
   - LM log-prob per decoded phone: finals -3.99 to -3.81 against basin -3.16 to -3.12 (AN-4 P1b). This separates at
     phi level but NOT at key level, where found keys beat gold per phone (E10). Read it only on phis;
   - non-SIL rate: 6.9-7.6 Hz against 8.4-8.5 Hz (P1c);
   - cross-seed agreement of the key.
   - Generative PER is reported as the project requires, but it must not select.

## Design implications for the coarse-to-fine inventory (mine unless cited)

1. **Criterion.** Label-free and independent of the current phi.
   - Second k-means or agglomerative clustering on the 500 centroids (Sicherman K-K/K-H), or context-vector cosine
     agglomeration (Copiale).
   - A Petrov-style Delta from phi's own statistics would inherit the found partition's errors, so avoid it for the
     first coarsening.
   - CR-weighting needs a vocoder and is out of reach.
2. **Count.** "About 100" is at the low end of the published evidence: Liu 2018 is best near 300 (about 6 per
   phoneme), Sicherman's V-measure peaks at 100-200, and HuBERT's PNMI is lower at 100 than at 500.
   - Register a sweep {100, 200, 300}.
   - Report each count's oracle many-to-one purity as a label-using analysis only, because merging can only lower the
     ceiling.
3. **Use.** Coarse-to-fine as training, not as the final inventory (Petrov: hierarchical beats direct).
   - Fit phi on the classes, then expand each class row to its units with m(u|s) = m(c(u)|s) p(u|c(u)), then continue
     EM on the 500 units.
   - Jin 2016 and Dasgupta & Schulman support getting the allocation of symbols to groups right at the coarse level
     first. They give no guarantee for a named inventory.
4. **Success read.** The same label-free battery as TP-B at the coarse and the fine stage, plus the purity ceiling at
   each count (analysis only).
5. **Needs a ruling (not proposed).** A broad-class allocation of names (vowel/stop/fricative/nasal/SIL) would be
   the direct analogue of Jin's group counts. It uses phonetic knowledge beyond durations, so the user must decide
   whether it is admissible.

## What this changes

- It supports proposing TP-B and the inventory. It reshapes TP-B: named merge + split + name choice, a context-based
  split criterion, 3-5 candidates per round, and refit length checked, not assumed. It adds a label-free battery beside
  held-out S, and a count sweep with a purity ceiling for the inventory. It does not justify a claim that either will
  reach the basin; no published result covers a named inventory.

## UNVERIFIED

- Minagawa, Tagawa & Tanaka 2002: abstract only (MIT Press full text not reachable).
- Kempton & Moore 2014 (Speech Communication 56): not reached.
- Müller et al., ICASSP 2017 (https://ml.informatik.uni-freiburg.de/wp-content/uploads/papers/17_ICASSP.pdf): read.
  It uses crosslingual supervised articulatory features and k-means with K set to the known inventory size, so it
  gives nothing label-free.
- Hughes & Sudderth 2013 memoized birth/merge and Petrov, Haghighi & Klein 2008: not read.
- The Liu 2018 accuracies below 300 clusters are in a figure only; I did not read values off it.

## Sources

- SMEM NIPS 1998: https://proceedings.neurips.cc/paper/1998/file/253f7b5d921338af34da817c00f42753-Paper.pdf
- SMEM VLSI 2000: https://mlg.eng.cam.ac.uk/pub/pdf/UedNakGha00b.pdf
- Minagawa 2002 abstract: https://mlanthology.org/neco/2002/minagawa2002neco-smem/
- RSEM 2012: http://www.sipu.uef.fi/pub/RandomSwapEM-PRL2012.pdf
- Jain & Neal: https://glizen.com/radfordneal/ftp/mixsplit.ps
- Hughes 2015: https://proceedings.neurips.cc/paper/2015/file/2e65f2f2fdaf6c699b223c61b1b5ab89-Paper.pdf
- Petrov 2006: https://aclanthology.org/P06-1055.pdf
- Stolcke & Omohundro 1994: https://arxiv.org/pdf/cmp-lg/9405017
- Jin 2016: https://arxiv.org/pdf/1609.00978
- Dasgupta & Schulman 2007: https://www.jmlr.org/papers/volume8/dasgupta07a/dasgupta07a.pdf
- BK&K 2013: https://aclanthology.org/D13-1087.pdf
- Sicherman & Adi 2023: https://arxiv.org/pdf/2301.00591
- HuBERT: https://arxiv.org/pdf/2106.07447
- Liu 2018: https://arxiv.org/pdf/1804.00316
- Wang 2023: https://arxiv.org/pdf/2306.07926
- DiscoPhon: https://arxiv.org/pdf/2603.18612
- Copiale: https://aclanthology.org/W11-1202.pdf
- Nuhn 2014: https://aclanthology.org/D14-1184.pdf
- Lee & Glass 2012: https://aclanthology.org/P12-1005.pdf
- Ondel 2016: https://www.fit.vut.cz/research/group/speech/public/publi/2016/ondel_sltu2016_17-8037.pdf
- wav2vec-U: https://arxiv.org/pdf/2105.11084
