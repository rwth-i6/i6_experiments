# The optimization criteria tried for the unsupervised cluster->phoneme map, variant by variant

Companion to OBJECTIVES.md (which frames them under the master objective sum_{c_1^T} p(c_1^T) log q(c_1^T)).

## Notation
- Audio corpus A: utterances of cluster ids x_1..x_T, x_t in {1..512}. Text corpus T: phoneme
  sequences c_1..c_L, c_t in {1..40}. Disjoint utterances, no pairing.
- Empirical distributions, estimated ONCE over the whole corpus and then fixed (joint, not conditional):
      p_A(x)        = (1/N) sum_u sum_t 1[x_t = x]
      p_A(x,x')     = (1/N') sum_u sum_t 1[x_t = x, x_{t+1} = x']
      p_A(x,x',x''), p_A(x1,x2,x3,x4)   analogously (stored sparse over the observed tuples)
      p(c), p(c,c'), p(c,c',c''), p(c1,c2,c3,c4)   the same on T (dense)
- Model = table q(c|x), the probability that cluster x is phoneme c.
      hard:  q_f(c|x)     = 1[c = f(x)],  f: {1..512} -> {1..40}
      soft:  q_theta(c|x) = exp(theta_{x,c}/tau) / sum_{c'} exp(theta_{x,c'}/tau),  theta in R^{512x40}
- Induced (predicted) text statistics, always written with the parameter as subscript:
      q_f(c)        = sum_x     p_A(x)     q_f(c|x)        = sum_{x: f(x)=c} p_A(x)
      q_f(c,c')     = sum_{x,x'} p_A(x,x') q_f(c|x) q_f(c'|x')
                    = sum_{x: f(x)=c} sum_{x': f(x')=c'} p_A(x,x')
      q_theta(c)    = sum_x     p_A(x)     q_theta(c|x)
      q_theta(c,c') = sum_{x,x'} p_A(x,x') q_theta(c|x) q_theta(c'|x')
      q_theta(c,c',c'')   = sum_{x,x',x''}   p_A(x,x',x'')   q_theta(c|x) q_theta(c'|x') q_theta(c''|x'')
      q_theta(c1,c2,c3,c4)= sum_{x1,x2,x3,x4} p_A(x1,x2,x3,x4) prod_i q_theta(c_i|x_i)
- Chain rule (why a joint-bigram KL already contains the unigram one):
      sum_{c,c'} p(c,c') log[p(c,c')/q(c,c')]
        = sum_c p(c) log[p(c)/q(c)] + sum_c p(c) sum_{c'} p(c'|c) log[p(c'|c)/q(c'|c)],   q(c'|c) = q(c,c')/q(c)
- Reported accuracy = frame accuracy of the hardened table against the oracle table (ceiling 0.740,
  = 26.0% PER; unigram chance 0.103). PER = 1 - acc.

--------------------------------------------------------------------------------------------------

## 1. Word-by-word translation (vecmap / Procrustes) -- no time, no p
Per-symbol embeddings e_A(x), e_T(c) in R^d, unit-normalized. Two embedding sources were tried:
  (i)  co-occurrence: top-d singular vectors of the PPMI matrix
          PPMI_A(x,x') = max(0, log p_A^{(w)}(x,x') / (p_A(x) p_A(x')))   (window w), same for text;
  (ii) averaged encoder states: e_A(x) = mean of the shared-encoder state over all frames with symbol x.
Dictionary D(x,c) in {0,1} plays the role of a hard q(c|x).

Supervised form (given D):
      min_W  sum_x sum_c D(x,c) ||e_A(x) W - e_T(c)||^2      s.t. W'W = I
    = max_W  sum_x sum_c D(x,c) (e_A(x) W) . e_T(c)          -> Procrustes: W = U V',  U S V' = svd(E_A' D E_T)

Unsupervised form (D unknown), the objective the self-learning loop ascends:
      max_W  sum_x  max_c  (e_A(x) W) . e_T(c)               s.t. W'W = I
Updates (alternation to a fixed point):
      D(x,c) <- 1[c = argmax_{c'} (e_A(x) W) . e_T(c')]      (nearest neighbour, all x at once)
      W      <- Procrustes(D)                                (closed form, global)
Bijective variant: k-means the e_A(x) into 40 groups g with means e_A(g), search a permutation pi:
      max_{W,pi} sum_g (e_A(g) W) . e_T(pi(g)),  pi via Hungarian, W via Procrustes, alternated.

Outcome: isometry premise holds (Mantel rho >> permutation null), embeddings carry the phoneme (probe
0.60 at pca-64), but the search finds W with a higher objective than the oracle's, and orthogonality
(the only anti-collapse device: without it every cosine can be made 1.0) forbids the 512->40
contraction. Self-learning acc 0.02-0.05 (chance 0.103); 100 anchors -> 0.341. Bijective variant 0.228.

--------------------------------------------------------------------------------------------------

## 2a. LM score of the induced output, hard table
      min_f   - sum_{c,c'} q_f(c,c') log p(c,c')   +   lam * sum_c p(c) log[ p(c) / q_f(c) ]        lam = 5
Decomposition of the bigram part:
      - sum q_f log p = sum_{c,c'} q_f(c,c') log[q_f(c,c')/p(c,c')] - sum_{c,c'} q_f(c,c') log q_f(c,c')
                      = KL(q_f || p) + H(q_f)
i.e. reverse KL plus a reward for lowering the entropy of the induced output. Its argmin over all q is a
point mass on the most frequent bigram. (The unigram term was the forward KL in this variant too.)
Updates: coordinate hill-climb, as in 2b.
Outcome: oracle 3.168 @ acc 0.740, best restart 2.980 @ 0.133, climb started AT the oracle walks to
2.892 @ 0.480 (229/512 entries moved). Merging phonemes lowers the score monotonically. Rejected.

## 2b. Distribution matching (forward KL), hard table
      min_f   sum_{c,c'} p(c,c') log[ p(c,c') / q_f(c,c') ]   +   lam * sum_c p(c) log[ p(c) / q_f(c) ]   lam = 5
with q_f as in the notation block (row/column sums of p_A(x,x') over the preimages of f).
Updates (Search.climb): one sweep visits all 512 clusters x in random order; for each x remove its
row/column/unigram mass from q_f, re-add it at each candidate v in {1..40}, evaluate the FULL criterion,
set f(x) = argmin_v (strict improvement only); repeat sweeps until none changes (<= 25).
Outcome: ranks the truth (climb from oracle settles at 0.073 vs ~0.21 for every random restart) and is
monotone in accuracy under corruption, but never increases accuracy from a partial start.

--------------------------------------------------------------------------------------------------

## 3. Distribution matching, soft table, gradient search
      min_theta   sum_{c,c'} p(c,c') log[ p(c,c') / q_theta(c,c') ]  +  lam * sum_c p(c) log[ p(c) / q_theta(c) ]   lam = 5
Optional (tested, dropped): + eps * sum_x p_A(x) sum_c q_theta(c|x) log q_theta(c|x).
Updates (full-batch, no per-sequence steps):
      for step in 0..S-1  (S = 3000):
          tau   = geomspace(1.0, 0.02)[step]
          M     = softmax(theta / tau)                # whole 512x40 table
          L     = criterion above, computed from the fixed count tables and M
          theta <- Adam(theta, grad_theta L)
          (anchored rows, if any, reset to fixed logits)
      report f(x) = argmax_c q_theta(c|x) and the criterion of 2b evaluated at that f ("hardened loss").
Init: theta ~ N(0, 2^2) (random), theta = 0 + 1e-3 noise (uniform), oracle, corrupted oracle, k-means.
Outcome: climbs uphill in accuracy from a partial start (corrupt 0.8: 0.166 -> 0.347); anchors give
lift (+0.09..+0.16 over their own contribution); cold start at chance (0.085); optimum displaced:
started at the oracle, loss 0.2125 -> ~0.14 while acc 0.740 -> 0.64.

--------------------------------------------------------------------------------------------------

## 4a. + trigram term
      q~_theta(c,c',c'') = (1 - b3) q_theta(c,c',c'') + b3 * q_theta(c,c') q_theta(c'')          b3 = 0.05

      min_theta   s3  * sum_{c,c',c''} p(c,c',c'') log[ p(c,c',c'') / q~_theta(c,c',c'') ]
                +       sum_{c,c'}     p(c,c')     log[ p(c,c')     / q_theta(c,c')     ]
                + lam * sum_c          p(c)        log[ p(c)        / q_theta(c)        ]          s3 in {1,3,10,30}, lam = 5
Back-off needed: 2.2% of the text trigram mass lies on cells even the oracle-mapped audio never produces
(6000 audio utts), so the forward KL of every hard table would be infinite. Updates as in 3;
q_theta(c,c',c'') is a sparse matmul over the 284,525 observed audio triples, then two small einsums.
Outcome: oracle-start optimum 0.707 (s3=1) / 0.717 (s3=10); basin: start 0.17 -> 0.67 (was 0.36);
anchors 25: 0.203 -> 0.302, 100: 0.405 -> 0.525; UNIFORM init 0.31-0.47 depending on s3/seed/steps
(random inits still at chance); hardened loss orders the seeds by accuracy.

## 4b. + 4-gram term
      q~_theta(c1,c2,c3,c4) = (1 - b4) q_theta(c1,c2,c3,c4) + b4 * q_theta(c1,c2,c3) q_theta(c2,c3,c4) / q_theta(c2,c3)   b4 = 0.2

      min_theta   s4  * sum_{c1..c4}   p(c1,c2,c3,c4) log[ p(c1,c2,c3,c4) / q~_theta(c1,c2,c3,c4) ]
                + s3  * sum_{c,c',c''} p(c,c',c'')    log[ p(c,c',c'')    / q~_theta(c,c',c'')    ]
                +       sum_{c,c'}     p(c,c')        log[ p(c,c')        / q_theta(c,c')        ]
                + lam * sum_c          p(c)           log[ p(c)           / q_theta(c)           ]      s3 = s4 = 10, lam = 5
14.2% of the text 4-gram mass unreachable from the oracle-mapped audio -> conditional back-off, b4 from
the oracle's KL4 curve (mild leakage). Updates as in 3; ~2 s/step, 6 GB (dense [512^2, 40, 40] intermediate).
Init theta = 0 (uniform), seeds 1-4; selection = lowest hardened criterion.
Outcome: oracle-start 0.727 (ceiling 0.740, 27.4% PER). Uniform seeds: 0.656 / 0.116 / 0.115 / 0.308,
hardened losses 13.7 / 33.2 / 30.0 / 26.0 -> loss-selection picks 0.656 = 34.4% PER without labels
(beats the supervised linear map on encoder embeddings, 36.5%). Trajectories flat after ~1000 steps.
