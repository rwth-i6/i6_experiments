# Unsupervised cluster->phoneme mapping: objectives, framed by one master objective

## Setting
- Audio: sequences of discrete cluster ids x_1^T, x_t in {1..512} (oracle GMM segmentation, so one
  audio token ~ one phoneme). Text: phoneme sequences c_1^T, c_t in {1..40}. The two corpora are
  UNPAIRED and cover disjoint utterances. Empirical distributions: p_A(.) on audio, p(.) on text.
- Model: a memoryless table q(c|x) (hard: q(c|x)=1[c=f(x)]; soft: q_theta(c|x)=softmax_c(theta_{x,c}/tau)),
  so q(c_1^T | x_1^T) = prod_t q(c_t|x_t). Oracle table: frame acc 0.740 = 26.0% PER; unigram chance 0.103.
- The distribution over phoneme sequences that the model INDUCES from audio:
      q(c_1^T) = sum_{x_1^T} p_A(x_1^T) prod_t q(c_t|x_t)

## Master objective
      max_q  sum_{c_1^T} p(c_1^T) log q(c_1^T)      ( = -KL(p || q) - H(p), i.e. min KL(p || q) )
Direction matters: p (text truth) in front, the induced q inside the log. It is mode-covering: every
phoneme sequence real text produces must be producible from audio at roughly the right rate.
Intractable as written (sum over all sequences; q(c_1^T) is itself a sum over audio sequences).

## What we actually optimize: n-gram marginal projections of the master objective
Replace p and q by their joint n-gram marginals p_n, q_n (all bigrams etc. are JOINT, e.g. p_2(c,c')):
      q_1(c)         = sum_x        p_A(x)        q(c|x)
      q_2(c,c')      = sum_{x,x'}   p_A(x,x')     q(c|x) q(c'|x')
      q_n(c_1..c_n)  = sum_{x_1..x_n} p_A(x_1..x_n) prod_i q(c_i|x_i)         (n = 3, 4)
      L(q) = sum_n s_n KL(p_n || q~_n),   s_1 = 5, s_2 = 1, s_3 = 10, s_4 = 10
      q~_3 = (1-b3) q_3 + b3 q_2(c,c') q_1(c''),                 b3 = 0.05
      q~_4 = (1-b4) q_4 + b4 q_3(c1,c2,c3) q_3(c2,c3,c4)/q_2(c2,c3), b4 = 0.2
Back-off is needed because 2.2% (trigram) / 14.2% (4-gram) of the text n-gram mass is unreachable even
from oracle-mapped audio (6000 audio utts) -> forward KL of any hard map would be infinite.
By the KL chain rule KL(p_n||q_n) = KL(p_{n-1}||q_{n-1}) + E_p[KL of the n-th conditional], so the s_n
effectively weight conditional orders. All statistics are precomputed once over the corpora; the
optimization is full-batch (no per-sequence updates).

Search: soft q_theta, Adam on theta (512x40 logits), temperature tau annealed 1.0 -> 0.02 over 3000 steps,
result hardened by argmax; report hardened loss/accuracy only (a soft q can match p_n as a mixture no
hard table realizes). Init: theta = 0 (uniform barycenter) + 1e-3 noise; several seeds; keep the run with
the LOWEST HARDENED L -- this ranking is label-free and has ordered every seed correctly so far.

## Results (frame accuracy of the hardened table; PER = 1 - acc)
- Optimum of L when started AT the oracle:  n<=2: 0.64 | n<=3: 0.707 | n<=4: 0.727  (ceiling 0.740)
- Basin: start at acc 0.17 -> 0.67 with n<=3 (n<=2 stalled at 0.36)
- Label-free (uniform init): n<=2 chance | n<=3 0.31-0.47 | n<=4 0.656 = 34.4% PER (1 of 4 seeds; the
  other seeds 0.12/0.12/0.31 have higher hardened loss, so loss-selection picks the good one)
- Beats a SUPERVISED linear map on per-symbol encoder embeddings (36.5% PER).

## Rejected alternatives and why (all in the same notation)
1. Reverse direction ("LM score"/perplexity of a text n-gram model on the induced output):
      max_f sum_{c,c'} q_f(c,c') log p(c,c')  =  -KL(q||p) - H(q)
   Frees the entropy of the induced output -> rewards collapse (argmin is a point mass on the most
   frequent bigram); strictly prefers lossy wrong maps over the oracle; a climb from the oracle walks
   away. Same failure as shallow LM fusion at decode time.
2. Master direction but hard table f + coordinate hill-climb (each of 512 entries re-set to its argmin
   over 40 phonemes, sweeps until fixed point): ranks the truth correctly and is monotone in accuracy,
   but never INCREASES accuracy from a partial start -- search, not objective, is the bottleneck.
3. Word-by-word translation (vecmap/Procrustes on per-symbol embeddings e_A(x), e_T(c)):
      max_W sum_x max_c (e_A(x) W) . e_T(c),  W'W = I
   No p in the objective; time enters only via the embeddings. Isometry premise verified, but the search
   finds W scoring above the oracle, and orthogonality (the only anti-collapse device) forbids the
   many-to-one 512->40 contraction. Self-learning at chance.

## Open
- Larger n / an actual sequence model for q (the master objective proper) vs. undersampled n-gram
  statistics at 6000 audio utts (more audio is the cheap lever).
- Success rate of the uniform start (1 of 4 seeds), and the remaining 0.07 to the optimum (annealing).
- Everything above is on the oracle segmentation; on real k-means clusters the alignment problem
  (many frames per phoneme) returns and q(c_1^T|x_1^T) is no longer a product over aligned positions.
