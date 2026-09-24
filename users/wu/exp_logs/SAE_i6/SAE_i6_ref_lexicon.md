# SAE_i6 reference: prior strength and the lexicon inside the marginalised lattice (k2 lexlat)

Part A covers prior strength: the prior-gap diagnostic (Step 0), the neural phone LM (Step 0b), the pre-launch falsifiers, and the score-function and soft (straight-through) training arms. Part B covers the lexicon inside the objective: the DP form and its cost probes E-1 / E0 / E1; the k2 second-graph form ("amendment 9"); the twelve-arm round and gate G4a.9; diagnoses D0-D18 and the training-length extension E60. Everything runs on the blank-free bed (recognizer theta and semi-Markov reverse model phi, trained jointly through an exact-marginal blank-free trigram lattice). `SAE_i6_ref_blankfree.md` defines the bed, the ctrl_20 / ctrl_50 controls and the N = 20 schedule. `SAE_i6_ref_objective.md` derives the loss.

**Conventions.** PER is plain greedy PER on dev-other: 2864 utterances, 177,275 gold tokens. A paired delta is candidate minus baseline, per utterance, with a 95 % speaker-clustered bootstrap (2000 resamples, 33 speakers). Negative means the candidate is better. "epN" is kept sub-epoch N. The cold-bed chance band is PER 0.83-0.91. Label-using reads are disclosed diagnostics that enter nothing.

**Standing constraints.** No label enters training or checkpoint selection. Main-line methods are pure unsupervised and GAN-free; supervised and GAN-lineage inits are analysis only. No gate bar is moved after its result.

---

# Part A: Prior strength

## A1. Question and premise

The budget control settles into a confident frame-level acoustic code, the "private code". It satisfies the live phone trigram nearly as well as phones do: prior per token is −3.98 on the dev-other decode against −3.20 for gold. A 40-symbol trigram accepts a large family of codes with English-like trigram statistics. A larger prior weight beta would make the code more trigram-typical, not more phone-like. The lever is a prior that accepts a smaller family: a higher n-gram order, or a lexicon under which phone strings must decompose into dictionary words. The phase asks two questions: Which prior widens the gold-minus-private gap? Can it enter training so that a cold arm leaves the content-free band? Literature anchor: the reproduced gains in unsupervised phone recognition come from prior strength and lexicalisation with a context-free emission. Klejch et al. 2022 went from a bigram to a 5-gram to a word trigram with a lexicon, and found the prior the most important factor. (src: SAE_4A_prior.md § Objective)

## A2. Prior resources and the SIL convention

**Live prior.** An interpolated Witten-Bell (WB) phone trigram over 39 ARPAbet phones plus SIL, with held-out perplexity 9.561 on its own 10,000 held lines. The older value, 9.469, used a different held set. Every prior is fitted on the same uniform-sample window of the LibriSpeech LM text (1,000,000 counted lines, 10,000 held), never on the alphabetical head. With the trigram inside the DP, the residual to a 4-gram is about 0.1 nats per phone. 4-gram rescoring of a bigram DP had been rejected earlier as an importance-sampling (IS) estimator, because its effective sample size was near zero.

**SIL convention.** The prior text is phonemised with sil_prob 0.5 plus sentence-edge SIL.

| quantity | value |
| --- | --- |
| gold SIL frames before / after the bed's rVAD | 20.3 % of 919,980 / 7.9 % of 781,130 |
| gold SIL runs on retained frames | 5.7 % of 186,083 runs; median 100 ms, 28 % are 1-2 frames |
| SIL tokens: ctrl_50 ep10 decode / prior text | 4.4 % / 13.8 % |

rVAD removes 60 % of gold silence, but SIL is still needed. The prior expects SIL 2.4x as often as gold shows it. wav2vec-U 2.0 also combines rVAD with sil_prob 0.5 (arXiv 2204.02492v2 §4.1), so this is the reference setting. wav2vec-U 1.0 swept the value and chose 0.25. A 0.25 text would match gold's SIL share; it would be a bed change and was never run. (src: SAE_4A_prior.md § Bed and constants)

## A3. Step 0: the prior-gap diagnostic (CPU, label-using)

**Design (pre-registered).** Three string sets are scored per dev-other utterance: (a) the MFA gold phones; (b) the private code: the ctrl_50 ep10 collapsed greedy decode; (c) nulls: each string's tokens shuffled within the utterance. All priors are fitted on the same window: WB phone n-grams of orders 1-3. KenLM modified Kneser-Ney (MKN) models at orders 4 and 6. The 8-gram was dropped because the installed KenLM caps at order 6. The exact lexicalised prior: a KenLM word trigram scores the best segmentation of the string into words from a 151,731-word pronunciation trie, with SIL as an optional word boundary. The trie takes each word's first bliss pronunciation and uses Sequitur G2P for missing words; this is the same chain that phonemises the prior text. The lexicalised prior is reported in two conventions: **STRICT** has no `<unk>` path. Its gap is computed on the subset where both strings segment. **ESCAPE** lets an escape word cover any non-SIL span. The escape costs the word LM's `<unk>` transition once per span, plus, per phone, the order-1 WB phone log-probability plus log 0.5. ESCAPE is the decision row, because a training reward must be finite on every string. The primary pairing is like-for-like: gold against the SIL-dropped decode. The spread is |gap on even utterances − gap on odd utterances|.

**Rule.** A prior discriminates more than the trigram if its gap exceeds the trigram's by more than the larger of the two spreads. Decisions: if the 4-gram discriminates more and its per-utterance IS log-weight sd is below 3 nats, use a 4-gram IS correction; if only the lexicon discriminates more, use a lexicon score-function term; if neither does, the phase closes.

**Result** (2864 utterances, nats per token, audited):

| prior | gold | private | gap like-for-like | spread | gap SIL-kept | IS log-weight sd per utt (gold / private) |
|---|---|---|---|---|---|---|
| unigram WB | −3.50 | −3.67 | 0.16 | 0.002 | 0.07 | – |
| bigram WB | −3.25 | −3.78 | 0.52 | 0.005 | −0.01 | – |
| trigram WB (live) | −3.20 | −4.63 | 1.39 | 0.017 | 0.49 | – |
| 4-gram MKN | −2.79 | −4.50 | 1.67 | 0.008 | 1.18 | 12.8 / 17.9 |
| 6-gram MKN | −2.57 | −4.26 | 1.62 | 0.014 | 1.25 | 22.8 / 20.0 |
| lexicon STRICT (n = 609) | −2.15 | −4.66 | 2.51 (trigram on the same subset: 1.09) | 0.060 | 2.23 | 31.0 / 20.9 |
| lexicon ESCAPE | −2.19 | −4.46 | 2.32 | 0.016 | 2.18 | 31.7 / 24.3 |

Strict-segmentable fractions: gold 0.876, private 0.237, gold null 0.024, private null 0.011. The audit added three findings: An MKN trigram gives a gap of 1.19. Smoothing alone therefore costs 0.20, and going to order 4 within one estimator adds +0.48. The 6-gram minus 4-gram gap is −0.044 (SE 0.005). In the escape row, the fixed escape prices carry 0.16 of the gap and lexical routing carries 2.17.

**Readings (audited).** The lexicon identifies the private code: it adds 0.93 over the trigram, against 0.27 for the 4-gram. Phone order is no proxy for the lexicon: the 6-gram discriminates less than the 4-gram. IS is closed: the per-utterance log-weight sd is 13-18 nats against the 3-nat ceiling. Only a score-function estimator can carry a lexicalised prior. As printed, the decision table fires no row: it did not anticipate "the 4-gram discriminates more but cannot be estimated". The amended reading, made after the numbers and audited as the table's own logic, is the lexicon score-function term. A word-bigram ESCAPE row gives a gap of 2.287 against 2.321 for the word trigram, so word-LM order above bigram buys only 0.03 nats per token. (src: SAE_4A_prior.md § Design / Step 0; § Results / Step 0; § Pre-launch falsifier (i))

## A4. Step 0b: can a neural phone LM carry the lexicon?

A GPU-batchable scorer is needed inside a training step. The candidate is a causal transformer phone LM trained on the same window, with the same SIL, sentence-start context, no end-of-sentence term and 512 positions.

**Rule (pre-registered).** A candidate is **adequate** if its gap is ≥ 2.01 (the trigram's 1.39 plus two thirds of the ESCAPE excess) and its strict-subset gap is within 0.30 of STRICT's 2.51. Between the 4-gram's gap and that bar it is a **partial proxy**; at or below the 4-gram's gap it is **not the scorer**. Held perplexity is the minimum over epochs on the benchmark's own held lines, a bias under 1 %.

| instance | params | epochs run / selected | held ppl | gap like-for-like | strict subset (vs 2.51) | verdict |
|---|---|---|---|---|---|---|
| first pass, 4L / w256, 3 ep | 3.3 M | 3 / 3 | 5.774 | 1.71 | 1.52 | partial proxy |
| (a) 4L / w256, 30-ep cap | 3.3 M | 11 / 10 (no-improvement abort) | 5.091 | 1.861 | 1.646 | partial proxy |
| (b) 6L / w384, 30-ep cap | 10.9 M | 11 / 10 (no-improvement abort) | 4.603 | 1.849 | 1.663 | partial proxy |
| (c) 10.09 M lines, 5 ep | 25.5 M | 5 / 5 (still falling) | 3.963 | 1.809 | 1.693 | partial proxy |

Benchmark perplexities on the same held lines are 9.557 (trigram WB), 7.271 (4-gram MKN) and 5.424 (6-gram MKN); the 4-gram gap is 1.666. Instance (c) changes text, capacity and epochs at once. Its architecture is recorded inconsistently: "6 layers / width 768-class" at registration and "8L / w512" in the result table, both at 25.5 M.

**Conclusion.** A stronger phone LM scores gold better but the decode better still: ten times the text lowered the gap from 1.849 to 1.809. The decode's errors are phonotactically fluent, and any phone-sequence model rewards that fluency; all gaps fall between 1.62 and 1.86. Step 0b closed on its third clause: the exact lexicon has to be made batchable. A later user ruling approved instance (a) as the scorer anyway (A7). (src: SAE_4A_prior.md § Step 0b; § Results / Step 0b)

## A5. Training-arm design (sf, soft) and gate G4a.7

The strong scorer cannot sit inside the DP, so it enters as a correction term outside it.

**Score-function arm (`sf`).** Per utterance, draw G = 8 strings by forward-filtering backward-sampling (FFBS) from the blank-free lattice posterior at the schedule's tau. The reward is summed over the string. Advantages are centred over G, with no std division. By the Fisher identity, the gradient is the advantage times the sampled path's own log weight (frame log-q gathers plus segment scores). log Z cancels, so no differentiable fixed-string DP is needed.

**Soft arm (`soft`, straight-through).** The forward pass runs on the hard max-plus string. The gradient runs through y_soft = onehot + (p − stopgrad(p)) into the scorer's embedding. Here p is the lattice's conditional posterior over 40 symbols per segment, with the neighbouring Viterbi symbols held fixed. The segmentation gets no gradient.

**Design-review amendments** (all applied before any job): **A1, reward.** log p_strong − log p_3 pays shuffled strings more than their originals, because the trigram punishes non-lexical order hardest. The reward becomes r(y) = log p_strong(y) − log p_uni(y), using the window unigram as an order-insensitive baseline: neural − unigram per token: gold +0.95, private −0.67, nulls −2.2; lexicon − unigram: +1.31, −0.79, −1.0. **A2, soft artefact ceiling.** The soft arm's reads are void if scorer(soft) − scorer(hard) exceeds 0.3 nats per token. **A3, UNINFORMATIVE.** It is read from two quantities: the label-using fraction of utterances with r(gold) > max_g r(y_g), and a probe-calibrated within-group std. A fixed std floor would never fire, because ctrl_50's sampler already draws 494-510 distinct strings of 512. **A4, SIL and length.** Both scorers see SIL-dropped strings; with SIL kept, the term pushes SIL out unopposed. Strings longer than 512 tokens are masked and counted. **A5, normalisation and lam.** The term is normalised per retained frame, as l_tau = mean_b(−log Z_b / retained_b) is, so it cannot pay for length. lam is set by probe so the term's gradient norm is 0.1x that of l_tau (preferred) or 0.3x (ceiling).

**Gate G4a.7** (per arm, dev-other, final sub-epoch, never best-PER). PASS requires PER < 0.50, emitted rate in [5.80, 14.49] per second, and a speaker-matched derangement gap > 0. Paired reads are against the control at the matched sub-epoch. A FAIL with the monitor engaged ends funding for the outside-the-DP correction. A PASS needs a fresh-context audit and a second seed. (src: SAE_4A_prior.md § Training arm; § Gate)

## A6. Pre-launch falsifiers

**(i) CPU neighbourhood probe.** Applied to the ctrl_50 ep4 and ep10 decodes, with K = 8 single-token edits per utterance (second-choice substitutions and deletions). Rewards are in nats per utterance. "Gold above max" is the fraction of utterances where r(gold) exceeds r of the decode and of all its edits.

| reward | std over neighbourhood ep4 / ep10 | corr with trigram delta | gold above max |
|---|---|---|---|
| neural − unigram (A1) | 4.00 / 4.23 | +0.65 / +0.68 | 0.9993 / 0.9920 |
| lexicon ESCAPE − unigram (A1) | 0.50 / 2.41 | +0.42 / +0.36 | 0.9997 / 0.9972 |
| original neural − trigram | 5.03 / 5.01 | −0.73 / −0.75 | 0.0552 / 0.6390 |

The sign trap is real, and A1 removes it. Locally, the neural reward restates the trigram. The lexicon reward is flat at ep4. Only 1.5 % of edits create a strict word, and decodes segment strictly in 5.2 % of utterances at ep4 (23.7 % at ep10).

**(ii) FFBS draws** on 300 dev-other utterances: G = 8 exact draws at the schedule's tau, reward first-pass 3.3 M neural − unigram. Audited CONFIRMED.

| ckpt (tau) | distinct of 8 | r per token gold / greedy / sample | within-group std | r(gold) > max_g |
|---|---|---|---|---|
| ep1 (8) | 8.00 | +0.84 / −4.12 / −1.42 | 14.9 | 1.0000 |
| ep4 (5.04) | 8.00 | +0.84 / −2.76 / −1.09 | 12.0 | 1.0000 |
| ep10 (2) | 7.36 | +0.84 / −0.72 / −0.48 | 5.1 | 0.9700 |

The rule: sf is not funded if r(gold) > max_g in more than 95 % of utterances at every checkpoint. It fired. The draws are diverse, but the best draw sits 45-75 nats per utterance below gold, and a gradient that never sees strings near the lexical region cannot supply the constraint. The next arm is therefore the lexicon inside the lattice (Part B). (src: SAE_4A_prior.md § Pre-launch falsifier (i), (ii))

## A7. The reopened soft / sf pack (N = 20)

**User rulings.** The 3.3 M instance (a) is approved as the scorer despite the Step 0b bar (epoch 10, held perplexity 5.0906, 3.31 M parameters). sf is also funded, as a hedge. Falsifier (ii) stays as a prediction, not a block. The ported package has no preset for these arms.

**Spec.** Kept checkpoints 1 / 4 / 10 / 20. The term is on from sub-epoch 1 with constant lam. The reward is A1, SIL is dropped, and normalisation follows A5. Controls: the frozen ctrl_20 and ctrl_20_s1. Arms: `soft_20` (0.1x, seed 0), `soft_20_s1` (0.1x, seed 1), `sf_20` (G = 8, 0.1x), and the null `softshuf_20`. In the null, both the scorer and the unigram see the 39 non-SIL symbols through a fixed derangement, so r_null(y) = r(σ⁻¹y). The reward's statistics are kept; phone identities are destroyed. The margin follows G4a.9: M = max(|seed band|, |null delta|, 0.010). The seed band ctrl_20 − ctrl_20_s1 at ep20 is −0.001 [−0.004, +0.001].

**lam probes** (ctrl_20 ep4, 3 batches of 88,000 padded frames, max_seqs 128):

| probe | grad-norm ratio to l_tau (per batch; median) | lam at 0.1x | other |
|---|---|---|---|
| soft | 0.298, 0.306, 0.335; 0.306148 | 0.326639 | artefact gap 0.0138 (ceiling 0.3); 0.3x = 0.979918 |
| sf (tau 2) | 2.368, 2.186, 2.203; 2.20294 | 0.0453938 | peak 23.6 GiB; 14-19 s per batch (two backwards); 8.0 distinct; within-group std 9.68 nats/utt |

Pitfall: a gather over an expanded view made the backward allocate a [B, U_max, K·d_cap·(S+1)] float64 tensor of 224.61 GiB. After a value-identical fix the backward peak is 1.14x the table.

**Cost** (one GH200 96 GB GPU per arm), seconds per sub-epoch: sf_20 717, soft_20 849, soft_20_s1 852, softshuf_20 849. That is 1.19-1.42x the bed's 601 s.

**Interim read at ep10** (not the gate). Gaps are gold − decode, nats per token.

| arm | gap trigram / 6-gram / lexicon-escape | paired PER vs control ep1 / ep4 / ep10 | dev reward_mean |
|---|---|---|---|
| sf_20 | 1.147 / 1.491 / 2.237 | +0.0075 / +0.0351 / +0.0306 | −0.515 |
| soft_20 | 1.295 / 1.600 / 2.313 | +0.0206 / −0.0004 / +0.0101 | −0.590 |
| soft_20_s1 | 1.055 / 1.537 / 2.294 | +0.0209 / +0.0099 / +0.0099 | −0.498 |
| softshuf_20 | 1.066 / 1.541 / 2.269 | +0.0710 / +0.0007 / +0.0121 | −2.025 |

No arm beats its control at any kept epoch. The scorer rates the real arms about 1.5 nats above the null, but the arms do not convert this into a smaller gap. JSD4 to the text at ep10: ctrl_20 0.739, soft_20 0.772, sf_20 0.789, softshuf_20 0.746, gold 0.230. soft_20 and sf_20 end farther from the text, and are less likely under its trigram (−3.81 / −3.71 against −3.52 nats per phone). Audited examples: every arm decodes fluent, English-shaped phone nonsense, with NMI(gold, hypothesis) 0.066-0.093. sf_20 alone is worse than the control band: PER 0.899, 74 % substitutions. Straight-through changes nothing beyond seed spread, and the score-function estimator degrades the code.

**Status: G4a.7 was never read at ep20; the record ends at this interim read.** (src: SAE_4A_prior.md § Training arm, reopened by user ruling; § Soft-arm lam probe; § Sampled-arm lam probe; § Soft pack, ep1 read; § Soft pack, interim read)

---

# Part B: The lexicon inside the marginalised lattice

## B1. Question, claim scope, constraints

**Question.** Does a pronunciation lexicon with a word LM, placed inside the marginalised objective so that the forward-backward sums over word-constrained phone strings, move a cold arm out of the content-free band at the same budget? This removes the sampling problem of A6.

**Claim scope (pre-registered).** A POSITIVE result is a paired PER gain beyond both the seed band and a shuffled-pronunciation null. A NEGATIVE result licenses "not at this bed, pruning budget and on-set", never "cannot work".

**Constraints and rulings.** One delta per arm against ctrl_20. The lexicon and the word LM are fixed; nothing on the lexicon side is learned, so the literature's alpha = 0.9 re-smoothing is not carried. The trie is used verbatim (bliss lexicon plus G2P). Mixing is ADD: beta log P3 + lam_lex × lexical increment, with lam_lex = 1 and no sweep. The shuffled null serves as the prior-weight control. Silence is a word boundary (Klejch). Main extrapolation risk: Klejch 2022 used a grapheme lexicon, 20 minutes of speech and 50 restarts, and still failed on 3 of 7 languages. (src: SAE_4A_lexlat.md § Objective; § Constraints; § Orchestrator rulings; § Design 9)

## B2. Resources

**Trie.** 151,731 words, 132,049 distinct pronunciations, maximum length 33, 291,476 nodes. Mean pronunciation length is 3.579 phones token-weighted (6.486 type-weighted), which fixes the phones-per-word band at 0.7-1.4x, i.e. [2.505, 5.010].

**In-house word trigram.** Built with `lmplz -o 3 --interpolate_unigrams True --discount_fallback` on the 1 M-line window. It has 151,734 unigrams, 3,393,577 bigrams and 10,419,405 trigrams, 3,545,312 CSR back-off states, and a 274,266,991-byte binary.

**Shuffled-pronunciation null.** A seed-0, single-cycle derangement of the word-to-pronunciation map with 0 fixed points. The trie arrays and the word-LM marginals are bit-identical to the treatment's, so the null destroys word-identity routing only, not segmentability. A random-pronunciation null was named but never run.

**Word-LM quality** on reference transcripts, in-vocabulary subset:

| LM | ppl dev-clean / dev-other | OOV % |
|---|---|---|
| in-house 1 M-line word trigram | 244.85 / 220.65 | 0.531 / 0.764 |
| official 3-gram.pruned.1e-7 | 244.27 / 222.92 | 0.349 / 0.579 |
| official 3-gram.pruned.3e-7 | 301.58 / 274.01 | 0.349 / 0.579 |
| official 4-gram, full | 148.42 / 137.34 | 0.349 / 0.579 |

The in-house trigram matches the official 1e-7 trigram within 1 %. Only the full 4-gram is substantially stronger. (src: SAE_4A_lexlat.md § Resource constants; § Official-LM benchmark)

## B3. DP form: lexical contexts inside the trigram lattice (abandoned on cost)

**State.** The bed's DP state (band offset × 41×41 phone history × repeat flag) gains a dynamic axis of C lexical contexts. A context holds a trie node, a word-LM state id (longest matched suffix, CSR back-off) and the last two phones.

**Transitions.** Emitting phone k continues the current word for free. At a word end the path may also close word w at log p_wordLM(w | state) and restart at the root. SIL forces a word close and closes an open escape. ESCAPE is priced exactly as in Step 0. The lexical increment is constant in theta and phi, so it enters the manual backward the same way beta log P3 does.

**Pruning.** The top C contexts per frame are kept by forward mass, with C_esc = 64 slots reserved for escape contexts. Only word-end, escape and root contexts are final. A NEG_INF log Z aborts the arm.

**Curriculum and guards.** lam_lex = 0 for sub-epochs 1-7, a linear ramp over 8-10, then full. The rate term is unchanged. The G4a.4 abort (rate < 0.6 rho for 5 sub-epochs) is re-armed at the on-set. Phones per word must stay in [2.505, 5.010].

| probe | rule | result |
|---|---|---|
| E-1 (CPU): max-plus trie DP vs Step 0 best segmentation | ≤ 1e-4 nats/token, else stop | PASS, worst 2.3e-07; gold −2.1945, private −4.4613, pooled gap 2.2667 (2.32 is the per-utterance aggregate) |
| E-1 (a): G2P-free trie (133,942 words) | disclosed | gap 2.2706 (+0.004): no loss |
| E0 census (ctrl_50 ep10, 100 utterances, tau 2) | median pruned mass < 0.05 at every lam; \|log Z(C) − log Z(4096)\| < 0.05 per retained frame; 0 NEG_INF | C 1024 misses at 0.0512; C re-declared 4096 |
| E0 kill (a): max-plus decode at lam 1, C 1024, vs greedy | paired PER ≤ −0.010 | PASS, −0.0509 (0.9095 → 0.8604); a lam-0 Viterbi gives −0.0330, so the lexicon's own share is about 0.018 (interpretation) |
| E0 kill (b): drop in E[N] | ≤ 10 % | PASS, marginal: 9.66 % (54.06 → 48.63) |
| E1 cost at run shape (one GH200, 88,000 frames) | ≤ 1202 s per sub-epoch (2.00x the bed's 601 s), ≤ 80 GiB | FAIL: C 4096 out of memory (> 95 GiB); C 1024 at 802-912 s per step (mean 856) and 41 GiB, about 48,800 s per sub-epoch, 41x the bar |

**E0 at lam 1** (log Z per retained frame / s per utterance at B = 1): C 256 −2.6507 / 0.69; C 1024 −2.6095 / 0.77; C 4096 −2.5594 / 2.07. Median pruned mass is 0.0000 in every cell, the escape phone fraction 0.649, phones per word 2.630.

**Profile at C 1024.** 873 s per step, against 12.3 s for the trigram-only step (71-76x). Each step runs three augmented DP passes, because the rate term's two central finite-difference passes also go through the augmented DP. The GPU is 97 % busy. The work is memory-bound elementwise work (where, log-sum-exp, gathers) over a [B, C, m_max, O] float64 tensor; GEMMs take under 1.5 %. Value-identical levers multiply to well under the needed 41x.

**Conclusion.** The DP core cannot reach the bar. The published GPU forward-backward rate (about 4.4e9 arc updates per second) predicts the measured cost from the arc count (about 2e8 per utterance-frame). Two routes remained: (A) a pruned marginal with gradients over H·L·G in k2, which became amendment 9; (B) max-plus lexicon decoding plus self-training, the wav2vec-U recipe, never run. The user tier still stands: once every cost fallback is exhausted, up to 2-3 days per 20 sub-epochs is acceptable (about 150-227 s per step, ≤ 80 GiB), with a resume test across the wall clamp. (src: SAE_4A_lexlat.md § Design 1-7; § E-1; § E0; § E1; § Cost analysis; § Profile)

## B4. The k2 form ("amendment 9", binding) and gate G4a.9

This is the form implemented in the ported package, `model/lexlat_k2*.py`.

**Term.** Per utterance, L_lex = log Z_HLG(e / tau) − log Z_H(e / tau). e: the recognizer's per-frame phone log-probs at the arm's tau. Z_HLG: the log-semiring total of the pruned intersection with the priced HLG, computed with `k2.intersect_dense_pruned` then `get_tot_scores(log_semiring=True)`. Z_H: the unpruned total under H, the bed's min-duration run-collapse phone topology. It is checked against the bed's transcript log-probability to 1e-4. −L_lex ≥ 0 is minus the log posterior mass of word-decomposable strings, escape included. The loss adds lam_lex × (−L_lex) / n_unit_frames, and the gradient reaches the recognizer only, through k2 autograd. The bed's lattice term (recognizer + reverse + beta P3) and the rate term are unchanged; the two graphs are added.

**Graph.** **L:** the pronunciation lexicon with ESCAPE and SIL at sil_prob 0.5. It has 3,120 `<unk>` arcs per graph, carried by the start state. Each is priced at the per-phone escape price plus the `<unk>` LM arc (−14.97), so −28.4 to −18.1 per arc; an escaped phone costs −8.562 to −3.125 nats. **G:** the word LM as an automaton with epsilon back-off, built from the CSR tables. theta is the LM pruning threshold in nats; 0.0 means unpruned. **No determinization.** k2's determinize is tropical-only, so it would under-count log-semiring totals, and it does not terminate on the escape LG. **Back-off `#0` loops** sit only on L's word-boundary states. Loops on every state were the defect found in B5. **Null graph:** the same compile with shuffled=True. Placement, escape, sil_prob and theta are asserted equal.

**Search.** Search beam 20, output beam 8, min_active 30, max_active 1000 (primary) or 3000. At most 16 sequences go into one intersect; more overflow k2's int32 arc count, and flat or high-tau posteriors overflow it regardless.

**Curriculum.** On-set at sub-epoch 8 (5 in the `_e5` arms), ramp 3, full_lam 1. The two sources disagree about sub-epoch 10: the D15 log shows lam = 1/3, 2/3 and 1 at sub-epochs 8, 9 and 10, while the DP design text puts full weight from sub-epoch 11. lam_lex = 1 means "the full constraint", but here it scales a log-mass rather than per-transition increments. At ep10, −L_lex per unit frame is 0.5-0.9, against 1.73-1.85 for the lattice term.

**Empty lattices** get no term and are counted. More than 10 % empty in a batch aborts the arm; a sub-epoch mean above 2 % reads UNINFORMATIVE.

**Monitors:** `lexlat_k2_stability` (|log Z(1000) − log Z(10000)| per retained frame), `_empty_frac`, `_expected_words`, `_expected_escape_words`, `_term_mean`, `_sec`, `_peak_reserved_gib`.

**Gate G4a.9** (pre-registered; per arm, dev-other, sub-epoch 20, never best-PER). Verdicts: **PASS:** PER < 0.50, rate in [5.80, 14.49] per second, treatment − ctrl_20 ≤ −M, treatment − null ≤ −M, and derangement gap > 0. The s1 pair must clear −M too. **FAIL:** health holds, the term was live, and a PASS condition is missed. **CANNOT_TELL:** health fails in both arms, an engagement clause fires, or the lexicon-free arms disagree at ep4 by more than 0.001 paired PER. Margin: M = max(arm bands, |F|, 0.010). B_ctrl = |ctrl_20 − ctrl_20_s1|. B_k2 and B_null are the seed pairs at ep20. F is the pack's common ep4 offset from ctrl_20. Engagement clauses (from sub-epoch 11, 3 consecutive sub-epochs): stability median > 0.05 → CANNOT_TELL; escape share of expected words > 0.90 → UNINFORMATIVE; phones per word (bed E[N] / expected words) outside [2.51, 5.01] → UNINFORMATIVE; treatment and null term magnitudes must match within 2x over sub-epochs 8-11; otherwise the null comparison is only disclosed. Abort: NaN, |surrogate| > 100, or rate < 0.6 rho for 5 sub-epochs after the on-set. (src: SAE_4A_lexlat.md § Design amendment 9; § Amendment 9 BINDING; § Gate; § Round chronology)

## B5. Over-count of the log-semiring graph (amendment 9.5)

With epsilon back-off, both the explicit n-gram path and the back-off path carry mass. On a fixture this gives −2.739 against the exact −2.803 nats, i.e. 0.064 over.

**Check.** The HLG is intersected with linear FSAs of gold and of the ctrl_50 ep10 decodes. Acceptance: median ≤ 0.05 nats per token per set. Strings with an adjacent repeat cannot pass H; they are excluded and counted (gold 772 of 2,864, private 22).

| graph | reference | gold median | private median | read |
|---|---|---|---|---|
| in-house, loops on all states | max-plus (as registered) | +0.287 | +0.162 | FAIL |
| in-house, loops on all states | exact log-sum under CSR back-off | 0.274 | 0.122 | FAIL |
| in-house, word-boundary loops (treatment graph) | max-plus (as registered) | +0.0500 | +0.0535 | FAIL on private by 0.0035 |
| in-house, word-boundary loops | exact log-sum (deciding) | +0.0357 | +0.0173 | PASS, audited |
| official 4-gram, word-boundary | exact / max-plus | +0.0221 / +0.0327 | +0.0159 / +0.0976 | PASS on exact |

**Audit split** of the all-states graph on a 60-string sample (gold / private): legitimate multi-parse mass +0.013 / +0.019; plain back-off route sum +0.033 / +0.017; positional multiplicity of `#0` loops on every L state +0.240 / +0.110. The sample's exact over-count was +0.274 / +0.117; the full job gave 0.274 / 0.122.

**Disclosed.** The deciding column was re-specified from max-plus to exact after max-plus had failed on the old graph, but before the rebuilt graph existed. p95 exceeds 0.05. 27 % of gold strings are excluded. The term is therefore an approximate word constraint, with a median double count of 0.02-0.04 nats per token. (src: SAE_4A_lexlat.md § Over-count read; § Official 4-gram word-boundary graph)

## B6. k2 cost numbers

**The amendment-9 bar.** Bed 601 s + lexicon leg ≤ 1202 s per sub-epoch, and ≤ 80 GiB, on one GH200 96 GB GPU. Probes use the ctrl_50 ep10 posterior at tau 2 and 9 batches of sub-epoch 10 (including the longest-T batch), with 57 steps per sub-epoch.

**Settling probes** time the lexicon leg only: forward, total, and backward to the emissions. Each cell is s per sub-epoch / peak GiB / median stability.

| graph (HLG states / arcs) | max_active 1000 | 3000 | 10000 |
|---|---|---|---|
| in-house, all-states loops (24.7 M / 143.9 M) | 309 / 43.3 / 0.034 | 353 / 46.3 / 0.014 | 476 / 62.3 / 0 |
| in-house, word-boundary loops (23.9 M / 98.6 M), treatment | 182 / 23.4 / 0.028 | 218 / 27.2 / 0.013 | 310 / 38.1 / 0 |
| official 3-gram 3e-7, all-states (9.8 M / 81.6 M) | 720 / 62.4 / 0.029 | 781 / 67.8 / 0.013 | 953 / 83.3 / 0 |
| official 3-gram 1e-7, all-states (20.4 M / 169.4 M) | 455 / 59.7 / 0.030 | 527 / 70.6 / 0.013 | 704 / 88.9 / 0 |
| official 4-gram theta 5, all-states (33.4 M / 181.1 M) | 981 / 69.7 / 0.032 | 1031 / 72.6 / 0.014 | 1189 / 88.5 / 0 |
| official 4-gram, word-boundary | 289.8 / 37.96 | 330.3 / 39.59 | – |
| official 3-gram 1e-7, word-boundary | 319.5 / 39.16 | 371.2 / 40.13 | – |

The rung rule (the smallest rung that passes the bar with median stability ≤ 0.05) picks 1000 on every graph. The maximum over utterances reaches 0.09-0.13. With the bed's 601 s added, the all-states official graphs fail on time; the word-boundary ones pass. The official 4-gram could only be built at theta 5.0 (145.3 M n-gram arcs pruned to 7.6 M). The size guard declined theta 0 / 0.5 / 2 at predicted sizes of 391 / 374 / 192 GiB. Build costs: in-house HLG 173.5 s / 26.0 GiB RSS; official 4-gram 347 s / 31.1 GiB; official 1e-7 119 s / 20.8 GiB.

**Real train step** (whole step with backward, treatment graph): rung 1000: 31.22 GiB, 3.302 s leg, 789.2 s per sub-epoch; rung 3000: 31.77 GiB, 3.955 s leg, 826.5 s per sub-epoch. Both pass. Rung 10000 was never tested with backward. In-run on-set reads show 17.4-21.9 GiB and 1.87-6.51 s per step.

**Measured wall time** per sub-epoch (sub-epochs 16-20): k2lat_20_ma3000 760.6 s, k2shuf_20_ma3000 948.6 s (the null graph is slower), k2lat_20 734.8 s, ctrl_20 610.0 s. A 4-arm, 20-sub-epoch pack on one exclusive 4-GPU node took 4:16-4:58 h. (src: SAE_4A_lexlat.md § k2 settling probes; § Step probes; § Official probes; § Packs, in-run kill reads; § E60 Cost)

## B7. Positive control of the k2 implementation

**Setup.** 100 utterances, tau 2.0, max_active 1000. Emissions are either gold-aligned (mass 0.98 on the MFA phone), the cold ctrl_50 ep10 posterior, or label-permuted noise.

| cell | −L_lex nats per gold token | escape share | E[words] |
|---|---|---|---|
| gold / treatment | 1.948 | 0.000 | 16.2 |
| cold / treatment | 2.127 | 0.003 | 15.9 |
| noise / treatment | 3.550 | 0.021 | 17.3 |
| gold / null | 2.811 | 0.000 | 12.1 |
| cold / null | 2.470 | 0.006 | 11.3 |
| noise / null | 3.741 | 0.063 | 11.2 |

The gradient pushes the gold phone up on 0.976 of frames for the treatment graph and 0.859 for the null.

**Original verdict: FAIL (6 of 10).** It stays on record as a mis-specified control. A debugger and an auditor, working independently from fresh contexts, showed the expectations were unreachable by any correct implementation: eps 0.02 tempered at tau 2 caps the frame peak at 0.529; the transcript's word cost is 1.219 nats per gold token even in the delta-emission limit; an escape costs about 20 nats per word more than tiling the string with deranged-inventory words.

**Amended control 9.7.** (a) log Z_H equals the k2-free closed form to 1.9e-12 on 100 of 100 utterances. (b) −L_lex is within the single-string bound on 99 of 100 (worst +1.5 nats, in the direction pruning moves it). (d) the escape arcs are reachable. (c) 5 of 6 orderings hold. The null-graph "gold < cold" ordering fails (2.811 vs 2.470), because a deranged lexicon has no reason to prefer gold; by the letter this is CONTROL_FAIL_9_7. The implementation itself is audited correct. (src: SAE_4A_lexlat.md § Design review amendment 9.7; § Implementation controls; § Positive control)

## B8. The twelve-arm round and the G4a.9 verdict

Every arm ran 20 sub-epochs with ctrl_20's seeds; the s1 arms used ctrl_20_s1's.

| node | arms | graph / rung |
|---|---|---|
| 1 | k2lat_20, k2shuf_20, k2lat_20_e5, k2lat_20_s1 | in-house word-boundary, 1000 |
| 2 | k2shuf_20_s1, k2shuf_20_e5, k2lat_20_ma3000, k2shuf_20_ma3000 | in-house; 1000 (ma3000 pair at 3000) |
| 3 | off4_k2lat_20, off4_k2shuf_20, off4_k2lat_20_s1, off3_k2lat_20 | official 4-gram theta 5 / official 3-gram 1e-7; 1000 |

**Ported presets.** `k2lat_20_ma3000` is the in-house trigram, theta 0, max_active 3000, full phone trigram. `off4_k2lat_20` is the official 4-gram at theta 5.0, max_active 1000, full phone trigram. There is no null preset.

**Pre-on-set identity.** At step 1 every arm reproduces ctrl_20 (l_tau −0.350, prior per token −5.657, expected tokens 63.821); each s1 arm reproduces ctrl_20_s1 (−0.347, −5.671, 57.498).

**PER at ep20.** Controls: ctrl_20 0.8746, ctrl_20_s1 0.8751. Treatments: 0.819 (k2lat_20_ma3000, 0.8186) to 0.843 (k2lat_20, 0.8435). Nulls: 0.814 (k2shuf_20_ma3000, 0.8142) to 0.843 (k2shuf_20_e5). The other per-arm values are not in the log.

| pair (ep20) | delta [95 % CI] |
|---|---|
| k2lat_20 − ctrl_20 | −0.0311 [−0.0363, −0.0262] |
| k2lat_20_s1 − ctrl_20_s1 | −0.0416 [−0.0486, −0.0349] |
| k2lat_20_e5 − ctrl_20 | −0.0437 [−0.0500, −0.0379] |
| k2lat_20_ma3000 − ctrl_20 | −0.0560 [−0.0628, −0.0499] |
| off3_k2lat_20 − ctrl_20 | −0.0475 [−0.0532, −0.0423] |
| off4_k2lat_20 − ctrl_20 | −0.0399 [−0.0464, −0.0340] |
| off4_k2lat_20_s1 − ctrl_20_s1 | −0.0349 [−0.0409, −0.0293] |
| k2shuf_20 − ctrl_20 | −0.0464 [−0.0541, −0.0393] |
| k2shuf_20_s1 − ctrl (control seed not stated in the source) | −0.0613 [−0.0684, −0.0547] |
| k2shuf_20_e5 − ctrl_20 | −0.0316 [−0.0377, −0.0262] |
| k2shuf_20_ma3000 − ctrl_20 | −0.0604 [−0.0672, −0.0540] |
| off4_k2shuf_20 − ctrl_20 | −0.0456 [−0.0534, −0.0387] |
| **k2lat_20 − k2shuf_20** | **+0.0153 [+0.0120, +0.0186]** (ep10 +0.0154) |
| **k2lat_20_s1 − k2shuf_20_s1** | **+0.0197 [+0.0174, +0.0220]** (ep10 +0.0147) |
| **k2lat_20_ma3000 − k2shuf_20_ma3000** | **+0.0044 [+0.0007, +0.0080]** (ep10 −0.0004 [−0.0036, +0.0026]) |
| **off4_k2lat_20 − off4_k2shuf_20** | **+0.0057 [+0.0033, +0.0084]** (ep10 −0.0072 [−0.0094, −0.0048]) |
| k2lat_20_ma3000 − k2lat_20 | −0.0249 [−0.0279, −0.0219] |
| off4_k2lat_20 − off3_k2lat_20 | +0.0077 [+0.0052, +0.0100] |
| off4_k2lat_20 − k2lat_20 | −0.0088 [−0.0113, −0.0064] |

**Bands.** B_k2 0.0092 [0.0069, 0.0120]; B_null 0.0136 [0.0118, 0.0154]; B_off4 0.0062. The ep4 full-seed band, ctrl_20 − ctrl_20_s1, is +0.0021 [+0.0002, +0.0039].

**The ep4 lexicon-free check failed in every node:** +0.0111, −0.0033 and −0.0146 against a 0.001 tolerance. The audited cause is the bed's run-to-run GPU nondeterminism, amplified in the cold phase: ep1 means agree to 2e-7; divergence starts at ep2, when lr and tau change; arms with identical pre-on-set configs differ by up to 0.0257 at ep4. F is therefore the identical-config ep4 spread, 0.013-0.015, which makes M = 0.015 for node 1.

**Engagement and health.** Stability ≤ 0.044 from sub-epoch 11; empty fraction ≤ 0.0010. Escape share ≤ 0.24. Treatments sit at 0.0001 (in-house), 0.05 (official 4-gram) and 0.20 (official 3-gram). The term-magnitude ratio of treatment to null is 1.2. Phones per word is 3.4-3.6 for treatments, inside the band, and 5.4-6.9 for every null. By the letter the clause fires for every null: the null matches the term's magnitude but emits about 40 % fewer words. Derangement gap 3.04-3.61; rate 7.3-8.1 Hz.

**VERDICT G4a.9 (audited): CANNOT_TELL as written, from the ep4 clause. PASS is unreachable under any admissible M:** clause 2 (treatment − null ≤ −M) is positive, with its whole interval above zero, in all four registered pairs, and every PER is 0.81-0.84.

**Licensed:** not funding the marginalised k2 term further at this operating point, i.e. rungs 1000 and 3000; the in-house trigram, the official 3-gram 1e-7 and the official 4-gram theta 5; on-sets 5 and 8; two seeds.

**Found:** the term lowers PER by 0.03-0.06 against ctrl_20 in every pack and seed. A magnitude-matched deranged lexicon lowers it as much or more, so on the cold line the gain is prior weight, not lexical content.

**Lesson for new gates:** identical cold configs differ by 0.01-0.03 PER by ep4 (D15: 0.013-0.020), so an ep4 tolerance of 0.001 cannot hold on this bed. (src: SAE_4A_lexlat.md § Launch round; § Packs, in-run kill reads; § Packs, registered paired PER reads; § VERDICT)

## B9. Diagnosis of the cold-line failure (D0-D9, D11)

These reads are descriptive, each with a rule registered before its number, on the ep20 dev-other greedy decodes. "GAN" means wav2vec-U 2.0 at update 148000.

**D0, frequency-stratified PER.** In word-frequency deciles 0-7 the null is at or below the treatment. In decile 9 the treatment wins 4 of 6 pairs (k2lat_20 0.741 vs 0.784; off4 0.692 vs 0.757) but loses at ma3000 (0.784 vs 0.714). D3 identifies this as mode-seeking onto frequent words.

**D1, n-gram JSD to the text** (count-matched at 134,710 phones), JSD3 / JSD4 at ep20: gold 0.073 / 0.248; GAN 0.084 / 0.277; ctrl_20 0.506 / 0.748; k2lat_20 0.498 / 0.714; k2shuf_20 0.456 / 0.695; off4_k2lat_20 0.521 / 0.734. Every arm, the control included, moves 0.04-0.09 from ep4, and the nulls move as far as the treatments. The treatments' trigram log P per phone rises to within 0.2-0.4 of gold (k2lat_20 −2.96, off4 −2.76, gold −2.54, ctrl −3.37) while their JSD does not: mode-seeking.

**D2, loss curves.** k2 PER drops about 0.04 from ep4 to ep10, then at most 0.015; ctrl_20 is flat (0.875 / 0.869 / 0.875). The lexicon term is lower for treatments (0.285-0.306) than for nulls (0.315-0.359), with 27-32 against 16-17 expected words. Any lexicon graph lowers agg (1.52 → 1.03-1.47) and rate (8.8 → 8.2-8.5 Hz) and leaves l_tau and the reverse term unchanged. The effect therefore runs through phone-level statistics that treatment and null share. D2's "not converged" reading was overturned by E60.

**D3, word structure** (coverage / word recall at ep20): k2lat_20 0.816 / 0.075; k2shuf_20 0.572 / 0.028; ctrl_20 0.517 / 0.036; off4_k2lat_20 0.891 / 0.069; k2lat_20_ma3000 0.708 / 0.032; GAN 0.922 / 0.617; gold 0.989 / 0.810. Word precision is 0.087 for k2lat_20, against 0.565 for the GAN. The paired treatment − null recall gain (+0.015 to +0.047) sits in deciles 8-9, and the top 100 types carry 0.61-0.63 of treatment tokens against 0.49 in gold. Structure is learned; content is not.

**D4, reverse-model preference** (500 utterances). Gold − decoded: −53.5 nats per utterance at ep1, −945 to −1187 at ep20 (audit-corrected range; no longer growing in the treatments). Gold − deranged gold: −5.6 at ep1, 15-74 at ep20. Paired treatment − null on the latter: +17.7 (node 1), +47.2 (ma3000), +46.2 (off4). The true lexicon teaches the reverse model some real phone order, 1-2 % of the utterance score, which PER cannot show.

**D5, word-LM conformance** (per-token HLG log P, 1,737 common strings). In-house graph: gold −2.36, GAN −3.21, k2lat_20 −3.93, k2shuf_20 −4.31, ctrl_20 −4.35. Official graph: gold −2.15, off4_k2lat_20 −3.43, ctrl −3.94. Treatments close 0.43-0.52 of the control-to-gold distance, nulls 0.04-0.12. Under the shuffled HLG all arms tie.

**D6, optimisation.** 1,140 updates is about 5 passes over the data. The GAN made about 830 passes (148,000 updates) with the same recognizer shape (1.42 M vs 1.46 M parameters). E60 later refuted "too short".

**D7, pruning mass** (73 H-admissible golds of 100). Rung 1000 is within 0.009-0.023 nats per frame of rung 10000 (audit-corrected range). Gold survives in 0/73 lattices at every rung, sitting 270-430 nats below log Z. The top-1 share is about 0. The objective prefers non-gold paths; pruning is not the cause. k2 prunes on the forward score only, a variant the MMI literature finds weaker (Qin and Rudnicky 2010).

**D8, identifiability channel.** sigma_min of the positional phone-unigram matrix is 7.05e-4 for the true lexicon and 3.48e-4 for the deranged one: ratio 1.948 [1.651, 2.249]. A first in-process read gave 2.03 [1.65, 2.25]. For the phone trigram it is 2.6e-18, i.e. rank-deficient. Verdict CANNOT_TELL: the factor of 2 is a length effect, since deranged sentences are 1.83x longer.

**D9, lattice oracle PER** (500 utterances). The HLG oracle at the own rung is 0.61-0.70 (k2lat_20_ma3000 ep20: 0.645 at 3000, 0.620 at 10000). Greedy is 0.82-0.88 and the lattice 1-best 0.80-0.83. The same posterior through H alone reaches 0.424 [0.420, 0.428]. The lexicon graph therefore excludes the recognizer's closest paths. The treatment's oracle worsens with training (0.611 at ep4, 0.645 at ep20). ctrl_20's posterior keeps a lower oracle than every treatment.

**D11, official 4-gram word perplexity** (best trie segmentation). Reference 140.12; segmented gold phones 857.8; GAN 2478; ctrl_20 18508. k2lat_20 10811 vs k2shuf_20 21195; off4_k2lat_20 12487 vs off4_k2shuf_20 17445. Treatments sit 13-33x above segmented gold: their words are chosen without word-sequence context.

**Literature constraining follow-ups.** Identifiability needs a full-rank positional phone-unigram matrix (Yang, Schlüter, Ney 2026; Wang et al. ACL 2023; ESPUM 2024). A deranged lexicon keeps a comparable matrix, so a null matching the treatment is the predicted outcome. In MMI a stronger LM is a reproduced negative: the LM scale is inert and the acoustic scale decides. The follow-up knob is therefore tau, not lam_lex. Only phone JSD4 has a known success threshold, about 0.27 (Lin et al. 2022). In the audited write-up, every k2 arm sits 0.03-0.06 below ctrl_20 and within 0.01 of its null. (src: SAE_4A_lexlat.md § Diagnosis; § D11)

## B10. Warm-start diagnostics D10 and D10e (label-using, analysis only)

**p0.** The bed's recognizer, trained on the labelled 10 h seed (2,821 train / 28 held-out) with −transcript_logprob. It is selected at minimum held-out loss, which falls at pass 1 (24 Adam steps): PER 0.1894. It overfits from pass 2 on; pass 30 reads 0.2067.

**Diagnostic pack** (p0 plus a random phi): tau is held at 2.0 for 8 sub-epochs, because at tau 8 the flat posterior overflows k2's int32. The k2 on-set is sub-epoch 1 with ramp 3. Kept checkpoints: 1 / 2 / 4 / 8. PER ep4 / ep8: sup_plain 0.8915 / 0.8915; sup_k2lat 0.8124 / 0.8139; sup_k2shuf 0.8199 / 0.8190; sup_off4 0.8059 / 0.8107. Paired at ep8: k2lat − plain −0.078; k2lat − k2shuf −0.0050 [−0.0081, −0.0015], which misses the −0.010 bar. The reading "the added weight protects" was **overturned by D10e**.

**D10a-d.** Collapse happens in sub-epoch 1: sup_plain is at 0.846 after 57 updates, mostly substitutions, and symbol entropy falls from 4.86 to 2.81 bits. Sub-epochs 2-8 rebuild the symbols along unit lines: NMI(symbol, unit) 0.358, NMI(symbol, phone) 0.068. No row is a relabelling. The unit-code clause is ambiguous as registered and was not resolved. At ep8 the reverse model prefers the decode over gold by 767.8-1077.8 nats per utterance.

**D10e, both-gold pack.** p0 plus a gold-fitted phi, refit with the supervised-reverse recipe on the 10 h seed. Precondition: gold − deranged gold under this phi is +1000.4 nats per utterance.

| arm | ep1 | ep2 | ep4 | ep8 | arm − p0 at ep8 |
|---|---|---|---|---|---|
| supphi_plain | 0.1931 | 0.2641 | 0.2936 | 0.2561 | +0.0666 [+0.0604, +0.0727] |
| supphi_frz (phi frozen) | 0.1868 | 0.2493 | 0.2663 | 0.2169 | +0.0275 [+0.0212, +0.0340] |
| supphi_k2lat | 0.1757 | 0.1881 | 0.2010 | 0.1797 | −0.0097 [−0.0151, −0.0045] |
| supphi_k2shuf | 0.2217 | 0.2870 | 0.3723 | 0.3656 | +0.1762 [+0.1689, +0.1833] |

Paired at ep8: frz − plain −0.039; k2lat − plain −0.076; k2lat − k2shuf −0.186 [−0.191, −0.180]; k2shuf − plain +0.110.

**Readings** (audited, with caveats): The untrained reverse model drove the collapse. Co-adaptation does not drive the collapse. A trainable phi still costs 0.039 and turns toward the decode (450 of 500 utterances, against 208 frozen). On a phone-like start the k2 control is CONFIRMED and lexicon-specific: the deranged lexicon hurts. supphi_k2lat PRESERVES p0; −0.0097 misses the REFINES bar by 0.0003. About half of the k2lat − k2shuf gap comes from the null losing to deletions. The term preserves or refines a phone-like start (NMI(symbol, phone) 0.86), but cannot create one from the cold unit code (0.07). This rests on one seed. (src: SAE_4A_lexlat.md § Supervised blank-free recognizer; § Diagnostic pack; § D10; § D10e)

## B11. Extension E60: training length

The user asked for training to the PER plateau. Four arms were continued from ep20 with their Adam state: k2lat_20_ma3000_x60, k2shuf_20_ma3000_x60, ctrl_20_x60 and k2lat_20_x60. They ran 40 more sub-epochs at lr 1e-4 held (a warm restart from ep20's 1e-5), with tau 2.0 and lam 1. Kept checkpoints: 21 / 30 / 40 / 50 / 60. The ported presets `ctrl_20_x60` and `k2lat_20_ma3000_x60` instead run one 60-sub-epoch job. The config is the same, but the trajectory is not bit-identical.

**Gate E60** (M = 0.015): continuity: |PER(ep21) − PER(ep20)| ≤ M; plateau: ep60 − ep50 within [−0.010, +0.010]; form: G4a.9's PASS clauses on k2lat_20_ma3000_x60.

| arm | ep20 | ep21 | ep30 | ep40 | ep50 | ep60 | ep60 − ep50 |
|---|---|---|---|---|---|---|---|
| ctrl_20_x60 | 0.8746 | 0.8605 | 0.8734 | 0.8748 | 0.8747 | 0.8735 | −0.0012 [−0.0029, +0.0001] |
| k2lat_20_x60 | 0.8435 | 0.8398 | 0.8490 | 0.8444 | 0.8441 | 0.8484 | +0.0043 [+0.0029, +0.0058] |
| k2lat_20_ma3000_x60 | 0.8186 | 0.8158 | 0.8244 | 0.8203 | 0.8205 | 0.8240 | +0.0035 [+0.0025, +0.0044] |
| k2shuf_20_ma3000_x60 | 0.8142 | 0.8135 | 0.8114 | 0.8142 | 0.8160 | 0.8169 | +0.0008 [−0.0007, +0.0025] |

Paired at ep60: ma3000 − ctrl −0.0495; k2shuf_ma3000 − ctrl −0.0566; k2lat_20 − ctrl −0.0251; ma3000 − null +0.0071 [+0.0035, +0.0107]; ma3000 − k2lat_20 −0.0244.

**Gate result (audited).** Continuity passes, and all four arms read PLATEAU. The treatments drift slightly upward, so the plateau is not convergence at their best point. Form is not PASS: PER 0.824, and the null delta is +0.0071. Rate 7.81 Hz, derangement gap 3.84. The run took 10 h 31 min on one exclusive node. "Too short" is refuted: 2,280 extra updates moved no arm by more than 0.01, so the objective sits at a fixed point. After the warm restart, l_tau recovers only part of its rise, the lexicon term worsens slightly (ma3000 0.310 → 0.320), and agg falls in the treatments while rising in the null. (src: SAE_4A_lexlat.md § Extension E60; § E60 continuity read; § E60 read at ep60; § E60 loss curves)

## B12. D12: screen for a word-histogram aggregate term

An external proposal added a word-level aggregate term, KL(c_text ‖ c_hat), on posterior expected word counts. Its companion ideas were not funded: per-word bonuses, and a one-substitution pronunciation channel. The paths are a unit code, not near-miss words, and relaxing the graph would dilute the only lexicon-specific signal.

**Rule.** SATISFIABLE_CONTENT_FREE if a row with word precision < 0.10 has K ≤ max(1.5 × K(gold phones), K(GAN)); otherwise ROOM.

**Read (audited): ROOM.** The bound is 0.3632 (gold phones 0.197, GAN 0.363). Content-free decodes range from 1.786 (off4_k2lat_20) to 2.229 (ctrl_20). Paired treatment − null K is inconsistent: −0.105, +0.178 and −0.359. ROOM only shows the term has something to pull on. An earlier order-3 phone aggregate descended while PER stayed content-free.

**Pending.** The drafted arm test waits on the user: warm start from k2lat_20 ep20 with lr held; treatment and deranged null, each with and without the word term; M = 0.015. Before any pack, K on the posterior histogram must also exceed the bound. (src: SAE_4A_lexlat.md § D12)

## B13. Objective-landscape reads D13-D18

**D13** (logged CV-holdout dev terms at matched settings; the reverse score is per frame, higher is better):

| run | l_tau | lexlat_k2 | agg | rate | reverse |
|---|---|---|---|---|---|
| gold-init supphi_k2lat ep8 | 1.749 | 0.247 | 0.211 | 0.017 | −3.160 |
| cold k2lat_20 ep20 | 1.845 | 0.297 | 1.319 | 0.016 | −3.353 |
| cold k2lat_20_x60 ep60 | 1.873 | 0.306 | 1.264 | 0.016 | −3.363 |

PHONES LOWER (no intervals): the cold failure is one of search. The user cancelled the dev-other eval.

**D14**, a phi fitted on p0's own seed decodes (seed-decode PER 0.1277). M_w = max(0.010, B_warm), with B_warm 0.0019 at ep4 and 0.0010 at ep8.

| arm | PER at ep8 |
|---|---|
| decphi_plain | 0.2416 |
| decphi_frz | 0.2220 |
| decphi_k2lat | 0.1822 |
| decphi_k2lat_frz | 0.1728 |
| supphi_k2lat_frz | 0.1701 |
| supphi_k2lat_rep | 0.1808 |
| decphi_k2shuf | 0.3618 |

Without k2 the arms DEGRADE from p0 (+0.0521 / +0.0326). The k2 term HELPS (−0.0593 / −0.0492) and is LEXICON-SPECIFIC (decphi_k2lat − decphi_k2shuf −0.1796). The frozen-phi k2 arms REFINE p0 (−0.0166 / −0.0193). With a gold phi, frozen beats joint by −0.0107. The label source hardly matters: within 0.003 of gold.

**D15**, fading the phone trigram where the word graph carries the text. beta(e) = 1 − lam(e) / full_lam, i.e. 1, then 2/3, 1/3, then 0 at full weight; beta scales P3 in the l_tau DP and in the rate term's tilted passes, while the agg target and the escape unigram price are unchanged. This is the ported `phone_trigram="rampout"`; with the ma3000 preset it gives `k2lat_20_ma3000_rp`. `"off"` (no text prior before the on-set) was never run. Cold arms (rung 3000; M_c = 0.0204, from the replicate's +0.0204 against its parent) read at ep20: k2lat_rp 0.8378, k2lat_rep 0.8390, k2shuf_rp 0.8449, k2shuf_rep 0.8174. rp − rep is −0.0012, so the trigram is NOT NEEDED. k2lat_rp − k2shuf_rp is −0.0072 [−0.0095, −0.0047], not lexicon-specific, so the double count was not what masked the lexicon. Identical cold configs differ by 0.013-0.020 at ep4. Warm: supphi_k2lat_rp − rep is +0.0372 at ep8, so there the trigram IS NEEDED.

**D16 / D16r**, crossing theta and phi between g (supphi_k2lat ep8) and c (k2lat_20_x60 ep60) on the training loss code, dev-other, identical batches. Identity checks reproduce logged values to 6 decimals. Totals (L + 0.1 agg per frame): gg 2.234974, gc 3.826603, cg 3.739597, cc 2.523745. A phi swap costs +1.59 / +1.22 and a theta swap +1.40 / +1.41; the interaction is I = −2.8119. Matched gap cc − gg: +0.1818 [+0.1704, +0.1935] on L.

**TWO BASINS** (audited): the pairs are co-adapted, with a swap costing 1.2-1.6 against a 0.18 gap. The reverse score alone agrees: phi swaps −1.797 / −0.889, matched gap −0.255. This is a discrete swap test, not proof of continuous local optima.

**D17**, a general-knowledge duration prior: a maximum-entropy law on [2, D_k] with mean m = (50 / rho) × retained/original = 4.4138 retained frames per phone. SIL stays uniform. Variants: durinit (trainable) and durfrz (frozen phone rows). On the random-phi pack every arm still collapses (0.75-0.80 at ep1, 0.82-0.86 at ep8). sup arms HELP AT THE EDGE (−0.0363 / −0.0406 against a 0.03 floor), but only from 0.89 to 0.85, inside the chance band. k2lat arms: NO EFFECT. The untrained phi's emissions carry the collapse. The same prior is used in lexlat_v2.

**D18.** Every D14, D15 and D17 arm is still descending at its last kept epoch. On one common objective, the frozen-phi arms are worse by +0.107 / +0.120 (mostly l_tau) despite better PER. Training phi lowers the objective by co-adaptation. The k2 term and the lexicon agree with PER. At beta 0 the objective prefers the worse-PER arm (warm −0.025, cold −0.160). Duration control lowers L by 0.015-0.030 while PER collapses. Mean non-SIL segment length is 4.24-4.52 retained frames (prior 4.41, MFA 4.14). (src: SAE_4A_lexlat.md § D13-D18)

## B14. Where the line stands

**Findings.** On the cold bed, a marginalised word graph, true or deranged, lowers PER by 0.03-0.06. The result stays in the chance band, has no lexicon-specific gain, and is at a fixed point by 60 sub-epochs. On a phone-like start the same term is strongly lexicon-specific and preserves or refines the start. The objective ranks the phone-like pair below the cold end point, but the two are co-adapted basins, so the cold failure is one of search. Training phi and removing the phone trigram each move the objective's optimum away from the better-PER solution. Cold-line escape work continues in lexlat v2 (`SAE_i6_ref_lexlat_v2.md`).

**Open user forks.** (1) The k2 form on the cold line, with a random-pronunciation null or rung 10000. (2) The word-aggregate arm (D12 = ROOM). (3) The double-counted prior: D15 ran, but the beta-2 control and the P3-divided graph did not.

**Not funded:** lam_lex arms, escape-price arms, the random-pronunciation null, rung 10000, and route (B), decode-and-self-train.

(src: SAE_4A_lexlat.md § State; § Launch round)
