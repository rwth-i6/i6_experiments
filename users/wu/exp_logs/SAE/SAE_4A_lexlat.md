# SAE 4A -- the lexicon inside the marginalised lattice

## State

Phase opened 2026-09-21 on the closure of `SAE_4A_prior.md` Step 0b and of its training arm: no phone LM trained on this text reaches the Step 0b bar (gaps 1.71 / 1.861 / 1.849 / 1.809 against the bar 2.01), and the score-function arm was not funded by falsifier (ii) -- r(gold) exceeds every one of G = 8 posterior draws in 100 / 100 / 97 % of utterances at ep1 / ep4 / ep10, above the pre-registered 95 % rule (`SAE_4A_prior.md` Results; audited `reports/audit_sf_probe_2026-09-21.md`). The only prior that discriminates the private code from gold is the word constraint itself (lexicon ESCAPE gap 2.32 nats per token against the trigram's 1.39, Step 0), so it has to enter the objective rather than a sampled reward. Design drafted from the banked survey and literature (`reports/draft_lexlat_phase_2026-09-21.md`), the five open points ruled by the orchestrator (section "Orchestrator rulings"). Design review 2026-09-21 (`reports/design_review_lexlat_2026-09-21.md`): APPROVE_WITH_AMENDMENTS, no ruling overturned; all eight amendments applied in place (section "Design review amendments"). Round 1 built 2026-09-21 (speech-llm 319ce90, new modules only: `sae/emc/lexlat.py`, `lexlat_jobs.py` with `LexiconTrieBuildJob` / `LexlatEquivalenceProbeJob` (E-1) / `LexlatCensusJob` (E0), `test_lexlat.py`, `configs/config_sae_4a_lexlat_probes_v1.py`, shim `config/sae_4a_lexlat_probes.py`; `reports/impl_lexlat_r1_2026-09-21.md`: 31 tests pass, lam_lex = 0 reproduces `lattice.py` log Z to 4e-16, string mode reproduces `prior_gap` to 3e-7 per token, manual backward vs autograd 3e-16). Code review (`reports/review_lexlat_r1_2026-09-21.md`): spec-faithful, escape parity / CSR back-off / reservation / finality / plumbing all verified; one crash in the E0 unpruned cell (arc guard below the 16384-ceiling candidate count), a wrong `exact` flag, and a missing lam_lex = 0 Viterbi control row for kill read (a) are being fixed (implementer, `reports/impl_lexlat_r1_fixes_2026-09-21.md`); a first probe launch was stopped before E0 could submit. Deferred to round 2: a pruned-path autograd test. No manager or watcher of this phase is live. NEXT: relaunch the probe graph after the fixes (executor, own manager + watcher); read E-1 (must reproduce gold −2.19 / private −4.46 per token) and E0's curve and two kill reads against Design 6 as written; then E1; the pack is funded only after E0 and E1 pass.

## Design review amendments (2026-09-21, applied before any job)

1. Gate takes G4a.4's form: PER < 0.50 and rate are PASS conditions, not a health precondition, so the likeliest outcome (a cold arm at PER 0.82–0.91) reads FAIL, not CANNOT_TELL.
2. No bit-equality claim across packs: arms run the banked `lattice.py` path before the on-set; step-1 loss match pre-launch; ep4 agreement of the in-pack lexicon-free arms defines the cross-pack floor F, which enters M.
3. Pruning: escape sub-budget C_esc = 64, stated finality, pruned mass reported as median / p95 / max with the NEG_INF count (one NEG_INF utterance aborts the arm).
4. E0 measured at lam_lex 1/3, 2/3, 1; the pruned-mass engagement clause counts from sub-epoch 11.
5. Null magnitude match read over sub-epochs 8–11 only; later divergence is the success signature.
6. E0 kill reads before any arm: augmented max-plus decode must be closer to gold than the plain decode (paired delta <= -0.010), and E[N] must not drop more than 10 % (else ADD is refused and the blend is re-registered).
7. S / D / I per arm at ep10 and ep20.
8. E1 pins ctrl_20 ep10 parameters and includes the largest-T batch; the cost table counts four arms.

Orchestrator note on the probe checkpoint (2026-09-21): E0 and E1 run at the BANKED ctrl_50 ep10 checkpoint, the Step 0 / falsifier operating point (same bed, same hold-phase LR, its decode and prior-gap rows already banked, so E0's kill read (a) compares directly with Step 0's private-code rows), because ctrl_20 ep10 does not exist until the prepro pack reaches it. Wherever the Design says "ctrl_20 ep10" for a probe, read ctrl_50 ep10; the arms' pairing against ctrl_20 is unchanged.

## Objective

Does a pronunciation lexicon with a word trigram, placed INSIDE the marginalised blank-free lattice objective so the forward-backward sums over word-constrained phone strings, move a cold blank-free arm out of the content-free band at the same budget?

What this builds on is a discrimination measurement, not a training result: the exact lexicon prices gold 2.32 nats per token above the private code where the live trigram prices it 1.39 (`SAE_4A_prior.md` Step 0, audited), and the audit attributes 2.17 of that 2.32 to lexical routing, 0.16 to the escape convention's fixed prices. A discrimination gap does not convert into a gradient: falsifier (ii) showed a sampled estimator never reaches strings in the lexical region, and falsifier (i) showed the lexicon reward nearly flat around the ep4 decode (std 0.50 nats per utterance; 5.2 % of decodes segment strictly against gold's 87.6 %). Marginalising the constraint removes the sampling problem -- every word-constrained path carries posterior mass by construction, so the term is felt at every frame with no string drawn.

A POSITIVE result (paired PER improvement over ctrl_20 beyond the seed band AND beyond the shuffled-pronunciation null) licenses the claim that the prior lever works when the constraint is marginalised rather than sampled, and funds a second seed and a scaling round; it licenses nothing about supervision cost or word decoding, neither of which this phase runs. A NEGATIVE result licenses "the marginalised lexicon at this bed, this pruning budget and this on-set does not improve PER, and prior-side work in this direction is not funded further"; it does not license "a lexicon in the objective cannot work". A result failing the health clause or the pruning monitors licenses neither.

## Constraints

- No transcripts, alignments or any label enter training or checkpoint selection (absolute, `SAE_ref.md` "Current research constraints"). Every label-using read below is a disclosed diagnostic entering nothing. Plain PER only, as scored, never a rescored variant.
- No new control. ctrl_20 and its seed band ctrl_20_s1 exist in the running prepro pack `PackedBlankfreeTrainJob.5EIGJJ1MkcO9` (Slurm 1921103, `SAE_4A_prepro.md` State) and are reused by utterance-paired reads at matched sub-epoch, exactly as G4a.8 reads them.
- Bed unchanged: blank-free trigram lattice DP over 39 ARPAbet phones + SIL, stride 3, band |s - 3t| <= 25, d_min 2, phone D = 25 / SIL D = 50, beta = 1, lambda_rate = 3, rho = 9.6619373279 Hz, lambda_agg = 0.1, batch 88,000 padded frames / max_seqs 128, theta LR 1e-4, phi LR 3e-3, partition_epoch 4, reduction/checkpoint stride 32 (`SAE_4A_blankfree.md` "Registered first model"). Schedule N = 20: anneal 4 sub-epochs tau 8 -> 2, LR warmup 2, hold to 12, linear decay to 20, kept 1 / 4 / 10 / 20, 601 s per sub-epoch, 3.34 h per arm (`SAE_4A_budget.md` "Sub-epoch count for future arms").
- Single delta: exactly one change against ctrl_20 per arm. One extra arm is allowed, for the on-set sub-epoch only (Design 5).
- The lexicon and the word LM are FIXED resources; nothing on the lexicon side is estimated, which is why the precedent's re-smoothing device is not carried (Design 4).
- No edit to any banked blank-free module while the prepro pack runs: running RETURNN jobs re-import the recipe tree on every resubmit, and adding a class to an existing file moves that file's `__sis_version__` and re-hashes its Jobs. New modules only (Design 8).
- G2P status, stated rather than assumed. The Step 0 trie is the phonemisation bliss lexicon (first pronunciation per word) PLUS a Sequitur G2P lexicon for the words it misses (`reports/survey_lexicon_scorer_2026-09-20.md` s1; `phon_lm.py:182-215`), and the bed's own prior text is phonemised through the same chain (`w2vu2/pipeline.py:138-145`, `PhonemizeWithSilJob(..., bliss_lexicon=lex, g2p_lexicon=g2p, ...)`), so G2P-derived pronunciations are already a text-side resource inside the live trigram prior. `SAE_ref.md` "Live reward: no G2P anywhere" is a source trace about the psi / GRPO reward path, not about this bed. The claim "the lexicon carries no G2P" is therefore NOT made here; probe E-1 prices a G2P-free trie and Open decision 1 asks the orchestrator to rule.

## Design (pre-registered 2026-09-21, before any job)

### 1. Scorer and state space

The bed's DP state is `fwd [B, O = 51, H = 1681, 2]`: batch x band offset o = s - 3t + 25 x phone trigram history 41 x 41 x CTC repeat flag, advanced by `_forward_step` (`lattice.py:861-902`), float64 accumulation in `_logmm` (`lattice.py:642-665`), TF32 disabled, finite NEG_INF floor, hand-written backward on detached tables (`lattice.py:1040-1160`); autograd through the expanded transition tensor was rejected at about 9 GB per utterance (`lattice.py:78-92`). Each new segment adds `[log q + beta log P3(k | h) + reverse_segment_score] / tau`.

The treatment replaces the fixed trigram-history axis by a DYNAMIC lexical-context axis of declared size C. A context c carries the trie node n_c (position inside a pronunciation), the word-LM state sigma_c (integer id: the longest matched suffix in the CSR back-off trigram), and the last two emitted phones h_c so the bed's trigram term is preserved exactly. On emitting phone k:

- `n' = child[n_c, k]`; if the child exists, the path continues inside a word at no word-LM cost.
- If n_c is a word end, a word close is an alternative branch: emit w at `log p_wordLM(w | sigma_c)`, sigma' = successor(sigma_c, w), n' = child[root, k]. Homophones are separate branches (19,682 words share a pronunciation, about 13 %; survey s1).
- SIL forces a word close, is never consumed inside a word, and closes an open escape -- Klejch's silence-as-boundary anchor, which `prior_gap.py:461-467` already implements.
- ESCAPE, verbatim from `prior_gap.py:235-237, 430-433, 1555`: one `<unk>` word-LM transition per contiguous non-SIL span plus, per phone, the live Witten-Bell order-1 phone log probability plus log 0.5. ESCAPE and not STRICT, because inside a marginalised sum a `-inf` on non-segmentable prefixes concentrates the forward mass on a vanishing set in the sub-epochs right after the on-set, which is Nuhn and Ney's zero-probability trap in another guise.

The added transition score is `lam_lex x (word-LM increment + escape increment)`, ADDED to `beta log P3(k | h_c)`, not replacing it (Design 3, Open decision 2). Resources: trie over 151,731 words / 132,049 distinct pronunciations, max pronunciation 33 phones; KenLM word trigram order 3 modified Kneser-Ney, 1,000,000 lines / 19,629,091 tokens / 3,302,936 distinct bigram types, `words_o3.bin` 274,266,991 bytes (survey s1, s4). A dense LM table is impossible (151,731 x about 3.45e6 states, about 2 PB); the CSR back-off form of the same binary, about 20M explicit n-grams, is a few hundred MB on device (survey s4).

Gradient: the lexical increment is constant in theta and phi -- nothing on the lexicon side is trained -- so it enters the manual backward exactly as `beta log P3` does today, as an additive per-transition constant. theta and phi receive gradient only through log q and the reverse segment score, reweighted by the lexicon-shaped posterior. The lexicon changes which paths carry mass, not the local derivatives.

### 2. Pruning, as a declared budget

Exact marginalisation is not on the table: Nuhn and Ney (ACL 2014, Table 4) measure exact forward-backward EM over a word-typed state space at 224.88 h for V = 200 with a bigram and state flatly that it is intractable above 200 word types; at V = 3,661 with a trigram only preselection search ran (19.68 h, accuracy 90.92 against beam's 91.16 at 15x the speed).

Declared budget, primary operating point: **C = 1024 lexical contexts per utterance per frame**, kept by highest forward mass (logsumexp over band offset and repeat flag), the rest floored to NEG_INF and the axis compacted. Design review amendment 3, fixed now: (a) an ESCAPE sub-budget of C_esc = 64 of the C slots is reserved for escape-state contexts, because escape paths are dearer per phone and pure mass ranking would prune them first, after which a SIL or the utterance end kills every mid-word prefix and log Z falls to NEG_INF; (b) finality is stated: at the utterance end only contexts at a word end, in an open escape or at the trie root carry final mass, mid-word contexts do not; (c) the census and the training log report pruned mass as median, p95 and max per frame and the count of utterances with log Z = NEG_INF (must be 0 in every sub-epoch; one such utterance is an abort of that arm). The pruned objective is optimised EXACTLY, as in beam EM; the forward sum is a lower bound on the unpruned log Z and is reported as such, not as an error on the true gradient. No LM-side preselection (Nuhn and Ney's B_LM = 50 / B_lex = 5) is carried: the trie already bounds the successor set at most max_len = 33 prefix matches per position times about 1.15 homophones (survey s4). The whole budget is C.

Where the survey does NOT carry over, stated explicitly: its measured 23-38 live states per position (`prior_gap.json:max_states`) is an exact-Viterbi count on a FIXED string, and its conclusion that "the state set per step is tiny compared to lattice.py's 51 x 1681" is derived for that case. Inside the marginalised lattice the label string is free, so the reachable (trie node, word-LM state) count per frame is bounded by the lattice, not by 38. C is therefore measured (probe E0), never assumed.

### 3. Curriculum: when the lexicon turns on, and how strong

Klejch et al. 2022 needed a prior curriculum (character bigram -> character 5-gram -> word trigram + grapheme lexicon) because the full composition "is not feasible to use from the beginning". Falsifier (i) measures the same thing from the other side: at ep4 the lexicon reward has std 0.50 nats per utterance around the decode and 5.2 % of decodes segment strictly; at ep10, std 2.41 and 23.7 % segment. Before the code has lexical structure the term has nothing to grade.

Fixed now: lam_lex = 0 for sub-epochs 1-7; linear ramp 0 -> 1 over sub-epochs 8, 9, 10; full lam_lex = 1 for 11-20. Ten sub-epochs at full strength; kept checkpoints 10 and 20 sit after the on-set, 1 and 4 before it. lam_lex = 1 is fixed and not swept: beta = 1 for the trigram, the lexical increment is in the same nats-per-phone units, and the gold magnitudes are comparable (-2.19 against -3.20 nats per token, Step 0). A sweep would be a second knob; the shuffled-pronunciation null is the control for the added prior weight (Design 6).

### 4. Normalisation and guards

- Per retained frame comes for free: the term lives inside log Z and the lattice term is already `l_tau = mean_b(-log Z_b / retained_b)`. No per-token mean appears anywhere, so the length exploit has no substrate (the `lm_prior_norm = "units"` sign guarantee). Unlike the score-function arm's A5, no separate normalisation is introduced.
- Anti-deletion guard, three parts. (a) The bed's rate term is unchanged (lambda_rate = 3, rho = 9.6619373279 Hz on the original-audio denominator) and is the standing guard. (b) The G4a.4 abort clause is RE-ARMED at the on-set: expected phone rate < 0.6 rho for 5 consecutive sub-epochs after the on-set reads FAIL (collapse). (c) New monitor `lexlat_expected_phones_per_word` (posterior-expected phones between word closes) must stay in [0.7, 1.4] x the lexicon's token-weighted mean pronunciation length -- a resource constant printed by the build job before any arm runs, not a result; the survey banks only max_len = 33, not the mean, so the number is filled from the build job and recorded before launch.
- No re-smoothing against uniform. Klejch's alpha = 0.9 and Nuhn and Ney's identical lambda = 0.9 exist because a LEARNED lexical table can hit zero and never recover. Nothing here is learned on the lexicon side, so the device has no target. Disclosed as a deliberate omission.

### 5. Arms (one pack, one arm per GPU, one exclusive 4-GPU node)

| arm | delta against ctrl_20 | why |
|---|---|---|
| `lexlat_20` | lexicon inside the DP, on-set sub-epoch 8, C = 1024, lam_lex 0 -> 1 over 8-10 | the treatment |
| `lexshuf_20` | identical, pronunciations permuted across words (fixed derangement, seed 0) | the null, and the prior-weight control |
| `lexlat_20_e5` | identical to `lexlat_20` with on-set sub-epoch 5, ramp 5-7 | the one extra knob |
| `lexlat_20_s1` | identical to `lexlat_20` with flat_seed 1 / random_seed 1 / random_seed_offset 1000 (the ctrl_20_s1 seed set) | the treatment's replicate; its ep4 identity check pairs against ctrl_20_s1 (ruling 4) |

Four arms fill the exclusive node. The extra on-set arm is justified and bounded: the on-set is the single constant the one speech precedent says decides the outcome, our own probe brackets it between "flat at ep4" and "awake at ep10", and it costs one GPU of a node that is exclusive anyway. It is the only second knob; if one arm must go, `lexlat_20_e5` goes, never the null.

Every arm keeps ctrl_20's seeds and data order (the second seed keeps ctrl_20_s1's), so the sub-epochs before each on-set are lexicon-free. Before the on-set every arm runs the BANKED `lattice.py` path, not the augmented DP at lam_lex = 0 (design review amendment 2: bit-equality across packs is not achievable, a forward-only rebuild already flipped 8 of 300 decodes, `SAE_4A_prior.md` falsifier (ii)). Pre-registered implementation checks: (a) pre-launch, the step-1 loss of `lexlat_20` matches ctrl_20's logged step-1 loss to 1e-4; (b) at ep4 the lexicon-free arms in the new pack (`lexlat_20`, `lexshuf_20`; `lexlat_20_e5` is already ramping at ep4 and is excluded) must agree with each other to within 0.001 paired PER, and their common paired offset from ctrl_20 is the cross-pack floor F entering M (Gate); disagreement between them is an implementation fault: the pack is stopped and the cause found before ep10 / ep20 are read. `lexlat_20_s1` is checked the same way against ctrl_20_s1.

### 6. Nulls and probes

**Shuffled-pronunciation null, as a PARALLEL ARM, not a pre-funding probe.** The permutation is a fixed-seed derangement of the word -> pronunciation assignment: the same 132,049 pronunciation strings, so trie topology and segmentable set are bit-identical, and the same word LM, so word n-gram marginals are unchanged; only the phone-string-to-word-identity correspondence is destroyed. It targets exactly the 2.17 of the 2.32 nats the Step 0 audit attributes to lexical routing. Three reasons it must be an arm: (i) the quantity being nulled is a paired PER delta at sub-epoch 20, which no probe can produce -- there is no gain to test before an arm runs; (ii) it doubles as the prior-weight control the ADD mixing needs, adding a term of comparable magnitude through the identical code path with no lexical content; (iii) it costs one GPU of an already-exclusive node, i.e. no marginal allocation. Magnitude match is measured, not assumed (`lexlat_term_mean`, reported for both arms). Disclosed limitation: with the trie unchanged the null preserves the "must decompose into dictionary pronunciations" constraint and isolates word-identity routing only; a null destroying segmentability too (random pronunciations at matched length) is named in Open decision 4 and is not funded here.

**E-1, pre-funding, CPU, about 10 min** (Step 0's whole job was 302.85 s on 4 CPUs / 16 GB). (a) The G2P question: re-run the Step 0 lexicon ESCAPE row with the trie restricted to bliss-lexicon entries only, reporting word count and gold-minus-private gap against the banked 151,731 / 2.32. (b) Equivalence: the new GPU trie DP in max-plus mode on the banked Step 0 strings must reproduce `prior_gap.best_segmentation` to 1e-4 nats per token on the ESCAPE row (gold -2.19, private -4.46). A failure of (b) stops the phase.

**E0, state-space census, GPU, about 1 h.** 100 fixed dev-other utterances at the banked ctrl_20 ep10 checkpoint; the trie DP at C in {256, 1024, 4096} and unpruned on the 20 shortest utterances, each at lam_lex in {1/3, 2/3, 1} (design review amendment 4: the ramp passes through these strengths and the pruning behaviour is read at each). Reported as the cost/quality curve the literature asks for (Shinozaki's beam-table analogue): per-frame pruned forward mass (median, p95, max), log Z = NEG_INF count and log Z per retained frame against C and against seconds per utterance. Funding rule for C = 1024, fixed now: median per-frame pruned mass < 0.05 at every lam_lex AND |log Z(1024) - log Z(4096)| < 0.05 nats per retained frame at lam_lex = 1 (about 3 % of the bed's lattice term, which sits at 1.73-1.85) AND zero NEG_INF utterances. If C = 1024 misses the rule, C is re-declared at the smallest measured value meeting it, the change recorded before launch, and E1 measured at that C. If no measured C meets the rule and also passes E1, the phase stops and reports the cost.

E0 also carries two kill reads on the same 100 utterances at ctrl_20 ep10, rules fixed now (amendment 6; these are the cheapest checks of the whole phase and run before any arm): (a) the augmented lattice's max-plus (Viterbi) decode at lam_lex = 1, C = 1024 against the plain bed's greedy decode, scored as PER against the MFA gold strings (disclosed label-using read): if the lexicon-constrained decode is NOT closer to gold than the plain decode (paired delta > -0.010), the constraint does not point toward gold at this operating point and the pack is not funded; (b) the posterior-expected phone count E[N] per utterance under the augmented lattice against the plain lattice: a drop of more than 10 % is confirmed deletion pressure from ADD's added prior weight, and the pack is not funded under ADD (the recorded fallback is the weight-preserving blend, a new pre-registration, not a silent switch).

### 7. Efficiency gate E1, BEFORE the pack is funded

Measured: seconds per step and peak GPU memory at the RUN shape (batch 88,000 padded frames, max_seqs 128, one GH200, lam_lex at full strength, C as declared by E0, parameters pinned to the banked ctrl_20 ep10 checkpoint so the posterior has the shape of a live run; amendment 8), over 100 steps sampled at four evenly spaced points of a sub-epoch under the schedule's own laplace ordering with a fixed recorded seed, plus the batch with the largest padded T of the sub-epoch, which sets the memory peak -- not the first 100 steps, since laplace sorting makes early batches short and the step rate periodic (standing user requirement, 2026-09-18).

Bar, fixed now: extrapolated **<= 1202 s per sub-epoch, i.e. <= 2.00 x the bed's 601 s**, and peak GPU memory **<= 80 GiB** against the per-arm marker of 96. Fail -> the pack is not funded at that C. Declared fallbacks, in order, each re-measured against the SAME bar: (1) C = 512; (2) the word BIGRAM CSR (151,731 states instead of about 3.45M, survey s4), which the banked word-bigram ESCAPE row prices at 0.03 nats per token of lost discrimination (2.287 against the word-trigram row's 2.321, `SAE_4A_prior.md` falsifier (i)); (3) stop and report. The bar is never moved to fit a measurement.

### 8. Code plan

New modules only; no edit to `lattice.py`, `blankfree.py`, `blankfree_train_jobs.py` or any other banked blank-free module while the prepro pack runs.

- `sae/emc/lexlat.py`: flat int32 trie (`child[node, phone]`, `is_word`, `word_id`) built from `prior_gap.Lexicon`, the CSR back-off word trigram with integer state ids replacing `kenlm.State`, the escape price vector, and the augmented forward / manual backward. Mirrors `lattice.py` throughout: same NEG_INF floor, `_exact_fp32_matmul` with TF32 off, float64 accumulation, checkpoint stride 32, manual backward on detached tables.
- `sae/emc/lexlat_jobs.py`: `LexiconTrieBuildJob` (CPU; reuses `prior_gap.load_phonemization_lexicon` / `restrict_to_word_lm` verbatim so the word set matches the banked rows, rebuilds the word trigram with the identical `lmplz -o 3 --interpolate_unigrams True --discount_fallback` call, asserts |V| = 151,731 and 3,302,936 bigram types, and prints the trie node count and the token-weighted mean pronunciation length, neither of which the survey banks); `LexlatCensusJob` (E0); `LexlatEfficiencyProbeJob` (E1). Arms go through the existing `blankfree_pack_jobs` packer unchanged.
- `sae/emc/test_lexlat.py`: (a) lam_lex = 0 with pruning off reproduces `lattice.py`'s log Z to 1e-9 in float64 on a toy shape; (b) max-plus mode reproduces `best_segmentation` on the `test_prior_gap.TINY_LEXICON` fixtures AND on 50 banked Step 0 strings through the REAL trie and the real KenLM (a fixture test cannot catch a wrong call into a shared primitive); (c) the escape price identities ported from `test_prior_gap.py:158` (one `<unk>` per contiguous span, SIL never escaped, SIL closes an escape); (d) finiteness and rows-sum-to-1 on the posterior at the production dtype and shape (float32 tables, float64 accumulation), not only in fp64; (e) the permutation is a true derangement and leaves the trie arrays bit-identical.

Compute-heavy work is sisyphus GPU jobs only, no login-node script; launches are verified by the on-disk job dir, since `add_alias` does not put a job in the graph.

### 9. Disclosed differences from the literature precedent

1. Klejch et al. 2022 compose a GRAPHEME (spelling) lexicon over a LEARNED phone-to-grapheme table; ours is a pronunciation lexicon with nothing learned on the lexicon side. Their alpha = 0.9 re-smoothing, and Nuhn and Ney's identical lambda = 0.9, exist only because a learned table can hit zero. Not carried (Design 4).
2. Their curriculum has intermediate character 5-gram stages; ours is two-stage (phone trigram -> phone trigram + lexicon). Reason: Step 0 prices phone order alone at +0.27 nats per token (4-gram) against the lexicon's +0.93.
3. Their deletion penalty sits at DECODE; our anti-deletion guard sits in training, and the read is plain greedy PER with no decode-side penalty.
4. Their training LM was the 100k most frequent words on 20 minutes of speech with 50 random restarts, and the method failed outright on 3 of 7 languages (Swahili WER 105.3, Swedish 93.5, Hausa 70.8). We run 151,731 words on the 100 h bed with one seed. This is the main extrapolation risk, it is not mitigated here, and it is why a PASS needs a second seed.
5. Nuhn and Ney's preselection budget is not carried (Design 2): the trie does the preselection. Klejch's silence-as-word-boundary anchor IS adopted, and is already what `prior_gap.py` does.
6. Yang, Barkoczi, Schlueter and Ney (arXiv 2603.02285) propose exactly this loss shape, with no ASR experiments and a simulation at |X| = 4; it is not treated as evidence.
7. The strongest published argument AGAINST: wav2vec-U 2.0 attributes the disease to the generator "consistently producing the most common n-grams" and answered it with an AUDIO-side anchor, and Ni et al. 2025 measured a word-level prior concentrating error onto rare words (error counts exceeding occurrence counts in the low-frequency bins). Hence the frequency-stratified read is pre-registered, not optional.
8. Nobody has run the trigram-only against trigram-plus-marginalised-lexicon ablation with the same acoustic model and the same DP. That ablation is this phase.

## Gate

**G4a.9** (pre-registered 2026-09-21, before any number exists; per arm, dev-other, read at sub-epoch 20, never best-PER over the kept set).

Form of G4a.4 (design review amendment 1: every cold arm of this bed sits at PER 0.82–0.91, `SAE_4A_budget.md` Results, so PER < 0.50 is part of the PASS condition, never a precondition for reading FAIL). Health clause: speaker-matched derangement gap > 0 (point estimate; the gap job emits no CI and the gap is positive in every content-free arm, so it cannot carry the decision).

Primary read: the utterance-paired PER delta (`PairedPerDeltaJob`) `lexlat_20` - `ctrl_20` at sub-epoch 20 on dev-other, plain PER, with S / D / I reported per arm at ep10 and ep20 (amendment 7). Band B = the paired delta `ctrl_20` - `ctrl_20_s1` from the prepro pack (full replicate band: init, order, sub-epoch composition). Cross-pack floor F = the common ep4 offset of the lexicon-free arms from ctrl_20 (Design 5, amendment 2). Margin M = max(|B|, |F|, 0.010 absolute PER); the 0.010 floor is fixed now so a band that comes back implausibly tight cannot manufacture a PASS.

- **PASS**: greedy PER < 0.50 AND greedy emitted rate in [5.80, 14.49] per second for `lexlat_20` AND `lexlat_20` - `ctrl_20` <= -M AND `lexlat_20` - `lexshuf_20` <= -M AND the health clause holds for `lexlat_20`. Both paired comparisons are required: beating ctrl alone is consistent with added prior weight, and the null is what separates the lexicon from the weight.
- **FAIL**: the health clause holds, the engagement monitors show the term was live, and any PASS condition is missed (including `lexlat_20` remaining at the cold band's PER). Licenses not funding the marginalised lexicon further at this bed; licenses nothing about a different pruning budget, on-set or bed.
- **CANNOT_TELL**: the health clause fails in BOTH `lexlat_20` and `ctrl_20`; or any engagement clause below fires; or E0 / E1 had to be re-declared in a way not recorded before launch; or the ep4 lexicon-free arms disagree with each other (Design 5).

Engagement clauses (label-free, per sub-epoch, in the training log), ceilings fixed now:

- `lexlat_pruned_mass` (median over utterances of forward mass discarded per frame by the C-truncation; p95 / max and the NEG_INF count are logged beside it) above 0.05 for 3 consecutive sub-epochs counted from sub-epoch 11 (full ramp; amendment 4) -> CANNOT_TELL at that budget; the next arm is a larger C, not a different conclusion.
- `lexlat_escape_phone_frac` (posterior-expected fraction of emitted phones inside an escape span) above 0.90 for 3 consecutive sub-epochs at full ramp -> the lexicon is inactive because every path escapes; UNINFORMATIVE, and the next arm raises the escape price.
- `lexlat_expected_phones_per_word` outside [0.7, 1.4] x the build job's token-weighted mean pronunciation length for 3 consecutive sub-epochs at full ramp -> the anti-deletion guard has fired; UNINFORMATIVE.
- `lexlat_term_mean` reported for `lexlat_20` and `lexshuf_20`; the magnitude match is read over sub-epochs 8–11 only (amendment 5): if the two differ by more than a factor of 2 in that window the null is not a magnitude-matched weight control, and the PASS clause's second comparison is reported as disclosed rather than decisive. Divergence after sub-epoch 11 is the expected signature of the lexicon taking hold and never demotes the null.

Abort rule, per arm, as G4a.4 and re-armed at the on-set: NaN, or |trained surrogate| > 100 in any sub-epoch, or expected phone rate < 0.6 rho for 5 consecutive sub-epochs after the on-set. An aborted arm reads FAIL (collapse) and is not restarted with a different constant.

Disclosed label-using reads at kept epochs 10 and 20, entering nothing:

- **Frequency-stratified PER.** Each gold phone inherits the word-frequency decile of its MFA word (deciles from the window text's word unigram counts, boundaries fixed before the read); substitutions and deletions attributed to the gold phone's decile, insertions to the preceding gold phone's decile; per-decile PER for `lexlat_20`, `lexshuf_20` and `ctrl_20`. This is what makes the frequent-word collapse visible in the first result.
- **Step 0 prior-gap rerun** (`PriorGapAnalysisJob` with each arm's decode as the private-code input), like-for-like pairing: the lexicon ESCAPE gap should shrink from 2.32 and the decode's strict segmentable fraction rise from 0.237 if the arm moved the code toward lexical structure.

Label-free reference read, as G4a.8: the wav2vec-U 2.0 selection statistic (4-gram phone-LM perplexity divided by the squared vocabulary-seen fraction, SIL stripped) per kept epoch.

A PASS is audited from a fresh context and needs a second seed before it is claimed (standing rule). A negative at n = 1 reads "no take-off in this seed".

## Cost

Per arm: 20 sub-epochs at the bed's 601 s = 3.34 h if the trie DP costs the bed's rate; 6.68 h at the E1 bar of 2.00x. Arms run one per GPU in one exclusive 4-GPU allocation (`blankfree_pack_jobs`: 4 arms max, per-arm cpu 16 / mem 64 GB / gpu_mem 96, `--exclusive`), so pack wall time is the longest arm: 3.3 h to 6.7 h, inside the 11.5 h clamp with no resume path.

| item | work | charged (exclusive 4-GPU node) |
|---|---|---|
| E-1 (CPU: G2P read + max-plus equivalence) | ~0.2 h CPU | 0 GPU-h |
| E0 census (C curve) | ~1 h on 1 GPU | 4 GPU-h |
| E1 efficiency probe (100 sampled steps) | ~0.5 h on 1 GPU | 2 GPU-h |
| pack of 4 arms (one exclusive node, arms in parallel) | 4 x 3.3 h .. 4 x 6.7 h of GPU work, 3.3 .. 6.7 h wall | 13.4 .. 26.7 GPU-h (the node is charged whole) |
| registered reads (PER with S/D/I, rate, gap, paired delta, prior-gap rerun, selection statistic; 4 arms x 4 kept epochs) | ~2.5 h | ~10 GPU-h |
| **phase total** | | **about 29 GPU-h at the bed's rate, about 43 GPU-h at the E1 bar** |

The second seed sits in the fourth slot (ruling 4), so a PASS needs no second round.

## Orchestrator rulings (2026-09-21, on the draft's five open points, before design review)

1. **G2P in the trie: keep the Step 0 trie verbatim** (bliss lexicon + Sequitur G2P fallback). The bed's own prior text is phonemised through the same chain, so G2P-derived pronunciations are already a text-side resource of the live trigram prior; `SAE_ref.md` "Live reward: no G2P anywhere" is the source trace of the retired grapheme scorer, not a rule for this bed. The G2P-derived subset is disclosed as a resource; E-1(a) prices a G2P-free trie as a disclosed read, not a stop.
2. **Mixing: ADD** (`beta log P3 + lam_lex x lexical increment`, lam_lex = 1, no sweep). The shuffled-pronunciation arm is the prior-weight control; a blend would remove part of the trigram, a second change.
3. **Word-LM order: trigram primary, bigram as the declared E1 fallback** (banked price 0.03 nats per token). The standing "prior order above bigram" rule holds either way because the phone trigram stays in the objective.
4. **Fourth GPU slot: the second seed of the treatment, `lexlat_20_s1`** (flat_seed 1 / random_seed 1 / random_seed_offset 1000, mirroring ctrl_20_s1), so a PASS arrives with its replicate. The segmentability-destroying null is named as a follow-up, not funded here.
5. **On-set: both arms kept** (8 primary, 5 as the one extra knob). If a slot must go, `lexlat_20_e5` goes, never `lexshuf_20` or the second seed.

## Results

### Resource constants (LexiconTrieBuildJob, first run 2026-09-21, `output/.../sae_4a_lexlat/resource/summary.txt`; the job reruns at a new hash after the review fixes, the constants are deterministic)

| constant | value | reference |
|---|---|---|
| words in the trie / distinct pronunciations / max length | 151,731 / 132,049 / 33 | survey s1, all matched |
| trie nodes | 291,476 (151,731 word ends) | new |
| mean pronunciation length, token-weighted (type-weighted) | 3.579 phones (6.486) | new; fixes the Design 4 monitor band `lexlat_expected_phones_per_word` in [2.505, 5.010] |
| word trigram | 151,734 unigrams, 3,393,577 bigrams, 10,419,405 trigrams, 3,545,312 CSR states, 274,266,991-byte binary | survey s4 (binary size matched) |
| distinct bigram types | 3,302,936 in-line (banked; with BOS 3,335,328, with BOS+EOS 3,393,577) | survey s4, matched by the in-line convention |
| shuffled-pronunciation null (seed 0, single-cycle derangement) | 0 fixed points; 1 word of 151,731 keeps a homophone of its own pronunciation; trie arrays bit-identical | Design 6 |

The monitor band is now a number written before any arm runs, as Design 4 requires.
