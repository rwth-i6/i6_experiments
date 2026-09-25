# SAE_i6 reference — setup, protocol, standing results and constraints

The shared basis of SAE_i6: what the method is, how it is run and read, which numbers a port must
reproduce, and what the JUPITER campaign (phase 4A of SAE, 2026-09-15 to 2026-09-24) established.
Details live in the topic files below; each fact is kept in one place. `(src: ...)` tags in those
files point to the frozen JUPITER logs in `recipe/i6_experiments/users/wu/exp_logs/SAE/` (provenance
only, jpt-specific; never needed to use these files).

| file | content |
|---|---|
| `SAE_i6_ref_objective.md` | the mathematics: the reverse-KL bound, the implemented loss (lattice, rate, aggregate terms), why tau ends at 2, the content-free stationary point and the private code ("the objective note") |
| `SAE_i6_ref_blankfree.md` | **the current bed** (blank-free, rVAD, stride 3; `ctrl_20`) with every constant, evaluation reads, defects; its cold rounds: first model, attribution 2x2, budget N = 50/100 and the N = 20 decision, InfoMax, waveform cut, context-dependent reverse model (deferred) |
| `SAE_i6_ref_lexicon.md` | prior strength (prior-gap diagnostic, neural phone LM, score-function and soft arms) and the lexicon inside the lattice: DP form (abandoned on cost), the k2 HLG second graph (`k2lat_*` presets), gate G4a.9, diagnoses D0-D18, E60 |
| `SAE_i6_ref_lexlat_v2.md` | the last phase before the move: phi-first EM decipherment, competence ladder L2-0, basin and key reads, **open experiments at the move** |
| `SAE_i6_ref_emc1.md` | history: the original 50 Hz CTC-blank cycle, round 1 (S1-S3b, S2d; degradation investigation; BT, content, consistency, rate remedies) |
| `SAE_i6_ref_emc2.md` | history: round 2 (seeded refinement S2e/S2f/S2g that works; S3c/S3d; float32 defect and float64 DP; six-gram prior and M512 path training) |

## 1. Problem and north star

Unsupervised phone recognition on LibriSpeech from unpaired speech (train-clean-100, no transcripts)
and unpaired text (the LibriSpeech LM corpus), with frozen self-supervised features (wav2vec 2.0
large-lv60, layer 15). The approach is a **cycle**: a recognizer theta proposes phone strings, a
frozen text prior scores them, a reverse model phi explains the acoustic units from them, and theta
and phi are trained jointly through an exact marginal over alignments (the "exact-marginal cycle",
EMC). User priority since 2026-09-16: **pure unsupervised ASR without GANs, improving cold start
within the cycle**. Unsupervised MT is a source of ideas for the cycle, not a replacement pipeline.
The reference point the cycle has not reached: wav2vec-U 2.0 (GAN, same recognizer shape) at
dev-other PER 0.214 (four seeds 0.168-0.215) on the same features.

## 2. Standing constraints

- **Label quarantine.** Transcripts, MFA alignments and the 10 h labelled seed appear only in
  evaluation (PER/WER) and in disclosed, analysis-only diagnostics. No label enters training,
  initialization, checkpoint selection, hyperparameter choice or a gate of a main-line arm.
  Supervised inits (p0, gold phi, corrupted-gold phis) are analysis only, never a route or fallback.
  Speaker IDs may train (disclosed supervision cost); transcripts and alignments never.
- **No GAN** component, GAN-derived init, pseudo-labels or teacher in main-line cycle arms.
- **Checkpoint rule.** Gates read the final sub-epoch or a label-free-selected checkpoint, never
  best-PER. The wav2vec-U 2.0 selector (4-gram phone-LM perplexity / squared vocabulary-seen
  fraction, SIL stripped) was never registered on a blank-free pack.
- **Paired evaluation.** Every model comparison is paired per utterance on the same items with a
  speaker-clustered bootstrap (2000 resamples, seed 0); never two pooled numbers.
- **One delta per arm** against a control in the same pack; each new pack carries its own `ctrl_20`.
- **Every n-gram** (prior, coverage target, text statistic, selection LM) is fit on the seeded
  uniform 1,010,000-line sample of the LM corpus, never the alphabetical head (defect in
  `SAE_i6_ref_blankfree.md` section 1).
- **Budget:** N = 20 sub-epochs for new arms (schedule in section 3); E60 showed nothing moves by 60.
- **Cost screens** use uniformly sampled real training batches across the schedule (fixed seed) plus
  the longest batch; never first-batch extrapolation (a 19 h projection became > 11.5 h per sub-epoch).
- **Numerics:** the lattice DP runs in float64 in arm and control alike; posteriors are never
  renormalised to pass a conservation check.
- Gates are pre-registered before results, never moved after; a positive derangement gap, a lower
  loss or a fitted relabelling is not evidence of phone learning.
- **Topology givens (user):** the reverse model's minimum duration `d_min >= 2` is standing and not
  revisitable; no cycle stage launches with a bigram prior (trigram at least).
- Closed or vetoed by the user (do not reopen without the user): SylCipher-style standalone
  initializer, K4/few-string candidate training, the context-dependent reverse model (deferred without
  limit). Design constraints of the last phase (lexlat v2; revisitable with a reason): no further full
  joint cold restarts of the existing recipe (every seed sits in the band), no recognizer-neutral
  alpha = 0 cold arm, no neural-refit split-merge search.

## 3. The bed and its reads (summary; constants in `SAE_i6_ref_blankfree.md` section 1)

- **Data.** train-clean-100 (28,539 utterances; 28,254 train + CV holdout), rVADfast mask applied to
  features and units after full-waveform extraction (retained train frames 15,427,853 of 18,088,388).
  Units: PCA-96 + k-means K = 500 on 50 Hz L15 features. Speaker vector eta: 16-dim projection of the
  utterance-mean features. Text: phonemised LM corpus, SIL at word boundaries p = 0.5 plus sentence
  edges; Witten-Bell phone trigram, BOS, no EOS, held-out ppl 9.56. Rate target rho = 9.6619 phones
  per original second.
- **Models.** theta: wav2vec-U 2.0 generator shape (BN, dropout 0.1, residual linear, one conv k 9
  stride 3), 40 outputs (39 phones + SIL), no blank, flat init (zero logits); strings are the run
  collapse. phi: segmental HSMM, d in [2, 25] (SIL 50), emission over 500 units conditioned on type,
  duration bucket, position bucket and eta; random init.
- **Loss.** l_tau (exact banded lattice, |s - 3t| <= 25, float64) + 3 x rate + 0.1 x agg; tau geometric
  8 -> 2 over the first 20 % of sub-epochs then 2; Adam (0.5, 0.98), clip 5, theta lr 1e-4, phi 3e-3;
  88,000 padded frames / 128 seqs per batch; a sub-epoch is a quarter of train-clean-100 (57 updates).
  `ctrl_20`: LR warmup to sub-epoch 2, hold to 12, linear decay to 0.1x at 20; kept 1/4/10/20.
- **Reads (dev-other, 2864 utterances, 33 speakers, 177,275 reference phones).** Greedy PER: frame
  argmax, collapse repeats including SIL, drop SIL without a second collapse, against MFA phones.
  Paired delta (corpus-ratio, speaker-clustered CI). Derangement gap (own vs speaker-matched donor
  decode under the arm's phi; positive in every content-free arm, does not discriminate). Chance
  null (length- and unigram-matched strings): the **content-free band is PER 0.83-0.91**; band exit
  needs PER more than 0.05 below the arm's own null. JS rows (n-gram JSD of decodes to the text;
  gold JSD4 about 0.25, GAN 0.28, cold arms 0.69-0.79).
- **Noise floor.** GPU training is not bit-reproducible: identical cold configs differ by 0.01-0.03 PER
  by sub-epoch 4 (0.013-0.020 measured); ctrl_20 vs its seed replicate differ by 0.001 at sub-epoch 20.
  Any ep4 tolerance tighter than about 0.015 cannot hold.

## 4. Banked numbers a port must reproduce (JUPITER, one GH200 96 GB GPU per arm)

| run (package entry point) | read | value |
|---|---|---|
| `ctrl_20` (`config/base.py`) | dev-other greedy PER at sub-epochs 1 / 4 / 10 / 20 | 0.855 / 0.875 / 0.869 / **0.874568** |
| `ctrl_20` step 1 | l_tau / prior per token / expected tokens | -0.350 / -5.657 / 63.821 |
| `ctrl_20_x60` | PER at 60 (banked run resumed from 20; port runs one 60-sub-epoch job) | 0.873490 |
| `k2lat_20_ma3000` (`config/k2_word_lm.py` preset) | PER at 20; paired vs ctrl_20 | **0.818615**; -0.0560 [-0.0628, -0.0499] |
| `k2lat_20_ma3000_x60` | PER at 60 | 0.823991 |
| `off4_k2lat_20` | paired vs ctrl_20 at 20 | -0.0399 [-0.0464, -0.0340] |
| supervised reverse init, gold phi (`config/supervised_init.py`, analysis only) | held-out NLL per frame, epoch 8 (banked value from the package README; not in the JUPITER logs) | 3.2888 |
| cost, `ctrl_20` | seconds per sub-epoch | about 601-610 |
| cost, `k2lat_20_ma3000` | seconds per sub-epoch; whole-step peak | 760.6; about 32 GiB (rung 3000) |

The package's default k2 arm `k2_word_lm` (official 4-gram HLG, max_active 1000, phone trigram
`rampout`) was never run and has no banked number.
Decision (user, 2026-09-25): k2 arms keep the phone trigram at full weight by default; `rampout` runs only as an
explicit single-delta arm. Reason: the trigram in `l_tau` is the only text signal that reaches phi. The k2 term
trains theta's emissions alone, through a pruned numerator. JUPITER D15: warm rampout minus full is +0.0372 at ep8;
cold it is −0.0012, inside the 0.013-0.020 spread of identical cold configs. At trigram weight 0 the objective
prefers the worse-PER arm (D18) (`SAE_i6_ref_lexicon.md`, D15 and D18). The package default changes after the P0
trainings end (`SAE_i6_P0.md`, State). Checked the same day: the running P0 `k2lat_20_ma3000` and the written P1 rt
arm configs carry `prior_weight: 1.0` and no weight schedule, so no k2 arm needed a restart.

## 5. What the campaign established (one line each; evidence in the topic files)

1. **No cold arm has left the content-free band** (PER 0.81-0.94 over every cold arm of both beds):
   rate, consistency, entropy, invariance, content auxiliaries, BT, coverage terms, six-gram prior,
   budget up to 67 sub-epochs, waveform cut, prior window, neural phone LM scorers (`_blankfree`,
   `_emc1`, `_emc2`, `_lexicon`).
2. The cold end state is a **confident, input-dependent private code** shared by theta and phi, not a
   diffuse stationary point and not a relabelling of phones (`_blankfree` section 7, objective note 7).
3. A marginalised word graph (k2 HLG) lowers cold PER by 0.03-0.06, but a **deranged lexicon does as
   well or better**: on the cold line the gain is prior weight, not lexical content (G4a.9, `_lexicon`).
4. From a phone-like start the same k2 term is strongly lexicon-specific and preserves/refines it; a
   competent phi anchors a random theta (L2-0: up to 70 % label noise still lifts to about gold level).
5. The objective ranks the phonetic solution lower than the cold end point (D13, A14 (ii)), and the
   two are co-adapted basins (D16): **the cold failure is one of search**, plus objective drift inside
   the phonetic basin (EM degrades a gold phi from PER 0.193 to 0.353).
6. Random-init phi EM finds mislabelled, merged partitions with real sub-phone-level content that
   chance-level decode PER hides (A15); the held-out objective S prefers them to gold (A10).
7. Seeded refinement works with the stabilized recipe (fixed tau 2, seed KL, no rate term, float64
   DP): S2f U_joint and S2g refine a 10 h seed (best dev-other WER 24.91 vs 26.50) — disclosed, not
   cold progress (`_emc2`).

## 6. Known defects and traps (fixed unless marked)

- float32 lattice breaks posterior conservation at the flat init -> float64 DP (`_emc2` section 5).
- Alphabetical prior window -> seeded uniform sample (`_blankfree` section 1).
- EMA-damped KL-form aggregate gradient (0.01x) and EMA-biased coverage reads (`_blankfree`).
- k2: int32 arc overflow on flat/high-tau posteriors (score one utterance per call, chunked);
  epsilon back-off `#0` loops on every state over-count -> word-boundary loops only; determinization
  is tropical-only and must not be used (`_lexicon` B4-B5, `_lexlat_v2`).
- CV-holdout overlap: 25 of 285 CV-holdout utterances lie in every ladder phi's fit set (`_lexlat_v2`).
- Repeats: a true adjacent repeat can only surface as X SIL X (PER floor about 0.56 %); 533 lexicon
  words with internal repeats are unreachable in k2 (`_lexlat_v2`).
- ffmpeg build changes the audio without changing hashes -> `FfmpegPinCheckJob` (package README);
  the reference list is from an aarch64 build.
- RETURNN resume under torch >= 2.6 needs `weights_only=False` (shipped patch).
- `PhoneNgramPrior.per_token_log_probs(order=3)` double-counts log P(y0 | BOS, BOS) for one-token
  strings (`model/prior.py:246`): affects only the order-3 held-out perplexity statistic and the
  prior-gap reader, not training (the lattice reads the tables). Pinned by P0 tests.
- The matmul reduction `_logmm` floors impossible products at about -708 instead of NEG_INF
  (`model/lattice.py:673-674`): an infeasible utterance gets a finite log Z and loss instead of z_zero.
  On the bed the only infeasible case (S = 1) is caught elsewhere. Pinned by P0 tests.
- SIL runs may split into several SIL tokens in the training lattice (objective note section 10).
- JUPITER's phone text is truncated (NOT fixed in the banked numbers). Its g2p lexicon
  (`ApplyG2PModelJob.myTIGtmrUIFq`) lacks every non-bliss word from DITCHLIKE to RIVAW: 388,780 of
  773,673 types, because g2p chunks 5-12 of 16 are empty. Sisyphus before d9e1ede (upstream PR #314)
  re-runs finished local tasks, and the reruns truncated the chunks just before the merge.
  `PhonemizeWithSilJob` then dropped the 788,091 lines holding such a word.
  - So the banked prior ppl 9.561, the HLG size and the trie word set belong to the truncated text, and
    so does rho 9.6619. The full text gives rho about 9.679. The literal is kept: every arm and its control
    share it.
  - The i6 text is complete, and the i6 Sisyphus (a567fa7) contains the fix. Any multi-task local job run
    under an older Sisyphus can lose output the same way; the package README still pins ddcd028.
  - Source: the JUPITER orchestrator's review (`reports/jupiter_port_review_2026-09-25.md`), where port
    test T7 reproduced JUPITER's prior.stats.txt on JUPITER's window, and `SAE_i6_P0.md`, G0.R0.

## 7. i6 port

The package README lists pins, entry points and the port's own limits. Deviations of the i6 runs
from the reference (hardware, audio generation, batch shape, env versions) are recorded in the
phase file that introduces them (`SAE_i6_P0.md` for the port).
