# Debug: i6 phone prior held-out ppl 9.6018 vs banked 9.5611 (2026-09-25)

Status: DONE (the stage and owning layer are identified; the split of the gap between corpus
composition and held-set sampling is NOT measured, see section 6).

## 1. Verdict

The prior code, the sampling code and the held-out split are not the cause. The first departure is
upstream, in the phonemised corpus, at the g2p / OOV stage:

- JUPITER `PhonemizeWithSilJob.DbFgvZOGZQ8F`: 40,418,261 lines in, 39,630,169 out, **788,092 lines
  dropped** because they held a word that neither the bliss lexicon nor the g2p lexicon covered.
- i6 `PhonemizeWithSilJob.NpoY1pGJWNUJ`: 40,418,261 in, 40,418,258 out, **3 dropped** (all three
  contain the word `HHH`, the one OOV word whose g2p pronunciation came out empty).

Because `SampleLinesJob` draws `random.Random(0).sample(range(n_in), 1_010_000)`, a different
`n_in` (39,630,169 vs 40,418,258) gives a completely different 1,010,000-line window. The i6 prior
is therefore fitted, and scored, on different lines, taken from a corpus with 2 % more lines
(3.4 % more tokens). Those extra lines are long and contain rare words.

Owning layer: **dataset / text serialisation, i.e. the reference's g2p output**. It is not model
code, the launcher or the prior port. The i6 chain does what the code intends (every OOV word gets
a g2p pronunciation). The JUPITER chain lost about half of its g2p coverage for a reason that is
not recorded in the frozen logs. The banked corpus cannot be regenerated from the public inputs
with this code. It can only be reproduced by importing a JUPITER artefact (section 8).

A separate, smaller defect in the i6 environment is in section 4: Sequitur training under numpy 2
aborts two of its ramp-ups early.

## 2. Stage-by-stage comparison

| stage | code / params: JUPITER vs i6 | banked JUPITER | i6 observed |
|---|---|---|---|
| LM text | `DownloadJob.g4jClO48cAvP`, same hash | 40,418,261 lines | 40,418,261 lines (stats lines_in) |
| bliss lexicon | `MergeLexiconJob.qKaOAPqURCkK`, same hash (JUPITER path in `exp_logs/SAE/reports/impl_prior_gap_2026-09-20.md:38`) | 200,000 words (`review_prior_gap_2026-09-20.md:39`) | 200,003 lemmas, 3 special -> 200,000 |
| g2p training | same wiring (`posterior_hmm/data/phon_lm.py:249-257` == `lm/lexicon.py:119-127`: BlissLexiconToG2PLexiconJob with no variants + TrainG2PModelJob defaults) | JUPITER train job hash and error rates not banked | `TrainG2PModelJob.pD4nbqFLWtbi`: ramp-ups 2 and 3 aborted (section 4); final dev symbol error 5.53 %, string error 22.79 % |
| OOV collection | identical code (`phon_lm.py:145-176` == `lm/lexicon.py:85-116`) | num_oov not banked | 773,673 OOV types |
| g2p apply | identical: `ApplyG2PModelJob(filter_empty_words=True, concurrent=16)`; JUPITER `myTIGtmrUIFq`, i6 `3eJqzOadjOqw` (the hash differs through the moved CollectOovWordsJob module) | bliss + g2p = **584,893 words** (`review_prior_gap_2026-09-20.md:40`, `survey_lexicon_scorer_2026-09-20.md:41`) -> **about 384,893 g2p words (inferred)** | **773,672 g2p words**; the only missing type is `HHH`; the stderr files hold no "failed to convert" |
| phonemise + SIL | identical `phonemize_line` (old `w2vu2/text.py:38-55` == `lm/phone_text.py:56-73`); p 0.5, surround, RandomState(0), one random_sample(W-1) per KEPT line | lines out 39,630,169; tokens 3,232,620,004; SIL 448,460,735 (rate 0.138730); non-SIL 2,784,159,269 | lines out 40,418,258; tokens 3,342,048,858; SIL 462,273,856 (rate 0.138320); non-SIL 2,879,775,002 |
| sample | identical algorithm; n_out 1,010,000, seed 0; the draw depends on n_in | `orN768ARKwlt`; initial phones DH 0.167 / HH 0.130 / IH 0.083 / AY 0.083 / AE 0.062 | `CrPgeKXsOosb`; DH 0.1677 / HH 0.1304 / IH 0.0829 / AY 0.0821 / AE 0.0615 |
| prior fit | identical: 1,000,000 counted + 10,000 held, `i % 101 == 0` held, interpolated WB, BOS, no EOS. The WB fit and the ppl are unit-tested against an independent oracle (`tests/wb_oracle.py`) | lines_counted 1,000,000, lines_held 10,000, stride 101; tokens_counted 81,559,944; held 808,146 raw tokens | same counts; tokens_counted 82,740,447 (I recounted it exactly with the `i % 101` rule); held 820,584 tokens |
| held ppl order 3 | | **9.561056** | **9.601838** (+0.0408; +0.00426 nats, +0.0061 bits per token) |

Arithmetic from the banked counts:
- Dropped lines: 788,092 of 40,418,261 (1.95 %).
- The lines that i6 keeps and JUPITER dropped: 788,089 lines carrying 109,428,854 tokens, i.e.
  **138.9 tokens per line** against 81.6 for the corpus.
- The SIL rate is lower in i6 because long rare words dilute the SIL tokens.
- The window's tokens per line rise by 1.45 %, in line with the full-corpus ratio (+1.37 %).

## 3. Why the JUPITER g2p lexicon is short (what is observed, what is inferred)

Observed:
- JUPITER dropped 788,092 lines as OOV (`exp_logs/SAE/reports/extract_sil_rate_full_2026-09-21.md:47-49`).
- `SAE_0.md:38-40`: "base lexicon covers 99.75 % of word tokens, residual unresolved types dropped
  ... (~0.1 % of tokens)".
- The phonemiser drops a line only if a word is missing from both bliss and g2p. The bliss lexicon
  and the text are the same jobs on both clusters.

Inferred: bliss has 200,000 words and the loaded union has 584,893. So the JUPITER g2p lexicon
held about 384,893 of the 773,673 OOV types, about 50 %. This inference depends on the JUPITER
OOV list equalling i6's. It should, because the code, the text and the lexicon are identical.

Not determinable from the logs: why the JUPITER g2p lost those words. Candidates are translation
failures in a differently built Sequitur (the port's own install notes say a SWIG >= 4.2 wrapper
is broken) or partially written run parts. Neither can be checked here, because `/e/scratch` is
not mounted.

## 4. i6 environment defect: Sequitur under numpy 2.4.6 (observed)

The log `work/i6_core/g2p/train/TrainG2PModelJob.pD4nbqFLWtbi/log.run.1` holds two tracebacks:
`AttributeError: np.sometrue was removed in the NumPy 2.0 release`. They are raised at
`sequitur.py:431` (`adjustHigherOrder`), inside the env's
`/work/asr4/hwu/conda/envs/sae/lib/python3.11/site-packages/sequitur.py`.

- Sequitur's `run()` (`sequitur.py:735-742`) catches the exception, prints "iteration failed." and
  breaks out of the loop. As a result, the ramp-up 2 log (`work/stdout.2`) stops at iteration 41
  and the ramp-up 3 log (`stdout.3`) at iteration 25.
- The job still reports "finished successfully".
- The dev symbol error per ramp-up is 43.74 / 18.84 / 10.70 / 6.69 / 5.53 %.

What this touches: only the pronunciations of the g2p'd words, not coverage. The i6 model
translated all words but one. The effect on the prior is expected to be small, because g2p'd
words are about 0.25 % of word tokens, but it is unmeasured. The reference README pins numpy
2.4.6 for the reference env too. Whether JUPITER's g2p job ran in that env is not recorded.

Upstream: sequitur-g2p master still calls `num.sometrue` in `adjustHigherOrder`. The GitHub
tracker has no issue matching "sometrue" (searched 2026-09-25). NumPy 2.0 removed `np.sometrue`
(migration guide).

Fix (environment layer): replace `num.sometrue` with `num.any` in the env's sequitur.py, or run
g2p with numpy < 2. Then clear and rerun `TrainG2PModelJob.pD4nbqFLWtbi`. Its hash does not
change, and every job down to the prior reruns.

## 5. Q2: the precise banked value

The banked value is `held_ppl_order3 = 9.561056` (`held_bits_per_phone_order3 = 3.2572`), from
`PhoneNgramPriorJob.RtzbESkOedsT/output/prior.stats.txt`, cited in
`exp_logs/SAE/reports/review_neural_phone_lm_2026-09-20.md:71-76`. It uses the same held-out
definition: 1,010,000-line window, `i % 101 == 0` held, 1,000,000 counted / 10,000 held
(ibid. :21-24). The held set is 808,146 raw tokens (`review_phone_lm_v2_2026-09-20.md:17`). It was
independently reproduced as 9.557 on the 512-truncated held set (`SAE_4A_prior.md:368`). "9.56" is
this value rounded.

## 6. Q4: size of the effect (not apportioned)

The whole gap is +0.0408 ppl (+0.00426 nats per token). It has two parts that the logs cannot
separate:
(a) Composition. The i6 corpus adds long rare-word lines, about 3.3 % of held tokens (about 197
held lines). If they alone explained the gap, they would have to cost about 0.13 nats per token
more than the other lines. That is plausible, and the direction (higher ppl) is certain.
(b) Held-set sampling. The i6 held lines are a different random 10,000 (line-length sd 66 tokens).
My unmeasured estimate of one held set's ppl SE is about 0.01-0.03. That is the same size as the
+-0.02 tolerance, which presumes an identical window.

## 7. Q3: rho

ctrl_20 `ReturnnTrainingJob.GiT88bxzoZbZ/output/returnn.config:51` sets
`"rate_rho_hz": 9.6619373279`, which equals the banked value. It is **hard-coded** at
`training/config.py:316` and not computed from this phone text.

Provenance of the literal: 2.7 x 2,784,159,269 / 778,025,128 (JUPITER T_phi).

On the i6 corpus the same definition would give **about 9.6794** (+0.18 %). This uses the word
estimate lines + 2 x internal SIL, which recovers JUPITER's 778,025,128 words as 778,030,963. So
ctrl_20 keeps the banked rho, and rho is no longer consistent with the i6 text. That is harmless
for reproduction.

`tests/test_artefacts.py:185-198` (lines_out == 39,630,169; non-SIL == 2,784,159,269) and
`:153-175` (ppl rounds to 9.561) will FAIL on the i6 artefacts. These are the pre-registered checks
that catch this.

## 8. Step-1 triple

i6 step 0 (`log.run.1:670`) against the banked values:
- l_tau: -0.352 vs -0.350.
- prior per token: -5.637 vs -5.657 (+0.020; outside +-0.01).
- expected tokens: 63.854 vs 63.821.

The prior-per-token shift is in the direction a flatter prior predicts. Under near-uniform strings
at tau 8, a higher-entropy prior scores them higher.

It is confounded, though. Under the audio label, the units and eta that enter G come from
different audio, and the banked `ctrl_20_s1` batch alone moves this value by 0.014. So the part
owed to the prior is not isolated.

## 9. Tests that would decide (specified, NOT run)

T0, cheapest and decisive, if JUPITER is still readable. Copy
`work/i6_core/g2p/apply/ApplyG2PModelJob.myTIGtmrUIFq/output/g2p.lexicon` and
`work/speech_llm/sae/emc/text_sample/SampleLinesJob.orN768ARKwlt/output/text.phn.gz` (about
50 MB), plus `PhoneNgramPriorJob.RtzbESkOedsT/output/prior.npz` (0.6 MB). Then:
(i) `cut -f1 | uniq | wc -l` on the g2p lexicon should give about 384,893 (confirms section 3);
(ii) `lm.phone_prior.fit_prior(<JUPITER window>)` should give 9.561056 exactly (confirms the port
code on the reference input);
(iii) feeding that g2p lexicon to the i6 `PhonemizeWithSilJob` should give lines_out 39,630,169.

T1, if JUPITER is unreadable (CPU, minutes, executor). Run
`SampleLinesJob(corpus=NpoY1pGJWNUJ text, n_out=1_010_000, seed=s)` + `PhoneNgramPriorJob` for
s = 1..4. The spread of held_ppl_order3 across windows says what +-0.02 means for a non-identical
window. A cheaper variant: bootstrap the 10,000 held lines (2000 resamples, seed 0) with the i6
prior.npz for the per-line SE.

T2 (composition bracket, CPU, about 15 min). Replay the words of the i6 window: kept lines = LM
lines without `HHH`, then the same seed-0 sample indices. Refit and rescore after dropping every
line that contains a word outside bliss. JUPITER dropped roughly half of those lines, so the
bliss-only ppl brackets the JUPITER-like value.

T3 (step 1). A one-step run of ctrl_20 with only prior_npz swapped (the JUPITER prior from T0, or
the T2 prior) isolates the prior's share of the -5.637 vs -5.657 gap.

## 10. What a fix would have to change

- To reproduce the banked prior exactly: pin an imported JUPITER artefact in the graph. Either
  replace the g2p lexicon returned by `lm/lexicon.py:151-154` (`lm_corpus_lexicon_and_g2p`) with
  the imported `myTIGtmrUIFq` g2p.lexicon, or pin `prior_npz` at `inputs.py:179`
  (`get_phone_prior()`) to the imported `RtzbESkOedsT/prior.npz`. Either one changes the ctrl_20
  hash, and ctrl_20, currently training on the i6 prior, would need a rerun.
- Otherwise: disclose the deviation. The i6 prior is fitted on the complete phonemisation, and the
  window and held set differ. rho stays as banked. Use T1 to decide whether the 0.041 ppl gap is
  distinguishable from window-sampling noise.
- Independently: the Sequitur numpy-2 patch (section 4). It changes only the g2p'd pronunciations
  and is optional for reproduction.

Sources: https://github.com/sequitur-g2p/sequitur-g2p (master sequitur.py, issues search
"sometrue"); https://numpy.org/doc/stable/numpy_2_0_migration_guide.html
