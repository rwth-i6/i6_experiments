# Audit: i6 phone prior held-out ppl 9.6018 vs banked 9.5611, and the debugger's attribution (2026-09-25)

Status: CANNOT_TELL. Fresh-context, read-only audit. Nothing was edited or rerun. The only computations
were token counts read off the existing sample files.

## 1. Verdict

The claim under audit (`reports/debug_prior_ppl_2026-09-25.md`) has two parts:
(A) the i6 text-prior pipeline computes what the reference computes;
(B) the ppl difference is caused ONLY by corpus composition.

- The gap is NOT window noise. This is supported: JUPITER lies outside the 95 % prediction interval of five
  i6 windows (p about 0.015).
- Corpus composition is a sufficient and likely main cause. This is supported: T2a shows a large composition
  effect in the right direction, and the emulations reproduce JUPITER's window token counts.
- "Only composition" is UNDETERMINED. T2b (n = 3) cannot tell JUPITER's value from the i6 value. Its
  uniform-loss premise is also contradicted by JUPITER's banked line count. A residual port-side effect about
  the size of the tolerance (+-0.02) is not excluded.
- (A) holds for the code and the parameters: the phonemiser, the lexicon loaders, OOV collection and the g2p
  wiring are verified against git. (A) is FALSE for the output of the g2p-apply stage: the same code and
  parameters gave 773,672 g2p words on i6 against about 384,893 on JUPITER. Calling JUPITER's side the defect
  is an inference; the cause is not in any log.
- G0.R0 prior clause, as pre-registered: FAIL (Tier A). The attribution explains the miss; it does not
  convert it. All five i6 windows (9.588 to 9.608) fall outside [9.54, 9.58], so the miss is systematic for
  the i6 corpus, not bad luck.

## 2. Numbers re-derived (sources)

- i6 prior: `work/i6_experiments/users/wu/experiments/unsupervised_asr/lm/phone_prior/PhoneNgramPriorJob.qJxXHgXLe31S/output/prior.json`
  gives held_ppl_order3 9.601837932100644, tokens_counted 82,740,447, SIL rate 0.1383042.
  Its info file: n_count 1,000,000, n_held 10,000, corpus `SampleLinesJob.CrPgeKXsOosb`.
- i6 sample stats: n_in 40,418,258, n_out 1,010,000, seed 0.
- i6 phonemiser (`PhonemizeWithSilJob.NpoY1pGJWNUJ`): stats lines_in 40,418,261, lines_out 40,418,258,
  tokens 3,342,048,858, SIL rate 0.138320. Info: sil_prob 0.5, surround True, seed 0, max_lines None.
  The 3 dropped lines are the empty line 0 plus 2 lines that contain `HHH`. The debugger's "all three contain
  HHH" is a minor error.
- i6 g2p: `CollectOovWordsJob.q0KCWART4B2t` num_oov 773,673. `ApplyG2PModelJob.3eJqzOadjOqw/output/g2p.lexicon`
  has 773,672 entries, all 4-field. `g2p.untranslated` holds only 16 "stack usage" lines.
  `TrainG2PModelJob.pD4nbqFLWtbi/log.run.1` holds the `np.sometrue` tracebacks (4 hits).
- JUPITER banked:
  - held ppl3 9.561056 (`exp_logs/SAE/reports/review_neural_phone_lm_2026-09-20.md:72`);
  - tokens_counted 81,559,944 (ibid. :105);
  - held 808,146 raw tokens (`review_phone_lm_v2_2026-09-20.md:17`);
  - corpus 39,630,169 of 40,418,261 lines, tokens 3,232,620,004, SIL 0.138730 (`extract_sil_rate_full_2026-09-21.md`);
  - bliss+g2p union 584,893 words, longest pronunciation 56 phones (`review_prior_gap_2026-09-20.md:40`).
- Analysis output: `/work/asr4/hwu/setups/librispeech-960/2026-09-24-unsupervised/analysis_out/prior_window_spread/prior_window_spread.{txt,json}`.
  Slurm 4349993 COMPLETED in 37 min.

| case | ppl3 | counted tokens | held tokens (my count) | window SIL rate | lines kept |
|---|---|---|---|---|---|
| JUPITER | 9.561056 | 81,559,944 | 808,146 | corpus 0.138730 | 39,630,169 |
| T1 s0..s4 | 9.6018 / 9.6005 / 9.5882 / 9.5909 / 9.6080 | 82.69-82.75 M | 820,584 / 826,675 / 825,723 / 830,916 / 824,576 | 0.13830-0.13834 | 40,418,258 |
| T2a bliss-only | 9.4988 | 80,733,859 | 812,800 | 0.139012 | 38,950,058 |
| T2b emu0 | 9.5785 | 81,555,539 | 808,857 | 0.138730 | 39,618,280 |
| T2b emu1 | 9.6010 | 81,327,695 | 816,483 | 0.138821 | 39,619,760 |
| T2b emu2 | 9.5592 | 81,575,614 | 817,830 | 0.138702 | 39,620,100 |

I recounted the held and counted tokens from `samples/*.phn.gz` (lines with i % 101 == 0 are held).
T1_seed0 reproduces the job's 82,740,447 and the debugger's 820,584.

## 3. Is the analysis valid?

- Reuse of the job code: yes. `analysis/prior_gap/prior_window_spread.py` calls the package's own functions:
  - `phone_text.sample_line_indices` and `iter_sampled`, followed by SampleLinesJob's write loop;
  - `phone_prior.fit_prior`;
  - `lexicon._load_lexicon_word_to_phon` and `_load_g2p_lexicon`, merged with setdefault.

  It reads its parameters from the jobs' info files, and it asserts the input chain and the line totals. For
  a filtered corpus it passes the stream of kept lines with n_in = the kept count, which equals SampleLinesJob
  run on such a file. The `reduceat` drop mask is correct, because every line that holds a non-bliss word has
  at least one type id.
- The seed-0 self-check (rel diff 0, sample sha256 identical) proves that the harness equals the i6 jobs. It
  is circular for (A): it compares the port with itself, not with JUPITER.
- What T1 licenses: the window noise of held ppl3 is SD 0.0082 (n = 5; 95 % CI for the SD about 0.005 to
  0.024). The 95 % prediction interval for a new window is 9.573 to 9.623. JUPITER's 9.5611 lies outside it
  (t = 4.1, df 4, p about 0.015). "4.5 SD" overstates the certainty, because n = 5; the conclusion "not window
  noise" still holds at about p = 0.015. The +-0.02 tolerance is about 2.4 window SDs, so it presumes a window
  of the same corpus.
- What T2b licenses: a mean of 9.5796 with SD 0.0209 (n = 3); the 95 % prediction interval is +-0.10.
  - JUPITER is 0.8 prediction-SDs from the T2b mean, but so is the i6 value (9.6018, +0.022).
  - T2b mean against T1 mean: Welch t = 1.45, not significant.
  - So T2b shows that a JUPITER-like drop CAN give 9.56 (emu2). It does not show that the emulated drop
    shifts the ppl, and it does not bound any residual.
  - The P0 reading "which types JUPITER lost moves the ppl by about 0.02" rests on n = 3 with a mis-specified
    emulation (next bullet).
- T2b fidelity: contradicted on one banked number.
  - JUPITER lost at least 388,780 types (773,673 - 384,893, inferred) and 788,092 lines.
  - The emulation drops FEWER types (386,836) yet MORE lines: 798,158 to 799,978, a spread of SD about 970
    lines. That is about 11 SDs away.
  - So JUPITER's loss was not a uniform random half of the types. It fell on types that occur in fewer lines.
  - A weak hint of length-dependent loss: the i6 g2p has 5 words with pronunciations over 56 phones (max 66),
    and JUPITER's union had none. Under uniform loss the chance of that is 1/32. The pronunciations come from
    different g2p models, so this is only a hint.
  - The g2p pronunciations of the retained words (i6's numpy-2-truncated model against JUPITER's unknown one)
    are not emulated. Their share of tokens is small.
- T2b corroboration:
  - emu0 and emu2 match JUPITER's window counted tokens (within 0.02 %).
  - emu0 matches JUPITER's held tokens (808,857 against 808,146).
  - emu0 and emu2 match JUPITER's corpus SIL rate (within 3e-5). The window SIL rate tracks the corpus rate
    to about 2e-5 in T1.
  - The T1 windows miss all three: counted tokens by -50 SD, held tokens by -4.7 SD, SIL rate by 4e-4.
  - This ties JUPITER's window to a JUPITER-like composition. Post hoc, the two composition-matched
    emulations average 9.569. This is not a test.

## 4. Port-side differences looked for

- Phonemiser: `phonemize_line` and `PhonemizeWithSilJob.run` are byte-identical to
  `git show d0c7d1e0e^:.../unsupervised_asr/w2vu2/text.py`.
- Lexicon side: the loaders, `CollectOovWordsJob` and `_train_g2p_model` are identical to
  `d0c7d1e0e^:.../posterior_hmm/data/phon_lm.py`. The ApplyG2PModelJob wiring (filter_empty_words=True,
  concurrent=16) is identical to `d0c7d1e0e^:.../unsupervised_asr/phonemize.py`.
- The whole `lm/` package entered in one commit (d0c7d1e0e), so there is no later drift. Git HEAD is not
  proven to be the exact version the JUPITER jobs ran, but phon_lm.py has 2 commits and text.py has 3.
- SampleLinesJob and WB/prior.py: the JUPITER source (speech-llm c49559ce) is NOT on i6. The equivalence rests
  on the port's provenance notes and on frozen-log descriptions: `random.Random(seed).sample(range(n),k)`
  (`review_phone_lm_v2:138`); held i % 101, with an empty line consuming an index; BOS, no EOS. The port
  matches those descriptions. Nothing checks the WB fit against a JUPITER input.
- Lexicon hash, text, SIL, sampling: the same MergeLexiconJob and DownloadJob hashes, and 200,000 bliss words
  on both sides. The SIL parameters match JUPITER info lines 7-9.
- Found: the only observed divergence is the g2p-apply output, in coverage. The i6 Sequitur environment also
  has the numpy-2 training defect.

## 5. Frame

- `SAE_i6_ref_blankfree.md:23` defines the reference text as 39,630,169 lines. The i6 prior is fitted on a
  window of a 40,418,258-line corpus. Its corpus size therefore does not trace to the reference setup.
- By construction the i6 prior is a different prior. The comparison cannot answer "did the port reproduce
  the reference prior". It answers only "is the difference explainable", and that only partly.
- Adopting it (orchestrator option b) makes it a disclosed deviation of the bed. As a result:
  - the G0.R1 step-1 prior-per-token clause (-5.637 against -5.657) and every prior-dependent comparison with
    banked JUPITER numbers are confounded by it;
  - rho stays JUPITER-derived (9.6619, where the i6 text would give about 9.679).

## 6. What would decide

- JUPITER artefacts (T0 of the debugger's report):
  - `SampleLinesJob.orN768ARKwlt` text.phn.gz: `fit_prior` on it must give 9.561056 exactly. This settles the
    port's fit code.
  - `ApplyG2PModelJob.myTIGtmrUIFq` g2p.lexicon: the i6 phonemiser must give 39,630,169 lines, and the full
    i6 chain must then give 9.561056. This settles "only".
- Without them: an emulation constrained to JUPITER's banked (types, lines) pair, and 10 or more emulation
  seeds, would narrow the residual but could not prove "only".
