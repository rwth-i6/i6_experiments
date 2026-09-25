# Implementation: prior window spread and coverage-loss emulation (2026-09-25)

Status: DONE_WITH_CONCERNS. The code is written and smoke-tested, and the seed-0 path reproduces the
i6 job exactly. The full run has NOT been launched; the executor launches it. The concerns are in
section 5.

## 1. Files

- `analysis/prior_gap/prior_window_spread.py` (new, 421 lines): the analysis.
- `analysis/prior_gap/run.sh` (new): an sbatch wrapper for cpu_modern, account hlt, 2 CPUs, 64G,
  2:30:00. The Slurm log and all outputs go to the out_dir given as argument 1. The wrapper refuses
  `/tmp` and `/var/tmp`.

Nothing under `recipe/` was edited.

## 2. How the job code is reused

- Parameters come from the job `info` files: n_out 1010000 and seed 0 from
  `SampleLinesJob.CrPgeKXsOosb`; n_count_lines 1000000 and n_held_lines 10000 from
  `PhoneNgramPriorJob.qJxXHgXLe31S`; and the text, bliss and g2p paths from
  `PhonemizeWithSilJob.NpoY1pGJWNUJ`. The script asserts that the input paths chain together
  correctly and that the job's `prior.json` value equals 9.601837932100644.
- Sampling calls `phone_text.sample_line_indices` and `phone_text.iter_sampled`, then uses the job's
  gzip write loop. Fitting calls `phone_prior.fit_prior` on the written file, which is what
  `PhoneNgramPriorJob.run` does. The job's `prior.save` stores each fitted prior as
  `priors/<case>.prior.npz`. The samples are kept as `samples/<case>.phn.gz`, about 55 MB each.
- For a filtered corpus, the in-order stream of its kept lines (`itertools.compress`) goes to
  `iter_sampled`, and n_in is the number of kept lines. This matches what SampleLinesJob would
  read from a file holding exactly those lines.
- The corpora are read once each, in one joint word+phone stream, with the phone lines held in
  memory. The lexicons are loaded with the package's own `lexicon._load_lexicon_word_to_phon` and
  `_load_g2p_lexicon`, merged the way `PhonemizeWithSilJob.run` merges them.

Alignment check: a word line is skipped exactly when the phonemiser would drop it (the line is
empty, or it holds a word with no pronunciation). For every other line, the number of non-SIL phone
tokens must equal the sum of that line's pronunciation lengths. The phone line must also start and
end with `<SIL>` and hold at most W+1 SILs. Any mismatch aborts. In full mode, the script also
asserts the totals: 40,418,261 word lines, and 40,418,258 phone lines, which equals the job's n_in.

Self-check: T1 seed 0 runs first. The script aborts if the relative difference in ppl3 exceeds
1e-9. It also records whether the seed-0 sample, decompressed, is identical to the job's output
(sha256).

## 3. Case definitions

- T1: seeds 0-4 on the full i6 phone text.
- T2a: drop every line that holds a non-bliss word.
- T2b: emulation seed e. The population is the non-bliss types that occur in the i6 phone corpus,
  sorted lexicographically. The script drops `random.Random(e).sample(types, N // 2)` and every line
  that holds one of those types, then samples with seed 0.

Outputs: `prior_window_spread.json` and `.txt`. Each case reports corpus lines, lines sampled,
counted and held, tokens counted, SIL token rate, and ppl for orders 1-3. The outputs also include
T1 and T2b mean/sd/min/max, the JUPITER-minus-T1-mean gap, and the T2 lines kept against 39,630,169.
A partial JSON is written after every case.

## 4. Checks run (desktop)

- Smoke test, 200k word lines, n_out 10100: rc 0, 11 s, 0.4 GB. Alignment passed; the 1 dropped
  line is word line 0, which is empty.
- Timing run, 2M word lines, full n_out 1,010,000: rc 0, 10:55 wall, 1.36 GB max RSS. The joint
  pass took 14 s; each case took about 71 s.
- Streaming seed-0 check. It uses the script's own `sample_and_fit` on the FULL phone text, streamed
  rather than held. Result: ppl3 9.601837932100644, relative difference 0.0. The sample's sha256
  matches the job's output. tokens_counted is 82,740,447 and the SIL rate is 0.1383042080978847,
  both equal to the job's `prior.json`. Took 102 s.
- `run.sh` dry run with a stub sbatch: the arguments and the `--wrap` string are correct.
  `/var/tmp` is refused.

The full-corpus joint pass itself has not been run: the full T2 alignment and the full-mode asserts
are exercised only on the cluster.

## 5. Concerns and undetermined points

- The i6 phonemiser's 3 dropped lines are the empty word line 0 plus 2 lines containing `HHH`.
  (`zcat | grep -wc HHH` gives 2.) The debugger report (section 1) says all three hold `HHH`. This
  does not affect any number here.
- T2b drops types uniformly at random. JUPITER's actual loss may have had structure: for example,
  whole failed parts of the 16-way concurrent `ApplyG2PModelJob`, which would be contiguous
  alphabetical chunks. Uniform dropping is the dispatch's definition; the lines kept show how
  close it comes. Choosing exactly N//2 types (rather than Bernoulli 0.5 per type) and using
  `random.Random` over the sorted type list were my implementation choices.
- `sbatch --test-only` (cpu_modern, hlt, 2 CPUs, 32G or 64G, 1 h or 2.5 h) estimated a start at
  2026-09-27 07:37, even though 18 nodes were idle. The executor should check the real queue wait.
- Expected wall time is about 25 min: a joint pass of about 5-8 min plus 9 cases at about 75 s
  each. Expected peak RSS is about 15-20 GB, extrapolated from 1.0 GB after 2M lines.

## 6. Launch

`/u/hwu/setups/librispeech-960/2026-09-24-unsupervised/analysis/prior_gap/run.sh /work/asr4/hwu/setups/librispeech-960/2026-09-24-unsupervised/analysis_out/prior_window_spread`

The out_dir is a suggestion; any shared path works.
