# Step 0b arm (c): a phone LM on ten times the text (2026-09-20)

Implementer report.  Dispatch: a third, larger neural phone LM trained on MORE text than the
1,010,000-line window, plus its `PriorGapAnalysisJob` read, in a new config.  First pass returned
BLOCKED (the training job replays its window and derives its held lines, so it can neither be fed a
filtered text nor be told which lines to hold out); the coordinator then fixed **Option C**: a NEW
subclass in a NEW module, `neural_phone_lm.py` untouched.  That is what is built here.  Nothing was
launched.

## The blocker, and what the subclass does about it

`NeuralPhoneLmTrainJob._texts()` (`neural_phone_lm.py:707-737`) does not read `window_phn` as a
corpus: it calls `prior_gap.replay_window_texts` (`prior_gap.py:873-934`), which re-derives the
sample with `sample_line_indices(n_kept, n_window_lines, sample_seed)` and asserts per line that the
phone line is the lexicon phonemization of the word line it recovers (`:919`).  Deleting one line
desynchronises that walk.  Its validation lines are `window_line_flags(window_phn, ...)[1]`, i.e.
`i % 101` of its OWN input (`:715`), so a fit on a larger sample would also be selected, aborted and
reported on lines that are not the benchmark's.

`NeuralPhoneLmTrainJobV2` overrides `_texts()` and NOTHING else: the two splits are inputs
(`train_phn`, `held_phn`), written into the job's work dir in the parent's own text form (`<SIL>`
respelled `SIL`, one line per line, in order) and handed to the inherited `encode_lines` /
`train_phone_lm` / selection / abort / `save_model` / `heldout.txt` path.  It also REFUSES to train
if `leakage_check` finds a held line's text in the training file (in the parent that count is
reported and tolerated -- there the split is by index; here a repeat would mean the exclusion did
not happen).

## Files

| file | delta |
|---|---|
| `speech_llm/sae/emc/neural_phone_lm_v2.py` | NEW. `HeldLinesJob` (writes `window_line_flags(...)[1]` of a window, asserts the count), `NeuralPhoneLmTrainJobV2(NeuralPhoneLmTrainJob)` (adds `train_phn` / `held_phn`, overrides `_texts`), `write_phone_lines`. |
| `speech_llm/sae/emc/text_filter.py` | NEW. `ExcludeLinesJob`: drops every line whose CONTENT is a line of `exclude`, keeps corpus order, banks `stats.json` / `stats.txt` (corpus lines, removed, distinct excluded lines hit, empty lines dropped, written).  Guards: `expect_corpus_lines`, optional `max_removed`. |
| `speech_llm/sae/emc/test_neural_phone_lm_v2.py` | NEW, 6 checks (a)-(e) below. |
| `speech_llm/sae/emc/test_text_filter.py` | NEW, 5 checks. |
| `configs/config_sae_4a_phone_lm_v2.py` | NEW (workspace shim `config/sae_4a_phone_lm.py`, like `config/sae_4a_prior_gap.py`).  The four-job arm and its read; `config_sae_4a_prior_gap_v1` is IMPORTED ONLY. |

`emc/neural_phone_lm.py`, `emc/prior_gap.py`, `emc/text_sample.py` and
`configs/config_sae_4a_prior_gap_v1.py` are untouched, as are the other implementer's files.

## The arm and its hashes (graph load from the setup dir)

```
held_lines            HeldLinesJob.SKxs9aPu2Sha            10,000 held lines of WINDOW_PHN
sample_10m            SampleLinesJob.tHBnfzwuo9ok          10,100,000 lines, seed 1, TEXT_PHN_SIL
sample_10m_no_held    ExcludeLinesJob.nOMT5eGtLKDV         that sample minus the held-line content
phone_lm_10m_l8w512   NeuralPhoneLmTrainJobV2.vkNGAOeLgNsy arm (c)
ctrl_50_ep10/dev-other_neural_10m_l8w512
                      PriorGapAnalysisJob.OO0iAEVLKgOO     the Step 0b read, scored with (c)
```

Aliases `sae/4a/phone_lm_v2/...`; outputs under `<setup>/librispeech/sae_4a_phone_lm_v2/...`.

* **parameters: 25,525,290** -- `n_parameters(build_model(job.config))` on the job's own config
  (8 layers, width 512, 8 heads, FFN 2048, 512 positions, vocab 42).  Head dim 64, the banked
  256 / 4.  The same script reproduces 3,312,170 and 10,876,458 for the existing arms.
* **the sample is a sibling of the window**: `SampleLinesJob(corpus=s1a.TEXT_PHN_SIL,
  n_out=1_010_000, seed=0)` rebuilt in the load script IS `SampleLinesJob.orN768ARKwlt`, so the
  corpus pin and the sampling of arm (c)'s text are the banked window's; only `n_out` and `seed`
  differ.
* **the read is the banked Step 0b read**: `PriorGapAnalysisJob(**analysis_args(pg.NEURAL_NAME,
  neural_lm=<banked model>))` IS `PriorGapAnalysisJob.Gct95xZHe0zt`, so arm (c)'s instance differs
  from the banked one in `name` and `neural_lm` alone (same decode, gold, prior, window, corpus,
  lexicons, KenLM orders, word LM order, `version=2`).
* **the v1 config's seven hashes are unchanged** (`2RkbKYl0v1XK`, `Iv6P6YVPNWmB`, `Gct95xZHe0zt`,
  `xObXEwRpvmzd`, `5wNIQs2lpC5P`, `pBozvj6c3l16`, `pg14aEYJyiva`), asserted in the same process
  after importing and building the new config.

## Held lines: the benchmark is the same 10,000 lines

`HeldLinesJob` takes `pg.WINDOW_PHN` and writes `window_line_flags(window, n_lines=1_010_000,
held_stride=101)[1]` -- the same function, the same defaults, that `PriorGapAnalysisJob` calls at
`prior_gap.py:1536` before replaying its held-line benchmark text -- and asserts there are exactly
`prior.DEFAULT_HELD_LINES` = 10,000.  Run against the real banked window here (read-only), the split
gives **counted 1,000,000 / held 10,000**, so the guard matches the live input.

That file is arm (c)'s `held_phn`, so `heldout.txt`, the best-epoch selection and the abort are
computed on exactly those lines.  Both sides truncate them the same way: `encode_lines` cuts at the
model's `max_positions` (512) in the train job, `held_line_benchmark` cuts every model's strings at
`scorer.max_positions` (512) in the analysis job.  The benchmark's cross-check
`train_job_minus_recomputed_nats` is therefore the same model on the same lines in the same
convention, as its docstring requires -- which it would NOT have been with a derived held split.

Training text: the 10.1 M-line seed-1 sample minus every line whose content is one of those 10,000
(about 2,500 expected: each corpus line is drawn with probability 10.1 M / 39.63 M).  No ceiling is
set on the removal count (the number of duplicate-content matches is not known in advance); the
count is banked in `stats.json` / `stats.txt` and registered as an output.  The corpus was checked
for a bare `SIL` token (there is none, 0 of 1,010,000 window lines), so the `<SIL>` -> `SIL`
respelling cannot create a content collision the exclusion missed; the train job asserts zero
leakage on the written text anyway.

## Budget

From the banked fit (41 s/epoch at 3,312,170 parameters over 1,000,000 counted lines): 7.71x the
parameters and 10.09x the lines is about 3,190 s/epoch, about 4.4 h for the 5 epochs.  With the
dispatch's 1.5x margin: `max_seconds = 24_000` (a constructor argument, hash-relevant) and
`job.rqmt = {..., "time": 8}` (an attribute, overwritten in the config, so it moves no hash);
`gpu 1, cpu 4, mem 32` inherited.  The Slurm hours also cover the two decompression passes
`_texts` makes.  Caveat: parameter-proportional scaling is an extrapolation, not a measurement; if
the cap is hit, the cosine is mid-schedule and the best epoch is still saved and scored
(`stopped_reason` says so).  Memory: `encode_lines` keeps uint8 ids, ~0.8 GB flat plus ~1.3 GB of
transient per-line arrays for 10.09 M lines -- inside the 32 GB.

**Disclosure for the read**: arm (c) differs from arms (a)/(b) in TWO things at once, the text
(10.09 M lines instead of 1.00 M) and the capacity/schedule (25.5 M parameters, 5 epochs).  Its row
answers "is a phone LM good enough with more of everything", not "what did the extra text buy".

## Checks run

`pytest test_text_filter.py test_neural_phone_lm_v2.py test_neural_phone_lm.py test_prior_gap.py`
-> **48 passed in 5.9 s** (11 new + the 13 and 24 that existed).  New:

* `HeldLinesJob` writes exactly `window_line_flags`' held lines of a synthetic window, in order,
  with an empty line on a stride index held by neither half, and its count guard fires;
* `_texts` reads the two handed files and never calls `replay_window_texts` (monkeypatched to
  raise), respells `<SIL>`, banks each file's provenance and line count, and the written text
  encodes with `encode_lines`;
* a held line's text inside the training file makes the job refuse to train;
* the architecture, `epochs`, `max_seconds`, the inherited optimiser constants and the inherited
  `rqmt` are what the arm asks for, and the built model has 25,525,290 parameters;
* END TO END: a tiny `NeuralPhoneLmTrainJobV2.run()` on two handed files produces `model.pt`,
  `train_log.json` and a `heldout.txt` in the banked two-key format, with `heldout_lines` = the
  handed held lines, zero leakage banked, and the saved model reproducing the reported held-out
  nats per token through `NeuralPhoneLmScorer` (1e-4);
* `ExcludeLinesJob` removes every excluded line (including a repeat and a respaced copy), keeps a
  superstring, drops an empty line, preserves order, reports removed / distinct-hit / written, and
  both guards plus the empty-exclusion-file check fire.

Plus the graph-load script (config imported and `build()` run from the setup dir): the five ids
above, the wiring assertions (`train_phn` is the `ExcludeLinesJob` output, `held_phn` and the
exclusion file are the `HeldLinesJob` output, the filter's input is the 10.1 M sample, `neural_lm`
is arm (c)'s `model.pt`, both window inputs are the banked `WINDOW_PHN` pin), the two identity
assertions above and the seven unchanged v1 ids.

Loading and these checks do not prove the GPU fit runs: no job was launched, and the manager on the
new config has not been started (not my scope).  The real removal count, the fit's step rate and
its held-out number are unknown until the arm runs.

## Commit

`d1c14cf` on `haotian_modality_matching_jupiter` in `recipe/2025-10-speech-llm`, five files staged
by explicit path; `config_sae_1g_v1.py`, `config_sae_3e1_d6_swap_cont_v1.py` and
`blankfree_sampler.py` (other people's live edits) left alone.  Not pushed.  The workspace shim
`config/sae_4a_phone_lm.py` lives in the setup dir, which that repository does not track.
