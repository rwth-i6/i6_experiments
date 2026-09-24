# Review: Step 0b arm (c) phone LM v2 (2026-09-20)

Read-only review of commit `d1c14cf` (`recipe/2025-10-speech-llm`, branch
`haotian_modality_matching_jupiter`) before the ~4.4 h GPU fit is funded.  Nothing was edited or
launched.  Verdict: **APPROVE_WITH_AMENDMENTS** -- the code does what the dispatch says; the
amendments are disclosures that must travel with the row, not code changes.

`git show --numstat d1c14cf`: five files, 894 insertions, **0 deletions and no modified file**.
`emc/neural_phone_lm.py`, `emc/prior_gap.py`, `emc/text_sample.py` and
`configs/config_sae_4a_prior_gap_v1.py` are untouched, so the diff cannot move a banked hash by
editing a shared file.

## 1. Leakage and the held set -- clean

* **The held set is the benchmark's, measured, not argued.**  Counted directly on the banked window
  (`work/.../SampleLinesJob.orN768ARKwlt/output/text.phn.gz`): the `i % 101` lines of the first
  1,010,000 are **10,000 lines / 806,207 tokens under the 512 cap** (raw 808,146).  The banked read
  `PriorGapAnalysisJob.Gct95xZHe0zt` banks `held_line_benchmark = {lines: 10000, tokens: 806207,
  truncated_lines: 12, max_positions: 512}`.  Same lines, same cap, same tokens.
* `HeldLinesJob.run` (`neural_phone_lm_v2.py:118-143`) writes those lines from
  `prior_gap.window_line_flags(...)[1]` in window order; the analysis job's benchmark text comes
  from `replay_window_texts(..., held_flags)` (`prior_gap.py:1556-1559`), which writes the same
  window phone lines in the same order with `<SIL>` -> `SIL`.  `write_phone_lines`
  (`neural_phone_lm_v2.py:64-79`) applies exactly that respelling, so the two files are identical
  byte for byte, in order, and both sides truncate at `max_positions=512` (`encode_lines`
  `neural_phone_lm.py:148-173` vs `held_line_benchmark` `prior_gap.py:541-543`).
  `train_job_minus_recomputed_nats` is therefore a real cross-check.
* **No held content can survive in the training text.**  Both texts descend from the SAME pin:
  the graph load shows `SampleLinesJob(corpus=s1a.TEXT_PHN_SIL, n_out=1_010_000, seed=0)` rehashes
  to `orN768ARKwlt`, the banked window, so arm (c)'s sample is a sibling draw of the same corpus.
  `SampleLinesJob.run` (`text_sample.py:157-176`) copies lines verbatim and asserts exactly `n_out`
  lines, so a corpus line has the same text in both samples; `ExcludeLinesJob`
  (`text_filter.py:58-86`) compares `" ".join(line.split())`, which absorbs any whitespace
  difference, and removes EVERY match -- duplicates of a held line elsewhere in the corpus are
  removed too (safe over-removal).  Empty lines are impossible downstream
  (`filter_lines` drops them, `write_phone_lines:76` and `encode_lines:162` assert).
  `_texts` then refuses to train on any residual repeat (`neural_phone_lm_v2.py:225-229`), a
  strictly stronger rule than the parent's report-and-tolerate.
* Residual, negligible: a training line whose FIRST 512 tokens equal a >512-token held line's first
  512 would leak after truncation (12 held lines exceed 512).  No practical path.

## 2. The override reaches the loop -- yes

`run()` (`neural_phone_lm.py:739-746`) calls `self._texts()` once and feeds both returned paths to
`encode_lines`; there is no second read of a window anywhere in `run`, `train_phone_lm` or
`save_model`.  The parent's four replay inputs are `None` in the V2 instance (verified on the built
job) and are touched only inside the overridden `_texts`.  `record["text"]["train"]` loses the
parent's `counted_lines` key and gains `source`/`source_lines`; nothing reads `counted_lines`, so
no consumer breaks.

`train_phn`/`held_phn` are assigned AFTER `super().__init__()` (`neural_phone_lm_v2.py:203-204`).
That is safe: sisyphus collects inputs from `self.__dict__` after `__init__` returns
(`tools/sisyphus/sisyphus/job.py:245`), and the built job's `_sis_inputs` is exactly
`{ExcludeLinesJob.nOMT5eGtLKDV/output/text.phn.gz, HeldLinesJob.SKxs9aPu2Sha/output/held.phn.gz}`,
so both are true dependencies and the fit cannot start before them.

## 3. Schedule vs cap -- comfortable, measured from the banked runs

Banked per-epoch seconds over 1,000,000 lines (`train_log.json`):

| job | params | s/epoch | outcome |
|---|---|---|---|
| `Iv6P6YVPNWmB` | 3,312,170 | 41-43 | ran the full 3 epochs (schedule-bound) |
| `xObXEwRpvmzd` | 3,312,170 | 42 | planned 30, aborted at 11 (no improvement), best epoch 10 |
| `pBozvj6c3l16` | 10,876,458 | 75.7 | planned 30, aborted at 11, best epoch 10, ppl 4.603 |

3.28x the parameters cost only **1.80x** the time, so the implementer's linear-in-parameters
estimate (3,190 s/epoch, 4.4 h) is conservative.  Extrapolating the measured slope to 25.5 M
parameters gives roughly 130-160 s/epoch at 1 M lines, i.e. ~1,300-1,600 s/epoch at 10.09 M lines
and **~1.8-2.2 h for the 5 epochs** -- far inside `max_seconds = 24,000 s` (6.67 h,
`config_sae_4a_phone_lm_v2.py:83`) and the 8 h Slurm request (`:85`, set as an attribute at `:143`,
so it moves no hash).  The non-training overhead (two decompression passes, `encode_lines` and
`leakage_check` over 10.09 M lines) is minutes, not hours.  A wall clamp truncating the schedule is
not the risk here; the cosine is laid over all 5 epochs (`neural_phone_lm.py:395-397`).

Checkpoint selection: the epoch with the lowest held-out nats per token (`train_phone_lm:479-484`),
and `heldout_nats_per_token` is that epoch's (`:508-511`).  The no-improvement abort (1e-4 nats)
is inherited; with only 5 epochs and a cosine decaying to zero it is unlikely to fire early.

Memory: `encode_lines` (~3 GB transient), the 5-epoch batch plan (~0.5 GB) and `leakage_check`'s
10.1 M-element sha1 set (~0.9 GB) fit the inherited 32 GB.

## 4. Hash discipline -- verified by graph load

Loaded both configs from the setup dir and printed ids only:

```
V1  2RkbKYl0v1XK  Iv6P6YVPNWmB  Gct95xZHe0zt  xObXEwRpvmzd  5wNIQs2lpC5P  pBozvj6c3l16  pg14aEYJyiva
V2  HeldLinesJob.SKxs9aPu2Sha   SampleLinesJob.tHBnfzwuo9ok   ExcludeLinesJob.nOMT5eGtLKDV
    NeuralPhoneLmTrainJobV2.vkNGAOeLgNsy   PriorGapAnalysisJob.OO0iAEVLKgOO
PriorGapAnalysisJob(**v2.analysis_args(pg.NEURAL_NAME, neural_lm=<banked model>)) -> Gct95xZHe0zt
SampleLinesJob(TEXT_PHN_SIL, n_out=1_010_000, seed=0)                            -> orN768ARKwlt
```

All seven v1 ids match the banked list, so no existing job is re-funded.  The banked read rebuilt
from the NEW config's `analysis_args` IS `Gct95xZHe0zt`, so `OO0iAEVLKgOO` differs from it in
`name` and `neural_lm` alone; every other argument was compared field by field on the two built
jobs and is equal, including the ones `analysis_args` leaves at their defaults
(`n_window_lines=1010000`, `held_stride=101`, `sample_seed=0`, `null_seed=0`, `beam=None`,
`escape_length_log_prob=-0.6931471805599453`, `kenlm_orders=(4,6)`, `word_lm_order=3`,
`version=2`).  The new classes live in NEW modules, so the file-sha `__sis_version__` trap does not
apply.

## 5. Anything beyond the delta

The built train job: `config = {n_layers 8, d_model 512, n_heads 8, d_ff 2048, max_positions 512,
dropout 0.1, vocab 42, bos 40, pad 41, eos None}`, `epochs 5`, `max_seconds 24000`,
`rqmt {gpu 1, cpu 4, mem 32, time 8}`, and every other constant at the parent's default:
lr 5e-4, warmup 500, weight decay 0.01, grad clip 1.0, 32,768 tokens/batch, sort window 8,192,
probe 10,000, abort 1e-4, seed 0, bf16 True, version 1.  `n_parameters(build_model(config))` =
**25,525,290**, the number the report claims.  No tokeniser, SIL-convention, truncation or
seed-semantics change anywhere.

## Amendments (disclosure, not code)

1. **Three knobs move at once vs arm (b)** (`config_sae_4a_phone_lm_v2.py:74-81`): text
   10.09 M vs 1.00 M lines, capacity 25.5 M vs 10.9 M parameters, epochs 5 vs 30.  The row answers
   "is a phone LM good enough with more of everything", never "what did the extra text buy".  The
   implementer discloses this; it must be carried into `SAE_4A_prior.md` with the number.
2. **The pre-registered Step 0b candidate is not this one.**  The read rule registered with the
   producing code (`neural_phone_lm.py` module docstring, "about 4 layers, width 256, 5 to 8 M
   parameters ... trained on the same priorshuf uniform window as every other prior") is deviated
   from on both capacity and training text.  The trigram / 4-gram / 6-gram rows of the held-line
   benchmark are still fitted on the window's 1,000,000 counted lines, so the neural-vs-n-gram
   held-line comparison is no longer like-for-like in training data.  The GATE itself is unharmed:
   the 2.01 bar and the 2.51 strict-subset reference are built from the trigram and lexicon rows,
   which are identical in every instance.
3. **The benchmark perplexity is a min over epochs on its own selection set** (`train_phone_lm`
   selects the best epoch on the same 10,000 lines that are reported).  This is the banked arms'
   protocol too (min over 3 and over 30), so the neural rows stay mutually comparable, but the
   n-gram rows carry no such selection.

## Notes (no failure path, recorded only)

* `ExcludeLinesJob` is built without `max_removed` (`config_sae_4a_phone_lm_v2.py:120-121`).
  Over-removal is harmless and the real guard (`expect_lines=10_000` on `HeldLinesJob`) is present;
  the removal count is banked in `stats.json`.  Expect ~2,575 direct hits plus duplicate matches.
* `sample_line_indices` (`text_sample.py:67-75`) uses `random.Random(seed).sample(range(n), k)`.
  At k = 10.1 M of 39.63 M CPython takes the POOL branch (`list(range(39.63M))`, ~1.4 GB) where the
  banked 1.01 M draw took the low-memory set branch.  It fits the job's 8 GB, but this is the first
  run of that branch at this scale.

## What I checked and did not find

No silent no-op (every new constructor argument reaches a read), no mismatch between the dispatch
and the effect, no constant that traces to nothing (10,100,000 = 10 x the banked window,
`held_stride`/`n_window_lines` inherited, 512 and the SIL convention unchanged), and no train/eval
contact beyond item 3 above.
