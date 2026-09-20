# BLOCKED: the 10 M-line phone LM arm (c) cannot be wired without editing `neural_phone_lm.py`

Implementer report, 2026-09-20.  Dispatch: a third, larger neural phone LM (8 layers, width 512,
8 heads, FFN 2048, 5 epochs) trained on a 10,100,000-line uniform sample (seed 1) of the same
phonemised corpus with the window's 10,000 held lines excluded, plus its `PriorGapAnalysisJob`
read, in a NEW config `config_sae_4a_phone_lm_v2.py`.

**Nothing was written into the recipe and nothing was committed.**  Two properties the dispatch
requires are not injectable through `NeuralPhoneLmTrainJob`'s constructor, and the dispatch's own
fallback for that case is BLOCKED.  Everything that IS determinate (architecture, parameter count,
time budget, the analysis instance) is below, so the arm can be built in one pass once the fork
below is decided.

## Blocker 1 -- the train job cannot consume a filtered text file at all

`NeuralPhoneLmTrainJob._texts()` (`sae/emc/neural_phone_lm.py:707-737`) does NOT read `window_phn`
as a corpus.  It calls `prior_gap.replay_window_texts` (`sae/emc/prior_gap.py:873-934`), which

* counts `n_kept` = the lines of `word_corpus` whose words are all in the lexicon vocabulary,
* re-derives the sample selection as `text_sample.sample_line_indices(n_kept, n_window_lines,
  sample_seed)`, and
* walks `word_corpus` and `window_phn` IN LOCKSTEP, asserting per line that the window's phone line
  with SIL removed equals the concatenation of the lexicon pronunciations of the recovered word
  line (`prior_gap.py:919`), and finally `n_checked == n_window_lines` (`:931`).

An `ExcludeLinesJob` output is a file with lines DELETED, so from the first removed line on the
phone line no longer corresponds to the word line the replay recovers, and the job dies in `run()`
on the assertion at `prior_gap.py:919`.  There is no constructor argument that bypasses the replay
(no train-text path input); the only knobs are `n_window_lines`, `held_stride`, `sample_seed`.
The unfiltered sample (`SampleLinesJob(n_out=10_100_000, seed=1)` output, `n_window_lines=
10_100_000`, `sample_seed=1`) WOULD replay cleanly -- but then about a quarter of the 10,000 held
lines (each drawn with probability 10.1 M / 39.63 M) land in training, which is exactly what the
exclusion job exists to prevent.

## Blocker 2 -- the held lines are derived, never handed in

The train job's validation set is `window_line_flags(window_phn, n_lines=n_window_lines,
held_stride=held_stride)[1]` -- the lines `i % held_stride == 0` of ITS OWN input
(`neural_phone_lm.py:715`, `prior_gap.py:805-828`).  The constructor has no held-lines path
(full signature: `window_phn, word_corpus, bliss_lexicon, g2p_lexicon, n_window_lines,
held_stride, sample_seed, n_layers, d_model, n_heads, d_ff, max_positions, dropout, epochs, lr,
warmup_steps, weight_decay, grad_clip, max_tokens_per_batch, sort_window_lines, max_seconds,
probe_lines, held_improvement_nats, seed, bf16, version`).  On a 10.1 M-line input its held set is
about 100,000 lines of the NEW sample, not the original 10,000, and that set drives

* `heldout.txt` / `heldout_nats_per_token` (the number the other two arms report on the original
  held lines -- no longer comparable),
* best-epoch selection and the no-improvement abort (`held_improvement_nats`), i.e. WHICH model is
  saved, and
* `held_line_benchmark`'s cross-check field `train_job_minus_recomputed_nats`
  (`prior_gap.py:517-590`, fed `train_meta=scorer.meta` at `:1645`), whose docstring says "the two
  are the same model on the same lines in the same convention, so a difference is a defect" -- it
  would bank a large spurious "defect" delta.

The dispatch's instruction for this case ("use an injectable held-lines input if one exists, else
report BLOCKED") applies: no such input exists.

**What is NOT affected:** the analysis instance (item 3) derives the benchmark lines from ITS OWN
`window_phn` pin (`prior_gap.py:1536`), which stays `WINDOW_PHN`.  So a `PriorGapAnalysisJob` with
`neural_lm = (c).out_model` still scores the new model on the ORIGINAL 10,000 held lines, whatever
the train job trained on.  Item 3 is buildable as specified.

## The fork (for the planner; I chose nothing)

**Option A -- the minimal recipe change (forbidden to me by the dispatch).**  In
`sae/emc/neural_phone_lm.py`, two optional inputs plus a branch:

```python
    train_phn: Optional[tk.Path] = None,   # a ready phone-line training text
    held_phn: Optional[tk.Path] = None,    # the benchmark held lines, handed in
    ...
    __sis_hash_exclude__ = {"train_phn": None, "held_phn": None}
```

and in `_texts()`, when both are given, copy/stream those two files instead of calling
`window_line_flags` + `replay_window_texts` (keep `leakage_check` and the disjointness assertion).
About 15 lines.  Excluding NEW parameters at their default is hash-neutral (the pattern already
used for `prior_gap`'s `neural_lm`), so `Iv6P6YVPNWmB`, `xObXEwRpvmzd`, `pBozvj6c3l16` and the four
analysis instances keep their hashes -- but the module is re-imported live by the two RUNNING train
jobs on every resubmit, so the edit must be made with that in mind.  With it, the dispatch's design
(SampleLinesJob 10.1 M seed 1 -> ExcludeLinesJob -> train job, held lines handed in) works exactly
as written and the train job validates, selects and reports on the ORIGINAL 10,000 held lines.

**Option B -- config-level only, but it needs constants the dispatch does not fix.**  To satisfy
the replay without a recipe edit, the new module would have to emit a phone file AND its WORD twin
in lockstep (recovering the word lines by replaying `sample_line_indices` over
`librispeech-lm-norm.txt.gz`), and hand the twin as `word_corpus` with `n_window_lines = N` and any
seed (`sample_line_indices(N, N, seed)` is the identity, `text_sample.py:67-75`).  Two sub-variants:
  * filter the 39.63 M-line phonemised corpus FIRST and sample 10,100,000 lines (seed 1) from the
    filtered corpus: determinate, fixes Blocker 1, leaves Blocker 2 (train job holds out ~100,000
    lines of the new sample);
  * additionally lay the file out so that the stride positions ARE the 10,000 original held lines
    (choose `held_stride = s` and a total line count `N` with `ceil(N / s) == 10_000`): this fixes
    Blocker 2 too, but `s`, `N` and the truncation policy for the surplus training lines are
    undetermined values that change the training data, and the post-exclusion line count is only
    known at run time.  I did not choose them.

## Determinate parts, measured

* Parameter count of arch (c) 8 / 512 / 8 / 2048, 512 positions, vocab 42:
  **25,525,290** (`n_parameters(build_model(model_config(...)))`, speech_llm conda env).  The same
  script reproduces the banked 3,312,170 (4/256/4/1024) and 10,876,458 (6/384/6/1536), matching the
  rerun report -- so the count is the job's own model, not an estimate.
* Time, scaled from the banked 41 s/epoch at 3,312,170 parameters over 1,000,000 counted lines:
  25,525,290 / 3,312,170 = 7.71x parameters, 10.09 M / 1.00 M = 10.09x lines -> about 3,190 s per
  epoch, 5 epochs = 15,950 s = 4.43 h of training.  With the dispatch's 1.5x margin:
  `max_seconds = 24_000` (6.7 h, a constructor argument, injectable) and Slurm
  `time = 8` h (`job.rqmt = {...}` overwritten from the config after construction -- `rqmt` is a
  plain attribute set at `neural_phone_lm.py:702`, read in `tasks()` and in `run()`, and is not part
  of the sisyphus hash, so no recipe edit is needed for it).  The 8 h also covers the replay, which
  runs TWICE over the 39.63 M-line word corpus and writes 10x more lines than the banked run's
  515.6 s.  Caveat: parameter-proportional scaling is an extrapolation, not a measurement; at 3.3 M
  the GH200 may have been partly launch-bound, in which case (c) is faster than this.
* Memory: `encode_lines` stores uint8 ids (`neural_phone_lm.py:148-174`), so 10.09 M lines at ~81
  tokens is ~0.8 GB flat plus ~1.3 GB of transient per-line arrays -- inside the job's 32 GB.

## Checks run

* Code reads of `neural_phone_lm.py` (ctor, `_texts`, `run`, `encode_lines`), `prior_gap.py`
  (`replay_window_texts`, `window_line_flags`, `held_line_benchmark`, `PriorGapAnalysisJob.run`),
  `text_sample.py` (`SampleLinesJob`, `sample_line_indices`) and `config_sae_4a_prior_gap_v1.py`.
* The parameter-count script above (speech_llm conda env, recipe on `PYTHONPATH`).
* No config was written, so no graph load, no hashes, no tests, no commit.
