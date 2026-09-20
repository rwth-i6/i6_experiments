# Implementation: SAE 4a Step 0b, the neural phone LM row — 2026-09-20

Built to `SAE_4A_prior.md` "Design", section "Step 0b" (pre-registered). Nothing was launched; the
manager on `config/sae_4a_prior_gap.py` picks the two new jobs up only after a restart.

## What was written

| file | delta |
|---|---|
| `speech_llm/sae/emc/neural_phone_lm.py` | NEW. `NeuralPhoneLmTrainJob` (GPU, in-process torch), the `CausalPhoneLm` model, the training loop, and `NeuralPhoneLmScorer` — the scorer `prior_gap` reads the row with. Reports perplexity, parameters, tokens seen, wall clock and the train-minus-held gap; aborts on a flat held-out epoch and keeps the best epoch; `leakage_check`. |
| `speech_llm/sae/emc/prior_gap.py` | optional `neural_lm` input (hash-excluded at `None`), the `neural_lm` row, Step 0b's read rule (`NEURAL_READ_RULE`, `neural_read`), the row in `PRIOR_KEYS`/`IS_KEYS`/the per-utterance dump/every rendered table; `counted_line_flags` split into `window_line_flags` → `(counted, held)`; `held_line_benchmark` + its rendered section. |
| `speech_llm/sae/emc/test_prior_gap.py` | section (g): the window split, the three branches of Step 0b's rule, the Step 0b row fires no row of Step 0's table, the row and its read render, the `neural_lm=None` record is unchanged, the banked hash is unchanged. Section (h): the held-line benchmark and its render. |
| `speech_llm/sae/emc/test_neural_phone_lm.py` | NEW, 12 checks (a)–(f) below (incl. the abort and the leakage count). |
| `configs/config_sae_4a_prior_gap_v1.py` | `neural_phone_lm()` (alias `sae/4a/prior_gap/neural_phone_lm`) and a second analysis instance (alias `sae/4a/prior_gap/ctrl_50_ep10/dev-other_neural`); the existing instance is untouched, built through a shared `_analysis_job(name, neural_lm=None)`. |

## Hashes

* `NeuralPhoneLmTrainJob.Iv6P6YVPNWmB` — the train job.
* `PriorGapAnalysisJob.Gct95xZHe0zt` — the Step 0b analysis instance.
  (Both moved from the first commit's `VhWt9AOPg5Pc` / `PahcN5xvg9QQ` when the abort tolerance
  `held_improvement_nats` became a job parameter; neither job has ever run, so nothing is orphaned.)
* `PriorGapAnalysisJob.2RkbKYl0v1XK` — the banked v2 instance, **unchanged** (asserted by
  `test_the_banked_v2_instance_keeps_its_hash`, which builds the config and reads `_sis_id()`).
  `__sis_hash_exclude__ = {"neural_lm": None}`: excluding a NEW parameter at its default is
  hash-neutral. `version` stays 2.

## The line source, not re-derived

`NeuralPhoneLmTrainJob._texts()` calls `prior_gap.replay_window_texts` twice on the same four pins
the analysis job uses (`WINDOW_PHN`, `WORD_CORPUS`, `BLISS_LEXICON`, `G2P_LEXICON`) — once with the
COUNTED flags (the lines `lmplz` is fed: the priorshuf uniform window, `SIL` spelling, the
line-by-line phonemization check) and once with the HELD flags (`i % 101 == 0`, the lines
`prior.count_ngrams` withholds). The counted/held split now lives in ONE function,
`prior_gap.window_line_flags`; `counted_line_flags` is its first half and behaves exactly as before.
Cost: the replay reads `librispeech-lm-norm.txt.gz` twice per call (the finished v2 job did the
whole replay plus three `lmplz` builds in 303 s), so the GPU job pays a few minutes of CPU before
training; `time` is 4 h against a 2 h training cap.

## Design constants

From the spec: 4 layers, d_model 256, 4 heads, FFN 1024, 512 learned positions, vocabulary 42
(40 phone ids + BOS 40 + PAD 41), **no end-of-sentence token in the target**, AdamW lr 5e-4 cosine,
3 epochs or 2 h, seed 0, bf16 autocast on CUDA, rqmt `gpu 1, cpu 4, mem 32, time 4`.

NOT fixed by the spec, chosen here, all job parameters (so in the hash) and none swept: token budget
per batch 32768, length-bucketing window 8192 lines, warmup 500 steps, weight decay 0.01, grad clip
1.0, dropout 0.1, train probe 10,000 lines. They are listed under "CONSTANTS THE SPEC DOES NOT FIX"
in the module docstring. A training line over 512 tokens is truncated and counted (0.13 % of the
window); the SCORER refuses a longer string instead (the longest dev-other string is 315 tokens).

Outputs: `model.pt` (state dict + the json config), `train_log.json` (per epoch: running train,
eval-mode train probe, held-out — all nats per token, BOS context, no EOS, pooled over tokens, and
also the mean over lines), `heldout.txt` with one `heldout_nats_per_token=…` line.

## Step 0b's read, as implemented

Per pairing (`like_for_like` primary, `sil_kept` disclosed, as for every other row): the bar is
recomputed from the run's own rows, `gap_3 + 2/3 (gap_escape − gap_3)`, and the spec's arithmetic
(1.39 + 0.62 = 2.01) is banked beside it as `bar_preregistered`. Clause 2 reads the neural gap on
the `lexicon_strict` paired subset against the strict row's gap there, tolerance 0.3. Outcomes and
their verbatim wording: `adequate_lexical_scorer`, `partial_proxy` ("a partial proxy and the arm's
design must say what it loses"), `not_the_scorer` ("a neural phone LM is not the scorer and the
exact lexicon must be made batchable (a GPU trie DP) before an arm exists"). The rule text is
printed verbatim in the report. **Assumption stated in `neural_read`'s docstring**: a row that
clears the bar but does not track the strict lexicon is not adequate and does beat the 4-gram, so it
falls in the middle branch — the only reading of the rule that covers that case.
The Step 0b row is excluded from Step 0's decision table (`STEP0B_KEYS`): that table was
pre-registered over the seven priors above it, and its outcome must not move.

## The benchmark (addition to the brief, same day)

"Is this LM any good?" is a question about TEXT, and the seven prior-gap rows are read on the
split's utterances — so both sides of it were added on the window's HELD lines (`i % 101 == 0`,
the lines `prior.count_ngrams` holds out and no model in either job is fitted on).

* **the train job** reports, per epoch and for the selected epoch, held-out nats per token AND
  `perplexity = exp(nats)` in the one convention (BOS context, no end-of-sentence term, pooled:
  total nats over total tokens), plus `parameters`, `training_tokens_seen`, `seconds`,
  `train_probe_nats_per_token` and `train_minus_held_nats_per_token` (the overfitting check, both
  sides in eval mode at fp32 on a 10 000-line probe of the training text). `heldout.txt` now carries
  `heldout_nats_per_token=` and `heldout_perplexity=`; `train_log.json` and `model.pt`'s meta carry
  all of it.
* **the abort**: an epoch that does not beat the best held-out number by `held_improvement_nats`
  (1e-4 nats, an implementer-chosen dead band, not fixed by the spec) stops the run with a printed
  `ABORT: …` line and a `stopped_reason` in the json. The model saved and scored is the BEST
  epoch's, and `selected_epoch` / `final_epoch` / `selection` say which and why. No label is
  involved: the held lines are LM text.
* **held lines never enter training**: the two splits come from one `window_line_flags` call and
  are asserted disjoint by index before either text is written. Exact TEXT repeats between the two
  halves are counted by `leakage_check` and banked (`text.leakage`) — reported, never repaired: the
  n-gram estimator this LM is compared against holds exactly the same lines out.
* **the analysis instance** (`held_line_benchmark`) replays the held text into `lms_held/`, truncates
  every line to the model's 512 learned positions FOR EVERY MODEL, and scores it with the live
  Witten-Bell trigram, each KenLM order (4, 6) and the neural LM: one table in `prior_gap.md` and
  the same block in the json (`neural_lm.held_line_benchmark`), each row carrying
  `nats_per_token`, `perplexity`, `log_prob_per_token` and `total_log_prob`, plus the line/token
  counts, the truncation count, and the train job's own held-out number beside the recomputed one
  as a cross-check. An instance without a neural LM renders no such section, so the v2 record shape
  is untouched.

Cost: the held replay is a second pass over `librispeech-lm-norm.txt.gz`; the whole banked v2 run
took 302.9 s wall clock, so the 4 h rqmt is not at risk. The analysis job has no GPU, so the neural
scorer runs on its 4 CPU cores.

## Checks run

`pytest test_prior_gap.py test_neural_phone_lm.py` → **36 passed in 5.0 s** (speech_llm env, from
the setup dir; 24 + 12). New: the window split partitions the non-empty lines; the three branches of Step
0b's rule; the Step 0b row leaves Step 0's outcome alone; the row, its IS sd, every field of
`neural_read` and of `neural_subset_gaps` render (the "every banked field rendered" test now covers
them); a `neural_lm=None` record carries neither Step 0b key; the config hashes above. In
`test_neural_phone_lm.py`: 200 synthetic lines train from 3.74 to well below (held-out nats fall by
> 0.5 and the last epoch is not worse than the first); the scorer's log-probability on a 3-token
string equals a hand computation off the model's own logits (BOS context, three conditionals, no
fourth term); batched == single; `model.pt` round-trips; the batcher covers every line within the
budget and is seed-reproducible; targets carry no EOS; the bf16 autocast path runs and leaves the
reported number an fp32 eval-mode number; the train/held texts come from `replay_window_texts`.
The benchmark checks added with it: the 20-epoch tiny run now ABORTS on its own (the held curve goes
flat within the 1e-4 dead band) and the kept epoch is the best one up to that band; a run with an
absurd tolerance aborts in epoch 2, says so in words, and hands back epoch 1's model (re-evaluating
it reproduces epoch 1's number); `leakage_check` counts 1 of 2 repeated held lines and 0 on disjoint
text; `held_line_benchmark` on a real KenLM 4-gram, a toy Witten-Bell prior and a real saved model
truncates every line the same way, matches each model's own scorer to 1e-4, and its three numbers
are one number (pooled log-probability per token, its sign flip, its exp); every field of the block
reaches the rendered report, and an instance without a neural LM renders no such section.

A scratch smoke (not committed) ran the run()-side block end to end on a tiny saved model: the
scorer scores all six string sets, the record assembles, `per_utt.json` carries the `neural_lm` row,
and the rendered report contains the Step 0b section, the verbatim rule and every banked field.
Loading and these checks do not prove the GPU training step itself runs on a GH200 — no job was
launched. Untested by anything executable here: the ~15 lines of `run()` that replay the held text,
read it back and hand it to `held_line_benchmark` (the helper, the scorers and the render are all
covered; the glue needs the real corpus, i.e. the job itself).

Note for the manager: the config now imports `emc.neural_phone_lm`, which imports torch at
module level (the idiom of `emc.supervised_reverse_init` / `emc.blankfree_seed_jobs`), so a
graph load costs a few extra seconds. Nothing else about the graph changed.

## Commit

`recipe/2025-10-speech-llm`, branch `haotian_modality_matching_jupiter`, explicit paths only, not
pushed.

* `739d9ed` — SAE 4a step 0b: a neural phone LM row in the prior gap.
* `e971603` — SAE 4a step 0b: benchmark the phone LM against the n-grams on the held lines.
