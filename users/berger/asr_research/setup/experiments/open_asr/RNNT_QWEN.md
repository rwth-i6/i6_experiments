# Parakeet RNN-T dynamic Qwen fusion

## Current scheduling (2026-10-15)

### Earnings22 diagnostic and long-run budgets (2026-10-19)

`config/earnings_qwen.py` runs ten intermediate pruning points plus existing
8/32 threshold-3 and 32/128 threshold-4 controls. See the entrypoint README
for the grid. LM scale 0.3, models, casing and rewards remain fixed. The aim
is WER about 8.3–8.4 at lower RTF; this is explicit Earnings22 tuning and is
labelled as such in reports. It is included in main, not in dev-only tuning.

GigaSpeech and SPGISpeech remain unsharded. Their measured audio durations are
35.143 and 100.002 hours. Fusion/candidate-generation requests now cover RTF 3
plus 20% margin and two startup hours: 129 and 363 hours. Resource changes do
not change recognition hashes. The settings-level cap has a targeted opt-in;
Slurm partition limits still apply. No jobs are automatically reset/requeued.

`config/open_asr.py` covers the eight held-out corpora plus Loquacious dev/test.
Text Qwen stays at scale 0.3, beams 32/128 and thresholds 2/2. Speech-Qwen
uses the new `speech-qwen-fast` condition: scale 0.3, beams 8/32, thresholds 3/3.
This achieved 3.93% WER at RTF 1.23 on dev.short. Scale 0.3 remains the validated
transfer setting; lower scales at this faster budget are still being tuned.
The old expensive held-out conditions are not registered by the suite.

`config/qwen_tuning.py` adds both dev-other and dev.short, using Qwen Base
0.6B/1.7B. Text scales 0.1/0.3/0.5 use beams 32/128 and threshold 2, with
beam-8/16, pre-32 controls at scale 0.3. Speech-Qwen scales 0.05/0.1/0.2/0.3
use beams 8/32 and threshold 3, plus a beam-16 control at scale 0.1.
Native and zero-LM controls are retained. No held-out model-size tuning.
The dedicated `rnnt_qwen.py` entrypoint runs only the focused 0.6B dev.short grid.
Original broad grids remain opt-in via `run(focused_tuning=False)`.
No submitted jobs are cancelled automatically. See QWEN.md for shared n-best work.

## Historical setup / results (superseded scheduling)

The original dedicated `rnnt_qwen.py` entrypoint called
`run(tuning_only=True)`. Only Loquacious dev.short
is decoded, with audio selected before recognition. Its standalone Speech-Qwen
and native RNN-T controls, zero-LLM controls, scale sweeps and pruning sweeps are
registered (25 recognition jobs total). No test sets or full Loquacious dev
recognitions are scheduled. Existing submitted jobs are not cancelled.
The `open_asr.py` suite now calls `run(tuning_only=False, evaluation_only=True)`
with the eight held-out corpora. This registers only native, text-Qwen and
Speech-Qwen conditions, not the historical controls or sweeps described below.
Frozen thresholds are 2/2 for text-Qwen and 4/4 for Speech-Qwen, with beams
32/128 and LM scale 0.3. `librispeech-dev-other` prepared data can also be supplied to
tuning mode; held-out dataset names are rejected.

Additional Speech-Qwen speed pilots target RTF below 2 on dev.short. First,
thresholds 1/1 and 1.5/1.5 retain caps 32/128. Then, at threshold 3/3
(the best completed development WER among 2/3/4), pre-LLM caps 64, 32 and 16
retain final cap 32; final caps 16 and 8 use pre-LLM cap 32. The existing
threshold-3 cap-32/128 result is the control. LM scale, model precision,
casing and scorer batch size remain fixed. These are seven new development
conditions, not new held-out settings. Compare WER and runtime before transfer;
the RTF target is not yet demonstrated. No time limits are increased.

The baseline entrypoint registers lexicon-free RNN-T experiments using RASR
`d000386c08a9b9213238e3cafdeb4291dc0b1775` (`llm-rnnt-search`), built by a
separate pinned CMake job. Existing CTC binaries and search configs are unchanged.

The acoustic model is `nvidia/parakeet-rnnt-1.1b`, revision
`2acc4c61eede2f52ddefe74935de1930a9064d4a`. This is standard RNN-T, not TDT.
The existing GPU encoder/predictor/joint adapter is reused. Raw acoustic pieces
are retained in model output order; blank is read from the model configuration,
not assumed to be zero. Labels can repeat and several labels can be emitted per
frame; only blank advances time.

Each of the eight Open-ASR corpora and Loquacious dev/test gets:

- Native greedy recognition (existing Open-ASR jobs reused).
- New-search zero-LLM controls with beams 1 and 32.
- First-pass text-Qwen Base fusion, with on-the-fly word tokenization.
- First-pass audio-conditioned Speech-Qwen fusion, dependent on the scored
  standalone Speech-Qwen baseline.

The base matrix has 46 RASR recognition jobs and two new Loquacious native jobs.
Six additional development-only pruning jobs compare final/pre-LLM thresholds
2/2, 3/3 and 4/4 for text-Qwen and Speech-Qwen at fixed scale .3 and beams
32/128. Existing 6/6 conditions are reused as controls, without changing their
job identities. Compare WER, insertion/deletion/substitution counts and RTF
before selecting thresholds and subsequently tightening beams.
Loquacious domain subsets and dev.short are scored from parent hypotheses.
The overview and individual scoring/runtime/traceback outputs are registered
under `open-asr/qwen-parakeet-rnnt`.

Initial final/pre-LLM beams are 32/128, thresholds 6/6, and the maximum labels
per frame is 10. LLM scale .3 is a transfer reference, not an RNN-T optimum;
Loquacious dev additionally sweeps .05, .1, .5 for both fusion methods. Text-Qwen
retains its tuned token reward .5 and word reward 1; Speech-Qwen starts with
neither reward. No pronunciation lexicon, word LM or subword LM is added.
Stepwise RASR statistics are disabled.

Both Qwen scorers use maximum GPU batch size eight and recompute full histories
without assembling KV caches. For Parakeet, RASR scores lowercase and capitalized
variants of every word (deduplicated), then carries the cheaper variant's token
history forward. At segment end the last word is selected together with EOS.
The surface ASR output is not recased. In future Granite suite runs, both scorers
preserve the capitalization hypothesized by Granite instead, with one variant.
Equal batching/casing removes those speed confounders, but Speech-Qwen still has
audio encoding and a longer multimodal prefix, so its RTF need not match text-Qwen.

A CPU native topology gate on the pinned binary must pass before any recognition:
it checks two repeated labels, multiple emissions in one frame, blank time
advancement, scaled word/EOS scores and selected casing history. An inventory job checks raw-piece
round trips. Speech continuation scoring is checked against full multimodal
forwarding on the first utterance. Compare zero-LLM controls with native before
interpreting improvements; beam-32 sums paths whereas beam-1 is Viterbi.
Pending-word costs remain deferred, so monitor word-boundary/deletion errors.
Tune further only on development data before selecting held-out conditions.

This binary uses the request-based LLM API: `tokenize(texts)` returns one token
sequence per text, and `score(requests)` reads each request's `prefix` and
`tokens`. An RNN-T-local adapter converts the existing single-variant Qwen
backends to that API. The older CTC scorer registrations remain unchanged.

Model/dataset downloads continue using `/work/asr4/berger/huggingface` for caches.
The jobs are registered for Sisyphus; no Slurm runs are submitted by this setup.
