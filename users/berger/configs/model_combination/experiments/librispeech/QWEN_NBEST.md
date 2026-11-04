# Qwen two-pass baseline

Registered by the LibriSpeech entrypoint instead of the closed-tree Qwen grid.
Existing job artifacts are preserved; no submitted jobs are cancelled.

Only **dev-other** is scheduled: pinned `openslr/librispeech_asr`, `other`,
`validation`, revision `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1` (verified
against the HF repository API when setting up these experiments). The manifest
records the split. No test-set references or reference-derived vocabulary are
used. Granite and Qwen repositories reuse the existing pinned benchmark/Base
checkpoints. All inference is float32 on GPU; existing CTC input processing and
the updated HF scoring normalizer are reused.

## First pass

Two extended-tree conditions, pinned RASR ff206f90 plus the existing separator
patch and acoustic/tokenizer preflight:

| Condition | Word LM | beta | alpha | Beam / word beam | Thresholds |
| --- | --- | ---: | ---: | --- | --- |
| open-no-lm | none | 1 | 2 | 256 / 256 | 16 / 16 |
| open-4gram | LibriSpeech 4gram, scale 0.3 | 6 | 2 | 256 / 256 | 16 / 16 |

`recognize_segment_n_best(features, 50)` extracts candidates from the surviving
lattice. Recombination stays on. Unique **surface** candidates are deduplicated
by lowest first-pass cost; fewer than 50 can survive. Generation reports the
actual mean/min/max candidate count and singleton fraction. Raw tracebacks,
AM/LM totals, native rank and fallback diagnostics are retained.

First-pass cost is the final cumulative AM plus LM cost, **not** a sum of
trace-item costs. The n-best first candidate must reproduce ordinary native
recognition's transcription and cost (0.01-nat tolerance), or generation fails.
This is important because epsilon/final-arc accounting and scaling in the native
lattice exporter must not silently change the objective. This runtime gate is
configured, not a claim that these new jobs have run successfully.

## Second pass

Qwen3-0.6B-Base scores complete strings with its native tokenizer/model:
`BOS + encode(surface, add_special_tokens=False) + EOS`. BOS itself is not a
prediction target; every surface token and EOS is scored. No chat prompt,
token-count averaging, truncated history, or fabricated UNK is used. GPU
full-history inference has no KV input assembly. Strings are deduplicated and
length-bucketed into bounded batches of eight.

Known lexical orthographies are uppercase while fallback pieces are lowercase.
For LM scoring only, lowercase the recovered surface uniformly to remove this
route artifact. Keep punctuation, apostrophes, numbers and word spacing; do not
apply the HF scoring normalizer before Qwen. Three deterministic policies:

- lowercase;
- lowercase, with the first alphabetic character capitalized;
- the preceding policy plus a terminal period unless `.`, `!` or `?` is present.

These are simple style diagnostics, **not** full truecasing or punctuation
restoration. Formatting changes affect only Qwen's history, not the recognized
word transcript or acoustic score. Save exact scored strings, IDs and per-token
costs for reproducibility.

Selection minimizes `first_pass_cost + scale * Qwen_NLL`. The 4gram condition
therefore retains its original LM contribution; this is additive rescoring,
not 4gram replacement. Report each first-pass condition separately.

## Evaluation

Evaluate prefixes n=1/10/50 and scales 0/0.05/0.1/0.2/0.3 under each style policy.
One lightweight evaluation job produces the whole grid per first pass; these
are not separate GPU recognition jobs. The scale-zero and n=1 results are
controls. An explicitly labelled reference-aware **oracle** minimizes edit
count among each candidate prefix; references never enter decoding, formatting,
LM scoring or non-oracle selection. Oracle WER is a diagnostic lower bound, not
a deployable condition. Report I/D/S, reference lengths and WER with the same
normalizer/compound-equivalence handling as the existing benchmark scorer.

Outputs: `output/librispeech/qwen-nbest/dev-other/{open-no-lm,open-4gram}/`
contains `candidates`, `generation`, `rasr.log`, `qwen-candidates`, `lm-scoring`,
`first-pass-score`, `summary` and all selected `hypotheses`.

Interpret the oracle first: if it is poor, improve candidate generation before
judging Qwen. If it is good but Qwen selection does not improve WER, inspect the
recorded full-string scores and style effects before implementing dynamic
in-search tokenization. No held-out evaluation is scheduled until dev settings
are selected.

## N-best exporter update

The first version used 8c1ac793. Its unscaled n-best LM scores disagreed with
the scaled one-best tracebacks and correctly triggered the parity gate.
The reported fix is `682423307f60ff3984e5ae669528780502394833` from `llm-search`.
It also fixes lattice
final-weight/EOS handling and gives vocabulary-text labels evaluation tokens
so n-best duplicate removal does not collapse every label path to one.

That `llm-search` commit did not contain the extended OOV tree. The fixes have
now been merged into `oov-fallback` at
`c1c3e28af68579d64ad92a2d58112ab608e2ec1c`. That run still failed parity:
the no-LM n-best cost was 3.0 below one-best, and the 4gram top transcription
differed, both on `3660-172183-0001`. The driver now uses the subsequent fixes
at `ff206f90c64fa5c44fa1acb7059f23722bfefc8c` through a separate pinned
CMake job (`tools/rasr-nbest-ff206f90`). These fixes still need runtime validation.
The approved separator XML patch remains in the
clone job hash. Existing accounting and pruning gates are scheduled afresh on
this same binary, and the prepared lexicon/state-tying validation uses it too.

Do **not** multiply the returned `lm_score` by the LM scale in Python: the fixed
native exporter already returns scaled scores. Final cumulative AM+LM is still
used directly, and transcription/total-cost parity remains mandatory for both
no-LM and 4gram conditions. Changed binary dependencies create new generation
and downstream rescoring jobs;
model/data downloads can be reused. The separate dynamic-search grid remains
pinned to 9804f5aa; the new `llm-search` commit
`36d66e5cd8ca631295f78148eda5a59bd174c02f` is not needed by these OOV-tree
n-best jobs and is not substituted into the completed dynamic one-best grid.

## Oracle-guided diagnostics (2026-10-06)

The driver now adds `diagnostics/` outputs for each first pass. The original
generation and full-dev Qwen scoring jobs retain their hashes and are reused.
Six utterances with the largest error gap to the 50-best oracle are selected
per condition. All candidates in each selected list are tested, not only the
current winner and the oracle. The oracle is sometimes still wrong, so reports
separate exact oracle-text agreement from matching its minimum error count.

Reference-independent formatting probes: lowercase, initial capitalization,
period/question mark, uppercase/title case, scoring-normalized text, capitalized
pronoun I, and coarse commas before conjunctions. Three fixed contexts are used:
no prompt, `Transcript: `, and a speech-transcription prefix. Prompt tokens are
conditioning only, not scored. Every formatting probe must preserve the updated
HF-normalized transcription; no reference-dependent punctuation or word rewrite
is permitted. GPU inference is float32 with full-sequence forwards and no KV cache.

Each shared configuration sweeps Qwen scale, acoustic weight, EOS weight,
word reward and token reward, and retaining/removing the first-pass LM component.
Removing that component also removes fallback penalties stored there: it is an
exploratory ablation, **not** a clean replacement of only the 4gram. Token rewards
are explicit diagnostic length biases, not probability-normalized LM scores.
Cached-score sweeps run without GPU; new formatting/context probes require one
small GPU job per first-pass condition. No AM decoding needs to be repeated.

Initial inspection of cached scores:

| Utterance | Current selection → oracle | Oracle minus selected AM cost | Qwen NLL difference |
| --- | --- | ---: | ---: |
| 8254-84205-0072 | OUT AND AWAKE → OUT IN A WEEK | +4.281 | -6.547 |
| 7697-105815-0000 | VISITED IT → THIS HE DID | +2.416 | -2.943 |
| 1585-131718-0000 | extra THE / A → reference articles | +0.457 | +12.344 |

These are no-LM lists under initial-capital-period formatting. In the first two
cases Qwen already prefers the oracle, but scale 0.2 is too small to overcome
the acoustic disadvantage (pairwise crossover approximately 0.65 and 0.82,
without extra rewards). In the third Qwen favors the less accurate but more
natural wording, so simply increasing its weight cannot fix the pair.

The cached shared-configuration sweep recovers oracle accuracy for only 2/6
selected no-LM examples and 1/6 selected 4gram examples at its best total-error
settings. New punctuation/context forwards have **not** been run yet.
This motivates the probes rather than establishing a globally useful setting.
These hand-selected, reference-guided diagnostics must never be reported as
unbiased WER improvement. A promising *shared* configuration needs validation
on the complete development set, followed by untouched held-out evaluation.

## Full-dev cached-score sweep (2026-10-06)

Evaluated 111 shared settings per first pass on all 2,864 dev-other utterances.
This uses the already completed three formatting policies and cached token NLLs;
the new GPU formatting/prompt probes are still pending. No new recognition or LM
forward is needed. The reusable `EvaluateQwenFullDevJob` is registered under
`qwen-nbest/dev-other/<condition>/full-dev-tuning/{summary,best-hypotheses}`.
Original job hashes are unchanged.

The objective is AM + retained first-pass LM + Qwen scale times (text NLL +
EOS weight times EOS NLL) + word penalty times surface-word count + token penalty
times Qwen text-token count. BOS, prompt and EOS tokens get no token reward.
All 50-best candidates compete; some lists contain fewer than 50 unique texts.

| First pass | First-pass WER | Previous Qwen best | New full-dev best | I / D / S | Errors |
| --- | ---: | ---: | ---: | --- | ---: |
| Open, no LM | 2.6307 | 2.5414 | **2.5220** | 102 / 71 / 1125 | 1298 |
| Open + 4gram | 2.5491 | 2.5491 (Qwen off) | **2.5255** | 112 / 72 / 1116 | 1300 |

Both winners use initial-capital-period formatting, full EOS weight and retain
the first-pass LM column. No-LM winner: Qwen scale 0.3, word penalty 0,
token penalty -0.5. 4gram winner: scale 0.1, word penalty -1, token penalty -0.5.
Versus the respective first passes, 96 utterances improve and 51 worsen in the
no-LM condition; 49 improve and 42 worsen with 4gram. The latter primarily
trades fewer deletions (104 to 72) for more insertions (89 to 112).

The strong sample-fitted ablations do not generalize: best removal-of-first-pass-LM
points are 2.6033 and 2.5859 respectively. Those are not a pure 4gram replacement,
because the column includes fallback penalties. They also use stronger rewards,
so the contrast does not isolate a single causal factor.

Local full-dev report files from the completed inspection:
`/var/tmp/qwen-full-dev-o2mkwffh/summary.json` (no LM) and
`/var/tmp/qwen-full-dev-xpmlwq61/summary.json` (4gram), with corresponding
`best_hypotheses.jsonl`. Sisyphus can reproduce/register them through the new CPU
jobs. These are development-selected results, not held-out gains; the 10–12
error improvements over previous best settings are small.
