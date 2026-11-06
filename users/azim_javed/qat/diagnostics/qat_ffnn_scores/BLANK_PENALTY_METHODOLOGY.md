# Blank-Penalty Regression Methodology

For the separate threshold-based decision-risk optimizer, see
[`OPTIMAL_BLANK_PENALTY_METHODOLOGY.md`](OPTIMAL_BLANK_PENALTY_METHODOLOGY.md).

This document describes the diagnostic implemented in
[`fit_blank_penalty.py`](fit_blank_penalty.py) and its Sisyphus integration in
[`reference_scoring.py`](reference_scoring.py). For the underlying transducer
alignment and score-extraction code, see
[`ANALYSIS_WALKTHROUGH.md`](ANALYSIS_WALKTHROUGH.md).

## Question and models

We want to understand whether a constant blank penalty can make the **fully
on-device FFNN transducer** score blank relative to tokens more like its
**digital reference**. In the saved files, the digital reference is
`digital_model` and the fully on-device model is `base_model`. Despite the
name `base_model`, it is the on-device target, not the digital reference. The
digital reference is a QAT model running digitally; this experiment does not
compare against a separate full-precision checkpoint.

The fit estimates a *single additive blank decoding cost*, called `b`. It is
a score-matching diagnostic, not a WER optimization or a model retraining step.
Both extraction configurations use `blank_penalty=0.0` and `ilm_scale=0.0`.

## 1. Prepare comparable scores

`prepare_scores(...)` in `reference_scoring.py` creates the digital extraction
job and whichever on-device extraction jobs were requested. For the regression,
only `model_names=["base_model"]` is needed. It returns a
`PreparedReferenceScores` object containing the score-directory paths and
corpus/segment information. `run_blank_penalty_fit(prepared)` creates the
regression job, consuming the **outputs of those extraction jobs**. The
independent `run_analysis(prepared)` entrypoint creates the heatmap/summary
analysis; it is not required for the regression.

The current `config/scorer_debug.py` prepares two `dev-other` runs: the first
fifth (`num_segments=0.2`) and the full corpus (`num_segments=1`), with
distinct output prefixes. For `num_segments <= 1`, preparation takes the
first file from a `SegmentCorpusJob` split into approximately
`round(1 / num_segments)` ordered parts. For `num_segments > 1`, it takes the
first specified number of segments. This is **not random sampling**. Explicit
`segments` bypasses the count/split selection.

For each selected utterance, the two extraction jobs save an NPZ file with:

| Array | Shape | Meaning |
|---|---|---|
| `logits` | `[T, U+1, V]` | Joint-network logits at each encoder frame and reference-token prefix |
| `labels` | `[U]` | Reference BPE token IDs |
| `blank_label` | scalar | Blank symbol ID, checked to be the final vocabulary entry by the fit |

Here `T` is the number of encoder frames, `U` the number of reference tokens,
and `V` the vocabulary size including blank. Both models score the **same
utterance, frame `t`, and correct reference history of length `u`**. Their
independently decoded hypotheses need not have the same histories: decoding
is not used to build these lattices. The fit checks that utterance sets,
filenames, lattice shapes, reference labels, and blank IDs match.

## 2. Obtain a digital soft alignment

The fit applies log-softmax to the digital logits and uses forward-backward to
sum over all paths that produce the complete reference transcript. It follows
the **time-synchronous monotonic** topology used by this diagnostic: each
encoder frame has one transition, either blank (stay at prefix `u`) or the
next reference token (advance to `u+1`). **Both** transitions consume a frame.
This differs from the topology of a conventional RNN-T that can emit several
labels without advancing time.

Forward-backward returns two transition-posterior arrays:

| Weight | Shape | Interpretation |
|---|---|---|
| `label_weights[t,u]` | `[T,U]` | Probability of emitting reference token `u` at frame `t`, conditioned on producing the complete reference |
| `blank_weights[t,u]` | `[T,U+1]` | Probability of taking blank at frame `t` from prefix `u`, under the same condition |

The `U+1`-th prefix is the state after the final reference token: it can still
take blank transitions. For each reference token, the emission weights sum
over frames to one. For each frame, all blank and label transition weights
together sum to one. The script checks both properties. Across the entire
utterance, emission weights sum to `U`, blank weights to `T-U`, and all path
weights to `T`.

These weights are always computed **once from the unpenalized digital model**
and reused when evaluating the on-device model. Otherwise each model could
choose different favorable frames, obscuring a paired score comparison. The
digital alignment is a *reference-conditioned model posterior*, not an
independent timing annotation and not a free-decoding path. The fit checks
posterior normalization; the separate heatmap analysis additionally checks
its forward likelihood against the native monotonic transducer loss.

## 3. Match a score difference in the decoder's units

At a matched position `(t,u)`, the fit considers two possible margins:

```text
reference-token margin = log P(next reference token) - log P(blank)
all-nonblank margin   = log P(any nonblank token) - log P(blank)
```

For the second margin, `P(any nonblank token)` is the **sum** of all nonblank
token probabilities; in log space it is a log-sum-exp. There is no next
reference token at prefix `u=U`, but the all-nonblank margin is still defined
there. The comparison is in log-score units, not raw probability points.
Although the script computes log-softmax, the common softmax normalizer
cancels in each margin. It can equivalently be calculated from the logits.

For every matched position, calculate:

```text
required correction = digital margin - on-device margin
```

The on-device decoder calculates `scores = -log_softmax(logits)` and adds `b`
to the **blank score/cost**. This raises either nonblank-versus-blank margin
by exactly `b`: a positive `b` makes blank less attractive. The decoder does
not re-normalize its scores after this addition. The fit therefore targets
the decoder's *relative blank cost*, not an absolute probability obtained by
subtracting `b` from `P(blank)`.

## 4. Four weighting/margin objectives

The fit repeats the same one-parameter regression under four definitions of
which positions matter:

| Report row | Margin compared | Digital weight at `(t,u)` | Question |
|---|---|---|---|
| `emission: reference token` | Next reference token vs blank | `label_weights[t,u]` | What penalty matches the correct token where digital is likely to emit it? |
| `emission: all nonblank` | All nonblank vs blank | `label_weights[t,u]` | What penalty matches general blank/token separation at likely emissions? |
| `blank: all nonblank` | All nonblank vs blank | `blank_weights[t,u]` | What penalty matches blank/token separation where digital is likely to wait? |
| `path: all nonblank` | All nonblank vs blank | `blank_weights[t,u] + label_weights[t,u]` | What penalty matches positions visited on the digital reference paths? |

In the path row, `label_weights[t,U]` is understood to be zero. Blank and
emission probabilities at one `(t,u)` are **weights for one observation** of
the same margin, not two independent observations. No hard alignment frame is
chosen. Nor is every cell of the rectangular `[T,U+1]` lattice weighted
equally: positions unlikely to be visited get little weight.

An emission-only fit may favor a large positive penalty while making blank
too expensive at waiting frames. The blank and path rows expose that
trade-off. A path fit naturally contains more blank than label transitions
when `T` greatly exceeds `U`; it does *not* give the two transition types
equal total influence.

## 5. Fit one constant and report errors

For one row, write `d[t,u] = digital margin - on-device margin` and let
`w[t,u]` be its weight from the table above. The script minimizes the
weighted squared difference between `d` and one constant `b`. Thus:

```text
fit b = sum(w * d) / sum(w)
```

There are two ways to combine utterances, and the report includes **both**:

| Averaging | Computation | Relative influence |
|---|---|---|
| `equal utterances` | Compute the weighted mean within each utterance; average those means | Every utterance has equal total weight |
| `pooled transitions` | Sum weighted differences over all utterances; divide by their total weight | Longer utterances contribute more transitions |

For pooled emission fits, one reference token contributes total weight one,
so an utterance with twice as many tokens contributes twice as much. Pooled
path fits weight exactly by encoder-frame count; pooled blank fits by
`T-U`. With equal utterances, weights are normalized *within* each utterance
before combining them. `RMSE before` is the square root of the weighted mean
of `d` squared. `RMSE after` is the same quantity for `d - fit b`. Each RMSE
uses **its row's own weighting scheme**, so compare before/after within a row.

The report has eight rows in total: four objectives times two averaging
methods. It is written as `blank_penalty_fit.txt`; a machine-readable copy is
written as `blank_penalty_fit.json` under the preparation's output prefix.

## 6. Interpreting a result

For example, the saved full-corpus `dev-other` report contains 2,864
utterances. Its pooled `emission: reference token` estimate is `+7.476`:
at frames where the digital reference alignment expects a token, the
on-device model's reference-token-vs-blank margin is, on average, about
7.476 log-score units below digital's. By contrast, its pooled
`blank: all nonblank` estimate is `-6.517`, and the pooled path estimate is
`-2.622`. These are **different objectives**, not competing measurements of
one universally optimal penalty. In particular, the negative path estimate
should not automatically be used for decoding when the emission problem
calls for a positive adjustment.

Opposite-signed emission and blank estimates indicate that a single constant
cannot match both types of decision well. A changed score sharpness, timing
differences, or other distortions may contribute; the fit alone does not
identify the cause. Score matching on teacher-forced, reference-conditioned
paths does not guarantee better free decoding: other tokens and erroneous
histories compete, and changing the penalty changes the decoder's paths.
Choose candidate penalties based on these diagnostics, then assess WER and
insertion/deletion behavior using the intended decoding settings.
