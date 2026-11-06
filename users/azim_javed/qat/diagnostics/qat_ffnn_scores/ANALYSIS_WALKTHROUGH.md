# Reference-Score Analysis Walkthrough

This document explains the analysis portion of
[`reference_scoring.py`](reference_scoring.py), currently lines 211-662. It is
intended to be read beside the source code. Exact line numbers can move as the
diagnostic evolves; the named variables and section order are the authoritative
references.

The analysis compares a digital FFNN transducer with several versions in which
different modules are mapped to memristor hardware. Its central question is:

> When the prediction network receives the correct reference history, how does
> each model score the next reference token relative to blank?

The code also infers a soft alignment between reference tokens and encoder
frames. No externally supplied forced alignment is used.

## 1. Speech And Transducer Concepts

### Audio, encoder frames, and tokens

The input waveform contains many audio samples. The encoder converts those
samples into a shorter sequence of representations called **encoder frames**.
If the encoder produces `T` frames, their indices are `0, ..., T-1`.

The transcript is represented by `U` BPE tokens rather than words. Their
indices are `0, ..., U-1`. A BPE token may represent a whole short word, part
of a word, or a frequently occurring character sequence.

There is not generally a known exact frame at which a BPE token should be
emitted. This unknown correspondence is the **alignment** problem.

### Encoder, prediction network, and joint network

The transducer has three conceptual components:

| Component | Information it sees | Purpose |
|---|---|---|
| Encoder | Audio | Produces acoustic representations over time |
| Prediction network | Previously emitted tokens | Represents linguistic history |
| Joint network | Encoder and prediction states | Scores the next vocabulary symbol |

During this diagnostic, the prediction network is teacher-forced. At reference
position `u`, it receives the correct reference prefix before token `u`.

For every encoder frame `t` and reference position `u`, the scorer produces one
logit for every vocabulary symbol. This gives a tensor with shape:

```text
[T, U + 1, V]
```

where `V` is the vocabulary size including blank. The extra reference position
is the state after the final reference token has been consumed.

### Blank

Blank means "consume this encoder frame without emitting a reference token."
It is necessary because there are usually many more encoder frames than output
tokens.

Blank being likely at most lattice positions is therefore normal. The useful
question is whether the next reference token decisively beats blank during its
plausible emission window.

### The monotonic transducer topology used here

At each encoder frame, this implementation takes one of two relevant
transitions:

```text
blank transition: keep the same reference position
label transition: emit the next reference token and advance one position
```

Both transition types consume one encoder frame. Consequently, a valid path
through the reference lattice chooses `U` frames at which to emit the `U`
tokens, in order; all other frames emit blank.

For a reference `A B` and four encoder frames, valid alignments include:

```text
A B _ _
A _ B _
A _ _ B
_ A B _
_ A _ B
_ _ A B
```

The code does not choose one of these paths in advance. It sums over all valid
paths with forward-backward.

### Logits, probabilities, and log probabilities

A **logit** is an unrestricted model score. It is not yet a probability.
Softmax converts the `V` logits at one lattice position into probabilities that
sum to one:

```text
P(k) = exp(logit[k]) / sum_j exp(logit[j])
```

The analysis performs most dynamic programming in log space. Multiplication of
many small probabilities then becomes addition:

```text
log(a * b) = log(a) + log(b)
```

This avoids numerical underflow on long utterances.

### Oracle terminology in this file

`oracle_probability` means the probability assigned to the **next reference
token**. It does not mean that an oracle timing alignment is available.

The transcript is known, but the frame at which each token should be emitted
is not.

## 2. Saved Inputs To The Analysis

The immediately preceding callback writes one NPZ file per utterance. Lines
190-197 save:

| Field | Shape | Meaning |
|---|---:|---|
| `logits` | `[T, U+1, V]` | Raw scorer outputs |
| `labels` | `[U]` | Reference BPE token IDs |
| `encoder_length` | scalar | Number of valid encoder frames |
| `label_length` | scalar | Number of reference tokens |
| `blank_label` | scalar | Vocabulary ID used for blank |

Lines 199-208 add each file to `manifest.json`, which lets the analysis find
the same utterances for every model.

Separately, `CorpusToTextDictJob` reads the original Bliss corpus orthography
for the selected segments. The analysis uses this dictionary to put the full
reference transcript in each figure title. The text comes from the corpus, not
from trying to reconstruct words from BPE token IDs.

## 3. Forward-Backward Function

### Function header and log-softmax: lines 211-216

```python
def _monotonic_rnnt_alignment_posteriors(logits, labels, blank_label):
```

Line 211 defines a helper that returns:

```text
log_likelihood:    log P(reference transcript | audio, model)
label_posteriors:  q[t, u], the posterior probability that token u is emitted at frame t
```

Line 212 converts logits to `float64` and subtracts the largest logit at every
`(t, u)` position:

```python
shifted_logits = logits.astype(np.float64) - np.max(logits, axis=-1, keepdims=True)
```

Subtracting the maximum changes no softmax probabilities, because the same
constant is subtracted from every vocabulary entry. It prevents `exp(logit)`
from overflowing.

Line 213 computes log-softmax:

```python
log_probs = shifted_logits - log(sum(exp(shifted_logits)))
```

Thus `log_probs[t, u, k]` is the log probability of vocabulary symbol `k` at
encoder frame `t`, given reference prefix `u`.

Lines 214-215 extract `T` and `U`. Line 216 checks that the lattice contains
exactly one prediction state before every token plus one after the final token:

```text
reference_positions == U + 1
```

### Forward variables: lines 218-228

The forward table has shape `[T, U+1]`:

```python
alpha[t, u]
```

Its meaning is:

> Log probability of all valid paths that have consumed frames `0..t` and
> emitted exactly `u` reference tokens.

Line 218 initializes every state to negative infinity. In log space,
`-infinity` represents zero probability.

Lines 219-220 visit every frame `t` and reference count `u`.

Line 221 finds the predecessor for a blank transition:

```python
previous_blank = alpha[t - 1, u]
```

A blank consumes the current frame but emits no token, so `u` does not change.
The special case `t == 0 and u == 0` uses log probability `0.0`, which means
ordinary probability `1.0`. This is the start state before any frame or token.

Line 222 adds the current blank log probability:

```python
blank_path = previous_blank + log_probs[t, u, blank_label]
```

Line 224 obtains the predecessor for a label transition. To end with `u`
emitted labels, the path previously had `u-1` emitted labels:

```python
previous_label = alpha[t - 1, u - 1]
```

The special case `t == 0 and u == 1` permits the first token to be emitted on
the first encoder frame.

Lines 225-227 add the score of reference label `labels[u-1]`. If `u == 0`, no
label can have been emitted, so this path receives `-infinity`.

Line 228 combines the blank and label paths:

```python
alpha[t, u] = np.logaddexp(blank_path, label_path)
```

`logaddexp(a, b)` computes `log(exp(a) + exp(b))` stably. It therefore sums the
probabilities of both ways of reaching the state.

Repeated application means `alpha` sums over every valid alignment rather than
retaining only the best alignment.

### Sequence likelihood: line 230

```python
log_likelihood = alpha[-1, -1]
```

The final state has consumed every encoder frame and emitted every reference
token. It is therefore:

```text
log P(reference transcript | audio)
```

with all valid alignments marginalized out.

### Backward variables: lines 231-239

The backward table uses the same `[T, U+1]` shape. Its meaning is:

> `beta[t, u]` is the log probability of completing the reference after the
> state that has consumed through frame `t` and emitted `u` tokens.

Line 231 initializes all states to `-infinity`.

Line 232 sets the completed final state to log probability zero, or ordinary
probability one. Once all frames and labels are consumed, there is nothing left
to score.

Line 233 walks backward from the penultimate frame. Lines 234-239 evaluate the
two possible transitions at frame `t+1`:

```text
blank: remain at u
label: advance from u to u+1 by emitting labels[u]
```

Line 235 scores the blank continuation. Lines 236-238 score the label
continuation when `u < U`. Line 239 uses `logaddexp` to sum both possible
suffixes.

The forward table summarizes all possible prefixes. The backward table
summarizes all possible suffixes.

### Label-transition posterior: lines 241-247

`label_posteriors[t, u]` is defined as:

```text
P(reference token u is emitted at frame t | audio, complete reference, model)
```

Line 241 allocates a `[T, U]` table. There is no column for the final prediction
state because there is no next reference token after `u == U-1`.

Lines 242-243 iterate over candidate emission frames and tokens.

Line 244 obtains the total probability of reaching the source state before the
label transition. The special initial state handles token zero at frame zero.

Lines 245-247 calculate:

```text
q(t,u) = exp(
    log probability of all prefixes reaching the transition
  + log probability of emitting labels[u]
  + log probability of all valid suffixes after the transition
  - log probability of the complete reference
)
```

Dividing by the complete-reference probability, implemented by subtracting its
log, conditions the alignment distribution on the reference being produced.

For each token `u`, the posterior should satisfy:

```text
sum_t q(t,u) = 1
```

Line 249 returns the marginalized sequence likelihood and the soft alignment.

### What this posterior is and is not

It is a **model-derived, reference-conditioned soft alignment**. Different
models can produce different posterior alignments.

It is not:

- A human annotation.
- A forced alignment from an independent acoustic system.
- A guarantee of what free decoding will do.
- One hard path through the lattice.

Because it is conditioned on completing the reference, it preferentially
weights frames at which the reference token can be emitted. This is important
when interpreting model-centered margins.

## 4. Analysis Job Setup

### Constructor: lines 252-265

Line 252 declares a Sisyphus job.

Lines 253-255 accept a mapping from model names to score directories, the Bliss
transcript dictionary, and a reference model. In the current experiment,
`digital_model` is passed as the reference when the job is instantiated.

Lines 256-258 store those inputs as job state.

Lines 259-265 declare five outputs:

| Output | Contents |
|---|---|
| `summary.json` | Full structured metrics, including per-token expected frames |
| `summary.tsv` | Flat table for quick inspection |
| `margin_heatmaps/` | Complete `T x U` margin lattice |
| `focused_margin_heatmaps/` | Windows centered on digital expected frames |
| `model_centered_margin_heatmaps/` | Windows centered on each model's expected frames |

### Task resources: lines 267-268

The job asks Sisyphus for two CPU cores, 8 GB of memory, one hour, and a small
GPU allocation. The analysis itself is mostly NumPy and plotting; the GPU
request is associated with loading/running the native loss in this environment.

### Transcript and native loss loading: lines 270-276

The `run` method performs the work. `ast.literal_eval` safely reads the Python
dictionary emitted by `CorpusToTextDictJob`. The method also imports the same
native monotonic RNN-T loss used by training. That implementation provides an
independent likelihood against which the handwritten forward recursion is
checked.

## 5. Loading And Matching Model Outputs

### Manifests: lines 278-285

Lines 278-281 load `manifest.json` from every model's score directory.

Line 283 takes the set of utterance tags from the reference model. Line 284
asserts that every model scored exactly the same utterances. Comparing tensors
from different utterances would otherwise produce meaningless results. Line
285 also verifies that corpus text is available for every selected segment.

### Display names and per-utterance storage: lines 287-302

Lines 287-293 define the human-readable model names used consistently in plots,
JSON, and TSV. `summary` will eventually be written as JSON.
`margin_heatmaps` retains the full label-versus-blank margin matrices for
plotting.

For each utterance, lines 297-302 allocate dictionaries holding:

| Dictionary | Stored object |
|---|---|
| `model_results` | Scalar and list metrics for JSON/TSV |
| `lattices` | Raw logits |
| `expected_emission_frames` | One posterior mean frame per token |
| `alignment_posteriors` | Full `[T,U]` soft alignments |
| `blank_probabilities_by_model` | `P(blank)` at every `[t,u]` |
| `oracle_probabilities_by_model` | `P(reference token u)` at every `[t,u]` |

### NPZ loading: lines 303-312

Line 303 loops over models. Line 304 looks up the utterance's NPZ filename.
Lines 305-308 load logits, labels, and the blank ID with explicit data types.

Lines 310-311 recover `T`, `U+1`, `V`, and `U`. Line 312 repeats the shape
consistency check.

## 6. Extracting Blank And Reference Probabilities

### Softmax: lines 314-317

Lines 314-316 apply numerically stable softmax to the logits.

Line 317 selects the blank vocabulary column for all real reference positions:

```text
blank_probabilities[t,u] = P(blank | frame t, reference prefix u)
```

Its shape is `[T,U]`.

### Gathering the next reference token: lines 318-322

Every reference position has a different target token. The code broadcasts
`labels[u]` across all `T` frames, then `np.take_along_axis` selects that token's
probability from the vocabulary dimension:

```text
oracle_probabilities[t,u]
  = P(labels[u] | frame t, reference prefix u)
```

This also has shape `[T,U]`.

These two matrices are the direct realization of the original diagnostic idea:
feed the correct reference history and compare the next reference token with
blank everywhere in the lattice.

## 7. Sequence Probability And Alignment Validation

### Native loss: lines 324-332

Line 324 flattens `[T,U+1,V]` to the format expected by the native operation.
`requires_grad_(True)` is required by this native operation even though this
analysis only reads the loss value.

Lines 325-331 pass the logits, reference labels, lengths, and blank ID to the
native monotonic RNN-T loss.

Loss is negative log likelihood:

```text
loss = -log P(reference | audio)
```

Line 332 negates it to recover log probability.

### Handwritten forward-backward check: lines 333-339

Lines 333-335 run the NumPy forward-backward implementation.

Lines 336-339 require its log likelihood to match the native loss within
`1e-3`. This is an important correctness check on the recurrence and boundary
conditions. A disagreement would invalidate all posterior-derived metrics.

## 8. Model-Specific Alignment Metrics

### Expected emission frame: lines 340-342

For token `u`, the expected emission frame is:

```text
E[t_u] = sum_t q(t,u) * t
```

This posterior mean can be fractional. It is not necessarily an actual frame,
a Viterbi alignment, or the mode of the posterior. If a posterior has two
distant peaks, its mean can lie between them.

### Alignment entropy: lines 343-345

For each token, entropy is:

```text
H(u) = -sum_t q(t,u) log q(t,u)
```

Low entropy means the model concentrates emission probability in a narrow
frame region. High entropy means it is uncertain across more frames.

`np.maximum(..., tiny)` prevents taking `log(0)`.

### Posterior-weighted probabilities: lines 346-351

These lines calculate, for each token:

```text
sum_t q(t,u) P(reference token u at t)
sum_t q(t,u) P(blank at t,u)
```

These metrics evaluate each model around its own reference-conditioned
alignment posterior.

### Posterior-weighted log margin: lines 352-359

At each lattice position the label-versus-blank log margin is:

```text
m(t,u) = log P(reference token u at t) - log P(blank at t,u)
       = log(P(reference token) / P(blank))
```

Interpretation:

| Margin | Meaning |
|---:|---|
| `> 0` | Reference token is more likely than blank |
| `= 0` | Equal probability |
| `< 0` | Blank is more likely |
| `+5` | Reference token has `exp(5)` times blank's probability |

Lines 352-359 average this margin over frames using the model's own alignment
posterior, producing one value per reference token.

This is an expectation of a log ratio. It is not equal to the log ratio of the
two separately averaged probabilities.

### Result fields: lines 360-387

Lines 360-387 package the model's metrics.

`model_display_name` stores the descriptive model name alongside the internal
dictionary key, so downstream readers do not need to reproduce the mapping.

`log_probability` is the complete reference score. Less negative is better.

`log_probability_per_token` divides by `U`, making utterances of different
lengths more comparable.

`mean_blank_probability` and `mean_oracle_token_probability` average the
entire `T x U` lattice without alignment weighting. Most cells are not plausible
emission points, so these are broad descriptive statistics rather than the
strongest diagnostic.

`blank_favored_over_oracle_fraction` is also unweighted over the complete
lattice. It is normally high because blank should dominate away from a token's
small emission window.

`mean_expected_emission_frame` averages token timing within the utterance.

`mean_expected_emission_frame_normalized` divides each expected frame by
`T-1`, mapping time approximately into `[0,1]`.

`expected_first_token_emission_frame` records the first token's timing.

`mean_alignment_posterior_entropy` summarizes alignment sharpness.

The three `mean_emission_weighted_*` fields summarize token probability, blank
probability, and their log margin under that model's own soft alignment.

`expected_emission_frames` preserves every token's posterior mean for plotting
and cross-model comparisons.

Lines 388-392 retain arrays needed after all models for the utterance have been
processed.

## 9. Shared-Reference Alignment Comparison

### Selecting the reference model: lines 394-397

These lines obtain metrics, logits, expected frames, and alignment posteriors
from `self.reference_model`. In the current experiment this is the completely
digital model.

The digital posterior is used as one shared weighting distribution for every
model. This asks:

> At frames preferred by the digital model, how does each model score the same
> reference token against blank?

### Per-model comparisons: lines 398-416

Line 399 subtracts raw logits from the reference logits. Raw-logit differences
describe numerical output changes but are not directly probability or WER
changes.

Line 400 calculates every token's expected-frame shift relative to digital.

Lines 401-402 retrieve the model's blank and reference probability matrices.

Lines 403-408 weight those probabilities by the **reference model's** posterior
rather than the evaluated model's own posterior.

Lines 409-416 similarly compute the reference-alignment-weighted log margin.

This shared alignment controls for timing. It separates two possible causes of
degradation:

```text
alignment change: the model prefers to emit at different frames
score change:     the model scores label versus blank differently at the same frames
```

### Delta fields: lines 417-437

Lines 417-422 store sequence-score differences relative to the reference model.
A positive delta means a higher reference score than the reference model for
that utterance.

Lines 423-425 summarize absolute raw-logit differences using mean absolute
error, maximum absolute error, and root mean square error.

Lines 426-428 summarize signed, absolute, and first-token alignment shifts.

Lines 429-437 average the shared-reference alignment probabilities and margin
over tokens.

### Heatmap data: lines 439-448

Line 439 stores all scalar and list results for the utterance.

Lines 440-448 construct each full margin matrix:

```text
margin_heatmap[t,u] = log P(reference token u at t) - log P(blank at t,u)
```

This matrix is not alignment-weighted. It shows the complete teacher-forced
lattice.

## 10. JSON And TSV Outputs

Lines 450-451 write the nested JSON summary. Every model object includes both
its internal key in the surrounding dictionary and `model_display_name` as a
field.

Lines 453-482 define the TSV column order, including `model_display_name`
immediately after `model`. Lines 483-488 write one row per utterance and model.

The JSON is preferable when per-token `expected_emission_frames` are needed.
The TSV is preferable for sorting and aggregate tabular analysis.

## 11. Plot Setup

### Matplotlib backend: lines 490-494

`Agg` is a noninteractive rendering backend. It allows PNG creation on cluster
workers without a display server. `textwrap` wraps long reference transcripts
across title lines.

### Model order: lines 496-503

The plots order models from digital through partial offload variants to fully
offloaded. The internal names are replaced with descriptions of which modules
remain QAT/digital and which modules execute on-device.

### Output directories and titles: lines 504-507

The three plot directories are created, then the code processes each utterance.
`figure_title` combines the segment ID with the wrapped full reference text;
the same title is used by all three chart families.

## 12. Complete-Lattice Heatmaps

### Shared color scale: lines 508-509

All model margins for one utterance are concatenated. The symmetric color limit
is the 95th percentile of absolute margin, with a minimum limit of one.

Using symmetric limits ensures that equal positive and negative margins have
equal color intensity. The percentile prevents a few extreme cells from making
the rest of the plot nearly white.

Values outside the limits saturate at the darkest colors. Also, each utterance
gets its own scale, so shades should be compared directly within one figure,
not numerically across different utterance figures.

### Figure and panels: lines 510-527

One vertical panel is created per model with shared axes.

`model_heatmaps[model_name]` has shape `[T,U]`. Line 520 transposes it to
`[U,T]`, so encoder frame is horizontal and reference-token index is vertical.

`RdBu_r` maps negative margins to blue, zero near white, and positive margins
to red.

`origin="lower"` places token zero at the bottom. `interpolation="nearest"`
shows discrete cells rather than smoothing them.

### Alignment overlay: lines 528-536

Lines 528-531 draw that model's expected emission frame against token index.
This black curve is a soft-alignment posterior mean, not a hard path.

Lines 532-536 add labels and the colorbar.

### Saving: lines 537-541

The utterance tag becomes the title and a filesystem-safe filename. The figure
is saved at 180 DPI and closed to release memory.

### Reading the full heatmap

Most of the lattice should be blue because most frames are inappropriate for a
given token. The important feature is whether a localized positive-margin
region appears near the alignment curve.

Horizontal patterns depend strongly on reference token and prediction history.
Vertical patterns reflect encoder-frame effects shared across many reference
positions.

## 13. Digital-Centered Focused Heatmaps

### Defining windows: lines 543-556

Line 543 chooses offsets `-8, ..., +8`, giving a 17-frame window.

Lines 544-546 load the digital/reference model's expected frame for every token
and round it to the nearest integer frame for array indexing.

For every model, lines 548-556 gather its margins at:

```text
round(digital expected frame for token u) + offset
```

All models are therefore viewed at the same frame windows.

Lines 551-555 handle utterance boundaries. Invalid offsets are clipped for safe
indexing and then replaced by `NaN` so they are not interpreted as real data.

### Focused color scale and rendering: lines 558-580

Finite cells from all models are used to calculate one shared, symmetric 95th
percentile color scale.

Each focused matrix has shape `[U,17]`, already in display orientation. The
`extent` argument labels columns by actual relative offsets `-8..+8` rather
than array indices `0..16`.

### Curves: lines 581-595

The black curve is:

```text
model expected frame - digital expected frame
```

The dotted vertical line at zero is the digital expected frame.

Thus the curve shows whether each model prefers an earlier or later frame than
digital for every token.

Lines 593-595 label the shared digital coordinate system and margin colorbar.
Lines 596-598 save and close the figure.

### What this view answers

This plot asks:

> At the same digital-model emission neighborhood, how do all models score the
> reference token against blank?

It is the cleaner view for isolating score changes from alignment changes.

## 14. Model-Centered Focused Heatmaps

### Defining per-model windows: lines 601-613

For each model, lines 602-607 load and round that model's own expected emission
frames. The gathered window is:

```text
round(that model's expected frame for token u) + offset
```

Boundary handling in lines 608-612 is the same as in the digital-centered
view.

### Scale and rendering: lines 615-642

All finite model-centered cells define one symmetric color scale for the
utterance. Lines 624-642 create the five panels.

The dotted zero line now means **that panel's model-specific expected emission
frame**, not the digital frame.

### Reciprocal timing curve: lines 643-657

The black curve is now:

```text
digital expected frame - model expected frame
```

It shows where the digital timing lies relative to the model-centered zero.
This is the reciprocal of the curve in the digital-centered figure.

Lines 655-657 label the model-specific coordinate system and margin colorbar.
Lines 658-662 save and close the figure.

### What this view answers

This plot asks:

> If each model is evaluated around its own reference-conditioned preferred
> timing, can it find a strong reference-token emission region?

This gives each model its most favorable timing. It therefore combines timing
and score effects and can make a degraded model look better than the shared
digital-centered comparison.

## 15. Why A Red Center Does Not Guarantee Good Decoding

The model-centered posterior is conditioned on the complete reference being
emitted. It naturally selects frames where reference transitions are relatively
favorable.

Free decoding is different:

- The decoder is not told that it must eventually emit every reference token.
- It does not know the correct reference prefix after making an error.
- Blank, other tokens, and hypotheses with different histories compete in the beam.
- A weak positive margin can be lost through accumulated scores or beam pruning.
- Some tokens can still have negative margins even if the panel looks lightly red on average.

The center heatmap cell is also only the nearest integer frame to a posterior
mean. The reported posterior-weighted metric integrates over all frames:

```text
sum_t q(t,u) m(t,u)
```

It is therefore not necessarily equal to the color of the center pixel.

## 16. Practical Interpretation Checklist

When reading these outputs, ask the following questions in order:

1. Does the full lattice show localized red regions, or is the margin structure washed out?
2. Does the alignment curve pass through those positive regions?
3. In the digital-centered view, does the model preserve positive margins at shared frames?
4. In the model-centered view, can the model recover positive margins by shifting timing?
5. Are timing shifts large enough to explain the change, or do scores change at nearly fixed timing?
6. Does the conclusion agree with free-decoding insertion, deletion, and substitution rates?

For the current experiment, the alignment shifts are small, while the joint
network offload sharply weakens the reference-token versus blank margin. The
model-centered view recovers a weak positive margin for many tokens, but the
shared digital view and high deletion rates show that this margin is not robust
enough during free decoding.

## 17. Main Limitations

The diagnostic evaluates only a small set of utterances. It is useful for
mechanistic investigation, not for estimating corpus-level accuracy.

The scorer is teacher-forced with the correct reference history. It does not
show scorer behavior after a decoding error changes the token history.

The posterior alignment is produced by the model being analyzed and is
conditioned on the reference. It is not independent acoustic ground truth.

The margin compares the reference token specifically with blank. It does not
by itself prove that blank is more likely than every other vocabulary token.

The complete-lattice averages include many intentionally irrelevant positions.
Posterior-weighted and focused metrics are usually more informative.

Finally, expected emission is a posterior mean. For a broad or multimodal
posterior, a Viterbi frame, posterior mode, or full posterior visualization may
describe timing more faithfully than the mean alone.
