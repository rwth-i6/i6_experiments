# Analytical Blank-Penalty Selection by Decision Risk

For a source-order explanation of every executable statement, including line
numbers, NumPy array shapes, and Sisyphus wiring, see
[`OPTIMAL_BLANK_PENALTY_WALKTHROUGH.md`](OPTIMAL_BLANK_PENALTY_WALKTHROUGH.md).

This document describes the method implemented in
[`optimal_blank_penalty.py`](optimal_blank_penalty.py), including its
`OptimalBlankPenaltyJob` Sisyphus integration. Equations use plain text and
Python-style notation.

For the earlier score-margin regression, see
[`BLANK_PENALTY_METHODOLOGY.md`](BLANK_PENALTY_METHODOLOGY.md). For score extraction
and the reference-conditioned alignment calculation, see
[`ANALYSIS_WALKTHROUGH.md`](ANALYSIS_WALKTHROUGH.md).

## 1. The question this method answers

The method asks:

> Given the digital model's reference-conditioned alignment weights, which
> constant blank penalty minimizes the offloaded model's expected local
> blank/token decision error at the same frames and correct token histories?

It is an analytical optimizer for a **specified token-transition error
objective**. It is not an analytical optimizer of free-decoding word error
rate (WER).

The distinction matters: the diagnostic knows the correct reference history
at every evaluated state. Actual decoding has to construct that history and
can leave the reference trajectory after an error.

The saved score directories have these meanings:

| Directory | Model |
|---|---|
| `digital_model` | Digital QAT reference model |
| `base_model` | Fully offloaded model whose blank penalty is being selected |

`base_model` is the device target, despite its name. The reference is not a
separate full-precision model.

## 2. Why replace emission-margin matching?

The earlier regression estimates an average difference between digital and
device token-versus-blank margins. Its emission-only objective asks how much
to penalize blank to restore the digital margin where a reference token should
be emitted.

That does not charge the penalty for emitting tokens where blank is needed.
It also does not explicitly test whether the best competing device token is
the correct reference token.

The saved full-corpus regression illustrates the trade-off:

- Emission/reference-token matching requests approximately `+7.36` with equal
  utterance weighting.
- Blank/all-nonblank matching requests approximately `-6.33`.

A single constant cannot reproduce both kinds of margin simultaneously.
The decision-risk method balances their consequences rather than trying to
reproduce the digital model's confidence values.

## 3. Inputs and paired states

Each utterance has paired NPZ files with:

| Field | Shape | Meaning |
|---|---|---|
| `logits` | `[T, U + 1, V]` | Scores for every symbol at each frame/reference prefix |
| `labels` | `[U]` | Correct reference BPE token IDs |
| `blank_label` | scalar | Blank ID, required to be the last vocabulary entry |

Here:

- `T` is the encoder-frame count.
- `U` is the reference-token count.
- `V` includes all nonblank symbols and blank.
- State `(t, u)` means frame `t` with the correct reference prefix of length `u`.
- At prefix `u < U`, the next correct token is `labels[u]`.
- Prefix `u = U` is the state after the final reference token.

Both models score the same audio, frame, and reference history. They are not
evaluated using their independently decoded histories.

`analyze_scores()` checks that manifests are nonempty and contain identical
utterance sets, filenames match, paired lattice shapes and labels match, and
the blank IDs agree. It also validates the lattice dimensions, token IDs,
reference length, and finite logits. Extraction must use the intended
unpenalized scores; the optimizer does not undo a penalty already baked into
its inputs.

## 4. Obtain digital blank and emission weights

The implementation reuses `log_softmax()` and `transition_posteriors()` from
`fit_blank_penalty.py`.

Digital log probabilities feed forward-backward over paths that produce the
complete reference transcript. This diagnostic uses the time-synchronous
monotonic topology:

```text
Blank transition:
    Consume one frame, keep the same reference prefix.

Label transition:
    Consume one frame, emit the next reference token, advance the prefix.
```

Both transitions advance time. This differs from a conventional RNN-T topology
that permits multiple label emissions without consuming another frame.

At each state, forward-backward supplies:

```text
blank_weight[t, u]:
    Posterior probability of taking the blank transition at this state,
    conditioned on eventually producing the complete reference transcript.

label_weight[t, u]:
    Posterior probability of taking the next-reference-token transition here,
    under the same condition.
```

These are **transition posterior masses**, not simply the model's local
`P(blank)` and `P(next token)`. They include the probabilities of reaching the
state and completing the reference afterward.

Important normalization properties are:

```text
For each frame:
    sum over prefixes of (blank_weight + label_weight) = 1

For each reference token:
    sum over frames of its label_weight = 1

For the whole utterance:
    total label_weight = U
    total blank_weight = T - U
    total transition weight = T
```

The code checks the first two properties. The label weight at final prefix
`u = U` is defined as zero; blank transitions after the final token are
included.

The digital weights are computed once and held fixed while evaluating all
device penalties. They are a model-derived soft alignment, not independent
timing annotations or the device's actual free-decoding occupancy.

## 5. Derive the device's switching threshold

The device scorer uses negative log probabilities, with lower costs better:

```python
costs = -log_softmax(device_logits)
costs[blank] += blank_penalty
```

For beam-size-one local decisions, first find the best nonblank token:

```python
best_token = argmax(device_logits[:-1])
best_token_logit = max(device_logits[:-1])
blank_logit = device_logits[-1]
```

Adding a penalty to blank cannot change the ordering of the nonblank tokens.
Their best token stays the same for every candidate penalty.

The common log-softmax normalizer cancels in the cost comparison. Therefore:

```python
switching_threshold = blank_logit - best_token_logit

emit_nonblank = blank_penalty > switching_threshold
```

The diagnostic chooses blank at exact blank/nonblank ties. Nonblank ties use
NumPy's first `argmax`, which selects the lowest tied token ID.

Example:

```text
Blank logit:       10.0
Best token logit:   7.5
Best token:        "the"

Switching threshold: 10.0 - 7.5 = 2.5

Penalty 2.0: choose blank.
Penalty 2.5: choose blank under this diagnostic's tie rule.
Penalty 3.0: emit "the".
```

The threshold uses the **best individual nonblank token**, not the sum of
nonblank probabilities. Summed nonblank mass does not determine which symbol
a greedy decoder selects.

## 6. Define the local error objective

The reference-conditioned transition at a state can be blank or the next
reference token. The device action is blank or its best nonblank token.

| Reference transition | Device action | Component charged |
|---|---|---|
| Blank | Blank | None |
| Blank | Any nonblank token | False emission |
| Next reference token | Blank | Missed emission |
| Next reference token | Correct token | None |
| Next reference token | Wrong token | Wrong-token emission |

At each state:

```python
correct = best_token == next_reference_token
emit = blank_penalty > switching_threshold

false_emission = blank_weight * emit
missed_emission = label_weight * (not emit)
wrong_token = label_weight * emit * (not correct)
```

At the final prefix, `label_weight` is zero and there is no next reference
token. Any nonblank emission there contributes only false-emission mass.

Aggregate the three components using the corpus weighting in section 9.
The total risk is:

```python
risk = (
    false_emission_cost * false_emission
    + missed_emission_cost * missed_emission
    + wrong_token_cost * wrong_token
)
```

Defaults assign cost `1.0` to every component. With these defaults and the
per-frame normalization below, risk is an expected symbol-decision error
rate between zero and one, apart from numerical roundoff.

These components are analogous to insertion, deletion, and substitution,
but they are **not transcript edit-distance counts**. A local missed emission
might be recovered at a later frame in real decoding. A local false emission
might alter subsequent histories and cause additional errors. BPE errors also
do not map one-to-one to word errors.

## 7. Turn each threshold into an error-change event

Imagine starting below every switching threshold. All evaluated states choose
blank. The initial component totals are:

```python
initial_false = 0
initial_missed = sum(label_weight)
initial_wrong = 0
```

When increasing the penalty past a state's threshold, its action changes from
blank to its best nonblank token. The component changes are:

```python
change_false = blank_weight
change_missed = -label_weight
change_wrong = label_weight * (not correct)
```

Consequently, with unit costs:

```python
if correct:
    change_in_risk = blank_weight - label_weight
else:
    change_in_risk = blank_weight
```

A correct-token switch can help or hurt depending on which transition has
more posterior mass. A wrong-token switch cannot reduce the unit-cost loss:
it replaces a missed emission with a wrong emission and adds false-emission
mass wherever the reference expects blank.

For custom costs:

```python
change_in_risk = (
    false_emission_cost * blank_weight
    - missed_emission_cost * label_weight
    + wrong_token_cost * label_weight * (not correct)
)
```

Costs must be finite and nonnegative. If wrong-token and missed-emission costs
differ, a wrong-token switch can improve or worsen that custom objective;
this reflects the chosen costs, not improved token correctness.

## 8. Find the optimum by sorting, not a penalty grid

Each state switches exactly once. Between adjacent switching thresholds,
every evaluated decision remains unchanged. The objective is therefore
piecewise constant.

The implementation:

1. Creates thresholds and component changes with `decision_events()`.
2. Removes states whose blank and label posterior weights are both zero.
3. Sorts thresholds and combines exactly equal thresholds with `group_events()`.
4. Groups events within each utterance, then groups them across utterances.
5. Cumulatively adds the component changes to the all-blank baseline.
6. Finds the smallest risk among the states allowed by the penalty bounds.
7. Reports all numerically tied minimum-risk intervals.

Every possible decision pattern induced by a constant penalty is represented
in this sweep. No arbitrary grid spacing can skip an optimal interval. The
global minimum is exact for the fixed-state objective, subject to the saved
numerical scores, posterior arithmetic, and numerical tie tolerance.

There is no assumption that the risk is convex or that it has a unique
minimum. Different, nonadjacent intervals can tie.

### Worked event example

Consider three illustrative weighted decision events, all with unit costs:

| Threshold | Blank weight | Label weight | Best token correct? | Risk change |
|---:|---:|---:|---|---:|
| 1.0 | 0.1 | 0.9 | Yes | -0.8 |
| 2.0 | 0.9 | 0.1 | Yes | +0.8 |
| 3.0 | 0.2 | 0.8 | No | +0.2 |

The all-blank risk is `0.9 + 0.1 + 0.8 = 1.8`. Sweeping gives:

| Penalty region | Unnormalized risk |
|---|---:|
| At or below 1.0 | 1.8 |
| Above 1.0, at or below 2.0 | **1.0** |
| Above 2.0, at or below 3.0 | 1.8 |
| Above 3.0 | 2.0 |

The optimal interval is `(1.0, 2.0]`, not a uniquely optimal scalar. The third
switch is harmful even though it removes missed-emission mass: the emitted
token is wrong.

### Computational cost

Building events requires examining the device vocabulary scores at every
lattice state; the digital forward-backward calculation also visits the
lattice. Sorting `M` active events takes approximately `M * log(M)` work.
Grouping duplicate thresholds before corpus aggregation reduces the final
event table, which is particularly useful with quantized outputs. Peak memory
still includes the current score lattice and retained grouped event arrays.

## 9. Corpus weighting and report units

The method reports two objectives.

### Pooled transitions

```python
pooled_component = sum(utterance_component_totals) / sum(utterance_frame_counts)
```

Each encoder frame has equal total influence. Longer utterances contribute
more total weight. The denominator is the total frame count, not the number
of rectangular lattice cells or reference tokens.

### Equal utterances

```python
equal_utterance_component = mean(
    utterance_component_total / utterance_frame_count
)
```

Each utterance contributes equally after its transition masses are divided
by its own frame count. This is an average of **per-utterance per-frame
errors**, not an average of raw utterance error counts.

In both cases, the three reported components are expected errors per frame.
For example, risk `0.09438` with unit costs means approximately 9.438% expected
local transition error under the specified weighting. It does not mean 9.438%
WER. With custom costs, risk is a weighted cost per frame rather than an
ordinary error fraction.

### Correct best-nonblank emission coverage

The report also measures whether the device's best nonblank token is the
correct reference token at digitally weighted emissions:

```python
coverage = weighted_correct_label_mass / weighted_total_label_mass
```

For pooled coverage, the masses are summed directly. For the equal-utterance
version, each utterance's masses are first divided by its frame count and
then summed. The latter is **not** the arithmetic mean of each utterance's
individual coverage ratio.

Coverage is independent of blank penalty. Low coverage shows errors that a
blank penalty cannot repair, because changing blank does not reorder tokens.
If there are no reference labels, coverage is undefined and saved as `null`.

## 10. Bounds, endpoints, and representative values

Defaults are:

```python
min_penalty = 0.0
max_penalty = None
penalties = (0.0, 2.0, 4.0, 7.3)
costs = (1.0, 1.0, 1.0)
```

The lower bound must be finite. Negative lower bounds are allowed. The upper
bound, if supplied, must be finite and at least the lower bound. Without an
upper bound, the search includes the final all-emitting interval.

Negative thresholds are handled correctly: their events have already been
crossed when starting the bounded search at penalty zero.

Because blank wins ties, an interval between thresholds is:

```text
(lower_threshold, upper_threshold]
```

It excludes the lower endpoint and includes the upper endpoint. The first
search interval includes `min_penalty`. An unbounded last interval has an open
infinite endpoint. A search with identical lower and upper bounds evaluates
that single penalty.

The representative penalty is usually the midpoint of a finite optimal
interval. For an unbounded interval, the code selects a finite value above
its lower endpoint. It is a convenient member of the interval, not a second
optimization criterion or an independently precise estimate.

Comparison penalties are evaluated even when outside the optimization bounds.
The `penalties` argument controls report comparisons; it is not the search grid.

## 11. Quantization and floating-point precision

The current joint network includes a final activation quantizer. Consequently,
device logits occupy a finite score grid, and switching thresholds cluster
around differences of those grid levels.

In an inspected full-corpus lattice, adjacent logit levels were spaced by
approximately `0.4822836`. Five such steps give a threshold near `2.411417`.

The saved scores are `float32`. The implementation converts them to `float64`
before subtracting, but conversion cannot recover the exact underlying
quantization-grid values. Nominally equivalent gaps can therefore differ:

```text
2.411417007446289
2.411417484283447
2.411417961120606
```

**Current behavior:** `group_events()` groups only exactly equal numerical
thresholds. It does not merge nearby thresholds by tolerance or infer a
quantization scale. These three numbers remain separate events.

The reported interval:

```text
(2.41141701, 2.41141748]
```

can therefore exploit a numerical split within one nominal quantization
boundary. It is meaningful for the saved numerical logits and diagnostic
tie rule, but does not establish a physically stable penalty to eight decimal
places. The deployed scorer's finite-precision log-softmax and cost addition
can change such near-tie decisions even though the logit derivation is exact
in real arithmetic.

The saved first-fifth and full-corpus reports both select this interval, with
139 and 142 unique numerical thresholds, respectively. This is possible
because both datasets share the quantization grid and choose the same
decision boundary. Their event weights and minimum risks are different; the
matching interval does not imply that the same data or report was reused.

A future quantization-aware diagnostic could merge nominally identical
boundaries and report behavior immediately below and above each group. Such
grouping would change the numerical decision model and must be explicitly
defined. It is **not implemented** by the current job.

Other numerical conventions in the current implementation:

- Zero posterior masses, including numerical underflow to zero, are omitted.
- Negative component residuals from cumulative roundoff are clamped to zero.
- Minimum risks are considered numerically tied using both relative and
  absolute tolerances of `1e-12`.
- That risk tie tolerance is not a threshold-grouping tolerance or a
  statistical confidence interval.

## 12. Sisyphus integration

The optimizer consumes extraction outputs through `tk.Path` dependencies.
Its entrypoint is:

```python
from diagnostics.qat_ffnn_scores.optimal_blank_penalty import run_optimal_blank_penalty

prepared = prepare_scores(...)
job = run_optimal_blank_penalty(prepared)
```

`config/scorer_debug.py` invokes it for both `first_fifth` and `full_corpus`
preparations, alongside the earlier regression job.

For example, customize the objective and search range with:

```python
job = run_optimal_blank_penalty(
    prepared,
    costs=(1.0, 1.0, 1.0),  # False emission, missed emission, wrong token.
    min_penalty=0.0,
    max_penalty=8.0,
    penalties=(0.0, 1.5, 2.0, 2.5, 3.0, 4.0, 7.3),
)
```

`OptimalBlankPenaltyJob` has one `run` task requesting two CPUs, 16 GB memory,
and two hours. It produces:

- `out_summary`: `optimal_blank_penalty.json`.
- `out_report`: `optimal_blank_penalty.txt`.

The helper adds alias `<output_prefix>/optimal_blank_penalty` and registers
both files under `prepared.output_prefix`. It uses the `digital_model` and
`base_model` entries in `prepared.score_dirs`.

The JSON includes input paths, utterance/frame counts, threshold count, costs,
bounds, tie conventions, and both averaging results. Each result contains the
minimum risk, all optimal intervals, their error components and representative
penalties, comparison rows, and correct-best-token coverage.

The text report displays all optimal intervals and uses the first interval's
representative value for its final representative-penalty line. If there are
multiple optima, inspect all intervals or the JSON rather than reading only
that last line.

## 13. How to interpret the current results

The saved full-corpus pooled report contains:

| Penalty | Total risk | False emission | Missed emission | Wrong token |
|---|---:|---:|---:|---:|
| 0 | 0.166095 | 0.004924 | 0.154992 | 0.006179 |
| 2 | 0.094664 | 0.032912 | 0.046507 | 0.015244 |
| Selected interval near 2.411417 | 0.094380 | 0.041975 | 0.036258 | 0.016147 |
| 4 | 0.135037 | 0.105077 | 0.010321 | 0.019639 |
| 7.3 | 0.542240 | 0.519715 | 0.000328 | 0.022197 |

The pattern explains the trade-off:

1. No penalty leaves many digitally expected emissions classified as blank.
2. A moderate penalty recovers substantial emission mass.
3. Larger penalties increasingly emit where the reference alignment expects blank.
4. At 7.3, almost all missed-emission mass disappears, but false emissions dominate.

The gain from penalty 2 to the selected numerical interval is small:
approximately `0.000284` expected errors per frame, or 0.0284 percentage points.
That small surrogate improvement does not by itself establish a WER gain.

## 14. Relationship to sequence length

In this time-synchronous topology, a complete path with `T` frames and `L`
emitted tokens has exactly `T - L` blank transitions. Under the unnormalized
additive blank-cost scheme:

```text
penalized_path_cost = original_path_cost + penalty * (T - L)
```

For a fixed utterance, `penalty * T` is common to every complete path, so the
same expression can be viewed as an output-token reward:

```text
relative_path_cost = original_path_cost - penalty * L
```

For a fixed transcript, every complete alignment has the same token count;
the blank penalty alone does not change their relative full-path costs.
Across different transcripts, it rewards longer token sequences. This helps
explain why an excessive penalty can produce insertion-heavy decoding.

The local diagnostic still changes decisions at fixed reference states. It
does not search whole competing transcripts or simulate their histories.

## 15. Scope, validation, and limitations

The analytical claim is precise: sorting thresholds finds the global minimum
of the defined local risk over a constant penalty and the allowed bounds.
It does not make the digital alignment correct ground truth or make local
error equivalent to transcript error.

Main limitations are:

- **Reference-conditioned timing:** Digital forward-backward is conditioned
  on completing the correct transcript. It can allocate emissions differently
  from unconstrained decoding.
- **Fixed correct histories:** Device mistakes do not change subsequent
  diagnostic states. Actual prediction-network histories do change.
- **Local rather than sequence loss:** Errors are scored at weighted states,
  not by aligning complete predicted transcripts to references.
- **Token rather than word units:** BPE transition errors are not WER.
- **Greedy decision model:** The threshold derivation targets a beam-size-one
  blank/token choice without additional token-dependent LM or ILM terms.
  Beam search and other scoring terms need their own objective or adjusted
  decision scores.
- **Tie assumptions:** The diagnostic's tie rules need not match all decoder
  implementations, especially around quantized score ties.
- **One hardware realization:** The job fits the supplied device score
  extraction. Robustness across cycles or realizations requires comparing
  their results or explicitly aggregating their events.

Implementation verification compared the event-sweep component totals against
direct decisions at 2,734 candidate penalties, checked selected optima in 100
random synthetic problems, and exercised bounded and unbounded searches.
The Sisyphus job's output registration and report serialization were also
checked using three actual paired score files. These checks validate the
implemented calculation, not a claim of WER optimality.

For experimental use, interpret the optimum alongside its component trade-off
and quantization boundary, then compare candidate penalties with free-decoding
WER and insertion/deletion/substitution counts under the intended decoder.
