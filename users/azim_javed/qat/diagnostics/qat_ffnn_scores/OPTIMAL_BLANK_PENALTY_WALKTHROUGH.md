# Optimal Blank Penalty: Line-by-Line Code Walkthrough

Read this beside [`optimal_blank_penalty.py`](optimal_blank_penalty.py).
The line numbers refer to the 315-line version of the file when this walkthrough
was written. Function names remain the useful anchors if later edits move lines.

The explanations follow source order and cover every executable statement.
Blank lines and multiline syntax are grouped where they do not introduce a
separate operation. Equations use Python-style notation, without LaTeX.

The broader motivation, assumptions, and interpretation are in
[`OPTIMAL_BLANK_PENALTY_METHODOLOGY.md`](OPTIMAL_BLANK_PENALTY_METHODOLOGY.md).

## Before reading: array shapes and vocabulary

| Name | Meaning |
|---|---|
| `T` | Encoder-frame count for one utterance |
| `U` | Correct reference BPE-token count |
| `V` | Vocabulary size including blank |
| `(t, u)` | Frame `t`, with the correct reference prefix of length `u` |
| `device_logits` | Shape `[T, U + 1, V]` |
| `labels` | Shape `[U]`; the next reference token at prefix `u` is `labels[u]` |
| `blank_weights` | Shape `[T, U + 1]`; digital blank-transition posterior masses |
| `label_weights` | Shape `[T, U]`; digital reference-token-transition posterior masses |
| `M` | Number of active state events after flattening |
| `K` | Number of distinct numerical thresholds after grouping |

A **logit** is a model score before softmax. Higher logits are better.
A decoding **cost** is a negative log probability. Lower costs are better.

The weights are not simply local token probabilities. Digital forward-backward
computes them conditioned on producing the complete reference. They include
the probability of visiting the state and completing the reference afterward.

The three error components always have this order:

```text
Column 0: false emission   — emit a token where the reference transition is blank
Column 1: missed emission  — choose blank where a reference token should be emitted
Column 2: wrong token      — emit the wrong token at a reference label transition
```

The method keeps these components separate, then multiplies them by error costs.

## Lines 1–29: description, imports, and shared ordering

| Line | What it does and why |
|---|---|
| 1–13 | The module docstring describes the fixed-reference-history objective, names the source score files, and links the methodology. It does not execute an analysis. |
| 15 | Imports `json` for reading manifests and writing the structured report. |
| 16 | Imports `Path` for joining input-directory paths and opening files. |
| 17 | Imports `TYPE_CHECKING` and `Optional` for type annotations. |
| 19 | Imports NumPy as `np` for array operations, sorting, and cumulative sums. |
| 20 | Imports Sisyphus `Job`, `Task`, and `tk`: job definition, task definition, and paths/output registration. |
| 22 | Reuses `log_softmax`, `read_pair`, and `transition_posteriors` from the earlier diagnostic. Their roles are explained at lines 167 and 175 below. |
| 24 | `TYPE_CHECKING` is false during ordinary execution. The following import is for static type checking only. |
| 25 | Makes the `PreparedReferenceScores` type available to a type checker without importing the heavier extraction module at runtime. |
| 28 | Defines the three component names in the same order used in every component vector. Changing this order without changing the vectors would mislabel the report. |
| 29 | Defines the two corpus-weighting schemes in their processing order: pooled first, equal-utterance second. |

## Lines 32–37: `group_events()`

This function turns several events at exactly the same switching threshold into
one event whose component changes are their sum.

| Line | Explanation |
|---|---|
| 32 | Defines a function accepting a one-dimensional threshold array and a two-dimensional change array. Each row of `changes` belongs to the threshold at the same position. |
| 33 | The docstring states that simultaneous switches are combined without losing their error-component contributions. |
| 34 | `np.argsort(thresholds)` returns the indices that would put the thresholds in ascending order. It does not yet rearrange the change rows. |
| 35 | Applies the same permutation to thresholds and change rows, keeping each event paired with its contributions. |
| 36 | `np.unique(..., return_index=True)` finds each distinct numerical threshold and the first row of its group in the sorted array. |
| 37 | `np.add.reduceat(..., axis=0)` sums change rows from each group start to the next group start. Returns the unique thresholds and grouped changes. |

For example:

```text
Input thresholds:       [2, 1, 2]
Input change rows:      [A, B, C]

After sorting:          thresholds [1, 2, 2], changes [B, A, C]
Unique thresholds:      [1, 2]
Group starts:           [0, 1]
Grouped change rows:    [B, A + C]
```

Grouping is by **exact numerical equality**. Values such as `2.411417007` and
`2.411417484` are different events even if they represent nominally identical
quantization gaps.

## Lines 40–63: `decision_events()`

This is where device scores and digital weights become switching events.

### Lines 40–50: best token and threshold

| Line | Explanation |
|---|---|
| 40 | Defines the function for one utterance's device lattice, reference labels, and digital transition weights. |
| 41–46 | Documents its four returned objects and the diagnostic's tie conventions. Blank wins blank/token ties; the first token ID wins tied nonblank maxima. |
| 47 | Converts logits to a NumPy `float64` array. Shape remains `[T, U + 1, V]`. This improves subsequent arithmetic precision but cannot undo rounding already present in saved `float32` logits. |
| 48 | `logits[..., :-1]` takes all frames and prefixes but excludes the final vocabulary entry, blank. `argmax(..., axis=-1)` finds the winning nonblank token ID at each state. Result shape: `[T, U + 1]`. |
| 49 | Finds that winning nonblank logit's value at each state. Same shape as `best_tokens`. |
| 50 | Subtracts the best nonblank logit from blank's logit. This produces the penalty at which their costs tie. The device emits when the penalty is strictly greater than this threshold. |

Why line 50 works:

```text
blank_cost = -blank_logit + shared_normalizer + penalty
token_cost = -best_token_logit + shared_normalizer

Token wins when token_cost < blank_cost.
Cancel the shared normalizer and rearrange:

penalty > blank_logit - best_token_logit
```

This is a real-arithmetic derivation of a greedy decision. The deployed
scorer's finite-precision cost calculation can differ at numerical near-ties.

### Lines 51–55: label masses and correctness

| Line | Explanation |
|---|---|
| 51 | Allocates `label_mass` as zeros with shape `[T, U + 1]`, matching blank weights. |
| 52 | Copies the `[T, U]` label weights into its first `U` columns. The final column stays zero because no next reference token exists after prefix `U`. |
| 53 | Allocates a Boolean correctness array of shape `[T, U + 1]`, initially false. |
| 54 | Compares each winning nonblank ID with the appropriate next reference token. `labels[None, :]` has shape `[1, U]` and broadcasts over all `T` frames. The final prefix remains false. |
| 55 | `~correct` is Boolean NOT. Multiplying it by `label_mass` retains only the digital label mass at states where the best nonblank token is wrong. |

Example for reference tokens `[A, B]`:

```text
Prefix 0: compare best token with A.
Prefix 1: compare best token with B.
Prefix 2: no next token; label mass is zero.
```

### Lines 57–63: error changes and return values

| Line | Explanation |
|---|---|
| 57 | Stacks the three error changes for switching from blank to the best token: add blank mass to false emissions, remove label mass from missed emissions, and add incorrect-token label mass to wrong tokens. Shape: `[T, U + 1, 3]`. |
| 58 | Marks states with any positive digital occupancy. A state with both weights zero cannot affect the objective. |
| 59 | Starts the tuple of four returned values. |
| 60 | Boolean indexing selects active states and flattens the frame/prefix dimensions. Returns thresholds with shape `[M]` and changes with shape `[M, 3]`. |
| 61 | Constructs the all-blank baseline: no false emissions, all label mass missed, no wrong tokens. Before normalization, missed mass equals `U` up to posterior arithmetic. |
| 62 | Sums correct-best-token label mass. This later becomes the numerator of emission coverage. Converts the NumPy scalar to a Python `float`. |
| 63 | Closes the return tuple. |

At one correct-token state with blank mass `0.1` and label mass `0.9`, the change
row is:

```text
[+0.1, -0.9, 0.0]
```

With unit costs, crossing that threshold improves risk by `0.8`.
If the winning token is wrong, the row becomes:

```text
[+0.1, -0.9, +0.9]
```

The missed emission becomes a wrong emission, so crossing the threshold adds
`0.1` risk instead of recovering the reference token.

## Lines 66–71: `component_dict()`

| Line | Explanation |
|---|---|
| 66 | Defines the conversion from a three-component vector to report fields. |
| 67 | Explains why the next line handles tiny negative numerical residuals. |
| 68 | Clamps negative values to zero. For example, a final missed-emission sum of `-1e-16` from accumulated roundoff becomes zero. |
| 69 | Pairs each value with its component name and converts scalars to JSON-friendly Python floats. |
| 70 | `values @ costs` is the dot product: multiply each error component by its cost and sum. Stores that weighted total as `risk`. |
| 71 | Returns the component dictionary plus total risk. |

Unit costs mean `risk = false + missed + wrong`. Different costs change the
trade-off, but not the device switching thresholds.

## Lines 74–87: `interval_representative()`

The optimum is usually an interval, not one unique penalty. This helper picks
one finite member of an optimal interval for convenience.

| Line | Explanation |
|---|---|
| 74 | Accepts lower and upper endpoints plus whether the lower endpoint is included. |
| 75 | Tests for an unbounded upper endpoint, represented by `None`. |
| 76 | Initially selects one unit above the lower endpoint. |
| 77 | Checks whether the addition overflowed or rounded back to the lower endpoint. |
| 78 | Uses the next representable floating-point number toward positive infinity as a fallback. |
| 79 | Tests for a single-point interval. |
| 80 | Selects its only possible penalty. |
| 81 | Otherwise handles a finite interval with different endpoints. |
| 82 | Selects its midpoint using two half-values, avoiding the potentially overflowing sum `lower + upper`. |
| 83 | Checks whether rounding placed the midpoint on an excluded lower endpoint. |
| 84 | Uses the included upper endpoint instead. Under the blank-wins-ties convention, that endpoint belongs to the interval. |
| 85 | Verifies that the chosen value is finite. |
| 86 | Raises an error if no finite representative was obtained. |
| 87 | Returns a Python float. |

The representative does not have a better risk than the other members of its
interval. Its additional decimal places are not independent evidence of
calibration precision.

## Lines 90–140: `optimize_events()`

This function receives sorted, grouped events for one weighting scheme and
finds the minimum-risk penalty intervals.

### Lines 90–107: arguments and validation

| Line | Explanation |
|---|---|
| 90 | Defines the optimizer. The `*` makes the following configuration arguments keyword-only. |
| 91 | Defaults to no upper bound and comparison penalties `0`, `2`, `4`, and `7.3`. These comparisons are not a search grid. |
| 92–98 | Documents sorted unique thresholds, normalized inputs, and interval endpoint conventions. |
| 99 | Converts the three error costs to `float64`. |
| 100 | Requires exactly three finite, nonnegative costs. |
| 101 | Raises an error for invalid costs. |
| 102 | Checks that the lower bound is finite; negative lower bounds are allowed. |
| 103 | Raises an error for a nonfinite lower bound. |
| 104 | If an upper bound exists, checks that it is finite and not below the lower bound. |
| 105 | Raises an error for invalid upper bounds. |
| 106 | Checks that every report-comparison penalty is finite. |
| 107 | Raises an error for invalid comparisons. |

### Lines 109–119: construct all risks and select minima

| Line | Explanation |
|---|---|
| 109 | Cumulatively adds event changes and prepends the all-blank baseline. With `K` thresholds, `states` has shape `[K + 1, 3]`. Row `i` represents crossing exactly the first `i` thresholds. |
| 110 | Clamps negative roundoff residuals in the component totals to zero. |
| 111 | Computes every state's weighted risk in one matrix-vector multiplication. Result shape: `[K + 1]`. |
| 112 | Finds how many thresholds are strictly below the minimum penalty. This identifies the first allowed state. `side="left"` does not count a threshold equal to the penalty, since blank wins the tie. |
| 113 | Without an upper bound, the last allowed state is `K`, after every event. |
| 114 | With an upper bound, finds the state at that exact penalty using the same strict-threshold convention. |
| 115 | Takes the minimum over allowed states. `last + 1` makes Python's otherwise exclusive slice include state `last`. |
| 116 | Notes that the next comparison handles numerical ties, not statistically near-optimal penalties. |
| 117 | `flatnonzero` obtains indices where the numerical minimum comparison is true. |
| 118 | Uses relative and absolute risk tolerances of `1e-12`. This tolerance compares risks; it does not merge switching thresholds. |
| 119 | Adds `first` because the matching indices were relative to a sliced risk array, not the full state array. |

For thresholds `[1, 2, 3]`, the state rows mean:

```text
states[0]: penalty at or below 1
states[1]: penalty above 1, at or below 2
states[2]: penalty above 2, at or below 3
states[3]: penalty above 3
```

At penalty `2`, `searchsorted(..., side="left")` returns `1`, so the optimizer
uses `states[1]`. The event at threshold `2` has not yet been crossed.

### Lines 120–133: turn optimal states into intervals

| Line | Explanation |
|---|---|
| 120 | Starts the list of optimal intervals. |
| 121 | Iterates over every minimum-risk state, including multiple or separated optima. |
| 122 | For the first allowed state, clips its lower endpoint to `min_penalty`. For later states, uses the preceding crossed threshold. |
| 123 | Uses the next un-crossed threshold as the upper endpoint. After the last threshold, uses the search upper bound, possibly `None`. |
| 124 | Checks whether the interval extends beyond the supplied maximum penalty. |
| 125 | Clips its upper endpoint to that maximum. |
| 126 | Includes the lower endpoint only for the first allowed interval. Later intervals exclude it because their last event must be strictly crossed. |
| 127 | Chooses a representative value using the helper explained above. |
| 128 | Starts appending the interval's report dictionary. |
| 129 | Stores numerical endpoints. `None` means unbounded above. |
| 130 | Stores explicit inclusion flags. Finite upper endpoints are included. |
| 131 | Stores the representative penalty. |
| 132 | Expands the state's named error components and risk into the dictionary. |
| 133 | Closes and appends the dictionary. |

An interval `(1, 2]` excludes penalty `1` and includes penalty `2`. A bounded
first interval `[0, 1]` includes both endpoints. Equal search bounds can produce
a single-point interval such as `[2, 2]`.

### Lines 135–140: evaluate comparison penalties

| Line | Explanation |
|---|---|
| 135 | Starts the comparison-row list. |
| 136 | Iterates over the requested comparison penalties. |
| 137 | Finds the corresponding cumulative state by counting thresholds strictly below the penalty. |
| 138 | Stores the penalty, three components, and total risk. These comparisons can be outside the optimization bounds. |
| 139 | Returns the minimum risk and optimal intervals. |
| 140 | Includes the comparison rows in that return value. |

The optimum is analytical because a constant penalty changes decisions only
at the collected thresholds. Evaluating all cumulative states covers every
possible fixed-state decision pattern; no dense penalty grid is necessary.

## Lines 143–217: `analyze_scores()`

This is the orchestration function: it loads utterances, constructs events,
builds both weighting schemes, and calls the optimizer.

### Lines 143–156: load manifests and select utterances

| Line | Explanation |
|---|---|
| 143 | Defines the analysis function with unit error costs. The directory arguments are `Path` objects. |
| 144 | Defines search-bound defaults. |
| 145 | Defines comparison defaults and an optional utterance limit for direct helper-level smoke checks. The Sisyphus job does not expose that limit. |
| 146 | Opens the digital manifest. |
| 147 | Loads its mapping from utterance tags to saved-score metadata. |
| 148 | Opens the device manifest. |
| 149 | Loads the device mapping. |
| 150 | Requires a nonempty digital manifest and identical tag sets. Dictionary iteration supplies keys, so `set(...)` compares utterance identities. |
| 151 | Raises an error if the inputs do not describe the same nonempty corpus. |
| 152 | Sorts tags for deterministic traversal. |
| 153 | Checks whether a helper-level utterance limit was supplied. |
| 154 | Requires that limit to be positive. |
| 155 | Raises an error for invalid limits. |
| 156 | Selects the first `N` sorted tags. This is deterministic truncation, not random sampling. |

### Lines 158–174: initialize accumulators and load pairs

| Line | Explanation |
|---|---|
| 158 | Creates lists for grouped per-utterance thresholds and changes. |
| 159 | Allocates two all-blank component vectors: row 0 for pooled weighting, row 1 for equal-utterance weighting. Shape `[2, 3]`. |
| 160 | Allocates correct-label-mass numerators for the two coverage calculations. |
| 161 | Allocates their label-mass denominators. |
| 162 | Initializes the total frame counter. |
| 163 | Starts processing each selected utterance. |
| 164 | Looks up its saved NPZ filename in the digital manifest. |
| 165 | Checks that the device manifest uses the same filename for this tag. |
| 166 | Raises an error on a filename mismatch. |
| 167 | `read_pair()` loads both NPZ files, checks paired shapes/reference labels/blank IDs, and returns both logits arrays, reference labels, and blank ID. |
| 168 | Requires a three-dimensional lattice, at least one frame, and at least two vocabulary entries. |
| 169 | Requires blank to be the final vocabulary entry. |
| 170 | Requires `U + 1` prefix columns and `U <= T`, since this topology emits at most one token per frame. |
| 171 | Requires every reference label to be a valid nonblank token ID. |
| 172 | Raises an error for those invalid lattice/reference conditions. |
| 173 | Checks that both models' logits are finite. |
| 174 | Raises an error for NaN or infinite logits. |

### Lines 175–189: alignment, events, and two weighting schemes

| Line | Explanation |
|---|---|
| 175 | Applies digital log-softmax, then digital reference-conditioned forward-backward. Returns blank and label transition masses. These weights are fixed for all candidate device penalties. |
| 176 | Calls `decision_events()` and unpacks thresholds, three-component changes, all-blank baseline, and correct-label mass. |
| 177 | Supplies the device logits, correct labels, and digital weights to that call. |
| 178 | Closes the call. |
| 179 | Explains the equal-utterance normalization: each utterance has total normalized occupancy one. |
| 180 | Explains pooled weighting: each encoder frame has equal total influence. |
| 181 | Sets `frames` to `T` for this utterance. |
| 182 | Concatenates raw changes and changes divided by `T`. The resulting `[M, 6]` matrix stores pooled components in columns 0–2 and equal-utterance components in columns 3–5. |
| 183 | Groups exactly identical thresholds within this utterance, summing all six columns. |
| 184 | Saves its grouped thresholds for later corpus aggregation. |
| 185 | Saves its grouped six-component change matrix. |
| 186 | Adds the raw baseline to row 0 and the baseline divided by `T` to row 1. |
| 187 | Accumulates correct-label mass directly and after division by `T`. |
| 188 | Accumulates total label mass as `U` and `U / T`, using forward-backward's label-mass normalization property. |
| 189 | Adds this utterance's frame count to the corpus total. |

`log_softmax()` and `transition_posteriors()` are imported rather than
reimplemented here. The latter computes forward and backward path scores,
then each transition's posterior mass from its prefix score, local score,
suffix score, and complete-reference likelihood. It verifies that transition
weights sum to one per frame and label weights sum to one per reference token.
The companion methodology explains this topology in greater detail.

The six columns from line 182 are:

```text
0: raw false-emission change
1: raw missed-emission change
2: raw wrong-token change
3: false-emission change / utterance frames
4: missed-emission change / utterance frames
5: wrong-token change / utterance frames
```

### Lines 191–206: combine utterances and optimize

| Line | Explanation |
|---|---|
| 191 | Starts corpus-level grouping. |
| 192 | Concatenates all utterance thresholds and change rows, then groups exact duplicate thresholds across utterances. |
| 193 | Closes the grouping call. |
| 194 | Sets the final divisors: total frames for pooled weighting, utterance count for equal-utterance weighting. |
| 195 | Starts the list of two optimization results. |
| 196 | Iterates over each weighting name and divisor, with index 0 or 1 selecting its three columns. |
| 197 | Calls the analytical optimizer for this weighting scheme. |
| 198 | Selects columns 0–2 or 3–5 and divides by the final normalization. |
| 199 | Divides the matching initial component vector by the same normalization and supplies error costs. |
| 200 | Supplies search bounds and report-comparison penalties. |
| 201 | Closes the optimizer call. |
| 202 | Labels the returned result with its weighting scheme. |
| 203 | Starts adding the correct-best-token coverage metric. |
| 204 | Divides correct emission mass by total emission mass. If no reference labels exist, stores `None` instead of dividing by zero. |
| 205 | Closes that expression. |
| 206 | Saves the completed result. |

The resulting error components are:

```text
Pooled:
    sum(raw utterance errors) / sum(utterance frames)

Equal utterances:
    sum(utterance errors / utterance frames) / number of utterances
```

Both are per-frame quantities. The latter gives equal influence to utterances
after normalizing each one's error by its own length.

Coverage also has two weightings. The equal-utterance version is the ratio of
the sums of frame-normalized correct and total label masses. It is not the
arithmetic mean of each utterance's individual coverage ratio.

### Lines 207–217: construct the full report object

| Line | Explanation |
|---|---|
| 207 | Starts returning the JSON-compatible summary dictionary. |
| 208 | Records both input-directory paths as strings. |
| 209 | Records processed utterance and frame counts. |
| 210 | Records the count of unique active numerical switching thresholds, before applying optimization bounds. |
| 211 | Stores named error costs as Python floats. |
| 212 | Records the search bounds. |
| 213 | Describes the optimized objective and distinguishes it from WER. |
| 214 | Records the assumed tie rules. |
| 215 | Records the equal-utterance normalization convention. |
| 216 | Stores both optimization results. |
| 217 | Closes the summary dictionary. |

## Lines 220–254: `format_results()`

This function formats already computed values. It does not perform another
optimization or modify the weights.

| Line | Explanation |
|---|---|
| 220 | Defines the text-report formatter. |
| 221 | Starts its list of output lines. |
| 222 | Writes the objective description. |
| 223 | Writes the digital-score path. |
| 224 | Writes the device-score path. |
| 225 | Writes the corpus size in utterances and frames. |
| 226 | Writes the unique threshold count. |
| 227 | Writes the configured costs. |
| 228 | Explains the penalty sign and blank tie rule. |
| 229 | Labels the components as per-frame errors, not WER percentages. |
| 230 | Explains equal-utterance weighting. |
| 231 | Closes the initial output-line list. |
| 232 | Iterates over the two weighting results. |
| 233 | Retrieves emission coverage. |
| 234 | Adds a blank separator and the weighting name. |
| 235 | Starts the coverage label. |
| 236 | Formats valid coverage as a percentage with four decimal places; otherwise explains that no labels were available. |
| 237 | Writes minimum risk with nine digits after the decimal point. |
| 238 | Starts the aligned table header: penalty/interval, risk, and false emissions. |
| 239 | Completes the header with missed emissions and wrong tokens. |
| 240 | Starts the table-row list. |
| 241 | Iterates over every optimal interval. |
| 242 | Uses `[` for an included lower endpoint and `(` for an excluded one. |
| 243 | Formats the upper endpoint as an included finite value or open infinity. `:.9g` means nine significant digits. |
| 244 | Constructs the interval label. |
| 245 | Adds that label and the interval's computed component dictionary to the table. |
| 246 | Appends the comparison-penalty rows. |
| 247 | Iterates over all table rows. |
| 248 | Formats the label, risk, and false-emission component. |
| 249 | Completes the row with missed-emission and wrong-token components. |
| 250 | Starts the representative-penalty line. |
| 251 | Displays the representative of the first optimal interval. Other tied optimal intervals remain in the table and JSON. |
| 252 | Starts the final scope reminder. |
| 253 | Explains that free-decoding histories change and transition error is not WER. |
| 254 | Joins the lines with newlines and adds a trailing newline. |

The display precision is a formatting choice. It does not establish that the
selected penalty is experimentally stable to nine significant digits.

## Lines 257–291: `OptimalBlankPenaltyJob`

This class lets Sisyphus schedule the analysis after score extraction.

### Lines 257–276: constructor and outputs

| Line | Explanation |
|---|---|
| 257 | Defines the job as a Sisyphus `Job` subclass. |
| 258 | Describes the job's objective. |
| 260 | Starts its constructor. |
| 261 | Declares the instance parameter `self`. |
| 262 | Accepts the digital score directory as a Sisyphus path. |
| 263 | Accepts the device score directory as a Sisyphus path. These path inputs carry extraction-job dependencies. |
| 264 | Accepts the three component costs, defaulting to equal costs. |
| 265 | Accepts the lower optimization bound. |
| 266 | Accepts the optional upper bound. |
| 267 | Accepts comparison penalties. |
| 268 | Closes the constructor signature. |
| 269 | Saves the digital input path on the job. |
| 270 | Saves the device input path. |
| 271 | Saves the error costs. |
| 272 | Saves the lower bound. |
| 273 | Saves the upper bound. |
| 274 | Saves comparison penalties. |
| 275 | Declares the job-owned JSON output path. The constructor does not yet write the report. |
| 276 | Declares the job-owned text output path. |

### Lines 278–291: task and execution

| Line | Explanation |
|---|---|
| 278 | Defines the task generator used by Sisyphus. |
| 279 | Yields one `run` task requesting two CPUs, 16 GB memory, and two hours. It does not run the analysis while loading the workflow config. |
| 281 | Defines the method executed by the scheduled task. |
| 282 | Calls the analysis orchestrator. |
| 283 | Resolves the Sisyphus input paths and converts their strings to `Path` objects. |
| 284 | Supplies the job's costs and lower bound. |
| 285 | Supplies its upper bound and comparisons. |
| 286 | Closes the analysis call. |
| 287 | Opens the declared JSON output for writing. |
| 288 | Serializes the results with indentation; `allow_nan=False` rejects nonstandard NaN/infinite JSON values. Unbounded endpoints use `None`, which becomes JSON `null`. |
| 289 | Appends a newline. |
| 290 | Opens the declared text output for writing. |
| 291 | Formats and writes the human-readable report. |

## Lines 294–315: `run_optimal_blank_penalty()`

This is the workflow entrypoint called by `config/scorer_debug.py`.

| Line | Explanation |
|---|---|
| 294 | Defines the helper that creates and registers an optimization job. |
| 295 | Accepts a `PreparedReferenceScores` object returned by `prepare_scores()`. The string annotation avoids a runtime import. |
| 296 | Makes subsequent options keyword-only. |
| 297 | Accepts the three error costs. |
| 298 | Accepts the lower bound. |
| 299 | Accepts the optional upper bound. |
| 300 | Accepts report-comparison penalties. |
| 301 | Specifies that the helper returns the created job. |
| 302–305 | Documents that it consumes prepared extraction outputs and states the cost ordering. |
| 306 | Retrieves the preparation's output prefix, distinguishing runs such as `first_fifth` and `full_corpus`. |
| 307 | Constructs the Sisyphus job. This declares computation; it does not call `run()` immediately. |
| 308 | Selects the prepared digital reference score output. |
| 309 | Selects the fully offloaded `base_model` score output. |
| 310 | Passes the chosen optimization/report options to the constructor. |
| 311 | Closes job construction. |
| 312 | Adds a discoverable job alias under the preparation's output prefix. |
| 313 | Registers the JSON summary under that prefix. |
| 314 | Registers the text report under that prefix. |
| 315 | Returns the job so callers can access its outputs or modify resource requirements. |

The workflow therefore follows this dependency chain:

```text
prepare_scores()
    -> digital extraction output
    -> fully offloaded extraction output

run_optimal_blank_penalty(prepared)
    -> OptimalBlankPenaltyJob
        -> analyze_scores()
            -> digital alignment weights
            -> device switching events
            -> threshold grouping and cumulative risks
            -> optimal intervals for both weighting schemes
        -> JSON and text outputs
```

## How the code explains the matching corpus thresholds

The relevant statements are:

- **Line 50:** thresholds are differences between quantized device logits.
- **Lines 36–37:** only exactly equal numerical thresholds are grouped.
- **Lines 109–119:** the optimum is selected among the resulting discrete states.
- **Lines 122–133:** an optimal state becomes a threshold-bounded interval.

More utterances change the weights of events and their cumulative risks, but
can retain the same winning boundary on the shared score grid. Float32
rounding can also split one nominal boundary into several nearby numerical
ones. Consequently, the narrow interval around `2.411417` is an exact result
for the saved-score decision model, not evidence that deployment should use
that value to eight decimal places.

## What the code does not simulate

The key limiting statements are **line 175**, which computes fixed digital
reference-conditioned weights, and **line 54**, which compares tokens against
the known correct reference history. There is no decoder loop that updates
the prefix after a device emission, and no word-level edit-distance calculation.

The threshold sweep is globally optimal for this local, fixed-state objective.
Its deployment value should be interpreted through free-decoding WER and
error counts, with attention to quantization boundaries and numerical ties.
