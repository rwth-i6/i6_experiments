# Statistical N-best score validation with fresh readout noise

[`validate_tree_nbest.py`](validate_tree_nbest.py) validates the first stage of
the proposed tree-search blank-penalty reranker. It consumes the existing probe
results and saved encoder features; it does not re-encode audio, launch a search,
submit jobs, or use digital alignment weights.

The current experiment uses **joint L2 ADC 2prec/6range**, with **1.44** as the
local estimated penalty/collection anchor. The probe and validator now require
that setting. The earlier `tree_nbest_probe/bp0` data came from different ADC
settings and must not be relabeled as data for the new configuration.

The validator leaves readout noise active. It does not reset seeds, restore RNG
states, disable noise, or replay noise draws. Naturally varying scores are
measured statistically, not treated as repeatability failures.

## GPU command

From `/u/azim.javed/experiments/training/qat`, on an allocated GPU node:

```bash
apptainer exec --nv --bind /u,/work/smt4,/work/asr4 \
  /work/asr4/hilmes/apptainer/torch-2.8_onnx-1.22_v3.sif \
  /work/smt4/azim.javed/envs/base_env/bin/python -u \
  diagnostics/qat_ffnn_scores/validate_tree_nbest.py \
  --probe-dir output/diagnostics/qat_ffnn_scores/tree_nbest_probe_jl2_2prec_6range/bp1.44 \
  --num-seqs 3 --max-candidates 2 --repeats 20 --penalty-delta 1 \
  --save-costs
```

First collect this new ADC-specific probe using the command in
[`TREE_NBEST_PROBE.md`](TREE_NBEST_PROBE.md). Existing encoder files are reused
within validation; audio is not re-encoded for every noisy scoring evaluation.

By default outputs go to `<probe-dir>/validation_statistical/`. Supply `--output-dir` to
retain separate validation runs. Output filenames are fixed and are replaced
when rerunning in the same directory.

CPU-only metadata inspection, without loading models or writing results:

```bash
/work/smt4/azim.javed/envs/base_env/bin/python \
  diagnostics/qat_ffnn_scores/validate_tree_nbest.py --inspect-only
```

## What is validated

### 1. Actual token reconstruction

The script uses the original traceback lemma sequence, not text retokenized
with an unrelated tokenizer. Each lemma must have one unique pronunciation.
Explicit `[BLANK]` pronunciation symbols are omitted from the nonblank sequence.
The sentence-end pronunciation must be empty.

Dense scorer output IDs are recovered from the lexicon phoneme inventory under
the checked RASR mapping:

```text
RASR phoneme IDs start at 1 in inventory parsing order.
One-state monophone emission index = phoneme ID - 1.
Therefore output IDs are zero-based inventory positions.
```

This is derived from RASR's `MonophoneStateTying` construction and classification,
not an assumption that alphabetically sorted BPE tokens have the correct IDs.
The saved probe's inventory has 185 entries, with `<blank>` at output index 184.
The script also checks the initial-history label and model output dimension.

### 2. Independent alignment rescoring

The validator loads the same scorer using the existing
`FixedContextTransducerPy` or `FixedContextTransducerPy_Cached` initialization.
It therefore reuses the checkpoint loading, model imports, hardware settings,
and recognition configuration used by the real decoder.

For a candidate with `U` tokens and `T` saved encoder frames:

```text
At prefix u:
    history = initial padding followed by the first u candidate tokens,
              truncated to the configured history length.

At each frame t:
    obtain the actual scorer cost for blank and the next candidate token.
```

These form arrays:

```text
blank costs: [T, U + 1]
label costs: [T, U]
```

Monotonic Viterbi considers two transitions, both consuming one frame:

```text
Take blank: keep prefix u.
Take label: advance to prefix u + 1.
```

It computes the minimum cost of emitting the whole candidate token sequence
over all complete alignments, including trailing blank frames. Backtracking
records each token's emission frame. An empty transcript is supported: all
`T` frames must be blank.

This is a full alignment search conditioned on one candidate transcript. It
does not reproduce the original lexical beam's pruning or replay the selected
N-best alignment: the N-best interface did not provide usable timestamps.

### 3. Fresh-read sampling

For each candidate, the default collects 20 fresh lattice evaluations at the
anchor and 20 at the shifted penalty. Dense score caches, encoder-device caches,
and prediction-state caches are cleared between evaluations. Normal caching is
retained within an evaluation. The random streams keep advancing naturally.

Conditions are interleaved:

```text
Round 1: anchor, shifted
Round 2: shifted, anchor
Round 3: anchor, shifted
...
```

The fixed checkpoint/programmed device state stays the same. Readout noise does
not. Each sample summarizes a complete lattice evaluation, which itself involves
multiple hardware reads; the lattice cells are not counted as independent
statistical replicates.

An additional independent pilot evaluation chooses a fixed alignment for replay.
The pilot is excluded from the statistical samples. This avoids choosing the
alignment on an anchor sample that also contributes to its mean cost.

### 4. Actual additive penalty behavior

For shifted evaluations, the script changes the scorer's own `blank_penalty` to:

```text
shifted penalty = anchor penalty + penalty delta
```

It reruns the neural scorer with cleared caches. It does not create shifted
scores by adding a constant to the first result.

The expected mean shifts are:

```text
Mean blank score increases by the penalty delta.
Mean nonblank label score stays unchanged.

Mean complete-path cost change = penalty delta * (T - U).
```

Individual fresh-read differences do not have to equal those targets. The code
reports the following read-level metrics under both conditions:

| Metric | Expected shift |
|---|---|
| Best complete-transcript Viterbi cost | `penalty_delta * blank_count` |
| Cost on the pilot's fixed alignment | `penalty_delta * blank_count` |
| Mean of all blank costs in the lattice | `penalty_delta` |
| Mean of next-candidate-token costs in the lattice | `0` |

The nonblank metric is not applicable to an empty transcript. Mean lattice
metrics are aggregate diagnostics, not simultaneous guarantees about every
individual state. They can hide state-level effects that cancel in an average.

Under the unchanged score distribution, every complete alignment of one
transcript has the same blank count. Adding the penalty shifts all its path
costs by the same amount. Its minimum-cost distribution therefore has the same
constant shift, even though independently sampled optima vary.

### 5. Confidence intervals and practical agreement

The code reports a mean, sample standard deviation, and standard error for each
condition. A Student-t interval describes uncertainty in each mean. An unpaired
Welch interval describes uncertainty in `shifted_mean - anchor_mean`, allowing
different variances under the two conditions.

These intervals assume independent, approximately stationary evaluations and
use a mean-based approximation. With a small sample or skewed/quantized costs,
coverage is approximate. `--confidence` defaults to `0.95`. Intervals are
per metric, not simultaneous across all candidates and checks.

The target's presence inside a wide interval is not proof of agreement. The
report also compares the residual interval to a practical agreement margin:

```text
residual = observed mean shift - expected shift
```

Default margins are `0.1` for path-cost shifts and `0.01` for mean lattice-score
shifts. They are explicit experimental choices, configurable with
`--cost-mean-margin` and `--score-mean-margin`; they are not measured noise levels
or confidence intervals.

| Per-metric assessment | Interpretation |
|---|---|
| `within_practical_margin` | The entire residual CI lies inside the chosen agreement margin |
| `material_difference_detected` | The entire residual CI lies beyond the agreement margin |
| `consistent_but_uncertain` | The target lies in the CI, but precision is insufficient to establish practical agreement |
| `difference_detected_margin_decision_uncertain` | The CI excludes the exact target but crosses a practical-margin boundary |

An overall candidate assessment is `within_practical_margins` only when all
applicable checks establish practical agreement. Otherwise it reports a material
difference if any check does so, or `uncertain_collect_more_reads`.

Twenty reads per condition are a starting point. Collecting more narrows mean
uncertainty under the sampling assumptions; it does not remove physical noise.
If increasing samples after inspecting results, regard the intervals as
exploratory rather than claiming a fixed-sample confirmatory significance level.

## Candidate selection

The default selects up to five distinct transcripts per utterance:

1. The highest-ranked exported candidate.
2. The lowest-word-error retained candidate, preferring lower cost in a tie.
3. Additional distinct transcripts in rank order.

In the earlier probe, this included `MARGARET` even though the top result was
`MART`. It avoids spending the diagnostic budget on many alignment variants
of the same words. `--max-candidates 1` selects only the highest-ranked result.

For the new ADC-specific probe, actual available transcripts may differ. Use
`--transcript` repeatedly to focus on exact retained alternatives, for example:

```text
--transcript MART --transcript MARGARET --transcript NO --transcript "NO I"
```

The script selects the cheapest exported alignment of each requested transcript
and errors if requested transcripts are not available among the selected records.
The recommended initial command instead selects two candidates automatically
per utterance, to avoid assuming the old configuration's alternatives are present.

## Supported first-stage configuration

The script explicitly requires:

- One fixed-context Python scorer with unit scale.
- RNA tree topology, no collapsed repeats, one-state monophone tying.
- No word LM and no ILM subtraction.
- No sentence-end fallback or nonempty sentence-end model pronunciation.
- Blank does not update the prediction history; no vertical label transitions.
- Zero acoustic loop/forward/exit transition costs and context-independent phones.
- Joint L2 ADC 2prec/6range, consistent between saved probe metadata and config.

For this no-LM configuration, a final hypothesis's word-LM contribution is zero.
With an empty sentence-end pronunciation and no remaining input frame, the
fixed-context scorer adds no final model-token cost. Thus the independent
complete-frame calculation does not add a fabricated sentence-end frame.
Adding LM support later requires validating the corresponding final score terms.

Unsupported settings are rejected rather than silently evaluated under the
wrong objective. A new probe under the required ADC setting is needed before
running this updated statistical validator.

## Outputs and interpretation

| Output | Contents |
|---|---|
| `validation.json` | Complete report, token mapping, all scalar samples, mean/SD/CI checks, and pilot alignments |
| `validation.txt` | Human-readable statistical report |
| `validation.jsonl` | Per-candidate results flushed during execution |
| `read_samples.jsonl` | Individual evaluation metrics flushed after every fresh read |
| `rasr.validation.config` | Private scorer config with diagnostic log paths/device |
| `costs_000_rankN.npz`, etc. | Optional pilot and first fresh score arrays per condition; not every repeated lattice |

Exported-versus-independent cost differences are descriptive only. The original
exported cost is a search-selected noisy observation; independent rescoring
uses different noise and may find different alignments. Neither equality nor
inequality with the independent mean certifies export bookkeeping. The reported
fraction of anchor samples at or below the exported value is not a p-value.

The obsolete `repeat_stable` and single-read `penalty_checks_passed` flags have
been replaced by variability estimates and statistical assessments. A completed
command still does not imply practical agreement: inspect the report and its CIs.

The diagnostic validates fixed-transcript score mechanics. It does not select
an optimal penalty, assess full-corpus candidate coverage, or establish a WER
gain from actual tree decoding.
