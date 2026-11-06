# Interactive LibRASR N-best probe

The current approach is direct reranking of exported candidate costs, described
in [`TREE_NBEST_APPROACH.md`](TREE_NBEST_APPROACH.md). No new alignment or model
rescoring is needed for that version. The independent rescoring diagnostic in
[`TREE_NBEST_VALIDATION.md`](TREE_NBEST_VALIDATION.md) is an optional investigation
of noisy scorer mechanics, not a required stage of direct reranking.

[`probe_tree_nbest.py`](probe_tree_nbest.py) exercises the existing N-best binding
using an already completed recognition job's config and checkpoint. It does not
submit jobs or change the recognition pipeline.

The current experiment uses joint L2 ADC **2prec/6range**, with local estimated
penalty **1.44**. The default source job is the matching cycle-1 separated-ADC
tree run. Other layers retain that run's settings (including joint L1 4prec/4range).

## Run on an allocated GPU node

From `/u/azim.javed/experiments/training/qat`:

```bash
apptainer exec --nv --bind /u,/work/smt4,/work/asr4 \
  /work/asr4/hilmes/apptainer/torch-2.8_onnx-1.22_v3.sif \
  /work/smt4/azim.javed/envs/base_env/bin/python -u \
  diagnostics/qat_ffnn_scores/probe_tree_nbest.py \
  --whole-corpus --num-seqs 3 --nbest 20 --blank-penalty 1.44 \
  --output-dir output/diagnostics/qat_ffnn_scores/tree_nbest_probe_jl2_2prec_6range/bp1.44 \
  --save-encoder
```

The default source is shard 1 of cycle 1 of the full-quant separated-ADC
`tree_mbs1024` run, without an external LM or ILM subtraction. `--whole-corpus`
removes the source shard's segment filter only for this small diagnostic, making
the same first three sorted utterances available as in the earlier probe.
It does not process the whole corpus: `--num-seqs` still limits the work.
Override the source with `--recog-job`, pointing to a matching recognition job
directory containing `work/rnn.sh`. The script checks the effective L2 ADC setting.

For a zero-penalty anchor, repeat with `--blank-penalty 0` and a different output
directory, such as `tree_nbest_probe_jl2_2prec_6range/bp0`. This is an interface/candidate probe,
not a penalty sweep or an optimization job.

By default the probe processes the first three sequences in the saved dataset
ordering. To select an exact utterance, use:

```text
--num-seqs 1 --seq-tag dev-other/116-288045-0000/116-288045-0000
```

`--seq-tag` can be repeated; tags must be present in the saved dataset.

For a CPU-only preflight:

```bash
/work/smt4/azim.javed/envs/base_env/bin/python \
  diagnostics/qat_ffnn_scores/probe_tree_nbest.py --inspect-only
```

## What it does

1. Resolves the saved RETURNN command and RASR config.
2. Inspects the actual lexicon's pronunciation variants.
3. Loads the saved model config, omitting its eager decoder construction in
   memory, and restores the encoder checkpoint.
4. Makes a private RASR config in the probe output directory.
5. Encodes a few utterances and calls `recognize_segment_n_best(features, n)`.
6. Retrieves the direct best traceback from the same search, without decoding
   a second time, and compares its transcript/cost to the first N-best result.
7. Prints candidates and saves their full traceback fields.

The private config forces an additive post-softmax penalty, CUDA scoring, and
`sentence-end-fall-back = no`. Disabling fallback is important: an unfinished
word can have hidden token emissions that cannot be recovered from returned
word lemmas. The probe requires the RNA topology without collapsed repeats.
Beam, lexicon, LM, and ILM settings otherwise come from the saved run.

## Outputs

| File | Contents |
|---|---|
| `nbest.jsonl` | One record per utterance, including reference, direct best, candidates, scores, token-count diagnostics, and complete returned tracebacks |
| `summary.json` | Input paths, hardware settings, effective penalty/scale, and per-utterance top-N-best/direct agreement |
| `rasr.probe.config` | Private effective config, leaving the source config unchanged |
| `rasr.probe.log` | Decoder log |
| `encoder_000.npy`, etc. | Optional encoder features for later interface-only checks |

Use a distinct output directory for each run: probe output filenames are fixed.
`nbest.jsonl` is flushed after each utterance so partial results remain if a
later utterance fails.

## What to inspect in the results

- Are multiple distinct candidates returned? The requested N-best size is an
  upper bound, not a guarantee.
- Does `top_nbest_matches_direct_transcript` hold?
- Is `top_nbest_minus_direct_cost` close to zero?
- Are the returned candidate costs consistent with their ranks?
- Is `token_count_identifiable` true, and are token/blank counts plausible?
- Are there empty results after disabling sentence-end fallback?
- Do references and candidate transcripts reconstruct correctly?

Token counts are inferred from the lexicon, not exported model-token paths.
For each lemma the script examines possible nonblank pronunciation lengths;
it reports a total only if every length is uniquely determined and the total
does not exceed the encoder-frame count. Blank pronunciation symbols do not
count as emitted nonblank tokens. For a complete RNA path, inferred blank
count is `encoder_frames - nonblank_token_count`.

The `lemma` field is not inherently word-level: other recognition configurations
can return BPE tokens and timed blank spans in the same Python object type.
However, the saved tree probe's 60 N-best candidates all have zero timestamps.
Even its direct-best tree tracebacks can hide blanks inside word spans: the
`MART` example has 19 explicitly timed blank frames, but 26 blanks inferred from
29 frames minus 3 model tokens. Summing explicit blank durations would miss seven
blanks inside the word span. See the interface audit in the approach document.

The shared lexicon preflight found 200,004 lemma names, each with one
pronunciation. That supports count inference for this particular lexicon;
the GPU results must still confirm returned lemma coverage and score behavior.

Earlier files under `tree_nbest_probe/bp0` use the old ADC configuration. Collect
new candidates under the ADC-specific output root; changing report labels or
only the penalty would not turn the earlier reads into this hardware setting.

`traceback_terminal_cost` is the last returned item's `am_score + lm_score`.
It is not yet certified as a complete-path cost including all epsilon/final
weights. The comparison against the direct best traceback is a useful first
check, not proof of correctness for every alternative.

`tentative_penalty_coefficient` and `tentative_zero_penalty_cost` illustrate
the intended additive reranking metadata. They remain tentative until score
scaling, total costs, and inferred counts are validated. Diagnostic word errors
use simple word edit distance; final experimental scoring should use the
existing Sclite pipeline.

The probe leaves readout noise active and does not set/reset seeds or replay
random draws. Separate searches use naturally advancing/new random streams.
Candidate changes across anchors can reflect readout noise as well as penalty.
The statistical validator measures that variation across fresh reads.
