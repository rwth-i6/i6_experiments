# ASR test suite

The [audit](TEST_AUDIT.md) and [per-test decisions](test_audit.tsv) describe the
original suite. Its historical grid counts, tuning-value snapshots, one-time
migration manifest checks and duplicate assertions have been removed. Mixed
checks were consolidated around behavior rather than preserving one replacement
for each old function. The old versions remain in Git at `b3055b34d`.

## What to run

From the repository, with `i6_core`, Sisyphus and `i6_models` on `PYTHONPATH`:

```bash
pytest -q tests/test_rasr_templates.py users/berger/configs/model_combination/tests
```

This runs small CPU models, real ONNX exports and sessions, actual file/job
transformations, and comparison/selection checks. It does not submit training
or recognition jobs, download model checkpoints or require a GPU. Native RASR
and the Granite tokenizer are optional local integration assets:

| Variable | Required capability |
| --- | --- |
| `RASR_LIB_DIR` | Multi-token tree LM scoring and CTC/AED native smoke cases |
| `RASR_DYNAMIC_LIB_DIR` | Dynamic LLM scorer with casing-variant tokenization |
| `RASR_BPE_LIB_DIR` | Dynamic LLM search with continuation-marked BPE support |
| `GRANITE_TOKENIZER_DIR` | Local Granite tokenizer repository |

Each RASR variable points to a directory containing `librasr*.so`, rather than a
source checkout. These are separate because historical prototype branches do
not all implement the same interface. Unset assets skip only their integration
checks; an explicitly supplied invalid/incompatible asset fails. Use the binary
that the corresponding Sisyphus gate and recognition jobs actually use. A pass
against a different build does not validate the experiment's build.

## Stronger coverage

- Real tiny Qwen-ASR audio-conditioned scores are compared with independent full
  multimodal forwards, across unequal histories and batches.
- The word Transformer cache is compared with full-history inference, including
  multiple layers, prefix splits, padding and last-position projection. Existing
  tiny Qwen ONNX export/parity checks remain.
- An actual forward-step instance is pickled with unpickleable native handles
  and loaded in a fresh worker; native state must rebuild lazily exactly once.
- The streaming loop must send every center frame exactly once as two-dimensional
  NumPy float32, excluding history/future frames and handling the final chunk.
- STM/CTM normalization uses real files: speaker columns, labels, segment
  boundaries, normalization expansions, empty hypotheses and edit counts.
- OOV jobs actually count/reduce a tiny lexicon; actual ONNX adaptation must
  preserve total probability, merge into the correct UNK column, suppress removed
  outputs and leave history IDs and known columns untouched (opsets 17 and 18).
- ARPA tests use independently computed probabilities, numeric words, UNK history,
  optional backoff weights, spelling mismatches and invalid factors.
- Rescoring checks summed likelihood rather than per-token means, EOS weighting,
  first-pass costs and n-best cutoff. Neural probes use nonuniform output scores.
- Config tests vary inputs and check serialized topology/LM wiring and job identity;
  they do not freeze production tuning values or grid sizes. Existing dev-only
  selection, provenance, dependency/gate and paired statistical tests remain.
- Report tests inspect structured table rows and matched model sizes/normalizers,
  pending/error/excluded results and idempotent registration rather than Markdown
  wording. Template imports must resolve without depending on user research.

## What these tests do not establish

Tiny CPU fixtures do not certify full-checkpoint WER, GPU execution-provider
availability, memory use at production vocabulary/history sizes, RTF or long
recording stability. Those still require the existing Sisyphus native/model
preflights and a short recognition on the target GPU/container before launching
an expensive grid. No GPU/checkpoint validation was silently substituted by a
mock. No new expensive preflight grid is scheduled by this test cleanup.

## Bug exposed by the new tests

`BoostArpaUnknownProbabilityJob` previously subtracted `log10(factor)`, dividing
rather than multiplying UNK probability. It also inferred the predicted token
by dropping numeric fields, misclassifying numeric final words when UNK occurred
in the history. Both are fixed: use the ARPA section order and add the log10
factor, leaving backoffs and unrelated n-grams unchanged. Factors must be finite
and positive. This operation remains a decoding bias, not a renormalized LM.

A class-specific hash version invalidates the affected boosted ARPA outputs.
Historical boost-sweep results therefore need the corrected mapping or rerunning
before interpreting their nominal factor. The active model-combination graph's
1,197 job IDs were compared with the pre-cleanup graph and are unchanged.

## Validation of this cleanup

The final combined run passed all 180 cases in 13.97 seconds with the local
RASR builds and tokenizer supplied. The suite now contains 142 test definitions
(180 parameterized cases), down from 220 definitions (257 cases). The generic six-test suite was also run on the isolated template
branch without the research import compatibility layer. The live checkout used
by running jobs was not edited.
