# Activation observer initialization: paired dev-other recognition

Target: `qat_ffnn_transducer_full_quant_v2_lowbs`, INT8, epoch 2000.
Saved no-LM tree recognition: 6.2% rounded WER, beam 1024, ILM 0.

`observer_wer.py` prepares isolated copies of the generated recognition config.
It instruments the shared activation class before the RASR scorer is created.
Current mode preserves the original forward; frozen mode calculates activation
qparams from the checkpoint observers before their first evaluation forward.
Both retain the existing weight path and PyTorch qparam formulas.

The main activation quantizer has now been patched to update observers only in
training. New `current` runs use that patched production forward, so they no
longer reproduce the old first-evaluation observer update. The existing digital
and memristor result reports were produced before this patch.

Run from the QAT setup on an allocated GPU, using the normal cluster runtime
and the base_env Python. The compiled `librasr` installation and other paths are
taken from the saved recognition config. No Sisyphus manager is required.

## One-utterance probe

```bash
export PYTHONPATH=/u/azim.javed/experiments/training/qat:$PYTHONPATH
/work/smt4/azim.javed/envs/base_env/bin/python diagnostics/observer_wer.py \
  --config alias/recognition/qat_ffnn_transducer_full_quant_v2_lowbs__recog_tree_mbs1024/dev-other/output/returnn.config \
  --output diagnostics/observer_wer_probe_nosis_gpu --device gpu --probe --run
```

Each mode writes `observer_report.json` and `console.log`. The probe deliberately
exits after one utterance and does **not** produce corpus WER.
Reports record saved and observed ranges, scales, zero points, and changed-grid
counts for the quantizers actually visited. In frozen mode `changed_qparams`
should be zero. Unvisited auxiliary-head quantizers are not initialized.

## Bounded paired decode (25 utterances per mode)

```bash
/work/smt4/azim.javed/envs/base_env/bin/python diagnostics/observer_wer.py \
  --config alias/recognition/qat_ffnn_transducer_full_quant_v2_lowbs__recog_tree_mbs1024/dev-other/output/returnn.config \
  --output diagnostics/observer_wer_sample25 --device gpu --max-sequences 25 --run
```

Use fresh output directories for new attempts. Each mode runs sequentially on
the same device and retains the original sorted dataset ordering. Bounded runs
use one utterance per batch so they do not decode a large batch before stopping.
The original offline step already forwards each utterance separately through
the encoder, so limiting batches does not itself turn batched encoder calibration
into single-utterance calibration. This remains a controlled sample experiment,
not an exact historical replay of the original code and CPU runtime.
The first 25 sorted utterances are also length-biased, not a representative
random sample. Results establish effects only on those utterances.
Use `--full-corpus` explicitly for original batching and full-corpus decoding.
Outputs
include `search_out.ctm`, `observer_report.json`, `evaluated_sequences.json`
(bounded runs), and runtime logs. Send the two
reports and CTMs back for exact word-error and hypothesis comparisons. For
sample WER, restrict the reference STM to the evaluated utterances; scoring a
partial CTM against the whole dev-other reference would give invalid results.

The old 6.2% is context, not the controlled baseline: this comparison uses the
current source code and GPU runtime in **both** modes. The original decode was
CPU. A later historical reproduction should retain its original runtime too.

Original scoring artifacts:

- Reference STM: `/work/smt4/azim.javed/sisyphus_work/i6_core/corpus/convert/CorpusToStmJob.cWMBwQxemXAT/output/corpus.stm`
- SCTK bin directory: `/work/smt4/azim.javed/sisyphus_work/i6_core/tools/compile/MakeJob.fiO6GhIoeLx4/output/bin`
- Original scoring job: `/work/smt4/azim.javed/sisyphus_work/i6_core/recognition/scoring/ScliteJob.il48bT6vOT1Z`

No production quantizer or existing recognition output is modified.

The generated config uses literal RASR paths and the saved serialized model cfg.
It does not import Sisyphus or the training workflow. Only runtime definitions
are extracted from `common/recog.py` to retain its decoding/CTM behavior without
importing the workflow Job classes.
Data loading uses `torch_dataloader_opts = {"num_workers": 0}` so the diagnostic
config and instrumentation do not need serialization into spawned workers.
