# Joiner activation histograms

For the independently recounted W4/A8 digital-versus-memristor comparison,
ADC measurement-point trace, and channel-wise variance analysis, see
[`AUDIT_W4_A8.md`](AUDIT_W4_A8.md). Its density-normalized figure and numerical
results are in `audit_w4_a8/`.

`plot_histograms.py` reads all appended dictionary records in the recognition
job's `joint_activations_debug.npy`, including records after an empty first
record. It counts every finite scalar activation without subsampling, using
two passes and loading one saved record at a time.

Run from `training/qat` using the base environment:

```bash
/work/smt4/azim.javed/envs/base_env/bin/python \
  diagnostics/qat_ffnn_joiner_activations/plot_histograms.py \
  --input output/recognition/memristor_qat_ffnn_transducer_full_quant_v2_lowbs__recog_num_cycle_1_batched_decoder_cache_lexfree_joint_debug/dev-other/joint_activations_debug.npy \
  --labels 'Memristor, ADC 4/4'
```

Outputs in `plots/`:

- `joiner_histograms.png`: both layers side by side.
- `linear1_histogram.png` / `.pdf`: first linear output before output
  quantization and tanh.
- `linear2_histogram.png` / `.pdf`: second linear output before output
  quantization and softmax, pooling all vocabulary entries including blank.
- `summary.json`: input path, record/batch counts, dimensions, finite/nonfinite
  counts, mean, population standard deviation, and range.
- `histogram_counts.npz`: exact counts and bin edges for reuse.

The default y-axis is probability density (histogram area equals one), with
100 equal-width bins spanning the full recorded range. Use
`--normalization count` for raw frequency or `--bins 60` to change binning.

To compare recordings, pass multiple paths to `--input` and the matching labels
to `--labels`. They will use shared bin edges for each layer. Separate search
runs describe each system's own visited histories, rather than a matched-input
hardware comparison.

Use `--exclude-special-tokens` to omit `</s>` (channel 0), `<unk>` (channel 1),
and blank (the last channel) from layer 2 in both recordings. Layer 1 has no
token channels and is unaffected. The filtered digital-versus-memristor plots
are saved in `plots_comparison_no_special_tokens/`; density is normalized using
only the retained values.

These are complete linear-layer outputs in model units, not internal hardware
PE/tile outputs. Empty records are skipped, and malformed/truncated records
raise an error. The plot includes exactly the saved calls; it cannot recover
any calls that the recognition job did not flush to disk.

## Replay the output quantizer

To process both finished recognition jobs and write each result beside its
original recording in the job's `/work/smt4/.../output/` directory, run:

```bash
bash diagnostics/qat_ffnn_joiner_activations/replay_both_output_quantizers.sh
```

Each job receives `output/joint_activations_post_quant.npy`. The wrapper runs
from that same output directory. Existing destination files are not overwritten.

`apply_output_quantizer.py` loads the scorer class, model settings, and checkpoint
specified in the run's RASR config. It monkey-patches `forward()` to call only
`joint_net[8]`, then processes the original layer-2 batches in order. This
preserves first-call observer initialization. Layer-1 arrays are retained as
recorded, and all layer-2 token channels are quantized before any filtering.
The current model code must match the implementation used by recognition.

From the setup root:

```bash
PYTHON=/work/smt4/azim.javed/envs/base_env/bin/python
WORK=/work/smt4/azim.javed/sisyphus_work
DIAG=diagnostics/qat_ffnn_joiner_activations

$PYTHON "$DIAG/apply_output_quantizer.py" \
  --input "$WORK/i6_core/returnn/forward/ReturnnForwardJobV2.9wPnV9PmMegQ/output/joint_activations_debug.npy" \
  --rasr-config "$WORK/i6_core/rasr/config/WriteRasrConfigJob.UdOpdxX7iNj1/output/rasr.config" \
  --output "$DIAG/digital_post_quant.npy"

$PYTHON "$DIAG/apply_output_quantizer.py" \
  --input "$WORK/i6_core/returnn/forward/ReturnnForwardJobV2.WzUWPLE5LRPZ/output/joint_activations_debug.npy" \
  --rasr-config "$WORK/i6_core/rasr/config/WriteRasrConfigJob.yD7XDmiGBtpZ/output/rasr.config" \
  --output "$DIAG/memristor_post_quant.npy"

$PYTHON "$DIAG/plot_histograms.py" \
  --input "$DIAG/digital_post_quant.npy" "$DIAG/memristor_post_quant.npy" \
  --labels 'Digital QAT' 'Memristor, ADC 4/4' \
  --post-output-quantization --exclude-special-tokens \
  --output-dir "$DIAG/plots_post_quant_no_special_tokens"
```

## Blank versus best competing token

`analyze_blank_margins.py --input DIGITAL.npy MEMRISTOR.npy --labels 'Digital QAT' 'Memristor, ADC 4/4'`
computes `blank_logit - max(nonblank_logits)` per call from the post-quantization
recordings. Positive means blank strictly wins, negative means a nonblank token
strictly wins, and zero is a top-score tie. Ties are reported separately because
their actual selection depends on decoder tie handling.

Results are in `blank_margins_post_quant/`: overlaid PNG/PDF histograms,
`summary.json`, and `per_call_margins.npz`. A second version omits `</s>` and
`<unk>` from the competing tokens. These statistics count recorded scorer
calls, not final traceback selections or matched-input decision changes.
