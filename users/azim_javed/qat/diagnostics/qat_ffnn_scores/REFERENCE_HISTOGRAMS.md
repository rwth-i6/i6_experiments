# Reference-conditioned logit histograms

Run from `training/qat`:

```bash
/work/smt4/azim.javed/envs/base_env/bin/python \
  diagnostics/qat_ffnn_scores/plot_reference_logit_histograms.py \
  --digital-dir output/diagnostics/qat_ffnn_scores/full_corpus/digital_model \
  --memristor-dir output/diagnostics/qat_ffnn_scores/full_corpus/base_model \
  --output-dir diagnostics/qat_ffnn_scores/reference_logit_histograms_full_corpus
```

The script uses every shared utterance and all reference-grid states, checks
reference labels/shapes/blank IDs, and produces both all-token and ordinary-token
histograms. It compares each experiment's own complete joint-network logits
before softmax. It does not sample, rescale logits, or share encoder vectors.

## Performance and interruption

- Reads each pair of NPZ files once; memory is bounded by the current utterance.
- Uses NumPy's fast uniform-bin histogram implementation.
- Counts all channels once, then subtracts the three special channels to obtain
  ordinary-token counts, instead of processing the full tensor twice.
- Computes float64 moments without allocating a full float64 tensor copy.
- Prints elapsed time and estimated remaining time.
- Atomically saves `checkpoint.npz` every 100 utterances and at completion.
  Change this interval with `--checkpoint-every`.

To continue after interruption, rerun the same command with `--resume`.
The checkpoint checks the source directories, manifests, and bin settings.
If a completed checkpoint is resumed, plots are regenerated without rereading
the logits. Use a different output directory for different bin settings.

The default bins are 128 equal-width bins over `[-64, 64]`. This fixed range
allows exact single-pass accumulation. Any value outside the range raises an
error; it is never silently excluded. Change it with `--value-range MIN MAX`.
The plotted x-axis is limited to the observed min/max, with all counts retained.

Outputs: `all_tokens_histogram.png` / `.pdf`,
`ordinary_tokens_histogram.png` / `.pdf`, `histogram_counts.npz`, `summary.json`,
and the resumable `checkpoint.npz`.
