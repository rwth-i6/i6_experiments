# Noise-controlled memristor observer probe

Target: encoder+predictor W8/A8 QAT finetune
`ffnn_transducer_qat_encoder_pred_bpe___ffnn_transducer_bpe`, cycle-1 converted checkpoint
`MemristorModelConversionJob.dM4lrkM4hNZa/output/memristor_converted_model.pt`.
Its joiner remains floating-point; predictor output activation quantizers still
use the first-evaluation observer-update path being investigated.
Use the saved cached `lexfree` greedy decode, ADC settings 8/4/4 and positional ADC 8/1/7.
Its RASR config uses `lexiconfree-timesync-beam-search`, `max-beam-size = 1`,
and `score-threshold = 0.0`. The report records 8.1% dev-other WER in all three
listed cycles (blank penalty 0), versus 43.2% for the full-QAT cycle-1 model.
The source configuration's `OfflineRasrRecogForwardStepV2` is retained.

The simulator draws fresh read noise using PyTorch `torch.randn` in both eager
and fast paths (`memristor_modules/memristor.py`). Conversion variability is
held fixed by reusing the existing converted checkpoint; no conversion is run.
Read noise remains enabled. Python, NumPy, and PyTorch CPU/CUDA RNGs are seeded
before configuration/model construction and reset immediately before inference.

Run on an allocated GPU from the QAT setup:

```bash
PYTHONPATH=/u/azim.javed/experiments/training/qat:$PYTHONPATH \
appexec hilmes \
/work/smt4/azim.javed/envs/base_env/bin/python diagnostics/observer_wer.py \
  --config alias/recognition/memristor_ffnn_transducer_qat_encoder_pred_bpe___ffnn_transducer_bpe__recog_num_cycle_1_batched_decoder_cache_lexfree/dev-other/output/returnn.config \
  --output diagnostics/observer_memristor_encpred_greedy_s17 \
  --device gpu --probe --seed 17 --repeat-current --run
```

This runs one utterance in each of `current`, `current_repeat`, and `frozen`,
sequentially. It is three bounded decodes, not three full-corpus passes.

Each directory contains:

- `observer_report.json`: first-call activation ranges/grids, including the
  encoder's training-only observer implementation and the scorer's cached one.
- `noise_report.json`: seed, inference RNG-state fingerprints, encoder input/
  output fingerprints, and first scorer input/output fingerprints.
- `search_out.ctm`, `evaluated_sequences.json`, and `console.log`.

## Interpretation

1. Compare current/current_repeat first. Their grids, fingerprints, and CTMs
   should match. If they do not, the seed control is insufficient; investigate
   remaining numerical nondeterminism or RNG sources before attributing WER
   differences to the observer policy.
2. Compare current/frozen. Their initial inference RNG states and encoder
   outputs should match. Encoder activation observers do not update in eval.
3. Inspect changed scorer ranges/scales. This directly establishes whether
   this noise realization triggers the first-input update.
4. Compare hypotheses and word errors for the evaluated utterance only.

Equal seeds do not guarantee identically matched reads after beam-search
paths diverge: changed scores can change later call shapes, hypotheses, and RNG
consumption. The first-scorer input/output fingerprints help localize the initial
difference before that divergence. Identical RNG end states alone are not proof
that every read was matched.

One seed/checkpoint/utterance is a mechanistic probe, not an expected noisy WER
estimate. If a difference appears, repeat paired seeds with fresh output dirs
before drawing distribution-level conclusions. V2's one-utterance batching is
also smaller than the historical batch and can change the noise broadcast
layout. This is a controlled probe, not an exact historical replay.
