# Sync-free QAT: original-source comparison

## Reference provenance

`original_quantizers.py` extracts independent classes and the range helper from
Git blobs, without importing or inheriting from optimized quantizers:

- Regular quantizers: `38e11aa25dba090010687d9cd4e38ec7f17434bd`.
- Mem-initialized activation: `6a2ed63015c2b09237bed54ad0e66b265bcc69e6`.

Before the optimization, the workspace already differed from the regular Git
blob: activation observers updated only during training, and qparams were
recomputed every forward. That exact forward behavior, recorded in the initial
workspace read, is reapplied explicitly to the reference. This is not a claimed
pristine snapshot of the uncommitted workspace. The original weight class and
mem-initialized activation class require no such adjustment.

The initial test reused optimized constructors and reconstructed the old
forward. It verified observer math but was not an independent module comparison.
Both tests and benchmark now instantiate independent original classes.

## Preserved behavior

- Nonempty, valid observer states: scale/zero-point formulas, shapes and dtypes;
  outputs, STE gradients, min/max accumulation and moving-average updates.
- Original integer conversion order in affine zero-point calculation.
- Separate original fake-quantization bounds and reduced observer bounds.
- Activation FP32 cast under disabled autocast.
- Existing train/eval policy, including weight scale caching and initial weight
  observer update when no scale exists, even in evaluation.
- Original checkpoint keys and values, strict loading in both directions.
- Explicit `set_scale_and_zp()` still uses the original observer calculation.
- Regular uncalibrated activation fallback `(scale=1, zero_point=0)`.
- Mem-initialized uncalibrated activation evaluation and empty uninitialized
  weights still raise the original initialization assertion. Cold mem-initialized
  eval and empty-input paths retain validation; normal training does not.

## Deliberate differences / limits

- Qparams are calculated on-device without observer scalar validation/materialization
  during ordinary forwards. Corrupt/inverted/NaN observer ranges are not claimed
  to reproduce original validation errors; removing hot-path validation is part
  of the optimization.
- Observer bounds are Python ints instead of CPU tensors. Quantizer public
  `quant_min`/`quant_max` attributes remain their original int32 tensors. No
  fractional scale/min/max is cast to an integer.
- The old CUDA observer path fails with CPU bounds for affine quantization,
  including per-tensor affine on the user's PyTorch version. All CUDA reference
  observers therefore convert fixed bounds to Python ints. Original observer
  qparam calculation and its synchronization remain intact. CPU per-tensor
  references receive no compatibility repairs.
- The old per-channel activation forward omitted `axis`. Per-channel references
  explicitly repair that argument as well; they check the intended observer
  math, not the broken raw forward.

## Verification

Run `python -m diagnostics.test_sync_free_qparams --require-cuda` on a GPU node.
The CPU checks pass locally; CUDA equivalence and wall-clock speedup require a
GPU run. The benchmark compares independent old/new per-tensor modules, applies
the same CUDA bound compatibility repair to the old modules before timing, and
prints that qualification. It does not use the optimized qparam helper in the
reference.

The expanded suite (`test_sync_free_qparams_edges.py`, automatically included)
also checks:

- Fresh zero/constant, positive-only, negative-only, tiny/subnormal and large
  finite ranges, plus seeded vector qparam ranges spanning 60 orders of magnitude.
- Precision labels 1.5, 2, 2.5, 3, 3.5, 4, 8; signed/unsigned modes; reduced ranges.
- Half-step rounding ties, adjacent representable inputs, clipping and STE masks.
- Scalar, transposed, sliced and channels-last inputs; FP16/BF16/FP32/FP64;
  CPU/CUDA autocast; dtype moves and CPU-to-CUDA move before first forward.
- Per-channel axes 0/1/2 and cumulative/moving observers with averaging constants
  None, 0, 0.01 and 1, including empty and uncalibrated cases.
- Actual torch.save/torch.load roundtrips into fresh modules, evaluation then
  resumed training, and loads into already-used modules.
- Multiple AdamW updates through a fake-quantized linear pipeline.
- Dispatched-op recording: no `_local_scalar_dense`/`item` in fast qparams or
  ordinary cumulative-MinMax training forwards; original forwards provide a
  positive control that does record scalar reads.
- No observer/qparam autograd history; original constructor rejection behavior.
- Finite comparisons include byte equality, distinguishing signed zero. Nonfinite
  qparams are checked separately with equal-NaN comparisons, not claimed as
  meaningful bit-exact quantized training outputs.

Important inherited behavior and scope limits:

- MovingAverageMinMaxObserver itself performs tensor-Boolean checks, so its
  updates can still synchronize. The optimization removes qparam scalar reads;
  it does not replace observer update logic.
- Weight scale/zero_point are ordinary cached attributes. Loading a checkpoint
  into an already-used weight quantizer during eval does not invalidate its old
  scale in the original implementation; this optimization preserves that policy.
  Fresh loads and subsequent training recomputation are tested separately.
- Quantization bounds are static constructor configuration. Runtime changes to
  observer bounds are not supported by the cached integer configuration.
- Distributed execution, torch.compile/TorchScript, and full model ONNX/hardware
  export are not exercised by these unit checks. Explicit conversion qparam
  calculation is tested, but that is not a full export integration test.
- Finite tests cannot prove equivalence for every possible input. CUDA cases are
  only confirmed after the suite is run on a GPU; GPU timing is a separate check.
