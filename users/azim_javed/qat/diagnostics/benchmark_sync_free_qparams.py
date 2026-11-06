"""CUDA wall-clock microbenchmark of old vs sync-free QAT quantizers.

Run from the QAT setup with python -m diagnostics.benchmark_sync_free_qparams.
Measures quantizer forward + STE backward, not full-model/epoch speedup. CUDA
events alone would miss CPU dispatch/synchronization costs: use perf_counter
and synchronize only at the timed region boundaries.
"""

import argparse
import copy
import statistics
import time

import torch

from .original_quantizers import OriginalWeightQuantizer, OriginalActivationQuantizer, prepare_reference, reference_forward
from recipe.model_pipelines.common.memristor_layers import ActivationQuantizer, WeightQuantizer


def benchmark(args, kind):
    if kind == "weight":
        prototype = WeightQuantizer(args.bits, torch.qint8, args.method)
        original = OriginalWeightQuantizer(args.bits, torch.qint8, args.method)
    else:
        prototype = ActivationQuantizer(args.bits, torch.qint8, args.method, None, None)
        original = OriginalActivationQuantizer(args.bits, torch.qint8, args.method, None, None)
    prototype = prototype.cuda().train()
    new = torch.nn.ModuleList(copy.deepcopy(prototype) for _ in range(args.quantizers))
    original = prepare_reference(original, "cuda").train()
    old = torch.nn.ModuleList(copy.deepcopy(original) for _ in range(args.quantizers))
    dtype = getattr(torch, args.dtype)
    x = torch.randn(*args.shape, device="cuda", dtype=dtype, requires_grad=True)

    def step(modules, optimized):
        x.grad = None
        loss = None
        for module in modules:
            out = module(x) if optimized else reference_forward(module, x)
            term = out.sum()
            loss = term if loss is None else loss + term
        loss.backward()

    # Calibrate both paths identically and exercise kernel/autograd initialization.
    for _ in range(args.warmup):
        step(old, False)
        step(new, True)
    old_ms, new_ms = [], []
    for repeat in range(args.repeats):
        # Alternate order to reduce consistent order/clock effects.
        order = [(old, False, old_ms), (new, True, new_ms)]
        if repeat % 2:
            order.reverse()
        for modules, optimized, measurements in order:
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(args.iterations):
                step(modules, optimized)
            torch.cuda.synchronize()
            measurements.append((time.perf_counter() - start) * 1000 / args.iterations)
    before, after = statistics.median(old_ms), statistics.median(new_ms)
    print(f"\n{kind}: {args.quantizers} quantizers, shape={tuple(args.shape)}, {args.dtype}, {args.bits} bits, {args.method}")
    print(f"  Original:  {before:.3f} ms/step (median)")
    print(f"  Optimized: {after:.3f} ms/step (median)")
    print(f"  Speedup:   {before / after:.3f}x; time reduction: {100 * (1 - after / before):.1f}%")
    print(f"  Original repeat ms:  {[round(t, 3) for t in old_ms]}")
    print(f"  Optimized repeat ms: {[round(t, 3) for t in new_ms]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quantizers", type=int, default=400)
    parser.add_argument("--shape", type=int, nargs=2, default=[128, 128])
    parser.add_argument("--bits", type=float, default=8)
    parser.add_argument("--method", choices=["per_tensor", "per_tensor_symmetric"], default="per_tensor_symmetric")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--kind", choices=["weight", "activation", "both"], default="both")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(args.quantizers, *args.shape, args.warmup, args.iterations, args.repeats) < 1:
        parser.error("counts and shape dimensions must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; run on an allocated GPU node")
    torch.manual_seed(19)
    print(f"PyTorch {torch.__version__}; GPU: {torch.cuda.get_device_name(0)}")
    print("Synchronized wall-clock timing, training forward + backward; excludes model/optimizer/data loading.")
    print("Original reference: fixed observer bounds converted to Python ints for CUDA compatibility; original qparam calculation retained.")
    for kind in (["weight", "activation"] if args.kind == "both" else [args.kind]):
        benchmark(args, kind)


if __name__ == "__main__":
    main()
