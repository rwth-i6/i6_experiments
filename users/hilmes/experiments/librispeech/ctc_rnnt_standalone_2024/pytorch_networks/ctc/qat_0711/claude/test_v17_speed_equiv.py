"""Equivalence and speed tests: v17 (vectorized, sync-free QAT) vs v16 quantizers.

Run from the repo root:
  PYTHONPATH=recipe:i6_models_full:/u/hilmes/src/MiniReturnn python3 \
      recipe/.../ctc/qat_0711/claude/test_v17_speed_equiv.py

Checks, for scalar WeightQuantizer, TiledWeightQuantizer (uniform + mixed grids),
and ActivationQuantizer (min/max + moving average, symmetric + affine):
  - identical outputs in train and eval mode over multiple steps (bit-exact)
  - identical weight gradients (STE)
  - identical state_dict keys and values (packed state synced back to observers)
  - cross-loading: v16 checkpoint -> v17 module and vice versa
  - get_tile_quantizer/set_scale_and_zp expose the same per-tile scales
plus a wall-clock micro-benchmark of the tiled quantizer.
"""

import sys
import time
from pathlib import Path

_here = Path(__file__).resolve()
_exp_root = _here.parents[11]  # .../experiments/asr_2023
for p in [_exp_root / "recipe", _exp_root / "i6_models_full", Path("/u/hilmes/src/MiniReturnn")]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch

_pkg = "i6_experiments.users.hilmes.experiments.librispeech.ctc_rnnt_standalone_2024.pytorch_networks.ctc.qat_0711.claude"
import importlib

SMP16 = importlib.import_module(f"{_pkg}.memristor_v16_dynmic_prec_cfg").SubMatrixPrecision
SMP17 = importlib.import_module(f"{_pkg}.memristor_v17_dynmic_prec_cfg").SubMatrixPrecision
m16 = importlib.import_module(f"{_pkg}.memristor_v16_dynmic_prec_modules")
m17 = importlib.import_module(f"{_pkg}.memristor_v17_dynmic_prec_modules")

torch.manual_seed(0)


def make_pair(grid, out_f, in_f, method="per_tensor_symmetric"):
    q16 = m16.TiledWeightQuantizer(
        SMP16(grid=grid, out_features=out_f, in_features=in_f, name="t16"), torch.qint8, method
    )
    q17 = m17.TiledWeightQuantizer(
        SMP17(grid=grid, out_features=out_f, in_features=in_f, name="t17"), torch.qint8, method
    )
    return q16, q17


def check(name, cond):
    print(f"  {'OK ' if cond else 'FAIL'} {name}")
    assert cond, name


def compare_tiled(grid, out_f, in_f, method="per_tensor_symmetric", steps=4):
    print(f"TiledWeightQuantizer {out_f}x{in_f} grid={grid[0]}... method={method}")
    q16, q17 = make_pair(grid, out_f, in_f, method)
    for step in range(steps):
        w = torch.randn(out_f, in_f) * (1.0 + 0.3 * step)
        w16 = w.clone().requires_grad_(True)
        w17 = w.clone().requires_grad_(True)
        out16, out17 = q16(w16), q17(w17)
        check(f"train step {step} outputs bit-equal", torch.equal(out16, out17))
        g = torch.randn_like(out16)
        out16.backward(g)
        out17.backward(g)
        check(f"train step {step} grads bit-equal", torch.equal(w16.grad, w17.grad))
    q16.eval(), q17.eval()
    w = torch.randn(out_f, in_f)
    check("eval outputs bit-equal", torch.equal(q16(w), q17(w)))
    q16.train(), q17.train()

    sd16, sd17 = q16.state_dict(), q17.state_dict()
    check("state_dict keys equal", set(sd16) == set(sd17))
    check("state_dict values equal", all(torch.equal(sd16[k], sd17[k]) for k in sd16))

    # cross-loading both directions
    q17b = m17.TiledWeightQuantizer(
        SMP17(grid=grid, out_features=out_f, in_features=in_f, name="t17b"), torch.qint8, method
    )
    q17b.load_state_dict(sd16)
    q16b = m16.TiledWeightQuantizer(
        SMP16(grid=grid, out_features=out_f, in_features=in_f, name="t16b"), torch.qint8, method
    )
    q16b.load_state_dict(sd17)
    q16.eval(), q17b.eval(), q16b.eval()
    w = torch.randn(out_f, in_f)
    check("v16 ckpt -> v17 module output equal", torch.equal(q16(w), q17b(w)))
    check("v17 ckpt -> v16 module output equal", torch.equal(q16(w), q16b(w)))

    q16.set_scale_and_zp()
    q17.set_scale_and_zp()
    scales_equal = all(
        torch.equal(q16.quantizers[r][c].scale, q17.get_tile_quantizer(r, c).scale)
        and torch.equal(q16.quantizers[r][c].zero_point, q17.get_tile_quantizer(r, c).zero_point)
        for r in range(len(grid)) for c in range(len(grid[0]))
    )
    check("per-tile scales/zps equal (conversion path)", scales_equal)


def compare_scalar_weight(method):
    print(f"WeightQuantizer scalar method={method}")
    q16 = m16.WeightQuantizer(6, torch.qint8, method)
    q17 = m17.WeightQuantizer(6, torch.qint8, method)
    for step in range(3):
        w = torch.randn(64, 64) * (1.0 + step)
        check(f"step {step} outputs bit-equal", torch.equal(q16(w.clone()), q17(w.clone())))
    q16.set_scale_and_zp(), q17.set_scale_and_zp()
    check("scale equal", torch.equal(q16.scale, q17.scale))
    check("zero_point equal", torch.equal(q16.zero_point, q17.zero_point))


def compare_activation(method, moving):
    print(f"ActivationQuantizer method={method} moving_avrg={moving}")
    q16 = m16.ActivationQuantizer(8, torch.qint8, method, channel_axis=None, moving_avrg=moving)
    q17 = m17.ActivationQuantizer(8, torch.qint8, method, channel_axis=None, moving_avrg=moving)
    for step in range(4):
        x = torch.randn(3, 50, 128) * (1.0 + 0.5 * step)
        check(f"step {step} outputs bit-equal", torch.equal(q16(x.clone()), q17(x.clone())))
    q16.eval(), q17.eval()
    x = torch.randn(3, 50, 128)
    check("eval outputs bit-equal", torch.equal(q16(x), q17(x)))


def benchmark(grid, out_f, in_f, iters=30):
    q16, q17 = make_pair(grid, out_f, in_f)
    w = torch.randn(out_f, in_f, requires_grad=True)
    for q in (q16, q17):
        for _ in range(3):
            q(w).sum().backward()  # warmup
    results = []
    for name, q in (("v16", q16), ("v17", q17)):
        t0 = time.perf_counter()
        for _ in range(iters):
            q(w).sum().backward()
        results.append((name, (time.perf_counter() - t0) / iters * 1000))
    print(f"benchmark {out_f}x{in_f} ({len(grid) * len(grid[0])} tiles, fwd+bwd, CPU): "
          + ", ".join(f"{n} {t:.1f}ms" for n, t in results)
          + f"  -> {results[0][1] / results[1][1]:.1f}x")


if __name__ == "__main__":
    uniform6 = [[6] * 4 for _ in range(16)]           # 2048x512 ff-style, one bit width
    mixed = [[8] * 4 for _ in range(4)] + [[6] * 4 for _ in range(6)] + [[4] * 4 for _ in range(2)]  # taper-style
    compare_tiled(uniform6, 2048, 512)
    compare_tiled(mixed, 1536, 512)
    compare_tiled([[6, 4], [4, 8]], 256, 256, method="per_tensor")  # affine path
    compare_scalar_weight("per_tensor_symmetric")
    compare_scalar_weight("per_tensor")
    compare_activation("per_tensor_symmetric", moving=None)
    compare_activation("per_tensor_symmetric", moving=0.01)
    compare_activation("per_tensor", moving=None)
    benchmark(uniform6, 2048, 512)
    benchmark(mixed, 1536, 512)
    print("all checks passed")
