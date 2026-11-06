"""Stored floating-point joiner weights: exact zeros, near-zero mass, and QAT grids."""

import argparse
import csv
import json
from pathlib import Path

import torch


@torch.no_grad()
def audit(label, bits, checkpoint):
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)["model"]
    names = sorted(k[:-len(".weight")] for k, w in state.items()
                   if k.startswith("joint_net.") and k.endswith(".weight") and w.ndim == 2)
    assert len(names) == 2, names
    rows = []
    for index, name in enumerate(names, 1):
        weight = state[name + ".weight"]
        assert torch.isfinite(weight).all(), name
        absolute = weight.abs().flatten()
        count = weight.numel()
        counts = {str(t): int((absolute < t).sum()) for t in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2)}
        quantiles = torch.quantile(absolute, torch.tensor([0.1, 0.5, 0.9]))
        row = {
            "model": label, "bits": bits or None, "layer": index, "name": name,
            "checkpoint": str(Path(checkpoint).resolve()), "num_weights": count,
            "exact_zero_count": int((weight == 0).sum()),
            "exact_zero_percent": 100 * int((weight == 0).sum()) / count,
            "near_zero_counts": counts,
            "near_zero_percent": {t: 100 * n / count for t, n in counts.items()},
            "p10_abs": float(quantiles[0]), "median_abs": float(quantiles[1]),
            "p90_abs": float(quantiles[2]), "max_abs": float(absolute.max()),
            "rms": float(weight.square().mean().sqrt()),
        }
        if bits:
            p = name + ".weight_quantizer.observer."
            bound = torch.maximum(state[p + "min_val"].abs(), state[p + "max_val"].abs())
            qmax = 2 ** (bits - 1) - 1
            scale = float(torch.maximum(bound / qmax, state[p + "eps"]))
            qw = torch.fake_quantize_per_tensor_affine(weight, scale, 0, -qmax, qmax)
            row["quantization_scale"] = scale
            row["zero_bin_half_width"] = scale / 2
            row["post_quant_zero_count"] = int((qw == 0).sum())
            row["post_quant_zero_percent"] = 100 * row["post_quant_zero_count"] / count
            # Reciprocal-scale float32 arithmetic can move a half-step boundary
            # by an ULP. Treat fake quantization as authoritative and check that
            # any discrepancy with the analytic zero bin is confined to ties.
            boundary_difference = (qw.flatten() == 0) != (absolute <= scale / 2)
            row["zero_bin_boundary_roundoff_count"] = int(boundary_difference.sum())
            assert torch.isclose(
                absolute[boundary_difference], torch.tensor(scale / 2, dtype=absolute.dtype),
                atol=4 * torch.finfo(absolute.dtype).eps * scale, rtol=0,
            ).all()
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", nargs=3, metavar=("LABEL", "BITS_OR_0_FOR_FP", "CHECKPOINT"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    rows = []
    for label, bits, checkpoint in args.model:
        print("Auditing", label, flush=True)
        rows.extend(audit(label, int(bits), checkpoint))
    args.output.write_text(json.dumps({"near_zero_definition": "strict absolute threshold, including exact zeros",
                                      "rows": rows}, indent=2) + "\n")
    lines = ["# Pre-quantization joiner weight distributions", "",
             "Measured from stored floating-point parameters, before fake quantization.",
             "Near-zero percentages include exact zeros; thresholds are strict absolute values.", "",
             "| Model | Joiner layer | Exact zero % | abs(w) < 0.001 % | abs(w) < 0.01 % | Median abs(w) | RMS |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f'| {r["model"]} | {r["layer"]} | {r["exact_zero_percent"]:.4f} '
                     f'| {r["near_zero_percent"]["0.001"]:.2f} | {r["near_zero_percent"]["0.01"]:.2f} '
                     f'| {r["median_abs"]:.6f} | {r["rms"]:.6f} |')
    lines += ["", "## Quantization zero bins", "",
              "Saved observer grids, without regridding. These are separate from exact floating-point zeros.", "",
              "| Model | Joiner layer | Zero-bin half-width | Post-quantization zero % |",
              "| --- | ---: | ---: | ---: |"]
    for r in rows:
        if r["bits"]:
            lines.append(f'| {r["model"]} | {r["layer"]} | {r["zero_bin_half_width"]:.6f} '
                         f'| {r["post_quant_zero_percent"]:.2f} |')
    lines += ["", "## Checkpoints", ""]
    lines += [f'- {r["model"]}: `{r["checkpoint"]}`' for r in rows if r["layer"] == 1]
    lines += ["", "Absolute thresholds compare raw weight units; models can have different learned scales.",
              "The zero-bin table shows why exact floating-point zeros and quantized zeros differ.", ""]
    args.output.with_suffix(".md").write_text("\n".join(lines))
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "layer", "exact_zero_percent", "abs_lt_0.001_percent", "abs_lt_0.01_percent",
                         "median_abs", "rms", "zero_bin_half_width", "post_quant_zero_percent"])
        for r in rows:
            writer.writerow([r["model"], r["layer"], r["exact_zero_percent"], r["near_zero_percent"]["0.001"],
                             r["near_zero_percent"]["0.01"], r["median_abs"], r["rms"],
                             r.get("zero_bin_half_width"), r.get("post_quant_zero_percent")])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
