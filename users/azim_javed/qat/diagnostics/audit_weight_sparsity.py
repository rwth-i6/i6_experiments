"""Compare saved observer-grid weight sparsity without conversion or decoding.

Only weights with saved weight-quantizer observers are quantized. Floating-point
predictor/joiner weights are reported separately, never silently quantized.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import torch


def group(name):
    if name.startswith("conformer."):
        if ".mhsa." in name:
            return "encoder_mhsa"
        if ".pointwise_conv" in name or ".depthwise_conv" in name:
            return "encoder_conv"
        if ".linear_ff" in name or ".linear_out" in name:
            return "encoder_ff"
        return "encoder_other"
    if name.startswith("prediction_net."):
        return "predictor"
    if name.startswith("joint_net."):
        return "joiner"
    return name.split(".")[0]


def aggregate(rows, bits):
    count = sum(r["num_weights"] for r in rows)
    nonzero = sum(r["nonzero_weights"] for r in rows)
    return {
        "layers": len(rows), "num_weights": count, "nonzero_weights": nonzero,
        "nonzero_percent": 100 * nonzero / count,
        "zero_percent": 100 * (count - nonzero) / count,
        "weighted_quantized_rms": math.sqrt(sum(r["quantized_rms"] ** 2 * r["num_weights"] for r in rows) / count),
        "bitplane_active_percent": [
            100 * sum(r["bitplane_active_counts"][bit] for r in rows) / count for bit in range(bits - 1)
        ],
        "layers_below_10_percent_nonzero": sum(r["nonzero_percent"] < 10 for r in rows),
        "layers_below_1_percent_nonzero": sum(r["nonzero_percent"] < 1 for r in rows),
        "layers_with_empty_highest_plane": sum(r["bitplane_active_counts"][-1] == 0 for r in rows),
        "layers_current_range_exceeds_saved_observer": sum(r["current_absmax"] > r["observer_absmax"] for r in rows),
    }


@torch.no_grad()
def audit(label, bits, checkpoint):
    print(f"Loading {label}: {checkpoint}", flush=True)
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    state = payload.get("model", payload)
    qmax = 2 ** (bits - 1) - 1
    rows, floating = [], []
    for key, weight in state.items():
        if not key.endswith(".weight") or weight.ndim < 2:
            continue
        name = key[:-len(".weight")]
        p = name + ".weight_quantizer.observer."
        if p + "min_val" not in state:
            if name == "token_embedding" or name.startswith(("prediction_net.", "joint_net.")):
                floating.append({"name": name, "shape": list(weight.shape), "num_weights": weight.numel(),
                                 "exact_zero_percent": 100 * float((weight == 0).float().mean()),
                                 "rms": float(weight.square().mean().sqrt()), "quantized": False})
            continue
        assert state[p + "min_val"].numel() == 1, f"Per-channel observer unsupported: {name}"
        bound = torch.maximum(state[p + "min_val"].abs(), state[p + "max_val"].abs())
        assert torch.isfinite(bound).all(), f"Uninitialized observer: {name}"
        scale = float(torch.maximum(bound / qmax, state[p + "eps"]))
        qw = torch.fake_quantize_per_tensor_affine(weight, scale, 0, -qmax, qmax)
        codes = (qw / scale).round().to(torch.int32)
        assert int(codes.abs().max()) <= qmax
        signed_counts = torch.bincount((codes + qmax).flatten().long(), minlength=2 * qmax + 1).tolist()
        absolute_counts = torch.bincount(codes.abs().flatten().long(), minlength=qmax + 1).tolist()
        count, nonzero = weight.numel(), weight.numel() - absolute_counts[0]
        flat = codes.reshape(codes.shape[0], -1)
        current_bound = weight.abs().max()
        current_scale = float(torch.maximum(current_bound / qmax, state[p + "eps"]))
        current_qw = torch.fake_quantize_per_tensor_affine(weight, current_scale, 0, -qmax, qmax)
        row = {
            "name": name, "group": group(name), "shape": list(weight.shape),
            "num_weights": count, "nonzero_weights": nonzero,
            "nonzero_percent": 100 * nonzero / count, "zero_percent": 100 * (count - nonzero) / count,
            "floating_exact_zero_percent": 100 * float((weight == 0).float().mean()),
            "scale": scale, "observer_absmax": float(bound), "current_absmax": float(current_bound),
            "observer_to_current_absmax": float(bound / current_bound) if current_bound > 0 else None,
            "floating_rms": float(weight.square().mean().sqrt()),
            "quantized_rms": float(qw.square().mean().sqrt()),
            "max_absolute_code": int(codes.abs().max()), "used_signed_codes": sum(n > 0 for n in signed_counts),
            "signed_code_counts": signed_counts, "absolute_code_counts": absolute_counts,
            "bitplane_active_counts": [sum(n for code, n in enumerate(absolute_counts) if code & (1 << bit))
                                       for bit in range(bits - 1)],
            "all_zero_output_rows": int((flat == 0).all(dim=1).sum()),
            "output_rows": flat.shape[0],
            # Hypothetical diagnostic only; no observers/checkpoints are modified.
            "current_range_regrid_nonzero_percent": 100 * float((current_qw != 0).float().mean()),
        }
        rows.append(row)
    assert rows, f"No observed quantized weights in {checkpoint}"
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    # The auxiliary encoder_output head is not used by recognition and must not
    # enter the conformer comparison (it is quantized only in the full-QAT model).
    encoder = [r for r in rows if r["name"].startswith("conformer.")]
    result = {
        "label": label, "bits": bits, "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_epoch": payload.get("epoch"),
        "quantization": "per-tensor symmetric; saved observer grid; no conversion-time embedding regrid",
        "summary": aggregate(rows, bits), "encoder_summary": aggregate(encoder, bits) if encoder else None,
        "groups": {name: aggregate(items, bits) for name, items in groups.items()},
        "layers": rows, "floating_prediction_joiner": floating,
        "sparsest_encoder_layers": [r["name"] for r in sorted(encoder, key=lambda r: r["nonzero_percent"])[:10]],
    }
    print(label, "encoder nonzero %", result["encoder_summary"]["nonzero_percent"],
          "encoder bitplane activity %", result["encoder_summary"]["bitplane_active_percent"], flush=True)
    return result


def write_report(results, output):
    output.write_text(json.dumps({"models": results}, indent=2) + "\n")
    fields = ["model", "name", "group", "num_weights", "nonzero_percent", "zero_percent", "scale",
              "observer_absmax", "current_absmax", "observer_to_current_absmax", "quantized_rms",
              "max_absolute_code", "all_zero_output_rows", "current_range_regrid_nonzero_percent"]
    with output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for model in results:
            for row in model["layers"]:
                writer.writerow(dict(model=model["label"], **{k: row[k] for k in fields if k != "model"}))
    lines = ["# W4/A8 fine-tuning sparsity diagnostic", "",
             "Sparsity is measured after quantizing weights on their saved observer grids.",
             "Percentages below are parameter-count weighted, not averages over layers.", "",
             "## Encoder comparison", "",
             "| Model | Epoch (subepoch) | Quantized layers | Nonzero % | Zero % | Bit 0 active % | Bit 1 active % | Bit 2 active % |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model in results:
        r = model["encoder_summary"]
        planes = r["bitplane_active_percent"]
        lines.append(f'| {model["label"]} | {model["checkpoint_epoch"]} | {r["layers"]} | {r["nonzero_percent"]:.3f} '
                     f'| {r["zero_percent"]:.3f} | {planes[0]:.3f} | {planes[1]:.3f} | {planes[2]:.3f} |')
    for model in results:
        lines += ["", f'## {model["label"]}', "", f'Checkpoint: `{model["checkpoint"]}`', "",
                  "| Group | Layers | Nonzero % | Layers below 10% nonzero | Empty highest bitplane |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for name, r in model["groups"].items():
            lines.append(f'| {name} | {r["layers"]} | {r["nonzero_percent"]:.3f} '
                         f'| {r["layers_below_10_percent_nonzero"]} | {r["layers_with_empty_highest_plane"]} |')
        scorer_rows = [r for r in model["layers"] if r["group"] in ("token_embedding", "predictor", "joiner")]
        if scorer_rows:
            lines += ["", "### Quantized embedding, predictor and joiner", "",
                      "| Layer | Nonzero % | Weight scale | Observer/current absmax | Quantized RMS | Highest bit active % |",
                      "| --- | ---: | ---: | ---: | ---: | ---: |"]
            for r in scorer_rows:
                highest = 100 * r["bitplane_active_counts"][-1] / r["num_weights"]
                lines.append(f'| `{r["name"]}` | {r["nonzero_percent"]:.3f} | {r["scale"]:.6f} '
                             f'| {r["observer_to_current_absmax"]:.3f} | {r["quantized_rms"]:.6f} | {highest:.5f} |')
        lines += ["", "### Ten sparsest encoder layers", "",
                  "| Layer | Nonzero % | Observer/current absmax | Zero output rows |",
                  "| --- | ---: | ---: | ---: |"]
        by_name = {r["name"]: r for r in model["layers"]}
        for name in model["sparsest_encoder_layers"]:
            r = by_name[name]
            lines.append(f'| `{name}` | {r["nonzero_percent"]:.3f} | {r["observer_to_current_absmax"]:.3f} '
                         f'| {r["all_zero_output_rows"]}/{r["output_rows"]} |')
        if model["floating_prediction_joiner"]:
            lines += ["", "Embedding/predictor/joiner weights are floating point in this model:"]
            lines += [f'- `{r["name"]}`: {r["exact_zero_percent"]:.4f}% exact zeros (not W4 sparsity).'
                      for r in model["floating_prediction_joiner"]]
    lines += ["", "## Interpretation", "",
              "This measures numerical sparsity, not explicit pruning or memristor programming error.",
              "An active bitplane percentage counts weights with that magnitude bit set, independently of sign.",
              "The CSV has every quantized layer; JSON additionally includes code histograms and bitplane counts.",
              "The current-range regrid column is a hypothetical comparison only, not a recommended model change.",
              "Saved observer ranges are held fixed, matching conversion; the JSON flags layers whose current",
              "weights exceed those ranges and would expand a lazily initialized eval observer.", ""]
    output.with_suffix(".md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", nargs=3, metavar=("LABEL", "BITS", "CHECKPOINT"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    results = []
    for label, bits, checkpoint in args.model:
        results.append(audit(label, int(bits), checkpoint))
        write_report(results, args.output)
    print("Results:", args.output, args.output.with_suffix(".csv"), args.output.with_suffix(".md"))


if __name__ == "__main__":
    main()
