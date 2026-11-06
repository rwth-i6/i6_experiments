"""Count zeros in weights actually passed to F.linear by loaded joiner modules.

Loads the model from its generated training configuration rather than guessing
its bit width or observer type. Compares actual first-eval weights against the
earlier saved-grid reconstruction. Does not execute a training workflow.
"""

import argparse
import gc
import json
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F


@torch.no_grad()
def audit(label, checkpoint):
    checkpoint = Path(checkpoint).resolve()
    config_file = checkpoint.parents[1] / "returnn.config"
    namespace = runpy.run_path(str(config_file))
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    model = namespace["get_model"](epoch=payload.get("epoch", 1), step=0)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    rows = []
    modules = [(name, module) for name, module in model.named_modules()
               if name.startswith(("joint_net.", "joiner.")) and hasattr(module, "weight") and module.weight.ndim == 2]
    assert len(modules) == 2, [name for name, _ in modules]
    for index, (name, layer) in enumerate(modules, 1):
        weight = layer.weight
        count = weight.numel()
        quantizer = getattr(layer, "weight_quantizer", None)
        row = {"model": label, "layer": index, "name": name, "checkpoint": str(checkpoint),
               "generated_config": str(config_file), "num_weights": count,
               "raw_zero_count": int((weight == 0).sum()), "quantized": quantizer is not None}
        if quantizer is not None:
            minimum = quantizer.observer.min_val.clone()
            maximum = quantizer.observer.max_val.clone()
            qmin, qmax = int(quantizer.quant_min), int(quantizer.quant_max)
            scale = float(torch.maximum(torch.maximum(minimum.abs(), maximum.abs()) / qmax,
                                        quantizer.observer.eps))
            manual = torch.fake_quantize_per_tensor_affine(weight, scale, 0, qmin, qmax)
            row.update(quant_min=qmin, quant_max=qmax, saved_grid_scale=scale,
                       saved_grid_zero_count=int((manual == 0).sum()),
                       initial_scale_is_none=quantizer.scale is None)
        else:
            manual = weight.detach()

        original_linear = F.linear
        captured = []

        def capture(inputs, effective_weight, bias=None):
            captured.append(effective_weight.detach().clone())
            return original_linear(inputs, effective_weight, bias)

        with patch.object(F, "linear", capture):
            layer(torch.zeros(1, weight.shape[1], dtype=weight.dtype))
        assert len(captured) == 1, len(captured)
        actual = captured[0]
        row["live_eval_zero_count"] = int((actual == 0).sum())
        row["raw_zero_percent"] = 100 * row["raw_zero_count"] / count
        row["live_eval_zero_percent"] = 100 * row["live_eval_zero_count"] / count
        row["different_zero_assignments_vs_saved_grid"] = int(((actual == 0) != (manual == 0)).sum())
        row["max_abs_weight_difference_vs_saved_grid"] = float((actual - manual).abs().max())
        if quantizer is not None:
            row["live_scale"] = float(quantizer.scale)
            row["live_zero_point"] = int(quantizer.zero_point)
            row["observer_range_changed"] = not (torch.equal(minimum, quantizer.observer.min_val)
                                                  and torch.equal(maximum, quantizer.observer.max_val))
            row["saved_grid_zero_percent"] = 100 * row["saved_grid_zero_count"] / count
        rows.append(row)
        print(label, name, "live zero %", row["live_eval_zero_percent"],
              "different zero assignments", row["different_zero_assignments_vs_saved_grid"], flush=True)
    del model, payload
    gc.collect()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", nargs=2, action="append", metavar=("LABEL", "CHECKPOINT"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
                   "/work/smt4/azim.javed/repositories/sisyphus_repo"]
    torch.set_num_threads(2)
    rows = []
    for label, checkpoint in args.model:
        print("Loading generated model:", label, flush=True)
        rows.extend(audit(label, checkpoint))
    args.output.write_text(json.dumps({"method": "exact equality on actual F.linear effective weights at first eval",
                                      "rows": rows}, indent=2) + "\n")
    lines = ["# Joiner sparsity: live-model validation", "",
             "Models are instantiated from their generated training configurations and checkpoints load strictly.",
             "Counts are taken on effective weights captured at the actual joiner F.linear call in eval mode.", "",
             "| Model | Layer | Raw exact zeros % | Saved-grid zeros % | Live eval zeros % | Different zero assignments |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        saved = f'{r["saved_grid_zero_percent"]:.5f}' if r["quantized"] else "N/A"
        lines.append(f'| {r["model"]} | {r["layer"]} | {r["raw_zero_percent"]:.5f} | {saved} '
                     f'| {r["live_eval_zero_percent"]:.5f} | {r["different_zero_assignments_vs_saved_grid"]} |')
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
