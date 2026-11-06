"""Audit existing full-QAT converted states without decoding or submitting jobs.

Constructs recognition modules on meta, checks every required checkpoint key and
shape, then audits predictor/joiner grids, biases, observers and bitplane signal.
Meta-only quantizer bounds are created on CPU because PyTorch validates them via
scalar conversion; production code is not patched outside this process.
"""

import argparse
import dataclasses
import json
import math
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch


@torch.no_grad()
def probe_second_prediction_linear(digital, converted, bits):
    """Matched real token-history inputs; isolate programming, with no read noise."""
    import synaptogen_ml
    from synaptogen_ml.memristor_modules import TiledMemristorLinear, DacAdcHardwareSettings

    def quant_weight(name):
        p = name + "."
        obs = p + "weight_quantizer.observer."
        qmax = 2 ** (bits - 1) - 1
        bound = torch.maximum(digital[obs + "min_val"].abs(), digital[obs + "max_val"].abs())
        scale = float(torch.maximum(bound / qmax, digital[obs + "eps"]))
        return torch.fake_quantize_per_tensor_affine(digital[p + "weight"], scale, 0, -qmax, qmax)

    def quant_activation(x, name):
        obs = name + ".observer."
        bound = torch.maximum(digital[obs + "min_val"].abs(), digital[obs + "max_val"].abs())
        scale = float(torch.maximum(bound / 127, digital[obs + "eps"]))
        return torch.fake_quantize_per_tensor_affine(x, scale, 0, -127, 127)

    ids = torch.linspace(0, digital["token_embedding.weight"].shape[0] - 1, 16).long()
    emb = torch.nn.functional.embedding(ids, quant_weight("token_embedding"))
    first = torch.nn.functional.linear(quant_activation(emb, "prediction_net.1"),
                                       quant_weight("prediction_net.2"), digital["prediction_net.2.bias"])
    x = torch.tanh(quant_activation(first, "prediction_net.3"))
    reference = torch.nn.functional.linear(quant_activation(x, "prediction_net.6"),
                                           quant_weight("prediction_net.7"), digital["prediction_net.7.bias"])
    layer = TiledMemristorLinear(640, 640, bits, DacAdcHardwareSettings(8, 4, 4, 0.6, 8020.0), 128, 128)
    prefix = "prediction_net.7."
    layer.load_state_dict({k[len(prefix):]: v for k, v in converted.items() if k.startswith(prefix)}, strict=True)
    layer.initialized = True
    layer.eval()
    for pair in layer.memristors:
        pair.pos.BW = pair.neg.BW = 0.0
    synaptogen_ml.set_fast_inference(False)
    # Exact signed digital integers on the grid used by the converted layer.
    scale = float(converted[prefix + "output_factor"]) * float(converted[prefix + "input_factor"])
    integers = (quant_weight("prediction_net.7") / scale).round().T
    original = [(pair.pos.r.detach().clone(), pair.neg.r.detach().clone()) for pair in layer.memristors]
    records = []
    for mode in ("saved_programming", "zero_bit_leakage_removed", "ideal_programming"):
        if mode != "saved_programming":
            for bi, bit in enumerate(reversed(range(bits - 1))):
                for ti in range(layer.input_tiling):
                    for to in range(layer.output_tiling):
                        idx = layer.get_memristor_index(bi, ti, to)
                        pair = layer.memristors[idx]
                        signed = integers[ti * 128:(ti + 1) * 128, to * 128:(to + 1) * 128]
                        expected = ((signed.abs().to(torch.int32) >> bit) & 1) * signed.sign()
                        assert torch.equal(pair.pos.resistance_weighted_poly_low, pair.neg.resistance_weighted_poly_low)
                        assert torch.equal(pair.pos.resistance_weighted_poly_high, pair.neg.resistance_weighted_poly_high)
                        if mode == "zero_bit_leakage_removed":
                            avg = (original[idx][0] + original[idx][1]) / 2
                            pair.pos.r.copy_(torch.where(expected == 0, avg, original[idx][0]))
                            pair.neg.r.copy_(torch.where(expected == 0, avg, original[idx][1]))
                        else:
                            pair.pos.r.copy_(1 - expected.clamp(0, 1))
                            pair.neg.r.copy_(1 + expected.clamp(-1, 0))
        y = layer(x)
        rmse = float((y - reference).square().mean().sqrt())
        post = torch.tanh(quant_activation(y, "prediction_net.8"))
        post_ref = torch.tanh(quant_activation(reference, "prediction_net.8"))
        records.append({"mode": mode, "pre_quant_rmse": rmse,
                        "relative_rmse": rmse / float(reference.std(unbiased=False)),
                        "post_tanh_rmse": float((post - post_ref).square().mean().sqrt())})
    return {"history_ids": ids.tolist(), "read_noise": False,
            "input": "same digital first-prediction-layer outputs for all modes", "modes": records}


def audit(args):
    sys.path[:0] = [str(Path(__file__).resolve().parents[1]), args.sisyphus_root, args.synaptogen_root]
    from recipe.experiments.librispeech.training.qat_ffnn_transducer_bpe import get_model_config
    from recipe.model_pipelines.common import memristor_layers, memristor_layers_mem_inited
    from recipe.model_pipelines.qat_ffnn_transducer.mem_inited.pytorch_modules import (
        QATFFNNTransducerEncoder,
        QATFFNNTransducerRecogConfig,
        QATFFNNTransducerScorer,
    )
    from synaptogen_ml.memristor_modules import DacAdcHardwareSettings
    import synaptogen_ml

    torch.set_num_threads(2)
    digital = torch.load(args.digital, map_location="cpu", mmap=True, weights_only=False)["model"]
    converted = torch.load(args.converted, map_location="cpu", mmap=True, weights_only=False)["model"]
    cfg = get_model_config(args.bits, 8, 0.0, None).with_replaced(
        converter_hardware_settings=DacAdcHardwareSettings(8, 4, 4, 0.6, 8020.0),
        pos_enc_converter_hardware_settings=DacAdcHardwareSettings(8, 1, 7, 0.6, 8020.0),
        num_cycles=1,
    )
    rcfg = QATFFNNTransducerRecogConfig(
        **{f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}, ilm_scale=0.0, blank_penalty=0.0
    )

    def cpu_bounds(original):
        def wrapped(*a, **kw):
            with torch.device("cpu"):
                return original(*a, **kw)
        return wrapped

    with ExitStack() as stack:
        for module in (memristor_layers, memristor_layers_mem_inited):
            name = "get_quantization_range_from_bit_precision"
            stack.enter_context(patch.object(module, name, cpu_bounds(getattr(module, name))))
        with torch.device("meta"):
            encoder = QATFFNNTransducerEncoder(cfg)
            scorer = QATFFNNTransducerScorer(rcfg)

    result = {"bits": args.bits, "digital": args.digital, "converted": args.converted,
              "synaptogen_file": synaptogen_ml.__file__, "state_schema": {}, "layers": []}
    for name, model in (("encoder", encoder), ("scorer", scorer)):
        expected = model.state_dict()
        result["state_schema"][name] = {
            "required_keys": len(expected),
            "missing": sorted(set(expected) - set(converted)),
            "shape_mismatches": [k for k, v in expected.items()
                                 if k in converted and v.shape != converted[k].shape
                                 and not (v.ndim == 0 and converted[k].shape == (1,))],
            # nn.Module._load_from_state_dict accepts legacy length-one tensors
            # for scalar parameters, so these are not loading failures.
            "legacy_scalar_shapes": sum(v.ndim == 0 and converted[k].shape == (1,)
                                        for k, v in expected.items() if k in converted),
        }
    # All untouched values should survive conversion exactly, including duplicated
    # observer registrations and frontend/layer-norm parameters.
    common = set(digital) & set(converted)
    result["preserved_state"] = {
        "common_keys": len(common),
        "changed": sorted(k for k in common if not torch.equal(digital[k], converted[k])),
    }

    for name in ("token_embedding", "prediction_net.2", "prediction_net.7", "joint_net.2", "joint_net.7"):
        p = name + "."
        weight = digital[p + "weight"]
        qmax = 2 ** (args.bits - 1) - 1
        obs = p + "weight_quantizer.observer."
        bound = torch.maximum(digital[obs + "min_val"].abs(), digital[obs + "max_val"].abs())
        original_scale = float(torch.maximum(bound / qmax, digital[obs + "eps"]))
        scale = float(converted[p + "output_factor"])
        if name != "token_embedding":
            scale *= float(converted[p + "input_factor"])
        original_qw = torch.fake_quantize_per_tensor_affine(weight, original_scale, 0, -qmax, qmax)
        qw = torch.fake_quantize_per_tensor_affine(weight, scale, 0, -qmax, qmax)
        integers = torch.round(qw / scale)
        if name != "token_embedding":
            integers = integers.T
        ni, no = integers.shape
        tile_i, tile_o = converted[p + "memristors.0.pos.r"].shape
        nti, nto = math.ceil(ni / tile_i), math.ceil(no / tile_o)
        effective = torch.zeros_like(integers)
        empty_plane_leakage = torch.zeros_like(integers)
        planes = []
        for bi, bit in enumerate(reversed(range(args.bits - 1))):
            count = on_count = 0
            sse = zero_sse = 0.0
            for ti in range(nti):
                for to in range(nto):
                    idx = bi * nti * nto + ti * nto + to
                    mp = p + f"memristors.{idx}."
                    nr, nc = min(tile_i, ni - ti * tile_i), min(tile_o, no - to * tile_o)
                    region = (slice(ti * tile_i, ti * tile_i + nr), slice(to * tile_o, to * tile_o + nc))
                    signed = integers[region]
                    expected = ((signed.abs().to(torch.int32) >> bit) & 1) * signed.sign()
                    actual = (converted[mp + "neg.r"] - converted[mp + "pos.r"])[:nr, :nc]
                    effective[region] += actual * 2 ** bit
                    zero = expected == 0
                    empty_plane_leakage[region] += actual * zero * 2 ** bit
                    sse += float((actual - expected).square().sum())
                    zero_sse += float(actual[zero].square().sum())
                    count += expected.numel()
                    on_count += int((~zero).sum())
            planes.append({"bit": bit, "active_fraction": on_count / count,
                           "signed_plane_rmse": math.sqrt(sse / count),
                           "zero_bit_leakage_rmse": math.sqrt(zero_sse / (count - on_count))
                           if count > on_count else None})
        rms = float(qw.square().mean().sqrt())
        row = {
            "name": name, "digital_scale": original_scale, "conversion_scale": scale,
            "current_weight_absmax": float(weight.abs().max()),
            "observed_weight_absmax": float(bound),
            "max_quantized_integer": int(integers.abs().max()),
            "regridding_weight_rmse": float((qw - original_qw).square().mean().sqrt()),
            "digital_nonzero_fraction": float((original_qw != 0).float().mean()),
            "converted_grid_nonzero_fraction": float((qw != 0).float().mean()),
            "quantized_weight_rms": rms,
            "programmed_weight_rmse": float((effective - integers).square().mean().sqrt()) * scale,
            "zero_bit_weight_leakage_rms": float(empty_plane_leakage.square().mean().sqrt()) * scale,
            "bias_unchanged": torch.equal(digital[p + "bias"], converted[p + "bias"])
            if p + "bias" in digital else None,
            "planes": planes,
        }
        if name != "token_embedding":
            aq = name.rsplit(".", 1)[0] + "." + str(int(name.rsplit(".", 1)[1]) - 1) + ".observer."
            act_bound = float(torch.maximum(digital[aq + "min_val"].abs(), digital[aq + "max_val"].abs()))
            row["expected_input_factor"] = 1 / act_bound
            row["saved_input_factor"] = float(converted[p + "input_factor"])
        result["layers"].append(row)
    if args.probe_predictor:
        result["matched_prediction_linear2"] = probe_second_prediction_linear(digital, converted, args.bits)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digital", required=True)
    parser.add_argument("--converted", required=True)
    parser.add_argument("--bits", type=int, required=True)
    parser.add_argument("--synaptogen-root", required=True)
    parser.add_argument("--sisyphus-root", default="/work/smt4/azim.javed/repositories/sisyphus_repo")
    parser.add_argument("--output", required=True)
    parser.add_argument("--probe-predictor", action="store_true")
    args = parser.parse_args()
    result = audit(args)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print("State schema:", result["state_schema"])
    print("Preserved state:", result["preserved_state"])
    for row in result["layers"]:
        print(row["name"], "nonzero", round(row["digital_nonzero_fraction"], 6),
              "weight RMS", round(row["quantized_weight_rms"], 6),
              "programming RMSE", round(row["programmed_weight_rmse"], 6),
              "zero-bit leakage RMS", round(row["zero_bit_weight_leakage_rms"], 6),
              "observed/current absmax", round(row["observed_weight_absmax"], 6),
              round(row["current_weight_absmax"], 6))
    print("Full results:", args.output)
    if "matched_prediction_linear2" in result:
        print("Matched predictor control:", result["matched_prediction_linear2"])


if __name__ == "__main__":
    main()
