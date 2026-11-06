"""Audit saved differential bitplanes without constructing full device models.

No-read-noise forward uses the exact polynomial I-V law and tile ADC operation,
factored into small matrix multiplications. Inputs are synthetic, not ASR data.
"""

import argparse
import json
import math
import time
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def polynomial(coefficients, x):
    y = torch.zeros_like(x)
    for c in reversed(coefficients):
        y = y * x + c
    return y


@torch.no_grad()
def audit(digital, converted, name, bits):
    prefix = name + "."
    embedding = name == "token_embedding"
    weight = digital[prefix + "weight"]
    obs = prefix + "weight_quantizer.observer."
    qmax = 2 ** (bits - 1) - 1
    observer_scale = torch.maximum(torch.maximum(digital[obs + "min_val"].abs(),
                                                digital[obs + "max_val"].abs()) / qmax,
                                   digital[obs + "eps"]).item()
    # These saved factors are the scale actually used in the conversion.
    scale = converted[prefix + "output_factor"].item()
    if not embedding:
        scale *= converted[prefix + "input_factor"].item()
    qw = torch.fake_quantize_per_tensor_affine(weight, scale, 0, -qmax, qmax)
    checkpoint_qw = torch.fake_quantize_per_tensor_affine(weight, observer_scale, 0, -qmax, qmax)
    intended = (qw / scale).round()
    if not embedding:
        intended = intended.T
    ni, no = intended.shape
    shape = converted[prefix + "memristors.0.pos.r"].shape
    tile_i, tile_o = shape
    nti, nto = math.ceil(ni / tile_i), math.ceil(no / tile_o)
    actual_int = torch.zeros_like(intended)
    ideal_int = torch.zeros_like(intended)
    on_sse = off_sse = 0.0
    on_count = off_count = 0
    planes = []
    torch.manual_seed(17)
    input_factor = converted[prefix + "input_factor"].item()
    if embedding:
        x = torch.zeros(8, ni)
        x[torch.arange(8), torch.linspace(0, max(0, ni - 2), 8).long()] = 1
    elif name == "joint_net.7":
        x = torch.tanh(torch.randn(8, ni))
    else:
        x = torch.randn(8, ni).clamp(-3, 3) / (6 * input_factor)
    voltage = F.pad(torch.round((x * input_factor).clamp(-1, 1) * 128) / 128 * 0.6,
                    (0, nti * tile_i - ni))
    reference_x = voltage[:, :ni] / (0.6 * input_factor)
    reference = reference_x @ (qw if embedding else qw.T)
    bias = converted.get(prefix + "bias")
    if bias is not None:
        reference += bias
    outputs = {"adc_4_4": torch.zeros(8, nto * tile_o),
               "adc_4_8": torch.zeros(8, nto * tile_o),
               "ideal_programming_adc_4_8": torch.zeros(8, nto * tile_o)}
    for bi, bit in enumerate(reversed(range(bits - 1))):
        sse = zero_sse = 0.0
        count = zero_count = 0
        for ti in range(nti):
            for to in range(nto):
                idx = bi * nti * nto + ti * nto + to
                mp = prefix + f"memristors.{idx}."
                actual = converted[mp + "neg.r"] - converted[mp + "pos.r"]
                nr, nc = min(tile_i, ni - ti * tile_i), min(tile_o, no - to * tile_o)
                region = (slice(ti * tile_i, ti * tile_i + nr), slice(to * tile_o, to * tile_o + nc))
                signed = intended[region]
                expected = ((signed.abs().to(torch.int32) >> bit) & 1) * signed.sign()
                delta = actual[:nr, :nc] - expected
                actual_int[region] += actual[:nr, :nc] * 2**bit
                ideal_int[region] += expected * 2**bit
                sse += delta.square().sum().item()
                count += delta.numel()
                zero = expected == 0
                zero_sse += delta[zero].square().sum().item()
                zero_count += int(zero.sum())
                on_sse += delta[~zero].square().sum().item()
                on_count += int((~zero).sum())
                off_sse += delta[zero].square().sum().item()
                off_count += int(zero.sum())
                v = voltage[:, ti * tile_i:(ti + 1) * tile_i]
                low = converted[mp + "pos.resistance_weighted_poly_low"]
                high = converted[mp + "pos.resistance_weighted_poly_high"]
                # This factorization requires identical positive/negative I-V polynomials.
                assert torch.equal(low, converted[mp + "neg.resistance_weighted_poly_low"])
                assert torch.equal(high, converted[mp + "neg.resistance_weighted_poly_high"])
                iv = polynomial(low, v) - polynomial(high, v)
                current = iv @ actual * 8020.0
                ideal_plane = torch.zeros_like(actual)
                ideal_plane[:nr, :nc] = expected
                ideal_current = iv @ ideal_plane * 8020.0
                out_slice = slice(to * tile_o, (to + 1) * tile_o)
                outputs["adc_4_4"][:, out_slice] += current.clamp(-8, 8).mul(16).round().div(16) * 2**bit
                outputs["adc_4_8"][:, out_slice] += current.clamp(-128, 128).mul(16).round().div(16) * 2**bit
                outputs["ideal_programming_adc_4_8"][:, out_slice] += ideal_current.clamp(-128, 128).mul(16).round().div(16) * 2**bit
        planes.append({"bit": bit, "rmse": math.sqrt(sse / count),
                       "zero_bit_rmse": math.sqrt(zero_sse / zero_count) if zero_count else None})
    assert torch.equal(ideal_int, intended), f"Ideal bitplane reconstruction failed: {name}"
    weight_rmse = (actual_int - intended).square().mean().sqrt().item() * scale
    weight_rms = qw.square().mean().sqrt().item()
    weight_gain = (actual_int * intended).sum().item() / intended.square().sum().item()
    residual_weight_rmse = (actual_int - weight_gain * intended).square().mean().sqrt().item() * scale
    forward = {}
    for mode, output in outputs.items():
        y = output[:, :no] * converted[prefix + "output_factor"]
        if bias is not None:
            y += bias
        rmse = (y - reference).square().mean().sqrt().item()
        reference_signal = reference if bias is None else reference - bias
        output_signal = y if bias is None else y - bias
        gain = (output_signal * reference_signal).sum().item() / reference_signal.square().sum().item()
        residual = (output_signal - gain * reference_signal).square().mean().sqrt().item()
        forward[mode] = {"rmse": rmse, "relative_to_reference_std": rmse / reference.std().item(),
                         "best_fit_signal_gain": gain, "gain_removed_rmse": residual,
                         "gain_removed_relative_rmse": residual / reference_signal.square().mean().sqrt().item()}
    return {"name": name, "weight_shape": list(weight.shape), "weight_scale": scale,
            "observer_scale": observer_scale, "scale_relative_difference": abs(scale - observer_scale) / observer_scale,
            "current_weight_absmax": weight.abs().max().item(),
            "conversion_regrid_rmse": (qw - checkpoint_qw).square().mean().sqrt().item(),
            "conversion_regrid_relative_rmse": (qw - checkpoint_qw).square().mean().sqrt().item() / checkpoint_qw.square().mean().sqrt().item(),
            "nonzero_fraction": (qw != 0).float().mean().item(), "weight_rms": weight_rms,
            "programming_weight_rmse": weight_rmse, "relative_weight_rmse": weight_rmse / weight_rms,
            "best_fit_weight_gain": weight_gain, "gain_removed_weight_rmse": residual_weight_rmse,
            "gain_removed_relative_weight_rmse": residual_weight_rmse / weight_rms,
            "zero_bit_rmse": math.sqrt(off_sse / off_count) if off_count else None,
            "on_bit_rmse": math.sqrt(on_sse / on_count) if on_count else None,
            "ideal_integer_reconstruction_max_error": (ideal_int - intended).abs().max().item(),
            "planes": planes, "synthetic_no_read_noise_forward": forward}


def fresh_conversion(digital, name, bits):
    """Execute the library's real programming routine for one selected layer."""
    import numpy as np
    from synaptogen_ml.memristor_modules import TiledMemristorLinear, TiledMemristorEmbedding, DacAdcHardwareSettings

    prefix = name + "."
    obs = prefix + "weight_quantizer.observer."
    qmax = 2 ** (bits - 1) - 1
    scale = torch.maximum(torch.maximum(digital[obs + "min_val"].abs(), digital[obs + "max_val"].abs()) / qmax,
                          digital[obs + "eps"])

    class FrozenWeightQuantizer:
        def __init__(self):
            self.scale = scale

        def __call__(self, weight):
            return torch.fake_quantize_per_tensor_affine(weight, float(self.scale), 0, -qmax, qmax)

    weight = digital[prefix + "weight"]
    proxy = SimpleNamespace(weight=weight, weight_quantizer=FrozenWeightQuantizer(),
                            bias=torch.nn.Parameter(digital[prefix + "bias"], requires_grad=False) if prefix + "bias" in digital else None)
    hs = DacAdcHardwareSettings(8, 4, 4, 0.6, 8020.0)
    np.random.seed(17)
    if name == "token_embedding":
        layer = TiledMemristorEmbedding(*weight.shape, bits, hs, 128, 128, padding_idx=weight.shape[0] - 1)
        layer.init_from_embedding_quant(proxy, num_cycles_init=1, correction_settings=None)
    else:
        parent, leaf = name.rsplit(".", 1)
        input_name = {"linear_ff": "lin_1_in_quant", "linear_out": "lin_2_in_quant",
                      "qkv_proj": "in_proj_in_quant", "2": "1", "7": "6"}[leaf]
        aq = parent + "." + input_name + ".observer."
        activation_scale = torch.maximum(torch.maximum(digital[aq + "min_val"].abs(), digital[aq + "max_val"].abs()) / 127,
                                         digital[aq + "eps"])
        layer = TiledMemristorLinear(weight.shape[1], weight.shape[0], bits, hs, 128, 128, bias=proxy.bias is not None)
        layer.init_from_linear_quant(SimpleNamespace(scale=activation_scale, quant_max=127), proxy,
                                     num_cycles_init=1, correction_settings=None)
    return {prefix + key: value for key, value in layer.state_dict().items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alias", required=True, type=Path)
    parser.add_argument("--bits", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--all-encoder-layers", action="store_true")
    parser.add_argument("--digital", type=Path, help="Override digital path from alias info")
    parser.add_argument("--fresh-from-digital", action="store_true", help="Fresh seeded selected-layer programming; not historical device states")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.digital:
        digital_path = str(args.digital)
    else:
        info = (args.alias / "info").read_text()
        digital_path = next(line[len("INPUT: "):] for line in info.splitlines() if line.startswith("INPUT: "))
    converted_path = args.alias / "output/memristor_converted_model.pt"
    print(f"Loading {args.alias.name}", flush=True)
    start = time.monotonic()
    digital = torch.load(digital_path, map_location="cpu", weights_only=False, mmap=True)["model"]
    print(f"Digital metadata loaded after {time.monotonic() - start:.1f}s", flush=True)
    if args.fresh_from_digital:
        sys.path.insert(0, "/work/smt4/azim.javed/sisyphus_work/i6_core/tools/git/CloneGitRepositoryJob.pxBD5qpvkBp6/output/synaptogen_ml")
        converted = {}
    else:
        converted = torch.load(converted_path, map_location="cpu", weights_only=False, mmap=True)["model"]
    print(f"Converted metadata loaded after {time.monotonic() - start:.1f}s", flush=True)
    names = []
    for key, value in digital.items():
        if not key.endswith(".weight") or value.ndim != 2:
            continue
        name = key[:-len(".weight")]
        if args.fresh_from_digital:
            if name not in ("token_embedding", "prediction_net.2", "prediction_net.7",
                            "conformer.module_list.0.module_list.0.linear_ff",
                            "conformer.module_list.0.module_list.2.mhsa.qkv_proj"):
                continue
        elif name + ".memristors.0.pos.r" not in converted:
            continue
        if name.startswith("conformer.") and not args.all_encoder_layers:
            if name.split(".")[2] not in ("0", "11"):
                continue
        names.append(name)
    result = {"alias": str(args.alias.resolve()), "digital": digital_path,
              "converted": None if args.fresh_from_digital else str(converted_path.resolve()),
              "fresh_seeded_programming": args.fresh_from_digital, "bits": args.bits, "layers": []}
    for name in names:
        if args.fresh_from_digital:
            print(f"Fresh programming {name}", flush=True)
            converted = fresh_conversion(digital, name, args.bits)
        row = audit(digital, converted, name, args.bits)
        result["layers"].append(row)
        print(f"{name}: weight RMSE={row['programming_weight_rmse']:.6g}, "
              f"relative={100 * row['relative_weight_rmse']:.2f}%, "
              f"nonzero={100 * row['nonzero_fraction']:.2f}%", flush=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {len(names)} layer audits to {args.output} in {time.monotonic() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
