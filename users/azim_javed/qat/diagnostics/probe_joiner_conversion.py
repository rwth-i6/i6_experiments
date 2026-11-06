"""Matched-input joiner audit; uses existing checkpoints, no workflow submission.

Synthetic inputs isolate layer arithmetic; these measurements are not WER.
"""

import argparse
import json
import sys

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synaptogen-root", required=True)
    parser.add_argument("--digital", required=True)
    parser.add_argument("--converted", required=True)
    parser.add_argument("--bits", type=int, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--metadata-only", action="store_true", help="Skip layer forward computations")
    args = parser.parse_args()
    torch.set_num_threads(2)
    print("Importing the recognition job's SynaptogenML", flush=True)
    sys.path.insert(0, args.synaptogen_root)
    import synaptogen_ml
    from synaptogen_ml.memristor_modules import TiledMemristorLinear, DacAdcHardwareSettings
    from synaptogen_ml.memristor_modules.memristor import MemristorArray

    synaptogen_ml.set_fast_inference(False)
    print("Loading checkpoint metadata", flush=True)
    digital = torch.load(args.digital, map_location="cpu", weights_only=False, mmap=True)["model"]
    converted = torch.load(args.converted, map_location="cpu", weights_only=False, mmap=True)["model"]
    result = {"synaptogen_file": synaptogen_ml.__file__, "bits": args.bits, "layers": []}
    for index in (2, 7):
        print(f"Auditing joint_net.{index}", flush=True)
        prefix = f"joint_net.{index}."
        weight = digital[prefix + "weight"]
        out_features, in_features = weight.shape
        qmax = 2 ** (args.bits - 1) - 1
        obs = prefix + "weight_quantizer.observer."
        bound = torch.maximum(digital[obs + "min_val"].abs(), digital[obs + "max_val"].abs())
        scale = torch.maximum(bound / qmax, digital[obs + "eps"])
        qw = torch.fake_quantize_per_tensor_affine(weight, float(scale), 0, -qmax, qmax)
        layer = TiledMemristorLinear(in_features, out_features, args.bits,
                                    DacAdcHardwareSettings(8, 4, 8, 0.6, 8020.0), 128, 128)
        layer.load_state_dict({k[len(prefix):]: v for k, v in converted.items() if k.startswith(prefix)})
        layer.initialized = True
        layer.eval()
        layer.requires_grad_(False)
        layer.to(args.device)
        qw = qw.to(args.device)
        scale = scale.to(args.device)
        aq = f"joint_net.{index - 1}.observer."
        ab = torch.maximum(digital[aq + "min_val"].abs(), digital[aq + "max_val"].abs())
        expected_factor = float(scale * ab)
        torch.manual_seed(17)
        # Layer 2 follows tanh. Layer 1 uses an explicit fraction of its observed bound.
        x = torch.tanh(torch.randn(8, in_features, device=args.device)) if index == 7 else torch.randn(8, in_features, device=args.device).clamp(-3, 3) * float(ab) / 6
        dac_x = layer.converter.dac(x * layer.input_factor) / 0.6 / layer.input_factor
        reference = F.linear(dac_x, qw, layer.bias)
        records = []
        arrays = [m for m in layer.modules() if isinstance(m, MemristorArray)]
        for mode in ("current", "no_read_noise", "physical_charge_same_bw", "physical_charge_100MHz"):
            if args.metadata_only:
                break
            for m in arrays:
                m.e = 2.718281828459045 if mode == "current" else 1.602176634e-19
                m.BW = 0.0 if mode == "no_read_noise" else (1e8 if mode == "physical_charge_100MHz" else 1e-8)
            torch.manual_seed(19)
            with torch.no_grad():
                y = layer(x)
            error = y - reference
            records.append({"mode": mode, "rmse": float(error.square().mean().sqrt()),
                            "relative_rmse": float(error.square().mean().sqrt() / reference.std()),
                            "max_abs_error": float(error.abs().max()), "output_std": float(y.std())})
        # Audit bitplane mapping directly using resistance states, independent of input/noise.
        effective = torch.zeros(in_features, out_features, device=args.device)
        programming = []
        for bi, bit in enumerate(reversed(range(args.bits - 1))):
            plane_errors = []
            off_errors = []
            for ti in range(layer.input_tiling):
                for to in range(layer.output_tiling):
                    pair = layer.memristors[layer.get_memristor_index(bi, ti, to)]
                    nr = min(128, in_features - ti * 128)
                    nc = min(128, out_features - to * 128)
                    region = (slice(ti*128, ti*128+nr), slice(to*128, to*128+nc))
                    signed_int = (qw / scale.to(args.device)).round().T[region]
                    expected = ((signed_int.abs().to(torch.int32) >> bit) & 1) * signed_int.sign()
                    actual = pair.neg.r[:nr, :nc] - pair.pos.r[:nr, :nc]
                    effective[region] += actual * (2 ** bit)
                    plane_errors.append((actual - expected).flatten())
                    off_errors.append(actual[expected == 0].flatten())
            pe = torch.cat(plane_errors)
            oe = torch.cat(off_errors)
            programming.append({"bit": bit, "signed_plane_rmse": float(pe.square().mean().sqrt()),
                                "zero_bit_leakage_rmse": float(oe.square().mean().sqrt()),
                                "signed_plane_error_mean": float(pe.mean())})
        integers = (qw / scale).round().T
        if not args.metadata_only:
            # Replace only resistance states by exactly the intended signed bitplanes.
            # Preserve device I-V curves, DAC, ADC, scales, and biases.
            for bi, bit in enumerate(reversed(range(args.bits - 1))):
                for ti in range(layer.input_tiling):
                    for to in range(layer.output_tiling):
                        pair = layer.memristors[layer.get_memristor_index(bi, ti, to)]
                        nr = min(128, in_features - ti * 128)
                        nc = min(128, out_features - to * 128)
                        signed_int = integers[ti*128:ti*128+nr, to*128:to*128+nc]
                        expected = ((signed_int.abs().to(torch.int32) >> bit) & 1) * signed_int.sign()
                        pair.pos.r.fill_(1)
                        pair.neg.r.fill_(1)
                        pair.pos.r[:nr, :nc] = 1 - expected.clamp(0, 1)
                        pair.neg.r[:nr, :nc] = 1 + expected.clamp(-1, 0)
            for m in arrays:
                m.BW = 0.0
            with torch.no_grad():
                y = layer(x)
            error = y - reference
            records.append({"mode": "ideal_programming_no_noise", "rmse": float(error.square().mean().sqrt()),
                            "relative_rmse": float(error.square().mean().sqrt() / reference.std()),
                            "max_abs_error": float(error.abs().max()), "output_std": float(y.std())})
        result["layers"].append({"index": index, "shape": list(weight.shape), "weight_scale": float(scale),
                                  "activation_bound": float(ab), "input_factor": float(layer.input_factor),
                                  "output_factor": float(layer.output_factor), "expected_output_factor": expected_factor,
                                  "weight_nonzero_fraction": float((qw != 0).float().mean()),
                                  "programmed_integer_rmse": float((effective - integers).square().mean().sqrt()),
                                  "programmed_weight_rmse": float((effective - integers).square().mean().sqrt() * scale),
                                  "digital_weight_rms": float(qw.square().mean().sqrt()),
                                  "bitplanes": programming,
                                  "reference_std": float(reference.std()), "modes": records})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
