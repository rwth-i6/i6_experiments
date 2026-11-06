"""Small device-level test of the unchanged SynaptogenML correction routine.

Uses one signed magnitude bitplane with 10% active entries, three fixed seeds,
and the real TiledMemristorLinear initialization method. No decoding, altered
correction algorithm, or ideal resistance assignments are used.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def polynomial(coefficients, voltage):
    result = torch.zeros_like(voltage)
    for c in reversed(coefficients):
        result = result * voltage + c
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synaptogen-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--recognition-gain", type=float, default=5476.0)
    args = parser.parse_args()
    sys.path.insert(0, args.synaptogen_root)
    from synaptogen_ml import synaptogen
    from synaptogen_ml.memristor_modules import TiledMemristorLinear, DacAdcHardwareSettings
    from synaptogen_ml.memristor_modules.config import CycleCorrectionSettings

    torch.set_num_threads(2)
    target_rng = np.random.default_rng(123)
    targets = torch.from_numpy(target_rng.choice([-1.0, 0.0, 1.0], size=(64, 64), p=[0.05, 0.9, 0.05])).float()

    class FrozenQuantizer:
        scale = torch.tensor([1.0])

        def __call__(self, weights):
            return weights

    proxy = SimpleNamespace(weight=targets.T, bias=None, weight_quantizer=FrozenQuantizer())
    activation = SimpleNamespace(scale=torch.tensor([1 / 127]), quant_max=127)
    settings = [(0, None), (5, 0.15), (10, 0.15), (10, 0.10), (20, 0.10)]
    rows = []
    for verify_gain in (5476.0, 8020.0):
        for rounds, tolerance in settings:
            samples = []
            for seed in (17, 18, 19):
                np.random.seed(seed)
                # rand/randn partials keep a reference to this same Generator.
                synaptogen.rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state
                layer = TiledMemristorLinear(64, 64, 2,
                                             DacAdcHardwareSettings(8, 4, 8, 0.6, verify_gain), 64, 64, bias=False)
                correction = None if not rounds else CycleCorrectionSettings(
                    num_cycles=rounds, test_input_value=0.6, relative_deviation=tolerance, ideal_programming=False,
                )
                layer.init_from_linear_quant(activation, proxy, num_cycles_init=1, correction_settings=correction)
                pair = layer.memristors[0]

                def current(array):
                    v = torch.full_like(array.r, 0.6)
                    return (polynomial(array.resistance_weighted_poly_low, v) * (1 - array.r)
                            + polynomial(array.resistance_weighted_poly_high, v) * array.r)

                pos, neg = current(pair.pos), current(pair.neg)
                differential = (pos - neg) * args.recognition_gain
                error = differential - targets
                zero, on = targets == 0, targets != 0
                positive_target, negative_target = targets.clamp(0, 1), -targets.clamp(-1, 0)
                pos_error = (pos * verify_gain - positive_target).abs()
                neg_error = (neg * verify_gain - negative_target).abs()
                samples.append({
                    "seed": seed,
                    "differential_rmse": float(error.square().mean().sqrt()),
                    "zero_bit_rmse": float(error[zero].square().mean().sqrt()),
                    "active_bit_rmse": float(error[on].square().mean().sqrt()),
                    "active_bit_gain": float((differential[on] * targets[on]).mean()),
                    "individual_devices_within_10pct": float(torch.cat([pos_error.flatten(), neg_error.flatten()]).le(0.10).float().mean()),
                    "individual_devices_within_requested_tolerance":
                        float(torch.cat([pos_error.flatten(), neg_error.flatten()]).le(tolerance).float().mean())
                        if tolerance is not None else None,
                })
            summary = {key: float(np.mean([s[key] for s in samples])) for key in samples[0]
                       if key != "seed" and samples[0][key] is not None}
            row = {"verify_gain": verify_gain, "recognition_gain": args.recognition_gain, "rounds": rounds,
                   "tolerance": tolerance, "test_voltage": 0.6, "mean": summary, "samples": samples}
            rows.append(row)
            print("verify gain", verify_gain, "rounds", rounds, "tolerance", tolerance, summary, flush=True)
    result = {"description": __doc__, "cells_per_polarity": targets.numel(),
              "active_target_fraction": float((targets != 0).float().mean()), "rows": rows}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
