"""Separate ideal resistance programming from I-V gain and digital equivalence.

Uses previously saved matched inputs. Calibration uses the analytic small-signal
I-V slope, not a fit to the test outputs. All controls disable read noise.
"""

import argparse
import json
from pathlib import Path

import torch

from probe_full_qat_joiner import JoinerProbe, metrics


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digital", required=True)
    parser.add_argument("--converted", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--bits", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print("Loading states and fixed inputs", flush=True)
    digital = torch.load(args.digital, map_location="cpu", mmap=True, weights_only=False)["model"]
    converted = torch.load(args.converted, map_location="cpu", mmap=True, weights_only=False)["model"]
    sample = torch.load(args.inputs, map_location="cpu", weights_only=False)
    assert sample["digital_checkpoint"] == str(Path(args.digital).resolve())
    probe = JoinerProbe(digital, converted, args.bits)
    inp = sample["joint_input"]
    ref1 = probe.digital_linear(inp, 2)
    h = torch.tanh(probe.activation(ref1, 3))
    ref2 = probe.digital_linear(h, 7)
    reference = probe.activation(ref2, 8)
    result = {"bits": args.bits, "read_noise": False, "audio": sample["audio"],
              "layers": {}, "cascade": []}
    gains = {}
    for index, x, ref in ((2, inp, ref1), (7, h, ref2)):
        planes = probe.prepare(index)[-1]
        low, high = planes[0][3:5]
        assert all(torch.equal(p[3], low) and torch.equal(p[4], high) for p in planes)
        gains[index] = float((low[1] - high[1]) * 0.6 * 8020.0)
        bias = converted[f"joint_net.{index}.bias"]
        signal = ref - bias
        rows = []
        for mode in ("saved", "ideal", "ideal_slope_calibrated", "ideal_linear_iv"):
            for precision in (4, 8, None):
                y, clipping = probe.analog_linear(
                    x, index, mode="saved" if mode == "saved" else "ideal", adc_range=8,
                    adc_precision=precision, current_gain=gains[index] if mode == "ideal_slope_calibrated" else 1.0,
                    linear_iv=mode == "ideal_linear_iv",
                )
                gain = float(((y - bias) * signal).sum() / signal.square().sum())
                rows.append({"mode": mode, "adc_precision": precision, **metrics(y, ref),
                             "bias_removed_signal_gain": gain, **clipping})
        result["layers"][str(index)] = {"analytic_small_signal_gain": gains[index], "controls": rows}

    for mode in ("saved", "ideal", "ideal_slope_calibrated", "ideal_linear_iv"):
        for precision in (4, 8, None):
            def forward(x, index):
                return probe.analog_linear(
                    x, index, mode="saved" if mode == "saved" else "ideal", adc_range=8,
                    adc_precision=precision, current_gain=gains[index] if mode == "ideal_slope_calibrated" else 1.0,
                    linear_iv=mode == "ideal_linear_iv",
                )[0]
            y1 = forward(inp, 2)
            y2 = forward(torch.tanh(probe.activation(y1, 3)), 7)
            logits = probe.activation(y2, 8)
            result["cascade"].append({"mode": mode, "adc_precision": precision,
                                      **metrics(logits, reference),
                                      "argmax_agreement": float((logits.argmax(-1) == reference.argmax(-1)).float().mean())})
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
