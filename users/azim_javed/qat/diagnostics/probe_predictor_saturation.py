"""Measure actual predictor converter/activation clipping for every history ID.

Uses an existing converted checkpoint and the recognition SynaptogenML clone.
Range controls replay the same read-noise seed, without submitting decode jobs.
"""

import argparse
import dataclasses
import json
import sys
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digital", required=True)
    parser.add_argument("--converted", required=True)
    parser.add_argument("--bits", required=True, type=int)
    parser.add_argument("--synaptogen-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
                   "/work/smt4/azim.javed/repositories/sisyphus_repo", args.synaptogen_root]
    from recipe.experiments.librispeech.training.qat_ffnn_transducer_bpe import get_model_config
    from recipe.model_pipelines.qat_ffnn_transducer.mem_inited.pytorch_modules import (
        QATFFNNTransducerRecogConfig, QATFFNNTransducerScorer,
    )
    from recipe.model_pipelines.common.memristor_layers import ActivationQuantizer
    from synaptogen_ml.memristor_modules.memristor import DacAdcPair, DacAdcHardwareSettings, MemristorArray
    import synaptogen_ml
    from probe_full_qat_joiner import JoinerProbe

    torch.set_num_threads(2)
    print("Loading checkpoint states", flush=True)
    digital = torch.load(args.digital, map_location="cpu", mmap=True, weights_only=False)["model"]
    converted = torch.load(args.converted, map_location="cpu", mmap=True, weights_only=False)["model"]
    cfg = get_model_config(args.bits, 8, 0.0, None).with_replaced(
        converter_hardware_settings=DacAdcHardwareSettings(8, 4, 4, 0.6, 8020.0),
        pos_enc_converter_hardware_settings=DacAdcHardwareSettings(8, 1, 7, 0.6, 8020.0), num_cycles=1,
    )
    scorer = QATFFNNTransducerScorer(QATFFNNTransducerRecogConfig(
        **{f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}, ilm_scale=0.0, blank_penalty=0.0,
    )).eval()
    required = scorer.state_dict().keys()
    scorer.load_state_dict({k: v for k, v in converted.items() if k in required}, strict=True)
    synaptogen_ml.set_fast_inference(False)
    converters = {"embedding": scorer.token_embedding.converter,
                  "prediction_linear1": scorer.prediction_net[2].converter,
                  "prediction_linear2": scorer.prediction_net[7].converter}
    converter_names = {id(c): name for name, c in converters.items()}
    settings = {name: c.hs for name, c in converters.items()}
    arrays = [(m, m.BW) for m in scorer.modules() if isinstance(m, MemristorArray)]
    probe = JoinerProbe(digital, converted, args.bits)
    ids = torch.arange(cfg.target_size).view(-1, 1)

    def qact(x, index):
        p = f"prediction_net.{index}.observer."
        bound = torch.maximum(digital[p + "min_val"].abs(), digital[p + "max_val"].abs())
        scale = float(torch.maximum(bound / 127, digital[p + "eps"]))
        return torch.fake_quantize_per_tensor_affine(x, scale, 0, -127, 127)

    reference = F.embedding(ids, probe.weight("token_embedding")).reshape(cfg.target_size, -1)
    for index in (2, 7):
        reference = F.linear(qact(reference, index - 1), probe.weight(f"prediction_net.{index}"),
                             digital[f"prediction_net.{index}.bias"])
        reference = torch.tanh(qact(reference, index + 1))

    stats = None

    def observe(key, values, scale, qmin, qmax):
        row = stats[key]
        codes = (values / scale).round()
        row["values"] += values.numel()
        row["clipped_values"] += int(((codes < qmin) | (codes > qmax)).sum())
        row["range_exceedances_before_rounding"] += int(((values < qmin * scale) | (values > qmax * scale)).sum())
        row["max_abs_input"] = max(row["max_abs_input"], float(values.abs().max()))
        row["lower_limit"], row["upper_limit"] = float(qmin * scale), float(qmax * scale)

    original_dac, original_adc = DacAdcPair.dac, DacAdcPair.adc

    def dac(converter, values):
        if id(converter) in converter_names:
            observe(converter_names[id(converter)] + ".dac", values,
                    converter.hs.dac_input_quant_scaling_factor, converter.hs.dac_min, converter.hs.dac_max)
        return original_dac(converter, values)

    def adc(converter, values):
        if id(converter) in converter_names:
            observe(converter_names[id(converter)] + ".adc", values * converter.hs.hardware_output_current_scaling,
                    1 / 2 ** converter.hs.output_precision_bits, converter.hs.adc_min, converter.hs.adc_max)
        return original_adc(converter, values)

    def hook(name):
        def pre(module, inputs):
            scale, _ = module.observer.calculate_qparams()
            observe(name, inputs[0], float(scale), int(module.quant_min), int(module.quant_max))
        return pre

    handles = [m.register_forward_pre_hook(hook(name)) for name, m in scorer.named_modules()
               if name.startswith("prediction_net.") and isinstance(m, ActivationQuantizer)]
    result = {"bits": args.bits, "digital": args.digital, "converted": args.converted,
              "histories": cfg.target_size, "seed": 17, "modes": []}
    baseline = {}
    try:
        with patch.object(DacAdcPair, "dac", dac), patch.object(DacAdcPair, "adc", adc):
            for read_noise in (False, True):
                for adc_range in (4, 8, 16):
                    stats = defaultdict(lambda: {"values": 0, "clipped_values": 0,
                                                 "range_exceedances_before_rounding": 0, "max_abs_input": 0.0})
                    for name, converter in converters.items():
                        converter.hs = dataclasses.replace(settings[name], output_range_bits=adc_range)
                    for module, bw in arrays:
                        module.BW = bw if read_noise else 0.0
                    torch.manual_seed(17)
                    pred = scorer.cached_prediction_net_forward(ids)
                    if adc_range == 4:
                        baseline[read_noise] = pred.clone()
                    rows = {}
                    for name, row in stats.items():
                        rows[name] = dict(row, clipped_fraction=row["clipped_values"] / row["values"])
                    rmse = float((pred - reference).square().mean().sqrt())
                    record = {"read_noise": read_noise, "adc_range": adc_range, "saturation": rows,
                              "prediction_rmse": rmse,
                              "prediction_relative_rmse": rmse / float(reference.std(unbiased=False)),
                              "max_output_change_vs_range4": float((pred - baseline[read_noise]).abs().max())}
                    result["modes"].append(record)
                    print("read_noise", read_noise, "range", adc_range, "RMSE", rmse,
                          "max change vs range4", record["max_output_change_vs_range4"], flush=True)
                    for name, row in rows.items():
                        print(name, "clipped", row["clipped_values"], "/", row["values"],
                              "max_abs", row["max_abs_input"], "limit", row["upper_limit"], flush=True)
    finally:
        for handle in handles:
            handle.remove()
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print("Full results:", args.output)


if __name__ == "__main__":
    main()
