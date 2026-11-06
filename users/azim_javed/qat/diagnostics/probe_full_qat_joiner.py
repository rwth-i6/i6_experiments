"""Matched-input joiner controls on a short real-audio excerpt, without decoding.

Factorizes the saved polynomial I-V law into matrix multiplications with read
noise disabled. ADC conversion remains per input tile and signed bitplane.
Validates the factorized result against the actual recognition-clone module.
"""

import argparse
import io
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def polynomial(coefficients, x):
    y = torch.zeros_like(x)
    for c in reversed(coefficients):
        y = y * x + c
    return y


class JoinerProbe:
    def __init__(self, digital, converted, bits):
        self.digital, self.converted, self.bits = digital, converted, bits
        self.planes = {}

    def weight(self, name):
        p = name + "."
        obs = p + "weight_quantizer.observer."
        qmax = 2 ** (self.bits - 1) - 1
        bound = torch.maximum(self.digital[obs + "min_val"].abs(), self.digital[obs + "max_val"].abs())
        scale = float(torch.maximum(bound / qmax, self.digital[obs + "eps"]))
        return torch.fake_quantize_per_tensor_affine(self.digital[p + "weight"], scale, 0, -qmax, qmax)

    def activation(self, x, index):
        obs = f"joint_net.{index}.observer."
        bound = torch.maximum(self.digital[obs + "min_val"].abs(), self.digital[obs + "max_val"].abs())
        scale = float(torch.maximum(bound / 127, self.digital[obs + "eps"]))
        return torch.fake_quantize_per_tensor_affine(x, scale, 0, -127, 127)

    def digital_linear(self, x, index):
        return F.linear(self.activation(x, index - 1), self.weight(f"joint_net.{index}"),
                        self.digital[f"joint_net.{index}.bias"])

    def prepare(self, index):
        if index in self.planes:
            return self.planes[index]
        p = f"joint_net.{index}."
        state = self.converted
        weight = self.weight(f"joint_net.{index}")
        no, ni = weight.shape
        tile_i, tile_o = state[p + "memristors.0.pos.r"].shape
        nti, nto = math.ceil(ni / tile_i), math.ceil(no / tile_o)
        scale = float(state[p + "input_factor"]) * float(state[p + "output_factor"])
        integers = F.pad((weight / scale).round().T, (0, nto * tile_o - no, 0, nti * tile_i - ni))
        planes = []
        for bi, bit in enumerate(reversed(range(self.bits - 1))):
            for ti in range(nti):
                for to in range(nto):
                    mp = p + f"memristors.{bi * nti * nto + ti * nto + to}."
                    low, high = state[mp + "pos.resistance_weighted_poly_low"], state[mp + "pos.resistance_weighted_poly_high"]
                    assert torch.equal(low, state[mp + "neg.resistance_weighted_poly_low"])
                    assert torch.equal(high, state[mp + "neg.resistance_weighted_poly_high"])
                    signed = integers[ti * tile_i:(ti + 1) * tile_i, to * tile_o:(to + 1) * tile_o]
                    expected = ((signed.abs().to(torch.int32) >> bit) & 1) * signed.sign()
                    actual = state[mp + "neg.r"] - state[mp + "pos.r"]
                    planes.append((bit, ti, to, low, high, expected, actual))
        self.planes[index] = (ni, no, tile_i, tile_o, nti, nto, planes)
        return self.planes[index]

    def analog_linear(self, x, index, mode="saved", adc_range=4, adc_precision=4,
                      current_gain=1.0, linear_iv=False):
        ni, no, tile_i, tile_o, nti, nto, planes = self.prepare(index)
        p = f"joint_net.{index}."
        # Recognition clone DAC: round at 128 levels, then apply 0.6 V.
        voltage = (x * self.converted[p + "input_factor"]).clamp(-1, 1).mul(128).round().div(128).mul(0.6)
        voltage = F.pad(voltage, (0, nti * tile_i - ni))
        out = torch.zeros(x.shape[0], nto * tile_o)
        clipped, count = 0, 0
        for bit, ti, to, low, high, expected, actual in planes:
            v = voltage[:, ti * tile_i:(ti + 1) * tile_i]
            if mode == "zero_bit_leakage_removed":
                states = torch.where(expected == 0, 0.0, actual)
            elif mode == "ideal":
                states = expected
            else:
                assert mode == "saved"
                states = actual
            if linear_iv:
                current = (v / 0.6) @ states
            else:
                current = (polynomial(low, v) - polynomial(high, v)) @ states * 8020.0 / current_gain
            # The clone rounds first and then clamps, at 1/16 resolution.
            adc_scale = 2 ** adc_precision if adc_precision is not None else 1
            adc = current.mul(adc_scale).round() if adc_precision is not None else current
            if adc_range is not None:
                limit = 2 ** (adc_range - 1) * adc_scale
                clipped += int((adc.abs() > limit).sum())
                adc = adc.clamp(-limit, limit)
            count += adc.numel()
            out[:, to * tile_o:(to + 1) * tile_o] += adc.div(adc_scale) * 2 ** bit
        out = out[:, :no] * self.converted[p + "output_factor"] + self.converted[p + "bias"]
        return out, {"clipped_partial_fraction": clipped / count, "partial_values": count}


def metrics(output, reference):
    rmse = float((output - reference).square().mean().sqrt())
    return {"rmse": rmse, "relative_rmse": rmse / float(reference.std(unbiased=False))}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digital", required=True)
    parser.add_argument("--converted", required=True)
    parser.add_argument("--bits", required=True, type=int)
    parser.add_argument("--synaptogen-root", required=True)
    parser.add_argument("--audio-zip", required=True)
    parser.add_argument("--audio-seconds", type=float, default=3.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--inputs", help="Optional previously saved matched input tensors")
    parser.add_argument("--predictor-adc-range", type=int, default=4)
    args = parser.parse_args()
    sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
                   "/work/smt4/azim.javed/repositories/sisyphus_repo", args.synaptogen_root]
    torch.set_num_threads(2)
    print("Loading digital and converted checkpoints", flush=True)
    digital = torch.load(args.digital, map_location="cpu", mmap=True, weights_only=False)["model"]
    converted = torch.load(args.converted, map_location="cpu", mmap=True, weights_only=False)["model"]
    if args.inputs:
        sample = torch.load(args.inputs, map_location="cpu", weights_only=False)
        assert sample["digital_checkpoint"] == str(Path(args.digital).resolve())
        joint_input = sample["joint_input"]
        audio_description = sample["audio"]
    else:
        import soundfile as sf
        from recipe.experiments.librispeech.training.qat_ffnn_transducer_bpe import get_model_config
        from recipe.model_pipelines.qat_ffnn_transducer.pytorch_modules import QATFFNNTransducerModel

        with zipfile.ZipFile(args.audio_zip) as archive:
            member = next(n for n in archive.namelist() if n.endswith(".ogg"))
            audio, sr = sf.read(io.BytesIO(archive.read(member)), dtype="float32")
        assert sr == 16000 and audio.ndim == 1
        # Match recognition peak normalization and preemphasis; use only a short
        # excerpt, so this is a layer diagnostic, not an utterance WER test.
        audio /= max(float(np.max(np.abs(audio))), 1e-20)
        audio[1:] -= 0.97 * audio[:-1].copy()
        audio = audio[:int(args.audio_seconds * sr)]
        model = QATFFNNTransducerModel(get_model_config(args.bits, 8, 0.0, None)).eval()
        model.load_state_dict(digital, strict=True)
        print("Computing matched digital encoder and prediction inputs", flush=True)
        enc, lengths = model.forward_encoder(torch.from_numpy(audio).view(1, -1, 1), torch.tensor([len(audio)]))
        frames = torch.linspace(0, int(lengths[0]) - 1, 16).long()
        enc = enc[0, frames]
        histories = torch.linspace(0, model.target_size - 1, 16).long()
        embedding = model.token_embedding(histories.view(-1, 1)).reshape(16, -1)
        pred = model.prediction_net(embedding)
        joint_input = torch.cat([enc[:, None].expand(-1, 16, -1), pred[None].expand(16, -1, -1)], dim=-1).reshape(256, -1)
        audio_description = {"zip": args.audio_zip, "member": member, "seconds": len(audio) / sr,
                             "encoder_frames": frames.tolist(), "history_ids": histories.tolist()}
        sample = {"digital_checkpoint": str(Path(args.digital).resolve()), "joint_input": joint_input,
                  "audio": audio_description}
        torch.save(sample, str(Path(args.output).with_suffix(".inputs.pt")))
        del model
    probe = JoinerProbe(digital, converted, args.bits)
    first = probe.digital_linear(joint_input, 2)
    hidden = torch.tanh(probe.activation(first, 3))
    second = probe.digital_linear(hidden, 7)
    reference_logits = probe.activation(second, 8)
    result = {"bits": args.bits, "digital": args.digital, "converted": args.converted,
              "audio": audio_description, "read_noise": False, "pairs": len(joint_input),
              "predictor_adc_range": args.predictor_adc_range, "layers": {}, "cascade": []}
    for index, x, reference in ((2, joint_input, first), (7, hidden, second)):
        records = []
        for mode in ("saved", "zero_bit_leakage_removed", "ideal"):
            for adc_range, adc_precision in ((4, 4), (8, 4), (None, 4), (8, 8), (8, None)):
                y, clipping = probe.analog_linear(x, index, mode, adc_range, adc_precision)
                row = {"mode": mode, "adc_range": adc_range, "adc_precision": adc_precision,
                       **metrics(y, reference), **clipping}
                post = probe.activation(y, index + 1)
                post_ref = probe.activation(reference, index + 1)
                if index == 2:
                    post, post_ref = torch.tanh(post), torch.tanh(post_ref)
                row["post_quant_rmse"] = metrics(post, post_ref)["rmse"]
                records.append(row)
        result["layers"][str(index)] = records

    for mode in ("saved", "zero_bit_leakage_removed", "ideal"):
        for adc_range, adc_precision in ((4, 4), (8, 4), (8, 8), (8, None)):
            y1, _ = probe.analog_linear(joint_input, 2, mode, adc_range, adc_precision)
            h = torch.tanh(probe.activation(y1, 3))
            y2, _ = probe.analog_linear(h, 7, mode, adc_range, adc_precision)
            logits = probe.activation(y2, 8)
            margin = logits[:, -1] - logits[:, :-1].max(dim=-1).values
            ref_margin = reference_logits[:, -1] - reference_logits[:, :-1].max(dim=-1).values
            result["cascade"].append({"mode": mode, "adc_range": adc_range, "adc_precision": adc_precision,
                                      **metrics(logits, reference_logits),
                                      "argmax_agreement": float((logits.argmax(-1) == reference_logits.argmax(-1)).float().mean()),
                                      "blank_margin_rmse": float((margin - ref_margin).square().mean().sqrt())})

    # Independent check against the actual recognition module, at the same inputs.
    import synaptogen_ml
    from synaptogen_ml.memristor_modules import TiledMemristorLinear, DacAdcHardwareSettings
    synaptogen_ml.set_fast_inference(False)
    result["implementation_checks"] = []
    for index, x in ((2, joint_input), (7, hidden)):
        no, ni = digital[f"joint_net.{index}.weight"].shape
        layer = TiledMemristorLinear(ni, no, args.bits, DacAdcHardwareSettings(8, 4, 4, 0.6, 8020.0), 128, 128)
        p = f"joint_net.{index}."
        layer.load_state_dict({k[len(p):]: v for k, v in converted.items() if k.startswith(p)}, strict=True)
        layer.initialized = True
        layer.eval()
        for pair in layer.memristors:
            pair.pos.BW = pair.neg.BW = 0.0
        actual = layer(x[:2])
        factorized, _ = probe.analog_linear(x[:2], index)
        error = float((actual - factorized).abs().max())
        result["implementation_checks"].append({"index": index, "max_abs_difference": error})
        torch.testing.assert_close(actual, factorized, atol=2e-4, rtol=2e-4)

    # Cross predictor/joiner offload while holding the acoustic states fixed.
    import dataclasses
    from recipe.experiments.librispeech.training.qat_ffnn_transducer_bpe import get_model_config
    from recipe.model_pipelines.qat_ffnn_transducer.mem_inited.pytorch_modules import (
        QATFFNNTransducerRecogConfig, QATFFNNTransducerScorer,
    )
    from synaptogen_ml.memristor_modules.memristor import MemristorArray

    cfg = get_model_config(args.bits, 8, 0.0, None).with_replaced(
        converter_hardware_settings=DacAdcHardwareSettings(8, 4, args.predictor_adc_range, 0.6, 8020.0),
        pos_enc_converter_hardware_settings=DacAdcHardwareSettings(8, 1, 7, 0.6, 8020.0), num_cycles=1,
    )
    scorer = QATFFNNTransducerScorer(QATFFNNTransducerRecogConfig(
        **{f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}, ilm_scale=0.0, blank_penalty=0.0,
    )).eval()
    required = scorer.state_dict().keys()
    scorer.load_state_dict({k: v for k, v in converted.items() if k in required}, strict=True)
    synaptogen_ml.set_fast_inference(False)
    for module in scorer.modules():
        if isinstance(module, MemristorArray):
            module.BW = 0.0

    ids = torch.tensor(audio_description["history_ids"])

    def predictor_activation(x, index):
        obs = f"prediction_net.{index}.observer."
        bound = torch.maximum(digital[obs + "min_val"].abs(), digital[obs + "max_val"].abs())
        scale = float(torch.maximum(bound / 127, digital[obs + "eps"]))
        return torch.fake_quantize_per_tensor_affine(x, scale, 0, -127, 127)

    qmax = 2 ** (args.bits - 1) - 1
    emb = torch.fake_quantize_per_tensor_affine(digital["token_embedding.weight"],
                                               float(converted["token_embedding.output_factor"]), 0, -qmax, qmax)[ids]
    for index in (2, 7):
        emb = F.linear(predictor_activation(emb, index - 1), probe.weight(f"prediction_net.{index}"),
                       digital[f"prediction_net.{index}.bias"])
        emb = torch.tanh(predictor_activation(emb, index + 1))
    predictors = {"digital": joint_input[:len(ids), 512:], "embedding_regrid_only": emb,
                  "analog": scorer.cached_prediction_net_forward(ids[:, None])}

    # Isolate the second-prediction-linear zero-bit effect within a full analog
    # predictor. Embedding and first linear stay in their saved programmed state.
    layer = scorer.prediction_net[7]
    p = "prediction_net.7."
    scale = float(converted[p + "input_factor"]) * float(converted[p + "output_factor"])
    integers = (probe.weight("prediction_net.7") / scale).round().T
    for bi, bit in enumerate(reversed(range(args.bits - 1))):
        for ti in range(layer.input_tiling):
            for to in range(layer.output_tiling):
                pair = layer.memristors[layer.get_memristor_index(bi, ti, to)]
                signed = integers[ti * 128:(ti + 1) * 128, to * 128:(to + 1) * 128]
                expected = ((signed.abs().to(torch.int32) >> bit) & 1) * signed.sign()
                avg = (pair.pos.r + pair.neg.r) / 2
                pair.pos.r.copy_(torch.where(expected == 0, avg, pair.pos.r))
                pair.neg.r.copy_(torch.where(expected == 0, avg, pair.neg.r))
    predictors["analog_prediction_linear2_zero_leakage"] = scorer.cached_prediction_net_forward(ids[:, None])

    result["predictor_outputs"] = {name: metrics(pred, predictors["digital"])
                                   for name, pred in predictors.items()}
    result["predictor_joiner_interaction"] = []
    for name, pred in predictors.items():
        inp = torch.cat([joint_input[:, :512], pred.repeat(len(joint_input) // len(pred), 1)], dim=-1)
        for joiner in ("digital", "analog_range4", "analog_range8"):
            if joiner == "digital":
                y1 = probe.digital_linear(inp, 2)
                y2 = probe.digital_linear(torch.tanh(probe.activation(y1, 3)), 7)
            else:
                adc_range = 4 if joiner == "analog_range4" else 8
                y1, _ = probe.analog_linear(inp, 2, adc_range=adc_range)
                y2, _ = probe.analog_linear(torch.tanh(probe.activation(y1, 3)), 7, adc_range=adc_range)
            logits = probe.activation(y2, 8)
            margin = logits[:, -1] - logits[:, :-1].max(dim=-1).values
            ref_margin = reference_logits[:, -1] - reference_logits[:, :-1].max(dim=-1).values
            result["predictor_joiner_interaction"].append({
                "predictor": name, "joiner": joiner, **metrics(logits, reference_logits),
                "argmax_agreement": float((logits.argmax(-1) == reference_logits.argmax(-1)).float().mean()),
                "blank_margin_rmse": float((margin - ref_margin).square().mean().sqrt()),
            })

    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    for index, records in result["layers"].items():
        print("Joiner linear", index)
        for row in records:
            print(row["mode"], "precision/range", row["adc_precision"], row["adc_range"],
                  "RMSE", round(row["rmse"], 6), "post-quant RMSE", round(row["post_quant_rmse"], 6),
                  "clipping", round(row["clipped_partial_fraction"], 6))
    print("Cascade:", result["cascade"])
    print("Implementation checks:", result["implementation_checks"])
    print("Predictor outputs:", result["predictor_outputs"])
    print("Predictor/joiner interaction:", result["predictor_joiner_interaction"])
    print("Full results:", args.output)


if __name__ == "__main__":
    main()
