"""Replay saved linear2 outputs through the recognition scorer's joint_net[8].

Run from training/qat with the base environment. --rasr-config is the actual
recognition job's rasr.config, so the scorer class and settings match that run.
The output retains the appended-record format used by plot_histograms.py.
"""

import argparse
import configparser
from pathlib import Path
import sys
from types import MethodType

import numpy as np
import torch

from plot_histograms import iter_records


def configure_import_paths():
    # The standalone process runs from a job output directory, without the
    # Sisyphus/RETURNN launcher that normally sets these import roots.
    for root in (
        Path("/work/smt4/azim.javed/repositories/sisyphus"),
        Path(__file__).resolve().parents[2],
    ):
        sys.path.insert(0, str(root))


def load_scorer(rasr_config_path):
    configure_import_paths()
    from recipe.model_pipelines.common.pyscorers import (
        _exec_model_imports,
        _get_recog_model_config,
        _parse_correction_settings,
        _parse_dac_adc_settings,
    )

    config = configparser.ConfigParser(interpolation=None)
    config.read(rasr_config_path)
    recognition = config["lib-rasr.label-scorer.recognition"]
    namespace = _exec_model_imports(recognition["imports"])
    qat = config["lib-rasr.label-scorer.recognition.qat"]
    if qat.get("weight-pruning-config", "None").lower() != "none":
        raise ValueError("This replay script expects weight_pruning_config=None")
    qat_params = dict(
        weight_bit_prec=int(qat["weight-bit-prec"]),
        activation_bit_prec=int(qat["activation-bit-prec"]),
        weight_dropout=float(qat["weight-dropout"]),
        weight_pruning_config=None,
    )
    memristor_params = {}
    section = "lib-rasr.label-scorer.recognition.memristor"
    if section in config:
        memristor = config[section]
        for key in ("converter-hardware-settings", "pos-enc-converter-hardware-settings"):
            if key in memristor:
                memristor_params[key.replace("-", "_")] = _parse_dac_adc_settings(memristor[key])
        if "num-cycles" in memristor:
            memristor_params["num_cycles"] = int(memristor["num-cycles"])
        if "correction-settings" in memristor:
            memristor_params["correction_settings"] = _parse_correction_settings(memristor["correction-settings"])
    cfg = _get_recog_model_config(
        namespace["RecogConfig"], namespace["get_model_config"], qat_params,
        float(recognition["ilm-scale"]), float(recognition["blank-penalty"]),
        memristor_params=memristor_params,
    )
    scorer = namespace["ScorerModel"](cfg=cfg)
    checkpoint = recognition["model-path"]
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    missing, _ = scorer.load_state_dict(data.get("model", data), strict=False)
    if missing:
        raise ValueError(f"Scorer checkpoint is missing parameters: {missing}")
    scorer.eval()

    def output_only_forward(self, inputs):
        return self.joint_net[8](inputs)

    scorer.forward = MethodType(output_only_forward, scorer)
    return scorer, checkpoint


def replay(scorer, input_path, output_path):
    records = batches = rows = 0
    # Each original batch is replayed in order, including all token channels.
    # This preserves first-call observer/scale initialization in eval mode.
    with output_path.open("xb") as stream, torch.inference_mode():
        for record in iter_records(input_path):
            outputs = []
            for batch in record["linear2"]:
                batch = np.asarray(batch)
                if batch.ndim != 2:
                    raise ValueError(f"Expected [batch, vocab] outputs, got {batch.shape}")
                output = scorer(torch.from_numpy(batch))
                outputs.append(output.detach().cpu().numpy().copy())
                batches += 1
                rows += batch.shape[0]
            np.save(stream, {**record, "linear2": outputs}, allow_pickle=True)
            records += 1
            if records % 500 == 0:
                print(f"Replayed {records} records, {batches} batches", flush=True)
    print(f"Saved {records} records, {batches} batches, {rows} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--rasr-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if not args.input.is_file() or not args.rasr_config.is_file():
        parser.error("Input recording and RASR config must exist")
    if not args.output.parent.is_dir():
        parser.error("Output parent directory must exist")
    torch.set_num_threads(1)
    scorer, checkpoint = load_scorer(args.rasr_config)
    print(f"Loaded {type(scorer).__name__} from {checkpoint}", flush=True)
    print(f"Applying {type(scorer.joint_net[8]).__name__} on CPU", flush=True)
    replay(scorer, args.input, args.output)


if __name__ == "__main__":
    main()
