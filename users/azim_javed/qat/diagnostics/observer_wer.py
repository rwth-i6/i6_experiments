"""Prepare isolated decodes using an existing, generated RETURNN recognition config.

Run from the QAT setup with the base_env Python. The instrumentation changes only
activation-qparam initialization; weight quantization and qparam math stay intact.
"""

import argparse
import ast
import atexit
import json
import hashlib
import random
import re
import subprocess
import sys
from pathlib import Path


_noise_report = {}


def seed_rng(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checksum(value):
    import torch
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().contiguous().cpu()
            digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def install(mode, report_path):
    import torch
    from recipe.model_pipelines.common.memristor_layers import ActivationQuantizer
    from recipe.model_pipelines.common.memristor_layers_mem_inited import ActivationQuantizer as MemActivationQuantizer

    originals = {cls: cls.forward for cls in (ActivationQuantizer, MemActivationQuantizer)}
    records = {}

    def values(tensor):
        return tensor.detach().cpu().reshape(-1).tolist()

    def forward(module, tensor):
        original = originals[type(module)]
        if module.training or id(module) in records:
            return original(module, tensor)
        before_min = module.observer.min_val.detach().clone()
        before_max = module.observer.max_val.detach().clone()
        # Derive the checkpoint grid without setting the cached attributes in A.
        before_scale, before_zp = module.observer.calculate_qparams()
        cached_before = module.scale is not None
        if mode == "frozen":
            module.set_scale_and_zp()
        output = original(module, tensor)
        record = {
            "name": getattr(module, "_observer_audit_name", "unnamed"),
            "input_shape": list(tensor.shape),
            "cached_before": cached_before,
            "min_before": values(before_min),
            "max_before": values(before_max),
            "min_after": values(module.observer.min_val),
            "max_after": values(module.observer.max_val),
            "scale_before": values(before_scale),
            "scale_after": values(module.scale),
            "zero_point_before": values(before_zp),
            "zero_point_after": values(module.zero_point),
            "range_changed": not (
                torch.equal(before_min, module.observer.min_val)
                and torch.equal(before_max, module.observer.max_val)
            ),
            "qparams_changed": not (
                torch.equal(before_scale.float().reshape(-1), module.scale.reshape(-1))
                and torch.equal(before_zp.int().reshape(-1), module.zero_point.reshape(-1))
            ),
        }
        records[id(module)] = record
        write_report()
        return output

    def write_report():
        rows = list(records.values())
        payload = {
            "mode": mode,
            "visited_quantizers": len(rows),
            "changed_ranges": sum(row["range_changed"] for row in rows),
            "changed_qparams": sum(row["qparams_changed"] for row in rows),
            "quantizers": rows,
        }
        path = Path(report_path)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(path)

    for cls in originals:
        cls.forward = forward
    atexit.register(write_report)

    # The Python label scorer constructs a separate model after loading.
    from recipe.model_pipelines.common import pyscorers
    original_load = pyscorers._load_eval_model

    def load(*args, **kwargs):
        model = original_load(*args, **kwargs)
        name_model(model, "scorer")
        return model

    pyscorers._load_eval_model = load


def name_model(model, prefix):
    from recipe.model_pipelines.common.memristor_layers import ActivationQuantizer
    from recipe.model_pipelines.common.memristor_layers_mem_inited import ActivationQuantizer as MemActivationQuantizer
    for name, module in model.named_modules():
        if isinstance(module, (ActivationQuantizer, MemActivationQuantizer)):
            module._observer_audit_name = f"{prefix}.{name}"
    original_forward = model.forward

    def forward(*args, **kwargs):
        first = f"{prefix}_first_output_sha256" not in _noise_report
        if first:
            _noise_report[f"{prefix}_first_input_sha256"] = checksum((args, sorted(kwargs.items())))
        output = original_forward(*args, **kwargs)
        if first:
            _noise_report[f"{prefix}_first_output_sha256"] = checksum(output)
        return output

    model.forward = forward


def decoding_runtime(cfg, rasr_path, step_name="OfflineRasrRecogForwardStep", seed=17, noise_path=None):
    """Load only runtime definitions, without importing workflow/Job code."""
    import numpy as np
    import torch
    import pprint
    from abc import ABC, abstractmethod
    from time import perf_counter
    from returnn.forward_iface import ForwardCallbackIface
    from returnn.tensor import Tensor
    from recipe.model_pipelines.common import pyscorers
    import importlib
    rasr_text = Path(rasr_path).read_text()
    def imported_class(alias):
        match = re.search(r"from (\S+) import (\S+) as " + alias + r"\b", rasr_text)
        if match is None:
            raise ValueError(f"Missing {alias} in saved RASR config")
        return getattr(importlib.import_module(match.group(1)), match.group(2))

    # The source RASR config imports a training workflow just to reconstruct cfg.
    # Use the exact cfg already serialized in the saved RETURNN config instead.
    pyscorers._exec_model_imports = lambda _: {
        "ScorerModel": imported_class("ScorerModel"),
        "RecogConfig": imported_class("RecogConfig"),
        "get_model_config": lambda **kwargs: cfg,
    }
    names = {
        "_samples_to_seconds", "_traceback_to_string", "_traceback_to_transcription",
        "_traceback_to_ctm_str", "_traceback_to_score", "SearchCallback",
        "OfflineSearchCallback", "RasrRecogForwardStep", "OfflineRasrRecogForwardStep",
        "OfflineRasrRecogForwardStepV2",
    }
    source = Path(__file__).resolve().parent.parent / "recipe/model_pipelines/common/recog.py"
    tree = ast.parse(source.read_text())
    definitions = [node for node in tree.body if getattr(node, "name", None) in names]
    if {node.name for node in definitions} != names:
        raise RuntimeError("Required recognition runtime definitions have changed")
    runtime = ast.Module(body=[ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )] + definitions, type_ignores=[])
    namespace = {
        "__name__": __name__, "np": np, "torch": torch, "pprint": pprint,
        "ABC": ABC, "abstractmethod": abstractmethod, "perf_counter": perf_counter,
        "ForwardCallbackIface": ForwardCallbackIface, "Tensor": Tensor,
        "register_pyscorers": pyscorers.register_pyscorers,
    }
    exec(compile(ast.fix_missing_locations(runtime), str(source), "exec"), namespace)
    globals().update({name: namespace[name] for name in names})

    class LiteralPath:
        def get(self):
            return rasr_path

    base_step = namespace[step_name]

    class SeededStep(base_step):
        def __call__(self, **kwargs):
            if not _noise_report.get("inference_started"):
                # Reset AFTER both checkpoint/model initialization paths. Keep
                # read noise enabled, with the same initial streams per run.
                seed_rng(seed)
                _noise_report.update(seed=seed, inference_started=True,
                                     cpu_rng_start_sha256=checksum(torch.get_rng_state()))
                if torch.cuda.is_available():
                    _noise_report["cuda_rng_start_sha256"] = checksum(torch.cuda.get_rng_state())
            result = super().__call__(**kwargs)
            _noise_report["cpu_rng_end_sha256"] = checksum(torch.get_rng_state())
            if torch.cuda.is_available():
                _noise_report["cuda_rng_end_sha256"] = checksum(torch.cuda.get_rng_state())
            if noise_path:
                Path(noise_path).write_text(json.dumps(_noise_report, indent=2) + "\n")
            return result

    step = SeededStep(
        recog_rasr_config_file=LiteralPath(), sample_rate=16000,
    )
    return namespace["OfflineSearchCallback"], step


def prepare(args):
    source = args.config.resolve()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    helper = Path(__file__).resolve()
    modes = ("current", "current_repeat", "frozen") if args.repeat_current else ("current", "frozen")
    for mode in modes:
        directory = root / mode
        directory.mkdir(exist_ok=True)
        if any((directory / name).exists() for name in (
            "console.log", "observer_report.json", "search_out.ctm"
        )):
            raise FileExistsError(f"Existing run artifacts in {directory}; choose a fresh --output")
        prefix = f'''
# Observer-initialization experiment, isolated from the production recipe.
import sys
sys.path.insert(0, {str(helper.parent.parent)!r})
import importlib.util as _audit_importlib
_audit_spec = _audit_importlib.spec_from_file_location("observer_wer_audit", {str(helper)!r})
_audit = _audit_importlib.module_from_spec(_audit_spec)
sys.modules[_audit_spec.name] = _audit
_audit_spec.loader.exec_module(_audit)
_audit.seed_rng({args.seed})
_audit.install({('current' if mode == 'current_repeat' else mode)!r}, {str(directory / "observer_report.json")!r})
'''
        epilog = '''
# Small diagnostic: avoid copying the instrumented config into data workers.
torch_dataloader_opts = {"num_workers": 0}
_original_get_model = get_model
def get_model(*args, **kwargs):
    model = _original_get_model(*args, **kwargs)
    _audit.name_model(model, "encoder")
    return model
'''
        source_text = source.read_text()
        match = re.search(r'recog_rasr_config_file=tk.Path\(\s*[\'"]([^\'"]+)', source_text)
        if match is None:
            raise ValueError("Cannot locate the saved recognition RASR config")
        rasr_source = Path(match.group(1))
        rasr_text = rasr_source.read_text()
        step_match = re.search(r"forward_step = (OfflineRasrRecogForwardStep(?:V2)?)\(", source_text)
        if step_match is None:
            raise ValueError("Unsupported recognition forward step")
        step_name = step_match.group(1)
        # Retain serialized model/dataset config, replace workflow-only tail.
        source_text = source_text.split("from sisyphus import tk", 1)[0]
        if "from sisyphus" in source_text:
            raise ValueError("Unexpected Sisyphus import in serialized model config")
        if args.device:
            epilog += f"\ndevice = {args.device!r}\n"
            # Override scorer device as well as encoder device, without modifying
            # the source RASR config or changing its search settings.
            scorer_device = "cuda" if args.device == "gpu" else "cpu"
            rasr_text = re.sub(r"(?m)^device\s*=.*$", f"device = {scorer_device}", rasr_text)
        rasr_copy = directory / "rasr.config"
        rasr_copy.write_text(rasr_text)
        epilog += f"\nforward_callback, forward_step = _audit.decoding_runtime(cfg, {str(rasr_copy)!r}, step_name={step_name!r}, seed={args.seed}, noise_path={str(directory / 'noise_report.json')!r})\n"
        limit = 1 if args.probe else args.max_sequences
        if limit is not None:
            # Bound compute as well as output count: one utterance per batch.
            # This intentionally differs from the historical batch calibration.
            epilog += f'''
max_seqs = 1
_original_forward_callback = forward_callback
class forward_callback(_original_forward_callback):
    def init(self, **kwargs):
        super().init(**kwargs)
        self._audit_seq_tags = []
    def process_seq(self, *, seq_tag, outputs):
        super().process_seq(seq_tag=seq_tag, outputs=outputs)
        self._audit_seq_tags.append(seq_tag)
        if len(self._audit_seq_tags) >= {limit}:
            import json as _audit_json
            from pathlib import Path as _audit_Path
            _audit_Path("evaluated_sequences.json").write_text(_audit_json.dumps(self._audit_seq_tags, indent=2))
            self.finish()
            raise SystemExit("Bounded observer comparison completed after {limit} sequences.")
'''
        # RETURNN selects Python-config parsing from the first line. Keep its
        # marker before instrumentation; prepending code ahead of it makes the
        # loader interpret Python statements as key/value text configuration.
        (directory / "returnn.config").write_text("#!rnn.py\n" + prefix + source_text + epilog)
    print(f"Prepared {root}/{{{','.join(modes)}}}/returnn.config")
    if args.run:
        for mode in modes:
            directory = root / mode
            with (directory / "console.log").open("w") as log:
                result = subprocess.run(
                    [sys.executable, str(args.returnn), str(directory / "returnn.config")],
                    cwd=directory, stdout=log, stderr=subprocess.STDOUT,
                )
            if result.returncode:
                # SystemExit(message) is intentional for probes, but do not hide
                # startup/inference errors just because a partial report exists.
                text = (directory / "console.log").read_text()
                if not (limit is not None and f"Bounded observer comparison completed after {limit} sequences." in text):
                    raise RuntimeError(f"{mode} failed: see {directory}/console.log")
            print(f"Finished {mode}: {directory}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"))
    parser.add_argument("--probe", action="store_true", help="Decode one utterance per mode; no corpus WER")
    parser.add_argument("--seed", type=int, default=17, help="Paired Python/NumPy/PyTorch inference seed")
    parser.add_argument("--repeat-current", action="store_true", help="Add same-seed current_repeat noise reproducibility control")
    limit_group = parser.add_mutually_exclusive_group()
    limit_group.add_argument("--max-sequences", type=int, default=25,
                            help="Decode at most this many sorted utterances per mode (default: 25)")
    limit_group.add_argument("--full-corpus", action="store_const", const=None, dest="max_sequences",
                            help="Explicitly opt into the original full-corpus batching")
    parser.add_argument("--run", action="store_true", help="Execute both prepared configurations sequentially")
    parser.add_argument("--returnn", type=Path, default=Path(
        "/work/smt4/azim.javed/sisyphus_work/i6_core/tools/git/CloneGitRepositoryJob.jTcfXGpNepaQ/output/returnn/rnn.py"
    ))
    args = parser.parse_args()
    if args.max_sequences is not None and args.max_sequences < 1:
        parser.error("--max-sequences must be positive")
    prepare(args)


if __name__ == "__main__":
    main()
