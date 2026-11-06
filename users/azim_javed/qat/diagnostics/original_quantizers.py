"""Independent pre-optimization quantizers, extracted from pinned Git blobs.

These definitions do not import or subclass the optimized quantizers. Only the
range helper and quantizer classes are executed, avoiding unrelated model deps.

The regular activation forward had an uncommitted train/eval fix BEFORE this
optimization. Reapply exactly that forward from the initial workspace read.
The mem-initialized activation class and weight class match the pinned blobs.
"""

import ast
from pathlib import Path
import subprocess
from typing import Optional

import torch
from torch import nn
import torch.ao.quantization as torch_quant
from torch.ao.quantization.utils import check_min_max_valid


REGULAR_BLOB = "38e11aa25dba090010687d9cd4e38ec7f17434bd"
MEM_INITED_BLOB = "6a2ed63015c2b09237bed54ad0e66b265bcc69e6"
SETUP_ROOT = Path(__file__).resolve().parents[1]

PRE_EXISTING_ACTIVATION_FORWARD = """
def forward(self, tensor: torch.Tensor):
    if self.training:
        tensor = self.observer(tensor)
    self.set_scale_and_zp()
    assert self.scale is not None and self.zero_point is not None, \
        "Need to calibrate before applying quant, disable apply_calibration"
    with torch.autocast(device_type=tensor.device.type, enabled=False, dtype=torch.bfloat16):
        tensor = self.quant_fn(tensor.float(), self.scale, self.zero_point, self.quant_min, self.quant_max)
    return tensor
"""


def _load(blob, class_names, restore_activation_forward=False):
    source = subprocess.check_output(["git", "show", blob], cwd=SETUP_ROOT, text=True)
    tree = ast.parse(source)
    names = {"get_quantization_range_from_bit_precision", *class_names}
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in definitions} != names:
        raise RuntimeError(f"Missing original quantizer definitions in Git blob {blob}")
    if restore_activation_forward:
        activation = next(node for node in definitions if node.name == "ActivationQuantizer")
        activation.body = [
            ast.parse(PRE_EXISTING_ACTIVATION_FORWARD).body[0]
            if isinstance(node, ast.FunctionDef) and node.name == "forward" else node
            for node in activation.body
        ]
    namespace = {
        "torch": torch, "nn": nn, "torch_quant": torch_quant,
        "Optional": Optional, "check_min_max_valid": check_min_max_valid,
        "__name__": __name__,
    }
    tree = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))
    exec(compile(tree, f"git-blob:{blob}", "exec"), namespace)
    return [namespace[name] for name in class_names]


OriginalWeightQuantizer, OriginalActivationQuantizer = _load(
    REGULAR_BLOB, ["WeightQuantizer", "ActivationQuantizer"], restore_activation_forward=True
)
(OriginalMemActivationQuantizer,) = _load(MEM_INITED_BLOB, ["ActivationQuantizer"])


def prepare_reference(module, device):
    """Apply explicit compatibility repairs needed to execute old code.

    Original per-channel forward is broken (missing axis), and original CUDA
    affine calculate_qparams (per-tensor too) fails with CPU tensor bounds on
    CUDA. Convert only the fixed observer bounds, retaining original qparam
    math and scalar synchronization. CPU per-tensor references are unmodified.
    """
    module = module.to(device)
    if module.observer.qscheme == torch.per_channel_affine:
        axis = module.channel_axis

        def quant_fn(tensor, scale, zero_point, quant_min, quant_max):
            return torch.fake_quantize_per_channel_affine(tensor, scale, zero_point, axis, quant_min, quant_max)

        module.quant_fn = quant_fn
    if torch.device(device).type == "cuda":
        module.observer.quant_min = int(module.observer.quant_min)
        module.observer.quant_max = int(module.observer.quant_max)
    return module


def reference_forward(module, tensor):
    return module(tensor)
