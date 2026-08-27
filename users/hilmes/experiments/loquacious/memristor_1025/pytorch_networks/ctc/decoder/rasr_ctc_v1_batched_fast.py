"""
Fast-inference variant of rasr_ctc_v1_batched: enables SynaptogenML's opt-in
fast inference path before the original decoder's init hook runs. Reusable
across any network module version -- no per-model _fast file copy needed
(contrast pytorch_networks/ctc/qat_0711/memristor_v11_fast_mem_inited.py).
"""

from .rasr_ctc_v1_batched import forward_step, forward_finish_hook
from .rasr_ctc_v1_batched import forward_init_hook as _base_forward_init_hook


def forward_init_hook(run_ctx, **kwargs):
    try:
        import torch_memristor as _synaptogen_pkg
    except ModuleNotFoundError:
        import synaptogen_ml as _synaptogen_pkg
    if hasattr(_synaptogen_pkg, "set_fast_inference"):
        _synaptogen_pkg.set_fast_inference(True)
    _base_forward_init_hook(run_ctx, **kwargs)
