"""
Fast-inference + noise-free readout variant of rasr_ctc_v1_batched: enables
SynaptogenML's fast path AND disables the memristor readout noise (Johnson +
shot terms) before the original decoder's init hook runs. Third arm of the
shot-noise-constant A/B: quantifies the total WER contribution of readout
noise. Requires the SynaptogenML v4 pin (set_readout_noise API).
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
    if hasattr(_synaptogen_pkg, "set_readout_noise"):
        _synaptogen_pkg.set_readout_noise(False)
    _base_forward_init_hook(run_ctx, **kwargs)
