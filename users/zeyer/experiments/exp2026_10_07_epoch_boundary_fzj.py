"""
Epoch boundary timing at FZJ for returnn#1939 (``torch_preload_next_train_epoch``),
see :func:`i6_experiments.users.zeyer.experiments.exp2026_05_23_returnn.py_epoch_boundary_timing_fzj`.
Registers only the timing jobs (the paper graph is built with output registration disabled).
"""

from __future__ import annotations


def py():
    """sis entry"""
    from i6_experiments.users.zeyer.experiments.exp2026_05_23_returnn import py_epoch_boundary_timing_fzj

    py_epoch_boundary_timing_fzj()
