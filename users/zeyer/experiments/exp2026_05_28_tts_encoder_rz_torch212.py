"""
TTS-encoder project, RZ single-GPU variant on the NEW PyTorch (torch 2.12 + cu130).

Role in the split: a separate RZ recipe (own Sis manager, torch-2.12 python),
for the base-ls-newtorch baseline only;
imports ``train_ls_base`` from the ``exp2026_05_28_tts_encoder`` library.

One experiment: the log-mel base re-run single-GPU under torch 2.12 (and the current RETURNN),
to isolate the torch-version effect on top of Run A (which moved RETURNN Aug-2025 -> Jun-2026 and torch 2.5 -> 2.7).
  base-ls (orig):    RETURNN Aug-2025 + torch 2.5  (4.06)
  base-ls-newrtrn:   current RETURNN + torch 2.7   (Run A, exp..._rz base recipe)
  base-ls-newtorch:  current RETURNN + torch 2.12  (this run)

This recipe runs under its OWN Sis manager, started with the torch-2.12 python:
settings.py has RETURNN_PYTHON_EXE = sys.executable,
so the manager's interpreter propagates as the training-job interpreter.
The _meta_hash_trigger differs from base-ls-newrtrn so this is a distinct job
(the python/torch version is not part of the Sisyphus hash).
"""

from __future__ import annotations

from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module
from i6_experiments.users.zeyer.experiments.exp2026_05_28_tts_encoder import train_ls_base

__all__ = ["py"]
__setup_root_prefix__ = "exp2026_05_28_tts_encoder_rz_torch212"


def py():
    prefix = get_setup_prefix_for_module(__name__)
    # Joint AED+CTC first-pass, as {"dev-clean", "dev-other", "test-clean", "test-other"}.
    # env-ablation dev-other: base-ls 4.06 (torch 2.5) -> newrtrn 4.14 (torch 2.7) -> newtorch 4.26 (torch 2.12).
    # {"dev-clean": 1.81, "dev-other": 4.26, "test-clean": 2.07, "test-other": 4.47}
    train_ls_base("base-ls-newtorch", prefix=prefix, config_updates_extra={"_meta_hash_trigger": "new-torch-2.12"})
