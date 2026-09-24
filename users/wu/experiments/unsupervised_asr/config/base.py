"""Entry point: the blank-free control ``ctrl_20`` (phone trigram only, no word graph) and its
60-sub-epoch variant ``ctrl_20_x60`` (the E60 extension as one job), with their reads.

Banked dev-other greedy PER: ``ctrl_20`` at sub-epoch 20 0.874568, ``ctrl_20_x60`` at 60 0.873490.

Reads (``config.common``): greedy dev-other PER at every kept checkpoint; at the final checkpoint
the derangement gap, the D4 decode gap and the JS rows (per-decode statistics, no contrast rows).
Also registers the input manifests (VAD, CV split, eta, prior, duration prior).
"""

from __future__ import annotations

from typing import Any, Dict

from sisyphus import tk

__all__ = ["ctrl_arms", "register_inputs", "py"]


def register_inputs(inputs) -> None:
    """The inputs' own manifests and small outputs."""
    tk.register_output("sae/4a/data/vad/manifest.json", inputs.vad.out_manifest)
    tk.register_output("sae/4a/data/cv_split/split.stats.txt", inputs.cv_split.out_stats)
    tk.register_output("sae/4a/data/speaker_eta/eta.stats.txt", inputs.eta_job.out_stats)
    tk.register_output("sae/4a/lm/prior.npz", inputs.prior_npz)
    tk.register_output("sae/4a/lexlat_durprior/prior.json", inputs.duration_prior)


def ctrl_arms(inputs=None) -> Dict[str, Dict[str, Any]]:
    """``{"ctrl_20": ..., "ctrl_20_x60": ...}``, each ``config.common.train_and_read``'s output."""
    from ..inputs import get_inputs
    from ..training.arms import ctrl_20
    from .common import train_and_read

    inputs = get_inputs() if inputs is None else inputs
    out = {}
    for n in (20, 60):
        arm = ctrl_20(data=inputs.data, num_sub_epochs=n)
        out[arm.name] = train_and_read(arm, inputs)
    return out


def py() -> Dict[str, Any]:
    from ..inputs import get_inputs
    from .common import js_rows

    inputs = get_inputs()
    register_inputs(inputs)
    arms = ctrl_arms(inputs)
    return {"inputs": inputs, "arms": arms, "js_rows": js_rows("ctrl", arms, inputs)}
