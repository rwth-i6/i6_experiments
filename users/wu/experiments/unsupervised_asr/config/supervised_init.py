"""Entry point, ANALYSIS ONLY (uses transcripts): the supervised inits fitted on the labelled 10 h
seed.  Neither ever initialises a main-line arm; they exist for the disclosed label-using diagnostics
(the L2-0 ladder's gold phi, the supervised-init analysis arms).

* the gold phi: the 10 h supervised reverse-model fit (``reverse_model.supervised``) as a
  ``ReturnnTrainingJob``, 8 epochs, epoch 8 the checkpoint used.  Its held-out NLL per frame is the
  ``dev_loss_nll_per_frame`` column of the registered ``learning_rates`` file; banked 3.2888 at
  epoch 8 (from 3.453 at epoch 1).
* p0: the supervised blank-free recognizer (``reverse_model.p0``), selected on its held-out loss
  (``GetBestPtCheckpointJob``), its ``recognizer.`` slice exported for ``flat_checkpoint``.
* p0's dev-other greedy PER (:func:`p0_per_read`): the arms' read chain (posterior dump ->
  ``analysis.per.BlankfreeGreedyPerJob``) on the SELECTED, exported recognizer; banked 0.1894,
  selected at epoch 1.  The selected epoch is registered next to it.
"""

from __future__ import annotations

from typing import Any, Dict

from sisyphus import tk

__all__ = ["gold_phi", "p0", "p0_per_read", "py"]

_PREFIX = "sae/4a/analysis_only"


def gold_phi(inputs=None, seed=None) -> Dict[str, Any]:
    """ANALYSIS ONLY.  ``{"data", "train", "checkpoint"}``: the seed-gold reverse fit and its epoch-8
    checkpoint (``tk.Path``)."""
    from ..inputs import get_inputs, get_seed_inputs
    from ..reverse_model.supervised import blankfree_supervised_reverse_init
    from ..reverse_model.supervised_steps import FIT

    inputs = get_inputs() if inputs is None else inputs
    seed = get_seed_inputs() if seed is None else seed
    s = seed.seed_inputs
    data, train = blankfree_supervised_reverse_init(
        gold_json=s["gold_json"], ids_json=s["ids_json"], targets_hdf=seed.targets.out_hdf,
        train_segments=s["train_segments"], cv_segments=s["cv_segments"],
        units_hdfs=inputs.data["train_units_hdfs"], eta_npz=inputs.data["eta_npz"])
    return {"data": data, "train": train, "checkpoint": train.out_checkpoints[FIT.epochs].path}


def p0(inputs=None, seed=None) -> Dict[str, Any]:
    """ANALYSIS ONLY.  ``reverse_model.p0.p0_recognizer``'s output on the seed inputs."""
    from ..inputs import get_inputs, get_seed_inputs
    from ..reverse_model.p0 import p0_recognizer

    inputs = get_inputs() if inputs is None else inputs
    seed = get_seed_inputs() if seed is None else seed
    s = seed.seed_inputs
    return p0_recognizer(data=inputs.data, gold_json=s["gold_json"], ids_json=s["ids_json"],
                         old_targets_hdf=seed.targets.out_hdf, seed_train_segments=s["train_segments"],
                         seed_held_segments=s["cv_segments"], train_units_hdfs=inputs.data["train_units_hdfs"])


def p0_per_read(rec: Dict[str, Any], inputs=None) -> Dict[str, Any]:
    """ANALYSIS ONLY.  ``{"post", "per"}``: the dev-other posterior dump of p0's selected, exported
    recognizer (``rec["checkpoint"]``, :func:`p0`) and its greedy PER, through the same
    ``analysis.posterior.posterior_dump`` -> ``analysis.per.BlankfreeGreedyPerJob`` chain (and the same
    RETURNN, ``NET_ARGS`` and dev stream) as every arm's reads (``analysis.per.epoch_reads``)."""
    from i6_core.returnn.training import PtCheckpoint

    from ..analysis.per import BlankfreeGreedyPerJob
    from ..analysis.posterior import posterior_dump
    from ..default_tools import RETURNN_EXE, RETURNN_ROOT
    from ..inputs import get_inputs
    from ..training.config import NET_ARGS
    from .common import READ_SPLIT

    inputs = get_inputs() if inputs is None else inputs
    stream = inputs.dev_stream(READ_SPLIT)
    read_name = f"analysis_only/p0/best/{READ_SPLIT}"
    post = posterior_dump(name=read_name, checkpoint=PtCheckpoint(rec["checkpoint"]),
                          feature_hdfs=stream["features"], returnn_exe=RETURNN_EXE,
                          returnn_root=RETURNN_ROOT, net_args=NET_ARGS)
    per = BlankfreeGreedyPerJob(posteriors=post.out_files["posteriors.hdf"], features=stream["features"],
                                originals=stream["originals"], gold=inputs.gold, split=READ_SPLIT)
    per.add_alias(f"sae/4a/blankfree/{read_name}/per")
    return {"post": post, "per": per}


def py() -> Dict[str, Any]:
    from ..inputs import get_seed_inputs

    seed = get_seed_inputs()
    tk.register_output(f"{_PREFIX}/seed/seed_gold.stats.txt", seed.gold_job.out_stats)
    tk.register_output(f"{_PREFIX}/seed/cv_split.stats.txt", seed.split.out_stats)
    tk.register_output(f"{_PREFIX}/seed/targets.stats.txt", seed.targets.out_stats)

    phi = gold_phi(seed=seed)
    # the held-out NLL per frame of every epoch: the dev_loss_nll_per_frame column of this file
    tk.register_output(f"{_PREFIX}/supervised_goldphi/learning_rates.dev_loss_nll_per_frame",
                       phi["train"].out_learning_rates)
    tk.register_output(f"{_PREFIX}/supervised_goldphi/phi_ep8.pt", phi["checkpoint"])

    rec = p0(seed=seed)
    tk.register_output(f"{_PREFIX}/p0/support.txt", rec["support"].out_summary)
    tk.register_output(f"{_PREFIX}/p0/learning_rates", rec["train"].out_learning_rates)
    tk.register_output(f"{_PREFIX}/p0/recognizer_init.pt", rec["checkpoint"])
    # G0.R3: p0's dev-other greedy PER at its selected checkpoint, and which epoch was selected
    tk.register_output(f"{_PREFIX}/p0/selected_epoch", rec["best"].out_epoch)
    per = p0_per_read(rec)
    tk.register_output(f"{_PREFIX}/p0/best/dev-other/per.json", per["per"].out_per)
    tk.register_output(f"{_PREFIX}/p0/best/dev-other/per.txt", per["per"].out_report)
    return {"seed": seed, "gold_phi": phi, "p0": rec, "p0_per": per}
