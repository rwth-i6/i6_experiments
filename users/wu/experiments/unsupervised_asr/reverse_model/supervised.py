"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/blankfree_reverse_init_jobs.py
(``BlankfreeSupervisedReverseInitJob``) and ``sae/emc/supervised_reverse_init.py`` (``FIT``,
``INIT_HOURS``), as a RETURNN training.

ANALYSIS ONLY (uses transcripts): the supervised reverse model phi, fit on the 10 h seed's gold phone
strings (the "gold phi" of the supervised diagnostic arms).  It never initialises a main-line arm.

:func:`supervised_reverse_config` / :func:`blankfree_supervised_reverse_init` build the fit as
``SupervisedReverseDataJob`` (the seed items in the source's batch order) plus an i6_core
``ReturnnTrainingJob`` whose ``get_model`` / ``train_step`` / ``torch_batching`` are
``supervised_steps``.  What is the SAME as the source (``FIT = FitConfig(seed=42)``):
``SegmentalReverseModel(ReverseConfig())`` reset with seed 42; 8 epochs; the source's 353 batches of
8 utterances per epoch in the source's order; the loss ``-logp.sum() / frames``; Adam lr 3e-3,
betas / eps at torch's defaults (0.9, 0.999) / 1e-8, weight decay 0; global-norm clip 5 after
backward and before the step; the 28-utterance held-out NLL per frame after every epoch; every
epoch's weights kept, epoch 8 the checkpoint used (no selection).

What differs (none of it changes the fitted weights, see ``supervised_steps``):

* the source's cost gate (timed profile updates, each followed by a full reset) and its untimed
  warm-up step are cut;
* the "gradient is None" check is not made (every parameter enters every likelihood); the
  finite-gradient check is a hook;
* the source's ``recovery/`` files and ``history.json`` are replaced by RETURNN's checkpoints,
  optimizer files and ``learning_rates`` file: the held-out NLL per frame of epoch e is its
  ``dev_loss_nll_per_frame`` (the train one ``train_loss_nll_per_frame``);
* the 4 h hard budget is the job's ``time_rqmt``, and a job interrupted by the cluster resumes from
  its last epoch checkpoint (RETURNN's resume) where the source ran with ``tries=1``;
* the reported pooled NLL is summed in a different float order (a sum of per-utterance ``-log p``
  instead of a sum of fp32 ``loss x frames``): the same number up to float32 rounding.

The checkpoint ``out_checkpoints[8]`` is ``{"model": state_dict, "epoch", "step", ...}`` with the
reverse model's unprefixed keys, the layout ``build_train_config(reverse_checkpoint_path=...)``
reads (the source's ``models/epoch.008.pt`` layout).
"""

from __future__ import annotations

from typing import Optional, Sequence

from sisyphus import tk

from .supervised_data import SupervisedReverseDataJob

__all__ = ["supervised_reverse_config", "blankfree_supervised_reverse_init", "MEM_RQMT", "CPU_RQMT",
           "GPU_MEM_RQMT"]

#: the source job's resources (``BlankfreeSupervisedReverseInitJob.rqmt``)
MEM_RQMT = 64
CPU_RQMT = 16
GPU_MEM_RQMT = 96

_PKG = __name__.rsplit(".", 2)[0]  # i6_experiments.users.wu.experiments.unsupervised_asr


def _seed_dataset(hdf: tk.Path, *, partition_epoch: int):
    from i6_experiments.common.setups.returnn.datasets.base import MetaDataset
    from i6_experiments.common.setups.returnn.datasets.generic import HDFDataset

    from .supervised_steps import BATCH_ID_KEY

    return MetaDataset(
        data_map={"units": ("seed", "data"), "phones": ("seed", "phones"), "eta": ("seed", "eta"),
                  BATCH_ID_KEY: ("seed", BATCH_ID_KEY)},
        datasets={"seed": HDFDataset(files=[hdf], partition_epoch=partition_epoch, seq_ordering="default")},
        seq_order_control_dataset="seed",
    ).as_returnn_opts()


def supervised_reverse_config(*, train_hdf: tk.Path, held_hdf: tk.Path):
    """The RETURNN config of the fit on ``SupervisedReverseDataJob``'s ``train.hdf`` / ``held.hdf``."""
    from i6_core.returnn.config import ReturnnConfig
    from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
    from i6_experiments.common.setups.serialization import Import

    from ..model.reverse import ReverseConfig
    from .supervised_steps import FIT

    cfg = ReverseConfig()
    extern_data = {
        "units": {"dim": cfg.n_units, "shape": (None,), "sparse": True, "dtype": "int32"},
        "phones": {"dim": cfg.n_types, "shape": (None,), "sparse": True, "dtype": "int32"},
        "eta": {"dim": cfg.eta_dim, "shape": (None, cfg.eta_dim), "dtype": "float32"},
    }
    ser = Collection(
        serializer_objects=[
            Import(code_object_path=f"{_PKG}.reverse_model.supervised_steps.get_model",
                   unhashed_package_root=f"{_PKG}.reverse_model", import_as="get_model"),
            Import(code_object_path=f"{_PKG}.reverse_model.supervised_steps.train_step",
                   unhashed_package_root=f"{_PKG}.reverse_model", import_as="train_step"),
            Import(code_object_path=f"{_PKG}.reverse_model.supervised_steps.SeedBatchIterDataPipe",
                   unhashed_package_root=f"{_PKG}.reverse_model", import_as="torch_batching"),
        ],
        make_local_package_copy=False,
        packages=None,
    )
    config = {
        "extern_data": extern_data,
        # sub-epoch e = the source's epoch e (the HDF holds the epochs back to back)
        "train": _seed_dataset(train_hdf, partition_epoch=FIT.epochs),
        "dev": _seed_dataset(held_hdf, partition_epoch=1),
        "optimizer": {"class": "adam", "weight_decay": FIT.weight_decay},
        "learning_rate": FIT.lr,
        "learning_rates": [FIT.lr] * FIT.epochs,
        "gradient_clip_global_norm": FIT.grad_clip,
        "accum_grad_multiple_step": 1,
    }
    return ReturnnConfig(
        config=config,
        post_config={"backend": "torch", "stop_on_nonfinite_train_score": True, "log_batch_size": True},
        python_prolog=[ser],
    )


def blankfree_supervised_reverse_init(
    *,
    gold_json: tk.Path,
    ids_json: tk.Path,
    targets_hdf: tk.Path,
    train_segments: tk.Path,
    cv_segments: tk.Path,
    units_hdfs: Sequence[tk.Path],
    eta_npz: tk.Path,
    returnn_python_exe: Optional[tk.Path] = None,
    returnn_root: Optional[tk.Path] = None,
    alias: Optional[str] = "sae/4a/supervised_goldphi/phi_init",
):
    """``(data_job, train_job)``; ``train_job.out_checkpoints[8]`` is the gold phi (module doc).

    ANALYSIS ONLY (uses transcripts).  The arguments are the source job's
    (``BlankfreeSupervisedReverseInitJob``): the seed gold / ids / theta targets / seed split, the
    bed's rVAD-masked train unit HDFs and its eta table.
    """
    from i6_core.returnn.training import ReturnnTrainingJob

    from ..default_tools import RETURNN_EXE, RETURNN_ROOT
    from .supervised_steps import FIT, INIT_HOURS

    data = SupervisedReverseDataJob(gold_json=gold_json, ids_json=ids_json, targets_hdf=targets_hdf,
                                    train_segments=train_segments, cv_segments=cv_segments,
                                    units_hdfs=units_hdfs, eta_npz=eta_npz)
    job = ReturnnTrainingJob(
        returnn_config=supervised_reverse_config(train_hdf=data.out_train_hdf, held_hdf=data.out_held_hdf),
        num_epochs=FIT.epochs,
        log_verbosity=5,
        time_rqmt=INIT_HOURS,
        mem_rqmt=MEM_RQMT,
        cpu_rqmt=CPU_RQMT,
        returnn_python_exe=RETURNN_EXE if returnn_python_exe is None else returnn_python_exe,
        returnn_root=RETURNN_ROOT if returnn_root is None else returnn_root,
        keep_epochs=list(range(1, FIT.epochs + 1)),
    )
    job.rqmt["gpu_mem"] = GPU_MEM_RQMT
    if alias:
        data.add_alias(f"{alias}/data")
        job.add_alias(alias)
    return data, job
