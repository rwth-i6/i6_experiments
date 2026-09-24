"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/blankfree_train_jobs.py (``blankfree_training``)
and the resources of ``sae/emc/blankfree_pack_jobs.py`` (``PackedBlankfreeTrainJob``, one arm per
job here).

:func:`train_arm` is the training job of one arm: a plain i6_core ``ReturnnTrainingJob`` with the
banked per-arm resources (log verbosity 5, 11.5 h, 64 GB, 16 CPUs, a 96 GB GPU).  The plain job keeps
RETURNN's ``resume="run"``, which a 60-sub-epoch arm needs (it runs past one 11.5 h allocation and
restarts from its last checkpoint).  The pack job, the continue job and the resume-less
``BoundedBlankfreeTrainingJob`` are dropped.
"""

from typing import Optional, Sequence

from sisyphus import tk

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import ReturnnTrainingJob

from ..default_tools import K2_PYTHON_EXE, RETURNN_ROOT, SAE_PYTHON_EXE

__all__ = ["TIME_RQMT", "MEM_RQMT", "CPU_RQMT", "GPU_MEM_RQMT", "get_model_args", "train_arm"]

#: the allocation's wall clock (the engine cap, ``DEFAULT_ALLOC_TIME_RQMT_HOURS``)
TIME_RQMT = 11.5
MEM_RQMT = 64
CPU_RQMT = 16
GPU_MEM_RQMT = 96


def get_model_args(returnn_config: ReturnnConfig) -> dict:
    """The hashed ``get_model`` arguments of a :func:`~.config.build_train_config` config (the one
    serializer ``Collection``, the one ``get_model`` import)."""
    from i6_experiments.common.setups.returnn_pytorch.serialization import Collection

    collections = [o for o in returnn_config.python_prolog if isinstance(o, Collection)]
    assert len(collections) == 1, f"expected one serializer Collection, got {len(collections)}"
    models = [o for o in collections[0].serializer_objects if getattr(o, "import_as", None) == "get_model"]
    assert len(models) == 1, f"expected one get_model import, got {len(models)}"
    return models[0].hashed_arguments


def train_arm(
    name: str,
    returnn_config: ReturnnConfig,
    num_epochs: int,
    *,
    keep_epochs: Sequence[int],
    returnn_python_exe: Optional[tk.Path] = None,
    returnn_root: Optional[tk.Path] = None,
    time_rqmt: float = TIME_RQMT,
    alias_prefix: Optional[str] = "sae/4a/training",
) -> ReturnnTrainingJob:
    """One arm's ``ReturnnTrainingJob``.

    ``num_epochs`` must equal the length of the config's two per-sub-epoch schedules (asserted).
    ``returnn_python_exe = None`` picks the interpreter from the config: :data:`K2_PYTHON_EXE` when
    the arm carries the k2 word-graph block (``lexlat_k2_hlg``; k2 is importable only there), else
    :data:`SAE_PYTHON_EXE`.  Either way the job is given it explicitly, so i6_core never falls back
    to ``settings.py``'s ``RETURNN_PYTHON_EXE``.  ``returnn_root`` defaults to :data:`RETURNN_ROOT`.
    """
    num_epochs = int(num_epochs)
    keep = [int(e) for e in keep_epochs]
    assert num_epochs >= 1, f"a run of {num_epochs} sub-epochs trains nothing"
    assert keep and all(1 <= e <= num_epochs for e in keep), (
        f"keep_epochs {keep} asks for a checkpoint outside the {num_epochs} sub-epochs of the run"
    )
    model_args = get_model_args(returnn_config)
    n_lr = len(returnn_config.config["learning_rates"])
    n_tau = len(model_args["temperature_schedule"])
    assert n_lr == n_tau == num_epochs, (
        f"{name}: {n_tau} temperatures and {n_lr} learning rates for a {num_epochs}-sub-epoch job; "
        "both are held at their last entry, so a mismatch trains part of the run on a constant"
    )
    if returnn_python_exe is None:
        returnn_python_exe = K2_PYTHON_EXE if "lexlat_k2_hlg" in model_args else SAE_PYTHON_EXE
    job = ReturnnTrainingJob(
        returnn_config=returnn_config,
        num_epochs=num_epochs,
        log_verbosity=5,
        time_rqmt=time_rqmt,
        mem_rqmt=MEM_RQMT,
        cpu_rqmt=CPU_RQMT,
        returnn_python_exe=returnn_python_exe,
        returnn_root=RETURNN_ROOT if returnn_root is None else returnn_root,
        keep_epochs=keep,
    )
    job.rqmt["gpu_mem"] = GPU_MEM_RQMT
    if alias_prefix:
        job.add_alias(f"{alias_prefix}/{name}")
    return job
