"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/eval_jobs.py (_forward_serializer :319,
build_posterior_forward_config :355, posterior_dump :391, POSTERIOR_BATCH_SIZE_FRAMES, POSTERIOR_MAX_SEQS).

The per-frame log-posterior dump of a recognizer checkpoint, as a plain i6_core ``ReturnnForwardJobV2``:

    ReturnnForwardJobV2 -> posteriors.hdf          [T, n_out] float32 per utterance
                        -> posteriors.frames.json  {seq tag: T}

The RETURNN-side ``forward_step`` and callback live in :mod:`.posterior_steps`; ``get_model`` is the
recognizer-only model of :mod:`..model.recognizer_only`.  Every reader downstream (greedy PER, decode
statistics, the D18 JS rows) is arithmetic on this dump, so no reader re-runs the network.

Output format, config keys and batching are the source's unchanged; only the code-object paths of
the serialized RETURNN entry points moved into this package.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from sisyphus import tk

from .posterior_steps import FORWARD_DATA_KEY

__all__ = [
    "POSTERIOR_BATCH_SIZE_FRAMES",
    "POSTERIOR_MAX_SEQS",
    "build_posterior_forward_config",
    "posterior_dump",
]

#: the package whose relative code-object paths are hashed (``unhashed_package_root``)
_PKG = __package__.rsplit(".", 1)[0]

_alias_prefix = "sae/4a/eval"

# Posterior dump batching: frames per batch.  The recognizer is a single conv layer over 1024-dim
# features, so the forward is cheap; 1.6e6 frames = 8 h of audio per batch at 50 Hz.
POSTERIOR_BATCH_SIZE_FRAMES = 1_600_000
POSTERIOR_MAX_SEQS = 200


def _forward_serializer(net_args: Dict[str, Any], extern_data: Dict[str, Any], callback_args: Dict[str, Any]):
    """``Collection`` for the posterior forward (no local package copy: the recipe dir is on sys.path)."""
    from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
    from i6_experiments.common.setups.serialization import Import, NonhashedCode, PartialImport
    from returnn.util.pprint import pformat

    return Collection(
        serializer_objects=[
            NonhashedCode(f"extern_data = {pformat(extern_data)}\n"),
            PartialImport(
                code_object_path=f"{_PKG}.model.recognizer_only.get_model",
                unhashed_package_root=_PKG,
                hashed_arguments=net_args,
                unhashed_arguments={},
                import_as="get_model",
            ),
            Import(
                code_object_path=f"{_PKG}.analysis.posterior_steps.posterior_forward_step",
                unhashed_package_root=_PKG,
                import_as="forward_step",
            ),
            PartialImport(
                code_object_path=f"{_PKG}.analysis.posterior_steps.PosteriorHdfCallback",
                unhashed_package_root=_PKG,
                hashed_arguments=callback_args,
                unhashed_arguments={},
                import_as="forward_callback",
            ),
        ],
        make_local_package_copy=False,
        packages=None,
    )


def build_posterior_forward_config(
    *,
    feature_hdfs: Sequence[tk.Path],
    net_args: Optional[Dict[str, Any]] = None,
    batch_size_frames: int = POSTERIOR_BATCH_SIZE_FRAMES,
    max_seqs: int = POSTERIOR_MAX_SEQS,
):
    """ReturnnConfig of the posterior dump over one split's L15 feature HDFs.

    ``net_args=None`` is the recognizer-only default ``RECOGNIZER_NET_ARGS``; the blank-free arms pass
    the training config's ``NET_ARGS`` (see :func:`.per.epoch_reads`).
    """
    from i6_core.returnn.config import ReturnnConfig
    from i6_experiments.common.setups.returnn.datasets.generic import HDFDataset

    if net_args is None:
        # imported only when needed: the model module is RETURNN-side code
        from ..model.recognizer_only import RECOGNIZER_NET_ARGS

        net_args = RECOGNIZER_NET_ARGS
    net_args = dict(net_args)
    # "default" order: the dump is consumed by seq tag, and an unsorted pass keeps the HDF's own
    # order so the frame table and the store line up utterance for utterance.
    data = HDFDataset(files=list(feature_hdfs), seq_ordering="default")
    # keyed by what the DATASET serves, not by what the training config calls it (FORWARD_DATA_KEY)
    extern_data = {
        FORWARD_DATA_KEY: {
            "dim": net_args["in_dim"], "shape": (None, net_args["in_dim"]), "dtype": "float16",
        },
    }
    config = {
        "forward_data": data.as_returnn_opts(),
        "batch_size": batch_size_frames,
        "max_seqs": max_seqs,
    }
    post_config = {"backend": "torch", "watch_memory": True}
    return ReturnnConfig(
        config=config,
        post_config=post_config,
        python_epilog=[_forward_serializer(net_args, extern_data, {"n_out": net_args["n_out"]})],
    )


def posterior_dump(
    *,
    name: str,
    checkpoint,
    feature_hdfs: Sequence[tk.Path],
    returnn_exe: tk.Path,
    returnn_root: tk.Path,
    net_args: Optional[Dict[str, Any]] = None,
    time_rqmt: float = 2.0,
    mem_rqmt: int = 24,
    cpu_rqmt: int = 4,
    gpu_mem: int = 24,
):
    """ReturnnForwardJobV2 dumping ``posteriors.hdf`` (+ frame table) for one checkpoint and split."""
    from i6_core.returnn.forward import ReturnnForwardJobV2

    job = ReturnnForwardJobV2(
        model_checkpoint=checkpoint,
        returnn_config=build_posterior_forward_config(feature_hdfs=feature_hdfs, net_args=net_args),
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        output_files=["posteriors.hdf", "posteriors.frames.json"],
        device="gpu",
        time_rqmt=time_rqmt,
        mem_rqmt=mem_rqmt,
        cpu_rqmt=cpu_rqmt,
    )
    job.rqmt["gpu_mem"] = gpu_mem
    job.add_alias(f"{_alias_prefix}/{name}/posteriors")
    return job
