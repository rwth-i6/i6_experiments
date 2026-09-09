"""
Pipeline for dumping a trained model's *encoder states* to HDF, so another setup can consume them
as plain input features instead of re-running the encoder in every training step.

This mirrors :mod:`analysis` (a ``ReturnnForwardJobV2`` whose ``forward_step``/``forward_callback``
do the work), but the output is an HDF in exactly the layout ``DumpNumpyFeaturesToHdfJobV2`` writes
for the wav2vec features -- so the dumped files drop into the existing
``HdfDataset(files=[...])`` + ``FeatureDatastream`` path unchanged. See
``models/dump/encoder_features/``.

The corpus is sharded over ``concurrent`` forward jobs via :class:`SplitSegmentFileJob`, giving one
HDF per shard (the same shape as ``dump_hdf_concurrent`` elsewhere). A full train-960 dump is on the
order of 85 GB in float32, hence the ``float16`` default.
"""

import copy
from typing import Any, Callable, Dict, List, Optional

from sisyphus import tk

from i6_core.corpus.segments import SplitSegmentFileJob
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.forward import ReturnnForwardJobV2

from .config import get_forward_config
from .default_tools import RETURNN_EXE, RETURNN_ROOT

ENCODER_FEATURES_FORWARD_STEP_MODULE = "dump.encoder_features.forward_step.forward_step"
ENCODER_FEATURES_CALLBACK_MODULE = "dump.encoder_features.callback.EncoderFeatureHdfCallback"

_HDF_FILE_NAME = "features.hdf"
_SUMMARY_FILE_NAME = "summary.txt"


def model_spec_from_config(config: Dict[str, Any], train_data) -> Dict[str, Any]:
    """
    Snapshot of everything a *forward* job needs to rebuild a model that some other config trained.

    Must be called with the config **before** it is passed to ``run_experiment``: ``run_train``
    pops ``__network_module`` & friends out of it.

    Configs that want their models to be reusable this way register the result under their
    ``training_name`` (next to ``checkpoints[training_name] = train_job.out_checkpoints``), and the
    consuming config reads both. See
    ``unsup.../configs/config_librispeech_960_w_sil_in_input_v1.py::model_specs``.
    """
    return {
        "network_module": config["__network_module"],
        "net_args": copy.deepcopy(config["model_args"]),
        "config": {**copy.deepcopy(config["general"]), **copy.deepcopy(config.get("recog", {}))},
        "datastreams": train_data.datastreams,
    }


def dump_encoder_features(
    *,
    alias_name: str,
    model_spec: Dict[str, Any],
    checkpoint,
    dataset_builder: Callable[[Optional[tk.Path]], Any],
    segment_file: Optional[tk.Path] = None,
    concurrent: int = 1,
    data_key: str = "data",
    modality: str = "audio",
    layer: Optional[int] = None,
    dtype: str = "float16",
    masking_opts: Optional[Dict[str, Any]] = None,
    batch_size: Optional[int] = None,
    extra_forward_config: Optional[ReturnnConfig] = None,
    rqmt: Optional[Dict[str, Any]] = None,
) -> List[tk.Path]:
    """
    Run the encoder over a corpus and dump its per-frame states to HDF.

    :param alias_name: alias / registered-output prefix.
    :param model_spec: from :func:`model_spec_from_config` -- the source model to run.
    :param checkpoint: the source model's checkpoint (e.g. ``train_job.out_checkpoints[500]``).
    :param dataset_builder: ``segment_file -> Dataset`` for the corpus to forward over. Called once
        per shard with that shard's segment file, so the caller decides how the (audio) input
        dataset is built.
    :param segment_file: full seq-tag list; required when ``concurrent > 1`` (it is what gets split).
    :param data_key: extern_data key holding the encoder input symbols.
    :param modality: ``"audio"`` (cluster ids) or ``"text"`` (phoneme ids).
    :param layer: 1-based encoder layer to dump; None = the last one (what recognition conditions on).
    :param dtype: on-disk dtype; ``float16`` halves the size. Consumers must cast (RETURNN hands the
        train step whatever dtype the HDF holds).
    :param masking_opts: optionally mask the encoder input like in training. Off by default.
    :param batch_size: forward batch size; defaults to the source model's recog batch size.
    :return: one HDF path per shard, ready for ``HdfDataset(files=...)``.
    """
    assert modality in ("audio", "text"), modality
    assert concurrent >= 1, concurrent

    forward_step_args: Dict[str, Any] = {"data_key": data_key, "modality": modality}
    # only pass the optional knobs when actually used, so switching them on later does not re-hash
    # the plain (last-layer, unmasked) dumps
    if layer is not None:
        forward_step_args["layer"] = layer
    if masking_opts is not None:
        forward_step_args["masking_opts"] = masking_opts

    callback_opts = {"out_file": _HDF_FILE_NAME, "dtype": dtype, "summary_file": _SUMMARY_FILE_NAME}

    config = dict(model_spec["config"])
    if batch_size is not None:
        config["batch_size"] = batch_size

    returnn_forward_config = get_forward_config(
        config=config,
        network_module=model_spec["network_module"],
        extra_config=extra_forward_config if extra_forward_config else ReturnnConfig({}),
        net_args=model_spec["net_args"],
        decoder_args=forward_step_args,
        decoder=ENCODER_FEATURES_FORWARD_STEP_MODULE,
        callback_module=ENCODER_FEATURES_CALLBACK_MODULE,
        datastreams=model_spec["datastreams"],
        callback_opts=callback_opts,
        # serialize_forward always injects a "vocab" into callback_opts; use the encoder input's own
        # datastream (the callback ignores it) instead of the default text target, so a text vocab
        # file is not dragged into the audio dump's hash.
        vocab_key=data_key,
    )

    if concurrent > 1:
        assert segment_file is not None, "sharding needs a segment file to split"
        split_job = SplitSegmentFileJob(segment_file, concurrent=concurrent)
        shard_segment_files = [split_job.out_single_segments[i] for i in range(1, concurrent + 1)]
    else:
        shard_segment_files = [segment_file]

    if rqmt is None:
        rqmt = {}

    out_hdfs: List[tk.Path] = []
    for shard_idx, shard_segment_file in enumerate(shard_segment_files):
        forward_config = copy.deepcopy(returnn_forward_config)
        forward_config.config["forward_data"] = dataset_builder(shard_segment_file).as_returnn_opts()

        prefix_name = f"{alias_name}/shard_{shard_idx}" if concurrent > 1 else alias_name
        forward_job = ReturnnForwardJobV2(
            model_checkpoint=checkpoint,
            returnn_config=forward_config,
            log_verbosity=5,
            mem_rqmt=rqmt.get("mem", 20),
            time_rqmt=rqmt.get("time", 4),
            device="gpu",
            cpu_rqmt=rqmt.get("cpu", 4),
            returnn_python_exe=RETURNN_EXE,
            returnn_root=RETURNN_ROOT,
            output_files=[_HDF_FILE_NAME, _SUMMARY_FILE_NAME],
        )
        gpu_mem = rqmt.get("gpu_mem", None)
        if gpu_mem is not None and gpu_mem != 11:
            forward_job.rqmt["gpu_mem"] = gpu_mem
        forward_job.add_alias(prefix_name + "/forward")
        tk.register_output(prefix_name + f"/{_SUMMARY_FILE_NAME}", forward_job.out_files[_SUMMARY_FILE_NAME])
        out_hdfs.append(forward_job.out_files[_HDF_FILE_NAME])

    return out_hdfs
