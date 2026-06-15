"""
Pipeline for running analysis forward jobs (e.g. the shared-encoder state PCA visualization).

This mirrors :func:`tune_eval.eval_model` but, instead of beam search + scoring, it just runs a
RETURNN forward job whose ``forward_step``/``forward_callback`` implement the analysis (see
``models.analysis.*``). All data loading is handled by the RETURNN backend.
"""

import copy
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union

from sisyphus import tk

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.forward import ReturnnForwardJobV2
from i6_core.returnn.training import ReturnnTrainingJob

from .config import get_forward_config
from .pipeline import get_checkpoint
from .default_tools import RETURNN_EXE, RETURNN_ROOT
from .data.common import TrainingDatasets

ENCODER_PCA_FORWARD_STEP_MODULE = "analysis.encoder_state_pca.forward_step.forward_step"
ENCODER_PCA_CALLBACK_MODULE = "analysis.encoder_state_pca.callback.EncoderStatePcaCallback"


@dataclass
class EncoderPcaConfig:
    """forward_init args passed to the analysis ``forward_step`` (hashed)."""

    audio_data_key: str = "data"
    text_data_key: str = "target"


def analyze_encoder_states(
    *,
    config: Dict[str, Any],
    training_name: str,
    train_job: Optional[ReturnnTrainingJob],
    train_args: Dict[str, Any],
    train_data: TrainingDatasets,
    test_data_dict: Dict[str, Any],
    checkpoints: List[Union[int, str]],
    analysis_name: str = "encoder_pca",
    forward_step_module: str = ENCODER_PCA_FORWARD_STEP_MODULE,
    callback_module: str = ENCODER_PCA_CALLBACK_MODULE,
    base_analysis_config: Optional[EncoderPcaConfig] = None,
    max_points_per_modality: int = 50_000,
    plot_seq_tags: Optional[List[str]] = None,
    max_plotted_seqs: int = 20,
    out_dir_name: str = "encoder_pca",
    loss_name: str = "dev_loss_ce",
    extra_forward_config: Optional[ReturnnConfig] = None,
    rqmt: Optional[Dict[str, Any]] = None,
):
    """
    Run the shared-encoder PCA analysis forward job for one or more checkpoints / test datasets.

    :param config: RETURNN config args, e.g. ``{**config["general"], **config["recog"]}``. Provides
        ``default_data_key`` (audio) and ``default_target_key`` (text).
    :param checkpoints: list of epochs (int) or "best"/"best4".
    :param base_analysis_config: forward_step keys; if None, derived from the config's
        ``default_data_key`` / ``default_target_key``.
    :param plot_seq_tags: if given, only these seq_tags are plotted; else the first
        ``max_plotted_seqs`` seqs are plotted (all seqs still contribute to the PCA pool).
    """
    if base_analysis_config is None:
        base_analysis_config = EncoderPcaConfig(
            audio_data_key=config.get("default_data_key", "data"),
            text_data_key=config.get("default_target_key", "target"),
        )

    callback_opts = {
        "out_dir": out_dir_name,
        "max_points_per_modality": max_points_per_modality,
        "plot_seq_tags": plot_seq_tags,
        "max_plotted_seqs": max_plotted_seqs,
    }

    # both modalities must be available to the forward_step, so declare the text key in extern_data
    returnn_forward_config = get_forward_config(
        config=config,
        network_module=train_args["network_module"],
        extra_config=extra_forward_config if extra_forward_config else ReturnnConfig({}),
        net_args=train_args["net_args"],
        decoder_args=asdict(base_analysis_config),
        decoder=forward_step_module,
        callback_module=callback_module,
        datastreams=train_data.datastreams,
        callback_opts=callback_opts,
        add_text_to_extern_data=True,
    )

    if rqmt is None:
        rqmt = {}

    for checkpoint_name in checkpoints:
        if isinstance(checkpoint_name, int):
            checkpoint = get_checkpoint(training_name, train_job, get_specific_checkpoint=checkpoint_name)
        elif checkpoint_name == "best":
            checkpoint = get_checkpoint(training_name, train_job, get_best_averaged_checkpoint=(1, loss_name))
        else:
            assert checkpoint_name == "best4", f"unknown checkpoint spec: {checkpoint_name!r}"
            checkpoint = get_checkpoint(training_name, train_job, get_best_averaged_checkpoint=(4, loss_name))

        for key, dataset in test_data_dict.items():
            forward_config = copy.deepcopy(returnn_forward_config)
            forward_config.config["forward_data"] = dataset.as_returnn_opts()

            prefix_name = f"{training_name}/{analysis_name}/{checkpoint_name}/{key}"
            forward_job = ReturnnForwardJobV2(
                model_checkpoint=checkpoint,
                returnn_config=forward_config,
                log_verbosity=5,
                mem_rqmt=rqmt.get("mem", 20),
                time_rqmt=rqmt.get("time", 1),
                device="gpu",
                cpu_rqmt=rqmt.get("cpu", 4),
                returnn_python_exe=RETURNN_EXE,
                returnn_root=RETURNN_ROOT,
                output_files=[out_dir_name],
            )
            gpu_mem = rqmt.get("gpu_mem", None)
            if gpu_mem is not None and gpu_mem != 11:
                forward_job.rqmt["gpu_mem"] = gpu_mem
            forward_job.add_alias(prefix_name + "/forward")
            tk.register_output(prefix_name + f"/{out_dir_name}", forward_job.out_files[out_dir_name])
