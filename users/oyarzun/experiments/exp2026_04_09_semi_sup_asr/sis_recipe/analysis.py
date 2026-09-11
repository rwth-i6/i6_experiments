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
    cosine_similarity_summary: bool = False,
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
    :param cosine_similarity_summary: if True, also summarize, for each plotted seq, the avg pairwise
        cosine similarity within audio, within text, and between modalities in the original (pre-PCA)
        feature space (written to ``cosine_similarities.txt`` + each ``*.npz``).
    """
    if base_analysis_config is None:
        base_analysis_config = EncoderPcaConfig(
            audio_data_key=config.get("default_data_key", "data"),
            text_data_key=config.get("default_target_key", "target"),
        )

    # When ADDING a new callback option, only put it into callback_opts when it is non-default
    # (like cosine_similarity_summary below). That keeps the hash of jobs that ran before the option
    # existed unchanged. Don't make the already-present opts conditional -- that would instead
    # change the hash of the jobs that already ran with them.
    callback_opts = {
        "out_dir": out_dir_name,
        "max_points_per_modality": max_points_per_modality,
        "plot_seq_tags": plot_seq_tags,
        "max_plotted_seqs": max_plotted_seqs,
    }
    if cosine_similarity_summary:
        callback_opts["cosine_similarity_summary"] = cosine_similarity_summary

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
                device=rqmt.get("device", "gpu"),
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

class PlotDiscLossSummaryJob(tk.Job):
    """
    Sisyphus Job to plot discriminator loss with variance bounds across multiple training runs.
    """
    def __init__(self, train_jobs: List[ReturnnTrainingJob], max_epoch: int, prefix_name: str):
        self.max_epoch = max_epoch
        self.train_jobs = train_jobs
        
        # Depend on specific checkpoints so the job runs only after all trainings reach max_epoch
        self.checkpoints = [
            get_checkpoint(f"train_job_{i}", job, get_specific_checkpoint=max_epoch)
            for i, job in enumerate(train_jobs)
        ]
        
        # Output plot
        self.out_plot = self.output_path(f"disc_loss_epoch_{max_epoch}.png")
        self.add_alias(f"{prefix_name}/plot_disc_loss_epoch_{max_epoch}")
        tk.register_output(f"{prefix_name}/disc_loss_epoch_{max_epoch}.png", self.out_plot)

    def tasks(self):
        yield tk.Task("plot", rqmt={"cpu": 1, "time": 1, "mem": 4})

    def plot(self):
        import os
        import subprocess
        
        # Get learning_rates paths
        lr_files = [str(job.out_files["learning_rates"]) for job in self.train_jobs if "learning_rates" in job.out_files]
        
        script_path = os.path.join(os.path.dirname(__file__), "plot_disc_loss_summary.py")
        
        cmd = [sys.executable, script_path, str(self.max_epoch), str(self.out_plot)] + lr_files
        subprocess.check_call(cmd)

CROSS_ATT_FORWARD_STEP_MODULE = "analysis.cross_attention.forward_step.forward_step"
CROSS_ATT_CALLBACK_MODULE = "analysis.cross_attention.callback.CrossAttentionWeightsCallback"

@dataclass
class CrossAttentionConfig:
    """forward_init args passed to the cross-attention ``forward_step`` (hashed)."""

    input_data_key: str = "data"
    target_data_key: str = "phon_indices"
    input_modality: str = "audio"
    output_modality: str = "text"


def analyze_cross_attention(
    *,
    config: Dict[str, Any],
    training_name: str,
    train_job: Optional[ReturnnTrainingJob],
    train_args: Dict[str, Any],
    train_data: TrainingDatasets,
    test_data_dict: Dict[str, Any],
    checkpoints: List[Union[int, str]],
    analysis_name: str = "cross_att",
    forward_step_module: str = CROSS_ATT_FORWARD_STEP_MODULE,
    callback_module: str = CROSS_ATT_CALLBACK_MODULE,
    input_modality: str = "audio",
    output_modality: str = "text",
    masking_opts: Optional[Dict[str, Any]] = None,
    expansion_opts: Optional[Dict[str, Any]] = None,
    plot_seq_tags: Optional[List[str]] = None,
    max_plotted_seqs: int = 20,
    plot_layers: Optional[List[int]] = None,
    plot_head_average: bool = True,
    max_num_seqs: Optional[int] = None,
    out_dir_name: str = "cross_att",
    loss_name: str = "dev_loss_ce",
    extra_forward_config: Optional[ReturnnConfig] = None,
    rqmt: Optional[Dict[str, Any]] = None,
):
    assert input_modality in ("audio", "text"), input_modality
    assert output_modality in ("audio", "text"), output_modality

    default_data_key = config.get("default_data_key", "data")
    default_target_key = config.get("default_target_key", "target")
    modality_to_key = {"audio": default_data_key, "text": default_target_key}

    base_att_config = CrossAttentionConfig(
        input_data_key=modality_to_key[input_modality],
        target_data_key=modality_to_key[output_modality],
        input_modality=input_modality,
        output_modality=output_modality,
    )

    forward_step_args = asdict(base_att_config)
    if masking_opts is not None:
        forward_step_args["masking_opts"] = masking_opts
    if expansion_opts is not None:
        forward_step_args["expansion_opts"] = expansion_opts
    if max_num_seqs is None:
        max_num_seqs = max_plotted_seqs
    forward_step_args["max_num_seqs"] = max_num_seqs

    callback_opts = {
        "out_dir": out_dir_name,
        "plot_seq_tags": plot_seq_tags,
        "max_plotted_seqs": max_plotted_seqs,
        "plot_layers": plot_layers,
        "plot_head_average": plot_head_average,
    }

    config = copy.deepcopy(config)
    config["default_data_key"] = default_data_key
    config["default_target_key"] = default_target_key

    returnn_forward_config = get_forward_config(
        config=config,
        network_module=train_args["network_module"],
        extra_config=extra_forward_config if extra_forward_config else ReturnnConfig({}),
        net_args=train_args["net_args"],
        decoder_args=forward_step_args,
        decoder=forward_step_module,
        callback_module=callback_module,
        datastreams=train_data.datastreams,
        callback_opts=callback_opts,
        vocab_key=base_att_config.target_data_key,
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
