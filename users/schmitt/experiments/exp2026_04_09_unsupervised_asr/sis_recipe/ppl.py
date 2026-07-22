"""
Pipeline for running perplexity (CE/PPL) forward jobs.

This mirrors :func:`analysis.analyze_encoder_states`: instead of beam search + sclite scoring, it
runs a RETURNN forward job whose ``forward_step``/``forward_callback`` compute the per-sequence
cross entropy / perplexity of a label sequence (see ``models.scoring.ppl.*``) and write the scores
to a file. It works both for the decoder-only phoneme LM (``input_modality=None``) and for the AED
models (``input_modality="audio"``/``"text"``, feeding an encoder input).
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

PPL_FORWARD_STEP_MODULE = "scoring.ppl.forward_step.forward_step"
PPL_CALLBACK_MODULE = "scoring.ppl.callback.PplScoresCallback"

PPL_SCORES_FILE = "ppl_scores.py.gz"
PPL_SUMMARY_FILE = "ppl_summary.txt"


@dataclass
class PplForwardConfig:
    """forward_init args passed to the PPL ``forward_step`` (hashed)."""

    target_data_key: str = "phon_indices"
    # AED-only (input_modality != None): feed an encoder input and score the output-modality decoder
    # conditioned on it. For the decoder-only LM leave input_modality at None.
    input_modality: Optional[str] = None
    input_data_key: str = "data"
    output_modality: str = "text"


def compute_ppl(
    *,
    config: Dict[str, Any],
    training_name: str,
    train_job: Optional[ReturnnTrainingJob],
    train_args: Dict[str, Any],
    train_data: TrainingDatasets,
    test_data_dict: Dict[str, Any],
    checkpoints: List[Union[int, str]],
    analysis_name: str = "ppl",
    forward_step_module: str = PPL_FORWARD_STEP_MODULE,
    callback_module: str = PPL_CALLBACK_MODULE,
    input_modality: Optional[str] = None,
    output_modality: str = "text",
    target_data_key: Optional[str] = None,
    input_data_key: str = "data",
    masking_opts: Optional[Dict[str, Any]] = None,
    loss_name: str = "dev_loss_ce",
    extra_forward_config: Optional[ReturnnConfig] = None,
    rqmt: Optional[Dict[str, Any]] = None,
):
    """
    Run the PPL forward job for one or more checkpoints / test datasets.

    :param config: RETURNN config args, e.g. ``{**config["general"], **config["recog"]}``. Provides
        ``default_data_key`` / ``default_target_key``.
    :param checkpoints: list of epochs (int) or "best"/"best4".
    :param input_modality: ``None`` (default) = decoder-only LM scoring; ``"audio"``/``"text"`` =
        AED conditional PPL (run the encoder over ``input_data_key`` and teacher-force the
        ``output_modality`` decoder conditioned on it). Use the *same* ``test_data_dict`` that
        provides both modalities (e.g. the encoder-PCA analysis dataset) for the AED path.
    :param target_data_key: the scored label sequence; defaults to the config's ``default_target_key``.
    :param masking_opts: if set (AED denoising PPL), mask the encoder input like in training.
    """
    if target_data_key is None:
        target_data_key = config.get("default_target_key", "phon_indices")

    base_ppl_config = PplForwardConfig(
        target_data_key=target_data_key,
        input_modality=input_modality,
        input_data_key=input_data_key,
        output_modality=output_modality,
    )

    forward_step_args = asdict(base_ppl_config)
    # only serialize masking when set (AED denoising PPL); keeps the plain LM/AED job hash minimal.
    if masking_opts is not None:
        forward_step_args["masking_opts"] = masking_opts

    # the AED path additionally feeds an encoder input, so the text target key must be declared in
    # extern_data next to the (audio) default_data_key. For the decoder-only LM the target key is
    # already the default_data_key, so nothing extra is declared.
    add_text_to_extern_data = base_ppl_config.input_modality is not None

    returnn_forward_config = get_forward_config(
        config=config,
        network_module=train_args["network_module"],
        extra_config=extra_forward_config if extra_forward_config else ReturnnConfig({}),
        net_args=train_args["net_args"],
        decoder_args=forward_step_args,
        decoder=forward_step_module,
        callback_module=callback_module,
        datastreams=train_data.datastreams,
        vocab_key=base_ppl_config.target_data_key,
        add_text_to_extern_data=add_text_to_extern_data,
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
                output_files=[PPL_SCORES_FILE, PPL_SUMMARY_FILE],
            )
            gpu_mem = rqmt.get("gpu_mem", None)
            if gpu_mem is not None and gpu_mem != 11:
                forward_job.rqmt["gpu_mem"] = gpu_mem
            forward_job.add_alias(prefix_name + "/forward")
            tk.register_output(prefix_name + f"/{PPL_SUMMARY_FILE}", forward_job.out_files[PPL_SUMMARY_FILE])
            tk.register_output(prefix_name + f"/{PPL_SCORES_FILE}", forward_job.out_files[PPL_SCORES_FILE])
