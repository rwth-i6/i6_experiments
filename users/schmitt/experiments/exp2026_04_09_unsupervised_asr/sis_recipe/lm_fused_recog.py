"""
Recognition with label-synchronous CTC search + phoneme-LM shallow fusion.

Unlike the normal recog path this needs **two** independently trained checkpoints in one forward
job: the CTC ASR model and the phoneme LM. RETURNN builds a single ``Model``, so we use the combined
``definitions.ctc_with_lm_v1.Model`` (``asr`` + ``lm`` submodules) and load each part from its own
file through ``preload_from_files`` (prefixes ``asr.`` / ``lm.``), with the forward job's
``model_checkpoint=None`` so nothing tries to load a full state dict over the top.

Everything downstream (SearchTakeBestJob, sclite scoring, score collection) is the standard path.
"""

import copy
from typing import Any, Dict, List, Optional, Sequence, Union

from sisyphus import tk

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import ReturnnTrainingJob

from i6_experiments.users.zeyer.datasets.score_results import join_score_results

from .config import get_forward_config
from .pipeline import get_checkpoint, search_single
from .data.common import TrainingDatasets
from .tune_eval import default_returnn, SummarizeScoreResultsJobV2

LM_FUSED_NETWORK_MODULE = "definitions.ctc_with_lm_v1.Model"
LM_FUSED_FORWARD_STEP_MODULE = "recognition.discrete_audio_ctc.forward_step_lm.forward_step"


def _checkpoint_path(checkpoint) -> tk.Path:
    """The ``.pt`` path of a (Pt)Checkpoint, for use as a ``preload_from_files`` filename."""
    return getattr(checkpoint, "path", checkpoint)


def run_lm_fused_recog(
    *,
    training_name: str,
    config: Dict[str, Any],
    train_job: Optional[ReturnnTrainingJob],
    train_args: Dict[str, Any],
    train_data: TrainingDatasets,
    test_data_dict: Dict[str, Any],
    checkpoints: List[Union[int, str]],
    lm_checkpoint: tk.Path,
    lm_args: Dict[str, Any],
    recog_name: str = "recog_lm",
    beam_size: int = 12,
    lm_scales: Sequence[float] = (0.0,),
    length_rewards: Sequence[float] = (0.0,),
    ctc_scale: float = 1.0,
    score_type: str = "sum",
    search_type: str = "label_sync",
    loss_name: str = "dev_loss_ce",
    main_eval_measure_key: str = "dev-other",
    extra_forward_config: Optional[ReturnnConfig] = None,
    rqmt: Optional[Dict[str, int]] = None,
    recog_post_proc_funcs=None,
):
    """
    One forward+score job per (checkpoint, lm_scale, length_reward).

    :param checkpoints: ASR epochs (or "best"/"best4").
    :param lm_checkpoint: ``.pt`` of the trained phoneme LM.
    :param lm_args: net_args of the phoneme LM (``model_dim``/``out_dim``/``num_layers``/``num_heads``).
    :param lm_scales: shallow-fusion weights to sweep. ``0.0`` = pure CTC prefix search, which is the
        control every sweep should include -- it isolates the effect of the LM from the effect of
        replacing greedy decoding with a beam.
    :param length_rewards: per-label bonus to sweep, counteracting the length bias fusion introduces.
    :param score_type: ``"sum"`` = CTC marginal, ``"max"`` = Viterbi / best-path. The marginal sums
        over all alignments, and since the alignment count grows with ``|y|`` it over-generates badly
        on a weak model (measured: 111 labels/utt vs a 62-label reference). ``"max"`` has no such
        bias and at ``lm_scale=0`` reduces to greedy decoding -- provided the beam is wide enough not
        to prune the best path away.
    :param search_type: ``"time_sync"`` (classic CTC prefix beam search -- **use this**) or
        ``"label_sync"`` (kept only for reproducing the earlier, invalid results). With
        ``time_sync`` + ``max`` the search reproduces greedy decoding exactly already at beam 4.
    """
    default_data_key = config.get("default_data_key", "data")
    default_target_key = config.get("default_target_key", "phon_indices")

    result_collections = {}
    for checkpoint_name in checkpoints:
        if isinstance(checkpoint_name, int):
            asr_checkpoint = get_checkpoint(training_name, train_job, get_specific_checkpoint=checkpoint_name)
        elif checkpoint_name == "best":
            asr_checkpoint = get_checkpoint(training_name, train_job, get_best_averaged_checkpoint=(1, loss_name))
        else:
            assert checkpoint_name == "best4", f"unknown checkpoint spec: {checkpoint_name!r}"
            asr_checkpoint = get_checkpoint(training_name, train_job, get_best_averaged_checkpoint=(4, loss_name))

        # both models are loaded by prefix; the forward job itself gets no main checkpoint
        preload_config = ReturnnConfig(
            config={
                "preload_from_files": {
                    "asr": {
                        "filename": _checkpoint_path(asr_checkpoint),
                        "prefix": "asr.",
                        "checkpoint_key": "model",
                    },
                    "lm": {
                        "filename": lm_checkpoint,
                        "prefix": "lm.",
                        "checkpoint_key": "model",
                    },
                }
            }
        )
        if extra_forward_config is not None:
            preload_config.update(copy.deepcopy(extra_forward_config))

        for lm_scale in lm_scales:
            for length_reward in length_rewards:
                forward_step_args: Dict[str, Any] = {"beam_size": beam_size}
                if score_type != "sum":
                    forward_step_args["score_type"] = score_type
                if search_type != "label_sync":
                    forward_step_args["search_type"] = search_type
                if ctc_scale != 1.0:
                    forward_step_args["ctc_scale"] = ctc_scale
                if lm_scale != 0.0:
                    forward_step_args["lm_scale"] = lm_scale
                if length_reward != 0.0:
                    forward_step_args["length_reward"] = length_reward

                returnn_config = get_forward_config(
                    config=config,
                    network_module=LM_FUSED_NETWORK_MODULE,
                    extra_config=preload_config,
                    net_args={"asr_args": train_args["net_args"], "lm_args": lm_args},
                    decoder_args=forward_step_args,
                    decoder=LM_FUSED_FORWARD_STEP_MODULE,
                    callback_module=config["__callback_module"],
                    datastreams=train_data.datastreams,
                    callback_opts={"include_beam": True},
                    vocab_key=default_target_key,
                )

                variant = f"lm-{lm_scale}_lr-{length_reward}_beam-{beam_size}"
                if score_type != "sum":
                    variant += f"_{score_type}"
                if search_type != "label_sync":
                    variant += f"_{search_type}"
                recog_path = f"{training_name}/{recog_name}/{variant}/{checkpoint_name}"
                outputs = {}
                for key, dataset in test_data_dict.items():
                    recog_dataset_path = f"{recog_path}/{key}"
                    score_result, _, _ = search_single(
                        recog_dataset_path,
                        returnn_config=returnn_config,
                        checkpoint=None,  # both parts come from preload_from_files
                        recognition_dataset=dataset,
                        dataset_name=key,
                        **default_returnn,
                        rqmt=rqmt,
                        vocab_opts=train_data.datastreams[default_target_key].as_returnn_targets_opts(),
                        recog_post_proc_funcs=recog_post_proc_funcs,
                        score_target_key=default_target_key,
                    )
                    outputs[key] = score_result
                    tk.register_output(f"{recog_dataset_path}/wer", score_result.main_measure_value)
                    tk.register_output(f"{recog_dataset_path}/report", score_result.report)
                result_collections[f"{variant}/{checkpoint_name}"] = join_score_results(
                    outputs, main_measure_key=main_eval_measure_key
                )

    summarize_job = SummarizeScoreResultsJobV2(result_collections)
    tk.register_output(f"{training_name}/{recog_name}/results_all", summarize_job.out_results_all_epochs_json)
