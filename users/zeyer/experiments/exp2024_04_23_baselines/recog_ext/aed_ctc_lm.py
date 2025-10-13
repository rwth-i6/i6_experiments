"""
Joint AED+CTC, but time-sync. Also with LM. Also with CTC prior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, Any, Callable, Sequence, Tuple, Dict
import functools

from sisyphus import tk
from sisyphus.delayed_ops import DelayedBase

from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from i6_experiments.users.zeyer.model_interfaces import (
    ModelWithCheckpoint,
    ModelDef,
    ModelDefWithCfg,
    RecogDef,
    RescoreDef,
)
from i6_experiments.users.zeyer.datasets.task import Task
from i6_experiments.users.zeyer.datasets.score_results import ScoreResultCollection, RecogOutput
from i6_experiments.users.zeyer.datasets.utils.vocab import (
    ExtractVocabLabelsJob,
    ExtractVocabSpecialLabelsJob,
    ExtendVocabLabelsByNewLabelJob,
)
from i6_experiments.users.zeyer.recog import recog_model, search_dataset
from i6_experiments.users.zeyer.decoding.rescoring import combine_scores, rescore
from i6_experiments.users.zeyer.decoding.prior_rescoring import prior_score, Prior, PriorRemoveLabelRenormJob
from i6_experiments.users.zeyer.decoding.lm_rescoring import lm_score

if TYPE_CHECKING:
    from returnn_common.datasets_old_2022_10.interface import DatasetConfig

from returnn.util.basic import NotSpecified
from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray
from returnn.frontend.decoder.transformer import TransformerDecoder

from ..aed import Model, _batch_size_factor, _aed_model_def_blank_idx, _aed_model_def_blank_label
from .aed_ctc import get_ctc_prior_probs


# like i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc_recog_ext.ctc_recog_recomb_labelwise_prior_auto_scale,
# following further defaults from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc_claix2023.recog_ext_with_lm,
# now without prior
def aed_ctc_lm_timesync_recog_recomb_auto_scale(
    *,
    prefix: str,
    task: Task,
    aed_ctc_model: ModelWithCheckpoint,
    aux_ctc_layer: int,
    vocab_file: tk.Path = NotSpecified,
    vocab_opts_file: tk.Path = NotSpecified,
    ctc_soft_collapse_threshold: Optional[float] = 0.8,  # default
    n_best_list_size: int = 64,
    first_pass_recog_beam_size: int = 64,
    first_pass_search_rqmt: Optional[Dict[str, int]] = None,
    recomb_type: str = "max",
    extra_config: Optional[Dict[str, Any]] = None,
) -> ScoreResultCollection:
    """
    Like :func:`aed_ctc_timesync_recog_recomb_labelwise_prior_auto_scale` but not using a prior.

    Recog with ``model_recog_with_recomb`` and recomb enabled to get N-best list on ``task.dev_dataset``,
    then calc scores with framewise prior and LM on N-best list,
    then tune optimal scales on N-best list,
    then rescore on all ``task.eval_datasets`` using those scales,
    and also do first-pass recog (``model_recog_with_recomb``) with those scales.
    """
    if vocab_file is NotSpecified:
        vocab_file = ExtractVocabLabelsJob(_get_vocab_opts_from_task(task)).out_vocab
        # tk.register_output(f"{prefix}/vocab/{vocab}/vocab.txt.gz", vocab_file)

    if vocab_opts_file is NotSpecified:
        vocab_opts_file = ExtractVocabSpecialLabelsJob(_get_vocab_opts_from_task(task)).out_vocab_special_labels_dict
        # tk.register_output(f"{prefix}/vocab/{vocab}/vocab_opts.py", vocab_opts_file)

    # For CTC-only and then also for joint AED+CTC+prior.
    base_config = {
        "behavior_version": 24,  # should make it independent from batch size
        "__env_updates": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},  # OOM maybe otherwise
        "recog_recomb": recomb_type,
        "ctc_soft_collapse_threshold": ctc_soft_collapse_threshold,
        "aux_loss_layers": [aux_ctc_layer],
    }
    if extra_config:
        base_config = dict_update_deep(base_config, extra_config)

    # Only use CTC for first search, no AED, no prior.
    ctc_model_only = get_aed_ctc_and_labelwise_prior(aed_ctc_model=aed_ctc_model, aed_scale=0.0)
    dataset = task.dev_dataset
    ctc_scores = search_dataset(
        dataset=dataset,
        model=ctc_model_only,
        recog_def=model_recog_with_recomb,
        config={**base_config, "beam_size": n_best_list_size},
        keep_beam=True,
    )
    aed_scores = aed_score(
        ctc_scores, dataset=dataset, aed_model=aed_ctc_model, vocab=vocab_file, vocab_opts_file=vocab_opts_file
    )

    # Also register the CTC-only results. (Will not do search again, should be same hash.)
    res = recog_model(
        task=task,
        model=ctc_model_only,
        recog_def=model_recog_with_recomb,
        config={**base_config, "beam_size": n_best_list_size},
    )
    tk.register_output(f"{prefix}/ctc-only-res.txt", res.output)

    from i6_experiments.users.zeyer.datasets.utils.serialize import ReturnnDatasetToTextDictJob
    from i6_experiments.users.zeyer.datasets.task import RecogOutput

    ref = RecogOutput(
        output=ReturnnDatasetToTextDictJob(
            returnn_dataset=dataset.get_main_dataset(), data_key=dataset.get_default_target()
        ).out_txt
    )

    for f in task.recog_post_proc_funcs:  # BPE to words or so
        ctc_scores = f(ctc_scores)
        aed_scores = f(aed_scores)
        ref = f(ref)

    from i6_experiments.users.zeyer.decoding.scale_tuning import ScaleTuningJob

    opt_scales_job = ScaleTuningJob(
        scores={"ctc": ctc_scores.output, "aed": aed_scores.output},
        ref=ref.output,
        fixed_scales={"ctc": 1.0},
        evaluation="edit_distance",
    )
    opt_scales_job.rqmt["engine"] = "short"  # should be fine
    tk.register_output(f"{prefix}/opt-real-scales", opt_scales_job.out_real_scales)
    tk.register_output(f"{prefix}/opt-rel-scales", opt_scales_job.out_scales)
    # We use the real scales.
    aed_scale = opt_scales_job.out_real_scale_per_name["aed"]

    # Rescore CTC results with optimal scales. Like recog_model with lm_framewise_prior_rescore.
    # (Will not do search again, should be same hash.)
    res = recog_model(
        task=task,
        model=ctc_model_only,
        recog_def=model_recog_with_recomb,
        config={**base_config, "beam_size": n_best_list_size},
        recog_pre_post_proc_funcs_ext=[
            functools.partial(
                aed_labelwise_prior_rescore,
                aed_model=aed_ctc_model,
                aed_scale=aed_scale,
                aed_rescore_rqmt={"cpu": 4, "mem": 30, "time": 24, "gpu_mem": 48},
                vocab=vocab_file,
                vocab_opts_file=vocab_opts_file,
            )
        ],
    )
    tk.register_output(f"{prefix}/rescore-res.txt", res.output)

    # Now do 1st-pass recog with optimal scales.
    model = get_aed_ctc_and_labelwise_prior(aed_ctc_model=aed_ctc_model, aed_scale=aed_scale)
    first_pass_search_rqmt = first_pass_search_rqmt.copy() if first_pass_search_rqmt else {}
    first_pass_search_rqmt.setdefault("time", 24)
    first_pass_search_rqmt.setdefault("mem", 50)
    res = recog_model(
        task=task,
        model=model,
        recog_def=model_recog_with_recomb,
        config={
            **base_config,
            "beam_size": first_pass_recog_beam_size,
            # Batch size was fitted on our small GPUs (1080) with 11GB for beam size 32.
            # So when the beam size is larger, reduce batch size.
            # (Linear is a bit wrong, because the encoder mem consumption is independent, but anyway...)
            "batch_size": int(
                20_000 * aed_ctc_model.definition.batch_size_factor * min(32 / first_pass_recog_beam_size, 1)
            ),
        },
        search_rqmt=first_pass_search_rqmt,
        name=f"{prefix}/recog-opt-1stpass",
    )
    tk.register_output(f"{prefix}/recog-1stpass-res.txt", res.output)
    return res


# like i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc_recog_ext.ctc_recog_recomb_labelwise_prior_auto_scale,
# following further defaults from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc_claix2023.recog_ext_with_lm
def aed_ctc_lm_timesync_recog_recomb_labelwise_prior_auto_scale(
    *,
    prefix: str,
    task: Task,
    aed_ctc_model: ModelWithCheckpoint,
    aux_ctc_layer: int,
    lm: ModelWithCheckpoint,
    labelwise_prior: Prior = NotSpecified,
    vocab_file: tk.Path = NotSpecified,
    vocab_opts_file: tk.Path = NotSpecified,
    ctc_soft_collapse_threshold: float = 0.8,  # default
    n_best_list_size: int = 64,
    first_pass_recog_beam_size: int = 64,
    first_pass_search_rqmt: Optional[Dict[str, int]] = None,
    recomb_type: str = "max",
    extra_config: Optional[Dict[str, Any]] = None,
) -> ScoreResultCollection:
    """
    Recog with ``model_recog_with_recomb`` and recomb enabled to get N-best list on ``task.dev_dataset``,
    then calc scores with framewise prior and LM on N-best list,
    then tune optimal scales on N-best list,
    then rescore on all ``task.eval_datasets`` using those scales,
    and also do first-pass recog (``model_recog_with_recomb``) with those scales.
    """
    if vocab_file is NotSpecified:
        vocab_file = ExtractVocabLabelsJob(_get_vocab_opts_from_task(task)).out_vocab
        # tk.register_output(f"{prefix}/vocab/{vocab}/vocab.txt.gz", vocab_file)

    if vocab_opts_file is NotSpecified:
        vocab_opts_file = ExtractVocabSpecialLabelsJob(_get_vocab_opts_from_task(task)).out_vocab_special_labels_dict
        # tk.register_output(f"{prefix}/vocab/{vocab}/vocab_opts.py", vocab_opts_file)

    if labelwise_prior is NotSpecified:
        prior = get_ctc_prior_probs(
            aed_ctc_model,
            task.train_dataset.copy_train_as_static(),
            config={
                "behavior_version": 24,
                "batch_size": 200_000 * _batch_size_factor,
                "max_seqs": 2000,
                "aux_loss_layers": [aux_ctc_layer],
            },
        )
        prior.creator.add_alias(f"{prefix}/prior")
        tk.register_output(f"{prefix}/prior.txt", prior)
        vocab_w_blank_file = ExtendVocabLabelsByNewLabelJob(
            vocab=vocab_file, new_label=_aed_model_def_blank_label, new_label_idx=_aed_model_def_blank_idx
        ).out_vocab
        # tk.register_output(f"{prefix}/vocab/{vocab}/vocab_w_blank.txt.gz", vocab_w_blank_file)
        log_prior_wo_blank = PriorRemoveLabelRenormJob(
            prior_file=prior,
            prior_type="prob",
            vocab=vocab_w_blank_file,
            remove_label=_aed_model_def_blank_label,
            out_prior_type="log_prob",
        ).out_prior
        tk.register_output(f"{prefix}/log_prior_wo_blank.txt", log_prior_wo_blank)
        labelwise_prior = Prior(file=log_prior_wo_blank, type="log_prob", vocab=vocab_file)

    # For CTC-only and then also for joint AED+CTC+prior.
    base_config = {
        "behavior_version": 24,  # should make it independent from batch size
        "__env_updates": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},  # OOM maybe otherwise
        "recog_recomb": recomb_type,
        "ctc_soft_collapse_threshold": ctc_soft_collapse_threshold,
        "aux_loss_layers": [aux_ctc_layer],
    }
    if extra_config:
        base_config = dict_update_deep(base_config, extra_config)

    # Only use CTC for first search, no AED, no prior.
    ctc_model_only = get_aed_ctc_lm_and_labelwise_prior(aed_ctc_model=aed_ctc_model, aed_scale=0.0)
    dataset = task.dev_dataset
    ctc_scores = search_dataset(
        dataset=dataset,
        model=ctc_model_only,
        recog_def=model_recog_with_recomb,
        config={**base_config, "beam_size": n_best_list_size},
        keep_beam=True,
    )
    prior_scores = prior_score(ctc_scores, prior=labelwise_prior)
    aed_scores = aed_score(
        ctc_scores, dataset=dataset, aed_model=aed_ctc_model, vocab=vocab_file, vocab_opts_file=vocab_opts_file
    )
    lm_scores = lm_score(ctc_scores, lm=lm, vocab=vocab_file, vocab_opts_file=vocab_opts_file)

    # Also register the CTC-only results. (Will not do search again, should be same hash.)
    res = recog_model(
        task=task,
        model=ctc_model_only,
        recog_def=model_recog_with_recomb,
        config={**base_config, "beam_size": n_best_list_size},
    )
    tk.register_output(f"{prefix}/ctc-only-res.txt", res.output)

    from i6_experiments.users.zeyer.datasets.utils.serialize import ReturnnDatasetToTextDictJob
    from i6_experiments.users.zeyer.datasets.task import RecogOutput

    ref = RecogOutput(
        output=ReturnnDatasetToTextDictJob(
            returnn_dataset=dataset.get_main_dataset(), data_key=dataset.get_default_target()
        ).out_txt
    )

    for f in task.recog_post_proc_funcs:  # BPE to words or so
        ctc_scores = f(ctc_scores)
        prior_scores = f(prior_scores)
        aed_scores = f(aed_scores)
        lm_scores = f(lm_scores)
        ref = f(ref)

    from i6_experiments.users.zeyer.decoding.scale_tuning import ScaleTuningJob

    opt_scales_job = ScaleTuningJob(
        scores={
            "ctc": ctc_scores.output,
            "prior": prior_scores.output,
            "aed": aed_scores.output,
            "lm": lm_scores.output,
        },
        ref=ref.output,
        fixed_scales={"ctc": 1.0},
        negative_scales={"prior"},
        scale_relative_to={"prior": "lm"},
        evaluation="edit_distance",
    )
    opt_scales_job.rqmt["engine"] = "short"  # should be fine
    tk.register_output(f"{prefix}/opt-real-scales", opt_scales_job.out_real_scales)
    tk.register_output(f"{prefix}/opt-rel-scales", opt_scales_job.out_scales)
    # We use the real scales.
    # But prior is still handled as negative in lm_framewise_prior_rescore and 1stpass model_recog below.
    # (The DelayedBase logic on the Sis Variable should handle this.)
    prior_scale = opt_scales_job.out_real_scale_per_name["prior"] * (-1)
    aed_scale = opt_scales_job.out_real_scale_per_name["aed"]
    lm_scale = opt_scales_job.out_real_scale_per_name["lm"]

    # Rescore CTC results with optimal scales. Like recog_model with lm_framewise_prior_rescore.
    # (Will not do search again, should be same hash.)
    res = recog_model(
        task=task,
        model=ctc_model_only,
        recog_def=model_recog_with_recomb,
        config={**base_config, "beam_size": n_best_list_size},
        recog_pre_post_proc_funcs_ext=[
            functools.partial(
                aed_ctc_lm_labelwise_prior_rescore,
                prior=labelwise_prior,
                prior_scale=prior_scale,
                aed_model=aed_ctc_model,
                aed_scale=aed_scale,
                rescore_rqmt={"cpu": 4, "mem": 30, "time": 24, "gpu_mem": 48},
                vocab=vocab_file,
                vocab_opts_file=vocab_opts_file,
                lm=lm,
                lm_scale=lm_scale,
            )
        ],
    )
    tk.register_output(f"{prefix}/rescore-res.txt", res.output)

    # Now do 1st-pass recog with optimal scales.
    model = get_aed_ctc_lm_and_labelwise_prior(
        aed_ctc_model=aed_ctc_model,
        aed_scale=aed_scale,
        prior=labelwise_prior.file,
        prior_type=labelwise_prior.type,
        prior_scale=prior_scale,
        language_model=lm,
        lm_scale=lm_scale,
    )
    first_pass_search_rqmt = first_pass_search_rqmt.copy() if first_pass_search_rqmt else {}
    first_pass_search_rqmt.setdefault("time", 24)
    first_pass_search_rqmt.setdefault("mem", 50)
    res = recog_model(
        task=task,
        model=model,
        recog_def=model_recog_with_recomb,
        config={
            **base_config,
            "beam_size": first_pass_recog_beam_size,
            # Batch size was fitted on our small GPUs (1080) with 11GB for beam size 32.
            # So when the beam size is larger, reduce batch size.
            # (Linear is a bit wrong, because the encoder mem consumption is independent, but anyway...)
            "batch_size": int(
                20_000 * aed_ctc_model.definition.batch_size_factor * min(32 / first_pass_recog_beam_size, 1)
            ),
        },
        search_rqmt=first_pass_search_rqmt,
        name=f"{prefix}/recog-opt-1stpass",
    )
    tk.register_output(f"{prefix}/recog-1stpass-res.txt", res.output)
    return res


def _get_vocab_opts_from_task(task: Task) -> Dict[str, Any]:
    dataset = task.dev_dataset
    extern_data_dict = dataset.get_extern_data()
    target_dict = extern_data_dict[dataset.get_default_target()]
    return target_dict["vocab"]


# like lm_score
def aed_score(
    recog_output: RecogOutput,
    *,
    dataset: DatasetConfig,  # for encoder inputs (e.g. audio)
    aed_model: ModelWithCheckpoint,
    vocab: tk.Path,
    vocab_opts_file: tk.Path,
    rescore_rqmt: Optional[Dict[str, Any]] = None,
) -> RecogOutput:
    """
    Scores the hyps with the LM.

    :param recog_output:
        The format of the JSON is: {"<seq_tag>": [(score, "<text>"), ...], ...},
        i.e. the standard RETURNN search output with beam.
        We ignore the scores here and just use the text of the hyps.
    :param dataset: the orig data which was used to generate recog_output
    :param aed_model: AED model
    :param vocab: labels (line-based, maybe gzipped)
    :param vocab_opts_file: for LM labels. contains info about EOS, BOS, etc
    :param rescore_rqmt:
    """
    return rescore(
        recog_output=recog_output,
        dataset=dataset,
        model=aed_model,
        vocab=vocab,
        vocab_opts_file=vocab_opts_file,
        rescore_def=aed_rescore_def,
        forward_rqmt=rescore_rqmt,
    )


# like lm_rescore_def
# somewhat like aed_training...
def aed_rescore_def(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
    targets: Tensor,
    targets_beam_dim: Dim,
    targets_spatial_dim: Dim,
    **_other,
):
    targets_beam_dim  # noqa  # unused here

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    assert not data.feature_dim  # raw audio

    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)

    input_labels, (targets_w_eos_spatial_dim,) = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=model.bos_idx
    )
    targets_w_eos, _ = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=model.eos_idx, out_dims=[targets_w_eos_spatial_dim]
    )

    logits, _ = model.decoder(
        input_labels,
        spatial_dim=targets_w_eos_spatial_dim,
        encoder=enc,
        state=model.decoder.default_initial_state(batch_dims=targets.remaining_dims(targets_spatial_dim)),
    )

    assert not model.out_eos_separated  # joint distrib, std case
    log_prob = rf.log_softmax(logits, axis=model.target_dim)
    log_prob_targets = rf.gather(
        log_prob, indices=targets_w_eos, axis=model.target_dim
    )  # [batch,beam,targets_spatial_w_eos]
    log_prob_targets_seq = rf.reduce_sum(log_prob_targets, axis=targets_w_eos_spatial_dim)  # [batch,beam]
    assert log_prob_targets_seq.dims_set == set(targets.remaining_dims(targets_spatial_dim))
    return log_prob_targets_seq


aed_rescore_def: RescoreDef


# like lm_labelwise_prior_rescore
def aed_ctc_lm_labelwise_prior_rescore(
    res: RecogOutput,
    *,
    dataset: DatasetConfig,
    raw_res_search_labels: RecogOutput,
    raw_res_labels: RecogOutput,
    orig_scale: Union[float, tk.Variable, DelayedBase] = 1.0,
    aed_model: ModelWithCheckpoint,
    aed_scale: Union[float, tk.Variable, DelayedBase],
    rescore_rqmt: Optional[Dict[str, Any]] = None,
    vocab: tk.Path,
    vocab_opts_file: tk.Path,
    prior: Optional[Prior] = None,
    prior_scale: Union[float, tk.Variable, DelayedBase] = 0.0,
    lm: ModelWithCheckpoint,
    lm_scale: Union[float, tk.Variable, DelayedBase] = 0.0,
    search_labels_to_labels: Optional[Callable[[RecogOutput], RecogOutput]] = None,
) -> RecogOutput:
    """
    With functools.partial, you can use this for ``recog_post_proc_funcs`` in :func:`recog_model` and co.

    If you also want to combine a prior, e.g. for CTC, you might want to use :func:`prior_rescore` first.

    :param res:
        The format of the JSON is: {"<seq_tag>": [(score, "<text>"), ...], ...},
        i.e. the standard RETURNN search output with beam.
    :param dataset: the orig data which was used to generate res
    :param raw_res_search_labels:
    :param raw_res_labels:
    :param orig_scale: scale for the original scores
    :param aed_model: AED model
    :param aed_scale: scale for the LM scores
    :param rescore_rqmt:
    :param vocab: for LM labels in res / raw_res_labels
    :param vocab_opts_file: for LM labels. contains info about EOS, BOS, etc
    :param prior:
    :param prior_scale: scale for the prior scores. this is used as the negative weight
    :param lm:
    :param lm_scale:
    :param search_labels_to_labels: function to convert the search labels to the labels
    """
    raw_res_search_labels, search_labels_to_labels  # noqa  # unused here
    res_labels_aed_scores = aed_score(
        raw_res_labels,
        dataset=dataset,
        aed_model=aed_model,
        vocab=vocab,
        vocab_opts_file=vocab_opts_file,
        rescore_rqmt=rescore_rqmt,
    )
    scores = [(orig_scale, res), (aed_scale, res_labels_aed_scores)]
    if prior and prior_scale:
        res_labels_prior_scores = prior_score(raw_res_labels, prior=prior)
        scores.append((prior_scale * (-1), res_labels_prior_scores))
    else:
        assert isinstance(prior_scale, (int, float)) and prior_scale == 0.0
    if lm and lm_scale:
        res_labels_lm_scores = lm_score(
            raw_res_labels, lm=lm, vocab=vocab, vocab_opts_file=vocab_opts_file, rescore_rqmt=rescore_rqmt
        )
        scores.append((lm_scale, res_labels_lm_scores))
    return combine_scores(scores)


def get_aed_ctc_lm_and_labelwise_prior(
    *,
    aed_ctc_model: ModelWithCheckpoint,
    prior: Optional[tk.Path] = None,
    prior_type: str = "prob",
    prior_scale: Optional[Union[float, tk.Variable, DelayedBase]] = None,
    aed_scale: Union[float, tk.Variable],
    ctc_scale: Union[float, tk.Variable] = 1.0,
    lm_scale: Optional[Union[float, tk.Variable]] = None,
    language_model: Optional[ModelWithCheckpoint] = None,
) -> ModelWithCheckpoint:
    """Combined CTC model with LM and prior"""
    # Keep CTC model config as-is, extend below for prior and LM.
    orig_model_def_ = aed_ctc_model.definition
    if isinstance(orig_model_def_, ModelDefWithCfg):
        config: Dict[str, Any] = orig_model_def_.config.copy()
        orig_model_def_ = orig_model_def_.model_def
    else:
        config = {}

    # Add prior.
    if prior is not None or prior_scale is not None:
        assert prior is not None and prior_scale is not None
        config.update({"labelwise_prior": {"type": prior_type, "file": prior, "scale": prior_scale}})

    config.update({"aed_scale": aed_scale})
    if ctc_scale != 1.0:  # dont break hash...
        config.update({"ctc_scale": ctc_scale})

    # Add LM.
    if language_model is not None or lm_scale is not None:
        assert language_model is not None and lm_scale is not None
        config.update(
            {
                "_lm_model_def_dict": language_model.definition.config["_model_def_dict"],
                "lm_scale": lm_scale,
            }
        )
        config["preload_from_files"] = config["preload_from_files"].copy() if config.get("preload_from_files") else {}
        config["preload_from_files"]["lm"] = {"prefix": "lm.", "filename": language_model.checkpoint}

    # Also see: denoising_lm_2024.sis_recipe.tts_model.get_asr_with_tts_model_def
    # noinspection PyTypeChecker
    combined_model_def: ModelDef = functools.partial(aed_model_ext_def, orig_ctc_model_def=orig_model_def_)
    # Make it a proper ModelDef
    combined_model_def.behavior_version = max(aed_model_ext_def.behavior_version, orig_model_def_.behavior_version)
    combined_model_def.backend = orig_model_def_.backend
    combined_model_def.batch_size_factor = orig_model_def_.batch_size_factor
    # Need new recog serialization for the partial.
    config["__serialization_version"] = max(2, config.get("__serialization_version", 0))

    return ModelWithCheckpoint(
        definition=ModelDefWithCfg(model_def=combined_model_def, config=config),
        checkpoint=aed_ctc_model.checkpoint,
    )


def aed_model_ext_def(*, epoch: int, in_dim: Dim, target_dim: Dim, orig_ctc_model_def: ModelDef) -> Model:
    """Function is run within RETURNN."""
    from returnn.config import get_global_config
    import numpy

    in_dim, epoch  # noqa
    config = get_global_config()  # noqa

    model = orig_ctc_model_def(epoch=epoch, in_dim=in_dim, target_dim=target_dim)

    labelwise_prior = config.typed_value("labelwise_prior", None)
    if labelwise_prior:
        assert isinstance(labelwise_prior, dict) and set(labelwise_prior.keys()) == {"type", "file", "scale"}
        v = numpy.loadtxt(labelwise_prior["file"])
        assert v.shape == (target_dim.dimension,), (
            f"invalid shape {v.shape} for labelwise_prior {labelwise_prior['file']!r}, expected dim {target_dim}"
        )
        # The `type` is about what is stored in the file.
        # We always store it in log prob here, so we potentially need to convert it.
        if labelwise_prior["type"] == "log_prob":
            pass  # already log prob
        elif labelwise_prior["type"] == "prob":
            v = numpy.log(v)
        else:
            raise ValueError(f"invalid static_prior type {labelwise_prior['type']!r}")
        v *= labelwise_prior["scale"]  # can already apply now
        model.labelwise_prior = rf.Parameter(
            rf.convert_to_tensor(v, dims=[target_dim], dtype=rf.get_default_float_dtype()),
            auxiliary=True,
            non_critical_for_restore=True,
        )
    else:
        model.labelwise_prior = None

    lm_model_def_dict = config.typed_value("_lm_model_def_dict", None)
    if lm_model_def_dict:
        model.lm = rf.build_from_dict(lm_model_def_dict, vocab_dim=target_dim)
    else:
        model.lm = None

    return model


aed_model_ext_def: ModelDef[Model]
aed_model_ext_def.behavior_version = 24
aed_model_ext_def.backend = "torch"
aed_model_ext_def.batch_size_factor = _batch_size_factor


def model_recog_with_recomb(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Function is run within RETURNN.

    Note, some potential further improvements:
    There are many align label seqs which correspond to the same label seq,
    but the LM score is calculated for each of them.
    We could make this somehow unique depending on the label seq.
    (But unclear how exactly to do this in a GPU friendly, batched way.)

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    import returnn
    from returnn.config import get_global_config
    from i6_experiments.users.zeyer.nn_rf.soft_collapse_repeated import soft_collapse_repeated

    config = get_global_config()
    beam_size = config.int("beam_size", 12)
    recomb = config.typed_value("recog_recomb", "max")  # None, "max", "sum"
    ctc_soft_collapse_threshold = config.typed_value("ctc_soft_collapse_threshold", None)  # e.g. 0.8
    ctc_soft_collapse_reduce_type = config.typed_value("ctc_soft_collapse_reduce_type", "max_renorm")
    aed_scale = config.float("aed_scale", 1.0)
    ctc_scale = config.float("ctc_scale", 1.0)
    lm_scale = config.typed_value("lm_scale", 0.0)

    # RETURNN version is like "1.20250115.110555"
    # There was an important fix in 2025-01-17 affecting masked_scatter.
    # And another important fix in 2025-01-24 affecting masked_scatter for old PyTorch versions.
    assert tuple(int(n) for n in returnn.__version__.split(".")) >= (1, 20250125, 0), returnn.__version__

    if data.feature_dim is not None:
        batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    else:
        batch_dims = data.remaining_dims(data_spatial_dim)
    enc_collected_outputs = {}
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim, collected_outputs=enc_collected_outputs)

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = Dim(1, name="initial-beam")
    batch_dims_ = [beam_dim] + batch_dims
    neg_inf = float("-inf")
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)  # Batch, Beam

    ctc_layer_idx = model.enc_aux_logits[-1]
    linear = getattr(model, f"enc_aux_logits_{ctc_layer_idx}")
    ctc_logits = linear(enc_collected_outputs[str(ctc_layer_idx - 1)])
    ctc_label_log_prob = rf.log_softmax(ctc_logits, axis=model.wb_target_dim)  # Batch, Spatial, VocabWB
    if ctc_soft_collapse_threshold is not None:
        ctc_label_log_prob, enc_spatial_dim = soft_collapse_repeated(
            ctc_label_log_prob,
            spatial_dim=enc_spatial_dim,
            classes_dim=model.wb_target_dim,
            threshold=ctc_soft_collapse_threshold,
            reduce_type=ctc_soft_collapse_reduce_type,
        )
    ctc_label_log_prob = rf.where(
        enc_spatial_dim.get_mask(),
        ctc_label_log_prob,
        rf.sparse_to_dense(model.blank_idx, axis=model.wb_target_dim, label_value=0.0, other_value=neg_inf),
    )
    if config.bool("use_eos_postfix", False):
        ctc_label_log_prob = rf.where(
            rf.range_over_dim(model.wb_target_dim) != model.eos_idx, ctc_label_log_prob, neg_inf
        )
    # No CTC scale needed.
    ctc_label_log_prob_ta = TensorArray.unstack(ctc_label_log_prob, axis=enc_spatial_dim)  # t -> Batch, VocabWB

    target = rf.constant(model.bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)  # Batch, InBeam -> Vocab
    target_wb = rf.constant(
        model.blank_idx, dims=batch_dims_, sparse_dim=model.wb_target_dim
    )  # Batch, InBeam -> VocabWB

    seq_label = _seq_label_history_init_state(vocab_dim=model.target_dim, batch_dims=batch_dims_)

    # noinspection PyUnresolvedReferences
    labelwise_prior: Optional[rf.Parameter] = model.labelwise_prior

    if aed_scale or labelwise_prior is not None or lm_scale:
        if aed_scale:
            # We usually have TransformerDecoder, but any other type would also be ok when it has the same API.
            decoder: Optional[TransformerDecoder] = model.decoder
            assert decoder is not None
            decoder_state = decoder.default_initial_state(batch_dims=batch_dims_)  # Batch, InBeam, ...
            decoder_logits, decoder_state = decoder(
                target,
                encoder=enc,
                spatial_dim=single_step_dim,
                state=decoder_state,
            )  # Batch, InBeam, Vocab / ...
            decoder_log_probs = rf.log_softmax(decoder_logits, axis=model.target_dim)  # Batch, InBeam, Vocab
            decoder_log_probs *= aed_scale
        else:
            decoder = None
            decoder_state = 0.0
            decoder_log_probs = 0.0

        if labelwise_prior is not None:
            decoder_log_probs -= labelwise_prior  # prior scale already applied

        if lm_scale:
            # We usually have TransformerDecoder, but any other type would also be ok when it has the same API.
            # noinspection PyUnresolvedReferences
            lm: Optional[TransformerDecoder] = model.lm
            assert lm is not None

            lm_state = lm.default_initial_state(batch_dims=batch_dims_)  # Batch, InBeam, ...
            lm_logits, lm_state = lm(
                target,
                spatial_dim=single_step_dim,
                state=lm_state,
            )  # Batch, InBeam, Vocab / ...
            lm_log_probs = rf.log_softmax(lm_logits, axis=model.target_dim)  # Batch, InBeam, Vocab
            lm_log_probs *= lm_scale
            decoder_log_probs += lm_log_probs
        else:
            lm = None
            lm_state = None

    else:  # aed_scale == 0 and no prior and lm_scale == 0
        decoder = None
        decoder_state = None
        decoder_log_probs = None
        lm = None
        lm_state = None

    max_seq_len = int(enc_spatial_dim.get_dim_value())
    seq_targets_wb = []
    seq_backrefs = []
    for t in range(max_seq_len):
        prev_target = target
        prev_target_wb = target_wb

        seq_log_prob = seq_log_prob + ctc_scale * ctc_label_log_prob_ta[t]  # Batch, InBeam, VocabWB

        if decoder_log_probs is not None:
            # Now add LM score. If prev align label (target_wb) is blank or != cur, add LM score, otherwise 0.
            seq_log_prob += rf.where(
                (prev_target_wb == model.blank_idx) | (prev_target_wb != rf.range_over_dim(model.wb_target_dim)),
                _target_dense_extend_blank(
                    decoder_log_probs,
                    target_dim=model.target_dim,
                    wb_target_dim=model.wb_target_dim,
                    blank_idx=model.blank_idx,
                    value=0.0,
                ),
                0.0,
            )  # Batch, InBeam, VocabWB

        seq_log_prob, (backrefs, target_wb), beam_dim = rf.top_k(
            seq_log_prob, k_dim=Dim(beam_size, name=f"dec-step{t}-beam"), axis=[beam_dim, model.wb_target_dim]
        )
        # seq_log_prob, backrefs, target_wb: Batch, Beam
        # backrefs -> InBeam.
        # target_wb -> VocabWB.
        seq_targets_wb.append(target_wb)
        seq_backrefs.append(backrefs)

        if decoder_log_probs is not None:
            decoder_log_probs = rf.gather(decoder_log_probs, indices=backrefs)  # Batch, Beam, Vocab
        if decoder_state is not None:
            decoder_state = rf.nested.gather_nested(decoder_state, indices=backrefs)
        if lm_state is not None:
            lm_state = rf.nested.gather_nested(lm_state, indices=backrefs)
        seq_label = rf.nested.gather_nested(seq_label, indices=backrefs)

        prev_target = rf.gather(prev_target, indices=backrefs)  # Batch, Beam -> Vocab
        prev_target_wb = rf.gather(prev_target_wb, indices=backrefs)  # Batch, Beam -> VocabWB

        got_new_label: Tensor = (target_wb != model.blank_idx) & (target_wb != prev_target_wb)  # Batch, Beam -> 0|1
        target = rf.where(
            got_new_label,
            _target_remove_blank(
                target_wb, target_dim=model.target_dim, wb_target_dim=model.wb_target_dim, blank_idx=model.blank_idx
            ),
            prev_target,
        )  # Batch, Beam -> Vocab
        got_new_label_cpu = rf.copy_to_device(got_new_label, "cpu")
        if got_new_label_cpu.raw_tensor.sum().item() > 0:
            seq_label = rf.nested.mask_nested(
                _seq_label_append(seq_label, target),
                mask=got_new_label,
                mask_cpu=got_new_label_cpu,
                mask_value=seq_label,
            )

            # Recombine paths with the same label seq.
            if not recomb:
                pass
            elif recomb in ("max", "sum"):
                # Set seq_log_prob for batch entries to neg_inf if they have the same label seq.
                same_seq_labels, beam_dual_dim = _same_seq_labels(
                    seq_label.history, spatial_dim=seq_label.hist_dim, beam_dim=beam_dim
                )
                seq_log_prob_ext = rf.where(
                    same_seq_labels, rf.replace_dim_v2(seq_log_prob, in_dim=beam_dim, out_dim=beam_dual_dim), neg_inf
                )  # Batch, Beam, BeamDual
                if recomb == "sum":
                    seq_log_prob = rf.reduce_logsumexp(seq_log_prob_ext, axis=beam_dual_dim)  # Batch, Beam
                argmax_seq_log_prob = rf.reduce_argmax(seq_log_prob_ext, axis=beam_dual_dim)  # Batch, Beam -> BeamDual
                mask = argmax_seq_log_prob == rf.range_over_dim(beam_dim)  # Batch, Beam -> 0|1
                seq_log_prob = rf.where(mask, seq_log_prob, neg_inf)
                got_new_label = got_new_label & mask  # don't re-eval the LM when masked out
                got_new_label_cpu = rf.copy_to_device(got_new_label, "cpu")
            else:
                raise ValueError(f"invalid recog_recomb {recomb!r}")

        if decoder is not None and got_new_label_cpu.raw_tensor.sum().item() > 0:
            (target_, decoder_state_, lm_state_, enc_), packed_new_label_dim, packed_new_label_dim_map = (
                rf.nested.masked_select_nested(
                    (target, decoder_state, lm_state, enc if decoder is not None else None),
                    mask=got_new_label,
                    mask_cpu=got_new_label_cpu,
                    dims=batch_dims + [beam_dim],
                )
            )
            # packed_new_label_dim_map: old dim -> new dim. see _masked_select_prepare_dims
            assert packed_new_label_dim.get_dim_value() > 0

            if decoder is not None:
                decoder_logits_, decoder_state_ = decoder(
                    target_,
                    encoder=enc_,
                    spatial_dim=single_step_dim,
                    state=decoder_state_,
                )  # Flat_Batch_Beam, Vocab / ...
                decoder_log_probs_ = rf.log_softmax(decoder_logits_, axis=model.target_dim)  # Flat_Batch_Beam, Vocab
                decoder_log_probs_ *= aed_scale
            else:
                decoder_log_probs_ = 0.0

            if labelwise_prior is not None:
                decoder_log_probs_ -= labelwise_prior  # prior scale already applied

            if lm is not None:
                lm_logits_, lm_state_ = lm(
                    target_,
                    spatial_dim=single_step_dim,
                    state=lm_state_,
                )  # Flat_Batch_Beam, Vocab / ...
                lm_log_probs_ = rf.log_softmax(lm_logits_, axis=model.target_dim)  # Flat_Batch_Beam, Vocab
                lm_log_probs_ *= lm_scale
                decoder_log_probs_ += lm_log_probs_

            decoder_log_probs, decoder_state, lm_state = rf.nested.masked_scatter_nested(
                (decoder_log_probs_, decoder_state_, lm_state_),
                (decoder_log_probs, decoder_state, lm_state),
                mask=got_new_label,
                mask_cpu=got_new_label_cpu,
                dims=batch_dims + [beam_dim],
                in_dim=packed_new_label_dim,
                masked_select_dim_map=packed_new_label_dim_map,
            )  # Batch, Beam, Vocab / ...

    if decoder_log_probs is not None:
        # seq_log_prob, lm_log_probs: Batch, Beam
        # Add LM EOS score at the end.
        decoder_eos_score = rf.gather(decoder_log_probs, indices=model.eos_idx, axis=model.target_dim)
        seq_log_prob += decoder_eos_score  # Batch, Beam -> VocabWB

    # Backtrack via backrefs, resolve beams.
    seq_targets_wb_ = []
    indices = rf.range_over_dim(beam_dim)  # FinalBeam -> FinalBeam
    for backrefs, target_wb in zip(seq_backrefs[::-1], seq_targets_wb[::-1]):
        # indices: FinalBeam -> Beam
        # backrefs: Beam -> PrevBeam
        seq_targets_wb_.insert(0, rf.gather(target_wb, indices=indices))
        indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam

    seq_targets_wb__ = TensorArray(seq_targets_wb_[0])
    for target_wb in seq_targets_wb_:
        seq_targets_wb__ = seq_targets_wb__.push_back(target_wb)
    out_spatial_dim = enc_spatial_dim
    seq_targets_wb = seq_targets_wb__.stack(axis=out_spatial_dim)

    # Select valid.
    mask = rf.is_finite(seq_log_prob)  # Batch, Beam
    mask_cpu = rf.copy_to_device(mask, "cpu")
    (seq_targets_wb, seq_log_prob, out_spatial_dim), beam_dim, _ = rf.nested.masked_select_nested(
        (seq_targets_wb, seq_log_prob, out_spatial_dim), mask=mask, mask_cpu=mask_cpu, dims=[beam_dim]
    )

    return seq_targets_wb, seq_log_prob, out_spatial_dim, beam_dim


# RecogDef API
model_recog_with_recomb: RecogDef[Model]
model_recog_with_recomb.output_with_beam = True
model_recog_with_recomb.output_blank_label = _aed_model_def_blank_label
model_recog_with_recomb.batch_size_dependent = True  # our models currently just are batch-size-dependent...


def _target_remove_blank(target: Tensor, *, target_dim: Dim, wb_target_dim: Dim, blank_idx: int) -> Tensor:
    assert target.sparse_dim == wb_target_dim
    assert blank_idx == target_dim.dimension  # currently just not implemented otherwise
    return rf.set_sparse_dim(target, target_dim)


def _target_dense_extend_blank(
    target: Tensor, *, target_dim: Dim, wb_target_dim: Dim, blank_idx: int, value: float
) -> Tensor:
    assert target_dim in target.dims
    assert blank_idx == target_dim.dimension  # currently just not implemented otherwise
    res, _ = rf.pad(target, axes=[target_dim], padding=[(0, 1)], out_dims=[wb_target_dim], value=value)
    return res


def _seq_label_history_init_state(*, vocab_dim: Dim, batch_dims: Sequence[Dim]) -> rf.State:
    hist_dim = Dim(0, name="hist0")
    history = rf.zeros(list(batch_dims) + [hist_dim], dtype="int64", sparse_dim=vocab_dim)
    return rf.State(hist_dim=hist_dim, history=history)


def _seq_label_append(state: rf.State, new_label: Tensor) -> rf.State:
    hist_dim: Dim = state.hist_dim
    new_history, new_hist_dim = rf.cum_concat_step(new_label, prev_accum=state.history, axis=hist_dim)
    return rf.State(hist_dim=new_hist_dim, history=new_history)


def _same_seq_labels(seq: Tensor, *, spatial_dim: Dim, beam_dim: Dim) -> Tuple[Tensor, Dim]:
    seq_label_dual, beam_dual_dim = rf.replace_dim(seq, in_dim=beam_dim)
    same_seq_labels = rf.compare_bc(seq, "==", seq_label_dual)  # Batch, Beam, BeamDual, Spatial
    same_seq_labels = rf.reduce_all(same_seq_labels, axis=spatial_dim)  # Batch, Beam, BeamDual
    if beam_dim in spatial_dim.get_size_tensor().dims:
        seq_labels_lens = spatial_dim.get_size_tensor(device=same_seq_labels.device)
        seq_labels_dual_lens = rf.replace_dim_v2(
            seq_labels_lens, in_dim=beam_dim, out_dim=beam_dual_dim
        )  # Batch, BeamDual
        same_seq_labels_lens = rf.compare_bc(seq_labels_lens, "==", seq_labels_dual_lens)  # Batch, Beam, BeamDual
        same_seq_labels = rf.logical_and(same_seq_labels, same_seq_labels_lens)
    return same_seq_labels, beam_dual_dim
