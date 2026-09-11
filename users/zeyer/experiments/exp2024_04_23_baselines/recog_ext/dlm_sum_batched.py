"""
Batched (full-node, sharded) DLM-sum recog: CTC + denoising LM,
using the denoising_lm_2024 search code unchanged
(:func:`denoising_lm_2024.recog.dlm_sum.ctc_model_with_dlm_sum_recog`)
inside the shared batched runner (FZJ bills per node, no single-GPU jobs).

The scale-tuning stages of the RZ pipeline
(:func:`denoising_lm_2024.sis_recipe.error_correction_model.recog_model_ext`)
are not ported yet; scales are passed in fixed
(e.g. copied from an RZ-tuned run, for the sanity reproduction).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sisyphus import tk


def ctc_dlm_sum_recog_fixed_scales_batched(
    *,
    prefix: str,
    task,
    ctc_model,
    dlm,
    lm_scale: float,
    prior_file: Optional[tk.Path] = None,
    prior_scale: Optional[float] = None,
    ctc_prefix_score_scale: float = 1.0,
    num_shards: int,
    extra_config: Optional[Dict[str, Any]] = None,
):
    """
    DLM-sum first pass over all ``task.eval_datasets`` with fixed scales.

    The config replicates the eval-branch ``dlm_sum_config`` of ``recog_model_ext``
    (ctc_num_hyps 20, soft collapse 0.9 max_renorm, ctc beam 128, DLM beam 12,
    length norm 0, dlm_max_seq_len "ctc", recog_version 3 with prior / 4 without).

    :param prior_scale: as the model wrapper expects it,
        i.e. the (negative) tuned real scale times -1 (so normally positive)
    :return: :class:`ScoreResultCollection` (WERs on the eval sets)
    """
    from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
    from denoising_lm_2024.recog.ctc_with_dlm import get_ctc_with_dlm_and_labelwise_prior
    from denoising_lm_2024.recog.dlm_sum import ctc_model_with_dlm_sum_recog
    from denoising_lm_2024.recog.dlm_sum_aed import aed_ctc_model_with_dlm_sum_recog
    from .aed_ctc_batched import _combined_recog_batched

    model = get_ctc_with_dlm_and_labelwise_prior(
        ctc_model=ctc_model,
        language_model=dlm,
        lm_scale=lm_scale,
        prior=prior_file,
        prior_type="log_prob" if prior_file is not None else None,
        prior_scale=prior_scale,
    )
    config: Dict[str, Any] = {
        "ctc_num_hyps": 20,
        "ctc_soft_collapse_threshold": 0.9,
        "ctc_soft_collapse_reduce_type": "max_renorm",
        "ctc_beam_size": 128,
        "beam_size": 12,
        "ctc_prefix_score_scale": ctc_prefix_score_scale,
        "length_normalization_exponent": 0.0,
        "dlm_max_seq_len": "ctc",
        "batch_size": int(10_000 * ctc_model.definition.batch_size_factor),
        "recog_version": 3 if prior_file is not None else 4,
        "behavior_version": 24,
        "__env_updates": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    }
    if extra_config:
        config = dict_update_deep(config, extra_config)
    recog_def = (
        aed_ctc_model_with_dlm_sum_recog if config.get("dlm_sum_use_aed_variant") else ctc_model_with_dlm_sum_recog
    )
    score = _combined_recog_batched(
        prefix=prefix,
        task=task,
        model=model,
        config=config,
        num_shards=num_shards,
        recog_def=recog_def,
    )
    tk.register_output(f"{prefix}/dlm-sum-res.txt", score.output)
    return score


def ctc_dlm_sum_recog_auto_scale_batched(
    *,
    prefix: str,
    task,
    asr_model,
    dlm,
    labelwise_prior,
    aux_ctc_layer: Optional[int] = None,
    num_shards: int,
    seed_lm_scale: float = 2.0,
    seed_prior_scale: float = 0.65,
    max_dlm_scale: float = 10.0,
    extra_config: Optional[Dict[str, Any]] = None,
):
    """
    DLM-sum with scale tuning, like the RZ ``recog_model_ext`` DLM-sum stages, batched:
    fused dev work items run the DLM-sum search (DLM-only scores via ``return_only_dlm_score``)
    and CTC-rescore the same N-best in-process; the prior is scored separately (CPU);
    then ScaleTuning {am 1 fixed, prior negative rel-to-lm, lm<=max_dlm_scale},
    then the first pass via :func:`ctc_dlm_sum_recog_fixed_scales_batched`.

    :param asr_model: CTC or AED+CTC model (``aux_ctc_layer`` selects the CTC head for the latter)
    :param dlm: the denoising LM
    :param labelwise_prior: :class:`Prior`, e.g. the transcription prior
    :param seed_lm_scale: DLM scale during the tuning search (guides the beam; RZ seeds these
        from its rescore stage, we expose them as parameters)
    :param seed_prior_scale: prior scale during the tuning search (positive, as the wrapper expects)
    :return: first-pass :class:`ScoreResultCollection` (WERs on the eval sets)
    """
    from i6_core.returnn.forward import ReturnnForwardJobV2
    from i6_experiments.users.zeyer import tools_paths
    from i6_experiments.users.zeyer.forward_batched import BatchedReturnnForwardJob, _ShardedDataset
    from i6_experiments.users.zeyer.recog_batched import MergeSearchOutputShardsJob
    from i6_experiments.users.zeyer.datasets.utils.serialize import ReturnnDatasetToTextDictJob
    from i6_experiments.users.zeyer.datasets.task import RecogOutput
    from i6_experiments.users.zeyer.decoding.lm_rescoring import prior_score
    from i6_experiments.users.zeyer.decoding.scale_tuning import ScaleTuningJob
    from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
    from denoising_lm_2024.recog.ctc_with_dlm import get_ctc_with_dlm_and_labelwise_prior
    from .ctc_lm_batched import _AM_OUT_FILENAME, _LM_OUT_FILENAME

    base_config: Dict[str, Any] = {
        "behavior_version": 24,
        "__env_updates": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    }
    if aux_ctc_layer is not None:
        base_config["aux_loss_layers"] = [aux_ctc_layer]
    if extra_config:
        base_config = dict_update_deep(base_config, extra_config)

    # Tuning search with the seed scales; the collected scores are per-source
    # (DLM-only, CTC rescore, prior), so the tuning itself is seed-independent.
    tune_model = get_ctc_with_dlm_and_labelwise_prior(
        ctc_model=asr_model,
        language_model=dlm,
        lm_scale=seed_lm_scale,
        prior=labelwise_prior.file,
        prior_type=labelwise_prior.type,
        prior_scale=seed_prior_scale,
    )
    tune_config = {
        **base_config,
        # The RZ recog_model_ext dev tune config (with_prior branch).
        "ctc_num_hyps": 10,
        "ctc_soft_collapse_threshold": 0.8,
        "ctc_soft_collapse_reduce_type": "max_renorm",
        "ctc_beam_size": 128,
        "beam_size": 12,
        "ctc_prefix_score_scale": 1.0,
        "length_normalization_exponent": 0.0,
        "dlm_max_seq_len": "ctc",
        "return_only_dlm_score": True,
        "recog_version": 3,
        "batch_size": int(20_000 * asr_model.definition.batch_size_factor),
    }
    if extra_config:
        # Re-apply, so callers can override the tune settings too
        # (e.g. the AED variant's matching ctc_num_hyps == ctc_beam_size).
        tune_config = dict_update_deep(tune_config, extra_config)

    dataset = task.dev_dataset
    work_items: Dict[str, Dict[str, Any]] = {}
    shard_keys: List[str] = []
    for s in range(num_shards):
        key = "shard_%03i" % s
        ds = _ShardedDataset(dataset, num_shards=num_shards, shard_index=s, seq_ordering="random")
        cfg = _fused_dlm_sum_tune_config(dataset=ds, model_def=tune_model.definition, config=tune_config)
        cfg = ReturnnForwardJobV2.create_returnn_config(
            model_checkpoint=tune_model.checkpoint.path, returnn_config=cfg, log_verbosity=5, device="gpu"
        )
        work_items[key] = {
            "returnn_config": cfg,
            "model_checkpoint": tune_model.checkpoint,
            "output_files": [_AM_OUT_FILENAME, _LM_OUT_FILENAME],
        }
        shard_keys.append(key)
    job = BatchedReturnnForwardJob(
        work_items,
        returnn_python_exe=tools_paths.get_returnn_python_exe(),
        returnn_root=tools_paths.get_returnn_root(),
    )
    job.add_alias(f"{prefix}/dev-dlm-sum-tune-batched")

    def merged(filename: str) -> tk.Path:
        if num_shards == 1:
            return job.out_files[shard_keys[0]][filename]
        return MergeSearchOutputShardsJob(
            [job.out_files[k][filename] for k in shard_keys], output_gzip=True
        ).out_search_results

    am_scores = RecogOutput(output=merged(_AM_OUT_FILENAME))
    dlm_scores = RecogOutput(output=merged(_LM_OUT_FILENAME))
    prior_scores = prior_score(am_scores, prior=labelwise_prior)

    ref = RecogOutput(
        output=ReturnnDatasetToTextDictJob(
            returnn_dataset=dataset.get_main_dataset(), data_key=dataset.get_default_target()
        ).out_txt
    )
    for f in task.recog_post_proc_funcs:  # BPE/SPM to words
        am_scores = f(am_scores)
        dlm_scores = f(dlm_scores)
        prior_scores = f(prior_scores)
        ref = f(ref)

    opt_scales_job = ScaleTuningJob(
        scores={"am": am_scores.output, "prior": prior_scores.output, "lm": dlm_scores.output},
        ref=ref.output,
        fixed_scales={"am": 1.0},
        negative_scales={"prior"},
        scale_relative_to={"prior": "lm"},
        max_scales={"lm": max_dlm_scale, "prior": 1.0},
        evaluation="edit_distance",
    )
    opt_scales_job.rqmt["engine"] = "short"
    tk.register_output(f"{prefix}/opt-dlm-sum-real-scales", opt_scales_job.out_real_scales)
    tk.register_output(f"{prefix}/opt-dlm-sum-rel-scales", opt_scales_job.out_scales)

    return ctc_dlm_sum_recog_fixed_scales_batched(
        prefix=prefix,
        task=task,
        ctc_model=asr_model,
        dlm=dlm,
        lm_scale=opt_scales_job.out_real_scale_per_name["lm"],
        prior_file=labelwise_prior.file,
        prior_scale=opt_scales_job.out_real_scale_per_name["prior"] * (-1),
        ctc_prefix_score_scale=1.0,
        num_shards=num_shards,
        extra_config=extra_config
        if aux_ctc_layer is None
        else dict_update_deep({"aux_loss_layers": [aux_ctc_layer]}, extra_config),
    )


def _fused_dlm_sum_tune_config(*, dataset, model_def, config: Optional[Dict[str, Any]] = None):
    """
    Fused per-shard tuning config: DLM-sum search (DLM-only scores)
    plus in-process CTC rescore of the same N-best, two score files.
    """
    from i6_experiments.users.zeyer.recog import search_config_v3
    from denoising_lm_2024.recog.dlm_sum import ctc_model_with_dlm_sum_recog
    from denoising_lm_2024.recog.dlm_sum_aed import aed_ctc_model_with_dlm_sum_recog

    config = dict(config or {})
    config["forward_step"] = _fused_dlm_sum_tune_forward_step
    config["forward_callback"] = _fused_dlm_sum_get_forward_callback
    recog_def = (
        aed_ctc_model_with_dlm_sum_recog if config.get("dlm_sum_use_aed_variant") else ctc_model_with_dlm_sum_recog
    )
    return search_config_v3(
        dataset=dataset,
        model_def=model_def,
        recog_def=recog_def,
        config=config,
    )


def _fused_dlm_sum_tune_forward_step(*, model, extern_data, **_kwargs_unused):
    """
    Run within RETURNN (eager forward). DLM-sum search (DLM-only scores via
    ``return_only_dlm_score``) -> CTC rescore of the N-best (second encoder pass).

    Marks three outputs: ``hyps`` (label seqs), ``am_scores``, ``lm_scores`` (DLM; [batch, beam]).
    """
    import returnn.frontend as rf
    from returnn.tensor import batch_dim
    from returnn.config import get_global_config
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc_recog_ext import ctc_model_rescore
    from denoising_lm_2024.recog.dlm_sum import ctc_model_with_dlm_sum_recog
    from denoising_lm_2024.recog.dlm_sum_aed import aed_ctc_model_with_dlm_sum_recog

    config = get_global_config()
    default_input_key = config.typed_value("default_input")
    data = extern_data[default_input_key]
    data_spatial_dim = data.get_time_dim_tag()

    recog_def = (
        aed_ctc_model_with_dlm_sum_recog
        if config.bool("dlm_sum_use_aed_variant", False)
        else ctc_model_with_dlm_sum_recog
    )
    hyps, dlm_scores, hyps_spatial_dim, beam_dim = recog_def(model=model, data=data, data_spatial_dim=data_spatial_dim)

    am_scores = ctc_model_rescore(
        model=model,
        data=data,
        data_spatial_dim=data_spatial_dim,
        targets=hyps,
        targets_beam_dim=beam_dim,
        targets_spatial_dim=hyps_spatial_dim,
    )

    run_ctx = rf.get_run_ctx()
    run_ctx.mark_as_output(hyps, "hyps", dims=[batch_dim, beam_dim, hyps_spatial_dim])
    run_ctx.mark_as_output(am_scores, "am_scores", dims=[batch_dim, beam_dim])
    run_ctx.mark_as_output(dlm_scores, "lm_scores", dims=[batch_dim, beam_dim])


def _fused_dlm_sum_get_forward_callback():
    """Same two-file callback as the CTC+LM fused step (am.py.gz + lm.py.gz)."""
    from .ctc_lm_batched import _fused_get_forward_callback

    return _fused_get_forward_callback()


def aed_ctc_dlm_sum_recog_auto_scale_batched(
    *,
    prefix: str,
    task,
    asr_model,
    dlm,
    labelwise_prior,
    aux_ctc_layer: Optional[int] = None,
    num_shards: int,
    aed_scale: float = 1.0,
    extra_config: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """
    Like :func:`ctc_dlm_sum_recog_auto_scale_batched`, but the initial search producing
    the DLM input hypotheses is label-synchronous CTC+AED
    (:func:`denoising_lm_2024.recog.dlm_sum_aed.aed_ctc_model_with_dlm_sum_recog`).
    The DLM-sum stage itself is unchanged.

    :param aed_scale: AED scale relative to the CTC prefix scores in the initial search
        (only shapes the hypothesis set; the hyp weights are renormalized anyway)
    """
    from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep

    extra_config = dict_update_deep(
        {
            "dlm_sum_use_aed_variant": True,
            # The numHyps32LS settings of the RZ pipeline
            # (label_sync needs ctc_num_hyps == ctc_beam_size).
            "initial_ctc_search_type": "label_sync",
            "ctc_num_hyps": 32,
            "ctc_beam_size": 32,
            "ctc_soft_collapse_threshold": 0.9,
            "aed_scale": aed_scale,
        },
        extra_config,
    )
    return ctc_dlm_sum_recog_auto_scale_batched(
        prefix=prefix,
        task=task,
        asr_model=asr_model,
        dlm=dlm,
        labelwise_prior=labelwise_prior,
        aux_ctc_layer=aux_ctc_layer,
        num_shards=num_shards,
        extra_config=extra_config,
        **kwargs,
    )
