from __future__ import annotations

from returnn.tensor import Tensor, Dim
import returnn.frontend as rf

from sisyphus import tk
from sisyphus.job_path import Variable

from i6_core.returnn.search import SearchOutputRawReplaceJob

from i6_experiments.users.mueller.datasets.task import Task
from i6_experiments.users.mueller.recog import search_dataset
from i6_experiments.users.mueller.experiments.language_models.ffnn import FeedForwardLm
from i6_experiments.users.mueller.experiments.ctc_baseline.ctc import model_recog as model_recog_ctc_only

from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint, RescoreDef
from i6_experiments.users.zeyer.recog import ctc_alignment_to_label_seq
from i6_experiments.users.zeyer.decoding.prior_rescoring import Prior
from i6_experiments.users.zeyer.datasets.score_results import RecogOutput
from i6_experiments.users.zeyer.decoding.scale_tuning import ScaleTuningJob
from i6_experiments.users.zeyer.datasets.utils.serialize import ReturnnDatasetToTextDictJob
from i6_experiments.users.zeyer.decoding.rescoring import rescore
from i6_experiments.users.zeyer.decoding.lm_rescoring import prior_score

def ctc_recog_framewise_prior_auto_scale(
    *,
    prefix: str,
    task: Task,
    ctc_model: ModelWithCheckpoint,
    prior_file: tk.Path,
    lm: ModelWithCheckpoint,
    vocab_file: tk.Path,
    vocab_opts_file: tk.Path,
    vocab_w_blank_file: tk.Path,
    num_shards: int,
    search_config: dict,
) -> tuple[Variable, Variable]:
    """
    Recog with ``model_recog_ctc_only`` to get N-best list on ``task.dev_dataset``,
    then calc scores with framewise prior and LM on N-best list,
    then tune optimal scales on N-best list,
    then rescore on all ``task.eval_datasets`` using those scales,
    and also do first-pass recog (``model_recog``) with those scales.
    """
    dataset = task.dev_dataset
    _, asr_scores, _ = search_dataset(
        decoder_hyperparameters={},
        dataset=dataset,
        model=ctc_model,
        recog_def=model_recog_ctc_only,
        prior_path=None,
        config=search_config,
        num_shards=num_shards,
        pseudo_label_alignment=True,
    )
    asr_scores = RecogOutput(output=asr_scores)
    
    framewise_prior = Prior(file=prior_file, type="log_prob", vocab=vocab_w_blank_file)
    prior_scores = prior_score(asr_scores, prior=framewise_prior)
    if model_recog_ctc_only.output_blank_label:
        asr_scores = ctc_alignment_to_label_seq(asr_scores, blank_label=model_recog_ctc_only.output_blank_label)
        prior_scores = ctc_alignment_to_label_seq(prior_scores, blank_label=model_recog_ctc_only.output_blank_label)
    lm_scores = rescore(
        recog_output=asr_scores,
        model=lm,
        vocab=vocab_file,
        vocab_opts_file=vocab_opts_file,
        rescore_def=lm_rescore_def,
        forward_device="cpu",
    )

    ref = RecogOutput(
        output=ReturnnDatasetToTextDictJob(
            returnn_dataset=dataset.get_main_dataset(), data_key=dataset.get_default_target()
        ).out_txt
    )

    for f in task.recog_post_proc_funcs:  # BPE to words or so
        asr_scores = f(asr_scores)
        prior_scores = f(prior_scores)
        lm_scores = f(lm_scores)
        ref = f(ref)
        
    asr_scores = SearchOutputRawReplaceJob(asr_scores.output, [("@@", "")], output_gzip=True).out_search_results
    prior_scores = SearchOutputRawReplaceJob(prior_scores.output, [("@@", "")], output_gzip=True).out_search_results
    lm_scores = SearchOutputRawReplaceJob(lm_scores.output, [("@@", "")], output_gzip=True).out_search_results
    ref = SearchOutputRawReplaceJob(ref.output, [("@@", "")], output_gzip=True).out_search_results

    opt_scales_job = ScaleTuningJob(
        scores={"am": asr_scores, "prior": prior_scores, "lm": lm_scores},
        ref=ref,
        fixed_scales={"am": 1.0},
        negative_scales={"prior"},
        scale_relative_to={"prior": "lm"},
        evaluation="edit_distance",
    )
    opt_scales_job.add_alias(prefix)
    tk.register_output(f"{prefix}/opt-real-scales", opt_scales_job.out_real_scales)
    tk.register_output(f"{prefix}/opt-rel-scales", opt_scales_job.out_scales)
    # We use the real scales.
    prior_scale = opt_scales_job.out_real_scale_per_name["prior"] * (-1)
    lm_scale = opt_scales_job.out_real_scale_per_name["lm"]

    return prior_scale, lm_scale

def ctc_recog_labelwise_prior_auto_scale(
    *,
    prefix: str,
    task: Task,
    ctc_model: ModelWithCheckpoint,
    prior_file: tk.Path,
    lm: ModelWithCheckpoint,
    vocab_file: tk.Path,
    vocab_opts_file: tk.Path,
    num_shards: int,
    search_config: dict,
) -> tuple[Variable, Variable]:
    """
    Recog with ``model_recog_ctc_only`` to get N-best list on ``task.dev_dataset``,
    then calc scores with labelwise prior and LM on N-best list,
    then tune optimal scales on N-best list,
    then rescore on all ``task.eval_datasets`` using those scales,
    and also do first-pass recog (``model_recog``) with those scales.
    """
    dataset = task.dev_dataset
    _, asr_scores, _ = search_dataset(
        decoder_hyperparameters={},
        dataset=dataset,
        model=ctc_model,
        recog_def=model_recog_ctc_only,
        prior_path=None,
        config=search_config,
        num_shards=num_shards,
    )
    asr_scores = RecogOutput(output=asr_scores)
    
    labelwise_prior = Prior(file=prior_file, type="log_prob", vocab=vocab_file)
    prior_scores = prior_score(asr_scores, prior=labelwise_prior)
    lm_scores = rescore(
        recog_output=asr_scores,
        model=lm,
        vocab=vocab_file,
        vocab_opts_file=vocab_opts_file,
        rescore_def=lm_rescore_def,
        forward_device="cpu",
    )

    ref = RecogOutput(
        output=ReturnnDatasetToTextDictJob(
            returnn_dataset=dataset.get_main_dataset(), data_key=dataset.get_default_target()
        ).out_txt
    )

    for f in task.recog_post_proc_funcs:  # BPE to words or so
        asr_scores = f(asr_scores)
        prior_scores = f(prior_scores)
        lm_scores = f(lm_scores)
        ref = f(ref)
        
    asr_scores = SearchOutputRawReplaceJob(asr_scores.output, [("@@", "")], output_gzip=True).out_search_results
    prior_scores = SearchOutputRawReplaceJob(prior_scores.output, [("@@", "")], output_gzip=True).out_search_results
    lm_scores = SearchOutputRawReplaceJob(lm_scores.output, [("@@", "")], output_gzip=True).out_search_results
    ref = SearchOutputRawReplaceJob(ref.output, [("@@", "")], output_gzip=True).out_search_results

    opt_scales_job = ScaleTuningJob(
        scores={"am": asr_scores, "prior": prior_scores, "lm": lm_scores},
        ref=ref,
        fixed_scales={"am": 1.0},
        negative_scales={"prior"},
        scale_relative_to={"prior": "lm"},
        evaluation="edit_distance",
    )
    opt_scales_job.add_alias(prefix)
    tk.register_output(f"{prefix}/opt-real-scales", opt_scales_job.out_real_scales)
    tk.register_output(f"{prefix}/opt-rel-scales", opt_scales_job.out_scales)
    # We use the real scales.
    prior_scale = opt_scales_job.out_real_scale_per_name["prior"] * (-1)
    lm_scale = opt_scales_job.out_real_scale_per_name["lm"]

    return prior_scale, lm_scale

def lm_rescore_def(*, model: rf.Module, targets: Tensor, targets_beam_dim: Dim, targets_spatial_dim: Dim, **_other):
    import returnn.frontend as rf

    targets_beam_dim  # noqa  # unused here

    # noinspection PyTypeChecker
    model: FeedForwardLm
    vocab = model.vocab_dim.vocab
    assert vocab.bos_label_id is not None and vocab.eos_label_id is not None

    targets_w_eos, (targets_w_eos_spatial_dim,) = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=vocab.eos_label_id
    )

    batch_dims = targets.remaining_dims(targets_spatial_dim)
    lm_state = model.default_initial_state(batch_dims=batch_dims)
    logits, _ = model(
        targets,
        spatial_dim=targets_spatial_dim,
        out_spatial_dim=targets_w_eos_spatial_dim,
        state=lm_state,
    )
    assert logits.dims == (*batch_dims, targets_w_eos_spatial_dim, model.vocab_dim)
    log_prob = rf.log_softmax(logits, axis=model.vocab_dim)
    log_prob_targets = rf.gather(
        log_prob, indices=targets_w_eos, axis=model.vocab_dim
    )  # [batch,beam,targets_spatial_w_eos]
    log_prob_targets_seq = rf.reduce_sum(log_prob_targets, axis=targets_w_eos_spatial_dim)  # [batch,beam]
    assert log_prob_targets_seq.dims_set == set(batch_dims)
    return log_prob_targets_seq

lm_rescore_def: RescoreDef