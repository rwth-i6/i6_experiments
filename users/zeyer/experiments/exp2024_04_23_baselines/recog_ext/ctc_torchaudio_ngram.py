"""
TorchAudio (internally using Flashlight) for CTC with ngram LM
"""

from __future__ import annotations
from typing import Optional, Any, Literal, Tuple, List, Dict
import functools

from sisyphus import tk

from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint, RecogDef
from i6_experiments.users.zeyer.datasets.task import Task, DatasetConfig
from i6_experiments.users.zeyer.datasets.score_results import ScoreResultCollection
from i6_experiments.users.zeyer.recog import recog_model, search_dataset, ctc_alignment_to_label_seq
from i6_experiments.users.zeyer.decoding.lm_rescoring import ngram_lm_framewise_prior_rescore, ngram_score_v2
from i6_experiments.users.zeyer.decoding.prior_rescoring import prior_score, Prior
from i6_experiments.users.zeyer.datasets.utils.vocab import ExtractLineBasedLexiconJob, get_vocab_opts_from_task

from ..ctc import Model
from ..ctc_recog_ext import get_ctc_with_ngram_lm_and_framewise_prior, get_ctc_prior

from returnn.util.basic import NotSpecified
from returnn.tensor import Tensor, Dim, batch_dim
import returnn.frontend as rf


def ctc_recog_ngram_lm_framewise_prior_auto_scale(
    *,
    prefix: str,
    task: Task,
    ctc_model: ModelWithCheckpoint,
    framewise_prior: Prior = NotSpecified,
    framewise_prior_dataset: Optional[DatasetConfig] = None,
    ngram_language_model: tk.Path,
    lm_label_level: Literal["word", "task"] = "word",
    lm_word_list: Optional[tk.Path] = None,
    n_best_list_size: int = 64,
    ctc_decoder_opts: Dict[str, Any],
    extra_config: Optional[Dict[str, Any]] = None,
    blank_label: Optional[str] = None,
    blank_idx: Optional[int] = None,
) -> ScoreResultCollection:
    """
    Recog with ``model_recog_ctc_only`` to get N-best list on ``task.dev_dataset``,
    then calc scores with framewise prior and LM on N-best list,
    then tune optimal scales on N-best list,
    then rescore on all ``task.eval_datasets`` using those scales,
    and also do first-pass recog (``model_recog``) with those scales.
    """
    if blank_label is None:
        from ..ctc import model_recog as model_recog_ctc_only

        blank_label = model_recog_ctc_only.output_blank_label

    if blank_idx is None:
        from ..ctc import _ctc_model_def_blank_idx

        blank_idx = _ctc_model_def_blank_idx

    ctc_decoder_opts = ctc_decoder_opts.copy()
    if "lexicon" not in ctc_decoder_opts:
        assert lm_label_level == "word"  # not implemented yet otherwise...
        assert lm_word_list is not None, "lm_word_list must be given if no lexicon is given"
        ctc_decoder_opts["lexicon"] = get_lexicon_from_task(task, lm_word_list=lm_word_list)

    config = {
        "behavior_version": 24,  # should make it independent from batch size
        "__env_updates": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},  # OOM maybe otherwise
        "batch_size": int(20_000 * ctc_model.definition.batch_size_factor),
    }
    if extra_config:
        config = dict_update_deep(config, extra_config)

    if framewise_prior is NotSpecified:
        framewise_prior = get_ctc_prior(
            ctc_model=ctc_model,
            extra_config=extra_config,
            task=task,
            framewise_prior_dataset=framewise_prior_dataset or task.dev_dataset,
            blank_label=blank_label,
            blank_idx=blank_idx,
        )
        framewise_prior.file.creator.add_alias(f"{prefix}/framewise-prior")
        tk.register_output(f"{prefix}/framewise-prior.txt", framewise_prior.file)

    ctc_decoder_opts_no_lm = ctc_decoder_opts.copy()
    ctc_decoder_opts_no_lm["nbest"] = n_best_list_size
    ctc_decoder_opts_no_lm["beam_size"] = max(n_best_list_size, ctc_decoder_opts_no_lm.get("beam_size", 50))
    model_no_lm = get_ctc_with_ngram_lm_and_framewise_prior(
        ctc_model=ctc_model, lm_scale=0, ctc_decoder_opts=ctc_decoder_opts_no_lm
    )

    # see recog_model, lm_labelwise_prior_rescore
    dataset = task.dev_dataset
    asr_scores = search_dataset(
        dataset=dataset,
        model=model_no_lm,
        recog_def=model_recog_torchaudio,
        config=config,
        search_rqmt={"time": 3, "mem": 12},
        keep_alignment_frames=True,
        keep_beam=True,
    )
    prior_scores = prior_score(asr_scores, prior=framewise_prior)
    asr_scores = ctc_alignment_to_label_seq(asr_scores, blank_label=blank_label)
    prior_scores = ctc_alignment_to_label_seq(prior_scores, blank_label=blank_label)

    from i6_experiments.users.zeyer.datasets.utils.serialize import ReturnnDatasetToTextDictJob
    from i6_experiments.users.zeyer.datasets.task import RecogOutput

    ref = RecogOutput(
        output=ReturnnDatasetToTextDictJob(
            returnn_dataset=dataset.get_main_dataset(), data_key=dataset.get_default_target()
        ).out_txt
    )

    for f in task.recog_post_proc_funcs:  # BPE to words or so
        asr_scores = f(asr_scores)
        prior_scores = f(prior_scores)
        ref = f(ref)

    assert lm_label_level == "word"  # not implemented yet otherwise...
    # Compute LM scores on the N-best list on word-level.
    lm_scores = ngram_score_v2(asr_scores, lm=ngram_language_model)

    from i6_experiments.users.zeyer.decoding.scale_tuning import ScaleTuningJob

    opt_scales_job = ScaleTuningJob(
        scores={"am": asr_scores.output, "prior": prior_scores.output, "lm": lm_scores.output},
        ref=ref.output,
        fixed_scales={"am": 1.0},
        negative_scales={"prior"},
        scale_relative_to={"prior": "lm"},
        evaluation="edit_distance",
    )
    tk.register_output(f"{prefix}/opt-real-scales", opt_scales_job.out_real_scales)
    tk.register_output(f"{prefix}/opt-rel-scales", opt_scales_job.out_scales)
    # We use the real scales.
    # But prior is still handled as negative in lm_framewise_prior_rescore and 1stpass model_recog below.
    # (The DelayedBase logic on the Sis Variable should handle this.)
    prior_scale = opt_scales_job.out_real_scale_per_name["prior"] * (-1)
    lm_scale = opt_scales_job.out_real_scale_per_name["lm"]

    # Rescore with optimal scales. Like recog_model with lm_framewise_prior_rescore.
    res = recog_model(
        task=task,
        model=model_no_lm,
        recog_def=model_recog_torchaudio,
        config=config,
        search_rqmt={"time": 3, "mem": 12},
        recog_pre_post_proc_funcs_ext=[
            functools.partial(
                ngram_lm_framewise_prior_rescore,
                prior=framewise_prior,
                prior_scale=prior_scale,
                ngram_language_model=ngram_language_model,
                lm_label_level=lm_label_level,
                labels_to_words=task.recog_post_proc_funcs,
                lm_scale=lm_scale,
            )
        ],
    )
    tk.register_output(f"{prefix}/rescore-res.txt", res.output)

    model_with_lm = get_ctc_with_ngram_lm_and_framewise_prior(
        ctc_model=ctc_model,
        prior=framewise_prior.file,
        prior_type=framewise_prior.type,
        prior_scale=prior_scale,
        ngram_language_model=ngram_language_model,
        lm_scale=lm_scale,
        ctc_decoder_opts=ctc_decoder_opts,
    )
    res = recog_model(
        task=task,
        model=model_with_lm,
        recog_def=model_recog_torchaudio,
        config=config,
        search_rqmt={"time": 24, "mem": 32},
        name=f"{prefix}/recog-opt-1stpass",
    )
    tk.register_output(f"{prefix}/recog-1stpass-res.txt", res.output)
    return res


def get_lexicon_from_task(task: Task, *, lm_word_list: tk.Path) -> tk.Path:
    return ExtractLineBasedLexiconJob(vocab_opts=get_vocab_opts_from_task(task), word_list=lm_word_list).out_lexicon


def model_recog_torchaudio(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    This function is run within RETURNN.

    Use together with
    :func:`i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc_recog_ext.get_ctc_with_ngram_lm_and_framewise_prior`
    and :func:`recog_model` or :func:`search_model` or so.

    Earlier we used the generic beam_search function,
    but now we just directly perform the search here,
    as this is overall simpler and shorter.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    import torch
    from returnn.config import get_global_config
    from returnn.util import basic as util
    from torchaudio.models.decoder import CTCDecoder

    config = get_global_config()
    ctc_decoder_opts = config.typed_value("ctc_decoder_opts", None)
    assert isinstance(ctc_decoder_opts, dict)  # e.g. lexicon, nbest, beam_size, etc, see below
    n_best = config.int("n_best", ctc_decoder_opts.get("nbest", 1))

    # Eager-mode implementation of beam search using TorchAudio/Flashlight.

    dev_s = rf.get_default_device()
    dev = torch.device(dev_s)

    total_mem = None
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
        _, total_mem = torch.cuda.mem_get_info(dev if dev.index is not None else None)

    def _collect_mem_stats():
        if dev.type == "cuda":
            return [
                f"alloc cur {util.human_bytes_size(torch.cuda.memory_allocated(dev))}",
                f"alloc peak {util.human_bytes_size(torch.cuda.max_memory_allocated(dev))}",
                f"reserved cur {util.human_bytes_size(torch.cuda.memory_reserved(dev))}",
                f"reserved peak {util.human_bytes_size(torch.cuda.max_memory_reserved(dev))}",
            ]
        return ["(unknown)"]

    print(
        f"Memory usage {dev_s} before encoder forward:",
        " ".join(_collect_mem_stats()),
        "total:",
        util.human_bytes_size(total_mem) if total_mem else "(unknown)",
    )

    # https://github.com/pytorch/audio/blob/main/src/torchaudio/models/decoder/_ctc_decoder.py
    # https://docs.pytorch.org/audio/main/generated/torchaudio.models.decoder.ctc_decoder.html#torchaudio.models.decoder.ctc_decoder

    # noinspection PyTypeHints
    decoder: CTCDecoder = model.decoder

    label_log_prob, enc, enc_spatial_dim = model.encode_and_get_ctc_log_probs(data, in_spatial_dim=data_spatial_dim)
    assert label_log_prob.dims_set == {batch_dim, enc_spatial_dim, model.wb_target_dim}

    # The label log probs include the AM and the (scaled) prior.
    label_log_prob = rf.where(
        enc_spatial_dim.get_mask(),
        label_log_prob,
        rf.sparse_to_dense(model.blank_idx, axis=model.wb_target_dim, label_value=0.0, other_value=-1.0e30),
    )
    label_log_prob = label_log_prob.copy_transpose((batch_dim, enc_spatial_dim, model.wb_target_dim))
    batch_size, max_seq_len = label_log_prob.raw_tensor.shape[:2]
    assert enc_spatial_dim.dyn_size_ext.dims == (batch_dim,)

    label_log_prob = rf.cast(label_log_prob, "float32")
    label_log_prob = rf.copy_to_device(label_log_prob, "cpu")
    label_log_prob_raw = label_log_prob.raw_tensor.contiguous()
    float_bytes = 4

    print(f"Memory usage {dev_s} after encoder forward:", " ".join(_collect_mem_stats()))

    hyps = []
    scores = []
    for batch_idx in range(batch_size):
        emissions_ptr = label_log_prob_raw.data_ptr() + float_bytes * batch_idx * label_log_prob_raw.stride(0)
        seq_len = enc_spatial_dim.dyn_size[batch_idx]
        assert seq_len <= max_seq_len
        results = decoder.decoder.decode(emissions_ptr, seq_len, model.wb_target_dim.dimension)
        # I get -1 (silence label?) at the beginning and end in the tokens? Filter those away.
        # These are also additional frames which don't correspond to the input frames?
        # When removing those two frames, the len of tokens (align labels) matches the emission frames
        # (as it should be).
        assert all(all(result.tokens[i] in {-1, model.blank_idx} for i in [0, -1]) for result in results), [
            (result.tokens[0], result.tokens[-1]) for result in results
        ]
        hyps_per_batch = [result.tokens[1:-1] for result in results]
        scores_per_batch = [result.score for result in results]
        print(
            f"batch {batch_idx + 1}/{batch_size}: {len(results)} hyps,"
            f" best score: {scores_per_batch[0]},"
            f" best seq {_format_align_label_seq(results[0].tokens, model.wb_target_dim)},"
            f" worst score: {scores_per_batch[-1]},"
            f" mem usage {dev_s}: {' '.join(_collect_mem_stats())}"
        )
        assert all(len(hyp) == seq_len for hyp in hyps_per_batch), (
            f"seq_len {seq_len}, hyps lens {[len(hyp) for hyp in hyps_per_batch]}"
        )
        if len(results) >= n_best:
            hyps_per_batch = hyps_per_batch[:n_best]
            scores_per_batch = scores_per_batch[:n_best]
        else:
            hyps_per_batch += [[]] * (n_best - len(results))
            scores_per_batch += [-1e30] * (n_best - len(results))
        assert len(hyps_per_batch) == len(scores_per_batch) == n_best
        hyps_per_batch = [hyp + [model.blank_idx] * (max_seq_len - len(hyp)) for hyp in hyps_per_batch]
        assert all(len(hyp) == max_seq_len for hyp in hyps_per_batch)
        hyps.append(hyps_per_batch)
        scores.append(scores_per_batch)
    hyps_pt = torch.tensor(hyps, dtype=torch.int32)
    assert hyps_pt.shape == (batch_size, n_best, max_seq_len)
    scores_pt = torch.tensor(scores, dtype=torch.float32)
    assert scores_pt.shape == (batch_size, n_best)

    beam_dim = Dim(n_best, name="beam")
    out_spatial_dim = enc_spatial_dim
    hyps_r = rf.convert_to_tensor(hyps_pt, dims=(batch_dim, beam_dim, out_spatial_dim), sparse_dim=model.wb_target_dim)
    scores_r = rf.convert_to_tensor(scores_pt, dims=(batch_dim, beam_dim))
    print(f"Memory usage ({dev_s}) after batch:", " ".join(_collect_mem_stats()))
    return hyps_r, scores_r, out_spatial_dim, beam_dim


# RecogDef API
model_recog_torchaudio: RecogDef[Model]
model_recog_torchaudio.output_with_beam = True
model_recog_torchaudio.output_blank_label = "<blank>"
model_recog_torchaudio.batch_size_dependent = True  # our models currently just are batch-size-dependent...


def _format_align_label_seq(align_label_seq: List[int], wb_target_dim: Dim) -> str:
    seq_label: List[str] = []  # list of label
    seq_label_idx: List[int] = []  # list of label index
    seq_label_count: List[int] = []  # list of label count
    for align_label in align_label_seq:
        if seq_label_idx and seq_label_idx[-1] == align_label:
            seq_label_count[-1] += 1
        else:
            seq_label.append(wb_target_dim.vocab.id_to_label(align_label) if align_label >= 0 else str(align_label))
            seq_label_idx.append(align_label)
            seq_label_count.append(1)
    return " ".join(f"{label}*{count}" if count > 1 else label for label, count in zip(seq_label, seq_label_count))
