"""
LM rescoring.

Given some hyps (JSON {seq_tag: [(score1, text1), (score2, text2), ...], ...}),
we use some new dataset in RETURNN which can read that directly,
and for one seq, the data looks like:
  tag: seq_tag
  data_flat: flatten(text1, text2, ...) (len: len(text1) + len(text2) + ...)
  data_seq_lens: [len(text1), len(text2), ...] (len: num hyps)
  scores: [score1, score2, ...] (len: num hyps)
  ...

Then use :class:`ReturnnForwardJobV2` and some custom forward function and the LM to score each hyp from this dataset,
and maybe directly write this out again as a new JSON file in the same format as the original hyps JSON.
The score would only be for the LM, not for the whole model.

Then we can have some job with combines multiple scores with some weights.
This job would combine the original scores with the LM scores.


Note: Another alternative would be:
Given some hyps (JSON {seq_tag: [(score1, text1), (score2, text2), ...], ...}),
we convert this into a standard RETURNN dataset in the format:
  seq_tag/1: "data" key: text1 (maybe also store "score" key: score1)
  seq_tag/2: "data" key: text2
  ...
I.e. individual seqs for each hyp.
However, this is problematic when you want to rescore with AED or CTC or so,
where there is still one shared input for multiple hyps,
and you ideally want to compute the encoder on the input only once.
Only for LM rescoring, this would not matter.
However, to keep it generic, we don't do this here.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union, Any, Literal, Callable, Dict, List

from sisyphus import tk
from sisyphus.delayed_ops import DelayedBase
from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint, ModelDefWithCfg, ModelDef, RescoreDef
from i6_experiments.users.zeyer.datasets.score_results import RecogOutput

from .rescoring import combine_scores, rescore
from .prior_rescoring import prior_score, Prior
from .concat_hyps import ReplaceHypsJob

if TYPE_CHECKING:
    from returnn_common.datasets_old_2022_10.interface import DatasetConfig
    from returnn.tensor import Tensor, Dim
    import returnn.frontend as rf
    from returnn.frontend.decoder.transformer import TransformerDecoder


def lm_framewise_prior_rescore(
    res: RecogOutput,
    *,
    dataset: DatasetConfig,
    raw_res_search_labels: RecogOutput,
    raw_res_labels: RecogOutput,
    orig_scale: Union[float, tk.Variable, DelayedBase] = 1.0,
    lm: ModelWithCheckpoint,
    lm_scale: Union[float, tk.Variable, DelayedBase],
    lm_rescore_rqmt: Optional[Dict[str, Any]] = None,
    vocab: tk.Path,
    vocab_opts_file: tk.Path,
    prior: Optional[Prior] = None,
    prior_scale: Union[float, tk.Variable, DelayedBase] = 0.0,
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
    :param lm: language model
    :param lm_scale: scale for the LM scores
    :param lm_rescore_rqmt:
    :param vocab: for LM labels in res / raw_res_labels
    :param vocab_opts_file: for LM labels. contains info about EOS, BOS, etc
    :param prior:
    :param prior_scale: scale for the prior scores. this is used as the negative weight
    :param search_labels_to_labels: function to convert the search labels to the labels
    """
    dataset  # noqa  # unused here
    res_labels_lm_scores = lm_score(
        raw_res_labels, lm=lm, vocab=vocab, vocab_opts_file=vocab_opts_file, rescore_rqmt=lm_rescore_rqmt
    )
    scores = [(orig_scale, res), (lm_scale, res_labels_lm_scores)]
    if prior and prior_scale:
        assert search_labels_to_labels
        res_search_labels_prior_scores = prior_score(raw_res_search_labels, prior=prior)
        res_labels_prior_scores = search_labels_to_labels(res_search_labels_prior_scores)
        scores.append((prior_scale * (-1), res_labels_prior_scores))
    else:
        assert isinstance(prior_scale, (int, float)) and prior_scale == 0.0
    return combine_scores(scores)


def lm_labelwise_prior_rescore(
    res: RecogOutput,
    *,
    dataset: DatasetConfig,
    raw_res_search_labels: RecogOutput,
    raw_res_labels: RecogOutput,
    orig_scale: Union[float, tk.Variable, DelayedBase] = 1.0,
    lm: ModelWithCheckpoint,
    lm_scale: Union[float, tk.Variable, DelayedBase],
    lm_rescore_config: Optional[Dict[str, Any]] = None,
    lm_rescore_rqmt: Optional[Dict[str, Any]] = None,
    vocab: tk.Path,
    vocab_opts_file: tk.Path,
    prior: Optional[Prior] = None,
    prior_scale: Union[float, tk.Variable, DelayedBase] = 0.0,
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
    :param lm: language model
    :param lm_scale: scale for the LM scores
    :param lm_rescore_config:
    :param lm_rescore_rqmt:
    :param vocab: for LM labels in res / raw_res_labels
    :param vocab_opts_file: for LM labels. contains info about EOS, BOS, etc
    :param prior:
    :param prior_scale: scale for the prior scores. this is used as the negative weight
    :param search_labels_to_labels: function to convert the search labels to the labels
    """
    dataset, raw_res_search_labels, search_labels_to_labels  # noqa  # unused here
    res_labels_lm_scores = lm_score(
        raw_res_labels,
        lm=lm,
        vocab=vocab,
        vocab_opts_file=vocab_opts_file,
        config=lm_rescore_config,
        rescore_rqmt=lm_rescore_rqmt,
    )
    scores = [(orig_scale, res), (lm_scale, res_labels_lm_scores)]
    if prior and prior_scale:
        res_labels_prior_scores = prior_score(raw_res_labels, prior=prior)
        scores.append((prior_scale * (-1), res_labels_prior_scores))
    else:
        assert isinstance(prior_scale, (int, float)) and prior_scale == 0.0
    return combine_scores(scores)


def lm_am_labelwise_prior_rescore(
    res: RecogOutput,
    *,
    dataset: DatasetConfig,
    raw_res_search_labels: RecogOutput,
    raw_res_labels: RecogOutput,
    am: ModelWithCheckpoint,
    am_rescore_def: RescoreDef,
    am_rescore_rqmt: Optional[Dict[str, Any]] = None,
    am_scale: Union[float, tk.Variable, DelayedBase] = 1.0,
    lm: ModelWithCheckpoint,
    lm_scale: Union[float, tk.Variable, DelayedBase],
    lm_rescore_rqmt: Optional[Dict[str, Any]] = None,
    vocab: tk.Path,
    vocab_opts_file: tk.Path,
    prior: Optional[Prior] = None,
    prior_scale: Union[float, tk.Variable, DelayedBase] = 0.0,
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
    :param am:
    :param am_rescore_def:
    :param am_rescore_rqmt:
    :param am_scale: scale for the new AM scores
    :param lm: language model
    :param lm_scale: scale for the LM scores
    :param lm_rescore_rqmt:
    :param vocab: for LM labels in res / raw_res_labels
    :param vocab_opts_file: for LM labels. contains info about EOS, BOS, etc
    :param prior:
    :param prior_scale: scale for the prior scores. this is used as the negative weight
    :param search_labels_to_labels: function to convert the search labels to the labels
    """
    res, raw_res_search_labels, search_labels_to_labels  # noqa  # unused here
    res_labels_lm_scores = lm_score(
        raw_res_labels, lm=lm, vocab=vocab, vocab_opts_file=vocab_opts_file, rescore_rqmt=lm_rescore_rqmt
    )
    res_labels_am_scores = rescore(
        recog_output=raw_res_labels,
        dataset=dataset,
        model=am,
        vocab=vocab,
        vocab_opts_file=vocab_opts_file,
        rescore_def=am_rescore_def,
        forward_rqmt=am_rescore_rqmt,
    )
    scores = [(am_scale, res_labels_am_scores), (lm_scale, res_labels_lm_scores)]
    if prior and prior_scale:
        res_labels_prior_scores = prior_score(raw_res_labels, prior=prior)
        scores.append((prior_scale * (-1), res_labels_prior_scores))
    else:
        assert isinstance(prior_scale, (int, float)) and prior_scale == 0.0
    return combine_scores(scores)


def lm_am_labelwise_prior_ngram_rescore(
    res: RecogOutput,
    *,
    dataset: DatasetConfig,
    raw_res_search_labels: RecogOutput,
    raw_res_labels: RecogOutput,
    am: ModelWithCheckpoint,
    am_rescore_def: RescoreDef,
    am_rescore_rqmt: Optional[Dict[str, Any]] = None,
    am_scale: Union[float, tk.Variable, DelayedBase] = 1.0,
    lm: ModelWithCheckpoint,
    lm_scale: Union[float, tk.Variable, DelayedBase],
    lm_rescore_rqmt: Optional[Dict[str, Any]] = None,
    vocab: tk.Path,
    vocab_opts_file: tk.Path,
    prior_ngram_lm: tk.Path,
    prior_scale: Union[float, tk.Variable, DelayedBase] = 0.0,
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
    :param am:
    :param am_rescore_def:
    :param am_rescore_rqmt:
    :param am_scale: scale for the new AM scores
    :param lm: language model
    :param lm_scale: scale for the LM scores
    :param lm_rescore_rqmt:
    :param vocab: for LM labels in res / raw_res_labels
    :param vocab_opts_file: for LM labels. contains info about EOS, BOS, etc
    :param prior_ngram_lm:
    :param prior_scale: scale for the prior scores. this is used as the negative weight
    :param search_labels_to_labels: function to convert the search labels to the labels
    """
    res, raw_res_search_labels, search_labels_to_labels  # noqa  # unused here
    res_labels_lm_scores = lm_score(
        raw_res_labels, lm=lm, vocab=vocab, vocab_opts_file=vocab_opts_file, rescore_rqmt=lm_rescore_rqmt
    )
    res_labels_am_scores = rescore(
        recog_output=raw_res_labels,
        dataset=dataset,
        model=am,
        vocab=vocab,
        vocab_opts_file=vocab_opts_file,
        rescore_def=am_rescore_def,
        forward_rqmt=am_rescore_rqmt,
    )
    scores = [(am_scale, res_labels_am_scores), (lm_scale, res_labels_lm_scores)]
    if prior_scale:
        res_labels_prior_scores = ngram_score(raw_res_labels, lm=prior_ngram_lm, vocab=vocab)
        scores.append((prior_scale * (-1), res_labels_prior_scores))
    else:
        assert isinstance(prior_scale, (int, float)) and prior_scale == 0.0
    return combine_scores(scores)


def ngram_lm_framewise_prior_rescore(
    res: RecogOutput,
    *,
    dataset: DatasetConfig,
    raw_res_search_labels: RecogOutput,
    raw_res_labels: RecogOutput,
    orig_scale: Union[float, tk.Variable, DelayedBase] = 1.0,
    ngram_language_model: tk.Path,
    lm_label_level: Literal["word", "task"] = "word",
    labels_to_words: Union[
        None, List[Callable[[RecogOutput], RecogOutput]], Callable[[RecogOutput], RecogOutput]
    ] = None,
    lm_scale: Union[float, tk.Variable, DelayedBase],
    prior: Optional[Prior] = None,
    prior_scale: Union[float, tk.Variable, DelayedBase] = 0.0,
    search_labels_to_labels: Optional[Callable[[RecogOutput], RecogOutput]] = None,
) -> RecogOutput:
    """
    With functools.partial, you can use this for ``recog_post_proc_funcs`` in :func:`recog_model` and co.

    :param res:
        The format of the JSON is: {"<seq_tag>": [(score, "<text>"), ...], ...},
        i.e. the standard RETURNN search output with beam.
    :param dataset: the orig data which was used to generate res
    :param raw_res_search_labels:
    :param raw_res_labels:
    :param orig_scale: scale for the original scores
    :param ngram_language_model: language model
    :param lm_label_level: "word" or "task"
    :param labels_to_words: function to convert the labels to words (only needed if lm_label_level=="word")
    :param lm_scale: scale for the LM scores
    :param prior:
    :param prior_scale: scale for the prior scores. this is used as the negative weight
    :param search_labels_to_labels: function to convert the search labels to the labels.
        (search labels are potentially with blanks, labels are without blanks;
         but labels are usually still on subword level)
    """
    dataset  # noqa  # unused here
    if lm_label_level == "word":
        assert labels_to_words is not None
        if isinstance(labels_to_words, list):
            lm_labels = raw_res_labels
            for f in labels_to_words:
                lm_labels = f(lm_labels)
        else:
            assert callable(labels_to_words)
            lm_labels = labels_to_words(raw_res_labels)
    elif lm_label_level == "task":
        lm_labels = raw_res_labels
    else:
        raise ValueError(f"invalid lm_label_level: {lm_label_level!r}")
    res_labels_lm_scores = ngram_score_v2(lm_labels, lm=ngram_language_model)
    if lm_label_level == "word":
        # Need to have consistent hyps for combine_scores
        res_labels_lm_scores = RecogOutput(
            output=ReplaceHypsJob(
                res_labels_lm_scores.output, new_hyps_py_output=raw_res_labels.output
            ).out_search_results
        )
    scores = [(orig_scale, res), (lm_scale, res_labels_lm_scores)]
    if prior and prior_scale:
        assert search_labels_to_labels
        res_search_labels_prior_scores = prior_score(raw_res_search_labels, prior=prior)
        # Need to have consistent hyps for combine_scores
        res_labels_prior_scores = search_labels_to_labels(res_search_labels_prior_scores)
        scores.append((prior_scale * (-1), res_labels_prior_scores))
    else:
        assert isinstance(prior_scale, (int, float)) and prior_scale == 0.0
    return combine_scores(scores)


def lm_score(
    recog_output: RecogOutput,
    *,
    lm: ModelWithCheckpoint,
    vocab: tk.Path,
    vocab_opts_file: tk.Path,
    config: Optional[Dict[str, Any]] = None,
    rescore_rqmt: Optional[Dict[str, Any]] = None,
) -> RecogOutput:
    """
    Scores the hyps with the LM.

    :param recog_output:
        The format of the JSON is: {"<seq_tag>": [(score, "<text>"), ...], ...},
        i.e. the standard RETURNN search output with beam.
        We ignore the scores here and just use the text of the hyps.
    :param lm: language model
    :param vocab: labels (line-based, maybe gzipped)
    :param vocab_opts_file: for LM labels. contains info about EOS, BOS, etc
    :param config: additional config
    :param rescore_rqmt:
    """
    return rescore(
        recog_output=recog_output,
        model=lm,
        vocab=vocab,
        vocab_opts_file=vocab_opts_file,
        rescore_def=lm_rescore_def,
        config=config,
        forward_rqmt=rescore_rqmt,
    )


def lm_rescore_def(*, model: rf.Module, targets: Tensor, targets_beam_dim: Dim, targets_spatial_dim: Dim, **_other):
    from returnn.tensor import Tensor, Dim
    import returnn.frontend as rf
    from returnn.config import get_global_config

    targets_beam_dim  # noqa  # unused here

    # noinspection PyTypeChecker
    model: TransformerDecoder

    returnn_config = get_global_config(return_empty_if_none=True)

    # This is supposed to follow the same API as used in
    # i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.ctc_delayed_fusion_v2.
    default_data_convert_labels_func = returnn_config.typed_dict.get("default_data_convert_labels_func")
    if default_data_convert_labels_func:
        new_lm_labels, new_lm_labels_spatial_dim, num_am_labels_converted = default_data_convert_labels_func(
            new_am_labels=targets,
            new_am_labels_spatial_dim=targets_spatial_dim,
            lm_target_dim=model.vocab_dim,
            last_am_frame=True,
        )
        assert (
            isinstance(new_lm_labels, Tensor)
            and isinstance(new_lm_labels_spatial_dim, Dim)
            and isinstance(num_am_labels_converted, Tensor)
        )
        assert new_lm_labels.sparse_dim == model.vocab_dim
        assert (num_am_labels_converted == targets_spatial_dim.get_size_tensor()).raw_tensor.all(), (
            f"expected all {targets_spatial_dim} {targets_spatial_dim.get_size_tensor().raw_tensor} AM labels"
            f" to be converted, got {num_am_labels_converted} {num_am_labels_converted.raw_tensor}"
            f" converted via {default_data_convert_labels_func}"
        )
        targets, targets_spatial_dim = new_lm_labels, new_lm_labels_spatial_dim

    assert targets.sparse_dim == model.vocab_dim

    vocab = model.vocab_dim.vocab
    assert vocab.bos_label_id is not None and vocab.eos_label_id is not None

    input_labels, (targets_w_eos_spatial_dim,) = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=vocab.bos_label_id
    )
    targets_w_eos, _ = rf.pad(
        targets,
        axes=[targets_spatial_dim],
        padding=[(0, 1)],
        value=vocab.eos_label_id,
        out_dims=[targets_w_eos_spatial_dim],
    )

    batch_dims = targets.remaining_dims(targets_spatial_dim)
    logits, _ = model(
        input_labels,
        spatial_dim=targets_w_eos_spatial_dim,
        encoder=None,
        state=model.default_initial_state(batch_dims=batch_dims),
    )

    log_prob = rf.log_softmax(logits, axis=model.vocab_dim)
    log_prob_targets = rf.gather(
        log_prob, indices=targets_w_eos, axis=model.vocab_dim
    )  # [batch,beam,targets_spatial_w_eos]
    log_prob_targets_seq = rf.reduce_sum(log_prob_targets, axis=targets_w_eos_spatial_dim)  # [batch,beam]
    assert log_prob_targets_seq.dims_set == set(batch_dims)
    return log_prob_targets_seq


lm_rescore_def: RescoreDef


def ngram_score(
    recog_output: RecogOutput,
    *,
    lm: tk.Path,
    vocab: tk.Path,
    rescore_rqmt: Optional[Dict[str, Any]] = None,
) -> RecogOutput:
    """
    Scores the hyps with the LM.

    :param recog_output:
        The format of the JSON is: {"<seq_tag>": [(score, "<text>"), ...], ...},
        i.e. the standard RETURNN search output with beam.
        We ignore the scores here and just use the text of the hyps.
    :param lm: language model
    :param vocab: labels (line-based, maybe gzipped)
    :param rescore_rqmt:
    """
    return rescore(
        recog_output=recog_output,
        model=ModelWithCheckpoint(
            definition=ModelDefWithCfg(model_def=ngram_model_def, config={"_lm_file": lm}), checkpoint=None
        ),
        vocab=vocab,
        rescore_def=ngram_rescore_def,
        forward_rqmt=rescore_rqmt,
        forward_device="cpu",
    )


def ngram_score_v2(
    recog_output: RecogOutput,
    *,
    lm: tk.Path,
    rescore_rqmt: Optional[Dict[str, Any]] = None,
) -> RecogOutput:
    """
    Scores the hyps with the LM.

    v2: use byte-based vocab internally, does not need a vocab file.
    We pass the raw text directly to KenLM anyway.
    The words are written out in the ARPA file (or LM binary).

    :param recog_output:
        The format of the JSON is: {"<seq_tag>": [(score, "<text>"), ...], ...},
        i.e. the standard RETURNN search output with beam.
        We ignore the scores here and just use the text of the hyps.
    :param lm: language model
    :param rescore_rqmt:
    """
    return rescore(
        recog_output=recog_output,
        model=ModelWithCheckpoint(
            definition=ModelDefWithCfg(model_def=ngram_model_def, config={"_lm_file": lm}), checkpoint=None
        ),
        vocab_opts={"class": "Utf8ByteTargets"},
        rescore_def=ngram_rescore_def,
        forward_rqmt=rescore_rqmt,
        forward_device="cpu",
    )


def ngram_model_def(**_other):
    import torch
    from returnn.config import get_global_config

    # pip install kenlm, or:
    # pip install git+https://github.com/kpu/kenlm.git
    # (note: make sure that CPATH is correct, does not point to some different Python version.
    # on RWTH HPC, `module load Python/3.12` or so will influence this)
    import kenlm

    config = get_global_config()

    class _NGramModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self._lm_file = config.typed_value("_lm_file")
            self.lm = kenlm.LanguageModel(self._lm_file)

    return _NGramModel()


ngram_model_def: ModelDef
ngram_model_def.behavior_version = 22
ngram_model_def.backend = "torch"
ngram_model_def.batch_size_factor = 1


def ngram_rescore_def(*, model: rf.Module, targets: Tensor, targets_beam_dim: Dim, targets_spatial_dim: Dim, **_other):
    import torch
    import kenlm
    import returnn.frontend as rf
    from returnn.tensor import batch_dim

    # noinspection PyUnresolvedReferences
    lm: kenlm.LanguageModel = model.lm
    vocab = targets.sparse_dim.vocab

    # https://github.com/kpu/kenlm/blob/master/python/example.py
    # https://github.com/kpu/kenlm/blob/master/python/kenlm.pyx

    assert targets.dims_set == {batch_dim, targets_beam_dim, targets_spatial_dim}
    targets = targets.copy_transpose((batch_dim, targets_beam_dim, targets_spatial_dim))

    res_raw = torch.zeros((batch_dim.get_dim_value(), targets_beam_dim.get_dim_value()))
    for i in range(batch_dim.get_dim_value()):
        targets_beam_size = targets_beam_dim.dyn_size_ext
        if batch_dim in targets_beam_size.dims:
            targets_beam_size = rf.gather(targets_beam_size, axis=batch_dim, indices=i)
        for j in range(targets_beam_size.raw_tensor.item()):
            seq_len = targets_spatial_dim.dyn_size_ext
            seq_len = rf.gather(seq_len, axis=targets_beam_dim, indices=j)
            seq_len = rf.gather(seq_len, axis=batch_dim, indices=i)
            assert seq_len.dims == ()
            targets_raw = targets.raw_tensor[i, j, : seq_len.raw_tensor]
            targets_str = vocab.get_seq_labels(targets_raw.numpy())
            res_raw[i, j] = lm.score(targets_str)

    # KenLM returns score in +log10 space.
    # We want to return in (natural) +log space.
    # 10 ** x = e ** (x * log(10))
    res_raw *= torch.log(torch.tensor(10.0))

    res = rf.convert_to_tensor(res_raw, dims=(batch_dim, targets_beam_dim))
    return res


ngram_rescore_def: RescoreDef
