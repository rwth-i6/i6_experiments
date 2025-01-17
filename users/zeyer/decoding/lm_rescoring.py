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
from typing import TYPE_CHECKING, Optional, Callable

from sisyphus import tk
from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint, RescoreDef
from i6_experiments.users.zeyer.datasets.score_results import RecogOutput

from .rescoring import combine_scores, rescore
from .prior_rescoring import prior_score, Prior

if TYPE_CHECKING:
    from returnn.tensor import Tensor, Dim
    import returnn.frontend as rf
    from returnn.frontend.decoder.transformer import TransformerDecoder


def lm_framewise_prior_rescore(
    res: RecogOutput,
    *,
    raw_res_search_labels: RecogOutput,
    raw_res_labels: RecogOutput,
    orig_scale: float = 1.0,
    lm: ModelWithCheckpoint,
    lm_scale: float,
    vocab: tk.Path,
    vocab_opts_file: tk.Path,
    prior: Optional[Prior] = None,
    prior_scale: float = 0.0,
    search_labels_to_labels: Optional[Callable[[RecogOutput], RecogOutput]] = None,
) -> RecogOutput:
    """
    With functools.partial, you can use this for ``recog_post_proc_funcs`` in :func:`recog_model` and co.

    If you also want to combine a prior, e.g. for CTC, you might want to use :func:`prior_rescore` first.

    :param res:
        The format of the JSON is: {"<seq_tag>": [(score, "<text>"), ...], ...},
        i.e. the standard RETURNN search output with beam.
    :param raw_res_search_labels:
    :param raw_res_labels:
    :param orig_scale: scale for the original scores
    :param lm: language model
    :param lm_scale: scale for the LM scores
    :param vocab: for LM labels in res / raw_res_labels
    :param vocab_opts_file: for LM labels. contains info about EOS, BOS, etc
    :param prior:
    :param prior_scale: scale for the prior scores. this is used as the negative weight
    :param search_labels_to_labels: function to convert the search labels to the labels
    """
    res_labels_lm_scores = lm_score(raw_res_labels, lm=lm, vocab=vocab, vocab_opts_file=vocab_opts_file)
    scores = [(orig_scale, res), (lm_scale, res_labels_lm_scores)]
    if prior and prior_scale:
        assert search_labels_to_labels
        res_search_labels_prior_scores = prior_score(raw_res_search_labels, prior=prior)
        res_labels_prior_scores = search_labels_to_labels(res_search_labels_prior_scores)
        scores.append((-prior_scale, res_labels_prior_scores))
    else:
        assert prior_scale == 0.0
    return combine_scores(scores)


def lm_score(
    recog_output: RecogOutput, *, lm: ModelWithCheckpoint, vocab: tk.Path, vocab_opts_file: tk.Path
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
    """
    return rescore(
        recog_output=recog_output, model=lm, vocab=vocab, vocab_opts_file=vocab_opts_file, rescore_def=lm_rescore_def
    )


def lm_rescore_def(*, model: rf.Module, targets: Tensor, targets_beam_dim: Dim, targets_spatial_dim: Dim, **_other):
    import returnn.frontend as rf

    targets_beam_dim  # noqa  # unused here

    # noinspection PyTypeChecker
    model: TransformerDecoder
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
