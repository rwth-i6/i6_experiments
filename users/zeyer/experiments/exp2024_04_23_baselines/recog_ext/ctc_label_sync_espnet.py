"""
Using the ESPnet label-sync code for CTC + other model (neural LM, or AED, or whatever).
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Any, Sequence, Tuple, List, Dict

import tree

from returnn.tensor import Tensor, Dim, batch_dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder
from returnn.frontend.tensor_array import TensorArray

from i6_experiments.users.zeyer.model_interfaces import RecogDef
from i6_experiments.users.zeyer.nn_rf.soft_collapse_repeated import soft_collapse_repeated
from i6_experiments.users.zeyer.nn_rf.top_k_and_random_choice_without_replacement import (
    top_k_and_random_choice_without_replacement,
)

from ..ctc import Model, _batch_size_factor

if TYPE_CHECKING:
    import torch
    from espnet.nets.scorer_interface import BatchScorerInterface


def model_recog_espnet(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Function is run within RETURNN.

    Earlier we used the generic beam_search function,
    but now we just directly perform the search here,
    as this is overall simpler and shorter.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    import time
    import torch
    from returnn.config import get_global_config

    config = get_global_config()

    # noinspection PyUnresolvedReferences
    lm: TransformerDecoder = model.lm
    # noinspection PyUnresolvedReferences
    lm_scale: float = model.lm_scale

    dev_s = rf.get_default_device()
    dev = torch.device(dev_s)
    start_time = time.perf_counter_ns()

    assert data.dims_set == {batch_dim, data_spatial_dim, data.feature_dim}
    logits, _, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim)
    assert logits.dims_set == {batch_dim, enc_spatial_dim, model.wb_target_dim}
    assert enc_spatial_dim.dyn_size_ext.dims_set == {batch_dim}
    enc_olens = enc_spatial_dim.dyn_size

    # The label log probs include the AM and the (scaled) prior.
    label_log_prob = model.log_probs_wb_from_logits(logits)  # Batch, Spatial, VocabWB
    label_log_prob = rf.where(
        enc_spatial_dim.get_mask(),
        label_log_prob,
        rf.sparse_to_dense(model.blank_idx, axis=model.wb_target_dim, label_value=0.0, other_value=-1.0e30),
    )
    label_log_prob = label_log_prob.copy_transpose((batch_dim, enc_spatial_dim, model.wb_target_dim))
    batch_size, max_seq_len = label_log_prob.raw_tensor.shape[:2]
    assert enc_spatial_dim.dyn_size_ext.dims == (batch_dim,)

    # Adapted from i6_experiments.users.zeyer.experiments.exp2023_04_25_rf.espnet.model_recog

    beam_search_opts = (config.typed_value("beam_search_opts", None) or {}).copy()
    print("beam search opts:", beam_search_opts)

    beam_size = beam_search_opts.pop("beam_size", 12)  # like RETURNN, not 60 for now...
    # no ctc_weight/lm_weight here, CTC scale (incl prior) is already in the model, lm_scale above
    # ngram_weight = beam_search_opts.pop("ngram_weight", 0.9)  # not used currently...
    penalty = beam_search_opts.pop("length_reward", 0.0)
    normalize_length = beam_search_opts.pop("normalize_length", False)  # note: only at the end
    maxlenratio = beam_search_opts.pop("maxlenratio", 0.0)
    minlenratio = beam_search_opts.pop("minlenratio", 0.0)
    assert not beam_search_opts, f"found unused opts: {beam_search_opts}"

    # Partly taking code from espnet2.bin.asr_inference.Speech2Text.

    from espnet.nets.scorers.length_bonus import LengthBonus
    from espnet.nets.scorer_interface import BatchScorerInterface
    from espnet.nets.batch_beam_search import BatchBeamSearch
    from espnet.nets.beam_search import Hypothesis

    token_list = model.target_dim.vocab.labels
    scorers = {"ctc": _ctc_scorer(blank_idx=model.blank_idx, eos_idx=model.eos_idx), "lm": _lm_scorer(lm)}
    # CTC scaled internally
    weights = {"ctc": 1.0, "lm": lm_scale}
    if penalty != 0:
        scorers["length_bonus"] = LengthBonus(len(token_list))
        weights["length_bonus"] = penalty

    assert all(isinstance(v, BatchScorerInterface) for k, v in scorers.items()), f"non-batch scorers: {scorers}"

    beam_search = BatchBeamSearch(
        beam_size=beam_size,
        weights=weights,
        scorers=scorers,
        sos=model.bos_idx,
        eos=model.eos_idx,
        vocab_size=len(token_list),
        token_list=token_list,
        pre_beam_score_key="full",  # None if ctc_weight == 1.0 else "full",
        normalize_length=normalize_length,
    )

    if data.raw_tensor.device.type == "cuda":
        # Just so that timing of encoder is correct.
        torch.cuda.synchronize(data.raw_tensor.device)

    enc_end_time = time.perf_counter_ns()

    beam_dim = Dim(beam_size, name="beam")
    olens = torch.zeros([batch_size, beam_size], dtype=torch.int32)
    out_spatial_dim = Dim(Tensor("out_spatial", [batch_dim, beam_dim], "int32", raw_tensor=olens))
    outputs = [[] for _ in range(batch_size)]
    oscores = torch.zeros([batch_size, beam_size], dtype=torch.float32)
    seq_log_prob = Tensor("scores", [batch_dim, beam_dim], "float32", raw_tensor=oscores)

    # BatchBeamSearch is misleading: It still only operates on a single sequence,
    # but just handles all hypotheses in a batched way.
    # So we must iterate over all the sequences here from the input.
    for i in range(batch_size):
        nbest_hyps: List[Hypothesis] = beam_search(
            # The `x` is only used by our wrapped CTC.
            x=label_log_prob.raw_tensor[i, : enc_olens[i]],
            maxlenratio=maxlenratio,
            minlenratio=minlenratio,
        )
        print("best:", " ".join(token_list[v] for v in nbest_hyps[0].yseq))
        # I'm not exactly sure why, but sometimes we get even more hyps?
        # And then also sometimes, we get less hyps?
        very_bad_score = min(-1e32, nbest_hyps[-1].score - 1)  # not -inf because of serialization issues
        while len(nbest_hyps) < beam_size:
            nbest_hyps.append(Hypothesis(score=very_bad_score, yseq=torch.zeros(0, dtype=torch.int32)))
        for j in range(beam_size):
            hyp: Hypothesis = nbest_hyps[j]
            olens[i, j] = hyp.yseq.size(0)
            outputs[i].append(hyp.yseq)
            oscores[i, j] = hyp.score

    search_end_time = time.perf_counter_ns()
    data_seq_len_sum = rf.reduce_sum(data_spatial_dim.dyn_size_ext, axis=data_spatial_dim.dyn_size_ext.dims)
    data_seq_len_sum_secs = data_seq_len_sum.raw_tensor / _batch_size_factor / 100.0
    data_seq_len_max_seqs = data_spatial_dim.get_dim_value() / _batch_size_factor / 100.0
    out_len_longest_sum = rf.reduce_sum(rf.reduce_max(out_spatial_dim.dyn_size_ext, axis=beam_dim), axis=batch_dim)
    print(
        "TIMINGS:",
        ", ".join(
            (
                f"batch size {data.get_batch_dim_tag().get_dim_value()}",
                f"data len max {data_spatial_dim.get_dim_value()} ({data_seq_len_max_seqs:.2f} secs)",
                f"data len sum {data_seq_len_sum.raw_tensor} ({data_seq_len_sum_secs:.2f} secs)",
                f"enc {enc_end_time - start_time} ns",
                f"enc len max {torch.max(enc_olens)}",
                f"dec {search_end_time - enc_end_time} ns",
                f"out len max {out_spatial_dim.get_dim_value()}",
                f"out len longest sum {out_len_longest_sum.raw_tensor}",
            )
        ),
    )

    outputs_t = torch.zeros([batch_size, beam_size, torch.max(olens)], dtype=torch.int32)
    for i in range(batch_size):
        for j in range(beam_size):
            outputs_t[i, j, : olens[i, j]] = outputs[i][j]
    seq_targets = Tensor(
        "outputs", [batch_dim, beam_dim, out_spatial_dim], "int32", sparse_dim=model.target_dim, raw_tensor=outputs_t
    )

    return seq_targets, seq_log_prob, out_spatial_dim, beam_dim


# RecogDef API
model_recog_espnet: RecogDef[Model]
model_recog_espnet.output_with_beam = True
model_recog_espnet.output_blank_label = None
model_recog_espnet.batch_size_dependent = True  # our models currently just are batch-size-dependent...


def _ctc_scorer(*, blank_idx: int, eos_idx: int) -> BatchScorerInterface:
    import torch
    from espnet.nets.ctc_prefix_score import CTCPrefixScoreTH
    from espnet.nets.scorer_interface import BatchPartialScorerInterface

    # Copied and adapted from espnet.nets.scorers.ctc.CTCPrefixScorer.
    class CTCPrefixScorer(BatchPartialScorerInterface):
        """Decoder interface wrapper for CTCPrefixScore."""

        def __init__(self):
            self.impl = None

        def init_state(self, x: torch.Tensor):
            """..."""
            raise NotImplementedError("Assuming batched beam search")

        def select_state(self, state, i, new_id=None):
            """Select state with relative ids in the main beam search.

            Args:
                state: Decoder state for prefix tokens
                i (int): Index to select a state in the main beam search
                new_id (int): New label id to select a state if necessary

            Returns:
                state: pruned state
            """
            if type(state) is tuple:
                # for CTCPrefixScoreTH (need new_id > 0)
                r, log_psi, f_min, f_max, scoring_idmap = state
                s = log_psi[i, new_id].expand(log_psi.size(1))
                if scoring_idmap is not None:
                    return r[:, :, i, scoring_idmap[i, new_id]], s, f_min, f_max
                else:
                    return r[:, :, i, new_id], s, f_min, f_max
            return None if state is None else state[i]

        def score_partial(self, y, ids, state, x):
            """..."""
            raise NotImplementedError("Assuming batched beam search")

        def batch_init_state(self, x: torch.Tensor):
            """Get an initial state for decoding.

            Args:
                x (torch.Tensor): The encoded feature tensor, already the log probs, shape [enc_len, wb_target_dim]

            Returns: initial state
            """
            logp = x.unsqueeze(0)  # assuming batch_size = 1
            xlen = torch.tensor([logp.size(1)])
            self.impl = CTCPrefixScoreTH(logp, xlen, blank_idx, eos_idx)
            return None

        def batch_score_partial(self, y, ids, state, x):
            """Score new token.

            Args:
                y (torch.Tensor): 1D prefix token
                ids (torch.Tensor): torch.int64 next token to score
                state: decoder state for prefix tokens
                x (torch.Tensor): 2D encoder feature that generates ys

            Returns:
                tuple[torch.Tensor, Any]:
                    Tuple of a score tensor for y that has a shape `(len(next_tokens),)`
                    and next state for ys

            """
            batch_state = (
                (
                    torch.stack([s[0] for s in state], dim=2),
                    torch.stack([s[1] for s in state]),
                    state[0][2],
                    state[0][3],
                )
                if state[0] is not None
                else None
            )
            log_probs, new_batch_state = self.impl(y, batch_state, ids)
            # log_probs is shape [B,V+1], no matter what ids is.
            # Comment from espnet.nets.batch_beam_search.BatchBeamSearch.search:
            #   NOTE(takaaki-hori): Unlike BeamSearch, we assume that score_partial returns
            #   full-size score matrices, which has non-zero scores for part_ids and zeros
            #   for others.
            # However, we want to return only V scores, not V+1, excluding the blank
            # (because the LM (or AED) also don't have the blank in the vocab).
            assert blank_idx == log_probs.size(1) - 1  # only for this case implemented here currently...
            log_probs = log_probs[:, :-1]  # [B,V]
            return log_probs, new_batch_state

        def extend_prob(self, x: torch.Tensor):
            """..."""
            raise NotImplementedError("extend prob not supported")

        def extend_state(self, state):
            """..."""
            raise NotImplementedError("extend state not supported")

    return CTCPrefixScorer()


def _lm_scorer(lm: TransformerDecoder) -> BatchScorerInterface:
    """
    :param lm: the LM
    :return: wrapped LM
    """
    import torch
    from espnet.nets.scorer_interface import BatchScorerInterface

    class _LmScorer(BatchScorerInterface):
        def batch_score(self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor) -> Tuple[torch.Tensor, List[Any]]:
            """Score new token batch (required).

            Args:
                ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
                states (List[Any]): Scorer states for prefix tokens.
                xs (torch.Tensor):
                    The encoder feature that generates ys (n_batch, xlen, n_feat).

            Returns:
                tuple[torch.Tensor, List[Any]]: Tuple of
                    batchfied scores for next token with shape of `(n_batch, n_vocab)`
                    and next state list for ys.
            """
            batch_size = ys.shape[0]
            assert batch_size == len(states)
            batch_dim_ = Dim(batch_size, name="batch")
            if any(s is None for s in states):
                # This is the initial state.
                assert all(s is None for s in states)
                states_batch = lm.default_initial_state(batch_dims=[batch_dim_])
            else:

                def _batchify(*args):
                    assert len(args) == batch_size
                    if isinstance(args[0], Tensor):
                        x, _ = rf.stack(args, out_dim=batch_dim_)
                        return x
                    if isinstance(args[0], Dim):
                        assert all(args[0] == s for s in args)
                        return args[0]
                    raise TypeError(f"unexpected type: {type(args[0])}")

                states_batch = tree.map_structure(_batchify, *states)

            logits, new_state = lm(
                rf.convert_to_tensor(ys[:, -1], dims=[batch_dim_], sparse_dim=lm.vocab_dim),
                spatial_dim=single_step_dim,
                encoder=None,
                state=states_batch,
            )
            log_probs = rf.log_softmax(logits, axis=lm.vocab_dim)  # [B,V]
            log_probs = log_probs.copy_transpose((batch_dim_, lm.vocab_dim))

            new_state_ls = []  # list over batch entries
            for i in range(batch_size):

                def _map(x):
                    if isinstance(x, Tensor):
                        if batch_dim_ in x.dims:
                            return rf.gather(x, axis=batch_dim_, indices=i)
                        return x
                    if isinstance(x, Dim):
                        if x.size is not None:  # static
                            return x
                        assert x.dyn_size_ext is not None  # weird? dim should be known
                        assert batch_dim_ not in x.dyn_size_ext.dims  # not supported
                        return x
                    raise TypeError(f"unexpected type: {type(x)}")

                new_state_ls.append(tree.map_structure(_map, new_state))

            return log_probs.raw_tensor, new_state_ls

    return _LmScorer()


def model_recog_label_sync_v2(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Code copied and adapted from
    :func:`i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.ctc.model_recog`.

    Function is run within RETURNN.

    Note, for debugging, see :func:`model_recog_debug` below.

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
    from returnn.config import get_global_config

    config = get_global_config()
    version = config.int("recog_version", 1)
    assert version == 1, f"invalid recog_version {version}"
    ctc_beam_size = config.int("beam_size", 12)
    ctc_soft_collapse_threshold = config.typed_value("ctc_soft_collapse_threshold", None)
    ctc_soft_collapse_reduce_type = config.typed_value("ctc_soft_collapse_reduce_type", "logmeanexp")
    ctc_top_k_with_random_sampling = config.float(
        "ctc_top_k_with_random_sampling", 0.0
    )  # 0 disabled, 1 enabled. but a smooth transition is possible
    ctc_top_k_with_random_sampling_opts: Optional[Dict[str, Any]] = None
    if ctc_top_k_with_random_sampling:
        ctc_top_k_with_random_sampling_opts = {"max_noise_scale": ctc_top_k_with_random_sampling}
    ctc_top_p = config.typed_value("ctc_top_p", None)  # 1.0 picks all (no effect). e.g. use 0.9.
    if ctc_top_p is not None:
        assert ctc_top_k_with_random_sampling_opts is not None
        ctc_top_k_with_random_sampling_opts["top_p"] = ctc_top_p
    if config.typed_value("ctc_top_k_with_random_sampling_opts", None):
        ctc_top_k_with_random_sampling_opts.update(config.typed_value("ctc_top_k_with_random_sampling_opts", None))
    if ctc_top_k_with_random_sampling_opts:
        for k in ["top_p"]:
            v = ctc_top_k_with_random_sampling_opts.get(k, None)
            if v is not None:
                ctc_top_k_with_random_sampling_opts[k] = rf.convert_to_tensor(v, device=data.device)

    neg_inf = float("-inf")

    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    logits, enc_out, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim)
    print("Encoder seq lens:", enc_spatial_dim.get_size_tensor().raw_tensor)

    # Eager-mode implementation of beam search.

    # The label log probs include the AM and the (scaled) prior.
    ctc_log_prob = model.log_probs_wb_from_logits(logits)  # Batch, Spatial, VocabWB
    if ctc_soft_collapse_threshold is not None:
        ctc_log_prob, enc_spatial_dim = soft_collapse_repeated(
            ctc_log_prob,
            spatial_dim=enc_spatial_dim,
            classes_dim=model.wb_target_dim,
            threshold=ctc_soft_collapse_threshold,
            reduce_type=ctc_soft_collapse_reduce_type,
        )
    ctc_log_prob = rf.where(
        enc_spatial_dim.get_mask(),
        ctc_log_prob,
        rf.sparse_to_dense(model.blank_idx, axis=model.wb_target_dim, label_value=0.0, other_value=neg_inf),
    )

    ctc_beam_dim = Dim(1, name="ctc_initial_beam")
    ctc_prefix_scorer = CtcPrefixScorer(
        log_probs=ctc_log_prob,
        batch_dims=batch_dims,
        enc_spatial_dim=enc_spatial_dim,
        vocab_wb_dim=model.wb_target_dim,
        vocab_dim=model.target_dim,
        blank_idx=model.blank_idx,
        eos_idx=model.eos_idx,
    )
    ctc_prefix_scorer_state = None
    ctc_seq_log_prob = rf.constant(0.0, dims=[ctc_beam_dim] + batch_dims)  # Batch, InBeam
    target = rf.constant(
        model.bos_idx, dims=[ctc_beam_dim] + batch_dims, sparse_dim=model.target_dim
    )  # Batch, InBeam -> Vocab
    ended = rf.constant(False, dims=[ctc_beam_dim] + batch_dims)
    out_seq_len = rf.constant(0, dims=[ctc_beam_dim] + batch_dims)

    if getattr(model, "lm", None) is None:
        lm: Optional[TransformerDecoder] = None
        lm_scale: Optional[float] = None
        lm_state = None

    else:
        # We usually have TransformerDecoder, but any other type would also be ok when it has the same API.
        # noinspection PyUnresolvedReferences
        lm: TransformerDecoder = model.lm
        # noinspection PyUnresolvedReferences
        lm_scale: float = model.lm_scale
        lm_state = lm.default_initial_state(batch_dims=batch_dims)  # Batch, InBeam, ...

    labelwise_prior: Optional[rf.Parameter] = getattr(model, "labelwise_prior", None)

    max_seq_len = enc_spatial_dim.get_size_tensor(device=data.device)

    i = 0
    seq_targets = []
    seq_backrefs = []
    while True:
        label_log_prob, ctc_prefix_scorer_state = ctc_prefix_scorer.score_and_update_state(
            prev_label=target, prev_state=ctc_prefix_scorer_state, beam_dim=ctc_beam_dim
        )

        if lm is not None:
            lm_logits, lm_state = lm(
                target,
                spatial_dim=single_step_dim,
                state=lm_state,
            )  # Batch, InBeam, Vocab / ...
            lm_log_probs = rf.log_softmax(lm_logits, axis=model.target_dim)  # Batch, InBeam, Vocab
            lm_log_probs *= lm_scale
            label_log_prob += lm_log_probs  # Batch, InBeam, Vocab

        if labelwise_prior is not None:
            label_log_prob -= labelwise_prior  # prior scale already applied

        # Filter out finished beams
        label_log_prob = rf.where(
            ended,
            rf.sparse_to_dense(model.eos_idx, axis=model.target_dim, label_value=0.0, other_value=neg_inf),
            label_log_prob,
        )
        ctc_seq_log_prob = ctc_seq_log_prob + label_log_prob  # Batch, InBeam, Vocab

        if ctc_top_k_with_random_sampling:
            ctc_seq_log_prob, (backrefs, target), ctc_beam_dim = top_k_and_random_choice_without_replacement(
                ctc_seq_log_prob,
                axis=[ctc_beam_dim, model.target_dim],
                k=Dim(ctc_beam_size, name=f"ctc_dec_step{i}_beam"),
                **ctc_top_k_with_random_sampling_opts,
            )  # ctc_seq_log_prob, backrefs, target: Batch, Beam. backrefs -> InBeam. target -> Vocab.
        else:
            ctc_seq_log_prob, (backrefs, target), ctc_beam_dim = rf.top_k(
                ctc_seq_log_prob,
                k_dim=Dim(ctc_beam_size, name=f"ctc_dec_step{i}_beam"),
                axis=[ctc_beam_dim, model.target_dim],
            )  # ctc_seq_log_prob, backrefs, target: Batch, Beam. backrefs -> InBeam. target -> Vocab.

        target = rf.cast(target, dtype=rf.get_default_int_dtype())
        seq_targets.append(target)
        seq_backrefs.append(backrefs)
        ended = rf.gather(ended, indices=backrefs)
        out_seq_len = rf.gather(out_seq_len, indices=backrefs)
        ctc_prefix_scorer_state = rf.nested.gather_nested(ctc_prefix_scorer_state, indices=backrefs)
        if lm is not None:
            lm_state = rf.nested.gather_nested(lm_state, indices=backrefs)

        i += 1
        ended = rf.logical_or(ended, target == model.eos_idx)
        ended = rf.logical_or(ended, i >= max_seq_len)
        if bool(rf.reduce_all(ended, axis=ended.dims).raw_tensor):
            break
        out_seq_len = out_seq_len + rf.where(ended, 0, 1)

    # Backtrack via backrefs, resolve beams.
    seq_targets_ = []
    indices = rf.range_over_dim(ctc_beam_dim)  # FinalBeam -> FinalBeam
    for backrefs, target in zip(seq_backrefs[::-1], seq_targets[::-1]):
        # indices: FinalBeam -> Beam
        # backrefs: Beam -> PrevBeam
        seq_targets_.insert(0, rf.gather(target, indices=indices))
        indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam

    seq_targets__ = TensorArray(seq_targets_[0])
    for target in seq_targets_:
        seq_targets__ = seq_targets__.push_back(target)
    labels_spatial_dim = Dim(out_seq_len, name="ctc_labels_spatial")
    ctc_seq_targets = seq_targets__.stack(axis=labels_spatial_dim)
    # Remove the remaining EOS labels.
    ctc_seq_targets, _ = rf.slice(ctc_seq_targets, axis=labels_spatial_dim, size=labels_spatial_dim)

    return ctc_seq_targets, ctc_seq_log_prob, labels_spatial_dim, ctc_beam_dim


# RecogDef API
model_recog_label_sync_v2: RecogDef[Model]
model_recog_label_sync_v2.output_with_beam = True
model_recog_label_sync_v2.output_blank_label = None  # label-sync, so no blank label
# Note: With behavior version >=24, it should be batch-size-independent...
model_recog_label_sync_v2.batch_size_dependent = True


# Copied from denoising LM...
# original derived from i6_experiments.users.zeyer.decoding.beam_search_torch.interface.LabelScorerIntf
class CtcPrefixScorer:
    """ESPnet label scorer"""

    # original: get_initial_state
    def __init__(
        self,
        *,
        log_probs: Tensor,
        batch_dims: Sequence[Dim],
        enc_spatial_dim: Dim,
        vocab_dim: Dim,
        vocab_wb_dim: Dim,
        blank_idx: int,
        eos_idx: int,
    ):
        """
        :param log_probs: shape [Batch, Spatial, Vocab]
        :param batch_dims: batch dims
        :param enc_spatial_dim: spatial dim
        :param vocab_dim: vocab dim, without blank
        :param vocab_wb_dim: vocab dim. we expect that this includes both blank and EOS.
        :param blank_idx: blank
        :param eos_idx: EOS
        """
        # ESPnet espnet.nets.batch_beam_search.BatchBeamSearch.init_hyp (slightly simplified):
        #         return self.batchfy(
        #             [
        #                 Hypothesis(
        #                     score=0.0,
        #                     scores={k: 0.0 for k in self.scorers},
        #                     states={k: d.batch_init_state(x) for k, d in self.scorers.items()},
        #                     hs=[],
        #                     yseq=torch.tensor([self.sos], device=x.device),
        #                 )
        #             ]
        #         )
        from espnet.nets.ctc_prefix_score import CTCPrefixScoreTH

        assert log_probs.dims_set == set(batch_dims) | {enc_spatial_dim, vocab_wb_dim}
        assert enc_spatial_dim.dyn_size_ext.dims_set == set(batch_dims)
        self.batch_dims = list(batch_dims)
        self.enc_spatial_dim = enc_spatial_dim
        self.vocab_dim = vocab_dim
        self.vocab_wb_dim = vocab_wb_dim
        self.state01_dim = Dim(2, name="state01")
        self.blank_idx = blank_idx
        self.eos_idx = eos_idx

        if len(self.batch_dims) == 1:
            self.batch_dim = self.batch_dims[0]
            enc_seq_lens = enc_spatial_dim.dyn_size_ext
        else:
            log_probs, self.batch_dim = rf.merge_dims(log_probs, dims=self.batch_dims)
            enc_seq_lens, _ = rf.merge_dims(enc_spatial_dim.dyn_size_ext, dims=self.batch_dims, out_dim=self.batch_dim)

        # espnet.nets.scorers.ctc.CTCPrefixScorer.batch_init_state incorrectly assumes batch_size=1,
        # and is wrong otherwise, thus we don't use that here.
        # Instead, directly use CTCPrefixScoreTH.
        self._espnet_ctc_prefix_score_th = CTCPrefixScoreTH(
            log_probs.copy_compatible_to_dims_raw([self.batch_dim, enc_spatial_dim, vocab_wb_dim]),
            enc_seq_lens.copy_compatible_to_dims_raw([self.batch_dim]),
            blank_idx,
            eos_idx,
        )

    @staticmethod
    def initial_state():
        return None

    def score_and_update_state(self, *, prev_state: Any, prev_label: Tensor, beam_dim: Dim) -> Tuple[Tensor, Any]:
        """
        :param prev_state: state of the scorer (decoder). any nested structure.
            all tensors are expected to have shape {Batch..., Beam, ...}.
        :param prev_label: shape {Batch..., Beam} -> index in [0...Label-1].
            Use some dummy value for the first step (e.g. SOS).
        :param beam_dim: beam dim
        :return: (scores, state).
            scores: shape {Batch..., Beam, Label}, log-prob-like scores.
            state: all tensors are expected to have shape {Batch..., Beam, ...}.
        """
        import torch
        import tree

        assert prev_label.dims_set == set(self.batch_dims) | {beam_dim} and prev_label.sparse_dim == self.vocab_dim
        if len(self.batch_dims) != 1:
            prev_label, _ = rf.merge_dims(prev_label, dims=self.batch_dims, out_dim=self.batch_dim)
        prev_label = _target_extend_blank(
            prev_label, target_dim=self.vocab_dim, wb_target_dim=self.vocab_wb_dim, blank_idx=self.blank_idx
        )
        prev_label_raw = prev_label.copy_compatible_to_dims_raw([self.batch_dim, beam_dim])
        batch_size, beam_size = prev_label_raw.shape

        if prev_state is not None:

            def _map(x):
                if x is None:
                    return None
                if isinstance(x, Dim):
                    assert not x.dyn_size_ext  # not implemented...
                    return x
                assert isinstance(x, Tensor) and x.dims_set.issuperset(self.batch_dims)
                x, _ = rf.merge_dims(x, dims=self.batch_dims, out_dim=self.batch_dim)
                return x

            if len(self.batch_dims) != 1:
                prev_state = tree.map_structure(_map, prev_state)
            ys, out_spatial_dim, prev_state = prev_state
            ys, out_spatial_dim = rf.cum_concat_step(
                prev_label, prev_accum=ys, axis=out_spatial_dim
            )  # [batch,beam,out_len]
        else:
            out_spatial_dim = Dim(1, name="out_spatial")
            ys = rf.expand_dim(prev_label, out_spatial_dim)  # [batch,beam,out_len]
        assert ys.dims_set == {self.batch_dim, beam_dim, out_spatial_dim}
        ys_raw = ys.copy_compatible_to_dims_raw([self.batch_dim, beam_dim, out_spatial_dim])
        ys_raw_flat = ys_raw.flatten(0, 1)  # [batch*beam,out_len]

        # Convert all [batch,beam,...] tensors to [batch*beam,...].
        def _map(x):
            if x is None:
                return None
            assert isinstance(x, Tensor) and x.dims_set.issuperset((self.batch_dim, beam_dim))
            x_raw = x.copy_compatible_to_dims_raw(
                [self.batch_dim, beam_dim] + x.remaining_dims([self.batch_dim, beam_dim])
            )
            return x_raw.flatten(0, 1)

        prev_state_raw = tree.map_structure(_map, prev_state)

        # if isinstance(espnet_scorer, CTCPrefixScorer):
        # Unfortunately the CTCPrefixScorer breaks our assumption that the batch dim is the first dim.
        # Thus, we must permute the corresponding entries in the state.
        # Also, the initial state is None, so we need to cover this case as well.
        if prev_state_raw is not None:
            # 4-tuple. first has batch in dim=2, second has batch in dim=0, third and forth don't have batch?
            # n_bh = self.batch * n_hyps. snum = odim.
            # first: r: (self.input_length, 2, n_bh, snum) in func,
            #   then with select_state resulting in: (in_len, 2, batch * new_n_hyps)
            #   or: r_prev: (self.input_length, 2, self.batch * n_hyps)
            # second: log_psi: (n_bh, self.odim) in func,
            #   then with select_state resulting in: (batch * new_n_hyps, self.odim) ?
            # third/forth: f_min, f_max: scalars, no batch, only used anyway with att_w, can just set 0 and 1.
            # we even get a fifth as output: scoring_idmap: but not used.
            # So, only care about first, second.
            # Apply the select_state logic here, i.e. espnet.nets.scorers.ctc.CTCPrefixScorer.select_state.
            r, log_psi = prev_state_raw
            r: torch.Tensor  # [batch*beam,in_len,2,snum]
            r = _batch_gather_torch(r, indices=prev_label_raw.flatten(), index_dim=3)  # [batch*beam,in_len,2]
            r = r.permute(1, 2, 0)  # [in_len,2,batch*beam]
            log_psi: torch.Tensor  # [batch*beam,odim]
            log_psi = _batch_gather_torch(log_psi, indices=prev_label_raw.flatten())  # [batch*beam]
            log_psi = log_psi[:, None]  # [batch*beam,1]. must broadcast to [batch*beam,odim]
            prev_state_raw = (r, log_psi, 0, 1)

        # Inline espnet.nets.scorers.ctc.CTCPrefixScorer.batch_score_partial,
        # as we already have it batched.
        scores, states = self._espnet_ctc_prefix_score_th(ys_raw_flat, prev_state_raw)
        # scores: (n_bh, vocab)
        scores = scores.unflatten(0, (batch_size, beam_size))  # [batch,beam,vocab]
        scores_rf = rf.convert_to_tensor(scores, dims=[self.batch_dim, beam_dim, self.vocab_wb_dim])
        r, log_psi = states[:2]
        r: torch.Tensor  # [in_len,2,batch*beam,snum]
        r = r.permute(2, 0, 1, 3)  # [batch*beam,in_len,2,snum]
        r = r.unflatten(0, (batch_size, beam_size))  # [batch,beam,in_len,2,snum]
        r_rf = rf.convert_to_tensor(
            r, dims=[self.batch_dim, beam_dim, self.enc_spatial_dim, self.state01_dim, self.vocab_wb_dim]
        )
        # log_psi: (n_bh, odim)
        log_psi = log_psi.unflatten(0, (batch_size, beam_size))  # [batch,beam,odim]
        log_psi_rf = rf.convert_to_tensor(log_psi, dims=[self.batch_dim, beam_dim, self.vocab_wb_dim])

        scores_rf = _target_dense_remove_blank(
            scores_rf, target_dim=self.vocab_dim, wb_target_dim=self.vocab_wb_dim, blank_idx=self.blank_idx
        )

        if len(self.batch_dims) != 1:

            def _map(x):
                if x is None:
                    return None
                if isinstance(x, Dim):
                    assert not x.dyn_size_ext or self.batch_dim not in x.dyn_size_ext.dims  # not implemented...
                    return x
                assert isinstance(x, Tensor) and self.batch_dim in x.dims
                return rf.split_dims(x, axis=self.batch_dim, dims=self.batch_dims)

            scores_rf, (ys, out_spatial_dim, (r_rf, log_psi_rf)) = tree.map_structure(
                _map, (scores_rf, (ys, out_spatial_dim, (r_rf, log_psi_rf)))
            )
        return scores_rf, (ys, out_spatial_dim, (r_rf, log_psi_rf))


def _target_remove_blank(target: Tensor, *, target_dim: Dim, wb_target_dim: Dim, blank_idx: int) -> Tensor:
    assert target.sparse_dim == wb_target_dim
    assert blank_idx == target_dim.dimension  # currently just not implemented otherwise
    return rf.set_sparse_dim(target, target_dim)


def _target_dense_remove_blank(target: Tensor, *, target_dim: Dim, wb_target_dim: Dim, blank_idx: int) -> Tensor:
    assert wb_target_dim in target.dims
    assert blank_idx == target_dim.dimension  # currently just not implemented otherwise
    res, _ = rf.slice(target, axis=wb_target_dim, size=target_dim)
    return res


def _target_extend_blank(target: Tensor, *, target_dim: Dim, wb_target_dim: Dim, blank_idx: int) -> Tensor:
    assert target.sparse_dim == target_dim
    assert blank_idx == target_dim.dimension  # currently just not implemented otherwise
    return rf.set_sparse_dim(target, wb_target_dim)


def _target_dense_extend_blank(
    target: Tensor, *, target_dim: Dim, wb_target_dim: Dim, blank_idx: int, value: float
) -> Tensor:
    assert target_dim in target.dims
    assert blank_idx == target_dim.dimension  # currently just not implemented otherwise
    res, _ = rf.pad(target, axes=[target_dim], padding=[(0, 1)], out_dims=[wb_target_dim], value=value)
    return res


# noinspection PyShadowingNames
def _batch_gather_torch(
    values: torch.Tensor, *, indices: torch.Tensor, batch_dim: int = 0, index_dim: int = 1
) -> torch.Tensor:
    """
    :param values: shape [Batch,Indices,ValuesDims...], e.g. [Batch,InBeam,...]
    :param indices: shape [Batch,IndicesDims...] -> Indices, e.g. [Batch,OutBeam] -> InBeam
    :param batch_dim: in values. in indices, batch is assumed first.
    :param index_dim: in values. must be >batch_dim (not implemented otherwise).
        in indices, index dims are expected after batch.
    :return: shape [Batch,IndicesDims...,ValuesDims...], e.g. [Batch,OutBeam,...],
        if batch_dim=0 and index_dim=1.
        Batch and index dim stays at the same place, index dim is replaced by indices dims from indices.
    """
    import torch

    # Derived from returnn.torch.frontend._backend.TorchBackend.gather.
    # Case indices.dims_set.intersection(source.dims_set - {axis}).
    # We cannot use index_select in this case. Need to fallback to gather.
    assert indices.shape[0] == values.shape[batch_dim] and batch_dim < index_dim
    num_index_own_dims = indices.ndim - 1
    if num_index_own_dims == 1:
        indices_flat = indices  # good, [Batch,IndexDim]
    elif num_index_own_dims == 0:
        indices_flat = indices[:, None]  # [Batch,IndexDim=1]
    else:
        indices_flat = indices.flatten(1)  # [Batch,FlatIndexDim]
    indices_flat_bc = indices_flat.reshape(
        [
            indices_flat.shape[0] if i == batch_dim else (indices_flat.shape[1] if i == index_dim else 1)
            for i, d in enumerate(values.shape)
        ]
    )  # batch_dim=0, index_dim=1 -> [Batch,IndexDim,1s...].
    indices_flat_exp = indices_flat_bc.expand(
        [
            indices_flat.shape[0] if i == batch_dim else (indices_flat.shape[1] if i == index_dim else d)
            for i, d in enumerate(values.shape)
        ]
    )  # batch_dim=0, index_dim=1 -> [Batch,IndexDim,ValuesDims...]
    out = torch.gather(values, dim=index_dim, index=indices_flat_exp.type(torch.int64))
    if num_index_own_dims == 1:
        pass  # nothing to do
    elif num_index_own_dims == 0:
        out = out.squeeze(index_dim)
    else:
        out = out.unflatten(index_dim, indices.shape[1:])
    if batch_dim == 0 and index_dim == 1:
        assert out.shape == indices.shape + values.shape[2:]
    return out
