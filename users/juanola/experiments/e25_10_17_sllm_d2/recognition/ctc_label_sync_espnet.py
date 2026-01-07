"""
Copied from i6_experiments/users/zeyer/experiments/exp2024_04_23_baselines/recog_ext/ctc_label_sync_espnet.py and
slightly adapted to work with
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Any, Sequence, Tuple, Dict
from functools import partial
from dataclasses import dataclass

import tree
import torch

from returnn.tensor import Tensor, Dim, batch_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray

from i6_experiments.users.zeyer.nn_rf.soft_collapse_repeated import soft_collapse_repeated
from i6_experiments.users.zeyer.nn_rf.top_k_and_random_choice_without_replacement import (
    top_k_and_random_choice_without_replacement,
)

from .beam_search import _gather_backrefs
from ..networks.interfaces.base_encoder_decoder_model import BaseEncoderDecoderModel

if TYPE_CHECKING:
    import torch
    from espnet.nets.scorer_interface import BatchScorerInterface


@dataclass
class DecoderConfig:
    # search related options:
    beam_size: int
    max_tokens_per_sec: int
    sample_rate: int
    ctc_soft_collapse_threshold: Optional[float] = None
    ctc_top_k_pruning: Optional[int] = None
    ctc_top_k_pruning_reduce_func: str = "mean"
    ctc_scale: float = 1.0
    prior_scale: float = 1.0
    lm_scale: float = 1.0

    def __str__(self) -> str:
        return f"beam-{self.beam_size}"


def ctc_label_sync_search_v1(
    *,
    model: BaseEncoderDecoderModel,
    data: torch.Tensor,
    data_seq_lens: torch.Tensor,
    beam_size: int,
    ctc_soft_collapse_threshold: Optional[float] = None,
    ctc_top_k_pruning: Optional[int] = None,
    ctc_top_k_pruning_reduce_func: str = "mean",
    ctc_scale: float = 1.0,
    prior_scale: float = 1.0,
    lm_scale: float = 1.0,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Code copied and adapted from
    :func:`i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.ctc.model_recog`.

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
    from returnn.config import get_global_config

    config = get_global_config()
    batch_dims = [batch_dim]
    ctc_beam_size = beam_size
    version = config.int("recog_version", 1)
    assert version == 1, f"invalid recog_version {version}"
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

    decoder_state, aux_logits, encoder_lens = model.forward_encoder(
        data,
        data_seq_lens,
        initial_beam_size=1,
    )
    ctc_log_prob = torch.nn.functional.log_softmax(aux_logits[-1], dim=-1)

    enc_spatial_dim = Dim(rf.convert_to_tensor(encoder_lens, dims=[batch_dim]), name="enc_spatial_dim")
    target_dim = Dim(model.num_labels, name="target_dim")
    wb_target_dim = Dim(model.num_labels + 1, name="wb_target_dim")  # Using num_labels + 1...

    if ctc_top_k_pruning is not None:
        reduce_func = getattr(torch, ctc_top_k_pruning_reduce_func)
        reduced_log_probs = reduce_func(ctc_log_prob[:, :, :-1], dim=1)
        if ctc_top_k_pruning_reduce_func in ("max", "min"):
            reduced_log_probs = reduced_log_probs[0]
        # get top k log probs for non-blank labels over reduced time frames
        _, pruned_indices = torch.topk(reduced_log_probs, k=ctc_top_k_pruning, dim=-1)
        # add EOS and blank to pruned indices
        pruned_indices = torch.cat(
            [
                pruned_indices,
                # EOS is needed for CTC prefix scoring
                torch.full((pruned_indices.size(0), 1), model.eos_idx, device=pruned_indices.device),
            ],
            dim=-1,
        )
        pruned_indices_wb = torch.cat(
            [
                pruned_indices,
                # EOS is needed for CTC prefix scoring
                torch.full((pruned_indices.size(0), 1), model.blank_idx, device=pruned_indices.device),
            ],
            dim=-1,
        )
        # gather selected log probs and re-normalize
        ctc_log_prob = torch.gather(
            ctc_log_prob, dim=-1, index=pruned_indices_wb.unsqueeze(1).expand(-1, ctc_log_prob.size(1), -1)
        )
        ctc_log_prob = torch.nn.functional.log_softmax(ctc_log_prob, dim=-1)
        pruned_eos_idx = pruned_indices.size(1) - 1  # last non-blank index
        pruned_bos_idx = pruned_eos_idx
        pruned_blank_idx = pruned_indices_wb.size(1) - 1  # last with blank idx
        pruned_wb_target_dim = Dim(pruned_indices_wb.size(1), name="pruned_wb_target_dim")
        wb_target_dim = pruned_wb_target_dim
        pruned_target_dim = Dim(pruned_indices.size(1), name="pruned_target_dim")
        pruned_indices_rf = rf.convert_to_tensor(
            pruned_indices, dims=[batch_dim, pruned_target_dim], sparse_dim=target_dim
        )
        ctc_log_prob = rf.convert_to_tensor(ctc_log_prob, dims=[batch_dim, enc_spatial_dim, pruned_wb_target_dim])
    else:
        ctc_log_prob = rf.convert_to_tensor(ctc_log_prob, dims=[batch_dim, enc_spatial_dim, wb_target_dim])

    # Eager-mode implementation of beam search.

    # The label log probs include the AM and the (scaled) prior.
    if ctc_soft_collapse_threshold is not None:
        ctc_log_prob, enc_spatial_dim = soft_collapse_repeated(
            ctc_log_prob,
            spatial_dim=enc_spatial_dim,
            classes_dim=wb_target_dim,
            threshold=ctc_soft_collapse_threshold,
            reduce_type=ctc_soft_collapse_reduce_type,
        )

    ctc_log_prob = rf.where(
        enc_spatial_dim.get_mask(),
        ctc_log_prob,
        rf.sparse_to_dense(model.blank_idx, axis=wb_target_dim, label_value=0.0, other_value=neg_inf),
    )

    ctc_beam_dim = Dim(1, name="ctc_initial_beam")
    ctc_prefix_scorer = CtcPrefixScorer(
        log_probs=ctc_log_prob,
        batch_dims=batch_dims,
        enc_spatial_dim=enc_spatial_dim,
        vocab_wb_dim=wb_target_dim,
        vocab_dim=target_dim if ctc_top_k_pruning is None else pruned_target_dim,
        blank_idx=model.blank_idx if ctc_top_k_pruning is None else pruned_blank_idx,
        eos_idx=model.eos_idx if ctc_top_k_pruning is None else pruned_eos_idx,
    )
    ctc_prefix_scorer_state = None
    ctc_seq_log_prob = rf.constant(0.0, dims=[ctc_beam_dim] + batch_dims)  # Batch, InBeam
    # differentiate between LM and CTC targets in case of pruning
    # in case of pruning, the lm target behaves as before but the ctc prefix scorer always gets the pruned indices
    target_lm = rf.constant(
        model.bos_idx, dims=[ctc_beam_dim] + batch_dims, sparse_dim=target_dim
    )  # Batch, InBeam -> Vocab
    target_ctc = rf.constant(
        model.bos_idx if ctc_top_k_pruning is None else pruned_bos_idx,
        dims=[ctc_beam_dim] + batch_dims,
        sparse_dim=target_dim if ctc_top_k_pruning is None else pruned_target_dim,
    )  # Batch, InBeam -> Vocab
    ended = rf.constant(False, dims=[ctc_beam_dim] + batch_dims)
    out_seq_len = rf.constant(0, dims=[ctc_beam_dim] + batch_dims)

    labelwise_prior: Optional[rf.Parameter] = getattr(model, "labelwise_prior", None)

    max_seq_len = enc_spatial_dim.get_size_tensor(device=data.device)

    lm_state_raw = decoder_state

    i = 0
    seq_targets = []
    seq_backrefs = []
    while True:
        label_log_prob, ctc_prefix_scorer_state = ctc_prefix_scorer.score_and_update_state(
            prev_label=target_ctc, prev_state=ctc_prefix_scorer_state, beam_dim=ctc_beam_dim
        )
        if ctc_top_k_pruning is not None:
            # scatter pruned log probs back to original vocab size with -inf for non-selected
            label_log_prob = rf.scatter(
                label_log_prob,
                fill_value=neg_inf,
                indices=pruned_indices_rf,
                indices_dim=pruned_target_dim,
                out_dim=target_dim,
                mode="max",
            )
        targets_lm_raw = target_lm.copy_compatible_to_dims_raw(batch_dims + [ctc_beam_dim])
        lm_logits_raw, lm_state_raw = model.step_decoder(targets_lm_raw.unsqueeze(-1), lm_state_raw)
        lm_logits_raw = lm_logits_raw.squeeze(-2)  # squeeze singleton time dim
        lm_logits = rf.convert_to_tensor(lm_logits_raw, dims=batch_dims + [ctc_beam_dim, target_dim])
        if ctc_top_k_pruning is not None:
            # gather selected lm logits
            lm_logits = rf.gather(
                lm_logits,
                indices=pruned_indices_rf,
                axis=target_dim,
            )
            # scatter back to original vocab size with -inf for non-selected
            lm_logits = rf.scatter(
                lm_logits,
                fill_value=neg_inf,
                indices=pruned_indices_rf,
                indices_dim=pruned_target_dim,
                out_dim=target_dim,
                mode="max",
            )
        lm_log_probs = rf.log_softmax(lm_logits, axis=target_dim)  # Batch, InBeam, Vocab
        lm_log_probs *= lm_scale
        label_log_prob += lm_log_probs  # Batch, InBeam, Vocab

        if labelwise_prior is not None:
            label_log_prob -= labelwise_prior  # prior scale already applied

        # Filter out finished beams
        label_log_prob = rf.where(
            ended,
            rf.sparse_to_dense(model.eos_idx, axis=target_dim, label_value=0.0, other_value=neg_inf),
            label_log_prob,
        )
        ctc_seq_log_prob = ctc_seq_log_prob + label_log_prob  # Batch, InBeam, Vocab

        if ctc_top_k_with_random_sampling:
            assert ctc_top_k_pruning is None, "not implemented for pruning case"
            ctc_seq_log_prob, (backrefs, target), ctc_beam_dim = top_k_and_random_choice_without_replacement(
                ctc_seq_log_prob,
                axis=[ctc_beam_dim, model.num_labels],
                k=Dim(ctc_beam_size, name=f"ctc_dec_step{i}_beam"),
                **ctc_top_k_with_random_sampling_opts,
            )  # ctc_seq_log_prob, backrefs, target: Batch, Beam. backrefs -> InBeam. target -> Vocab.
        else:
            k_dim = None
            if ctc_top_k_pruning is not None:
                _, (_, target_ctc), ctc_beam_dim_ = rf.top_k(
                    rf.gather(
                        ctc_seq_log_prob,
                        indices=pruned_indices_rf,
                        axis=target_dim,
                    ),
                    k_dim=Dim(ctc_beam_size, name=f"ctc_dec_step{i}_beam"),
                    axis=[ctc_beam_dim, pruned_target_dim],
                )  # ctc_seq_log_prob, backrefs, target: Batch, Beam. backrefs -> InBeam. target -> Vocab.
                k_dim = ctc_beam_dim_
            ctc_seq_log_prob, (backrefs, target_lm), ctc_beam_dim = rf.top_k(
                ctc_seq_log_prob,
                k_dim=Dim(ctc_beam_size, name=f"ctc_dec_step{i}_beam") if k_dim is None else k_dim,
                axis=[ctc_beam_dim, target_dim],
            )  # ctc_seq_log_prob, backrefs, target: Batch, Beam. backrefs -> InBeam. target -> Vocab.
            if ctc_top_k_pruning is None:
                target_ctc = target_lm

        target_lm = rf.cast(target_lm, dtype=rf.get_default_int_dtype())
        target_ctc = rf.cast(target_ctc, dtype=rf.get_default_int_dtype())
        seq_targets.append(target_lm)
        seq_backrefs.append(backrefs)
        ended = rf.gather(ended, indices=backrefs)
        out_seq_len = rf.gather(out_seq_len, indices=backrefs)
        ctc_prefix_scorer_state = rf.nested.gather_nested(ctc_prefix_scorer_state, indices=backrefs)

        lm_state_raw = tree.map_structure(
            partial(_gather_backrefs, backrefs=backrefs.raw_tensor, beam_size=beam_size), lm_state_raw
        )

        i += 1
        ended = rf.logical_or(ended, target_lm == model.eos_idx)
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


def _target_dense_remove_blank(target: Tensor, *, target_dim: Dim, wb_target_dim: Dim, blank_idx: int) -> Tensor:
    assert wb_target_dim in target.dims
    assert blank_idx == target_dim.dimension  # currently just not implemented otherwise
    res, _ = rf.slice(target, axis=wb_target_dim, size=target_dim)
    return res


def _target_extend_blank(target: Tensor, *, target_dim: Dim, wb_target_dim: Dim, blank_idx: int) -> Tensor:
    assert target.sparse_dim == target_dim
    assert blank_idx == target_dim.dimension  # currently just not implemented otherwise
    return rf.set_sparse_dim(target, wb_target_dim)


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
