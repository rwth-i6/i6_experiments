__all__ = ["State", "beam_search_v1"]

from abc import abstractmethod
from functools import partial
from typing import Generic, List, Protocol, Sequence, Tuple, TypeVar, Union, Optional
from dataclasses import dataclass

import returnn.frontend as rf
from returnn.tensor import Dim as ReturnnDim, Tensor as ReturnnTensor, batch_dim
from returnn.frontend.tensor_array import TensorArray

from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.networks.interfaces.base_encoder_decoder_model import \
    BaseEncoderDecoderModel
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.aed_ctc import (
    _seq_label_history_init_state,
    _target_dense_extend_blank,
    _target_remove_blank,
    _seq_label_append,
    _same_seq_labels,
)
from i6_experiments.users.zeyer.nn_rf.soft_collapse_repeated import (
    soft_collapse_repeated,
)

import torch
from torch import Tensor
import torch.nn.functional as F
import tree

State = TypeVar("State")


@dataclass
class DecoderConfig:
    # search related options:
    beam_size: int
    max_tokens_per_sec: int
    sample_rate: int

    def __str__(self) -> str:
        return f"beam-{self.beam_size}"


@dataclass
class DecoderConfigV2:
    # search related options:
    beam_size: int
    max_tokens_per_sec: int
    sample_rate: int
    use_dec_aux_log_probs: bool = False

    def __str__(self) -> str:
        return f"beam-{self.beam_size}"


def _batch_gather(values: Tensor, *, indices: Tensor, batch_dim: int = 0, index_dim: int = 1) -> Tensor:
    """
    torch.gather, but with support for an additional leading batch dim.

    :param values: shape [Batch,Indices,ValuesDims...], e.g. [Batch,InBeam,...]
    :param indices: shape [Batch,IndicesDims...] -> Indices, e.g. [Batch,OutBeam] -> InBeam
    :param batch_dim: in values. in indices, batch is assumed first.
    :param index_dim: in values. must be >batch_dim (not implemented otherwise).
        in indices, index dims are expected after batch.
    :return: shape [Batch,IndicesDims...,ValuesDims...], e.g. [Batch,OutBeam,...],
        if batch_dim=0 and index_dim=1.
        Batch and index dim stays at the same place, index dim is replaced by indices dims from indices.
    """
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


_T = TypeVar("T")


def _gather_backrefs(state: _T, *, backrefs: Tensor, beam_size: int) -> _T:
    """
    Specialized gather for backrefs, respecting broadcasted data in the decoder state.

    :param state: recurrent decoder state
    :param backrefs: backref indices
    :param beam_size: beam size for asserting state shapes
    """

    if not isinstance(state, Tensor) or state.ndim == 0:
        return state
    assert state.ndim >= 2, "need at least batch and beam dims"
    assert state.shape[1] == 1 or state.shape[1] == beam_size, (
        f"Beam dim must either be 1 or beam_size ({beam_size}) but is {state.shape[1]}"
    )
    if state.shape[1] == 1:
        return state  # broadcast, e.g. encoder state in [B,1,T,F]
    assert backrefs.ndim == 2, "need at least batch and beam dims in backrefs"
    return _batch_gather(state, indices=backrefs, index_dim=1)


def _top_k_nd(source: Tensor, *, k: int, dim: Sequence[int], sorted: bool = True) -> Tuple[Tensor, List[Tensor]]:
    """
    torch.top_k, but with support for search over multiple dimensions.

    :param source: [Batch...,SourceDims...]
    :param k: how many
    :param dim: axes of SourceDims, multiple dims to search in
    :param sorted: sorted output, see :func:`torch.topk`
    :return: values [Batch...,k], list of indices per dim, each [Batch...,k]
    """
    # Derived from returnn.torch.frontend._backend.TorchBackend.top_k.
    # Move axis to the end, in the right order.
    dim = [(d + source.ndim) % source.ndim for d in dim]
    source = source.permute([d for d in range(source.ndim) if d not in dim] + list(dim))
    source_flat = source.flatten(start_dim=source.ndim - len(dim))
    values, indices = torch.topk(source_flat, k=k, dim=-1, largest=True, sorted=sorted)
    indices_out = []
    for i in reversed(list(range(len(dim)))):
        a_dim = source.shape[source.ndim - len(dim) + i]
        indices_out_ = indices % a_dim
        indices = indices // a_dim
        indices_out.insert(0, indices_out_)
    return values, indices_out


def _get_collapsed_out_seqs(
        seq_targets: Tensor,
        blank_idx: int,
):
    """

    :param seq_targets: (B, beam, Time)
    :return:
    """
    ctc_batch_indices = []
    ctc_output_lens = []
    max_ctc_output_len = 0
    B, beam_size = seq_targets.shape[:2]  # noqa
    for b in range(B):
        ctc_beam_indices = []
        ctc_beam_lens = []
        max_ctc_output_len_beam = 0
        for beam in range(beam_size):
            seq_wo_reps = torch.unique_consecutive(seq_targets[b, beam], dim=0)
            seq_wo_reps_wo_blank = seq_wo_reps[seq_wo_reps != blank_idx]
            ctc_beam_indices.append(seq_wo_reps_wo_blank)
            ctc_beam_lens.append(len(seq_wo_reps_wo_blank))
            max_ctc_output_len_beam = max(max_ctc_output_len_beam, ctc_beam_lens[-1])
        max_ctc_output_len = max(max_ctc_output_len, max_ctc_output_len_beam)

        ctc_beam_indices = [F.pad(bi, (0, max_ctc_output_len_beam - bi.size(0)), value=0) for bi in ctc_beam_indices]

        ctc_batch_indices.append(torch.stack(ctc_beam_indices, dim=0))
        ctc_output_lens.append(torch.LongTensor(ctc_beam_lens))
    ctc_batch_indices = [F.pad(bi, (0, max_ctc_output_len - bi.size(1)), value=0) for bi in ctc_batch_indices]

    ctc_batch_indices = torch.stack(ctc_batch_indices, dim=0)
    ctc_output_lens = torch.stack(ctc_output_lens, dim=0)

    return ctc_batch_indices, ctc_output_lens


# @torch.no_grad
def beam_search_v1(
        *,
        model: BaseEncoderDecoderModel,
        raw_audio: Tensor,
        raw_audio_lens: Tensor,
        beam_size: int,
        batch_size: int,
        use_dec_aux_log_probs: bool = False,
        device: torch.device,
        ctc_soft_collapse_threshold: Optional[float] = None,
        ctc_soft_collapse_reduce_type: str = "logmeanexp",
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Eager-mode implementation of beam search.

    :param model: interface to the decoder model
    :param beam_size: the beam size, 16 is a good value
    :param batch_size: the batch size
    :param decoder_state: initial recurrent decoder state
    :param device: the device the computation happens on
    :param length_norm_exponent: scaling exponent for the length normalization factor
    :param max_seq_len: how long the decoded seqs can be at max, use e.g. encoder time dim.
        Shape: [B,]
    """
    with torch.no_grad():
        assert beam_size > 0
        assert batch_size > 0

        beam_dim = ReturnnDim(1, name="initial-beam")
        seq_log_prob = rf.constant(0.0, dims=[batch_dim, beam_dim])  # Batch, Beam

        (
            llm_audio_features_in,
            aux_log_probs,
            adapter_output_lengths,
            _,
            encoder_output_lens,
        ) = model.forward(raw_audio, raw_audio_lens)
        if use_dec_aux_log_probs:
            _, dec_aux_log_probs = model.decode_seq(
                x=torch.empty((raw_audio.size(0), 0), dtype=torch.long, device=raw_audio.device),
                x_lens=torch.zeros((raw_audio.size(0),), dtype=torch.long, device=raw_audio.device),
                audio_features=llm_audio_features_in,
                audio_features_lens=adapter_output_lengths,
            )
            label_log_prob = dec_aux_log_probs[-1]  # Batch, Time, Vocab
            encoder_output_lens = adapter_output_lengths
        else:
            # hard code last aux logit for now. make adjustable later if needed.
            label_log_prob = aux_log_probs[-1]  # Batch, Time, Vocab

        enc_spatial_dim = ReturnnDim(
            rf.convert_to_tensor(encoder_output_lens, dims=[batch_dim]),
            name="enc-spatial",
        )
        vocab_dim = ReturnnDim(model.wb_target_dim, name="vocab")
        label_log_prob = rf.convert_to_tensor(label_log_prob, dims=[batch_dim, enc_spatial_dim, vocab_dim])

        if ctc_soft_collapse_threshold is not None:
            label_log_prob, enc_spatial_dim = soft_collapse_repeated(
                label_log_prob,
                spatial_dim=enc_spatial_dim,
                classes_dim=vocab_dim,
                threshold=ctc_soft_collapse_threshold,
                reduce_type=ctc_soft_collapse_reduce_type,
            )

        label_log_prob = rf.where(
            enc_spatial_dim.get_mask(),
            label_log_prob,
            rf.sparse_to_dense(model.blank_idx, axis=vocab_dim, label_value=0.0, other_value=-1.0e30),
        )
        label_log_prob_pre_filter, (backrefs_pre_filter,), pre_filter_beam_dim = rf.top_k(
            label_log_prob,
            k_dim=ReturnnDim(min(beam_size, model.wb_target_dim), name="pre-filter-beam"),
            axis=[vocab_dim],
        )  # seq_log_prob, backrefs_global: Batch, Spatial, PreFilterBeam. backrefs_pre_filter -> Vocab
        label_log_prob_pre_filter_ta = TensorArray.unstack(
            label_log_prob_pre_filter, axis=enc_spatial_dim
        )  # t -> Batch, PreFilterBeam
        backrefs_pre_filter_ta = TensorArray.unstack(
            backrefs_pre_filter, axis=enc_spatial_dim
        )  # t -> Batch, PreFilterBeam

        max_seq_len = int(enc_spatial_dim.get_dim_value())
        seq_targets = []
        seq_backrefs = []
        for t in range(max_seq_len):
            # Filter out finished beams
            seq_log_prob = seq_log_prob + label_log_prob_pre_filter_ta[t]  # Batch, InBeam, PreFilterBeam
            seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
                seq_log_prob,
                k_dim=ReturnnDim(
                    min(beam_size, beam_dim.dimension * pre_filter_beam_dim.dimension),
                    name=f"dec-step{t}-beam",
                ),
                axis=[beam_dim, pre_filter_beam_dim],
            )  # seq_log_prob, backrefs, target: Batch, Beam. backrefs -> InBeam. target -> PreFilterBeam.
            target = rf.gather(backrefs_pre_filter_ta[t], indices=target)  # Batch, Beam -> Vocab
            seq_targets.append(target)
            seq_backrefs.append(backrefs)

        # Backtrack via backrefs, resolve beams.
        seq_targets_ = []
        indices = rf.range_over_dim(beam_dim)  # FinalBeam -> FinalBeam
        for backrefs, target in zip(seq_backrefs[::-1], seq_targets[::-1]):
            # indices: FinalBeam -> Beam
            # backrefs: Beam -> PrevBeam
            seq_targets_.insert(0, rf.gather(target, indices=indices))
            indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam

        seq_targets__ = TensorArray(seq_targets_[0])
        for target in seq_targets_:
            seq_targets__ = seq_targets__.push_back(target)
        out_spatial_dim = enc_spatial_dim
        seq_targets = seq_targets__.stack(axis=out_spatial_dim)

        seq_targets = seq_targets.copy_transpose([batch_dim, beam_dim, out_spatial_dim])
        seq_log_prob = seq_log_prob.copy_transpose([batch_dim, beam_dim])

    ctc_batch_indices, ctc_output_lens = _get_collapsed_out_seqs(
        seq_targets=seq_targets.raw_tensor,
        blank_idx=model.blank_idx,
    )

    return ctc_batch_indices, seq_log_prob.raw_tensor, ctc_output_lens


def beam_search_with_recomb_v1(
        *,
        model: BaseEncoderDecoderModel,
        raw_audio: Tensor,
        raw_audio_lens: Tensor,
        beam_size: int,
        batch_size: int,
        device: torch.device,
        original_blank_idx: Optional[int] = None,
        ctc_soft_collapse_threshold: Optional[float] = None,
        ctc_soft_collapse_reduce_type: str = "logmeanexp",
        ctc_top_k_pruning: Optional[int] = None,
        ctc_top_k_pruning_reduce_func: str = "mean",
        use_dec_aux_log_probs: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Copied and modified from i6_experiments/users/zeyer/experiments/exp2024_04_23_baselines/recog_ext/aed_ctc.py
    -> model_recog_with_recomb
    """
    import returnn
    from returnn.config import get_global_config

    llm_audio_features_in, aux_log_probs, adapter_output_lengths, _, _ = model.forward_v2(
        raw_audio.float(), raw_audio_lens
    )

    if use_dec_aux_log_probs:
        _, dec_aux_log_probs = model.decode_seq(
            x=torch.empty((raw_audio.size(0), 0), dtype=torch.long, device=raw_audio.device),
            x_lens=torch.zeros((raw_audio.size(0),), dtype=torch.long, device=raw_audio.device),
            audio_features=llm_audio_features_in,
            audio_features_lens=adapter_output_lengths,
        )
        ctc_label_log_prob = dec_aux_log_probs[-1]  # Batch, Time, Vocab
        ctc_label_log_prob_lens = adapter_output_lengths
    else:
        # aux_log_probs is a list of (logits, lens) tuples, one for each auxiliary CTC loss. We use the last one here.
        ctc_label_log_prob_lens = aux_log_probs[-1][1]
        # hard code last aux logit for now. make adjustable later if needed.
        ctc_label_log_prob = aux_log_probs[-1][0]

    if original_blank_idx is not None:
        assert model.blank_idx == ctc_label_log_prob.size(-1) - 1, (
            f"Expected blank_idx {model.blank_idx} to be the last index in the vocab"
        )
        assert original_blank_idx == 0, f"Expected original_blank_idx to be 0, got {original_blank_idx}. "
        # move blank log prob to the last index, as expected by the rest of the code. This is needed if the model was trained with a different blank_idx.
        blank_log_prob = ctc_label_log_prob[..., original_blank_idx]
        ctc_label_log_prob = torch.cat(
            [
                ctc_label_log_prob[..., :original_blank_idx],
                ctc_label_log_prob[..., original_blank_idx + 1 :],
                blank_log_prob.unsqueeze(-1),
            ],
            dim=-1,
        )

    config = get_global_config()
    recomb = config.typed_value("recog_recomb", "max")  # None, "max", "sum"

    # RETURNN version is like "1.20250115.110555"
    # There was an important fix in 2025-01-17 affecting masked_scatter.
    # And another important fix in 2025-01-24 affecting masked_scatter for old PyTorch versions.
    assert tuple(int(n) for n in returnn.__version__.split(".")) >= (1, 20250125, 0), returnn.__version__

    batch_dims = [batch_dim]
    enc_spatial_dim = ReturnnDim(
        rf.convert_to_tensor(ctc_label_log_prob_lens, dims=[batch_dim]),
        name="enc-spatial",
    )
    vocab_wb_dim = ReturnnDim(model.wb_target_dim, name="vocab")
    vocab_dim = ReturnnDim(model.wb_target_dim - 1, name="vocab")

    if ctc_top_k_pruning is not None:
        reduce_func = getattr(torch, ctc_top_k_pruning_reduce_func)
        # assumes that blank is the last index in the vocab
        reduced_log_probs = reduce_func(ctc_label_log_prob[:, :, :-1], dim=1)
        if ctc_top_k_pruning_reduce_func in ("max", "min"):
            reduced_log_probs = reduced_log_probs[0]
        # get top k log probs for non-blank labels over reduced time frames
        _, pruned_indices = torch.topk(reduced_log_probs, k=ctc_top_k_pruning, dim=-1)
        # add blank to pruned indices
        pruned_indices_wb = torch.cat(
            [
                pruned_indices,
                # EOS is needed for CTC prefix scoring
                torch.full(
                    (pruned_indices.size(0), 1),
                    model.blank_idx,
                    device=pruned_indices.device,
                ),
            ],
            dim=-1,
        )
        # gather selected log probs and re-normalize
        ctc_log_prob = torch.gather(
            ctc_label_log_prob,
            dim=-1,
            index=pruned_indices_wb.unsqueeze(1).expand(-1, ctc_label_log_prob.size(1), -1),
        )
        ctc_log_prob = torch.nn.functional.log_softmax(ctc_log_prob, dim=-1)
        pruned_wb_target_dim = ReturnnDim(pruned_indices_wb.size(1), name="pruned_wb_target_dim")
        pruned_indices_wb_rf = rf.convert_to_tensor(
            pruned_indices_wb,
            dims=[batch_dim, pruned_wb_target_dim],
            sparse_dim=vocab_wb_dim,
        )
        ctc_log_prob = rf.convert_to_tensor(ctc_log_prob, dims=[batch_dim, enc_spatial_dim, pruned_wb_target_dim])
        # scatter pruned log probs back to original vocab size with -inf for non-selected
        ctc_label_log_prob = rf.scatter(
            ctc_log_prob,
            fill_value=float("-inf"),
            indices=pruned_indices_wb_rf,
            indices_dim=pruned_wb_target_dim,
            out_dim=vocab_wb_dim,
            mode="max",
        )
        # ctc_log_prob = ctc_log_prob.copy_transpose((batch_dim, enc_spatial_dim, wb_target_dim)).raw_tensor
    else:
        ctc_label_log_prob = rf.convert_to_tensor(ctc_label_log_prob, dims=[batch_dim, enc_spatial_dim, vocab_wb_dim])

    if ctc_soft_collapse_threshold is not None:
        ctc_label_log_prob, enc_spatial_dim = soft_collapse_repeated(
            ctc_label_log_prob,
            spatial_dim=enc_spatial_dim,
            classes_dim=vocab_wb_dim,
            threshold=ctc_soft_collapse_threshold,
            reduce_type=ctc_soft_collapse_reduce_type,
        )

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = ReturnnDim(1, name="initial-beam")
    batch_dims_ = [beam_dim] + batch_dims
    neg_inf = float("-inf")
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)  # Batch, Beam

    ctc_label_log_prob = rf.where(
        enc_spatial_dim.get_mask(),
        ctc_label_log_prob,
        rf.sparse_to_dense(model.blank_idx, axis=vocab_wb_dim, label_value=0.0, other_value=neg_inf),
    )
    # No CTC scale needed.
    ctc_label_log_prob_ta = TensorArray.unstack(ctc_label_log_prob, axis=enc_spatial_dim)  # t -> Batch, VocabWB

    target = rf.constant(model.bos_idx, dims=batch_dims_, sparse_dim=vocab_dim)  # Batch, InBeam -> Vocab
    target_wb = rf.constant(model.blank_idx, dims=batch_dims_, sparse_dim=vocab_wb_dim)  # Batch, InBeam -> VocabWB

    seq_label = _seq_label_history_init_state(vocab_dim=vocab_dim, batch_dims=batch_dims_)

    ctc_scale = 1.0

    max_seq_len = int(enc_spatial_dim.get_dim_value())
    seq_targets_wb = []
    seq_backrefs = []
    for t in range(max_seq_len):
        prev_target = target
        prev_target_wb = target_wb

        seq_log_prob = seq_log_prob + ctc_scale * ctc_label_log_prob_ta[t]  # Batch, InBeam, VocabWB

        seq_log_prob, (backrefs, target_wb), beam_dim = rf.top_k(
            seq_log_prob,
            k_dim=ReturnnDim(beam_size, name=f"dec-step{t}-beam"),
            axis=[beam_dim, vocab_wb_dim],
        )
        # seq_log_prob, backrefs, target_wb: Batch, Beam
        # backrefs -> InBeam.
        # target_wb -> VocabWB.
        seq_targets_wb.append(target_wb)
        seq_backrefs.append(backrefs)

        seq_label = rf.nested.gather_nested(seq_label, indices=backrefs)

        prev_target = rf.gather(prev_target, indices=backrefs)  # Batch, Beam -> Vocab
        prev_target_wb = rf.gather(prev_target_wb, indices=backrefs)  # Batch, Beam -> VocabWB

        got_new_label: Tensor = (target_wb != model.blank_idx) & (target_wb != prev_target_wb)  # Batch, Beam -> 0|1
        target = rf.where(
            got_new_label,
            _target_remove_blank(
                target_wb,
                target_dim=vocab_dim,
                wb_target_dim=vocab_wb_dim,
                blank_idx=model.blank_idx,
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
                    same_seq_labels,
                    rf.replace_dim_v2(seq_log_prob, in_dim=beam_dim, out_dim=beam_dual_dim),
                    neg_inf,
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
        (seq_targets_wb, seq_log_prob, out_spatial_dim),
        mask=mask,
        mask_cpu=mask_cpu,
        dims=[beam_dim],
    )

    seq_targets_wb = seq_targets_wb.copy_transpose([batch_dim, beam_dim, out_spatial_dim])
    seq_log_prob = seq_log_prob.copy_transpose([batch_dim, beam_dim])
    # print("seq_targets_wb: ", seq_targets_wb.raw_tensor[0, 0])
    # raise Exception("stop here for debugging")
    ctc_batch_indices, ctc_output_lens = _get_collapsed_out_seqs(
        seq_targets=seq_targets_wb.raw_tensor,
        blank_idx=model.blank_idx,
    )

    if original_blank_idx is not None:
        # since we assume original_blank_idx is 0, we need to shift the indices one up
        # blank is already removed here, so we just need to add 1 to all indices to get back to the original vocab space.
        ctc_batch_indices += 1

    return ctc_batch_indices, seq_log_prob.raw_tensor, ctc_output_lens