__all__ = ["beam_search_decode"]

from functools import partial
from typing import List, Sequence, Tuple, TypeVar

import torch
import torch.nn.functional as F
import tree
from torch import Tensor

from ..networks.interfaces.label_scorer_protocol import LabelScorerProtocol, State


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
    assert (
            state.shape[1] == 1 or state.shape[1] == beam_size
    ), f"Beam dim must either be 1 or beam_size ({beam_size}) but is {state.shape[1]}"
    if state.shape[1] == 1:
        return state  # broadcast, e.g. encoder state in [B,1,T,F]
    assert backrefs.ndim == 2, "need at least batch and beam dims in backrefs"

    return _batch_gather(state, indices=backrefs, index_dim=1)


def _top_k_nd(source: Tensor, *, k: int, dim: Sequence[int], sorted: bool = True) -> Tuple[Tensor, List[Tensor]]:
    """
    torch.top_k, but with support for search over multiple dimensions.
    (Derived from returnn.torch.frontend._backend.TorchBackend.top_k.)

    This runs a top-k search across all beams and across all vocab entries,
    returning the top `k` entries per batch.

    :param source: [Batch...,SourceDims...]
    :param k: how many
    :param dim: axes of SourceDims, multiple dims to search in
    :param sorted: sorted output, see :func:`torch.topk`
    :return: values [Batch...,k], list of indices per dim, each [Batch...,k]
    """
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


# when using torch.no_grad here, it leads to an error when running ./sis m:
# TypeError: no_grad.__init__() takes 1 positional argument but 2 were given
# -> use context manager instead (see below)
#  @torch.no_grad
def beam_search_decode(
        *,
        model: LabelScorerProtocol,
        decoder_state: State,

        beam_size: int,
        batch_size: int,

        device: torch.device,
        max_seq_len: Tensor,
        length_norm_exponent: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Eager-mode implementation of beam search.

    :param model: interface to the decoder model
    :param decoder_state: initial recurrent decoder state

    :param beam_size: the beam size, 16 is a good value
    :param batch_size: the batch size

    :param device: the device the computation happens on
    :param length_norm_exponent: scaling exponent for the length normalization factor
    :param max_seq_len: how long the decoded seqs can be at max, use e.g. encoder time dim.

    :return:

    Uses initial_beam_size = beam_size



    """
    assert beam_size > 0, "beam_size must be positive"
    assert batch_size > 0, "batch_size must be positive"
    assert (max_seq_len > 0).all(), "all max_seq_len must be positive"

    with torch.no_grad():
        # TODO: state could be extracted (for cleaner main method)
        target = torch.full([batch_size, beam_size], model.bos_idx, dtype=torch.int32, device=device) # [B, b]
        ended = torch.full([batch_size, beam_size], False, device=device) # [B, b]
        seq_log_prob = torch.full([batch_size, beam_size], 0.0, dtype=torch.float32, device=device) # [B, b]
        out_seq_len = torch.full([batch_size, beam_size], 0, dtype=torch.int32, device=device) # [B, b]

        ended_beams_log_prob_mask = F.one_hot(torch.tensor(model.eos_idx, device=device), num_classes=model.num_labels)
        ended_beams_log_prob_mask = torch.where(ended_beams_log_prob_mask.bool(), 0.0, float('-inf')) # [V]

        label_log_probs = []
        seq_backrefs = []
        seq_targets = []

        # Decoding Loop
        step = torch.tensor(0, device=device, dtype=torch.int32)
        while True:
            # DECODER (FORWARD) STEP (for inference)
            logits, decoder_state = model.step_decoder(target.unsqueeze(-1), decoder_state)

            label_log_prob = F.log_softmax(logits, dim=-1)  # [Batch, Beam, ?, Vocab]
            assert label_log_prob.shape[-2] == 1, f"time dim mismatch, is {label_log_prob.shape[-2]} but should be 1" #TODO: why??
            label_log_prob = label_log_prob.squeeze(-2)
            label_log_prob = torch.where(ended[:, :, None], ended_beams_log_prob_mask[None, None, :], label_log_prob)  # filter out finished beams

            seq_log_prob = seq_log_prob[:, :, None] + label_log_prob  # [B, b, V]
            assert seq_log_prob.ndim == 3

            # `backrefs` gives us the index of the beam the top-k entry came from
            # `target` gives us the index of the top-k entry label.
            seq_log_prob, (backrefs, target) = _top_k_nd(seq_log_prob, k=beam_size, dim=[1, 2])

            # Store iteration info
            label_log_probs.append(label_log_prob)
            seq_backrefs.append(backrefs)
            seq_targets.append(target)

            # Update structures with selected top-k beams (using backrefs structure)
            decoder_state = tree.map_structure(
                partial(_gather_backrefs, backrefs=backrefs, beam_size=beam_size), decoder_state
            )
            ended = _gather_backrefs(ended, backrefs=backrefs, beam_size=beam_size)
            out_seq_len = _gather_backrefs(out_seq_len, backrefs=backrefs, beam_size=beam_size)

            step += 1

            # Termination Condition
            ended = ended | (target == model.eos_idx)
            ended = ended | (step >= max_seq_len)[:, None]
            if ended.all():
                break

            # Update output sequence length (for those not ended)
            out_seq_len = out_seq_len + torch.where(ended, 0, 1)

            seq_log_prob = apply_length_normalization(ended, length_norm_exponent, seq_log_prob, step)

        if step > 1 and length_norm_exponent != 0:
            seq_log_prob *= (1 / step) ** length_norm_exponent

        # Backtrack via backrefs, resolve beams.
        seq_targets_ = []
        indices = torch.arange(beam_size, device=device)[None, :].expand(batch_size, -1)  # [B, FinalBeam] -> FinalBeam
        for backrefs, target in zip(seq_backrefs[::-1], seq_targets[::-1]):
            # indices: [Batch,FinalBeam] -> Beam
            # backrefs: [Batch,Beam] -> PrevBeam
            seq_targets_.insert(0, _batch_gather(target, indices=indices))  # [Batch,FinalBeam]
            indices = _batch_gather(backrefs, indices=indices)  # [Batch,FinalBeam] -> PrevBeam

        seq_targets = torch.stack(seq_targets_, dim=2)  # [Batch, FinalBeam, OutSeqLen]
        label_log_probs_expanded = [l_p.expand(l_p.shape[0], beam_size, *l_p.shape[2:]) for l_p in label_log_probs]
        label_log_probs = torch.stack(label_log_probs_expanded, dim=2)  # Batch, FinalBeam, OutSeqLen, Label

        return seq_targets, seq_log_prob, label_log_probs, out_seq_len


def apply_length_normalization(ended: Tensor, length_norm_exponent: float, seq_log_prob: Tensor, step: Tensor) -> Tensor:
    """
    Length-normalized scores, so we evaluate score_t/len.
    If seq ended, score_i/i == score_{i-1}/(i-1), thus score_i = score_{i-1}*(i/(i-1))
    Because we count with EOS symbol, shifted by one.

    :param ended:
    :param length_norm_exponent:
    :param seq_log_prob:
    :param step:
    :return:
    """
    if step > 1 and length_norm_exponent != 0:
        seq_log_prob = seq_log_prob * torch.where(ended, (step / (step - 1)) ** length_norm_exponent, 1.0)
    return seq_log_prob

def beam_search_v2( # TODO: check this!!
        *,
        model: LabelScorerProtocol,
        beam_size: int,
        batch_size: int,
        decoder_state: State,
        device: torch.device,
        length_norm_exponent: float = 1.0,
        max_seq_len: Tensor,
        initial_target: Tensor | None = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Eager-mode implementation of beam search.

    Fix initial beam size to be 1.

    :param model: interface to the decoder model
    :param beam_size: the beam size, 16 is a good value
    :param batch_size: the batch size
    :param decoder_state: initial recurrent decoder state
    :param device: the device the computation happens on
    :param length_norm_exponent: scaling exponent for the length normalization factor
    :param max_seq_len: how long the decoded seqs can be at max, use e.g. encoder time dim.
        Shape: [B,]
    :param initial_target: start decoding with these BOS tokens.
        Shape: [B,]
    """
    with torch.no_grad():
        assert beam_size > 0
        assert batch_size > 0
        assert (max_seq_len > 0).all()

        # First step uses beam=1, since the start state is the same for all beams, and multiple
        # beams containing the same contents cause issues in top-k search.
        initial_beam = 1
        if initial_target is None:
            target = torch.full([batch_size, initial_beam], model.bos_idx, dtype=torch.int32, device=device)  # Batch, Beam
        else:
            assert initial_target.shape == (batch_size,)
            target = initial_target[:, None]  # Batch, Beam
        ended = torch.full([batch_size, initial_beam], False, device=device)  # Batch, Beam
        seq_log_prob = torch.full([batch_size, initial_beam], 0.0, dtype=torch.float32, device=device)  # Batch, Beam
        out_seq_len = torch.full([batch_size, initial_beam], 0, dtype=torch.int32, device=device)  # Batch, Beam

        ended_default = F.one_hot(torch.tensor(model.eos_idx, device=device), num_classes=model.num_labels)
        ended_default = torch.where(ended_default.bool(), 0.0, -1e30)

        seq_targets = []
        seq_backrefs = []
        label_log_probs = []
        step = torch.tensor(0, device=device, dtype=torch.int32)

        while True:
            logits, decoder_state = model.step_decoder(target.unsqueeze(-1), decoder_state)
            label_log_prob = F.log_softmax(logits, dim=-1)  # Batch, Beam, Vocab
            assert label_log_prob.shape[-2] == 1, f"time dim mismatch, is {label_log_prob.shape[-2]} but should be 1"
            label_log_prob = label_log_prob.squeeze(-2)
            label_log_prob = torch.where(
                ended[:, :, None], ended_default[None, None, :], label_log_prob
            )  # filter out finished beams

            seq_log_prob = seq_log_prob[:, :, None] + label_log_prob  # Batch, Beam, Vocab
            assert seq_log_prob.ndim == 3

            # This runs a top-k search across all beams and across all vocab entries,
            # returning the top `beam_size` entries per batch.
            #
            # `backrefs` gives us the index of the beam the top-k entry came from, while
            # `target` gives us the index of the top-k entry label.
            seq_log_prob, (backrefs, target) = _top_k_nd(seq_log_prob, k=beam_size, dim=[1, 2])

            label_log_probs.append(label_log_prob)  # Note: this is wrong, also needs to be re-gathered by backrefs
            seq_targets.append(target)
            seq_backrefs.append(backrefs)

            # now select from the decoder state those top-k beams
            decoder_state = tree.map_structure(
                partial(_gather_backrefs, backrefs=backrefs, beam_size=beam_size), decoder_state
            )
            ended = _gather_backrefs(ended, backrefs=backrefs, beam_size=beam_size)
            out_seq_len = _gather_backrefs(out_seq_len, backrefs=backrefs, beam_size=beam_size)

            step += 1

            ended = ended | (target == model.eos_idx)
            ended = ended | (step >= max_seq_len)[:, None]
            if ended.all():
                break

            out_seq_len = out_seq_len + torch.where(ended, 0, 1)

            # Length-normalized scores, so we evaluate score_t/len.
            # If seq ended, score_i/i == score_{i-1}/(i-1), thus score_i = score_{i-1}*(i/(i-1))
            # Because we count with EOS symbol, shifted by one.
            if step > 1 and length_norm_exponent != 0:
                seq_log_prob = seq_log_prob * torch.where(ended, (step / (step - 1)) ** length_norm_exponent, 1.0)

        if step > 1 and length_norm_exponent != 0:
            seq_log_prob *= (1 / step) ** length_norm_exponent

        # Backtrack via backrefs, resolve beams.
        seq_targets_ = []
        indices = torch.arange(beam_size, device=device)[None, :].expand(batch_size, -1)  # [Batch,FinalBeam] -> FinalBeam
        for backrefs, target in zip(seq_backrefs[::-1], seq_targets[::-1]):
            # indices: [Batch,FinalBeam] -> Beam
            # backrefs: [Batch,Beam] -> PrevBeam
            seq_targets_.insert(0, _batch_gather(target, indices=indices))  # [Batch,FinalBeam]
            indices = _batch_gather(backrefs, indices=indices)  # [Batch,FinalBeam] -> PrevBeam

        seq_targets = torch.stack(seq_targets_, dim=2)  # [Batch, FinalBeam, OutSeqLen]
        label_log_probs_expanded = [l_p.expand(l_p.shape[0], beam_size, *l_p.shape[2:]) for l_p in label_log_probs]
        label_log_probs = torch.stack(label_log_probs_expanded, dim=2)  # Batch, FinalBeam, OutSeqLen, Label
        return seq_targets, seq_log_prob, label_log_probs, out_seq_len
