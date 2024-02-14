"""
Beam search
"""

from typing import Union, Sequence, Dict, Tuple

from dataclasses import dataclass
import torch

from .interface_torch import LabelScorer


@dataclass
class BeamSearchOpts:
    beam_size: int
    length_normalization_exponent: float
    bos_label: int
    eos_label: int
    num_labels: int


def beam_search(
    label_scorer: LabelScorer,
    *,
    batch_size: int,
    max_seq_len: torch.Tensor,
    device: torch.device,
    opts: BeamSearchOpts,
):
    """
    beam search

    :param beam_size: e.g. 12
    :param length_normalization_exponent: e.g. 1 to enable, 0 to disable
    :param max_seq_len: e.g. use encoder length. shape [Batch]

    """
    # Eager-mode implementation of beam search.
    # Initial state.
    cur_beam_size = 1
    state = label_scorer.get_initial_state(batch_size=batch_size, device=device)
    target = torch.full([batch_size, cur_beam_size], opts.bos_label, device=device)
    ended = torch.full([batch_size, cur_beam_size], False, device=device)
    out_seq_len = torch.full([batch_size, cur_beam_size], 0, device=device)
    seq_log_prob = torch.full([batch_size, cur_beam_size], 0.0, device=device)

    masked_finished_log_prob = torch.where(torch.range(0, opts.num_labels) == opts.eos_label, 0.0, -1.0e30)

    i = 0
    seq_targets = []
    seq_backrefs = []
    while True:
        new_state = label_scorer.update_state(prev_state=state, prev_label=target)
        label_log_prob = label_scorer.score(
            prev_state=state, prev_label=target, state=new_state
        )  # [Batch,InBeam,Vocab]

        # Filter out finished beams
        label_log_prob = torch.where(ended, masked_finished_log_prob, label_log_prob)
        seq_log_prob = seq_log_prob[:, :, None] + label_log_prob  # Batch, InBeam, Vocab
        seq_log_prob, (backrefs, target) = top_k(
            seq_log_prob, k=opts.beam_size, axis=[1, 2]
        )  # seq_log_prob, backrefs, target: Batch, Beam
        seq_targets.append(target)
        seq_backrefs.append(backrefs)
        decoder_state = tree.map_structure(functools.partial(_gather_backrefs, backrefs=backrefs), decoder_state)
        ended = rf.gather(ended, indices=backrefs)
        out_seq_len = rf.gather(out_seq_len, indices=backrefs)
        i += 1

        ended = rf.logical_or(ended, target == model.eos_idx)
        ended = rf.logical_or(ended, rf.copy_to_device(i >= max_seq_len))
        if bool(rf.reduce_all(ended, axis=ended.dims).raw_tensor):
            break
        out_seq_len = out_seq_len + rf.where(ended, 0, 1)

        if i > 1 and opts.length_normalization_exponent != 0:
            # Length-normalized scores, so we evaluate score_t/len.
            # If seq ended, score_i/i == score_{i-1}/(i-1), thus score_i = score_{i-1}*(i/(i-1))
            # Because we count with EOS symbol, shifted by one.
            seq_log_prob *= rf.where(
                ended,
                (i / (i - 1)) ** length_normalization_exponent,
                1.0,
            )

    if i > 0 and opts.length_normalization_exponent != 0:
        seq_log_prob *= (1 / i) ** opts.length_normalization_exponent

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
    out_spatial_dim = Dim(out_seq_len, name="out-spatial")
    seq_targets = seq_targets__.stack(axis=out_spatial_dim)

    return seq_targets, seq_log_prob, out_spatial_dim, beam_dim


def top_k(values: torch.Tensor, *, k: int, axis: Union[int, Sequence[int]]):
    pass


def gather(values: torch.Tensor, *, indices: torch.Tensor) -> torch.Tensor:
    """
    :param values: shape [Batch,Indices,ValuesDims...], e.g. [Batch,InBeam,...]
    :param indices: shape [Batch,IndicesDims...] -> Indices, e.g. [Batch,OutBeam] -> InBeam
    :return: shape [Batch,IndicesDims...,ValuesDims...]
    """
    pass
