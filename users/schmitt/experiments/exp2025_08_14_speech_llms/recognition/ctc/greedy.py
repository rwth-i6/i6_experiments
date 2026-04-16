__all__ = ["State", "LabelScorer", "greedy_decoding_v1"]

from abc import abstractmethod
from functools import partial
from typing import Generic, List, Protocol, Sequence, Tuple, TypeVar, Dict, Optional
from dataclasses import dataclass

from sisyphus import Path

import torch
from torch import Tensor
import torch.nn.functional as F
from torch import nn
import tree


State = TypeVar("State")


class LabelScorer(Protocol, Generic[State]):
    """
    Interface for scoring labels.

    Given existing labels (initial: BOS) and the recurrent state, scores labels.

    Generic over a recurrent decoder state of type `State`.
    `State` can be any PyTree (nested structure composed of the primitive containers list, dict and tuple).

    All tensors in the state are expected to have shape [Batch, Beam, ...Features].
    The data is reprocessed during decoding when e.g. selecting beam backrefs.
    To store beam-indepentent data like the encoder output, the Beam entry can also be set to 1.
    In that case the values are treated as broadcast over all beams and left untouched.
    """

    bos_idx: int
    """Index of the beginning-of-sentence label."""
    eos_idx: int
    """Index of the end-of-sentence label."""
    num_labels: int
    """Number of labels including BOS/EOS."""

    external_lm: Optional[nn.Module]

    @abstractmethod
    def step_decoder(self, labels: Tensor, state: State) -> Tuple[Tensor, State]:
        """
        Run one decoder step, given the labels and recurrent state.

        :param labels: current labels, shape [Batch, Beam, Time=1],
            sparse dim L
        :param state: recurrent decoder state, where the initial state is obtained
            from `forward_encoder`.
        :return: tuple of:
            - logits of the next labels, shape [Batch, Beam, Time=1, L]
            - decoder state for decoding the next step.
        """
        raise NotImplementedError


@dataclass
class DecoderConfig:
    # search related options:
    beam_size: int
    max_tokens_per_sec: int
    sample_rate: int

    def __str__(self) -> str:
        return f"beam-{self.beam_size}-lmw-{0.0}-ilmw-{0.0}"


_T = TypeVar("T")


# when using torch.no_grad here, it leads to an error when running ./sis m:
# TypeError: no_grad.__init__() takes 1 positional argument but 2 were given
# -> use context manager instead (see below)
#  @torch.no_grad
def greedy_decoding_v1(
    *,
    model: LabelScorer,
    batch_size: int,
    device: torch.device,
    max_seq_len: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
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
        assert batch_size > 0
        assert (max_seq_len > 0).all()

        seq_log_prob = torch.full([batch_size], 0.0, dtype=torch.float32, device=device)  # Batch, Beam

        while True:
            logits, decoder_state = model.step_decoder(target.unsqueeze(-1), decoder_state)
            label_log_prob = F.log_softmax(logits, dim=-1)  # Batch, Beam, Vocab
            assert label_log_prob.shape[-2] == 1, f"time dim mismatch, is {label_log_prob.shape[-2]} but should be 1"
            label_log_prob = label_log_prob.squeeze(-2)

            if model.external_lm is not None:
                target_merged = target.reshape(-1)  # (B, beam) -> (B*beam)
                if lm_cache is not None:
                    lm_cache = tree.map_structure(lambda x: x.reshape(-1, *x.shape[2:]), lm_cache)  # (B, beam, ...F) -> (B*beam, ...F)
                lm_logits, lm_cache = model.external_lm(
                    target_merged.unsqueeze(-1),
                    torch.ones_like(target_merged),
                    lm_cache,
                    out_seq_len.reshape(-1)
                )
                lm_log_prob = F.log_softmax(lm_logits, dim=-1)
                lm_log_prob = lm_log_prob.reshape(label_log_prob.shape)  # (B*beam, V) -> (B, beam, V)
                lm_cache = tree.map_structure(
                    lambda x: x.reshape(label_log_prob.shape[0], label_log_prob.shape[1], *x.shape[1:]), lm_cache
                )

                label_log_prob = label_log_prob + lm_log_prob * lm_scale

            if ilm_decoder_state is not None:
                ilm_logits, ilm_decoder_state = model.step_decoder(target.unsqueeze(-1), ilm_decoder_state)
                ilm_log_prob = F.log_softmax(ilm_logits, dim=-1)  # Batch, Beam, Vocab
                assert ilm_log_prob.shape[-2] == 1, f"time dim mismatch, is {ilm_log_prob.shape[-2]} but should be 1"
                ilm_log_prob = ilm_log_prob.squeeze(-2)
                label_log_prob = label_log_prob - ilm_log_prob * ilm_scale

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

            label_log_probs.append(label_log_prob)
            seq_targets.append(target)
            seq_backrefs.append(backrefs)

            # now select from the decoder state those top-k beams
            decoder_state = tree.map_structure(
                partial(_gather_backrefs, backrefs=backrefs, beam_size=beam_size), decoder_state
            )
            if lm_cache is not None:
                lm_cache = tree.map_structure(
                    partial(_gather_backrefs, backrefs=backrefs, beam_size=beam_size), lm_cache
                )
            if ilm_decoder_state is not None:
                ilm_decoder_state = tree.map_structure(
                    partial(_gather_backrefs, backrefs=backrefs, beam_size=beam_size), ilm_decoder_state
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
