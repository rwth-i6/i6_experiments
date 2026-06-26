__all__ = ["CtcModel", "forward_step"]

from abc import abstractmethod
from typing import Generic, Optional, Dict, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from .greedy import LabelScorer, State, greedy_decoding_v1
from ... import PACKAGE
import returnn.frontend as rf
from returnn.tensor import Dim, TensorDict, batch_dim


class CtcModel:
    """
    Interface for an encoder and a decoder that scores labels.

    Can encode acoustic data into a higher level representation and, as part of that, generate
    an initial decoder state.
    This state also stores the higher level representations.
    """

    @abstractmethod
    def forward(self, raw_audio: Tensor, raw_audio_lens: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward the raw audio data through the encoder and initialize decoder state from it.

        :param raw_audio: audio data, shape [B,T,1]
        :param raw_audio_lens: lengths of the audio in `raw_audio`, shape [B,]
        :return: decoder state initialized by passing the `raw_audio` through the encoder and
            initializing a fresh decoder state with it.
        """
        raise NotImplementedError


def forward_step(
    *,
    model: CtcModel,
    extern_data: TensorDict,
    max_tokens_per_sec: Optional[int] = None,
    sample_rate: Optional[int] = None,
    **kwargs,
):
    """Runs full recognition on the given data."""

    data = extern_data["data"].raw_tensor
    seq_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor.to(device=data.device)

    max_seq_len = seq_len
    if max_tokens_per_sec is not None and sample_rate is not None:
        assert max_tokens_per_sec > 0 and sample_rate > 0
        max_seq_len = max_tokens_per_sec * (seq_len / sample_rate)

    out_logits, _, out_seq_len, out_mask = model.forward(data, seq_len)
    out_log_prob = F.log_softmax(out_logits, dim=-1)
    seq_targets = torch.argmax(out_log_prob, dim=-1)

    seq_log_prob = torch.max(out_log_prob, dim=-1).values
    seq_log_prob[~out_mask] = 0.0
    seq_log_prob = torch.sum(seq_log_prob, dim=1)

    # seq_targets, seq_log_prob, _label_log_probs, out_seq_len = greedy_decoding_v1(
    #     model=model,
    #     batch_size=data.shape[0],
    #     device=data.device,
    #     max_seq_len=max_seq_len,
    # )

    out_seq_len = out_seq_len.unsqueeze(1) # (B,) -> (B,1)
    seq_targets = seq_targets.unsqueeze(1)  # (B,T) -> (B,1,T)
    seq_log_prob = seq_log_prob.unsqueeze(1)  # (B,) -> (B,1)

    beam_dim = Dim(1, name="beam")  # greedy
    vocab_dim = Dim(model.out_dim_w_blank, name="vocab")
    lens_data = rf.convert_to_tensor(out_seq_len, dims=[batch_dim, beam_dim])
    lens_dim = Dim(lens_data, name="seq_len")

    ctx = rf.get_run_ctx()
    seq_targets_rf = rf.convert_to_tensor(seq_targets, dims=[batch_dim, beam_dim, lens_dim], sparse_dim=vocab_dim)
    ctx.mark_as_output(seq_targets_rf, "tokens", dims=[batch_dim, beam_dim, lens_dim])
    ctx.mark_as_output(seq_log_prob, "scores", dims=[batch_dim, beam_dim])
