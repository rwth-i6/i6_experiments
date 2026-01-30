__all__ = ["EncoderDecoderModel", "forward_step", "aed_recog_ctc_rescore_forward_step_v1"]

from abc import abstractmethod
from typing import Generic, Optional, Callable, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from .beam_search import LabelScorer, State, beam_search_v1
import returnn.frontend as rf
from returnn.tensor import Dim, TensorDict, batch_dim

from i6_models.assemblies.transformer.transformer_decoder_v1 import TransformerDecoderV1, TransformerDecoderV1State


class EncoderDecoderModel(LabelScorer[State], Generic[State]):
    """
    Interface for an encoder and a decoder that scores labels.

    Can encode acoustic data into a higher level representation and, as part of that, generate
    an initial decoder state.
    This state also stores the higher level representations.
    """

    text_decoder: TransformerDecoderV1

    @abstractmethod
    def forward_encoder(
            self, indices: Tensor, indices_lens: Tensor, decoder: TransformerDecoderV1, forward_func: Callable
    ) -> TransformerDecoderV1State:
        pass

    @abstractmethod
    def forward_audio(self, indices: Tensor, seq_lens: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        pass


def forward_step(
    *,
    model: EncoderDecoderModel,
    extern_data: TensorDict,
    beam_size: int,
    max_tokens_per_sec: Optional[int] = None,
    sample_rate: Optional[int] = None,
    **kwargs,
):
    """Runs full recognition on the given data."""

    assert beam_size > 0

    data = extern_data["data"].raw_tensor
    seq_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor.to(device=data.device)

    max_seq_len = seq_len

    decoder_state = model.forward_encoder(
        data,
        seq_len,
        decoder=model.text_decoder,
        forward_func=model.forward_audio
    )
    seq_targets, seq_log_prob, _label_log_probs, out_seq_len = beam_search_v1(
        model=model,
        beam_size=beam_size,
        batch_size=data.shape[0],
        decoder_state=decoder_state,
        device=data.device,
        max_seq_len=max_seq_len,
        decoder=model.text_decoder,
        bos_idx=model.text_bos_idx,
        eos_idx=model.text_eos_idx,
        out_dim=model.text_out_dim,
    )

    beam_dim = Dim(beam_size, name="beam")
    vocab_dim = Dim(model.text_out_dim, name="vocab")
    lens_data = rf.convert_to_tensor(out_seq_len, dims=[batch_dim, beam_dim])
    lens_dim = Dim(lens_data, name="seq_len")

    ctx = rf.get_run_ctx()
    seq_targets_rf = rf.convert_to_tensor(seq_targets, dims=[batch_dim, beam_dim, lens_dim], sparse_dim=vocab_dim)
    ctx.mark_as_output(seq_targets_rf, "tokens", dims=[batch_dim, beam_dim, lens_dim])
    ctx.mark_as_output(seq_log_prob, "scores", dims=[batch_dim, beam_dim])


def aed_recog_ctc_rescore_forward_step_v1(
    *,
    model: EncoderDecoderModel,
    extern_data: TensorDict,
    beam_size: int,
    ctc_weight: float = 0.3,
    max_tokens_per_sec: Optional[int] = None,
    sample_rate: Optional[int] = None,
    **kwargs,
):
    """Runs full recognition on the given data."""

    assert beam_size > 0
    assert 0 < ctc_weight <= 1, "ctc_weight must be in (0, 1]"

    data = extern_data["data"].raw_tensor
    seq_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor.to(device=data.device)

    max_seq_len = seq_len
    if max_tokens_per_sec is not None and sample_rate is not None:
        assert max_tokens_per_sec > 0 and sample_rate > 0
        max_seq_len = max_tokens_per_sec * (seq_len / sample_rate)

    decoder_state, ctc_logits, ctc_logit_lens = model.forward_encoder_with_ctc(data, seq_len)
    seq_targets, seq_log_prob, _label_log_probs, out_seq_len = beam_search_v1(
        model=model,
        beam_size=beam_size,
        batch_size=data.shape[0],
        decoder_state=decoder_state,
        device=data.device,
        max_seq_len=max_seq_len,
    )

    ctc_logits_expanded = (
        ctc_logits.unsqueeze(1)
        .expand(-1, beam_size, -1, -1)
        .reshape(-1, *ctc_logits.shape[1:])
        .transpose(0, 1)
        .to(dtype=torch.float32)
    )
    input_lengths_expanded = ctc_logit_lens.unsqueeze(1).expand(-1, beam_size).reshape(-1)
    ctc_scores = -F.ctc_loss(
        ctc_logits_expanded,
        seq_targets.reshape(-1, seq_targets.shape[-1]),
        input_lengths=input_lengths_expanded,
        target_lengths=out_seq_len.reshape(-1),
        blank=model.blank_idx,
        reduction="none",
        zero_infinity=True,
    )
    ctc_scores = ctc_scores.reshape(*seq_targets.shape[:2])
    ctc_scores = ctc_scores / out_seq_len

    fused_score = seq_log_prob + ctc_weight * ctc_scores

    beam_dim = Dim(beam_size, name="beam")
    vocab_dim = Dim(model.num_labels, name="vocab")
    lens_data = rf.convert_to_tensor(out_seq_len, dims=[batch_dim, beam_dim])
    lens_dim = Dim(lens_data, name="seq_len")

    ctx = rf.get_run_ctx()
    seq_targets_rf = rf.convert_to_tensor(seq_targets, dims=[batch_dim, beam_dim, lens_dim], sparse_dim=vocab_dim)
    ctx.mark_as_output(seq_targets_rf, "tokens", dims=[batch_dim, beam_dim, lens_dim])
    ctx.mark_as_output(fused_score, "scores", dims=[batch_dim, beam_dim])
