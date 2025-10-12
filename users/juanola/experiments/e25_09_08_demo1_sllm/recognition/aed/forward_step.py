__all__ = ["EncoderDecoderModel", "forward_step"]

from abc import abstractmethod
from typing import Generic, Optional

import returnn.frontend as rf
from returnn.tensor import Dim, TensorDict, batch_dim
from torch import Tensor

from .beam_search import LabelScorer, State, beam_search_v1


class EncoderDecoderModel(LabelScorer[State], Generic[State]):
    """
    Interface for an encoder and a decoder that scores labels.

    Can encode acoustic data into a higher level representation and, as part of that, generate
    an initial decoder state.
    This state also stores the higher level representations.
    """

    @abstractmethod
    def forward_encoder(self, raw_audio: Tensor, raw_audio_lens: Tensor) -> State:
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
    model: EncoderDecoderModel,
    extern_data: TensorDict,
    beam_size: int,
    max_tokens_per_sec: Optional[int] = None,
    sample_rate: Optional[int] = None,
    **kwargs,
):
    """
    Runs full recognition on the given data.

    CALLED FROM RETURNN.CONFIG!!
    """

    assert beam_size > 0

    data = extern_data["data"].raw_tensor
    seq_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor.to(device=data.device)

    max_seq_len = seq_len
    if max_tokens_per_sec is not None and sample_rate is not None:
        assert max_tokens_per_sec > 0 and sample_rate > 0
        max_seq_len = max_tokens_per_sec * (seq_len / sample_rate)

    decoder_state = model.forward_encoder(data, seq_len)
    seq_targets, seq_log_prob, _label_log_probs, out_seq_len = beam_search_v1(
        model=model,
        beam_size=beam_size,
        batch_size=data.shape[0],
        decoder_state=decoder_state,
        device=data.device,
        max_seq_len=max_seq_len,
    )

    beam_dim = Dim(beam_size, name="beam")
    vocab_dim = Dim(model.num_labels, name="vocab")
    lens_data = rf.convert_to_tensor(out_seq_len, dims=[batch_dim, beam_dim])
    lens_dim = Dim(lens_data, name="seq_len")

    ctx = rf.get_run_ctx()
    seq_targets_rf = rf.convert_to_tensor(seq_targets, dims=[batch_dim, beam_dim, lens_dim], sparse_dim=vocab_dim)
    ctx.mark_as_output(seq_targets_rf, "tokens", dims=[batch_dim, beam_dim, lens_dim])
    ctx.mark_as_output(seq_log_prob, "scores", dims=[batch_dim, beam_dim])