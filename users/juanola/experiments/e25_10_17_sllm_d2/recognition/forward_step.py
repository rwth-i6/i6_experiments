__all__ = ["forward_step"]

from typing import Optional

import returnn.frontend as rf
from returnn.tensor import Dim, TensorDict, batch_dim
from returnn.tensor import Tensor as ReturnnTensor
from torch import Tensor

from .beam_search import beam_search_v1
from ..networks.conformer_qwen_v1 import Qwen2DecoderState
from ..networks.interfaces.base_encoder_decoder_model import BaseEncoderDecoderModel


def forward_step(
    *,
    model: BaseEncoderDecoderModel,
    extern_data: TensorDict,

    beam_size: int,
    max_tokens_per_sec: Optional[int] = None,
    sample_rate: Optional[int] = None,

    **kwargs,
):
    """
    Runs full recognition on the given data.

    RETURNN ENTRYPOINT!!
    """
    assert beam_size > 0

    initial_beam_size = beam_size

    data_: ReturnnTensor = extern_data["data"]
    data: Tensor = data_.raw_tensor
    seq_len: Tensor = data_.dims[1].dyn_size_ext.raw_tensor.to(device=data.device)

    max_seq_len = seq_len
    if max_tokens_per_sec is not None and sample_rate is not None:
        assert max_tokens_per_sec > 0 and sample_rate > 0
        max_seq_len = max_tokens_per_sec * (seq_len / sample_rate)

    # ENCODER (FORWARD) STEP (for inference)
    decoder_state: Qwen2DecoderState = model.forward_encoder(data, seq_len, initial_beam_size)

    # BEAM SEARCH (contains DECODER (FORWARD) STEPs)
    seq_targets, seq_log_prob, _label_log_probs, out_seq_len = beam_search_v1( # TODO: should receive init_size by param
        model=model,
        decoder_state=decoder_state,
        beam_size=beam_size,
        initial_beam_size=initial_beam_size,
        batch_size=data.shape[0],
        device=data.device,
        max_seq_len=max_seq_len,
    )

    beam_dim = Dim(beam_size, name="beam")
    vocab_dim = Dim(model.num_labels, name="vocab")
    lens_data = rf.convert_to_tensor(out_seq_len, dims=[batch_dim, beam_dim])
    lens_dim = Dim(lens_data, name="seq_len")

    seq_targets_rf = rf.convert_to_tensor(seq_targets, dims=[batch_dim, beam_dim, lens_dim], sparse_dim=vocab_dim)

    ctx = rf.get_run_ctx()
    ctx.mark_as_output(seq_targets_rf, "tokens", dims=[batch_dim, beam_dim, lens_dim])
    ctx.mark_as_output(seq_log_prob, "scores", dims=[batch_dim, beam_dim])