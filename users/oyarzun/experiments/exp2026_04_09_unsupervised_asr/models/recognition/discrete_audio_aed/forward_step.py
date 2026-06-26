__all__ = ["forward_step"]

from ....models.definitions.conformer_aed_discrete_shared_v1 import Model
from .beam_search import State, beam_search_v1
import returnn.frontend as rf
from returnn.tensor import Dim, TensorDict, batch_dim


def forward_step(
    *,
    model: Model,
    extern_data: TensorDict,
    beam_size: int,
    **kwargs,
):
    """Runs full recognition on the given data."""

    assert beam_size > 0

    data = extern_data["data"].raw_tensor
    seq_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor.to(device=data.device)

    max_seq_len = seq_len

    decoder_state = model.forward_encoder(data, seq_len, decoder=model.text_decoder, forward_func=model.forward_audio)
    seq_targets, seq_log_prob, _label_log_probs, out_seq_len = beam_search_v1(
        model=model,
        beam_size=beam_size,
        batch_size=data.shape[0],
        decoder_state=decoder_state,
        device=data.device,
        max_seq_len=max_seq_len,
        # decoder=model.text_decoder,
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
