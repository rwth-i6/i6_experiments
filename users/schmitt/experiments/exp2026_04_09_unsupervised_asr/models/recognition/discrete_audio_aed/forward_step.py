__all__ = ["forward_step"]

from typing import Dict, Optional

from ....models.definitions.conformer_aed_discrete_shared_v1 import Model
from ....models.train_steps.util import get_random_mask, mask_sequence, expand_sequence
from .beam_search import State, beam_search_v1
import returnn.frontend as rf
from returnn.tensor import Dim, TensorDict, batch_dim


def forward_step(
    *,
    model: Model,
    extern_data: TensorDict,
    beam_size: int,
    input_data_key: str = "data",
    input_modality: str = "audio",
    output_modality: str = "text",
    masking_opts: Optional[Dict] = None,
    expansion_opts: Optional[Dict] = None,
    **kwargs,
):
    """Runs full recognition / reconstruction on the given data.

    By default audio cluster ids (``input_data_key="data"``, ``input_modality="audio"``) are
    encoded and decoded with the text decoder (``output_modality="text"``) -- i.e. standard ASR.

    Setting ``input_modality`` and ``output_modality`` to the *same* modality turns this into a
    same-modality reconstruction (audio->audio or text->text), probing how well the shared denoising
    model reconstructs a given input. ``input_modality`` selects which embedding/encoder path (and,
    for masking, which mask token) is used; ``output_modality`` selects which decoder, vocab and
    bos/eos symbols beam search uses. The scoring side (see ``pipeline.search_single`` /
    ``tune_eval.eval_model``) must reference the matching data key + vocab.

    If ``masking_opts`` is given with ``mask_prob > 0``, the encoder input is masked exactly like in
    training (random spans collapsed to a single mask token, see ``train_steps.util``) before being
    fed to the encoder, while the scoring reference stays the unmasked sequence -- so this measures
    the denoising reconstruction quality. The decoder may still emit up to the *original* (unmasked)
    input length.

    If ``expansion_opts`` is given (``{"min_dup", "max_dup"}``), the (masked) encoder input is
    additionally upsampled by duplicating tokens exactly as in training (see
    ``train_steps.util.expand_sequence``), applied *after* masking. This matches the train-time text
    upsampling so the encoder sees the same longer sequence at recognition; the scoring reference /
    decode length stay at the original (un-expanded, unmasked) length.
    """

    assert beam_size > 0
    assert input_modality in ("audio", "text"), input_modality
    assert output_modality in ("audio", "text"), output_modality

    data = extern_data[input_data_key].raw_tensor
    seq_len = extern_data[input_data_key].dims[1].dyn_size_ext.raw_tensor.to(device=data.device)

    # the decoder may produce up to the original (unmasked) input length
    max_seq_len = seq_len

    # input modality -> which encoder path / mask token
    if input_modality == "audio":
        forward_func = model.forward_audio
        mask_idx = model.audio_mask_idx
    else:
        forward_func = model.forward_text
        mask_idx = model.text_mask_idx

    # output modality -> which decoder / vocab / special symbols
    if output_modality == "text":
        out_decoder = model.text_decoder
        step_func = model.step_text_decoder
        bos_idx = model.text_bos_idx
        eos_idx = model.text_eos_idx
        out_dim = model.text_out_dim
    else:
        out_decoder = model.audio_decoder
        step_func = model.step_audio_decoder
        bos_idx = model.audio_bos_idx
        eos_idx = model.audio_eos_idx
        out_dim = model.audio_out_dim

    # optionally mask the encoder input the same way as during training
    enc_indices, enc_lens = data, seq_len
    if masking_opts is not None and masking_opts.get("mask_prob", 0.0) > 0.0:
        mask = get_random_mask(seq_len, **masking_opts)
        enc_indices, enc_lens = mask_sequence(data, seq_len, mask, mask_value=mask_idx)

    # optionally upsample the (masked) encoder input like in training (after masking), so the encoder
    # sees the same longer sequence; the decode/score length stays at the original (max_seq_len).
    if expansion_opts is not None:
        enc_indices, enc_lens = expand_sequence(enc_indices, enc_lens, **expansion_opts)

    decoder_state = model.forward_encoder(enc_indices, enc_lens, decoder=out_decoder, forward_func=forward_func)
    seq_targets, seq_log_prob, _label_log_probs, out_seq_len = beam_search_v1(
        model=model,
        beam_size=beam_size,
        batch_size=data.shape[0],
        decoder_state=decoder_state,
        device=data.device,
        max_seq_len=max_seq_len,
        step_func=step_func,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
        out_dim=out_dim,
    )

    beam_dim = Dim(beam_size, name="beam")
    vocab_dim = Dim(out_dim, name="vocab")
    lens_data = rf.convert_to_tensor(out_seq_len, dims=[batch_dim, beam_dim])
    lens_dim = Dim(lens_data, name="seq_len")

    ctx = rf.get_run_ctx()
    seq_targets_rf = rf.convert_to_tensor(seq_targets, dims=[batch_dim, beam_dim, lens_dim], sparse_dim=vocab_dim)
    ctx.mark_as_output(seq_targets_rf, "tokens", dims=[batch_dim, beam_dim, lens_dim])
    ctx.mark_as_output(seq_log_prob, "scores", dims=[batch_dim, beam_dim])
