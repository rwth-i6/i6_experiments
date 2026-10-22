__all__ = ["forward_step"]

import returnn.frontend as rf
from returnn.tensor import Dim, TensorDict, batch_dim


def _lang_present(data) -> bool:
    """A language stream is present in the batch if its (padded) time dim is non-empty and at least one
    sequence has a non-zero length."""
    if data is None:
        return False
    seq_lens = data.dims[1].dyn_size_ext.raw_tensor
    return data.raw_tensor.shape[1] > 0 and bool(seq_lens.max() > 0)


def forward_step(
    *,
    model,
    extern_data: TensorDict,
    src_data_key: str = "data",
    trg_data_key: str = "target",
    audio_data_key: str = "data",
    text_data_key: str = "target",
    **kwargs,
):
    """
    Forward step for the cross-lingual shared-encoder state PCA analysis.

    For each sequence in the batch, the shared encoder is run over whichever languages are
    present -- the source language token indices (``src_data_key``, e.g. German) via
    ``model.forward_src_encoder`` (or ``model.forward_audio``) and/or the target language
    token indices (``trg_data_key``, e.g. English) via ``model.forward_trg_encoder``
    (or ``model.forward_text``). The per-token encoder states are exposed as the forward
    outputs ``src_states`` (and ``audio_states``) and/or ``trg_states`` (and ``text_states``),
    both ``[B, T, F]``. The actual PCA projection and plotting happens in
    :class:`...callback.EncoderStatePcaCallback`.
    """
    ctx = rf.get_run_ctx()
    feat_dim = None

    s_key = src_data_key if src_data_key in extern_data.data else audio_data_key
    t_key = trg_data_key if trg_data_key in extern_data.data else text_data_key

    src_data = extern_data.data.get(s_key)
    if _lang_present(src_data):
        src_time_dim = src_data.dims[1]
        src_indices = src_data.raw_tensor
        src_lens = src_time_dim.dyn_size_ext.raw_tensor.to(device=src_indices.device)
        if hasattr(model, "forward_src_encoder"):
            src_enc, _, _, _ = model.forward_src_encoder(src_indices.long(), src_lens)
        else:
            src_enc, _, _, _ = model.forward_audio(src_indices.long(), src_lens)
        feat_dim = Dim(int(src_enc.shape[-1]), name="enc_feat")
        src_enc_rf = rf.convert_to_tensor(src_enc, dims=[batch_dim, src_time_dim, feat_dim])
        ctx.mark_as_output(src_enc_rf, "audio_states", dims=[batch_dim, src_time_dim, feat_dim])

    trg_data = extern_data.data.get(t_key)
    if _lang_present(trg_data):
        trg_time_dim = trg_data.dims[1]
        trg_indices = trg_data.raw_tensor
        trg_lens = trg_time_dim.dyn_size_ext.raw_tensor.to(device=trg_indices.device)
        if hasattr(model, "forward_trg_encoder"):
            trg_enc, _, _, _ = model.forward_trg_encoder(trg_indices.long(), trg_lens)
        else:
            trg_enc, _, _, _ = model.forward_text(trg_indices.long(), trg_lens)
        if feat_dim is None:
            feat_dim = Dim(int(trg_enc.shape[-1]), name="enc_feat")
        trg_enc_rf = rf.convert_to_tensor(trg_enc, dims=[batch_dim, trg_time_dim, feat_dim])
        ctx.mark_as_output(trg_enc_rf, "text_states", dims=[batch_dim, trg_time_dim, feat_dim])
