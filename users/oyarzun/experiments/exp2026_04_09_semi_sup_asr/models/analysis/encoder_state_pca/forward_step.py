__all__ = ["forward_step"]

import returnn.frontend as rf
from returnn.tensor import Dim, TensorDict, batch_dim

from ....models.definitions.conformer_aed_discrete_shared_v1 import Model


def _modality_present(data) -> bool:
    """A modality is present in the batch if its (padded) time dim is non-empty and at least one
    sequence has a non-zero length. With the ``CombinedDataset`` an audio-only batch carries
    length-0 text sequences (and vice versa), which we must not feed to the encoder."""
    if data is None:
        return False
    seq_lens = data.dims[1].dyn_size_ext.raw_tensor
    return data.raw_tensor.shape[1] > 0 and bool(seq_lens.max() > 0)


def forward_step(
    *,
    model: Model,
    extern_data: TensorDict,
    audio_data_key: str = "data",
    text_data_key: str = "phon_indices",
    **kwargs,
):
    """
    Forward step for the shared-encoder state analysis.

    For each sequence in the batch, the shared encoder is run over whichever modalities are
    present -- the audio cluster indices (``audio_data_key``) via :meth:`Model.forward_audio`
    and/or the phoneme indices (``text_data_key``) via :meth:`Model.forward_text`. The per-frame
    encoder states are exposed as the forward outputs ``audio_states`` and/or ``text_states``
    (both ``[B, T, F]``). The actual PCA projection and plotting happens in the forward callback
    (:class:`...callback.EncoderStatePcaCallback`).

    Both modalities are handled independently so this works regardless of the dataset:
    a paired ``MetaDataset`` yields both modalities in every batch, while a ``CombinedDataset``
    (alternate batching) yields audio-only or text-only batches -- the absent modality has
    length-0 sequences and is simply skipped here (its output is not marked for that batch).

    Unlike ``notebooks/visualize_embeds.py`` all data loading / batching / seq filtering is handled
    by the RETURNN backend, so this step only has to run the encoder and mark the outputs.
    """
    ctx = rf.get_run_ctx()
    feat_dim = None

    audio = extern_data.data.get(audio_data_key)
    if _modality_present(audio):
        audio_time_dim = audio.dims[1]
        audio_indices = audio.raw_tensor
        audio_lens = audio_time_dim.dyn_size_ext.raw_tensor.to(device=audio_indices.device)
        # encoder has no subsampling frontend, so the encoder time dim equals the input time dim
        # and we can reuse the existing dynamic dim for the output.
        audio_enc, _, _, _ = model.forward_audio(audio_indices.long(), audio_lens)
        feat_dim = Dim(int(audio_enc.shape[-1]), name="enc_feat")
        audio_enc_rf = rf.convert_to_tensor(audio_enc, dims=[batch_dim, audio_time_dim, feat_dim])
        ctx.mark_as_output(audio_enc_rf, "audio_states", dims=[batch_dim, audio_time_dim, feat_dim])

    text = extern_data.data.get(text_data_key)
    if _modality_present(text):
        text_time_dim = text.dims[1]
        text_indices = text.raw_tensor
        text_lens = text_time_dim.dyn_size_ext.raw_tensor.to(device=text_indices.device)
        text_enc, _, _, _ = model.forward_text(text_indices.long(), text_lens)
        if feat_dim is None:
            feat_dim = Dim(int(text_enc.shape[-1]), name="enc_feat")
        text_enc_rf = rf.convert_to_tensor(text_enc, dims=[batch_dim, text_time_dim, feat_dim])
        ctx.mark_as_output(text_enc_rf, "text_states", dims=[batch_dim, text_time_dim, feat_dim])
