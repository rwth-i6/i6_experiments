__all__ = ["forward_step"]

from typing import Dict, Optional

import returnn.frontend as rf
from returnn.tensor import Dim, TensorDict, batch_dim

from ....models.definitions.conformer_aed_discrete_shared_v1 import Model
from ....models.train_steps.util import get_random_mask, mask_sequence, expand_sequence


def _modality_present(data) -> bool:
    """A modality is present in the batch if its (padded) time dim is non-empty and at least one
    sequence has a non-zero length. With the ``CombinedDataset`` an audio-only batch carries
    length-0 text sequences (and vice versa), which we must not feed to the encoder."""
    if data is None:
        return False
    seq_lens = data.dims[1].dyn_size_ext.raw_tensor
    return data.raw_tensor.shape[1] > 0 and bool(seq_lens.max() > 0)


def _maybe_transform(indices, lens, mask_idx, masking_opts: Optional[Dict], expansion_opts: Optional[Dict]):
    """Optionally mask (collapse random spans to ``mask_idx``) and then upsample the encoder input
    exactly as in training. Returns ``(indices, lens, changed)`` where ``changed`` is True iff the
    sequence length was altered (in which case the caller must build a fresh time dim)."""
    changed = False
    if masking_opts is not None and masking_opts.get("mask_prob", 0.0) > 0.0:
        mask = get_random_mask(lens, **masking_opts)
        indices, lens = mask_sequence(indices, lens, mask, mask_value=mask_idx)
        changed = True
    if expansion_opts is not None:
        indices, lens = expand_sequence(indices, lens, **expansion_opts)
        changed = True
    return indices, lens, changed


def forward_step(
    *,
    model: Model,
    extern_data: TensorDict,
    audio_data_key: str = "data",
    text_data_key: str = "phon_indices",
    audio_masking_opts: Optional[Dict] = None,
    text_masking_opts: Optional[Dict] = None,
    text_expansion_opts: Optional[Dict] = None,
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

    ``audio_masking_opts`` / ``text_masking_opts`` (``mask_prob``/``min_span``/``max_span``) and
    ``text_expansion_opts`` (``{"min_dup", "max_dup"}``) optionally mask / upsample the encoder
    input exactly as in training (masking first, then upsampling), so the analysis reflects what the
    encoder sees during training. When set, the encoder time dim changes, so a fresh dynamic dim is
    built from the transformed lengths. All default to None -> the raw input is fed unchanged.

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
        audio_indices, audio_lens, audio_changed = _maybe_transform(
            audio_indices.long(), audio_lens, model.audio_mask_idx, audio_masking_opts, None
        )
        # encoder has no subsampling frontend, so the encoder time dim equals the (transformed) input
        # time dim: reuse the existing dynamic dim when unchanged, else build a fresh one.
        audio_enc, _, _, _ = model.forward_audio(audio_indices, audio_lens)
        feat_dim = Dim(int(audio_enc.shape[-1]), name="enc_feat")
        out_time_dim = _time_dim(audio_lens, "audio_enc_time") if audio_changed else audio_time_dim
        audio_enc_rf = rf.convert_to_tensor(audio_enc, dims=[batch_dim, out_time_dim, feat_dim])
        ctx.mark_as_output(audio_enc_rf, "audio_states", dims=[batch_dim, out_time_dim, feat_dim])

    text = extern_data.data.get(text_data_key)
    if _modality_present(text):
        text_time_dim = text.dims[1]
        text_indices = text.raw_tensor
        text_lens = text_time_dim.dyn_size_ext.raw_tensor.to(device=text_indices.device)
        text_indices, text_lens, text_changed = _maybe_transform(
            text_indices.long(), text_lens, model.text_mask_idx, text_masking_opts, text_expansion_opts
        )
        text_enc, _, _, _ = model.forward_text(text_indices, text_lens)
        if feat_dim is None:
            feat_dim = Dim(int(text_enc.shape[-1]), name="enc_feat")
        out_time_dim = _time_dim(text_lens, "text_enc_time") if text_changed else text_time_dim
        text_enc_rf = rf.convert_to_tensor(text_enc, dims=[batch_dim, out_time_dim, feat_dim])
        ctx.mark_as_output(text_enc_rf, "text_states", dims=[batch_dim, out_time_dim, feat_dim])


def _time_dim(lens, name: str) -> Dim:
    """Build a fresh dynamic time dim from per-seq lengths (used when masking/upsampling changed the
    sequence length). Dynamic dim sizes live on CPU as int32, like RETURNN's own ``dyn_size_ext``."""
    import torch

    lens_rf = rf.convert_to_tensor(lens.to(device="cpu", dtype=torch.int32), dims=[batch_dim])
    return Dim(lens_rf, name=name)
