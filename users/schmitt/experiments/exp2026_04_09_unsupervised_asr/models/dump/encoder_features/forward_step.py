__all__ = ["forward_step"]

from typing import Dict, Optional, Tuple

import torch

import returnn.frontend as rf
from returnn.tensor import Dim, TensorDict, batch_dim

from ....models.definitions.conformer_aed_discrete_shared_v1 import Model
from ....models.train_steps.util import get_random_mask, mask_sequence


def _num_enc_layers(model: Model, modality: str) -> int:
    """Number of encoder layers. ``_out_fetch_layers_*`` always contains ``num_layers - 1``
    (it is built as ``{*aux_loss_layers, num_layers} - 1``), so its max + 1 is the layer count."""
    fetch = model._out_fetch_layers_audio if modality == "audio" else model._out_fetch_layers_text
    return fetch[-1] + 1


def _encode(
    model: Model, modality: str, indices: torch.Tensor, lens: torch.Tensor, layer: Optional[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the shared encoder over ``indices`` and return ``(states [B,T,F], out_lens [B])``.

    For the last layer (``layer`` None or == num_enc_layers) this goes through
    :meth:`Model.forward_audio` / :meth:`Model.forward_text`, i.e. exactly the path training and
    recognition take -- including the quantizer, if the model has a codebook. An intermediate
    ``layer`` is fetched straight from the encoder instead; note that the quantizer only ever
    applies to the final layer, so selecting an intermediate one bypasses it by construction.
    """
    forward = model.forward_audio if modality == "audio" else model.forward_text
    if layer is None or layer == _num_enc_layers(model, modality):
        states, _, out_lens, _ = forward(indices, lens)
        return states, out_lens

    assert 1 <= layer < _num_enc_layers(model, modality), (
        f"layer {layer} out of range for {_num_enc_layers(model, modality)} encoder layers"
    )
    embedding = model.audio_embedding if modality == "audio" else model.text_embedding
    data = embedding(indices)  # [B, T, F]
    data_mask = torch.less(torch.arange(data.shape[-2], device=data.device)[None, :], lens[:, None])
    encoder_outputs, out_mask = model.encoder.forward(data, data_mask, return_layers=[layer - 1])
    return encoder_outputs[-1], out_mask.sum(dim=-1)


def forward_step(
    *,
    model: Model,
    extern_data: TensorDict,
    data_key: str = "data",
    modality: str = "audio",
    layer: Optional[int] = None,
    masking_opts: Optional[Dict] = None,
    **kwargs,
):
    """
    Dump the shared encoder's per-frame states so they can be used as *input features* by another
    setup (see ``sis_recipe/dump_features.py`` and the wav2vec-U encoder-feature configs), instead
    of re-running the encoder in every training step.

    The encoder has no subsampling frontend, so the output is frame-synchronous with the input
    symbols (the collapsed cluster ids for ``modality="audio"``): one 512-d state per input token.

    :param data_key: extern_data key holding the encoder input symbols.
    :param modality: ``"audio"`` (cluster ids, the ASR input side) or ``"text"`` (phoneme ids).
    :param layer: 1-based encoder layer to dump; None (default) = the last one, i.e. exactly the
        states recognition conditions on.
    :param masking_opts: if given (``mask_prob``/``min_span``/``max_span``), mask the encoder input
        like in training. Off by default -- the dumped features are meant to be used as clean
        inputs, so they should be the states the model produces at inference time.
    """
    assert modality in ("audio", "text"), modality

    data = extern_data[data_key]
    time_dim = data.dims[1]
    indices = data.raw_tensor.long()
    lens = time_dim.dyn_size_ext.raw_tensor.to(device=indices.device)

    changed = False
    if masking_opts is not None and masking_opts.get("mask_prob", 0.0) > 0.0:
        mask_idx = model.audio_mask_idx if modality == "audio" else model.text_mask_idx
        mask = get_random_mask(lens, **masking_opts)
        indices, lens = mask_sequence(indices, lens, mask, mask_value=mask_idx)
        changed = True

    states, out_lens = _encode(model, modality, indices, lens, layer)

    # no subsampling -> the encoder time dim equals the (possibly masked) input time dim; reuse the
    # existing dynamic dim unless masking changed the lengths.
    if changed:
        lens_rf = rf.convert_to_tensor(out_lens.to(device="cpu", dtype=torch.int32), dims=[batch_dim])
        out_time_dim = Dim(lens_rf, name="enc_time")
    else:
        out_time_dim = time_dim
    feat_dim = Dim(int(states.shape[-1]), name="enc_feat")
    states_rf = rf.convert_to_tensor(states, dims=[batch_dim, out_time_dim, feat_dim])
    rf.get_run_ctx().mark_as_output(states_rf, "features", dims=[batch_dim, out_time_dim, feat_dim])
