from typing import Dict, Optional, Sequence

from returnn.tensor import TensorDict

from . import aed_denoising_discrete
from .aed_denoising_discrete_shared import SharedDenoisingAedModel


def train_step(
    *,
    model: SharedDenoisingAedModel,
    extern_data: TensorDict,
    text_ce_loss_scale: Optional[float] = None,
    text_masked_ce_loss_scale: Optional[float] = None,
    label_smoothing: float = 0.0,
    label_smoothing_start_epoch: int = 0,
    text_masking_opts: Optional[Dict] = None,
    aux_loss_scales: Optional[Sequence[float]] = None,
    codebook_diversity_loss_scale: float = 0.0,
    **_kwargs,
):
    """
    Single-task (text-only) denoising training: same shared model, but only the text modality is
    trained (the audio task is dropped). Serves as a reference to measure how much the multi-task
    text+audio setup (:func:`aed_denoising_discrete_shared.train_step`) hurts text reconstruction
    vs. training text alone.

    Expects a text-only dataset providing the phoneme indices under the ``phon_indices`` key (no
    audio / no alternate batching). Mirrors the text half of
    :func:`aed_denoising_discrete_shared.train_step`.

    NOTE: the function MUST be named ``train_step`` -- the config serializer imports
    ``<__train_step_module>`` under its own name and RETURNN looks for a global ``train_step``.
    """
    assert set(extern_data.data.keys()) == {"phon_indices", "seq_tag"}
    phon_indices_ = extern_data["phon_indices"]

    model.decode_seq = model.decode_text_seq
    model.forward = model.forward_text
    model.mask_idx = model.text_mask_idx
    model.bos_idx = model.text_bos_idx
    model.eos_idx = model.text_eos_idx
    model.embedding = model.text_embedding
    model.decoder = model.text_decoder
    model.blank_idx = model.text_blank_idx
    aed_denoising_discrete.train_step(
        model=model,
        extern_data=TensorDict({"data": phon_indices_, "seq_tag": extern_data["seq_tag"]}),
        ce_loss_scale=text_ce_loss_scale,
        masked_ce_loss_scale=text_masked_ce_loss_scale,
        label_smoothing=label_smoothing,
        label_smoothing_start_epoch=label_smoothing_start_epoch,
        masking_opts=text_masking_opts,
        aux_loss_scales=aux_loss_scales,
        codebook_diversity_loss_scale=codebook_diversity_loss_scale,
        loss_name="text",
    )
